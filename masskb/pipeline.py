"""检索编排：向量与稀疏写入、文档骨架，以及可选的查询改写、混合检索、图扩展、重排、引用回答和有界工具循环。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client.http.exceptions import UnexpectedResponse

from masskb.agent import (
    MAX_AGENT_STEPS,
    build_agent_messages,
    format_health_observation,
    format_invalid_observation,
    format_search_observation,
    parse_agent_action,
)
from masskb.answer import (
    AnswerStatus,
    ChatClient,
    build_answer_messages,
    no_hits_answer,
    parse_answer,
    skipped_answer,
)
from masskb.config import Settings, get_settings
from masskb.embeddings import EmbeddingClient
from masskb.entities import (
    EntityMention,
    EntityParseError,
    build_entity_messages,
    parse_entity_response,
)
from masskb.fusion import RRF_K, FusedHit, candidate_limit, reciprocal_rank_fusion
from masskb.graph_store import GraphStore, RelatedChunk
from masskb.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    Document,
    IngestReport,
    ingest_path,
)
from masskb.rerank import build_rerank_messages, parse_rerank_order
from masskb.rewrite import build_rewrite_messages, parse_rewrite
from masskb.sparse import encode_text
from masskb.vector_store import (
    DEFAULT_COLLECTION,
    VectorStore,
    point_id_for_chunk,
)

EMBED_BATCH_SIZE = 64
_PROBE_TEXT = "masskb"


class DimensionMismatchError(RuntimeError):
    """探测、分块或查询向量的维度与配置的维度不一致。默认配置是 1536。"""


class GraphSyncError(RuntimeError):
    """向量已按文档提交，图谱事务未完成。整次 ingest 可重试。"""


class EntitySyncError(RuntimeError):
    """向量和文档骨架已提交，实体事务未完成。整次 ingest 可重试。"""


@dataclass(frozen=True)
class SearchHit:
    """一条近邻命中。默认的 `score` 来自 Qdrant，正文来自 payload，前后 id 来自图。

    重排成功后 `score` 改为 `1/名次`。`dense_score` 与 `sparse_score` 仍是重排前的值。
    """

    score: float
    text: str | None
    source: str | None
    chunk_id: str | None
    doc_id: str | None
    index: int | None
    start: int | None
    end: int | None
    prev_chunk_id: str | None
    next_chunk_id: str | None
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    graph_via: str | None = None


@dataclass(frozen=True)
class GraphStatus:
    """这次检索有没有补上前后邻。`skipped` 表示没有去查 Neo4j。"""

    ok: bool
    skipped: bool
    error: str | None = None


@dataclass(frozen=True)
class HybridStatus:
    """这次有没有做稀疏腿。`skipped` 表示没有去查 bm25。"""

    ok: bool
    skipped: bool
    error: str | None = None
    rrf_k: int | None = None
    candidate_k: int | None = None


@dataclass(frozen=True)
class RerankStatus:
    """这次有没有做 listwise 重排。`skipped` 表示没有请求模型。"""

    ok: bool
    skipped: bool
    error: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class GraphHopStatus:
    """这次有没有沿已有边扩展命中。`skipped` 表示没有去做。

    `found` 是排序后的候选数，最多 8。`expanded` 是进入命中列表的条数，为 0 或 1。
    跳过、没有种子或失败时，这两个计数是 None。
    """

    ok: bool
    skipped: bool
    error: str | None = None
    found: int | None = None
    expanded: int | None = None


@dataclass(frozen=True)
class RewriteStatus:
    """这次有没有改写查询。`skipped` 表示没有请求模型。

    成功时 `text` 是拿去 embedding 和稀疏编码的检索文本。
    `SearchResult.query` 仍是用户原文。
    """

    ok: bool
    skipped: bool
    text: str | None
    model: str | None
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """一次向量检索。`doc_id` 是过滤条件，未过滤时为 None。"""

    query: str
    top_k: int
    collection: str
    embedding_model: str
    doc_id: str | None
    hits: tuple[SearchHit, ...]
    graph: GraphStatus
    mode: str = "dense"
    score_kind: str = "cosine"
    hybrid: HybridStatus | None = None
    answer: AnswerStatus | None = None
    rerank: RerankStatus | None = None
    rewrite: RewriteStatus | None = None
    graph_hops: GraphHopStatus | None = None

    def __post_init__(self) -> None:
        if self.hybrid is None:
            object.__setattr__(self, "hybrid", _skipped_hybrid())
        if self.answer is None:
            object.__setattr__(self, "answer", skipped_answer())
        if self.rerank is None:
            object.__setattr__(self, "rerank", _skipped_rerank())
        if self.rewrite is None:
            object.__setattr__(self, "rewrite", _skipped_rewrite())
        if self.graph_hops is None:
            object.__setattr__(self, "graph_hops", _skipped_graph_hops())


@dataclass(frozen=True)
class AgentStep:
    """工具循环里的一步。`search` 里嵌的是一次完整检索结果。"""

    index: int
    action: str
    ok: bool
    error: str | None = None
    query: str | None = None
    search: SearchResult | None = None
    health: dict[str, Any] | None = None


@dataclass(frozen=True)
class AskResult:
    """一次工具循环。`query` 是用户原文。`answer` 是收束后的回答。"""

    query: str
    max_steps: int
    model: str | None
    stopped: str
    steps: tuple[AgentStep, ...]
    answer: AnswerStatus
    top_k: int
    doc_id: str | None
    hybrid: bool
    rewrite: bool
    rerank: bool
    graph_hops: bool = False


@dataclass(frozen=True)
class IngestWriteResult:
    """一次写入的结果。`points_upserted` 是本次 upsert 的点数。"""

    report: IngestReport
    collection: str
    vector_size: int
    points_upserted: int
    graph_documents: int
    graph_chunks: int
    graph_skipped: bool
    entities_skipped: bool = True
    graph_persons: int = 0
    graph_concepts: int = 0
    graph_mentions: int = 0
    entities_dropped: int = 0


def _validate_search(query: str, top_k: int, doc_id: str | None) -> None:
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("查询不能为空")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，收到 {top_k!r}")
    if doc_id is not None and (not isinstance(doc_id, str) or doc_id.strip() == ""):
        raise ValueError("doc_id 不能为空")


def _validate_max_steps(max_steps: int) -> None:
    allowed = range(1, MAX_AGENT_STEPS + 1)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps not in allowed:
        raise ValueError(f"max_steps 必须是 1 到 {MAX_AGENT_STEPS} 的整数，收到 {max_steps!r}")


def _ask_result(
    query: str,
    max_steps: int,
    model: str | None,
    stopped: str,
    steps: tuple[AgentStep, ...],
    answer: AnswerStatus,
    *,
    top_k: int,
    doc_id: str | None,
    hybrid: bool,
    rewrite: bool,
    rerank: bool,
    graph_hops: bool,
) -> AskResult:
    return AskResult(
        query=query,
        max_steps=max_steps,
        model=model,
        stopped=stopped,
        steps=steps,
        answer=answer,
        top_k=top_k,
        doc_id=doc_id,
        hybrid=hybrid,
        rewrite=rewrite,
        rerank=rerank,
        graph_hops=graph_hops,
    )


def _search_result(
    embedding_model: str,
    query: str,
    top_k: int,
    doc_id: str | None,
    hits: tuple[SearchHit, ...],
    graph: GraphStatus,
    *,
    mode: str = "dense",
    score_kind: str = "cosine",
    hybrid: HybridStatus | None = None,
    answer: AnswerStatus | None = None,
    rerank: RerankStatus | None = None,
    rewrite: RewriteStatus | None = None,
) -> SearchResult:
    return SearchResult(
        query=query,
        top_k=top_k,
        collection=DEFAULT_COLLECTION,
        embedding_model=embedding_model,
        doc_id=doc_id,
        hits=hits,
        graph=graph,
        mode=mode,
        score_kind=score_kind,
        hybrid=hybrid,
        answer=answer,
        rerank=rerank,
        rewrite=rewrite,
    )


def _skipped_graph() -> GraphStatus:
    return GraphStatus(ok=True, skipped=True, error=None)


def _skipped_hybrid() -> HybridStatus:
    return HybridStatus(ok=True, skipped=True, error=None, rrf_k=None, candidate_k=None)


def _skipped_rerank() -> RerankStatus:
    return RerankStatus(ok=True, skipped=True, error=None, model=None)


def _skipped_rewrite() -> RewriteStatus:
    return RewriteStatus(ok=True, skipped=True, text=None, model=None, error=None)


def _skipped_graph_hops() -> GraphHopStatus:
    return GraphHopStatus(ok=True, skipped=True, error=None)


def _graph_hop_error(exc: Exception) -> GraphHopStatus:
    return GraphHopStatus(
        ok=False,
        skipped=False,
        error=f"{type(exc).__name__}: {exc}",
    )


def _graph_via(item: RelatedChunk) -> str:
    if item.mention_count > 0 and item.via_next:
        return "mention+next_chunk"
    if item.mention_count > 0:
        return "mention"
    return "next_chunk"


def _first_graph_payload(
    related: Sequence[RelatedChunk],
    payloads: dict[str, dict[str, Any]],
    doc_id: str | None,
) -> tuple[RelatedChunk, dict[str, Any]] | None:
    for item in related:
        payload = payloads.get(item.chunk_id)
        if payload is None:
            continue
        text = payload.get("text")
        if not isinstance(text, str) or text == "":
            continue
        if doc_id is not None and payload.get("doc_id") != doc_id:
            continue
        return item, payload
    return None


def _hit_from_payload(payload: dict[str, Any], item: RelatedChunk) -> SearchHit:
    chunk_id = payload.get("chunk_id")
    return SearchHit(
        score=0.0,
        text=payload.get("text"),
        source=payload.get("source"),
        chunk_id=chunk_id if isinstance(chunk_id, str) and chunk_id else item.chunk_id,
        doc_id=payload.get("doc_id"),
        index=payload.get("index"),
        start=payload.get("start"),
        end=payload.get("end"),
        prev_chunk_id=None,
        next_chunk_id=None,
        graph_via=_graph_via(item),
    )


def _no_hits_rerank() -> RerankStatus:
    return RerankStatus(ok=True, skipped=False, error="no_hits", model=None)


def _hybrid_error(exc: Exception) -> HybridStatus:
    return HybridStatus(
        ok=False,
        skipped=False,
        error=f"{type(exc).__name__}: {exc}",
        rrf_k=None,
        candidate_k=None,
    )


def _copy_result(
    result: SearchResult,
    *,
    hits: tuple[SearchHit, ...] | None = None,
    graph: GraphStatus | None = None,
    answer: AnswerStatus | None = None,
    score_kind: str | None = None,
    rerank: RerankStatus | None = None,
    rewrite: RewriteStatus | None = None,
    graph_hops: GraphHopStatus | None = None,
) -> SearchResult:
    return SearchResult(
        query=result.query,
        top_k=result.top_k,
        collection=result.collection,
        embedding_model=result.embedding_model,
        doc_id=result.doc_id,
        hits=result.hits if hits is None else hits,
        graph=result.graph if graph is None else graph,
        mode=result.mode,
        score_kind=result.score_kind if score_kind is None else score_kind,
        hybrid=result.hybrid,
        answer=result.answer if answer is None else answer,
        rerank=result.rerank if rerank is None else rerank,
        rewrite=result.rewrite if rewrite is None else rewrite,
        graph_hops=result.graph_hops if graph_hops is None else graph_hops,
    )


def _ranked_pairs(hits: tuple[SearchHit, ...]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id and hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            pairs.append((hit.chunk_id, hit.score))
    return pairs


def _hits_by_chunk_id(*groups: tuple[SearchHit, ...]) -> dict[str, SearchHit]:
    found: dict[str, SearchHit] = {}
    for group in groups:
        for hit in group:
            if hit.chunk_id and hit.chunk_id not in found:
                found[hit.chunk_id] = hit
    return found


def _fused_hit(hit: SearchHit, fused: FusedHit) -> SearchHit:
    return SearchHit(
        score=fused.score,
        text=hit.text,
        source=hit.source,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        index=hit.index,
        start=hit.start,
        end=hit.end,
        prev_chunk_id=None,
        next_chunk_id=None,
        dense_score=fused.dense_score,
        sparse_score=fused.sparse_score,
        dense_rank=fused.dense_rank,
        sparse_rank=fused.sparse_rank,
        graph_via=hit.graph_via,
    )


def _with_model(status: AnswerStatus, model: str) -> AnswerStatus:
    return AnswerStatus(
        ok=status.ok,
        skipped=status.skipped,
        text=status.text,
        citations=status.citations,
        model=model,
        error=status.error,
    )


def _search_hit(raw: dict[str, Any]) -> SearchHit:
    payload = raw.get("payload") or {}
    return SearchHit(
        score=float(raw["score"]),
        text=payload.get("text"),
        source=payload.get("source"),
        chunk_id=payload.get("chunk_id"),
        doc_id=payload.get("doc_id"),
        index=payload.get("index"),
        start=payload.get("start"),
        end=payload.get("end"),
        prev_chunk_id=None,
        next_chunk_id=None,
        dense_score=None,
        sparse_score=None,
        dense_rank=None,
        sparse_rank=None,
    )


def _hit_with_score(hit: SearchHit, score: float) -> SearchHit:
    return SearchHit(
        score=score,
        text=hit.text,
        source=hit.source,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        index=hit.index,
        start=hit.start,
        end=hit.end,
        prev_chunk_id=hit.prev_chunk_id,
        next_chunk_id=hit.next_chunk_id,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        graph_via=hit.graph_via,
    )


def _hit_with_neighbors(
    hit: SearchHit,
    prev_chunk_id: str | None,
    next_chunk_id: str | None,
) -> SearchHit:
    return SearchHit(
        score=hit.score,
        text=hit.text,
        source=hit.source,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        index=hit.index,
        start=hit.start,
        end=hit.end,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        graph_via=hit.graph_via,
    )


def _graph_failure(doc_id: str, exc: Exception) -> GraphSyncError:
    return GraphSyncError(
        f"文档 {doc_id} 的 Qdrant 已写入，Neo4j 骨架未完成，可重试。"
        f"{type(exc).__name__}: {exc}"
    )


def _entity_failure(doc_id: str, exc: Exception, *, index: int | None = None) -> EntitySyncError:
    where = f"文档 {doc_id}"
    if index is not None:
        where += f" 分块 {index}"
    return EntitySyncError(
        f"{where} 的 Qdrant 与文档骨架已保留，实体未写入，可重试。"
        f"{type(exc).__name__}: {exc}"
    )


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
    }


class RetrievalPipeline:
    """组装向量、图谱与 Embedding。`ingest` 先写 Qdrant 再同步骨架。

    `ask` 在有界步数内调用现有的 `search` 与 `healthcheck`。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = VectorStore(self.settings)
        self.graph_store = GraphStore(self.settings)
        self.embeddings = EmbeddingClient(self.settings)

    def healthcheck(self) -> dict[str, Any]:
        """检查向量库与图库连通性（不强制测 embedding）。"""
        qdrant = self.vector_store.healthcheck()
        neo4j = self.graph_store.healthcheck()
        return {
            "ok": bool(qdrant.get("ok") and neo4j.get("ok")),
            "qdrant": qdrant,
            "neo4j": neo4j,
            "openai_key_configured": self.settings.has_real_openai_key,
        }

    def ingest(
        self,
        path: Path,
        *,
        root: Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        entities: bool = False,
    ) -> IngestWriteResult:
        """切分并写入 Qdrant，再同步这篇文档的 Neo4j 骨架。不做检索。

        有分块时先探测一条向量的维度，再确保集合存在。维度或集合配置不符时中止，
        不 upsert，也不删除，也不连接 Neo4j。探测用的那条向量不会写入。
        每个文档的向量 upsert 和过期点删除都成功后，才写该文档的图。
        `entities` 为真时，骨架成功后才按分块抽取人物和概念，并替换这些分块上的提及。
        抽取或提及写入失败时不回滚已经写入的向量和骨架。
        """
        report = ingest_path(path, root=root, chunk_size=chunk_size, overlap=overlap)
        if not any(doc.chunks for doc in report.files):
            return self._clear_docs_if_collection_exists(report, entities=entities)

        probe = self.embeddings.embed_one(_PROBE_TEXT)
        if len(probe) != self.settings.embedding_dimensions:
            raise self._dimension_error(len(probe), before_collection=True)

        self.vector_store.ensure_collection(vector_size=self.settings.embedding_dimensions)
        upserted = 0
        graph_documents = 0
        graph_chunks = 0
        person_ids: set[str] = set()
        concept_ids: set[str] = set()
        graph_mentions = 0
        entities_dropped = 0
        for doc in report.files:
            if not doc.chunks:
                self.vector_store.delete_by_doc_id(doc.doc_id)
                self._delete_graph_document(doc.doc_id)
                graph_documents += 1
                continue
            vectors = self._embed_document(doc.chunks)
            ids = [point_id_for_chunk(chunk.chunk_id) for chunk in doc.chunks]
            payloads = [_payload(chunk) for chunk in doc.chunks]
            sparse_vectors = [encode_text(chunk.text) for chunk in doc.chunks]
            self.vector_store.upsert_points(ids, vectors, payloads, sparse_vectors)
            self.vector_store.delete_by_doc_id(doc.doc_id, keep_ids=ids)
            graph_chunks += self._sync_graph_document(doc)
            graph_documents += 1
            upserted += len(ids)
            if entities:
                found, dropped = self._extract_mentions(doc)
                entities_dropped += dropped
                self._replace_mentions(doc, found)
                person_ids.update(item.entity_id for item in found if item.kind == "person")
                concept_ids.update(item.entity_id for item in found if item.kind == "concept")
                graph_mentions += len(found)
        return self._result(
            report,
            upserted,
            graph_documents=graph_documents,
            graph_chunks=graph_chunks,
            graph_skipped=False,
            entities_skipped=not entities,
            graph_persons=len(person_ids),
            graph_concepts=len(concept_ids),
            graph_mentions=graph_mentions,
            entities_dropped=entities_dropped,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        hybrid: bool = False,
        answer: bool = False,
        rerank: bool = False,
        rewrite: bool = False,
        graph_hops: bool = False,
    ) -> SearchResult:
        """用与入库相同的模型做近邻检索，再按 chunk_id 补前后邻。

        先可选地把查询改写成一条检索文本，再生成查询向量，然后看集合是否存在。
        集合不存在、没有点或过滤后没有命中时，返回空的 `hits`，并且不连接 Neo4j。
        缺 Key、维度不符、集合配置不符或 Qdrant 连不上时抛错。
        Neo4j 补邻接失败时仍返回向量命中。
        `rewrite` 为真时，改写发生在 embedding 和稀疏编码之前。失败时仍用原文检索。
        `SearchResult.query` 始终是用户原文。
        `hybrid` 为真时，在补邻之前用进程内 RRF 融合 bm25。
        `graph_hops` 为真时，在截成 `top_k` 之后、重排之前沿已有边最多替换 1 条命中。
        图扩展失败时仍返回向量命中。
        `rerank` 为真时，在图扩展之后、补邻之前做 listwise 重排。
        `answer` 为真时，补邻之后才请求聊天，且只引用本次命中的 chunk_id。
        重排和回答的提示词使用用户原文。这些开关默认关闭。
        """
        _validate_search(query, top_k, doc_id)
        retrieval_text, rewrite_status = self._maybe_rewrite(query, enabled=rewrite)
        vector = self.embeddings.embed_one(retrieval_text)
        expected = self.settings.embedding_dimensions
        if len(vector) != expected:
            raise DimensionMismatchError(
                f"向量维度不符：模型 {self.settings.embedding_model} 返回 {len(vector)} 维，"
                f"期望 {expected}。未执行检索。"
            )
        mode = "hybrid" if hybrid else "dense"
        if not self.vector_store.has_collection():
            result = _search_result(
                self.settings.embedding_model,
                query,
                top_k,
                doc_id,
                (),
                _skipped_graph(),
                mode=mode,
                hybrid=_skipped_hybrid(),
                rewrite=rewrite_status,
            )
            return self._maybe_answer(
                self._maybe_rerank(
                    self._maybe_graph_hops(result, enabled=graph_hops),
                    enabled=rerank,
                ),
                enabled=answer,
            )
        self.vector_store.require_compatible_collection(
            vector_size=self.settings.embedding_dimensions
        )
        limit = candidate_limit(top_k) if hybrid else top_k
        try:
            raw_hits = self.vector_store.search(vector, top_k=limit, doc_id=doc_id)
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            raw_hits = []
        hits = tuple(_search_hit(item) for item in raw_hits)
        if hybrid:
            result = self._hybrid_hits(
                query,
                retrieval_text,
                top_k,
                doc_id,
                hits,
                limit,
                rewrite_status,
            )
        else:
            result = _search_result(
                self.settings.embedding_model,
                query,
                top_k,
                doc_id,
                hits,
                _skipped_graph(),
                mode="dense",
                hybrid=_skipped_hybrid(),
                rewrite=rewrite_status,
            )
        return self._maybe_answer(
            self._attach_neighbors(
                self._maybe_rerank(
                    self._maybe_graph_hops(result, enabled=graph_hops),
                    enabled=rerank,
                )
            ),
            enabled=answer,
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        hybrid: bool = False,
        rewrite: bool = False,
        rerank: bool = False,
        graph_hops: bool = False,
        max_steps: int = MAX_AGENT_STEPS,
    ) -> AskResult:
        """用有界工具循环回答问题。默认不由 `search` 进入。

        每一步只接受一个 JSON 动作：`search`、`healthcheck` 或 `finish`。
        `search` 转发调用方的 `top_k`、`doc_id`、`hybrid`、`rewrite`、`rerank`、
        `graph_hops`，并固定 `answer=False`。引用只能来自本轮带正文的命中。
        空白问题、非法 `top_k`、`doc_id` 或 `max_steps` 在第一次聊天前抛出 `ValueError`。
        缺 Key、聊天失败，以及 `search` 的硬错误会继续抛出。
        走满步数仍未 `finish` 时返回 `stopped="max_steps"`，不抛错。
        """
        _validate_search(query, top_k, doc_id)
        _validate_max_steps(max_steps)
        observations: list[str] = []
        steps: list[AgentStep] = []
        seen_ids: set[str] = set()
        model = self.settings.llm_model
        for index in range(1, max_steps + 1):
            system, user = build_agent_messages(
                query,
                observations,
                top_k=top_k,
                doc_id=doc_id,
                hybrid=hybrid,
                rewrite=rewrite,
                rerank=rerank,
                graph_hops=graph_hops,
            )
            raw = self._complete_agent(system, user)
            action = parse_agent_action(raw)
            if not action.ok or action.kind is None:
                error = action.error or "invalid_shape"
                steps.append(AgentStep(index=index, action="invalid", ok=False, error=error))
                observations.append(format_invalid_observation(error))
                continue
            if action.kind == "finish":
                parsed = _with_model(parse_answer(action.raw or "", seen_ids), model)
                steps.append(
                    AgentStep(
                        index=index,
                        action="finish",
                        ok=parsed.ok,
                        error=parsed.error,
                    )
                )
                return _ask_result(
                    query,
                    max_steps,
                    model,
                    "finish",
                    tuple(steps),
                    parsed,
                    top_k=top_k,
                    doc_id=doc_id,
                    hybrid=hybrid,
                    rewrite=rewrite,
                    rerank=rerank,
                    graph_hops=graph_hops,
                )
            if action.kind == "healthcheck":
                payload = self.healthcheck()
                steps.append(AgentStep(index=index, action="healthcheck", ok=True, health=payload))
                observations.append(format_health_observation(payload))
                continue
            tool_query = action.query
            if tool_query is None:
                steps.append(
                    AgentStep(index=index, action="invalid", ok=False, error="invalid_shape")
                )
                observations.append(format_invalid_observation("invalid_shape"))
                continue
            result = self.search(
                tool_query,
                top_k=top_k,
                doc_id=doc_id,
                hybrid=hybrid,
                answer=False,
                rerank=rerank,
                rewrite=rewrite,
                graph_hops=graph_hops,
            )
            for hit in result.hits:
                if hit.chunk_id and hit.text:
                    seen_ids.add(hit.chunk_id)
            steps.append(
                AgentStep(
                    index=index,
                    action="search",
                    ok=True,
                    query=tool_query,
                    search=result,
                )
            )
            observations.append(format_search_observation(tool_query, result))
        return _ask_result(
            query,
            max_steps,
            model,
            "max_steps",
            tuple(steps),
            AnswerStatus(
                ok=False,
                skipped=False,
                text=None,
                citations=(),
                model=model,
                error="max_steps",
            ),
            top_k=top_k,
            doc_id=doc_id,
            hybrid=hybrid,
            rewrite=rewrite,
            rerank=rerank,
            graph_hops=graph_hops,
        )

    def _complete_agent(self, system: str, user: str) -> str:
        return ChatClient(self.settings).complete(system, user, max_tokens=512)

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()

    def _clear_docs_if_collection_exists(
        self,
        report: IngestReport,
        *,
        entities: bool = False,
    ) -> IngestWriteResult:
        """没有可写入的分块时不探测、不建集合。集合已在则先删点，再删骨架。"""
        if not self.vector_store.has_collection() or not report.files:
            return self._result(
                report,
                0,
                graph_documents=0,
                graph_chunks=0,
                graph_skipped=True,
                entities_skipped=not entities,
            )
        for doc in report.files:
            self.vector_store.delete_by_doc_id(doc.doc_id)
            self._delete_graph_document(doc.doc_id)
        return self._result(
            report,
            0,
            graph_documents=len(report.files),
            graph_chunks=0,
            graph_skipped=False,
            entities_skipped=not entities,
        )

    def _sync_graph_document(self, doc: Document) -> int:
        try:
            self.graph_store.ensure_schema()
            return self.graph_store.upsert_document(doc)
        except GraphSyncError:
            raise
        except Exception as exc:  # 图失败不回滚已写入的向量
            raise _graph_failure(doc.doc_id, exc) from exc

    def _extract_mentions(self, doc: Document) -> tuple[tuple[EntityMention, ...], int]:
        """按分块请求聊天。任一块失败则整篇不写提及。"""
        found: list[EntityMention] = []
        dropped = 0
        for chunk in doc.chunks:
            try:
                system, user = build_entity_messages(chunk.chunk_id, chunk.text)
                raw = self._complete_entities(system, user)
                mentions, dropped_here = parse_entity_response(
                    raw,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                )
            except EntitySyncError:
                raise
            except EntityParseError as exc:
                raise _entity_failure(doc.doc_id, exc, index=chunk.index) from exc
            except Exception as exc:  # 抽取失败不改已有提及
                raise _entity_failure(doc.doc_id, exc, index=chunk.index) from exc
            found.extend(mentions)
            dropped += dropped_here
        return tuple(found), dropped

    def _replace_mentions(self, doc: Document, mentions: Sequence[EntityMention]) -> None:
        """骨架已经写入。失败时不回滚向量和骨架，也不改已有提及（事务未提交）。"""
        try:
            self.graph_store.ensure_entity_schema()
            self.graph_store.replace_mentions(
                doc.doc_id,
                [chunk.chunk_id for chunk in doc.chunks],
                mentions,
            )
        except EntitySyncError:
            raise
        except Exception as exc:  # 提及写入失败不回滚已提交的骨架
            raise _entity_failure(doc.doc_id, exc) from exc

    def _complete_entities(self, system: str, user: str) -> str:
        return ChatClient(self.settings).complete(system, user, max_tokens=256)

    def _delete_graph_document(self, doc_id: str) -> None:
        try:
            self.graph_store.delete_document(doc_id)
        except GraphSyncError:
            raise
        except Exception as exc:  # 点已删，骨架失败时可重试
            raise _graph_failure(doc_id, exc) from exc

    def _maybe_graph_hops(self, result: SearchResult, *, enabled: bool) -> SearchResult:
        """在重排之前沿已有边最多换入 1 条分块。关闭或失败时不改向量命中。"""
        if not enabled:
            return _copy_result(result, graph_hops=_skipped_graph_hops())
        seeds: list[str] = []
        seen: set[str] = set()
        for hit in result.hits:
            if hit.chunk_id and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                seeds.append(hit.chunk_id)
        if not seeds:
            return _copy_result(
                result,
                graph_hops=GraphHopStatus(ok=True, skipped=False, error="no_hits"),
            )
        try:
            related = self.graph_store.related_chunks(seeds, doc_id=result.doc_id)
        except Exception as exc:  # noqa: BLE001 — 图扩展失败仍返回向量命中
            return _copy_result(result, graph_hops=_graph_hop_error(exc))
        found = len(related)
        if found == 0 or result.top_k < 2:
            return _copy_result(
                result,
                graph_hops=GraphHopStatus(
                    ok=True,
                    skipped=False,
                    error=None,
                    found=found,
                    expanded=0,
                ),
            )
        try:
            payloads = self.vector_store.get_by_chunk_ids(tuple(item.chunk_id for item in related))
        except Exception as exc:  # noqa: BLE001 — 取正文失败仍返回向量命中
            return _copy_result(result, graph_hops=_graph_hop_error(exc))
        chosen = _first_graph_payload(related, payloads, result.doc_id)
        if chosen is None:
            return _copy_result(
                result,
                graph_hops=GraphHopStatus(
                    ok=True,
                    skipped=False,
                    error=None,
                    found=found,
                    expanded=0,
                ),
            )
        item, payload = chosen
        expansion = _hit_from_payload(payload, item)
        kept = result.hits[: result.top_k - 1]
        hits = (*kept, expansion)[: result.top_k]
        expanded = 1 if any(hit.graph_via for hit in hits) else 0
        return _copy_result(
            result,
            hits=hits,
            graph_hops=GraphHopStatus(
                ok=True,
                skipped=False,
                error=None,
                found=found,
                expanded=expanded,
            ),
        )

    def _attach_neighbors(self, result: SearchResult) -> SearchResult:
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for hit in result.hits:
            if hit.chunk_id and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                chunk_ids.append(hit.chunk_id)
        if not chunk_ids:
            return result
        try:
            neighbors = self.graph_store.neighbor_chunk_ids(chunk_ids)
        except Exception as exc:  # noqa: BLE001 — 邻接失败仍返回向量命中
            cleared = tuple(_hit_with_neighbors(hit, None, None) for hit in result.hits)
            status = GraphStatus(
                ok=False,
                skipped=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _copy_result(result, hits=cleared, graph=status)
        enriched = []
        for hit in result.hits:
            pair = neighbors.get(hit.chunk_id) if hit.chunk_id else None
            if pair is None:
                enriched.append(hit)
            else:
                enriched.append(_hit_with_neighbors(hit, pair[0], pair[1]))
        return _copy_result(
            result,
            hits=tuple(enriched),
            graph=GraphStatus(ok=True, skipped=False, error=None),
        )

    def _hybrid_hits(
        self,
        query: str,
        retrieval_text: str,
        top_k: int,
        doc_id: str | None,
        hits: tuple[SearchHit, ...],
        limit: int,
        rewrite: RewriteStatus,
    ) -> SearchResult:
        """有稠密命中时才查稀疏。失败或分词为空时保留余弦命中。"""
        if not hits:
            return _search_result(
                self.settings.embedding_model,
                query,
                top_k,
                doc_id,
                (),
                _skipped_graph(),
                mode="hybrid",
                hybrid=_skipped_hybrid(),
                rewrite=rewrite,
            )
        indices, values = encode_text(retrieval_text)
        if not indices:
            return self._dense_hybrid_result(
                query,
                top_k,
                doc_id,
                hits,
                HybridStatus(ok=True, skipped=False, error=None),
                rewrite,
            )
        try:
            ready = self.vector_store.sparse_ready()
        except Exception as exc:  # noqa: BLE001 — 稀疏槽读失败仍返回稠密命中
            return self._dense_hybrid_result(
                query, top_k, doc_id, hits, _hybrid_error(exc), rewrite
            )
        if not ready:
            return self._dense_hybrid_result(
                query,
                top_k,
                doc_id,
                hits,
                HybridStatus(
                    ok=False,
                    skipped=False,
                    error=f"集合 {DEFAULT_COLLECTION} 没有稀疏向量 bm25",
                ),
                rewrite,
            )
        try:
            raw_sparse = self.vector_store.search_sparse(
                indices,
                values,
                top_k=limit,
                doc_id=doc_id,
            )
        except Exception as exc:  # noqa: BLE001 — 稀疏查询失败仍返回稠密命中
            return self._dense_hybrid_result(
                query, top_k, doc_id, hits, _hybrid_error(exc), rewrite
            )
        sparse_hits = tuple(_search_hit(item) for item in raw_sparse)
        fused = reciprocal_rank_fusion(
            _ranked_pairs(hits),
            _ranked_pairs(sparse_hits),
            top_k=top_k,
        )
        by_id = _hits_by_chunk_id(hits, sparse_hits)
        fused_hits = tuple(_fused_hit(by_id[item.chunk_id], item) for item in fused)
        return _search_result(
            self.settings.embedding_model,
            query,
            top_k,
            doc_id,
            fused_hits,
            _skipped_graph(),
            mode="hybrid",
            score_kind="rrf",
            hybrid=HybridStatus(
                ok=True,
                skipped=False,
                error=None,
                rrf_k=RRF_K,
                candidate_k=limit,
            ),
            rewrite=rewrite,
        )

    def _dense_hybrid_result(
        self,
        query: str,
        top_k: int,
        doc_id: str | None,
        hits: tuple[SearchHit, ...],
        hybrid: HybridStatus,
        rewrite: RewriteStatus,
    ) -> SearchResult:
        return _search_result(
            self.settings.embedding_model,
            query,
            top_k,
            doc_id,
            hits[:top_k],
            _skipped_graph(),
            mode="hybrid",
            score_kind="cosine",
            hybrid=hybrid,
            rewrite=rewrite,
        )

    def _maybe_rewrite(self, query: str, *, enabled: bool) -> tuple[str, RewriteStatus]:
        """关闭时不创建聊天客户端。失败时退回原文。"""
        if not enabled:
            return query, _skipped_rewrite()
        try:
            system, user = build_rewrite_messages(query)
            raw = self._complete_rewrite(system, user)
            parsed = parse_rewrite(raw)
        except Exception as exc:  # noqa: BLE001 — 改写失败仍用原文检索
            status = RewriteStatus(
                ok=False,
                skipped=False,
                text=None,
                model=self.settings.llm_model,
                error=f"{type(exc).__name__}: {exc}",
            )
            return query, status
        if not parsed.ok or parsed.text is None:
            status = RewriteStatus(
                ok=False,
                skipped=False,
                text=None,
                model=self.settings.llm_model,
                error=parsed.error,
            )
            return query, status
        status = RewriteStatus(
            ok=True,
            skipped=False,
            text=parsed.text,
            model=self.settings.llm_model,
            error=None,
        )
        return parsed.text, status

    def _complete_rewrite(self, system: str, user: str) -> str:
        return ChatClient(self.settings).complete(system, user, max_tokens=256)

    def _maybe_rerank(self, result: SearchResult, *, enabled: bool) -> SearchResult:
        """在补邻之前重排已经截断的命中。关闭或没有可排序命中时不创建聊天客户端。"""
        if not enabled:
            return _copy_result(result, rerank=_skipped_rerank())
        eligible: list[int] = []
        allowed: list[str] = []
        seen: set[str] = set()
        for index, hit in enumerate(result.hits):
            if hit.chunk_id and hit.text and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                allowed.append(hit.chunk_id)
                eligible.append(index)
        if not allowed:
            return _copy_result(result, rerank=_no_hits_rerank())
        try:
            system, user = build_rerank_messages(
                result.query,
                tuple(result.hits[index] for index in eligible),
            )
            raw = self._complete_rerank(system, user)
            parsed = parse_rerank_order(raw, allowed)
        except Exception as exc:  # noqa: BLE001 — 重排失败仍返回原顺序
            status = RerankStatus(
                ok=False,
                skipped=False,
                error=f"{type(exc).__name__}: {exc}",
                model=self.settings.llm_model,
            )
            return _copy_result(result, rerank=status)
        if not parsed.ok:
            status = RerankStatus(
                ok=False,
                skipped=False,
                error=parsed.error,
                model=self.settings.llm_model,
            )
            return _copy_result(result, rerank=status)
        by_id = {result.hits[index].chunk_id: result.hits[index] for index in eligible}
        chosen = set(eligible)
        ranked = [by_id[chunk_id] for chunk_id in parsed.ids]
        tail = [hit for index, hit in enumerate(result.hits) if index not in chosen]
        ordered = [*ranked, *tail]
        scored = tuple(
            _hit_with_score(hit, 1.0 / rank) for rank, hit in enumerate(ordered, start=1)
        )
        return _copy_result(
            result,
            hits=scored,
            score_kind="rerank",
            rerank=RerankStatus(
                ok=True,
                skipped=False,
                error=None,
                model=self.settings.llm_model,
            ),
        )

    def _complete_rerank(self, system: str, user: str) -> str:
        return ChatClient(self.settings).complete(system, user)

    def _maybe_answer(self, result: SearchResult, *, enabled: bool) -> SearchResult:
        if not enabled:
            return _copy_result(result, answer=skipped_answer())
        usable = tuple(hit for hit in result.hits if hit.chunk_id and hit.text)
        if not usable:
            return _copy_result(result, answer=no_hits_answer())
        try:
            system, user = build_answer_messages(result.query, usable)
            raw = self._complete_answer(system, user)
            parsed = parse_answer(raw, {hit.chunk_id for hit in usable if hit.chunk_id})
            parsed = _with_model(parsed, self.settings.llm_model)
        except Exception as exc:  # noqa: BLE001 — 回答失败仍返回检索命中
            parsed = AnswerStatus(
                ok=False,
                skipped=False,
                text=None,
                citations=(),
                model=self.settings.llm_model,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _copy_result(result, answer=parsed)

    def _complete_answer(self, system: str, user: str) -> str:
        return ChatClient(self.settings).complete(system, user)

    def _embed_document(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in chunks]
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embedded = self.embeddings.embed(batch)
            if len(embedded) != len(batch):
                raise DimensionMismatchError(
                    f"embedding 返回 {len(embedded)} 条，期望 {len(batch)} 条。未写入该文档。"
                )
            expected = self.settings.embedding_dimensions
            for vector in embedded:
                if len(vector) != expected:
                    raise self._dimension_error(len(vector), before_collection=False)
            vectors.extend(embedded)
        return vectors

    def _dimension_error(self, actual: int, *, before_collection: bool) -> DimensionMismatchError:
        suffix = "未创建集合，也未写入。" if before_collection else "未写入该文档。"
        model = self.settings.embedding_model
        expected = self.settings.embedding_dimensions
        return DimensionMismatchError(
            f"向量维度不符：模型 {model} 返回 {actual} 维，期望 {expected}。{suffix}"
        )

    def _result(
        self,
        report: IngestReport,
        points_upserted: int,
        *,
        graph_documents: int,
        graph_chunks: int,
        graph_skipped: bool,
        entities_skipped: bool = True,
        graph_persons: int = 0,
        graph_concepts: int = 0,
        graph_mentions: int = 0,
        entities_dropped: int = 0,
    ) -> IngestWriteResult:
        return IngestWriteResult(
            report=report,
            collection=DEFAULT_COLLECTION,
            vector_size=self.settings.embedding_dimensions,
            points_upserted=points_upserted,
            graph_documents=graph_documents,
            graph_chunks=graph_chunks,
            graph_skipped=graph_skipped,
            entities_skipped=entities_skipped,
            graph_persons=graph_persons,
            graph_concepts=graph_concepts,
            graph_mentions=graph_mentions,
            entities_dropped=entities_dropped,
        )

"""进程内倒数排名融合。不调用 Qdrant 的 FusionQuery。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

RRF_K = 60
FUSION_MIN_CANDIDATES = 20
_MISSING_RANK = 10**9


@dataclass(frozen=True)
class FusedHit:
    """一条融合结果。某一腿没有这条时，对应的分和名次是 None。"""

    chunk_id: str
    score: float
    dense_score: float | None
    sparse_score: float | None
    dense_rank: int | None
    sparse_rank: int | None


def candidate_limit(top_k: int) -> int:
    """每一腿取回的条数。融合后再截断到调用方的 top_k。"""
    return max(top_k, FUSION_MIN_CANDIDATES)


def reciprocal_rank_fusion(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[FusedHit]:
    """按两列已经排好的名次做 RRF。同一列里重复的 chunk_id 只保留第一次。"""
    dense_ranks = _first_ranks(dense)
    sparse_ranks = _first_ranks(sparse)
    hits: list[FusedHit] = []
    for chunk_id in dict.fromkeys((*dense_ranks, *sparse_ranks)):
        dense_rank, dense_score = dense_ranks.get(chunk_id, (None, None))
        sparse_rank, sparse_score = sparse_ranks.get(chunk_id, (None, None))
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (rrf_k + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (rrf_k + sparse_rank)
        hits.append(
            FusedHit(
                chunk_id=chunk_id,
                score=score,
                dense_score=dense_score,
                sparse_score=sparse_score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
            )
        )
    hits.sort(key=_fusion_sort_key)
    return hits[:top_k]


def _first_ranks(
    pairs: Sequence[tuple[str, float]],
) -> dict[str, tuple[int, float]]:
    found: dict[str, tuple[int, float]] = {}
    for chunk_id, score in pairs:
        if chunk_id in found:
            continue
        found[chunk_id] = (len(found) + 1, float(score))
    return found


def _fusion_sort_key(hit: FusedHit) -> tuple[float, int, str]:
    dense_rank = _MISSING_RANK if hit.dense_rank is None else hit.dense_rank
    return (-hit.score, dense_rank, hit.chunk_id)

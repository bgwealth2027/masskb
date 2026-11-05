# 架构图

本目录是 MassKB 的目标设计图，源文件为 draw.io（`.drawio`）。在 [diagrams.net](https://app.diagrams.net/) 中选择 **Open Existing Diagram** 即可编辑。GitHub 不会内嵌渲染这些文件。

图中的方框是规划中的形态。已经落地的范围以仓库根目录 [README 的「当前状态」](../../README.md#当前状态) 为准：当前代码提供配置、Embedding 客户端、Qdrant / Neo4j 健康检查、本地文本切分、写入 Qdrant 的稠密向量和 `bm25` 稀疏向量、Neo4j 文档骨架（`Document`、`Chunk`、`HAS_CHUNK`、`NEXT_CHUNK`），以及同一模型上的近邻检索（集合 `masskb_chunks`，余弦距离，默认 1536 维，命中可附前后 `chunk_id`）。`search --hybrid` 在进程内融合稀疏名次，`search --answer` 可以生成带 `chunk_id` 引用的回答，两者默认关闭。`ingest --entities` 可以把人物和概念提及挂到 `Chunk` 上，默认关闭。`search --rerank` 用 `LLM_MODEL` 对已经截断的命中做 listwise 重排，默认关闭。`ingest --dry-run` 仍只预览、不连接外部服务。`masskb serve` 提供 `GET /health`、`POST /search`、`POST /ask` 和 `POST /ingest`；宿主机可直接运行，Compose 里的应用容器跑同一条命令。`POST /ingest` 读取服务端可见路径，不挂语料卷。进程内有界工具循环已由 `masskb ask` 落地，只调用现有检索与健康检查，并可把调用方的 `graph_hops` 传进检索。`search --graph-hops` 已能沿文档骨架和人物/概念提及做有界扩展。图中的意图理解分叉、独立图谱工具、书目关系、别名合并、独立 Agent 服务和完整 SDK 仍是目标。应用容器已进入 Compose。可选嵌入容器在 profile `embed` 中，默认不启动，也不拉取模型。图中的 bge-m3 与 GPU 仍是目标。Python 客户端已覆盖 `GET /health`、`POST /search`、`POST /ask` 和 `POST /ingest`。设计图里的独立检索引擎、完整 REST 和多用户鉴权仍是目标。书目关系仍未实现。查询改写已由 `search --rewrite` 在检索前做单次改写，不是图里的意图理解。图中的「融合重排」和「工具调用：检索 / 图谱 / 重排」仍是目标形态，不表示独立重排服务或图谱工具。

| 文件 | 图名 | 阅读时注意 |
|------|------|------------|
| [01-总体架构.drawio](01-总体架构.drawio) | 总体架构 | 图中的完整 SDK 与完整 REST 仍是目标。Python 客户端已覆盖现有四条路由，含 `POST /ingest`。进程内有界工具循环已由 `masskb ask` 落地，只调用现有检索与健康检查；图中的图谱工具和独立 Agent 服务仍是目标。本机 HTTP 由 `masskb serve` 提供。图中的独立 BM25 引擎没有落地；代码使用 Qdrant 稀疏向量，并在进程内融合 |
| [02-数据流.drawio](02-数据流.drawio) | 数据流 | 向量入库、`bm25` 稀疏槽、文档骨架与近邻查询已实现（集合 `masskb_chunks`，余弦，1536）。`--hybrid` 做进程内融合，`--answer` 可生成引用。`--entities` 写入可选的 Person/Concept 提及，不是图中的完整书目关系。查询改写已由 `search --rewrite` 在检索前做单次改写。`search --graph-hops` 已能沿文档骨架和人物/概念提及做有界扩展。图中的意图理解，以及改写后再分叉去图谱查询，仍是目标。重排已由 `search --rerank` 以 listwise 完成，不是图里的独立重排服务。`ingest --dry-run` 仍只做本地预览。图中向量模型写作 bge-m3；代码默认是 OpenAI 兼容的 `text-embedding-3-small`。Compose profile `embed` 的默认模型是 `BAAI/bge-small-zh-v1.5`（512 维），不是 bge-m3 |
| [03-部署架构.drawio](03-部署架构.drawio) | 部署架构 | 当前 [`docker-compose.yml`](../../docker-compose.yml) 含 Qdrant、Neo4j 和 `masskb serve` 应用服务。可选嵌入推理容器在 profile `embed` 中，默认 `docker compose up` 不启动它，也不拉取模型。宿主机仍可直接跑 CLI |
| [04-集成与社区.drawio](04-集成与社区.drawio) | 集成与社区 | Python 客户端已覆盖现有四条路由，含路径入库。图中的完整 SDK、多语言 SDK 和完整 REST 仍是目标。`masskb serve` 覆盖本机健康检查、检索、`POST /ask` 和 `POST /ingest`。Fork → Pull Request → CI → 合并这条路径中，CI 已由 [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) 执行离线 `pytest` |

密钥、`.env` 和本地语料不要放进仓库。`.env.example` 只保留占位值。

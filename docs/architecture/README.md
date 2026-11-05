# 架构图

本目录是 MassKB 的目标设计图，源文件为 draw.io（`.drawio`）。在 [diagrams.net](https://app.diagrams.net/) 中选择 **Open Existing Diagram** 即可编辑。GitHub 不会内嵌渲染这些文件。

图中的方框是规划中的形态。已经落地的范围以仓库根目录 [README 的「当前状态」](../../README.md#当前状态) 为准：当前代码提供配置、Embedding 客户端、Qdrant / Neo4j 健康检查、本地文本切分、写入 Qdrant 的向量入库、Neo4j 文档骨架（`Document`、`Chunk`、`HAS_CHUNK`、`NEXT_CHUNK`），以及同一模型上的近邻检索（集合 `masskb_chunks`，余弦距离，1536 维，命中可附前后 `chunk_id`）。`ingest --dry-run` 仍只预览、不连接外部服务。BM25、混合检索和实体抽取仍未实现。

| 文件 | 图名 | 阅读时注意 |
|------|------|------------|
| [01-总体架构.drawio](01-总体架构.drawio) | 总体架构 | SDK、REST API、BM25、混合检索和 Agent 工具调用尚未实现 |
| [02-数据流.drawio](02-数据流.drawio) | 数据流 | 向量入库、文档骨架与近邻查询已实现（集合 `masskb_chunks`，余弦，1536）。实体抽取、查询改写和融合重排尚未实现。`ingest --dry-run` 仍只做本地预览。图中向量模型写作 bge-m3；代码默认是 OpenAI 兼容的 `text-embedding-3-small` |
| [03-部署架构.drawio](03-部署架构.drawio) | 部署架构 | 当前 [`docker-compose.yml`](../../docker-compose.yml) 只有 Qdrant 和 Neo4j。应用容器与可选嵌入推理容器不在该文件中，Python 包在宿主机运行 |
| [04-集成与社区.drawio](04-集成与社区.drawio) | 集成与社区 | SDK / REST 仍是规划。Fork → Pull Request → CI → 合并这条路径中，CI 已由 [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) 执行离线 `pytest` |

密钥、`.env` 和本地语料不要放进仓库。`.env.example` 只保留占位值。

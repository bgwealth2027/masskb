# MassKB

面向海量知识库的智能检索中间件：向量检索、知识图谱与智能代理。

*Intelligent retrieval middleware for large knowledge bases — vector search, knowledge graph, and an agent layer.*

[![CI](https://github.com/bgwealth2027/masskb/actions/workflows/ci.yml/badge.svg)](https://github.com/bgwealth2027/masskb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)

MassKB 是一层检索中间件，目标是把向量检索、知识图谱和智能代理放在同一条调用链上。当前版本是 **0.1.0**。仓库里已有可安装的 Python 包、Qdrant 与 Neo4j 的 Docker Compose、离线单元测试，以及在 `main` 上运行的 GitHub Actions。语料入库和混合检索仍未完成，调用 `RetrievalPipeline.search` 会抛出 `NotImplementedError`。

## 目录

- [项目简介](#项目简介)
- [架构](#架构)
- [当前状态](#当前状态)
- [安装与运行](#安装与运行)
- [配置](#配置)
- [项目结构](#项目结构)
- [测试与持续集成](#测试与持续集成)
- [许可证](#许可证)

## 项目简介

本仓库交付的是 Python 库、命令行，以及依赖服务的 Compose 文件。三个组成部分的设计职责和 0.1.0 的实际范围如下：

| 组成部分 | 目标职责 | 本仓库里的落点 |
|----------|----------|----------------|
| 向量检索 | 对文本做语义相似度查找 | `VectorStore` 连接 Qdrant，目前只做连通性检查 |
| 知识图谱 | 保存实体与关系，并支持图查询 | `GraphStore` 连接 Neo4j，目前只做 Bolt 连通性检查 |
| 智能代理 | 用大模型改写查询、调度检索并组织回答 | 配置项已预留；没有对话客户端，也没有工具调用循环 |

向量由 OpenAI 兼容的 Embedding API 生成。`pyproject.toml` 的依赖是 `qdrant-client`、`neo4j`、`openai`、`python-dotenv`、`httpx`、`pydantic`，安装清单里没有 `sentence-transformers` 或 `torch`。

## 架构

### 当前运行时

Docker Compose 只启动 Qdrant 和 Neo4j。Python 包跑在宿主机的虚拟环境里，直接访问本机端口。Embedding 请求发往 `LLM_BASE_URL`（默认 `https://api.openai.com/v1`）。健康检查不调用 Embedding API。

```mermaid
flowchart LR
  pkg["masskb 包\n宿主机虚拟环境"]
  qdrant["Qdrant\nlocalhost:6333"]
  neo4j["Neo4j Bolt\nlocalhost:7687"]
  api["OpenAI 兼容 API\nEmbedding"]

  pkg -->|"healthcheck"| qdrant
  pkg -->|"healthcheck"| neo4j
  pkg -->|"EmbeddingClient"| api
```

| 组件 | 实现 | 说明 |
|------|------|------|
| Qdrant | `qdrant/qdrant:latest` | HTTP `6333`，gRPC `6334`；容器名 `masskb-qdrant` |
| Neo4j 5 | `neo4j:5` | Browser `7474`，Bolt `7687`；容器名 `masskb-neo4j`；Compose 会安装 APOC，业务代码尚未调用 |
| Embedding / LLM | OpenAI 兼容 HTTP API | `EmbeddingClient` 会请求 embeddings；`LLM_MODEL` 已读入配置，没有对应的对话调用 |
| 编排 | `RetrievalPipeline` | `healthcheck()` 汇总两个数据库的状态；`search()` 未实现 |
| 部署 | `docker-compose.yml` | 数据卷：`qdrant_data`、`neo4j_data`、`neo4j_logs` |

### 目标架构图

设计图放在 [`docs/architecture/`](docs/architecture/README.md)，源文件是 draw.io，可用 [diagrams.net](https://app.diagrams.net/) 打开。这些图描述的是目标形态。图中的 REST API、BM25、混合检索、语料入库、Agent 工具调用、应用容器和本地嵌入推理容器，以 [当前状态](#当前状态) 为准。

| 文件 | 图名 | 图里有什么 |
|------|------|------------|
| [01-总体架构.drawio](docs/architecture/01-总体架构.drawio) | 总体架构 | 检索层、图谱层、智能层，以及面向集成方的 SDK / REST |
| [02-数据流.drawio](docs/architecture/02-数据流.drawio) | 数据流 | 索引与图谱构建、查询路径。图中向量模型写作 bge-m3；代码默认模型是 `text-embedding-3-small` |
| [03-部署架构.drawio](docs/architecture/03-部署架构.drawio) | 部署架构 | Compose 网络。图中还有应用容器和可选嵌入容器；当前 Compose 文件没有这两项 |
| [04-集成与社区.drawio](docs/architecture/04-集成与社区.drawio) | 集成与社区 | 第三方经 SDK / API 集成，以及 Fork → PR → CI → 合并。SDK / REST 仍是规划；CI 已跑离线测试 |

## 当前状态

`main` 上已经有核心包、离线测试和 CI。入库与检索的 MVP 仍在推进，下表只统计仓库里已经存在的代码。

| 能力 | 状态 | 说明 |
|------|------|------|
| 环境配置 | 已实现 | `masskb.config.Settings` 读取仓库根目录 `.env` 与环境变量 |
| Embedding 客户端 | 已实现 | `EmbeddingClient.embed` / `embed_one`；占位 API Key 会直接报错，测试里客户端被 mock |
| Qdrant 封装 | 部分实现 | `get_collections` 健康检查；没有建集合、写入或相似度查询 |
| Neo4j 封装 | 部分实现 | `verify_connectivity` 与 `RETURN 1`；没有实体模型或业务 Cypher |
| 健康检查 CLI | 已实现 | `masskb healthcheck`、`python -m masskb`、`scripts/healthcheck.py` |
| 离线测试与 CI | 已实现 | `tests/` 不连接真实服务；`.github/workflows/ci.yml` 在 `main` 推送和 Pull Request 上运行 |
| 语料入库 | 未提供 | 仓库中没有 ingest 模块或样例语料流水线 |
| `RetrievalPipeline.search` | 未实现 | 调用即抛出 `NotImplementedError` |
| 混合检索、图谱构建、Agent、REST API | 未实现 | 只出现在 `docs/architecture/` 的设计图中 |

## 安装与运行

需要 Python 3.10 或更高版本（CI 使用 3.12）以及 Docker Compose。Python 包不在容器里运行。

### 1. 克隆

```bash
git clone https://github.com/bgwealth2027/masskb.git
cd masskb
```

### 2. 启动 Qdrant 与 Neo4j

```bash
docker compose up -d
docker compose ps
```

| 服务 | 地址 |
|------|------|
| Qdrant HTTP | http://localhost:6333 |
| Qdrant gRPC | `localhost:6334` |
| Neo4j Browser | http://localhost:7474 |
| Neo4j Bolt | `bolt://localhost:7687` |

Compose 里的开发凭据是 `NEO4J_AUTH=neo4j/masskb-dev-pass`。这是本地示例密码，不要直接用于共享或生产环境。首次启动 Neo4j 时会拉取 APOC，健康检查的 `start_period` 是 40 秒。

### 3. 准备 `.env`

```bash
cp .env.example .env
```

至少把 `NEO4J_PASSWORD` 改成与 Compose 一致的 `masskb-dev-pass`。`.env.example` 里的 `your-neo4j-password-here` 和 `sk-your-api-key-here` 都只是占位符。没有真实 `OPENAI_API_KEY` 时，数据库健康检查仍然可以执行；`EmbeddingClient` 在真正请求前会拒绝占位 Key。`.env` 已被 `.gitignore` 排除。

### 4. 创建虚拟环境并安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

`".[dev]"` 会安装本包，以及 `pytest` 和 `ruff`。只装运行依赖时可以使用 `pip install -e .` 或 `pip install -r requirements.txt`。

### 5. 健康检查

确认容器 `masskb-qdrant` 和 `masskb-neo4j` 已启动后执行：

```bash
python -m masskb healthcheck
```

安装完成后也可以直接运行 `masskb healthcheck`。不带子命令的 `python -m masskb` 同样进入健康检查。未执行可编辑安装时，可以用：

```bash
python scripts/healthcheck.py
```

Qdrant 与 Neo4j 都成功时，输出里各有一行 `[OK]`，进程退出码为 0，并附上 JSON 结果。任一数据库失败则退出码为 1。检查过程不会请求 Embedding API；Key 未配置时会多一行 `[INFO]`，这一行不改变退出码。

### 6. 运行测试

```bash
python -m pytest
```

测试不连接 Qdrant、Neo4j，也不会访问 OpenAI。

## 配置

导入 `masskb.config` 时会加载仓库根目录的 `.env`，并把 `localhost`、`127.0.0.1` 和 `::1` 写入 `NO_PROXY` 与 `no_proxy`，避免本机 Docker 端口被系统 HTTP 代理截走。

| 变量 | 未设置时的代码默认值 | 说明 |
|------|----------------------|------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP 地址 |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt 地址 |
| `NEO4J_USER` | `neo4j` | 与 Compose 中的用户名一致 |
| `NEO4J_PASSWORD` | 空字符串 | 本地 Compose 请填 `masskb-dev-pass` |
| `OPENAI_API_KEY` | 空字符串 | 以 `sk-your-` 开头的占位值视为未配置 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容网关，Embedding 客户端使用 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `.env.example` 中为注释项 |
| `LLM_MODEL` | `gpt-4o-mini` | 已读入配置；当前没有聊天或代理调用 |

## 项目结构

```
masskb/
├── docker-compose.yml          # Qdrant + Neo4j
├── pyproject.toml              # 包元数据、依赖、pytest / ruff 配置
├── requirements.txt            # 与 pyproject 运行依赖对齐的清单
├── .env.example                # 环境变量模板（不含可用密钥）
├── LICENSE                     # MIT
├── README.md
├── masskb/
│   ├── __init__.py             # 版本号 0.1.0
│   ├── __main__.py             # CLI，默认 healthcheck
│   ├── config.py               # Settings
│   ├── embeddings.py           # OpenAI 兼容 Embedding
│   ├── vector_store.py         # Qdrant 客户端
│   ├── graph_store.py          # Neo4j 驱动
│   └── pipeline.py             # 编排；search 未实现
├── scripts/healthcheck.py      # 可在未安装包时运行
├── tests/                      # 离线单元测试
├── docs/architecture/          # 目标架构图（drawio）及说明
└── .github/
    ├── workflows/ci.yml        # 安装 .[dev] 并运行 pytest
    ├── CODEOWNERS
    ├── PULL_REQUEST_TEMPLATE.md
    └── ISSUE_TEMPLATE/
```

`.gitignore` 排除 `.env`、虚拟环境、`data/` 以及仓库根目录上的若干内部规划文档。`data/` 预留给不入库的本地语料，仓库本身不附带语料。

| 模块 | 职责 |
|------|------|
| `masskb/config.py` | 读取配置；判断 API Key 是否为占位符 |
| `masskb/embeddings.py` | 按返回的 `index` 排序后给出向量 |
| `masskb/vector_store.py` | 懒加载 `QdrantClient`，`prefer_grpc=False` |
| `masskb/graph_store.py` | 懒加载 Bolt 驱动 |
| `masskb/pipeline.py` | 组装上述客户端；对外提供 `healthcheck` 与未实现的 `search` |
| `masskb/__main__.py` | `masskb` 控制台脚本的入口 |

## 测试与持续集成

| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_config.py` | 版本号、API Key 判断、环境变量、配置缓存、本机 `NO_PROXY` |
| `tests/test_embeddings.py` | 空输入、缺少真实 Key、客户端构造、按 index 排序 |
| `tests/test_stores.py` | 两个存储的健康检查成功、失败和 `close` |
| `tests/test_pipeline.py` | 离线健康检查、任一侧失败、`search` 抛出 `NotImplementedError` |
| `tests/test_cli.py` | CLI 退出码、默认子命令、帮助文本、`scripts/healthcheck.py` |

工作流文件是 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。触发条件：

- 推送到 `main`
- 任意 Pull Request

作业在 `ubuntu-latest` 上使用 Python 3.12，执行 `pip install -e ".[dev]"` 和 `python -m pytest`。CI 不启动 Docker，也不运行 `ruff`。本地若要检查风格，可在安装开发依赖后执行 `ruff check`。

## 许可证

本项目采用 [MIT License](LICENSE)。

Copyright (c) 2026 Black God.

# MassKB-检索

面向海量知识库的智能检索中间件（向量检索 + 知识图谱 + 智能代理）。

## 功能特性

- **向量检索**：基于 Qdrant 的语义相似度检索
- **知识图谱**：基于 Neo4j 的实体关系建模与图查询
- **智能代理**：结合 LLM 的检索编排与问答能力
- **可插拔后端**：支持 OpenAI 兼容 API，便于切换模型服务

## 技术栈

| 层级 | 技术 |
|------|------|
| 向量库 | Qdrant |
| 图数据库 | Neo4j 5 |
| 语言 / 运行时 | Python ≥ 3.10、Node.js ≥ 18（可选） |
| LLM / Embedding | OpenAI 兼容 API（openai SDK，不依赖本地 torch） |
| 编排 | Docker Compose |

## 快速开始

### 1. 启动依赖服务（Docker）

```bash
docker compose up -d
```

- Qdrant：`http://localhost:6333`（gRPC `6334`）
- Neo4j Browser：`http://localhost:7474`（Bolt `7687`）
  - 默认开发凭据见 `docker-compose.yml` 中 `NEO4J_AUTH`（示例密码，生产请修改）

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env：
#   - NEO4J_PASSWORD 与 docker-compose 中 NEO4J_AUTH 一致（本地开发示例：masskb-dev-pass）
#   - OPENAI_API_KEY 填入真实密钥后，才能做 embedding / 问答
#   - LLM_BASE_URL 默认为 https://api.openai.com/v1，可换成兼容网关
```

> **说明**：未配置真实 `OPENAI_API_KEY` 时，Qdrant / Neo4j 健康检查仍可通过；embedding 与问答不可用。

### 3. Python 虚拟环境与依赖

```bash
python3.12 -m venv .venv   # 或 python3.10+
source .venv/bin/activate
pip install -U pip
pip install -e .           # 或: pip install -r requirements.txt
```

核心依赖见 `pyproject.toml` / `requirements.txt`（`qdrant-client`、`neo4j`、`openai`、`python-dotenv`、`httpx`、`pydantic`）。  
**不**安装 `sentence-transformers` / `torch`：向量由 OpenAI 兼容 Embedding API 生成。

### 4. 健康检查

确认 Docker 中 `masskb-qdrant`、`masskb-neo4j` 已启动后：

```bash
# 方式一：独立脚本
.venv/bin/python scripts/healthcheck.py

# 方式二：模块入口
.venv/bin/python -m masskb healthcheck
```

成功时会打印 Qdrant / Neo4j 的 `[OK]`；若缺少真实 API Key，会提示 embedding 暂不可用（不阻塞）。

### 5. Node（可选）

本机需 Node.js v18+（推荐 v22）。

```bash
node --version
```

## 目录结构

```
MassKB-检索/
├── docker-compose.yml      # Qdrant + Neo4j
├── pyproject.toml          # 包元数据与依赖
├── requirements.txt        # pip 依赖清单（与 pyproject 对齐）
├── .env.example            # 环境变量模板
├── .gitignore
├── LICENSE                 # MIT
├── README.md
├── masskb/                 # Python 应用包
│   ├── __init__.py
│   ├── __main__.py         # CLI：healthcheck
│   ├── config.py           # 读取 .env
│   ├── vector_store.py     # Qdrant 封装
│   ├── graph_store.py      # Neo4j 封装
│   ├── embeddings.py       # OpenAI embeddings 客户端
│   └── pipeline.py         # 检索编排占位
├── scripts/
│   └── healthcheck.py      # 连通性检查脚本
├── docs/
│   └── architecture/       # 架构图（drawio）
├── .github/
│   ├── CODEOWNERS
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
└── data/                   # 本地语料（不入库）
```

## 许可证

本项目采用 [MIT License](LICENSE)。

Copyright (c) 2026 Black God. All rights reserved under the terms of the MIT License.

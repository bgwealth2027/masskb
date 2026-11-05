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
| 语言 / 运行时 | Python ≥ 3.10、Node.js ≥ 18 |
| LLM | OpenAI 兼容 API（openai SDK） |
| 编排 | Docker Compose |

## 快速开始

### 1. 启动依赖服务（Docker）

```bash
docker compose up -d
```

- Qdrant：`http://localhost:6333`（gRPC `6334`）
- Neo4j Browser：`http://localhost:7474`（Bolt `7687`）
  - 默认开发凭据见 `docker-compose.yml` 中 `NEO4J_AUTH`（示例密码，生产请修改）

健康检查：

```bash
curl -sf http://localhost:6333/readyz
# Neo4j：浏览器打开 7474，或使用 cypher-shell / Bolt 连通性检查
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 Neo4j 密码、OPENAI_API_KEY、LLM_BASE_URL 等
```

### 3. Python 虚拟环境

```bash
python3.12 -m venv .venv   # 或 python3.10+
source .venv/bin/activate
pip install -U pip
pip install qdrant-client neo4j openai sentence-transformers
```

### 4. Node（可选）

本机需 Node.js v18+（推荐 v22）。

```bash
node --version
```

## 目录结构

```
MassKB-检索/
├── docker-compose.yml      # Qdrant + Neo4j
├── .env.example            # 环境变量模板
├── .gitignore
├── LICENSE                 # MIT
├── README.md
├── .github/
│   ├── CODEOWNERS
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
└── data/                   # 本地数据（不入库）
```

## 许可证

本项目采用 [MIT License](LICENSE)。

Copyright (c) 2026 Black God. All rights reserved under the terms of the MIT License.

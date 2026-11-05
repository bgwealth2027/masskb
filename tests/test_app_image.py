"""应用镜像与 Compose 的静态契约。不启动 Docker，不读取密钥。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DEV_PASSWORD = "${NEO4J_DEV_PASSWORD:-masskb-dev-pass}"


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    cleaned = text.replace(",", " ").replace("[", " ").replace("]", " ").replace('"', " ")
    return " ".join(cleaned.split())


def _service_blocks(text: str) -> dict[str, str]:
    """按顶层 services 下的两空格服务名切块。不解析完整 YAML。"""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    in_services = False
    for line in text.splitlines():
        if line.startswith("services:"):
            in_services = True
            current = None
            continue
        if in_services and line and not line.startswith(" ") and not line.startswith("#"):
            in_services = False
            current = None
        if not in_services:
            continue
        if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":"):
            current = line.strip()[:-1]
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(line)
    return {name: "\n".join(body) for name, body in blocks.items()}


def test_dockerfile_runs_readonly_serve() -> None:
    text = _read("Dockerfile")
    assert text.startswith("FROM python:3.12-slim\n")
    assert '".[api]"' in text
    assert "masskb serve --host 0.0.0.0 --port 8080" in _flat(text)
    forbidden = (
        "sentence-transformers",
        "torch",
        "requirements.txt",
        "sk-",
        "OPENAI_API_KEY=",
    )
    for item in forbidden:
        assert item not in text


def test_dockerignore_excludes_env_file() -> None:
    assert ".env" in _read(".dockerignore").splitlines()


def test_compose_app_service_contract() -> None:
    text = _read("docker-compose.yml")
    blocks = _service_blocks(text)
    assert set(blocks) >= {"qdrant", "neo4j", "app"}

    for published in ("6333:6333", "6334:6334", "7474:7474", "7687:7687"):
        assert published in text

    app = blocks["app"]
    neo4j = blocks["neo4j"]
    assert "image: masskb-app:local" in app
    assert "container_name: masskb-app" in app
    assert "build: ." in app
    assert "QDRANT_URL: http://qdrant:6333" in app
    assert "NEO4J_URI: bolt://neo4j:7687" in app
    assert "masskb serve --host 0.0.0.0 --port 8080" in _flat(app)
    assert "127.0.0.1:8080:8080" in app
    assert "qdrant:" in app and "neo4j:" in app
    assert app.count("condition: service_healthy") == 2
    assert f"NEO4J_AUTH: neo4j/{_DEV_PASSWORD}" in neo4j or (
        f'"neo4j/{_DEV_PASSWORD}"' in neo4j
    )
    assert f"NEO4J_PASSWORD: {_DEV_PASSWORD}" in app or f'"{_DEV_PASSWORD}"' in app
    assert "OPENAI_API_KEY: ${OPENAI_API_KEY:-}" in app or '"${OPENAI_API_KEY:-}"' in app
    assert "sk-" not in text
    assert "env_file" not in text
    assert "ingest" not in text
    assert "bge" not in text.lower()

#!/bin/bash
# 公开部署沿用 NEO4J_DEV_PASSWORD。空值或仓库里的开发默认密码在启动 Neo4j 前拒绝。
# --check 只做判断，给测试用，不进入镜像的默认命令。
set -eu
password="${NEO4J_DEV_PASSWORD:-}"
if [ -z "$password" ] || [ "$password" = "masskb-dev-pass" ]; then
  echo "拒绝空密码或默认开发密码" >&2
  exit 1
fi
if [ "${1:-}" = "--check" ]; then
  exit 0
fi
export NEO4J_AUTH="neo4j/${password}"
exec /startup/docker-entrypoint.sh neo4j

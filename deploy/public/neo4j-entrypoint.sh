#!/bin/sh
# 公开部署沿用 NEO4J_DEV_PASSWORD。空值或开发默认密码时，任何启动命令都不会执行。
# --check 只做判断。--neo4j 进入官方镜像入口。其它参数原样 exec，供应用、网关和入库使用。
set -eu
password="${NEO4J_DEV_PASSWORD:-}"
if [ -z "$password" ] || [ "$password" = "masskb-dev-pass" ]; then
  echo "拒绝空密码或默认开发密码" >&2
  exit 1
fi
if [ "${1:-}" = "--check" ]; then
  exit 0
fi
if [ "${1:-}" = "--neo4j" ]; then
  export NEO4J_AUTH="neo4j/${password}"
  exec /startup/docker-entrypoint.sh neo4j
fi
exec "$@"

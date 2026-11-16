#!/bin/sh
# LangGraph runtime: ensure runtime_checkpoints DB + migrations, then run the wrapped command.
set -eu

/bin/sh /opt/askpdf/docker_compose_runtime_db_bootstrap.sh

exec /bin/sh /opt/askpdf/docker_load_env_secrets.sh "$@"
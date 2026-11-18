#!/bin/sh
# LangGraph runtime: ensure runtime_checkpoints DB + migrations, then run the wrapped command.
set -eu

/bin/sh /opt/askpdf/docker_compose_runtime_db_bootstrap.sh

if [ -f "${ASKPDF_ENV_SECRETS_READY:-/secrets/ready}" ]; then
  exec /bin/sh /opt/askpdf/docker_load_env_secrets.sh "$@"
fi
exec "$@"
#!/bin/sh
# Product rag-service: apply app DB migrations (advisory lock), then run the wrapped command.
set -eu

if [ "${RUN_DB_MIGRATIONS:-true}" != "false" ] && [ "${RUN_DB_MIGRATIONS:-true}" != "0" ]; then
  /bin/sh /opt/askpdf/docker_compose_db_migrate.sh app.db.migrate
fi

exec /bin/sh /opt/askpdf/docker_load_env_secrets.sh "$@"
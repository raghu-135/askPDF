#!/bin/sh
# Load Compose-generated service secrets, then exec the container command.
set -eu

READY_FILE="${ASKPDF_ENV_SECRETS_READY:-/secrets/ready}"
SECRETS_FILE="${ASKPDF_ENV_SECRETS_FILE:-/secrets/runtime.env}"
WAIT_ATTEMPTS="${ASKPDF_ENV_SECRETS_WAIT_ATTEMPTS:-150}"

i=0
while [ ! -f "$READY_FILE" ]; do
    i=$((i + 1))
    if [ "$i" -gt "$WAIT_ATTEMPTS" ]; then
        echo "timed out waiting for $READY_FILE" >&2
        exit 1
    fi
    sleep 0.2
done

if [ ! -f "$SECRETS_FILE" ]; then
    echo "missing service secrets file $SECRETS_FILE" >&2
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
        '' | \#*)
            continue
            ;;
    esac
    key=${line%%=*}
    value=${line#*=}
    export "$key=$value"
done <"$SECRETS_FILE"

if [ -n "${HERMES_API_TOKEN:-}" ]; then
    export API_SERVER_KEY="$HERMES_API_TOKEN"
fi

clear_list=${ASKPDF_CLEAR_ENV:-}
old_ifs=$IFS
IFS=,
for name in $clear_list; do
    name=$(printf '%s' "$name" | tr -d ' ')
    if [ -n "$name" ]; then
        unset "$name" || true
    fi
done
IFS=$old_ifs

if [ "$#" -eq 0 ]; then
    echo "docker_load_env_secrets.sh requires a command" >&2
    exit 1
fi

exec "$@"

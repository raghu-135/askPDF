#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"

docker run --rm askpdf-runtime-control-plane python -c 'import importlib.util; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm askpdf-langgraph-runtime python -c 'import importlib.util, langgraph, runtime_protocol, langgraph_runtime; assert importlib.util.find_spec("app") is None'
docker run --rm askpdf-hermes-runtime python -c 'import importlib.util, hermes_runtime, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("app") is None'
docker run --rm askpdf-control-plane-tests python -c 'import importlib.util, app, runtime_protocol; assert importlib.util.find_spec("langgraph") is None; assert importlib.util.find_spec("langgraph_runtime") is None'
docker run --rm askpdf-langgraph-runtime-tests python -c 'import importlib.util, langgraph, langgraph_runtime, runtime_protocol; assert importlib.util.find_spec("app") is None'

echo "Framework image boundary checks passed"
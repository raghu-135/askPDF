#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="${1:?sweep config json name without path}"
shift
PATCH="${REPO_ROOT}/evals/sweeps/${SWEEP}.json"
test -f "$PATCH" || { echo "missing $PATCH" >&2; exit 1; }
BASE_CFG="${REPO_ROOT}/benchmarks/benchmark_config.eval.json"
MERGED="$(python3 -c "import json; from pathlib import Path; b=json.loads(Path('${BASE_CFG}').read_text()); p=json.loads(Path('${PATCH}').read_text());
for k,v in p.items():
  if isinstance(v,dict) and isinstance(b.get(k),dict): b[k]={**b[k],**v}
  else: b[k]=v
print(json.dumps(b))")"
export ASKPDF_BENCHMARK_CONFIG="${REPO_ROOT}/benchmarks/.sweep-merged.json"
echo "$MERGED" > "$ASKPDF_BENCHMARK_CONFIG"
ASKPDF_BENCHMARK_CONFIG="$ASKPDF_BENCHMARK_CONFIG" \
  "${REPO_ROOT}/evals/promptfoo/run_eval.sh" --suite sweep_core --verify-settings --label "sweep-${SWEEP}" "$@"
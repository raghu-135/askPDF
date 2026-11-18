#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="${ASKPDF_EVAL_ARTIFACTS:-$HOME/askpdf-worktrees/evals-artifacts}"
mkdir -p "$ARTIFACTS"
OUT="$ARTIFACTS/override-verification.jsonl"
: >"$OUT"
SWEEPS=(
  document_ratio_0.20
  document_ratio_0.36
  recent_ratio_0.14
  recent_ratio_0.30
  coverage_1_3
  coverage_3_5
  web_search_bonus_0
  web_search_bonus_4
)
for name in "${SWEEPS[@]}"; do
  label="verify-${name}"
  log="$ARTIFACTS/verify-${name}.log"
  set +e
  "$REPO_ROOT/evals/promptfoo/run_sweep.sh" "$name" --judge --limit 1 --workflows router_rag_agent --label "$label" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  # Keep going when a sweep fails verification (recorded below).
  python3 - "$name" "$label" "$rc" "$log" >>"$OUT" <<'PY'
import json, sys
from pathlib import Path
name, label, rc, log_path = sys.argv[1:5]
rc = int(rc)
row = {"sweep": name, "label": label, "exit_code": rc, "applied": None, "reason": "", "effective": None}
text = Path(log_path).read_text(encoding="utf-8", errors="replace")
for line in text.splitlines():
    if line.startswith("{") and "effective_settings" in line:
        try:
            case = json.loads(line)
            row["effective"] = case.get("effective_settings")
            row["applied"] = case.get("settings_override_verified")
            row["reason"] = case.get("settings_verify_reason", "")
            break
        except json.JSONDecodeError:
            pass
if rc != 0 and row["applied"] is None:
    row["applied"] = False
    row["reason"] = row["reason"] or "run failed"
print(json.dumps(row))
PY
done
echo "wrote $OUT"
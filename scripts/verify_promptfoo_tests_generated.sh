#!/usr/bin/env bash
# Same gate as CI: questions.json must match committed promptfoo tests.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
GENERATED="${REPO_ROOT}/evals/promptfoo/tests.generated.yaml"

python3 "${REPO_ROOT}/scripts/generate_promptfoo_tests.py"
if ! git diff --exit-code "$GENERATED"; then
    echo "error: $GENERATED is out of date; commit the generator output" >&2
    exit 1
fi

python3 "${REPO_ROOT}/scripts/generate_promptfoo_tests.py"
if ! git diff --exit-code "$GENERATED"; then
    echo "error: generator is not idempotent for $GENERATED" >&2
    exit 1
fi

echo "promptfoo tests.generated.yaml OK"
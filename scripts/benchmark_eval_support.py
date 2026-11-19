"""Shared RAG eval orchestration: readiness, pacing, tiers, checkpoints, profiles."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = REPO / "benchmarks" / "questions.json"
HOLDOUT_PATH = REPO / "benchmarks" / "holdout_questions.json"
REPORTS_DIR = REPO / "benchmarks" / "reports"

TIER_ORDER = ("smoke", "core", "advanced", "holdout")


@dataclass(frozen=True)
class EvalProfile:
    name: str
    parallel: bool
    rpm: float | None
    min_interval_s: float
    max_workers: int
    daily_request_budget: int | None


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def default_max_workers() -> int:
    explicit = os.getenv("ASKPDF_EVAL_MAX_WORKERS", "").strip()
    if explicit:
        return max(1, _int_env("ASKPDF_EVAL_MAX_WORKERS", 1))
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2 or 1))


def resolve_profile(name: str | None = None) -> EvalProfile:
    profile = (name or os.getenv("ASKPDF_EVAL_PROFILE", "free")).strip().lower()
    if profile == "fast":
        rpm = _float_env("ASKPDF_EVAL_FAST_RPM", 0) or None
        interval = _float_env("ASKPDF_EVAL_MIN_INTERVAL_S", 0.25) if rpm is None else 60.0 / rpm
        return EvalProfile(
            name="fast",
            parallel=True,
            rpm=rpm,
            min_interval_s=max(0.0, interval),
            max_workers=default_max_workers(),
            daily_request_budget=_int_env("ASKPDF_EVAL_DAILY_REQUEST_BUDGET", 0) or None,
        )
    rpm = _float_env("ASKPDF_EVAL_FREE_RPM", 20.0)
    return EvalProfile(
        name="free",
        parallel=False,
        rpm=rpm,
        min_interval_s=max(0.0, 60.0 / rpm),
        max_workers=1,
        daily_request_budget=_int_env("ASKPDF_EVAL_DAILY_REQUEST_BUDGET", 0) or None,
    )


class RequestPacer:
    """Throttle and retry live LLM HTTP with 429 / timeout backoff."""

    def __init__(self, profile: EvalProfile) -> None:
        self._profile = profile
        self._lock = threading.Lock()
        self._last_at = 0.0
        self._request_count = 0
        self._count_day = datetime.now(timezone.utc).date()

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._count_day:
            self._count_day = today
            self._request_count = 0

    def budget_remaining(self) -> int | None:
        budget = self._profile.daily_request_budget
        if budget is None or budget <= 0:
            return None
        self._maybe_reset_daily()
        return max(0, budget - self._request_count)

    def wait_turn(self) -> None:
        budget = self.budget_remaining()
        if budget is not None and budget <= 0:
            raise RuntimeError(
                "ASKPDF_EVAL_DAILY_REQUEST_BUDGET exhausted for today; resume tomorrow or raise the budget"
            )
        with self._lock:
            now = time.monotonic()
            wait_s = self._profile.min_interval_s - (now - self._last_at)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_at = time.monotonic()
            self._maybe_reset_daily()
            self._request_count += 1

    def call(
        self,
        fn: Callable[[], Any],
        *,
        max_retries: int | None = None,
    ) -> Any:
        retries = max_retries if max_retries is not None else _int_env("ASKPDF_EVAL_HTTP_RETRIES", 8)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            self.wait_turn()
            try:
                return fn()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 429 and attempt < retries:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after and retry_after.replace(".", "", 1).isdigit():
                        delay = max(float(retry_after), self._profile.min_interval_s)
                    else:
                        delay = min(self._profile.min_interval_s * (2**attempt), 120.0)
                    time.sleep(delay)
                    continue
                raise
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(min(5.0 * (attempt + 1), 60.0))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("request pacing failed without exception")


def stack_api_base() -> str:
    return os.getenv("ASKPDF_BENCHMARK_API", os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api")).rstrip("/")


def product_base_url() -> str:
    api = stack_api_base()
    if api.endswith("/api"):
        return api[: -len("/api")]
    return api.replace("/api", "")


def wait_stack_ready(*, timeout_s: float | None = None, client: httpx.Client | None = None) -> None:
    """Poll /health and /ready until the product stack accepts traffic."""
    timeout = timeout_s if timeout_s is not None else _float_env("ASKPDF_EVAL_STACK_READY_TIMEOUT_S", 900.0)
    base = product_base_url()
    deadline = time.time() + timeout
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0))
    assert client is not None
    last_detail = ""
    try:
        while time.time() < deadline:
            try:
                health = client.get(f"{base}/health")
                health.raise_for_status()
                if health.json().get("status") != "ok":
                    last_detail = f"health status={health.json()}"
                    time.sleep(2.0)
                    continue
                ready = client.get(f"{base}/ready")
                if ready.status_code == 200 and ready.json().get("status") == "ok":
                    return
                last_detail = f"ready status_code={ready.status_code} body={ready.text[:500]}"
            except httpx.HTTPError as exc:
                last_detail = str(exc)
            time.sleep(2.0)
        raise TimeoutError(
            f"stack not ready after {timeout:.0f}s (base={base}; last={last_detail}). "
            "Start the app (e.g. bash scripts/cloud-agent-setup.sh stack) or increase ASKPDF_EVAL_STACK_READY_TIMEOUT_S"
        )
    finally:
        if own_client:
            client.close()


def load_question_bank(*, tier: str, include_holdout_file: bool = False) -> list[dict[str, Any]]:
    tier_norm = tier.strip().lower()
    if tier_norm == "holdout":
        data = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
        return list(data.get("questions") or [])
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = list(data.get("questions") or [])
    if tier_norm == "all":
        return cases
    filtered = [c for c in cases if str(c.get("tier") or "core").lower() == tier_norm]
    if include_holdout_file and tier_norm != "holdout":
        return filtered
    return filtered


def load_cases(
    *,
    tier: str,
    suite: str | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Select cases by eval tier and optional legacy suite tag (e.g. sweep_core)."""
    cases = load_question_bank(tier=tier)
    if suite:
        cases = [c for c in cases if suite in (c.get("suites") or ["baseline"])]
    if limit > 0:
        cases = cases[:limit]
    return cases


def bundled_fixture_names(cases: Iterable[dict[str, Any]]) -> list[str]:
    from benchmark_fixtures import list_bundled_fixtures

    allowed = set(list_bundled_fixtures())
    names: set[str] = set()
    for case in cases:
        for name in case.get("fixture_files") or []:
            if name not in allowed:
                raise ValueError(
                    f"case {case.get('id')}: fixture {name} is not in bundled_manifest.json "
                    "(eval corpus must be redistributable; see benchmarks/CORPUS.md)"
                )
            names.add(name)
    return sorted(names)


def checkpoint_path(label: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    return REPORTS_DIR / f".checkpoint-{safe}.json"


def load_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    done = data.get("completed_keys") or []
    return {str(x) for x in done}


def save_checkpoint(path: Path, completed: set[str], *, meta: dict[str, Any] | None = None) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_keys": sorted(completed),
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def case_result_key(case_id: str, turn_id: str | None, workflow: str) -> str:
    turn = turn_id or "single"
    return f"{workflow}:{case_id}:{turn}"


def resolve_chat_models() -> tuple[str, str]:
    primary = os.getenv("ASKPDF_BENCHMARK_MODEL", "nvidia/nemotron-3.5-super-vl:free")
    fallback = os.getenv("ASKPDF_BENCHMARK_MODEL_FALLBACK", "particle-ai/particle-1:free")
    if primary.strip().lower() in {"openrouter/free", "openrouter:free"}:
        primary = fallback
    return primary, fallback


def apply_reasoning_effort(effort: str | None) -> None:
    value = (effort or "").strip()
    if not value or value.lower() == "default":
        os.environ.pop("ASKPDF_EVAL_REASONING_EFFORT", None)
        os.environ.pop("ASKPDF_OPENROUTER_CHAT_EXTRA_BODY_JSON", None)
        return
    os.environ["ASKPDF_EVAL_REASONING_EFFORT"] = value
    os.environ.pop("ASKPDF_OPENROUTER_CHAT_EXTRA_BODY_JSON", None)


def run_parallel(
    items: list[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int,
) -> list[Any]:
    if max_workers <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results
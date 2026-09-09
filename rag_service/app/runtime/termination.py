"""Product cancellation intent, separate from authoritative runtime outcomes."""

from typing import Any



def cancellation_reason(task: Any, run: Any) -> str:
    request = (getattr(run, "run_metadata_json", None) or {}).get("cancellation_request") or {}
    return str(request.get("reason") or "cancelled_by_user")


def confirmed_cancellation_details(task: Any, run: Any) -> dict[str, Any]:
    request = dict((getattr(run, "run_metadata_json", None) or {}).get("cancellation_request") or {})
    return {**request, "reason": cancellation_reason(task, run), "runtime_confirmation": "confirmed"}

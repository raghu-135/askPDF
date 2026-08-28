"""Product-owned grounding evaluation for agent task results."""

from __future__ import annotations

from typing import Any, Mapping


DOCUMENT_EVIDENCE_TOOLS = frozenset({"search_documents", "search_document_by_id"})
RESEARCH_EVIDENCE_TOOLS = frozenset({
    "search_durable_memory",
    "search_web",
    "wikipedia",
    "wikidata",
    "arxiv",
    "pubmed",
    "semantic_scholar",
    "stack_exchange",
    "yahoo_finance_news",
})


def _event_payload(event: Any) -> Mapping[str, Any]:
    payload = getattr(event, "payload_json", None) or getattr(event, "payload", None) or {}
    return payload if isinstance(payload, Mapping) else {}


class AgentGroundingEvaluator:
    """Evaluate askPDF's evidence policy from neutral runtime records."""

    def evaluate(
        self,
        result: Mapping[str, Any],
        events: list[Any],
        *,
        documents_present: bool,
        artifacts: list[Any] | None = None,
    ) -> dict[str, Any]:
        successful_tools: list[Mapping[str, Any]] = []
        failures: list[Mapping[str, Any]] = []
        for event in events:
            payload = _event_payload(event)
            kind = str(getattr(event, "kind", ""))
            if kind == "tool.completed" and payload.get("ok") is True and int(payload.get("result_count") or 0) > 0:
                successful_tools.append(payload)
            elif kind == "tool.failed":
                failures.append(payload)

        eligible = DOCUMENT_EVIDENCE_TOOLS if documents_present else DOCUMENT_EVIDENCE_TOOLS | RESEARCH_EVIDENCE_TOOLS
        qualifying = [item for item in successful_tools if item.get("tool_name") in eligible]
        evidence = result.get("task_evidence_manifest") if isinstance(result, Mapping) else None
        report = result.get("grounding_report") if isinstance(result, Mapping) else None
        verified_claims = report.get("verified_claims") if isinstance(report, Mapping) else None
        fallback_count = (
            len(verified_claims)
            if isinstance(verified_claims, list)
            else self._evidence_artifact_count(evidence, artifacts)
            if isinstance(evidence, list)
            else 0
        )
        result_count = sum(int(item.get("result_count") or 0) for item in qualifying)
        return {
            "requirement": "document" if documents_present else "research",
            "grounded": bool(qualifying) or (not successful_tools and fallback_count > 0),
            "evidence_result_count": result_count or fallback_count,
            "successful_evidence_tools": sorted({str(item.get("tool_name")) for item in qualifying}),
            "failed_tool_count": len(failures),
            "failure_codes": sorted({str((item.get("error") or {}).get("code") or "tool_failed") for item in failures}),
        }

    @staticmethod
    def _evidence_artifact_count(evidence: list[Any], artifacts: list[Any] | None) -> int:
        if artifacts is None:
            return len(evidence)
        artifact_ids = {
            str(getattr(artifact, "id", "") or "")
            for artifact in artifacts
            if str(getattr(artifact, "validity", "valid") or "valid") == "valid"
            and getattr(artifact, "deleted_at", None) is None
        }
        return sum(1 for item in evidence if isinstance(item, Mapping) and str(item.get("id") or "") in artifact_ids)

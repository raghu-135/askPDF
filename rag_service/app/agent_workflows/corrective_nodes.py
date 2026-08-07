from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from app.agent_workflows.corrective_contracts import normalized_corrective_policy
from app.agent_workflows.evidence import corrective_evidence_packets
from app.agent_workflows.enums import CorrectiveRetrievalRoute, GroundedAnswerRoute


def _strings(values: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())[:500]
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _contradictions(values: Any) -> list[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [
        {
            "claim": " ".join(str(item.get("claim") or item.get("reason") or "conflicting evidence").split())[:1_000],
            "source_ids": _strings(item.get("source_ids"), limit=12),
            "claim_ids": _strings(item.get("claim_ids"), limit=20),
        }
        for item in values[:20]
        if isinstance(item, dict)
    ]


def retrieval_quality_contract_errors(value: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value.get("packet_assessments"), list):
        errors.append("packet_assessments must be an array")
    if not isinstance(value.get("missing_requirements"), list):
        errors.append("missing_requirements must be an array")
    if not isinstance(value.get("material_contradictions"), list):
        errors.append("material_contradictions must be an array")
    return errors


def normalize_retrieval_quality_report(parsed: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    packet_by_id = {
        str(packet.get("id")): packet for packet in state.get("evidence_packets", [])
        if isinstance(packet, dict) and packet.get("id")
    }
    assessments: list[Dict[str, Any]] = []
    assessed: set[str] = set()
    unknown_ids: list[str] = []
    for raw in parsed.get("packet_assessments", []) if isinstance(parsed.get("packet_assessments"), list) else []:
        if not isinstance(raw, dict):
            continue
        packet_id = str(raw.get("packet_id") or "")
        if packet_id not in packet_by_id:
            if packet_id:
                unknown_ids.append(packet_id)
            continue
        try:
            confidence = max(0.0, min(float(raw.get("confidence") or 0.0), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        assessment = {
            "packet_id": packet_id,
            "source_ids": list(packet_by_id[packet_id].get("source_ids") or []),
            "relevant": raw.get("relevant") is True,
            "confidence": confidence,
            "provenance_complete": raw.get("provenance_complete") is True and bool(packet_by_id[packet_id].get("source_ids")),
            "instruction_injection_risk": raw.get("instruction_injection_risk") is True,
            "coverage": _strings(raw.get("coverage")),
            "contradiction_signals": _strings(raw.get("contradiction_signals")),
        }
        assessments.append(assessment)
        assessed.add(packet_id)
    for packet_id, packet in packet_by_id.items():
        if packet_id not in assessed:
            assessments.append({
                "packet_id": packet_id,
                "source_ids": list(packet.get("source_ids") or []),
                "relevant": False,
                "confidence": 0.0,
                "provenance_complete": False,
                "instruction_injection_risk": False,
                "coverage": [],
                "contradiction_signals": ["grader did not return a valid assessment"],
            })
    contradictions = _contradictions(parsed.get("material_contradictions"))
    gaps = _strings(parsed.get("missing_requirements"))
    policy = normalized_corrective_policy(state.get("corrective_policy"))
    for item in assessments:
        reasons = []
        if not item["relevant"]:
            reasons.append("irrelevant")
        if not item["provenance_complete"]:
            reasons.append("incomplete_provenance")
        if item["instruction_injection_risk"]:
            reasons.append("instruction_injection_risk")
        if item["confidence"] < policy["minimum_relevance_confidence"]:
            reasons.append("below_relevance_threshold")
        if len(item["source_ids"]) != 1:
            reasons.append("invalid_source_binding")
        item["eligible"] = not reasons
        item["rejection_reasons"] = reasons
    eligible = [item for item in assessments if item["eligible"]]
    covered = {value.casefold() for item in eligible for value in item["coverage"]}
    required = _strings(state.get("evidence_gaps"))
    uncovered = [item for item in required if item.casefold() not in covered]
    gaps = list(dict.fromkeys([*gaps, *uncovered]))
    confidence = min((item["confidence"] for item in eligible), default=0.0)
    if not eligible:
        verdict = "incorrect"
    elif gaps or confidence < policy["minimum_relevance_confidence"] or contradictions:
        verdict = "ambiguous"
    else:
        verdict = "correct"
    return {
        "verdict": verdict,
        "confidence": confidence,
        "packet_assessments": assessments,
        "source_assessments": [{"source_id": source_id, "packet_id": item["packet_id"], "eligible": item["eligible"], "rejection_reasons": item["rejection_reasons"]} for item in assessments for source_id in item["source_ids"]],
        "missing_requirements": gaps,
        "material_contradictions": contradictions,
        "unknown_packet_ids": sorted(set(unknown_ids)),
        "reason": str(parsed.get("reason") or "")[:800],
    }


def corrective_route_for_report(report: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[str, str]:
    if report.get("verdict") == "correct":
        return CorrectiveRetrievalRoute.SYNTHESIZE.value, ""
    policy = normalized_corrective_policy(state.get("corrective_policy"))
    wave = max(0, int(state.get("corrective_wave") or 0))
    used_work = len({str(item.get("work_id")) for item in state.get("worker_result_packets", []) if isinstance(item, dict) and item.get("work_id")})
    used_attempts = len([item for item in state.get("parallel_attempt_records", []) if isinstance(item, dict)])
    if wave >= policy["max_corrective_waves"]:
        return CorrectiveRetrievalRoute.INSUFFICIENT.value, "max_corrective_waves"
    if used_work >= policy["max_total_work_items"]:
        return CorrectiveRetrievalRoute.INSUFFICIENT.value, "max_total_work_items"
    if used_attempts >= policy["max_total_tool_attempts"]:
        return CorrectiveRetrievalRoute.INSUFFICIENT.value, "max_total_tool_attempts"
    return CorrectiveRetrievalRoute.CORRECT.value, ""


def grounded_answer_contract_errors(value: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("claims", "citation_violations", "contradictions", "unresolved_gaps"):
        if not isinstance(value.get(key), list):
            errors.append(f"{key} must be an array")
    if not isinstance(value.get("usefulness_score"), int):
        errors.append("usefulness_score must be an integer")
    claims = value.get("claims") if isinstance(value.get("claims"), list) else []
    claim_ids = [str(item.get("claim_id") or "") for item in claims if isinstance(item, dict)]
    if any(not claim_id for claim_id in claim_ids) or len(set(claim_ids)) != len(claim_ids):
        errors.append("claims must have unique non-empty claim_id values")
    for item in value.get("contradictions") or []:
        if isinstance(item, dict) and not isinstance(item.get("claim_ids"), list):
            errors.append("each contradiction must contain claim_ids")
    return errors


def normalize_grounding_report(parsed: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Any]:
    valid_source_ids = {
        str(source_id) for packet in corrective_evidence_packets(dict(state))
        if isinstance(packet, dict) for source_id in packet.get("source_ids") or [] if source_id
    }
    claims: list[Dict[str, Any]] = []
    unknown_ids: set[str] = set()
    invalid_claim_ids: set[str] = set()
    seen_claim_ids: set[str] = set()
    for index, raw in enumerate(parsed.get("claims", [])[:20] if isinstance(parsed.get("claims"), list) else []):
        if not isinstance(raw, dict) or not str(raw.get("claim") or "").strip():
            continue
        claim_id = " ".join(str(raw.get("claim_id") or "").split())[:100]
        if not claim_id or claim_id in seen_claim_ids:
            invalid_claim_ids.add(claim_id or f"missing:{index}")
            claim_id = f"invalid:{index}"
        seen_claim_ids.add(claim_id)
        supplied = _strings(raw.get("source_ids"))
        unknown_ids.update(source_id for source_id in supplied if source_id not in valid_source_ids)
        source_ids = sorted(source_id for source_id in supplied if source_id in valid_source_ids)
        support = str(raw.get("support") or "none")
        if support not in {"full", "partial", "none"} or (support == "full" and not source_ids):
            support = "none"
        claims.append({
            "claim_id": claim_id,
            "claim": " ".join(str(raw.get("claim") or "").split())[:1_000],
            "support": support,
            "source_ids": source_ids,
            "contradicted": raw.get("contradicted") is True,
        })
    try:
        usefulness = max(1, min(int(parsed.get("usefulness_score") or 1), 5))
    except (TypeError, ValueError):
        usefulness = 1
    violations = _strings(parsed.get("citation_violations"))
    contradictions = _contradictions(parsed.get("contradictions"))
    claim_ids = {item["claim_id"] for item in claims}
    contradicted_claim_ids = {item["claim_id"] for item in claims if item["contradicted"]}
    unmapped_contradiction = False
    for contradiction in contradictions:
        supplied = list(contradiction.get("source_ids") or [])
        unknown_ids.update(source_id for source_id in supplied if source_id not in valid_source_ids)
        contradiction["source_ids"] = sorted(source_id for source_id in supplied if source_id in valid_source_ids)
        supplied_claim_ids = list(contradiction.get("claim_ids") or [])
        unknown_claim_ids = sorted(claim_id for claim_id in supplied_claim_ids if claim_id not in claim_ids)
        mapped_claim_ids = sorted(claim_id for claim_id in supplied_claim_ids if claim_id in claim_ids)
        contradiction["claim_ids"] = mapped_claim_ids
        if unknown_claim_ids or not mapped_claim_ids:
            unmapped_contradiction = True
            violations.append(
                "Unmapped contradiction claim ids: " + ", ".join(unknown_claim_ids or ["missing"])
            )
        contradicted_claim_ids.update(mapped_claim_ids)
    if unknown_ids:
        violations.append("Unknown citation ids: " + ", ".join(sorted(unknown_ids)))
    if invalid_claim_ids:
        violations.append("Invalid or duplicate claim ids: " + ", ".join(sorted(invalid_claim_ids)))
    prior_gaps = _strings(state.get("unresolved_gaps")) if state.get("corrective_retrieval_route") == "insufficient" else []
    prior_contradictions = _contradictions(state.get("contradiction_report")) if state.get("corrective_retrieval_route") == "insufficient" else []
    contradictions = [*contradictions, *(item for item in prior_contradictions if item not in contradictions)]
    if any(not item.get("claim_ids") for item in prior_contradictions):
        unmapped_contradiction = True
    full = [] if unmapped_contradiction else [
        item for item in claims
        if item["support"] == "full" and item["claim_id"] not in contradicted_claim_ids
    ]
    ratio = len(full) / len(claims) if claims else 0.0
    return {
        "claims": claims,
        "verified_claims": full,
        "supported_claim_ratio": ratio,
        "citation_violations": violations,
        "contradictions": contradictions,
        "unresolved_gaps": list(dict.fromkeys([*_strings(parsed.get("unresolved_gaps")), *prior_gaps])),
        "usefulness_score": usefulness,
        "valid_source_ids": sorted(valid_source_ids),
    }


def grounded_route_for_report(report: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[str, str]:
    policy = normalized_corrective_policy(state.get("corrective_policy"))
    passed = (
        float(report.get("supported_claim_ratio") or 0.0) >= policy["minimum_supported_claim_ratio"]
        and int(report.get("usefulness_score") or 0) >= policy["minimum_usefulness_score"]
        and not report.get("citation_violations")
        and not report.get("contradictions")
        and not report.get("unresolved_gaps")
    )
    if passed:
        return GroundedAnswerRoute.PASS.value, ""
    revision_count = max(0, int(state.get("answer_revision_count") or 0))
    if report.get("verified_claims") and revision_count < policy["max_answer_revisions"]:
        return GroundedAnswerRoute.REVISE.value, ""
    correction_route, reason = corrective_route_for_report({"verdict": "ambiguous"}, state)
    if correction_route == CorrectiveRetrievalRoute.CORRECT.value and (report.get("unresolved_gaps") or not report.get("verified_claims")):
        return GroundedAnswerRoute.CORRECT.value, ""
    return GroundedAnswerRoute.FINALIZE_CAUTIOUS.value, reason or "grounding_failed"

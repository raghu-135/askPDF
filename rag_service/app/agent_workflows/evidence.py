from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.agent_workflows.enums import EvidenceCompressionMode
from app.agent_workflows.corrective_contracts import CORRECTIVE_WORKFLOW_ID, normalized_corrective_policy
from app.agent.evidence_contract import canonical_source_id, normalized_canonical_source_id, normalized_source_url
from app.agent_workflows.trace import (
    available_document_refs,
    compact_preview,
    compact_refs,
    refs_from_documents,
    refs_from_messages,
    refs_from_web,
)
from app.time_utils import iso_utc_z, utc_now


NODE_RUNTIME_CONFIG_KEY = "agent_workflow_node_runtime"
EVIDENCE_PACKET_LIMIT = 12
EVIDENCE_PACKET_CONTENT_LIMIT = 2_000
EVIDENCE_TEXT_LIMIT = EVIDENCE_PACKET_LIMIT * (EVIDENCE_PACKET_CONTENT_LIMIT + 128)
FINAL_CONTEXT_CHAR_LIMIT = EVIDENCE_TEXT_LIMIT

_INSTRUCTION_INJECTION_PATTERNS = (
    (
        "instruction_override",
        re.compile(r"\b(?:ignore|disregard|override|bypass|forget)\b.{0,80}\b(?:system|developer|previous|prior)\b.{0,40}\b(?:instruction|message|prompt|policy)s?\b", re.IGNORECASE | re.DOTALL),
    ),
    (
        "secret_exfiltration",
        re.compile(r"\b(?:reveal|expose|print|return|show|leak)\b.{0,80}\b(?:secret|api[- ]?key|password|credential|system prompt)s?\b", re.IGNORECASE | re.DOTALL),
    ),
    (
        "tool_authorization",
        re.compile(
            r"\b(?:(?:authorize|enable|allow)\b.{0,40}\b(?:assistant|agent|model)\b.{0,40}\b(?:invoke|call|run|execute)"
            r"|(?:you|assistant|agent|model)\b.{0,30}\b(?:must|should|may|can)\b.{0,30}\b(?:invoke|call|run|execute))"
            r"\b.{0,40}\b(?:tool|web search|shell|command)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_manipulation",
        re.compile(r"\b(?:you are now|act as|assume the role of|switch (?:your )?role)\b", re.IGNORECASE),
    ),
    (
        "citation_fabrication",
        re.compile(r"\b(?:fabricate|invent|make up|fake)\b.{0,60}\b(?:citation|source|reference)s?\b", re.IGNORECASE | re.DOTALL),
    ),
)


def instruction_injection_reason_codes(value: Any) -> List[str]:
    """Return high-confidence reason codes without treating benign instructions as unsafe."""

    text = normalized_evidence_text(value)[:EVIDENCE_PACKET_CONTENT_LIMIT]
    if not text:
        return []
    return [code for code, pattern in _INSTRUCTION_INJECTION_PATTERNS if pattern.search(text)]


def _runtime_node_id(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = (((config or {}).get("configurable") or {}).get(NODE_RUNTIME_CONFIG_KEY))
    runtime = runtime if isinstance(runtime, dict) else {}
    node_id = runtime.get("node_id")
    return str(node_id) if isinstance(node_id, str) and node_id else fallback


def _runtime_node_type(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = (((config or {}).get("configurable") or {}).get(NODE_RUNTIME_CONFIG_KEY))
    runtime = runtime if isinstance(runtime, dict) else {}
    node_type = runtime.get("node_type")
    return str(node_type) if isinstance(node_type, str) and node_type else fallback


def _runtime_visit_index(config: Optional[RunnableConfig]) -> Optional[int]:
    runtime = (((config or {}).get("configurable") or {}).get(NODE_RUNTIME_CONFIG_KEY))
    runtime = runtime if isinstance(runtime, dict) else {}
    try:
        value = int(runtime.get("visit_index"))
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def format_prefetch_summary(bundle: Dict[str, Any]) -> str:
    parts = []
    if bundle.get("thread_shape_text"):
        parts.append("Thread shape tool output:\n" + str(bundle["thread_shape_text"]))
    if bundle.get("recent_history_text"):
        parts.append("Recent conversation:\n" + bundle["recent_history_text"])
    if bundle.get("semantic_history_text"):
        parts.append("Semantic memory:\n" + bundle["semantic_history_text"])
    if bundle.get("durable_memory_text"):
        parts.append(bundle["durable_memory_text"])
    if bundle.get("document_evidence_text"):
        parts.append("Document evidence:\n" + bundle["document_evidence_text"])
    documents = bundle.get("documents") or []
    if documents:
        names = [
            "- " + str(doc.get("file_name"))
            + f" ({doc.get('file_hash')})"
            + (f" — {doc.get('page_count')} pages" if doc.get("page_count") not in (None, "") else "")
            for doc in documents[:12]
        ]
        parts.append("Available documents:\n" + "\n".join(names))
    return "\n\n".join(parts).strip() or "No pre-fetched context is available."


def combine_evidence(existing: Any, addition: Any, *, label: str, limit: Optional[int] = None) -> str:
    existing_text = str(existing or "").strip()
    addition_text = str(addition or "").strip()
    if not addition_text:
        return existing_text
    labeled = f"[{label}]\n{addition_text}"
    combined = "\n\n".join(part for part in (existing_text, labeled) if part).strip()
    if isinstance(limit, int) and limit > 0 and len(combined) > limit:
        return combined[-limit:].lstrip()
    return combined


def context_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    policy = state.get("context_policy")
    return policy if isinstance(policy, dict) else {}


def context_policy_int(state: Dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(context_policy(state).get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def evidence_packet_limit(state: Dict[str, Any]) -> int:
    return context_policy_int(state, "evidence_packet_limit", EVIDENCE_PACKET_LIMIT)


def evidence_packet_content_limit(state: Dict[str, Any]) -> int:
    return context_policy_int(state, "evidence_packet_content_limit", EVIDENCE_PACKET_CONTENT_LIMIT)


def evidence_text_limit(state: Dict[str, Any]) -> int:
    packet_limit = evidence_packet_limit(state)
    content_limit = evidence_packet_content_limit(state)
    if "context_policy" not in state:
        return EVIDENCE_TEXT_LIMIT
    return max(1, packet_limit * (content_limit + 128))


def final_context_char_limit(state: Dict[str, Any]) -> int:
    return context_policy_int(state, "final_context_char_limit", evidence_text_limit(state) or FINAL_CONTEXT_CHAR_LIMIT)


def evidence_dedupe_enabled(state: Dict[str, Any]) -> bool:
    value = context_policy(state).get("evidence_dedupe", True)
    return value is not False


def evidence_compression_mode(state: Dict[str, Any]) -> str:
    mode = context_policy(state).get("evidence_compression", EvidenceCompressionMode.COMPACT.value)
    return str(mode or EvidenceCompressionMode.COMPACT.value)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=True)


def normalized_evidence_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def short_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def packet_source_ids(packet: Dict[str, Any]) -> List[str]:
    refs = packet.get("refs") if isinstance(packet.get("refs"), dict) else {}
    candidates = [packet]
    for value in refs.values():
        if isinstance(value, dict):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    values = {canonical_source_id(item) for item in candidates}
    return sorted(item for item in values if item)


def packet_fingerprint(*, kind: str, content: str, refs: Dict[str, Any]) -> str:
    return short_hash({"kind": kind, "content": normalized_evidence_text(content), "refs": refs or {}})


def dedupe_evidence_packets(packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped_reversed: List[Dict[str, Any]] = []
    for packet in reversed(packets):
        fingerprint = str(packet.get("fingerprint") or "")
        if not fingerprint:
            fingerprint = packet_fingerprint(
                kind=str(packet.get("kind") or packet.get("producer_node_type") or "evidence"),
                content=str(packet.get("content") or ""),
                refs=packet.get("refs") if isinstance(packet.get("refs"), dict) else {},
            )
            packet = {**packet, "fingerprint": fingerprint}
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped_reversed.append(packet)
    return list(reversed(deduped_reversed))


def compact_context_text(text: str, *, limit: int, mode: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if mode == EvidenceCompressionMode.COMPACT.value:
        lines: List[str] = []
        seen_lines: set[str] = set()
        for raw_line in value.splitlines():
            line = " ".join(raw_line.split())
            if not line:
                if lines and lines[-1]:
                    lines.append("")
                continue
            key = line.lower()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(line)
        value = "\n".join(lines).strip()
    if isinstance(limit, int) and limit > 0 and len(value) > limit:
        return value[-limit:].lstrip()
    return value


def evidence_packets(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    packets = state.get("evidence_packets")
    normalized = [item for item in packets if isinstance(item, dict)] if isinstance(packets, list) else []
    if evidence_dedupe_enabled(state):
        normalized = dedupe_evidence_packets(normalized)
    return normalized


def corrective_evidence_packets(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the bounded, grade-eligible evidence view for corrective synthesis."""

    if state.get("workflow_id") != CORRECTIVE_WORKFLOW_ID:
        return evidence_packets(state)[-evidence_packet_limit(state):]
    policy = normalized_corrective_policy(state.get("corrective_policy"))
    assessments = {
        str(item.get("packet_id")): item
        for item in state.get("evidence_assessments") or []
        if isinstance(item, dict) and item.get("packet_id")
    }
    selected: List[Dict[str, Any]] = []
    for packet in evidence_packets(state):
        packet_id = str(packet.get("id") or "")
        assessment = assessments.get(packet_id)
        if not assessment:
            continue
        try:
            confidence = float(assessment.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        source_ids = sorted({
            normalized
            for item in packet.get("source_ids") or []
            if (normalized := normalized_canonical_source_id(item))
        })
        if not (
            assessment.get("relevant") is True
            and assessment.get("provenance_complete") is True
            and assessment.get("instruction_injection_risk") is not True
            and confidence >= policy["minimum_relevance_confidence"]
            and len(source_ids) == 1
        ):
            continue
        selected.append({
            **packet,
            "source_ids": source_ids,
            "assessment_ref": packet_id,
            "evaluator_confidence": confidence,
            "coverage": sorted({str(item) for item in assessment.get("coverage") or [] if str(item).strip()}),
        })
    selected.sort(key=lambda item: (
        -len(item.get("coverage") or []),
        -float(item.get("evaluator_confidence") or 0.0),
        int(item.get("wave_id") or 0),
        int(item.get("work_ordinal") or 0),
        str((item.get("source_ids") or [""])[0]),
        str(item.get("id") or ""),
    ))
    return selected[:evidence_packet_limit(state)]


def corrective_evidence_context(state: Dict[str, Any]) -> str:
    parts: List[str] = []
    for packet in corrective_evidence_packets(state):
        content = compact_preview(packet.get("content"), limit=evidence_packet_content_limit(state))
        if not content:
            continue
        source_ids = ", ".join(packet.get("source_ids") or [])
        parts.append(f"[packet {packet.get('id')} | sources: {source_ids}]\n{content}")
    limit = final_context_char_limit(state)
    selected: List[str] = []
    used = 0
    for part in parts:
        normalized = compact_context_text(
            part,
            limit=len(part),
            mode=evidence_compression_mode(state),
        )
        separator_chars = 2 if selected else 0
        remaining = limit - used - separator_chars
        if remaining <= 0:
            break
        # Never admit a partial packet because truncation could sever its source
        # binding. Lower-ranked packets are dropped once the context is full.
        if len(normalized) > remaining:
            continue
        selected.append(normalized)
        used += separator_chars + len(normalized)
    return "\n\n".join(selected)


def evidence_context_from_packets(state: Dict[str, Any]) -> str:
    parts = []
    for packet in evidence_packets(state)[-evidence_packet_limit(state):]:
        content = compact_preview(packet.get("content"), limit=evidence_packet_content_limit(state))
        if not content:
            continue
        kind = str(packet.get("kind") or packet.get("producer_node_type") or "evidence")
        producer = str(packet.get("producer_node_id") or packet.get("producer_node_type") or "unknown")
        parts.append(f"[{kind} evidence from {producer}]\n{content}")
    return compact_context_text(
        "\n\n".join(parts).strip(),
        limit=final_context_char_limit(state),
        mode=evidence_compression_mode(state),
    )


def final_context_from_state(state: Dict[str, Any]) -> tuple[str, str]:
    policy = context_policy(state)
    if state.get("workflow_id") == CORRECTIVE_WORKFLOW_ID and state.get("retrieval_quality_report"):
        return corrective_evidence_context(state), "corrective_eligible_evidence"
    if policy.get("final_prompt_assembly") == "evidence_packets":
        packet_context = evidence_context_from_packets(state)
        if packet_context:
            return packet_context, "evidence_packets"
    if state.get("evidence"):
        return compact_context_text(
            str(state.get("evidence") or ""),
            limit=final_context_char_limit(state),
            mode=evidence_compression_mode(state),
        ), "worker_evidence"
    return compact_context_text(
        format_prefetch_summary(state.get("pre_fetch_bundle") or {}),
        limit=final_context_char_limit(state),
        mode=evidence_compression_mode(state),
    ), "prefetch"


def append_evidence_packet(
    state: Dict[str, Any],
    config: RunnableConfig,
    *,
    kind: str,
    content: Any,
    refs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    text = compact_preview(content, limit=evidence_packet_content_limit(state))
    if not text:
        return evidence_packets(state)
    node_id = _runtime_node_id(config, kind)
    node_type = _runtime_node_type(config, kind)
    visit_index = _runtime_visit_index(config) or 1
    refs = refs or {}
    fingerprint = packet_fingerprint(kind=kind, content=text, refs=refs)
    existing_packets = evidence_packets(state)
    if evidence_dedupe_enabled(state):
        existing_packets = [packet for packet in existing_packets if packet.get("fingerprint") != fingerprint]
    packet = {
        "id": f"{node_id}:visit:{visit_index}:{kind}:{len(existing_packets) + 1}",
        "producer_node_id": node_id,
        "producer_node_type": node_type,
        "visit_index": visit_index,
        "kind": kind,
        "content": text,
        "content_hash": short_hash(normalized_evidence_text(text)),
        "fingerprint": fingerprint,
        "refs": refs,
        "source_ids": packet_source_ids({"refs": refs}),
        "created_at": iso_utc_z(utc_now()),
    }
    return [*existing_packets, packet][-evidence_packet_limit(state):]


def append_corrective_evidence_packets(
    state: Dict[str, Any],
    config: RunnableConfig,
    *,
    segments: Any,
) -> List[Dict[str, Any]]:
    """Append source-bound packets; malformed or multi-source input is ignored."""

    existing_packets = evidence_packets(state)
    node_id = _runtime_node_id(config, "evidence")
    node_type = _runtime_node_type(config, "evidence")
    visit_index = _runtime_visit_index(config) or 1
    appended: List[Dict[str, Any]] = []
    for raw in segments if isinstance(segments, list) else []:
        if not isinstance(raw, dict):
            continue
        source_id = normalized_canonical_source_id(raw.get("source_id"))
        content = compact_preview(raw.get("content"), limit=evidence_packet_content_limit(state))
        if not source_id or not content:
            continue
        kind = str(raw.get("kind") or "evidence")
        fingerprint = packet_fingerprint(kind=kind, content=content, refs={"source_id": source_id})
        packet_id = f"{node_id}:visit:{visit_index}:{kind}:{short_hash({'source_id': source_id, 'content': content})}"
        appended.append({
            "id": packet_id,
            "producer_node_id": node_id,
            "producer_node_type": node_type,
            "visit_index": visit_index,
            "kind": kind,
            "content": content,
            "content_hash": short_hash(normalized_evidence_text(content)),
            "fingerprint": fingerprint,
            "refs": {"source": {"source_id": source_id, **(raw.get("display") or {}), **(raw.get("locator") or {})}},
            "source_ids": [source_id],
            "raw_retriever_score": raw.get("raw_score"),
            "created_at": iso_utc_z(utc_now()),
        })
    return dedupe_evidence_packets([*existing_packets, *appended])[-evidence_packet_limit(state):]


def prefetch_refs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return compact_refs(
        {
            "recent_messages": refs_from_messages(bundle.get("recent_message_refs")),
            "semantic_matches": refs_from_messages(bundle.get("semantic_memory_refs") or bundle.get("used_chat_ids")),
            "memories": bundle.get("durable_memory_refs") or [],
            "document_matches": refs_from_documents(bundle.get("document_sources")),
            "web_sources": refs_from_web(bundle.get("web_sources")),
            "available_documents": available_document_refs(bundle.get("documents")),
        }
    )


def state_evidence_refs(state: Dict[str, Any]) -> Dict[str, Any]:
    return compact_refs(
        {
            "document_matches": refs_from_documents(state.get("document_sources")),
            "web_sources": refs_from_web(state.get("web_sources")),
            "messages": refs_from_messages(state.get("used_chat_ids")),
            "memories": [{"memory_id": item} for item in state.get("used_memory_ids", []) if item],
        }
    )

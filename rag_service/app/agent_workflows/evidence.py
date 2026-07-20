from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.agent_workflows.enums import EvidenceCompressionMode
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
    if bundle.get("recent_history_text"):
        parts.append("Recent conversation:\n" + bundle["recent_history_text"])
    if bundle.get("semantic_history_text"):
        parts.append("Semantic memory:\n" + bundle["semantic_history_text"])
    if bundle.get("document_evidence_text"):
        parts.append("Document evidence:\n" + bundle["document_evidence_text"])
    documents = bundle.get("documents") or []
    if documents:
        names = [f"- {doc.get('file_name')} ({doc.get('file_hash')})" for doc in documents[:12]]
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
        "created_at": iso_utc_z(utc_now()),
    }
    return [*existing_packets, packet][-evidence_packet_limit(state):]


def prefetch_refs(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return compact_refs(
        {
            "recent_messages": refs_from_messages(bundle.get("recent_message_refs")),
            "semantic_matches": refs_from_messages(bundle.get("semantic_memory_refs") or bundle.get("used_chat_ids")),
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
        }
    )

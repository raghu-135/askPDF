from __future__ import annotations

from typing import Any, Dict

from app.agent_workflows.enums import EvidenceCompressionMode
from app.agent_workflows.parallel_contracts import (
    DEFAULT_PARALLEL_POLICY,
    PARALLEL_POLICY_BOOLEAN_FIELDS,
    PARALLEL_POLICY_LIMITS,
)
from app.agent_workflows.workflow_runtime import ALLOWED_WORKFLOW_CONFIG_KEYS
from app.models.llm_server_client import (
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    MAX_SYSTEM_ROLE_CHARS,
    REPLANS_LIMIT,
)
from app.agent_workflows.execution_contracts import PREFETCH_MODES
from app.agent_workflows.corrective_contracts import collect_corrective_policy_errors


CONTEXT_FINAL_PROMPT_ASSEMBLIES = {"evidence_packets"}
CONTEXT_EVIDENCE_COMPRESSION_MODES = {mode.value for mode in EvidenceCompressionMode}
def collect_config_errors(config: Dict[str, Any], workflow_id: Any) -> list[str]:
    errors: list[str] = []
    errors.extend(collect_corrective_policy_errors(config.get("corrective_policy")))
    unknown_keys = sorted(set(config) - ALLOWED_WORKFLOW_CONFIG_KEYS)
    if unknown_keys:
        errors.append(f"unknown config keys: {', '.join(unknown_keys)}")

    for key in ("use_web_search", "use_reranker"):
        if key in config and not isinstance(config[key], bool):
            errors.append(f"{key} must be a boolean")

    parallel_policy = config.get("parallel_policy")
    if parallel_policy is not None:
        if not isinstance(parallel_policy, dict):
            errors.append("parallel_policy must be an object")
        else:
            known = {*PARALLEL_POLICY_BOOLEAN_FIELDS, *PARALLEL_POLICY_LIMITS}
            unknown = sorted(set(parallel_policy) - known)
            if unknown:
                errors.append(f"parallel_policy has unknown keys: {', '.join(unknown)}")
            for key in PARALLEL_POLICY_BOOLEAN_FIELDS:
                if key in parallel_policy and not isinstance(parallel_policy[key], bool):
                    errors.append(f"parallel_policy.{key} must be a boolean")
            for key, (minimum, maximum) in PARALLEL_POLICY_LIMITS.items():
                if key not in parallel_policy:
                    continue
                value = parallel_policy[key]
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"parallel_policy.{key} must be an integer")
                elif value < minimum or value > maximum:
                    errors.append(f"parallel_policy.{key} must be between {minimum} and {maximum}")
            successes = parallel_policy.get("minimum_successes", DEFAULT_PARALLEL_POLICY["minimum_successes"])
            work_items = parallel_policy.get("max_work_items", DEFAULT_PARALLEL_POLICY["max_work_items"])
            if isinstance(successes, int) and isinstance(work_items, int) and successes > work_items:
                errors.append("parallel_policy.minimum_successes cannot exceed max_work_items")

    if "replans" in config:
        replans = config.get("replans")
        if not isinstance(replans, int):
            errors.append("replans must be an integer")
        elif replans < 1 or replans > REPLANS_LIMIT:
            errors.append(f"replans must be between 1 and {REPLANS_LIMIT}")

    system_role = config.get("system_role", "")
    if not isinstance(system_role, str) or len(system_role) > MAX_SYSTEM_ROLE_CHARS:
        errors.append(f"system_role must be a string up to {MAX_SYSTEM_ROLE_CHARS} characters")

    custom_instructions = config.get("custom_instructions", "")
    if not isinstance(custom_instructions, str) or len(custom_instructions) > MAX_CUSTOM_INSTRUCTIONS_CHARS:
        errors.append(f"custom_instructions must be a string up to {MAX_CUSTOM_INSTRUCTIONS_CHARS} characters")

    tool_instructions = config.get("tool_instructions", {})
    if not isinstance(tool_instructions, dict):
        errors.append("tool_instructions must be an object")
    elif not all(isinstance(k, str) and isinstance(v, str) for k, v in tool_instructions.items()):
        errors.append("tool_instructions keys and values must be strings")

    prefetch_policy = config.get("prefetch_policy", {})
    if not isinstance(prefetch_policy, dict):
        errors.append("prefetch_policy must be an object")
    else:
        unknown_prefetch_keys = set(prefetch_policy) - {"enabled", "mode"}
        if unknown_prefetch_keys:
            errors.append("prefetch_policy only supports enabled and mode")
        if "enabled" in prefetch_policy and not isinstance(prefetch_policy["enabled"], bool):
            errors.append("prefetch_policy.enabled must be a boolean")
        if "mode" in prefetch_policy and prefetch_policy["mode"] not in PREFETCH_MODES:
            errors.append("prefetch_policy.mode must be evidence or routing")

    context_policy = config.get("context_policy", {})
    if not isinstance(context_policy, dict):
        errors.append("context_policy must be an object")
    else:
        unknown_context_keys = sorted(
            set(context_policy)
            - {
                "evidence_packet_limit",
                "evidence_packet_content_limit",
                "final_prompt_assembly",
                "evidence_dedupe",
                "evidence_compression",
                "final_context_char_limit",
            }
        )
        if unknown_context_keys:
            errors.append(f"context_policy has unknown keys: {', '.join(unknown_context_keys)}")
        for key in ("evidence_packet_limit", "evidence_packet_content_limit", "final_context_char_limit"):
            if key in context_policy:
                try:
                    value = int(context_policy[key])
                except (TypeError, ValueError):
                    value = 0
                if value < 1:
                    errors.append(f"context_policy.{key} must be a positive integer")
        if "final_prompt_assembly" in context_policy and not isinstance(context_policy["final_prompt_assembly"], str):
            errors.append("context_policy.final_prompt_assembly must be a string")
        elif (
            "final_prompt_assembly" in context_policy
            and context_policy["final_prompt_assembly"] not in CONTEXT_FINAL_PROMPT_ASSEMBLIES
        ):
            errors.append(
                "context_policy.final_prompt_assembly must be one of: "
                f"{', '.join(sorted(CONTEXT_FINAL_PROMPT_ASSEMBLIES))}"
            )
        if "evidence_dedupe" in context_policy and not isinstance(context_policy["evidence_dedupe"], bool):
            errors.append("context_policy.evidence_dedupe must be a boolean")
        if "evidence_compression" in context_policy and not isinstance(context_policy["evidence_compression"], str):
            errors.append("context_policy.evidence_compression must be a string")
        elif (
            "evidence_compression" in context_policy
            and context_policy["evidence_compression"] not in CONTEXT_EVIDENCE_COMPRESSION_MODES
        ):
            errors.append(
                "context_policy.evidence_compression must be one of: "
                f"{', '.join(sorted(CONTEXT_EVIDENCE_COMPRESSION_MODES))}"
            )
    return errors

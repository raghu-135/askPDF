"""Exact LangGraph deployment and definition capability profiles."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeFeatureDescriptor,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def checkpoint_database_url(environ: Mapping[str, str] | None = None) -> str:
    values = environ or os.environ
    url = values.get("AGENT_CHECKPOINT_DATABASE_URL") or values.get("DATABASE_URL") or ""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    return url


@dataclass(frozen=True)
class LangGraphDeploymentProfile:
    runtime_mode: str
    checkpointer_backend: str
    checkpoint_available: bool
    durable_persistence: bool
    runtime_available: bool
    configuration_error: str | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "LangGraphDeploymentProfile":
        values = environ or os.environ
        runtime_mode = str(values.get("AGENT_RUNTIME_MODE") or "external").strip().lower()
        if runtime_mode not in {"external", "in_process"}:
            return cls(
                runtime_mode=runtime_mode,
                checkpointer_backend=str(values.get("ASKPDF_AGENT_CHECKPOINTER") or "memory"),
                checkpoint_available=False,
                durable_persistence=False,
                runtime_available=False,
                configuration_error="AGENT_RUNTIME_MODE must be 'external' or 'in_process'",
            )

        backend = str(values.get("ASKPDF_AGENT_CHECKPOINTER") or "memory").strip().lower()
        if backend == "memory":
            available = _module_available("langgraph.checkpoint.memory")
            return cls(
                runtime_mode=runtime_mode,
                checkpointer_backend=backend,
                checkpoint_available=available,
                durable_persistence=False,
                runtime_available=available,
                configuration_error=None if available else "LangGraph memory checkpointer is unavailable",
            )
        if backend == "postgres":
            url_available = bool(checkpoint_database_url(values))
            saver_available = _module_available("langgraph.checkpoint.postgres.aio")
            available = url_available and saver_available
            error = None
            if not url_available:
                error = "LangGraph Postgres checkpointer requires a database URL"
            elif not saver_available:
                error = "LangGraph Postgres checkpointer is unavailable"
            return cls(
                runtime_mode=runtime_mode,
                checkpointer_backend=backend,
                checkpoint_available=available,
                durable_persistence=available,
                runtime_available=available,
                configuration_error=error,
            )
        return cls(
            runtime_mode=runtime_mode,
            checkpointer_backend=backend,
            checkpoint_available=False,
            durable_persistence=False,
            runtime_available=False,
            configuration_error=f"Unsupported ASKPDF_AGENT_CHECKPOINTER value: {backend!r}",
        )

    def deployment_metadata(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "checkpointer_backend": self.checkpointer_backend,
            "checkpoint_available": self.checkpoint_available,
            "durable_persistence": self.durable_persistence,
            "runtime_available": self.runtime_available,
            "configuration_error": self.configuration_error,
        }


def _unsupported() -> RuntimeOperationDescriptor:
    return RuntimeOperationDescriptor(
        RuntimeSupportLevel.UNSUPPORTED,
        RuntimeOperationOwner.RUNTIME,
        False,
        disabled_reason="runtime_capability_unsupported",
    )


def _feature(
    enabled: bool,
    *,
    semantics: str,
    details: Mapping[str, Any] | None = None,
) -> RuntimeFeatureDescriptor:
    return RuntimeFeatureDescriptor(
        RuntimeSupportLevel.NATIVE if enabled else RuntimeSupportLevel.UNSUPPORTED,
        enabled,
        disabled_reason=None if enabled else "definition_capability_unavailable",
        semantics=semantics,
        details=dict(details or {}),
    )


def _deep_agents_features(definition: AgentDefinition) -> dict[str, RuntimeFeatureDescriptor]:
    metadata = definition.definition_metadata
    node_types = {str(value) for value in metadata.get("graph_node_types", [])}
    tools = {str(value) for value in metadata.get("allowed_tool_ids", [])}
    profiles = {str(value) for value in metadata.get("task_profiles", [])}
    features = definition.capabilities
    is_deep = (
        definition.category == "deep"
        or definition.definition_id == "deep_research_agent"
        or "deep_research_subagent" in node_types
    )
    if not is_deep:
        return {}
    planning = bool(features.get("supports_replans")) or "deep_task_planner" in node_types
    parallel = bool(features.get("supports_parallel_dispatch")) or "parallel_dispatch" in node_types
    artifacts = bool(features.get("supports_artifacts"))
    subagents = "deep_research_subagent" in node_types and bool(profiles)
    memory = "durable_memory" in tools
    return {
        "planning": _feature(planning, semantics="definition_planner_nodes"),
        "parallel_dispatch": _feature(parallel, semantics="definition_parallel_dispatch"),
        "artifacts": _feature(artifacts, semantics="definition_artifact_policy"),
        "subagent_orchestration": _feature(
            subagents,
            semantics="product_managed_subagents",
            details={"profiles": sorted(profiles)},
        ),
        "memory": _feature(memory, semantics="definition_tool_policy", details={"tool_id": "durable_memory"}),
        "tools": _feature(bool(tools), semantics="definition_tool_policy", details={"count": len(tools)}),
    }


def langgraph_definition_features(definition: AgentDefinition) -> dict[str, RuntimeFeatureDescriptor]:
    """Return definition-owned Deep Agent features for central reconciliation."""

    return _deep_agents_features(definition)


def langgraph_deployment_capabilities(
    *,
    profile: LangGraphDeploymentProfile | None = None,
) -> RuntimeCapabilities:
    """Return deployment declarations without definition or run policy."""

    return langgraph_capabilities(None, profile=profile)


def langgraph_capabilities(
    definition: AgentDefinition | None,
    *,
    profile: LangGraphDeploymentProfile | None = None,
) -> RuntimeCapabilities:
    profile = profile or LangGraphDeploymentProfile.from_environment()
    checkpoint = profile.checkpoint_available
    deployment_reason = "runtime_configuration_invalid" if profile.configuration_error else "runtime_unavailable"

    def enabled_descriptor(descriptor: RuntimeOperationDescriptor) -> RuntimeOperationDescriptor:
        if profile.runtime_available:
            return descriptor
        return RuntimeOperationDescriptor(
            descriptor.support,
            descriptor.owner,
            False,
            disabled_reason=deployment_reason,
            modes=descriptor.modes,
            semantics=descriptor.semantics,
            confirmation=descriptor.confirmation,
            terminal_states=descriptor.terminal_states,
            preserves_run_id=descriptor.preserves_run_id,
            preserves_session_id=descriptor.preserves_session_id,
            requires_runtime_binding=descriptor.requires_runtime_binding,
        )

    operations: dict[str, RuntimeOperationDescriptor] = {
        RuntimeOperationId.RUN_START.value: enabled_descriptor(RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True)),
        RuntimeOperationId.RUN_CANCEL.value: enabled_descriptor(RuntimeOperationDescriptor(
            RuntimeSupportLevel.NATIVE,
            RuntimeOperationOwner.RUNTIME,
            True,
            modes=("interrupt",),
            confirmation="asynchronous",
            terminal_states=("cancelled", "interrupted"),
        )),
        RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.RUNTIME,
            checkpoint,
            semantics="resume_from_interrupt",
            disabled_reason=None if checkpoint else "checkpoint_store_unavailable",
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.RUNTIME,
            checkpoint,
            semantics="checkpoint_state_inspection",
            disabled_reason=None if checkpoint else "checkpoint_store_unavailable",
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_UPDATE_STATE.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.RUNTIME,
            checkpoint,
            semantics="checkpoint_boundary_update",
            disabled_reason=None if checkpoint else "checkpoint_store_unavailable",
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.RUNTIME,
            checkpoint,
            semantics="checkpoint_thread_cleanup",
            disabled_reason=None if checkpoint else "checkpoint_store_unavailable",
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.RUN_APPROVAL_RESPOND.value: _unsupported(),
        RuntimeOperationId.RUN_STEER_LIVE.value: _unsupported(),
        RuntimeOperationId.RUN_SEND_FOLLOWUP.value: _unsupported(),
        RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT.value: _unsupported(),
        RuntimeOperationId.RUN_REPLAY.value: _unsupported(),
        RuntimeOperationId.RUN_FORK.value: _unsupported(),
        RuntimeOperationId.SUBAGENT_LIST.value: _unsupported(),
        RuntimeOperationId.SUBAGENT_SEND.value: _unsupported(),
        RuntimeOperationId.SUBAGENT_CANCEL.value: _unsupported(),
    }
    return RuntimeCapabilities(
        operations=operations,
        features=_deep_agents_features(definition) if definition is not None else {},
        deployment=profile.deployment_metadata(),
    )

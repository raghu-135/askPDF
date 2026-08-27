"""LangGraph implementation of the framework-neutral builder provider."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from app.runtime.builder import (
    BuilderCapabilities,
    BuilderCatalog,
    UnsupportedRequestOverrideError,
)
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    validated_disabled_operation_ids,
)
from app.runtime.mode import external_runtime_enabled


class LangGraphBuilderProvider:
    framework = "langgraph"
    builder_id = "langgraph_graph"

    def filter_request_overrides(
        self,
        definition: AgentDefinition,
        overrides: Mapping[str, Any] | None,
        *,
        reject_unsupported: bool,
    ) -> Mapping[str, Any]:
        from app.agent_workflows.workflow_runtime import ALLOWED_WORKFLOW_CONFIG_KEYS

        supplied = {
            str(key): value
            for key, value in dict(overrides or {}).items()
            if value is not None
        }
        unsupported = set(supplied) - ALLOWED_WORKFLOW_CONFIG_KEYS
        if unsupported and reject_unsupported:
            raise UnsupportedRequestOverrideError(unsupported)
        return {
            key: value
            for key, value in supplied.items()
            if key in ALLOWED_WORKFLOW_CONFIG_KEYS
        }

    @staticmethod
    def _external_runtime_enabled() -> bool:
        return external_runtime_enabled()

    @staticmethod
    def _normalize_external_hitl_policy(policy: Any, thread_settings: Mapping[str, Any]) -> dict[str, Any]:
        """Apply thread HITL settings without importing the LangGraph graph."""
        normalized = deepcopy(policy) if isinstance(policy, dict) else {}
        if not bool(thread_settings.get("hitl_web_approval")):
            return normalized
        normalized["enabled"] = True
        gates = dict(normalized.get("gates") or {})
        gate = dict(gates.get("web_approval_gate") or {})
        gate.setdefault("mode", "approval")
        gate.setdefault("phase", "before")
        gates["web_approval_gate"] = gate
        normalized["gates"] = gates
        return normalized

    async def capabilities(self, definition: AgentDefinition) -> BuilderCapabilities:
        return BuilderCapabilities(
            framework=self.framework,
            builder_id=self.builder_id,
            authoring=True,
            transient_tests=True,
            runtime_capabilities=dict(definition.capabilities or {}),
        )

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult:
        from app.runtime.langgraph.validator import WorkflowValidator

        report = WorkflowValidator().report(dict(spec))
        runtime = spec.get("runtime") if isinstance(spec.get("runtime"), Mapping) else {}
        features = runtime.get("features") if isinstance(runtime.get("features"), Mapping) else {}
        capability_issues: list[RuntimeValidationIssue] = []
        try:
            validated_disabled_operation_ids(features.get("disabled_operations"))
        except ValueError as exc:
            capability_issues.append(RuntimeValidationIssue(
                code="invalid_disabled_operation",
                message=str(exc),
                path="runtime.features.disabled_operations",
            ))
        issues = tuple(
            RuntimeValidationIssue(
                code=str(issue.get("code") or "invalid_workflow"),
                message=str(issue.get("message") or "Invalid workflow"),
                path=issue.get("path"),
                severity=str(issue.get("severity") or "error"),
                details=dict(issue),
            )
            for issue in report.get("issues") or []
            if isinstance(issue, Mapping)
        )
        issues = tuple(capability_issues) + issues
        # Older validators expose only an errors list. Preserve those errors
        # in the neutral result rather than silently returning valid=True.
        if not issues:
            issues = tuple(
                RuntimeValidationIssue(code="invalid_workflow", message=str(error))
                for error in report.get("errors") or []
            )
        return RuntimeValidationResult(
            valid=not issues and bool(report.get("valid", False)),
            issues=issues,
            diagnostics=dict(report),
            runtime_metadata={
                "framework": self.framework,
                "builder_id": self.builder_id,
                "definition_id": definition.definition_id,
            },
        )

    async def normalize(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        from app.runtime.langgraph.validator import WorkflowValidationError

        candidate = dict(spec)
        validation = await self.validate(definition, candidate, options=options)
        if not validation.valid:
            message = "; ".join(issue.message for issue in validation.issues)
            raise WorkflowValidationError(message or "Invalid workflow")

        if self._external_runtime_enabled():
            # Materialization is framework-owned and happens in
            # langgraph-runtime. The control plane stores the validated,
            # resolved neutral spec without importing the local compiler.
            return candidate

        from app.runtime.langgraph.compiler import WorkflowCompiler

        return WorkflowCompiler().materialize_spec(candidate)

    async def resolve(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        thread_settings: Mapping[str, Any] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        from app.runtime.langgraph.validator import WorkflowResolver

        filtered_overrides = self.filter_request_overrides(
            definition,
            request_overrides,
            reject_unsupported=False,
        )
        resolved = WorkflowResolver().resolve(
            dict(spec),
            thread_settings=thread_settings,
            request_overrides=dict(filtered_overrides),
        )
        config = dict(resolved.get("config") or {})
        if self._external_runtime_enabled():
            # The control plane only freezes neutral workflow inputs. The
            # runtime container owns LangGraph validation/materialization.
            config["hitl_policy"] = self._normalize_external_hitl_policy(
                config.get("hitl_policy"), thread_settings or {}
            )
            resolved["config"] = config
            return resolved

        from app.runtime.langgraph.graph import normalize_hitl_policy_for_thread_settings

        config["hitl_policy"] = normalize_hitl_policy_for_thread_settings(
            config.get("hitl_policy"), thread_settings or {}
        )
        resolved["config"] = config
        return await self.normalize(definition, resolved)

    async def catalog(self, definition: AgentDefinition | None = None) -> BuilderCatalog:
        from app.agent.tool_registry import tool_contracts_by_id
        from app.agent_workflows.corrective_contracts import corrective_policy_catalog
        from app.agent_workflows.node_catalog import get_node_catalog
        from app.agent_workflows.parallel_contracts import parallel_policy_catalog
        from app.agent_workflows.route_registry import get_route_function_registry

        capabilities = await self.capabilities(definition or AgentDefinition(
            definition_id="catalog",
            framework=self.framework,
            builder_id=self.builder_id,
        ))
        node_catalog = get_node_catalog()
        payload = {
            "schema_version": 1,
            "spec_schema_version": 1,
            "graph_spec": {
                "required_schema_version": 1,
                "requires_explicit_route_fn": True,
                "reserved_node_ids": ["START", "END"],
                "start_node": "START",
                "end_node": "END",
            },
            "node_catalog": {
                node_type: {
                    **metadata,
                    "authorable": metadata.get("builtin_only") is not True,
                    "allowed_parent_types": list(metadata.get("allowed_parent_types", [])),
                    "allowed_child_types": list(metadata.get("allowed_child_types", [])),
                }
                for node_type, metadata in node_catalog.items()
            },
            "route_functions": get_route_function_registry(),
            "tool_contracts": self._tool_contract_catalog(tool_contracts_by_id()),
            "defaults": {
                "context_policy": {
                    "evidence_packet_limit": 12,
                    "evidence_packet_content_limit": 2000,
                    "final_prompt_assembly": "evidence_packets",
                },
                "loop_policy": {"default_max_node_visits": 1},
                "parallel_policy": parallel_policy_catalog(),
                "corrective_policy": corrective_policy_catalog(),
            },
        }
        return BuilderCatalog(
            framework=self.framework,
            builder_id=self.builder_id,
            capabilities=capabilities,
            payload=payload,
        )

    @staticmethod
    def _tool_contract_catalog(records_by_id: Mapping[str, Any]) -> dict[str, Any]:
        contracts: dict[str, Any] = {}
        for contract_id, records in sorted(records_by_id.items()):
            records = list(records or [])
            first = records[0] if records else {}
            contracts[contract_id] = {
                "id": contract_id,
                "category": first.get("category"),
                "display_name": first.get("display_name"),
                "description": first.get("description"),
                "canonical_tools": sorted(
                    str(record.get("tool_name"))
                    for record in records
                    if isinstance(record.get("tool_name"), str) and record.get("tool_name")
                ),
                "allowed_node_types": sorted({
                    str(node_type)
                    for record in records
                    for node_type in record.get("allowed_node_types", [])
                    if node_type
                }),
                "required_node_capabilities": sorted({
                    str(capability)
                    for record in records
                    for capability in record.get("required_node_capabilities", [])
                    if capability
                }),
                "artifact_keys": sorted({
                    str(key)
                    for record in records
                    for key in record.get("artifact_keys", [])
                    if key
                }),
            }
        return contracts

    async def source(self, definition_id: str) -> Mapping[str, Any]:
        from app.agent_workflows.builtin_workflows import load_builtin_workflows

        workflow = next(
            (item for item in load_builtin_workflows() if item.get("builtin_key") == definition_id),
            None,
        )
        if workflow is None:
            raise KeyError(definition_id)
        return {
            "builtin_key": definition_id,
            "name": workflow.get("name") or definition_id,
            "description": workflow.get("description") or "",
            "spec_json": workflow["spec_json"],
        }

    async def transient_test(
        self,
        request: AgentRuntimeRequest,
        *,
        context: Any = None,
        event_sink: Any = None,
    ) -> AgentRuntimeResult:
        from app.runtime.langgraph.studio_runtime import stream_builder_test

        result: AgentRuntimeResult | None = None
        async for event in stream_builder_test(
            run=context.run,
            request=context.request,
            embedding_model=context.embedding_model,
            checkpointer=context.checkpointer,
        ):
            if event_sink is not None:
                await event_sink.emit(event)
            if event.get("event") in {"run.completed", "run.failed", "run.interrupted"}:
                from app.runtime.langgraph_adapter import _result_from_graph

                result = _result_from_graph(event.get("data") or {})
        if result is None:
            raise RuntimeError("LangGraph builder test ended without a terminal result")
        return result

    async def resume_transient_test(
        self,
        request: AgentRuntimeRequest,
        *,
        context: Any = None,
        event_sink: Any = None,
    ) -> AgentRuntimeResult:
        from app.runtime.langgraph.studio_runtime import stream_builder_test

        result: AgentRuntimeResult | None = None
        async for event in stream_builder_test(
            run=context.run,
            request=context.request,
            embedding_model=context.embedding_model,
            checkpointer=context.checkpointer,
            resume_decision=context.resume_decision,
        ):
            if event_sink is not None:
                await event_sink.emit(event)
            if event.get("event") in {"run.completed", "run.failed", "run.interrupted"}:
                from app.runtime.langgraph_adapter import _result_from_graph

                result = _result_from_graph(event.get("data") or {})
        if result is None:
            raise RuntimeError("LangGraph builder resume ended without a terminal result")
        return result

    async def cleanup_transient_test(self, request: AgentRuntimeRequest) -> Any:
        from app.runtime.langgraph.checkpointing import delete_agent_checkpoints

        checkpoint_id = request.continuation.payload.get("checkpoint_thread_id") if request.continuation else None
        return await delete_agent_checkpoints([str(checkpoint_id)]) if checkpoint_id else []

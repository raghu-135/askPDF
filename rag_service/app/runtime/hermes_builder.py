"""Framework-specific builder provider for the Hermes proof definition."""

from __future__ import annotations

from typing import Any, Mapping

from app.runtime.builder import BuilderCapabilities, BuilderCatalog
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeValidationIssue,
    RuntimeValidationResult,
)
from app.runtime.errors import RuntimeError


class HermesBuilderProvider:
    framework = "hermes"
    builder_id = "hermes_agent"

    @staticmethod
    def _issues(spec: Mapping[str, Any]) -> list[RuntimeValidationIssue]:
        issues: list[RuntimeValidationIssue] = []
        if spec.get("schema_version") != 2:
            issues.append(RuntimeValidationIssue("unsupported_schema_version", "Hermes definitions must use schema_version 2", "schema_version"))
        runtime = spec.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("kind") != "hermes_agent":
            issues.append(RuntimeValidationIssue("invalid_runtime_kind", "Hermes definitions must use runtime.kind=hermes_agent", "runtime.kind"))
        config = spec.get("config")
        if not isinstance(config, Mapping):
            issues.append(RuntimeValidationIssue("missing_config", "Hermes definitions require a config object", "config"))
        else:
            for key in ("system_prompt", "mcp_server", "allowed_tool_ids"):
                if not config.get(key):
                    issues.append(RuntimeValidationIssue("missing_config_field", f"Hermes config requires {key}", f"config.{key}"))
            if any(key in config for key in ("graph", "nodes", "edges", "route_fn")):
                issues.append(RuntimeValidationIssue("graph_fields_not_supported", "Hermes definitions cannot contain graph fields", "config"))
        return issues

    async def capabilities(self, definition: AgentDefinition) -> BuilderCapabilities:
        return BuilderCapabilities(
            framework=self.framework,
            builder_id=self.builder_id,
            schema_versions=(2,),
            authoring=False,
            transient_tests=False,
            runtime_capabilities=dict(definition.capabilities or {}),
        )

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult:
        issues = self._issues(spec)
        return RuntimeValidationResult(
            valid=not issues,
            issues=tuple(issues),
            normalized_spec=dict(spec) if not issues else None,
            runtime_metadata={"framework": self.framework, "builder_id": self.builder_id, "definition_id": definition.definition_id},
        )

    def report(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        issues = self._issues(spec)
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "valid": not issues,
            "errors": [issue.code for issue in issues],
        }

    async def normalize(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        validation = await self.validate(definition, spec, options=options)
        if not validation.valid:
            raise ValueError("; ".join(issue.message for issue in validation.issues))
        return dict(spec)

    async def resolve(self, definition: AgentDefinition, spec: Mapping[str, Any], *, thread_settings: Mapping[str, Any] | None = None, request_overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        resolved = dict(await self.normalize(definition, spec))
        config = dict(resolved.get("config") or {})
        config.update(dict(request_overrides or {}))
        resolved["config"] = config
        return resolved

    async def catalog(self, definition: AgentDefinition | None = None) -> BuilderCatalog:
        capabilities = await self.capabilities(definition or AgentDefinition("catalog", self.framework, self.builder_id))
        return BuilderCatalog(
            framework=self.framework,
            builder_id=self.builder_id,
            capabilities=capabilities,
            payload={"schema_version": 2, "definition_ids": ["hermes_rag_agent"], "graph": {"supported": False}},
        )

    async def source(self, definition_id: str) -> Mapping[str, Any]:
        if definition_id != "hermes_rag_agent":
            raise KeyError(definition_id)
        from app.agent_workflows.builtin_workflows import load_builtin_workflows
        return next(item for item in load_builtin_workflows() if item["builtin_key"] == definition_id)

    async def transient_test(self, request: AgentRuntimeRequest, *, context: Any = None, event_sink: Any = None) -> AgentRuntimeResult:
        raise RuntimeError("runtime_capability_unsupported", "Hermes builder tests are not enabled")

    async def resume_transient_test(self, request: AgentRuntimeRequest, *, context: Any = None, event_sink: Any = None) -> AgentRuntimeResult:
        raise RuntimeError("runtime_capability_unsupported", "Hermes resume is not enabled")

    async def cleanup_transient_test(self, request: AgentRuntimeRequest) -> Any:
        return {"status": "unsupported"}

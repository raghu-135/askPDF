"""Framework-specific builder provider for versioned Hermes definitions."""

from __future__ import annotations

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
)
from app.runtime.errors import RuntimeError
from app.runtime.hermes_config import hermes_model_provider
from app.runtime.hermes_profile import HERMES_DEFINITION_VERSION, HERMES_SUPPORTED_DEFINITION_VERSIONS, resolve_hermes_profile
from app.prompts.loaders import DEEP_RESEARCH_POLICY_ID, get_deep_research_policy


class HermesBuilderProvider:
    framework = "hermes"
    builder_id = "hermes_agent"
    _allowed_config_keys = {
        "research_policy_id", "system_prompt", "model", "provider", "mcp_server", "allowed_tool_ids",
        "max_output_chars", "max_duration_seconds", "max_event_count",
        "allow_subagents", "allow_persistent_memory", "cancellation_mode",
        "skills", "task_policy", "use_web_search", "context_window",
    }
    _supported_request_override_keys: frozenset[str] = frozenset({"llm_model", "context_window", "use_web_search"})

    def _issues(self, spec: Mapping[str, Any]) -> list[RuntimeValidationIssue]:
        issues: list[RuntimeValidationIssue] = []
        if spec.get("definition_version") not in HERMES_SUPPORTED_DEFINITION_VERSIONS:
            issues.append(RuntimeValidationIssue("unsupported_definition_version", f"Hermes definitions must use one of {sorted(HERMES_SUPPORTED_DEFINITION_VERSIONS)}", "definition_version"))
        if spec.get("schema_version") != 2:
            issues.append(RuntimeValidationIssue("unsupported_schema_version", "Hermes definitions must use schema_version 2", "schema_version"))
        runtime = spec.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("kind") != "hermes_agent":
            issues.append(RuntimeValidationIssue("invalid_runtime_kind", "Hermes definitions must use runtime.kind=hermes_agent", "runtime.kind"))
        config = spec.get("config")
        if not isinstance(config, Mapping):
            issues.append(RuntimeValidationIssue("missing_config", "Hermes definitions require a config object", "config"))
        else:
            for key in ("research_policy_id", "system_prompt", "mcp_server", "allowed_tool_ids"):
                if not config.get(key):
                    issues.append(RuntimeValidationIssue("missing_config_field", f"Hermes config requires {key}", f"config.{key}"))
            if config.get("research_policy_id") not in (None, DEEP_RESEARCH_POLICY_ID):
                issues.append(RuntimeValidationIssue("unsupported_research_policy", "Hermes requires the current shared Deep Research policy", "config.research_policy_id"))
            if not isinstance(config.get("allowed_tool_ids"), list) or not all(isinstance(item, str) and item for item in config.get("allowed_tool_ids") or []):
                issues.append(RuntimeValidationIssue("invalid_tool_allowlist", "Hermes allowed_tool_ids must be a non-empty string list", "config.allowed_tool_ids"))
            unknown = sorted(set(config) - self._allowed_config_keys)
            if unknown:
                issues.append(RuntimeValidationIssue("unsupported_config_field", f"Hermes config does not support: {', '.join(unknown)}", "config"))
            for key, minimum in (("max_output_chars", 1), ("max_duration_seconds", 1), ("max_event_count", 1)):
                if key in config and (not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < minimum):
                    issues.append(RuntimeValidationIssue("invalid_limit", f"{key} must be a positive integer", f"config.{key}"))
            if any(key in config for key in ("graph", "nodes", "edges", "route_fn")) or any(key in spec for key in ("graph", "nodes", "edges", "route_fn")):
                issues.append(RuntimeValidationIssue("graph_fields_not_supported", "Hermes definitions cannot contain graph fields", "config"))
            try:
                resolve_hermes_profile(spec)
            except ValueError as exc:
                issues.append(RuntimeValidationIssue("invalid_hermes_profile", str(exc), "config"))
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

    def filter_request_overrides(
        self,
        definition: AgentDefinition,
        overrides: Mapping[str, Any] | None,
        *,
        reject_unsupported: bool,
    ) -> Mapping[str, Any]:
        supplied = {
            str(key): value
            for key, value in dict(overrides or {}).items()
            if value is not None
        }
        unsupported = set(supplied) - self._supported_request_override_keys
        if unsupported and reject_unsupported:
            raise UnsupportedRequestOverrideError(unsupported)
        return {
            key: value
            for key, value in supplied.items()
            if key in self._supported_request_override_keys
        }

    async def resolve(self, definition: AgentDefinition, spec: Mapping[str, Any], *, thread_settings: Mapping[str, Any] | None = None, request_overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        resolved = dict(await self.normalize(definition, spec))
        config = dict(resolved.get("config") or {})
        filtered_overrides = dict(self.filter_request_overrides(
            definition,
            request_overrides,
            reject_unsupported=False,
        ))
        selected_model = str(
            filtered_overrides.pop("llm_model", None)
            or (thread_settings or {}).get("llm_model")
            or ""
        ).strip()
        if selected_model:
            config["model"] = selected_model
            config["provider"] = hermes_model_provider()
        config.update(filtered_overrides)
        policy_id = str(config.get("research_policy_id") or "")
        runtime_instructions = str(config.get("system_prompt") or "").strip()
        config["system_prompt"] = get_deep_research_policy(policy_id) + "\n\n" + runtime_instructions
        resolved["config"] = config
        resolved["managed_profile"] = resolve_hermes_profile(resolved)
        return await self.normalize(definition, resolved)

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

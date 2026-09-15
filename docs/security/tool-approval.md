# Human-gated tools

Tool handlers do not implement approval. The MCP server wraps registered
handlers with policy enforcement and an invocation journal. Product adapters
compile backend settings into a signed execution context; the model cannot
change that policy through tool arguments.

## Policy

`hitl_policy` has two independent surfaces:

- `tools`: per-MCP-tool wrapping (`allow`, `ask`, `deny`) with `run` or `task`
  scope. This map is compiled whenever it is present. Graph-gate
  `hitl_policy.enabled` does not discard it.
- `gates` plus `enabled`: workflow interrupts (pause, review, choice). `enabled`
  applies only to those gates.

    {
      "hitl_policy": {
        "enabled": false,
        "tools": {
          "search_knowledge": {"mode": "ask", "scope": "run"}
        }
      }
    }

Unknown tools and invalid policies are rejected. `scope: task` is rejected
unless an AgentTask id is present. An explicit denial takes precedence over an
allowance.

Web search remains a category shortcut. `web_search_mode` (`off` / `ask` /
`on`) is the product field. `hitl_web_approval` and `use_web_search` are
derived only when `web_search_mode` is absent. `off` is an admission deny for
every `CAT_WEB` and `CAT_EXTERNAL_RESEARCH` tool, even if the tools map asked
to gate them. That deny is not a human skip.

Thread setting `hitl_canvas_publish` (default false) is the product shortcut
for `publish_canvas`. When true, the compiler adds `{ "mode": "ask", "scope":
"run" }` unless the tools map already names that tool. Canvas publish is not a
web-search category tool.

## Approval lifecycle

An ask call returns the exact tool name, arguments, argument digest, and
invocation identity before the handler runs. The product saves the decision;
the selected runtime then resumes through its native pause mechanism and the
same MCP boundary.

LangGraph tool gates continue with `run.resume`. Hermes tool gates continue
with `run.approval.respond`. Native resume values alone are not authorization.
If submitting that continuation fails, the pending tool-approval interrupt is
restored so the human can retry.

- Approve once authorizes only that invocation and argument digest.
- Approve for this run/task authorizes that tool within the displayed scope.
- Skip this tool for this run/task denies it within that scope.

MCP execution grants are re-issued on resume so a long human wait cannot rely
on the start-time token TTL. Invocation identities are deterministic so a
crash between ask and retry cannot silently mint a new fence.

## Replay and failure handling

Completed invocation results are replayed from the journal for every MCP call
tied to a persisted AgentRun, including calls without a tool approval rule.
Ephemeral curator correlation ids are not fenced. Concurrent duplicates,
cancellation, and process death must not automatically re-execute an
invocation with an unknown outcome. The product returns a non-retryable error
until the outcome is inspected.

Approval decisions and accepted interrupt responses are committed together.
Audit events are history, never an authorization source.

## Adding a tool

Register the handler and contract in the backend MCP catalog, configure its
approval policy, and add contract/approval/replay tests. Runtime-specific
frontends or planner heuristics should not be needed. Graph-internal tools
(`mcp_enabled: false`) cannot use this wrapper.

The relevant implementation is in rag_service/app/mcp/ and rag_service/app/tools/.

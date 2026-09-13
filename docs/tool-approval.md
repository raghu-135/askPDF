# Human-gated tools

Tool handlers do not implement approval. The MCP server wraps every registered
handler with policy enforcement and an invocation journal. Product adapters
compile backend settings into a signed execution context; the model cannot
change that policy through tool arguments.

For example, a workflow can require approval for document retrieval:

```json
{
  "hitl_policy": {
    "enabled": true,
    "tools": {
      "search_documents": {"mode": "ask", "scope": "run"}
    }
  }
}
```

Modes are `allow`, `ask`, and `deny`. Reusable scopes are `run` and `task`.
Unknown tool names and invalid policies are rejected. The web-search settings
compile policies for the web and external-research tools from the backend tool
catalog. An explicit web denial takes precedence over an allowance.

An `ask` call returns an approval request containing its exact tool name,
arguments, argument digest, and invocation identity before the handler runs.
LangGraph pauses with a native interrupt outside the MCP transport's task
groups. Hermes uses its native tool approval API. Both return to the same MCP
boundary after the product saves the human decision; a native resume value
alone cannot authorize execution.

The pinned Hermes Runs API originally discarded its event queue when an SSE
subscriber disconnected. Its small transport patch now retains frames with
stable IDs across human pauses and retires MCP connections after terminal
delivery. Live approvals survive the native transport's idle sweep; terminal
buffers still use the native retention limit. This adds replayable delivery,
not another agent scheduler.

- **Approve once** authorizes that invocation and argument digest only.
- **Approve for this run/task** authorizes that tool within the displayed scope.
- **Skip this tool for this run/task** denies the tool within that scope, including
  subsequent calls created by replanning. It does not authorize other tools.

Completed invocation results are replayed from the journal. Concurrent duplicate
calls, cancellation, and process death cannot cause automatic re-execution of a
call with an unknown outcome. Such a call returns a non-retryable error so its
outcome can be inspected before a new invocation is attempted. LangGraph also
checkpoints tool results and deep-agent action selection, resumes individual
parallel interrupts by ID, and excludes human waiting time from dispatch deadlines.

To add another tool, register its handler and contract in the existing backend
MCP catalog and configure its approval policy. No runtime-specific gate node,
planner heuristic, or tool-specific frontend component is required. A new
runtime needs an adapter from the common approval request to its native pause
mechanism and must reuse the invocation identity when continuing the call.

The startup migration creates `tool_approval_decisions` and `tool_invocations`.
Approval decisions and accepted interrupt responses are committed together.
The run journal records permission waits and denials without claiming that the
handler executed. Generic graph review and choice gates continue to support
workflow-level review independently of tool permission.

Task permissions use the same decision store as normal chat. The consolidation
migration imports each task’s latest historical web decision for the external
tools it covered. Audit events remain history, never an authorization source.
Workflow review gates use their authored targets and do not grant tool access.

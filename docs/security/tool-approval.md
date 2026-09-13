# Human-gated tools

Tool handlers do not implement approval. The MCP server wraps registered
handlers with policy enforcement and an invocation journal. Product adapters
compile backend settings into a signed execution context; the model cannot
change that policy through tool arguments.

## Policy

Modes are allow, ask, and deny. Scopes are run and task.

    {
      "hitl_policy": {
        "enabled": true,
        "tools": {
          "search_documents": {"mode": "ask", "scope": "run"}
        }
      }
    }

Unknown tools and invalid policies are rejected. An explicit denial takes
precedence over an allowance.

## Approval lifecycle

An ask call returns the exact tool name, arguments, argument digest, and
invocation identity before the handler runs. The product saves the decision;
the selected runtime then resumes through its native pause mechanism and the
same MCP boundary.

- Approve once authorizes only that invocation and argument digest.
- Approve for this run/task authorizes that tool within the displayed scope.
- Skip this tool for this run/task denies it within that scope.

Native resume values alone are not authorization.

## Replay and failure handling

Completed invocation results are replayed from the journal. Concurrent
duplicates, cancellation, and process death must not automatically re-execute
an invocation with an unknown outcome. The product returns a non-retryable
error until the outcome is inspected.

Approval decisions and accepted interrupt responses are committed together.
Audit events are history, never an authorization source.

## Adding a tool

Register the handler and contract in the backend MCP catalog, configure its
approval policy, and add contract/approval/replay tests. Runtime-specific
frontends or planner heuristics should not be needed.

The relevant implementation is in rag_service/app/mcp/ and rag_service/app/tools/.

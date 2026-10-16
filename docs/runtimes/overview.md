# Agent runtime overview

askPDF has a product control plane and pluggable external execution runtimes:

    frontend → product APIs → runtime_protocol HTTP/SSE → LangGraph runtime
                                      └───────────────→ Hermes runtime

The control plane owns workflow definitions, product policy, authentication,
tasks, plans, todos, subagents, budgets, artifacts (including research
canvases), and trace projections.

Runtimes own framework validation, compilation, execution, checkpoints,
runtime leases, execution journals, dependency monitoring, and opaque
continuations.

The shared runtime_protocol package is dependency-neutral and independently
installed by each runtime image. It contains JSON-serializable contracts,
events, errors, authentication, tool approval, and continuation types.

LangGraph is the default execution engine for the main agent workflows.
Hermes is an additional engine for supported deep-research task definitions.
`.env.example` enables the Compose `hermes` profile. Their capability
surfaces are intentionally different; callers must use capability admission
rather than assuming every runtime supports every operation.

See:

- [LangGraph runtime](langgraph.md)
- [Hermes runtime](hermes.md)
- [Runtime operations](operations.md)
- [Capability matrix](capability-matrix.md)
- [Runtime protocol](../contracts/runtime-protocol.md)

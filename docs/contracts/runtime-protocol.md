# Runtime protocol

runtime_protocol is the only shared Python package between the control plane
and external runtimes. It is dependency-neutral and independently installed by
each service image.

It defines:

- runtime requests and results
- event envelopes and terminal semantics
- typed runtime errors
- authentication parsing
- shared LLM server URL and optional API key
- tool contracts and approval context
- continuation bindings
- task results and evidence
- transport and validation helpers

The control plane uses HTTP/SSE adapters to communicate with LangGraph and
Hermes. Framework-specific identifiers remain inside the owning runtime.
Product records store only opaque runtime bindings.

The protocol is strictly validated and JSON-oriented. It currently has no
protocol-version negotiation, so changes require coordinated updates to the
control plane, runtime adapters, and runtime services.

Related source:

- runtime_protocol/
- rag_service/app/runtime/
- langgraph_runtime/api.py
- hermes_runtime/api.py

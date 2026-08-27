"""LangGraph-owned implementation surface.

Keep this package initializer framework-free. The control-plane image does
not install LangGraph; importing a lightweight module such as
``app.runtime.langgraph.validator`` must therefore not eagerly import the
compiler or execution graph. Public implementation symbols remain available
through lazy attribute loading for runtime-container callers.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "NodeRegistry": ("app.runtime.langgraph.graph", "NodeRegistry"),
    "WorkflowCompiler": ("app.runtime.langgraph.compiler", "WorkflowCompiler"),
    "continue_compiled_rag_chat": ("app.runtime.langgraph.router_runtime", "continue_compiled_rag_chat"),
    "execute_compiled_rag_chat": ("app.runtime.langgraph.router_runtime", "execute_compiled_rag_chat"),
    "resume_compiled_rag_chat": ("app.runtime.langgraph.router_runtime", "resume_compiled_rag_chat"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "NodeRegistry",
    "WorkflowCompiler",
    "continue_compiled_rag_chat",
    "execute_compiled_rag_chat",
    "resume_compiled_rag_chat",
]

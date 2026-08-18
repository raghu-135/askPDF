"""Compatibility entry point for LangGraph execution and projection."""

from app.agent_workflows import router_runtime as _legacy
from app.agent_workflows.router_runtime import *  # noqa: F401,F403


async def execute_compiled_rag_chat(*args, **kwargs):
    return await _legacy.execute_compiled_rag_chat(*args, **kwargs)


async def resume_compiled_rag_chat(*args, **kwargs):
    return await _legacy.resume_compiled_rag_chat(*args, **kwargs)


async def continue_compiled_rag_chat(*args, **kwargs):
    return await _legacy.continue_compiled_rag_chat(*args, **kwargs)


async def project_agent_task_result(*args, **kwargs):
    return await _legacy.project_agent_task_result(*args, **kwargs)

__all__ = [
    "continue_compiled_rag_chat",
    "execute_compiled_rag_chat",
    "project_agent_task_result",
    "resume_compiled_rag_chat",
]

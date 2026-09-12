"""
Prompt file loaders - Simple utilities to load prompt markdown files.

Prompt composition lives in langgraph_runtime/workflows/prompting.py. This module just
handles file I/O and narrow shared prompt fragments.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
RUNTIME_PROMPT_NAMESPACE = "agent_workflows"
DEEP_RESEARCH_POLICY_ID = "deep_research_v1"
RUNTIME_PROMPT_FILES = (
    "corrective_grounded_verifier.md",
    "corrective_retrieval_grader.md",
    "deep_research_policy.md",
    "evaluator_replanner_evaluator.md",
    "evaluator_replanner_replanner.md",
    "final_answer.md",
    "plan_execute_planner.md",
    "router_rag_router.md",
    "web_search_mandate.md",
)


def runtime_prompt_path(filename: str) -> Path:
    """Return a path for a prompt owned by the LangGraph runtime."""
    return PROMPTS_DIR / RUNTIME_PROMPT_NAMESPACE / filename


def validate_runtime_prompt_assets() -> None:
    """Fail startup if a runtime prompt asset is missing from the image."""
    missing = [
        filename
        for filename in RUNTIME_PROMPT_FILES
        if not runtime_prompt_path(filename).is_file()
    ]
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"LangGraph runtime prompt assets are missing: {names}")


def load_runtime_prompt(filename: str) -> str:
    """Load a prompt markdown file from the runtime-owned prompt namespace."""
    path = runtime_prompt_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def get_web_search_mandate() -> str:
    """Load the web search mandate section."""
    return load_runtime_prompt("web_search_mandate.md")


def get_deep_research_policy(policy_id: str = DEEP_RESEARCH_POLICY_ID) -> str:
    """Load the versioned policy shared by Deep Research runtimes."""
    if policy_id != DEEP_RESEARCH_POLICY_ID:
        raise ValueError(f"Unsupported Deep Research policy: {policy_id}")
    return load_runtime_prompt("deep_research_policy.md").strip()

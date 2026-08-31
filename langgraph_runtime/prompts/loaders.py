"""
Prompt file loaders - Simple utilities to load prompt markdown files.

Prompt composition lives in agent_workflows/prompting.py. This module just
handles file I/O and narrow shared prompt fragments.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
DEEP_RESEARCH_POLICY_ID = "deep_research_v1"


def load_prompt(filename: str) -> str:
    """Load a prompt markdown file from the prompts directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def get_web_search_mandate() -> str:
    """Load the web search mandate section."""
    return load_prompt("agent_workflows/web_search_mandate.md")


def get_deep_research_policy(policy_id: str = DEEP_RESEARCH_POLICY_ID) -> str:
    """Load the versioned policy shared by Deep Research runtimes."""
    if policy_id != DEEP_RESEARCH_POLICY_ID:
        raise ValueError(f"Unsupported Deep Research policy: {policy_id}")
    return load_prompt("agent_workflows/deep_research_policy.md").strip()

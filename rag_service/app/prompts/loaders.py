"""
Prompt file loaders - Simple utilities to load prompt markdown files.

Prompt composition lives in agent_workflows/prompting.py. This module just
handles file I/O and narrow shared prompt fragments.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> str:
    """Load a prompt markdown file from the prompts directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def get_web_search_mandate() -> str:
    """Load the web search mandate section."""
    return load_prompt("agent_workflows/web_search_mandate.md")

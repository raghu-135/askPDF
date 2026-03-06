"""
Prompt file loaders - Simple utilities to load prompt markdown files.

All composition logic lives in agent.py. This module just handles file I/O.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt markdown file from the prompts directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def get_orchestrator_prompt() -> str:
    """Load the orchestrator agent system prompt."""
    return load_prompt("orchestrator/system.md")

def get_orchestrator_prompt_compact() -> str:
    """Load the compact orchestrator agent system prompt."""
    return load_prompt("orchestrator/system_compact.md")

def get_orchestrator_phase0_prompt() -> str:
    """Load the orchestrator Phase 0 preprocessing prompt."""
    return load_prompt("orchestrator/phase0.md")

def get_orchestrator_phase0_prompt_compact() -> str:
    """Load the compact orchestrator Phase 0 preprocessing prompt."""
    return load_prompt("orchestrator/phase0_compact.md")


def get_intent_agent_prompt() -> str:
    """Load the intent agent system prompt."""
    return load_prompt("intent_agent/system.md")

def get_intent_agent_prompt_compact() -> str:
    """Load the compact intent agent system prompt."""
    return load_prompt("intent_agent/system_compact.md")


def get_web_search_mandate() -> str:
    """Load the web search mandate section."""
    return load_prompt("orchestrator/web_search_mandate.md")

"""
prompt_defaults.py - Default values for customizable prompt sections

This module defines all user-customizable defaults for the Orchestrator and Intent Agent prompts.
The database can override these on a per-thread basis by storing only the customized values.
"""

# Default system role for the Orchestrator Agent
DEFAULT_SYSTEM_ROLE = "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."

# Default custom instructions (empty by default — user can add)
DEFAULT_CUSTOM_INSTRUCTIONS = ""

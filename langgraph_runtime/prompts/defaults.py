"""
prompt_defaults.py - Default values for customizable prompt sections

This module defines user-customizable defaults for agent-workflow prompts.
The database can override these on a per-thread basis by storing only the customized values.
"""

# Default system role for agent-workflow prompts.
DEFAULT_SYSTEM_ROLE = "Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers."

# Default custom instructions (empty by default — user can add)
DEFAULT_CUSTOM_INSTRUCTIONS = ""

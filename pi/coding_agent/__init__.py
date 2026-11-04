"""pi.coding_agent: coding agent on top of pi.agent.

Python port of the core of ``packages/coding-agent``.
"""

from .session import SessionManager
from .system_prompt import SystemPromptOptions, build_system_prompt
from .tools import create_default_tools, tool_prompt_contributions

__all__ = [
    "SessionManager",
    "SystemPromptOptions",
    "build_system_prompt",
    "create_default_tools",
    "tool_prompt_contributions",
    "cli",
]

"""Built-in coding tools: read, write, edit, bash.

Python port of ``packages/coding-agent/src/core/tools``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ...agent.types import AgentTool
from .bash import BASH_PROMPT_SNIPPET, create_bash_tool
from .edit import EDIT_PROMPT_GUIDELINES, EDIT_PROMPT_SNIPPET, create_edit_tool
from .read import READ_PROMPT_GUIDELINES, READ_PROMPT_SNIPPET, create_read_tool
from .write import WRITE_PROMPT_GUIDELINES, WRITE_PROMPT_SNIPPET, create_write_tool
from .grep import GREP_PROMPT_SNIPPET, create_grep_tool
from .find import FIND_PROMPT_SNIPPET, create_find_tool
from .ls import LS_PROMPT_SNIPPET, create_ls_tool
from .powershell import POWERSHELL_PROMPT_SNIPPET, create_powershell_tool

__all__ = [
    "create_default_tools",
    "create_read_tool",
    "create_bash_tool",
    "create_edit_tool",
    "create_write_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
    "create_powershell_tool",
    "tool_prompt_contributions",
    "READ_PROMPT_SNIPPET",
    "BASH_PROMPT_SNIPPET",
    "EDIT_PROMPT_SNIPPET",
    "WRITE_PROMPT_SNIPPET",
]


def create_default_tools(cwd: str) -> List[AgentTool]:
    """Create built-in tools bound to ``cwd``."""
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
        create_powershell_tool(cwd),
    ]


def tool_prompt_contributions(tools: List[AgentTool]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Snippet and guideline contributions for the system prompt, by tool name."""
    snippets = {
        "read": READ_PROMPT_SNIPPET,
        "bash": BASH_PROMPT_SNIPPET,
        "edit": EDIT_PROMPT_SNIPPET,
        "write": WRITE_PROMPT_SNIPPET,
        "grep": GREP_PROMPT_SNIPPET,
        "find": FIND_PROMPT_SNIPPET,
        "ls": LS_PROMPT_SNIPPET,
        "powershell": POWERSHELL_PROMPT_SNIPPET,
    }
    guidelines: Dict[str, List[str]] = {
        "read": list(READ_PROMPT_GUIDELINES),
        "edit": list(EDIT_PROMPT_GUIDELINES),
        "write": list(WRITE_PROMPT_GUIDELINES),
    }
    selected = {t.name for t in tools}
    return (
        {name: text for name, text in snippets.items() if name in selected},
        {name: rules for name, rules in guidelines.items() if name in selected},
    )

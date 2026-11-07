"""Explicit PowerShell execution through the shared shell runner."""

from ...agent.types import AgentTool
from .bash import create_bash_tool, powershell_command

POWERSHELL_PROMPT_SNIPPET = "Execute PowerShell commands with streaming output and optional timeout"


def create_powershell_tool(cwd: str) -> AgentTool:
    tool = create_bash_tool(cwd, shell_command=powershell_command)
    tool.name = "powershell"
    tool.label = "powershell"
    tool.description = "Execute PowerShell commands in the working directory. Supports timeout in seconds; stdout/stderr stream and use the shared output limits."
    return tool

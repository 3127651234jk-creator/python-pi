"""Explicit PowerShell execution through the shared shell runner."""

import base64
import shutil

from ...agent.types import AgentTool
from .bash import create_bash_tool

POWERSHELL_PROMPT_SNIPPET = "Execute PowerShell commands with streaming output and optional timeout"


def create_powershell_tool(cwd: str) -> AgentTool:
    def command_argv(command: str):
        executable = shutil.which("pwsh") or shutil.which("powershell")
        if executable is None:
            raise FileNotFoundError("PowerShell is unavailable; install pwsh and add it to PATH")
        prefix = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n$OutputEncoding = [Console]::OutputEncoding\n"
        encoded = base64.b64encode((prefix + command).encode("utf-16-le")).decode("ascii")
        return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded], False

    tool = create_bash_tool(cwd, shell_command=command_argv)
    tool.name = "powershell"
    tool.label = "powershell"
    tool.description = "Execute PowerShell commands in the working directory. Supports timeout in seconds; stdout/stderr stream and use the shared output limits."
    return tool

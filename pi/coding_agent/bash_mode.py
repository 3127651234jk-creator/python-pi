"""User-facing shell escape: the ``!`` / ``!!`` input mode (F-TUI-04).

``!`` runs one shell command through the shared bash runner (streaming,
truncation, ANSI stripping) outside the model; its output enters the session
and the model context as a ``bashExecution`` message. ``!!`` keeps the output
out of the model context. Port of the TS interactive-mode bash path and the
RPC ``bash`` command's execution half.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..agent.types import BashExecutionMessage
from ..util import AbortSignal
from .tools.bash import create_bash_tool

_TIMEOUT_STATUS = re.compile(r"Command timed out after \d+(?:\.\d+)? seconds$")
_ABORT_STATUS = "Command aborted"


def _split_status(text: str) -> "tuple[str, str]":
    """Split a bash-tool failure message into (output, status)."""
    if text.endswith(_ABORT_STATUS):
        return text[:-len(_ABORT_STATUS)].rstrip("\n"), "aborted"
    match = _TIMEOUT_STATUS.search(text)
    if match:
        return text[:match.start()].rstrip("\n"), "timeout"
    return text, ""


async def execute_bang(text: str, cwd: str, signal: Optional[AbortSignal] = None,
                       exclude_from_context: Optional[bool] = None) -> BashExecutionMessage:
    """Run one ``!`` / ``!!`` command; raises ValueError on an empty command.

    Leading whitespace is accepted. ``exclude_from_context`` forces the ``!!``
    semantics explicitly; the RPC ``bash`` command uses ``execute_shell``
    because its payload is a raw command without a bang prefix."""
    stripped = text.lstrip()
    excluded = stripped.startswith("!!") if exclude_from_context is None else exclude_from_context
    command = stripped[2:].strip() if stripped.startswith("!!") else stripped[1:].strip()
    return await _execute(command, cwd, signal, excluded, bang=True)


async def execute_shell(command: str, cwd: str, signal: Optional[AbortSignal] = None,
                        exclude_from_context: bool = False) -> BashExecutionMessage:
    """Run a raw shell command (RPC ``bash``; no ``!``/``!!`` prefix)."""

    return await _execute(command.strip(), cwd, signal, exclude_from_context, bang=False)


async def _execute(command: str, cwd: str, signal: Optional[AbortSignal], excluded: bool,
                   bang: bool) -> BashExecutionMessage:
    if not command:
        if bang:
            raise ValueError("usage: ! <command>  (or !! <command> to keep the output out of context)")
        raise ValueError("bash requires a command string")
    tool = create_bash_tool(cwd)
    timestamp = int(time.time() * 1000)
    try:
        result = await tool.execute(f"bang-{timestamp}", {"command": command}, signal, None)
    except RuntimeError as exc:
        output, status = _split_status(str(exc))
        return BashExecutionMessage(command=command, output=output, exit_code=None,
                                    cancelled=status == "aborted", truncated=False, timestamp=timestamp,
                                    exclude_from_context=excluded)
    structured = result.structured_content or {}
    return BashExecutionMessage(
        command=command, output=structured.get("output", ""),
        exit_code=structured.get("exit_code"), cancelled=False,
        truncated=bool(structured.get("truncated")),
        full_output_path=structured.get("full_output_path"),
        timestamp=timestamp, exclude_from_context=excluded)


def render_bang_output(message: BashExecutionMessage) -> str:
    """Interactive rendering: dim-marked header for ``!!``, then the output
    (the #10557 header-dim semantics)."""
    marker = "!!" if message.exclude_from_context else "!"
    lines = [f"{marker} {message.command}"]
    if message.output:
        lines.append(message.output)
    if message.cancelled:
        lines.append("(command cancelled)")
    elif message.exit_code not in (None, 0):
        lines.append(f"Command exited with code {message.exit_code}")
    return "\n".join(lines)

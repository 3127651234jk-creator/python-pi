"""bash tool: execute shell commands with streaming, truncated output.

Python port of ``packages/coding-agent/src/core/tools/bash.ts``. On Windows
commands run through Git Bash when it is installed (the shell the TS client
prefers), otherwise through PowerShell; elsewhere through bash. Output
accumulates streaming and is tail-truncated to the shared line/byte limits;
full output is saved to a temp file when truncation occurs.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import signal
import sys
import tempfile
import time
from typing import Any, Awaitable, Callable, Optional, Tuple

from ...ai.types import TextContent
from ...agent.types import AgentTool, AgentToolResult
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_tail,
)

MAX_TIMEOUT_SECONDS = 2147483
STRUCTURED_OUTPUT_MAX_BYTES = 1024 * 1024
UPDATE_THROTTLE_MS = 300

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

_WINDOWS_BASH: Optional[str] = None


def _find_windows_bash() -> Optional[str]:
    """Locate Git Bash: the shell the TS client prefers on Windows.

    Checks the standard Git install locations, the install next to ``git`` on
    PATH (covers non-default installs), and ``bash.exe`` itself.
    """
    candidates = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
        base = os.environ.get(variable)
        if base:
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
    git = shutil.which("git")
    if git:
        install = os.path.dirname(os.path.dirname(git))
        candidates.append(os.path.join(install, "bin", "bash.exe"))
        candidates.append(os.path.join(os.path.dirname(git), "bash.exe"))
    found = shutil.which("bash.exe")
    if found:
        candidates.append(found)
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


if sys.platform == "win32":
    _WINDOWS_BASH = _find_windows_bash()

_SHELL_LABEL = "PowerShell" if sys.platform == "win32" and _WINDOWS_BASH is None else "bash"
_SHELL_EXAMPLES = ("Get-ChildItem, Select-String, etc." if _SHELL_LABEL == "PowerShell"
                   else "ls, grep, find, etc.")
BASH_DESCRIPTION = (
    f"Execute a shell command in the current working directory (this machine runs {_SHELL_LABEL}). "
    "Returns stdout and stderr. Output is "
    f"truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit "
    "first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds."
)

BASH_PROMPT_SNIPPET = f"Execute {_SHELL_LABEL} commands ({_SHELL_EXAMPLES})"


def powershell_command(command: str) -> Tuple[list, bool]:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        raise FileNotFoundError("PowerShell is unavailable; install pwsh and add it to PATH")
    prefix = (
        "$ProgressPreference = 'SilentlyContinue'\n"
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "$OutputEncoding = [Console]::OutputEncoding\n"
    )
    encoded = base64.b64encode((prefix + command).encode("utf-16-le")).decode("ascii")
    return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-OutputFormat", "Text",
            "-EncodedCommand", encoded], False


class OutputAccumulator:
    """Collect streamed output; tail-truncate snapshots; persist overflow."""

    def __init__(self) -> None:
        self._chunks: list = []
        self._bytes = 0
        self._full_output_path: Optional[str] = None
        self._persisted_bytes = 0

    def append(self, data: bytes) -> None:
        self._chunks.append(data)
        self._bytes += len(data)
        if self._full_output_path is not None:
            with open(self._full_output_path, "ab") as handle:
                handle.write(data)
            self._persisted_bytes += len(data)

    def snapshot(self) -> Tuple[str, TruncationResult, Optional[str]]:
        content = b"".join(self._chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(content)
        full_output_path = self._full_output_path
        if truncation.truncated and full_output_path is None:
            handle = tempfile.NamedTemporaryFile(prefix="pi-py-bash-", suffix=".log", delete=False)
            handle.write(b"".join(self._chunks))
            handle.close()
            full_output_path = handle.name
            self._full_output_path = full_output_path
            self._persisted_bytes = self._bytes
        return content, truncation, full_output_path

    def close(self) -> None:
        pass


def _shell_command(command: str) -> Tuple[list, bool]:
    """Return (argv, needs_shell_wrap).

    Windows prefers Git Bash (POSIX commands work) and falls back to
    PowerShell; other platforms use bash.
    """
    if sys.platform == "win32":
        bash = _WINDOWS_BASH or _find_windows_bash()
        if bash is not None:
            return [bash, "-c", command], False
        return powershell_command(command)
    return ["bash", "-c", command], False


def _kill_tree(pid: int) -> None:
    try:
        if sys.platform == "win32":
            os.system(f"taskkill /F /T /PID {pid} >nul 2>&1")
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def create_bash_tool(
    cwd: str, command_prefix: Optional[str] = None, *,
    shell_command: Optional[Callable[[str], Tuple[list, bool]]] = None,
) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: Any,
        signal: Optional[Any] = None,
        on_update: Optional[Callable[[AgentToolResult], Awaitable[None]]] = None,
    ) -> AgentToolResult:
        command: str = params["command"]
        timeout: Optional[float] = params.get("timeout")
        if signal is not None and getattr(signal, "aborted", False):
            raise RuntimeError("Command aborted")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
                raise ValueError("Invalid timeout: must be a positive number of seconds")
        if command_prefix:
            command = f"{command_prefix}\n{command}"

        if not os.path.isdir(cwd):
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")

        output = OutputAccumulator()
        last_update_at = 0.0
        update_dirty = False

        def handle_data(data: bytes) -> None:
            nonlocal update_dirty
            output.append(data)
            update_dirty = True

        async def maybe_emit_update() -> None:
            nonlocal update_dirty, last_update_at
            if on_update is None or not update_dirty:
                return
            now = time.time()
            if now - last_update_at < UPDATE_THROTTLE_MS / 1000:
                return
            update_dirty = False
            last_update_at = now
            content, truncation, full_path = output.snapshot()
            await on_update(
                AgentToolResult(
                    content=[TextContent(text=truncation.content)],
                    details={
                        "truncation": truncation if truncation.truncated else None,
                        "fullOutputPath": full_path,
                    },
                )
            )

        argv, _ = (shell_command or _shell_command)(command)
        started_at = time.monotonic()
        timed_out = False
        aborted = False

        if sys.platform != "win32":
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        assert proc.stdout is not None

        async def read_stream() -> None:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    return
                handle_data(chunk)

        read_task = asyncio.ensure_future(read_stream())
        try:
            await asyncio.sleep(0)
            while True:
                if signal is not None and getattr(signal, "aborted", False):
                    aborted = True
                    break
                if proc.returncode is not None:
                    break
                if timeout is not None and (time.monotonic() - started_at) > timeout:
                    timed_out = True
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=0.1)
                except asyncio.TimeoutError:
                    pass
                await maybe_emit_update()
        finally:
            if aborted or timed_out or proc.returncode is None:
                _kill_tree(proc.pid)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
            await read_task
            output.close()

        content, truncation, full_path = output.snapshot()
        text = truncation.content or "(no output)"
        details: Optional[dict] = None
        if truncation.truncated:
            details = {"truncation": truncation, "fullOutputPath": full_path}
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines
            if truncation.last_line_partial:
                text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line}. "
                    f"Full output: {full_path}]"
                )
            elif truncation.truncated_by == "lines":
                text += f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. Full output: {full_path}]"
            else:
                text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {full_path}]"
                )

        def append_status(t: str, status: str) -> str:
            return f"{t}\n\n{status}" if t else status

        if aborted:
            raise RuntimeError(append_status(content, "Command aborted"))
        if timed_out:
            raise RuntimeError(append_status(content, f"Command timed out after {timeout} seconds"))

        exit_code = proc.returncode
        wall_time_seconds = round(time.monotonic() - started_at, 1)
        if exit_code is None:
            raise RuntimeError(append_status(text, "Command terminated without an exit code"))

        structured_content = {
            "output": content[:STRUCTURED_OUTPUT_MAX_BYTES],
            "truncated": truncation.truncated,
            "exit_code": exit_code,
            "wall_time_seconds": wall_time_seconds,
        }
        if truncation.truncated and full_path:
            structured_content["full_output_path"] = full_path

        if exit_code != 0:
            return AgentToolResult(
                content=[TextContent(text=append_status(text, f"Command exited with code {exit_code}"))],
                details=details,
                structured_content=structured_content,
                is_error=True,
            )
        return AgentToolResult(content=[TextContent(text=text)], details=details, structured_content=structured_content)

    return AgentTool(
        name="bash",
        label="bash",
        description=BASH_DESCRIPTION,
        parameters=BASH_SCHEMA,
        execute=execute,
    )

"""bash tool (port of ``tools/bash.ts``).

Runs a command through the environment's shell. Its output streams to
``api.output()``, where the caller keeps the tail within its limits; the result
content is that retained output. A nonzero exit or timeout raises, which makes
an error result that still carries the output and diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

from ..env import Context, ShellExecOptions, ShellSpillOptions, ShellOutputWindow
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES
from .base import ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration, define_tool, require_env

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

BASH_DESCRIPTION = (
    "Execute a bash command in the current working directory. Returns combined stdout and stderr. "
    f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
    "(whichever is hit first). If truncated, full output is saved to a temp file. "
    "Optionally provide a timeout in seconds."
)

DEFAULT_OUTPUT_LIMITS = {"maxBytes": DEFAULT_MAX_BYTES, "maxLines": DEFAULT_MAX_LINES, "retain": "tail"}


def _validate_timeout(timeout: Optional[float]) -> None:
    if timeout is None:
        return
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(
    options: Optional[Dict[str, Any]] = None,
    command_prefix: Optional[str] = None,
    prepare: Optional[Callable[[dict, ToolExecutionApi, Context], Optional[Any]]] = None,
) -> ToolRegistration:
    """``options`` carries the TS ``BashToolOptions`` shape: ``{"commandPrefix", "prepare"}``."""
    if options is not None:
        command_prefix = options.get("commandPrefix", command_prefix)
        prepare = options.get("prepare", prepare)

    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        command: str = args["command"]
        timeout = args.get("timeout")
        _validate_timeout(timeout)
        env = require_env(api)
        execution = {
            "command": f"{command_prefix}\n{command}" if command_prefix else command,
            "cwd": env.cwd,
            "env": {},
            "inherit_env": True,
        }
        if prepare is not None:
            result = prepare(execution, api, context)
            if hasattr(result, "__await__"):
                await result

        exec_options = ShellExecOptions(
            cwd=execution["cwd"],
            env=execution["env"],
            inherit_env=execution["inherit_env"],
            on_output=lambda text, _context, info: api.output(text, info.skipped),
            spill=ShellSpillOptions(after_bytes=DEFAULT_MAX_BYTES, after_lines=DEFAULT_MAX_LINES),
        )
        if timeout is not None:
            exec_options.timeout = timeout
        # An environment may then omit output outside the retained tail and report the omission.
        window: Optional[ShellOutputWindow] = api.output_window
        if window is not None:
            exec_options.window = window

        exec_result = await env.exec(execution["command"], exec_options, context)
        if exec_result.ok:
            spill_path = exec_result.value.spill_path
        else:
            spill_path = getattr(exec_result.error, "spill_path", None)
        if spill_path is not None:
            api.diagnostic(ToolDiagnostic(severity="info", code="full_output", message=f"Full output: {spill_path}"))
        if not exec_result.ok:
            error = exec_result.error
            if error.code == "aborted" and context.aborted:
                raise error
            if error.code == "timeout":
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            if error.code == "aborted":
                raise RuntimeError("Command aborted")
            raise error
        if exec_result.value.exit_code != 0:
            raise RuntimeError(f"Command exited with code {exec_result.value.exit_code}")
        return ToolExecutionResult()

    return define_tool(
        name="bash",
        description=BASH_DESCRIPTION,
        parameters=BASH_SCHEMA,
        execute=execute,
        output_limits=DEFAULT_OUTPUT_LIMITS,
    )

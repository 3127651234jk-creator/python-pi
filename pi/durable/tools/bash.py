"""bash and powershell tools (port of ``tools/bash.ts``).

``bash`` runs a command through the environment's shell; ``powershell`` runs
one through PowerShell instead: ``pwsh`` (PowerShell 7), else Windows
PowerShell, started directly with the command as an argument, so no other
shell parses it. Their output streams to ``api.output()``, where the caller
keeps the tail within its limits; the result content is that retained output.
A nonzero exit or timeout raises, which makes an error result that still
carries the output and diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..env import Context, ExecutionEnv, ShellExecOptions, ShellSpillOptions, ShellOutputWindow
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

POWERSHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "PowerShell command to execute"},
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

POWERSHELL_DESCRIPTION = (
    "Execute a PowerShell command in the current working directory. Returns combined stdout and stderr. "
    f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
    "(whichever is hit first). If truncated, full output is saved to a temp file. "
    "Optionally provide a timeout in seconds."
)

DEFAULT_OUTPUT_LIMITS = {"maxBytes": DEFAULT_MAX_BYTES, "maxLines": DEFAULT_MAX_LINES, "retain": "tail"}

# Output in UTF-8 whatever the console's code page, as the coding agent's `powershell` tool does.
UTF8_OUTPUT = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}"
POWERSHELL_ARGS = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]

DEFAULT_POWERSHELL_PROGRAMS: List[str] = ["pwsh", "powershell"]


def _validate_timeout(timeout: Optional[float]) -> None:
    if timeout is None:
        return
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


async def _prepare_execution(
    command: str,
    options: Optional[Dict[str, Any]],
    api: ToolExecutionApi,
    context: Context,
) -> Dict[str, Any]:
    """The execution of one call: the command with its prefix, in the environment's working directory, then ``prepare``."""
    command_prefix = (options or {}).get("commandPrefix")
    prepare = (options or {}).get("prepare")
    execution: Dict[str, Any] = {
        "command": f"{command_prefix}\n{command}" if command_prefix else command,
        "cwd": require_env(api).cwd,
        "env": {},
        "inherit_env": True,
    }
    if prepare is not None:
        result = prepare(execution, api, context)
        if hasattr(result, "__await__"):
            await result
    return execution


async def _run_command(
    commands: Sequence[Union[str, Sequence[str]]],
    execution: Dict[str, Any],
    timeout: Optional[float],
    api: ToolExecutionApi,
    context: Context,
) -> None:
    """Run each command form in turn until one starts, streaming output to ``api.output()`` within the retained
    window, and turn the result into the tool's outcome: a spill diagnostic, and a thrown error for a failure or a
    nonzero exit."""
    env: ExecutionEnv = require_env(api)
    result = None
    for command in commands:
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
        result = await env.exec(command, exec_options, context)
        # A program that could not start produced no output; the next form may.
        if result.ok or result.error.code != "spawn_error":
            break
    if result is None:
        raise RuntimeError("No command to run")
    spill_path = result.value.spill_path if result.ok else getattr(result.error, "spill_path", None)
    if spill_path is not None:
        api.diagnostic(ToolDiagnostic(severity="info", code="full_output", message=f"Full output: {spill_path}"))
    if not result.ok:
        error = result.error
        if error.code == "aborted" and context.aborted:
            raise error
        if error.code == "timeout":
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        if error.code == "aborted":
            raise RuntimeError("Command aborted")
        raise error
    if result.value.exit_code != 0:
        raise RuntimeError(f"Command exited with code {result.value.exit_code}")


def create_bash_tool(
    options: Optional[Dict[str, Any]] = None,
    command_prefix: Optional[str] = None,
    prepare: Optional[Callable[[dict, ToolExecutionApi, Context], Optional[Any]]] = None,
) -> ToolRegistration:
    """``options`` carries the TS ``BashToolOptions`` shape: ``{"commandPrefix", "prepare"}``."""
    if options is not None:
        command_prefix = options.get("commandPrefix", command_prefix)
        prepare = options.get("prepare", prepare)
    merged: Dict[str, Any] = {"commandPrefix": command_prefix, "prepare": prepare}

    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        command: str = args["command"]
        timeout = args.get("timeout")
        _validate_timeout(timeout)
        execution = await _prepare_execution(command, merged, api, context)
        await _run_command([execution["command"]], execution, timeout, api, context)
        return ToolExecutionResult()

    return define_tool(
        name="bash",
        description=BASH_DESCRIPTION,
        parameters=BASH_SCHEMA,
        execute=execute,
        output_limits=DEFAULT_OUTPUT_LIMITS,
    )


def create_powershell_tool(
    options: Optional[Dict[str, Any]] = None,
    command_prefix: Optional[str] = None,
    prepare: Optional[Callable[[dict, ToolExecutionApi, Context], Optional[Any]]] = None,
    programs: Optional[Sequence[str]] = None,
) -> ToolRegistration:
    """Runs a PowerShell command, like ``bash`` but through PowerShell instead of the environment's shell:
    ``pwsh`` (PowerShell 7), else Windows PowerShell, started directly with the command as an argument, so no
    other shell parses it. ``options`` carries the TS ``PowerShellToolOptions`` shape:
    ``{"commandPrefix", "prepare", "programs"}``. A program that cannot start is skipped."""
    if options is not None:
        command_prefix = options.get("commandPrefix", command_prefix)
        prepare = options.get("prepare", prepare)
        programs = options.get("programs", programs)
    merged: Dict[str, Any] = {
        "commandPrefix": command_prefix,
        "prepare": prepare,
        "programs": list(programs) if programs is not None else list(DEFAULT_POWERSHELL_PROGRAMS),
    }

    async def execute(args: dict, api: ToolExecutionApi, context: Context) -> ToolExecutionResult:
        command: str = args["command"]
        timeout = args.get("timeout")
        _validate_timeout(timeout)
        execution = await _prepare_execution(command, merged, api, context)
        script = f"{UTF8_OUTPUT}\n{execution['command']}"
        commands = [[program, *POWERSHELL_ARGS, script] for program in merged["programs"]]
        await _run_command(commands, execution, timeout, api, context)
        return ToolExecutionResult()

    return define_tool(
        name="powershell",
        description=POWERSHELL_DESCRIPTION,
        parameters=POWERSHELL_SCHEMA,
        execute=execute,
        output_limits=DEFAULT_OUTPUT_LIMITS,
    )

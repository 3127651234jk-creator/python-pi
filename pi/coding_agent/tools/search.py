"""Bounded, cancellable ripgrep subprocesses shared by grep and find."""

import asyncio
import re
import shutil
from typing import Callable, List, Optional

from ...agent.types import AgentToolResult
from ...ai.types import TextContent
from ...util import AbortSignal
from .truncate import DEFAULT_MAX_BYTES


def positive_limit(value: object, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or value <= 0:
        raise ValueError("limit must be a positive integer")
    return value


def bounded_result(lines: List[str], limited: bool, empty: str, byte_limited: bool = False) -> AgentToolResult:
    text = "\n".join(lines)
    raw = text.encode("utf-8")
    byte_limited = byte_limited or len(raw) > DEFAULT_MAX_BYTES
    if byte_limited:
        text = raw[:DEFAULT_MAX_BYTES].decode("utf-8", errors="ignore")
    notices = []
    if limited:
        notices.append("Result limit reached; increase limit or narrow the search")
    if byte_limited:
        notices.append("50KB output limit reached; narrow the search")
    if notices:
        text += "\n\n[" + ". ".join(notices) + "]"
    return AgentToolResult(content=[TextContent(text=text or empty)],
                           details={"resultLimitReached": limited, "byteLimitReached": byte_limited})


async def run_rg(cwd: str, args: List[str], limit: int, signal: Optional[AbortSignal], matches: bool = False,
                 include: Optional[Callable[[str], bool]] = None) -> AgentToolResult:
    if signal:
        signal.throw_if_aborted()
    executable = shutil.which("rg")
    if executable is None:
        raise FileNotFoundError("ripgrep is unavailable; install rg and add it to PATH")
    process = await asyncio.create_subprocess_exec(
        executable, *args, cwd=cwd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=1024 * 1024,
    )

    def stop():
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    remove_listener = signal.add_listener(stop) if signal else None

    async def errors():
        output = bytearray()
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                return output.decode("utf-8", errors="replace")
            if len(output) < 4096:
                output.extend(chunk[:4096 - len(output)])

    error_task = asyncio.create_task(errors())
    lines = []
    count = 0
    size = 0
    limited = False
    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            if signal:
                signal.throw_if_aborted()
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if include is not None and not include(text):
                continue
            is_match = not matches or re.match(r"^.*:\d+:", text) is not None
            if is_match:
                count += 1
                if count > limit:
                    limited = True
                    break
            lines.append(text)
            size += len(line)
            if size > DEFAULT_MAX_BYTES:
                break
        if limited or size > DEFAULT_MAX_BYTES:
            stop()
            await process.stdout.read()
        await process.wait()
        error = await error_task
        if signal:
            signal.throw_if_aborted()
        if not limited and size <= DEFAULT_MAX_BYTES and process.returncode not in (0, 1):
            raise RuntimeError(error.strip() or f"ripgrep exited with code {process.returncode}")
        return bounded_result(lines, limited, "No matches found" if matches else "No files found matching pattern",
                              byte_limited=size > DEFAULT_MAX_BYTES)
    finally:
        if remove_listener:
            remove_listener()
        stop()
        await process.stdout.read()
        await process.wait()
        await error_task

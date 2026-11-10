"""Bounded, cancellable ripgrep subprocesses shared by grep and find.

When ripgrep is not installed the tools fall back to a pure-Python scan that
keeps the same output shape and limit behaviour (slower, but works offline).
"""

import asyncio
import fnmatch
import os
import re
import shutil
from dataclasses import dataclass
from typing import Callable, List, Optional

from ...agent.types import AgentToolResult
from ...ai.types import TextContent
from ...util import AbortSignal
from .truncate import DEFAULT_MAX_BYTES

FALLBACK_MAX_FILE_BYTES = 16 * 1024 * 1024
FALLBACK_MAX_LINE_CHARS = 500
SKIPPED_DIRS = {".git", "node_modules"}


def matches_glob(path: str, pattern: str) -> bool:
    path = path.replace("\\", "/").removeprefix("./")
    pattern = pattern.replace("\\", "/").removeprefix("./")
    if "/" not in pattern:
        return fnmatch.fnmatchcase(path.rsplit("/", 1)[-1], pattern)

    def match(parts, patterns):
        if not patterns:
            return not parts
        if patterns[0] == "**":
            return match(parts, patterns[1:]) or bool(parts) and match(parts[1:], patterns)
        return bool(parts) and fnmatch.fnmatchcase(parts[0], patterns[0]) and match(parts[1:], patterns[1:])

    return match(path.split("/"), pattern.split("/"))


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


@dataclass
class _IgnoreRule:
    pattern: str
    negate: bool
    directory_only: bool
    base: str


def _load_ignore_rules(root: str) -> List[_IgnoreRule]:
    """Read .gitignore rules with their directory scope, parents first."""
    rules: List[_IgnoreRule] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIPPED_DIRS]
        if ".gitignore" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, ".gitignore"), "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        base = os.path.relpath(dirpath, root).replace(os.sep, "/")
        base = "" if base == "." else base + "/"
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:].lstrip()
            directory_only = line.endswith("/")
            pattern = line.rstrip("/")
            if pattern:
                rules.append(_IgnoreRule(pattern, negate, directory_only, base))
    return rules


def _matches_ignore(rel_path: str, pattern: str) -> bool:
    if pattern.startswith("/"):
        anchored_pattern = pattern[1:]
        if "/" in anchored_pattern:
            return matches_glob(rel_path, anchored_pattern)
        return fnmatch.fnmatchcase(rel_path.split("/")[0], anchored_pattern)
    if "/" not in pattern:
        return any(fnmatch.fnmatchcase(part, pattern) for part in rel_path.split("/"))
    # A slash makes the pattern relative to the directory containing .gitignore.
    return matches_glob(rel_path, pattern)


def _is_ignored(rel_path: str, is_dir: bool, rules: List[_IgnoreRule]) -> bool:
    """Last matching rule wins, like git; dir-only rules also cover ancestors."""
    ignored = False
    for rule in rules:
        if not rel_path.startswith(rule.base):
            continue
        scoped_path = rel_path[len(rule.base):]
        parts = scoped_path.split("/")
        directories = ["/".join(parts[:index]) for index in range(1, len(parts))]
        if is_dir:
            directories.append(scoped_path)
        if rule.directory_only:
            matched = any(_matches_ignore(directory, rule.pattern) for directory in directories)
        else:
            matched = _matches_ignore(scoped_path, rule.pattern)
        if matched:
            ignored = not rule.negate
    return ignored


def _walk_files(root: str) -> List[str]:
    rules = _load_ignore_rules(root)
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative = os.path.relpath(dirpath, root)
        relative = "" if relative == "." else relative.replace(os.sep, "/")
        kept = []
        for name in dirnames:
            if name in SKIPPED_DIRS:
                continue
            rel = f"{relative}/{name}" if relative else name
            if not _is_ignored(rel, True, rules):
                kept.append(name)
        dirnames[:] = sorted(kept)
        for name in sorted(filenames):
            rel = f"{relative}/{name}" if relative else name
            if not _is_ignored(rel, False, rules):
                files.append(os.path.join(dirpath, name))
    return files


async def _git_files(root: str) -> Optional[List[str]]:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        process = await asyncio.create_subprocess_exec(
            git, "-C", root, "ls-files", "-z", "-c", "-o", "--exclude-standard",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        data, _ = await process.communicate()
    except OSError:
        return None
    if process.returncode != 0:
        return None
    files = []
    for name in data.decode("utf-8", errors="replace").split("\0"):
        if not name:
            continue
        relative = name.replace("\\", "/")
        if any(part in SKIPPED_DIRS for part in relative.split("/")):
            continue
        files.append(os.path.join(root, relative))
    if not files:
        # An empty .git directory (or a repo with nothing to list) makes git
        # return success with no output; the walk still has to find the files.
        return None
    return sorted(files)


async def list_files(root: str) -> List[str]:
    """Files under root, honouring .gitignore through git when possible."""
    if os.path.isfile(root):
        return [root]
    files = await _git_files(root)
    if files is not None:
        return files
    return await asyncio.to_thread(_walk_files, root)


def _read_text(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > FALLBACK_MAX_FILE_BYTES:
            return None
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if b"\x00" in data[:8192]:  # binary file
        return None
    return data.decode("utf-8", errors="replace")


async def grep_python(root: str, pattern: str, *, glob: Optional[str] = None, ignore_case: bool = False,
                      literal: bool = False, context: int = 0, limit: int = 100,
                      signal: Optional[AbortSignal] = None) -> AgentToolResult:
    """ripgrep-compatible grep used when rg is unavailable."""
    try:
        regex = re.compile(re.escape(pattern) if literal else pattern,
                           re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise RuntimeError(f"invalid pattern: {exc}") from exc
    if signal:
        signal.throw_if_aborted()
    base = root if os.path.isdir(root) else os.path.dirname(root) or "."
    lines: List[str] = []
    count = 0
    size = 0
    limited = False
    byte_limited = False
    for path in await list_files(root):
        if signal:
            signal.throw_if_aborted()
        if not os.path.isfile(path):
            continue
        relative = os.path.relpath(path, base).replace(os.sep, "/")
        if glob and not matches_glob(relative, glob):
            continue
        text = await asyncio.to_thread(_read_text, path)
        if text is None:
            continue
        content = text.splitlines()
        matches = [index for index, line in enumerate(content) if regex.search(line)]
        if not matches:
            continue
        matching_lines = set(matches)
        emitted = set()
        for index in matches:
            count += 1
            if count > limit:
                limited = True
                break
            start = max(0, index - context)
            end = min(len(content), index + context + 1)
            for line_number in range(start, end):
                if line_number in emitted:
                    continue
                emitted.add(line_number)
                line = content[line_number]
                if len(line) > FALLBACK_MAX_LINE_CHARS:
                    line = line[:FALLBACK_MAX_LINE_CHARS] + " [line truncated]"
                marker = ":" if line_number in matching_lines else "-"
                rendered = f"{relative}{marker}{line_number + 1}{marker}{line}"
                lines.append(rendered)
                size += len(rendered.encode("utf-8")) + 1
                byte_limited = size > DEFAULT_MAX_BYTES
            if byte_limited:
                break
        if limited or byte_limited:
            break
    if signal:
        signal.throw_if_aborted()
    return bounded_result(lines, limited, "No matches found", byte_limited=byte_limited)


async def find_python(root: str, pattern: str, limit: int = 1000,
                      signal: Optional[AbortSignal] = None) -> AgentToolResult:
    """ripgrep-compatible file glob used when rg is unavailable."""
    if signal:
        signal.throw_if_aborted()
    found: List[str] = []
    limited = False
    for path in await list_files(root):
        if signal:
            signal.throw_if_aborted()
        if not os.path.isfile(path):
            continue
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        if not matches_glob(relative, pattern):
            continue
        if len(found) >= limit:
            limited = True
            break
        found.append(relative)
    if signal:
        signal.throw_if_aborted()
    return bounded_result(found, limited, "No files found matching pattern")

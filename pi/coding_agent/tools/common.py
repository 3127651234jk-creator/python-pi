"""Shared tool helpers: path resolution and the file mutation queue.

Port of ``path-utils.ts`` and ``file-mutation-queue.ts``: write/edit serialize
per absolute path so concurrent tool calls cannot interleave on one file.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from urllib.parse import unquote, urlsplit
from typing import Any, Awaitable, Callable, Optional
from weakref import WeakValueDictionary

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _normalize_path(path: str, *, tool_input: bool = False) -> str:
    if tool_input:
        path = re.sub(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", path)
        if path.startswith("@"):
            path = path[1:]
    windows = sys.platform == "win32"
    if windows and path.startswith("/") and not path.startswith("//") and "\\" not in path:
        match = re.fullmatch(r"/(?:mnt/|cygdrive/)?([a-z])(?:/(.*))?", path, re.IGNORECASE)
        if match:
            path = match[1].upper() + ":\\" + (match[2] or "").replace("/", "\\")
    if path == "~" or path.startswith("~/") or (windows and path.startswith("~\\")):
        return os.path.join(os.path.expanduser("~"), path[2:]) if path != "~" else os.path.expanduser("~")
    if path.startswith("file://"):
        url = urlsplit(path)
        if re.search(r"%2f", url.path, re.IGNORECASE) or (windows and re.search(r"%5c", url.path, re.IGNORECASE)):
            raise ValueError("File URL must not contain encoded path separators")
        path = unquote(url.path, errors="strict")
        if windows:
            if url.hostname and url.hostname != "localhost":
                path = "\\\\" + url.hostname + path.replace("/", "\\")
            elif re.match(r"^/[a-z]:", path, re.IGNORECASE):
                path = path[1:]
            else:
                raise ValueError("File URL must contain an absolute Windows path")
        elif url.hostname and url.hostname != "localhost":
            raise ValueError("File URL host must be localhost or empty")
    return path


def resolve_to_cwd(path: str, cwd: str) -> str:
    """Normalize tool paths and resolve relative paths against ``cwd``."""
    path = _normalize_path(path, tool_input=True)
    cwd = _normalize_path(cwd)
    return os.path.abspath(os.path.join(cwd, path))


def detect_image_mime_type(path: str):
    return IMAGE_MIME_TYPES.get(os.path.splitext(path)[1].lower())


# Hold locks strongly only while a holder or queued caller needs them.
_MUTATION_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_LOCKS_GUARD: Optional[asyncio.Lock] = None


def _locks_guard() -> asyncio.Lock:
    """The guard, created on first use: an asyncio primitive instantiated at
    import time would spin up (and leak) an event loop on Python 3.9."""
    global _LOCKS_GUARD
    if _LOCKS_GUARD is None:
        _LOCKS_GUARD = asyncio.Lock()
    return _LOCKS_GUARD


async def with_file_mutation_queue(absolute_path: str, fn: Callable[[], Awaitable[Any]]) -> Any:
    """Run ``fn`` while holding the mutation lock for ``absolute_path``."""

    # Resolve aliases (including Windows junctions, SUBST drives and short names)
    # before case folding. Non-strict realpath also resolves existing parents
    # when a write is creating a file that does not exist yet.
    key = os.path.normcase(os.path.realpath(absolute_path))
    async with _locks_guard():
        lock = _MUTATION_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        return await fn()

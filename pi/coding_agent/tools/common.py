"""Shared tool helpers: path resolution and the file mutation queue.

Port of ``path-utils.ts`` and ``file-mutation-queue.ts``: write/edit serialize
per absolute path so concurrent tool calls cannot interleave on one file.
"""

from __future__ import annotations

import asyncio
import os
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


def resolve_to_cwd(path: str, cwd: str) -> str:
    """Resolve a tool path argument to an absolute path under ``cwd``."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(cwd, path))


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

    async with _locks_guard():
        lock = _MUTATION_LOCKS.setdefault(absolute_path, asyncio.Lock())
    async with lock:
        return await fn()

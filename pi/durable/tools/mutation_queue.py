"""Per-file mutation serialization (port of ``tools/file-mutation-queue.ts``).

``edit`` and ``write`` mutations of one file serialize within this process:
same file system id and canonical path, whichever environment object the call
got. Other files, and other file systems, never wait. Not a lock against
``bash`` or other processes.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict

from ..env import Context, ExecutionEnv, Result, get_or_throw

_QUEUES: Dict[str, asyncio.Lock] = {}
_GUARD = asyncio.Lock()


async def _mutation_key(env: ExecutionEnv, path: str, context: Context) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path, context))
    return f"{env.id}\0{await _canonical(env, absolute_path, context)}"


async def _canonical(env: ExecutionEnv, absolute_path: str, context: Context) -> str:
    """The canonical path; for a missing file, its canonical parent joined with its name."""
    result: Result = await env.canonical_path(absolute_path, context)
    if result.ok:
        return result.value
    code = getattr(result.error, "code", None)
    if code == "not_supported":
        return absolute_path
    if code != "not_found":
        raise result.error
    # The file system splits the path, so a name may contain characters that are separators elsewhere.
    parent = get_or_throw(await env.join_path([absolute_path, ".."], context))
    if parent == absolute_path or not absolute_path.startswith(parent):
        return absolute_path
    name = absolute_path[len(parent) + (0 if parent.endswith(("/", "\\")) else 1):]
    return get_or_throw(await env.join_path([await _canonical(env, parent, context), name], context))


async def with_file_mutation_queue(
    env: ExecutionEnv, path: str, fn: Callable[[], Awaitable[Any]], context: Context
) -> Any:
    """Run ``fn`` while holding the mutation slot of one file; concurrent calls queue in key order."""
    key = await _mutation_key(env, path, context)
    async with _GUARD:
        lock = _QUEUES.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _QUEUES[key] = lock
    async with lock:
        try:
            return await fn()
        finally:
            async with _GUARD:
                # Only the current holder cleans up, so a queued waiter never drops a fresh lock.
                if _QUEUES.get(key) is lock and not lock.locked():
                    _QUEUES.pop(key, None)

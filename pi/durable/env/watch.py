"""Polling file watching (port of the ``FileWatcher`` contract of ``env/index.ts``).

Porting deviation: the TS node implementation uses native FS watchers with a
polling fallback; the Python port ships the polling implementation only. It
compares snapshots of the watched trees each tick, so a change undone between
two snapshots can be missed (``mode`` is ``"polling"``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import Context, FileError, FileInfo, WatchChange, WatchTarget


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


class _Snapshot:
    """One scan of the watched trees: path -> (kind, size, mtime_ns)."""

    def __init__(self) -> None:
        self.entries: Dict[str, Tuple[str, int, int]] = {}

    def add(self, path: str, kind: str, size: int, mtime_ns: int) -> None:
        self.entries[path] = (kind, size, mtime_ns)


class PollingFileWatcher:
    """Watches ``targets`` by rescanning them each interval; reports batched path changes."""

    def __init__(
        self,
        targets: List[WatchTarget],
        resolve: Callable[[str], str],
        on_change: Callable[[WatchChange], None],
        interval_ms: int = 250,
    ) -> None:
        self.mode = "polling"
        self._targets = targets
        self._resolve = resolve
        self._on_change = on_change
        self._interval = interval_ms / 1000
        self._closed = False
        # Coverage is established when the watcher opens: the first snapshot is taken now, so a change
        # made right after watching starts is a diff, not a swallowed baseline.
        self._previous: Optional[_Snapshot] = self._scan()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self, context: Context) -> None:
        """Stop watching; no ``on_change`` call starts after this resolves. Idempotent."""
        self._closed = True
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            while not self._closed:
                snapshot = self._scan()
                if self._closed:
                    return
                if self._previous is not None:
                    changed = _diff(self._previous, snapshot)
                    if changed:
                        self._on_change({"paths": changed})
                self._previous = snapshot
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as the watcher's terminal error
            if not self._closed:
                self._on_change({"error": FileError("unknown", str(exc))})

    def _scan(self) -> _Snapshot:
        snapshot = _Snapshot()
        for target in self._targets:
            absolute = self._resolve(target.path)
            exclude = target.exclude or {}
            hidden = bool(exclude.get("hidden", False))
            excluded_names = set(exclude.get("names") or [])

            def excluded(name: str) -> bool:
                return (hidden and _is_hidden(name)) or name in excluded_names

            self._scan_path(snapshot, absolute, bool(target.recursive), excluded)
        return snapshot

    def _scan_path(self, snapshot: _Snapshot, path: str, recursive: bool, excluded: Callable[[str], bool]) -> None:
        try:
            stat = os.lstat(path)
        except OSError:
            return  # Missing targets contribute nothing until they exist.
        kind = _kind_of(stat)
        name = os.path.basename(path)
        if excluded(name):
            return
        snapshot.add(path, kind, stat.st_size, stat.st_mtime_ns)
        if kind != "directory":
            return
        try:
            with os.scandir(path) as entries:
                children = list(entries)
        except OSError:
            return
        for child in children:
            if excluded(child.name):
                continue
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError:
                continue  # Removed between enumeration and stat: not part of the listing any more.
            child_kind = _kind_of_stat(child_stat)
            child_path = os.path.join(path, child.name)
            snapshot.add(child_path, child_kind, child_stat.st_size, child_stat.st_mtime_ns)
            if recursive and child_kind == "directory":
                self._scan_path(snapshot, child_path, True, excluded)


def _kind_of(stat: os.stat_result) -> str:
    import stat as stat_module

    if stat_module.S_ISREG(stat.st_mode):
        return "file"
    if stat_module.S_ISDIR(stat.st_mode):
        return "directory"
    if stat_module.S_ISLNK(stat.st_mode):
        return "symlink"
    return "invalid"


def _kind_of_stat(stat: os.stat_result) -> str:
    return _kind_of(stat)


def _diff(previous: _Snapshot, current: _Snapshot) -> List[str]:
    """Paths whose entry, content, or children changed, reported at or above the change."""
    changed: Set[str] = set()
    for path, identity in previous.entries.items():
        if path not in current.entries:
            changed.add(_parent_or_self(path))
        elif current.entries[path] != identity:
            changed.add(path)
    for path in current.entries:
        if path not in previous.entries:
            changed.add(path)
    return sorted(changed)


def _parent_or_self(path: str) -> str:
    parent = os.path.dirname(path)
    return parent if parent else path

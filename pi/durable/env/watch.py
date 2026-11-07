"""File watching (port of the ``FileWatcher`` contract of ``env/index.ts`` and the
``NodeFileWatcher`` of ``env/node-watch.ts``).

Watches by snapshots: native events only trigger a debounced rescan, and changes
are the difference between snapshots plus the event paths. A replaced file, a
renamed or recreated ancestor, or a directory created together with its contents
therefore never depends on which events an operating system sends. New
directories get their watchers before they are scanned again, so nothing
written into them before the watcher existed is missed.

Native backends live in ``native_backends`` (Windows ``ReadDirectoryChangesW``,
Linux ``inotify``, macOS FSEvents). When a native backend cannot start or dies,
the watcher switches to comparing snapshots on an interval and reports
``{"overflow": True}`` once.

Porting deviations from the TS implementation:
- Windows chooses ``native`` by default, where TS polls: Windows refuses to
  rename a directory while a handle below it is open, so no handle is ever held
  on a watched target itself; handles sit on each target's nearest existing
  ancestor directory instead, and renaming the target or its parent keeps
  working. Renaming an ancestor above that is denied by Windows while watching.
  A two-second rescan (the contract's "within about two seconds") heals
  coverage after such renames on every platform.
- ``poll_interval_ms`` defaults to 250 (TS: 2000), matching the interval the
  Python port has always polled with.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import stat as stat_module
import time
from typing import Callable, Dict, FrozenSet, List, Optional, NamedTuple, Set

from . import FileError, WatchChange, WatchTarget
from .native_backends import (
    DARWIN,
    LINUX,
    WINDOWS,
    ancestors_of,
    any_unreliable_file_system,
    default_backend,
    read_shared,
)

DEBOUNCE_SECONDS = 0.05
# On macOS, FSEvents may not be live when a watch is installed; changes made in
# between are caught by a rescan this long after installing watchers.
FSEVENTS_SETTLE_SECONDS = 0.5
# Native watchers also rescan on this cadence: it re-establishes coverage that a
# renamed ancestor or a missed event broke, within the reported latency contract.
NATIVE_RESYNC_SECONDS = 2.0
DEFAULT_POLL_INTERVAL_MS = 250
DEFAULT_MAX_DIRECTORIES = 10_000
# In polling mode, recently modified small files are also compared by content: a
# second write within the file system's timestamp granularity can keep size and
# modification time.
HASH_MAX_BYTES = 256 * 1024
HASH_RECENT_MS = 5000.0

_IGNORED_WATCH_ERRNOS = frozenset({errno.ENOENT, errno.EACCES, errno.EPERM})


class BudgetExceeded(Exception):
    pass


class ResolvedTarget(NamedTuple):
    path: str
    recursive: bool
    hidden: bool
    names: FrozenSet[str]


class _Entry(NamedTuple):
    """What a snapshot remembers of one path. Ancestors compare by identity only."""

    kind: str  # "file" | "directory" | "symlink" | "other"
    dev: int
    ino: int
    size: int
    mtime_ns: int
    hash: Optional[str] = None


def _entry_of(stats: os.stat_result, content_hash: Optional[str] = None) -> _Entry:
    kind = "file"
    if stat_module.S_ISDIR(stats.st_mode):
        kind = "directory"
    elif stat_module.S_ISLNK(stats.st_mode):
        kind = "symlink"
    elif not stat_module.S_ISREG(stats.st_mode):
        kind = "other"
    # Directories are compared by identity only: their own timestamps change with
    # every unrelated child.
    size = 0 if kind == "directory" else stats.st_size
    mtime_ns = 0 if kind == "directory" else stats.st_mtime_ns
    return _Entry(kind, stats.st_dev, stats.st_ino, size, mtime_ns, content_hash)


def _is_within(path: str, ancestor: str) -> bool:
    """Whether ``path`` is ``ancestor`` or below it."""
    if path == ancestor:
        return True
    prefix = ancestor if ancestor.endswith(os.sep) else ancestor + os.sep
    if WINDOWS:
        return path.lower().startswith(prefix.lower())
    return path.startswith(prefix)


def _excluded(target: ResolvedTarget, name: str) -> bool:
    return (target.hidden and name.startswith(".")) or name in target.names


def _is_directory(snapshot: Dict[str, _Entry], path: str) -> bool:
    entry = snapshot.get(path)
    return entry is not None and entry.kind == "directory"


class LocalFileWatcher:
    """Watches resolved targets in ``native`` or ``polling`` mode; open with ``open()``."""

    def __init__(
        self,
        targets: List[ResolvedTarget],
        on_change: Callable[[WatchChange], None],
        mode: str,
        poll_interval_ms: int,
        max_directories: int,
    ) -> None:
        self.mode = mode
        self._targets = targets
        self._on_change = on_change
        self._poll_interval = poll_interval_ms / 1000
        self._max_directories = max_directories
        self._loop = asyncio.get_running_loop()
        self._closed = False
        self._dirty = False
        self._pending_events: Set[str] = set()
        self._snapshot: Dict[str, _Entry] = {}
        self._watched: Dict[str, bool] = {}
        self._backend = None
        self._flush_handle: Optional[asyncio.TimerHandle] = None
        self._poll_handle: Optional[asyncio.TimerHandle] = None
        self._resync_handle: Optional[asyncio.TimerHandle] = None
        self._settle_handle: Optional[asyncio.TimerHandle] = None
        self._flush_task: Optional[asyncio.Task] = None

    @classmethod
    async def open(
        cls,
        targets: List[WatchTarget],
        resolve_path: Callable[[str], str],
        on_change: Callable[[WatchChange], None],
        mode: str = "auto",
        poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
        max_directories: int = DEFAULT_MAX_DIRECTORIES,
    ) -> "LocalFileWatcher":
        """Establish coverage: watchers first, then the snapshot later changes are compared with."""
        resolved = [
            ResolvedTarget(
                path=resolve_path(target.path),
                recursive=target.recursive is True,
                hidden=bool((target.exclude or {}).get("hidden", False)),
                names=frozenset((target.exclude or {}).get("names") or []),
            )
            for target in targets
        ]
        if mode not in ("native", "polling"):  # "auto" and anything unrecognized
            if default_backend() is None:
                chosen = "polling"
            elif LINUX and any_unreliable_file_system([target.path for target in resolved]):
                chosen = "polling"
            else:
                chosen = "native"
        else:
            chosen = mode
        watcher = cls(resolved, on_change, chosen, poll_interval_ms, max_directories)
        try:
            watcher._sync(report=False)
        except BudgetExceeded as exc:
            watcher._stop()
            raise FileError("invalid", str(exc)) from None
        except BaseException:
            watcher._stop()
            raise
        watcher._schedule_poll()
        watcher._schedule_resync()
        return watcher

    async def close(self, context) -> None:
        """Stop watching; no ``on_change`` call starts after this resolves. Idempotent."""
        if self._closed:
            return
        self._stop()
        task = self._flush_task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    def _stop(self) -> None:
        self._closed = True
        for handle in (self._flush_handle, self._poll_handle, self._resync_handle, self._settle_handle):
            if handle is not None:
                handle.cancel()
        self._flush_handle = None
        self._poll_handle = None
        self._resync_handle = None
        self._settle_handle = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._watched.clear()

    def _deliver(self, change: WatchChange) -> None:
        if self._closed:
            return
        try:
            self._on_change(change)
        except Exception:
            pass  # A throwing callback must not stop watching.

    # -- scheduling -----------------------------------------------------------

    def _schedule_poll(self) -> None:
        if self._closed or self.mode != "polling":
            return
        self._poll_handle = self._loop.call_later(self._poll_interval, self._on_poll)

    def _on_poll(self) -> None:
        self._poll_handle = None
        if self._closed:
            return
        self._flush().add_done_callback(self._reschedule_poll)

    def _reschedule_poll(self, task: asyncio.Task) -> None:
        try:
            task.exception()
        except BaseException:
            pass
        self._schedule_poll()

    def _schedule_resync(self) -> None:
        if self._closed or self.mode != "native":
            return
        self._resync_handle = self._loop.call_later(NATIVE_RESYNC_SECONDS, self._on_resync)

    def _on_resync(self) -> None:
        self._resync_handle = None
        if self._closed:
            return
        self._flush().add_done_callback(self._reschedule_resync)

    def _reschedule_resync(self, task: asyncio.Task) -> None:
        try:
            task.exception()
        except BaseException:
            pass
        self._schedule_resync()

    def _schedule_settle(self) -> None:
        if self._closed:
            return
        if self._settle_handle is not None:
            self._settle_handle.cancel()
        self._settle_handle = self._loop.call_later(FSEVENTS_SETTLE_SECONDS, self._on_settle)

    def _on_settle(self) -> None:
        self._settle_handle = None
        if self._closed:
            return
        self._flush()

    def _schedule_flush(self) -> None:
        if self._closed or self._flush_handle is not None:
            return
        self._flush_handle = self._loop.call_later(DEBOUNCE_SECONDS, self._on_debounce)

    def _on_debounce(self) -> None:
        self._flush_handle = None
        if self._closed:
            return
        self._flush()

    def _flush(self) -> asyncio.Task:
        if self._flush_task is not None and not self._flush_task.done():
            self._dirty = True
            return self._flush_task
        self._flush_task = self._loop.create_task(self._flush_body())
        return self._flush_task

    async def _flush_body(self) -> None:
        try:
            while not self._closed:
                self._dirty = False
                events = self._pending_events
                self._pending_events = set()
                changed = self._sync(report=True)
                changed.update(events)
                if changed:
                    self._deliver({"paths": sorted(changed)})
                if not self._dirty or self._closed:
                    break
                await asyncio.sleep(0)
        except BaseException as exc:
            error = exc if isinstance(exc, FileError) else FileError("invalid", str(exc))
            self._deliver({"error": error})
            self._stop()
        finally:
            self._flush_task = None

    # -- events ---------------------------------------------------------------

    def _on_backend_event(self, directory: Optional[str], filename: Optional[str]) -> None:
        """Runs on backend threads: filter cheaply, then hand over to the loop.

        With ``filename`` absent, ``directory`` is a full path that changed;
        with both absent it is a pure wake-up. Excluded or unrelated events are
        dropped here so churn below excluded directories costs no rescans.
        """
        if directory is None:
            self._hop(self._record_event, None)
            return
        path = os.path.join(directory, filename) if filename is not None else directory
        if not self._in_scope(path):
            return
        self._hop(self._record_event, self._reported(path))

    def _hop(self, callback: Callable[[Optional[str]], None], value: Optional[str]) -> None:
        try:
            self._loop.call_soon_threadsafe(callback, value)
        except RuntimeError:
            pass  # The loop is already closed; close() has taken the watcher down.

    def _record_event(self, path: Optional[str]) -> None:
        if self._closed:
            return
        if path is not None:
            self._pending_events.add(path)
        self._schedule_flush()

    def _in_scope(self, path: str) -> bool:
        """Whether an event about ``path`` can affect what is reported."""
        for target in self._targets:
            if _is_within(target.path, path):
                return True
            if not _is_within(path, target.path) or path == target.path:
                continue
            try:
                components = os.path.relpath(path, target.path).split(os.sep)
            except ValueError:
                continue
            if not target.recursive and len(components) > 1:
                continue
            if any(_excluded(target, name) for name in components):
                continue
            return True
        return False

    def _reported(self, path: str) -> str:
        """An ancestor that changed identity moved every target below it; report those targets."""
        for target in self._targets:
            if _is_within(path, target.path):
                return path
        for target in self._targets:
            if _is_within(target.path, path):
                return target.path
        return path

    # -- scanning -------------------------------------------------------------

    def _sync(self, report: bool) -> Set[str]:
        """Rescan, report differences, and install watchers for new directories, until none are new."""
        changed: Set[str] = set()
        for _round in range(10):
            if self._closed:
                break
            snapshot = self._scan()
            if report:
                changed.update(self._diff(self._snapshot, snapshot))
            self._snapshot = snapshot
            if self.mode == "polling" or not self._reconcile(snapshot):
                break
            if DARWIN:
                self._schedule_settle()
            # Something written into a new directory before its watcher existed
            # shows up in the next round.
            report = True
        return changed

    def _diff(self, previous: Dict[str, _Entry], current: Dict[str, _Entry]) -> List[str]:
        changed: List[str] = []
        for path, entry in current.items():
            before = previous.get(path)
            if before is None or before != entry:
                changed.append(self._reported(path))
        for path in previous:
            if path not in current:
                changed.append(self._reported(path))
        return changed

    def _scan(self) -> Dict[str, _Entry]:
        snapshot: Dict[str, _Entry] = {}
        directories = 0

        def count_directory() -> None:
            nonlocal directories
            directories += 1
            if directories > self._max_directories:
                raise BudgetExceeded(f"Watched paths exceed {self._max_directories} directories")

        for target in self._targets:
            for ancestor in ancestors_of(target.path):
                if ancestor in snapshot:
                    continue
                try:
                    stats = os.lstat(ancestor)
                except OSError:
                    continue  # Missing ancestors contribute nothing until they exist.
                # Identity only: an ancestor's own timestamps change with every unrelated sibling.
                snapshot[ancestor] = _entry_of(stats)._replace(size=0, mtime_ns=0)
            # The target itself may be a symbolic link to what is watched; follow it.
            try:
                stats = os.stat(target.path)
            except OSError:
                continue
            self._record(snapshot, target.path, stats)
            if stat_module.S_ISDIR(stats.st_mode):
                count_directory()
                self._scan_directory(snapshot, target, target.path, count_directory)
        return snapshot

    def _record(self, snapshot: Dict[str, _Entry], path: str, stats: os.stat_result) -> None:
        content_hash: Optional[str] = None
        if (
            self.mode == "polling"
            and stat_module.S_ISREG(stats.st_mode)
            and stats.st_size <= HASH_MAX_BYTES
            and time.time() * 1000 - stats.st_mtime * 1000 < HASH_RECENT_MS
        ):
            content = read_shared(path)
            if content is not None:
                content_hash = hashlib.sha256(content).hexdigest()
        snapshot[path] = _entry_of(stats, content_hash)

    def _scan_directory(
        self,
        snapshot: Dict[str, _Entry],
        target: ResolvedTarget,
        directory: str,
        count_directory: Callable[[], None],
    ) -> None:
        try:
            children = list(os.scandir(directory))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return  # Gone or unreadable since the parent listing: nothing to record.
        for child in children:
            if _excluded(target, child.name):
                continue
            path = os.path.join(directory, child.name)
            if path in snapshot:
                continue
            try:
                stats = child.stat(follow_symlinks=False)
            except OSError:
                continue  # Removed between enumeration and stat: not part of the listing any more.
            self._record(snapshot, path, stats)
            if target.recursive and stat_module.S_ISDIR(stats.st_mode):
                count_directory()
                self._scan_directory(snapshot, target, path, count_directory)

    # -- native watchers ------------------------------------------------------

    def _reconcile(self, snapshot: Dict[str, _Entry]) -> bool:
        """Watch what the snapshot needs; returns whether a watcher was added."""
        wanted = self._wanted_dirs(snapshot)
        for path in list(self._watched):
            if path not in wanted:
                self._backend.unwatch(path)
                del self._watched[path]
        added = False
        try:
            if self._backend is None:
                factory = default_backend()
                if factory is None:
                    raise OSError(errno.ENOSYS, "No native watch backend on this platform")
                self._backend = factory(self._on_backend_event)
            for path in wanted:
                if path in self._watched:
                    continue
                self._backend.watch(path)
                self._watched[path] = True
                added = True
            if added:
                self._backend.commit()
        except OSError as exc:
            # Out of watches or a vanished directory: the latter is re-armed by the
            # next rescan; anything else means compare snapshots from now on.
            if exc.errno in _IGNORED_WATCH_ERRNOS:
                return added
            self._switch_to_polling()
            return False
        return added

    def _wanted_dirs(self, snapshot: Dict[str, _Entry]) -> Dict[str, bool]:
        """Directories to hold native watches on. Windows never holds a handle on a
        target itself (that would block renaming its parent), so there the nearest
        existing ancestor is watched recursively; elsewhere TS's shape applies."""
        wanted: Dict[str, bool] = {}
        if WINDOWS:
            for target in self._targets:
                candidate = self._nearest_existing_ancestor(snapshot, target.path)
                if candidate is None:
                    continue
                if any(_is_within(candidate, watched) for watched in wanted):
                    continue
                wanted[candidate] = True
                for watched in list(wanted):
                    if watched != candidate and _is_within(watched, candidate):
                        del wanted[watched]
            return wanted
        for target in self._targets:
            for ancestor in ancestors_of(target.path):
                if _is_directory(snapshot, ancestor) and ancestor not in wanted:
                    wanted[ancestor] = False
            if _is_directory(snapshot, target.path):
                wanted[target.path] = bool(target.recursive)
                if LINUX and target.recursive:
                    # Linux inotify has no recursion: one watch per directory below.
                    for path, entry in snapshot.items():
                        if entry.kind == "directory" and path != target.path and _is_within(path, target.path):
                            wanted[path] = False
        return wanted

    def _nearest_existing_ancestor(self, snapshot: Dict[str, _Entry], path: str) -> Optional[str]:
        current = path
        while True:
            parent = os.path.dirname(current)
            if _is_directory(snapshot, parent):
                return parent
            if parent == current:
                return None
            current = parent

    def _switch_to_polling(self) -> None:
        if self.mode == "polling":
            return
        self.mode = "polling"
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._watched.clear()
        self._deliver({"overflow": True})
        if self._resync_handle is not None:
            self._resync_handle.cancel()
            self._resync_handle = None
        self._schedule_poll()

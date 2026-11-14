"""File watcher tests: watch conformance cases in both modes, budget, and fallback.

Mirrors the TS ``env-node-conformance.test.ts`` double registration (native and
polling watches) plus its watch-limits describe block.
"""

from __future__ import annotations

import asyncio
import errno
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.durable.env import Context, LocalExecutionEnv, WatchTarget
from pi.durable.env import watch as watch_module
from pi.durable.testing import create_env_conformance
from pi.durable.testing.types import ConformanceAssertions, EnvConformanceOptions

WATCH_CASE_NAMES = [
    "watch reports a missing file's creation, changes, replacement and removal",
    "watch reports a missing target whose ancestors are created",
    "watch follows directories created together with their contents",
    "watch keeps watching a path whose parent is renamed and recreated",
    "watch skips excluded entries and reports a rename out of them",
    "watch keeps recursive coverage where a non-recursive target overlaps",
    "watch follows a directory replaced at the same path",
    "watch stops reporting once closed",
    "watch reports changes to the file a watched symbolic link points to",
]

SYMLINK_CASE_PREFIX = "watch reports changes to the file a watched symbolic link"

MODES = ["native", "polling"]


def _watch_case_names():
    options = EnvConformanceOptions(
        assertions=ConformanceAssertions(), with_env=None, shell=["sh", "-c"], symlinks=True
    )
    return [case.name for case in create_env_conformance(options) if case.name.startswith("watch ")]


def test_watch_case_names_match_conformance():
    assert _watch_case_names() == WATCH_CASE_NAMES


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case_name", WATCH_CASE_NAMES)
def test_watch_conformance(case_name, mode, tmp_path):
    if case_name.startswith(SYMLINK_CASE_PREFIX) and os.name == "nt":
        pytest.skip("symbolic links need privileges on Windows")

    async def main():
        env = LocalExecutionEnv(cwd=str(tmp_path), watch_mode=mode, watch_interval_ms=100)
        try:

            async def with_env(use):
                await use(env)

            options = EnvConformanceOptions(
                assertions=ConformanceAssertions(), with_env=with_env, shell=["sh", "-c"], symlinks=True
            )
            cases = create_env_conformance(options)
            target = next(candidate for candidate in cases if candidate.name == case_name)
            await asyncio.wait_for(target.run(), timeout=30.0)
        finally:
            await env.cleanup(Context())

    asyncio.run(main())


def test_auto_mode_chooses_native_where_supported(tmp_path):
    async def main():
        context = Context()
        env = LocalExecutionEnv(cwd=str(tmp_path))
        try:
            changes = []
            opened = await env.watch([WatchTarget(path="file.txt")], changes.append, context)
            assert opened.ok
            watcher = opened.value
            try:
                supported = (
                    sys.platform == "win32" or sys.platform == "darwin" or sys.platform.startswith("linux")
                )
                assert watcher.mode == ("native" if supported else "polling")
            finally:
                await watcher.close(context)
        finally:
            await env.cleanup(context)

    asyncio.run(main())


def test_watch_refuses_and_reports_directory_budget(tmp_path):
    async def main():
        context = Context()
        env = LocalExecutionEnv(cwd=str(tmp_path), watch_max_directories=3)
        try:
            for path in ("tree/a/b", "tree/c"):
                created = await env.create_dir(path, None, context)
                assert created.ok
            refused = await env.watch(
                [WatchTarget(path="tree", recursive=True)], lambda change: None, context
            )
            assert not refused.ok
            assert refused.error.code == "invalid"

            removed = await env.remove("tree/c", {"recursive": True}, context)
            assert removed.ok
            changes = []
            opened = await env.watch(
                [WatchTarget(path="tree", recursive=True)], changes.append, context
            )
            assert opened.ok
            watcher = opened.value
            try:
                created = await env.create_dir("tree/d", None, context)
                assert created.ok
                deadline = time.monotonic() + 3.0
                while not changes or "error" not in changes[-1]:
                    if time.monotonic() > deadline:
                        raise AssertionError(f"No budget error; got {changes!r}")
                    await asyncio.sleep(0.02)
                assert changes[-1]["error"].code == "invalid"
            finally:
                await watcher.close(context)
        finally:
            await env.cleanup(context)

    asyncio.run(main())


class _FailingBackend:
    """A native backend whose watches always fail with an unexpected error."""

    def __init__(self, on_event):
        self._on_event = on_event

    def watch(self, directory):
        raise OSError(errno.EIO, "backend unavailable")

    def unwatch(self, directory):
        pass

    def commit(self):
        pass

    def close(self):
        pass


def test_watch_reports_a_target_it_cannot_read_as_permission_denied(tmp_path, monkeypatch):
    # A target that exists but cannot be reached is denied, not missing; a rescan that meets
    # the same denial reports the same code instead of "invalid".
    secret = os.path.join(str(tmp_path), "secret")
    original_stat = os.stat

    def denied(path, *args, **kwargs):
        if os.path.abspath(str(path)) == os.path.abspath(secret):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_stat(path, *args, **kwargs)

    async def main():
        context = Context()
        env = LocalExecutionEnv(cwd=str(tmp_path), watch_mode="polling", watch_interval_ms=50)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(watch_module.os, "stat", denied)
                refused = await env.watch([WatchTarget(path="secret")], lambda change: None, context)
            assert not refused.ok
            assert refused.error.code == "permission_denied"

            changes = []
            opened = await env.watch([WatchTarget(path="secret")], changes.append, context)
            assert opened.ok
            watcher = opened.value
            try:
                with monkeypatch.context() as patch:
                    patch.setattr(watch_module.os, "stat", denied)
                    deadline = time.monotonic() + 3.0
                    while not any("error" in change for change in changes):
                        if time.monotonic() > deadline:
                            raise AssertionError(f"No error change; got {changes!r}")
                        await asyncio.sleep(0.02)
                assert changes[-1]["error"].code == "permission_denied"
            finally:
                await watcher.close(context)
        finally:
            await env.cleanup(Context())

    asyncio.run(main())


def test_windows_does_not_watch_the_file_a_target_links_to(tmp_path, monkeypatch):
    # Windows cannot watch a file handle (ReadDirectoryChangesW fails), and a failed watch
    # would degrade the whole watcher to polling; the rescan covers the linked file instead.

    async def wanted(platform_windows):
        monkeypatch.setattr(watch_module, "WINDOWS", platform_windows)
        monkeypatch.setattr(watch_module, "LINUX", False)
        link = os.path.join(str(tmp_path), "config", "AGENTS.md")
        config = os.path.dirname(link)
        watcher = watch_module.LocalFileWatcher(
            [watch_module.ResolvedTarget(path=link, recursive=False, hidden=False, names=frozenset())],
            lambda change: None,
            "native",
            100,
            watch_module.DEFAULT_MAX_DIRECTORIES,
        )
        snapshot = {
            config: watch_module._Entry("directory", 1, 1, 0, 0, None),
            link: watch_module._Entry("file", 1, 2, 3, 4, None),
        }
        return watcher._wanted_dirs(snapshot, {link})

    windows_wanted = asyncio.run(wanted(True))
    link = os.path.join(str(tmp_path), "config", "AGENTS.md")
    assert link not in windows_wanted
    posix_wanted = asyncio.run(wanted(False))
    assert posix_wanted.get(link) is False


def test_native_backend_failure_switches_to_polling(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_module, "default_backend", lambda: _FailingBackend)

    async def main():
        context = Context()
        env = LocalExecutionEnv(cwd=str(tmp_path), watch_mode="native")
        try:
            changes = []
            opened = await env.watch([WatchTarget(path="file.txt")], changes.append, context)
            assert opened.ok
            watcher = opened.value
            try:
                assert watcher.mode == "polling"
                assert any("overflow" in change for change in changes)

                target = os.path.join(str(tmp_path), "file.txt")
                wrote = await env.write_file("file.txt", "one", context)
                assert wrote.ok
                deadline = time.monotonic() + 3.0
                while not any(
                    "paths" in change and any(path == target for path in change["paths"])
                    for change in changes
                ):
                    if time.monotonic() > deadline:
                        raise AssertionError(f"No change reported {target}; got {changes!r}")
                    await asyncio.sleep(0.02)
            finally:
                await watcher.close(context)
        finally:
            await env.cleanup(context)

    asyncio.run(main())

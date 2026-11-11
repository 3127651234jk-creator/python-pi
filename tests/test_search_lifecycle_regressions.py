import asyncio

import pytest

from pi.coding_agent.tools import search
from pi.util import AbortSignal


def test_removing_listener_after_abort_does_not_call_it_again():
    async def check():
        signal = AbortSignal()
        signal.abort()
        calls = []
        remove = signal.add_listener(lambda: calls.append("called"))
        remove()
        remove()
        assert calls == ["called"]

    asyncio.run(check())


def test_fallback_grep_bounds_context_before_collecting_all_lines(tmp_path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_text("needle\n" + "x\n" * 10000, encoding="utf-8")
    monkeypatch.setattr(search.shutil, "which", lambda name: None)
    original = search.bounded_result
    collected = []

    def bounded(lines, *args, **kwargs):
        collected.extend(lines)
        return original(lines, *args, **kwargs)

    monkeypatch.setattr(search, "bounded_result", bounded)
    result = asyncio.run(search.grep_python(str(tmp_path), "needle", context=10000))
    assert result.details["byteLimitReached"]
    assert sum(len(line.encode("utf-8")) + 1 for line in collected) < search.DEFAULT_MAX_BYTES + 1024


def test_cancelled_git_discovery_reaps_subprocess(monkeypatch):
    async def check():
        started = asyncio.Event()

        class Process:
            returncode = None
            killed = False
            waited = False

            async def communicate(self):
                if self.killed:
                    return b"", None
                started.set()
                await asyncio.Event().wait()

            def kill(self):
                self.killed = True
                self.returncode = -1

            async def wait(self):
                self.waited = True
                return self.returncode

        process = Process()

        async def spawn(*args, **kwargs):
            return process

        monkeypatch.setattr(search.shutil, "which", lambda name: "git")
        monkeypatch.setattr(search.asyncio, "create_subprocess_exec", spawn)
        task = asyncio.create_task(search._git_files("."))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed
        assert process.waited

    asyncio.run(check())

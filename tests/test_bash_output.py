"""Regression coverage for bounded shell output."""
import asyncio
from ctypes import wintypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi.coding_agent.tools import bash as bash_module
from pi.coding_agent.tools.bash import DEFAULT_MAX_BYTES, OutputAccumulator, create_bash_tool
from pi.coding_agent.tools.truncate import truncate_tail
from pi.coding_agent.tools.windows_process import WindowsJob


def test_accumulator_bounds_memory_and_preserves_log():
    output = OutputAccumulator()
    data = ("中文 line\n" * 100000).encode()
    try:
        for offset in range(0, len(data), 4096):
            output.append(data[offset:offset + 4096])
        output.close()
        prefix, truncation, path = output.snapshot()
        assert len(output._tail) <= 2 * DEFAULT_MAX_BYTES
        assert len(prefix) <= 1024 * 1024
        assert truncation == truncate_tail(data.decode())
        assert Path(path).read_bytes() == data
    finally:
        output.close()
        os.unlink(output._log.name)


@pytest.mark.parametrize("abort", [False, True])
def test_interrupted_output_is_bounded(tmp_path, abort):
    script = tmp_path / "emit.py"
    script.write_text("import sys, time\nsys.stdout.write('x' * 200000)\nsys.stdout.flush()\ntime.sleep(30)\n")
    signal = SimpleNamespace(aborted=False)
    tool = create_bash_tool(str(tmp_path), shell_command=lambda command: ([sys.executable, str(script)], False))

    async def scenario():
        async def update(result):
            if abort:
                signal.aborted = True
        with pytest.raises(RuntimeError, match="aborted" if abort else "timed out") as error:
            await tool.execute("test", {"command": "unused", "timeout": 1}, signal, update)
        message = str(error.value)
        assert len(message) < DEFAULT_MAX_BYTES + 1000
        assert "Full output:" in message
        path = message.split("Full output: ", 1)[1].split("]", 1)[0]
        assert Path(path).read_bytes() == b"x" * 200000
        os.unlink(path)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["shell", "spawn"])
def test_startup_failure_cleans_log(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(bash_module.tempfile, "tempdir", str(tmp_path))

    def shell(command):
        if failure == "shell":
            raise FileNotFoundError("shell unavailable")
        return [str(tmp_path / "missing-executable")], False

    tool = create_bash_tool(str(tmp_path), shell_command=shell)
    with pytest.raises(FileNotFoundError):
        asyncio.run(tool.execute("test", {"command": "unused"}))
    assert not list(tmp_path.glob("pi-py-bash-*.log"))


def test_command_reuses_one_wait_task(tmp_path):
    script = tmp_path / "ticks.py"
    script.write_text("import time\nfor _ in range(12):\n print('tick', flush=True)\n time.sleep(0.1)\n")
    tool = create_bash_tool(str(tmp_path), shell_command=lambda command: ([sys.executable, str(script)], False))

    async def scenario():
        counts = []

        async def update(result):
            counts.append(sum(task.get_coro().__qualname__ == "Process.wait" for task in asyncio.all_tasks()))

        await tool.execute("test", {"command": "unused"}, on_update=update)
        assert len(counts) >= 3
        assert max(counts) <= 1
        assert not any(task.get_coro().__qualname__ == "Process.wait" for task in asyncio.all_tasks())

    asyncio.run(scenario())



@pytest.mark.parametrize("abort", [False, True])
def test_parent_exit_does_not_disable_interrupts(tmp_path, abort):
    script = tmp_path / "parent.py"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])\n"
    )
    tool = create_bash_tool(str(tmp_path), shell_command=lambda command: ([sys.executable, str(script)], False))
    signal = SimpleNamespace(aborted=False)

    async def scenario():
        loop = asyncio.get_running_loop()
        if abort:
            loop.call_later(0.5, setattr, signal, "aborted", True)
        start = loop.time()
        with pytest.raises(RuntimeError, match="aborted" if abort else "timed out"):
            await tool.execute("test", {"command": "unused", "timeout": 10 if abort else 0.5}, signal)
        assert loop.time() - start < 1.5

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [True, False, float("nan"), float("inf"), -float("inf"), 0, -1, "1", 10 ** 1000])
def test_invalid_timeout_rejected_before_start(tmp_path, timeout):
    def shell(command):
        pytest.fail("Invalid timeout must be rejected before launching a command")

    tool = create_bash_tool(str(tmp_path), shell_command=shell)
    with pytest.raises(ValueError, match="Invalid timeout"):
        asyncio.run(tool.execute("test", {"command": "unused", "timeout": timeout}))


@pytest.mark.parametrize("text", ["中", "😀", "a中😀"])
def test_structured_prefix_is_bounded_utf8(text):
    limit = bash_module.STRUCTURED_OUTPUT_MAX_BYTES
    data = (text * limit).encode("utf-8")
    output = OutputAccumulator()
    try:
        for offset in range(0, len(data), 4096):
            output.append(data[offset:offset + 4096])
        output.append(b"suffix")
        output.close()
        prefix, _, _ = output.snapshot()
        assert len(prefix.encode("utf-8")) <= limit
        assert prefix == data[:limit].decode("utf-8", errors="ignore")
        assert "\ufffd" not in prefix
    finally:
        output.close()
        os.unlink(output._log.name)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job lifetime")
@pytest.mark.parametrize("abort", [False, True])
def test_windows_interrupt_kills_orphan_descendants(tmp_path, abort):
    child = tmp_path / "child.py"
    child.write_text(
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "Path('pids').write_text(str(os.getpid()) + ' ' + str(grandchild.pid))\n"
        "time.sleep(30)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text("import subprocess, sys\nsubprocess.Popen([sys.executable, 'child.py'])\n")
    signal = SimpleNamespace(aborted=False)
    tool = create_bash_tool(str(tmp_path), shell_command=lambda command: ([sys.executable, str(parent)], False))

    async def scenario():
        async def interrupt():
            while not (tmp_path / "pids").exists():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.2)
            signal.aborted = True

        interrupt_task = asyncio.create_task(interrupt()) if abort else None
        try:
            with pytest.raises(RuntimeError, match="aborted" if abort else "timed out"):
                await asyncio.wait_for(tool.execute("test", {"command": "unused", "timeout": 10 if abort else 1}, signal), 5)
        finally:
            if interrupt_task is not None:
                interrupt_task.cancel()
                await asyncio.gather(interrupt_task, return_exceptions=True)

    asyncio.run(scenario())
    job = WindowsJob()
    try:
        for pid in map(int, (tmp_path / "pids").read_text().split()):
            handle = job._api.OpenProcess(0x00100000, False, pid)
            if handle:
                try:
                    job._api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                    assert job._api.WaitForSingleObject(handle, 1000) == 0
                finally:
                    job._api.CloseHandle(handle)
    finally:
        job.close()


def test_output_failure_terminates_command_immediately(tmp_path, monkeypatch):
    processes = []
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        return process

    failure = OSError("disk full")

    def append(self, data):
        raise failure

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(OutputAccumulator, "append", append)
    monkeypatch.setattr(bash_module.tempfile, "tempdir", str(tmp_path))
    script = tmp_path / "output.py"
    script.write_text("import time\nprint('output', flush=True)\ntime.sleep(30)\n")
    tool = create_bash_tool(str(tmp_path), shell_command=lambda command: ([sys.executable, str(script)], False))

    async def scenario():
        start = asyncio.get_running_loop().time()
        with pytest.raises(OSError, match="disk full") as error:
            await asyncio.wait_for(tool.execute("test", {"command": "unused"}), 3)
        assert error.value is failure
        assert asyncio.get_running_loop().time() - start < 2
        assert processes and all(process.returncode is not None for process in processes)
        assert not list(tmp_path.glob("pi-py-bash-*.log"))
        assert not any(task.get_coro().__qualname__ == "Process.wait" for task in asyncio.all_tasks())

    asyncio.run(scenario())

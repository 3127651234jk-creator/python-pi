"""Regression coverage for bounded shell output."""
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pi.coding_agent.tools import bash as bash_module
from pi.coding_agent.tools.bash import DEFAULT_MAX_BYTES, OutputAccumulator, create_bash_tool
from pi.coding_agent.tools.truncate import truncate_tail


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

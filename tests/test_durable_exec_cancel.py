import asyncio
import sys

import pytest

from pi.durable.env import Context, LocalExecutionEnv, ShellExecOptions


def test_cancel_exec_stops_output_and_reaps_process(tmp_path, monkeypatch):
    script = tmp_path / "emit.py"
    child = tmp_path / "child.py"
    child.write_text("import time\nwith open('heartbeat', 'ab', buffering=0) as f:\n while True:\n  f.write(b'x')\n  time.sleep(0.01)\n")
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "subprocess.Popen([sys.executable, 'child.py'])\n"
        "while not Path('heartbeat').exists(): time.sleep(0.01)\n"
        "while True:\n print('tick', flush=True)\n time.sleep(0.01)\n")
    processes = []
    spawn = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        if args[0] == sys.executable:
            processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)

    async def scenario():
        env = LocalExecutionEnv(str(tmp_path))
        output = []
        ready = asyncio.Event()

        def on_output(text, context, info):
            output.append(text)
            ready.set()

        task = asyncio.create_task(env.exec(
            [sys.executable, str(script)], ShellExecOptions(on_output=on_output), Context()))
        try:
            await asyncio.wait_for(ready.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            count = len(output)
            heartbeat_size = (tmp_path / "heartbeat").stat().st_size
            await asyncio.sleep(0.1)
            assert len(output) == count
            assert (tmp_path / "heartbeat").stat().st_size == heartbeat_size
            assert not env._active_children
            assert processes and all(p.returncode is not None for p in processes)
            assert not any("_CommandRunner._pump" in t.get_coro().__qualname__
                           or "_CommandRunner._watch_abort" in t.get_coro().__qualname__
                           for t in asyncio.all_tasks())
        finally:
            await env.cleanup(Context())

    asyncio.run(scenario())

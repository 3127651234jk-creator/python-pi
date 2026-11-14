import asyncio
import io
import threading
from types import SimpleNamespace

from pi.ai.models import Model
from pi.chord import StateEncoder
from pi.chord.provider import RemoteServiceProvider, ReplicatedState
from pi.coding_agent import cli, output_guard
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage


def test_protocol_stdout_uses_utf8(monkeypatch):
    data = io.BytesIO()
    stream = io.TextIOWrapper(data, encoding="gbk", newline="\n")
    monkeypatch.setattr("sys.stdout", stream)
    try:
        output_guard.take_over_stdout()
        output_guard.write_raw_stdout("\U0001f600\n")
        assert data.getvalue() == "\U0001f600\n".encode("utf-8")
    finally:
        output_guard.restore_stdout()


def test_closed_keyed_state_does_not_break_encoder():
    provider = RemoteServiceProvider([{"id": "test", "mode": "keyed"}])
    old = ReplicatedState(0)
    live = ReplicatedState(0)
    close = provider.spawn("test", "old", SimpleNamespace(state=old))
    provider.spawn("test", "live", SimpleNamespace(state=live))
    encoder = StateEncoder()
    subscription = provider.subscribe("test", "keyed", encoder.encode_update)
    encoder.encode_snapshot(subscription.snapshot)
    subscription.activate()
    close()
    old.replace(1)
    live.replace(1)
    assert not old._listeners
    provider.dispose()
    assert not live._listeners


def test_cancelled_console_read_does_not_block_loop_shutdown(monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def blocked_input(prompt):
        entered.set()
        release.wait(3)
        return ""

    monkeypatch.setattr("builtins.input", blocked_input)

    async def scenario():
        task = asyncio.create_task(cli.read_console_input())
        while not entered.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    try:
        asyncio.run(scenario())
        assert not release.is_set()
    finally:
        release.set()


def test_abort_interrupts_running_generation(tmp_path, monkeypatch):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "abort.jsonl")))
        harness = Harness(session, Model(id="faux", name="Faux", api="faux", provider="faux", base_url=""))
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_stream(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr("pi.durable.harness._stream_once", blocked_stream)
        cid = await harness.create_conversation()
        tid = await harness.submit_input(cid, "hello")
        running = asyncio.create_task(harness.run_pending_tasks())
        await asyncio.wait_for(entered.wait(), 2)
        await harness.abort_task(tid)
        await asyncio.wait_for(running, 2)
        assert cancelled.is_set()
        record = await harness._get_task(tid)
        assert record.state.outcome.status == "aborted"
        assert not harness._signals
        await session.close()

    asyncio.run(scenario())

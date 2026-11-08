"""Codemode host tests: pure helpers plus the real subprocess lifecycle.

Covers the source/identifier helpers, a real ``python -I`` round trip with
out-of-order replies and record backfill, snapshot publish timing, pending
cancellation on normal completion, timeout, abort, worker crash, a synthetic
bridge failure, and store/write plumbing.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.codemode import host
from pi.util import AbortError, AbortSignal


def wmeta(name, js_name=""):
    return {"name": name, "js_name": js_name, "namespace": "", "description": name}


async def run_host(code, tools=None, on_call=None, on_update=None, runtime=None,
                   timeout_ms=None, signal=None, store=None, root_id="call"):
    return await host.execute_codemode(
        code=code, root_id=root_id, tools_meta=tools or [wmeta("echo"), wmeta("slow"), wmeta("fast")],
        store=store or {}, timeout_ms=timeout_ms, signal=signal,
        on_call=on_call or _echo_call, on_update=on_update, runtime=runtime or host.prepare_runtime())


async def _echo_call(name, args, signal, record):
    record.id = "assigned/" + name
    return {"called": name, "args": args}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_codemode_source_without_options():
    assert host.parse_codemode_source("x = 1") == ({}, "x = 1")
    assert host.parse_codemode_source("# a comment\n#@options: {}\n") == ({}, "# a comment\n#@options: {}\n")


def test_parse_codemode_source_reads_first_line():
    options, rest = host.parse_codemode_source(
        '# @options: {"timeout_ms": 5, "max_output_tokens": 100}\n\ncode = 1')
    assert options == {"timeout_ms": 5, "max_output_tokens": 100}
    assert rest == "code = 1"

    options, rest = host.parse_codemode_source('#@options:{"timeout_ms": 7}\r\ncode = 2')
    assert options == {"timeout_ms": 7}
    assert rest == "code = 2"

    options, rest = host.parse_codemode_source('   # @options: {"timeout_ms": 1}\n\ncode = 3')
    assert options == {"timeout_ms": 1}
    assert rest == "code = 3"


@pytest.mark.parametrize("line", [
    '# @options: not json',
    '# @options: [1]',
    '#@options: {"unknown": 1}',
    '# @options: {"timeout_ms": -1}',
    '# @options: {"timeout_ms": 0}',
    '# @options: {"timeout_ms": "5"}',
    '# @options: {"timeout_ms": 1.5}',
    '# @options: {"timeout_ms": true}',
])
def test_parse_codemode_source_rejects(line):
    with pytest.raises(ValueError) as excinfo:
        host.parse_codemode_source(line + "\ncode")
    message = str(excinfo.value)
    assert message.startswith("codemode @options ")
    assert ("not valid JSON" in message) or ('supports "timeout_ms"' in message) \
        or ("must be a positive integer" in message)


def test_to_codemode_identifier_rules():
    assert host.to_codemode_identifier("echo") == "echo"
    assert host.to_codemode_identifier("mcp__dev__query") == "mcp__dev__query"
    assert host.to_codemode_identifier("a-b") == "a_b"
    assert host.to_codemode_identifier("with space") == "with_space"
    assert host.to_codemode_identifier("caf\u00e9") == "caf_"
    assert host.to_codemode_identifier("1abc") == "_1abc"
    assert host.to_codemode_identifier("9") == "_9"
    assert host.to_codemode_identifier("") == "_"
    assert host.to_codemode_identifier("___") == "___"


# ---------------------------------------------------------------------------
# Real subprocess lifecycle
# ---------------------------------------------------------------------------


def test_round_trip_with_out_of_order_replies_and_backfill():
    async def scenario():
        seen = []
        release = asyncio.Event()

        async def on_call(name, args, signal, record):
            seen.append((name, args, record.id))
            record.id = "assigned/" + name
            if name == "fast":
                release.set()
                return "fast-value"
            await release.wait()
            return "slow-value"

        runtime = host.prepare_runtime()
        code = (
            "text('starting')\n"
            "slow_handle = tools.slow.start({'n': 1})\n"
            "fast_handle = tools.fast.start({'n': 2})\n"
            "store('kept', [1, 2])\n"
            "store('drop', 1)\n"
            "store('drop')\n"
            "result = {'picked': wait_any(fast_handle, slow_handle), 'slow': wait(slow_handle),\n"
            "          'init': load('init'), 'kept': load('kept')}\n"
        )
        outcome = await run_host(code, runtime=runtime, on_call=on_call, store={"init": "v"})

        assert outcome.status == host.STATUS_OK
        assert outcome.value == {"picked": "fast-value", "slow": "slow-value",
                                 "init": "v", "kept": [1, 2]}
        assert seen == [("slow", {"n": 1}, "call/1"), ("fast", {"n": 2}, "call/2")]
        assert [record["id"] for record in outcome.calls] == ["assigned/slow", "assigned/fast"]
        assert all(record["status"] == "ok" for record in outcome.calls)
        assert all(isinstance(record["duration_ms"], int) for record in outcome.calls)
        assert all(record["error"] is None for record in outcome.calls)
        assert outcome.output_items == [{"type": "text", "text": "starting"}]
        assert outcome.writes == {"set": {"kept": [1, 2]}, "delete": ["drop"]}
        assert isinstance(outcome.wall_time, float) and outcome.wall_time > 0

        # Process reaped before the outcome is returned.
        assert outcome.exit_code is not None

    asyncio.run(scenario())


def test_snapshot_publish_timing_and_running_state():
    async def scenario():
        frames = []

        async def on_call(name, args, signal, record):
            record.id = "assigned/slow"
            await asyncio.sleep(0.05)
            return "done"

        def on_update(snapshot):
            frames.append(snapshot)

        runtime = host.prepare_runtime()
        code = "handle = tools.slow.start({})\nresult = wait(handle)\n"
        outcome = await run_host(code, runtime=runtime, on_call=on_call, on_update=on_update)

        assert outcome.status == host.STATUS_OK
        assert len(frames) >= 2
        first = frames[0]["calls"][0]
        assert first["id"] == "call/1"  # assigned before the executor runs
        assert first["status"] == "running"
        assert first["duration_ms"] is None
        last = frames[-1]["calls"][0]
        assert last["id"] == "assigned/slow"
        assert last["status"] == "ok"
        assert isinstance(last["duration_ms"], int)

    asyncio.run(scenario())


def test_async_update_listener_receives_frames():
    async def scenario():
        frames = []
        drained = asyncio.Event()

        async def on_update(snapshot):
            frames.append(snapshot)
            if len(frames) >= 2:
                drained.set()

        code = "result = tools.echo({})\n"
        await run_host(code, on_update=on_update)
        await asyncio.wait_for(drained.wait(), timeout=5)
        assert frames[0]["calls"][0]["status"] == "running"
        assert frames[-1]["calls"][0]["status"] == "ok"

    asyncio.run(scenario())


def test_normal_completion_cancels_pending_calls():
    async def scenario():
        cancelled = []
        started = asyncio.Event()
        never = asyncio.Event()

        async def on_call(name, args, signal, record):
            started.set()
            try:
                await never.wait()
                return "unreachable"
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        code = "tools.slow.start({})\ntext('leaving')\n"
        outcome = await run_host(code, on_call=on_call)

        assert outcome.status == host.STATUS_OK
        assert outcome.value is None
        assert [item["text"] for item in outcome.output_items] == ["leaving"]
        assert outcome.calls[0]["status"] == "cancelled"
        assert outcome.calls[0]["duration_ms"] is not None
        assert cancelled == ["slow"]  # the executor task was cancelled, not leaked

    asyncio.run(scenario())


def test_timeout_cancels_pending_and_reports():
    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        async def on_call(name, args, signal, record):
            started.set()
            await never.wait()
            return "unreachable"

        code = "result = wait(tools.slow.start({}))\n"
        outcome = await asyncio.wait_for(
            run_host(code, on_call=on_call, timeout_ms=300), timeout=15)

        assert outcome.status == host.STATUS_TIMEOUT
        assert outcome.value is None
        assert outcome.calls[0]["status"] == "cancelled"
        assert started.is_set()
        assert outcome.exit_code is not None  # worker was killed and reaped

    asyncio.run(scenario())


def test_abort_mid_script_cancels_calls():
    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        async def on_call(name, args, signal, record):
            started.set()
            assert signal is not None
            signal.abort()  # abort while the host loop is waiting
            await never.wait()
            return "unreachable"

        code = "result = wait(tools.slow.start({}))\n"
        outcome = await asyncio.wait_for(
            run_host(code, on_call=on_call, signal=AbortSignal()), timeout=15)

        assert outcome.status == host.STATUS_ABORTED
        assert outcome.calls[0]["status"] == "cancelled"

    asyncio.run(scenario())


def test_pre_aborted_signal_raises_before_script_runs():
    async def scenario():
        signal = AbortSignal()
        signal.abort()
        calls = []

        async def on_call(name, args, sig, record):
            calls.append(name)
            return None

        with pytest.raises(AbortError):
            await run_host("result = 1\n", on_call=on_call, signal=signal)
        assert calls == []

    asyncio.run(scenario())


def test_worker_crash_is_classified_with_exit_code():
    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        async def on_call(name, args, signal, record):
            started.set()
            await never.wait()
            return "unreachable"

        code = "text('before')\ntools.slow.start({})\nimport os\nos._exit(7)\n"
        outcome = await asyncio.wait_for(run_host(code, on_call=on_call), timeout=15)

        assert outcome.status == host.STATUS_CRASHED
        assert outcome.exit_code == 7
        assert outcome.value is None
        assert outcome.error is None
        assert [item["text"] for item in outcome.output_items] == ["before"]
        assert outcome.calls[0]["status"] == "cancelled"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Bridge failure (synthetic: pump raises while the host waits)
# ---------------------------------------------------------------------------


class FakeStream:
    def __init__(self, script=None):
        self.script = list(script or [])

    def write(self, data):
        pass

    async def drain(self):
        pass

    async def readline(self):
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return b""


class FakeProcess:
    def __init__(self):
        self.stdin = FakeStream()
        self.stdout = FakeStream([RuntimeError("pipe exploded")])
        self.returncode = None
        self.killed = False

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = 3

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_bridge_failure_is_reported_and_everything_is_reaped(monkeypatch):
    async def scenario():
        fake = FakeProcess()

        async def fake_exec(*args, **kwargs):
            return fake

        calls = []

        async def on_call(name, args, signal, record):
            calls.append(name)
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        outcome = await run_host("result = 1\n", on_call=on_call)

        assert outcome.status == host.STATUS_SCRIPT_ERROR
        assert outcome.error["name"] == "SandboxError"
        assert "codemode bridge failed" in outcome.error["message"]
        assert "RuntimeError" in outcome.error["message"]
        assert calls == []  # the failure happened before any call
        assert fake.killed
        assert fake.returncode is not None

    asyncio.run(scenario())


def test_bridge_failure_after_calls_still_reaps_pending(monkeypatch):
    async def scenario():
        release = asyncio.Event()

        class DelayedProcess(FakeProcess):
            def __init__(self):
                super().__init__()
                # First read: a call message; second: the pipe explodes.
                call_line = (json.dumps({"type": "call", "id": 1, "tool": "slow",
                                         "args": {}}) + "\n").encode()
                self.stdout = FakeStream([call_line, RuntimeError("pipe exploded")])

        fake = DelayedProcess()

        async def fake_exec(*args, **kwargs):
            return fake

        async def on_call(name, args, signal, record):
            await release.wait()
            return "unreachable"

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        outcome = await run_host("tools.slow.start({})\n", on_call=on_call)

        assert outcome.status == host.STATUS_SCRIPT_ERROR
        assert outcome.error["name"] == "SandboxError"
        assert outcome.calls[0]["status"] == "cancelled"
        assert fake.killed

    asyncio.run(scenario())

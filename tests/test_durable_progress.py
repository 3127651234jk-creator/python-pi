"""Progress policy, adaptive throttling, and waiter settlement (P01-P07)."""

from __future__ import annotations

import asyncio
import math

import pytest

from pi.durable.output import Progress
from pi.durable.types import ProgressPolicy, resolve_progress_policy


def test_p01_defaults_and_field_by_field_resolution():
    assert resolve_progress_policy(None) == ProgressPolicy(100, 100)
    assert resolve_progress_policy({}) == ProgressPolicy(100, 100)
    assert resolve_progress_policy({"partial_interval_ms": 0}) == ProgressPolicy(0, 100)
    assert resolve_progress_policy({"output_interval_ms": 50}) == ProgressPolicy(100, 50)
    assert resolve_progress_policy({"partial_interval_ms": None}) == ProgressPolicy(100, 100)
    assert resolve_progress_policy({"partialIntervalMs": 25, "outputIntervalMs": 0}) == ProgressPolicy(25, 0)


@pytest.mark.parametrize("value", [
    {"partial_interval_ms": -1},
    {"partial_interval_ms": float("nan")},
    {"partial_interval_ms": float("inf")},
    {"output_interval_ms": True},
    {"output_interval_ms": "100"},
    {"output_interval_ms": []},
    [],
])
def test_p02_invalid_policies_fail_explicitly(value):
    with pytest.raises(ValueError, match="progress"):
        resolve_progress_policy(value)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Handle:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeScheduler:
    def __init__(self, clock):
        self.clock = clock
        self.entries = []

    def __call__(self, delay, callback):
        handle = Handle()
        self.entries.append([self.clock.now + delay, callback, handle])
        return handle

    def next_delay(self):
        active = [entry for entry in self.entries if not entry[2].cancelled]
        return min(entry[0] - self.clock.now for entry in active) if active else None

    def run_due(self):
        due = [entry for entry in self.entries if not entry[2].cancelled and entry[0] <= self.clock.now]
        self.entries = [entry for entry in self.entries
                        if not (not entry[2].cancelled and entry[0] <= self.clock.now)]
        for _at, callback, _handle in due:
            callback()


async def settle():
    for _ in range(6):
        await asyncio.sleep(0)


def test_p03_first_mark_immediate_and_writes_coalesce():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock
        started = []
        release = asyncio.Event()

        async def write():
            started.append(clock.now)
            if len(started) == 1:
                await release.wait()
            return 0

        progress = Progress(write, lambda exc: None, 100, clock=clock, schedule=scheduler)
        progress.mark()
        await settle()
        assert started == [1000.0]

        progress.mark()
        progress.mark()
        await settle()
        assert started == [1000.0]  # one commit in flight, changes coalesce

        release.set()
        await settle()
        assert progress._in_flight is None
        assert scheduler.next_delay() == pytest.approx(0.1)
        clock.advance(0.1)
        scheduler.run_due()
        await settle()
        assert len(started) == 2

    asyncio.run(scenario())


def test_p04_adaptive_interval_uses_written_utf8_bytes():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock

        async def write():
            return 204800

        progress = Progress(write, lambda exc: None, 100, clock=clock, schedule=scheduler)
        progress.mark()
        await settle()
        progress.mark()
        assert scheduler.next_delay() == pytest.approx(2.0)

    asyncio.run(scenario())


def test_p05_mark_and_wait_is_not_settled_by_an_earlier_commit():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock
        started = []
        release = asyncio.Event()

        async def write():
            started.append(clock.now)
            if len(started) == 1:
                await release.wait()
            return 0

        progress = Progress(write, lambda exc: None, 100, clock=clock, schedule=scheduler)
        progress.mark()
        await settle()
        waiter = progress.mark_and_wait()
        assert not waiter.done()
        release.set()
        await settle()
        assert not waiter.done()  # the first commit did not include this change
        assert scheduler.next_delay() == pytest.approx(0.1)
        clock.advance(0.1)
        scheduler.run_due()
        await settle()
        assert waiter.done() and waiter.exception() is None

    asyncio.run(scenario())


def test_p07_write_failure_rejects_waiter_and_reports():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock
        errors = []

        async def write():
            raise RuntimeError("storage unavailable")

        progress = Progress(write, errors.append, 100, clock=clock, schedule=scheduler)
        waiter = progress.mark_and_wait()
        await settle()
        assert waiter.done()
        assert isinstance(waiter.exception(), RuntimeError)
        assert errors and isinstance(errors[0], RuntimeError)

    asyncio.run(scenario())


def test_p06_stop_waits_for_in_flight_and_returns_waiters():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock
        release = asyncio.Event()
        started = []

        async def write():
            started.append(clock.now)
            await release.wait()
            return 0

        progress = Progress(write, lambda exc: None, 100, clock=clock, schedule=scheduler)
        progress.mark()
        await settle()
        waiter = progress.mark_and_wait()
        stopping = asyncio.create_task(progress.stop())
        await settle()
        assert not stopping.done()
        release.set()
        pending = await stopping
        assert pending == [waiter]
        assert not waiter.done()

        # Nothing is scheduled or written after stop.
        count = len(started)
        progress.mark()
        await settle()
        assert len(started) == count
        assert scheduler.next_delay() is None
        waiter.set_result(None)

    asyncio.run(scenario())


def test_p06_stop_cancels_a_pending_timer():
    async def scenario():
        clock, scheduler = FakeClock(), FakeScheduler(FakeClock())
        scheduler.clock = clock
        calls = []

        async def write():
            calls.append(1)
            return 0

        progress = Progress(write, lambda exc: None, 100, clock=clock, schedule=scheduler)
        progress.mark()
        await settle()
        progress.mark()
        assert scheduler.next_delay() == pytest.approx(0.1)
        await progress.stop()
        assert scheduler.next_delay() is None

    asyncio.run(scenario())


def test_policy_rejects_non_finite_even_when_math_imported():
    # Guard against a regression that compares NaN with < 0 (always false).
    with pytest.raises(ValueError):
        resolve_progress_policy({"partial_interval_ms": math.nan})

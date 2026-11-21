"""Session shutdown must drain work even when a caller fails or cancels."""

import asyncio

import pytest

from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage


def test_failing_close_listener_does_not_fail_the_close():
    # A close listener that throws is reported and dropped; it must not fail
    # the close or leave later listeners unrun.
    async def scenario():
        storage = MemoryStorage()
        session = create_session(storage)
        notified = []
        reported = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: reported.append(context))

        def fail():
            raise RuntimeError("listener failed")

        session.subscribe_close(fail)
        session.subscribe_close(lambda: notified.append(True))
        await session.close()
        assert storage.closed
        assert notified == [True]
        assert any(context.get("exception") is not None for context in reported)
        # A repeated close resolves instead of re-raising the listener error.
        await asyncio.wait_for(session.close(), 1)
    asyncio.run(scenario())


def test_cancelling_close_waiter_does_not_cancel_shutdown():
    async def scenario():
        storage = MemoryStorage()
        session = create_session(storage)
        started, release = asyncio.Event(), asyncio.Event()

        async def change(tx):
            started.set()
            await release.wait()
            await tx.create_conversation({"kind": "ownerless"})

        commit = asyncio.create_task(session.commit(change))
        await started.wait()
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not storage.closed
        release.set()
        await commit
        await asyncio.wait_for(session.close(), 1)
        assert storage.closed
    asyncio.run(scenario())


def test_throwing_session_clock_falls_back_to_wall_time():
    # A broken host clock must not stop commits (TS safeNow / Session #clock).
    async def scenario():
        def broken_clock():
            raise RuntimeError("clock broken")

        session = create_session(MemoryStorage(), now=broken_clock)
        value = session.now()
        assert isinstance(value, int) and value > 0

        async def change(tx):
            await tx.create_conversation({"kind": "ownerless"})

        await session.commit(change)
        await session.close()
    asyncio.run(scenario())


def test_throwing_harness_clock_reports_once_and_falls_back():
    from pi.ai.models import Model
    from pi.durable.harness import Harness
    from pi.durable.session import create_session as _create

    async def scenario():
        session = _create(MemoryStorage())
        reports = []

        def broken_clock():
            raise RuntimeError("clock broken")

        harness = Harness(session, Model(id="faux", name="Faux", api="faux", provider="faux",
                                         base_url=""),
                          on_report=reports.append, now=broken_clock)
        assert isinstance(harness.now(), int) and harness.now() > 0
        assert len(reports) == 1  # the first throw is reported, later ones are not
        await session.close()
    asyncio.run(scenario())

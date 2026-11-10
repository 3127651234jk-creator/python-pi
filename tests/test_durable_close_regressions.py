"""Session shutdown must drain work even when a caller fails or cancels."""

import asyncio

import pytest

from pi.durable.session import create_session
from pi.durable.storage import MemoryStorage


def test_failing_close_listener_does_not_leave_session_hanging():
    async def scenario():
        storage = MemoryStorage()
        session = create_session(storage)
        notified = []

        def fail():
            raise RuntimeError("listener failed")

        session.subscribe_close(fail)
        session.subscribe_close(lambda: notified.append(True))
        with pytest.raises(RuntimeError, match="listener failed"):
            await session.close()
        assert storage.closed
        assert notified == [True]
        with pytest.raises(RuntimeError, match="listener failed"):
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

"""Regression tests for cancellation and payload boundaries."""

import asyncio

import pytest

from pi.coding_agent.oauth_callback import start_oauth_callback_server
from pi.util import EventStream


def test_none_event_is_delivered_before_completion():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        stream.push(None)
        stream.push("next")
        stream.end("result")
        assert [event async for event in stream] == [None, "next"]
        assert await stream.result() == "result"

    asyncio.run(scenario())


def test_cancelled_oauth_waiter_does_not_cancel_other_waiters():
    async def scenario():
        server = await start_oauth_callback_server("test", "127.0.0.1", 0, "/callback")
        try:
            first = asyncio.create_task(server.wait())
            second = asyncio.create_task(server.wait())
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            server.cancel()
            assert await second is None
            assert await server.wait() is None
        finally:
            await server.close()

    asyncio.run(scenario())

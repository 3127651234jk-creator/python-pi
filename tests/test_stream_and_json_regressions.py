"""Regression coverage for stream completion and message persistence."""

import asyncio

import pytest

from pi.ai.json_io import message_from_json, to_json
from pi.ai.types import SystemMessage, TextContent, ToolResultMessage, UserMessage, get_system_message_text
from pi.util import EventStream


def test_system_text_blocks_roundtrip():
    message = SystemMessage(content=[TextContent(text="instructions", text_signature="signed")], timestamp=1)
    restored = message_from_json(to_json(message))
    assert restored == message
    assert get_system_message_text(restored) == "instructions"


def test_terminal_event_completes_stream():
    async def scenario():
        stream = EventStream(lambda event: event == "done", lambda event: "result")
        stream.push("data")
        stream.push("done")
        assert await asyncio.wait_for(stream.result(), 0.1) == "result"
        assert [event async for event in stream] == ["data", "done"]
    asyncio.run(scenario())


def test_result_waiters_do_not_consume_events():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        first = asyncio.create_task(stream.result())
        second = asyncio.create_task(stream.result())
        await asyncio.sleep(0)
        stream.push("data")
        stream.end("result")
        assert await asyncio.wait_for(asyncio.gather(first, second), 0.1) == ["result", "result"]
        assert [event async for event in stream] == ["data"]
    asyncio.run(scenario())


def test_exhausted_stream_stays_exhausted():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        stream.end("result")
        for _ in range(2):
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(stream.__anext__(), 0.1)
    asyncio.run(scenario())


def test_none_is_a_valid_stream_result():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        stream.end(None)
        assert await stream.result() is None
    asyncio.run(scenario())


def test_cancelled_result_waiter_does_not_cancel_completion():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        waiter = asyncio.create_task(stream.result())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        stream.push("data")
        stream.end("result")
        assert await stream.result() == "result"
        assert [event async for event in stream] == ["data"]
    asyncio.run(scenario())


def test_end_releases_all_iterator_waiters():
    async def scenario():
        stream = EventStream(lambda event: False, lambda event: event)
        waiters = [asyncio.create_task(stream.__anext__()) for _ in range(2)]
        await asyncio.sleep(0)
        stream.end("result")
        results = await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), 0.1)
        assert all(isinstance(result, StopAsyncIteration) for result in results)
    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["user", "toolResult"])
def test_text_signatures_survive_message_roundtrip(role):
    content = [TextContent(text="signed text", text_signature="signature")]
    if role == "user":
        message = UserMessage(content=content, timestamp=1)
    else:
        message = ToolResultMessage(tool_call_id="call", tool_name="read", content=content,
                                    is_error=False, timestamp=1)
    assert message_from_json(to_json(message)) == message

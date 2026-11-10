"""Tool progress must reach listeners before execution completes."""

import asyncio

import pytest

from pi.agent.agent_loop import execute_prepared_tool_call, _maybe_await
from pi.agent.types import AgentTool, AgentToolResult
from pi.ai.types import TextContent, ToolCall


def test_progress_is_delivered_while_tool_is_running():
    async def scenario():
        delivered = asyncio.Event()

        async def update(result):
            delivered.set()

        async def execute(call_id, args, signal, on_update):
            await on_update(AgentToolResult(content=[TextContent(text="working")]))
            await asyncio.wait_for(delivered.wait(), 0.5)
            return AgentToolResult(content=[TextContent(text="done")])

        result = await execute_prepared_tool_call(
            {"tool": AgentTool(name="progress", description="progress", execute=execute),
             "tool_call": ToolCall(id="call", name="progress", arguments={}), "args": {}},
            None, update,
        )
        assert not result["is_error"]
        assert result["result"].content[0].text == "done"

    asyncio.run(scenario())


def test_callback_future_is_awaited():
    async def scenario():
        future = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(future.set_result, "resolved")
        assert await _maybe_await(future) == "resolved"

    asyncio.run(scenario())


def test_cancellation_stops_pending_progress_callbacks():
    async def scenario():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def update(result):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def execute(call_id, args, signal, on_update):
            await on_update(AgentToolResult(content=[]))
            await asyncio.Event().wait()

        task = asyncio.create_task(execute_prepared_tool_call(
            {"tool": AgentTool(name="progress", description="progress", execute=execute),
             "tool_call": ToolCall(id="call", name="progress", arguments={}), "args": {}},
            None, update,
        ))
        try:
            await asyncio.wait_for(started.wait(), 1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert stopped.is_set()

    asyncio.run(scenario())

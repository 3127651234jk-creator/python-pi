"""Nested tool-call recording, pipeline reuse and usage aggregation tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.agent.agent_loop import default_convert_to_llm, run_agent_loop
from pi.agent.types import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi.ai.models import Model
from pi.ai.providers import faux
from pi.ai.providers.pi_messages import wire_message
from pi.ai.types import (
    NestedToolCallRecord,
    NestedToolCalls,
    TextContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi.codemode import create_codemode_tool
from pi.coding_agent.nested_tool_calls import (
    NestedCallContext,
    NestedCallRecorder,
    NestedToolCallHost,
    NestedToolCallOutcome,
    NestedToolCallRunner,
    execute_tool,
    reset_current_call,
    set_current_call,
)
from pi.coding_agent.session import SessionManager
from pi.coding_agent.usage import session_stats, usage_cost_breakdown


@pytest.fixture(autouse=True)
def faux_script_env():
    yield
    os.environ.pop("FAUX_SCRIPT", None)
    faux.reset_script_index()


def use_script(script):
    os.environ["FAUX_SCRIPT"] = json.dumps(script)
    faux.reset_script_index()


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


async def stream_fn(model, context, options=None):
    async for event in faux.stream_faux(model, context, options):
        yield event


def usage(input_tokens: int = 0) -> Usage:
    return Usage(input=input_tokens)


def leaf_tool(name: str = "leaf", own_usage: int = 0, executions=None,
              mode: str = None) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        if executions is not None:
            executions.append(params)
        return AgentToolResult(content=[TextContent(text=str(params.get("text", "leaf")))],
                               usage=usage(own_usage) if own_usage else None)

    return AgentTool(
        name=name,
        label=name,
        description="leaf tool",
        parameters={"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]},
        execute=execute,
        execution_mode=mode,
    )


def outer_tool(child: str, own_usage: int = 0, calls=None,
               mode: str = None) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        outcome = await execute_tool(child, {"text": params.get("text", "x")}, signal)
        if calls is not None:
            calls.append(outcome.tool_call_id)
        text = "\n".join(block.text for block in outcome.result.content
                         if isinstance(block, TextContent))
        return AgentToolResult(content=[TextContent(text=text)],
                               usage=usage(own_usage) if own_usage else None)

    return AgentTool(
        name="outer",
        label="outer",
        description="outer tool",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
        execution_mode=mode,
    )


async def run_once(tools, tool_calls, config_kwargs=None, events=None):
    """Run one model-issued batch and return transcript plus emitted events."""
    use_script([
        {"text": "", "toolCalls": tool_calls},
        {"text": "done"},
    ])
    collected = events if events is not None else []

    async def emit(event):
        collected.append(event)

    return await run_agent_loop(
        prompts=[UserMessage(content="go", timestamp=0)],
        context=AgentContext(messages=[], tools=tools),
        config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm,
                               **(config_kwargs or {})),
        signal=None,
        emit=emit,
        stream_fn=stream_fn,
    )


def tool_results(messages):
    return [message for message in messages if isinstance(message, ToolResultMessage)]


# ---------------------------------------------------------------------------
# A -> B -> C
# ---------------------------------------------------------------------------


def test_nested_ids_parents_and_single_root_result():
    async def scenario():
        async def mid_execute(tool_call_id, params, signal=None, on_update=None):
            outcome = await execute_tool("leaf", {"text": params.get("text", "x")}, signal)
            text = "\n".join(block.text for block in outcome.result.content
                             if isinstance(block, TextContent))
            return AgentToolResult(content=[TextContent(text=text)], usage=usage(10))

        mid = AgentTool(name="mid", label="mid", description="mid", parameters={
            "type": "object", "properties": {"text": {"type": "string"}}}, execute=mid_execute)
        tools = [outer_tool("mid", own_usage=5), mid, leaf_tool(own_usage=20)]

        messages = await run_once(tools, [
            {"id": "t1", "name": "outer", "arguments": {"text": "hi"}},
        ])

        results = tool_results(messages)
        assert len(results) == 1
        result = results[0]
        assert result.tool_call_id == "t1"
        assert result.nested_calls is not None
        assert result.nested_calls.complete is True
        assert [record.id for record in result.nested_calls.calls] == ["t1/1", "t1/1/1"]
        assert [record.parent_tool_call_id for record in result.nested_calls.calls] == ["t1", "t1/1"]
        assert [record.name for record in result.nested_calls.calls] == ["mid", "leaf"]
        # Own 5 + mid 10 + leaf 20, counted once.
        assert result.usage.input == 35
        assert result.usage.total_tokens == 35

        totals = usage_cost_breakdown(messages)
        tool_totals = totals.by_model["Tools/summaries"]
        assert tool_totals.input == 35
        assert tool_totals.requests == 1
        assert totals.requests == 3  # two assistant turns plus one tool result
        stats = session_stats(messages)
        assert stats["tool_calls"] == 1
        assert stats["nested_tool_calls"] == 2
        assert stats["nested_tool_calls_complete"] is True

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Failures still use the normal pipeline
# ---------------------------------------------------------------------------


def test_nested_invalid_arguments_and_unknown_tool():
    async def scenario():
        executions = []
        leaf = leaf_tool(executions=executions)

        async def outer_execute(tool_call_id, params, signal=None, on_update=None):
            bad = await execute_tool("leaf", {"wrong": 1}, signal)
            missing = await execute_tool("missing", {}, signal)
            assert bad.is_error and missing.is_error
            return AgentToolResult(content=[TextContent(
                text="\n".join(block.text for block in bad.result.content
                                if isinstance(block, TextContent))
            )])

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=outer_execute)
        events = []
        messages = await run_once([leaf, outer], [
            {"id": "t1", "name": "outer", "arguments": {}},
        ], events=events)

        assert executions == []
        result = tool_results(messages)[0]
        assert result.nested_calls is not None
        records = result.nested_calls.calls
        assert [record.status for record in records] == ["error", "error"]
        assert "missing required property 'text'" in records[0].error
        assert "not found" in records[1].error
        starts = [event for event in events if isinstance(event, ToolExecutionStartEvent)]
        ends = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
        assert [event.tool_call_id for event in starts] == ["t1", "t1/1", "t1/2"]
        assert [event.tool_call_id for event in ends if event.tool_call_id != "t1"] == ["t1/1", "t1/2"]
        assert starts[1].parent_tool_call_id == "t1"
        assert ends[1].is_error is True

    asyncio.run(scenario())


def test_nested_permission_block_runs_no_executor():
    async def scenario():
        executions = []
        leaf = leaf_tool(executions=executions)
        seen = []

        async def before(context, signal=None):
            seen.append((context.tool_name, context.parent_tool_call_id))
            if context.tool_name == "leaf":
                return BeforeToolCallResult(block=True, reason="not allowed")
            return None

        async def execute_outer(tool_call_id, params, signal=None, on_update=None):
            return (await execute_tool("leaf", {"text": "x"}, signal)).result

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=execute_outer)
        messages = await run_once([leaf, outer], [
            {"id": "t1", "name": "outer", "arguments": {}},
        ], config_kwargs={"before_tool_call": before})

        assert executions == []
        result = tool_results(messages)[0]
        assert result.nested_calls.calls[0].status == "error"
        assert result.nested_calls.calls[0].error == "not allowed"
        assert ("outer", None) in seen
        assert ("leaf", "t1") in seen

    asyncio.run(scenario())


def test_hook_modified_usage_counted_once():
    async def scenario():
        leaf = leaf_tool(own_usage=1)

        async def after(context, signal=None):
            if context.tool_name == "leaf":
                return AfterToolCallResult(usage=usage(7))
            return None

        async def execute_outer(tool_call_id, params, signal=None, on_update=None):
            outcome = await execute_tool("leaf", {"text": "x"}, signal)
            # The caller reports only its own usage, never the child's.
            return AgentToolResult(content=outcome.result.content)

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=execute_outer)
        messages = await run_once([leaf, outer], [
            {"id": "t1", "name": "outer", "arguments": {}},
        ], config_kwargs={"after_tool_call": after})
        result = tool_results(messages)[0]
        assert result.usage.input == 7
        assert result.nested_calls.calls[0].status == "ok"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_record_limits_drop_and_incomplete():
    recorder = NestedCallRecorder()
    for index in range(257):
        recorder.start(f"r/{index + 1}", "r", "leaf", {"index": index})
    snapshot = recorder.snapshot()
    assert len(snapshot.calls) == 256
    assert snapshot.complete is False

    large = NestedCallRecorder()
    arguments = {"text": "\u4e2d" * 3000}
    record = large.start("r/1", "r", "leaf", arguments)
    expected = len(json.dumps(arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert record.arguments is None
    assert record.arguments_bytes == expected
    assert large.snapshot().complete is False

    small = NestedCallRecorder()
    small_record = small.start("r/1", "r", "leaf", {"text": "ok"})
    assert small_record.arguments == {"text": "ok"}
    small.finish(small_record, "ok")
    assert small.snapshot().complete is True


def test_all_dropped_call_usage_still_counted():
    async def scenario():
        async def outer_execute(tool_call_id, params, signal=None, on_update=None):
            for _ in range(257):
                await execute_tool("leaf", {"text": "x"}, signal)
            return AgentToolResult(content=[TextContent(text="done")])

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=outer_execute)
        messages = await run_once([leaf_tool(own_usage=1), outer], [
            {"id": "t1", "name": "outer", "arguments": {}},
        ])
        result = tool_results(messages)[0]
        assert len(result.nested_calls.calls) == 256
        assert result.nested_calls.complete is False
        assert result.usage.input == 257

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_parallel_siblings_and_roots_do_not_share_usage():
    async def scenario():
        async def outer_execute(tool_call_id, params, signal=None, on_update=None):
            first, second = await asyncio.gather(
                execute_tool("leaf", {"text": "a"}, signal),
                execute_tool("leaf", {"text": "b"}, signal),
            )
            return AgentToolResult(content=[TextContent(
                text=first.result.content[0].text + second.result.content[0].text
            )])

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=outer_execute)
        tools = [leaf_tool(own_usage=10), outer]
        messages = await run_once(tools, [
            {"id": "t1", "name": "outer", "arguments": {}},
            {"id": "t2", "name": "outer", "arguments": {}},
        ])

        results = {result.tool_call_id: result for result in tool_results(messages)}
        assert set(results) == {"t1", "t2"}
        for result in results.values():
            assert result.usage.input == 20
            calls = result.nested_calls.calls
            assert [call.id for call in calls] == [f"{result.tool_call_id}/1", f"{result.tool_call_id}/2"]
            assert [call.parent_tool_call_id for call in calls] == [result.tool_call_id] * 2

        totals = usage_cost_breakdown(messages)
        tool_totals = totals.by_model["Tools/summaries"]
        assert tool_totals.input == 40
        assert tool_totals.requests == 2

    asyncio.run(scenario())


def test_sequential_mode_does_not_deadlock():
    async def scenario():
        async def outer_execute(tool_call_id, params, signal=None, on_update=None):
            outcome = await execute_tool("leaf", {"text": "x"}, signal)
            return AgentToolResult(content=outcome.result.content)

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=outer_execute)
        messages = await asyncio.wait_for(
            run_once([leaf_tool(own_usage=3, mode="sequential"), outer], [
                {"id": "t1", "name": "outer", "arguments": {}},
            ], config_kwargs={"tool_execution": "sequential"}),
            timeout=10,
        )
        result = tool_results(messages)[0]
        assert result.nested_calls.calls[0].id == "t1/1"
        assert result.usage.input == 3

    asyncio.run(scenario())


class _QueuedHost(NestedToolCallHost):
    """Runs calls only after the test releases them; every call is exclusive."""

    def __init__(self) -> None:
        self.events = []
        self.release = asyncio.Event()

    def get_tools(self):
        async def execute(*args, **kwargs):
            return AgentToolResult(content=[])

        return [AgentTool(name="slow", label="slow", description="slow",
                          parameters={"type": "object"}, execute=execute,
                          execution_mode="sequential")]

    def is_sequential(self):
        return True

    async def run_tool_call(self, tool_call, parent_tool_call_id, signal, on_update):
        await self.release.wait()
        return {"tool_call": tool_call,
                "result": AgentToolResult(content=[TextContent(text="ok")]), "is_error": False}

    async def emit(self, event):
        self.events.append(event)


def test_cancel_while_waiting_for_exclusive_queue():
    async def scenario():
        host = _QueuedHost()
        runner = NestedToolCallRunner(host)
        context = NestedCallContext(
            root_id="r", caller_id="r", assistant_message=None,
            context=AgentContext(messages=[]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None, holds_queue=False, runner=runner,
        )
        token = set_current_call(context)
        try:
            first = asyncio.create_task(runner.execute("r", "slow", {}))
            await asyncio.sleep(0)
            second = asyncio.create_task(runner.execute("r", "slow", {}))
            await asyncio.sleep(0)
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)

            host.release.set()
            third = await runner.execute("r", "slow", {})
            await first
            assert third.tool_call_id == "r/3"

            summary = runner.take_record("r")
            assert [call.id for call in summary.calls.calls] == ["r/1", "r/2", "r/3"]
            assert summary.calls.calls[1].status == "cancelled"
            assert summary.calls.calls[2].status == "ok"
            starts = [event.tool_call_id for event in host.events
                      if isinstance(event, ToolExecutionStartEvent)]
            ends = [event.tool_call_id for event in host.events
                    if isinstance(event, ToolExecutionEndEvent)]
            assert starts == ["r/1", "r/2", "r/3"]
            assert sorted(ends) == ["r/1", "r/2", "r/3"]
            assert len(set(ends)) == len(ends)
        finally:
            reset_current_call(token)
            await runner.close()

    asyncio.run(scenario())


def test_close_cancels_in_flight_calls():
    async def scenario():
        host = _QueuedHost()
        runner = NestedToolCallRunner(host)
        context = NestedCallContext(
            root_id="r", caller_id="r", assistant_message=None,
            context=AgentContext(messages=[]),
            config=AgentLoopConfig(model=make_model(), convert_to_llm=default_convert_to_llm),
            signal=None, holds_queue=False, runner=runner,
        )
        token = set_current_call(context)
        try:
            task = asyncio.create_task(runner.execute("r", "slow", {}))
            await asyncio.sleep(0)
            await runner.close()
            await asyncio.gather(task, return_exceptions=True)
            assert task.cancelled()
            ends = [event for event in host.events if isinstance(event, ToolExecutionEndEvent)]
            assert len(ends) == 1 and ends[0].is_error is True
            assert runner.take_record("r") is None
        finally:
            reset_current_call(token)

    asyncio.run(scenario())


def test_root_return_cancels_detached_nested_calls():
    async def scenario():
        started = asyncio.Event()
        never = asyncio.Event()

        async def slow_execute(tool_call_id, params, signal=None, on_update=None):
            started.set()
            await never.wait()
            return AgentToolResult(content=[TextContent(text="unreachable")])

        slow = AgentTool(name="slow", label="slow", description="slow", parameters={
            "type": "object", "properties": {}}, execute=slow_execute)

        async def outer_execute(tool_call_id, params, signal=None, on_update=None):
            asyncio.ensure_future(execute_tool("slow", {}, signal))
            await started.wait()
            return AgentToolResult(content=[TextContent(text="outer done")])

        outer = AgentTool(name="outer", label="outer", description="outer", parameters={
            "type": "object", "properties": {}}, execute=outer_execute)
        messages = await asyncio.wait_for(run_once([slow, outer], [
            {"id": "t1", "name": "outer", "arguments": {}},
        ]), timeout=10)

        result = tool_results(messages)[0]
        assert result.content[0].text == "outer done"
        assert result.nested_calls.calls[0].status == "cancelled"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Codemode
# ---------------------------------------------------------------------------


def test_codemode_uses_outcome_and_records_real_id():
    async def scenario():
        seen = []

        async def fake_execute_tool(name, args, signal):
            seen.append((name, args))
            return NestedToolCallOutcome(
                tool_call_id="c/1",
                result=AgentToolResult(content=[TextContent(text="hello")]),
                is_error=False,
            )

        tool = create_codemode_tool([leaf_tool()], execute_tool=fake_execute_tool)
        result = await tool.execute("c", {"code": "text(tools.leaf({'text': 'x'}))"}, None, None)
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        assert result.is_error is False
        assert "hello" in text
        assert seen == [("leaf", {"text": "x"})]
        assert result.details["calls"][0]["id"] == "c/1"
        assert result.details["calls"][0]["status"] == "ok"

    asyncio.run(scenario())


def test_codemode_error_keeps_script_semantics():
    async def scenario():
        async def failing_execute_tool(name, args, signal):
            return NestedToolCallOutcome(
                tool_call_id="c/1",
                result=AgentToolResult(content=[TextContent(text="boom")], is_error=True),
                is_error=True,
            )

        tool = create_codemode_tool([leaf_tool()], execute_tool=failing_execute_tool)
        result = await tool.execute("c", {"code": "tools.leaf({'text': 'x'})"}, None, None)
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        assert result.is_error is True
        assert "boom" in text

    asyncio.run(scenario())


def test_codemode_through_agent_loop_records_nested_call():
    async def scenario():
        from pi.coding_agent.nested_tool_calls import execute_tool as unified_execute_tool

        leaf = leaf_tool(own_usage=10)
        codemode = create_codemode_tool([leaf], execute_tool=unified_execute_tool)
        messages = await run_once([leaf, codemode], [
            {"id": "t1", "name": "codemode",
             "arguments": {"code": "text(tools.leaf({'text': 'hi'}))"}},
        ])
        result = tool_results(messages)[0]
        assert result.tool_name == "codemode"
        assert result.is_error is False
        assert result.nested_calls is not None
        assert [record.id for record in result.nested_calls.calls] == ["t1/1"]
        assert result.nested_calls.calls[0].name == "leaf"
        assert result.nested_calls.calls[0].parent_tool_call_id == "t1"
        assert result.usage.input == 10

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Persistence and provider payloads
# ---------------------------------------------------------------------------


def _nested_message(usage_tokens: int = 35) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="t1",
        tool_name="outer",
        content=[TextContent(text="ok")],
        is_error=False,
        timestamp=1,
        usage=usage(usage_tokens),
        nested_calls=NestedToolCalls(calls=[
            NestedToolCallRecord(id="t1/1", parent_tool_call_id="t1", name="leaf",
                                 arguments={"text": "\u4e2d"}, status="ok", duration_ms=1),
        ], complete=True),
    )


def test_session_and_ts_round_trip(tmp_path):
    session = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
    session.start_new()
    session.append(_nested_message())
    loaded = session.load(session.session_file)
    result = tool_results(loaded)[0]
    assert result.usage.input == 35
    assert result.nested_calls.calls[0].id == "t1/1"
    assert result.nested_calls.calls[0].arguments == {"text": "\u4e2d"}
    assert session_stats(loaded)["tokens"]["input"] == 35

    ts_path = tmp_path / "export.jsonl"
    session.export_ts(str(ts_path))
    reader = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
    restored = reader.load(str(ts_path))
    restored_result = tool_results(restored)[0]
    assert restored_result.usage.input == 35
    assert restored_result.nested_calls.complete is True
    record = restored_result.nested_calls.calls[0]
    assert record.id == "t1/1"
    # TS keeps no direct parent field; the hierarchical id restores it.
    assert record.parent_tool_call_id == "t1"
    assert record.arguments == {"text": "\u4e2d"}
    assert usage_cost_breakdown(restored).input == 35


def test_provider_payload_omits_nested_calls():
    message = _nested_message()
    wire = wire_message(message)
    assert "nested_calls" not in wire
    assert "nestedCalls" not in wire
    assert wire["content"] == [{"type": "text", "text": "ok"}]

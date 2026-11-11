"""Agent loop.

Python port of ``packages/agent/src/agent-loop.ts``. Works with transcript
messages throughout and converts to LLM messages only at the stream boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from ..ai.models import Model
from ..ai.providers import get_default_stream_fn
from ..ai.types import (
    AssistantMessage,
    DoneEvent,
    StreamErrorEvent,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    combine_usage,
)
from ..ai.utils import (
    NO_CHANGES,
    ToolStateChanges,
    get_current_tools,
    get_tool_state_changes,
    validate_tool_arguments,
    with_tool_changes,
)
from ..util import AbortError, AbortSignal
from ..ai.types import TextContent
from .types import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)

EmitFn = Callable[[Any], Any]


async def run_agent_loop(
    prompts: List[Any],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EmitFn,
    signal: Optional[AbortSignal],
    stream_fn: Optional[Callable[..., AsyncIterator[Any]]] = None,
) -> List[Any]:
    initial_messages = declare_tool_changes(context, prompts)
    new_messages: List[Any] = list(initial_messages)
    current_context = AgentContext(messages=[*context.messages, *initial_messages], tools=context.tools)

    await emit(AgentStartEventType())
    await emit(TurnStartEventType())
    for message in initial_messages:
        await emit(MessageStart(message))
        await emit(MessageEnd(message))

    config, owned_runner = _ensure_nested_runner(config, emit)
    try:
        await run_loop(
            current_context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn()
        )
    finally:
        if owned_runner is not None:
            await owned_runner.close()
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: EmitFn,
    signal: Optional[AbortSignal],
    stream_fn: Optional[Callable[..., AsyncIterator[Any]]] = None,
) -> List[Any]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: List[Any] = []
    current_context = AgentContext(messages=list(context.messages), tools=context.tools)

    await emit(AgentStartEventType())
    await emit(TurnStartEventType())

    config, owned_runner = _ensure_nested_runner(config, emit)
    try:
        await run_loop(current_context, new_messages, config, signal, emit, stream_fn or get_default_stream_fn())
    finally:
        if owned_runner is not None:
            await owned_runner.close()
    return new_messages


# Event shorthands (imported lazily to keep the top imports readable)
def AgentStartEventType():
    from .types import AgentStartEvent

    return AgentStartEvent()


def TurnStartEventType():
    from .types import TurnStartEvent

    return TurnStartEvent()


def MessageStart(message: Any):
    from .types import MessageStartEvent

    return MessageStartEvent(message=message)


def MessageEnd(message: Any):
    from .types import MessageEndEvent

    return MessageEndEvent(message=message)


async def run_loop(
    initial_context: AgentContext,
    new_messages: List[Any],
    initial_config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: EmitFn,
    stream_function: Callable[..., AsyncIterator[Any]],
) -> None:
    from .types import AgentEndEvent, TurnEndEvent

    current_context = initial_context
    config = initial_config
    last_completed_turn = None
    explicit_continuation = False
    pending_messages: List[Any] = []
    if config.get_steering_messages is not None:
        pending_messages = list(await config.get_steering_messages() or [])

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            prepared_messages: List[Any] = []
            if last_completed_turn is not None:
                if config.prepare_next_turn is not None:
                    next_turn_snapshot = await config.prepare_next_turn(last_completed_turn)
                    if next_turn_snapshot is not None:
                        if next_turn_snapshot.context is not None:
                            current_context = next_turn_snapshot.context
                        prepared_messages = next_turn_snapshot.messages or []
                        config = _replace_config(
                            config,
                            model=next_turn_snapshot.model or config.model,
                            reasoning=(
                                config.reasoning
                                if next_turn_snapshot.thinking_level is None
                                else (
                                    None
                                    if next_turn_snapshot.thinking_level == "off"
                                    else next_turn_snapshot.thinking_level
                                )
                            ),
                        )
                if not pending_messages and config.get_steering_messages is not None:
                    pending_messages = list(await config.get_steering_messages() or [])
                await emit(TurnStartEventType())

            for message in declare_tool_changes(current_context, [*prepared_messages, *pending_messages]):
                await emit(MessageStart(message))
                await emit(MessageEnd(message))
                current_context.messages.append(message)
                new_messages.append(message)
            pending_messages = []

            if config.prepare_request is not None:
                from .types import AgentRequestContext

                request_update = await config.prepare_request(
                    AgentRequestContext(
                        context=current_context,
                        model=config.model,
                        thinking_level=config.reasoning or "off",
                    ),
                    signal,
                )
                if request_update is not None:
                    if request_update.context is not None:
                        current_context = request_update.context
                    config = _replace_config(
                        config,
                        model=request_update.model or config.model,
                        reasoning=(
                            config.reasoning
                            if request_update.thinking_level is None
                            else (None if request_update.thinking_level == "off" else request_update.thinking_level)
                        ),
                    )

            message = await stream_assistant_response(current_context, config, signal, emit, stream_function)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                # The provider may have streamed tool_use blocks before the
                # request failed; answer them so the persisted transcript
                # stays valid for the next request.
                streamed_calls = [
                    c for c in message.content if isinstance(c, ToolCall) and c.raw_item is None
                ]
                aborted_results: List[ToolResultMessage] = []
                for finalized in await _finalize_aborted_tool_calls(streamed_calls, emit):
                    result_message = create_tool_result_message(finalized)
                    await emit(MessageStart(result_message))
                    await emit(MessageEnd(result_message))
                    current_context.messages.append(result_message)
                    new_messages.append(result_message)
                    aborted_results.append(result_message)
                last_completed_turn = _turn_context(message, aborted_results, current_context, new_messages)
                if config.finish_turn is not None:
                    await config.finish_turn(last_completed_turn, signal)
                await emit(TurnEndEvent(message=message, tool_results=aborted_results))
                await emit(AgentEndEvent(messages=new_messages))
                return

            # Hosted (provider-executed) calls carry their result server-side
            # and replay via raw_item; the loop neither executes nor answers them.
            tool_calls = [c for c in message.content if isinstance(c, ToolCall) and c.raw_item is None]

            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                if message.stop_reason == "length":
                    batch = await fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    batch = await execute_tool_calls(current_context, message, config, signal, emit)
                tool_results.extend(batch["messages"])
                has_more_tool_calls = not batch["terminate"]

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            last_completed_turn = _turn_context(message, tool_results, current_context, new_messages)
            decision = None
            if config.finish_turn is not None:
                decision = await config.finish_turn(last_completed_turn, signal)
            await emit(TurnEndEvent(message=message, tool_results=tool_results))

            if decision is not None and getattr(decision, "action", None) == "end":
                await emit(AgentEndEvent(messages=new_messages))
                return

            explicit_continuation = decision is not None and getattr(decision, "action", None) == "continue"
            pending_messages = []
            if config.get_steering_messages is not None:
                pending_messages = list(await config.get_steering_messages() or [])
            if has_more_tool_calls or pending_messages:
                explicit_continuation = False

        follow_up_messages: List[Any] = []
        if config.get_follow_up_messages is not None:
            follow_up_messages = list(await config.get_follow_up_messages() or [])
        if follow_up_messages:
            explicit_continuation = False
            pending_messages = follow_up_messages
            continue

        if explicit_continuation:
            explicit_continuation = False
            continue

        break

    await emit(AgentEndEvent(messages=new_messages))


def _turn_context(message, tool_results, context, new_messages):
    from .types import AgentTurnContext

    return AgentTurnContext(
        message=message, tool_results=tool_results, context=context, new_messages=new_messages
    )


def _replace_config(config: AgentLoopConfig, **changes: Any) -> AgentLoopConfig:
    import dataclasses

    return dataclasses.replace(config, **changes)


def _ensure_nested_runner(config: AgentLoopConfig, emit: EmitFn):
    """Use the configured session runner, or own one for this run."""
    runner = getattr(config, "nested_tool_call_runner", None)
    if runner is not None and not getattr(runner, "closed", False):
        return config, None
    from ..coding_agent.nested_tool_calls import LoopToolHost, NestedToolCallRunner

    runner = NestedToolCallRunner(LoopToolHost(emit))
    return _replace_config(config, nested_tool_call_runner=runner), runner


def declare_tool_changes(context: AgentContext, pending_messages: List[Any]) -> List[Any]:
    """Declare tool loadout changes to the model via system messages."""
    system_index = -1
    for i in range(len(pending_messages) - 1, -1, -1):
        if getattr(pending_messages[i], "role", None) == "system":
            system_index = i
            break

    pending: Optional[SystemMessage] = (
        pending_messages[system_index] if system_index >= 0 else None  # type: ignore[assignment]
    )
    if pending is not None:
        baseline = [
            with_tool_changes(pending, NO_CHANGES) if i == system_index else message
            for i, message in enumerate(pending_messages)
        ]
    else:
        baseline = pending_messages

    declared = get_current_tools([*context.messages, *baseline])
    # Rebuild declaration-only tools (dropping execute callables), preserving
    # hosted tool payloads so they re-declare with their provider shape.
    executable = [AgentTool(name=t.name, description=t.description, parameters=t.parameters,
                            hosted=getattr(t, "hosted", None)) for t in (context.tools or [])]
    changes = get_tool_state_changes(declared, executable)
    unchanged = not changes.tools_added and not changes.tools_removed

    if pending is not None:
        if unchanged and not (pending.tools_added or pending.tools_removed):
            return pending_messages
        return [
            with_tool_changes(pending, changes) if i == system_index else message
            for i, message in enumerate(pending_messages)
        ]
    if unchanged:
        return pending_messages

    update = with_tool_changes(SystemMessage(content="", timestamp=_now_ms()), changes)
    insert_index = next(
        (i for i, message in enumerate(pending_messages) if getattr(message, "role", None) != "system"),
        len(pending_messages),
    )
    return [*pending_messages[:insert_index], update, *pending_messages[insert_index:]]


async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: EmitFn,
    stream_function: Callable[..., AsyncIterator[Any]],
) -> AssistantMessage:
    from .types import MessageEndEvent, MessageStartEvent, MessageUpdateEvent

    messages = context.messages
    if config.transform_context is not None:
        messages = await config.transform_context(messages, signal)

    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    resolved_api_key = config.api_key
    if config.get_api_key is not None:
        resolved_api_key = await _maybe_await(config.get_api_key(config.model.provider)) or config.api_key

    options = _build_stream_options(config, resolved_api_key, signal)
    response = stream_function(config.model, TranscriptContext(messages=list(llm_messages)), options)

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        if isinstance(event, (DoneEvent, StreamErrorEvent)):
            final = event.message if isinstance(event, DoneEvent) else event.error
            final = _finalize_message(final, config)
            if added_partial:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await emit(MessageStartEvent(message=final))
            await emit(MessageEndEvent(message=final))
            return final

        partial = event.partial
        if event.type == "start":
            partial_message = partial
            context.messages.append(partial_message)
            added_partial = True
            await emit(MessageStartEvent(message=partial))
        elif partial_message is not None:
            partial_message = partial
            context.messages[-1] = partial
            await emit(MessageUpdateEvent(message=partial, assistant_message_event=event))

    raise RuntimeError("Provider stream ended without a done/error event")


def _finalize_message(message: AssistantMessage, config: AgentLoopConfig) -> AssistantMessage:
    message.thinking_level = config.reasoning or "off"
    return message


def _build_stream_options(config: AgentLoopConfig, api_key: Optional[str], signal: Optional[AbortSignal]):
    from ..ai.providers import SimpleStreamOptions

    return SimpleStreamOptions(
        signal=signal,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        reasoning=config.reasoning,
        telemetry_context=config.telemetry_context,
        cache_retention=getattr(config, "cache_retention", None),
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def fail_tool_calls_from_truncated_message(
    tool_calls: List[ToolCall], emit: EmitFn
) -> Dict[str, Any]:
    from .types import ToolExecutionStartEvent, ToolExecutionEndEvent

    messages: List[ToolResultMessage] = []
    for tool_call in tool_calls:
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
            )
        )
        finalized = {
            "tool_call": tool_call,
            "result": create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token '
                "limit, so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            "is_error": True,
        }
        await emit(
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=finalized["result"],
                is_error=True,
            )
        )
        tool_result_message = create_tool_result_message(finalized)
        await emit(MessageStart(tool_result_message))
        await emit(MessageEnd(tool_result_message))
        messages.append(tool_result_message)
    return {"messages": messages, "terminate": False}


async def _finalize_aborted_tool_calls(
    tool_calls: List[ToolCall], emit: EmitFn
) -> List[Dict[str, Any]]:
    """Answer calls an abort left unstarted so every tool_use has a result.

    Providers reject a transcript whose assistant ``toolUse`` blocks lack a
    matching ``toolResult``, so an aborted batch must still answer every call.
    """
    from .types import ToolExecutionStartEvent, ToolExecutionEndEvent

    finalized_calls: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
            )
        )
        finalized = {
            "tool_call": tool_call,
            "result": create_error_tool_result("Operation aborted"),
            "is_error": True,
        }
        await emit(
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=finalized["result"],
                is_error=True,
            )
        )
        finalized_calls.append(finalized)
    return finalized_calls


def _enter_call_context(config: AgentLoopConfig, context: AgentContext,
                        assistant_message: AssistantMessage, signal: Optional[AbortSignal],
                        tool_call: ToolCall, holds_queue: bool):
    """Make ``tool_call`` the active call for execute_tool and nested runners."""
    runner = getattr(config, "nested_tool_call_runner", None)
    if runner is None:
        return None
    from ..coding_agent.nested_tool_calls import NestedCallContext, set_current_call

    return set_current_call(NestedCallContext(
        root_id=tool_call.id,
        caller_id=tool_call.id,
        assistant_message=assistant_message,
        context=context,
        config=config,
        signal=signal,
        holds_queue=holds_queue,
        runner=runner,
    ))


def _leave_call_context(token) -> None:
    if token is None:
        return
    from ..coding_agent.nested_tool_calls import reset_current_call

    reset_current_call(token)


async def _attach_nested_record(config: AgentLoopConfig, root_id: str,
                                finalized: Dict[str, Any]) -> Dict[str, Any]:
    """Consume one top-level call's nested record so content and usage land once."""
    runner = getattr(config, "nested_tool_call_runner", None)
    if runner is None:
        return finalized
    summary = await runner.collect_record(root_id)
    if summary is None:
        return finalized
    if summary.calls is not None:
        finalized["nested_calls"] = summary.calls
    if summary.usage is not None:
        finalized["nested_usage"] = summary.usage
    return finalized


async def run_single_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    on_update,
    parent_tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare, execute and finalize one call through the shared pipeline."""
    preparation = await prepare_tool_call(
        current_context, assistant_message, tool_call, config, signal,
        parent_tool_call_id=parent_tool_call_id,
    )
    if preparation["kind"] == "immediate":
        return {"tool_call": tool_call, "result": preparation["result"],
                "is_error": preparation["is_error"]}
    executed = await execute_prepared_tool_call(preparation, signal, on_update)
    return await finalize_executed_tool_call(
        current_context, assistant_message, preparation, executed, config, signal,
        parent_tool_call_id=parent_tool_call_id,
    )


async def execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: EmitFn,
) -> Dict[str, Any]:
    from .types import ToolExecutionStartEvent

    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall) and c.raw_item is None]
    tools = current_context.tools or []

    def find_tool(name: str) -> Optional[AgentTool]:
        return next((t for t in tools if t.name == name), None)

    has_sequential = any(
        (find_tool(tc.name) is not None and find_tool(tc.name).execution_mode == "sequential")  # type: ignore[union-attr]
        for tc in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, signal, emit)
    return await execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, signal, emit)


async def execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[ToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: EmitFn,
) -> Dict[str, Any]:
    from .types import ToolExecutionEndEvent, ToolExecutionStartEvent

    finalized_calls: List[Dict[str, Any]] = []
    messages: List[ToolResultMessage] = []

    for index, tool_call in enumerate(tool_calls):
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
            )
        )
        # A sequential batch owns the tool pipeline; nested exclusive calls
        # inherit that lease instead of waiting on it.
        token = _enter_call_context(
            config, current_context, assistant_message, signal, tool_call, holds_queue=True
        )
        try:
            preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
            if preparation["kind"] == "immediate":
                finalized = {"tool_call": tool_call, "result": preparation["result"],
                             "is_error": preparation["is_error"]}
            else:
                executed = await execute_prepared_tool_call(preparation, signal, _update_sink(tool_call, emit))
                finalized = await finalize_executed_tool_call(
                    current_context, assistant_message, preparation, executed, config, signal
                )
        finally:
            _leave_call_context(token)
        finalized = await _attach_nested_record(config, tool_call.id, finalized)

        await emit(
            ToolExecutionEndEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=finalized["result"],
                is_error=finalized["is_error"],
            )
        )
        tool_result_message = create_tool_result_message(finalized)
        await emit(MessageStart(tool_result_message))
        await emit(MessageEnd(tool_result_message))
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if signal is not None and signal.aborted:
            # Every tool_use block in the assistant message needs a matching
            # tool_result, otherwise the next request is rejected by the
            # provider. Answer the calls the abort prevented from running.
            for finalized in await _finalize_aborted_tool_calls(tool_calls[index + 1:], emit):
                tool_result_message = create_tool_result_message(finalized)
                await emit(MessageStart(tool_result_message))
                await emit(MessageEnd(tool_result_message))
                finalized_calls.append(finalized)
                messages.append(tool_result_message)
            break

    return {"messages": messages, "terminate": _should_terminate_batch(finalized_calls)}


async def execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[ToolCall],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    emit: EmitFn,
) -> Dict[str, Any]:
    from .types import ToolExecutionEndEvent, ToolExecutionStartEvent

    entries: List[Any] = []
    accounted = 0

    for index, tool_call in enumerate(tool_calls):
        await emit(
            ToolExecutionStartEvent(
                tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments
            )
        )
        preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        accounted = index + 1
        if preparation["kind"] == "immediate":
            finalized = {
                "tool_call": tool_call,
                "result": preparation["result"],
                "is_error": preparation["is_error"],
            }
            entries.append(finalized)
            await emit(
                ToolExecutionEndEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=finalized["result"],
                    is_error=finalized["is_error"],
                )
            )
            if signal is not None and signal.aborted:
                break
            continue

        async def run_one(preparation: Dict[str, Any] = preparation, tool_call: ToolCall = tool_call) -> Dict[str, Any]:
            from .types import ToolExecutionEndEvent as TEE

            if signal is not None and signal.aborted:
                finalized = {
                    "tool_call": tool_call,
                    "result": create_error_tool_result("Operation aborted"),
                    "is_error": True,
                }
                finalized = await _attach_nested_record(config, tool_call.id, finalized)
                await emit(TEE(tool_call_id=tool_call.id, tool_name=tool_call.name, result=finalized["result"], is_error=True))
                return finalized
            token = _enter_call_context(
                config, current_context, assistant_message, signal, tool_call, holds_queue=False
            )
            try:
                executed = await execute_prepared_tool_call(preparation, signal, _update_sink(tool_call, emit))
                finalized = await finalize_executed_tool_call(
                    current_context, assistant_message, preparation, executed, config, signal
                )
            finally:
                _leave_call_context(token)
            finalized = await _attach_nested_record(config, tool_call.id, finalized)
            await emit(
                TEE(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=finalized["result"],
                    is_error=finalized["is_error"],
                )
            )
            return finalized

        entries.append(run_one)
        if signal is not None and signal.aborted:
            break

    ordered = await asyncio.gather(*[_maybe_call(entry) for entry in entries])
    ordered.extend(await _finalize_aborted_tool_calls(tool_calls[accounted:], emit))
    messages: List[ToolResultMessage] = []
    for finalized in ordered:
        tool_result_message = create_tool_result_message(finalized)
        await emit(MessageStart(tool_result_message))
        await emit(MessageEnd(tool_result_message))
        messages.append(tool_result_message)

    return {"messages": messages, "terminate": _should_terminate_batch(list(ordered))}


async def _maybe_call(entry: Any) -> Dict[str, Any]:
    if asyncio.iscoroutinefunction(entry):
        return await entry()
    return entry


def _update_sink(tool_call: ToolCall, emit: EmitFn) -> Callable[[AgentToolResult], Any]:
    from .types import ToolExecutionUpdateEvent

    async def sink(partial_result: AgentToolResult) -> None:
        await emit(
            ToolExecutionUpdateEvent(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
                partial_result=partial_result,
            )
        )

    return sink


def _should_terminate_batch(finalized_calls: List[Dict[str, Any]]) -> bool:
    return bool(finalized_calls) and all(f["result"].terminate is True for f in finalized_calls)


async def prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    tools: Optional[List[AgentTool]] = None,
    parent_tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    tools = tools if tools is not None else (current_context.tools or [])
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        return {
            "kind": "immediate",
            "result": create_error_tool_result(f"Tool {tool_call.name} not found"),
            "is_error": True,
        }
    if tool.execute is None:
        return {
            "kind": "immediate",
            "result": create_error_tool_result(f"Tool {tool_call.name} has no executor"),
            "is_error": True,
        }

    try:
        args = tool_call.arguments
        if tool.prepare_arguments is not None:
            args = tool.prepare_arguments(args)
        validated_args = validate_tool_arguments(tool, args)
        if config.before_tool_call is not None:
            from .types import BeforeToolCallContext

            before_result = await config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    tool_arguments=tool_call.arguments,
                    args=validated_args,
                    context=current_context,
                    parent_tool_call_id=parent_tool_call_id,
                ),
                signal,
            )
            if signal is not None and signal.aborted:
                return {
                    "kind": "immediate",
                    "result": create_error_tool_result("Operation aborted"),
                    "is_error": True,
                }
            if before_result is not None and getattr(before_result, "block", False):
                reason = getattr(before_result, "reason", None) or "Tool execution was blocked"
                result = create_error_tool_result(reason)
                if getattr(before_result, "terminate", False):
                    result.terminate = True
                return {"kind": "immediate", "result": result, "is_error": True}
        if signal is not None and signal.aborted:
            return {
                "kind": "immediate",
                "result": create_error_tool_result("Operation aborted"),
                "is_error": True,
            }
        return {"kind": "prepared", "tool_call": tool_call, "tool": tool, "args": validated_args}
    except AbortError:
        return {"kind": "immediate", "result": create_error_tool_result("Operation aborted"), "is_error": True}
    except Exception as exc:  # noqa: BLE001
        return {"kind": "immediate", "result": create_error_tool_result(str(exc)), "is_error": True}


async def execute_prepared_tool_call(
    prepared: Dict[str, Any],
    signal: Optional[AbortSignal],
    on_update: Callable[[AgentToolResult], Any],
) -> Dict[str, Any]:
    update_tasks: List[Any] = []
    accepting_updates = True

    async def sink(partial_result: AgentToolResult) -> None:
        if accepting_updates:
            update_tasks.append(asyncio.create_task(_maybe_await(on_update(partial_result))))

    tool: AgentTool = prepared["tool"]
    try:
        result = await tool.execute(prepared["tool_call"].id, prepared["args"], signal, sink)
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return {"result": result, "is_error": bool(result.is_error)}
    except AbortError:
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return {"result": create_error_tool_result("Operation aborted"), "is_error": True}
    except Exception as exc:  # noqa: BLE001
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return {"result": create_error_tool_result(str(exc)), "is_error": True}
    finally:
        accepting_updates = False
        for task in update_tasks:
            if not task.done():
                task.cancel()
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)


async def finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: Dict[str, Any],
    executed: Dict[str, Any],
    config: AgentLoopConfig,
    signal: Optional[AbortSignal],
    parent_tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    from .types import AfterToolCallContext

    result: AgentToolResult = executed["result"]
    is_error: bool = executed["is_error"]

    if config.after_tool_call is not None:
        try:
            after_result = await config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call_id=prepared["tool_call"].id,
                    tool_name=prepared["tool_call"].name,
                    tool_arguments=prepared["tool_call"].arguments,
                    args=prepared["args"],
                    result=result,
                    is_error=is_error,
                    context=current_context,
                    parent_tool_call_id=parent_tool_call_id,
                ),
                signal,
            )
            if after_result is not None and isinstance(after_result, AfterToolCallResult):
                structured = after_result.structured_content
                if structured is None and after_result.content is not None:
                    structured = None
                result = AgentToolResult(
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    structured_content=structured if structured is not None else result.structured_content,
                    usage=after_result.usage if after_result.usage is not None else result.usage,
                    is_error=result.is_error,
                    terminate=(
                        after_result.terminate if after_result.terminate is not None else result.terminate
                    ),
                )
                if after_result.is_error is not None:
                    is_error = after_result.is_error
        except Exception as exc:  # noqa: BLE001
            result = create_error_tool_result(str(exc))
            is_error = True

    return {"tool_call": prepared["tool_call"], "result": result, "is_error": is_error}


def create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def create_tool_result_message(finalized: Dict[str, Any]) -> ToolResultMessage:
    result: AgentToolResult = finalized["result"]
    # The top-level message is the only persisted result: its usage is the
    # tool's own usage plus the recorder's sum of nested tool usage.
    usage = result.usage
    nested_usage: Optional[Usage] = finalized.get("nested_usage")
    if nested_usage is not None:
        usage = combine_usage(usage, nested_usage) if usage is not None else nested_usage
    return ToolResultMessage(
        tool_call_id=finalized["tool_call"].id,
        tool_name=finalized["tool_call"].name,
        content=result.content or [],
        is_error=finalized["is_error"],
        timestamp=_now_ms(),
        details=result.details,
        usage=usage,
        nested_calls=finalized.get("nested_calls"),
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


IMAGE_DISABLED_TEXT = "Image reading is disabled."


def _filter_message_images(message: Any) -> Any:
    """Replace image blocks with a text placeholder, collapsing consecutive duplicates."""
    import dataclasses  # noqa: PLC0415

    content = getattr(message, "content", None)
    if not isinstance(content, list) or not any(getattr(block, "type", None) == "image" for block in content):
        return message
    filtered: List[Any] = []
    for block in content:
        if getattr(block, "type", None) == "image":
            previous = filtered[-1] if filtered else None
            if isinstance(previous, TextContent) and previous.text == IMAGE_DISABLED_TEXT:
                continue
            filtered.append(TextContent(text=IMAGE_DISABLED_TEXT))
        else:
            filtered.append(block)
    return dataclasses.replace(message, content=filtered)


def default_convert_to_llm(messages: List[Any], block_images: bool = False) -> List[Any]:
    """Pass through the four LLM roles, filtering UI-only custom messages.

    With ``block_images`` set, image blocks in user and toolResult messages are
    swapped for a text placeholder (defense-in-depth for the block_images
    setting). The filter runs at the stream boundary so setting changes take
    effect on the next request without touching stored history.
    """
    converted = [m for m in messages if getattr(m, "role", None) in ("system", "user", "assistant", "toolResult")]
    if not block_images:
        return converted
    return [
        _filter_message_images(m) if getattr(m, "role", None) in ("user", "toolResult") else m for m in converted
    ]

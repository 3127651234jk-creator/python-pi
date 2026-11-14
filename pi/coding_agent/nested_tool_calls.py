"""Nested tool calls made while another tool runs.

Port of ``packages/coding-agent/src/core/nested-tool-calls.ts``. A tool such as
codemode calls :func:`execute_tool`; the runner executes the call through the
agent loop pipeline (validation, hooks, permission decisions), emits
``tool_execution_*`` events carrying ``parent_tool_call_id``, records the call
on the model-issued call's tool result message, and sums the usage of every
nested result exactly once.

Nothing here runs until a tool calls :func:`execute_tool`. Calls made through
the tool's direct fallback path do not reach this module and therefore have no
parent, hooks, or session record.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from ..agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentToolResult,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from ..ai.types import (
    NestedToolCallRecord,
    NestedToolCalls,
    TextContent,
    ToolCall,
    Usage,
    combine_usage,
)
from ..util import AbortSignal

# Record limits of one model-issued call: arguments over the per-call or total
# size are omitted, calls beyond the count are dropped, and the record is marked
# incomplete when any of that happens.
MAX_CALLS = 256
MAX_ARGUMENT_BYTES_PER_CALL = 8 * 1024
MAX_ARGUMENT_BYTES_TOTAL = 32 * 1024
MAX_ERROR_CHARS = 500


def _arguments_json(arguments: Any) -> str:
    """Compact JSON used for both the size limit and the retained arguments."""
    return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def _text_of(result: AgentToolResult) -> str:
    return "\n".join(block.text for block in (result.content or []) if isinstance(block, TextContent))


@dataclass
class NestedCallSummary:
    """What the nested calls of one model-issued tool call leave on its result."""

    # Becomes ``nested_calls``. None when no nested call was made.
    calls: Optional[NestedToolCalls]
    # Summed usage of the nested results, added to the message's usage.
    usage: Optional[Usage]


@dataclass
class NestedToolCallOutcome:
    """One nested call's full result, before any script-facing conversion."""

    tool_call_id: str
    result: AgentToolResult
    is_error: bool


class NestedCallRecorder:
    """Collects the nested calls of one model-issued call, including deeper calls."""

    def __init__(self) -> None:
        self._calls: List[NestedToolCallRecord] = []
        self._started_at: Dict[int, float] = {}
        self._complete = True
        self._argument_bytes = 0
        # Summed usage of every nested result, including calls dropped from the record.
        self._usage: Optional[Usage] = None

    def start(self, call_id: str, parent_tool_call_id: str, name: str,
              arguments: Any) -> Optional[NestedToolCallRecord]:
        """Record a call as it starts. Returns None when the call is dropped."""
        if len(self._calls) >= MAX_CALLS:
            self._complete = False
            return None
        record = NestedToolCallRecord(id=call_id, parent_tool_call_id=parent_tool_call_id, name=name)
        encoded = _arguments_json(arguments)
        size = len(encoded.encode("utf-8"))
        if (size > MAX_ARGUMENT_BYTES_PER_CALL
                or self._argument_bytes + size > MAX_ARGUMENT_BYTES_TOTAL):
            record.arguments_bytes = size
            self._complete = False
        else:
            parsed = json.loads(encoded)
            record.arguments = parsed if isinstance(parsed, dict) else {}
            self._argument_bytes += size
        self._calls.append(record)
        self._started_at[id(record)] = time.monotonic()
        return record

    def finish(self, record: Optional[NestedToolCallRecord], status: str,
               error_text: str = "") -> None:
        if record is None:
            return
        record.status = status
        started = self._started_at.pop(id(record), time.monotonic())
        record.duration_ms = round((time.monotonic() - started) * 1000)
        if status == "error" and error_text:
            record.error = error_text[:MAX_ERROR_CHARS]

    def add_usage(self, usage: Usage) -> None:
        self._usage = combine_usage(self._usage, usage) if self._usage is not None else usage

    @property
    def total_usage(self) -> Optional[Usage]:
        return self._usage

    def snapshot(self) -> Optional[NestedToolCalls]:
        """Copy of the record so far, or None when no nested call was made."""
        if not self._calls and self._complete:
            return None
        calls = [replace(call) for call in self._calls]
        return NestedToolCalls(
            calls=calls,
            complete=self._complete and all(call.status != "unfinished" for call in calls),
        )


@dataclass
class NestedCallContext:
    """The tool call currently running, carried by a ContextVar.

    Tools reach this through :func:`execute_tool`. ``caller_id`` is the direct
    parent; ``root_id`` names the model-issued call every nested call belongs to.
    """

    root_id: str
    caller_id: str
    assistant_message: Any
    context: AgentContext
    config: AgentLoopConfig
    signal: Optional[AbortSignal]
    holds_queue: bool
    runner: "NestedToolCallRunner"


_CURRENT_CALL: ContextVar[Optional[NestedCallContext]] = ContextVar(
    "pi_nested_tool_call", default=None
)


def current_call() -> Optional[NestedCallContext]:
    return _CURRENT_CALL.get()


def set_current_call(call: NestedCallContext):
    """Bind ``call`` for the current task; pass the token to reset_current_call."""
    return _CURRENT_CALL.set(call)


def reset_current_call(token) -> None:
    _CURRENT_CALL.reset(token)


@dataclass
class _CallScope:
    recorder: NestedCallRecorder
    next_id: int = 1
    # Set inside a call that holds the exclusive queue, so its own nested calls
    # do not wait on it.
    holds_queue: bool = False


class NestedToolCallHost:
    """What the runner needs from the agent loop."""

    def get_tools(self) -> List[Any]:
        raise NotImplementedError

    def is_sequential(self) -> bool:
        raise NotImplementedError

    async def run_tool_call(self, tool_call: Any, parent_tool_call_id: str,
                            signal: Optional[AbortSignal], on_update) -> Dict[str, Any]:
        raise NotImplementedError

    async def emit(self, event: Any) -> None:
        raise NotImplementedError


class LoopToolHost(NestedToolCallHost):
    """Host bound to the agent loop currently running."""

    def __init__(self, emit) -> None:
        self._emit_fn = emit

    def _call(self) -> NestedCallContext:
        call = _CURRENT_CALL.get()
        if call is None:
            raise RuntimeError("Nested tool call has no active agent-loop context")
        return call

    def get_tools(self) -> List[Any]:
        return list(self._call().context.tools or [])

    def is_sequential(self) -> bool:
        return self._call().config.tool_execution == "sequential"

    async def run_tool_call(self, tool_call: Any, parent_tool_call_id: str,
                            signal: Optional[AbortSignal], on_update) -> Dict[str, Any]:
        # Imported lazily: agent_loop imports this module to set the context.
        from ..agent.agent_loop import run_single_tool_call

        call = self._call()
        return await run_single_tool_call(
            call.context, call.assistant_message, tool_call, call.config, signal,
            on_update, parent_tool_call_id,
        )

    async def emit(self, event: Any) -> None:
        await self._emit_fn(event)


class NestedToolCallRunner:
    """Executes the nested calls below one or more model-issued calls."""

    def __init__(self, host: NestedToolCallHost) -> None:
        self._host = host
        self._scopes: Dict[str, _CallScope] = {}
        # Serializes nested calls that must not run concurrently. Top-level
        # sequential execution makes its calls holders of this queue.
        self._exclusive = asyncio.Lock()
        self._tasks: set = set()
        self._root_tasks: Dict[str, set] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _scope_for(self, caller_id: str) -> _CallScope:
        scope = self._scopes.get(caller_id)
        if scope is None:
            call = _CURRENT_CALL.get()
            scope = _CallScope(
                recorder=NestedCallRecorder(),
                holds_queue=bool(call.holds_queue) if call is not None else False,
            )
            self._scopes[caller_id] = scope
        return scope

    def _is_sequential_tool(self, name: str) -> bool:
        return any(
            getattr(tool, "name", None) == name and getattr(tool, "execution_mode", None) == "sequential"
            for tool in self._host.get_tools()
        )

    async def execute(
        self,
        caller_id: str,
        name: str,
        args: Any,
        signal: Optional[AbortSignal] = None,
        on_update=None,
    ) -> NestedToolCallOutcome:
        """Run ``name`` on behalf of ``caller_id``; tool failures come back as is_error.

        The nested call gets the id ``<caller_id>/<n>``. The number is assigned
        before the first await and never reused.
        """
        call = _CURRENT_CALL.get()
        if call is None:
            raise RuntimeError("execute_tool() requires an active tool call")
        if self._closed:
            raise RuntimeError("Nested tool call runner is closed")

        scope = self._scope_for(caller_id)
        tool_call_id = f"{caller_id}/{scope.next_id}"
        scope.next_id += 1
        reserved_ids = {block.id for block in getattr(call.assistant_message, "content", []) if isinstance(block, ToolCall)}
        while tool_call_id in reserved_ids or tool_call_id in self._scopes:
            tool_call_id = f"{caller_id}/{scope.next_id}"
            scope.next_id += 1
        arguments = args if isinstance(args, dict) else {}
        record = scope.recorder.start(tool_call_id, caller_id, name, arguments)

        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
            self._root_tasks.setdefault(call.root_id, set()).add(task)

        started = False
        release = False
        token = None
        call_scope_added = False
        outcome: Optional[Dict[str, Any]] = None
        result: Optional[AgentToolResult] = None
        is_error = True
        status = "error"
        error_text = ""
        cancelled = False

        try:
            await self._host.emit(ToolExecutionStartEvent(
                tool_call_id=tool_call_id, tool_name=name, args=arguments,
                parent_tool_call_id=caller_id,
            ))
            started = True

            exclusive = (not scope.holds_queue
                         and (self._host.is_sequential() or self._is_sequential_tool(name)))
            if exclusive:
                await self._exclusive.acquire()
                release = True

            effective_signal = signal if signal is not None else call.signal
            self._scopes[tool_call_id] = _CallScope(
                recorder=scope.recorder, holds_queue=scope.holds_queue or exclusive
            )
            call_scope_added = True
            token = set_current_call(NestedCallContext(
                root_id=call.root_id,
                caller_id=tool_call_id,
                assistant_message=call.assistant_message,
                context=call.context,
                config=call.config,
                signal=effective_signal,
                holds_queue=scope.holds_queue or exclusive,
                runner=self,
            ))

            async def forward_update(partial_result: AgentToolResult) -> None:
                if on_update is not None:
                    forwarded = on_update(partial_result)
                    if inspect.isawaitable(forwarded):
                        await forwarded
                await self._host.emit(ToolExecutionUpdateEvent(
                    tool_call_id=tool_call_id, tool_name=name, args=arguments,
                    partial_result=partial_result, parent_tool_call_id=caller_id,
                ))

            outcome = await self._host.run_tool_call(
                ToolCall(id=tool_call_id, name=name, arguments=arguments),
                caller_id, effective_signal, forward_update,
            )
            result = outcome["result"]
            is_error = bool(outcome.get("is_error"))
            if effective_signal is not None and effective_signal.aborted:
                status = "cancelled"
            else:
                status = "error" if is_error else "ok"
                if is_error:
                    error_text = _text_of(result)
        except asyncio.CancelledError:
            cancelled = True
            status = "cancelled"
            result = AgentToolResult(content=[TextContent(text="Operation aborted")], details={})
            is_error = True
        except Exception as exc:  # noqa: BLE001 - the runner reports failures as outcomes
            status = "error"
            error_text = str(exc)
            result = AgentToolResult(content=[TextContent(text=str(exc))], details={})
            is_error = True
        finally:
            if token is not None:
                reset_current_call(token)
            if call_scope_added:
                self._scopes.pop(tool_call_id, None)
            if release:
                self._exclusive.release()
            if task is not None:
                self._tasks.discard(task)
                root_tasks = self._root_tasks.get(call.root_id)
                if root_tasks is not None:
                    root_tasks.discard(task)
                    if not root_tasks:
                        self._root_tasks.pop(call.root_id, None)

        if started:
            scope.recorder.finish(record, status, error_text)
            if result is not None and result.usage is not None:
                scope.recorder.add_usage(result.usage)
            await self._host.emit(ToolExecutionEndEvent(
                tool_call_id=tool_call_id, tool_name=name, result=result,
                is_error=is_error, parent_tool_call_id=caller_id,
            ))
        if cancelled:
            raise asyncio.CancelledError()
        if outcome is None:
            # An unexpected exception was converted to an error outcome above.
            return NestedToolCallOutcome(
                tool_call_id=tool_call_id,
                result=result if result is not None else AgentToolResult(content=[], details={}),
                is_error=True,
            )
        return NestedToolCallOutcome(tool_call_id=tool_call_id, result=result, is_error=is_error)

    async def collect_record(self, root_id: str) -> Optional[NestedCallSummary]:
        """Cancel detached calls of ``root_id``, wait for them, then take the record.

        A tool that leaves nested calls running must not let them change the
        record after its own result was persisted.
        """
        current = asyncio.current_task()
        pending = [task for task in self._root_tasks.pop(root_id, set())
                   if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return self.take_record(root_id)

    def take_record(self, root_id: str) -> Optional[NestedCallSummary]:
        """Remove and return the record of the nested calls a model-issued call made."""
        scope = self._scopes.pop(root_id, None)
        if scope is None:
            return None
        return NestedCallSummary(calls=scope.recorder.snapshot(), usage=scope.recorder.total_usage)

    async def close(self) -> None:
        """Cancel and await in-flight nested calls, then clear every scope."""
        self._closed = True
        current = asyncio.current_task()
        pending = [task for task in self._tasks if task is not current and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
        self._root_tasks.clear()
        self._scopes.clear()


async def execute_tool(
    name: str,
    args: Any,
    signal: Optional[AbortSignal] = None,
    on_update=None,
) -> NestedToolCallOutcome:
    """Run a tool on behalf of the currently running tool call.

    Raises RuntimeError when no tool call is active, so a tool cannot silently
    execute outside the agent pipeline.
    """
    call = _CURRENT_CALL.get()
    if call is None:
        raise RuntimeError("execute_tool() requires an active tool call")
    return await call.runner.execute(call.caller_id, name, args, signal, on_update)

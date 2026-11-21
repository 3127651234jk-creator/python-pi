"""Durable harness: scheduler and the built-in pi.generation task.

Python port of the core of ``packages/durable/src/harness``: task reservations,
the phase state machine, waiting/join policy, abort marks, crash recovery
(running resets to pending on reopen), and the generation task that commits a
model turn to the transcript before it becomes visible.

Named extensions provide tools, tasks and hooks. Busy inputs and passive writes
are admitted through the transactional inbox. ``pi.tool`` and ``pi.compact``
live in ``tasks.py``; the scheduler supports bounded concurrency.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..ai.json_io import message_from_json, to_json as ai_to_json
from ..ai.models import Model
from ..ai.retry import RetryPolicy, is_retryable_assistant_error, retry_delay_ms
from ..ai.types import AssistantMessage, SystemMessage, Tool, ToolCall, TranscriptContext, UserMessage
from ..ai.providers import get_default_stream_fn
from ..ai.utils import normalize_context
from ..agent.compaction import is_context_overflow
from .context import ContextRange, ContextView, read_context_from
from .entries import ASSISTANT_ENTRY, USER_ENTRY
from .errors import StorageRejected, error_message
from .extensions import AgentDoc, ProviderDoc, Extension, HookApi, Registry, invoke
from .inbox import InboxDoc, apply_boundary, prepare_boundary, withdraw_queued_inputs
from .live import LiveDoc, clear_generation, clear_tools, end_run, finish_slot, generation_state, tool_slot
from .output import Progress
from .prompt import INSTRUCTIONS_KEY, plan_system_entries
from .session import Session
from .truncate import utf8_byte_length
from .types import (
    ProgressPolicy,
    TaskDefinition,
    TaskOutcome,
    TaskRecord,
    TaskState,
    define_task,
    record_to_dict,
    resolve_context_retention_ms,
    resolve_progress_policy,
)


class TaskRuntime:
    """Operations of one task invocation (compact port of TaskRuntime)."""

    def __init__(self, harness: "Harness", task_id: int, signal: asyncio.Event) -> None:
        self._harness = harness
        self.task_id = task_id
        self.signal = signal
        self._in_abort = False

    @property
    def aborted(self) -> bool:
        return self.signal.is_set()

    async def context(self, conversation_id: int, at: Optional[int] = None) -> ContextView:
        """Model context of one conversation during task execution, cut off at
        the visible entry ``at`` when given (TS 1.1.0 options shape
        ``context(conversationId, {at})``; the cached scan layer is shared with
        generation's own reads)."""

        return await self._harness.context(conversation_id, at)

    async def sleep(self, until_ms: int) -> None:
        """Sleep until the wall-clock deadline ``until_ms``; wakes early on abort.

        The abort mark routes through the scheduler: after a wait cut short by
        ``abort_task``, the next loop iteration invokes the definition's abort
        handler instead of the phase the task would resume.
        """
        delay = (until_ms - int(time.time() * 1000)) / 1000
        if delay <= 0:
            return
        tick = asyncio.ensure_future(asyncio.sleep(delay))
        guard = asyncio.ensure_future(self.signal.wait())
        try:
            await asyncio.wait({tick, guard}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            tick.cancel()
            guard.cancel()

    async def commit(self, change: Callable[[Any, TaskRecord], Optional[TaskState]]) -> None:
        """Commit after rereading the task; a returned state replaces its state.

        Abort routing lives in the scheduler: when the abort mark is set, the
        scheduler invokes the definition's abort handler instead of the next
        phase, so commits from inside an abort handler flow through here
        without re-entering it.
        """

        scope = _scope_for(self._harness, self.task_id)

        async def tx_change(tx):
            current = await tx.task(self.task_id)
            if current is None:
                raise RuntimeError(f"Task {self.task_id} vanished")
            if current.state.status in ("completing", "terminal"):
                return None
            # Reads must precede writes: snapshot the wait-for graph before the
            # callback runs, so a returned waiting state can be validated.
            graph = await _wait_graph(tx)
            next_state = change(tx, current)
            if hasattr(next_state, "__await__"):
                next_state = await next_state
            if next_state is None:
                return None
            if next_state.status in ("terminal", "completing") and next_state.outcome is None:
                raise ValueError(f"Task state {next_state.status!r} requires an outcome")
            if next_state.status == "waiting":
                _reject_wait_cycles(graph, current.id, list(next_state.on or []))
            if next_state.status == "terminal":
                # Terminal outcomes stage as completing; the scheduler finalizes
                # them once no ordinary owned work below is live (spec 5.5).
                replaced = dataclasses.replace(
                    current, state=TaskState(status="completing", outcome=next_state.outcome))
                from .types import write_task

                tx._writes.append(write_task(replaced))
                tx._mark_table_write()
                return None
            replaced = dataclasses.replace(current, state=next_state)
            from .types import write_task

            tx._writes.append(write_task(replaced))
            tx._mark_table_write()
            return None

        await self._harness.session.commit(tx_change, scope)


async def _scan_all(scan, query):
    items = []
    cursor = None
    while True:
        page = await scan(query, 1000, cursor)
        items.extend(page.items)
        if page.next is None:
            return items
        cursor = page.next


async def _wait_graph(tx):
    """Snapshot of the wait-for graph: (records by id, owner -> live children).

    Read before the caller's transaction writes; tasks created afterwards have
    no outgoing edges, so the snapshot stays exact for cycle detection.
    """
    records = await _scan_all(tx.scan_tasks, {})
    by_id = {record.id: record for record in records}
    owned: Dict[int, List[int]] = {}
    for record in records:
        # Background work never holds its owner (TS #waitCycle skips it), so it
        # is not an edge in the wait-for graph either.
        if (record.owner is not None and record.state.status != "terminal"
                and record.id != record.owner and not record.background):
            owned.setdefault(record.owner, []).append(record.id)
    return by_id, owned


def _reject_wait_cycles(graph, task_id: int, on_ids: List[int]) -> None:
    """Reject a wait that closes a cycle in the wait-for graph (TS #validateWait).

    Edges are wait joins (``state.on`` of waiting tasks) plus live owned work
    (non-terminal tasks owned by the node). Waiting on a task that can reach
    back would deadlock every task on the cycle forever, since a waiting task
    only resumes once its joins settle.
    """

    by_id, owned = graph
    if task_id in on_ids:
        raise ValueError(f"Task {task_id} cannot wait on itself")

    def edges(node: int) -> List[int]:
        record = by_id.get(node)
        out = []
        if record is not None and record.state.status == "waiting":
            out.extend(record.state.on or [])
        out.extend(owned.get(node, []))
        return out

    # BFS from the join targets over wait and owned-work edges; reaching the
    # waiting task again closes a cycle, and the error prints the whole loop.
    parent: Dict[int, int] = {target: task_id for target in on_ids}
    queue = list(on_ids)
    while queue:
        node = queue.pop(0)
        for nxt in edges(node):
            if nxt == task_id:
                path = [task_id]
                walk = node
                while walk != task_id:
                    path.append(walk)
                    walk = parent[walk]
                path.append(task_id)
                chain = " -> ".join(str(step) for step in path)
                raise ValueError(f"Waiting on {on_ids} would close a wait cycle: {chain}")
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)


def _scope_for(harness: "Harness", task_id: int):
    from .transaction import TransactionScope

    return TransactionScope(task_id=task_id)


def _scope_for_conversation(conversation_id: int):
    from .transaction import TransactionScope

    return TransactionScope(conversation_id=conversation_id)


def _safe_now(now: Callable[[], int], on_report: Callable[[BaseException], None]) -> Callable[[], int]:
    """The host clock, falling back to the wall clock when it throws.

    A broken clock must not stop commits: the clock is read on every task
    change, so the first throw is reported once and later ones are not
    (port of TS safeNow).
    """
    reported = {"value": False}

    def read() -> int:
        try:
            return int(now())
        except Exception as exc:  # noqa: BLE001 - the clock must not fail its caller
            if not reported["value"]:
                # Set first: a report handler that reads the clock again must
                # not report again.
                reported["value"] = True
                try:
                    on_report(exc)
                except Exception:  # noqa: BLE001 - onReport is the last place an error goes
                    pass
            return int(time.time() * 1000)

    return read


class Harness:
    """Durable agent harness over one Session (compact port of Harness)."""

    def __init__(self, session: Session, model: Model, retry: Optional[RetryPolicy] = None,
                 progress: Optional[ProgressPolicy] = None,
                 on_report: Optional[Callable[[BaseException], None]] = None,
                 registry: Optional[Registry] = None, extensions: Optional[List[Extension]] = None,
                 conversation_created: Optional[Callable] = None,
                 steering_mode: str = "one-at-a-time", follow_up_mode: str = "one-at-a-time",
                 now: Optional[Callable[[], int]] = None,
                 context_retention_ms: Optional[float] = None) -> None:
        self.session = session
        self.model = model
        self.retry = retry
        # Wall-clock ms source for timers (TS HarnessOptions.now); the Session
        # keeps its own injection for task lifecycle stamps. A broken clock
        # must not stop commits: the first throw is reported, later reads fall
        # back to the wall clock silently (TS safeNow).
        self.now = _safe_now(now or (lambda: int(time.time() * 1000)),
                             on_report or self._report_progress_error)
        # Per-conversation context cache retention (TS settings.contextRetentionMs).
        self.context_retention_ms = resolve_context_retention_ms(context_retention_ms)
        # Accepts a ProgressPolicy or the settings mapping (validated field by field).
        self.progress = resolve_progress_policy(progress)
        self.on_report = on_report or self._report_progress_error
        self.definitions: Dict[str, TaskDefinition] = {}
        self.tools: Dict[str, Any] = {}
        self.registry = registry or Registry(extensions)
        self.default_extensions = [extension.name for extension in (extensions or [])]
        self.steering_mode, self.follow_up_mode = steering_mode, follow_up_mode
        self.conversation_created = conversation_created
        previous = getattr(session, "_harness_creation_hook", None)
        if previous in session._creation_hooks:
            session._creation_hooks.remove(previous)
        session._harness_creation_hook = self._conversation_created
        session._creation_hooks.append(self._conversation_created)
        self.register_definition(generation_definition())
        from .tasks import compact_definition

        self.register_definition(compact_definition())
        from .tasks import tool_definition
        self.register_definition(tool_definition())
        # Created on first use inside the running loop: on Python 3.9
        # asyncio.Event binds to the loop current at construction, and
        # harnesses are often built before the loop that runs them.
        self.__stop: Optional[asyncio.Event] = None
        self._signals: Dict[int, asyncio.Event] = {}
        # Tool progress waiters survive from the execute phase to the settle
        # transaction, which is the commit that makes the result visible.
        self._progress_waiters: Dict[int, List[asyncio.Future]] = {}
        # Per-conversation context cache (F-AG-09): the last read range plus
        # its derived view, reused while the head marker stays put.
        self._context_cache: Dict[int, ContextRange] = {}
        self._context_expiry_tasks: Dict[int, asyncio.Task] = {}

    @property
    def _stop(self) -> asyncio.Event:
        if self.__stop is None:
            self.__stop = asyncio.Event()
        return self.__stop

    @staticmethod
    def _report_progress_error(error: BaseException) -> None:
        """Default reporter: a failed progress commit must not look like success."""

        logging.getLogger(__name__).warning("durable progress commit failed: %s", error)

    def hold_progress_waiters(self, task_id: int, waiters: List[asyncio.Future]) -> None:
        if waiters:
            self._progress_waiters[task_id] = waiters

    def settle_progress_waiters(self, task_id: int, error: Optional[BaseException] = None) -> None:
        """Resolve (or reject) waiters with the task's final transaction outcome."""

        for waiter in self._progress_waiters.pop(task_id, []):
            if waiter.done():
                continue
            if error is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(error)

    # -- registration ---------------------------------------------------------

    async def _conversation_created(self, tx, record) -> None:
        for definition in (LiveDoc, InboxDoc, AgentDoc, ProviderDoc):
            draft = await tx.doc(definition, record.id)
            if definition is AgentDoc:
                draft.value["extensions"] = list(self.default_extensions)
                if record.parent:
                    parent = record.parent
                    address = {"kind": AgentDoc.kind, "scope": {"kind": "conversation", "conversation_id": parent["conversation_id"]}, "key": None}
                    visible = await self.session.storage.entry_visible(parent["conversation_id"], parent["at"])
                    found = await self.session.storage.find_document(address, visible[1])
                    if found:
                        stored = await self.session.storage.document(found.id, visible[1])
                        draft.value = copy.deepcopy(stored.value)
            if definition is ProviderDoc:
                draft.value["session_id"] = str(uuid.uuid4())
            draft.dirty = True
        for extension in self.registry.installed():
            for definition in extension.documents:
                if definition.scope == "conversation":
                    draft = await tx.doc(definition, record.id)
                    if record.parent and definition.fork != "initial":
                        parent = record.parent
                        visible = await self.session.storage.entry_visible(parent["conversation_id"], parent["at"])
                        at = "current" if definition.fork == "current" else visible[1]
                        address = {"kind": definition.kind, "scope": {"kind": "conversation", "conversation_id": parent["conversation_id"]}, "key": None}
                        found = await self.session.storage.find_document(address, at)
                        if found:
                            stored = await self.session.storage.document(found.id, at)
                            draft.value = copy.deepcopy(stored.value)
                    draft.dirty = True
        if self.conversation_created:
            await invoke(self.conversation_created, tx, record)

    async def configure(self, conversation_id: int, extensions: Optional[List[str]] = None,
                        instructions: Optional[str] = None) -> None:
        async def change(tx):
            if await tx.conversation(conversation_id) is None:
                raise StorageRejected(f"Unknown conversation: {conversation_id}")
            agent = await tx.doc(AgentDoc, conversation_id)
            if extensions is not None:
                if any(self.registry.extension(name) is None for name in extensions):
                    raise ValueError("Unknown extension")
                agent.value["extensions"] = list(extensions)
            if instructions is not None:
                agent.value["instructions"] = instructions
            agent.dirty = True
        await self.session.commit(change)

    async def selected_extensions(self, conversation_id: int) -> List[Extension]:
        agent = await self.session.snapshot(AgentDoc, conversation_id) or {}
        result = []
        for name in agent.get("extensions", self.default_extensions):
            extension = self.registry.extension(name)
            if extension is None:
                raise ValueError(f"Missing extension: {name}")
            result.append(extension)
        return result

    async def run_hooks(self, name: str, runtime: TaskRuntime, *args):
        task = await self._get_task(runtime.task_id)
        result = None
        for extension in await self.selected_extensions(task.conversation_id):
            callback = extension.hooks.get(name)
            if callback:
                value = await invoke(callback, *args, HookApi(runtime), runtime)
                if value is not None:
                    result = value
                    if name in ("on_yield", "before_compact") or (isinstance(value, dict) and value.get("block")):
                        break
                    if name == "before_request":
                        args = (value,)
                    elif name == "before_tool" and "arguments" in value:
                        args = ({**args[0], "arguments": value["arguments"]},)
                    elif name == "after_tool":
                        args = (args[0], value)
        return result

    async def resolved_tools(self, conversation_id: int) -> dict:
        tools = dict(self.tools)
        selected = await self.selected_extensions(conversation_id)
        for extension in selected:
            for tool in extension.tools:
                if tool.name in tools:
                    raise ValueError(f"Duplicate tool: {tool.name}")
                tools[tool.name] = tool
        for extension in selected:
            for name, wrap in extension.wraps.items():
                if name in tools:
                    tools[name] = wrap(tools[name])
        return tools

    def register_definition(self, definition: TaskDefinition) -> None:
        self.definitions[definition.name] = definition

    def register_tools(self, tools) -> None:
        """Expose agent tools to the ``pi.tool`` durable task."""
        self.tools = dict(tools)
        from .tasks import tool_definition

        if "pi.tool" not in self.definitions:
            self.register_definition(tool_definition())

    # -- conversations ---------------------------------------------------------

    async def create_conversation(self) -> int:
        async def change(tx):
            record = await tx.create_conversation({"kind": "ownerless"})
            return record.id

        return await self.session.commit(change)

    async def submit_input(self, conversation_id: int, text: str, request_id: Optional[str] = None,
                           mode: str = "steer") -> int:
        """Admit one user input durably: submission, entry, and its run task
        commit together before anything becomes visible."""

        async def change(tx):
            if await tx.conversation(conversation_id) is None:
                raise StorageRejected(f"Unknown conversation: {conversation_id}")
            if mode not in ("steer", "followUp"):
                raise ValueError("input mode must be steer or followUp")
            existing = await tx.submission_by_request(conversation_id, request_id) if request_id else None
            tasks = await _scan_all(tx.scan_tasks, {"conversation_id": conversation_id})
            if existing:
                return next((task.id for task in tasks if existing.id in (task.input or {}).get("submission_ids", [(task.input or {}).get("submission_id")])), existing.id)
            boundary = await prepare_boundary(tx, conversation_id, self.steering_mode, self.follow_up_mode)
            busy = any(task.kind == "pi.generation" and task.state.status not in ("terminal", "completing") for task in tasks)
            submission = await tx.create_submission(
                {
                    "conversation_id": conversation_id,
                    "type": "input",
                    "status": "queued",
                    "request_id": request_id,
                }
            )
            if busy:
                boundary.inbox.value["items"].append({"id": submission.id, "mode": mode, "content": text})
                boundary.inbox.dirty = True
                return submission.id
            boundary.inbox.value["items"].append({"id": submission.id, "mode": mode, "content": text})
            boundary.inbox.dirty = True
            placed = await apply_boundary(tx, boundary, "final")
            return await self._schedule_placed(tx, conversation_id, placed["users"])

        return await self.session.commit(change, _scope_for_conversation(conversation_id))

    async def transcript(self, conversation_id: int) -> List[Any]:
        """Visible entries, oldest first (fork-aware, active context only)."""

        async def change(tx):
            items = list(reversed(await _scan_all(tx.scan_entries, {"conversation_id": conversation_id})))
            for entry in reversed(items):
                if entry.head is not None:
                    start = entry.head
                    return [e for e in items if e.id >= start]
            return items

        return await self.session.commit(change, _scope_for_conversation(conversation_id))

    async def context(self, conversation_id: int, at: Optional[int] = None) -> ContextView:
        """Committed model context of one conversation (port of TS
        ``Conversation.context(context, {at})``, #10512): the active transcript
        and its derived messages, cut off at the visible entry ``at`` when
        given — the same view ``fork(at)`` would start with, without creating
        a conversation. Reuses the per-conversation cached scan range."""

        view, context_range = await read_context_from(
            self.session, self.session.storage, conversation_id, at,
            self._context_cache.get(conversation_id))
        self._remember_context_range(conversation_id, context_range)
        return view

    async def context_entries(self, conversation_id: int, at: Optional[int] = None) -> List[Any]:
        """Raw active-context entries, oldest first — the transcript range the
        context view derives from, read through the same cached scan layer
        (generation/compaction entry scans)."""

        view, context_range = await read_context_from(
            self.session, self.session.storage, conversation_id, at,
            self._context_cache.get(conversation_id))
        self._remember_context_range(conversation_id, context_range)
        return [] if context_range is None else list(context_range.entries)

    def _remember_context_range(self, conversation_id: int, context_range: Optional[ContextRange]) -> None:
        if context_range is None:
            self._context_cache.pop(conversation_id, None)
            return
        self._context_cache[conversation_id] = context_range
        # A read does not extend the idle window: the pending expiry (if any)
        # keeps running, as TS keeps the run's idleSince across reads.

    def _schedule_context_expiry(self, conversation_id: int) -> None:
        """Schedule this conversation's idle expiry unless one is pending."""

        if conversation_id in self._context_expiry_tasks:
            return
        delay = self.context_retention_ms / 1000
        if delay <= 0:
            return

        async def expire() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            self._context_expiry_tasks.pop(conversation_id, None)
            self._context_cache.pop(conversation_id, None)

        self._context_expiry_tasks[conversation_id] = asyncio.ensure_future(expire())

    async def _settle_context_idle(self) -> None:
        """Expire cached context ranges of idle conversations (port of TS
        #settleIdle): busy conversations keep theirs, retention 0 drops at
        once, otherwise a timer drops the range after the retention window."""

        if not self._context_cache:
            return
        live = set()
        for status in ("pending", "running", "waiting", "completing"):
            for record in await _scan_all(self.session.storage.scan_tasks, {"status": status}):
                if record.conversation_id is not None:
                    live.add(record.conversation_id)
        for conversation_id in list(self._context_cache):
            if conversation_id in live:
                continue
            if self.context_retention_ms <= 0:
                self._context_cache.pop(conversation_id, None)
                continue
            self._schedule_context_expiry(conversation_id)

    def _cancel_context_expiry(self) -> None:
        for task in self._context_expiry_tasks.values():
            task.cancel()
        self._context_expiry_tasks.clear()

    async def live(self, conversation_id: int) -> Dict[str, Any]:
        """Committed live state: the generated partial and running tool slots."""

        value = await self.session.snapshot(LiveDoc, conversation_id)
        return value or {}

    async def inbox(self, conversation_id: int) -> dict:
        return await self.session.snapshot(InboxDoc, conversation_id) or {"items": []}

    async def submit_write(self, conversation_id: int, entry: dict, request_id: Optional[str] = None) -> int:
        async def change(tx):
            if await tx.conversation(conversation_id) is None:
                raise StorageRejected(f"Unknown conversation: {conversation_id}")
            existing = await tx.submission_by_request(conversation_id, request_id) if request_id else None
            if existing:
                return existing.id
            tasks = await _scan_all(tx.scan_tasks, {"conversation_id": conversation_id})
            boundary = await prepare_boundary(tx, conversation_id, self.steering_mode, self.follow_up_mode)
            submission = await tx.create_submission({"conversation_id": conversation_id, "type": "write", "request_id": request_id})
            boundary.inbox.value["items"].append({"id": submission.id, "mode": "write", "entry": ai_to_json(entry)})
            boundary.inbox.dirty = True
            if not any(task.kind == "pi.generation" and task.state.status not in ("terminal", "completing") for task in tasks):
                placed = await apply_boundary(tx, boundary, "final")
                await self._schedule_placed(tx, conversation_id, placed["users"])
            return submission.id
        return await self.session.commit(change, _scope_for_conversation(conversation_id))

    async def _schedule_placed(self, tx, conversation_id: int, users: list) -> Optional[int]:
        if not users:
            return None
        task = await tx.create_task(self.definitions["pi.generation"],
                                   {"submission_id": users[0]["id"], "entry_id": users[0]["entry"],
                                    "submission_ids": [user["id"] for user in users]},
                                   {"ownership": {"kind": "conversation"}, "conversation_id": conversation_id})
        return task.id

    async def submission(self, submission_id: int):
        async def change(tx):
            return await tx._latest_submission(submission_id)
        return await self.session.commit(change)

    async def withdraw_submission(self, submission_id: int) -> None:
        async def change(tx):
            submission = await tx._latest_submission(submission_id)
            inbox = await tx.doc(InboxDoc, submission.conversation_id)
            if submission.status != "queued":
                return
            inbox.value["items"] = [item for item in inbox.value["items"] if item["id"] != submission_id]
            inbox.dirty = True
            await tx.settle_submission(submission_id, {"status": "unanswered", "reason": "aborted"})
        await self.session.commit(change)

    async def abort_conversation(self, conversation_id: int) -> None:
        async def change(tx):
            tasks = await _scan_all(tx.scan_tasks, {"conversation_id": conversation_id})
            await withdraw_queued_inputs(tx, conversation_id)
            return [task.id for task in tasks if task.state.status not in ("completing", "terminal")]
        for task_id in await self.session.commit(change):
            await self.abort_task(task_id)

    async def submit_tool_task(self, conversation_id: int, tool: str, arguments: dict,
                               call_id: Optional[str] = None) -> int:
        """Submit a durable tool execution for a conversation."""

        if "pi.tool" not in self.definitions:
            raise RuntimeError("Register tools first: harness.register_tools(...)")
        async def change(tx):
            if await tx.conversation(conversation_id) is None:
                raise StorageRejected(f"Unknown conversation: {conversation_id}")
            task = await tx.create_task(
                self.definitions["pi.tool"],
                {"tool": tool, "arguments": arguments, "call_id": call_id},
                {"ownership": {"kind": "conversation"}, "conversation_id": conversation_id},
            )
            return task.id

        return await self.session.commit(change, _scope_for_conversation(conversation_id))

    async def submit_compact_task(self, conversation_id: int) -> int:
        """Submit a pi.compact task that summarizes and shrinks the context."""

        async def change(tx):
            if await tx.conversation(conversation_id) is None:
                raise StorageRejected(f"Unknown conversation: {conversation_id}")
            task = await tx.create_task(
                self.definitions["pi.compact"],
                {"conversation_id": conversation_id},
                {"ownership": {"kind": "conversation"}, "conversation_id": conversation_id},
            )
            return task.id

        return await self.session.commit(change, _scope_for_conversation(conversation_id))

    async def list_tasks(self, conversation_id: Optional[int] = None) -> List[TaskRecord]:
        async def change(tx):
            query = {"conversation_id": conversation_id} if conversation_id is not None else {}
            return await _scan_all(tx.scan_tasks, query)

        return await self.session.commit(change)

    # -- abort ------------------------------------------------------------------

    async def abort_task(self, task_id: int) -> None:
        async def change(tx):
            from .types import write_task

            current = await tx.task(task_id)
            if current is None or current.state.status in ("completing", "terminal"):
                return
            marked = dataclasses.replace(current, abort_requested=True)
            tx._writes.append(write_task(marked))
            tx._mark_table_write()

        await self.session.commit(change, _scope_for(self, task_id))
        # Wake a running phase (e.g. a retry backoff sleep) so the abort mark is
        # honored promptly instead of after the full wait.
        signal = self._signals.get(task_id)
        if signal is not None:
            signal.set()

    # -- scheduler ---------------------------------------------------------------

    async def run_pending_tasks(self, concurrency: int = 1) -> int:
        """Run every schedulable task to quiescence; returns tasks executed.

        With ``concurrency > 1`` up to that many pending tasks run at once.
        The batch takes at most one task per conversation, so tasks that share
        a conversation serialize while other conversations still run in
        parallel. Overlapping scheduler calls on the same Session serialize,
        including calls from different Harness instances.
        """

        # Recovery is safe only after the previous scheduler has finished or
        # cancelled all its phase handlers. Share ownership across harnesses
        # over this session, not just concurrent calls on this harness.
        async with self.session._scheduler_lock:
            return await self._drain_pending_tasks(concurrency)

    async def _drain_pending_tasks(self, concurrency: int) -> int:

        executed = 0
        await self._recover_running_tasks()
        self._cancel_context_expiry()
        try:
            while not self._stop.is_set():
                progressed = False
                progressed = await self._resume_waiting_tasks() or progressed
                progressed = await self._finalize_completing() or progressed
                batch = await self._pending_batch(max(1, concurrency))
                if batch:
                    results = await asyncio.gather(*[self._run_claimed(task) for task in batch])
                    executed += sum(results)
                    progressed = True
                if not progressed:
                    break
        finally:
            # Idle expiry runs even when the drain unwinds through an error.
            await self._settle_context_idle()
        return executed

    async def _run_claimed(self, task: TaskRecord) -> int:
        """Re-check one batched task and run it if still pending."""
        current = await self._get_task(task.id)
        if current is None or current.state.status != "pending":
            return 0
        await self._run_task(current)
        return 1

    async def _pending_batch(self, limit: int) -> List[TaskRecord]:
        async def change(tx):
            page = await tx.scan_tasks({"status": "pending"}, max(limit, 1) * 8)
            return page.items

        tasks = await self.session.commit(change)
        batch: List[TaskRecord] = []
        conversations = set()
        for task in tasks:
            conversation = task.conversation_id
            if conversation is not None:
                if conversation in conversations:
                    continue
                conversations.add(conversation)
            batch.append(task)
            if len(batch) >= limit:
                break
        return batch

    async def _next_pending_task(self) -> Optional[TaskRecord]:
        batch = await self._pending_batch(1)
        return batch[0] if batch else None

    async def _recover_running_tasks(self) -> None:
        """Recover abandoned running tasks under exclusive scheduler ownership.

        This covers reopening and retrying after scheduler cancellation.
        """

        async def change(tx):
            from .types import write_task

            records = await _scan_all(tx.scan_tasks, {"status": "running"})
            for record in records:
                replaced = dataclasses.replace(
                    record, state=TaskState(status="pending", checkpoint=record.state.checkpoint))
                tx._writes.append(write_task(replaced))
                tx._mark_table_write()
            return len(records)

        await self.session.commit(change)

    async def _run_task(self, task: TaskRecord) -> None:
        definition = self.definitions.get(task.kind)
        for extension in self.registry.installed():
            for candidate in extension.tasks:
                if candidate.name == task.kind:
                    definition = candidate
        if definition is None:
            await self._fault_or_orphan(task, "orphaned", f"No definition registered for task kind {task.kind}")
            return

        signal = asyncio.Event()
        self._signals[task.id] = signal
        runtime = TaskRuntime(self, task.id, signal)
        runtime.conversation_id = task.conversation_id
        try:
            # Reserve: pending -> running. On reopen, running resets to pending.
            # A task that cannot be reserved (for example a version migration)
            # faults here instead of escaping the scheduler on every cycle.
            reserved = await self._reserve(task.id, definition, task)
            if reserved is None:
                return
            while True:
                current = await self._get_task(task.id)
                if current is None or current.state.status in ("completing", "terminal"):
                    return
                if current.abort_requested:
                    await self._abort_now(current, definition)
                    return
                checkpoint = current.state.checkpoint or {}
                phase = checkpoint.get("phase")
                handler = definition.phases.get(phase)
                if handler is None:
                    await self._fault_or_orphan(current, "faulted", f"Task {current.kind} has no phase {phase!r}")
                    return
                phase_task = asyncio.create_task(handler(current, runtime, self))
                abort_wait = asyncio.create_task(signal.wait())
                try:
                    done, _ = await asyncio.wait((phase_task, abort_wait), return_when=asyncio.FIRST_COMPLETED)
                    if phase_task not in done:
                        phase_task.cancel()
                        await asyncio.gather(phase_task, return_exceptions=True)
                        latest = await self._get_task(task.id)
                        if latest is not None:
                            await self._abort_now(latest, definition)
                        return
                    await phase_task
                finally:
                    abort_wait.cancel()
                    if not phase_task.done():
                        phase_task.cancel()
                    await asyncio.gather(phase_task, abort_wait, return_exceptions=True)
                after = await self._get_task(task.id)
                if after is None or after.state.status in ("completing", "terminal", "waiting"):
                    return
                if after.state.checkpoint == current.state.checkpoint:
                    # No durable progress: contract violation (spec 5.5).
                    await self._fault_or_orphan(after, "faulted", f"Phase {phase!r} made no durable progress")
                    return
        except Exception as exc:  # noqa: BLE001 - encoded as a durable outcome
            current = await self._get_task(task.id)
            if current is not None and current.state.status not in ("completing", "terminal"):
                await self._fault_or_orphan(
                    current, "faulted", f"{type(exc).__name__}: {error_message(exc)}"
                )
        finally:
            self._signals.pop(task.id, None)

    async def _reserve(self, task_id: int, definition: TaskDefinition, record: TaskRecord) -> Optional[TaskRecord]:
        """Claim a pending task (or recover a stale running one), migrating input."""

        async def change(tx):
            from .types import write_task

            current = await tx.task(task_id)
            if current is None or current.state.status != "pending":
                return None
            input_value = current.input
            checkpoint = current.state.checkpoint
            if current.version < definition.version:
                raise RuntimeError(
                    f"Task {current.kind} stored by version {current.version} needs migration"
                )
            replaced = dataclasses.replace(
                current, state=TaskState(status="running", checkpoint=checkpoint))
            tx._writes.append(write_task(replaced))
            tx._mark_table_write()
            return replaced

        scope = _scope_for(self, task_id)
        return await self.session.commit(change, scope)

    async def _get_task(self, task_id: int) -> Optional[TaskRecord]:
        async def change(tx):
            return await tx.task(task_id)

        return await self.session.commit(change, _scope_for(self, task_id))

    async def _abort_now(self, task: TaskRecord, definition: TaskDefinition) -> None:
        """Run the abort handler outside any transaction (mirrors the TS
        scheduler's #runAbort): its runtime.commit opens its own commit on
        the line. Invoking the handler inside session.commit would deadlock
        on the non-reentrant line lock."""

        signal = asyncio.Event()
        signal.set()
        runtime = TaskRuntime(self, task.id, signal)
        runtime._in_abort = True
        try:
            current = await self._get_task(task.id)
            if current is not None and current.state.status not in ("completing", "terminal"):
                if (current.state.checkpoint or {}).get("phase") == "settle":
                    await definition.phases["settle"](current, runtime, self)
                else:
                    await definition.abort(current, runtime, self)
        except Exception as exc:  # noqa: BLE001
            current = await self._get_task(task.id)
            if current is not None and current.state.status not in ("completing", "terminal"):
                await self._fault_or_orphan(current, "faulted", f"abort failed: {error_message(exc)}")

    async def _run_abort_handler(self, tx, current: TaskRecord) -> None:
        definition = self.definitions.get(current.kind)
        if definition is None:
            await _stage_terminal(tx, current, TaskOutcome(status="orphaned", reason="definition missing"))
            return
        signal = asyncio.Event()
        runtime = TaskRuntime(self, current.id, signal)
        runtime._in_abort = True
        await definition.abort(current, runtime, self)

    async def _finalize_completing(self) -> bool:
        async def change(tx):
            from .types import write_task

            completing = await _scan_all(tx.scan_tasks, {"status": "completing"})
            all_tasks = await _scan_all(tx.scan_tasks, {})
            # Background children never block their owner's finalization
            # (TS #ownedLive skips them).
            live_owner_ids = {task.owner for task in all_tasks
                              if task.owner is not None and task.state.status != "terminal"
                              and not task.background}
            finalized = 0
            for record in completing:
                if record.id in live_owner_ids:
                    continue
                terminal = dataclasses.replace(
                    record, state=TaskState(status="terminal", outcome=record.state.outcome))
                tx._writes.append(write_task(terminal))
                tx._mark_table_write()
                finalized += 1
            return finalized

        count = await self.session.commit(change)
        return count > 0

    async def _resume_waiting_tasks(self) -> bool:
        async def change(tx):
            from .types import write_task

            records = await _scan_all(tx.scan_tasks, {"status": "waiting"})
            replacements = []
            resumed = 0
            for record in records:
                # Abort must reach the scheduler's normal abort handler even
                # when the joined tasks never finish. Keep the checkpoint so
                # an already committed settle phase still takes precedence.
                ids = [] if record.abort_requested else (record.state.on or [])
                terminal_tasks = []
                for tid in ids:
                    other = await tx.task(tid)
                    if other is None or other.state.status != "terminal":
                        continue
                    terminal_tasks.append(other)
                if record.state.policy == "failFast":
                    # Persisted malformed terminal states are failed dependencies,
                    # never successful joins or a reason to stall every tick.
                    failed = [t for t in terminal_tasks
                              if t.state.outcome is None or t.state.outcome.status != "completed"]
                    if not failed and len(terminal_tasks) != len(ids):
                        continue
                    if failed:
                        outcome = TaskOutcome(status="aborted", reason=f"join failed: task {failed[0].id}")
                        replaced = dataclasses.replace(
                            record, state=TaskState(status="terminal", outcome=outcome))
                        replacements.append(replaced)
                        resumed += 1
                        continue
                if len(terminal_tasks) != len(ids):
                    continue
                # allSettled: resume at the stored checkpoint.
                replaced = dataclasses.replace(
                    record, state=TaskState(status="pending", checkpoint=record.state.checkpoint))
                replacements.append(replaced)
                resumed += 1
            for replaced in replacements:
                tx._writes.append(write_task(replaced))
                tx._mark_table_write()
            return resumed

        resumed = await self.session.commit(change)
        return resumed > 0

    async def _fault_or_orphan(self, task: TaskRecord, status: str, detail: str) -> None:
        async def change(tx):
            current = await tx.task(task.id)
            if current is None or current.state.status in ("completing", "terminal"):
                return
            if status == "orphaned":
                outcome = TaskOutcome(status="orphaned", reason=detail)
            else:
                outcome = TaskOutcome(status="faulted", error={"message": detail})
            await _cleanup_live_for_terminal(tx, current)
            await _stage_terminal(tx, current, outcome)

        try:
            await self.session.commit(change, _scope_for(self, task.id))
        except Exception as commit_exc:  # noqa: BLE001
            # A fault write that cannot commit leaves the task running, to be
            # reserved and run again in this process. End the invocation and
            # fail the Session instead, so later work is rejected
            # (TS #step catch → #failSession).
            self.session.fail(commit_exc)
            self.settle_progress_waiters(task.id, RuntimeError(detail))
            return
        self.settle_progress_waiters(task.id, RuntimeError(detail))


async def _cleanup_live_for_terminal(tx, current: TaskRecord) -> None:
    """Clear a task's live presentation before its terminal state is written."""

    if current.kind == "pi.generation":
        await _convert_generation_partial(tx, current)
        return
    if current.kind == "pi.tool":
        live = await tx.doc(LiveDoc, current.conversation_id)
        slot = tool_slot(live.value, current.id)
        if slot is not None:
            finish_slot(slot, None)
            live.dirty = True


async def _stage_terminal(tx, current: TaskRecord, outcome: TaskOutcome) -> None:
    from .types import write_task

    replaced = dataclasses.replace(
        current, state=TaskState(status="completing", outcome=outcome))
    tx._writes.append(write_task(replaced))
    tx._mark_table_write()


# ---------------------------------------------------------------------------
# pi.generation: one durable model turn
# ---------------------------------------------------------------------------


def generation_definition(stream_fn=None) -> TaskDefinition:
    """The built-in generation task (compact port of GenerationTask).

    Phases: prepare (transcript -> LLM context), stream (provider request,
    response committed as a checkpoint), retry (transient-error backoff, when
    a retry policy is configured), settle (assistant entry + submission
    settlement). A crash between phases resumes from the durable checkpoint.
    """

    async def prepare(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        input_value = task.input
        conversation_id = task.conversation_id
        attempt = int((task.state.checkpoint or {}).get("attempt") or 1)
        entries = await harness.context_entries(conversation_id)
        # Desired prompt state: extension sections + the instructions pseudo
        # section, and the declared tools. Planned as positional pi.system
        # entries so the prompt lives in the transcript (fork/audit visible).
        agent = await harness.session.snapshot(AgentDoc, conversation_id) or {}
        section_map: Dict[str, str] = {}
        for extension in await harness.selected_extensions(conversation_id):
            for name, render in extension.sections.items():
                text = await invoke(render, HookApi(runtime), runtime)
                # None omits the section (TS renderSections); "" still emits nothing.
                if text is not None:
                    section_map[name] = str(text)
        if agent.get("instructions"):
            section_map[INSTRUCTIONS_KEY] = agent["instructions"]
        declared = [Tool(name=t.name, description=t.description, parameters=t.parameters,
                         hosted=getattr(t, "hosted", None))
                    for t in (await harness.resolved_tools(conversation_id)).values() if t.declared]
        drafts, messages = plan_system_entries(entries, section_map, declared)
        stored = [ai_to_json(m) for m in messages]

        async def change(tx, current):
            for draft in drafts:
                await tx.append_entry(conversation_id, draft)
            return TaskState(
                status="running", checkpoint={"phase": "stream", "messages": stored, "attempt": attempt}
            )

        await runtime.commit(change)

    async def stream(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        checkpoint = task.state.checkpoint
        messages = [message_from_json(m) for m in (checkpoint.get("messages") or [])]
        replacement = await harness.run_hooks("before_request", runtime, {"messages": messages})
        if replacement is not None:
            messages = list(replacement["messages"])
        attempt = int(checkpoint.get("attempt") or 1)
        conversation_id = task.conversation_id
        latest: Dict[str, Any] = {"message": None}

        async def write_partial() -> int:
            message = latest["message"]
            if message is None:
                return 0

            async def change(tx, current):
                live = await tx.doc(LiveDoc, conversation_id)
                generation = live.value.setdefault("generation", {})
                generation["attempt"] = attempt
                generation["message"] = message
                live.dirty = True
                return None

            await runtime.commit(change)
            return utf8_byte_length(json.dumps(message, ensure_ascii=False))

        progress = Progress(write_partial, lambda exc: harness.on_report(exc),
                            harness.progress.partial_interval_ms)

        def on_partial(partial: AssistantMessage) -> None:
            if not partial.content:
                return
            # Copy synchronously: the provider keeps mutating its partial.
            latest["message"] = ai_to_json([copy.deepcopy(partial)])[0]
            progress.mark()

        try:
            # Tool declarations ride in the pi.system entries' tools_added and
            # are folded into the leading system message by normalize_context.
            response = await _call_stream_once(harness.model, messages, harness, on_partial)
            await harness.run_hooks("after_response", runtime, response)
        finally:
            # No stale partial lands after the outcome: wait for the in-flight
            # commit, then the settle/retry commit clears or replaces it.
            await progress.stop()
        # Transient failures retry with backoff before the response becomes
        # visible; an overflow is never retried (only compaction can fit it).
        if response.stop_reason == "error" and not is_context_overflow(response):
            policy = harness.retry
            if (
                policy is not None
                and policy.enabled
                and is_retryable_assistant_error(response)
                and attempt <= policy.max_retries
            ):
                until = int(time.time() * 1000) + retry_delay_ms(policy, attempt)

                async def retry_change(tx, current):
                    live = await tx.doc(LiveDoc, conversation_id)
                    clear_generation(live.value)
                    live.dirty = True
                    return TaskState(
                        status="running",
                        checkpoint={
                            "phase": "retry",
                            "attempt": attempt,
                            "until": until,
                            "messages": checkpoint.get("messages") or [],
                        },
                    )

                await runtime.commit(retry_change)
                return

        async def settle_change(tx, current):
            live = await tx.doc(LiveDoc, conversation_id)
            clear_generation(live.value)
            live.dirty = True
            return TaskState(
                status="running",
                checkpoint={"phase": "settle", "response": ai_to_json([response])[0]},
            )

        await runtime.commit(settle_change)

    async def retry(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        await runtime.sleep(int(task.state.checkpoint["until"]))
        # Re-prepare, like TS: sections and the tool loadout are rendered for the
        # retried request, and a changed prompt becomes a new pi.system entry.
        await runtime.commit(
            lambda tx, current: TaskState(
                status="running",
                checkpoint={
                    "phase": "prepare",
                    "attempt": int(current.state.checkpoint["attempt"]) + 1,
                },
            )
        )

    async def settle(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        response = message_from_json(task.state.checkpoint["response"])
        continuation = None
        calls = [block for block in response.content if isinstance(block, ToolCall)]
        if response.stop_reason not in ("error", "aborted") and not calls:
            continuation = await harness.run_hooks("on_yield", runtime, response)
        async def tx_change(tx, current):
            live = await tx.doc(LiveDoc, current.conversation_id)
            boundary = await prepare_boundary(tx, current.conversation_id, harness.steering_mode, harness.follow_up_mode)
            # The attempt is over: whether this settles the run or starts a tool
            # round, older slots are not part of the new presentation.
            end_run(live.value)
            live.dirty = True
            response = message_from_json(current.state.checkpoint["response"])
            entry = await tx.append_entry(
                current.conversation_id,
                {
                    "kind": ASSISTANT_ENTRY.kind,
                    "model": [response],
                    "data": {"stop_reason": response.stop_reason},
                },
            )
            if calls and response.stop_reason not in ("error", "aborted"):
                children = []
                slots = []
                for call in calls:
                    child = await tx.create_task(harness.definitions["pi.tool"],
                        {"tool": call.name, "arguments": call.arguments, "call_id": call.id},
                        {"ownership": {"kind": "conversation"}, "conversation_id": current.conversation_id})
                    children.append(child.id)
                    slots.append({"task_id": child.id, "call_id": call.id,
                                  "name": call.name, "status": "pending"})
                # The round's presentation belongs to the generation (TS
                # startToolRound): slots exist only while their round runs, and
                # the tool tasks never create them.
                live.value["tools"] = slots
                return TaskState(status="waiting", on=children, policy="allSettled",
                                 checkpoint={"phase": "after_tools", "assistant": entry.id, "tools": children})
            if continuation is not None:
                await tx.append_entry(current.conversation_id, {
                    "kind": USER_ENTRY.kind, "model": [UserMessage(content=continuation["continue"], timestamp=int(time.time() * 1000))]})
                return TaskState(status="running", checkpoint={"phase": "prepare"})
            for submission_id in current.input.get("submission_ids", [current.input["submission_id"]]):
                settlement = ({"status": "unanswered", "reason": response.stop_reason}
                              if response.stop_reason in ("error", "aborted") else {"status": "done", "answer": entry.id})
                await tx.settle_submission(submission_id, settlement)
            if response.stop_reason not in ("error", "aborted"):
                placed = await apply_boundary(tx, boundary, "final")
                await harness._schedule_placed(tx, current.conversation_id, placed["users"])
            return TaskState(
                status="terminal", outcome=TaskOutcome(status="completed", result={"entry_id": entry.id})
            )

        await runtime.commit(tx_change)

    async def after_tools(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        children = [await harness._get_task(task_id) for task_id in task.state.checkpoint["tools"]]
        results = [(child.state.outcome.result or {}).get("entry_id") for child in children
                   if child is not None and child.state.outcome is not None]
        if len(results) != len(children) or any(result is None for result in results):
            raise RuntimeError("A tool round ended without a result entry")
        await harness.run_hooks("after_tools", runtime, task.state.checkpoint["assistant"], results)
        async def change(tx, current):
            live = await tx.doc(LiveDoc, current.conversation_id)
            clear_tools(live.value)
            live.dirty = True
            boundary = await prepare_boundary(tx, current.conversation_id, harness.steering_mode, harness.follow_up_mode)
            placed = await apply_boundary(tx, boundary, "postTools")
            ids = list(current.input.get("submission_ids", [current.input["submission_id"]]))
            if placed["reset"]:
                for submission_id in ids:
                    await tx.settle_submission(submission_id, {"status": "unanswered", "reason": "reset"})
                await harness._schedule_placed(tx, current.conversation_id, placed["users"])
                return TaskState(status="terminal", outcome=TaskOutcome(status="completed", result={"reset": True}))
            ids.extend(user["id"] for user in placed["users"])
            current.input["submission_ids"] = ids
            # Preserve updated run membership in the same durable checkpoint.
            return TaskState(status="running", checkpoint={"phase": "prepare", "submission_ids": ids})
        await runtime.commit(change)

    async def abort(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        async def tx_change(tx, current):
            await _abort_generation(tx, current)
            return None

        await runtime.commit(tx_change)

    return define_task(
        name="pi.generation",
        version=1,
        initial=lambda input: {"phase": "prepare"},
        phases={"prepare": prepare, "stream": stream, "retry": retry, "settle": settle, "after_tools": after_tools},
        abort=abort,
    )


async def _abort_generation(tx, current: TaskRecord) -> None:
    await _convert_generation_partial(tx, current)
    submission_ids = (current.input or {}).get("submission_ids", [(current.input or {}).get("submission_id")])
    for submission_id in submission_ids:
        if submission_id is None:
            continue
        await tx.settle_submission(
            submission_id, {"status": "unanswered", "reason": "generation aborted"}
        )
    await _stage_terminal(tx, current, TaskOutcome(status="aborted", reason="aborted"))


async def _convert_generation_partial(tx, current: TaskRecord) -> None:
    """Keep a committed partial as an aborted assistant entry and end the run's live presentation."""

    if current.kind != "pi.generation":
        return
    live = await tx.doc(LiveDoc, current.conversation_id)
    partial = (generation_state(live.value) or {}).get("message")
    if partial is not None:
        message = message_from_json(partial)
        message.stop_reason = "aborted"
        await tx.append_entry(
            current.conversation_id,
            {
                "kind": ASSISTANT_ENTRY.kind,
                "model": [message],
                "data": {"stop_reason": "aborted", "partial": True},
            },
        )
    end_run(live.value)
    live.dirty = True


async def _stream_once(model: Model, messages: List[Any], harness: "Harness",
                       on_partial: Optional[Callable[[AssistantMessage], None]] = None,
                       tools: Optional[List[Any]] = None
                       ) -> AssistantMessage:
    if tools is not None:
        declarations = [Tool(name=tool.name, description=tool.description, parameters=tool.parameters,
                             hosted=getattr(tool, "hosted", None))
                        for tool in tools if tool.declared]
        messages = [SystemMessage(content="", timestamp=0, tools_added=declarations), *messages]
    system, rest = normalize_context(TranscriptContext(messages=messages))
    if system is not None:
        rest = [system, *rest]
    fn = get_default_stream_fn()
    final: Optional[AssistantMessage] = None
    async for event in fn(model, TranscriptContext(messages=rest), None):
        if event.type in ("done", "error"):
            final = event.message if event.type == "done" else event.error
            continue
        partial = getattr(event, "partial", None)
        if on_partial is not None and partial is not None and getattr(partial, "content", None):
            on_partial(partial)
    if final is None:
        raise RuntimeError("Provider stream ended without a final message")
    return final


async def _call_stream_once(model: Model, messages: List[Any], harness: "Harness",
                            on_partial: Callable[[AssistantMessage], None],
                            tools: Optional[List[Any]] = None) -> AssistantMessage:
    """Call ``_stream_once`` with partials when the (possibly patched) function accepts them.

    Existing tests and callers patch ``_stream_once(model, messages, harness)``;
    they keep working without the optional callback.
    """

    try:
        parameters = inspect.signature(_stream_once).parameters
        accepts = "on_partial" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    except (TypeError, ValueError):
        accepts = False
        parameters = {}
    kwargs = {"on_partial": on_partial} if accepts else {}
    if "tools" in parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        kwargs["tools"] = tools
    return await _stream_once(model, messages, harness, **kwargs)

"""Durable harness: scheduler and the built-in pi.generation task.

Python port of the core of ``packages/durable/src/harness``: task reservations,
the phase state machine, waiting/join policy, abort marks, crash recovery
(running resets to pending on reopen), and the generation task that commits a
model turn to the transcript before it becomes visible.

Porting deviations from the TS harness (documented in the README):
- hooks, extensions, documents-per-conversation (pi.live/pi.inbox/...),
  and full tool-task surfaces are not ported; ``pi.tool``/``pi.compact``
  live in ``tasks.py`` and the scheduler supports bounded concurrency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from ..ai.json_io import message_from_json, to_json as ai_to_json
from ..ai.models import Model
from ..ai.types import AssistantMessage, ToolCall, TranscriptContext, UserMessage
from ..ai.providers import get_default_stream_fn
from ..ai.utils import normalize_context
from .entries import ASSISTANT_ENTRY, SYSTEM_ENTRY, USER_ENTRY
from .errors import error_message
from .session import Session
from .types import (
    TaskDefinition,
    TaskOutcome,
    TaskRecord,
    TaskState,
    define_task,
    record_to_dict,
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
            next_state = change(tx, current)
            if hasattr(next_state, "__await__"):
                next_state = await next_state
            if next_state is None:
                return None
            if next_state.status == "terminal":
                # Terminal outcomes stage as completing; the scheduler finalizes
                # them once no ordinary owned work below is live (spec 5.5).
                replaced = TaskRecord(
                    id=current.id,
                    conversation_id=current.conversation_id,
                    kind=current.kind,
                    version=current.version,
                    input=current.input,
                    owner=current.owner,
                    background=current.background,
                    abort_requested=current.abort_requested,
                    state=TaskState(status="completing", outcome=next_state.outcome),
                )
                from .types import write_task

                tx._writes.append(write_task(replaced))
                tx._mark_table_write()
                return None
            replaced = TaskRecord(
                id=current.id,
                conversation_id=current.conversation_id,
                kind=current.kind,
                version=current.version,
                input=current.input,
                owner=current.owner,
                background=current.background,
                abort_requested=current.abort_requested,
                state=next_state,
                memos=current.memos,
            )
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


def _scope_for(harness: "Harness", task_id: int):
    from .transaction import TransactionScope

    return TransactionScope(task_id=task_id)


def _scope_for_conversation(conversation_id: int):
    from .transaction import TransactionScope

    return TransactionScope(conversation_id=conversation_id)


class Harness:
    """Durable agent harness over one Session (compact port of Harness)."""

    def __init__(self, session: Session, model: Model) -> None:
        self.session = session
        self.model = model
        self.definitions: Dict[str, TaskDefinition] = {}
        self.tools: Dict[str, Any] = {}
        self.register_definition(generation_definition())
        from .tasks import compact_definition

        self.register_definition(compact_definition())
        self._stop = asyncio.Event()

    # -- registration ---------------------------------------------------------

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

    async def submit_input(self, conversation_id: int, text: str, request_id: Optional[str] = None) -> int:
        """Admit one user input durably: submission, entry, and its run task
        commit together before anything becomes visible."""

        async def change(tx):
            submission = await tx.create_submission(
                {
                    "conversation_id": conversation_id,
                    "type": "input",
                    "status": "queued",
                    "request_id": request_id,
                }
            )
            entry = await tx.append_entry(
                conversation_id,
                {
                    "kind": USER_ENTRY.kind,
                    "model": [UserMessage(content=text, timestamp=int(time.time() * 1000))],
                },
            )
            tx_place = tx.place_submission(submission.id, entry.id)
            if hasattr(tx_place, "__await__"):
                await tx_place
            task = await tx.create_task(
                self.definitions["pi.generation"],
                {"submission_id": submission.id, "entry_id": entry.id},
                {"ownership": {"kind": "conversation"}, "conversation_id": conversation_id},
            )
            return task.id

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

    async def submit_tool_task(self, conversation_id: int, tool: str, arguments: dict,
                               call_id: Optional[str] = None) -> int:
        """Submit a durable tool execution for a conversation."""

        if "pi.tool" not in self.definitions:
            raise RuntimeError("Register tools first: harness.register_tools(...)")
        async def change(tx):
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
            marked = TaskRecord(
                id=current.id,
                conversation_id=current.conversation_id,
                kind=current.kind,
                version=current.version,
                input=current.input,
                owner=current.owner,
                background=current.background,
                abort_requested=True,
                state=current.state,
                memos=current.memos,
            )
            tx._writes.append(write_task(marked))
            tx._mark_table_write()

        await self.session.commit(change, _scope_for(self, task_id))

    # -- scheduler ---------------------------------------------------------------

    async def run_pending_tasks(self, concurrency: int = 1) -> int:
        """Run every schedulable task to quiescence; returns tasks executed.

        With ``concurrency > 1`` up to that many pending tasks run at once.
        The batch takes at most one task per conversation, so tasks that share
        a conversation serialize while other conversations still run in
        parallel.
        """

        executed = 0
        await self._recover_running_tasks()
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
        """Reset stale running tasks to pending (crash recovery on reopen)."""

        async def change(tx):
            from .types import write_task

            records = await _scan_all(tx.scan_tasks, {"status": "running"})
            for record in records:
                replaced = TaskRecord(
                    id=record.id,
                    conversation_id=record.conversation_id,
                    kind=record.kind,
                    version=record.version,
                    input=record.input,
                    owner=record.owner,
                    background=record.background,
                    abort_requested=record.abort_requested,
                    state=TaskState(status="pending", checkpoint=record.state.checkpoint),
                    memos=record.memos,
                )
                tx._writes.append(write_task(replaced))
                tx._mark_table_write()
            return len(records)

        await self.session.commit(change)

    async def _run_task(self, task: TaskRecord) -> None:
        definition = self.definitions.get(task.kind)
        if definition is None:
            await self._fault_or_orphan(task, "orphaned", f"No definition registered for task kind {task.kind}")
            return

        signal = asyncio.Event()
        runtime = TaskRuntime(self, task.id, signal)
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
                await handler(current, runtime, self)
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
            replaced = TaskRecord(
                id=current.id,
                conversation_id=current.conversation_id,
                kind=current.kind,
                version=current.version,
                input=input_value,
                owner=current.owner,
                background=current.background,
                abort_requested=current.abort_requested,
                state=TaskState(status="running", checkpoint=checkpoint),
                memos=current.memos,
            )
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
        runtime = TaskRuntime(self, task.id, signal)
        try:
            current = await self._get_task(task.id)
            if current is not None and current.state.status not in ("completing", "terminal"):
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

            page = await tx.scan_tasks({"status": "completing"}, 10)
            for record in page.items:
                terminal = TaskRecord(
                    id=record.id,
                    conversation_id=record.conversation_id,
                    kind=record.kind,
                    version=record.version,
                    input=record.input,
                    owner=record.owner,
                    background=record.background,
                    abort_requested=record.abort_requested,
                    state=TaskState(status="terminal", outcome=record.state.outcome),
                )
                tx._writes.append(write_task(terminal))
                tx._mark_table_write()
            return len(page.items)

        count = await self.session.commit(change)
        return count > 0

    async def _resume_waiting_tasks(self) -> bool:
        async def change(tx):
            from .types import write_task

            records = await _scan_all(tx.scan_tasks, {"status": "waiting"})
            replacements = []
            resumed = 0
            for record in records:
                ids = record.state.on or []
                terminal_tasks = []
                for tid in ids:
                    other = await tx.task(tid)
                    if other is None or other.state.status != "terminal":
                        continue
                    terminal_tasks.append(other)
                if record.state.policy == "failFast":
                    failed = [t for t in terminal_tasks if t.state.outcome.status != "completed"]
                    if not failed and len(terminal_tasks) != len(ids):
                        continue
                    if failed:
                        outcome = TaskOutcome(status="aborted", reason=f"join failed: task {failed[0].id}")
                        replaced = TaskRecord(
                            id=record.id,
                            conversation_id=record.conversation_id,
                            kind=record.kind,
                            version=record.version,
                            input=record.input,
                            owner=record.owner,
                            background=record.background,
                            abort_requested=record.abort_requested,
                            state=TaskState(status="terminal", outcome=outcome),
                        )
                        replacements.append(replaced)
                        resumed += 1
                        continue
                if len(terminal_tasks) != len(ids):
                    continue
                # allSettled: resume at the stored checkpoint.
                replaced = TaskRecord(
                    id=record.id,
                    conversation_id=record.conversation_id,
                    kind=record.kind,
                    version=record.version,
                    input=record.input,
                    owner=record.owner,
                    background=record.background,
                    abort_requested=record.abort_requested,
                    state=TaskState(status="pending", checkpoint=record.state.checkpoint),
                    memos=record.memos,
                )
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
            await _stage_terminal(tx, current, outcome)

        await self.session.commit(change, _scope_for(self, task.id))


async def _stage_terminal(tx, current: TaskRecord, outcome: TaskOutcome) -> None:
    from .types import write_task

    replaced = TaskRecord(
        id=current.id,
        conversation_id=current.conversation_id,
        kind=current.kind,
        version=current.version,
        input=current.input,
        owner=current.owner,
        background=current.background,
        abort_requested=current.abort_requested,
        state=TaskState(status="completing", outcome=outcome),
    )
    tx._writes.append(write_task(replaced))
    tx._mark_table_write()


# ---------------------------------------------------------------------------
# pi.generation: one durable model turn
# ---------------------------------------------------------------------------


def generation_definition(stream_fn=None) -> TaskDefinition:
    """The built-in generation task (compact port of GenerationTask).

    Phases: prepare (transcript -> LLM context), stream (provider request,
    response committed as a checkpoint), settle (assistant entry + submission
    settlement). A crash between phases resumes from the durable checkpoint.
    """

    async def prepare(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        input_value = task.input
        conversation_id = task.conversation_id
        entries = await harness.transcript(conversation_id)
        llm_messages: List[Any] = []
        for entry in entries:
            if entry.kind in (USER_ENTRY.kind, ASSISTANT_ENTRY.kind, SYSTEM_ENTRY.kind) and entry.model:
                llm_messages.extend(entry.model)
        stored = [ai_to_json(m) for m in llm_messages]
        await runtime.commit(
            lambda tx, current: TaskState(
                status="running", checkpoint={"phase": "stream", "messages": stored}
            )
        )

    async def stream(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        messages = [message_from_json(m) for m in (task.state.checkpoint.get("messages") or [])]
        response = await _stream_once(harness.model, messages, harness)
        await runtime.commit(
            lambda tx, current: TaskState(
                status="running",
                checkpoint={"phase": "settle", "response": ai_to_json([response])[0]},
            )
        )

    async def settle(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        async def tx_change(tx, current):
            response = message_from_json(current.state.checkpoint["response"])
            entry = await tx.append_entry(
                current.conversation_id,
                {
                    "kind": ASSISTANT_ENTRY.kind,
                    "model": [response],
                    "data": {"stop_reason": response.stop_reason},
                },
            )
            await tx.settle_submission(current.input["submission_id"], {"status": "done", "answer": entry.id})
            return TaskState(
                status="terminal", outcome=TaskOutcome(status="completed", result={"entry_id": entry.id})
            )

        await runtime.commit(tx_change)

    async def abort(task: TaskRecord, runtime: TaskRuntime, harness: "Harness") -> None:
        async def tx_change(tx, current):
            await _abort_generation(tx, current)
            return None

        await runtime.commit(tx_change)

    return define_task(
        name="pi.generation",
        version=1,
        initial=lambda input: {"phase": "prepare"},
        phases={"prepare": prepare, "stream": stream, "settle": settle},
        abort=abort,
    )


async def _abort_generation(tx, current: TaskRecord) -> None:
    submission_id = (current.input or {}).get("submission_id")
    if submission_id is not None:
        await tx.settle_submission(
            submission_id, {"status": "unanswered", "reason": "generation aborted"}
        )
    await _stage_terminal(tx, current, TaskOutcome(status="aborted", reason="aborted"))


async def _stream_once(model: Model, messages: List[Any], harness: "Harness") -> AssistantMessage:
    system, rest = normalize_context(TranscriptContext(messages=messages))
    if system is not None:
        rest = [system, *rest]
    fn = get_default_stream_fn()
    final: Optional[AssistantMessage] = None
    async for event in fn(model, TranscriptContext(messages=rest), None):
        if event.type in ("done", "error"):
            final = event.message if event.type == "done" else event.error
    if final is None:
        raise RuntimeError("Provider stream ended without a final message")
    return final

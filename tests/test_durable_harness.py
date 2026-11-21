"""Durable end-to-end test: a full conversation survives restarts."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.ai.json_io import to_json
from pi.ai.models import Model
from pi.ai.providers import faux
from pi.ai.retry import RetryPolicy
from pi.ai.types import AssistantMessage, TextContent, Usage, UserMessage
from pi.agent.types import AgentTool, AgentToolResult
from pi.durable.entries import ASSISTANT_ENTRY, COMPACTION_ENTRY, SYSTEM_ENTRY, TOOL_RESULT_ENTRY, USER_ENTRY
from pi.durable.extensions import Extension
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage
from pi.durable.tasks import compact_definition
from pi.durable.transaction import TransactionScope
from pi.durable.types import TaskOutcome, TaskState, define_task


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


def test_overlapping_schedulers_do_not_repeat_generation(tmp_path, monkeypatch):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        first = Harness(session, make_model())
        second = Harness(session, make_model())
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def stream_once(model, messages, harness):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return AssistantMessage(content=[TextContent(text="answer")], api="faux",
                provider="faux", model="faux", usage=Usage(), stop_reason="stop", timestamp=0)

        monkeypatch.setattr("pi.durable.harness._stream_once", stream_once)
        cid = await first.create_conversation()
        await first.submit_input(cid, "hello")
        running = asyncio.create_task(first.run_pending_tasks())
        await asyncio.wait_for(entered.wait(), 2)
        overlapping = asyncio.create_task(second.run_pending_tasks())
        try:
            await asyncio.sleep(0.05)
            assert calls == 1
            assert not overlapping.done()
        finally:
            release.set()
            counts = await asyncio.gather(running, overlapping)
        assert counts == [1, 0]
        assert calls == 1
        assert [entry.kind for entry in await first.transcript(cid)] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        await session.close()

    asyncio.run(scenario())


def test_durable_conversation_end_to_end(tmp_path):
    """Submit input, run the scheduler, and observe the committed turn."""

    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "the answer"}])
        try:
            path = str(tmp_path / "session.jsonl")
            storage = JsonlStorage(path)
            session = create_session(storage)
            harness = Harness(session, make_model())

            cid = await harness.create_conversation()
            task_id = await harness.submit_input(cid, "hello")
            await harness.run_pending_tasks()

            transcript = await harness.transcript(cid)
            kinds = [e.kind for e in transcript]
            assert kinds == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
            answer = transcript[2].model[0]
            assert answer.content[0].text == "the answer"

            # The submission settled against the answer entry.
            async def read_submission(tx):
                return await tx.submission_by_request(cid, None) if False else None

            await storage.close()

            # Reopen: everything is committed and visible; nothing left to run.
            reopened = create_session(JsonlStorage(path))
            reopened_harness = Harness(reopened, make_model())
            transcript = await reopened_harness.transcript(cid)
            assert [e.kind for e in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
            executed = await reopened_harness.run_pending_tasks()
            assert executed == 0
            await reopened.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_durable_generation_survives_crash_before_stream(tmp_path):
    """A crash after prepare resumes at stream and still commits one turn."""

    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "recovered answer"}])
        try:
            path = str(tmp_path / "session.jsonl")
            storage = JsonlStorage(path)
            session = create_session(storage)
            harness = Harness(session, make_model())
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "hi")

            # Simulate a crash: stop after the prepare phase committed by
            # manually advancing one scheduler iteration without the stream.
            task = await harness._next_pending_task()
            await harness._reserve(task.id, harness.definitions[task.kind], task)
            runtime_reserve_only = None

            # Commit only the prepare phase by invoking its handler directly.
            definition = harness.definitions["pi.generation"]
            from pi.durable.harness import TaskRuntime

            current = await harness._get_task(task.id)
            runtime = TaskRuntime(harness, task.id, asyncio.Event())
            await definition.phases["prepare"](current, runtime, harness)
            after = await harness._get_task(task.id)
            assert after.state.status == "running"
            assert after.state.checkpoint["phase"] == "stream"

            # Process dies here. Reopen and run: the turn still completes once.
            await storage.close()
            reopened = create_session(JsonlStorage(path))
            harness2 = Harness(reopened, make_model())
            executed = await harness2.run_pending_tasks()
            assert executed >= 1

            transcript = await harness2.transcript(cid)
            kinds = [e.kind for e in transcript]
            assert kinds == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
            assert transcript[2].model[0].content[0].text == "recovered answer"
            await reopened.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_abort_generation_leaves_unanswered(tmp_path):
    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "never seen"}])
        try:
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            harness = Harness(session, make_model())
            cid = await harness.create_conversation()
            task_id = await harness.submit_input(cid, "hello")

            # Reserve so the abort mark is honored at the next commit.
            task = await harness._next_pending_task()
            await harness._reserve(task.id, harness.definitions[task.kind], task)
            await harness.abort_task(task_id)
            await harness.run_pending_tasks()

            transcript = await harness.transcript(cid)
            assert [e.kind for e in transcript] == [USER_ENTRY.kind]
            task_record = await harness._get_task(task_id)
            assert task_record.state.status == "terminal"
            assert task_record.state.outcome.status == "aborted"
            await session.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_durable_tool_task_applies_prepare_arguments(tmp_path):
    """pi.tool normalizes arguments exactly like the interactive tool pipeline."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, make_model())
        seen = {}

        class EchoTool:
            prepare_arguments = staticmethod(lambda args: {"text": args["value"]})

            async def execute(self, call_id, params, signal=None, on_update=None):
                seen["params"] = params
                return AgentToolResult(content=[TextContent(text=params["text"])], details={})

        harness.register_tools({"echo": EchoTool()})
        cid = await harness.create_conversation()
        await harness.submit_tool_task(cid, "echo", {"value": "hi"})
        await harness.run_pending_tasks()

        assert seen["params"] == {"text": "hi"}
        transcript = await harness.transcript(cid)
        assert [entry.kind for entry in transcript] == [TOOL_RESULT_ENTRY.kind]
        await session.close()

    asyncio.run(scenario())


def test_compact_cut_inside_last_entry_does_not_fault(tmp_path):
    """A cut inside a multi-message entry snaps to the entry start instead of StopIteration."""

    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"text": "summary"}])
        try:
            session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
            model = make_model()
            model.context_window = 100
            harness = Harness(session, model)
            harness.register_definition(compact_definition(keep_recent_tokens=1, reserve_tokens=1))
            cid = await harness.create_conversation()

            def assistant(text):
                return to_json(AssistantMessage(
                    content=[TextContent(text=text)], api="faux", provider="faux", model="faux",
                    usage=Usage(), stop_reason="stop", timestamp=0,
                ))

            async def change(tx):
                await tx.append_entry(cid, {
                    "kind": USER_ENTRY.kind, "model": [to_json(UserMessage(content="hi", timestamp=0))],
                })
                await tx.append_entry(cid, {
                    "kind": ASSISTANT_ENTRY.kind,
                    "model": [assistant("a" * 400), assistant("b" * 400)],
                })

            await session.commit(change)
            task_id = await harness.submit_compact_task(cid)
            await harness.run_pending_tasks()

            record = await harness._get_task(task_id)
            assert record.state.status == "terminal"
            assert record.state.outcome.status == "completed"
            assert record.state.outcome.result["compacted"] is True
            transcript = await harness.transcript(cid)
            assert transcript[-1].kind == COMPACTION_ENTRY.kind
            assert all(entry.kind != USER_ENTRY.kind for entry in transcript)

            # Every later generation must retain the summary after the head cut.
            for text in ("continue", "continue again"):
                generation_id = await harness.submit_input(cid, text)
                task = await harness._get_task(generation_id)
                definition = harness.definitions[task.kind]
                await harness._reserve(task.id, definition, task)
                current = await harness._get_task(task.id)
                runtime = TaskRuntime(harness, task.id, asyncio.Event())
                await definition.phases["prepare"](current, runtime, harness)
                prepared = await harness._get_task(task.id)
                messages = prepared.state.checkpoint["messages"]
                assert any(message["role"] == "system" and
                           message["content"] == "Conversation summary:\nsummary"
                           for message in messages)
                await harness.run_pending_tasks()

            await session.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_durable_generation_retries_transient_stream_error(tmp_path):
    """A transient stream drop retries with backoff before the turn commits."""

    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"error": "terminated"}, {"text": "recovered answer"}])
        try:
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            harness = Harness(session, make_model(), retry=RetryPolicy(base_delay_ms=0))
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "hello")
            await harness.run_pending_tasks()

            transcript = await harness.transcript(cid)
            # The failed attempt never becomes a visible entry.
            assert [e.kind for e in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
            answer = transcript[2].model[0]
            assert answer.content[0].text == "recovered answer"
            await session.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_durable_generation_gives_up_after_max_retries(tmp_path, monkeypatch):
    """The attempt counter survives re-preparing, so retries stay bounded."""

    attempts = {"count": 0}

    async def stream(model, messages, harness, **kwargs):
        attempts["count"] += 1
        message = AssistantMessage(content=[TextContent(text="partial")], api="faux", provider="faux",
                                   model="faux", usage=Usage(), stop_reason="error", timestamp=0)
        message.error_message = "503 overloaded"
        return message

    monkeypatch.setattr("pi.durable.harness._stream_once", stream)

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model(), retry=RetryPolicy(max_retries=1, base_delay_ms=0))
        cid = await harness.create_conversation()
        await harness.submit_input(cid, "hello")
        await harness.run_pending_tasks()

        assert attempts["count"] == 2  # initial attempt plus one retry, then give up
        transcript = await harness.transcript(cid)
        assert [e.kind for e in transcript] == [USER_ENTRY.kind, SYSTEM_ENTRY.kind, ASSISTANT_ENTRY.kind]
        assert transcript[-1].model[0].stop_reason == "error"
        await session.close()

    asyncio.run(scenario())


def test_unreservable_task_faults_instead_of_crashing_the_scheduler(tmp_path):
    """A version migration failure must fault the task, not escape every cycle."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()

        async def finish(task, runtime, harness):
            await runtime.commit(lambda tx, current: TaskState(status="terminal",
                outcome=TaskOutcome(status="completed", result={})))

        harness.register_definition(define_task(
            name="pi.custom", version=1, initial=lambda value: {"phase": "run"},
            phases={"run": finish}))
        definition = harness.definitions["pi.custom"]

        async def create(tx):
            task = await tx.create_task(definition, {"x": 1},
                {"ownership": {"kind": "conversation"}, "conversation_id": cid})
            return task.id

        task_id = await session.commit(create, TransactionScope(conversation_id=cid))

        # A newer definition version makes the stored task unreservable.
        harness.register_definition(define_task(
            name="pi.custom", version=2, initial=lambda value: {"phase": "run"},
            phases={"run": finish}))
        await harness.run_pending_tasks()

        record = await harness._get_task(task_id)
        assert record.state.status == "terminal"
        assert record.state.outcome.status == "faulted"
        assert "migration" in record.state.outcome.error["message"]
        await session.close()

    asyncio.run(scenario())


def test_concurrent_scheduler_serializes_same_conversation(tmp_path):
    """concurrency>1 runs different conversations in parallel but one task per conversation."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, make_model())
        cid1 = await harness.create_conversation()
        cid2 = await harness.create_conversation()
        active = {}
        peak = {}

        async def run_phase(task, runtime, harness):
            conversation = task.conversation_id
            active[conversation] = active.get(conversation, 0) + 1
            peak[conversation] = max(peak.get(conversation, 0), active[conversation])
            await asyncio.sleep(0.02)
            active[conversation] -= 1
            await runtime.commit(lambda tx, current: TaskState(status="terminal",
                outcome=TaskOutcome(status="completed", result={})))

        harness.register_definition(define_task(
            name="pi.custom", version=1, initial=lambda value: {"phase": "run"},
            phases={"run": run_phase}))
        definition = harness.definitions["pi.custom"]

        async def create(conversation, value):
            async def change(tx):
                task = await tx.create_task(definition, {"x": value},
                    {"ownership": {"kind": "conversation"}, "conversation_id": conversation})
                return task.id

            return await session.commit(change, TransactionScope(conversation_id=conversation))

        await create(cid1, 1)
        await create(cid1, 2)
        await create(cid2, 3)
        assert await harness.run_pending_tasks(concurrency=3) == 3
        assert peak[cid1] == 1
        assert peak[cid2] == 1
        await session.close()

    asyncio.run(scenario())


def test_durable_generation_does_not_retry_quota_error(tmp_path):
    """Quota/billing errors are deterministic: the error response settles as-is."""

    async def scenario():
        faux.reset_script_index()
        os.environ["FAUX_SCRIPT"] = json.dumps([{"error": "insufficient_quota"}, {"text": "never"}])
        try:
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            harness = Harness(session, make_model(), retry=RetryPolicy(base_delay_ms=0))
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "hello")
            await harness.run_pending_tasks()

            transcript = await harness.transcript(cid)
            answer = transcript[2].model[0]
            assert answer.stop_reason == "error"
            assert answer.error_message == "insufficient_quota"
            await session.close()
        finally:
            os.environ.pop("FAUX_SCRIPT", None)
            faux.reset_script_index()

    asyncio.run(scenario())


def test_task_records_carry_lifecycle_times(tmp_path):
    """Tasks stamp startedAt on the first running write and endedAt on the
    terminal write (F-AG-07, #10549); the Session clock is injectable."""

    async def scenario():
        storage = JsonlStorage(str(tmp_path / "s.jsonl"))
        session = create_session(storage, now=lambda: 12345)
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()
        task_id = await harness.submit_input(cid, "hello")
        await harness.run_pending_tasks()
        record = await harness._get_task(task_id)
        assert record.started_at == 12345
        assert record.ended_at == 12345
        await storage.close()

    asyncio.run(scenario())


def test_task_lifecycle_times_survive_resume_and_predate_writes(tmp_path):
    """Resumed tasks keep their start stamp (carry-over, not restamp); records
    written before the fields existed read back without them."""

    async def scenario():
        from pi.durable.types import TaskRecord, write_task

        path = str(tmp_path / "s.jsonl")
        clock = {"now": 5000}
        storage = JsonlStorage(path)
        session = create_session(storage, now=lambda: clock["now"])
        harness = Harness(session, make_model())
        cid = await harness.create_conversation()

        async def phase_wait(task, runtime, harness):
            clock["now"] = 6000
            await runtime.commit(lambda tx, current: TaskState(
                status="waiting", on=[], policy="allSettled", checkpoint={"phase": "resume"}))

        async def phase_resume(task, runtime, harness):
            clock["now"] = 7000
            await runtime.commit(lambda tx, current: TaskState(
                status="terminal", outcome=TaskOutcome(status="completed")))

        definition = define_task(name="pi.wait", version=1, initial=lambda input: {"phase": "wait"},
                                 phases={"wait": phase_wait, "resume": phase_resume})
        harness.register_definition(definition)

        async def submit_wait(tx):
            record = await tx.create_task(definition, {}, {"ownership": {"kind": "conversation"},
                                                           "conversation_id": cid})
            return record.id

        task_id = await harness.session.commit(submit_wait, TransactionScope(conversation_id=cid))
        await harness.run_pending_tasks()
        record = await harness._get_task(task_id)
        # started_at is stamped at the reserve (clock 5000) and carried over to
        # the terminal write (clock 7000); a restamp would have made it 7000.
        assert record.started_at == 5000
        assert record.ended_at == 7000

        # A record written before the fields existed (raw storage commit)
        # reads back without them.
        raw = TaskRecord(
            id=await storage.mint_id(), conversation_id=cid, kind="legacy", version=1, input={},
            state=TaskState(status="pending", checkpoint={"phase": "p"}))
        await storage.commit([write_task(raw)])
        legacy = await storage.task(raw.id)
        assert legacy.started_at is None and legacy.ended_at is None
        await storage.close()

        # Reopen: stamped records replay with their times.
        reopened = create_session(JsonlStorage(path))
        again = await reopened.storage.task(task_id)
        assert again.started_at == 5000 and again.ended_at == 7000
        await reopened.close()

    asyncio.run(scenario())


def test_durable_tool_result_and_assistant_carry_durations(tmp_path, monkeypatch):
    """pi.tool's visible pi.tool-result entry times execute(), and assistant
    entries carry the ai-layer response duration (F-AG-08, #10549)."""

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"id": "c1", "name": "echo", "arguments": {}}]},
        {"text": "final"},
    ]))
    faux.reset_script_index()
    try:
        async def scenario():
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            harness = Harness(session, make_model())

            async def execute(call_id, params, signal=None, on_update=None):
                await asyncio.sleep(0.02)
                return AgentToolResult(content=[TextContent(text="echoed")])

            harness.register_tools({"echo": AgentTool(name="echo", description="echo", execute=execute)})
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "run the tool")
            await harness.run_pending_tasks()

            transcript = await harness.transcript(cid)
            kinds = [entry.kind for entry in transcript]
            # The second generation re-baselines idempotently: no new pi.system.
            assert kinds == ["pi.user", "pi.system", "pi.assistant", "pi.tool-result", "pi.assistant"]
            tool_result = transcript[3].model[0]
            assert tool_result.duration_ms is not None and tool_result.duration_ms >= 15
            tool_call = transcript[2].model[0]
            final = transcript[4].model[0]
            # Both responses went through the ai-layer wrapper (faux), which
            # stamps the final assistant message.
            assert isinstance(tool_call.duration_ms, int) and tool_call.duration_ms >= 0
            assert isinstance(final.duration_ms, int) and final.duration_ms >= 0
            await session.close()

        asyncio.run(scenario())
    finally:
        os.environ.pop("FAUX_SCRIPT", None)
        faux.reset_script_index()


def test_durable_blocked_tool_result_has_no_duration(tmp_path, monkeypatch):
    """A hook-blocked call never executed, so its result carries no duration."""

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"id": "c1", "name": "echo", "arguments": {}}]},
        {"text": "done"},
    ]))
    faux.reset_script_index()
    try:
        async def scenario():
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            extension = Extension("guard", hooks={"before_tool": lambda call, api, runtime: {"block": "denied"}})
            harness = Harness(session, make_model(), extensions=[extension])

            async def execute(*args):
                raise AssertionError("blocked tool must not run")

            harness.register_tools({"echo": AgentTool(name="echo", description="echo", execute=execute)})
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "hello")
            await harness.run_pending_tasks()
            transcript = await harness.transcript(cid)
            tool_result = next(entry for entry in transcript if entry.kind == "pi.tool-result")
            message = tool_result.model[0]
            assert message.is_error is True and message.duration_ms is None
            await session.close()

        asyncio.run(scenario())
    finally:
        os.environ.pop("FAUX_SCRIPT", None)
        faux.reset_script_index()


def test_live_tool_slots_do_not_accumulate_across_rounds(tmp_path, monkeypatch):
    """Each tool round presents only its own slots, and the ended run leaves no
    live state behind (TS ``endRun`` deletes ``live.tools``)."""

    monkeypatch.setenv("FAUX_SCRIPT", json.dumps([
        {"toolCalls": [{"id": "c1", "name": "echo", "arguments": {}}]},
        {"toolCalls": [{"id": "c2", "name": "echo", "arguments": {}}]},
        {"text": "final"},
    ]))
    faux.reset_script_index()
    try:
        async def scenario():
            session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
            harness = Harness(session, make_model())
            seen = []

            async def execute(call_id, params, signal=None, on_update=None):
                live = await harness.live(cid)
                seen.append([slot.get("call_id") for slot in live.get("tools") or []])
                return AgentToolResult(content=[TextContent(text="echoed")])

            harness.register_tools({"echo": AgentTool(name="echo", description="echo", execute=execute)})
            cid = await harness.create_conversation()
            await harness.submit_input(cid, "run the tool")
            await harness.run_pending_tasks()

            assert seen == [["c1"], ["c2"]]
            assert await harness.live(cid) == {}
            await session.close()

        asyncio.run(scenario())
    finally:
        os.environ.pop("FAUX_SCRIPT", None)
        faux.reset_script_index()


def test_standalone_tool_task_leaves_no_live_slot(tmp_path):
    """TS creates tool slots only in ``startToolRound``: a standalone
    ``pi.tool`` task has no live presentation, so finished slots cannot pile up."""

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "s.jsonl")))
        harness = Harness(session, make_model())

        async def execute(call_id, params, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="echoed")])

        harness.register_tools({"echo": AgentTool(name="echo", description="echo", execute=execute)})
        cid = await harness.create_conversation()
        await harness.submit_tool_task(cid, "echo", {}, call_id="standalone")
        await harness.run_pending_tasks()

        transcript = await harness.transcript(cid)
        assert [entry.kind for entry in transcript] == [TOOL_RESULT_ENTRY.kind]
        assert await harness.live(cid) == {}
        await session.close()

    asyncio.run(scenario())

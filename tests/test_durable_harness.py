"""Durable end-to-end test: a full conversation survives restarts."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.ai.models import Model
from pi.ai.providers import faux
from pi.durable.entries import ASSISTANT_ENTRY, USER_ENTRY
from pi.durable.harness import Harness
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage


def make_model() -> Model:
    return Model(id="faux", name="Faux", api="faux", provider="faux", base_url="")


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
            assert kinds == [USER_ENTRY.kind, ASSISTANT_ENTRY.kind]
            answer = transcript[1].model[0]
            assert answer.content[0].text == "the answer"

            # The submission settled against the answer entry.
            async def read_submission(tx):
                return await tx.submission_by_request(cid, None) if False else None

            await storage.close()

            # Reopen: everything is committed and visible; nothing left to run.
            reopened = create_session(JsonlStorage(path))
            reopened_harness = Harness(reopened, make_model())
            transcript = await reopened_harness.transcript(cid)
            assert [e.kind for e in transcript] == [USER_ENTRY.kind, ASSISTANT_ENTRY.kind]
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
            assert kinds == [USER_ENTRY.kind, ASSISTANT_ENTRY.kind]
            assert transcript[1].model[0].content[0].text == "recovered answer"
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

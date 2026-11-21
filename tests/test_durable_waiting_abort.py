import asyncio

import pytest

from pi.ai.models import Model
from pi.durable.entries import TOOL_RESULT_ENTRY
from pi.durable.harness import Harness, TaskRuntime
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage
from pi.durable.types import TaskOutcome, TaskState, define_task
from pi.durable.worker import SessionEndpoint, SessionServices


@pytest.mark.parametrize("policy", ["allSettled", "failFast"])
@pytest.mark.parametrize("reopen", [False, True])
def test_waiting_abort_settles_submission_without_join(tmp_path, policy, reopen):
    async def scenario():
        path = str(tmp_path / "session.jsonl")
        session = create_session(JsonlStorage(path))
        model = Model("faux", "Faux", "faux", "faux", "")
        harness = Harness(session, model)
        cid = await harness.create_conversation()
        task_id = await harness.submit_input(cid, "hello")
        task = await harness._get_task(task_id)
        submission_id = task.input["submission_id"]
        await harness._reserve(task_id, harness.definitions[task.kind], task)
        runtime = TaskRuntime(harness, task_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "prepare"}, on=[999999], policy=policy))
        # Unsatisfied joins stay waiting until an abort is requested.
        assert await harness.run_pending_tasks() == 0
        assert (await harness._get_task(task_id)).state.status == "waiting"
        await harness.abort_task(task_id)
        if reopen:
            await session.close()
            session = create_session(JsonlStorage(path))
            harness = Harness(session, model)
        await asyncio.wait_for(harness.run_pending_tasks(), 2)
        record = await harness._get_task(task_id)
        assert record.state.status == "terminal"
        assert record.state.outcome.status == "aborted"
        submission = await session.storage.submission(submission_id)
        assert submission.status == "unanswered"
        assert await harness.run_pending_tasks() == 0
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_task_id", [False, True])
def test_worker_abort_settles_waiting_generation_and_its_tools(tmp_path, explicit_task_id):
    """The worker abort must find a waiting generation and cascade to its tools.

    A waiting generation is blocked on its tool round, so marking only it would
    leave the child pi.tool tasks running; they would later commit tool results
    for a turn that was already aborted.
    """

    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        generation_id = await harness.submit_input(cid, "hello")
        generation = await harness._get_task(generation_id)
        await harness._reserve(generation_id, harness.definitions[generation.kind], generation)
        child_id = await harness.submit_tool_task(cid, "echo", {"text": "x"})
        runtime = TaskRuntime(harness, generation_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "after_tools", "tools": [child_id]},
            on=[child_id], policy="allSettled"))
        assert (await harness._get_task(generation_id)).state.status == "waiting"

        endpoint = SessionEndpoint(SessionServices(harness, cid, asyncio.Event()))
        args = [{"taskId": generation_id}] if explicit_task_id else []
        reply = await endpoint.invoke_service({"member": "abort", "args": args}, None, None)
        assert reply == {"aborted": True, "taskId": generation_id}
        # Every non-terminal task of the conversation is marked, not only the
        # generation that the reply reports.
        assert (await harness._get_task(generation_id)).abort_requested
        assert (await harness._get_task(child_id)).abort_requested

        await asyncio.wait_for(harness.run_pending_tasks(), 2)
        generation = await harness._get_task(generation_id)
        child = await harness._get_task(child_id)
        assert (generation.state.status, generation.state.outcome.status) == ("terminal", "aborted")
        assert (child.state.status, child.state.outcome.status) == ("terminal", "aborted")
        # The tool never ran, so no tool result was committed.
        assert TOOL_RESULT_ENTRY.kind not in [entry.kind for entry in await harness.transcript(cid)]

        # A terminal task is no longer abortable.
        reply = await endpoint.invoke_service(
            {"member": "abort", "args": [{"taskId": generation_id}]}, None, None)
        assert reply == {"aborted": False}
        await session.close()

    asyncio.run(scenario())


def _flat_definition(name):
    async def finish(task, runtime, harness):
        await runtime.commit(lambda tx, current: TaskState(status="terminal",
            outcome=TaskOutcome(status="completed", result={})))
    return define_task(name=name, version=1, initial=lambda value: {"phase": "run"},
                       phases={"run": finish})


def _create_custom(session, harness, definition, cid):
    from pi.durable.transaction import TransactionScope

    async def create(tx):
        task = await tx.create_task(definition, {"x": 1},
            {"ownership": {"kind": "conversation"}, "conversation_id": cid})
        return task.id

    return session.commit(create, TransactionScope(conversation_id=cid))


def test_rejects_the_wait_that_closes_a_cycle(tmp_path):
    # A -> B and B -> A would deadlock both tasks forever.
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        definition = _flat_definition("pi.custom")
        harness.register_definition(definition)
        a_id, b_id = await _create_custom(session, harness, definition, cid), \
            await _create_custom(session, harness, definition, cid)
        for task_id in (a_id, b_id):
            task = await harness._get_task(task_id)
            await harness._reserve(task_id, definition, task)

        runtime_a = TaskRuntime(harness, a_id, asyncio.Event())
        await runtime_a.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "run"}, on=[b_id], policy="allSettled"))

        runtime_b = TaskRuntime(harness, b_id, asyncio.Event())
        with pytest.raises(ValueError, match="wait cycle"):
            await runtime_b.commit(lambda tx, current: TaskState(
                status="waiting", checkpoint={"phase": "run"}, on=[a_id], policy="allSettled"))
        # The rejecting task keeps its prior state; the drain cannot stall on it.
        assert (await harness._get_task(b_id)).state.status == "running"
        assert (await harness._get_task(a_id)).state.status == "waiting"
        await session.close()

    asyncio.run(scenario())


def test_rejects_a_longer_cycle_of_waits(tmp_path):
    # C -> A closes A -> B -> C.
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        definition = _flat_definition("pi.custom")
        harness.register_definition(definition)
        ids = [await _create_custom(session, harness, definition, cid) for _ in range(3)]
        for task_id in ids:
            task = await harness._get_task(task_id)
            await harness._reserve(task_id, definition, task)

        for waiter, target in zip(ids, ids[1:]):  # A->B, B->C
            runtime = TaskRuntime(harness, waiter, asyncio.Event())
            await runtime.commit(lambda tx, current, target=target: TaskState(
                status="waiting", checkpoint={"phase": "run"}, on=[target], policy="allSettled"))

        runtime_c = TaskRuntime(harness, ids[2], asyncio.Event())
        with pytest.raises(ValueError, match="wait cycle.*-> "):
            await runtime_c.commit(lambda tx, current: TaskState(
                status="waiting", checkpoint={"phase": "run"}, on=[ids[0]], policy="allSettled"))
        await session.close()

    asyncio.run(scenario())


def test_rejects_waiting_on_self_and_on_the_owner(tmp_path):
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        definition = _flat_definition("pi.custom")
        harness.register_definition(definition)
        a_id = await _create_custom(session, harness, definition, cid)
        task = await harness._get_task(a_id)
        await harness._reserve(a_id, definition, task)

        runtime = TaskRuntime(harness, a_id, asyncio.Event())
        with pytest.raises(ValueError, match="cannot wait on itself"):
            await runtime.commit(lambda tx, current: TaskState(
                status="waiting", checkpoint={"phase": "run"}, on=[a_id], policy="allSettled"))

        # A live owned child blocks its owner's finalization, so a child waiting
        # back on its owner would deadlock (TS allows this only for background).
        child_id = await _create_custom(session, harness, definition, cid)
        async def adopt(tx):
            from pi.durable.types import write_task
            import dataclasses
            child = await tx.task(child_id)
            replaced = dataclasses.replace(child, owner=a_id)
            tx._writes.append(write_task(replaced))
            tx._mark_table_write()
        await session.commit(adopt)

        child_runtime = TaskRuntime(harness, child_id, asyncio.Event())
        with pytest.raises(ValueError, match="wait cycle"):
            await child_runtime.commit(lambda tx, current: TaskState(
                status="waiting", checkpoint={"phase": "run"}, on=[a_id], policy="allSettled"))
        await session.close()

    asyncio.run(scenario())


async def _create_background_child(session, definition, owner_id, cid):
    from pi.durable.transaction import TransactionScope

    async def create(tx):
        task = await tx.create_task(definition, {"x": 2}, {
            "ownership": {"kind": "task", "task_id": owner_id}, "background": True})
        return task.id

    return await session.commit(create, TransactionScope(conversation_id=cid))


def test_background_child_does_not_hold_completing_owner(tmp_path):
    # Background work never holds its owner: the owner reaches terminal while
    # the child is still live.
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        definition = _flat_definition("pi.custom")
        harness.register_definition(definition)
        parent_id = await _create_custom(session, harness, definition, cid)
        parent = await harness._get_task(parent_id)
        await harness._reserve(parent_id, definition, parent)
        child_id = await _create_background_child(session, definition, parent_id, cid)

        runtime = TaskRuntime(harness, parent_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="terminal", outcome=TaskOutcome(status="completed", result={})))
        assert (await harness._get_task(parent_id)).state.status == "completing"
        await harness._finalize_completing()

        assert (await harness._get_task(parent_id)).state.status == "terminal"
        assert (await harness._get_task(child_id)).state.status != "terminal"
        await session.close()

    asyncio.run(scenario())


def test_background_child_may_wait_on_its_owner(tmp_path):
    # The owner does not hold a background child, so waiting on the owner adds
    # no cycle edge (TS #10411/#10533).
    async def scenario():
        session = create_session(JsonlStorage(str(tmp_path / "session.jsonl")))
        harness = Harness(session, Model("faux", "Faux", "faux", "faux", ""))
        cid = await harness.create_conversation()
        definition = _flat_definition("pi.custom")
        harness.register_definition(definition)
        parent_id = await _create_custom(session, harness, definition, cid)
        child_id = await _create_background_child(session, definition, parent_id, cid)

        runtime = TaskRuntime(harness, child_id, asyncio.Event())
        await runtime.commit(lambda tx, current: TaskState(
            status="waiting", checkpoint={"phase": "run"}, on=[parent_id], policy="allSettled"))
        assert (await harness._get_task(child_id)).state.status == "waiting"
        await session.close()

    asyncio.run(scenario())

"""Runtime regressions found by the second Python audit."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from pi.coding_agent.tools import common
from pi.coding_agent.rpc_client import RpcClient
from pi.durable.env import Context, LocalExecutionEnv, StreamDecoder, range_decoder
from pi.durable.tools import mutation_queue
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage


@pytest.mark.parametrize("data", [
    "中文🙂".encode(), b"\xef\xbb\xbfhello", b"\xc3\xa9", b"\xf0\x9f\x99\x82",
    b"\xe0\x80\x80a", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x82",
])
def test_utf8_decoders_match_whole_input_at_every_chunk_boundary(data):
    for split in range(len(data) + 1):
        for factory, encoding in ((range_decoder, "utf-8"), (StreamDecoder, "utf-8-sig")):
            decoder = factory()
            actual = decoder.decode(data[:split]) + decoder.decode(data[split:]) + decoder.decode()
            assert actual == data.decode(encoding, "replace"), (data, split, factory)


@pytest.mark.parametrize("durable", [False, True])
def test_mutation_queue_releases_unused_locks(tmp_path, durable):
    async def scenario():
        env = LocalExecutionEnv(cwd=str(tmp_path))
        for index in range(20):
            path = str(tmp_path / f"file-{index}")
            if durable:
                await mutation_queue.with_file_mutation_queue(env, path, lambda: asyncio.sleep(0), Context())
            else:
                await common.with_file_mutation_queue(path, lambda: asyncio.sleep(0))
        locks = mutation_queue._QUEUES if durable else common._MUTATION_LOCKS
        assert not locks
    asyncio.run(scenario())


@pytest.mark.parametrize("durable", [False, True])
def test_mutation_queue_serializes_after_waiter_cancellation(tmp_path, durable):
    async def scenario():
        env = LocalExecutionEnv(cwd=str(tmp_path))
        path = str(tmp_path / "shared")
        started, release = asyncio.Event(), asyncio.Event()
        order = []

        async def enqueue(fn):
            if durable:
                return await mutation_queue.with_file_mutation_queue(env, path, fn, Context())
            return await common.with_file_mutation_queue(path, fn)

        async def first():
            started.set()
            await release.wait()
            order.append("first")

        async def later():
            order.append("later")

        holder = asyncio.create_task(enqueue(first))
        await started.wait()
        cancelled = asyncio.create_task(enqueue(later))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        waiter = asyncio.create_task(enqueue(later))
        await asyncio.sleep(0)
        assert order == []
        release.set()
        await asyncio.wait_for(asyncio.gather(holder, waiter), 2)
        assert order == ["first", "later"]

    # A new loop must not reuse a lock bound during the previous contention.
    asyncio.run(scenario())
    asyncio.run(scenario())


@pytest.mark.parametrize("blocked_at", ["lock", "drain"])
def test_rpc_timeout_covers_sending(blocked_at):
    async def scenario():
        client = RpcClient()
        client.reader = asyncio.get_running_loop().create_future()
        blocked = asyncio.Event()
        client.process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(
            write=Mock(), drain=AsyncMock(side_effect=blocked.wait)))
        if blocked_at == "lock":
            await client.lock.acquire()
        task = asyncio.create_task(client.request("get_state", timeout=0.01))
        try:
            done, _ = await asyncio.wait([task], timeout=0.5)
            assert task in done, "request timeout did not cover sending"
            with pytest.raises(asyncio.TimeoutError):
                await task
            assert not client.pending
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if client.lock.locked():
                client.lock.release()
    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "jsonl", "sqlite"])
@pytest.mark.parametrize("operation", ["answer", "place_twice", "settle_twice"])
def test_submission_updates_see_previous_changes_in_same_transaction(tmp_path, backend, operation):
    async def scenario():
        if backend == "memory":
            storage = MemoryStorage()
        elif backend == "jsonl":
            storage = JsonlStorage(str(tmp_path / "storage.jsonl"))
        else:
            storage = SqliteStorage(str(tmp_path / "storage.sqlite"))
        session = create_session(storage)
        try:
            conversation = await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))
            submission = await session.commit(lambda tx: tx.create_submission({
                "conversation_id": conversation.id, "type": "input", "request_id": "request",
            }))
            first = await session.commit(lambda tx: tx.append_entry(conversation.id, {"kind": "pi.user"}))
            second = await session.commit(lambda tx: tx.append_entry(conversation.id, {"kind": "pi.user"}))

            async def change(tx):
                if operation == "settle_twice":
                    await tx.settle_submission(submission.id, {"status": "unanswered", "reason": "first"})
                    await tx.settle_submission(submission.id, {"status": "unanswered", "reason": "second"})
                else:
                    await tx.place_submission(submission.id, first.id)
                    if operation == "answer":
                        await tx.settle_submission(submission.id, {"status": "done", "answer": second.id})
                    else:
                        await tx.place_submission(submission.id, second.id)

            await session.commit(change)
            stored = await storage.submission(submission.id)
            if operation == "settle_twice":
                assert stored.reason == "first"
            else:
                assert stored.entry == first.id
                assert stored.status == ("done" if operation == "answer" else "placed")
                if operation == "answer":
                    assert stored.answer == second.id
        finally:
            await session.close()
    asyncio.run(scenario())

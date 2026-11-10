"""Document transaction and snapshot isolation regressions."""

import asyncio

import pytest

from pi.durable.documents import define_doc, define_doc_family
from pi.durable.session import create_session
from pi.durable.storage import JsonlStorage, MemoryStorage, SqliteStorage


@pytest.fixture(params=["memory", "jsonl", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        storage = MemoryStorage()
    elif request.param == "jsonl":
        storage = JsonlStorage(str(tmp_path / "store.jsonl"))
    else:
        storage = SqliteStorage(str(tmp_path / "store.sqlite"))
    return storage


async def save(session, definition, *args):
    async def change(tx):
        draft = await tx.doc(definition, *args)
        draft.dirty = True
    await session.commit(change)


def test_snapshot_and_failed_draft_do_not_mutate_cached_state(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.isolation", 1, "session", lambda seed: {"nested": {"n": 1}})
        try:
            await save(session, definition)
            snapshot = await session.snapshot(definition)
            snapshot["nested"]["n"] = 10
            assert await session.snapshot(definition) == {"nested": {"n": 1}}

            async def fail(tx):
                draft = await tx.doc(definition)
                draft.value["nested"]["n"] = 20
                draft.dirty = True
                raise RuntimeError("rollback")

            with pytest.raises(RuntimeError, match="rollback"):
                await session.commit(fail)
            assert await session.snapshot(definition) == {"nested": {"n": 1}}
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("existing", [False, True])
def test_repeated_doc_returns_same_draft(storage, existing):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.reuse", 1, "session", lambda seed: {"n": 0})
        try:
            if existing:
                await save(session, definition)

            async def change(tx):
                first = await tx.doc(definition)
                first.value["n"] += 1
                first.dirty = True
                second = await tx.doc(definition)
                assert second is first
                second.value["n"] += 1

            await session.commit(change)
            assert await session.snapshot(definition) == {"n": 2}
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["session", "conversation"])
def test_family_address_is_preserved(storage, scope):
    async def scenario():
        session = create_session(storage)
        definition = define_doc_family("audit.family", 1, scope, lambda seed: {"seed": seed})
        try:
            owners = [[]]
            if scope == "conversation":
                owners = []
                for _ in range(2):
                    owner = await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))
                    owners.append([owner.id])
            for index, owner in enumerate(owners):
                for key in ("a", "b"):
                    await save(session, definition, *owner, key, f"{index}:{key}")
            for index, owner in enumerate(owners):
                for key in ("a", "b"):
                    assert await session.snapshot(definition, *owner, key) == {"seed": f"{index}:{key}"}
        finally:
            await session.close()
    asyncio.run(scenario())


def test_retire_without_loading_draft_can_publish(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.retire", 1, "session", lambda seed: {"n": 1})
        publications = []
        session.subscribe_commits(publications.append)
        try:
            await save(session, definition)
            await session.commit(lambda tx: tx.retire_doc(definition))
            assert await session.snapshot(definition) is None
            assert publications[-1].changes[0]["version"] is None
            # Retirement must leave the session usable for subsequent commits.
            await session.commit(lambda tx: tx.create_conversation({"kind": "ownerless"}))
        finally:
            await session.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("existing", [False, True])
def test_retiring_dirty_draft_does_not_recreate_document(storage, existing):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.retire-dirty", 1, "session", lambda seed: {"n": 1})
        try:
            if existing:
                await save(session, definition)

            async def change(tx):
                draft = await tx.doc(definition)
                draft.value["n"] = 2
                draft.dirty = True
                await tx.retire_doc(definition)

            await session.commit(change)
            assert await session.snapshot(definition) is None
        finally:
            await session.close()
    asyncio.run(scenario())


def test_snapshot_rejects_newer_version_and_migration_is_detached(storage):
    async def scenario():
        session = create_session(storage)
        definition = define_doc("audit.version", 2, "session", lambda seed: {"nested": {"n": 1}})
        try:
            await save(session, definition)
            old = define_doc("audit.version", 1, "session", lambda seed: {})
            with pytest.raises(ValueError, match="newer than definition"):
                await session.snapshot(old)

            def migrate(value, version):
                value["nested"]["n"] = version + 1
                return value

            new = define_doc("audit.version", 3, "session", lambda seed: {}, migrate=migrate)
            assert await session.snapshot(new) == {"nested": {"n": 3}}
            assert await session.snapshot(definition) == {"nested": {"n": 1}}
        finally:
            await session.close()
    asyncio.run(scenario())

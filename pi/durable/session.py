"""Session kernel.

Python port of the core of ``packages/durable/src/session/session.ts``: one
serialized mutation line over a Storage backend, transactional commits, and
synchronous post-adoption publication. Watch/state attachment (the chord
replicated-state layer) is not ported; snapshots cover the read surface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from .documents import DocDefinition, materialize_document, resolve_address
from .errors import StorageRejected
from .transaction import Transaction, TransactionScope
from .types import CommitPublication


class SessionHost:
    """Adapter handed to transactions (storage access and hooks)."""

    def __init__(self, session: "Session") -> None:
        self._session = session

    @property
    def storage(self):
        return self._session.storage

    async def load_document(self, definition: DocDefinition, address: Dict[str, Any]):
        return await self._session._load_document(definition, address)

    async def conversation_created(self, tx: Transaction, record) -> None:
        await self._session.conversation_created(tx, record)


class Session:
    """Owner of one mutation line, its records, and its tracked documents."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self._commit_listeners: List[Callable[[CommitPublication], None]] = []
        self._close_listeners: List[Callable[[], None]] = []
        self._line_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_task: Optional[asyncio.Task] = None
        self._poison: Optional[BaseException] = None
        self._document_cache: Dict[str, Optional[Any]] = {}
        self.host = SessionHost(self)

    # -- hooks for subclasses (Harness stages built-in documents) --------------

    async def conversation_created(self, tx: Transaction, record) -> None:
        return None

    # -- commit line -------------------------------------------------------------

    async def commit(self, change: Callable[[Transaction], Any], scope: Optional[TransactionScope] = None) -> Any:
        """Run one atomic transaction on the Session mutation line."""

        if self._closing or self._closed:
            raise RuntimeError("Session is closed")
        if self._poison is not None:
            raise RuntimeError(
                "Session is poisoned by a failed commit after storage admission; reopen it"
            ) from self._poison

        async with self._line_lock:
            if self._poison is not None:
                raise RuntimeError(
                    "Session is poisoned by a failed commit after storage admission; reopen it"
                ) from self._poison
            tx = Transaction(self.host, scope)
            try:
                result = change(tx)
                if hasattr(result, "__await__"):
                    result = await result
            except BaseException:
                await tx.settle_failure()
                raise

            writes = await tx.settle_success()
            if not writes:
                tx.discard()
                return result

            try:
                seq = await self.storage.commit(writes)
            except BaseException as exc:
                tx.discard()
                if not isinstance(exc, StorageRejected):
                    # No batch effect committed, but the session can no longer
                    # guarantee consistency between memory and storage.
                    self._poison = exc
                raise

            documents = tx.adopt(seq)
            # Committed document changes invalidate loaded trackers.
            if documents:
                self._document_cache.clear()
            self._publish(seq, writes, documents)
            return result

    def _publish(self, seq: int, writes: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> None:
        if not self._commit_listeners:
            return
        changes: List[Dict[str, Any]] = [
            w for w in writes if w["type"] in ("conversation", "entry", "task", "submission")
        ]
        changes.extend(documents)
        publication = CommitPublication(seq=seq, changes=changes)
        for listener in list(self._commit_listeners):
            try:
                listener(publication)
            except Exception as exc:  # noqa: BLE001 - listeners must not break the commit
                # Storage already committed. Report the listener failure to the
                # event loop instead of making the caller retry a commit that
                # already happened.
                try:
                    asyncio.get_running_loop().call_exception_handler({
                        "message": "durable commit listener failed",
                        "exception": exc,
                    })
                except RuntimeError:
                    pass

    def subscribe_commits(self, listener: Callable[[CommitPublication], None]) -> Callable[[], None]:
        self._assert_usable()
        self._commit_listeners.append(listener)
        return lambda: self._commit_listeners.remove(listener)

    def subscribe_close(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._assert_usable()
        self._close_listeners.append(listener)
        return lambda: self._close_listeners.remove(listener)

    # -- document reads -----------------------------------------------------------

    async def _load_document(self, definition: DocDefinition, address: Dict[str, Any]) -> Optional[Any]:
        from .transaction import _LoadedDocument

        address_id = _address_key_id(address)
        record = await self.storage.find_document(address, "current")
        if record is None:
            self._document_cache[address_id] = None
            return None
        stored = await self.storage.document(record.id, "current")
        if stored is None:
            raise StorageRejected(f"Current document {record.id} ({record.kind}) cannot be read")
        cached = self._document_cache.get(address_id)
        if (cached is not None and cached.record == stored.record
                and cached.stored_version == stored.version and cached.value == stored.value):
            return cached
        loaded = _LoadedDocument(record=stored.record, stored_version=stored.version, value=stored.value)
        self._document_cache[address_id] = loaded
        return loaded

    async def snapshot(self, definition: DocDefinition, *args: Any) -> Optional[Dict[str, Any]]:
        """Materialized current value of a document, or None when absent."""
        if self._closing or self._closed:
            raise RuntimeError("Session is closed")
        _address_id, address, _next = resolve_address(definition, list(args))
        async with self._line_lock:
            if self._poison is not None:
                raise RuntimeError("Session is poisoned; reopen it") from self._poison
            loaded = await self._load_document(definition, address)
        if loaded is None:
            return None
        from .types import StoredDocument

        stored = StoredDocument(record=loaded.record, version=loaded.stored_version, value=loaded.value)
        return materialize_document(definition, stored)

    async def snapshot_as_of(self, definition: DocDefinition, conversation_id: int, at: int, *rest: Any) -> Optional[Dict[str, Any]]:
        """Historical value of a rewindable conversation document at one entry."""
        if self._closing or self._closed:
            raise RuntimeError("Session is closed")
        args = [conversation_id, *rest]
        _address_id, address, _next = resolve_address(definition, args)
        if address["scope"].get("kind") != "conversation":
            raise TypeError("Session.snapshot_as_of() requires a conversation document")
        async with self._line_lock:
            stored_entry = await self.storage.entry_visible(conversation_id, at)
            if stored_entry is None:
                raise StorageRejected(f"Entry {at} is not visible from conversation {conversation_id}")
            entry, commit_seq = stored_entry
            scoped_address = dict(address)
            scoped_address["scope"] = {"kind": "conversation", "conversation_id": entry.conversation_id}
            record = await self.storage.find_document(scoped_address, commit_seq)
            if record is None:
                return None
            stored = await self.storage.document(record.id, commit_seq)
            if stored is None:
                raise StorageRejected(f"Historical document {record.id} ({record.kind}) cannot be read")
        return materialize_document(definition, stored)

    # -- lifecycle -----------------------------------------------------------------

    async def close(self) -> None:
        """Seal admission, drain the line, then close storage."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        listeners = list(self._close_listeners)
        self._close_listeners.clear()
        listener_error: Optional[BaseException] = None
        for listener in listeners:
            try:
                listener()
            except BaseException as exc:
                if listener_error is None:
                    listener_error = exc
        try:
            async with self._line_lock:
                self._commit_listeners.clear()
                self._document_cache.clear()
                await self.storage.close()
        finally:
            self._closed = True
        if listener_error is not None:
            raise listener_error

    def _assert_usable(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("Session is closed")


def _address_key_id(address: Dict[str, Any]) -> str:
    from .documents import _address_id

    return _address_id(address)


def create_session(storage) -> Session:
    """Open a Session kernel over one storage backend."""
    return Session(storage)

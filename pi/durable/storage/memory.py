"""In-memory reference storage backend.

Python port of ``packages/durable/src/storage/memory.ts``. Reads return
detached records: the caller may mutate what it received without corrupting
storage, as the ``Storage`` contract requires.
"""

from __future__ import annotations

import copy as _copy
from typing import Any, Dict, List, Optional

from ..errors import StorageRejected
from ..types import Page
from .base import MAX_SAFE_ID, BaseStorage


def _detach(value: Any) -> Any:
    return _copy.deepcopy(value)


class MemoryStorage(BaseStorage):
    """Detached in-memory backend; reads return detached copies of the stored records."""

    async def mint_id(self) -> int:
        self.assert_open()
        if self.next_id > MAX_SAFE_ID:
            raise StorageRejected("ID space is exhausted")
        value = self.next_id
        self.next_id += 1
        return value

    async def commit(self, writes: List[Dict[str, Any]]) -> int:
        self.assert_open()
        return self.apply_commit(writes, self.next_seq)

    async def conversation(self, cid: int) -> Optional[Any]:
        self.assert_open()
        found = self.get_conversation(cid)
        return None if found is None else _detach(found)

    async def scan_conversations(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        page = self.scan_conversations_sync(query, limit, cursor)
        return Page(items=[_detach(item) for item in page.items], next=page.next)

    async def entry_global(self, eid: int):
        self.assert_open()
        found = self.get_entry_global(eid)
        return None if found is None else (_detach(found[0]), found[1])

    async def entry_visible(self, cid: int, eid: int):
        self.assert_open()
        found = self.get_entry_visible(cid, eid)
        return None if found is None else (_detach(found[0]), found[1])

    async def find_latest_head_marker(self, cid: int, at_or_before: Optional[int] = None):
        self.assert_open()
        found = self.find_latest_head_marker_sync(cid, at_or_before)
        return None if found is None else _detach(found)

    async def scan_entries(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        page = self.scan_entries_sync(query, limit, cursor)
        return Page(items=[_detach(item) for item in page.items], next=page.next)

    async def task(self, tid: int) -> Optional[Any]:
        self.assert_open()
        found = self.get_task(tid)
        return None if found is None else _detach(found)

    async def scan_tasks(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        page = self.scan_tasks_sync(query, limit, cursor)
        return Page(items=[_detach(item) for item in page.items], next=page.next)

    async def submission(self, sid: int) -> Optional[Any]:
        self.assert_open()
        found = self.get_submission(sid)
        return None if found is None else _detach(found)

    async def scan_submissions(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        page = self.scan_submissions_sync(query, limit, cursor)
        return Page(items=[_detach(item) for item in page.items], next=page.next)

    async def submission_by_request(self, cid: int, request_id: str) -> Optional[Any]:
        self.assert_open()
        found = self.get_submission_by_request(cid, request_id)
        return None if found is None else _detach(found)

    async def find_document(self, address: Dict[str, Any], at="current") -> Optional[Any]:
        self.assert_open()
        found = self.find_document_sync(address, at)
        return None if found is None else _detach(found)

    async def document(self, rid: int, at="current") -> Optional[Any]:
        self.assert_open()
        found = self.get_document(rid, at)
        return None if found is None else _detach(found)

    async def scan_documents(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        page = self.scan_documents_sync(query, limit, cursor)
        return Page(items=[_detach(item) for item in page.items], next=page.next)

    async def close(self) -> None:
        self.closed = True

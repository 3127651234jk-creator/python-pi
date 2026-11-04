"""In-memory reference storage backend.

Python port of ``packages/durable/src/storage/memory.ts``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..types import Page
from .base import BaseStorage


class MemoryStorage(BaseStorage):
    """Detached in-memory backend; reads return the live immutable records."""

    async def commit(self, writes: List[Dict[str, Any]]) -> int:
        self.assert_open()
        return self.apply_commit(writes, self.next_seq)

    async def mint_id(self) -> int:
        self.assert_open()
        value = self.next_id
        self.next_id += 1
        return value

    async def conversation(self, cid: int) -> Optional[Any]:
        self.assert_open()
        return self.get_conversation(cid)

    async def scan_conversations(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        return self.scan_conversations_sync(query, limit, cursor)

    async def entry_global(self, eid: int):
        self.assert_open()
        return self.get_entry_global(eid)

    async def entry_visible(self, cid: int, eid: int):
        self.assert_open()
        return self.get_entry_visible(cid, eid)

    async def find_latest_head_marker(self, cid: int, at_or_before: Optional[int] = None):
        self.assert_open()
        return self.find_latest_head_marker_sync(cid, at_or_before)

    async def scan_entries(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        return self.scan_entries_sync(query, limit, cursor)

    async def task(self, tid: int) -> Optional[Any]:
        self.assert_open()
        return self.get_task(tid)

    async def scan_tasks(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        return self.scan_tasks_sync(query, limit, cursor)

    async def submission(self, sid: int) -> Optional[Any]:
        self.assert_open()
        return self.get_submission(sid)

    async def scan_submissions(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        return self.scan_submissions_sync(query, limit, cursor)

    async def submission_by_request(self, cid: int, request_id: str) -> Optional[Any]:
        self.assert_open()
        return self.get_submission_by_request(cid, request_id)

    async def find_document(self, address: Dict[str, Any], at="current") -> Optional[Any]:
        self.assert_open()
        return self.find_document_sync(address, at)

    async def document(self, rid: int, at="current") -> Optional[Any]:
        self.assert_open()
        return self.get_document(rid, at)

    async def scan_documents(self, query: Dict[str, Any], limit: int, cursor: Optional[Dict[str, Any]] = None) -> Page:
        self.assert_open()
        return self.scan_documents_sync(query, limit, cursor)

    async def close(self) -> None:
        self.closed = True

"""SQLite storage backend.

Python port of ``packages/durable/src/storage/sqlite``, using the standard
library ``sqlite3`` module instead of ``node:sqlite``. One table per durable
record type plus a commit log table; each commit is one SQL transaction.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from ..types import Page, record_from_dict, record_to_dict
from .base import BaseStorage
from .memory import MemoryStorage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ids (kind TEXT PRIMARY KEY, next_id INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS commits (seq INTEGER PRIMARY KEY, writes TEXT NOT NULL);
"""


class SqliteStorage(MemoryStorage):
    """Commit-per-transaction backend over a sqlite database file."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.executescript(_SCHEMA)
        self._replay()

    def _replay(self) -> None:
        cursor = self.conn.execute("SELECT seq, writes FROM commits ORDER BY seq")
        for seq, payload in cursor:
            batch = json.loads(payload)
            self.apply_commit(self._decode_writes(batch), seq)

    def _encode_writes(self, writes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        encoded: List[Dict[str, Any]] = []
        for write in writes:
            item = dict(write)
            if "value" in item and not isinstance(item["value"], dict):
                item["value"] = record_to_dict(item["value"])
            encoded.append(item)
        return encoded

    def _decode_writes(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decoded: List[Dict[str, Any]] = []
        for item in items:
            write = dict(item)
            if "value" in write and isinstance(write["value"], dict):
                write["value"] = record_from_dict(write["value"])
            decoded.append(write)
        return decoded

    async def commit(self, writes: List[Dict[str, Any]]) -> int:
        self.assert_open()
        resolved = [self.resolve_document_copy(w) if w["type"] == "document.copy" else w for w in writes]
        seq = self.next_seq
        # Validate before opening the transaction so rejections leave no rows.
        self.check_global_ids(resolved)
        actions = self.prepare_document_actions(resolved)
        self.check_document_actions(actions)
        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "INSERT INTO commits (seq, writes) VALUES (?, ?)",
                (seq, json.dumps(self._encode_writes(resolved), ensure_ascii=False)),
            )
            self.conn.execute("COMMIT")
        except sqlite3.Error as exc:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise RuntimeError(f"sqlite commit failed: {exc}") from exc
        self.apply_writes(resolved, actions, seq)
        self.next_seq = seq + 1
        return seq

    async def close(self) -> None:
        self.closed = True
        self.conn.close()

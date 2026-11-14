"""JSONL commit-log storage backend.

Python port of ``packages/durable/src/storage/jsonl``. Every commit appends one
JSON line (``{"seq", "writes"}``) and is flushed before the commit resolves;
opening the file replays the log into the in-memory indexes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..errors import StorageRejected
from ..types import Page, record_from_dict, record_to_dict
from .base import BaseStorage
from .memory import MemoryStorage


class JsonlStorage(MemoryStorage):
    """Commit-log backend over a JSONL file."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            self._replay()
        else:
            with open(self.path, "a", encoding="utf-8"):
                pass

    def _replay(self) -> None:
        with open(self.path, "r+b") as handle:
            while True:
                start = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                # The newline completes a log record. A crash can leave a
                # partial JSON value or even a partial UTF-8 character at EOF.
                if not raw_line.endswith(b"\n"):
                    handle.truncate(start)
                    handle.flush()
                    os.fsync(handle.fileno())
                    break
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                batch = json.loads(line)
                self.apply_commit(self._decode_writes(batch["writes"]), batch["seq"])

    def _encode_writes(self, writes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        encoded: List[Dict[str, Any]] = []
        for write in writes:
            item = dict(write)
            if "value" in item and not isinstance(item["value"], dict):
                item["value"] = record_to_dict(item["value"])
            if "record" in item and isinstance(item["record"], object) and not isinstance(item["record"], dict):
                item["record"] = record_to_dict(item["record"])
            encoded.append(item)
        return encoded

    def _decode_writes(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from ..types import DocumentRecord

        decoded: List[Dict[str, Any]] = []
        for item in items:
            write = dict(item)
            if "value" in write and isinstance(write["value"], dict):
                write["value"] = record_from_dict(write["value"])
            if "record" in write and isinstance(write["record"], dict):
                write["record"] = (
                    write["record"]
                    if write["type"] not in ("document.create", "document.copy")
                    else write["record"]
                )
            decoded.append(write)
        return decoded

    async def commit(self, writes: List[Dict[str, Any]]) -> int:
        self.assert_open()
        self.reject_copy_source_conflicts(writes)
        resolved = [self.resolve_document_copy(w) if w["type"] == "document.copy" else w for w in writes]
        seq = self.next_seq
        # Validate first so a rejected commit leaves no log line behind.
        self.check_global_ids(resolved)
        actions = self.prepare_document_actions(resolved)
        self.check_document_actions(actions)
        try:
            payload = json.dumps({"seq": seq, "writes": self._encode_writes(resolved)}, ensure_ascii=True) + "\n"
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            raise StorageRejected(f"Commit is not JSON-serializable: {exc}") from exc
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.apply_writes(resolved, actions, seq)
        self.next_seq = seq + 1
        return seq

    async def close(self) -> None:
        self.closed = True

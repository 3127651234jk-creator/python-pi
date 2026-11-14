"""Append-only JSONL session trees and active-branch export."""

from __future__ import annotations

import json
import os
import time
import uuid
import html
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional
from ..ai.json_io import message_from_json, nested_calls_from_json, usage_from_json
from ..ai.types import Message, get_message_text

from ..ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolReference,
    ToolResultMessage,
    Tool,
    Usage,
    UserMessage,
)


def _to_dict(message: Any) -> Any:
    if is_dataclass(message) and not isinstance(message, type):
        return asdict(message)
    if isinstance(message, dict):
        return message
    raise TypeError(f"Cannot serialize message of type {type(message).__name__}")


def _from_dict(data: dict) -> Any:
    role = data.get("role")
    if role == "system":
        return SystemMessage(
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0),
            sections=data.get("sections"),
            tools_added=[Tool(**t) for t in (data.get("tools_added") or [])] or None,
            tools_removed=[ToolReference(**t) for t in (data.get("tools_removed") or [])] or None,
        )
    if role == "user":
        return UserMessage(content=data.get("content", ""), timestamp=data.get("timestamp", 0))
    if role == "assistant":
        content = []
        for block in data.get("content", []):
            kind = block.get("type")
            if kind == "text":
                content.append(TextContent(text=block.get("text", "")))
            elif kind == "thinking":
                content.append(
                    ThinkingContent(
                        thinking=block.get("thinking", ""),
                        thinking_signature=block.get("thinking_signature"),
                    )
                )
            elif kind == "toolCall":
                content.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("arguments", {})))
        usage_data = data.get("usage", {})
        return AssistantMessage(
            content=content,
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage=Usage(**{k: v for k, v in usage_data.items() if k in Usage.__dataclass_fields__}),
            stop_reason=data.get("stop_reason", "stop"),
            timestamp=data.get("timestamp", 0),
            error_message=data.get("error_message"),
            raw_stop_reason=data.get("raw_stop_reason"),
            thinking_level=data.get("thinking_level"),
            retried=bool(data.get("retried", False)),
        )
    if role == "toolResult":
        content = []
        for block in data.get("content", []):
            if block.get("type") == "image":
                content.append(ImageContent(data=block.get("data", ""), mime_type=block.get("mime_type", "")))
            else:
                content.append(TextContent(text=block.get("text", "")))
        usage_data = data.get("usage")
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=content,
            is_error=data.get("is_error", False),
            timestamp=data.get("timestamp", 0),
            details=data.get("details"),
            usage=Usage(**{k: v for k, v in usage_data.items() if k in Usage.__dataclass_fields__}) if usage_data else None,
            nested_calls=nested_calls_from_json(data.get("nested_calls")),
        )
    raise ValueError(f"Unknown message role: {role!r}")


@dataclass
class SessionEntry:
    id: str
    parent_id: Optional[str]
    message: Optional[Message]  # None for custom (non-message) entries
    custom: Optional[dict] = None  # {"kind": str, "data": dict}
    timestamp: str = ""  # UTC ISO timestamp (TS session files always carry one)


class SessionManager:
    """Append transcript messages to a JSONL session file."""

    def __init__(self, session_dir: Optional[str] = None, cwd: Optional[str] = None) -> None:
        self.cwd = cwd or os.getcwd()
        self.session_dir = session_dir or os.path.join(self.cwd, ".pi", "sessions")
        self.session_file: Optional[str] = None
        self.entries: List[SessionEntry] = []
        self.leaf_id: Optional[str] = None
        self.name = ""
        self._incomplete_tail_offset: Optional[int] = None
        os.makedirs(self.session_dir, exist_ok=True)

    def start_new(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = os.path.join(self.session_dir, f"{stamp}-{os.getpid()}")
        suffix = 0
        while True:
            path = f"{candidate}{f'-{suffix}' if suffix else ''}.jsonl"
            if not os.path.exists(path):
                break
            suffix += 1
        # Touch the file so session pickers can find it before the first append.
        with open(path, "a", encoding="utf-8"):
            pass
        self.session_file = path
        self._incomplete_tail_offset = None
        self.entries = []
        self.leaf_id = None
        self.name = ""
        return path

    def _write(self, record: dict) -> None:
        if self.session_file is None:
            self.start_new()
        assert self.session_file is not None
        if self._incomplete_tail_offset is not None:
            with open(self.session_file, "r+b") as handle:
                handle.truncate(self._incomplete_tail_offset)
            self._incomplete_tail_offset = None
        with open(self.session_file, "rb+") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
        with open(self.session_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, message: Message) -> str:
        if self.session_file is None:
            self.start_new()
        entry = SessionEntry(uuid.uuid4().hex, self.leaf_id, message, timestamp=_now_iso())
        self._write({"type": "message", "id": entry.id, "parent_id": entry.parent_id,
                     "timestamp": entry.timestamp, "message": _to_dict(message)})
        self.entries.append(entry)
        self.leaf_id = entry.id
        return entry.id

    def messages(self) -> List[Message]:
        return [entry.message for entry in self.get_branch() if entry.message is not None]

    def get_branch(self, entry_id: Optional[str] = None) -> List[SessionEntry]:
        index = {entry.id: entry for entry in self.entries}
        cursor = self.leaf_id if entry_id is None else entry_id
        branch = []
        seen = set()
        while cursor is not None:
            if cursor not in index or cursor in seen:
                raise ValueError("Invalid session ancestry")
            seen.add(cursor)
            entry = index[cursor]
            branch.append(entry)
            cursor = entry.parent_id
        return list(reversed(branch))

    # -- custom / metadata entries -----------------------------------------

    def append_custom(self, custom_type: str, data: dict) -> str:
        """State-only entry: replayed on load, never enters the LLM context."""
        if self.session_file is None:
            self.start_new()
        entry = SessionEntry(uuid.uuid4().hex, self.leaf_id, None, {"kind": custom_type, "data": data},
                             timestamp=_now_iso())
        self._write({"type": "custom", "id": entry.id, "parent_id": entry.parent_id,
                     "timestamp": entry.timestamp, "customType": custom_type, "data": data})
        self.entries.append(entry)
        self.leaf_id = entry.id
        return entry.id

    def append_custom_message(self, custom_type: str, content, display: bool = True,
                              details=None) -> str:
        """Custom entry that does enter the LLM context as a user message."""
        message = UserMessage(content=content, timestamp=_now_ms())
        metadata = {"display": display}
        if details is not None:
            metadata["details"] = details
        entry = SessionEntry(uuid.uuid4().hex, self.leaf_id, message, {"kind": custom_type, "data": metadata},
                             timestamp=_now_iso())
        self._write({"type": "custom_message", "id": entry.id, "parent_id": entry.parent_id,
                     "timestamp": entry.timestamp, "customType": custom_type, "display": display,
                     "details": details, "message": _to_dict(message)})
        self.entries.append(entry)
        self.leaf_id = entry.id
        return entry.id

    def append_compaction(self, summary: str, tokens_before: int = 0, checkpoint: Optional[dict] = None,
                          usage: Optional[Usage] = None) -> str:
        data: dict = {"summary": summary, "tokens_before": tokens_before}
        if checkpoint is not None:
            data["checkpoint"] = checkpoint
        if usage is not None:
            data["usage"] = usage.to_dict()
        return self.append_custom("compaction", data)

    def append_model_change(self, provider: str, model_id: str) -> str:
        return self.append_custom("model_change", {"provider": provider, "modelId": model_id})

    def append_thinking_change(self, level: str) -> str:
        return self.append_custom("thinking_level_change", {"thinkingLevel": level})

    def append_branch_summary(self, from_id: str, summary: str) -> str:
        return self.append_custom("branch_summary", {"fromId": from_id, "summary": summary})

    def custom_entries(self, custom_type: Optional[str] = None) -> List[SessionEntry]:
        return [entry for entry in self.entries
                if entry.custom is not None and (custom_type is None or entry.custom["kind"] == custom_type)]

    def compaction_usages(self) -> List[Usage]:
        """Usage recorded on this branch's compaction entries (summary-request billing)."""
        result: List[Usage] = []
        for entry in self.get_branch():
            if entry.custom is not None and entry.custom["kind"] == "compaction":
                data = entry.custom["data"].get("usage")
                if isinstance(data, dict):
                    result.append(usage_from_json(data))
        return result

    def latest_custom(self, custom_type: str) -> Optional[dict]:
        """Last custom entry payload of a type on the active branch."""
        found = None
        for entry in self.get_branch():
            if entry.custom is not None and entry.custom["kind"] == custom_type:
                found = entry.custom["data"]
        return found

    def branch(self, entry_id: str) -> List[Message]:
        matches = [entry.id for entry in self.entries if entry.id.startswith(entry_id)]
        if len(matches) != 1:
            raise ValueError("Entry ID is unknown or ambiguous")
        target = matches[0]
        self._validate_branch(self.get_branch(target))
        current = {entry.id for entry in self.get_branch()}
        kept = {entry.id for entry in self.get_branch(target)}
        abandoned = [entry for entry in self.get_branch() if entry.id not in kept]
        self._write({"type": "leaf", "id": target})
        self.leaf_id = target
        if abandoned:
            # Leave a digest of the abandoned tail so the model knows what is
            # no longer in context after the switch.
            digest = _branch_digest(abandoned)
            if digest:
                self.append_branch_summary(abandoned[-1].id, digest)
                self.append(SystemMessage(
                    content="Switched to an earlier branch. Work done on the abandoned branch, "
                            "no longer in context:\n" + digest, timestamp=_now_ms()))
        return self.messages()

    def _validate_branch(self, branch: List[SessionEntry]) -> None:
        pending = set()
        for entry in branch:
            message = entry.message
            if message is None:
                continue  # custom entries carry no tool protocol state
            if message.role == "assistant":
                pending.update(block.id for block in message.content
                               if isinstance(block, ToolCall) and block.raw_item is None)
            elif isinstance(message, ToolResultMessage):
                pending.discard(message.tool_call_id)
        if pending:
            raise ValueError("Select an entry after all tool results, or before the tool call")

    def tree(self) -> str:
        depths = {}
        lines = []
        for entry in self.entries:
            depth = depths.get(entry.parent_id, -1) + 1
            depths[entry.id] = depth
            if entry.message is None:
                preview = f"[{entry.custom['kind'] if entry.custom else 'custom'}]"
            else:
                preview = get_message_text(entry.message.content).replace("\n", " ")[:80]
            marker = "*" if entry.id == self.leaf_id else " "
            lines.append(f"{marker} {'  ' * depth}{entry.id} {entry.message.role if entry.message else 'custom'}: {preview}")
        return "\n".join(lines)

    def rename(self, name: str) -> None:
        self._write({"type": "name", "name": name})
        self.name = name

    def list_sessions(self) -> List[dict]:
        result = []
        for filename in os.listdir(self.session_dir):
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(self.session_dir, filename)
            reader = SessionManager(self.session_dir, self.cwd)
            try:
                messages = reader.load(path)
                result.append({"id": filename[:-6], "path": path, "name": reader.name,
                               "messages": len(messages), "modified": os.path.getmtime(path)})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda item: item["modified"], reverse=True)

    def resume(self, identifier: str) -> List[Message]:
        if os.path.isfile(identifier):
            return self.load(identifier)
        sessions = self.list_sessions()
        matches = [item for item in sessions if item["id"] == identifier]
        if not matches:
            matches = [item for item in sessions if item["id"].startswith(identifier)]
        if len(matches) != 1:
            raise ValueError("Session ID is unknown or ambiguous")
        return self.load(matches[0]["path"])

    def fork(self, entry_id: Optional[str] = None) -> List[Message]:
        if entry_id:
            matches = [entry.id for entry in self.entries if entry.id.startswith(entry_id)]
            if len(matches) != 1:
                raise ValueError("Entry ID is unknown or ambiguous")
            branch = self.get_branch(matches[0])
        else:
            branch = self.get_branch()
        self._validate_branch(branch)
        self.start_new()
        for entry in branch:
            if entry.message is not None and entry.custom is not None:
                copied = SessionEntry(uuid.uuid4().hex, self.leaf_id, entry.message, entry.custom,
                                      timestamp=entry.timestamp)
                self._write({"type": "custom_message", "id": copied.id, "parent_id": copied.parent_id,
                             "timestamp": copied.timestamp, "customType": copied.custom["kind"],
                             "display": copied.custom["data"].get("display", True),
                             "details": copied.custom["data"].get("details"),
                             "message": _to_dict(copied.message)})
                self.entries.append(copied)
                self.leaf_id = copied.id
            elif entry.message is not None:
                self.append(entry.message)
            elif entry.custom is not None:
                self.append_custom(entry.custom["kind"], entry.custom["data"])
        return self.messages()

    def branch_jsonl(self) -> str:
        """Active branch as JSONL records (same shape as a .jsonl export)."""
        records = [{"type": "name", "name": self.name}]
        for entry in self.get_branch():
            common = {"id": entry.id, "parent_id": entry.parent_id, "timestamp": entry.timestamp}
            if entry.custom is not None and entry.message is None:
                records.append({"type": "custom", **common,
                                "customType": entry.custom["kind"], "data": entry.custom["data"]})
            elif entry.custom is not None:
                records.append({"type": "custom_message", **common,
                                "customType": entry.custom["kind"], "display": entry.custom["data"].get("display", True),
                                "details": entry.custom["data"].get("details"),
                                "message": _to_dict(entry.message)})
            else:
                records.append({"type": "message", **common,
                                "message": _to_dict(entry.message)})
        return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)

    def export(self, path: str) -> str:
        path = os.path.abspath(path)
        if self.session_file and os.path.normcase(path) == os.path.normcase(os.path.abspath(self.session_file)):
            raise ValueError("Cannot export over the active session")
        extension = os.path.splitext(path)[1].lower()
        if extension not in (".jsonl", ".html", ".md"):
            raise ValueError("Export extension must be .jsonl, .html or .md")
        branch = self.get_branch()
        if extension == ".jsonl":
            content = self.branch_jsonl()
        else:
            text = "\n\n".join(f"## {e.message.role}\n\n{get_message_text(e.message.content)}"
                               for e in branch if e.message is not None)
            content = ("<!doctype html><meta charset='utf-8'><title>Pi session</title><pre>"
                       + html.escape(text) + "</pre>") if extension == ".html" else text
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def load(self, path: str) -> List[Any]:
        entries = []
        leaf = None
        name = ""
        incomplete_tail_offset = None
        with open(path, "rb") as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                try:
                    line = raw_line.decode("utf-8").strip()
                    record = json.loads(line) if line else None
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if not raw_line.endswith(b"\n"):
                        incomplete_tail_offset = offset
                        break
                    raise ValueError(f"Invalid session JSON in {path} at byte {offset}") from None
                if not line:
                    continue
                if not isinstance(record, dict):
                    raise ValueError("Session line must be a JSON object")
                kind = record.get("type")
                if kind == "session":
                    # TypeScript pi session header; delegate to the TS reader.
                    return self.load_ts(path)
                if kind == "name":
                    name = record["name"]
                elif kind == "leaf":
                    leaf = record["id"]
                    if leaf not in {e.id for e in entries}:
                        raise ValueError("Unknown active entry")
                elif kind == "custom" or kind == "custom_message":
                    entry_id = record["id"]
                    parent = record.get("parent_id")
                    if entry_id in {e.id for e in entries} or (parent is not None and parent not in {e.id for e in entries}):
                        raise ValueError("Invalid session entry")
                    if kind == "custom":
                        entries.append(SessionEntry(entry_id, parent, None,
                            {"kind": record.get("customType", ""), "data": record.get("data", {})},
                            timestamp=record.get("timestamp", "")))
                    else:
                        metadata = {"display": record.get("display", True)}
                        if record.get("details") is not None:
                            metadata["details"] = record["details"]
                        entries.append(SessionEntry(entry_id, parent, message_from_json(record["message"]),
                            {"kind": record.get("customType", ""), "data": metadata},
                            timestamp=record.get("timestamp", "")))
                    leaf = entry_id
                else:
                    entry_id = record["id"] if kind == "message" else f"legacy-{len(entries)}"
                    parent = record.get("parent_id") if kind == "message" else leaf
                    if entry_id in {e.id for e in entries} or (parent is not None and parent not in {e.id for e in entries}):
                        raise ValueError("Invalid session entry")
                    message = message_from_json(record["message"] if kind == "message" else record)
                    entries.append(SessionEntry(entry_id, parent, message, timestamp=record.get("timestamp", "")))
                    leaf = entry_id
        self.session_file = path
        self._incomplete_tail_offset = incomplete_tail_offset
        self.entries = entries
        self.leaf_id = leaf
        self.name = name
        return self.messages()

    def load_ts(self, path: str) -> List[Any]:
        """Load a TypeScript pi session file (version 3 JSONL)."""
        from . import ts_interop

        parsed = ts_interop.read_ts_file(path)
        entries = []
        leaf = None
        for record in parsed["records"]:
            entry = SessionEntry(record["id"], record["parent_id"], record.get("message"), record.get("custom"),
                                 timestamp=record.get("timestamp", ""))
            if record["parent_id"] is not None and record["parent_id"] not in {e.id for e in entries}:
                raise ValueError("Invalid TS session entry")
            entries.append(entry)
            leaf = entry.id
        self.session_file = path
        self.entries = entries
        self.leaf_id = leaf
        self.name = parsed["name"]
        return self.messages()

    def export_ts(self, path: str) -> str:
        """Export the active branch as a TypeScript pi session file."""
        from . import ts_interop

        if self.session_file and os.path.normcase(path) == os.path.normcase(os.path.abspath(self.session_file)):
            raise ValueError("Cannot export over the active session")
        if os.path.splitext(path)[1].lower() != ".jsonl":
            raise ValueError("TS export requires a .jsonl path")
        return ts_interop.write_ts_file(path, self)

    def latest_session_file(self) -> Optional[str]:
        if not os.path.isdir(self.session_dir):
            return None
        candidates = [
            os.path.join(self.session_dir, name)
            for name in os.listdir(self.session_dir)
            if name.endswith(".jsonl")
        ]
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    """UTC ISO timestamp in the TS ``toISOString`` shape (millisecond precision)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + ("%03dZ" % (now.microsecond // 1000))


def _branch_digest(entries: List[SessionEntry]) -> str:
    """Compact digest of abandoned branch entries for the leave summary."""
    lines = []
    for entry in entries[-12:]:
        if entry.message is not None:
            text = " ".join(get_message_text(entry.message.content).split())[:120]
            lines.append(f"- {entry.message.role}: {text}" if text else f"- {entry.message.role}")
        elif entry.custom is not None and entry.custom["kind"] in ("model_change", "compaction"):
            lines.append(f"- [{entry.custom['kind']}]")
    return "\n".join(lines)

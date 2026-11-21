"""TypeScript session-file interop tests: fixture reading and round-trips."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi.coding_agent.session import SessionManager
from pi.coding_agent.ts_interop import TS_VERSION, read_ts_file, write_ts_file

# Hand-written TS pi v3 session: header, name, three messages, metadata
# entries, a custom pair and an unknown label entry that must round-trip.
FIXTURE_LINES = [
    {"type": "session", "version": 3, "id": "s1", "timestamp": 1700000000000, "cwd": "/tmp"},
    {"type": "session_info", "id": "i1", "parentId": None, "timestamp": 1700000000000, "name": "my session"},
    {"type": "message", "id": "e1", "parentId": None, "timestamp": 1,
     "message": {"role": "user", "content": "hello", "timestamp": 1}},
    {"type": "message", "id": "e2", "parentId": "e1", "timestamp": 2,
     "message": {"role": "assistant", "timestamp": 2, "api": "anthropic-messages", "provider": "anthropic",
                 "model": "claude-test", "stopReason": "toolUse",
                 "usage": {"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1, "totalTokens": 18,
                           "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.01, "cacheWrite": 0.02, "total": 0.33}},
                 "content": [{"type": "text", "text": "hi"},
                             {"type": "toolCall", "id": "t1", "name": "read", "arguments": {"path": "x"}}]}},
    {"type": "message", "id": "e3", "parentId": "e2", "timestamp": 3,
     "message": {"role": "toolResult", "toolCallId": "t1", "toolName": "read", "isError": False, "timestamp": 3,
                 "details": {"k": 1},
                 "content": [{"type": "text", "text": "contents"}]}},
    {"type": "model_change", "id": "e4", "parentId": "e3", "timestamp": 4, "provider": "faux", "modelId": "faux"},
    {"type": "thinking_level_change", "id": "e5", "parentId": "e4", "timestamp": 5, "thinkingLevel": "high"},
    {"type": "compaction", "id": "e6", "parentId": "e5", "timestamp": 6, "summary": "compressed", "tokensBefore": 99},
    {"type": "custom_message", "id": "e7", "parentId": "e6", "timestamp": 7, "customType": "note",
     "display": True, "content": "hello"},
    {"type": "custom", "id": "e8", "parentId": "e7", "timestamp": 8, "customType": "state", "data": {"n": 1}},
    {"type": "label", "id": "e9", "parentId": "e8", "timestamp": 9, "entryId": "e1", "label": "pin"},
]

FIXTURE_TEXT = "".join(json.dumps(line) + "\n" for line in FIXTURE_LINES)


def write_fixture(tmp_path, text: str = FIXTURE_TEXT) -> str:
    path = tmp_path / "ts-session.jsonl"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_read_ts_fixture(tmp_path):
    parsed = read_ts_file(write_fixture(tmp_path))
    assert parsed["name"] == "my session"
    records = parsed["records"]
    assert [r["id"] for r in records] == ["i1", *[f"e{i}" for i in range(1, 10)]]
    assert records[0]["custom"]["kind"] == "ts_session_info"
    records = records[1:]
    assert [r["parent_id"] for r in records] == [None, "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]

    user, assistant, result = records[0]["message"], records[1]["message"], records[2]["message"]
    assert user.content == "hello"
    assert isinstance(assistant, AssistantMessage)
    assert assistant.content[1] == ToolCall(id="t1", name="read", arguments={"path": "x"})
    assert assistant.usage == Usage(input=10, output=5, cache_read=2, cache_write=1,
                                    cost={"input": 0.1, "output": 0.2, "cache_read": 0.01,
                                          "cache_write": 0.02, "cache_write_1h": 0.0, "total": 0.33})
    assert isinstance(result, ToolResultMessage)
    assert result.details == {"k": 1} and result.is_error is False

    # Metadata entries map onto custom kinds with snake_case payloads.
    assert records[3]["custom"] == {"kind": "model_change", "data": {"provider": "faux", "modelId": "faux"}}
    assert records[4]["custom"] == {"kind": "thinking_level_change", "data": {"thinkingLevel": "high"}}
    assert records[5]["custom"] == {"kind": "compaction", "data": {"summary": "compressed", "tokens_before": 99}}
    # custom_message becomes a user message plus a custom marker.
    assert records[6]["message"].content == "hello"
    assert records[6]["custom"] == {"kind": "note", "data": {"display": True}}
    assert records[7]["custom"] == {"kind": "state", "data": {"n": 1}}
    # Unknown entry types round-trip opaquely under a ts_ prefix.
    assert records[8]["custom"]["kind"] == "ts_label"
    assert records[8]["custom"]["data"]["label"] == "pin"


def test_read_ts_tolerates_bom_and_missing_ids(tmp_path):
    lines = [FIXTURE_LINES[0],
             {"type": "message", "timestamp": 1, "message": {"role": "user", "content": "no id", "timestamp": 1}}]
    text = "".join(json.dumps(line) + "\n" for line in lines)
    path = tmp_path / "bom.jsonl"
    path.write_text("\ufeff" + text, encoding="utf-8")
    parsed = read_ts_file(str(path))
    assert len(parsed["records"]) == 1
    assert parsed["records"][0]["message"].content == "no id"
    assert parsed["records"][0]["id"]  # generated when the fixture has none


def test_session_info_in_active_chain_remains_loadable(tmp_path):
    lines = [
        FIXTURE_LINES[0],
        {"type": "message", "id": "m1", "parentId": None, "timestamp": 1,
         "message": {"role": "user", "content": "before", "timestamp": 1}},
        {"type": "session_info", "id": "n1", "parentId": "m1", "timestamp": 2,
         "name": "renamed"},
        {"type": "message", "id": "m2", "parentId": "n1", "timestamp": 3,
         "message": {"role": "user", "content": "after", "timestamp": 3}},
    ]
    path = write_fixture(tmp_path, "".join(json.dumps(line) + "\n" for line in lines))
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))

    assert [message.content for message in manager.load(path)] == ["before", "after"]
    assert manager.name == "renamed"
    assert manager.custom_entries("ts_session_info")[0].id == "n1"


def test_context_edits_project_active_messages(tmp_path):
    lines = [
        FIXTURE_LINES[0],
        {"type": "message", "id": "m1", "parentId": None, "timestamp": 1,
         "message": {"role": "user", "content": "remove", "timestamp": 1}},
        {"type": "message", "id": "m2", "parentId": "m1", "timestamp": 2,
         "message": {"role": "user", "content": "old", "timestamp": 2}},
        {"type": "context_edit", "id": "x1", "parentId": "m2", "timestamp": 3,
         "targetId": "m1", "replacement": None},
        {"type": "context_edit", "id": "x2", "parentId": "x1", "timestamp": 4,
         "targetId": "m2", "replacement": {"content": "new"}},
    ]
    path = write_fixture(tmp_path, "".join(json.dumps(line) + "\n" for line in lines))
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))

    assert [message.content for message in manager.load(path)] == ["new"]


def test_session_roundtrip_through_ts_format(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    manager.rename("roundtrip")
    manager.append(UserMessage(content="hello", timestamp=1))
    manager.append(AssistantMessage(
        content=[TextContent(text="using a tool"), ToolCall(id="t1", name="read", arguments={"path": "x"})],
        api="anthropic-messages", provider="anthropic", model="claude-test",
        usage=Usage(input=10, output=5), stop_reason="toolUse", timestamp=2))
    manager.append(ToolResultMessage(
        tool_call_id="t1", tool_name="read", content=[TextContent(text="contents")],
        is_error=False, timestamp=3, details={"k": 1}))
    manager.append_model_change("faux", "faux")
    manager.append_compaction("compressed", tokens_before=99)
    manager.append_custom_message("note", "imported")

    exported = manager.export_ts(str(tmp_path / "out.jsonl"))
    header = json.loads(open(exported, encoding="utf-8").readline())
    assert header["type"] == "session" and header["version"] == TS_VERSION

    loaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    messages = loaded.load_ts(exported)
    assert loaded.name == "roundtrip"
    assert [m.role for m in messages] == ["user", "assistant", "toolResult", "user"]
    assert messages[1].content[1] == ToolCall(id="t1", name="read", arguments={"path": "x"})
    assert messages[1].usage.input == 10 and messages[1].stop_reason == "toolUse"
    assert messages[2].details == {"k": 1}
    assert loaded.latest_custom("compaction") == {"summary": "compressed", "tokens_before": 99}
    assert loaded.latest_custom("model_change") == {"provider": "faux", "modelId": "faux"}
    # The custom message re-enters as its own entry with the original kind.
    notes = loaded.custom_entries("note")
    assert len(notes) == 1 and notes[0].message.content == "imported"


def test_append_after_ts_load_stays_in_ts_tree(tmp_path):
    """A resumed TS session must keep writing camelCase ``parentId`` records."""
    lines = [
        FIXTURE_LINES[0],
        {"type": "message", "id": "u1", "parentId": None, "timestamp": "2024-01-01T00:00:01.000Z",
         "message": {"role": "user", "content": "first", "timestamp": 1}},
        {"type": "message", "id": "a1", "parentId": "u1", "timestamp": "2024-01-01T00:00:02.000Z",
         "message": {"role": "assistant", "api": "faux", "provider": "faux", "model": "faux",
                     "stopReason": "stop", "timestamp": 2,
                     "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                               "totalTokens": 0, "cost": {"total": 0}},
                     "content": [{"type": "text", "text": "reply"}]}},
    ]
    path = write_fixture(tmp_path, "".join(json.dumps(line) + "\n" for line in lines))
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    assert [m.role for m in manager.load(path)] == ["user", "assistant"]
    manager.append(UserMessage(content="third", timestamp=3))

    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert records[0]["type"] == "session"
    assert all("parentId" in record and "parent_id" not in record for record in records[1:])
    assert records[-1]["parentId"] == "a1"
    assert records[-1]["message"]["role"] == "user"

    reloaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    messages = reloaded.load(path)
    assert [(m.role, m.content if isinstance(m.content, str) else None) for m in messages] == [
        ("user", "first"), ("assistant", None), ("user", "third")]


def test_load_ts_resets_incomplete_tail_offset(tmp_path):
    """Loading a TS session must drop a pending truncation offset."""
    native = tmp_path / "native.jsonl"
    native.write_text(json.dumps({"type": "message", "id": "n1", "parent_id": None, "timestamp": "t",
                                  "message": {"role": "user", "content": "native", "timestamp": 1}}) + "\n"
                      + '{"type": "mess', encoding="utf-8")
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.load(str(native))
    assert manager._incomplete_tail_offset is not None

    path = write_fixture(tmp_path)
    manager.load_ts(path)
    assert manager._incomplete_tail_offset is None
    size_before = os.path.getsize(path)
    manager.append(UserMessage(content="after", timestamp=9))
    text = (tmp_path / "ts-session.jsonl").read_text(encoding="utf-8")
    assert os.path.getsize(path) > size_before
    assert text.startswith('{"type": "session"')
    assert "hello" in text and "after" in text


def test_metadata_appends_after_ts_load_use_ts_records(tmp_path):
    path = write_fixture(tmp_path)
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.load(path)
    manager.rename("renamed")
    manager.append_custom("state", {"n": 2})
    manager.append_compaction("sum", tokens_before=5, checkpoint={"length": 1, "last": "x"})

    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert [record["type"] for record in records[-3:]] == ["session_info", "custom", "compaction"]
    assert records[-3]["name"] == "renamed"
    assert all("parentId" in record for record in records[-3:])
    compaction = records[-1]
    assert compaction["firstKeptEntryId"] == "e2"
    assert compaction["checkpoint"] == {"length": 1, "last": "x"}

    reloaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    reloaded.load(path)
    assert reloaded.name == "renamed"
    assert reloaded.latest_custom("state") == {"n": 2}
    assert reloaded.latest_custom("compaction")["checkpoint"] == {"length": 1, "last": "x"}


def test_ts_prefixed_custom_kinds_roundtrip_after_ts_load(tmp_path):
    """Raw TS passthrough records keep their shape; extension ts_* kinds stay custom."""
    path = write_fixture(tmp_path)
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.load(path)
    # Opaque passthrough of the label record read from the file, re-appended.
    raw_label = next(e.custom["data"] for e in manager.entries if e.custom and e.custom["kind"] == "ts_label")
    manager.append_custom("ts_label", raw_label)
    # An extension's own ts_* kind carries no "type" key and must not lose it.
    manager.append_custom("ts_extension_state", {"count": 1})

    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert records[-2]["type"] == "label" and records[-2]["label"] == "pin"
    assert records[-1]["type"] == "custom" and records[-1]["customType"] == "ts_extension_state"

    reloaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    reloaded.load(path)
    labels = reloaded.custom_entries("ts_label")
    assert len(labels) == 2 and labels[-1].custom["data"]["label"] == "pin"
    state = reloaded.custom_entries("ts_extension_state")
    assert len(state) == 1 and state[0].custom["data"] == {"count": 1}


def test_export_ts_writes_compaction_first_kept_entry_id(tmp_path):
    """export_ts emits the TS-required firstKeptEntryId on compaction records."""
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    first = manager.append(UserMessage(content="one", timestamp=1))
    second = manager.append(UserMessage(content="two", timestamp=2))
    manager.append(UserMessage(content="three", timestamp=3))

    # Partial consumption: checkpoint length 1 keeps from the second message.
    manager.append_compaction("cut one", tokens_before=9, checkpoint={"length": 1, "last": "x"})
    exported = manager.export_ts(str(tmp_path / "cut.jsonl"))
    records = [json.loads(line) for line in open(exported, encoding="utf-8") if line.strip()]
    compaction = next(r for r in records if r.get("type") == "compaction")
    assert compaction["firstKeptEntryId"] == second
    assert compaction["firstKeptEntryId"] != first

    # Full consumption keeps nothing, which TS spells as the entry's own id.
    manager.append_compaction("cut all", tokens_before=9, checkpoint={"length": 3, "last": "y"})
    exported_all = manager.export_ts(str(tmp_path / "cut-all.jsonl"))
    records_all = [json.loads(line) for line in open(exported_all, encoding="utf-8") if line.strip()]
    compactions = [r for r in records_all if r.get("type") == "compaction"]
    assert len(compactions) == 2
    assert compactions[-1]["firstKeptEntryId"] == compactions[-1]["id"]

    # No checkpoint: unknown cut point falls back to the entry's own id.
    manager.append_compaction("no checkpoint", tokens_before=9)
    exported_legacy = manager.export_ts(str(tmp_path / "cut-legacy.jsonl"))
    legacy = [r for r in (json.loads(line) for line in open(exported_legacy, encoding="utf-8") if line.strip())
              if r.get("type") == "compaction"][-1]
    assert legacy["firstKeptEntryId"] == legacy["id"]

    # The exported cut point round-trips back into the portable checkpoint.
    reloaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    reloaded.load(exported)
    assert reloaded.latest_custom("compaction")["checkpoint"] == {"length": 1, "last": "x"}


def test_compaction_first_kept_respects_context_edits(tmp_path):
    """checkpoint indexes the projected transcript, not the raw entry list."""
    lines = [
        FIXTURE_LINES[0],
        {"type": "message", "id": "m1", "parentId": None, "timestamp": 1,
         "message": {"role": "user", "content": "dropped", "timestamp": 1}},
        {"type": "message", "id": "m2", "parentId": "m1", "timestamp": 2,
         "message": {"role": "user", "content": "kept", "timestamp": 2}},
        {"type": "message", "id": "m3", "parentId": "m2", "timestamp": 3,
         "message": {"role": "user", "content": "tail", "timestamp": 3}},
        {"type": "context_edit", "id": "x1", "parentId": "m3", "timestamp": 4,
         "targetId": "m1", "replacement": None},
    ]
    path = write_fixture(tmp_path, "".join(json.dumps(line) + "\n" for line in lines))
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    assert [m.content for m in manager.load(path)] == ["kept", "tail"]

    # checkpoint length 1 consumed the first *projected* message (m2/"kept"),
    # so TS must keep from m3, not from the omitted m1's successor in raw order.
    manager.append_compaction("sum", tokens_before=9, checkpoint={"length": 1, "last": "x"})
    exported = manager.export_ts(str(tmp_path / "out.jsonl"))
    compaction = next(r for r in (json.loads(line) for line in open(exported, encoding="utf-8") if line.strip())
                      if r.get("type") == "compaction")
    assert compaction["firstKeptEntryId"] == "m3"

    reloaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    reloaded.load(exported)
    assert reloaded.latest_custom("compaction")["checkpoint"] == {"length": 1, "last": "x"}


def test_compaction_usage_roundtrip(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    manager.append(UserMessage(content="hello", timestamp=1))
    manager.append_compaction("compressed", tokens_before=99,
                              usage=Usage(input=11, output=4, cost={"total": 0.02}))

    exported = manager.export_ts(str(tmp_path / "out.jsonl"))
    records = [json.loads(line) for line in open(exported, encoding="utf-8") if line.strip()]
    entry = next(r for r in records if r.get("type") == "compaction")
    assert entry["usage"]["input"] == 11 and entry["usage"]["output"] == 4
    assert entry["usage"]["totalTokens"] == 15
    assert entry["usage"]["cost"]["total"] == 0.02

    # A TS file whose compaction entry carries usage imports it into the payload.
    lines = [FIXTURE_LINES[0],
             {"type": "compaction", "id": "c1", "parentId": None, "timestamp": 5, "summary": "compressed",
              "tokensBefore": 99,
              "usage": {"input": 7, "output": 2, "cacheRead": 1, "cacheWrite": 0, "totalTokens": 10,
                        "cost": {"total": 0.03}}}]
    path = tmp_path / "usage.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    loaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    loaded.load(str(path))
    data = loaded.latest_custom("compaction")
    assert data["usage"]["input"] == 7 and data["usage"]["output"] == 2
    assert data["usage"]["cost"]["total"] == 0.03
    assert [u.input for u in loaded.compaction_usages()] == [7]


@pytest.mark.parametrize("content", ["plain 中文", [TextContent(text="block 中文")],
    [TextContent(text="look"), ImageContent(data="aGVsbG8=", mime_type="image/png")]])
def test_custom_message_export_after_native_reload(tmp_path, content):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    path = manager.start_new()
    manager.append_custom_message("note", content, display=False, details={"source": "test"})
    loaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    loaded.load(path)
    exported = loaded.export_ts(str(tmp_path / "custom.jsonl"))
    records = [json.loads(line) for line in (tmp_path / "custom.jsonl").read_text(encoding="utf-8").splitlines()]
    custom = next(record for record in records if record["type"] == "custom_message")
    assert custom["customType"] == "note"
    assert custom["display"] is False
    assert custom["details"] == {"source": "test"}
    if isinstance(content, list) and len(content) > 1:
        assert custom["content"][1]["mimeType"] == "image/png"
    restored = read_ts_file(exported)["records"][0]
    assert restored["message"].content == content
    assert restored["custom"]["data"]["details"] == {"source": "test"}


def test_export_refuses_non_jsonl_and_overwrite(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    manager.append(UserMessage(content="hello", timestamp=1))
    with pytest.raises(ValueError):
        manager.export_ts(str(tmp_path / "out.md"))
    exported = manager.export_ts(str(tmp_path / "out.jsonl"))
    with pytest.raises(FileExistsError):
        write_ts_file(exported, manager)


def test_tool_result_duration_ms_ts_roundtrip(tmp_path):
    """ToolResultMessage.duration_ms maps to durationMs in TS session files (#10549: F-AI-02)."""
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()
    manager.append(
        ToolResultMessage(tool_call_id="t1", tool_name="bash", content=[TextContent(text="ok")],
                          is_error=False, timestamp=1, duration_ms=1500)
    )
    exported = manager.export_ts(str(tmp_path / "out.jsonl"))
    records = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    message = next(r["message"] for r in records if r["type"] == "message")
    assert message["durationMs"] == 1500
    restored = read_ts_file(exported)["records"][0]["message"]
    assert restored.duration_ms == 1500

    lines = [
        {"type": "session", "version": TS_VERSION, "id": "s1", "timestamp": 1700000000000, "cwd": "/tmp"},
        {"type": "message", "id": "e1", "parentId": None, "timestamp": 1,
         "message": {"role": "toolResult", "toolCallId": "t1", "toolName": "bash", "isError": False,
                     "timestamp": 1, "durationMs": 42,
                     "content": [{"type": "text", "text": "ok"}]}},
    ]
    path = tmp_path / "ts-duration.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    imported = read_ts_file(str(path))["records"][0]["message"]
    assert isinstance(imported, ToolResultMessage)
    assert imported.duration_ms == 42

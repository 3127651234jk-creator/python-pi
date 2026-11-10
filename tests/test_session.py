"""Session persistence tests: JSONL roundtrip."""

from __future__ import annotations

import os
import sys
import time
import asyncio
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.ai.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi.coding_agent.session import SessionManager
from pi.coding_agent.cli import async_main, handle_session_command
from pi.agent.agent import Agent
from pi.agent.compaction import prefix_checkpoint
from pi.ai.models import resolve_model
from pi.ai.providers.faux import reset_script_index


def test_session_roundtrip(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path))
    manager.start_new()

    manager.append(SystemMessage(content="prompt", timestamp=1))
    manager.append(UserMessage(content="hello", timestamp=2))
    manager.append(
        AssistantMessage(
            content=[TextContent(text="using a tool"), ToolCall(id="t1", name="read", arguments={"path": "x"})],
            api="faux",
            provider="faux",
            model="faux",
            usage=Usage(input=10, output=5),
            stop_reason="toolUse",
            timestamp=3,
        )
    )
    manager.append(
        ToolResultMessage(
            tool_call_id="t1",
            tool_name="read",
            content=[TextContent(text="contents")],
            is_error=False,
            timestamp=4,
            details={"k": 1},
        )
    )

    loaded = SessionManager(session_dir=str(tmp_path / "sessions"), cwd=str(tmp_path)).load(manager.session_file)
    assert len(loaded) == 4
    assert loaded[0].content == "prompt"
    assert loaded[1].content == "hello"
    assert loaded[2].content[1].name == "read"
    assert loaded[2].usage.input == 10
    assert loaded[3].is_error is False
    assert loaded[3].details == {"k": 1}


def test_latest_session_file(tmp_path):
    manager = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path))
    first = manager.start_new()
    time.sleep(0.02)
    second = SessionManager(session_dir=str(tmp_path), cwd=str(tmp_path)).start_new()
    latest = manager.latest_session_file()
    assert latest == second
    assert latest != first


def test_custom_message_can_start_session(tmp_path):
    manager = SessionManager(str(tmp_path))
    entry_id = manager.append_custom_message("notice", "first", display=False, details={"source": "extension"})
    assert manager.leaf_id == entry_id
    assert len(manager.entries) == 1
    reader = SessionManager(str(tmp_path))
    reader.load(manager.session_file)
    assert reader.entries == manager.entries


def test_fork_preserves_custom_message_metadata(tmp_path):
    manager = SessionManager(str(tmp_path))
    manager.start_new()
    manager.append_custom_message("notice", "hidden", display=False, details={"source": "extension"})
    original = manager.entries[0]
    manager.fork()
    assert manager.entries[0].custom == original.custom
    assert manager.messages() == [original.message]
    reader = SessionManager(str(tmp_path))
    reader.load(manager.session_file)
    assert reader.entries[0].custom == original.custom


def test_tree_branches_survive_reopen(tmp_path):
    manager = SessionManager(str(tmp_path))
    root = manager.append(UserMessage(content="root", timestamp=0))
    old = manager.append(UserMessage(content="old branch", timestamp=1))
    manager.branch(root)
    new = manager.append(UserMessage(content="new branch", timestamp=2))
    reader = SessionManager(str(tmp_path))
    messages = reader.load(manager.session_file)
    # Branching leaves a digest of the abandoned tail before continuing.
    assert [m.content for m in messages if m.role == "user"] == ["root", "new branch"]
    assert any(m.role == "system" and "old branch" in m.content for m in messages)
    assert len(reader.entries) == 5
    assert old in reader.tree() and new in reader.tree()
    rebranched = reader.branch(old)
    assert [m.content for m in rebranched if m.role == "user"] == ["root", "old branch"]
    assert SessionManager(str(tmp_path)).load(manager.session_file)[-1].role == "system"
    assert "new branch" in SessionManager(str(tmp_path)).load(manager.session_file)[-1].content


def test_fork_export_and_session_switch(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"))
    manager.append(UserMessage(content="<script>alert(1)</script>", timestamp=0))
    original = manager.session_file
    manager.rename("original")
    manager.fork()
    manager.append(UserMessage(content="fork only", timestamp=1))
    fork = manager.session_file
    exported = manager.export(str(tmp_path / "export.jsonl"))
    reader = SessionManager(str(tmp_path / "sessions"))
    assert len(reader.load(exported)) == 2
    manager.export(str(tmp_path / "export.html"))
    assert "<script>" not in (tmp_path / "export.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        manager.export(str(tmp_path / "export.html"))
    with pytest.raises(ValueError):
        manager.export(fork)
    assert len(manager.resume(os.path.basename(original)[:-6])) == 1
    assert manager.name == "original"
    assert len(manager.list_sessions()) == 2
    manager.resume(fork)
    assert len(manager.messages()) == 2


def test_invalid_load_does_not_change_active_session(tmp_path):
    manager = SessionManager(str(tmp_path))
    manager.append(UserMessage(content="safe", timestamp=0))
    original = manager.session_file
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps({"type": "leaf", "id": "missing"}), encoding="utf-8")
    with pytest.raises(ValueError):
        manager.load(str(broken))
    assert manager.session_file == original
    assert manager.messages()[0].content == "safe"


def test_cli_session_commands(tmp_path):
    manager = SessionManager(str(tmp_path))
    root = manager.append(UserMessage(content="root", timestamp=0))
    manager.append(UserMessage(content="later", timestamp=1))
    agent = Agent(resolve_model("faux"))
    handle_session_command("/tree", root, agent, manager)
    assert [m.content for m in agent.messages if m.role == "user"] == ["root"]
    assert any(m.role == "system" and "later" in m.content for m in agent.messages)
    handle_session_command("/clone", "", agent, manager)
    assert [m.content for m in agent.messages if m.role == "user"] == ["root"]
    handle_session_command("/name", "cloned", agent, manager)
    assert manager.name == "cloned"
    with pytest.raises(ValueError):
        handle_session_command("/resume", "missing", agent, manager)
    assert [m.content for m in agent.messages if m.role == "user"] == ["root"]


def test_cli_persists_system_prompt_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_script_index()
    assert asyncio.run(async_main(["--provider", "faux", "hello"])) == 0
    manager = SessionManager(cwd=str(tmp_path))
    path = manager.latest_session_file()
    messages = manager.load(path)
    assert messages[0].role == "system"
    before = len(messages)
    reset_script_index()
    assert asyncio.run(async_main(["--provider", "faux", "--session", path, "again"])) == 0
    assert len(manager.load(path)) > before


def test_resume_restores_compaction_checkpoint(tmp_path):
    manager = SessionManager(str(tmp_path))
    manager.start_new()
    first = UserMessage(content="old" * 4000, timestamp=0)
    second = UserMessage(content="latest", timestamp=1)
    manager.append(first)
    manager.append(second)
    prefix = [first]
    manager.append_compaction("summary so far", checkpoint=prefix_checkpoint(prefix))
    session_id = os.path.basename(manager.session_file)[:-6]

    agent = Agent(resolve_model("faux"))
    agent.compactor.summary = "stale"
    agent.compactor.prefix = []
    handle_session_command("/resume", session_id, agent, manager)
    assert agent.compactor.summary == "summary so far"
    assert agent.compactor.prefix == prefix


def test_branch_requires_complete_tool_results(tmp_path):
    manager = SessionManager(str(tmp_path))
    manager.append(UserMessage(content="read file", timestamp=0))
    call = manager.append(AssistantMessage(
        content=[ToolCall(id="call", name="read", arguments={"path": "a"})],
        api="faux", provider="faux", model="faux", usage=Usage(), stop_reason="toolUse", timestamp=1,
    ))
    result = manager.append(ToolResultMessage(
        tool_call_id="call", tool_name="read", content=[TextContent(text="done")], is_error=False, timestamp=2,
    ))
    with pytest.raises(ValueError):
        manager.branch(call)
    assert manager.leaf_id == result
    with pytest.raises(ValueError):
        manager.fork(call)
    assert len(manager.list_sessions()) == 1
    assert manager.branch(result)[-1].role == "toolResult"

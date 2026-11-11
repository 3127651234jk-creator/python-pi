"""Session branch and export regressions found during runtime review."""

import pytest

from pi.ai.types import AssistantMessage, TextContent, ToolCall, Usage, UserMessage
from pi.coding_agent.session import SessionManager


@pytest.mark.parametrize("operation", ["branch", "fork"])
def test_hosted_calls_do_not_require_local_results(tmp_path, operation):
    manager = SessionManager(str(tmp_path / "sessions"))
    manager.append(UserMessage(content="search", timestamp=0))
    completed = manager.append(AssistantMessage(
        content=[ToolCall(id="search", name="web_search", arguments={},
                          raw_item={"type": "web_search_call", "id": "search", "status": "completed"}),
                 TextContent(text="answer")],
        api="openai-responses", provider="openai", model="test", usage=Usage(),
        stop_reason="stop", timestamp=1,
    ))
    reader = SessionManager(str(tmp_path / "sessions"))
    reader.load(manager.session_file)
    messages = getattr(reader, operation)(completed)
    assert messages[-1].content[-1].text == "answer"


def test_branch_export_preserves_entry_timestamps(tmp_path):
    manager = SessionManager(str(tmp_path / "sessions"))
    manager.append(UserMessage(content="hello", timestamp=0))
    manager.append_custom("state", {"value": 1})
    manager.append_custom_message("notice", "message", display=False)
    exported = manager.export(str(tmp_path / "export.jsonl"))
    reader = SessionManager(str(tmp_path / "sessions"))
    reader.load(exported)
    assert reader.entries == manager.entries

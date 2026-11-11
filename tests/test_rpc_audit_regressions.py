"""RPC regressions for session entries and assistant text retrieval."""

import asyncio
import io
import json

import pytest

from pi.agent.agent import Agent
from pi.ai.models import resolve_model
from pi.ai.types import AssistantMessage, TextContent, ThinkingContent, Usage
from pi.coding_agent.rpc import JsonWriter, RpcServer
from pi.coding_agent.session import SessionManager


def assistant(content, stop_reason="stop"):
    return AssistantMessage(content=content, api="faux", provider="faux", model="faux",
                            usage=Usage(), stop_reason=stop_reason, timestamp=0)


@pytest.mark.parametrize("tail,expected", [
    ([assistant([TextContent(text=" first"), TextContent(text=" second ")])], "first second"),
    ([assistant([ThinkingContent(thinking="thinking")])], None),
    ([assistant([])], None),
    ([assistant([], "aborted")], "old"),
    ([assistant([TextContent(text="   ")])], None),
])
def test_last_assistant_text_uses_complete_latest_message(tail, expected):
    async def scenario():
        agent = Agent(resolve_model("faux"))
        agent.messages = [assistant([TextContent(text="old")]), *tail]
        output = io.StringIO()
        server = RpcServer(agent, JsonWriter(output))
        await server.dispatch({"type": "get_last_assistant_text"})
        assert json.loads(output.getvalue())["data"]["text"] == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("command", ["get_entries", "get_tree"])
def test_rpc_accepts_custom_messages_with_string_content(tmp_path, command):
    async def scenario():
        session = SessionManager(str(tmp_path / "sessions"))
        session.append_custom_message("notice", "plain text", display=False)
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), session=session)
        await server.dispatch({"type": command})
        response = json.loads(output.getvalue())
        assert response["success"]
        entry = response["data"]["entries"][0] if command == "get_entries" else response["data"]["tree"][0]["entry"]
        assert entry["content"] == "plain text"
        assert entry["display"] is False

    asyncio.run(scenario())

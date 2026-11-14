import re

import pytest

from pi.ai.json_io import to_json
from pi.ai.models import Model
from pi.ai.providers import amazon_bedrock, anthropic
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, TranscriptContext, Usage


@pytest.mark.parametrize("module,api", [
    (anthropic, "anthropic-messages"), (amazon_bedrock, "bedrock-converse-stream"),
])
@pytest.mark.parametrize("call_id,expected", [
    ("call_abc|fc_" + "x" * 120, "call_abc"),
    ("a" * 132, "a" * 64),
    ("call.bad:/中文", "call_bad____"),
    ("valid-ID_123", "valid-ID_123"),
    ("", "call"),
])
@pytest.mark.parametrize("has_result", [False, True])
def test_call_and_result_ids_match_without_mutating_history(module, api, call_id, expected, has_result):
    history = [AssistantMessage(
        content=[ToolCall(id=call_id, name="echo", arguments={})],
        api="openai-responses", provider="gateway", model="other", usage=Usage(),
        stop_reason="toolUse", timestamp=0,
    )]
    if has_result:
        history.append(ToolResultMessage(tool_call_id=call_id, tool_name="echo",
                       content=[TextContent(text="ok")], is_error=False, timestamp=0))
    original = to_json(history)
    body = module.build_request(Model("target", "Target", api, "target", "https://example.test"),
                                TranscriptContext(history), SimpleStreamOptions())
    if module is anthropic:
        actual = body["messages"][0]["content"][0]["id"]
        result_id = body["messages"][1]["content"][0]["tool_use_id"]
    else:
        actual = body["messages"][0]["content"][0]["toolUse"]["toolUseId"]
        result_id = body["messages"][1]["content"][0]["toolResult"]["toolUseId"]
    assert actual == result_id == expected
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", actual)
    assert to_json(history) == original

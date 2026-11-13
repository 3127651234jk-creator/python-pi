"""Request text must preserve the TypeScript transcript rendering semantics."""

import pytest

from pi.ai.models import Model
from pi.ai.providers import anthropic, openai_completions
from pi.ai.types import (AssistantMessage, SystemMessage, TextContent, ThinkingContent,
                         ToolCall, TranscriptContext, Usage, get_system_message_text)


@pytest.mark.parametrize("content", ["base", [TextContent(text="base")]])
def test_system_sections_are_plain_text_in_order(content):
    message = SystemMessage(content=content, timestamp=0,
                            sections={"skills": "skills text", "removed": None,
                                      "empty": "", "space": " ", "custom": "<existing>text</existing>"})
    expected = "base\n\nskills text\n\n \n\n<existing>text</existing>"
    assert get_system_message_text(message) == expected
    model = Model("test", "test", "anthropic-messages", "test", "https://example.test")
    context = TranscriptContext([message])
    assert anthropic.build_request(model, context, None)["system"] == expected
    assert openai_completions.build_request(model, context, None)["messages"][0]["content"] == expected


@pytest.mark.parametrize("signature", [None, "", " \t"])
def test_unsigned_thinking_keeps_text_before_tool_use(signature):
    thinking = ThinkingContent(thinking=" keep reasoning ", thinking_signature=signature)
    message = AssistantMessage(content=[thinking, ToolCall(id="call", name="echo", arguments={})],
                               api="anthropic-messages", provider="test", model="test",
                               usage=Usage(), stop_reason="toolUse", timestamp=0)
    assert anthropic.convert_messages([message])[0]["content"] == [
        {"type": "text", "text": " keep reasoning "},
        {"type": "tool_use", "id": "call", "name": "echo", "input": {}},
    ]
    assert message.content[0] is thinking


def test_signed_and_redacted_thinking_keep_original_payloads():
    message = AssistantMessage(content=[
        ThinkingContent(thinking="", thinking_signature="signed"),
        ThinkingContent(thinking="hidden", thinking_signature="opaque", redacted=True),
        ThinkingContent(thinking=" \n"), TextContent(text="answer"),
    ], api="anthropic-messages", provider="test", model="test", usage=Usage(), stop_reason="stop", timestamp=0)
    assert anthropic.convert_messages([message])[0]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": "answer"},
    ]

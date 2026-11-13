"""Empty text snapshots must remain safe to replay after stream interruption."""

import pytest

from pi.ai.models import resolve_model
from pi.ai.providers.anthropic import build_request
from pi.ai.types import AssistantMessage, ImageContent, TextContent, TranscriptContext, Usage, UserMessage


@pytest.mark.parametrize("empty", ["", " \n\t"])
@pytest.mark.parametrize("only_empty", [False, True])
def test_aborted_assistant_empty_text_replay(empty, only_empty):
    blocks = [TextContent(text=empty)]
    if not only_empty:
        blocks.append(TextContent(text=" preserved text "))
    message = AssistantMessage(content=blocks, api="anthropic-messages", provider="anthropic",
                               model="test", usage=Usage(), stop_reason="aborted", timestamp=0)
    body = build_request(resolve_model("anthropic", api_key="test-key"), TranscriptContext([
        UserMessage(content="hello", timestamp=0), message,
        UserMessage(content="continue", timestamp=1),
    ]), None)
    replay = body["messages"][1]["content"]
    assert all(block["text"].strip() for block in replay if block["type"] == "text")
    if not only_empty:
        assert replay == [{"type": "text", "text": " preserved text "}]
    assert message.content[0].text == empty


@pytest.mark.parametrize("empty", ["", " \n\t"])
def test_user_empty_text_filtered_while_images_survive(empty):
    image = ImageContent(data="image", mime_type="image/png")
    body = build_request(resolve_model("anthropic", api_key="test-key"), TranscriptContext([
        UserMessage(content=[TextContent(text=empty)], timestamp=0),
        UserMessage(content=[TextContent(text=empty), image, TextContent(text="look")], timestamp=1),
    ]), None)
    assert len(body["messages"]) == 1
    assert [block["type"] for block in body["messages"][0]["content"]] == ["image", "text"]

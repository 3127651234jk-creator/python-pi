import json

import pytest

from pi.ai.json_io import message_from_json, to_json
from pi.ai.models import Model
from pi.ai.providers import amazon_bedrock, anthropic, google, openai_completions, openai_responses, pi_messages
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import AssistantMessage, SystemMessage, TextContent, ToolCall, ToolResultMessage, TranscriptContext, Usage, UserMessage
from pi.ai.utils import repair_tool_results


def interrupted_history():
    # JSON round-trip represents the messages persisted before a hard kill.
    message = AssistantMessage(
        content=[ToolCall(id="done", name="echo", arguments={}),
                 ToolCall(id="missing", name="echo", arguments={})],
        api="openai-responses", provider="test", model="test", usage=Usage(),
        stop_reason="toolUse", timestamp=0,
    )
    return [message_from_json(to_json(message)), ToolResultMessage(
        tool_call_id="done", tool_name="echo", content=[TextContent(text="real result")], is_error=False, timestamp=0,
    )]


@pytest.mark.parametrize("boundary", [None, "user", "assistant"])
def test_repair_partial_batch_at_end_or_next_turn(boundary):
    history = interrupted_history()
    if boundary == "user":
        history.append(UserMessage(content="continue", timestamp=0))
    elif boundary == "assistant":
        history.append(AssistantMessage(content=[TextContent(text="next")], api="test", provider="test",
                                        model="test", usage=Usage(), stop_reason="stop", timestamp=0))
    original = to_json(history)
    repaired = repair_tool_results(history)
    results = [m for m in repaired if isinstance(m, ToolResultMessage)]
    assert [m.tool_call_id for m in results] == ["done", "missing"]
    assert results[0].content[0].text == "real result"
    assert results[1].content[0].text == "No result provided"
    assert results[1].is_error
    assert repaired[2] is results[1]
    assert to_json(history) == original
    assert to_json(repair_tool_results(repaired)) == to_json(repaired)


def test_system_between_call_and_real_result_does_not_duplicate():
    history = interrupted_history()
    system = SystemMessage(content="update", timestamp=0)
    history.insert(1, system)
    history.append(ToolResultMessage(tool_call_id="missing", tool_name="echo",
                                     content=[TextContent(text="finished")], is_error=False, timestamp=0))
    repaired = repair_tool_results(history)
    assert len(repaired) == len(history)
    assert repaired[-1] is system
    assert "No result provided" not in json.dumps(to_json(repaired))


@pytest.mark.parametrize("module,api", [
    (anthropic, "anthropic-messages"), (amazon_bedrock, "bedrock-converse-stream"),
    (google, "google-generative-ai"), (openai_completions, "openai-completions"),
    (openai_responses, "openai-responses"), (pi_messages, "pi-messages"),
])
def test_provider_request_repairs_reopened_history(module, api):
    model = Model("test", "Test", api, "test", "https://example.test")
    history = interrupted_history()
    history.append(UserMessage(content="continue", timestamp=0))
    body = module.build_request(model, TranscriptContext(history), SimpleStreamOptions())
    assert "No result provided" in json.dumps(body)


def test_hosted_calls_do_not_need_client_results():
    history = interrupted_history()[:1]
    history[0].content = [ToolCall(id="hosted", name="search", arguments={}, raw_item={"type": "web_search_call"})]
    assert repair_tool_results(history) == history

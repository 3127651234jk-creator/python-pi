"""Regression tests for provider conversion gaps found in the static audit.

Covers Gemini thought-signature replay, compound Responses tool-call IDs,
tool-result images on Gemini/Bedrock/Chat Completions, TS text signatures, and
the RPC state/model wire fields.
"""

import asyncio
import base64
import io
import json

import httpx

from pi.agent.agent import Agent
from pi.ai.models import Model, ModelCost, resolve_model
from pi.ai.providers import amazon_bedrock, anthropic, google, openai_completions, openai_responses
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import (
    AssistantMessage,
    DoneEvent,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
)
from pi.coding_agent import rpc_wire, ts_interop
from pi.coding_agent.rpc import JsonWriter, RpcServer


def base_context():
    return TranscriptContext([
        SystemMessage(content="system", timestamp=0, tools_added=[Tool("echo", "Echo", {"type": "object"})]),
        UserMessage(content="hello", timestamp=0),
    ])


def assistant(content, model, api="google-generative-ai", provider=None):
    return AssistantMessage(content=content, api=api, provider=provider or model.provider,
                            model=model.id, usage=Usage(), stop_reason="stop", timestamp=0)


async def _stream_payload(module, model, payload):
    async def send(self, request, **kwargs):
        return httpx.Response(200, request=request,
                              content="".join("data: " + json.dumps(item) + "\n\n" for item in payload).encode())
    original = httpx.AsyncClient.send

    async def run():
        httpx.AsyncClient.send = send
        try:
            stream = module.stream_google if module is google else module.stream_openai_responses
            return [event async for event in stream(model, base_context())]
        finally:
            httpx.AsyncClient.send = original

    return await run()


def _google_payload(parts, finish="STOP"):
    return [{"candidates": [{"content": {"parts": parts}, "finishReason": finish}]}]


# ---------------------------------------------------------------------------
# Gemini signatures and content order
# ---------------------------------------------------------------------------


def test_google_text_thought_signature_parsed_and_replayed():
    model = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test",
                  "https://example.test", api_key="key", input=["text", "image"])
    events = asyncio.run(_stream_payload(google, model, _google_payload([
        {"text": "answer", "thoughtSignature": "dGV4dA=="},
    ])))
    done = next(event for event in events if isinstance(event, DoneEvent))
    text = done.message.content[0]
    assert isinstance(text, TextContent)
    assert text.text == "answer" and text.text_signature == "dGV4dA=="

    body = google.build_request(model, TranscriptContext([done.message]), SimpleStreamOptions())
    assert body["contents"][-1]["parts"] == [{"text": "answer", "thoughtSignature": "dGV4dA=="}]


def test_google_signature_only_empty_text_is_replayed():
    model = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test",
                  "https://example.test", input=["text"])
    message = assistant([TextContent(text="", text_signature="c2lnbmF0dXJl")], model)
    body = google.build_request(model, TranscriptContext([message]), SimpleStreamOptions())
    assert body["contents"][-1]["parts"] == [{"text": "", "thoughtSignature": "c2lnbmF0dXJl"}]

    plain = assistant([TextContent(text="   ")], model)
    body = google.build_request(model, TranscriptContext([plain]), SimpleStreamOptions())
    assert body["contents"] == []


def test_google_duplicate_function_call_ids_are_regenerated():
    model = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test",
                  "https://example.test", api_key="key")
    events = asyncio.run(_stream_payload(google, model, _google_payload([
        {"functionCall": {"id": "same", "name": "echo", "args": {"n": 1}}},
        {"functionCall": {"id": "same", "name": "echo", "args": {"n": 2}}},
    ])))
    done = next(event for event in events if isinstance(event, DoneEvent))
    calls = [block for block in done.message.content if isinstance(block, ToolCall)]
    assert len(calls) == 2
    assert calls[0].id == "same"
    assert calls[1].id != "same" and calls[1].id.startswith("echo_")


def test_google_text_around_tool_call_keeps_wire_order():
    model = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test",
                  "https://example.test", api_key="key")
    events = asyncio.run(_stream_payload(google, model, _google_payload([
        {"text": "before"},
        {"functionCall": {"name": "echo", "args": {}}},
        {"text": "after"},
    ])))
    done = next(event for event in events if isinstance(event, DoneEvent))
    assert [type(block).__name__ for block in done.message.content] == ["TextContent", "ToolCall", "TextContent"]
    assert done.message.content[0].text == "before"
    assert done.message.content[2].text == "after"


def test_google_tool_call_ids_only_for_models_that_require_them():
    gemini3 = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test", "https://example.test")
    gemini2 = Model("gemini-2.0-flash", "t", "google-generative-ai", "test", "https://example.test")
    call = ToolCall(id="call", name="echo", arguments={})
    for model, expected in ((gemini3, True), (gemini2, False)):
        body = google.build_request(model, TranscriptContext([assistant([call], model, provider=model.provider)]),
                                    SimpleStreamOptions())
        function_call = body["contents"][0]["parts"][0]["functionCall"]
        assert ("id" in function_call) is expected


# ---------------------------------------------------------------------------
# Tool-result images
# ---------------------------------------------------------------------------


def _tool_result():
    return ToolResultMessage(
        tool_call_id="call", tool_name="read",
        content=[TextContent(text="text result"),
                 ImageContent(data=base64.b64encode(b"image-bytes").decode(), mime_type="image/png")],
        is_error=False, timestamp=0,
    )


def test_google_tool_result_images():
    model = Model("gemini-3-pro-preview", "t", "google-generative-ai", "test",
                  "https://example.test", input=["text", "image"])
    body = google.build_request(model, TranscriptContext([_tool_result()]), SimpleStreamOptions())
    response = body["contents"][0]["parts"][0]["functionResponse"]
    assert response["id"] == "call"
    assert response["response"] == {"output": "text result"}
    assert response["parts"] == [{"inlineData": {
        "mimeType": "image/png", "data": base64.b64encode(b"image-bytes").decode()}}]

    legacy = Model("gemini-2.0-flash", "t", "google-generative-ai", "test",
                   "https://example.test", input=["text", "image"])
    body = google.build_request(legacy, TranscriptContext([_tool_result()]), SimpleStreamOptions())
    response = body["contents"][0]["parts"][0]["functionResponse"]
    assert "id" not in response and "parts" not in response
    # Older Gemini needs the images in a separate user turn.
    assert body["contents"][1] == {"role": "user", "parts": [
        {"text": "Tool result image:"},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(b"image-bytes").decode()}},
    ]}


def test_bedrock_tool_result_images():
    model = Model("anthropic.claude-sonnet-4-5", "t", "bedrock-converse-stream", "amazon-bedrock",
                  "https://bedrock.us-east-1.amazonaws.com", input=["text", "image"])
    body = amazon_bedrock.build_request(model, TranscriptContext([_tool_result()]), SimpleStreamOptions())
    content = body["messages"][0]["content"][0]["toolResult"]["content"]
    assert content[0] == {"text": "text result"}
    assert content[1] == {"image": {"format": "png", "source": {"bytes": b"image-bytes"}}}


def test_completions_tool_result_images_are_attached_after_tool_messages():
    model = Model("test", "t", "openai-completions", "test", "https://example.test",
                  input=["text", "image"])
    body = openai_completions.build_request(model, TranscriptContext([_tool_result()]), SimpleStreamOptions())
    assert body["messages"][0] == {"role": "tool", "tool_call_id": "call", "content": "text result"}
    attachment = body["messages"][1]
    assert attachment["role"] == "user"
    assert attachment["content"][0] == {"type": "text", "text": "Attached image(s) from tool result:"}
    assert attachment["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    text_only = Model("test", "t", "openai-completions", "test", "https://example.test", input=["text"])
    body = openai_completions.build_request(text_only, TranscriptContext([_tool_result()]), SimpleStreamOptions())
    assert body["messages"] == [{"role": "tool", "tool_call_id": "call", "content": "text result"}]


# ---------------------------------------------------------------------------
# Compound Responses tool-call IDs
# ---------------------------------------------------------------------------


def test_responses_stream_keeps_call_and_item_ids():
    model = Model("test", "t", "openai-responses", "test", "https://example.test", api_key="key")
    payload = [
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "echo", "arguments": ""}},
        {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{}"},
        {"type": "response.completed", "response": {"id": "response", "usage": {"input_tokens": 1, "output_tokens": 1}}},
    ]
    events = asyncio.run(_stream_payload(openai_responses, model, payload))
    done = next(event for event in events if isinstance(event, DoneEvent))
    call = next(block for block in done.message.content if isinstance(block, ToolCall))
    assert call.id == "call_1|fc_1"

    result = ToolResultMessage(tool_call_id=call.id, tool_name="echo",
                               content=[TextContent(text="ok")], is_error=False, timestamp=0)
    body = openai_responses.build_request(model, TranscriptContext([done.message, result]), SimpleStreamOptions())
    function_call = next(item for item in body["input"] if item.get("type") == "function_call")
    assert function_call["call_id"] == "call_1" and function_call["id"] == "fc_1"
    output = next(item for item in body["input"] if item.get("type") == "function_call_output")
    assert output["call_id"] == "call_1"


def test_responses_replay_omits_item_id_for_other_models():
    model = Model("target", "t", "openai-responses", "test", "https://example.test")
    message = AssistantMessage(content=[ToolCall(id="call_1|fc_1", name="echo", arguments={})],
                               api="openai-responses", provider="test", model="other",
                               usage=Usage(), stop_reason="toolUse", timestamp=0)
    body = openai_responses.build_request(model, TranscriptContext([message]), SimpleStreamOptions())
    function_call = next(item for item in body["input"] if item.get("type") == "function_call")
    assert function_call["call_id"] == "call_1" and "id" not in function_call


def test_compound_ids_stay_distinct_for_anthropic_and_bedrock():
    message = AssistantMessage(content=[
        ToolCall(id="call_1|fc_2", name="echo", arguments={}),
        ToolCall(id="call_1|fc_3", name="echo", arguments={}),
    ], api="openai-responses", provider="gateway", model="other",
       usage=Usage(), stop_reason="toolUse", timestamp=0)
    for module, api in ((anthropic, "anthropic-messages"),
                        (amazon_bedrock, "bedrock-converse-stream")):
        model = Model("target", "t", api, "test", "https://example.test")
        content = module.build_request(model, TranscriptContext([message]), SimpleStreamOptions())["messages"][0]["content"]
        ids = [block["id"] if module is anthropic else block["toolUse"]["toolUseId"] for block in content]
        assert ids == ["call_1_fc_2", "call_1_fc_3"]


def test_completions_normalizes_compound_ids_consistently():
    model = Model("test", "t", "openai-completions", "openai", "https://api.openai.com")
    message = AssistantMessage(content=[
        ToolCall(id="call_1|fc_2", name="echo", arguments={}),
        ToolCall(id="call_1|fc_3", name="echo", arguments={}),
    ], api="openai-completions", provider="openai", model="test",
       usage=Usage(), stop_reason="toolUse", timestamp=0)
    result = ToolResultMessage(tool_call_id="call_1|fc_2", tool_name="echo",
                               content=[TextContent(text="ok")], is_error=False, timestamp=0)
    body = openai_completions.build_request(model, TranscriptContext([message, result]), SimpleStreamOptions())
    ids = [call["id"] for call in body["messages"][0]["tool_calls"]]
    assert ids == ["call_1_fc_2", "call_1_fc_3"]
    assert body["messages"][1]["tool_call_id"] == "call_1_fc_2"

    long_id = "call_" + "a" * 80 + "|fc_" + "b" * 80
    short = openai_completions.normalize_completions_tool_call_id(model, long_id)
    assert len(short) <= 40
    assert short == openai_completions.normalize_completions_tool_call_id(model, long_id)

    assert openai_completions.normalize_completions_tool_call_id(
        model, "c" * 60) == "c" * 40


# ---------------------------------------------------------------------------
# TS interop and RPC wire
# ---------------------------------------------------------------------------


def test_ts_interop_preserves_text_signature():
    message = AssistantMessage(content=[TextContent(text="answer", text_signature="sig")],
                               api="google-generative-ai", provider="test", model="gemini-3-pro-preview",
                               usage=Usage(), stop_reason="stop", timestamp=0)
    data = ts_interop.our_message_to_ts(message)
    assert data["content"][0]["textSignature"] == "sig"
    restored = ts_interop.ts_message_to_ours(data)
    assert restored.content[0].text_signature == "sig"


def test_rpc_wire_includes_cost_tiers():
    model = Model("tiered", "t", "faux", "test", "",
                  costs=ModelCost(input=1.0, output=2.0, cache_read=0.1, cache_write=0.2,
                                  tiers=[{"input_tokens_above": 100_000, "input": 4.0, "output": 6.0,
                                          "cache_read": 0.4, "cache_write": 0.8}]))
    assert rpc_wire.model_to_ts(model)["cost"]["tiers"] == [{
        "inputTokensAbove": 100_000, "input": 4.0, "output": 6.0,
        "cacheRead": 0.4, "cacheWrite": 0.8,
    }]


def test_rpc_get_state_reports_compaction():
    async def check():
        output = io.StringIO()
        server = RpcServer(Agent(resolve_model("faux")), JsonWriter(output), registry=None)
        server.agent.compactor.compacting = True
        await server.dispatch({"id": "s", "type": "get_state"})
        return [json.loads(line) for line in output.getvalue().splitlines()]

    records = asyncio.run(check())
    assert records[0]["data"]["isCompacting"] is True


# ---------------------------------------------------------------------------
# Unicode surrogates
# ---------------------------------------------------------------------------


def test_anthropic_sanitizes_lone_surrogates_before_json_encoding():
    lone = "\ud83d"
    message = AssistantMessage(
        content=[TextContent(text=f"answer {lone}"), ThinkingContent(thinking=f"think {lone}", thinking_signature="sig")],
        api="anthropic-messages", provider="anthropic", model="test",
        usage=Usage(), stop_reason="toolUse", timestamp=0)
    result = ToolResultMessage(tool_call_id="call", tool_name="echo",
                               content=[TextContent(text=f"result {lone}")], is_error=False, timestamp=0)
    context = TranscriptContext([
        SystemMessage(content=f"system {lone}", timestamp=0),
        UserMessage(content=f"hello 😀 {lone}", timestamp=0),
        UserMessage(content=[TextContent(text=f"block {lone}")], timestamp=1),
        message, result,
    ])
    body = anthropic.build_request(Model("test", "t", "anthropic-messages", "test", "https://example.test"),
                                   context, SimpleStreamOptions())
    # httpx encodes JSON with ensure_ascii=False; a lone surrogate would raise here.
    json.dumps(body, ensure_ascii=False).encode("utf-8")
    assert body["system"].startswith("system ")
    assert body["messages"][0]["content"].startswith("hello 😀 ")
    assert body["messages"][1]["content"][0]["text"].startswith("block ")
    assistant_blocks = body["messages"][2]["content"]
    assert assistant_blocks[0]["text"].startswith("answer ")
    assert assistant_blocks[1]["thinking"].startswith("think ")
    assert body["messages"][3]["content"][0]["content"][0]["text"].startswith("result ")

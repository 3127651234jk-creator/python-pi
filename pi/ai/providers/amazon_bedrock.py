"""Bedrock ConverseStream through optional boto3 and its AWS credential chain."""

import asyncio
import base64
import json
import os
import re
import time

try:
    import boto3
except ImportError:
    boto3 = None

from ..types import TextContent, ImageContent, AudioContent, VideoContent, ThinkingContent, ToolCall, ToolResultMessage, StreamStartEvent, StreamErrorEvent, get_system_message_text
from ..utils import normalize_context
from ...util import AbortError
from .base import ProviderState, SimpleStreamOptions, usage_from_counts, thinking_budget
from .blocks import BlockEvents

# Converse top-level keys the adapter always builds itself.
_STRUCTURAL_KEYS = ("modelId", "messages", "system")


def _is_nova(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.startswith("amazon.") and "nova" in lowered


def _apply_model_params(body: dict, model, options) -> None:
    """Merge ``Model.params`` into the Converse request.

    Top-level keys pass through verbatim (``inferenceConfig``,
    ``additionalModelRequestFields``, ``guardrailConfig``, ...) and merge one
    level deep with built keys so params extend rather than replace. The
    ``top_k`` shorthand routes to the family-specific field: Nova nests it as
    ``additionalModelRequestFields.inferenceConfig.topK``, everything else takes
    the flat ``additionalModelRequestFields.top_k``. Options the caller set
    explicitly (temperature, max tokens, reasoning) win over params.
    """
    params = model.params or {}
    for key, value in params.items():
        if key in _STRUCTURAL_KEYS or key == "top_k":
            continue
        current = body.get(key)
        body[key] = {**current, **value} if isinstance(current, dict) and isinstance(value, dict) else value
    config = body.get("inferenceConfig")
    if isinstance(config, dict) and options is not None:
        # Explicit caller options keep precedence over params.
        if options.temperature is not None:
            config["temperature"] = options.temperature
        if options.max_tokens:
            config["maxTokens"] = options.max_tokens
    top_k = params.get("top_k")
    if top_k is not None:
        fields = body.setdefault("additionalModelRequestFields", {})
        if isinstance(fields, dict):
            if _is_nova(model.id):
                fields.setdefault("inferenceConfig", {})["topK"] = top_k
            else:
                fields["top_k"] = top_k


def build_request(model, context, options):
    system, messages = normalize_context(context)
    cache_point = None
    if options is not None and options.cache_retention in ("5m", "1h") and _supports_prompt_caching(model.id):
        cache_point = {"cachePoint": {"type": "default", **({"ttl": "1h"} if options.cache_retention == "1h" else {})}}
    output = []
    for message in messages:
        content = []
        if isinstance(message, ToolResultMessage):
            content = [{"toolResult": {"toolUseId": message.tool_call_id, "status": "error" if message.is_error else "success",
                "content": [{"text": b.text} for b in message.content if isinstance(b, TextContent)] or [{"text": "(empty result)"}]}}]
        else:
            for block in [TextContent(text=message.content)] if isinstance(message.content, str) else message.content:
                if isinstance(block, TextContent):
                    content.append({"text": block.text})
                elif isinstance(block, ImageContent):
                    content.append({"image": {"format": block.mime_type.split("/")[-1].replace("jpg", "jpeg"), "source": {"bytes": base64.b64decode(block.data)}}})
                elif isinstance(block, (AudioContent, VideoContent)):
                    raise ValueError("Bedrock Converse does not support audio or video input")
                elif isinstance(block, ToolCall):
                    if block.raw_item is not None:
                        continue  # Hosted calls are provider-executed; not replayable on Bedrock.
                    content.append({"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.arguments}})
                elif isinstance(block, ThinkingContent) and block.thinking_signature:
                    content.append({"reasoningContent": {"reasoningText": {"text": block.thinking, "signature": block.thinking_signature}}})
        role = "assistant" if message.role == "assistant" else "user"
        if content:
            if output and output[-1]["role"] == role:
                output[-1]["content"].extend(content)
            else:
                output.append({"role": role, "content": content})
    body = {"modelId": model.id, "messages": output, "inferenceConfig": {"maxTokens": options.max_tokens or model.max_tokens}}
    if options.temperature is not None:
        body["inferenceConfig"]["temperature"] = options.temperature
    _apply_model_params(body, model, options)
    if model.reasoning and "anthropic." in model.id and options.reasoning not in (None, "off"):
        budget = min(thinking_budget(options.reasoning), body["inferenceConfig"]["maxTokens"] - 1024)
        if budget < 1024:
            raise ValueError("Bedrock thinking requires an output budget of at least 2048 tokens")
        body.setdefault("additionalModelRequestFields", {})["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["inferenceConfig"].pop("temperature", None)
    if system:
        body["system"] = [{"text": get_system_message_text(system)}]
        if cache_point:
            body["system"].append(dict(cache_point))
        tools = [{"toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": t.parameters}}}
                 for t in (system.tools_added or []) if not t.hosted]
        if tools:
            body["toolConfig"] = {"tools": tools}
    if cache_point and output and output[-1]["role"] == "user":
        output[-1]["content"].append(dict(cache_point))
    return body


def _supports_prompt_caching(model_id: str) -> bool:
    """Cache points are accepted by Claude models; others reject the block."""
    lowered = model_id.lower()
    return "claude" in lowered


async def stream_bedrock(model, context, options=None):
    options = options or SimpleStreamOptions()
    state = ProviderState(model, int(time.time() * 1000))
    blocks = BlockEvents(state)
    client = None
    stream = None
    try:
        if options.signal:
            options.signal.throw_if_aborted()
        if boto3 is None:
            raise RuntimeError("Bedrock requires optional dependencies: pip install 'pi-python[bedrock]'")
        match = re.search(r"bedrock-runtime\.([^.]+)\.", model.base_url)
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or (match.group(1) if match else "us-east-1")
        client = await asyncio.to_thread(boto3.client, "bedrock-runtime", region_name=region, endpoint_url=model.base_url)
        response = await asyncio.to_thread(client.converse_stream, **build_request(model, context, options))
        stream = response["stream"]
        iterator = iter(stream)
        yield StreamStartEvent(partial=state.snapshot())
        reason = None
        while True:
            if options.signal:
                options.signal.throw_if_aborted()
            item = await asyncio.to_thread(next, iterator, None)
            if item is None:
                break
            events = []
            if "contentBlockStart" in item:
                start = item["contentBlockStart"]
                call = start.get("start", {}).get("toolUse")
                if call:
                    events = blocks.tool(start["contentBlockIndex"], call["toolUseId"], call["name"])
            elif "contentBlockDelta" in item:
                chunk = item["contentBlockDelta"]
                key, delta = chunk["contentBlockIndex"], chunk["delta"]
                if "text" in delta:
                    events = blocks.text(key, delta["text"])
                elif "toolUse" in delta:
                    events = blocks.tool(key, "", "", delta["toolUse"].get("input", ""))
                elif "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    events = blocks.text(key, reasoning.get("text", ""), thinking=True)
                    if reasoning.get("signature"):
                        state.partial.content[blocks.indices[key]].thinking_signature = reasoning["signature"]
            elif "messageStop" in item:
                raw = item["messageStop"]["stopReason"]
                state.set_raw_stop_reason(raw)
                reason = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "toolUse", "max_tokens": "length"}.get(raw)
                if reason is None:
                    raise RuntimeError(f"Bedrock stopped: {raw}")
            elif "metadata" in item:
                usage = item["metadata"].get("usage", {})
                write = usage.get("cacheWriteInputTokens", 0) or 0
                write_1h = sum((detail.get("inputTokens") or 0) for detail in (usage.get("cacheDetails") or [])
                               if detail.get("ttl") == "1h")
                if write_1h:
                    write = max(0, write - write_1h)
                state.set_usage(usage_from_counts(usage.get("inputTokens", 0), usage.get("outputTokens", 0),
                    usage.get("cacheReadInputTokens", 0), write, cache_write_1h=write_1h))
            elif any(key.endswith("Exception") for key in item):
                raise RuntimeError("Bedrock stream error: " + next(iter(item)))
            for event in events:
                yield event
        if reason is None:
            raise RuntimeError("Bedrock stream ended without messageStop")
        state.partial.usage.cost = model.calculate_cost(state.partial.usage)
        for event in blocks.finish(reason):
            yield event
    except Exception as exc:
        state.partial.stop_reason = "aborted" if isinstance(exc, AbortError) else "error"
        state.partial.error_message = str(exc)
        yield StreamErrorEvent(reason=state.partial.stop_reason, error=state.snapshot())
    finally:
        if stream:
            await asyncio.to_thread(stream.close)
        if client:
            await asyncio.to_thread(client.close)

"""Bedrock ConverseStream through optional boto3 and its AWS credential chain."""

import asyncio
import base64
import json
import os
import re
import time
from typing import List, Optional

try:
    import boto3
except ImportError:
    boto3 = None

from ..types import TextContent, ImageContent, AudioContent, VideoContent, ThinkingContent, ToolCall, ToolResultMessage, StreamStartEvent, StreamErrorEvent, get_system_message_text
from ..utils import normalize_context, normalize_tool_call_id
from ...util import AbortError
from .base import ProviderState, SimpleStreamOptions, usage_from_counts, thinking_budget, resolve_max_tokens
from .blocks import BlockEvents

# Converse top-level keys the adapter always builds itself.
_STRUCTURAL_KEYS = ("modelId", "messages", "system")

# OpenAI GPT models (GPT-5.x, GPT-6) take a nested `reasoning.effort` and reject `minimal`.
_GPT_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}
# gpt-oss takes a flat `reasoning_effort` and only accepts low, medium and high.
_GPT_OSS_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def _is_nova(model_id: str) -> bool:
    lowered = model_id.lower()
    return lowered.startswith("amazon.") and "nova" in lowered


def _model_match_candidates(model) -> List[str]:
    """Id/name match candidates (port of TS ``getModelMatchCandidates``):
    lowercased values, plus a variant folding whitespace, underscores, dots and
    colons to dashes so catalog spellings like ``GPT_OSS`` still match."""
    values = [model.id, model.name] if model.name else [model.id]
    candidates: List[str] = []
    for value in values:
        lowered = value.lower()
        candidates.append(lowered)
        candidates.append(re.sub(r"[\s_.:]+", "-", lowered))
    return candidates


def _openai_reasoning_fields(model, reasoning: str) -> Optional[dict]:
    """Additional fields that carry the reasoning level to OpenAI models on
    Bedrock (port of the gpt branches in TS ``buildAdditionalModelRequestFields``,
    #9331/#10142): without them these models always run at the Bedrock default
    effort. ``model.thinking_level_map`` takes priority over the static maps; a
    level outside the known set is ignored (TS drops the undefined field) so
    unvalidated entry points such as env/settings/session replay cannot raise."""
    candidates = _model_match_candidates(model)
    if any("gpt-oss" in candidate for candidate in candidates):
        effort = _GPT_OSS_EFFORT.get(reasoning)
        return {"reasoning_effort": effort} if effort is not None else None
    if any("gpt-" in candidate for candidate in candidates):
        mapped = (model.thinking_level_map or {}).get(reasoning)
        effort = mapped if isinstance(mapped, str) else _GPT_EFFORT.get(reasoning)
        if effort is not None:
            return {"reasoning": {"effort": effort}}
    return None


def _apply_model_params(body: dict, model, options, max_tokens: Optional[int] = None) -> None:
    """Merge ``Model.params`` into the Converse request.

    Top-level keys pass through verbatim (``inferenceConfig``,
    ``additionalModelRequestFields``, ``guardrailConfig``, ...) and merge one
    level deep with built keys so params extend rather than replace. The
    ``top_k`` shorthand routes to the family-specific field: Nova nests it as
    ``additionalModelRequestFields.inferenceConfig.topK``, everything else takes
    the flat ``additionalModelRequestFields.top_k``. Options the caller set
    explicitly (temperature, max tokens, reasoning) win over params; the caller
    passes the already-clamped output budget so params cannot override it.
    """
    params = model.params or {}
    if "additionalModelRequestFields" in params and not isinstance(params["additionalModelRequestFields"], dict):
        # Both reasoning branches write into this object; reject an invalid
        # shape here instead of failing later with a bare TypeError.
        raise ValueError("Model params.additionalModelRequestFields must be an object")
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
        if max_tokens:
            config["maxTokens"] = max_tokens
    top_k = params.get("top_k")
    if top_k is not None:
        fields = body.setdefault("additionalModelRequestFields", {})
        if isinstance(fields, dict):
            if _is_nova(model.id):
                fields.setdefault("inferenceConfig", {})["topK"] = top_k
            else:
                fields["top_k"] = top_k


def _supports_adaptive_thinking(model_id: str) -> bool:
    """Bedrock Claude families that run adaptive thinking with a native
    xhigh/max effort (Claude 5+; haiku-5 included)."""
    lowered = model_id.lower()
    return any(marker in lowered for marker in ("haiku-5", "opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5"))


def _map_thinking_effort(model, level) -> str:
    """Port of mapThinkingLevelToEffort: native xhigh for the effort families,
    then the model's thinking_level_map, then the static fallback ladder."""
    if level == "xhigh" and _supports_adaptive_thinking(model.id):
        return "xhigh"
    mapped = (model.thinking_level_map or {}).get(level)
    if isinstance(mapped, str):
        return mapped
    return {"minimal": "low", "low": "low", "medium": "medium"}.get(level, "high")


def build_request(model, context, options):
    system, messages = normalize_context(context)
    cache_point = None
    if options is not None and options.cache_retention in ("5m", "1h") and _supports_prompt_caching(model.id):
        cache_point = {"cachePoint": {"type": "default", **({"ttl": "1h"} if options.cache_retention == "1h" else {})}}
    output = []
    for message in messages:
        same_model = (message.role == "assistant" and message.provider == model.provider
                      and message.api == model.api and message.model == model.id)
        content = []
        if isinstance(message, ToolResultMessage):
            content = [{"toolResult": {"toolUseId": normalize_tool_call_id(message.tool_call_id), "status": "error" if message.is_error else "success",
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
                    content.append({"toolUse": {"toolUseId": normalize_tool_call_id(block.id), "name": block.name, "input": block.arguments}})
                elif isinstance(block, ThinkingContent):
                    if same_model and block.thinking_signature:
                        content.append({"reasoningContent": {"reasoningText": {"text": block.thinking, "signature": block.thinking_signature}}})
                    elif not block.redacted and block.thinking.strip():
                        content.append({"text": block.thinking})
        role = "assistant" if message.role == "assistant" else "user"
        if content:
            if output and output[-1]["role"] == role:
                output[-1]["content"].extend(content)
            else:
                output.append({"role": role, "content": content})
    max_tokens = resolve_max_tokens(model, context, options)
    body = {"modelId": model.id, "messages": output, "inferenceConfig": {"maxTokens": max_tokens}}
    if options.temperature is not None:
        body["inferenceConfig"]["temperature"] = options.temperature
    _apply_model_params(body, model, options,
                        max_tokens=max_tokens if options and options.max_tokens else None)
    if model.reasoning and options.reasoning not in (None, "off"):
        if "anthropic." in model.id:
            if _supports_adaptive_thinking(model.id):
                # Claude Haiku 5.5+ effort families on Bedrock: adaptive
                # thinking with a native xhigh/max effort instead of a budget.
                fields = body.setdefault("additionalModelRequestFields", {})
                fields["thinking"] = {"type": "adaptive"}
                fields["output_config"] = {"effort": _map_thinking_effort(model, options.reasoning)}
            else:
                budget = min(thinking_budget(options.reasoning), body["inferenceConfig"]["maxTokens"] - 1024)
                if budget < 1024:
                    raise ValueError("Bedrock thinking requires an output budget of at least 2048 tokens")
                body.setdefault("additionalModelRequestFields", {})["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["inferenceConfig"].pop("temperature", None)
        else:
            openai_fields = _openai_reasoning_fields(model, options.reasoning)
            if openai_fields:
                fields = body.setdefault("additionalModelRequestFields", {})
                for key, value in openai_fields.items():
                    # Explicit params keep precedence over the derived value.
                    if key not in fields:
                        fields[key] = value
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

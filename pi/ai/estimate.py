"""Context token estimation for request sizing.

Python port of ``packages/ai/src/utils/estimate.ts``. Character-based
heuristic: 3.5 characters per token (#10497), replacing the old bytes/4
division. Consumers (provider output-limit clamp, compaction, durable
compaction tasks) share this one estimator instead of each carrying its own
divisor.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, List, Optional, Union

from .types import (
    AssistantMessage,
    AudioContent,
    ImageContent,
    SystemMessage,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UserMessage,
    VideoContent,
    get_system_message_text,
)

CHARS_PER_TOKEN = 3.5
# Non-text blocks (images, audio, video) count as this many characters; their
# base64 payload is never serialized into the estimate.
ESTIMATED_MEDIA_CHARS = 4800


@dataclass
class ContextUsageEstimate:
    """Total context estimate, split into the usage-anchored and estimated parts."""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: Optional[int]


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def safe_json(value: Any) -> str:
    """JSON size of a value, serializing dataclass blocks like TS JSON.stringify."""
    try:
        return json.dumps(
            value,
            default=lambda item: asdict(item) if is_dataclass(item) else str(item),
        ) or "undefined"
    except (TypeError, ValueError):
        return "[unserializable]"


def estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_content_chars(content: Union[str, List[Any]]) -> int:
    """Character count of user/toolResult content; media blocks count as fixed-size."""
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        if isinstance(block, (ImageContent, AudioContent, VideoContent)):
            chars += ESTIMATED_MEDIA_CHARS
        else:
            chars += len(block.text)
    return chars


def estimate_tools_tokens(tools: Optional[List[Any]]) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(safe_json(tools))


def estimate_message_tokens(message: Any) -> int:
    """Token estimate for one transcript message (port of estimateMessageTokens)."""
    if isinstance(message, SystemMessage):
        return (
            estimate_text_tokens(get_system_message_text(message))
            + estimate_tools_tokens(message.tools_added)
            + estimate_tools_tokens(message.tools_removed)
        )
    if isinstance(message, (UserMessage, ToolResultMessage)):
        return math.ceil(estimate_content_chars(message.content) / CHARS_PER_TOKEN)

    # Assistant: text and thinking count as text, tool calls as name + arguments JSON.
    chars = 0
    for block in message.content:
        if isinstance(block, ToolCall):
            chars += len(block.name) + len(safe_json(block.arguments))
        elif isinstance(block, ThinkingContent):
            chars += len(block.thinking)
        else:
            chars += len(block.text)
    return math.ceil(chars / CHARS_PER_TOKEN)


def _usage_applies(message: AssistantMessage) -> bool:
    return (
        message.stop_reason not in ("aborted", "error")
        and calculate_context_tokens(message.usage) > 0
    )


def estimate_context_tokens(context: Union[TranscriptContext, List[Any]]) -> ContextUsageEstimate:
    """Estimate the context size of a request.

    Anchors on the most recent applicable assistant usage when available (real
    token counts) and estimates only the messages after it; without usable
    usage every message is estimated. A newer prefix message after the response
    (for example a compaction summary) invalidates its usage, matching the TS
    getLastAssistantUsageInfo timestamp guard.
    """
    messages = context.messages if isinstance(context, TranscriptContext) else context

    latest_prefix_timestamp = float("-inf")
    usage_index: Optional[int] = None
    usage: Optional[Usage] = None
    for index, message in enumerate(messages):
        if (
            isinstance(message, AssistantMessage)
            and message.timestamp >= latest_prefix_timestamp
            and _usage_applies(message)
        ):
            usage_index = index
            usage = message.usage
        latest_prefix_timestamp = max(latest_prefix_timestamp, getattr(message, "timestamp", 0))

    if usage_index is None or usage is None:
        tokens = sum(estimate_message_tokens(message) for message in messages)
        return ContextUsageEstimate(tokens=tokens, usage_tokens=0, trailing_tokens=tokens, last_usage_index=None)

    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = sum(estimate_message_tokens(message) for message in messages[usage_index + 1:])
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=usage_index,
    )

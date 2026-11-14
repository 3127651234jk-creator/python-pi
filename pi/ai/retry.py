"""Retry classification and the policy-driven retry loop.

Python port of ``packages/ai/src/utils/retry.ts``: the
``RETRYABLE_PROVIDER_ERROR_PATTERN`` / ``NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN``
tables, ``isRetryableAssistantError``, and ``retryAssistantCall`` (bounded
attempts with exponential backoff and abort normalization).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Pattern

from ..util import AbortSignal
from .types import AssistantMessage


def _build_provider_error_pattern(patterns: List[str]) -> Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


# OpenCode Go/free-tier limits returned as 429 JSON error types by OpenCode's
# Zen API. These are subscription/account limits, not transient throttles.
# Generic quota/budget/billing exhaustion. ``insufficient_quota`` is OpenAI's
# quota/billing error code; the other strings cover common gateway wording.
NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN = _build_provider_error_pattern([
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
    # Sign in with ChatGPT: the subscription's shared usage limit, which resets
    # after hours rather than seconds.
    "subscription_sharing_usage_limit_exceeded",
])

RETRYABLE_PROVIDER_ERROR_PATTERN = _build_provider_error_pattern([
    # Generic provider load, HTTP status, and server-side transient failures.
    "overloaded",
    "currently experiencing high demand",
    "model is at capacity",
    "rate.?limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "520",
    "524",
    "service.?unavailable",
    "server.?error",
    "internal.?error",
    # Wrapper/provider text for transient upstream failures, including OpenRouter
    # "Provider returned error" responses (#2264).
    "provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    # Network, proxy, and fetch transport failures. This includes OpenAI Codex
    # raw-fetch failures such as "upstream connect", "connection refused", and
    # "reset before headers" (#733), plus OpenRouter connection drops (#3317).
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    # WebSocket transports can report close/error text instead of HTTP/fetch text.
    "websocket.?closed",
    "websocket.?error",
    # Premature stream endings from SDKs and transports. Anthropic can throw
    # "stream ended without ..." and "Anthropic stream ended before message_stop"
    # (#4433); Bedrock/Smithy can throw an HTTP/2 no-response error (#3594).
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    # HTTP/2 ERR_HTTP2_STREAM_CANCEL: the HTTP/2 session died before the request
    # was sent, e.g. after the Bedrock SDK's 5-minute session timeout (#10379).
    "pending stream has been canceled",
    # Provider-requested retry delay cap failures should flow through the outer
    # retry policy so callers can surface/abort the backoff (#1123).
    "retry delay",
    # Explicit retry guidance emitted mid-stream by OpenAI Responses and Bedrock
    # stream exceptions (#6019).
    "you can retry your request",
    "try your request again",
    "please retry your request",
    # gRPC based providers (e.g. NVIDIA NIM)
    "ResourceExhausted",
    # Sign in with ChatGPT: usage or user data temporarily unavailable. Usage
    # failures can arrive mid-stream without an HTTP 503 in the message.
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
])


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """Classify whether a failed assistant message looks like a transient
    provider or transport error, so callers can decide if the last assistant
    turn should be restarted.

    This does not implement retry policy. Callers should first handle context
    overflow separately, then apply their own retry budget, backoff, and
    reporting before restarting the assistant turn.
    """
    if message.stop_reason != "error" or not message.error_message:
        return False
    error_message = message.error_message
    if NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.search(error_message):
        return False
    return RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message) is not None


# ---------------------------------------------------------------------------
# Retry policy (port of RetryPolicy / retryDelayMs / retryAssistantCall)
# ---------------------------------------------------------------------------

DEFAULT_MAX_AGENT_RETRY_DELAY_MS = 60_000


@dataclass
class RetryPolicy:
    """Bounded attempts with exponential backoff (``base_delay_ms * 2^(attempt-1)``).

    Matches ``settings.retry`` (``enabled``, ``max_retries``, ``base_delay_ms``,
    ``max_agent_delay_ms``) in coding-agent. ``max_retries`` counts retries only:
    the initial call never counts as a retry. ``max_agent_delay_ms`` caps each
    computed agent-level delay.
    """

    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000
    max_agent_delay_ms: int = DEFAULT_MAX_AGENT_RETRY_DELAY_MS


def retry_delay_ms(policy: RetryPolicy, attempt: int) -> int:
    """Per-attempt delay ``base_delay_ms * 2^(attempt-1)`` before jitter, capped."""
    delay = policy.base_delay_ms * 2 ** max(0, attempt - 1)
    return min(delay, policy.max_agent_delay_ms)


@dataclass
class RetryCallbacks:
    """Optional callbacks emitted by ``retry_assistant_call`` around each retry."""

    # (attempt, max_attempts, delay_ms, error_message); before each backoff sleep (1-indexed).
    on_retry_scheduled: Optional[Callable[..., object]] = None
    # After the backoff sleep, immediately before the retried call starts.
    on_retry_attempt_start: Optional[Callable[..., object]] = None
    # (success, attempt, final_error=None); once when the loop ends.
    on_retry_finished: Optional[Callable[..., object]] = None


class RetrySleepAbortError(Exception):
    """Raised when the backoff sleep of ``retry_assistant_call`` is aborted."""


async def sleep_ms(ms: int, signal: Optional[AbortSignal]) -> None:
    """Sleep ``ms`` milliseconds, returning early (raising) when ``signal`` aborts."""
    if signal is not None:
        if signal.aborted:
            raise RetrySleepAbortError()
        remainder = ms / 1000
        if remainder <= 0:
            return
        try:
            await asyncio.wait_for(signal.wait(), timeout=remainder)
        except asyncio.TimeoutError:
            return
        raise RetrySleepAbortError()
    await asyncio.sleep(ms / 1000)


async def _emit(callbacks: Optional[RetryCallbacks], name: str, *args: object) -> None:
    callback = getattr(callbacks, name, None) if callbacks is not None else None
    if callback is None:
        return
    result = callback(*args)
    if hasattr(result, "__await__"):
        await result


async def retry_assistant_call(
    produce: Callable[[], AssistantMessage],
    policy: Optional[RetryPolicy],
    signal: Optional[AbortSignal],
    callbacks: Optional[RetryCallbacks] = None,
) -> AssistantMessage:
    """Run a single assistant-producing call with bounded retry on transient errors.

    Behavior (port of ``retryAssistantCall``):
    - A successful response is returned immediately. Aborts are terminal and
      never retried, but reported as unsuccessful if they happen after a retry
      was scheduled. Aborts during the backoff sleep are normalized to an
      aborted ``AssistantMessage`` too, so callers do not care when
      cancellation happened.
    - A non-retryable error (per ``is_retryable_assistant_error``, including
      quota/billing exhaustion) is returned immediately so deterministic errors
      fail fast.
    - Otherwise retries up to ``policy.max_retries`` times with exponential
      backoff, emitting ``on_retry_scheduled`` before each sleep,
      ``on_retry_attempt_start`` after each sleep, and ``on_retry_finished``
      once at the end.

    When ``policy`` is None or disabled, the first response is returned
    unchanged (equivalent to calling ``produce()`` directly).
    """
    max_attempts = policy.max_retries if policy is not None and policy.enabled else 0

    attempt = 0
    last_retry: Optional[dict] = None
    while True:
        response = await produce()

        # Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == "aborted":
            if last_retry is not None:
                await _emit(callbacks, "on_retry_finished", False, last_retry["attempt"])
            return response

        # Success: non-error, non-abort responses return as-is.
        if response.stop_reason != "error":
            if last_retry is not None:
                await _emit(callbacks, "on_retry_finished", True, last_retry["attempt"])
            return response

        # Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None:
                await _emit(callbacks, "on_retry_finished", False, last_retry["attempt"], response.error_message)
            return response

        attempt += 1
        last_retry = {"attempt": attempt, "error_message": response.error_message or "Unknown error"}
        delay = retry_delay_ms(policy, attempt)
        await _emit(callbacks, "on_retry_scheduled", attempt, max_attempts, delay, last_retry["error_message"])

        # Normalize aborts during retry backoff to the same AssistantMessage
        # shape as provider stream aborts, so callers do not need to care when
        # cancellation happened.
        try:
            await sleep_ms(delay, signal)
        except RetrySleepAbortError:
            await _emit(callbacks, "on_retry_finished", False, attempt, last_retry["error_message"])
            return replace(response, stop_reason="aborted", error_message=None)
        await _emit(callbacks, "on_retry_attempt_start")

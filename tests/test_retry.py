"""Provider retry classification and retry-policy loop tests.

Python port of ``packages/ai/test/retry.test.ts``.
"""

import asyncio

import pytest

from pi.ai.retry import (
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
)
from pi.ai.types import AssistantMessage, TextContent, Usage
from pi.util import AbortSignal

OPENAI_EXPLICIT_RETRY_MESSAGE = (
    "An error occurred while processing your request. You can retry your request, or contact us through our "
    "help center at help.openai.com if the error persists. Please include the request ID req_******** in your message."
)
BEDROCK_EXPLICIT_RETRY_MESSAGE = (
    '{"message":"The system encountered an unexpected error during processing. Try your request again."}'
)
NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE = "ResourceExhausted: Worker local total request limit reached (288/48)"
BUN_FETCH_SOCKET_CLOSED_MESSAGE = (
    "The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second "
    "argument to fetch()"
)
OPENAI_RESPONSES_EARLY_EOF_MESSAGE = "OpenAI Responses stream ended before a terminal response event"
WRAPPED_DNS_LOOKUP_ERROR = (
    "The pending stream has been canceled (caused by: getaddrinfo ENOTFOUND bedrock-runtime.us-east-1.amazonaws.com)"
)
AZURE_PEAK_LOAD_ERROR = (
    "The system is currently experiencing high demand and cannot process your request. Your request exceeds the "
    "maximum usage size allowed during peak load. For improved capacity reliability, consider switching to "
    "Provisioned Throughput."
)


def faux_assistant_message(text="", stop_reason="stop", error_message=None):
    message = AssistantMessage(
        content=[TextContent(text=text)] if text else [],
        api="faux",
        provider="faux",
        model="faux",
        usage=Usage(),
        stop_reason=stop_reason,
        timestamp=0,
    )
    if error_message is not None:
        message.error_message = error_message
    return message


def error_message(text):
    return faux_assistant_message(stop_reason="error", error_message=text)


def aborted_message():
    return faux_assistant_message(stop_reason="aborted")


# ---------------------------------------------------------------------------
# retry_assistant_call / retry_delay_ms
# ---------------------------------------------------------------------------


def test_retry_delay_ms_caps_agent_retry_delay():
    # Regression for #8826.
    assert retry_delay_ms(RetryPolicy(base_delay_ms=2000), 6) == 60000
    assert retry_delay_ms(RetryPolicy(base_delay_ms=2000, max_agent_delay_ms=5000), 5) == 5000
    assert retry_delay_ms(RetryPolicy(base_delay_ms=2000, max_agent_delay_ms=0), 5) == 0


def _producer(script, calls):
    index = {"n": 0}

    async def produce():
        index["n"] += 1
        calls.append(index["n"])
        return script[min(index["n"], len(script)) - 1]()

    return produce


def test_retry_assistant_call_returns_success_immediately():
    async def check():
        calls = []
        produce = _producer([lambda: faux_assistant_message("ok")], calls)
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None)
        assert res.content[0].text == "ok"
        assert calls == [1]

    asyncio.run(check())


def test_retry_assistant_call_does_not_retry_an_aborted_message():
    async def check():
        calls = []
        scheduled = []
        produce = _producer([aborted_message], calls)
        callbacks = RetryCallbacks(on_retry_scheduled=lambda *a: scheduled.append(a))
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.stop_reason == "aborted"
        assert calls == [1]
        assert scheduled == []

    asyncio.run(check())


def test_retry_assistant_call_does_not_retry_a_non_retryable_error():
    async def check():
        calls = []
        scheduled = []
        finished = []
        produce = _producer([lambda: error_message("insufficient_quota")], calls)
        callbacks = RetryCallbacks(
            on_retry_scheduled=lambda *a: scheduled.append(a),
            on_retry_finished=lambda *a: finished.append(a),
        )
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.stop_reason == "error"
        assert calls == [1]
        assert scheduled == []
        assert finished == []

    asyncio.run(check())


def test_retry_assistant_call_retries_transient_error_up_to_max_retries():
    async def check():
        calls = []
        scheduled = []
        finished = []

        async def produce():
            calls.append(1)
            return error_message("terminated")

        callbacks = RetryCallbacks(
            on_retry_scheduled=lambda *a: scheduled.append(a),
            on_retry_finished=lambda *a: finished.append(a),
        )
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.stop_reason == "error"
        assert len(calls) == 4  # 1 initial + 3 retries
        assert len(scheduled) == 3
        assert finished == [(False, 3, "terminated")]

    asyncio.run(check())


def test_retry_assistant_call_reports_capped_retry_delays():
    async def check():
        # Regression for #8826.
        counter = {"n": 0}

        async def produce():
            counter["n"] += 1
            if counter["n"] < 5:
                return error_message("terminated")
            return faux_assistant_message("recovered")

        scheduled = []

        def on_retry_scheduled(attempt, max_attempts, delay_ms, error_message):
            scheduled.append(delay_ms)

        policy = RetryPolicy(enabled=True, max_retries=4, base_delay_ms=10, max_agent_delay_ms=15)
        await retry_assistant_call(produce, policy, None, RetryCallbacks(on_retry_scheduled=on_retry_scheduled))
        assert scheduled == [10, 15, 15, 15]

    asyncio.run(check())


def test_retry_assistant_call_stops_retrying_once_a_call_succeeds():
    async def check():
        counter = {"n": 0}
        finished = []

        async def produce():
            counter["n"] += 1
            if counter["n"] < 3:
                return error_message("terminated")
            return faux_assistant_message("recovered")

        callbacks = RetryCallbacks(on_retry_finished=lambda *a: finished.append(a))
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.content[0].text == "recovered"
        assert counter["n"] == 3
        assert finished == [(True, 2)]

    asyncio.run(check())


def test_retry_assistant_call_reports_an_aborted_retried_call_as_unsuccessful():
    async def check():
        counter = {"n": 0}
        finished = []

        async def produce():
            counter["n"] += 1
            if counter["n"] == 1:
                return error_message("terminated")
            return aborted_message()

        callbacks = RetryCallbacks(on_retry_finished=lambda *a: finished.append(a))
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.stop_reason == "aborted"
        assert counter["n"] == 2
        assert finished == [(False, 1)]

    asyncio.run(check())


def test_retry_assistant_call_does_not_retry_when_policy_is_disabled():
    async def check():
        calls = []
        scheduled = []
        finished = []

        async def produce():
            calls.append(1)
            return error_message("terminated")

        callbacks = RetryCallbacks(
            on_retry_scheduled=lambda *a: scheduled.append(a),
            on_retry_finished=lambda *a: finished.append(a),
        )
        res = await retry_assistant_call(
            produce, RetryPolicy(enabled=False, max_retries=3, base_delay_ms=0), None, callbacks
        )
        assert res.stop_reason == "error"
        assert calls == [1]
        assert scheduled == []
        assert finished == []

    asyncio.run(check())


def test_retry_assistant_call_emits_attempt_start_after_backoff():
    async def check():
        events = []
        counter = {"n": 0}

        async def produce():
            events.append(f"produce:{counter['n']}")
            counter["n"] += 1
            if counter["n"] < 3:
                return error_message("terminated")
            return faux_assistant_message("recovered")

        callbacks = RetryCallbacks(
            on_retry_scheduled=lambda attempt, *_: events.append(f"retry:{attempt}"),
            on_retry_attempt_start=lambda: events.append("attempt-start"),
        )
        res = await retry_assistant_call(produce, RetryPolicy(base_delay_ms=0), None, callbacks)
        assert res.content[0].text == "recovered"
        assert events == [
            "produce:0",
            "retry:1",
            "attempt-start",
            "produce:1",
            "retry:2",
            "attempt-start",
            "produce:2",
        ]

    asyncio.run(check())


def test_retry_assistant_call_aborts_backoff_sleep_and_returns_aborted_message():
    async def check():
        signal = AbortSignal()
        calls = []
        finished = []

        async def produce():
            calls.append(1)
            return error_message("terminated")

        callbacks = RetryCallbacks(on_retry_finished=lambda *a: finished.append(a))
        task = asyncio.ensure_future(
            retry_assistant_call(
                produce, RetryPolicy(enabled=True, max_retries=5, base_delay_ms=10_000), signal, callbacks
            )
        )
        # Let one error call resolve and the first backoff sleep start, then abort.
        while not calls:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        signal.abort()
        res = await task
        assert res.stop_reason == "aborted"
        assert res.error_message is None
        assert calls == [1]
        assert finished == [(False, 1, "terminated")]

    asyncio.run(check())


def test_matches_explicit_provider_retry_guidance():
    assert is_retryable_assistant_error(error_message(OPENAI_EXPLICIT_RETRY_MESSAGE)) is True
    assert is_retryable_assistant_error(error_message(BEDROCK_EXPLICIT_RETRY_MESSAGE)) is True
    assert is_retryable_assistant_error(error_message(NVIDIA_NIM_RESOURCE_EXHAUSTED_MESSAGE)) is True


def test_matches_bun_fetch_socket_drop_wording():
    assert is_retryable_assistant_error(error_message(BUN_FETCH_SOCKET_CLOSED_MESSAGE)) is True


def test_matches_upstream_request_buffer_exhaustion_wording():
    assert (
        is_retryable_assistant_error(error_message("Error: exceeded request buffer limit while retrying upstream"))
        is True
    )


@pytest.mark.parametrize(
    "text",
    [
        WRAPPED_DNS_LOOKUP_ERROR,
        "connect ENOTFOUND api.example.com",
        "EAI_AGAIN api.example.com",
        "getaddrinfo failed for api.example.com",
    ],
)
def test_matches_dns_transport_failure_wording(text):
    assert is_retryable_assistant_error(error_message(text)) is True


@pytest.mark.parametrize(
    "text",
    ["The pending stream has been canceled", "The pending stream has been canceled (caused by: socket closed)"],
)
def test_matches_http2_pending_stream_cancellation(text):
    # Regression for #10379.
    assert is_retryable_assistant_error(error_message(text)) is True


def test_matches_openai_responses_streams_that_end_before_terminal_events():
    assert is_retryable_assistant_error(error_message(OPENAI_RESPONSES_EARLY_EOF_MESSAGE)) is True


def test_matches_azure_peak_load_capacity_errors():
    # Regression for #9669.
    assert is_retryable_assistant_error(error_message(AZURE_PEAK_LOAD_ERROR)) is True


def test_keeps_provider_limit_errors_non_retryable():
    assert is_retryable_assistant_error(error_message("429 quota exceeded")) is False


def test_keeps_the_chatgpt_subscription_usage_limit_non_retryable():
    text = (
        'OpenAI API error (429): {"code":"subscription_sharing_usage_limit_exceeded",'
        '"message":"Usage limit reached."}'
    )
    assert is_retryable_assistant_error(error_message(text)) is False


@pytest.mark.parametrize(
    "text",
    [
        "subscription_sharing_usage_unavailable: Usage cannot be checked.",
        "subscription_sharing_user_unavailable: User cannot be loaded.",
    ],
)
def test_retries_temporary_chatgpt_subscription_errors(text):
    assert is_retryable_assistant_error(error_message(text)) is True


def test_classifies_assistant_error_messages():
    assert is_retryable_assistant_error(error_message("overloaded_error")) is True
    # Regression for #9627.
    assert is_retryable_assistant_error(error_message("520 status code (no body)")) is True
    assert is_retryable_assistant_error(error_message("524 status code (no body)")) is True
    assert is_retryable_assistant_error(faux_assistant_message("not an error")) is False

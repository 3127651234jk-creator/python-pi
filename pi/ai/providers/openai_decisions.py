"""OpenAI Decisions classifier API (port of packages/ai/src/api/openai-decisions.ts).

``POST {baseUrl}/decisions`` with ``{model, input, questions}``: the state is
sent as JSON text; with images the input becomes one user message with the
state as ``input_text`` followed by ``input_image`` data URLs. Questions map
``bool`` to ``predicate`` (the true/false meanings join the instructions).
Only OpenAI API keys work on this route; Sign in with ChatGPT tokens are
rejected, and 504 gateway timeouts on very large inputs are not retried.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import Any, Dict, List, Optional

import httpx

from ..types import (
    ClassifierBoolAnswer,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierQuestion,
    ClassifierResult,
    ClassifierScoreAnswer,
    ImageContent,
    Usage,
)
from ..models import Model
from ..utils import normalize_provider_error_body
from .base import SimpleStreamOptions

LABEL = "OpenAI Decisions"
MAX_IMAGES = 128
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
# Cloudflare in front of api.openai.com answers 504 with an HTML page when a
# request runs longer than about five seconds; retrying runs into it again.
NO_RETRY_STATUSES = (504,)


class ClassifierHttpError(Exception):
    def __init__(self, message: str, status: Optional[int], headers: Dict[str, str], body: str) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers
        self.body = body


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _required_number(label: str, value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{LABEL} returned an invalid {field}")
    return value


def _predicate_instructions(question) -> str:
    meanings = []
    if question.criteria.get("true"):
        meanings.append(f"True means: {question.criteria['true']}")
    if question.criteria.get("false"):
        meanings.append(f"False means: {question.criteria['false']}")
    return f"{question.instructions}\n\n" + "\n".join(meanings) if meanings else question.instructions


def _wire_question(name: str, question: ClassifierQuestion) -> dict:
    if question.type == "choice":
        return {"type": "choice", "name": name, "instructions": question.instructions,
                "choices": [{"value": value, **({"description": text} if text else {})}
                            for value, text in question.criteria.items()]}
    if question.type == "score":
        return {"type": "score", "name": name, "instructions": question.instructions,
                "levels": [{"label": label} for label in question.criteria]}
    return {"type": "predicate", "name": name, "instructions": _predicate_instructions(question)}


def _wire_input(context: ClassifierContext) -> Any:
    state = json.dumps(context.state, ensure_ascii=False)
    images = context.images or []
    if not images:
        return state
    if len(images) > MAX_IMAGES:
        raise ValueError(f"{LABEL} accepts at most {MAX_IMAGES} images, got {len(images)}")
    return [{"role": "user", "content": [{"type": "input_text", "text": state}] + [
        {"type": "input_image", "image_url": f"data:{image.mime_type};base64,{image.data}"}
        for image in images]}]


def _choice_probabilities(value: Any, question_id: str) -> Dict[str, float]:
    if not isinstance(value, list):
        raise ValueError(f"{LABEL} returned invalid probabilities for {question_id}")
    result = {}
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
            raise ValueError(f"{LABEL} returned invalid probabilities for {question_id}")
        result[entry["value"]] = _required_number(LABEL, entry.get("probability"),
                                                  f"probability for {question_id}.{entry['value']}")
    return result


def _parse_answer(question_id: str, question: ClassifierQuestion, answer: dict) -> Any:
    if answer.get("type") == "refusal":
        raise ValueError(f"{LABEL} refused to answer {question_id}")
    if question.type == "choice":
        if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
            raise ValueError(f"{LABEL} did not return a choice answer for {question_id}")
        return ClassifierChoiceAnswer(
            choice=answer["choice"],
            probabilities=_choice_probabilities(answer.get("probabilities"), question_id),
            confidence=_required_number(LABEL, answer.get("confidence"), f"confidence for {question_id}"),
        )
    if question.type == "score":
        if answer.get("type") != "score":
            raise ValueError(f"{LABEL} did not return a score answer for {question_id}")
        return ClassifierScoreAnswer(
            score=_required_number(LABEL, answer.get("score"), f"score for {question_id}"),
            confidence=_required_number(LABEL, answer.get("confidence"), f"confidence for {question_id}"),
        )
    if answer.get("type") != "predicate":
        raise ValueError(f"{LABEL} did not return a predicate answer for {question_id}")
    return ClassifierBoolAnswer(
        probability=_required_number(LABEL, answer.get("probability"), f"probability for {question_id}"))


def _parse_answers(value: Any, context: ClassifierContext) -> Dict[str, Any]:
    if not isinstance(value, list):
        raise ValueError(f"{LABEL} returned an unexpected response")
    by_name = {answer["name"]: answer for answer in value
               if isinstance(answer, dict) and isinstance(answer.get("name"), str)}
    answers: Dict[str, Any] = {}
    for question_id, question in context.questions.items():
        answer = by_name.get(question_id)
        if answer is None:
            raise ValueError(f"{LABEL} did not return an answer for {question_id}")
        answers[question_id] = _parse_answer(question_id, question, answer)
    return answers


def _retry_delay_ms(error: ClassifierHttpError, retry_index: int) -> float:
    headers = {name.lower(): value for name, value in (error.headers or {}).items()}
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return min(float(retry_after_ms), DEFAULT_MAX_RETRY_DELAY_MS)
        except ValueError:
            pass
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after) * 1000, DEFAULT_MAX_RETRY_DELAY_MS)
        except ValueError:
            pass
    return min(0.5 * 2 ** retry_index, 8.0) * 1000 * (1 - random.random() * 0.25)


def _is_retryable(error: ClassifierHttpError, no_retry: tuple) -> bool:
    headers = {name.lower(): value for name, value in (error.headers or {}).items()}
    should_retry = headers.get("x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    if error.status is None:
        return True
    if error.status in no_retry:
        return False
    return error.status in (408, 409, 429) or error.status >= 500


async def _post_classifier_request(url: str, model: Model, body: dict,
                                   options: Optional[SimpleStreamOptions]) -> Any:
    """POST one JSON classifier request with bearer auth and provider-style
    retries (port of postClassifierRequest + retryProviderRequest)."""
    if options is None or not options.api_key:
        raise ValueError(f"No API key for provider: {model.provider}")
    max_retries = DEFAULT_MAX_RETRIES
    retries_remaining = max_retries
    signal = options.signal
    while True:
        if signal is not None and signal.aborted:
            raise asyncio.CancelledError()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0),
                                         trust_env=False) as client:
                request = client.build_request(
                    "POST", url, json=body,
                    headers={"authorization": f"Bearer {options.api_key}", "content-type": "application/json",
                             **{name.lower(): value for name, value in model.headers.items()}})
                if options.on_payload is not None:
                    replacement = options.on_payload(body, model)
                    if hasattr(replacement, "__await__"):
                        replacement = await replacement
                    if replacement is not None:
                        request = client.build_request(
                            "POST", url, json=replacement,
                            headers={"authorization": f"Bearer {options.api_key}", "content-type": "application/json",
                                     **{name.lower(): value for name, value in model.headers.items()}})
                response = await client.send(request)
                response_body = response.text
                if response.status_code >= 400:
                    raise ClassifierHttpError(f"{LABEL} returned {response.status_code}", response.status_code,
                                              dict(response.headers), response_body)
                await response.aread()
                if options.on_response is not None:
                    hook = options.on_response({"status": response.status_code, "headers": dict(response.headers)})
                    if hasattr(hook, "__await__"):
                        await hook
                return json.loads(response_body)
        except ClassifierHttpError as error:
            if signal is not None and signal.aborted:
                raise asyncio.CancelledError()
            if retries_remaining <= 0 or not _is_retryable(error, NO_RETRY_STATUSES):
                raise
            retries_remaining -= 1
            await asyncio.sleep(_retry_delay_ms(error, max_retries - retries_remaining - 1) / 1000)


def _parse_usage(value: Any, model: Model) -> Optional[Usage]:
    if not _is_record(value) or ("input_tokens" not in value and "output_tokens" not in value):
        return None

    def token_count(item: Any) -> int:
        return item if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item > 0 else 0

    input_tokens = token_count(value.get("input_tokens"))
    output_tokens = token_count(value.get("output_tokens"))
    usage = Usage(input=input_tokens, output=output_tokens)
    usage.cost = model.calculate_cost(usage)
    return usage


def _error_message(error: Exception) -> str:
    if isinstance(error, ClassifierHttpError) and error.status == 504:
        return (f"{LABEL} error (504): the request timed out at the gateway. "
                "Very large inputs (above roughly 600K tokens) currently exceed its time limit.")
    if isinstance(error, ClassifierHttpError):
        detail = normalize_provider_error_body(error.body) or "(empty response body)"
        return f"{LABEL} error ({error.status}): {detail}"
    return f"{LABEL} error: {error}"


async def classify_openai_decisions(model: Model, context: ClassifierContext,
                                    options: Optional[SimpleStreamOptions] = None) -> ClassifierResult:
    """Classification through OpenAI's Decisions API."""
    result = ClassifierResult(api=model.api, provider=model.provider, model=model.id,
                              timestamp=int(time.time() * 1000))
    try:
        if model.api != "openai-decisions":
            raise ValueError(f"Unsupported classifier API: {model.api}")
        body = await _post_classifier_request(
            model.base_url.rstrip("/") + "/decisions", model,
            {"model": model.id, "input": _wire_input(context),
             "questions": [_wire_question(name, question) for name, question in context.questions.items()]},
            options)
        if not _is_record(body):
            raise ValueError(f"{LABEL} returned an unexpected response")
        # Parse usage before answers: a malformed or refused answer set was still billed.
        result.usage = _parse_usage(body.get("usage"), model)
        result.answers = _parse_answers(body.get("answers"), context)
        return result
    except asyncio.CancelledError:
        result.answers = {}
        result.stop_reason = "aborted"
        result.error_message = "Request aborted"
        return result
    except Exception as exc:  # noqa: BLE001 - encoded as an error result, like the TS contract
        result.answers = {}
        aborted = options is not None and options.signal is not None and options.signal.aborted
        result.stop_reason = "aborted" if aborted else "error"
        result.error_message = _error_message(exc)
        return result

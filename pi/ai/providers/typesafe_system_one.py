"""TypeSafe System One classifier protocol (port of packages/ai/src/api/system-one-shared.ts
and typesafe-system-one.ts).

``POST {base}/systemone`` with ``{model, state, questions}``: public ``bool``
questions map to the wire-level ``noul`` type, and bool answers come back as
``noul``. OpenRouter serves the same protocol, so both providers use this API
with different base URLs. The llama.cpp extension's decision-only models (0.6.0+)
also expose it at ``/v1/systemone``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..models import Model
from ..types import (
    ClassifierBoolAnswer,
    ClassifierChoiceAnswer,
    ClassifierContext,
    ClassifierResult,
    ClassifierScoreAnswer,
    Usage,
)
from ..utils import normalize_provider_error_body
from .base import SimpleStreamOptions
from .openai_decisions import (
    ClassifierHttpError,
    _is_record,
    _parse_usage,
    _post_classifier_request,
)

LABEL = "System One API"


def _probabilities(value: Any, question_id: str) -> Dict[str, float]:
    if not _is_record(value):
        raise ValueError(f"{LABEL} returned invalid probabilities for {question_id}")
    result: Dict[str, float] = {}
    for key, probability in value.items():
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"{LABEL} returned an invalid probability for {question_id}.{key}")
        result[key] = probability
    return result


def _parse_answers(value: Any, context: ClassifierContext) -> Dict[str, Any]:
    if not _is_record(value):
        raise ValueError(f"{LABEL} returned an unexpected response")
    answers: Dict[str, Any] = {}
    for question_id, question in context.questions.items():
        answer = value.get(question_id)
        if not _is_record(answer):
            raise ValueError(f"{LABEL} did not return an answer for {question_id}")
        if question.type == "choice":
            if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str):
                raise ValueError(f"{LABEL} did not return a choice answer for {question_id}")
            answers[question_id] = ClassifierChoiceAnswer(
                choice=answer["choice"],
                probabilities=_probabilities(answer.get("probabilities"), question_id),
                confidence=answer.get("confidence") if isinstance(answer.get("confidence"), (int, float)) else 0.0,
            )
        elif question.type == "score":
            if answer.get("type") != "score":
                raise ValueError(f"{LABEL} did not return a score answer for {question_id}")
            answers[question_id] = ClassifierScoreAnswer(score=answer.get("score", 0.0))
        else:
            # Public bool questions map to wire-level noul answers.
            if answer.get("type") != "noul":
                raise ValueError(f"{LABEL} did not return a bool answer for {question_id}")
            answers[question_id] = ClassifierBoolAnswer(probability=answer.get("noul", 0.0))
    return answers


def _wire_request(context: ClassifierContext) -> dict:
    questions: Dict[str, Any] = {}
    for question_id, question in context.questions.items():
        if question.type == "bool":
            questions[question_id] = {"type": "noul", "instructions": question.instructions,
                                      "criteria": dict(question.criteria)}
        elif question.type == "choice":
            questions[question_id] = {"type": "choice", "instructions": question.instructions,
                                      "criteria": dict(question.criteria)}
        else:
            questions[question_id] = {"type": "score", "instructions": question.instructions,
                                      "criteria": list(question.criteria)}
    return {"state": context.state, "questions": questions}


async def classify_system_one(model: Model, context: ClassifierContext,
                              options: Optional[SimpleStreamOptions] = None) -> ClassifierResult:
    """One System One classification over the HTTP transport."""
    result = ClassifierResult(api=model.api, provider=model.provider, model=model.id,
                              timestamp=int(time.time() * 1000))
    try:
        if model.api != "typesafe-system-one":
            raise ValueError(f"Unsupported classifier API: {model.api}")
        if context.images:
            raise ValueError(f"{LABEL} does not support image input")
        body = await _post_classifier_request(
            model.base_url.rstrip("/") + "/systemone", model,
            {"model": model.id, **_wire_request(context)}, options)
        if not _is_record(body):
            raise ValueError(f"{LABEL} returned an unexpected response")
        # Parse usage before answers: a request with malformed answers was still billed.
        result.usage = _parse_usage(body.get("usage"), model)
        result.answers = _parse_answers(body.get("answers"), context)
        return result
    except ClassifierHttpError as error:
        result.answers = {}
        result.stop_reason = "aborted" if (options is not None and options.signal is not None
                                           and options.signal.aborted) else "error"
        detail = normalize_provider_error_body(error.body) or "(empty response body)"
        result.error_message = f"{LABEL} error ({error.status}): {detail}"
        return result
    except Exception as exc:  # noqa: BLE001 - encoded as an error result, like the TS contract
        result.answers = {}
        result.stop_reason = "aborted" if (options is not None and options.signal is not None
                                           and options.signal.aborted) else "error"
        result.error_message = f"{LABEL} error: {exc}"
        return result

"""Provider registry: maps an API name to its stream function.

Extensions add custom streaming protocols two ways: register a stream
function for a new ``api`` name, or attach a ``stream_simple`` callable to a
registered provider, which overrides dispatch for that provider's models.

Image generation is dispatched the same way through ``IMAGES_FNS``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable, Dict, Optional

from ..models import Model
from ..types import (
    AssistantImages,
    ClassifierContext,
    ClassifierResult,
    ImagesContext,
    ImagesOptions,
    TranscriptContext,
)
from .amazon_bedrock import stream_bedrock
from .anthropic import stream_anthropic
from .base import DoneEvent, SimpleStreamOptions, StreamErrorEvent, StreamFn
from .faux import stream_faux
from .google import stream_google
from .openai_completions import stream_openai_completions
from .openai_decisions import classify_openai_decisions
from .openai_responses import stream_openai_responses
from .openrouter_images import generate_images_openrouter
from .typesafe_system_one import classify_system_one
from .vertex import stream_vertex
from .pi_messages import stream_pi_messages

STREAM_FNS: Dict[str, StreamFn] = {
    "pi-messages": stream_pi_messages,
    "azure-openai-responses": stream_openai_responses,
    "anthropic-messages": stream_anthropic,
    "openai-completions": stream_openai_completions,
    "faux": stream_faux,
    "openai-responses": stream_openai_responses,
    "openai-codex-responses": stream_openai_responses,
    "google-generative-ai": stream_google,
    "google-vertex": stream_vertex,
    "azure-openai": stream_openai_completions,
    "bedrock-converse-stream": stream_bedrock,
}

PROVIDER_STREAM_FNS: Dict[str, StreamFn] = {}

ImagesFn = Callable[[Model, ImagesContext, Optional[ImagesOptions]], Awaitable[AssistantImages]]

IMAGES_FNS: Dict[str, ImagesFn] = {}

# Classifier APIs (port of the TS provider.classify hooks): api name -> classify fn.
ClassifierFn = Callable[[Model, "ClassifierContext", Optional[SimpleStreamOptions]], Awaitable["ClassifierResult"]]

CLASSIFIER_FNS: Dict[str, ClassifierFn] = {}


def register_classifier_fn(api: str, fn: ClassifierFn) -> None:
    """Register (or replace) the classifier function for an API name."""

    if not callable(fn):
        raise ValueError("Classifier function must be callable")
    CLASSIFIER_FNS[api] = fn


register_classifier_fn("openai-decisions", classify_openai_decisions)
register_classifier_fn("typesafe-system-one", classify_system_one)


async def classify(
    model: Model, context: ClassifierContext, options: Optional[SimpleStreamOptions] = None
) -> ClassifierResult:
    """Classify via the classifier API registry, resolving auth when the caller
    did not pass an explicit API key (port of TS Models.classify()). Like the
    TS wrapper, failures — including images on a text-only classifier — come
    back as an error result instead of raising."""
    result = ClassifierResult(api=model.api, provider=model.provider, model=model.id,
                              timestamp=int(time.time() * 1000))
    try:
        fn = CLASSIFIER_FNS.get(model.api)
        if fn is None:
            raise ValueError(f"No classifier API registered for api: {model.api}")
        if context.images and "image" not in model.input:
            raise ValueError(f"Model {model.provider}/{model.id} does not accept image inputs")
        resolved = options or SimpleStreamOptions()
        if not resolved.api_key:
            key = None
            if model.get_api_key is not None:
                key = await model.get_api_key(model.provider)
            resolved = replace(resolved, api_key=key or model.api_key)
        return await fn(model, context, resolved)
    except asyncio.CancelledError:
        result.stop_reason = "aborted"
        result.error_message = "Request aborted"
        return result
    except Exception as exc:  # noqa: BLE001 - encoded as an error result, like the TS contract
        result.stop_reason = "aborted" if (options is not None and options.signal is not None
                                           and options.signal.aborted) else "error"
        result.error_message = str(exc)
        return result


def register_stream_fn(api: str, fn: StreamFn) -> None:
    """Register (or replace) the stream function for an API name."""

    if not callable(fn):
        raise ValueError("Stream function must be callable")
    STREAM_FNS[api] = fn


def register_provider_stream_fn(provider: str, fn: StreamFn) -> None:
    """Override streaming for one provider, whatever API its models declare."""

    if not callable(fn):
        raise ValueError("Provider stream function must be callable")
    PROVIDER_STREAM_FNS[provider] = fn


def unregister_provider_stream_fn(provider: str) -> None:
    PROVIDER_STREAM_FNS.pop(provider, None)


def register_images_fn(api: str, fn: ImagesFn) -> None:
    """Register (or replace) the image generation function for an API name."""

    if not callable(fn):
        raise ValueError("Images function must be callable")
    IMAGES_FNS[api] = fn


register_images_fn("openrouter-images", generate_images_openrouter)


async def generate_images(
    model: Model, context: ImagesContext, options: Optional[ImagesOptions] = None
) -> AssistantImages:
    """Generate images via the images API registry.

    Resolves provider auth when the caller did not pass an explicit API key,
    mirroring the TS ``Models.generateImages()`` convenience layer.
    """
    fn = IMAGES_FNS.get(model.api)
    if fn is None:
        raise ValueError(f"No images API provider registered for api: {model.api}")
    resolved = options or ImagesOptions()
    if not resolved.api_key:
        key = None
        if model.get_api_key is not None:
            key = await model.get_api_key(model.provider)
        resolved = replace(resolved, api_key=key or model.api_key)
    return await fn(model, context, resolved)


def get_stream_fn(api: str):
    fn = STREAM_FNS.get(api)
    if fn is None:
        raise KeyError(f"No stream function registered for API '{api}'")
    return fn


def get_default_stream_fn():
    """Default StreamFn: a provider-level override first, then dispatch on ``model.api``.

    Every terminating event gets the response's wall time stamped into the
    final message (port of the TS AssistantMessageEventStream timing): a
    monotonic clock runs from wrapper creation to the done/error event, and
    messages that already carry a duration or predate the wrapper (deferred
    results fetched later) are left untouched.
    """

    async def stream(model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions] = None):
        override = PROVIDER_STREAM_FNS.get(model.provider)
        fn = override if override is not None else get_stream_fn(model.api)
        started_at_ms = int(time.time() * 1000)
        started_monotonic = time.perf_counter()
        finished = False

        def time_final(message) -> None:
            if message is None or message.duration_ms is not None or message.timestamp < started_at_ms:
                return
            message.duration_ms = max(0, round((time.perf_counter() - started_monotonic) * 1000))

        async for event in fn(model, context, options):
            if not finished and isinstance(event, DoneEvent):
                time_final(event.message)
                finished = True
            elif not finished and isinstance(event, StreamErrorEvent):
                time_final(event.error)
                finished = True
            yield event

    return stream


__all__ = [
    "STREAM_FNS",
    "PROVIDER_STREAM_FNS",
    "IMAGES_FNS",
    "CLASSIFIER_FNS",
    "ImagesFn",
    "ClassifierFn",
    "SimpleStreamOptions",
    "StreamFn",
    "get_stream_fn",
    "get_default_stream_fn",
    "register_stream_fn",
    "register_provider_stream_fn",
    "unregister_provider_stream_fn",
    "register_images_fn",
    "register_classifier_fn",
    "generate_images",
    "classify",
    "stream_anthropic",
    "stream_openai_completions",
    "stream_faux",
]

"""Provider registry: maps an API name to its stream function.

Extensions add custom streaming protocols two ways: register a stream
function for a new ``api`` name, or attach a ``stream_simple`` callable to a
registered provider, which overrides dispatch for that provider's models.

Image generation is dispatched the same way through ``IMAGES_FNS``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable, Dict, Optional

from ..models import Model
from ..types import AssistantImages, ImagesContext, ImagesOptions, TranscriptContext
from .amazon_bedrock import stream_bedrock
from .anthropic import stream_anthropic
from .base import SimpleStreamOptions, StreamFn
from .faux import stream_faux
from .google import stream_google
from .openai_completions import stream_openai_completions
from .openai_responses import stream_openai_responses
from .openrouter_images import generate_images_openrouter
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
    """Default StreamFn: a provider-level override first, then dispatch on ``model.api``."""

    async def stream(model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions] = None):
        override = PROVIDER_STREAM_FNS.get(model.provider)
        fn = override if override is not None else get_stream_fn(model.api)
        async for event in fn(model, context, options):
            yield event

    return stream


__all__ = [
    "STREAM_FNS",
    "PROVIDER_STREAM_FNS",
    "IMAGES_FNS",
    "ImagesFn",
    "SimpleStreamOptions",
    "StreamFn",
    "get_stream_fn",
    "get_default_stream_fn",
    "register_stream_fn",
    "register_provider_stream_fn",
    "unregister_provider_stream_fn",
    "register_images_fn",
    "generate_images",
    "stream_anthropic",
    "stream_openai_completions",
    "stream_faux",
]

"""Provider registry: maps an API name to its stream function.

Extensions add custom streaming protocols two ways: register a stream
function for a new ``api`` name, or attach a ``stream_simple`` callable to a
registered provider, which overrides dispatch for that provider's models.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Dict, Optional

from ..models import Model
from ..types import TranscriptContext
from .amazon_bedrock import stream_bedrock
from .anthropic import stream_anthropic
from .base import SimpleStreamOptions, StreamFn
from .faux import stream_faux
from .google import stream_google
from .openai_completions import stream_openai_completions
from .openai_responses import stream_openai_responses
from .vertex import stream_vertex

STREAM_FNS: Dict[str, StreamFn] = {
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
    "SimpleStreamOptions",
    "StreamFn",
    "get_stream_fn",
    "get_default_stream_fn",
    "register_stream_fn",
    "register_provider_stream_fn",
    "unregister_provider_stream_fn",
    "stream_anthropic",
    "stream_openai_completions",
    "stream_faux",
]

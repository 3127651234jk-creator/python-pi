"""Model definitions and a small environment-driven catalog.

Python port of the ``Model`` type from ``packages/ai/src/types.ts`` and a
minimal replacement for the generated model catalog. Providers are configured
through environment variables or CLI flags; unknown model ids fall back to
sane defaults so any OpenAI-compatible endpoint works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class ModelCost:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    cache_write_1h: float = 0.0


@dataclass
class ModelImageResizeOptions:
    """Per-model resize profile applied before a new image enters history."""

    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_bytes: Optional[int] = None
    jpeg_quality: Optional[int] = None


@dataclass
class ModelImageInputLimits:
    """Image limits for one model (port of ModelImageInputLimits)."""

    resize: Optional[ModelImageResizeOptions] = None
    max_per_message: Optional[int] = None
    max_per_request: Optional[int] = None


@dataclass
class ModelInputLimits:
    """Provider input limits (port of ModelInputLimits)."""

    max_request_bytes: Optional[int] = None
    images: Optional[ModelImageInputLimits] = None


def parse_input_limits(raw: Optional[dict]) -> Optional[ModelInputLimits]:
    """Build ModelInputLimits from a catalog entry (camelCase JSON keys)."""
    if not raw:
        return None
    images = raw.get("images") or {}
    resize = images.get("resize") or {}
    return ModelInputLimits(
        max_request_bytes=raw.get("maxRequestBytes"),
        images=ModelImageInputLimits(
            resize=ModelImageResizeOptions(
                max_width=resize.get("maxWidth"),
                max_height=resize.get("maxHeight"),
                max_bytes=resize.get("maxBytes"),
                jpeg_quality=resize.get("jpegQuality"),
            )
            if resize
            else None,
            max_per_message=images.get("maxPerMessage"),
            max_per_request=images.get("maxPerRequest"),
        )
        if images
        else None,
    )


@dataclass
class Model:
    id: str
    name: str
    api: str  # "anthropic-messages" | "openai-completions" | "faux"
    provider: str
    base_url: str
    api_key: Optional[str] = field(default=None, repr=False)
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8_192
    input: List[str] = field(default_factory=lambda: ["text"])
    output: Optional[List[str]] = None  # image models: ["image"] or ["image", "text"]
    input_limits: Optional[ModelInputLimits] = field(default=None, repr=False)
    costs: ModelCost = field(default_factory=ModelCost)
    headers: Dict[str, str] = field(default_factory=dict, repr=False)
    params: Dict[str, Any] = field(default_factory=dict, repr=False)
    auth_type: str = field(default="api_key", repr=False)
    get_api_key: Optional[Callable[[str], Awaitable[Optional[str]]]] = field(default=None, repr=False, compare=False)
    # Provider-specific thinking-level mapping (pi level -> provider level or None),
    # carried verbatim from catalogs such as the Radius gateway /v1/config response.
    thinking_level_map: Optional[Dict[str, Optional[str]]] = None

    def calculate_cost(self, usage) -> Dict[str, float]:
        m = 1_000_000.0
        input_cost = usage.input * self.costs.input / m
        output_cost = usage.output * self.costs.output / m
        cache_read_cost = usage.cache_read * self.costs.cache_read / m
        cache_write_cost = usage.cache_write * self.costs.cache_write / m
        # Anthropic extended-TTL rule: 1h cache writes bill at 2x base input
        # unless the catalog carries a dedicated 1h price. Only Anthropic
        # reports cache_write_1h tokens, so the fallback is safe elsewhere.
        write_1h_price = self.costs.cache_write_1h or 2 * self.costs.input
        cache_write_1h_cost = usage.cache_write_1h * write_1h_price / m
        return {
            "input": input_cost,
            "output": output_cost,
            "cache_read": cache_read_cost,
            "cache_write": cache_write_cost,
            "cache_write_1h": cache_write_1h_cost,
            "total": input_cost + output_cost + cache_read_cost + cache_write_cost + cache_write_1h_cost,
        }


class ModelError(Exception):
    pass


# A handful of well-known models; anything else uses the provider defaults.
_CATALOG: Dict[str, Dict] = {
    "anthropic/claude-sonnet-4-5-20250929": {
        "name": "Claude Sonnet 4.5",
        "reasoning": True,
        "context_window": 200_000,
        "max_tokens": 64_000,
        "costs": ModelCost(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    },
    "anthropic/claude-opus-4-1-20250805": {
        "name": "Claude Opus 4.1",
        "reasoning": True,
        "context_window": 200_000,
        "max_tokens": 32_000,
        "costs": ModelCost(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
    },
    "anthropic/claude-haiku-4-5-20251001": {
        "name": "Claude Haiku 4.5",
        "reasoning": True,
        "context_window": 200_000,
        "max_tokens": 64_000,
        "costs": ModelCost(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25),
    },
    "openai/gpt-4o": {
        "name": "GPT-4o",
        "context_window": 128_000,
        "max_tokens": 16_384,
        "costs": ModelCost(input=2.5, output=10.0, cache_read=1.25),
    },
    "openai/gpt-4o-mini": {
        "name": "GPT-4o mini",
        "context_window": 128_000,
        "max_tokens": 16_384,
        "costs": ModelCost(input=0.15, output=0.6, cache_read=0.075),
    },
    "openai/gpt-5.1": {
        "name": "GPT-5.1",
        "reasoning": True,
        "context_window": 400_000,
        "max_tokens": 128_000,
        "costs": ModelCost(input=1.25, output=10.0, cache_read=0.125),
    },
}

_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o-mini",
    "faux": "faux",
}


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else None


def resolve_model(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Model:
    """Resolve a Model from explicit values or environment variables.

    Environment variables: ``PI_PY_PROVIDER``, ``PI_PY_MODEL``,
    ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` (or ``PI_PY_API_KEY``),
    ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL`` (or ``PI_PY_BASE_URL``).
    """

    provider = provider or _env("PI_PY_PROVIDER") or "openai"
    provider = provider.lower()
    if provider in ("google", "openai-responses", "amazon-bedrock"):
        defaults = {
            "google": ("gemini-2.5-flash", "google-generative-ai", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
            "openai-responses": ("gpt-4o-mini", "openai-responses", "https://api.openai.com/v1", "OPENAI_API_KEY"),
            "amazon-bedrock": ("anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock-converse-stream",
                "https://bedrock-runtime." + (_env("AWS_REGION") or _env("AWS_DEFAULT_REGION") or "us-east-1") + ".amazonaws.com", ""),
        }
        default_id, api, endpoint, key_name = defaults[provider]
        key = api_key or _env("PI_PY_API_KEY") or (_env(key_name) if key_name else None)
        if not key and provider != "amazon-bedrock":
            raise ModelError(f"Missing API key: set {key_name}")
        identifier = model_id or _env("PI_PY_MODEL") or default_id
        return Model(id=identifier, name=identifier, api=api, provider=provider,
            base_url=(base_url or _env("PI_PY_BASE_URL") or endpoint).rstrip("/"), api_key=key,
            input=["text", "image"], reasoning=provider == "google")
    if provider not in ("anthropic", "openai", "faux"):
        raise ModelError(
            f"Unknown provider '{provider}'. Supported: anthropic, openai (OpenAI-compatible), faux."
        )

    model_id = model_id or _env("PI_PY_MODEL") or _PROVIDER_DEFAULT_MODEL[provider]

    if provider == "faux":
        return Model(id=model_id, name="Faux", api="faux", provider="faux", base_url="")

    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    resolved_key = api_key or _env("PI_PY_API_KEY") or _env(key_env)
    if not resolved_key:
        raise ModelError(f"Missing API key: set {key_env} (or pass --api-key / PI_PY_API_KEY).")

    default_base = "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1"
    resolved_base = base_url or _env("PI_PY_BASE_URL") or _env(
        "ANTHROPIC_BASE_URL" if provider == "anthropic" else "OPENAI_BASE_URL"
    ) or default_base
    resolved_base = resolved_base.rstrip("/")

    entry = _CATALOG.get(f"{provider}/{model_id}", {})
    api = "anthropic-messages" if provider == "anthropic" else "openai-completions"
    return Model(
        id=model_id,
        name=entry.get("name", model_id),
        api=api,
        provider=provider,
        base_url=resolved_base,
        api_key=resolved_key,
        reasoning=entry.get("reasoning", False),
        context_window=entry.get("context_window", 200_000 if provider == "anthropic" else 128_000),
        max_tokens=entry.get("max_tokens", 8_192),
        input=entry.get("input", ["text"]),
        input_limits=parse_input_limits(entry.get("input_limits")),
        costs=entry.get("costs", ModelCost()),
    )


def resolve_image_model(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Model:
    """Resolve an image-generation model (currently openrouter-images only)."""

    provider = (provider or _env("PI_PY_PROVIDER") or "openrouter").lower()
    if provider != "openrouter":
        raise ModelError(f"Unknown image provider '{provider}'. Supported: openrouter.")
    key = api_key or _env("PI_PY_API_KEY") or _env("OPENROUTER_API_KEY")
    if not key:
        raise ModelError("Missing API key: set OPENROUTER_API_KEY (or pass PI_PY_API_KEY).")
    identifier = model_id or _env("PI_PY_MODEL") or "google/gemini-2.5-flash-image"
    endpoint = (
        base_url or _env("PI_PY_BASE_URL") or _env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    )
    return Model(
        id=identifier,
        name=identifier,
        api="openrouter-images",
        provider="openrouter",
        base_url=endpoint.rstrip("/"),
        api_key=key,
        input=["text", "image"],
        output=["image", "text"],
    )


def list_catalog() -> List[str]:
    return sorted(_CATALOG.keys())

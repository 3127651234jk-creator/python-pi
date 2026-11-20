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
    # Request-wide prompt-length pricing tiers; the highest matching input
    # threshold applies to the full request (port of ModelCost.tiers).
    tiers: Optional[List[Dict[str, Any]]] = None


def parse_cost_tiers(raw: Any, base: Optional[Dict[str, float]] = None) -> Optional[List[Dict[str, Any]]]:
    """Normalize pricing tiers from catalogs or models.json into snake_case dicts.

    Accepts the models.dev wire shape (``{"tier": {"type": "context", "size":
    N}, "input": ...}``), the normalized shape (``{"input_tokens_above": N,
    ...}``), the gateway shape (``{"inputTokensAbove": N, ...}``), and
    camelCase rate keys. Rates a tier does not list keep the base price;
    entries without a valid numeric threshold are dropped. Returns None when
    nothing valid remains.
    """
    if not isinstance(raw, list):
        return None
    tiers: List[Dict[str, Any]] = []
    for tier in raw:
        if not isinstance(tier, dict):
            continue
        above = tier.get("input_tokens_above")
        if above is None:
            above = tier.get("inputTokensAbove")
        context = tier.get("tier")
        if above is None and isinstance(context, dict) and context.get("type") == "context":
            above = context.get("size")
        if isinstance(above, bool) or not isinstance(above, (int, float)) or above < 0:
            continue
        entry: Dict[str, Any] = {"input_tokens_above": above}
        for key in ("input", "output", "cache_read", "cache_write"):
            value = tier.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                value = tier.get("cacheRead" if key == "cache_read" else "cacheWrite" if key == "cache_write" else key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                value = (base or {}).get(key)
            if value is not None:
                entry[key] = value
        tiers.append(entry)
    return tiers or None


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
    # Per-thinking-level sampling parameters (port of Model.samplingParamsByThinkingLevel):
    # pi level -> request-body keys merged over ``params`` by OpenAI-compatible adapters.
    sampling_params_by_thinking_level: Optional[Dict[str, Dict[str, Any]]] = None
    # Chat Completions compatibility overrides, using the TS catalog keys.
    compat: Dict[str, object] = field(default_factory=dict)
    # Advertised prompt-cache TTLs per retention ("5m"/"1h" -> seconds); absent
    # metadata stops cache warming (the request intent alone is not proof).
    prompt_cache: Optional[Dict[str, int]] = None

    def calculate_cost(self, usage) -> Dict[str, float]:
        m = 1_000_000.0
        # Prompt-length tiers: the highest matching input threshold prices the
        # whole request (not just the part above it); the threshold counts
        # every input-side token the provider bills. Rates a tier does not
        # list keep the base price.
        input_side = usage.input + usage.cache_read + usage.cache_write + usage.cache_write_1h
        rates = {"input": self.costs.input, "output": self.costs.output,
                 "cache_read": self.costs.cache_read, "cache_write": self.costs.cache_write}
        matched = -1
        for tier in self.costs.tiers or []:
            if not isinstance(tier, dict):
                continue
            above = tier.get("input_tokens_above")
            if isinstance(above, bool) or not isinstance(above, (int, float)):
                continue
            if input_side > above and above > matched:
                for key in rates:
                    value = tier.get(key)
                    if not isinstance(value, bool) and isinstance(value, (int, float)):
                        rates[key] = value
                matched = above
        input_cost = usage.input * rates["input"] / m
        output_cost = usage.output * rates["output"] / m
        cache_read_cost = usage.cache_read * rates["cache_read"] / m
        cache_write_cost = usage.cache_write * rates["cache_write"] / m
        # Anthropic extended-TTL rule: 1h cache writes bill at 2x the effective
        # input rate unless the catalog carries a dedicated 1h price. Only
        # Anthropic reports cache_write_1h tokens, so the fallback is safe
        # elsewhere.
        write_1h_price = self.costs.cache_write_1h or 2 * rates["input"]
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


# Port of EXTENDED_THINKING_LEVELS / getSupportedThinkingLevels / clampThinkingLevel
# from packages/ai/src/models.ts.
THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def supported_thinking_levels(model: Model) -> List[str]:
    """Port of ``getSupportedThinkingLevels``: levels the model can actually run."""
    if not model.reasoning:
        return ["off"]
    mapping = model.thinking_level_map or {}
    levels: List[str] = []
    for level in THINKING_LEVELS:
        mapped = mapping.get(level)
        if mapped is None:
            # Explicitly mapped away, or (xhigh/max) never offered unless named.
            if level in mapping or level in ("xhigh", "max"):
                continue
        levels.append(level)
    return levels


def clamp_thinking_level(model: Model, level: Optional[str]) -> str:
    """Snap a requested thinking level to one the model actually supports.

    Non-reasoning models only support ``off``; reasoning models support every
    extended level except those mapped away by ``thinking_level_map``, and
    ``xhigh``/``max`` only when the map names them. A too-high request steps
    down to the nearest supported level.
    """
    requested = level if level in THINKING_LEVELS else "off"
    available = supported_thinking_levels(model)
    if requested in available:
        return requested
    index = THINKING_LEVELS.index(requested)
    for candidate in THINKING_LEVELS[index:] + list(reversed(THINKING_LEVELS[:index])):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def resolve_sampling_params(
    model: Model,
    thinking_level: Optional[str],
    request_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Merge the model's flat params, per-level params, and per-request params.

    Port of ``resolveSamplingParams`` (simple-options.ts): level params override
    the flat ``Model.params`` per key, per-request params override both. Returns
    None when no layer has anything to apply.
    """
    flat = model.params or None
    level_params = (model.sampling_params_by_thinking_level or {}).get(
        clamp_thinking_level(model, thinking_level if thinking_level else "off"))
    if flat or level_params or request_params:
        return {**(flat or {}), **(level_params or {}), **(request_params or {})}
    return None


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
    # Claude Haiku 5.5 (until catalogs include it). Prompts over 100k input
    # tokens are billed at 5x for the whole request; adaptive thinking with a
    # native xhigh/max effort replaces budget thinking.
    "anthropic/claude-haiku-5-5": {
        "name": "Claude Haiku 5.5",
        "reasoning": True,
        "thinking_level_map": {"off": None, "minimal": None, "low": "low", "medium": "medium",
                               "high": "high", "xhigh": "xhigh", "max": "max"},
        "context_window": 1_000_000,
        "max_tokens": 128_000,
        "costs": ModelCost(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125,
                           tiers=[{"input_tokens_above": 100_000, "input": 0.5, "output": 2.5,
                                   "cache_read": 0.05, "cache_write": 0.625}]),
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
        thinking_level_map=entry.get("thinking_level_map"),
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

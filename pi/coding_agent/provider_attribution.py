"""Provider attribution headers (port of provider-attribution.ts).

Some gateways identify the calling client through request headers. The
headers are merged into ``Model.headers`` when a model is resolved; explicit
models.json / provider headers keep precedence.
"""

import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

OPENROUTER_HOST = "openrouter.ai"
NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
CLOUDFLARE_API_HOST = "api.cloudflare.com"
CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
OPENCODE_HOST = "opencode.ai"


def matches_host(base_url: str, expected_host: str) -> bool:
    try:
        return urlparse(base_url).hostname == expected_host
    except ValueError:
        return False


def is_openrouter_model(model) -> bool:
    return model.provider == "openrouter" or matches_host(model.base_url, OPENROUTER_HOST)


def is_opencode_like(model) -> bool:
    """Providers that accept per-session attribution headers."""
    return (model.provider in ("opencode", "opencode-go")
            or matches_host(model.base_url, OPENCODE_HOST))


def is_nvidia_nim_model(model) -> bool:
    return model.provider == "nvidia" or matches_host(model.base_url, NVIDIA_NIM_HOST)


def is_cloudflare_model(model) -> bool:
    return (model.provider in ("cloudflare-workers-ai", "cloudflare-ai-gateway")
            or matches_host(model.base_url, CLOUDFLARE_API_HOST)
            or matches_host(model.base_url, CLOUDFLARE_AI_GATEWAY_HOST))


def is_install_telemetry_enabled(settings: Optional[dict] = None, env: Optional[str] = None) -> bool:
    """PI_TELEMETRY wins over the ``enable_install_telemetry`` setting (default True)."""
    value = os.environ.get("PI_TELEMETRY") if env is None else env
    if value is not None:
        return value in ("1", "true", "yes")
    return bool((settings or {}).get("enable_install_telemetry", True))


def default_attribution_headers(model, settings: Optional[dict] = None) -> Optional[Dict[str, str]]:
    if not is_install_telemetry_enabled(settings):
        return None
    if is_openrouter_model(model):
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }
    if is_nvidia_nim_model(model):
        return {"X-BILLING-INVOKE-ORIGIN": "Pi"}
    if is_cloudflare_model(model):
        return {"User-Agent": "pi-coding-agent"}
    return None


def session_headers(model, session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id:
        return None
    if not is_opencode_like(model):
        return None
    return {"x-opencode-session": session_id, "x-opencode-client": "pi"}


def merge_provider_attribution_headers(model, settings: Optional[dict] = None,
                                       session_id: Optional[str] = None,
                                       *header_sources: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    merged: Dict[str, str] = {}
    for source in (session_headers(model, session_id), default_attribution_headers(model, settings)):
        if source:
            merged.update(source)
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged or None


def apply_attribution(model, settings: Optional[dict] = None,
                      session_id: Optional[str] = None) -> List[str]:
    """Merge attribution headers into ``model.headers`` in place.

    Returns the header names that were added, so callers can report or remove
    them (the per-session header changes whenever the session changes).
    """
    headers = merge_provider_attribution_headers(model, settings, session_id, model.headers or {})
    added = []
    if headers:
        for name, value in headers.items():
            if (model.headers or {}).get(name) != value:
                added.append(name)
            model.headers[name] = value
    return added

"""Radius gateway model discovery.

Fetches the gateway model catalog from ``/v1/config`` (port of
``packages/ai/src/providers/radius-config.ts``), maps it onto registry model
entries, and caches the result in the agent directory. The cache overlays the
registry on reload; model ids the user configured in models.json still win.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .config import read_json, write_json
from .subscription_auth import DEFAULT_RADIUS_GATEWAY, normalize_radius_gateway

CACHE_FILE = "radius-cache.json"
CACHE_MAX_AGE = 7 * 24 * 3600

# /v1/config cost keys -> registry ModelCost fields (USD per million tokens).
_COST_KEYS = (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
              ("cache_write", "cacheWrite"), ("cache_write_1h", "cacheWrite1h"))


def cache_path(agent_dir: Path) -> Path:
    return agent_dir / CACHE_FILE


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value)


def _is_gateway_model(model: Any) -> bool:
    """Port of isRadiusGatewayModel: required fields with the right types."""
    if not isinstance(model, dict):
        return False
    return (isinstance(model.get("id"), str) and isinstance(model.get("name"), str)
            and isinstance(model.get("reasoning"), bool) and isinstance(model.get("input"), list)
            and isinstance(model.get("cost"), dict)
            and _positive_int(model.get("contextWindow")) is not None
            and _positive_int(model.get("maxTokens")) is not None)


def normalize_gateway_config(data: Any, gateway: str) -> Dict[str, Any]:
    """Map the /v1/config response onto registry model entries.

    Invalid entries are dropped; a response without a string ``baseUrl`` and a
    model list is rejected entirely.
    """
    if not isinstance(data, dict) or not isinstance(data.get("baseUrl"), str) or not isinstance(data.get("models"), list):
        raise ValueError(f"Invalid Radius config from {gateway}")
    models: List[dict] = []
    for raw in data["models"]:
        if not _is_gateway_model(raw):
            continue
        inputs = [value for value in raw["input"] if value in ("text", "image")]
        if "text" not in inputs:
            inputs = ["text"] + inputs
        entry: dict = {
            "id": raw["id"],
            "name": raw["name"],
            "reasoning": raw["reasoning"],
            "input": inputs,
            "context_window": int(raw["contextWindow"]),
            "max_tokens": int(raw["maxTokens"]),
            "base_url": data["baseUrl"],
        }
        costs = {key: raw["cost"][wire] for key, wire in _COST_KEYS
                 if isinstance(raw["cost"].get(wire), (int, float)) and not isinstance(raw["cost"].get(wire), bool)
                 and raw["cost"][wire] >= 0}
        if costs:
            entry["costs"] = costs
        level_map = raw.get("thinkingLevelMap")
        if isinstance(level_map, dict) and all(isinstance(key, str) for key in level_map):
            entry["thinking_level_map"] = level_map
        models.append(entry)
    return {"base_url": data["baseUrl"], "models": models}


def truncate_http_body(body: str) -> str:
    trimmed = body.strip()
    return trimmed if len(trimmed) <= 512 else trimmed[:512] + "…"


async def fetch_radius_gateway_config(gateway: str, api_key: Optional[str] = None,
                                      timeout: float = 20.0) -> Dict[str, Any]:
    """Fetch and normalize the gateway catalog from GET /v1/config."""
    gateway = normalize_radius_gateway(gateway)
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = "Bearer " + api_key
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(gateway + "/v1/config", headers=headers)
    if response.status_code >= 400:
        raise ValueError(f"Could not load Radius config from {gateway}: "
                         f"{response.status_code}: {truncate_http_body(response.text)}")
    try:
        data = response.json()
    except ValueError:
        data = None
    return normalize_gateway_config(data, gateway)


def save_cache(agent_dir: Path, gateway: str, config: Dict[str, Any]) -> None:
    write_json(cache_path(agent_dir), {"fetched_at": time.time(), "gateway": gateway,
                                       "base_url": config["base_url"], "models": config["models"]})


def load_cache(agent_dir: Path, max_age: float = CACHE_MAX_AGE) -> Optional[dict]:
    data = read_json(cache_path(agent_dir))
    fetched = data.get("fetched_at")
    if (not isinstance(fetched, (int, float)) or not isinstance(data.get("base_url"), str)
            or not isinstance(data.get("models"), list)):
        return None
    if time.time() - fetched > max_age:
        return None
    return data


def radius_overlay(agent_dir: Path, enabled: bool = True) -> dict:
    """Cached gateway catalog as a provider overlay for ModelRegistry.reload()."""
    if not enabled:
        return {}
    cached = load_cache(agent_dir)
    if not cached:
        return {}
    return {"radius": {"models": list(cached["models"])}}


async def refresh_radius_catalog(registry, timeout: float = 20.0) -> int:
    """Refresh the cached gateway catalog and reload the registry.

    The gateway is taken from the stored credential record, then RADIUS_GATEWAY,
    then the provider base_url. Credentials come from the stored Radius OAuth
    token (refreshed when expired) or RADIUS_API_KEY; unauthenticated requests
    are valid, like the TS implementation. Returns the discovered model count.
    """
    if "radius" not in registry.providers:
        return 0
    record = registry.auth.get("radius") or {}
    gateway = ((record.get("gateway") if isinstance(record, dict) else None)
               or os.environ.get("RADIUS_GATEWAY")
               or registry.providers["radius"].get("base_url")
               or DEFAULT_RADIUS_GATEWAY)
    api_key = await registry.auth.key("radius", None) or os.environ.get("RADIUS_API_KEY")
    config = await fetch_radius_gateway_config(gateway, api_key=api_key, timeout=timeout)
    save_cache(registry.config.agent_dir, gateway, config)
    registry.reload()
    return len(config["models"])

"""Online model catalog sync.

Fetches the public models.dev catalog, maps the providers this port supports
onto registry model entries, and caches the result in the agent directory.
The cache merges into the registry overlay on reload (explicit user
configuration in models.json still wins because it is applied afterwards).
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .config import read_json, write_json

MODELS_DEV_URL = "https://models.dev/api.json"
CACHE_FILE = "models-cache.json"
CACHE_MAX_AGE = 7 * 24 * 3600

# models.dev provider ids this port can serve, mapped to our api names.
PROVIDER_APIS: Dict[str, str] = {
    "anthropic": "anthropic-messages",
    "openai": "openai-completions",
    "google": "google-generative-ai",
    "google-vertex": "google-vertex",
    "azure-openai": "azure-openai",
    "amazon-bedrock": "bedrock-converse-stream",
    "openrouter": "openai-completions",
}


def cache_path(agent_dir: Path) -> Path:
    return agent_dir / CACHE_FILE


def load_cache(agent_dir: Path, max_age: float = CACHE_MAX_AGE) -> Optional[dict]:
    try:
        data = read_json(cache_path(agent_dir))
    except ValueError as exc:
        logging.getLogger(__name__).warning("%s", exc)
        return None
    fetched = data.get("fetched_at")
    providers = data.get("providers")
    if not isinstance(fetched, (int, float)) or not isinstance(providers, dict):
        return None
    if time.time() - fetched > max_age:
        return None
    return providers


async def fetch_catalog(timeout: float = 20.0) -> Dict[str, dict]:
    """Fetch and normalize the models.dev catalog for supported providers."""
    async with httpx.AsyncClient(timeout=timeout, headers={"user-agent": "pi-python"}) as client:
        response = await client.get(MODELS_DEV_URL)
    if response.status_code >= 400:
        raise ValueError(f"Model catalog fetch failed (HTTP {response.status_code})")
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Model catalog response is not an object")
    return normalize_catalog(data)


def normalize_catalog(data: dict) -> Dict[str, dict]:
    """Map models.dev providers to registry overlay entries."""
    providers: Dict[str, dict] = {}
    for name, api in PROVIDER_APIS.items():
        source = data.get(name)
        if not isinstance(source, dict) or not isinstance(source.get("models"), dict):
            continue
        entries: List[dict] = []
        for model_id, model in source["models"].items():
            if not isinstance(model, dict):
                continue
            limit = model.get("limit") or {}
            cost = model.get("cost") or {}
            context = limit.get("context")
            output = limit.get("output")
            if not isinstance(context, int) or context <= 0:
                continue
            inputs = model.get("input")
            inputs = [value for value in inputs if value in ("text", "image")] if isinstance(inputs, list) else ["text"]
            if "text" not in inputs:
                inputs = ["text"] + inputs
            entry: dict = {
                "id": model_id,
                "name": model.get("name") or model_id,
                "reasoning": bool(model.get("reasoning")),
                "context_window": context,
                "max_tokens": output if isinstance(output, int) and output > 0 else min(8192, context // 4),
                "input": inputs,
            }
            rates = {k: cost.get(k) for k in ("input", "output", "cache_read", "cache_write")}
            if all(isinstance(v, (int, float)) for v in rates.values()):
                entry["costs"] = rates
            entries.append(entry)
        if entries:
            providers[name] = {"models": entries}
    return providers


def save_cache(agent_dir: Path, providers: dict) -> None:
    write_json(cache_path(agent_dir), {"fetched_at": time.time(), "providers": providers})


def sync_overlay(agent_dir: Path, enabled: bool = True) -> dict:
    """Cached catalog as a provider overlay for ModelRegistry.reload()."""
    if not enabled:
        return {}
    providers = load_cache(agent_dir)
    if not providers:
        return {}
    return {name: {"models": entry["models"]} for name, entry in providers.items() if isinstance(entry, dict) and isinstance(entry.get("models"), list)}

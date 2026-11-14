"""Validated model catalog with runtime credential resolution."""

import os
import re
from dataclasses import replace
from typing import Dict, Optional
from urllib.parse import urlsplit

from ..ai.models import Model, ModelCost, ModelError, list_catalog, resolve_model
from .auth import AuthStore
from .catalog_sync import sync_overlay
from .config import Configuration, read_json
from .radius_catalog import radius_overlay
from .subscription_auth import COPILOT_HEADERS, copilot_endpoint


def expand(value: str) -> str:
    def substitute(match):
        name = match.group(1) or match.group(2)
        if name not in os.environ:
            raise ModelError(f"Missing environment variable: {name}")
        return os.environ[name]
    return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)", substitute, value)


def validate_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ModelError("Endpoint must be an HTTP(S) URL without embedded credentials")


class ModelRegistry:
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.auth = AuthStore(config.agent_dir / "auth.json")
        self.extension_providers: Dict[str, dict] = {}
        self.virtual_models: Dict[tuple, dict] = {}
        self.reload()

    def register_extension_providers(self, providers: Dict[str, dict]) -> None:
        """Merge extension-registered providers below user models.json settings."""
        self.extension_providers = dict(providers)
        self.reload()

    def register_virtual_models(self, models: Dict[tuple, dict]) -> None:
        self.virtual_models = dict(models)
        self.reload()

    def reload(self) -> None:
        data = read_json(self.config.agent_dir / "models.json")
        providers = data.get("providers", {})
        if not isinstance(providers, dict):
            raise ModelError("models.json providers must be an object")
        merged = {"openai-codex": {"api": "openai-codex-responses", "base_url": "https://chatgpt.com/backend-api", "models": [{"id": "gpt-5.1-codex", "reasoning": True, "input": ["text", "image"], "context_window": 400000}]},
            "github-copilot": {"api": "openai-completions", "api_key_env": "COPILOT_GITHUB_TOKEN", "base_url": "https://api.individual.githubcopilot.com", "models": [{"id": "gpt-4o", "input": ["text", "image"], "context_window": 128000}]},
            # Subscription sign-in for the "openai" provider is Sign in with ChatGPT; the access token is used as the API key.
            "kimi-coding": {"api": "anthropic-messages", "base_url": "https://api.kimi.com/coding", "api_key_env": "KIMI_API_KEY", "models": [{"id": "k3", "name": "Kimi K3", "reasoning": True, "input": ["text", "image"], "context_window": 1048576, "max_tokens": 131072}]},
            "xai": {"api": "openai-responses", "base_url": "https://api.x.ai/v1", "api_key_env": "XAI_API_KEY", "models": [{"id": "grok-4.3", "name": "Grok 4.3", "reasoning": True, "input": ["text", "image"], "context_window": 1000000, "max_tokens": 30000}]},
            "radius": {"api": "pi-messages", "base_url": "https://radius.pi.dev", "api_key_env": "RADIUS_API_KEY"},
            "azure-openai-responses": {"api": "azure-openai-responses", "api_key_env": "AZURE_OPENAI_API_KEY", "base_url": "",
                "models": [{"id": "gpt-4o", "input": ["text", "image"], "context_window": 128000}]},
            "openai-responses": {"api": "openai-responses", "api_key_env": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1", "models": [{"id": "gpt-4o-mini", "input": ["text", "image"], "context_window": 128000}]},
            "google": {"api": "google-generative-ai", "api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta", "models": [{"id": "gemini-2.5-flash", "reasoning": True, "input": ["text", "image"], "context_window": 1048576}]},
            "google-vertex": {"api": "google-vertex", "auth_required": False, "base_url": "",
                "models": [{"id": "gemini-2.5-flash", "reasoning": True, "input": ["text", "image"], "context_window": 1048576}]},
            "azure-openai": {"api": "azure-openai", "api_key_env": "AZURE_OPENAI_API_KEY", "base_url": "",
                "models": [{"id": "gpt-4o", "input": ["text", "image"], "context_window": 128000}]},
            "amazon-bedrock": {"api": "bedrock-converse-stream", "auth_required": False,
                "base_url": "https://bedrock-runtime." + (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1") + ".amazonaws.com",
                "models": [{"id": "anthropic.claude-3-5-sonnet-20241022-v2:0", "input": ["text", "image"], "context_window": 200000}]},
            "openai": {}, "anthropic": {}, "faux": {}, "openrouter": {
            "api": "openai-completions", "base_url": "https://openrouter.ai/api/v1",
            "models": [{"id": "openrouter/auto", "name": "OpenRouter Auto"}],
        }}
        for name, provider in self.extension_providers.items():
            if not isinstance(name, str) or not name or not isinstance(provider, dict):
                raise ModelError("Extension provider settings must be an object")
            merged[name] = {**merged.get(name, {}), **provider}
        # Cached online catalog fills in extra models for providers neither the
        # user nor an extension configured explicitly.
        for name, overlay in sync_overlay(self.config.agent_dir, self.config.settings.get("sync_catalog", True)).items():
            if name in providers or name in self.extension_providers:
                continue
            known = {entry.get("id") for entry in merged.get(name, {}).get("models", [])}
            extra = [entry for entry in overlay.get("models", []) if entry.get("id") not in known]
            if extra:
                merged[name] = {**merged.get(name, {}), "models": merged.get(name, {}).get("models", []) + extra}
        # Discovered Radius gateway models overlay the provider (including a user
        # models.json configuration) with real capability and cost data; ids the
        # user declared keep their explicit settings.
        for name, overlay in radius_overlay(self.config.agent_dir, self.config.settings.get("sync_catalog", True)).items():
            entry = merged.get(name, {})
            known = {model.get("id") for model in entry.get("models", [])}
            extra = [model for model in overlay["models"] if isinstance(model, dict) and model.get("id") not in known]
            if extra:
                merged[name] = {**entry, "models": entry.get("models", []) + extra}
        for name, provider in providers.items():
            if not isinstance(name, str) or not name or not isinstance(provider, dict):
                raise ModelError("Each provider must be an object")
            unknown = set(provider) - {"api", "base_url", "api_key", "api_key_env", "headers", "models", "oauth", "auth_required"}
            if unknown:
                raise ModelError(f"Unknown provider settings: {', '.join(sorted(unknown))}")
            entry = merged.get(name, {})
            if name == "radius" and isinstance(provider.get("models"), list):
                user_ids = {model.get("id") for model in provider["models"] if isinstance(model, dict)}
                discovered = [model for model in entry.get("models", []) if model.get("id") not in user_ids]
                merged[name] = {**entry, **provider, "models": discovered + provider["models"]}
            else:
                merged[name] = {**entry, **provider}
        models = {}
        for identifier in list_catalog():
            provider, model_id = identifier.split("/", 1)
            models[identifier] = resolve_model(provider, model_id, api_key="catalog-only")
        models["faux/faux"] = resolve_model("faux")
        for name, provider in merged.items():
            stream_simple = provider.get("stream_simple")
            if stream_simple is not None and not callable(stream_simple):
                raise ModelError("stream_simple must be callable")
            api = provider.get("api", "anthropic-messages" if name == "anthropic" else "faux" if name == "faux" else "openai-completions")
            # A provider with a custom stream function implements its own protocol,
            # so it may name its api freely and keep its endpoint empty.
            if stream_simple is None and api not in ("pi-messages", "azure-openai-responses", "anthropic-messages", "openai-completions", "faux", "openai-responses", "openai-codex-responses", "google-generative-ai", "google-vertex", "azure-openai", "bedrock-converse-stream"):
                raise ModelError(f"Unsupported API: {api}")
            # Providers may leave the endpoint empty: Vertex and Azure build it from
            # the environment at request time, and catalog models carry their own
            # base_url. Validate only a URL that is actually declared.
            if provider.get("base_url"):
                validate_url(provider["base_url"])
            for key in ("base_url", "api_key", "api_key_env"):
                if key in provider and not isinstance(provider[key], str):
                    raise ModelError(f"{key} must be a string")
            if not isinstance(provider.get("auth_required", True), bool):
                raise ModelError("auth_required must be boolean")
            headers = provider.get("headers", {})
            if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
                raise ModelError("headers must contain string values")
            oauth = provider.get("oauth")
            if oauth is not None:
                if not isinstance(oauth, dict) or not all(isinstance(oauth.get(k), str) and oauth[k] for k in ("device_url", "token_url", "client_id")):
                    raise ModelError("OAuth requires device_url, token_url and client_id")
                for key in ("device_url", "token_url"):
                    validate_url(oauth[key])
                    if urlsplit(oauth[key]).scheme != "https":
                        raise ModelError("OAuth endpoints must use HTTPS")
            entries = provider.get("models", [])
            if not isinstance(entries, list):
                raise ModelError("models must be a list")
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
                    raise ModelError("Model requires a nonempty id")
                if set(entry) - {"id", "name", "reasoning", "context_window", "max_tokens", "input", "costs", "params", "base_url", "thinking_level_map", "compat"}:
                    raise ModelError("Unknown model capability field")
                compat = entry.get("compat", {})
                if (not isinstance(compat, dict)
                        or set(compat) - {"maxTokensField", "supportsUsageInStreaming"}
                        or ("maxTokensField" in compat and compat["maxTokensField"] not in ("max_tokens", "max_completion_tokens"))
                        or ("supportsUsageInStreaming" in compat and not isinstance(compat["supportsUsageInStreaming"], bool))):
                    raise ModelError("Invalid model compat settings")
                if "params" in entry and (not isinstance(entry["params"], dict) or not all(isinstance(k, str) for k in entry["params"])):
                    raise ModelError("Model params must be an object with string keys")
                if "base_url" in entry and (not isinstance(entry["base_url"], str) or not entry["base_url"]):
                    raise ModelError("Model base_url must be a nonempty string")
                if "thinking_level_map" in entry and (not isinstance(entry["thinking_level_map"], dict)
                        or not all(isinstance(k, str) for k in entry["thinking_level_map"])):
                    raise ModelError("Model thinking_level_map must be an object with string keys")
                identifier = name + "/" + entry["id"]
                base = models.get(identifier) or Model(id=entry["id"], name=entry["id"], api=api, provider=name, base_url="")
                changes = {k: v for k, v in entry.items() if k not in ("id", "costs")}
                if "costs" in entry:
                    costs = entry["costs"]
                    if not isinstance(costs, dict) or set(costs) - set(ModelCost.__dataclass_fields__) or not all(type(v) in (int, float) and v >= 0 for v in costs.values()):
                        raise ModelError("Invalid model costs")
                    changes["costs"] = ModelCost(**costs)
                model = replace(base, **changes)
                if any(type(getattr(model, field)) is not int or getattr(model, field) <= 0 for field in ("context_window", "max_tokens")):
                    raise ModelError("Context window and output limit must be positive integers")
                if not isinstance(model.reasoning, bool) or not isinstance(model.name, str):
                    raise ModelError("Invalid model name or reasoning capability")
                if not isinstance(model.input, list) or "text" not in model.input or not all(isinstance(value, str) and value in ("text", "image") for value in model.input):
                    raise ModelError("Model input must contain text and optionally image")
                models[identifier] = model
            for identifier, model in list(models.items()):
                if model.provider != name:
                    continue
                # pi-messages models keep a catalog-provided base_url (e.g. the
                # baseUrl a Radius gateway advertises) over the provider default.
                if api == "pi-messages" and model.base_url:
                    base_url = model.base_url
                else:
                    base_url = provider.get("base_url", model.base_url)
                if stream_simple is None and (api not in ("faux", "google-vertex", "azure-openai", "azure-openai-responses") or base_url):
                    validate_url(base_url)
                models[identifier] = replace(model, api=api, base_url=base_url.rstrip("/"), api_key=None)
        for (provider, model_id), definition in self.virtual_models.items():
            merged.setdefault(provider, {"auth_required": False})
            levels = definition.get("thinkingLevels") or ["off"]
            models[provider + "/" + model_id] = Model(
                id=model_id, name=definition.get("name") or model_id, api="pi-virtual",
                provider=provider, base_url="", reasoning=any(level != "off" for level in levels),
                context_window=definition.get("contextWindow") or 1,
                max_tokens=definition.get("maxTokens") or 1,
                input=list(definition.get("input") or ["text", "image"]),
            )
        self.providers = merged
        self.models = models

    def resolve(self, provider: str, model_id: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None) -> Model:
        if provider not in self.providers:
            raise ModelError(f"Unknown provider: {provider}")
        entries = [m for m in self.models.values() if m.provider == provider]
        if model_id is None:
            model_id = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5-20250929", "faux": "faux"}.get(provider)
            if model_id is None and entries:
                model_id = entries[0].id
        if model_id is None:
            raise ModelError(f"No model configured for {provider}")
        model = self.models.get(provider + "/" + model_id)
        settings = self.providers[provider]
        if model is None:
            if provider in ("openai", "anthropic", "faux"):
                model = resolve_model(provider, model_id, api_key="catalog-only")
            elif settings.get("api") == "pi-messages":
                model = Model(model_id, model_id, "pi-messages", provider, settings["base_url"])
            elif provider in ("google", "amazon-bedrock", "openai-responses", "openai-codex", "github-copilot", "google-vertex", "azure-openai", "azure-openai-responses"):
                model = replace(entries[0], id=model_id, name=model_id)
            else:
                raise ModelError(f"Unknown model: {provider}/{model_id}")
        endpoint = base_url or settings.get("base_url", model.base_url)
        # pi-messages models keep their catalog-provided endpoint (e.g. the
        # baseUrl a Radius gateway advertises) unless the caller overrides it.
        if base_url is None and settings.get("api", model.api) == "pi-messages" and model.base_url:
            endpoint = model.base_url
        if settings.get("api", model.api) == "azure-openai" and not endpoint:
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if settings.get("api", model.api) == "azure-openai-responses":
            endpoint = endpoint or os.environ.get("AZURE_OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            if not endpoint and os.environ.get("AZURE_OPENAI_RESOURCE_NAME"):
                endpoint = "https://" + os.environ["AZURE_OPENAI_RESOURCE_NAME"] + ".openai.azure.com"
            if not endpoint:
                raise ModelError("Azure Responses requires an endpoint or AZURE_OPENAI_RESOURCE_NAME")
            parsed = urlsplit(endpoint)
            if parsed.path.rstrip("/") in ("", "/openai", "/openai/v1/responses"):
                endpoint = parsed._replace(path="/openai/v1", query="").geturl()
        if model.api != "faux" and endpoint:
            validate_url(endpoint)
        if settings.get("api", model.api) == "azure-openai" and not endpoint:
            raise ModelError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT or a models.json base_url")

        resolved = replace(model, api=settings.get("api", model.api), base_url=endpoint.rstrip("/"), api_key=api_key,
                       headers={k: expand(v) for k, v in settings.get("headers", {}).items()})
        baseline_headers = dict(resolved.headers)

        async def credentials(_provider):
            resolved.base_url = endpoint.rstrip("/")
            resolved.headers = dict(baseline_headers)
            resolved.auth_type = "api_key"
            if api_key:
                return api_key
            key = await self.auth.key(provider, settings.get("oauth"))
            if key:
                record = self.auth.get(provider)
                resolved.auth_type = record.get("type", "api_key")
                if provider == "openai-codex":
                    resolved.headers.update({"chatgpt-account-id": record["account_id"], "originator": "pi"})
                elif provider == "github-copilot":
                    resolved.base_url = copilot_endpoint(key)
                    resolved.headers.update({**COPILOT_HEADERS, "x-github-api-version": "2026-06-01", "openai-intent": "conversation-panel", "X-Initiator": "user"})
                return key
            if resolved.api in ("azure-openai", "azure-openai-responses"):
                token = os.environ.get("AZURE_ACCESS_TOKEN", "")
                if token:
                    resolved.auth_type = "entra"
                    return token
            if settings.get("api_key"):
                return expand(settings["api_key"])
            env_name = settings.get("api_key_env", provider.upper().replace("-", "_") + "_API_KEY")
            key = os.environ.get("PI_PY_API_KEY") or os.environ.get(env_name)
            if not key and settings.get("auth_required", True) and model.api != "faux":
                raise ModelError(f"No credentials for {provider}; use --login or {env_name}")
            return key or ""

        resolved.get_api_key = credentials
        return resolved

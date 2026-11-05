"""Provider/model configuration probes for evals.

Port of ``packages/evals/evals/configured-runtime.ts``: after the agent has
been prompted to configure a provider or model and the harness reloaded the
agent directory, these helpers inspect the refreshed model configuration and
run one real completion against a fixture provider.

Structured outputs keep the TypeScript camelCase field names because they are
compared verbatim by the structured-output judge; errors are structured
results (``{"error": ...}``) rather than raised, so the judge scores them like
any other response.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pi.ai.models import Model
from pi.ai.providers import get_default_stream_fn
from pi.ai.providers.base import SimpleStreamOptions
from pi.ai.types import DoneEvent, StreamErrorEvent, TextContent, TranscriptContext
from pi.coding_agent.config import Configuration
from pi.coding_agent.model_registry import ModelError, ModelRegistry


@dataclass
class ProviderScenario:
    """One probe: provider/model ids, a context factory, and the probe flag."""

    provider_id: str
    model_id: str
    create_context: Callable[[], TranscriptContext]
    options: Optional[SimpleStreamOptions] = None
    valid_request_received: Callable[[], bool] = field(default=lambda: False)


class ConfiguredModelRuntime:
    """ModelRegistry over one agent directory, tolerating load failures.

    ``ModelRegistry`` reloads eagerly in its constructor, while the TypeScript
    ``ModelRuntime`` records a malformed ``models.json`` and exposes it through
    ``getError()``. This wrapper keeps the failure and replays it on the next
    reload, so probes can observe a broken configuration as a structured error.
    """

    def __init__(self, agent_dir: str, cwd: Optional[str] = None) -> None:
        directory = Path(agent_dir)
        self._configuration = Configuration(str(cwd or directory), str(directory))
        self._registry: Optional[ModelRegistry] = None

    def reload(self) -> None:
        """Re-read the agent directory; raises when models.json is invalid."""
        if self._registry is None:
            self._registry = ModelRegistry(self._configuration)
        else:
            self._registry.reload()

    @property
    def providers(self) -> Dict[str, dict]:
        return self._registry.providers if self._registry else {}

    @property
    def models(self) -> Dict[str, Model]:
        return self._registry.models if self._registry else {}

    def resolve(self, provider: str, model_id: Optional[str] = None) -> Model:
        if self._registry is None:
            raise ModelError("Model configuration has not loaded successfully.")
        return self._registry.resolve(provider, model_id)


def load_configured_model_runtime(agent_dir: str, cwd: Optional[str] = None) -> ConfiguredModelRuntime:
    """Registry over an isolated agent directory, like TS loadConfiguredModelRuntime.

    The agent directory is exactly where the agent was asked to write
    ``models.json``; the catalog overlay reads only a local cache file, so
    loading never touches the network.
    """
    return ConfiguredModelRuntime(agent_dir, cwd)


def model_fields(model: Model) -> Dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "provider": model.provider,
        "reasoning": model.reasoning,
        "input": list(model.input),
        "cost": {
            "input": model.costs.input,
            "output": model.costs.output,
            "cacheRead": model.costs.cache_read,
            "cacheWrite": model.costs.cache_write,
        },
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
    }


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block.text for block in content if isinstance(block, TextContent))


async def _complete_simple(model: Model, context: TranscriptContext,
                           options: Optional[SimpleStreamOptions]) -> Any:
    """Final assistant message from one provider stream (TS completeSimple).

    Stream-level errors come back as a message with ``stop_reason="error"``
    instead of raising, mirroring the TypeScript completion contract.
    """
    final = None
    async for event in get_default_stream_fn()(model, context, options):
        if isinstance(event, DoneEvent):
            final = event.message
            # The Python adapters carry the stop reason on the event only.
            final.stop_reason = event.reason
        elif isinstance(event, StreamErrorEvent):
            final = event.error
    if final is None:
        raise RuntimeError("Provider stream ended without a final message.")
    return final


def inspect_provider(runtime: ConfiguredModelRuntime, scenario: ProviderScenario) -> Dict[str, Any]:
    """Reload offline, resolve the model, and complete once.

    Returns a coroutine; await it (or run under ``asyncio.run``).
    """

    async def probe() -> Dict[str, Any]:
        try:
            runtime.reload()
        except (ModelError, ValueError) as error:
            return {"result": {"error": str(error)}}
        identifier = f"{scenario.provider_id}/{scenario.model_id}"
        if scenario.provider_id not in runtime.providers or identifier not in runtime.models:
            return {"result": {"error": f"Model {identifier} is unavailable after reload."}}
        try:
            model = runtime.resolve(scenario.provider_id, scenario.model_id)
            api_key = await model.get_api_key(model.provider)
            options = replace(scenario.options or SimpleStreamOptions(), api_key=api_key or None)
            response = await _complete_simple(model, scenario.create_context(), options)
        except Exception as error:  # noqa: BLE001 - structured errors are the contract
            return {"result": {"error": str(error)}}
        return {"result": {
            "validRequestReceived": bool(scenario.valid_request_received()),
            "model": model_fields(model),
            "response": {
                "text": _content_text(response.content),
                "stopReason": response.stop_reason,
                "inputTokens": response.usage.input,
                "outputTokens": response.usage.output,
            },
        }}

    return probe()


def inspect_added_model(runtime: ConfiguredModelRuntime, provider_id: str, model_id: str) -> Dict[str, Any]:
    """Resolve an added model and verify the built-in catalog was preserved.

    Returns a coroutine; await it (or run under ``asyncio.run``).
    """

    async def probe() -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="pi-eval-pristine-") as pristine_dir:
            pristine = load_configured_model_runtime(pristine_dir)
            pristine.reload()
            existing_model_ids = [model.id for model in pristine.models.values()
                                  if model.provider == provider_id]
        if provider_id not in pristine.providers:
            return {"result": {"error": f"Built-in provider {provider_id} is unavailable."}}
        if not existing_model_ids:
            return {"result": {"error": f"Built-in provider {provider_id} has no models."}}
        try:
            runtime.reload()
        except (ModelError, ValueError) as error:
            return {"result": {"error": str(error)}}
        identifier = f"{provider_id}/{model_id}"
        if provider_id not in runtime.providers or identifier not in runtime.models:
            return {"result": {"error": f"Model {identifier} is unavailable after reload."}}
        model = runtime.resolve(provider_id, model_id)
        return {"result": {
            "model": model_fields(model),
            "existingModelsPreserved": all(f"{provider_id}/{existing}" in runtime.models
                                           for existing in existing_model_ids),
        }}

    return probe()

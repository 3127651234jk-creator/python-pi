"""Image generation over OpenRouter's chat completions endpoint.

Port of ``packages/ai/src/api/openrouter-images.ts``: one non-streaming chat
completion with ``modalities`` requesting image output; generated images come
back as base64 data URLs on ``message.images``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from ..models import Model
from ..types import AssistantImages, ImageContent, ImagesContext, ImagesOptions, TextContent, Usage
from ...util import AbortError


async def generate_images_openrouter(
    model: Model, context: ImagesContext, options: Optional[ImagesOptions] = None
) -> AssistantImages:
    """Generate images; errors are encoded on the result, never raised."""
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        timestamp=int(time.time() * 1000),
    )
    resolved = options or ImagesOptions()
    signal = resolved.signal
    try:
        if signal is not None:
            signal.throw_if_aborted()
        api_key = resolved.api_key
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")
        headers = {**model.headers, **resolved.headers, "Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(resolved.timeout_ms / 1000 if resolved.timeout_ms else 600.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                model.base_url + "/chat/completions",
                json=build_params(model, context),
                headers=headers,
            )
        if response.status_code >= 400:
            body = response.text[:2000]
            raise ValueError(f"HTTP {response.status_code} from {model.provider}: {body}")
        parse_image_response(response.json(), model, output)
        return output
    except AbortError:
        output.stop_reason = "aborted"
        return output
    except Exception as exc:  # noqa: BLE001 - encoded on the result, like the TS contract
        output.stop_reason = "aborted" if signal is not None and signal.aborted else "error"
        output.error_message = str(exc)
        return output


def build_params(model: Model, context: ImagesContext) -> Dict[str, Any]:
    """Build the chat completion payload; ``modalities`` follows model.output."""
    content = []
    for item in context.input:
        if isinstance(item, TextContent):
            content.append({"type": "text", "text": _sanitize_surrogates(item.text)})
        else:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
            )
    return {
        "model": model.id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["image", "text"] if model.output and "text" in model.output else ["image"],
    }


def parse_image_response(payload: Dict[str, Any], model: Model, output: AssistantImages) -> None:
    """Fill an AssistantImages result from a chat completion payload."""
    output.response_id = payload.get("id")
    if payload.get("usage"):
        output.usage = _parse_usage(payload["usage"], model)
    choices = payload.get("choices") or []
    if not choices:
        return
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content:
        output.output.append(TextContent(text=content))
    for image in message.get("images") or []:
        image_url = image.get("image_url") if isinstance(image, dict) else None
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str) or not image_url.startswith("data:"):
            continue
        header, separator, data = image_url.partition(",")
        if not separator or not header.endswith(";base64"):
            continue
        mime_type = header[len("data:") : -len(";base64")]
        if not mime_type or not data:
            continue
        output.output.append(ImageContent(data=data, mime_type=mime_type))


def _parse_usage(raw: Dict[str, Any], model: Model) -> Usage:
    details = raw.get("prompt_tokens_details") or {}
    prompt_tokens = raw.get("prompt_tokens") or 0
    reported_cached = details.get("cached_tokens") or 0
    cache_write = details.get("cache_write_tokens") or 0
    cache_read = max(0, reported_cached - cache_write) if cache_write > 0 else reported_cached
    usage = Usage(
        input=max(0, prompt_tokens - cache_read - cache_write),
        output=raw.get("completion_tokens") or 0,
        cache_read=cache_read,
        cache_write=cache_write,
    )
    usage.cost = model.calculate_cost(usage)
    return usage


def _sanitize_surrogates(text: str) -> str:
    """Replace unpaired surrogates so JSON encoding cannot fail."""
    return text.encode("utf-8", "replace").decode("utf-8")

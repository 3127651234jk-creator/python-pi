"""Tool path resolution (port of ``tools/path-utils.ts``)."""

from __future__ import annotations

import re
import unicodedata
from typing import List

from ..env import Context, ExecutionEnv, get_or_throw

# U+00A0 NBSP, U+2000-U+200A various spaces, U+202F narrow NBSP, U+205F medium space, U+3000 ideographic space.
UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
NARROW_NO_BREAK_SPACE = "\u202f"


def _normalize_tool_path(path: str) -> str:
    normalized = UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    return get_or_throw(await env.absolute_path(_normalize_tool_path(path), context))


async def resolve_read_tool_path(env: ExecutionEnv, path: str, context: Context) -> str:
    """Resolve like ``resolve_tool_path``, then try spelling variants the file system may hold."""
    resolved = await resolve_tool_path(env, path, context)
    variants: List[str] = [
        resolved,
        re.sub(r" (AM|PM)\.", NARROW_NO_BREAK_SPACE + r"\1.", resolved, flags=re.IGNORECASE),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "\u2019"),
        unicodedata.normalize("NFD", resolved).replace("'", "\u2019"),
    ]

    seen = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        if get_or_throw(await env.exists(variant, context)):
            return variant
    return resolved

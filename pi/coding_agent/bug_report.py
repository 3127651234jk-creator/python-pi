"""Bug report bundle: metadata, diagnostics, summary, zip export and upload.

Port of ``bug-report.ts``, ``bug-report-upload.ts`` and the ``/bug`` flow from
``modes/interactive/bug-report.ts``. The bundle never includes conversation
content unless the user explicitly shares the session transcript; everything
else is metadata with credentials redacted.
"""

import json
import os
import platform
import re
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from ..ai.providers import get_default_stream_fn
from ..ai.providers.base import SimpleStreamOptions
from ..ai.types import (AssistantMessage, SystemMessage, TextContent, TranscriptContext,
                        UserMessage, get_message_text)
from ..ai.utils import normalize_context
from .crash_log import pi_version
from .subscription_auth import DEFAULT_RADIUS_GATEWAY

BUG_REPORT_CUSTOM_ENTRY_TYPE = "pi.bug-report"
BUG_REPORT_SCHEMA_VERSION = 1
REDACTED = "<redacted>"
SENSITIVE_KEY = re.compile(r"(?:^|[-_])(api[-_]?key|secret|token|password|passwd|credential|authorization|cookie)(?:$|[-_])", re.IGNORECASE)

BUG_SUMMARY_SYSTEM_PROMPT = (
    "You are helping a user file a bug report about pi, the coding agent they are talking to. "
    "You will be shown the conversation transcript. Write a report for the pi developers describing "
    "what the user was doing and what went wrong.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. "
    "ONLY output the report."
)
BUG_SUMMARY_INSTRUCTIONS = """Write the bug report in Markdown with these sections:

## What the user was doing
One short paragraph.

## What went wrong
Concrete description of the failure: wrong output, errors, hangs, tool failures, unexpected behavior. Quote error messages and tool output verbatim where they exist.

## Steps to reproduce
Numbered list, as specific as the transcript allows.

## Relevant details
Tool calls involved, files touched, model behavior, anything else that helps a developer reproduce or locate the problem.

Do not include file contents, secrets, or credentials from the transcript; refer to files by path only. Keep the report factual and concise."""


def _camel_to_snake(key: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)


def is_sensitive_key(key: str) -> bool:
    return bool(SENSITIVE_KEY.search(_camel_to_snake(key)))


def redact_url(value: str) -> str:
    """Strip credentials and secret-looking query parameters from a URL."""
    parts = urlsplit(value)
    changed = False
    if parts.username or parts.password:
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        parts = parts._replace(netloc=netloc)
        changed = True
    if parts.query:
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        if any(is_sensitive_key(key) for key, _ in pairs):
            parts = parts._replace(query=urlencode(
                [(key, REDACTED if is_sensitive_key(key) else pair_value) for key, pair_value in pairs]))
            changed = True
    return urlunsplit(parts) if changed else value


def _redact(key: str, value: Any) -> Any:
    if value is not None and is_sensitive_key(key):
        return REDACTED
    if isinstance(value, str):
        return redact_url(value)
    return value


def redact_json_value(value: Any) -> Any:
    """Copy a JSON value, redacting sensitive keys and URLs in strings."""
    if isinstance(value, dict):
        return {key: _redact(key, redact_json_value(child)) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_json_value(child) for child in value]
    if isinstance(value, str):
        return redact_url(value)
    return value


def collect_environment() -> dict:
    def env(name: str) -> Optional[str]:
        return os.environ.get(name) or None

    shell = env("SHELL") or env("COMSPEC")
    return {
        "version": pi_version(),
        "runtime": f"python/{sys.version.split()[0]}",
        "platform": sys.platform,
        "machine": platform.machine(),
        "osRelease": platform.release() or None,
        "shell": shell.replace("\\", "/").split("/")[-1] if shell else None,
        "terminal": {
            "term": env("TERM"),
            "program": env("TERM_PROGRAM"),
            "programVersion": env("TERM_PROGRAM_VERSION"),
            "colorterm": env("COLORTERM"),
            "tmux": bool(os.environ.get("TMUX")),
            "ssh": bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY")),
            "ci": bool(os.environ.get("CI")),
        },
        # Names help diagnose configuration; values never leave the machine.
        "piEnvironmentVariables": sorted(name for name in os.environ if name.startswith("PI_PY_")),
    }


def describe_model(model) -> Optional[dict]:
    if model is None:
        return None
    return {
        "provider": model.provider,
        "id": model.id,
        "name": model.name,
        "api": model.api,
        "baseUrl": redact_url(model.base_url),
        "reasoning": model.reasoning,
        "input": list(model.input),
        "contextWindow": model.context_window,
        "maxTokens": model.max_tokens,
        "samplingParams": redact_json_value(model.params) if getattr(model, "params", None) else None,
        "headerNames": sorted((model.headers or {}).keys()),
    }


def collect_bug_report_metadata(hint: Optional[str], session_id: str, cwd: str, include_session: bool,
                                include_summary: bool, message_count: int, model, thinking_level: str,
                                extensions: List[str], extension_errors: List[dict],
                                global_settings: dict, project_settings: dict,
                                id: Optional[str] = None) -> dict:
    return {
        "schemaVersion": BUG_REPORT_SCHEMA_VERSION,
        "id": id or f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{os.getpid()}",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "hint": (hint or "").strip() or None,
        "environment": collect_environment(),
        "session": {
            "id": session_id,
            "included": include_session,
            "summaryIncluded": include_summary,
            "messageCount": message_count,
            **({"cwd": cwd} if include_session else {}),
        },
        "model": describe_model(model),
        "thinkingLevel": thinking_level,
        "extensions": [str(path) for path in extensions],
        "extensionErrors": extension_errors,
        "settings": {
            "global": redact_json_value({key: value for key, value in global_settings.items()
                                         if key not in ("tracking_id", "device_id")}),
            "project": redact_json_value({key: value for key, value in project_settings.items()
                                          if key not in ("tracking_id", "device_id")}),
        },
    }


def collect_bug_report_diagnostics(session, crashes: Optional[List[dict]] = None) -> dict:
    """Failed assistant turns without collecting conversation content."""
    entries = session.get_branch() if session is not None else []
    assistant = []
    assistant_message_count = 0
    for entry in entries:
        message = entry.message
        if not isinstance(message, AssistantMessage):
            continue
        assistant_message_count += 1
        if message.stop_reason not in ("error", "aborted") and not message.error_message:
            continue
        assistant.append({
            "entryId": entry.id,
            "timestamp": message.timestamp,
            "provider": message.provider,
            "model": message.model,
            "api": message.api,
            "stopReason": message.stop_reason,
            **({"rawStopReason": message.raw_stop_reason} if getattr(message, "raw_stop_reason", None) else {}),
            **({"errorMessage": message.error_message} if message.error_message else {}),
        })
    return {
        "schemaVersion": BUG_REPORT_SCHEMA_VERSION,
        "sessionId": session.session_file if session is not None else None,
        "entryCount": len(entries),
        "assistantMessageCount": assistant_message_count,
        "assistant": assistant,
        "crashes": [{key: value for key, value in record.items() if key != "notified"}
                    for record in (crashes or [])],
    }


def bug_report_files(bundle: dict) -> List[Tuple[str, str, str]]:
    """(name, content type, data) files shared by upload and zip export."""
    files = [
        ("report.json", "application/json", json.dumps(bundle["metadata"], ensure_ascii=False, indent=2) + "\n"),
        ("diagnostics.json", "application/json", json.dumps(bundle["diagnostics"], ensure_ascii=False, indent=2) + "\n"),
    ]
    if bundle.get("sessionJsonl") is not None:
        files.append(("session.jsonl", "application/x-ndjson", bundle["sessionJsonl"]))
    if bundle.get("summary") is not None:
        summary = bundle["summary"]
        files.append(("summary.md", "text/markdown", summary if summary.endswith("\n") else summary + "\n"))
    return files


def bug_report_archive_file_name(report_id: str) -> str:
    return f"pi-bug-report-{report_id}.zip"


def write_bug_report_archive(bundle: dict, file_path: str) -> str:
    with zipfile.ZipFile(file_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, _content_type, data in bug_report_files(bundle):
            archive.writestr(name, data)
    return file_path


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _select_messages(messages: List[Any], token_budget: int) -> List[Any]:
    selected: List[Any] = []
    tokens = 0
    for message in reversed(messages):
        cost = _estimate_tokens(get_message_text(message.content) if hasattr(message, "content") else "")
        if selected and tokens + cost > token_budget:
            break
        selected.append(message)
        tokens += cost
    return list(reversed(selected))


async def generate_bug_report_summary(messages: List[Any], model, hint: Optional[str] = None,
                                      thinking_level: Optional[str] = None) -> str:
    """Ask the session model for a report when the user does not share the transcript."""
    context_window = model.context_window if model.context_window > 0 else 128_000
    selected = _select_messages(messages, context_window * 3 // 5)
    conversation = "\n\n".join(
        f"## {getattr(message, 'role', 'user')}\n\n{get_message_text(message.content)}"
        for message in selected)
    parts = [
        f"Note: only the last {len(selected)} of {len(messages)} messages are shown." if len(selected) < len(messages) else None,
        f"<conversation>\n{conversation}\n</conversation>",
        f"<user-report>\n{hint.strip()}\n</user-report>" if hint and hint.strip() else None,
        BUG_SUMMARY_INSTRUCTIONS,
    ]
    prompt = "\n\n".join(part for part in parts if part)
    options = SimpleStreamOptions(
        api_key=getattr(model, "api_key", None),
        max_tokens=min(4096, model.max_tokens if model.max_tokens > 0 else 4096),
        reasoning=thinking_level if model.reasoning and thinking_level not in (None, "off") else None,
    )
    system, normalized = normalize_context(TranscriptContext(messages=[
        SystemMessage(content=BUG_SUMMARY_SYSTEM_PROMPT, timestamp=0),
        UserMessage(content=[TextContent(text=prompt)], timestamp=0),
    ]))
    context = TranscriptContext(messages=([system, *normalized] if system is not None else normalized))
    final: Optional[AssistantMessage] = None
    error: Optional[str] = None
    async for event in get_default_stream_fn()(model, context, options):
        reason = getattr(event, "reason", None)
        if reason is not None:
            message = getattr(event, "message", None) or getattr(event, "error", None)
            if reason == "error" or getattr(event, "error", None) is not None:
                error = getattr(message, "error_message", None) or "Bug report summary failed"
            elif message is not None:
                final = message
    if error:
        raise ValueError(error)
    if final is None:
        raise ValueError("Bug report summary failed")
    if final.stop_reason == "aborted":
        raise ValueError("Bug report summary was cancelled")
    if any(getattr(block, "type", None) == "toolCall" for block in final.content):
        raise ValueError("Bug report summary attempted to call a tool")
    text = "".join(block.text for block in final.content if isinstance(block, TextContent)).strip()
    if not text:
        raise ValueError("Bug report summary was empty")
    return text


async def upload_bug_report(bundle: dict, token: Optional[str] = None,
                            gateway_url: Optional[str] = None) -> str:
    """Upload the report as multipart form data; returns the report id."""
    url = (gateway_url or DEFAULT_RADIUS_GATEWAY).rstrip("/") + "/v1/bug-reports"
    files = []
    for name, content_type, data in bug_report_files(bundle):
        files.append(("files", (name, data.encode("utf-8"), content_type)))
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, files=files, headers=headers)
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code < 400 and isinstance(body, dict) and body.get("ok") is True \
            and isinstance(body.get("bug_report", {}).get("id"), str):
        return body["bug_report"]["id"]
    detail = body.get("description") or body.get("error") if isinstance(body, dict) else None
    raise ValueError(f"Bug report upload failed: {detail or response.reason_phrase or response.status_code}")

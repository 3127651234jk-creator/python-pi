"""In-process fake LLM provider server for provider/model evals.

Port of ``packages/evals/evals/acme-server.ts``. "Acme" stands in for any
third-party provider; the server simulates two API shapes and is used by
provider-class evals and their unit tests:

- ``"openai"`` mode: OpenAI Chat Completions SSE at ``POST /v1/chat/completions``.
- ``"stream"`` mode: a fictional NDJSON streaming protocol at ``POST /generate``
  (the target protocol agents must implement from ``STREAM_API_DOCUMENTATION``).

The probe flag distinguishes "the agent's provider really hit the API exactly
as specified" from "some request got through": it is set only for a request
whose path, method, credential, model, prompt and stream flag all match the
constants below, and ``reset()`` clears it between cases.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

OPENAI_PROVIDER_ID = "acme"
OPENAI_MODEL_ID = "acme-chat"
OPENAI_PROBE_PROMPT = "Reply with ACME_OK."
OPENAI_PROBE_RESPONSE = "ACME_OK"
STREAM_PROVIDER_ID = "acme-stream"
STREAM_MODEL_ID = "acme-stream-chat"
STREAM_PROBE_PROMPT = "Reply with ACME_STREAM_OK."
STREAM_PROBE_RESPONSE = "ACME_STREAM_OK"

STREAM_API_DOCUMENTATION = json.dumps({
    "name": "Acme Streaming API",
    "request": {
        "method": "POST",
        "path": "/generate",
        "headers": {"content-type": "application/json", "x-acme-key": "resolved credential"},
        "body": {"model": STREAM_MODEL_ID, "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    },
    "response": {
        "contentType": "application/x-ndjson",
        "events": [
            {"type": "text_delta", "text": "Hello"},
            {"type": "usage", "input_tokens": 3, "output_tokens": 2},
            {"type": "done", "reason": "stop"},
        ],
    },
}, indent=2) + "\n"


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _parse_prompt(payload: Any) -> Optional[str]:
    """The first user message's string content, or None."""
    if not _is_record(payload) or not isinstance(payload.get("messages"), list):
        return None
    for message in payload["messages"]:
        if _is_record(message) and message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _ndjson(event: Dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _sse(event: Dict[str, Any]) -> bytes:
    return (f"data: {json.dumps(event, ensure_ascii=False)}\n\n").encode("utf-8")


class AcmeServer:
    """Fake provider HTTP server; one instance per eval suite or test file."""

    def __init__(self, mode: str) -> None:
        if mode not in ("openai", "stream"):
            raise ValueError(f"Unknown Acme server mode: {mode!r}")
        self._mode = mode
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._origin = ""
        self._received_valid_request = False

    def _reject(self, handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
        handler.send_response(status)
        handler.send_header("content-type", "text/plain")
        handler.end_headers()
        handler.wfile.write(message.encode("utf-8"))

    def _respond_stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("content-type", "application/x-ndjson")
        handler.end_headers()
        handler.wfile.write(_ndjson({"type": "text_delta", "text": "ACME_"}))
        handler.wfile.write(_ndjson({"type": "text_delta", "text": "STREAM_OK"}))
        handler.wfile.write(_ndjson({"type": "usage", "input_tokens": 4, "output_tokens": 3}))
        handler.wfile.write(_ndjson({"type": "done", "reason": "stop"}))

    def _respond_openai(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.send_header("cache-control", "no-cache")
        handler.end_headers()
        handler.wfile.write(_sse({
            "id": "chatcmpl-acme",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": OPENAI_MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": OPENAI_PROBE_RESPONSE},
                         "finish_reason": None}],
        }))
        handler.wfile.write(_sse({
            "id": "chatcmpl-acme",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": OPENAI_MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }))
        handler.wfile.write(b"data: [DONE]\n\n")

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        content_type = handler.headers.get("content-type") or ""
        length = int(handler.headers.get("content-length") or 0)
        raw_body = handler.rfile.read(length) if length else b""

        expected_path = "/v1/chat/completions" if self._mode == "openai" else "/generate"
        if handler.path != expected_path:
            return self._reject(handler, 404, "Unknown endpoint")
        if handler.command != "POST":
            return self._reject(handler, 405, "Expected POST")
        if not content_type.startswith("application/json"):
            return self._reject(handler, 415, "Expected application/json")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self._reject(handler, 400, "Invalid JSON")
        if not _is_record(payload):
            return self._reject(handler, 422, "Expected a JSON object")
        prompt = _parse_prompt(payload)

        if self._mode == "stream":
            if handler.headers.get("x-acme-key") != "resolved-stream-key":
                return self._reject(handler, 401, "Invalid Acme Stream credential")
            if payload.get("model") != STREAM_MODEL_ID or prompt is None or payload.get("stream") is not True:
                return self._reject(handler, 422, "Invalid Acme Stream request")
            self._received_valid_request = prompt == STREAM_PROBE_PROMPT
            return self._respond_stream(handler)

        if handler.headers.get("authorization") != "Bearer resolved-acme-key":
            return self._reject(handler, 401, "Invalid Acme credential")
        if payload.get("model") != OPENAI_MODEL_ID or prompt is None or payload.get("stream") is not True:
            return self._reject(handler, 422, "Invalid OpenAI-compatible request")
        self._received_valid_request = prompt == OPENAI_PROBE_PROMPT
        return self._respond_openai(handler)

    def _make_handler(self) -> type:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def _dispatch(self) -> None:
                try:
                    server._handle(self)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                pass

            do_POST = _dispatch
            do_GET = _dispatch
            do_PUT = _dispatch
            do_DELETE = _dispatch
            do_PATCH = _dispatch

        return Handler

    def start(self) -> None:
        """Bind 127.0.0.1 on a random free port and serve in a daemon thread."""
        if self._server is not None:
            raise RuntimeError("Acme server is already running.")
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server = server
        self._thread = thread
        self._origin = f"http://127.0.0.1:{server.server_address[1]}"

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def reset(self) -> None:
        """Clear the probe flag; call before every case."""
        self._received_valid_request = False

    def origin(self) -> str:
        return self._origin

    def base_url(self) -> str:
        return f"{self._origin}/v1"

    def valid_request_received(self) -> bool:
        return self._received_valid_request


def create_acme_server(mode: str) -> AcmeServer:
    return AcmeServer(mode)

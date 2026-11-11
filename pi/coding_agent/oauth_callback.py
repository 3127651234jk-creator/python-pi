"""Loopback OAuth callback server and shared device-code polling; all tokens remain local."""

import asyncio
import base64
import hashlib
import math
import secrets
import time
from html import escape
from urllib.parse import parse_qs, urlsplit

MIN_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
SLOW_DOWN_INCREMENT_SECONDS = 5.0
DEVICE_FLOW_TIMEOUT_MESSAGE = "Device flow timed out"
DEVICE_FLOW_SLOW_DOWN_TIMEOUT_MESSAGE = ("Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or "
    "VM environments. Please sync or restart the VM clock and try again.")
RESPONSE_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 409: "Conflict", 502: "Bad Gateway"}


def oauth_page(message, details=""):
    detail = f'<div class="details">{escape(details)}</div>' if details else ""
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8" /><title>pi</title><style>'
        "body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#09090b;"
        "color:#fafafa;font-family:system-ui,sans-serif;text-align:center}main{max-width:560px;padding:24px}"
        "h1{font-size:28px;margin:0 0 10px}p{margin:0;line-height:1.7;color:#a1a1aa;font-size:15px}"
        ".details{margin-top:16px;font-family:monospace;font-size:13px;color:#a1a1aa;white-space:pre-wrap;"
        'word-break:break-word}</style></head><body><main><h1>pi</h1><p>' + escape(message) + f"</p>{detail}</main></body></html>")


def oauth_success_html(message):
    return oauth_page(message)


def oauth_error_html(message, details=""):
    return oauth_page(message, details)


def generate_pkce():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


class OAuthCallbackServer:
    """Loopback redirect handler. The browser page is sent after `complete` runs, so it can show exchange failures.

    `wait()` resolves with the result of `complete`, or with the callback query parameters when no `complete`
    was given, or with None after `cancel()`. It rejects when the provider redirects with an error, `complete`
    fails, the timeout elapses, or the server closes.
    """

    def __init__(self, provider_name, path, state=None, complete=None):
        self.provider_name = provider_name
        self.path = path
        self.state = state
        self.complete = complete
        self.redirect_uri = ""
        self.claimed = False
        self.settled = False

    async def wait(self):
        return await asyncio.shield(self._future)

    def cancel(self):
        """Stop waiting for the browser unless a callback is already being completed."""
        if not self.claimed:
            self._finish_value(None)

    async def close(self):
        """Stop the server; on Python 3.9 (Windows proactor) wait_closed() may never return, so skip it."""
        self._finish_error(ValueError("OAuth callback server closed"))
        self._tcp.close()

    def _finish_value(self, value):
        if not self.settled:
            self.settled = True
            if not self._future.cancelled():
                self._future.set_result(value)

    def _finish_error(self, error):
        if not self.settled:
            self.settled = True
            if not self._future.cancelled():
                self._future.set_exception(error)

    async def _respond(self, writer, status, body):
        payload = body.encode("utf-8")
        writer.write((f"HTTP/1.1 {status} {RESPONSE_REASONS.get(status, 'OK')}\r\ncontent-type: text/html; charset=utf-8\r\n"
            f"cache-control: no-store\r\nconnection: close\r\ncontent-length: {len(payload)}\r\n\r\n").encode("ascii") + payload)
        try:
            await writer.drain()
        except (ConnectionError, OSError):
            pass

    async def _handle(self, reader, writer):
        try:
            try:
                request = await asyncio.wait_for(reader.readline(), 5)
            except asyncio.TimeoutError:
                return
            parts = request.decode("latin-1").split()
            target = urlsplit(parts[1]) if len(parts) == 3 and parts[0] == "GET" else None
            if target is None or target.path != self.path:
                await self._respond(writer, 404, oauth_error_html("Callback route not found."))
                return
            params = {key: values[0] for key, values in parse_qs(target.query).items() if values}
            if self.state is not None and params.get("state") != self.state:
                await self._respond(writer, 400, oauth_error_html("State mismatch."))
                return
            if self.claimed or self.settled:
                await self._respond(writer, 409, oauth_error_html("This sign-in has already been handled."))
                return
            error = params.get("error")
            if error:
                description = params.get("error_description") or error
                await self._respond(writer, 400, oauth_error_html(f"{self.provider_name} authorization failed.", description))
                self._finish_error(ValueError(f"{self.provider_name} authorization failed: {description}"))
                return
            code = params.get("code")
            if not code:
                await self._respond(writer, 400, oauth_error_html("Missing authorization code."))
                return
            self.claimed = True
            try:
                value = await self.complete(code) if self.complete else params
            except Exception as exc:
                await self._respond(writer, 502, oauth_error_html(f"{self.provider_name} sign-in failed.", str(exc)))
                self._finish_error(exc)
                return
            await self._respond(writer, 200, oauth_success_html(f"Signed in to {self.provider_name}. You may now close this page."))
            self._finish_value(value)
        except (OSError, ValueError, UnicodeError):
            pass
        finally:
            writer.close()


async def start_oauth_callback_server(provider_name, host, port, path, state=None, complete=None, timeout_ms=None):
    server = OAuthCallbackServer(provider_name, path, state, complete)
    server._loop = asyncio.get_running_loop()
    server._future = server._loop.create_future()
    # A wait that is cancelled or closed may never be observed; consume the exception.
    server._future.add_done_callback(lambda future: None if future.cancelled() else future.exception())
    if timeout_ms is not None:
        server._loop.call_later(timeout_ms / 1000, server._finish_error, ValueError(f"{provider_name} sign-in timed out"))
    server._tcp = await asyncio.start_server(server._handle, host, port)
    bound = server._tcp.sockets[0].getsockname()[1]
    display_host = f"[{host}]" if ":" in host else host
    server.redirect_uri = f"http://{display_host}:{bound}{path}"
    return server


async def poll_oauth_device_code_flow(poll, interval_seconds=None, expires_in_seconds=None, wait_before_first_poll=False):
    """RFC 8628 section 3.2/3.5 polling: wait between polls, treat slow_down as a 5-second (or server-given) increase."""
    deadline = time.monotonic() + expires_in_seconds if expires_in_seconds is not None else math.inf
    interval = max(MIN_POLL_INTERVAL_SECONDS, float(interval_seconds) if interval_seconds else DEFAULT_POLL_INTERVAL_SECONDS)
    slow_downs = 0
    if wait_before_first_poll and deadline - time.monotonic() > 0:
        await asyncio.sleep(min(interval, deadline - time.monotonic()))
    while time.monotonic() < deadline:
        result = await poll()
        if result["status"] == "complete":
            return result["value"]
        if result["status"] == "failed":
            raise ValueError(result["message"])
        if result["status"] == "slow_down":
            slow_downs += 1
            given = result.get("interval_seconds")
            interval = max(MIN_POLL_INTERVAL_SECONDS, float(given)) if given else max(MIN_POLL_INTERVAL_SECONDS, interval + SLOW_DOWN_INCREMENT_SECONDS)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(interval, remaining))
    raise ValueError(DEVICE_FLOW_SLOW_DOWN_TIMEOUT_MESSAGE if slow_downs else DEVICE_FLOW_TIMEOUT_MESSAGE)

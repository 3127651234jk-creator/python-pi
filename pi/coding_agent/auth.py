"""Credential persistence and OpenRouter PKCE authentication."""

import asyncio
import base64
import hashlib
import secrets
import time
import math
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from .config import read_json, write_json
from .subscription_auth import refresh_subscription


class AuthStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, provider: str) -> Optional[dict]:
        credential = read_json(self.path).get(provider)
        if credential is not None and not isinstance(credential, dict):
            raise ValueError("Invalid credential record")
        if credential is not None:
            key = credential.get("key") if credential.get("type") == "api_key" else credential.get("access")
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid credential token")
            if credential.get("type") == "oauth" and (type(credential.get("expires")) not in (int, float) or not math.isfinite(credential["expires"])):
                raise ValueError("Invalid credential expiration")
        return credential

    def set(self, provider: str, credential: dict) -> None:
        credentials = read_json(self.path)
        credentials[provider] = credential
        write_json(self.path, credentials)

    def logout(self, provider: str) -> None:
        credentials = read_json(self.path)
        credentials.pop(provider, None)
        write_json(self.path, credentials)

    async def key(self, provider: str, oauth: Optional[dict] = None) -> Optional[str]:
        credential = self.get(provider)
        if not credential:
            return None
        if credential.get("type") == "api_key":
            return credential["key"]
        if credential.get("type") != "oauth":
            raise ValueError("Unknown credential type")
        if credential.get("expires", 0) <= time.time() + 60:
            if provider in ("anthropic", "openai-codex", "github-copilot", "openai", "kimi-coding", "xai", "radius"):
                credential = await refresh_subscription(provider, credential)
                self.set(provider, credential)
                return credential["access"]
            if not oauth or not credential.get("refresh"):
                raise ValueError(f"OAuth credential expired; log in again: {provider}")
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(oauth["token_url"], data={
                    "grant_type": "refresh_token", "client_id": oauth["client_id"],
                    "refresh_token": credential["refresh"],
                })
            credential = token_credential(response, credential.get("refresh", ""))
            self.set(provider, credential)
        return credential["access"]


def token_credential(response: httpx.Response, refresh: str = "") -> dict:
    if response.status_code >= 400:
        raise ValueError(f"OAuth token request failed (HTTP {response.status_code})")
    data = response_object(response)
    if not isinstance(data.get("access_token"), str) or not data["access_token"]:
        raise ValueError("OAuth response has no access token")
    expires = data.get("expires_in", 3600)
    if type(expires) not in (int, float) or not math.isfinite(expires) or expires <= 0:
        raise ValueError("Invalid OAuth expiration")
    return {"type": "oauth", "access": data["access_token"],
            "refresh": data.get("refresh_token", refresh), "expires": time.time() + expires}


def response_object(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError("OAuth returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("OAuth response must be an object")
    return data


async def login_openrouter(notify: Callable[[str], None], timeout: float = 300) -> dict:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    path = "/callback/" + secrets.token_urlsafe(24)
    future = asyncio.get_running_loop().create_future()

    async def callback(reader, writer):
        try:
            request = await asyncio.wait_for(reader.readline(), 5)
            parts = request.decode("ascii").split(" ")
            target = urlsplit(parts[1]) if len(parts) == 3 else None
            params = parse_qs(target.query) if target else {}
            valid = target is not None and target.path == path and parts[0] == "GET"
            code = params.get("code", [""])[0]
            if valid and code and not future.done():
                future.set_result(code)
                writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 18\r\n\r\nReturn to terminal")
            else:
                writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        except (ValueError, UnicodeError, asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(callback, "127.0.0.1", 0, limit=8192)
    try:
        port = server.sockets[0].getsockname()[1]
        notify("https://openrouter.ai/auth?" + urlencode({
            "callback_url": f"http://127.0.0.1:{port}{path}",
            "code_challenge": challenge, "code_challenge_method": "S256",
        }))
        code = await asyncio.wait_for(future, timeout)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post("https://openrouter.ai/api/v1/auth/keys", json={
                "code": code, "code_verifier": verifier, "code_challenge_method": "S256",
            })
        if response.status_code >= 400:
            raise ValueError(f"OAuth exchange failed (HTTP {response.status_code})")
        key = response_object(response).get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("OAuth exchange returned no key")
        return {"type": "oauth", "access": key, "refresh": "", "expires": 1e30}
    finally:
        server.close()
        await server.wait_closed()


async def login_device(oauth: dict, notify: Callable[[str], None]) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(oauth["device_url"], data={
            "client_id": oauth["client_id"], "scope": oauth.get("scope", ""),
        })
        if response.status_code >= 400:
            raise ValueError(f"Device authorization failed (HTTP {response.status_code})")
        data = response_object(response)
        notify(f"{data['verification_uri']}  code: {data['user_code']}")
        deadline = time.monotonic() + float(data["expires_in"])
        interval = max(1.0, float(data.get("interval", 5)))
        while time.monotonic() < deadline:
            await asyncio.sleep(min(interval, max(0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                break
            response = await client.post(oauth["token_url"], data={
                "client_id": oauth["client_id"], "device_code": data["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            })
            error = response_object(response).get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            return token_credential(response)
    raise ValueError("Device authorization expired")

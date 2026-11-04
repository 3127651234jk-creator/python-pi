"""Google Vertex AI adapter.

Reuses the Gemini ``generateContent`` wire format (same request builder and
SSE converter as the Google AI adapter) but authenticates with a Google Cloud
service account: a signed RS256 JWT exchanged for an OAuth access token at
``oauth2.googleapis.com``. The signing is implemented with pure stdlib
(minimal DER parsing + RSA PKCS#1 v1.5) so no extra dependency is required.

Environment:
  GOOGLE_APPLICATION_CREDENTIALS  path to the service-account JSON key
  GOOGLE_CLOUD_PROJECT / VERTEX_PROJECT / GCP_PROJECT
  GOOGLE_CLOUD_REGION / VERTEX_REGION        (default us-central1)
  GOOGLE_OAUTH_ACCESS_TOKEN                  pre-minted token (short-circuits JWT)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from ..models import Model
from ..types import TranscriptContext
from .base import SimpleStreamOptions, run_provider_stream
from .google import build_request, make_google_converter

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"

_TOKEN_CACHE: Dict[str, tuple] = {}
_TOKEN_LOCK: Optional[asyncio.Lock] = None


# ---------------------------------------------------------------------------
# Minimal DER + RSA PKCS#1 v1.5 signing (stdlib only)
# ---------------------------------------------------------------------------

def _der_read(data: bytes, offset: int):
    """Read one DER TLV; return (tag, content, next_offset)."""
    tag = data[offset]
    offset += 1
    length = data[offset]
    offset += 1
    if length & 0x80:
        count = length & 0x7F
        length = int.from_bytes(data[offset:offset + count], "big")
        offset += count
    return tag, data[offset:offset + length], offset + length


def _der_sequence_integers(content: bytes):
    """Yield INTEGER values at the top level of a DER sequence."""
    values = []
    offset = 0
    while offset < len(content):
        tag, body, offset = _der_read(content, offset)
        if tag == 0x02:
            values.append(int.from_bytes(body, "big"))
        elif tag in (0x30, 0xA0):  # nested sequence / context tag: descend
            values.extend(_der_sequence_integers(body))
        elif tag == 0x04 and not values:  # PKCS#8 octet string wrapping PKCS#1
            values.extend(_der_sequence_integers(body))
    return values


def load_private_key(pem: str):
    """Return (n, d) from a PKCS#1 or PKCS#8 PEM RSA private key."""
    lines = [line for line in pem.strip().splitlines() if "-----" not in line]
    der = base64.b64decode("".join(lines))
    integers = _der_sequence_integers(der)
    if len(integers) < 4:
        raise ValueError("Unsupported RSA private key format")
    return integers[1], integers[3]  # version, n, e, d, ...


def _pkcs1_v15_signature(private, message: bytes) -> bytes:
    n, d = private
    k = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(message).digest()
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + digest
    padding = b"\xff" * (k - len(digest_info) - 3)
    em = b"\x00\x01" + padding + b"\x00" + digest_info
    signature = pow(int.from_bytes(em, "big"), d, n)
    return signature.to_bytes(k, "big")


def service_account_jwt(account: dict, scope: str = SCOPE, now: Optional[float] = None) -> str:
    """Build a signed RS256 assertion for the service-account token exchange."""
    issued = int(now if now is not None else time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    claims = {"iss": account["client_email"], "scope": scope, "aud": TOKEN_URL,
              "iat": issued, "exp": issued + 3600}
    private = load_private_key(account["private_key"])

    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=")

    signing_input = segment(header) + b"." + segment(claims)
    signature = base64.urlsafe_b64encode(_pkcs1_v15_signature(private, signing_input)).rstrip(b"=")
    return (signing_input + b"." + signature).decode()


# ---------------------------------------------------------------------------
# Access-token acquisition with in-process caching
# ---------------------------------------------------------------------------

def vertex_settings() -> Dict[str, str]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT") or os.environ.get("GCP_PROJECT") or ""
    region = os.environ.get("GOOGLE_CLOUD_REGION") or os.environ.get("VERTEX_REGION") or "us-central1"
    return {"project": project, "region": region}


def vertex_base_url(settings: Optional[Dict[str, str]] = None) -> str:
    settings = settings or vertex_settings()
    if not settings["project"]:
        raise ValueError("Vertex AI requires GOOGLE_CLOUD_PROJECT (or VERTEX_PROJECT/GCP_PROJECT)")
    region = settings["region"]
    return f"https://{region}-aiplatform.googleapis.com/v1/projects/{settings['project']}/locations/{region}/publishers/google"


async def vertex_access_token(account_path: Optional[str] = None) -> str:
    """Return a valid access token, minting one via JWT exchange when needed."""
    direct = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if direct:
        return direct
    path = account_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or ""
    if not path:
        raise ValueError("Vertex AI requires GOOGLE_APPLICATION_CREDENTIALS (service-account JSON)")
    with open(path, "r", encoding="utf-8") as handle:
        account = json.load(handle)
    if not isinstance(account, dict) or not account.get("client_email") or not account.get("private_key"):
        raise ValueError("Service-account JSON is missing client_email/private_key")
    cache_key = account["client_email"] + "|" + SCOPE
    cached = _TOKEN_CACHE.get(cache_key)
    if cached and cached[1] > time.time() + 60:
        return cached[0]
    global _TOKEN_LOCK
    if _TOKEN_LOCK is None:
        _TOKEN_LOCK = asyncio.Lock()
    async with _TOKEN_LOCK:
        cached = _TOKEN_CACHE.get(cache_key)
        if cached and cached[1] > time.time() + 60:
            return cached[0]
        assertion = service_account_jwt(account)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion})
        if response.status_code >= 400:
            raise ValueError(f"Vertex token exchange failed (HTTP {response.status_code})")
        data = response.json()
        token = data.get("access_token")
        expires = data.get("expires_in")
        if not isinstance(token, str) or not token or not isinstance(expires, (int, float)):
            raise ValueError("Vertex token response is missing access_token/expires_in")
        _TOKEN_CACHE[cache_key] = (token, time.time() + float(expires))
        return token


# ---------------------------------------------------------------------------
# Stream adapter
# ---------------------------------------------------------------------------

async def stream_vertex(model: Model, context: TranscriptContext, options: Optional[SimpleStreamOptions] = None):
    options = options or SimpleStreamOptions()
    endpoint = model.base_url or vertex_base_url()
    streaming_model = replace(model, base_url=endpoint.rstrip("/"))
    convert = make_google_converter(streaming_model)
    path = "/models/" + quote(streaming_model.id.removeprefix("models/").removeprefix("publishers/google/models/"), safe="") + ":streamGenerateContent?alt=sse"
    token = await vertex_access_token()

    def headers_factory():
        # run_provider_stream expects a synchronous headers callable.
        return {"Authorization": "Bearer " + token, "content-type": "application/json", **streaming_model.headers}


    async for event in run_provider_stream(streaming_model,
        lambda: build_request(streaming_model, context, options), path, headers_factory, convert, options,
        int(time.time() * 1000)):
        yield event

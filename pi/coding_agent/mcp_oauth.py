"""OAuth 2.1 discovery, dynamic client registration and PKCE flows for HTTP MCP servers.

Mirrors ``packages/mcp/src/oauth`` at pi v1.0.4:

- Protected resource metadata discovery (``/.well-known/oauth-protected-resource<path>``
  with a root fallback for path-carrying servers).
- Authorization server metadata discovery in the RFC 8414 / OpenID Connect order,
  with issuer validation and an explicit trusted metadata URL.
- Dynamic client registration, including the OpenID Connect ``application_type``
  derived from the redirect URIs (MCP SEP-837).
- Authorization-code + PKCE S256 and refresh-token grants.

The module is transport-agnostic about persistence: callers pass and receive the
per-server record that ``mcp-auth.json`` stores.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
from typing import Dict, Optional
from urllib.parse import urlencode, urlsplit

import httpx

LATEST_PROTOCOL_VERSION = "2025-03-26"
# Bounds each request to the authorization server, so an unresponsive one cannot
# hold the refresh lock, delay shutdown, or keep a sign-in waiting for long
# (alignment of the TS OAUTH_REQUEST_TIMEOUT_MS).
OAUTH_REQUEST_TIMEOUT_SECONDS = 15
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")


class McpOAuthError(RuntimeError):
    """An OAuth protocol error reported by the authorization server."""

    def __init__(self, code: str, description: Optional[str] = None, uri: Optional[str] = None) -> None:
        super().__init__(description or code)
        self.code = code
        self.description = description or code
        self.uri = uri


class McpAuthorizationRequired(McpOAuthError):
    """The MCP server rejected a request with 401/403 and sent a challenge."""

    def __init__(self, status: int, challenge: dict) -> None:
        super().__init__("authorization_required", f"MCP authorization required (HTTP {status})")
        self.status = status
        self.challenge = challenge


class McpLoginRequired(RuntimeError):
    """No usable token and no refresh token; the user must run ``/mcp login``."""


class OAuthIssuerMismatchError(McpOAuthError):
    def __init__(self, expected: str, actual: Optional[str]) -> None:
        super().__init__("issuer_mismatch", f"Authorization server issuer {actual!r} does not match {expected!r}")
        self.expected = expected
        self.actual = actual


class OAuthInsecureEndpointError(McpOAuthError):
    def __init__(self, url: str) -> None:
        super().__init__("insecure_endpoint", f"OAuth endpoint must use https (except on loopback): {url}")
        self.url = url


class OAuthRegistrationError(McpOAuthError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__("registration_failed", f"Dynamic client registration failed (HTTP {status}): {body}")
        self.status = status
        self.body = body


def loopback(hostname: str) -> bool:
    return hostname.lower() in LOOPBACK_HOSTS


def application_type(redirect_uris) -> str:
    """OpenID Connect client type for redirect URIs: loopback or a custom scheme is native."""

    for uri in redirect_uris:
        try:
            parts = urlsplit(uri)
        except ValueError:
            continue
        if not parts.scheme:
            continue
        if parts.scheme not in ("http", "https"):
            return "native"
        if not parts.netloc:
            continue
        if loopback(parts.hostname or ""):
            return "native"
    return "web"


def secure_endpoint(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "https" and not loopback(parts.hostname or ""):
        raise OAuthInsecureEndpointError(value)
    return value


def parse_www_authenticate(header: Optional[str]) -> dict:
    """Bearer/DPoP challenge fields. DPoP is recognized but not implemented."""

    if not header:
        return {}
    scheme = header.strip().split(None, 1)[0].lower() if header.strip() else ""
    if scheme not in ("bearer", "dpop"):
        return {}

    def field(name: str) -> Optional[str]:
        match = re.search(r'(?:^|[,\s])' + name + r'=(?:"([^"]*)"|([^\s,]+))', header, re.I)
        if not match:
            return None
        return match.group(1) or match.group(2) or None

    return {
        "resource_metadata": field("resource_metadata"),
        "scope": field("scope"),
        "error": field("error"),
        "error_description": field("error_description"),
    }


def _required_string(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {name}")
    return value


def _optional_string(value, name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    return _required_string(value, name)


def _optional_strings(value, name: str) -> Optional[list]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid {name}")
    return list(value)


def _safe_url(value, name: str) -> str:
    text = _required_string(value, name)
    parts = urlsplit(text)
    if parts.scheme in ("javascript", "data", "vbscript") or not parts.scheme:
        raise ValueError(f"Invalid {name}")
    if parts.scheme in ("http", "https") and not parts.netloc:
        raise ValueError(f"Invalid {name}")
    return text


def _optional_url(value, name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    return _safe_url(value, name)


def parse_protected_resource_metadata(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Invalid OAuth protected resource metadata")
    return {
        **value,
        "resource": _safe_url(value.get("resource"), "OAuth protected resource metadata resource"),
        "authorization_servers": _optional_strings(value.get("authorization_servers"), "authorization_servers"),
        "scopes_supported": _optional_strings(value.get("scopes_supported"), "scopes_supported"),
    }


def parse_authorization_server_metadata(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Invalid authorization server metadata")
    response_types = _optional_strings(value.get("response_types_supported"), "response_types_supported")
    if not response_types:
        raise ValueError("Invalid response_types_supported")
    return {
        **value,
        "issuer": _safe_url(value.get("issuer"), "authorization server issuer"),
        "authorization_endpoint": _safe_url(value.get("authorization_endpoint"), "authorization endpoint"),
        "token_endpoint": _safe_url(value.get("token_endpoint"), "token endpoint"),
        "registration_endpoint": _optional_url(value.get("registration_endpoint"), "registration endpoint"),
        "scopes_supported": _optional_strings(value.get("scopes_supported"), "scopes_supported"),
        "response_types_supported": response_types,
        "grant_types_supported": _optional_strings(value.get("grant_types_supported"), "grant_types_supported"),
        "token_endpoint_auth_methods_supported": _optional_strings(
            value.get("token_endpoint_auth_methods_supported"), "token_endpoint_auth_methods_supported"),
        "code_challenge_methods_supported": _optional_strings(
            value.get("code_challenge_methods_supported"), "code_challenge_methods_supported"),
        "client_id_metadata_document_supported": (
            value.get("client_id_metadata_document_supported")
            if isinstance(value.get("client_id_metadata_document_supported"), bool) else None),
        "authorization_response_iss_parameter_supported": (
            value.get("authorization_response_iss_parameter_supported")
            if isinstance(value.get("authorization_response_iss_parameter_supported"), bool) else None),
    }


def parse_tokens(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Invalid OAuth token response")
    expires = value.get("expires_in")
    if expires is None or expires == "":
        expires = None
    else:
        try:
            expires = float(expires)
        except (TypeError, ValueError):
            raise ValueError("Invalid expires_in") from None
        if expires != expires or expires in (float("inf"), float("-inf")):
            raise ValueError("Invalid expires_in")
    return {
        "access_token": _required_string(value.get("access_token"), "access_token"),
        "token_type": _required_string(value.get("token_type"), "token_type"),
        "expires_in": expires,
        "scope": _optional_string(value.get("scope"), "scope"),
        "refresh_token": _optional_string(value.get("refresh_token"), "refresh_token"),
        "id_token": _optional_string(value.get("id_token"), "id_token"),
    }


def parse_client_information(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Invalid OAuth client registration response")
    return {
        **value,
        "client_id": _required_string(value.get("client_id"), "client_id"),
        "client_secret": _optional_string(value.get("client_secret"), "client_secret"),
        "client_id_issued_at": value.get("client_id_issued_at")
        if isinstance(value.get("client_id_issued_at"), (int, float)) else None,
        "client_secret_expires_at": value.get("client_secret_expires_at")
        if isinstance(value.get("client_secret_expires_at"), (int, float)) else None,
        "redirect_uris": _optional_strings(value.get("redirect_uris"), "redirect_uris") or [],
    }


def _origin(parts) -> str:
    return f"{parts.scheme}://{parts.netloc}"


def _path_suffix(pathname: str) -> str:
    return pathname[:-1] if pathname.endswith("/") else pathname


def _is_discovery_miss(status: int) -> bool:
    return 400 <= status < 500 or status == 502


async def _fetch_metadata(client: httpx.AsyncClient, url: str, protocol_version: str) -> httpx.Response:
    return await client.get(url, headers={"Accept": "application/json",
                                          "MCP-Protocol-Version": protocol_version})


async def discover_protected_resource_metadata(
    server_url: str,
    client: httpx.AsyncClient,
    resource_metadata_url: Optional[str] = None,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> dict:
    server = urlsplit(server_url)
    origin = _origin(server)
    if resource_metadata_url:
        response = await _fetch_metadata(client, resource_metadata_url, protocol_version)
    else:
        suffix = _path_suffix(server.path or "/")
        response = await _fetch_metadata(
            client, f"{origin}/.well-known/oauth-protected-resource{suffix}", protocol_version)
        if suffix and _is_discovery_miss(response.status_code):
            response = await _fetch_metadata(
                client, f"{origin}/.well-known/oauth-protected-resource", protocol_version)
    if response.status_code >= 400:
        raise McpOAuthError("server_error", f"HTTP {response.status_code} loading OAuth protected resource metadata")
    return parse_protected_resource_metadata(response.json())


def build_authorization_server_discovery_urls(authorization_server_url: str) -> list:
    issuer = urlsplit(authorization_server_url)
    origin = _origin(issuer)
    path = _path_suffix(issuer.path or "/")
    urls = [
        (f"{origin}/.well-known/oauth-authorization-server{path}", "oauth"),
        (f"{origin}/.well-known/openid-configuration{path}", "oidc"),
    ]
    if path:
        urls.append((f"{origin}{path}/.well-known/openid-configuration", "oidc"))
    return urls


async def discover_authorization_server_metadata(
    authorization_server_url: str,
    client: httpx.AsyncClient,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
    skip_issuer_validation: bool = False,
) -> Optional[dict]:
    for url, _kind in build_authorization_server_discovery_urls(authorization_server_url):
        response = await _fetch_metadata(client, url, protocol_version)
        if response.status_code >= 400:
            if _is_discovery_miss(response.status_code):
                continue
            raise McpOAuthError("server_error",
                                f"HTTP {response.status_code} loading authorization server metadata from {url}")
        metadata = parse_authorization_server_metadata(response.json())
        if not skip_issuer_validation:
            expected = authorization_server_url.rstrip("/")
            actual = metadata["issuer"].rstrip("/")
            if actual != expected:
                raise OAuthIssuerMismatchError(expected, metadata["issuer"])
        return metadata
    return None


async def discover_oauth_server_info(
    server_url: str,
    client: httpx.AsyncClient,
    resource_metadata_url: Optional[str] = None,
    authorization_server_metadata_url: Optional[str] = None,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
    skip_issuer_validation: bool = False,
) -> dict:
    resource_metadata = None
    try:
        resource_metadata = await discover_protected_resource_metadata(
            server_url, client, resource_metadata_url, protocol_version)
    except (McpOAuthError, ValueError):
        # Discovery failures mean "no protected resource metadata"; network failures propagate.
        resource_metadata = None
    if authorization_server_metadata_url:
        response = await _fetch_metadata(client, authorization_server_metadata_url, protocol_version)
        if response.status_code >= 400:
            raise McpOAuthError(
                "server_error",
                f"HTTP {response.status_code} loading authorization server metadata from {authorization_server_metadata_url}")
        metadata = parse_authorization_server_metadata(response.json())
        return {"authorization_server_url": metadata["issuer"],
                "authorization_server_metadata": metadata,
                "resource_metadata": resource_metadata}
    servers = (resource_metadata or {}).get("authorization_servers") or []
    server = urlsplit(server_url)
    authorization_server_url = servers[0] if servers else _origin(server) + "/"
    return {
        "authorization_server_url": authorization_server_url,
        "authorization_server_metadata": await discover_authorization_server_metadata(
            authorization_server_url, client, protocol_version, skip_issuer_validation),
        "resource_metadata": resource_metadata,
    }


def select_resource(server_url: str, metadata: Optional[dict]) -> Optional[str]:
    if not metadata:
        return None
    requested = urlsplit(server_url)
    configured = urlsplit(metadata["resource"])
    if _origin(requested) != _origin(configured):
        raise McpOAuthError(
            "invalid_resource",
            f"Protected resource {metadata['resource']} does not match MCP server {server_url}")
    requested_path = requested.path or "/"
    configured_path = configured.path or "/"
    if not requested_path.endswith("/"):
        requested_path += "/"
    if not configured_path.endswith("/"):
        configured_path += "/"
    if not requested_path.startswith(configured_path):
        raise McpOAuthError(
            "invalid_resource",
            f"Protected resource {metadata['resource']} does not match MCP server {server_url}")
    return metadata["resource"]


def merge_scopes(*scopes: Optional[str]) -> Optional[str]:
    merged = []
    for scope in scopes:
        for item in (scope or "").split():
            if item and item not in merged:
                merged.append(item)
    return " ".join(merged) if merged else None


def step_up_scope(granted: Optional[str], challenged: Optional[str]) -> Optional[str]:
    if not challenged:
        return None
    return merge_scopes(granted, challenged)


def pkce() -> tuple:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def start_authorization(
    authorization_server_url: str,
    client_information: dict,
    redirect_url: str,
    metadata: Optional[dict] = None,
    scope: Optional[str] = None,
    state: Optional[str] = None,
    resource: Optional[str] = None,
    authorize_url: Optional[str] = None,
) -> tuple:
    if metadata and "code" not in metadata.get("response_types_supported", []):
        raise McpOAuthError("unsupported_response_type", "Authorization server does not support authorization codes")
    methods = (metadata or {}).get("code_challenge_methods_supported")
    if methods and "S256" not in methods:
        raise McpOAuthError("unsupported_code_challenge", "Authorization server does not support PKCE S256")
    endpoint = authorize_url or (metadata or {}).get("authorization_endpoint") \
        or authorization_server_url.rstrip("/") + "/authorize"
    verifier, challenge = pkce()
    params = {
        "response_type": "code",
        "client_id": client_information["client_id"],
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "redirect_uri": redirect_url,
    }
    if state:
        params["state"] = state
    if scope:
        params["scope"] = scope
        if "offline_access" in scope.split():
            params["prompt"] = "consent"
    if resource:
        params["resource"] = resource
    separator = "&" if "?" in endpoint else "?"
    return endpoint + separator + urlencode(params), verifier


def select_client_auth_method(information: dict, supported: list) -> str:
    hinted = information.get("token_endpoint_auth_method")
    if hinted in ("client_secret_basic", "client_secret_post", "none") and (not supported or hinted in supported):
        return hinted
    if not supported:
        return "client_secret_basic" if information.get("client_secret") else "none"
    if information.get("client_secret") and "client_secret_basic" in supported:
        return "client_secret_basic"
    if information.get("client_secret") and "client_secret_post" in supported:
        return "client_secret_post"
    if "none" in supported:
        return "none"
    return "client_secret_post" if information.get("client_secret") else "none"


def apply_client_authentication(method: str, information: dict, headers: dict, params: dict) -> None:
    if method == "client_secret_basic":
        secret = information.get("client_secret")
        if not secret:
            raise McpOAuthError("invalid_client", "client_secret_basic requires a client secret")
        raw = f"{information['client_id']}:{secret}".encode()
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        return
    params["client_id"] = information["client_id"]
    if method == "client_secret_post" and information.get("client_secret"):
        params["client_secret"] = information["client_secret"]


async def token_request(
    client: httpx.AsyncClient,
    token_url: str,
    client_information: dict,
    params: Dict[str, str],
    metadata: Optional[dict] = None,
    resource: Optional[str] = None,
) -> dict:
    url = secure_endpoint(token_url)
    headers = {"Accept": "application/json", "content-type": "application/x-www-form-urlencoded"}
    data = dict(params)
    if resource:
        data["resource"] = resource
    method = select_client_auth_method(
        client_information, (metadata or {}).get("token_endpoint_auth_methods_supported") or [])
    apply_client_authentication(method, client_information, headers, data)
    response = await client.post(url, headers=headers, data=data)
    text = response.text
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("error"), str):
        raise McpOAuthError(value["error"],
                            value.get("error_description") if isinstance(value.get("error_description"), str) else None,
                            value.get("error_uri") if isinstance(value.get("error_uri"), str) else None)
    if response.status_code >= 400:
        raise McpOAuthError("server_error", f"HTTP {response.status_code}: {text}")
    return parse_tokens(value)


async def register_client(
    client: httpx.AsyncClient,
    authorization_server_url: str,
    client_metadata: dict,
    metadata: Optional[dict] = None,
    scope: Optional[str] = None,
) -> dict:
    endpoint = (metadata or {}).get("registration_endpoint")
    if metadata is not None and not endpoint:
        raise McpOAuthError("registration_not_supported",
                            "Authorization server does not support dynamic client registration")
    url = endpoint or authorization_server_url.rstrip("/") + "/register"
    body = {
        **client_metadata,
        "application_type": client_metadata.get("application_type")
        or application_type(client_metadata.get("redirect_uris") or []),
    }
    if scope:
        body["scope"] = scope
    response = await client.post(url, headers={"Accept": "application/json", "content-type": "application/json"},
                                 content=json.dumps(body))
    if response.status_code >= 400:
        raise OAuthRegistrationError(response.status_code, response.text)
    return parse_client_information(response.json())


async def exchange_authorization_code(
    client: httpx.AsyncClient,
    authorization_server_url: str,
    client_information: dict,
    code: str,
    code_verifier: str,
    redirect_url: str,
    metadata: Optional[dict] = None,
    resource: Optional[str] = None,
    token_url: Optional[str] = None,
) -> dict:
    endpoint = token_url or (metadata or {}).get("token_endpoint") \
        or authorization_server_url.rstrip("/") + "/token"
    return await token_request(client, endpoint, client_information, {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_url,
    }, metadata, resource)


async def refresh_authorization(
    client: httpx.AsyncClient,
    authorization_server_url: str,
    client_information: dict,
    refresh_token: str,
    metadata: Optional[dict] = None,
    resource: Optional[str] = None,
    token_url: Optional[str] = None,
) -> dict:
    endpoint = token_url or (metadata or {}).get("token_endpoint") \
        or authorization_server_url.rstrip("/") + "/token"
    tokens = await token_request(client, endpoint, client_information, {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }, metadata, resource)
    if not tokens.get("refresh_token"):
        tokens["refresh_token"] = refresh_token
    return tokens


def oauth_settings(oauth: Optional[dict]) -> dict:
    """Normalize snake_case Python config and the camelCase TS shape."""

    value = oauth or {}
    if not isinstance(value, dict):
        raise ValueError("oauth must be an object")
    return {
        "authorize_url": value.get("authorize_url") or value.get("authorizeUrl"),
        "token_url": value.get("token_url") or value.get("tokenUrl"),
        "client_id": value.get("client_id") or value.get("clientId"),
        "client_secret": value.get("client_secret") or value.get("clientSecret"),
        "callback_port": value.get("callback_port", value.get("callbackPort")),
        "callback_url": value.get("callback_url") or value.get("callbackUrl"),
        "scope": value.get("scope"),
        "client_name": value.get("client_name") or value.get("clientName"),
        "application_type": value.get("application_type") or value.get("applicationType"),
        "client_registration": value.get("client_registration") or value.get("clientRegistration"),
        "auth_server_metadata_url": value.get("auth_server_metadata_url") or value.get("authServerMetadataUrl"),
        "timeout": value.get("timeout", 600),
    }


def validate_oauth_settings(oauth: dict) -> dict:
    settings = oauth_settings(oauth)
    authorize_url, token_url = settings["authorize_url"], settings["token_url"]
    if bool(authorize_url) != bool(token_url):
        raise ValueError("oauth.authorize_url and oauth.token_url must be configured together")
    if authorize_url and not settings["client_id"]:
        raise ValueError("oauth.client_id is required when authorize_url and token_url are configured")
    for key in ("authorize_url", "token_url", "auth_server_metadata_url"):
        value = settings[key]
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"oauth.{key} must be a string")
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"oauth.{key} must be an http or https URL")
        if key == "auth_server_metadata_url" and parts.scheme != "https" and not loopback(parts.hostname or ""):
            raise ValueError("oauth.auth_server_metadata_url must use https (except on loopback)")
    for key in ("client_id", "client_secret", "scope"):
        if settings[key] is not None and not isinstance(settings[key], str):
            raise ValueError(f"oauth.{key} must be a string")
    if settings["client_name"] is not None and (
            not isinstance(settings["client_name"], str) or not settings["client_name"].strip()):
        raise ValueError("oauth.client_name must be a non-empty string")
    port = settings["callback_port"]
    if port is not None and (type(port) is not int or not 0 < port <= 65535):
        raise ValueError("oauth.callback_port must be a port number")
    callback_url = settings["callback_url"]
    if callback_url is not None:
        if not isinstance(callback_url, str):
            raise ValueError("oauth.callback_url must be a string")
        parts = urlsplit(callback_url)
        if (parts.scheme != "http" or not loopback(parts.hostname or "")
                or parts.query or parts.fragment):
            raise ValueError("oauth.callback_url must be an http URI on localhost, 127.0.0.1, or [::1] "
                             "without query or fragment")
        if parts.port and port is not None and parts.port != port:
            raise ValueError("oauth.callback_url and oauth.callback_port name different ports")
    application = settings["application_type"]
    if application is not None and application not in ("native", "web"):
        raise ValueError('oauth.application_type must be "native" or "web"')
    registration = settings["client_registration"]
    if registration not in (None, "dcr"):
        raise ValueError('oauth.client_registration must be "dcr" (Client ID Metadata Documents are not '
                         "implemented by the Python port)")
    timeout = settings["timeout"]
    if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
        raise ValueError("oauth.timeout must be between 0 and 86400 seconds")
    return settings


def callback_settings(oauth: dict) -> dict:
    settings = oauth_settings(oauth)
    raw = settings["callback_url"] or "http://127.0.0.1/callback"
    parts = urlsplit(raw)
    host = (parts.hostname or "127.0.0.1").strip("[]")
    if host == "localhost":
        listen_host = "127.0.0.1"
    elif host in ("127.0.0.1", "::1"):
        listen_host = host
    else:
        raise ValueError("oauth.callback_url must be on localhost, 127.0.0.1, or [::1]")
    path = parts.path or "/callback"
    port = parts.port if parts.port else settings["callback_port"]
    fixed = None
    if parts.port:
        # The callback server only answers on `path`, so a configured URL without
        # one would advertise a redirect the local server refuses with 404.
        fixed = f"{parts.scheme}://{parts.netloc}{path}"
    elif port is not None:
        display = f"[{host}]" if ":" in host else host
        fixed = f"http://{display}:{port}{path}"
    return {"listen_host": listen_host, "redirect_host": host, "port": port, "path": path,
            "fixed_redirect_url": fixed}


def client_metadata(oauth: dict, redirect_uri: str) -> dict:
    settings = oauth_settings(oauth)
    secret = settings["client_secret"]
    metadata = {
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post" if secret else "none",
        "client_name": settings["client_name"] or "pi",
    }
    if settings["application_type"]:
        metadata["application_type"] = settings["application_type"]
    return metadata


def static_client(oauth: dict) -> Optional[dict]:
    settings = oauth_settings(oauth)
    if not settings["client_id"]:
        return None
    client = {"client_id": settings["client_id"]}
    if settings["client_secret"]:
        client["client_secret"] = settings["client_secret"]
    return client


def stored_tokens(record: Optional[dict]) -> dict:
    record = record or {}
    return {
        "access_token": record.get("access", ""),
        "token_type": record.get("token_type", "Bearer"),
        "expires_in": None,
        "scope": record.get("scope"),
        "refresh_token": record.get("refresh"),
    }


def record_from_tokens(record: dict, tokens: dict, scope: Optional[str] = None) -> dict:
    updated = dict(record)
    updated["access"] = tokens["access_token"]
    updated["token_type"] = tokens.get("token_type") or "Bearer"
    updated["scope"] = tokens.get("scope") or scope or record.get("scope")
    if tokens.get("refresh_token"):
        updated["refresh"] = tokens["refresh_token"]
    expires_in = tokens.get("expires_in")
    if expires_in is None:
        # No advertised lifetime: the grant does not expire on its own.
        updated.pop("expires", None)
    else:
        updated["expires"] = time.time() + float(expires_in)
    return updated

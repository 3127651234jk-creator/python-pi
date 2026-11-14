"""Provider-specific subscription OAuth exchanges; all tokens remain local."""

import asyncio
import base64
import errno
import hashlib
import json
import math
import os
import secrets
import re
import time
import uuid
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from .oauth_callback import generate_pkce, poll_oauth_device_code_flow, start_oauth_callback_server

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
COPILOT_HEADERS = {"User-Agent": "GitHubCopilotChat/0.35.0", "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0", "Copilot-Integration-Id": "vscode-chat"}
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
# Sign in with ChatGPT: public client, the issued client ID arrives in the callback query.
CHATGPT_DYNAMIC_CLIENT_ID = "dynamic_agent_client"
CHATGPT_AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
CHATGPT_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
CHATGPT_RESOURCE = "https://api.openai.com/v1"
CHATGPT_DIRECT_TOKEN_SCOPE = "chatgpt.tokens.use.direct"
CHATGPT_SCOPE = f"openid profile email offline_access resource.invoke {CHATGPT_DIRECT_TOKEN_SCOPE}"
CHATGPT_CALLBACK_PORT = 1455
CHATGPT_CALLBACK_PATH = "/auth/callback"
CHATGPT_REDIRECT_URI = f"http://127.0.0.1:{CHATGPT_CALLBACK_PORT}{CHATGPT_CALLBACK_PATH}"
# Refresh this long before the real expiry so a request never starts with a token about to expire.
CHATGPT_EXPIRY_MARGIN_SECONDS = 180
KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
KIMI_DEVICE_TIMEOUT_SECONDS = 15 * 60
KIMI_REFRESH_RETRIES = 3
# Radius is a pi-messages gateway; only the interactive authorization endpoint is discovered.
RADIUS_CLIENT_ID = "pi-gateway"
RADIUS_SCOPE = "gateway offline_access"
RADIUS_DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
RADIUS_REDIRECT_URI = "http://127.0.0.1:1456/oauth/callback"
RADIUS_TOKEN_SKEW_SECONDS = 60
DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"
XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
XAI_REFRESH_SKEW_SECONDS = 300
XAI_DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


def checked(response):
    if response.status_code >= 400:
        raise ValueError(f"Subscription authentication failed (HTTP {response.status_code})")
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid subscription authentication response")
    return data


def account_id(token: str) -> str:
    try:
        payload = token.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        value = data["https://api.openai.com/auth"]["chatgpt_account_id"]
        if not isinstance(value, str) or not value:
            raise ValueError()
        return value
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("Codex token has no account identifier") from exc


def token_record(data, provider, previous=None):
    previous = previous or {}
    access = data.get("access_token")
    refresh = data.get("refresh_token", previous.get("refresh"))
    expires = data.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise ValueError("OAuth response is missing access/refresh tokens")
    if type(expires) not in (int, float) or not math.isfinite(expires) or expires <= 0:
        raise ValueError("Invalid OAuth token expiry")
    record = {"type": "oauth", "access": access, "refresh": refresh, "expires": time.time() + expires}
    if provider == "radius":
        record["expires"] -= RADIUS_TOKEN_SKEW_SECONDS
    if provider == "openai-codex":
        record["account_id"] = account_id(access)
    return record


def copilot_endpoint(token):
    match = re.search(r"(?:^|;)proxy-ep=([^;]+)", token)
    host = match.group(1).replace("proxy.", "api.", 1) if match else "api.individual.githubcopilot.com"
    if not host.endswith(".githubcopilot.com") or not all(c.isalnum() or c in ".-" for c in host):
        raise ValueError("Invalid Copilot proxy endpoint")
    return "https://" + host


async def copilot_exchange(github_token):
    async with httpx.AsyncClient(timeout=30) as client:
        data = checked(await client.get("https://api.github.com/copilot_internal/v2/token", headers={
            "Authorization": "Bearer " + github_token, "Accept": "application/json", **COPILOT_HEADERS}))
    if not isinstance(data.get("token"), str) or not data["token"] or type(data.get("expires_at")) not in (int, float):
        raise ValueError("Invalid Copilot token response")
    return {"type": "oauth", "access": data["token"], "refresh": github_token,
        "expires": data["expires_at"], "base_url": copilot_endpoint(data["token"])}


async def refresh_subscription(provider, credential):
    if provider == "github-copilot":
        return await copilot_exchange(credential["refresh"])
    if provider == "openai":
        return await refresh_chatgpt(credential)
    if provider == "kimi-coding":
        return await refresh_kimi(credential)
    if provider == "xai":
        return await refresh_xai(credential)
    if provider == "radius":
        return await refresh_radius(credential)
    url, client_id = ("https://auth.openai.com/oauth/token", CODEX_CLIENT_ID) if provider == "openai-codex" else (
        "https://platform.claude.com/v1/oauth/token", CLAUDE_CLIENT_ID)
    body = {"grant_type": "refresh_token", "client_id": client_id, "refresh_token": credential["refresh"]}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, **({"data": body} if provider == "openai-codex" else {"json": body}))
    return token_record(checked(response), provider, credential)


async def login_claude(notify, prompt):
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    redirect = "https://platform.claude.com/oauth/code/callback"
    notify("https://claude.ai/oauth/authorize?" + urlencode({"code": "true", "client_id": CLAUDE_CLIENT_ID,
        "response_type": "code", "redirect_uri": redirect, "scope": "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload",
        "code_challenge": challenge, "code_challenge_method": "S256", "state": verifier}))
    value = (await prompt("Paste Claude authorization code (code#state) or redirect URL: ")).strip()
    state = verifier
    if "://" in value:
        params = parse_qs(urlsplit(value).query)
        code, state = params.get("code", [""])[0], params.get("state", [""])[0]
    elif "#" in value:
        code, state = value.split("#", 1)
    else:
        code = value
    if state != verifier or not code:
        raise ValueError("Missing authorization code or OAuth state mismatch")
    async with httpx.AsyncClient(timeout=30) as client:
        data = checked(await client.post("https://platform.claude.com/v1/oauth/token", json={
            "grant_type": "authorization_code", "client_id": CLAUDE_CLIENT_ID, "code": code,
            "state": verifier, "code_verifier": verifier, "redirect_uri": redirect}))
    return token_record(data, "anthropic")


async def login_codex(notify):
    async with httpx.AsyncClient(timeout=30) as client:
        device = checked(await client.post("https://auth.openai.com/api/accounts/deviceauth/usercode", json={"client_id": CODEX_CLIENT_ID}))
        interval = max(1, float(device.get("interval", 5)))
        if not math.isfinite(interval):
            raise ValueError("Invalid device polling interval")
        notify("https://auth.openai.com/codex/device  code: " + device["user_code"])
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            response = await client.post("https://auth.openai.com/api/accounts/deviceauth/token", json={
                "device_auth_id": device["device_auth_id"], "user_code": device["user_code"]})
            if response.status_code in (403, 404):
                continue
            data = response.json()
            error = data.get("error")
            error = error.get("code") if isinstance(error, dict) else error
            if error in ("deviceauth_authorization_pending", "authorization_pending"):
                continue
            if error == "slow_down":
                interval += 5
                continue
            data = checked(response)
            result = checked(await client.post("https://auth.openai.com/oauth/token", data={
                "grant_type": "authorization_code", "client_id": CODEX_CLIENT_ID,
                "code": data["authorization_code"], "code_verifier": data["code_verifier"],
                "redirect_uri": "https://auth.openai.com/deviceauth/callback"}))
            return token_record(result, "openai-codex")
    raise ValueError("Codex device login expired")


async def login_copilot(notify):
    async with httpx.AsyncClient(timeout=30, headers={"Accept": "application/json", **COPILOT_HEADERS}) as client:
        device = checked(await client.post("https://github.com/login/device/code", data={"client_id": COPILOT_CLIENT_ID, "scope": "read:user"}))
        notify(device["verification_uri"] + "  code: " + device["user_code"])
        interval = max(1, float(device.get("interval", 5)))
        deadline = time.monotonic() + float(device["expires_in"])
        while time.monotonic() < deadline:
            await asyncio.sleep(interval)
            data = checked(await client.post("https://github.com/login/oauth/access_token", data={
                "client_id": COPILOT_CLIENT_ID, "device_code": device["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}))
            if data.get("error") == "authorization_pending":
                continue
            if data.get("error") == "slow_down":
                interval += 5
                continue
            if not data.get("access_token"):
                raise ValueError("GitHub device authorization denied or expired")
            return await copilot_exchange(data["access_token"])
    raise ValueError("GitHub device login expired")


def get_or_create_device_id(agent_dir):
    """OpenAI identifies each installation ("agent host") by a stable UUID persisted in the agent directory."""
    path = agent_dir / "device-id"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if UUID_PATTERN.match(value):
        return value
    value = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def oauth_json(response):
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def oauth_error_detail(data):
    parts = [data[key] for key in ("error", "error_description") if isinstance(data.get(key), str) and data[key]]
    return ", ".join(parts)


def trusted_url(value, https_only=False):
    """The verification URI is opened in the user's browser; only http(s) URLs are trusted."""
    if not isinstance(value, str) or not value:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in (("https",) if https_only else ("https", "http")) or not parsed.hostname:
        return None
    return value


def positive_number(value, default=None):
    return value if type(value) in (int, float) and math.isfinite(value) and value > 0 else default


def chatgpt_record(data, client_id):
    access, refresh, expires, scope = (data.get(key) for key in ("access_token", "refresh_token", "expires_in", "scope"))
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise ValueError("OpenAI OAuth token response has invalid access/refresh token")
    if not positive_number(expires):
        raise ValueError("OpenAI OAuth token response has invalid expires_in")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("OpenAI OAuth token response has invalid scope")
    scopes = scope.split()
    if CHATGPT_DIRECT_TOKEN_SCOPE not in scopes:
        raise ValueError(f"OpenAI OAuth grant did not include {CHATGPT_DIRECT_TOKEN_SCOPE}")
    return {"type": "oauth", "access": access, "refresh": refresh, "expires": time.time() + expires - CHATGPT_EXPIRY_MARGIN_SECONDS,
        "client_id": client_id, "scopes": scopes}


async def refresh_chatgpt(credential):
    client_id = credential.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(CHATGPT_TOKEN_URL, data={"grant_type": "refresh_token", "client_id": client_id,
            "refresh_token": credential["refresh"], "resource": CHATGPT_RESOURCE})
    return chatgpt_record(checked(response), client_id)


def chatgpt_callback_result(params, state):
    if params.get("state") != state:
        raise ValueError("OAuth state mismatch")
    if not params.get("code"):
        raise ValueError("Missing authorization code")
    client_id = (params.get("client_id") or "").strip()
    if not client_id:
        raise ValueError("OpenAI OAuth registration callback did not contain an issued client ID")
    return {"code": params["code"], "client_id": client_id}


def chatgpt_manual_result(value, state):
    try:
        url = urlsplit(value.strip())
    except ValueError as exc:
        raise ValueError("Paste the full callback URL from the browser") from exc
    if not url.scheme or not url.hostname:
        raise ValueError("Paste the full callback URL from the browser")
    expected = urlsplit(CHATGPT_REDIRECT_URI)
    if (url.scheme, url.hostname, url.port, url.path) != (expected.scheme, expected.hostname, expected.port, expected.path):
        raise ValueError(f"The pasted callback URL must start with {CHATGPT_REDIRECT_URI}")
    params = {key: values[0] for key, values in parse_qs(url.query).items() if values}
    if params.get("error"):
        raise ValueError(f"ChatGPT authorization failed: {params['error']}")
    return chatgpt_callback_result(params, state)


async def login_openai_chatgpt(notify, prompt, device_id):
    if not UUID_PATTERN.match(device_id or ""):
        raise ValueError("Sign in with ChatGPT requires a device ID (UUID) for this installation")
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    authorize = CHATGPT_AUTHORIZE_URL + "?" + urlencode({
        "client_id": CHATGPT_DYNAMIC_CLIENT_ID, "agent_name_hint": "Pi", "ext_agent_host_id": f"urn:uuid:{device_id.lower()}",
        "response_type": "code", "redirect_uri": CHATGPT_REDIRECT_URI, "resource": CHATGPT_RESOURCE, "scope": CHATGPT_SCOPE,
        "state": state, "code_challenge": challenge, "code_challenge_method": "S256", "nonce": secrets.token_urlsafe(32)})
    # A fixed port: without this server the callback would reach whatever else holds the port (another pending
    # login or the Codex CLI), which rejects it as a state mismatch. Fail with a clear error instead.
    try:
        callback = await start_oauth_callback_server("OpenAI", "127.0.0.1", CHATGPT_CALLBACK_PORT, CHATGPT_CALLBACK_PATH, state=state)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        raise ValueError("Port 1455 is in use, probably by an unfinished login in another pi session or by the Codex CLI. "
            "Cancel that login and try again.") from exc
    notify(authorize)
    notify("Complete sign-in in your browser. If the callback does not complete, paste the final redirect URL here.")
    # Race the loopback callback against the pasted redirect URL, for when the browser cannot reach 127.0.0.1 (SSH).
    manual = asyncio.ensure_future(prompt("Complete login in your browser, or paste the final redirect URL here: "))
    try:
        pending = {asyncio.ensure_future(callback.wait()), manual}
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if manual in done:
            result = chatgpt_manual_result(manual.result(), state)
        else:
            result = chatgpt_callback_result(next(iter(done)).result(), state)
        notify("Exchanging authorization code for tokens...")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(CHATGPT_TOKEN_URL, data={"grant_type": "authorization_code",
                "client_id": result["client_id"], "code": result["code"], "code_verifier": verifier,
                "redirect_uri": CHATGPT_REDIRECT_URI, "resource": CHATGPT_RESOURCE})
        data = checked(response)
        # Pi does not use the ID token to identify the user; keep the presence check as part of the contract.
        if not isinstance(data.get("id_token"), str) or not data["id_token"].strip():
            raise ValueError("OpenAI OAuth token response did not contain an ID token")
        return chatgpt_record(data, result["client_id"])
    finally:
        manual.cancel()
        await callback.close()


def kimi_oauth_host():
    override = os.environ.get("KIMI_CODE_OAUTH_HOST") or os.environ.get("KIMI_OAUTH_HOST")
    return (override or KIMI_DEFAULT_OAUTH_HOST).rstrip("/")


async def login_kimi_coding(notify):
    host = kimi_oauth_host()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(host + "/api/oauth/device_authorization", data={"client_id": KIMI_CLIENT_ID})
        if response.status_code >= 400:
            raise ValueError(f"Kimi Code device authorization failed with status {response.status_code}")
        device = oauth_json(response)
        verification, complete = trusted_url(device.get("verification_uri")), trusted_url(device.get("verification_uri_complete"))
        if not isinstance(device.get("device_code"), str) or not device["device_code"] or not isinstance(device.get("user_code"), str) \
                or not device["user_code"] or not verification or not complete:
            raise ValueError("Invalid Kimi Code device authorization response")
        notify(complete + "  code: " + device["user_code"])
        async def poll():
            token = await client.post(host + "/api/oauth/token", data={"client_id": KIMI_CLIENT_ID, "device_code": device["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
            if token.status_code >= 500:
                return {"status": "failed", "message": f"Kimi Code device token request failed with status {token.status_code}"}
            data = oauth_json(token)
            if token.status_code < 400 and isinstance(data.get("access_token"), str) and data["access_token"]:
                try:
                    return {"status": "complete", "value": token_record(data, "kimi-coding")}
                except ValueError as exc:
                    return {"status": "failed", "message": str(exc)}
            error = data.get("error")
            if error == "authorization_pending":
                return {"status": "pending"}
            if error == "slow_down":
                return {"status": "slow_down", "interval_seconds": positive_number(data.get("interval"))}
            if error == "expired_token":
                return {"status": "failed", "message": "Kimi Code device authorization expired. Please restart login."}
            if error == "access_denied":
                return {"status": "failed", "message": "Kimi Code login was denied."}
            detail = f": {error}{', ' + data['error_description'] if isinstance(data.get('error_description'), str) else ''}" if isinstance(error, str) else ""
            return {"status": "failed", "message": f"Kimi Code device token request failed (status {token.status_code}){detail}"}
        return await poll_oauth_device_code_flow(poll, interval_seconds=positive_number(device.get("interval"), 5),
            expires_in_seconds=positive_number(device.get("expires_in"), KIMI_DEVICE_TIMEOUT_SECONDS), wait_before_first_poll=True)


async def refresh_kimi(credential):
    host = kimi_oauth_host()
    last = None
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(KIMI_REFRESH_RETRIES + 1):
            if attempt:
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                response = await client.post(host + "/api/oauth/token", data={"client_id": KIMI_CLIENT_ID,
                    "grant_type": "refresh_token", "refresh_token": credential["refresh"]})
            except (httpx.HTTPError, OSError) as exc:
                last = exc
                continue
            data = oauth_json(response)
            if response.status_code < 400:
                return token_record(data, "kimi-coding", credential)
            if response.status_code in (401, 403) or data.get("error") == "invalid_grant":
                description = f": {data['error_description']}" if isinstance(data.get("error_description"), str) else ""
                raise ValueError(f"Kimi Code token refresh unauthorized (status {response.status_code}){description}")
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < KIMI_REFRESH_RETRIES:
                    last = ValueError(f"Kimi Code token refresh failed with status {response.status_code}")
                    continue
                raise ValueError(f"Kimi Code token refresh failed with status {response.status_code}")
            raise ValueError(f"Kimi Code token refresh failed with status {response.status_code}: {oauth_error_detail(data) or response.status_code}")
    raise last if isinstance(last, Exception) else ValueError("Kimi Code token refresh failed")


def xai_record(data, previous=None):
    access = data.get("access_token")
    if not isinstance(access, str) or not access:
        raise ValueError("Invalid xAI OAuth response field: access_token")
    if "refresh_token" in data:
        refresh = data["refresh_token"]
    elif previous:
        # xAI may omit refresh_token on refresh when the token is not rotated.
        refresh = previous.get("refresh")
    else:
        refresh = None
    if not isinstance(refresh, str) or not refresh:
        raise ValueError("Invalid xAI OAuth response field: refresh_token")
    expires = data.get("expires_in", XAI_DEFAULT_TOKEN_LIFETIME_SECONDS)
    if not positive_number(expires):
        raise ValueError("Invalid xAI OAuth response field: expires_in")
    return {"type": "oauth", "access": access, "refresh": refresh, "expires": time.time() + expires - XAI_REFRESH_SKEW_SECONDS}


def xai_error(action, response):
    return ValueError(f"xAI OAuth {action} failed (HTTP {response.status_code}): {oauth_error_detail(oauth_json(response))}")


async def login_xai(notify):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(XAI_DEVICE_CODE_URL, data={"client_id": XAI_CLIENT_ID, "scope": XAI_SCOPE, "referrer": "pi"})
        if response.status_code >= 400:
            raise xai_error("device authorization", response)
        device = oauth_json(response)
        verification = trusted_url(device.get("verification_uri"), https_only=True)
        if verification is None or not isinstance(device.get("device_code"), str) or not device["device_code"] \
                or not isinstance(device.get("user_code"), str) or not device["user_code"] or not positive_number(device.get("expires_in")):
            raise ValueError("Invalid xAI OAuth response field")
        complete = trusted_url(device.get("verification_uri_complete"), https_only=True)
        notify((complete or verification) + "  code: " + device["user_code"])
        async def poll():
            token = await client.post(XAI_TOKEN_URL, data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID, "device_code": device["device_code"]})
            body = oauth_json(token)
            if token.status_code < 400:
                return {"status": "complete", "value": xai_record(body)}
            error = body.get("error")
            if error == "authorization_pending":
                return {"status": "pending"}
            if error == "slow_down":
                return {"status": "slow_down", "interval_seconds": positive_number(body.get("interval"))}
            if error in ("access_denied", "authorization_denied"):
                return {"status": "failed", "message": "xAI device authorization was denied"}
            if error == "expired_token":
                return {"status": "failed", "message": "xAI device code expired"}
            return {"status": "failed", "message": str(xai_error("device token polling", token))}
        return await poll_oauth_device_code_flow(poll, interval_seconds=positive_number(device.get("interval")),
            expires_in_seconds=device["expires_in"], wait_before_first_poll=True)


async def refresh_xai(credential):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(XAI_TOKEN_URL, data={"grant_type": "refresh_token", "client_id": XAI_CLIENT_ID,
            "refresh_token": credential["refresh"]})
    if response.status_code >= 400:
        raise xai_error("token refresh", response)
    return xai_record(oauth_json(response), credential)


def normalize_radius_gateway(value):
    value = value or DEFAULT_RADIUS_GATEWAY
    if not re.match(r"^https?://", value):
        value = "https://" + value
    return value.rstrip("/")


class RadiusOAuthError(ValueError):
    def __init__(self, message, status, oauth_error=None):
        super().__init__(message)
        self.status = status
        self.oauth_error = oauth_error


async def radius_token(gateway, fields):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(gateway + "/v1/oauth/token", data=fields)
    if response.status_code >= 400:
        oauth_error = description = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            oauth_error = body.get("error") if isinstance(body.get("error"), str) else None
            description = body.get("error_description") if isinstance(body.get("error_description"), str) else None
        else:
            description = response.text or None
        detail = f"{oauth_error}: {description}" if oauth_error and description else (oauth_error or description or str(response.status_code))
        raise RadiusOAuthError(f"Radius OAuth token request failed: {detail}", response.status_code, oauth_error)
    body = checked(response)
    record = token_record(body, "radius")
    record["scope"] = body.get("scope")
    record["gateway"] = gateway
    return record


async def login_radius(notify, prompt, gateway=None):
    gateway = normalize_radius_gateway(gateway or os.environ.get("RADIUS_GATEWAY"))
    method = (await prompt("Sign in to Radius: press Enter for browser, or type 'device' to sign in from another device: ")).strip().lower()
    if method in ("", "browser"):
        return await radius_login_browser(notify, gateway)
    if method in ("device", "device-code"):
        return await radius_login_device(notify, gateway)
    raise ValueError(f"Unknown Radius sign-in method: {method}")


async def radius_login_browser(notify, gateway):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(gateway + "/v1/oauth", headers={"accept": "application/json"})
        if response.status_code >= 400:
            raise ValueError(f"Could not load Radius OAuth config from {gateway}: {response.status_code} {response.text}")
        endpoint = oauth_json(response).get("authorizationEndpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(f"Invalid Radius OAuth config from {gateway}")
    verifier, challenge = generate_pkce()
    state = str(uuid.uuid4())
    authorize = endpoint + ("&" if "?" in endpoint else "?") + urlencode({
        "response_type": "code", "client_id": RADIUS_CLIENT_ID, "redirect_uri": RADIUS_REDIRECT_URI, "scope": RADIUS_SCOPE,
        "code_challenge": challenge, "code_challenge_method": "S256", "handoff": "url", "state": state})
    async def complete(code):
        return await radius_token(gateway, {"grant_type": "authorization_code", "client_id": RADIUS_CLIENT_ID,
            "redirect_uri": RADIUS_REDIRECT_URI, "code": code, "code_verifier": verifier})
    callback = await start_oauth_callback_server("Radius", "127.0.0.1", 1456, "/oauth/callback", state=state, complete=complete)
    notify("Listening for OAuth callback on " + RADIUS_REDIRECT_URI)
    notify(authorize)
    try:
        credential = await callback.wait()
        if credential is None:
            raise ValueError("OAuth callback did not complete.")
        return credential
    finally:
        await callback.close()


async def radius_login_device(notify, gateway):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(gateway + "/v1/oauth/device", data={"client_id": RADIUS_CLIENT_ID, "scope": RADIUS_SCOPE})
        if response.status_code >= 400:
            raise RadiusOAuthError(f"Radius OAuth device authorization failed: {response.status_code} {response.text}", response.status_code)
        device = oauth_json(response)
        if not isinstance(device.get("device_code"), str) or not device["device_code"] or not isinstance(device.get("user_code"), str) \
                or not device["user_code"] or not isinstance(device.get("verification_uri"), str) or not device["verification_uri"] \
                or not positive_number(device.get("expires_in")):
            raise ValueError("Radius OAuth device authorization response is missing required fields")
        notify(device["verification_uri"] + "  code: " + device["user_code"])
        async def poll():
            try:
                return {"status": "complete", "value": await radius_token(gateway, {"grant_type": RADIUS_DEVICE_CODE_GRANT,
                    "client_id": RADIUS_CLIENT_ID, "device_code": device["device_code"]})}
            except RadiusOAuthError as exc:
                if exc.oauth_error == "authorization_pending":
                    return {"status": "pending"}
                if exc.oauth_error == "slow_down":
                    return {"status": "slow_down"}
                if exc.oauth_error == "expired_token":
                    return {"status": "failed", "message": "Device authorization expired."}
                if exc.oauth_error == "access_denied":
                    return {"status": "failed", "message": "Device authorization was denied."}
                raise
        return await poll_oauth_device_code_flow(poll, interval_seconds=positive_number(device.get("interval")),
            expires_in_seconds=device["expires_in"])


async def refresh_radius(credential):
    gateway = credential.get("gateway")
    if not isinstance(gateway, str) or not gateway:
        raise ValueError("Radius credential has no gateway; log in again")
    return await radius_token(normalize_radius_gateway(gateway), {"grant_type": "refresh_token",
        "client_id": RADIUS_CLIENT_ID, "refresh_token": credential["refresh"]})

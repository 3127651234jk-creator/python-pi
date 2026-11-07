# Azure Responses and Radius

Azure Responses uses provider/API `azure-openai-responses`:

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://RESOURCE.openai.azure.com"
$env:AZURE_OPENAI_API_KEY = "..."
python -m pi --provider azure-openai-responses --model DEPLOYMENT "hello"
```

Alternatively set `AZURE_OPENAI_BASE_URL` or `AZURE_OPENAI_RESOURCE_NAME`.
Root endpoints are normalized to `/openai/v1`; requests use `/responses` and
`AZURE_OPENAI_API_VERSION` (default `v1`). Existing `AZURE_ACCESS_TOKEN`
credentials use Bearer authentication. This does not acquire Entra tokens.
Model `params.deployment_name` separates a configured model ID from its
deployment; `params.api_version` overrides the environment API version.
Responses text, reasoning replay, tool calls and usage use the shared adapter.

Radius uses provider `radius`, API `pi-messages`:

```powershell
$env:RADIUS_API_KEY = "..."
python -m pi --provider radius --model MODEL_ID "hello"
```

Existing Radius login credentials also work. Model discovery fetches the
gateway catalog from `/v1/config`: at startup, after Radius login, and on
`--sync-models` / `/sync-models` (skipped with `PI_OFFLINE` or
`sync_catalog: false`; failures fall back to the cached catalog in
`radius-cache.json`, and `RADIUS_GATEWAY` overrides the gateway). Discovered
models carry the gateway-advertised `baseUrl`, capability flags, context and
output limits, costs, and thinking-level map; model IDs configured in
`models.json` keep their explicit settings.

Custom providers can set `api` to `pi-messages` with a `base_url` and model list.
The adapter sends `{model, context, options}` to `/messages`, serializes the
transcript with TS camelCase fields (including text and thinking signatures,
provider thinking levels and diagnostics), and accepts text, thinking, tool
calls, usage, response IDs, gateway rewrite diagnostics and terminal
done/error events; unknown event types are forwarded instead of failing.
Gateway costs are preserved. Missing terminal events produce an error while
preserving partial content. Cache TTLs map `5m` to `short` and `1h` to `long`,
with the legacy `PI_CACHE_RETENTION=long` opt-in; `sessionId`, `toolChoice`
and `debug` (`?debug=1`) request options are supported, plus the TS
`onPayload` / `onResponse` / `onProviderStreamEvent` hooks. Non-2xx responses
surface a `pi_messages_response_failure` diagnostic on the error message.

Limitations: the shared driver has no per-request timeout/retry controls, and
a stream that ends without a terminal event reports the generic driver error
message rather than the TS wording. Verification uses offline HTTP/SSE
fixtures; no real Azure or Radius account requests have been performed.

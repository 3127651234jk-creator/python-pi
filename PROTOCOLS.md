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

Existing Radius login credentials also work. Specify a model ID or configure
models in `models.json`; automatic Radius gateway model discovery is not
implemented. Configure accurate capability and context limits for your model.

Custom providers can set `api` to `pi-messages` with a `base_url` and model list.
The adapter sends `{model, context, options}` to `/messages`, serializes the
transcript with TS camelCase fields, and accepts text, thinking, tool calls,
usage, response IDs and terminal done/error events. Gateway costs are preserved.
Missing terminal events produce an error while preserving partial content.
Cache TTLs map `5m` to `short` and `1h` to `long`.

Limitations: gateway rewrite diagnostics, text signatures, provider thinking
metadata, debug/session/tool-choice options and full TS sampling controls are
not represented by this adapter. Verification uses offline HTTP/SSE fixtures;
no real Azure or Radius account requests have been performed.

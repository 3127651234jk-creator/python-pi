"""Custom API model login: form validation, candidate configuration and save.

Terminal-independent core of the "Add custom API model" login wizard
(docs/specs/custom-api-login.md). The wizard drives a small :class:`WizardIO`
surface so the enhanced TUI and the plain REPL share identical validation,
cancel and save rules. Saving reuses the existing adapters and atomic JSON
writer; no new request protocol is introduced.
"""

import copy
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlsplit

from .config import Configuration, read_json, write_json
from .model_registry import ModelRegistry

# UI label -> models.json "api" value, in wizard display order.
API_FORMATS: Tuple[Tuple[str, str], ...] = (
    ("openai-completions", "OpenAI Chat Completions"),
    ("anthropic-messages", "Claude Messages"),
    ("openai-responses", "OpenAI Responses"),
)
# Path suffix each adapter appends to the base URL (mirrors the providers).
API_PATH_SUFFIXES = {
    "openai-completions": "/chat/completions",
    "anthropic-messages": "/v1/messages",
    "openai-responses": "/responses",
}
PROVIDER_ID_RE = re.compile(r"[a-z][a-z0-9-]{0,63}")
# "custom" is reserved for this login entry (/login custom, --login custom).
RESERVED_PROVIDER_ID = "custom"
DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_MAX_TOKENS = 4096
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ConcurrentConfigChange(Exception):
    """The on-disk configuration changed while the wizard was open."""


class SaveError(Exception):
    """Persistence failed; the message names the file that needs repair."""


class Cancelled(Exception):
    """The user cancelled the wizard (Escape, Ctrl+C, EOF or :cancel)."""


@dataclass
class CustomApiForm:
    api: str = "openai-completions"
    provider_id: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model_id: str = ""
    display_name: str = ""
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    image_input: bool = False
    reasoning: bool = False


class WizardIO(Protocol):
    """Async UI surface the wizard drives; implemented per terminal mode."""

    async def choose(self, title: str, values: List[Tuple[str, str]], default: Optional[str] = None) -> Optional[str]: ...

    async def text_input(self, title: str, default: str = "", password: bool = False, allow_cancel_word: bool = True,
                         validate: Optional[Callable[[str], str]] = None) -> Optional[str]: ...

    async def confirm(self, title: str, default: bool) -> Optional[bool]: ...

    def say(self, message: str) -> None: ...


def display_name(form: CustomApiForm) -> str:
    return form.display_name or form.model_id


def validate_api(value: str) -> str:
    if value not in API_PATH_SUFFIXES:
        raise ValueError("Choose one of the three API formats")
    return value


def validate_provider_id(value: str, existing_providers, config: Configuration) -> str:
    provider_id = value.strip()
    if not provider_id:
        raise ValueError("Provider ID is required")
    if not PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError("Provider ID must match ^[a-z][a-z0-9-]{0,63}$: lowercase letters, digits and hyphens, starting with a letter")
    if provider_id == RESERVED_PROVIDER_ID:
        raise ValueError("provider ID 'custom' is reserved for this login entry; pick another ID")
    if provider_id in existing_providers:
        raise ValueError(f"a provider named '{provider_id}' already exists; choose another ID")
    try:
        stored = read_json(config.agent_dir / "auth.json")
    except ValueError as exc:
        raise ValueError(f"{exc}; repair it before adding a provider") from exc
    if provider_id in stored:
        raise ValueError(f"credentials for '{provider_id}' exist without a model configuration; run /logout {provider_id} to remove them, then retry")
    return provider_id


def validate_base_url(value: str) -> str:
    url = value.strip()
    if not url:
        raise ValueError("Base URL is required")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Base URL must be an absolute http(s) address")
    if not parsed.hostname:
        raise ValueError("Base URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("Base URL must not contain a username or password")
    if parsed.query:
        raise ValueError("Base URL must not contain query parameters")
    if parsed.fragment:
        raise ValueError("Base URL must not contain a fragment")
    return url.rstrip("/")


def is_local_base_url(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in LOCAL_HOSTS


def is_plain_http(url: str) -> bool:
    return urlsplit(url).scheme == "http"


@dataclass
class SuffixCheck:
    matching_api: Optional[str] = None  # protocol whose full endpoint the URL looks like
    suggested_root: Optional[str] = None  # root address to use when the suffix matches


def check_endpoint_suffix(url: str, api: str) -> SuffixCheck:
    """Recognize pasted full endpoints and protocol/suffix mismatches."""
    for known_api, suffix in API_PATH_SUFFIXES.items():
        if len(url) > len(suffix) and url.endswith(suffix):
            if known_api == api:
                return SuffixCheck(suggested_root=url[: -len(suffix)].rstrip("/") or None)
            return SuffixCheck(matching_api=known_api)
    if api == "anthropic-messages" and url.endswith("/v1"):
        raise ValueError(f"the final request URL would be {url}/v1/messages; remove the /v1 suffix or choose a different API format")
    return SuffixCheck()


def endpoint_preview(api: str, base_url: str) -> str:
    return base_url.rstrip("/") + API_PATH_SUFFIXES[api]


def validate_api_key(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("API key must not contain line breaks")
    key = value.strip()
    if not key:
        raise ValueError("API key is required")
    return key


def validate_model_id(value: str) -> str:
    model_id = value.strip()
    if not model_id:
        raise ValueError("Model ID is required")
    if any(ord(char) < 32 or ord(char) == 127 for char in model_id):
        raise ValueError("Model ID must not contain control characters")
    return model_id


def validate_display_name(value: str, fallback: str) -> str:
    name = value.strip()
    if not name:
        name = fallback.strip()
    if not name:
        raise ValueError("Display name is required")
    return name


def validate_positive_int(value: str, label: str) -> int:
    text = value.strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(text)


def validate_limits(context_window: int, max_tokens: int) -> None:
    if context_window <= 0 or max_tokens <= 0:
        raise ValueError("Context window and max output tokens must be positive integers")
    if context_window <= max_tokens:
        raise ValueError("Context window must be greater than max output tokens")


@dataclass
class FormValidation:
    """Whole-form result: the normalized form, or one error per bad field."""

    form: Optional[CustomApiForm]
    errors: Dict[str, str]
    suggestion: Optional[SuffixCheck] = None
    request_url: Optional[str] = None


def form_values(form: CustomApiForm) -> Dict[str, object]:
    """Raw values of an already-parsed form, for whole-form validation."""
    return {
        "api": form.api,
        "provider_id": form.provider_id,
        "base_url": form.base_url,
        "api_key": form.api_key,
        "model_id": form.model_id,
        "display_name": form.display_name,
        "context_window": str(form.context_window),
        "max_tokens": str(form.max_tokens),
        "image_input": form.image_input,
        "reasoning": form.reasoning,
    }


def validate_form_values(values: Dict[str, object], existing_providers, config: Configuration) -> FormValidation:
    """Validate every raw string as one unit, without prompting.

    The single-page form calls this after each edit and on save; the text
    wizard uses the same underlying field rules. Cross-field checks live here
    so both surfaces agree. No network access and no writes.
    """
    errors: Dict[str, str] = {}

    api = "openai-completions"
    try:
        api = validate_api(str(values.get("api", "")))
    except ValueError as exc:
        errors["api"] = str(exc)

    provider_id = ""
    try:
        provider_id = validate_provider_id(str(values.get("provider_id", "")), existing_providers, config)
    except ValueError as exc:
        errors["provider_id"] = str(exc)

    base_url = ""
    request_url = None
    suggestion = None
    try:
        base_url = validate_base_url(str(values.get("base_url", "")))
    except ValueError as exc:
        errors["base_url"] = str(exc)
    else:
        if "api" not in errors:
            try:
                suggestion = check_endpoint_suffix(base_url, api)
            except ValueError as exc:
                errors["base_url"] = str(exc)
            else:
                if suggestion.matching_api is not None:
                    errors["base_url"] = (
                        f"This looks like a {dict(API_FORMATS)[suggestion.matching_api]} endpoint; "
                        "choose that API format or change the address")
                elif suggestion.suggested_root is not None:
                    errors["base_url"] = (
                        f"This looks like a full endpoint; use the root {suggestion.suggested_root}")
        if "api" not in errors:
            request_url = endpoint_preview(api, base_url)

    api_key = ""
    try:
        api_key = validate_api_key(str(values.get("api_key", "")))
    except ValueError as exc:
        errors["api_key"] = str(exc)

    model_id = ""
    try:
        model_id = validate_model_id(str(values.get("model_id", "")))
    except ValueError as exc:
        errors["model_id"] = str(exc)

    name = ""
    raw_name = str(values.get("display_name", ""))
    raw_model = str(values.get("model_id", ""))
    try:
        name = validate_display_name(raw_name, raw_model)
    except ValueError as exc:
        # An empty display name falls back to Model ID and is not an error on
        # its own; only a bad explicit value needs its own message.
        if raw_name.strip():
            errors["display_name"] = str(exc)

    context_window = 0
    try:
        context_window = validate_positive_int(str(values.get("context_window", "")), "Context window")
    except ValueError as exc:
        errors["context_window"] = str(exc)
    max_tokens = 0
    try:
        max_tokens = validate_positive_int(str(values.get("max_tokens", "")), "Max output tokens")
    except ValueError as exc:
        errors["max_tokens"] = str(exc)
    if "context_window" not in errors and "max_tokens" not in errors:
        try:
            validate_limits(context_window, max_tokens)
        except ValueError as exc:
            errors["context_window"] = str(exc)

    form = None
    if not errors:
        form = CustomApiForm(
            api=api, provider_id=provider_id, base_url=base_url, api_key=api_key,
            model_id=model_id, display_name=name,
            context_window=context_window, max_tokens=max_tokens,
            image_input=bool(values.get("image_input", False)),
            reasoning=bool(values.get("reasoning", False)),
        )
    return FormValidation(form=form, errors=errors, suggestion=suggestion, request_url=request_url)


def provider_configuration(form: CustomApiForm) -> dict:
    return {"api": form.api, "base_url": form.base_url, "models": [model_configuration(form)]}


def model_configuration(form: CustomApiForm) -> dict:
    return {
        "id": form.model_id,
        "name": display_name(form),
        "context_window": form.context_window,
        "max_tokens": form.max_tokens,
        "input": ["text", "image"] if form.image_input else ["text"],
        "reasoning": form.reasoning,
    }


def build_candidate_models(models_data: dict, form: CustomApiForm) -> dict:
    """models.json content after the merge: other providers, models and
    unknown fields are preserved untouched."""
    candidate = copy.deepcopy(models_data)
    providers = candidate.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("models.json providers must be an object")
    providers[form.provider_id] = provider_configuration(form)
    return candidate


def validate_candidate(candidate: dict, form: CustomApiForm) -> None:
    """Load the candidate models.json from a scratch directory and resolve the
    new model, reusing the registry's full validation. No network, no writes
    to the real configuration."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        write_json(root / "models.json", candidate)
        registry = ModelRegistry(Configuration(str(root), str(root)))
        registry.resolve(form.provider_id, form.model_id, api_key="candidate-check")


def build_review_text(form: CustomApiForm) -> str:
    labels = dict(API_FORMATS)
    return "\n".join([
        f"API format: {labels[form.api]}",
        f"Provider ID: {form.provider_id}",
        f"Base URL: {form.base_url}",
        f"Request URL: {endpoint_preview(form.api, form.base_url)}",
        f"Model ID: {form.model_id}",
        f"Display name: {display_name(form)}",
        f"Context window: {form.context_window} tokens",
        f"Max output tokens: {form.max_tokens} tokens",
        f"Image input: {'yes' if form.image_input else 'no'}",
        f"Reasoning: {'yes' if form.reasoning else 'no'}",
        "API Key: filled (hidden)",
        "Costs are not configured; usage lines will not show a price.",
        "The connection has not been verified yet.",
    ])


class CustomApiSaver:
    """Limited transaction across auth.json and models.json.

    Read validation snapshot -> re-check on save -> validate the candidate in
    memory -> write key, then models -> reload. On failure the added entries
    are rolled back when the files still match this run's writes.
    """

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.models_path = config.agent_dir / "models.json"
        self.auth_path = config.agent_dir / "auth.json"
        self.models_snapshot = read_json(self.models_path)
        self.auth_snapshot = read_json(self.auth_path)

    def save(self, registry: ModelRegistry, form: CustomApiForm) -> None:
        models_data = read_json(self.models_path)
        auth_data = read_json(self.auth_path)
        if models_data != self.models_snapshot or auth_data != self.auth_snapshot:
            raise ConcurrentConfigChange("models.json or auth.json changed while the wizard was open; review your entries and save again")
        candidate = build_candidate_models(models_data, form)
        validate_candidate(candidate, form)
        auth_written = {**auth_data, form.provider_id: {"type": "api_key", "key": form.api_key}}
        write_json(self.auth_path, auth_written)
        try:
            write_json(self.models_path, candidate)
        except BaseException:
            self._rollback(form, [(self.auth_path, auth_written, auth_data), (self.models_path, None, models_data)])
            raise
        try:
            registry.reload()
        except BaseException:
            self._rollback(form, [(self.auth_path, auth_written, auth_data), (self.models_path, candidate, models_data)])
            raise

    def adopt_current(self) -> None:
        """After a reported concurrent change, take the on-disk state as the
        new baseline once the user re-confirms the save."""
        self.models_snapshot = read_json(self.models_path)
        self.auth_snapshot = read_json(self.auth_path)

    def _rollback(self, form: CustomApiForm, written: List[Tuple[Path, Optional[dict], dict]]) -> None:
        """Restore each file we wrote to its pre-write content, but only while
        it still matches this run's write (never clobber a concurrent change)."""
        problems = []
        for path, data, previous in written:
            if data is None:
                continue
            try:
                current = read_json(path)
            except ValueError:
                problems.append(f"{path} could not be re-read; remove the '{form.provider_id}' entries manually")
                continue
            if current != data:
                problems.append(f"{path} changed after the write; remove the '{form.provider_id}' entries manually")
                continue
            try:
                write_json(path, previous)
            except OSError as exc:
                problems.append(f"{path} could not be restored ({exc}); remove the '{form.provider_id}' entries manually")
        if problems:
            raise SaveError("save failed and could not be rolled back: " + "; ".join(problems))


async def run_custom_api_wizard(config: Configuration, registry: ModelRegistry, io: WizardIO) -> Optional[CustomApiForm]:
    """Collect, review and save one custom provider. Returns the form, or
    None when the user cancelled. Never writes anything before Save."""
    saver = CustomApiSaver(config)
    form = CustomApiForm()
    try:
        while True:
            await _collect_fields(io, form, registry, config)
            while True:
                action = await _review(io, form)
                if action is None or action == "cancel":
                    return None
                if action == "back":
                    break  # re-enter the fields; values kept, key stays in memory
                validation = validate_form_values(form_values(form), registry.providers, config)
                if validation.form is None:
                    # The on-disk configuration drifted since collection (for
                    # example a provider now conflicts); re-enter the fields.
                    for message in validation.errors.values():
                        io.say(f"error: {message}")
                    break
                try:
                    saver.save(registry, form)
                except ConcurrentConfigChange as exc:
                    # Keep the in-memory form and require re-confirmation; the
                    # re-confirmed save merges onto the current on-disk state.
                    io.say(f"warning: {exc}")
                    registry.reload()
                    saver.adopt_current()
                    continue
                return form
    except (Cancelled, KeyboardInterrupt, EOFError):
        return None


async def _review(io: WizardIO, form: CustomApiForm) -> Optional[str]:
    io.say("Review the new custom API model:\n" + build_review_text(form))
    return await io.choose("Review", [("save", "Save"), ("back", "Back"), ("cancel", "Cancel")], default="save")


async def _prompt_validated(io: WizardIO, title: str, current: str, validate: Callable[[str], str],
                            password: bool = False, allow_cancel_word: bool = True) -> str:
    """Re-prompt the same field on validation failure, explaining the reason.

    ``validate`` is handed to the input surface so the terminal dialog can
    reject the value without closing (keeping the text and showing the rule);
    the loop below covers the plain REPL and any other WizardIO implementation.
    """
    while True:
        try:
            raw = await io.text_input(title, current, password=password,
                                      allow_cancel_word=allow_cancel_word, validate=validate)
            if raw is None:
                raise Cancelled()
            return validate(raw)
        except ValueError as exc:
            io.say(f"error: {exc}")


async def _prompt_int(io: WizardIO, title: str, current: int) -> int:
    text = await _prompt_validated(io, title, str(current),
                                   lambda value: str(validate_positive_int(value, title)))
    return int(text)


async def _collect_fields(io: WizardIO, form: CustomApiForm, registry: ModelRegistry, config: Configuration) -> None:
    """One editing round over all fields; Back re-enters here with the values
    kept. Raises Cancelled to unwind on Escape, Ctrl+C, EOF or :cancel."""
    # API format
    value = await io.choose("API format", list(API_FORMATS), default=form.api)
    if value is None:
        raise Cancelled()
    form.api = validate_api(value)
    # Provider ID
    form.provider_id = await _prompt_validated(
        io, "Provider ID", form.provider_id,
        lambda raw: validate_provider_id(raw, registry.providers, config))
    # Base URL
    while True:
        raw = await io.text_input("Base URL", form.base_url)
        if raw is None:
            raise Cancelled()
        try:
            url = validate_base_url(raw)
        except ValueError as exc:
            io.say(f"error: {exc}")
            continue
        try:
            check = check_endpoint_suffix(url, form.api)
        except ValueError as exc:
            io.say(f"error: {exc}")
            continue
        if check.matching_api is not None:
            # A full endpoint of another protocol: never silently replace it.
            other = check.matching_api
            switch = await io.confirm(f"The address looks like a {dict(API_FORMATS)[other]} endpoint. Use that API format instead?", default=True)
            if switch is None:
                raise Cancelled()
            if not switch:
                io.say("error: the address does not match the selected API format; enter a different base URL")
                continue
            form.api = other
            check = check_endpoint_suffix(url, other)
        if check.suggested_root is not None:
            use_root = await io.confirm(f"The address looks like a full endpoint. Use the root {check.suggested_root} instead?", default=True)
            if use_root is None:
                raise Cancelled()
            if use_root:
                url = check.suggested_root
        if is_plain_http(url) and not is_local_base_url(url):
            proceed = await io.confirm("The API key will be sent over an unencrypted connection. Continue anyway?", default=False)
            if proceed is None:
                raise Cancelled()
            if not proceed:
                continue
        form.base_url = url
        break
    # API key: hidden input, never in any history; ":cancel" stays a legal key.
    while True:
        raw = await io.text_input("API Key", "", password=True, allow_cancel_word=False)
        if raw is None:
            raise Cancelled()
        if not raw.strip() and form.api_key:
            break  # review-Back keeps the key in memory; empty input keeps it
        try:
            form.api_key = validate_api_key(raw)
            break
        except ValueError as exc:
            io.say(f"error: {exc}")
    # Model ID and display name
    form.model_id = await _prompt_validated(io, "Model ID", form.model_id, validate_model_id)
    raw = await io.text_input("Display name", display_name(form))
    if raw is None:
        raise Cancelled()
    form.display_name = validate_display_name(raw, form.model_id)
    # Advanced settings
    advanced = await io.confirm("Advanced settings?", default=False)
    if advanced is None:
        raise Cancelled()
    if not advanced:
        return
    io.say("The limits below are client-side settings, not the server's real capabilities; adjust them to the service documentation.")
    while True:
        context_window = await _prompt_int(io, "Context window", form.context_window)
        max_tokens = await _prompt_int(io, "Max output tokens", form.max_tokens)
        try:
            validate_limits(context_window, max_tokens)
            form.context_window, form.max_tokens = context_window, max_tokens
            break
        except ValueError as exc:
            io.say(f"error: {exc}")
    image_input = await io.confirm("Image input?", default=form.image_input)
    if image_input is None:
        raise Cancelled()
    form.image_input = image_input
    reasoning = await io.confirm("Reasoning?", default=form.reasoning)
    if reasoning is None:
        raise Cancelled()
    form.reasoning = reasoning

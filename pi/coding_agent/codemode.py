"""Codemode: organize, compose and execute tool calls through Python code.

Port of the codemode tool from ``packages/codemode`` and the coding-agent
extension: the model writes a Python script that calls the agent's tools
programmatically instead of one tool call at a time. The script runs in a fresh
``python -I`` subprocess that can only reach the outside world through a
JSON-line protocol to this module, which executes tool calls through the
agent's normal pipeline (so validation, hooks, and permission checks apply).
No network/process/timer primitives are provided; isolation is process-level.

Script contract (worker globals, also reachable through ``pi.``):
  tools.<name>(args) / tools["<name>"](args)   call one tool (blocks)
  tools.<name>.start(args) / pi.call(...)      start a call, returns a handle
  wait(handle) / wait_all(*handles) / wait_any(*handles)
                                               compose concurrent calls
  ALL_TOOLS                                    [{name, description}, ...]
  text(value) / image(dataUriOrBlock) / print(...) / console.*
                                               append output items
  exit(value)                                  finish immediately with a value
  store(key, value) / store(key) / load(key)   persist small JSON values
  search_tools(query, {limit?, namespace?})    find tools by topic
  describe_tool(name) / describe_namespace(n)  tool and namespace lookup
  result = ...                                 module variable returned as value
  # @options: {"timeout_ms": .., "max_output_tokens": ..}
                                               optional first line

Store writes are reported to the caller and appended to the session as one
``codemode-store`` custom entry per successful script; failed scripts keep
their output but write nothing. ``store``/``load`` are limited to small values
(256k per value, 1M total of JSON), output to 16M characters / 100k items.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..agent.types import AgentTool, AgentToolResult
from ..ai.types import ImageContent, TextContent
from .session import SessionManager

STORE_TYPE = "codemode-store"
MAX_OUTPUT_CHARS = 16 * 1024 * 1024
MAX_OUTPUT_ITEMS = 100_000
MAX_STORE_VALUE_CHARS = 256 * 1024
MAX_STORE_TOTAL_CHARS = 1024 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_INLINE_BUDGET = 3_000
CHARS_PER_TOKEN = 4
ARGS_PREVIEW_CHARS = 200
ERROR_PREVIEW_CHARS = 500
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}

WORKER_SOURCE = textwrap.dedent('''
    """Codemode worker: runs user code with a threaded tool-call bridge."""
    import concurrent.futures
    import json
    import math
    import os
    import re
    import sys
    import threading
    import traceback

    NL = chr(10)
    MAX_OUTPUT_CHARS = 16 * 1024 * 1024
    MAX_OUTPUT_ITEMS = 100000
    MAX_STORE_VALUE_CHARS = 256 * 1024
    MAX_STORE_TOTAL_CHARS = 1024 * 1024

    IMAGE_SIGNATURES = [
        ("image/png", re.compile(r"^iVBORw0KGg")),
        ("image/jpeg", re.compile(r"^[/]9j[/](?!9)")),
        ("image/gif", re.compile(r"^R0lGOD[dl]h")),
        ("image/webp", re.compile(r"^UklG.{8}RUJQ")),
    ]

    class ScriptExit(Exception):
        def __init__(self, value):
            self.value = value

    def _comparable(text):
        return re.sub(r"[^a-z0-9]", "", text.lower())

    class Call:
        def __init__(self, name, future):
            self.name = name
            self.future = future

        def done(self):
            return self.future.done()

        def result(self, timeout=None):
            return self.future.result(timeout)

        def error(self):
            if not self.future.done():
                return None
            return self.future.exception()

    class Bridge:
        def __init__(self, stream_in, stream_out):
            self.stream_in, self.stream_out = stream_in, stream_out
            self.send_lock = threading.Lock()
            self.map_lock = threading.Lock()
            self.futures = {}
            self.next_id = 0
            self.closed = False
            self.reader = threading.Thread(target=self._read_replies, daemon=True)
            self.reader.start()

        def _read_replies(self):
            while True:
                line = self.stream_in.readline()
                if not line:
                    break
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                with self.map_lock:
                    future = self.futures.pop(reply.get("id"), None)
                if future is None:
                    continue
                if reply.get("ok"):
                    future.set_result(reply.get("value"))
                else:
                    future.set_exception(RuntimeError(str(reply.get("error", "tool call failed"))))
            with self.map_lock:
                self.closed = True
                for future in self.futures.values():
                    if not future.done():
                        future.set_exception(RuntimeError("codemode bridge closed"))
                self.futures.clear()

        def _send(self, payload):
            data = json.dumps(payload, ensure_ascii=False, default=str)
            with self.send_lock:
                self.stream_out.write(data + NL)
                self.stream_out.flush()

        def start(self, name, args=None):
            canonical = self.tools.get(name)
            if canonical is None:
                raise NameError(f"tool not available: {name}")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise TypeError("tool arguments must be an object")
            with self.map_lock:
                if self.closed:
                    raise RuntimeError("codemode bridge closed")
                self.next_id += 1
                call_id = self.next_id
                future = concurrent.futures.Future()
                self.futures[call_id] = future
            self._send({"type": "call", "id": call_id, "tool": canonical, "args": args})
            return Call(name, future)

    class ToolFn:
        def __init__(self, bridge, name):
            self._bridge = bridge
            self._name = name

        def __call__(self, args=None):
            return self._bridge.start(self._name, args).result()

        def start(self, args=None):
            return self._bridge.start(self._name, args)

        def __repr__(self):
            return f"<tool {self._name}>"

    class ToolsNamespace:
        def __init__(self, bridge, names):
            self._bridge = bridge
            self._names = names

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return self._lookup(name)

        def __getitem__(self, name):
            return self._lookup(name)

        def _lookup(self, name):
            canonical = self._bridge.tools.get(name)
            if canonical is not None:
                return ToolFn(self._bridge, canonical)
            wanted = _comparable(name)
            exact = [n for n in self._names if _comparable(n) == wanted]
            close = exact or [n for n in self._names
                              if wanted and (wanted in _comparable(n) or _comparable(n) in wanted)]
            message = f"tools.{name} does not exist."
            if close:
                message += " Did you mean " + ", ".join(f"tools.{n}" for n in close[:5]) + "?"
            elif len(self._names) <= 20:
                message += " Available: " + ", ".join(self._names) + "."
            message += " Check for a member with ALL_TOOLS or dir(tools)."
            raise AttributeError(message)

        def __dir__(self):
            return list(self._names)

    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_rank(query, documents, limit):
        """Rank [(name, description)] documents against the query with BM25."""
        doc_tokens = [(_tokenize(name) + _tokenize(description)) for name, description in documents]
        total = sum(len(tokens) for tokens in doc_tokens)
        average = (total / len(doc_tokens)) if doc_tokens else 0.0
        counts = []
        for tokens in doc_tokens:
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            counts.append(freq)
        df = {}
        for freq in counts:
            for token in freq:
                df[token] = df.get(token, 0) + 1
        scores = []
        for index, freq in enumerate(counts):
            score = 0.0
            for token in _tokenize(query):
                if token not in freq:
                    continue
                idf = math.log(1 + (len(counts) - df[token] + 0.5) / (df[token] + 0.5))
                tf = freq[token]
                size = len(doc_tokens[index])
                norm = 1 - 0.75 + 0.75 * (size / average if average else 0)
                score += idf * (tf * 2.2) / (tf + 1.2 * norm)
            scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [documents[index] for score, index in scores[:limit] if score > 0]

    class NamespaceGroup:
        def __init__(self, name):
            self.name = name
            self.members = []

    def main():
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        init = json.loads(sys.stdin.readline())
        bridge = Bridge(sys.stdin, sys.stdout)
        bridge.tools = {}
        for meta in init.get("tools", []):
            bridge.tools[meta["name"]] = meta["name"]
            if meta.get("js_name"):
                bridge.tools[meta["js_name"]] = meta["name"]
        meta_by_name = {meta["name"]: meta for meta in init.get("tools", [])}
        namespaces = {}
        for meta in init.get("tools", []):
            if meta.get("namespace"):
                group = namespaces.setdefault(meta["namespace"], NamespaceGroup(meta["namespace"]))
                group.members.append(meta)

        all_tools = []
        js_names = {}
        for meta in init.get("tools", []):
            js = meta.get("js_name") or meta["name"]
            if js in js_names:
                continue
            js_names[js] = meta["name"]
            all_tools.append({"name": js, "description": meta.get("description", "")})

        stored = dict(init.get("store", {}))
        writes_set = {}
        writes_delete = set()
        output_items = []
        output_chars = 0
        output_items_count = 0
        reported = {"error": False}

        def emit(item):
            nonlocal output_chars, output_items_count
            if reported["error"]:
                return
            output_chars += len(item.get("text") or item.get("data") or "")
            output_items_count += 1
            if output_chars > MAX_OUTPUT_CHARS or output_items_count > MAX_OUTPUT_ITEMS:
                reported["error"] = True
                failure = {"name": "RangeError", "stack": None, "message": (
                    "script output exceeded the limit of %d characters or %d text(), image(), "
                    "and print/console calls. Print a summary instead, or write large data "
                    "to a file with a tool." % (MAX_OUTPUT_CHARS, MAX_OUTPUT_ITEMS))}
                bridge._send({"type": "script_error", "error": failure})
                os._exit(1)
            bridge._send({"type": "output", "item": item})

        def stringify(value):
            if value is None or isinstance(value, (str, bool, int, float)):
                return str(value) if not isinstance(value, str) else value
            return json.dumps(value, ensure_ascii=False, default=str)

        def text(value):
            emit({"type": "text", "text": stringify(value)})

        def image(value):
            if isinstance(value, str):
                url = value
            elif isinstance(value, dict) and isinstance(value.get("image_url"), str):
                url = value["image_url"]
            elif isinstance(value, dict) and value.get("type") == "image" and isinstance(value.get("data"), str) and value["data"]:
                url = value["data"] if value["data"].lower().startswith("data:") else "data:;base64," + value["data"]
            else:
                raise TypeError(
                    "image expects a non-empty image URL string, an object with image_url, "
                    "or a raw MCP image block")
            if not url:
                raise TypeError(
                    "image expects a non-empty image URL string, an object with image_url, "
                    "or a raw MCP image block")
            colon = url.find(":")
            scheme = url[:colon].lower() if colon != -1 else ""
            if scheme in ("http", "https"):
                raise TypeError("remote image URLs are not supported in tool outputs. Pass a base64 data URI instead")
            comma = url.find(",")
            header = url[colon + 1:comma].split(";") if comma != -1 else []
            if scheme != "data" or comma == -1 or all(part.lower() != "base64" for part in header[1:]):
                raise TypeError("invalid image output. Pass a base64 data URI instead")
            data = re.sub(r"\\s+", "", url[comma + 1:])
            if len(data) % 4 != 0 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", data):
                raise TypeError("invalid image output. The image data is not valid base64 (truncated or corrupted?)")
            signature = next((entry for entry in IMAGE_SIGNATURES if entry[1].match(data[:16])), None)
            if signature is None:
                raise TypeError("invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image")
            emit({"type": "image", "data": data, "mime_type": signature[0]})

        _DELETE = object()

        def store(key, value=_DELETE):
            if not isinstance(key, str):
                raise TypeError("store() key must be a string")
            if value is _DELETE:
                stored.pop(key, None)
                writes_delete.add(key)
                writes_set.pop(key, None)
                return
            previous = len(key) + len(json.dumps(stored[key], ensure_ascii=False)) if key in stored else 0
            try:
                encoded = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"store({key!r}) value is not JSON-serializable: {exc}") from None
            if len(encoded) > MAX_STORE_VALUE_CHARS:
                raise ValueError(
                    f"store({key!r}) value has {len(encoded)} characters of JSON, more than the "
                    f"limit of {MAX_STORE_VALUE_CHARS}. store() is for small state such as IDs or "
                    "summaries; keep large data in variables or write it to a file with a tool.")
            total = sum(len(k) + len(json.dumps(v, ensure_ascii=False)) for k, v in stored.items())
            total = total - previous + len(key) + len(encoded)
            if total > MAX_STORE_TOTAL_CHARS:
                raise ValueError(
                    f"store is full: stored values would exceed {MAX_STORE_TOTAL_CHARS} characters "
                    "of JSON. Delete keys with store(key). store() is for small state such as IDs "
                    "or summaries.")
            stored[key] = value
            writes_delete.discard(key)
            writes_set[key] = value

        def load(key):
            if not isinstance(key, str):
                raise TypeError("load() key must be a string")
            return stored.get(key)

        def wait(handle):
            return handle.result()

        def wait_all(*handles):
            return [handle.result() for handle in handles]

        def wait_any(*handles):
            if not handles:
                raise ValueError("wait_any() expects at least one call handle")
            while True:
                for handle in handles:
                    if handle.done():
                        return handle.result()
                threading.Event().wait(0.01)

        def search_tools(query, options=None):
            if not isinstance(query, str):
                raise ValueError("search_tools() expects a query string")
            limit = 20
            namespace = None
            if isinstance(options, dict):
                if isinstance(options.get("limit"), int) and options["limit"] > 0:
                    limit = options["limit"]
                if isinstance(options.get("namespace"), str):
                    namespace = options["namespace"]
            documents = []
            for meta in init.get("tools", []):
                if namespace and meta.get("namespace") != namespace:
                    continue
                js = meta.get("js_name") or meta["name"]
                documents.append((js, meta.get("description", "")))
            return [{"name": name, "description": description}
                    for name, description in _bm25_rank(query, documents, limit)]

        def describe_tool(name):
            meta = next((m for m in init.get("tools", [])
                         if (m.get("js_name") or m["name"]) == name or m["name"] == name), None)
            return meta.get("description") if meta else None

        def describe_namespace(name):
            group = namespaces.get(name)
            if group is None:
                for key, candidate in namespaces.items():
                    if _comparable(key) == _comparable(name):
                        group = candidate
                        break
            if group is None:
                return None
            return {"name": group.name,
                    "tools": [(m.get("js_name") or m["name"]) for m in group.members]}

        def exit(value=None):
            raise ScriptExit(value)

        class Console:
            @staticmethod
            def _log(*args):
                emit({"type": "text", "text": " ".join(stringify(a) for a in args)})

        console = type("ConsoleType", (), {
            "log": staticmethod(lambda *a: Console._log(*a)),
            "info": staticmethod(lambda *a: Console._log(*a)),
            "warn": staticmethod(lambda *a: Console._log(*a)),
            "error": staticmethod(lambda *a: Console._log(*a)),
            "debug": staticmethod(lambda *a: Console._log(*a)),
        })()

        class Pi:
            pass

        pi = Pi()
        pi.tools = ToolsNamespace(bridge, [m.get("js_name") or m["name"] for m in init.get("tools", [])])
        pi.ALL_TOOLS = all_tools
        pi.text = text
        pi.image = image
        pi.exit = exit
        pi.store = store
        pi.load = load
        pi.wait = wait
        pi.wait_all = wait_all
        pi.wait_any = wait_any
        pi.search_tools = search_tools
        pi.describe_tool = describe_tool
        pi.describe_namespace = describe_namespace
        pi.console = console

        tools = pi.tools
        environment = {"pi": pi, "tools": tools, "ALL_TOOLS": all_tools, "text": text, "image": image,
                       "exit": exit, "store": store, "load": load, "wait": wait, "wait_all": wait_all,
                       "wait_any": wait_any, "search_tools": search_tools, "describe_tool": describe_tool,
                       "describe_namespace": describe_namespace, "console": console, "print": Console._log,
                       "__name__": "__codemode__"}
        value = None
        try:
            with open(sys.argv[1], "r", encoding="utf-8") as handle:
                source = handle.read()
            exec(compile(source, sys.argv[1], "exec"), environment)
            if environment.get("result") is not None:
                value = environment["result"]
        except ScriptExit as stop:
            value = stop.value
        except BaseException:
            stack = traceback.format_exc(limit=12)
            head = stack.splitlines()[-1] if stack else "script failed"
            bridge._send({"type": "script_error", "error": {"name": head.split(":")[0].strip() or "Error",
                                                            "message": head.strip(), "stack": stack}})
            return 1
        bridge._send({"type": "done", "value": value,
                      "writes": {"set": writes_set, "delete": sorted(writes_delete)}})
        return 0

    sys.exit(main())
''')


def load_codemode_store(session: Optional[SessionManager]) -> Dict[str, Any]:
    """Values of ``load()``: the ``codemode-store`` entries on the session, applied in order."""

    data: Dict[str, Any] = {}
    if session is None:
        return data
    for entry in session.custom_entries(STORE_TYPE):
        custom = entry.custom["data"] or {}
        for key in custom.get("delete") or []:
            data.pop(key, None)
        for key, value in (custom.get("set") or {}).items():
            data[key] = value
    return data


def to_codemode_identifier(name: str) -> str:
    """The script identifier for a tool name: invalid characters become ``_``."""

    identifier = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not identifier or not re.match(r"[A-Za-z_]", identifier[0]):
        identifier = "_" + identifier
    return identifier


def _schema_type(schema: Any) -> str:
    if schema is True or schema == {} or schema is None:
        return "Any"
    if schema is False:
        return "Never"
    if not isinstance(schema, dict):
        return "Any"
    kind = schema.get("type")
    if isinstance(kind, list):
        return " | ".join(_schema_type({"type": item}) for item in kind)
    if kind == "string":
        return "str"
    if kind == "integer":
        return "int"
    if kind == "number":
        return "float"
    if kind == "boolean":
        return "bool"
    if kind == "array":
        items = schema.get("items")
        return "list[" + _schema_type(items) + "]" if items is not None else "list"
    if kind == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            required = set(schema.get("required") or [])
            fields = []
            for key, sub in properties.items():
                if not isinstance(key, str):
                    continue
                rendered = f"{key}: {_schema_type(sub)}"
                fields.append(rendered if key in required else rendered + " = None")
            return "{" + ", ".join(fields) + "}"
        return "dict"
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        values = schema["enum"]
        head = ", ".join(repr(value) for value in values[:6])
        return head + (", ..." if len(values) > 6 else "")
    if isinstance(schema.get("anyOf"), list):
        return " | ".join(_schema_type(sub) for sub in schema["anyOf"])
    return "Any"


def _describe_output(output_schema: Optional[dict]) -> str:
    """What a script call resolves to: a string, or the object's field names."""

    if not isinstance(output_schema, dict) or output_schema.get("type") != "object":
        return "str"
    properties = output_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "str"
    required = set(output_schema.get("required") or [])
    fields = [key if key in required else key + "?" for key in properties if isinstance(key, str)]
    return "{ " + ", ".join(fields) + " }"


def render_tool_sample(name: str, description: str, parameters: dict,
                       output_schema: Optional[dict] = None) -> str:
    """The declaration a script sees for one tool: signature plus description."""

    identifier = to_codemode_identifier(name)
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    required = set(parameters.get("required") or []) if isinstance(parameters, dict) else set()
    args_text = ""
    if isinstance(properties, dict) and properties:
        rendered = []
        for key, sub in properties.items():
            if not isinstance(key, str):
                continue
            arg = f"{key}: {_schema_type(sub)}"
            rendered.append(arg if key in required else arg + " = None")
        args_text = ", ".join(rendered)
    signature = f"tools.{identifier}({args_text}) -> {_describe_output(output_schema)}"
    lines = [signature]
    body = (description or "").strip()
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _select_catalog(groups: List[dict], budget: Optional[int]) -> set:
    """Pick tool sections that fit the budget: every group places its cheapest
    remaining section each round, so every namespace is represented first."""

    if budget is None:
        return {entry["name"] for group in groups for entry in group["entries"]}
    queues = [sorted(group["entries"], key=lambda entry: entry["cost"]) for group in groups]
    shown: set = set()
    remaining = budget
    active = [queue for queue in queues if queue]
    while active:
        surviving = []
        for queue in active:
            entry = queue[0]
            if entry["cost"] > remaining:
                continue
            remaining -= entry["cost"]
            shown.add(entry["name"])
            queue.pop(0)
            if queue:
                surviving.append(queue)
        active = surviving
    return shown


def _tool_namespace(name: str) -> Optional[str]:
    """Server namespace of an MCP tool name (``mcp__dev__tool`` -> ``dev``)."""

    match = re.match(r"mcp__([^_]+(?:_[^_]+)*)__", name)
    return match.group(1) if match else None


DESCRIPTION_INTRO = (
    "Run Python that calls other tools. The input is raw Python (not JSON, no code fence), "
    "executed in an isolated subprocess: no network, file system, or timers beyond its tool calls.\n"
    "- tools.<name>(args) calls one tool and blocks until it finishes; tools.<name>.start(args) "
    "starts a call and returns a handle, composed with wait_all(...) / wait_any(...). "
    "Set the module variable `result` to return a value. Calls still running when the script ends are cancelled.\n"
    '- Optional first line: # @options: {"max_output_tokens": 10000, "timeout_ms": 60000}'
)

DESCRIPTION_GLOBALS = (
    "Globals:\n"
    "- text(value), image(dataUriOrImageBlock), print(...), and `result` add output; exit(value) ends the script.\n"
    "- store(key, value) and load(key) keep JSON values across codemode calls; store(key) deletes.\n"
    "- ALL_TOOLS lists every tool; search_tools(query, {limit?, namespace?}), "
    "describe_tool(name), describe_namespace(name) find unlisted tools, such as MCP tools."
)


def create_codemode_description(tools: List[AgentTool], inline_budget: Optional[int] = DEFAULT_INLINE_BUDGET) -> str:
    """Model-facing description: the helper list plus one section per listed tool,
    grouped by MCP namespace and limited to the token budget."""

    sections = [DESCRIPTION_INTRO, DESCRIPTION_GLOBALS]
    declarations = []
    for tool in tools:
        if tool.name == "codemode":
            continue
        sample = render_tool_sample(tool.name, tool.description or "",
                                    tool.parameters or {}, tool.output_schema)
        declarations.append({"name": tool.name, "sample": sample})
    if not declarations:
        return "\n\n".join(sections)

    groups: Dict[str, dict] = {"": {"namespace": None, "entries": []}}
    for declaration in declarations:
        namespace = _tool_namespace(declaration["name"])
        key = "ns:" + namespace if namespace else ""
        group = groups.setdefault(key, {"namespace": namespace, "entries": []})
        group["entries"].append({
            "name": declaration["name"], "sample": declaration["sample"],
            "cost": math.ceil(len(declaration["sample"]) / CHARS_PER_TOKEN),
        })
    ordered = sorted(groups.values(),
                     key=lambda group: (group["namespace"] is not None, group["namespace"] or ""))
    shown = _select_catalog(ordered, inline_budget)

    tool_sections = ["Nested tools:"]
    for group in ordered:
        namespace, entries = group["namespace"], group["entries"]
        visible = [entry for entry in entries if entry["name"] in shown]
        if namespace:
            listing = ""
            if len(visible) != len(entries):
                listing = " (tools not listed)" if not visible else " (some tools not listed)"
            tool_sections.append(f"## {namespace}{listing}")
        for entry in visible:
            identifier = to_codemode_identifier(entry["name"])
            heading = f"### `{identifier}`" if identifier == entry["name"] \
                else f"### `{identifier}` (`{entry['name']}`)"
            tool_sections.append(f"{heading}\n{entry['sample'].strip()}")
    sections.append("\n\n".join(tool_sections))
    return "\n\n".join(sections)


def parse_codemode_source(code: str) -> tuple:
    """Split the optional ``# @options:`` first line from the script source."""

    stripped = code.lstrip()
    if not stripped.startswith("#@options:") and not stripped.startswith("# @options:"):
        return {}, code
    first = stripped.splitlines()[0]
    payload = first.split(":", 1)[1].strip()
    try:
        options = json.loads(payload)
    except ValueError as exc:
        raise ValueError(f"codemode @options line is not valid JSON: {exc}") from None
    if not isinstance(options, dict) or set(options) - {"timeout_ms", "max_output_tokens"}:
        raise ValueError('codemode @options supports "timeout_ms" and "max_output_tokens"')
    for key, value in options.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"codemode @options {key} must be a positive integer")
    rest = code[code.index(first) + len(first):].lstrip("\r\n")
    return options, rest


class _Truncation:
    """Token-budget truncation: keep head and tail of the combined text, spill the rest."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def apply(self, items: List[Union[TextContent, ImageContent]], max_tokens: int, call_id: str
              ) -> List[Union[TextContent, ImageContent]]:
        texts = [item.text for item in items if isinstance(item, TextContent)]
        combined = "\n".join(texts)
        budget = max_tokens * CHARS_PER_TOKEN
        if not texts or len(combined) <= budget:
            return items
        head = combined[:budget // 2]
        tail = combined[len(combined) - (budget - budget // 2):] if budget - budget // 2 > 0 else ""
        removed = len(combined) - budget
        warning = (f"Warning: truncated output (original token count: {math.ceil(len(combined) / CHARS_PER_TOKEN)})\n"
                   f"Total output lines: {combined.count(chr(10)) + 1}\n\n"
                   f"{head}...{math.ceil(removed / CHARS_PER_TOKEN)} tokens truncated...{tail}")
        spill = self.workdir / f"codemode_{call_id}_full_output.txt"
        try:
            spill.write_text(combined, encoding="utf-8")
            warning += f"\n\n[Full output: {spill} (read with offset/limit)]"
        except OSError as exc:
            warning += f"\n\n[Could not save the full output: {exc}]"
        return [TextContent(text=warning)] + [item for item in items if isinstance(item, ImageContent)]


def _save_images(items: List[Union[TextContent, ImageContent]], workdir: Path, call_id: str
                 ) -> List[Union[TextContent, ImageContent]]:
    """Save each output image to a temp file and put a text item with its path before it."""

    saved: Dict[str, str] = {}
    result: List[Union[TextContent, ImageContent]] = []
    counter = 0
    for item in items:
        if not isinstance(item, ImageContent):
            result.append(item)
            continue
        path = saved.get(item.data)
        if path is None:
            extension = IMAGE_EXTENSIONS.get(item.mime_type)
            if extension is None:
                result.append(TextContent(text=f"[Image ({item.mime_type}) has no known file extension]"))
                result.append(item)
                continue
            counter += 1
            path = str(workdir / f"codemode_{call_id}_img{counter}{extension}")
            try:
                Path(path).write_bytes(base64.b64decode(item.data))
            except OSError as exc:
                path = ""
                result.append(TextContent(text=f"[Image ({item.mime_type}) could not be saved: {exc}]"))
                result.append(item)
                continue
            saved[item.data] = path
        if path:
            size = len(base64.b64decode(item.data))
            result.append(TextContent(text=f"[Image saved to {path} ({item.mime_type}, {size} bytes)]"))
        result.append(item)
    return result


def create_codemode_tool(tools: List[AgentTool], session: Optional[SessionManager] = None,
                         settings: Optional[dict] = None, execute_tool=None) -> AgentTool:
    """Build the ``codemode`` AgentTool over the given callable tools.

    ``execute_tool(name, args, signal)`` defaults to invoking the matching tool
    from ``tools`` directly; callers route through the agent pipeline so nested
    calls run the same validation, hooks, and permission checks as direct calls.
    """

    settings = settings or {}
    configured_timeout = settings.get("timeout_ms")
    inline_budget = settings.get("inline_budget", DEFAULT_INLINE_BUDGET)
    callable_tools = [tool for tool in tools if tool.name != "codemode"]
    allowed = {tool.name: tool for tool in callable_tools}
    workdir = Path(tempfile.mkdtemp(prefix="pi-codemode-"))
    worker_path = workdir / "codemode_worker.py"
    worker_path.write_text(WORKER_SOURCE, encoding="utf-8")

    js_names: Dict[str, str] = {}
    tool_meta: List[dict] = []
    for tool in callable_tools:
        js = to_codemode_identifier(tool.name)
        if js in js_names:
            js = ""
        else:
            js_names[js] = tool.name
        tool_meta.append({
            "name": tool.name, "js_name": js,
            "namespace": _tool_namespace(tool.name),
            "description": render_tool_sample(tool.name, tool.description or "",
                                              tool.parameters or {}, tool.output_schema),
        })

    async def run_tool(name: str, args: dict, signal) -> Any:
        if execute_tool is not None:
            return await execute_tool(name, args, signal)
        tool = allowed.get(name)
        if tool is None:
            raise ValueError(f"tool not available: {name}")
        result = await tool.execute("codemode-" + name, args, signal, None)
        if getattr(result, "is_error", False):
            text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
            raise RuntimeError(text or f'tool "{name}" failed')
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        return "\n".join(block.text for block in result.content if isinstance(block, TextContent))

    async def execute(call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("codemode requires a code string")
        try:
            options, code = parse_codemode_source(code)
        except ValueError as exc:
            return _failed_result(call_id, [], [], f"Script error:\n{exc}")
        timeout_ms = options.get("timeout_ms", configured_timeout)
        max_tokens = options.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        code_path = workdir / f"script_{call_id}.py"
        code_path.write_text(code, encoding="utf-8")
        init = {"tools": tool_meta, "store": load_codemode_store(session)}
        started_at = asyncio.get_event_loop().time()

        calls: List[dict] = []
        writes = {"set": {}, "delete": []}
        output_items: List[dict] = []
        pending: Dict[int, asyncio.Future] = {}
        state = {"value": None, "error": None, "finished": False, "timed_out": False, "crashed": False}

        def snapshot() -> dict:
            return {"calls": [dict(entry) for entry in calls]}

        def publish() -> None:
            if on_update is not None:
                update = AgentToolResult(content=[], details=snapshot())
                outcome = on_update(update)
                if asyncio.iscoroutine(outcome):
                    asyncio.ensure_future(outcome)

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(worker_path), str(code_path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            limit=64 * 1024 * 1024)

        async def feed():
            stream = process.stdin
            stream.write((json.dumps(init, ensure_ascii=False) + "\n").encode())
            await stream.drain()

        async def pump():
            while True:
                line = await process.stdout.readline()
                if not line:
                    state["crashed"] = not state["finished"]
                    return
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                kind = message.get("type")
                if kind == "call":
                    await _start_call(message)
                elif kind == "output":
                    output_items.append(message.get("item") or {})
                elif kind == "script_error":
                    state["error"] = message.get("error") or {"message": "script failed"}
                    return
                elif kind == "done":
                    state["value"] = message.get("value")
                    state["finished"] = True
                    worker_writes = message.get("writes") or {}
                    if isinstance(worker_writes.get("set"), dict):
                        writes["set"].update(worker_writes["set"])
                    if isinstance(worker_writes.get("delete"), list):
                        for key in worker_writes["delete"]:
                            writes["set"].pop(key, None)
                            if key not in writes["delete"]:
                                writes["delete"].append(key)
                    return

        async def _start_call(message):
            index = len(calls) + 1
            args = message.get("args") or {}
            try:
                preview = json.dumps(args, ensure_ascii=False)[:ARGS_PREVIEW_CHARS]
            except (TypeError, ValueError):
                preview = ""
            record = {"id": f"{call_id}/{index}", "name": message["tool"], "args": preview,
                      "status": "running", "duration_ms": None, "error": None}
            calls.append(record)
            publish()
            started = asyncio.get_event_loop().time()
            future = asyncio.ensure_future(run_tool(message["tool"], args, signal))
            pending[message["id"]] = future

            def done(fut):
                record["duration_ms"] = round((asyncio.get_event_loop().time() - started) * 1000)
                if fut.cancelled():
                    record["status"] = "cancelled"
                elif fut.exception() is not None:
                    record["status"] = "cancelled" if signal is not None and signal.aborted else "error"
                    record["error"] = str(fut.exception())[:ERROR_PREVIEW_CHARS]
                else:
                    record["status"] = "ok"
                pending.pop(message["id"], None)
                publish()
                asyncio.ensure_future(_write_reply(process, _reply_from_future(message["id"], fut)))

            future.add_done_callback(done)

        def _reply_from_future(message_id, fut) -> dict:
            if fut.cancelled():
                return {"id": message_id, "ok": False, "error": "aborted"}
            if fut.exception() is not None:
                return {"id": message_id, "ok": False, "error": str(fut.exception())}
            return {"id": message_id, "ok": True, "value": fut.result()}

        feeder = asyncio.ensure_future(feed())
        pump_task = asyncio.ensure_future(pump())
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_ms / 1000 if timeout_ms else None
        aborted = False
        try:
            if signal is not None:
                signal.throw_if_aborted()
            while not pump_task.done():
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    state["timed_out"] = True
                    break
                wait_timeout = 0.5 if remaining is None else min(remaining, 0.5)
                done_set, _ = await asyncio.wait({pump_task}, timeout=wait_timeout)
                if pump_task in done_set:
                    break
                if signal is not None and signal.aborted:
                    aborted = True
                    break
            if pump_task.done() and not pump_task.cancelled() and pump_task.exception() is not None:
                state["error"] = {"name": "SandboxError",
                                  "message": f"codemode bridge failed: {pump_task.exception()!r}"}
        finally:
            feeder.cancel()
            pump_task.cancel()
            for future in list(pending.values()):
                future.cancel()
            if not feeder.done():
                try:
                    await feeder
                except (asyncio.CancelledError, OSError, ConnectionError, RuntimeError):
                    pass
            for task in (feeder, pump_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            for future in list(pending.values()):
                if future.cancelled():
                    continue
                try:
                    await future
                except BaseException:
                    pass
            if process.returncode is None:
                process.kill()
            await process.wait()
        for record in calls:
            if record["status"] == "running":
                record["status"] = "cancelled"

        wall_time = asyncio.get_event_loop().time() - started_at
        if aborted:
            return _failed_result(call_id, output_items, calls,
                                  "Script aborted: codemode script was aborted", wall_time)
        if state["timed_out"]:
            return _failed_result(call_id, output_items, calls,
                                  f"Script timed out: exceeded the {timeout_ms} ms deadline", wall_time)
        if state["crashed"] and not state["finished"] and state["error"] is None:
            code_value = process.returncode if process.returncode is not None else "unknown"
            return _failed_result(call_id, output_items, calls,
                                  f"Script sandbox failed: codemode worker exited unexpectedly (exit code {code_value})",
                                  wall_time)
        if state["error"] is not None:
            error = state["error"]
            stack = error.get("stack") or f'{error.get("name", "Error")}: {error.get("message", "")}'
            return _failed_result(call_id, output_items, calls, "Script error:\n" + stack.strip(),
                                  wall_time)

        if writes["set"] or writes["delete"]:
            if session is not None:
                session.append_custom(STORE_TYPE, {"set": dict(writes["set"]), "delete": list(writes["delete"])})

        value = state["value"]
        extra = "" if value is None else (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str))
        items = _build_output_items(output_items, extra)
        items = _save_images(items, workdir, call_id)
        items = _Truncation(workdir).apply(items, max_tokens, call_id)
        return AgentToolResult(
            content=[TextContent(text=f"Script completed\nWall time {wall_time:.1f} seconds\nOutput:")] + items,
            details=snapshot())

    def _failed_result(call_id, output_items, calls, error_text, wall_time=0.0) -> AgentToolResult:
        if calls:
            summary = "Tool calls made before the failure (they are not undone): " + ", ".join(
                f"{record['name']} ({record['status']})" for record in calls)
        else:
            summary = "No tool calls were made."
        header = f"Script failed\nWall time {wall_time:.1f} seconds\nOutput:"
        items = _build_output_items(output_items, f"{error_text}\n\n{summary}".strip())
        return AgentToolResult(content=[TextContent(text=header)] + items,
                               details={"calls": calls}, is_error=True)

    return AgentTool(
        name="codemode",
        description=create_codemode_description(callable_tools, inline_budget),
        parameters={"type": "object", "properties": {"code": {"type": "string", "description": "Raw Python source."}},
                    "required": ["code"]},
        execute=execute,
    )


def _build_output_items(output_items: List[dict], extra_text: str) -> List[Union[TextContent, ImageContent]]:
    """Output items in production order: text first, images after, then the extra text."""

    items: List[Union[TextContent, ImageContent]] = []
    for item in output_items:
        if item.get("type") == "text":
            items.append(TextContent(text=str(item.get("text", ""))))
    if extra_text:
        items.append(TextContent(text=extra_text))
    for item in output_items:
        if item.get("type") == "image" and isinstance(item.get("data"), str):
            items.append(ImageContent(data=item["data"], mime_type=str(item.get("mime_type", "image/png"))))
    return items


async def _write_reply(process, reply: dict) -> None:
    if process.stdin is None or process.returncode is not None:
        return
    try:
        process.stdin.write((json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode())
        await process.stdin.drain()
    except (ConnectionError, RuntimeError):
        pass

"""Codemode worker program, embedded as a standalone source string.

TS counterpart: ``packages/codemode/src/runtime/worker.ts`` plus
``prelude-source.ts``. The string is written to a temp file at tool-creation
time and started with ``python -I`` per script run, so it must run without the
``pi`` package, the source tree, or ``PYTHONPATH``: only the standard library.
It carries its own copies of the output and store quotas; boundary tests keep
the two sides from drifting. Never build this string with templates or inject
host state into it.
"""

from __future__ import annotations

import textwrap

WORKER_SOURCE = textwrap.dedent('''
    """Codemode worker: runs user code with a threaded tool-call bridge."""
    import ast
    import builtins
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

    # User scripts get the normal builtins minus every file, import and
    # code-evaluation entry point; the tool bridge is their only I/O.
    BLOCKED_BUILTINS = {
        "open", "input", "eval", "exec", "compile", "__import__", "breakpoint",
        "help", "exit", "quit", "getattr", "setattr", "delattr", "globals", "locals", "vars",
        "print", "license", "credits", "copyright",
    }

    def restricted_builtins():
        return {name: value for name, value in vars(builtins).items()
                if name not in BLOCKED_BUILTINS}

    def compile_script(source, filename):
        tree = ast.parse(source, filename, "exec")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr.startswith("_")
                    and node.attr != "__name__"):
                raise SyntaxError(f"attribute access is disabled: {node.attr}")
        return compile(tree, filename, "exec")

    class ScriptCallable:
        __slots__ = ("_target",)

        def __init__(self, target):
            object.__setattr__(self, "_target", target)

        def __call__(self, *args, **kwargs):
            target = object.__getattribute__(self, "_target")
            return target(*args, **kwargs)

        def __getattribute__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return object.__getattribute__(self, name)

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

        def __call__(self, args=None, /, **kwargs):
            return self.start(args, **kwargs).result()

        def start(self, args=None, /, **kwargs):
            if kwargs:
                if args is not None:
                    raise TypeError("pass either an argument object or keyword arguments, not both")
                args = kwargs
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
        pi.text = ScriptCallable(text)
        pi.image = ScriptCallable(image)
        pi.exit = ScriptCallable(exit)
        pi.store = ScriptCallable(store)
        pi.load = ScriptCallable(load)
        pi.wait = ScriptCallable(wait)
        pi.wait_all = ScriptCallable(wait_all)
        pi.wait_any = ScriptCallable(wait_any)
        pi.search_tools = ScriptCallable(search_tools)
        pi.describe_tool = ScriptCallable(describe_tool)
        pi.describe_namespace = ScriptCallable(describe_namespace)
        pi.console = console

        tools = pi.tools
        environment = {"pi": pi, "tools": tools, "ALL_TOOLS": all_tools,
                       "text": pi.text, "image": pi.image, "exit": pi.exit,
                       "store": pi.store, "load": pi.load, "wait": pi.wait,
                       "wait_all": pi.wait_all, "wait_any": pi.wait_any,
                       "search_tools": pi.search_tools, "describe_tool": pi.describe_tool,
                       "describe_namespace": pi.describe_namespace, "console": console,
                       "print": ScriptCallable(Console._log),
                       "__name__": "__codemode__", "__builtins__": restricted_builtins()}
        value = None
        try:
            with open(sys.argv[1], "r", encoding="utf-8") as handle:
                source = handle.read()
            exec(compile_script(source, sys.argv[1]), environment)
            if environment.get("result") is not None:
                value = environment["result"]
        except ScriptExit as stop:
            value = stop.value
        except SystemExit:
            raise
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

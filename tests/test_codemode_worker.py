"""Codemode worker tests: the standalone ``python -I`` subprocess program.

Drives the real worker through a minimal line-protocol client: globals, call
handles, queries, output, store quotas and write sets, script errors and
exit. The worker must run without the pi package, the source tree, or
PYTHONPATH, so the driver writes it to a scratch dir and spawns it there.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from pi.codemode.worker_source import WORKER_SOURCE

# Every module import inside the executed worker program; the set is exact on
# purpose: new imports must be standard library and stay dependency-free.
WORKER_IMPORTS = {"builtins", "concurrent.futures", "json", "math", "os", "re", "sys", "threading", "traceback"}

PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
           "AAAABJRU5ErkJggg==")


def meta(name, description="", js_name="", namespace=""):
    return {"name": name, "js_name": js_name, "namespace": namespace, "description": description}


class WorkerRun:
    """What one worker run produced, in protocol order."""

    def __init__(self):
        self.value = None
        self.writes = None
        self.output = []
        self.error = None
        self.crashed = False
        self.exit_code = None


class WorkerDriver:
    """Minimal JSON-line driver for a real worker subprocess."""

    def __init__(self, tmp_path, code, tools=None, store=None, name="script.py"):
        self.tmp_path = tmp_path
        self.code = code
        self.tools = tools or []
        self.store = store or {}
        self.name = name

    async def run(self, on_call, timeout=30):
        script = self.tmp_path / self.name
        script.write_text(self.code, encoding="utf-8")
        worker = self.tmp_path / "codemode_worker.py"
        worker.write_text(WORKER_SOURCE, encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(worker), str(script),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=64 * 1024 * 1024,
            cwd=str(self.tmp_path))
        run = WorkerRun()

        async def drive():
            handlers = []

            async def handle(message):
                try:
                    value = await on_call(message["tool"], message.get("args") or {})
                    reply = {"id": message["id"], "ok": True, "value": value}
                except Exception as exc:  # noqa: BLE001 - mirrors the host reply contract
                    reply = {"id": message["id"], "ok": False, "error": str(exc)}
                process.stdin.write((json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode())
                await process.stdin.drain()

            init = json.dumps({"tools": self.tools, "store": self.store}, ensure_ascii=False)
            process.stdin.write((init + "\n").encode())
            await process.stdin.drain()
            while True:
                line = await process.stdout.readline()
                if not line:
                    run.crashed = True
                    break
                message = json.loads(line)
                kind = message.get("type")
                if kind == "call":
                    # Calls run concurrently so replies can complete out of
                    # order, like the real host's pending-call tasks.
                    handlers.append(asyncio.ensure_future(handle(message)))
                elif kind == "output":
                    run.output.append(message.get("item") or {})
                elif kind == "script_error":
                    run.error = message.get("error")
                    break
                elif kind == "done":
                    run.value = message.get("value")
                    run.writes = message.get("writes")
                    break
            for task in handlers:
                task.cancel()
            if handlers:
                await asyncio.gather(*handlers, return_exceptions=True)

        try:
            await asyncio.wait_for(drive(), timeout=timeout)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
            run.exit_code = process.returncode
        return run


async def run_script(tmp_path, code, tools=None, store=None, on_call=None, **kwargs):
    async def default_call(name, args):
        return {"called": name, "args": args}

    return await WorkerDriver(tmp_path, code, tools=tools, store=store, **kwargs).run(
        on_call or default_call)


def texts(run):
    return [item.get("text") for item in run.output if item.get("type") == "text"]


def test_worker_keyword_tool_arguments(tmp_path):
    async def scenario():
        run = await run_script(tmp_path, (
            "result = [tools.echo(x='hi'), wait(tools.read.start(path='a.txt')),\n"
            "          tools.echo(args='value'), tools.echo({'x': 'dict'}), tools.echo()]\n"
        ), tools=[meta("echo"), meta("read")])
        assert run.error is None
        assert run.value == [
            {"called": "echo", "args": {"x": "hi"}},
            {"called": "read", "args": {"path": "a.txt"}},
            {"called": "echo", "args": {"args": "value"}},
            {"called": "echo", "args": {"x": "dict"}},
            {"called": "echo", "args": {}},
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("code", [
    "tools.echo({'x': 1}, x=2)", "tools.echo.start({'x': 1}, x=2)",
])
def test_worker_rejects_mixed_tool_arguments(tmp_path, code):
    async def scenario():
        run = await run_script(tmp_path, code, tools=[meta("echo")])
        assert run.error["name"] == "TypeError"
        assert "not both" in run.error["message"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------


def test_worker_source_imports_only_stdlib():
    modules = set()
    for node in ast.walk(ast.parse(WORKER_SOURCE)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert modules == WORKER_IMPORTS


def test_worker_scripts_are_denied_file_import_and_eval_escapes(tmp_path):
    async def scenario():
        code = (
            "blocked = {}\n"
            "for name, call in {\n"
            "    'open': lambda: open('x'),\n"
            "    'import': lambda: __import__('socket'),\n"
            "    'eval': lambda: eval('1 + 1'),\n"
            "    'compile': lambda: compile('1', '<x>', 'eval'),\n"
            "}.items():\n"
            "    try:\n"
            "        call()\n"
            "        blocked[name] = 'allowed'\n"
            "    except Exception as exc:\n"
            "        blocked[name] = type(exc).__name__\n"
            "result = {'blocked': blocked, 'sum': sum(range(5)), 'label': str({'a': 1})}\n"
        )
        run = await run_script(tmp_path, code)
        assert run.error is None
        assert run.value["blocked"] == {
            "open": "NameError", "import": "NameError", "eval": "NameError", "compile": "NameError",
        }
        assert run.value["sum"] == 10
        assert run.value["label"] == "{'a': 1}"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Globals, handles and calls
# ---------------------------------------------------------------------------


def test_worker_globals_all_tools_and_queries(tmp_path):
    async def scenario():
        tools = [
            meta("echo", "Echo the arguments back."),
            meta("mcp__dev__query", "Query the dev service.", js_name="dev_query", namespace="dev"),
            meta("mcp__dev__other", "Another dev tool.", namespace="dev"),
        ]
        code = (
            "result = {\n"
            "    'dir': dir(tools),\n"
            "    'all': ALL_TOOLS,\n"
            "    'desc': describe_tool('echo'),\n"
            "    'desc_alias': describe_tool('dev_query'),\n"
            "    'desc_missing': describe_tool('nope'),\n"
            "    'ns': describe_namespace('dev'),\n"
            "    'ns_case': describe_namespace('DEV'),\n"
            "    'ns_missing': describe_namespace('nope'),\n"
            "}\n"
        )
        run = await run_script(tmp_path, code, tools=tools)
        assert run.error is None
        # dir() returns the script-visible names, sorted like any dir() call.
        assert run.value["dir"] == ["dev_query", "echo", "mcp__dev__other"]
        assert run.value["all"] == [
            {"name": "echo", "description": "Echo the arguments back."},
            {"name": "dev_query", "description": "Query the dev service."},
            {"name": "mcp__dev__other", "description": "Another dev tool."},
        ]
        assert run.value["desc"] == "Echo the arguments back."
        assert run.value["desc_alias"] == "Query the dev service."
        assert run.value["desc_missing"] is None
        assert run.value["ns"] == {"name": "dev", "tools": ["dev_query", "mcp__dev__other"]}
        assert run.value["ns_case"] == run.value["ns"]
        assert run.value["ns_missing"] is None

    asyncio.run(scenario())


def test_worker_search_tools_ranking_and_filters(tmp_path):
    async def scenario():
        tools = [
            meta("mcp__dev__query", "Query the dev service.", namespace="dev"),
            meta("mcp__stats__count", "Count things.", namespace="stats"),
            meta("echo", "Echo the arguments back."),
        ]
        code = (
            "result = {\n"
            "    'ranked': search_tools('query'),\n"
            "    'limited': search_tools('query', {'limit': 1}),\n"
            "    'scoped': search_tools('count', {'namespace': 'stats'}),\n"
            "    'empty': search_tools('zzz'),\n"
            "}\n"
        )
        run = await run_script(tmp_path, code, tools=tools)
        assert run.error is None
        assert run.value["ranked"][0]["name"] == "mcp__dev__query"
        assert run.value["limited"] == [run.value["ranked"][0]]
        assert [entry["name"] for entry in run.value["scoped"]] == ["mcp__stats__count"]
        assert run.value["empty"] == []

    asyncio.run(scenario())


def test_worker_calls_and_handle_composition_out_of_order(tmp_path):
    async def scenario():
        seen = []
        release = asyncio.Event()

        async def on_call(name, args):
            seen.append((name, args))
            if name == "fast":
                release.set()
                return "fast:" + json.dumps(args, sort_keys=True)
            await release.wait()
            return "slow"

        tools = [meta("slow"), meta("fast")]
        # slow takes call id 1 but replies after fast (id 2): out-of-order.
        code = (
            "slow_handle = tools.slow.start({'n': 1})\n"
            "fast_handle = tools.fast.start({'n': 2})\n"
            "picked = wait_any(fast_handle, slow_handle)\n"
            "blocking = tools.fast()\n"
            "result = {'picked': picked, 'slow': wait(slow_handle),\n"
            "          'both': wait_all(fast_handle, slow_handle)[0], 'blocking': blocking}\n"
        )
        run = await run_script(tmp_path, code, tools=tools, on_call=on_call)
        assert run.error is None
        assert run.value["picked"] == "fast:{\"n\": 2}"
        assert run.value["slow"] == "slow"
        assert run.value["blocking"] == "fast:{}"  # missing args default to {}
        assert run.value["both"] == "fast:{\"n\": 2}"
        assert seen == [("slow", {"n": 1}), ("fast", {"n": 2}), ("fast", {})]

    asyncio.run(scenario())


def test_worker_unknown_tool_messages(tmp_path):
    async def scenario():
        tools = [meta("echo", "Echo tool.")]
        code = (
            "problems = []\n"
            "try:\n"
            "    tools.ech\n"
            "except AttributeError as exc:\n"
            "    problems.append(str(exc))\n"
            "try:\n"
            "    tools['missing']\n"
            "except AttributeError as exc:\n"
            "    problems.append(str(exc))\n"
            "try:\n"
            "    tools.echo(5)\n"
            "except TypeError as exc:\n"
            "    problems.append(str(exc))\n"
            "result = problems\n"
        )
        run = await run_script(tmp_path, code, tools=tools)
        assert run.error is None
        close, missing, bad_args = run.value
        assert close.startswith("tools.ech does not exist. Did you mean tools.echo?")
        assert missing.startswith("tools.missing does not exist. Available: echo.")
        assert "Check for a member with ALL_TOOLS or dir(tools)." in missing
        assert bad_args == "tool arguments must be an object"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_worker_output_items_and_serialization(tmp_path):
    async def scenario():
        code = (
            "text('plain')\n"
            "text({'k': [1, True, None]})\n"
            "print('a', 1, None, {'x': 'y'})\n"
            "console.log('log')\n"
            "console.error('err')\n"
            "result = 'end'\n"
        )
        run = await run_script(tmp_path, code)
        assert run.error is None
        assert run.value == "end"
        assert texts(run) == ["plain", '{"k": [1, true, null]}',
                              'a 1 None {"x": "y"}', "log", "err"]

    asyncio.run(scenario())


def test_worker_image_validation(tmp_path):
    async def scenario():
        code = (
            "errors = []\n"
            "image('data:image/png;base64," + PNG_1PX + "')\n"
            "image({'type': 'image', 'data': '" + PNG_1PX + "'})\n"
            "image({'image_url': 'data:image/png;base64," + PNG_1PX + "'})\n"
            "for bad in ('https://example.com/x.png', 'data:image/png;base64,not@@base64,',\n"
            "            'data:image/png;base64,SGVsbG8=', 'data:image/png;base64,', None):\n"
            "    try:\n"
            "        image(bad)\n"
            "    except TypeError as exc:\n"
            "        errors.append(str(exc))\n"
            "result = len(errors)\n"
        )
        run = await run_script(tmp_path, code)
        assert run.error is None
        assert run.value == 5
        images = [item for item in run.output if item.get("type") == "image"]
        assert len(images) == 3
        assert all(item["mime_type"] == "image/png" and item["data"] == PNG_1PX for item in images)
        # No output is emitted for rejected values.
        assert len(texts(run)) == 0

    asyncio.run(scenario())


def test_worker_output_quota_enforced(tmp_path):
    async def scenario():
        code = "text('x' * (16 * 1024 * 1024 + 1))\ntext('unreached')\n"
        run = await run_script(tmp_path, code)
        assert run.error is not None
        assert run.error["name"] == "RangeError"
        assert "script output exceeded the limit" in run.error["message"]
        assert texts(run) == []  # the over-quota item is dropped, not emitted

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_worker_store_writes_and_delete(tmp_path):
    async def scenario():
        code = (
            "base = load('base')\n"
            "store('a', {'x': 1})\n"
            "store('base', 2)\n"
            "store('gone', 1)\n"
            "store('gone')\n"
            "try:\n"
            "    store(5, 1)\n"
            "except TypeError:\n"
            "    key_type_error = True\n"
            "result = {'base': base, 'a': load('a'), 'gone': load('gone'),\n"
            "          'key_type_error': key_type_error}\n"
        )
        run = await run_script(tmp_path, code, store={"base": 1})
        assert run.error is None
        assert run.value == {"base": 1, "a": {"x": 1}, "gone": None, "key_type_error": True}
        assert run.writes == {"set": {"a": {"x": 1}, "base": 2}, "delete": ["gone"]}

    asyncio.run(scenario())


def test_worker_store_value_and_total_quota(tmp_path):
    async def scenario():
        code = (
            "errors = []\n"
            "try:\n"
            "    store('big', 'y' * (256 * 1024 + 1))\n"
            "except ValueError as exc:\n"
            "    errors.append(str(exc))\n"
            "count = 0\n"
            "try:\n"
            "    while True:\n"
            "        store('k' + str(count), 'x' * 4000)\n"
            "        count += 1\n"
            "except ValueError as exc:\n"
            "    errors.append(str(exc))\n"
            "result = {'count': count, 'errors': errors}\n"
        )
        run = await run_script(tmp_path, code)
        assert run.error is None
        value_errors, full_error = run.value["errors"]
        assert "more than the" in value_errors and "262144" in value_errors
        assert "store is full" in full_error
        assert run.value["count"] >= 200
        assert len(run.writes["set"]) == run.value["count"]
        assert run.writes["delete"] == []

    asyncio.run(scenario())


def test_worker_store_non_serializable_value(tmp_path):
    async def scenario():
        code = (
            "try:\n"
            "    store('bad', {('tuple',): 1})\n"
            "except TypeError as exc:\n"
            "    result = 'not JSON-serializable' in str(exc)\n"
        )
        run = await run_script(tmp_path, code)
        assert run.error is None
        assert run.value is True
        assert run.writes == {"set": {}, "delete": []}

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Errors and exit
# ---------------------------------------------------------------------------


def test_worker_script_error_carries_traceback_and_keeps_output(tmp_path):
    async def scenario():
        code = "text('before')\n1/0\ntext('unreached')\n"
        run = await run_script(tmp_path, code)
        assert run.error is not None
        assert run.error["name"] == "ZeroDivisionError"
        assert "division by zero" in run.error["message"]
        assert run.error["stack"].startswith("Traceback")
        assert texts(run) == ["before"]
        assert run.writes is None

    asyncio.run(scenario())


def test_worker_exit_and_result_variable(tmp_path):
    async def scenario():
        code = "exit('bye')\ntext('unreached')\n"
        run = await run_script(tmp_path, code)
        assert run.error is None
        assert run.value == "bye"
        assert texts(run) == []

        run = await run_script(tmp_path, "result = {'x': 1}\n")
        assert run.value == {"x": 1}

        run = await run_script(tmp_path, "exit()\n")
        assert run.value is None

        run = await run_script(tmp_path, "text('kept')\n")
        assert run.value is None  # no result variable
        assert texts(run) == ["kept"]

    asyncio.run(scenario())


def test_worker_refuses_remote_image_and_bad_data_uri_with_script_error(tmp_path):
    async def scenario():
        code = "image('https://example.com/x.png')\n"
        run = await run_script(tmp_path, code)
        assert run.error is not None
        assert run.error["name"] == "TypeError"
        assert "remote image URLs are not supported" in run.error["message"]

    asyncio.run(scenario())


def test_worker_store_failure_is_script_error_not_write(tmp_path):
    async def scenario():
        code = "store('a', 1)\nstore('b', 'z' * (256 * 1024 + 1))\n"
        run = await run_script(tmp_path, code)
        assert run.error is not None
        assert run.error["name"] == "ValueError"
        assert run.writes is None  # nothing reported when the script fails

    asyncio.run(scenario())


def test_worker_reports_bridge_closure(tmp_path):
    async def scenario():
        # A handler that fails after the driver already saw `done` cannot
        # happen; instead verify a reply error surfaces as a script-side
        # RuntimeError on the waiting handle.
        release = asyncio.Event()

        async def on_call(name, args):
            await release.wait()
            raise RuntimeError("host exploded")

        code = (
            "handle = tools.slow.start({})\n"
            "try:\n"
            "    wait(handle)\n"
            "except RuntimeError as exc:\n"
            "    result = str(exc)\n"
        )
        driver = WorkerDriver(tmp_path, code, tools=[meta("slow")])

        async def releaser():
            await asyncio.sleep(0.2)
            release.set()

        asyncio.ensure_future(releaser())
        run = await driver.run(on_call)
        assert run.error is None
        assert run.value == "host exploded"

    asyncio.run(scenario())


@pytest.mark.parametrize("code", [
    "tools.missing({})\n",
    "wait_any()\n",
    "store(5, 1)\n",
])
def test_worker_failures_report_script_error(tmp_path, code):
    async def scenario():
        run = await run_script(tmp_path, code, tools=[meta("echo")])
        assert run.error is not None
        assert run.error["stack"].startswith("Traceback")

    asyncio.run(scenario())

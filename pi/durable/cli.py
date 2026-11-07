"""Command-line access to the durable harness (``pi durable <command>``).

Storage is a sqlite database (or jsonl file) chosen with ``--storage``. The
model is resolved with the same flags/environment as the main CLI. This wires
the durable runtime into the CLI surface: create conversations, submit inputs
and tool tasks, run the scheduler, and inspect transcripts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from ..ai.models import ModelError, resolve_model
from ..coding_agent.tools import create_default_tools
from .harness import Harness
from .session import Session
from .storage import JsonlStorage, MemoryStorage, SqliteStorage
from .tasks import register_builtin_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi durable", description="Durable conversation runtime")
    parser.add_argument("--storage", default="durable.sqlite", help="sqlite path or .jsonl file (memory: ':memory:')")
    parser.add_argument("--provider", help="anthropic | openai | faux | ...")
    parser.add_argument("--model", help="Model id")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", help="Provider base URL override")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent task executions")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Create a conversation; prints its id")
    submit = sub.add_parser("input", help="Submit user input and run tasks")
    submit.add_argument("conversation_id", type=int)
    submit.add_argument("text", nargs="+")
    tool = sub.add_parser("tool", help="Submit a durable pi.tool task")
    tool.add_argument("conversation_id", type=int)
    tool.add_argument("name")
    tool.add_argument("arguments", nargs="?", default="{}", help="JSON object of tool arguments")
    compact = sub.add_parser("compact", help="Submit a pi.compact task for a conversation")
    compact.add_argument("conversation_id", type=int)
    run = sub.add_parser("run", help="Run pending tasks to quiescence")
    run.add_argument("--tasks", type=int, default=None, help="Run only the first N tasks")
    sub.add_parser("transcript", help="Print a conversation transcript")
    transcript = sub.add_parser("tasks", help="List task records")
    transcript.add_argument("--conversation", type=int, default=None)
    abort = sub.add_parser("abort", help="Abort a task")
    abort.add_argument("task_id", type=int)
    return parser


def open_storage(path: str):
    if path == ":memory:":
        return MemoryStorage()
    if path.endswith(".jsonl"):
        return JsonlStorage(path)
    return SqliteStorage(path)


def transcript_text(entries) -> str:
    lines = []
    for entry in entries:
        for message in (entry.model or []):
            role = getattr(message, "role", None) or message.get("role", "?")
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text = content
            else:
                parts = []
                for block in content or []:
                    text_value = getattr(block, "text", None)
                    parts.append(text_value if text_value is not None else json.dumps(block, ensure_ascii=False, default=str))
                text = "\n".join(parts)
            lines.append(f"[{entry.id} {entry.kind} {role}] {text}")
    return "\n".join(lines)


async def async_main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    storage = open_storage(args.storage)
    session = Session(storage)
    try:
        model = resolve_model(args.provider, args.model, args.api_key, args.base_url)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    harness = Harness(session, model)
    register_builtin_tasks(harness, tools={tool.name: tool for tool in create_default_tools(".")})

    if args.command == "create":
        print(await harness.create_conversation())
        return 0
    if args.command == "input":
        task_id = await harness.submit_input(args.conversation_id, " ".join(args.text))
        print(f"task: {task_id}")
    elif args.command == "tool":
        try:
            arguments = json.loads(args.arguments)
        except ValueError:
            print("error: arguments must be a JSON object", file=sys.stderr)
            return 2
        if not isinstance(arguments, dict):
            print("error: arguments must be a JSON object", file=sys.stderr)
            return 2
        task_id = await harness.submit_tool_task(args.conversation_id, args.name, arguments)
        print(f"task: {task_id}")
    elif args.command == "compact":
        task_id = await harness.submit_compact_task(args.conversation_id)
        print(f"task: {task_id}")
    elif args.command == "abort":
        await harness.abort_task(args.task_id)
    executed = await harness.run_pending_tasks(max(1, args.concurrency))
    if executed and args.command != "create":
        print(f"ran {executed} task(s)")
    if args.command == "transcript":
        entries = await harness.transcript(args.conversation_id)
        text = transcript_text(entries)
        print(text if text else "(empty)")
    elif args.command == "tasks":
        for record in await harness.list_tasks(args.conversation):
            state = record.state
            detail = state.outcome.status if state.outcome else state.status
            print(f"{record.id}  {record.kind}  {detail}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

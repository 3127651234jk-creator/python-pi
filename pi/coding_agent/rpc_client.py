"""Async client for the Python CLI's subprocess RPC protocol."""

import asyncio
import json
import sys
from typing import List, Optional


class RpcClient:
    def __init__(self) -> None:
        self.process = None
        self.reader = None
        self.pending = {}
        self.events = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.sequence = 0

    async def start(self, args: Optional[List[str]] = None, cwd: Optional[str] = None):
        if self.process:
            raise RuntimeError("Client already started")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pi", "--mode", "rpc", *(args or []), cwd=cwd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        self.reader = asyncio.create_task(self.read())
        return self

    async def read(self):
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError("RPC process exited")
                record = json.loads(line)
                if record.get("type") == "response" and record.get("id") in self.pending:
                    future = self.pending[record["id"]]
                    if not future.done():
                        future.set_result(record)
                else:
                    await self.events.put(record)
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)
            await self.events.put({"type": "connection_closed", "error": str(exc)})

    async def request(self, command: str, timeout: float = 60, **params):
        if self.process is None or self.process.returncode is not None or self.reader.done():
            raise RuntimeError("RPC client is disconnected")
        self.sequence += 1
        request_id = str(self.sequence)
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            async with self.lock:
                self.process.stdin.write((json.dumps({**params, "id": request_id, "type": command}) + "\n").encode())
                await self.process.stdin.drain()
            response = await asyncio.wait_for(future, timeout)
            if not response["success"]:
                raise RuntimeError(response.get("error", "RPC command failed"))
            return response.get("data")
        finally:
            self.pending.pop(request_id, None)

    async def prompt_and_wait(self, message: str, timeout: float = 300):
        disposition = await self.request("prompt", message=message)
        if disposition and disposition.get("disposition") == "handled":
            return []
        async def collect():
            records = []
            while True:
                record = await self.events.get()
                if record["type"] == "connection_closed":
                    raise RuntimeError(record["error"])
                records.append(record)
                if record["type"] == "agent_settled":
                    return records
        return await asyncio.wait_for(collect(), timeout)

    async def respond_ui(self, request_id: str, value=None) -> None:
        """Answer an extension_ui_request from the server."""
        if self.process is None or self.process.stdin is None:
            return
        async with self.lock:
            self.process.stdin.write((json.dumps(
                {"type": "extension_ui_response", "id": request_id, "value": value}) + "\n").encode())
            await self.process.stdin.drain()

    async def close(self):
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 7)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            if self.reader:
                await self.reader

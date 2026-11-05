"""Shared pytest helpers: asyncio runner and faux-agent fixtures."""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def direct_httpx(monkeypatch):
    """Make httpx talk to in-process fixture servers directly.

    Default ``trust_env`` honors environment proxy configuration, which
    intercepts even 127.0.0.1 traffic on hosts running a transparent proxy
    layer. Tests that exercise real sockets against localhost fixtures opt in
    here so they behave the same on proxied and unproxied machines.
    """
    original = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("trust_env", False)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)

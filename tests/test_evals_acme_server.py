"""Port of packages/evals/test/acme-server.test.ts."""

import asyncio
import json

import httpx
import pytest

from evals.acme_server import (
    OPENAI_MODEL_ID,
    OPENAI_PROBE_PROMPT,
    OPENAI_PROBE_RESPONSE,
    STREAM_MODEL_ID,
    STREAM_PROBE_PROMPT,
    STREAM_PROBE_RESPONSE,
    create_acme_server,
)


class Fixture:
    def __init__(self, mode, url, headers, unauthorized_headers, model, prompt, response,
                 content_type, unauthorized, invalid):
        self.mode = mode
        self.url = url
        self.headers = headers
        self.unauthorized_headers = unauthorized_headers
        self.model = model
        self.prompt = prompt
        self.response = response
        self.content_type = content_type
        self.unauthorized = unauthorized
        self.invalid = invalid


FIXTURES = [
    Fixture(
        mode="openai",
        url=lambda server: f"{server.base_url()}/chat/completions",
        headers={"authorization": "Bearer resolved-acme-key"},
        unauthorized_headers={"authorization": "Bearer wrong-key"},
        model=OPENAI_MODEL_ID,
        prompt=OPENAI_PROBE_PROMPT,
        response=OPENAI_PROBE_RESPONSE,
        content_type="text/event-stream",
        unauthorized="Invalid Acme credential",
        invalid="Invalid OpenAI-compatible request",
    ),
    Fixture(
        mode="stream",
        url=lambda server: f"{server.origin()}/generate",
        headers={"x-acme-key": "resolved-stream-key"},
        unauthorized_headers={"x-acme-key": "wrong-key"},
        model=STREAM_MODEL_ID,
        prompt=STREAM_PROBE_PROMPT,
        response=STREAM_PROBE_RESPONSE,
        content_type="application/x-ndjson",
        unauthorized="Invalid Acme Stream credential",
        invalid="Invalid Acme Stream request",
    ),
]


def body(model, prompt, stream=True):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}


async def post(url, headers, payload):
    # trust_env=False: talk to the in-process fixture directly instead of any
    # environment-level proxy layer.
    async with httpx.AsyncClient(trust_env=False) as client:
        return await client.post(url, headers={"content-type": "application/json", **headers},
                                 content=json.dumps(payload).encode("utf-8"))


def streamed_text(mode, raw):
    if mode == "stream":
        events = [json.loads(line) for line in raw.splitlines() if line]
        return "".join(event.get("text", "") for event in events if event.get("type") == "text_delta")
    chunks = []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: "):])
        chunks.append((chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
    return "".join(chunks)


@pytest.fixture(scope="module")
def server(request):
    """One started Acme server per fixture mode; yields (fixture, server)."""
    fixture = request.param
    acme = create_acme_server(fixture.mode)
    acme.start()
    yield fixture, acme
    acme.stop()


@pytest.fixture(autouse=True)
def reset_probe(server):
    server[1].reset()


@pytest.mark.parametrize("server", FIXTURES, ids=lambda fixture: fixture.mode, indirect=True)
class TestAcmeFixture:
    def test_accepts_the_probe_request_and_records_it(self, server):
        fixture, acme = server
        response = asyncio.run(post(fixture.url(acme), fixture.headers, body(fixture.model, fixture.prompt)))
        assert response.status_code == 200
        assert fixture.content_type in response.headers["content-type"]
        assert streamed_text(fixture.mode, response.text) == fixture.response
        assert acme.valid_request_received() is True

    def test_rejects_a_bad_credential_with_401_and_does_not_record_a_probe(self, server):
        fixture, acme = server
        response = asyncio.run(post(fixture.url(acme), fixture.unauthorized_headers,
                                    body(fixture.model, fixture.prompt)))
        assert response.status_code == 401
        assert response.text == fixture.unauthorized
        assert acme.valid_request_received() is False

    def test_rejects_a_malformed_request_with_422_and_does_not_record_a_probe(self, server):
        fixture, acme = server
        response = asyncio.run(post(fixture.url(acme), fixture.headers, body(fixture.model, fixture.prompt, False)))
        assert response.status_code == 422
        assert response.text == fixture.invalid
        assert acme.valid_request_received() is False

    def test_does_not_treat_a_non_probe_success_as_the_probe(self, server):
        fixture, acme = server
        response = asyncio.run(post(fixture.url(acme), fixture.headers, body(fixture.model, "hello")))
        assert response.status_code == 200
        assert acme.valid_request_received() is False

    def test_clears_the_probe_flag_on_reset(self, server):
        fixture, acme = server
        response = asyncio.run(post(fixture.url(acme), fixture.headers, body(fixture.model, fixture.prompt)))
        assert response.status_code == 200
        assert acme.valid_request_received() is True
        acme.reset()
        assert acme.valid_request_received() is False

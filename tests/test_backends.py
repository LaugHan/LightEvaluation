import asyncio
from types import SimpleNamespace

from eval_pipeline.core.backends import APIBackend, GenResult, api_headers


def test_api_headers_uses_openai_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert api_headers()["Authorization"] == "Bearer secret"


def test_api_headers_is_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert api_headers() == {}


def test_api_backend_runs_requests_concurrently_within_batch():
    class ProbeBackend(APIBackend):
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def _call(self, prompt, config):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return GenResult(prompt, "stop")

    backend = ProbeBackend()
    config = SimpleNamespace(
        model=SimpleNamespace(max_retries=1, concurrency=2),
        generation=SimpleNamespace(max_tokens=1, temperature=0.0, stop=None),
    )

    results = backend.generate(["a", "b", "c", "d"], config)

    assert [result.text for result in results] == ["a", "b", "c", "d"]
    assert backend.max_active == 2

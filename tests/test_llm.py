from __future__ import annotations

import json

import httpx
import pytest

from mafialab import paths
from mafialab.llm import (
    CostTracker,
    DiskCache,
    FakeLLM,
    LLMError,
    LLMRequest,
    ManagedClient,
    Pricing,
    RetryableLLMError,
    Usage,
    cache_key,
)
from mafialab.llm.providers import AnthropicClient, OllamaClient, OpenAICompatClient
from mafialab.llm.ratelimit import RateLimiter, retry_with_backoff
from mafialab.llm.registry import build_raw_client, load_models


def test_fake_llm_sequence_and_records_calls():
    llm = FakeLLM(["one", "two"])
    assert llm("a") == "one"
    assert llm("b") == "two"
    assert [c.prompt_text() for c in llm.calls] == ["a", "b"]
    with pytest.raises(LLMError):
        llm("c")


def test_fake_llm_callable_and_default():
    llm = FakeLLM(["a"], default="?")
    assert [llm("x"), llm("y")] == ["a", "?"]
    echo = FakeLLM(lambda req: req.prompt_text().upper())
    assert echo("abc") == "ABC"


def test_cache_key_depends_on_all_parts():
    base = LLMRequest.from_prompt("p", params={"temperature": 0.7}, sample_index=0)
    k = cache_key("openai_compat", "m", base)
    assert k == cache_key("openai_compat", "m", base.model_copy(update={"tags": {"game": "x"}}))
    assert k != cache_key("ollama", "m", base)
    assert k != cache_key("openai_compat", "m2", base)
    assert k != cache_key("openai_compat", "m", base.model_copy(update={"sample_index": 1}))
    assert k != cache_key(
        "openai_compat", "m", base.model_copy(update={"params": {"temperature": 0}})
    )
    assert k != cache_key(
        "openai_compat", "m", LLMRequest.from_prompt("p2", params={"temperature": 0.7})
    )


def test_managed_client_caches_and_tracks(tmp_path):
    fake = FakeLLM(["first", "second"])
    tracker = CostTracker(tmp_path / "costs.jsonl")
    client = ManagedClient(fake, cache=DiskCache(tmp_path / "c.sqlite"), tracker=tracker)
    req = LLMRequest.from_prompt("hello", tags={"game_id": "g1"})
    assert client.complete(req).text == "first"
    again = client.complete(req)
    assert again.text == "first" and again.cached
    assert len(fake.calls) == 1
    assert client.complete(req.model_copy(update={"sample_index": 1})).text == "second"
    totals = tracker.totals()
    assert totals["calls"] == 3 and totals["cached_calls"] == 1
    assert len((tmp_path / "costs.jsonl").read_text().splitlines()) == 3


def test_cache_survives_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    ManagedClient(FakeLLM(["x"]), cache=DiskCache(path)).complete(LLMRequest.from_prompt("q"))
    fresh = FakeLLM([])  # would raise if called
    assert (
        ManagedClient(fresh, cache=DiskCache(path)).complete(LLMRequest.from_prompt("q")).text
        == "x"
    )


def test_cache_misses_when_the_models_settings_change(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["options"])
        return httpx.Response(200, json={"message": {"content": "ok"}, "done_reason": "stop"})

    def client(temperature):
        inner = OllamaClient(
            "m",
            base_url="http://ollama.test",
            default_params={"temperature": temperature},
            http=_mock(handler),
        )
        return ManagedClient(inner, cache=DiskCache(tmp_path / "c.sqlite"))

    req = LLMRequest.from_prompt("q")
    client(0.8).complete(req)
    assert client(0.8).complete(req).cached
    assert not client(0.2).complete(req).cached
    assert calls == [{"temperature": 0.8}, {"temperature": 0.2}]


def test_retry_with_backoff_then_success():
    sleeps = []
    fake = FakeLLM(["ok"], fail_times=2, failure=RetryableLLMError("429"))
    client = ManagedClient(fake, retries=3, base_delay=1.0, sleep=sleeps.append)
    assert client.complete(LLMRequest.from_prompt("x")).text == "ok"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_retry_gives_up_and_respects_retry_after():
    sleeps = []

    def always():
        raise RetryableLLMError("503", retry_after=7)

    with pytest.raises(RetryableLLMError):
        retry_with_backoff(always, retries=2, sleep=sleeps.append, jitter=0)
    assert sleeps == [7, 7]


def test_rate_limiter_waits_when_bucket_empty():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)  # 1 request/second, burst 5
    for _ in range(5):
        assert limiter.acquire() == 0
    assert limiter.acquire() == pytest.approx(1.0)


def test_pricing_cost_with_cached_tokens():
    p = Pricing(input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.3)
    cost = p.cost(Usage(input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=500_000))
    assert cost == pytest.approx(1.5 + 0.15 + 1.5)


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_client_request_and_usage():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '"hello"'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
            },
        )

    client = OpenAICompatClient(
        "openai/gpt-5-mini",
        base_url="https://example.test/api/v1",
        api_key="k",
        default_params={"reasoning_effort": "minimal"},
        pricing=Pricing(input_per_mtok=1, output_per_mtok=2),
        http=_mock(handler),
    )
    resp = client.complete(LLMRequest.from_prompt("hi"))
    assert resp.text == '"hello"'
    assert seen["url"] == "https://example.test/api/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["reasoning_effort"] == "minimal"
    assert resp.usage.input_tokens == 1000
    assert resp.cost_usd == pytest.approx((1000 * 1 + 10 * 2) / 1e6)


def test_openai_compat_429_is_retryable():
    client = OpenAICompatClient(
        "m",
        base_url="https://x.test",
        http=_mock(lambda r: httpx.Response(429, headers={"retry-after": "3"})),
    )
    with pytest.raises(RetryableLLMError) as info:
        client.complete(LLMRequest.from_prompt("hi"))
    assert info.value.retry_after == 3
    bad = OpenAICompatClient(
        "m", base_url="https://x.test", http=_mock(lambda r: httpx.Response(400))
    )
    with pytest.raises(LLMError):
        bad.complete(LLMRequest.from_prompt("hi"))


def test_anthropic_client_prompt_cache_marker():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Bob"}],
                "usage": {"input_tokens": 50, "output_tokens": 2, "cache_read_input_tokens": 900},
                "stop_reason": "end_turn",
            },
        )

    client = AnthropicClient(
        "some-model",
        base_url="https://api.test",
        default_params={"prompt_cache": True},
        http=_mock(handler),
    )
    req = LLMRequest(
        messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "vote"}]
    )
    resp = client.complete(req)
    assert resp.text == "Bob"
    assert seen["body"]["system"] == "rules"
    assert seen["body"]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert resp.usage.cached_input_tokens == 900 and resp.usage.input_tokens == 950


def test_ollama_widens_num_ctx_for_long_prompts():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}, "done_reason": "stop"})

    def client(**params):
        return OllamaClient(
            "m",
            base_url="http://ollama.test",
            default_params={"max_tokens": 5, **params},
            http=_mock(handler),
        )

    client().complete(LLMRequest.from_prompt("short"))
    client().complete(LLMRequest.from_prompt("x" * 30000))
    client(options={"num_ctx": 8192}).complete(LLMRequest.from_prompt("x" * 30000))
    assert "num_ctx" not in bodies[0]["options"]
    assert bodies[1]["options"]["num_ctx"] == 10240  # 30000 / 3 + 5, rounded up to 2048s
    assert bodies[2]["options"]["num_ctx"] == 8192


def test_ollama_client_options():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": "Alice"},
                "prompt_eval_count": 12,
                "eval_count": 1,
                "done_reason": "stop",
            },
        )

    client = OllamaClient(
        "qwen2.5:7b",
        base_url="http://ollama.test",
        default_params={"temperature": 0.2, "max_tokens": 5},
        http=_mock(handler),
    )
    resp = client.complete(LLMRequest.from_prompt("x"))
    assert resp.text == "Alice" and resp.cost_usd == 0
    assert seen["body"]["options"] == {"temperature": 0.2, "num_predict": 5}
    assert seen["body"]["stream"] is False


def test_models_yaml_loads_and_builds_all_clients():
    cfg = load_models()
    assert {s.provider for s in cfg.models.values()} <= {"ollama", "fake", "nim", "openai_compat"}
    open_weight_paper = {
        s.paper_model for s in cfg.models.values() if s.provider == "ollama" and s.paper_model
    }
    assert open_weight_paper == {
        "Llama 3.1 8B Instruct",
        "Mistral 7B Instruct",
        "Qwen2.5 7B Instruct",
    }
    for spec in cfg.models.values():
        assert build_raw_client(spec).model == spec.model


def test_every_model_plays_at_the_same_sampling_settings():
    for alias, spec in load_models().models.items():
        if spec.provider == "fake":
            continue
        params = spec.params
        if spec.provider == "ollama":
            params = params | params["options"]
        assert (params["temperature"], params["top_p"]) == (0.8, 0.9), alias
        assert spec.provider != "ollama" or params["top_k"] == 40, alias


def test_paid_models_yaml_has_the_papers_ten_api_models():
    cfg = load_models(paths.configs_dir() / "paid-models.yaml")
    assert cfg.get("or-gpt-5-mini").params["reasoning_effort"] == "minimal"
    api_paper = {s.paper_model for s in cfg.models.values() if s.provider == "openai_compat"}
    assert len(api_paper) == 10
    for spec in cfg.models.values():
        assert build_raw_client(spec).model == spec.model

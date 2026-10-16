from __future__ import annotations

import time
from typing import Any

import httpx

from mafialab.llm.base import LLMClient, LLMError, RetryableLLMError
from mafialab.llm.cost import Pricing
from mafialab.llm.types import LLMRequest, LLMResponse, Usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    retry_after = resp.headers.get("retry-after")
    msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
    if resp.status_code in RETRYABLE_STATUS:
        try:
            ra = float(retry_after) if retry_after else None
        except ValueError:
            ra = None
        raise RetryableLLMError(msg, retry_after=ra)
    raise LLMError(msg)


class _HTTPClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        default_params: dict[str, Any] | None = None,
        pricing: Pricing | None = None,
        timeout: float = 120.0,
        http: httpx.Client | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_params = dict(default_params or {})
        self.pricing = pricing or Pricing()
        self.http = http or httpx.Client(timeout=timeout)

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        return {**self.default_params, **request.params}

    def _post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            resp = self.http.post(url, json=json, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableLLMError(f"transport error: {exc}") from exc
        _raise_for_status(resp)
        return resp.json()

    def _finish(
        self,
        text: str,
        usage: Usage,
        data: dict,
        started: float,
        finish: str | None,
        thinking: str | None = None,
    ):
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=usage,
            cost_usd=self.pricing.cost(usage),
            latency_s=time.monotonic() - started,
            finish_reason=finish,
            raw=data,
            thinking=thinking,
        )


class OpenAICompatClient(_HTTPClient):
    # OpenRouter, OpenAI, vLLM, LM Studio...
    provider = "openai_compat"

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        params = self._params(request)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        for key in ("temperature", "top_p", "max_tokens", "seed", "reasoning_effort"):
            if params.get(key) is not None:
                body[key] = params[key]
        body.update(params.get("extra_body", {}))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._post(f"{self.base_url}/chat/completions", body, headers)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            thinking = choice["message"].get("thinking")
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc
        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            cached_input_tokens=cached,
        )
        return self._finish(
            text, usage, data, started, choice.get("finish_reason"), thinking=thinking
        )


class AnthropicClient(_HTTPClient):
    provider = "anthropic"

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        params = self._params(request)
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                continue
            messages.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
        # params.prompt_cache: let the provider cache the start of the prompt
        if params.get("prompt_cache") and messages:
            messages[0]["content"][0]["cache_control"] = {"type": "ephemeral"}
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.get("max_tokens", 1024),
            "messages": messages,
        }
        if system:
            body["system"] = system
        for key in ("temperature", "top_p"):
            if params.get(key) is not None:
                body[key] = params[key]
        if params.get("thinking"):
            body["thinking"] = params["thinking"]
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": params.get("anthropic_version", "2023-06-01"),
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        data = self._post(f"{self.base_url}/v1/messages", body, headers)
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        thinking = (
            "".join(
                b.get("thinking", "")
                for b in data.get("content", [])
                if b.get("type") == "thinking"
            )
            or None
        )
        u = data.get("usage") or {}
        cache_read = u.get("cache_read_input_tokens", 0) or 0
        cache_write = u.get("cache_creation_input_tokens", 0) or 0
        usage = Usage(
            input_tokens=u.get("input_tokens", 0) + cache_read + cache_write,
            output_tokens=u.get("output_tokens", 0),
            cached_input_tokens=cache_read,
        )
        return self._finish(text, usage, data, started, data.get("stop_reason"), thinking=thinking)


class OllamaClient(_HTTPClient):
    provider = "ollama"

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        params = self._params(request)
        options: dict[str, Any] = dict(params.get("options", {}))
        if params.get("temperature") is not None:
            options["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            options["num_predict"] = params["max_tokens"]
        if params.get("seed") is not None:
            options["seed"] = params["seed"]
        if "num_ctx" not in options:
            # Ollama silently truncates prompts beyond its context window (4096 by default)
            need = len(request.prompt_text()) // 3 + (params.get("max_tokens") or 0)
            if need > 4096:
                options["num_ctx"] = -(-need // 2048) * 2048
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": options,
        }
        if "think" in params:
            body["think"] = params["think"]
        data = self._post(f"{self.base_url}/api/chat", body, {"Content-Type": "application/json"})
        text = (data.get("message") or {}).get("content", "")
        thinking = (data.get("message") or {}).get("thinking")
        usage = Usage(
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )
        return self._finish(text, usage, data, started, data.get("done_reason"), thinking=thinking)


class NVIDIAClient(OpenAICompatClient):
    provider = "nim"

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


class LLMRequest(BaseModel):
    messages: list[ChatMessage]
    params: dict[str, Any] = Field(default_factory=dict)
    sample_index: int = 0  # separates repeated samples of the same prompt
    # only for cost reporting, not part of the cache key
    tags: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs: Any) -> LLMRequest:
        return cls(messages=[ChatMessage(role="user", content=prompt)], **kwargs)

    def prompt_text(self) -> str:
        return "\n\n".join(m.content for m in self.messages)


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    cached: bool = False
    latency_s: float = 0.0
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None
    thinking: str | None = None  # model's reasoning process if available


def cache_key(provider: str, model: str, request: LLMRequest) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "params": request.params,
        "messages": [m.model_dump() for m in request.messages],
        "sample_index": request.sample_index,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    # ~4 characters per token
    return max(1, (len(text) + 3) // 4) if text else 0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from mafialab import paths
from mafialab.llm.base import LLMClient, LLMError
from mafialab.llm.cache import DiskCache
from mafialab.llm.cost import CostTracker, Pricing
from mafialab.llm.fake import FakeLLM
from mafialab.llm.managed import ManagedClient
from mafialab.llm.providers import AnthropicClient, OllamaClient, OpenAICompatClient
from mafialab.llm.ratelimit import RateLimiter

ProviderName = Literal["openai_compat", "anthropic", "ollama", "fake"]


class ModelSpec(BaseModel):
    alias: str
    provider: ProviderName
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing = Field(default_factory=Pricing)
    rpm: float = 0
    paper_model: str | None = None


class ModelsConfig(BaseModel):
    models: dict[str, ModelSpec]

    def get(self, alias: str) -> ModelSpec:
        try:
            return self.models[alias]
        except KeyError as exc:
            raise LLMError(f"unknown model alias {alias!r}; see configs/models.yaml") from exc


def load_models(path: str | Path | None = None) -> ModelsConfig:
    path = Path(path) if path else paths.models_yaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models = {
        alias: ModelSpec(alias=alias, **spec) for alias, spec in (raw.get("models") or {}).items()
    }
    return ModelsConfig(models=models)


DEFAULT_BASE_URLS = {
    "openai_compat": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
}


def build_raw_client(spec: ModelSpec, **http_kwargs) -> LLMClient:
    if spec.provider == "fake":
        return FakeLLM(cycle=True, default='"I have nothing to add."', model=spec.model)
    base_url = spec.base_url or DEFAULT_BASE_URLS[spec.provider]
    common = dict(
        base_url=base_url,
        api_key=os.environ.get(spec.api_key_env) if spec.api_key_env else None,
        default_params=spec.params,
        pricing=spec.pricing,
        **http_kwargs,
    )
    if spec.provider == "openai_compat":
        return OpenAICompatClient(spec.model, **common)
    if spec.provider == "anthropic":
        return AnthropicClient(spec.model, **common)
    if spec.provider == "ollama":
        common["base_url"] = spec.base_url or os.environ.get("OLLAMA_BASE_URL", base_url)
        return OllamaClient(spec.model, **common)
    raise LLMError(f"unsupported provider {spec.provider}")


def build_client(
    alias: str,
    *,
    config: ModelsConfig | None = None,
    cache: DiskCache | Literal["default"] | None = "default",
    tracker: CostTracker | None = None,
    **http_kwargs,
) -> ManagedClient:
    config = config or load_models()
    spec = config.get(alias)
    if cache == "default":
        cache = DiskCache(paths.cache_dir() / "llm_cache.sqlite")
    limiter = RateLimiter(spec.rpm) if spec.rpm else None
    return ManagedClient(
        build_raw_client(spec, **http_kwargs), cache=cache, tracker=tracker, limiter=limiter
    )

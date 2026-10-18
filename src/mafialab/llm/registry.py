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
from mafialab.llm.providers import AnthropicClient, NVIDIAClient, OllamaClient, OpenAICompatClient
from mafialab.llm.ratelimit import RateLimiter

# which API a model is called through: openai_compat is OpenRouter or any OpenAI-style server,
# nim is NVIDIA's API and fake is the test model
ProviderName = Literal["openai_compat", "anthropic", "ollama", "nim", "fake"]


# one entry of configs/models.yaml: the provider and model id behind an alias, plus its sampling
# params, price, rate limit and, for the paper's models, the name the paper uses
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


# where each provider's API lives, unless the model entry sets base_url
DEFAULT_BASE_URLS = {
    "openai_compat": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
    "nim": "https://integrate.api.nvidia.com/v1",
}


def build_raw_client(spec: ModelSpec, **http_kwargs) -> LLMClient:
    """
    Make the bare client for one model entry, without cache, rate limit or retries. The API key
    is read from the environment variable named in api_key_env.
    :param spec: The model entry.
    :param http_kwargs: Passed on to the client, e.g. timeout, or http with a mock client in tests.
    """
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
    if spec.provider == "nim":
        return NVIDIAClient(spec.model, **common)
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
    """
    The usual way to get a model: look up its alias and return a client that is ready to use,
    with cache, rate limit, retries and cost tracking.
    :param alias: A name from configs/models.yaml, e.g. "ollama-llama3.1-8b".
    :param config: The loaded models file; read from configs/models.yaml if not given.
    :param cache: "default" uses data/cache/llm_cache.sqlite, None turns the cache off.
    :param tracker: Where to record costs; None records nothing.
    :param http_kwargs: Passed on to the client, see build_raw_client.
    """
    config = config or load_models()
    spec = config.get(alias)
    if cache == "default":
        cache = DiskCache(paths.cache_dir() / "llm_cache.sqlite")
    limiter = RateLimiter(spec.rpm) if spec.rpm else None
    return ManagedClient(
        build_raw_client(spec, **http_kwargs), cache=cache, tracker=tracker, limiter=limiter
    )

from __future__ import annotations

from mafialab.llm.base import LLMClient
from mafialab.llm.cache import DiskCache
from mafialab.llm.cost import CostTracker
from mafialab.llm.ratelimit import RateLimiter, retry_with_backoff
from mafialab.llm.types import LLMRequest, LLMResponse, cache_key


class ManagedClient(LLMClient):
    def __init__(
        self,
        inner: LLMClient,
        *,
        cache: DiskCache | None = None,
        tracker: CostTracker | None = None,
        limiter: RateLimiter | None = None,
        retries: int = 5,
        base_delay: float = 1.0,
        sleep=None,
    ):
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.cache = cache
        self.tracker = tracker
        self.limiter = limiter
        self.retries = retries
        self.base_delay = base_delay
        self._sleep = sleep

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = cache_key(self.provider, self.model, request)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                response = hit.model_copy(update={"cached": True, "latency_s": 0.0})
                if self.tracker:
                    self.tracker.record(response, request.tags)
                return response

        def call() -> LLMResponse:
            if self.limiter:
                self.limiter.acquire()
            return self.inner.complete(request)

        kwargs = {"retries": self.retries, "base_delay": self.base_delay}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        response = retry_with_backoff(call, **kwargs)
        if self.cache is not None:
            self.cache.put(key, response)
        if self.tracker:
            self.tracker.record(response, request.tags)
        return response

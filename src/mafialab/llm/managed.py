from __future__ import annotations

import threading

from mafialab.llm.base import LLMClient
from mafialab.llm.cache import DiskCache
from mafialab.llm.cost import CostTracker
from mafialab.llm.ratelimit import RateLimiter, retry_with_backoff
from mafialab.llm.types import LLMRequest, LLMResponse, cache_key


# wraps a real client and adds the extras every model needs: cache, rate limit, retries and
# cost tracking
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
        # replies of games still being played, saved to the cache by end_game
        self._held: dict[str, list[tuple[str, LLMResponse]]] = {}
        self._held_lock = threading.Lock()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Answer from the cache if this exact request was made before. Otherwise wait for the rate
        limiter, call the model (trying again on temporary errors), save the reply in the cache
        and record its cost. A request tagged with a game_id is held back from the cache until
        end_game says that game finished, so a halted game leaves nothing cached.
        """
        # key on the settings the provider really sends, so editing models.yaml is a cache miss
        params = {**getattr(self.inner, "default_params", {}), **request.params}
        key = cache_key(self.provider, self.model, request.model_copy(update={"params": params}))
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
            if game := request.tags.get("game_id"):
                with self._held_lock:
                    self._held.setdefault(game, []).append((key, response))
            else:
                self.cache.put(key, response)
        if self.tracker:
            self.tracker.record(response, request.tags)
        return response

    def end_game(self, game_id: str, completed: bool) -> None:
        """
        Save a finished game's replies in the cache, or forget them if it did not finish.
        :param game_id: The game_id tag of the game's requests.
        :param completed: Whether the game reached its end.
        """
        with self._held_lock:
            held = self._held.pop(game_id, [])
        if completed and self.cache is not None:
            for key, response in held:
                self.cache.put(key, response)

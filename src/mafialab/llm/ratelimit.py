from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from mafialab.llm.base import RetryableLLMError

T = TypeVar("T")


class RateLimiter:
    # token bucket; rpm <= 0 disables it
    def __init__(
        self,
        rpm: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = rpm
        self.capacity = max(1.0, rpm / 60.0 * 5) if rpm > 0 else 0.0
        self.tokens = self.capacity
        self.clock = clock
        self.sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        if self.rpm <= 0:
            return 0.0
        waited = 0.0
        with self._lock:
            while True:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rpm / 60.0)
                self._last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return waited
                delay = (1.0 - self.tokens) * 60.0 / self.rpm
                self.sleep(delay)
                waited += delay


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    rng = rng or random.Random(0)
    attempt = 0
    while True:
        try:
            return fn()
        except RetryableLLMError as exc:
            if attempt >= retries:
                raise
            delay = exc.retry_after if exc.retry_after is not None else base_delay * 2**attempt
            delay = min(max_delay, delay) * (1 + jitter * rng.random())
            sleep(delay)
            attempt += 1

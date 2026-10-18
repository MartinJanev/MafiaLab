from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from mafialab.llm.base import RetryableLLMError

# whatever fn returns, so retry_with_backoff returns the same type
T = TypeVar("T")


# keeps calls under rpm per minute: a bucket refills by rpm/60 tokens per second, each call takes
# one token and waits if the bucket is empty; it holds up to 5 seconds' worth, so short bursts pass
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
        """
        Take one token, sleeping until there is one.
        :return: Seconds spent waiting.
        """
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
    """
    Call fn, and when it fails with RetryableLLMError wait and call it again. The wait doubles
    every time (1s, 2s, 4s, ...) up to max_delay, unless the server said how long to wait.
    :param fn: The call to make.
    :param retries: How many extra tries before the error is raised.
    :param base_delay: Wait before the first retry, in seconds.
    :param max_delay: Longest wait, in seconds.
    :param jitter: Each wait is made up to this fraction longer at random (0.1 = up to 10%), so
        many clients do not all retry at the same moment.
    :param sleep: How to wait; tests pass a fake one so they do not really sleep.
    :param rng: Source of the random jitter; the default has a fixed seed.
    :return: What fn returned.
    """
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

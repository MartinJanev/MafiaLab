from __future__ import annotations

from abc import ABC, abstractmethod

from mafialab.llm.types import LLMRequest, LLMResponse


class LLMError(RuntimeError):
    pass


class RetryableLLMError(LLMError):
    # rate limits, 5xx, timeouts
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.complete(LLMRequest.from_prompt(prompt, **kwargs)).text

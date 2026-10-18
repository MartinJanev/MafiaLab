from __future__ import annotations

from abc import ABC, abstractmethod

from mafialab.llm.types import LLMRequest, LLMResponse


# a model call failed in a way that trying again will not fix (bad request, wrong key, ...)
class LLMError(RuntimeError):
    pass


# a model call failed but may work if we wait and try again; retry_after is the server's
# suggested wait in seconds
class RetryableLLMError(LLMError):
    # rate limits, 5xx, timeouts
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# what every model looks like to the rest of the code: give it an LLMRequest, get an LLMResponse
class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.complete(LLMRequest.from_prompt(prompt, **kwargs)).text

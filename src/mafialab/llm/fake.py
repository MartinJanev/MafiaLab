from __future__ import annotations

from collections.abc import Callable, Sequence

from mafialab.llm.base import LLMClient, LLMError
from mafialab.llm.types import LLMRequest, LLMResponse, Usage, estimate_tokens


class FakeLLM(LLMClient):
    # script: replies returned in order, or a function request -> reply
    provider = "fake"

    def __init__(
        self,
        script: Sequence[str] | Callable[[LLMRequest], str] = (),
        *,
        model: str = "fake-model",
        cycle: bool = False,
        default: str | None = None,
        fail_times: int = 0,
        failure: Exception | None = None,
    ):
        self.model = model
        self._script = script
        self._cycle = cycle
        self._default = default
        self._index = 0
        self._fail_times = fail_times
        self._failure = failure
        self.calls: list[LLMRequest] = []

    def _next_text(self, request: LLMRequest) -> str:
        if callable(self._script):
            return self._script(request)
        items = list(self._script)
        if self._index < len(items):
            text = items[self._index]
        elif self._cycle and items:
            text = items[self._index % len(items)]
        elif self._default is not None:
            text = self._default
        else:
            raise LLMError(f"FakeLLM script exhausted after {len(items)} responses")
        self._index += 1
        return text

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._failure or LLMError("FakeLLM scripted failure")
        text = self._next_text(request)
        usage = Usage(
            input_tokens=estimate_tokens(request.prompt_text()),
            output_tokens=estimate_tokens(text),
        )
        return LLMResponse(text=text, provider=self.provider, model=self.model, usage=usage)

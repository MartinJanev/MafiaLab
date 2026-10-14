from __future__ import annotations

import threading
import time
from pathlib import Path

from pydantic import BaseModel

from mafialab.llm.types import LLMResponse, Usage


class Pricing(BaseModel):
    # USD per million tokens
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cached_input_per_mtok: float | None = None

    def cost(self, usage: Usage) -> float:
        cached = usage.cached_input_tokens
        cached_rate = (
            self.cached_input_per_mtok
            if self.cached_input_per_mtok is not None
            else self.input_per_mtok
        )
        uncached = max(0, usage.input_tokens - cached)
        return (
            uncached * self.input_per_mtok
            + cached * cached_rate
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


class CallRecord(BaseModel):
    timestamp: float
    provider: str
    model: str
    usage: Usage
    cost_usd: float
    cached: bool
    latency_s: float
    tags: dict[str, str] = {}


class CostTracker:
    def __init__(self, log_path: str | Path | None = None):
        self.records: list[CallRecord] = []
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, response: LLMResponse, tags: dict[str, str] | None = None) -> CallRecord:
        rec = CallRecord(
            timestamp=time.time(),
            provider=response.provider,
            model=response.model,
            usage=response.usage,
            cost_usd=0.0 if response.cached else response.cost_usd,
            cached=response.cached,
            latency_s=response.latency_s,
            tags=dict(tags or {}),
        )
        with self._lock:
            self.records.append(rec)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(rec.model_dump_json() + "\n")
        return rec

    def totals(self) -> dict[str, float]:
        usage = sum((r.usage for r in self.records if not r.cached), Usage())
        return {
            "calls": len(self.records),
            "cached_calls": sum(r.cached for r in self.records),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(sum(r.cost_usd for r in self.records), 6),
        }

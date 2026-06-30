"""Lightweight local cost tracker — no API key required."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from langchain_tokonomics.pricing import calculate_cost, get_pricing


@dataclass
class CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Thread-safe accumulator for LLM call costs.

    Usage:
        tracker = CostTracker(budget_usd=10.0)
        tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
        print(tracker.total_cost)
        print(tracker.summary())
    """

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd
        self._calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> CallRecord:
        cost = calculate_cost(model, input_tokens, output_tokens)
        rec = CallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        with self._lock:
            self._calls.append(rec)
        return rec

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(c.cost_usd for c in self._calls)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(c.input_tokens + c.output_tokens for c in self._calls)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._calls)

    @property
    def budget_remaining(self) -> float | None:
        if self.budget_usd is None:
            return None
        return self.budget_usd - self.total_cost

    @property
    def budget_exceeded(self) -> bool:
        if self.budget_usd is None:
            return False
        return self.total_cost >= self.budget_usd

    def cost_by_model(self) -> dict[str, float]:
        with self._lock:
            breakdown: dict[str, float] = {}
            for c in self._calls:
                breakdown[c.model] = breakdown.get(c.model, 0.0) + c.cost_usd
            return breakdown

    def summary(self) -> str:
        lines = [
            f"Total cost: ${self.total_cost:.6f}",
            f"Total tokens: {self.total_tokens:,}",
            f"API calls: {self.call_count}",
        ]
        if self.budget_usd is not None:
            pct = (self.total_cost / self.budget_usd * 100) if self.budget_usd > 0 else 0
            lines.append(f"Budget: ${self.budget_usd:.2f} ({pct:.1f}% used)")
        breakdown = self.cost_by_model()
        if breakdown:
            lines.append("Cost by model:")
            for model, cost in sorted(breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"  {model}: ${cost:.6f}")
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()

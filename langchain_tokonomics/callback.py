"""LangChain callback handler for automatic cost tracking."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_tokonomics.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class TokonomicsCallbackHandler(BaseCallbackHandler):
    """LangChain callback that tracks token usage and costs for every LLM call.

    Works with any LangChain chat model (OpenAI, Anthropic, Google, etc.).
    No API key required — runs fully local. Optionally set a budget to get
    warnings when spending approaches the limit.

    Usage:
        from langchain_tokonomics import TokonomicsCallbackHandler

        handler = TokonomicsCallbackHandler(budget_usd=5.00)

        # Use with any LangChain model
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
        llm.invoke("Hello!")

        # Check costs
        print(handler.total_cost)
        print(handler.summary())

    With Tokonomics API (optional — for persistent tracking and alerts):
        handler = TokonomicsCallbackHandler(
            api_key="mk_your_key_here",
            base_url="https://tokonomics.ca",
            budget_usd=50.00,
        )
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        api_key: str | None = None,
        base_url: str = "https://tokonomics.ca",
        on_budget_warning: Any | None = None,
        warning_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.tracker = CostTracker(budget_usd=budget_usd)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.on_budget_warning = on_budget_warning
        self.warning_threshold = warning_threshold
        self._warning_fired = False

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not response.llm_output:
            return

        token_usage = response.llm_output.get("token_usage", {})
        model = response.llm_output.get("model_name", "")

        if not model:
            model = response.llm_output.get("model", "unknown")

        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            return

        record = self.tracker.record(model, input_tokens, output_tokens)

        logger.debug(
            "LLM call: model=%s input=%d output=%d cost=$%.6f total=$%.6f",
            model,
            input_tokens,
            output_tokens,
            record.cost_usd,
            self.tracker.total_cost,
        )

        self._check_budget()

        if self.api_key:
            self._send_to_api(model, input_tokens, output_tokens, record.cost_usd)

    def _check_budget(self) -> None:
        if self.tracker.budget_usd is None:
            return

        pct = self.tracker.total_cost / self.tracker.budget_usd
        if pct >= self.warning_threshold and not self._warning_fired:
            self._warning_fired = True
            msg = (
                f"[Tokonomics] Budget warning: ${self.tracker.total_cost:.4f} "
                f"/ ${self.tracker.budget_usd:.2f} ({pct * 100:.1f}% used)"
            )
            logger.warning(msg)
            if self.on_budget_warning:
                self.on_budget_warning(self.tracker.total_cost, self.tracker.budget_usd)

        if self.tracker.budget_exceeded:
            logger.error(
                "[Tokonomics] Budget exceeded: $%.4f / $%.2f",
                self.tracker.total_cost,
                self.tracker.budget_usd,
            )

    def _send_to_api(
        self, model: str, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        try:
            import requests

            requests.post(
                f"{self.base_url}/proxy/usage",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "source": "langchain",
                },
                timeout=5,
            )
        except Exception:
            logger.debug("Failed to send usage to Tokonomics API", exc_info=True)

    @property
    def total_cost(self) -> float:
        return self.tracker.total_cost

    @property
    def total_tokens(self) -> int:
        return self.tracker.total_tokens

    @property
    def call_count(self) -> int:
        return self.tracker.call_count

    def cost_by_model(self) -> dict[str, float]:
        return self.tracker.cost_by_model()

    def summary(self) -> str:
        return self.tracker.summary()

    def reset(self) -> None:
        self.tracker.reset()
        self._warning_fired = False

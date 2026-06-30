"""Tests for TokonomicsCallbackHandler."""

from unittest.mock import MagicMock
from langchain_core.outputs import LLMResult
from langchain_tokonomics import TokonomicsCallbackHandler


def _make_llm_result(model: str, prompt_tokens: int, completion_tokens: int) -> LLMResult:
    return LLMResult(
        generations=[[]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "model_name": model,
        },
    )


def test_on_llm_end_tracks_cost():
    handler = TokonomicsCallbackHandler()
    result = _make_llm_result("gpt-4o", 1000, 500)
    handler.on_llm_end(result)
    assert handler.total_cost > 0
    assert handler.total_tokens == 1500
    assert handler.call_count == 1


def test_budget_warning_fires():
    warned = []
    handler = TokonomicsCallbackHandler(
        budget_usd=0.001,
        warning_threshold=0.5,
        on_budget_warning=lambda spent, budget: warned.append((spent, budget)),
    )
    result = _make_llm_result("gpt-4o", 100_000, 50_000)
    handler.on_llm_end(result)
    assert len(warned) == 1


def test_warning_fires_only_once():
    count = []
    handler = TokonomicsCallbackHandler(
        budget_usd=0.001,
        on_budget_warning=lambda s, b: count.append(1),
    )
    for _ in range(5):
        handler.on_llm_end(_make_llm_result("gpt-4o", 100_000, 50_000))
    assert len(count) == 1


def test_no_tracking_without_tokens():
    handler = TokonomicsCallbackHandler()
    result = _make_llm_result("gpt-4o", 0, 0)
    handler.on_llm_end(result)
    assert handler.call_count == 0


def test_no_tracking_without_llm_output():
    handler = TokonomicsCallbackHandler()
    result = LLMResult(generations=[[]], llm_output=None)
    handler.on_llm_end(result)
    assert handler.call_count == 0


def test_reset():
    handler = TokonomicsCallbackHandler(budget_usd=100)
    handler.on_llm_end(_make_llm_result("gpt-4o", 1000, 500))
    handler.reset()
    assert handler.total_cost == 0
    assert handler.call_count == 0


def test_cost_by_model():
    handler = TokonomicsCallbackHandler()
    handler.on_llm_end(_make_llm_result("gpt-4o", 100, 50))
    handler.on_llm_end(_make_llm_result("gpt-4o-mini", 100, 50))
    breakdown = handler.cost_by_model()
    assert len(breakdown) == 2

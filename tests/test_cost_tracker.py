"""Tests for CostTracker."""

import threading
from langchain_tokonomics.cost_tracker import CostTracker


def test_record_and_totals():
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
    assert tracker.total_cost > 0
    assert tracker.total_tokens == 1500
    assert tracker.call_count == 1


def test_budget_tracking():
    tracker = CostTracker(budget_usd=0.01)
    tracker.record("gpt-4o", input_tokens=1_000_000, output_tokens=500_000)
    assert tracker.budget_exceeded is True
    assert tracker.budget_remaining < 0


def test_no_budget():
    tracker = CostTracker()
    assert tracker.budget_exceeded is False
    assert tracker.budget_remaining is None


def test_cost_by_model():
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
    tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
    breakdown = tracker.cost_by_model()
    assert "gpt-4o" in breakdown
    assert "gpt-4o-mini" in breakdown
    assert len(breakdown) == 2


def test_reset():
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
    tracker.reset()
    assert tracker.total_cost == 0
    assert tracker.call_count == 0


def test_summary_format():
    tracker = CostTracker(budget_usd=10.0)
    tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
    summary = tracker.summary()
    assert "Total cost" in summary
    assert "Budget" in summary


def test_thread_safety():
    tracker = CostTracker()
    threads = []
    for _ in range(10):
        t = threading.Thread(
            target=tracker.record, args=("gpt-4o", 100, 50)
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    assert tracker.call_count == 10

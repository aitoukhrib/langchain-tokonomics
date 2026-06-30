"""Tests for pricing module."""

from langchain_tokonomics.pricing import calculate_cost, get_pricing, PRICING, ALIASES


def test_known_model_pricing():
    pricing = get_pricing("gpt-4o")
    assert pricing["input"] == 2.50
    assert pricing["output"] == 10.00


def test_alias_resolution():
    pricing = get_pricing("claude-sonnet-4")
    assert pricing == get_pricing("claude-sonnet-4-20250514")


def test_fuzzy_match():
    pricing = get_pricing("gpt-4o-2024-08-06")
    assert pricing["input"] == 2.50


def test_unknown_model_fallback():
    pricing = get_pricing("totally-unknown-model-xyz")
    assert pricing["input"] == 0.15
    assert pricing["output"] == 0.60


def test_calculate_cost_gpt4o():
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
    assert abs(cost - expected) < 1e-10


def test_calculate_cost_zero_tokens():
    cost = calculate_cost("gpt-4o", input_tokens=0, output_tokens=0)
    assert cost == 0.0


def test_all_aliases_resolve():
    for alias, target in ALIASES.items():
        assert target in PRICING, f"Alias {alias} points to unknown model {target}"

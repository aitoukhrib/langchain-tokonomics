"""LLM pricing data per 1M tokens. Updated June 2026."""

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 10.00, "output": 40.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # DeepSeek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Meta (via providers)
    "llama-4-maverick": {"input": 0.25, "output": 0.50},
    "llama-4-scout": {"input": 0.15, "output": 0.30},
    # Mistral
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    # Cohere
    "command-a-03-2025": {"input": 2.50, "output": 10.00},
    # xAI
    "grok-3": {"input": 3.00, "output": 15.00},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
    # Cerebras
    "cerebras-llama-4-scout": {"input": 0.09, "output": 0.20},
    # Fireworks
    "accounts/fireworks/models/llama-v3p3-70b-instruct": {"input": 0.90, "output": 0.90},
}

# Aliases for common model name variations
ALIASES: dict[str, str] = {
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-11-20": "gpt-4o",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-haiku-3.5": "claude-haiku-3-5-20241022",
    "deepseek-v3": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
}

FALLBACK_PRICING = {"input": 0.15, "output": 0.60}


def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model. Falls back to gpt-4o-mini rates for unknown models."""
    model_lower = model.lower()
    if model_lower in PRICING:
        return PRICING[model_lower]
    if model_lower in ALIASES:
        return PRICING[ALIASES[model_lower]]
    for key in PRICING:
        if key in model_lower or model_lower in key:
            return PRICING[key]
    return FALLBACK_PRICING


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single LLM call."""
    pricing = get_pricing(model)
    return (input_tokens * pricing["input"] / 1_000_000) + (
        output_tokens * pricing["output"] / 1_000_000
    )

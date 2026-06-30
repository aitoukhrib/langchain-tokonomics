# langchain-tokonomics

LangChain integration for [Tokonomics](https://tokonomics.ca) — automatic cost tracking, budget alerts, and spending caps for LLM API calls.

## Features

- **Automatic cost tracking** for every LangChain LLM call (OpenAI, Anthropic, Google, DeepSeek, Mistral, Cohere, xAI, and more)
- **30+ models** with up-to-date pricing (June 2026)
- **Budget alerts** — get warnings when spending approaches a threshold
- **Per-model breakdowns** — see which models cost the most
- **Thread-safe** — works with async and parallel chains
- **No API key required** — runs fully local by default
- **Optional Tokonomics API** — for persistent tracking, team dashboards, and hard spending caps

## Installation

```bash
pip install langchain-tokonomics
```

## Quick Start

### Track costs locally (no API key needed)

```python
from langchain_openai import ChatOpenAI
from langchain_tokonomics import TokonomicsCallbackHandler

# Create handler with a $5 budget
handler = TokonomicsCallbackHandler(budget_usd=5.00)

# Use with any LangChain model
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
response = llm.invoke("Explain quantum computing in 3 sentences.")

# Check costs
print(f"Cost: ${handler.total_cost:.6f}")
print(f"Tokens: {handler.total_tokens:,}")
print(handler.summary())
```

### With budget warnings

```python
def on_warning(spent, budget):
    print(f"WARNING: ${spent:.2f} of ${budget:.2f} budget used!")

handler = TokonomicsCallbackHandler(
    budget_usd=10.00,
    warning_threshold=0.8,  # Warn at 80%
    on_budget_warning=on_warning,
)

llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

for question in my_questions:
    if handler.tracker.budget_exceeded:
        print("Budget exceeded, stopping.")
        break
    llm.invoke(question)
```

### With Tokonomics API (persistent tracking)

Sign up at [tokonomics.ca](https://tokonomics.ca) for a free API key. This gives you a dashboard with historical cost data, team breakdowns, and hard spending caps.

```python
handler = TokonomicsCallbackHandler(
    api_key="mk_your_api_key_here",
    budget_usd=50.00,
)

llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
```

### Cost estimation without making API calls

```python
from langchain_tokonomics.pricing import calculate_cost, get_pricing

# Check pricing for a model
pricing = get_pricing("gpt-4o")
print(f"GPT-4o: ${pricing['input']}/1M input, ${pricing['output']}/1M output")

# Estimate cost
cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
print(f"Estimated cost: ${cost:.6f}")
```

### Per-model breakdown

```python
handler = TokonomicsCallbackHandler()

# Run multiple models...
llm_4o = ChatOpenAI(model="gpt-4o", callbacks=[handler])
llm_mini = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])

llm_4o.invoke("Complex analysis task...")
llm_mini.invoke("Simple formatting task...")

# See cost breakdown
for model, cost in handler.cost_by_model().items():
    print(f"  {model}: ${cost:.6f}")
```

## Supported Models

30+ models across 8 providers:

| Provider | Models |
|---|---|
| OpenAI | GPT-4o, GPT-4o Mini, GPT-4.1, o3, o3-mini, o4-mini |
| Anthropic | Claude Sonnet 4, Claude Opus 4, Claude Haiku 3.5 |
| Google | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash |
| DeepSeek | DeepSeek V3, DeepSeek R1 |
| Meta | Llama 4 Maverick, Llama 4 Scout |
| Mistral | Mistral Large, Mistral Small |
| xAI | Grok 3, Grok 3 Mini |
| Cohere | Command A |

Unknown models fall back to GPT-4o Mini pricing.

## Why Tokonomics?

LangChain makes it easy to build LLM applications, but production costs can spiral without visibility. `langchain-tokonomics` gives you:

1. **Real-time cost awareness** during development and production
2. **Budget guardrails** to prevent unexpected bills
3. **Model cost comparison** to pick the right model for each task
4. **Team accountability** when connected to the [Tokonomics dashboard](https://tokonomics.ca)

## Links

- [Tokonomics](https://tokonomics.ca) — AI cost metering API
- [Tokonomics Blog](https://tokonomics.ca/blog) — LLM cost analysis and optimization guides
- [LLM Cost Calculator](https://huggingface.co/spaces/tokonomics/llm-api-cost-calculator-2026) — Compare pricing across models
- [Documentation](https://tokonomics.ca/docs)

## License

MIT

# Basic vs Advanced Prompt Strategy

## Compared Strategies

### Basic Prompt
- Requires answers from retrieved context.
- Rejects unavailable information.
- Prioritizes concise output.

### Advanced Prompt
- Requires every factual statement to be supported.
- Explicitly separates conversation-history interpretation from document evidence.
- Prevents history from overriding retrieved context.
- Encourages structured output only when useful.

## Evaluation Method

The runnable comparison uses the same questions, retrieved chunks, Gemini model, and evaluation metrics for both strategies. It records:

- Context-grounding score
- Expected-answer similarity
- Basic answer
- Advanced answer

Run:

```bash
python -m src.prompt_comparison
```

The command generates:

- `evaluation_results/prompt_strategy_comparison.csv`
- `evaluation_results/prompt_strategy_comparison.md`

## Design Result

The project uses the **advanced prompt** by default because it has stricter grounding rules, explicitly limits conversation history to reference resolution, and requires unsupported answers to be rejected. The executable comparison remains the source of numeric results because Gemini responses may vary between runs.

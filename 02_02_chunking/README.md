# 02_02_chunking

Four text chunking strategies compared side-by-side.

## Run

```bash
npm run lesson7:chunking
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY` (needed for context-enriched and topic-based strategies).

## What it does

1. Reads `workspace/example.md`
2. Runs four chunking strategies on the same text:
   - **Characters** — fixed-size windows with overlap
   - **Separators** — splits on headings and paragraph boundaries
   - **Context** — separator-based chunks enriched with an LLM-generated context prefix
   - **Topics** — LLM identifies logical topic boundaries and groups text accordingly
3. Saves each result as JSONL in `workspace/example-[strategy].jsonl`

## Notes

The character and separator strategies are purely local. The context and topic strategies call the LLM, so they consume tokens. Pre-generated outputs are already present in `workspace/`.

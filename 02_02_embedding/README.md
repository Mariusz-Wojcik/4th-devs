# 02_02_embedding

Interactive embedding demo with a pairwise similarity matrix.

## Run

```bash
npm run lesson7:embedding
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

## What it does

1. Opens an interactive REPL
2. Each text you type is embedded with `text-embedding-3-small`
3. After two or more entries, prints a color-coded similarity matrix (cosine similarity)
4. Makes it easy to see which inputs cluster together

## Notes

Type `exit` or press Enter on an empty line to quit. The matrix uses green (≥ 0.60 similar), yellow (≥ 0.35 related), and red (< 0.35 distant) to visualize scores.

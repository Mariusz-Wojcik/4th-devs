# 02_02_hybrid_rag

Hybrid RAG agent with SQLite FTS5 full-text search and sqlite-vec vector similarity.

## Run

```bash
npm run lesson7:hybrid_rag
```

## Required setup

1. Copy `env.example` to `.env` in the repo root.
2. Set one Responses API key: `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.

## What it does

1. Reads `.md`/`.txt` files from `workspace/`, chunks them, generates embeddings
2. Stores everything in SQLite with FTS5 full-text index and sqlite-vec vector index
3. At query time, combines BM25 keyword search with cosine vector similarity via Reciprocal Rank Fusion (RRF)
4. Runs an agentic loop with conversation history

## Notes

Documents in `workspace/` are indexed automatically on startup. Use `reindex` to re-scan, `clear` to reset conversation, and `exit` to quit.

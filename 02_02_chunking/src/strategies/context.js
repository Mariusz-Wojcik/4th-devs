/**
 * Context-enriched chunking (Anthropic-style contextual retrieval)
 * Splits with separators first, then uses LLM to generate a context prefix per chunk.
 */

import { chat } from "../api.js";
import { chunkBySeparators } from "./separators.js";

const enrichChunk = async (chunk) => {
  const context = await chat(
    `<chunk>${chunk.content}</chunk>`,
    "Generate a very short (1-2 sentence) context that situates this chunk within the overall document. Return ONLY the context, nothing else."
  );
  return {
    content: chunk.content,
    metadata: { ...chunk.metadata, strategy: "context", context },
  };
};

export const chunkWithContext = async (text, opts = {}) => {
  const base = chunkBySeparators(text, opts);
  const enriched = [];

  for (const [i, chunk] of base.entries()) {
    process.stdout.write(`  context: enriching ${i + 1}/${base.length}\r`);
    enriched.push(await enrichChunk(chunk));
  }
  console.log();

  return enriched;
};

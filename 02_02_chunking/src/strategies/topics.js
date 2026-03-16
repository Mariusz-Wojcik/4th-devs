/**
 * Topic-based (AI-driven) chunking
 * Uses the LLM to identify logical topic boundaries and returns chunks per topic.
 */

import { chat } from "../api.js";
import { buildHeadingIndex, findSection } from "../utils.js";

export const chunkByTopics = async (text, { source } = {}) => {
  const raw = await chat(
    text,
    `You are a document chunking expert. Break the provided document into logical topic-based chunks.

Rules:
- Each chunk must contain ONE coherent topic or idea
- Preserve the original text — do NOT summarise or rewrite
- Return a JSON array of objects: [{ "topic": "short topic label", "content": "original text for this topic" }]
- Return ONLY the JSON array, no markdown fences or explanation`
  );

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    const cleaned = raw.replace(/```json?\n?/g, "").replace(/```/g, "").trim();
    parsed = JSON.parse(cleaned);
  }

  const headings = buildHeadingIndex(text);

  return parsed.map((item, i) => ({
    content: item.content,
    metadata: {
      strategy: "topics",
      index: i,
      topic: item.topic,
      chars: item.content.length,
      section: findSection(text, item.content, headings),
      source: source ?? null,
    },
  }));
};

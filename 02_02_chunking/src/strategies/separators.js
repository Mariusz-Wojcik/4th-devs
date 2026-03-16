/**
 * Separator-based (recursive) chunking
 * Splits text using a hierarchy of separators: headers → paragraphs → sentences → words.
 */

import { buildHeadingIndex, findSection } from "../utils.js";

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "];

const pickOverlap = (text, overlap, sep) => {
  if (overlap <= 0) return "";

  const start = Math.max(0, text.length - overlap);
  const tail = text.slice(start);

  let idx = tail.search(/\n/);
  if (idx === -1) idx = tail.search(/\s/);

  if (idx === -1) return "";

  let overlapText = text.slice(start + idx + 1);

  if (sep && overlapText.startsWith(sep)) {
    overlapText = overlapText.slice(sep.length);
  }

  return overlapText;
};

const split = (text, size, overlap, separators, stats) => {
  if (text.length <= size) return [text];

  const sep = separators.find((s) => text.includes(s));
  if (!sep) return [text];

  const parts = text.split(sep);
  const chunks = [];
  let current = "";

  for (const part of parts) {
    const candidate = current ? current + sep + part : part;
    if (candidate.length > size && current) {
      chunks.push(current);
      const overlapText = pickOverlap(current, overlap, sep);
      if (!overlapText) stats.dropped += 1;
      if (overlapText && overlapText.length < overlap) stats.trimmed += 1;
      current = overlapText ? overlapText + sep + part : part;
    } else {
      current = candidate;
    }
  }
  if (current) chunks.push(current);

  const remaining = separators.slice(separators.indexOf(sep) + 1);
  return chunks.flatMap((c) =>
    c.length > size && remaining.length ? split(c, size, overlap, remaining, stats) : [c]
  );
};

export const chunkBySeparators = (text, { source, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP } = {}) => {
  const stats = { trimmed: 0, dropped: 0 };
  const chunks = split(text, size, overlap, SEPARATORS, stats);
  const headings = buildHeadingIndex(text);
  console.log(`[separators] overlap trimmed: ${stats.trimmed}, dropped: ${stats.dropped}`);

  return chunks.map((content, i) => ({
    content,
    metadata: {
      strategy: "separators",
      index: i,
      chars: content.length,
      section: findSection(text, content, headings),
      source: source ?? null,
    },
  }));
};

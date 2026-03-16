/**
 * Separator-based (recursive) chunking.
 * Splits text using a hierarchy of separators: headers → paragraphs → sentences → words.
 *
 * Ported from 02_02_chunking/src/strategies/separators.js
 */

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "];

// ── Heading index utilities ────────────────────────────────────

export const buildHeadingIndex = (text) => {
  const headings = [];
  let match;

  const mdRegex = /^(#{1,6})\s+(.+)$/gm;
  while ((match = mdRegex.exec(text)) !== null) {
    headings.push({ position: match.index, level: match[1].length, title: match[2].trim() });
  }

  const plainRegex = /(?:^|\n\n)([^\n]{1,80})\n(?=[A-Za-z"'\[(])/gm;
  const mdTitles = new Set(headings.map((h) => h.title));

  while ((match = plainRegex.exec(text)) !== null) {
    const title = match[1].trim();
    if (!title || title === "Conclusion:" || mdTitles.has(title)) continue;

    const offset = match[0].startsWith("\n") ? 2 : 0;
    headings.push({ position: match.index + offset, level: 1, title });
  }

  return headings.sort((a, b) => a.position - b.position);
};

export const findSection = (text, chunkContent, headings) => {
  if (!headings.length) return null;

  const mid = Math.floor(chunkContent.length * 0.4);
  const sample = chunkContent.slice(mid, mid + 100);
  const pos = text.indexOf(sample);
  if (pos === -1) return null;

  let current = null;
  for (const h of headings) {
    if (h.position <= pos) current = h;
    else break;
  }

  return current ? `${"#".repeat(current.level)} ${current.title}` : null;
};

// ── Overlap helper ─────────────────────────────────────────────

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

// ── Recursive split ────────────────────────────────────────────

const split = (text, size, overlap, separators) => {
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
      current = overlapText ? overlapText + sep + part : part;
    } else {
      current = candidate;
    }
  }
  if (current) chunks.push(current);

  const remaining = separators.slice(separators.indexOf(sep) + 1);
  return chunks.flatMap((c) =>
    c.length > size && remaining.length ? split(c, size, overlap, remaining) : [c]
  );
};

// ── Public API ─────────────────────────────────────────────────

export const chunkBySeparators = (text, { source, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP } = {}) => {
  const chunks = split(text, size, overlap, SEPARATORS);
  const headings = buildHeadingIndex(text);

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

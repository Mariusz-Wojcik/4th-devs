/**
 * Builds a heading index from markdown text.
 * Detects both markdown `#` headings and plain-text headings
 * (short standalone lines followed immediately by content).
 *
 * Returns a sorted array of { position, level, title }.
 */
export const buildHeadingIndex = (text) => {
  const headings = [];
  let match;

  // 1. Markdown # headings
  const mdRegex = /^(#{1,6})\s+(.+)$/gm;
  while ((match = mdRegex.exec(text)) !== null) {
    headings.push({
      position: match.index,
      level: match[1].length,
      title: match[2].trim(),
    });
  }

  // 2. Plain-text headings: short line after blank line (or start),
  //    followed by content on the very next line (single \n, not \n\n).
  const plainRegex = /(?:^|\n\n)([^\n]{1,80})\n(?=[A-Za-z"'\[(])/gm;
  const mdTitles = new Set(headings.map((h) => h.title));

  while ((match = plainRegex.exec(text)) !== null) {
    const title = match[1].trim();
    if (!title || title === "Conclusion:" || mdTitles.has(title)) continue;

    const offset = match[0].startsWith("\n") ? 2 : 0;
    headings.push({
      position: match.index + offset,
      level: 1,
      title,
    });
  }

  return headings.sort((a, b) => a.position - b.position);
};

/**
 * Finds the most recent heading for a chunk based on its position in the source text.
 * Samples from the middle of the chunk to avoid overlap-related false matches.
 */
export const findSection = (text, chunkContent, headings) => {
  if (!headings.length) return null;

  // Use a mid-chunk sample so overlapping start/end don't mislead indexOf
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

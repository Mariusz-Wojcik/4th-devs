/**
 * Character-based chunking
 * Splits text into fixed-size chunks by character count with overlap.
 */

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;

export const chunkByCharacters = (text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) => {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    chunks.push(text.slice(start, start + size));
    start += size - overlap;
  }

  return chunks.map((content, i) => ({
    content,
    metadata: { strategy: "characters", index: i, chars: content.length, size, overlap },
  }));
};

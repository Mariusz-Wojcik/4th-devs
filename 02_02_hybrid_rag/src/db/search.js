/**
 * Hybrid search: FTS5 (BM25) + sqlite-vec (cosine distance), combined with Reciprocal Rank Fusion.
 */

import { embed } from "./embeddings.js";
import log from "../helpers/logger.js";

const RRF_K = 60; // Reciprocal Rank Fusion constant

const toVecBuffer = (arr) => {
  const f32 = new Float32Array(arr);
  return Buffer.from(f32.buffer);
};

/**
 * Sanitize a query string for FTS5 MATCH syntax.
 * Strips special characters and joins terms with OR for broad matching.
 */
const toFtsQuery = (query) => {
  const terms = query
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 1);

  if (!terms.length) return null;
  return terms.map((t) => `"${t}"`).join(" OR ");
};

/**
 * Extract unique matched terms from FTS5 highlight() output.
 */
const extractMatchedTerms = (highlighted) => {
  const matches = [...highlighted.matchAll(/«([^»]+)»/g)];
  return [...new Set(matches.map((m) => m[1].toLowerCase()))];
};

/**
 * Full-text search using FTS5 with BM25 ranking.
 * Uses highlight() to identify which keywords triggered each match.
 */
export const searchFts = (db, query, limit = 10) => {
  const ftsQuery = toFtsQuery(query);
  if (!ftsQuery) return [];

  try {
    const rows = db
      .prepare(
        `SELECT c.id, c.content, c.section, c.chunk_index, d.source,
                rank AS fts_score,
                highlight(chunks_fts, 0, '«', '»') AS highlighted
         FROM chunks_fts
         JOIN chunks c ON c.id = chunks_fts.rowid
         JOIN documents d ON d.id = c.document_id
         WHERE chunks_fts MATCH ?
         ORDER BY rank
         LIMIT ?`
      )
      .all(ftsQuery, limit);

    return rows.map(({ highlighted, ...rest }) => ({
      ...rest,
      matched_terms: extractMatchedTerms(highlighted),
    }));
  } catch {
    return [];
  }
};

/**
 * Vector similarity search using sqlite-vec.
 */
export const searchVector = (db, queryEmbedding, limit = 10) => {
  const rows = db
    .prepare(
      `SELECT chunk_id, distance
       FROM chunks_vec
       WHERE embedding MATCH ?
       ORDER BY distance
       LIMIT ?`
    )
    .all(toVecBuffer(queryEmbedding), limit);

  if (!rows.length) return [];

  const ids = rows.map((r) => r.chunk_id);
  const placeholders = ids.map(() => "?").join(",");

  const chunks = db
    .prepare(
      `SELECT c.id, c.content, c.section, c.chunk_index, d.source
       FROM chunks c
       JOIN documents d ON d.id = c.document_id
       WHERE c.id IN (${placeholders})`
    )
    .all(...ids);

  const chunkMap = new Map(chunks.map((c) => [c.id, c]));

  return rows
    .map((r) => ({ ...chunkMap.get(r.chunk_id), vec_distance: r.distance }))
    .filter(Boolean);
};

/**
 * Hybrid search: runs FTS5 + vector search with separate queries, merges with RRF.
 *
 * @param {object} db - SQLite database
 * @param {{ keywords: string, semantic: string }} query - Separate queries for each search type
 * @param {number} limit - Max results
 */
export const hybridSearch = async (db, { keywords, semantic }, limit = 5) => {
  const ftsLimit = limit * 3;

  log.searchHeader(keywords, semantic);

  // FTS runs first — synchronous, always available
  const ftsResults = searchFts(db, keywords, ftsLimit);
  log.searchFts(ftsResults);

  // Vector search — may fail (API issues), degrade gracefully to FTS-only
  let vecResults = [];
  try {
    const [queryEmbedding] = await embed(semantic);
    vecResults = searchVector(db, queryEmbedding, ftsLimit);
  } catch (err) {
    log.warn(`Semantic search unavailable: ${err.message}`);
  }
  log.searchVec(vecResults);

  // Build RRF scores
  const scores = new Map();

  const upsert = (id, data) => {
    if (!scores.has(id)) scores.set(id, { ...data, rrf: 0 });
    return scores.get(id);
  };

  ftsResults.forEach((r, rank) => {
    const entry = upsert(r.id, r);
    entry.rrf += 1 / (RRF_K + rank + 1);
    entry.fts_rank = rank + 1;
  });

  vecResults.forEach((r, rank) => {
    const entry = upsert(r.id, r);
    entry.rrf += 1 / (RRF_K + rank + 1);
    entry.vec_rank = rank + 1;
    entry.vec_distance = r.vec_distance;
  });

  const merged = [...scores.values()]
    .sort((a, b) => b.rrf - a.rrf)
    .slice(0, limit);

  log.searchRrf(merged);

  return merged.map(({ rrf, fts_score, ...rest }) => rest);
};

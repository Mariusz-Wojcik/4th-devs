/**
 * Workspace indexer.
 * Reads text files from a directory, chunks them, generates embeddings,
 * and inserts into SQLite (documents + chunks + FTS5 + vec0).
 */

import { readdir, readFile, mkdir } from "fs/promises";
import { join } from "path";
import { createHash } from "crypto";
import { chunkBySeparators } from "./chunking.js";
import { embed } from "./embeddings.js";
import log from "../helpers/logger.js";

const BATCH_SIZE = 20;
const SUPPORTED_EXT = new Set([".md", ".txt"]);

const hashContent = (content) =>
  createHash("sha256").update(content).digest("hex");

const toVecBuffer = (arr) => {
  const f32 = new Float32Array(arr);
  return Buffer.from(f32.buffer);
};

/**
 * Remove all indexed data for a document (vec, chunks/fts, document row).
 */
const removeDocument = (db, docId) => {
  db.prepare(
    "DELETE FROM chunks_vec WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)"
  ).run(docId);
  db.prepare("DELETE FROM chunks WHERE document_id = ?").run(docId);
  db.prepare("DELETE FROM documents WHERE id = ?").run(docId);
};

/**
 * Index a single file: chunk → embed → insert.
 */
const indexFile = async (db, filePath, fileName) => {
  const content = await readFile(filePath, "utf-8");
  if (!content.trim()) {
    log.warn(`Skipping empty file: ${fileName}`);
    return;
  }

  const hash = hashContent(content);

  const existing = db
    .prepare("SELECT id, hash FROM documents WHERE source = ?")
    .get(fileName);

  if (existing?.hash === hash) {
    log.info(`Skipping ${fileName} (unchanged)`);
    return;
  }

  if (existing) {
    log.info(`Re-indexing ${fileName} (changed)`);
    removeDocument(db, existing.id);
  }

  // 1. Chunk
  const chunks = chunkBySeparators(content, { source: fileName });
  log.info(`${fileName}: ${chunks.length} chunks`);

  // 2. Insert document
  const docId = db
    .prepare("INSERT INTO documents (source, content, hash) VALUES (?, ?, ?)")
    .run(fileName, content, hash).lastInsertRowid;

  // 3. Insert chunks (triggers populate FTS5 automatically)
  const insertChunk = db.prepare(
    "INSERT INTO chunks (document_id, content, chunk_index, section, chars) VALUES (?, ?, ?, ?, ?)"
  );

  const chunkIds = chunks.map((chunk) => {
    const { lastInsertRowid } = insertChunk.run(
      docId,
      chunk.content,
      chunk.metadata.index,
      chunk.metadata.section,
      chunk.metadata.chars
    );
    return BigInt(lastInsertRowid);
  });

  // 4. Generate embeddings in batches
  const contents = chunks.map((c) => c.content);
  const embeddings = [];

  for (let i = 0; i < contents.length; i += BATCH_SIZE) {
    const batch = contents.slice(i, i + BATCH_SIZE);
    const batchEmbeddings = await embed(batch);
    embeddings.push(...batchEmbeddings);
    process.stdout.write(`  embeddings: ${embeddings.length}/${contents.length}\r`);
  }
  if (contents.length > BATCH_SIZE) console.log();

  // 5. Insert vectors
  const insertVec = db.prepare(
    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)"
  );

  for (let i = 0; i < chunkIds.length; i++) {
    insertVec.run(chunkIds[i], toVecBuffer(embeddings[i]));
  }

  log.success(`Indexed ${fileName}: ${chunks.length} chunks`);
};

/**
 * Index all supported files in the workspace directory.
 */
export const indexWorkspace = async (db, workspacePath) => {
  await mkdir(workspacePath, { recursive: true });

  const files = (await readdir(workspacePath)).filter((f) => {
    const ext = f.slice(f.lastIndexOf("."));
    return SUPPORTED_EXT.has(ext);
  });

  if (!files.length) {
    log.warn(`No .md/.txt files found in ${workspacePath}`);
    return;
  }

  log.info(`Found ${files.length} file(s) in ${workspacePath}`);

  for (const file of files) {
    await indexFile(db, join(workspacePath, file), file);
  }

  // Prune documents that no longer exist on disk
  const indexed = db.prepare("SELECT id, source FROM documents").all();
  const onDisk = new Set(files);

  for (const doc of indexed) {
    if (!onDisk.has(doc.source)) {
      log.info(`Removing stale index: ${doc.source}`);
      removeDocument(db, doc.id);
    }
  }
};

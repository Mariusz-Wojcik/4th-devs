/**
 * SQLite database with FTS5 (full-text search) and sqlite-vec (vector search).
 */

import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { mkdirSync } from "fs";
import { dirname } from "path";
import log from "../helpers/logger.js";

const EMBEDDING_DIM = 1536; // openai/text-embedding-3-small

export const initDb = (dbPath = "data/hybrid.db") => {
  mkdirSync(dirname(dbPath), { recursive: true });

  const db = new Database(dbPath);
  sqliteVec.load(db);

  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");
  db.pragma("foreign_keys = ON");
  db.pragma("busy_timeout = 5000");

  const version = db.prepare("SELECT vec_version() AS v").get();
  log.info(`sqlite-vec ${version.v}`);

  db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source TEXT NOT NULL UNIQUE,
      content TEXT NOT NULL,
      hash TEXT NOT NULL,
      indexed_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      content TEXT NOT NULL,
      chunk_index INTEGER NOT NULL,
      section TEXT,
      chars INTEGER NOT NULL
    );

    -- FTS5 external-content table backed by chunks
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      content,
      content='chunks',
      content_rowid='id'
    );

    -- Triggers to keep FTS5 in sync with chunks table
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
      INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END;

    -- sqlite-vec virtual table for vector similarity search
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
      chunk_id INTEGER PRIMARY KEY,
      embedding float[${EMBEDDING_DIM}]
    );
  `);

  return db;
};

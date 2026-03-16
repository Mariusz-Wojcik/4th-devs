/**
 * Hybrid RAG Agent
 *
 * Indexes workspace text files into SQLite (FTS5 + sqlite-vec),
 * then runs an interactive agent that searches via hybrid retrieval.
 */

import { initDb } from "./src/db/index.js";
import { indexWorkspace } from "./src/db/indexer.js";
import { createTools } from "./src/agent/tools.js";
import { createReadline, runRepl } from "./src/repl.js";
import { onShutdown } from "./src/helpers/shutdown.js";
import { logStats } from "./src/helpers/stats.js";
import log from "./src/helpers/logger.js";

const main = async () => {
  log.box("Hybrid RAG Agent\nCommands: 'exit' | 'clear' | 'reindex'");

  // 1. Database
  log.start("Initializing database...");
  const db = initDb();
  log.success("Database ready");

  // 2. Index workspace
  log.start("Indexing workspace...");
  await indexWorkspace(db, "workspace");
  log.success("Indexing complete");

  // 3. Agent tools
  const tools = createTools(db);

  // 4. REPL
  const rl = createReadline();

  const shutdown = onShutdown(async () => {
    logStats();
    rl?.close();
    db.close();
  });

  await runRepl({ tools, rl, db });
  await shutdown();
};

main().catch((err) => {
  log.error("Startup error", err.message);
  process.exit(1);
});

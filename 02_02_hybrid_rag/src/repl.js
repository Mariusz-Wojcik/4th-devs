/**
 * Interactive REPL for the hybrid RAG agent.
 */

import * as readline from "readline/promises";
import { run, createConversation } from "./agent/index.js";
import { indexWorkspace } from "./db/indexer.js";
import { resetStats } from "./helpers/stats.js";
import log from "./helpers/logger.js";

export const createReadline = () =>
  readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

/**
 * @param {object} options
 * @param {object} options.tools - Native tools from createTools()
 * @param {object} options.rl - Readline interface
 * @param {object} options.db - SQLite database instance
 */
export const runRepl = async ({ tools, rl, db }) => {
  let conversation = createConversation();

  while (true) {
    const input = await rl.question("You: ").catch(() => "exit");

    if (input.toLowerCase() === "exit") break;

    if (input.toLowerCase() === "clear") {
      conversation = createConversation();
      resetStats();
      log.success("Conversation cleared\n");
      continue;
    }

    if (input.toLowerCase() === "reindex") {
      log.start("Re-indexing workspace...");
      await indexWorkspace(db, "workspace");
      log.success("Re-indexing complete\n");
      continue;
    }

    if (!input.trim()) continue;

    try {
      const result = await run(input, {
        tools,
        conversationHistory: conversation.history,
      });

      conversation.history = result.conversationHistory;
      console.log(`\nAssistant: ${result.response}\n`);
    } catch (err) {
      log.error("Error", err.message);
      console.log("");
    }
  }
};

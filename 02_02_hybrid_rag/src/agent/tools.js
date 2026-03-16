/**
 * Native tools for the agent.
 * Provides a single hybrid search tool over the indexed document database.
 */

import { hybridSearch } from "../db/search.js";
import log from "../helpers/logger.js";

const SEARCH_TOOL = {
  type: "function",
  name: "search",
  description:
    "Search the indexed knowledge base using hybrid search (full-text BM25 + semantic vector similarity). " +
    "Returns the most relevant document chunks with content, source file, and section heading. " +
    "Provide BOTH a keyword query for full-text search AND a natural language query for semantic search.",
  parameters: {
    type: "object",
    properties: {
      keywords: {
        type: "string",
        description:
          "Keywords for full-text search (BM25) — specific terms, names, and phrases that should appear in the text",
      },
      semantic: {
        type: "string",
        description:
          "Natural language query for semantic/vector search — a question or description of the concept you're looking for",
      },
      limit: {
        type: "number",
        description: "Maximum number of results to return (default 5, max 20)",
      },
    },
    required: ["keywords", "semantic"],
  },
  strict: false,
};

/**
 * Creates the tool interface consumed by the agent.
 *
 * @param {import("better-sqlite3").Database} db
 * @returns {{ definitions: object[], handle: (name: string, args: object) => Promise<any> }}
 */
export const createTools = (db) => {
  const handlers = {
    search: async ({ keywords, semantic, limit = 5 }) => {
      const results = await hybridSearch(db, { keywords, semantic }, Math.min(limit, 20));

      return results.map((r) => ({
        source: r.source,
        section: r.section,
        content: r.content,
      }));
    },
  };

  return {
    definitions: [SEARCH_TOOL],

    handle: async (name, args) => {
      const handler = handlers[name];
      if (!handler) throw new Error(`Unknown tool: ${name}`);

      log.tool(name, args);

      try {
        const result = await handler(args);
        const output = JSON.stringify(result);
        log.toolResult(name, true, output);
        return output;
      } catch (error) {
        const output = JSON.stringify({ error: error.message });
        log.toolResult(name, false, error.message);
        return output;
      }
    },
  };
};

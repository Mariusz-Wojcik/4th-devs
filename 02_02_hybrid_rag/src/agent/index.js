/**
 * Agent Loop
 *
 * Executes chat → tool calls → results cycle until completion.
 * Uses native tools (hybrid search) for document retrieval.
 * MCP tools can be added later alongside native tools.
 */

import { chat, extractToolCalls, extractText, extractReasoning } from "../helpers/api.js";
import log from "../helpers/logger.js";

const MAX_STEPS = 30;

// ─────────────────────────────────────────────────────────────
// Tool Execution
// ─────────────────────────────────────────────────────────────

const runTool = async (tools, toolCall) => {
  const args = JSON.parse(toolCall.arguments);
  const output = await tools.handle(toolCall.name, args);
  return { type: "function_call_output", call_id: toolCall.call_id, output };
};

const runTools = (tools, toolCalls) =>
  Promise.all(toolCalls.map((tc) => runTool(tools, tc)));

// ─────────────────────────────────────────────────────────────
// Agent Loop
// ─────────────────────────────────────────────────────────────

/**
 * Run the agent with a query.
 *
 * @param {string} query - User query
 * @param {object} options
 * @param {object} options.tools - { definitions, handle } from createTools()
 * @param {array}  options.conversationHistory - Previous messages
 * @returns {{ response: string, conversationHistory: object[] }}
 */
export const run = async (query, { tools, conversationHistory = [] }) => {
  const toolDefs = tools.definitions;
  const messages = [...conversationHistory, { role: "user", content: query }];

  log.query(query);

  for (let step = 1; step <= MAX_STEPS; step++) {
    log.api(`Step ${step}`, messages.length);
    const response = await chat({ input: messages, tools: toolDefs });
    log.apiDone(response.usage);
    log.reasoning(extractReasoning(response));

    const toolCalls = extractToolCalls(response);

    if (toolCalls.length === 0) {
      const text = extractText(response) ?? "No response";
      log.response(text);

      messages.push(...response.output);

      return {
        response: text,
        conversationHistory: messages,
      };
    }

    messages.push(...response.output);

    const results = await runTools(tools, toolCalls);
    messages.push(...results);
  }

  throw new Error(`Max steps (${MAX_STEPS}) reached`);
};

/**
 * Create a new conversation context.
 */
export const createConversation = () => ({
  history: [],
});

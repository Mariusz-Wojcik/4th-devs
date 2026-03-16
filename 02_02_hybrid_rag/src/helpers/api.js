import { AI_API_KEY, RESPONSES_API_ENDPOINT, EXTRA_API_HEADERS } from "../../../config.js";
import { api } from "../config.js";
import { recordUsage } from "./stats.js";

export const chat = async ({ 
  model = api.model, 
  input, 
  tools, 
  toolChoice = "auto", 
  instructions = api.instructions, 
  maxOutputTokens = api.maxOutputTokens,
  reasoning = api.reasoning
}) => {
  const body = { model, input };

  if (tools?.length) body.tools = tools;
  if (tools?.length) body.tool_choice = toolChoice;
  if (instructions) body.instructions = instructions;
  if (maxOutputTokens) body.max_output_tokens = maxOutputTokens;
  if (reasoning) body.reasoning = reasoning;

  const response = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();

  if (data.error) {
    throw new Error(data.error.message);
  }

  recordUsage(data.usage);
  return data;
};

/**
 * Extracts function calls from response.
 */
export const extractToolCalls = (response) =>
  response.output.filter((item) => item.type === "function_call");

/**
 * Extracts text content from response.
 */
export const extractText = (response) => {
  const message = response.output.find((item) => item.type === "message");
  return message?.content?.[0]?.text ?? null;
};

/**
 * Extracts reasoning summaries from response output.
 */
export const extractReasoning = (response) =>
  response.output
    .filter((item) => item.type === "reasoning")
    .flatMap((item) => item.summary ?? [])
    .map((s) => s.text)
    .filter(Boolean);

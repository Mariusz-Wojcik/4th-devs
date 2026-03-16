/**
 * Token usage statistics tracker.
 */

let totalTokens = { input: 0, output: 0, reasoning: 0, cached: 0, requests: 0 };

export const recordUsage = (usage) => {
  if (!usage) return;
  totalTokens.input += usage.input_tokens || 0;
  totalTokens.output += usage.output_tokens || 0;
  totalTokens.reasoning += usage.output_tokens_details?.reasoning_tokens || 0;
  totalTokens.cached += usage.input_tokens_details?.cached_tokens || 0;
  totalTokens.requests += 1;
};

export const getStats = () => ({ ...totalTokens });

export const logStats = () => {
  const { input, output, reasoning, cached, requests } = totalTokens;
  const visible = output - reasoning;
  let summary = `${requests} requests, ${input} in`;
  if (cached > 0) summary += ` (${cached} cached)`;
  summary += `, ${output} out`;
  if (reasoning > 0) summary += ` (${reasoning} reasoning + ${visible} visible)`;
  console.log(`\n📊 Stats: ${summary}\n`);
};

export const resetStats = () => {
  totalTokens = { input: 0, output: 0, reasoning: 0, cached: 0, requests: 0 };
};

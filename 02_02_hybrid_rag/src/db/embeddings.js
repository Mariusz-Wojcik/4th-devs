import { AI_API_KEY, EMBEDDINGS_API_ENDPOINT, EXTRA_API_HEADERS, resolveModelForProvider } from "../../../config.js";

const MODEL = resolveModelForProvider("text-embedding-3-small");

export const embed = async (texts) => {
  const input = Array.isArray(texts) ? texts : [texts];

  const response = await fetch(EMBEDDINGS_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({ model: MODEL, input }),
  });

  const data = await response.json();

  if (data.error) {
    throw new Error(`Embedding error: ${data.error.message ?? JSON.stringify(data.error)}`);
  }

  return data.data
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
};

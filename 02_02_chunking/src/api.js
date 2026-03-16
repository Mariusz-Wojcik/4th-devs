import { AI_API_KEY, RESPONSES_API_ENDPOINT, EXTRA_API_HEADERS, resolveModelForProvider } from "../../config.js";

const DEFAULT_MODEL = resolveModelForProvider("gpt-4.1-mini");

export const chat = async (input, instructions, model = DEFAULT_MODEL) => {
  const body = { model, input };
  if (instructions) body.instructions = instructions;

  const res = await fetch(RESPONSES_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify(body),
  });

  const data = await res.json();
  if (data.error) throw new Error(data.error.message);

  const message = data.output.find((item) => item.type === "message");
  return message?.content?.[0]?.text ?? "";
};

/**
 * Embedding demo with similarity matrix.
 * Each input is embedded and a full pairwise similarity matrix is displayed,
 * making it easy to see which inputs cluster together.
 */

import * as readline from "readline/promises";
import { AI_API_KEY, EMBEDDINGS_API_ENDPOINT, EXTRA_API_HEADERS, resolveModelForProvider } from "../config.js";

const MODEL = resolveModelForProvider("text-embedding-3-small");

// ── Colors ─────────────────────────────────────────────────────

const c = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  bold: "\x1b[1m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  cyan: "\x1b[36m",
  bg: "\x1b[7m",
};

const colorFor = (score) =>
  score >= 0.6 ? c.green : score >= 0.35 ? c.yellow : c.red;

// ── Embedding API ──────────────────────────────────────────────

const embed = async (text) => {
  const response = await fetch(EMBEDDINGS_API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AI_API_KEY}`,
      ...EXTRA_API_HEADERS
    },
    body: JSON.stringify({ model: MODEL, input: text }),
  });

  const data = await response.json();
  if (data.error) throw new Error(data.error.message ?? JSON.stringify(data.error));

  return data.data[0].embedding;
};

// ── Math ───────────────────────────────────────────────────────

const cosineSimilarity = (a, b) => {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
};

// ── Display ────────────────────────────────────────────────────

const preview = (embedding) => {
  const head = embedding.slice(0, 4).map((v) => v.toFixed(4)).join(", ");
  const tail = embedding.slice(-2).map((v) => v.toFixed(4)).join(", ");
  return `${c.dim}[${head}, …, ${tail}]${c.reset} ${c.cyan}(${embedding.length}d)${c.reset}`;
};

const LABEL_WIDTH = 14;

const truncate = (text, width) =>
  text.length > width ? text.slice(0, width - 1) + "…" : text;

const pad = (text, width) => text.padEnd(width);
const padStart = (text, width) => text.padStart(width);

const printMatrix = (entries) => {
  const labels = entries.map((e) => truncate(e.text, LABEL_WIDTH));
  const colWidth = Math.max(LABEL_WIDTH, ...labels.map((l) => l.length)) + 1;

  // Header row
  const header =
    pad("", LABEL_WIDTH + 2) +
    labels.map((l) => `${c.bold}${padStart(l, colWidth)}${c.reset}`).join("");

  console.log(`\n${header}`);

  // Matrix rows
  for (let i = 0; i < entries.length; i++) {
    const rowLabel = `${c.bold}${pad(labels[i], LABEL_WIDTH)}${c.reset}  `;
    const cells = entries.map((_, j) => {
      if (i === j) return padStart(`${c.dim}  ——${c.reset}`, colWidth + 8);

      const score = cosineSimilarity(entries[i].embedding, entries[j].embedding);
      const color = colorFor(score);
      const bar = "█".repeat(Math.round(score * 8));
      const value = score.toFixed(2);

      return padStart(`${color}${bar} ${value}${c.reset}`, colWidth + 8);
    });

    console.log(rowLabel + cells.join(""));
  }

  // Legend
  console.log(
    `\n  ${c.dim}Legend:${c.reset} ${c.green}███ ≥0.60 similar${c.reset}  ${c.yellow}███ ≥0.35 related${c.reset}  ${c.red}███ <0.35 distant${c.reset}`
  );
};

// ── REPL ───────────────────────────────────────────────────────

const main = async () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const entries = [];

  console.log(`\n${c.cyan}Embedding + Similarity Matrix${c.reset} (model: ${MODEL})`);
  console.log(`Type 'exit' to quit.\n`);

  while (true) {
    const input = await rl.question("Text: ").catch(() => "exit");

    if (input.toLowerCase() === "exit" || !input.trim()) break;

    try {
      const embedding = await embed(input);
      entries.push({ text: input, embedding });

      console.log(`\n  "${input}" → ${preview(embedding)}`);

      if (entries.length === 1) {
        console.log(`${c.dim}  Add more to see similarities.${c.reset}`);
        console.log();
        continue;
      }

      printMatrix(entries);
      console.log();
    } catch (err) {
      console.error(`Error: ${err.message}\n`);
    }
  }

  rl.close();
};

main();

/**
 * Runs all four chunking strategies on workspace/example.md
 * and saves results as JSONL to workspace/example-[type].jsonl
 */

import { createInterface } from "node:readline/promises";
import { readFile, writeFile } from "fs/promises";
import { chunkByCharacters } from "./src/strategies/characters.js";
import { chunkBySeparators } from "./src/strategies/separators.js";
import { chunkWithContext } from "./src/strategies/context.js";
import { chunkByTopics } from "./src/strategies/topics.js";

const INPUT = "workspace/example.md";
const DEMO_DIR = "workspace/";

const confirmRun = async () => {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("\n⚠️  UWAGA: Uruchomienie tego przykładu zużyje tokeny (strategie context i topics używają LLM).");
  console.log("   Jeśli nie chcesz uruchamiać go teraz, najpierw sprawdź gotowe wyniki:");
  console.log(`   Demo: ${DEMO_DIR}example-*.jsonl`);
  console.log("");

  const answer = await rl.question("Czy chcesz kontynuować? (yes/y): ");
  rl.close();

  const normalized = answer.trim().toLowerCase();
  if (normalized !== "yes" && normalized !== "y") {
    console.log("Przerwano.");
    process.exit(0);
  }
};

const toJsonl = (chunks) =>
  chunks.map((chunk) => JSON.stringify(chunk)).join("\n");

const save = async (name, chunks) => {
  const path = `workspace/example-${name}.jsonl`;
  await writeFile(path, toJsonl(chunks), "utf-8");
  console.log(`  ✓ ${path} (${chunks.length} chunks)`);
};

const main = async () => {
  await confirmRun();
  const text = await readFile(INPUT, "utf-8");
  const opts = { source: INPUT };
  console.log(`Source: ${INPUT} (${text.length} chars)\n`);

  console.log("1. Characters...");
  await save("characters", chunkByCharacters(text));

  console.log("2. Separators...");
  await save("separators", chunkBySeparators(text, opts));

  console.log("3. Context (LLM-enriched)...");
  await save("context", await chunkWithContext(text, opts));

  console.log("4. Topics (AI-driven)...");
  await save("topics", await chunkByTopics(text, opts));

  console.log("\nDone.");
};

main().catch(console.error);

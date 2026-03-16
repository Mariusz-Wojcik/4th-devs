/**
 * Simple colored logger for terminal output.
 */

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  white: "\x1b[37m",
  bgBlue: "\x1b[44m",
  bgMagenta: "\x1b[45m"
};

const timestamp = () => new Date().toLocaleTimeString("en-US", { hour12: false });

const log = {
  info: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.green}✓${colors.reset} ${msg}`),
  error: (title, msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.red}✗ ${title}${colors.reset} ${msg || ""}`),
  warn: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.yellow}⚠${colors.reset} ${msg}`),
  start: (msg) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.cyan}→${colors.reset} ${msg}`),
  
  box: (text) => {
    const lines = text.split("\n");
    const width = Math.max(...lines.map(l => l.length)) + 4;
    console.log(`\n${colors.cyan}${"─".repeat(width)}${colors.reset}`);
    for (const line of lines) {
      console.log(`${colors.cyan}│${colors.reset} ${colors.bright}${line.padEnd(width - 3)}${colors.reset}${colors.cyan}│${colors.reset}`);
    }
    console.log(`${colors.cyan}${"─".repeat(width)}${colors.reset}\n`);
  },

  query: (q) => console.log(`\n${colors.bgBlue}${colors.white} QUERY ${colors.reset} ${q}\n`),
  response: (r) => console.log(`\n${colors.green}Response:${colors.reset} ${r.substring(0, 500)}${r.length > 500 ? "..." : ""}\n`),
  
  api: (step, msgCount) => console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.magenta}◆${colors.reset} ${step} (${msgCount} messages)`),
  apiDone: (usage) => {
    if (!usage) return;
    const cached = usage.input_tokens_details?.cached_tokens ?? 0;
    const reasoning = usage.output_tokens_details?.reasoning_tokens ?? 0;
    const visible = usage.output_tokens - reasoning;

    let parts = [`${usage.input_tokens} in`];
    if (cached > 0) parts.push(`${cached} cached`);
    parts.push(`${usage.output_tokens} out`);
    if (reasoning > 0) parts.push(`${colors.cyan}${reasoning} reasoning${colors.dim} + ${visible} visible`);

    console.log(`${colors.dim}         tokens: ${parts.join(" / ")}${colors.reset}`);
  },

  reasoning: (summaries) => {
    if (!summaries?.length) return;
    console.log(`${colors.dim}         ${colors.cyan}reasoning:${colors.reset}`);
    for (const summary of summaries) {
      const lines = summary.split("\n");
      for (const line of lines) {
        console.log(`${colors.dim}           ${line}${colors.reset}`);
      }
    }
  },

  tool: (name, args) => {
    const argStr = JSON.stringify(args);
    const truncated = argStr.length > 300 ? argStr.substring(0, 300) + "..." : argStr;
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.yellow}⚡${colors.reset} ${name} ${colors.dim}${truncated}${colors.reset}`);
  },
  
  toolResult: (name, success, output) => {
    const icon = success ? `${colors.green}✓${colors.reset}` : `${colors.red}✗${colors.reset}`;
    const truncated = output.length > 500 ? output.substring(0, 500) + "..." : output;
    console.log(`${colors.dim}         ${icon} ${truncated}${colors.reset}`);
  },

  vision: (path, question) => {
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.blue}👁${colors.reset} Vision: ${path}`);
    console.log(`${colors.dim}         Q: ${question}${colors.reset}`);
  },

  visionResult: (answer) => {
    const truncated = answer.length > 200 ? answer.substring(0, 200) + "..." : answer;
    console.log(`${colors.dim}         A: ${truncated}${colors.reset}`);
  },

  gemini: (action, detail) => {
    console.log(`${colors.dim}[${timestamp()}]${colors.reset} ${colors.bgMagenta}${colors.white} GEMINI ${colors.reset} ${action}`);
    if (detail) console.log(`${colors.dim}         ${detail}${colors.reset}`);
  },

  geminiResult: (success, msg) => {
    const icon = success ? `${colors.green}✓${colors.reset}` : `${colors.red}✗${colors.reset}`;
    console.log(`${colors.dim}         ${icon} ${msg}${colors.reset}`);
  },

  // ── Search-specific logs ───────────────────────────────────

  searchHeader: (keywords, semantic) => {
    console.log(`${colors.dim}         ${colors.cyan}fts:${colors.reset}${colors.dim}      "${keywords}"${colors.reset}`);
    console.log(`${colors.dim}         ${colors.cyan}semantic:${colors.reset}${colors.dim} "${semantic}"${colors.reset}`);
  },

  searchFts: (results) => {
    const label = `${colors.blue}FTS${colors.reset}`;
    if (!results.length) {
      console.log(`${colors.dim}         ${label} ${colors.dim}(no matches)${colors.reset}`);
      return;
    }
    console.log(`${colors.dim}         ${label} ${results.length} hit(s)${colors.reset}`);
    for (const r of results.slice(0, 5)) {
      const src = r.source;
      const section = r.section ? ` → ${r.section}` : "";
      const score = r.fts_score?.toFixed(2) ?? "?";
      const terms = r.matched_terms?.length ? ` ${colors.yellow}[${r.matched_terms.join(", ")}]${colors.reset}${colors.dim}` : "";
      console.log(`${colors.dim}           #${r.chunk_index} ${src}${section} (bm25: ${score})${terms}${colors.reset}`);
    }
    if (results.length > 5) console.log(`${colors.dim}           ... +${results.length - 5} more${colors.reset}`);
  },

  searchVec: (results) => {
    const label = `${colors.magenta}VEC${colors.reset}`;
    if (!results.length) {
      console.log(`${colors.dim}         ${label} ${colors.dim}(no matches)${colors.reset}`);
      return;
    }
    console.log(`${colors.dim}         ${label} ${results.length} hit(s)${colors.reset}`);
    for (const r of results.slice(0, 5)) {
      const src = r.source;
      const section = r.section ? ` → ${r.section}` : "";
      const dist = r.vec_distance?.toFixed(3) ?? "?";
      console.log(`${colors.dim}           #${r.chunk_index} ${src}${section} (dist: ${dist})${colors.reset}`);
    }
    if (results.length > 5) console.log(`${colors.dim}           ... +${results.length - 5} more${colors.reset}`);
  },

  searchRrf: (results) => {
    const label = `${colors.green}RRF${colors.reset}`;
    console.log(`${colors.dim}         ${label} ${results.length} merged result(s)${colors.reset}`);
    for (const r of results) {
      const src = r.source;
      const section = r.section ? ` → ${r.section}` : "";
      const fts = r.fts_rank ? `fts:#${r.fts_rank}` : "—";
      const vec = r.vec_rank ? `vec:#${r.vec_rank}` : "—";
      const rrf = r.rrf?.toFixed(4) ?? "?";
      console.log(`${colors.dim}           ${src}${section} [${fts} ${vec}] rrf=${rrf}${colors.reset}`);
    }
  }
};

export default log;

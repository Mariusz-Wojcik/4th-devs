import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { readFile } from "fs/promises";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import log from "../helpers/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, "../..");

/**
 * Loads MCP server configuration from mcp.json
 */
const loadMcpConfig = async () => {
  const configPath = join(PROJECT_ROOT, "mcp.json");
  const content = await readFile(configPath, "utf-8");
  return JSON.parse(content);
};

/**
 * Creates an MCP client for a specific server from config.
 */
export const createMcpClient = async (serverName = "files") => {
  const config = await loadMcpConfig();
  const serverConfig = config.mcpServers[serverName];

  if (!serverConfig) {
    throw new Error(`MCP server "${serverName}" not found in mcp.json`);
  }

  const client = new Client(
    { name: "video-generation-client", version: "1.0.0" },
    { capabilities: {} }
  );

  log.info(`Spawning MCP server: ${serverName}`);
  log.info(`Command: ${serverConfig.command} ${serverConfig.args.join(" ")}`);

  const transport = new StdioClientTransport({
    command: serverConfig.command,
    args: serverConfig.args,
    env: {
      PATH: process.env.PATH,
      HOME: process.env.HOME,
      NODE_ENV: process.env.NODE_ENV,
      ...serverConfig.env
    },
    cwd: PROJECT_ROOT,
    stderr: "inherit"
  });

  await client.connect(transport);
  log.success(`Connected to ${serverName} via stdio`);

  return client;
};

/**
 * Lists all tools available on the MCP server.
 */
export const listMcpTools = async (client) => {
  const result = await client.listTools();
  return result.tools;
};

/**
 * Calls a tool on the MCP server.
 */
export const callMcpTool = async (client, name, args) => {
  const result = await client.callTool({ name, arguments: args });

  const textContent = result.content.find((c) => c.type === "text");
  if (textContent) {
    try {
      return JSON.parse(textContent.text);
    } catch {
      return textContent.text;
    }
  }
  return result;
};

/**
 * Converts MCP tool definitions to OpenAI function format.
 */
export const mcpToolsToOpenAI = (mcpTools) =>
  mcpTools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
    strict: false
  }));

/**
 * Closes the MCP client.
 */
export const closeMcpClient = async (client) => {
  try {
    await client.close();
    log.info("MCP client closed");
  } catch (e) {
    log.warn(`Error closing client: ${e.message}`);
  }
};

import OpenAI from 'openai'
import { join } from 'node:path'
// @ts-ignore — root config is untyped JS
import { AI_API_KEY, CHAT_API_BASE_URL, EXTRA_API_HEADERS, resolveModelForProvider } from '../../config.js'

export { resolveModelForProvider }

export const WORKSPACE = join(process.cwd(), 'workspace')

export const openai = new OpenAI({
  apiKey: AI_API_KEY as string,
  baseURL: CHAT_API_BASE_URL as string,
  defaultHeaders: EXTRA_API_HEADERS as Record<string, string>,
})

export const SERVER_PORT = 3001

export const AGENT_MAX_TURNS = 25

export const TOKEN_CHARS_PER_TOKEN = 4
export const TOKEN_SAFETY_MARGIN = 1.2

export const OBSERVER_MAX_SECTION_CHARS = 6_000
export const OBSERVER_MAX_TOOL_PAYLOAD_CHARS = 3_000
export const OBSERVER_MAX_OUTPUT_TOKENS = 8_000

export const REFLECTOR_MAX_OUTPUT_TOKENS = 10_000

export const MEMORY_DIR = join(WORKSPACE, 'memory')

import type { MemoryConfig } from './types.js'

// Thresholds are intentionally low so the observer/reflector cycle
// triggers within a short demo conversation. In production you'd
// raise these significantly (e.g. 4000 / 2000 / 1200).
export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  observationThresholdTokens: 400,
  reflectionThresholdTokens: 400,
  reflectionTargetTokens: 200,
  observerModel: 'gpt-4.1-mini',
  reflectorModel: 'gpt-4.1-mini',
}

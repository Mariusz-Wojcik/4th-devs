/**
 * Reflector — compresses and reorganizes observations when they exceed token budget.
 *
 * Based on Mastra's Observational Memory system.
 * https://mastra.ai/blog/observational-memory
 */

import OpenAI from 'openai'
import type { CalibrationState, ReflectorResult } from '../types.js'
import { estimateTokens } from '../tokens.js'
import { extractTag } from '../utils.js'
import { log } from '../log.js'
import { REFLECTOR_MAX_OUTPUT_TOKENS } from '../config.js'
import { REFLECTOR_SYSTEM_PROMPT, REFLECTOR_COMPRESSION_LEVELS, buildReflectorPrompt } from './prompts.js'

export const runReflector = async (
  openai: OpenAI,
  model: string,
  observations: string,
  targetTokens: number,
  calibration?: CalibrationState,
): Promise<ReflectorResult> => {
  let bestObservations = observations
  let bestTokens = estimateTokens(observations, calibration)
  let bestRaw = observations
  let bestLevel = -1

  log('reflector', `Compressing observations (${bestTokens} → target ${targetTokens} tokens)`)

  for (let level = 0; level < REFLECTOR_COMPRESSION_LEVELS.length; level += 1) {
    const response = await openai.responses.create({
      model,
      instructions: REFLECTOR_SYSTEM_PROMPT,
      input: buildReflectorPrompt(observations, REFLECTOR_COMPRESSION_LEVELS[level]),
      temperature: 0,
      max_output_tokens: REFLECTOR_MAX_OUTPUT_TOKENS,
      store: false,
    })

    const raw = response.output_text ?? ''
    const compressed = extractTag(raw, 'observations') ?? raw.trim()
    if (!compressed) continue

    const tokens = estimateTokens(compressed, calibration)
    if (tokens < bestTokens) {
      bestObservations = compressed
      bestTokens = tokens
      bestRaw = raw
      bestLevel = level
    }

    if (tokens <= targetTokens) {
      log('reflector', `Compressed to ${tokens} tokens (level ${level})`)
      return { observations: compressed, tokenCount: tokens, raw, compressionLevel: level }
    }
  }

  log('reflector', `Best: ${bestTokens} tokens (level ${bestLevel})`)
  return { observations: bestObservations, tokenCount: bestTokens, raw: bestRaw, compressionLevel: bestLevel }
}

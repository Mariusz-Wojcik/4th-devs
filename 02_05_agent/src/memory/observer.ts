/**
 * Observer — extracts structured observations from conversation history.
 *
 * Based on Mastra's Observational Memory system.
 * https://mastra.ai/blog/observational-memory
 */

import OpenAI from 'openai'
import type { Message, ObserverResult } from '../types.js'
import { isTextMessage, isFunctionCall, isFunctionCallOutput } from '../types.js'
import { estimateTokensRaw } from '../tokens.js'
import { truncate, extractTag } from '../utils.js'
import { log } from '../log.js'
import { OBSERVER_MAX_SECTION_CHARS, OBSERVER_MAX_TOOL_PAYLOAD_CHARS, OBSERVER_MAX_OUTPUT_TOKENS } from '../config.js'
import { OBSERVER_SYSTEM_PROMPT, buildObserverPrompt } from './prompts.js'

export const serializeMessages = (messages: Message[]): string =>
  messages
    .map((msg, i) => {
      if (isFunctionCallOutput(msg)) {
        return `**Tool Result (#${i + 1}):**\n${truncate(msg.output, OBSERVER_MAX_TOOL_PAYLOAD_CHARS)}`
      }

      if (isFunctionCall(msg)) {
        return `**Assistant Tool Call (#${i + 1}):**\n[Tool: ${msg.name}] ${truncate(msg.arguments, OBSERVER_MAX_TOOL_PAYLOAD_CHARS)}`
      }

      if (isTextMessage(msg)) {
        const label = msg.role.charAt(0).toUpperCase() + msg.role.slice(1)
        const content = typeof msg.content === 'string' ? msg.content : ''
        return `**${label} (#${i + 1}):**\n${truncate(content, OBSERVER_MAX_SECTION_CHARS) || '[empty]'}`
      }

      return ''
    })
    .filter(Boolean)
    .join('\n\n---\n\n')

export const parseObserverOutput = (raw: string): ObserverResult => ({
  observations: extractTag(raw, 'observations') ?? raw.trim(),
  currentTask: extractTag(raw, 'current-task'),
  suggestedResponse: extractTag(raw, 'suggested-response'),
  raw,
})

export const runObserver = async (
  openai: OpenAI,
  model: string,
  previousObservations: string,
  messages: Message[],
): Promise<ObserverResult> => {
  const history = serializeMessages(messages)
  if (!history.trim()) {
    return { observations: '', raw: '' }
  }

  log('observer', `Extracting from ${messages.length} messages (~${estimateTokensRaw(history)} tokens)`)

  const response = await openai.responses.create({
    model,
    instructions: OBSERVER_SYSTEM_PROMPT,
    input: buildObserverPrompt(previousObservations, history),
    temperature: 0.3,
    max_output_tokens: OBSERVER_MAX_OUTPUT_TOKENS,
    store: false,
  })

  const raw = response.output_text ?? ''
  const result = parseObserverOutput(raw)
  const lineCount = result.observations.split('\n').filter((l) => l.trim()).length

  log('observer', `Extracted ${lineCount} observation lines (~${estimateTokensRaw(result.observations)} tokens)`)

  return result
}

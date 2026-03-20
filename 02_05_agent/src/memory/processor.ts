/**
 * Memory processor — orchestrates the observer/reflector cycle.
 *
 * Based on Mastra's Observational Memory system.
 * https://mastra.ai/blog/observational-memory
 *
 * Context window layout:
 * ┌──────────────────────────────────────────────────────┐
 * │  Observations (system prompt)  │  Unobserved tail    │
 * │  Compressed history            │  Raw recent messages │
 * └──────────────────────────────────────────────────────┘
 */

import type OpenAI from 'openai'
import { mkdir, writeFile } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import type { Message, Session, CalibrationState, MemoryConfig, ProcessedContext } from '../types.js'
import { isFunctionCallOutput } from '../types.js'
import { estimateTokens, estimateMessagesTokensRaw, estimateMessageTokens } from '../tokens.js'
import { runObserver } from './observer.js'
import { runReflector } from './reflector.js'
import { MEMORY_DIR, resolveModelForProvider, DEFAULT_MEMORY_CONFIG } from '../config.js'
import { log, logError } from '../log.js'
import { CONTINUATION_HINT, buildObservationAppendix } from './prompts.js'

// ============================================================================
// File persistence
// ============================================================================

const pad = (n: number): string => String(n).padStart(3, '0')

const persistMemoryLog = async (
  prefix: string,
  seq: number,
  body: string,
  metadata: Record<string, string | number>,
): Promise<void> => {
  const filename = `${prefix}-${pad(seq)}.md`
  const path = join(MEMORY_DIR, filename)

  const frontmatter = Object.entries(metadata)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')

  const content = `---\n${frontmatter}\ncreated: ${new Date().toISOString()}\n---\n\n${body}\n`

  try {
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, content, 'utf-8')
    log('memory', `💾 ${filename}`)
  } catch (err) {
    logError('memory', `Failed to write ${filename}:`, err)
  }
}

// ============================================================================
// Context shaping helpers
// ============================================================================

/**
 * Split messages into head (to observe) and tail (to keep as raw context).
 * Tail budget is 30% of observation threshold (min 120 tokens).
 * Keeps tool call/result pairs together.
 */
const splitByTailBudget = (
  messages: Message[],
  tailBudget: number,
  calibration?: CalibrationState,
): { head: Message[]; tail: Message[] } => {
  let tailTokens = 0
  let splitIndex = messages.length

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const tokens = estimateMessageTokens(messages[i], calibration)
    if (tailTokens + tokens > tailBudget && splitIndex < messages.length) break
    tailTokens += tokens
    splitIndex = i
  }

  while (splitIndex > 0 && splitIndex < messages.length) {
    if (isFunctionCallOutput(messages[splitIndex])) {
      splitIndex -= 1
    } else {
      break
    }
  }

  return { head: messages.slice(0, splitIndex), tail: messages.slice(splitIndex) }
}

// ============================================================================
// Observer + Reflector execution
// ============================================================================

const runObservation = async (
  openai: OpenAI,
  session: Session,
  config: MemoryConfig,
): Promise<{ contextHint?: { currentTask?: string; suggestedResponse?: string } } | null> => {
  const { messages, memory } = session
  const unobserved = messages.slice(memory.lastObservedIndex)

  const tailBudget = Math.max(120, Math.floor(config.observationThresholdTokens * 0.3))
  const { head } = splitByTailBudget(unobserved, tailBudget, memory.calibration)
  const toObserve = head.length > 0 ? head : unobserved

  const observed = await runObserver(openai, resolveModelForProvider(config.observerModel) as string, memory.activeObservations, toObserve)
  if (!observed.observations) return null

  const prevIndex = memory.lastObservedIndex

  memory.activeObservations = memory.activeObservations
    ? `${memory.activeObservations.trim()}\n\n${observed.observations.trim()}`
    : observed.observations.trim()
  memory.lastObservedIndex = head.length > 0
    ? memory.lastObservedIndex + head.length
    : messages.length
  memory.observationTokenCount = estimateTokens(memory.activeObservations, memory.calibration)

  const sealed = memory.lastObservedIndex - prevIndex
  log('memory', `Sealed ${sealed} messages (indices ${prevIndex}–${memory.lastObservedIndex - 1})`)
  log('memory', `Thread: ${memory.lastObservedIndex} sealed | ${messages.length - memory.lastObservedIndex} active`)

  memory.observerLogSeq += 1
  await persistMemoryLog('observer', memory.observerLogSeq, observed.observations, {
    type: 'observation',
    session: session.id,
    sequence: memory.observerLogSeq,
    generation: memory.generationCount,
    tokens: estimateTokens(observed.observations, memory.calibration),
    messages_observed: toObserve.length,
    sealed_range: `${prevIndex}–${memory.lastObservedIndex - 1}`,
  })

  return { contextHint: { currentTask: observed.currentTask, suggestedResponse: observed.suggestedResponse } }
}

const runReflection = async (
  openai: OpenAI,
  session: Session,
  config: MemoryConfig,
): Promise<void> => {
  const { memory } = session

  log('memory', `Reflecting (${memory.observationTokenCount} > ${config.reflectionThresholdTokens})`)

  const reflected = await runReflector(
    openai,
    resolveModelForProvider(config.reflectorModel) as string,
    memory.activeObservations,
    config.reflectionTargetTokens,
    memory.calibration,
  )

  memory.activeObservations = reflected.observations
  memory.observationTokenCount = reflected.tokenCount
  memory._lastReflectionOutputTokens = reflected.tokenCount
  memory.generationCount += 1

  memory.reflectorLogSeq += 1
  await persistMemoryLog('reflector', memory.reflectorLogSeq, reflected.observations, {
    type: 'reflection',
    session: session.id,
    sequence: memory.reflectorLogSeq,
    generation: memory.generationCount,
    tokens: reflected.tokenCount,
    compression_level: reflected.compressionLevel,
  })
}

// ============================================================================
// Main entry point — called before each provider call in the agent loop
// ============================================================================

/**
 * Core memory processor.
 *
 * 1. Below threshold → pass through (observations in system prompt if they exist)
 * 2. Above threshold → observer seals head, keeps tail
 * 3. Observations too large → reflector compresses
 *
 * Observer runs at most once per HTTP request (flag on session.memory).
 */
export const processMemory = async (
  openai: OpenAI,
  session: Session,
  baseSystemPrompt: string,
  config: MemoryConfig = DEFAULT_MEMORY_CONFIG,
): Promise<ProcessedContext> => {
  const { messages, memory } = session
  const unobserved = messages.slice(memory.lastObservedIndex)
  const pendingTokens = estimateMessagesTokensRaw(unobserved)
  const hasObservations = memory.activeObservations.length > 0

  log('memory', `Pending: ${pendingTokens} tokens (${unobserved.length} msgs) | Observations: ${memory.observationTokenCount} tokens (gen ${memory.generationCount})`)

  const passthroughContext = (): ProcessedContext => ({
    systemPrompt: hasObservations
      ? `${baseSystemPrompt}\n\n${buildObservationAppendix(memory.activeObservations)}`
      : baseSystemPrompt,
    messages: hasObservations ? unobserved : messages,
  })

  if (pendingTokens < config.observationThresholdTokens) {
    return passthroughContext()
  }

  if (memory._observerRanThisRequest) {
    log('memory', `Observer already ran this request, skipping`)
    return passthroughContext()
  }

  // --- Observation ---
  log('memory', `Threshold exceeded (${pendingTokens} >= ${config.observationThresholdTokens}), running observer`)

  try {
    await runObservation(openai, session, config)
    memory._observerRanThisRequest = true
  } catch (err) {
    logError('memory', 'Observer failed:', err)
    return { systemPrompt: baseSystemPrompt, messages }
  }

  // --- Reflection (only if observations grew meaningfully since last reflection) ---
  const grewSinceReflection = memory.observationTokenCount - (memory._lastReflectionOutputTokens ?? 0)
  const shouldReflect = memory.observationTokenCount > config.reflectionThresholdTokens
    && grewSinceReflection >= config.reflectionTargetTokens

  if (shouldReflect) {
    try {
      await runReflection(openai, session, config)
    } catch (err) {
      logError('memory', 'Reflector failed:', err)
    }
  } else if (memory.observationTokenCount > config.reflectionThresholdTokens) {
    log('memory', `Skipping reflection (grew ${grewSinceReflection} tokens since last, need ${config.reflectionTargetTokens})`)
  }

  // --- Return reshaped context ---
  const remaining = messages.slice(memory.lastObservedIndex)
  const finalMessages: Message[] = remaining.length > 0
    ? remaining
    : [{ role: 'user' as const, content: CONTINUATION_HINT }]

  log('memory', `Context: ${finalMessages.length} active msgs + observations (gen ${memory.generationCount}) | ${memory.lastObservedIndex} sealed`)

  return {
    systemPrompt: `${baseSystemPrompt}\n\n${buildObservationAppendix(memory.activeObservations)}`,
    messages: finalMessages,
  }
}

// ============================================================================
// Flush — force-observe remaining messages at end of session/demo
// ============================================================================

export const flushMemory = async (
  openai: OpenAI,
  session: Session,
  config: MemoryConfig = DEFAULT_MEMORY_CONFIG,
): Promise<void> => {
  const { messages, memory } = session
  const unobserved = messages.slice(memory.lastObservedIndex)
  if (unobserved.length === 0) return

  log('flush', `Observing ${unobserved.length} remaining messages`)

  await runObservation(openai, session, config)

  if (memory.observationTokenCount > config.reflectionThresholdTokens) {
    try {
      await runReflection(openai, session, config)
    } catch (err) {
      logError('flush', 'Reflector failed:', err)
    }
  }
}

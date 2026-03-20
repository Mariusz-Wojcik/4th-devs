import type { Message, CalibrationState, UsageTotals } from './types.js'
import { isTextMessage, isFunctionCall, isFunctionCallOutput } from './types.js'
import { TOKEN_CHARS_PER_TOKEN, TOKEN_SAFETY_MARGIN } from './config.js'

/**
 * Raw chars/4 estimate — stable, no calibration applied.
 * Used for threshold checks where predictability matters.
 */
export const estimateTokensRaw = (text: string): number => {
  if (!text) return 0
  return Math.ceil(text.length / TOKEN_CHARS_PER_TOKEN)
}

/**
 * Calibrated estimate — adjusts based on actual API-reported usage.
 * Used for display/budget calculations.
 */
export const estimateTokens = (text: string, cal?: CalibrationState): number => {
  const base = estimateTokensRaw(text)
  if (!base) return 0

  if (cal && cal.cumulativeActual > 500 && cal.cumulativeEstimated > 0) {
    const ratio = cal.cumulativeActual / cal.cumulativeEstimated
    return Math.ceil(base * ratio)
  }

  return base
}

export const withSafetyMargin = (tokens: number): number =>
  Math.ceil(tokens * TOKEN_SAFETY_MARGIN)

export const estimateMessageTokens = (message: Message, cal?: CalibrationState): number => {
  let tokens = 4

  if (isTextMessage(message)) {
    if (typeof message.content === 'string') {
      tokens += estimateTokens(message.content, cal)
    }
    return tokens
  }

  if (isFunctionCall(message)) {
    tokens += estimateTokens(message.name, cal)
    tokens += estimateTokens(message.arguments, cal)
    tokens += 10
    return tokens
  }

  if (isFunctionCallOutput(message)) {
    tokens += estimateTokens(message.output, cal)
    return tokens
  }

  return tokens
}

export const estimateMessagesTokens = (messages: Message[], cal?: CalibrationState): { raw: number; safe: number } => {
  let raw = 0
  for (const msg of messages) {
    raw += estimateMessageTokens(msg, cal)
  }
  return { raw, safe: withSafetyMargin(raw) }
}

/**
 * Raw (uncalibrated) message token estimate for stable threshold comparisons.
 */
export const estimateMessagesTokensRaw = (messages: Message[]): number => {
  let total = 0
  for (const msg of messages) {
    total += estimateMessageTokens(msg)
  }
  return total
}

export const recordActualUsage = (cal: CalibrationState, estimated: number, actual: number): void => {
  cal.cumulativeEstimated += estimated
  cal.cumulativeActual += actual
}

export const trackUsage = (
  usage: { input_tokens: number; output_tokens: number } | null,
  cal: CalibrationState,
  estimatedSafe: number,
  totals: UsageTotals,
): number | null => {
  totals.estimated += estimatedSafe
  if (!usage) return null

  const actual = usage.input_tokens + usage.output_tokens
  totals.actual += actual
  recordActualUsage(cal, estimatedSafe, actual)
  return actual
}

export const getCalibration = (cal: CalibrationState): { ratio: number | null; samples: number } => {
  if (cal.cumulativeActual < 100 || cal.cumulativeEstimated === 0) {
    return { ratio: null, samples: cal.cumulativeActual }
  }
  return { ratio: cal.cumulativeActual / cal.cumulativeEstimated, samples: cal.cumulativeActual }
}

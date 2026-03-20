export const truncate = (s: string, max = 100): string =>
  s.length > max ? s.slice(0, max - 1) + '…' : s

export const extractTag = (text: string, tag: string): string | undefined => {
  const match = text.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, 'i'))
  return match?.[1]?.trim() || undefined
}

export const parseArgs = (raw: string): Record<string, unknown> => {
  const parsed: unknown = JSON.parse(raw || '{}')
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`Expected JSON object, got ${Array.isArray(parsed) ? 'array' : typeof parsed}`)
  }
  return parsed as Record<string, unknown>
}

export const getMessageText = (message: { content: Array<{ type: string; text?: string }> }): string => {
  let text = ''
  for (const part of message.content) {
    if (part.type === 'output_text' && part.text) text += part.text
  }
  return text
}

export const formatError = (err: unknown): string =>
  err instanceof Error ? err.message : String(err)

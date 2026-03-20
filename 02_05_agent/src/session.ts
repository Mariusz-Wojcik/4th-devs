import { freshMemory } from './types.js'
import type { Session } from './types.js'

const sessions = new Map<string, Session>()

export const getSession = (sessionId: string): Session | undefined =>
  sessions.get(sessionId)

export const getOrCreateSession = (sessionId: string): Session => {
  const existing = sessions.get(sessionId)
  if (existing) return existing

  const session: Session = { id: sessionId, messages: [], memory: freshMemory() }
  sessions.set(sessionId, session)
  return session
}

export const listSessions = (): Session[] =>
  [...sessions.values()]

export const buildMemorySummary = (session: Session) => ({
  observationTokens: session.memory.observationTokenCount,
  generation: session.memory.generationCount,
  totalMessages: session.messages.length,
  sealedMessages: session.memory.lastObservedIndex,
  activeMessages: session.messages.length - session.memory.lastObservedIndex,
})

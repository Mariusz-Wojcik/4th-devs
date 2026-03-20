import { Hono } from 'hono'
import { serve } from '@hono/node-server'
import { cors } from 'hono/cors'
import { randomUUID } from 'node:crypto'
import { runAgent } from './agent.js'
import { flushMemory } from './memory/processor.js'
import { openai, SERVER_PORT } from './config.js'
import { truncate } from './utils.js'
import { log, logError } from './log.js'
import { getSession, getOrCreateSession, listSessions, buildMemorySummary } from './session.js'

const app = new Hono()
app.use(cors())

app.post('/api/chat', async (c) => {
  const body = await c.req.json()
  const sessionId = typeof body.session_id === 'string' ? body.session_id : randomUUID()
  const message = typeof body.message === 'string' ? body.message : null

  if (!message) {
    return c.json({ error: 'message is required' }, 400)
  }

  const session = getOrCreateSession(sessionId)
  log('session', `${sessionId.slice(0, 8)} "${truncate(message, 60)}"`)

  try {
    const result = await runAgent(session, message)

    return c.json({
      session_id: sessionId,
      response: result.response,
      memory: {
        hasObservations: session.memory.activeObservations.length > 0,
        ...buildMemorySummary(session),
      },
      usage: result.usage,
    })
  } catch (err) {
    logError('session', 'Agent execution failed:', err)
    return c.json({ error: 'Agent execution failed' }, 500)
  }
})

app.get('/api/sessions', (c) => {
  const list = listSessions().map((s) => ({
    id: s.id,
    messageCount: s.messages.length,
    observationTokens: s.memory.observationTokenCount,
    generation: s.memory.generationCount,
  }))
  return c.json(list)
})

app.get('/api/sessions/:id/memory', (c) => {
  const session = getSession(c.req.param('id'))
  if (!session) return c.json({ error: 'Session not found' }, 404)

  return c.json({
    session_id: session.id,
    messageCount: session.messages.length,
    memory: session.memory,
  })
})

app.post('/api/sessions/:id/flush', async (c) => {
  const session = getSession(c.req.param('id'))
  if (!session) return c.json({ error: 'Session not found' }, 404)

  log('session', `${c.req.param('id').slice(0, 8)} Flushing remaining messages to observations`)

  try {
    await flushMemory(openai, session)
    return c.json({ session_id: session.id, memory: buildMemorySummary(session) })
  } catch (err) {
    logError('session', 'Flush failed:', err)
    return c.json({ error: 'Flush failed' }, 500)
  }
})

const port = parseInt(process.env.PORT ?? String(SERVER_PORT), 10)

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`\n========================================`)
  console.log(`  02_05 Agent — Context Engineering Demo`)
  console.log(`  http://localhost:${info.port}`)
  console.log(`========================================\n`)
})

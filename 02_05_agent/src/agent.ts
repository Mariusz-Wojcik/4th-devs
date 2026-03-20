import type OpenAI from 'openai'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import matter from 'gray-matter'
import type { AgentResult, AgentTemplate, FunctionCallItem, ResponseOutputItem, Session } from './types.js'
import { findTool, resolveAgentTools } from './tools.js'
import { processMemory } from './memory/processor.js'
import { estimateMessagesTokens, trackUsage, getCalibration } from './tokens.js'
import { openai, resolveModelForProvider, WORKSPACE, AGENT_MAX_TURNS, DEFAULT_MEMORY_CONFIG } from './config.js'
import { truncate, parseArgs, getMessageText, formatError } from './utils.js'
import { log, logError } from './log.js'

const loadAgent = async (name: string): Promise<AgentTemplate> => {
  const raw = await readFile(join(WORKSPACE, 'agents', `${name}.agent.md`), 'utf-8')
  const { data, content } = matter(raw)
  return {
    name: data.name ?? name,
    model: typeof data.model === 'string' ? data.model : 'gpt-4.1-mini',
    tools: Array.isArray(data.tools) ? data.tools : [],
    systemPrompt: content.trim(),
  }
}

const applyResponseOutput = (session: Session, output: ResponseOutputItem[]): FunctionCallItem[] => {
  const pendingCalls: FunctionCallItem[] = []

  for (const item of output) {
    if (item.type === 'message') {
      const text = getMessageText(item)
      if (text) session.messages.push({ role: 'assistant', content: text })
      continue
    }

    if (item.type === 'function_call') {
      const call: FunctionCallItem = { type: 'function_call', call_id: item.call_id, name: item.name, arguments: item.arguments }
      session.messages.push(call)
      pendingCalls.push(call)
    }
  }

  return pendingCalls
}

const executeToolCall = async (session: Session, call: FunctionCallItem): Promise<void> => {
  let args: Record<string, unknown>
  try {
    args = parseArgs(call.arguments)
  } catch (err) {
    logError('agent', `Tool: ${call.name} — bad arguments:`, err)
    session.messages.push({ type: 'function_call_output', call_id: call.call_id, output: `Error parsing arguments: ${formatError(err)}` })
    return
  }

  log('agent', `Tool: ${call.name}(${truncate(JSON.stringify(args))})`)

  const tool = findTool(call.name)
  const output = tool ? await tool.handler(args) : `Unknown tool: ${call.name}`
  session.messages.push({ type: 'function_call_output', call_id: call.call_id, output })
}

export const runAgent = async (session: Session, userMessage: string): Promise<AgentResult> => {
  const template = await loadAgent('alice')
  const model = resolveModelForProvider(template.model) as string
  const responsesTools = resolveAgentTools(template.tools)
  const cal = session.memory.calibration

  session.messages.push({ role: 'user', content: userMessage })
  session.memory._observerRanThisRequest = false

  const totals = { estimated: 0, actual: 0 }

  const buildUsage = (turns: number): AgentResult['usage'] => ({
    totalEstimatedTokens: totals.estimated,
    totalActualTokens: totals.actual,
    calibration: getCalibration(cal),
    turns,
  })

  for (let turn = 0; turn < AGENT_MAX_TURNS; turn += 1) {
    const context = await processMemory(openai, session, template.systemPrompt, DEFAULT_MEMORY_CONFIG)
    const estimated = estimateMessagesTokens(context.messages, cal)
    log('agent', `Turn ${turn + 1}, ${context.messages.length} items (~${estimated.safe} tokens)`)

    const response = await openai.responses.create({
      model,
      instructions: context.systemPrompt,
      input: context.messages as OpenAI.Responses.ResponseInputItem[],
      tools: responsesTools.length > 0 ? responsesTools : undefined,
      store: false,
    })

    const actualTokens = trackUsage(response.usage ?? null, cal, estimated.safe, totals)
    if (actualTokens !== null) {
      log('agent', `API usage — estimated: ${estimated.safe}, actual: ${actualTokens}`)
    }

    const pendingCalls = applyResponseOutput(session, response.output as ResponseOutputItem[])

    if (pendingCalls.length === 0) {
      log('agent', `Done (${turn + 1} turns)`)
      return { response: response.output_text ?? '', usage: buildUsage(turn + 1) }
    }

    for (const call of pendingCalls) {
      await executeToolCall(session, call)
    }
  }

  return { response: 'Exceeded maximum turns', usage: buildUsage(AGENT_MAX_TURNS) }
}

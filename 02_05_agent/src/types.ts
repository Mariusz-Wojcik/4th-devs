export type TextMessage = {
  role: 'user' | 'assistant' | 'system' | 'developer'
  content: string | null
}

export type FunctionCallItem = {
  type: 'function_call'
  call_id: string
  name: string
  arguments: string
}

export type FunctionCallOutputItem = {
  type: 'function_call_output'
  call_id: string
  output: string
}

export type Message = TextMessage | FunctionCallItem | FunctionCallOutputItem

export const isTextMessage = (m: Message): m is TextMessage => 'role' in m && !('type' in m)
export const isFunctionCall = (m: Message): m is FunctionCallItem => 'type' in m && m.type === 'function_call'
export const isFunctionCallOutput = (m: Message): m is FunctionCallOutputItem => 'type' in m && m.type === 'function_call_output'

export interface AgentTemplate {
  name: string
  model: string
  tools: string[]
  systemPrompt: string
}

export interface CalibrationState {
  cumulativeEstimated: number
  cumulativeActual: number
}

export interface MemoryState {
  activeObservations: string
  lastObservedIndex: number
  observationTokenCount: number
  generationCount: number
  observerLogSeq: number
  reflectorLogSeq: number
  calibration: CalibrationState
  _observerRanThisRequest?: boolean
  _lastReflectionOutputTokens?: number
}

export const freshMemory = (): MemoryState => ({
  activeObservations: '',
  lastObservedIndex: 0,
  observationTokenCount: 0,
  generationCount: 0,
  observerLogSeq: 0,
  reflectorLogSeq: 0,
  calibration: { cumulativeEstimated: 0, cumulativeActual: 0 },
})

export interface Session {
  id: string
  messages: Message[]
  memory: MemoryState
}

export interface ToolDefinition {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface Tool {
  definition: ToolDefinition
  handler: (args: Record<string, unknown>) => Promise<string>
}

export type ResponseOutputItem =
  | { type: 'message'; content: Array<{ type: string; text?: string }> }
  | { type: 'function_call'; call_id: string; name: string; arguments: string }

export interface AgentResult {
  response: string
  usage: {
    totalEstimatedTokens: number
    totalActualTokens: number
    calibration: { ratio: number | null; samples: number }
    turns: number
  }
}

export interface ResolvedTool {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
  strict: boolean
}

export interface MemoryConfig {
  observationThresholdTokens: number
  reflectionThresholdTokens: number
  reflectionTargetTokens: number
  observerModel: string
  reflectorModel: string
}

export interface UsageTotals {
  estimated: number
  actual: number
}

export interface ObserverResult {
  observations: string
  currentTask?: string
  suggestedResponse?: string
  raw: string
}

export interface ReflectorResult {
  observations: string
  tokenCount: number
  raw: string
  compressionLevel: number
}

export interface ProcessedContext {
  systemPrompt: string
  messages: Message[]
}

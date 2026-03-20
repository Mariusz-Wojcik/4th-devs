import { formatError } from './utils.js'

type Tag = 'agent' | 'memory' | 'observer' | 'reflector' | 'flush' | 'session'

const prefix = (tag: Tag): string => `  [${tag}]`

export const log = (tag: Tag, message: string): void => {
  console.log(`${prefix(tag)} ${message}`)
}

export const logError = (tag: Tag, message: string, err?: unknown): void => {
  if (err !== undefined) {
    console.error(`${prefix(tag)} ${message}`, formatError(err))
  } else {
    console.error(`${prefix(tag)} ${message}`)
  }
}

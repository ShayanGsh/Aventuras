import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import type { CodexToolCall, CodexToolExecutor, CodexToolResult } from './codex'

export interface CodexDirectAccount {
  authMode: string
  email: string | null
  planType: string | null
}

export interface CodexDirectAccountState {
  account: CodexDirectAccount | null
  requiresOpenaiAuth: boolean
}

export interface CodexDirectLoginStart {
  loginType: string
  loginId: string
  authUrl: string
  verificationUrl: string
  userCode: string
}

export interface CodexDirectModel {
  id: string
  reasoning: boolean
}

export interface CodexDirectTurnHandle {
  threadId: string
  turnId: string
}

export interface CodexDirectTurnDelta {
  threadId: string
  turnId: string
  content: string
  reasoning: string | null
  toolCall?: CodexToolCall
  toolResult?: CodexToolResult
}

export interface CodexDirectTurnCompleted {
  threadId: string
  turnId: string
  status: string
  error: string | null
}

export interface CodexDirectTurnRequest {
  model: string
  system: string
  prompt: string
  input?: unknown[]
  reasoningEffort: string
  outputSchema?: unknown
  tools?: unknown[]
  toolChoice?: unknown
  toolExecutor?: CodexToolExecutor
  signal?: AbortSignal
}

type CodexDirectTurnStreamEvent =
  | { type: 'delta'; payload: CodexDirectTurnDelta }
  | { type: 'completed'; payload: CodexDirectTurnCompleted }

function createAbortError(): Error {
  const error = new Error('Codex direct turn interrupted')
  error.name = 'AbortError'
  return error
}

class CodexDirectService {
  async readAccount(): Promise<CodexDirectAccountState> {
    return invoke('codex_direct_account_read')
  }

  async startLogin(): Promise<CodexDirectLoginStart> {
    return invoke('codex_direct_login_start')
  }

  async logout(): Promise<void> {
    return invoke('codex_direct_logout')
  }

  async listModels(): Promise<CodexDirectModel[]> {
    return invoke('codex_direct_list_models')
  }

  async disconnect(): Promise<void> {
    return invoke('codex_direct_disconnect')
  }

  async startTurn(request: CodexDirectTurnRequest): Promise<CodexDirectTurnHandle> {
    return invoke('codex_direct_turn_start', {
      model: request.model,
      system: request.system,
      prompt: request.prompt,
      input: request.input ?? null,
      reasoningEffort: request.reasoningEffort,
      outputSchema: request.outputSchema ?? null,
      tools: request.tools ?? null,
      toolChoice: request.toolChoice ?? null,
    })
  }

  async interruptTurn(handle: CodexDirectTurnHandle): Promise<void> {
    return invoke('codex_direct_turn_interrupt', { turnId: handle.turnId })
  }

  async generateText(request: CodexDirectTurnRequest): Promise<string> {
    let content = ''
    for await (const delta of this.streamTurn(request)) {
      content += delta.content
    }
    return content
  }

  async *streamTurn(request: CodexDirectTurnRequest): AsyncIterable<CodexDirectTurnDelta> {
    if (request.signal?.aborted) throw createAbortError()

    const buffered: CodexDirectTurnStreamEvent[] = []
    const queued: CodexDirectTurnStreamEvent[] = []
    let handle: CodexDirectTurnHandle | null = null
    let closed = false
    let wake: ((event: CodexDirectTurnStreamEvent | null) => void) | null = null

    const push = (event: CodexDirectTurnStreamEvent) => {
      if (closed) return
      if (wake) {
        const resolve = wake
        wake = null
        resolve(event)
      } else {
        queued.push(event)
      }
    }

    const accept = (event: CodexDirectTurnStreamEvent) => {
      if (!handle) {
        buffered.push(event)
        return
      }
      const payload = event.payload
      if (payload.threadId === handle.threadId && payload.turnId === handle.turnId) {
        push(event)
      }
    }

    const take = async (): Promise<CodexDirectTurnStreamEvent | null> => {
      if (queued.length > 0) return queued.shift() ?? null
      if (closed) return null
      return new Promise((resolve) => {
        wake = resolve
      })
    }

    let unlistenDelta: (() => void) | undefined
    let unlistenCompleted: (() => void) | undefined
    let abortHandler: (() => void) | undefined

    try {
      unlistenDelta = await listen<CodexDirectTurnDelta>('codex-direct-turn-delta', (event) => {
        accept({ type: 'delta', payload: event.payload })
      })
      unlistenCompleted = await listen<CodexDirectTurnCompleted>(
        'codex-direct-turn-completed',
        (event) => accept({ type: 'completed', payload: event.payload }),
      )

      handle = await this.startTurn(request)
      for (const event of buffered.splice(0)) accept(event)

      if (request.signal) {
        abortHandler = () => {
          if (handle) void this.interruptTurn(handle).catch(() => undefined)
          push({
            type: 'completed',
            payload: {
              threadId: handle?.threadId ?? '',
              turnId: handle?.turnId ?? '',
              status: 'interrupted',
              error: null,
            },
          })
        }
        request.signal.addEventListener('abort', abortHandler, { once: true })
        if (request.signal.aborted) abortHandler()
      }

      while (true) {
        const event = await take()
        if (!event) return
        if (event.type === 'delta') {
          yield event.payload
          continue
        }

        if (event.payload.status === 'completed') return
        if (event.payload.status === 'interrupted') throw createAbortError()
        throw new Error(event.payload.error || `Codex direct turn ${event.payload.status}`)
      }
    } finally {
      closed = true
      if (request.signal && abortHandler) {
        request.signal.removeEventListener('abort', abortHandler)
      }
      unlistenDelta?.()
      unlistenCompleted?.()
      if (wake) {
        const resolve = wake
        wake = null
        resolve(null)
      }
    }
  }
}

export const codexDirectService = new CodexDirectService()

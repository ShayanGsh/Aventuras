import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import type { JSONArray } from '@ai-sdk/provider'

export interface CodexToolCall {
  id: string
  name: string
  input: unknown
  providerExecuted?: boolean
  dynamic?: boolean
}

export interface CodexReasoningItem {
  id: string
  encryptedContent: string
  summary: JSONArray
}

export interface CodexAccount {
  authMode: string
  email: string | null
  planType: string | null
}

export interface CodexAccountState {
  account: CodexAccount | null
  requiresOpenaiAuth: boolean
}

export interface CodexLoginStart {
  loginType: string
  loginId: string
  authUrl: string
  verificationUrl: string
  userCode: string
}

export interface CodexModel {
  id: string
  reasoning: boolean
}

export interface CodexTurnHandle {
  threadId: string
  turnId: string
}

export interface CodexTurnDelta {
  threadId: string
  turnId: string
  content: string
  reasoning: string | null
  toolCall?: CodexToolCall
  reasoningItem?: CodexReasoningItem
}

export interface CodexTurnCompleted {
  threadId: string
  turnId: string
  status: string
  error: string | null
}

export interface CodexTurnRequest {
  model: string
  system: string
  prompt: string
  input?: unknown[]
  reasoningEffort: string
  outputSchema?: unknown
  tools?: unknown[]
  toolChoice?: unknown
  signal?: AbortSignal
}

type CodexTurnStreamEvent =
  { type: 'delta'; payload: CodexTurnDelta } | { type: 'completed'; payload: CodexTurnCompleted }

function createAbortError(): Error {
  const error = new Error('Codex turn interrupted')
  error.name = 'AbortError'
  return error
}

class CodexService {
  async readAccount(): Promise<CodexAccountState> {
    return invoke('codex_account_read')
  }

  async startLogin(): Promise<CodexLoginStart> {
    return invoke('codex_login_start')
  }

  async cancelLogin(loginId: string): Promise<void> {
    return invoke('codex_login_cancel', { loginId })
  }

  async logout(): Promise<void> {
    return invoke('codex_logout')
  }

  async listModels(): Promise<CodexModel[]> {
    return invoke('codex_list_models')
  }

  async startTurn(request: CodexTurnRequest): Promise<CodexTurnHandle> {
    return invoke('codex_turn_start', {
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

  async interruptTurn(handle: CodexTurnHandle): Promise<void> {
    return invoke('codex_turn_interrupt', { turnId: handle.turnId })
  }

  async generateText(request: CodexTurnRequest): Promise<string> {
    let content = ''
    for await (const delta of this.streamTurn(request)) {
      content += delta.content
    }
    return content
  }

  async *streamTurn(request: CodexTurnRequest): AsyncIterable<CodexTurnDelta> {
    if (request.signal?.aborted) throw createAbortError()

    const buffered: CodexTurnStreamEvent[] = []
    const queued: CodexTurnStreamEvent[] = []
    let handle: CodexTurnHandle | null = null
    let closed = false
    let wake: ((event: CodexTurnStreamEvent | null) => void) | null = null

    const push = (event: CodexTurnStreamEvent) => {
      if (closed) return
      if (wake) {
        const resolve = wake
        wake = null
        resolve(event)
      } else {
        queued.push(event)
      }
    }

    const accept = (event: CodexTurnStreamEvent) => {
      if (!handle) {
        buffered.push(event)
        return
      }
      const payload = event.payload
      if (payload.threadId === handle.threadId && payload.turnId === handle.turnId) {
        push(event)
      }
    }

    const take = async (): Promise<CodexTurnStreamEvent | null> => {
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
      unlistenDelta = await listen<CodexTurnDelta>('codex-turn-delta', (event) => {
        accept({ type: 'delta', payload: event.payload })
      })
      unlistenCompleted = await listen<CodexTurnCompleted>('codex-turn-completed', (event) =>
        accept({ type: 'completed', payload: event.payload }),
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
        throw new Error(event.payload.error || `Codex turn ${event.payload.status}`)
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

export const codexService = new CodexService()

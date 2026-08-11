import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'

export interface CodexHermesAccount {
  authMode: string
  email: string | null
  planType: string | null
}

export interface CodexHermesAccountState {
  account: CodexHermesAccount | null
  requiresOpenaiAuth: boolean
}

export interface CodexHermesLoginStart {
  loginType: string
  loginId: string
  authUrl: string
  verificationUrl: string
  userCode: string
}

export interface CodexHermesModel {
  id: string
  reasoning: boolean
}

export interface CodexHermesTurnHandle {
  threadId: string
  turnId: string
}

export interface CodexHermesTurnDelta {
  threadId: string
  turnId: string
  content: string
  reasoning: string | null
}

export interface CodexHermesTurnCompleted {
  threadId: string
  turnId: string
  status: string
  error: string | null
}

export interface CodexHermesTurnRequest {
  model: string
  system: string
  prompt: string
  reasoningEffort: string
  outputSchema?: unknown
  signal?: AbortSignal
}

type CodexHermesTurnStreamEvent =
  | { type: 'delta'; payload: CodexHermesTurnDelta }
  | { type: 'completed'; payload: CodexHermesTurnCompleted }

function createAbortError(): Error {
  const error = new Error('Codex-Hermes turn interrupted')
  error.name = 'AbortError'
  return error
}

class CodexHermesService {
  async readAccount(): Promise<CodexHermesAccountState> {
    return invoke('codex_hermes_account_read')
  }

  async startLogin(): Promise<CodexHermesLoginStart> {
    return invoke('codex_hermes_login_start')
  }

  async logout(): Promise<void> {
    return invoke('codex_hermes_logout')
  }

  async listModels(): Promise<CodexHermesModel[]> {
    return invoke('codex_hermes_list_models')
  }

  async disconnect(): Promise<void> {
    return invoke('codex_hermes_disconnect')
  }

  async startTurn(request: CodexHermesTurnRequest): Promise<CodexHermesTurnHandle> {
    return invoke('codex_hermes_turn_start', {
      model: request.model,
      system: request.system,
      prompt: request.prompt,
      reasoningEffort: request.reasoningEffort,
      outputSchema: request.outputSchema ?? null,
    })
  }

  async interruptTurn(handle: CodexHermesTurnHandle): Promise<void> {
    return invoke('codex_hermes_turn_interrupt', { turnId: handle.turnId })
  }

  async generateText(request: CodexHermesTurnRequest): Promise<string> {
    let content = ''
    for await (const delta of this.streamTurn(request)) {
      content += delta.content
    }
    return content
  }

  async *streamTurn(request: CodexHermesTurnRequest): AsyncIterable<CodexHermesTurnDelta> {
    if (request.signal?.aborted) throw createAbortError()

    const buffered: CodexHermesTurnStreamEvent[] = []
    const queued: CodexHermesTurnStreamEvent[] = []
    let handle: CodexHermesTurnHandle | null = null
    let closed = false
    let wake: ((event: CodexHermesTurnStreamEvent | null) => void) | null = null

    const push = (event: CodexHermesTurnStreamEvent) => {
      if (closed) return
      if (wake) {
        const resolve = wake
        wake = null
        resolve(event)
      } else {
        queued.push(event)
      }
    }

    const accept = (event: CodexHermesTurnStreamEvent) => {
      if (!handle) {
        buffered.push(event)
        return
      }
      const payload = event.payload
      if (payload.threadId === handle.threadId && payload.turnId === handle.turnId) {
        push(event)
      }
    }

    const take = async (): Promise<CodexHermesTurnStreamEvent | null> => {
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
      unlistenDelta = await listen<CodexHermesTurnDelta>('codex-hermes-turn-delta', (event) => {
        accept({ type: 'delta', payload: event.payload })
      })
      unlistenCompleted = await listen<CodexHermesTurnCompleted>(
        'codex-hermes-turn-completed',
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
        throw new Error(event.payload.error || `Codex-Hermes turn ${event.payload.status}`)
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

export const codexHermesService = new CodexHermesService()

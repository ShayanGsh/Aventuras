import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Prompt,
  LanguageModelV4StreamPart,
  LanguageModelV4Usage,
} from '@ai-sdk/provider'
import type { ProviderType } from '$lib/types'
import { codexService, type CodexTurnRequest } from '$lib/services/codex'
import { codexDirectService } from '$lib/services/codexDirect'

interface CodexTransport {
  generateText(request: CodexTurnRequest): Promise<string>
  streamTurn(
    request: CodexTurnRequest,
  ): AsyncIterable<{ content: string; reasoning: string | null }>
}

const EMPTY_USAGE: LanguageModelV4Usage = {
  inputTokens: {
    total: undefined,
    noCache: undefined,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: {
    total: undefined,
    text: undefined,
    reasoning: undefined,
  },
}

function contentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''

  return content
    .map((part) => {
      if (!part || typeof part !== 'object') return ''
      const value = part as { type?: string; text?: unknown; input?: unknown; result?: unknown }
      if (value.type === 'text' || value.type === 'reasoning') {
        return typeof value.text === 'string' ? value.text : ''
      }
      if (value.type === 'tool-result') {
        return typeof value.result === 'string' ? value.result : JSON.stringify(value.result)
      }
      if (value.type === 'tool-call') {
        return `[tool call: ${String(value.input ?? '')}]`
      }
      return ''
    })
    .filter(Boolean)
    .join('')
}

function buildPrompt(prompt: LanguageModelV4Prompt): { system: string; prompt: string } {
  const system = prompt
    .filter((message) => message.role === 'system')
    .map((message) => message.content)
    .join('\n\n')

  const conversation = prompt
    .filter((message) => message.role !== 'system')
    .map((message) => {
      const text = contentText(message.content)
      if (prompt.filter((item) => item.role !== 'system').length === 1) return text
      return `${message.role}:\n${text}`
    })
    .filter(Boolean)
    .join('\n\n')

  return { system, prompt: conversation }
}

function reasoningEffort(options: LanguageModelV4CallOptions): string {
  const effort = options.reasoning
  return effort && effort !== 'provider-default' ? effort : 'medium'
}

function outputSchema(options: LanguageModelV4CallOptions): unknown {
  return options.responseFormat?.type === 'json' ? options.responseFormat.schema : undefined
}

function unsupportedToolsError(): Error {
  return new Error(
    'OpenAI Codex profiles do not support Vercel AI SDK tool-loop calls. Assign a tool-capable provider to this service.',
  )
}

function getTransport(providerType: ProviderType): CodexTransport {
  return providerType === 'openai-codex' ? codexService : codexDirectService
}

export function createCodexLanguageModel(
  providerType: 'openai-codex' | 'openai-codex-direct',
  modelId: string,
): LanguageModelV4 {
  const transport = getTransport(providerType)

  return {
    specificationVersion: 'v4',
    provider: providerType,
    modelId,
    supportedUrls: {},
    doGenerate: async (options) => {
      if (options.tools?.length) throw unsupportedToolsError()

      const { system, prompt } = buildPrompt(options.prompt)
      const text = await transport.generateText({
        model: modelId,
        system,
        prompt,
        reasoningEffort: reasoningEffort(options),
        outputSchema: outputSchema(options),
        signal: options.abortSignal,
      })

      return {
        content: [{ type: 'text', text }],
        finishReason: { unified: 'stop', raw: 'stop' },
        usage: EMPTY_USAGE,
        warnings: [],
      }
    },
    doStream: async (options) => {
      if (options.tools?.length) throw unsupportedToolsError()

      const { system, prompt } = buildPrompt(options.prompt)
      const request: CodexTurnRequest = {
        model: modelId,
        system,
        prompt,
        reasoningEffort: reasoningEffort(options),
        outputSchema: outputSchema(options),
        signal: options.abortSignal,
      }

      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        async start(controller) {
          const textId = 'codex-text'
          const reasoningId = 'codex-reasoning'
          let textStarted = false
          let reasoningStarted = false

          controller.enqueue({ type: 'stream-start', warnings: [] })

          try {
            for await (const delta of transport.streamTurn(request)) {
              if (delta.reasoning) {
                if (!reasoningStarted) {
                  reasoningStarted = true
                  controller.enqueue({ type: 'reasoning-start', id: reasoningId })
                }
                controller.enqueue({
                  type: 'reasoning-delta',
                  id: reasoningId,
                  delta: delta.reasoning,
                })
              }

              if (delta.content) {
                if (!textStarted) {
                  textStarted = true
                  controller.enqueue({ type: 'text-start', id: textId })
                }
                controller.enqueue({ type: 'text-delta', id: textId, delta: delta.content })
              }
            }

            if (reasoningStarted) controller.enqueue({ type: 'reasoning-end', id: reasoningId })
            if (textStarted) controller.enqueue({ type: 'text-end', id: textId })
            controller.enqueue({
              type: 'finish',
              finishReason: { unified: 'stop', raw: 'stop' },
              usage: EMPTY_USAGE,
            })
            controller.close()
          } catch (error) {
            controller.enqueue({ type: 'error', error })
            controller.close()
          }
        },
      })

      return { stream }
    },
  }
}

import type {
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Prompt,
  LanguageModelV4StreamPart,
  LanguageModelV4Usage,
} from '@ai-sdk/provider'
import {
  codexService,
  type CodexToolCall,
  type CodexToolExecutor,
  type CodexReasoningItem,
  type CodexToolResult,
  type CodexTurnRequest,
} from '$lib/services/codex'
import type {
  JSONObject,
  SharedV4ProviderMetadata,
  SharedV4ProviderOptions,
} from '@ai-sdk/provider'

interface CodexTransport {
  streamTurn(request: CodexTurnRequest): AsyncIterable<{
    content: string
    reasoning: string | null
    toolCall?: CodexToolCall
    toolResult?: CodexToolResult
    reasoningItem?: CodexReasoningItem
  }>
}

const CODEX_PROVIDER_METADATA_KEY = 'openaiCodex'

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

function jsonString(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value) ?? '{}'
  } catch {
    return '{}'
  }
}

function toolResultText(output: unknown): string {
  if (!output || typeof output !== 'object') return jsonString(output)

  const value = output as { type?: string; value?: unknown }
  if (value.type === 'text' || value.type === 'error-text') {
    return typeof value.value === 'string' ? value.value : jsonString(value.value)
  }
  if (value.type === 'json' || value.type === 'error-json') {
    return jsonString(value.value)
  }
  if (value.type === 'content') return jsonString(value.value)
  return jsonString(output)
}

function readReasoningItems(part: {
  providerOptions?: SharedV4ProviderOptions
}): CodexReasoningItem[] {
  const providerOptions = part.providerOptions?.[CODEX_PROVIDER_METADATA_KEY]
  const reasoningItems = providerOptions?.reasoningItems
  if (!Array.isArray(reasoningItems)) return []

  return reasoningItems.flatMap((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const value = item as { id?: unknown; encryptedContent?: unknown }
    if (typeof value.id !== 'string' || typeof value.encryptedContent !== 'string') return []
    return [{ id: value.id, encryptedContent: value.encryptedContent }]
  })
}

export function buildPrompt(prompt: LanguageModelV4Prompt): {
  system: string
  prompt: string
  input: unknown[]
} {
  const system = prompt
    .filter((message) => message.role === 'system')
    .map((message) => message.content)
    .join('\n\n')

  const conversationMessages = prompt.filter((message) => message.role !== 'system')
  const conversation = conversationMessages
    .map((message) => {
      const text = contentText(message.content)
      if (conversationMessages.length === 1) return text
      return `${message.role}:\n${text}`
    })
    .filter(Boolean)
    .join('\n\n')

  const input: unknown[] = []
  for (const message of conversationMessages) {
    if (message.role === 'user') {
      const text = contentText(message.content)
      if (text) {
        input.push({
          role: 'user',
          content: [{ type: 'input_text', text }],
        })
      }
      continue
    }

    if (message.role === 'assistant') {
      let textParts: string[] = []
      const flushText = () => {
        const text = textParts.join('')
        textParts = []
        if (text) {
          input.push({
            role: 'assistant',
            content: [{ type: 'output_text', text }],
          })
        }
      }

      for (const part of message.content) {
        if (part.type === 'text') {
          textParts.push(part.text)
        } else if (part.type === 'reasoning') {
          flushText()
          for (const item of readReasoningItems(part)) {
            input.push({
              type: 'reasoning',
              id: item.id,
              encrypted_content: item.encryptedContent,
            })
          }
        } else if (part.type === 'tool-call') {
          flushText()
          input.push({
            type: 'function_call',
            call_id: part.toolCallId,
            name: part.toolName,
            arguments: jsonString(part.input),
          })
        }
      }
      flushText()
      continue
    }

    for (const part of message.content) {
      if (part.type !== 'tool-result') continue
      input.push({
        type: 'function_call_output',
        call_id: part.toolCallId,
        output: toolResultText(part.output),
      })
    }
  }

  return { system, prompt: conversation, input }
}

function reasoningEffort(options: LanguageModelV4CallOptions): string {
  const effort = options.reasoning
  return effort && effort !== 'provider-default' ? effort : 'medium'
}

function reasoningMetadata(items: CodexReasoningItem[]): SharedV4ProviderMetadata | undefined {
  if (items.length === 0) return undefined
  const metadata: JSONObject = {
    reasoningItems: items.map((item): JSONObject => ({
      id: item.id,
      encryptedContent: item.encryptedContent,
    })),
  }
  return { [CODEX_PROVIDER_METADATA_KEY]: metadata }
}

function outputSchema(options: LanguageModelV4CallOptions): unknown {
  return options.responseFormat?.type === 'json' ? options.responseFormat.schema : undefined
}

function functionTools(options: LanguageModelV4CallOptions): unknown[] | undefined {
  const tools = options.tools?.filter((tool) => tool.type === 'function')
  return tools?.length ? tools : undefined
}

export function createCodexLanguageModel(
  providerType: 'openai-codex',
  modelId: string,
  toolExecutor?: CodexToolExecutor,
): LanguageModelV4 {
  const transport: CodexTransport = codexService

  function buildRequest(options: LanguageModelV4CallOptions): CodexTurnRequest {
    const { system, prompt, input } = buildPrompt(options.prompt)
    return {
      model: modelId,
      system,
      prompt,
      input,
      reasoningEffort: reasoningEffort(options),
      outputSchema: outputSchema(options),
      tools: functionTools(options),
      toolChoice: options.toolChoice,
      toolExecutor,
      signal: options.abortSignal,
    }
  }

  async function collect(request: CodexTurnRequest) {
    let text = ''
    let reasoning = ''
    const toolCalls: CodexToolCall[] = []
    const toolResults: CodexToolResult[] = []
    const reasoningItems: CodexReasoningItem[] = []

    for await (const delta of transport.streamTurn(request)) {
      text += delta.content
      if (delta.reasoning) reasoning += delta.reasoning
      if (delta.toolCall) toolCalls.push(delta.toolCall)
      if (delta.toolResult) toolResults.push(delta.toolResult)
      if (
        delta.reasoningItem &&
        !reasoningItems.some((item) => item.id === delta.reasoningItem?.id)
      ) {
        reasoningItems.push(delta.reasoningItem)
      }
    }

    return { text, reasoning, toolCalls, toolResults, reasoningItems }
  }

  function finishReason(toolCalls: CodexToolCall[]) {
    return toolCalls.some((toolCall) => !toolCall.providerExecuted)
      ? { unified: 'tool-calls' as const, raw: 'tool_calls' }
      : { unified: 'stop' as const, raw: 'stop' }
  }

  function resultValue(value: unknown): any {
    if (value === undefined) return ''
    if (value === null) return ''
    return value
  }

  return {
    specificationVersion: 'v4',
    provider: providerType,
    modelId,
    supportedUrls: {},
    doGenerate: async (options) => {
      const result = await collect(buildRequest(options))
      const content: Array<any> = []
      if (result.reasoning || result.reasoningItems.length > 0) {
        content.push({
          type: 'reasoning',
          text: result.reasoning,
          providerMetadata: reasoningMetadata(result.reasoningItems),
        })
      }
      if (result.text) content.push({ type: 'text', text: result.text })
      for (const toolCall of result.toolCalls) {
        content.push({
          type: 'tool-call',
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          input: jsonString(toolCall.input),
          providerExecuted: toolCall.providerExecuted,
          dynamic: toolCall.dynamic,
        })
      }
      for (const toolResult of result.toolResults) {
        content.push({
          type: 'tool-result',
          toolCallId: toolResult.id,
          toolName: toolResult.name,
          result: resultValue(toolResult.result),
          isError: toolResult.isError,
          dynamic: toolResult.dynamic,
        })
      }
      return {
        content,
        finishReason: finishReason(result.toolCalls),
        usage: EMPTY_USAGE,
        warnings: [],
      }
    },
    doStream: async (options) => {
      const request = buildRequest(options)

      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        async start(controller) {
          const textId = 'codex-text'
          const reasoningId = 'codex-reasoning'
          let textStarted = false
          let reasoningStarted = false
          const toolCalls: CodexToolCall[] = []
          const reasoningItems: CodexReasoningItem[] = []

          controller.enqueue({ type: 'stream-start', warnings: [] })

          try {
            for await (const delta of transport.streamTurn(request)) {
              if (delta.reasoning || delta.reasoningItem) {
                if (!reasoningStarted) {
                  reasoningStarted = true
                  controller.enqueue({ type: 'reasoning-start', id: reasoningId })
                }
                if (delta.reasoning) {
                  controller.enqueue({
                    type: 'reasoning-delta',
                    id: reasoningId,
                    delta: delta.reasoning,
                  })
                }
                if (
                  delta.reasoningItem &&
                  !reasoningItems.some((item) => item.id === delta.reasoningItem?.id)
                ) {
                  reasoningItems.push(delta.reasoningItem)
                }
              }

              if (delta.content) {
                if (!textStarted) {
                  textStarted = true
                  controller.enqueue({ type: 'text-start', id: textId })
                }
                controller.enqueue({ type: 'text-delta', id: textId, delta: delta.content })
              }

              if (delta.toolCall) {
                toolCalls.push(delta.toolCall)
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: delta.toolCall.id,
                  toolName: delta.toolCall.name,
                  input: jsonString(delta.toolCall.input),
                  providerExecuted: delta.toolCall.providerExecuted,
                  dynamic: delta.toolCall.dynamic,
                })
              }

              if (delta.toolResult) {
                controller.enqueue({
                  type: 'tool-result',
                  toolCallId: delta.toolResult.id,
                  toolName: delta.toolResult.name,
                  result: resultValue(delta.toolResult.result),
                  isError: delta.toolResult.isError,
                  dynamic: delta.toolResult.dynamic,
                })
              }
            }

            if (reasoningStarted) {
              controller.enqueue({
                type: 'reasoning-end',
                id: reasoningId,
                providerMetadata: reasoningMetadata(reasoningItems),
              })
            }
            if (textStarted) controller.enqueue({ type: 'text-end', id: textId })
            controller.enqueue({
              type: 'finish',
              finishReason: finishReason(toolCalls),
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

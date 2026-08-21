import type { LanguageModelV4Prompt } from '@ai-sdk/provider'
import { describe, expect, it } from 'vitest'
import { buildPrompt } from './codexModel'

describe('buildPrompt', () => {
  it('replays encrypted reasoning items without sending their visible text', () => {
    const prompt = [
      { role: 'system', content: 'Be concise.' },
      { role: 'user', content: [{ type: 'text', text: 'Find the answer.' }] },
      {
        role: 'assistant',
        content: [
          {
            type: 'reasoning',
            text: 'This internal explanation must not be replayed as plain text.',
            providerOptions: {
              openaiCodex: {
                reasoningItems: [
                  {
                    id: 'rs_123',
                    encryptedContent: 'opaque-reasoning',
                    summary: [{ type: 'summary_text', text: 'A summary.' }],
                  },
                ],
              },
            },
          },
          { type: 'text', text: 'The answer.' },
          {
            type: 'tool-call',
            toolCallId: 'call_123',
            toolName: 'lookup',
            input: { query: 'answer' },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            type: 'tool-result',
            toolCallId: 'call_123',
            toolName: 'lookup',
            output: { type: 'json', value: { found: true } },
          },
        ],
      },
    ] as LanguageModelV4Prompt

    expect(buildPrompt(prompt).input).toEqual([
      {
        role: 'user',
        content: [{ type: 'input_text', text: 'Find the answer.' }],
      },
      {
        type: 'reasoning',
        id: 'rs_123',
        encrypted_content: 'opaque-reasoning',
        summary: [{ type: 'summary_text', text: 'A summary.' }],
      },
      {
        role: 'assistant',
        content: [{ type: 'output_text', text: 'The answer.' }],
      },
      {
        type: 'function_call',
        call_id: 'call_123',
        name: 'lookup',
        arguments: '{"query":"answer"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_123',
        output: '{"found":true}',
      },
    ])
  })
})

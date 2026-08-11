import { invoke } from '@tauri-apps/api/core'

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
  loginId: string | null
  authUrl: string | null
  verificationUrl: string | null
  userCode: string | null
}

export interface CodexModelReasoningEffort {
  reasoningEffort: string
  description: string | null
}

export interface CodexModel {
  id: string
  model: string | null
  displayName: string | null
  hidden: boolean
  defaultReasoningEffort: string | null
  supportedReasoningEfforts: CodexModelReasoningEffort[]
  inputModalities: string[]
  isDefault: boolean
}

class CodexService {
  async readAccount(): Promise<CodexAccountState> {
    return invoke('codex_account_read')
  }

  async startLogin(): Promise<CodexLoginStart> {
    return invoke('codex_login_start')
  }

  async logout(): Promise<void> {
    return invoke('codex_logout')
  }

  async listModels(): Promise<CodexModel[]> {
    return invoke('codex_list_models')
  }

  async disconnect(): Promise<void> {
    return invoke('codex_disconnect')
  }
}

export const codexService = new CodexService()

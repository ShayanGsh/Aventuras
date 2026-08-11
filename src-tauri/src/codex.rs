use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{broadcast, oneshot, Mutex};
use tokio::time::timeout;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const NOTIFICATION_BUFFER: usize = 256;

#[derive(Debug, Clone)]
pub struct CodexNotification {
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexAccount {
    #[serde(rename = "type")]
    pub auth_mode: String,
    pub email: Option<String>,
    pub plan_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexAccountState {
    pub account: Option<CodexAccount>,
    pub requires_openai_auth: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexLoginStart {
    #[serde(rename = "type")]
    pub login_type: String,
    pub login_id: Option<String>,
    pub auth_url: Option<String>,
    pub verification_url: Option<String>,
    pub user_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexModelReasoningEffort {
    pub reasoning_effort: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexModel {
    pub id: String,
    pub model: Option<String>,
    pub display_name: Option<String>,
    #[serde(default)]
    pub hidden: bool,
    pub default_reasoning_effort: Option<String>,
    #[serde(default)]
    pub supported_reasoning_efforts: Vec<CodexModelReasoningEffort>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub is_default: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexModelListPage {
    #[serde(default)]
    data: Vec<CodexModel>,
    next_cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexLoginCompleted {
    login_id: Option<String>,
    success: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexTurnHandle {
    pub thread_id: String,
    pub turn_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexTurnDelta {
    thread_id: String,
    turn_id: String,
    content: String,
    reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexTurnCompleted {
    thread_id: String,
    turn_id: String,
    status: String,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CodexThreadStartResponse {
    thread: CodexThreadReference,
}

#[derive(Debug, Deserialize)]
struct CodexThreadReference {
    id: String,
}

#[derive(Debug, Deserialize)]
struct CodexTurnStartResponse {
    turn: CodexTurnReference,
}

#[derive(Debug, Deserialize)]
struct CodexTurnReference {
    id: String,
    status: String,
    #[serde(default)]
    error: Option<CodexTurnError>,
}

#[derive(Debug, Deserialize)]
struct CodexTurnError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct RpcErrorPayload {
    code: i64,
    message: String,
    #[serde(default)]
    data: Option<Value>,
}

impl Display for RpcErrorPayload {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(data) = &self.data {
            write!(formatter, "{} ({}): {}", self.message, self.code, data)
        } else {
            write!(formatter, "{} ({})", self.message, self.code)
        }
    }
}

type PendingResponse = Result<Value, String>;

pub struct CodexConnection {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Mutex<HashMap<u64, oneshot::Sender<PendingResponse>>>,
    next_request_id: AtomicU64,
    notification_tx: broadcast::Sender<CodexNotification>,
}

impl CodexConnection {
    async fn spawn() -> Result<Arc<Self>, String> {
        let mut child = spawn_codex_process()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Codex app-server did not expose stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Codex app-server did not expose stdout".to_string())?;
        let (notification_tx, _) = broadcast::channel(NOTIFICATION_BUFFER);

        let connection = Arc::new(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending: Mutex::new(HashMap::new()),
            next_request_id: AtomicU64::new(1),
            notification_tx,
        });

        let reader_connection = Arc::clone(&connection);
        tauri::async_runtime::spawn(async move {
            reader_connection.read_stdout(stdout).await;
        });

        Ok(connection)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<CodexNotification> {
        self.notification_tx.subscribe()
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();

        self.pending.lock().await.insert(id, sender);

        let request = build_request(id, method, params);
        if let Err(error) = self.send_message(request).await {
            self.pending.lock().await.remove(&id);
            return Err(error);
        }

        match timeout(REQUEST_TIMEOUT, receiver).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(format!("Codex app-server canceled request: {method}")),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(format!("Codex app-server request timed out: {method}"))
            }
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        self.send_message(build_notification(method, params)).await
    }

    async fn send_message(&self, message: Value) -> Result<(), String> {
        let mut stdin = self.stdin.lock().await;
        let encoded = serde_json::to_string(&message)
            .map_err(|error| format!("Failed to encode Codex request: {error}"))?;

        stdin
            .write_all(encoded.as_bytes())
            .await
            .map_err(|error| format!("Failed to write to Codex app-server: {error}"))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|error| format!("Failed to terminate Codex request: {error}"))?;
        stdin
            .flush()
            .await
            .map_err(|error| format!("Failed to flush Codex request: {error}"))
    }

    async fn read_stdout(self: Arc<Self>, stdout: ChildStdout) {
        let mut lines = BufReader::new(stdout).lines();

        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => {
                    self.fail_pending("Codex app-server closed its output stream".to_string())
                        .await;
                    return;
                }
                Err(error) => {
                    self.fail_pending(format!("Failed to read from Codex app-server: {error}"))
                        .await;
                    return;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            let message: Value = match serde_json::from_str(&line) {
                Ok(message) => message,
                Err(error) => {
                    eprintln!("[Codex] Ignoring invalid app-server message: {error}");
                    continue;
                }
            };

            if let Some(method) = message.get("method").and_then(Value::as_str) {
                let params = message.get("params").cloned().unwrap_or_else(|| json!({}));
                let _ = self.notification_tx.send(CodexNotification {
                    method: method.to_string(),
                    params,
                });

                // The first generation slice does not expose approval or other
                // server-initiated request UX. Rejecting these explicitly keeps the
                // app-server from waiting indefinitely if a turn requests one.
                if let Some(id) = message.get("id") {
                    let _ = self
                        .send_message(json!({
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": "Aventuras does not support this server request yet"
                            }
                        }))
                        .await;
                }
                continue;
            }

            let Some(id) = message.get("id").and_then(Value::as_u64) else {
                continue;
            };

            let Some(sender) = self.pending.lock().await.remove(&id) else {
                continue;
            };

            let result = if let Some(error) = message.get("error") {
                match serde_json::from_value::<RpcErrorPayload>(error.clone()) {
                    Ok(error) => Err(error.to_string()),
                    Err(parse_error) => {
                        Err(format!("Codex returned an invalid error: {parse_error}"))
                    }
                }
            } else if let Some(result) = message.get("result") {
                Ok(result.clone())
            } else {
                Err("Codex returned a response without a result or error".to_string())
            };

            let _ = sender.send(result);
        }
    }

    async fn fail_pending(&self, error: String) {
        let mut pending = self.pending.lock().await;
        for (_, sender) in pending.drain() {
            let _ = sender.send(Err(error.clone()));
        }
    }

    async fn is_alive(&self) -> bool {
        self.child
            .lock()
            .await
            .try_wait()
            .map(|status| status.is_none())
            .unwrap_or(false)
    }

    async fn shutdown(&self) {
        let mut child = self.child.lock().await;
        if let Err(error) = child.kill().await {
            eprintln!("[Codex] Failed to stop app-server: {error}");
        }
    }
}

#[derive(Default)]
pub struct CodexState {
    connection: Mutex<Option<Arc<CodexConnection>>>,
}

impl CodexState {
    pub async fn connection(&self) -> Result<Arc<CodexConnection>, String> {
        let mut slot = self.connection.lock().await;

        if let Some(existing) = slot.as_ref() {
            if existing.is_alive().await {
                return Ok(Arc::clone(existing));
            }
        }

        if let Some(previous) = slot.take() {
            previous.shutdown().await;
        }

        let connection = CodexConnection::spawn().await?;
        connection
            .request(
                "initialize",
                json!({
                    "clientInfo": {
                        "name": "aventuras",
                        "title": "Aventuras",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;
        connection.notify("initialized", json!({})).await?;

        *slot = Some(Arc::clone(&connection));
        Ok(connection)
    }

    pub async fn disconnect(&self) {
        if let Some(connection) = self.connection.lock().await.take() {
            connection.shutdown().await;
        }
    }
}

#[tauri::command]
pub async fn codex_account_read(state: State<'_, CodexState>) -> Result<CodexAccountState, String> {
    let connection = state.connection().await?;
    let response = connection.request("account/read", json!({})).await?;
    serde_json::from_value(response)
        .map_err(|error| format!("Codex returned an invalid account response: {error}"))
}

#[tauri::command]
pub async fn codex_login_start(
    app: AppHandle,
    state: State<'_, CodexState>,
) -> Result<CodexLoginStart, String> {
    let connection = state.connection().await?;
    let mut notifications = connection.subscribe();
    let response = connection
        .request(
            "account/login/start",
            json!({
                "type": "chatgpt",
                "useHostedLoginSuccessPage": true,
                "appBrand": "chatgpt"
            }),
        )
        .await?;
    let login: CodexLoginStart = serde_json::from_value(response)
        .map_err(|error| format!("Codex returned an invalid login response: {error}"))?;

    let login_id = login.login_id.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            let notification = match notifications.recv().await {
                Ok(notification) => notification,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            };

            if notification.method != "account/login/completed" {
                continue;
            }

            let notification_login_id = notification
                .params
                .get("loginId")
                .and_then(Value::as_str)
                .map(String::from);
            if login_id.is_some() && notification_login_id != login_id {
                continue;
            }

            let success = notification
                .params
                .get("success")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let error = notification
                .params
                .get("error")
                .and_then(Value::as_str)
                .map(String::from);

            let _ = app.emit(
                "codex-login-completed",
                CodexLoginCompleted {
                    login_id: notification_login_id,
                    success,
                    error,
                },
            );
            break;
        }
    });

    Ok(login)
}

#[tauri::command]
pub async fn codex_logout(state: State<'_, CodexState>) -> Result<(), String> {
    let connection = state.connection().await?;
    connection.request("account/logout", json!({})).await?;
    Ok(())
}

#[tauri::command]
pub async fn codex_list_models(state: State<'_, CodexState>) -> Result<Vec<CodexModel>, String> {
    let connection = state.connection().await?;
    let mut models = Vec::new();
    let mut cursor: Option<String> = None;

    loop {
        let mut params = json!({
            "limit": 100,
            "includeHidden": false
        });
        if let Some(cursor) = &cursor {
            params["cursor"] = Value::String(cursor.clone());
        }

        let response = connection.request("model/list", params).await?;
        let page: CodexModelListPage = serde_json::from_value(response)
            .map_err(|error| format!("Codex returned an invalid model list: {error}"))?;
        models.extend(page.data);

        match page.next_cursor {
            Some(next_cursor) if !next_cursor.is_empty() => cursor = Some(next_cursor),
            _ => break,
        }
    }

    Ok(models)
}

#[tauri::command]
pub async fn codex_turn_start(
    app: AppHandle,
    state: State<'_, CodexState>,
    model: String,
    system: String,
    prompt: String,
    reasoning_effort: String,
) -> Result<CodexTurnHandle, String> {
    let connection = state.connection().await?;
    let thread_response = connection
        .request(
            "thread/start",
            json!({
                "model": model,
                "ephemeral": true,
                "baseInstructions": system,
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "serviceName": "aventuras"
            }),
        )
        .await?;
    let thread: CodexThreadStartResponse = serde_json::from_value(thread_response)
        .map_err(|error| format!("Codex returned an invalid thread response: {error}"))?;

    let notifications = connection.subscribe();
    let turn_response = connection
        .request(
            "turn/start",
            json!({
                "threadId": thread.thread.id,
                "model": model,
                "effort": reasoning_effort,
                "approvalPolicy": "never",
                "sandboxPolicy": {
                    "type": "readOnly",
                    "networkAccess": false
                },
                "input": [{
                    "type": "text",
                    "text": prompt
                }]
            }),
        )
        .await?;
    let turn: CodexTurnStartResponse = serde_json::from_value(turn_response)
        .map_err(|error| format!("Codex returned an invalid turn response: {error}"))?;

    let handle = CodexTurnHandle {
        thread_id: thread.thread.id,
        turn_id: turn.turn.id,
    };

    if turn.turn.status == "inProgress" {
        let app = app.clone();
        let thread_id = handle.thread_id.clone();
        let turn_id = handle.turn_id.clone();
        tauri::async_runtime::spawn(async move {
            monitor_turn(app, notifications, thread_id, turn_id).await;
        });
    } else {
        emit_turn_completed(
            &app,
            CodexTurnCompleted {
                thread_id: handle.thread_id.clone(),
                turn_id: handle.turn_id.clone(),
                status: turn.turn.status,
                error: turn.turn.error.map(|error| error.message),
            },
        );
    }

    Ok(handle)
}

#[tauri::command]
pub async fn codex_turn_interrupt(
    state: State<'_, CodexState>,
    thread_id: String,
    turn_id: String,
) -> Result<(), String> {
    let connection = state.connection().await?;
    connection
        .request(
            "turn/interrupt",
            json!({
                "threadId": thread_id,
                "turnId": turn_id
            }),
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn codex_disconnect(state: State<'_, CodexState>) -> Result<(), String> {
    state.disconnect().await;
    Ok(())
}

fn build_request(id: u64, method: &str, params: Value) -> Value {
    json!({ "method": method, "id": id, "params": params })
}

fn build_notification(method: &str, params: Value) -> Value {
    json!({ "method": method, "params": params })
}

async fn monitor_turn(
    app: AppHandle,
    mut notifications: broadcast::Receiver<CodexNotification>,
    thread_id: String,
    turn_id: String,
) {
    loop {
        let notification = match notifications.recv().await {
            Ok(notification) => notification,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => {
                emit_turn_completed(
                    &app,
                    CodexTurnCompleted {
                        thread_id,
                        turn_id,
                        status: "failed".to_string(),
                        error: Some("Codex app-server closed its notification stream".to_string()),
                    },
                );
                return;
            }
        };

        let notification_thread_id = notification.params.get("threadId").and_then(Value::as_str);
        if notification_thread_id != Some(thread_id.as_str()) {
            continue;
        }

        match notification.method.as_str() {
            "item/agentMessage/delta" => {
                if notification.params.get("turnId").and_then(Value::as_str)
                    != Some(turn_id.as_str())
                {
                    continue;
                }
                if let Some(delta) = notification.params.get("delta").and_then(Value::as_str) {
                    emit_turn_delta(
                        &app,
                        CodexTurnDelta {
                            thread_id: thread_id.clone(),
                            turn_id: turn_id.clone(),
                            content: delta.to_string(),
                            reasoning: None,
                        },
                    );
                }
            }
            "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
                if notification.params.get("turnId").and_then(Value::as_str)
                    != Some(turn_id.as_str())
                {
                    continue;
                }
                if let Some(delta) = notification.params.get("delta").and_then(Value::as_str) {
                    emit_turn_delta(
                        &app,
                        CodexTurnDelta {
                            thread_id: thread_id.clone(),
                            turn_id: turn_id.clone(),
                            content: String::new(),
                            reasoning: Some(delta.to_string()),
                        },
                    );
                }
            }
            "turn/completed" => {
                let completed_turn_id = notification
                    .params
                    .get("turn")
                    .and_then(|turn| turn.get("id"))
                    .and_then(Value::as_str);
                if completed_turn_id != Some(turn_id.as_str()) {
                    continue;
                }
                let status = notification
                    .params
                    .get("turn")
                    .and_then(|turn| turn.get("status"))
                    .and_then(Value::as_str)
                    .unwrap_or("failed")
                    .to_string();
                let error = notification
                    .params
                    .get("turn")
                    .and_then(|turn| turn.get("error"))
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .map(String::from);

                emit_turn_completed(
                    &app,
                    CodexTurnCompleted {
                        thread_id,
                        turn_id,
                        status,
                        error,
                    },
                );
                return;
            }
            _ => {}
        }
    }
}

fn emit_turn_delta(app: &AppHandle, delta: CodexTurnDelta) {
    let _ = app.emit("codex-turn-delta", delta);
}

fn emit_turn_completed(app: &AppHandle, completed: CodexTurnCompleted) {
    let _ = app.emit("codex-turn-completed", completed);
}

fn spawn_codex_process() -> Result<Child, String> {
    let mut candidates = Vec::new();
    if let Ok(path) = std::env::var("CODEX_BIN") {
        if !path.trim().is_empty() {
            candidates.push(path);
        }
    }
    candidates.extend([
        "codex".to_string(),
        "/opt/homebrew/bin/codex".to_string(),
        "/usr/local/bin/codex".to_string(),
        "/home/linuxbrew/.linuxbrew/bin/codex".to_string(),
    ]);

    let mut errors = Vec::new();
    for executable in candidates {
        let result = Command::new(&executable)
            .args(["app-server", "--stdio"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();

        match result {
            Ok(child) => return Ok(child),
            Err(error) => errors.push(format!("{executable}: {error}")),
        }
    }

    Err(format!(
        "Could not start Codex app-server. Install the Codex CLI or set CODEX_BIN. Attempts: {}",
        errors.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::{build_notification, build_request};
    use serde_json::json;

    #[test]
    fn builds_json_rpc_request() {
        assert_eq!(
            build_request(7, "model/list", json!({ "limit": 20 })),
            json!({
                "method": "model/list",
                "id": 7,
                "params": { "limit": 20 }
            })
        );
    }

    #[test]
    fn builds_json_rpc_notification_without_an_id() {
        assert_eq!(
            build_notification("initialized", json!({})),
            json!({ "method": "initialized", "params": {} })
        );
    }

    #[test]
    fn parses_account_state_from_app_server_response() {
        let account: super::CodexAccountState = serde_json::from_value(json!({
            "account": {
                "type": "chatgpt",
                "email": "writer@example.com",
                "planType": "plus"
            },
            "requiresOpenaiAuth": true
        }))
        .expect("account response should parse");

        assert_eq!(
            account
                .account
                .expect("account should be present")
                .auth_mode,
            "chatgpt"
        );
        assert!(account.requires_openai_auth);
    }

    #[test]
    fn parses_model_capabilities_from_app_server_response() {
        let page: super::CodexModelListPage = serde_json::from_value(json!({
            "data": [{
                "id": "gpt-5.6-sol",
                "model": "gpt-5.6-sol",
                "displayName": "GPT-5.6-Sol",
                "hidden": false,
                "defaultReasoningEffort": "low",
                "supportedReasoningEfforts": [{
                    "reasoningEffort": "max",
                    "description": "Highest effort"
                }],
                "inputModalities": ["text", "image"],
                "isDefault": true
            }],
            "nextCursor": null
        }))
        .expect("model response should parse");

        assert_eq!(page.data[0].id, "gpt-5.6-sol");
        assert_eq!(
            page.data[0].supported_reasoning_efforts[0].reasoning_effort,
            "max"
        );
        assert!(page.next_cursor.is_none());
    }
}

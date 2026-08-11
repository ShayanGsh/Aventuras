use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
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

fn build_request(id: u64, method: &str, params: Value) -> Value {
    json!({ "method": method, "id": id, "params": params })
}

fn build_notification(method: &str, params: Value) -> Value {
    json!({ "method": method, "params": params })
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
}

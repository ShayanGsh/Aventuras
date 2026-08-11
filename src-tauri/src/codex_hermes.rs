use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::{oneshot, Mutex};
use tokio::time::sleep;
use uuid::Uuid;

const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const CODEX_MODELS_URL: &str = "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0";
const OAUTH_ISSUER: &str = "https://auth.openai.com";
const OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OAUTH_DEVICE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const OAUTH_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const OAUTH_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const AUTH_FILE_NAME: &str = "codex-hermes-auth.json";
const CODEX_USER_AGENT: &str = "codex_cli_rs/0.0.0 (Aventuras)";
const CODEX_ORIGINATOR: &str = "codex_cli_rs";
const TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const LOGIN_TIMEOUT: Duration = Duration::from_secs(15 * 60);

#[derive(Default, Clone)]
pub struct CodexHermesState {
    active_turns: Arc<Mutex<HashMap<String, oneshot::Sender<()>>>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexHermesAccount {
    #[serde(rename = "type")]
    pub auth_mode: String,
    pub email: Option<String>,
    pub plan_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexHermesAccountState {
    pub account: Option<CodexHermesAccount>,
    pub requires_openai_auth: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexHermesLoginStart {
    pub login_type: String,
    pub login_id: String,
    pub auth_url: String,
    pub verification_url: String,
    pub user_code: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexHermesModel {
    pub id: String,
    pub reasoning: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexHermesTurnHandle {
    pub thread_id: String,
    pub turn_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexHermesTurnDelta {
    thread_id: String,
    turn_id: String,
    content: String,
    reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexHermesTurnCompleted {
    thread_id: String,
    turn_id: String,
    status: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexHermesLoginCompleted {
    login_id: String,
    success: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredCodexAuth {
    access_token: String,
    refresh_token: Option<String>,
    expires_at: Option<i64>,
    account_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    user_code: String,
    device_auth_id: String,
    interval: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct DeviceAuthTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
}

enum TurnResult {
    Completed,
    Interrupted,
    Failed(String),
}

fn auth_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map(|path| path.join(AUTH_FILE_NAME))
        .map_err(|error| format!("Failed to resolve Codex-Hermes auth directory: {error}"))
}

fn read_auth(app: &AppHandle) -> Result<Option<StoredCodexAuth>, String> {
    let path = auth_file_path(app)?;
    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Failed to read Codex-Hermes auth: {error}")),
    };

    let auth = serde_json::from_str(&contents)
        .map_err(|error| format!("Codex-Hermes auth is invalid: {error}"))?;
    Ok(Some(auth))
}

fn save_auth(app: &AppHandle, auth: &StoredCodexAuth) -> Result<(), String> {
    let path = auth_file_path(app)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create Codex-Hermes auth directory: {error}"))?;
    }

    let encoded = serde_json::to_vec_pretty(auth)
        .map_err(|error| format!("Failed to encode Codex-Hermes auth: {error}"))?;
    fs::write(&path, encoded)
        .map_err(|error| format!("Failed to save Codex-Hermes auth: {error}"))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("Failed to protect Codex-Hermes auth: {error}"))?;
    }

    Ok(())
}

fn delete_auth(app: &AppHandle) -> Result<(), String> {
    let path = auth_file_path(app)?;
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("Failed to remove Codex-Hermes auth: {error}")),
    }
}

fn jwt_claims(access_token: &str) -> Option<Value> {
    let payload = access_token.split('.').nth(1)?;
    let padded = format!("{payload}{}", "=".repeat((4 - payload.len() % 4) % 4));
    let bytes = URL_SAFE.decode(padded).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn account_id_from_token(access_token: &str) -> Option<String> {
    jwt_claims(access_token)?
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(String::from)
}

fn expiry_from_token(access_token: &str) -> Option<i64> {
    jwt_claims(access_token)?.get("exp")?.as_i64()
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn token_needs_refresh(auth: &StoredCodexAuth) -> bool {
    auth.expires_at
        .map(|expires_at| expires_at <= unix_now() + TOKEN_REFRESH_SKEW_SECONDS)
        .unwrap_or(false)
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .user_agent(CODEX_USER_AGENT)
        .build()
        .map_err(|error| format!("Failed to create Codex-Hermes HTTP client: {error}"))
}

async fn response_error(context: &str, response: reqwest::Response) -> String {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let detail = body.trim();
    if detail.is_empty() {
        format!("{context} (HTTP {status})")
    } else {
        format!(
            "{context} (HTTP {status}): {}",
            detail.chars().take(500).collect::<String>()
        )
    }
}

async fn refresh_auth(app: &AppHandle, auth: &StoredCodexAuth) -> Result<StoredCodexAuth, String> {
    let refresh_token = auth
        .refresh_token
        .as_deref()
        .filter(|token| !token.trim().is_empty())
        .ok_or_else(|| "Codex-Hermes sign-in has expired. Please sign in again.".to_string())?;
    let client = http_client()?;
    let response = client
        .post(OAUTH_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", OAUTH_CLIENT_ID),
        ])
        .send()
        .await
        .map_err(|error| format!("Failed to refresh Codex-Hermes sign-in: {error}"))?;

    if !response.status().is_success() {
        return Err(response_error("Codex-Hermes sign-in refresh failed", response).await);
    }

    let token_response: OAuthTokenResponse = response
        .json()
        .await
        .map_err(|error| format!("Codex-Hermes returned an invalid refresh response: {error}"))?;
    let expires_at = token_response
        .expires_in
        .map(|expires_in| unix_now() + expires_in)
        .or_else(|| expiry_from_token(&token_response.access_token));
    let next = StoredCodexAuth {
        account_id: account_id_from_token(&token_response.access_token)
            .or_else(|| auth.account_id.clone()),
        access_token: token_response.access_token,
        refresh_token: token_response
            .refresh_token
            .or_else(|| auth.refresh_token.clone()),
        expires_at,
    };
    save_auth(app, &next)?;
    Ok(next)
}

async fn resolve_auth(app: &AppHandle) -> Result<StoredCodexAuth, String> {
    let mut auth = read_auth(app)?.ok_or_else(|| {
        "No Codex-Hermes sign-in found. Sign in with ChatGPT to use this provider.".to_string()
    })?;

    if auth.access_token.trim().is_empty() {
        return Err("Codex-Hermes sign-in has no access token. Please sign in again.".to_string());
    }

    if token_needs_refresh(&auth) {
        auth = refresh_auth(app, &auth).await?;
    }

    if auth.account_id.is_none() {
        auth.account_id = account_id_from_token(&auth.access_token);
        save_auth(app, &auth)?;
    }

    Ok(auth)
}

fn apply_codex_headers(
    request: reqwest::RequestBuilder,
    auth: &StoredCodexAuth,
) -> reqwest::RequestBuilder {
    let request = request
        .bearer_auth(&auth.access_token)
        .header("originator", CODEX_ORIGINATOR)
        .header("Accept", "application/json");
    match auth.account_id.as_deref() {
        Some(account_id) if !account_id.is_empty() => {
            request.header("ChatGPT-Account-Id", account_id)
        }
        _ => request,
    }
}

#[tauri::command]
pub async fn codex_hermes_account_read(app: AppHandle) -> Result<CodexHermesAccountState, String> {
    if read_auth(&app)?.is_none() {
        return Ok(CodexHermesAccountState {
            account: None,
            requires_openai_auth: true,
        });
    }

    let _auth = resolve_auth(&app).await?;
    Ok(CodexHermesAccountState {
        account: Some(CodexHermesAccount {
            auth_mode: "chatgpt".to_string(),
            email: None,
            plan_type: None,
        }),
        requires_openai_auth: false,
    })
}

#[tauri::command]
pub async fn codex_hermes_login_start(app: AppHandle) -> Result<CodexHermesLoginStart, String> {
    let client = http_client()?;
    let response = client
        .post(OAUTH_DEVICE_URL)
        .header("Content-Type", "application/json")
        .json(&json!({ "client_id": OAUTH_CLIENT_ID }))
        .send()
        .await
        .map_err(|error| format!("Failed to request Codex-Hermes sign-in: {error}"))?;

    if !response.status().is_success() {
        return Err(response_error("Codex-Hermes sign-in request failed", response).await);
    }

    let body = response
        .text()
        .await
        .map_err(|error| format!("Failed to read Codex-Hermes sign-in response: {error}"))?;
    let device: DeviceCodeResponse = serde_json::from_str(&body)
        .map_err(|error| format!("Codex-Hermes returned an invalid device code: {error}"))?;
    if device.user_code.trim().is_empty() || device.device_auth_id.trim().is_empty() {
        return Err("Codex-Hermes sign-in response was missing the device code.".to_string());
    }

    let login = CodexHermesLoginStart {
        login_type: "device_code".to_string(),
        login_id: device.device_auth_id.clone(),
        auth_url: format!("{OAUTH_ISSUER}/codex/device"),
        verification_url: format!("{OAUTH_ISSUER}/codex/device"),
        user_code: device.user_code.clone(),
    };

    let app_for_login = app.clone();
    let interval = device
        .interval
        .as_ref()
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
        .unwrap_or(5)
        .max(3);
    let login_id = device.device_auth_id.clone();
    let user_code = device.user_code.clone();
    tauri::async_runtime::spawn(async move {
        let result = complete_device_login(&app_for_login, &login_id, &user_code, interval).await;
        let event = match result {
            Ok(()) => CodexHermesLoginCompleted {
                login_id: login_id.clone(),
                success: true,
                error: None,
            },
            Err(error) => CodexHermesLoginCompleted {
                login_id: login_id.clone(),
                success: false,
                error: Some(error),
            },
        };
        let _ = app_for_login.emit("codex-hermes-login-completed", event);
    });

    Ok(login)
}

async fn complete_device_login(
    app: &AppHandle,
    device_auth_id: &str,
    user_code: &str,
    interval_seconds: u64,
) -> Result<(), String> {
    let client = http_client()?;
    let started = Instant::now();
    let device_token = loop {
        if started.elapsed() >= LOGIN_TIMEOUT {
            return Err("Codex-Hermes sign-in timed out. Please try again.".to_string());
        }
        sleep(Duration::from_secs(interval_seconds)).await;
        let response = client
            .post(OAUTH_DEVICE_TOKEN_URL)
            .header("Content-Type", "application/json")
            .json(&json!({
                "device_auth_id": device_auth_id,
                "user_code": user_code
            }))
            .send()
            .await
            .map_err(|error| format!("Failed while waiting for Codex-Hermes sign-in: {error}"))?;
        let status = response.status();
        if status.is_success() {
            let body = response.text().await.map_err(|error| {
                format!("Failed to read Codex-Hermes device authorization: {error}")
            })?;
            break serde_json::from_str::<DeviceAuthTokenResponse>(&body).map_err(|error| {
                format!("Codex-Hermes returned an invalid device authorization: {error}")
            })?;
        }
        if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        return Err(response_error("Codex-Hermes device authorization failed", response).await);
    };

    let response = client
        .post(OAUTH_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", device_token.authorization_code.as_str()),
            ("redirect_uri", OAUTH_REDIRECT_URI),
            ("client_id", OAUTH_CLIENT_ID),
            ("code_verifier", device_token.code_verifier.as_str()),
        ])
        .send()
        .await
        .map_err(|error| format!("Failed to exchange Codex-Hermes sign-in: {error}"))?;

    if !response.status().is_success() {
        return Err(response_error("Codex-Hermes token exchange failed", response).await);
    }

    let token_response: OAuthTokenResponse = response
        .json()
        .await
        .map_err(|error| format!("Codex-Hermes returned an invalid token response: {error}"))?;
    let expires_at = token_response
        .expires_in
        .map(|expires_in| unix_now() + expires_in)
        .or_else(|| expiry_from_token(&token_response.access_token));
    let auth = StoredCodexAuth {
        account_id: account_id_from_token(&token_response.access_token),
        access_token: token_response.access_token,
        refresh_token: token_response.refresh_token,
        expires_at,
    };
    save_auth(app, &auth)
}

#[tauri::command]
pub async fn codex_hermes_logout(app: AppHandle) -> Result<(), String> {
    delete_auth(&app)
}

#[tauri::command]
pub async fn codex_hermes_list_models(app: AppHandle) -> Result<Vec<CodexHermesModel>, String> {
    let auth = resolve_auth(&app).await?;
    let client = http_client()?;
    let response = apply_codex_headers(client.get(CODEX_MODELS_URL), &auth)
        .send()
        .await
        .map_err(|error| format!("Failed to fetch Codex-Hermes models: {error}"))?;
    if !response.status().is_success() {
        return Err(response_error("Codex-Hermes model discovery failed", response).await);
    }

    let body = response
        .json::<Value>()
        .await
        .map_err(|error| format!("Codex-Hermes returned an invalid model list: {error}"))?;
    let entries = body
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| "Codex-Hermes returned a model list without models.".to_string())?;

    let mut sortable = entries
        .iter()
        .filter_map(|entry| {
            let id = entry.get("slug")?.as_str()?.trim();
            if id.is_empty() {
                return None;
            }
            let visibility = entry
                .get("visibility")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            if visibility == "hide" || visibility == "hidden" {
                return None;
            }
            let priority = entry
                .get("priority")
                .and_then(Value::as_i64)
                .unwrap_or(10_000);
            Some((priority, id.to_string()))
        })
        .collect::<Vec<_>>();
    sortable.sort_by(|left, right| left.cmp(right));

    let mut seen = HashSet::new();
    Ok(sortable
        .into_iter()
        .filter_map(|(_, id)| {
            if seen.insert(id.clone()) {
                Some(CodexHermesModel {
                    id,
                    reasoning: true,
                })
            } else {
                None
            }
        })
        .collect())
}

#[tauri::command]
pub async fn codex_hermes_turn_start(
    app: AppHandle,
    state: State<'_, CodexHermesState>,
    model: String,
    system: String,
    prompt: String,
    reasoning_effort: String,
    output_schema: Option<Value>,
) -> Result<CodexHermesTurnHandle, String> {
    let auth = resolve_auth(&app).await?;
    let turn_id = Uuid::new_v4().to_string();
    let handle = CodexHermesTurnHandle {
        thread_id: format!("hermes-{turn_id}"),
        turn_id: turn_id.clone(),
    };
    let (cancel_tx, cancel_rx) = oneshot::channel();
    state
        .active_turns
        .lock()
        .await
        .insert(turn_id.clone(), cancel_tx);

    let app_for_turn = app.clone();
    let state_for_turn = state.inner().clone();
    let handle_for_turn = handle.clone();
    tauri::async_runtime::spawn(async move {
        let result = run_turn(
            &app_for_turn,
            &handle_for_turn,
            auth,
            &model,
            &system,
            &prompt,
            &reasoning_effort,
            output_schema,
            cancel_rx,
        )
        .await;

        let (status, error) = match result {
            TurnResult::Completed => ("completed".to_string(), None),
            TurnResult::Interrupted => ("interrupted".to_string(), None),
            TurnResult::Failed(error) => ("failed".to_string(), Some(error)),
        };
        emit_turn_completed(&app_for_turn, &handle_for_turn, status, error);
        state_for_turn
            .active_turns
            .lock()
            .await
            .remove(&handle_for_turn.turn_id);
    });

    Ok(handle)
}

#[tauri::command]
pub async fn codex_hermes_turn_interrupt(
    state: State<'_, CodexHermesState>,
    turn_id: String,
) -> Result<(), String> {
    if let Some(cancel) = state.active_turns.lock().await.remove(&turn_id) {
        let _ = cancel.send(());
    }
    Ok(())
}

#[tauri::command]
pub async fn codex_hermes_disconnect(state: State<'_, CodexHermesState>) -> Result<(), String> {
    let cancels = std::mem::take(&mut *state.active_turns.lock().await);
    for cancel in cancels.into_values() {
        let _ = cancel.send(());
    }
    Ok(())
}

async fn run_turn(
    app: &AppHandle,
    handle: &CodexHermesTurnHandle,
    auth: StoredCodexAuth,
    model: &str,
    system: &str,
    prompt: &str,
    reasoning_effort: &str,
    output_schema: Option<Value>,
    mut cancel: oneshot::Receiver<()>,
) -> TurnResult {
    let body = build_turn_body(model, system, prompt, reasoning_effort, output_schema);
    let client = match http_client() {
        Ok(client) => client,
        Err(error) => return TurnResult::Failed(error),
    };
    let response = match apply_codex_headers(
        client
            .post(format!("{CODEX_BASE_URL}/responses"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body),
        &auth,
    )
    .send()
    .await
    {
        Ok(response) => response,
        Err(error) => return TurnResult::Failed(format!("Codex-Hermes request failed: {error}")),
    };

    if !response.status().is_success() {
        return TurnResult::Failed(
            response_error("Codex-Hermes request was rejected", response).await,
        );
    }

    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    let mut emitted_text = false;

    loop {
        let chunk = tokio::select! {
            _ = &mut cancel => return TurnResult::Interrupted,
            chunk = stream.next() => chunk,
        };
        let Some(chunk) = chunk else { break };
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                return TurnResult::Failed(format!("Codex-Hermes stream failed: {error}"))
            }
        };
        buffer.extend_from_slice(&chunk);

        while let Some(event) = take_sse_event(&mut buffer) {
            let event = match event {
                Ok(event) => event,
                Err(error) => return TurnResult::Failed(error),
            };
            if let Some(result) = handle_stream_event(app, handle, &event, &mut emitted_text) {
                return result;
            }
        }
    }

    if !buffer.is_empty() {
        if let Some(event) = parse_sse_line(&buffer) {
            match event {
                Ok(event) => {
                    if let Some(result) =
                        handle_stream_event(app, handle, &event, &mut emitted_text)
                    {
                        return result;
                    }
                }
                Err(error) => return TurnResult::Failed(error),
            }
        }
    }

    if emitted_text {
        TurnResult::Completed
    } else {
        TurnResult::Failed("Codex-Hermes stream ended without a response.".to_string())
    }
}

fn build_turn_body(
    model: &str,
    system: &str,
    prompt: &str,
    reasoning_effort: &str,
    output_schema: Option<Value>,
) -> Value {
    let mut body = json!({
        "model": model,
        "instructions": system,
        "input": [{
            "role": "user",
            "content": [{ "type": "input_text", "text": prompt }]
        }],
        "store": false,
        "stream": true
    });

    match wire_reasoning_effort(reasoning_effort) {
        Some(effort) => {
            body["reasoning"] = json!({ "effort": effort, "summary": "auto" });
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
        None => body["include"] = json!([]),
    }

    if let Some(schema) = output_schema {
        body["text"] = json!({
            "format": {
                "type": "json_schema",
                "name": "aventuras_output",
                "strict": true,
                "schema": schema
            }
        });
    }

    body
}

fn wire_reasoning_effort(effort: &str) -> Option<&'static str> {
    match effort {
        "none" => None,
        "minimal" => Some("low"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("high"),
        "max" => Some("max"),
        _ => Some("medium"),
    }
}

fn take_sse_event(buffer: &mut Vec<u8>) -> Option<Result<Value, String>> {
    let newline = buffer.iter().position(|byte| *byte == b'\n')?;
    let line = buffer.drain(..=newline).collect::<Vec<_>>();
    parse_sse_line(&line)
}

fn parse_sse_line(line: &[u8]) -> Option<Result<Value, String>> {
    let line = String::from_utf8_lossy(line).trim().to_string();
    let data = line.strip_prefix("data:")?.trim();
    if data == "[DONE]" || data.is_empty() {
        return None;
    }
    Some(
        serde_json::from_str(data)
            .map_err(|error| format!("Codex-Hermes returned invalid stream data: {error}")),
    )
}

fn handle_stream_event(
    app: &AppHandle,
    handle: &CodexHermesTurnHandle,
    event: &Value,
    emitted_text: &mut bool,
) -> Option<TurnResult> {
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    if event_type == "response.completed" {
        return Some(TurnResult::Completed);
    }
    if event_type == "response.failed" || event_type == "error" {
        let error = event
            .get("error")
            .and_then(|value| value.get("message").or(Some(value)))
            .and_then(Value::as_str)
            .unwrap_or("Codex-Hermes returned an error")
            .to_string();
        return Some(TurnResult::Failed(error));
    }
    if event_type == "response.incomplete" {
        let reason = event
            .get("response")
            .and_then(|response| response.get("incomplete_details"))
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
            .unwrap_or("unknown reason");
        return Some(TurnResult::Failed(format!(
            "Codex-Hermes response was incomplete: {reason}"
        )));
    }

    if event_type == "response.output_text.delta" {
        if let Some(delta) = event.get("delta").and_then(Value::as_str) {
            *emitted_text = true;
            emit_turn_delta(app, handle, delta.to_string(), None);
        }
    } else if event_type == "response.output_text.done" && !*emitted_text {
        if let Some(text) = event.get("text").and_then(Value::as_str) {
            *emitted_text = true;
            emit_turn_delta(app, handle, text.to_string(), None);
        }
    } else if event_type.contains("reasoning") && event_type.ends_with(".delta") {
        if let Some(delta) = event.get("delta").and_then(Value::as_str) {
            emit_turn_delta(app, handle, String::new(), Some(delta.to_string()));
        }
    }

    None
}

fn emit_turn_delta(
    app: &AppHandle,
    handle: &CodexHermesTurnHandle,
    content: String,
    reasoning: Option<String>,
) {
    let _ = app.emit(
        "codex-hermes-turn-delta",
        CodexHermesTurnDelta {
            thread_id: handle.thread_id.clone(),
            turn_id: handle.turn_id.clone(),
            content,
            reasoning,
        },
    );
}

fn emit_turn_completed(
    app: &AppHandle,
    handle: &CodexHermesTurnHandle,
    status: String,
    error: Option<String>,
) {
    let _ = app.emit(
        "codex-hermes-turn-completed",
        CodexHermesTurnCompleted {
            thread_id: handle.thread_id.clone(),
            turn_id: handle.turn_id.clone(),
            status,
            error,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{build_turn_body, parse_sse_line, wire_reasoning_effort};
    use serde_json::json;

    #[test]
    fn builds_conservative_codex_responses_body() {
        let body = build_turn_body("gpt-5.6-terra", "system", "hello", "high", None);
        assert_eq!(body["model"], "gpt-5.6-terra");
        assert_eq!(body["stream"], true);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert!(body.get("temperature").is_none());
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn adds_json_schema_format_when_requested() {
        let body = build_turn_body(
            "gpt-5.6-luna",
            "system",
            "hello",
            "none",
            Some(json!({ "type": "object" })),
        );
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["include"], json!([]));
    }

    #[test]
    fn parses_text_delta_sse_line() {
        let event = parse_sse_line(br#"data: {"type":"response.output_text.delta","delta":"Hi"}"#)
            .expect("data line should produce an event")
            .expect("event should be valid JSON");
        assert_eq!(event["delta"], "Hi");
    }

    #[test]
    fn maps_unsupported_efforts_to_supported_wire_values() {
        assert_eq!(wire_reasoning_effort("minimal"), Some("low"));
        assert_eq!(wire_reasoning_effort("xhigh"), Some("high"));
        assert_eq!(wire_reasoning_effort("none"), None);
    }
}

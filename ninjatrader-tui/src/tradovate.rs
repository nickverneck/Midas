use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use anyhow::{Context, Result, bail};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::tungstenite::Message;

#[derive(Debug, Clone)]
pub enum ServiceCommand {
    Connect(AppConfig),
    SelectAccount { account_id: i64 },
    SearchContracts { query: String, limit: usize },
    SubscribeBars { contract: ContractSuggestion },
}

#[derive(Debug, Clone)]
pub enum ServiceEvent {
    Status(String),
    Error(String),
    Connected {
        env: TradingEnvironment,
        user_name: Option<String>,
        auth_mode: AuthMode,
    },
    Disconnected,
    AccountsLoaded(Vec<AccountInfo>),
    AccountSnapshotsLoaded(Vec<AccountSnapshot>),
    ContractSearchResults {
        query: String,
        results: Vec<ContractSuggestion>,
    },
    MarketSnapshot(MarketSnapshot),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccessTokenResponse {
    error_text: Option<String>,
    access_token: Option<String>,
    md_access_token: Option<String>,
    expiration_time: Option<String>,
    user_id: Option<i64>,
    name: Option<String>,
    has_live: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenCacheFile {
    token: String,
    #[serde(rename = "accessToken")]
    access_token: Option<String>,
    #[serde(rename = "mdAccessToken")]
    md_access_token: Option<String>,
    #[serde(rename = "expirationTime")]
    expiration_time: Option<String>,
    #[serde(rename = "userId")]
    user_id: Option<i64>,
    name: Option<String>,
    #[serde(rename = "hasLive")]
    has_live: Option<bool>,
}

#[derive(Debug, Clone)]
struct TokenBundle {
    access_token: String,
    md_access_token: String,
    expiration_time: Option<String>,
    user_id: Option<i64>,
    user_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub id: i64,
    pub name: String,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSuggestion {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketSnapshot {
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub bars: Vec<Bar>,
    pub history_loaded: usize,
    pub live_bars: usize,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub account_id: i64,
    pub account_name: String,
    pub balance: Option<f64>,
    pub cash_balance: Option<f64>,
    pub net_liq: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub intraday_margin: Option<f64>,
    pub open_position_qty: Option<f64>,
    pub raw_account: Option<Value>,
    pub raw_risk: Option<Value>,
    pub raw_cash: Option<Value>,
    pub raw_positions: Vec<Value>,
}

struct ServiceState {
    client: Client,
    session: Option<SessionState>,
    user_task: Option<JoinHandle<()>>,
    market_task: Option<JoinHandle<()>>,
}

struct SessionState {
    cfg: AppConfig,
    tokens: TokenBundle,
    accounts: Vec<AccountInfo>,
    user_store: UserSyncStore,
    selected_account_id: Option<i64>,
    selected_contract: Option<ContractSuggestion>,
}

#[derive(Default)]
struct UserSyncStore {
    accounts: BTreeMap<i64, Value>,
    risk: BTreeMap<i64, Value>,
    cash: BTreeMap<i64, Value>,
    positions: BTreeMap<i64, BTreeMap<i64, Value>>,
}

enum InternalEvent {
    UserEntities(Vec<EntityEnvelope>),
    UserSocketStatus(String),
    Market(MarketSnapshot),
    Error(String),
}

#[derive(Debug, Clone)]
struct EntityEnvelope {
    entity_type: String,
    deleted: bool,
    entity: Value,
}

struct LiveSeries {
    closed_bars: Vec<Bar>,
    forming_bar: Option<Bar>,
}

impl LiveSeries {
    fn new() -> Self {
        Self {
            closed_bars: Vec::new(),
            forming_bar: None,
        }
    }

    fn push_closed_bar(&mut self, bar: &Bar) {
        if let Some(last) = self.closed_bars.last_mut() {
            if bar.ts_ns == last.ts_ns {
                *last = bar.clone();
                return;
            }
            if bar.ts_ns < last.ts_ns {
                return;
            }
        }
        self.closed_bars.push(bar.clone());
    }

    fn render_bars(&self) -> Vec<Bar> {
        let mut out = self.closed_bars.clone();
        if let Some(forming) = &self.forming_bar {
            out.push(forming.clone());
        }
        out
    }
}

pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut state = ServiceState {
        client: Client::new(),
        session: None,
        user_task: None,
        market_task: None,
    };

    while let Some(next) = tokio::select! {
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
    } {
        match next {
            Either::Command(cmd) => {
                if let Err(err) =
                    handle_command(cmd, &mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::Internal(internal) => {
                if let Err(err) = handle_internal(internal, &mut state, &event_tx).await {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
        }
    }

    shutdown_state(&mut state, &event_tx);
}

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
}

async fn handle_command(
    cmd: ServiceCommand,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            shutdown_tasks(state);
            let _ = event_tx.send(ServiceEvent::Status(format!(
                "Authenticating against {}...",
                cfg.env.label()
            )));

            let tokens = authenticate(&state.client, &cfg).await?;
            save_token_cache(&cfg.session_cache_path, &tokens)?;

            let _ = event_tx.send(ServiceEvent::Connected {
                env: cfg.env,
                user_name: tokens.user_name.clone(),
                auth_mode: cfg.auth_mode,
            });

            let accounts = list_accounts(&state.client, &cfg.env, &tokens.access_token).await?;
            let mut user_store = UserSyncStore::default();
            seed_user_store(
                &state.client,
                &cfg.env,
                &tokens.access_token,
                &mut user_store,
            )
            .await;

            let selected_account_id = accounts.first().map(|account| account.id);
            let snapshots = user_store.build_snapshots(&accounts);

            let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts.clone()));
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));

            let account_ids = accounts
                .iter()
                .map(|account| account.id)
                .collect::<Vec<_>>();
            let user_cfg = cfg.clone();
            let user_tokens = tokens.clone();
            state.user_task = Some(tokio::spawn(user_sync_worker(
                user_cfg,
                user_tokens,
                account_ids,
                internal_tx.clone(),
            )));

            state.session = Some(SessionState {
                cfg,
                tokens,
                accounts,
                user_store,
                selected_account_id,
                selected_contract: None,
            });
        }
        ServiceCommand::SelectAccount { account_id } => {
            let Some(session) = state.session.as_mut() else {
                bail!("not connected");
            };
            session.selected_account_id = Some(account_id);
            let snapshots = session.user_store.build_snapshots(&session.accounts);
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
        }
        ServiceCommand::SearchContracts { query, limit } => {
            let Some(session) = state.session.as_ref() else {
                bail!("connect first");
            };
            let results = search_contracts(
                &state.client,
                &session.cfg.env,
                &session.tokens.access_token,
                &query,
                limit,
            )
            .await?;
            let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
        }
        ServiceCommand::SubscribeBars { contract } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let Some(task) = state.market_task.take() {
                task.abort();
            }
            session.selected_contract = Some(contract.clone());
            let cfg = session.cfg.clone();
            let token = session.tokens.md_access_token.clone();
            state.market_task = Some(tokio::spawn(market_data_worker(
                cfg,
                token,
                contract,
                internal_tx,
            )));
        }
    }
    Ok(())
}

async fn handle_internal(
    internal: InternalEvent,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::UserEntities(entities) => {
            let Some(session) = state.session.as_mut() else {
                return Ok(());
            };
            for envelope in entities {
                session.user_store.apply(envelope);
            }
            let snapshots = session.user_store.build_snapshots(&session.accounts);
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
        }
        InternalEvent::UserSocketStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::Market(snapshot) => {
            let _ = event_tx.send(ServiceEvent::MarketSnapshot(snapshot));
        }
        InternalEvent::Error(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
    }
    Ok(())
}

fn shutdown_state(state: &mut ServiceState, event_tx: &UnboundedSender<ServiceEvent>) {
    shutdown_tasks(state);
    state.session = None;
    let _ = event_tx.send(ServiceEvent::Disconnected);
}

fn shutdown_tasks(state: &mut ServiceState) {
    if let Some(task) = state.user_task.take() {
        task.abort();
    }
    if let Some(task) = state.market_task.take() {
        task.abort();
    }
}

async fn authenticate(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    if let Some(token) = empty_as_none(&cfg.token_override) {
        let user_name = fetch_auth_me(client, &cfg.env, token)
            .await
            .ok()
            .and_then(|value| {
                value
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            });
        return Ok(TokenBundle {
            access_token: token.to_string(),
            md_access_token: token.to_string(),
            expiration_time: None,
            user_id: None,
            user_name,
        });
    }

    match cfg.auth_mode {
        AuthMode::TokenFile => {
            let tokens = load_token_file(&cfg.token_path)
                .or_else(|_| load_token_file(&cfg.session_cache_path))
                .with_context(|| {
                    format!(
                        "load token from {} or {}",
                        cfg.token_path.display(),
                        cfg.session_cache_path.display()
                    )
                })?;
            let user_name = fetch_auth_me(client, &cfg.env, &tokens.access_token)
                .await
                .ok()
                .and_then(|value| {
                    value
                        .get("name")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .or(tokens.user_name.clone());
            Ok(TokenBundle {
                user_name,
                ..tokens
            })
        }
        AuthMode::Credentials => request_access_token(client, cfg).await,
    }
}

fn load_token_file(path: &Path) -> Result<TokenBundle> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse token JSON {}", path.display()))?;

    let access_token = parsed
        .get("token")
        .and_then(Value::as_str)
        .or_else(|| parsed.get("accessToken").and_then(Value::as_str))
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("token JSON missing token/accessToken")?;
    let md_access_token = parsed
        .get("mdAccessToken")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| access_token.clone());

    Ok(TokenBundle {
        access_token,
        md_access_token,
        expiration_time: parsed
            .get("expirationTime")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        user_id: parsed.get("userId").and_then(Value::as_i64),
        user_name: parsed
            .get("name")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    })
}

fn save_token_cache(path: &Path, tokens: &TokenBundle) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }

    let body = TokenCacheFile {
        token: tokens.access_token.clone(),
        access_token: Some(tokens.access_token.clone()),
        md_access_token: Some(tokens.md_access_token.clone()),
        expiration_time: tokens.expiration_time.clone(),
        user_id: tokens.user_id,
        name: tokens.user_name.clone(),
        has_live: None,
    };
    fs::write(path, serde_json::to_string_pretty(&body)?)
        .with_context(|| format!("write token cache {}", path.display()))?;
    Ok(())
}

async fn request_access_token(client: &Client, cfg: &AppConfig) -> Result<TokenBundle> {
    let url = format!("{}/auth/accesstokenrequest", cfg.env.rest_url());
    let payload = json!({
        "name": cfg.username,
        "password": cfg.password,
        "appId": empty_as_none(&cfg.app_id),
        "appVersion": empty_as_none(&cfg.app_version),
        "cid": empty_as_none(&cfg.cid),
        "sec": empty_as_none(&cfg.secret),
    });

    let response = client.post(url).json(&payload).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("auth request failed ({status}): {body}");
    }

    let parsed: AccessTokenResponse =
        serde_json::from_str(&body).context("parse access token response")?;
    if let Some(error_text) = parsed.error_text.as_deref() {
        if !error_text.trim().is_empty() {
            bail!("access token request rejected: {error_text}");
        }
    }

    let access_token = parsed
        .access_token
        .filter(|token| !token.trim().is_empty())
        .context("missing accessToken in auth response")?;
    let md_access_token = parsed
        .md_access_token
        .filter(|token| !token.trim().is_empty())
        .unwrap_or_else(|| access_token.clone());

    Ok(TokenBundle {
        access_token,
        md_access_token,
        expiration_time: parsed.expiration_time,
        user_id: parsed.user_id,
        user_name: parsed.name,
    })
}

async fn fetch_auth_me(client: &Client, env: &TradingEnvironment, token: &str) -> Result<Value> {
    let url = format!("{}/auth/me", env.rest_url());
    let response = client.get(url).bearer_auth(token).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("auth/me failed ({status}): {body}");
    }
    Ok(serde_json::from_str(&body)?)
}

async fn list_accounts(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
) -> Result<Vec<AccountInfo>> {
    let payload = fetch_entity_list(client, env, token, "account").await?;
    Ok(payload
        .into_iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_i64)?;
            let name = item.get("name").and_then(Value::as_str)?.to_string();
            Some(AccountInfo {
                id,
                name,
                raw: item,
            })
        })
        .collect())
}

async fn search_contracts(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<ContractSuggestion>> {
    let url = format!("{}/contract/suggest", env.rest_url());
    let response = client
        .get(url)
        .bearer_auth(token)
        .query(&[("t", query), ("l", &limit.to_string())])
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("contract/suggest failed ({status}): {body}");
    }
    let value: Value = serde_json::from_str(&body)?;
    let mut seen = HashMap::<i64, ()>::new();
    let mut out = Vec::new();
    if let Some(items) = value.as_array() {
        for item in items {
            let Some(id) = item.get("id").and_then(Value::as_i64) else {
                continue;
            };
            if seen.insert(id, ()).is_some() {
                continue;
            }
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("UNKNOWN")
                .to_string();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "contractMaturityId={}",
                        json_i64(item, "contractMaturityId").unwrap_or_default()
                    )
                });
            out.push(ContractSuggestion {
                id,
                name,
                description,
                raw: item.clone(),
            });
        }
    }
    Ok(out)
}

async fn fetch_entity_list(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    entity: &str,
) -> Result<Vec<Value>> {
    let url = format!("{}/{entity}/list", env.rest_url());
    let response = client.get(url).bearer_auth(token).send().await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{entity}/list failed ({status}): {body}");
    }
    let parsed: Value = serde_json::from_str(&body)?;
    Ok(match parsed {
        Value::Array(items) => items,
        Value::Object(_) => vec![parsed],
        _ => Vec::new(),
    })
}

async fn seed_user_store(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    store: &mut UserSyncStore,
) {
    for entity in ["account", "accountRiskStatus", "cashBalance", "position"] {
        let Ok(items) = fetch_entity_list(client, env, token, entity).await else {
            continue;
        };
        for item in items {
            store.apply(EntityEnvelope {
                entity_type: entity.to_string(),
                deleted: false,
                entity: item,
            });
        }
    }
}

async fn user_sync_worker(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) = user_sync_worker_inner(cfg, tokens, account_ids, internal_tx.clone()).await {
        let _ = internal_tx.send(InternalEvent::Error(format!("user sync: {err}")));
    }
}

async fn user_sync_worker_inner(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let (ws_stream, _) = tokio_tungstenite::connect_async(cfg.env.user_ws_url())
        .await
        .with_context(|| format!("connect {}", cfg.env.user_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let mut message_id = 1_u64;
    let authorize_id = message_id;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, tokens.access_token
        )))
        .await?;

    let mut sync_id = None;
    let mut authorized = false;
    let mut heartbeat = time::interval(Duration::from_millis(cfg.heartbeat_ms.max(250)));
    heartbeat.tick().await;

    loop {
        tokio::select! {
            _ = heartbeat.tick(), if authorized => {
                let _ = write.send(Message::Text("[]".to_string())).await;
            }
            next = read.next() => {
                let raw = match next {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => String::from_utf8_lossy(&bytes).to_string(),
                    Some(Ok(Message::Close(_))) => {
                        let _ = internal_tx.send(InternalEvent::UserSocketStatus("User-data websocket closed".to_string()));
                        break;
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(err)) => bail!("user websocket read error: {err}"),
                    None => break,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);

                    if !authorized && status == Some(200) && response_id == Some(authorize_id) {
                        authorized = true;
                        message_id += 1;
                        sync_id = Some(message_id);
                        let body = json!({
                            "splitResponses": true,
                            "accounts": account_ids,
                            "entityTypes": [
                                "account",
                                "accountRiskStatus",
                                "cashBalance",
                                "position",
                                "order",
                                "executionReport",
                                "fill"
                            ]
                        });
                        write
                            .send(Message::Text(create_message(
                                "user/syncrequest",
                                message_id,
                                Some(&body),
                            )))
                            .await?;
                        let _ = internal_tx.send(InternalEvent::UserSocketStatus("User sync authorized".to_string()));
                        continue;
                    }

                    if status == Some(200) && response_id == sync_id {
                        let envelopes = extract_entity_envelopes(&item);
                        if !envelopes.is_empty() {
                            let _ = internal_tx.send(InternalEvent::UserEntities(envelopes));
                        }
                        continue;
                    }

                    let envelopes = extract_entity_envelopes(&item);
                    if !envelopes.is_empty() {
                        let _ = internal_tx.send(InternalEvent::UserEntities(envelopes));
                    }
                }
            }
        }
    }

    Ok(())
}

async fn market_data_worker(
    cfg: AppConfig,
    access_token: String,
    contract: ContractSuggestion,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) =
        market_data_worker_inner(cfg, access_token, contract, internal_tx.clone()).await
    {
        let _ = internal_tx.send(InternalEvent::Error(format!("market data: {err}")));
    }
}

async fn market_data_worker_inner(
    cfg: AppConfig,
    access_token: String,
    contract: ContractSuggestion,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let (ws_stream, _) = tokio_tungstenite::connect_async(cfg.env.market_ws_url())
        .await
        .with_context(|| format!("connect {}", cfg.env.market_ws_url()))?;
    let (mut write, mut read) = ws_stream.split();

    let mut message_id = 1_u64;
    let authorize_id = message_id;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, access_token
        )))
        .await?;

    let mut chart_req_id = None;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut authorized = false;
    let mut series = LiveSeries::new();
    let mut live_bars = 0_usize;

    let mut heartbeat = time::interval(Duration::from_millis(cfg.heartbeat_ms.max(250)));
    heartbeat.tick().await;

    loop {
        tokio::select! {
            _ = heartbeat.tick(), if authorized => {
                let _ = write.send(Message::Text("[]".to_string())).await;
            }
            next = read.next() => {
                let raw = match next {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => String::from_utf8_lossy(&bytes).to_string(),
                    Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => continue,
                    Some(Err(err)) => bail!("market websocket read error: {err}"),
                    None => break,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);

                    if !authorized && status == Some(200) && response_id == Some(authorize_id) {
                        authorized = true;
                        message_id += 1;
                        chart_req_id = Some(message_id);
                        let body = json!({
                            "symbol": contract.name,
                            "chartDescription": {
                                "underlyingType": "MinuteBar",
                                "elementSize": 1,
                                "elementSizeUnit": "UnderlyingUnits",
                                "withHistogram": false
                            },
                            "timeRange": {
                                "asMuchAsElements": cfg.history_bars
                            }
                        });
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                message_id,
                                Some(&body),
                            )))
                            .await?;
                        continue;
                    }

                    if status == Some(200) && response_id == chart_req_id {
                        if let Some(d) = item.get("d") {
                            historical_id = d.get("historicalId").and_then(Value::as_i64).or(historical_id);
                            realtime_id = d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                    }

                    let Some(charts) = item
                        .get("d")
                        .and_then(|d| d.get("charts"))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };

                    for chart in charts {
                        let chart_id = chart.get("id").and_then(Value::as_i64);
                        let Some(bars) = chart.get("bars").and_then(Value::as_array) else {
                            continue;
                        };

                        for bar_json in bars {
                            let Some(bar) = parse_bar(bar_json) else {
                                continue;
                            };
                            let is_historical = chart_id.is_some()
                                && historical_id.is_some()
                                && chart_id == historical_id;
                            let is_realtime =
                                chart_id.is_some() && realtime_id.is_some() && chart_id == realtime_id;

                            if is_historical || (historical_id.is_none() && realtime_id.is_none()) {
                                series.push_closed_bar(&bar);
                                continue;
                            }

                            if is_realtime {
                                if let Some(current_ts) =
                                    series.forming_bar.as_ref().map(|current| current.ts_ns)
                                {
                                    if bar.ts_ns == current_ts {
                                        if let Some(current) = series.forming_bar.as_mut() {
                                            *current = bar;
                                        }
                                    } else if bar.ts_ns > current_ts {
                                        let closed =
                                            series.forming_bar.take().expect("forming bar exists");
                                        series.push_closed_bar(&closed);
                                        series.forming_bar = Some(bar);
                                        live_bars = live_bars.saturating_add(1);
                                    }
                                } else {
                                    series.forming_bar = Some(bar);
                                }
                            }
                        }
                    }

                    let _ = internal_tx.send(InternalEvent::Market(MarketSnapshot {
                        contract_id: Some(contract.id),
                        contract_name: Some(contract.name.clone()),
                        bars: series.render_bars(),
                        history_loaded: series.closed_bars.len(),
                        live_bars,
                        status: format!("Subscribed to 1m bars for {}", contract.name),
                    }));
                }
            }
        }
    }

    Ok(())
}

impl UserSyncStore {
    fn apply(&mut self, envelope: EntityEnvelope) {
        let entity_type = envelope.entity_type.to_ascii_lowercase();
        let Some(entity_id) = extract_entity_id(&envelope.entity) else {
            return;
        };

        match entity_type.as_str() {
            "account" => {
                if envelope.deleted {
                    self.accounts.remove(&entity_id);
                } else {
                    self.accounts.insert(entity_id, envelope.entity);
                }
            }
            "accountriskstatus" => {
                let Some(account_id) = extract_account_id("accountRiskStatus", &envelope.entity)
                else {
                    return;
                };
                if envelope.deleted {
                    self.risk.remove(&account_id);
                } else {
                    self.risk.insert(account_id, envelope.entity);
                }
            }
            "cashbalance" => {
                let Some(account_id) = extract_account_id("cashBalance", &envelope.entity) else {
                    return;
                };
                if envelope.deleted {
                    self.cash.remove(&account_id);
                } else {
                    self.cash.insert(account_id, envelope.entity);
                }
            }
            "position" => {
                let Some(account_id) = extract_account_id("position", &envelope.entity) else {
                    return;
                };
                let bucket = self.positions.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
                }
            }
            _ => {}
        }
    }

    fn build_snapshots(&self, accounts: &[AccountInfo]) -> Vec<AccountSnapshot> {
        accounts
            .iter()
            .map(|account| {
                let raw_account = self
                    .accounts
                    .get(&account.id)
                    .cloned()
                    .or_else(|| Some(account.raw.clone()));
                let raw_risk = self.risk.get(&account.id).cloned();
                let raw_cash = self.cash.get(&account.id).cloned();
                let raw_positions = self
                    .positions
                    .get(&account.id)
                    .map(|items| items.values().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();

                let balance = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "balance",
                                "netLiq",
                                "netLiquidationValue",
                                "netLiquidation",
                                "cashBalance",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_cash.as_ref().and_then(|value| {
                            pick_number(value, &["cashBalance", "totalCashValue", "amount"])
                        })
                    })
                    .or_else(|| {
                        raw_account
                            .as_ref()
                            .and_then(|value| pick_number(value, &["balance", "netLiq"]))
                    });
                let cash_balance = raw_cash.as_ref().and_then(|value| {
                    pick_number(
                        value,
                        &["cashBalance", "totalCashValue", "amount", "balance"],
                    )
                });
                let net_liq = raw_risk.as_ref().and_then(|value| {
                    pick_number(
                        value,
                        &[
                            "netLiq",
                            "netLiquidationValue",
                            "netLiquidation",
                            "balance",
                            "cashBalance",
                        ],
                    )
                });
                let intraday_margin = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "intradayMargin",
                                "dayMargin",
                                "dayTradeMargin",
                                "dayTradeMarginReq",
                                "marginRequirement",
                                "marginUsed",
                                "totalMargin",
                                "initialMarginReq",
                                "requiredIntradayMargin",
                                "initialMargin",
                                "maintenanceMargin",
                                "maintenanceMarginReq",
                                "marginReq",
                                "margin",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_account.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "intradayMargin",
                                    "dayTradeMargin",
                                    "dayTradeMarginReq",
                                    "initialMargin",
                                    "maintenanceMargin",
                                    "marginRequirement",
                                ],
                            )
                        })
                    });
                let unrealized_pnl = sum_position_metric(
                    &raw_positions,
                    &[
                        "unrealizedPnL",
                        "unrealizedPnl",
                        "floatingPnL",
                        "floatingPnl",
                        "openProfitAndLoss",
                        "netPnL",
                        "netPnl",
                        "openPnL",
                        "openPnl",
                    ],
                );
                let net_liq = net_liq.or_else(|| match (balance, unrealized_pnl) {
                    (Some(balance), Some(unrealized)) => Some(balance + unrealized),
                    _ => None,
                });
                let open_position_qty = sum_position_metric(
                    &raw_positions,
                    &["netPos", "netPosition", "qty", "quantity", "netQty"],
                );

                AccountSnapshot {
                    account_id: account.id,
                    account_name: account.name.clone(),
                    balance,
                    cash_balance,
                    net_liq,
                    unrealized_pnl,
                    intraday_margin,
                    open_position_qty,
                    raw_account,
                    raw_risk,
                    raw_cash,
                    raw_positions,
                }
            })
            .collect()
    }
}

fn pick_number(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| json_number(value, key))
}

fn sum_position_metric(positions: &[Value], keys: &[&str]) -> Option<f64> {
    let values = positions
        .iter()
        .filter_map(|position| pick_number(position, keys))
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum())
    }
}

fn extract_entity_envelopes(item: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if item.get("e").and_then(Value::as_str) == Some("props") {
        if let Some(d) = item.get("d") {
            let deleted = matches!(
                d.get("eventType").and_then(Value::as_str),
                Some("Deleted") | Some("deleted")
            );
            if let Some(entity_type) = d.get("entityType").and_then(Value::as_str) {
                if let Some(entity) = d.get("entity") {
                    out.push(EntityEnvelope {
                        entity_type: entity_type.to_string(),
                        deleted,
                        entity: entity.clone(),
                    });
                }
                if let Some(entities) = d.get("entities").and_then(Value::as_array) {
                    for entity in entities {
                        out.push(EntityEnvelope {
                            entity_type: entity_type.to_string(),
                            deleted,
                            entity: entity.clone(),
                        });
                    }
                }
            }
        }
    }

    if let Some(d) = item.get("d") {
        out.extend(extract_response_entities(d));
    }

    out
}

fn extract_response_entities(payload: &Value) -> Vec<EntityEnvelope> {
    let mut out = Vec::new();

    if let Some(items) = payload.as_array() {
        for item in items {
            out.extend(extract_response_entities(item));
        }
        return out;
    }

    let Some(obj) = payload.as_object() else {
        return out;
    };

    if let Some(entity_type) = obj.get("entityType").and_then(Value::as_str) {
        if let Some(entity) = obj.get("entity") {
            out.push(EntityEnvelope {
                entity_type: entity_type.to_string(),
                deleted: false,
                entity: entity.clone(),
            });
        }
        if let Some(entities) = obj.get("entities").and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: entity_type.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    for key in ["account", "accountRiskStatus", "cashBalance", "position"] {
        if let Some(entity) = obj.get(key) {
            if entity.is_object() {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
        let plural = format!("{key}s");
        if let Some(entities) = obj.get(&plural).and_then(Value::as_array) {
            for entity in entities {
                out.push(EntityEnvelope {
                    entity_type: key.to_string(),
                    deleted: false,
                    entity: entity.clone(),
                });
            }
        }
    }

    out
}

fn parse_status_code(msg: &Value) -> Option<i64> {
    if let Some(code) = msg.get("s").and_then(Value::as_i64) {
        return Some(code);
    }
    msg.get("s")
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<i64>().ok())
}

fn parse_frame(raw: &str) -> (char, Option<Value>) {
    let mut chars = raw.chars();
    let frame_type = chars.next().unwrap_or('\0');
    let offset = frame_type.len_utf8();
    let payload = raw.get(offset..).unwrap_or("");
    let value = if payload.is_empty() {
        None
    } else {
        serde_json::from_str(payload).ok()
    };
    (frame_type, value)
}

fn create_message(endpoint: &str, id: u64, body: Option<&Value>) -> String {
    if let Some(body) = body {
        format!("{endpoint}\n{id}\n\n{}", body)
    } else {
        format!("{endpoint}\n{id}\n\n")
    }
}

fn parse_bar(value: &Value) -> Option<Bar> {
    let ts = value.get("timestamp")?.as_str()?;
    let ts_ns = chrono::DateTime::parse_from_rfc3339(ts)
        .ok()?
        .with_timezone(&Utc)
        .timestamp_nanos_opt()?;
    Some(Bar {
        ts_ns,
        open: json_number(value, "open")?,
        high: json_number(value, "high")?,
        low: json_number(value, "low")?,
        close: json_number(value, "close")?,
    })
}

fn json_number(value: &Value, key: &str) -> Option<f64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_f64() {
        return Some(v);
    }
    if let Some(v) = raw.as_i64() {
        return Some(v as f64);
    }
    if let Some(v) = raw.as_u64() {
        return Some(v as f64);
    }
    raw.as_str().and_then(|text| text.parse::<f64>().ok())
}

fn json_i64(value: &Value, key: &str) -> Option<i64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_i64() {
        return Some(v);
    }
    if let Some(v) = raw.as_u64() {
        return i64::try_from(v).ok();
    }
    raw.as_str().and_then(|text| text.parse::<i64>().ok())
}

fn extract_entity_id(value: &Value) -> Option<i64> {
    json_i64(value, "id")
}

fn extract_account_id(entity_type: &str, value: &Value) -> Option<i64> {
    if entity_type.eq_ignore_ascii_case("account") {
        return json_i64(value, "id");
    }
    json_i64(value, "accountId")
        .or_else(|| {
            value
                .get("account")
                .and_then(|account| account.get("id"))
                .and_then(Value::as_i64)
        })
        .or_else(|| value.get("account").and_then(Value::as_i64))
        .or_else(|| json_i64(value, "id"))
}

fn empty_as_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

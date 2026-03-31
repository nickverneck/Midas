use crate::broker::{
    AccountInfo, AccountSnapshot, Bar, BarType, BrokerCapabilities, BrokerKind, ContractSuggestion,
    InstrumentSessionProfile, InstrumentSessionWindow, LatencySnapshot, ManualOrderAction,
    MarketSnapshot, ReplaySpeed, ServiceCommand, ServiceEvent, SessionKind, TradeMarker,
    TradeMarkerSide,
};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, ExecutionStrategyConfig, NativeReversalMode,
    NativeSignalTiming, NativeStrategyKind, StrategyKind,
};
use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::{
    mpsc::{UnboundedReceiver, UnboundedSender},
    watch,
};
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_BAR_LIMIT: usize = 256;
const PENDING_TARGET_WATCHDOG_SECS: u64 = 3;
const FOLLOWUP_REFRESH_DELAY_MS: u64 = 600;
const TRADE_MARKER_LIMIT: usize = 200;

pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: watch::Sender<MarketSnapshot>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut state = IronbeamState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        session: None,
        latency: LatencySnapshot::default(),
    };

    while let Some(next) = tokio::select! {
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
    } {
        let result = match next {
            Either::Command(cmd) => {
                handle_command(cmd, &mut state, &event_tx, &market_tx, internal_tx.clone()).await
            }
            Either::Internal(internal) => {
                handle_internal(
                    internal,
                    &mut state,
                    &event_tx,
                    &market_tx,
                    internal_tx.clone(),
                )
                .await
            }
        };
        if let Err(err) = result {
            let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
        }
    }

    shutdown_session(state.session.as_mut(), &market_tx);
}

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
}

enum InternalEvent {
    StreamPayload(String),
    StreamStatus(String),
    StreamError(String),
    RefreshAccountState { reason: Option<String> },
}

struct IronbeamState {
    client: Client,
    session: Option<IronbeamSession>,
    latency: LatencySnapshot,
}

struct IronbeamSession {
    cfg: AppConfig,
    token: String,
    user_name: Option<String>,
    accounts: Vec<AccountInfo>,
    selected_account_id: Option<i64>,
    selected_contract: Option<ContractSuggestion>,
    account_state: AccountState,
    account_snapshots: Vec<AccountSnapshot>,
    execution_config: ExecutionStrategyConfig,
    execution_runtime: ExecutionRuntimeState,
    managed_protection: BTreeMap<ProtectionKey, ManagedProtectionOrders>,
    market: MarketSnapshot,
    market_task: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IronbeamTokenCache {
    token: String,
    #[serde(rename = "accessToken", skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

struct AuthResult {
    token: String,
    user_name: Option<String>,
}

#[derive(Debug, Clone)]
struct PendingNativeReversalEntry {
    target_qty: i32,
    reason: String,
}

#[derive(Debug, Clone, Default)]
struct ExecutionRuntimeState {
    armed: bool,
    last_closed_bar_ts: Option<i64>,
    pending_target_qty: Option<i32>,
    pending_target_started_at: Option<Instant>,
    pending_reversal_entry: Option<PendingNativeReversalEntry>,
    last_summary: String,
    hma_execution: HmaAngleExecutionState,
    ema_execution: EmaCrossExecutionState,
}

impl ExecutionRuntimeState {
    fn snapshot(&self) -> ExecutionRuntimeSnapshot {
        ExecutionRuntimeSnapshot {
            armed: self.armed,
            last_closed_bar_ts: self.last_closed_bar_ts,
            pending_target_qty: self.pending_target_qty,
            last_summary: self.last_summary.clone(),
        }
    }

    fn reset_execution(&mut self) {
        self.pending_reversal_entry = None;
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
    }

    fn set_pending_target(&mut self, target_qty: Option<i32>) {
        self.pending_target_qty = target_qty;
        self.pending_target_started_at = target_qty.map(|_| Instant::now());
    }

    fn clear_pending_target(&mut self) {
        self.pending_target_qty = None;
        self.pending_target_started_at = None;
    }
}

#[derive(Debug, Clone, Default)]
struct AccountState {
    balances: BTreeMap<String, Value>,
    positions: BTreeMap<String, BTreeMap<String, Value>>,
    risks: BTreeMap<String, Value>,
    orders: BTreeMap<String, BTreeMap<String, Value>>,
    fills: BTreeMap<String, BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ProtectionKey {
    account_id: i64,
    contract_id: i64,
}

#[derive(Debug, Clone)]
struct ManagedProtectionOrders {
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    last_requested_take_profit_price: Option<f64>,
    last_requested_stop_price: Option<f64>,
    take_profit_order_id: Option<String>,
    stop_order_id: Option<String>,
}

#[derive(Debug, Clone)]
struct ProtectionOrderCandidate {
    order_id: String,
    price: Option<f64>,
    quantity: Option<i32>,
}

enum OrderDispatchOutcome {
    NoOp { message: String },
    Queued { target_qty: Option<i32> },
}

struct OrderContext<'a> {
    account: &'a AccountInfo,
    contract: &'a ContractSuggestion,
}

enum ProtectionSyncAction {
    None,
    Replace {
        take_profit_price: Option<f64>,
        stop_price: Option<f64>,
    },
    Clear,
}

#[derive(Clone)]
struct AccountRefresh {
    balances: Vec<Value>,
    positions: Vec<Value>,
    risks: Vec<Value>,
    orders: BTreeMap<String, Vec<Value>>,
    fills: BTreeMap<String, Vec<Value>>,
}

async fn handle_command(
    cmd: ServiceCommand,
    state: &mut IronbeamState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            shutdown_session(state.session.as_mut(), market_tx);
            state.session = None;
            state.latency = LatencySnapshot::default();
            let _ = market_tx.send(MarketSnapshot::default());
            let _ = event_tx.send(ServiceEvent::Status(format!(
                "Authenticating against Ironbeam {}...",
                cfg.env.label()
            )));

            let auth = authenticate(&state.client, &cfg, &mut state.latency).await?;
            save_token_cache(&cfg.session_cache_path, &auth)?;
            let accounts =
                list_accounts(&state.client, &cfg, &auth.token, &mut state.latency).await?;
            let selected_account_id = accounts.first().map(|account| account.id);

            let mut session = IronbeamSession {
                cfg: cfg.clone(),
                token: auth.token,
                user_name: auth.user_name.clone(),
                accounts: accounts.clone(),
                selected_account_id,
                selected_contract: None,
                account_state: AccountState::default(),
                account_snapshots: Vec::new(),
                execution_config: ExecutionStrategyConfig::default(),
                execution_runtime: ExecutionRuntimeState::default(),
                managed_protection: BTreeMap::new(),
                market: MarketSnapshot::default(),
                market_task: None,
            };
            refresh_account_state(&state.client, &mut session, &mut state.latency).await?;
            emit_account_snapshots(event_tx, &session);
            state.session = Some(session);

            let _ = event_tx.send(ServiceEvent::Connected {
                broker: BrokerKind::Ironbeam,
                env: cfg.env,
                user_name: auth.user_name,
                auth_mode: cfg.auth_mode,
                session_kind: SessionKind::Live,
                capabilities: ironbeam_capabilities(),
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            if let Some(session) = state.session.as_ref() {
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::EnterReplayMode { .. } => {
            let _ = event_tx.send(ServiceEvent::Error(
                "Ironbeam replay is not available in this build.".to_string(),
            ));
        }
        ServiceCommand::ReplayState => {
            let Some(session) = state.session.as_ref() else {
                let _ = event_tx.send(ServiceEvent::Disconnected);
                return Ok(());
            };
            let _ = event_tx.send(ServiceEvent::Connected {
                broker: BrokerKind::Ironbeam,
                env: session.cfg.env,
                user_name: session.user_name.clone(),
                auth_mode: session.cfg.auth_mode,
                session_kind: SessionKind::Live,
                capabilities: ironbeam_capabilities(),
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(session.accounts.clone()));
            emit_account_snapshots(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::SelectAccount { account_id } => {
            let session = require_session_mut(state.session.as_mut())?;
            if !session
                .accounts
                .iter()
                .any(|account| account.id == account_id)
            {
                bail!("unknown Ironbeam account id `{account_id}`");
            }
            session.selected_account_id = Some(account_id);
            handle_execution_account_sync(&state.client, session, &mut state.latency, internal_tx)
                .await?;
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SearchContracts { query, limit } => {
            let session = require_session(state.session.as_ref())?;
            let trimmed = query.trim();
            if trimmed.len() < 3 {
                let _ = event_tx.send(ServiceEvent::Status(
                    "Ironbeam symbol search expects at least 3 characters.".to_string(),
                ));
                let _ = event_tx.send(ServiceEvent::ContractSearchResults {
                    query,
                    results: Vec::new(),
                });
                return Ok(());
            }
            let results = search_contracts(
                &state.client,
                &session.cfg,
                &session.token,
                trimmed,
                limit,
                &mut state.latency,
            )
            .await?;
            let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            let session = require_session_mut(state.session.as_mut())?;
            if bar_type != BarType::Minute1 {
                let _ = event_tx.send(ServiceEvent::Status(
                    "Ironbeam currently supports 1-minute bars only; using 1 Min.".to_string(),
                ));
            }

            shutdown_market_task(session, market_tx);
            let stream_id = create_stream_id(
                &state.client,
                &session.cfg,
                &session.token,
                &mut state.latency,
            )
            .await?;
            subscribe_time_bars(
                &state.client,
                &session.cfg,
                &session.token,
                &stream_id,
                &contract,
                &mut state.latency,
            )
            .await?;

            session.selected_contract = Some(contract.clone());
            session.execution_runtime.clear_pending_target();
            session.execution_runtime.last_closed_bar_ts = None;
            session.execution_runtime.reset_execution();
            session.execution_runtime.last_summary =
                "Selected contract changed; waiting for Ironbeam bars.".to_string();
            session.market = MarketSnapshot {
                contract_id: Some(contract.id),
                contract_name: Some(contract.name.clone()),
                bars: Vec::new(),
                trade_markers: selected_trade_markers(&session.account_state, &contract),
                session_profile: Some(contract_session_profile(&contract)),
                value_per_point: pick_number(
                    &contract.raw,
                    &["pipValue", "pointValue", "pointValueUsd", "valuePerPoint"],
                ),
                tick_size: pick_number(&contract.raw, &["minTick", "pipSize", "tickSize"]),
                history_loaded: 0,
                live_bars: 0,
                status: format!("Subscribing to {} 1-minute bars...", contract.name),
            };
            refresh_managed_protection(session);
            rebuild_account_snapshots(session);
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = market_tx.send(session.market.clone());
            let bar_limit = market_bar_limit(session);
            session.market_task = Some(tokio::spawn(run_market_stream(
                session.cfg.env,
                stream_id,
                session.token.clone(),
                contract,
                internal_tx,
                bar_limit,
            )));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetReplaySpeed { .. } => {
            let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(ReplaySpeed::default()));
        }
        ServiceCommand::ManualOrder { action } => {
            let session = require_session_mut(state.session.as_mut())?;
            match dispatch_manual_order(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                action,
            )
            .await?
            {
                OrderDispatchOutcome::NoOp { message } => {
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                OrderDispatchOutcome::Queued { target_qty } => {
                    if let Some(target_qty) = target_qty {
                        session
                            .execution_runtime
                            .set_pending_target(Some(target_qty));
                        session.execution_runtime.last_summary =
                            "Manual close requested; waiting for Ironbeam fill.".to_string();
                    }
                    emit_execution_state(event_tx, session);
                }
            }
            emit_account_snapshots(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetTargetPosition {
            target_qty,
            automated,
            reason,
        } => {
            let session = require_session_mut(state.session.as_mut())?;
            match dispatch_target_position_order(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                target_qty,
                automated,
                &reason,
            )
            .await?
            {
                OrderDispatchOutcome::NoOp { message } => {
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                OrderDispatchOutcome::Queued { target_qty } => {
                    session.execution_runtime.set_pending_target(target_qty);
                }
            }
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
        } => {
            let session = require_session_mut(state.session.as_mut())?;
            sync_native_protection(
                &state.client,
                session,
                &mut state.latency,
                internal_tx,
                signed_qty,
                take_profit_price,
                stop_price,
                &reason,
            )
            .await?;
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            let session = require_session_mut(state.session.as_mut())?;
            if session.execution_config != config {
                session.execution_config = config;
                if session.execution_runtime.armed {
                    disarm_execution_strategy(
                        session,
                        "Native strategy config changed; press Continue to re-arm.".to_string(),
                    );
                }
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::ArmExecutionStrategy => {
            let session = require_session_mut(state.session.as_mut())?;
            arm_execution_strategy(session);
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            let session = require_session_mut(state.session.as_mut())?;
            disarm_execution_strategy(session, reason);
            emit_execution_state(event_tx, session);
        }
    }

    Ok(())
}

async fn handle_internal(
    internal: InternalEvent,
    state: &mut IronbeamState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::StreamStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::StreamError(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
        InternalEvent::RefreshAccountState { reason } => {
            let Some(session) = state.session.as_mut() else {
                return Ok(());
            };
            if let Some(reason) = reason {
                let _ = event_tx.send(ServiceEvent::Status(reason));
            }
            refresh_account_state(&state.client, session, &mut state.latency).await?;
            handle_execution_account_sync(
                &state.client,
                session,
                &mut state.latency,
                internal_tx.clone(),
            )
            .await?;
            let _ = market_tx.send(session.market.clone());
            emit_account_snapshots(event_tx, session);
            emit_execution_state(event_tx, session);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::StreamPayload(raw) => {
            let Ok(parsed) = serde_json::from_str::<Value>(&raw) else {
                return Ok(());
            };
            let Some(session) = state.session.as_mut() else {
                return Ok(());
            };
            let stream_effect = apply_stream_payload(session, &parsed);
            if let Some(reset) = parsed.get("r") {
                let message = pick_str(reset, &["message", "m"]).unwrap_or("stream reset");
                let _ = event_tx.send(ServiceEvent::Status(format!(
                    "Ironbeam stream reset: {message}"
                )));
                schedule_refresh(
                    internal_tx.clone(),
                    Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
                    Some("Refreshing Ironbeam account state after reset.".to_string()),
                );
            }

            if stream_effect.account_changed {
                handle_execution_account_sync(
                    &state.client,
                    session,
                    &mut state.latency,
                    internal_tx.clone(),
                )
                .await?;
                emit_account_snapshots(event_tx, session);
                emit_execution_state(event_tx, session);
            } else if stream_effect.market_changed {
                maybe_run_execution_strategy(
                    &state.client,
                    session,
                    &mut state.latency,
                    internal_tx.clone(),
                    event_tx,
                )
                .await?;
                emit_execution_state(event_tx, session);
            }

            if stream_effect.market_changed || stream_effect.account_changed {
                let _ = market_tx.send(session.market.clone());
            }

            if stream_effect.refresh_recommended {
                schedule_refresh(
                    internal_tx,
                    Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
                    None,
                );
            }
        }
    }

    Ok(())
}

fn ironbeam_capabilities() -> BrokerCapabilities {
    BrokerCapabilities {
        replay: false,
        manual_orders: true,
        automated_orders: true,
        native_protection: true,
    }
}

fn require_session(session: Option<&IronbeamSession>) -> Result<&IronbeamSession> {
    session.context("Connect to Ironbeam first.")
}

fn require_session_mut(session: Option<&mut IronbeamSession>) -> Result<&mut IronbeamSession> {
    session.context("Connect to Ironbeam first.")
}

fn shutdown_session(
    session: Option<&mut IronbeamSession>,
    market_tx: &watch::Sender<MarketSnapshot>,
) {
    if let Some(session) = session {
        shutdown_market_task(session, market_tx);
    }
}

fn shutdown_market_task(session: &mut IronbeamSession, market_tx: &watch::Sender<MarketSnapshot>) {
    if let Some(task) = session.market_task.take() {
        task.abort();
    }
    session.market = MarketSnapshot::default();
    let _ = market_tx.send(MarketSnapshot::default());
}

fn emit_account_snapshots(event_tx: &UnboundedSender<ServiceEvent>, session: &IronbeamSession) {
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(
        session.account_snapshots.clone(),
    ));
}

fn emit_execution_state(event_tx: &UnboundedSender<ServiceEvent>, session: &IronbeamSession) {
    let (take_profit_price, stop_price) = selected_protection_prices(session);
    let snapshot = ExecutionStateSnapshot {
        config: session.execution_config.clone(),
        runtime: session.execution_runtime.snapshot(),
        selected_account_id: session.selected_account_id,
        selected_contract_name: session
            .selected_contract
            .as_ref()
            .map(|contract| contract.name.clone()),
        market_position_qty: selected_market_position_qty(session),
        market_entry_price: selected_market_entry_price(session),
        selected_contract_take_profit_price: take_profit_price,
        selected_contract_stop_price: stop_price,
    };
    let _ = event_tx.send(ServiceEvent::ExecutionState(snapshot));
}

async fn authenticate(
    client: &Client,
    cfg: &AppConfig,
    latency: &mut LatencySnapshot,
) -> Result<AuthResult> {
    if let Some(token) = empty_as_none(&cfg.token_override) {
        latency.rest_rtt_ms = Some(0);
        return Ok(AuthResult {
            token: token.to_string(),
            user_name: empty_as_none(&cfg.username).map(ToString::to_string),
        });
    }

    match cfg.auth_mode {
        AuthMode::TokenFile => load_token_file(&cfg.token_path)
            .or_else(|_| load_token_file(&cfg.session_cache_path))
            .with_context(|| {
                format!(
                    "load token from {} or {}",
                    cfg.token_path.display(),
                    cfg.session_cache_path.display()
                )
            }),
        AuthMode::Credentials => request_access_token(client, cfg, latency).await,
    }
}

fn load_token_file(path: &Path) -> Result<AuthResult> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read token file {}", path.display()))?;
    let parsed: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse token JSON {}", path.display()))?;
    let token = pick_str(&parsed, &["token", "accessToken"])
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("token JSON missing token/accessToken")?;
    Ok(AuthResult {
        token,
        user_name: pick_str(&parsed, &["name"]).map(ToString::to_string),
    })
}

fn save_token_cache(path: &Path, auth: &AuthResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let body = IronbeamTokenCache {
        token: auth.token.clone(),
        access_token: Some(auth.token.clone()),
        name: auth.user_name.clone(),
    };
    fs::write(path, serde_json::to_string_pretty(&body)?)
        .with_context(|| format!("write token cache {}", path.display()))?;
    Ok(())
}

async fn request_access_token(
    client: &Client,
    cfg: &AppConfig,
    latency: &mut LatencySnapshot,
) -> Result<AuthResult> {
    let payload = json!({
        "username": cfg.username,
        "password": cfg.password,
        "apikey": empty_as_none(&cfg.api_key),
    });
    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!("{}/auth", ironbeam_rest_url(cfg.env)))
            .json(&payload),
        "Ironbeam auth",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    let token = pick_str(&parsed, &["token", "accessToken"])
        .map(ToString::to_string)
        .filter(|token| !token.trim().is_empty())
        .context("missing token in Ironbeam auth response")?;
    Ok(AuthResult {
        token,
        user_name: empty_as_none(&cfg.username).map(ToString::to_string),
    })
}

async fn list_accounts(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    latency: &mut LatencySnapshot,
) -> Result<Vec<AccountInfo>> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllAccounts",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token),
        "Ironbeam account list",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(parsed
        .get("accounts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|account_id| AccountInfo {
            id: stable_id(account_id),
            name: account_id.to_string(),
            raw: json!({ "accountId": account_id }),
        })
        .collect())
}

async fn search_contracts(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    query: &str,
    limit: usize,
    latency: &mut LatencySnapshot,
) -> Result<Vec<ContractSuggestion>> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!("{}/info/symbols", ironbeam_rest_url(cfg.env)))
            .bearer_auth(token)
            .query(&[
                ("text", query.to_string()),
                ("limit", limit.min(1000).max(1).to_string()),
                ("preferActive", "true".to_string()),
            ]),
        "Ironbeam symbol search",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(parsed
        .get("symbols")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let symbol = pick_str(item, &["symbol", "exchSym"])?.trim();
            if symbol.is_empty() {
                return None;
            }
            let description =
                pick_str(item, &["description", "symbolType", "contractText"]).unwrap_or(symbol);
            Some(ContractSuggestion {
                id: stable_id(symbol),
                name: symbol.to_string(),
                description: description.to_string(),
                raw: item.clone(),
            })
        })
        .collect())
}

async fn refresh_account_state(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let refresh = fetch_account_refresh(client, session, latency).await?;
    session.account_state = build_account_state(&refresh);
    refresh_managed_protection(session);
    rebuild_account_snapshots(session);
    session.market.trade_markers = session
        .selected_contract
        .as_ref()
        .map(|contract| selected_trade_markers(&session.account_state, contract))
        .unwrap_or_default();
    Ok(())
}

async fn fetch_account_refresh(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
) -> Result<AccountRefresh> {
    let (balances, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllBalances",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token)
            .query(&[("balanceType", "CURRENT_OPEN")]),
        "Ironbeam balances",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let (positions, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllPositions",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token),
        "Ironbeam positions",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let (risks, elapsed_ms) = request_json(
        client
            .get(format!(
                "{}/account/getAllRiskInfo",
                ironbeam_rest_url(session.cfg.env)
            ))
            .bearer_auth(&session.token),
        "Ironbeam risk info",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);

    let mut orders = BTreeMap::new();
    let mut fills = BTreeMap::new();
    for account in &session.accounts {
        let (parsed_orders, elapsed_ms) = request_json(
            client
                .get(format!(
                    "{}/order/{}/ANY",
                    ironbeam_rest_url(session.cfg.env),
                    account.name
                ))
                .bearer_auth(&session.token),
            &format!("Ironbeam orders for {}", account.name),
        )
        .await?;
        latency.rest_rtt_ms = Some(elapsed_ms);
        orders.insert(
            account.name.clone(),
            parsed_orders
                .get("orders")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        );

        let (parsed_fills, elapsed_ms) = request_json(
            client
                .get(format!(
                    "{}/order/{}/fills",
                    ironbeam_rest_url(session.cfg.env),
                    account.name
                ))
                .bearer_auth(&session.token),
            &format!("Ironbeam fills for {}", account.name),
        )
        .await?;
        latency.rest_rtt_ms = Some(elapsed_ms);
        fills.insert(
            account.name.clone(),
            parsed_fills
                .get("fills")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        );
    }

    Ok(AccountRefresh {
        balances: balances
            .get("balances")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        positions: positions
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        risks: risks
            .get("risks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        orders,
        fills,
    })
}

fn build_account_state(refresh: &AccountRefresh) -> AccountState {
    let mut state = AccountState::default();

    for balance in &refresh.balances {
        if let Some(account_id) = account_id_string(balance) {
            state.balances.insert(account_id, balance.clone());
        }
    }

    for risk in &refresh.risks {
        if let Some(account_id) = account_id_string(risk) {
            state.risks.insert(account_id, risk.clone());
        }
    }

    for item in &refresh.positions {
        let Some(account_id) = account_id_string(item) else {
            continue;
        };
        let positions = item
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let entries = positions
            .into_iter()
            .filter_map(|position| Some((position_key(&position)?, position)))
            .collect::<BTreeMap<_, _>>();
        state.positions.insert(account_id, entries);
    }

    for (account_id, orders) in &refresh.orders {
        let entries = orders
            .iter()
            .filter_map(|order| Some((order_id_string(order)?, order.clone())))
            .collect::<BTreeMap<_, _>>();
        state.orders.insert(account_id.clone(), entries);
    }

    for (account_id, fills) in &refresh.fills {
        let entries = fills
            .iter()
            .filter_map(|fill| Some((fill_key(fill)?, fill.clone())))
            .collect::<BTreeMap<_, _>>();
        state.fills.insert(account_id.clone(), entries);
    }

    state
}

fn rebuild_account_snapshots(session: &mut IronbeamSession) {
    let selected_symbol = session.selected_contract.as_ref().map(contract_symbol);
    let selected_contract_id = session
        .selected_contract
        .as_ref()
        .map(|contract| contract.id);
    session.account_snapshots = session
        .accounts
        .iter()
        .map(|account| {
            let balance = session.account_state.balances.get(&account.name).cloned();
            let risk = session.account_state.risks.get(&account.name).cloned();
            let raw_positions = session
                .account_state
                .positions
                .get(&account.name)
                .map(|positions| positions.values().cloned().collect::<Vec<_>>())
                .unwrap_or_default();

            let selected_position = selected_symbol.and_then(|symbol| {
                raw_positions
                    .iter()
                    .find(|position| position_symbol(position) == Some(symbol))
            });
            let open_position_qty = raw_positions
                .iter()
                .filter_map(signed_position_qty)
                .map(f64::abs)
                .sum::<f64>();
            let unrealized_pnl = selected_position
                .and_then(|position| pick_number(position, &["unrealizedPL", "unrealizedPnl"]))
                .or_else(|| {
                    balance
                        .as_ref()
                        .and_then(|item| pick_number(item, &["openTradeEquity"]))
                });

            let protection = selected_contract_id.and_then(|contract_id| {
                session.managed_protection.get(&ProtectionKey {
                    account_id: account.id,
                    contract_id,
                })
            });

            AccountSnapshot {
                account_id: account.id,
                account_name: account.name.clone(),
                balance: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["totalEquity", "balance"]))
                    .or_else(|| {
                        balance
                            .as_ref()
                            .and_then(|item| pick_number(item, &["netLiquidity"]))
                    })
                    .or_else(|| {
                        risk.as_ref()
                            .and_then(|item| pick_number(item, &["currentNetLiquidationValue"]))
                    }),
                cash_balance: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["cashBalance"])),
                net_liq: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["netLiquidity"]))
                    .or_else(|| {
                        risk.as_ref()
                            .and_then(|item| pick_number(item, &["currentNetLiquidationValue"]))
                    }),
                realized_pnl: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["realizedPL", "realizedPnl"])),
                unrealized_pnl,
                intraday_margin: balance
                    .as_ref()
                    .and_then(|item| item.get("marginInfo"))
                    .and_then(|item| {
                        pick_number(item, &["initialTotalMargin", "maintenanceTotalMargin"])
                    }),
                open_position_qty: (open_position_qty > 0.0).then_some(open_position_qty),
                market_position_qty: selected_position.and_then(signed_position_qty),
                market_entry_price: selected_position.and_then(position_entry_price),
                selected_contract_take_profit_price: protection.and_then(|state| {
                    state
                        .take_profit_price
                        .or(state.last_requested_take_profit_price)
                }),
                selected_contract_stop_price: protection
                    .and_then(|state| state.stop_price.or(state.last_requested_stop_price)),
                raw_account: balance.clone(),
                raw_risk: risk.clone(),
                raw_cash: balance,
                raw_positions,
            }
        })
        .collect();
}

async fn create_stream_id(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    latency: &mut LatencySnapshot,
) -> Result<String> {
    let (parsed, elapsed_ms) = request_json(
        client
            .get(format!("{}/stream/create", ironbeam_rest_url(cfg.env)))
            .bearer_auth(token),
        "Ironbeam stream creation",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    pick_str(&parsed, &["streamId"])
        .map(ToString::to_string)
        .filter(|stream_id| !stream_id.trim().is_empty())
        .context("missing streamId in Ironbeam stream creation response")
}

async fn subscribe_time_bars(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    stream_id: &str,
    contract: &ContractSuggestion,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let payload = json!({
        "symbol": contract_symbol(contract),
        "period": 1,
        "barType": "MINUTE",
        "loadSize": cfg.history_bars.max(1),
    });
    let (_, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/indicator/{stream_id}/timeBars/subscribe",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam time bars subscription",
    )
    .await?;
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

async fn run_market_stream(
    env: TradingEnvironment,
    stream_id: String,
    token: String,
    contract: ContractSuggestion,
    internal_tx: UnboundedSender<InternalEvent>,
    _bar_limit: usize,
) {
    let ws_url = format!(
        "{}/stream/{stream_id}?token={token}",
        ironbeam_ws_base_url(env)
    );

    let Ok((mut socket, _)) = connect_async(&ws_url).await else {
        let _ = internal_tx.send(InternalEvent::StreamError(format!(
            "Failed to open Ironbeam stream for {}.",
            contract.name
        )));
        return;
    };

    let _ = internal_tx.send(InternalEvent::StreamStatus(format!(
        "Ironbeam market stream open for {}.",
        contract.name
    )));

    while let Some(frame) = socket.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let _ = internal_tx.send(InternalEvent::StreamPayload(text.to_string()));
            }
            Ok(Message::Binary(bytes)) => {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    let _ = internal_tx.send(InternalEvent::StreamPayload(text));
                }
            }
            Ok(Message::Close(frame)) => {
                let reason = frame
                    .as_ref()
                    .map(|frame| frame.reason.to_string())
                    .filter(|reason| !reason.trim().is_empty())
                    .unwrap_or_else(|| "no reason provided".to_string());
                let _ = internal_tx.send(InternalEvent::StreamStatus(format!(
                    "Ironbeam market stream closed for {}: {}",
                    contract.name, reason
                )));
                break;
            }
            Ok(Message::Ping(_) | Message::Pong(_)) => {}
            Ok(Message::Frame(_)) => {}
            Err(err) => {
                let _ = internal_tx.send(InternalEvent::StreamError(format!(
                    "Ironbeam market stream error for {}: {}",
                    contract.name, err
                )));
                break;
            }
        }
    }
}

struct StreamEffect {
    market_changed: bool,
    account_changed: bool,
    refresh_recommended: bool,
}

fn apply_stream_payload(session: &mut IronbeamSession, parsed: &Value) -> StreamEffect {
    let mut market_changed = false;
    let mut account_changed = false;
    let mut refresh_recommended = false;

    if let Some(items) = parsed.get("ti").and_then(Value::as_array) {
        let incoming = parse_time_bars(items);
        if !incoming.is_empty() {
            let bar_limit = market_bar_limit(session);
            let appended = merge_bars(&mut session.market.bars, incoming, bar_limit);
            if session.market.history_loaded == 0 && session.market.live_bars == 0 {
                session.market.history_loaded = session.market.bars.len();
                session.market.live_bars = 0;
            } else if appended > 0 {
                session.market.live_bars = session.market.live_bars.saturating_add(appended);
            } else {
                session.market.live_bars = session.market.live_bars.saturating_add(1);
            }
            session.market.status = format!(
                "Streaming {} 1-minute bars from Ironbeam",
                session.market.contract_name.as_deref().unwrap_or("market")
            );
            market_changed = true;
        }
    }

    if apply_stream_balances(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_positions(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_risks(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_orders(&mut session.account_state, parsed) {
        account_changed = true;
        refresh_recommended = true;
    }
    if apply_stream_fills(&mut session.account_state, parsed) {
        account_changed = true;
        refresh_recommended = true;
    }

    if account_changed {
        refresh_managed_protection(session);
        rebuild_account_snapshots(session);
        if let Some(contract) = session.selected_contract.as_ref() {
            session.market.trade_markers = selected_trade_markers(&session.account_state, contract);
        }
    }

    StreamEffect {
        market_changed,
        account_changed,
        refresh_recommended,
    }
}

fn apply_stream_balances(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("ba").and_then(Value::as_array) {
        for item in items {
            if let Some(account_id) = account_id_string(item) {
                state.balances.insert(account_id, item.clone());
                changed = true;
            }
        }
    }
    if let Some(item) = parsed.get("b") {
        for balance in value_items(item) {
            if let Some(account_id) = account_id_string(balance) {
                state.balances.insert(account_id, balance.clone());
                changed = true;
            }
        }
    }
    changed
}

fn apply_stream_positions(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(snapshot) = parsed.get("psa") {
        for (account_id, positions, replace) in position_update_groups(snapshot) {
            let entry = state.positions.entry(account_id).or_default();
            if replace {
                entry.clear();
            }
            for position in positions {
                if let Some(key) = position_key(&position) {
                    entry.insert(key, position);
                    changed = true;
                }
            }
        }
    }
    if let Some(update) = parsed.get("ps") {
        for (account_id, positions, _) in position_update_groups(update) {
            let entry = state.positions.entry(account_id).or_default();
            for position in positions {
                if let Some(key) = position_key(&position) {
                    if signed_position_qty(&position)
                        .map(|qty| qty.abs() <= f64::EPSILON)
                        .unwrap_or(false)
                    {
                        entry.remove(&key);
                    } else {
                        entry.insert(key, position);
                    }
                    changed = true;
                }
            }
        }
    }
    changed
}

fn apply_stream_risks(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("ria").and_then(Value::as_array) {
        for item in items {
            if let Some(account_id) = account_id_string(item) {
                state.risks.insert(account_id, item.clone());
                changed = true;
            }
        }
    }
    if let Some(item) = parsed.get("ri") {
        for risk in value_items(item) {
            if let Some(account_id) = account_id_string(risk) {
                state.risks.insert(account_id, risk.clone());
                changed = true;
            }
        }
    }
    changed
}

fn apply_stream_orders(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("o") {
        for order in value_items(items) {
            let Some(account_id) = account_id_string(order) else {
                continue;
            };
            let Some(order_id) = order_id_string(order) else {
                continue;
            };
            state
                .orders
                .entry(account_id)
                .or_default()
                .insert(order_id, order.clone());
            changed = true;
        }
    }
    changed
}

fn apply_stream_fills(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("f") {
        for fill in value_items(items) {
            let Some(account_id) = account_id_string(fill) else {
                continue;
            };
            let Some(key) = fill_key(fill) else {
                continue;
            };
            state
                .fills
                .entry(account_id)
                .or_default()
                .insert(key, fill.clone());
            changed = true;
        }
    }
    changed
}

async fn dispatch_manual_order(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
    action: ManualOrderAction,
) -> Result<OrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account_name = order_ctx.account.name.clone();
    let contract_name = order_ctx.contract.name.clone();
    let symbol = contract_symbol(order_ctx.contract).to_string();
    let current_qty =
        account_contract_position_qty(session, &account_name, order_ctx.contract).round() as i32;

    match action {
        ManualOrderAction::Buy => {
            if current_qty != 0 {
                cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            }
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                "BUY",
                session.cfg.order_qty.max(1),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued { target_qty: None })
        }
        ManualOrderAction::Sell => {
            if current_qty != 0 {
                cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            }
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                "SELL",
                session.cfg.order_qty.max(1),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued { target_qty: None })
        }
        ManualOrderAction::Close => {
            if current_qty == 0 {
                return Ok(OrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract_name, account_name
                    ),
                });
            }
            cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            let side = if current_qty > 0 { "SELL" } else { "BUY" };
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                side,
                current_qty.abs(),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued {
                target_qty: Some(0),
            })
        }
    }
}

async fn dispatch_target_position_order(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
    target_qty: i32,
    automated: bool,
    reason: &str,
) -> Result<OrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account_name = order_ctx.account.name.clone();
    let contract_name = order_ctx.contract.name.clone();
    let symbol = contract_symbol(order_ctx.contract).to_string();
    let current_qty =
        account_contract_position_qty(session, &account_name, order_ctx.contract).round() as i32;
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(OrderDispatchOutcome::NoOp {
            message: format!(
                "Target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract_name, account_name
            ),
        });
    }

    let is_reversal =
        current_qty != 0 && target_qty != 0 && current_qty.signum() != target_qty.signum();
    if automated
        && is_reversal
        && session.execution_config.native_reversal_mode == NativeReversalMode::FlattenConfirmEnter
    {
        cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
        let flatten_side = if current_qty > 0 { "SELL" } else { "BUY" };
        submit_market_order(
            client,
            &session.cfg,
            &session.token,
            &account_name,
            &symbol,
            flatten_side,
            current_qty.abs(),
            latency,
        )
        .await?;
        schedule_followup_refresh(internal_tx);
        session.execution_runtime.pending_reversal_entry = Some(PendingNativeReversalEntry {
            target_qty,
            reason: reason.to_string(),
        });
        return Ok(OrderDispatchOutcome::Queued {
            target_qty: Some(0),
        });
    }

    if current_qty != 0 {
        cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
    }

    let side = if delta > 0 { "BUY" } else { "SELL" };
    submit_market_order(
        client,
        &session.cfg,
        &session.token,
        &account_name,
        &symbol,
        side,
        delta.abs(),
        latency,
    )
    .await?;
    schedule_followup_refresh(internal_tx);
    Ok(OrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

async fn cancel_selected_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(key) = selected_protection_key(session) else {
        return Ok(());
    };
    let Some(account) = session.selected_account_id.and_then(|account_id| {
        session
            .accounts
            .iter()
            .find(|account| account.id == account_id)
    }) else {
        return Ok(());
    };
    let ids = protection_order_ids(session, key);
    if ids.is_empty() {
        return Ok(());
    }
    cancel_multiple_orders(
        client,
        &session.cfg,
        &session.token,
        &account.name,
        &ids,
        latency,
    )
    .await?;
    session.managed_protection.remove(&key);
    schedule_followup_refresh(internal_tx);
    Ok(())
}

async fn sync_native_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    _reason: &str,
) -> Result<()> {
    let Some(key) = selected_protection_key(session) else {
        return Ok(());
    };
    let order_ctx = resolve_order_context(session)?;
    let exit_side = if signed_qty > 0 { "SELL" } else { "BUY" };
    let quantity = signed_qty.abs().max(1);
    let desired_take_profit = sanitize_price(take_profit_price);
    let desired_stop = sanitize_price(stop_price);

    let existing = session.managed_protection.get(&key).cloned();
    let take_profit_candidate = resolve_take_profit_candidate(
        session,
        key,
        &order_ctx.account.name,
        order_ctx.contract,
        exit_side,
    );
    let stop_candidate = resolve_stop_candidate(
        session,
        key,
        &order_ctx.account.name,
        order_ctx.contract,
        exit_side,
    );

    let action = if signed_qty == 0 || (desired_take_profit.is_none() && desired_stop.is_none()) {
        ProtectionSyncAction::Clear
    } else if existing.as_ref().is_some_and(|current| {
        current.signed_qty == signed_qty
            && prices_match(
                current.last_requested_take_profit_price,
                desired_take_profit,
            )
            && prices_match(current.last_requested_stop_price, desired_stop)
            && (desired_take_profit.is_none()
                || current.take_profit_order_id.is_some()
                || take_profit_candidate.is_some())
            && (desired_stop.is_none()
                || current.stop_order_id.is_some()
                || stop_candidate.is_some())
    }) {
        ProtectionSyncAction::None
    } else {
        ProtectionSyncAction::Replace {
            take_profit_price: desired_take_profit,
            stop_price: desired_stop,
        }
    };

    match action {
        ProtectionSyncAction::None => {}
        ProtectionSyncAction::Clear => {
            let ids = protection_order_ids(session, key);
            if !ids.is_empty() {
                cancel_multiple_orders(
                    client,
                    &session.cfg,
                    &session.token,
                    &order_ctx.account.name,
                    &ids,
                    latency,
                )
                .await?;
            }
            session.managed_protection.remove(&key);
            schedule_followup_refresh(internal_tx);
        }
        ProtectionSyncAction::Replace {
            take_profit_price,
            stop_price,
        } => {
            let mut next_state = existing.unwrap_or(ManagedProtectionOrders {
                signed_qty,
                take_profit_price,
                stop_price,
                last_requested_take_profit_price: take_profit_price,
                last_requested_stop_price: stop_price,
                take_profit_order_id: None,
                stop_order_id: None,
            });
            next_state.signed_qty = signed_qty;
            next_state.take_profit_price =
                take_profit_candidate.as_ref().and_then(|item| item.price);
            next_state.stop_price = stop_candidate.as_ref().and_then(|item| item.price);
            next_state.last_requested_take_profit_price = take_profit_price;
            next_state.last_requested_stop_price = stop_price;

            sync_take_profit_order(
                client,
                session,
                latency,
                &order_ctx.account.name,
                order_ctx.contract,
                exit_side,
                quantity,
                take_profit_price,
                take_profit_candidate,
                &mut next_state,
            )
            .await?;
            sync_stop_order(
                client,
                session,
                latency,
                &order_ctx.account.name,
                order_ctx.contract,
                exit_side,
                quantity,
                stop_price,
                stop_candidate,
                &mut next_state,
            )
            .await?;
            session.managed_protection.insert(key, next_state);
            schedule_followup_refresh(internal_tx);
        }
    }

    rebuild_account_snapshots(session);
    Ok(())
}

async fn sync_take_profit_order(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
    quantity: i32,
    desired_price: Option<f64>,
    existing: Option<ProtectionOrderCandidate>,
    state: &mut ManagedProtectionOrders,
) -> Result<()> {
    match (desired_price, existing) {
        (None, Some(existing)) => {
            cancel_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                &existing.order_id,
                latency,
            )
            .await?;
            state.take_profit_order_id = None;
            state.take_profit_price = None;
        }
        (None, None) => {
            state.take_profit_order_id = None;
            state.take_profit_price = None;
        }
        (Some(price), Some(existing)) => {
            if !prices_match(existing.price, Some(price)) || existing.quantity != Some(quantity) {
                update_order(
                    client,
                    &session.cfg,
                    &session.token,
                    account_name,
                    &existing.order_id,
                    quantity,
                    Some(price),
                    None,
                    latency,
                )
                .await?;
            }
            state.take_profit_order_id = Some(existing.order_id);
            state.take_profit_price = Some(price);
        }
        (Some(price), None) => {
            let order_id = place_protection_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                contract_symbol(contract),
                exit_side,
                quantity,
                "LIMIT",
                Some(price),
                None,
                latency,
            )
            .await?;
            state.take_profit_order_id = Some(order_id);
            state.take_profit_price = Some(price);
        }
    }
    Ok(())
}

async fn sync_stop_order(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
    quantity: i32,
    desired_price: Option<f64>,
    existing: Option<ProtectionOrderCandidate>,
    state: &mut ManagedProtectionOrders,
) -> Result<()> {
    match (desired_price, existing) {
        (None, Some(existing)) => {
            cancel_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                &existing.order_id,
                latency,
            )
            .await?;
            state.stop_order_id = None;
            state.stop_price = None;
        }
        (None, None) => {
            state.stop_order_id = None;
            state.stop_price = None;
        }
        (Some(price), Some(existing)) => {
            if !prices_match(existing.price, Some(price)) || existing.quantity != Some(quantity) {
                update_order(
                    client,
                    &session.cfg,
                    &session.token,
                    account_name,
                    &existing.order_id,
                    quantity,
                    None,
                    Some(price),
                    latency,
                )
                .await?;
            }
            state.stop_order_id = Some(existing.order_id);
            state.stop_price = Some(price);
        }
        (Some(price), None) => {
            let order_id = place_protection_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                contract_symbol(contract),
                exit_side,
                quantity,
                "STOP",
                None,
                Some(price),
                latency,
            )
            .await?;
            state.stop_order_id = Some(order_id);
            state.stop_price = Some(price);
        }
    }
    Ok(())
}

async fn submit_market_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    symbol: &str,
    side: &str,
    quantity: i32,
    latency: &mut LatencySnapshot,
) -> Result<Option<String>> {
    let payload = json!({
        "accountId": account_name,
        "exchSym": symbol,
        "side": side,
        "quantity": quantity,
        "orderType": "MARKET",
        "duration": ironbeam_duration_code(&cfg.time_in_force),
        "waitForOrderId": true,
    });
    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/order/{account_name}/place",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam market order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(pick_str(&parsed, &["orderId"]).map(ToString::to_string))
}

async fn place_protection_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    symbol: &str,
    side: &str,
    quantity: i32,
    order_type: &str,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    latency: &mut LatencySnapshot,
) -> Result<String> {
    let mut payload = Map::new();
    payload.insert("accountId".to_string(), json!(account_name));
    payload.insert("exchSym".to_string(), json!(symbol));
    payload.insert("side".to_string(), json!(side));
    payload.insert("quantity".to_string(), json!(quantity));
    payload.insert("orderType".to_string(), json!(order_type));
    payload.insert(
        "duration".to_string(),
        json!(ironbeam_duration_code(&cfg.time_in_force)),
    );
    payload.insert("waitForOrderId".to_string(), json!(true));
    if let Some(limit_price) = limit_price {
        payload.insert("limitPrice".to_string(), json!(limit_price));
    }
    if let Some(stop_price) = stop_price {
        payload.insert("stopPrice".to_string(), json!(stop_price));
    }

    let (parsed, elapsed_ms) = request_json(
        client
            .post(format!(
                "{}/order/{account_name}/place",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&Value::Object(payload)),
        "Ironbeam protection order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    pick_str(&parsed, &["orderId"])
        .map(ToString::to_string)
        .context("missing orderId in Ironbeam protection order response")
}

async fn update_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_id: &str,
    quantity: i32,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let mut payload = Map::new();
    payload.insert("orderId".to_string(), json!(order_id));
    payload.insert("quantity".to_string(), json!(quantity));
    if let Some(limit_price) = limit_price {
        payload.insert("limitPrice".to_string(), json!(limit_price));
    }
    if let Some(stop_price) = stop_price {
        payload.insert("stopPrice".to_string(), json!(stop_price));
    }

    let (_, elapsed_ms) = request_json(
        client
            .put(format!(
                "{}/order/{account_name}/update/{order_id}",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&Value::Object(payload)),
        "Ironbeam update order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

async fn cancel_order(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_id: &str,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let (_, elapsed_ms) = request_json(
        client
            .delete(format!(
                "{}/order/{account_name}/cancel/{order_id}",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token),
        "Ironbeam cancel order",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

async fn cancel_multiple_orders(
    client: &Client,
    cfg: &AppConfig,
    token: &str,
    account_name: &str,
    order_ids: &[String],
    latency: &mut LatencySnapshot,
) -> Result<()> {
    if order_ids.is_empty() {
        return Ok(());
    }
    let payload = json!({
        "accountId": account_name,
        "orderIds": order_ids,
    });
    let (_, elapsed_ms) = request_json(
        client
            .delete(format!(
                "{}/order/{account_name}/cancelMultiple",
                ironbeam_rest_url(cfg.env)
            ))
            .bearer_auth(token)
            .json(&payload),
        "Ironbeam cancel multiple orders",
    )
    .await?;
    latency.last_order_ack_ms = Some(elapsed_ms);
    latency.rest_rtt_ms = Some(elapsed_ms);
    Ok(())
}

fn resolve_order_context(session: &IronbeamSession) -> Result<OrderContext<'_>> {
    let account_id = session
        .selected_account_id
        .context("select an account before sending orders")?;
    let account = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .as_ref()
        .context("select a contract before sending orders")?;
    Ok(OrderContext { account, contract })
}

fn selected_protection_key(session: &IronbeamSession) -> Option<ProtectionKey> {
    let account_id = session.selected_account_id?;
    let contract_id = session.selected_contract.as_ref()?.id;
    Some(ProtectionKey {
        account_id,
        contract_id,
    })
}

fn protection_order_ids(session: &IronbeamSession, key: ProtectionKey) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(state) = session.managed_protection.get(&key) {
        if let Some(order_id) = state.take_profit_order_id.clone() {
            ids.push(order_id);
        }
        if let Some(order_id) = state.stop_order_id.clone() {
            if !ids.contains(&order_id) {
                ids.push(order_id);
            }
        }
    }

    if let Some(account_name) = account_name_by_id(session, key.account_id) {
        let symbol = session
            .selected_contract
            .as_ref()
            .map(contract_symbol)
            .unwrap_or_default();
        let exit_side = if selected_market_position_qty(session) > 0 {
            "SELL"
        } else {
            "BUY"
        };
        for order in active_orders_for_account(&session.account_state, account_name) {
            if order_symbol(order) != Some(symbol) {
                continue;
            }
            if order_side(order) != Some(exit_side) {
                continue;
            }
            if !is_protection_order_type(order) {
                continue;
            }
            if let Some(order_id) = order_id_string(order) {
                if !ids.contains(&order_id) {
                    ids.push(order_id);
                }
            }
        }
    }

    ids
}

fn refresh_managed_protection(session: &mut IronbeamSession) {
    let Some(contract) = session.selected_contract.as_ref() else {
        session.managed_protection.clear();
        return;
    };
    let symbol = contract_symbol(contract).to_string();
    let contract_id = contract.id;

    for account in &session.accounts {
        let key = ProtectionKey {
            account_id: account.id,
            contract_id,
        };
        let signed_qty =
            account_contract_position_qty(session, &account.name, contract).round() as i32;
        if signed_qty == 0 {
            session.managed_protection.remove(&key);
            continue;
        }

        let exit_side = if signed_qty > 0 { "SELL" } else { "BUY" };
        let take_profit = find_protection_candidate(
            session,
            &account.name,
            &symbol,
            exit_side,
            ProtectionOrderKind::TakeProfit,
            session
                .managed_protection
                .get(&key)
                .and_then(|state| state.take_profit_order_id.as_deref()),
        );
        let stop = find_protection_candidate(
            session,
            &account.name,
            &symbol,
            exit_side,
            ProtectionOrderKind::StopLoss,
            session
                .managed_protection
                .get(&key)
                .and_then(|state| state.stop_order_id.as_deref()),
        );

        let entry = session
            .managed_protection
            .entry(key)
            .or_insert(ManagedProtectionOrders {
                signed_qty,
                take_profit_price: None,
                stop_price: None,
                last_requested_take_profit_price: None,
                last_requested_stop_price: None,
                take_profit_order_id: None,
                stop_order_id: None,
            });
        entry.signed_qty = signed_qty;
        if let Some(candidate) = take_profit {
            entry.take_profit_order_id = Some(candidate.order_id);
            entry.take_profit_price = candidate.price;
        } else {
            entry.take_profit_order_id = None;
            if entry.last_requested_take_profit_price.is_none() {
                entry.take_profit_price = None;
            }
        }
        if let Some(candidate) = stop {
            entry.stop_order_id = Some(candidate.order_id);
            entry.stop_price = candidate.price;
        } else {
            entry.stop_order_id = None;
            if entry.last_requested_stop_price.is_none() {
                entry.stop_price = None;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ProtectionOrderKind {
    TakeProfit,
    StopLoss,
}

fn find_protection_candidate(
    session: &IronbeamSession,
    account_name: &str,
    symbol: &str,
    exit_side: &str,
    kind: ProtectionOrderKind,
    preferred_id: Option<&str>,
) -> Option<ProtectionOrderCandidate> {
    let mut candidates = active_orders_for_account(&session.account_state, account_name)
        .into_iter()
        .filter(|order| order_symbol(order) == Some(symbol))
        .filter(|order| order_side(order) == Some(exit_side))
        .filter_map(|order| {
            let order_type = normalized_order_type(order)?;
            let matches = match kind {
                ProtectionOrderKind::TakeProfit => order_type == "LIMIT",
                ProtectionOrderKind::StopLoss => order_type == "STOP" || order_type == "STOP_LIMIT",
            };
            if !matches {
                return None;
            }
            Some(ProtectionOrderCandidate {
                order_id: order_id_string(order)?,
                price: order_price(order),
                quantity: order_quantity(order),
            })
        })
        .collect::<Vec<_>>();
    if let Some(preferred_id) = preferred_id {
        if let Some(index) = candidates
            .iter()
            .position(|candidate| candidate.order_id == preferred_id)
        {
            return Some(candidates.remove(index));
        }
    }
    candidates.into_iter().next()
}

fn resolve_take_profit_candidate(
    session: &IronbeamSession,
    key: ProtectionKey,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
) -> Option<ProtectionOrderCandidate> {
    find_protection_candidate(
        session,
        account_name,
        contract_symbol(contract),
        exit_side,
        ProtectionOrderKind::TakeProfit,
        session
            .managed_protection
            .get(&key)
            .and_then(|state| state.take_profit_order_id.as_deref()),
    )
}

fn resolve_stop_candidate(
    session: &IronbeamSession,
    key: ProtectionKey,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
) -> Option<ProtectionOrderCandidate> {
    find_protection_candidate(
        session,
        account_name,
        contract_symbol(contract),
        exit_side,
        ProtectionOrderKind::StopLoss,
        session
            .managed_protection
            .get(&key)
            .and_then(|state| state.stop_order_id.as_deref()),
    )
}

fn active_orders_for_account<'a>(state: &'a AccountState, account_name: &str) -> Vec<&'a Value> {
    state
        .orders
        .get(account_name)
        .into_iter()
        .flat_map(|orders| orders.values())
        .filter(|order| order_is_active(order))
        .collect()
}

fn selected_protection_prices(session: &IronbeamSession) -> (Option<f64>, Option<f64>) {
    let Some(key) = selected_protection_key(session) else {
        return (None, None);
    };
    session
        .managed_protection
        .get(&key)
        .map(|state| {
            (
                state
                    .take_profit_price
                    .or(state.last_requested_take_profit_price),
                state.stop_price.or(state.last_requested_stop_price),
            )
        })
        .unwrap_or((None, None))
}

fn selected_market_position_qty(session: &IronbeamSession) -> i32 {
    let Some(account_name) = session
        .selected_account_id
        .and_then(|account_id| account_name_by_id(session, account_id))
    else {
        return 0;
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return 0;
    };
    account_contract_position_qty(session, account_name, contract).round() as i32
}

fn selected_market_entry_price(session: &IronbeamSession) -> Option<f64> {
    let account_name = session
        .selected_account_id
        .and_then(|account_id| account_name_by_id(session, account_id))?;
    let contract = session.selected_contract.as_ref()?;
    let symbol = contract_symbol(contract);
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in account_positions_for_symbol(&session.account_state, account_name, symbol) {
        let qty = signed_position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = position_entry_price(position)?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }
    (total_qty > f64::EPSILON).then_some(weighted_sum / total_qty)
}

fn account_contract_position_qty(
    session: &IronbeamSession,
    account_name: &str,
    contract: &ContractSuggestion,
) -> f64 {
    account_positions_for_symbol(
        &session.account_state,
        account_name,
        contract_symbol(contract),
    )
    .into_iter()
    .filter_map(signed_position_qty)
    .sum::<f64>()
}

fn account_positions_for_symbol<'a>(
    state: &'a AccountState,
    account_name: &str,
    symbol: &str,
) -> Vec<&'a Value> {
    state
        .positions
        .get(account_name)
        .into_iter()
        .flat_map(|positions| positions.values())
        .filter(|position| position_symbol(position) == Some(symbol))
        .collect()
}

fn selected_has_live_entry_path(session: &IronbeamSession) -> bool {
    let Some(account_name) = session
        .selected_account_id
        .and_then(|account_id| account_name_by_id(session, account_id))
    else {
        return false;
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return false;
    };
    let protection_ids = selected_protection_key(session)
        .and_then(|key| session.managed_protection.get(&key))
        .map(|protection| {
            [
                protection.take_profit_order_id.as_deref(),
                protection.stop_order_id.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    active_orders_for_account(&session.account_state, account_name)
        .into_iter()
        .filter(|order| order_symbol(order) == Some(contract_symbol(contract)))
        .any(|order| {
            let Some(order_id) = order_id_string(order) else {
                return true;
            };
            !protection_ids.iter().any(|known| *known == order_id)
        })
}

fn selected_trade_markers(
    account_state: &AccountState,
    contract: &ContractSuggestion,
) -> Vec<TradeMarker> {
    let symbol = contract_symbol(contract);
    let mut markers = account_state
        .fills
        .iter()
        .flat_map(|(account_name, fills)| {
            fills
                .values()
                .filter(move |fill| order_symbol(fill) == Some(symbol))
                .filter_map(move |fill| trade_marker_from_fill(account_name, contract, fill))
        })
        .collect::<Vec<_>>();
    markers.sort_by_key(|marker| marker.ts_ns);
    if markers.len() > TRADE_MARKER_LIMIT {
        let excess = markers.len() - TRADE_MARKER_LIMIT;
        markers.drain(..excess);
    }
    markers
}

fn trade_marker_from_fill(
    account_name: &str,
    contract: &ContractSuggestion,
    fill: &Value,
) -> Option<TradeMarker> {
    let side = match order_side(fill)?.to_ascii_uppercase().as_str() {
        "BUY" => TradeMarkerSide::Buy,
        "SELL" => TradeMarkerSide::Sell,
        _ => return None,
    };
    let ts_ns = pick_i64(fill, &["timeOrderEvent"])
        .and_then(normalize_unix_timestamp_ns)
        .or_else(|| parse_timestamp_ns(fill.get("fillDate")?))?;
    let price = pick_number(fill, &["fillPrice", "avgFillPrice", "price"])?;
    let qty = pick_number(fill, &["fillQuantity", "quantity"])?.round() as i32;
    let fill_id = fill_key(fill).map(|value| stable_id(&value));
    Some(TradeMarker {
        fill_id: fill_id.filter(|value| *value > 0),
        account_id: Some(stable_id(account_name)),
        contract_id: Some(contract.id),
        contract_name: Some(contract.name.clone()),
        ts_ns,
        price,
        qty: qty.max(1),
        side,
    })
}

async fn handle_execution_account_sync(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);
    refresh_managed_protection(session);

    if let Some(pending) = session.execution_runtime.pending_target_qty {
        if actual_qty == pending {
            session.execution_runtime.clear_pending_target();
            if !continue_staged_reversal(client, session, latency, internal_tx.clone()).await? {
                session.execution_runtime.last_summary =
                    format!("Position confirmed at target {actual_qty}");
            }
        } else if !selected_has_live_entry_path(session) {
            let timed_out = session
                .execution_runtime
                .pending_target_started_at
                .is_some_and(|started_at| {
                    started_at.elapsed().as_secs() >= PENDING_TARGET_WATCHDOG_SECS
                });
            if timed_out {
                session.execution_runtime.clear_pending_target();
                session.execution_runtime.last_closed_bar_ts =
                    latest_strategy_bar_ts(session).map(|last_ts| last_ts.saturating_sub(1));
                session.execution_runtime.last_summary = format!(
                    "Pending target {pending} cleared after Ironbeam order path went idle; re-evaluating."
                );
            }
        }
    }

    if session.execution_runtime.pending_target_qty.is_none() {
        let _ = continue_staged_reversal(client, session, latency, internal_tx.clone()).await?;
    }

    if session.execution_runtime.armed && session.execution_config.kind == StrategyKind::Native {
        sync_execution_protection(client, session, latency, internal_tx, None).await?;
    }

    rebuild_account_snapshots(session);
    Ok(())
}

async fn maybe_run_execution_strategy(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let actual_market_qty = selected_market_position_qty(session);
    let actual_market_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_market_qty, actual_market_entry);

    if session.execution_runtime.pending_target_qty.is_none() {
        if continue_staged_reversal(client, session, latency, internal_tx.clone()).await? {
            rebuild_account_snapshots(session);
            return Ok(());
        }
    }

    let max_automated_qty = session.execution_config.order_qty.max(1);
    if actual_market_qty.abs() > max_automated_qty {
        if selected_has_live_entry_path(session) {
            session.execution_runtime.last_summary = format!(
                "Waiting for Ironbeam broker sync: temporary position {actual_market_qty} exceeds max {max_automated_qty} while an order path is still active."
            );
        } else {
            disarm_execution_strategy(
                session,
                format!(
                    "Automation disarmed: position drifted to {actual_market_qty}, above configured max {max_automated_qty}."
                ),
            );
        }
        rebuild_account_snapshots(session);
        return Ok(());
    }

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        session.execution_runtime.last_summary = format!(
            "Waiting for prior Ironbeam order to settle (actual {actual_market_qty}, pending target {pending_target_qty})."
        );
        rebuild_account_snapshots(session);
        return Ok(());
    }

    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        rebuild_account_snapshots(session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current {}; waiting for next update.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        rebuild_account_snapshots(session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts)
    {
        return Ok(());
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let current_qty = effective_market_position_qty(session);
    let (signal_bar, signal, summary) = {
        let bars = strategy_bars(session);
        let signal_bar = bars
            .last()
            .cloned()
            .context("latest Ironbeam strategy bar disappeared during evaluation")?;
        let (signal, summary) = evaluate_active_execution_strategy(session, bars, current_qty);
        (signal_bar, signal, summary)
    };

    if let Some(window) = session_window_at(session, signal_bar.ts_ns) {
        if window.hold_entries {
            if actual_market_qty != 0 {
                match dispatch_target_position_order(
                    client,
                    session,
                    latency,
                    internal_tx.clone(),
                    0,
                    true,
                    &session_hold_reason(session, window, actual_market_qty),
                )
                .await?
                {
                    OrderDispatchOutcome::NoOp { message } => {
                        session.execution_runtime.last_summary = message.clone();
                        let _ = event_tx.send(ServiceEvent::Status(message));
                    }
                    OrderDispatchOutcome::Queued { target_qty } => {
                        session.execution_runtime.set_pending_target(target_qty);
                        session.execution_runtime.last_summary = if window.session_open {
                            format!(
                                "Session hold active; flattening {} {:.0}m before close.",
                                actual_market_qty,
                                window.minutes_to_close.unwrap_or_default()
                            )
                        } else {
                            format!(
                                "Session closed; flattening {} and holding until reopen.",
                                actual_market_qty
                            )
                        };
                    }
                }
                rebuild_account_snapshots(session);
                return Ok(());
            }

            sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar))
                .await?;
            session.execution_runtime.last_summary = if window.session_open {
                format!(
                    "Session hold active; no new entries with {:.0}m to close.",
                    window.minutes_to_close.unwrap_or_default()
                )
            } else {
                "Session closed; holding flat until reopen.".to_string()
            };
            rebuild_account_snapshots(session);
            return Ok(());
        }
    }

    session.execution_runtime.last_summary = summary.clone();

    let Some(target_qty) =
        target_qty_for_signal(signal, current_qty, session.execution_config.order_qty)
    else {
        sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar)).await?;
        rebuild_account_snapshots(session);
        return Ok(());
    };

    if target_qty == current_qty {
        sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar)).await?;
        rebuild_account_snapshots(session);
        return Ok(());
    }

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        current_qty,
        target_qty
    )));

    let reason = format!(
        "{} {} | {}",
        active_native_slug(session),
        signal.label(),
        summary
    );
    match dispatch_target_position_order(
        client,
        session,
        latency,
        internal_tx,
        target_qty,
        true,
        &reason,
    )
    .await?
    {
        OrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message.clone();
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        OrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.set_pending_target(target_qty);
        }
    }
    rebuild_account_snapshots(session);
    Ok(())
}

fn session_hold_reason(
    session: &IronbeamSession,
    window: InstrumentSessionWindow,
    actual_market_qty: i32,
) -> String {
    if window.session_open {
        format!(
            "{} session auto-close {:.0}m before {} close (qty {})",
            active_native_slug(session),
            window.minutes_to_close.unwrap_or_default(),
            session
                .market
                .session_profile
                .map(|profile| profile.label())
                .unwrap_or("session"),
            actual_market_qty
        )
    } else {
        format!(
            "{} session hold until {} reopen (qty {})",
            active_native_slug(session),
            session
                .market
                .session_profile
                .map(|profile| profile.label())
                .unwrap_or("session"),
            actual_market_qty
        )
    }
}

async fn continue_staged_reversal(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<bool> {
    let Some(staged) = session.execution_runtime.pending_reversal_entry.clone() else {
        return Ok(false);
    };

    let actual_qty = selected_market_position_qty(session);
    if actual_qty.signum() == staged.target_qty.signum() && actual_qty != 0 {
        session.execution_runtime.pending_reversal_entry = None;
        session.execution_runtime.last_summary =
            format!("Staged reversal resolved at target {actual_qty}");
        return Ok(true);
    }

    if actual_qty == 0 {
        if selected_has_live_entry_path(session) {
            session.execution_runtime.last_summary = format!(
                "Staged reversal flat; waiting for Ironbeam order path to clear before entering {}.",
                staged.target_qty
            );
            return Ok(true);
        }

        match dispatch_target_position_order(
            client,
            session,
            latency,
            internal_tx,
            staged.target_qty,
            false,
            &staged.reason,
        )
        .await?
        {
            OrderDispatchOutcome::NoOp { message } => {
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = message;
            }
            OrderDispatchOutcome::Queued { target_qty } => {
                session.execution_runtime.set_pending_target(target_qty);
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = format!(
                    "Flat confirmed; submitting staged reversal entry to {}.",
                    staged.target_qty
                );
            }
        }
        return Ok(true);
    }

    match dispatch_target_position_order(
        client,
        session,
        latency,
        internal_tx,
        0,
        false,
        &format!(
            "{} | staged reversal flatten {} -> 0 before {}",
            staged.reason, actual_qty, staged.target_qty
        ),
    )
    .await?
    {
        OrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message;
        }
        OrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.set_pending_target(target_qty);
            session.execution_runtime.last_summary = format!(
                "Flattening {} before staged reversal to {}.",
                actual_qty, staged.target_qty
            );
        }
    }
    Ok(true)
}

fn arm_execution_strategy(session: &mut IronbeamSession) {
    session.execution_runtime.clear_pending_target();
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
    session.execution_runtime.last_summary =
        if session.execution_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        } else {
            format!(
                "Native {} armed; waiting for first {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        };
}

fn disarm_execution_strategy(session: &mut IronbeamSession, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.clear_pending_target();
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

fn closed_bars(session: &IronbeamSession) -> &[Bar] {
    let closed_len = session.market.history_loaded.min(session.market.bars.len());
    &session.market.bars[..closed_len]
}

fn strategy_bars(session: &IronbeamSession) -> &[Bar] {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        &session.market.bars
    } else {
        closed_bars(session)
    }
}

fn latest_strategy_bar_ts(session: &IronbeamSession) -> Option<i64> {
    strategy_bars(session).last().map(|bar| bar.ts_ns)
}

fn active_native_slug(session: &IronbeamSession) -> &'static str {
    session.execution_config.native_strategy.slug()
}

fn active_native_label(session: &IronbeamSession) -> &'static str {
    session.execution_config.native_strategy.label()
}

fn active_signal_timing_label(session: &IronbeamSession) -> &'static str {
    match session.execution_config.native_signal_timing {
        NativeSignalTiming::ClosedBar => "closed bar",
        NativeSignalTiming::LiveBar => "live bar",
    }
}

fn session_window_at(session: &IronbeamSession, ts_ns: i64) -> Option<InstrumentSessionWindow> {
    session
        .market
        .session_profile
        .map(|profile| profile.evaluate(ts_ns))
}

fn evaluate_active_execution_strategy(
    session: &IronbeamSession,
    bars: &[Bar],
    current_qty: i32,
) -> (StrategySignal, String) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            let evaluation = session
                .execution_config
                .native_hma
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
        NativeStrategyKind::EmaCross => {
            let evaluation = session
                .execution_config
                .native_ema
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
    }
}

fn target_qty_for_signal(signal: StrategySignal, current_qty: i32, base_qty: i32) -> Option<i32> {
    let base_qty = base_qty.max(1);
    match signal {
        StrategySignal::Hold => None,
        StrategySignal::EnterLong => Some(base_qty),
        StrategySignal::EnterShort => Some(-base_qty),
        StrategySignal::ExitLongOnShortSignal => {
            if current_qty > 0 {
                Some(0)
            } else {
                None
            }
        }
    }
}

fn effective_market_position_qty(session: &IronbeamSession) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

fn sync_active_execution_position(
    session: &mut IronbeamSession,
    signed_qty: i32,
    entry_price: Option<f64>,
) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.sync_position(
            &mut session.execution_runtime.hma_execution,
            signed_qty,
            entry_price,
        ),
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.sync_position(
            &mut session.execution_runtime.ema_execution,
            signed_qty,
            entry_price,
        ),
    }
}

fn active_native_uses_protection(session: &IronbeamSession) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            session.execution_config.native_hma.uses_native_protection()
        }
        NativeStrategyKind::EmaCross => {
            session.execution_config.native_ema.uses_native_protection()
        }
    }
}

fn take_profit_price(session: &IronbeamSession, entry_price: f64, signed_qty: i32) -> Option<f64> {
    let side = side_from_signed_qty(signed_qty)?;
    let offset = match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session
            .execution_config
            .native_hma
            .take_profit_offset(session.market.tick_size)?,
        NativeStrategyKind::EmaCross => session
            .execution_config
            .native_ema
            .take_profit_offset(session.market.tick_size)?,
    };
    Some(match side {
        crate::strategies::PositionSide::Long => entry_price + offset,
        crate::strategies::PositionSide::Short => entry_price - offset,
    })
}

fn combined_stop_price(session: &mut IronbeamSession, trailing_bar: Option<&Bar>) -> Option<f64> {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_hma
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.hma_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_hma
                .current_effective_stop_price(
                    &session.execution_runtime.hma_execution,
                    session.market.tick_size,
                )
        }
        NativeStrategyKind::EmaCross => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_ema
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.ema_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_ema
                .current_effective_stop_price(
                    &session.execution_runtime.ema_execution,
                    session.market.tick_size,
                )
        }
    }
}

async fn sync_execution_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<InternalEvent>,
    trailing_bar: Option<&Bar>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }
    if !active_native_uses_protection(session) {
        return Ok(());
    }
    if session.execution_runtime.pending_target_qty.is_some() {
        return Ok(());
    }

    let signed_qty = selected_market_position_qty(session);
    let entry_price = selected_market_entry_price(session);
    sync_active_execution_position(session, signed_qty, entry_price);

    let (take_profit_price, stop_price) = if signed_qty == 0 {
        (None, None)
    } else if let Some(entry_price) = entry_price {
        (
            take_profit_price(session, entry_price, signed_qty),
            combined_stop_price(session, trailing_bar),
        )
    } else {
        return Ok(());
    };

    sync_native_protection(
        client,
        session,
        latency,
        internal_tx,
        signed_qty,
        take_profit_price,
        stop_price,
        "native execution sync",
    )
    .await
}

async fn request_json(request: reqwest::RequestBuilder, context: &str) -> Result<(Value, u64)> {
    let started = Instant::now();
    let response = request
        .send()
        .await
        .with_context(|| format!("{context}: send request"))?;
    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!("{context} failed ({status}): {body}");
    }
    if body.trim().is_empty() {
        return Ok((Value::Null, elapsed_ms));
    }
    let parsed: Value =
        serde_json::from_str(&body).with_context(|| format!("{context}: parse JSON response"))?;
    ensure_api_ok(&parsed, context)?;
    Ok((parsed, elapsed_ms))
}

fn ensure_api_ok(parsed: &Value, context: &str) -> Result<()> {
    if let Some(status) = parsed.get("status").and_then(Value::as_str) {
        if !status.eq_ignore_ascii_case("OK") {
            let message = parsed
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default();
            bail!("{context} rejected ({status}): {message}");
        }
    }
    Ok(())
}

fn parse_time_bars(items: &[Value]) -> Vec<Bar> {
    let mut bars = items
        .iter()
        .filter_map(|item| {
            Some(Bar {
                ts_ns: parse_timestamp_ns(item.get("t")?)?,
                open: pick_number(item, &["o", "open"])?,
                high: pick_number(item, &["h", "high"])?,
                low: pick_number(item, &["l", "low"])?,
                close: pick_number(item, &["c", "close"])?,
            })
        })
        .collect::<Vec<_>>();
    bars.sort_by_key(|bar| bar.ts_ns);
    bars
}

fn merge_bars(existing: &mut Vec<Bar>, incoming: Vec<Bar>, bar_limit: usize) -> usize {
    let mut appended = 0;
    for bar in incoming {
        match existing.binary_search_by_key(&bar.ts_ns, |current| current.ts_ns) {
            Ok(index) => existing[index] = bar,
            Err(index) => {
                existing.insert(index, bar);
                appended += 1;
            }
        }
    }
    if existing.len() > bar_limit {
        let excess = existing.len() - bar_limit;
        existing.drain(..excess);
    }
    appended
}

fn market_bar_limit(session: &IronbeamSession) -> usize {
    session
        .cfg
        .history_bars
        .saturating_add(512)
        .max(DEFAULT_BAR_LIMIT)
}

fn contract_symbol(contract: &ContractSuggestion) -> &str {
    pick_str(&contract.raw, &["symbol", "exchSym"]).unwrap_or(contract.name.as_str())
}

fn contract_session_profile(contract: &ContractSuggestion) -> InstrumentSessionProfile {
    let symbol_type = pick_str(&contract.raw, &["symbolType", "productType"])
        .unwrap_or_default()
        .to_ascii_lowercase();
    if symbol_type.contains("equity") || symbol_type.contains("stock") {
        InstrumentSessionProfile::EquityRth
    } else {
        InstrumentSessionProfile::FuturesGlobex
    }
}

fn position_symbol(position: &Value) -> Option<&str> {
    pick_str(position, &["exchSym", "symbol", "s"])
}

fn account_id_string(value: &Value) -> Option<String> {
    pick_str(value, &["accountId", "a"]).map(ToString::to_string)
}

fn position_key(position: &Value) -> Option<String> {
    if let Some(id) = pick_str(position, &["positionId", "id"]) {
        return Some(id.to_string());
    }
    let symbol = position_symbol(position)?;
    let side = pick_str(position, &["side", "sd", "s"]).unwrap_or("LONG");
    Some(format!("{symbol}:{side}"))
}

fn order_id_string(order: &Value) -> Option<String> {
    pick_str(order, &["orderId", "oid", "id"]).map(ToString::to_string)
}

fn fill_key(fill: &Value) -> Option<String> {
    if let Some(update_id) = pick_str(fill, &["orderUpdateId", "fillId", "id"]) {
        return Some(update_id.to_string());
    }
    let order_id = order_id_string(fill)?;
    let fill_date = pick_str(fill, &["fillDate"]).unwrap_or("unknown");
    let fill_qty = pick_number(fill, &["fillQuantity", "quantity"]).unwrap_or_default();
    Some(format!("{order_id}:{fill_date}:{fill_qty:.4}"))
}

fn order_symbol(order: &Value) -> Option<&str> {
    pick_str(order, &["exchSym", "symbol", "s"])
}

fn order_side(order: &Value) -> Option<&str> {
    pick_str(order, &["side", "sd"])
}

fn order_quantity(order: &Value) -> Option<i32> {
    pick_number(order, &["quantity", "q"]).map(|qty| qty.round() as i32)
}

fn order_price(order: &Value) -> Option<f64> {
    pick_number(order, &["limitPrice", "stopPrice", "price", "lp", "sp"])
}

fn normalized_order_type(order: &Value) -> Option<&'static str> {
    match pick_str(order, &["orderType", "ot"])?
        .trim()
        .to_ascii_uppercase()
        .as_str()
    {
        "1" | "MARKET" => Some("MARKET"),
        "2" | "LIMIT" => Some("LIMIT"),
        "3" | "STOP" => Some("STOP"),
        "4" | "STOP_LIMIT" | "STOPLIMIT" => Some("STOP_LIMIT"),
        _ => None,
    }
}

fn is_protection_order_type(order: &Value) -> bool {
    matches!(
        normalized_order_type(order),
        Some("LIMIT" | "STOP" | "STOP_LIMIT")
    )
}

fn order_is_active(order: &Value) -> bool {
    let status = pick_str(order, &["status", "st"])
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase();
    if matches!(
        status.as_str(),
        "FILLED"
            | "CANCELLED"
            | "CANCELED"
            | "REJECTED"
            | "EXPIRED"
            | "DONE"
            | "CLOSED"
            | "LIQUIDATED"
            | "INVALID"
    ) {
        return false;
    }

    let quantity = order_quantity(order).unwrap_or_default();
    let filled = pick_number(order, &["fillQuantity", "fillTotalQuantity"])
        .unwrap_or_default()
        .round() as i32;
    quantity <= 0 || filled < quantity
}

fn signed_position_qty(position: &Value) -> Option<f64> {
    let qty = pick_number(position, &["quantity", "q", "netPos"])?;
    let side = pick_str(position, &["side", "sd"]).unwrap_or("LONG");
    if side.eq_ignore_ascii_case("SHORT") || qty < 0.0 {
        Some(-qty.abs())
    } else {
        Some(qty)
    }
}

fn position_entry_price(position: &Value) -> Option<f64> {
    pick_number(position, &["price", "avgPrice", "averagePrice", "netPrice"])
}

fn value_items(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![value],
        _ => Vec::new(),
    }
}

fn position_update_groups(value: &Value) -> Vec<(String, Vec<Value>, bool)> {
    match value {
        Value::Array(items) => items
            .iter()
            .flat_map(position_update_groups)
            .collect::<Vec<_>>(),
        Value::Object(object) => {
            if let Some(items) = object.get("positions").and_then(Value::as_array) {
                let Some(account_id) = account_id_string(value) else {
                    return Vec::new();
                };
                return vec![(account_id, items.to_vec(), true)];
            }
            let Some(account_id) = account_id_string(value) else {
                return Vec::new();
            };
            vec![(account_id, vec![Value::Object(object.clone())], false)]
        }
        _ => Vec::new(),
    }
}

fn pick_str<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| value.get(key)?.as_str())
}

fn pick_number(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|key| value.get(key))
        .and_then(value_f64)
}

fn pick_i64(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .find_map(|key| value.get(key))
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_u64().map(|value| value as i64))
                .or_else(|| value.as_str()?.parse::<i64>().ok())
        })
}

fn value_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|number| number as f64))
        .or_else(|| value.as_u64().map(|number| number as f64))
        .or_else(|| value.as_str()?.parse::<f64>().ok())
}

fn parse_timestamp_ns(value: &Value) -> Option<i64> {
    if let Some(raw) = value
        .as_i64()
        .and_then(normalize_unix_timestamp_ns)
        .or_else(|| value.as_str()?.parse::<i64>().ok())
        .and_then(normalize_unix_timestamp_ns)
    {
        return Some(raw);
    }

    let text = value.as_str()?;
    chrono::DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|dt| dt.timestamp_nanos_opt().unwrap_or_default())
        .filter(|ts| *ts > 0)
}

fn normalize_unix_timestamp_ns(raw: i64) -> Option<i64> {
    let magnitude = raw.unsigned_abs();
    if magnitude >= 1_000_000_000_000_000_000 {
        Some(raw)
    } else if magnitude >= 1_000_000_000_000_000 {
        raw.checked_mul(1_000)
    } else if magnitude >= 1_000_000_000_000 {
        raw.checked_mul(1_000_000)
    } else if magnitude >= 1_000_000_000 {
        raw.checked_mul(1_000_000_000)
    } else {
        None
    }
}

fn account_name_by_id<'a>(session: &'a IronbeamSession, account_id: i64) -> Option<&'a str> {
    session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .map(|account| account.name.as_str())
}

fn sanitize_price(price: Option<f64>) -> Option<f64> {
    price.filter(|price| price.is_finite() && *price > 0.0)
}

fn prices_match(left: Option<f64>, right: Option<f64>) -> bool {
    match (sanitize_price(left), sanitize_price(right)) {
        (None, None) => true,
        (Some(left), Some(right)) => (left - right).abs() <= 1e-6,
        _ => false,
    }
}

fn stable_id(raw: &str) -> i64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash & i64::MAX as u64) as i64
}

fn empty_as_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn ironbeam_duration_code(time_in_force: &str) -> &'static str {
    match time_in_force.trim().to_ascii_lowercase().as_str() {
        "gtc" | "gtt" | "good_till_cancel" | "good till cancel" => "1",
        _ => "0",
    }
}

fn ironbeam_rest_url(env: TradingEnvironment) -> &'static str {
    match env {
        TradingEnvironment::Sim => "https://demo.ironbeamapi.com/v2",
        TradingEnvironment::Live => "https://live.ironbeamapi.com/v2",
    }
}

fn ironbeam_ws_base_url(env: TradingEnvironment) -> &'static str {
    match env {
        TradingEnvironment::Sim => "wss://demo.ironbeamapi.com/v2",
        TradingEnvironment::Live => "wss://live.ironbeamapi.com/v2",
    }
}

fn schedule_followup_refresh(internal_tx: UnboundedSender<InternalEvent>) {
    schedule_refresh(
        internal_tx,
        Duration::from_millis(FOLLOWUP_REFRESH_DELAY_MS),
        Some("Refreshing Ironbeam account state after order activity.".to_string()),
    );
}

fn schedule_refresh(
    internal_tx: UnboundedSender<InternalEvent>,
    delay: Duration,
    reason: Option<String>,
) {
    tokio::spawn(async move {
        time::sleep(delay).await;
        let _ = internal_tx.send(InternalEvent::RefreshAccountState { reason });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::ExecutionStrategyConfig;

    #[test]
    fn duration_codes_match_supported_inputs() {
        assert_eq!(ironbeam_duration_code("Day"), "0");
        assert_eq!(ironbeam_duration_code("GTC"), "1");
        assert_eq!(ironbeam_duration_code("good till cancel"), "1");
    }

    #[test]
    fn target_qty_for_signal_maps_expected_transitions() {
        assert_eq!(target_qty_for_signal(StrategySignal::Hold, 0, 1), None);
        assert_eq!(
            target_qty_for_signal(StrategySignal::EnterLong, 0, 2),
            Some(2)
        );
        assert_eq!(
            target_qty_for_signal(StrategySignal::EnterShort, 0, 3),
            Some(-3)
        );
        assert_eq!(
            target_qty_for_signal(StrategySignal::ExitLongOnShortSignal, 2, 1),
            Some(0)
        );
    }

    #[test]
    fn active_order_statuses_exclude_terminal_states() {
        assert!(order_is_active(
            &json!({"status": "WORKING", "quantity": 1})
        ));
        assert!(!order_is_active(
            &json!({"status": "FILLED", "quantity": 1, "fillQuantity": 1})
        ));
        assert!(!order_is_active(
            &json!({"status": "CANCELED", "quantity": 1})
        ));
    }

    #[test]
    fn selected_trade_markers_filter_to_selected_symbol() {
        let mut account_state = AccountState::default();
        account_state.fills.insert(
            "SIM".to_string(),
            BTreeMap::from([
                (
                    "one".to_string(),
                    json!({
                        "orderUpdateId": "one",
                        "accountId": "SIM",
                        "exchSym": "XCME:ESM6",
                        "side": "BUY",
                        "fillPrice": 5000.0,
                        "fillQuantity": 1,
                        "fillDate": "2026-03-30T14:30:00Z",
                    }),
                ),
                (
                    "two".to_string(),
                    json!({
                        "orderUpdateId": "two",
                        "accountId": "SIM",
                        "exchSym": "XCME:NQM6",
                        "side": "SELL",
                        "fillPrice": 18000.0,
                        "fillQuantity": 1,
                        "fillDate": "2026-03-30T14:31:00Z",
                    }),
                ),
            ]),
        );

        let markers = selected_trade_markers(
            &account_state,
            &ContractSuggestion {
                id: stable_id("XCME:ESM6"),
                name: "XCME:ESM6".to_string(),
                description: "ES".to_string(),
                raw: json!({"symbol": "XCME:ESM6"}),
            },
        );

        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].price, 5000.0);
        assert_eq!(markers[0].side, TradeMarkerSide::Buy);
    }

    #[test]
    fn execution_runtime_snapshot_hides_internal_timers() {
        let mut runtime = ExecutionRuntimeState::default();
        runtime.armed = true;
        runtime.set_pending_target(Some(2));
        runtime.last_summary = "waiting".to_string();

        let snapshot = runtime.snapshot();

        assert!(snapshot.armed);
        assert_eq!(snapshot.pending_target_qty, Some(2));
        assert_eq!(snapshot.last_summary, "waiting");
    }

    #[test]
    fn selected_position_qty_uses_selected_account_and_contract() {
        let contract = ContractSuggestion {
            id: stable_id("XCME:ESM6"),
            name: "XCME:ESM6".to_string(),
            description: "ES".to_string(),
            raw: json!({"symbol": "XCME:ESM6"}),
        };
        let session = IronbeamSession {
            cfg: AppConfig::default(),
            token: "token".to_string(),
            user_name: None,
            accounts: vec![AccountInfo {
                id: stable_id("SIM"),
                name: "SIM".to_string(),
                raw: json!({}),
            }],
            selected_account_id: Some(stable_id("SIM")),
            selected_contract: Some(contract.clone()),
            account_state: AccountState {
                positions: BTreeMap::from([(
                    "SIM".to_string(),
                    BTreeMap::from([(
                        "XCME:ESM6:LONG".to_string(),
                        json!({
                            "accountId": "SIM",
                            "exchSym": "XCME:ESM6",
                            "quantity": 2,
                            "side": "LONG",
                            "price": 5000.0,
                        }),
                    )]),
                )]),
                ..AccountState::default()
            },
            account_snapshots: Vec::new(),
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            managed_protection: BTreeMap::new(),
            market: MarketSnapshot::default(),
            market_task: None,
        };

        assert_eq!(selected_market_position_qty(&session), 2);
        assert_eq!(selected_market_entry_price(&session), Some(5000.0));
    }
}

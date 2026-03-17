use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, ExecutionStrategyConfig, NativeStrategyKind,
    StrategyKind,
};
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::net::ToSocketAddrs;
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceCommand {
    Connect(AppConfig),
    ReplayState,
    SelectAccount {
        account_id: i64,
    },
    SearchContracts {
        query: String,
        limit: usize,
    },
    SubscribeBars {
        contract: ContractSuggestion,
        bar_type: BarType,
    },
    ManualOrder {
        action: ManualOrderAction,
    },
    SetTargetPosition {
        target_qty: i32,
        automated: bool,
        reason: String,
    },
    SyncNativeProtection {
        signed_qty: i32,
        take_profit_price: Option<f64>,
        stop_price: Option<f64>,
        reason: String,
    },
    SetExecutionStrategyConfig(ExecutionStrategyConfig),
    ArmExecutionStrategy,
    DisarmExecutionStrategy {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ManualOrderAction {
    Buy,
    Sell,
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    TradeMarkersUpdated(Vec<TradeMarker>),
    Latency(LatencySnapshot),
    ExecutionState(ExecutionStateSnapshot),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarType {
    Minute1,
    Range1,
}

impl BarType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Minute1 => "1 Min",
            Self::Range1 => "1 Range",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Minute1 => Self::Range1,
            Self::Range1 => Self::Minute1,
        }
    }

    pub fn chart_description(self) -> Value {
        match self {
            Self::Minute1 => json!({
                "underlyingType": "MinuteBar",
                "elementSize": 1,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }),
            Self::Range1 => json!({
                "underlyingType": "Tick",
                "elementSize": 1,
                "elementSizeUnit": "Range",
                "withHistogram": false
            }),
        }
    }
}

impl Default for BarType {
    fn default() -> Self {
        Self::Minute1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSuggestion {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InstrumentSessionProfile {
    FuturesGlobex,
    EquityRth,
}

impl InstrumentSessionProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::FuturesGlobex => "Globex",
            Self::EquityRth => "RTH",
        }
    }

    pub fn evaluate(self, ts_ns: i64) -> InstrumentSessionWindow {
        if ts_ns <= 0 {
            return InstrumentSessionWindow {
                session_open: true,
                minutes_to_close: None,
                hold_entries: false,
            };
        }

        let dt_et = DateTime::<Utc>::from_timestamp_nanos(ts_ns).with_timezone(&New_York);
        match self {
            Self::FuturesGlobex => futures_globex_window(dt_et),
            Self::EquityRth => equity_rth_window(dt_et),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstrumentSessionWindow {
    pub session_open: bool,
    pub minutes_to_close: Option<f64>,
    pub hold_entries: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketSnapshot {
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub bars: Vec<Bar>,
    pub trade_markers: Vec<TradeMarker>,
    pub session_profile: Option<InstrumentSessionProfile>,
    pub value_per_point: Option<f64>,
    pub tick_size: Option<f64>,
    pub history_loaded: usize,
    pub live_bars: usize,
    pub status: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TradeMarkerSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeMarker {
    pub fill_id: Option<i64>,
    pub account_id: Option<i64>,
    pub contract_id: Option<i64>,
    pub contract_name: Option<String>,
    pub ts_ns: i64,
    pub price: f64,
    pub qty: i32,
    pub side: TradeMarkerSide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub account_id: i64,
    pub account_name: String,
    pub balance: Option<f64>,
    pub cash_balance: Option<f64>,
    pub net_liq: Option<f64>,
    pub realized_pnl: Option<f64>,
    pub unrealized_pnl: Option<f64>,
    pub intraday_margin: Option<f64>,
    pub open_position_qty: Option<f64>,
    pub market_position_qty: Option<f64>,
    pub market_entry_price: Option<f64>,
    pub selected_contract_take_profit_price: Option<f64>,
    pub selected_contract_stop_price: Option<f64>,
    pub raw_account: Option<Value>,
    pub raw_risk: Option<Value>,
    pub raw_cash: Option<Value>,
    pub raw_positions: Vec<Value>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LatencySnapshot {
    pub rest_rtt_ms: Option<u64>,
    pub last_order_ack_ms: Option<u64>,
    pub last_order_seen_ms: Option<u64>,
    pub last_exec_report_ms: Option<u64>,
    pub last_fill_ms: Option<u64>,
}

struct ServiceState {
    client: Client,
    broker_tx: UnboundedSender<BrokerCommand>,
    session: Option<SessionState>,
    user_task: Option<JoinHandle<()>>,
    market_task: Option<JoinHandle<()>>,
    rest_probe_task: Option<JoinHandle<()>>,
    latency: LatencySnapshot,
    snapshot_revision: u64,
}

struct SessionState {
    cfg: AppConfig,
    tokens: TokenBundle,
    accounts: Vec<AccountInfo>,
    request_tx: UnboundedSender<UserSocketCommand>,
    execution_config: ExecutionStrategyConfig,
    execution_runtime: ExecutionRuntimeState,
    order_latency_tracker: Option<OrderLatencyTracker>,
    order_submit_in_flight: bool,
    user_store: UserSyncStore,
    selected_account_id: Option<i64>,
    selected_contract: Option<ContractSuggestion>,
    bar_type: BarType,
    market: MarketSnapshot,
    managed_protection: BTreeMap<StrategyProtectionKey, ManagedProtectionOrders>,
    next_strategy_order_nonce: u64,
}

#[derive(Debug, Clone, Default)]
struct ExecutionRuntimeState {
    armed: bool,
    last_closed_bar_ts: Option<i64>,
    pending_target_qty: Option<i32>,
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
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
    }
}

#[derive(Clone, Default)]
struct UserSyncStore {
    accounts: BTreeMap<i64, Value>,
    risk: BTreeMap<i64, Value>,
    cash: BTreeMap<i64, Value>,
    positions: BTreeMap<i64, BTreeMap<i64, Value>>,
    orders: BTreeMap<i64, BTreeMap<i64, Value>>,
    fills: BTreeMap<i64, BTreeMap<i64, Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StrategyProtectionKey {
    account_id: i64,
    contract_id: i64,
}

#[derive(Debug, Clone)]
struct ManagedProtectionOrders {
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    take_profit_cl_ord_id: Option<String>,
    stop_cl_ord_id: Option<String>,
    take_profit_order_id: Option<i64>,
    stop_order_id: Option<i64>,
}

const TOKEN_REFRESH_LEAD_SECS: i64 = 300;
const SESSION_MAINTENANCE_INTERVAL_SECS: u64 = 30;
pub const AUTO_CLOSE_MINUTES_BEFORE_SESSION_END: f64 = 15.0;

fn infer_session_profile(product: &Value) -> InstrumentSessionProfile {
    match product
        .get("productType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "futures" => InstrumentSessionProfile::FuturesGlobex,
        _ => InstrumentSessionProfile::EquityRth,
    }
}

fn futures_globex_window(dt_et: DateTime<chrono_tz::Tz>) -> InstrumentSessionWindow {
    let hour = fractional_hour(&dt_et);
    let session_open = match dt_et.weekday() {
        Weekday::Sun => hour >= 18.0,
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu => hour < 17.0 || hour >= 18.0,
        Weekday::Fri => hour < 17.0,
        Weekday::Sat => false,
    };

    if !session_open {
        return InstrumentSessionWindow {
            session_open: false,
            minutes_to_close: None,
            hold_entries: true,
        };
    }

    let close_date = if dt_et.weekday() == Weekday::Sun || hour >= 18.0 {
        dt_et
            .date_naive()
            .succ_opt()
            .unwrap_or_else(|| dt_et.date_naive())
    } else {
        dt_et.date_naive()
    };
    let close_at = new_york_close(close_date, 17, 0, 0);
    let minutes_to_close = close_at.map(|close| minutes_until(dt_et, close));
    let hold_entries = minutes_to_close
        .map(|minutes| minutes <= AUTO_CLOSE_MINUTES_BEFORE_SESSION_END)
        .unwrap_or(false);

    InstrumentSessionWindow {
        session_open: true,
        minutes_to_close,
        hold_entries,
    }
}

fn equity_rth_window(dt_et: DateTime<chrono_tz::Tz>) -> InstrumentSessionWindow {
    let hour = fractional_hour(&dt_et);
    let session_open = matches!(
        dt_et.weekday(),
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu | Weekday::Fri
    ) && hour >= 9.5
        && hour < 16.0;

    if !session_open {
        return InstrumentSessionWindow {
            session_open: false,
            minutes_to_close: None,
            hold_entries: true,
        };
    }

    let close_at = new_york_close(dt_et.date_naive(), 16, 0, 0);
    let minutes_to_close = close_at.map(|close| minutes_until(dt_et, close));
    let hold_entries = minutes_to_close
        .map(|minutes| minutes <= AUTO_CLOSE_MINUTES_BEFORE_SESSION_END)
        .unwrap_or(false);

    InstrumentSessionWindow {
        session_open: true,
        minutes_to_close,
        hold_entries,
    }
}

fn fractional_hour(dt_et: &DateTime<chrono_tz::Tz>) -> f64 {
    dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0 + dt_et.second() as f64 / 3600.0
}

fn new_york_close(
    date: chrono::NaiveDate,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<DateTime<chrono_tz::Tz>> {
    let naive = date.and_hms_opt(hour, minute, second)?;
    New_York.from_local_datetime(&naive).single()
}

fn minutes_until(start: DateTime<chrono_tz::Tz>, end: DateTime<chrono_tz::Tz>) -> f64 {
    ((end - start).num_seconds() as f64 / 60.0).max(0.0)
}

enum InternalEvent {
    UserEntities(Vec<EntityEnvelope>),
    SnapshotsBuilt {
        revision: u64,
        snapshots: Vec<AccountSnapshot>,
    },
    RestLatencyMeasured(u64),
    UserSocketStatus(String),
    Market(MarketUpdate),
    BrokerOrderAck(BrokerOrderAck),
    BrokerOrderFailed(BrokerOrderFailure),
    Error(String),
}

struct UserSocketCommand {
    endpoint: String,
    query: Option<String>,
    body: Option<Value>,
    response_tx: oneshot::Sender<Result<Value, String>>,
}

struct OrderLatencyTracker {
    started_at: time::Instant,
    cl_ord_id: String,
    order_id: Option<i64>,
    seen_recorded: bool,
    exec_report_recorded: bool,
    fill_recorded: bool,
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

#[derive(Debug, Clone)]
struct MarketUpdate {
    contract_id: i64,
    contract_name: String,
    session_profile: Option<InstrumentSessionProfile>,
    value_per_point: Option<f64>,
    tick_size: Option<f64>,
    history_loaded: usize,
    live_bars: usize,
    status: String,
    bars: MarketBarsUpdate,
}

#[derive(Debug, Clone)]
enum MarketBarsUpdate {
    Snapshot {
        closed_bars: Vec<Bar>,
        forming_bar: Option<Bar>,
    },
    Forming {
        forming_bar: Bar,
    },
    Closed {
        closed_bar: Bar,
        forming_bar: Option<Bar>,
    },
}

struct BrokerCommand {
    request_tx: UnboundedSender<UserSocketCommand>,
    order: PendingMarketOrder,
}

struct PendingMarketOrder {
    cl_ord_id: String,
    payload: Value,
    action_label: String,
    order_action: String,
    order_qty: i32,
    contract_name: String,
    account_name: String,
    reason_suffix: Option<String>,
    target_qty: Option<i32>,
}

struct BrokerOrderAck {
    cl_ord_id: String,
    order_id: Option<i64>,
    submit_rtt_ms: u64,
    message: String,
}

struct BrokerOrderFailure {
    cl_ord_id: String,
    message: String,
    target_qty: Option<i32>,
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
}

fn spawn_user_sync_task(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> (UnboundedSender<UserSocketCommand>, JoinHandle<()>) {
    let (request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(user_sync_worker(
        cfg,
        tokens,
        account_ids,
        request_rx,
        internal_tx,
    ));
    (request_tx, task)
}

fn spawn_broker_gateway_task(
    request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(broker_gateway_worker(request_rx, internal_tx))
}

async fn broker_gateway_worker(
    mut request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    while let Some(command) = request_rx.recv().await {
        match submit_market_order_via_gateway(command).await {
            Ok(ack) => {
                let _ = internal_tx.send(InternalEvent::BrokerOrderAck(ack));
            }
            Err(failure) => {
                let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
            }
        }
    }
}

async fn submit_market_order_via_gateway(
    command: BrokerCommand,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    let started_at = time::Instant::now();
    let parsed = match request_order_json(
        &command.request_tx,
        "order/placeorder",
        &command.order.payload,
    )
    .await
    {
        Ok(parsed) => parsed,
        Err(err) => {
            return Err(BrokerOrderFailure {
                cl_ord_id: command.order.cl_ord_id,
                message: err.to_string(),
                target_qty: command.order.target_qty,
            });
        }
    };
    let submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    let order_id = json_i64(&parsed, "orderId").or_else(|| json_i64(&parsed, "id"));
    let mut message = format!(
        "{} submitted: {} {} {} on {}",
        command.order.action_label,
        command.order.order_action,
        command.order.order_qty,
        command.order.contract_name,
        command.order.account_name
    );
    if let Some(reason) = command.order.reason_suffix.as_deref() {
        message.push_str(&format!(" [{reason}]"));
    }
    if let Some(order_id) = order_id {
        message.push_str(&format!(" (order {order_id})"));
    }
    message.push_str(&format!(" [clOrdId {}]", command.order.cl_ord_id));
    Ok(BrokerOrderAck {
        cl_ord_id: command.order.cl_ord_id,
        order_id,
        submit_rtt_ms,
        message,
    })
}

fn spawn_rest_probe_task(
    client: Client,
    cfg: AppConfig,
    access_token: String,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        interval.tick().await;
        loop {
            interval.tick().await;
            if let Ok(rest_rtt_ms) = measure_rest_rtt_ms(&client, &cfg.env, &access_token).await {
                let _ = internal_tx.send(InternalEvent::RestLatencyMeasured(rest_rtt_ms));
            }
        }
    })
}

fn request_snapshot_refresh(
    state: &mut ServiceState,
    internal_tx: &UnboundedSender<InternalEvent>,
) {
    let Some(session) = state.session.as_ref() else {
        return;
    };
    state.snapshot_revision = state.snapshot_revision.saturating_add(1);
    let revision = state.snapshot_revision;
    let accounts = session.accounts.clone();
    let market = session.market.clone();
    let managed_protection = session.managed_protection.clone();
    let user_store = session.user_store.clone();
    let internal_tx = internal_tx.clone();
    tokio::spawn(async move {
        let snapshots = user_store.build_snapshots(&accounts, Some(&market), &managed_protection);
        let _ = internal_tx.send(InternalEvent::SnapshotsBuilt {
            revision,
            snapshots,
        });
    });
}

fn market_last_closed_ts(market: &MarketSnapshot) -> Option<i64> {
    let closed_len = market.history_loaded.min(market.bars.len());
    closed_len
        .checked_sub(1)
        .and_then(|idx| market.bars.get(idx))
        .map(|bar| bar.ts_ns)
}

fn build_market_update(
    contract: &ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    history_loaded: usize,
    live_bars: usize,
    status: String,
    before_closed_len: usize,
    before_last_closed: Option<Bar>,
    before_forming: Option<Bar>,
    series: &LiveSeries,
) -> Option<MarketUpdate> {
    let bars = if before_closed_len == 0 && history_loaded > 0 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len + 1 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.closed_bars.last() != before_last_closed.as_ref() {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.forming_bar != before_forming {
        series
            .forming_bar
            .clone()
            .map(|forming_bar| MarketBarsUpdate::Forming { forming_bar })
    } else {
        None
    }?;

    Some(MarketUpdate {
        contract_id: contract.id,
        contract_name: contract.name.clone(),
        session_profile: market_specs.and_then(|specs| specs.session_profile),
        value_per_point: market_specs.and_then(|specs| specs.value_per_point),
        tick_size: market_specs.and_then(|specs| specs.tick_size),
        history_loaded,
        live_bars,
        status,
        bars,
    })
}

fn apply_market_update(market: &mut MarketSnapshot, update: MarketUpdate) -> bool {
    let prev_last_closed_ts = market_last_closed_ts(market);
    market.contract_id = Some(update.contract_id);
    market.contract_name = Some(update.contract_name);
    market.session_profile = update.session_profile;
    market.value_per_point = update.value_per_point;
    market.tick_size = update.tick_size;
    market.live_bars = update.live_bars;
    market.status = update.status;

    let closed_bar_advanced = match update.bars {
        MarketBarsUpdate::Snapshot {
            closed_bars,
            forming_bar,
        } => {
            let next_last_closed_ts = closed_bars.last().map(|bar| bar.ts_ns);
            market.history_loaded = update.history_loaded.min(closed_bars.len());
            market.bars = closed_bars;
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            next_last_closed_ts.is_some_and(|ts| prev_last_closed_ts.is_none_or(|prev| ts > prev))
        }
        MarketBarsUpdate::Forming { forming_bar } => {
            let closed_len = update.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            market.history_loaded = closed_len;
            market.bars.push(forming_bar);
            false
        }
        MarketBarsUpdate::Closed {
            closed_bar,
            forming_bar,
        } => {
            let closed_len = market.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            match market.bars.last_mut() {
                Some(last) if last.ts_ns == closed_bar.ts_ns => {
                    *last = closed_bar.clone();
                }
                Some(last) if closed_bar.ts_ns > last.ts_ns => {
                    market.bars.push(closed_bar.clone());
                }
                None => market.bars.push(closed_bar.clone()),
                _ => {}
            }
            market.history_loaded = update.history_loaded.min(market.bars.len());
            market.bars.truncate(market.history_loaded);
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            prev_last_closed_ts.is_none_or(|prev| closed_bar.ts_ns > prev)
        }
    };

    closed_bar_advanced
}

fn execution_state_snapshot(session: &SessionState) -> ExecutionStateSnapshot {
    ExecutionStateSnapshot {
        config: session.execution_config.clone(),
        runtime: session.execution_runtime.snapshot(),
    }
}

fn emit_execution_state(event_tx: &UnboundedSender<ServiceEvent>, session: &SessionState) {
    let _ = event_tx.send(ServiceEvent::ExecutionState(execution_state_snapshot(
        session,
    )));
}

fn closed_bars(session: &SessionState) -> &[Bar] {
    let closed_len = session.market.history_loaded.min(session.market.bars.len());
    &session.market.bars[..closed_len]
}

fn latest_closed_bar_ts(session: &SessionState) -> Option<i64> {
    closed_bars(session).last().map(|bar| bar.ts_ns)
}

fn session_window_at(session: &SessionState, ts_ns: i64) -> Option<InstrumentSessionWindow> {
    session
        .market
        .session_profile
        .map(|profile| profile.evaluate(ts_ns))
}

fn selected_contract_positions<'a>(session: &'a SessionState) -> Vec<&'a Value> {
    let Some(account_id) = session.selected_account_id else {
        return Vec::new();
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return Vec::new();
    };
    session
        .user_store
        .positions
        .get(&account_id)
        .into_iter()
        .flat_map(|positions| positions.values())
        .filter(|position| position_matches_contract(position, contract))
        .collect()
}

fn selected_market_position_qty(session: &SessionState) -> i32 {
    let qty = selected_contract_positions(session)
        .into_iter()
        .filter_map(position_qty)
        .sum::<f64>();
    qty.round() as i32
}

fn selected_market_entry_price(session: &SessionState) -> Option<f64> {
    let positions = selected_contract_positions(session);
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in positions {
        let qty = position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }

    if total_qty <= f64::EPSILON {
        None
    } else {
        Some(weighted_sum / total_qty)
    }
}

fn active_native_slug(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.slug()
}

fn active_native_label(session: &SessionState) -> &'static str {
    session.execution_config.native_strategy.label()
}

fn active_native_uses_protection(session: &SessionState) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            session.execution_config.native_hma.uses_native_protection()
        }
        NativeStrategyKind::EmaCross => {
            session.execution_config.native_ema.uses_native_protection()
        }
    }
}

fn sync_active_execution_position(
    session: &mut SessionState,
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

fn take_profit_price(session: &SessionState, entry_price: f64, signed_qty: i32) -> Option<f64> {
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

fn combined_stop_price(session: &mut SessionState, trailing_bar: Option<&Bar>) -> Option<f64> {
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
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
    trailing_bar: Option<&Bar>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }
    if !active_native_uses_protection(session) {
        return Ok(());
    }

    let signed_qty = selected_market_position_qty(session);
    let entry_price = selected_market_entry_price(session);
    sync_active_execution_position(session, signed_qty, entry_price);
    let reason = if signed_qty == 0 {
        format!("{} flat", active_native_slug(session))
    } else if trailing_bar.is_some() {
        format!("{} bar sync", active_native_slug(session))
    } else {
        format!("{} position sync", active_native_slug(session))
    };

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

    if let Some(message) =
        sync_native_protection(session, signed_qty, take_profit_price, stop_price, &reason).await?
    {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }

    Ok(())
}

fn effective_market_position_qty(session: &SessionState) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

fn evaluate_active_execution_strategy(
    session: &SessionState,
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

fn arm_execution_strategy(session: &mut SessionState) {
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_closed_bar_ts(session);
    session.execution_runtime.last_summary =
        if session.execution_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current closed bar.",
                active_native_label(session)
            )
        } else {
            format!(
                "Native {} armed; waiting for first closed bar.",
                active_native_label(session)
            )
        };
}

fn disarm_execution_strategy(session: &mut SessionState, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

async fn handle_execution_account_sync(
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);

    let mut runtime_changed = false;
    if let Some(pending) = session.execution_runtime.pending_target_qty {
        let reached = pending == actual_qty;
        // Position overshot past target — entry filled, then something else
        // (e.g. orphaned orders) pushed position further.  Release pending
        // so the strategy can correct on the next bar.
        let overshot = (pending > 0 && actual_qty > pending)
            || (pending < 0 && actual_qty < pending);
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
            session.execution_runtime.last_summary = if reached {
                format!("Position confirmed at target {actual_qty}")
            } else {
                format!(
                    "Position at {actual_qty} (target was {pending}); re-evaluating on next bar"
                )
            };
            runtime_changed = true;
        }
    }

    if session.execution_runtime.armed && session.execution_config.kind == StrategyKind::Native {
        sync_execution_protection(session, event_tx, None).await?;
    }

    if runtime_changed {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

async fn maybe_run_execution_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let actual_market_qty = selected_market_position_qty(session);
    let actual_market_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_market_qty, actual_market_entry);

    if session.execution_runtime.pending_target_qty.is_some() {
        session.execution_runtime.last_summary =
            "Waiting for prior automated order to settle.".to_string();
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let Some(last_closed_ts) = latest_closed_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for market data.",
            active_native_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_closed_ts);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current bar; waiting for next close.",
            active_native_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_runtime.last_closed_bar_ts == Some(last_closed_ts) {
        return Ok(());
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_closed_ts);

    let current_qty = effective_market_position_qty(session);
    let (last_closed, signal, summary) = {
        let closed = closed_bars(session);
        let last_closed = closed
            .last()
            .cloned()
            .context("latest closed bar disappeared during strategy evaluation")?;
        let (signal, summary) = evaluate_active_execution_strategy(session, closed, current_qty);
        (last_closed, signal, summary)
    };

    if let Some(window) = session_window_at(session, last_closed.ts_ns) {
        if window.hold_entries {
            if actual_market_qty != 0 {
                if let Some(message) = sync_native_protection(
                    session,
                    0,
                    None,
                    None,
                    &format!("{} session auto-close", active_native_slug(session)),
                )
                .await?
                {
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                let reason = if window.session_open {
                    format!(
                        "{} session auto-close {:.0}m before {} close",
                        active_native_slug(session),
                        window.minutes_to_close.unwrap_or_default(),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                } else {
                    format!(
                        "{} session hold until {} reopen",
                        active_native_slug(session),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                };
                match dispatch_target_position_order(session, broker_tx, 0, true, &reason).await? {
                    MarketOrderDispatchOutcome::NoOp { message } => {
                        let _ = event_tx.send(ServiceEvent::Status(message));
                    }
                    MarketOrderDispatchOutcome::Queued { target_qty } => {
                        session.execution_runtime.pending_target_qty = target_qty;
                    }
                }
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
                emit_execution_state(event_tx, session);
                return Ok(());
            }

            sync_execution_protection(session, event_tx, Some(&last_closed)).await?;
            session.execution_runtime.last_summary = if window.session_open {
                format!(
                    "Session hold active; no new entries with {:.0}m to close.",
                    window.minutes_to_close.unwrap_or_default()
                )
            } else {
                "Session closed; holding flat until reopen.".to_string()
            };
            emit_execution_state(event_tx, session);
            return Ok(());
        }
    }

    session.execution_runtime.last_summary = summary.clone();

    let Some(target_qty) =
        target_qty_for_signal(signal, current_qty, session.execution_config.order_qty)
    else {
        sync_execution_protection(session, event_tx, Some(&last_closed)).await?;
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if target_qty == current_qty {
        sync_execution_protection(session, event_tx, Some(&last_closed)).await?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        current_qty,
        target_qty
    )));

    if current_qty != 0 {
        if let Some(message) = sync_native_protection(
            session,
            0,
            None,
            None,
            &format!(
                "{} target transition {} -> {}",
                active_native_slug(session),
                current_qty,
                target_qty
            ),
        )
        .await?
        {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
    }

    let reason = format!(
        "{} {} | {}",
        active_native_slug(session),
        signal.label(),
        summary
    );
    match dispatch_target_position_order(session, broker_tx, target_qty, true, &reason).await? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
        }
    }
    emit_execution_state(event_tx, session);
    Ok(())
}

pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let _broker_task = spawn_broker_gateway_task(broker_rx, internal_tx.clone());
    let mut state = ServiceState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        broker_tx,
        session: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    };
    let mut maintenance_tick =
        time::interval(Duration::from_secs(SESSION_MAINTENANCE_INTERVAL_SECS));
    maintenance_tick.tick().await;

    while let Some(next) = tokio::select! {
        biased;
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
        _ = maintenance_tick.tick() => Some(Either::MaintenanceTick),
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
                if let Err(err) =
                    handle_internal(internal, &mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::MaintenanceTick => {
                if let Err(err) = maintain_session(&mut state, &event_tx, internal_tx.clone()).await
                {
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
    MaintenanceTick,
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
            state.latency = LatencySnapshot::default();
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
            let snapshots = user_store.build_snapshots(&accounts, None, &BTreeMap::new());

            let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts.clone()));
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));

            let account_ids = accounts
                .iter()
                .map(|account| account.id)
                .collect::<Vec<_>>();
            let user_cfg = cfg.clone();
            let user_tokens = tokens.clone();
            let (request_tx, user_task) =
                spawn_user_sync_task(user_cfg, user_tokens, account_ids, internal_tx.clone());
            let rest_probe_task = spawn_rest_probe_task(
                state.client.clone(),
                cfg.clone(),
                tokens.access_token.clone(),
                internal_tx.clone(),
            );
            state.user_task = Some(user_task);
            state.rest_probe_task = Some(rest_probe_task);

            state.session = Some(SessionState {
                cfg,
                tokens,
                accounts,
                request_tx,
                execution_config: ExecutionStrategyConfig::default(),
                execution_runtime: ExecutionRuntimeState::default(),
                order_latency_tracker: None,
                order_submit_in_flight: false,
                user_store,
                selected_account_id,
                selected_contract: None,
                bar_type: BarType::default(),
                market: MarketSnapshot::default(),
                managed_protection: BTreeMap::new(),
                next_strategy_order_nonce: 1,
            });
            if let Some(session) = state.session.as_ref() {
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::ReplayState => {
            let Some(session) = state.session.as_ref() else {
                let _ = event_tx.send(ServiceEvent::Disconnected);
                return Ok(());
            };
            let _ = event_tx.send(ServiceEvent::Connected {
                env: session.cfg.env,
                user_name: session.tokens.user_name.clone(),
                auth_mode: session.cfg.auth_mode,
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(session.accounts.clone()));
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(
                session.user_store.build_snapshots(
                    &session.accounts,
                    Some(&session.market),
                    &session.managed_protection,
                ),
            ));
            if session.market.contract_id.is_some()
                || session.market.contract_name.is_some()
                || !session.market.bars.is_empty()
                || !session.market.status.is_empty()
            {
                let _ = event_tx.send(ServiceEvent::MarketSnapshot(session.market.clone()));
            }
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::SelectAccount { account_id } => {
            {
                let Some(session) = state.session.as_mut() else {
                    bail!("not connected");
                };
                session.selected_account_id = Some(account_id);
                handle_execution_account_sync(session, event_tx).await?;
            }
            request_snapshot_refresh(state, &internal_tx);
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
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let Some(task) = state.market_task.take() {
                task.abort();
            }
            session.market = MarketSnapshot::default();
            let market_specs = fetch_contract_specs(
                &state.client,
                &session.cfg.env,
                &session.tokens.access_token,
                &contract,
            )
            .await
            .ok();
            session.selected_contract = Some(contract.clone());
            session.bar_type = bar_type;
            session.execution_runtime.last_closed_bar_ts = None;
            session.execution_runtime.pending_target_qty = None;
            session.execution_runtime.reset_execution();
            session.execution_runtime.last_summary =
                "Selected contract changed; waiting for market data.".to_string();
            emit_execution_state(event_tx, session);
            let cfg = session.cfg.clone();
            let token = session.tokens.md_access_token.clone();
            state.market_task = Some(tokio::spawn(market_data_worker(
                cfg,
                token,
                contract,
                market_specs,
                bar_type,
                internal_tx,
            )));
        }
        ServiceCommand::ManualOrder { action } => {
            let broker_tx = state.broker_tx.clone();
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let MarketOrderDispatchOutcome::NoOp { message } =
                dispatch_manual_order(session, &broker_tx, action).await?
            {
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
        }
        ServiceCommand::SetTargetPosition {
            target_qty,
            automated,
            reason,
        } => {
            let broker_tx = state.broker_tx.clone();
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let MarketOrderDispatchOutcome::NoOp { message } =
                dispatch_target_position_order(session, &broker_tx, target_qty, automated, &reason)
                    .await?
            {
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
        }
        ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
        } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let Some(message) =
                sync_native_protection(session, signed_qty, take_profit_price, stop_price, &reason)
                    .await?
            {
                request_snapshot_refresh(state, &internal_tx);
                let _ = event_tx.send(ServiceEvent::Status(message));
                let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            }
        }
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if session.execution_config != config {
                session.execution_config = config;
                if session.execution_runtime.armed {
                    session.execution_runtime.armed = false;
                    session.execution_runtime.pending_target_qty = None;
                    session.execution_runtime.last_closed_bar_ts = None;
                    session.execution_runtime.reset_execution();
                    session.execution_runtime.last_summary =
                        "Native strategy config changed; press Continue to re-arm.".to_string();
                }
                emit_execution_state(event_tx, session);
            }
        }
        ServiceCommand::ArmExecutionStrategy => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            arm_execution_strategy(session);
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            disarm_execution_strategy(session, reason);
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

async fn handle_internal(
    internal: InternalEvent,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::UserEntities(entities) => {
            let mut latency_changed = false;
            let mut trade_markers_changed = false;
            {
                let Some(session) = state.session.as_mut() else {
                    return Ok(());
                };
                for envelope in &entities {
                    latency_changed |=
                        update_latency_from_envelope(session, &mut state.latency, &envelope);
                    session.user_store.apply(envelope.clone());
                }
                for envelope in &entities {
                    if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                        continue;
                    }
                    if let Some(marker) = trade_marker_from_fill(session, &envelope.entity) {
                        trade_markers_changed |= record_trade_marker(session, marker);
                    }
                }
                if trade_markers_changed {
                    let _ = event_tx.send(ServiceEvent::TradeMarkersUpdated(
                        session.market.trade_markers.clone(),
                    ));
                }
                handle_execution_account_sync(session, event_tx).await?;
            }
            request_snapshot_refresh(state, &internal_tx);
            if latency_changed {
                let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            }
        }
        InternalEvent::SnapshotsBuilt {
            revision,
            snapshots,
        } => {
            if revision == state.snapshot_revision && state.session.is_some() {
                let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
            }
        }
        InternalEvent::RestLatencyMeasured(rest_rtt_ms) => {
            state.latency.rest_rtt_ms = Some(rest_rtt_ms);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::UserSocketStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::Market(update) => {
            if state.session.is_some() {
                let broker_tx = state.broker_tx.clone();
                let (snapshot, closed_bar_advanced) = {
                    let session = state.session.as_mut().expect("checked session above");
                    let closed_bar_advanced = apply_market_update(&mut session.market, update);
                    maybe_run_execution_strategy(session, &broker_tx, event_tx).await?;
                    (session.market.clone(), closed_bar_advanced)
                };
                if closed_bar_advanced {
                    request_snapshot_refresh(state, &internal_tx);
                }
                let _ = event_tx.send(ServiceEvent::MarketSnapshot(snapshot));
                return Ok(());
            }
        }
        InternalEvent::BrokerOrderAck(ack) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if let Some(tracker) = session.order_latency_tracker.as_mut() {
                    if tracker.cl_ord_id == ack.cl_ord_id {
                        tracker.order_id = ack.order_id;
                    }
                }
            }
            state.latency.last_order_ack_ms = Some(ack.submit_rtt_ms);
            state.latency.last_order_seen_ms = None;
            state.latency.last_exec_report_ms = None;
            state.latency.last_fill_ms = None;
            let _ = event_tx.send(ServiceEvent::Status(ack.message));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::BrokerOrderFailed(failure) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if session
                    .order_latency_tracker
                    .as_ref()
                    .is_some_and(|tracker| tracker.cl_ord_id == failure.cl_ord_id)
                {
                    session.order_latency_tracker = None;
                }
                if let Some(target_qty) = failure.target_qty {
                    if session.execution_runtime.pending_target_qty == Some(target_qty) {
                        session.execution_runtime.pending_target_qty = None;
                        session.execution_runtime.last_summary = failure.message.clone();
                        emit_execution_state(event_tx, session);
                    }
                }
            }
            let _ = event_tx.send(ServiceEvent::Error(failure.message));
        }
        InternalEvent::Error(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
    }
    Ok(())
}

async fn maintain_session(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        return Ok(());
    };

    let refresh_action = next_token_maintenance_action(&session.cfg, &session.tokens)?;
    let mut forced_restart = false;
    let mut status_message = None;

    if let Some(action) = refresh_action {
        let next_tokens = match action {
            TokenMaintenanceAction::RefreshCredentials => {
                request_access_token(&state.client, &session.cfg).await?
            }
            TokenMaintenanceAction::ReloadTokenFile => load_runtime_token_bundle(&session.cfg)?,
        };

        if let Some(session) = state.session.as_mut() {
            if token_bundle_changed(&session.tokens, &next_tokens) {
                session.tokens = next_tokens;
                save_token_cache(&session.cfg.session_cache_path, &session.tokens)?;
                refresh_session_state(&state.client, session, event_tx).await?;
                if let Some(task) = state.user_task.take() {
                    task.abort();
                }
                if let Some(task) = state.market_task.take() {
                    task.abort();
                }
                if let Some(task) = state.rest_probe_task.take() {
                    task.abort();
                }
                forced_restart = true;
                status_message = Some(match action {
                    TokenMaintenanceAction::RefreshCredentials => {
                        "Session token refreshed; reconnecting background streams.".to_string()
                    }
                    TokenMaintenanceAction::ReloadTokenFile => {
                        "Session token reloaded from file; reconnecting background streams."
                            .to_string()
                    }
                });
            }
        }
    }

    let restart = ensure_background_tasks(state, internal_tx).await?;

    if let Some(message) = status_message {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }
    if restart.user_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "User sync stream restarted.".to_string(),
        ));
    }
    if restart.market_restarted && !forced_restart {
        let contract_name = state
            .session
            .as_ref()
            .and_then(|session| session.selected_contract.as_ref())
            .map(|contract| contract.name.clone())
            .unwrap_or_else(|| "selected contract".to_string());
        let _ = event_tx.send(ServiceEvent::Status(format!(
            "Market data stream restarted for {contract_name}."
        )));
    }
    if restart.rest_probe_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "REST latency probe restarted.".to_string(),
        ));
    }

    Ok(())
}

async fn refresh_session_state(
    client: &Client,
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let accounts = list_accounts(client, &session.cfg.env, &session.tokens.access_token).await?;
    let mut user_store = UserSyncStore::default();
    seed_user_store(
        client,
        &session.cfg.env,
        &session.tokens.access_token,
        &mut user_store,
    )
    .await;

    session.accounts = accounts.clone();
    if let Some(selected_account_id) = session.selected_account_id {
        if !session
            .accounts
            .iter()
            .any(|account| account.id == selected_account_id)
        {
            session.selected_account_id = session.accounts.first().map(|account| account.id);
        }
    } else {
        session.selected_account_id = session.accounts.first().map(|account| account.id);
    }
    session.user_store = user_store;

    let snapshots = session.user_store.build_snapshots(
        &session.accounts,
        Some(&session.market),
        &session.managed_protection,
    );
    let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts));
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenMaintenanceAction {
    RefreshCredentials,
    ReloadTokenFile,
}

fn next_token_maintenance_action(
    cfg: &AppConfig,
    tokens: &TokenBundle,
) -> Result<Option<TokenMaintenanceAction>> {
    if empty_as_none(&cfg.token_override).is_some() {
        return Ok(None);
    }

    match cfg.auth_mode {
        AuthMode::Credentials => {
            if token_refresh_due(tokens, Utc::now()) {
                Ok(Some(TokenMaintenanceAction::RefreshCredentials))
            } else {
                Ok(None)
            }
        }
        AuthMode::TokenFile => {
            let loaded = load_runtime_token_bundle(cfg)?;
            if token_bundle_changed(tokens, &loaded) {
                Ok(Some(TokenMaintenanceAction::ReloadTokenFile))
            } else {
                Ok(None)
            }
        }
    }
}

fn load_runtime_token_bundle(cfg: &AppConfig) -> Result<TokenBundle> {
    load_token_file(&cfg.token_path)
        .or_else(|_| load_token_file(&cfg.session_cache_path))
        .with_context(|| {
            format!(
                "load token from {} or {}",
                cfg.token_path.display(),
                cfg.session_cache_path.display()
            )
        })
}

fn token_refresh_due(tokens: &TokenBundle, now: DateTime<Utc>) -> bool {
    let Some(expires_at) = token_expires_at(tokens) else {
        return false;
    };
    expires_at <= now + chrono::Duration::seconds(TOKEN_REFRESH_LEAD_SECS)
}

fn token_expires_at(tokens: &TokenBundle) -> Option<DateTime<Utc>> {
    tokens
        .expiration_time
        .as_deref()
        .and_then(parse_expiration_time)
        .or_else(|| jwt_expiration_time(&tokens.access_token))
}

fn parse_expiration_time(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }

    let numeric = raw.trim().parse::<i64>().ok()?;
    let seconds = if numeric > 10_000_000_000 {
        numeric / 1000
    } else {
        numeric
    };
    DateTime::<Utc>::from_timestamp(seconds, 0)
}

fn jwt_expiration_time(token: &str) -> Option<DateTime<Utc>> {
    let claims = token.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD
        .decode(claims)
        .or_else(|_| URL_SAFE.decode(claims))
        .ok()?;
    let parsed: Value = serde_json::from_slice(&payload).ok()?;
    let seconds = parsed.get("exp").and_then(Value::as_i64)?;
    DateTime::<Utc>::from_timestamp(seconds, 0)
}

fn token_bundle_changed(current: &TokenBundle, next: &TokenBundle) -> bool {
    current.access_token != next.access_token
        || current.md_access_token != next.md_access_token
        || current.expiration_time != next.expiration_time
        || current.user_id != next.user_id
        || current.user_name != next.user_name
}

#[derive(Debug, Clone, Copy, Default)]
struct TaskRestartState {
    user_restarted: bool,
    market_restarted: bool,
    rest_probe_restarted: bool,
}

async fn ensure_background_tasks(
    state: &mut ServiceState,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<TaskRestartState> {
    let Some(session) = state.session.as_ref() else {
        return Ok(TaskRestartState::default());
    };

    let user_needed = state
        .user_task
        .as_ref()
        .is_none_or(tokio::task::JoinHandle::is_finished);
    let market_needed = session.selected_contract.is_some()
        && state
            .market_task
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished);
    let rest_probe_needed = state
        .rest_probe_task
        .as_ref()
        .is_none_or(tokio::task::JoinHandle::is_finished);

    let user_spawn = if user_needed {
        Some((
            session.cfg.clone(),
            session.tokens.clone(),
            session
                .accounts
                .iter()
                .map(|account| account.id)
                .collect::<Vec<_>>(),
        ))
    } else {
        None
    };
    let market_spawn = if market_needed {
        session.selected_contract.as_ref().map(|contract| {
            (
                session.cfg.clone(),
                session.tokens.access_token.clone(),
                session.tokens.md_access_token.clone(),
                contract.clone(),
            )
        })
    } else {
        None
    };
    let rest_probe_spawn = if rest_probe_needed {
        Some((
            state.client.clone(),
            session.cfg.clone(),
            session.tokens.access_token.clone(),
        ))
    } else {
        None
    };

    if user_needed {
        if let Some(task) = state.user_task.take() {
            task.abort();
        }
        if let Some((cfg, tokens, account_ids)) = user_spawn {
            let (request_tx, user_task) =
                spawn_user_sync_task(cfg, tokens, account_ids, internal_tx.clone());
            if let Some(session) = state.session.as_mut() {
                session.request_tx = request_tx;
            }
            state.user_task = Some(user_task);
        }
    }

    if market_needed {
        if let Some(task) = state.market_task.take() {
            task.abort();
        }
        if let Some((cfg, access_token, md_access_token, contract)) = market_spawn {
            let market_specs =
                fetch_contract_specs(&state.client, &cfg.env, &access_token, &contract)
                    .await
                    .ok();
            let bar_type = state
                .session
                .as_ref()
                .map(|s| s.bar_type)
                .unwrap_or_default();
            state.market_task = Some(tokio::spawn(market_data_worker(
                cfg,
                md_access_token,
                contract,
                market_specs,
                bar_type,
                internal_tx.clone(),
            )));
        }
    }

    if rest_probe_needed {
        if let Some(task) = state.rest_probe_task.take() {
            task.abort();
        }
        if let Some((client, cfg, access_token)) = rest_probe_spawn {
            state.rest_probe_task = Some(spawn_rest_probe_task(
                client,
                cfg,
                access_token,
                internal_tx.clone(),
            ));
        }
    }

    Ok(TaskRestartState {
        user_restarted: user_needed,
        market_restarted: market_needed,
        rest_probe_restarted: rest_probe_needed,
    })
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
    if let Some(task) = state.rest_probe_task.take() {
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

async fn measure_rest_rtt_ms(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
) -> Result<u64> {
    let started = time::Instant::now();
    let _ = fetch_auth_me(client, env, token).await?;
    Ok(started.elapsed().as_millis() as u64)
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

enum MarketOrderDispatchOutcome {
    NoOp { message: String },
    Queued { target_qty: Option<i32> },
}

async fn dispatch_manual_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    action: ManualOrderAction,
) -> Result<MarketOrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    ensure_no_market_order_submit_in_flight(session)?;

    let (order_action, order_qty, action_label, automated, reason_suffix) = match action {
        ManualOrderAction::Buy => ("Buy", session.cfg.order_qty, "Buy", false, None),
        ManualOrderAction::Sell => ("Sell", session.cfg.order_qty, "Sell", false, None),
        ManualOrderAction::Close => {
            let Some(net_qty) = session
                .user_store
                .contract_position_qty(account.id, &contract)
            else {
                return Ok(MarketOrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract.name, account.name
                    ),
                });
            };
            let close_qty = net_qty.abs().round() as i32;
            if close_qty <= 0 {
                return Ok(MarketOrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract.name, account.name
                    ),
                });
            }
            let close_action = if net_qty > 0.0 { "Sell" } else { "Buy" };
            (close_action, close_qty, "Close", false, None)
        }
    };

    cancel_strategy_protection_for_selected(session).await?;

    let order = build_market_order_request(
        session,
        &account,
        &contract,
        order_action,
        order_qty,
        action_label,
        automated,
        reason_suffix,
        None,
    );
    enqueue_market_order(session, broker_tx, order)?;
    Ok(MarketOrderDispatchOutcome::Queued { target_qty: None })
}

async fn dispatch_target_position_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    target_qty: i32,
    automated: bool,
    reason: &str,
) -> Result<MarketOrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    ensure_no_market_order_submit_in_flight(session)?;
    let current_qty = session
        .user_store
        .contract_position_qty(account.id, &contract)
        .unwrap_or(0.0)
        .round() as i32;
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(MarketOrderDispatchOutcome::NoOp {
            message: format!(
                "Strategy target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract.name, account.name
            ),
        });
    }

    cancel_strategy_protection_for_selected(session).await?;

    let order_action = if delta > 0 { "Buy" } else { "Sell" };
    let order_qty = delta.unsigned_abs() as i32;
    let action_label = "Strategy";
    let reason_suffix = Some(format!(
        "target {} -> {} ({reason})",
        current_qty, target_qty
    ));

    let order = build_market_order_request(
        session,
        &account,
        &contract,
        order_action,
        order_qty,
        action_label,
        automated,
        reason_suffix.as_deref(),
        Some(target_qty),
    );
    enqueue_market_order(session, broker_tx, order)?;
    Ok(MarketOrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

struct OrderContext<'a> {
    account: &'a AccountInfo,
    contract: &'a ContractSuggestion,
}

fn resolve_order_context<'a>(session: &'a SessionState) -> Result<OrderContext<'a>> {
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

async fn sync_native_protection(
    session: &mut SessionState,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: &str,
) -> Result<Option<String>> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    let key = StrategyProtectionKey {
        account_id: account.id,
        contract_id: contract.id,
    };
    let take_profit_price = sanitize_price(take_profit_price);
    let stop_price = sanitize_price(stop_price);

    if signed_qty == 0 || (take_profit_price.is_none() && stop_price.is_none()) {
        let cleared = cancel_strategy_protection_by_key(session, key, &account, &contract).await?;
        if cleared {
            return Ok(Some(format!(
                "Native protection cleared for {} on {} ({reason})",
                contract.name, account.name
            )));
        }
        return Ok(None);
    }

    let exit_action = if signed_qty > 0 { "Sell" } else { "Buy" };
    let order_qty = signed_qty.abs().max(1);

    refresh_managed_protection_order_ids(session, key);
    if let Some(existing) = session.managed_protection.get(&key).cloned() {
        let same_position = existing.signed_qty == signed_qty;
        let same_take_profit = prices_match(existing.take_profit_price, take_profit_price);
        let same_stop = prices_match(existing.stop_price, stop_price);
        if same_position && same_take_profit && same_stop {
            return Ok(None);
        }

        if same_position
            && same_take_profit
            && stop_price.is_some()
            && existing.stop_order_id.is_some()
            && existing.take_profit_price.is_some() == take_profit_price.is_some()
        {
            let stop_order_id = existing.stop_order_id.expect("checked is_some");
            let next_stop_price = stop_price.expect("checked is_some");
            modify_native_stop_order(session, stop_order_id, order_qty, next_stop_price).await?;
            if let Some(state) = session.managed_protection.get_mut(&key) {
                state.stop_price = Some(next_stop_price);
            }
            return Ok(Some(format!(
                "Native stop updated to {:.2} on {} ({reason})",
                next_stop_price, contract.name
            )));
        }
    }

    cancel_strategy_protection_by_key(session, key, &account, &contract).await?;

    let tp_cl_ord_id = take_profit_price.map(|_| next_strategy_cl_ord_id(session, "tp"));
    let stop_cl_ord_id = stop_price.map(|_| next_strategy_cl_ord_id(session, "sl"));

    let (take_profit_order_id, stop_order_id, action_label) = match (take_profit_price, stop_price)
    {
        (Some(tp), Some(stop)) => {
            let ids = place_native_oco_orders(
                session,
                &account,
                &contract,
                exit_action,
                order_qty,
                tp,
                tp_cl_ord_id.as_deref(),
                stop,
                stop_cl_ord_id.as_deref(),
            )
            .await?;
            (ids.0, ids.1, "TP/SL")
        }
        (Some(tp), None) => {
            let order_id = place_native_limit_order(
                session,
                &account,
                &contract,
                exit_action,
                order_qty,
                tp,
                tp_cl_ord_id.as_deref(),
            )
            .await?;
            (order_id, None, "TP")
        }
        (None, Some(stop)) => {
            let order_id = place_native_stop_order(
                session,
                &account,
                &contract,
                exit_action,
                order_qty,
                stop,
                stop_cl_ord_id.as_deref(),
            )
            .await?;
            (None, order_id, "SL")
        }
        (None, None) => unreachable!("checked above"),
    };

    session.managed_protection.insert(
        key,
        ManagedProtectionOrders {
            signed_qty,
            take_profit_price,
            stop_price,
            take_profit_cl_ord_id: tp_cl_ord_id,
            stop_cl_ord_id,
            take_profit_order_id,
            stop_order_id,
        },
    );

    let mut parts = Vec::new();
    if let Some(price) = take_profit_price {
        parts.push(format!("tp {:.2}", price));
    }
    if let Some(price) = stop_price {
        parts.push(format!("sl {:.2}", price));
    }
    Ok(Some(format!(
        "Native {action_label} protection live for {} on {}: {} ({reason})",
        contract.name,
        account.name,
        parts.join(", ")
    )))
}

fn ensure_no_market_order_submit_in_flight(session: &SessionState) -> Result<()> {
    if session.order_submit_in_flight {
        bail!("order submission already in flight");
    }
    Ok(())
}

fn build_market_order_request(
    session: &mut SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    order_action: &str,
    order_qty: i32,
    action_label: &str,
    automated: bool,
    reason_suffix: Option<&str>,
    target_qty: Option<i32>,
) -> PendingMarketOrder {
    let cl_ord_id = next_strategy_cl_ord_id(session, "entry");
    let payload = with_cl_ord_id(
        json!({
            "accountSpec": account.name,
            "accountId": account.id,
            "action": order_action,
            "symbol": contract.name,
            "orderQty": order_qty,
            "orderType": "Market",
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": automated
        }),
        Some(cl_ord_id.as_str()),
    );

    PendingMarketOrder {
        cl_ord_id,
        payload,
        action_label: action_label.to_string(),
        order_action: order_action.to_string(),
        order_qty,
        contract_name: contract.name.clone(),
        account_name: account.name.clone(),
        reason_suffix: reason_suffix.map(ToString::to_string),
        target_qty,
    }
}

fn enqueue_market_order(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    order: PendingMarketOrder,
) -> Result<()> {
    session.order_submit_in_flight = true;
    session.order_latency_tracker = Some(OrderLatencyTracker {
        started_at: time::Instant::now(),
        cl_ord_id: order.cl_ord_id.clone(),
        order_id: None,
        seen_recorded: false,
        exec_report_recorded: false,
        fill_recorded: false,
    });
    let request_tx = session.request_tx.clone();
    if broker_tx.send(BrokerCommand { request_tx, order }).is_err() {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        bail!("broker gateway is closed");
    }
    Ok(())
}

async fn cancel_strategy_protection_for_selected(session: &mut SessionState) -> Result<bool> {
    let order_ctx = resolve_order_context(session)?;
    let account = order_ctx.account.clone();
    let contract = order_ctx.contract.clone();
    let key = StrategyProtectionKey {
        account_id: account.id,
        contract_id: contract.id,
    };
    cancel_strategy_protection_by_key(session, key, &account, &contract).await
}

async fn cancel_strategy_protection_by_key(
    session: &mut SessionState,
    key: StrategyProtectionKey,
    _account: &AccountInfo,
    _contract: &ContractSuggestion,
) -> Result<bool> {
    refresh_managed_protection_order_ids(session, key);
    let existing = session.managed_protection.remove(&key);

    let session = &*session;

    // Collect ALL active strategy orders for this contract from the user store.
    // This catches orphaned orders whose IDs we never captured from the OCO
    // placement response (the stop leg ID often arrives later via WebSocket).
    let orphan_ids: Vec<i64> = session
        .user_store
        .orders
        .get(&key.account_id)
        .into_iter()
        .flat_map(|orders| orders.iter())
        .filter(|(_, order)| {
            order_is_active(order)
                && order_contract_id(order) == Some(key.contract_id)
                && order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .is_some_and(|id| id.starts_with("midas-"))
        })
        .filter_map(|(id, _)| Some(*id))
        .collect();

    let mut cancelled = false;
    for order_id in &orphan_ids {
        if cancel_order_if_active(session, key.account_id, *order_id).await? {
            cancelled = true;
        }
    }

    // Also cancel by tracked order IDs in case they haven't propagated to
    // user_store.orders yet (covers the window between placement and the
    // first WebSocket entity sync).
    if let Some(existing) = existing {
        let tracked_ids: Vec<i64> = [existing.stop_order_id, existing.take_profit_order_id]
            .into_iter()
            .flatten()
            .filter(|id| !orphan_ids.contains(id))
            .collect();
        for order_id in tracked_ids {
            if cancel_order_if_active(session, key.account_id, order_id).await? {
                cancelled = true;
            }
        }
    }

    Ok(cancelled)
}

fn refresh_managed_protection_order_ids(session: &mut SessionState, key: StrategyProtectionKey) {
    let Some(state) = session.managed_protection.get_mut(&key) else {
        return;
    };
    if state.take_profit_order_id.is_none() {
        if let Some(cl_ord_id) = state.take_profit_cl_ord_id.as_deref() {
            state.take_profit_order_id = session
                .user_store
                .order_id_by_client_id(key.account_id, cl_ord_id);
        }
    }
    if state.stop_order_id.is_none() {
        if let Some(cl_ord_id) = state.stop_cl_ord_id.as_deref() {
            state.stop_order_id = session
                .user_store
                .order_id_by_client_id(key.account_id, cl_ord_id);
        }
    }
}

fn next_strategy_cl_ord_id(session: &mut SessionState, suffix: &str) -> String {
    let nonce = session.next_strategy_order_nonce;
    session.next_strategy_order_nonce = session.next_strategy_order_nonce.saturating_add(1);
    let ts = Utc::now().timestamp_millis();
    format!("midas-{ts}-{nonce}-{suffix}")
}

async fn place_native_limit_order(
    session: &SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    action: &str,
    order_qty: i32,
    price: f64,
    cl_ord_id: Option<&str>,
) -> Result<Option<i64>> {
    let payload = with_cl_ord_id(
        json!({
        "accountSpec": account.name,
        "accountId": account.id,
        "action": action,
        "symbol": contract.name,
        "orderQty": order_qty,
        "orderType": "Limit",
        "price": price,
        "timeInForce": session.cfg.time_in_force,
        "isAutomated": true,
        }),
        cl_ord_id,
    );
    let parsed = request_order_json(&session.request_tx, "order/placeorder", &payload).await?;
    Ok(first_known_order_id(&parsed))
}

async fn place_native_stop_order(
    session: &SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    action: &str,
    order_qty: i32,
    stop_price: f64,
    cl_ord_id: Option<&str>,
) -> Result<Option<i64>> {
    let payload = with_cl_ord_id(
        json!({
        "accountSpec": account.name,
        "accountId": account.id,
        "action": action,
        "symbol": contract.name,
        "orderQty": order_qty,
        "orderType": "Stop",
        "stopPrice": stop_price,
        "timeInForce": session.cfg.time_in_force,
        "isAutomated": true,
        }),
        cl_ord_id,
    );
    let parsed = request_order_json(&session.request_tx, "order/placeorder", &payload).await?;
    Ok(first_known_order_id(&parsed))
}

async fn place_native_oco_orders(
    session: &SessionState,
    account: &AccountInfo,
    contract: &ContractSuggestion,
    action: &str,
    order_qty: i32,
    take_profit_price: f64,
    take_profit_cl_ord_id: Option<&str>,
    stop_price: f64,
    stop_cl_ord_id: Option<&str>,
) -> Result<(Option<i64>, Option<i64>)> {
    let mut payload = with_cl_ord_id(
        json!({
        "accountSpec": account.name,
        "accountId": account.id,
        "action": action,
        "symbol": contract.name,
        "orderQty": order_qty,
        "orderType": "Limit",
        "price": take_profit_price,
        "timeInForce": session.cfg.time_in_force,
        "isAutomated": true,
        "other": {
            "accountSpec": account.name,
            "accountId": account.id,
            "action": action,
            "symbol": contract.name,
            "orderQty": order_qty,
            "orderType": "Stop",
            "stopPrice": stop_price,
            "timeInForce": session.cfg.time_in_force,
            "isAutomated": true,
        }
        }),
        take_profit_cl_ord_id,
    );
    if let Some(other) = payload.get_mut("other").and_then(Value::as_object_mut) {
        if let Some(cl_ord_id) = stop_cl_ord_id {
            other.insert("clOrdId".to_string(), Value::String(cl_ord_id.to_string()));
        }
    }
    let parsed = request_order_json(&session.request_tx, "order/placeOCO", &payload).await?;
    Ok((
        first_known_order_id(&parsed),
        known_order_id(&parsed, &["otherId", "stopOrderId"]),
    ))
}

async fn modify_native_stop_order(
    session: &SessionState,
    order_id: i64,
    order_qty: i32,
    stop_price: f64,
) -> Result<()> {
    let payload = json!({
        "orderId": order_id,
        "orderQty": order_qty,
        "orderType": "Stop",
        "stopPrice": stop_price,
        "timeInForce": session.cfg.time_in_force,
        "isAutomated": true,
    });
    let _ = request_order_json(&session.request_tx, "order/modifyorder", &payload).await?;
    Ok(())
}

async fn cancel_order_if_active(
    session: &SessionState,
    account_id: i64,
    order_id: i64,
) -> Result<bool> {
    if let Some(orders) = session.user_store.orders.get(&account_id) {
        if let Some(order) = orders.get(&order_id) {
            if !order_is_active(order) {
                return Ok(false);
            }
        }
    }

    let payload = json!({
        "orderId": order_id,
        "isAutomated": true,
    });
    match request_order_json(&session.request_tx, "order/cancelorder", &payload).await {
        Ok(_) => Ok(true),
        Err(err) => {
            let msg = err.to_string();
            if msg.contains("TooLate") || msg.contains("Already cancelled") {
                Ok(false)
            } else {
                Err(err)
            }
        }
    }
}

async fn request_order_json(
    request_tx: &UnboundedSender<UserSocketCommand>,
    endpoint: &str,
    payload: &Value,
) -> Result<Value> {
    let (response_tx, response_rx) = oneshot::channel();
    request_tx
        .send(UserSocketCommand {
            endpoint: endpoint.to_string(),
            query: None,
            body: Some(payload.clone()),
            response_tx,
        })
        .map_err(|_| anyhow::anyhow!("user websocket request channel is closed"))?;

    let parsed = response_rx
        .await
        .map_err(|_| anyhow::anyhow!("user websocket response channel was dropped"))?
        .map_err(anyhow::Error::msg)?;
    if let Some(failure) = parsed.get("failureReason").and_then(Value::as_str) {
        if !failure.trim().is_empty() {
            bail!("{endpoint} rejected: {failure}");
        }
    }
    if let Some(err_text) = parsed.get("errorText").and_then(Value::as_str) {
        if !err_text.trim().is_empty() {
            bail!("{endpoint} errorText: {err_text}");
        }
    }
    Ok(parsed)
}

fn update_latency_from_envelope(
    session: &mut SessionState,
    latency: &mut LatencySnapshot,
    envelope: &EntityEnvelope,
) -> bool {
    if envelope.deleted {
        return false;
    }
    let Some(tracker) = session.order_latency_tracker.as_mut() else {
        return false;
    };

    match envelope.entity_type.to_ascii_lowercase().as_str() {
        "order" => {
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.seen_recorded {
                return false;
            }
            tracker.seen_recorded = true;
            latency.last_order_seen_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            true
        }
        "executionreport" => {
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.exec_report_recorded {
                return false;
            }
            tracker.exec_report_recorded = true;
            latency.last_exec_report_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            true
        }
        "fill" => {
            if !tracker_matches_entity(tracker, &envelope.entity) || tracker.fill_recorded {
                return false;
            }
            tracker.fill_recorded = true;
            latency.last_fill_ms = Some(tracker.started_at.elapsed().as_millis() as u64);
            true
        }
        _ => false,
    }
}

fn tracker_matches_entity(tracker: &mut OrderLatencyTracker, entity: &Value) -> bool {
    let entity_cl_ord_id = entity.get("clOrdId").and_then(Value::as_str);
    let entity_order_id = json_i64(entity, "orderId").or_else(|| json_i64(entity, "id"));

    if let Some(expected_order_id) = tracker.order_id {
        if entity_order_id == Some(expected_order_id) {
            return true;
        }
    }

    if entity_cl_ord_id.is_some_and(|value| value == tracker.cl_ord_id) {
        if tracker.order_id.is_none() {
            tracker.order_id = entity_order_id;
        }
        return true;
    }

    false
}

fn record_trade_marker(session: &mut SessionState, marker: TradeMarker) -> bool {
    if let Some(fill_id) = marker.fill_id {
        if session
            .market
            .trade_markers
            .iter()
            .any(|existing| existing.fill_id == Some(fill_id))
        {
            return false;
        }
    }

    session.market.trade_markers.push(marker);
    if session.market.trade_markers.len() > 200 {
        let overflow = session.market.trade_markers.len() - 200;
        session.market.trade_markers.drain(0..overflow);
    }
    true
}

fn trade_marker_from_fill(session: &SessionState, fill: &Value) -> Option<TradeMarker> {
    let fill_id = extract_entity_id(fill)?;
    let account_id = extract_account_id("fill", fill)?;
    let order_id = json_i64(fill, "orderId");
    let order = order_id.and_then(|order_id| session.user_store.find_order(account_id, order_id));
    let side = order
        .and_then(trade_side_from_order)
        .or_else(|| trade_side_from_fill(fill))?;
    let price = pick_number(fill, &["price", "fillPrice", "lastPrice", "avgPrice"])?;
    let qty = pick_number(fill, &["qty", "fillQty", "lastQty", "quantity"])?
        .abs()
        .round() as i32;
    if qty <= 0 {
        return None;
    }

    let ts_ns = json_timestamp_ns(fill, &["timestamp", "fillTime", "createdTime"])
        .or_else(|| session.market.bars.last().map(|bar| bar.ts_ns))?;
    let contract_id = json_i64(fill, "contractId")
        .or_else(|| {
            fill.get("contract")
                .and_then(|contract| json_i64(contract, "id"))
        })
        .or_else(|| order.and_then(order_contract_id));
    let contract_name = fill
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| fill.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| fill.get("name").and_then(Value::as_str))
        .or_else(|| order.and_then(order_symbol))
        .map(ToString::to_string);

    Some(TradeMarker {
        fill_id: Some(fill_id),
        account_id: Some(account_id),
        contract_id,
        contract_name,
        ts_ns,
        price,
        qty,
        side,
    })
}

fn trade_side_from_order(order: &Value) -> Option<TradeMarkerSide> {
    order
        .get("action")
        .and_then(Value::as_str)
        .and_then(trade_side_from_text)
}

fn trade_side_from_fill(fill: &Value) -> Option<TradeMarkerSide> {
    ["buySell", "side", "action"]
        .iter()
        .find_map(|key| fill.get(*key).and_then(Value::as_str))
        .and_then(trade_side_from_text)
}

fn trade_side_from_text(value: &str) -> Option<TradeMarkerSide> {
    match value.trim().to_ascii_lowercase().as_str() {
        "buy" | "bot" | "b" | "long" => Some(TradeMarkerSide::Buy),
        "sell" | "sld" | "s" | "short" => Some(TradeMarkerSide::Sell),
        _ => None,
    }
}

fn order_contract_id(order: &Value) -> Option<i64> {
    json_i64(order, "contractId").or_else(|| {
        order
            .get("contract")
            .and_then(|contract| json_i64(contract, "id"))
    })
}

fn order_symbol(order: &Value) -> Option<&str> {
    order
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| order.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| order.get("name").and_then(Value::as_str))
        .or_else(|| {
            order
                .get("contract")
                .and_then(|contract| contract.get("name"))
                .and_then(Value::as_str)
        })
}

fn json_timestamp_ns(value: &Value, keys: &[&str]) -> Option<i64> {
    for key in keys {
        let Some(raw) = value.get(*key) else {
            continue;
        };
        if let Some(timestamp) = raw.as_i64().and_then(normalize_unix_timestamp_ns) {
            return Some(timestamp);
        }
        if let Some(text) = raw.as_str() {
            if let Some(timestamp) = parse_bar_timestamp_ns(text) {
                return Some(timestamp);
            }
            if let Ok(parsed) = text.parse::<i64>() {
                if let Some(timestamp) = normalize_unix_timestamp_ns(parsed) {
                    return Some(timestamp);
                }
            }
        }
    }
    None
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

#[derive(Debug, Clone, Copy)]
struct MarketSpecs {
    session_profile: Option<InstrumentSessionProfile>,
    value_per_point: Option<f64>,
    tick_size: Option<f64>,
}

async fn fetch_contract_specs(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract: &ContractSuggestion,
) -> Result<MarketSpecs> {
    let contract_maturity_id = json_i64(&contract.raw, "contractMaturityId")
        .context("selected contract is missing contractMaturityId")?;
    let maturity_url = format!("{}/contractMaturity/item", env.rest_url());
    let maturity_response = client
        .get(&maturity_url)
        .bearer_auth(token)
        .query(&[("id", contract_maturity_id.to_string())])
        .send()
        .await?;
    let maturity_status = maturity_response.status();
    let maturity_body = maturity_response.text().await.unwrap_or_default();
    if !maturity_status.is_success() {
        bail!("contractMaturity/item failed ({maturity_status}): {maturity_body}");
    }
    let maturity: Value = serde_json::from_str(&maturity_body)?;
    let product_id =
        json_i64(&maturity, "productId").context("contractMaturity/item missing productId")?;

    let product_url = format!("{}/product/item", env.rest_url());
    let product_response = client
        .get(&product_url)
        .bearer_auth(token)
        .query(&[("id", product_id.to_string())])
        .send()
        .await?;
    let product_status = product_response.status();
    let product_body = product_response.text().await.unwrap_or_default();
    if !product_status.is_success() {
        bail!("product/item failed ({product_status}): {product_body}");
    }
    let product: Value = serde_json::from_str(&product_body)?;
    Ok(MarketSpecs {
        session_profile: Some(infer_session_profile(&product)),
        value_per_point: json_number(&product, "valuePerPoint"),
        tick_size: json_number(&product, "tickSize").or_else(|| json_number(&product, "minTick")),
    })
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
    for entity in [
        "account",
        "accountRiskStatus",
        "cashBalance",
        "position",
        "order",
        "fill",
    ] {
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

/// Create a low-latency WebSocket connection with TCP_NODELAY and TCP_QUICKACK (Linux).
async fn connect_low_latency_ws(
    url: &str,
    ws_config: WebSocketConfig,
) -> Result<(
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    tokio_tungstenite::tungstenite::http::Response<Option<Vec<u8>>>,
)> {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let request = url.into_client_request()?;
    let host = request
        .uri()
        .host()
        .ok_or_else(|| anyhow::anyhow!("no host in URL"))?;
    let port = request.uri().port_u16().unwrap_or(443);

    let addr = format!("{host}:{port}")
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| anyhow::anyhow!("DNS resolution failed for {host}"))?;

    let socket = socket2::Socket::new(
        socket2::Domain::for_address(addr),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    socket.set_nodelay(true)?;
    socket.set_nonblocking(true)?;

    // Minimize socket buffer sizes to reduce kernel-side latency
    let _ = socket.set_recv_buffer_size(64 * 1024);
    let _ = socket.set_send_buffer_size(64 * 1024);

    // Start non-blocking connect (returns EINPROGRESS)
    let _ = socket.connect(&addr.into());

    let std_stream: std::net::TcpStream = socket.into();
    let tcp_stream = tokio::net::TcpStream::from_std(std_stream)?;
    // Wait for the async connect to complete
    tcp_stream.writable().await?;

    // Re-apply TCP_NODELAY after connect (some OS reset it)
    tcp_stream.set_nodelay(true)?;

    // Linux: disable delayed ACKs
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let fd = tcp_stream.as_raw_fd();
        unsafe {
            let val: i32 = 1;
            libc::setsockopt(
                fd,
                libc::IPPROTO_TCP,
                libc::TCP_QUICKACK,
                &val as *const _ as *const libc::c_void,
                std::mem::size_of::<i32>() as libc::socklen_t,
            );
        }
    }

    let (ws, resp) =
        tokio_tungstenite::client_async_tls_with_config(request, tcp_stream, Some(ws_config), None)
            .await?;
    Ok((ws, resp))
}

async fn user_sync_worker(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    request_rx: UnboundedReceiver<UserSocketCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) =
        user_sync_worker_inner(cfg, tokens, account_ids, request_rx, internal_tx.clone()).await
    {
        let _ = internal_tx.send(InternalEvent::Error(format!("user sync: {err}")));
    }
}

async fn user_sync_worker_inner(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    mut request_rx: UnboundedReceiver<UserSocketCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.user_ws_url(), ws_config)
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
    let mut pending_requests = HashMap::<u64, oneshot::Sender<Result<Value, String>>>::new();
    let mut heartbeat = time::interval(Duration::from_millis(cfg.heartbeat_ms.max(250)));
    heartbeat.tick().await;

    loop {
        tokio::select! {
            biased;

            outbound = request_rx.recv() => {
                let Some(outbound) = outbound else {
                    break;
                };
                if !authorized {
                    let _ = outbound
                        .response_tx
                        .send(Err("user websocket is not authorized".to_string()));
                    continue;
                }

                message_id += 1;
                let request_id = message_id;
                pending_requests.insert(request_id, outbound.response_tx);
                let frame = create_message(
                    &outbound.endpoint,
                    request_id,
                    outbound.query.as_deref(),
                    outbound.body.as_ref(),
                );
                if let Err(err) = write.send(Message::Text(frame)).await {
                    if let Some(response_tx) = pending_requests.remove(&request_id) {
                        let _ = response_tx.send(Err(format!(
                            "user websocket send error: {err}"
                        )));
                    }
                    bail!("user websocket send error: {err}");
                }
            }
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

                    if !authorized && response_id == Some(authorize_id) {
                        let Some(status) = status else {
                            continue;
                        };
                        if status != 200 {
                            bail!("user websocket authorize failed ({status})");
                        }

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
                                None,
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

                    if let Some(request_id) = response_id {
                        if let Some(response_tx) = pending_requests.remove(&request_id) {
                            let _ = response_tx.send(parse_socket_response(&item));
                            continue;
                        }
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
    market_specs: Option<MarketSpecs>,
    bar_type: BarType,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    if let Err(err) = market_data_worker_inner(
        cfg,
        access_token,
        contract,
        market_specs,
        bar_type,
        internal_tx.clone(),
    )
    .await
    {
        let _ = internal_tx.send(InternalEvent::Error(format!("market data: {err}")));
    }
}

async fn market_data_worker_inner(
    cfg: AppConfig,
    access_token: String,
    contract: ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    bar_type: BarType,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
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
                let before_closed_len = series.closed_bars.len();
                let before_last_closed = series.closed_bars.last().cloned();
                let before_forming = series.forming_bar.clone();

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);

                    if !authorized && status == Some(200) && response_id == Some(authorize_id) {
                        authorized = true;
                        message_id += 1;
                        chart_req_id = Some(message_id);
                        let body = json!({
                            "symbol": contract.id,
                            "chartDescription": bar_type.chart_description(),
                            "timeRange": {
                                "asMuchAsElements": cfg.history_bars
                            }
                        });
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                message_id,
                                None,
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
                }

                if let Some(update) = build_market_update(
                    &contract,
                    market_specs,
                    series.closed_bars.len(),
                    live_bars,
                    format!("Subscribed to {} bars for {}", bar_type.label(), contract.name),
                    before_closed_len,
                    before_last_closed,
                    before_forming,
                    &series,
                ) {
                    let _ = internal_tx.send(InternalEvent::Market(update));
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
            "order" => {
                let Some(account_id) = extract_account_id("order", &envelope.entity) else {
                    return;
                };
                let bucket = self.orders.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
                }
            }
            "fill" => {
                let Some(account_id) = extract_account_id("fill", &envelope.entity) else {
                    return;
                };
                let bucket = self.fills.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
                }
            }
            _ => {}
        }
    }

    fn build_snapshots(
        &self,
        accounts: &[AccountInfo],
        market: Option<&MarketSnapshot>,
        managed_protection: &BTreeMap<StrategyProtectionKey, ManagedProtectionOrders>,
    ) -> Vec<AccountSnapshot> {
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
                let realized_pnl = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "realizedPnL",
                                "realizedPnl",
                                "realizedProfitAndLoss",
                                "realizedProfitLoss",
                                "sessionRealizedPnL",
                                "sessionRealizedPnl",
                                "todayRealizedPnL",
                                "todayRealizedPnl",
                                "closedPnL",
                                "closedPnl",
                                "dayPnL",
                                "dayPnl",
                                "dailyPnL",
                                "dailyPnl",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_account.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "realizedPnL",
                                    "realizedPnl",
                                    "realizedProfitAndLoss",
                                    "realizedProfitLoss",
                                    "sessionRealizedPnL",
                                    "sessionRealizedPnl",
                                    "todayRealizedPnL",
                                    "todayRealizedPnl",
                                    "closedPnL",
                                    "closedPnl",
                                ],
                            )
                        })
                    })
                    .or_else(|| {
                        raw_cash.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "realizedPnL",
                                    "realizedPnl",
                                    "sessionRealizedPnL",
                                    "sessionRealizedPnl",
                                    "todayRealizedPnL",
                                    "todayRealizedPnl",
                                ],
                            )
                        })
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
                let unrealized_pnl = unrealized_pnl.or_else(|| {
                    market.and_then(|market| fallback_unrealized_pnl(&raw_positions, market))
                });
                let net_liq = net_liq.or_else(|| match (balance, unrealized_pnl) {
                    (Some(balance), Some(unrealized)) => Some(balance + unrealized),
                    _ => None,
                });
                let open_position_qty = sum_position_metric(
                    &raw_positions,
                    &["netPos", "netPosition", "qty", "quantity", "netQty"],
                );
                let market_position_qty = market.and_then(|market| {
                    let values = raw_positions
                        .iter()
                        .filter(|position| position_matches_market(position, market))
                        .filter_map(position_qty)
                        .collect::<Vec<_>>();
                    if values.is_empty() {
                        None
                    } else {
                        Some(values.iter().sum())
                    }
                });
                let market_entry_price =
                    market.and_then(|market| weighted_market_entry_price(&raw_positions, market));
                let (selected_contract_take_profit_price, selected_contract_stop_price) = market
                    .and_then(|market| {
                        market.contract_id.map(|contract_id| StrategyProtectionKey {
                            account_id: account.id,
                            contract_id,
                        })
                    })
                    .and_then(|key| managed_protection.get(&key))
                    .map(|orders| (orders.take_profit_price, orders.stop_price))
                    .unwrap_or((None, None));

                AccountSnapshot {
                    account_id: account.id,
                    account_name: account.name.clone(),
                    balance,
                    cash_balance,
                    net_liq,
                    realized_pnl,
                    unrealized_pnl,
                    intraday_margin,
                    open_position_qty,
                    market_position_qty,
                    market_entry_price,
                    selected_contract_take_profit_price,
                    selected_contract_stop_price,
                    raw_account,
                    raw_risk,
                    raw_cash,
                    raw_positions,
                }
            })
            .collect()
    }

    fn find_order(&self, account_id: i64, order_id: i64) -> Option<&Value> {
        self.orders.get(&account_id)?.get(&order_id)
    }

    fn contract_position_qty(&self, account_id: i64, contract: &ContractSuggestion) -> Option<f64> {
        let values = self
            .positions
            .get(&account_id)
            .into_iter()
            .flat_map(|positions| positions.values())
            .filter(|position| position_matches_contract(position, contract))
            .filter_map(position_qty)
            .collect::<Vec<_>>();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum())
        }
    }

    fn order_id_by_client_id(&self, account_id: i64, cl_ord_id: &str) -> Option<i64> {
        self.orders
            .get(&account_id)
            .into_iter()
            .flat_map(|orders| orders.values())
            .find(|order| {
                order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .map(|value| value == cl_ord_id)
                    .unwrap_or(false)
            })
            .and_then(extract_entity_id)
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

fn fallback_unrealized_pnl(positions: &[Value], market: &MarketSnapshot) -> Option<f64> {
    let last_close = market.bars.last().map(|bar| bar.close)?;
    let value_per_point = market.value_per_point?;
    let values = positions
        .iter()
        .filter(|position| position_matches_market(position, market))
        .filter_map(|position| {
            let qty = position_qty(position)?;
            let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
            Some((last_close - entry_price) * qty * value_per_point)
        })
        .collect::<Vec<_>>();

    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum())
    }
}

fn weighted_market_entry_price(positions: &[Value], market: &MarketSnapshot) -> Option<f64> {
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in positions
        .iter()
        .filter(|position| position_matches_market(position, market))
    {
        let qty = position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }

    if total_qty <= f64::EPSILON {
        None
    } else {
        Some(weighted_sum / total_qty)
    }
}

fn position_matches_contract(position: &Value, contract: &ContractSuggestion) -> bool {
    let contract_id_match = position_contract_id(position) == Some(contract.id);
    let contract_maturity_match = json_i64(&contract.raw, "contractMaturityId")
        .zip(position_contract_maturity_id(position))
        .is_some_and(|(expected, actual)| expected == actual);
    let symbol_match =
        position_symbol(position).is_some_and(|symbol| symbol.eq_ignore_ascii_case(&contract.name));

    contract_id_match || contract_maturity_match || symbol_match
}

fn position_matches_market(position: &Value, market: &MarketSnapshot) -> bool {
    let contract_id_match = market
        .contract_id
        .is_some_and(|contract_id| position_contract_id(position) == Some(contract_id));
    let symbol_match = market
        .contract_name
        .as_deref()
        .zip(position_symbol(position))
        .is_some_and(|(expected, actual)| actual.eq_ignore_ascii_case(expected));
    contract_id_match || symbol_match
}

fn position_qty(position: &Value) -> Option<f64> {
    pick_number(
        position,
        &["netPos", "netPosition", "qty", "quantity", "netQty"],
    )
}

fn position_contract_id(position: &Value) -> Option<i64> {
    json_i64(position, "contractId").or_else(|| {
        position
            .get("contract")
            .and_then(|contract| json_i64(contract, "id"))
    })
}

fn position_contract_maturity_id(position: &Value) -> Option<i64> {
    json_i64(position, "contractMaturityId").or_else(|| {
        position
            .get("contract")
            .and_then(|contract| json_i64(contract, "contractMaturityId"))
    })
}

fn position_symbol(position: &Value) -> Option<&str> {
    position
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| position.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| position.get("name").and_then(Value::as_str))
        .or_else(|| {
            position
                .get("contract")
                .and_then(|contract| contract.get("name"))
                .and_then(Value::as_str)
        })
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

    for key in [
        "account",
        "accountRiskStatus",
        "cashBalance",
        "position",
        "order",
        "executionReport",
        "fill",
    ] {
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

fn create_message(endpoint: &str, id: u64, query: Option<&str>, body: Option<&Value>) -> String {
    match (query, body) {
        (Some(query), Some(body)) => format!("{endpoint}\n{id}\n{query}\n{body}"),
        (Some(query), None) => format!("{endpoint}\n{id}\n{query}"),
        (None, Some(body)) => format!("{endpoint}\n{id}\n\n{body}"),
        (None, None) => format!("{endpoint}\n{id}\n\n"),
    }
}

fn parse_socket_response(message: &Value) -> Result<Value, String> {
    let Some(status) = parse_status_code(message) else {
        return Err("websocket response missing status code".to_string());
    };
    let payload = message.get("d").cloned().unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        Ok(payload)
    } else if let Some(text) = payload.as_str() {
        Err(format!("websocket request failed ({status}): {text}"))
    } else {
        Err(format!("websocket request failed ({status}): {payload}"))
    }
}

fn parse_bar(value: &Value) -> Option<Bar> {
    let ts = value.get("timestamp")?.as_str()?;
    let ts_ns = parse_bar_timestamp_ns(ts)?;
    Some(Bar {
        ts_ns,
        open: json_number(value, "open")?,
        high: json_number(value, "high")?,
        low: json_number(value, "low")?,
        close: json_number(value, "close")?,
    })
}

fn parse_bar_timestamp_ns(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::DateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M%:z")
                .map(|dt| dt.with_timezone(&Utc))
        })
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%MZ").map(|dt| dt.and_utc())
        })
        .ok()?
        .timestamp_nanos_opt()
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

fn sanitize_price(price: Option<f64>) -> Option<f64> {
    price.filter(|value| value.is_finite() && *value > 0.0)
}

fn with_cl_ord_id(mut payload: Value, cl_ord_id: Option<&str>) -> Value {
    if let Some(cl_ord_id) = cl_ord_id {
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("clOrdId".to_string(), Value::String(cl_ord_id.to_string()));
        }
    }
    payload
}

fn prices_match(lhs: Option<f64>, rhs: Option<f64>) -> bool {
    match (lhs, rhs) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn known_order_id(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| json_i64(value, key))
}

fn first_known_order_id(value: &Value) -> Option<i64> {
    known_order_id(value, &["orderId", "id", "otherId", "stopOrderId"])
}

fn order_is_active(order: &Value) -> bool {
    let Some(status) = order
        .get("ordStatus")
        .and_then(Value::as_str)
        .or_else(|| order.get("status").and_then(Value::as_str))
    else {
        return true;
    };

    !matches!(
        status.to_ascii_lowercase().as_str(),
        "filled" | "cancelled" | "canceled" | "rejected" | "expired" | "stopped" | "finished"
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn parse_bar_accepts_minute_precision_utc_timestamp() {
        let bar = parse_bar(&json!({
            "timestamp": "2026-03-11T22:38Z",
            "open": 6738.5,
            "high": 6739.5,
            "low": 6736.75,
            "close": 6738.0
        }))
        .expect("bar should parse");

        let expected_ts = chrono::DateTime::parse_from_rfc3339("2026-03-11T22:38:00Z")
            .unwrap()
            .with_timezone(&Utc)
            .timestamp_nanos_opt()
            .unwrap();

        assert_eq!(bar.ts_ns, expected_ts);
        assert_eq!(bar.close, 6738.0);
    }

    #[test]
    fn contract_position_qty_matches_selected_contract() {
        let mut store = UserSyncStore::default();
        store.positions.insert(
            42,
            BTreeMap::from([
                (
                    1,
                    json!({
                        "id": 1,
                        "accountId": 42,
                        "contractId": 3570918,
                        "netPos": 2
                    }),
                ),
                (
                    2,
                    json!({
                        "id": 2,
                        "accountId": 42,
                        "symbol": "ESM6",
                        "netPos": -1
                    }),
                ),
            ]),
        );

        let contract = ContractSuggestion {
            id: 3570918,
            name: "ESH6".to_string(),
            description: String::new(),
            raw: json!({ "contractMaturityId": 53951 }),
        };

        assert_eq!(store.contract_position_qty(42, &contract), Some(2.0));
    }

    #[test]
    fn fallback_unrealized_pnl_uses_latest_close_and_value_per_point() {
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 6725.0,
                high: 6730.0,
                low: 6724.0,
                close: 6727.25,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };

        let positions = vec![json!({
            "accountId": 42,
            "contractId": 3570918,
            "netPos": 1,
            "netPrice": 6725.75
        })];

        assert_eq!(fallback_unrealized_pnl(&positions, &market), Some(75.0));
    }

    #[test]
    fn build_market_update_emits_snapshot_then_forming_delta() {
        let contract = ContractSuggestion {
            id: 3570918,
            name: "ESH6".to_string(),
            description: "ES Jun 2026".to_string(),
            raw: json!({ "id": 3570918 }),
        };
        let closed_bar = Bar {
            ts_ns: 1,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        };
        let forming_bar = Bar {
            ts_ns: 2,
            open: 5000.5,
            high: 5002.0,
            low: 5000.25,
            close: 5001.5,
        };
        let mut series = LiveSeries::new();
        series.closed_bars.push(closed_bar.clone());

        let initial = build_market_update(
            &contract,
            None,
            series.closed_bars.len(),
            0,
            "status".to_string(),
            0,
            None,
            None,
            &series,
        )
        .expect("initial snapshot should be emitted");
        assert!(matches!(
            initial.bars,
            MarketBarsUpdate::Snapshot {
                ref closed_bars,
                forming_bar: None
            } if closed_bars == &vec![closed_bar.clone()]
        ));

        let before_last_closed = series.closed_bars.last().cloned();
        series.forming_bar = Some(forming_bar.clone());
        let update = build_market_update(
            &contract,
            None,
            series.closed_bars.len(),
            0,
            "status".to_string(),
            series.closed_bars.len(),
            before_last_closed,
            None,
            &series,
        )
        .expect("forming update should be emitted");
        assert!(matches!(
            update.bars,
            MarketBarsUpdate::Forming { forming_bar: ref bar } if bar == &forming_bar
        ));
    }

    #[test]
    fn apply_market_update_keeps_bars_incremental() {
        let closed_bar = Bar {
            ts_ns: 1,
            open: 5000.0,
            high: 5001.0,
            low: 4999.0,
            close: 5000.5,
        };
        let forming_bar = Bar {
            ts_ns: 2,
            open: 5000.5,
            high: 5002.0,
            low: 5000.25,
            close: 5001.5,
        };
        let next_forming_bar = Bar {
            ts_ns: 3,
            open: 5001.5,
            high: 5003.0,
            low: 5001.0,
            close: 5002.5,
        };
        let marker = TradeMarker {
            fill_id: Some(7),
            account_id: Some(42),
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            ts_ns: 1,
            price: 5000.5,
            qty: 1,
            side: TradeMarkerSide::Buy,
        };
        let mut market = MarketSnapshot {
            trade_markers: vec![marker],
            ..MarketSnapshot::default()
        };

        let initial = MarketUpdate {
            contract_id: 3570918,
            contract_name: "ESH6".to_string(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: "initial".to_string(),
            bars: MarketBarsUpdate::Snapshot {
                closed_bars: vec![closed_bar.clone()],
                forming_bar: Some(forming_bar.clone()),
            },
        };
        assert!(apply_market_update(&mut market, initial));
        assert_eq!(market.history_loaded, 1);
        assert_eq!(market.bars, vec![closed_bar.clone(), forming_bar.clone()]);
        assert_eq!(market.trade_markers.len(), 1);

        let next = MarketUpdate {
            contract_id: 3570918,
            contract_name: "ESH6".to_string(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 2,
            live_bars: 1,
            status: "realtime".to_string(),
            bars: MarketBarsUpdate::Closed {
                closed_bar: forming_bar.clone(),
                forming_bar: Some(next_forming_bar.clone()),
            },
        };
        assert!(apply_market_update(&mut market, next));
        assert_eq!(market.history_loaded, 2);
        assert_eq!(market.bars, vec![closed_bar, forming_bar, next_forming_bar]);
        assert_eq!(market.trade_markers.len(), 1);
    }

    #[test]
    fn jwt_expiration_time_reads_exp_claim() {
        let token = "eyJhbGciOiJub25lIn0.eyJleHAiOjE4OTM0NTYwMDB9.sig";
        let expires_at = jwt_expiration_time(token).expect("jwt exp should parse");

        let expected = DateTime::<Utc>::from_timestamp(1_893_456_000, 0).unwrap();
        assert_eq!(expires_at, expected);
    }

    #[test]
    fn token_refresh_due_uses_jwt_exp_when_expiration_time_missing() {
        let tokens = TokenBundle {
            access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
            md_access_token: "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzM0MzYwNDR9.sig".to_string(),
            expiration_time: None,
            user_id: Some(42),
            user_name: Some("demo".to_string()),
        };

        let now = DateTime::<Utc>::from_timestamp(1_773_436_044 - 60, 0).unwrap();
        assert!(token_refresh_due(&tokens, now));
    }

    #[test]
    fn create_message_formats_body_only_requests_for_websocket() {
        let body = json!({
            "accountId": 42,
            "orderQty": 1
        });

        let actual = create_message("order/placeorder", 7, None, Some(&body));

        assert_eq!(
            actual,
            "order/placeorder\n7\n\n{\"accountId\":42,\"orderQty\":1}"
        );
    }

    #[test]
    fn parse_socket_response_maps_status_and_payload() {
        let ok = json!({
            "i": 3,
            "s": 200,
            "d": { "orderId": 99 }
        });
        let err = json!({
            "i": 4,
            "s": 400,
            "d": "bad request"
        });

        assert_eq!(
            parse_socket_response(&ok).expect("success payload"),
            json!({ "orderId": 99 })
        );
        assert_eq!(
            parse_socket_response(&err).expect_err("error payload"),
            "websocket request failed (400): bad request"
        );
    }

    #[test]
    fn tracker_matches_entity_binds_order_id_from_cl_ord_id() {
        let mut tracker = OrderLatencyTracker {
            started_at: time::Instant::now(),
            cl_ord_id: "midas-1-entry".to_string(),
            order_id: None,
            seen_recorded: false,
            exec_report_recorded: false,
            fill_recorded: false,
        };
        let entity = json!({
            "orderId": 42,
            "clOrdId": "midas-1-entry"
        });

        assert!(tracker_matches_entity(&mut tracker, &entity));
        assert_eq!(tracker.order_id, Some(42));
    }

    #[test]
    fn update_latency_from_envelope_records_seen_ack_and_fill() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = SessionState {
            cfg: AppConfig::default(),
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            order_latency_tracker: Some(OrderLatencyTracker {
                started_at: time::Instant::now(),
                cl_ord_id: "midas-1-entry".to_string(),
                order_id: Some(42),
                seen_recorded: false,
                exec_report_recorded: false,
                fill_recorded: false,
            }),
            order_submit_in_flight: false,
            user_store: UserSyncStore::default(),
            selected_account_id: None,
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            next_strategy_order_nonce: 1,
        };
        let mut latency = LatencySnapshot::default();

        let order = EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42 }),
        };
        let exec_report = EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42, "execType": "New" }),
        };
        let fill = EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({ "orderId": 42, "price": 5000.25, "qty": 1 }),
        };

        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &order
        ));
        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &exec_report
        ));
        assert!(update_latency_from_envelope(
            &mut session,
            &mut latency,
            &fill
        ));
        assert!(latency.last_order_seen_ms.is_some());
        assert!(latency.last_exec_report_ms.is_some());
        assert!(latency.last_fill_ms.is_some());
    }

    #[test]
    fn trade_marker_from_fill_uses_order_action_for_side_and_contract() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut user_store = UserSyncStore::default();
        user_store.orders.insert(
            42,
            BTreeMap::from([(
                77,
                json!({
                    "id": 77,
                    "accountId": 42,
                    "action": "Sell",
                    "contractId": 3570918,
                    "symbol": "ESH6"
                }),
            )]),
        );
        let session = SessionState {
            cfg: AppConfig::default(),
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            order_latency_tracker: None,
            order_submit_in_flight: false,
            user_store,
            selected_account_id: Some(42),
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            next_strategy_order_nonce: 1,
        };
        let fill = json!({
            "id": 501,
            "accountId": 42,
            "orderId": 77,
            "price": 5000.25,
            "qty": 1,
            "timestamp": "2026-03-15T13:45:00Z"
        });

        let marker = trade_marker_from_fill(&session, &fill).expect("fill marker should resolve");

        assert_eq!(marker.fill_id, Some(501));
        assert_eq!(marker.account_id, Some(42));
        assert_eq!(marker.contract_id, Some(3570918));
        assert_eq!(marker.contract_name.as_deref(), Some("ESH6"));
        assert_eq!(marker.side, TradeMarkerSide::Sell);
        assert_eq!(marker.price, 5000.25);
        assert_eq!(marker.qty, 1);
    }

    #[test]
    fn record_trade_marker_deduplicates_fill_ids() {
        let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = SessionState {
            cfg: AppConfig::default(),
            tokens: TokenBundle {
                access_token: "access".to_string(),
                md_access_token: "md".to_string(),
                expiration_time: None,
                user_id: None,
                user_name: None,
            },
            accounts: Vec::new(),
            request_tx,
            execution_config: ExecutionStrategyConfig::default(),
            execution_runtime: ExecutionRuntimeState::default(),
            order_latency_tracker: None,
            order_submit_in_flight: false,
            user_store: UserSyncStore::default(),
            selected_account_id: Some(42),
            selected_contract: None,
            bar_type: BarType::default(),
            market: MarketSnapshot::default(),
            managed_protection: BTreeMap::new(),
            next_strategy_order_nonce: 1,
        };
        let marker = TradeMarker {
            fill_id: Some(501),
            account_id: Some(42),
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            ts_ns: 1,
            price: 5000.25,
            qty: 1,
            side: TradeMarkerSide::Buy,
        };

        assert!(record_trade_marker(&mut session, marker.clone()));
        assert!(!record_trade_marker(&mut session, marker));
        assert_eq!(session.market.trade_markers.len(), 1);
    }

    #[test]
    fn build_snapshots_include_realized_pnl_and_protection_prices() {
        let mut store = UserSyncStore::default();
        store.risk.insert(
            42,
            json!({
                "accountId": 42,
                "balance": 10000.0,
                "realizedPnL": 125.5
            }),
        );
        store.positions.insert(
            42,
            BTreeMap::from([(
                1,
                json!({
                    "id": 1,
                    "accountId": 42,
                    "contractId": 3570918,
                    "netPos": 1,
                    "netPrice": 5000.0
                }),
            )]),
        );
        let market = MarketSnapshot {
            contract_id: Some(3570918),
            contract_name: Some("ESH6".to_string()),
            bars: vec![Bar {
                ts_ns: 0,
                open: 4999.0,
                high: 5002.0,
                low: 4998.0,
                close: 5001.0,
            }],
            trade_markers: Vec::new(),
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(50.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 0,
            status: String::new(),
        };
        let managed_protection = BTreeMap::from([(
            StrategyProtectionKey {
                account_id: 42,
                contract_id: 3570918,
            },
            ManagedProtectionOrders {
                signed_qty: 1,
                take_profit_price: Some(5004.0),
                stop_price: Some(4998.0),
                take_profit_cl_ord_id: None,
                stop_cl_ord_id: None,
                take_profit_order_id: None,
                stop_order_id: None,
            },
        )]);
        let accounts = vec![AccountInfo {
            id: 42,
            name: "sim".to_string(),
            raw: json!({ "id": 42, "name": "sim" }),
        }];

        let snapshots = store.build_snapshots(&accounts, Some(&market), &managed_protection);
        let snapshot = snapshots.first().expect("snapshot should exist");

        assert_eq!(snapshot.realized_pnl, Some(125.5));
        assert_eq!(snapshot.market_entry_price, Some(5000.0));
        assert_eq!(snapshot.selected_contract_take_profit_price, Some(5004.0));
        assert_eq!(snapshot.selected_contract_stop_price, Some(4998.0));
    }

    #[test]
    fn parse_expiration_time_accepts_rfc3339() {
        let parsed = parse_expiration_time("2026-03-13T15:04:05Z").expect("timestamp should parse");
        let expected = DateTime::parse_from_rfc3339("2026-03-13T15:04:05Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(parsed, expected);
    }

    #[test]
    fn futures_globex_preclose_window_holds_entries() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 16, 50, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(window.hold_entries);
        assert!(window.minutes_to_close.unwrap() <= 10.0);
    }

    #[test]
    fn futures_globex_daily_break_holds_until_reopen() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 17, 30, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(!window.session_open);
        assert!(window.hold_entries);
        assert_eq!(window.minutes_to_close, None);
    }

    #[test]
    fn futures_globex_reopens_after_break() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 18, 5, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::FuturesGlobex.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(!window.hold_entries);
        assert!(window.minutes_to_close.unwrap() > 1_300.0);
    }

    #[test]
    fn equity_rth_preclose_window_holds_entries() {
        let dt = New_York
            .with_ymd_and_hms(2026, 3, 9, 15, 50, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc);
        let window =
            InstrumentSessionProfile::EquityRth.evaluate(dt.timestamp_nanos_opt().unwrap());

        assert!(window.session_open);
        assert!(window.hold_entries);
        assert!(window.minutes_to_close.unwrap() <= 10.0);
    }
}

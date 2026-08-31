//! A deliberately small, offline Tradovate protocol façade.
//!
//! This process is not part of the trader binary and has no HTTP client or
//! broker credentials.  It only binds loopback listeners, reads one explicit
//! fixture, and exposes the REST/user-market WebSocket surfaces that the
//! native Tradovate path already consumes.  That makes it useful for replaying
//! the same fixture against binaries built from different commits.

use anyhow::{Context, Result, bail};
use arrow_array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, Stream, StreamExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Semaphore, broadcast, mpsc};
use tokio::time::{Instant, sleep, timeout};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use crate::Args;
use crate::protection::{
    BracketStatus, ExitPrecedence, FillIntent, MarketTick, Position, PositionSide,
    ProtectionBracket, ProtectionEvent, ProtectionLeg, ProtectionParams, TrailingConfig,
};
use crate::tick::{DuplicateTickIds, RawTick, TickLoadOptions, TickOrdering};

const PROXY_SCHEMA_VERSION: u32 = 1;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 2 * 1024 * 1024;
const MAX_TRACE_VALUE_BYTES: usize = 4096;
const DEFAULT_PARQUET_BATCH_ROWS: usize = 4096;
const MAX_SEEN_RESPONSES: usize = 4096;
const MAX_FIXTURE_RECORD_BYTES: usize = 2 * 1024 * 1024;
const MAX_SYNTHETIC_ID_SEED: i64 = i64::MAX / 100_000;
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(10);
const WS_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_TIMESTAMP_DRIFT_NS: i128 = 1_000_000;

#[derive(Debug, Clone)]
struct Settings {
    rest_bind: SocketAddr,
    user_ws_bind: SocketAddr,
    market_ws_bind: SocketAddr,
    history_bars: usize,
    max_bars: usize,
    start_file: Option<PathBuf>,
    speed: f64,
    max_sleep_ms: u64,
    loop_boundary_delay_ms: u64,
    rest_delay_ms: u64,
    ack_delay_ms: u64,
    fill_delay_ms: u64,
    loop_replay: bool,
    account_id: i64,
    account_name: String,
    contract: String,
    contract_id: i64,
    contract_id_explicit: bool,
    tick_size: f64,
    value_per_point: f64,
    initial_balance: f64,
    max_clients: usize,
    max_state_entities: usize,
    trace_max_bytes: usize,
    fixture_path: PathBuf,
    tick_fixture_path: Option<PathBuf>,
    fixture_manifest: Option<PathBuf>,
    allow_unverified_fixture: bool,
    require_quote_ticks: bool,
    max_ticks: usize,
    raw_tick_bar_timestamps: RawTickBarTimestampMode,
    protection_precedence: ExitPrecedence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawTickBarTimestampMode {
    Start,
    Close,
}

impl Settings {
    fn from_args(args: &Args) -> Result<Self> {
        for (name, address) in [
            ("rest_bind", args.rest_bind),
            ("user_ws_bind", args.user_ws_bind),
            ("market_ws_bind", args.market_ws_bind),
        ] {
            if !address.ip().is_loopback() {
                bail!("{name} must bind to a loopback address; refusing {address}");
            }
        }
        if args.history_bars == 0 {
            bail!("history_bars must be greater than zero");
        }
        if !args.speed.is_finite() || args.speed < 0.0 {
            bail!("speed must be finite and non-negative");
        }
        if args.loop_replay && args.speed == 0.0 {
            bail!("loop_replay requires speed > 0 to avoid an unpaced hot loop");
        }
        if args.loop_replay && args.ticks.is_some() {
            bail!(
                "loop_replay cannot be combined with --ticks; raw-tick replay is a single deterministic broker lifecycle"
            );
        }
        if args.require_quote_ticks && args.ticks.is_none() {
            bail!("require_quote_ticks requires --ticks");
        }
        if args.max_sleep_ms == 0 {
            bail!("max_sleep_ms must be greater than zero");
        }
        if args.loop_boundary_delay_ms == 0 {
            bail!("loop_boundary_delay_ms must be greater than zero");
        }
        let contract_id = args.contract_id.unwrap_or(910_002);
        if args.account_id <= 0 || contract_id <= 0 {
            bail!("account_id and contract_id must be positive");
        }
        if contract_id > MAX_SYNTHETIC_ID_SEED {
            bail!(
                "contract_id is too large for bounded synthetic entity IDs; use a value <= {MAX_SYNTHETIC_ID_SEED}"
            );
        }
        if args.account_name.trim().is_empty() || args.contract.trim().is_empty() {
            bail!("account_name and contract must not be empty");
        }
        if !args.tick_size.is_finite() || args.tick_size <= 0.0 {
            bail!("tick_size must be finite and greater than zero");
        }
        if !args.value_per_point.is_finite() || args.value_per_point <= 0.0 {
            bail!("value_per_point must be finite and greater than zero");
        }
        if !args.initial_balance.is_finite() || args.initial_balance <= 0.0 {
            bail!("initial_balance must be finite and greater than zero");
        }
        if args.max_clients == 0 {
            bail!("max_clients must be greater than zero");
        }
        if args.max_state_entities == 0 {
            bail!("max_state_entities must be greater than zero");
        }
        if args.trace_max_bytes == 0 {
            bail!("trace_max_bytes must be greater than zero");
        }
        let raw_tick_bar_timestamps = match args
            .raw_tick_bar_timestamps
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "start" => RawTickBarTimestampMode::Start,
            "close" => RawTickBarTimestampMode::Close,
            other => bail!("invalid raw_tick_bar_timestamps `{other}`; expected start or close"),
        };
        let protection_precedence = match args
            .protection_precedence
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "stop" | "stop_first" | "conservative" => ExitPrecedence::StopFirst,
            "target" | "take_profit" | "take_profit_first" | "optimistic" => {
                ExitPrecedence::TakeProfitFirst
            }
            other => bail!("invalid protection_precedence `{other}`; expected stop or target"),
        };
        Ok(Self {
            rest_bind: args.rest_bind,
            user_ws_bind: args.user_ws_bind,
            market_ws_bind: args.market_ws_bind,
            history_bars: args.history_bars,
            max_bars: args.max_bars,
            start_file: args.start_file.clone(),
            speed: args.speed,
            max_sleep_ms: args.max_sleep_ms,
            loop_boundary_delay_ms: args.loop_boundary_delay_ms,
            rest_delay_ms: args.rest_delay_ms,
            ack_delay_ms: args.ack_delay_ms,
            fill_delay_ms: args.fill_delay_ms,
            loop_replay: args.loop_replay,
            account_id: args.account_id,
            account_name: args.account_name.clone(),
            contract: args.contract.clone(),
            contract_id,
            contract_id_explicit: args.contract_id.is_some(),
            tick_size: args.tick_size,
            value_per_point: args.value_per_point,
            initial_balance: args.initial_balance,
            max_clients: args.max_clients,
            max_state_entities: args.max_state_entities,
            trace_max_bytes: args.trace_max_bytes,
            fixture_path: args.bars.clone(),
            tick_fixture_path: args.ticks.clone(),
            fixture_manifest: args.fixture_manifest.clone(),
            allow_unverified_fixture: args.allow_unverified_fixture,
            require_quote_ticks: args.require_quote_ticks,
            max_ticks: args.max_ticks,
            raw_tick_bar_timestamps,
            protection_precedence,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct FixtureMetadata {
    path: String,
    sha256: String,
    rows: usize,
    first_ts_ns: Option<i64>,
    last_ts_ns: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
struct TickFixtureMetadata {
    path: String,
    sha256: String,
    rows: usize,
    first_ts_ns: Option<i64>,
    last_ts_ns: Option<i64>,
    quote_rows: usize,
}

#[derive(Debug, Clone, Serialize, Default)]
struct FixtureIdentityMetadata {
    verified: bool,
    manifests: Vec<String>,
    contract_symbols: Vec<String>,
    contract_ids: Vec<i64>,
    tick_sizes: Vec<f64>,
    value_per_points: Vec<f64>,
    bar_shapes: Vec<ChartShape>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ChartShape {
    underlying_type: &'static str,
    element_size: u32,
    element_size_unit: &'static str,
}

impl ChartShape {
    fn from_bar_kind(kind: &str, element_size: u32) -> Option<Self> {
        let (underlying_type, element_size_unit) = match kind.to_ascii_lowercase().as_str() {
            "minute" => ("MinuteBar", "UnderlyingUnits"),
            "second" => ("Tick", "Seconds"),
            "tick" => ("Tick", "UnderlyingUnits"),
            "volume" => ("Tick", "Volume"),
            "range" => ("Tick", "Range"),
            _ => return None,
        };
        (element_size > 0).then_some(Self {
            underlying_type,
            element_size,
            element_size_unit,
        })
    }

    fn matches_description(&self, description: &Value) -> bool {
        description.get("underlyingType").and_then(Value::as_str)
            == Some(self.underlying_type)
            && description.get("elementSize").and_then(Value::as_u64)
                == Some(self.element_size as u64)
            && description.get("elementSizeUnit").and_then(Value::as_str)
                == Some(self.element_size_unit)
    }
}

fn chart_request_matches_contract(body: &Value, settings: &Settings) -> bool {
    let Some(symbol) = body.get("symbol") else {
        return false;
    };
    symbol.as_i64() == Some(settings.contract_id)
        || symbol
            .as_str()
            .is_some_and(|value| value.eq_ignore_ascii_case(&settings.contract))
        || symbol
            .as_str()
            .and_then(|value| value.parse::<i64>().ok())
            .is_some_and(|value| value == settings.contract_id)
}

fn value_as_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| {
            value.as_f64().and_then(|value| {
                (value.is_finite() && value.fract() == 0.0 && value >= 0.0)
                    .then_some(value as u64)
                    .and_then(|value| usize::try_from(value).ok())
            })
        })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct Bar {
    timestamp: DateTime<Utc>,
    ts_ns: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(default)]
    volume: Option<f64>,
}

#[derive(Debug, Clone)]
struct Fixture {
    bars: Arc<Vec<Bar>>,
    metadata: FixtureMetadata,
    identity: FixtureIdentityMetadata,
    bar_shape: Option<ChartShape>,
    ticks: Option<Arc<Vec<RawTick>>>,
    tick_metadata: Option<TickFixtureMetadata>,
    tick_seed_index: usize,
    tick_end_index: usize,
}

#[derive(Debug, Serialize)]
struct TraceRecord {
    seq: u64,
    wall_ns: u128,
    protocol: &'static str,
    direction: &'static str,
    connection_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<u64>,
    summary: String,
}

#[derive(Debug, Clone, Default)]
struct TraceSink {
    writer: Option<Arc<Mutex<std::io::BufWriter<File>>>>,
    next_sequence: Arc<AtomicU64>,
    written_bytes: Arc<AtomicUsize>,
    truncated: Arc<AtomicBool>,
    max_bytes: usize,
}

impl TraceSink {
    fn new(path: Option<&Path>, settings: &Settings, fixture: &Fixture) -> Result<Self> {
        let Some(path) = path else {
            return Ok(Self {
                writer: None,
                next_sequence: Arc::new(AtomicU64::new(0)),
                written_bytes: Arc::new(AtomicUsize::new(0)),
                truncated: Arc::new(AtomicBool::new(false)),
                max_bytes: settings.trace_max_bytes,
            });
        };
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create trace directory {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .with_context(|| format!("open trace {}", path.display()))?;
        let sink = Self {
            writer: Some(Arc::new(Mutex::new(std::io::BufWriter::new(file)))),
            next_sequence: Arc::new(AtomicU64::new(0)),
            written_bytes: Arc::new(AtomicUsize::new(0)),
            truncated: Arc::new(AtomicBool::new(false)),
            max_bytes: settings.trace_max_bytes,
        };
        sink.record(TraceRecord {
            seq: 0,
            wall_ns: wall_ns(),
            protocol: "proxy",
            direction: "meta",
            connection_id: 0,
            endpoint: None,
            request_id: None,
            summary: serde_json::to_string(&json!({
                "schema_version": PROXY_SCHEMA_VERSION,
                "fixture": fixture.metadata,
                "fixture_identity": fixture.identity,
                "bar_shape": fixture.bar_shape,
                "tick_fixture": fixture.tick_metadata,
                "tick_seed_index": fixture.tick_seed_index,
                "tick_end_index": fixture.tick_end_index,
                "account_id": settings.account_id,
                "contract": settings.contract,
                "contract_id": settings.contract_id,
                "speed": settings.speed,
                "history_bars": settings.history_bars,
                "max_bars": settings.max_bars,
                "start_file": settings.start_file,
                "max_sleep_ms": settings.max_sleep_ms,
                "loop_boundary_delay_ms": settings.loop_boundary_delay_ms,
                "loop_replay": settings.loop_replay,
                "max_clients": settings.max_clients,
                "max_state_entities": settings.max_state_entities,
                "trace_max_bytes": settings.trace_max_bytes,
                "initial_balance": settings.initial_balance,
                "tick_size": settings.tick_size,
                "value_per_point": settings.value_per_point,
                "require_quote_ticks": settings.require_quote_ticks,
                "raw_tick_bar_timestamps": match settings.raw_tick_bar_timestamps {
                    RawTickBarTimestampMode::Start => "start",
                    RawTickBarTimestampMode::Close => "close",
                },
                "protection_precedence": match settings.protection_precedence {
                    ExitPrecedence::StopFirst => "stop",
                    ExitPrecedence::TakeProfitFirst => "target",
                },
                "delays_ms": {
                    "rest": settings.rest_delay_ms,
                    "ack": settings.ack_delay_ms,
                    "fill": settings.fill_delay_ms,
                },
                "source_commit": std::env::var("TRADER_PROXY_SOURCE_COMMIT").ok(),
                "source_workspace_dirty": std::env::var("TRADER_PROXY_SOURCE_WORKSPACE_DIRTY").ok(),
                "client_commit": std::env::var("TRADER_PROXY_CLIENT_COMMIT").ok(),
                "run_id": std::env::var("TRADER_PROXY_RUN_ID")
                    .ok()
                    .unwrap_or_else(|| format!("proxy-{}-{}", std::process::id(), wall_ns())),
                "proxy_toolchain": std::env::var("TRADER_PROXY_TOOLCHAIN").ok(),
                "proxy_binary_sha256": std::env::var("TRADER_PROXY_BINARY_SHA256").ok(),
                "client_binary_sha256": std::env::var("TRADER_PROXY_CLIENT_BINARY_SHA256").ok(),
                "client_toolchain": std::env::var("TRADER_PROXY_CLIENT_TOOLCHAIN").ok(),
            }))?,
        });
        Ok(sink)
    }

    fn record(&self, mut record: TraceRecord) {
        let Some(writer) = self.writer.as_ref() else {
            return;
        };
        let Ok(mut writer) = writer.lock() else {
            return;
        };
        record.seq = self.next_sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let Ok(mut encoded) = serde_json::to_vec(&record) else {
            return;
        };
        encoded.push(b'\n');
        let current = self.written_bytes.load(Ordering::Relaxed);
        if current.saturating_add(encoded.len()) > self.max_bytes {
            self.truncated.store(true, Ordering::Relaxed);
            return;
        }
        if writer.write_all(&encoded).is_ok() {
            // A trace is a diagnostic artifact used during crash/regression
            // investigation. Flush each record so a forced process stop does
            // not erase the very request that exposed the fault. The trace is
            // optional and intentionally not part of the low-latency path.
            let _ = writer.flush();
            self.written_bytes
                .fetch_add(encoded.len(), Ordering::Relaxed);
        }
    }

    fn finish(&self) {
        if let Some(writer) = self.writer.as_ref()
            && let Ok(mut writer) = writer.lock()
        {
            let _ = writer.flush();
        }
        if self.truncated.load(Ordering::Relaxed) {
            eprintln!(
                "warning: proxy trace reached --trace-max-bytes ({}); later records were omitted",
                self.max_bytes
            );
        }
    }

    fn text_summary(raw: &str) -> String {
        if raw.len() <= MAX_TRACE_VALUE_BYTES {
            return raw.to_string();
        }
        // MAX_TRACE_VALUE_BYTES is a byte budget, but Rust strings must be
        // sliced on UTF-8 boundaries. The first character boundary at or
        // after the budget gives a bounded, valid prefix.
        let end = raw
            .char_indices()
            .map(|(index, _)| index)
            .find(|index| *index >= MAX_TRACE_VALUE_BYTES)
            .unwrap_or(raw.len());
        format!("{}… [truncated {} bytes]", &raw[..end], raw.len())
    }
}

#[derive(Debug, Clone)]
struct Shared {
    settings: Settings,
    fixture: Fixture,
    broker: Arc<Mutex<BrokerState>>,
    events: broadcast::Sender<Value>,
    fill_tx: mpsc::Sender<i64>,
    trace: TraceSink,
}

#[derive(Debug, Clone)]
struct BrokerState {
    last_price: Option<f64>,
    last_bid: Option<f64>,
    last_ask: Option<f64>,
    last_ts_ns: Option<i64>,
    market_sequence: u64,
    position_qty: i64,
    average_price: Option<f64>,
    position_generation: u64,
    realized_pnl: f64,
    next_order_id: i64,
    next_fill_id: i64,
    next_execution_id: i64,
    next_strategy_id: i64,
    next_command_id: i64,
    next_link_id: i64,
    orders: BTreeMap<i64, Value>,
    fills: BTreeMap<i64, Value>,
    execution_reports: BTreeMap<i64, Value>,
    fill_fees: BTreeMap<i64, Value>,
    strategies: BTreeMap<i64, Value>,
    strategy_links: BTreeMap<i64, Value>,
    cl_ord_ids: HashMap<String, i64>,
    strategy_uuids: HashMap<String, i64>,
    pending_fill_quotes: BTreeMap<i64, (f64, i64, u64)>,
    protections: BTreeMap<i64, ProtectionRuntime>,
}

impl BrokerState {
    fn new(settings: &Settings) -> Self {
        Self {
            last_price: None,
            last_bid: None,
            last_ask: None,
            last_ts_ns: None,
            market_sequence: 0,
            position_qty: 0,
            average_price: None,
            position_generation: 0,
            realized_pnl: 0.0,
            next_order_id: settings.contract_id.saturating_mul(100).saturating_add(1),
            next_fill_id: settings.contract_id.saturating_mul(1000).saturating_add(1),
            next_execution_id: settings
                .contract_id
                .saturating_mul(10_000)
                .saturating_add(1),
            next_strategy_id: settings.contract_id.saturating_mul(10).saturating_add(1),
            next_command_id: settings
                .contract_id
                .saturating_mul(20_000)
                .saturating_add(1),
            next_link_id: settings
                .contract_id
                .saturating_mul(30_000)
                .saturating_add(1),
            orders: BTreeMap::new(),
            fills: BTreeMap::new(),
            execution_reports: BTreeMap::new(),
            fill_fees: BTreeMap::new(),
            strategies: BTreeMap::new(),
            strategy_links: BTreeMap::new(),
            cl_ord_ids: HashMap::new(),
            strategy_uuids: HashMap::new(),
            pending_fill_quotes: BTreeMap::new(),
            protections: BTreeMap::new(),
        }
    }

    fn set_market(&mut self, bar: &Bar) {
        self.last_price = Some(bar.close);
        self.last_ts_ns = Some(bar.ts_ns);
        self.last_bid = Some(bar.close);
        self.last_ask = Some(bar.close);
        self.market_sequence = self.market_sequence.saturating_add(1);
    }

    fn set_market_tick(&mut self, tick: &RawTick, sequence: u64) {
        self.last_price = Some(tick.price);
        self.last_bid = tick.bid_price.or(Some(tick.price));
        self.last_ask = tick.ask_price.or(Some(tick.price));
        self.last_ts_ns = Some(tick.ts_ns);
        self.market_sequence = sequence;
    }

    fn position_entity(&self, settings: &Settings) -> Value {
        let unrealized = self
            .last_price
            .zip(self.average_price)
            .map(|(last, entry)| {
                (last - entry) * self.position_qty as f64 * settings.value_per_point
            })
            .unwrap_or(0.0);
        json!({
            "id": settings.contract_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "symbol": settings.contract,
            "netPos": self.position_qty,
            "netPrice": self.average_price.unwrap_or_default(),
            "avgPrice": self.average_price.unwrap_or_default(),
            "unrealizedPnL": unrealized,
        })
    }

    fn cash_entity(&self, settings: &Settings) -> Value {
        let balance = settings.initial_balance + self.realized_pnl;
        json!({
            "id": settings.account_id,
            "accountId": settings.account_id,
            "cashBalance": balance,
            "balance": balance,
            "realizedPnL": self.realized_pnl,
            "unrealizedPnL": self.position_entity(settings)
                .get("unrealizedPnL")
                .and_then(Value::as_f64)
                .unwrap_or_default(),
        })
    }

    fn account_entity(&self, settings: &Settings) -> Value {
        let balance = settings.initial_balance + self.realized_pnl;
        json!({
            "id": settings.account_id,
            "name": settings.account_name,
            "userId": settings.account_id,
            "active": true,
            "balance": balance,
            "cashBalance": balance,
            "realizedPnL": self.realized_pnl,
        })
    }

    fn snapshot(&self, settings: &Settings) -> Value {
        let positions = vec![self.position_entity(settings)];
        let orders = self.orders.values().cloned().collect::<Vec<_>>();
        let fills = self.fills.values().cloned().collect::<Vec<_>>();
        let execution_reports = self.execution_reports.values().cloned().collect::<Vec<_>>();
        let fees = self.fill_fees.values().cloned().collect::<Vec<_>>();
        let strategies = self.strategies.values().cloned().collect::<Vec<_>>();
        let links = self.strategy_links.values().cloned().collect::<Vec<_>>();
        json!({
            "accounts": [self.account_entity(settings)],
            "accountRiskStatuses": [{
                "id": settings.account_id,
                "accountId": settings.account_id,
                "marginUsed": 0.0
            }],
            "cashBalances": [self.cash_entity(settings)],
            "positions": positions,
            "commands": [],
            "commandReports": [],
            "orders": orders,
            "fills": fills,
            "executionReports": execution_reports,
            "fillFees": fees,
            "orderStrategies": strategies,
            "orderStrategyLinks": links,
        })
    }

    fn next_timestamp_ms(&self) -> i64 {
        self.last_ts_ns
            .map(|ts| ts / 1_000_000)
            .unwrap_or_else(|| wall_ns().min(i64::MAX as u128) as i64 / 1_000_000)
    }

    fn accept_order(
        &mut self,
        request: OrderRequest<'_>,
    ) -> Result<(Value, i64, Vec<Value>, bool)> {
        let OrderRequest {
            settings,
            action,
            quantity,
            account_id,
            contract_id,
            symbol,
            cl_ord_id,
            strategy_id,
            bracket,
        } = request;
        if account_id != settings.account_id {
            bail!("proxy rejected unknown accountId {account_id}");
        }
        if contract_id != settings.contract_id {
            bail!("proxy rejected unknown contractId {contract_id}");
        }
        if !symbol.is_empty() && !symbol.eq_ignore_ascii_case(&settings.contract) {
            bail!("proxy rejected unknown symbol {symbol}");
        }
        let action = normalize_action(action)?;
        if quantity <= 0 {
            bail!("proxy rejected non-positive order quantity");
        }
        if let Some(bracket) = bracket.as_ref() {
            // Validate before acknowledging the parent. A malformed bracket
            // must never produce an unprotected Working strategy.
            parse_protection_params(settings, bracket)
                .context("proxy rejected malformed broker-owned bracket")?;
        }
        let signed_quantity = if action == "Buy" { quantity } else { -quantity };
        if self.position_qty.checked_add(signed_quantity).is_none() {
            bail!("proxy rejected order quantity because synthetic position would overflow");
        }
        if let Some(cl_ord_id) = cl_ord_id.as_deref()
            && let Some(order_id) = self.cl_ord_ids.get(cl_ord_id).copied()
        {
            let response = json!({"id": order_id, "orderId": order_id, "duplicate": true});
            return Ok((response, order_id, Vec::new(), false));
        }

        self.prune_terminal_state(settings.max_state_entities);
        if self.entity_count().saturating_add(1) > settings.max_state_entities {
            bail!("proxy synthetic state limit reached");
        }

        let Some(price) = executable_market_price(self, action) else {
            bail!("proxy cannot accept a market order before a replay quote");
        };
        let timestamp = self.last_ts_ns.unwrap_or_default();

        let order_id = self.next_order_id;
        self.next_order_id = self.next_order_id.saturating_add(1);
        let order = json!({
            "id": order_id,
            "orderId": order_id,
            "accountId": account_id,
            "contractId": contract_id,
            "symbol": settings.contract,
            "action": action,
            "orderQty": quantity,
            "filledQty": 0,
            "ordStatus": "Working",
            "orderType": "Market",
            "timeInForce": "Day",
            "isAutomated": true,
            "clOrdId": cl_ord_id,
            "orderStrategyId": strategy_id,
            "bracket": bracket,
            "acceptedMarketSequence": self.market_sequence,
        });
        self.orders.insert(order_id, order.clone());
        if let Some(cl_ord_id) = cl_ord_id {
            self.cl_ord_ids.insert(cl_ord_id, order_id);
        }
        self.pending_fill_quotes.insert(
            order_id,
            (price, timestamp, self.market_sequence.saturating_add(1)),
        );
        let response = json!({"id": order_id, "orderId": order_id, "orderStatus": "Working"});
        Ok((response, order_id, vec![props_event("order", order)], true))
    }

    fn fill_order(&mut self, settings: &Settings, order_id: i64) -> Vec<Value> {
        let Some((price, timestamp_ns, _eligible_sequence)) =
            self.pending_fill_quotes.remove(&order_id)
        else {
            return Vec::new();
        };
        let mut events =
            self.fill_order_at_price(settings, order_id, price, timestamp_ns, None, "proxy", None);
        // Bar-only mode has no raw-tick callback in which to notice that a
        // direct flatten/reversal changed the position under an older
        // broker-owned bracket. Apply the same stale-protection teardown used
        // by the tick clock before returning the fill events.
        self.cancel_stale_protections(settings, &mut events);
        self.prune_terminal_state(settings.max_state_entities);
        events
    }

    fn fill_order_at_tick(
        &mut self,
        settings: &Settings,
        order_id: i64,
        tick: &RawTick,
        sequence: u64,
    ) -> Vec<Value> {
        let Some((_, _, eligible_sequence)) = self.pending_fill_quotes.get(&order_id).copied()
        else {
            return Vec::new();
        };
        if sequence < eligible_sequence {
            return Vec::new();
        }
        let Some((price, fill_source)) =
            raw_tick_executable_price(tick, self.order_action(order_id))
        else {
            return Vec::new();
        };
        self.pending_fill_quotes.remove(&order_id);
        self.fill_order_at_price(
            settings,
            order_id,
            price,
            tick.ts_ns,
            Some(sequence),
            fill_source,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fill_order_at_price(
        &mut self,
        settings: &Settings,
        order_id: i64,
        price: f64,
        timestamp_ns: i64,
        market_sequence: Option<u64>,
        fill_source: &'static str,
        exit_reason: Option<&'static str>,
    ) -> Vec<Value> {
        let Some(order) = self.orders.get(&order_id).cloned() else {
            return Vec::new();
        };
        if order
            .get("ordStatus")
            .and_then(Value::as_str)
            .is_some_and(|status| {
                matches!(
                    status,
                    "Filled" | "Cancelled" | "Canceled" | "Rejected" | "Expired"
                )
            })
        {
            return Vec::new();
        }
        let quantity = order
            .get("orderQty")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        if quantity <= 0 {
            return Vec::new();
        }
        let action = order.get("action").and_then(Value::as_str).unwrap_or("Buy");
        let signed_qty = if action.eq_ignore_ascii_case("Buy") {
            quantity
        } else {
            -quantity
        };
        if !price.is_finite() || price <= 0.0 {
            return Vec::new();
        }
        if self.position_qty.checked_add(signed_qty).is_none() {
            let mut rejected_order = order;
            if let Some(object) = rejected_order.as_object_mut() {
                object.insert("ordStatus".to_string(), json!("Rejected"));
                object.insert(
                    "failureReason".to_string(),
                    json!("synthetic position would overflow"),
                );
            }
            self.orders.insert(order_id, rejected_order.clone());
            self.prune_terminal_state(settings.max_state_entities);
            return vec![props_event("order", rejected_order)];
        }
        self.apply_fill(settings, signed_qty, price);
        self.position_generation = self.position_generation.saturating_add(1);

        let strategy_id = order.get("orderStrategyId").and_then(Value::as_i64);
        let has_bracket = order
            .get("bracket")
            .is_some_and(|bracket| !bracket.is_null());
        let fill_id = self.next_fill_id;
        self.next_fill_id = self.next_fill_id.saturating_add(1);
        let timestamp = timestamp_ns
            .checked_div(1_000_000)
            .unwrap_or_else(|| self.next_timestamp_ms());
        let execution_id = self.next_execution_id;
        self.next_execution_id = self.next_execution_id.saturating_add(1);
        let order_strategy_id = order.get("orderStrategyId").cloned().unwrap_or(Value::Null);
        let cl_ord_id = order.get("clOrdId").cloned().unwrap_or(Value::Null);
        let execution = json!({
            "id": execution_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "orderId": order_id,
            "clOrdId": cl_ord_id,
            "orderStrategyId": order_strategy_id,
            "status": "Filled",
            "price": price,
            "timestamp": timestamp_ns,
            "replayFillSource": fill_source,
            "replayMarketSequence": market_sequence,
            "replayPositionGeneration": self.position_generation,
            "replayExitReason": exit_reason,
        });
        let fill = json!({
            "id": fill_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "symbol": settings.contract,
            "orderId": order_id,
            "orderStrategyId": order.get("orderStrategyId").cloned().unwrap_or(Value::Null),
            "clOrdId": order.get("clOrdId").cloned().unwrap_or(Value::Null),
            "action": action,
            "buySell": action,
            "qty": quantity,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "source": "replay",
            "replayFillSource": fill_source,
            "replayExecutionPrecision": replay_execution_precision(fill_source),
            "replayFillTimestampNs": timestamp_ns,
            "replayProtectionOrderId": exit_reason.map(|_| order_id),
            "replayAmbiguousTick": matches!(
                fill_source,
                "tick_partial_quote" | "tick_trade_fallback"
            ),
            "replayMarketSequence": market_sequence,
            "replayPositionGeneration": self.position_generation,
            "replayExitReason": exit_reason,
        });
        let original_order = order.clone();
        let mut filled_order = order;
        if let Some(object) = filled_order.as_object_mut() {
            object.insert("ordStatus".to_string(), Value::String("Filled".to_string()));
            object.insert("filledQty".to_string(), json!(quantity));
            object.insert("avgFillPrice".to_string(), json!(price));
        }
        self.orders.insert(order_id, filled_order.clone());
        self.fills.insert(fill_id, fill.clone());
        self.execution_reports
            .insert(execution_id, execution.clone());
        let fee_id = self.next_command_id;
        self.next_command_id = self.next_command_id.saturating_add(1);
        let fee = json!({
            "id": fee_id,
            "accountId": settings.account_id,
            "fillId": fill_id,
            "amount": 0.0,
            "commission": 0.0,
        });
        self.fill_fees.insert(fee_id, fee.clone());
        let mut events = vec![
            props_event("order", filled_order),
            props_event("executionReport", execution),
            props_event("fill", fill),
            props_event("fillFee", fee),
            props_event("position", self.position_entity(settings)),
            props_event("cashBalance", self.cash_entity(settings)),
            props_event("account", self.account_entity(settings)),
        ];
        if let Some(strategy_id) = strategy_id
            && has_bracket
        {
            match self.arm_protection(settings, strategy_id, order_id, price, &original_order) {
                Ok(protection_events) => events.extend(protection_events),
                Err(error) => {
                    let error = error.to_string();
                    events.push(self.mark_protection_failure(strategy_id, error.clone()));
                    events.extend(self.fail_closed_after_protection_failure(
                        settings,
                        strategy_id,
                        price,
                        timestamp_ns,
                        market_sequence,
                        fill_source,
                        &error,
                    ));
                }
            }
        }
        events
    }

    fn order_action(&self, order_id: i64) -> &str {
        self.orders
            .get(&order_id)
            .and_then(|order| order.get("action"))
            .and_then(Value::as_str)
            .unwrap_or("Buy")
    }

    fn arm_protection(
        &mut self,
        settings: &Settings,
        strategy_id: i64,
        entry_order_id: i64,
        entry_price: f64,
        entry_order: &Value,
    ) -> Result<Vec<Value>> {
        let Some(position_qty) = (!self.position_qty.eq(&0)).then_some(self.position_qty) else {
            bail!("position was flat while arming broker-owned protection");
        };
        let side = if position_qty > 0 {
            PositionSide::Long
        } else {
            PositionSide::Short
        };
        let quantity = position_qty.unsigned_abs();
        let Some(raw_bracket) = entry_order.get("bracket") else {
            bail!("parent order has no broker-owned bracket");
        };
        let params = parse_protection_params(settings, raw_bracket)
            .context("parse broker-owned protection after parent fill")?;
        // Broker-owned protection covers the resulting net position. For a
        // same-side add, that basis is the weighted average, not the price of
        // only the newest fill. Reversals reset average_price in apply_fill.
        let position_entry_price = self.average_price.unwrap_or(entry_price);
        let entry_price_ticks = price_to_ticks(position_entry_price, settings.tick_size)
            .context("convert broker-owned protection entry to ticks")?;
        let position = Position::new(side, quantity, entry_price_ticks)
            .map_err(|error| anyhow::anyhow!("arm broker-owned protection: {error:?}"))?;
        let protection = ProtectionBracket::new(position, params)
            .map_err(|error| anyhow::anyhow!("arm broker-owned protection: {error:?}"))?;

        // A strategy UUID can be reused for a same-side add or a re-arm. The
        // old children are no longer valid for the new weighted-average
        // position and must be canceled before replacing the runtime entry.
        let mut events = Vec::new();
        if let Some(previous) = self.protections.remove(&strategy_id) {
            for order_id in [previous.take_profit_order_id, previous.stop_order_id]
                .into_iter()
                .flatten()
            {
                if let Some(order) = self.cancel_order(order_id) {
                    events.push(props_event("order", order));
                }
            }
        }

        let exit_action = match side.exit_order_side() {
            crate::protection::OrderSide::Buy => "Buy",
            crate::protection::OrderSide::Sell => "Sell",
        };
        let take_profit_order_id = protection.take_profit_ticks().map(|price_ticks| {
            let price = ticks_to_price(price_ticks, settings.tick_size);
            let (order_id, order, link) = self.insert_protection_child(
                settings,
                strategy_id,
                entry_order_id,
                entry_order,
                ProtectionLeg::TakeProfit,
                exit_action,
                quantity,
                "Limit",
                Some(price),
                None,
            );
            events.push(props_event("order", order));
            events.push(props_event("orderStrategyLink", link));
            order_id
        });
        let stop_order_id = protection.active_stop_ticks().map(|price_ticks| {
            let price = ticks_to_price(price_ticks, settings.tick_size);
            let (order_id, order, link) = self.insert_protection_child(
                settings,
                strategy_id,
                entry_order_id,
                entry_order,
                ProtectionLeg::StopLoss,
                exit_action,
                quantity,
                "Stop",
                None,
                Some(price),
            );
            events.push(props_event("order", order));
            events.push(props_event("orderStrategyLink", link));
            order_id
        });
        self.protections.insert(
            strategy_id,
            ProtectionRuntime {
                strategy_id,
                bracket: protection,
                take_profit_order_id,
                stop_order_id,
                position_generation: self.position_generation,
            },
        );
        Ok(events)
    }

    fn mark_protection_failure(&mut self, strategy_id: i64, error: String) -> Value {
        let strategy = self.strategies.entry(strategy_id).or_insert_with(|| {
            json!({
                "id": strategy_id,
                "orderStrategyId": strategy_id,
                "status": "Active",
                "ordStatus": "Working",
            })
        });
        if let Some(object) = strategy.as_object_mut() {
            object.insert("status".to_string(), json!("Interrupted"));
            object.insert("ordStatus".to_string(), json!("Interrupted"));
            object.insert("exitReason".to_string(), json!("protection_arm_failed"));
            object.insert("failureReason".to_string(), json!(error));
            object.insert("replayProtectionStatus".to_string(), json!("unprotected"));
            object.insert("replaySafetyAction".to_string(), json!("flatten"));
        }
        props_event("orderStrategy", strategy.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn fail_closed_after_protection_failure(
        &mut self,
        settings: &Settings,
        strategy_id: i64,
        price: f64,
        timestamp_ns: i64,
        market_sequence: Option<u64>,
        fill_source: &'static str,
        failure_reason: &str,
    ) -> Vec<Value> {
        let signed_position = self.position_qty;
        let Ok(quantity) = i64::try_from(signed_position.unsigned_abs()) else {
            return vec![self.mark_protection_failure(
                strategy_id,
                format!("{failure_reason}; safety flatten quantity overflowed"),
            )];
        };
        if quantity == 0 {
            return Vec::new();
        }
        let action = if signed_position > 0 { "Sell" } else { "Buy" };
        let order_id = self.next_order_id;
        self.next_order_id = self.next_order_id.saturating_add(1);
        let order = json!({
            "id": order_id,
            "orderId": order_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "symbol": settings.contract,
            "action": action,
            "orderQty": quantity,
            "filledQty": 0,
            "ordStatus": "Working",
            "orderType": "Market",
            "timeInForce": "Day",
            "isAutomated": true,
            "clOrdId": format!("proxy-safety-flatten-{strategy_id}"),
            "orderStrategyId": strategy_id,
            "replaySafetyFlatten": true,
            "replayProtectionFailure": failure_reason,
        });
        self.orders.insert(order_id, order);
        let events = self.fill_order_at_price(
            settings,
            order_id,
            price,
            timestamp_ns,
            market_sequence,
            fill_source,
            Some("protection_arm_failed_flatten"),
        );
        self.prune_terminal_state(settings.max_state_entities);
        events
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_protection_child(
        &mut self,
        settings: &Settings,
        strategy_id: i64,
        entry_order_id: i64,
        entry_order: &Value,
        leg: ProtectionLeg,
        action: &str,
        quantity: u64,
        order_type: &str,
        price: Option<f64>,
        stop_price: Option<f64>,
    ) -> (i64, Value, Value) {
        let order_id = self.next_order_id;
        self.next_order_id = self.next_order_id.saturating_add(1);
        let suffix = match leg {
            ProtectionLeg::TakeProfit => "tp",
            ProtectionLeg::StopLoss | ProtectionLeg::TrailingStop => "sl",
        };
        let cl_ord_id = entry_order
            .get("clOrdId")
            .and_then(Value::as_str)
            .map(|id| format!("{id}-{suffix}"))
            .unwrap_or_else(|| format!("proxy-strategy-{strategy_id}-{suffix}"));
        let mut order = json!({
            "id": order_id,
            "orderId": order_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "symbol": settings.contract,
            "action": action,
            "orderQty": quantity,
            "filledQty": 0,
            "ordStatus": "Working",
            "orderType": order_type,
            "timeInForce": "GTC",
            "isAutomated": true,
            "clOrdId": cl_ord_id,
            "orderStrategyId": strategy_id,
            "parentId": entry_order_id,
            "ocoId": strategy_id,
            "protectionLeg": suffix,
            "replayBrokerOwned": true,
            "replayPositionGeneration": self.position_generation,
        });
        if let Some(price) = price {
            order["price"] = json!(price);
        }
        if let Some(stop_price) = stop_price {
            order["stopPrice"] = json!(stop_price);
        }
        self.orders.insert(order_id, order.clone());
        self.cl_ord_ids.insert(cl_ord_id, order_id);
        let link_id = self.next_link_id;
        self.next_link_id = self.next_link_id.saturating_add(1);
        let link = json!({
            "id": link_id,
            "orderStrategyId": strategy_id,
            "orderId": order_id,
            "parentOrderId": entry_order_id,
            "accountId": settings.account_id,
            "contractId": settings.contract_id,
            "role": suffix,
        });
        self.strategy_links.insert(link_id, link.clone());
        (order_id, order, link)
    }

    fn process_raw_tick(
        &mut self,
        settings: &Settings,
        tick: &RawTick,
        sequence: u64,
    ) -> Vec<Value> {
        let mut events = Vec::new();

        // A crossed quote is not an executable market state. Validate it
        // before pending entries are considered so a malformed source row
        // cannot fill an order and then be rejected only for protection.
        let Some((bid, ask)) = raw_tick_bid_ask(tick) else {
            return events;
        };
        // Do not let a crossed/invalid source row mutate the executable quote
        // carried into the next tick. `raw_tick_bid_ask` is intentionally the
        // single validation gate for both matching and protection.
        self.set_market_tick(tick, sequence);

        // Snapshot before entry fills. A newly filled parent bracket is
        // armed after the fill, so it must not retroactively trigger against
        // the same source tick that filled its entry.
        let protection_ids = self.protections.keys().copied().collect::<Vec<_>>();

        // Entry market orders become eligible only on a source tick strictly
        // after their acceptance sequence. This removes the wall-clock race
        // between the old fill worker and the replay clock.
        let pending = self
            .pending_fill_quotes
            .iter()
            .filter_map(|(order_id, (_, _, eligible_sequence))| {
                (*eligible_sequence <= sequence).then_some(*order_id)
            })
            .collect::<Vec<_>>();
        for order_id in pending {
            events.extend(self.fill_order_at_tick(settings, order_id, tick, sequence));
        }

        // A direct liquidation, a reversal, or a same-side quantity change
        // invalidates the old bracket's position/entry basis. Cancel its
        // broker-owned children before evaluating the pre-existing bracket
        // snapshot, so a stale protection order cannot close a new position.
        self.cancel_stale_protections(settings, &mut events);

        let Ok(bid_ticks) = price_to_ticks(bid, settings.tick_size) else {
            return events;
        };
        let Ok(ask_ticks) = price_to_ticks(ask, settings.tick_size) else {
            return events;
        };
        let market_tick = MarketTick::new(sequence, bid_ticks, ask_ticks);
        for strategy_id in protection_ids {
            let Some(mut runtime) = self.protections.remove(&strategy_id) else {
                continue;
            };
            let protection_events = match runtime.bracket.on_tick(market_tick) {
                Ok(events) => events,
                Err(error) => {
                    // A malformed/crossed quote must never manufacture a
                    // protection fill. Keep the bracket armed and leave a
                    // diagnostic order event for the trace consumer.
                    self.protections.insert(strategy_id, runtime);
                    let _ = error;
                    continue;
                }
            };
            let mut terminal = false;
            for protection_event in protection_events {
                match protection_event {
                    ProtectionEvent::TrailingActivated { new_stop_ticks, .. }
                    | ProtectionEvent::TrailingRatchet { new_stop_ticks, .. } => {
                        if let Some(order_id) = runtime.stop_order_id
                            && let Some(order) = self.orders.get_mut(&order_id)
                        {
                            if let Some(object) = order.as_object_mut() {
                                object.insert(
                                    "stopPrice".to_string(),
                                    json!(ticks_to_price(new_stop_ticks, settings.tick_size)),
                                );
                                object.insert("replayTrailingActive".to_string(), json!(true));
                            }
                            events.push(props_event("order", order.clone()));
                        }
                    }
                    ProtectionEvent::FillIntent(intent) => {
                        let protection_order_id = match intent.leg {
                            ProtectionLeg::TakeProfit => runtime.take_profit_order_id,
                            ProtectionLeg::StopLoss | ProtectionLeg::TrailingStop => {
                                runtime.stop_order_id
                            }
                        };
                        events.extend(self.fill_protection_order(
                            settings,
                            &runtime,
                            intent,
                            tick.ts_ns,
                            sequence,
                            protection_fill_source(tick, intent.side),
                        ));
                        terminal = protection_order_id.is_some_and(|order_id| {
                            self.orders
                                .get(&order_id)
                                .and_then(|order| order.get("ordStatus"))
                                .and_then(Value::as_str)
                                == Some("Filled")
                        });
                        if !terminal {
                            events.push(self.mark_protection_failure(
                                strategy_id,
                                "broker-owned protection child did not fill".to_string(),
                            ));
                        }
                        if terminal {
                            // There is only one net position in this
                            // synthetic account. A successful protection exit
                            // makes any second pre-snapshot bracket stale.
                            break;
                        }
                    }
                    ProtectionEvent::SiblingCancelled { .. } => {}
                }
            }
            if !terminal && runtime.bracket.status() == BracketStatus::Armed {
                self.protections.insert(strategy_id, runtime);
            }
        }
        // Defer eviction until the complete tick has been processed. In
        // particular, a protection child may be marked Filled by
        // `fill_order_at_price`; pruning it before the caller verifies the
        // terminal child would make the OCO lifecycle appear orphaned when a
        // deliberately small state cap is used in a stress test.
        self.prune_terminal_state(settings.max_state_entities);
        events
    }

    fn cancel_stale_protections(&mut self, settings: &Settings, events: &mut Vec<Value>) {
        let current_position_ticks = self
            .average_price
            .and_then(|price| price_to_ticks(price, settings.tick_size).ok());
        let stale_ids = self
            .protections
            .iter()
            .filter_map(|(strategy_id, runtime)| {
                let protected = runtime.bracket.position();
                let protected_signed = match protected.side {
                    PositionSide::Long => i64::try_from(protected.quantity).ok(),
                    PositionSide::Short => i64::try_from(protected.quantity)
                        .ok()
                        .map(|quantity| -quantity),
                }?;
                let stale = self.position_qty != protected_signed
                    || current_position_ticks != Some(protected.entry_price_ticks)
                    || runtime.position_generation != self.position_generation;
                stale.then_some(*strategy_id)
            })
            .collect::<Vec<_>>();
        for strategy_id in stale_ids {
            let Some(runtime) = self.protections.remove(&strategy_id) else {
                continue;
            };
            for order_id in [runtime.take_profit_order_id, runtime.stop_order_id]
                .into_iter()
                .flatten()
            {
                if let Some(order) = self.cancel_order(order_id) {
                    events.push(props_event("order", order));
                }
            }
            if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
                if let Some(object) = strategy.as_object_mut() {
                    object.insert("status".to_string(), json!("Interrupted"));
                    object.insert("ordStatus".to_string(), json!("Interrupted"));
                    object.insert("exitReason".to_string(), json!("position_changed"));
                }
                events.push(props_event("orderStrategy", strategy.clone()));
            }
        }
    }

    fn fill_protection_order(
        &mut self,
        settings: &Settings,
        runtime: &ProtectionRuntime,
        intent: FillIntent,
        timestamp_ns: i64,
        sequence: u64,
        fill_source: &'static str,
    ) -> Vec<Value> {
        let order_id = match intent.leg {
            ProtectionLeg::TakeProfit => runtime.take_profit_order_id,
            ProtectionLeg::StopLoss | ProtectionLeg::TrailingStop => runtime.stop_order_id,
        };
        let Some(order_id) = order_id else {
            return Vec::new();
        };
        let price = ticks_to_price(intent.fill_price_ticks, settings.tick_size);
        let mut events = self.fill_order_at_price(
            settings,
            order_id,
            price,
            timestamp_ns,
            Some(sequence),
            fill_source,
            Some(match intent.leg {
                ProtectionLeg::TakeProfit => "take_profit",
                ProtectionLeg::StopLoss => "stop_loss",
                ProtectionLeg::TrailingStop => "trailing_stop",
            }),
        );
        let child_filled = self
            .orders
            .get(&order_id)
            .and_then(|order| order.get("ordStatus"))
            .and_then(Value::as_str)
            == Some("Filled");
        if !child_filled {
            return events;
        }
        for sibling_id in [runtime.take_profit_order_id, runtime.stop_order_id]
            .into_iter()
            .flatten()
            .filter(|sibling_id| *sibling_id != order_id)
        {
            if let Some(sibling) = self.cancel_order(sibling_id) {
                events.push(props_event("order", sibling));
            }
        }
        if let Some(strategy) = self.strategies.get_mut(&runtime.strategy_id) {
            if let Some(object) = strategy.as_object_mut() {
                object.insert("status".to_string(), json!("Completed"));
                object.insert("ordStatus".to_string(), json!("Filled"));
                object.insert(
                    "exitReason".to_string(),
                    json!(match intent.leg {
                        ProtectionLeg::TakeProfit => "take_profit",
                        ProtectionLeg::StopLoss => "stop_loss",
                        ProtectionLeg::TrailingStop => "trailing_stop",
                    }),
                );
            }
            events.push(props_event("orderStrategy", strategy.clone()));
        }
        events
    }

    fn cancel_order(&mut self, order_id: i64) -> Option<Value> {
        let order = self.orders.get_mut(&order_id)?;
        let status = order
            .get("ordStatus")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if matches!(
            status,
            "Filled" | "Cancelled" | "Canceled" | "Rejected" | "Expired"
        ) {
            return None;
        }
        if let Some(object) = order.as_object_mut() {
            object.insert("ordStatus".to_string(), json!("Cancelled"));
        }
        self.pending_fill_quotes.remove(&order_id);
        Some(order.clone())
    }

    fn protection_strategy_for_order(&self, order_id: i64) -> Option<i64> {
        self.protections.iter().find_map(|(strategy_id, runtime)| {
            (runtime.take_profit_order_id == Some(order_id)
                || runtime.stop_order_id == Some(order_id))
            .then_some(*strategy_id)
        })
    }

    fn teardown_protection_after_cancel(
        &mut self,
        strategy_id: i64,
        canceled_order_id: i64,
        events: &mut Vec<Value>,
    ) {
        let Some(runtime) = self.protections.remove(&strategy_id) else {
            return;
        };
        for sibling_id in [runtime.take_profit_order_id, runtime.stop_order_id]
            .into_iter()
            .flatten()
            .filter(|sibling_id| *sibling_id != canceled_order_id)
        {
            if let Some(sibling) = self.cancel_order(sibling_id) {
                events.push(props_event("order", sibling));
            }
        }
        if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
            if let Some(object) = strategy.as_object_mut() {
                object.insert("status".to_string(), json!("Interrupted"));
                object.insert("ordStatus".to_string(), json!("Interrupted"));
                object.insert("exitReason".to_string(), json!("protection_cancelled"));
            }
            events.push(props_event("orderStrategy", strategy.clone()));
        }
    }

    fn interrupt_strategy_after_cancel(&mut self, strategy_id: i64, events: &mut Vec<Value>) {
        if let Some(strategy) = self.strategies.get_mut(&strategy_id) {
            if let Some(object) = strategy.as_object_mut() {
                object.insert("status".to_string(), json!("Interrupted"));
                object.insert("ordStatus".to_string(), json!("Interrupted"));
                object.insert("exitReason".to_string(), json!("parent_cancelled"));
            }
            events.push(props_event("orderStrategy", strategy.clone()));
        }
    }

    fn prune_terminal_state(&mut self, max_entities: usize) {
        let terminal = |value: &Value| {
            value
                .get("ordStatus")
                .and_then(Value::as_str)
                .is_some_and(|status| {
                    matches!(
                        status,
                        "Filled" | "Cancelled" | "Canceled" | "Rejected" | "Expired"
                    )
                })
        };
        while self.entity_count() > max_entities {
            if let Some(id) = self
                .orders
                .iter()
                .find_map(|(id, order)| terminal(order).then_some(*id))
            {
                self.orders.remove(&id);
                self.pending_fill_quotes.remove(&id);
                self.cl_ord_ids.retain(|_, order_id| *order_id != id);
                continue;
            }
            if let Some(id) = self.execution_reports.keys().next().copied() {
                self.execution_reports.remove(&id);
                continue;
            }
            if let Some(id) = self.fills.keys().next().copied() {
                self.fills.remove(&id);
                continue;
            }
            if let Some(id) = self.fill_fees.keys().next().copied() {
                self.fill_fees.remove(&id);
                continue;
            }
            let evictable_link = self.strategy_links.iter().find_map(|(id, link)| {
                let strategy_id = link.get("orderStrategyId").and_then(Value::as_i64);
                let active_strategy = strategy_id
                    .and_then(|strategy_id| self.strategies.get(&strategy_id))
                    .and_then(|strategy| strategy.get("status"))
                    .and_then(Value::as_str)
                    == Some("Active");
                let active_protection = strategy_id
                    .is_some_and(|strategy_id| self.protections.contains_key(&strategy_id));
                (!active_strategy && !active_protection).then_some(*id)
            });
            if let Some(id) = evictable_link {
                self.strategy_links.remove(&id);
                continue;
            }
            if let Some(id) = self.strategies.iter().find_map(|(id, strategy)| {
                (strategy.get("status").and_then(Value::as_str) != Some("Active")).then_some(*id)
            }) {
                if let Some(strategy) = self.strategies.remove(&id)
                    && let Some(uuid) = strategy.get("uuid").and_then(Value::as_str)
                {
                    self.strategy_uuids.remove(uuid);
                }
                continue;
            }
            break;
        }
    }

    fn entity_count(&self) -> usize {
        self.orders.len()
            + self.fills.len()
            + self.execution_reports.len()
            + self.fill_fees.len()
            + self.strategies.len()
            + self.strategy_links.len()
    }

    fn apply_fill(&mut self, settings: &Settings, signed_qty: i64, price: f64) {
        if self.position_qty == 0 || self.position_qty.signum() == signed_qty.signum() {
            let old_abs = self.position_qty.unsigned_abs() as f64;
            let new_abs = old_abs + signed_qty.unsigned_abs() as f64;
            let old_average = self.average_price.unwrap_or(price);
            self.average_price = Some(
                ((old_average * old_abs) + (price * signed_qty.unsigned_abs() as f64))
                    / new_abs.max(1.0),
            );
            self.position_qty = self.position_qty.saturating_add(signed_qty);
            return;
        }

        let old_qty = self.position_qty;
        let old_average = self.average_price.unwrap_or(price);
        let close_qty = old_qty.unsigned_abs().min(signed_qty.unsigned_abs()) as f64;
        self.realized_pnl += if old_qty > 0 {
            (price - old_average) * close_qty * settings.value_per_point
        } else {
            (old_average - price) * close_qty * settings.value_per_point
        };
        self.position_qty = self
            .position_qty
            .checked_add(signed_qty)
            .expect("position overflow checked before applying synthetic fill");
        if self.position_qty == 0 {
            self.average_price = None;
        } else if signed_qty.unsigned_abs() > old_qty.unsigned_abs() {
            self.average_price = Some(price);
        }
    }
}

#[derive(Debug, Clone)]
struct UserRequest {
    endpoint: String,
    request_id: u64,
    body: Option<Value>,
}

struct OrderRequest<'a> {
    settings: &'a Settings,
    action: &'a str,
    quantity: i64,
    account_id: i64,
    contract_id: i64,
    symbol: &'a str,
    cl_ord_id: Option<String>,
    strategy_id: Option<i64>,
    bracket: Option<Value>,
}

#[derive(Debug, Clone)]
struct ProtectionRuntime {
    strategy_id: i64,
    bracket: ProtectionBracket,
    take_profit_order_id: Option<i64>,
    stop_order_id: Option<i64>,
    position_generation: u64,
}

fn normalize_action(action: &str) -> Result<&'static str> {
    if action.eq_ignore_ascii_case("buy") {
        Ok("Buy")
    } else if action.eq_ignore_ascii_case("sell") {
        Ok("Sell")
    } else {
        bail!("proxy rejected unsupported action {action}");
    }
}

fn executable_market_price(broker: &BrokerState, action: &str) -> Option<f64> {
    match action.trim().to_ascii_lowercase().as_str() {
        "buy" => broker.last_ask.or(broker.last_price),
        "sell" => broker.last_bid.or(broker.last_price),
        _ => None,
    }
    .filter(|price| price.is_finite() && *price > 0.0)
}

fn raw_tick_executable_price(tick: &RawTick, action: &str) -> Option<(f64, &'static str)> {
    let (bid, ask) = raw_tick_bid_ask(tick)?;
    let quote = if action.eq_ignore_ascii_case("Buy") {
        tick.ask_price
    } else {
        tick.bid_price
    };
    let quote_source = if raw_tick_has_valid_quote(tick) {
        "tick_bid_ask"
    } else if quote.is_some_and(valid_positive_price) {
        "tick_partial_quote"
    } else {
        "tick_trade_fallback"
    };
    quote
        .filter(|price| valid_positive_price(*price))
        .map(|price| (price, quote_source))
        .or_else(|| {
            Some((
                (if action.eq_ignore_ascii_case("Buy") {
                    ask
                } else {
                    bid
                }),
                if raw_tick_has_valid_quote(tick) {
                    "tick_bid_ask"
                } else {
                    "tick_trade_fallback"
                },
            ))
        })
}

fn protection_fill_source(tick: &RawTick, side: crate::protection::OrderSide) -> &'static str {
    let quote = match side {
        crate::protection::OrderSide::Buy => tick.ask_price,
        crate::protection::OrderSide::Sell => tick.bid_price,
    };
    if raw_tick_has_valid_quote(tick) {
        "tick_bid_ask"
    } else if quote.is_some_and(valid_positive_price) {
        "tick_partial_quote"
    } else {
        "tick_trade_fallback"
    }
}

fn replay_execution_precision(fill_source: &str) -> &'static str {
    match fill_source {
        "tick_bid_ask" => "quote_exact",
        "tick_partial_quote" => "tick_partial_quote",
        "tick_trade_fallback" => "trade_time_only",
        _ => "bar_approximate",
    }
}

fn valid_positive_price(price: f64) -> bool {
    price.is_finite() && price > 0.0
}

fn raw_tick_has_valid_quote(tick: &RawTick) -> bool {
    let (Some(bid), Some(ask)) = (tick.bid_price, tick.ask_price) else {
        return false;
    };
    valid_positive_price(bid) && valid_positive_price(ask) && bid <= ask
}

fn raw_tick_bid_ask(tick: &RawTick) -> Option<(f64, f64)> {
    let bid = tick
        .bid_price
        .filter(|price| valid_positive_price(*price))
        .unwrap_or(tick.price);
    let ask = tick
        .ask_price
        .filter(|price| valid_positive_price(*price))
        .unwrap_or(tick.price);
    (bid.is_finite() && ask.is_finite() && bid > 0.0 && ask > 0.0 && bid <= ask)
        .then_some((bid, ask))
}

fn price_to_ticks(price: f64, tick_size: f64) -> Result<i64> {
    if !price.is_finite() || price <= 0.0 || !tick_size.is_finite() || tick_size <= 0.0 {
        bail!("invalid price/tick size for tick conversion");
    }
    let raw = price / tick_size;
    if !raw.is_finite() || raw < i64::MIN as f64 || raw > i64::MAX as f64 {
        bail!("price exceeds integer tick range");
    }
    Ok(raw.round() as i64)
}

fn ticks_to_price(price_ticks: i64, tick_size: f64) -> f64 {
    price_ticks as f64 * tick_size
}

fn parse_protection_params(settings: &Settings, raw: &Value) -> Result<ProtectionParams> {
    let bracket = raw
        .as_array()
        .and_then(|brackets| brackets.first())
        .or_else(|| raw.is_object().then_some(raw))
        .context("proxy strategy brackets must contain an object")?;
    let offset = |key: &str| -> Result<Option<u64>> {
        let Some(value) = bracket.get(key).and_then(Value::as_f64) else {
            return Ok(None);
        };
        if !value.is_finite() || value == 0.0 {
            bail!("proxy bracket {key} must be finite and non-zero");
        }
        let ticks = (value.abs() / settings.tick_size).round();
        if !ticks.is_finite() || ticks < 1.0 || ticks > u64::MAX as f64 {
            bail!("proxy bracket {key} is outside tick range");
        }
        Ok(Some(ticks as u64))
    };
    let take_profit_ticks = offset("profitTarget")?;
    let explicit_stop_ticks = offset("stopLoss")?;
    let trailing = bracket
        .get("autoTrail")
        .and_then(Value::as_object)
        .map(|auto_trail| {
            let trigger = auto_trail
                .get("trigger")
                .and_then(Value::as_f64)
                .context("proxy autoTrail.trigger is missing")?;
            let stop = auto_trail
                .get("stopLoss")
                .and_then(Value::as_f64)
                .context("proxy autoTrail.stopLoss is missing")?;
            let to_ticks = |value: f64, key: &'static str| -> Result<u64> {
                if !value.is_finite() || value <= 0.0 {
                    bail!("proxy autoTrail.{key} must be finite and positive");
                }
                let ticks = (value / settings.tick_size).round();
                if !ticks.is_finite() || ticks < 1.0 || ticks > u64::MAX as f64 {
                    bail!("proxy autoTrail.{key} is outside tick range");
                }
                Ok(ticks as u64)
            };
            Ok::<_, anyhow::Error>(TrailingConfig {
                activation_ticks: to_ticks(trigger, "trigger")?,
                offset_ticks: to_ticks(stop, "stopLoss")?,
                frequency_ticks: to_ticks(
                    auto_trail
                        .get("freq")
                        .and_then(Value::as_f64)
                        .unwrap_or(settings.tick_size),
                    "freq",
                )?,
            })
        })
        .transpose()?;
    let stop_loss_ticks = explicit_stop_ticks.or_else(|| {
        trailing.and_then(|trail| trail.activation_ticks.checked_add(trail.offset_ticks))
    });
    if take_profit_ticks.is_none() && stop_loss_ticks.is_none() && trailing.is_none() {
        bail!("proxy strategy bracket has no usable TP, SL, or trailing leg");
    }
    Ok(ProtectionParams {
        take_profit_ticks,
        stop_loss_ticks,
        trailing,
        exit_precedence: settings.protection_precedence,
    })
}

fn props_event(entity_type: &str, entity: Value) -> Value {
    json!({
        "e": "props",
        "d": {
            "entityType": entity_type,
            "eventType": "Updated",
            "entity": entity,
        }
    })
}

fn response_frame(request_id: u64, status: i64, payload: Value) -> String {
    format!("a{}", json!([{"i": request_id, "s": status, "d": payload}]))
}

fn parse_authorize(raw: &str) -> Option<u64> {
    let mut lines = raw.splitn(4, '\n');
    (lines.next()? == "authorize")
        .then(|| lines.next()?.trim().parse::<u64>().ok())
        .flatten()
}

fn parse_request(raw: &str) -> Result<UserRequest> {
    let mut parts = raw.splitn(4, '\n');
    let endpoint = parts.next().unwrap_or_default().trim();
    let request_id = parts
        .next()
        .context("request missing id")?
        .trim()
        .parse::<u64>()?;
    let _query = parts.next().unwrap_or_default();
    let body = parts
        .next()
        .map(str::trim)
        .filter(|body| !body.is_empty())
        .map(serde_json::from_str)
        .transpose()
        .context("parse request JSON body")?;
    if endpoint.is_empty() {
        bail!("request endpoint is empty");
    }
    Ok(UserRequest {
        endpoint: endpoint.to_string(),
        request_id,
        body,
    })
}

fn request_signature(request: &UserRequest) -> String {
    format!(
        "{}\n{}",
        request.endpoint,
        request
            .body
            .as_ref()
            .map(Value::to_string)
            .unwrap_or_default()
    )
}

fn parse_body_i64(body: Option<&Value>, key: &str, default: i64) -> i64 {
    body.and_then(|body| body.get(key))
        .and_then(|value| {
            value.as_i64().or_else(|| {
                let value = value.as_f64()?;
                (value.is_finite() && value.fract() == 0.0)
                    .then_some(value as i128)
                    .and_then(|value| i64::try_from(value).ok())
            })
        })
        .unwrap_or(default)
}

fn parse_body_string(body: Option<&Value>, key: &str) -> String {
    body.and_then(|body| body.get(key))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn parse_strategy_quantity(params: &Value) -> Result<i64> {
    let entry = params
        .get("entryVersion")
        .context("proxy strategy params are missing entryVersion")?;
    let quantity = entry
        .get("orderQty")
        .and_then(Value::as_i64)
        .context("proxy strategy entryVersion.orderQty must be an integer")?;
    if quantity <= 0 {
        bail!("proxy strategy entryVersion.orderQty must be positive");
    }
    Ok(quantity)
}

pub async fn run(args: Args) -> Result<()> {
    let mut settings = Settings::from_args(&args)?;
    let fixture = load_fixture(
        &settings,
        &settings.fixture_path,
        settings.max_bars,
        settings.tick_fixture_path.as_deref(),
        settings.max_ticks,
        settings.history_bars,
    )?;
    if !settings.contract_id_explicit
        && let Some(contract_id) = fixture.identity.contract_ids.first().copied()
    {
        if contract_id <= 0 || contract_id > MAX_SYNTHETIC_ID_SEED {
            bail!(
                "fixture manifest contract id {contract_id} is outside the bounded synthetic entity ID range"
            );
        }
        settings.contract_id = contract_id;
    }
    if settings.loop_replay && settings.history_bars >= fixture.bars.len() {
        bail!(
            "loop_replay requires at least one realtime fixture bar; reduce history_bars or increase max_bars"
        );
    }
    let trace = TraceSink::new(args.trace.as_deref(), &settings, &fixture)?;
    let (events, _) = broadcast::channel(2048);
    let (fill_tx, fill_rx) = mpsc::channel(1024);
    let mut broker = BrokerState::new(&settings);
    // Seed from the last historical bar, not the first fixture row. This is
    // the quote the engine has actually received by the time it can submit an
    // order after md/getChart completes.
    let history_end = settings.history_bars.min(fixture.bars.len());
    if let Some(ticks) = fixture.ticks.as_ref() {
        if let Some(tick) = ticks.get(fixture.tick_seed_index.saturating_sub(1)) {
            broker.set_market_tick(tick, fixture.tick_seed_index as u64);
        }
    } else if let Some(history_bar) = fixture
        .bars
        .get(history_end.saturating_sub(1))
        .or_else(|| fixture.bars.first())
    {
        broker.set_market(history_bar);
    }
    let shared = Arc::new(Shared {
        broker: Arc::new(Mutex::new(broker)),
        settings: settings.clone(),
        fixture,
        events,
        fill_tx,
        trace,
    });

    tokio::spawn(fill_worker(shared.clone(), fill_rx));

    let rest_listener = TcpListener::bind(settings.rest_bind)
        .await
        .with_context(|| format!("bind REST listener {}", settings.rest_bind))?;
    let user_listener = TcpListener::bind(settings.user_ws_bind)
        .await
        .with_context(|| format!("bind user WebSocket listener {}", settings.user_ws_bind))?;
    let market_listener = TcpListener::bind(settings.market_ws_bind)
        .await
        .with_context(|| format!("bind market WebSocket listener {}", settings.market_ws_bind))?;

    println!("trader-replay-proxy ready");
    println!("fixture: {}", shared.fixture.metadata.path);
    println!("fixture rows: {}", shared.fixture.metadata.rows);
    if let Some(tick_metadata) = shared.fixture.tick_metadata.as_ref() {
        println!("tick fixture: {}", tick_metadata.path);
        println!("tick rows: {}", tick_metadata.rows);
    }
    println!("REST: http://{}/v1", settings.rest_bind);
    println!("user WS: ws://{}/v1/websocket", settings.user_ws_bind);
    println!("market WS: ws://{}/v1/websocket", settings.market_ws_bind);
    println!(
        "source commit: {}",
        std::env::var("TRADER_PROXY_SOURCE_COMMIT").unwrap_or_else(|_| "unknown".to_string())
    );

    let rest_task = tokio::spawn(serve_rest(
        rest_listener,
        shared.clone(),
        Arc::new(Semaphore::new(settings.max_clients)),
    ));
    let user_task = tokio::spawn(serve_user_ws(
        user_listener,
        shared.clone(),
        Arc::new(Semaphore::new(settings.max_clients)),
    ));
    let market_task = tokio::spawn(serve_market_ws(
        market_listener,
        shared.clone(),
        Arc::new(Semaphore::new(settings.max_clients)),
    ));

    tokio::select! {
        result = rest_task => result.context("REST proxy task join")??,
        result = user_task => result.context("user WebSocket proxy task join")??,
        result = market_task => result.context("market WebSocket proxy task join")??,
        result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
    }
    shared.trace.finish();
    Ok(())
}

async fn fill_worker(shared: Arc<Shared>, mut requests: mpsc::Receiver<i64>) {
    // One FIFO worker is intentional. Independent per-order tasks let a
    // staged flatten/confirm/enter sequence fill in different orders across
    // runs, which makes commit comparisons meaningless.
    while let Some(order_id) = requests.recv().await {
        if shared.fixture.ticks.is_some() {
            // Raw-tick replay owns the clock. Filling from a detached task
            // would reintroduce the wall-clock race this mode is designed to
            // remove and could fill an order between two source ticks.
            continue;
        }
        if shared.settings.fill_delay_ms > 0 {
            sleep(Duration::from_millis(shared.settings.fill_delay_ms)).await;
        }
        let events = {
            let Ok(mut broker) = shared.broker.lock() else {
                continue;
            };
            broker.fill_order(&shared.settings, order_id)
        };
        for event in events {
            let _ = shared.events.send(event);
        }
    }
}

async fn serve_rest(
    listener: TcpListener,
    shared: Arc<Shared>,
    clients: Arc<Semaphore>,
) -> Result<()> {
    let mut next_connection_id = 1_u64;
    loop {
        let (stream, peer) = listener.accept().await.context("accept REST client")?;
        let Some(permit) = clients.clone().try_acquire_owned().ok() else {
            let _ = reject_http_client(stream).await;
            continue;
        };
        let shared = shared.clone();
        let connection_id = next_connection_id;
        next_connection_id = next_connection_id.saturating_add(1);
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(err) = handle_rest(stream, shared.clone(), connection_id).await {
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(),
                    protocol: "rest",
                    direction: "error",
                    connection_id,
                    endpoint: None,
                    request_id: None,
                    summary: format!("{peer}: {err}"),
                });
            }
        });
    }
}

async fn reject_http_client(mut stream: TcpStream) -> Result<()> {
    write_http_response(
        &mut stream,
        503,
        json!({"error": "proxy client limit reached"}),
    )
    .await
}

async fn serve_user_ws(
    listener: TcpListener,
    shared: Arc<Shared>,
    clients: Arc<Semaphore>,
) -> Result<()> {
    let mut next_connection_id = 10_000_u64;
    loop {
        let (mut stream, peer) = listener.accept().await.context("accept user WebSocket")?;
        let Some(permit) = clients.clone().try_acquire_owned().ok() else {
            let _ = stream.shutdown().await;
            continue;
        };
        let shared = shared.clone();
        let connection_id = next_connection_id;
        next_connection_id = next_connection_id.saturating_add(1);
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(err) = handle_user_ws(stream, shared.clone(), connection_id).await {
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(),
                    protocol: "user_ws",
                    direction: "error",
                    connection_id,
                    endpoint: None,
                    request_id: None,
                    summary: format!("{peer}: {err}"),
                });
            }
        });
    }
}

async fn serve_market_ws(
    listener: TcpListener,
    shared: Arc<Shared>,
    clients: Arc<Semaphore>,
) -> Result<()> {
    let mut next_connection_id = 20_000_u64;
    loop {
        let (mut stream, peer) = listener.accept().await.context("accept market WebSocket")?;
        let Some(permit) = clients.clone().try_acquire_owned().ok() else {
            let _ = stream.shutdown().await;
            continue;
        };
        let shared = shared.clone();
        let connection_id = next_connection_id;
        next_connection_id = next_connection_id.saturating_add(1);
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(err) = handle_market_ws(stream, shared.clone(), connection_id).await {
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(),
                    protocol: "market_ws",
                    direction: "error",
                    connection_id,
                    endpoint: None,
                    request_id: None,
                    summary: format!("{peer}: {err}"),
                });
            }
        });
    }
}

async fn handle_user_ws(stream: TcpStream, shared: Arc<Shared>, connection_id: u64) -> Result<()> {
    let ws = timeout(WS_HANDSHAKE_TIMEOUT, accept_async(stream))
        .await
        .context("user WebSocket handshake timed out")?
        .context("upgrade user WebSocket")?;
    let (mut write, mut read) = ws.split();
    let mut events = shared.events.subscribe();
    let mut authorized = false;
    let mut seen_responses = HashMap::<u64, (String, String)>::new();
    let mut seen_response_order = VecDeque::<u64>::new();

    loop {
        tokio::select! {
            event = events.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        bail!("user event subscriber lagged by {skipped}; closing fail-closed")
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                let frame = format!("a{}", json!([event]));
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(), protocol: "user_ws", direction: "out", connection_id,
                    endpoint: Some("props".to_string()), request_id: None,
                    summary: TraceSink::text_summary(&frame),
                });
                write.send(Message::Text(frame)).await.context("send user event")?;
            }
            message = read.next() => {
                let Some(message) = message else { break; };
                let message = message.context("read user WebSocket")?;
                match message {
                    Message::Text(raw) => {
                        if raw.trim() == "[]" { continue; }
                        if let Some(request_id) = parse_authorize(&raw) {
                            authorized = true;
                            let frame = response_frame(request_id, 200, json!({"userId": shared.settings.account_id, "name": shared.settings.account_name}));
                            shared.trace.record(TraceRecord {
                                seq: 0,
                                wall_ns: wall_ns(), protocol: "user_ws", direction: "out", connection_id,
                                endpoint: Some("authorize".to_string()), request_id: Some(request_id),
                                summary: TraceSink::text_summary(&frame),
                            });
                            write.send(Message::Text(frame)).await.context("send user authorization")?;
                            continue;
                        }
                        if !authorized { bail!("user request received before authorization"); }
                        let request = parse_request(&raw)?;
                        shared.trace.record(TraceRecord {
                            seq: 0,
                            wall_ns: wall_ns(), protocol: "user_ws", direction: "in", connection_id,
                            endpoint: Some(request.endpoint.clone()), request_id: Some(request.request_id),
                            summary: request.body.as_ref().map(|body| TraceSink::text_summary(&body.to_string())).unwrap_or_default(),
                        });
                        let signature = request_signature(&request);
                        if let Some((seen_signature, frame)) =
                            seen_responses.get(&request.request_id).cloned()
                        {
                            if seen_signature == signature {
                                write
                                    .send(Message::Text(frame))
                                    .await
                                    .context("send duplicate user response")?;
                            } else {
                                let frame = response_frame(
                                    request.request_id,
                                    409,
                                    json!({"failureReason": "request id was reused for a different request"}),
                                );
                                write
                                    .send(Message::Text(frame))
                                    .await
                                    .context("send request id collision response")?;
                            }
                            continue;
                        }
                        let (status, payload, fill_order_id) = handle_user_request(&shared, &request).await;
                        // Queue an accepted fill before attempting to write the
                        // acknowledgement. A client disconnect is not a broker
                        // cancellation; otherwise a dropped socket can leave a
                        // permanently Working order that never reaches the
                        // synthetic fill worker.
                        if let Some(order_id) = fill_order_id {
                            shared
                                .fill_tx
                                .send(order_id)
                                .await
                                .context("queue deterministic synthetic fill")?;
                        }
                        if shared.settings.ack_delay_ms > 0 {
                            sleep(Duration::from_millis(shared.settings.ack_delay_ms)).await;
                        }
                        let frame = response_frame(request.request_id, status, payload);
                        if seen_responses
                            .insert(request.request_id, (signature, frame.clone()))
                            .is_none()
                        {
                            seen_response_order.push_back(request.request_id);
                        }
                        while seen_response_order.len() > MAX_SEEN_RESPONSES {
                            if let Some(old_request_id) = seen_response_order.pop_front() {
                                seen_responses.remove(&old_request_id);
                            }
                        }
                        shared.trace.record(TraceRecord {
                            seq: 0,
                            wall_ns: wall_ns(), protocol: "user_ws", direction: "out", connection_id,
                            endpoint: Some(request.endpoint.clone()), request_id: Some(request.request_id),
                            summary: TraceSink::text_summary(&frame),
                        });
                        write.send(Message::Text(frame)).await.context("send user response")?;
                    }
                    Message::Ping(payload) => { write.send(Message::Pong(payload)).await?; }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

async fn handle_user_request(shared: &Shared, request: &UserRequest) -> (i64, Value, Option<i64>) {
    match request.endpoint.as_str() {
        "user/syncrequest" => {
            let snapshot = shared
                .broker
                .lock()
                .map(|broker| broker.snapshot(&shared.settings))
                .unwrap_or_else(|_| json!({}));
            (200, snapshot, None)
        }
        "order/placeorder" => {
            let body = request.body.as_ref();
            let account_id = parse_body_i64(body, "accountId", 0);
            let contract_id = parse_body_i64(body, "contractId", shared.settings.contract_id);
            let symbol = parse_body_string(body, "symbol");
            let action = parse_body_string(body, "action");
            let quantity = parse_body_i64(body, "orderQty", 0);
            let cl_ord_id = body
                .and_then(|body| body.get("clOrdId"))
                .and_then(Value::as_str)
                .map(ToString::to_string);
            let accepted = shared.broker.lock().ok().and_then(|mut broker| {
                broker
                    .accept_order(OrderRequest {
                        settings: &shared.settings,
                        action: &action,
                        quantity,
                        account_id,
                        contract_id,
                        symbol: &symbol,
                        cl_ord_id,
                        strategy_id: None,
                        bracket: None,
                    })
                    .ok()
            });
            match accepted {
                Some((payload, order_id, events, should_schedule)) => {
                    for event in events {
                        let _ = shared.events.send(event);
                    }
                    (200, payload, should_schedule.then_some(order_id))
                }
                None => (
                    400,
                    json!({"failureReason": "invalid proxy order payload or identity"}),
                    None,
                ),
            }
        }
        "order/liquidateposition" => {
            let liquidation = match shared.broker.lock() {
                Ok(broker) if broker.position_qty > 0 => {
                    Some(("Sell".to_string(), broker.position_qty))
                }
                Ok(broker) if broker.position_qty < 0 => broker
                    .position_qty
                    .checked_abs()
                    .map(|quantity| ("Buy".to_string(), quantity)),
                Ok(_) => Some(("Buy".to_string(), 0)),
                Err(_) => None,
            };
            let Some((action, quantity)) = liquidation else {
                return (
                    500,
                    json!({"failureReason": "proxy broker state is unavailable"}),
                    None,
                );
            };
            if quantity == 0 {
                return (
                    200,
                    json!({"orderId": 0, "id": 0, "alreadyFlat": true}),
                    None,
                );
            }
            let accepted = shared.broker.lock().ok().and_then(|mut broker| {
                broker
                    .accept_order(OrderRequest {
                        settings: &shared.settings,
                        action: &action,
                        quantity,
                        account_id: parse_body_i64(
                            request.body.as_ref(),
                            "accountId",
                            shared.settings.account_id,
                        ),
                        contract_id: parse_body_i64(
                            request.body.as_ref(),
                            "contractId",
                            shared.settings.contract_id,
                        ),
                        symbol: &shared.settings.contract,
                        cl_ord_id: Some(format!("proxy-liquidate-{}", request.request_id)),
                        strategy_id: None,
                        bracket: None,
                    })
                    .ok()
            });
            match accepted {
                Some((payload, order_id, events, should_schedule)) => {
                    for event in events {
                        let _ = shared.events.send(event);
                    }
                    (200, payload, should_schedule.then_some(order_id))
                }
                None => (
                    400,
                    json!({"failureReason": "invalid proxy liquidation payload"}),
                    None,
                ),
            }
        }
        "order/cancelorder" => {
            let order_id = parse_body_i64(request.body.as_ref(), "orderId", 0);
            let mut events = Vec::new();
            let found = shared.broker.lock().ok().and_then(|mut broker| {
                let protection_strategy_id = broker.protection_strategy_for_order(order_id);
                let parent_strategy_id = broker
                    .orders
                    .get(&order_id)
                    .and_then(|order| order.get("orderStrategyId"))
                    .and_then(Value::as_i64)
                    .filter(|strategy_id| Some(*strategy_id) != protection_strategy_id);
                let result = if let Some(order) = broker.cancel_order(order_id) {
                    events.push(props_event("order", order));
                    if let Some(strategy_id) = protection_strategy_id {
                        broker.teardown_protection_after_cancel(strategy_id, order_id, &mut events);
                    }
                    if let Some(strategy_id) = parent_strategy_id {
                        broker.interrupt_strategy_after_cancel(strategy_id, &mut events);
                    }
                    Some(true)
                } else if broker.orders.contains_key(&order_id) {
                    Some(false)
                } else {
                    None
                };
                broker.prune_terminal_state(shared.settings.max_state_entities);
                result
            });
            for event in events {
                let _ = shared.events.send(event);
            }
            match found {
                Some(true) => (
                    200,
                    json!({"orderId": order_id, "id": order_id, "ordStatus": "Cancelled"}),
                    None,
                ),
                Some(false) => (409, json!({"failureReason": "TooLate"}), None),
                None => (404, json!({"failureReason": "unknown order"}), None),
            }
        }
        "orderStrategy/interruptorderstrategy" => {
            let strategy_id = parse_body_i64(request.body.as_ref(), "orderStrategyId", 0);
            let mut events = Vec::new();
            let found = shared.broker.lock().ok().and_then(|mut broker| {
                let strategy = broker.strategies.get_mut(&strategy_id)?;
                if let Some(object) = strategy.as_object_mut() {
                    object.insert("status".to_string(), json!("Interrupted"));
                    object.insert("ordStatus".to_string(), json!("Interrupted"));
                }
                events.push(props_event("orderStrategy", strategy.clone()));
                let linked_order_ids = broker
                    .orders
                    .iter()
                    .filter_map(|(order_id, order)| {
                        (order.get("orderStrategyId").and_then(Value::as_i64) == Some(strategy_id))
                            .then_some(*order_id)
                    })
                    .collect::<Vec<_>>();
                for order_id in linked_order_ids {
                    if let Some(order) = broker.cancel_order(order_id) {
                        events.push(props_event("order", order));
                    }
                }
                // A strategy interrupt is also the broker-side OCO teardown.
                // Do not let a detached in-memory bracket emit an exit on a
                // later raw tick after all of its child orders were canceled.
                broker.protections.remove(&strategy_id);
                broker.prune_terminal_state(shared.settings.max_state_entities);
                Some(())
            });
            for event in events {
                let _ = shared.events.send(event);
            }
            if found.is_some() {
                (
                    200,
                    json!({"orderStrategyId": strategy_id, "id": strategy_id}),
                    None,
                )
            } else {
                (409, json!({"failureReason": "Already inactive"}), None)
            }
        }
        "orderStrategy/startorderstrategy" => {
            let body = request.body.as_ref();
            let action = parse_body_string(body, "action");
            let symbol = parse_body_string(body, "symbol");
            let account_id = parse_body_i64(body, "accountId", 0);
            let contract_id = parse_body_i64(body, "contractId", shared.settings.contract_id);
            let uuid = parse_body_string(body, "uuid");
            let params_raw = body.and_then(|body| body.get("params"));
            let params_raw = match params_raw {
                None => None,
                Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
                    Ok(Value::Object(params)) => Some((raw.clone(), Value::Object(params))),
                    Ok(_) => {
                        return (
                            400,
                            json!({"failureReason": "proxy strategy params must encode a JSON object"}),
                            None,
                        );
                    }
                    Err(_) => {
                        return (
                            400,
                            json!({"failureReason": "proxy strategy params are invalid JSON"}),
                            None,
                        );
                    }
                },
                Some(_) => {
                    return (
                        400,
                        json!({"failureReason": "proxy strategy params must be a serialized JSON string"}),
                        None,
                    );
                }
            };
            let quantity = match params_raw.as_ref() {
                Some((_, params)) => match parse_strategy_quantity(params) {
                    Ok(quantity) => quantity,
                    Err(error) => {
                        return (400, json!({"failureReason": error.to_string()}), None);
                    }
                },
                None => 1,
            };
            let bracket = params_raw
                .as_ref()
                .and_then(|(_, params)| params.get("brackets").cloned());
            let strategy_cl_ord_id = if uuid.is_empty() {
                format!("proxy-strategy-{}", request.request_id)
            } else {
                uuid.clone()
            };
            let accepted = shared.broker.lock().ok().and_then(|mut broker| {
                if !uuid.is_empty()
                    && let Some(strategy_id) = broker.strategy_uuids.get(&uuid).copied()
                {
                    let strategy = broker.strategies.get(&strategy_id)?.clone();
                    let link = broker
                        .strategy_links
                        .values()
                        .find(|link| {
                            link.get("orderStrategyId").and_then(Value::as_i64)
                                == Some(strategy_id)
                        })
                        .cloned()
                        .unwrap_or_else(|| json!({"id": strategy_id.saturating_add(1_000_000), "orderStrategyId": strategy_id}));
                    let order_id = broker
                        .orders
                        .values()
                        .find(|order| {
                            order.get("orderStrategyId").and_then(Value::as_i64)
                                == Some(strategy_id)
                        })
                        .and_then(|order| order.get("id").and_then(Value::as_i64))?;
                    let response = json!({
                        "id": strategy_id,
                        "orderStrategyId": strategy_id,
                        "orderId": order_id,
                        "duplicate": true,
                        "brackets": bracket.clone(),
                    });
                    return Some((
                        strategy_id,
                        (response, order_id, Vec::new(), false),
                        strategy,
                        link,
                        bracket,
                    ));
                }
                // Terminal fills/cancellations are evictable state. Prune
                // before reserving space so a long replay cannot reject a
                // new strategy merely because old terminal entities remain.
                broker.prune_terminal_state(shared.settings.max_state_entities);
                let reservation = if bracket.is_some() { 7 } else { 3 };
                if broker.entity_count().saturating_add(reservation)
                    > shared.settings.max_state_entities
                {
                    return None;
                }
                let strategy_id = broker.next_strategy_id;
                broker.next_strategy_id = broker.next_strategy_id.saturating_add(1);
                let accepted = broker
                    .accept_order(OrderRequest {
                        settings: &shared.settings,
                        action: &action,
                        quantity,
                        account_id,
                        contract_id,
                        symbol: if symbol.is_empty() {
                            &shared.settings.contract
                        } else {
                            &symbol
                        },
                        cl_ord_id: Some(strategy_cl_ord_id.clone()),
                        strategy_id: Some(strategy_id),
                        bracket: bracket.clone(),
                    })
                    .ok()?;
                let (payload, order_id, order_events, should_schedule) = accepted;
                let strategy = json!({
                    "id": strategy_id,
                    "accountId": account_id,
                    "contractId": contract_id,
                    "symbol": if symbol.is_empty() { shared.settings.contract.clone() } else { symbol.clone() },
                    "uuid": uuid,
                    "status": "Active",
                    "ordStatus": "Working",
                    "orderStrategyTypeId": 2,
                    "source": "replay",
                    "params": params_raw.as_ref().map(|(raw, _)| raw.clone()),
                });
                broker.strategies.insert(strategy_id, strategy.clone());
                if !uuid.is_empty() {
                    broker.strategy_uuids.insert(uuid.clone(), strategy_id);
                }
                let link_id = strategy_id.saturating_add(1_000_000);
                let link = json!({
                    "id": link_id,
                    "orderStrategyId": strategy_id,
                    "orderId": order_id,
                    "accountId": account_id,
                    "contractId": contract_id,
                });
                broker.strategy_links.insert(link_id, link.clone());
                Some((
                    strategy_id,
                    (payload, order_id, order_events, should_schedule),
                    strategy,
                    link,
                    bracket,
                ))
            });
            match accepted {
                Some((
                    strategy_id,
                    (_payload, order_id, events, should_schedule),
                    strategy,
                    link,
                    bracket,
                )) => {
                    for event in events {
                        let _ = shared.events.send(event);
                    }
                    let _ = shared.events.send(props_event("orderStrategy", strategy));
                    let _ = shared.events.send(props_event("orderStrategyLink", link));
                    (
                        200,
                        json!({"id": strategy_id, "orderStrategyId": strategy_id, "orderId": order_id, "brackets": bracket}),
                        should_schedule.then_some(order_id),
                    )
                }
                None => (
                    400,
                    json!({"failureReason": "invalid proxy strategy payload or identity"}),
                    None,
                ),
            }
        }
        _ => (
            404,
            json!({"failureReason": format!("proxy does not implement {}", request.endpoint)}),
            None,
        ),
    }
}

async fn handle_market_ws(
    stream: TcpStream,
    shared: Arc<Shared>,
    connection_id: u64,
) -> Result<()> {
    let ws = timeout(WS_HANDSHAKE_TIMEOUT, accept_async(stream))
        .await
        .context("market WebSocket handshake timed out")?
        .context("upgrade market WebSocket")?;
    let (mut write, mut read) = ws.split();
    let mut authorized = false;
    loop {
        let Some(message) = read.next().await else {
            break;
        };
        let message = message.context("read market WebSocket")?;
        match message {
            Message::Text(raw) => {
                if raw.trim() == "[]" {
                    continue;
                }
                if let Some(request_id) = parse_authorize(&raw) {
                    authorized = true;
                    let frame = response_frame(request_id, 200, json!({}));
                    write.send(Message::Text(frame.clone())).await?;
                    shared.trace.record(TraceRecord {
                        seq: 0,
                        wall_ns: wall_ns(),
                        protocol: "market_ws",
                        direction: "out",
                        connection_id,
                        endpoint: Some("authorize".to_string()),
                        request_id: Some(request_id),
                        summary: TraceSink::text_summary(&frame),
                    });
                    continue;
                }
                if !authorized {
                    bail!("market request received before authorization");
                }
                let request = parse_request(&raw)?;
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(),
                    protocol: "market_ws",
                    direction: "in",
                    connection_id,
                    endpoint: Some(request.endpoint.clone()),
                    request_id: Some(request.request_id),
                    summary: request
                        .body
                        .as_ref()
                        .map(|body| TraceSink::text_summary(&body.to_string()))
                        .unwrap_or_default(),
                });
                if request.endpoint != "md/getChart" {
                    let frame = response_frame(
                        request.request_id,
                        404,
                        json!({"failureReason": "proxy only implements md/getChart"}),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                }
                let Some(body) = request.body.as_ref() else {
                    let frame = response_frame(
                        request.request_id,
                        400,
                        json!({"failureReason": "md/getChart requires a request body"}),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                };
                if !chart_request_matches_contract(body, &shared.settings) {
                    let frame = response_frame(
                        request.request_id,
                        409,
                        json!({"failureReason": "md/getChart contract identity does not match the fixture"}),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                }
                let Some(expected_shape) = shared.fixture.bar_shape.as_ref() else {
                    let frame = response_frame(
                        request.request_id,
                        409,
                        json!({
                            "failureReason": "fixture chart shape is unknown; use a manifest market_shape or a bar filename such as 10range"
                        }),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                };
                let shape_matches = body
                    .get("chartDescription")
                    .is_some_and(|description| expected_shape.matches_description(description));
                if !shape_matches {
                    let frame = response_frame(
                        request.request_id,
                        409,
                        json!({
                            "failureReason": format!(
                                "chart fixture shape mismatch: expected {} {} {}",
                                expected_shape.element_size,
                                expected_shape.element_size_unit,
                                expected_shape.underlying_type
                            )
                        }),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                }
                let requested_history = body
                    .get("timeRange")
                    .and_then(|range| range.get("asMuchAsElements"))
                    .and_then(value_as_usize);
                if requested_history != Some(shared.settings.history_bars) {
                    let frame = response_frame(
                        request.request_id,
                        409,
                        json!({
                            "failureReason": format!(
                                "chart fixture history mismatch: expected {} bars, request asked {:?}",
                                shared.settings.history_bars,
                                requested_history
                            )
                        }),
                    );
                    write.send(Message::Text(frame)).await?;
                    continue;
                }
                let (historical, realtime) = shared
                    .fixture
                    .bars
                    .split_at(shared.settings.history_bars.min(shared.fixture.bars.len()));
                let historical_json = historical.iter().map(bar_json).collect::<Vec<_>>();
                let response = response_frame(
                    request.request_id,
                    200,
                    json!({
                        "historicalId": 700_001_i64,
                        "realtimeId": 700_002_i64,
                        "charts": [{"id": 700_001_i64, "bars": historical_json}],
                    }),
                );
                write.send(Message::Text(response.clone())).await?;
                shared.trace.record(TraceRecord {
                    seq: 0,
                    wall_ns: wall_ns(),
                    protocol: "market_ws",
                    direction: "out",
                    connection_id,
                    endpoint: Some("md/getChart".to_string()),
                    request_id: Some(request.request_id),
                    summary: format!(
                        "historical_bars={} realtime_bars={}",
                        historical.len(),
                        realtime.len()
                    ),
                });
                if let Some(start_file) = shared.settings.start_file.as_deref() {
                    shared.trace.record(TraceRecord {
                        seq: 0,
                        wall_ns: wall_ns(),
                        protocol: "market_ws",
                        direction: "wait",
                        connection_id,
                        endpoint: Some("replay/start-file".to_string()),
                        request_id: None,
                        summary: format!("waiting for marker {}", start_file.display()),
                    });
                    if !wait_for_start_file(&mut write, &mut read, start_file).await? {
                        break;
                    }
                    shared.trace.record(TraceRecord {
                        seq: 0,
                        wall_ns: wall_ns(),
                        protocol: "market_ws",
                        direction: "start",
                        connection_id,
                        endpoint: Some("replay/start-file".to_string()),
                        request_id: None,
                        summary: format!(
                            "marker present {}; starting realtime",
                            start_file.display()
                        ),
                    });
                }
                let stream_open = stream_realtime_bars(
                    &mut write,
                    &mut read,
                    &shared,
                    connection_id,
                    historical.last().map(|bar| bar.ts_ns),
                    realtime,
                )
                .await?;
                if !stream_open {
                    break;
                }
                if !shared.settings.loop_replay {
                    // Finish a finite replay with a proper WebSocket close
                    // frame. Dropping the socket here looks like a transport
                    // reset to clients and masks a successful finite test.
                    write.send(Message::Close(None)).await?;
                    break;
                }
            }
            Message::Ping(payload) => {
                write.send(Message::Pong(payload)).await?;
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

struct TickCursor {
    ticks: Arc<Vec<RawTick>>,
    next_index: usize,
    end_index: usize,
    sequence: u64,
    previous_ts_ns: Option<i64>,
}

async fn stream_realtime_bars<S, R>(
    write: &mut S,
    read: &mut R,
    shared: &Shared,
    connection_id: u64,
    initial_previous_ts_ns: Option<i64>,
    bars: &[Bar],
) -> Result<bool>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
    R: Stream<Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let mut previous_ts_ns = initial_previous_ts_ns;
    let mut tick_cursor = shared.fixture.ticks.as_ref().map(|ticks| TickCursor {
        ticks: Arc::clone(ticks),
        next_index: shared.fixture.tick_seed_index,
        end_index: shared.fixture.tick_end_index,
        sequence: shared.fixture.tick_seed_index as u64,
        previous_ts_ns: ticks
            .get(shared.fixture.tick_seed_index.saturating_sub(1))
            .map(|tick| tick.ts_ns),
    });
    loop {
        if bars.is_empty() {
            if shared.settings.loop_replay {
                // Avoid a hot loop when history_bars consumes the whole
                // fixture. An empty realtime stream is still a valid
                // finite replay, not permission to spin at 100% CPU.
                if !wait_for_replay_delay(
                    write,
                    read,
                    Duration::from_millis(shared.settings.loop_boundary_delay_ms),
                )
                .await?
                {
                    return Ok(false);
                }
                continue;
            }
            return Ok(true);
        }
        let mut first_bar_in_cycle = true;
        for bar in bars {
            if let Some(cursor) = tick_cursor.as_mut() {
                // Raw ticks are the execution clock. The chart bar remains
                // the strategy's input, while every source tick between chart
                // updates advances the synthetic broker and can fill entries,
                // TP/SL, or trailing protection.
                let inclusive = matches!(
                    shared.settings.raw_tick_bar_timestamps,
                    RawTickBarTimestampMode::Close
                );
                if !replay_ticks_until(
                    write,
                    read,
                    shared,
                    connection_id,
                    cursor,
                    bar.ts_ns,
                    inclusive,
                )
                .await?
                {
                    return Ok(false);
                }
                // A chart timestamp is also a replay-clock event. If no raw
                // tick landed exactly at this boundary, advance the virtual
                // clock to the bar before publishing it; otherwise bars can
                // burst together after a quiet tick interval.
                if let Some(previous_ts_ns) = cursor.previous_ts_ns
                    && previous_ts_ns < bar.ts_ns
                {
                    let delay_ms = replay_delay_ms(
                        previous_ts_ns,
                        bar.ts_ns,
                        shared.settings.speed,
                        shared.settings.max_sleep_ms,
                    );
                    if delay_ms > 0
                        && !wait_for_replay_delay(write, read, Duration::from_millis(delay_ms))
                            .await?
                    {
                        return Ok(false);
                    }
                    cursor.previous_ts_ns = Some(bar.ts_ns);
                }
            } else {
                if let Some(previous_ts_ns) = previous_ts_ns {
                    let delay_ms = if shared.settings.speed > 0.0 {
                        replay_delay_ms(
                            previous_ts_ns,
                            bar.ts_ns,
                            shared.settings.speed,
                            shared.settings.max_sleep_ms,
                        )
                    } else {
                        0
                    };
                    // A one-bar loop (or a correction with the same timestamp)
                    // still needs a bounded yield. Without this, --loop-replay
                    // can flood a client at several thousand frames per second.
                    let delay_ms =
                        if shared.settings.loop_replay && first_bar_in_cycle && delay_ms == 0 {
                            shared.settings.loop_boundary_delay_ms
                        } else {
                            delay_ms
                        };
                    if delay_ms > 0
                        && !wait_for_replay_delay(write, read, Duration::from_millis(delay_ms))
                            .await?
                    {
                        return Ok(false);
                    }
                }
                previous_ts_ns = Some(bar.ts_ns);
                if let Ok(mut broker) = shared.broker.lock() {
                    broker.set_market(bar);
                }
            }
            first_bar_in_cycle = false;
            let frame = format!(
                "a{}",
                json!([{"d": {"charts": [{"id": 700_002_i64, "bars": [bar_json(bar)]}]}}])
            );
            shared.trace.record(TraceRecord {
                seq: 0,
                wall_ns: wall_ns(),
                protocol: "market_ws",
                direction: "out",
                connection_id,
                endpoint: Some("md/realtime".to_string()),
                request_id: None,
                summary: format!("bar ts_ns={} close={}", bar.ts_ns, bar.close),
            });
            write
                .send(Message::Text(frame))
                .await
                .map_err(|err| anyhow::anyhow!(err))?;
            // Give the strategy/user socket a scheduling opportunity after
            // each chart update even for unpaced fixtures. This is a yield,
            // not an artificial trading delay.
            tokio::task::yield_now().await;
        }
        if !shared.settings.loop_replay {
            break;
        }
        if let Some(cursor) = tick_cursor.as_mut() {
            cursor.next_index = shared.fixture.tick_seed_index;
            cursor.previous_ts_ns = shared
                .fixture
                .tick_seed_index
                .checked_sub(1)
                .and_then(|index| cursor.ticks.get(index).map(|tick| tick.ts_ns));
        }
        previous_ts_ns = initial_previous_ts_ns;
    }
    Ok(true)
}

async fn wait_for_start_file<S, R>(write: &mut S, read: &mut R, path: &Path) -> Result<bool>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
    R: Stream<Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        if path.is_file() {
            return Ok(true);
        }
        tokio::select! {
            _ = sleep(Duration::from_millis(25)) => {}
            message = read.next() => match message {
                Some(Ok(Message::Ping(payload))) => {
                    write.send(Message::Pong(payload)).await.map_err(|err| anyhow::anyhow!(err))?;
                }
                Some(Ok(Message::Close(_))) | None => return Ok(false),
                Some(Ok(_)) => {}
                Some(Err(err)) => return Err(anyhow::anyhow!("market websocket read failed while waiting for replay start: {err}")),
            }
        }
    }
}

async fn replay_ticks_until<S, R>(
    write: &mut S,
    read: &mut R,
    shared: &Shared,
    connection_id: u64,
    cursor: &mut TickCursor,
    boundary_ts_ns: i64,
    inclusive: bool,
) -> Result<bool>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
    R: Stream<Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    while cursor.next_index < cursor.end_index
        && let Some(tick) = cursor.ticks.get(cursor.next_index).cloned()
    {
        let eligible = if inclusive {
            tick.ts_ns <= boundary_ts_ns
        } else {
            tick.ts_ns < boundary_ts_ns
        };
        if !eligible {
            break;
        }
        if let Some(previous_ts_ns) = cursor.previous_ts_ns {
            let delay_ms = replay_delay_ms(
                previous_ts_ns,
                tick.ts_ns,
                shared.settings.speed,
                shared.settings.max_sleep_ms,
            );
            if delay_ms > 0
                && !wait_for_replay_delay(write, read, Duration::from_millis(delay_ms)).await?
            {
                return Ok(false);
            }
        }
        cursor.sequence = cursor.sequence.saturating_add(1);
        let sequence = cursor.sequence;
        let events = {
            let Ok(mut broker) = shared.broker.lock() else {
                return Ok(false);
            };
            broker.process_raw_tick(&shared.settings, &tick, sequence)
        };
        for event in events {
            let _ = shared.events.send(event);
        }
        shared.trace.record(TraceRecord {
            seq: 0,
            wall_ns: wall_ns(),
            protocol: "raw_tick",
            direction: "out",
            connection_id,
            endpoint: Some("replay/tick".to_string()),
            request_id: None,
            summary: format!(
                "sequence={} tick_id={} ts_ns={} price={} bid={} ask={}",
                sequence,
                tick.tick_id
                    .map(|id| id.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                tick.ts_ns,
                tick.price,
                tick.bid_price
                    .map(|price| price.to_string())
                    .unwrap_or_else(|| "fallback".to_string()),
                tick.ask_price
                    .map(|price| price.to_string())
                    .unwrap_or_else(|| "fallback".to_string()),
            ),
        });
        cursor.previous_ts_ns = Some(tick.ts_ns);
        cursor.next_index = cursor.next_index.saturating_add(1);
        tokio::task::yield_now().await;
    }
    Ok(true)
}

async fn wait_for_replay_delay<S, R>(write: &mut S, read: &mut R, delay: Duration) -> Result<bool>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::error::Error + Send + Sync + 'static,
    R: Stream<Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    if delay.is_zero() {
        tokio::task::yield_now().await;
        return Ok(true);
    }

    let deadline = Instant::now() + delay;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(true);
        }
        tokio::select! {
            _ = sleep(remaining) => return Ok(true),
            message = read.next() => match message {
                Some(Ok(Message::Ping(payload))) => {
                    write.send(Message::Pong(payload)).await.map_err(|err| anyhow::anyhow!(err))?;
                }
                Some(Ok(Message::Close(_))) | None => return Ok(false),
                Some(Ok(_)) => {}
                Some(Err(err)) => return Err(anyhow::anyhow!("market websocket read failed during replay pacing: {err}")),
            }
        }
    }
}

fn replay_delay_ms(previous_ts_ns: i64, current_ts_ns: i64, speed: f64, max_sleep_ms: u64) -> u64 {
    if speed <= 0.0 || !speed.is_finite() {
        return 0;
    }
    let delta_ns = current_ts_ns.saturating_sub(previous_ts_ns).max(0) as f64;
    ((delta_ns / 1_000_000.0) / speed)
        .ceil()
        .min(max_sleep_ms as f64) as u64
}

fn bar_json(bar: &Bar) -> Value {
    let mut value = json!({
        "timestamp": bar.timestamp.to_rfc3339(),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
    });
    if let Some(volume) = bar.volume {
        value["volume"] = json!(volume);
    }
    value
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

async fn handle_rest(mut stream: TcpStream, shared: Arc<Shared>, connection_id: u64) -> Result<()> {
    let request = timeout(HTTP_READ_TIMEOUT, read_http_request(&mut stream))
        .await
        .context("HTTP request read timed out")??;
    let path = request.path.split('?').next().unwrap_or(&request.path);
    shared.trace.record(TraceRecord {
        seq: 0,
        wall_ns: wall_ns(),
        protocol: "rest",
        direction: "in",
        connection_id,
        endpoint: Some(path.to_string()),
        request_id: None,
        summary: format!("{} body_bytes={}", request.method, request.body.len()),
    });
    if shared.settings.rest_delay_ms > 0 {
        sleep(Duration::from_millis(shared.settings.rest_delay_ms)).await;
    }
    let (status, payload) = rest_response(&request, &shared);
    write_http_response(&mut stream, status, payload.clone()).await?;
    shared.trace.record(TraceRecord {
        seq: 0,
        wall_ns: wall_ns(),
        protocol: "rest",
        direction: "out",
        connection_id,
        endpoint: Some(path.to_string()),
        request_id: None,
        summary: format!(
            "status={status} payload={}",
            TraceSink::text_summary(&payload.to_string())
        ),
    });
    Ok(())
}

fn rest_response(request: &HttpRequest, shared: &Shared) -> (u16, Value) {
    let path = request.path.split('?').next().unwrap_or(&request.path);
    let path = path.strip_prefix("/v1").unwrap_or(path);
    let broker = match shared.broker.lock() {
        Ok(broker) => broker,
        Err(_) => return (503, json!({"failureReason": "proxy state lock poisoned"})),
    };
    match (request.method.as_str(), path) {
        ("POST", "/auth/accesstokenrequest") => (
            200,
            json!({
                "accessToken": "replay-proxy-token",
                "mdAccessToken": "replay-proxy-token",
                "expirationTime": "2099-01-01T00:00:00Z",
                "userId": shared.settings.account_id,
                "name": shared.settings.account_name,
            }),
        ),
        ("GET", "/auth/me") => (
            200,
            json!({
                "id": shared.settings.account_id,
                "userId": shared.settings.account_id,
                "name": shared.settings.account_name,
            }),
        ),
        ("GET", "/auth/renewAccessToken") => (
            200,
            json!({
                "accessToken": "replay-proxy-token",
                "mdAccessToken": "replay-proxy-token",
                "expirationTime": "2099-01-01T00:00:00Z",
                "userId": shared.settings.account_id,
                "name": shared.settings.account_name,
            }),
        ),
        ("GET", "/account/list") => (200, json!([broker.account_entity(&shared.settings)])),
        ("GET", "/accountRiskStatus/list") => (
            200,
            json!([{
                "id": shared.settings.account_id,
                "accountId": shared.settings.account_id,
                "marginUsed": 0.0
            }]),
        ),
        ("GET", "/cashBalance/list") => (200, json!([broker.cash_entity(&shared.settings)])),
        ("GET", "/position/list") => (200, json!([broker.position_entity(&shared.settings)])),
        ("GET", "/order/list") => (
            200,
            json!(broker.orders.values().cloned().collect::<Vec<_>>()),
        ),
        ("GET", "/order/item") => match query_i64(&request.path, "id")
            .and_then(|order_id| broker.orders.get(&order_id).cloned())
        {
            Some(order) => (200, order),
            None => (404, json!({"failureReason": "unknown order"})),
        },
        ("GET", "/orderVersion/deps") => match query_i64(&request.path, "masterid")
            .and_then(|order_id| broker.orders.get(&order_id))
        {
            Some(order) => {
                let version = json!({
                    "id": order
                        .get("id")
                        .and_then(Value::as_i64)
                        .unwrap_or_default()
                        .saturating_mul(10),
                    "orderId": order.get("id").and_then(Value::as_i64),
                    "orderQty": order.get("orderQty").and_then(Value::as_i64),
                    "orderType": order.get("orderType").and_then(Value::as_str),
                    "price": order.get("avgFillPrice").and_then(Value::as_f64),
                });
                (200, json!([version]))
            }
            None => (404, json!({"failureReason": "unknown order"})),
        },
        ("GET", "/orderStrategyLink/deps") => {
            let strategy_id = query_i64(&request.path, "masterid");
            let links = broker
                .strategy_links
                .values()
                .filter(|link| {
                    strategy_id.is_none()
                        || link.get("orderStrategyId").and_then(Value::as_i64) == strategy_id
                })
                .cloned()
                .collect::<Vec<_>>();
            (200, json!(links))
        }
        ("GET", "/fill/list") => (
            200,
            json!(broker.fills.values().cloned().collect::<Vec<_>>()),
        ),
        ("GET", "/executionReport/list") => (
            200,
            json!(
                broker
                    .execution_reports
                    .values()
                    .cloned()
                    .collect::<Vec<_>>()
            ),
        ),
        ("GET", "/fillFee/list") => (
            200,
            json!(broker.fill_fees.values().cloned().collect::<Vec<_>>()),
        ),
        ("GET", "/orderStrategy/list") => (
            200,
            json!(broker.strategies.values().cloned().collect::<Vec<_>>()),
        ),
        ("GET", "/orderStrategyLink/list") => (
            200,
            json!(broker.strategy_links.values().cloned().collect::<Vec<_>>()),
        ),
        ("GET", "/command/list") => (200, json!([])),
        ("GET", "/contract/suggest") => (200, json!([contract_json(&shared.settings)])),
        ("GET", "/contractMaturity/items") => (200, json!([maturity_json(&shared.settings)])),
        ("GET", "/contractMaturity/item") => (200, maturity_json(&shared.settings)),
        ("GET", "/product/item") => (200, product_json(&shared.settings)),
        _ => (
            404,
            json!({"failureReason": format!("proxy does not implement {} {}", request.method, request.path)}),
        ),
    }
}

fn contract_json(settings: &Settings) -> Value {
    json!({
        "id": settings.contract_id,
        "name": settings.contract,
        "description": format!("offline replay contract {}", settings.contract),
        "contractMaturityId": settings.contract_id.saturating_add(10),
        "productId": settings.contract_id.saturating_add(20),
    })
}

fn maturity_json(settings: &Settings) -> Value {
    json!({
        "id": settings.contract_id.saturating_add(10),
        "productId": settings.contract_id.saturating_add(20),
        "contractId": settings.contract_id,
        "name": settings.contract,
        "expirationDate": "2099-12-31T00:00:00Z",
    })
}

fn product_json(settings: &Settings) -> Value {
    json!({
        "id": settings.contract_id.saturating_add(20),
        "name": settings.contract.trim_end_matches(['U', 'Z', 'N', 'Q', 'G', 'J', 'M', 'V', 'H', 'K']).to_string(),
        "productType": "futures",
        "valuePerPoint": settings.value_per_point,
        "tickSize": settings.tick_size,
        "minTick": settings.tick_size,
    })
}

fn query_i64(path: &str, key: &str) -> Option<i64> {
    path.split_once('?')?
        .1
        .split('&')
        .filter_map(|part| part.split_once('='))
        .find_map(|(name, value)| (name == key).then(|| value.parse().ok()))
        .flatten()
}

async fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::with_capacity(4096);
    let header_end;
    loop {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.context("read HTTP request")?;
        if read == 0 {
            bail!("HTTP client closed before headers");
        }
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.len() > MAX_HTTP_HEADER_BYTES {
            bail!("HTTP headers exceed proxy limit");
        }
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
    }
    let header_text =
        std::str::from_utf8(&buffer[..header_end]).context("HTTP headers are not UTF-8")?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().context("HTTP request line missing")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .context("HTTP method missing")?
        .to_string();
    let path = request_parts
        .next()
        .context("HTTP path missing")?
        .to_string();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find_map(|(key, value)| {
            key.eq_ignore_ascii_case("content-length")
                .then_some(value.trim())
        })
        .map(|value| value.parse::<usize>().context("parse Content-Length"))
        .transpose()?
        .unwrap_or(0);
    if content_length > MAX_HTTP_BODY_BYTES {
        bail!("HTTP body exceeds proxy limit");
    }
    while buffer.len() < header_end + content_length {
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.context("read HTTP body")?;
        if read == 0 {
            bail!("HTTP client closed in body");
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: buffer[header_end..header_end + content_length].to_vec(),
    })
}

async fn write_http_response(stream: &mut TcpStream, status: u16, payload: Value) -> Result<()> {
    let body = serde_json::to_vec(&payload)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        503 => "Service Unavailable",
        _ => "Error",
    };
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    Ok(())
}

fn load_fixture(
    settings: &Settings,
    path: &Path,
    max_bars: usize,
    tick_path: Option<&Path>,
    max_ticks: usize,
    history_bars: usize,
) -> Result<Fixture> {
    let sha256 = hash_file(path)?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let bars = match extension.as_str() {
        "parquet" => read_parquet(path, max_bars)?,
        "csv" => read_csv_file(path, max_bars)?,
        _ => read_jsonl_file(path, max_bars)?,
    };
    if bars.is_empty() {
        bail!("replay fixture {} contains no bars", path.display());
    }
    for (index, bar) in bars.iter().enumerate() {
        validate_bar(bar).with_context(|| format!("validate fixture row {index}"))?;
    }
    if tick_path.is_some()
        && bars
            .windows(2)
            .any(|window| window[1].ts_ns < window[0].ts_ns)
    {
        bail!(
            "raw-tick replay requires non-decreasing chart timestamps; range-bar corrections must be exported as explicit correction events before they can be served tick-by-tick"
        );
    }
    let inferred_bar_shape = infer_bar_shape(path);
    let identity = validate_fixture_identity(
        settings,
        path,
        tick_path,
        settings.fixture_manifest.as_deref(),
        settings.allow_unverified_fixture,
        inferred_bar_shape.as_ref(),
    )?;
    let metadata = FixtureMetadata {
        path: path.display().to_string(),
        sha256,
        rows: bars.len(),
        first_ts_ns: bars.first().map(|bar| bar.ts_ns),
        last_ts_ns: bars.last().map(|bar| bar.ts_ns),
    };
    let (ticks, tick_metadata) = if let Some(tick_path) = tick_path {
        let tick_fixture = crate::tick::load(
            tick_path,
            TickLoadOptions {
                max_ticks,
                ordering: TickOrdering::NonDecreasingTimestamp,
                duplicate_tick_ids: DuplicateTickIds::Reject,
            },
        )?;
        let ticks = tick_fixture.into_ticks();
        let quote_rows = ticks
            .iter()
            .filter(|tick| raw_tick_has_valid_quote(tick))
            .count();
        if settings.require_quote_ticks && quote_rows != ticks.len() {
            bail!(
                "raw-tick fixture {} contains {} rows without a valid bid/ask quote; {} quote rows available (omit --require-quote-ticks to use the explicit trade/partial-quote fallback)",
                tick_path.display(),
                ticks.len().saturating_sub(quote_rows),
                quote_rows,
            );
        }
        if quote_rows != ticks.len() {
            eprintln!(
                "warning: raw-tick fixture {} has {}/{} rows with both bid and ask; fills without a complete quote are explicitly marked as tick fallback",
                tick_path.display(),
                quote_rows,
                ticks.len(),
            );
        }
        let tick_metadata = TickFixtureMetadata {
            path: tick_path.display().to_string(),
            sha256: hash_file(tick_path)?,
            rows: ticks.len(),
            first_ts_ns: ticks.first().map(|tick| tick.ts_ns),
            last_ts_ns: ticks.last().map(|tick| tick.ts_ns),
            quote_rows,
        };
        (Some(Arc::new(ticks)), Some(tick_metadata))
    } else {
        (None, None)
    };
    let history_end = history_bars.min(bars.len());
    let first_realtime_ts_ns = bars.get(history_end).map(|bar| bar.ts_ns);
    // The cursor starts at the first tick belonging to the realtime portion.
    // The timestamp mode is applied by the stream when it decides whether
    // the current bar consumes ticks before or through its timestamp. Seeding
    // with `<=` for close-stamped bars would silently consume the first live
    // bar's ticks before the first realtime update was published.
    let tick_seed_index = ticks.as_ref().map_or(0, |ticks| {
        first_realtime_ts_ns.map_or(ticks.len(), |boundary| {
            ticks.partition_point(|tick| tick.ts_ns < boundary)
        })
    });
    let tick_end_index = ticks.as_ref().map_or(0, |ticks| {
        bars.last().map_or(0, |last_bar| match settings.raw_tick_bar_timestamps {
            RawTickBarTimestampMode::Start => {
                ticks.partition_point(|tick| tick.ts_ns < last_bar.ts_ns)
            }
            RawTickBarTimestampMode::Close => {
                ticks.partition_point(|tick| tick.ts_ns <= last_bar.ts_ns)
            }
        })
    });
    if let Some(ticks) = ticks.as_ref()
        && first_realtime_ts_ns.is_some()
    {
        validate_tick_coverage(
            ticks,
            &bars,
            history_end,
            tick_seed_index,
            tick_end_index,
            tick_path.expect("tick path exists when ticks are loaded"),
        )?;
    }
    let bar_shape = inferred_bar_shape.or_else(|| identity.bar_shapes.first().cloned());
    Ok(Fixture {
        bars: Arc::new(bars),
        metadata,
        identity,
        bar_shape,
        ticks,
        tick_metadata,
        tick_seed_index,
        tick_end_index,
    })
}

fn validate_tick_coverage(
    ticks: &[RawTick],
    bars: &[Bar],
    history_end: usize,
    tick_seed_index: usize,
    tick_end_index: usize,
    tick_path: &Path,
) -> Result<()> {
    let Some(first_realtime_bar) = bars.get(history_end) else {
        return Ok(());
    };
    let Some(last_realtime_bar) = bars.last() else {
        return Ok(());
    };
    let valid = |tick: &RawTick| raw_tick_bid_ask(tick).is_some();
    let seed_end = tick_seed_index.min(ticks.len());
    if !ticks[..seed_end].iter().any(valid) {
        bail!(
            "raw-tick fixture {} has no valid executable tick strictly before the first realtime bar (ts_ns={}); refusing a replay with an unseeded broker quote",
            tick_path.display(),
            first_realtime_bar.ts_ns,
        );
    }

    let end_index = tick_end_index.min(ticks.len());
    let start_index = tick_seed_index.min(end_index);
    if !ticks[start_index..end_index].iter().any(valid) {
        bail!(
            "raw-tick fixture {} has no valid executable ticks in the selected realtime window",
            tick_path.display()
        );
    }

    let has_tick_at_or_after_last = ticks
        .iter()
        .any(|tick| tick.ts_ns >= last_realtime_bar.ts_ns && valid(tick));
    if !has_tick_at_or_after_last {
        bail!(
            "raw-tick fixture {} ends before the last realtime bar (ts_ns={}); refusing an incomplete execution tape",
            tick_path.display(),
            last_realtime_bar.ts_ns,
        );
    }
    Ok(())
}

fn infer_bar_shape(path: &Path) -> Option<ChartShape> {
    let stem = path.file_stem()?.to_str()?;
    for component in stem.split('_').rev() {
        let Some(kind) = ["minute", "second", "range", "volume", "tick"]
            .into_iter()
            .find(|kind| component.ends_with(kind))
        else {
            continue;
        };
        let Some(number) = component
            .strip_suffix(kind)
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        if number == 0 {
            return None;
        }
        let (underlying_type, element_size_unit) = match kind {
            "minute" => ("MinuteBar", "UnderlyingUnits"),
            "second" => ("Tick", "Seconds"),
            "range" => ("Tick", "Range"),
            "volume" => ("Tick", "Volume"),
            "tick" => ("Tick", "UnderlyingUnits"),
            _ => return None,
        };
        return Some(ChartShape {
            underlying_type,
            element_size: number,
            element_size_unit,
        });
    }
    None
}

#[derive(Debug, Clone, Default)]
struct ManifestIdentity {
    path: PathBuf,
    contract_symbol: Option<String>,
    contract_id: Option<i64>,
    tick_size: Option<f64>,
    value_per_point: Option<f64>,
    files: Vec<ManifestFile>,
}

#[derive(Debug, Clone)]
struct ManifestFile {
    relative_path: String,
    source_kind: Option<String>,
    first_ts_ns: Option<i64>,
    last_ts_ns: Option<i64>,
    row_count: Option<usize>,
    data_hash_algorithm: Option<String>,
    data_hash_value: Option<String>,
    bar_shape: Option<ChartShape>,
}

impl ManifestFile {
    fn window(&self) -> Option<(i64, i64)> {
        let (Some(first), Some(last)) = (self.first_ts_ns, self.last_ts_ns) else {
            return None;
        };
        (first <= last).then_some((first, last))
    }
}

fn validate_fixture_identity(
    settings: &Settings,
    bar_path: &Path,
    tick_path: Option<&Path>,
    explicit_manifest: Option<&Path>,
    allow_unverified: bool,
    inferred_bar_shape: Option<&ChartShape>,
) -> Result<FixtureIdentityMetadata> {
    let mut manifest_paths = Vec::<PathBuf>::new();
    if let Some(path) = explicit_manifest {
        manifest_paths.push(path.to_path_buf());
    } else {
        if let Some(path) = discover_manifest(bar_path) {
            manifest_paths.push(path);
        }
        if let Some(path) = tick_path.and_then(discover_manifest)
            && !manifest_paths.iter().any(|existing| existing == &path)
        {
            manifest_paths.push(path);
        }
    }

    let identities = manifest_paths
        .iter()
        .map(|path| read_manifest_identity(path))
        .collect::<Result<Vec<_>>>()?;

    let bar_binding = find_manifest_file(&identities, bar_path, "server_bars")?;
    let tick_binding = if let Some(path) = tick_path {
        find_manifest_file(&identities, path, "raw_ticks")?
    } else {
        None
    };
    if let (Some((bar_manifest, _)), Some((tick_manifest, _))) = (bar_binding, tick_binding)
        && bar_manifest.path != tick_manifest.path
    {
        bail!(
            "bars and raw ticks must come from the same fixture manifest; got {} and {}",
            bar_manifest.path.display(),
            tick_manifest.path.display()
        );
    }
    if let (Some((_, bar_file)), Some((_, tick_file))) = (bar_binding, tick_binding)
        && let (Some((bar_first, bar_last)), Some((tick_first, tick_last))) =
            (bar_file.window(), tick_file.window())
        && (tick_last < bar_first || tick_first > bar_last)
    {
        bail!(
            "raw-tick fixture {} does not overlap the selected bar fixture {}",
            tick_path.map_or_else(
                || "<unknown>".to_string(),
                |path| path.display().to_string()
            ),
            bar_path.display()
        );
    }

    let mut metadata = FixtureIdentityMetadata {
        manifests: identities
            .iter()
            .map(|identity| identity.path.display().to_string())
            .collect(),
        contract_symbols: identities
            .iter()
            .filter_map(|identity| identity.contract_symbol.clone())
            .collect(),
        contract_ids: identities
            .iter()
            .filter_map(|identity| identity.contract_id)
            .collect(),
        tick_sizes: identities
            .iter()
            .filter_map(|identity| identity.tick_size)
            .collect(),
        value_per_points: identities
            .iter()
            .filter_map(|identity| identity.value_per_point)
            .collect(),
        bar_shapes: bar_binding
            .and_then(|(_, file)| file.bar_shape.clone())
            .into_iter()
            .collect(),
        verified: false,
    };

    if let Some(manifest_shape) = metadata.bar_shapes.first()
        && let Some(inferred_shape) = inferred_bar_shape
        && manifest_shape != inferred_shape
    {
        bail!(
            "fixture manifest chart shape does not match the selected bar path: manifest {:?}, inferred {:?}",
            manifest_shape,
            inferred_shape
        );
    }

    if let Some(symbol) = metadata.contract_symbols.first()
        && !symbol.eq_ignore_ascii_case(&settings.contract)
    {
        bail!(
            "fixture manifest contract `{symbol}` does not match proxy contract `{}`",
            settings.contract
        );
    }
    if metadata
        .contract_symbols
        .iter()
        .any(|symbol| !symbol.eq_ignore_ascii_case(&settings.contract))
    {
        bail!("fixture manifests disagree with the selected proxy contract");
    }
    if let Some(contract_id) = metadata.contract_ids.first()
        && metadata
            .contract_ids
            .iter()
            .any(|candidate| candidate != contract_id)
    {
        bail!("fixture manifests disagree on contract id");
    }
    if settings.contract_id_explicit
        && let Some(contract_id) = metadata.contract_ids.first()
        && *contract_id != settings.contract_id
    {
        bail!(
            "fixture manifest contract id {contract_id} does not match proxy --contract-id {}",
            settings.contract_id
        );
    }
    if let Some(tick_size) = metadata.tick_sizes.first()
        && metadata
            .tick_sizes
            .iter()
            .any(|candidate| !approximately_equal(*candidate, *tick_size))
    {
        bail!("fixture manifests disagree on tick size");
    }
    if let Some(value_per_point) = metadata.value_per_points.first()
        && metadata
            .value_per_points
            .iter()
            .any(|candidate| !approximately_equal(*candidate, *value_per_point))
    {
        bail!("fixture manifests disagree on value_per_point");
    }
    if let Some(tick_size) = metadata.tick_sizes.first()
        && !approximately_equal(*tick_size, settings.tick_size)
    {
        bail!(
            "fixture tick size {tick_size} does not match proxy --tick-size {}",
            settings.tick_size
        );
    }
    if let Some(value_per_point) = metadata.value_per_points.first()
        && !approximately_equal(*value_per_point, settings.value_per_point)
    {
        bail!(
            "fixture value_per_point {value_per_point} does not match proxy --value-per-point {}",
            settings.value_per_point
        );
    }

    let bar_content_verified = bar_binding
        .map(|(_, file)| verify_manifest_file(bar_path, file, "server_bars", inferred_bar_shape))
        .transpose()?
        .unwrap_or(false);
    let tick_content_verified = tick_path
        .zip(tick_binding)
        .map(|(path, (_, file))| verify_manifest_file(path, file, "raw_ticks", None))
        .transpose()?
        .unwrap_or(true);
    let complete_identity = !identities.is_empty()
        && identities.iter().all(|identity| {
            identity.contract_symbol.is_some()
                && identity.contract_id.is_some()
                && identity.tick_size.is_some()
                && identity.value_per_point.is_some()
        });
    let files_verified = bar_content_verified
        && tick_path.is_none_or(|_| tick_binding.is_some() && tick_content_verified);
    metadata.verified = complete_identity && files_verified;
    if !metadata.verified && !allow_unverified {
        bail!(
            "fixture identity could not be verified; provide --fixture-manifest or pass --allow-unverified-fixture"
        );
    }
    if !metadata.verified && !metadata.manifests.is_empty() {
        eprintln!(
            "warning: fixture manifest was found but did not contain complete contract identity; replay is unverified"
        );
    }
    if !metadata.verified
        && metadata.manifests.is_empty()
        && (tick_path.is_some() || !allow_unverified)
    {
        eprintln!(
            "warning: no fixture manifest found; replay identity is unverified (use --fixture-manifest for cached data)"
        );
    }
    Ok(metadata)
}

fn discover_manifest(path: &Path) -> Option<PathBuf> {
    let mut directory = path.parent();
    while let Some(current) = directory {
        let candidate = current.join("manifest.json");
        if candidate.is_file() {
            return Some(candidate);
        }
        directory = current.parent();
    }
    None
}

fn find_manifest_file<'a>(
    identities: &'a [ManifestIdentity],
    selected_path: &Path,
    expected_source_kind: &str,
) -> Result<Option<(&'a ManifestIdentity, &'a ManifestFile)>> {
    let selected_path = fs::canonicalize(selected_path)
        .with_context(|| format!("canonicalize selected fixture {}", selected_path.display()))?;
    let has_file_entries = identities.iter().any(|identity| !identity.files.is_empty());
    for identity in identities {
        let Some(manifest_root) = identity.path.parent() else {
            continue;
        };
        let manifest_root = fs::canonicalize(manifest_root).with_context(|| {
            format!(
                "canonicalize manifest directory {}",
                manifest_root.display()
            )
        })?;
        for file in &identity.files {
            let manifest_file_path = Path::new(&file.relative_path);
            let manifest_file_path = if manifest_file_path.is_absolute() {
                manifest_file_path.to_path_buf()
            } else {
                manifest_root.join(manifest_file_path)
            };
            let Ok(manifest_file_path) = fs::canonicalize(&manifest_file_path) else {
                continue;
            };
            if manifest_file_path != selected_path {
                continue;
            }
            if file.source_kind.as_deref() != Some(expected_source_kind) {
                bail!(
                    "fixture {} is listed as source kind {:?}, expected {expected_source_kind}",
                    selected_path.display(),
                    file.source_kind
                );
            }
            return Ok(Some((identity, file)));
        }
    }
    if has_file_entries {
        bail!(
            "selected {} fixture is not listed in the discovered manifest; refusing an unbound replay input",
            selected_path.display()
        );
    }
    Ok(None)
}

fn read_manifest_identity(path: &Path) -> Result<ManifestIdentity> {
    let file =
        File::open(path).with_context(|| format!("open fixture manifest {}", path.display()))?;
    let root = serde_json::from_reader::<_, Value>(BufReader::new(file))
        .with_context(|| format!("parse fixture manifest {}", path.display()))?;
    let contract_symbol = value_string(
        &root,
        &[
            &["contract", "symbol"],
            &["contract_metadata", "contract", "payload", "name"],
        ],
    );
    let contract_id = value_i64(
        &root,
        &[
            &["contract", "id"],
            &["contract_metadata", "contract", "payload", "id"],
        ],
    );
    let tick_size = value_f64(
        &root,
        &[
            &["tick_specs", "tick_size"],
            &[
                "contract_metadata",
                "contract",
                "payload",
                "providerTickSize",
            ],
        ],
    );
    let value_per_point = value_f64(&root, &[&["tick_specs", "value_per_point"]]);
    let files = root
        .get("files")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .map(|entry| {
                    let relative_path = entry
                        .get("relative_path")
                        .and_then(Value::as_str)
                        .context("fixture manifest file is missing relative_path")?
                        .to_string();
                    let first_ts_ns = manifest_file_timestamp(
                        entry
                            .get("first_timestamp")
                            .or_else(|| entry.get("first_ts_ns")),
                        "first_timestamp",
                        path,
                    )?;
                    let last_ts_ns = manifest_file_timestamp(
                        entry
                            .get("last_timestamp")
                            .or_else(|| entry.get("last_ts_ns")),
                        "last_timestamp",
                        path,
                    )?;
                    let row_count = entry
                        .get("row_count")
                        .map(|value| parse_manifest_usize(value, "row_count", path))
                        .transpose()?;
                    let (data_hash_algorithm, data_hash_value) = entry
                        .get("data_hash")
                        .map(|value| {
                            let object = value
                                .as_object()
                                .context("fixture manifest data_hash must be an object")?;
                            let algorithm = object
                                .get("algorithm")
                                .and_then(Value::as_str)
                                .context("fixture manifest data_hash is missing algorithm")?
                                .to_string();
                            let value = object
                                .get("value")
                                .and_then(Value::as_str)
                                .context("fixture manifest data_hash is missing value")?
                                .to_string();
                            Ok::<_, anyhow::Error>((Some(algorithm), Some(value)))
                        })
                        .transpose()?
                        .unwrap_or((None, None));
                    let bar_shape = manifest_bar_shape(entry, path)?;
                    Ok(ManifestFile {
                        relative_path,
                        source_kind: entry
                            .get("source_kind")
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                        first_ts_ns,
                        last_ts_ns,
                        row_count,
                        data_hash_algorithm,
                        data_hash_value,
                        bar_shape,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(ManifestIdentity {
        path: path.to_path_buf(),
        contract_symbol,
        contract_id,
        tick_size,
        value_per_point,
        files,
    })
}

fn manifest_file_timestamp(
    value: Option<&Value>,
    field: &str,
    manifest_path: &Path,
) -> Result<Option<i64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if let Some(timestamp) = value.as_str() {
        return parse_timestamp(timestamp)
            .and_then(|timestamp| {
                timestamp
                    .timestamp_nanos_opt()
                    .context("manifest timestamp is outside nanosecond range")
            })
            .map(Some)
            .with_context(|| format!("parse {field} in manifest {}", manifest_path.display()));
    }
    if let Some(timestamp) = value.as_i64() {
        return Ok(Some(timestamp));
    }
    if let Some(timestamp) = value.as_f64()
        && timestamp.is_finite()
        && timestamp.fract() == 0.0
        && timestamp >= i64::MIN as f64
        && timestamp <= i64::MAX as f64
    {
        return Ok(Some(timestamp as i64));
    }
    bail!(
        "manifest {} has an invalid {field} value",
        manifest_path.display()
    )
}

fn parse_manifest_usize(value: &Value, field: &str, manifest_path: &Path) -> Result<usize> {
    let number = value
        .as_u64()
        .or_else(|| {
            value.as_f64().and_then(|number| {
                (number.is_finite() && number.fract() == 0.0 && number >= 0.0)
                    .then_some(number as u64)
            })
    })
    .context("manifest row count must be a non-negative integer")?;
    usize::try_from(number).with_context(|| {
        format!(
            "manifest {} {field} does not fit in this platform's usize",
            manifest_path.display(),
        )
    })
}

fn manifest_bar_shape(entry: &Value, manifest_path: &Path) -> Result<Option<ChartShape>> {
    let Some(bar_type) = entry
        .get("market_shape")
        .and_then(Value::as_object)
        .and_then(|shape| shape.get("bar_type"))
    else {
        return Ok(None);
    };
    let kind = bar_type
        .get("kind")
        .and_then(Value::as_str)
        .context("fixture manifest bar_type is missing kind")?;
    let value = bar_type
        .get("value")
        .map(|value| parse_manifest_usize(value, "market_shape.bar_type.value", manifest_path))
        .transpose()?
        .and_then(|value| u32::try_from(value).ok())
        .context("fixture manifest bar_type value is outside u32 range")?;
    ChartShape::from_bar_kind(kind, value)
        .map(Some)
        .with_context(|| format!("fixture manifest has unsupported bar kind `{kind}`"))
}

fn value_string(root: &Value, paths: &[&[&str]]) -> Option<String> {
    paths
        .iter()
        .find_map(|path| root.pointer(&format!("/{}", path.join("/")))?.as_str())
        .map(ToString::to_string)
}

fn value_i64(root: &Value, paths: &[&[&str]]) -> Option<i64> {
    paths.iter().find_map(|path| {
        let value = root.pointer(&format!("/{}", path.join("/")))?;
        value.as_i64().or_else(|| {
            let number = value.as_f64()?;
            (number.is_finite() && number.fract() == 0.0).then_some(number as i64)
        })
    })
}

fn value_f64(root: &Value, paths: &[&[&str]]) -> Option<f64> {
    paths
        .iter()
        .find_map(|path| root.pointer(&format!("/{}", path.join("/")))?.as_f64())
}

fn verify_manifest_file(
    path: &Path,
    file: &ManifestFile,
    expected_source_kind: &str,
    inferred_bar_shape: Option<&ChartShape>,
) -> Result<bool> {
    let (Some(row_count), Some(algorithm), Some(expected_hash), Some(first_ts_ns), Some(last_ts_ns)) = (
        file.row_count,
        file.data_hash_algorithm.as_deref(),
        file.data_hash_value.as_deref(),
        file.first_ts_ns,
        file.last_ts_ns,
    ) else {
        return Ok(false);
    };
    if first_ts_ns > last_ts_ns {
        bail!(
            "manifest window for {} is inverted: {} > {}",
            path.display(),
            first_ts_ns,
            last_ts_ns
        );
    }
    let actual_row_count = count_fixture_rows(path)?;
    if actual_row_count != row_count {
        bail!(
            "fixture row-count mismatch for {}: manifest={} actual={actual_row_count}",
            path.display(),
            row_count
        );
    }
    let actual_hash = match algorithm {
        "fnv1a64" => fnv1a64_file_hex(path)?,
        other => {
            eprintln!(
                "warning: cannot verify {} fixture {} with unsupported manifest hash algorithm {other}",
                expected_source_kind,
                path.display()
            );
            return Ok(false);
        }
    };
    if actual_hash != expected_hash {
        bail!(
            "fixture data hash mismatch for {}: manifest={} actual={actual_hash}",
            path.display(),
            expected_hash
        );
    }
    if expected_source_kind == "server_bars" {
        let manifest_shape = file.bar_shape.as_ref().context(
            "server-bars manifest entry has no market_shape.bar_type; refusing verified chart replay",
        )?;
        if let Some(inferred_shape) = inferred_bar_shape
            && inferred_shape != manifest_shape
        {
            bail!(
                "server-bars fixture {} shape differs from its manifest: path={:?} manifest={:?}",
                path.display(),
                inferred_shape,
                manifest_shape
            );
        }
    }
    Ok(true)
}

fn count_fixture_rows(path: &Path) -> Result<usize> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "parquet" => {
            let file = File::open(path)
                .with_context(|| format!("open fixture for row count {}", path.display()))?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .with_context(|| format!("read fixture metadata {}", path.display()))?;
            usize::try_from(reader.metadata().file_metadata().num_rows())
                .with_context(|| format!("fixture row count does not fit usize {}", path.display()))
        }
        "csv" => {
            let file = File::open(path)
                .with_context(|| format!("open CSV fixture for row count {}", path.display()))?;
            let mut reader = csv::ReaderBuilder::new().flexible(false).from_reader(file);
            let mut rows = 0_usize;
            for record in reader.records() {
                record.with_context(|| format!("read CSV fixture row count {}", path.display()))?;
                rows = rows.saturating_add(1);
            }
            Ok(rows)
        }
        _ => {
            let file = File::open(path)
                .with_context(|| format!("open JSONL fixture for row count {}", path.display()))?;
            let mut rows = 0_usize;
            for line in BufReader::new(file).lines() {
                if !line
                    .with_context(|| format!("read JSONL fixture row count {}", path.display()))?
                    .trim()
                    .is_empty()
                {
                    rows = rows.saturating_add(1);
                }
            }
            Ok(rows)
        }
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= scale * 1e-9
}

#[derive(Debug, Deserialize)]
struct JsonBar {
    #[serde(default)]
    timestamp: Option<String>,
    #[serde(default)]
    ts_ns: Option<i64>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    #[serde(default)]
    volume: Option<f64>,
}

#[cfg(test)]
fn read_jsonl(bytes: &[u8]) -> Result<Vec<Bar>> {
    read_jsonl_reader(BufReader::new(bytes), 0)
}

fn read_jsonl_file(path: &Path, max_bars: usize) -> Result<Vec<Bar>> {
    let file =
        File::open(path).with_context(|| format!("open JSONL fixture {}", path.display()))?;
    read_jsonl_reader(BufReader::new(file), max_bars)
}

fn read_jsonl_reader<R: BufRead>(reader: R, max_bars: usize) -> Result<Vec<Bar>> {
    let mut bars = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        if max_bars > 0 && bars.len() >= max_bars {
            break;
        }
        let line = line.with_context(|| format!("read JSONL fixture line {}", line_index + 1))?;
        if line.len() > MAX_FIXTURE_RECORD_BYTES {
            bail!(
                "JSONL fixture line {} exceeds {} bytes",
                line_index + 1,
                MAX_FIXTURE_RECORD_BYTES
            );
        }
        if line.trim().is_empty() {
            continue;
        }
        let row: JsonBar = serde_json::from_str(line.trim())
            .with_context(|| format!("parse JSONL fixture line {}", line_index + 1))?;
        bars.push(
            json_bar_to_bar(row)
                .with_context(|| format!("decode JSONL fixture line {}", line_index + 1))?,
        );
    }
    Ok(bars)
}

fn json_bar_to_bar(row: JsonBar) -> Result<Bar> {
    let timestamp = row
        .timestamp
        .as_deref()
        .map(parse_timestamp)
        .transpose()?
        .or_else(|| row.ts_ns.map(DateTime::<Utc>::from_timestamp_nanos))
        .context("bar is missing timestamp or ts_ns")?;
    let ts_ns = row
        .ts_ns
        .or_else(|| timestamp.timestamp_nanos_opt())
        .context("bar timestamp outside nanosecond range")?;
    Ok(Bar {
        timestamp,
        ts_ns,
        open: row.open,
        high: row.high,
        low: row.low,
        close: row.close,
        volume: row.volume,
    })
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .or_else(|_| {
            DateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f%:z")
                .map(|value| value.with_timezone(&Utc))
        })
        .with_context(|| format!("parse timestamp {value}"))
}

fn read_csv_file(path: &Path, max_bars: usize) -> Result<Vec<Bar>> {
    let file = File::open(path).with_context(|| format!("open CSV fixture {}", path.display()))?;
    read_csv_reader(BufReader::new(file), max_bars)
}

fn read_csv_reader<R: Read>(reader: R, max_bars: usize) -> Result<Vec<Bar>> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .flexible(false)
        .from_reader(reader);
    let header = csv_reader
        .headers()
        .context("read CSV fixture header")?
        .clone();
    let columns = header
        .iter()
        .map(|value| value.trim_start_matches('\u{feff}').to_ascii_lowercase())
        .collect::<Vec<_>>();
    let index = |name: &str| {
        columns
            .iter()
            .position(|column| column == name)
            .with_context(|| format!("CSV fixture missing {name} column"))
    };
    let timestamp_index = columns
        .iter()
        .position(|column| column == "timestamp" || column == "time" || column == "ts");
    let ts_ns_index = columns
        .iter()
        .position(|column| column == "ts_ns" || column == "timestamp_ns");
    let open_index = index("open")?;
    let high_index = index("high")?;
    let low_index = index("low")?;
    let close_index = index("close")?;
    let volume_index = columns.iter().position(|column| column == "volume");
    let mut bars = Vec::new();
    for (row_index, row_result) in csv_reader.records().enumerate() {
        if max_bars > 0 && bars.len() >= max_bars {
            break;
        }
        let row = row_result.with_context(|| format!("read CSV fixture row {}", row_index + 2))?;
        if row.as_slice().trim().is_empty() {
            continue;
        }
        let field = |at: usize| {
            row.get(at)
                .filter(|value| !value.is_empty())
                .context("CSV row has a missing field")
        };
        let timestamp = timestamp_index
            .map(|at| field(at).and_then(parse_timestamp))
            .transpose()?;
        let ts_ns = ts_ns_index
            .map(|at| field(at).and_then(|value| value.parse::<i64>().context("parse ts_ns")))
            .transpose()?;
        let timestamp = timestamp
            .or_else(|| ts_ns.map(DateTime::<Utc>::from_timestamp_nanos))
            .context("CSV row missing timestamp/ts_ns")?;
        let ts_ns = ts_ns
            .or_else(|| timestamp.timestamp_nanos_opt())
            .context("CSV timestamp outside nanosecond range")?;
        let parse_number =
            |at: usize| -> Result<f64> { field(at)?.parse::<f64>().context("parse CSV number") };
        let volume = volume_index
            .map(|at| {
                Ok::<_, anyhow::Error>(match row.get(at).map(str::trim) {
                    Some("") | None => None,
                    Some(value) => Some(value.parse::<f64>().context("parse CSV volume")?),
                })
            })
            .transpose()?
            .flatten();
        bars.push(Bar {
            timestamp,
            ts_ns,
            open: parse_number(open_index)?,
            high: parse_number(high_index)?,
            low: parse_number(low_index)?,
            close: parse_number(close_index)?,
            volume,
        });
    }
    Ok(bars)
}

fn read_parquet(path: &Path, max_bars: usize) -> Result<Vec<Bar>> {
    let file =
        File::open(path).with_context(|| format!("open parquet fixture {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?
        .with_batch_size(DEFAULT_PARQUET_BATCH_ROWS)
        .build()
        .with_context(|| format!("build parquet reader {}", path.display()))?;
    let mut bars = Vec::new();
    for batch in reader {
        let batch = batch.with_context(|| format!("read parquet batch {}", path.display()))?;
        append_parquet_batch(&mut bars, &batch, path, max_bars)?;
        if max_bars > 0 && bars.len() >= max_bars {
            break;
        }
    }
    Ok(bars)
}

fn append_parquet_batch(
    bars: &mut Vec<Bar>,
    batch: &RecordBatch,
    path: &Path,
    max_bars: usize,
) -> Result<()> {
    let schema = batch.schema();
    let column = |name: &str| -> Result<&Arc<dyn Array>> {
        let index = schema
            .index_of(name)
            .with_context(|| format!("parquet fixture missing {name} column"))?;
        Ok(batch.column(index))
    };
    let timestamps = column("timestamp")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("parquet timestamp is not UTF-8")?;
    let ts_ns = column("ts_ns")?
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("parquet ts_ns is not int64")?;
    let opens = column("open")?
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("parquet open is not float64")?;
    let highs = column("high")?
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("parquet high is not float64")?;
    let lows = column("low")?
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("parquet low is not float64")?;
    let closes = column("close")?
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("parquet close is not float64")?;
    let volume_column = column("volume").ok();
    let volumes = volume_column
        .as_ref()
        .map(|column| {
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("parquet volume is not float64")
        })
        .transpose()?;
    for index in 0..batch.num_rows() {
        if max_bars > 0 && bars.len() >= max_bars {
            break;
        }
        for (name, array) in [
            ("timestamp", timestamps as &dyn Array),
            ("ts_ns", ts_ns as &dyn Array),
            ("open", opens as &dyn Array),
            ("high", highs as &dyn Array),
            ("low", lows as &dyn Array),
            ("close", closes as &dyn Array),
        ] {
            if array.is_null(index) {
                bail!("parquet {name} is null at row {index}");
            }
        }
        let timestamp = parse_timestamp(timestamps.value(index)).with_context(|| {
            format!("parse parquet timestamp row {index} in {}", path.display())
        })?;
        bars.push(Bar {
            timestamp,
            ts_ns: ts_ns.value(index),
            open: opens.value(index),
            high: highs.value(index),
            low: lows.value(index),
            close: closes.value(index),
            volume: volumes
                .and_then(|volumes| (!volumes.is_null(index)).then(|| volumes.value(index))),
        });
    }
    Ok(())
}

fn validate_bar(bar: &Bar) -> Result<()> {
    if bar.ts_ns <= 0 {
        bail!("bar timestamp must be positive");
    }
    if !(bar.open.is_finite()
        && bar.high.is_finite()
        && bar.low.is_finite()
        && bar.close.is_finite()
        && bar.volume.is_none_or(f64::is_finite))
    {
        bail!("bar contains a non-finite OHLCV value");
    }
    if bar.volume.is_some_and(|volume| volume < 0.0) {
        bail!("bar volume must not be negative");
    }
    if let Some(timestamp_ns) = bar.timestamp.timestamp_nanos_opt() {
        let drift = i128::from(timestamp_ns) - i128::from(bar.ts_ns);
        if drift.abs() > MAX_TIMESTAMP_DRIFT_NS {
            bail!(
                "bar timestamp and ts_ns differ by {} ns",
                drift.unsigned_abs()
            );
        }
    }
    if bar.high < bar.low
        || bar.open < bar.low
        || bar.open > bar.high
        || bar.close < bar.low
        || bar.close > bar.high
    {
        bail!("bar OHLC values fall outside high/low range");
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let file =
        File::open(path).with_context(|| format!("read replay fixture {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash replay fixture {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn fnv1a64_file_hex(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("read fixture hash {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hash = 0xcbf29ce484222325_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash fixture {}", path.display()))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3_u64);
        }
    }
    Ok(format!("{hash:016x}"))
}

fn wall_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn settings() -> Settings {
        Settings {
            rest_bind: "127.0.0.1:18100".parse().unwrap(),
            user_ws_bind: "127.0.0.1:18101".parse().unwrap(),
            market_ws_bind: "127.0.0.1:18102".parse().unwrap(),
            history_bars: 1,
            max_bars: 0,
            start_file: None,
            speed: 0.0,
            max_sleep_ms: 1,
            loop_boundary_delay_ms: 1,
            rest_delay_ms: 0,
            ack_delay_ms: 0,
            fill_delay_ms: 0,
            loop_replay: false,
            account_id: 1,
            account_name: "PROXY".to_string(),
            contract: "GCZ6".to_string(),
            contract_id: 2,
            contract_id_explicit: true,
            tick_size: 0.1,
            value_per_point: 100.0,
            initial_balance: 100_000.0,
            max_clients: 1,
            max_state_entities: 100,
            trace_max_bytes: 64 * 1024,
            fixture_path: PathBuf::from("fixture.jsonl"),
            tick_fixture_path: None,
            fixture_manifest: None,
            allow_unverified_fixture: true,
            require_quote_ticks: false,
            max_ticks: 0,
            raw_tick_bar_timestamps: RawTickBarTimestampMode::Start,
            protection_precedence: ExitPrecedence::StopFirst,
        }
    }

    fn bar(ts: i64, close: f64) -> Bar {
        Bar {
            timestamp: Utc.timestamp_nanos(ts),
            ts_ns: ts,
            open: close,
            high: close,
            low: close,
            close,
            volume: None,
        }
    }

    fn raw_tick(ts: i64, price: f64, bid_price: Option<f64>, ask_price: Option<f64>) -> RawTick {
        RawTick {
            timestamp: Utc.timestamp_nanos(ts),
            ts_ns: ts,
            tick_id: Some(ts),
            price,
            size: 1.0,
            bid_price,
            bid_size: bid_price.map(|_| 1.0),
            ask_price,
            ask_size: ask_price.map(|_| 1.0),
            chart_id: None,
            trade_date: None,
            packet_source: None,
            packet_base_ts_ms: None,
            packet_base_price_ticks: None,
        }
    }

    #[test]
    fn parser_keeps_wire_request_body() {
        let request = parse_request("order/placeorder\n7\n\n{\"accountId\":1}").unwrap();
        assert_eq!(request.endpoint, "order/placeorder");
        assert_eq!(request.request_id, 7);
        assert_eq!(request.body.unwrap()["accountId"], 1);
    }

    #[test]
    fn bar_shape_is_inferred_from_replay_cache_filename() {
        assert_eq!(
            infer_bar_shape(Path::new("session_10range_v1.parquet")),
            Some(ChartShape {
                underlying_type: "Tick",
                element_size: 10,
                element_size_unit: "Range",
            })
        );
        assert_eq!(
            infer_bar_shape(Path::new("session_1minute_v1.parquet")),
            Some(ChartShape {
                underlying_type: "MinuteBar",
                element_size: 1,
                element_size_unit: "UnderlyingUnits",
            })
        );
        assert_eq!(infer_bar_shape(Path::new("session.parquet")), None);
    }

    #[test]
    fn raw_tick_coverage_rejects_unseeded_or_early_tapes() {
        let bars = vec![bar(10, 100.0), bar(20, 100.0), bar(30, 100.0)];
        let complete = vec![
            raw_tick(10, 100.0, Some(99.9), Some(100.1)),
            raw_tick(20, 100.0, Some(99.9), Some(100.1)),
            raw_tick(30, 100.0, Some(99.9), Some(100.1)),
        ];
        let seed = complete.partition_point(|tick| tick.ts_ns < 20);
        let end = complete.partition_point(|tick| tick.ts_ns <= 30);
        assert!(validate_tick_coverage(
            &complete,
            &bars,
            1,
            seed,
            end,
            Path::new("complete.parquet"),
        )
        .is_ok());

        let no_seed = vec![
            raw_tick(20, 100.0, Some(99.9), Some(100.1)),
            raw_tick(30, 100.0, Some(99.9), Some(100.1)),
        ];
        let seed = no_seed.partition_point(|tick| tick.ts_ns < 20);
        let end = no_seed.partition_point(|tick| tick.ts_ns <= 30);
        assert!(validate_tick_coverage(
            &no_seed,
            &bars,
            1,
            seed,
            end,
            Path::new("no-seed.parquet"),
        )
        .is_err());

        let early_end = vec![
            raw_tick(10, 100.0, Some(99.9), Some(100.1)),
            raw_tick(20, 100.0, Some(99.9), Some(100.1)),
            raw_tick(25, 100.0, Some(99.9), Some(100.1)),
        ];
        let seed = early_end.partition_point(|tick| tick.ts_ns < 20);
        let end = early_end.partition_point(|tick| tick.ts_ns <= 30);
        assert!(validate_tick_coverage(
            &early_end,
            &bars,
            1,
            seed,
            end,
            Path::new("early-end.parquet"),
        )
        .is_err());
    }

    #[test]
    fn strategy_quantity_requires_a_positive_integer_entry_quantity() {
        assert_eq!(
            parse_strategy_quantity(&json!({"entryVersion": {"orderQty": 2}})).unwrap(),
            2
        );
        assert!(parse_strategy_quantity(&json!({})).is_err());
        assert!(parse_strategy_quantity(&json!({"entryVersion": {"orderQty": 1.5}})).is_err());
        assert!(parse_strategy_quantity(&json!({"entryVersion": {"orderQty": 0}})).is_err());
    }

    #[test]
    fn bar_loader_preserves_input_order_for_corrections() {
        let raw = br#"{"timestamp":"1970-01-01T00:00:00.000000002Z","ts_ns":2,"open":2,"high":2,"low":2,"close":2}
{"timestamp":"1970-01-01T00:00:00.000000001Z","ts_ns":1,"open":1,"high":1,"low":1,"close":1}
"#;
        let bars = read_jsonl(raw).unwrap();
        assert_eq!(
            bars.iter().map(|bar| bar.ts_ns).collect::<Vec<_>>(),
            vec![2, 1]
        );
    }

    #[test]
    fn fill_math_handles_reversal_and_realized_pnl() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: None,
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        broker.fill_order(&settings, order_id);
        broker.set_market(&bar(2_000_000_000, 12.0));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Sell",
                quantity: 2,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: None,
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        broker.fill_order(&settings, order_id);
        assert_eq!(broker.position_qty, -1);
        assert_eq!(broker.realized_pnl, 200.0);
        assert_eq!(broker.average_price, Some(12.0));
    }

    #[test]
    fn snapshot_has_authoritative_flat_position() {
        let settings = settings();
        let broker = BrokerState::new(&settings);
        let snapshot = broker.snapshot(&settings);
        assert_eq!(snapshot["positions"][0]["netPos"], 0);
        assert_eq!(snapshot["accounts"][0]["id"], 1);
    }

    #[test]
    fn response_frame_is_tradovate_array_frame() {
        let frame = response_frame(4, 200, json!({"ok": true}));
        assert!(frame.starts_with("a["));
        assert_eq!(
            serde_json::from_str::<Value>(&frame[1..]).unwrap()[0]["i"],
            4
        );
    }

    #[test]
    fn fills_use_the_quote_available_when_the_order_was_accepted() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: None,
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        broker.set_market(&bar(2_000_000_000, 20.0));
        let events = broker.fill_order(&settings, order_id);
        let fill_price = events
            .iter()
            .find(|event| event["d"]["entityType"] == "fill")
            .and_then(|event| event["d"]["entity"]["price"].as_f64());
        assert_eq!(fill_price, Some(10.0));
    }

    #[test]
    fn fills_emit_native_order_execution_and_replay_identity() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("strategy-entry".to_string()),
                strategy_id: Some(77),
                bracket: None,
            })
            .unwrap();
        let events = broker.fill_order(&settings, order_id);
        let execution = events
            .iter()
            .find(|event| event["d"]["entityType"] == "executionReport")
            .expect("execution report event")["d"]["entity"]
            .clone();
        let fill = events
            .iter()
            .find(|event| event["d"]["entityType"] == "fill")
            .expect("fill event")["d"]["entity"]
            .clone();
        assert_eq!(execution["orderId"], order_id);
        assert_eq!(execution["orderStrategyId"], 77);
        assert_eq!(execution["clOrdId"], "strategy-entry");
        assert_eq!(fill["orderId"], order_id);
        assert_eq!(fill["orderStrategyId"], 77);
        assert_eq!(fill["clOrdId"], "strategy-entry");
        assert_eq!(fill["source"], "replay");
        assert_eq!(fill["replayFillSource"], "proxy");
        assert_eq!(broker.execution_reports.len(), 1);
    }

    #[test]
    fn canceling_a_parent_marks_its_strategy_interrupted() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        broker
            .strategies
            .insert(77, json!({"id": 77, "status": "Active", "ordStatus": "Working"}));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("cancel-parent".to_string()),
                strategy_id: Some(77),
                bracket: None,
            })
            .unwrap();
        assert!(broker.cancel_order(order_id).is_some());
        let mut events = Vec::new();
        broker.interrupt_strategy_after_cancel(77, &mut events);
        assert_eq!(broker.strategies[&77]["status"], "Interrupted");
        assert_eq!(broker.strategies[&77]["exitReason"], "parent_cancelled");
        assert!(events.iter().any(|event| {
            event["d"]["entityType"] == "orderStrategy"
                && event["d"]["entity"]["status"] == "Interrupted"
        }));
    }

    #[test]
    fn raw_ticks_fill_entries_on_the_next_quote_and_broker_owned_tp_on_bid() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let bracket = json!([{
            "qty": 1,
            "profitTarget": 1.0,
            "stopLoss": -0.5
        }]);
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("raw-entry".to_string()),
                strategy_id: Some(77),
                bracket: Some(bracket),
            })
            .unwrap();

        let entry_events = broker.process_raw_tick(
            &settings,
            // The bid is already below the eventual stop while the ask fills
            // the entry. The new bracket must not evaluate this same tick.
            &raw_tick(2_000_000_000, 100.2, Some(99.0), Some(100.2)),
            2,
        );
        let entry_fill = entry_events
            .iter()
            .find(|event| event["d"]["entityType"] == "fill")
            .expect("entry fill")["d"]["entity"]
            .clone();
        assert_eq!(entry_fill["orderId"], order_id);
        assert_eq!(entry_fill["price"], 100.2);
        assert_eq!(entry_fill["replayFillSource"], "tick_bid_ask");
        assert!(
            entry_events
                .iter()
                .filter(|event| event["d"]["entityType"] == "fill")
                .all(|event| event["d"]["entity"]["replayExitReason"].is_null())
        );
        let children = broker
            .orders
            .values()
            .filter(|order| order["orderStrategyId"] == 77 && order["id"] != order_id)
            .collect::<Vec<_>>();
        assert_eq!(children.len(), 2);
        assert!(children.iter().all(|order| {
            order["ordStatus"] == "Working" && order["replayBrokerOwned"] == true
        }));

        let exit_events = broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 101.2, Some(101.2), Some(101.3)),
            3,
        );
        let exit_fill = exit_events
            .iter()
            .find(|event| {
                event["d"]["entityType"] == "fill"
                    && event["d"]["entity"]["replayExitReason"] == "take_profit"
            })
            .expect("take-profit fill")["d"]["entity"]
            .clone();
        assert_eq!(exit_fill["price"], 101.2);
        assert_eq!(broker.position_qty, 0);
        assert!(broker.protections.is_empty());
        let stop = broker
            .orders
            .values()
            .find(|order| order["protectionLeg"] == "sl")
            .expect("sibling stop");
        assert_eq!(stop["ordStatus"], "Cancelled");
    }

    #[test]
    fn partial_quote_fill_is_marked_ambiguous() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("partial-quote".to_string()),
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        let events = broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, None, Some(100.2)),
            2,
        );
        let fill = events
            .iter()
            .find(|event| event["d"]["entityType"] == "fill")
            .expect("partial quote fill")["d"]["entity"]
            .clone();
        assert_eq!(fill["orderId"], order_id);
        assert_eq!(fill["replayFillSource"], "tick_partial_quote");
        assert_eq!(fill["replayExecutionPrecision"], "tick_partial_quote");
        assert_eq!(fill["replayAmbiguousTick"], true);
    }

    #[test]
    fn raw_tick_trailing_stop_amends_child_then_exits_at_bid() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let bracket = json!([{
            "qty": 1,
            "stopLoss": -5.0,
            "autoTrail": {"trigger": 1.0, "stopLoss": 0.5, "freq": 0.1}
        }]);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("trail-entry".to_string()),
                strategy_id: Some(88),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, Some(100.1), Some(100.2)),
            2,
        );
        let tightened = broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 101.2, Some(101.2), Some(101.3)),
            3,
        );
        let stop_id = broker.protections[&88].stop_order_id.expect("stop child");
        assert_eq!(broker.orders[&stop_id]["stopPrice"], json!(100.7));
        assert!(tightened.iter().any(|event| {
            event["d"]["entityType"] == "order"
                && event["d"]["entity"]["id"] == stop_id
                && event["d"]["entity"]["replayTrailingActive"] == true
        }));

        let exit_events = broker.process_raw_tick(
            &settings,
            &raw_tick(4_000_000_000, 100.7, Some(100.7), Some(100.8)),
            4,
        );
        assert!(exit_events.iter().any(|event| {
            event["d"]["entityType"] == "fill"
                && event["d"]["entity"]["replayExitReason"] == "trailing_stop"
        }));
        assert_eq!(broker.position_qty, 0);
    }

    #[test]
    fn canceled_protection_is_torn_down_before_a_later_quote() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let bracket = json!([{"profitTarget": 1.0, "stopLoss": -0.5}]);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("cancel-entry".to_string()),
                strategy_id: Some(89),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, Some(100.1), Some(100.2)),
            2,
        );
        let stop_id = broker.protections[&89].stop_order_id.expect("stop child");
        let mut cancel_events = Vec::new();
        let strategy_id = broker
            .protection_strategy_for_order(stop_id)
            .expect("protection owner");
        assert!(broker.cancel_order(stop_id).is_some());
        broker.teardown_protection_after_cancel(strategy_id, stop_id, &mut cancel_events);
        let tp = broker
            .orders
            .values()
            .find(|order| order["protectionLeg"] == "tp")
            .expect("take-profit sibling");
        assert_eq!(tp["ordStatus"], "Cancelled");
        assert!(broker.protections.is_empty());

        let events = broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 99.0, Some(99.0), Some(99.1)),
            3,
        );
        assert_eq!(broker.position_qty, 1);
        assert!(!events.iter().any(|event| {
            event["d"]["entityType"] == "fill"
                && event["d"]["entity"]["replayExitReason"].is_string()
        }));
    }

    #[test]
    fn bar_fill_tears_down_stale_broker_protection_after_flatten() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 100.0));
        let bracket = json!([{"profitTarget": 1.0, "stopLoss": -0.5}]);
        let (_, entry_order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("bar-entry".to_string()),
                strategy_id: Some(94),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.fill_order(&settings, entry_order_id);
        let stop_id = broker.protections[&94].stop_order_id.expect("stop child");
        let (_, flatten_order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Sell",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("bar-flatten".to_string()),
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        broker.fill_order(&settings, flatten_order_id);

        assert_eq!(broker.position_qty, 0);
        assert!(broker.protections.is_empty());
        assert_eq!(broker.orders[&stop_id]["ordStatus"], "Cancelled");
    }

    #[test]
    fn position_generation_invalidates_a_same_price_old_bracket() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let bracket = json!([{"profitTarget": 1.0, "stopLoss": -0.5}]);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("generation-entry".to_string()),
                strategy_id: Some(90),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, Some(100.1), Some(100.2)),
            2,
        );
        let old_stop_id = broker.protections[&90]
            .stop_order_id
            .expect("old stop child");
        broker.set_market_tick(&raw_tick(3_000_000_000, 100.2, Some(100.1), Some(100.2)), 3);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("same-price-add".to_string()),
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(4_000_000_000, 100.2, Some(100.1), Some(100.2)),
            4,
        );
        assert_eq!(broker.position_qty, 2);
        assert!(!broker.protections.contains_key(&90));
        assert_eq!(broker.orders[&old_stop_id]["ordStatus"], "Cancelled");
    }

    #[test]
    fn same_side_add_arms_new_protection_from_weighted_average() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let bracket = json!([{"profitTarget": 1.0, "stopLoss": -0.5}]);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("weighted-first".to_string()),
                strategy_id: Some(91),
                bracket: Some(bracket.clone()),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, Some(100.1), Some(100.2)),
            2,
        );
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("weighted-second".to_string()),
                strategy_id: Some(92),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 102.0, Some(101.9), Some(102.0)),
            3,
        );
        assert_eq!(broker.position_qty, 2);
        assert_eq!(broker.average_price, Some(101.1));
        assert!(broker.protections.contains_key(&92));
        assert_eq!(broker.protections[&92].bracket.position().entry_price_ticks, 1011);
        assert!(!broker.protections.contains_key(&91));
    }

    #[test]
    fn state_cap_cannot_prune_a_protection_child_before_oco_resolution() {
        let mut settings = settings();
        settings.max_state_entities = 6;
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(
            &raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)),
            1,
        );
        let bracket = json!([{"profitTarget": 1.0, "stopLoss": -0.5}]);
        let (_, _, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("state-cap-entry".to_string()),
                strategy_id: Some(93),
                bracket: Some(bracket),
            })
            .unwrap();
        broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.2, Some(100.1), Some(100.2)),
            2,
        );
        let exit_events = broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 101.2, Some(101.2), Some(101.3)),
            3,
        );

        assert_eq!(broker.position_qty, 0);
        assert!(broker.protections.is_empty());
        assert!(exit_events.iter().any(|event| {
            event["d"]["entityType"] == "fill"
                && event["d"]["entity"]["replayExitReason"] == "take_profit"
        }));
        let stop = broker
            .orders
            .values()
            .find(|order| order["protectionLeg"] == "sl");
        assert!(stop.is_none() || stop.is_some_and(|order| order["ordStatus"] == "Cancelled"));
    }

    #[test]
    fn crossed_quote_cannot_fill_a_pending_entry() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market_tick(&raw_tick(1_000_000_000, 100.0, Some(99.9), Some(100.1)), 1);
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("crossed-entry".to_string()),
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        let crossed = broker.process_raw_tick(
            &settings,
            &raw_tick(2_000_000_000, 100.0, Some(100.2), Some(100.1)),
            2,
        );
        assert_eq!(broker.position_qty, 0);
        assert_eq!(broker.last_bid, Some(99.9));
        assert_eq!(broker.last_ask, Some(100.1));
        assert_eq!(broker.last_ts_ns, Some(1_000_000_000));
        assert_eq!(broker.market_sequence, 1);
        assert!(broker.pending_fill_quotes.contains_key(&order_id));
        assert!(crossed.is_empty());
        broker.process_raw_tick(
            &settings,
            &raw_tick(3_000_000_000, 100.2, Some(100.1), Some(100.2)),
            3,
        );
        assert_eq!(broker.position_qty, 1);
    }

    #[test]
    fn canceled_orders_cannot_fill_after_they_are_queued() {
        let settings = settings();
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        let (_, order_id, _, _) = broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: None,
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        assert!(broker.cancel_order(order_id).is_some());
        assert!(broker.fill_order(&settings, order_id).is_empty());
    }

    #[test]
    fn synthetic_state_limit_rejects_new_orders_when_full() {
        let mut settings = settings();
        settings.max_state_entities = 1;
        let mut broker = BrokerState::new(&settings);
        broker.set_market(&bar(1_000_000_000, 10.0));
        broker
            .accept_order(OrderRequest {
                settings: &settings,
                action: "Buy",
                quantity: 1,
                account_id: 1,
                contract_id: 2,
                symbol: "GCZ6",
                cl_ord_id: Some("first".to_string()),
                strategy_id: None,
                bracket: None,
            })
            .unwrap();
        assert!(
            broker
                .accept_order(OrderRequest {
                    settings: &settings,
                    action: "Buy",
                    quantity: 1,
                    account_id: 1,
                    contract_id: 2,
                    symbol: "GCZ6",
                    cl_ord_id: Some("second".to_string()),
                    strategy_id: None,
                    bracket: None,
                })
                .is_err()
        );
    }

    #[test]
    fn fractional_replay_speeds_are_safe_and_precise() {
        assert_eq!(replay_delay_ms(0, 1_000_000_000, 0.5, 10_000), 2_000);
        assert_eq!(replay_delay_ms(0, 1_000_000_000, 1.5, 10_000), 667);
        assert_eq!(replay_delay_ms(0, 1_000_000_000, 0.0, 10_000), 0);
    }

    #[test]
    fn csv_reader_handles_bom_quotes_and_optional_volume() {
        let raw = b"\xEF\xBB\xBFtimestamp,open,high,low,close,volume\n\"2026-08-28T12:00:00Z\",1,2,0,1.5,\n";
        let bars = read_csv_reader(&raw[..], 0).expect("CSV fixture");
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].close, 1.5);
        assert_eq!(bars[0].volume, None);
    }

    #[test]
    fn body_integer_parser_rejects_fractional_values() {
        let body = json!({"quantity": 1.9, "accountId": 1.0});
        assert_eq!(parse_body_i64(Some(&body), "quantity", 0), 0);
        assert_eq!(parse_body_i64(Some(&body), "accountId", 0), 1);
    }

    #[test]
    fn trace_truncation_never_splits_utf8() {
        let raw = "é".repeat(MAX_TRACE_VALUE_BYTES);
        let summary = TraceSink::text_summary(&raw);
        assert!(summary.is_char_boundary(summary.len()));
        assert!(summary.starts_with(&"é".repeat((MAX_TRACE_VALUE_BYTES - 1) / 2)));
    }
}

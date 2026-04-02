use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategy::{ExecutionStateSnapshot, ExecutionStrategyConfig};
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerKind {
    Tradovate,
    Ironbeam,
}

impl BrokerKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Tradovate => "Tradovate",
            Self::Ironbeam => "Ironbeam",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrokerCapabilities {
    pub replay: bool,
    pub manual_orders: bool,
    pub automated_orders: bool,
    pub native_protection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceCommand {
    Connect(AppConfig),
    EnterReplayMode {
        config: AppConfig,
        bar_type: BarType,
    },
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
    SetReplaySpeed {
        speed: ReplaySpeed,
    },
    ManualOrder {
        action: ManualOrderAction,
    },
    SetTargetPosition {
        target_qty: i32,
        automated: bool,
        reason: String,
    },
    ProfileLegacyOrderStrategyTarget {
        target_qty: i32,
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
    ProbeExecution {
        tag: String,
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
    DebugLog(String),
    Error(String),
    Connected {
        broker: BrokerKind,
        env: TradingEnvironment,
        user_name: Option<String>,
        auth_mode: AuthMode,
        session_kind: SessionKind,
        capabilities: BrokerCapabilities,
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
    ExecutionProbe(ExecutionProbeSnapshot),
    ReplaySpeedUpdated(ReplaySpeed),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionKind {
    Live,
    Replay,
}

impl SessionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Replay => "Replay",
        }
    }
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplaySpeed {
    Realtime,
    X2,
    X5,
    X10,
    X25,
}

impl ReplaySpeed {
    pub fn label(self) -> &'static str {
        match self {
            Self::Realtime => "Realtime",
            Self::X2 => "2x",
            Self::X5 => "5x",
            Self::X10 => "10x",
            Self::X25 => "25x",
        }
    }

    #[cfg(feature = "replay")]
    pub fn multiplier(self) -> f64 {
        match self {
            Self::Realtime => 1.0,
            Self::X2 => 2.0,
            Self::X5 => 5.0,
            Self::X10 => 10.0,
            Self::X25 => 25.0,
        }
    }

    pub fn faster(self) -> Self {
        match self {
            Self::Realtime => Self::X2,
            Self::X2 => Self::X5,
            Self::X5 => Self::X10,
            Self::X10 => Self::X25,
            Self::X25 => Self::X25,
        }
    }

    pub fn slower(self) -> Self {
        match self {
            Self::Realtime => Self::Realtime,
            Self::X2 => Self::Realtime,
            Self::X5 => Self::X2,
            Self::X10 => Self::X5,
            Self::X25 => Self::X10,
        }
    }
}

impl Default for ReplaySpeed {
    fn default() -> Self {
        Self::Realtime
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

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LatencySnapshot {
    pub rest_rtt_ms: Option<u64>,
    pub last_order_ack_ms: Option<u64>,
    pub last_order_seen_ms: Option<u64>,
    pub last_exec_report_ms: Option<u64>,
    pub last_fill_ms: Option<u64>,
    pub last_signal_submit_ms: Option<u64>,
    pub last_signal_seen_ms: Option<u64>,
    pub last_signal_ack_ms: Option<u64>,
    pub last_signal_fill_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeSnapshot {
    pub tag: String,
    pub captured_at_utc: DateTime<Utc>,
    pub execution_state: ExecutionStateSnapshot,
    pub latency: LatencySnapshot,
    pub order_submit_in_flight: bool,
    pub protection_sync_in_flight: bool,
    pub tracker_order_id: Option<i64>,
    pub tracker_order_is_active: bool,
    pub tracker_order_strategy_id: Option<i64>,
    pub tracker_strategy_has_live_orders: bool,
    pub tracker_within_strategy_grace: bool,
    pub tracked_order_strategy_id: Option<i64>,
    pub broker_order_strategy_id: Option<i64>,
    pub broker_order_strategy_status: Option<String>,
    pub broker_strategy_entry_order_qty: Option<i32>,
    pub broker_strategy_bracket_qtys: Vec<i32>,
    pub selected_working_orders: Vec<ExecutionProbeOrder>,
    pub linked_active_orders: Vec<ExecutionProbeOrder>,
    pub managed_protection: Option<ExecutionProbeManagedProtection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeOrder {
    pub order_id: Option<i64>,
    pub order_strategy_id: Option<i64>,
    pub cl_ord_id: Option<String>,
    pub order_type: Option<String>,
    pub action: Option<String>,
    pub order_qty: Option<i32>,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProbeManagedProtection {
    pub signed_qty: i32,
    pub take_profit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub take_profit_order_id: Option<i64>,
    pub stop_order_id: Option<i64>,
    pub take_profit_cl_ord_id: Option<String>,
    pub stop_cl_ord_id: Option<String>,
}

pub const AUTO_CLOSE_MINUTES_BEFORE_SESSION_END: f64 = 15.0;

pub fn infer_session_profile(product: &Value) -> InstrumentSessionProfile {
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

use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategy::{ExecutionStateSnapshot, ExecutionStrategyConfig};
use chrono::NaiveDate;
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
#[cfg(feature = "replay")]
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "replay")]
static NEXT_REPLAY_DOWNLOAD_OPERATION_ID: OnceLock<AtomicU64> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDownloadOperationId(pub u64);

impl ReplayDownloadOperationId {
    #[cfg(feature = "replay")]
    pub fn next() -> Self {
        let counter = NEXT_REPLAY_DOWNLOAD_OPERATION_ID.get_or_init(|| {
            let epoch_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(1);
            let process_seed = u64::from(std::process::id()).rotate_left(32);
            AtomicU64::new((epoch_nanos ^ process_seed).max(1))
        });
        Self(counter.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayDownloadCacheTarget {
    pub dataset_dir: PathBuf,
    pub manifest_path: PathBuf,
}

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
        candle_mode: CandleMode,
        replay_dataset_manifest: Option<std::path::PathBuf>,
    },
    DownloadReplayData {
        operation_id: ReplayDownloadOperationId,
        config: crate::config::AppConfig,
        instrument: String,
        contract: ContractSuggestion,
        target: Option<ReplayDownloadCacheTarget>,
        start_date: NaiveDate,
        end_date: NaiveDate,
        source_kind: String,
        bar_type: BarType,
        candle_mode: CandleMode,
        display_name: Option<String>,
        tags: Vec<String>,
    },
    SearchReplayDownloadContracts {
        operation_id: ReplayDownloadOperationId,
        config: crate::config::AppConfig,
        query: String,
        limit: usize,
    },
    InspectReplayDownloadContract {
        operation_id: ReplayDownloadOperationId,
        config: crate::config::AppConfig,
        contract: ContractSuggestion,
    },
    CancelReplayDownloadOperation {
        operation_id: ReplayDownloadOperationId,
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
        candle_mode: CandleMode,
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
    BrokerRejection(String),
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
    EngineHistoryUpdated(EngineHistorySnapshot),
    Latency(LatencySnapshot),
    ExecutionState(ExecutionStateSnapshot),
    ExecutionProbe(ExecutionProbeSnapshot),
    ReplaySpeedUpdated(ReplaySpeed),
    ReplayDownloadProgress {
        operation_id: ReplayDownloadOperationId,
        phase: ReplayDownloadPhase,
        message: String,
        estimated_rows: Option<u64>,
        estimated_bytes: Option<u64>,
    },
    ReplayDownloadContractSearchResults {
        operation_id: ReplayDownloadOperationId,
        query: String,
        results: Vec<ContractSuggestion>,
    },
    ReplayDownloadContractInspected {
        operation_id: ReplayDownloadOperationId,
        contract: ContractSuggestion,
        suggested_start_date: Option<NaiveDate>,
        suggested_end_date: Option<NaiveDate>,
        suggestion_basis: Option<String>,
    },
    ReplayDownloadCompleted {
        operation_id: ReplayDownloadOperationId,
        cache_root: PathBuf,
        manifest_path: PathBuf,
        data_path: PathBuf,
        rows: u64,
        bytes: u64,
    },
    ReplayDownloadFailed {
        operation_id: ReplayDownloadOperationId,
        phase: ReplayDownloadPhase,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDownloadPhase {
    Idle,
    Searching,
    InspectingContract,
    Ready,
    Authenticating,
    Downloading,
    WritingCache,
    Busy,
    Cancelling,
    Cancelled,
    Complete,
    Failed,
}

impl ReplayDownloadPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::Searching => "Searching",
            Self::InspectingContract => "Inspecting contract",
            Self::Ready => "Ready",
            Self::Authenticating => "Authenticating",
            Self::Downloading => "Downloading",
            Self::WritingCache => "Writing cache",
            Self::Busy => "Busy",
            Self::Cancelling => "Cancelling",
            Self::Cancelled => "Cancelled",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
        }
    }

    #[cfg(feature = "replay")]
    pub fn is_busy(self) -> bool {
        matches!(
            self,
            Self::Searching
                | Self::InspectingContract
                | Self::Authenticating
                | Self::Downloading
                | Self::WritingCache
                | Self::Cancelling
        )
    }
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
#[serde(rename_all = "snake_case")]
pub enum BarKind {
    Minute,
    Second,
    Tick,
    Volume,
    Range,
}

impl BarKind {
    const ALL: [Self; 5] = [
        Self::Minute,
        Self::Second,
        Self::Tick,
        Self::Volume,
        Self::Range,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Minute => "Minute",
            Self::Second => "Seconds",
            Self::Tick => "Tick Count",
            Self::Volume => "Volume",
            Self::Range => "Range",
        }
    }

    fn short_label(self) -> &'static str {
        match self {
            Self::Minute => "Min",
            Self::Second => "Sec",
            Self::Tick => "Tick",
            Self::Volume => "Vol",
            Self::Range => "Range",
        }
    }

    fn next(self) -> Self {
        let index = Self::ALL.iter().position(|kind| *kind == self).unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    fn previous(self) -> Self {
        let index = Self::ALL.iter().position(|kind| *kind == self).unwrap_or(0);
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarType {
    kind: BarKind,
    value: u32,
}

impl BarType {
    pub fn new(kind: BarKind, value: u32) -> Self {
        Self {
            kind,
            value: value.max(1),
        }
    }

    pub fn minute(value: u32) -> Self {
        Self::new(BarKind::Minute, value)
    }

    pub fn second(value: u32) -> Self {
        Self::new(BarKind::Second, value)
    }

    pub fn tick(value: u32) -> Self {
        Self::new(BarKind::Tick, value)
    }

    pub fn volume(value: u32) -> Self {
        Self::new(BarKind::Volume, value)
    }

    pub fn range(value: u32) -> Self {
        Self::new(BarKind::Range, value)
    }

    pub fn kind(self) -> BarKind {
        self.kind
    }

    pub fn value(self) -> u32 {
        self.value
    }

    pub fn is_range(self) -> bool {
        self.kind == BarKind::Range
    }

    pub fn is_one_minute(self) -> bool {
        self.kind == BarKind::Minute && self.value == 1
    }

    pub fn is_time_based(self) -> bool {
        matches!(self.kind, BarKind::Minute | BarKind::Second)
    }

    pub fn supports_candle_mode(self) -> bool {
        !self.is_range()
    }

    pub fn effective_candle_mode(self, candle_mode: CandleMode) -> CandleMode {
        if self.supports_candle_mode() {
            candle_mode
        } else {
            CandleMode::Standard
        }
    }

    pub fn with_value(self, value: u32) -> Self {
        Self::new(self.kind, value)
    }

    pub fn next_kind(self) -> Self {
        match self.kind.next() {
            BarKind::Minute => Self::minute(self.value),
            BarKind::Second => Self::second(self.value),
            BarKind::Tick => Self::tick(self.value),
            BarKind::Volume => Self::volume(self.value),
            BarKind::Range => Self::range(self.value),
        }
    }

    pub fn previous_kind(self) -> Self {
        match self.kind.previous() {
            BarKind::Minute => Self::minute(self.value),
            BarKind::Second => Self::second(self.value),
            BarKind::Tick => Self::tick(self.value),
            BarKind::Volume => Self::volume(self.value),
            BarKind::Range => Self::range(self.value),
        }
    }

    pub fn label(self) -> String {
        format!("{} {}", self.value, self.kind.short_label())
    }

    pub fn mode_label(self, candle_mode: CandleMode) -> String {
        if self.supports_candle_mode() {
            format!("{} {}", candle_mode.label(), self.label())
        } else {
            self.label()
        }
    }

    pub fn chart_description(self) -> Value {
        match self.kind {
            BarKind::Minute => json!({
                "underlyingType": "MinuteBar",
                "elementSize": self.value,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }),
            BarKind::Second => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Seconds",
                "withHistogram": false
            }),
            BarKind::Tick => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            }),
            BarKind::Volume => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Volume",
                "withHistogram": false
            }),
            BarKind::Range => json!({
                "underlyingType": "Tick",
                "elementSize": self.value,
                "elementSizeUnit": "Range",
                "withHistogram": false
            }),
        }
    }
}

impl Default for BarType {
    fn default() -> Self {
        Self::minute(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandleMode {
    Standard,
    HeikinAshi,
}

impl CandleMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "OHLC",
            Self::HeikinAshi => "Heikin Ashi",
        }
    }

    pub fn toggle(self) -> Self {
        match self {
            Self::Standard => Self::HeikinAshi,
            Self::HeikinAshi => Self::Standard,
        }
    }
}

impl Default for CandleMode {
    fn default() -> Self {
        Self::Standard
    }
}

pub fn transform_bars_for_candle_mode(bars: &[Bar], candle_mode: CandleMode) -> Vec<Bar> {
    match candle_mode {
        CandleMode::Standard => bars.to_vec(),
        CandleMode::HeikinAshi => heikin_ashi_bars(bars),
    }
}

fn heikin_ashi_bars(bars: &[Bar]) -> Vec<Bar> {
    let mut transformed = Vec::with_capacity(bars.len());
    let mut previous_open = None::<f64>;
    let mut previous_close = None::<f64>;

    for bar in bars {
        let ha_close = (bar.open + bar.high + bar.low + bar.close) / 4.0;
        let ha_open = match (previous_open, previous_close) {
            (Some(open), Some(close)) => (open + close) / 2.0,
            _ => (bar.open + bar.close) / 2.0,
        };
        let ha_high = bar.high.max(ha_open).max(ha_close);
        let ha_low = bar.low.min(ha_open).min(ha_close);

        transformed.push(Bar {
            ts_ns: bar.ts_ns,
            open: ha_open,
            high: ha_high,
            low: ha_low,
            close: ha_close,
            volume: bar.volume,
        });

        previous_open = Some(ha_open);
        previous_close = Some(ha_close);
    }

    transformed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_candle_mode_preserves_raw_bars() {
        let bars = vec![Bar {
            ts_ns: 1,
            open: 10.0,
            high: 12.0,
            low: 9.0,
            close: 11.0,
            volume: Some(125.0),
        }];

        assert_eq!(
            transform_bars_for_candle_mode(&bars, CandleMode::Standard),
            bars
        );
    }

    #[test]
    fn heikin_ashi_candle_mode_transforms_ohlc_recursively() {
        let bars = vec![
            Bar {
                ts_ns: 1,
                open: 10.0,
                high: 14.0,
                low: 8.0,
                close: 12.0,
                volume: Some(100.0),
            },
            Bar {
                ts_ns: 2,
                open: 12.0,
                high: 16.0,
                low: 11.0,
                close: 15.0,
                volume: Some(150.0),
            },
        ];

        let transformed = transform_bars_for_candle_mode(&bars, CandleMode::HeikinAshi);

        assert_eq!(
            transformed,
            vec![
                Bar {
                    ts_ns: 1,
                    open: 11.0,
                    high: 14.0,
                    low: 8.0,
                    close: 11.0,
                    volume: Some(100.0),
                },
                Bar {
                    ts_ns: 2,
                    open: 11.0,
                    high: 16.0,
                    low: 11.0,
                    close: 13.5,
                    volume: Some(150.0),
                },
            ]
        );
    }

    #[test]
    fn bar_type_chart_descriptions_match_tradovate_units() {
        assert_eq!(
            BarType::minute(3).chart_description(),
            json!({
                "underlyingType": "MinuteBar",
                "elementSize": 3,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::second(15).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 15,
                "elementSizeUnit": "Seconds",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::tick(100).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 100,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::volume(1000).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 1000,
                "elementSizeUnit": "Volume",
                "withHistogram": false
            })
        );
        assert_eq!(
            BarType::range(4).chart_description(),
            json!({
                "underlyingType": "Tick",
                "elementSize": 4,
                "elementSizeUnit": "Range",
                "withHistogram": false
            })
        );
    }

    #[test]
    fn range_bars_force_standard_effective_candle_mode() {
        assert_eq!(
            BarType::range(1).effective_candle_mode(CandleMode::HeikinAshi),
            CandleMode::Standard
        );
        assert_eq!(
            BarType::volume(1000).effective_candle_mode(CandleMode::HeikinAshi),
            CandleMode::HeikinAshi
        );
    }

    #[test]
    fn contract_trade_status_blocks_first_intent_safety_window() {
        let contract = ContractSuggestion {
            id: 4_095_561,
            name: "GCQ6".to_string(),
            description: "Gold August 2026".to_string(),
            raw: json!({
                "contractMaturityId": 59107,
                "_midasContractMaturity": {
                    "id": 59107,
                    "expirationDate": "2026-08-27T17:30Z",
                    "firstIntentDate": "2026-07-31T00:00Z",
                    "archived": false
                }
            }),
        };
        let now = Utc.with_ymd_and_hms(2026, 7, 28, 12, 0, 0).unwrap();

        let status = contract.trade_status_at(now);

        assert!(status.is_blocked());
        assert!(status.label().contains("first intent 2026-07-31"));
        assert!(status.label().contains("5-day opening safety window"));
    }

    #[test]
    fn contract_trade_status_allows_maturity_outside_safety_window() {
        let contract = ContractSuggestion {
            id: 3_267_701,
            name: "GCZ6".to_string(),
            description: "Gold December 2026".to_string(),
            raw: json!({
                "contractMaturityId": 49223,
                "_midasContractMaturity": {
                    "id": 49223,
                    "expirationDate": "2026-12-29T18:30Z",
                    "firstIntentDate": "2026-11-30T00:00Z",
                    "archived": false
                }
            }),
        };
        let now = Utc.with_ymd_and_hms(2026, 7, 28, 12, 0, 0).unwrap();

        let status = contract.trade_status_at(now);

        assert!(!status.is_blocked());
        assert!(matches!(status, ContractTradeStatus::Ready(_)));
        assert!(status.label().contains("first intent 2026-11-30"));
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

pub const CONTRACT_OPENING_SAFETY_DAYS: i64 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractTradeStatus {
    Ready(String),
    Blocked(String),
    Unknown(String),
}

impl ContractTradeStatus {
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Blocked(_))
    }

    pub fn label(&self) -> String {
        match self {
            Self::Ready(detail) => format!("READY — {detail}"),
            Self::Blocked(detail) => format!("BLOCKED — {detail}"),
            Self::Unknown(detail) => format!("UNKNOWN — {detail}"),
        }
    }
}

impl ContractSuggestion {
    pub fn trade_status(&self) -> ContractTradeStatus {
        self.trade_status_at(Utc::now())
    }

    pub fn trade_status_at(&self, now: DateTime<Utc>) -> ContractTradeStatus {
        let Some(maturity) = self.raw.get("_midasContractMaturity") else {
            return ContractTradeStatus::Unknown(
                "maturity metadata unavailable; selection is allowed".to_string(),
            );
        };

        if maturity
            .get("archived")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return ContractTradeStatus::Blocked("contract maturity is archived".to_string());
        }

        if let Some(first_intent) = maturity_timestamp(maturity, "firstIntentDate") {
            let date = first_intent.format("%Y-%m-%d");
            if now >= first_intent {
                return ContractTradeStatus::Blocked(format!("first intent date passed on {date}"));
            }
            if now >= first_intent - chrono::Duration::days(CONTRACT_OPENING_SAFETY_DAYS) {
                return ContractTradeStatus::Blocked(format!(
                    "first intent {date} is within the {CONTRACT_OPENING_SAFETY_DAYS}-day opening safety window"
                ));
            }
            let days = (first_intent.date_naive() - now.date_naive()).num_days();
            return ContractTradeStatus::Ready(format!(
                "first intent {date} ({days} days away); volume becomes available after subscription"
            ));
        }

        if let Some(expiration) = maturity_timestamp(maturity, "expirationDate") {
            let date = expiration.format("%Y-%m-%d");
            if now >= expiration {
                return ContractTradeStatus::Blocked(format!("expired on {date}"));
            }
            if now >= expiration - chrono::Duration::days(CONTRACT_OPENING_SAFETY_DAYS) {
                return ContractTradeStatus::Blocked(format!(
                    "expiration {date} is within the {CONTRACT_OPENING_SAFETY_DAYS}-day opening safety window"
                ));
            }
            return ContractTradeStatus::Ready(format!(
                "expires {date}; first intent date unavailable"
            ));
        }

        ContractTradeStatus::Unknown("maturity dates unavailable; selection is allowed".to_string())
    }
}

fn maturity_timestamp(maturity: &Value, key: &str) -> Option<DateTime<Utc>> {
    let raw = maturity.get(key).and_then(Value::as_str)?;
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%MZ")
                .ok()
                .map(|timestamp| Utc.from_utc_datetime(&timestamp))
        })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bar {
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
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

    pub fn evaluate_with_blockout(
        self,
        ts_ns: i64,
        blockout_minutes_before_close: f64,
    ) -> InstrumentSessionWindow {
        if ts_ns <= 0 {
            return InstrumentSessionWindow {
                session_open: true,
                minutes_to_close: None,
                hold_entries: false,
            };
        }

        let dt_et = DateTime::<Utc>::from_timestamp_nanos(ts_ns).with_timezone(&New_York);
        let blockout_minutes_before_close = blockout_minutes_before_close.max(0.0);
        match self {
            Self::FuturesGlobex => futures_globex_window(dt_et, blockout_minutes_before_close),
            Self::EquityRth => equity_rth_window(dt_et, blockout_minutes_before_close),
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
    pub candle_mode: CandleMode,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineHistoryFill {
    pub fill_id: i64,
    pub order_id: i64,
    pub ts_ns: i64,
    pub side: TradeMarkerSide,
    pub qty: i32,
    pub price: f64,
    pub realized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineHistorySnapshot {
    pub run_id: String,
    pub started_at_utc: DateTime<Utc>,
    pub account_id: i64,
    pub account_name: String,
    pub contract_id: i64,
    pub contract_name: String,
    pub position_qty: i32,
    pub average_entry_price: Option<f64>,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub fees: f64,
    pub wins: usize,
    pub losses: usize,
    pub fills: Vec<EngineHistoryFill>,
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

fn futures_globex_window(
    dt_et: DateTime<chrono_tz::Tz>,
    blockout_minutes_before_close: f64,
) -> InstrumentSessionWindow {
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
        .map(|minutes| minutes <= blockout_minutes_before_close)
        .unwrap_or(false);

    InstrumentSessionWindow {
        session_open: true,
        minutes_to_close,
        hold_entries,
    }
}

fn equity_rth_window(
    dt_et: DateTime<chrono_tz::Tz>,
    blockout_minutes_before_close: f64,
) -> InstrumentSessionWindow {
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
        .map(|minutes| minutes <= blockout_minutes_before_close)
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

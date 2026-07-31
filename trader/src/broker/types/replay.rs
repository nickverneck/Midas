use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(feature = "replay")]
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "replay")]
static NEXT_REPLAY_DOWNLOAD_OPERATION_ID: OnceLock<AtomicU64> = OnceLock::new();

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayEngineMode {
    /// Original async bar loop retained during the deterministic-engine migration.
    #[default]
    Legacy,
    /// Market-time ordering is owned by the deterministic virtual event queue.
    Deterministic,
}

impl ReplayEngineMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Legacy => "Legacy compatibility",
            Self::Deterministic => "Deterministic virtual time",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayWindowSnapshot {
    pub preset: String,
    pub input_timezone: String,
    pub warmup_start: DateTime<Utc>,
    pub evaluation_start: DateTime<Utc>,
    pub evaluation_end: DateTime<Utc>,
    pub warmup_rows: usize,
    pub evaluation_rows_total: usize,
    pub evaluation_rows_processed: usize,
}

impl ReplayWindowSnapshot {
    pub fn local_range_label(&self) -> String {
        let timezone = self.input_timezone.parse::<Tz>().unwrap_or(chrono_tz::UTC);
        format!(
            "{} to {}",
            self.evaluation_start
                .with_timezone(&timezone)
                .format("%Y-%m-%d %H:%M:%S %Z"),
            self.evaluation_end
                .with_timezone(&timezone)
                .format("%Y-%m-%d %H:%M:%S %Z")
        )
    }

    pub fn evaluation_rows_remaining(&self) -> usize {
        self.evaluation_rows_total
            .saturating_sub(self.evaluation_rows_processed)
    }
}

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

pub const REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayLatencyModel {
    #[default]
    IgnoredLegacy,
    Fixed,
    ObservedMean,
    ObservedP95,
    ObservedP99,
    SeededObserved,
}

impl ReplayLatencyModel {
    pub fn label(self) -> &'static str {
        match self {
            Self::IgnoredLegacy => "ignored in Legacy",
            Self::Fixed => "fixed",
            Self::ObservedMean => "observed mean",
            Self::ObservedP95 => "observed p95",
            Self::ObservedP99 => "observed p99",
            Self::SeededObserved => "seeded observed sample",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayLatencyConfig {
    pub model: ReplayLatencyModel,
    pub fixed_latency_ms: u64,
    pub observed_samples_ms: Vec<u64>,
    pub seed: u64,
}

impl Default for ReplayLatencyConfig {
    fn default() -> Self {
        Self {
            model: ReplayLatencyModel::Fixed,
            fixed_latency_ms: 0,
            observed_samples_ms: Vec::new(),
            seed: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBarProtectionPolicy {
    /// When OHLC reaches both bracket legs, assume the adverse stop happened first.
    #[default]
    Conservative,
    /// When OHLC reaches both bracket legs, assume the profit target happened first.
    Optimistic,
    /// Select the reachable leg nearest the raw bar open; ties choose the stop.
    NearestOpen,
}

impl ReplayBarProtectionPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Conservative => "conservative stop-first",
            Self::Optimistic => "optimistic target-first",
            Self::NearestOpen => "nearest raw open",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayFillPriceSource {
    #[default]
    LegacyReferencePrice,
    RawBarOpen,
    RawBarOhlc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayExecutionFill {
    pub sequence: u64,
    pub lifecycle_sequence: Option<u64>,
    pub fill_id: i64,
    pub order_id: i64,
    pub order_strategy_id: Option<i64>,
    pub protection_order_id: Option<i64>,
    pub account_id: i64,
    pub contract_id: i64,
    pub contract_name: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub signal_timestamp_ns: Option<i64>,
    pub submission_timestamp_ns: Option<i64>,
    pub exchange_arrival_timestamp_ns: Option<i64>,
    pub acknowledgement_timestamp_ns: Option<i64>,
    pub fill_timestamp_ns: i64,
    pub fill_price_source: ReplayFillPriceSource,
    pub exit_reason: Option<String>,
    pub latency_ms: u64,
    pub tick_size: Option<f64>,
    pub value_per_point: Option<f64>,
    pub gross_realized_pnl_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ReplayExecutionLedgerSnapshot {
    pub schema_version: u32,
    pub fee_neutral: bool,
    pub engine_mode: ReplayEngineMode,
    pub latency_model: ReplayLatencyModel,
    pub fixed_latency_ms: u64,
    pub latency_seed: Option<u64>,
    pub observed_latency_sample_count: usize,
    pub bar_protection_policy: ReplayBarProtectionPolicy,
    pub signal_source: String,
    pub gross_realized_pnl: f64,
    pub fills: Vec<ReplayExecutionFill>,
}

impl Default for ReplayExecutionLedgerSnapshot {
    fn default() -> Self {
        Self {
            schema_version: REPLAY_EXECUTION_LEDGER_SCHEMA_VERSION,
            fee_neutral: true,
            engine_mode: ReplayEngineMode::Legacy,
            latency_model: ReplayLatencyModel::IgnoredLegacy,
            fixed_latency_ms: 0,
            latency_seed: None,
            observed_latency_sample_count: 0,
            bar_protection_policy: ReplayBarProtectionPolicy::NearestOpen,
            signal_source: String::new(),
            gross_realized_pnl: 0.0,
            fills: Vec::new(),
        }
    }
}

impl ReplayExecutionLedgerSnapshot {
    pub fn summary(&self) -> ReplayExecutionLedgerSummary {
        ReplayExecutionLedgerSummary {
            schema_version: self.schema_version,
            engine_mode: self.engine_mode,
            latency_model: self.latency_model,
            fixed_latency_ms: self.fixed_latency_ms,
            latency_seed: self.latency_seed,
            observed_latency_sample_count: self.observed_latency_sample_count,
            bar_protection_policy: self.bar_protection_policy,
            signal_source: self.signal_source.clone(),
            fill_count: self.fills.len(),
            gross_realized_pnl: self.gross_realized_pnl,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ReplayExecutionLedgerSummary {
    pub schema_version: u32,
    pub engine_mode: ReplayEngineMode,
    pub latency_model: ReplayLatencyModel,
    pub fixed_latency_ms: u64,
    pub latency_seed: Option<u64>,
    pub observed_latency_sample_count: usize,
    pub bar_protection_policy: ReplayBarProtectionPolicy,
    pub signal_source: String,
    pub fill_count: usize,
    pub gross_realized_pnl: f64,
}

impl Default for ReplayExecutionLedgerSummary {
    fn default() -> Self {
        ReplayExecutionLedgerSnapshot::default().summary()
    }
}

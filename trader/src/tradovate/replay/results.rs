use super::fees::ReplayFeeSchedule;
use super::liquidation::{
    ReplayLiquidationAnalysis, ReplayLiquidationConfig, liquidation_csv,
    simulate_replay_liquidation,
};
use super::risk::{ReplayMarginAnalysis, ReplayMarginConfig, compute_margin_analysis};
use super::state::ReplayState;
use crate::broker::{
    BarType, CandleMode, MarketSnapshot, ReplayExecutionFill, ReplayExecutionLedgerSnapshot,
    ReplayFillPriceSource, ReplaySignalDiagnostic,
};
use crate::config::{AppConfig, TradingEnvironment};
use crate::strategy::{ExecutionStrategyConfig, StrategyKind};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const REPLAY_RESULT_SCHEMA_VERSION: u32 = 3;
pub(crate) const FEE_NEUTRAL_SCENARIO_NAME: &str = "fee_neutral";
const SIGNAL_DIAGNOSTICS_CSV_HEADER: &str = "bar_timestamp_ns,bar_open,bar_high,bar_low,bar_close,bar_index,bar_count,strategy,execution_path,signal_timing,signal_delay_bars,signal,raw_signal,effective_signal,raw_buy_signal,raw_sell_signal,effective_buy_signal,effective_sell_signal,current_position_qty,effective_position_qty,target_qty,decision,gate_reason,order_action,order_qty,indicator_name,previous_fast_indicator,previous_slow_indicator,fast_indicator,slow_indicator,auxiliary_name,auxiliary_value,hold_reason,strategy_detail,fingerprint\n";

#[derive(Debug, Clone)]
pub(crate) struct ReplayResultWriteOutcome {
    pub(crate) result_path: PathBuf,
    pub(crate) fill_count: usize,
    pub(crate) trade_count: usize,
    pub(crate) required_starting_capital: Option<f64>,
    pub(crate) initial_capital_sufficient: Option<bool>,
}

/// One durable replay result discovered under the configured result root.
/// The path is retained so the analytics screen can identify the exact run
/// directory without opening a sidecar file.
#[derive(Debug, Clone)]
pub(crate) struct ReplayResultEntry {
    pub(crate) path: PathBuf,
    pub(crate) document: ReplayResultDocument,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ReplayResultLibrarySnapshot {
    pub(crate) entries: Vec<ReplayResultEntry>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReplayEquityPoint {
    pub(crate) timestamp_ns: i64,
    pub(crate) equity: f64,
    pub(crate) cumulative_net_pnl: f64,
    pub(crate) position_qty: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayResultInput<'a> {
    pub(crate) config: &'a AppConfig,
    pub(crate) replay: &'a ReplayState,
    pub(crate) market: &'a MarketSnapshot,
    pub(crate) ledger: &'a ReplayExecutionLedgerSnapshot,
    pub(crate) strategy: &'a ExecutionStrategyConfig,
    pub(crate) bar_type: BarType,
    pub(crate) candle_mode: CandleMode,
    pub(crate) run_id: &'a str,
    pub(crate) started_at_utc: DateTime<Utc>,
    pub(crate) completed_at_utc: DateTime<Utc>,
    pub(crate) error: Option<&'a str>,
    pub(crate) signal_diagnostics: Option<&'a [ReplaySignalDiagnostic]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayResultStatus {
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ReplayResultDocument {
    pub(crate) schema_version: u32,
    pub(crate) result_type: String,
    pub(crate) run_id: String,
    pub(crate) status: ReplayResultStatus,
    pub(crate) error: Option<String>,
    pub(crate) created_at_utc: DateTime<Utc>,
    pub(crate) started_at_utc: DateTime<Utc>,
    pub(crate) completed_at_utc: DateTime<Utc>,
    pub(crate) metadata: ReplayResultMetadata,
    pub(crate) summary: ReplayResultSummary,
    pub(crate) artifacts: ReplayResultArtifacts,
    #[serde(default = "default_active_fee_scenario")]
    pub(crate) active_fee_scenario: String,
    #[serde(default)]
    pub(crate) fee_scenarios: Vec<ReplayFeeScenario>,
    #[serde(default)]
    pub(crate) margin_analysis: Option<ReplayMarginAnalysis>,
    #[serde(default)]
    pub(crate) trade_excursions: Option<Vec<ReplayTradeExcursion>>,
    #[serde(default)]
    pub(crate) liquidation_analysis: Option<ReplayLiquidationAnalysis>,
    /// Optional parent-sweep metadata. Keeping this in the typed document
    /// prevents accounting-only repricing from dropping the metadata that
    /// makes a completed child safe to resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) sweep: Option<Value>,
    /// The fee-neutral ledger is embedded so a result is self-contained even
    /// when the CSV sidecars are moved or inspected independently.
    pub(crate) ledger: ReplayExecutionLedgerSnapshot,
}

/// Load the saved result documents below a replay result root for the
/// read-only analytics screen. Invalid individual runs are skipped and
/// returned as warnings so one damaged artifact does not hide other runs.
pub(crate) fn load_replay_result_entries(root: &Path) -> ReplayResultLibrarySnapshot {
    let mut snapshot = ReplayResultLibrarySnapshot::default();
    let mut candidates = Vec::new();

    if root.is_file() {
        candidates.push(root.to_path_buf());
    } else if root.is_dir() {
        match fs::read_dir(root) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() && path.file_name().is_some_and(|name| name == "result.json")
                    {
                        candidates.push(path);
                    } else if path.is_dir() {
                        let result_path = path.join("result.json");
                        if result_path.is_file() {
                            candidates.push(result_path);
                        }
                    }
                }
            }
            Err(error) => snapshot.warnings.push(format!(
                "Could not scan replay result directory {}: {error}",
                root.display()
            )),
        }
    }

    for path in candidates {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) => {
                snapshot
                    .warnings
                    .push(format!("Could not read {}: {error}", path.display()));
                continue;
            }
        };
        match serde_json::from_slice::<ReplayResultDocument>(&bytes) {
            Ok(document) => snapshot.entries.push(ReplayResultEntry { path, document }),
            Err(error) => snapshot.warnings.push(format!(
                "Could not parse replay result {}: {error}",
                path.display()
            )),
        }
    }

    snapshot.entries.sort_by(|left, right| {
        right
            .document
            .completed_at_utc
            .cmp(&left.document.completed_at_utc)
            .then_with(|| right.path.cmp(&left.path))
    });
    snapshot
}

/// Load the optional signal diagnostics sidecar for one saved replay result.
/// The result index remains cheap because this is intentionally lazy: the
/// Analytics screen calls it only after a run is selected.
pub(crate) fn load_replay_signal_diagnostics(
    entry: &ReplayResultEntry,
) -> Result<Vec<ReplaySignalDiagnostic>> {
    let Some(file_name) = entry.document.artifacts.signals_csv.as_deref() else {
        return Ok(Vec::new());
    };
    let directory = entry
        .path
        .parent()
        .context("replay result path has no parent directory")?;
    let path = directory.join(file_name);
    let bytes = fs::read(&path)
        .with_context(|| format!("read replay signal diagnostics {}", path.display()))?;
    let records = parse_csv_records(&bytes)?;
    let Some(header) = records.first() else {
        return Ok(Vec::new());
    };
    let expected_header = SIGNAL_DIAGNOSTICS_CSV_HEADER
        .trim_end_matches('\n')
        .split(',')
        .collect::<Vec<_>>();
    if header.iter().map(String::as_str).collect::<Vec<_>>() != expected_header {
        anyhow::bail!(
            "unexpected signal diagnostics CSV header in {}",
            path.display()
        );
    }

    records
        .iter()
        .skip(1)
        .enumerate()
        .map(|(row_index, fields)| parse_signal_diagnostic_row(fields, row_index + 2))
        .collect()
}

/// Load the selected run's equity sidecar lazily for chart/hourly analytics.
/// The JSON index stays inexpensive and older results without an equity file
/// simply report an unavailable chart.
pub(crate) fn load_replay_equity(entry: &ReplayResultEntry) -> Result<Vec<ReplayEquityPoint>> {
    let directory = entry
        .path
        .parent()
        .context("replay result path has no parent directory")?;
    let path = directory.join(&entry.document.artifacts.equity_csv);
    let bytes =
        fs::read(&path).with_context(|| format!("read replay equity {}", path.display()))?;
    let records = parse_csv_records(&bytes)?;
    let Some(header) = records.first() else {
        return Ok(Vec::new());
    };
    let expected = [
        "timestamp_ns",
        "equity",
        "initial_capital",
        "cumulative_gross_realized_pnl",
        "cumulative_fees",
        "cumulative_net_pnl",
        "position_qty",
        "mark_price",
        "execution_precision",
    ];
    if header.iter().map(String::as_str).collect::<Vec<_>>() != expected {
        anyhow::bail!("unexpected equity CSV header in {}", path.display());
    }
    records
        .iter()
        .skip(1)
        .enumerate()
        .map(|(row_index, fields)| {
            if fields.len() != expected.len() {
                anyhow::bail!(
                    "equity row {} has {} fields; expected {}",
                    row_index + 2,
                    fields.len(),
                    expected.len()
                );
            }
            Ok(ReplayEquityPoint {
                timestamp_ns: fields[0].parse().with_context(|| {
                    format!("invalid equity timestamp at row {}", row_index + 2)
                })?,
                equity: fields[1]
                    .parse()
                    .with_context(|| format!("invalid equity value at row {}", row_index + 2))?,
                cumulative_net_pnl: fields[5]
                    .parse()
                    .with_context(|| format!("invalid equity net PnL at row {}", row_index + 2))?,
                position_qty: fields[6]
                    .parse()
                    .with_context(|| format!("invalid equity position at row {}", row_index + 2))?,
            })
        })
        .collect()
}

fn parse_signal_diagnostic_row(
    fields: &[String],
    row_number: usize,
) -> Result<ReplaySignalDiagnostic> {
    if fields.len() != 35 {
        anyhow::bail!(
            "signal diagnostics row {row_number} has {} fields; expected 35",
            fields.len()
        );
    }
    let field = |index: usize, name: &str| -> Result<&str> {
        fields
            .get(index)
            .map(String::as_str)
            .with_context(|| format!("missing signal diagnostics field {name}"))
    };
    let required =
        |index: usize, name: &str| -> Result<String> { Ok(field(index, name)?.to_string()) };

    Ok(ReplaySignalDiagnostic {
        bar_timestamp_ns: parse_csv_value(fields, 0, "bar_timestamp_ns", row_number)?,
        bar_open: parse_csv_value(fields, 1, "bar_open", row_number)?,
        bar_high: parse_csv_value(fields, 2, "bar_high", row_number)?,
        bar_low: parse_csv_value(fields, 3, "bar_low", row_number)?,
        bar_close: parse_csv_value(fields, 4, "bar_close", row_number)?,
        bar_index: parse_csv_optional(fields, 5, "bar_index", row_number)?,
        bar_count: parse_csv_value(fields, 6, "bar_count", row_number)?,
        strategy: required(7, "strategy")?,
        execution_path: required(8, "execution_path")?,
        signal_timing: required(9, "signal_timing")?,
        signal_delay_bars: parse_csv_value(fields, 10, "signal_delay_bars", row_number)?,
        signal: required(11, "signal")?,
        raw_signal: required(12, "raw_signal")?,
        effective_signal: required(13, "effective_signal")?,
        raw_buy_signal: parse_csv_value(fields, 14, "raw_buy_signal", row_number)?,
        raw_sell_signal: parse_csv_value(fields, 15, "raw_sell_signal", row_number)?,
        effective_buy_signal: parse_csv_value(fields, 16, "effective_buy_signal", row_number)?,
        effective_sell_signal: parse_csv_value(fields, 17, "effective_sell_signal", row_number)?,
        current_position_qty: parse_csv_value(fields, 18, "current_position_qty", row_number)?,
        effective_position_qty: parse_csv_value(fields, 19, "effective_position_qty", row_number)?,
        target_qty: parse_csv_optional(fields, 20, "target_qty", row_number)?,
        decision: required(21, "decision")?,
        gate_reason: required(22, "gate_reason")?,
        order_action: parse_csv_optional_string(fields, 23, "order_action")?,
        order_qty: parse_csv_optional(fields, 24, "order_qty", row_number)?,
        indicator_name: required(25, "indicator_name")?,
        previous_fast_indicator: parse_csv_optional(
            fields,
            26,
            "previous_fast_indicator",
            row_number,
        )?,
        previous_slow_indicator: parse_csv_optional(
            fields,
            27,
            "previous_slow_indicator",
            row_number,
        )?,
        fast_indicator: parse_csv_optional(fields, 28, "fast_indicator", row_number)?,
        slow_indicator: parse_csv_optional(fields, 29, "slow_indicator", row_number)?,
        auxiliary_name: parse_csv_optional_string(fields, 30, "auxiliary_name")?,
        auxiliary_value: parse_csv_optional(fields, 31, "auxiliary_value", row_number)?,
        hold_reason: parse_csv_optional_string(fields, 32, "hold_reason")?,
        strategy_detail: required(33, "strategy_detail")?,
        fingerprint: parse_csv_optional(fields, 34, "fingerprint", row_number)?,
    })
}

fn parse_csv_value<T>(fields: &[String], index: usize, name: &str, row: usize) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let value = fields
        .get(index)
        .with_context(|| format!("missing {name} in signal diagnostics row {row}"))?;
    value
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("invalid {name} in signal diagnostics row {row}: {error}"))
}

fn parse_csv_optional<T>(
    fields: &[String],
    index: usize,
    name: &str,
    row: usize,
) -> Result<Option<T>>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    let value = fields
        .get(index)
        .with_context(|| format!("missing {name} in signal diagnostics row {row}"))?;
    if value.is_empty() {
        Ok(None)
    } else {
        value.parse::<T>().map(Some).map_err(|error| {
            anyhow::anyhow!("invalid {name} in signal diagnostics row {row}: {error}")
        })
    }
}

fn parse_csv_optional_string(
    fields: &[String],
    index: usize,
    name: &str,
) -> Result<Option<String>> {
    let value = fields
        .get(index)
        .with_context(|| format!("missing {name} in signal diagnostics row"))?;
    Ok((!value.is_empty()).then(|| value.clone()))
}

fn parse_csv_records(bytes: &[u8]) -> Result<Vec<Vec<String>>> {
    let input = String::from_utf8(bytes.to_vec()).context("signal diagnostics CSV is not UTF-8")?;
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(character) = chars.next() {
        if in_quotes {
            if character == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(character);
            }
            continue;
        }

        match character {
            '"' => in_quotes = true,
            ',' => {
                record.push(std::mem::take(&mut field));
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                if !(record.len() == 1 && record[0].is_empty()) {
                    records.push(std::mem::take(&mut record));
                } else {
                    record.clear();
                }
            }
            '\r' => {}
            other => field.push(other),
        }
    }

    if in_quotes {
        anyhow::bail!("unterminated quoted field in signal diagnostics CSV");
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ReplayResultMetadata {
    pub(crate) app_version: String,
    pub(crate) package_version: String,
    pub(crate) git_commit: Option<String>,
    pub(crate) broker: String,
    pub(crate) environment: TradingEnvironment,
    pub(crate) run_mode: String,
    /// Backend used by a sweep child.  These optional fields intentionally
    /// live in the typed result document so accounting-only repricing keeps
    /// the execution provenance instead of dropping JSON extensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) execution_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) fallback_reason: Option<String>,
    pub(crate) strategy: ExecutionStrategyConfig,
    #[serde(default)]
    pub(crate) path_dependence: ReplayPathDependence,
    pub(crate) signal_source: String,
    pub(crate) fill_model: crate::broker::ReplayFillModel,
    pub(crate) fill_price_sources: Vec<String>,
    pub(crate) execution_precisions: Vec<String>,
    pub(crate) latency_model: crate::broker::ReplayLatencyModel,
    pub(crate) fixed_latency_ms: u64,
    pub(crate) latency_seed: Option<u64>,
    pub(crate) observed_latency_sample_count: usize,
    pub(crate) bar_protection_policy: crate::broker::ReplayBarProtectionPolicy,
    pub(crate) fee_model: String,
    pub(crate) initial_capital: f64,
    pub(crate) account_currency: String,
    pub(crate) margin_model: String,
    pub(crate) account_id: i64,
    pub(crate) account_name: String,
    pub(crate) contract_id: i64,
    pub(crate) contract_name: String,
    pub(crate) tick_size: Option<f64>,
    pub(crate) value_per_point: Option<f64>,
    pub(crate) bar_type: BarType,
    pub(crate) candle_mode: CandleMode,
    pub(crate) timezone: String,
    pub(crate) replay_file_path: String,
    pub(crate) replay_dom_file_path: Option<String>,
    pub(crate) replay_cache_dir: String,
    #[serde(default)]
    pub(crate) post_exit_continuation_horizon_bars: usize,
    pub(crate) dataset_view: Option<String>,
    pub(crate) evaluation_start_utc: Option<DateTime<Utc>>,
    pub(crate) evaluation_end_utc: Option<DateTime<Utc>>,
    pub(crate) warmup_start_utc: Option<DateTime<Utc>>,
    pub(crate) market_first_timestamp_ns: Option<i64>,
    pub(crate) market_last_timestamp_ns: Option<i64>,
    pub(crate) warmup_rows: usize,
    pub(crate) evaluation_rows_total: usize,
    pub(crate) evaluation_rows_processed: usize,
    #[serde(default)]
    pub(crate) signal_diagnostics_enabled: bool,
    #[serde(default)]
    pub(crate) signal_diagnostic_count: usize,
}

/// Records whether accounting overlays can be applied without changing the
/// strategy execution path.  Native fixed-quantity strategies currently meet
/// that condition; unknown/legacy results fail closed and request a rerun.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayPathDependence {
    pub(crate) fee_path_independent: bool,
    pub(crate) account_state_dependent: bool,
    pub(crate) reasons: Vec<String>,
}

impl Default for ReplayPathDependence {
    fn default() -> Self {
        Self {
            fee_path_independent: false,
            account_state_dependent: true,
            reasons: vec!["path-dependence metadata was not recorded".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ReplayResultSummary {
    pub(crate) initial_capital: f64,
    pub(crate) ending_equity: f64,
    pub(crate) gross_pnl: f64,
    pub(crate) net_pnl: f64,
    pub(crate) fees: f64,
    pub(crate) return_on_initial_capital_pct: Option<f64>,
    pub(crate) fill_count: usize,
    pub(crate) trade_count: usize,
    pub(crate) closed_trade_count: usize,
    pub(crate) win_count: usize,
    pub(crate) loss_count: usize,
    pub(crate) win_rate_pct: Option<f64>,
    pub(crate) profit_factor: Option<f64>,
    pub(crate) max_drawdown: f64,
    pub(crate) max_drawdown_pct: Option<f64>,
    pub(crate) max_open_position: f64,
    pub(crate) precision: Vec<String>,
    pub(crate) exit_reason_counts: BTreeMap<String, usize>,
    pub(crate) evaluation_rows_processed: usize,
    #[serde(default)]
    pub(crate) required_starting_capital: Option<f64>,
    #[serde(default)]
    pub(crate) peak_margin_requirement: Option<f64>,
    #[serde(default)]
    pub(crate) minimum_equity_buffer_over_margin: Option<f64>,
    #[serde(default)]
    pub(crate) initial_capital_sufficient: Option<bool>,
    #[serde(default)]
    pub(crate) average_mfe_pnl: Option<f64>,
    #[serde(default)]
    pub(crate) average_mae_pnl: Option<f64>,
    #[serde(default)]
    pub(crate) average_giveback: Option<f64>,
    #[serde(default)]
    pub(crate) median_giveback: Option<f64>,
    #[serde(default)]
    pub(crate) largest_giveback: Option<f64>,
    #[serde(default)]
    pub(crate) average_mfe_capture_ratio: Option<f64>,
    #[serde(default)]
    pub(crate) average_post_exit_favorable_pnl: Option<f64>,
    #[serde(default)]
    pub(crate) largest_post_exit_favorable_pnl: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ReplayResultArtifacts {
    pub(crate) result_json: String,
    pub(crate) trades_csv: String,
    pub(crate) fills_csv: String,
    pub(crate) equity_csv: String,
    #[serde(default)]
    pub(crate) trades_parquet: Option<String>,
    #[serde(default)]
    pub(crate) fills_parquet: Option<String>,
    #[serde(default)]
    pub(crate) equity_parquet: Option<String>,
    #[serde(default)]
    pub(crate) fee_scenarios_csv: Option<String>,
    #[serde(default)]
    pub(crate) margin_csv: Option<String>,
    #[serde(default)]
    pub(crate) trade_excursions_csv: Option<String>,
    #[serde(default)]
    pub(crate) signals_csv: Option<String>,
    #[serde(default)]
    pub(crate) trade_excursions_parquet: Option<String>,
    #[serde(default)]
    pub(crate) signals_parquet: Option<String>,
    #[serde(default)]
    pub(crate) liquidation_csv: Option<String>,
}

/// Per-trade opportunity and adverse-path measurements. Prices are always
/// based on the tradable replay path; synthetic candle values are never used
/// for excursion pricing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayTradeExcursion {
    pub(crate) trade_id: usize,
    pub(crate) account_id: i64,
    pub(crate) contract_id: i64,
    pub(crate) contract_name: String,
    pub(crate) side: String,
    pub(crate) quantity: f64,
    pub(crate) entry_timestamp_ns: i64,
    pub(crate) entry_price: f64,
    pub(crate) exit_timestamp_ns: Option<i64>,
    pub(crate) exit_price: Option<f64>,
    pub(crate) realized_gross_pnl: f64,
    #[serde(default)]
    pub(crate) realized_net_pnl: f64,
    #[serde(default)]
    pub(crate) exit_reason: Option<String>,
    pub(crate) mfe_points: f64,
    pub(crate) mae_points: f64,
    pub(crate) mfe_price: f64,
    pub(crate) mae_price: f64,
    pub(crate) mfe_pnl: Option<f64>,
    pub(crate) mae_pnl: Option<f64>,
    pub(crate) giveback: Option<f64>,
    pub(crate) mfe_capture_ratio: Option<f64>,
    pub(crate) mfe_timestamp_ns: Option<i64>,
    pub(crate) time_to_mfe_ns: Option<i64>,
    pub(crate) time_from_mfe_to_exit_ns: Option<i64>,
    pub(crate) bars_to_mfe: Option<usize>,
    pub(crate) ticks_to_mfe: Option<usize>,
    pub(crate) bars_from_mfe_to_exit: Option<usize>,
    pub(crate) ticks_from_mfe_to_exit: Option<usize>,
    #[serde(default)]
    pub(crate) post_exit_favorable_points: Option<f64>,
    #[serde(default)]
    pub(crate) post_exit_favorable_price: Option<f64>,
    #[serde(default)]
    pub(crate) post_exit_favorable_pnl: Option<f64>,
    #[serde(default)]
    pub(crate) post_exit_favorable_timestamp_ns: Option<i64>,
    #[serde(default)]
    pub(crate) post_exit_bars_observed: Option<usize>,
    pub(crate) path_precision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(crate) struct ReplayFeeScenario {
    pub(crate) schedule: ReplayFeeSchedule,
    pub(crate) fees: f64,
    pub(crate) gross_pnl: f64,
    pub(crate) net_pnl: f64,
    pub(crate) ending_equity: f64,
    pub(crate) return_on_initial_capital_pct: Option<f64>,
    pub(crate) max_drawdown: f64,
    pub(crate) max_drawdown_pct: Option<f64>,
    pub(crate) profit_factor: Option<f64>,
}

impl Default for ReplayFeeScenario {
    fn default() -> Self {
        Self {
            schedule: ReplayFeeSchedule::default(),
            fees: 0.0,
            gross_pnl: 0.0,
            net_pnl: 0.0,
            ending_equity: 0.0,
            return_on_initial_capital_pct: None,
            max_drawdown: 0.0,
            max_drawdown_pct: None,
            profit_factor: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TradeRow {
    pub(super) account_id: i64,
    pub(super) contract_id: i64,
    pub(super) contract_name: String,
    pub(super) side: String,
    pub(super) quantity: f64,
    pub(super) entry_timestamp_ns: i64,
    pub(super) entry_price: f64,
    pub(super) exit_timestamp_ns: Option<i64>,
    pub(super) exit_price: Option<f64>,
    pub(super) gross_realized_pnl: f64,
    pub(super) fees: f64,
    pub(super) net_realized_pnl: f64,
    pub(super) exit_reason: Option<String>,
    pub(super) fill_count: usize,
    pub(super) execution_precision: String,
}

#[derive(Debug, Clone)]
struct OpenTrade {
    account_id: i64,
    contract_id: i64,
    contract_name: String,
    side: String,
    quantity: f64,
    entry_timestamp_ns: i64,
    entry_price: f64,
    exit_timestamp_ns: Option<i64>,
    exit_price: Option<f64>,
    gross_realized_pnl: f64,
    fees: f64,
    exit_reason: Option<String>,
    fill_count: usize,
    execution_precisions: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub(super) struct EquityRow {
    pub(super) timestamp_ns: i64,
    pub(super) initial_capital: f64,
    pub(super) equity: f64,
    pub(super) cumulative_gross_realized_pnl: f64,
    pub(super) cumulative_fees: f64,
    pub(super) cumulative_net_pnl: f64,
    pub(super) position_qty: f64,
    pub(super) mark_price: Option<f64>,
    pub(super) execution_precision: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PositionKey {
    account_id: i64,
    contract_id: i64,
}

#[derive(Debug, Clone, Copy)]
struct ExcursionPoint {
    timestamp_ns: i64,
    high: f64,
    low: f64,
    bar_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExcursionPathPrecision {
    TickExact,
    BarApproximate,
}

impl ExcursionPathPrecision {
    fn label(self) -> &'static str {
        match self {
            Self::TickExact => "tick_exact",
            Self::BarApproximate => "bar_approximate",
        }
    }

    fn is_tick_exact(self) -> bool {
        matches!(self, Self::TickExact)
    }
}

pub(crate) fn write_replay_result(
    input: ReplayResultInput<'_>,
) -> Result<ReplayResultWriteOutcome> {
    let fills = sorted_fills(input.ledger);
    let fee_schedule = ReplayFeeSchedule::default();
    let trades = build_trade_rows(&fills, &fee_schedule);
    let initial_capital = replay_initial_capital(input.replay, input.config.replay_initial_capital);
    let trade_excursions = if input.error.is_none() {
        build_trade_excursions(
            input.replay,
            input.bar_type,
            &trades,
            input.market,
            input.config.replay_post_exit_continuation_bars,
        )?
    } else {
        None
    };
    let equity = build_equity_rows(&fills, initial_capital, input.started_at_utc, &fee_schedule);
    let mut summary = build_summary(
        &fills,
        &trades,
        &equity,
        initial_capital,
        input
            .replay
            .replay_window
            .as_ref()
            .map_or(0, |value| value.evaluation_rows_processed),
    );
    apply_excursion_summary(&mut summary, trade_excursions.as_deref());
    let fee_scenario = fee_scenario_from_summary(&fee_schedule, &summary);
    let metadata = build_metadata(&input, initial_capital, &fills);
    let status = if input.error.is_some() {
        ReplayResultStatus::Failed
    } else {
        ReplayResultStatus::Completed
    };
    let margin_config = if matches!(status, ReplayResultStatus::Completed) {
        configured_margin_config(input.config)?
    } else {
        None
    };
    let margin_analysis = margin_config
        .as_ref()
        .map(|config| {
            let margin_fills = margin_fills_for_identity(
                &fills,
                input.replay.account.id,
                input.replay.contract.id,
            );
            compute_margin_analysis(
                &margin_fills,
                initial_capital,
                input.started_at_utc,
                &fee_schedule,
                &fee_schedule.name,
                config,
            )
        })
        .transpose()?;
    if let Some(analysis) = margin_analysis.as_ref() {
        apply_margin_to_summary(&mut summary, analysis);
    }
    let liquidation_analysis = if input.config.replay_liquidation_enabled {
        let margin = margin_config
            .as_ref()
            .context("replay liquidation simulation requires replay_margin_per_contract")?;
        Some(simulate_replay_liquidation(
            &fills,
            initial_capital,
            input.started_at_utc,
            &fee_schedule,
            &fee_schedule.name,
            &ReplayLiquidationConfig {
                margin: margin.clone(),
                slippage_points: input.config.replay_liquidation_slippage_points,
            },
        )?)
    } else {
        None
    };

    let directory = input
        .config
        .replay_result_dir
        .join(safe_path_component(input.run_id));
    fs::create_dir_all(&directory)
        .with_context(|| format!("create replay result directory {}", directory.display()))?;
    let parquet_artifacts = super::result_parquet::write_replay_parquet_artifacts(
        &directory,
        &fills,
        &trades,
        &equity,
        trade_excursions.as_deref(),
        input.signal_diagnostics,
    )?;

    let document = ReplayResultDocument {
        schema_version: REPLAY_RESULT_SCHEMA_VERSION,
        result_type: "midas_replay".to_string(),
        run_id: input.run_id.to_string(),
        status,
        error: input.error.map(ToString::to_string),
        created_at_utc: input.completed_at_utc,
        started_at_utc: input.started_at_utc,
        completed_at_utc: input.completed_at_utc,
        metadata,
        summary,
        artifacts: ReplayResultArtifacts {
            result_json: "result.json".to_string(),
            trades_csv: "trades.csv".to_string(),
            fills_csv: "fills.csv".to_string(),
            equity_csv: "equity.csv".to_string(),
            trades_parquet: parquet_artifacts.trades.clone(),
            fills_parquet: parquet_artifacts.fills.clone(),
            equity_parquet: parquet_artifacts.equity.clone(),
            fee_scenarios_csv: Some("fee-scenarios.csv".to_string()),
            margin_csv: margin_analysis.as_ref().map(|_| "margin.csv".to_string()),
            trade_excursions_csv: trade_excursions
                .as_ref()
                .map(|_| "trade-excursions.csv".to_string()),
            signals_csv: input.signal_diagnostics.map(|_| "signals.csv".to_string()),
            trade_excursions_parquet: parquet_artifacts.excursions.clone(),
            signals_parquet: parquet_artifacts.signals.clone(),
            liquidation_csv: liquidation_analysis
                .as_ref()
                .map(|_| "liquidation.csv".to_string()),
        },
        active_fee_scenario: fee_schedule.name.clone(),
        fee_scenarios: vec![fee_scenario],
        margin_analysis: margin_analysis.clone(),
        trade_excursions: trade_excursions.clone(),
        liquidation_analysis: liquidation_analysis.clone(),
        sweep: None,
        ledger: input.ledger.clone(),
    };

    let result_path = directory.join("result.json");
    let fills_path = directory.join("fills.csv");
    let trades_path = directory.join("trades.csv");
    let equity_path = directory.join("equity.csv");
    write_atomic(
        &result_path,
        &serde_json::to_vec_pretty(&document).context("serialize replay result JSON")?,
    )?;
    write_atomic(&fills_path, &fills_csv(&fills))?;
    write_atomic(
        &trades_path,
        &trades_csv(&trades, trade_excursions.as_deref()),
    )?;
    write_atomic(&equity_path, &equity_csv(&equity, initial_capital))?;
    write_atomic(
        &directory.join("fee-scenarios.csv"),
        &fee_scenarios_csv(&document.fee_scenarios),
    )?;
    if let Some(analysis) = margin_analysis.as_ref() {
        write_atomic(
            &directory.join("margin.csv"),
            &margin_analysis_csv(analysis),
        )?;
    }
    if let Some(excursions) = trade_excursions.as_ref() {
        write_atomic(
            &directory.join("trade-excursions.csv"),
            &trade_excursions_csv(excursions),
        )?;
    }
    if let Some(diagnostics) = input.signal_diagnostics {
        write_atomic(&directory.join("signals.csv"), &signals_csv(diagnostics))?;
    }
    if let Some(analysis) = liquidation_analysis.as_ref() {
        write_atomic(
            &directory.join("liquidation.csv"),
            &liquidation_csv(analysis),
        )?;
    }

    Ok(ReplayResultWriteOutcome {
        result_path,
        fill_count: fills.len(),
        trade_count: trades.len(),
        required_starting_capital: margin_analysis
            .as_ref()
            .map(|analysis| analysis.required_starting_capital),
        initial_capital_sufficient: margin_analysis
            .as_ref()
            .map(|analysis| analysis.initial_capital_sufficient),
    })
}

fn build_trade_excursions(
    replay: &ReplayState,
    bar_type: BarType,
    trades: &[TradeRow],
    market: &MarketSnapshot,
    post_exit_horizon_bars: usize,
) -> Result<Option<Vec<ReplayTradeExcursion>>> {
    if trades.is_empty() {
        return Ok(None);
    }

    let frames = replay.frames_for_type(bar_type)?;
    let has_ticks = frames.iter().any(|frame| !frame.ticks.is_empty());
    let precision = if has_ticks {
        ExcursionPathPrecision::TickExact
    } else {
        ExcursionPathPrecision::BarApproximate
    };
    let mut points = Vec::new();
    if has_ticks {
        for (bar_index, frame) in frames.iter().enumerate() {
            for tick in frame.ticks.iter().copied() {
                if tick.last.is_finite() {
                    points.push(ExcursionPoint {
                        timestamp_ns: tick.ts_ns,
                        high: tick.last,
                        low: tick.last,
                        bar_index,
                    });
                }
            }
        }
    } else {
        for (bar_index, frame) in frames.iter().enumerate() {
            if !frame.bar.high.is_finite() || !frame.bar.low.is_finite() {
                continue;
            }
            points.push(ExcursionPoint {
                timestamp_ns: frame.bar.ts_ns,
                high: frame.bar.high.max(frame.bar.low),
                low: frame.bar.low.min(frame.bar.high),
                bar_index,
            });
        }
    }
    if points.is_empty() {
        return Ok(None);
    }
    points.sort_by_key(|point| point.timestamp_ns);

    let value_per_point = market
        .value_per_point
        .or(replay.market_specs.value_per_point)
        .filter(|value| value.is_finite() && *value > 0.0);
    let mut excursions = Vec::with_capacity(trades.len());
    for (index, trade) in trades.iter().enumerate() {
        excursions.push(excursion_for_trade(
            index + 1,
            trade,
            &points,
            precision,
            value_per_point,
            post_exit_horizon_bars,
        ));
    }
    Ok(Some(excursions))
}

fn excursion_for_trade(
    trade_id: usize,
    trade: &TradeRow,
    points: &[ExcursionPoint],
    precision: ExcursionPathPrecision,
    value_per_point: Option<f64>,
    post_exit_horizon_bars: usize,
) -> ReplayTradeExcursion {
    let entry_timestamp_ns = trade.entry_timestamp_ns;
    let end_timestamp_ns = trade.exit_timestamp_ns.unwrap_or(i64::MAX);
    let start_index = points.partition_point(|point| point.timestamp_ns < entry_timestamp_ns);
    let end_index = points.partition_point(|point| point.timestamp_ns <= end_timestamp_ns);
    let mut mfe_points = 0.0;
    let mut mae_points = 0.0;
    let mut mfe_price = trade.entry_price;
    let mut mae_price = trade.entry_price;
    let mut mfe_timestamp_ns = Some(entry_timestamp_ns);
    let mut mfe_index = None;

    for (offset, point) in points[start_index..end_index].iter().enumerate() {
        let favorable_points;
        let adverse_points;
        let favorable_price;
        let adverse_price;
        if trade.side.eq_ignore_ascii_case("long") {
            favorable_points = (point.high - trade.entry_price).max(0.0);
            adverse_points = (trade.entry_price - point.low).max(0.0);
            favorable_price = point.high;
            adverse_price = point.low;
        } else {
            favorable_points = (trade.entry_price - point.low).max(0.0);
            adverse_points = (point.high - trade.entry_price).max(0.0);
            favorable_price = point.low;
            adverse_price = point.high;
        }
        if favorable_points > mfe_points {
            mfe_points = favorable_points;
            mfe_price = favorable_price;
            mfe_timestamp_ns = Some(point.timestamp_ns);
            mfe_index = Some(start_index + offset);
        }
        if adverse_points > mae_points {
            mae_points = adverse_points;
            mae_price = adverse_price;
        }
    }

    let mfe_pnl = value_per_point.map(|value| mfe_points * trade.quantity * value);
    let mae_pnl = value_per_point.map(|value| mae_points * trade.quantity * value);
    let giveback = mfe_pnl.map(|mfe| mfe - trade.gross_realized_pnl);
    let mfe_capture_ratio = mfe_pnl
        .filter(|mfe| *mfe > 0.0)
        .map(|mfe| trade.gross_realized_pnl / mfe);
    let time_to_mfe_ns =
        mfe_timestamp_ns.map(|timestamp| timestamp.saturating_sub(entry_timestamp_ns));
    let time_from_mfe_to_exit_ns = trade
        .exit_timestamp_ns
        .zip(mfe_timestamp_ns)
        .map(|(exit, mfe)| exit.saturating_sub(mfe));
    let observations_to_mfe = mfe_index.map(|index| index.saturating_sub(start_index) + 1);
    let observations_from_mfe_to_exit = trade
        .exit_timestamp_ns
        .and_then(|_| mfe_index.map(|index| end_index.saturating_sub(index).saturating_sub(1)));
    let (bars_to_mfe, ticks_to_mfe) = if precision.is_tick_exact() {
        (None, observations_to_mfe)
    } else {
        (observations_to_mfe, None)
    };
    let (bars_from_mfe_to_exit, ticks_from_mfe_to_exit) = if precision.is_tick_exact() {
        (None, observations_from_mfe_to_exit)
    } else {
        (observations_from_mfe_to_exit, None)
    };
    let (
        post_exit_favorable_points,
        post_exit_favorable_price,
        post_exit_favorable_pnl,
        post_exit_favorable_timestamp_ns,
        post_exit_bars_observed,
    ) = post_exit_continuation(trade, points, post_exit_horizon_bars, value_per_point);

    ReplayTradeExcursion {
        trade_id,
        account_id: trade.account_id,
        contract_id: trade.contract_id,
        contract_name: trade.contract_name.clone(),
        side: trade.side.clone(),
        quantity: trade.quantity,
        entry_timestamp_ns,
        entry_price: trade.entry_price,
        exit_timestamp_ns: trade.exit_timestamp_ns,
        exit_price: trade.exit_price,
        realized_gross_pnl: trade.gross_realized_pnl,
        realized_net_pnl: trade.net_realized_pnl,
        exit_reason: trade.exit_reason.clone(),
        mfe_points,
        mae_points,
        mfe_price,
        mae_price,
        mfe_pnl,
        mae_pnl,
        giveback,
        mfe_capture_ratio,
        mfe_timestamp_ns,
        time_to_mfe_ns,
        time_from_mfe_to_exit_ns,
        bars_to_mfe,
        ticks_to_mfe,
        bars_from_mfe_to_exit,
        ticks_from_mfe_to_exit,
        post_exit_favorable_points,
        post_exit_favorable_price,
        post_exit_favorable_pnl,
        post_exit_favorable_timestamp_ns,
        post_exit_bars_observed,
        path_precision: precision.label().to_string(),
    }
}

fn post_exit_continuation(
    trade: &TradeRow,
    points: &[ExcursionPoint],
    horizon_bars: usize,
    value_per_point: Option<f64>,
) -> (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<i64>,
    Option<usize>,
) {
    let Some(exit_timestamp_ns) = trade.exit_timestamp_ns else {
        return (None, None, None, None, None);
    };
    let Some(exit_price) = trade.exit_price.filter(|price| price.is_finite()) else {
        return (None, None, None, None, None);
    };
    if horizon_bars == 0 {
        return (None, None, None, None, None);
    }

    let start_index = points.partition_point(|point| point.timestamp_ns <= exit_timestamp_ns);
    let Some(first_point) = points.get(start_index) else {
        return (None, None, None, None, None);
    };
    let first_bar_index = first_point.bar_index;
    let last_bar_index = first_bar_index.saturating_add(horizon_bars - 1);
    let mut favorable_points = 0.0;
    let mut favorable_price = exit_price;
    let mut favorable_timestamp_ns = None;
    let mut last_observed_bar_index = first_bar_index;

    for point in points.iter().skip(start_index) {
        if point.bar_index > last_bar_index {
            break;
        }
        last_observed_bar_index = point.bar_index;
        let (candidate_points, candidate_price) = if trade.side.eq_ignore_ascii_case("long") {
            ((point.high - exit_price).max(0.0), point.high)
        } else {
            ((exit_price - point.low).max(0.0), point.low)
        };
        if candidate_points > favorable_points {
            favorable_points = candidate_points;
            favorable_price = candidate_price;
            favorable_timestamp_ns = Some(point.timestamp_ns);
        }
    }

    (
        Some(favorable_points),
        Some(favorable_price),
        value_per_point.map(|value| favorable_points * trade.quantity * value),
        favorable_timestamp_ns,
        Some(
            last_observed_bar_index
                .saturating_sub(first_bar_index)
                .saturating_add(1),
        ),
    )
}

fn configured_margin_config(config: &AppConfig) -> Result<Option<ReplayMarginConfig>> {
    if config.replay_margin_per_contract <= 0.0 {
        return Ok(None);
    }
    let margin = ReplayMarginConfig {
        model: config.replay_margin_model.clone(),
        currency: config.replay_account_currency.clone(),
        margin_per_contract: config.replay_margin_per_contract,
        safety_buffer: config.replay_safety_buffer,
        safety_buffer_percent: config.replay_safety_buffer_percent,
    };
    margin.validate()?;
    Ok(Some(margin))
}

fn build_metadata(
    input: &ReplayResultInput<'_>,
    initial_capital: f64,
    fills: &[ReplayExecutionFill],
) -> ReplayResultMetadata {
    let window = input.replay.replay_window.as_ref();
    let evaluation_range = input.replay.evaluation_range;
    let mut fill_price_sources = BTreeSet::new();
    let mut execution_precisions = BTreeSet::new();
    for fill in fills {
        fill_price_sources.insert(fill_price_source_label(fill.fill_price_source).to_string());
        execution_precisions.insert(fill.execution_precision.label().to_string());
    }
    ReplayResultMetadata {
        app_version: input.config.app_version.clone(),
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: option_env!("GIT_COMMIT")
            .or(option_env!("MIDAS_GIT_COMMIT"))
            .map(ToString::to_string),
        broker: input.config.broker.label().to_string(),
        environment: input.config.env,
        run_mode: "single".to_string(),
        execution_backend: None,
        fallback_reason: None,
        strategy: input.strategy.clone(),
        path_dependence: replay_path_dependence(input.strategy),
        signal_source: input.ledger.signal_source.clone(),
        fill_model: input.ledger.fill_model,
        fill_price_sources: fill_price_sources.into_iter().collect(),
        execution_precisions: execution_precisions.into_iter().collect(),
        latency_model: input.ledger.latency_model,
        fixed_latency_ms: input.ledger.fixed_latency_ms,
        latency_seed: input.ledger.latency_seed,
        observed_latency_sample_count: input.ledger.observed_latency_sample_count,
        bar_protection_policy: input.ledger.bar_protection_policy,
        fee_model: "fee_neutral".to_string(),
        initial_capital,
        account_currency: input.config.replay_account_currency.clone(),
        margin_model: if input.config.replay_margin_per_contract > 0.0 {
            input.config.replay_margin_model.clone()
        } else {
            "not_configured".to_string()
        },
        account_id: input.replay.account.id,
        account_name: input.replay.account.name.clone(),
        contract_id: input.replay.contract.id,
        contract_name: input.replay.contract.name.clone(),
        tick_size: input
            .market
            .tick_size
            .or(input.replay.market_specs.tick_size),
        value_per_point: input
            .market
            .value_per_point
            .or(input.replay.market_specs.value_per_point),
        bar_type: input.bar_type,
        candle_mode: input.candle_mode,
        timezone: window
            .map(|value| value.input_timezone.clone())
            .unwrap_or_else(|| "UTC".to_string()),
        replay_file_path: input.config.replay_file_path.display().to_string(),
        replay_dom_file_path: input
            .config
            .replay_dom_file_path
            .as_ref()
            .map(|path| path.display().to_string()),
        replay_cache_dir: input.config.replay_cache_dir.display().to_string(),
        post_exit_continuation_horizon_bars: input.config.replay_post_exit_continuation_bars,
        dataset_view: window.map(|value| value.preset.clone()),
        evaluation_start_utc: window
            .map(|value| value.evaluation_start)
            .or_else(|| evaluation_range.map(|value| value.start)),
        evaluation_end_utc: window
            .map(|value| value.evaluation_end)
            .or_else(|| evaluation_range.map(|value| value.end)),
        warmup_start_utc: window.map(|value| value.warmup_start),
        market_first_timestamp_ns: input.market.bars.first().map(|bar| bar.ts_ns),
        market_last_timestamp_ns: input.market.bars.last().map(|bar| bar.ts_ns),
        warmup_rows: window.map_or(0, |value| value.warmup_rows),
        evaluation_rows_total: window.map_or(0, |value| value.evaluation_rows_total),
        evaluation_rows_processed: window.map_or(input.market.live_bars, |value| {
            value.evaluation_rows_processed
        }),
        signal_diagnostics_enabled: input.signal_diagnostics.is_some(),
        signal_diagnostic_count: input.signal_diagnostics.map_or(0, |rows| rows.len()),
    }
}

fn replay_path_dependence(strategy: &ExecutionStrategyConfig) -> ReplayPathDependence {
    if strategy.kind == StrategyKind::Native {
        ReplayPathDependence {
            fee_path_independent: true,
            account_state_dependent: false,
            reasons: Vec::new(),
        }
    } else {
        ReplayPathDependence {
            fee_path_independent: false,
            account_state_dependent: true,
            reasons: vec![format!(
                "{} strategy inputs are not proven independent of fees, equity, or margin",
                strategy.kind.label()
            )],
        }
    }
}

fn replay_initial_capital(replay: &ReplayState, fallback: f64) -> f64 {
    ["startingBalance", "balance", "netLiq"]
        .iter()
        .find_map(|key| replay.account.raw.get(*key).and_then(json_f64))
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(fallback)
}

fn json_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|number| number as f64))
        .or_else(|| value.as_u64().map(|number| number as f64))
}

fn sorted_fills(ledger: &ReplayExecutionLedgerSnapshot) -> Vec<ReplayExecutionFill> {
    let mut fills = ledger.fills.clone();
    fills.sort_by_key(|fill| (fill.fill_timestamp_ns, fill.sequence));
    fills
}

fn margin_fills_for_identity(
    fills: &[ReplayExecutionFill],
    account_id: i64,
    contract_id: i64,
) -> Vec<ReplayExecutionFill> {
    fills
        .iter()
        .filter(|fill| fill.account_id == account_id && fill.contract_id == contract_id)
        .cloned()
        .collect()
}

fn margin_fills_for_result(
    document: &ReplayResultDocument,
    fills: &[ReplayExecutionFill],
) -> Vec<ReplayExecutionFill> {
    margin_fills_for_identity(
        fills,
        document.metadata.account_id,
        document.metadata.contract_id,
    )
}

fn build_trade_rows(
    fills: &[ReplayExecutionFill],
    fee_schedule: &ReplayFeeSchedule,
) -> Vec<TradeRow> {
    let mut open: BTreeMap<PositionKey, OpenTrade> = BTreeMap::new();
    let mut completed = Vec::new();

    for fill in fills {
        let quantity = fill.quantity.abs();
        if quantity <= f64::EPSILON {
            continue;
        }
        let fill_fee = fee_schedule.fee_for_quantity(quantity);
        let mut remaining_fee = fill_fee;
        let is_buy = fill.side.eq_ignore_ascii_case("buy");
        let incoming_side = if is_buy { "long" } else { "short" };
        let signed = if is_buy { quantity } else { -quantity };
        let key = PositionKey {
            account_id: fill.account_id,
            contract_id: fill.contract_id,
        };
        let mut remaining = quantity;
        if let Some(current) = open.get_mut(&key) {
            let current_signed = if current.side == "long" {
                current.quantity
            } else {
                -current.quantity
            };
            if current_signed.signum() == signed.signum() {
                let next_quantity = current.quantity + quantity;
                current.entry_price = (current.entry_price * current.quantity
                    + fill.price * quantity)
                    / next_quantity.max(f64::EPSILON);
                current.quantity = next_quantity;
                current.fill_count += 1;
                current.fees += fill_fee;
                remaining_fee = 0.0;
                current
                    .execution_precisions
                    .insert(fill.execution_precision.label().to_string());
                remaining = 0.0;
            }
        }

        if remaining > f64::EPSILON {
            if let Some(mut current) = open.remove(&key) {
                let close_quantity = current.quantity.min(remaining);
                let close_fee = if quantity > 0.0 {
                    fill_fee * close_quantity / quantity
                } else {
                    0.0
                };
                let points = if current.side == "long" {
                    fill.price - current.entry_price
                } else {
                    current.entry_price - fill.price
                };
                let realized = fill
                    .value_per_point
                    .map(|value_per_point| points * close_quantity * value_per_point)
                    .or(fill.gross_realized_pnl_delta)
                    .unwrap_or_default();
                current.quantity -= close_quantity;
                current.exit_timestamp_ns = Some(fill.fill_timestamp_ns);
                current.exit_price = Some(fill.price);
                current.gross_realized_pnl += realized;
                current.fees += close_fee;
                current.exit_reason = fill.exit_reason.clone();
                current.fill_count += 1;
                current
                    .execution_precisions
                    .insert(fill.execution_precision.label().to_string());
                remaining -= close_quantity;
                remaining_fee -= close_fee;
                if current.quantity <= f64::EPSILON {
                    // Preserve the quantity represented by this completed
                    // trade row. The open position is decremented above, so
                    // leaving the zero remainder here would make closed
                    // trades (and their excursion PnL) report quantity zero.
                    current.quantity = close_quantity;
                    completed.push(trade_row_from_open(current));
                } else {
                    open.insert(key, current);
                }
            }
        }

        if remaining > f64::EPSILON {
            open.insert(
                key,
                OpenTrade {
                    account_id: fill.account_id,
                    contract_id: fill.contract_id,
                    contract_name: fill.contract_name.clone(),
                    side: incoming_side.to_string(),
                    quantity: remaining,
                    entry_timestamp_ns: fill.fill_timestamp_ns,
                    entry_price: fill.price,
                    exit_timestamp_ns: None,
                    exit_price: None,
                    gross_realized_pnl: 0.0,
                    fees: remaining_fee,
                    exit_reason: None,
                    fill_count: 1,
                    execution_precisions: [fill.execution_precision.label().to_string()]
                        .into_iter()
                        .collect(),
                },
            );
        }
    }

    completed.extend(open.into_values().map(trade_row_from_open));
    completed.sort_by_key(|trade| {
        (
            trade.entry_timestamp_ns,
            trade.exit_timestamp_ns.unwrap_or(i64::MAX),
            trade.account_id,
            trade.contract_id,
        )
    });
    completed
        .into_iter()
        .enumerate()
        .map(|(_, trade)| trade)
        .collect()
}

fn trade_row_from_open(open: OpenTrade) -> TradeRow {
    let execution_precision = if open.execution_precisions.len() == 1 {
        open.execution_precisions
            .into_iter()
            .next()
            .unwrap_or_else(|| "bar approximate".to_string())
    } else {
        "mixed".to_string()
    };
    TradeRow {
        account_id: open.account_id,
        contract_id: open.contract_id,
        contract_name: open.contract_name,
        side: open.side,
        quantity: open.quantity,
        entry_timestamp_ns: open.entry_timestamp_ns,
        entry_price: open.entry_price,
        exit_timestamp_ns: open.exit_timestamp_ns,
        exit_price: open.exit_price,
        gross_realized_pnl: open.gross_realized_pnl,
        fees: open.fees,
        net_realized_pnl: open.gross_realized_pnl - open.fees,
        exit_reason: open.exit_reason,
        fill_count: open.fill_count,
        execution_precision,
    }
}

fn build_equity_rows(
    fills: &[ReplayExecutionFill],
    initial_capital: f64,
    started_at_utc: DateTime<Utc>,
    fee_schedule: &ReplayFeeSchedule,
) -> Vec<EquityRow> {
    let mut positions: BTreeMap<PositionKey, f64> = BTreeMap::new();
    let mut cumulative = 0.0;
    let mut cumulative_fees = 0.0;
    let mut rows = vec![EquityRow {
        timestamp_ns: started_at_utc.timestamp_nanos_opt().unwrap_or_default(),
        initial_capital,
        equity: initial_capital,
        cumulative_gross_realized_pnl: 0.0,
        cumulative_fees: 0.0,
        cumulative_net_pnl: 0.0,
        position_qty: 0.0,
        mark_price: None,
        execution_precision: "baseline".to_string(),
    }];
    for fill in fills {
        let quantity = fill.quantity.abs();
        let signed = if fill.side.eq_ignore_ascii_case("buy") {
            quantity
        } else {
            -quantity
        };
        let key = PositionKey {
            account_id: fill.account_id,
            contract_id: fill.contract_id,
        };
        *positions.entry(key).or_default() += signed;
        cumulative += fill.gross_realized_pnl_delta.unwrap_or_default();
        cumulative_fees += fee_schedule.fee_for_quantity(quantity);
        let cumulative_net_pnl = cumulative - cumulative_fees;
        let position_qty = positions.values().copied().sum();
        rows.push(EquityRow {
            timestamp_ns: fill.fill_timestamp_ns,
            initial_capital,
            equity: initial_capital + cumulative_net_pnl,
            cumulative_gross_realized_pnl: cumulative,
            cumulative_fees,
            cumulative_net_pnl,
            position_qty,
            mark_price: Some(fill.price),
            execution_precision: fill.execution_precision.label().to_string(),
        });
    }
    rows
}

fn build_summary(
    fills: &[ReplayExecutionFill],
    trades: &[TradeRow],
    equity: &[EquityRow],
    initial_capital: f64,
    evaluation_rows_processed: usize,
) -> ReplayResultSummary {
    let gross_pnl = fills
        .iter()
        .filter_map(|fill| fill.gross_realized_pnl_delta)
        .sum::<f64>();
    let fees = trades.iter().map(|trade| trade.fees).sum::<f64>();
    let net_pnl = gross_pnl - fees;
    let closed = trades
        .iter()
        .filter(|trade| trade.exit_timestamp_ns.is_some())
        .collect::<Vec<_>>();
    let wins = closed
        .iter()
        .filter(|trade| trade.net_realized_pnl > 0.0)
        .count();
    let losses = closed
        .iter()
        .filter(|trade| trade.net_realized_pnl < 0.0)
        .count();
    let gross_wins = closed
        .iter()
        .filter(|trade| trade.net_realized_pnl > 0.0)
        .map(|trade| trade.net_realized_pnl)
        .sum::<f64>();
    let gross_losses = closed
        .iter()
        .filter(|trade| trade.net_realized_pnl < 0.0)
        .map(|trade| trade.net_realized_pnl.abs())
        .sum::<f64>();
    let mut peak = initial_capital;
    let mut max_drawdown: f64 = 0.0;
    for row in equity {
        peak = peak.max(row.equity);
        max_drawdown = max_drawdown.max(peak - row.equity);
    }
    let mut precision = BTreeSet::new();
    let mut exit_reason_counts = BTreeMap::new();
    for fill in fills {
        precision.insert(fill.execution_precision.label().to_string());
        if let Some(reason) = fill.exit_reason.as_deref() {
            *exit_reason_counts.entry(reason.to_string()).or_insert(0) += 1;
        }
    }
    let ending_equity = initial_capital + net_pnl;
    ReplayResultSummary {
        initial_capital,
        ending_equity,
        gross_pnl,
        net_pnl,
        fees,
        return_on_initial_capital_pct: (initial_capital > 0.0)
            .then_some(net_pnl / initial_capital * 100.0),
        fill_count: fills.len(),
        trade_count: trades.len(),
        closed_trade_count: closed.len(),
        win_count: wins,
        loss_count: losses,
        win_rate_pct: (!closed.is_empty()).then_some(wins as f64 / closed.len() as f64 * 100.0),
        profit_factor: (gross_losses > 0.0).then_some(gross_wins / gross_losses),
        max_drawdown,
        max_drawdown_pct: (initial_capital > 0.0).then_some(max_drawdown / initial_capital * 100.0),
        max_open_position: equity
            .iter()
            .map(|row| row.position_qty.abs())
            .fold(0.0, f64::max),
        precision: precision.into_iter().collect(),
        exit_reason_counts,
        evaluation_rows_processed,
        required_starting_capital: None,
        peak_margin_requirement: None,
        minimum_equity_buffer_over_margin: None,
        initial_capital_sufficient: None,
        average_mfe_pnl: None,
        average_mae_pnl: None,
        average_giveback: None,
        median_giveback: None,
        largest_giveback: None,
        average_mfe_capture_ratio: None,
        average_post_exit_favorable_pnl: None,
        largest_post_exit_favorable_pnl: None,
    }
}

fn fee_scenario_from_summary(
    schedule: &ReplayFeeSchedule,
    summary: &ReplayResultSummary,
) -> ReplayFeeScenario {
    ReplayFeeScenario {
        schedule: schedule.clone(),
        fees: summary.fees,
        gross_pnl: summary.gross_pnl,
        net_pnl: summary.net_pnl,
        ending_equity: summary.ending_equity,
        return_on_initial_capital_pct: summary.return_on_initial_capital_pct,
        max_drawdown: summary.max_drawdown,
        max_drawdown_pct: summary.max_drawdown_pct,
        profit_factor: summary.profit_factor,
    }
}

fn fee_scenarios_csv(scenarios: &[ReplayFeeScenario]) -> Vec<u8> {
    let mut output = String::from(
        "name,currency,commission_per_contract,exchange_per_contract,clearing_per_contract,regulatory_per_contract,misc_per_contract,total_per_contract,fees,gross_pnl,net_pnl,ending_equity,return_on_initial_capital_pct,max_drawdown,max_drawdown_pct,profit_factor\n",
    );
    for scenario in scenarios {
        let schedule = &scenario.schedule;
        let row = [
            schedule.name.clone(),
            schedule.currency.clone(),
            schedule.commission_per_contract.to_string(),
            schedule.exchange_per_contract.to_string(),
            schedule.clearing_per_contract.to_string(),
            schedule.regulatory_per_contract.to_string(),
            schedule.misc_per_contract.to_string(),
            schedule.total_per_contract().to_string(),
            scenario.fees.to_string(),
            scenario.gross_pnl.to_string(),
            scenario.net_pnl.to_string(),
            scenario.ending_equity.to_string(),
            optional(scenario.return_on_initial_capital_pct),
            scenario.max_drawdown.to_string(),
            optional(scenario.max_drawdown_pct),
            optional(scenario.profit_factor),
        ];
        push_csv_row(&mut output, &row);
    }
    output.into_bytes()
}

fn apply_excursion_summary(
    summary: &mut ReplayResultSummary,
    excursions: Option<&[ReplayTradeExcursion]>,
) {
    let Some(excursions) = excursions else {
        return;
    };
    summary.average_mfe_pnl = average_optional(excursions.iter().map(|row| row.mfe_pnl));
    summary.average_mae_pnl = average_optional(excursions.iter().map(|row| row.mae_pnl));
    let givebacks = excursions
        .iter()
        .filter_map(|row| row.giveback)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    summary.average_giveback =
        (!givebacks.is_empty()).then(|| givebacks.iter().sum::<f64>() / givebacks.len() as f64);
    summary.largest_giveback = givebacks.iter().copied().reduce(f64::max);
    summary.median_giveback = median(givebacks);
    summary.average_mfe_capture_ratio = average_optional(
        excursions
            .iter()
            .map(|row| row.mfe_capture_ratio)
            .filter(|value| value.is_none_or(|value| value.is_finite())),
    );
    let post_exit_favorable_pnl = excursions
        .iter()
        .filter_map(|row| row.post_exit_favorable_pnl)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    summary.average_post_exit_favorable_pnl = (!post_exit_favorable_pnl.is_empty()).then(|| {
        post_exit_favorable_pnl.iter().sum::<f64>() / post_exit_favorable_pnl.len() as f64
    });
    summary.largest_post_exit_favorable_pnl =
        post_exit_favorable_pnl.iter().copied().reduce(f64::max);
}

fn average_optional(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values
        .flatten()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

fn apply_margin_to_summary(summary: &mut ReplayResultSummary, analysis: &ReplayMarginAnalysis) {
    summary.required_starting_capital = Some(analysis.required_starting_capital);
    summary.peak_margin_requirement = Some(analysis.peak_margin_requirement);
    summary.minimum_equity_buffer_over_margin = Some(analysis.minimum_equity_buffer_over_margin);
    summary.initial_capital_sufficient = Some(analysis.initial_capital_sufficient);
}

fn margin_analysis_csv(analysis: &ReplayMarginAnalysis) -> Vec<u8> {
    let mut output = String::from(
        "fee_scenario,model,currency,margin_per_contract,safety_buffer,safety_buffer_percent,initial_capital,max_open_position,peak_margin_requirement,minimum_equity_buffer_over_margin,required_starting_capital,initial_capital_sufficient,first_breach_timestamp_ns\n",
    );
    let row = [
        analysis.fee_scenario.clone(),
        analysis.model.clone(),
        analysis.currency.clone(),
        analysis.margin_per_contract.to_string(),
        analysis.safety_buffer.to_string(),
        analysis.safety_buffer_percent.to_string(),
        analysis.initial_capital.to_string(),
        analysis.max_open_position.to_string(),
        analysis.peak_margin_requirement.to_string(),
        analysis.minimum_equity_buffer_over_margin.to_string(),
        analysis.required_starting_capital.to_string(),
        analysis.initial_capital_sufficient.to_string(),
        optional(analysis.first_breach_timestamp_ns),
    ];
    push_csv_row(&mut output, &row);
    output.into_bytes()
}

fn trade_excursions_csv(excursions: &[ReplayTradeExcursion]) -> Vec<u8> {
    let mut output = String::from(
        "trade_id,account_id,contract_id,contract_name,side,quantity,entry_timestamp_ns,entry_price,exit_timestamp_ns,exit_price,realized_gross_pnl,realized_net_pnl,exit_reason,mfe_points,mae_points,mfe_price,mae_price,mfe_pnl,mae_pnl,giveback,mfe_capture_ratio,mfe_timestamp_ns,time_to_mfe_ns,time_from_mfe_to_exit_ns,bars_to_mfe,ticks_to_mfe,bars_from_mfe_to_exit,ticks_from_mfe_to_exit,post_exit_favorable_points,post_exit_favorable_price,post_exit_favorable_pnl,post_exit_favorable_timestamp_ns,post_exit_bars_observed,path_precision\n",
    );
    for excursion in excursions {
        let row = [
            excursion.trade_id.to_string(),
            excursion.account_id.to_string(),
            excursion.contract_id.to_string(),
            excursion.contract_name.clone(),
            excursion.side.clone(),
            excursion.quantity.to_string(),
            excursion.entry_timestamp_ns.to_string(),
            excursion.entry_price.to_string(),
            optional(excursion.exit_timestamp_ns),
            optional(excursion.exit_price),
            excursion.realized_gross_pnl.to_string(),
            excursion.realized_net_pnl.to_string(),
            excursion.exit_reason.clone().unwrap_or_default(),
            excursion.mfe_points.to_string(),
            excursion.mae_points.to_string(),
            excursion.mfe_price.to_string(),
            excursion.mae_price.to_string(),
            optional(excursion.mfe_pnl),
            optional(excursion.mae_pnl),
            optional(excursion.giveback),
            optional(excursion.mfe_capture_ratio),
            optional(excursion.mfe_timestamp_ns),
            optional(excursion.time_to_mfe_ns),
            optional(excursion.time_from_mfe_to_exit_ns),
            optional(excursion.bars_to_mfe),
            optional(excursion.ticks_to_mfe),
            optional(excursion.bars_from_mfe_to_exit),
            optional(excursion.ticks_from_mfe_to_exit),
            optional(excursion.post_exit_favorable_points),
            optional(excursion.post_exit_favorable_price),
            optional(excursion.post_exit_favorable_pnl),
            optional(excursion.post_exit_favorable_timestamp_ns),
            optional(excursion.post_exit_bars_observed),
            excursion.path_precision.clone(),
        ];
        push_csv_row(&mut output, &row);
    }
    output.into_bytes()
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayResultRepriceOutcome {
    pub(crate) result_path: PathBuf,
    pub(crate) scenario_name: String,
    pub(crate) fees: f64,
    pub(crate) net_pnl: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayMarginAnalysisOutcome {
    pub(crate) result_path: PathBuf,
    pub(crate) required_starting_capital: f64,
    pub(crate) initial_capital_sufficient: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayLiquidationSimulationOutcome {
    pub(crate) result_path: PathBuf,
    pub(crate) triggered: bool,
    pub(crate) event_count: usize,
    pub(crate) equity_after: f64,
}

/// Apply an accounting-only fee schedule to a completed result.
///
/// The embedded execution ledger is never changed.  The active summary and
/// human-readable trade/equity exports are regenerated from that immutable
/// ledger, while every selected schedule remains available in the result JSON
/// and `fee-scenarios.csv`.
pub(crate) fn reprice_replay_result(
    result_path: &Path,
    schedule: ReplayFeeSchedule,
) -> Result<ReplayResultRepriceOutcome> {
    schedule.validate()?;
    let bytes = fs::read(result_path)
        .with_context(|| format!("read replay result {}", result_path.display()))?;
    let mut document: ReplayResultDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse replay result {}", result_path.display()))?;
    if !matches!(document.status, ReplayResultStatus::Completed) {
        anyhow::bail!("cannot reprice a failed replay result");
    }
    if !document.ledger.fee_neutral {
        anyhow::bail!("replay result ledger is not marked fee-neutral");
    }
    let existing_margin_config =
        document
            .margin_analysis
            .as_ref()
            .map(|analysis| ReplayMarginConfig {
                model: analysis.model.clone(),
                currency: analysis.currency.clone(),
                margin_per_contract: analysis.margin_per_contract,
                safety_buffer: analysis.safety_buffer,
                safety_buffer_percent: analysis.safety_buffer_percent,
            });
    let existing_liquidation_config = document
        .liquidation_analysis
        .as_ref()
        .map(|analysis| analysis.config.clone());
    let directory = result_path
        .parent()
        .context("replay result path has no parent directory")?;
    let fills = sorted_fills(&document.ledger);
    let initial_capital = document.metadata.initial_capital;
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        anyhow::bail!("replay result has invalid initial capital");
    }
    let trades = build_trade_rows(&fills, &schedule);
    let equity = build_equity_rows(&fills, initial_capital, document.started_at_utc, &schedule);
    let mut summary = build_summary(
        &fills,
        &trades,
        &equity,
        initial_capital,
        document.summary.evaluation_rows_processed,
    );
    apply_excursion_summary(&mut summary, document.trade_excursions.as_deref());
    let scenario = fee_scenario_from_summary(&schedule, &summary);

    if document.fee_scenarios.is_empty() {
        let neutral = ReplayFeeSchedule::default();
        document
            .fee_scenarios
            .push(fee_scenario_from_summary(&neutral, &document.summary));
    }
    document
        .fee_scenarios
        .retain(|existing| existing.schedule.name != schedule.name);
    document.fee_scenarios.push(scenario.clone());
    document.schema_version = REPLAY_RESULT_SCHEMA_VERSION;
    document.active_fee_scenario = schedule.name.clone();
    document.metadata.fee_model = schedule.name.clone();
    document.summary = summary;
    document.artifacts.fee_scenarios_csv = Some("fee-scenarios.csv".to_string());

    let parquet_artifacts = super::result_parquet::write_replay_parquet_artifacts(
        directory,
        &fills,
        &trades,
        &equity,
        document.trade_excursions.as_deref(),
        None,
    )?;
    document.artifacts.trades_parquet = parquet_artifacts.trades;
    document.artifacts.fills_parquet = parquet_artifacts.fills;
    document.artifacts.equity_parquet = parquet_artifacts.equity;
    document.artifacts.trade_excursions_parquet = parquet_artifacts.excursions;

    if let Some(margin_config) = existing_margin_config {
        let margin_fills = margin_fills_for_result(&document, &fills);
        let margin_analysis = compute_margin_analysis(
            &margin_fills,
            initial_capital,
            document.started_at_utc,
            &schedule,
            &schedule.name,
            &margin_config,
        )?;
        apply_margin_to_summary(&mut document.summary, &margin_analysis);
        document.margin_analysis = Some(margin_analysis.clone());
        document.artifacts.margin_csv = Some("margin.csv".to_string());
        write_atomic(
            &directory.join("margin.csv"),
            &margin_analysis_csv(&margin_analysis),
        )?;
    }
    if let Some(liquidation_config) = existing_liquidation_config {
        let liquidation_analysis = simulate_replay_liquidation(
            &fills,
            initial_capital,
            document.started_at_utc,
            &schedule,
            &schedule.name,
            &liquidation_config,
        )?;
        document.liquidation_analysis = Some(liquidation_analysis.clone());
        document.artifacts.liquidation_csv = Some("liquidation.csv".to_string());
        write_atomic(
            &directory.join("liquidation.csv"),
            &liquidation_csv(&liquidation_analysis),
        )?;
    }

    write_atomic(
        &directory.join("trades.csv"),
        &trades_csv(&trades, document.trade_excursions.as_deref()),
    )?;
    write_atomic(
        &directory.join("equity.csv"),
        &equity_csv(&equity, initial_capital),
    )?;
    write_atomic(
        &directory.join("fee-scenarios.csv"),
        &fee_scenarios_csv(&document.fee_scenarios),
    )?;
    if let Some(excursions) = document.trade_excursions.as_ref() {
        write_atomic(
            &directory.join("trade-excursions.csv"),
            &trade_excursions_csv(excursions),
        )?;
    }
    write_atomic(
        result_path,
        &serde_json::to_vec_pretty(&document).context("serialize repriced replay result")?,
    )?;

    Ok(ReplayResultRepriceOutcome {
        result_path: result_path.to_path_buf(),
        scenario_name: schedule.name,
        fees: document.summary.fees,
        net_pnl: document.summary.net_pnl,
    })
}

/// Add or refresh the account-size overlay for a completed result without
/// replaying market data or changing the immutable execution ledger.
pub(crate) fn analyze_replay_margin(
    result_path: &Path,
    config: ReplayMarginConfig,
) -> Result<ReplayMarginAnalysisOutcome> {
    config.validate()?;
    let bytes = fs::read(result_path)
        .with_context(|| format!("read replay result {}", result_path.display()))?;
    let mut document: ReplayResultDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse replay result {}", result_path.display()))?;
    if !matches!(document.status, ReplayResultStatus::Completed) {
        anyhow::bail!("cannot analyze a failed replay result");
    }
    if !document.ledger.fee_neutral {
        anyhow::bail!("replay result ledger is not marked fee-neutral");
    }
    let directory = result_path
        .parent()
        .context("replay result path has no parent directory")?;
    let fee_schedule = if let Some(scenario) = document
        .fee_scenarios
        .iter()
        .find(|scenario| scenario.schedule.name == document.active_fee_scenario)
    {
        scenario.schedule.clone()
    } else if document.active_fee_scenario == FEE_NEUTRAL_SCENARIO_NAME {
        ReplayFeeSchedule::default()
    } else {
        anyhow::bail!(
            "active fee scenario '{}' is not present in the result",
            document.active_fee_scenario
        );
    };
    let fills = sorted_fills(&document.ledger);
    let margin_fills = margin_fills_for_result(&document, &fills);
    let analysis = compute_margin_analysis(
        &margin_fills,
        document.metadata.initial_capital,
        document.started_at_utc,
        &fee_schedule,
        &document.active_fee_scenario,
        &config,
    )?;
    apply_margin_to_summary(&mut document.summary, &analysis);
    document.metadata.margin_model = config.model.clone();
    document.metadata.account_currency = config.currency.clone();
    document.margin_analysis = Some(analysis.clone());
    document.artifacts.margin_csv = Some("margin.csv".to_string());
    write_atomic(
        &directory.join("margin.csv"),
        &margin_analysis_csv(&analysis),
    )?;
    write_atomic(
        result_path,
        &serde_json::to_vec_pretty(&document).context("serialize replay margin result")?,
    )?;
    Ok(ReplayMarginAnalysisOutcome {
        result_path: result_path.to_path_buf(),
        required_starting_capital: analysis.required_starting_capital,
        initial_capital_sufficient: analysis.initial_capital_sufficient,
    })
}

/// Run the opt-in margin liquidation overlay against a saved result.  This
/// never changes the embedded execution ledger or starts a replay worker.
pub(crate) fn simulate_replay_liquidation_result(
    result_path: &Path,
    config: ReplayLiquidationConfig,
    fee_scenario: Option<&str>,
) -> Result<ReplayLiquidationSimulationOutcome> {
    config.validate()?;
    let bytes = fs::read(result_path)
        .with_context(|| format!("read replay result {}", result_path.display()))?;
    let mut document: ReplayResultDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse replay result {}", result_path.display()))?;
    if !matches!(document.status, ReplayResultStatus::Completed) {
        anyhow::bail!("cannot simulate liquidation for a failed replay result");
    }
    let selected_name = fee_scenario.unwrap_or(&document.active_fee_scenario);
    let fee_schedule = document
        .fee_scenarios
        .iter()
        .find(|scenario| scenario.schedule.name.eq_ignore_ascii_case(selected_name))
        .map(|scenario| scenario.schedule.clone())
        .or_else(|| {
            (selected_name.eq_ignore_ascii_case(FEE_NEUTRAL_SCENARIO_NAME))
                .then(ReplayFeeSchedule::default)
        })
        .with_context(|| format!("fee scenario `{selected_name}` is not present in result"))?;
    let fills = sorted_fills(&document.ledger);
    let analysis = simulate_replay_liquidation(
        &fills,
        document.metadata.initial_capital,
        document.started_at_utc,
        &fee_schedule,
        &fee_schedule.name,
        &config,
    )?;
    let directory = result_path
        .parent()
        .context("replay result path has no parent directory")?;
    write_atomic(
        &directory.join("liquidation.csv"),
        &liquidation_csv(&analysis),
    )?;
    document.liquidation_analysis = Some(analysis.clone());
    document.artifacts.liquidation_csv = Some("liquidation.csv".to_string());
    document.schema_version = REPLAY_RESULT_SCHEMA_VERSION;
    write_atomic(
        result_path,
        &serde_json::to_vec_pretty(&document).context("serialize replay liquidation result")?,
    )?;
    Ok(ReplayLiquidationSimulationOutcome {
        result_path: result_path.to_path_buf(),
        triggered: analysis.triggered,
        event_count: analysis.event_count,
        equity_after: analysis.equity_after,
    })
}

fn fills_csv(fills: &[ReplayExecutionFill]) -> Vec<u8> {
    let mut output = String::from(
        "sequence,lifecycle_sequence,fill_id,order_id,order_strategy_id,protection_order_id,account_id,contract_id,contract_name,side,quantity,price,signal_timestamp_ns,submission_timestamp_ns,exchange_arrival_timestamp_ns,acknowledgement_timestamp_ns,fill_timestamp_ns,fill_price_source,execution_precision,exit_reason,latency_ms,tick_size,value_per_point,gross_realized_pnl_delta\n",
    );
    for fill in fills {
        let row = [
            fill.sequence.to_string(),
            optional(fill.lifecycle_sequence),
            fill.fill_id.to_string(),
            fill.order_id.to_string(),
            optional(fill.order_strategy_id),
            optional(fill.protection_order_id),
            fill.account_id.to_string(),
            fill.contract_id.to_string(),
            fill.contract_name.clone(),
            fill.side.clone(),
            fill.quantity.to_string(),
            fill.price.to_string(),
            optional(fill.signal_timestamp_ns),
            optional(fill.submission_timestamp_ns),
            optional(fill.exchange_arrival_timestamp_ns),
            optional(fill.acknowledgement_timestamp_ns),
            fill.fill_timestamp_ns.to_string(),
            fill_price_source_label(fill.fill_price_source).to_string(),
            fill.execution_precision.label().to_string(),
            fill.exit_reason.clone().unwrap_or_default(),
            fill.latency_ms.to_string(),
            optional(fill.tick_size),
            optional(fill.value_per_point),
            optional(fill.gross_realized_pnl_delta),
        ];
        push_csv_row(&mut output, &row);
    }
    output.into_bytes()
}

fn signals_csv(rows: &[ReplaySignalDiagnostic]) -> Vec<u8> {
    let mut output = String::from(SIGNAL_DIAGNOSTICS_CSV_HEADER);
    for row in rows {
        let fields = [
            row.bar_timestamp_ns.to_string(),
            row.bar_open.to_string(),
            row.bar_high.to_string(),
            row.bar_low.to_string(),
            row.bar_close.to_string(),
            optional(row.bar_index),
            row.bar_count.to_string(),
            row.strategy.clone(),
            row.execution_path.clone(),
            row.signal_timing.clone(),
            row.signal_delay_bars.to_string(),
            row.signal.clone(),
            row.raw_signal.clone(),
            row.effective_signal.clone(),
            row.raw_buy_signal.to_string(),
            row.raw_sell_signal.to_string(),
            row.effective_buy_signal.to_string(),
            row.effective_sell_signal.to_string(),
            row.current_position_qty.to_string(),
            row.effective_position_qty.to_string(),
            optional(row.target_qty),
            row.decision.clone(),
            row.gate_reason.clone(),
            row.order_action.clone().unwrap_or_default(),
            optional(row.order_qty),
            row.indicator_name.clone(),
            optional(row.previous_fast_indicator),
            optional(row.previous_slow_indicator),
            optional(row.fast_indicator),
            optional(row.slow_indicator),
            row.auxiliary_name.clone().unwrap_or_default(),
            optional(row.auxiliary_value),
            row.hold_reason.clone().unwrap_or_default(),
            row.strategy_detail.clone(),
            optional(row.fingerprint),
        ];
        push_csv_row(&mut output, &fields);
    }
    output.into_bytes()
}

pub(super) fn fill_price_source_label(source: ReplayFillPriceSource) -> &'static str {
    match source {
        ReplayFillPriceSource::LegacyReferencePrice => "legacy_reference_price",
        ReplayFillPriceSource::RawBarOpen => "raw_bar_open",
        ReplayFillPriceSource::RawBarOhlc => "raw_bar_ohlc",
        ReplayFillPriceSource::TickTradeFallback => "tick_trade_fallback",
        ReplayFillPriceSource::TickBidAsk => "tick_bid_ask",
        ReplayFillPriceSource::DomVisibleLevels => "dom_visible_levels",
        ReplayFillPriceSource::DomTopOfBook => "dom_top_of_book",
    }
}

fn trades_csv(trades: &[TradeRow], excursions: Option<&[ReplayTradeExcursion]>) -> Vec<u8> {
    let mut output = String::from(
        "trade_id,account_id,contract_id,contract_name,side,quantity,entry_timestamp_ns,entry_price,exit_timestamp_ns,exit_price,gross_realized_pnl,fees,net_realized_pnl,exit_reason,fill_count,execution_precision,mfe_points,mae_points,mfe_pnl,mae_pnl,giveback,mfe_capture_ratio,mfe_timestamp_ns,time_to_mfe_ns,time_from_mfe_to_exit_ns,bars_to_mfe,ticks_to_mfe,bars_from_mfe_to_exit,ticks_from_mfe_to_exit,post_exit_favorable_points,post_exit_favorable_price,post_exit_favorable_pnl,post_exit_favorable_timestamp_ns,post_exit_bars_observed,path_precision\n",
    );
    for (index, trade) in trades.iter().enumerate() {
        let excursion = excursions.and_then(|rows| rows.get(index));
        let row = [
            (index + 1).to_string(),
            trade.account_id.to_string(),
            trade.contract_id.to_string(),
            trade.contract_name.clone(),
            trade.side.clone(),
            trade.quantity.to_string(),
            trade.entry_timestamp_ns.to_string(),
            trade.entry_price.to_string(),
            optional(trade.exit_timestamp_ns),
            optional(trade.exit_price),
            trade.gross_realized_pnl.to_string(),
            trade.fees.to_string(),
            trade.net_realized_pnl.to_string(),
            trade.exit_reason.clone().unwrap_or_default(),
            trade.fill_count.to_string(),
            trade.execution_precision.clone(),
            excursion.map_or_else(String::new, |row| row.mfe_points.to_string()),
            excursion.map_or_else(String::new, |row| row.mae_points.to_string()),
            excursion.map_or_else(String::new, |row| optional(row.mfe_pnl)),
            excursion.map_or_else(String::new, |row| optional(row.mae_pnl)),
            excursion.map_or_else(String::new, |row| optional(row.giveback)),
            excursion.map_or_else(String::new, |row| optional(row.mfe_capture_ratio)),
            excursion.map_or_else(String::new, |row| optional(row.mfe_timestamp_ns)),
            excursion.map_or_else(String::new, |row| optional(row.time_to_mfe_ns)),
            excursion.map_or_else(String::new, |row| optional(row.time_from_mfe_to_exit_ns)),
            excursion.map_or_else(String::new, |row| optional(row.bars_to_mfe)),
            excursion.map_or_else(String::new, |row| optional(row.ticks_to_mfe)),
            excursion.map_or_else(String::new, |row| optional(row.bars_from_mfe_to_exit)),
            excursion.map_or_else(String::new, |row| optional(row.ticks_from_mfe_to_exit)),
            excursion.map_or_else(String::new, |row| optional(row.post_exit_favorable_points)),
            excursion.map_or_else(String::new, |row| optional(row.post_exit_favorable_price)),
            excursion.map_or_else(String::new, |row| optional(row.post_exit_favorable_pnl)),
            excursion.map_or_else(String::new, |row| {
                optional(row.post_exit_favorable_timestamp_ns)
            }),
            excursion.map_or_else(String::new, |row| optional(row.post_exit_bars_observed)),
            excursion.map_or_else(String::new, |row| row.path_precision.clone()),
        ];
        push_csv_row(&mut output, &row);
    }
    output.into_bytes()
}

fn equity_csv(rows: &[EquityRow], initial_capital: f64) -> Vec<u8> {
    let mut output = String::from(
        "timestamp_ns,equity,initial_capital,cumulative_gross_realized_pnl,cumulative_fees,cumulative_net_pnl,position_qty,mark_price,execution_precision\n",
    );
    for row in rows {
        let fields = [
            row.timestamp_ns.to_string(),
            row.equity.to_string(),
            initial_capital.to_string(),
            row.cumulative_gross_realized_pnl.to_string(),
            row.cumulative_fees.to_string(),
            row.cumulative_net_pnl.to_string(),
            row.position_qty.to_string(),
            optional(row.mark_price),
            row.execution_precision.clone(),
        ];
        push_csv_row(&mut output, &fields);
    }
    output.into_bytes()
}

fn push_csv_row(output: &mut String, fields: &[String]) {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&csv_escape(field));
    }
    output.push('\n');
}

fn csv_escape(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional<T: Display>(value: Option<T>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn default_active_fee_scenario() -> String {
    FEE_NEUTRAL_SCENARIO_NAME.to_string()
}

fn safe_path_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| match character {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => character,
            _ => '_',
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "replay-run".to_string()
    } else {
        sanitized
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("result artifact path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("create temporary result artifact {}", temporary.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("write temporary result artifact {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("flush temporary result artifact {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("commit result artifact {}", path.display()))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;
    use crate::broker::{
        AccountInfo, ContractSuggestion, ReplayEngineMode, ReplayExecutionPrecision,
        ReplayFillModel, ReplayFillPriceSource, ReplayLatencyModel, ReplayMarketDom,
        ReplaySignalDiagnostic, ReplayWindowSnapshot,
    };
    use crate::config::TradingEnvironment;
    use crate::tradovate::MarketSpecs;
    use crate::tradovate::replay::state::ReplayDataSource;
    use crate::tradovate::replay::ticks::ReplayTick;
    use serde_json::json;
    use std::fs;

    fn replay_state() -> ReplayState {
        ReplayState {
            evaluation_range: None,
            replay_window: Some(ReplayWindowSnapshot {
                preset: "test".to_string(),
                input_timezone: "UTC".to_string(),
                warmup_start: DateTime::from_timestamp(0, 0).unwrap(),
                evaluation_start: DateTime::from_timestamp(1, 0).unwrap(),
                evaluation_end: DateTime::from_timestamp(3, 0).unwrap(),
                warmup_rows: 1,
                evaluation_rows_total: 2,
                evaluation_rows_processed: 2,
            }),
            contract: ContractSuggestion {
                id: 99,
                name: "MESU6".to_string(),
                description: "test contract".to_string(),
                raw: json!({}),
            },
            account: AccountInfo {
                id: 1,
                name: "REPLAY".to_string(),
                raw: json!({"startingBalance": 10000.0}),
            },
            market_specs: MarketSpecs {
                session_profile: None,
                value_per_point: Some(5.0),
                tick_size: Some(0.25),
            },
            dom_updates: std::sync::Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
            data: ReplayDataSource::PriceTicks(std::sync::Arc::from(
                Vec::<ReplayTick>::new().into_boxed_slice(),
            )),
            shared_frames: None,
        }
    }

    fn replay_state_with_ticks() -> ReplayState {
        let mut replay = replay_state();
        replay.data = ReplayDataSource::PriceTicks(std::sync::Arc::from(
            vec![
                ReplayTick {
                    ts_ns: 1_000_000_000,
                    last: 100.0,
                    size: Some(1.0),
                },
                ReplayTick {
                    ts_ns: 1_500_000_000,
                    last: 103.0,
                    size: Some(1.0),
                },
                ReplayTick {
                    ts_ns: 2_000_000_000,
                    last: 98.0,
                    size: Some(1.0),
                },
                ReplayTick {
                    ts_ns: 2_500_000_000,
                    last: 104.0,
                    size: Some(1.0),
                },
            ]
            .into_boxed_slice(),
        ));
        replay
    }

    fn fill(
        id: i64,
        side: &str,
        quantity: f64,
        price: f64,
        timestamp: i64,
        realized: Option<f64>,
    ) -> ReplayExecutionFill {
        ReplayExecutionFill {
            sequence: id as u64,
            lifecycle_sequence: None,
            fill_id: id,
            order_id: id,
            order_strategy_id: None,
            protection_order_id: None,
            account_id: 1,
            contract_id: 99,
            contract_name: "MESU6".to_string(),
            side: side.to_string(),
            quantity,
            price,
            signal_timestamp_ns: None,
            submission_timestamp_ns: None,
            exchange_arrival_timestamp_ns: None,
            acknowledgement_timestamp_ns: None,
            fill_timestamp_ns: timestamp,
            fill_price_source: ReplayFillPriceSource::RawBarOpen,
            execution_precision: ReplayExecutionPrecision::BarApproximate,
            exit_reason: None,
            latency_ms: 0,
            tick_size: Some(0.25),
            value_per_point: Some(5.0),
            gross_realized_pnl_delta: realized,
        }
    }

    #[test]
    fn excursion_math_reports_tick_exact_opportunity_and_giveback() {
        let trade = TradeRow {
            account_id: 1,
            contract_id: 99,
            contract_name: "MESU6".to_string(),
            side: "long".to_string(),
            quantity: 1.0,
            entry_timestamp_ns: 10,
            entry_price: 100.0,
            exit_timestamp_ns: Some(30),
            exit_price: Some(101.6),
            gross_realized_pnl: 8.0,
            fees: 0.0,
            net_realized_pnl: 8.0,
            exit_reason: None,
            fill_count: 2,
            execution_precision: "tick_exact".to_string(),
        };
        let points = [
            ExcursionPoint {
                timestamp_ns: 10,
                high: 100.0,
                low: 100.0,
                bar_index: 0,
            },
            ExcursionPoint {
                timestamp_ns: 20,
                high: 103.0,
                low: 99.0,
                bar_index: 1,
            },
            ExcursionPoint {
                timestamp_ns: 30,
                high: 102.0,
                low: 98.0,
                bar_index: 2,
            },
        ];

        let excursion = excursion_for_trade(
            7,
            &trade,
            &points,
            ExcursionPathPrecision::TickExact,
            Some(5.0),
            0,
        );

        assert_eq!(excursion.trade_id, 7);
        assert_eq!(excursion.mfe_points, 3.0);
        assert_eq!(excursion.mae_points, 2.0);
        assert_eq!(excursion.mfe_price, 103.0);
        assert_eq!(excursion.mae_price, 98.0);
        assert_eq!(excursion.mfe_pnl, Some(15.0));
        assert_eq!(excursion.mae_pnl, Some(10.0));
        assert_eq!(excursion.giveback, Some(7.0));
        assert_eq!(excursion.mfe_capture_ratio, Some(8.0 / 15.0));
        assert_eq!(excursion.mfe_timestamp_ns, Some(20));
        assert_eq!(excursion.time_to_mfe_ns, Some(10));
        assert_eq!(excursion.time_from_mfe_to_exit_ns, Some(10));
        assert_eq!(excursion.ticks_to_mfe, Some(2));
        assert_eq!(excursion.ticks_from_mfe_to_exit, Some(1));
        assert_eq!(excursion.bars_to_mfe, None);
        assert_eq!(excursion.path_precision, "tick_exact");
    }

    #[test]
    fn excursion_math_marks_bar_path_as_approximate_and_handles_shorts() {
        let trade = TradeRow {
            account_id: 1,
            contract_id: 99,
            contract_name: "MESU6".to_string(),
            side: "short".to_string(),
            quantity: 1.0,
            entry_timestamp_ns: 10,
            entry_price: 100.0,
            exit_timestamp_ns: Some(30),
            exit_price: Some(97.0),
            gross_realized_pnl: 15.0,
            fees: 0.0,
            net_realized_pnl: 15.0,
            exit_reason: None,
            fill_count: 2,
            execution_precision: "bar_approximate".to_string(),
        };
        let points = [
            ExcursionPoint {
                timestamp_ns: 10,
                high: 100.0,
                low: 100.0,
                bar_index: 0,
            },
            ExcursionPoint {
                timestamp_ns: 20,
                high: 102.0,
                low: 96.0,
                bar_index: 1,
            },
            ExcursionPoint {
                timestamp_ns: 30,
                high: 99.0,
                low: 97.0,
                bar_index: 2,
            },
        ];

        let excursion = excursion_for_trade(
            8,
            &trade,
            &points,
            ExcursionPathPrecision::BarApproximate,
            Some(5.0),
            0,
        );

        assert_eq!(excursion.mfe_points, 4.0);
        assert_eq!(excursion.mae_points, 2.0);
        assert_eq!(excursion.mfe_price, 96.0);
        assert_eq!(excursion.mae_price, 102.0);
        assert_eq!(excursion.mfe_pnl, Some(20.0));
        assert_eq!(excursion.mae_pnl, Some(10.0));
        assert_eq!(excursion.giveback, Some(5.0));
        assert_eq!(excursion.bars_to_mfe, Some(2));
        assert_eq!(excursion.bars_from_mfe_to_exit, Some(1));
        assert_eq!(excursion.ticks_to_mfe, None);
        assert_eq!(excursion.path_precision, "bar_approximate");
    }

    #[test]
    fn replay_result_capital_prefers_account_value_then_config_fallback() {
        let replay = replay_state();
        assert_eq!(replay_initial_capital(&replay, 25_000.0), 10_000.0);

        let mut missing_account_value = replay;
        missing_account_value.account.raw = json!({});
        assert_eq!(
            replay_initial_capital(&missing_account_value, 25_000.0),
            25_000.0
        );
    }

    #[test]
    fn result_writer_saves_versioned_json_and_csv_sidecars() {
        let root = std::env::temp_dir().join(format!(
            "trader-replay-results-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = AppConfig::default();
        config.broker = crate::broker::BrokerKind::Tradovate;
        config.env = TradingEnvironment::Sim;
        config.replay_result_dir = root.clone();
        config.replay_initial_capital = 10_000.0;
        config.replay_margin_per_contract = 1_000.0;
        config.replay_safety_buffer = 100.0;
        config.replay_liquidation_enabled = true;
        config.replay_liquidation_slippage_points = 0.25;
        config.replay_post_exit_continuation_bars = 2;
        config.replay_signal_diagnostics = true;
        let mut ledger = ReplayExecutionLedgerSnapshot::default();
        ledger.schema_version = 3;
        ledger.engine_mode = ReplayEngineMode::Deterministic;
        ledger.fill_model = ReplayFillModel::RawBarOpen;
        ledger.latency_model = ReplayLatencyModel::Fixed;
        ledger.signal_source = "OHLC 1 Min".to_string();
        ledger.fills = vec![
            fill(1, "Buy", 1.0, 100.0, 1_000_000_000, None),
            fill(2, "Sell", 1.0, 101.0, 2_000_000_000, Some(5.0)),
        ];
        ledger.gross_realized_pnl = 5.0;
        let strategy = ExecutionStrategyConfig::default();
        let market = MarketSnapshot {
            contract_id: Some(99),
            contract_name: Some("MESU6".to_string()),
            candle_mode: CandleMode::Standard,
            bars: vec![],
            trade_markers: vec![],
            session_profile: None,
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 2,
            replay_window: None,
            status: "complete".to_string(),
        };
        let replay = replay_state_with_ticks();
        let diagnostic = ReplaySignalDiagnostic {
            bar_timestamp_ns: 1_000_000_000,
            bar_open: 100.0,
            bar_high: 102.0,
            bar_low: 99.0,
            bar_close: 101.0,
            bar_index: Some(1),
            bar_count: 2,
            strategy: "ema_cross".to_string(),
            execution_path: "guarded".to_string(),
            signal_timing: "closed bar".to_string(),
            signal_delay_bars: 1,
            signal: "Buy".to_string(),
            raw_signal: "buy".to_string(),
            effective_signal: "buy".to_string(),
            raw_buy_signal: true,
            raw_sell_signal: false,
            effective_buy_signal: true,
            effective_sell_signal: false,
            current_position_qty: 0,
            effective_position_qty: 0,
            target_qty: Some(1),
            decision: "dispatching".to_string(),
            gate_reason: "target delta passed all guarded execution gates".to_string(),
            order_action: Some("Buy".to_string()),
            order_qty: Some(1),
            indicator_name: "EMA".to_string(),
            previous_fast_indicator: Some(99.0),
            previous_slow_indicator: Some(100.0),
            fast_indicator: Some(101.0),
            slow_indicator: Some(100.5),
            auxiliary_name: None,
            auxiliary_value: None,
            hold_reason: None,
            strategy_detail: "Signal: Buy,detail".to_string(),
            fingerprint: Some(42),
        };
        let outcome = write_replay_result(ReplayResultInput {
            config: &config,
            replay: &replay,
            market: &market,
            ledger: &ledger,
            strategy: &strategy,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            run_id: "test-run",
            started_at_utc: DateTime::from_timestamp(0, 0).unwrap(),
            completed_at_utc: DateTime::from_timestamp(3, 0).unwrap(),
            error: None,
            signal_diagnostics: Some(std::slice::from_ref(&diagnostic)),
        })
        .expect("write result");

        assert_eq!(outcome.fill_count, 2);
        assert_eq!(outcome.trade_count, 1);
        let document: ReplayResultDocument =
            serde_json::from_slice(&fs::read(&outcome.result_path).expect("read result JSON"))
                .expect("parse result JSON");
        let library = load_replay_result_entries(&root);
        assert_eq!(library.entries.len(), 1);
        assert!(library.warnings.is_empty());
        assert_eq!(library.entries[0].document.run_id, "test-run");
        let loaded_signals =
            load_replay_signal_diagnostics(&library.entries[0]).expect("load signal diagnostics");
        assert_eq!(loaded_signals, vec![diagnostic.clone()]);
        assert_eq!(document.schema_version, REPLAY_RESULT_SCHEMA_VERSION);
        assert_eq!(document.summary.net_pnl, 5.0);
        assert_eq!(document.metadata.initial_capital, 10_000.0);
        assert_eq!(document.metadata.margin_model, "fixed_per_contract");
        assert_eq!(document.metadata.account_currency, "USD");
        assert_eq!(document.metadata.post_exit_continuation_horizon_bars, 2);
        assert!(document.metadata.signal_diagnostics_enabled);
        assert_eq!(document.metadata.signal_diagnostic_count, 1);
        assert!(document.metadata.path_dependence.fee_path_independent);
        assert!(document.liquidation_analysis.is_some());
        assert_eq!(document.summary.required_starting_capital, Some(1_100.0));
        assert_eq!(document.summary.initial_capital_sufficient, Some(true));
        let excursion = document
            .trade_excursions
            .as_ref()
            .and_then(|rows| rows.first())
            .expect("tick-path excursion");
        assert_eq!(excursion.path_precision, "tick_exact");
        assert_eq!(excursion.mfe_points, 3.0);
        assert_eq!(excursion.mae_points, 2.0);
        assert_eq!(excursion.mfe_pnl, Some(15.0));
        assert_eq!(excursion.mae_pnl, Some(10.0));
        assert_eq!(excursion.giveback, Some(10.0));
        assert_eq!(excursion.mfe_timestamp_ns, Some(1_500_000_000));
        assert_eq!(excursion.ticks_to_mfe, Some(2));
        assert_eq!(excursion.ticks_from_mfe_to_exit, Some(1));
        assert_eq!(excursion.post_exit_favorable_points, Some(3.0));
        assert_eq!(excursion.post_exit_favorable_price, Some(104.0));
        assert_eq!(excursion.post_exit_favorable_pnl, Some(15.0));
        assert_eq!(
            excursion.post_exit_favorable_timestamp_ns,
            Some(2_500_000_000)
        );
        assert_eq!(excursion.post_exit_bars_observed, Some(1));
        assert_eq!(document.summary.average_mfe_pnl, Some(15.0));
        assert_eq!(document.summary.average_mae_pnl, Some(10.0));
        assert_eq!(document.summary.average_giveback, Some(10.0));
        assert_eq!(document.summary.median_giveback, Some(10.0));
        assert_eq!(document.summary.largest_giveback, Some(10.0));
        assert_eq!(document.summary.average_mfe_capture_ratio, Some(5.0 / 15.0));
        assert_eq!(document.summary.average_post_exit_favorable_pnl, Some(15.0));
        assert_eq!(document.summary.largest_post_exit_favorable_pnl, Some(15.0));
        assert_eq!(outcome.required_starting_capital, Some(1_100.0));
        assert_eq!(outcome.initial_capital_sufficient, Some(true));
        assert!(document.margin_analysis.is_some());
        assert_eq!(document.active_fee_scenario, FEE_NEUTRAL_SCENARIO_NAME);
        assert_eq!(document.fee_scenarios.len(), 1);
        let result_dir = outcome.result_path.parent().expect("result directory");
        assert!(result_dir.join("fills.csv").is_file());
        assert!(result_dir.join("trades.csv").is_file());
        assert!(result_dir.join("equity.csv").is_file());
        assert!(result_dir.join("trades.parquet").is_file());
        assert!(result_dir.join("fills.parquet").is_file());
        assert!(result_dir.join("equity.parquet").is_file());
        assert!(result_dir.join("fee-scenarios.csv").is_file());
        assert!(result_dir.join("signals.csv").is_file());
        assert!(result_dir.join("signals.parquet").is_file());
        assert!(result_dir.join("liquidation.csv").is_file());
        let signals_csv = fs::read_to_string(result_dir.join("signals.csv")).expect("read signals");
        assert!(signals_csv.starts_with("bar_timestamp_ns,bar_open,bar_high"));
        assert!(signals_csv.contains("dispatching"));
        assert!(signals_csv.contains("\"Signal: Buy,detail\""));
        let excursions_csv =
            fs::read_to_string(result_dir.join("trade-excursions.csv")).expect("read excursions");
        assert!(excursions_csv.starts_with("trade_id,account_id,contract_id"));
        assert!(excursions_csv.contains("tick_exact"));
        assert_eq!(
            document.artifacts.trade_excursions_csv.as_deref(),
            Some("trade-excursions.csv")
        );
        assert_eq!(document.artifacts.margin_csv.as_deref(), Some("margin.csv"));
        assert!(result_dir.join("margin.csv").is_file());
        let mut legacy = serde_json::to_value(&document).expect("encode result");
        let legacy_object = legacy.as_object_mut().expect("result object");
        legacy_object.remove("active_fee_scenario");
        legacy_object.remove("fee_scenarios");
        legacy_object.remove("margin_analysis");
        legacy_object
            .get_mut("summary")
            .and_then(Value::as_object_mut)
            .expect("summary object")
            .retain(|key, _| {
                !matches!(
                    key.as_str(),
                    "required_starting_capital"
                        | "peak_margin_requirement"
                        | "minimum_equity_buffer_over_margin"
                        | "initial_capital_sufficient"
                        | "average_mfe_pnl"
                        | "average_mae_pnl"
                        | "average_giveback"
                        | "median_giveback"
                        | "largest_giveback"
                        | "average_mfe_capture_ratio"
                        | "average_post_exit_favorable_pnl"
                        | "largest_post_exit_favorable_pnl"
                )
            });
        legacy_object
            .get_mut("artifacts")
            .and_then(Value::as_object_mut)
            .expect("artifact object")
            .retain(|key, _| {
                key != "fee_scenarios_csv"
                    && key != "margin_csv"
                    && key != "trade_excursions_csv"
                    && key != "signals_csv"
            });
        legacy_object
            .get_mut("metadata")
            .and_then(Value::as_object_mut)
            .expect("metadata object")
            .retain(|key, _| {
                key != "signal_diagnostics_enabled" && key != "signal_diagnostic_count"
            });
        legacy_object.remove("trade_excursions");
        let legacy_document: ReplayResultDocument =
            serde_json::from_value(legacy).expect("parse v1 result");
        assert_eq!(
            legacy_document.active_fee_scenario,
            FEE_NEUTRAL_SCENARIO_NAME
        );
        assert!(legacy_document.fee_scenarios.is_empty());
        fs::remove_dir_all(root).expect("cleanup result directory");
    }

    #[test]
    fn repricing_updates_accounting_without_mutating_ledger() {
        let root = std::env::temp_dir().join(format!(
            "trader-replay-reprice-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = AppConfig::default();
        config.broker = crate::broker::BrokerKind::Tradovate;
        config.env = TradingEnvironment::Sim;
        config.replay_result_dir = root.clone();
        config.replay_margin_per_contract = 1_000.0;
        config.replay_liquidation_enabled = true;
        config.replay_post_exit_continuation_bars = 2;
        let mut ledger = ReplayExecutionLedgerSnapshot::default();
        ledger.schema_version = 3;
        ledger.engine_mode = ReplayEngineMode::Deterministic;
        ledger.fill_model = ReplayFillModel::RawBarOpen;
        ledger.signal_source = "OHLC 1 Min".to_string();
        ledger.fills = vec![
            fill(1, "Buy", 1.0, 100.0, 1_000_000_000, None),
            fill(2, "Sell", 1.0, 101.0, 2_000_000_000, Some(5.0)),
        ];
        ledger.gross_realized_pnl = 5.0;
        let strategy = ExecutionStrategyConfig::default();
        let market = MarketSnapshot {
            contract_id: Some(99),
            contract_name: Some("MESU6".to_string()),
            candle_mode: CandleMode::Standard,
            bars: vec![],
            trade_markers: vec![],
            session_profile: None,
            value_per_point: Some(5.0),
            tick_size: Some(0.25),
            history_loaded: 1,
            live_bars: 2,
            replay_window: None,
            status: "complete".to_string(),
        };
        let replay = replay_state_with_ticks();
        let written = write_replay_result(ReplayResultInput {
            config: &config,
            replay: &replay,
            market: &market,
            ledger: &ledger,
            strategy: &strategy,
            bar_type: BarType::minute(1),
            candle_mode: CandleMode::Standard,
            run_id: "reprice-test",
            started_at_utc: DateTime::from_timestamp(0, 0).unwrap(),
            completed_at_utc: DateTime::from_timestamp(3, 0).unwrap(),
            error: None,
            signal_diagnostics: None,
        })
        .expect("write result");
        let mut neutral_document: ReplayResultDocument =
            serde_json::from_slice(&fs::read(&written.result_path).expect("read neutral result"))
                .expect("parse neutral result");
        assert!(neutral_document.margin_analysis.is_some());
        assert_eq!(neutral_document.metadata.margin_model, "fixed_per_contract");
        neutral_document.metadata.execution_backend = Some("prepared_cpu".to_string());
        neutral_document.metadata.fallback_reason = Some("test fallback".to_string());
        fs::write(
            &written.result_path,
            serde_json::to_vec_pretty(&neutral_document).expect("encode backend metadata"),
        )
        .expect("write backend metadata");
        assert_eq!(
            neutral_document
                .trade_excursions
                .as_ref()
                .and_then(|rows| rows.first())
                .map(|row| row.mfe_pnl),
            Some(Some(15.0))
        );
        let margin_config = ReplayMarginConfig {
            margin_per_contract: 1_000.0,
            safety_buffer: 100.0,
            ..ReplayMarginConfig::default()
        };
        let margin = analyze_replay_margin(&written.result_path, margin_config)
            .expect("analyze replay margin");
        assert_eq!(margin.required_starting_capital, 1_100.0);
        assert!(margin.initial_capital_sufficient);
        let margin_document: ReplayResultDocument =
            serde_json::from_slice(&fs::read(&written.result_path).expect("read margin result"))
                .expect("parse margin result");
        assert_eq!(
            margin_document.summary.required_starting_capital,
            Some(1_100.0)
        );
        assert!(margin_document.artifacts.margin_csv.is_some());
        assert!(
            written
                .result_path
                .parent()
                .expect("result directory")
                .join("margin.csv")
                .is_file()
        );
        let schedule = ReplayFeeSchedule {
            name: "broker_standard".to_string(),
            commission_per_contract: 0.50,
            ..ReplayFeeSchedule::default()
        };
        let repriced = reprice_replay_result(&written.result_path, schedule).expect("reprice");
        assert_eq!(repriced.scenario_name, "broker_standard");
        assert!((repriced.fees - 1.0).abs() < f64::EPSILON);
        assert!((repriced.net_pnl - 4.0).abs() < f64::EPSILON);

        let document: ReplayResultDocument =
            serde_json::from_slice(&fs::read(&written.result_path).expect("read result"))
                .expect("parse repriced result");
        assert_eq!(document.schema_version, REPLAY_RESULT_SCHEMA_VERSION);
        assert_eq!(document.active_fee_scenario, "broker_standard");
        assert_eq!(
            document.metadata.execution_backend.as_deref(),
            Some("prepared_cpu")
        );
        assert_eq!(
            document.metadata.fallback_reason.as_deref(),
            Some("test fallback")
        );
        assert_eq!(document.fee_scenarios.len(), 2);
        assert_eq!(document.ledger.fills, ledger.fills);
        assert_eq!(
            document
                .trade_excursions
                .as_ref()
                .and_then(|rows| rows.first())
                .map(|row| row.giveback),
            Some(Some(10.0))
        );
        assert_eq!(
            document
                .trade_excursions
                .as_ref()
                .and_then(|rows| rows.first())
                .map(|row| row.post_exit_favorable_pnl),
            Some(Some(15.0))
        );
        assert_eq!(document.summary.gross_pnl, 5.0);
        assert_eq!(document.summary.fees, 1.0);
        assert_eq!(document.summary.net_pnl, 4.0);
        assert_eq!(
            document
                .margin_analysis
                .as_ref()
                .expect("repriced margin analysis")
                .fee_scenario,
            "broker_standard"
        );
        assert_eq!(
            document
                .liquidation_analysis
                .as_ref()
                .expect("repriced liquidation analysis")
                .fee_scenario,
            "broker_standard"
        );
        assert!(
            document
                .summary
                .required_starting_capital
                .expect("required starting capital")
                > 1_100.0
        );
        let result_dir = written.result_path.parent().expect("result directory");
        let fee_csv =
            fs::read_to_string(result_dir.join("fee-scenarios.csv")).expect("read fee scenarios");
        assert!(fee_csv.contains("broker_standard"));
        let equity_csv = fs::read_to_string(result_dir.join("equity.csv")).expect("read equity");
        assert!(equity_csv.contains("cumulative_net_pnl"));
        let trades_csv = fs::read_to_string(result_dir.join("trades.csv")).expect("read trades");
        assert!(trades_csv.contains("net_realized_pnl"));
        assert!(trades_csv.contains("mfe_points"));
        assert!(trades_csv.contains(",15,"));
        assert!(result_dir.join("trade-excursions.csv").is_file());
        assert!(result_dir.join("liquidation.csv").is_file());
        fs::remove_dir_all(root).expect("cleanup result directory");
    }

    #[test]
    fn csv_escape_quotes_commas_and_newlines() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_escape("plain"), "plain");
    }

    #[test]
    fn margin_overlay_filters_to_result_account_and_contract() {
        let selected = fill(1, "Buy", 1.0, 100.0, 1_000_000_000, None);
        let mut other_contract = fill(2, "Buy", 5.0, 100.0, 2_000_000_000, None);
        other_contract.contract_id = 100;
        let mut other_account = fill(3, "Buy", 5.0, 100.0, 3_000_000_000, None);
        other_account.account_id = 2;
        let filtered = margin_fills_for_identity(
            &[selected.clone(), other_contract, other_account],
            selected.account_id,
            selected.contract_id,
        );
        assert_eq!(filtered, vec![selected]);
    }
}

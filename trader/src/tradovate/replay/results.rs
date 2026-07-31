use super::state::ReplayState;
use crate::broker::{
    BarType, CandleMode, MarketSnapshot, ReplayExecutionFill, ReplayExecutionLedgerSnapshot,
    ReplayFillPriceSource,
};
use crate::config::{AppConfig, TradingEnvironment};
use crate::strategy::ExecutionStrategyConfig;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const REPLAY_RESULT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub(crate) struct ReplayResultWriteOutcome {
    pub(crate) result_path: PathBuf,
    pub(crate) fill_count: usize,
    pub(crate) trade_count: usize,
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
    /// The fee-neutral ledger is embedded so a result is self-contained even
    /// when the CSV sidecars are moved or inspected independently.
    pub(crate) ledger: ReplayExecutionLedgerSnapshot,
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
    pub(crate) strategy: ExecutionStrategyConfig,
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
    pub(crate) dataset_view: Option<String>,
    pub(crate) evaluation_start_utc: Option<DateTime<Utc>>,
    pub(crate) evaluation_end_utc: Option<DateTime<Utc>>,
    pub(crate) warmup_start_utc: Option<DateTime<Utc>>,
    pub(crate) market_first_timestamp_ns: Option<i64>,
    pub(crate) market_last_timestamp_ns: Option<i64>,
    pub(crate) warmup_rows: usize,
    pub(crate) evaluation_rows_total: usize,
    pub(crate) evaluation_rows_processed: usize,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ReplayResultArtifacts {
    pub(crate) result_json: String,
    pub(crate) trades_csv: String,
    pub(crate) fills_csv: String,
    pub(crate) equity_csv: String,
}

#[derive(Debug, Clone)]
struct TradeRow {
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
    exit_reason: Option<String>,
    fill_count: usize,
    execution_precision: String,
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
    exit_reason: Option<String>,
    fill_count: usize,
    execution_precisions: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct EquityRow {
    timestamp_ns: i64,
    equity: f64,
    cumulative_gross_realized_pnl: f64,
    position_qty: f64,
    mark_price: Option<f64>,
    execution_precision: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PositionKey {
    account_id: i64,
    contract_id: i64,
}

pub(crate) fn write_replay_result(
    input: ReplayResultInput<'_>,
) -> Result<ReplayResultWriteOutcome> {
    let fills = sorted_fills(input.ledger);
    let trades = build_trade_rows(&fills);
    let initial_capital = replay_initial_capital(input.replay);
    let equity = build_equity_rows(&fills, initial_capital, input.started_at_utc);
    let summary = build_summary(
        &fills,
        &trades,
        &equity,
        initial_capital,
        input.replay.replay_window.as_ref(),
    );
    let metadata = build_metadata(&input, initial_capital, &fills);
    let status = if input.error.is_some() {
        ReplayResultStatus::Failed
    } else {
        ReplayResultStatus::Completed
    };

    let directory = input
        .config
        .replay_result_dir
        .join(safe_path_component(input.run_id));
    fs::create_dir_all(&directory)
        .with_context(|| format!("create replay result directory {}", directory.display()))?;

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
        },
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
    write_atomic(&trades_path, &trades_csv(&trades))?;
    write_atomic(&equity_path, &equity_csv(&equity, initial_capital))?;

    Ok(ReplayResultWriteOutcome {
        result_path,
        fill_count: fills.len(),
        trade_count: trades.len(),
    })
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
        strategy: input.strategy.clone(),
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
        account_currency: "USD".to_string(),
        margin_model: "not_configured".to_string(),
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
    }
}

fn replay_initial_capital(replay: &ReplayState) -> f64 {
    ["startingBalance", "balance", "netLiq"]
        .iter()
        .find_map(|key| replay.account.raw.get(*key).and_then(json_f64))
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(100_000.0)
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

fn build_trade_rows(fills: &[ReplayExecutionFill]) -> Vec<TradeRow> {
    let mut open: BTreeMap<PositionKey, OpenTrade> = BTreeMap::new();
    let mut completed = Vec::new();

    for fill in fills {
        let quantity = fill.quantity.abs();
        if quantity <= f64::EPSILON {
            continue;
        }
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
                current
                    .execution_precisions
                    .insert(fill.execution_precision.label().to_string());
                remaining = 0.0;
            }
        }

        if remaining > f64::EPSILON {
            if let Some(mut current) = open.remove(&key) {
                let close_quantity = current.quantity.min(remaining);
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
                current.exit_reason = fill.exit_reason.clone();
                current.fill_count += 1;
                current
                    .execution_precisions
                    .insert(fill.execution_precision.label().to_string());
                remaining -= close_quantity;
                if current.quantity <= f64::EPSILON {
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
        exit_reason: open.exit_reason,
        fill_count: open.fill_count,
        execution_precision,
    }
}

fn build_equity_rows(
    fills: &[ReplayExecutionFill],
    initial_capital: f64,
    started_at_utc: DateTime<Utc>,
) -> Vec<EquityRow> {
    let mut positions: BTreeMap<PositionKey, f64> = BTreeMap::new();
    let mut cumulative = 0.0;
    let mut rows = vec![EquityRow {
        timestamp_ns: started_at_utc.timestamp_nanos_opt().unwrap_or_default(),
        equity: initial_capital,
        cumulative_gross_realized_pnl: 0.0,
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
        let position_qty = positions.values().copied().sum();
        rows.push(EquityRow {
            timestamp_ns: fill.fill_timestamp_ns,
            equity: initial_capital + cumulative,
            cumulative_gross_realized_pnl: cumulative,
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
    window: Option<&crate::broker::ReplayWindowSnapshot>,
) -> ReplayResultSummary {
    let gross_pnl = fills
        .iter()
        .filter_map(|fill| fill.gross_realized_pnl_delta)
        .sum::<f64>();
    let closed = trades
        .iter()
        .filter(|trade| trade.exit_timestamp_ns.is_some())
        .collect::<Vec<_>>();
    let wins = closed
        .iter()
        .filter(|trade| trade.gross_realized_pnl > 0.0)
        .count();
    let losses = closed
        .iter()
        .filter(|trade| trade.gross_realized_pnl < 0.0)
        .count();
    let gross_wins = closed
        .iter()
        .filter(|trade| trade.gross_realized_pnl > 0.0)
        .map(|trade| trade.gross_realized_pnl)
        .sum::<f64>();
    let gross_losses = closed
        .iter()
        .filter(|trade| trade.gross_realized_pnl < 0.0)
        .map(|trade| trade.gross_realized_pnl.abs())
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
    let ending_equity = initial_capital + gross_pnl;
    ReplayResultSummary {
        initial_capital,
        ending_equity,
        gross_pnl,
        net_pnl: gross_pnl,
        fees: 0.0,
        return_on_initial_capital_pct: (initial_capital > 0.0)
            .then_some(gross_pnl / initial_capital * 100.0),
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
        evaluation_rows_processed: window.map_or(0, |value| value.evaluation_rows_processed),
    }
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

fn fill_price_source_label(source: ReplayFillPriceSource) -> &'static str {
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

fn trades_csv(trades: &[TradeRow]) -> Vec<u8> {
    let mut output = String::from(
        "trade_id,account_id,contract_id,contract_name,side,quantity,entry_timestamp_ns,entry_price,exit_timestamp_ns,exit_price,gross_realized_pnl,exit_reason,fill_count,execution_precision\n",
    );
    for (index, trade) in trades.iter().enumerate() {
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
            trade.exit_reason.clone().unwrap_or_default(),
            trade.fill_count.to_string(),
            trade.execution_precision.clone(),
        ];
        push_csv_row(&mut output, &row);
    }
    output.into_bytes()
}

fn equity_csv(rows: &[EquityRow], initial_capital: f64) -> Vec<u8> {
    let mut output = String::from(
        "timestamp_ns,equity,initial_capital,cumulative_gross_realized_pnl,fees,position_qty,mark_price,execution_precision\n",
    );
    for row in rows {
        let fields = [
            row.timestamp_ns.to_string(),
            row.equity.to_string(),
            initial_capital.to_string(),
            row.cumulative_gross_realized_pnl.to_string(),
            "0".to_string(),
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
        ReplayWindowSnapshot,
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
        }
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
        let replay = replay_state();
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
        })
        .expect("write result");

        assert_eq!(outcome.fill_count, 2);
        assert_eq!(outcome.trade_count, 1);
        let document: ReplayResultDocument =
            serde_json::from_slice(&fs::read(&outcome.result_path).expect("read result JSON"))
                .expect("parse result JSON");
        assert_eq!(document.schema_version, REPLAY_RESULT_SCHEMA_VERSION);
        assert_eq!(document.summary.net_pnl, 5.0);
        let result_dir = outcome.result_path.parent().expect("result directory");
        assert!(result_dir.join("fills.csv").is_file());
        assert!(result_dir.join("trades.csv").is_file());
        assert!(result_dir.join("equity.csv").is_file());
        fs::remove_dir_all(root).expect("cleanup result directory");
    }

    #[test]
    fn csv_escape_quotes_commas_and_newlines() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
        assert_eq!(csv_escape("plain"), "plain");
    }
}

//! Durable Parquet sidecars for replay results.
//!
//! Result JSON remains the portable metadata/index artifact and CSV remains
//! the human-readable export.  These sidecars keep the same rows in a typed,
//! columnar format for large replays and notebook/Arrow consumers without
//! changing the immutable execution ledger.

use super::results::{EquityRow, TradeRow, fill_price_source_label};
use crate::broker::{ReplayExecutionFill, ReplaySignalDiagnostic};
use anyhow::{Context, Result};
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use std::fs::{self, File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default)]
pub(super) struct ReplayParquetArtifacts {
    pub(super) trades: Option<String>,
    pub(super) fills: Option<String>,
    pub(super) equity: Option<String>,
    pub(super) excursions: Option<String>,
    pub(super) signals: Option<String>,
}

pub(super) fn write_replay_parquet_artifacts(
    directory: &Path,
    fills: &[ReplayExecutionFill],
    trades: &[TradeRow],
    equity: &[EquityRow],
    excursions: Option<&[super::results::ReplayTradeExcursion]>,
    signals: Option<&[ReplaySignalDiagnostic]>,
) -> Result<ReplayParquetArtifacts> {
    let artifacts = ReplayParquetArtifacts {
        trades: Some("trades.parquet".to_string()),
        fills: Some("fills.parquet".to_string()),
        equity: Some("equity.parquet".to_string()),
        excursions: excursions.map(|_| "trade-excursions.parquet".to_string()),
        signals: signals.map(|_| "signals.parquet".to_string()),
    };

    write_parquet_atomic(&directory.join("trades.parquet"), |file| {
        write_batches(file, trade_schema(), trades_record_batch(trades))
    })?;
    write_parquet_atomic(&directory.join("fills.parquet"), |file| {
        write_batches(file, fill_schema(), fills_record_batch(fills))
    })?;
    write_parquet_atomic(&directory.join("equity.parquet"), |file| {
        write_batches(file, equity_schema(), equity_record_batch(equity))
    })?;
    if let Some(excursions) = excursions {
        write_parquet_atomic(&directory.join("trade-excursions.parquet"), |file| {
            write_batches(
                file,
                excursion_schema(),
                excursions_record_batch(excursions),
            )
        })?;
    }
    if let Some(signals) = signals {
        write_parquet_atomic(&directory.join("signals.parquet"), |file| {
            write_batches(file, signal_schema(), signals_record_batch(signals))
        })?;
    }

    Ok(artifacts)
}

fn write_batches(
    file: File,
    schema: Arc<Schema>,
    batch: Result<RecordBatch, anyhow::Error>,
) -> Result<()> {
    write_record_batch(file, schema, batch?)
}

/// Write one typed Arrow batch through the same atomic/compressed writer used
/// by replay result sidecars. Sweep summaries reuse this to keep one Parquet
/// encoding policy across single runs and large sweeps.
pub(super) fn write_parquet_record_batch(path: &Path, batch: RecordBatch) -> Result<()> {
    write_parquet_atomic(path, |file| {
        let schema = batch.schema();
        write_record_batch(file, schema, batch)
    })
}

fn write_record_batch(file: File, schema: Arc<Schema>, batch: RecordBatch) -> Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_size(65_536)
        .set_write_batch_size(8_192)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub(super) fn write_parquet_atomic<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(File) -> Result<()>,
{
    let parent = path
        .parent()
        .context("result Parquet path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact.parquet");
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let result = (|| -> Result<()> {
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .with_context(|| format!("create temporary result artifact {}", temporary.display()))?;
        writer(file)?;
        fs::rename(&temporary, path)
            .with_context(|| format!("commit result artifact {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn fill_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("sequence", DataType::Int64, false),
        Field::new("lifecycle_sequence", DataType::Int64, true),
        Field::new("fill_id", DataType::Int64, false),
        Field::new("order_id", DataType::Int64, false),
        Field::new("order_strategy_id", DataType::Int64, true),
        Field::new("protection_order_id", DataType::Int64, true),
        Field::new("account_id", DataType::Int64, false),
        Field::new("contract_id", DataType::Int64, false),
        Field::new("contract_name", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("signal_timestamp_ns", DataType::Int64, true),
        Field::new("submission_timestamp_ns", DataType::Int64, true),
        Field::new("exchange_arrival_timestamp_ns", DataType::Int64, true),
        Field::new("acknowledgement_timestamp_ns", DataType::Int64, true),
        Field::new("fill_timestamp_ns", DataType::Int64, false),
        Field::new("fill_price_source", DataType::Utf8, false),
        Field::new("execution_precision", DataType::Utf8, false),
        Field::new("exit_reason", DataType::Utf8, true),
        Field::new("latency_ms", DataType::Int64, false),
        Field::new("tick_size", DataType::Float64, true),
        Field::new("value_per_point", DataType::Float64, true),
        Field::new("gross_realized_pnl_delta", DataType::Float64, true),
    ]))
}

fn fills_record_batch(rows: &[ReplayExecutionFill]) -> Result<RecordBatch> {
    let schema = fill_schema();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.sequence as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.lifecycle_sequence.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.fill_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.order_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.order_strategy_id)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.protection_order_id)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.account_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.contract_id).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.contract_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|row| row.side.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.quantity).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.price).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.signal_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.submission_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.exchange_arrival_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.acknowledgement_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.fill_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| fill_price_source_label(row.fill_price_source))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.execution_precision.label())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.exit_reason.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.latency_ms as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.tick_size).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.value_per_point)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.gross_realized_pnl_delta)
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build fills Parquet record batch")
}

fn trade_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("trade_id", DataType::Int64, false),
        Field::new("account_id", DataType::Int64, false),
        Field::new("contract_id", DataType::Int64, false),
        Field::new("contract_name", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("entry_timestamp_ns", DataType::Int64, false),
        Field::new("entry_price", DataType::Float64, false),
        Field::new("exit_timestamp_ns", DataType::Int64, true),
        Field::new("exit_price", DataType::Float64, true),
        Field::new("gross_realized_pnl", DataType::Float64, false),
        Field::new("fees", DataType::Float64, false),
        Field::new("net_realized_pnl", DataType::Float64, false),
        Field::new("exit_reason", DataType::Utf8, true),
        Field::new("fill_count", DataType::Int64, false),
        Field::new("execution_precision", DataType::Utf8, false),
    ]))
}

fn trades_record_batch(rows: &[TradeRow]) -> Result<RecordBatch> {
    let schema = trade_schema();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            (0..rows.len())
                .map(|value| value as i64 + 1)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.account_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.contract_id).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.contract_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|row| row.side.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.quantity).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.entry_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.entry_price).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.exit_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.exit_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.gross_realized_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.fees).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.net_realized_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.exit_reason.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.fill_count as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.execution_precision.clone())
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build trades Parquet record batch")
}

fn equity_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_ns", DataType::Int64, false),
        Field::new("equity", DataType::Float64, false),
        Field::new("initial_capital", DataType::Float64, false),
        Field::new("cumulative_gross_realized_pnl", DataType::Float64, false),
        Field::new("cumulative_fees", DataType::Float64, false),
        Field::new("cumulative_net_pnl", DataType::Float64, false),
        Field::new("position_qty", DataType::Float64, false),
        Field::new("mark_price", DataType::Float64, true),
        Field::new("execution_precision", DataType::Utf8, false),
    ]))
}

fn equity_record_batch(rows: &[EquityRow]) -> Result<RecordBatch> {
    let schema = equity_schema();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.timestamp_ns).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.equity).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.initial_capital)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.cumulative_gross_realized_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.cumulative_fees)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.cumulative_net_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.position_qty).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mark_price).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.execution_precision.clone())
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build equity Parquet record batch")
}

fn excursion_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("trade_id", DataType::Int64, false),
        Field::new("account_id", DataType::Int64, false),
        Field::new("contract_id", DataType::Int64, false),
        Field::new("contract_name", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("entry_timestamp_ns", DataType::Int64, false),
        Field::new("entry_price", DataType::Float64, false),
        Field::new("exit_timestamp_ns", DataType::Int64, true),
        Field::new("exit_price", DataType::Float64, true),
        Field::new("realized_gross_pnl", DataType::Float64, false),
        Field::new("realized_net_pnl", DataType::Float64, false),
        Field::new("exit_reason", DataType::Utf8, true),
        Field::new("mfe_points", DataType::Float64, false),
        Field::new("mae_points", DataType::Float64, false),
        Field::new("mfe_price", DataType::Float64, false),
        Field::new("mae_price", DataType::Float64, false),
        Field::new("mfe_pnl", DataType::Float64, true),
        Field::new("mae_pnl", DataType::Float64, true),
        Field::new("giveback", DataType::Float64, true),
        Field::new("mfe_capture_ratio", DataType::Float64, true),
        Field::new("mfe_timestamp_ns", DataType::Int64, true),
        Field::new("time_to_mfe_ns", DataType::Int64, true),
        Field::new("time_from_mfe_to_exit_ns", DataType::Int64, true),
        Field::new("bars_to_mfe", DataType::Int64, true),
        Field::new("ticks_to_mfe", DataType::Int64, true),
        Field::new("bars_from_mfe_to_exit", DataType::Int64, true),
        Field::new("ticks_from_mfe_to_exit", DataType::Int64, true),
        Field::new("post_exit_favorable_points", DataType::Float64, true),
        Field::new("post_exit_favorable_price", DataType::Float64, true),
        Field::new("post_exit_favorable_pnl", DataType::Float64, true),
        Field::new("post_exit_favorable_timestamp_ns", DataType::Int64, true),
        Field::new("post_exit_bars_observed", DataType::Int64, true),
        Field::new("path_precision", DataType::Utf8, false),
    ]))
}

fn excursions_record_batch(rows: &[super::results::ReplayTradeExcursion]) -> Result<RecordBatch> {
    let schema = excursion_schema();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.trade_id as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.account_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.contract_id).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.contract_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|row| row.side.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.quantity).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.entry_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.entry_price).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.exit_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.exit_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.realized_gross_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.realized_net_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.exit_reason.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mfe_points).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mae_points).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mfe_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mae_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mfe_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.mae_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.giveback).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.mfe_capture_ratio)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.mfe_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.time_to_mfe_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.time_from_mfe_to_exit_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.bars_to_mfe.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.ticks_to_mfe.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.bars_from_mfe_to_exit.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.ticks_from_mfe_to_exit.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.post_exit_favorable_points)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.post_exit_favorable_price)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.post_exit_favorable_pnl)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.post_exit_favorable_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.post_exit_bars_observed.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.path_precision.clone())
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build trade excursions Parquet record batch")
}

fn signal_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("bar_timestamp_ns", DataType::Int64, false),
        Field::new("bar_open", DataType::Float64, false),
        Field::new("bar_high", DataType::Float64, false),
        Field::new("bar_low", DataType::Float64, false),
        Field::new("bar_close", DataType::Float64, false),
        Field::new("bar_index", DataType::Int64, true),
        Field::new("bar_count", DataType::Int64, false),
        Field::new("strategy", DataType::Utf8, false),
        Field::new("execution_path", DataType::Utf8, false),
        Field::new("signal_timing", DataType::Utf8, false),
        Field::new("signal_delay_bars", DataType::Int64, false),
        Field::new("signal", DataType::Utf8, false),
        Field::new("raw_signal", DataType::Utf8, false),
        Field::new("effective_signal", DataType::Utf8, false),
        Field::new("raw_buy_signal", DataType::Boolean, false),
        Field::new("raw_sell_signal", DataType::Boolean, false),
        Field::new("effective_buy_signal", DataType::Boolean, false),
        Field::new("effective_sell_signal", DataType::Boolean, false),
        Field::new("current_position_qty", DataType::Int32, false),
        Field::new("effective_position_qty", DataType::Int32, false),
        Field::new("target_qty", DataType::Int32, true),
        Field::new("decision", DataType::Utf8, false),
        Field::new("gate_reason", DataType::Utf8, false),
        Field::new("order_action", DataType::Utf8, true),
        Field::new("order_qty", DataType::Int32, true),
        Field::new("indicator_name", DataType::Utf8, false),
        Field::new("previous_fast_indicator", DataType::Float64, true),
        Field::new("previous_slow_indicator", DataType::Float64, true),
        Field::new("fast_indicator", DataType::Float64, true),
        Field::new("slow_indicator", DataType::Float64, true),
        Field::new("auxiliary_name", DataType::Utf8, true),
        Field::new("auxiliary_value", DataType::Float64, true),
        Field::new("hold_reason", DataType::Utf8, true),
        Field::new("strategy_detail", DataType::Utf8, false),
        Field::new("fingerprint", DataType::Int64, true),
    ]))
}

fn signals_record_batch(rows: &[ReplaySignalDiagnostic]) -> Result<RecordBatch> {
    let schema = signal_schema();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.bar_timestamp_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bar_open).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bar_high).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bar_low).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bar_close).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.bar_index.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.bar_count as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.strategy.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.execution_path.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.signal_timing.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.signal_delay_bars as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.signal.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.raw_signal.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.effective_signal.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter()
                .map(|row| row.raw_buy_signal)
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter()
                .map(|row| row.raw_sell_signal)
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter()
                .map(|row| row.effective_buy_signal)
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter()
                .map(|row| row.effective_sell_signal)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter()
                .map(|row| row.current_position_qty)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter()
                .map(|row| row.effective_position_qty)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.target_qty).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.decision.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.gate_reason.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.order_action.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.order_qty).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.indicator_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.previous_fast_indicator)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.previous_slow_indicator)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.fast_indicator)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.slow_indicator)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.auxiliary_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.auxiliary_value)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.hold_reason.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.strategy_detail.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.fingerprint.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build signals Parquet record batch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::{ReplayExecutionPrecision, ReplayFillPriceSource};

    // Keep the writer smoke test local to this module; it proves the Arrow
    // schemas remain compatible with Parquet without needing a replay run.
    #[test]
    fn writes_empty_typed_result_sidecars() {
        let root = std::env::temp_dir().join(format!(
            "midas-result-parquet-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp result directory");
        let fill = ReplayExecutionFill {
            sequence: 1,
            lifecycle_sequence: None,
            fill_id: 1,
            order_id: 1,
            order_strategy_id: None,
            protection_order_id: None,
            account_id: 1,
            contract_id: 2,
            contract_name: "MESU6".to_string(),
            side: "Buy".to_string(),
            quantity: 1.0,
            price: 100.0,
            signal_timestamp_ns: None,
            submission_timestamp_ns: None,
            exchange_arrival_timestamp_ns: None,
            acknowledgement_timestamp_ns: None,
            fill_timestamp_ns: 1,
            fill_price_source: ReplayFillPriceSource::RawBarOpen,
            execution_precision: ReplayExecutionPrecision::BarApproximate,
            exit_reason: None,
            latency_ms: 0,
            tick_size: Some(0.25),
            value_per_point: Some(5.0),
            gross_realized_pnl_delta: None,
        };
        let artifacts = write_replay_parquet_artifacts(&root, &[fill], &[], &[], None, None)
            .expect("write Parquet sidecars");
        assert_eq!(artifacts.fills.as_deref(), Some("fills.parquet"));
        assert!(root.join("fills.parquet").is_file());
        assert!(root.join("trades.parquet").is_file());
        assert!(root.join("equity.parquet").is_file());
        let _ = std::fs::remove_dir_all(root);
    }
}

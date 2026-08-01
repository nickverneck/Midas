//! Typed Parquet exports for headless sweep summaries.
//!
//! Child replay results already write their own Parquet sidecars. These
//! summary tables keep the parent sweep cheap to inspect without opening every
//! child result directory and use the same atomic Snappy writer as single-run
//! artifacts.

use super::result_parquet::write_parquet_record_batch;
use super::sweep_runner::{
    ReplaySweepFeeScenarioSummary, ReplaySweepRunSummary, ReplaySweepSummaryDocument,
};
use anyhow::{Context, Result};
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use std::path::Path;
use std::sync::Arc;

pub(super) fn write_sweep_parquet_outputs(
    output_root: &Path,
    document: &ReplaySweepSummaryDocument,
) -> Result<()> {
    write_parquet_record_batch(
        &output_root.join("sweep-summary.parquet"),
        summary_record_batch(&document.runs)?,
    )?;
    if !document.fee_scenarios.is_empty() {
        write_parquet_record_batch(
            &output_root.join("sweep-fee-scenarios.parquet"),
            fee_scenario_record_batch(&document.fee_scenarios)?,
        )?;
    }
    Ok(())
}

fn summary_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("run_index", DataType::Int64, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("skipped", DataType::Boolean, false),
        Field::new("result_path", DataType::Utf8, true),
        Field::new("error", DataType::Utf8, true),
        Field::new("gross_pnl", DataType::Float64, true),
        Field::new("net_pnl", DataType::Float64, true),
        Field::new("fees", DataType::Float64, true),
        Field::new("max_drawdown", DataType::Float64, true),
        Field::new("trade_count", DataType::Int64, true),
        Field::new("fill_count", DataType::Int64, true),
    ]))
}

fn summary_record_batch(rows: &[ReplaySweepRunSummary]) -> Result<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.run_id.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.run_index as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.status.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            rows.iter().map(|row| row.skipped).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| {
                    row.result_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                })
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|row| row.error.clone()).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.gross_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.net_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.fees).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.max_drawdown).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.trade_count.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.fill_count.map(|value| value as i64))
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(summary_schema(), arrays).context("build sweep summary Parquet batch")
}

fn fee_scenario_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("run_index", DataType::Int64, false),
        Field::new("scenario_name", DataType::Utf8, false),
        Field::new("currency", DataType::Utf8, false),
        Field::new("total_per_contract", DataType::Float64, false),
        Field::new("fees", DataType::Float64, false),
        Field::new("gross_pnl", DataType::Float64, false),
        Field::new("net_pnl", DataType::Float64, false),
        Field::new("ending_equity", DataType::Float64, false),
        Field::new("return_on_initial_capital_pct", DataType::Float64, true),
        Field::new("max_drawdown", DataType::Float64, false),
        Field::new("max_drawdown_pct", DataType::Float64, true),
        Field::new("profit_factor", DataType::Float64, true),
    ]))
}

fn fee_scenario_record_batch(rows: &[ReplaySweepFeeScenarioSummary]) -> Result<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.run_id.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.run_index as i64)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.scenario_name.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.currency.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.total_per_contract)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.fees).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.gross_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.net_pnl).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.ending_equity).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.return_on_initial_capital_pct)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.max_drawdown).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter()
                .map(|row| row.max_drawdown_pct)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.profit_factor).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(fee_scenario_schema(), arrays)
        .context("build sweep fee-scenario Parquet batch")
}

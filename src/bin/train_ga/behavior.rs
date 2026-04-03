use anyhow::Result;
use midas_env::ml::ResolvedTrainingStack;
use std::path::Path;

use crate::{
    args::Args,
    backends,
    config::CandidateConfig,
    data::DataSet,
    generation::EvaluatedCandidate,
    setup::RunResources,
    types::{BehaviorRow, CandidateResult},
};

pub(crate) fn maybe_capture_behavior(
    args: &Args,
    stack: &ResolvedTrainingStack,
    resources: &RunResources,
    base_cfg: &CandidateConfig,
    generation: usize,
    best: &EvaluatedCandidate,
) -> Result<()> {
    let capture_behavior = args.behavior_every > 0 && generation % args.behavior_every == 0;
    if !capture_behavior {
        return Ok(());
    }

    let (train_behavior_metrics, train_history) = backends::evaluate_candidate_with_history(
        stack,
        &best.genome,
        &resources.datasets.train,
        &resources.windows.train,
        base_cfg,
    )?;
    let train_path = resources
        .behavior_dir
        .join(format!("train_gen{}_idx{}.csv", generation, best.idx));
    write_behavior_csv(
        &train_path,
        generation,
        best.idx,
        "train",
        &resources.datasets.train,
        &train_behavior_metrics,
        &train_history,
    )?;

    if !args.skip_val_eval {
        let (val_behavior_metrics, val_history) = backends::evaluate_candidate_with_history(
            stack,
            &best.genome,
            &resources.datasets.val,
            &resources.windows.val,
            base_cfg,
        )?;
        let val_path = resources
            .behavior_dir
            .join(format!("val_gen{}_idx{}.csv", generation, best.idx));
        write_behavior_csv(
            &val_path,
            generation,
            best.idx,
            "val",
            &resources.datasets.val,
            &val_behavior_metrics,
            &val_history,
        )?;
    }

    Ok(())
}

fn write_behavior_csv(
    path: &Path,
    generation: usize,
    candidate_idx: usize,
    split: &str,
    data: &DataSet,
    metrics: &CandidateResult,
    rows: &[BehaviorRow],
) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "gen",
        "idx",
        "split",
        "window",
        "step",
        "data_idx",
        "datetime_ns",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "action",
        "effective_action",
        "action_idx",
        "position_before",
        "position_after",
        "equity_before",
        "equity_after",
        "cash",
        "unrealized_pnl",
        "realized_pnl",
        "pnl_change",
        "realized_pnl_change",
        "reward",
        "commission_paid",
        "slippage_paid",
        "drawdown_penalty",
        "session_close_penalty",
        "auto_close_executed",
        "invalid_revert_penalty",
        "hold_duration_penalty",
        "flat_hold_penalty",
        "session_open",
        "margin_ok",
        "minutes_to_close",
        "session_closed_violation",
        "margin_call_violation",
        "position_limit_violation",
        "fitness",
        "pnl",
        "pnl_realized",
        "pnl_total",
        "sortino",
        "drawdown",
        "ret_mean",
    ])?;
    let symbol = data.symbol.as_str();
    for row in rows {
        let data_idx = row.data_idx;
        let datetime_ns = data
            .datetime_ns
            .as_ref()
            .and_then(|values| values.get(data_idx))
            .copied();
        let open = data.open.get(data_idx).copied();
        let high = data._high.get(data_idx).copied();
        let low = data._low.get(data_idx).copied();
        let close = data.close.get(data_idx).copied();
        let volume = data
            .volume
            .as_ref()
            .and_then(|values| values.get(data_idx))
            .copied();

        wtr.write_record([
            generation.to_string(),
            candidate_idx.to_string(),
            split.to_string(),
            row.window_idx.to_string(),
            row.step.to_string(),
            row.data_idx.to_string(),
            datetime_ns
                .map(|value| value.to_string())
                .unwrap_or_default(),
            symbol.to_string(),
            open.map(|value| value.to_string()).unwrap_or_default(),
            high.map(|value| value.to_string()).unwrap_or_default(),
            low.map(|value| value.to_string()).unwrap_or_default(),
            close.map(|value| value.to_string()).unwrap_or_default(),
            volume.map(|value| value.to_string()).unwrap_or_default(),
            row.action.clone(),
            row.effective_action.clone(),
            row.action_idx.to_string(),
            row.position_before.to_string(),
            row.position_after.to_string(),
            row.equity_before.to_string(),
            row.equity_after.to_string(),
            row.cash.to_string(),
            row.unrealized_pnl.to_string(),
            row.realized_pnl.to_string(),
            row.pnl_change.to_string(),
            row.realized_pnl_change.to_string(),
            row.reward.to_string(),
            row.commission_paid.to_string(),
            row.slippage_paid.to_string(),
            row.drawdown_penalty.to_string(),
            row.session_close_penalty.to_string(),
            row.auto_close_executed.to_string(),
            row.invalid_revert_penalty.to_string(),
            row.hold_duration_penalty.to_string(),
            row.flat_hold_penalty.to_string(),
            row.session_open.to_string(),
            row.margin_ok.to_string(),
            row.minutes_to_close
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.session_closed_violation.to_string(),
            row.margin_call_violation.to_string(),
            row.position_limit_violation.to_string(),
            metrics.fitness.to_string(),
            metrics.eval_pnl.to_string(),
            metrics.eval_pnl_realized.to_string(),
            metrics.eval_pnl_total.to_string(),
            metrics.eval_sortino.to_string(),
            metrics.eval_drawdown.to_string(),
            metrics.eval_ret_mean.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

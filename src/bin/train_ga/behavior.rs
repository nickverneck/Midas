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
    let diagnostics = metrics.diagnostics(rows);
    let penalties = diagnostics.penalties;
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
        "net_equity_delta",
        "cash",
        "unrealized_pnl",
        "gross_realized_pnl",
        "gross_mark_to_market_pnl_change",
        "gross_realized_pnl_change",
        "net_realized_pnl_change",
        "reward",
        "commission_paid",
        "slippage_paid",
        "drawdown_penalty",
        "session_close_penalty",
        "early_exit_penalty",
        "early_flip_penalty",
        "auto_close_executed",
        "invalid_revert_penalty",
        "hold_duration_penalty",
        "flat_hold_penalty",
        "violation_penalty",
        "session_open",
        "margin_ok",
        "minutes_to_close",
        "session_closed_violation",
        "margin_call_violation",
        "position_limit_violation",
        "terminal_liquidation",
        "terminal_liquidation_cost",
        "fitness",
        "net_objective_pnl",
        "net_realized_pnl_after_costs_and_penalties",
        "total_net_equity_delta",
        "gross_realized_pnl_candidate_mean",
        "execution_costs_candidate_mean",
        "shaping_penalties_candidate_mean",
        "terminal_liquidation_cost_candidate_mean",
        "sortino",
        "drawdown",
        "ret_mean",
        "penalty_drawdown_total",
        "penalty_session_close_total",
        "penalty_early_exit_total",
        "penalty_early_flip_total",
        "penalty_invalid_revert_total",
        "penalty_hold_duration_total",
        "penalty_flat_hold_total",
        "penalty_violation_total",
        "penalty_total",
        "account_equity_delta_total",
        "net_equity_delta_total",
        "net_equity_reconciliation_error",
        "behavior_window_count",
        "behavior_net_equity_delta_mean",
        "candidate_net_equity_reconciliation_error",
        "terminal_liquidation_cost_behavior_total",
        "terminal_liquidation_cost_behavior_mean",
        "terminal_liquidation_reconciliation_error",
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
            row.net_equity_delta.to_string(),
            row.cash.to_string(),
            row.unrealized_pnl.to_string(),
            row.gross_realized_pnl.to_string(),
            row.gross_mark_to_market_pnl_change.to_string(),
            row.gross_realized_pnl_change.to_string(),
            row.net_realized_pnl_change.to_string(),
            row.reward.to_string(),
            row.commission_paid.to_string(),
            row.slippage_paid.to_string(),
            row.drawdown_penalty.to_string(),
            row.session_close_penalty.to_string(),
            row.early_exit_penalty.to_string(),
            row.early_flip_penalty.to_string(),
            row.auto_close_executed.to_string(),
            row.invalid_revert_penalty.to_string(),
            row.hold_duration_penalty.to_string(),
            row.flat_hold_penalty.to_string(),
            row.violation_penalty.to_string(),
            row.session_open.to_string(),
            row.margin_ok.to_string(),
            row.minutes_to_close
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.session_closed_violation.to_string(),
            row.margin_call_violation.to_string(),
            row.position_limit_violation.to_string(),
            row.terminal_liquidation.to_string(),
            row.terminal_liquidation_cost.to_string(),
            metrics.fitness.to_string(),
            metrics.eval_pnl.to_string(),
            metrics.eval_pnl_realized.to_string(),
            metrics.eval_pnl_total.to_string(),
            metrics.eval_gross_realized_pnl.to_string(),
            metrics.eval_execution_costs.to_string(),
            metrics.eval_shaping_penalties.to_string(),
            metrics.terminal_liquidation_cost.to_string(),
            metrics.eval_sortino.to_string(),
            metrics.eval_drawdown.to_string(),
            metrics.eval_ret_mean.to_string(),
            penalties.drawdown_penalty.to_string(),
            penalties.session_close_penalty.to_string(),
            penalties.early_exit_penalty.to_string(),
            penalties.early_flip_penalty.to_string(),
            penalties.invalid_revert_penalty.to_string(),
            penalties.hold_duration_penalty.to_string(),
            penalties.flat_hold_penalty.to_string(),
            penalties.violation_penalty.to_string(),
            penalties.total_penalty().to_string(),
            penalties.actual_equity_delta.to_string(),
            penalties.net_equity_delta.to_string(),
            penalties.reconciliation_error.to_string(),
            penalties.window_count.to_string(),
            penalties.net_equity_delta_mean().to_string(),
            diagnostics.candidate_reconciliation_error.to_string(),
            penalties.terminal_liquidation_cost.to_string(),
            if penalties.window_count == 0 {
                0.0f64.to_string()
            } else {
                (penalties.terminal_liquidation_cost / penalties.window_count as f64).to_string()
            },
            diagnostics
                .terminal_liquidation_reconciliation_error
                .to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

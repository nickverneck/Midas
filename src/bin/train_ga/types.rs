use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateResult {
    pub fitness: f64,
    /// Mean per-window net objective PnL from the reconciled net equity curve.
    pub eval_pnl: f64,
    /// Net realized PnL after every evaluation window is explicitly flattened.
    ///
    /// This includes the final mark-to-market of a position that was still
    /// open at the end of a window, less execution costs and shaping penalties.
    pub eval_pnl_realized: f64,
    /// Mean per-window total net equity delta from the same curve as
    /// `eval_pnl`. This is the account-equity view of the objective result.
    pub eval_pnl_total: f64,
    /// Mean per-window gross realized PnL before execution costs or penalties.
    pub eval_gross_realized_pnl: f64,
    /// Mean per-window commission plus slippage, including terminal costs.
    pub eval_execution_costs: f64,
    /// Mean per-window shaping penalties, separate from trading costs.
    pub eval_shaping_penalties: f64,
    /// Mean per-window cost of the explicit end-of-window liquidation event.
    pub terminal_liquidation_cost: f64,
    pub eval_sortino: f64,
    pub eval_drawdown: f64,
    /// Mean per-step return generated from successive net equity values.
    pub eval_ret_mean: f64,
    pub debug_non_hold: usize,
    pub debug_non_zero_pos: usize,
    pub debug_mean_abs_pnl: f64,
    pub debug_buy: usize,
    pub debug_sell: usize,
    pub debug_hold: usize,
    pub debug_revert: usize,
    pub debug_session_violations: usize,
    pub debug_margin_violations: usize,
    pub debug_position_violations: usize,
    pub debug_drawdown_penalty: f64,
    pub debug_invalid_revert_penalty: f64,
    pub debug_hold_duration_penalty: f64,
    pub debug_flat_hold_penalty: f64,
    pub debug_session_close_penalty: f64,
}

/// Aggregate shaping-penalty diagnostics for a captured candidate evaluation.
///
/// `CandidateResult` is shared by the Torch, Burn, and Candle evaluators.  The
/// latter two construct it in backend-specific modules, so the detailed
/// penalty breakdown is deliberately kept as a report-side summary over the
/// common behavior rows rather than adding backend-specific fields to that
/// shared result struct.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PenaltySummary {
    pub drawdown_penalty: f64,
    pub session_close_penalty: f64,
    pub early_exit_penalty: f64,
    pub early_flip_penalty: f64,
    pub invalid_revert_penalty: f64,
    pub hold_duration_penalty: f64,
    pub flat_hold_penalty: f64,
    pub violation_penalty: f64,
    pub actual_equity_delta: f64,
    pub net_equity_delta: f64,
    /// Sum of the explicit end-of-window liquidation rows.
    pub terminal_liquidation_cost: f64,
    /// Number of distinct evaluation windows represented by the rows.
    pub window_count: usize,
    pub reconciliation_error: f64,
}

impl PenaltySummary {
    pub fn from_rows(rows: &[BehaviorRow]) -> Self {
        let mut summary = Self::default();
        let mut windows = BTreeSet::new();
        for row in rows {
            windows.insert(row.window_idx);
            summary.drawdown_penalty += row.drawdown_penalty;
            summary.session_close_penalty += row.session_close_penalty;
            summary.early_exit_penalty += row.early_exit_penalty;
            summary.early_flip_penalty += row.early_flip_penalty;
            summary.invalid_revert_penalty += row.invalid_revert_penalty;
            summary.hold_duration_penalty += row.hold_duration_penalty;
            summary.flat_hold_penalty += row.flat_hold_penalty;
            summary.violation_penalty += row.violation_penalty;
            summary.actual_equity_delta += row.equity_after - row.equity_before;
            summary.net_equity_delta += row.net_equity_delta;
            summary.terminal_liquidation_cost += row.terminal_liquidation_cost;
            summary.reconciliation_error += row.accounting_reconciliation_error();
        }
        summary.window_count = windows.len();
        summary
    }

    pub fn total_penalty(self) -> f64 {
        self.drawdown_penalty
            + self.session_close_penalty
            + self.early_exit_penalty
            + self.early_flip_penalty
            + self.invalid_revert_penalty
            + self.hold_duration_penalty
            + self.flat_hold_penalty
            + self.violation_penalty
    }

    pub fn expected_net_equity_delta(self) -> f64 {
        self.actual_equity_delta - self.total_penalty()
    }

    pub fn net_equity_delta_mean(self) -> f64 {
        if self.window_count == 0 {
            0.0
        } else {
            self.net_equity_delta / self.window_count as f64
        }
    }
}

/// The aggregate diagnostics associated with a `CandidateResult` report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateDiagnostics<'a> {
    pub result: &'a CandidateResult,
    pub penalties: PenaltySummary,
    /// Difference between CandidateResult's mean window PnL and the mean of
    /// the captured behavior rows.  This includes terminal liquidation rows.
    pub candidate_reconciliation_error: f64,
    /// Difference between CandidateResult's mean terminal cost and the
    /// captured rows' mean terminal cost.
    pub terminal_liquidation_reconciliation_error: f64,
}

impl CandidateResult {
    pub fn diagnostics<'a>(&'a self, rows: &[BehaviorRow]) -> CandidateDiagnostics<'a> {
        let penalties = PenaltySummary::from_rows(rows);
        CandidateDiagnostics {
            result: self,
            candidate_reconciliation_error: self.eval_pnl - penalties.net_equity_delta_mean(),
            terminal_liquidation_reconciliation_error: self.terminal_liquidation_cost
                - if penalties.window_count == 0 {
                    0.0
                } else {
                    penalties.terminal_liquidation_cost / penalties.window_count as f64
                },
            penalties,
        }
    }
}

/// Accounting for one evaluator transition.
///
/// `equity_after - equity_before` is the account-level change and therefore
/// already includes execution costs charged by `TradingEnv`.  The evaluator
/// subtracts the environment's shaping penalties from that change exactly
/// once so its net equity curve, PnL, and return series reconcile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StepAccounting {
    pub actual_equity_delta: f64,
    pub penalty_total: f64,
    pub violation_penalty: f64,
    pub net_equity_delta: f64,
}

impl StepAccounting {
    pub(crate) fn from_transition(
        equity_before: f64,
        equity_after: f64,
        info: &midas_env::env::StepInfo,
    ) -> Self {
        let violation_count = [
            info.session_closed_violation,
            info.margin_call_violation,
            info.position_limit_violation,
        ]
        .into_iter()
        .filter(|violation| *violation)
        .count() as f64;
        let violation_penalty = violation_count * midas_env::env::VIOLATION_PENALTY;
        let penalty_total = info.drawdown_penalty
            + info.session_close_penalty
            + info.early_exit_penalty
            + info.early_flip_penalty
            + info.invalid_revert_penalty
            + info.hold_duration_penalty
            + info.flat_hold_penalty
            + violation_penalty;
        let actual_equity_delta = equity_after - equity_before;

        Self {
            actual_equity_delta,
            penalty_total,
            violation_penalty,
            net_equity_delta: actual_equity_delta - penalty_total,
        }
    }
}

#[derive(Clone)]
pub struct BehaviorRow {
    pub window_idx: usize,
    pub step: usize,
    pub data_idx: usize,
    pub action_idx: i32,
    pub action: String,
    pub effective_action: String,
    pub position_before: i32,
    pub position_after: i32,
    pub equity_before: f64,
    pub equity_after: f64,
    pub net_equity_delta: f64,
    pub cash: f64,
    pub unrealized_pnl: f64,
    /// Gross realized PnL reported by the environment, before costs.
    pub gross_realized_pnl: f64,
    /// Gross mark-to-market PnL for this transition, before costs.
    pub gross_mark_to_market_pnl_change: f64,
    /// Gross PnL realized on this transition, before costs.
    pub gross_realized_pnl_change: f64,
    /// Gross realized change less commission, slippage, and shaping penalties.
    pub net_realized_pnl_change: f64,
    pub reward: f64,
    pub commission_paid: f64,
    pub slippage_paid: f64,
    pub drawdown_penalty: f64,
    pub session_close_penalty: f64,
    pub early_exit_penalty: f64,
    pub early_flip_penalty: f64,
    pub invalid_revert_penalty: f64,
    pub hold_duration_penalty: f64,
    pub flat_hold_penalty: f64,
    pub violation_penalty: f64,
    pub auto_close_executed: bool,
    pub session_open: bool,
    pub margin_ok: bool,
    pub minutes_to_close: Option<f64>,
    pub session_closed_violation: bool,
    pub margin_call_violation: bool,
    pub position_limit_violation: bool,
    /// True only for the synthetic end-of-window liquidation event.
    pub terminal_liquidation: bool,
    /// Commission plus slippage charged by the synthetic liquidation event.
    pub terminal_liquidation_cost: f64,
}

impl BehaviorRow {
    /// Construct the one explicit accounting event that closes a position
    /// left open at the end of an evaluation window.
    ///
    /// The environment already includes unrealized PnL in `equity_before`.
    /// Therefore this event adds only the final exit commission/slippage to
    /// the net equity curve.  Its realized-PnL fields do explicitly move the
    /// open mark-to-market into realized PnL, but that amount must never be
    /// added to the account-equity curve a second time.
    pub fn terminal_liquidation(
        window_idx: usize,
        step: usize,
        data_idx: usize,
        position_before: i32,
        equity_before: f64,
        unrealized_pnl_before: f64,
        realized_pnl_before: f64,
        commission_paid: f64,
        slippage_paid: f64,
        session_open: bool,
        margin_ok: bool,
        minutes_to_close: Option<f64>,
    ) -> Self {
        let terminal_liquidation_cost = commission_paid + slippage_paid;
        let equity_after = equity_before - terminal_liquidation_cost;

        Self {
            window_idx,
            step,
            data_idx,
            action_idx: -1,
            action: "terminal_liquidation".to_string(),
            effective_action: "terminal_liquidation".to_string(),
            position_before,
            position_after: 0,
            equity_before,
            equity_after,
            net_equity_delta: -terminal_liquidation_cost,
            cash: equity_after,
            unrealized_pnl: 0.0,
            gross_realized_pnl: realized_pnl_before + unrealized_pnl_before,
            gross_mark_to_market_pnl_change: 0.0,
            gross_realized_pnl_change: unrealized_pnl_before,
            net_realized_pnl_change: unrealized_pnl_before - terminal_liquidation_cost,
            reward: -terminal_liquidation_cost,
            commission_paid,
            slippage_paid,
            drawdown_penalty: 0.0,
            session_close_penalty: 0.0,
            early_exit_penalty: 0.0,
            early_flip_penalty: 0.0,
            invalid_revert_penalty: 0.0,
            hold_duration_penalty: 0.0,
            flat_hold_penalty: 0.0,
            violation_penalty: 0.0,
            auto_close_executed: false,
            session_open,
            margin_ok,
            minutes_to_close,
            session_closed_violation: false,
            margin_call_violation: false,
            position_limit_violation: false,
            terminal_liquidation: true,
            terminal_liquidation_cost,
        }
    }

    pub fn accounting_reconciliation_error(&self) -> f64 {
        let actual_equity_delta = self.equity_after - self.equity_before;
        let penalty_total = self.drawdown_penalty
            + self.session_close_penalty
            + self.early_exit_penalty
            + self.early_flip_penalty
            + self.invalid_revert_penalty
            + self.hold_duration_penalty
            + self.flat_hold_penalty
            + self.violation_penalty;
        self.net_equity_delta - (actual_equity_delta - penalty_total)
    }
}

#[cfg(test)]
mod tests {
    use super::{BehaviorRow, CandidateResult, PenaltySummary, StepAccounting};
    use midas_env::env::{Action, StepInfo, VIOLATION_PENALTY};

    fn info() -> StepInfo {
        StepInfo {
            effective_action: Action::Hold,
            action_overridden: false,
            commission_paid: 0.0,
            slippage_paid: 0.0,
            pnl_change: 0.0,
            realized_pnl_change: 0.0,
            drawdown_penalty: 1.0,
            session_close_penalty: 2.0,
            early_exit_penalty: 3.0,
            early_flip_penalty: 4.0,
            invalid_revert_penalty: 5.0,
            hold_duration_penalty: 6.0,
            flat_hold_penalty: 7.0,
            auto_close_executed: false,
            margin_call_violation: false,
            position_limit_violation: false,
            session_closed_violation: false,
        }
    }

    #[test]
    fn net_delta_uses_account_equity_and_subtracts_penalties_once() {
        let accounting = StepAccounting::from_transition(1_000.0, 990.0, &info());

        assert_eq!(accounting.actual_equity_delta, -10.0);
        assert_eq!(accounting.penalty_total, 28.0);
        assert_eq!(accounting.violation_penalty, 0.0);
        assert_eq!(accounting.net_equity_delta, -38.0);
    }

    #[test]
    fn violation_penalty_is_part_of_net_delta_without_equity_change() {
        let mut step = info();
        step.session_closed_violation = true;
        let accounting = StepAccounting::from_transition(1_000.0, 1_000.0, &step);

        assert_eq!(accounting.violation_penalty, VIOLATION_PENALTY);
        assert_eq!(accounting.net_equity_delta, -VIOLATION_PENALTY - 28.0);
    }

    fn behavior_row() -> BehaviorRow {
        BehaviorRow {
            window_idx: 0,
            step: 0,
            data_idx: 0,
            action_idx: 0,
            action: "hold".to_string(),
            effective_action: "hold".to_string(),
            position_before: 0,
            position_after: 0,
            equity_before: 100.0,
            equity_after: 92.0,
            net_equity_delta: -44.0,
            cash: 92.0,
            unrealized_pnl: 0.0,
            gross_realized_pnl: 0.0,
            gross_mark_to_market_pnl_change: 0.0,
            gross_realized_pnl_change: 0.0,
            net_realized_pnl_change: 0.0,
            reward: 0.0,
            commission_paid: 0.0,
            slippage_paid: 0.0,
            drawdown_penalty: 1.0,
            session_close_penalty: 2.0,
            early_exit_penalty: 3.0,
            early_flip_penalty: 4.0,
            invalid_revert_penalty: 5.0,
            hold_duration_penalty: 6.0,
            flat_hold_penalty: 7.0,
            violation_penalty: 8.0,
            auto_close_executed: false,
            session_open: true,
            margin_ok: true,
            minutes_to_close: None,
            session_closed_violation: false,
            margin_call_violation: false,
            position_limit_violation: false,
            terminal_liquidation: false,
            terminal_liquidation_cost: 0.0,
        }
    }

    #[test]
    fn penalty_summary_exposes_all_buckets_and_reconciles_net_delta() {
        let summary = PenaltySummary::from_rows(&[behavior_row()]);

        assert_eq!(summary.total_penalty(), 36.0);
        assert_eq!(summary.actual_equity_delta, -8.0);
        assert_eq!(summary.expected_net_equity_delta(), -44.0);
        assert_eq!(summary.net_equity_delta, -44.0);
        assert_eq!(summary.reconciliation_error, 0.0);
        assert_eq!(summary.early_exit_penalty, 3.0);
        assert_eq!(summary.early_flip_penalty, 4.0);
        assert_eq!(summary.violation_penalty, 8.0);
        assert_eq!(summary.window_count, 1);
        assert_eq!(summary.terminal_liquidation_cost, 0.0);
    }

    #[test]
    fn terminal_liquidation_row_reconciles_candidate_net_delta_once() {
        let ordinary = behavior_row();
        let terminal = BehaviorRow::terminal_liquidation(
            0,
            1,
            1,
            1,
            ordinary.equity_after,
            12.0,
            ordinary.gross_realized_pnl,
            0.80,
            0.25,
            true,
            true,
            None,
        );
        let rows = [ordinary, terminal.clone()];
        let summary = PenaltySummary::from_rows(&rows);

        assert!((terminal.terminal_liquidation_cost - 1.05).abs() < 1e-12);
        assert!((summary.terminal_liquidation_cost - 1.05).abs() < 1e-12);
        assert!((summary.actual_equity_delta + 9.05).abs() < 1e-12);
        assert!((summary.net_equity_delta + 45.05).abs() < 1e-12);
        assert!((summary.net_equity_delta_mean() + 45.05).abs() < 1e-12);
        assert!(summary.reconciliation_error.abs() < 1e-12);

        let result = CandidateResult {
            fitness: 0.0,
            eval_pnl: -45.05,
            eval_pnl_realized: 0.0,
            eval_pnl_total: -45.05,
            eval_gross_realized_pnl: 0.0,
            eval_execution_costs: 0.0,
            eval_shaping_penalties: 0.0,
            terminal_liquidation_cost: 1.05,
            eval_sortino: 0.0,
            eval_drawdown: 0.0,
            eval_ret_mean: 0.0,
            debug_non_hold: 0,
            debug_non_zero_pos: 0,
            debug_mean_abs_pnl: 0.0,
            debug_buy: 0,
            debug_sell: 0,
            debug_hold: 0,
            debug_revert: 0,
            debug_session_violations: 0,
            debug_margin_violations: 0,
            debug_position_violations: 0,
            debug_drawdown_penalty: 0.0,
            debug_invalid_revert_penalty: 0.0,
            debug_hold_duration_penalty: 0.0,
            debug_flat_hold_penalty: 0.0,
            debug_session_close_penalty: 0.0,
        };
        let diagnostics = result.diagnostics(&rows);
        assert!(diagnostics.candidate_reconciliation_error.abs() < 1e-12);
        assert!(diagnostics.terminal_liquidation_reconciliation_error.abs() < 1e-12);
    }

    #[test]
    fn terminal_liquidation_row_is_still_emitted_when_execution_cost_is_zero() {
        let terminal = BehaviorRow::terminal_liquidation(
            0, 1, 1, 1, 100.0, 12.0, 3.0, 0.0, 0.0, true, true, None,
        );

        assert!(terminal.terminal_liquidation);
        assert_eq!(terminal.terminal_liquidation_cost, 0.0);
        assert_eq!(terminal.net_equity_delta, 0.0);
        assert_eq!(terminal.gross_realized_pnl_change, 12.0);
        assert_eq!(terminal.net_realized_pnl_change, 12.0);
    }
}

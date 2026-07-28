use super::*;

pub(crate) fn target_qty_for_signal(
    signal: StrategySignal,
    current_qty: i32,
    base_qty: i32,
) -> Option<i32> {
    let base_qty = base_qty.max(1);
    match signal {
        StrategySignal::Hold => None,
        StrategySignal::EnterLong => Some(base_qty),
        StrategySignal::EnterShort => Some(-base_qty),
        StrategySignal::ExitLongOnShortSignal => {
            if current_qty > 0 {
                Some(0)
            } else {
                None
            }
        }
    }
}

pub(crate) fn entry_signal_consumed_while_flat(
    session: &SessionState,
    signal: StrategySignal,
    current_qty: i32,
) -> bool {
    session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && current_qty == 0
        && matches!(
            signal,
            StrategySignal::EnterLong | StrategySignal::EnterShort
        )
        && session.execution_runtime.last_dispatched_entry_signal == Some(signal)
}

pub(crate) fn mark_closed_bar_signal_dispatched(
    session: &mut SessionState,
    signal_bar_ts: i64,
    signal: StrategySignal,
) {
    if session.execution_config.native_signal_timing != NativeSignalTiming::ClosedBar {
        return;
    }
    session.execution_runtime.last_dispatched_signal_bar_ts = Some(signal_bar_ts);
    if matches!(
        signal,
        StrategySignal::EnterLong | StrategySignal::EnterShort
    ) {
        session.execution_runtime.last_dispatched_entry_signal = Some(signal);
    }
}

pub(super) fn closed_bar_signal_already_dispatched(
    session: &SessionState,
    signal_bar_ts: i64,
) -> bool {
    session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_dispatched_signal_bar_ts == Some(signal_bar_ts)
}

pub(super) fn seed_hma_cross_observed_side(session: &mut SessionState) {
    if session.execution_config.kind != StrategyKind::Native
        || session.execution_config.native_strategy != NativeStrategyKind::HmaCross
    {
        return;
    }

    let bars = signal_evaluation_bars(session).to_vec();
    if bars.is_empty() {
        return;
    }
    let current_side = side_from_signed_qty(effective_market_position_qty(session));
    let config = session.execution_config.native_hma_cross.clone();
    let _ = config.evaluate_current_cross(
        &mut session.execution_runtime.hma_cross_execution,
        &bars,
        current_side,
    );
}

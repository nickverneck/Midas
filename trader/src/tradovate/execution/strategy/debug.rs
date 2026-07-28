use super::*;
use crate::strategy_debug::{StrategyDecisionDebug, format_strategy_decision};

pub(super) fn guarded_strategy_eval_context(session: &SessionState, actual_qty: i32) -> String {
    let mut context = execution_observability_context(session);
    if session.execution_config.native_strategy == NativeStrategyKind::HmaCross {
        context.push_str(" | ");
        context.push_str(&hma_cross_market_debug(session, actual_qty));
    }
    context
}

pub(crate) fn strategy_bar_debug_position(
    session: &SessionState,
    signal_bar_ts: Option<i64>,
) -> (Option<String>, Option<String>) {
    let bars = signal_evaluation_bars(session);
    let bar_count = Some(bars.len().to_string());
    let bar_index = signal_bar_ts.and_then(|ts| {
        bars.iter()
            .position(|bar| bar.ts_ns == ts)
            .map(|idx| idx + 1)
    });
    (bar_index.map(|idx| idx.to_string()), bar_count)
}

pub(crate) fn debug_pending_target(session: &SessionState) -> String {
    session
        .execution_runtime
        .pending_target_qty
        .map(|qty| qty.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(crate) fn debug_target_qty(target_qty: Option<i32>) -> String {
    target_qty
        .map(|qty| qty.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(crate) struct TradovateStrategyDecisionDebug<'a> {
    pub path: &'a str,
    pub decision: &'a str,
    pub signal: Option<StrategySignal>,
    pub bar_ts: Option<i64>,
    pub actual_qty: i32,
    pub effective_qty: i32,
    pub target_qty: Option<i32>,
    pub strategy_detail: &'a str,
    pub gate_detail: String,
    pub fingerprint: Option<u64>,
}

pub(crate) fn format_tradovate_strategy_decision(
    session: &SessionState,
    debug: TradovateStrategyDecisionDebug<'_>,
) -> String {
    let (bar_index, bar_count) = strategy_bar_debug_position(session, debug.bar_ts);
    format_strategy_decision(&StrategyDecisionDebug {
        strategy: Some(active_native_slug(session).to_string()),
        broker: Some("tradovate".to_string()),
        path: Some(debug.path.to_string()),
        decision: Some(debug.decision.to_string()),
        timing: Some(active_signal_timing_label(session).to_string()),
        signal_delay_bars: Some(signal_delay_bars(session).to_string()),
        bar_ts: debug.bar_ts.map(|ts| ts.to_string()),
        bar_index,
        bar_count,
        fingerprint: debug.fingerprint.map(|fingerprint| fingerprint.to_string()),
        actual_qty: Some(debug.actual_qty.to_string()),
        effective_qty: Some(debug.effective_qty.to_string()),
        pending_target: Some(debug_pending_target(session)),
        target_qty: Some(debug_target_qty(debug.target_qty)),
        signal: debug.signal.map(|signal| signal.label().to_string()),
        reason: Some(debug.decision.to_string()),
        strategy_detail: Some(debug.strategy_detail.to_string()),
        gate_detail: Some(debug.gate_detail),
    })
}

pub(super) fn emit_guarded_strategy_eval_debug(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    decision: &str,
    signal: StrategySignal,
    signal_bar_ts: i64,
    actual_qty: i32,
    effective_qty: i32,
    target_qty: Option<i32>,
    debug_summary: &str,
) {
    let legacy = format!(
        "strategy eval | {} | {decision} | signal {} | bar_ts {} | actual_qty {} | effective_qty {} | target_qty {}",
        active_native_slug(session),
        signal.label(),
        signal_bar_ts,
        actual_qty,
        effective_qty,
        debug_target_qty(target_qty)
    );
    let gate_detail = format!(
        "{legacy} | {}",
        guarded_strategy_eval_context(session, actual_qty)
    );
    let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
        session,
        TradovateStrategyDecisionDebug {
            path: "guarded",
            decision,
            signal: Some(signal),
            bar_ts: Some(signal_bar_ts),
            actual_qty,
            effective_qty,
            target_qty,
            strategy_detail: debug_summary,
            gate_detail,
            fingerprint: latest_strategy_bar_fingerprint(session),
        },
    )));
}

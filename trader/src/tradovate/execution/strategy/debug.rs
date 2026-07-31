use super::*;
use crate::strategy_debug::{StrategyDecisionDebug, format_strategy_decision};

/// Append one structured strategy decision row when replay diagnostics are
/// explicitly enabled. The helper is deliberately a no-op for live sessions
/// and for the default replay configuration.
pub(crate) fn record_replay_signal_diagnostic(
    session: &mut SessionState,
    signal_bar_ts: i64,
    signal: StrategySignal,
    actual_qty: i32,
    effective_qty: i32,
    target_qty: Option<i32>,
    decision: &str,
    gate_reason: &str,
    order_action: Option<&str>,
    order_qty: Option<i32>,
    strategy_detail: &str,
) {
    if !session.replay_enabled || !session.cfg.replay_signal_diagnostics {
        return;
    }

    let (bar, bar_index, bar_count, snapshot) = {
        let bars = signal_evaluation_bars(session);
        let Some(index) = bars.iter().position(|bar| bar.ts_ns == signal_bar_ts) else {
            return;
        };
        let bar = bars[index].clone();
        let snapshot = snapshot_active_execution_strategy(session, &bars[..=index], effective_qty);
        (bar, Some(index + 1), bars.len(), snapshot)
    };
    let raw_signal = signal_direction(snapshot.raw_buy_signal, snapshot.raw_sell_signal);
    let effective_signal = signal_direction(
        snapshot.effective_buy_signal,
        snapshot.effective_sell_signal,
    );
    let strategy = active_native_slug(session).to_string();
    let signal_timing = active_signal_timing_label(session).to_string();
    let signal_delay_bars = signal_delay_bars(session);
    let fingerprint = Some(bar_fingerprint(&bar));

    session.execution_runtime.replay_signal_diagnostics.push(
        crate::broker::ReplaySignalDiagnostic {
            bar_timestamp_ns: bar.ts_ns,
            bar_open: bar.open,
            bar_high: bar.high,
            bar_low: bar.low,
            bar_close: bar.close,
            bar_index,
            bar_count,
            strategy,
            execution_path: decision_path(session),
            signal_timing,
            signal_delay_bars,
            signal: signal.label().to_string(),
            raw_signal,
            effective_signal,
            raw_buy_signal: snapshot.raw_buy_signal,
            raw_sell_signal: snapshot.raw_sell_signal,
            effective_buy_signal: snapshot.effective_buy_signal,
            effective_sell_signal: snapshot.effective_sell_signal,
            current_position_qty: actual_qty,
            effective_position_qty: effective_qty,
            target_qty,
            decision: decision.to_string(),
            gate_reason: gate_reason.to_string(),
            order_action: order_action.map(ToString::to_string),
            order_qty,
            indicator_name: snapshot.indicator_name.to_string(),
            previous_fast_indicator: snapshot.previous_fast_indicator,
            previous_slow_indicator: snapshot.previous_slow_indicator,
            fast_indicator: snapshot.fast_indicator,
            slow_indicator: snapshot.slow_indicator,
            auxiliary_name: snapshot.auxiliary_name.map(ToString::to_string),
            auxiliary_value: snapshot.auxiliary_value,
            hold_reason: snapshot.hold_reason.map(ToString::to_string),
            strategy_detail: strategy_detail.to_string(),
            fingerprint,
        },
    );
}

fn signal_direction(buy: bool, sell: bool) -> String {
    match (buy, sell) {
        (true, true) => "buy+sell".to_string(),
        (true, false) => "buy".to_string(),
        (false, true) => "sell".to_string(),
        (false, false) => "none".to_string(),
    }
}

fn decision_path(session: &SessionState) -> String {
    if session.execution_config.kind != StrategyKind::Native {
        "non_native".to_string()
    } else {
        match session.execution_config.native_execution_path {
            NativeExecutionPath::Guarded => "guarded".to_string(),
            NativeExecutionPath::SimpleDiagnostic => "simple diagnostic".to_string(),
            NativeExecutionPath::HmaDirect => "hma direct".to_string(),
        }
    }
}

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

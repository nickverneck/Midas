use super::*;
use crate::strategy_debug::{StrategyDecisionDebug, format_strategy_decision};

pub(crate) fn effective_market_position_qty(session: &SessionState) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

pub(crate) fn strategy_has_live_broker_path(
    session: &SessionState,
    key: StrategyProtectionKey,
    order_strategy_id: i64,
) -> bool {
    session
        .user_store
        .linked_strategy_orders(key.account_id, order_strategy_id)
        .into_iter()
        .any(|order| order_is_active(order) && order_contract_id(order) == Some(key.contract_id))
}

pub(crate) const ORDER_STRATEGY_HYDRATION_GRACE_MS: u128 = 1_500;
pub(crate) const MARKET_ORDER_POSITION_SYNC_GRACE_MS: u128 = 3_000;
pub(crate) const ORDER_STRATEGY_POSITION_SYNC_GRACE_MS: u128 = 10_000;

pub(crate) fn tracker_within_broker_path_grace(
    session: &SessionState,
    order_strategy_id: i64,
) -> bool {
    session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.order_strategy_id == Some(order_strategy_id)
                && tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_HYDRATION_GRACE_MS
        })
}

pub(crate) fn selected_contract_has_live_broker_path(session: &SessionState) -> bool {
    if session.order_submit_in_flight {
        return true;
    }

    let Some(account_id) = session.selected_account_id else {
        return false;
    };
    let Some(key) = selected_strategy_key(session).ok() else {
        return false;
    };

    if let Some(tracker) = session.order_latency_tracker.as_ref() {
        if let Some(order_id) = tracker.order_id.or_else(|| {
            session
                .user_store
                .order_id_by_client_id(account_id, &tracker.cl_ord_id)
        }) {
            if session
                .user_store
                .find_order(account_id, order_id)
                .is_some_and(order_is_active)
            {
                return true;
            }
        }

        if let Some(order_strategy_id) = tracker.order_strategy_id {
            if strategy_has_live_broker_path(session, key, order_strategy_id) {
                return true;
            }
            if tracker_within_broker_path_grace(session, order_strategy_id) {
                return true;
            }
        }
    }

    active_order_strategy_matches_selected(session).is_some_and(|tracked| {
        strategy_has_live_broker_path(session, key, tracked.order_strategy_id)
    })
}

pub(crate) fn pending_target_has_live_broker_path(session: &SessionState) -> bool {
    selected_contract_has_live_broker_path(session)
}

fn selected_contract_position_qty(session: &SessionState) -> Option<i32> {
    let account_id = session.selected_account_id?;
    let contract = session.selected_contract.as_ref()?;
    session
        .user_store
        .contract_position_qty(account_id, contract)
        .map(|qty| qty.round() as i32)
}

fn flat_broker_path_should_wait(session: &SessionState) -> bool {
    if !selected_contract_has_live_broker_path(session) {
        return false;
    }
    if selected_contract_position_qty(session) != Some(0) {
        return true;
    }
    if session.order_submit_in_flight || session.protection_sync_in_flight {
        return true;
    }
    session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker.started_at.elapsed().as_millis() <= ORDER_STRATEGY_POSITION_SYNC_GRACE_MS
        })
}

fn emit_pending_target_gate_debug(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    source: &str,
    pending_target_qty: i32,
    actual_qty: i32,
    reached: bool,
    overshot: bool,
    has_live_broker_path: bool,
    waiting_for_position_sync: bool,
) {
    let effective_qty = effective_market_position_qty(session);
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "strategy pending target gate | {source} | pending target {pending_target_qty} | actual {actual_qty} | effective {effective_qty} | reached {reached} | overshot {overshot} | live broker path {has_live_broker_path} | waiting position sync {waiting_for_position_sync} | {}",
        execution_observability_context(session)
    )));
}

pub(crate) fn should_wait_for_automated_position_sync(
    session: &SessionState,
    pending_target_qty: i32,
    actual_qty: i32,
) -> bool {
    if pending_target_qty == 0 || actual_qty == pending_target_qty {
        return false;
    }

    let Some(tracker) = session.order_latency_tracker.as_ref() else {
        return false;
    };
    let has_submission_progress = tracker.order_strategy_id.is_some()
        || tracker.order_id.is_some()
        || tracker.seen_recorded
        || tracker.exec_report_recorded
        || tracker.fill_recorded;
    if !has_submission_progress {
        return false;
    }

    let grace_ms = if tracker.order_strategy_id.is_some() {
        ORDER_STRATEGY_POSITION_SYNC_GRACE_MS
    } else {
        MARKET_ORDER_POSITION_SYNC_GRACE_MS
    };

    tracker.started_at.elapsed().as_millis() <= grace_ms
}

pub(crate) fn clear_stale_pending_target(
    session: &mut SessionState,
    pending: i32,
    actual_qty: i32,
    event_tx: &UnboundedSender<ServiceEvent>,
) {
    let observability = execution_observability_context(session);
    session.execution_runtime.pending_target_qty = None;
    session.pending_signal_context = None;
    session.order_latency_tracker = None;
    session.execution_runtime.last_summary = format!(
        "Pending target {pending} cleared: broker has no active order path and position is still {actual_qty}; re-evaluating."
    );
    force_reevaluate_pending_window(session);
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Pending target {pending} cleared: broker has no active order path; re-evaluating."
    )));
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "pending target cleared | target {pending} | actual {actual_qty} | broker has no active order path | {observability}"
    )));
}

fn force_reevaluate_pending_window(session: &mut SessionState) {
    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        return;
    }

    let Some(latest_strategy_ts) = latest_strategy_bar_ts(session) else {
        return;
    };
    if session
        .execution_runtime
        .last_closed_bar_ts
        .is_some_and(|last_seen| last_seen < latest_strategy_ts)
    {
        return;
    }
    session.execution_runtime.last_closed_bar_ts = Some(latest_strategy_ts.saturating_sub(1));
    session.execution_runtime.last_closed_bar_fingerprint = None;
}

pub(crate) fn evaluate_active_execution_strategy(
    session: &SessionState,
    bars: &[Bar],
    current_qty: i32,
) -> (StrategySignal, String, String) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            let evaluation = session
                .execution_config
                .native_hma
                .evaluate(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal,
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
        NativeStrategyKind::EmaCross => {
            let evaluation = session
                .execution_config
                .native_ema
                .evaluate(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal,
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
        NativeStrategyKind::HmaCross => {
            let evaluation = session
                .execution_config
                .native_hma_cross
                .evaluate(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal,
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
    }
}

pub(crate) fn evaluate_active_execution_strategy_since(
    session: &SessionState,
    bars: &[Bar],
    current_qty: i32,
    after_ts: Option<i64>,
) -> (Bar, StrategySignal, String, String) {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        let signal_bar = bars
            .last()
            .expect("strategy bars must not be empty")
            .clone();
        let (signal, summary, debug_summary) =
            evaluate_active_execution_strategy(session, bars, current_qty);
        return (signal_bar, signal, summary, debug_summary);
    }

    let current_side = side_from_signed_qty(current_qty);
    let start_idx = after_ts
        .and_then(|ts| bars.iter().position(|bar| bar.ts_ns > ts))
        .unwrap_or_else(|| bars.len().saturating_sub(1));
    let mut latest = None;
    for idx in start_idx..bars.len() {
        let window = &bars[..=idx];
        let signal_bar = bars[idx].clone();
        let (signal, summary, debug_summary) = match session.execution_config.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let evaluation = session
                    .execution_config
                    .native_hma
                    .evaluate(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::EmaCross => {
                let evaluation = session
                    .execution_config
                    .native_ema
                    .evaluate(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::HmaCross => {
                let evaluation = session
                    .execution_config
                    .native_hma_cross
                    .evaluate(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
        };
        if signal != StrategySignal::Hold {
            latest = Some((signal_bar, signal, summary, debug_summary));
        } else if latest.is_none() {
            latest = Some((signal_bar, signal, summary, debug_summary));
        }
    }
    latest.expect("strategy bars must not be empty")
}

pub(crate) fn evaluate_active_execution_strategy_since_mut(
    session: &mut SessionState,
    bars: &[Bar],
    current_qty: i32,
    after_ts: Option<i64>,
) -> (Bar, StrategySignal, String, String) {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        let signal_bar = bars
            .last()
            .expect("strategy bars must not be empty")
            .clone();
        let current_side = side_from_signed_qty(current_qty);
        let (signal, summary, debug_summary) = match session.execution_config.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let evaluation = session
                    .execution_config
                    .native_hma
                    .evaluate(bars, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::EmaCross => {
                let evaluation = session
                    .execution_config
                    .native_ema
                    .evaluate(bars, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::HmaCross => {
                let config = session.execution_config.native_hma_cross.clone();
                let evaluation = config.evaluate_current_cross(
                    &mut session.execution_runtime.hma_cross_execution,
                    bars,
                    current_side,
                );
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
        };
        return (signal_bar, signal, summary, debug_summary);
    }

    let current_side = side_from_signed_qty(current_qty);
    let start_idx = after_ts
        .and_then(|ts| bars.iter().position(|bar| bar.ts_ns > ts))
        .unwrap_or_else(|| bars.len().saturating_sub(1));
    let mut latest = None;
    for idx in start_idx..bars.len() {
        let window = &bars[..=idx];
        let signal_bar = bars[idx].clone();
        let (signal, summary, debug_summary) = match session.execution_config.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let evaluation = session
                    .execution_config
                    .native_hma
                    .evaluate(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::EmaCross => {
                let evaluation = session
                    .execution_config
                    .native_ema
                    .evaluate(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::HmaCross => {
                let config = session.execution_config.native_hma_cross.clone();
                let evaluation = config.evaluate_current_cross(
                    &mut session.execution_runtime.hma_cross_execution,
                    window,
                    current_side,
                );
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
        };
        if signal != StrategySignal::Hold {
            latest = Some((signal_bar, signal, summary, debug_summary));
        } else if latest.is_none() {
            latest = Some((signal_bar, signal, summary, debug_summary));
        }
    }
    latest.expect("strategy bars must not be empty")
}

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

fn closed_bar_signal_already_dispatched(session: &SessionState, signal_bar_ts: i64) -> bool {
    session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_dispatched_signal_bar_ts == Some(signal_bar_ts)
}

fn seed_hma_cross_observed_side(session: &mut SessionState) {
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

fn guarded_strategy_eval_context(session: &SessionState, actual_qty: i32) -> String {
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

fn emit_guarded_strategy_eval_debug(
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

fn staged_reversal_timing(entry: &PendingNativeReversalEntry) -> String {
    let elapsed_ms = entry.started_at.elapsed().as_millis();
    let flat_wait = entry
        .flat_seen_at
        .map(|flat_seen_at| format!(" | flat_wait {}ms", flat_seen_at.elapsed().as_millis()))
        .unwrap_or_default();
    format!("elapsed {elapsed_ms}ms{flat_wait}")
}

pub(crate) fn continue_staged_reversal(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
    actual_qty: i32,
) -> Result<bool> {
    let Some(staged) = session.execution_runtime.pending_reversal_entry.clone() else {
        return Ok(false);
    };

    if actual_qty.signum() == staged.target_qty.signum() && actual_qty != 0 {
        session.execution_runtime.pending_reversal_entry = None;
        let next_summary = format!(
            "Staged reversal resolved at target {actual_qty} ({}).",
            staged_reversal_timing(&staged)
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution staged reversal resolved",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(true);
    }

    if actual_qty == 0 {
        if let Some(entry) = session.execution_runtime.pending_reversal_entry.as_mut() {
            if entry.flat_seen_at.is_none() {
                entry.flat_seen_at = Some(time::Instant::now());
            }
        }
        let staged = session
            .execution_runtime
            .pending_reversal_entry
            .clone()
            .unwrap_or(staged);
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Staged reversal flat; waiting for broker path to clear before entering {} ({}).",
                staged.target_qty,
                staged_reversal_timing(&staged)
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution staged reversal wait",
            );
            session.execution_runtime.last_summary = next_summary;
            emit_execution_state(event_tx, session);
            return Ok(true);
        }

        let entry_summary = format!(
            "Flat confirmed and broker path clear; submitting staged reversal entry to {} ({}).",
            staged.target_qty,
            staged_reversal_timing(&staged)
        );
        let dispatch_outcome = dispatch_target_position_order(
            session,
            broker_tx,
            staged.target_qty,
            true,
            &staged.reason,
        )?;
        match dispatch_outcome {
            MarketOrderDispatchOutcome::NoOp { message } => {
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = message.clone();
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
            MarketOrderDispatchOutcome::Queued { target_qty } => {
                session.execution_runtime.pending_target_qty = target_qty;
                session.execution_runtime.pending_reversal_entry = None;
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &entry_summary,
                    "execution staged reversal entry submit",
                );
                session.execution_runtime.last_summary = entry_summary;
            }
        }
        emit_execution_state(event_tx, session);
        return Ok(true);
    }

    let flatten_reason = format!(
        "{} | staged reversal flatten {} -> 0 before {}",
        staged.reason, actual_qty, staged.target_qty
    );
    let dispatch_outcome =
        dispatch_target_position_order(session, broker_tx, 0, true, &flatten_reason)?;
    match dispatch_outcome {
        MarketOrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message.clone();
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            session.execution_runtime.last_summary = format!(
                "Flattening {} before staged reversal to {}.",
                actual_qty, staged.target_qty
            );
        }
    }
    emit_execution_state(event_tx, session);
    Ok(true)
}

pub(crate) fn arm_execution_strategy(session: &mut SessionState) {
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_closed_bar_fingerprint = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
    session.execution_runtime.last_closed_bar_fingerprint =
        latest_strategy_bar_fingerprint(session);
    seed_hma_cross_observed_side(session);
    session.execution_runtime.last_summary =
        if session.execution_runtime.last_closed_bar_ts.is_some() {
            format!(
                "Native {} armed from current {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        } else {
            format!(
                "Native {} armed; waiting for first {}.",
                active_native_label(session),
                active_signal_timing_label(session)
            )
        };
}

pub(crate) fn disarm_execution_strategy(session: &mut SessionState, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.last_closed_bar_fingerprint = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

pub(crate) fn handle_execution_account_sync(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);
    reconcile_selected_active_order_strategy(session);
    hydrate_selected_order_strategy_protection(session);

    let mut runtime_changed = false;
    let max_automated_qty = session.execution_config.order_qty.max(1);
    if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && actual_qty.abs() > max_automated_qty
    {
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Waiting for broker sync: temporary position {actual_qty} exceeds max {max_automated_qty} while an order path is still active."
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution broker sync wait",
            );
            session.execution_runtime.last_summary = next_summary;
        } else {
            let reason = format!(
                "Automation disarmed: position drifted to {actual_qty}, above configured max {max_automated_qty}."
            );
            emit_execution_transition_debug(event_tx, session, &reason, "execution drift disarm");
            disarm_execution_strategy(session, reason);
        }
        runtime_changed = true;
    }
    if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && actual_qty == 0
        && session.execution_runtime.pending_target_qty.is_none()
        && flat_broker_path_should_wait(session)
    {
        let next_summary =
            "Waiting for broker sync: flat position reported while an order path is still active."
                .to_string();
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution flat broker sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }
    if let Some(pending) = session.execution_runtime.pending_target_qty {
        let reached = pending == actual_qty;
        // Position overshot past target — entry filled, then something else
        // (e.g. orphaned orders) pushed position further.  Release pending
        // so the strategy can correct on the next bar.
        let overshot =
            (pending > 0 && actual_qty > pending) || (pending < 0 && actual_qty < pending);
        let has_live_broker_path = pending_target_has_live_broker_path(session);
        let waiting_for_position_sync = !has_live_broker_path
            && should_wait_for_automated_position_sync(session, pending, actual_qty);
        emit_pending_target_gate_debug(
            event_tx,
            session,
            "account sync",
            pending,
            actual_qty,
            reached,
            overshot,
            has_live_broker_path,
            waiting_for_position_sync,
        );
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
            force_reevaluate_pending_window(session);
            if continue_staged_reversal(session, broker_tx, event_tx, actual_qty)? {
                runtime_changed = true;
            } else {
                let next_summary = if reached {
                    format!("Position confirmed at target {actual_qty}")
                } else {
                    format!(
                        "Position at {actual_qty} (target was {pending}); re-evaluating on next bar"
                    )
                };
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution pending target settled",
                );
                session.execution_runtime.last_summary = next_summary;
                runtime_changed = true;
            }
        } else if pending != 0 && !has_live_broker_path {
            if waiting_for_position_sync {
                let next_summary = format!(
                    "Waiting for position sync after automated order settle (actual {actual_qty}, pending target {pending})."
                );
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution market position sync wait",
                );
                session.execution_runtime.last_summary = next_summary;
                runtime_changed = true;
            } else {
                clear_stale_pending_target(session, pending, actual_qty, event_tx);
                runtime_changed = true;
            }
        }
    }

    if session.execution_runtime.pending_target_qty.is_none()
        && continue_staged_reversal(session, broker_tx, event_tx, actual_qty)?
    {
        runtime_changed = true;
    }

    let waiting_for_native_protection_sync = session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
        && session.execution_runtime.pending_target_qty.is_none()
        && selected_managed_protection_waiting_for_position_sync(session);
    if waiting_for_native_protection_sync {
        let next_summary = format!(
            "Waiting for broker sync after native protection activity (position still {actual_qty})."
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution native protection sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        runtime_changed = true;
    } else if session.execution_runtime.armed
        && session.execution_config.kind == StrategyKind::Native
    {
        sync_execution_protection(session, broker_tx, None)?;
    }

    if runtime_changed {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

pub(crate) fn maybe_run_execution_strategy(
    session: &mut SessionState,
    broker_tx: &UnboundedSender<BrokerCommand>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let actual_market_qty = selected_market_position_qty(session);
    let actual_market_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_market_qty, actual_market_entry);
    if session.execution_runtime.pending_target_qty.is_none()
        && continue_staged_reversal(session, broker_tx, event_tx, actual_market_qty)?
    {
        return Ok(());
    }
    let max_automated_qty = session.execution_config.order_qty.max(1);
    if actual_market_qty.abs() > max_automated_qty {
        if selected_contract_has_live_broker_path(session) {
            let next_summary = format!(
                "Waiting for broker sync: temporary position {actual_market_qty} exceeds max {max_automated_qty} while an order path is still active."
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution broker sync wait",
            );
            session.execution_runtime.last_summary = next_summary;
        } else {
            let reason = format!(
                "Automation disarmed: position drifted to {actual_market_qty}, above configured max {max_automated_qty}."
            );
            emit_execution_transition_debug(event_tx, session, &reason, "execution drift disarm");
            disarm_execution_strategy(session, reason);
        }
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if actual_market_qty == 0
        && session.execution_runtime.pending_target_qty.is_none()
        && flat_broker_path_should_wait(session)
    {
        let next_summary =
            "Waiting for broker sync: flat position reported while an order path is still active."
                .to_string();
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution flat broker sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        let reached = pending_target_qty == actual_market_qty;
        let overshot = (pending_target_qty > 0 && actual_market_qty > pending_target_qty)
            || (pending_target_qty < 0 && actual_market_qty < pending_target_qty);
        let has_live_broker_path = pending_target_has_live_broker_path(session);
        let waiting_for_position_sync = !has_live_broker_path
            && should_wait_for_automated_position_sync(
                session,
                pending_target_qty,
                actual_market_qty,
            );
        emit_pending_target_gate_debug(
            event_tx,
            session,
            "strategy loop",
            pending_target_qty,
            actual_market_qty,
            reached,
            overshot,
            has_live_broker_path,
            waiting_for_position_sync,
        );
        if reached || overshot {
            session.execution_runtime.pending_target_qty = None;
            force_reevaluate_pending_window(session);
            let next_summary = if reached {
                format!("Position confirmed at target {actual_market_qty}; re-evaluating.")
            } else {
                format!(
                    "Position at {actual_market_qty} (target was {pending_target_qty}); re-evaluating."
                )
            };
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution pending target settled",
            );
            session.execution_runtime.last_summary = next_summary;
            emit_execution_state(event_tx, session);
        } else if pending_target_qty != 0 && !has_live_broker_path {
            if waiting_for_position_sync {
                let next_summary = format!(
                    "Waiting for position sync after automated order settle (actual {actual_market_qty}, pending target {pending_target_qty})."
                );
                emit_execution_transition_debug(
                    event_tx,
                    session,
                    &next_summary,
                    "execution market position sync wait",
                );
                session.execution_runtime.last_summary = next_summary;
                emit_execution_state(event_tx, session);
                return Ok(());
            }

            clear_stale_pending_target(session, pending_target_qty, actual_market_qty, event_tx);
            emit_execution_state(event_tx, session);
            return Ok(());
        } else {
            let next_summary = format!(
                "Waiting for prior automated order to settle (actual {actual_market_qty}, pending target {pending_target_qty})."
            );
            emit_execution_transition_debug(
                event_tx,
                session,
                &next_summary,
                "execution pending target wait",
            );
            session.execution_runtime.last_summary = next_summary;
            emit_execution_state(event_tx, session);
            return Ok(());
        }
    }

    if selected_managed_protection_waiting_for_position_sync(session) {
        let next_summary = format!(
            "Waiting for broker sync after native protection activity (position still {actual_market_qty})."
        );
        emit_execution_transition_debug(
            event_tx,
            session,
            &next_summary,
            "execution native protection sync wait",
        );
        session.execution_runtime.last_summary = next_summary;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);
        session.execution_runtime.last_closed_bar_fingerprint =
            latest_strategy_bar_fingerprint(session);
        seed_hma_cross_observed_side(session);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current {}; waiting for next update.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
        let latest_fingerprint = latest_strategy_bar_fingerprint(session);
        if session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts) {
            if session.execution_config.native_strategy != NativeStrategyKind::HmaCross
                || session.execution_runtime.last_closed_bar_fingerprint == latest_fingerprint
            {
                let reason =
                    if session.execution_config.native_strategy == NativeStrategyKind::HmaCross {
                        "closed-bar fingerprint unchanged"
                    } else {
                        "closed-bar timing waiting for next bar"
                    };
                let gate_detail = format!(
                    "strategy gate | {} | {reason} | last_bar_ts {} | fingerprint {:?} | actual_qty {} | {}",
                    active_native_slug(session),
                    last_strategy_ts,
                    latest_fingerprint,
                    actual_market_qty,
                    guarded_strategy_eval_context(session, actual_market_qty)
                );
                let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
                    session,
                    TradovateStrategyDecisionDebug {
                        path: "guarded",
                        decision: "blocked",
                        signal: None,
                        bar_ts: Some(last_strategy_ts),
                        actual_qty: actual_market_qty,
                        effective_qty: effective_market_position_qty(session),
                        target_qty: None,
                        strategy_detail: "n/a",
                        gate_detail,
                        fingerprint: latest_fingerprint,
                    },
                )));
                return Ok(());
            }

            let gate_detail = format!(
                "strategy closed-bar revision | {} | same timestamp fingerprint changed | bar_ts {} | previous_fingerprint {:?} | latest_fingerprint {:?} | actual_qty {} | {}",
                active_native_slug(session),
                last_strategy_ts,
                session.execution_runtime.last_closed_bar_fingerprint,
                latest_fingerprint,
                actual_market_qty,
                guarded_strategy_eval_context(session, actual_market_qty)
            );
            let _ = event_tx.send(ServiceEvent::DebugLog(format_tradovate_strategy_decision(
                session,
                TradovateStrategyDecisionDebug {
                    path: "guarded",
                    decision: "closed-bar revision",
                    signal: None,
                    bar_ts: Some(last_strategy_ts),
                    actual_qty: actual_market_qty,
                    effective_qty: effective_market_position_qty(session),
                    target_qty: None,
                    strategy_detail: "n/a",
                    gate_detail,
                    fingerprint: latest_fingerprint,
                },
            )));
        }
        session.execution_runtime.last_closed_bar_fingerprint = latest_fingerprint;
    }
    let previous_strategy_ts = session.execution_runtime.last_closed_bar_ts;
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let current_qty = effective_market_position_qty(session);
    let (signal_bar, signal, summary, debug_summary) = {
        let bars = signal_evaluation_bars(session).to_vec();
        if bars.is_empty() {
            bail!("latest strategy bar disappeared during strategy evaluation");
        }
        evaluate_active_execution_strategy_since_mut(
            session,
            &bars,
            current_qty,
            previous_strategy_ts,
        )
    };
    let protection_bar = strategy_bars(session)
        .last()
        .cloned()
        .unwrap_or_else(|| signal_bar.clone());

    if let Some(window) = session_window_at(session, protection_bar.ts_ns) {
        if window.hold_entries {
            if actual_market_qty != 0 {
                emit_guarded_strategy_eval_debug(
                    event_tx,
                    session,
                    "session hold flattening position",
                    signal,
                    signal_bar.ts_ns,
                    actual_market_qty,
                    current_qty,
                    Some(0),
                    &debug_summary,
                );
                if !native_order_strategy_enabled(session) {
                    sync_native_protection(
                        session,
                        broker_tx,
                        0,
                        None,
                        None,
                        &format!("{} session auto-close", active_native_slug(session)),
                    )?;
                }
                let reason = if window.session_open {
                    format!(
                        "{} session auto-close {:.0}m before {} close",
                        active_native_slug(session),
                        window.minutes_to_close.unwrap_or_default(),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                } else {
                    format!(
                        "{} session hold until {} reopen",
                        active_native_slug(session),
                        session
                            .market
                            .session_profile
                            .map(|profile| profile.label())
                            .unwrap_or("session")
                    )
                };
                match dispatch_target_position_order(session, broker_tx, 0, true, &reason)? {
                    MarketOrderDispatchOutcome::NoOp { message } => {
                        let _ = event_tx.send(ServiceEvent::Status(message));
                    }
                    MarketOrderDispatchOutcome::Queued { target_qty } => {
                        session.execution_runtime.pending_target_qty = target_qty;
                    }
                }
                session.execution_runtime.last_summary = if window.session_open {
                    format!(
                        "Session hold active; flattening {} {:.0}m before close.",
                        actual_market_qty,
                        window.minutes_to_close.unwrap_or_default()
                    )
                } else {
                    format!(
                        "Session closed; flattening {} and holding until reopen.",
                        actual_market_qty
                    )
                };
                emit_execution_state(event_tx, session);
                return Ok(());
            }

            emit_guarded_strategy_eval_debug(
                event_tx,
                session,
                "session hold blocked entries",
                signal,
                signal_bar.ts_ns,
                actual_market_qty,
                current_qty,
                target_qty_for_signal(signal, current_qty, session.execution_config.order_qty),
                &debug_summary,
            );
            sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
            session.execution_runtime.last_summary = if window.session_open {
                format!(
                    "Session hold active; no new entries with {:.0}m to close.",
                    window.minutes_to_close.unwrap_or_default()
                )
            } else {
                "Session closed; holding flat until reopen.".to_string()
            };
            emit_execution_state(event_tx, session);
            return Ok(());
        }
    }

    session.execution_runtime.last_summary = summary.clone();

    let Some(target_qty) =
        target_qty_for_signal(signal, current_qty, session.execution_config.order_qty)
    else {
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "no target",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            None,
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    };

    if target_qty == current_qty {
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "target already current",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if closed_bar_signal_already_dispatched(session, signal_bar.ts_ns) {
        session.execution_runtime.last_summary = format!(
            "Signal on closed bar {} already dispatched; waiting for a new signal bar.",
            signal_bar.ts_ns
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "closed-bar already dispatched",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    if entry_signal_consumed_while_flat(session, signal, current_qty) {
        session.execution_runtime.last_summary = format!(
            "{} entry side already dispatched while flat; waiting for the opposite entry signal before another {}.",
            signal.label(),
            signal.label()
        );
        emit_guarded_strategy_eval_debug(
            event_tx,
            session,
            "flat entry side already consumed",
            signal,
            signal_bar.ts_ns,
            actual_market_qty,
            current_qty,
            Some(target_qty),
            &debug_summary,
        );
        sync_execution_protection(session, broker_tx, Some(&protection_bar))?;
        emit_execution_state(event_tx, session);
        return Ok(());
    }

    emit_guarded_strategy_eval_debug(
        event_tx,
        session,
        "dispatching",
        signal,
        signal_bar.ts_ns,
        actual_market_qty,
        current_qty,
        Some(target_qty),
        &debug_summary,
    );

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} on {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        active_signal_timing_label(session),
        current_qty,
        target_qty
    )));

    if current_qty != 0 && !native_order_strategy_enabled(session) {
        sync_native_protection(
            session,
            broker_tx,
            0,
            None,
            None,
            &format!(
                "{} target transition {} -> {}",
                active_native_slug(session),
                current_qty,
                target_qty
            ),
        )?;
    }

    let reason = format!(
        "{} {} on {} | {}",
        active_native_slug(session),
        signal.label(),
        active_signal_timing_label(session),
        summary
    );
    let signal_context = PendingSignalLatencyContext {
        started_at: time::Instant::now(),
        description: format!(
            "{} {} (qty {} -> {})",
            active_native_slug(session),
            signal.label(),
            current_qty,
            target_qty
        ),
    };
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "signal | {} | {}",
        signal_context.description, summary
    )));
    session.pending_signal_context = Some(signal_context);
    let dispatch_outcome =
        match dispatch_target_position_order(session, broker_tx, target_qty, true, &reason) {
            Ok(outcome) => outcome,
            Err(err) => {
                session.pending_signal_context = None;
                return Err(err);
            }
        };
    match dispatch_outcome {
        MarketOrderDispatchOutcome::NoOp { message } => {
            session.pending_signal_context = None;
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            mark_closed_bar_signal_dispatched(session, signal_bar.ts_ns, signal);
        }
    }
    emit_execution_state(event_tx, session);
    Ok(())
}

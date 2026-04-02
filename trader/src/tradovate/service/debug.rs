use super::*;

pub(super) fn emit_debug_logs_from_latency_delta(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    previous: LatencySnapshot,
    current: LatencySnapshot,
) {
    emit_debug_latency_stage(
        event_tx,
        session,
        "seen",
        previous.last_order_seen_ms,
        current.last_order_seen_ms,
    );
    emit_debug_latency_stage(
        event_tx,
        session,
        "ack",
        previous.last_exec_report_ms,
        current.last_exec_report_ms,
    );
    emit_debug_latency_stage(
        event_tx,
        session,
        "fill",
        previous.last_fill_ms,
        current.last_fill_ms,
    );
}

fn emit_debug_latency_stage(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    stage: &str,
    previous: Option<u64>,
    current: Option<u64>,
) {
    if current.is_none() || previous == current {
        return;
    }
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "{stage} {}{} | {}",
        format_debug_latency_ms(current.unwrap_or_default()),
        debug_signal_latency_suffix(
            session
                .order_latency_tracker
                .as_ref()
                .and_then(|tracker| tracker.signal_started_at)
                .map(|started_at| started_at.elapsed().as_millis() as u64),
            session
                .order_latency_tracker
                .as_ref()
                .and_then(|tracker| tracker.signal_context.as_deref()),
        ),
        debug_tracker_context(session.order_latency_tracker.as_ref(), session)
    )));
}

pub(super) fn format_debug_latency_ms(value: u64) -> String {
    if value >= 60_000 {
        format!("{:.1}m", value as f64 / 60_000.0)
    } else if value >= 1_000 {
        format!("{:.1}s", value as f64 / 1_000.0)
    } else {
        format!("{value}ms")
    }
}

pub(super) fn debug_signal_latency_suffix(
    signal_latency_ms: Option<u64>,
    signal_context: Option<&str>,
) -> String {
    let Some(signal_latency_ms) = signal_latency_ms else {
        return String::new();
    };
    let mut suffix = format!(" | signal {}", format_debug_latency_ms(signal_latency_ms));
    if let Some(signal_context) = signal_context {
        suffix.push_str(&format!(" [{signal_context}]"));
    }
    suffix
}

fn debug_tracker_context(tracker: Option<&OrderLatencyTracker>, session: &SessionState) -> String {
    let mut parts = Vec::new();

    if let Some(contract) = session.selected_contract.as_ref() {
        parts.push(contract.name.clone());
    }
    if let Some(account_name) = session
        .selected_account_id
        .and_then(|selected_id| {
            session
                .accounts
                .iter()
                .find(|account| account.id == selected_id)
        })
        .map(|account| account.name.clone())
    {
        parts.push(format!("on {account_name}"));
    }

    if let Some(tracker) = tracker {
        parts.push(format!("[request {}]", tracker.cl_ord_id));
        if let Some(order_id) = tracker.order_id {
            parts.push(format!("(order {order_id})"));
        }
        if let Some(order_strategy_id) = tracker.order_strategy_id {
            parts.push(format!("(strategy {order_strategy_id})"));
        }
    }

    if parts.is_empty() {
        "selected market".to_string()
    } else {
        parts.join(" ")
    }
}

use super::debug::{
    debug_signal_latency_suffix, emit_debug_logs_from_latency_delta, emit_service_debug_log,
    emit_service_operational_status, format_debug_latency_ms,
};
use super::*;

const SNAPSHOT_IN_FLIGHT_MASK: u64 = 1 << 63;
const SNAPSHOT_REVISION_MASK: u64 = !SNAPSHOT_IN_FLIGHT_MASK;

pub(super) async fn handle_internal(
    internal: InternalEvent,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: InternalEventSender,
) -> Result<()> {
    match internal {
        InternalEvent::UserEntities(entities) => {
            handle_user_entities(entities, state, event_tx, internal_tx)?
        }
        InternalEvent::SnapshotsBuilt {
            generation,
            revision,
            snapshots,
        } => {
            if generation != state.snapshot_generation {
                // A detached build from a previous connection/replay session
                // must not consume or await the current generation's task.
                return Ok(());
            }
            // A completion must match the one active build. This protects the
            // owned current task if an older completion was delayed in a
            // queue; it also means a stale result can never be published
            // after a newer build.
            let Some(current_revision) =
                accept_snapshot_completion(&mut state.snapshot_revision, revision)
            else {
                return Ok(());
            };
            // The builder sends its completion event before returning. Await
            // the owned handle here so no snapshot clone/scan can overlap a
            // session reset or remain detached after this event is handled.
            if let Some(task) = state.snapshot_task.take() {
                let _ = task.await;
            }
            state.snapshot_revision = current_revision;
            let refresh_pending = std::mem::take(&mut state.snapshot_refresh_pending);
            if state.session.is_some() {
                let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
            }
            if refresh_pending && state.session.is_some() {
                // Publish useful progress first, then build once from the
                // latest store state. This bounded handoff avoids starvation
                // while preserving build/completion order.
                request_snapshot_refresh(state, &internal_tx);
            }
        }
        InternalEvent::SnapshotsBuildFailed {
            generation,
            revision,
            failure,
        } => {
            if generation != state.snapshot_generation {
                // A detached failure from a previous connection/replay
                // session must not consume or await the current generation's
                // task, release its reservation, or report a stale error.
                return Ok(());
            }
            let Some(current_revision) =
                accept_snapshot_completion(&mut state.snapshot_revision, revision)
            else {
                // A stale completion must not touch the current connection's
                // reservation or trigger a refresh from an old session.
                return Ok(());
            };
            // The worker has already completed its bounded build attempt. A
            // failure completion is the only path available when the normal
            // SnapshotsBuilt event could not be admitted. Await the owned
            // task so no build remains detached, then release its reservation.
            if let Some(task) = state.snapshot_task.take() {
                let _ = task.await;
            }
            state.snapshot_revision = current_revision;
            let refresh_pending = std::mem::take(&mut state.snapshot_refresh_pending);

            let _ = event_tx.send(ServiceEvent::Error(format!(
                "account snapshot refresh delivery failed (generation {generation}, revision {revision}): {}",
                failure.description()
            )));
            if refresh_pending && state.session.is_some() {
                // Preserve the existing coalescing contract: a request that
                // arrived while the failed build was running gets one fresh
                // build from current service state.
                request_snapshot_refresh(state, &internal_tx);
            }
        }
        InternalEvent::RestLatencyMeasured(rest_rtt_ms) => {
            state.latency.rest_rtt_ms = Some(rest_rtt_ms);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::UserSocketStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::Market(update) => {
            handle_market_update(update, state, event_tx, market_tx, internal_tx)?
        }
        #[cfg(feature = "replay")]
        InternalEvent::ReplayMarket {
            update,
            response_tx,
        } => {
            let result = handle_market_update(update, state, event_tx, market_tx, internal_tx);
            let response = result.as_ref().map(|_| ()).map_err(ToString::to_string);
            let _ = response_tx.send(response);
        }
        #[cfg(feature = "replay")]
        InternalEvent::ReplayBarrier(response_tx) => {
            let _ = response_tx.send(());
        }
        #[cfg(feature = "replay")]
        InternalEvent::ReplayCompleted { error } => {
            persist_replay_result(error, state, event_tx)?;
        }
        InternalEvent::BrokerOrderAck(ack) => {
            handle_broker_order_ack(ack, state, event_tx, internal_tx)
        }
        InternalEvent::BrokerOrderFailed(failure) => {
            handle_broker_order_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::OrderStrategyAck(ack) => {
            handle_order_strategy_ack(ack, state, event_tx, internal_tx)
        }
        InternalEvent::OrderStrategyFailed(failure) => {
            handle_order_strategy_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::ProtectionSyncApplied(ack) => {
            handle_protection_sync_applied(ack, state, event_tx, internal_tx)?
        }
        InternalEvent::ProtectionSyncFailed(failure) => {
            handle_protection_sync_failed(failure, state, event_tx, internal_tx)?
        }
        InternalEvent::PendingTargetWatchdog => handle_pending_target_watchdog(state, event_tx)?,
        InternalEvent::Error(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
    }
    Ok(())
}

fn accept_snapshot_completion(
    snapshot_revision: &mut u64,
    completion_revision: u64,
) -> Option<u64> {
    if *snapshot_revision & SNAPSHOT_IN_FLIGHT_MASK == 0 {
        return None;
    }
    let current_revision = *snapshot_revision & SNAPSHOT_REVISION_MASK;
    (completion_revision == current_revision).then_some(current_revision)
}

#[cfg(feature = "replay")]
fn persist_replay_result(
    error: Option<String>,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state
        .session
        .as_ref()
        .filter(|session| session.replay_enabled)
    else {
        return Ok(());
    };
    let Some(replay) = state.replay.as_ref() else {
        return Ok(());
    };

    let config = session.cfg.clone();
    let market = session.market.clone();
    let strategy = session.execution_config.clone();
    let bar_type = session.bar_type;
    let candle_mode = session.candle_mode;
    let signal_diagnostics = session.cfg.replay_signal_diagnostics.then_some(
        session
            .execution_runtime
            .replay_signal_diagnostics
            .as_slice(),
    );
    let ledger = state.replay_execution_ledger.snapshot().clone();
    let completed_at_utc = Utc::now();
    let run_id = session
        .engine_run
        .as_ref()
        .map(|run| run.run_id.clone())
        .unwrap_or_else(|| {
            format!(
                "replay-{}-{}",
                completed_at_utc.timestamp_millis(),
                std::process::id()
            )
        });
    let started_at_utc = session
        .engine_run
        .as_ref()
        .map(|run| run.started_at_utc)
        .or_else(|| {
            market.bars.first().and_then(|bar| {
                DateTime::<Utc>::from_timestamp(
                    bar.ts_ns.div_euclid(1_000_000_000),
                    bar.ts_ns.rem_euclid(1_000_000_000) as u32,
                )
            })
        })
        .unwrap_or(completed_at_utc);

    let outcome = replay::write_replay_result(replay::ReplayResultInput {
        config: &config,
        replay,
        market: &market,
        ledger: &ledger,
        strategy: &strategy,
        bar_type,
        candle_mode,
        run_id: &run_id,
        started_at_utc,
        completed_at_utc,
        error: error.as_deref(),
        signal_diagnostics,
    })?;
    let result_status = if error.is_some() {
        "failed"
    } else {
        "completed"
    };
    let status_label = if error.is_some() {
        "failed"
    } else {
        "complete"
    };
    let margin_suffix = match (
        outcome.required_starting_capital,
        outcome.initial_capital_sufficient,
    ) {
        (Some(required), Some(sufficient)) => format!(
            "; required capital {:.2} (selected capital {})",
            required,
            if sufficient {
                "sufficient"
            } else {
                "insufficient"
            }
        ),
        _ => String::new(),
    };
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Replay {status_label}; result saved to {} ({} fills, {} trades){}",
        outcome.result_path.display(),
        outcome.fill_count,
        outcome.trade_count,
        margin_suffix
    )));
    let _ = event_tx.send(ServiceEvent::ReplayResultSaved {
        run_id,
        result_path: outcome.result_path,
        status: result_status.to_string(),
        fill_count: outcome.fill_count,
        trade_count: outcome.trade_count,
    });
    Ok(())
}

fn handle_user_entities(
    entities: Vec<EntityEnvelope>,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    #[cfg(feature = "replay")]
    let replay_ledger_changed = state
        .session
        .as_ref()
        .filter(|session| session.replay_enabled)
        .map(|session| replay::ReplayLedgerMarketContext {
            contract_name: session
                .selected_contract
                .as_ref()
                .map(|contract| contract.name.clone())
                .or_else(|| session.market.contract_name.clone())
                .unwrap_or_default(),
            tick_size: session.market.tick_size,
            value_per_point: session.market.value_per_point,
        })
        .map(|context| {
            state
                .replay_execution_ledger
                .append_entities(&entities, &context)
                > 0
        })
        .unwrap_or(false);
    let history_entities_changed = entities.iter().any(|envelope| {
        matches!(
            envelope.entity_type.to_ascii_lowercase().as_str(),
            "order"
                | "command"
                | "orderstrategy"
                | "orderstrategylink"
                | "executionreport"
                | "fill"
                | "fillfee"
        )
    });
    let mut latency_changed = false;
    let mut trade_markers_changed = false;
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        for envelope in &entities {
            let previous_latency = state.latency;
            latency_changed |= update_latency_from_envelope(session, &mut state.latency, envelope);
            emit_debug_logs_from_latency_delta(event_tx, session, previous_latency, state.latency);
            session.user_store.apply(envelope.clone());
        }
        if let Some(settlement) = settle_replay_protected_exit(session, &entities) {
            let summary = format!(
                "Replay protected exit settled: {} for strategy {} on contract {}.",
                settlement.reason, settlement.order_strategy_id, settlement.contract_id,
            );
            session.execution_runtime.last_summary = summary.clone();
            let _ = event_tx.send(ServiceEvent::Status(summary));
            emit_service_debug_log(event_tx, Some(session), || {
                format!(
                    "replay protected exit settlement | reason {} | strategy {} | account {} | contract {} | lifecycle tracker and pending target released",
                    settlement.reason,
                    settlement.order_strategy_id,
                    settlement.account_id,
                    settlement.contract_id,
                )
            });
            emit_execution_state(event_tx, session);
        }
        let broker_rejections = collect_new_broker_rejections(session);
        for envelope in &entities {
            if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                continue;
            }
            if let Some(marker) = trade_marker_from_fill(session, &envelope.entity) {
                let emit_fill_detail =
                    fill_matches_active_latency_tracker(session, &envelope.entity);
                let fill_detail =
                    emit_fill_detail.then(|| fill_debug_detail(session, &marker, &envelope.entity));
                if record_trade_marker(session, marker) {
                    trade_markers_changed = true;
                    if let Some(detail) = fill_detail {
                        emit_service_debug_log(event_tx, Some(session), || {
                            format!("fill detail | {detail}")
                        });
                    }
                }
            }
        }
        if trade_markers_changed {
            let _ = event_tx.send(ServiceEvent::TradeMarkersUpdated(
                session.market.trade_markers.clone(),
            ));
        }
        match session.execution_config.native_execution_path {
            NativeExecutionPath::SimpleDiagnostic | NativeExecutionPath::HmaDirect => {
                handle_simple_execution_account_sync(session, event_tx);
            }
            NativeExecutionPath::Guarded => {
                handle_execution_account_sync(session, &broker_tx, event_tx)?;
            }
        }
        for rejection in broker_rejections {
            apply_broker_rejection(session, event_tx, rejection);
        }
        if history_entities_changed {
            refresh_engine_history(session);
            emit_engine_history(event_tx, session);
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    if latency_changed {
        let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    }
    #[cfg(feature = "replay")]
    if replay_ledger_changed {
        let _ = event_tx.send(ServiceEvent::ReplayExecutionLedgerUpdated(
            state.replay_execution_ledger.summary(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct BrokerRejectionNotice {
    pub(super) message: String,
    pub(super) debug: String,
    pub(super) affects_active_submission: bool,
    pub(super) affects_active_strategy: bool,
    pub(super) matches_selected_instrument: bool,
}

fn collect_new_broker_rejections(session: &mut SessionState) -> Vec<BrokerRejectionNotice> {
    let pending = session
        .user_store
        .command_reports
        .iter()
        .filter(|(report_id, report)| {
            !session
                .user_store
                .reported_command_rejections
                .contains(report_id)
                && command_report_is_rejection(report)
        })
        .map(|(report_id, report)| (*report_id, report.clone()))
        .collect::<Vec<_>>();

    pending
        .into_iter()
        .map(|(report_id, report)| {
            let notice = broker_rejection_notice(session, report_id, &report);
            session
                .user_store
                .reported_command_rejections
                .insert(report_id);
            notice
        })
        .collect()
}

fn command_report_is_rejection(report: &Value) -> bool {
    report
        .get("commandStatus")
        .and_then(Value::as_str)
        .is_some_and(|status| {
            matches!(
                status.trim().to_ascii_lowercase().as_str(),
                "riskrejected" | "executionrejected"
            )
        })
        || report
            .get("ordStatus")
            .and_then(Value::as_str)
            .is_some_and(|status| status.eq_ignore_ascii_case("Rejected"))
}

pub(super) fn broker_rejection_notice(
    session: &SessionState,
    report_id: i64,
    report: &Value,
) -> BrokerRejectionNotice {
    let command_id = json_i64(report, "commandId");
    let command = command_id.and_then(|id| session.user_store.commands.get(&id));
    let order_id = json_i64(report, "orderId")
        .or_else(|| command.and_then(|command| json_i64(command, "orderId")))
        .or(command_id);
    let order = order_id.and_then(|id| session.user_store.find_order_by_id(id));
    let account_id = order.and_then(|order| extract_account_id("order", order));
    let contract_id = order.and_then(|order| json_i64(order, "contractId"));
    let action = order
        .and_then(|order| order.get("action"))
        .and_then(Value::as_str);
    let order_qty = order
        .and_then(|order| pick_number(order, &["orderQty", "qty", "quantity"]))
        .map(|qty| qty.abs().round() as i32)
        .filter(|qty| *qty > 0);
    let account_name = account_id.and_then(|account_id| {
        session
            .accounts
            .iter()
            .find(|account| account.id == account_id)
            .map(|account| account.name.as_str())
    });
    let contract_name = contract_id
        .and_then(|contract_id| {
            session
                .selected_contract
                .as_ref()
                .filter(|contract| contract.id == contract_id)
                .map(|contract| contract.name.as_str())
        })
        .or_else(|| {
            order.and_then(|order| {
                order
                    .get("symbol")
                    .or_else(|| order.get("contractName"))
                    .and_then(Value::as_str)
            })
        });

    let mut subject = "Broker rejected".to_string();
    if let Some(action) = action {
        subject.push(' ');
        subject.push_str(action);
    }
    if let Some(order_qty) = order_qty {
        subject.push(' ');
        subject.push_str(&order_qty.to_string());
    }
    if let Some(contract_name) = contract_name {
        subject.push(' ');
        subject.push_str(contract_name);
    } else if let Some(contract_id) = contract_id {
        subject.push_str(&format!(" contract {contract_id}"));
    } else if let Some(order_id) = order_id {
        subject.push_str(&format!(" order {order_id}"));
    }
    if let Some(account_name) = account_name {
        subject.push_str(" on ");
        subject.push_str(account_name);
    } else if let Some(account_id) = account_id {
        subject.push_str(&format!(" on account {account_id}"));
    }

    let command_status = report
        .get("commandStatus")
        .and_then(Value::as_str)
        .unwrap_or("Rejected");
    let reject_reason = report
        .get("rejectReason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.trim().is_empty());
    let broker_text = report
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty());
    let reason = reject_reason.unwrap_or(command_status);
    let mut message = format!("{subject}: {reason}");
    if let Some(broker_text) = broker_text.filter(|text| !text.eq_ignore_ascii_case(reason)) {
        message.push_str(" — ");
        message.push_str(broker_text);
    }

    let key = account_id
        .zip(contract_id)
        .map(|(account_id, contract_id)| StrategyProtectionKey {
            account_id,
            contract_id,
        });
    let strategy_match = key.is_some_and(|key| {
        session
            .active_order_strategy
            .as_ref()
            .is_some_and(|tracked| tracked.key == key)
    });
    let matches_selected_instrument = key.is_some_and(|key| {
        session.selected_account_id == Some(key.account_id)
            && session
                .selected_contract
                .as_ref()
                .is_some_and(|contract| contract.id == key.contract_id)
    });
    let tracked_order_match = session
        .order_latency_tracker
        .as_ref()
        .is_some_and(|tracker| {
            tracker
                .order_id
                .is_some_and(|tracked_id| Some(tracked_id) == order_id)
                || command
                    .and_then(|command| command.get("clOrdId"))
                    .and_then(Value::as_str)
                    .is_some_and(|cl_ord_id| cl_ord_id == tracker.cl_ord_id)
                || order
                    .and_then(|order| order.get("clOrdId"))
                    .and_then(Value::as_str)
                    .is_some_and(|cl_ord_id| cl_ord_id == tracker.cl_ord_id)
        });
    let debug = format!(
        "broker rejection | command report {report_id} | command {} | order {} | account {} | contract {} | status {command_status} | reason {} | {}",
        command_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        order_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        account_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        contract_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string()),
        reject_reason.unwrap_or("none"),
        broker_text.unwrap_or("no broker text"),
    );

    BrokerRejectionNotice {
        message,
        debug,
        affects_active_submission: strategy_match || tracked_order_match,
        affects_active_strategy: strategy_match,
        matches_selected_instrument,
    }
}

pub(super) fn apply_broker_rejection(
    session: &mut SessionState,
    event_tx: &ServiceEventSender,
    rejection: BrokerRejectionNotice,
) {
    if rejection.affects_active_submission {
        session.order_submit_in_flight = false;
        session.order_latency_tracker = None;
        session.execution_runtime.pending_target_qty = None;
        if rejection.affects_active_strategy {
            clear_selected_order_strategy_state(session);
        }
        session.execution_runtime.last_summary = rejection.message.clone();
        emit_execution_state(event_tx, session);
    }
    emit_service_debug_log(event_tx, Some(session), || rejection.debug);
    if rejection.matches_selected_instrument || rejection.affects_active_submission {
        let _ = event_tx.send(ServiceEvent::BrokerRejection(rejection.message));
    }
}

fn handle_market_update(
    update: MarketUpdate,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: InternalEventSender,
) -> Result<()> {
    if state.session.is_none() {
        return Ok(());
    }

    let replay_warmup_only = state.session.as_ref().is_some_and(|session| {
        session.replay_enabled
            && update.live_bars == 0
            && update.history_update == MarketHistoryUpdate::Snapshot
            && update
                .replay_window
                .as_ref()
                .is_some_and(|window| window.evaluation_rows_processed == 0)
    });
    let broker_tx = state.broker_tx.clone();
    let (display_snapshot, closed_bar_advanced, history_update, engine_history) = {
        let session = state.session.as_mut().expect("checked session above");
        let history_sequence = update.history_sequence;
        let history_update = update.history_update;
        let closed_bar_advanced = apply_market_update(&mut session.market, update);
        session.execution_runtime.market_update_sequence = Some(history_sequence);
        session.execution_runtime.market_update_kind = history_update;
        if replay_warmup_only {
            // Seed closed-bar timing at the warmup boundary.  This keeps the
            // first evaluation bar eligible without allowing the seed
            // snapshot itself to place an order.
            if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar {
                session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
                session.execution_runtime.last_closed_bar_fingerprint =
                    latest_strategy_bar_fingerprint(session);
            }
            session.execution_runtime.last_summary =
                "Replay warmup loaded; waiting for evaluation bars.".to_string();
        } else {
            match session.execution_config.native_execution_path {
                NativeExecutionPath::Guarded => {
                    maybe_run_execution_strategy(session, &broker_tx, event_tx)?;
                }
                NativeExecutionPath::SimpleDiagnostic => {
                    maybe_run_simple_execution_strategy(session, &broker_tx, event_tx)?;
                }
                NativeExecutionPath::HmaDirect => {
                    maybe_run_hma_direct_execution_strategy(session, &broker_tx, event_tx)?;
                }
            }
        }
        refresh_engine_history_mark(session);
        (
            display_market_snapshot(&session.market),
            closed_bar_advanced,
            history_update,
            session.engine_run.as_ref().map(|run| run.history.clone()),
        )
    };
    // Keep F6/account snapshots fresh at closed-bar cadence. Forming-bar
    // updates can arrive much faster than the bar interval (especially for
    // range bars), so do not clone and scan UserSyncStore for every quote.
    // The snapshot revision path coalesces a refresh that is still building;
    // corrections are included even when their timestamp did not advance.
    let headless_replay = state
        .session
        .as_ref()
        .is_some_and(|session| session.replay_enabled && session.cfg.replay_headless);
    if should_refresh_account_snapshot(closed_bar_advanced, history_update, headless_replay) {
        request_snapshot_refresh(state, &internal_tx);
    }
    let _ = market_tx.send(display_snapshot);
    if let Some(history) = engine_history {
        let _ = event_tx.send(ServiceEvent::EngineHistoryUpdated(history));
    }
    Ok(())
}

fn should_refresh_account_snapshot(
    closed_bar_advanced: bool,
    history_update: MarketHistoryUpdate,
    headless_replay: bool,
) -> bool {
    !headless_replay && (closed_bar_advanced || history_update == MarketHistoryUpdate::Correction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn account_snapshot_refresh_is_bounded_to_closed_bar_changes() {
        assert!(should_refresh_account_snapshot(
            true,
            MarketHistoryUpdate::Append,
            false,
        ));
        assert!(should_refresh_account_snapshot(
            false,
            MarketHistoryUpdate::Correction,
            false,
        ));
        assert!(!should_refresh_account_snapshot(
            false,
            MarketHistoryUpdate::Unchanged,
            false,
        ));
        assert!(!should_refresh_account_snapshot(
            false,
            MarketHistoryUpdate::Snapshot,
            false,
        ));
        assert!(!should_refresh_account_snapshot(
            true,
            MarketHistoryUpdate::Append,
            true,
        ));
    }

    #[test]
    fn snapshot_completion_requires_the_active_revision() {
        let mut snapshot_revision = 12 | SNAPSHOT_IN_FLIGHT_MASK;

        assert_eq!(accept_snapshot_completion(&mut snapshot_revision, 11), None);
        assert_eq!(snapshot_revision, 12 | SNAPSHOT_IN_FLIGHT_MASK);
        assert_eq!(
            accept_snapshot_completion(&mut snapshot_revision, 12),
            Some(12)
        );
    }

    #[test]
    fn stale_snapshot_completion_does_not_release_active_build() {
        let mut snapshot_revision = 13 | SNAPSHOT_IN_FLIGHT_MASK;
        assert_eq!(accept_snapshot_completion(&mut snapshot_revision, 12), None);
        assert_eq!(snapshot_revision, 13 | SNAPSHOT_IN_FLIGHT_MASK);
    }

    #[tokio::test]
    async fn failed_snapshot_completion_surfaces_error_and_releases_reservation() {
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (replay_speed_tx, _replay_speed_rx) =
            tokio::sync::watch::channel(ReplaySpeed::default());
        let mut state = ServiceState {
            client: Client::builder().build().expect("client"),
            broker_tx,
            broker_task: None,
            replay_speed_tx,
            replay_speed: ReplaySpeed::default(),
            replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
            session: None,
            replay: None,
            user_task: None,
            market_task: None,
            rest_probe_task: None,
            replay_lookup_job: None,
            replay_download_job: None,
            latency: LatencySnapshot::default(),
            snapshot_generation: 9,
            snapshot_revision: 5 | SNAPSHOT_IN_FLIGHT_MASK,
            snapshot_refresh_pending: false,
            snapshot_task: None,
        };
        let (event_tx, mut event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
        let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
        let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

        handle_internal(
            InternalEvent::SnapshotsBuildFailed {
                generation: 9,
                revision: 5,
                failure: SnapshotDeliveryFailure::QueueFull,
            },
            &mut state,
            &event_tx,
            &market_tx,
            internal_tx,
        )
        .await
        .expect("snapshot failure completion should be handled");

        assert_eq!(state.snapshot_revision, 5);
        assert!(matches!(
            event_rx.try_recv(),
            Ok(ServiceEvent::Error(message)) if message.contains("account snapshot refresh delivery failed")
                && message.contains("internal event queue is full")
        ));
    }

    #[tokio::test]
    async fn stale_snapshot_completion_does_not_consume_new_generation_task() {
        let (broker_tx, _broker_rx) = tokio::sync::mpsc::unbounded_channel();
        let (replay_speed_tx, _replay_speed_rx) =
            tokio::sync::watch::channel(ReplaySpeed::default());
        let new_generation_task = tokio::spawn(async {
            std::future::pending::<()>().await;
        });
        let mut state = ServiceState {
            client: Client::builder().build().expect("client"),
            broker_tx,
            broker_task: None,
            replay_speed_tx,
            replay_speed: ReplaySpeed::default(),
            replay_execution_ledger: replay::ReplayExecutionLedgerState::default(),
            session: None,
            replay: None,
            user_task: None,
            market_task: None,
            rest_probe_task: None,
            replay_lookup_job: None,
            replay_download_job: None,
            latency: LatencySnapshot::default(),
            snapshot_generation: 2,
            snapshot_revision: 8 | SNAPSHOT_IN_FLIGHT_MASK,
            snapshot_refresh_pending: false,
            snapshot_task: Some(new_generation_task),
        };
        let (event_tx, _event_rx) = service_event_channel(SERVICE_EVENT_QUEUE_CAPACITY);
        let (market_tx, _market_rx) = tokio::sync::watch::channel(MarketSnapshot::default());
        let (internal_tx, _internal_rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);

        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            handle_internal(
                InternalEvent::SnapshotsBuilt {
                    generation: 1,
                    revision: 7,
                    snapshots: Vec::new(),
                },
                &mut state,
                &event_tx,
                &market_tx,
                internal_tx.clone(),
            ),
        )
        .await
        .expect("stale snapshot completion must not await the new task")
        .expect("stale snapshot completion should be ignored");
        assert!(
            state.snapshot_task.is_some(),
            "the new generation task must remain owned after a stale success"
        );

        let task = state
            .snapshot_task
            .take()
            .expect("new task should be owned");
        task.abort();
        let _ = task.await;
        state.snapshot_task = Some(tokio::spawn(async {
            std::future::pending::<()>().await;
        }));

        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            handle_internal(
                InternalEvent::SnapshotsBuildFailed {
                    generation: 1,
                    revision: 7,
                    failure: SnapshotDeliveryFailure::QueueFull,
                },
                &mut state,
                &event_tx,
                &market_tx,
                internal_tx,
            ),
        )
        .await
        .expect("stale snapshot failure must not await the new task")
        .expect("stale snapshot failure should be ignored");
        assert!(
            state.snapshot_task.is_some(),
            "the new generation task must remain owned after a stale failure"
        );
        assert_eq!(state.snapshot_revision, 8 | SNAPSHOT_IN_FLIGHT_MASK);

        let task = state
            .snapshot_task
            .take()
            .expect("new task should be owned");
        task.abort();
        let _ = task.await;
    }
}

pub(super) fn handle_broker_order_ack(
    ack: BrokerOrderAck,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) {
    let mut signal_submit_ms = None;
    let mut signal_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if let Some(run) = session.engine_run.as_mut()
            && ack.cl_ord_id.starts_with(&run.order_prefix)
            && let Some(order_id) = ack.order_id
        {
            run.owned_order_ids.insert(order_id);
        }
        if let Some(tracker) = session.order_latency_tracker.as_mut() {
            if tracker.cl_ord_id == ack.cl_ord_id {
                tracker.order_id = ack.order_id;
                signal_submit_ms = tracker
                    .signal_started_at
                    .map(|started_at| started_at.elapsed().as_millis() as u64);
                signal_context = tracker.signal_context.clone();
            }
        }
    }

    apply_submit_latency(&mut state.latency, ack.submit_rtt_ms, signal_submit_ms);
    let ack_message = ack.message.clone();
    emit_service_operational_status(event_tx, state.session.as_ref(), || ack.message);
    emit_service_debug_log(event_tx, state.session.as_ref(), || {
        format!(
            "submit {}{} | endpoint {} | {}",
            format_debug_latency_ms(ack.submit_rtt_ms),
            debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
            ack.endpoint,
            ack_message
        )
    });
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    schedule_pending_target_watchdog(internal_tx);
}

fn handle_broker_order_failed(
    failure: BrokerOrderFailure,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let mut stale_interrupt_recovered = false;
    let mut observability_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if session
            .order_latency_tracker
            .as_ref()
            .is_some_and(|tracker| tracker.cl_ord_id == failure.cl_ord_id)
        {
            session.order_latency_tracker = None;
        }
        if let Some(target_qty) = failure.target_qty {
            if session.execution_runtime.pending_target_qty == Some(target_qty) {
                session.execution_runtime.pending_target_qty = None;
                if failure.stale_interrupt {
                    clear_selected_order_strategy_state(session);
                    session.execution_runtime.last_summary =
                        "Previous strategy was already inactive; retrying current signal after broker sync."
                            .to_string();
                    if let Some(last_closed_ts) = latest_strategy_bar_ts(session) {
                        session.execution_runtime.last_closed_bar_ts =
                            Some(last_closed_ts.saturating_sub(1));
                    }
                    stale_interrupt_recovered = true;
                    emit_execution_state(event_tx, session);
                } else {
                    session.execution_runtime.last_summary = failure.message.clone();
                    emit_execution_state(event_tx, session);
                }
            }
        }
        observability_context = Some(execution_observability_context(session));
    }

    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        emit_service_debug_log(event_tx, state.session.as_ref(), || {
            format!(
                "submit stale | {}",
                format_broker_order_failure_debug(&failure, observability_context.as_deref())
            )
        });
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        emit_service_debug_log(event_tx, state.session.as_ref(), || {
            format!(
                "submit failed | {}",
                format_broker_order_failure_debug(&failure, observability_context.as_deref())
            )
        });
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
}

fn format_broker_order_failure_debug(
    failure: &BrokerOrderFailure,
    observability_context: Option<&str>,
) -> String {
    let mut message = format!(
        "endpoint {} | clOrdId {} | target {:?} | {}",
        failure.endpoint, failure.cl_ord_id, failure.target_qty, failure.message
    );
    if let Some(context) = observability_context {
        message.push_str(" | ");
        message.push_str(context);
    }
    message
}

fn handle_order_strategy_ack(
    ack: BrokerOrderStrategyAck,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) {
    let mut signal_submit_ms = None;
    let mut signal_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if let Some(tracker) = session.order_latency_tracker.as_mut() {
            if tracker.cl_ord_id == ack.uuid {
                tracker.order_strategy_id = ack.order_strategy_id;
                signal_submit_ms = tracker
                    .signal_started_at
                    .map(|started_at| started_at.elapsed().as_millis() as u64);
                signal_context = tracker.signal_context.clone();
            }
        }
        if let Some(order_strategy_id) = ack.order_strategy_id {
            session.active_order_strategy = Some(TrackedOrderStrategy {
                key: ack.key,
                order_strategy_id,
                target_qty: ack.target_qty,
            });
        }
    }

    apply_submit_latency(&mut state.latency, ack.submit_rtt_ms, signal_submit_ms);
    let ack_message = ack.message.clone();
    emit_service_operational_status(event_tx, state.session.as_ref(), || ack.message);
    emit_service_debug_log(event_tx, state.session.as_ref(), || {
        format!(
            "submit {}{} | endpoint {} | {}",
            format_debug_latency_ms(ack.submit_rtt_ms),
            debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
            ack.endpoint,
            ack_message
        )
    });
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    schedule_pending_target_watchdog(internal_tx);
}

fn handle_order_strategy_failed(
    failure: BrokerOrderStrategyFailure,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let mut stale_interrupt_recovered = false;
    let mut observability_context = None;
    if let Some(session) = state.session.as_mut() {
        session.order_submit_in_flight = false;
        if session
            .order_latency_tracker
            .as_ref()
            .is_some_and(|tracker| tracker.cl_ord_id == failure.uuid)
        {
            session.order_latency_tracker = None;
        }
        if session.execution_runtime.pending_target_qty == Some(failure.target_qty) {
            session.execution_runtime.pending_target_qty = None;
        }
        if failure.stale_interrupt {
            clear_selected_order_strategy_state(session);
            session.execution_runtime.last_summary =
                "Previous strategy was already inactive; retrying current signal after broker sync."
                    .to_string();
            if let Some(last_closed_ts) = latest_strategy_bar_ts(session) {
                session.execution_runtime.last_closed_bar_ts =
                    Some(last_closed_ts.saturating_sub(1));
            }
            stale_interrupt_recovered = true;
            emit_execution_state(event_tx, session);
        } else if session.execution_runtime.pending_target_qty.is_none() {
            session.execution_runtime.last_summary = failure.message.clone();
            emit_execution_state(event_tx, session);
        }
        observability_context = Some(execution_observability_context(session));
    }

    if stale_interrupt_recovered {
        request_snapshot_refresh(state, &internal_tx);
        emit_service_debug_log(event_tx, state.session.as_ref(), || {
            format!(
                "submit stale | {}",
                format_order_strategy_failure_debug(&failure, observability_context.as_deref())
            )
        });
        let _ = event_tx.send(ServiceEvent::Status(failure.message));
    } else {
        emit_service_debug_log(event_tx, state.session.as_ref(), || {
            format!(
                "submit failed | {}",
                format_order_strategy_failure_debug(&failure, observability_context.as_deref())
            )
        });
        let _ = event_tx.send(ServiceEvent::Error(failure.message));
    }

    Ok(())
}

fn format_order_strategy_failure_debug(
    failure: &BrokerOrderStrategyFailure,
    observability_context: Option<&str>,
) -> String {
    let mut message = format!(
        "endpoint {} | uuid {} | target {} | {}",
        failure.endpoint, failure.uuid, failure.target_qty, failure.message
    );
    if let Some(context) = observability_context {
        message.push_str(" | ");
        message.push_str(context);
    }
    message
}

fn handle_protection_sync_applied(
    ack: ProtectionSyncAck,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        session.protection_sync_in_flight = false;
        match ack.next_state {
            Some(next_state) => {
                session.managed_protection.insert(ack.key, next_state);
            }
            None => {
                session.managed_protection.remove(&ack.key);
            }
        }

        if let Some(desired) = session.pending_protection_sync.take() {
            sync_native_protection_target(session, &broker_tx, desired)?;
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    if let Some(message) = ack.message {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }
    emit_service_debug_log(event_tx, state.session.as_ref(), || {
        format!("protection sync applied | endpoint {}", ack.endpoint)
    });
    Ok(())
}

fn handle_protection_sync_failed(
    failure: ProtectionSyncFailure,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            return Ok(());
        };
        session.protection_sync_in_flight = false;
        if let Some(desired) = session.pending_protection_sync.take() {
            sync_native_protection_target(session, &broker_tx, desired)?;
        }
    }
    request_snapshot_refresh(state, &internal_tx);
    emit_service_debug_log(event_tx, state.session.as_ref(), || {
        format!(
            "protection sync failed | endpoint {} | {}",
            failure.endpoint, failure.message
        )
    });
    let _ = event_tx.send(ServiceEvent::Error(failure.message));
    Ok(())
}

fn handle_pending_target_watchdog(
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        return Ok(());
    };
    let Some(pending) = session.execution_runtime.pending_target_qty else {
        return Ok(());
    };
    // A staged reversal uses target 0 for its flatten leg. Never clear that
    // lifecycle on a wall-clock timeout: an absent user-stream update does
    // not prove that the broker-owned strategy/flatten is gone, and forgetting
    // it could allow a duplicate order. Only an authoritative broker event
    // or an explicit user reconciliation may release this state.
    if pending == 0 || selected_contract_has_live_broker_path(session) {
        return Ok(());
    }

    let actual_qty = selected_market_position_qty(session);
    if should_wait_for_automated_position_sync(session, pending, actual_qty) {
        return Ok(());
    }
    clear_stale_pending_target(session, pending, actual_qty, event_tx);
    emit_execution_state(event_tx, session);
    Ok(())
}

fn apply_submit_latency(
    latency: &mut LatencySnapshot,
    submit_rtt_ms: u64,
    signal_submit_ms: Option<u64>,
) {
    latency.last_order_ack_ms = Some(submit_rtt_ms);
    latency.last_order_seen_ms = None;
    latency.last_exec_report_ms = None;
    latency.last_fill_ms = None;
    latency.last_signal_submit_ms = signal_submit_ms;
    latency.last_signal_seen_ms = None;
    latency.last_signal_ack_ms = None;
    latency.last_signal_fill_ms = None;
}

fn schedule_pending_target_watchdog(internal_tx: InternalEventSender) {
    tokio::spawn(async move {
        time::sleep(Duration::from_secs(PENDING_TARGET_WATCHDOG_DELAY_SECS)).await;
        let _ = internal_tx.send(InternalEvent::PendingTargetWatchdog);
    });
}

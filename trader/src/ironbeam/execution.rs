use super::account::{
    rebuild_account_snapshots, selected_has_live_entry_path, selected_market_entry_price,
    selected_market_position_qty,
};
use super::orders::{dispatch_target_position_order, sync_native_protection};
use super::state::{IronbeamSession, OrderDispatchOutcome};
use crate::broker::{Bar, InstrumentSessionWindow, LatencySnapshot, ServiceEvent};
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{NativeSignalTiming, NativeStrategyKind, StrategyKind};
use anyhow::{Context, Result};
use reqwest::Client;
use tokio::sync::mpsc::UnboundedSender;

const PENDING_TARGET_WATCHDOG_SECS: u64 = 3;

pub(super) async fn handle_execution_account_sync(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
) -> Result<()> {
    let actual_qty = selected_market_position_qty(session);
    let actual_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_qty, actual_entry);
    super::orders::refresh_managed_protection(session);

    if let Some(pending) = session.execution_runtime.pending_target_qty {
        if actual_qty == pending {
            session.execution_runtime.clear_pending_target();
            if !continue_staged_reversal(client, session, latency, internal_tx.clone()).await? {
                session.execution_runtime.last_summary =
                    format!("Position confirmed at target {actual_qty}");
            }
        } else if !selected_has_live_entry_path(session) {
            let timed_out = session
                .execution_runtime
                .pending_target_started_at
                .is_some_and(|started_at| {
                    started_at.elapsed().as_secs() >= PENDING_TARGET_WATCHDOG_SECS
                });
            if timed_out {
                session.execution_runtime.clear_pending_target();
                session.execution_runtime.last_closed_bar_ts =
                    latest_strategy_bar_ts(session).map(|last_ts| last_ts.saturating_sub(1));
                session.execution_runtime.last_summary = format!(
                    "Pending target {pending} cleared after Ironbeam order path went idle; re-evaluating."
                );
            }
        }
    }

    if session.execution_runtime.pending_target_qty.is_none() {
        let _ = continue_staged_reversal(client, session, latency, internal_tx.clone()).await?;
    }

    if session.execution_runtime.armed && session.execution_config.kind == StrategyKind::Native {
        sync_execution_protection(client, session, latency, internal_tx, None).await?;
    }

    rebuild_account_snapshots(session);
    Ok(())
}

pub(super) async fn maybe_run_execution_strategy(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }

    let actual_market_qty = selected_market_position_qty(session);
    let actual_market_entry = selected_market_entry_price(session);
    sync_active_execution_position(session, actual_market_qty, actual_market_entry);

    if session.execution_runtime.pending_target_qty.is_none()
        && continue_staged_reversal(client, session, latency, internal_tx.clone()).await?
    {
        rebuild_account_snapshots(session);
        return Ok(());
    }

    let max_automated_qty = session.execution_config.order_qty.max(1);
    if actual_market_qty.abs() > max_automated_qty {
        if selected_has_live_entry_path(session) {
            session.execution_runtime.last_summary = format!(
                "Waiting for Ironbeam broker sync: temporary position {actual_market_qty} exceeds max {max_automated_qty} while an order path is still active."
            );
        } else {
            disarm_execution_strategy(
                session,
                format!(
                    "Automation disarmed: position drifted to {actual_market_qty}, above configured max {max_automated_qty}."
                ),
            );
        }
        rebuild_account_snapshots(session);
        return Ok(());
    }

    if let Some(pending_target_qty) = session.execution_runtime.pending_target_qty {
        session.execution_runtime.last_summary = format!(
            "Waiting for prior Ironbeam order to settle (actual {actual_market_qty}, pending target {pending_target_qty})."
        );
        rebuild_account_snapshots(session);
        return Ok(());
    }

    let Some(last_strategy_ts) = latest_strategy_bar_ts(session) else {
        session.execution_runtime.last_summary = format!(
            "Native {} armed; waiting for first {}.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        rebuild_account_snapshots(session);
        return Ok(());
    };

    if session.execution_runtime.last_closed_bar_ts.is_none() {
        session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);
        session.execution_runtime.last_summary = format!(
            "Native {} anchored to current {}; waiting for next update.",
            active_native_label(session),
            active_signal_timing_label(session)
        );
        rebuild_account_snapshots(session);
        return Ok(());
    }

    if session.execution_config.native_signal_timing == NativeSignalTiming::ClosedBar
        && session.execution_runtime.last_closed_bar_ts == Some(last_strategy_ts)
    {
        return Ok(());
    }
    session.execution_runtime.last_closed_bar_ts = Some(last_strategy_ts);

    let current_qty = effective_market_position_qty(session);
    let (signal_bar, signal, summary) = {
        let bars = strategy_bars(session);
        let signal_bar = bars
            .last()
            .cloned()
            .context("latest Ironbeam strategy bar disappeared during evaluation")?;
        let (signal, summary) = evaluate_active_execution_strategy(session, bars, current_qty);
        (signal_bar, signal, summary)
    };

    if let Some(window) = session_window_at(session, signal_bar.ts_ns)
        && window.hold_entries
    {
        if actual_market_qty != 0 {
            match dispatch_target_position_order(
                client,
                session,
                latency,
                internal_tx.clone(),
                0,
                true,
                &session_hold_reason(session, window, actual_market_qty),
            )
            .await?
            {
                OrderDispatchOutcome::NoOp { message } => {
                    session.execution_runtime.last_summary = message.clone();
                    let _ = event_tx.send(ServiceEvent::Status(message));
                }
                OrderDispatchOutcome::Queued { target_qty } => {
                    session.execution_runtime.set_pending_target(target_qty);
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
                }
            }
            rebuild_account_snapshots(session);
            return Ok(());
        }

        sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar)).await?;
        session.execution_runtime.last_summary = if window.session_open {
            format!(
                "Session hold active; no new entries with {:.0}m to close.",
                window.minutes_to_close.unwrap_or_default()
            )
        } else {
            "Session closed; holding flat until reopen.".to_string()
        };
        rebuild_account_snapshots(session);
        return Ok(());
    }

    session.execution_runtime.last_summary = summary.clone();

    let Some(target_qty) =
        target_qty_for_signal(signal, current_qty, session.execution_config.order_qty)
    else {
        sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar)).await?;
        rebuild_account_snapshots(session);
        return Ok(());
    };

    if target_qty == current_qty {
        sync_execution_protection(client, session, latency, internal_tx, Some(&signal_bar)).await?;
        rebuild_account_snapshots(session);
        return Ok(());
    }

    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Strategy {} signal: {} (qty {} -> {})",
        active_native_slug(session),
        signal.label(),
        current_qty,
        target_qty
    )));

    let reason = format!(
        "{} {} | {}",
        active_native_slug(session),
        signal.label(),
        summary
    );
    match dispatch_target_position_order(
        client,
        session,
        latency,
        internal_tx,
        target_qty,
        true,
        &reason,
    )
    .await?
    {
        OrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message.clone();
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        OrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.set_pending_target(target_qty);
        }
    }
    rebuild_account_snapshots(session);
    Ok(())
}

pub(super) fn arm_execution_strategy(session: &mut IronbeamSession) {
    session.execution_runtime.clear_pending_target();
    session.execution_runtime.reset_execution();
    if session.execution_config.kind != StrategyKind::Native {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_closed_bar_ts = None;
        session.execution_runtime.last_summary =
            "Selected strategy is not an armed native runtime.".to_string();
        return;
    }

    session.execution_runtime.armed = true;
    session.execution_runtime.last_closed_bar_ts = latest_strategy_bar_ts(session);
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

pub(super) fn disarm_execution_strategy(session: &mut IronbeamSession, reason: String) {
    if !session.execution_runtime.armed && session.execution_runtime.last_summary == reason {
        return;
    }
    session.execution_runtime.armed = false;
    session.execution_runtime.clear_pending_target();
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary = reason;
}

fn session_hold_reason(
    session: &IronbeamSession,
    window: InstrumentSessionWindow,
    actual_market_qty: i32,
) -> String {
    if window.session_open {
        format!(
            "{} session auto-close {:.0}m before {} close (qty {})",
            active_native_slug(session),
            window.minutes_to_close.unwrap_or_default(),
            session
                .market
                .session_profile
                .map(|profile| profile.label())
                .unwrap_or("session"),
            actual_market_qty
        )
    } else {
        format!(
            "{} session hold until {} reopen (qty {})",
            active_native_slug(session),
            session
                .market
                .session_profile
                .map(|profile| profile.label())
                .unwrap_or("session"),
            actual_market_qty
        )
    }
}

async fn continue_staged_reversal(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
) -> Result<bool> {
    let Some(staged) = session.execution_runtime.pending_reversal_entry.clone() else {
        return Ok(false);
    };

    let actual_qty = selected_market_position_qty(session);
    if actual_qty.signum() == staged.target_qty.signum() && actual_qty != 0 {
        session.execution_runtime.pending_reversal_entry = None;
        session.execution_runtime.last_summary =
            format!("Staged reversal resolved at target {actual_qty}");
        return Ok(true);
    }

    if actual_qty == 0 {
        if selected_has_live_entry_path(session) {
            session.execution_runtime.last_summary = format!(
                "Staged reversal flat; waiting for Ironbeam order path to clear before entering {}.",
                staged.target_qty
            );
            return Ok(true);
        }

        match dispatch_target_position_order(
            client,
            session,
            latency,
            internal_tx,
            staged.target_qty,
            false,
            &staged.reason,
        )
        .await?
        {
            OrderDispatchOutcome::NoOp { message } => {
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = message;
            }
            OrderDispatchOutcome::Queued { target_qty } => {
                session.execution_runtime.set_pending_target(target_qty);
                session.execution_runtime.pending_reversal_entry = None;
                session.execution_runtime.last_summary = format!(
                    "Flat confirmed; submitting staged reversal entry to {}.",
                    staged.target_qty
                );
            }
        }
        return Ok(true);
    }

    match dispatch_target_position_order(
        client,
        session,
        latency,
        internal_tx,
        0,
        false,
        &format!(
            "{} | staged reversal flatten {} -> 0 before {}",
            staged.reason, actual_qty, staged.target_qty
        ),
    )
    .await?
    {
        OrderDispatchOutcome::NoOp { message } => {
            session.execution_runtime.last_summary = message;
        }
        OrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.set_pending_target(target_qty);
            session.execution_runtime.last_summary = format!(
                "Flattening {} before staged reversal to {}.",
                actual_qty, staged.target_qty
            );
        }
    }
    Ok(true)
}

fn closed_bars(session: &IronbeamSession) -> &[Bar] {
    let closed_len = session.market.history_loaded.min(session.market.bars.len());
    &session.market.bars[..closed_len]
}

fn strategy_bars(session: &IronbeamSession) -> &[Bar] {
    if session.execution_config.native_signal_timing == NativeSignalTiming::LiveBar {
        &session.market.bars
    } else {
        closed_bars(session)
    }
}

fn latest_strategy_bar_ts(session: &IronbeamSession) -> Option<i64> {
    strategy_bars(session).last().map(|bar| bar.ts_ns)
}

fn active_native_slug(session: &IronbeamSession) -> &'static str {
    session.execution_config.native_strategy.slug()
}

fn active_native_label(session: &IronbeamSession) -> &'static str {
    session.execution_config.native_strategy.label()
}

fn active_signal_timing_label(session: &IronbeamSession) -> &'static str {
    match session.execution_config.native_signal_timing {
        NativeSignalTiming::ClosedBar => "closed bar",
        NativeSignalTiming::LiveBar => "live bar",
    }
}

fn session_window_at(
    session: &IronbeamSession,
    ts_ns: i64,
) -> Option<crate::broker::InstrumentSessionWindow> {
    session
        .market
        .session_profile
        .map(|profile| profile.evaluate(ts_ns))
}

fn evaluate_active_execution_strategy(
    session: &IronbeamSession,
    bars: &[Bar],
    current_qty: i32,
) -> (StrategySignal, String) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            let evaluation = session
                .execution_config
                .native_hma
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
        NativeStrategyKind::EmaCross => {
            let evaluation = session
                .execution_config
                .native_ema
                .evaluate(bars, side_from_signed_qty(current_qty));
            (evaluation.signal, evaluation.summary())
        }
    }
}

pub(super) fn target_qty_for_signal(
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

fn effective_market_position_qty(session: &IronbeamSession) -> i32 {
    session
        .execution_runtime
        .pending_target_qty
        .unwrap_or_else(|| selected_market_position_qty(session))
}

fn sync_active_execution_position(
    session: &mut IronbeamSession,
    signed_qty: i32,
    entry_price: Option<f64>,
) {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session.execution_config.native_hma.sync_position(
            &mut session.execution_runtime.hma_execution,
            signed_qty,
            entry_price,
        ),
        NativeStrategyKind::EmaCross => session.execution_config.native_ema.sync_position(
            &mut session.execution_runtime.ema_execution,
            signed_qty,
            entry_price,
        ),
    }
}

fn active_native_uses_protection(session: &IronbeamSession) -> bool {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            session.execution_config.native_hma.uses_native_protection()
        }
        NativeStrategyKind::EmaCross => {
            session.execution_config.native_ema.uses_native_protection()
        }
    }
}

fn take_profit_price(session: &IronbeamSession, entry_price: f64, signed_qty: i32) -> Option<f64> {
    let side = side_from_signed_qty(signed_qty)?;
    let offset = match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => session
            .execution_config
            .native_hma
            .take_profit_offset(session.market.tick_size)?,
        NativeStrategyKind::EmaCross => session
            .execution_config
            .native_ema
            .take_profit_offset(session.market.tick_size)?,
    };
    Some(match side {
        crate::strategies::PositionSide::Long => entry_price + offset,
        crate::strategies::PositionSide::Short => entry_price - offset,
    })
}

fn combined_stop_price(session: &mut IronbeamSession, trailing_bar: Option<&Bar>) -> Option<f64> {
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_hma
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.hma_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_hma
                .current_effective_stop_price(
                    &session.execution_runtime.hma_execution,
                    session.market.tick_size,
                )
        }
        NativeStrategyKind::EmaCross => {
            if let Some(bar) = trailing_bar {
                let _ = session
                    .execution_config
                    .native_ema
                    .desired_trailing_stop_price(
                        &mut session.execution_runtime.ema_execution,
                        bar,
                        session.market.tick_size,
                    );
            }
            session
                .execution_config
                .native_ema
                .current_effective_stop_price(
                    &session.execution_runtime.ema_execution,
                    session.market.tick_size,
                )
        }
    }
}

async fn sync_execution_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
    trailing_bar: Option<&Bar>,
) -> Result<()> {
    if !session.execution_runtime.armed || session.execution_config.kind != StrategyKind::Native {
        return Ok(());
    }
    if !active_native_uses_protection(session) {
        return Ok(());
    }
    if session.execution_runtime.pending_target_qty.is_some() {
        return Ok(());
    }

    let signed_qty = selected_market_position_qty(session);
    let entry_price = selected_market_entry_price(session);
    sync_active_execution_position(session, signed_qty, entry_price);

    let (take_profit_price, stop_price) = if signed_qty == 0 {
        (None, None)
    } else if let Some(entry_price) = entry_price {
        (
            take_profit_price(session, entry_price, signed_qty),
            combined_stop_price(session, trailing_bar),
        )
    } else {
        return Ok(());
    };

    sync_native_protection(
        client,
        session,
        latency,
        internal_tx,
        signed_qty,
        take_profit_price,
        stop_price,
        "native execution sync",
    )
    .await
}

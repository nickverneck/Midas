#[cfg(feature = "replay")]
use super::virtual_time::{ReplayBarSchedule, ReplayVirtualEventKind};
use super::*;

pub(crate) fn spawn_replay_market_task(
    replay: ReplayState,
    cfg: AppConfig,
    contract: ContractSuggestion,
    bar_type: BarType,
    candle_mode: CandleMode,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_rx: tokio::sync::watch::Receiver<ReplaySpeed>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (replay, cfg, contract, broker_tx, replay_speed_rx);
        tokio::spawn(async move {
            let _ = internal_tx.send(InternalEvent::Error(
                "replay mode is not enabled in this build".to_string(),
            ));
            let _ = (bar_type, candle_mode);
        })
    }

    #[cfg(feature = "replay")]
    {
        tokio::spawn(async move {
            let mut replay_speed_rx = replay_speed_rx;
            if let Err(err) = replay_market_worker_inner(
                replay,
                cfg,
                contract,
                bar_type,
                candle_mode,
                broker_tx,
                &mut replay_speed_rx,
                internal_tx.clone(),
            )
            .await
            {
                let _ = internal_tx.send(InternalEvent::Error(format!("replay data: {err}")));
            }
        })
    }
}

#[cfg(feature = "replay")]
async fn replay_market_worker_inner(
    replay: ReplayState,
    cfg: AppConfig,
    contract: ContractSuggestion,
    bar_type: BarType,
    candle_mode: CandleMode,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_rx: &mut tokio::sync::watch::Receiver<ReplaySpeed>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let bars = replay.bars_for_type(bar_type)?;
    if bars.is_empty() {
        bail!("no {} bars available in replay dataset", bar_type.label());
    }

    let total_bars = bars.len();
    let evaluation_start_ns = replay.evaluation_start_ns()?;
    let history_loaded = replay_history_loaded(&bars, cfg.history_bars, evaluation_start_ns);
    let evaluation_rows_total = total_bars.saturating_sub(history_loaded);
    if replay.replay_window.is_some() && evaluation_rows_total == 0 {
        bail!("replay dataset view contains warmup rows but no evaluation rows");
    }
    let mut replay_window = replay_window_progress(
        replay.replay_window.as_ref(),
        history_loaded,
        evaluation_rows_total,
        0,
    );
    let mut series = LiveSeries::new();
    for bar in &bars[..history_loaded] {
        series.push_closed_bar_capped(bar, ENGINE_MARKET_BAR_LIMIT);
    }

    let initial_status = replay_window.as_ref().map_or_else(
        || {
            format!(
                "Replay {} loaded for {} ({}/{}) [{}]",
                bar_type.mode_label(candle_mode),
                contract.name,
                history_loaded,
                total_bars,
                cfg.replay_engine_mode.label()
            )
        },
        |window| {
            format!(
                "Replay {} loaded for {} (warmup {} | evaluation 0/{}) [{}]",
                bar_type.mode_label(candle_mode),
                contract.name,
                window.warmup_rows,
                window.evaluation_rows_total,
                cfg.replay_engine_mode.label()
            )
        },
    );
    // A dataset view's pre-evaluation rows seed indicators only. The first
    // market event is emitted for the first evaluation bar, preventing a
    // warmup row from triggering strategy execution.
    if evaluation_start_ns.is_none() {
        if let Some(update) = build_market_update(
            &contract,
            Some(replay.market_specs),
            candle_mode,
            series.closed_bars.len(),
            0,
            initial_status,
            0,
            None,
            None,
            &series,
        ) {
            let _ = internal_tx.send(InternalEvent::Market(update));
        }
    } else {
        let _ = internal_tx.send(InternalEvent::UserSocketStatus(initial_status));
    }

    let mut live_bars = 0usize;
    let evaluation_bars = &bars[history_loaded..];
    let mut schedule = ReplayBarSchedule::new(cfg.replay_engine_mode, evaluation_bars)?;
    while let Some(event) = schedule.next_bar(evaluation_bars) {
        let ReplayVirtualEventKind::BarClose { bar_index } = event.kind else {
            bail!("replay bar schedule emitted a non-bar event")
        };
        let bar = evaluation_bars
            .get(bar_index)
            .context("replay bar schedule referenced a missing bar")?;
        if bar.ts_ns != event.market_ts_ns {
            bail!("replay bar schedule timestamp does not match source bar")
        }
        wait_for_replay_bar(
            series.closed_bars.last().map(|previous| previous.ts_ns),
            event.market_ts_ns,
            cfg.replay_bar_interval_ms,
            replay_speed_rx,
        )
        .await;
        process_replay_bar(&broker_tx, bar).await?;
        let before_closed_len = series.closed_bars.len();
        let before_last_closed = series.closed_bars.last().cloned();
        let before_forming = series.forming_bar.clone();
        series.push_closed_bar_capped(bar, ENGINE_MARKET_BAR_LIMIT);
        live_bars = live_bars.saturating_add(1);
        replay_window = replay_window_progress(
            replay_window.as_ref(),
            history_loaded,
            evaluation_rows_total,
            live_bars,
        );
        let status = replay_window.as_ref().map_or_else(
            || {
                format!(
                    "Replay {} streaming for {} ({}/{}) [{}]",
                    bar_type.mode_label(candle_mode),
                    contract.name,
                    history_loaded + live_bars,
                    total_bars,
                    cfg.replay_engine_mode.label()
                )
            },
            |window| {
                format!(
                    "Replay {} streaming for {} (warmup {} | evaluation {}/{}) [{}]",
                    bar_type.mode_label(candle_mode),
                    contract.name,
                    window.warmup_rows,
                    window.evaluation_rows_processed,
                    window.evaluation_rows_total,
                    cfg.replay_engine_mode.label()
                )
            },
        );
        if let Some(mut update) = build_market_update(
            &contract,
            Some(replay.market_specs),
            candle_mode,
            series.closed_bars.len(),
            live_bars,
            status,
            before_closed_len,
            before_last_closed,
            before_forming,
            &series,
        ) {
            update.replay_window = replay_window.clone();
            let _ = internal_tx.send(InternalEvent::Market(update));
        }
    }

    let _ = internal_tx.send(InternalEvent::UserSocketStatus(format!(
        "Replay complete for {} ({}) [{}]",
        contract.name,
        bar_type.label(),
        cfg.replay_engine_mode.label()
    )));
    Ok(())
}

#[cfg(feature = "replay")]
pub(super) fn replay_window_progress(
    template: Option<&ReplayWindowSnapshot>,
    warmup_rows: usize,
    evaluation_rows_total: usize,
    evaluation_rows_processed: usize,
) -> Option<ReplayWindowSnapshot> {
    template.cloned().map(|mut window| {
        window.warmup_rows = warmup_rows;
        window.evaluation_rows_total = evaluation_rows_total;
        window.evaluation_rows_processed = evaluation_rows_processed.min(evaluation_rows_total);
        window
    })
}

#[cfg(feature = "replay")]
pub(super) fn replay_history_loaded(
    bars: &[Bar],
    configured_history_bars: usize,
    evaluation_start_ns: Option<i64>,
) -> usize {
    if let Some(evaluation_start_ns) = evaluation_start_ns {
        return bars.partition_point(|bar| bar.ts_ns < evaluation_start_ns);
    }
    if bars.len() > 1 {
        configured_history_bars.max(1).min(bars.len() - 1)
    } else {
        bars.len()
    }
}

#[cfg(feature = "replay")]
async fn process_replay_bar(broker_tx: &UnboundedSender<BrokerCommand>, bar: &Bar) -> Result<()> {
    let (response_tx, response_rx) = oneshot::channel();
    broker_tx
        .send(BrokerCommand::ReplayBar {
            bar: bar.clone(),
            response_tx,
        })
        .map_err(|_| anyhow::anyhow!("replay broker task is unavailable"))?;
    let _ = response_rx.await;
    Ok(())
}

#[cfg(feature = "replay")]
async fn wait_for_replay_bar(
    previous_ts_ns: Option<i64>,
    next_ts_ns: i64,
    fallback_ms: u64,
    replay_speed_rx: &mut tokio::sync::watch::Receiver<ReplaySpeed>,
) {
    let mut remaining = replay_gap_duration(previous_ts_ns, next_ts_ns, fallback_ms);
    while !remaining.is_zero() {
        let speed = *replay_speed_rx.borrow();
        let wall_sleep = scale_duration(remaining, 1.0 / speed.multiplier());
        if wall_sleep.is_zero() {
            break;
        }

        let started_at = time::Instant::now();
        tokio::select! {
            _ = tokio::time::sleep(wall_sleep) => break,
            changed = replay_speed_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let consumed_market = scale_duration(started_at.elapsed(), speed.multiplier());
                remaining = remaining.saturating_sub(consumed_market);
            }
        }
    }
}

#[cfg(feature = "replay")]
pub(super) fn replay_gap_duration(
    previous_ts_ns: Option<i64>,
    next_ts_ns: i64,
    fallback_ms: u64,
) -> Duration {
    match previous_ts_ns {
        Some(previous_ts_ns) if next_ts_ns > previous_ts_ns => {
            Duration::from_nanos((next_ts_ns - previous_ts_ns) as u64)
        }
        Some(_) => Duration::ZERO,
        None => Duration::from_millis(fallback_ms),
    }
}

#[cfg(feature = "replay")]
pub(super) fn scale_duration(duration: Duration, factor: f64) -> Duration {
    if !factor.is_finite() || factor <= 0.0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64((duration.as_secs_f64() * factor).max(0.0))
}

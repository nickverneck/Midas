use super::*;

/// Indicator values and gate inputs associated with one strategy evaluation.
/// This is kept separate from the execution tuple so existing order paths can
/// continue to use their compact return values while replay diagnostics opt in
/// to the richer snapshot.
#[derive(Debug, Clone, Default)]
pub(crate) struct StrategyEvaluationSnapshot {
    pub(crate) indicator_name: &'static str,
    pub(crate) previous_fast_indicator: Option<f64>,
    pub(crate) previous_slow_indicator: Option<f64>,
    pub(crate) fast_indicator: Option<f64>,
    pub(crate) slow_indicator: Option<f64>,
    pub(crate) auxiliary_name: Option<&'static str>,
    pub(crate) auxiliary_value: Option<f64>,
    pub(crate) raw_buy_signal: bool,
    pub(crate) raw_sell_signal: bool,
    pub(crate) effective_buy_signal: bool,
    pub(crate) effective_sell_signal: bool,
    pub(crate) hold_reason: Option<&'static str>,
}

/// Re-evaluate a selected bar window without mutating execution state and
/// expose the indicator inputs needed by the replay signal artifact.
pub(crate) fn snapshot_active_execution_strategy(
    session: &SessionState,
    bars: &[Bar],
    current_qty: i32,
) -> StrategyEvaluationSnapshot {
    let current_side = side_from_signed_qty(current_qty);
    match session.execution_config.native_strategy {
        NativeStrategyKind::HmaAngle => {
            let evaluation = session
                .execution_config
                .native_hma
                .evaluate(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "HMA",
                fast_indicator: evaluation.latest_hma,
                slow_indicator: evaluation.lookback_hma,
                auxiliary_name: Some("angle"),
                auxiliary_value: evaluation.angle,
                raw_buy_signal: evaluation.raw_buy_signal,
                raw_sell_signal: evaluation.raw_sell_signal,
                effective_buy_signal: evaluation.effective_buy_signal,
                effective_sell_signal: evaluation.effective_sell_signal,
                hold_reason: evaluation.hold_reason,
                ..Default::default()
            }
        }
        NativeStrategyKind::EmaCross => {
            let config = session.execution_config.native_ema.clone();
            let evaluation = if session.replay_enabled
                && session.cfg.replay_evaluator_mode
                    == crate::broker::ReplayEvaluatorMode::Streaming
            {
                // Diagnostics must not re-enter the legacy O(N²) evaluator.
                // Clone the already-advanced indicator state so a snapshot of
                // the current bar is O(1); historical corrections naturally
                // fall back to the streaming state's linear rebuild.
                let mut runtime = session.execution_runtime.ema_execution.clone();
                config.evaluate_streaming_with_market_update(
                    &mut runtime,
                    bars,
                    current_side,
                    session.execution_runtime.market_update_sequence,
                    session.execution_runtime.market_update_kind,
                )
            } else {
                config.evaluate(bars, current_side)
            };
            StrategyEvaluationSnapshot {
                indicator_name: "EMA",
                previous_fast_indicator: evaluation.previous_fast_ema,
                previous_slow_indicator: evaluation.previous_slow_ema,
                fast_indicator: evaluation.fast_ema,
                slow_indicator: evaluation.slow_ema,
                raw_buy_signal: evaluation.raw_buy_signal,
                raw_sell_signal: evaluation.raw_sell_signal,
                effective_buy_signal: evaluation.effective_buy_signal,
                effective_sell_signal: evaluation.effective_sell_signal,
                hold_reason: evaluation.hold_reason,
                ..Default::default()
            }
        }
        NativeStrategyKind::HmaCross => {
            let evaluation = session
                .execution_config
                .native_hma_cross
                .evaluate(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "HMA",
                previous_fast_indicator: evaluation.previous_fast_hma,
                previous_slow_indicator: evaluation.previous_slow_hma,
                fast_indicator: evaluation.fast_hma,
                slow_indicator: evaluation.slow_hma,
                raw_buy_signal: evaluation.raw_buy_signal,
                raw_sell_signal: evaluation.raw_sell_signal,
                effective_buy_signal: evaluation.effective_buy_signal,
                effective_sell_signal: evaluation.effective_sell_signal,
                hold_reason: evaluation.hold_reason,
                ..Default::default()
            }
        }
    }
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
                let config = session.execution_config.native_ema.clone();
                let evaluation = if session.replay_enabled
                    && session.cfg.replay_evaluator_mode
                        == crate::broker::ReplayEvaluatorMode::Streaming
                {
                    config.evaluate_streaming(
                        &mut session.execution_runtime.ema_execution,
                        bars,
                        current_side,
                    )
                } else {
                    config.evaluate(bars, current_side)
                };
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
                let config = session.execution_config.native_ema.clone();
                let evaluation = if session.replay_enabled
                    && session.cfg.replay_evaluator_mode
                        == crate::broker::ReplayEvaluatorMode::Streaming
                {
                    config.evaluate_streaming(
                        &mut session.execution_runtime.ema_execution,
                        window,
                        current_side,
                    )
                } else {
                    config.evaluate(window, current_side)
                };
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

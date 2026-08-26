use super::*;
use crate::strategies::hma_cross::{HmaCrossConfig, HmaCrossExecutionState};

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
        NativeStrategyKind::HeikinAshiColor => {
            let evaluation = session
                .execution_config
                .native_heikin_ashi
                .evaluate_from_source_bars(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "Heikin-Ashi",
                previous_fast_indicator: evaluation.previous_ha_open,
                previous_slow_indicator: evaluation.previous_ha_close,
                fast_indicator: evaluation.current_ha_open,
                slow_indicator: evaluation.current_ha_close,
                raw_buy_signal: evaluation.raw_buy_signal,
                raw_sell_signal: evaluation.raw_sell_signal,
                effective_buy_signal: evaluation.effective_buy_signal,
                effective_sell_signal: evaluation.effective_sell_signal,
                hold_reason: evaluation.hold_reason,
                ..Default::default()
            }
        }
        NativeStrategyKind::VolumeAdaptiveHmaCross => {
            let evaluation = session
                .execution_config
                .native_volume_hma_cross
                .evaluate(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "HMA+RVOL",
                previous_fast_indicator: evaluation.hma.previous_fast_hma,
                previous_slow_indicator: evaluation.hma.previous_slow_hma,
                fast_indicator: evaluation.hma.fast_hma,
                slow_indicator: evaluation.hma.slow_hma,
                auxiliary_name: Some("relative_volume"),
                auxiliary_value: evaluation.relative_volume,
                raw_buy_signal: evaluation.hma.raw_buy_signal,
                raw_sell_signal: evaluation.hma.raw_sell_signal,
                effective_buy_signal: evaluation.hma.effective_buy_signal,
                effective_sell_signal: evaluation.hma.effective_sell_signal,
                hold_reason: evaluation.hma.hold_reason,
            }
        }
        NativeStrategyKind::VolumeAdaptiveEmaCross => {
            let evaluation = session
                .execution_config
                .native_volume_ema_cross
                .evaluate(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "EMA+RVOL",
                previous_fast_indicator: evaluation.ema.previous_fast_ema,
                previous_slow_indicator: evaluation.ema.previous_slow_ema,
                fast_indicator: evaluation.ema.fast_ema,
                slow_indicator: evaluation.ema.slow_ema,
                auxiliary_name: Some("relative_volume"),
                auxiliary_value: evaluation.relative_volume,
                raw_buy_signal: evaluation.ema.raw_buy_signal,
                raw_sell_signal: evaluation.ema.raw_sell_signal,
                effective_buy_signal: evaluation.ema.effective_buy_signal,
                effective_sell_signal: evaluation.ema.effective_sell_signal,
                hold_reason: evaluation.ema.hold_reason,
            }
        }
        NativeStrategyKind::Adx => {
            let evaluation = session
                .execution_config
                .native_adx
                .evaluate(bars, current_side);
            StrategyEvaluationSnapshot {
                indicator_name: "ADX",
                previous_fast_indicator: evaluation.previous_adx,
                fast_indicator: evaluation.adx,
                slow_indicator: evaluation.plus_di,
                auxiliary_name: Some("signed_trend_score"),
                auxiliary_value: evaluation.signed_trend_score,
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
        NativeStrategyKind::HeikinAshiColor => {
            let evaluation = session
                .execution_config
                .native_heikin_ashi
                .evaluate_from_source_bars(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal,
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
        NativeStrategyKind::VolumeAdaptiveHmaCross => {
            let evaluation = session
                .execution_config
                .native_volume_hma_cross
                .evaluate(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal(),
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
        NativeStrategyKind::VolumeAdaptiveEmaCross => {
            let evaluation = session
                .execution_config
                .native_volume_ema_cross
                .evaluate(bars, side_from_signed_qty(current_qty));
            (
                evaluation.signal(),
                evaluation.summary(),
                evaluation.debug_summary(),
            )
        }
        NativeStrategyKind::Adx => {
            let evaluation = session
                .execution_config
                .native_adx
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
            NativeStrategyKind::HeikinAshiColor => {
                let evaluation = session
                    .execution_config
                    .native_heikin_ashi
                    .evaluate_from_source_bars(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                let evaluation = session
                    .execution_config
                    .native_volume_hma_cross
                    .evaluate(window, current_side);
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                let evaluation = session
                    .execution_config
                    .native_volume_ema_cross
                    .evaluate(window, current_side);
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::Adx => {
                let evaluation = session
                    .execution_config
                    .native_adx
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
            NativeStrategyKind::HeikinAshiColor => {
                let evaluation = session
                    .execution_config
                    .native_heikin_ashi
                    .evaluate_from_source_bars(bars, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                let config = session.execution_config.native_volume_hma_cross.clone();
                let evaluation = config.evaluate_current_cross(
                    &mut session.execution_runtime.volume_hma_cross_execution,
                    bars,
                    current_side,
                );
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                let config = session.execution_config.native_volume_ema_cross.clone();
                let evaluation = if session.replay_enabled
                    && session.cfg.replay_evaluator_mode
                        == crate::broker::ReplayEvaluatorMode::Streaming
                {
                    config.evaluate_streaming(
                        &mut session.execution_runtime.volume_ema_cross_execution,
                        bars,
                        current_side,
                    )
                } else {
                    config.evaluate(bars, current_side)
                };
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::Adx => {
                let evaluation = session
                    .execution_config
                    .native_adx
                    .evaluate(bars, current_side);
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
            NativeStrategyKind::HeikinAshiColor => {
                let evaluation = session
                    .execution_config
                    .native_heikin_ashi
                    .evaluate_from_source_bars(window, current_side);
                (
                    evaluation.signal,
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                let config = session.execution_config.native_volume_hma_cross.clone();
                let evaluation = config.evaluate_current_cross(
                    &mut session.execution_runtime.volume_hma_cross_execution,
                    window,
                    current_side,
                );
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                let config = session.execution_config.native_volume_ema_cross.clone();
                let evaluation = if session.replay_enabled
                    && session.cfg.replay_evaluator_mode
                        == crate::broker::ReplayEvaluatorMode::Streaming
                {
                    config.evaluate_streaming(
                        &mut session.execution_runtime.volume_ema_cross_execution,
                        window,
                        current_side,
                    )
                } else {
                    config.evaluate(window, current_side)
                };
                (
                    evaluation.signal(),
                    evaluation.summary(),
                    evaluation.debug_summary(),
                )
            }
            NativeStrategyKind::Adx => {
                let evaluation = session
                    .execution_config
                    .native_adx
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

/// Evaluate an incremental HMA crossover directly against the retained market
/// slice. Live and streaming-replay callers use this to avoid cloning the
/// retained bar window; legacy replay still uses the reference evaluator.
pub(crate) fn evaluate_incremental_hma_cross_since(
    config: &HmaCrossConfig,
    runtime: &mut HmaCrossExecutionState,
    signal_timing: NativeSignalTiming,
    bars: &[Bar],
    current_qty: i32,
    after_ts: Option<i64>,
    include_debug: bool,
) -> (Bar, StrategySignal, String, String) {
    let current_side = side_from_signed_qty(current_qty);
    if signal_timing == NativeSignalTiming::LiveBar {
        let signal_bar = bars
            .last()
            .expect("strategy bars must not be empty")
            .clone();
        let evaluation = config.evaluate_current_cross(runtime, bars, current_side);
        return (
            signal_bar,
            evaluation.signal,
            evaluation.summary(),
            if include_debug {
                evaluation.debug_summary()
            } else {
                String::new()
            },
        );
    }

    let start_idx = after_ts
        .and_then(|ts| bars.iter().position(|bar| bar.ts_ns > ts))
        .unwrap_or_else(|| bars.len().saturating_sub(1));
    let mut latest = None;
    for idx in start_idx..bars.len() {
        let evaluation = config.evaluate_current_cross(runtime, &bars[..=idx], current_side);
        let candidate = (
            bars[idx].clone(),
            evaluation.signal,
            evaluation.summary(),
            if include_debug {
                evaluation.debug_summary()
            } else {
                String::new()
            },
        );
        if candidate.1 != StrategySignal::Hold || latest.is_none() {
            latest = Some(candidate);
        }
    }
    latest.expect("strategy bars must not be empty")
}

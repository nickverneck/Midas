use super::*;

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

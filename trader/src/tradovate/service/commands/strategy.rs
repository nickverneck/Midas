use super::*;

pub(super) async fn set_execution_strategy_config(
    mut config: ExecutionStrategyConfig,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        bail!("connect first");
    };
    if config.kind == StrategyKind::Lua {
        if session.session_kind != SessionKind::Replay
            || !session.replay_enabled
            || session.hosted_replay_orders
        {
            bail!("Lua strategies can only be configured for cached Tradovate replay");
        }
        crate::lua_replay::validate_source(&config.lua).context("validate Lua replay strategy")?;
        config.native_execution_path = NativeExecutionPath::Guarded;
    }
    normalize_broker_owned_protection_config(&mut config);
    if session.execution_config == config {
        return Ok(());
    }

    let old_history = session
        .execution_config
        .market_history_requirement(session.cfg.history_bars);
    let new_history = config.market_history_requirement(session.cfg.history_bars);
    let selected_contract = session.selected_contract.clone();
    let should_refresh_market_history = selected_contract.is_some()
        && !session.replay_enabled
        && !session.execution_runtime.armed
        && new_history.requested_bars > old_history.requested_bars;

    if session.execution_runtime.armed
        && selected_contract.is_some()
        && new_history.requested_bars > old_history.requested_bars
    {
        bail!(
            "cannot increase market history while the strategy is armed; disarm before changing to a longer indicator"
        );
    }

    if let Some(session) = state.session.as_mut() {
        session.execution_config = config;
        emit_execution_state(event_tx, session);
    }

    if should_refresh_market_history {
        let (contract, bar_type, candle_mode) = {
            let session = state
                .session
                .as_ref()
                .context("session disappeared before market history refresh")?;
            (
                selected_contract.expect("selected contract checked above"),
                session.bar_type,
                session.candle_mode,
            )
        };
        super::market::subscribe_bars(
            contract,
            bar_type,
            candle_mode,
            state,
            event_tx,
            market_tx,
            internal_tx,
        )
        .await?;
    }
    Ok(())
}

fn normalize_broker_owned_protection_config(config: &mut ExecutionStrategyConfig) {
    if config.kind != StrategyKind::Native {
        return;
    }

    let uses_protection = match config.native_strategy {
        NativeStrategyKind::HmaAngle => config.native_hma.uses_native_protection(),
        NativeStrategyKind::EmaCross => config.native_ema.uses_native_protection(),
        NativeStrategyKind::HmaCross => config.native_hma_cross.uses_native_protection(),
        NativeStrategyKind::HeikinAshiColor => false,
        NativeStrategyKind::VolumeAdaptiveHmaCross => {
            config.native_volume_hma_cross.uses_native_protection()
        }
        NativeStrategyKind::VolumeAdaptiveEmaCross => {
            config.native_volume_ema_cross.uses_native_protection()
        }
        NativeStrategyKind::Adx => config.native_adx.uses_native_protection(),
    };
    if uses_protection && config.native_reversal_mode == NativeReversalMode::Direct {
        config.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    }
    if uses_protection || config.native_reversal_mode != NativeReversalMode::Direct {
        config.native_execution_path = NativeExecutionPath::Guarded;
    }
}

pub(super) fn arm_execution_strategy_command(
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    if session.hosted_replay_orders && !session.hosted_replay_running {
        bail!("NinjaTrader hosted Replay has ended; start a new Replay session before arming");
    }
    if session.execution_config.kind == StrategyKind::Lua
        && (session.session_kind != SessionKind::Replay
            || !session.replay_enabled
            || session.hosted_replay_orders
            || session.execution_config.native_execution_path != NativeExecutionPath::Guarded)
    {
        bail!("Lua strategies can only arm on the guarded cached Tradovate replay path");
    }
    let was_armed = session.execution_runtime.armed;
    arm_execution_strategy(session);
    if session.execution_runtime.armed
        && !was_armed
        && let Err(err) = start_engine_run(session)
    {
        session.execution_runtime.armed = false;
        session.execution_runtime.last_summary =
            format!("Strategy arm failed before history start: {err}");
        emit_execution_state(event_tx, session);
        return Err(err);
    }
    emit_execution_state(event_tx, session);
    emit_engine_history(event_tx, session);
    Ok(())
}

pub(super) fn disarm_execution_strategy_command(
    reason: String,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    disarm_execution_strategy(session, reason);
    emit_execution_state(event_tx, session);
    Ok(())
}

pub(super) fn probe_execution(
    tag: String,
    state: &ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        bail!("connect first");
    };
    let _ = event_tx.send(ServiceEvent::ExecutionProbe(execution_probe_snapshot(
        session,
        state.latency,
        tag,
    )));
    Ok(())
}

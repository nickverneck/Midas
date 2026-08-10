use super::*;

pub(super) fn set_execution_strategy_config(
    mut config: ExecutionStrategyConfig,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    normalize_broker_owned_protection_config(&mut config);
    if session.execution_config != config {
        session.execution_config = config;
        emit_execution_state(event_tx, session);
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
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
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
    event_tx: &UnboundedSender<ServiceEvent>,
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
    event_tx: &UnboundedSender<ServiceEvent>,
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

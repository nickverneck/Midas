use super::*;

pub(super) fn select_account(
    account_id: i64,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    {
        let Some(session) = state.session.as_mut() else {
            bail!("not connected");
        };
        session.selected_account_id = Some(account_id);
        handle_execution_account_sync(session, &broker_tx, event_tx)?;
    }
    request_snapshot_refresh(state, &internal_tx);
    Ok(())
}

pub(super) async fn search_contracts_command(
    query: String,
    limit: usize,
    state: &ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        bail!("connect first");
    };
    if session.replay_enabled {
        let results = state
            .replay
            .as_ref()
            .map(|replay| replay::search_replay_contracts(replay, &query, limit))
            .unwrap_or_default();
        let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
        return Ok(());
    }
    let rest_url = session.cfg.broker_rest_url();
    let results = search_contracts(
        &state.client,
        &rest_url,
        &session.tokens.access_token,
        &query,
        limit,
    )
    .await?;
    let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
    Ok(())
}

pub(super) async fn subscribe_bars(
    contract: ContractSuggestion,
    bar_type: BarType,
    candle_mode: CandleMode,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: InternalEventSender,
) -> Result<()> {
    let candle_mode = bar_type.effective_candle_mode(candle_mode);
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    if let Some(task) = state.market_task.take() {
        task.abort();
        let _ = task.await;
    }
    session.market = MarketSnapshot::default();
    let _ = market_tx.send(MarketSnapshot::default());
    session.selected_contract = Some(contract.clone());
    session.engine_run = None;
    session.active_order_strategy = None;
    session.bar_type = bar_type;
    session.candle_mode = candle_mode;
    session.execution_runtime.last_closed_bar_ts = None;
    session.execution_runtime.pending_target_qty = None;
    session.execution_runtime.reset_execution();
    session.execution_runtime.last_summary =
        "Selected contract changed; waiting for market data.".to_string();
    emit_execution_state(event_tx, session);

    if session.replay_enabled {
        let replay = state
            .replay
            .clone()
            .context("replay dataset is unavailable")?;
        let cfg = session.cfg.clone();
        state.market_task = Some(replay::spawn_replay_market_task(
            replay,
            cfg,
            contract,
            bar_type,
            candle_mode,
            state.broker_tx.clone(),
            state.replay_speed_tx.subscribe(),
            internal_tx,
        ));
    } else {
        let market_specs = fetch_contract_specs(
            &state.client,
            &session.cfg.broker_rest_url(),
            &session.tokens.access_token,
            &contract,
        )
        .await
        .ok();
        let cfg = session.cfg.clone();
        let token = session.tokens.md_access_token.clone();
        state.market_task = Some(tokio::spawn(market_data_worker(
            cfg,
            token,
            contract,
            market_specs,
            bar_type,
            candle_mode,
            internal_tx,
        )));
    }

    Ok(())
}

pub(super) fn set_replay_speed(
    speed: ReplaySpeed,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        return Ok(());
    };
    if !session.replay_enabled || state.replay_speed == speed {
        if session.replay_enabled {
            let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(state.replay_speed));
        }
        return Ok(());
    }
    state.replay_speed = speed;
    let _ = state.replay_speed_tx.send(speed);
    let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(speed));
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Replay speed set to {}",
        speed.label()
    )));
    Ok(())
}

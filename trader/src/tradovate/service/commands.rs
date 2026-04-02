use super::*;

pub(super) async fn handle_command(
    cmd: ServiceCommand,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            connect_live_session(cfg, state, event_tx, market_tx, internal_tx).await
        }
        ServiceCommand::EnterReplayMode {
            config: cfg,
            bar_type,
        } => enter_replay_mode(cfg, bar_type, state, event_tx, market_tx, internal_tx).await,
        ServiceCommand::ReplayState => replay_state(state, event_tx),
        ServiceCommand::SelectAccount { account_id } => {
            select_account(account_id, state, event_tx, internal_tx)
        }
        ServiceCommand::SearchContracts { query, limit } => {
            search_contracts_command(query, limit, state, event_tx).await
        }
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            subscribe_bars(contract, bar_type, state, event_tx, market_tx, internal_tx).await
        }
        ServiceCommand::SetReplaySpeed { speed } => set_replay_speed(speed, state, event_tx),
        ServiceCommand::ManualOrder { action } => manual_order(action, state, event_tx),
        ServiceCommand::SetTargetPosition {
            target_qty,
            automated,
            reason,
        } => set_target_position(target_qty, automated, reason, state, event_tx),
        ServiceCommand::ProfileLegacyOrderStrategyTarget { target_qty, reason } => {
            profile_legacy_order_strategy_target(target_qty, reason, state, event_tx)
        }
        ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
        } => sync_native_protection_command(
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
            state,
            internal_tx,
        ),
        ServiceCommand::SetExecutionStrategyConfig(config) => {
            set_execution_strategy_config(config, state, event_tx)
        }
        ServiceCommand::ArmExecutionStrategy => arm_execution_strategy_command(state, event_tx),
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            disarm_execution_strategy_command(reason, state, event_tx)
        }
        ServiceCommand::ProbeExecution { tag } => probe_execution(tag, state, event_tx),
    }
}

async fn connect_live_session(
    cfg: AppConfig,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    reset_state_for_new_session(state, market_tx);
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Authenticating against {}...",
        cfg.env.label()
    )));

    let tokens = authenticate(&state.client, &cfg).await?;
    save_token_cache(&cfg.session_cache_path, &tokens)?;

    let _ = event_tx.send(ServiceEvent::Connected {
        broker: BrokerKind::Tradovate,
        env: cfg.env,
        user_name: tokens.user_name.clone(),
        auth_mode: cfg.auth_mode,
        session_kind: SessionKind::Live,
        capabilities: tradovate_capabilities(),
    });

    let accounts = list_accounts(&state.client, &cfg.env, &tokens.access_token).await?;
    let mut user_store = UserSyncStore::default();
    seed_user_store(
        &state.client,
        &cfg.env,
        &tokens.access_token,
        &mut user_store,
    )
    .await;

    let selected_account_id = accounts.first().map(|account| account.id);
    let snapshots = user_store.build_snapshots(&accounts, None, &BTreeMap::new());

    let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts.clone()));
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));

    let account_ids = accounts
        .iter()
        .map(|account| account.id)
        .collect::<Vec<_>>();
    let (request_tx, user_task) = spawn_user_sync_task(
        cfg.clone(),
        tokens.clone(),
        account_ids,
        internal_tx.clone(),
    );
    let rest_probe_task = spawn_rest_probe_task(
        state.client.clone(),
        cfg.clone(),
        tokens.access_token.clone(),
        internal_tx,
    );
    state.user_task = Some(user_task);
    state.rest_probe_task = Some(rest_probe_task);

    state.session = Some(SessionState {
        cfg,
        session_kind: SessionKind::Live,
        replay_enabled: false,
        tokens,
        accounts,
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store,
        selected_account_id,
        selected_contract: None,
        bar_type: BarType::default(),
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    });
    if let Some(session) = state.session.as_ref() {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

async fn enter_replay_mode(
    cfg: AppConfig,
    bar_type: BarType,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    reset_state_for_new_session(state, market_tx);
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Loading replay dataset from {}...",
        cfg.replay_file_path.display()
    )));

    let replay = replay::load_replay_state(&cfg).await?;
    let accounts = replay::replay_accounts(&replay);
    let contract = replay::replay_contract(&replay);
    let selected_account_id = accounts.first().map(|account| account.id);
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut user_store = UserSyncStore::default();
    seed_replay_user_store(&accounts, &mut user_store);

    state.replay = Some(replay.clone());
    state.session = Some(SessionState {
        cfg: cfg.clone(),
        session_kind: SessionKind::Replay,
        replay_enabled: true,
        tokens: TokenBundle {
            access_token: String::new(),
            md_access_token: String::new(),
            expiration_time: None,
            user_id: None,
            user_name: Some("Replay".to_string()),
        },
        accounts: accounts.clone(),
        request_tx,
        execution_config: ExecutionStrategyConfig::default(),
        execution_runtime: ExecutionRuntimeState::default(),
        pending_signal_context: None,
        order_latency_tracker: None,
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        pending_protection_sync: None,
        user_store,
        selected_account_id,
        selected_contract: Some(contract.clone()),
        bar_type,
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
    });

    let _ = event_tx.send(ServiceEvent::Connected {
        broker: BrokerKind::Tradovate,
        env: cfg.env,
        user_name: Some("Replay".to_string()),
        auth_mode: cfg.auth_mode,
        session_kind: SessionKind::Replay,
        capabilities: tradovate_capabilities(),
    });
    let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts.clone()));
    let _ = event_tx.send(ServiceEvent::ContractSearchResults {
        query: "replay".to_string(),
        results: vec![contract.clone()],
    });
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(state.replay_speed));
    if let Some(session) = state.session.as_ref() {
        emit_execution_state(event_tx, session);
    }
    request_snapshot_refresh(state, &internal_tx);
    state.market_task = Some(replay::spawn_replay_market_task(
        replay,
        cfg,
        contract,
        bar_type,
        state.broker_tx.clone(),
        state.replay_speed_tx.subscribe(),
        internal_tx,
    ));

    Ok(())
}

fn replay_state(state: &ServiceState, event_tx: &UnboundedSender<ServiceEvent>) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        let _ = event_tx.send(ServiceEvent::Disconnected);
        return Ok(());
    };
    let _ = event_tx.send(ServiceEvent::Connected {
        broker: BrokerKind::Tradovate,
        env: session.cfg.env,
        user_name: session.tokens.user_name.clone(),
        auth_mode: session.cfg.auth_mode,
        session_kind: session.session_kind,
        capabilities: tradovate_capabilities(),
    });
    let _ = event_tx.send(ServiceEvent::AccountsLoaded(session.accounts.clone()));
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(
        session.user_store.build_snapshots(
            &session.accounts,
            Some(&session.market),
            &session.managed_protection,
        ),
    ));
    let _ = event_tx.send(ServiceEvent::Latency(state.latency));
    if session.replay_enabled {
        let _ = event_tx.send(ServiceEvent::ReplaySpeedUpdated(state.replay_speed));
    }
    emit_execution_state(event_tx, session);
    Ok(())
}

fn select_account(
    account_id: i64,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
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

async fn search_contracts_command(
    query: String,
    limit: usize,
    state: &ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
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
    let results = search_contracts(
        &state.client,
        &session.cfg.env,
        &session.tokens.access_token,
        &query,
        limit,
    )
    .await?;
    let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
    Ok(())
}

async fn subscribe_bars(
    contract: ContractSuggestion,
    bar_type: BarType,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    if let Some(task) = state.market_task.take() {
        task.abort();
    }
    session.market = MarketSnapshot::default();
    let _ = market_tx.send(MarketSnapshot::default());
    session.selected_contract = Some(contract.clone());
    session.active_order_strategy = None;
    session.bar_type = bar_type;
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
            state.broker_tx.clone(),
            state.replay_speed_tx.subscribe(),
            internal_tx,
        ));
    } else {
        let market_specs = fetch_contract_specs(
            &state.client,
            &session.cfg.env,
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
            internal_tx,
        )));
    }

    Ok(())
}

fn set_replay_speed(
    speed: ReplaySpeed,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
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

fn manual_order(
    action: ManualOrderAction,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    match dispatch_manual_order(session, &broker_tx, action)? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            if let Some(target_qty) = target_qty {
                session.execution_runtime.pending_target_qty = Some(target_qty);
                session.execution_runtime.last_summary =
                    "Manual close requested; waiting for flat position.".to_string();
                emit_execution_state(event_tx, session);
            }
        }
    }
    Ok(())
}

fn set_target_position(
    target_qty: i32,
    automated: bool,
    reason: String,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    match dispatch_target_position_order(session, &broker_tx, target_qty, automated, &reason)? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

fn profile_legacy_order_strategy_target(
    target_qty: i32,
    reason: String,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    match dispatch_profile_legacy_order_strategy_target(session, &broker_tx, target_qty, &reason)? {
        MarketOrderDispatchOutcome::NoOp { message } => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        MarketOrderDispatchOutcome::Queued { target_qty } => {
            session.execution_runtime.pending_target_qty = target_qty;
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

fn sync_native_protection_command(
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: String,
    state: &mut ServiceState,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let broker_tx = state.broker_tx.clone();
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    sync_native_protection(
        session,
        &broker_tx,
        signed_qty,
        take_profit_price,
        stop_price,
        &reason,
    )?;
    request_snapshot_refresh(state, &internal_tx);
    Ok(())
}

fn set_execution_strategy_config(
    config: ExecutionStrategyConfig,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    if session.execution_config != config {
        session.execution_config = config;
        if session.execution_runtime.armed {
            session.execution_runtime.armed = false;
            session.execution_runtime.pending_target_qty = None;
            session.execution_runtime.last_closed_bar_ts = None;
            session.execution_runtime.reset_execution();
            session.execution_runtime.last_summary =
                "Native strategy config changed; press Continue to re-arm.".to_string();
        }
        emit_execution_state(event_tx, session);
    }
    Ok(())
}

fn arm_execution_strategy_command(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        bail!("connect first");
    };
    arm_execution_strategy(session);
    emit_execution_state(event_tx, session);
    Ok(())
}

fn disarm_execution_strategy_command(
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

fn probe_execution(
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

fn reset_state_for_new_session(
    state: &mut ServiceState,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
) {
    shutdown_tasks(state);
    state.latency = LatencySnapshot::default();
    state.replay_speed = ReplaySpeed::default();
    let _ = state.replay_speed_tx.send(state.replay_speed);
    let _ = market_tx.send(MarketSnapshot::default());
}

fn seed_replay_user_store(accounts: &[AccountInfo], store: &mut UserSyncStore) {
    for account in accounts {
        store.apply(EntityEnvelope {
            entity_type: "account".to_string(),
            deleted: false,
            entity: json!({
                "id": account.id,
                "source": "replay",
                "name": account.name,
                "startingBalance": 100000.0,
                "balance": 100000.0,
                "netLiq": 100000.0
            }),
        });
        store.apply(EntityEnvelope {
            entity_type: "accountRiskStatus".to_string(),
            deleted: false,
            entity: json!({
                "id": account.id,
                "accountId": account.id,
                "source": "replay",
                "startingBalance": 100000.0,
                "balance": 100000.0,
                "netLiq": 100000.0,
                "cashBalance": 100000.0,
                "realizedPnL": 0.0
            }),
        });
        store.apply(EntityEnvelope {
            entity_type: "cashBalance".to_string(),
            deleted: false,
            entity: json!({
                "id": account.id,
                "accountId": account.id,
                "source": "replay",
                "startingBalance": 100000.0,
                "cashBalance": 100000.0,
                "realizedPnL": 0.0
            }),
        });
    }
}

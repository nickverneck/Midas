use super::*;

pub(super) async fn connect_live_session(
    cfg: AppConfig,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    reset_state_for_new_session(state, market_tx).await;
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Authenticating against {}...",
        cfg.env.label()
    )));

    let tokens = authenticate(&state.client, &cfg).await?;
    let token_file_snapshot = if matches!(cfg.auth_mode, AuthMode::TokenFile) {
        load_runtime_token_bundle(&cfg)
            .ok()
            .and_then(|loaded| loaded.file_snapshot)
    } else {
        None
    };
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

    let candle_mode = cfg.candle_mode;
    state.session = Some(SessionState {
        cfg,
        session_kind: SessionKind::Live,
        replay_enabled: false,
        tokens,
        token_file_snapshot,
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
        candle_mode,
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
        engine_run: None,
    });
    if let Some(session) = state.session.as_ref() {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

pub(super) async fn enter_replay_mode(
    cfg: AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
    replay_dataset_manifest: Option<std::path::PathBuf>,
    replay_dataset_view: Option<std::path::PathBuf>,
    shared_frames: Option<Arc<ReplayFrameSet>>,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let candle_mode = bar_type.effective_candle_mode(candle_mode);
    reset_state_for_new_session(state, market_tx).await;
    let _ = event_tx.send(ServiceEvent::Status(format!(
        "Loading replay dataset for {} from cache or {}...",
        bar_type.mode_label(candle_mode),
        cfg.replay_file_path.display()
    )));

    let replay = replay::load_replay_state_with_shared_frames(
        &cfg,
        bar_type,
        candle_mode,
        replay_dataset_manifest.as_deref(),
        replay_dataset_view.as_deref(),
        shared_frames,
    )
    .await?;
    let accounts = replay::replay_accounts(&replay);
    let contract = replay::replay_contract(&replay);
    let selected_account_id = accounts.first().map(|account| account.id);
    let (request_tx, _request_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut user_store = UserSyncStore::default();
    seed_replay_user_store(&accounts, &mut user_store, cfg.replay_initial_capital);

    state.replay = Some(replay.clone());
    state.replay_execution_ledger = replay::ReplayExecutionLedgerState::new_with_fill_config(
        cfg.replay_engine_mode,
        cfg.replay_fill_model,
        &cfg.replay_latency_config(),
        cfg.replay_bar_protection_policy,
        bar_type,
        candle_mode,
    );
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
        token_file_snapshot: None,
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
        candle_mode,
        market: MarketSnapshot::default(),
        managed_protection: BTreeMap::new(),
        active_order_strategy: None,
        next_strategy_order_nonce: 1,
        engine_run: None,
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
    let _ = event_tx.send(ServiceEvent::ReplayExecutionLedgerUpdated(
        state.replay_execution_ledger.summary(),
    ));
    let _ = event_tx.send(ServiceEvent::ReplayExecutionLedgerSnapshot(
        state.replay_execution_ledger.snapshot().clone(),
    ));
    if let Some(session) = state.session.as_ref() {
        emit_execution_state(event_tx, session);
    }
    request_snapshot_refresh(state, &internal_tx);
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

    Ok(())
}

pub(in crate::tradovate::service) async fn replay_state(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_mut() else {
        let _ = event_tx.send(ServiceEvent::Disconnected);
        return Ok(());
    };
    refresh_engine_history_from_broker(&state.client, session).await;
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
        let _ = event_tx.send(ServiceEvent::ReplayExecutionLedgerUpdated(
            state.replay_execution_ledger.summary(),
        ));
        let _ = event_tx.send(ServiceEvent::ReplayExecutionLedgerSnapshot(
            state.replay_execution_ledger.snapshot().clone(),
        ));
    }
    emit_execution_state(event_tx, session);
    emit_engine_history(event_tx, session);
    Ok(())
}

async fn reset_state_for_new_session(
    state: &mut ServiceState,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
) {
    shutdown_tasks(state).await;
    state.snapshot_generation = state.snapshot_generation.wrapping_add(1);
    state.snapshot_revision = 0;
    state.latency = LatencySnapshot::default();
    state.replay_speed = ReplaySpeed::default();
    state.replay_execution_ledger = replay::ReplayExecutionLedgerState::default();
    let _ = state.replay_speed_tx.send(state.replay_speed);
    let _ = market_tx.send(MarketSnapshot::default());
}

fn seed_replay_user_store(
    accounts: &[AccountInfo],
    store: &mut UserSyncStore,
    initial_capital: f64,
) {
    for account in accounts {
        store.apply(EntityEnvelope {
            entity_type: "account".to_string(),
            deleted: false,
            entity: json!({
                "id": account.id,
                "source": "replay",
                "name": account.name,
                "startingBalance": initial_capital,
                "balance": initial_capital,
                "netLiq": initial_capital
            }),
        });
        store.apply(EntityEnvelope {
            entity_type: "accountRiskStatus".to_string(),
            deleted: false,
            entity: json!({
                "id": account.id,
                "accountId": account.id,
                "source": "replay",
                "startingBalance": initial_capital,
                "balance": initial_capital,
                "netLiq": initial_capital,
                "cashBalance": initial_capital,
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
                "startingBalance": initial_capital,
                "cashBalance": initial_capital,
                "realizedPnL": 0.0
            }),
        });
    }
}

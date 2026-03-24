pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let _broker_task = spawn_broker_gateway_task(broker_rx, internal_tx.clone());
    let mut state = ServiceState {
        client: Client::builder()
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(300))
            .pool_max_idle_per_host(4)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .unwrap(),
        broker_tx,
        session: None,
        user_task: None,
        market_task: None,
        rest_probe_task: None,
        latency: LatencySnapshot::default(),
        snapshot_revision: 0,
    };
    let mut maintenance_tick =
        time::interval(Duration::from_secs(SESSION_MAINTENANCE_INTERVAL_SECS));
    maintenance_tick.tick().await;

    while let Some(next) = tokio::select! {
        biased;
        cmd = cmd_rx.recv() => cmd.map(Either::Command),
        internal = internal_rx.recv() => internal.map(Either::Internal),
        _ = maintenance_tick.tick() => Some(Either::MaintenanceTick),
    } {
        match next {
            Either::Command(cmd) => {
                if let Err(err) =
                    handle_command(cmd, &mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::Internal(internal) => {
                if let Err(err) =
                    handle_internal(internal, &mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::MaintenanceTick => {
                if let Err(err) = maintain_session(&mut state, &event_tx, internal_tx.clone()).await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
        }
    }

    shutdown_state(&mut state, &event_tx);
}

enum Either {
    Command(ServiceCommand),
    Internal(InternalEvent),
    MaintenanceTick,
}

async fn handle_command(
    cmd: ServiceCommand,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            shutdown_tasks(state);
            state.latency = LatencySnapshot::default();
            let _ = event_tx.send(ServiceEvent::Status(format!(
                "Authenticating against {}...",
                cfg.env.label()
            )));

            let tokens = authenticate(&state.client, &cfg).await?;
            save_token_cache(&cfg.session_cache_path, &tokens)?;

            let _ = event_tx.send(ServiceEvent::Connected {
                env: cfg.env,
                user_name: tokens.user_name.clone(),
                auth_mode: cfg.auth_mode,
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
            let user_cfg = cfg.clone();
            let user_tokens = tokens.clone();
            let (request_tx, user_task) =
                spawn_user_sync_task(user_cfg, user_tokens, account_ids, internal_tx.clone());
            let rest_probe_task = spawn_rest_probe_task(
                state.client.clone(),
                cfg.clone(),
                tokens.access_token.clone(),
                internal_tx.clone(),
            );
            state.user_task = Some(user_task);
            state.rest_probe_task = Some(rest_probe_task);

            state.session = Some(SessionState {
                cfg,
                tokens,
                accounts,
                request_tx,
                execution_config: ExecutionStrategyConfig::default(),
                execution_runtime: ExecutionRuntimeState::default(),
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
        }
        ServiceCommand::ReplayState => {
            let Some(session) = state.session.as_ref() else {
                let _ = event_tx.send(ServiceEvent::Disconnected);
                return Ok(());
            };
            let _ = event_tx.send(ServiceEvent::Connected {
                env: session.cfg.env,
                user_name: session.tokens.user_name.clone(),
                auth_mode: session.cfg.auth_mode,
            });
            let _ = event_tx.send(ServiceEvent::AccountsLoaded(session.accounts.clone()));
            let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(
                session.user_store.build_snapshots(
                    &session.accounts,
                    Some(&session.market),
                    &session.managed_protection,
                ),
            ));
            if session.market.contract_id.is_some()
                || session.market.contract_name.is_some()
                || !session.market.bars.is_empty()
                || !session.market.status.is_empty()
            {
                let _ = event_tx.send(ServiceEvent::MarketSnapshot(session.market.clone()));
            }
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::SelectAccount { account_id } => {
            let broker_tx = state.broker_tx.clone();
            {
                let Some(session) = state.session.as_mut() else {
                    bail!("not connected");
                };
                session.selected_account_id = Some(account_id);
                handle_execution_account_sync(session, &broker_tx, event_tx)?;
            }
            request_snapshot_refresh(state, &internal_tx);
        }
        ServiceCommand::SearchContracts { query, limit } => {
            let Some(session) = state.session.as_ref() else {
                bail!("connect first");
            };
            let results = search_contracts(
                &state.client,
                &session.cfg.env,
                &session.tokens.access_token,
                &query,
                limit,
            )
            .await?;
            let _ = event_tx.send(ServiceEvent::ContractSearchResults { query, results });
        }
        ServiceCommand::SubscribeBars { contract, bar_type } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let Some(task) = state.market_task.take() {
                task.abort();
            }
            session.market = MarketSnapshot::default();
            let market_specs = fetch_contract_specs(
                &state.client,
                &session.cfg.env,
                &session.tokens.access_token,
                &contract,
            )
            .await
            .ok();
            session.selected_contract = Some(contract.clone());
            session.active_order_strategy = None;
            session.bar_type = bar_type;
            session.execution_runtime.last_closed_bar_ts = None;
            session.execution_runtime.pending_target_qty = None;
            session.execution_runtime.reset_execution();
            session.execution_runtime.last_summary =
                "Selected contract changed; waiting for market data.".to_string();
            emit_execution_state(event_tx, session);
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
        ServiceCommand::ManualOrder { action } => {
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
        }
        ServiceCommand::SetTargetPosition {
            target_qty,
            automated,
            reason,
        } => {
            let broker_tx = state.broker_tx.clone();
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            if let MarketOrderDispatchOutcome::NoOp { message } =
                dispatch_target_position_order(session, &broker_tx, target_qty, automated, &reason)?
            {
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
        }
        ServiceCommand::SyncNativeProtection {
            signed_qty,
            take_profit_price,
            stop_price,
            reason,
        } => {
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
        }
        ServiceCommand::SetExecutionStrategyConfig(config) => {
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
        }
        ServiceCommand::ArmExecutionStrategy => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            arm_execution_strategy(session);
            emit_execution_state(event_tx, session);
        }
        ServiceCommand::DisarmExecutionStrategy { reason } => {
            let Some(session) = state.session.as_mut() else {
                bail!("connect first");
            };
            disarm_execution_strategy(session, reason);
            emit_execution_state(event_tx, session);
        }
    }
    Ok(())
}

async fn handle_internal(
    internal: InternalEvent,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match internal {
        InternalEvent::UserEntities(entities) => {
            let mut latency_changed = false;
            let mut trade_markers_changed = false;
            let broker_tx = state.broker_tx.clone();
            {
                let Some(session) = state.session.as_mut() else {
                    return Ok(());
                };
                for envelope in &entities {
                    latency_changed |=
                        update_latency_from_envelope(session, &mut state.latency, &envelope);
                    session.user_store.apply(envelope.clone());
                }
                for envelope in &entities {
                    if envelope.deleted || !envelope.entity_type.eq_ignore_ascii_case("fill") {
                        continue;
                    }
                    if let Some(marker) = trade_marker_from_fill(session, &envelope.entity) {
                        trade_markers_changed |= record_trade_marker(session, marker);
                    }
                }
                if trade_markers_changed {
                    let _ = event_tx.send(ServiceEvent::TradeMarkersUpdated(
                        session.market.trade_markers.clone(),
                    ));
                }
                handle_execution_account_sync(session, &broker_tx, event_tx)?;
            }
            request_snapshot_refresh(state, &internal_tx);
            if latency_changed {
                let _ = event_tx.send(ServiceEvent::Latency(state.latency));
            }
        }
        InternalEvent::SnapshotsBuilt {
            revision,
            snapshots,
        } => {
            if revision == state.snapshot_revision && state.session.is_some() {
                let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
            }
        }
        InternalEvent::RestLatencyMeasured(rest_rtt_ms) => {
            state.latency.rest_rtt_ms = Some(rest_rtt_ms);
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::UserSocketStatus(message) => {
            let _ = event_tx.send(ServiceEvent::Status(message));
        }
        InternalEvent::Market(update) => {
            if state.session.is_some() {
                let broker_tx = state.broker_tx.clone();
                let (snapshot, closed_bar_advanced) = {
                    let session = state.session.as_mut().expect("checked session above");
                    let closed_bar_advanced = apply_market_update(&mut session.market, update);
                    maybe_run_execution_strategy(session, &broker_tx, event_tx)?;
                    (session.market.clone(), closed_bar_advanced)
                };
                if closed_bar_advanced {
                    request_snapshot_refresh(state, &internal_tx);
                }
                let _ = event_tx.send(ServiceEvent::MarketSnapshot(snapshot));
                return Ok(());
            }
        }
        InternalEvent::BrokerOrderAck(ack) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if let Some(tracker) = session.order_latency_tracker.as_mut() {
                    if tracker.cl_ord_id == ack.cl_ord_id {
                        tracker.order_id = ack.order_id;
                    }
                }
            }
            state.latency.last_order_ack_ms = Some(ack.submit_rtt_ms);
            state.latency.last_order_seen_ms = None;
            state.latency.last_exec_report_ms = None;
            state.latency.last_fill_ms = None;
            let _ = event_tx.send(ServiceEvent::Status(ack.message));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::BrokerOrderFailed(failure) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if session
                    .order_latency_tracker
                    .as_ref()
                    .is_some_and(|tracker| tracker.cl_ord_id == failure.cl_ord_id)
                {
                    session.order_latency_tracker = None;
                }
                if let Some(target_qty) = failure.target_qty {
                    if session.execution_runtime.pending_target_qty == Some(target_qty) {
                        session.execution_runtime.pending_target_qty = None;
                        session.execution_runtime.last_summary = failure.message.clone();
                        emit_execution_state(event_tx, session);
                    }
                }
            }
            let _ = event_tx.send(ServiceEvent::Error(failure.message));
        }
        InternalEvent::OrderStrategyAck(ack) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if let Some(tracker) = session.order_latency_tracker.as_mut() {
                    if tracker.cl_ord_id == ack.uuid {
                        tracker.order_strategy_id = ack.order_strategy_id;
                    }
                }
                if let Some(order_strategy_id) = ack.order_strategy_id {
                    session.active_order_strategy = Some(TrackedOrderStrategy {
                        key: ack.key,
                        order_strategy_id,
                        target_qty: ack.target_qty,
                    });
                }
            }
            state.latency.last_order_ack_ms = Some(ack.submit_rtt_ms);
            state.latency.last_order_seen_ms = None;
            state.latency.last_exec_report_ms = None;
            state.latency.last_fill_ms = None;
            let _ = event_tx.send(ServiceEvent::Status(ack.message));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::OrderStrategyFailed(failure) => {
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if session
                    .order_latency_tracker
                    .as_ref()
                    .is_some_and(|tracker| tracker.cl_ord_id == failure.uuid)
                {
                    session.order_latency_tracker = None;
                }
                if session.execution_runtime.pending_target_qty == Some(failure.target_qty) {
                    session.execution_runtime.pending_target_qty = None;
                    session.execution_runtime.last_summary = failure.message.clone();
                    emit_execution_state(event_tx, session);
                }
            }
            let _ = event_tx.send(ServiceEvent::Error(failure.message));
        }
        InternalEvent::ProtectionSyncApplied(ack) => {
            let broker_tx = state.broker_tx.clone();
            {
                let Some(session) = state.session.as_mut() else {
                    return Ok(());
                };
                session.protection_sync_in_flight = false;
                match ack.next_state {
                    Some(next_state) => {
                        session.managed_protection.insert(ack.key, next_state);
                    }
                    None => {
                        session.managed_protection.remove(&ack.key);
                    }
                }

                if let Some(desired) = session.pending_protection_sync.take() {
                    sync_native_protection_target(session, &broker_tx, desired)?;
                }
            }
            request_snapshot_refresh(state, &internal_tx);
            if let Some(message) = ack.message {
                let _ = event_tx.send(ServiceEvent::Status(message));
            }
        }
        InternalEvent::ProtectionSyncFailed(failure) => {
            let broker_tx = state.broker_tx.clone();
            {
                let Some(session) = state.session.as_mut() else {
                    return Ok(());
                };
                session.protection_sync_in_flight = false;
                if let Some(desired) = session.pending_protection_sync.take() {
                    sync_native_protection_target(session, &broker_tx, desired)?;
                }
            }
            request_snapshot_refresh(state, &internal_tx);
            let _ = event_tx.send(ServiceEvent::Error(failure.message));
        }
        InternalEvent::Error(message) => {
            let _ = event_tx.send(ServiceEvent::Error(message));
        }
    }
    Ok(())
}

async fn maintain_session(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        return Ok(());
    };

    let refresh_action = next_token_maintenance_action(&session.cfg, &session.tokens)?;
    let mut forced_restart = false;
    let mut status_message = None;

    if let Some(action) = refresh_action {
        let next_tokens = match action {
            TokenMaintenanceAction::RefreshCredentials => {
                request_access_token(&state.client, &session.cfg).await?
            }
            TokenMaintenanceAction::ReloadTokenFile => load_runtime_token_bundle(&session.cfg)?,
        };

        if let Some(session) = state.session.as_mut() {
            if token_bundle_changed(&session.tokens, &next_tokens) {
                session.tokens = next_tokens;
                save_token_cache(&session.cfg.session_cache_path, &session.tokens)?;
                refresh_session_state(&state.client, session, event_tx).await?;
                if let Some(task) = state.user_task.take() {
                    task.abort();
                }
                if let Some(task) = state.market_task.take() {
                    task.abort();
                }
                if let Some(task) = state.rest_probe_task.take() {
                    task.abort();
                }
                forced_restart = true;
                status_message = Some(match action {
                    TokenMaintenanceAction::RefreshCredentials => {
                        "Session token refreshed; reconnecting background streams.".to_string()
                    }
                    TokenMaintenanceAction::ReloadTokenFile => {
                        "Session token reloaded from file; reconnecting background streams."
                            .to_string()
                    }
                });
            }
        }
    }

    let restart = ensure_background_tasks(state, internal_tx).await?;

    if let Some(message) = status_message {
        let _ = event_tx.send(ServiceEvent::Status(message));
    }
    if restart.user_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "User sync stream restarted.".to_string(),
        ));
    }
    if restart.market_restarted && !forced_restart {
        let contract_name = state
            .session
            .as_ref()
            .and_then(|session| session.selected_contract.as_ref())
            .map(|contract| contract.name.clone())
            .unwrap_or_else(|| "selected contract".to_string());
        let _ = event_tx.send(ServiceEvent::Status(format!(
            "Market data stream restarted for {contract_name}."
        )));
    }
    if restart.rest_probe_restarted && !forced_restart {
        let _ = event_tx.send(ServiceEvent::Status(
            "REST latency probe restarted.".to_string(),
        ));
    }

    Ok(())
}

async fn refresh_session_state(
    client: &Client,
    session: &mut SessionState,
    event_tx: &UnboundedSender<ServiceEvent>,
) -> Result<()> {
    let accounts = list_accounts(client, &session.cfg.env, &session.tokens.access_token).await?;
    let mut user_store = UserSyncStore::default();
    seed_user_store(
        client,
        &session.cfg.env,
        &session.tokens.access_token,
        &mut user_store,
    )
    .await;

    session.accounts = accounts.clone();
    if let Some(selected_account_id) = session.selected_account_id {
        if !session
            .accounts
            .iter()
            .any(|account| account.id == selected_account_id)
        {
            session.selected_account_id = session.accounts.first().map(|account| account.id);
        }
    } else {
        session.selected_account_id = session.accounts.first().map(|account| account.id);
    }
    session.user_store = user_store;

    let snapshots = session.user_store.build_snapshots(
        &session.accounts,
        Some(&session.market),
        &session.managed_protection,
    );
    let _ = event_tx.send(ServiceEvent::AccountsLoaded(accounts));
    let _ = event_tx.send(ServiceEvent::AccountSnapshotsLoaded(snapshots));
    Ok(())
}

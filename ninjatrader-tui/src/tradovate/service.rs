pub async fn service_loop(
    mut cmd_rx: UnboundedReceiver<ServiceCommand>,
    event_tx: UnboundedSender<ServiceEvent>,
    market_tx: tokio::sync::watch::Sender<MarketSnapshot>,
) {
    let (internal_tx, mut internal_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broker_tx, broker_rx) = tokio::sync::mpsc::unbounded_channel();
    let (replay_speed_tx, _replay_speed_rx) =
        tokio::sync::watch::channel(ReplaySpeed::default());
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
        replay_speed_tx,
        replay_speed: ReplaySpeed::default(),
        session: None,
        replay: None,
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
                if let Err(err) = handle_command(
                    cmd,
                    &mut state,
                    &event_tx,
                    &market_tx,
                    internal_tx.clone(),
                )
                .await
                {
                    let _ = event_tx.send(ServiceEvent::Error(err.to_string()));
                }
            }
            Either::Internal(internal) => {
                if let Err(err) = handle_internal(
                    internal,
                    &mut state,
                    &event_tx,
                    &market_tx,
                    internal_tx.clone(),
                )
                .await
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
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            shutdown_tasks(state);
            state.latency = LatencySnapshot::default();
            state.replay_speed = ReplaySpeed::default();
            let _ = state.replay_speed_tx.send(state.replay_speed);
            let _ = market_tx.send(MarketSnapshot::default());
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
                session_kind: SessionKind::Live,
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
        }
        ServiceCommand::EnterReplayMode { config: cfg, bar_type } => {
            shutdown_tasks(state);
            state.latency = LatencySnapshot::default();
            state.replay_speed = ReplaySpeed::default();
            let _ = state.replay_speed_tx.send(state.replay_speed);
            let _ = market_tx.send(MarketSnapshot::default());
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
                env: cfg.env,
                user_name: Some("Replay".to_string()),
                auth_mode: cfg.auth_mode,
                session_kind: SessionKind::Replay,
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
                session_kind: session.session_kind,
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
        }
        ServiceCommand::SubscribeBars { contract, bar_type } => {
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
        }
        ServiceCommand::SetReplaySpeed { speed } => {
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
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
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
                    let previous_latency = state.latency;
                    latency_changed |=
                        update_latency_from_envelope(session, &mut state.latency, &envelope);
                    emit_debug_logs_from_latency_delta(
                        event_tx,
                        session,
                        previous_latency,
                        state.latency,
                    );
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
                let (display_snapshot, closed_bar_advanced) = {
                    let session = state.session.as_mut().expect("checked session above");
                    let closed_bar_advanced = apply_market_update(&mut session.market, update);
                    maybe_run_execution_strategy(session, &broker_tx, event_tx)?;
                    (display_market_snapshot(&session.market), closed_bar_advanced)
                };
                if closed_bar_advanced {
                    request_snapshot_refresh(state, &internal_tx);
                }
                let _ = market_tx.send(display_snapshot);
                return Ok(());
            }
        }
        InternalEvent::BrokerOrderAck(ack) => {
            let mut signal_submit_ms = None;
            let mut signal_context = None;
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if let Some(tracker) = session.order_latency_tracker.as_mut() {
                    if tracker.cl_ord_id == ack.cl_ord_id {
                        tracker.order_id = ack.order_id;
                        signal_submit_ms = tracker
                            .signal_started_at
                            .map(|started_at| started_at.elapsed().as_millis() as u64);
                        signal_context = tracker.signal_context.clone();
                    }
                }
            }
            state.latency.last_order_ack_ms = Some(ack.submit_rtt_ms);
            state.latency.last_order_seen_ms = None;
            state.latency.last_exec_report_ms = None;
            state.latency.last_fill_ms = None;
            state.latency.last_signal_submit_ms = signal_submit_ms;
            state.latency.last_signal_seen_ms = None;
            state.latency.last_signal_ack_ms = None;
            state.latency.last_signal_fill_ms = None;
            let debug_message = format!(
                "submit {}{} | {}",
                format_debug_latency_ms(ack.submit_rtt_ms),
                debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
                ack.message
            );
            let _ = event_tx.send(ServiceEvent::Status(ack.message));
            let _ = event_tx.send(ServiceEvent::DebugLog(debug_message));
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
            let _ = event_tx.send(ServiceEvent::DebugLog(format!(
                "submit failed | {}",
                failure.message
            )));
            let _ = event_tx.send(ServiceEvent::Error(failure.message));
        }
        InternalEvent::OrderStrategyAck(ack) => {
            let mut signal_submit_ms = None;
            let mut signal_context = None;
            if let Some(session) = state.session.as_mut() {
                session.order_submit_in_flight = false;
                if let Some(tracker) = session.order_latency_tracker.as_mut() {
                    if tracker.cl_ord_id == ack.uuid {
                        tracker.order_strategy_id = ack.order_strategy_id;
                        signal_submit_ms = tracker
                            .signal_started_at
                            .map(|started_at| started_at.elapsed().as_millis() as u64);
                        signal_context = tracker.signal_context.clone();
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
            state.latency.last_signal_submit_ms = signal_submit_ms;
            state.latency.last_signal_seen_ms = None;
            state.latency.last_signal_ack_ms = None;
            state.latency.last_signal_fill_ms = None;
            let debug_message = format!(
                "submit {}{} | {}",
                format_debug_latency_ms(ack.submit_rtt_ms),
                debug_signal_latency_suffix(signal_submit_ms, signal_context.as_deref()),
                ack.message
            );
            let _ = event_tx.send(ServiceEvent::Status(ack.message));
            let _ = event_tx.send(ServiceEvent::DebugLog(debug_message));
            let _ = event_tx.send(ServiceEvent::Latency(state.latency));
        }
        InternalEvent::OrderStrategyFailed(failure) => {
            let mut stale_interrupt_recovered = false;
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
                }
                if failure.stale_interrupt {
                    clear_selected_order_strategy_state(session);
                    session.execution_runtime.last_summary =
                        "Previous strategy was already inactive; retrying current signal after broker sync."
                            .to_string();
                    if let Some(last_closed_ts) = latest_closed_bar_ts(session) {
                        session.execution_runtime.last_closed_bar_ts =
                            Some(last_closed_ts.saturating_sub(1));
                    }
                    stale_interrupt_recovered = true;
                    emit_execution_state(event_tx, session);
                } else if session.execution_runtime.pending_target_qty.is_none() {
                    session.execution_runtime.last_summary = failure.message.clone();
                    emit_execution_state(event_tx, session);
                }
            }
            if stale_interrupt_recovered {
                request_snapshot_refresh(state, &internal_tx);
                let _ = event_tx.send(ServiceEvent::DebugLog(format!(
                    "submit stale | {}",
                    failure.message
                )));
                let _ = event_tx.send(ServiceEvent::Status(failure.message));
            } else {
                let _ = event_tx.send(ServiceEvent::DebugLog(format!(
                    "submit failed | {}",
                    failure.message
                )));
                let _ = event_tx.send(ServiceEvent::Error(failure.message));
            }
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

fn emit_debug_logs_from_latency_delta(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    previous: LatencySnapshot,
    current: LatencySnapshot,
) {
    emit_debug_latency_stage(
        event_tx,
        session,
        "seen",
        previous.last_order_seen_ms,
        current.last_order_seen_ms,
    );
    emit_debug_latency_stage(
        event_tx,
        session,
        "ack",
        previous.last_exec_report_ms,
        current.last_exec_report_ms,
    );
    emit_debug_latency_stage(
        event_tx,
        session,
        "fill",
        previous.last_fill_ms,
        current.last_fill_ms,
    );
}

fn emit_debug_latency_stage(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
    stage: &str,
    previous: Option<u64>,
    current: Option<u64>,
) {
    if current.is_none() || previous == current {
        return;
    }
    let _ = event_tx.send(ServiceEvent::DebugLog(format!(
        "{stage} {}{} | {}",
        format_debug_latency_ms(current.unwrap_or_default()),
        debug_signal_latency_suffix(
            session
                .order_latency_tracker
                .as_ref()
                .and_then(|tracker| tracker.signal_started_at)
                .map(|started_at| started_at.elapsed().as_millis() as u64),
            session
                .order_latency_tracker
                .as_ref()
                .and_then(|tracker| tracker.signal_context.as_deref()),
        ),
        debug_tracker_context(session.order_latency_tracker.as_ref(), session)
    )));
}

fn format_debug_latency_ms(value: u64) -> String {
    if value >= 60_000 {
        format!("{:.1}m", value as f64 / 60_000.0)
    } else if value >= 1_000 {
        format!("{:.1}s", value as f64 / 1_000.0)
    } else {
        format!("{value}ms")
    }
}

fn debug_signal_latency_suffix(signal_latency_ms: Option<u64>, signal_context: Option<&str>) -> String {
    let Some(signal_latency_ms) = signal_latency_ms else {
        return String::new();
    };
    let mut suffix = format!(" | signal {}", format_debug_latency_ms(signal_latency_ms));
    if let Some(signal_context) = signal_context {
        suffix.push_str(&format!(" [{signal_context}]"));
    }
    suffix
}

fn debug_tracker_context(
    tracker: Option<&OrderLatencyTracker>,
    session: &SessionState,
) -> String {
    let mut parts = Vec::new();

    if let Some(contract) = session.selected_contract.as_ref() {
        parts.push(contract.name.clone());
    }
    if let Some(account_name) = session
        .selected_account_id
        .and_then(|selected_id| session.accounts.iter().find(|account| account.id == selected_id))
        .map(|account| account.name.clone())
    {
        parts.push(format!("on {account_name}"));
    }

    if let Some(tracker) = tracker {
        parts.push(format!("[request {}]", tracker.cl_ord_id));
        if let Some(order_id) = tracker.order_id {
            parts.push(format!("(order {order_id})"));
        }
        if let Some(order_strategy_id) = tracker.order_strategy_id {
            parts.push(format!("(strategy {order_strategy_id})"));
        }
    }

    if parts.is_empty() {
        "selected market".to_string()
    } else {
        parts.join(" ")
    }
}

async fn maintain_session(
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let Some(session) = state.session.as_ref() else {
        return Ok(());
    };
    if session.replay_enabled {
        return Ok(());
    }

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

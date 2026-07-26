use super::*;
use crate::broker::{ReplayDownloadCacheTarget, ReplayDownloadOperationId, ReplayDownloadPhase};

#[cfg(feature = "replay")]
use crate::replay_cache::{
    ReplayCacheContract, ReplayCacheInstrument, ReplayCacheRawTicksWrite,
    ReplayCacheServerBarsWrite, ReplayCacheSourceKind, write_raw_ticks_parquet_cache,
    write_server_bars_parquet_cache,
};
#[cfg(feature = "replay")]
use chrono::{Datelike, TimeZone, Utc};

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
            candle_mode,
            replay_dataset_manifest,
        } => {
            enter_replay_mode(
                cfg,
                bar_type,
                candle_mode,
                replay_dataset_manifest,
                state,
                event_tx,
                market_tx,
                internal_tx,
            )
            .await
        }
        ServiceCommand::DownloadReplayData {
            operation_id,
            config,
            instrument,
            contract,
            target,
            start_date,
            end_date,
            source_kind,
            bar_type,
            candle_mode,
            display_name,
            tags,
        } => {
            reap_finished_replay_download(state).await;
            if let Some(active) = state.replay_download_job.as_ref() {
                let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                    operation_id,
                    phase: ReplayDownloadPhase::Busy,
                    message: format!(
                        "Replay download {} is still {}; cancel it or wait for it to finish.",
                        active.operation_id.0,
                        match active.stage() {
                            ReplayDownloadJobStage::Network => "collecting network data",
                            ReplayDownloadJobStage::CancelRequested => "cancelling network data",
                            ReplayDownloadJobStage::Committing => "committing cache data",
                            ReplayDownloadJobStage::Finished => "finishing",
                        }
                    ),
                });
                return Ok(());
            }
            let event_tx = event_tx.clone();
            let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
            let stage = Arc::new(AtomicU8::new(ReplayDownloadJobStage::Network as u8));
            let task_stage = stage.clone();
            let task = tokio::spawn(async move {
                let result = download_replay_data(
                    operation_id,
                    config,
                    instrument,
                    contract,
                    target,
                    start_date,
                    end_date,
                    source_kind,
                    bar_type,
                    candle_mode,
                    display_name,
                    tags,
                    &event_tx,
                    cancel_rx,
                    task_stage.clone(),
                )
                .await;
                if let Err((phase, err)) = result {
                    let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                        operation_id,
                        phase,
                        message: err.to_string(),
                    });
                }
                task_stage.store(ReplayDownloadJobStage::Finished as u8, Ordering::Release);
            });
            state.replay_download_job = Some(ReplayDownloadJob {
                operation_id,
                cancel_tx,
                stage,
                task,
            });
            Ok(())
        }
        ServiceCommand::SearchReplayDownloadContracts {
            operation_id,
            config,
            query,
            limit,
        } => {
            #[cfg(feature = "replay")]
            {
                replace_replay_lookup(state).await;
                let event_tx = event_tx.clone();
                let task = tokio::spawn(async move {
                    let result =
                        crate::tradovate::search_replay_download_contracts(&config, &query, limit)
                            .await;
                    match result {
                        Ok(results) => {
                            let _ =
                                event_tx.send(ServiceEvent::ReplayDownloadContractSearchResults {
                                    operation_id,
                                    query,
                                    results,
                                });
                        }
                        Err(err) => {
                            let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                                operation_id,
                                phase: ReplayDownloadPhase::Searching,
                                message: err.to_string(),
                            });
                        }
                    }
                });
                state.replay_lookup_job = Some(ReplayLookupJob { operation_id, task });
            }
            #[cfg(not(feature = "replay"))]
            {
                let _ = (config, query, limit);
                let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                    operation_id,
                    phase: ReplayDownloadPhase::Searching,
                    message: "replay downloader requires a replay-enabled build".to_string(),
                });
            }
            Ok(())
        }
        ServiceCommand::InspectReplayDownloadContract {
            operation_id,
            config,
            contract,
        } => {
            #[cfg(feature = "replay")]
            {
                replace_replay_lookup(state).await;
                let event_tx = event_tx.clone();
                let task = tokio::spawn(async move {
                    let result =
                        crate::tradovate::inspect_replay_download_contract(&config, contract).await;
                    match result {
                        Ok(inspection) => {
                            let coverage = inspection.suggested_coverage;
                            let _ = event_tx.send(ServiceEvent::ReplayDownloadContractInspected {
                                operation_id,
                                contract: inspection.contract,
                                suggested_start_date: coverage
                                    .as_ref()
                                    .map(|value| value.start_date),
                                suggested_end_date: coverage.as_ref().map(|value| value.end_date),
                                suggestion_basis: coverage.map(|value| value.basis),
                            });
                        }
                        Err(err) => {
                            let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                                operation_id,
                                phase: ReplayDownloadPhase::InspectingContract,
                                message: err.to_string(),
                            });
                        }
                    }
                });
                state.replay_lookup_job = Some(ReplayLookupJob { operation_id, task });
            }
            #[cfg(not(feature = "replay"))]
            {
                let _ = (config, contract);
                let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
                    operation_id,
                    phase: ReplayDownloadPhase::InspectingContract,
                    message: "replay downloader requires a replay-enabled build".to_string(),
                });
            }
            Ok(())
        }
        ServiceCommand::CancelReplayDownloadOperation { operation_id } => {
            cancel_replay_operation(operation_id, state, event_tx).await;
            Ok(())
        }
        ServiceCommand::ReplayState => replay_state(state, event_tx),
        ServiceCommand::SelectAccount { account_id } => {
            select_account(account_id, state, event_tx, internal_tx)
        }
        ServiceCommand::SearchContracts { query, limit } => {
            search_contracts_command(query, limit, state, event_tx).await
        }
        ServiceCommand::SubscribeBars {
            contract,
            bar_type,
            candle_mode,
        } => {
            subscribe_bars(
                contract,
                bar_type,
                candle_mode,
                state,
                event_tx,
                market_tx,
                internal_tx,
            )
            .await
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

#[cfg(feature = "replay")]
pub(super) async fn replace_replay_lookup(state: &mut ServiceState) {
    if let Some(previous) = state.replay_lookup_job.take() {
        previous.task.abort();
        let _ = previous.task.await;
    }
}

pub(super) async fn reap_finished_replay_download(state: &mut ServiceState) {
    let finished = state
        .replay_download_job
        .as_ref()
        .is_some_and(|job| job.task.is_finished());
    if finished && let Some(job) = state.replay_download_job.take() {
        let _ = job.task.await;
    }
}

pub(super) async fn cancel_replay_operation(
    operation_id: ReplayDownloadOperationId,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
) {
    if state
        .replay_lookup_job
        .as_ref()
        .is_some_and(|job| job.operation_id == operation_id)
        && let Some(job) = state.replay_lookup_job.take()
    {
        job.task.abort();
        let _ = job.task.await;
        let _ = event_tx.send(ServiceEvent::ReplayDownloadFailed {
            operation_id,
            phase: ReplayDownloadPhase::Cancelled,
            message: "Replay contract lookup cancelled.".to_string(),
        });
        return;
    }

    reap_finished_replay_download(state).await;
    let Some(job) = state.replay_download_job.as_ref() else {
        return;
    };
    if job.operation_id != operation_id {
        return;
    }
    match job.request_cancellation() {
        ReplayDownloadJobStage::CancelRequested => {
            let _ = event_tx.send(ServiceEvent::ReplayDownloadProgress {
                operation_id,
                phase: ReplayDownloadPhase::Cancelling,
                message: "Cancelling replay download before cache commit...".to_string(),
                estimated_rows: None,
                estimated_bytes: None,
            });
        }
        ReplayDownloadJobStage::Network => unreachable!("cancellation claim returned network"),
        ReplayDownloadJobStage::Committing => {
            let _ = event_tx.send(ServiceEvent::ReplayDownloadProgress {
                operation_id,
                phase: ReplayDownloadPhase::WritingCache,
                message: "Cancellation cannot interrupt an atomic cache commit that has already begun; the commit will finish.".to_string(),
                estimated_rows: None,
                estimated_bytes: None,
            });
        }
        ReplayDownloadJobStage::Finished => {
            reap_finished_replay_download(state).await;
        }
    }
}

#[cfg(feature = "replay")]
pub(super) fn begin_replay_cache_commit(
    stage: &AtomicU8,
) -> std::result::Result<(), (ReplayDownloadPhase, anyhow::Error)> {
    let result = stage
        .compare_exchange(
            ReplayDownloadJobStage::Network as u8,
            ReplayDownloadJobStage::Committing as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map(|_| ())
        .map_err(ReplayDownloadJobStage::from_raw);
    result.map_err(|current| match current {
        ReplayDownloadJobStage::CancelRequested => (
            ReplayDownloadPhase::Cancelled,
            anyhow::anyhow!("replay download cancelled before cache commit"),
        ),
        other => (
            ReplayDownloadPhase::WritingCache,
            anyhow::anyhow!("replay download could not begin cache commit from {other:?} state"),
        ),
    })
}

#[cfg(feature = "replay")]
pub(super) async fn wait_for_replay_download_cancellation(
    cancel_rx: &mut tokio::sync::watch::Receiver<bool>,
) {
    if *cancel_rx.borrow() {
        return;
    }
    while cancel_rx.changed().await.is_ok() {
        if *cancel_rx.borrow() {
            return;
        }
    }
}

async fn download_replay_data(
    operation_id: ReplayDownloadOperationId,
    cfg: AppConfig,
    instrument: String,
    contract: ContractSuggestion,
    target: Option<ReplayDownloadCacheTarget>,
    start_date: chrono::NaiveDate,
    end_date: chrono::NaiveDate,
    source_kind: String,
    bar_type: BarType,
    candle_mode: CandleMode,
    display_name: Option<String>,
    tags: Vec<String>,
    event_tx: &UnboundedSender<ServiceEvent>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    stage: Arc<AtomicU8>,
) -> std::result::Result<(), (ReplayDownloadPhase, anyhow::Error)> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (
            operation_id,
            cfg,
            instrument,
            contract,
            target,
            start_date,
            end_date,
            source_kind,
            bar_type,
            candle_mode,
            display_name,
            tags,
            event_tx,
            cancel_rx,
            stage,
        );
        return Err((
            ReplayDownloadPhase::Authenticating,
            anyhow::anyhow!("replay downloader requires `--features replay`"),
        ));
    }

    #[cfg(feature = "replay")]
    {
        let mut cancel_rx = cancel_rx;
        if end_date < start_date {
            return Err((
                ReplayDownloadPhase::Ready,
                anyhow::anyhow!("download end date cannot be before start date"),
            ));
        }
        let start = Utc
            .with_ymd_and_hms(
                start_date.year(),
                start_date.month(),
                start_date.day(),
                0,
                0,
                0,
            )
            .single()
            .context("compose download start date")
            .map_err(|err| (ReplayDownloadPhase::Ready, err))?;
        let end_exclusive_date = end_date
            .succ_opt()
            .context("download end date overflowed")
            .map_err(|err| (ReplayDownloadPhase::Ready, err))?;
        let end = Utc
            .with_ymd_and_hms(
                end_exclusive_date.year(),
                end_exclusive_date.month(),
                end_exclusive_date.day(),
                0,
                0,
                0,
            )
            .single()
            .context("compose download end date")
            .map_err(|err| (ReplayDownloadPhase::Ready, err))?;
        let source_kind = match source_kind.as_str() {
            "server-bars" => ReplayCacheSourceKind::ServerBars,
            "raw-ticks" => ReplayCacheSourceKind::RawTicks,
            other => {
                return Err((
                    ReplayDownloadPhase::Ready,
                    anyhow::anyhow!("unsupported replay download source kind: {other}"),
                ));
            }
        };

        let _ = event_tx.send(ServiceEvent::ReplayDownloadProgress {
            operation_id,
            phase: ReplayDownloadPhase::Authenticating,
            message: format!(
                "Authenticating for {} {} {} to {}.",
                instrument, contract.name, start_date, end_date
            ),
            estimated_rows: None,
            estimated_bytes: None,
        });

        if source_kind == ReplayCacheSourceKind::ServerBars {
            let authenticated = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let authenticated_callback = authenticated.clone();
            let progress_tx = event_tx.clone();
            let download = crate::tradovate::download_replay_server_bars_after_auth(
                &cfg,
                crate::tradovate::TradovateServerBarDownloadRequest {
                    contract: contract.name.clone(),
                    exact_contract: Some(contract),
                    start,
                    end,
                    bar_type,
                },
                move || {
                    authenticated_callback.store(true, std::sync::atomic::Ordering::Release);
                    let _ = progress_tx.send(ServiceEvent::ReplayDownloadProgress {
                        operation_id,
                        phase: ReplayDownloadPhase::Downloading,
                        message: format!("Downloading server bars: {}.", bar_type.label()),
                        estimated_rows: None,
                        estimated_bytes: None,
                    });
                },
            );
            tokio::pin!(download);
            let download = tokio::select! {
                biased;
                _ = wait_for_replay_download_cancellation(&mut cancel_rx) => {
                    return Err((
                        ReplayDownloadPhase::Cancelled,
                        anyhow::anyhow!("replay download cancelled before cache commit"),
                    ));
                }
                result = &mut download => result,
            }
            .map_err(|err| {
                let phase = if authenticated.load(std::sync::atomic::Ordering::Acquire) {
                    ReplayDownloadPhase::Downloading
                } else {
                    ReplayDownloadPhase::Authenticating
                };
                (phase, err)
            })?;
            begin_replay_cache_commit(&stage)?;
            let _ = event_tx.send(ServiceEvent::ReplayDownloadProgress {
                operation_id,
                phase: ReplayDownloadPhase::WritingCache,
                message: "Writing server bars to the replay cache...".to_string(),
                estimated_rows: None,
                estimated_bytes: None,
            });
            let write = ReplayCacheServerBarsWrite {
                cache_root: cfg.replay_cache_dir.clone(),
                target,
                provider: cfg.broker,
                env: cfg.env,
                instrument: ReplayCacheInstrument {
                    symbol: instrument,
                    name: None,
                    exchange: None,
                },
                contract: ReplayCacheContract {
                    symbol: download.contract.name,
                    id: Some(download.contract.id),
                    expiration: None,
                },
                request_start: start,
                request_end: end,
                source_kind,
                download_request: download.request_body,
                bar_type,
                tick_specs: download.tick_specs,
                contract_metadata: Some(download.contract_metadata),
                session_template: download.session_template,
                bars: download.bars,
                warnings: download.warnings,
                display_name,
                tags: Some(tags),
                notes: Some(format!(
                    "Downloaded from the TUI as {} through Tradovate read-only metadata/account REST endpoints and the md/getChart market-data WebSocket; no user sync, account stream, or order path was started.",
                    bar_type.mode_label(candle_mode)
                )),
            };
            let outcome =
                tokio::task::spawn_blocking(move || write_server_bars_parquet_cache(write))
                    .await
                    .map_err(|err| (ReplayDownloadPhase::WritingCache, anyhow::Error::new(err)))?
                    .map_err(|err| (ReplayDownloadPhase::WritingCache, err))?;
            let bytes = std::fs::metadata(&outcome.data_path)
                .map(|metadata| metadata.len())
                .with_context(|| {
                    format!("read cache file metadata {}", outcome.data_path.display())
                })
                .map_err(|err| (ReplayDownloadPhase::WritingCache, err))?;
            let _ = event_tx.send(ServiceEvent::ReplayDownloadCompleted {
                operation_id,
                cache_root: cfg.replay_cache_dir.clone(),
                manifest_path: outcome.manifest_path,
                data_path: outcome.data_path,
                rows: outcome.row_count,
                bytes,
            });
        } else {
            let authenticated = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let authenticated_callback = authenticated.clone();
            let progress_tx = event_tx.clone();
            let download = crate::tradovate::download_replay_raw_ticks_after_auth(
                &cfg,
                crate::tradovate::TradovateRawTickDownloadRequest {
                    contract: contract.name.clone(),
                    exact_contract: Some(contract),
                    start,
                    end,
                },
                move || {
                    authenticated_callback.store(true, std::sync::atomic::Ordering::Release);
                    let _ = progress_tx.send(ServiceEvent::ReplayDownloadProgress {
                        operation_id,
                        phase: ReplayDownloadPhase::Downloading,
                        message: "Downloading raw ticks...".to_string(),
                        estimated_rows: None,
                        estimated_bytes: None,
                    });
                },
            );
            tokio::pin!(download);
            let download = tokio::select! {
                biased;
                _ = wait_for_replay_download_cancellation(&mut cancel_rx) => {
                    return Err((
                        ReplayDownloadPhase::Cancelled,
                        anyhow::anyhow!("replay download cancelled before cache commit"),
                    ));
                }
                result = &mut download => result,
            }
            .map_err(|err| {
                let phase = if authenticated.load(std::sync::atomic::Ordering::Acquire) {
                    ReplayDownloadPhase::Downloading
                } else {
                    ReplayDownloadPhase::Authenticating
                };
                (phase, err)
            })?;
            begin_replay_cache_commit(&stage)?;
            let _ = event_tx.send(ServiceEvent::ReplayDownloadProgress {
                operation_id,
                phase: ReplayDownloadPhase::WritingCache,
                message: "Writing raw ticks to the replay cache...".to_string(),
                estimated_rows: None,
                estimated_bytes: None,
            });
            let write = ReplayCacheRawTicksWrite {
                cache_root: cfg.replay_cache_dir.clone(),
                target,
                provider: cfg.broker,
                env: cfg.env,
                instrument: ReplayCacheInstrument {
                    symbol: instrument,
                    name: None,
                    exchange: None,
                },
                contract: ReplayCacheContract {
                    symbol: download.contract.name,
                    id: Some(download.contract.id),
                    expiration: None,
                },
                request_start: start,
                request_end: end,
                download_request: download.request_body,
                tick_specs: download.tick_specs,
                contract_metadata: Some(download.contract_metadata),
                session_template: download.session_template,
                ticks: download.ticks,
                warnings: download.warnings,
                display_name,
                tags: Some(tags),
                notes: Some(
                    "Downloaded from the TUI through Tradovate read-only metadata/account REST endpoints and the md/getChart market-data WebSocket; no user sync, account stream, or order path was started."
                        .to_string(),
                ),
            };
            let outcome = tokio::task::spawn_blocking(move || write_raw_ticks_parquet_cache(write))
                .await
                .map_err(|err| (ReplayDownloadPhase::WritingCache, anyhow::Error::new(err)))?
                .map_err(|err| (ReplayDownloadPhase::WritingCache, err))?;
            let bytes = std::fs::metadata(&outcome.data_path)
                .map(|metadata| metadata.len())
                .with_context(|| {
                    format!("read cache file metadata {}", outcome.data_path.display())
                })
                .map_err(|err| (ReplayDownloadPhase::WritingCache, err))?;
            let _ = event_tx.send(ServiceEvent::ReplayDownloadCompleted {
                operation_id,
                cache_root: cfg.replay_cache_dir.clone(),
                manifest_path: outcome.manifest_path,
                data_path: outcome.data_path,
                rows: outcome.row_count,
                bytes,
            });
        }
        Ok(())
    }
}

async fn connect_live_session(
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
    });
    if let Some(session) = state.session.as_ref() {
        emit_execution_state(event_tx, session);
    }

    Ok(())
}

async fn enter_replay_mode(
    cfg: AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
    replay_dataset_manifest: Option<std::path::PathBuf>,
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

    let replay = replay::load_replay_state(
        &cfg,
        bar_type,
        candle_mode,
        replay_dataset_manifest.as_deref(),
    )
    .await?;
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
        candle_mode,
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
    candle_mode: CandleMode,
    state: &mut ServiceState,
    event_tx: &UnboundedSender<ServiceEvent>,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let candle_mode = bar_type.effective_candle_mode(candle_mode);
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
            candle_mode,
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
    #[cfg(not(feature = "manual-orders"))]
    {
        let _ = action;
        let _ = state;
        let _ = event_tx.send(ServiceEvent::Error(
            "Manual order commands are disabled; rebuild with --features manual-orders."
                .to_string(),
        ));
        return Ok(());
    }

    #[cfg(feature = "manual-orders")]
    {
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
    };
    if uses_protection && config.native_reversal_mode == NativeReversalMode::Direct {
        config.native_reversal_mode = NativeReversalMode::CloseAllEnter;
    }
    if uses_protection || config.native_reversal_mode != NativeReversalMode::Direct {
        config.native_execution_path = NativeExecutionPath::Guarded;
    }
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

async fn reset_state_for_new_session(
    state: &mut ServiceState,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
) {
    shutdown_tasks(state).await;
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

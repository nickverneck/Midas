use super::{connection::*, market::*, orders::*, replay_jobs::*, strategy::*, *};

pub(in crate::tradovate::service) async fn handle_command(
    cmd: ServiceCommand,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
    market_tx: &tokio::sync::watch::Sender<MarketSnapshot>,
    internal_tx: InternalEventSender,
) -> Result<()> {
    match cmd {
        ServiceCommand::Connect(cfg) => {
            // The login form can change env after the initial config load.
            // Revalidate at the actual network boundary so an enabled local
            // proxy can never silently fall through to the live endpoint.
            cfg.validate()?;
            connect_live_session(cfg, state, event_tx, market_tx, internal_tx).await
        }
        ServiceCommand::EnterReplayMode {
            config: cfg,
            bar_type,
            candle_mode,
            replay_dataset_manifest,
            replay_dataset_view,
        } => {
            enter_replay_mode(
                cfg,
                bar_type,
                candle_mode,
                replay_dataset_manifest,
                replay_dataset_view,
                None,
                state,
                event_tx,
                market_tx,
                internal_tx,
            )
            .await
        }
        #[cfg(feature = "replay")]
        ServiceCommand::EnterReplayModeWithSharedFrames {
            config: cfg,
            bar_type,
            candle_mode,
            replay_dataset_manifest,
            replay_dataset_view,
            replay_shared_frames,
        } => {
            enter_replay_mode(
                cfg,
                bar_type,
                candle_mode,
                replay_dataset_manifest,
                replay_dataset_view,
                replay_shared_frames,
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
        ServiceCommand::InspectState => inspect_state(state, event_tx),
        ServiceCommand::ReplayState => replay_state(state, event_tx, &internal_tx).await,
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

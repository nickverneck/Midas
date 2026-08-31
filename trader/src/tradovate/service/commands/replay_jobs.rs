use super::*;

#[cfg(feature = "replay")]
pub(in crate::tradovate::service) async fn replace_replay_lookup(state: &mut ServiceState) {
    if let Some(previous) = state.replay_lookup_job.take() {
        previous.task.abort();
        let _ = previous.task.await;
    }
}

pub(in crate::tradovate::service) async fn reap_finished_replay_download(state: &mut ServiceState) {
    let finished = state
        .replay_download_job
        .as_ref()
        .is_some_and(|job| job.task.is_finished());
    if finished && let Some(job) = state.replay_download_job.take() {
        let _ = job.task.await;
    }
}

pub(in crate::tradovate::service) async fn cancel_replay_operation(
    operation_id: ReplayDownloadOperationId,
    state: &mut ServiceState,
    event_tx: &ServiceEventSender,
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
                message: "Cancellation cannot interrupt the atomic cache commit already in progress. That commit will finish; any remaining resumable chunks will stop afterward.".to_string(),
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
pub(in crate::tradovate::service) fn begin_replay_cache_commit(
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

pub(super) async fn download_replay_data(
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
    event_tx: &ServiceEventSender,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    stage: Arc<AtomicU8>,
) -> std::result::Result<(), (ReplayDownloadPhase, anyhow::Error)> {
    if cfg.simulation_proxy.enabled {
        return Err((
            ReplayDownloadPhase::Ready,
            anyhow::anyhow!(
                "replay downloads are unavailable in simulation_proxy mode; use the standalone proxy fixture"
            ),
        ));
    }
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
            let auth_progress_tx = event_tx.clone();
            let chunk_progress_tx = event_tx.clone();
            let stage_before_commit = stage.clone();
            let stage_after_commit = stage.clone();
            let outcome = crate::tradovate::download_replay_raw_ticks_chunked_to_cache(
                &cfg,
                crate::tradovate::TradovateChunkedRawTickCacheRequest {
                    instrument,
                    contract: contract.name.clone(),
                    exact_contract: Some(contract),
                    target,
                    start,
                    end,
                    cache_root: cfg.replay_cache_dir.clone(),
                    chunk_duration: chrono::Duration::minutes(60),
                    minimum_split: chrono::Duration::minutes(5),
                    display_name,
                    tags: Some(tags),
                    notes: Some(
                        "Downloaded from the TUI through bounded, resumable Tradovate read-only metadata/account REST and md/getChart market-data requests; no user sync, account stream, or order path was started."
                            .to_string(),
                    ),
                },
                Some(cancel_rx.clone()),
                move || {
                    authenticated_callback.store(true, std::sync::atomic::Ordering::Release);
                    let _ = auth_progress_tx.send(ServiceEvent::ReplayDownloadProgress {
                        operation_id,
                        phase: ReplayDownloadPhase::Downloading,
                        message: "Authenticated; preparing resumable raw-tick chunks...".to_string(),
                        estimated_rows: None,
                        estimated_bytes: None,
                    });
                },
                move |progress| {
                    let phase = match progress.phase {
                        crate::tradovate::TradovateChunkedRawTickPhase::CommittingChunk
                        | crate::tradovate::TradovateChunkedRawTickPhase::Finalizing => {
                            ReplayDownloadPhase::WritingCache
                        }
                        _ => ReplayDownloadPhase::Downloading,
                    };
                    let _ = chunk_progress_tx.send(ServiceEvent::ReplayDownloadProgress {
                        operation_id,
                        phase,
                        message: format!(
                            "{} ({}/{} chunks complete)",
                            progress.message,
                            progress.completed_chunks,
                            progress.total_leaf_chunks
                        ),
                        estimated_rows: None,
                        estimated_bytes: None,
                    });
                },
                move || begin_replay_cache_commit(&stage_before_commit).map_err(|(_, err)| err),
                move |finalizing, succeeded| {
                    if !finalizing && succeeded {
                        let _ = stage_after_commit.compare_exchange(
                            ReplayDownloadJobStage::Committing as u8,
                            ReplayDownloadJobStage::Network as u8,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                    }
                },
            )
            .await
            .map_err(|err| {
                let phase = if *cancel_rx.borrow() {
                    ReplayDownloadPhase::Cancelled
                } else if ReplayDownloadJobStage::from_raw(stage.load(Ordering::Acquire))
                    == ReplayDownloadJobStage::Committing
                {
                    ReplayDownloadPhase::WritingCache
                } else if authenticated.load(std::sync::atomic::Ordering::Acquire) {
                    ReplayDownloadPhase::Downloading
                } else {
                    ReplayDownloadPhase::Authenticating
                };
                (phase, err)
            })?;
            let data_path = outcome.data_paths.first().cloned().ok_or_else(|| {
                (
                    ReplayDownloadPhase::WritingCache,
                    anyhow::anyhow!("completed raw-tick manifest has no cache data files"),
                )
            })?;
            let bytes = outcome
                .data_paths
                .iter()
                .try_fold(0_u64, |total, path| {
                    std::fs::metadata(path)
                        .map(|metadata| total.saturating_add(metadata.len()))
                        .with_context(|| format!("read cache file metadata {}", path.display()))
                })
                .map_err(|err| (ReplayDownloadPhase::WritingCache, err))?;
            let _ = event_tx.send(ServiceEvent::ReplayDownloadCompleted {
                operation_id,
                cache_root: cfg.replay_cache_dir.clone(),
                manifest_path: outcome.manifest_path,
                data_path,
                rows: outcome.row_count,
                bytes,
            });
        }
        Ok(())
    }
}

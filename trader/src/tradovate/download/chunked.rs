use super::*;

pub async fn download_replay_raw_ticks_chunked_to_cache<F, P, B, A>(
    cfg: &AppConfig,
    mut request: TradovateChunkedRawTickCacheRequest,
    mut cancel_rx: Option<tokio::sync::watch::Receiver<bool>>,
    on_authenticated: F,
    mut on_progress: P,
    mut before_commit: B,
    mut after_commit: A,
) -> Result<ReplayCacheChunkedWriteOutcome>
where
    F: FnOnce(),
    P: FnMut(TradovateChunkedRawTickProgress),
    B: FnMut() -> Result<()>,
    A: FnMut(bool, bool),
{
    let requested_end = request.end;
    request.end = effective_raw_tick_request_end(
        request.start,
        request.end,
        Utc::now(),
        request.chunk_duration,
    )?;
    let end_was_clamped = request.end < requested_end;
    let root_window = DownloadWindow::new(request.start, request.end)?;
    let initial_windows = plan_fixed_windows(root_window, request.chunk_duration)?;
    let session = prepare_replay_download_session_after_auth(
        cfg,
        &request.contract,
        request.exact_contract.as_ref(),
        on_authenticated,
    )
    .await?;
    let contract = session.contract.clone();
    let root_request = TradovateRawTickDownloadRequest {
        contract: contract.name.clone(),
        exact_contract: Some(contract.clone()),
        start: request.start,
        end: request.end,
    };
    let mut warnings = session.warnings.clone();
    if end_was_clamped {
        warnings.push(format!(
            "Requested raw-tick end {requested_end} was later than the download snapshot; coverage was clamped to {} so future intervals remain refreshable.",
            request.end
        ));
    }
    let plan_write = ReplayCacheRawTickChunkPlanWrite {
        cache_root: request.cache_root,
        target: request.target,
        identity: ReplayCacheRawTickCheckpointIdentity {
            provider: cfg.broker,
            env: cfg.env,
            instrument: ReplayCacheInstrument {
                symbol: request.instrument,
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: contract.name.clone(),
                id: Some(contract.id),
                expiration: None,
            },
            request: root_window,
            chunk_seconds: request.chunk_duration.num_seconds(),
            request_protocol_version: 6,
        },
        download_request: raw_tick_chart_request_body(&contract, &root_request),
        tick_specs: session.tick_specs.clone(),
        contract_metadata: Some(session.contract_metadata.clone()),
        session_template: session.session_template.clone(),
        warnings,
        display_name: request.display_name,
        tags: request.tags,
        notes: request.notes,
    };
    let plan_for_prepare = plan_write.clone();
    let initial_for_prepare = initial_windows.clone();
    let mut state = tokio::task::spawn_blocking(move || {
        prepare_raw_tick_chunk_cache(&plan_for_prepare, &initial_for_prepare)
    })
    .await
    .context("join raw-tick checkpoint preparation")??;
    on_progress(TradovateChunkedRawTickProgress {
        phase: TradovateChunkedRawTickPhase::Resuming,
        window: None,
        completed_chunks: state.checkpoint.completed_leaf_count(),
        total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint),
        message: format!(
            "Raw-tick checkpoint: {} completed, {} pending.",
            state.checkpoint.completed_leaf_count(),
            state.checkpoint.pending_windows().len()
        ),
    });

    while let Some(window) = state.checkpoint.pending_windows().first().copied() {
        if replay_download_cancelled(cancel_rx.as_ref()) {
            bail!("replay raw-tick download cancelled; completed chunks remain resumable");
        }
        let mut retry = 0_u8;
        loop {
            on_progress(TradovateChunkedRawTickProgress {
                phase: TradovateChunkedRawTickPhase::Downloading,
                window: Some(window),
                completed_chunks: state.checkpoint.completed_leaf_count(),
                total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint),
                message: format!(
                    "Downloading raw ticks for [{} , {}).",
                    window.start, window.end
                ),
            });
            let download = session.download_raw_ticks(window.start, window.end);
            tokio::pin!(download);
            let result = if let Some(receiver) = cancel_rx.as_mut() {
                tokio::select! {
                    biased;
                    _ = wait_for_chunked_download_cancellation(receiver) => {
                        bail!("replay raw-tick download cancelled; completed chunks remain resumable");
                    }
                    result = &mut download => result,
                }
            } else {
                download.await
            };
            match result {
                Ok(download) => {
                    before_commit()?;
                    on_progress(TradovateChunkedRawTickProgress {
                        phase: TradovateChunkedRawTickPhase::CommittingChunk,
                        window: Some(window),
                        completed_chunks: state.checkpoint.completed_leaf_count(),
                        total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint),
                        message: format!(
                            "Committing completed raw-tick chunk [{} , {}).",
                            window.start, window.end
                        ),
                    });
                    let write_for_chunk = plan_write.clone();
                    let telemetry = download.telemetry;
                    let ticks = download.ticks;
                    let commit = tokio::task::spawn_blocking(move || {
                        write_raw_tick_chunk_cache(&write_for_chunk, window, ticks, telemetry)
                    })
                    .await
                    .context("join raw-tick chunk cache commit")?;
                    let committed = commit.is_ok();
                    after_commit(false, committed);
                    commit?;
                    let plan_for_reload = plan_write.clone();
                    let initial_for_reload = initial_windows.clone();
                    state = tokio::task::spawn_blocking(move || {
                        prepare_raw_tick_chunk_cache(&plan_for_reload, &initial_for_reload)
                    })
                    .await
                    .context("join raw-tick checkpoint reload")??;
                    break;
                }
                Err(error) => {
                    let Some(failure) = error.downcast_ref::<HistoricalDownloadFailure>() else {
                        return Err(error);
                    };
                    let evidence = serde_json::to_value(failure)
                        .context("serialize sanitized raw-tick failure evidence")?;
                    if matches!(
                        failure.kind,
                        HistoricalDownloadFailureKind::Authentication
                            | HistoricalDownloadFailureKind::Authorization
                    ) {
                        let plan_for_attempt = plan_write.clone();
                        tokio::task::spawn_blocking(move || {
                            record_raw_tick_chunk_attempt(&plan_for_attempt, window, evidence)
                        })
                        .await
                        .context("join raw-tick auth failure checkpoint")??;
                        return Err(error);
                    }
                    if failure.kind.is_transient() && retry < 2 {
                        let delay_secs = 5_u64.saturating_mul(1_u64 << retry);
                        retry = retry.saturating_add(1);
                        let plan_for_attempt = plan_write.clone();
                        let evidence_for_attempt = evidence.clone();
                        tokio::task::spawn_blocking(move || {
                            record_raw_tick_chunk_attempt(
                                &plan_for_attempt,
                                window,
                                evidence_for_attempt,
                            )
                        })
                        .await
                        .context("join raw-tick retry checkpoint")??;
                        on_progress(TradovateChunkedRawTickProgress {
                            phase: TradovateChunkedRawTickPhase::Retrying,
                            window: Some(window),
                            completed_chunks: state.checkpoint.completed_leaf_count(),
                            total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint),
                            message: format!(
                                "Transient raw-tick failure; retrying in {delay_secs}s (attempt {retry}/2)."
                            ),
                        });
                        tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                        continue;
                    }
                    if raw_tick_failure_should_split(failure) {
                        let plan_for_split = plan_write.clone();
                        let minimum = request.minimum_split;
                        let split = tokio::task::spawn_blocking(move || {
                            split_raw_tick_checkpoint_chunk(
                                &plan_for_split,
                                window,
                                &[],
                                minimum,
                                evidence,
                            )
                        })
                        .await
                        .context("join raw-tick checkpoint split")?;
                        match split {
                            Ok((left, right)) => {
                                on_progress(TradovateChunkedRawTickProgress {
                                    phase: TradovateChunkedRawTickPhase::Splitting,
                                    window: Some(window),
                                    completed_chunks: state.checkpoint.completed_leaf_count(),
                                    total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint) + 1,
                                    message: format!(
                                        "Split incomplete raw-tick request into [{} , {}) and [{} , {}).",
                                        left.start, left.end, right.start, right.end
                                    ),
                                });
                                let plan_for_reload = plan_write.clone();
                                let initial_for_reload = initial_windows.clone();
                                state = tokio::task::spawn_blocking(move || {
                                    prepare_raw_tick_chunk_cache(
                                        &plan_for_reload,
                                        &initial_for_reload,
                                    )
                                })
                                .await
                                .context("join split checkpoint reload")??;
                                break;
                            }
                            Err(split_error) => return Err(split_error.context(error.to_string())),
                        }
                    }
                    let plan_for_attempt = plan_write.clone();
                    tokio::task::spawn_blocking(move || {
                        record_raw_tick_chunk_attempt(&plan_for_attempt, window, evidence)
                    })
                    .await
                    .context("join raw-tick failure checkpoint")??;
                    return Err(error);
                }
            }
        }
    }
    if replay_download_cancelled(cancel_rx.as_ref()) {
        bail!("replay raw-tick download cancelled; completed chunks remain resumable");
    }
    before_commit()?;
    on_progress(TradovateChunkedRawTickProgress {
        phase: TradovateChunkedRawTickPhase::Finalizing,
        window: None,
        completed_chunks: state.checkpoint.completed_leaf_count(),
        total_leaf_chunks: raw_tick_leaf_count(&state.checkpoint),
        message: "Publishing complete raw-tick manifest coverage.".to_string(),
    });
    let write_for_finalize = plan_write;
    let finalized =
        tokio::task::spawn_blocking(move || finalize_raw_tick_chunk_cache(&write_for_finalize))
            .await
            .context("join raw-tick manifest finalization")?;
    after_commit(true, finalized.is_ok());
    finalized
}

pub(super) fn raw_tick_failure_should_split(failure: &HistoricalDownloadFailure) -> bool {
    failure.kind == HistoricalDownloadFailureKind::SizeRejected
        || (failure.kind.is_splittable()
            && (failure.telemetry.historical_id.is_some() || failure.telemetry.provider_rows > 0))
}

pub(super) fn effective_raw_tick_request_end(
    start: DateTime<Utc>,
    requested_end: DateTime<Utc>,
    snapshot_now: DateTime<Utc>,
    chunk_duration: chrono::Duration,
) -> Result<DateTime<Utc>> {
    if chunk_duration <= chrono::Duration::zero() {
        bail!("raw-tick chunk duration must be positive");
    }
    let effective_end = if requested_end <= snapshot_now {
        requested_end
    } else {
        let elapsed_ms = (snapshot_now - start).num_milliseconds().max(0);
        let chunk_ms = chunk_duration.num_milliseconds();
        if chunk_ms <= 0 {
            bail!("raw-tick chunk duration must be at least one millisecond");
        }
        let completed_chunks = elapsed_ms / chunk_ms;
        start + chrono::Duration::milliseconds(completed_chunks.saturating_mul(chunk_ms))
    };
    if start >= effective_end {
        bail!(
            "raw-tick request starts at {start}, but no completed historical interval exists before snapshot {snapshot_now}"
        );
    }
    Ok(effective_end)
}

fn raw_tick_leaf_count(checkpoint: &crate::replay_cache::ReplayCacheRawTickCheckpoint) -> usize {
    checkpoint
        .chunks
        .iter()
        .filter(|chunk| {
            !matches!(
                chunk.status,
                crate::replay_cache::ReplayCacheRawTickChunkStatus::Split { .. }
            )
        })
        .count()
}

pub(super) fn replay_download_cancelled(
    cancel_rx: Option<&tokio::sync::watch::Receiver<bool>>,
) -> bool {
    cancel_rx.is_some_and(|receiver| *receiver.borrow())
}

async fn wait_for_chunked_download_cancellation(
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

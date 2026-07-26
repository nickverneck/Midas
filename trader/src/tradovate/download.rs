use super::*;
use crate::replay_cache::{
    ReplayCacheChunkedWriteOutcome, ReplayCacheContract, ReplayCacheContractMetadata,
    ReplayCacheInstrument, ReplayCacheMetadataAccount, ReplayCacheMetadataContext,
    ReplayCacheMetadataSnapshot, ReplayCacheRawTickChunkPlanWrite,
    ReplayCacheRawTickCheckpointIdentity, ReplayCacheRawTickRow, ReplayCacheSuggestedCoverage,
    ReplayCacheTickSpecs, finalize_raw_tick_chunk_cache, prepare_raw_tick_chunk_cache,
    record_raw_tick_chunk_attempt, split_raw_tick_checkpoint_chunk, write_raw_tick_chunk_cache,
};
use crate::replay_download::{
    DownloadCompletionEvidence, DownloadWindow, HistoricalDownloadFailure,
    HistoricalDownloadFailureKind, HistoricalDownloadTelemetry, plan_fixed_windows,
};

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownloadRequest {
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub bar_type: BarType,
}

#[derive(Debug, Clone)]
pub struct TradovateServerBarDownload {
    pub contract: ContractSuggestion,
    pub bars: Vec<Bar>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub contract_metadata: ReplayCacheContractMetadata,
    pub warnings: Vec<String>,
    pub telemetry: HistoricalDownloadTelemetry,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownloadRequest {
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct TradovateRawTickDownload {
    pub contract: ContractSuggestion,
    pub ticks: Vec<ReplayCacheRawTickRow>,
    pub request_body: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub contract_metadata: ReplayCacheContractMetadata,
    pub warnings: Vec<String>,
    pub telemetry: HistoricalDownloadTelemetry,
}

#[derive(Debug, Clone)]
pub struct TradovateReplayContractInspection {
    pub contract: ContractSuggestion,
    pub suggested_coverage: Option<ReplayCacheSuggestedCoverage>,
}

#[derive(Debug, Clone)]
pub struct TradovateChunkedRawTickCacheRequest {
    pub instrument: String,
    pub contract: String,
    pub exact_contract: Option<ContractSuggestion>,
    pub target: Option<ReplayDownloadCacheTarget>,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub cache_root: PathBuf,
    pub chunk_duration: chrono::Duration,
    pub minimum_split: chrono::Duration,
    pub display_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradovateChunkedRawTickPhase {
    Resuming,
    Downloading,
    Retrying,
    Splitting,
    CommittingChunk,
    Finalizing,
}

#[derive(Debug, Clone)]
pub struct TradovateChunkedRawTickProgress {
    pub phase: TradovateChunkedRawTickPhase,
    pub window: Option<DownloadWindow>,
    pub completed_chunks: usize,
    pub total_leaf_chunks: usize,
    pub message: String,
}

/// Read-only authenticated context shared by bounded historical requests. The
/// market-data token intentionally has no Debug/Serialize surface.
pub struct TradovateReplayDownloadSession {
    cfg: AppConfig,
    md_access_token: String,
    contract: ContractSuggestion,
    tick_specs: ReplayCacheTickSpecs,
    session_template: Option<String>,
    contract_metadata: ReplayCacheContractMetadata,
    warnings: Vec<String>,
}

impl TradovateReplayDownloadSession {
    pub fn contract(&self) -> &ContractSuggestion {
        &self.contract
    }

    pub fn contract_metadata(&self) -> &ReplayCacheContractMetadata {
        &self.contract_metadata
    }

    pub async fn download_server_bars(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        bar_type: BarType,
    ) -> Result<TradovateServerBarDownload> {
        let request = TradovateServerBarDownloadRequest {
            contract: self.contract.name.clone(),
            exact_contract: Some(self.contract.clone()),
            start,
            end,
            bar_type,
        };
        let request_body = server_bar_chart_request_body(&self.contract, &request);
        let history = fetch_server_bars_over_market_ws(
            &self.cfg,
            &self.md_access_token,
            request_body.clone(),
            start,
            end,
        )
        .await?;
        let mut warnings = self.warnings.clone();
        if history.rows.is_empty() {
            warnings.push("Tradovate returned zero historical server bars with explicit end-of-history.".to_string());
        }
        Ok(TradovateServerBarDownload {
            contract: self.contract.clone(),
            bars: history.rows,
            request_body,
            tick_specs: self.tick_specs.clone(),
            session_template: self.session_template.clone(),
            contract_metadata: self.contract_metadata.clone(),
            warnings,
            telemetry: history.telemetry,
        })
    }

    pub async fn download_raw_ticks(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<TradovateRawTickDownload> {
        let request = TradovateRawTickDownloadRequest {
            contract: self.contract.name.clone(),
            exact_contract: Some(self.contract.clone()),
            start,
            end,
        };
        let request_body = raw_tick_chart_request_body(&self.contract, &request);
        let tick_size = (self.tick_specs.tick_size.is_finite() && self.tick_specs.tick_size > 0.0)
            .then_some(self.tick_specs.tick_size);
        let history = fetch_raw_ticks_over_market_ws(
            &self.cfg,
            &self.md_access_token,
            request_body.clone(),
            start,
            end,
            tick_size,
        )
        .await?;
        let mut warnings = self.warnings.clone();
        if history.rows.is_empty() {
            warnings.push("Tradovate returned zero historical raw ticks with explicit end-of-history.".to_string());
        }
        Ok(TradovateRawTickDownload {
            contract: self.contract.clone(),
            ticks: history.rows,
            request_body,
            tick_specs: self.tick_specs.clone(),
            session_template: self.session_template.clone(),
            contract_metadata: self.contract_metadata.clone(),
            warnings,
            telemetry: history.telemetry,
        })
    }
}

pub async fn prepare_replay_download_session(
    cfg: &AppConfig,
    contract_symbol: &str,
    exact_contract: Option<&ContractSuggestion>,
) -> Result<TradovateReplayDownloadSession> {
    prepare_replay_download_session_after_auth(cfg, contract_symbol, exact_contract, || {}).await
}

pub async fn prepare_replay_download_session_after_auth<F>(
    cfg: &AppConfig,
    contract_symbol: &str,
    exact_contract: Option<&ContractSuggestion>,
    on_authenticated: F,
) -> Result<TradovateReplayDownloadSession>
where
    F: FnOnce(),
{
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    on_authenticated();
    let contract = resolve_requested_download_contract(
        &client,
        &cfg.env,
        &tokens.access_token,
        contract_symbol,
        exact_contract,
        cfg.contract_suggest_limit.max(12),
    )
    .await?;
    let (contract_metadata, warnings) =
        fetch_download_contract_metadata(&client, cfg, &tokens, &contract).await;
    validate_contract_identity(&contract, &contract_metadata.contract.payload)?;
    let specs = market_specs_from_metadata(&contract_metadata);
    let tick_specs = ReplayCacheTickSpecs {
        tick_size: specs
            .as_ref()
            .and_then(|market_specs| market_specs.tick_size)
            .unwrap_or(0.0),
        value_per_point: specs
            .as_ref()
            .and_then(|market_specs| market_specs.value_per_point)
            .unwrap_or(0.0),
    };
    let session_template = specs.as_ref().and_then(|market_specs| {
        market_specs
            .session_profile
            .map(|profile| profile.label().to_string())
    });
    Ok(TradovateReplayDownloadSession {
        cfg: cfg.clone(),
        md_access_token: tokens.md_access_token,
        contract,
        tick_specs,
        session_template,
        contract_metadata,
        warnings,
    })
}

pub async fn download_replay_raw_ticks_chunked_to_cache<F, P, B, A>(
    cfg: &AppConfig,
    request: TradovateChunkedRawTickCacheRequest,
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
    A: FnMut(),
{
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
        },
        download_request: raw_tick_chart_request_body(&contract, &root_request),
        tick_specs: session.tick_specs.clone(),
        contract_metadata: Some(session.contract_metadata.clone()),
        session_template: session.session_template.clone(),
        warnings: session.warnings.clone(),
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
                    after_commit();
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
                    if failure.kind.is_splittable() {
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
    let finalized = tokio::task::spawn_blocking(move || {
        finalize_raw_tick_chunk_cache(&write_for_finalize)
    })
    .await
    .context("join raw-tick manifest finalization")?;
    after_commit();
    finalized
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

fn replay_download_cancelled(
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

pub async fn search_replay_download_contracts(
    cfg: &AppConfig,
    query: &str,
    limit: usize,
) -> Result<Vec<ContractSuggestion>> {
    let query = query.trim();
    if query.is_empty() {
        bail!("replay contract search query cannot be empty");
    }
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let snapshot = fetch_metadata_snapshot(
        &client,
        &cfg.env,
        &tokens.access_token,
        "contract/suggest",
        &[("t", query.to_string()), ("l", limit.max(1).to_string())],
    )
    .await?;
    let mut seen = std::collections::BTreeSet::new();
    Ok(snapshot
        .payload
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = item.get("id")?.as_i64()?;
            if !seen.insert(id) {
                return None;
            }
            let name = item.get("name")?.as_str()?.to_string();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "contractMaturityId={}",
                        item.get("contractMaturityId")
                            .and_then(Value::as_i64)
                            .unwrap_or_default()
                    )
                });
            Some(ContractSuggestion {
                id,
                name,
                description,
                raw: item.clone(),
            })
        })
        .collect())
}

pub async fn inspect_replay_download_contract(
    cfg: &AppConfig,
    contract: ContractSuggestion,
) -> Result<TradovateReplayContractInspection> {
    let client = Client::new();
    let tokens = authenticate(&client, cfg).await?;
    let (metadata, _) = fetch_download_contract_metadata(&client, cfg, &tokens, &contract).await;
    validate_contract_identity(&contract, &metadata.contract.payload)?;
    Ok(TradovateReplayContractInspection {
        contract,
        suggested_coverage: metadata.suggested_coverage,
    })
}

pub async fn download_replay_server_bars(
    cfg: &AppConfig,
    request: TradovateServerBarDownloadRequest,
) -> Result<TradovateServerBarDownload> {
    download_replay_server_bars_after_auth(cfg, request, || {}).await
}

pub async fn download_replay_server_bars_after_auth<F>(
    cfg: &AppConfig,
    request: TradovateServerBarDownloadRequest,
    on_authenticated: F,
) -> Result<TradovateServerBarDownload>
where
    F: FnOnce(),
{
    let session = prepare_replay_download_session_after_auth(
        cfg,
        &request.contract,
        request.exact_contract.as_ref(),
        on_authenticated,
    )
    .await?;
    session
        .download_server_bars(request.start, request.end, request.bar_type)
        .await
}

pub async fn download_replay_raw_ticks(
    cfg: &AppConfig,
    request: TradovateRawTickDownloadRequest,
) -> Result<TradovateRawTickDownload> {
    download_replay_raw_ticks_after_auth(cfg, request, || {}).await
}

pub async fn download_replay_raw_ticks_after_auth<F>(
    cfg: &AppConfig,
    request: TradovateRawTickDownloadRequest,
    on_authenticated: F,
) -> Result<TradovateRawTickDownload>
where
    F: FnOnce(),
{
    let session = prepare_replay_download_session_after_auth(
        cfg,
        &request.contract,
        request.exact_contract.as_ref(),
        on_authenticated,
    )
    .await?;
    session.download_raw_ticks(request.start, request.end).await
}

async fn fetch_download_contract_metadata(
    client: &Client,
    cfg: &AppConfig,
    tokens: &TokenBundle,
    contract: &ContractSuggestion,
) -> (ReplayCacheContractMetadata, Vec<String>) {
    let fetched_at = Utc::now();
    let mut warnings = Vec::new();
    let (accounts, accounts_fetched_at) =
        match fetch_replay_metadata_accounts(client, &cfg.env, &tokens.access_token).await {
            Ok(accounts) => (accounts, Some(Utc::now())),
            Err(err) => {
                warnings.push(optional_metadata_warning("account/list", &err));
                (Vec::new(), None)
            }
        };
    let context = ReplayCacheMetadataContext {
        provider: cfg.broker,
        env: cfg.env,
        user_id: tokens.user_id,
        user_name: tokens.user_name.clone(),
        accounts,
        accounts_endpoint: Some("account/list".to_string()),
        accounts_fetched_at,
    };
    let mut contract_snapshot = ReplayCacheMetadataSnapshot {
        endpoint: "contract/suggest".to_string(),
        fetched_at,
        source_timestamp: metadata_source_timestamp(&contract.raw),
        payload: contract.raw.clone(),
    };

    match fetch_metadata_snapshot(
        client,
        &cfg.env,
        &tokens.access_token,
        "contract/item",
        &[("id", contract.id.to_string())],
    )
    .await
    {
        Ok(snapshot) => contract_snapshot = snapshot,
        Err(err) => warnings.push(optional_metadata_warning("contract/item", &err)),
    }

    let contract_payload = &contract_snapshot.payload;
    let maturity_id = json_i64(contract_payload, "contractMaturityId")
        .or_else(|| json_i64(&contract.raw, "contractMaturityId"));
    let maturity = if let Some(maturity_id) = maturity_id {
        fetch_optional_metadata(
            client,
            &cfg.env,
            &tokens.access_token,
            "contractMaturity/item",
            &[("id", maturity_id.to_string())],
            &mut warnings,
        )
        .await
    } else {
        warnings.push(
            "Optional replay metadata contractMaturity/item was skipped: contractMaturityId was unavailable."
                .to_string(),
        );
        None
    };
    let product_id = maturity
        .as_ref()
        .and_then(|snapshot| json_i64(&snapshot.payload, "productId"));

    let (maturity_chain, product, product_sessions, product_margins, fee_params) =
        if let Some(product_id) = product_id {
            let maturity_chain = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "contractMaturity/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "product/item",
                &[("id", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product_sessions = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "productSession/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let product_margins = fetch_optional_metadata(
                client,
                &cfg.env,
                &tokens.access_token,
                "productMargin/deps",
                &[("masterid", product_id.to_string())],
                &mut warnings,
            )
            .await;
            let fee_params = match fetch_fee_metadata_snapshot(
                client,
                &cfg.env,
                &tokens.access_token,
                product_id,
            )
            .await
            {
                Ok(snapshot) => Some(snapshot),
                Err(err) => {
                    warnings.push(optional_metadata_warning(
                        "contract/getproductfeeparams",
                        &err,
                    ));
                    None
                }
            };
            (
                maturity_chain,
                product,
                product_sessions,
                product_margins,
                fee_params,
            )
        } else {
            warnings.push(
                "Optional replay product metadata was skipped: productId was unavailable."
                    .to_string(),
            );
            (None, None, None, None, None)
        };
    let contract_margins = fetch_optional_metadata(
        client,
        &cfg.env,
        &tokens.access_token,
        "contractMargin/deps",
        &[("masterid", contract.id.to_string())],
        &mut warnings,
    )
    .await;
    let suggested_coverage = suggested_contract_coverage(
        maturity.as_ref().map(|snapshot| &snapshot.payload),
        maturity_chain.as_ref().map(|snapshot| &snapshot.payload),
    );
    if suggested_coverage.is_none() {
        warnings.push(
            "A suggested broad contract coverage range could not be derived from maturity metadata."
                .to_string(),
        );
    }

    (
        ReplayCacheContractMetadata {
            context,
            contract: contract_snapshot,
            maturity,
            maturity_chain,
            product,
            product_sessions,
            product_margins,
            contract_margins,
            fee_params,
            suggested_coverage,
        },
        warnings,
    )
}

async fn fetch_replay_metadata_accounts(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
) -> Result<Vec<ReplayCacheMetadataAccount>> {
    let endpoint = "account/list";
    let response = client
        .get(format!("{}/{endpoint}", env.rest_url()))
        .bearer_auth(token)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    let accounts = match payload {
        Value::Array(accounts) => accounts,
        Value::Object(_) => vec![payload],
        _ => Vec::new(),
    };
    Ok(accounts
        .into_iter()
        .filter_map(|account| {
            Some(ReplayCacheMetadataAccount {
                id: account.get("id")?.as_i64()?,
                name: account.get("name")?.as_str()?.to_string(),
            })
        })
        .collect())
}

async fn fetch_optional_metadata(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    endpoint: &str,
    query: &[(&'static str, String)],
    warnings: &mut Vec<String>,
) -> Option<ReplayCacheMetadataSnapshot> {
    match fetch_metadata_snapshot(client, env, token, endpoint, query).await {
        Ok(snapshot) => Some(snapshot),
        Err(err) => {
            warnings.push(optional_metadata_warning(endpoint, &err));
            None
        }
    }
}

async fn fetch_metadata_snapshot(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    endpoint: &str,
    query: &[(&'static str, String)],
) -> Result<ReplayCacheMetadataSnapshot> {
    let url = format!("{}/{}", env.rest_url(), endpoint);
    let response = client
        .get(url)
        .bearer_auth(token)
        .query(query)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    Ok(ReplayCacheMetadataSnapshot {
        endpoint: endpoint.to_string(),
        fetched_at: Utc::now(),
        source_timestamp: metadata_source_timestamp(&payload),
        payload,
    })
}

async fn fetch_fee_metadata_snapshot(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    product_id: i64,
) -> Result<ReplayCacheMetadataSnapshot> {
    let endpoint = "contract/getproductfeeparams";
    let url = format!("{}/{}", env.rest_url(), endpoint);
    let response = client
        .post(url)
        .bearer_auth(token)
        .json(&fee_metadata_request_body(product_id))
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(sanitized_metadata_http_error(endpoint, status, &body));
    }
    let payload: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse {endpoint} metadata response"))?;
    Ok(ReplayCacheMetadataSnapshot {
        endpoint: endpoint.to_string(),
        fetched_at: Utc::now(),
        source_timestamp: metadata_source_timestamp(&payload),
        payload,
    })
}

fn fee_metadata_request_body(product_id: i64) -> Value {
    json!({ "productIds": [product_id] })
}

fn optional_metadata_warning(endpoint: &str, err: &anyhow::Error) -> String {
    format!("Optional replay metadata {endpoint} failed: {err}")
}

fn sanitized_metadata_http_error(
    endpoint: &str,
    status: reqwest::StatusCode,
    _provider_body: &str,
) -> anyhow::Error {
    let classification = match status.as_u16() {
        401 => "authentication rejected",
        403 => "authorization rejected",
        404 => "endpoint or entity not found",
        408 => "request timed out",
        429 => "provider rate limit",
        400..=499 => "provider rejected the request",
        500..=599 => "provider server error",
        _ => "unexpected provider response",
    };
    anyhow::anyhow!(
        "{endpoint} failed with HTTP {} ({classification}); provider response body omitted",
        status.as_u16()
    )
}

fn metadata_source_timestamp(payload: &Value) -> Option<DateTime<Utc>> {
    match payload {
        Value::Array(items) => items.iter().find_map(metadata_source_timestamp),
        Value::Object(fields) => {
            for key in [
                "timestamp",
                "updatedAt",
                "lastModified",
                "definitionTimestamp",
            ] {
                if let Some(timestamp) = fields
                    .get(key)
                    .and_then(Value::as_str)
                    .and_then(parse_metadata_timestamp)
                {
                    return Some(timestamp);
                }
            }
            fields.values().find_map(metadata_source_timestamp)
        }
        _ => None,
    }
}

fn parse_metadata_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|timestamp| timestamp.with_timezone(&Utc))
}

fn metadata_expiration_date(payload: &Value) -> Option<chrono::NaiveDate> {
    let raw = payload.get("expirationDate")?.as_str()?;
    parse_metadata_timestamp(raw)
        .map(|timestamp| timestamp.date_naive())
        .or_else(|| chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").ok())
        .or_else(|| {
            raw.get(..10)
                .and_then(|date| chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
        })
}

fn suggested_contract_coverage(
    selected_maturity: Option<&Value>,
    maturity_chain: Option<&Value>,
) -> Option<ReplayCacheSuggestedCoverage> {
    let selected_maturity = selected_maturity?;
    let selected_expiration = metadata_expiration_date(selected_maturity)?;
    let selected_id = json_i64(selected_maturity, "id");
    let chain = maturity_chain?.as_array()?;
    let previous_expiration = chain
        .iter()
        .filter(|maturity| {
            selected_id.is_none_or(|selected_id| json_i64(maturity, "id") != Some(selected_id))
        })
        .filter_map(metadata_expiration_date)
        .filter(|expiration| *expiration < selected_expiration)
        .max()?;
    Some(ReplayCacheSuggestedCoverage {
        start_date: previous_expiration,
        end_date: selected_expiration,
        basis: "previous maturity expiration through selected maturity expiration; actual liquidity and provider retention may differ"
            .to_string(),
        estimated: true,
    })
}

fn market_specs_from_metadata(metadata: &ReplayCacheContractMetadata) -> Option<MarketSpecs> {
    let product = metadata.product.as_ref().map(|snapshot| &snapshot.payload);
    let contract = &metadata.contract.payload;
    let tick_size = product
        .and_then(|product| {
            json_number(product, "tickSize").or_else(|| json_number(product, "minTick"))
        })
        .or_else(|| json_number(contract, "providerTickSize"))
        .or_else(|| json_number(contract, "tickSize"));
    let value_per_point = product.and_then(|product| json_number(product, "valuePerPoint"));
    if product.is_none() && tick_size.is_none() && value_per_point.is_none() {
        return None;
    }
    Some(MarketSpecs {
        session_profile: product.map(infer_session_profile),
        value_per_point,
        tick_size,
    })
}

pub fn server_bar_chart_request_body(
    contract: &ContractSuggestion,
    request: &TradovateServerBarDownloadRequest,
) -> Value {
    json!({
        "symbol": contract.id,
        "chartDescription": request.bar_type.chart_description(),
        "timeRange": {
            "closestTimestamp": request.end.to_rfc3339(),
            "asFarAsTimestamp": request.start.to_rfc3339()
        }
    })
}

pub fn raw_tick_chart_request_body(
    contract: &ContractSuggestion,
    request: &TradovateRawTickDownloadRequest,
) -> Value {
    json!({
        "symbol": contract.id,
        "chartDescription": {
            "underlyingType": "Tick",
            "elementSize": 1,
            "elementSizeUnit": "UnderlyingUnits",
            "withHistogram": false
        },
        "timeRange": {
            "closestTimestamp": request.end.to_rfc3339(),
            "asFarAsTimestamp": request.start.to_rfc3339()
        }
    })
}

pub fn extract_historical_server_bars_from_chart_message(
    item: &Value,
    historical_id: Option<i64>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> (Vec<Bar>, bool) {
    let mut bars = Vec::new();
    let mut end_of_history = false;
    let Some(charts) = item
        .get("d")
        .and_then(|d| d.get("charts"))
        .and_then(Value::as_array)
    else {
        return (bars, end_of_history);
    };
    let start_ns = start.timestamp_nanos_opt().unwrap_or(i64::MIN);
    let end_ns = end.timestamp_nanos_opt().unwrap_or(i64::MAX);
    for chart in charts {
        let chart_id = chart.get("id").and_then(Value::as_i64);
        let is_historical = historical_id.is_none_or(|id| chart_id == Some(id));
        if !is_historical {
            continue;
        }
        if chart.get("eoh").and_then(Value::as_bool).unwrap_or(false) {
            end_of_history = true;
        }
        let Some(raw_bars) = chart.get("bars").and_then(Value::as_array) else {
            continue;
        };
        for raw_bar in raw_bars {
            let Some(bar) = parse_bar(raw_bar) else {
                continue;
            };
            if bar.ts_ns >= start_ns && bar.ts_ns < end_ns {
                bars.push(bar);
            }
        }
    }
    (bars, end_of_history)
}

pub fn extract_historical_raw_ticks_from_chart_message(
    item: &Value,
    historical_id: Option<i64>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fallback_tick_size: Option<f64>,
) -> (Vec<ReplayCacheRawTickRow>, bool) {
    let mut ticks = Vec::new();
    let mut end_of_history = false;
    let Some(charts) = item
        .get("d")
        .and_then(|d| d.get("charts"))
        .and_then(Value::as_array)
    else {
        return (ticks, end_of_history);
    };
    let start_ns = start.timestamp_nanos_opt().unwrap_or(i64::MIN);
    let end_ns = end.timestamp_nanos_opt().unwrap_or(i64::MAX);
    for chart in charts {
        let chart_id = chart.get("id").and_then(Value::as_i64);
        let is_historical = historical_id.is_none_or(|id| chart_id == Some(id));
        if !is_historical {
            continue;
        }
        if chart.get("eoh").and_then(Value::as_bool).unwrap_or(false) {
            end_of_history = true;
        }
        let Some(packet_ticks) = parse_raw_tick_packet(chart, start_ns, end_ns, fallback_tick_size)
        else {
            continue;
        };
        ticks.extend(packet_ticks);
    }
    (ticks, end_of_history)
}

fn parse_raw_tick_packet(
    chart: &Value,
    start_ns: i64,
    end_ns: i64,
    fallback_tick_size: Option<f64>,
) -> Option<Vec<ReplayCacheRawTickRow>> {
    let raw_ticks = chart.get("tks").and_then(Value::as_array)?;
    let base_price_ticks = json_i64(chart, "bp")?;
    let base_ts_ms = json_i64(chart, "bt")?;
    let tick_size = json_number(chart, "ts")
        .filter(|tick_size| tick_size.is_finite() && *tick_size > 0.0)
        .or(fallback_tick_size)?;
    let chart_id = chart.get("id").and_then(Value::as_i64);
    let trade_date = json_i64(chart, "td").and_then(|value| i32::try_from(value).ok());
    let packet_source = chart
        .get("s")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let mut rows = Vec::with_capacity(raw_ticks.len());

    for raw_tick in raw_ticks {
        let Some(relative_ts_ms) = json_i64(raw_tick, "t") else {
            continue;
        };
        let Some(price_offset_ticks) = json_i64(raw_tick, "p") else {
            continue;
        };
        let Some(size) = json_number(raw_tick, "s") else {
            continue;
        };
        let Some(ts_ms) = base_ts_ms.checked_add(relative_ts_ms) else {
            continue;
        };
        let Some(ts_ns) = ts_ms.checked_mul(1_000_000) else {
            continue;
        };
        if ts_ns < start_ns || ts_ns >= end_ns {
            continue;
        }
        let price_ticks = base_price_ticks.saturating_add(price_offset_ticks);
        let price = price_ticks as f64 * tick_size;
        let bid_size = json_number(raw_tick, "bs");
        let ask_size = json_number(raw_tick, "as");
        let bid_price = bid_size
            .and_then(|_| json_i64(raw_tick, "b"))
            .map(|offset| base_price_ticks.saturating_add(offset) as f64 * tick_size);
        let ask_price = ask_size
            .and_then(|_| json_i64(raw_tick, "a"))
            .map(|offset| base_price_ticks.saturating_add(offset) as f64 * tick_size);

        rows.push(ReplayCacheRawTickRow {
            timestamp: DateTime::<Utc>::from_timestamp_nanos(ts_ns),
            ts_ns,
            tick_id: json_i64(raw_tick, "id"),
            price,
            size,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            chart_id,
            trade_date,
            packet_source: packet_source.clone(),
            packet_base_ts_ms: Some(base_ts_ms),
            packet_base_price_ticks: Some(base_price_ticks),
        });
    }

    Some(rows)
}

async fn resolve_download_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract_symbol: &str,
    limit: usize,
) -> Result<ContractSuggestion> {
    let contracts = search_contracts(client, env, token, contract_symbol, limit).await?;
    contracts
        .iter()
        .find(|contract| contract.name.eq_ignore_ascii_case(contract_symbol))
        .cloned()
        .with_context(|| {
            let available = contracts
                .iter()
                .map(|contract| contract.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if available.is_empty() {
                format!("contract search returned no results for {contract_symbol}")
            } else {
                format!(
                    "contract search did not return exact symbol {contract_symbol}; suggestions: {available}"
                )
            }
        })
}

async fn resolve_requested_download_contract(
    client: &Client,
    env: &TradingEnvironment,
    token: &str,
    contract_symbol: &str,
    exact_contract: Option<&ContractSuggestion>,
    limit: usize,
) -> Result<ContractSuggestion> {
    if let Some(contract) = exact_contract {
        if !contract.name.eq_ignore_ascii_case(contract_symbol) {
            bail!(
                "selected contract name {} does not match requested symbol {contract_symbol}",
                contract.name
            );
        }
        return Ok(contract.clone());
    }
    resolve_download_contract(client, env, token, contract_symbol, limit).await
}

fn validate_contract_identity(contract: &ContractSuggestion, payload: &Value) -> Result<()> {
    if let Some(id) = payload.get("id").and_then(Value::as_i64)
        && id != contract.id
    {
        bail!(
            "provider contract id {id} does not match selected contract id {}",
            contract.id
        );
    }
    if let Some(name) = payload.get("name").and_then(Value::as_str)
        && !name.eq_ignore_ascii_case(&contract.name)
    {
        bail!(
            "provider contract name {name} does not match selected contract name {}",
            contract.name
        );
    }
    Ok(())
}

struct HistoricalRows<T> {
    rows: Vec<T>,
    telemetry: HistoricalDownloadTelemetry,
}

fn historical_failure(
    kind: HistoricalDownloadFailureKind,
    message: impl Into<String>,
    mut telemetry: HistoricalDownloadTelemetry,
    started: std::time::Instant,
) -> anyhow::Error {
    telemetry.elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    anyhow::Error::new(HistoricalDownloadFailure {
        kind,
        message: message.into(),
        telemetry,
        retry_after_ms: None,
    })
}

fn websocket_status_failure_kind(status: Option<i64>) -> HistoricalDownloadFailureKind {
    match status {
        Some(401) => HistoricalDownloadFailureKind::Authentication,
        Some(403) => HistoricalDownloadFailureKind::Authorization,
        Some(408) => HistoricalDownloadFailureKind::Timeout,
        Some(413 | 414 | 422) => HistoricalDownloadFailureKind::SizeRejected,
        Some(429) => HistoricalDownloadFailureKind::RateLimited,
        _ => HistoricalDownloadFailureKind::Provider,
    }
}

fn sanitized_websocket_failure(endpoint: &str, status: Option<i64>) -> String {
    match status {
        Some(status) => format!(
            "{endpoint} failed with websocket status {status}; provider response body omitted"
        ),
        None => format!(
            "{endpoint} response did not include a status code; provider response body omitted"
        ),
    }
}

fn record_wire_message(telemetry: &mut HistoricalDownloadTelemetry, message: &Message) {
    telemetry.packet_count = telemetry.packet_count.saturating_add(1);
    let bytes = match message {
        Message::Text(text) => text.len(),
        Message::Binary(bytes) => bytes.len(),
        Message::Ping(bytes) | Message::Pong(bytes) => bytes.len(),
        Message::Close(frame) => frame
            .as_ref()
            .map(|frame| frame.reason.len().saturating_add(2))
            .unwrap_or_default(),
        Message::Frame(frame) => frame.payload().len(),
    };
    telemetry.wire_bytes = telemetry.wire_bytes.saturating_add(bytes as u64);
}

fn historical_chart_row_count(item: &Value, historical_id: Option<i64>, field: &str) -> u64 {
    item.get("d")
        .and_then(|value| value.get("charts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|chart| {
            historical_id.is_none_or(|id| chart.get("id").and_then(Value::as_i64) == Some(id))
        })
        .filter_map(|chart| chart.get(field).and_then(Value::as_array))
        .map(|rows| rows.len() as u64)
        .sum()
}

fn finish_telemetry<T>(
    telemetry: &mut HistoricalDownloadTelemetry,
    rows: &[T],
    timestamp: impl Fn(&T) -> DateTime<Utc>,
    started: std::time::Instant,
) {
    telemetry.elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    telemetry.normalized_rows = rows.len() as u64;
    telemetry.dropped_rows = telemetry
        .provider_rows
        .saturating_sub(telemetry.normalized_rows);
    telemetry.first_timestamp = rows.iter().map(&timestamp).min();
    telemetry.last_timestamp = rows.iter().map(timestamp).max();
}

async fn fetch_server_bars_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<HistoricalRows<Bar>> {
    let request_window = DownloadWindow::new(start, end)?;
    let mut telemetry = HistoricalDownloadTelemetry::new(request_window);
    let started = std::time::Instant::now();
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .map_err(|_| {
            historical_failure(
                HistoricalDownloadFailureKind::Transport,
                "connect market-data websocket failed; transport details omitted",
                telemetry.clone(),
                started,
            )
        })?;
    let (mut write, mut read) = ws_stream.split();

    let authorize_id = 1_u64;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, md_access_token
        )))
        .await?;

    let chart_req_id = 2_u64;
    let mut authorized = false;
    let mut chart_requested = false;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut bars = Vec::new();
    let timeout = time::sleep(Duration::from_secs(35));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                telemetry.timed_out = true;
                telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                return Err(historical_failure(
                    HistoricalDownloadFailureKind::Timeout,
                    "timed out waiting for explicit Tradovate server-bar end-of-history",
                    telemetry,
                    started,
                ));
            }
            next = read.next() => {
                let message = match next {
                    Some(Ok(message)) => message,
                    Some(Err(_)) => {
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::Transport,
                            "market websocket read failed; transport details omitted",
                            telemetry,
                            started,
                        ));
                    }
                    None => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket ended before explicit Tradovate server-bar end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                };
                record_wire_message(&mut telemetry, &message);
                let raw = match message {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Message::Close(_) => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket closed before explicit Tradovate server-bar end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                    _ => continue,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);
                    if response_id == Some(authorize_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("market websocket authorize", status),
                                telemetry,
                                started,
                            ));
                        }
                        authorized = true;
                    }

                    if authorized && !chart_requested {
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                chart_req_id,
                                None,
                                Some(&request_body),
                            )))
                            .await?;
                        chart_requested = true;
                    }

                    if response_id == Some(chart_req_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("md/getChart", status),
                                telemetry,
                                started,
                            ));
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                        telemetry.historical_id = historical_id;
                        telemetry.realtime_id = realtime_id;
                    }

                    telemetry.provider_rows = telemetry.provider_rows.saturating_add(
                        historical_chart_row_count(&item, historical_id, "bars"),
                    );
                    let (mut packet_bars, end_of_history) =
                        extract_historical_server_bars_from_chart_message(
                            &item,
                            historical_id,
                            start,
                            end,
                        );
                    bars.append(&mut packet_bars);
                    if end_of_history {
                        telemetry.end_of_history = true;
                        telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        finish_telemetry(&mut telemetry, &bars, |bar| DateTime::<Utc>::from_timestamp_nanos(bar.ts_ns), started);
                        return Ok(HistoricalRows { rows: bars, telemetry });
                    }
                }
            }
        }
    }
}

async fn fetch_raw_ticks_over_market_ws(
    cfg: &AppConfig,
    md_access_token: &str,
    request_body: Value,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    fallback_tick_size: Option<f64>,
) -> Result<HistoricalRows<ReplayCacheRawTickRow>> {
    let request_window = DownloadWindow::new(start, end)?;
    let mut telemetry = HistoricalDownloadTelemetry::new(request_window);
    let started = std::time::Instant::now();
    let ws_config = WebSocketConfig {
        write_buffer_size: 0,
        max_write_buffer_size: usize::MAX,
        ..Default::default()
    };
    let (ws_stream, _) = connect_low_latency_ws(cfg.env.market_ws_url(), ws_config)
        .await
        .map_err(|_| {
            historical_failure(
                HistoricalDownloadFailureKind::Transport,
                "connect market-data websocket failed; transport details omitted",
                telemetry.clone(),
                started,
            )
        })?;
    let (mut write, mut read) = ws_stream.split();

    let authorize_id = 1_u64;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            authorize_id, md_access_token
        )))
        .await?;

    let chart_req_id = 2_u64;
    let mut authorized = false;
    let mut chart_requested = false;
    let mut historical_id = None;
    let mut realtime_id = None;
    let mut ticks = Vec::new();
    let timeout = time::sleep(Duration::from_secs(60));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            _ = &mut timeout => {
                telemetry.timed_out = true;
                telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                return Err(historical_failure(
                    HistoricalDownloadFailureKind::Timeout,
                    "timed out waiting for explicit Tradovate raw-tick end-of-history",
                    telemetry,
                    started,
                ));
            }
            next = read.next() => {
                let message = match next {
                    Some(Ok(message)) => message,
                    Some(Err(_)) => {
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::Transport,
                            "market websocket read failed; transport details omitted",
                            telemetry,
                            started,
                        ));
                    }
                    None => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket ended before explicit Tradovate raw-tick end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                };
                record_wire_message(&mut telemetry, &message);
                let raw = match message {
                    Message::Text(text) => text,
                    Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Message::Close(_) => {
                        telemetry.socket_closed = true;
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        return Err(historical_failure(
                            HistoricalDownloadFailureKind::ClosedWithoutCompletion,
                            "market websocket closed before explicit Tradovate raw-tick end-of-history",
                            telemetry,
                            started,
                        ));
                    }
                    _ => continue,
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(Value::as_u64);
                    if response_id == Some(authorize_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("market websocket authorize", status),
                                telemetry,
                                started,
                            ));
                        }
                        authorized = true;
                    }

                    if authorized && !chart_requested {
                        write
                            .send(Message::Text(create_message(
                                "md/getChart",
                                chart_req_id,
                                None,
                                Some(&request_body),
                            )))
                            .await?;
                        chart_requested = true;
                    }

                    if response_id == Some(chart_req_id) {
                        if !status.is_some_and(|code| (200..300).contains(&code)) {
                            return Err(historical_failure(
                                websocket_status_failure_kind(status),
                                sanitized_websocket_failure("md/getChart", status),
                                telemetry,
                                started,
                            ));
                        }
                        if let Some(d) = item.get("d") {
                            historical_id = d
                                .get("historicalId")
                                .and_then(Value::as_i64)
                                .or(historical_id);
                            realtime_id =
                                d.get("realtimeId").and_then(Value::as_i64).or(realtime_id);
                        }
                        telemetry.historical_id = historical_id;
                        telemetry.realtime_id = realtime_id;
                    }

                    telemetry.provider_rows = telemetry.provider_rows.saturating_add(
                        historical_chart_row_count(&item, historical_id, "tks"),
                    );
                    let (mut packet_ticks, end_of_history) =
                        extract_historical_raw_ticks_from_chart_message(
                            &item,
                            historical_id,
                            start,
                            end,
                            fallback_tick_size,
                        );
                    ticks.append(&mut packet_ticks);
                    if end_of_history {
                        telemetry.end_of_history = true;
                        telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
                        telemetry.cancellation_sent = cancel_chart_subscriptions(&mut write, historical_id, realtime_id).await;
                        finish_telemetry(&mut telemetry, &ticks, |tick| tick.timestamp, started);
                        return Ok(HistoricalRows { rows: ticks, telemetry });
                    }
                }
            }
        }
    }
}

async fn cancel_chart_subscriptions<S>(
    write: &mut S,
    historical_id: Option<i64>,
    realtime_id: Option<i64>,
) -> bool
where
    S: futures_util::Sink<Message> + Unpin,
{
    let mut cancel_id = 10_000_u64;
    let mut sent = false;
    for subscription_id in [historical_id, realtime_id].into_iter().flatten() {
        cancel_id += 1;
        let body = json!({ "subscriptionId": subscription_id });
        if write
            .send(Message::Text(create_message(
                "md/cancelChart",
                cancel_id,
                None,
                Some(&body),
            )))
            .await
            .is_ok()
        {
            sent = true;
        }
    }
    sent
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dt(raw: &str) -> DateTime<Utc> {
        raw.parse().expect("valid timestamp")
    }

    #[test]
    fn server_bar_request_uses_chart_timerange_without_account_streams() {
        let contract = ContractSuggestion {
            id: 123,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({ "id": 123, "name": "MESU6" }),
        };
        let request = TradovateServerBarDownloadRequest {
            contract: "MESU6".to_string(),
            exact_contract: None,
            start: dt("2026-07-23T00:00:00Z"),
            end: dt("2026-07-24T00:00:00Z"),
            bar_type: BarType::minute(1),
        };

        let body = server_bar_chart_request_body(&contract, &request);

        assert_eq!(body["symbol"], 123);
        assert_eq!(
            body["chartDescription"],
            BarType::minute(1).chart_description()
        );
        assert_eq!(
            body["timeRange"]["asFarAsTimestamp"],
            "2026-07-23T00:00:00+00:00"
        );
        assert_eq!(
            body["timeRange"]["closestTimestamp"],
            "2026-07-24T00:00:00+00:00"
        );
        assert!(body.get("replayCache").is_none());
        assert!(body.get("accounts").is_none());
        assert!(body.get("entityTypes").is_none());
    }

    #[test]
    fn raw_tick_request_uses_one_tick_chart_without_account_streams() {
        let contract = ContractSuggestion {
            id: 123,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({ "id": 123, "name": "MESU6" }),
        };
        let request = TradovateRawTickDownloadRequest {
            contract: "MESU6".to_string(),
            exact_contract: None,
            start: dt("2026-07-23T00:00:00Z"),
            end: dt("2026-07-24T00:00:00Z"),
        };

        let body = raw_tick_chart_request_body(&contract, &request);

        assert_eq!(body["symbol"], 123);
        assert_eq!(body["chartDescription"]["underlyingType"], "Tick");
        assert_eq!(body["chartDescription"]["elementSize"], 1);
        assert_eq!(
            body["chartDescription"]["elementSizeUnit"],
            "UnderlyingUnits"
        );
        assert_eq!(
            body["timeRange"]["asFarAsTimestamp"],
            "2026-07-23T00:00:00+00:00"
        );
        assert_eq!(
            body["timeRange"]["closestTimestamp"],
            "2026-07-24T00:00:00+00:00"
        );
        assert!(body.get("accounts").is_none());
        assert!(body.get("entityTypes").is_none());
    }

    #[tokio::test]
    async fn exact_download_contract_is_used_without_symbol_search() {
        let client = Client::new();
        let exact = ContractSuggestion {
            id: 4_399_631,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({"id": 4_399_631, "name": "MESU6"}),
        };

        let resolved = resolve_requested_download_contract(
            &client,
            &TradingEnvironment::Sim,
            "unused-token",
            "MESU6",
            Some(&exact),
            12,
        )
        .await
        .expect("exact contract should not require a search request");

        assert_eq!(resolved.id, exact.id);
        assert_eq!(resolved.name, exact.name);
    }

    #[tokio::test]
    async fn exact_download_contract_rejects_name_mismatch_before_network_io() {
        let client = Client::new();
        let exact = ContractSuggestion {
            id: 4_399_631,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({"id": 4_399_631, "name": "MESU6"}),
        };

        let err = resolve_requested_download_contract(
            &client,
            &TradingEnvironment::Sim,
            "unused-token",
            "ESU6",
            Some(&exact),
            12,
        )
        .await
        .expect_err("mismatched exact contract must fail closed");

        assert!(err.to_string().contains("does not match requested symbol"));
    }

    #[test]
    fn provider_contract_snapshot_must_match_selected_exact_identity() {
        let contract = ContractSuggestion {
            id: 4_399_631,
            name: "MESU6".to_string(),
            description: "Micro E-mini".to_string(),
            raw: json!({}),
        };

        let id_err = validate_contract_identity(&contract, &json!({"id": 7, "name": "MESU6"}))
            .expect_err("provider id mismatch");
        assert!(
            id_err
                .to_string()
                .contains("does not match selected contract id")
        );

        let name_err =
            validate_contract_identity(&contract, &json!({"id": 4_399_631, "name": "ESU6"}))
                .expect_err("provider name mismatch");
        assert!(
            name_err
                .to_string()
                .contains("does not match selected contract name")
        );
    }

    #[test]
    fn extracts_only_historical_bars_inside_requested_range() {
        let item = json!({
            "d": {
                "charts": [
                    {
                        "id": 7,
                        "bars": [
                            {"timestamp":"2026-07-22T23:59:00Z","open":1,"high":1,"low":1,"close":1,"volume":1},
                            {"timestamp":"2026-07-23T00:00:00Z","open":2,"high":3,"low":1,"close":2.5,"upVolume":5,"downVolume":4}
                        ]
                    },
                    {
                        "id": 8,
                        "bars": [
                            {"timestamp":"2026-07-23T00:00:00Z","open":9,"high":9,"low":9,"close":9,"volume":1}
                        ]
                    },
                    {"id": 7, "eoh": true}
                ]
            }
        });

        let (bars, eoh) = extract_historical_server_bars_from_chart_message(
            &item,
            Some(7),
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-24T00:00:00Z"),
        );

        assert!(eoh);
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].open, 2.0);
        assert_eq!(bars[0].volume, Some(9.0));
    }

    #[test]
    fn extracts_raw_ticks_from_historical_tick_packets() {
        let item = json!({
            "d": {
                "charts": [
                    {
                        "id": 7,
                        "s": "db",
                        "td": 20260723,
                        "bp": 29700,
                        "bt": 1784764800000i64,
                        "ts": 0.25,
                        "tks": [
                            {"t": 0, "p": 0, "s": 1, "b": -1, "a": 0, "bs": 12, "as": 14, "id": 1001},
                            {"t": 1, "p": 1, "s": 2, "id": 1002},
                            {"t": -1, "p": 9, "s": 1, "id": 999}
                        ]
                    },
                    {
                        "id": 8,
                        "bp": 999,
                        "bt": 1784764800000i64,
                        "ts": 0.25,
                        "tks": [
                            {"t": 0, "p": 0, "s": 1, "id": 2001}
                        ]
                    },
                    {"id": 7, "eoh": true}
                ]
            }
        });

        let (ticks, eoh) = extract_historical_raw_ticks_from_chart_message(
            &item,
            Some(7),
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-23T00:00:01Z"),
            None,
        );

        assert!(eoh);
        assert_eq!(ticks.len(), 2);
        assert_eq!(ticks[0].tick_id, Some(1001));
        assert_eq!(ticks[0].price, 7425.0);
        assert_eq!(ticks[0].bid_price, Some(7424.75));
        assert_eq!(ticks[0].ask_price, Some(7425.0));
        assert_eq!(ticks[0].size, 1.0);
        assert_eq!(ticks[0].chart_id, Some(7));
        assert_eq!(ticks[0].trade_date, Some(20260723));
        assert_eq!(ticks[0].packet_source.as_deref(), Some("db"));
        assert_eq!(ticks[1].tick_id, Some(1002));
        assert_eq!(ticks[1].price, 7425.25);
    }

    #[test]
    fn derives_broad_coverage_from_adjacent_maturity_expirations() {
        let selected = json!({
            "id": 62531,
            "expirationDate": "2026-09-18T13:30Z"
        });
        let chain = json!([
            {"id": 61200, "expirationDate": "2026-03-20T13:30Z"},
            {"id": 61800, "expirationDate": "2026-06-19T13:30Z"},
            {"id": 62531, "expirationDate": "2026-09-18T13:30Z"},
            {"id": 63200, "expirationDate": "2026-12-18T14:30Z"}
        ]);

        let coverage = suggested_contract_coverage(Some(&selected), Some(&chain))
            .expect("adjacent expiration coverage");

        assert_eq!(coverage.start_date.to_string(), "2026-06-19");
        assert_eq!(coverage.end_date.to_string(), "2026-09-18");
        assert!(coverage.estimated);
        assert!(coverage.basis.contains("actual liquidity"));
    }

    #[test]
    fn broad_coverage_fails_closed_without_an_earlier_maturity() {
        let selected = json!({
            "id": 62531,
            "expirationDate": "2026-09-18T13:30:00Z"
        });
        let chain = json!([
            {"id": 62531, "expirationDate": "2026-09-18T13:30:00Z"},
            {"id": 63200, "expirationDate": "2026-12-18T14:30:00Z"}
        ]);

        assert!(suggested_contract_coverage(Some(&selected), Some(&chain)).is_none());
    }

    #[test]
    fn metadata_source_timestamp_finds_provider_timestamp() {
        let payload = json!([{
            "id": 1,
            "initialMargin": 2761.96,
            "timestamp": "2026-07-24T03:50:43.545Z"
        }]);

        assert_eq!(
            metadata_source_timestamp(&payload),
            Some(dt("2026-07-24T03:50:43.545Z"))
        );
    }

    #[test]
    fn market_specs_prefer_product_fields_and_keep_contract_tick_fallback() {
        let metadata = ReplayCacheContractMetadata {
            context: ReplayCacheMetadataContext {
                provider: BrokerKind::Tradovate,
                env: TradingEnvironment::Sim,
                user_id: Some(9),
                user_name: Some("tester".to_string()),
                accounts: Vec::new(),
                accounts_endpoint: Some("account/list".to_string()),
                accounts_fetched_at: Some(dt("2026-07-24T00:00:00Z")),
            },
            contract: ReplayCacheMetadataSnapshot {
                endpoint: "contract/item".to_string(),
                fetched_at: dt("2026-07-24T00:00:00Z"),
                source_timestamp: None,
                payload: json!({"providerTickSize": 0.5}),
            },
            maturity: None,
            maturity_chain: None,
            product: Some(ReplayCacheMetadataSnapshot {
                endpoint: "product/item".to_string(),
                fetched_at: dt("2026-07-24T00:00:01Z"),
                source_timestamp: None,
                payload: json!({
                    "tickSize": 0.25,
                    "valuePerPoint": 5.0,
                    "productType": "Futures"
                }),
            }),
            product_sessions: None,
            product_margins: None,
            contract_margins: None,
            fee_params: None,
            suggested_coverage: None,
        };

        let specs = market_specs_from_metadata(&metadata).expect("metadata market specs");
        assert_eq!(specs.tick_size, Some(0.25));
        assert_eq!(specs.value_per_point, Some(5.0));
        assert!(specs.session_profile.is_some());
    }

    #[test]
    fn optional_metadata_warning_names_the_failed_endpoint() {
        let warning =
            optional_metadata_warning("productMargin/deps", &anyhow::anyhow!("synthetic timeout"));

        assert!(warning.contains("productMargin/deps"));
        assert!(warning.contains("synthetic timeout"));
    }

    #[test]
    fn authenticated_metadata_http_warnings_omit_provider_response_bodies() {
        let provider_body = r#"{"accessToken":"secret-token","accountName":"DEMO4769136"}"#;
        let error = sanitized_metadata_http_error(
            "account/list",
            reqwest::StatusCode::UNAUTHORIZED,
            provider_body,
        );
        let warning = optional_metadata_warning("account/list", &error);

        assert!(warning.contains("account/list"));
        assert!(warning.contains("HTTP 401"));
        assert!(warning.contains("authentication rejected"));
        assert!(warning.contains("provider response body omitted"));
        assert!(!warning.contains("accessToken"));
        assert!(!warning.contains("secret-token"));
        assert!(!warning.contains("DEMO4769136"));
    }

    #[test]
    fn fee_metadata_request_uses_required_product_ids_array() {
        assert_eq!(
            fee_metadata_request_body(1_878_809),
            json!({"productIds": [1_878_809]})
        );
    }
}

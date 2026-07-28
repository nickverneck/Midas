use super::metadata::{
    fetch_download_contract_metadata, market_specs_from_metadata,
    resolve_requested_download_contract, validate_contract_identity,
};
use super::websocket::{fetch_raw_ticks_over_market_ws, fetch_server_bars_over_market_ws};
use super::*;

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
            warnings.push(
                "Tradovate returned zero historical server bars with explicit end-of-history."
                    .to_string(),
            );
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
        let root_window = DownloadWindow::new(start, end)?;
        let root_request = TradovateRawTickDownloadRequest {
            contract: self.contract.name.clone(),
            exact_contract: Some(self.contract.clone()),
            start,
            end,
        };
        let request_body = raw_tick_chart_request_body(&self.contract, &root_request);
        let tick_size = (self.tick_specs.tick_size.is_finite() && self.tick_specs.tick_size > 0.0)
            .then_some(self.tick_specs.tick_size);
        let mut cursor_end = end;
        let mut ticks = Vec::new();
        let mut telemetry = HistoricalDownloadTelemetry::new(root_window);

        loop {
            pace_raw_tick_request(&self.raw_tick_last_request).await;
            let page_request = TradovateRawTickDownloadRequest {
                contract: self.contract.name.clone(),
                exact_contract: Some(self.contract.clone()),
                start,
                end: cursor_end,
            };
            let page_body = raw_tick_chart_request_body(&self.contract, &page_request);
            let page = fetch_raw_ticks_over_market_ws(
                &self.cfg,
                &self.md_access_token,
                page_body,
                start,
                end,
                tick_size,
            )
            .await?;
            let next_cursor = raw_tick_next_page_cursor(&page.telemetry, start, cursor_end)?;
            merge_raw_tick_page_telemetry(&mut telemetry, &page.telemetry);
            ticks.extend(page.rows);
            let Some(next_cursor) = next_cursor else {
                break;
            };
            cursor_end = next_cursor;
        }

        telemetry.end_of_history = true;
        telemetry.completion_evidence = Some(DownloadCompletionEvidence::EndOfHistory);
        telemetry.cap = DownloadCapClassification::No;
        telemetry.cap_reason = format!(
            "Tradovate raw-tick history completed after {} backward page(s) at the effective {}-element page limit.",
            telemetry.page_count,
            super::protocol::RAW_TICK_REQUEST_MAX_ELEMENTS,
        );
        telemetry.normalized_rows = ticks.len() as u64;
        telemetry.first_timestamp = ticks.iter().map(|tick| tick.timestamp).min();
        telemetry.last_timestamp = ticks.iter().map(|tick| tick.timestamp).max();
        let mut warnings = self.warnings.clone();
        if ticks.is_empty() {
            warnings.push(
                "Tradovate returned zero historical raw ticks with explicit end-of-history."
                    .to_string(),
            );
        }
        Ok(TradovateRawTickDownload {
            contract: self.contract.clone(),
            ticks,
            request_body,
            tick_specs: self.tick_specs.clone(),
            session_template: self.session_template.clone(),
            contract_metadata: self.contract_metadata.clone(),
            warnings,
            telemetry,
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
        raw_tick_last_request: tokio::sync::Mutex::new(None),
    })
}

const RAW_TICK_REQUEST_MIN_INTERVAL: Duration = Duration::from_millis(1_250);

async fn pace_raw_tick_request(gate: &tokio::sync::Mutex<Option<tokio::time::Instant>>) {
    let mut last_request = gate.lock().await;
    if let Some(last_request) = *last_request {
        let elapsed = last_request.elapsed();
        if elapsed < RAW_TICK_REQUEST_MIN_INTERVAL {
            tokio::time::sleep(RAW_TICK_REQUEST_MIN_INTERVAL - elapsed).await;
        }
    }
    *last_request = Some(tokio::time::Instant::now());
}

pub(super) fn raw_tick_next_page_cursor(
    telemetry: &HistoricalDownloadTelemetry,
    requested_start: DateTime<Utc>,
    cursor_end: DateTime<Utc>,
) -> Result<Option<DateTime<Utc>>> {
    if telemetry.cap != DownloadCapClassification::Yes {
        return Ok(None);
    }
    let oldest = telemetry
        .provider_first_timestamp
        .context("capped Tradovate raw-tick page did not expose its oldest timestamp")?;
    if oldest <= requested_start {
        return Ok(None);
    }
    if oldest >= cursor_end {
        bail!(
            "Tradovate raw-tick pagination made no backward progress from {cursor_end}; coverage was not committed"
        );
    }
    Ok(Some(oldest))
}

fn merge_raw_tick_page_telemetry(
    aggregate: &mut HistoricalDownloadTelemetry,
    page: &HistoricalDownloadTelemetry,
) {
    aggregate.elapsed_ms = aggregate.elapsed_ms.saturating_add(page.elapsed_ms);
    aggregate.wire_bytes = aggregate.wire_bytes.saturating_add(page.wire_bytes);
    aggregate.packet_count = aggregate.packet_count.saturating_add(page.packet_count);
    aggregate.provider_rows = aggregate.provider_rows.saturating_add(page.provider_rows);
    aggregate.normalized_rows = aggregate
        .normalized_rows
        .saturating_add(page.normalized_rows);
    aggregate.duplicate_rows = aggregate.duplicate_rows.saturating_add(page.duplicate_rows);
    aggregate.dropped_rows = aggregate.dropped_rows.saturating_add(page.dropped_rows);
    aggregate.page_count = aggregate.page_count.saturating_add(page.page_count.max(1));
    aggregate.provider_first_timestamp = match (
        aggregate.provider_first_timestamp,
        page.provider_first_timestamp,
    ) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    };
    aggregate.provider_last_timestamp = match (
        aggregate.provider_last_timestamp,
        page.provider_last_timestamp,
    ) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    };
    aggregate.historical_id = page.historical_id.or(aggregate.historical_id);
    aggregate.realtime_id = page.realtime_id.or(aggregate.realtime_id);
    aggregate.socket_closed |= page.socket_closed;
    aggregate.timed_out |= page.timed_out;
    aggregate.cancellation_sent |= page.cancellation_sent;
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

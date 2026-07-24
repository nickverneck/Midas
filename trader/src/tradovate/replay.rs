use super::*;
#[cfg(feature = "replay")]
use crate::replay_cache::{ReplayCacheLibrary, ReplayCacheLoadedServerBars};
#[cfg(feature = "replay")]
use anyhow::{Context, Result, bail};
#[cfg(not(feature = "replay"))]
use anyhow::{Result, bail};
#[cfg(feature = "replay")]
use chrono::{NaiveDate, NaiveTime, TimeZone, Timelike};
#[cfg(feature = "replay")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "replay")]
use std::fs::File;
#[cfg(feature = "replay")]
use std::hash::{Hash, Hasher};
#[cfg(feature = "replay")]
use std::io::{BufRead, BufReader};
#[cfg(feature = "replay")]
use std::path::{Path, PathBuf};
#[cfg(feature = "replay")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct ReplayState {
    #[cfg(feature = "replay")]
    contract: ContractSuggestion,
    #[cfg(feature = "replay")]
    account: AccountInfo,
    #[cfg(feature = "replay")]
    market_specs: MarketSpecs,
    #[cfg(feature = "replay")]
    data: ReplayDataSource,
}

pub(crate) async fn load_replay_state(
    cfg: &AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
) -> Result<ReplayState> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (cfg, bar_type, candle_mode);
        bail!("replay mode is not enabled in this build; rebuild with `--features replay`");
    }

    #[cfg(feature = "replay")]
    {
        let cfg = cfg.clone();
        tokio::task::spawn_blocking(move || load_replay_state_blocking(&cfg, bar_type, candle_mode))
            .await
            .context("join replay parser task")?
    }
}

pub(crate) fn replay_accounts(state: &ReplayState) -> Vec<AccountInfo> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = state;
        Vec::new()
    }

    #[cfg(feature = "replay")]
    {
        vec![state.account.clone()]
    }
}

pub(crate) fn replay_contract(state: &ReplayState) -> ContractSuggestion {
    #[cfg(not(feature = "replay"))]
    {
        let _ = state;
        ContractSuggestion {
            id: 0,
            name: "Replay".to_string(),
            description: "Replay".to_string(),
            raw: json!({}),
        }
    }

    #[cfg(feature = "replay")]
    {
        state.contract.clone()
    }
}

pub(crate) fn search_replay_contracts(
    state: &ReplayState,
    query: &str,
    limit: usize,
) -> Vec<ContractSuggestion> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (state, query, limit);
        Vec::new()
    }

    #[cfg(feature = "replay")]
    {
        let needle = query.trim().to_ascii_lowercase();
        let haystack = format!(
            "{} {}",
            state.contract.name.to_ascii_lowercase(),
            state.contract.description.to_ascii_lowercase()
        );
        if needle.is_empty() || haystack.contains(&needle) {
            vec![state.contract.clone()]
                .into_iter()
                .take(limit.max(1))
                .collect()
        } else {
            Vec::new()
        }
    }
}

pub(crate) fn spawn_replay_market_task(
    replay: ReplayState,
    cfg: AppConfig,
    contract: ContractSuggestion,
    bar_type: BarType,
    candle_mode: CandleMode,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_rx: tokio::sync::watch::Receiver<ReplaySpeed>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (replay, cfg, contract, broker_tx, replay_speed_rx);
        tokio::spawn(async move {
            let _ = internal_tx.send(InternalEvent::Error(
                "replay mode is not enabled in this build".to_string(),
            ));
            let _ = (bar_type, candle_mode);
        })
    }

    #[cfg(feature = "replay")]
    {
        tokio::spawn(async move {
            let mut replay_speed_rx = replay_speed_rx;
            if let Err(err) = replay_market_worker_inner(
                replay,
                cfg,
                contract,
                bar_type,
                candle_mode,
                broker_tx,
                &mut replay_speed_rx,
                internal_tx.clone(),
            )
            .await
            {
                let _ = internal_tx.send(InternalEvent::Error(format!("replay data: {err}")));
            }
        })
    }
}

#[cfg(feature = "replay")]
fn load_replay_state_blocking(
    cfg: &AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
) -> Result<ReplayState> {
    let library = ReplayCacheLibrary::scan(&cfg.replay_cache_dir);
    if let Some(cached) = library.load_first_server_bars_jsonl(bar_type, candle_mode, None)? {
        return replay_state_from_cached_server_bars(cached, bar_type, candle_mode);
    }

    load_local_tick_replay_state_blocking(&cfg.replay_file_path)
}

#[cfg(feature = "replay")]
fn load_local_tick_replay_state_blocking(path: &Path) -> Result<ReplayState> {
    let resolved_path = resolve_replay_path(path)?;
    let file = File::open(&resolved_path)
        .with_context(|| format!("open replay file {}", resolved_path.display()))?;
    let reader = BufReader::new(file);
    let contract_name = infer_contract_name(&resolved_path);
    let tick_size = infer_tick_size(&contract_name);
    let value_per_point = infer_value_per_point(&contract_name);
    let mut ticks = Vec::new();

    let mut line_count = 0usize;
    for line in reader.lines() {
        let line = line.with_context(|| format!("read replay file {}", resolved_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let tick = parse_tick_line(&line).with_context(|| {
            format!(
                "parse replay tick {line_count} in {}",
                resolved_path.display()
            )
        })?;
        ticks.push(tick);
        line_count = line_count.saturating_add(1);
    }

    if line_count == 0 {
        bail!("replay file {} contained no ticks", resolved_path.display());
    }
    ticks.sort_by_key(|tick| tick.ts_ns);

    let contract_id = replay_contract_id(&resolved_path);
    let description = format!("Replay Dataset ({})", resolved_path.display());
    let contract = ContractSuggestion {
        id: contract_id,
        name: contract_name.clone(),
        description: description.clone(),
        raw: json!({
            "source": "replay",
            "path": resolved_path.display().to_string(),
            "configuredPath": path.display().to_string(),
        }),
    };
    let account = AccountInfo {
        id: 1,
        name: "REPLAY".to_string(),
        raw: json!({
            "id": 1,
            "name": "REPLAY",
            "source": "replay",
            "startingBalance": 100000.0,
            "balance": 100000.0,
        }),
    };

    Ok(ReplayState {
        contract,
        account,
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(value_per_point),
            tick_size: Some(tick_size),
        },
        data: ReplayDataSource::LocalTicks(Arc::from(ticks.into_boxed_slice())),
    })
}

#[cfg(feature = "replay")]
fn replay_state_from_cached_server_bars(
    cached: ReplayCacheLoadedServerBars,
    requested_bar_type: BarType,
    requested_candle_mode: CandleMode,
) -> Result<ReplayState> {
    let contract_name = if cached.manifest.contract.symbol.trim().is_empty() {
        "Replay Cache".to_string()
    } else {
        cached.manifest.contract.symbol.clone()
    };
    let contract_id = cached
        .manifest
        .contract
        .id
        .unwrap_or_else(|| replay_contract_id(&cached.data_path));
    let data_bar_type = cached
        .file
        .market_shape
        .bar_type
        .unwrap_or(requested_bar_type);
    let source_label = format!("cache {}", cached.manifest.display_name);
    let description = format!("Cached Replay Dataset ({})", cached.manifest.display_name);
    let session_profile = match cached.file.market_shape.session_template.as_deref() {
        Some(template) if template.eq_ignore_ascii_case("rth") => {
            InstrumentSessionProfile::EquityRth
        }
        _ => InstrumentSessionProfile::FuturesGlobex,
    };
    let bars = cached.bars;
    if bars.is_empty() {
        bail!(
            "cached replay dataset {} contained no bars",
            cached.data_path.display()
        );
    }

    Ok(ReplayState {
        contract: ContractSuggestion {
            id: contract_id,
            name: contract_name,
            description,
            raw: json!({
                "source": "replay-cache",
                "manifestPath": cached.manifest_path.display().to_string(),
                "dataPath": cached.data_path.display().to_string(),
                "requestedBarType": requested_bar_type,
                "requestedCandleMode": requested_candle_mode,
            }),
        },
        account: AccountInfo {
            id: 1,
            name: "REPLAY".to_string(),
            raw: json!({
                "id": 1,
                "name": "REPLAY",
                "source": "replay-cache",
                "startingBalance": 100000.0,
                "balance": 100000.0,
            }),
        },
        market_specs: MarketSpecs {
            session_profile: Some(session_profile),
            value_per_point: Some(cached.manifest.tick_specs.value_per_point),
            tick_size: Some(cached.manifest.tick_specs.tick_size),
        },
        data: ReplayDataSource::CachedServerBars {
            bars: Arc::from(bars.into_boxed_slice()),
            bar_type: data_bar_type,
            source_label,
        },
    })
}

#[cfg(feature = "replay")]
fn resolve_replay_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        if path.is_file() {
            return Ok(path.to_path_buf());
        }
        bail!("replay file {} was not found", path.display());
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cwd = std::env::current_dir().ok();
    let candidates = replay_path_candidates(path, cwd.as_deref(), manifest_dir);
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }

    let tried = candidates
        .into_iter()
        .map(|candidate| candidate.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "replay file {} was not found; tried {}",
        path.display(),
        tried
    );
}

#[cfg(feature = "replay")]
fn replay_path_candidates(path: &Path, cwd: Option<&Path>, manifest_dir: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(cwd) = cwd {
        push_replay_candidate(&mut candidates, cwd.join(path));
    }
    push_replay_candidate(&mut candidates, manifest_dir.join(path));
    if let Some(workspace_root) = manifest_dir.parent() {
        push_replay_candidate(&mut candidates, workspace_root.join(path));
    }
    candidates
}

#[cfg(feature = "replay")]
fn push_replay_candidate(candidates: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !candidates.iter().any(|existing| existing == &candidate) {
        candidates.push(candidate);
    }
}

#[cfg(feature = "replay")]
async fn replay_market_worker_inner(
    replay: ReplayState,
    cfg: AppConfig,
    contract: ContractSuggestion,
    bar_type: BarType,
    candle_mode: CandleMode,
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_rx: &mut tokio::sync::watch::Receiver<ReplaySpeed>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let bars = replay.bars_for_type(bar_type)?;
    if bars.is_empty() {
        bail!("no {} bars available in replay dataset", bar_type.label());
    }

    let total_bars = bars.len();
    let history_loaded = if total_bars > 1 {
        cfg.history_bars.max(1).min(total_bars - 1)
    } else {
        total_bars
    };
    let mut series = LiveSeries::new();
    for bar in &bars[..history_loaded] {
        series.push_closed_bar_capped(bar, ENGINE_MARKET_BAR_LIMIT);
    }

    let initial_status = format!(
        "Replay {} loaded for {} ({}/{})",
        bar_type.mode_label(candle_mode),
        contract.name,
        history_loaded,
        total_bars
    );
    if let Some(update) = build_market_update(
        &contract,
        Some(replay.market_specs),
        candle_mode,
        series.closed_bars.len(),
        0,
        initial_status,
        0,
        None,
        None,
        &series,
    ) {
        let _ = internal_tx.send(InternalEvent::Market(update));
    }

    let mut live_bars = 0usize;
    for bar in &bars[history_loaded..] {
        wait_for_replay_bar(
            series.closed_bars.last().map(|previous| previous.ts_ns),
            bar.ts_ns,
            cfg.replay_bar_interval_ms,
            replay_speed_rx,
        )
        .await;
        process_replay_bar(&broker_tx, bar).await?;
        let before_closed_len = series.closed_bars.len();
        let before_last_closed = series.closed_bars.last().cloned();
        let before_forming = series.forming_bar.clone();
        series.push_closed_bar_capped(bar, ENGINE_MARKET_BAR_LIMIT);
        live_bars = live_bars.saturating_add(1);
        let status = format!(
            "Replay {} streaming for {} ({}/{})",
            bar_type.mode_label(candle_mode),
            contract.name,
            history_loaded + live_bars,
            total_bars
        );
        if let Some(update) = build_market_update(
            &contract,
            Some(replay.market_specs),
            candle_mode,
            series.closed_bars.len(),
            live_bars,
            status,
            before_closed_len,
            before_last_closed,
            before_forming,
            &series,
        ) {
            let _ = internal_tx.send(InternalEvent::Market(update));
        }
    }

    let _ = internal_tx.send(InternalEvent::UserSocketStatus(format!(
        "Replay complete for {} ({})",
        contract.name,
        bar_type.label()
    )));
    Ok(())
}

#[cfg(feature = "replay")]
async fn process_replay_bar(broker_tx: &UnboundedSender<BrokerCommand>, bar: &Bar) -> Result<()> {
    let (response_tx, response_rx) = oneshot::channel();
    broker_tx
        .send(BrokerCommand::ReplayBar {
            bar: bar.clone(),
            response_tx,
        })
        .map_err(|_| anyhow::anyhow!("replay broker task is unavailable"))?;
    let _ = response_rx.await;
    Ok(())
}

#[cfg(feature = "replay")]
async fn wait_for_replay_bar(
    previous_ts_ns: Option<i64>,
    next_ts_ns: i64,
    fallback_ms: u64,
    replay_speed_rx: &mut tokio::sync::watch::Receiver<ReplaySpeed>,
) {
    let mut remaining = replay_gap_duration(previous_ts_ns, next_ts_ns, fallback_ms);
    while !remaining.is_zero() {
        let speed = *replay_speed_rx.borrow();
        let wall_sleep = scale_duration(remaining, 1.0 / speed.multiplier());
        if wall_sleep.is_zero() {
            break;
        }

        let started_at = time::Instant::now();
        tokio::select! {
            _ = tokio::time::sleep(wall_sleep) => break,
            changed = replay_speed_rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let consumed_market = scale_duration(started_at.elapsed(), speed.multiplier());
                remaining = remaining.saturating_sub(consumed_market);
            }
        }
    }
}

#[cfg(feature = "replay")]
fn replay_gap_duration(previous_ts_ns: Option<i64>, next_ts_ns: i64, fallback_ms: u64) -> Duration {
    match previous_ts_ns {
        Some(previous_ts_ns) if next_ts_ns > previous_ts_ns => {
            Duration::from_nanos((next_ts_ns - previous_ts_ns) as u64)
        }
        Some(_) => Duration::ZERO,
        None => Duration::from_millis(fallback_ms),
    }
}

#[cfg(feature = "replay")]
fn scale_duration(duration: Duration, factor: f64) -> Duration {
    if !factor.is_finite() || factor <= 0.0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64((duration.as_secs_f64() * factor).max(0.0))
}

#[cfg(feature = "replay")]
impl ReplayState {
    fn bars_for_type(&self, bar_type: BarType) -> Result<Vec<Bar>> {
        match &self.data {
            ReplayDataSource::LocalTicks(ticks) => {
                let ticks = ticks.as_ref();
                match bar_type.kind() {
                    BarKind::Minute => {
                        let interval = i64::from(bar_type.value()) * 60 * 1_000_000_000;
                        Ok(build_time_bars(ticks, interval.max(1)))
                    }
                    BarKind::Second => {
                        let interval = i64::from(bar_type.value()) * 1_000_000_000;
                        Ok(build_time_bars(ticks, interval.max(1)))
                    }
                    BarKind::Tick => {
                        let mut bars = build_tick_count_bars(ticks, bar_type.value());
                        make_bar_timestamps_strictly_increasing(&mut bars);
                        Ok(bars)
                    }
                    BarKind::Range => {
                        let mut bars = build_range_bars(
                            ticks,
                            self.market_specs.tick_size.unwrap_or(0.25).max(0.01)
                                * f64::from(bar_type.value()),
                        );
                        make_bar_timestamps_strictly_increasing(&mut bars);
                        Ok(bars)
                    }
                    BarKind::Volume => {
                        bail!(
                            "volume bars require trade size; local replay file only has trusted last prices"
                        )
                    }
                }
            }
            ReplayDataSource::CachedServerBars {
                bars,
                bar_type: cached_bar_type,
                source_label,
            } => {
                if *cached_bar_type != bar_type {
                    bail!(
                        "cached server-bar replay from {source_label} contains {}; requested {}",
                        cached_bar_type.label(),
                        bar_type.label()
                    );
                }
                Ok(bars.as_ref().to_vec())
            }
        }
    }
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone)]
enum ReplayDataSource {
    LocalTicks(Arc<[ReplayTick]>),
    CachedServerBars {
        bars: Arc<[Bar]>,
        bar_type: BarType,
        source_label: String,
    },
}

#[cfg(feature = "replay")]
#[derive(Debug, Clone, Copy)]
struct ReplayTick {
    ts_ns: i64,
    last: f64,
}

#[cfg(feature = "replay")]
fn parse_tick_line(line: &str) -> Result<ReplayTick> {
    let mut fields = line.split(';');
    let prefix = fields
        .next()
        .context("replay line missing timestamp prefix")?;
    let last = fields
        .next()
        .context("replay line missing last price")?
        .trim()
        .parse::<f64>()
        .context("parse last price")?;

    let mut prefix_parts = prefix.split_whitespace();
    let date_raw = prefix_parts.next().context("replay line missing date")?;
    let time_raw = prefix_parts.next().context("replay line missing time")?;
    let fraction_raw = prefix_parts
        .next()
        .context("replay line missing fractional time")?;

    let date = NaiveDate::parse_from_str(date_raw, "%Y%m%d").context("parse replay date")?;
    let time = NaiveTime::parse_from_str(time_raw, "%H%M%S").context("parse replay time")?;
    let fraction = fraction_raw
        .trim()
        .parse::<u32>()
        .context("parse replay fractional time")?;
    let nanos = fraction
        .checked_mul(100)
        .context("replay fractional time overflowed nanoseconds")?;
    let naive = date
        .and_hms_nano_opt(time.hour(), time.minute(), time.second(), nanos)
        .context("compose replay timestamp")?;
    let ts_ns = New_York
        .from_local_datetime(&naive)
        .single()
        .or_else(|| New_York.from_local_datetime(&naive).earliest())
        .or_else(|| New_York.from_local_datetime(&naive).latest())
        .context("resolve replay timestamp in America/New_York")?
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
        .context("convert replay timestamp to nanoseconds")?;

    Ok(ReplayTick { ts_ns, last })
}

#[cfg(feature = "replay")]
fn build_time_bars(ticks: &[ReplayTick], interval_ns: i64) -> Vec<Bar> {
    let mut builder = TimeBarBuilder::new(interval_ns);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last);
    }
    builder.finish()
}

#[cfg(feature = "replay")]
fn build_tick_count_bars(ticks: &[ReplayTick], ticks_per_bar: u32) -> Vec<Bar> {
    let mut builder = TickCountBarBuilder::new(ticks_per_bar);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last);
    }
    builder.finish()
}

#[cfg(feature = "replay")]
fn build_range_bars(ticks: &[ReplayTick], range_size: f64) -> Vec<Bar> {
    let mut builder = RangeBarBuilder::new(range_size);
    for tick in ticks {
        builder.push_tick(tick.ts_ns, tick.last);
    }
    builder.finish()
}

#[cfg(feature = "replay")]
fn make_bar_timestamps_strictly_increasing(bars: &mut [Bar]) {
    let mut last_ts = None::<i64>;
    for bar in bars {
        if let Some(last) = last_ts
            && bar.ts_ns <= last
        {
            bar.ts_ns = last.saturating_add(1);
        }
        last_ts = Some(bar.ts_ns);
    }
}

#[cfg(feature = "replay")]
struct TimeBarBuilder {
    interval_ns: i64,
    current_period_ts_ns: Option<i64>,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

#[cfg(feature = "replay")]
impl TimeBarBuilder {
    fn new(interval_ns: i64) -> Self {
        Self {
            interval_ns: interval_ns.max(1),
            current_period_ts_ns: None,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    fn push_tick(&mut self, ts_ns: i64, price: f64) {
        let period_ts_ns = ts_ns - ts_ns.rem_euclid(self.interval_ns);
        match self.current_bar.as_mut() {
            Some(current) if self.current_period_ts_ns == Some(period_ts_ns) => {
                current.high = current.high.max(price);
                current.low = current.low.min(price);
                current.close = price;
            }
            Some(_) => {
                if let Some(current) = self.current_bar.take() {
                    self.bars.push(current);
                }
                self.current_period_ts_ns = Some(period_ts_ns);
                self.current_bar = Some(Bar {
                    ts_ns: period_ts_ns,
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                    volume: None,
                });
            }
            None => {
                self.current_period_ts_ns = Some(period_ts_ns);
                self.current_bar = Some(Bar {
                    ts_ns: period_ts_ns,
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                    volume: None,
                });
            }
        }
    }

    fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

#[cfg(feature = "replay")]
struct TickCountBarBuilder {
    ticks_per_bar: usize,
    current_tick_count: usize,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

#[cfg(feature = "replay")]
impl TickCountBarBuilder {
    fn new(ticks_per_bar: u32) -> Self {
        Self {
            ticks_per_bar: ticks_per_bar.max(1) as usize,
            current_tick_count: 0,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    fn push_tick(&mut self, ts_ns: i64, price: f64) {
        if self.current_bar.is_none() || self.current_tick_count >= self.ticks_per_bar {
            if let Some(current) = self.current_bar.take() {
                self.bars.push(current);
            }
            self.current_tick_count = 0;
            self.current_bar = Some(Bar {
                ts_ns,
                open: price,
                high: price,
                low: price,
                close: price,
                volume: None,
            });
        }

        if let Some(current) = self.current_bar.as_mut() {
            current.ts_ns = ts_ns;
            current.high = current.high.max(price);
            current.low = current.low.min(price);
            current.close = price;
            self.current_tick_count = self.current_tick_count.saturating_add(1);
        }
    }

    fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

#[cfg(feature = "replay")]
struct RangeBarBuilder {
    range_size: f64,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

#[cfg(feature = "replay")]
impl RangeBarBuilder {
    fn new(range_size: f64) -> Self {
        Self {
            range_size,
            current_bar: None,
            bars: Vec::new(),
        }
    }

    fn push_tick(&mut self, ts_ns: i64, price: f64) {
        const EPSILON: f64 = 1e-9;
        let mut current = self.current_bar.take().unwrap_or(Bar {
            ts_ns,
            open: price,
            high: price,
            low: price,
            close: price,
            volume: None,
        });

        loop {
            let tentative_high = current.high.max(price);
            let tentative_low = current.low.min(price);
            let breaks_up =
                price > current.high && (price - tentative_low) >= self.range_size - EPSILON;
            let breaks_down =
                price < current.low && (tentative_high - price) >= self.range_size - EPSILON;

            if breaks_up {
                let close = tentative_low + self.range_size;
                current.high = close;
                current.close = close;
                current.ts_ns = ts_ns;
                self.bars.push(current.clone());
                current = Bar {
                    ts_ns,
                    open: close,
                    high: close,
                    low: close,
                    close,
                    volume: None,
                };
                if price <= close + EPSILON {
                    break;
                }
                continue;
            }

            if breaks_down {
                let close = tentative_high - self.range_size;
                current.low = close;
                current.close = close;
                current.ts_ns = ts_ns;
                self.bars.push(current.clone());
                current = Bar {
                    ts_ns,
                    open: close,
                    high: close,
                    low: close,
                    close,
                    volume: None,
                };
                if price >= close - EPSILON {
                    break;
                }
                continue;
            }

            current.high = tentative_high;
            current.low = tentative_low;
            current.close = price;
            current.ts_ns = ts_ns;
            break;
        }

        self.current_bar = Some(current);
    }

    fn finish(mut self) -> Vec<Bar> {
        if let Some(current) = self.current_bar.take() {
            self.bars.push(current);
        }
        self.bars
    }
}

#[cfg(feature = "replay")]
fn infer_contract_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("Replay");
    stem.trim_end_matches(".Last").to_string()
}

#[cfg(feature = "replay")]
fn replay_contract_id(path: &Path) -> i64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    (hasher.finish() & 0x3FFF_FFFF_FFFF_FFFF) as i64
}

#[cfg(feature = "replay")]
fn infer_tick_size(contract_name: &str) -> f64 {
    let symbol = contract_name
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    match symbol.as_str() {
        "ES" | "MES" | "NQ" | "MNQ" | "RTY" | "M2K" | "YM" | "MYM" => 0.25,
        "CL" | "MCL" => 0.01,
        "GC" | "MGC" => 0.1,
        _ => 0.25,
    }
}

#[cfg(feature = "replay")]
fn infer_value_per_point(contract_name: &str) -> f64 {
    let symbol = contract_name
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    match symbol.as_str() {
        "ES" => 50.0,
        "MES" => 5.0,
        "NQ" => 20.0,
        "MNQ" => 2.0,
        "RTY" => 50.0,
        "M2K" => 5.0,
        "YM" => 5.0,
        "MYM" => 0.5,
        "CL" => 1000.0,
        "MCL" => 100.0,
        "GC" => 100.0,
        "MGC" => 10.0,
        _ => 1.0,
    }
}

#[cfg(all(test, feature = "replay"))]
mod replay_tests {
    use super::*;
    use crate::replay_cache::{
        ReplayCacheContract, ReplayCacheInstrument, ReplayCacheServerBarsWrite,
        ReplayCacheSourceKind, ReplayCacheTickSpecs, write_server_bars_jsonl_cache,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dt(raw: &str) -> chrono::DateTime<chrono::Utc> {
        raw.parse().expect("valid timestamp")
    }

    fn temp_cache_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("trader-replay-state-{name}-{nonce}"))
    }

    #[test]
    fn parse_tick_line_reads_ninjatrader_last_format() {
        let tick = parse_tick_line("20260324 040000 1800000;6603;6603;6603.25;1")
            .expect("tick should parse");
        assert_eq!(tick.last, 6603.0);
        assert!(tick.ts_ns > 0);
    }

    #[test]
    fn time_bar_builder_groups_ticks_by_interval() {
        let mut builder = TimeBarBuilder::new(2_000_000_000);
        let base = 1_700_000_000_000_000_000i64;
        builder.push_tick(base, 100.0);
        builder.push_tick(base + 1_000_000_000, 101.0);
        builder.push_tick(base + 2_000_000_000, 99.5);
        let bars = builder.finish();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].high, 101.0);
        assert_eq!(bars[0].close, 101.0);
        assert_eq!(bars[1].open, 99.5);
    }

    #[test]
    fn tick_count_bar_builder_groups_by_number_of_ticks() {
        let mut builder = TickCountBarBuilder::new(2);
        let base = 1_700_000_000_000_000_000i64;
        builder.push_tick(base, 100.0);
        builder.push_tick(base + 1, 101.0);
        builder.push_tick(base + 2, 99.5);
        let bars = builder.finish();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].high, 101.0);
        assert_eq!(bars[0].close, 101.0);
        assert_eq!(bars[1].open, 99.5);
        assert_eq!(bars[1].close, 99.5);
    }

    #[test]
    fn range_bar_builder_rolls_on_one_tick_range_breaks() {
        let mut builder = RangeBarBuilder::new(0.25);
        let base = 1_700_000_000_000_000_000i64;
        builder.push_tick(base, 100.0);
        builder.push_tick(base + 1, 100.25);
        builder.push_tick(base + 2, 100.5);
        let bars = builder.finish();

        assert!(bars.len() >= 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].close, 100.25);
        assert_eq!(bars[1].open, 100.25);
    }

    #[test]
    fn replay_state_derives_requested_local_file_bar_types() {
        let base = 1_700_000_000_000_000_000i64;
        let state = ReplayState {
            contract: ContractSuggestion {
                id: 1,
                name: "MESU6".to_string(),
                description: "test replay".to_string(),
                raw: json!({}),
            },
            account: AccountInfo {
                id: 1,
                name: "REPLAY".to_string(),
                raw: json!({}),
            },
            market_specs: MarketSpecs {
                session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
                value_per_point: Some(5.0),
                tick_size: Some(0.25),
            },
            data: ReplayDataSource::LocalTicks(Arc::from(
                vec![
                    ReplayTick {
                        ts_ns: base,
                        last: 100.0,
                    },
                    ReplayTick {
                        ts_ns: base + 1_000_000_000,
                        last: 100.25,
                    },
                    ReplayTick {
                        ts_ns: base + 60_000_000_000,
                        last: 100.5,
                    },
                ]
                .into_boxed_slice(),
            )),
        };

        assert_eq!(state.bars_for_type(BarType::second(2)).unwrap().len(), 2);
        assert_eq!(state.bars_for_type(BarType::minute(1)).unwrap().len(), 2);
        assert_eq!(state.bars_for_type(BarType::tick(2)).unwrap().len(), 2);
        assert!(state.bars_for_type(BarType::range(1)).unwrap().len() >= 2);
        assert!(
            state
                .bars_for_type(BarType::volume(100))
                .expect_err("volume is unsupported for price-only replay")
                .to_string()
                .contains("volume bars require trade size")
        );
    }

    #[test]
    fn replay_state_keeps_duplicate_timestamp_derived_bars_distinct() {
        let base = 1_700_000_000_000_000_000i64;
        let state = ReplayState {
            contract: ContractSuggestion {
                id: 1,
                name: "MESU6".to_string(),
                description: "test replay".to_string(),
                raw: json!({}),
            },
            account: AccountInfo {
                id: 1,
                name: "REPLAY".to_string(),
                raw: json!({}),
            },
            market_specs: MarketSpecs {
                session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
                value_per_point: Some(5.0),
                tick_size: Some(0.25),
            },
            data: ReplayDataSource::LocalTicks(Arc::from(
                vec![
                    ReplayTick {
                        ts_ns: base,
                        last: 100.0,
                    },
                    ReplayTick {
                        ts_ns: base,
                        last: 100.25,
                    },
                    ReplayTick {
                        ts_ns: base,
                        last: 101.0,
                    },
                ]
                .into_boxed_slice(),
            )),
        };

        let tick_bars = state.bars_for_type(BarType::tick(1)).unwrap();
        assert_eq!(tick_bars.len(), 3);
        assert!(
            tick_bars
                .windows(2)
                .all(|window| window[0].ts_ns < window[1].ts_ns)
        );

        let range_bars = state.bars_for_type(BarType::range(1)).unwrap();
        assert!(range_bars.len() >= 3);
        assert!(
            range_bars
                .windows(2)
                .all(|window| window[0].ts_ns < window[1].ts_ns)
        );
    }

    #[test]
    fn replay_path_candidates_cover_launch_crate_and_workspace_roots() {
        let launch_dir = Path::new("/tmp/launch");
        let manifest_dir = Path::new("/tmp/workspace/trader");
        let candidates = replay_path_candidates(
            Path::new("market replay/ES 06-26.Last.txt"),
            Some(launch_dir),
            manifest_dir,
        );

        assert_eq!(
            candidates,
            vec![
                PathBuf::from("/tmp/launch/market replay/ES 06-26.Last.txt"),
                PathBuf::from("/tmp/workspace/trader/market replay/ES 06-26.Last.txt"),
                PathBuf::from("/tmp/workspace/market replay/ES 06-26.Last.txt"),
            ]
        );
    }

    #[test]
    fn replay_path_candidates_support_workspace_relative_inputs() {
        let launch_dir = Path::new("/tmp/launch");
        let manifest_dir = Path::new("/tmp/workspace/trader");
        let candidates = replay_path_candidates(
            Path::new("trader/market replay/ES 06-26.Last.txt"),
            Some(launch_dir),
            manifest_dir,
        );

        assert!(
            candidates.contains(&PathBuf::from(
                "/tmp/workspace/trader/market replay/ES 06-26.Last.txt"
            )),
            "workspace-root-relative replay paths should be accepted"
        );
    }

    #[test]
    fn replay_gap_duration_uses_market_timestamps() {
        let gap = replay_gap_duration(Some(1_000_000_000), 61_000_000_000, 5);
        assert_eq!(gap, Duration::from_secs(60));

        let zero_gap = replay_gap_duration(Some(61_000_000_000), 61_000_000_000, 5);
        assert_eq!(zero_gap, Duration::ZERO);
    }

    #[test]
    fn cached_server_bar_state_serves_only_cached_bar_shape() {
        let base = 1_700_000_000_000_000_000i64;
        let state = ReplayState {
            contract: ContractSuggestion {
                id: 1,
                name: "MESU6".to_string(),
                description: "cached replay".to_string(),
                raw: json!({}),
            },
            account: AccountInfo {
                id: 1,
                name: "REPLAY".to_string(),
                raw: json!({}),
            },
            market_specs: MarketSpecs {
                session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
                value_per_point: Some(5.0),
                tick_size: Some(0.25),
            },
            data: ReplayDataSource::CachedServerBars {
                bar_type: BarType::volume(6500),
                source_label: "cache fixture".to_string(),
                bars: Arc::from(
                    vec![Bar {
                        ts_ns: base,
                        open: 100.0,
                        high: 101.0,
                        low: 99.0,
                        close: 100.5,
                        volume: Some(6500.0),
                    }]
                    .into_boxed_slice(),
                ),
            },
        };

        assert_eq!(state.bars_for_type(BarType::volume(6500)).unwrap().len(), 1);
        assert!(
            state
                .bars_for_type(BarType::minute(1))
                .expect_err("cached server bars are exact-shape data")
                .to_string()
                .contains("contains 6500 Vol; requested 1 Min")
        );
    }

    #[test]
    fn load_replay_state_prefers_matching_cached_server_bars() {
        let cache_root = temp_cache_dir("cache-load");
        write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
            cache_root: cache_root.clone(),
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(25866054),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({"source": "unit-test"}),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            session_template: Some("Globex".to_string()),
            bars: vec![Bar {
                ts_ns: dt("2026-07-23T00:00:00Z")
                    .timestamp_nanos_opt()
                    .expect("timestamp ns"),
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: Some(1000.0),
            }],
            warnings: Vec::new(),
            notes: None,
        })
        .expect("write cache");

        let mut cfg = AppConfig::default();
        cfg.replay_cache_dir = cache_root;
        cfg.replay_file_path = PathBuf::from("/tmp/trader-replay-missing-local.Last.txt");
        let state = load_replay_state_blocking(&cfg, BarType::minute(1), CandleMode::HeikinAshi)
            .expect("load replay state from cache");

        assert_eq!(replay_contract(&state).name, "MESU6");
        let bars = state
            .bars_for_type(BarType::minute(1))
            .expect("cached bars");
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].volume, Some(1000.0));
    }

    #[test]
    fn scale_duration_applies_replay_multiplier() {
        assert_eq!(
            scale_duration(Duration::from_secs(60), 0.1),
            Duration::from_secs(6)
        );
        assert_eq!(
            scale_duration(Duration::from_millis(250), 2.0),
            Duration::from_millis(500)
        );
    }
}

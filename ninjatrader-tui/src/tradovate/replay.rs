use super::*;
#[cfg(feature = "replay")]
use anyhow::{Context, Result, bail};
#[cfg(not(feature = "replay"))]
use anyhow::{Result, bail};
#[cfg(feature = "replay")]
use chrono::{NaiveDate, NaiveTime, TimeZone};
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
    minute1_bars: Arc<[Bar]>,
    #[cfg(feature = "replay")]
    range1_bars: Arc<[Bar]>,
}

pub(crate) async fn load_replay_state(cfg: &AppConfig) -> Result<ReplayState> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = cfg;
        bail!("replay mode is not enabled in this build; rebuild with `--features replay`");
    }

    #[cfg(feature = "replay")]
    {
        let replay_path = cfg.replay_file_path.clone();
        tokio::task::spawn_blocking(move || load_replay_state_blocking(&replay_path))
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
            let _ = bar_type;
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
fn load_replay_state_blocking(path: &Path) -> Result<ReplayState> {
    let resolved_path = resolve_replay_path(path)?;
    let file = File::open(&resolved_path)
        .with_context(|| format!("open replay file {}", resolved_path.display()))?;
    let reader = BufReader::new(file);
    let contract_name = infer_contract_name(&resolved_path);
    let tick_size = infer_tick_size(&contract_name);
    let value_per_point = infer_value_per_point(&contract_name);
    let mut minute_builder = MinuteBarBuilder::default();
    let mut range_builder = RangeBarBuilder::new(tick_size.max(0.01));

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
        minute_builder.push_tick(tick.ts_ns, tick.last);
        range_builder.push_tick(tick.ts_ns, tick.last);
        line_count = line_count.saturating_add(1);
    }

    if line_count == 0 {
        bail!("replay file {} contained no ticks", resolved_path.display());
    }

    let minute1_bars = minute_builder.finish();
    let range1_bars = range_builder.finish();
    if minute1_bars.is_empty() || range1_bars.is_empty() {
        bail!(
            "replay file {} did not produce minute/range bars",
            resolved_path.display()
        );
    }

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
        minute1_bars: Arc::from(minute1_bars.into_boxed_slice()),
        range1_bars: Arc::from(range1_bars.into_boxed_slice()),
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
    broker_tx: UnboundedSender<BrokerCommand>,
    replay_speed_rx: &mut tokio::sync::watch::Receiver<ReplaySpeed>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> Result<()> {
    let bars = replay.bars_for_type(bar_type);
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
        bar_type.label(),
        contract.name,
        history_loaded,
        total_bars
    );
    if let Some(update) = build_market_update(
        &contract,
        Some(replay.market_specs),
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
            bar_type.label(),
            contract.name,
            history_loaded + live_bars,
            total_bars
        );
        if let Some(update) = build_market_update(
            &contract,
            Some(replay.market_specs),
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
    fn bars_for_type(&self, bar_type: BarType) -> &[Bar] {
        match bar_type {
            BarType::Minute1 => &self.minute1_bars,
            BarType::Range1 => &self.range1_bars,
        }
    }
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
#[derive(Default)]
struct MinuteBarBuilder {
    current_minute_ts_ns: Option<i64>,
    current_bar: Option<Bar>,
    bars: Vec<Bar>,
}

#[cfg(feature = "replay")]
impl MinuteBarBuilder {
    fn push_tick(&mut self, ts_ns: i64, price: f64) {
        let minute_ts_ns = ts_ns - ts_ns.rem_euclid(60 * 1_000_000_000);
        match self.current_bar.as_mut() {
            Some(current) if self.current_minute_ts_ns == Some(minute_ts_ns) => {
                current.high = current.high.max(price);
                current.low = current.low.min(price);
                current.close = price;
            }
            Some(_) => {
                if let Some(current) = self.current_bar.take() {
                    self.bars.push(current);
                }
                self.current_minute_ts_ns = Some(minute_ts_ns);
                self.current_bar = Some(Bar {
                    ts_ns: minute_ts_ns,
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                });
            }
            None => {
                self.current_minute_ts_ns = Some(minute_ts_ns);
                self.current_bar = Some(Bar {
                    ts_ns: minute_ts_ns,
                    open: price,
                    high: price,
                    low: price,
                    close: price,
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

    #[test]
    fn parse_tick_line_reads_ninjatrader_last_format() {
        let tick = parse_tick_line("20260324 040000 1800000;6603;6603;6603.25;1")
            .expect("tick should parse");
        assert_eq!(tick.last, 6603.0);
        assert!(tick.ts_ns > 0);
    }

    #[test]
    fn minute_bar_builder_groups_ticks_by_minute() {
        let mut builder = MinuteBarBuilder::default();
        let base = 1_700_000_000_000_000_000i64;
        builder.push_tick(base, 100.0);
        builder.push_tick(base + 1_000_000_000, 101.0);
        builder.push_tick(base + 60_000_000_000, 99.5);
        let bars = builder.finish();

        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].high, 101.0);
        assert_eq!(bars[0].close, 101.0);
        assert_eq!(bars[1].open, 99.5);
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
    fn replay_path_candidates_cover_launch_crate_and_workspace_roots() {
        let launch_dir = Path::new("/tmp/launch");
        let manifest_dir = Path::new("/tmp/workspace/ninjatrader-tui");
        let candidates = replay_path_candidates(
            Path::new("market replay/ES 06-26.Last.txt"),
            Some(launch_dir),
            manifest_dir,
        );

        assert_eq!(
            candidates,
            vec![
                PathBuf::from("/tmp/launch/market replay/ES 06-26.Last.txt"),
                PathBuf::from("/tmp/workspace/ninjatrader-tui/market replay/ES 06-26.Last.txt"),
                PathBuf::from("/tmp/workspace/market replay/ES 06-26.Last.txt"),
            ]
        );
    }

    #[test]
    fn replay_path_candidates_support_workspace_relative_inputs() {
        let launch_dir = Path::new("/tmp/launch");
        let manifest_dir = Path::new("/tmp/workspace/ninjatrader-tui");
        let candidates = replay_path_candidates(
            Path::new("ninjatrader-tui/market replay/ES 06-26.Last.txt"),
            Some(launch_dir),
            manifest_dir,
        );

        assert!(
            candidates.contains(&PathBuf::from(
                "/tmp/workspace/ninjatrader-tui/market replay/ES 06-26.Last.txt"
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

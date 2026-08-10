use super::*;
use chrono::{
    Datelike, Duration as ChronoDuration, LocalResult, NaiveDateTime, NaiveTime, TimeZone, Weekday,
};
use chrono_tz::Tz;
use std::str::FromStr;

pub const REPLAY_DATASET_VIEW_VERSION: u32 = 1;
pub const REPLAY_DATASET_VIEWS_DIR: &str = ".views";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayDatasetSourceRef {
    /// Cache-root-relative path to the source `manifest.json`.
    pub manifest_id: String,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: String,
    pub contract: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDatasetSessionPreset {
    FullSource,
    FuturesGlobex,
    FuturesRthNewYork,
    FuturesRthChicago,
    CustomLocal,
    CustomUtc,
}

impl ReplayDatasetSessionPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::FullSource => "Full source",
            Self::FuturesGlobex => "Futures Globex",
            Self::FuturesRthNewYork => "Futures RTH (New York)",
            Self::FuturesRthChicago => "Futures RTH (Chicago)",
            Self::CustomLocal => "Custom local time",
            Self::CustomUtc => "Custom UTC",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::FullSource => Self::FuturesGlobex,
            Self::FuturesGlobex => Self::FuturesRthNewYork,
            Self::FuturesRthNewYork => Self::FuturesRthChicago,
            Self::FuturesRthChicago => Self::CustomLocal,
            Self::CustomLocal => Self::CustomUtc,
            Self::CustomUtc => Self::FullSource,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Self::FullSource => Self::CustomUtc,
            Self::FuturesGlobex => Self::FullSource,
            Self::FuturesRthNewYork => Self::FuturesGlobex,
            Self::FuturesRthChicago => Self::FuturesRthNewYork,
            Self::CustomLocal => Self::FuturesRthChicago,
            Self::CustomUtc => Self::CustomLocal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "preset", rename_all = "snake_case")]
pub enum ReplayDatasetSessionSelection {
    FullSource,
    FuturesGlobex {
        trading_date: NaiveDate,
    },
    FuturesRthNewYork {
        trading_date: NaiveDate,
    },
    FuturesRthChicago {
        trading_date: NaiveDate,
    },
    CustomLocal {
        start: NaiveDateTime,
        end: NaiveDateTime,
        timezone: String,
    },
    CustomUtc {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedReplayDatasetSession {
    pub preset: ReplayDatasetSessionPreset,
    pub evaluation_range: ReplayCacheTimeRange,
    pub input_timezone: String,
}

impl ReplayDatasetSessionSelection {
    pub fn resolve(
        &self,
        source_coverage: &ReplayCacheCoverage,
    ) -> Result<ResolvedReplayDatasetSession> {
        match self {
            Self::FullSource => Ok(ResolvedReplayDatasetSession {
                preset: ReplayDatasetSessionPreset::FullSource,
                evaluation_range: ReplayCacheTimeRange::new(
                    source_coverage.start,
                    source_coverage.end,
                )?,
                input_timezone: "UTC".to_string(),
            }),
            Self::FuturesGlobex { trading_date } => {
                validate_futures_trading_date(*trading_date)?;
                let open_date = trading_date
                    .pred_opt()
                    .context("Globex trading date has no preceding calendar date")?;
                resolve_named_local_session(
                    ReplayDatasetSessionPreset::FuturesGlobex,
                    "America/New_York",
                    open_date
                        .and_hms_opt(18, 0, 0)
                        .context("compose Globex open")?,
                    trading_date
                        .and_hms_opt(17, 0, 0)
                        .context("compose Globex close")?,
                )
            }
            Self::FuturesRthNewYork { trading_date } => {
                validate_futures_trading_date(*trading_date)?;
                resolve_named_local_session(
                    ReplayDatasetSessionPreset::FuturesRthNewYork,
                    "America/New_York",
                    trading_date
                        .and_hms_opt(9, 30, 0)
                        .context("compose New York RTH open")?,
                    trading_date
                        .and_hms_opt(16, 0, 0)
                        .context("compose New York RTH close")?,
                )
            }
            Self::FuturesRthChicago { trading_date } => {
                validate_futures_trading_date(*trading_date)?;
                resolve_named_local_session(
                    ReplayDatasetSessionPreset::FuturesRthChicago,
                    "America/Chicago",
                    trading_date
                        .and_hms_opt(8, 30, 0)
                        .context("compose Chicago RTH open")?,
                    trading_date
                        .and_hms_opt(15, 0, 0)
                        .context("compose Chicago RTH close")?,
                )
            }
            Self::CustomLocal {
                start,
                end,
                timezone,
            } => resolve_named_local_session(
                ReplayDatasetSessionPreset::CustomLocal,
                timezone,
                *start,
                *end,
            ),
            Self::CustomUtc { start, end } => Ok(ResolvedReplayDatasetSession {
                preset: ReplayDatasetSessionPreset::CustomUtc,
                evaluation_range: ReplayCacheTimeRange::new(*start, *end)?,
                input_timezone: "UTC".to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayWarmupTradingPolicy {
    /// Indicators receive warmup data, but execution remains flat until evaluation starts.
    FlatUntilEvaluation,
    /// Reserved for an explicit opt-in once warmup execution is implemented by RBT-022.
    CarryWarmupPosition,
}

impl Default for ReplayWarmupTradingPolicy {
    fn default() -> Self {
        Self::FlatUntilEvaluation
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayDatasetWarmupPolicy {
    #[serde(default)]
    pub duration_seconds: u64,
    #[serde(default)]
    pub trading: ReplayWarmupTradingPolicy,
}

impl Default for ReplayDatasetWarmupPolicy {
    fn default() -> Self {
        Self {
            duration_seconds: 0,
            trading: ReplayWarmupTradingPolicy::FlatUntilEvaluation,
        }
    }
}

/// Optional recurring local-time filter applied to replay bars after the
/// source range is loaded. The immutable source cache is never rewritten.
/// Weekends are excluded because this filter models a weekday trading window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayDatasetDailySessionFilter {
    pub timezone: String,
    pub start_local: NaiveTime,
    pub end_local: NaiveTime,
}

impl ReplayDatasetDailySessionFilter {
    pub fn validate(&self) -> Result<()> {
        Tz::from_str(self.timezone.trim())
            .with_context(|| format!("daily session timezone is invalid: {}", self.timezone))?;
        if self.start_local >= self.end_local {
            bail!(
                "daily session start {} must be before end {}",
                self.start_local,
                self.end_local
            );
        }
        Ok(())
    }

    pub fn label(&self) -> String {
        format!(
            "{} {}-{} weekdays",
            self.timezone, self.start_local, self.end_local
        )
    }

    pub fn contains_timestamp(&self, ts_ns: i64) -> Result<bool> {
        self.validate()?;
        if ts_ns <= 0 {
            return Ok(false);
        }
        let timezone = Tz::from_str(self.timezone.trim())
            .with_context(|| format!("daily session timezone is invalid: {}", self.timezone))?;
        let local = DateTime::<Utc>::from_timestamp_nanos(ts_ns).with_timezone(&timezone);
        if matches!(local.weekday(), Weekday::Sat | Weekday::Sun) {
            return Ok(false);
        }
        let time = local.time();
        Ok(time >= self.start_local && time < self.end_local)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayDatasetView {
    pub view_version: u32,
    pub id: String,
    pub source: ReplayDatasetSourceRef,
    pub evaluation_start: DateTime<Utc>,
    pub evaluation_end: DateTime<Utc>,
    pub input_timezone: String,
    pub session_preset: ReplayDatasetSessionPreset,
    /// Optional recurring local-time filter, evaluated on every loaded bar.
    /// Older view documents omit this field and retain full-source behavior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub daily_session: Option<ReplayDatasetDailySessionFilter>,
    #[serde(default)]
    pub warmup: ReplayDatasetWarmupPolicy,
}

impl ReplayDatasetView {
    pub fn from_session_selection(
        cache_root: &Path,
        dataset: &ReplayCacheDataset,
        id: impl Into<String>,
        selection: &ReplayDatasetSessionSelection,
        warmup: ReplayDatasetWarmupPolicy,
    ) -> Result<Self> {
        let resolved = selection.resolve(&dataset.manifest.coverage)?;
        Self::for_dataset(
            cache_root,
            dataset,
            id,
            resolved.evaluation_range,
            resolved.input_timezone,
            resolved.preset,
            warmup,
        )
    }

    pub fn for_dataset(
        cache_root: &Path,
        dataset: &ReplayCacheDataset,
        id: impl Into<String>,
        evaluation: ReplayCacheTimeRange,
        input_timezone: impl Into<String>,
        session_preset: ReplayDatasetSessionPreset,
        warmup: ReplayDatasetWarmupPolicy,
    ) -> Result<Self> {
        let manifest_id = manifest_id_for_path(cache_root, &dataset.manifest_path)?;
        let view = Self {
            view_version: REPLAY_DATASET_VIEW_VERSION,
            id: id.into(),
            source: ReplayDatasetSourceRef {
                manifest_id,
                provider: dataset.manifest.provider,
                env: dataset.manifest.env,
                instrument: dataset.manifest.instrument.symbol.clone(),
                contract: dataset.manifest.contract.symbol.clone(),
            },
            evaluation_start: evaluation.start,
            evaluation_end: evaluation.end,
            input_timezone: input_timezone.into(),
            session_preset,
            daily_session: None,
            warmup,
        };
        view.validate_model()?;
        validate_view_source_coverage(&view, &dataset.manifest)?;
        Ok(view)
    }

    pub fn evaluation_range(&self) -> Result<ReplayCacheTimeRange> {
        ReplayCacheTimeRange::new(self.evaluation_start, self.evaluation_end)
    }

    pub fn load_range(&self) -> Result<ReplayCacheTimeRange> {
        let evaluation = self.evaluation_range()?;
        let seconds = i64::try_from(self.warmup.duration_seconds)
            .context("dataset view warmup duration is too large")?;
        let start = evaluation
            .start
            .checked_sub_signed(ChronoDuration::seconds(seconds))
            .context("dataset view warmup starts outside the supported timestamp range")?;
        ReplayCacheTimeRange::new(start, evaluation.end)
    }

    pub fn input_tz(&self) -> Result<Tz> {
        Tz::from_str(self.input_timezone.trim()).with_context(|| {
            format!(
                "dataset view input timezone is invalid: {}",
                self.input_timezone
            )
        })
    }

    pub fn evaluation_label(&self) -> Result<String> {
        let timezone = self.input_tz()?;
        let start = self.evaluation_start.with_timezone(&timezone);
        let end = self.evaluation_end.with_timezone(&timezone);
        let range = format!(
            "{}: {} to {}",
            self.session_preset.label(),
            start.format("%Y-%m-%d %H:%M:%S %Z"),
            end.format("%Y-%m-%d %H:%M:%S %Z")
        );
        Ok(match self.daily_session.as_ref() {
            Some(filter) => format!("{range} | daily {}", filter.label()),
            None => range,
        })
    }

    pub fn validate_model(&self) -> Result<()> {
        if self.view_version != REPLAY_DATASET_VIEW_VERSION {
            bail!(
                "unsupported replay dataset view version {}; expected {}",
                self.view_version,
                REPLAY_DATASET_VIEW_VERSION
            );
        }
        validate_view_id(&self.id)?;
        validate_manifest_id(&self.source.manifest_id)?;
        if self.source.instrument.trim().is_empty() || self.source.contract.trim().is_empty() {
            bail!("dataset view source instrument and contract must not be empty");
        }
        self.input_tz()?;
        if let Some(filter) = self.daily_session.as_ref() {
            filter.validate()?;
        }
        self.load_range()?;
        if self.warmup.trading == ReplayWarmupTradingPolicy::CarryWarmupPosition {
            bail!("carrying a warmup position is not implemented; use flat_until_evaluation");
        }
        Ok(())
    }

    /// Apply this view's recurring session filter to an already range-bounded
    /// server-bar sequence. Filtering is stable and never mutates the cache.
    pub fn filter_server_bars(&self, bars: Vec<Bar>) -> Result<Vec<Bar>> {
        let Some(filter) = self.daily_session.as_ref() else {
            return Ok(bars);
        };
        filter.validate()?;
        bars.into_iter()
            .filter_map(|bar| match filter.contains_timestamp(bar.ts_ns) {
                Ok(true) => Some(Ok(bar)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }
}

fn resolve_named_local_session(
    preset: ReplayDatasetSessionPreset,
    timezone: &str,
    start: NaiveDateTime,
    end: NaiveDateTime,
) -> Result<ResolvedReplayDatasetSession> {
    let timezone = Tz::from_str(timezone.trim())
        .with_context(|| format!("session timezone is invalid: {timezone}"))?;
    let start = resolve_local_datetime(timezone, start, "start")?;
    let end = resolve_local_datetime(timezone, end, "end")?;
    Ok(ResolvedReplayDatasetSession {
        preset,
        evaluation_range: ReplayCacheTimeRange::new(
            start.with_timezone(&Utc),
            end.with_timezone(&Utc),
        )?,
        input_timezone: timezone.name().to_string(),
    })
}

fn resolve_local_datetime(
    timezone: Tz,
    local: NaiveDateTime,
    boundary: &str,
) -> Result<DateTime<Tz>> {
    match timezone.from_local_datetime(&local) {
        LocalResult::Single(value) => Ok(value),
        LocalResult::Ambiguous(earliest, latest) => bail!(
            "session {boundary} {local} is ambiguous in {} ({} or {} UTC); choose an unambiguous time or use custom UTC",
            timezone.name(),
            earliest.with_timezone(&Utc),
            latest.with_timezone(&Utc)
        ),
        LocalResult::None => bail!(
            "session {boundary} {local} does not exist in {} because of a timezone transition; choose another time or use custom UTC",
            timezone.name()
        ),
    }
}

fn validate_futures_trading_date(date: NaiveDate) -> Result<()> {
    if matches!(date.weekday(), Weekday::Sat | Weekday::Sun) {
        bail!(
            "futures session presets require a Monday-Friday trading date; {date} is {}",
            date.weekday()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedReplayDatasetView {
    pub view_path: PathBuf,
    pub view: ReplayDatasetView,
    pub dataset: ReplayCacheDataset,
    pub evaluation_range: ReplayCacheTimeRange,
    pub load_range: ReplayCacheTimeRange,
}

impl ResolvedReplayDatasetView {
    pub fn requested_coverage(&self) -> ReplayCacheCoverage {
        ReplayCacheCoverage {
            start: self.load_range.start,
            end: self.load_range.end,
            trading_date: None,
        }
    }

    pub fn load_server_bars(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
    ) -> Result<ReplayCacheLoadedServerBars> {
        let coverage = self.requested_coverage();
        let mut loaded = load_server_bars_cache_file_range(
            &self.dataset,
            bar_type,
            candle_mode,
            Some(&coverage),
            Some(&self.load_range),
        )?;
        loaded.bars = self.view.filter_server_bars(loaded.bars)?;
        Ok(loaded)
    }

    pub fn load_raw_ticks(&self) -> Result<ReplayCacheLoadedRawTicks> {
        let coverage = self.requested_coverage();
        let library = ReplayCacheLibrary {
            root: self
                .dataset
                .manifest_path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            datasets: vec![self.dataset.clone()],
            warnings: Vec::new(),
        };
        library.load_raw_ticks_parquet_dataset_range(
            &self.dataset,
            Some(&coverage),
            Some(&self.load_range),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayDatasetViewStore {
    cache_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayDatasetViewLibrary {
    pub views: Vec<ResolvedReplayDatasetView>,
    pub warnings: Vec<String>,
}

impl ReplayDatasetViewStore {
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        Self {
            cache_root: cache_root.into(),
        }
    }

    pub fn path_for(&self, id: &str) -> Result<PathBuf> {
        validate_view_id(id)?;
        Ok(self
            .cache_root
            .join(REPLAY_DATASET_VIEWS_DIR)
            .join(format!("{id}.json")))
    }

    pub fn save(&self, view: &ReplayDatasetView) -> Result<PathBuf> {
        view.validate_model()?;
        self.resolve_model(view, self.path_for(&view.id)?)?;
        let path = self.path_for(&view.id)?;
        let bytes = serde_json::to_vec_pretty(view).context("serialize replay dataset view")?;
        write_bytes_atomically(&path, &bytes)?;
        Ok(path)
    }

    pub fn list_for_dataset(&self, dataset: &ReplayCacheDataset) -> ReplayDatasetViewLibrary {
        let mut library = ReplayDatasetViewLibrary {
            views: Vec::new(),
            warnings: Vec::new(),
        };
        let expected_manifest_id =
            match manifest_id_for_path(&self.cache_root, &dataset.manifest_path) {
                Ok(id) => id,
                Err(error) => {
                    library.warnings.push(error.to_string());
                    return library;
                }
            };
        let views_dir = self.cache_root.join(REPLAY_DATASET_VIEWS_DIR);
        let entries = match fs::read_dir(&views_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return library,
            Err(error) => {
                library.warnings.push(format!(
                    "read replay dataset views {}: {error}",
                    views_dir.display()
                ));
                return library;
            }
        };

        for entry in entries {
            let path = match entry {
                Ok(entry) => entry.path(),
                Err(error) => {
                    library
                        .warnings
                        .push(format!("read replay dataset view directory entry: {error}"));
                    continue;
                }
            };
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let view = match self.load_path(&path) {
                Ok(view) => view,
                Err(error) => {
                    library.warnings.push(error.to_string());
                    continue;
                }
            };
            if view.source.manifest_id != expected_manifest_id {
                continue;
            }
            match self.resolve_model(&view, path) {
                Ok(resolved) => library.views.push(resolved),
                Err(error) => library.warnings.push(error.to_string()),
            }
        }
        library
            .views
            .sort_by(|left, right| left.view.id.cmp(&right.view.id));
        library
    }

    pub fn load(&self, id: &str) -> Result<ReplayDatasetView> {
        let path = self.path_for(id)?;
        self.load_path(&path)
    }

    pub fn load_path(&self, path: &Path) -> Result<ReplayDatasetView> {
        let expected_parent = self.cache_root.join(REPLAY_DATASET_VIEWS_DIR);
        if path.parent() != Some(expected_parent.as_path()) {
            bail!(
                "dataset view {} is not in {}",
                path.display(),
                expected_parent.display()
            );
        }
        let bytes = fs::read(path)
            .with_context(|| format!("read replay dataset view {}", path.display()))?;
        let view: ReplayDatasetView = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse replay dataset view {}", path.display()))?;
        view.validate_model()?;
        let expected_path = self.path_for(&view.id)?;
        if path != expected_path {
            bail!(
                "dataset view id {} does not match file {}",
                view.id,
                path.display()
            );
        }
        Ok(view)
    }

    pub fn resolve(&self, id: &str) -> Result<ResolvedReplayDatasetView> {
        let path = self.path_for(id)?;
        let view = self.load_path(&path)?;
        self.resolve_model(&view, path)
    }

    pub fn resolve_model(
        &self,
        view: &ReplayDatasetView,
        view_path: PathBuf,
    ) -> Result<ResolvedReplayDatasetView> {
        view.validate_model()?;
        let manifest_path = resolve_manifest_path(&self.cache_root, &view.source.manifest_id)?;
        let manifest = ReplayCacheManifest::from_path(&manifest_path)?;
        if manifest.provider != view.source.provider
            || manifest.env != view.source.env
            || !manifest
                .instrument
                .symbol
                .eq_ignore_ascii_case(&view.source.instrument)
            || !manifest
                .contract
                .symbol
                .eq_ignore_ascii_case(&view.source.contract)
        {
            bail!(
                "dataset view source identity no longer matches manifest {}",
                manifest_path.display()
            );
        }
        let dataset = ReplayCacheDataset {
            dataset_dir: manifest_path
                .parent()
                .context("source manifest has no dataset directory")?
                .to_path_buf(),
            manifest_path,
            manifest,
        };
        let evaluation_range = view.evaluation_range()?;
        let load_range = view.load_range()?;
        validate_view_source_coverage(view, &dataset.manifest)?;
        Ok(ResolvedReplayDatasetView {
            view_path,
            view: view.clone(),
            dataset,
            evaluation_range,
            load_range,
        })
    }
}

fn validate_view_source_coverage(
    view: &ReplayDatasetView,
    manifest: &ReplayCacheManifest,
) -> Result<()> {
    let load_range = view.load_range()?;
    // Generic view validation uses manifest coverage. A raw-tick load
    // additionally requires completed_raw_tick_coverage in the dataset
    // resolver, while a mixed manifest may have wider server-bar data.
    let available = &manifest.coverage;
    let requested = ReplayCacheCoverage {
        start: load_range.start,
        end: load_range.end,
        trading_date: None,
    };
    if !available.contains(&requested) {
        bail!(
            "dataset view range {} to {} (including warmup) is outside source coverage {} to {}",
            load_range.start,
            load_range.end,
            available.start,
            available.end
        );
    }
    Ok(())
}

fn manifest_id_for_path(cache_root: &Path, manifest_path: &Path) -> Result<String> {
    let relative = manifest_path.strip_prefix(cache_root).with_context(|| {
        format!(
            "source manifest {} is not inside cache root {}",
            manifest_path.display(),
            cache_root.display()
        )
    })?;
    validate_safe_relative_path(relative, "source manifest")?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn resolve_manifest_path(cache_root: &Path, manifest_id: &str) -> Result<PathBuf> {
    validate_manifest_id(manifest_id)?;
    let root = fs::canonicalize(cache_root)
        .with_context(|| format!("resolve replay cache root {}", cache_root.display()))?;
    let path = fs::canonicalize(root.join(manifest_id))
        .with_context(|| format!("resolve replay dataset source manifest {manifest_id}"))?;
    if !path.starts_with(&root)
        || path.file_name().and_then(|name| name.to_str()) != Some(MANIFEST_FILE_NAME)
    {
        bail!("dataset view source manifest escapes the replay cache root");
    }
    Ok(path)
}

fn validate_manifest_id(manifest_id: &str) -> Result<()> {
    if manifest_id.trim().is_empty() {
        bail!("dataset view source manifest id must not be empty");
    }
    let path = Path::new(manifest_id);
    validate_safe_relative_path(path, "source manifest id")?;
    if path.file_name().and_then(|name| name.to_str()) != Some(MANIFEST_FILE_NAME) {
        bail!("dataset view source manifest id must end in {MANIFEST_FILE_NAME}");
    }
    Ok(())
}

fn validate_safe_relative_path(path: &Path, label: &str) -> Result<()> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("dataset view {label} must be a safe relative path");
    }
    Ok(())
}

fn validate_view_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 96
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("dataset view id must be 1-96 ASCII letters, numbers, hyphens, or underscores");
    }
    Ok(())
}

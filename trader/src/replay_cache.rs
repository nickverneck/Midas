use crate::broker::{Bar, BarKind, BarType, BrokerKind, CandleMode};
use crate::config::TradingEnvironment;
use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE_NAME: &str = "manifest.json";
pub const MANIFEST_VERSION: u32 = 1;
pub const SERVER_BARS_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheSourceKind {
    ServerBars,
    RawTicks,
    DerivedBars,
    DomStream,
    LocalText,
}

impl ReplayCacheSourceKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ServerBars => "server bars",
            Self::RawTicks => "raw ticks",
            Self::DerivedBars => "derived bars",
            Self::DomStream => "L2/DOM",
            Self::LocalText => "local text",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::ServerBars => "server-bars",
            Self::RawTicks => "raw-ticks",
            Self::DerivedBars => "derived-bars",
            Self::DomStream => "l2-dom",
            Self::LocalText => "local-text",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCacheFileFormat {
    Parquet,
    Jsonl,
    Csv,
    Text,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheInstrument {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchange: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheContract {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiration: Option<NaiveDate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheTickSpecs {
    pub tick_size: f64,
    pub value_per_point: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheCoverage {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trading_date: Option<NaiveDate>,
}

impl ReplayCacheCoverage {
    pub fn contains(&self, requested: &Self) -> bool {
        self.start <= requested.start && self.end >= requested.end
    }

    pub fn label(&self) -> String {
        match self.trading_date {
            Some(date) => format!(
                "{} {} to {}",
                date,
                self.start.format("%H:%M:%S UTC"),
                self.end.format("%H:%M:%S UTC")
            ),
            None => format!(
                "{} to {}",
                self.start.format("%Y-%m-%d %H:%M:%S UTC"),
                self.end.format("%Y-%m-%d %H:%M:%S UTC")
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheMarketShape {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bar_type: Option<BarType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_mode: Option<CandleMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_template: Option<String>,
}

impl ReplayCacheMarketShape {
    pub fn supports(&self, bar_type: BarType, candle_mode: CandleMode) -> bool {
        let bar_matches = self.bar_type.is_none_or(|cached| cached == bar_type);
        let candle_matches = self
            .chart_mode
            .is_none_or(|cached| cached == bar_type.effective_candle_mode(candle_mode));
        bar_matches && candle_matches
    }

    pub fn label(&self) -> String {
        match (self.bar_type, self.chart_mode) {
            (Some(bar_type), Some(candle_mode)) => bar_type.mode_label(candle_mode),
            (Some(bar_type), None) => bar_type.label(),
            (None, Some(candle_mode)) => candle_mode.label().to_string(),
            (None, None) => "unshaped market data".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheDataHash {
    pub algorithm: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheDataFile {
    pub relative_path: PathBuf,
    pub source_kind: ReplayCacheSourceKind,
    pub format: ReplayCacheFileFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<String>,
    pub market_shape: ReplayCacheMarketShape,
    pub row_count: u64,
    pub first_timestamp: DateTime<Utc>,
    pub last_timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_hash: Option<ReplayCacheDataHash>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCacheAppMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheManifest {
    pub manifest_version: u32,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub display_name: String,
    pub coverage: ReplayCacheCoverage,
    pub source_kind: ReplayCacheSourceKind,
    #[serde(default)]
    pub download_request: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    #[serde(default)]
    pub files: Vec<ReplayCacheDataFile>,
    #[serde(default)]
    pub app: Option<ReplayCacheAppMetadata>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub badges: Vec<String>,
    #[serde(default)]
    pub available_bar_shapes: Vec<BarType>,
    #[serde(default)]
    pub available_chart_modes: Vec<CandleMode>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl ReplayCacheManifest {
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut manifest: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        manifest.normalize_derived_fields();
        Ok(manifest)
    }

    pub fn supports_replay(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        if !self.errors.is_empty() {
            return false;
        }
        if requested_coverage.is_some_and(|requested| !self.coverage.contains(requested)) {
            return false;
        }
        self.files
            .iter()
            .any(|file| file.errors.is_empty() && file.market_shape.supports(bar_type, candle_mode))
    }

    pub fn normalize_derived_fields(&mut self) {
        if self.available_bar_shapes.is_empty() {
            let mut shapes = Vec::new();
            for file in &self.files {
                if let Some(bar_type) = file.market_shape.bar_type
                    && !shapes.contains(&bar_type)
                {
                    shapes.push(bar_type);
                }
            }
            self.available_bar_shapes = shapes;
        }

        if self.available_chart_modes.is_empty() {
            let mut modes = Vec::new();
            for file in &self.files {
                if let Some(mode) = file.market_shape.chart_mode
                    && !modes.contains(&mode)
                {
                    modes.push(mode);
                }
            }
            self.available_chart_modes = modes;
        }

        if self.badges.is_empty() {
            self.badges = self.derived_badges();
        }
    }

    pub fn derived_badges(&self) -> Vec<String> {
        let mut badges = BTreeSet::new();
        badges.insert(self.source_kind.badge().to_string());
        for file in &self.files {
            badges.insert(file.source_kind.badge().to_string());
            if file.market_shape.bar_type.is_some_and(|bar_type| {
                matches!(
                    bar_type.kind(),
                    BarKind::Tick | BarKind::Volume | BarKind::Range
                )
            }) {
                badges.insert(file.market_shape.label().to_ascii_lowercase());
            }
            if let Some(chart_mode) = file.market_shape.chart_mode {
                badges.insert(chart_mode.label().to_ascii_lowercase());
            }
        }
        badges.into_iter().collect()
    }

    pub fn row_count_total(&self) -> u64 {
        self.files.iter().map(|file| file.row_count).sum()
    }

    pub fn available_shapes_label(&self) -> String {
        if self.available_bar_shapes.is_empty() {
            return "none listed".to_string();
        }
        self.available_bar_shapes
            .iter()
            .map(|bar_type| bar_type.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn available_chart_modes_label(&self) -> String {
        if self.available_chart_modes.is_empty() {
            return "none listed".to_string();
        }
        self.available_chart_modes
            .iter()
            .map(|mode| mode.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn badges_label(&self) -> String {
        if self.badges.is_empty() {
            return "none".to_string();
        }
        self.badges.join(", ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheServerBarRow {
    pub timestamp: DateTime<Utc>,
    pub ts_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
}

impl ReplayCacheServerBarRow {
    pub fn from_bar(bar: &Bar) -> Self {
        Self {
            timestamp: DateTime::<Utc>::from_timestamp_nanos(bar.ts_ns),
            ts_ns: bar.ts_ns,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplayCacheServerBarsWrite {
    pub cache_root: PathBuf,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub request_start: DateTime<Utc>,
    pub request_end: DateTime<Utc>,
    pub source_kind: ReplayCacheSourceKind,
    pub download_request: Value,
    pub bar_type: BarType,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub bars: Vec<Bar>,
    pub warnings: Vec<String>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCacheWriteOutcome {
    pub dataset_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub data_path: PathBuf,
    pub row_count: u64,
}

pub fn write_server_bars_jsonl_cache(
    write: ReplayCacheServerBarsWrite,
) -> Result<ReplayCacheWriteOutcome> {
    if write.source_kind != ReplayCacheSourceKind::ServerBars {
        bail!("JSONL server-bar cache writer only accepts server-bars source data");
    }

    let rows = normalize_server_bar_rows(write.bars);
    if rows.is_empty() {
        bail!("server-bar download returned no usable bars");
    }

    let dataset_dir = replay_cache_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract.symbol,
        write.request_start.date_naive(),
    );
    let relative_path =
        server_bars_relative_path(write.request_start, write.request_end, write.bar_type);
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut data = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut data, row)?;
        data.push(b'\n');
    }
    fs::write(&data_path, &data).with_context(|| format!("write {}", data_path.display()))?;

    let first_timestamp = rows
        .first()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let last_timestamp = rows
        .last()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let row_count = rows.len() as u64;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::ServerBars,
        format: ReplayCacheFileFormat::Jsonl,
        schema_version: Some(SERVER_BARS_SCHEMA_VERSION),
        compression: None,
        market_shape: ReplayCacheMarketShape {
            bar_type: Some(write.bar_type),
            chart_mode: None,
            session_template: write.session_template.clone(),
        },
        row_count,
        first_timestamp,
        last_timestamp,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: fnv1a64_hex(&data),
        }),
        warnings: write.warnings.clone(),
        errors: Vec::new(),
    };

    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.provider,
            env: write.env,
            instrument: write.instrument.clone(),
            contract: write.contract.clone(),
            display_name: replay_cache_display_name(
                &write.contract.symbol,
                write.request_start,
                write.request_end,
                write.bar_type,
            ),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: Vec::new(),
            notes: write.notes.clone(),
        }
    };

    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument;
    manifest.contract = write.contract;
    manifest.source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = write.warnings;
    manifest.errors.clear();
    manifest.notes = write.notes;
    manifest
        .files
        .retain(|file| file.relative_path != relative_path);
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.request_start);
    manifest.available_bar_shapes.clear();
    for file in &manifest.files {
        if let Some(bar_type) = file.market_shape.bar_type
            && !manifest.available_bar_shapes.contains(&bar_type)
        {
            manifest.available_bar_shapes.push(bar_type);
        }
    }
    manifest.available_chart_modes = if write.bar_type.supports_candle_mode() {
        vec![CandleMode::Standard, CandleMode::HeikinAshi]
    } else {
        vec![CandleMode::Standard]
    };
    manifest.badges = manifest.derived_badges();

    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;

    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path,
        data_path,
        row_count,
    })
}

pub fn replay_cache_dataset_dir(
    root: &Path,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument: &str,
    contract: &str,
    start_date: NaiveDate,
) -> PathBuf {
    root.join(provider.label().to_ascii_lowercase())
        .join(match env {
            TradingEnvironment::Sim => "sim",
            TradingEnvironment::Live => "live",
        })
        .join(safe_cache_segment(instrument))
        .join(safe_cache_segment(contract))
        .join(start_date.to_string())
}

pub fn server_bars_relative_path(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> PathBuf {
    PathBuf::from("server-bars").join(format!(
        "{}_to_{}_{}.jsonl",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type_file_label(bar_type)
    ))
}

pub fn normalize_server_bar_rows(bars: Vec<Bar>) -> Vec<ReplayCacheServerBarRow> {
    let mut rows: Vec<_> = bars
        .into_iter()
        .filter(|bar| {
            bar.ts_ns > 0
                && bar.open.is_finite()
                && bar.high.is_finite()
                && bar.low.is_finite()
                && bar.close.is_finite()
                && bar.volume.is_none_or(f64::is_finite)
        })
        .map(|bar| ReplayCacheServerBarRow::from_bar(&bar))
        .collect();
    rows.sort_by_key(|row| row.ts_ns);
    rows.dedup_by_key(|row| row.ts_ns);
    rows
}

fn manifest_coverage_from_files(
    files: &[ReplayCacheDataFile],
    fallback_start: DateTime<Utc>,
) -> ReplayCacheCoverage {
    let start = files
        .iter()
        .map(|file| file.first_timestamp)
        .min()
        .unwrap_or(fallback_start);
    let end = files
        .iter()
        .map(|file| file.last_timestamp)
        .max()
        .unwrap_or(start);
    ReplayCacheCoverage {
        start,
        end,
        trading_date: Some(start.date_naive()),
    }
}

fn replay_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> String {
    format!(
        "{} {} to {} {} server bars",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type.label()
    )
}

fn bar_type_file_label(bar_type: BarType) -> String {
    let kind = match bar_type.kind() {
        BarKind::Minute => "minute",
        BarKind::Second => "second",
        BarKind::Tick => "tick",
        BarKind::Volume => "volume",
        BarKind::Range => "range",
    };
    format!("{}{}", bar_type.value(), kind)
}

fn safe_cache_segment(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheDataset {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub manifest: ReplayCacheManifest,
}

impl ReplayCacheDataset {
    pub fn can_serve(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        self.manifest
            .supports_replay(bar_type, candle_mode, requested_coverage)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayCacheLibrary {
    pub root: PathBuf,
    pub datasets: Vec<ReplayCacheDataset>,
    pub warnings: Vec<String>,
}

impl ReplayCacheLibrary {
    pub fn scan(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let mut library = Self {
            root: root.clone(),
            datasets: Vec::new(),
            warnings: Vec::new(),
        };
        scan_manifest_paths(
            &root,
            &mut |path| match ReplayCacheManifest::from_path(path) {
                Ok(manifest) => library.datasets.push(ReplayCacheDataset {
                    dataset_dir: path.parent().unwrap_or(root.as_path()).to_path_buf(),
                    manifest_path: path.to_path_buf(),
                    manifest,
                }),
                Err(err) => library
                    .warnings
                    .push(format!("{}: {err:#}", path.display())),
            },
        );
        library.datasets.sort_by(|left, right| {
            right
                .manifest
                .coverage
                .end
                .cmp(&left.manifest.coverage.end)
                .then_with(|| left.manifest.display_name.cmp(&right.manifest.display_name))
        });
        library
    }

    pub fn first_serving(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets
            .iter()
            .find(|dataset| dataset.can_serve(bar_type, candle_mode, requested_coverage))
    }
}

fn scan_manifest_paths(root: &Path, on_manifest: &mut impl FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|value| value.to_str()) == Some(MANIFEST_FILE_NAME) {
            on_manifest(&path);
            continue;
        }
        if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            scan_manifest_paths(&path, on_manifest);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dt(raw: &str) -> DateTime<Utc> {
        raw.parse().expect("valid timestamp")
    }

    fn temp_cache_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("trader-cache-{name}-{nonce}"))
    }

    fn sample_manifest() -> ReplayCacheManifest {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: Some("Micro E-mini S&P 500".to_string()),
                exchange: Some("CME".to_string()),
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: NaiveDate::from_ymd_opt(2026, 9, 18),
            },
            display_name: "MESU6 2026-07-23 1m HA".to_string(),
            coverage: ReplayCacheCoverage {
                start: dt("2026-07-23T13:30:00Z"),
                end: dt("2026-07-23T20:00:00Z"),
                trading_date: NaiveDate::from_ymd_opt(2026, 7, 23),
            },
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({
                "md": "getChart",
                "chartDescription": BarType::minute(1).chart_description(),
            }),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            files: vec![ReplayCacheDataFile {
                relative_path: PathBuf::from("server-bars/1m-heikin.parquet"),
                source_kind: ReplayCacheSourceKind::ServerBars,
                format: ReplayCacheFileFormat::Parquet,
                schema_version: Some(1),
                compression: Some("zstd".to_string()),
                market_shape: ReplayCacheMarketShape {
                    bar_type: Some(BarType::minute(1)),
                    chart_mode: Some(CandleMode::HeikinAshi),
                    session_template: Some("Globex".to_string()),
                },
                row_count: 390,
                first_timestamp: dt("2026-07-23T13:30:00Z"),
                last_timestamp: dt("2026-07-23T20:00:00Z"),
                data_hash: Some(ReplayCacheDataHash {
                    algorithm: "sha256".to_string(),
                    value: "abc123".to_string(),
                }),
                warnings: Vec::new(),
                errors: Vec::new(),
            }],
            app: Some(ReplayCacheAppMetadata {
                app_version: Some("0.1.0".to_string()),
                git_commit: Some("test".to_string()),
                generated_at: Some(dt("2026-07-24T00:00:00Z")),
            }),
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: vec!["regression".to_string()],
            notes: Some("sample".to_string()),
        }
    }

    fn bar(raw: &str, open: f64) -> Bar {
        Bar {
            ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            open,
            high: open + 1.0,
            low: open - 1.0,
            close: open + 0.5,
            volume: Some(10.0),
        }
    }

    #[test]
    fn manifest_parsing_derives_library_badges_and_shapes() {
        let manifest = sample_manifest();
        let raw = serde_json::to_string(&manifest).expect("serialize manifest");
        let mut parsed: ReplayCacheManifest = serde_json::from_str(&raw).expect("parse manifest");
        parsed.normalize_derived_fields();

        assert_eq!(parsed.provider, BrokerKind::Tradovate);
        assert_eq!(parsed.row_count_total(), 390);
        assert_eq!(parsed.available_bar_shapes, vec![BarType::minute(1)]);
        assert_eq!(parsed.available_chart_modes, vec![CandleMode::HeikinAshi]);
        assert!(parsed.badges.contains(&"server-bars".to_string()));
        assert!(parsed.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
        assert!(!parsed.supports_replay(BarType::minute(5), CandleMode::HeikinAshi, None));
    }

    #[test]
    fn manifest_errors_make_dataset_unservable() {
        let mut manifest = sample_manifest();
        manifest.errors.push("partial download".to_string());

        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));

        let mut manifest = sample_manifest();
        manifest.files[0]
            .errors
            .push("missing data page".to_string());

        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
    }

    #[test]
    fn cache_library_discovers_manifests_without_reading_data_files() {
        let root = temp_cache_dir("scan");
        let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
        fs::create_dir_all(dataset_dir.join("server-bars")).expect("create dirs");
        fs::write(
            dataset_dir.join(MANIFEST_FILE_NAME),
            serde_json::to_vec_pretty(&sample_manifest()).expect("serialize manifest"),
        )
        .expect("write manifest");
        fs::write(
            dataset_dir.join("server-bars/1m-heikin.parquet"),
            b"not parquet",
        )
        .expect("write ignored data file");

        let library = ReplayCacheLibrary::scan(&root);

        assert_eq!(library.datasets.len(), 1);
        assert!(library.warnings.is_empty());
        assert_eq!(
            library.datasets[0].manifest.display_name,
            "MESU6 2026-07-23 1m HA"
        );
        assert!(
            library
                .first_serving(BarType::minute(1), CandleMode::HeikinAshi, None)
                .is_some()
        );
    }

    #[test]
    fn server_bar_cache_path_shape_is_deterministic_and_sanitized() {
        let root = PathBuf::from("/tmp/cache-root");
        let dir = replay_cache_dataset_dir(
            &root,
            BrokerKind::Tradovate,
            TradingEnvironment::Sim,
            "ME S",
            "MES/U6",
            NaiveDate::from_ymd_opt(2026, 7, 23).expect("date"),
        );
        let relative = server_bars_relative_path(
            dt("2026-07-23T00:00:00Z"),
            dt("2026-07-25T00:00:00Z"),
            BarType::volume(6500),
        );

        assert_eq!(
            dir,
            PathBuf::from("/tmp/cache-root/tradovate/sim/ME_S/MESU6/2026-07-23")
        );
        assert_eq!(
            relative,
            PathBuf::from("server-bars/2026-07-23_to_2026-07-25_6500volume.jsonl")
        );
    }

    #[test]
    fn normalize_server_bars_filters_sorts_and_deduplicates_rows() {
        let rows = normalize_server_bar_rows(vec![
            bar("2026-07-23T00:01:00Z", 2.0),
            Bar {
                open: f64::NAN,
                ..bar("2026-07-23T00:02:00Z", 3.0)
            },
            bar("2026-07-23T00:00:00Z", 1.0),
            bar("2026-07-23T00:01:00Z", 4.0),
        ]);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].open, 1.0);
        assert_eq!(rows[1].open, 2.0);
    }

    #[test]
    fn write_server_bars_jsonl_cache_writes_manifest_and_data_file() {
        let root = temp_cache_dir("write");
        let outcome = write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
            cache_root: root.clone(),
            provider: BrokerKind::Tradovate,
            env: TradingEnvironment::Sim,
            instrument: ReplayCacheInstrument {
                symbol: "MES".to_string(),
                name: None,
                exchange: None,
            },
            contract: ReplayCacheContract {
                symbol: "MESU6".to_string(),
                id: Some(123),
                expiration: None,
            },
            request_start: dt("2026-07-23T00:00:00Z"),
            request_end: dt("2026-07-24T00:00:00Z"),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: json!({
                "md": "getChart",
                "chartDescription": BarType::minute(1).chart_description()
            }),
            bar_type: BarType::minute(1),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            session_template: Some("Globex".to_string()),
            bars: vec![bar("2026-07-23T00:00:00Z", 1.0)],
            warnings: vec!["synthetic fixture".to_string()],
            notes: Some("test write".to_string()),
        })
        .expect("write cache");

        assert_eq!(outcome.row_count, 1);
        assert!(outcome.data_path.exists());
        assert!(outcome.manifest_path.exists());

        let data = fs::read_to_string(&outcome.data_path).expect("read data");
        assert_eq!(data.lines().count(), 1);
        assert!(data.contains("\"timestamp\":\"2026-07-23T00:00:00Z\""));

        let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
        assert_eq!(manifest.provider, BrokerKind::Tradovate);
        assert_eq!(manifest.source_kind, ReplayCacheSourceKind::ServerBars);
        assert_eq!(manifest.files[0].format, ReplayCacheFileFormat::Jsonl);
        assert_eq!(manifest.files[0].row_count, 1);
        assert_eq!(
            manifest.files[0]
                .data_hash
                .as_ref()
                .map(|hash| hash.algorithm.as_str()),
            Some("fnv1a64")
        );
        assert_eq!(
            manifest.available_chart_modes,
            vec![CandleMode::Standard, CandleMode::HeikinAshi]
        );
        assert!(manifest.supports_replay(BarType::minute(1), CandleMode::HeikinAshi, None));
    }
}

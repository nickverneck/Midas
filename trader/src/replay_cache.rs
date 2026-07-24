use crate::broker::{Bar, BarKind, BarType, BrokerKind, CandleMode};
use crate::config::TradingEnvironment;
use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub const MANIFEST_FILE_NAME: &str = "manifest.json";
pub const MANIFEST_VERSION: u32 = 1;
pub const SERVER_BARS_SCHEMA_VERSION: u32 = 1;
pub const RAW_TICKS_SCHEMA_VERSION: u32 = 1;
const PARQUET_COMPRESSION_LABEL: &str = "snappy";

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
            .any(|file| replay_file_can_serve(file, bar_type, candle_mode))
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

fn replay_file_can_serve(
    file: &ReplayCacheDataFile,
    bar_type: BarType,
    candle_mode: CandleMode,
) -> bool {
    file.errors.is_empty()
        && matches!(
            file.source_kind,
            ReplayCacheSourceKind::ServerBars
                | ReplayCacheSourceKind::DerivedBars
                | ReplayCacheSourceKind::LocalText
        )
        && file.market_shape.supports(bar_type, candle_mode)
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplayCacheRawTickRow {
    pub timestamp: DateTime<Utc>,
    pub ts_ns: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tick_id: Option<i64>,
    pub price: f64,
    pub size: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_price: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chart_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trade_date: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_ts_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_base_price_ticks: Option<i64>,
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

#[derive(Debug, Clone)]
pub struct ReplayCacheRawTicksWrite {
    pub cache_root: PathBuf,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub request_start: DateTime<Utc>,
    pub request_end: DateTime<Utc>,
    pub download_request: Value,
    pub tick_specs: ReplayCacheTickSpecs,
    pub session_template: Option<String>,
    pub ticks: Vec<ReplayCacheRawTickRow>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheRawTicksNormalizeOutcome {
    pub rows: Vec<ReplayCacheRawTickRow>,
    pub duplicate_tick_ids: usize,
    pub dropped_rows: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheResolvedServerBarsFile {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheLoadedServerBars {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub data_path: PathBuf,
    pub manifest: ReplayCacheManifest,
    pub file: ReplayCacheDataFile,
    pub bars: Vec<Bar>,
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

pub fn write_raw_ticks_parquet_cache(
    write: ReplayCacheRawTicksWrite,
) -> Result<ReplayCacheWriteOutcome> {
    let normalized = normalize_raw_tick_rows(write.ticks);
    if normalized.rows.is_empty() {
        bail!("raw tick download returned no usable ticks");
    }

    let mut warnings = write.warnings;
    if normalized.duplicate_tick_ids > 0 {
        warnings.push(format!(
            "Dropped {} duplicate raw tick id(s) while normalizing cache rows.",
            normalized.duplicate_tick_ids
        ));
    }
    if normalized.dropped_rows > 0 {
        warnings.push(format!(
            "Dropped {} malformed raw tick row(s) while normalizing cache rows.",
            normalized.dropped_rows
        ));
    }

    let dataset_dir = replay_cache_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract.symbol,
        write.request_start.date_naive(),
    );
    let relative_path = raw_ticks_relative_path(write.request_start, write.request_end);
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_raw_ticks_parquet_file(&data_path, &normalized.rows)?;

    let data = fs::read(&data_path).with_context(|| format!("read {}", data_path.display()))?;
    let first_timestamp = normalized
        .rows
        .first()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let last_timestamp = normalized
        .rows
        .last()
        .map(|row| row.timestamp)
        .expect("rows are non-empty");
    let row_count = normalized.rows.len() as u64;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::RawTicks,
        format: ReplayCacheFileFormat::Parquet,
        schema_version: Some(RAW_TICKS_SCHEMA_VERSION),
        compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
        market_shape: ReplayCacheMarketShape {
            bar_type: None,
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
        warnings: warnings.clone(),
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
            display_name: raw_ticks_cache_display_name(
                &write.contract.symbol,
                write.request_start,
                write.request_end,
            ),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            source_kind: ReplayCacheSourceKind::RawTicks,
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
    manifest.source_kind = ReplayCacheSourceKind::RawTicks;
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = warnings;
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
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect::<Vec<_>>();
    manifest.available_bar_shapes.sort_by_key(|bar_type| {
        (
            match bar_type.kind() {
                BarKind::Minute => 0,
                BarKind::Second => 1,
                BarKind::Tick => 2,
                BarKind::Volume => 3,
                BarKind::Range => 4,
            },
            bar_type.value(),
        )
    });
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.chart_mode)
        .collect::<Vec<_>>();
    manifest.available_chart_modes.dedup();
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

pub fn raw_ticks_relative_path(start: DateTime<Utc>, end: DateTime<Utc>) -> PathBuf {
    PathBuf::from("raw-ticks").join(format!(
        "{}_to_{}_ticks.parquet",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
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

pub fn normalize_raw_tick_rows(
    ticks: Vec<ReplayCacheRawTickRow>,
) -> ReplayCacheRawTicksNormalizeOutcome {
    let mut dropped_rows = 0usize;
    let mut rows = Vec::with_capacity(ticks.len());
    for tick in ticks {
        if validate_raw_tick_row(&tick).is_ok() {
            rows.push(tick);
        } else {
            dropped_rows = dropped_rows.saturating_add(1);
        }
    }

    rows.sort_by(|left, right| {
        left.ts_ns.cmp(&right.ts_ns).then_with(|| {
            left.tick_id
                .unwrap_or(i64::MAX)
                .cmp(&right.tick_id.unwrap_or(i64::MAX))
        })
    });

    let mut seen_tick_ids = BTreeSet::new();
    let before_dedup = rows.len();
    rows.retain(|row| {
        row.tick_id
            .is_none_or(|tick_id| seen_tick_ids.insert(tick_id))
    });
    let duplicate_tick_ids = before_dedup.saturating_sub(rows.len());

    ReplayCacheRawTicksNormalizeOutcome {
        rows,
        duplicate_tick_ids,
        dropped_rows,
    }
}

pub fn read_server_bars_jsonl_file(path: &Path) -> Result<Vec<Bar>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut bars = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line =
            line.with_context(|| format!("read line {line_number} in {}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: ReplayCacheServerBarRow = serde_json::from_str(trimmed)
            .with_context(|| format!("parse JSONL bar line {line_number} in {}", path.display()))?;
        bars.push(
            server_bar_row_to_bar(row)
                .with_context(|| format!("validate JSONL bar line {line_number}"))?,
        );
    }

    if bars.is_empty() {
        bail!("server-bar cache file {} contained no bars", path.display());
    }
    normalize_read_server_bars(bars)
}

pub fn write_raw_ticks_parquet_file(path: &Path, rows: &[ReplayCacheRawTickRow]) -> Result<()> {
    if rows.is_empty() {
        bail!("raw tick parquet writer requires at least one row");
    }

    let schema = raw_ticks_parquet_schema();
    let timestamp_values = rows
        .iter()
        .map(|row| row.timestamp.to_rfc3339())
        .collect::<Vec<_>>();
    let packet_sources = rows
        .iter()
        .map(|row| row.packet_source.as_deref())
        .collect::<Vec<_>>();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(timestamp_values)),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.ts_ns).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.size).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bid_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.bid_size).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.ask_price).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.ask_size).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.chart_id).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|row| row.trade_date).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(packet_sources)),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.packet_base_ts_ms)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| row.packet_base_price_ticks)
                .collect::<Vec<_>>(),
        )),
    ];
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

#[allow(dead_code)]
pub fn read_raw_ticks_parquet_file(path: &Path) -> Result<Vec<ReplayCacheRawTickRow>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?;
    let reader = builder
        .build()
        .with_context(|| format!("build parquet reader {}", path.display()))?;
    let mut rows = Vec::new();

    for batch in reader {
        let batch = batch.with_context(|| format!("read parquet batch {}", path.display()))?;
        let timestamps = parquet_column::<StringArray>(&batch, 0, "timestamp")?;
        let ts_ns = parquet_column::<Int64Array>(&batch, 1, "ts_ns")?;
        let tick_ids = parquet_column::<Int64Array>(&batch, 2, "tick_id")?;
        let prices = parquet_column::<Float64Array>(&batch, 3, "price")?;
        let sizes = parquet_column::<Float64Array>(&batch, 4, "size")?;
        let bid_prices = parquet_column::<Float64Array>(&batch, 5, "bid_price")?;
        let bid_sizes = parquet_column::<Float64Array>(&batch, 6, "bid_size")?;
        let ask_prices = parquet_column::<Float64Array>(&batch, 7, "ask_price")?;
        let ask_sizes = parquet_column::<Float64Array>(&batch, 8, "ask_size")?;
        let chart_ids = parquet_column::<Int64Array>(&batch, 9, "chart_id")?;
        let trade_dates = parquet_column::<Int32Array>(&batch, 10, "trade_date")?;
        let packet_sources = parquet_column::<StringArray>(&batch, 11, "packet_source")?;
        let packet_base_ts_ms = parquet_column::<Int64Array>(&batch, 12, "packet_base_ts_ms")?;
        let packet_base_price_ticks =
            parquet_column::<Int64Array>(&batch, 13, "packet_base_price_ticks")?;

        for row_index in 0..batch.num_rows() {
            let timestamp = DateTime::parse_from_rfc3339(timestamps.value(row_index))
                .with_context(|| format!("parse raw tick timestamp row {row_index}"))?
                .with_timezone(&Utc);
            rows.push(ReplayCacheRawTickRow {
                timestamp,
                ts_ns: ts_ns.value(row_index),
                tick_id: optional_i64(tick_ids, row_index),
                price: prices.value(row_index),
                size: sizes.value(row_index),
                bid_price: optional_f64(bid_prices, row_index),
                bid_size: optional_f64(bid_sizes, row_index),
                ask_price: optional_f64(ask_prices, row_index),
                ask_size: optional_f64(ask_sizes, row_index),
                chart_id: optional_i64(chart_ids, row_index),
                trade_date: optional_i32(trade_dates, row_index),
                packet_source: optional_string(packet_sources, row_index),
                packet_base_ts_ms: optional_i64(packet_base_ts_ms, row_index),
                packet_base_price_ticks: optional_i64(packet_base_price_ticks, row_index),
            });
        }
    }

    let normalized = normalize_raw_tick_rows(rows);
    if normalized.rows.is_empty() {
        bail!(
            "raw tick parquet file {} contained no usable ticks",
            path.display()
        );
    }
    Ok(normalized.rows)
}

pub fn load_server_bars_jsonl_cache_file(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
) -> Result<ReplayCacheLoadedServerBars> {
    let resolved =
        dataset.resolve_server_bars_jsonl_file(bar_type, candle_mode, requested_coverage)?;
    let bars = read_server_bars_jsonl_file(&resolved.data_path)?;
    validate_loaded_server_bars_metadata(&resolved, &bars)?;

    Ok(ReplayCacheLoadedServerBars {
        manifest_path: resolved.manifest_path,
        dataset_dir: resolved.dataset_dir,
        data_path: resolved.data_path,
        manifest: resolved.manifest,
        file: resolved.file,
        bars,
    })
}

fn server_bar_row_to_bar(row: ReplayCacheServerBarRow) -> Result<Bar> {
    if row.ts_ns <= 0 {
        bail!("bar timestamp nanoseconds must be positive");
    }
    let timestamp_ns = row
        .timestamp
        .timestamp_nanos_opt()
        .context("bar timestamp is outside supported nanosecond range")?;
    if timestamp_ns != row.ts_ns {
        bail!(
            "bar timestamp {} disagrees with ts_ns {}",
            row.timestamp,
            row.ts_ns
        );
    }
    let bar = Bar {
        ts_ns: row.ts_ns,
        open: row.open,
        high: row.high,
        low: row.low,
        close: row.close,
        volume: row.volume,
    };
    validate_server_bar(&bar)?;
    Ok(bar)
}

fn normalize_read_server_bars(mut bars: Vec<Bar>) -> Result<Vec<Bar>> {
    bars.sort_by_key(|bar| bar.ts_ns);
    bars.dedup_by_key(|bar| bar.ts_ns);
    for bar in &bars {
        validate_server_bar(bar)?;
    }
    Ok(bars)
}

fn validate_server_bar(bar: &Bar) -> Result<()> {
    if bar.ts_ns <= 0 {
        bail!("bar timestamp nanoseconds must be positive");
    }
    if !(bar.open.is_finite()
        && bar.high.is_finite()
        && bar.low.is_finite()
        && bar.close.is_finite()
        && bar.volume.is_none_or(f64::is_finite))
    {
        bail!("bar contains a non-finite price or volume");
    }
    if bar.high < bar.low {
        bail!("bar high is below low");
    }
    const EPSILON: f64 = 1e-9;
    if bar.open > bar.high + EPSILON
        || bar.close > bar.high + EPSILON
        || bar.open < bar.low - EPSILON
        || bar.close < bar.low - EPSILON
    {
        bail!("bar OHLC values fall outside high/low range");
    }
    Ok(())
}

fn validate_raw_tick_row(row: &ReplayCacheRawTickRow) -> Result<()> {
    if row.ts_ns <= 0 {
        bail!("raw tick timestamp nanoseconds must be positive");
    }
    let timestamp_ns = row
        .timestamp
        .timestamp_nanos_opt()
        .context("raw tick timestamp is outside supported nanosecond range")?;
    if timestamp_ns != row.ts_ns {
        bail!(
            "raw tick timestamp {} disagrees with ts_ns {}",
            row.timestamp,
            row.ts_ns
        );
    }
    if !(row.price.is_finite() && row.price > 0.0 && row.size.is_finite() && row.size > 0.0) {
        bail!("raw tick price and size must be finite positive values");
    }
    validate_optional_positive(row.bid_price, "bid price")?;
    validate_optional_nonnegative(row.bid_size, "bid size")?;
    validate_optional_positive(row.ask_price, "ask price")?;
    validate_optional_nonnegative(row.ask_size, "ask size")?;
    Ok(())
}

fn validate_optional_positive(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        bail!("raw tick {label} must be finite and positive");
    }
    Ok(())
}

fn validate_optional_nonnegative(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        bail!("raw tick {label} must be finite and non-negative");
    }
    Ok(())
}

fn raw_ticks_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("tick_id", DataType::Int64, true),
        Field::new("price", DataType::Float64, false),
        Field::new("size", DataType::Float64, false),
        Field::new("bid_price", DataType::Float64, true),
        Field::new("bid_size", DataType::Float64, true),
        Field::new("ask_price", DataType::Float64, true),
        Field::new("ask_size", DataType::Float64, true),
        Field::new("chart_id", DataType::Int64, true),
        Field::new("trade_date", DataType::Int32, true),
        Field::new("packet_source", DataType::Utf8, true),
        Field::new("packet_base_ts_ms", DataType::Int64, true),
        Field::new("packet_base_price_ticks", DataType::Int64, true),
    ]))
}

#[allow(dead_code)]
fn parquet_column<'a, T: 'static>(
    batch: &'a RecordBatch,
    index: usize,
    name: &str,
) -> Result<&'a T> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("parquet column {name} had an unexpected type"))
}

#[allow(dead_code)]
fn optional_i64(array: &Int64Array, index: usize) -> Option<i64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_i32(array: &Int32Array, index: usize) -> Option<i32> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_f64(array: &Float64Array, index: usize) -> Option<f64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
fn optional_string(array: &StringArray, index: usize) -> Option<String> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index).to_string())
    }
}

fn validate_loaded_server_bars_metadata(
    resolved: &ReplayCacheResolvedServerBarsFile,
    bars: &[Bar],
) -> Result<()> {
    if bars.is_empty() {
        bail!(
            "server-bar cache file {} contained no bars",
            resolved.data_path.display()
        );
    }
    if bars.len() as u64 != resolved.file.row_count {
        bail!(
            "server-bar cache row count mismatch for {}: manifest={} actual={}",
            resolved.data_path.display(),
            resolved.file.row_count,
            bars.len()
        );
    }
    let first_timestamp = DateTime::<Utc>::from_timestamp_nanos(bars[0].ts_ns);
    let last_timestamp =
        DateTime::<Utc>::from_timestamp_nanos(bars.last().expect("bars are non-empty").ts_ns);
    if first_timestamp != resolved.file.first_timestamp
        || last_timestamp != resolved.file.last_timestamp
    {
        bail!(
            "server-bar cache timestamp range mismatch for {}: manifest={}..{} actual={}..{}",
            resolved.data_path.display(),
            resolved.file.first_timestamp,
            resolved.file.last_timestamp,
            first_timestamp,
            last_timestamp
        );
    }
    Ok(())
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

fn raw_ticks_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> String {
    format!(
        "{} raw ticks {} to {}",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
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

    pub fn server_bars_jsonl_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty() {
            return None;
        }
        if requested_coverage.is_some_and(|requested| !self.manifest.coverage.contains(requested)) {
            return None;
        }
        self.manifest.files.iter().find(|file| {
            file.errors.is_empty()
                && file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.format == ReplayCacheFileFormat::Jsonl
                && file.market_shape.chart_mode != Some(CandleMode::HeikinAshi)
                && file
                    .schema_version
                    .is_none_or(|version| version == SERVER_BARS_SCHEMA_VERSION)
                && file.market_shape.supports(bar_type, candle_mode)
        })
    }

    pub fn resolve_server_bars_jsonl_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = self
            .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
            .with_context(|| {
                format!(
                    "no cached JSONL server bars for {} in {}",
                    bar_type.mode_label(candle_mode),
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
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

    #[allow(dead_code)]
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

    pub fn first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets.iter().find(|dataset| {
            dataset.can_serve(bar_type, candle_mode, requested_coverage)
                && dataset
                    .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
                    .is_some()
        })
    }

    pub fn load_first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedServerBars>> {
        let Some(dataset) = self.first_server_bars_jsonl(bar_type, candle_mode, requested_coverage)
        else {
            return Ok(None);
        };
        load_server_bars_jsonl_cache_file(dataset, bar_type, candle_mode, requested_coverage)
            .map(Some)
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

fn resolve_cache_data_path(dataset_dir: &Path, relative_path: &Path) -> Result<PathBuf> {
    if relative_path.is_absolute() {
        bail!(
            "cache data path {} must be relative to its manifest",
            relative_path.display()
        );
    }
    for component in relative_path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!(
                "cache data path {} is not a safe manifest-relative path",
                relative_path.display()
            ),
        }
    }
    let data_path = dataset_dir.join(relative_path);
    if !data_path.is_file() {
        bail!("cache data file {} was not found", data_path.display());
    }
    Ok(data_path)
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

    fn bar_row_json(raw: &str, open: f64) -> String {
        json!({
            "timestamp": raw,
            "ts_ns": dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            "open": open,
            "high": open + 1.0,
            "low": open - 1.0,
            "close": open + 0.5,
            "volume": 10.0
        })
        .to_string()
    }

    fn raw_tick(raw: &str, tick_id: Option<i64>, price: f64) -> ReplayCacheRawTickRow {
        ReplayCacheRawTickRow {
            timestamp: dt(raw),
            ts_ns: dt(raw).timestamp_nanos_opt().expect("timestamp ns"),
            tick_id,
            price,
            size: 2.0,
            bid_price: Some(price - 0.25),
            bid_size: Some(10.0),
            ask_price: Some(price),
            ask_size: Some(12.0),
            chart_id: Some(77),
            trade_date: Some(20260723),
            packet_source: Some("db".to_string()),
            packet_base_ts_ms: Some(1_785_000_000_000),
            packet_base_price_ticks: Some(29_700),
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
    fn normalize_raw_ticks_sorts_deduplicates_ids_and_drops_bad_rows() {
        let mut bad = raw_tick("2026-07-23T00:02:00Z", Some(3), 100.0);
        bad.size = f64::NAN;
        let normalized = normalize_raw_tick_rows(vec![
            raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
            bad,
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
            raw_tick("2026-07-23T00:01:01Z", Some(2), 102.0),
        ]);

        assert_eq!(normalized.rows.len(), 2);
        assert_eq!(normalized.duplicate_tick_ids, 1);
        assert_eq!(normalized.dropped_rows, 1);
        assert_eq!(normalized.rows[0].tick_id, Some(1));
        assert_eq!(normalized.rows[1].tick_id, Some(2));
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

        let library = ReplayCacheLibrary::scan(root);
        let loaded = library
            .load_first_server_bars_jsonl(BarType::minute(1), CandleMode::HeikinAshi, None)
            .expect("load first matching cached server bars")
            .expect("matching cached server bars");
        assert_eq!(loaded.bars.len(), 1);
        assert_eq!(loaded.bars[0].close, 1.5);
        assert_eq!(
            loaded.file.relative_path,
            PathBuf::from("server-bars/2026-07-23_to_2026-07-24_1minute.jsonl")
        );
    }

    #[test]
    fn raw_tick_parquet_file_round_trips_rows() {
        let root = temp_cache_dir("raw-parquet-round-trip");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("ticks.parquet");
        let rows = vec![
            raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
            raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
        ];

        write_raw_ticks_parquet_file(&path, &rows).expect("write parquet");
        let loaded = read_raw_ticks_parquet_file(&path).expect("read parquet");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].tick_id, Some(1));
        assert_eq!(loaded[0].price, 100.0);
        assert_eq!(loaded[0].bid_price, Some(99.75));
        assert_eq!(loaded[1].packet_source.as_deref(), Some("db"));
    }

    #[test]
    fn write_raw_ticks_parquet_cache_writes_manifest_without_direct_replay_support() {
        let root = temp_cache_dir("raw-parquet-cache");
        let outcome = write_raw_ticks_parquet_cache(ReplayCacheRawTicksWrite {
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
            download_request: json!({
                "md": "getChart",
                "chartDescription": {
                    "underlyingType": "Tick",
                    "elementSize": 1,
                    "elementSizeUnit": "UnderlyingUnits"
                }
            }),
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            session_template: Some("Globex".to_string()),
            ticks: vec![
                raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
                raw_tick("2026-07-23T00:00:01Z", Some(1), 100.25),
            ],
            warnings: Vec::new(),
            notes: Some("raw tick test".to_string()),
        })
        .expect("write raw tick cache");

        assert_eq!(outcome.row_count, 1);
        assert!(outcome.data_path.exists());
        assert_eq!(
            outcome
                .data_path
                .extension()
                .and_then(|value| value.to_str()),
            Some("parquet")
        );

        let manifest = ReplayCacheManifest::from_path(&outcome.manifest_path).expect("manifest");
        assert_eq!(manifest.source_kind, ReplayCacheSourceKind::RawTicks);
        assert_eq!(manifest.files[0].format, ReplayCacheFileFormat::Parquet);
        assert_eq!(
            manifest.files[0].compression.as_deref(),
            Some(PARQUET_COMPRESSION_LABEL)
        );
        assert!(manifest.badges.contains(&"raw-ticks".to_string()));
        assert!(
            manifest
                .warnings
                .iter()
                .any(|warning| warning.contains("duplicate raw tick id"))
        );
        assert!(!manifest.supports_replay(BarType::minute(1), CandleMode::Standard, None));

        let loaded_ticks = read_raw_ticks_parquet_file(&outcome.data_path).expect("read ticks");
        assert_eq!(loaded_ticks.len(), 1);
        assert_eq!(loaded_ticks[0].tick_id, Some(1));
    }

    #[test]
    fn read_server_bars_jsonl_file_parses_sorts_and_deduplicates_rows() {
        let root = temp_cache_dir("read-jsonl");
        fs::create_dir_all(&root).expect("create temp dir");
        let path = root.join("bars.jsonl");
        fs::write(
            &path,
            [
                bar_row_json("2026-07-23T00:01:00Z", 2.0),
                bar_row_json("2026-07-23T00:00:00Z", 1.0),
                bar_row_json("2026-07-23T00:01:00Z", 2.0),
            ]
            .join("\n"),
        )
        .expect("write jsonl");

        let bars = read_server_bars_jsonl_file(&path).expect("read bars");

        assert_eq!(bars.len(), 2);
        assert!(bars[0].ts_ns < bars[1].ts_ns);
        assert_eq!(bars[0].open, 1.0);
        assert_eq!(bars[1].open, 2.0);
    }

    #[test]
    fn read_server_bars_jsonl_file_rejects_invalid_rows() {
        let root = temp_cache_dir("read-bad-jsonl");
        fs::create_dir_all(&root).expect("create temp dir");
        let bad_ohlc = root.join("bad-ohlc.jsonl");
        fs::write(
            &bad_ohlc,
            json!({
                "timestamp": "2026-07-23T00:00:00Z",
                "ts_ns": dt("2026-07-23T00:00:00Z").timestamp_nanos_opt().expect("timestamp ns"),
                "open": 10.0,
                "high": 9.0,
                "low": 8.0,
                "close": 8.5
            })
            .to_string(),
        )
        .expect("write bad ohlc");
        let err = read_server_bars_jsonl_file(&bad_ohlc).expect_err("bad ohlc should fail");
        assert!(err.to_string().contains("validate JSONL bar line 1"));

        let bad_timestamp = root.join("bad-timestamp.jsonl");
        fs::write(
            &bad_timestamp,
            json!({
                "timestamp": "2026-07-23T00:00:00Z",
                "ts_ns": dt("2026-07-23T00:01:00Z").timestamp_nanos_opt().expect("timestamp ns"),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5
            })
            .to_string(),
        )
        .expect("write bad timestamp");
        let err = read_server_bars_jsonl_file(&bad_timestamp)
            .expect_err("timestamp mismatch should fail");
        assert!(err.to_string().contains("validate JSONL bar line 1"));
    }

    #[test]
    fn server_bars_jsonl_resolver_rejects_path_escape() {
        let root = temp_cache_dir("resolve-escape");
        let mut manifest = sample_manifest();
        manifest.files[0].relative_path = PathBuf::from("../escape.jsonl");
        manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
        manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
        manifest.files[0].market_shape.chart_mode = None;
        let dataset = ReplayCacheDataset {
            manifest_path: root.join(MANIFEST_FILE_NAME),
            dataset_dir: root,
            manifest,
        };

        let err = dataset
            .resolve_server_bars_jsonl_file(BarType::minute(1), CandleMode::Standard, None)
            .expect_err("escaping relative path should fail");

        assert!(
            err.to_string()
                .contains("not a safe manifest-relative path")
        );
    }

    #[test]
    fn load_server_bars_jsonl_cache_file_validates_manifest_metadata() {
        let root = temp_cache_dir("metadata-mismatch");
        let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
        let data_dir = dataset_dir.join("server-bars");
        fs::create_dir_all(&data_dir).expect("create data dir");
        let relative_path = PathBuf::from("server-bars/bars.jsonl");
        fs::write(
            dataset_dir.join(&relative_path),
            bar_row_json("2026-07-23T00:00:00Z", 1.0),
        )
        .expect("write jsonl");

        let mut manifest = sample_manifest();
        manifest.files[0].relative_path = relative_path;
        manifest.files[0].format = ReplayCacheFileFormat::Jsonl;
        manifest.files[0].source_kind = ReplayCacheSourceKind::ServerBars;
        manifest.files[0].market_shape.chart_mode = None;
        manifest.files[0].row_count = 2;
        manifest.files[0].first_timestamp = dt("2026-07-23T00:00:00Z");
        manifest.files[0].last_timestamp = dt("2026-07-23T00:00:00Z");
        let dataset = ReplayCacheDataset {
            manifest_path: dataset_dir.join(MANIFEST_FILE_NAME),
            dataset_dir,
            manifest,
        };

        let err = load_server_bars_jsonl_cache_file(
            &dataset,
            BarType::minute(1),
            CandleMode::HeikinAshi,
            None,
        )
        .expect_err("row count mismatch should fail");

        assert!(err.to_string().contains("row count mismatch"));
    }
}

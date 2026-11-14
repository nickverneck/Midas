//! Cache-backed chart API adapter for future web clients.
//!
//! This is a data adapter, not an HTTP server. It serves only completed stored
//! server bars and labels their provenance so callers cannot mistake them for
//! a live feed. Raw-tick aggregation remains owned by replay.

use super::*;
use crate::broker::transform_bars_for_candle_mode;
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CHART_API_MAX_PAGE_SIZE: usize = 5_000;

/// Selects the stored input used to produce a chart series. Raw ticks are
/// aggregated locally from the cache; this does not create a trade feed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChartApiInput {
    ServerBars,
    RawTicks,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChartApiSeries {
    pub input: ChartApiInput,
    pub bar_type: BarType,
    pub candle_mode: CandleMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChartApiDataset {
    /// Stable opaque identifier derived from the manifest path.
    pub id: String,
    pub display_name: String,
    pub provider: BrokerKind,
    pub environment: TradingEnvironment,
    pub instrument: ReplayCacheInstrument,
    pub contract: ReplayCacheContract,
    pub coverage: ReplayCacheCoverage,
    /// Always `stored_cache`; this adapter never reports a live subscription.
    pub source_status: String,
    pub series: Vec<ChartApiSeries>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartApiBar {
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartApiPage {
    pub dataset_id: String,
    pub series: ChartApiSeries,
    pub source_status: String,
    pub offset: usize,
    pub limit: usize,
    pub total: usize,
    pub bars: Vec<ChartApiBar>,
}

/// Scan the replay cache and describe chart series that can actually be read.
pub fn chart_api_catalog(cache_root: impl Into<PathBuf>) -> ChartApiCatalog {
    let library = ReplayCacheLibrary::scan(cache_root);
    let datasets = library
        .datasets
        .iter()
        .filter_map(|dataset| chart_api_dataset(dataset))
        .collect();
    ChartApiCatalog {
        datasets,
        warnings: library.warnings,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChartApiCatalog {
    pub datasets: Vec<ChartApiDataset>,
    pub warnings: Vec<String>,
}

fn chart_api_dataset(dataset: &ReplayCacheDataset) -> Option<ChartApiDataset> {
    let mut series = Vec::new();
    if !dataset.manifest.errors.is_empty() {
        return None;
    }
    for file in &dataset.manifest.files {
        if file.source_kind != ReplayCacheSourceKind::ServerBars {
            continue;
        }
        let Some(bar_type) = file.market_shape.bar_type else {
            continue;
        };
        // Use the same resolver predicates as reads. This keeps unsupported
        // formats, schemas, and manifest/file errors out of the catalog.
        for candle_mode in [CandleMode::Standard, CandleMode::HeikinAshi] {
            if !bar_type.supports_candle_mode() && candle_mode != CandleMode::Standard {
                continue;
            }
            if dataset.can_serve(bar_type, candle_mode, None)
                && dataset
                    .server_bars_file_for(bar_type, candle_mode, None)
                    .is_some()
            {
                let item = ChartApiSeries {
                    input: ChartApiInput::ServerBars,
                    bar_type,
                    candle_mode,
                };
                if !series.contains(&item) {
                    series.push(item);
                }
            }
        }
    }
    // Raw ticks are a separate, explicit input capability. Validate the
    // resolver up front so incomplete, corrupt, or unsupported caches are not
    // advertised as readable chart data.
    #[cfg(feature = "replay")]
    if let Ok(resolved) = dataset.resolve_raw_ticks_parquet_files(None) {
        let tick_size = dataset.manifest.tick_specs.tick_size;
        if tick_size.is_finite() && tick_size > 0.0 {
            for kind in [
                BarKind::Minute,
                BarKind::Second,
                BarKind::Tick,
                BarKind::Volume,
                BarKind::Range,
            ] {
                let bar_type = match kind {
                    BarKind::Minute => BarType::minute(1),
                    BarKind::Second => BarType::second(1),
                    BarKind::Tick => BarType::tick(1),
                    BarKind::Volume => BarType::volume(1),
                    BarKind::Range => BarType::range(1),
                };
                for candle_mode in [CandleMode::Standard, CandleMode::HeikinAshi] {
                    if !bar_type.supports_candle_mode() && candle_mode != CandleMode::Standard {
                        continue;
                    }
                    series.push(ChartApiSeries {
                        input: ChartApiInput::RawTicks,
                        bar_type,
                        candle_mode,
                    });
                }
            }
        }
        drop(resolved);
    }
    if series.is_empty() {
        return None;
    }
    Some(ChartApiDataset {
        id: chart_api_dataset_id(&dataset.manifest_path),
        display_name: dataset.manifest.display_name.clone(),
        provider: dataset.manifest.provider,
        environment: dataset.manifest.env,
        instrument: dataset.manifest.instrument.clone(),
        contract: dataset.manifest.contract.clone(),
        coverage: dataset.manifest.coverage.clone(),
        source_status: "stored_cache".into(),
        series,
    })
}

fn chart_api_dataset_id(path: &Path) -> String {
    let digest = Sha256::digest(path.to_string_lossy().as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Read a bounded page from a selected dataset and cached candle series.
/// `offset` is applied after optional half-open timestamp filtering.
pub fn chart_api_read_page(
    cache_root: impl Into<PathBuf>,
    dataset_id: &str,
    series: ChartApiSeries,
    start_ns: Option<i64>,
    end_ns: Option<i64>,
    offset: usize,
    limit: usize,
) -> Result<ChartApiPage> {
    ensure!(
        limit > 0 && limit <= CHART_API_MAX_PAGE_SIZE,
        "chart page limit must be between 1 and {CHART_API_MAX_PAGE_SIZE}"
    );
    if let (Some(start), Some(end)) = (start_ns, end_ns) {
        ensure!(start < end, "chart range must have start < end");
    }
    ensure!(
        series.bar_type.supports_candle_mode() || series.candle_mode == CandleMode::Standard,
        "requested bar type does not support Heikin Ashi candles"
    );
    let root = cache_root.into();
    let library = ReplayCacheLibrary::scan(&root);
    let dataset = library
        .datasets
        .iter()
        .find(|dataset| chart_api_dataset_id(&dataset.manifest_path) == dataset_id)
        .with_context(|| {
            format!(
                "chart dataset `{dataset_id}` was not found in {}",
                root.display()
            )
        })?;
    if series.input == ChartApiInput::ServerBars {
        ensure!(
            dataset.can_serve(series.bar_type, series.candle_mode, None),
            "dataset does not contain requested server-bar chart series"
        );
    } else {
        ensure!(
            dataset.resolve_raw_ticks_parquet_files(None).is_ok(),
            "dataset does not contain complete readable raw ticks"
        );
    }
    let time_range = match (series.input, start_ns, end_ns) {
        // Heikin Ashi is recursive, so load its complete history before
        // transforming and applying the requested output range.
        (ChartApiInput::ServerBars, Some(start), Some(end))
            if series.candle_mode == CandleMode::Standard =>
        {
            Some(ReplayCacheTimeRange::new(
                chrono::DateTime::from_timestamp_nanos(start),
                chrono::DateTime::from_timestamp_nanos(end),
            )?)
        }
        _ => None,
    };
    let (bars, source_status) = match series.input {
        ChartApiInput::ServerBars => (
            load_server_bars_cache_file_range(
                dataset,
                series.bar_type,
                series.candle_mode,
                None,
                time_range.as_ref(),
            )?
            .bars,
            "stored_cache",
        ),
        ChartApiInput::RawTicks => {
            #[cfg(feature = "replay")]
            {
                let resolved = dataset.resolve_raw_ticks_parquet_files(None)?;
                let tick_size = dataset.manifest.tick_specs.tick_size;
                ensure!(
                    tick_size.is_finite() && tick_size > 0.0,
                    "dataset has invalid tick size"
                );
                (
                    crate::tradovate::replay::derive_cached_raw_tick_chart_bars(
                        &resolved,
                        None,
                        series.bar_type,
                        tick_size,
                    )?,
                    "stored_raw_ticks",
                )
            }
            #[cfg(not(feature = "replay"))]
            anyhow::bail!("raw-tick chart aggregation requires the replay feature")
        }
    };
    let transformed = transform_bars_for_candle_mode(&bars, series.candle_mode)
        .into_iter()
        .filter(|bar| start_ns.is_none_or(|start| bar.ts_ns >= start))
        .filter(|bar| end_ns.is_none_or(|end| bar.ts_ns < end));
    let transformed = transformed.collect::<Vec<_>>();
    let total = transformed.len();
    let page = transformed
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|bar| ChartApiBar {
            timestamp_ns: bar.ts_ns,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        })
        .collect();
    Ok(ChartApiPage {
        dataset_id: dataset_id.to_owned(),
        series,
        source_status: source_status.into(),
        offset,
        limit,
        total,
        bars: page,
    })
}

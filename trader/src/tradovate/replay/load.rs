use super::*;

#[cfg(feature = "replay")]
use super::instrument::{
    infer_contract_name, infer_tick_size, infer_value_per_point, replay_contract_id,
};
#[cfg(feature = "replay")]
use super::state::ReplayDataSource;
#[cfg(feature = "replay")]
use super::ticks::parse_tick_line;
#[cfg(feature = "replay")]
use crate::replay_cache::{
    ReplayCacheLibrary, ReplayCacheLoadedServerBars, ReplayCacheResolvedRawTicks,
};
#[cfg(feature = "replay")]
use anyhow::Context;
#[cfg(feature = "replay")]
use std::fs::File;
#[cfg(feature = "replay")]
use std::io::{BufRead, BufReader};
use std::path::Path;
#[cfg(feature = "replay")]
use std::path::PathBuf;
#[cfg(feature = "replay")]
use std::sync::Arc;

pub(crate) async fn load_replay_state(
    cfg: &AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
    selected_manifest_path: Option<&Path>,
    selected_view_path: Option<&Path>,
) -> Result<ReplayState> {
    #[cfg(not(feature = "replay"))]
    {
        let _ = (
            cfg,
            bar_type,
            candle_mode,
            selected_manifest_path,
            selected_view_path,
        );
        bail!("replay mode is not enabled in this build; rebuild with `--features replay`");
    }

    #[cfg(feature = "replay")]
    {
        let cfg = cfg.clone();
        let selected_manifest_path = selected_manifest_path.map(Path::to_path_buf);
        let selected_view_path = selected_view_path.map(Path::to_path_buf);
        tokio::task::spawn_blocking(move || {
            load_replay_state_blocking(
                &cfg,
                bar_type,
                candle_mode,
                selected_manifest_path.as_deref(),
                selected_view_path.as_deref(),
            )
        })
        .await
        .context("join replay parser task")?
    }
}

#[cfg(feature = "replay")]
pub(super) fn load_replay_state_blocking(
    cfg: &AppConfig,
    bar_type: BarType,
    candle_mode: CandleMode,
    selected_manifest_path: Option<&Path>,
    selected_view_path: Option<&Path>,
) -> Result<ReplayState> {
    let library = ReplayCacheLibrary::scan(&cfg.replay_cache_dir);
    if let Some(view_path) = selected_view_path {
        let store = crate::replay_cache::ReplayDatasetViewStore::new(&cfg.replay_cache_dir);
        let view = store.load_path(view_path)?;
        let resolved = store.resolve_model(&view, view_path.to_path_buf())?;
        let replay_window = ReplayWindowSnapshot {
            preset: resolved.view.session_preset.label().to_string(),
            input_timezone: resolved.view.input_timezone.clone(),
            warmup_start: resolved.load_range.start,
            evaluation_start: resolved.evaluation_range.start,
            evaluation_end: resolved.evaluation_range.end,
            warmup_rows: 0,
            evaluation_rows_total: 0,
            evaluation_rows_processed: 0,
        };
        if selected_manifest_path.is_some_and(|path| path != resolved.dataset.manifest_path) {
            bail!("selected replay dataset view does not reference the selected source manifest");
        }
        if resolved
            .dataset
            .server_bars_file_for(bar_type, candle_mode, Some(&resolved.requested_coverage()))
            .is_some()
        {
            return replay_state_from_cached_server_bars(
                resolved.load_server_bars(bar_type, candle_mode)?,
                bar_type,
                candle_mode,
                Some(resolved.evaluation_range),
                Some(replay_window),
            );
        }
        if resolved
            .dataset
            .raw_ticks_parquet_file_for(Some(&resolved.requested_coverage()))
            .is_some()
        {
            let coverage = resolved.requested_coverage();
            let cached = resolved
                .dataset
                .resolve_raw_ticks_parquet_files(Some(&coverage))
                .context("load raw-tick replay dataset view")?;
            return replay_state_from_cached_raw_ticks(
                cached,
                Some(resolved.load_range),
                Some(resolved.evaluation_range),
                Some(replay_window),
            );
        }
        bail!(
            "selected replay dataset view does not support {}",
            bar_type.mode_label(candle_mode)
        );
    }
    if let Some(selected_path) = selected_manifest_path {
        let dataset = library
            .datasets
            .iter()
            .find(|dataset| dataset.manifest_path == selected_path)
            .with_context(|| {
                format!(
                    "selected replay dataset not found: {}",
                    selected_path.display()
                )
            })?;
        if dataset
            .server_bars_file_for(bar_type, candle_mode, None)
            .is_some()
        {
            return replay_state_from_cached_server_bars(
                crate::replay_cache::load_server_bars_cache_file(
                    dataset,
                    bar_type,
                    candle_mode,
                    None,
                )?,
                bar_type,
                candle_mode,
                None,
                None,
            );
        }
        if dataset.raw_ticks_parquet_file_for(None).is_some() {
            let cached = dataset
                .resolve_raw_ticks_parquet_files(None)
                .context("load selected raw-tick replay dataset")?;
            return replay_state_from_cached_raw_ticks(cached, None, None, None);
        }
        bail!(
            "selected replay dataset does not support {}",
            bar_type.mode_label(candle_mode)
        );
    }
    if let Some(cached) = library.load_first_server_bars(bar_type, candle_mode, None)? {
        return replay_state_from_cached_server_bars(cached, bar_type, candle_mode, None, None);
    }
    if let Some(cached) = library.resolve_unique_raw_ticks_parquet_files(None)? {
        return replay_state_from_cached_raw_ticks(cached, None, None, None);
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
    let account = replay_account("replay");

    Ok(ReplayState {
        evaluation_range: None,
        replay_window: None,
        contract,
        account,
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(value_per_point),
            tick_size: Some(tick_size),
        },
        data: ReplayDataSource::PriceTicks(Arc::from(ticks.into_boxed_slice())),
    })
}

#[cfg(feature = "replay")]
pub(super) fn replay_state_from_cached_raw_ticks(
    cached: ReplayCacheResolvedRawTicks,
    timestamp_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    evaluation_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    replay_window: Option<ReplayWindowSnapshot>,
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
        .unwrap_or_else(|| replay_contract_id(&cached.manifest_path));
    Ok(ReplayState {
        evaluation_range,
        replay_window,
        contract: ContractSuggestion {
            id: contract_id,
            name: contract_name.clone(),
            description: format!("Cached Raw Tick Dataset ({})", cached.manifest.display_name),
            raw: json!({
                "source": "replay-cache",
                "sourceKind": "raw-ticks",
                "manifestPath": cached.manifest_path.display().to_string(),
                "dataPaths": cached.files.iter().map(|file| file.data_path.display().to_string()).collect::<Vec<_>>(),
            }),
        },
        account: replay_account("replay-cache"),
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(cached.manifest.tick_specs.value_per_point),
            tick_size: Some(cached.manifest.tick_specs.tick_size),
        },
        data: ReplayDataSource::CachedRawTicks {
            resolved: cached,
            timestamp_range,
        },
    })
}

#[cfg(feature = "replay")]
fn replay_state_from_cached_server_bars(
    cached: ReplayCacheLoadedServerBars,
    requested_bar_type: BarType,
    requested_candle_mode: CandleMode,
    evaluation_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    replay_window: Option<ReplayWindowSnapshot>,
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
        evaluation_range,
        replay_window,
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
        account: replay_account("replay-cache"),
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
fn replay_account(source: &str) -> AccountInfo {
    AccountInfo {
        id: 1,
        name: "REPLAY".to_string(),
        raw: json!({
            "id": 1,
            "name": "REPLAY",
            "source": source,
            "startingBalance": 100000.0,
            "balance": 100000.0,
        }),
    }
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
pub(super) fn replay_path_candidates(
    path: &Path,
    cwd: Option<&Path>,
    manifest_dir: &Path,
) -> Vec<PathBuf> {
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

use super::*;

#[cfg(feature = "replay")]
use crate::broker::{ReplayDomLevel, ReplayFillModel};

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
    ReplayCacheLibrary, ReplayCacheResolvedRawTicks, ReplayCacheResolvedServerBarsFile,
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
    let dom_updates = if cfg.replay_fill_model == ReplayFillModel::Dom {
        load_replay_dom_updates(cfg.replay_dom_file_path.as_deref())?
    } else {
        Vec::new()
    };
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
            return attach_dom_updates(
                {
                    let (resolved_file, bars) =
                        crate::replay_cache::load_server_bars_cache_file_range_shared(
                            &resolved.dataset,
                            bar_type,
                            candle_mode,
                            Some(&resolved.requested_coverage()),
                            Some(&resolved.load_range),
                        )?;
                    replay_state_from_shared_server_bars(
                        resolved_file,
                        bars,
                        bar_type,
                        candle_mode,
                        Some(resolved.evaluation_range),
                        Some(replay_window),
                        cfg.replay_initial_capital,
                    )?
                },
                &dom_updates,
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
            return attach_dom_updates(
                replay_state_from_cached_raw_ticks(
                    cached,
                    Some(resolved.load_range),
                    Some(resolved.evaluation_range),
                    Some(replay_window),
                    cfg.replay_initial_capital,
                )?,
                &dom_updates,
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
            return attach_dom_updates(
                {
                    let (resolved_file, bars) =
                        crate::replay_cache::load_server_bars_cache_file_range_shared(
                            dataset,
                            bar_type,
                            candle_mode,
                            None,
                            None,
                        )?;
                    replay_state_from_shared_server_bars(
                        resolved_file,
                        bars,
                        bar_type,
                        candle_mode,
                        None,
                        None,
                        cfg.replay_initial_capital,
                    )?
                },
                &dom_updates,
            );
        }
        if dataset.raw_ticks_parquet_file_for(None).is_some() {
            let cached = dataset
                .resolve_raw_ticks_parquet_files(None)
                .context("load selected raw-tick replay dataset")?;
            return attach_dom_updates(
                replay_state_from_cached_raw_ticks(
                    cached,
                    None,
                    None,
                    None,
                    cfg.replay_initial_capital,
                )?,
                &dom_updates,
            );
        }
        bail!(
            "selected replay dataset does not support {}",
            bar_type.mode_label(candle_mode)
        );
    }
    if let Some(dataset) = library.first_server_bars(bar_type, candle_mode, None) {
        return attach_dom_updates(
            {
                let (resolved_file, bars) =
                    crate::replay_cache::load_server_bars_cache_file_range_shared(
                        dataset,
                        bar_type,
                        candle_mode,
                        None,
                        None,
                    )?;
                replay_state_from_shared_server_bars(
                    resolved_file,
                    bars,
                    bar_type,
                    candle_mode,
                    None,
                    None,
                    cfg.replay_initial_capital,
                )?
            },
            &dom_updates,
        );
    }
    if let Some(cached) = library.resolve_unique_raw_ticks_parquet_files(None)? {
        return attach_dom_updates(
            replay_state_from_cached_raw_ticks(
                cached,
                None,
                None,
                None,
                cfg.replay_initial_capital,
            )?,
            &dom_updates,
        );
    }

    attach_dom_updates(
        load_local_tick_replay_state_blocking(&cfg.replay_file_path, cfg.replay_initial_capital)?,
        &dom_updates,
    )
}

#[cfg(feature = "replay")]
fn load_local_tick_replay_state_blocking(path: &Path, initial_capital: f64) -> Result<ReplayState> {
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
    let account = replay_account("replay", initial_capital);

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
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::PriceTicks(Arc::from(ticks.into_boxed_slice())),
    })
}

#[cfg(feature = "replay")]
pub(super) fn replay_state_from_cached_raw_ticks(
    cached: ReplayCacheResolvedRawTicks,
    timestamp_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    evaluation_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    replay_window: Option<ReplayWindowSnapshot>,
    initial_capital: f64,
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
        account: replay_account("replay-cache", initial_capital),
        market_specs: MarketSpecs {
            session_profile: Some(InstrumentSessionProfile::FuturesGlobex),
            value_per_point: Some(cached.manifest.tick_specs.value_per_point),
            tick_size: Some(cached.manifest.tick_specs.tick_size),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::CachedRawTicks {
            resolved: cached,
            timestamp_range,
        },
    })
}

#[cfg(feature = "replay")]
fn replay_state_from_shared_server_bars(
    cached: ReplayCacheResolvedServerBarsFile,
    bars: Arc<[Bar]>,
    requested_bar_type: BarType,
    requested_candle_mode: CandleMode,
    evaluation_range: Option<crate::replay_cache::ReplayCacheTimeRange>,
    replay_window: Option<ReplayWindowSnapshot>,
    initial_capital: f64,
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
        account: replay_account("replay-cache", initial_capital),
        market_specs: MarketSpecs {
            session_profile: Some(session_profile),
            value_per_point: Some(cached.manifest.tick_specs.value_per_point),
            tick_size: Some(cached.manifest.tick_specs.tick_size),
        },
        dom_updates: Arc::from(Vec::<ReplayMarketDom>::new().into_boxed_slice()),
        data: ReplayDataSource::CachedServerBars {
            bars,
            bar_type: data_bar_type,
            source_label,
        },
    })
}

#[cfg(feature = "replay")]
fn attach_dom_updates(
    mut state: ReplayState,
    dom_updates: &[ReplayMarketDom],
) -> Result<ReplayState> {
    state.dom_updates = Arc::from(dom_updates.to_vec().into_boxed_slice());
    Ok(state)
}

#[cfg(feature = "replay")]
fn load_replay_dom_updates(path: Option<&Path>) -> Result<Vec<ReplayMarketDom>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let resolved_path = resolve_replay_path(path)?;
    let file = File::open(&resolved_path)
        .with_context(|| format!("open replay DOM file {}", resolved_path.display()))?;
    let reader = BufReader::new(file);
    let mut updates = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("read replay DOM file {}", resolved_path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut dom = serde_json::from_str::<ReplayMarketDom>(&line).with_context(|| {
            format!(
                "parse replay DOM snapshot {} in {}",
                line_index,
                resolved_path.display()
            )
        })?;
        normalize_replay_dom(&mut dom).with_context(|| {
            format!(
                "validate replay DOM snapshot {} in {}",
                line_index,
                resolved_path.display()
            )
        })?;
        updates.push(dom);
    }
    if updates.is_empty() {
        bail!(
            "replay DOM file {} contained no snapshots",
            resolved_path.display()
        );
    }
    updates.sort_by_key(|dom| dom.ts_ns);
    Ok(updates)
}

#[cfg(feature = "replay")]
fn normalize_replay_dom(dom: &mut ReplayMarketDom) -> Result<()> {
    if dom.ts_ns <= 0 {
        bail!("DOM timestamp must be positive nanoseconds");
    }
    normalize_replay_dom_side(&mut dom.bids, true)?;
    normalize_replay_dom_side(&mut dom.asks, false)?;
    if dom.bids.is_empty() && dom.asks.is_empty() {
        bail!("DOM snapshot has no positive-size bid or ask levels");
    }
    Ok(())
}

#[cfg(feature = "replay")]
fn normalize_replay_dom_side(levels: &mut Vec<ReplayDomLevel>, descending: bool) -> Result<()> {
    levels.retain(|level| level.size.is_finite() && level.size > 0.0);
    if levels
        .iter()
        .any(|level| !level.price.is_finite() || level.price <= 0.0)
    {
        bail!("DOM levels require finite positive prices");
    }
    levels.sort_by(|left, right| {
        if descending {
            right.price.total_cmp(&left.price)
        } else {
            left.price.total_cmp(&right.price)
        }
    });
    Ok(())
}

#[cfg(feature = "replay")]
fn replay_account(source: &str, initial_capital: f64) -> AccountInfo {
    AccountInfo {
        id: 1,
        name: "REPLAY".to_string(),
        raw: json!({
            "id": 1,
            "name": "REPLAY",
            "source": source,
            "startingBalance": initial_capital,
            "balance": initial_capital,
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

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn replay_dom_jsonl_loader_sorts_and_normalizes_levels() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "trader-replay-dom-{}-{suffix}.jsonl",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "{\"ts_ns\":2,\"bids\":[{\"price\":100.0,\"size\":0.0},{\"price\":99.75,\"size\":3.0}],\"asks\":[{\"price\":100.5,\"size\":2.0},{\"price\":100.25,\"size\":1.0}]}\n",
                "{\"ts_ns\":1,\"bids\":[{\"price\":99.5,\"size\":1.0}],\"asks\":[{\"price\":100.0,\"size\":1.0}]}\n",
            ),
        )
        .expect("write DOM fixture");

        let updates = load_replay_dom_updates(Some(&path)).expect("load DOM fixture");
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].ts_ns, 1);
        assert_eq!(updates[1].ts_ns, 2);
        assert_eq!(updates[1].bids[0].price, 99.75);
        assert_eq!(updates[1].asks[0].price, 100.25);
        assert_eq!(updates[1].bids.len(), 1);

        std::fs::remove_file(path).expect("remove DOM fixture");
    }

    #[test]
    fn replay_dom_jsonl_loader_rejects_empty_books() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "trader-replay-dom-empty-{}-{suffix}.jsonl",
            std::process::id()
        ));
        std::fs::write(&path, r#"{"ts_ns":1,"bids":[],"asks":[]}"#).expect("write DOM fixture");

        let error = load_replay_dom_updates(Some(&path)).expect_err("empty DOM should fail");
        assert!(format!("{error:#}").contains("no positive-size"));

        std::fs::remove_file(path).expect("remove DOM fixture");
    }
}

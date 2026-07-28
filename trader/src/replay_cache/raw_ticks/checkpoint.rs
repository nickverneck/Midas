use super::super::*;

pub fn write_raw_ticks_parquet_cache(
    write: ReplayCacheRawTicksWrite,
) -> Result<ReplayCacheWriteOutcome> {
    let normalized = normalize_raw_tick_rows(write.ticks);
    if normalized.rows.is_empty() {
        bail!("raw tick download returned no usable ticks");
    }
    let missing_tick_ids = normalized
        .rows
        .iter()
        .filter(|row| row.tick_id.is_none())
        .count();
    if missing_tick_ids > 0 {
        bail!(
            "raw tick provider response omitted stable tick IDs for {missing_tick_ids} of {} usable row(s); schema v2 cache was not written because replay cannot distinguish duplicate provider trades safely",
            normalized.rows.len()
        );
    }
    validate_raw_tick_id_write_invariant(&normalized.rows)?;

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

    let dataset_dir = replay_cache_write_dataset_dir(
        &write.cache_root,
        write.provider,
        write.env,
        &write.instrument.symbol,
        &write.contract,
        write.request_start.date_naive(),
        write.target.as_ref(),
    )?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let relative_path = versioned_cache_relative_path(&raw_ticks_relative_path(
        write.request_start,
        write.request_end,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_raw_ticks_parquet_file(&data_path, &normalized.rows)?;

    let data_hash = fnv1a64_file_hex(&data_path)?;
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
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: data_hash,
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
            display_name: write.display_name.clone().unwrap_or_else(|| {
                raw_ticks_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::RawTicks,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };

    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument;
    manifest.contract = write.contract;
    manifest.source_kind = ReplayCacheSourceKind::RawTicks;
    manifest.completed_raw_tick_coverage = None;
    manifest.completed_raw_tick_windows.clear();
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.contract_metadata =
        merge_contract_metadata(manifest.contract_metadata.take(), write.contract_metadata);
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = warnings;
    manifest.errors.clear();
    if let Some(display_name) = write.display_name.filter(|name| !name.trim().is_empty()) {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes;
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::RawTicks
                && file.format == ReplayCacheFileFormat::Parquet
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::RawTicks
            && file.format == ReplayCacheFileFormat::Parquet)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::RawTicks);
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
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);

    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path,
        data_path,
        row_count,
    })
}

pub(super) const RAW_TICK_CHECKPOINT_FILE_PREFIX: &str = "download-checkpoint";

pub fn prepare_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
    initial_windows: &[DownloadWindow],
) -> Result<ReplayCacheRawTickCheckpointState> {
    validate_exact_window_coverage(write.identity.request, initial_windows)?;
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = raw_tick_checkpoint_path(&dataset_dir, &write.identity);
    let checkpoint = if checkpoint_path.exists() {
        let checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
        validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
        checkpoint
    } else {
        let now = Utc::now();
        let checkpoint = ReplayCacheRawTickCheckpoint {
            checkpoint_version: RAW_TICK_CHECKPOINT_VERSION,
            identity_key: write.identity.stable_key(),
            identity: write.identity.clone(),
            created_at: now,
            updated_at: now,
            chunks: initial_windows
                .iter()
                .copied()
                .map(|window| ReplayCacheRawTickCheckpointChunk {
                    window,
                    status: ReplayCacheRawTickChunkStatus::Pending,
                    attempts: Vec::new(),
                })
                .collect(),
        };
        write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
        checkpoint
    };
    Ok(ReplayCacheRawTickCheckpointState {
        dataset_dir,
        checkpoint_path,
        checkpoint,
    })
}

pub fn record_raw_tick_chunk_attempt(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    sanitized_evidence: Value,
) -> Result<ReplayCacheRawTickCheckpointState> {
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = raw_tick_checkpoint_path(&dataset_dir, &write.identity);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint_identity(write, &checkpoint)?;
    let chunk = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    chunk.attempts.push(sanitized_evidence);
    checkpoint.updated_at = Utc::now();
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok(ReplayCacheRawTickCheckpointState {
        dataset_dir,
        checkpoint_path,
        checkpoint,
    })
}

pub fn split_raw_tick_checkpoint_chunk(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    session_boundaries: &[DateTime<Utc>],
    minimum: chrono::Duration,
    sanitized_evidence: Value,
) -> Result<(DownloadWindow, DownloadWindow)> {
    let (left, right) = split_download_window(window, session_boundaries, minimum)
        .context("raw-tick chunk cannot be split without violating the minimum interval")?;
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = raw_tick_checkpoint_path(&dataset_dir, &write.identity);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint_identity(write, &checkpoint)?;
    let parent = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    if !matches!(parent.status, ReplayCacheRawTickChunkStatus::Pending) {
        bail!("only a pending raw-tick checkpoint chunk can be split");
    }
    parent.attempts.push(sanitized_evidence);
    parent.status = ReplayCacheRawTickChunkStatus::Split { left, right };
    for child in [left, right] {
        checkpoint.chunks.push(ReplayCacheRawTickCheckpointChunk {
            window: child,
            status: ReplayCacheRawTickChunkStatus::Pending,
            attempts: Vec::new(),
        });
    }
    checkpoint.updated_at = Utc::now();
    validate_checkpoint_leaf_coverage(&checkpoint)?;
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok((left, right))
}

pub fn write_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
    window: DownloadWindow,
    ticks: Vec<ReplayCacheRawTickRow>,
    mut telemetry: HistoricalDownloadTelemetry,
) -> Result<ReplayCacheRawTickChunkWriteOutcome> {
    if telemetry.request != window {
        bail!("raw-tick chunk telemetry request does not match checkpoint window");
    }
    if !telemetry.completed_by_eoh() {
        bail!("raw-tick chunk cannot commit without explicit provider end-of-history");
    }
    let normalized = normalize_raw_tick_rows(ticks);
    let missing_tick_ids = normalized
        .rows
        .iter()
        .filter(|row| row.tick_id.is_none())
        .count();
    if missing_tick_ids > 0 {
        bail!(
            "raw tick provider response omitted stable tick IDs for {missing_tick_ids} of {} usable row(s); chunk was not committed",
            normalized.rows.len()
        );
    }
    validate_raw_tick_id_write_invariant(&normalized.rows)?;
    let start_ns = window
        .start
        .timestamp_nanos_opt()
        .context("raw-tick chunk start is outside nanosecond range")?;
    let end_ns = window
        .end
        .timestamp_nanos_opt()
        .context("raw-tick chunk end is outside nanosecond range")?;
    if normalized
        .rows
        .iter()
        .any(|row| row.ts_ns < start_ns || row.ts_ns >= end_ns)
    {
        bail!("raw-tick chunk contains rows outside its exact half-open request window");
    }
    telemetry.normalized_rows = normalized.rows.len() as u64;
    telemetry.duplicate_rows = normalized.duplicate_tick_ids as u64;
    telemetry.dropped_rows = telemetry
        .dropped_rows
        .saturating_add(normalized.dropped_rows as u64);
    telemetry.first_timestamp = normalized.rows.first().map(|row| row.timestamp);
    telemetry.last_timestamp = normalized.rows.last().map(|row| row.timestamp);

    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = raw_tick_checkpoint_path(&dataset_dir, &write.identity);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
    let existing = checkpoint
        .chunks
        .iter()
        .find(|chunk| chunk.window == window)
        .with_context(|| format!("raw-tick checkpoint has no chunk for {window:?}"))?;
    if let ReplayCacheRawTickChunkStatus::Completed { file, .. } = &existing.status {
        let data_path = file
            .as_ref()
            .map(|file| resolve_cache_data_path(&dataset_dir, &file.relative_path))
            .transpose()?;
        return Ok(ReplayCacheRawTickChunkWriteOutcome {
            dataset_dir,
            checkpoint_path,
            data_path,
            row_count: file.as_ref().map(|file| file.row_count).unwrap_or_default(),
        });
    }
    if !matches!(existing.status, ReplayCacheRawTickChunkStatus::Pending) {
        bail!("raw-tick checkpoint chunk is not a pending leaf");
    }

    let (data_file, data_path) = if normalized.rows.is_empty() {
        (None, None)
    } else {
        let relative_path = versioned_cache_relative_path(&raw_tick_chunk_relative_path(window));
        let data_path = dataset_dir.join(&relative_path);
        if let Some(parent) = data_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        write_raw_ticks_parquet_file(&data_path, &normalized.rows)?;
        let data_hash = fnv1a64_file_hex(&data_path)?;
        let first_timestamp = normalized
            .rows
            .first()
            .expect("rows are non-empty")
            .timestamp;
        let last_timestamp = normalized
            .rows
            .last()
            .expect("rows are non-empty")
            .timestamp;
        let mut warnings = Vec::new();
        if normalized.duplicate_tick_ids > 0 {
            warnings.push(format!(
                "Dropped {} duplicate raw tick id(s) from this chunk.",
                normalized.duplicate_tick_ids
            ));
        }
        if normalized.dropped_rows > 0 {
            warnings.push(format!(
                "Dropped {} malformed raw tick row(s) from this chunk.",
                normalized.dropped_rows
            ));
        }
        (
            Some(ReplayCacheDataFile {
                relative_path,
                source_kind: ReplayCacheSourceKind::RawTicks,
                format: ReplayCacheFileFormat::Parquet,
                schema_version: Some(RAW_TICKS_SCHEMA_VERSION),
                compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
                market_shape: ReplayCacheMarketShape {
                    bar_type: None,
                    chart_mode: None,
                    session_template: write.session_template.clone(),
                },
                row_count: normalized.rows.len() as u64,
                first_timestamp,
                last_timestamp,
                request_start: Some(window.start),
                request_end: Some(window.end),
                data_hash: Some(ReplayCacheDataHash {
                    algorithm: "fnv1a64".to_string(),
                    value: data_hash,
                }),
                warnings,
                errors: Vec::new(),
            }),
            Some(data_path),
        )
    };
    let chunk = checkpoint
        .chunks
        .iter_mut()
        .find(|chunk| chunk.window == window)
        .expect("chunk checked above");
    chunk.status = ReplayCacheRawTickChunkStatus::Completed {
        file: data_file,
        telemetry,
    };
    checkpoint.updated_at = Utc::now();
    write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
    Ok(ReplayCacheRawTickChunkWriteOutcome {
        dataset_dir,
        checkpoint_path,
        data_path,
        row_count: normalized.rows.len() as u64,
    })
}

pub fn finalize_raw_tick_chunk_cache(
    write: &ReplayCacheRawTickChunkPlanWrite,
) -> Result<ReplayCacheChunkedWriteOutcome> {
    let dataset_dir = raw_tick_chunk_dataset_dir(write)?;
    let _manifest_lock = ReplayManifestLock::acquire(&dataset_dir)?;
    let checkpoint_path = raw_tick_checkpoint_path(&dataset_dir, &write.identity);
    let mut checkpoint = read_raw_tick_checkpoint(&checkpoint_path)?;
    validate_raw_tick_checkpoint(write, &dataset_dir, &checkpoint)?;
    validate_checkpoint_leaf_coverage(&checkpoint)?;
    if !checkpoint.pending_windows().is_empty() {
        bail!("raw-tick checkpoint is incomplete; requested coverage was not published");
    }
    let mut files = checkpoint
        .chunks
        .iter()
        .filter_map(|chunk| match &chunk.status {
            ReplayCacheRawTickChunkStatus::Completed {
                file: Some(file), ..
            } => Some(file.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|file| file.request_start);
    if files.is_empty() {
        for chunk in &mut checkpoint.chunks {
            let ReplayCacheRawTickChunkStatus::Completed {
                file: None,
                telemetry,
            } = &chunk.status
            else {
                continue;
            };
            chunk.attempts.push(serde_json::json!({
                "kind": "empty_provider_history",
                "telemetry": telemetry,
            }));
            chunk.status = ReplayCacheRawTickChunkStatus::Pending;
        }
        checkpoint.updated_at = Utc::now();
        write_raw_tick_checkpoint(&checkpoint_path, &checkpoint)?;
        bail!(
            "Tradovate acknowledged the raw-tick chunks with end-of-history but returned zero provider ticks for the entire requested range; no replay dataset was published and the chunks remain retryable. Historical Tick charts may be unavailable for this account, contract, or date even when minute server bars are available"
        );
    }
    validate_raw_tick_chunk_file_sequence(&dataset_dir, &files)?;

    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        let first = files.first().expect("files are non-empty");
        let last = files.last().expect("files are non-empty");
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.identity.provider,
            env: write.identity.env,
            instrument: write.identity.instrument.clone(),
            contract: write.identity.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                raw_ticks_cache_display_name(
                    &write.identity.contract.symbol,
                    write.identity.request.start,
                    write.identity.request.end,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first.first_timestamp,
                end: last.last_timestamp,
                trading_date: Some(write.identity.request.start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::RawTicks,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };
    if manifest.provider != write.identity.provider
        || manifest.env != write.identity.env
        || !manifest
            .instrument
            .symbol
            .eq_ignore_ascii_case(&write.identity.instrument.symbol)
        || !manifest
            .contract
            .symbol
            .eq_ignore_ascii_case(&write.identity.contract.symbol)
        || (manifest.contract.id.is_some()
            && write.identity.contract.id.is_some()
            && manifest.contract.id != write.identity.contract.id)
    {
        bail!("raw-tick manifest identity does not match checkpoint identity");
    }
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::RawTicks
                && file.format == ReplayCacheFileFormat::Parquet
                && !files
                    .iter()
                    .any(|replacement| replacement.relative_path == file.relative_path)
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::RawTicks
            && file.format == ReplayCacheFileFormat::Parquet)
    });
    manifest.files.extend(files.clone());
    manifest.files.sort_by(|left, right| {
        left.request_start
            .cmp(&right.request_start)
            .then_with(|| left.relative_path.cmp(&right.relative_path))
    });
    manifest.provider = write.identity.provider;
    manifest.env = write.identity.env;
    manifest.instrument = write.identity.instrument.clone();
    manifest.contract = write.identity.contract.clone();
    manifest.download_request = write.download_request.clone();
    manifest.tick_specs = write.tick_specs.clone();
    manifest.contract_metadata = merge_contract_metadata(
        manifest.contract_metadata.take(),
        write.contract_metadata.clone(),
    );
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.identity.request.start);
    manifest.completed_raw_tick_coverage = Some(ReplayCacheCoverage {
        start: write.identity.request.start,
        end: write.identity.request.end,
        trading_date: Some(write.identity.request.start.date_naive()),
    });
    manifest.completed_raw_tick_windows = checkpoint
        .chunks
        .iter()
        .filter(|chunk| {
            matches!(
                chunk.status,
                ReplayCacheRawTickChunkStatus::Completed { .. }
            )
        })
        .map(|chunk| chunk.window)
        .collect();
    manifest
        .completed_raw_tick_windows
        .sort_by_key(|window| window.start);
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::RawTicks);
    manifest.warnings = write.warnings.clone();
    manifest.errors.clear();
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    if let Some(display_name) = write
        .display_name
        .clone()
        .filter(|name| !name.trim().is_empty())
    {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags.clone() {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes.clone();
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect();
    manifest
        .available_bar_shapes
        .sort_by_key(|bar_type| bar_type.value());
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.chart_mode)
        .collect();
    manifest.available_chart_modes.dedup();
    manifest.badges = manifest.derived_badges();
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);
    let data_paths = files
        .iter()
        .map(|file| resolve_cache_data_path(&dataset_dir, &file.relative_path))
        .collect::<Result<Vec<_>>>()?;
    Ok(ReplayCacheChunkedWriteOutcome {
        dataset_dir,
        manifest_path,
        data_paths,
        row_count: files.iter().map(|file| file.row_count).sum(),
    })
}

pub(super) fn raw_tick_chunk_dataset_dir(
    write: &ReplayCacheRawTickChunkPlanWrite,
) -> Result<PathBuf> {
    replay_cache_write_dataset_dir(
        &write.cache_root,
        write.identity.provider,
        write.identity.env,
        &write.identity.instrument.symbol,
        &write.identity.contract,
        write.identity.request.start.date_naive(),
        write.target.as_ref(),
    )
}

pub(super) fn raw_tick_chunk_relative_path(window: DownloadWindow) -> PathBuf {
    PathBuf::from("raw-ticks").join("chunks").join(format!(
        "{}_to_{}_ticks.parquet",
        window.start.format("%Y%m%dT%H%M%S%.3fZ"),
        window.end.format("%Y%m%dT%H%M%S%.3fZ")
    ))
}

pub(super) fn raw_tick_checkpoint_path(
    dataset_dir: &Path,
    identity: &ReplayCacheRawTickCheckpointIdentity,
) -> PathBuf {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in identity.stable_key().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    dataset_dir.join("raw-ticks").join(format!(
        "{RAW_TICK_CHECKPOINT_FILE_PREFIX}-{hash:016x}.json"
    ))
}

pub(super) fn read_raw_tick_checkpoint(path: &Path) -> Result<ReplayCacheRawTickCheckpoint> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn write_raw_tick_checkpoint(
    path: &Path,
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    write_bytes_atomically(
        path,
        &serde_json::to_vec_pretty(checkpoint).context("serialize raw-tick checkpoint")?,
    )
}

pub(super) fn validate_raw_tick_checkpoint_identity(
    write: &ReplayCacheRawTickChunkPlanWrite,
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    if checkpoint.checkpoint_version != RAW_TICK_CHECKPOINT_VERSION {
        bail!(
            "unsupported raw-tick checkpoint version {}",
            checkpoint.checkpoint_version
        );
    }
    if checkpoint.identity_key != write.identity.stable_key()
        || checkpoint.identity != write.identity
    {
        bail!(
            "raw-tick checkpoint request identity mismatch; use a new dataset or remove the incompatible checkpoint"
        );
    }
    Ok(())
}

pub(super) fn validate_raw_tick_checkpoint(
    write: &ReplayCacheRawTickChunkPlanWrite,
    dataset_dir: &Path,
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    validate_raw_tick_checkpoint_identity(write, checkpoint)?;
    validate_checkpoint_leaf_coverage(checkpoint)?;
    for chunk in &checkpoint.chunks {
        let ReplayCacheRawTickChunkStatus::Completed {
            file: Some(file),
            telemetry,
        } = &chunk.status
        else {
            continue;
        };
        if file.request_start != Some(chunk.window.start)
            || file.request_end != Some(chunk.window.end)
            || telemetry.request != chunk.window
            || !telemetry.completed_by_eoh()
        {
            bail!("raw-tick checkpoint completed-chunk evidence mismatch");
        }
        let path = resolve_cache_data_path(dataset_dir, &file.relative_path)?;
        validate_cache_file_hash(&path, file)?;
    }
    Ok(())
}

pub(super) fn validate_cache_file_hash(path: &Path, file: &ReplayCacheDataFile) -> Result<()> {
    let expected = file
        .data_hash
        .as_ref()
        .context("checkpoint data file has no hash")?;
    if expected.algorithm != "fnv1a64" {
        bail!(
            "unsupported checkpoint hash algorithm {}",
            expected.algorithm
        );
    }
    let actual = fnv1a64_file_hex(path)?;
    if actual != expected.value {
        bail!(
            "checkpoint data hash mismatch for {}: expected {} actual {}",
            path.display(),
            expected.value,
            actual
        );
    }
    Ok(())
}

pub(super) fn validate_checkpoint_leaf_coverage(
    checkpoint: &ReplayCacheRawTickCheckpoint,
) -> Result<()> {
    let mut leaves = checkpoint
        .chunks
        .iter()
        .filter(|chunk| !matches!(chunk.status, ReplayCacheRawTickChunkStatus::Split { .. }))
        .map(|chunk| chunk.window)
        .collect::<Vec<_>>();
    leaves.sort_by_key(|window| window.start);
    validate_exact_window_coverage(checkpoint.identity.request, &leaves)
}

pub(in crate::replay_cache) fn validate_exact_window_coverage(
    parent: DownloadWindow,
    windows: &[DownloadWindow],
) -> Result<()> {
    let first = windows
        .first()
        .context("raw-tick chunk plan has no windows")?;
    if first.start != parent.start {
        bail!("raw-tick chunk plan does not begin at requested start");
    }
    let mut expected_start = parent.start;
    for window in windows {
        if window.start != expected_start || window.start >= window.end || window.end > parent.end {
            bail!("raw-tick chunk plan has a gap, overlap, or out-of-range window");
        }
        expected_start = window.end;
    }
    if expected_start != parent.end {
        bail!("raw-tick chunk plan does not end at requested end");
    }
    Ok(())
}

pub(super) fn validate_raw_tick_chunk_file_sequence(
    dataset_dir: &Path,
    files: &[ReplayCacheDataFile],
) -> Result<()> {
    let mut previous_ts = None;
    let mut previous_id = None;
    for file in files {
        let request = DownloadWindow::new(
            file.request_start
                .context("raw-tick chunk file has no request start")?,
            file.request_end
                .context("raw-tick chunk file has no request end")?,
        )?;
        let path = resolve_cache_data_path(dataset_dir, &file.relative_path)?;
        validate_cache_file_hash(&path, file)?;
        let mut rows = 0_u64;
        stream_raw_ticks_parquet_file(&path, None, |row| {
            if row.timestamp < request.start || row.timestamp >= request.end {
                bail!("raw-tick chunk row is outside its manifest request window");
            }
            if previous_ts.is_some_and(|timestamp| row.ts_ns < timestamp) {
                bail!("raw-tick chunk files are not globally timestamp ordered");
            }
            let tick_id = row.tick_id.context("schema-v2 chunk row has no tick id")?;
            if previous_id.is_some_and(|id| tick_id <= id) {
                bail!("raw-tick chunk files are not globally tick-id ordered");
            }
            previous_ts = Some(row.ts_ns);
            previous_id = Some(tick_id);
            rows = rows.saturating_add(1);
            Ok(())
        })?;
        if rows != file.row_count {
            bail!(
                "raw-tick chunk row-count mismatch for {}: manifest={} actual={rows}",
                path.display(),
                file.row_count
            );
        }
    }
    Ok(())
}

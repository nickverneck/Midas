use super::*;

#[allow(dead_code)]
pub fn write_server_bars_jsonl_cache(
    write: ReplayCacheServerBarsWrite,
) -> Result<ReplayCacheWriteOutcome> {
    if write.source_kind != ReplayCacheSourceKind::ServerBars {
        bail!("JSONL server-bar cache writer only accepts server-bars source data");
    }

    let rows = normalize_server_bar_rows(write.bars.clone());
    if rows.is_empty() {
        bail!("server-bar download returned no usable bars");
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
    let relative_path = versioned_cache_relative_path(&server_bars_relative_path(
        write.request_start,
        write.request_end,
        write.bar_type,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    let mut data = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut data, row)?;
        data.push(b'\n');
    }
    write_bytes_atomically(&data_path, &data)?;

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
        request_start: None,
        request_end: None,
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
            display_name: write.display_name.clone().unwrap_or_else(|| {
                replay_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                    write.bar_type,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::ServerBars,
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
    manifest.source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.download_request = write.download_request;
    manifest.tick_specs = write.tick_specs;
    manifest.contract_metadata =
        merge_contract_metadata(manifest.contract_metadata.take(), write.contract_metadata);
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = write.warnings;
    manifest.errors.clear();
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
    manifest.notes = write.notes;
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.market_shape.bar_type == Some(write.bar_type)
                && file.format == ReplayCacheFileFormat::Jsonl
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::ServerBars
            && file.market_shape.bar_type == Some(write.bar_type)
            && file.format == ReplayCacheFileFormat::Jsonl)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::ServerBars);
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

pub fn write_server_bars_parquet_cache(
    write: ReplayCacheServerBarsWrite,
) -> Result<ReplayCacheWriteOutcome> {
    if write.source_kind != ReplayCacheSourceKind::ServerBars {
        bail!("Parquet server-bar cache writer only accepts server-bars source data");
    }

    let rows = normalize_server_bar_rows(write.bars.clone());
    if rows.is_empty() {
        bail!("server-bar download returned no usable bars");
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
    let relative_path = versioned_cache_relative_path(&server_bars_parquet_relative_path(
        write.request_start,
        write.request_end,
        write.bar_type,
    ));
    let data_path = dataset_dir.join(&relative_path);
    if let Some(parent) = data_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    write_server_bars_parquet_file(&data_path, &rows)?;
    let data_hash = fnv1a64_file_hex(&data_path)?;
    let first_timestamp = rows.first().expect("rows are non-empty").timestamp;
    let last_timestamp = rows.last().expect("rows are non-empty").timestamp;
    let data_file = ReplayCacheDataFile {
        relative_path: relative_path.clone(),
        source_kind: ReplayCacheSourceKind::ServerBars,
        format: ReplayCacheFileFormat::Parquet,
        schema_version: Some(SERVER_BARS_SCHEMA_VERSION),
        compression: Some(PARQUET_COMPRESSION_LABEL.to_string()),
        market_shape: ReplayCacheMarketShape {
            bar_type: Some(write.bar_type),
            chart_mode: None,
            session_template: write.session_template.clone(),
        },
        row_count: rows.len() as u64,
        first_timestamp,
        last_timestamp,
        request_start: None,
        request_end: None,
        data_hash: Some(ReplayCacheDataHash {
            algorithm: "fnv1a64".to_string(),
            value: data_hash,
        }),
        warnings: write.warnings.clone(),
        errors: Vec::new(),
    };
    let outcome = upsert_server_bars_manifest(
        &write,
        &dataset_dir,
        data_file,
        first_timestamp,
        last_timestamp,
    )?;
    Ok(ReplayCacheWriteOutcome {
        dataset_dir,
        manifest_path: outcome.0,
        data_path,
        row_count: rows.len() as u64,
    })
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

pub(super) fn server_bars_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, true),
    ]))
}

pub fn write_server_bars_parquet_file(path: &Path, rows: &[ReplayCacheServerBarRow]) -> Result<()> {
    write_server_bars_parquet_file_with_limits(
        path,
        rows,
        PARQUET_ROW_GROUP_ROWS,
        PARQUET_WRITE_BATCH_ROWS,
    )
}

pub(super) fn write_server_bars_parquet_file_with_limits(
    path: &Path,
    rows: &[ReplayCacheServerBarRow],
    row_group_rows: usize,
    batch_rows: usize,
) -> Result<()> {
    if rows.is_empty() {
        bail!("server-bar parquet writer requires at least one row");
    }
    if row_group_rows == 0 || batch_rows == 0 {
        bail!("parquet row-group and write-batch limits must be positive");
    }
    let schema = server_bars_parquet_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_size(row_group_rows)
        .set_write_batch_size(batch_rows)
        .build();
    write_file_atomically(path, |file| {
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        for chunk in rows.chunks(batch_rows) {
            writer.write(&server_bars_record_batch(schema.clone(), chunk)?)?;
        }
        writer.close()?;
        Ok(())
    })
}

pub(super) fn server_bars_record_batch(
    schema: Arc<Schema>,
    rows: &[ReplayCacheServerBarRow],
) -> Result<RecordBatch> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.timestamp.to_rfc3339())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.ts_ns).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.open).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.high).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.low).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.close).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|row| row.volume).collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(schema, arrays).context("build server-bar parquet record batch")
}

pub fn read_server_bars_parquet_file(path: &Path) -> Result<Vec<Bar>> {
    let (bars, _) = read_server_bars_parquet_file_range(path, None)?;
    if bars.is_empty() {
        bail!(
            "server-bar parquet file {} contained no bars",
            path.display()
        );
    }
    Ok(bars)
}

pub fn read_server_bars_parquet_file_range(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<Bar>, ReplayCacheParquetReadStats)> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?;
    let (selected_row_groups, mut stats) = parquet_row_groups_for_range(&builder, range)?;
    if selected_row_groups.is_empty() {
        return Ok((Vec::new(), stats));
    }
    let reader = builder
        .with_row_groups(selected_row_groups)
        .with_batch_size(PARQUET_READ_BATCH_ROWS)
        .build()
        .with_context(|| format!("build parquet reader {}", path.display()))?;
    let mut bars = Vec::new();
    let mut last_ts_ns = None;
    for batch in reader {
        let batch = batch.with_context(|| format!("read parquet batch {}", path.display()))?;
        stats.record_batches = stats.record_batches.saturating_add(1);
        stats.max_record_batch_rows = stats.max_record_batch_rows.max(batch.num_rows());
        stats.decoded_rows = stats.decoded_rows.saturating_add(batch.num_rows() as u64);
        let timestamps = parquet_column::<StringArray>(&batch, 0, "timestamp")?;
        let ts_ns = parquet_column::<Int64Array>(&batch, 1, "ts_ns")?;
        let opens = parquet_column::<Float64Array>(&batch, 2, "open")?;
        let highs = parquet_column::<Float64Array>(&batch, 3, "high")?;
        let lows = parquet_column::<Float64Array>(&batch, 4, "low")?;
        let closes = parquet_column::<Float64Array>(&batch, 5, "close")?;
        let volumes = parquet_column::<Float64Array>(&batch, 6, "volume")?;
        for index in 0..batch.num_rows() {
            let timestamp = DateTime::parse_from_rfc3339(timestamps.value(index))
                .with_context(|| format!("parse server bar timestamp row {index}"))?
                .with_timezone(&Utc);
            let bar = server_bar_row_to_bar(ReplayCacheServerBarRow {
                timestamp,
                ts_ns: ts_ns.value(index),
                open: opens.value(index),
                high: highs.value(index),
                low: lows.value(index),
                close: closes.value(index),
                volume: optional_f64(volumes, index),
            })?;
            if last_ts_ns.is_some_and(|last| bar.ts_ns <= last) {
                bail!(
                    "server-bar parquet file {} is not strictly ordered at timestamp {}",
                    path.display(),
                    bar.ts_ns
                );
            }
            last_ts_ns = Some(bar.ts_ns);
            if timestamp_range_contains(range, bar.ts_ns)? {
                record_emitted_timestamp(&mut stats, bar.ts_ns);
                bars.push(bar);
            }
        }
    }
    Ok((bars, stats))
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
    validate_loaded_server_bars_metadata(&resolved, &bars, None)?;

    Ok(ReplayCacheLoadedServerBars {
        manifest_path: resolved.manifest_path,
        dataset_dir: resolved.dataset_dir,
        data_path: resolved.data_path,
        manifest: resolved.manifest,
        file: resolved.file,
        bars,
    })
}

pub fn load_server_bars_cache_file(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
) -> Result<ReplayCacheLoadedServerBars> {
    load_server_bars_cache_file_range(dataset, bar_type, candle_mode, requested_coverage, None)
}

pub fn load_server_bars_cache_file_range(
    dataset: &ReplayCacheDataset,
    bar_type: BarType,
    candle_mode: CandleMode,
    requested_coverage: Option<&ReplayCacheCoverage>,
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<ReplayCacheLoadedServerBars> {
    let resolved = dataset.resolve_server_bars_file(bar_type, candle_mode, requested_coverage)?;
    let bars = match resolved.file.format {
        ReplayCacheFileFormat::Parquet => {
            read_server_bars_parquet_file_range(&resolved.data_path, timestamp_range)?.0
        }
        ReplayCacheFileFormat::Jsonl => read_server_bars_jsonl_file(&resolved.data_path)?
            .into_iter()
            .filter(|bar| timestamp_range_contains(timestamp_range, bar.ts_ns).unwrap_or(false))
            .collect(),
        format => bail!("unsupported server-bar cache format: {format:?}"),
    };
    validate_loaded_server_bars_metadata(&resolved, &bars, timestamp_range)?;
    Ok(ReplayCacheLoadedServerBars {
        manifest_path: resolved.manifest_path,
        dataset_dir: resolved.dataset_dir,
        data_path: resolved.data_path,
        manifest: resolved.manifest,
        file: resolved.file,
        bars,
    })
}

pub(super) fn server_bar_row_to_bar(row: ReplayCacheServerBarRow) -> Result<Bar> {
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

pub(super) fn normalize_read_server_bars(mut bars: Vec<Bar>) -> Result<Vec<Bar>> {
    bars.sort_by_key(|bar| bar.ts_ns);
    bars.dedup_by_key(|bar| bar.ts_ns);
    for bar in &bars {
        validate_server_bar(bar)?;
    }
    Ok(bars)
}

pub(super) fn validate_server_bar(bar: &Bar) -> Result<()> {
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

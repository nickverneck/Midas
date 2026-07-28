use super::super::*;

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug, Clone)]
pub(in crate::replay_cache) struct PositionIndependentFile {
    file: Arc<File>,
    len: u64,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl PositionIndependentFile {
    pub(in crate::replay_cache) fn new(file: Arc<File>) -> Result<Self> {
        let len = file
            .metadata()
            .context("inspect leased replay cache file")?
            .len();
        Ok(Self { file, len })
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Length for PositionIndependentFile {
    fn len(&self) -> u64 {
        self.len
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[derive(Debug)]
pub(in crate::replay_cache) struct PositionIndependentRead {
    file: Arc<File>,
    offset: u64,
    len: u64,
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Read for PositionIndependentRead {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() || self.offset >= self.len {
            return Ok(0);
        }
        let remaining = usize::try_from((self.len - self.offset).min(buffer.len() as u64))
            .unwrap_or(buffer.len());
        loop {
            match self.file.read_at(&mut buffer[..remaining], self.offset) {
                Ok(read) => {
                    self.offset = self.offset.saturating_add(read as u64);
                    return Ok(read);
                }
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ChunkReader for PositionIndependentFile {
    type T = PositionIndependentRead;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        if start > self.len {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Parquet read starts beyond end of file: {start} > {}",
                    self.len
                ),
            )
            .into());
        }
        Ok(PositionIndependentRead {
            file: self.file.clone(),
            offset: start,
            len: self.len,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let end = start.checked_add(length as u64).ok_or_else(|| {
            std::io::Error::new(ErrorKind::InvalidInput, "Parquet byte range overflow")
        })?;
        if end > self.len {
            return Err(std::io::Error::new(
                ErrorKind::UnexpectedEof,
                format!(
                    "Parquet byte range exceeds file length: {start}..{end} > {}",
                    self.len
                ),
            )
            .into());
        }

        let mut bytes = vec![0_u8; length];
        let mut read = 0usize;
        while read < length {
            match self.file.read_at(&mut bytes[read..], start + read as u64) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        ErrorKind::UnexpectedEof,
                        format!(
                            "short position-independent read at byte {}",
                            start + read as u64
                        ),
                    )
                    .into());
                }
                Ok(count) => read += count,
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) => return Err(err.into()),
            }
        }
        Ok(Bytes::from(bytes))
    }
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

pub(in crate::replay_cache) fn validate_raw_tick_id_write_invariant(
    rows: &[ReplayCacheRawTickRow],
) -> Result<()> {
    let mut previous = None;
    for (index, row) in rows.iter().enumerate() {
        let tick_id = row.tick_id.with_context(|| {
            format!(
                "raw tick row {index} has no tick id; schema v2 cache was not written because replay cannot prove global uniqueness"
            )
        })?;
        if previous.is_some_and(|last| tick_id <= last) {
            bail!(
                "raw tick ids must be strictly increasing in timestamp order; row {index} has id {tick_id} after {:?}",
                previous
            );
        }
        previous = Some(tick_id);
    }
    Ok(())
}

pub fn write_raw_ticks_parquet_file(path: &Path, rows: &[ReplayCacheRawTickRow]) -> Result<()> {
    write_raw_ticks_parquet_file_with_limits(
        path,
        rows,
        PARQUET_ROW_GROUP_ROWS,
        PARQUET_WRITE_BATCH_ROWS,
    )
}

pub(crate) fn write_raw_ticks_parquet_file_with_limits(
    path: &Path,
    rows: &[ReplayCacheRawTickRow],
    row_group_rows: usize,
    batch_rows: usize,
) -> Result<()> {
    if rows.is_empty() {
        bail!("raw tick parquet writer requires at least one row");
    }
    if row_group_rows == 0 || batch_rows == 0 {
        bail!("parquet row-group and write-batch limits must be positive");
    }

    let schema = raw_ticks_parquet_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_size(row_group_rows)
        .set_write_batch_size(batch_rows)
        .build();
    write_file_atomically(path, |file| {
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        for chunk in rows.chunks(batch_rows) {
            writer.write(&raw_ticks_record_batch(schema.clone(), chunk)?)?;
        }
        writer.close()?;
        Ok(())
    })
}

pub(in crate::replay_cache) fn raw_ticks_record_batch(
    schema: Arc<Schema>,
    rows: &[ReplayCacheRawTickRow],
) -> Result<RecordBatch> {
    let timestamp_values = rows
        .iter()
        .map(|row| row.timestamp.to_rfc3339())
        .collect::<Vec<_>>();
    let packet_sources = rows
        .iter()
        .map(|row| row.packet_source.as_deref())
        .collect::<Vec<_>>();
    let tick_ids: ArrayRef = if schema.field(2).is_nullable() {
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
        ))
    } else {
        let values = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.tick_id.with_context(|| {
                    format!("raw tick row {index} has no tick id; cache schema requires one")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Arc::new(Int64Array::from(values))
    };
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(timestamp_values)),
        Arc::new(Int64Array::from(
            rows.iter().map(|row| row.ts_ns).collect::<Vec<_>>(),
        )),
        tick_ids,
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
    RecordBatch::try_new(schema, arrays).context("build raw-tick parquet record batch")
}

#[allow(dead_code)]
pub fn read_raw_ticks_parquet_file(path: &Path) -> Result<Vec<ReplayCacheRawTickRow>> {
    let mut rows = Vec::new();
    let stats = stream_raw_ticks_parquet_file(path, None, |row| {
        rows.push(row);
        Ok(())
    })?;
    if stats.emitted_rows == 0 {
        bail!(
            "raw tick parquet file {} contained no usable ticks",
            path.display()
        );
    }
    Ok(rows)
}

pub fn read_raw_ticks_parquet_file_range(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<ReplayCacheRawTickRow>, ReplayCacheParquetReadStats)> {
    let mut rows = Vec::new();
    let stats = stream_raw_ticks_parquet_file(path, range, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok((rows, stats))
}

pub fn stream_raw_ticks_parquet_file<F>(
    path: &Path,
    range: Option<&ReplayCacheTimeRange>,
    on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let file = Arc::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
        return stream_raw_ticks_parquet_reader(
            PositionIndependentFile::new(file)?,
            path,
            range,
            None,
            on_tick,
        );
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (path, range, on_tick);
        bail!(
            "position-independent replay cache reads are unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

pub(super) fn stream_raw_ticks_parquet_reader<T, F>(
    source: T,
    display_path: &Path,
    range: Option<&ReplayCacheTimeRange>,
    declared_schema_version: Option<u32>,
    mut on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    T: ChunkReader + 'static,
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    let builder = ParquetRecordBatchReaderBuilder::try_new(source)
        .with_context(|| format!("open parquet reader {}", display_path.display()))?;
    let read_policy = raw_tick_read_policy(&builder, display_path, declared_schema_version)?;
    if read_policy == RawTickReadPolicy::StrictV2ProviderIds {
        validate_raw_tick_id_row_groups(&builder, display_path)?;
    }
    let (selected_row_groups, mut stats) = if read_policy == RawTickReadPolicy::LegacyV1SequenceIds
    {
        let total_row_groups = builder.metadata().num_row_groups();
        (
            (0..total_row_groups).collect::<Vec<_>>(),
            ReplayCacheParquetReadStats {
                total_row_groups,
                selected_row_groups: total_row_groups,
                ..ReplayCacheParquetReadStats::default()
            },
        )
    } else {
        parquet_row_groups_for_range(&builder, range)?
    };
    if selected_row_groups.is_empty() {
        return Ok(stats);
    }
    let reader = builder
        .with_row_groups(selected_row_groups)
        .with_batch_size(PARQUET_READ_BATCH_ROWS)
        .build()
        .with_context(|| format!("build parquet reader {}", display_path.display()))?;
    let mut last_ts_ns = None;
    let mut last_tick_id = None;
    let mut legacy_sequence_id = 0_i64;

    for batch in reader {
        let batch =
            batch.with_context(|| format!("read parquet batch {}", display_path.display()))?;
        stats.record_batches = stats.record_batches.saturating_add(1);
        stats.max_record_batch_rows = stats.max_record_batch_rows.max(batch.num_rows());
        stats.decoded_rows = stats.decoded_rows.saturating_add(batch.num_rows() as u64);
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
            let mut row = ReplayCacheRawTickRow {
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
            };
            validate_raw_tick_row(&row).with_context(|| {
                format!(
                    "validate raw tick row {row_index} in {}",
                    display_path.display()
                )
            })?;
            if last_ts_ns.is_some_and(|last| row.ts_ns < last) {
                bail!(
                    "raw tick parquet file {} is not ordered by timestamp at row {}",
                    display_path.display(),
                    row_index
                );
            }
            last_ts_ns = Some(row.ts_ns);
            match read_policy {
                RawTickReadPolicy::LegacyV1SequenceIds => {
                    legacy_sequence_id = legacy_sequence_id.checked_add(1).with_context(|| {
                        format!(
                            "raw tick schema v1 file {} exceeds deterministic sequence-id capacity",
                            display_path.display()
                        )
                    })?;
                    row.tick_id = Some(legacy_sequence_id);
                }
                RawTickReadPolicy::StrictV2ProviderIds => {
                    let tick_id = row.tick_id.with_context(|| {
                        format!(
                            "raw tick schema v2 file {} has no tick id at row {}",
                            display_path.display(),
                            row_index
                        )
                    })?;
                    if last_tick_id == Some(tick_id) {
                        bail!(
                            "raw tick parquet file {} contains duplicate tick id {} at timestamp {}",
                            display_path.display(),
                            tick_id,
                            row.ts_ns
                        );
                    }
                    if last_tick_id.is_some_and(|last| tick_id < last) {
                        bail!(
                            "raw tick parquet file {} cannot prove duplicate-id safety because tick ids are not strictly increasing: {} follows {:?}",
                            display_path.display(),
                            tick_id,
                            last_tick_id
                        );
                    }
                    last_tick_id = Some(tick_id);
                }
            }
            if timestamp_range_contains(range, row.ts_ns)? {
                record_emitted_timestamp(&mut stats, row.ts_ns);
                on_tick(row)?;
            }
        }
    }
    Ok(stats)
}

pub(super) fn raw_tick_read_policy<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    path: &Path,
    declared_schema_version: Option<u32>,
) -> Result<RawTickReadPolicy> {
    let legacy_schema = parquet_column_matches_expected_i64(builder, 2, "tick_id", true);
    let strict_schema = parquet_column_matches_expected_i64(builder, 2, "tick_id", false);
    match declared_schema_version {
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION) if legacy_schema => {
            Ok(RawTickReadPolicy::LegacyV1SequenceIds)
        }
        Some(RAW_TICKS_SCHEMA_VERSION) if strict_schema => {
            Ok(RawTickReadPolicy::StrictV2ProviderIds)
        }
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION) => bail!(
            "raw tick schema v1 file {} must contain a nullable INT64 tick_id column",
            path.display()
        ),
        Some(RAW_TICKS_SCHEMA_VERSION) => bail!(
            "raw tick schema v2 file {} must contain a required INT64 tick_id column",
            path.display()
        ),
        Some(version) => bail!(
            "raw tick parquet file {} declares unsupported schema version {}",
            path.display(),
            version
        ),
        None if strict_schema => Ok(RawTickReadPolicy::StrictV2ProviderIds),
        None if legacy_schema => bail!(
            "raw tick schema v1 file {} requires a manifest-declared schema version and verified data hash; re-download the dataset if its manifest is unavailable",
            path.display()
        ),
        None => bail!(
            "raw tick parquet file {} has an unsupported tick_id schema",
            path.display()
        ),
    }
}

pub(in crate::replay_cache) fn parquet_row_groups_for_range<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    range: Option<&ReplayCacheTimeRange>,
) -> Result<(Vec<usize>, ReplayCacheParquetReadStats)> {
    let total_row_groups = builder.metadata().num_row_groups();
    let mut stats = ReplayCacheParquetReadStats {
        total_row_groups,
        ..ReplayCacheParquetReadStats::default()
    };
    let Some(range) = range.copied() else {
        let selected = (0..total_row_groups).collect::<Vec<_>>();
        stats.selected_row_groups = selected.len();
        return Ok((selected, stats));
    };
    let (start_ns, end_ns) = range.bounds_ns()?;
    let trusted_ts_ns_schema = parquet_column_matches_expected_i64(builder, 1, "ts_ns", false);
    let mut selected = Vec::new();
    for (index, row_group) in builder.metadata().row_groups().iter().enumerate() {
        let overlaps = trusted_ts_ns_schema
            .then(|| row_group.columns().get(1))
            .flatten()
            .and_then(|column| exact_non_null_i64_bounds(column.statistics()))
            .map(|(min, max)| max >= start_ns && min < end_ns)
            .unwrap_or(true);
        if overlaps {
            selected.push(index);
        }
    }
    stats.selected_row_groups = selected.len();
    stats.pruned_row_groups = total_row_groups.saturating_sub(selected.len());
    Ok((selected, stats))
}

pub(super) fn parquet_column_matches_expected_i64<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    index: usize,
    expected_name: &str,
    nullable: bool,
) -> bool {
    let Some(arrow_field) = builder.schema().fields().get(index) else {
        return false;
    };
    let Some(parquet_column) = builder.parquet_schema().columns().get(index) else {
        return false;
    };
    arrow_field.name() == expected_name
        && arrow_field.data_type() == &DataType::Int64
        && arrow_field.is_nullable() == nullable
        && parquet_column.name() == expected_name
        && parquet_column.path().parts() == [expected_name]
        && parquet_column.physical_type() == PhysicalType::INT64
        && parquet_column.logical_type().is_none()
        && parquet_column.converted_type() == ConvertedType::NONE
        && parquet_column.max_rep_level() == 0
        && parquet_column.max_def_level() == i16::from(nullable)
}

pub(in crate::replay_cache) fn exact_non_null_i64_bounds(
    statistics: Option<&Statistics>,
) -> Option<(i64, i64)> {
    let Statistics::Int64(values) = statistics? else {
        return None;
    };
    if !values.min_is_exact() || !values.max_is_exact() || values.null_count_opt() != Some(0) {
        return None;
    }
    let (min, max) = (*values.min_opt()?, *values.max_opt()?);
    (min <= max).then_some((min, max))
}

/// Raw tick cache schema v2 relies on provider tick IDs being globally unique and
/// strictly increasing in timestamp order. Exact, non-null row-group statistics
/// prove that ID ranges are disjoint before any range pruning occurs. Within each
/// decoded group the stream enforces the same monotonic invariant in constant space.
pub(super) fn validate_raw_tick_id_row_groups<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    path: &Path,
) -> Result<()> {
    if !parquet_column_matches_expected_i64(builder, 2, "tick_id", false) {
        bail!(
            "raw tick parquet file {} cannot prove duplicate-id safety: expected required INT64 tick_id schema column",
            path.display()
        );
    }

    let mut previous_max = None;
    for (index, row_group) in builder.metadata().row_groups().iter().enumerate() {
        let (min, max) = row_group
            .columns()
            .get(2)
            .and_then(|column| exact_non_null_i64_bounds(column.statistics()))
            .with_context(|| {
                format!(
                    "raw tick parquet file {} cannot prove duplicate-id safety: row group {} lacks exact non-null tick_id statistics",
                    path.display(),
                    index
                )
            })?;
        if previous_max.is_some_and(|previous| min <= previous) {
            bail!(
                "raw tick parquet file {} has overlapping or non-monotonic tick-id ranges at row group {}: min {} follows {:?}",
                path.display(),
                index,
                min,
                previous_max
            );
        }
        previous_max = Some(max);
    }
    Ok(())
}

pub(in crate::replay_cache) fn timestamp_range_contains(
    range: Option<&ReplayCacheTimeRange>,
    ts_ns: i64,
) -> Result<bool> {
    range
        .copied()
        .map_or(Ok(true), |range| range.contains_ns(ts_ns))
}

pub(in crate::replay_cache) fn record_emitted_timestamp(
    stats: &mut ReplayCacheParquetReadStats,
    ts_ns: i64,
) {
    stats.emitted_rows = stats.emitted_rows.saturating_add(1);
    stats.first_timestamp_ns.get_or_insert(ts_ns);
    stats.last_timestamp_ns = Some(ts_ns);
}

#[allow(dead_code)]
pub fn stream_resolved_raw_ticks_parquet<F>(
    resolved: &ReplayCacheResolvedRawTicksFile,
    timestamp_range: Option<&ReplayCacheTimeRange>,
    on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let source = PositionIndependentFile::new(resolved.data_file.clone())?;
        let stats = stream_raw_ticks_parquet_reader(
            source,
            &resolved.data_path,
            timestamp_range,
            resolved.file.schema_version,
            on_tick,
        )?;
        validate_streamed_raw_ticks_metadata(resolved, &stats, timestamp_range)?;
        return Ok(stats);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (resolved, timestamp_range, on_tick);
        bail!(
            "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

pub fn stream_resolved_raw_ticks<F>(
    resolved: &ReplayCacheResolvedRawTicks,
    timestamp_range: Option<&ReplayCacheTimeRange>,
    mut on_tick: F,
) -> Result<ReplayCacheParquetReadStats>
where
    F: FnMut(ReplayCacheRawTickRow) -> Result<()>,
{
    let mut aggregate = ReplayCacheParquetReadStats::default();
    let mut previous_ts = None;
    let mut previous_id = None;
    let multi_file = resolved.files.len() > 1;
    for file in &resolved.files {
        if let Some(range) = timestamp_range
            && let (Some(start), Some(end)) = (file.file.request_start, file.file.request_end)
            && (end <= range.start || start >= range.end)
        {
            continue;
        }
        if multi_file && file.file.schema_version != Some(RAW_TICKS_SCHEMA_VERSION) {
            bail!("multi-file raw-tick replay requires schema-v2 chunk files");
        }
        let stats = stream_resolved_raw_ticks_parquet(file, timestamp_range, |row| {
            if previous_ts.is_some_and(|timestamp| row.ts_ns < timestamp) {
                bail!("multi-file raw-tick replay is not globally timestamp ordered");
            }
            let tick_id = row
                .tick_id
                .context("raw-tick replay row has no sequence id")?;
            if multi_file && previous_id.is_some_and(|id| tick_id <= id) {
                bail!("multi-file raw-tick replay is not globally tick-id ordered");
            }
            previous_ts = Some(row.ts_ns);
            previous_id = Some(tick_id);
            on_tick(row)
        })?;
        aggregate.total_row_groups = aggregate
            .total_row_groups
            .saturating_add(stats.total_row_groups);
        aggregate.selected_row_groups = aggregate
            .selected_row_groups
            .saturating_add(stats.selected_row_groups);
        aggregate.pruned_row_groups = aggregate
            .pruned_row_groups
            .saturating_add(stats.pruned_row_groups);
        aggregate.record_batches = aggregate
            .record_batches
            .saturating_add(stats.record_batches);
        aggregate.max_record_batch_rows = aggregate
            .max_record_batch_rows
            .max(stats.max_record_batch_rows);
        aggregate.decoded_rows = aggregate.decoded_rows.saturating_add(stats.decoded_rows);
        aggregate.emitted_rows = aggregate.emitted_rows.saturating_add(stats.emitted_rows);
        aggregate.first_timestamp_ns = aggregate.first_timestamp_ns.or(stats.first_timestamp_ns);
        aggregate.last_timestamp_ns = stats.last_timestamp_ns.or(aggregate.last_timestamp_ns);
    }
    Ok(aggregate)
}

pub(super) fn validate_streamed_raw_ticks_metadata(
    resolved: &ReplayCacheResolvedRawTicksFile,
    stats: &ReplayCacheParquetReadStats,
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<()> {
    if let Some(range) = timestamp_range {
        if let Some(first) = stats.first_timestamp_ns
            && !range.contains_ns(first)?
        {
            bail!(
                "raw tick cache file {} returned its first timestamp outside the requested range",
                resolved.data_path.display()
            );
        }
        if let Some(last) = stats.last_timestamp_ns
            && !range.contains_ns(last)?
        {
            bail!(
                "raw tick cache file {} returned its last timestamp outside the requested range",
                resolved.data_path.display()
            );
        }
        return Ok(());
    }
    if stats.emitted_rows == 0 {
        bail!(
            "raw tick cache file {} contained no usable ticks",
            resolved.data_path.display()
        );
    }
    if stats.emitted_rows != resolved.file.row_count {
        bail!(
            "raw tick cache row count mismatch for {}: manifest={} actual={}",
            resolved.data_path.display(),
            resolved.file.row_count,
            stats.emitted_rows
        );
    }
    let first_timestamp = DateTime::<Utc>::from_timestamp_nanos(
        stats
            .first_timestamp_ns
            .context("raw tick stream did not report a first timestamp")?,
    );
    let last_timestamp = DateTime::<Utc>::from_timestamp_nanos(
        stats
            .last_timestamp_ns
            .context("raw tick stream did not report a last timestamp")?,
    );
    if first_timestamp != resolved.file.first_timestamp
        || last_timestamp != resolved.file.last_timestamp
    {
        bail!(
            "raw tick cache timestamp range mismatch for {}: manifest={}..{} actual={}..{}",
            resolved.data_path.display(),
            resolved.file.first_timestamp,
            resolved.file.last_timestamp,
            first_timestamp,
            last_timestamp
        );
    }
    Ok(())
}

pub(super) fn validate_raw_tick_row(row: &ReplayCacheRawTickRow) -> Result<()> {
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

pub(super) fn validate_optional_positive(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value <= 0.0)
    {
        bail!("raw tick {label} must be finite and positive");
    }
    Ok(())
}

pub(super) fn validate_optional_nonnegative(value: Option<f64>, label: &str) -> Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        bail!("raw tick {label} must be finite and non-negative");
    }
    Ok(())
}

pub(in crate::replay_cache) fn raw_ticks_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Utf8, false),
        Field::new("ts_ns", DataType::Int64, false),
        Field::new("tick_id", DataType::Int64, false),
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
pub(in crate::replay_cache) fn parquet_column<'a, T: 'static>(
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
pub(super) fn optional_i64(array: &Int64Array, index: usize) -> Option<i64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
pub(super) fn optional_i32(array: &Int32Array, index: usize) -> Option<i32> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
pub(in crate::replay_cache) fn optional_f64(array: &Float64Array, index: usize) -> Option<f64> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index))
    }
}

#[allow(dead_code)]
pub(super) fn optional_string(array: &StringArray, index: usize) -> Option<String> {
    if array.is_null(index) {
        None
    } else {
        Some(array.value(index).to_string())
    }
}

use super::*;

#[test]
fn raw_tick_parquet_file_round_trips_rows() {
    let root = temp_cache_dir("raw-parquet-round-trip");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let rows = vec![
        raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
        raw_tick("2026-07-23T00:01:00Z", Some(2), 101.0),
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
fn raw_tick_stream_prunes_row_groups_and_uses_half_open_ranges() {
    let root = temp_cache_dir("raw-parquet-range");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let rows = vec![
        raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0),
        raw_tick("2026-07-23T00:00:01Z", Some(2), 100.25),
        raw_tick("2026-07-23T00:00:02Z", Some(3), 100.5),
        raw_tick("2026-07-23T00:00:03Z", Some(4), 100.75),
        raw_tick("2026-07-23T00:00:04Z", Some(5), 101.0),
        raw_tick("2026-07-23T00:00:05Z", Some(6), 101.25),
    ];
    write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1)
        .expect("write multi-row-group parquet");
    let range = ReplayCacheTimeRange::new(dt("2026-07-23T00:00:02Z"), dt("2026-07-23T00:00:04Z"))
        .expect("range");

    let (selected, stats) =
        read_raw_ticks_parquet_file_range(&path, Some(&range)).expect("stream range");

    assert_eq!(
        selected.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
        vec![Some(3), Some(4)]
    );
    assert_eq!(stats.total_row_groups, 3);
    assert_eq!(stats.selected_row_groups, 1);
    assert_eq!(stats.pruned_row_groups, 2);
    assert_eq!(stats.decoded_rows, 2);
    assert_eq!(stats.emitted_rows, 2);
    assert_eq!(read_raw_ticks_parquet_file(&path).expect("full read"), rows);
}

#[test]
fn parquet_pruning_bounds_fail_open_for_missing_inexact_and_reversed_statistics() {
    use parquet::file::statistics::ValueStatistics;

    assert_eq!(exact_non_null_i64_bounds(None), None);
    let missing = Statistics::int64(None, None, None, Some(0), false);
    assert_eq!(exact_non_null_i64_bounds(Some(&missing)), None);
    let inexact = Statistics::Int64(
        ValueStatistics::new(Some(10_i64), Some(20_i64), None, Some(0), false)
            .with_min_is_exact(false),
    );
    assert_eq!(exact_non_null_i64_bounds(Some(&inexact)), None);
    let reversed = Statistics::int64(Some(20), Some(10), None, Some(0), false);
    assert_eq!(exact_non_null_i64_bounds(Some(&reversed)), None);
    let nulls = Statistics::int64(Some(10), Some(20), None, Some(1), false);
    assert_eq!(exact_non_null_i64_bounds(Some(&nulls)), None);
    let trusted = Statistics::int64(Some(10), Some(20), None, Some(0), false);
    assert_eq!(exact_non_null_i64_bounds(Some(&trusted)), Some((10, 20)));
}

#[test]
fn parquet_pruning_decodes_all_groups_when_ts_ns_schema_or_statistics_are_uncertain() {
    let root = temp_cache_dir("parquet-pruning-fail-open");
    fs::create_dir_all(&root).expect("create temp dir");
    let rows = sequential_raw_ticks(4);
    let range = ReplayCacheTimeRange::new(dt("2026-07-24T00:00:00Z"), dt("2026-07-24T00:01:00Z"))
        .expect("range");

    let no_stats_path = root.join("no-stats.parquet");
    write_raw_ticks_with_schema(
        &no_stats_path,
        &rows,
        raw_ticks_parquet_schema(),
        EnabledStatistics::None,
        2,
    );
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        File::open(&no_stats_path).expect("open no-stats fixture"),
    )
    .expect("read no-stats metadata");
    let (selected, stats) =
        parquet_row_groups_for_range(&builder, Some(&range)).expect("select row groups");
    assert_eq!(selected, vec![0, 1]);
    assert_eq!(stats.pruned_row_groups, 0);

    let mut fields = raw_ticks_parquet_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[1] = Field::new("not_ts_ns", DataType::Int64, false);
    let wrong_schema = Arc::new(Schema::new(fields));
    let wrong_column_path = root.join("wrong-column.parquet");
    write_raw_ticks_with_schema(
        &wrong_column_path,
        &rows,
        wrong_schema,
        EnabledStatistics::Chunk,
        2,
    );
    let builder = ParquetRecordBatchReaderBuilder::try_new(
        File::open(&wrong_column_path).expect("open wrong-column fixture"),
    )
    .expect("read wrong-column metadata");
    let (selected, stats) =
        parquet_row_groups_for_range(&builder, Some(&range)).expect("select row groups");
    assert_eq!(selected, vec![0, 1]);
    assert_eq!(stats.pruned_row_groups, 0);
}

#[test]
fn raw_tick_stream_rejects_duplicate_ids_across_row_groups() {
    let root = temp_cache_dir("raw-parquet-cross-group-duplicate");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let mut rows = sequential_raw_ticks(4);
    rows[2].tick_id = rows[1].tick_id;
    write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1)
        .expect("write duplicate-id fixture");

    let err = stream_raw_ticks_parquet_file(&path, None, |_| Ok(()))
        .expect_err("cross-group duplicate id must fail closed");
    assert!(
        err.to_string()
            .contains("overlapping or non-monotonic tick-id ranges")
    );
}

#[test]
fn raw_tick_schema_v1_requires_manifest_provenance_and_v2_rejects_unprovable_ids() {
    let root = temp_cache_dir("raw-parquet-legacy-id-schema");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let rows = sequential_raw_ticks(2);
    let mut fields = raw_ticks_parquet_schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect::<Vec<_>>();
    fields[2] = Field::new("tick_id", DataType::Int64, true);
    write_raw_ticks_with_schema(
        &path,
        &rows,
        Arc::new(Schema::new(fields)),
        EnabledStatistics::Chunk,
        2,
    );
    let err = stream_raw_ticks_parquet_file(&path, None, |_| Ok(()))
        .expect_err("nullable legacy tick IDs require manifest provenance");
    assert!(
        err.to_string()
            .contains("requires a manifest-declared schema version")
    );

    let mut missing = rows.clone();
    missing[0].tick_id = None;
    assert!(
        validate_raw_tick_id_write_invariant(&missing)
            .expect_err("missing tick id")
            .to_string()
            .contains("schema v2 cache was not written")
    );
    let mut non_monotonic = rows;
    non_monotonic[1].tick_id = Some(0);
    assert!(
        validate_raw_tick_id_write_invariant(&non_monotonic)
            .expect_err("non-monotonic tick id")
            .to_string()
            .contains("strictly increasing")
    );
}

#[test]
fn schema_v1_manifest_hash_enables_sequence_ids_and_rejects_missing_or_modified_data() {
    let root = temp_cache_dir("raw-parquet-v1-compatibility");
    let rows = vec![
        raw_tick("2026-07-23T00:00:00Z", Some(900), 100.0),
        raw_tick("2026-07-23T00:00:01Z", None, 100.25),
        raw_tick("2026-07-23T00:00:02Z", Some(100), 100.5),
    ];
    let dataset = write_legacy_v1_raw_tick_dataset(&root, &rows);

    let resolved = dataset
        .resolve_raw_ticks_parquet_file(None)
        .expect("resolve hash-verified schema v1 cache");
    let mut replayed = Vec::new();
    stream_resolved_raw_ticks_parquet(&resolved, None, |row| {
        replayed.push(row);
        Ok(())
    })
    .expect("stream schema v1 cache");
    assert_eq!(
        replayed.iter().map(|row| row.tick_id).collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        replayed.iter().map(|row| row.price).collect::<Vec<_>>(),
        rows.iter().map(|row| row.price).collect::<Vec<_>>()
    );

    let mut missing_hash = dataset.clone();
    missing_hash.manifest.files[0].data_hash = None;
    let err = missing_hash
        .resolve_raw_ticks_parquet_file(None)
        .expect_err("schema v1 without writer hash must fail");
    assert!(err.to_string().contains("has no writer data hash"));
    assert!(err.to_string().contains("re-download"));

    let data_path = dataset
        .dataset_dir
        .join(&dataset.manifest.files[0].relative_path);
    OpenOptions::new()
        .append(true)
        .open(&data_path)
        .expect("open legacy file for corruption")
        .write_all(b"modified")
        .expect("modify legacy file");
    let err = dataset
        .resolve_raw_ticks_parquet_file(None)
        .expect_err("modified schema v1 cache must fail hash verification");
    assert!(err.to_string().contains("hash mismatch"));
    assert!(err.to_string().contains("re-download"));
}

#[test]
fn schema_v2_download_with_missing_provider_id_fails_before_cache_commit() {
    let root = temp_cache_dir("raw-parquet-v2-missing-provider-id");
    let err = write_raw_ticks_parquet_cache(raw_tick_cache_write(
        root.clone(),
        vec![raw_tick("2026-07-23T00:00:00Z", None, 100.0)],
    ))
    .expect_err("schema v2 must reject a missing provider tick id");

    assert!(
        err.to_string()
            .contains("provider response omitted stable tick IDs")
    );
    assert!(err.to_string().contains("schema v2 cache was not written"));
    assert!(!root.join(MANIFEST_FILE_NAME).exists());
    assert!(ReplayCacheLibrary::scan(root).datasets.is_empty());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn concurrent_derivations_share_an_unlinked_lease_without_cursor_races() {
    let root = temp_cache_dir("raw-parquet-concurrent-lease");
    let original_rows = sequential_raw_ticks(PARQUET_ROW_GROUP_ROWS + 1);
    write_raw_ticks_parquet_cache(raw_tick_cache_write(root.clone(), original_rows.clone()))
        .expect("write original raw tick cache");
    let resolved = ReplayCacheLibrary::scan(&root)
        .resolve_unique_raw_ticks_parquet(None)
        .expect("resolve original cache")
        .expect("raw tick dataset");
    let old_path = resolved.data_path.clone();

    let mut refreshed_rows = original_rows;
    for row in &mut refreshed_rows {
        row.price += 10.0;
    }
    let refreshed = write_raw_ticks_parquet_cache(raw_tick_cache_write(root, refreshed_rows))
        .expect("refresh raw tick cache");
    assert_ne!(refreshed.data_path, old_path);
    assert!(!old_path.exists(), "refresh should unlink the old pathname");

    let barrier = Arc::new(std::sync::Barrier::new(3));
    let mut readers = Vec::new();
    for _ in 0..2 {
        let resolved = resolved.clone();
        let barrier = barrier.clone();
        readers.push(std::thread::spawn(move || {
            barrier.wait();
            let mut count = 0_u64;
            let mut first = None;
            let mut last = None;
            stream_resolved_raw_ticks_parquet(&resolved, None, |row| {
                count += 1;
                first.get_or_insert((row.tick_id, row.price));
                last = Some((row.tick_id, row.price));
                Ok(())
            })?;
            Ok::<_, anyhow::Error>((count, first, last))
        }));
    }
    barrier.wait();

    for reader in readers {
        let (count, first, last) = reader
            .join()
            .expect("reader thread")
            .expect("concurrent position-independent derivation");
        assert_eq!(count, (PARQUET_ROW_GROUP_ROWS + 1) as u64);
        assert_eq!(first, Some((Some(1), 100.0)));
        assert_eq!(
            last,
            Some((
                Some((PARQUET_ROW_GROUP_ROWS + 1) as i64),
                100.0 + (PARQUET_ROW_GROUP_ROWS % 8) as f64 * 0.25
            ))
        );
    }
}

#[test]
fn raw_tick_stream_propagates_callback_failure_and_supports_empty_ranges() {
    let root = temp_cache_dir("raw-parquet-callback");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let rows = sequential_raw_ticks(4);
    write_raw_ticks_parquet_file_with_limits(&path, &rows, 2, 1).expect("write fixture");

    let mut callbacks = 0;
    let err = stream_raw_ticks_parquet_file(&path, None, |_| {
        callbacks += 1;
        if callbacks == 2 {
            bail!("synthetic callback failure");
        }
        Ok(())
    })
    .expect_err("callback failure must stop streaming");
    assert!(err.to_string().contains("synthetic callback failure"));
    assert_eq!(callbacks, 2);

    let range = ReplayCacheTimeRange::new(dt("2026-07-24T00:00:00Z"), dt("2026-07-24T00:01:00Z"))
        .expect("range");
    let (selected, stats) =
        read_raw_ticks_parquet_file_range(&path, Some(&range)).expect("empty range");
    assert!(selected.is_empty());
    assert_eq!(stats.selected_row_groups, 0);
    assert_eq!(stats.emitted_rows, 0);
}

#[test]
fn raw_tick_parquet_honors_production_row_group_and_batch_bounds() {
    let root = temp_cache_dir("raw-parquet-production-bounds");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("ticks.parquet");
    let rows = sequential_raw_ticks(PARQUET_ROW_GROUP_ROWS + 1);
    write_raw_ticks_parquet_file(&path, &rows).expect("write production-sized parquet");

    let mut emitted = 0_u64;
    let stats = stream_raw_ticks_parquet_file(&path, None, |_| {
        emitted += 1;
        Ok(())
    })
    .expect("stream production-sized parquet");
    assert_eq!(stats.total_row_groups, 2);
    assert_eq!(stats.record_batches, 9);
    assert!(stats.max_record_batch_rows <= PARQUET_READ_BATCH_ROWS);
    assert_eq!(stats.max_record_batch_rows, PARQUET_READ_BATCH_ROWS);
    assert_eq!(emitted, (PARQUET_ROW_GROUP_ROWS + 1) as u64);
}

#[test]
fn server_bar_range_reader_prunes_row_groups_and_rejects_malformed_rows() {
    let root = temp_cache_dir("server-parquet-range");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("bars.parquet");
    let rows = [
        bar("2026-07-23T00:00:00Z", 100.0),
        bar("2026-07-23T00:01:00Z", 101.0),
        bar("2026-07-23T00:02:00Z", 102.0),
        bar("2026-07-23T00:03:00Z", 103.0),
        bar("2026-07-23T00:04:00Z", 104.0),
        bar("2026-07-23T00:05:00Z", 105.0),
    ]
    .iter()
    .map(ReplayCacheServerBarRow::from_bar)
    .collect::<Vec<_>>();
    write_server_bars_parquet_file_with_limits(&path, &rows, 2, 1)
        .expect("write multi-row-group parquet");
    let range = ReplayCacheTimeRange::new(dt("2026-07-23T00:02:00Z"), dt("2026-07-23T00:04:00Z"))
        .expect("range");

    let (selected, stats) =
        read_server_bars_parquet_file_range(&path, Some(&range)).expect("read range");
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].open, 102.0);
    assert_eq!(selected[1].open, 103.0);
    assert_eq!(stats.pruned_row_groups, 2);

    let malformed_path = root.join("malformed-ticks.parquet");
    let mut malformed = raw_tick("2026-07-23T00:00:00Z", Some(1), 100.0);
    malformed.ts_ns += 1;
    write_raw_ticks_parquet_file_with_limits(&malformed_path, &[malformed], 1, 1)
        .expect("write malformed fixture");
    let err = stream_raw_ticks_parquet_file(&malformed_path, None, |_| Ok(()))
        .expect_err("malformed tick must fail");
    assert!(err.to_string().contains("validate raw tick row"));
}

#[test]
fn atomic_writer_preserves_existing_file_when_replacement_fails() {
    let root = temp_cache_dir("atomic-replacement");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join("manifest.json");
    fs::write(&path, b"working-cache").expect("write original");

    let err = write_file_atomically(&path, |file| {
        file.write_all(b"partial-replacement")?;
        bail!("synthetic write failure")
    })
    .expect_err("replacement must fail");

    assert!(err.to_string().contains("synthetic write failure"));
    assert_eq!(fs::read(&path).expect("read original"), b"working-cache");
    assert!(
        fs::read_dir(&root)
            .expect("read temp dir")
            .flatten()
            .all(|entry| !entry.file_name().to_string_lossy().contains(".tmp-"))
    );
}

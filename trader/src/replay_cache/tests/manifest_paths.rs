use super::*;
use serde_json::json;

#[test]
fn write_server_bars_jsonl_cache_writes_manifest_and_data_file() {
    let root = temp_cache_dir("write");
    let outcome = write_server_bars_jsonl_cache(ReplayCacheServerBarsWrite {
        cache_root: root.clone(),
        target: None,
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
        contract_metadata: None,
        session_template: Some("Globex".to_string()),
        bars: vec![bar("2026-07-23T00:00:00Z", 1.0)],
        warnings: vec!["synthetic fixture".to_string()],
        display_name: None,
        tags: None,
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

    let library = ReplayCacheLibrary::scan(root.clone());
    let loaded = library
        .load_first_server_bars_jsonl(BarType::minute(1), CandleMode::HeikinAshi, None)
        .expect("load first matching cached server bars")
        .expect("matching cached server bars");
    assert_eq!(loaded.bars.len(), 1);
    assert_eq!(loaded.bars[0].close, 1.5);
    assert_eq!(
        loaded.file.relative_path.parent(),
        Some(Path::new("server-bars"))
    );
    assert!(
        loaded
            .file
            .relative_path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| {
                name.starts_with("2026-07-23_to_2026-07-24_1minute_v") && name.ends_with(".jsonl")
            })
    );
}

#[test]
fn concurrent_manifest_writers_preserve_distinct_shapes_with_advisory_locking() {
    fn write_for(root: PathBuf, bar_type: BarType, price: f64) -> ReplayCacheWriteOutcome {
        write_server_bars_parquet_cache(ReplayCacheServerBarsWrite {
            cache_root: root,
            target: None,
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
            download_request: json!({"md": "getChart"}),
            bar_type,
            tick_specs: ReplayCacheTickSpecs {
                tick_size: 0.25,
                value_per_point: 5.0,
            },
            contract_metadata: None,
            session_template: Some("Globex".to_string()),
            bars: vec![bar("2026-07-23T00:00:00Z", price)],
            warnings: Vec::new(),
            display_name: None,
            tags: None,
            notes: None,
        })
        .expect("concurrent cache write")
    }

    let root = temp_cache_dir("concurrent-manifest-writers");
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let mut tasks = Vec::new();
    for (bar_type, price) in [(BarType::minute(1), 1.0), (BarType::volume(6500), 2.0)] {
        let root = root.clone();
        let barrier = barrier.clone();
        tasks.push(std::thread::spawn(move || {
            barrier.wait();
            write_for(root, bar_type, price)
        }));
    }
    barrier.wait();
    let outcomes = tasks
        .into_iter()
        .map(|task| task.join().expect("writer thread"))
        .collect::<Vec<_>>();

    let manifest =
        ReplayCacheManifest::from_path(&outcomes[0].manifest_path).expect("concurrent manifest");
    assert_eq!(manifest.files.len(), 2);
    assert!(manifest.available_bar_shapes.contains(&BarType::minute(1)));
    assert!(
        manifest
            .available_bar_shapes
            .contains(&BarType::volume(6500))
    );
    assert!(
        outcomes[0]
            .dataset_dir
            .join(MANIFEST_LOCK_FILE_NAME)
            .is_file()
    );
    for entry in fs::read_dir(&outcomes[0].dataset_dir).expect("dataset directory") {
        let name = entry
            .expect("dataset entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        assert!(!name.contains(".tmp-"), "orphan temporary file: {name}");
    }
    for entry in
        fs::read_dir(outcomes[0].dataset_dir.join("server-bars")).expect("server bars directory")
    {
        let name = entry
            .expect("server bar entry")
            .file_name()
            .to_string_lossy()
            .into_owned();
        assert!(!name.contains(".tmp-"), "orphan temporary file: {name}");
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn manifest_lock_contention_times_out_without_removing_the_stable_lock_file() {
    let root = temp_cache_dir("manifest-lock-timeout");
    fs::create_dir_all(&root).expect("lock directory");
    let lock_path = root.join(MANIFEST_LOCK_FILE_NAME);
    let lock = ReplayManifestLock::acquire(&root).expect("first lock");
    let started = Instant::now();
    let err = ReplayManifestLock::acquire_with_timeout(&root, Duration::from_millis(60))
        .err()
        .expect("contending lock should time out");

    assert!(err.to_string().contains("timed out waiting"));
    assert!(started.elapsed() >= Duration::from_millis(40));
    assert!(lock_path.exists());
    drop(lock);
    assert!(lock_path.exists());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn manifest_lock_waiter_acquires_after_raii_release() {
    let root = temp_cache_dir("manifest-lock-release");
    let lock = ReplayManifestLock::acquire(&root).expect("first lock");
    let ready = Arc::new(std::sync::Barrier::new(2));
    let ready_waiter = ready.clone();
    let waiter_root = root.clone();
    let waiter = std::thread::spawn(move || {
        ready_waiter.wait();
        ReplayManifestLock::acquire_with_timeout(&waiter_root, Duration::from_millis(500))
            .expect("waiter acquires released lock")
    });

    ready.wait();
    std::thread::sleep(Duration::from_millis(40));
    assert!(!waiter.is_finished());
    drop(lock);
    drop(waiter.join().expect("waiter thread"));
    assert!(root.join(MANIFEST_LOCK_FILE_NAME).is_file());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn manifest_lock_rejects_a_symlink_lock_file() {
    use std::os::unix::fs::symlink;

    let root = temp_cache_dir("manifest-lock-symlink");
    let outside = temp_cache_dir("manifest-lock-symlink-outside");
    fs::create_dir_all(&root).expect("lock directory");
    fs::create_dir_all(&outside).expect("outside directory");
    let outside_file = outside.join("outside.lock");
    fs::write(&outside_file, b"unchanged").expect("outside lock file");
    symlink(&outside_file, root.join(MANIFEST_LOCK_FILE_NAME)).expect("lock symlink");

    let err = ReplayManifestLock::acquire(&root)
        .err()
        .expect("symlink lock must be rejected");

    assert!(err.to_string().contains("open replay cache manifest lock"));
    assert_eq!(fs::read(&outside_file).expect("outside file"), b"unchanged");
}

#[cfg(unix)]
#[test]
fn generated_cache_dataset_rejects_symlink_path_escape_before_creation() {
    use std::os::unix::fs::symlink;

    let root = temp_cache_dir("generated-path-escape");
    let outside = temp_cache_dir("generated-path-outside");
    fs::create_dir_all(&root).expect("cache root");
    fs::create_dir_all(&outside).expect("outside root");
    symlink(&outside, root.join("tradovate")).expect("escape symlink");
    let canonical_root = fs::canonicalize(&root).expect("canonical root");
    let dataset_dir = replay_cache_dataset_dir(
        &canonical_root,
        BrokerKind::Tradovate,
        TradingEnvironment::Sim,
        "MES",
        "MESU6",
        NaiveDate::from_ymd_opt(2026, 7, 23).expect("date"),
    );

    let err = create_cache_dataset_without_symlinks(&canonical_root, &dataset_dir)
        .expect_err("symlink cache component must fail closed");

    assert!(err.to_string().contains("contains symlink"));
    assert!(!outside.join("sim").exists());
}

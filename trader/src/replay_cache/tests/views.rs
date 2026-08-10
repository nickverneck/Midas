use super::*;

fn naive(raw: &str) -> chrono::NaiveDateTime {
    chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%dT%H:%M:%S").expect("naive timestamp")
}

fn source_dataset(root: &Path) -> ReplayCacheDataset {
    let dataset_dir = root.join("tradovate/sim/MES/MESU6/2026-07-23");
    fs::create_dir_all(&dataset_dir).expect("create dataset");
    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let manifest = sample_manifest();
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
    ReplayCacheDataset {
        manifest_path,
        dataset_dir,
        manifest,
    }
}

fn sample_view(root: &Path, dataset: &ReplayCacheDataset) -> ReplayDatasetView {
    ReplayDatasetView::for_dataset(
        root,
        dataset,
        "mes_open",
        ReplayCacheTimeRange::new(dt("2026-07-23T14:00:00Z"), dt("2026-07-23T15:00:00Z"))
            .expect("range"),
        "America/New_York",
        ReplayDatasetSessionPreset::CustomLocal,
        ReplayDatasetWarmupPolicy {
            duration_seconds: 30 * 60,
            trading: ReplayWarmupTradingPolicy::FlatUntilEvaluation,
        },
    )
    .expect("view")
}

#[test]
fn dataset_view_round_trips_and_resolves_warmup_without_mutating_source() {
    let root = temp_cache_dir("view-round-trip");
    let dataset = source_dataset(&root);
    let source_before = fs::read(&dataset.manifest_path).expect("read source before");
    let store = ReplayDatasetViewStore::new(&root);
    let view = sample_view(&root, &dataset);

    let path = store.save(&view).expect("save view");
    let loaded = store.load("mes_open").expect("load view");
    let resolved = store.resolve("mes_open").expect("resolve view");

    assert_eq!(path, root.join(".views/mes_open.json"));
    assert_eq!(loaded, view);
    assert_eq!(resolved.evaluation_range.start, dt("2026-07-23T14:00:00Z"));
    assert_eq!(resolved.load_range.start, dt("2026-07-23T13:30:00Z"));
    assert_eq!(resolved.load_range.end, dt("2026-07-23T15:00:00Z"));
    assert_eq!(
        fs::read(&dataset.manifest_path).expect("read source after"),
        source_before
    );
    assert_eq!(
        fs::read_dir(&dataset.dataset_dir)
            .expect("dataset files")
            .count(),
        1
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn dataset_view_library_lists_only_valid_views_for_the_selected_source() {
    let root = temp_cache_dir("view-library");
    let dataset = source_dataset(&root);
    let store = ReplayDatasetViewStore::new(&root);
    let mut later = sample_view(&root, &dataset);
    later.id = "z_later".to_string();
    let mut earlier = sample_view(&root, &dataset);
    earlier.id = "a_earlier".to_string();
    store.save(&later).expect("save later");
    store.save(&earlier).expect("save earlier");
    fs::write(root.join(".views/broken.json"), b"not json").expect("write broken view");

    let library = store.list_for_dataset(&dataset);

    assert_eq!(
        library
            .views
            .iter()
            .map(|resolved| resolved.view.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a_earlier", "z_later"]
    );
    assert_eq!(library.warnings.len(), 1);

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn dataset_view_rejects_warmup_outside_completed_source_coverage() {
    let root = temp_cache_dir("view-warmup-coverage");
    let dataset = source_dataset(&root);
    let store = ReplayDatasetViewStore::new(&root);
    let mut view = sample_view(&root, &dataset);
    view.warmup.duration_seconds = 31 * 60;

    let error = store.save(&view).expect_err("warmup must fit");
    assert!(error.to_string().contains("including warmup"));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn dataset_view_rejects_invalid_timezone_and_unsafe_ids() {
    let root = temp_cache_dir("view-validation");
    let dataset = source_dataset(&root);
    let mut view = sample_view(&root, &dataset);
    view.input_timezone = "New_York-ish".to_string();
    assert!(
        view.validate_model()
            .expect_err("timezone")
            .to_string()
            .contains("timezone")
    );

    view.input_timezone = "UTC".to_string();
    view.id = "../escape".to_string();
    assert!(
        view.validate_model()
            .expect_err("id")
            .to_string()
            .contains("view id")
    );

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn dataset_view_detects_source_identity_replacement() {
    let root = temp_cache_dir("view-identity");
    let dataset = source_dataset(&root);
    let store = ReplayDatasetViewStore::new(&root);
    let view = sample_view(&root, &dataset);
    store.save(&view).expect("save view");

    let mut replacement = sample_manifest();
    replacement.contract.symbol = "GCZ6".to_string();
    fs::write(
        &dataset.manifest_path,
        serde_json::to_vec_pretty(&replacement).expect("serialize replacement"),
    )
    .expect("replace manifest");

    let error = store.resolve("mes_open").expect_err("identity mismatch");
    assert!(error.to_string().contains("identity no longer matches"));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn carry_warmup_position_is_explicitly_blocked_until_execution_support_exists() {
    let root = temp_cache_dir("view-carry-policy");
    let dataset = source_dataset(&root);
    let mut view = sample_view(&root, &dataset);
    view.warmup.trading = ReplayWarmupTradingPolicy::CarryWarmupPosition;

    let error = view.validate_model().expect_err("carry policy blocked");
    assert!(error.to_string().contains("not implemented"));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn full_source_selection_uses_exact_manifest_coverage() {
    let manifest = sample_manifest();
    let resolved = ReplayDatasetSessionSelection::FullSource
        .resolve(&manifest.coverage)
        .expect("full source");

    assert_eq!(resolved.preset, ReplayDatasetSessionPreset::FullSource);
    assert_eq!(resolved.input_timezone, "UTC");
    assert_eq!(resolved.evaluation_range.start, manifest.coverage.start);
    assert_eq!(resolved.evaluation_range.end, manifest.coverage.end);
}

#[test]
fn new_york_and_chicago_rth_follow_dst_and_resolve_to_the_same_utc_window() {
    let coverage = ReplayCacheCoverage {
        start: dt("2026-01-01T00:00:00Z"),
        end: dt("2027-01-01T00:00:00Z"),
        trading_date: None,
    };
    let before_dst = NaiveDate::from_ymd_opt(2026, 3, 6).unwrap();
    let after_dst = NaiveDate::from_ymd_opt(2026, 3, 9).unwrap();

    let ny_before = ReplayDatasetSessionSelection::FuturesRthNewYork {
        trading_date: before_dst,
    }
    .resolve(&coverage)
    .unwrap();
    let ny_after = ReplayDatasetSessionSelection::FuturesRthNewYork {
        trading_date: after_dst,
    }
    .resolve(&coverage)
    .unwrap();
    let chicago_after = ReplayDatasetSessionSelection::FuturesRthChicago {
        trading_date: after_dst,
    }
    .resolve(&coverage)
    .unwrap();

    assert_eq!(ny_before.evaluation_range.start, dt("2026-03-06T14:30:00Z"));
    assert_eq!(ny_before.evaluation_range.end, dt("2026-03-06T21:00:00Z"));
    assert_eq!(ny_after.evaluation_range.start, dt("2026-03-09T13:30:00Z"));
    assert_eq!(ny_after.evaluation_range.end, dt("2026-03-09T20:00:00Z"));
    assert_eq!(chicago_after.evaluation_range, ny_after.evaluation_range);
    assert_eq!(chicago_after.input_timezone, "America/Chicago");
}

#[test]
fn globex_monday_starts_sunday_and_friday_ends_before_the_weekend() {
    let coverage = ReplayCacheCoverage {
        start: dt("2026-01-01T00:00:00Z"),
        end: dt("2027-01-01T00:00:00Z"),
        trading_date: None,
    };
    let monday = ReplayDatasetSessionSelection::FuturesGlobex {
        trading_date: NaiveDate::from_ymd_opt(2026, 3, 9).unwrap(),
    }
    .resolve(&coverage)
    .unwrap();
    let friday = ReplayDatasetSessionSelection::FuturesGlobex {
        trading_date: NaiveDate::from_ymd_opt(2026, 3, 13).unwrap(),
    }
    .resolve(&coverage)
    .unwrap();

    assert_eq!(monday.evaluation_range.start, dt("2026-03-08T22:00:00Z"));
    assert_eq!(monday.evaluation_range.end, dt("2026-03-09T21:00:00Z"));
    assert_eq!(friday.evaluation_range.start, dt("2026-03-12T22:00:00Z"));
    assert_eq!(friday.evaluation_range.end, dt("2026-03-13T21:00:00Z"));
}

#[test]
fn futures_presets_reject_weekend_trading_dates() {
    let coverage = sample_manifest().coverage;
    let error = ReplayDatasetSessionSelection::FuturesGlobex {
        trading_date: NaiveDate::from_ymd_opt(2026, 3, 8).unwrap(),
    }
    .resolve(&coverage)
    .expect_err("Sunday is not a trading date");

    assert!(error.to_string().contains("Monday-Friday"));
}

#[test]
fn custom_local_selection_rejects_dst_gaps_and_ambiguous_times() {
    let coverage = sample_manifest().coverage;
    let gap = ReplayDatasetSessionSelection::CustomLocal {
        start: naive("2026-03-08T02:30:00"),
        end: naive("2026-03-08T04:00:00"),
        timezone: "America/New_York".to_string(),
    }
    .resolve(&coverage)
    .expect_err("spring-forward gap");
    let ambiguous = ReplayDatasetSessionSelection::CustomLocal {
        start: naive("2026-11-01T01:30:00"),
        end: naive("2026-11-01T03:00:00"),
        timezone: "America/New_York".to_string(),
    }
    .resolve(&coverage)
    .expect_err("fall-back ambiguity");

    assert!(gap.to_string().contains("does not exist"));
    assert!(ambiguous.to_string().contains("ambiguous"));
    assert!(ambiguous.to_string().contains("custom UTC"));
}

#[test]
fn preset_view_construction_persists_timezone_and_displays_zone_label() {
    let root = temp_cache_dir("view-rth-preset");
    let dataset = source_dataset(&root);
    let selection = ReplayDatasetSessionSelection::FuturesRthNewYork {
        trading_date: NaiveDate::from_ymd_opt(2026, 7, 23).unwrap(),
    };
    let view = ReplayDatasetView::from_session_selection(
        &root,
        &dataset,
        "rth_2026_07_23",
        &selection,
        ReplayDatasetWarmupPolicy::default(),
    )
    .expect("preset view");
    let store = ReplayDatasetViewStore::new(&root);
    store.save(&view).expect("save preset view");
    let loaded = store.load("rth_2026_07_23").expect("reload preset view");

    assert_eq!(loaded.input_timezone, "America/New_York");
    assert_eq!(loaded.evaluation_start, dt("2026-07-23T13:30:00Z"));
    assert_eq!(loaded.evaluation_end, dt("2026-07-23T20:00:00Z"));
    let label = loaded.evaluation_label().expect("display label");
    assert!(label.contains("Futures RTH (New York)"));
    assert!(label.contains("09:30:00 EDT"));
    assert!(label.contains("16:00:00 EDT"));

    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn preset_view_construction_rejects_a_session_outside_source_coverage() {
    let root = temp_cache_dir("view-preset-coverage");
    let dataset = source_dataset(&root);
    let selection = ReplayDatasetSessionSelection::FuturesRthNewYork {
        trading_date: NaiveDate::from_ymd_opt(2026, 7, 24).unwrap(),
    };

    let error = ReplayDatasetView::from_session_selection(
        &root,
        &dataset,
        "outside_source",
        &selection,
        ReplayDatasetWarmupPolicy::default(),
    )
    .expect_err("preset must fit source coverage");

    assert!(error.to_string().contains("outside source coverage"));
    fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn recurring_daily_session_filter_uses_new_york_clock_and_excludes_weekends() {
    let filter = ReplayDatasetDailySessionFilter {
        timezone: "America/New_York".to_string(),
        start_local: chrono::NaiveTime::from_hms_opt(8, 30, 0).unwrap(),
        end_local: chrono::NaiveTime::from_hms_opt(16, 15, 0).unwrap(),
    };
    assert!(
        filter
            .contains_timestamp(dt("2026-07-23T12:30:00Z").timestamp_nanos_opt().unwrap())
            .unwrap()
    );
    assert!(
        filter
            .contains_timestamp(dt("2026-07-23T20:14:00Z").timestamp_nanos_opt().unwrap())
            .unwrap()
    );
    assert!(
        !filter
            .contains_timestamp(dt("2026-07-23T20:15:00Z").timestamp_nanos_opt().unwrap())
            .unwrap()
    );
    assert!(
        !filter
            .contains_timestamp(dt("2026-07-23T12:29:00Z").timestamp_nanos_opt().unwrap())
            .unwrap()
    );
    assert!(
        !filter
            .contains_timestamp(dt("2026-07-25T14:00:00Z").timestamp_nanos_opt().unwrap())
            .unwrap()
    );
}

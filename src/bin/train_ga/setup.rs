use anyhow::{Context, Result, bail};
use midas_env::bars::BarSelection;
use midas_env::env::MarginMode;
use midas_env::ml::{self, ResolvedTrainingStack};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use std::path::PathBuf;

use crate::{
    args::Args,
    backends,
    config::{CandidateConfig, ExecutionTarget},
    data::{self, DataSet},
    evolution, util,
};

pub(crate) struct Datasets {
    pub(crate) train: DataSet,
    pub(crate) val: DataSet,
    pub(crate) test: DataSet,
}

pub(crate) struct WindowSets {
    pub(crate) train: Vec<(usize, usize)>,
    pub(crate) val: Vec<(usize, usize)>,
    pub(crate) test: Vec<(usize, usize)>,
}

pub(crate) struct MarginSettings {
    pub(crate) margin_mode: MarginMode,
    pub(crate) contract_multiplier: f64,
    pub(crate) margin_per_contract: f64,
}

pub(crate) struct RunResources {
    pub(crate) device: ExecutionTarget,
    pub(crate) behavior_dir: PathBuf,
    pub(crate) datasets: Datasets,
    pub(crate) windows: WindowSets,
    pub(crate) margin: MarginSettings,
    pub(crate) obs_dim: usize,
    pub(crate) genome_len: usize,
}

pub(crate) struct PopulationState {
    pub(crate) start_gen: usize,
    pub(crate) target_pop_size: usize,
    pub(crate) pop: Vec<Vec<f32>>,
    pub(crate) rng: StdRng,
}

pub(crate) struct EvolutionPlan {
    pub(crate) target_pop_size: usize,
    pub(crate) elite_n: usize,
    pub(crate) parent_pool_n: usize,
    pub(crate) immigrant_n: usize,
}

pub(crate) fn initialize_runtime(
    args: &Args,
    stack: &mut ResolvedTrainingStack,
) -> Result<(ExecutionTarget, PathBuf)> {
    if args.selection_use_eval && args.train_only_selection {
        bail!("--selection-use-eval and --train-only-selection are mutually exclusive");
    }
    if args.skip_val_eval && args.validation_participates_in_selection() {
        bail!(
            "validation-aware selection requires validation evaluation; remove --skip-val-eval or pass --train-only-selection"
        );
    }
    if args.skip_val_eval {
        println!(
            "warn: validation evaluation is disabled; no out-of-sample metric will be available during this run"
        );
    } else if args.validation_participates_in_selection() {
        println!(
            "info: validation is evaluated every generation and participates in policy selection"
        );
    } else {
        println!(
            "warn: validation is evaluated for diagnostics only; policy selection uses training fitness because --train-only-selection was set"
        );
    }

    std::fs::create_dir_all(&args.outdir)?;
    println!("info: run directory {}", args.outdir.display());
    let behavior_dir = args.outdir.join("behavior");
    std::fs::create_dir_all(&behavior_dir)?;

    let device = backends::resolve_device(stack)?;
    stack.effective_runtime = device.effective_runtime();
    ml::write_run_metadata(
        &args.outdir.join("training_stack.json"),
        &stack,
        Some("ga"),
        None,
    )?;
    println!(
        "info: effective runtime resolved to {}",
        stack.effective_runtime
    );
    backends::print_device(stack, device)?;

    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()
            .context("configure rayon thread pool")?;
        println!("info: using {} worker threads", args.workers);
    }

    Ok((device, behavior_dir))
}

pub(crate) fn load_run_resources(
    args: &Args,
    stack: &ResolvedTrainingStack,
    device: ExecutionTarget,
    behavior_dir: PathBuf,
) -> Result<RunResources> {
    let (train_path, val_path, test_path) = util::resolve_paths(args)?;
    validate_split_paths(args, &train_path, &val_path, &test_path)?;
    let train_symbol = data::read_symbol(&train_path)
        .with_context(|| format!("read symbol from {}", train_path.display()))?;
    let (margin_cfg, session_cfg) = util::load_symbol_config(&args.symbol_config, &train_symbol)?;
    let use_globex = if let Some(session) = session_cfg {
        match session.as_str() {
            "rth" => false,
            "globex" => true,
            _ => !args.rth,
        }
    } else {
        !args.rth
    };
    let bar_selection = BarSelection {
        bar_kind: args.bar_kind,
        volume_bar_size: args.volume_bar_size,
        price_source: args.price_source,
    };
    let train = data::load_dataset_with_bars(&train_path, use_globex, bar_selection)?;
    let val = data::load_dataset_with_bars(&val_path, use_globex, bar_selection)?;
    let test = data::load_dataset_with_bars(&test_path, use_globex, bar_selection)?;
    if args.debug_data {
        data::dump_dataset_stats("train", &train);
        data::dump_dataset_stats("val", &val);
        data::dump_dataset_stats("test", &test);
    }

    let margin_mode = match args.margin_mode.as_str() {
        "per-contract" => MarginMode::PerContract,
        "price" => MarginMode::Price,
        _ => util::infer_margin_mode(&train_symbol, margin_cfg),
    };
    let contract_multiplier = if args.contract_multiplier > 0.0 {
        args.contract_multiplier
    } else {
        1.0
    };
    let margin_per_contract = args
        .margin_per_contract
        .or(margin_cfg)
        .unwrap_or_else(|| util::infer_margin(&train_symbol));

    let train = train.with_session(use_globex);
    let val = val.with_session(use_globex);
    let test = test.with_session(use_globex);
    validate_loaded_splits(
        args,
        (&train_path, &train),
        (&val_path, &val),
        (&test_path, &test),
    )?;

    let full_file = if args.parquet.is_some() {
        args.full_file
    } else {
        args.full_file || !args.windowed
    };
    let raw_windows_train = if full_file {
        vec![(0, train.close.len())]
    } else {
        midas_env::sampler::windows(train.close.len(), args.window, args.step)
    };
    let raw_windows_val = if full_file {
        vec![(0, val.close.len())]
    } else {
        midas_env::sampler::windows(val.close.len(), args.window, args.step)
    };
    let raw_windows_test = if full_file {
        vec![(0, test.close.len())]
    } else {
        midas_env::sampler::windows(test.close.len(), args.window, args.step)
    };

    let feature_warmup = midas_env::features::feature_warmup_bars();
    let min_window_start = feature_warmup.saturating_sub(1);
    let windows = WindowSets {
        train: adjust_windows("train", raw_windows_train, feature_warmup, min_window_start),
        val: adjust_windows("val", raw_windows_val, feature_warmup, min_window_start),
        test: adjust_windows("test", raw_windows_test, feature_warmup, min_window_start),
    };

    require_windows("training", &windows.train, feature_warmup)?;
    require_windows("validation", &windows.val, feature_warmup)?;
    require_windows("test", &windows.test, feature_warmup)?;

    println!(
        "info: split validation passed | train {} bars/{} windows | val {} bars/{} windows | test {} bars/{} windows | eval cap {}",
        train.close.len(),
        windows.train.len(),
        val.close.len(),
        windows.val.len(),
        test.close.len(),
        windows.test.len(),
        if args.eval_windows == 0 {
            "all".to_string()
        } else {
            args.eval_windows.to_string()
        }
    );

    let obs_dim = train.obs_dim;
    let genome_len = backends::param_count(stack, obs_dim, args.hidden, args.layers)?;

    Ok(RunResources {
        device,
        behavior_dir,
        datasets: Datasets { train, val, test },
        windows,
        margin: MarginSettings {
            margin_mode,
            contract_multiplier,
            margin_per_contract,
        },
        obs_dim,
        genome_len,
    })
}

pub(crate) fn initialize_population(args: &Args, genome_len: usize) -> Result<PopulationState> {
    let mut rng = args
        .seed
        .map(StdRng::seed_from_u64)
        .unwrap_or_else(StdRng::from_entropy);
    let normal = Normal::<f32>::new(0.0, args.init_sigma as f32)?;
    let mut start_gen = 0usize;
    let mut target_pop_size = args.pop_size;
    let pop: Vec<Vec<f32>> = if let Some(checkpoint) = args.load_checkpoint.as_ref() {
        if !checkpoint.exists() {
            anyhow::bail!("checkpoint not found: {}", checkpoint.display());
        }
        let (resume_gen, loaded_pop) = evolution::load_checkpoint(checkpoint)
            .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;
        start_gen = resume_gen;
        if loaded_pop.is_empty() {
            anyhow::bail!("checkpoint population is empty");
        }
        if loaded_pop.len() != target_pop_size {
            println!(
                "warn: checkpoint population size {} does not match --pop-size {}; using {}",
                loaded_pop.len(),
                target_pop_size,
                loaded_pop.len()
            );
            target_pop_size = loaded_pop.len();
        }
        println!(
            "info: resuming from checkpoint {} at generation {}",
            checkpoint.display(),
            start_gen
        );
        loaded_pop
    } else {
        (0..target_pop_size)
            .map(|_| (0..genome_len).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    };

    if pop.iter().any(|genome| genome.len() != genome_len) {
        anyhow::bail!(
            "checkpoint genome length mismatch (expected {}, found {})",
            genome_len,
            pop.first().map(|genome| genome.len()).unwrap_or(0)
        );
    }

    Ok(PopulationState {
        start_gen,
        target_pop_size,
        pop,
        rng,
    })
}

pub(crate) fn build_candidate_config(
    args: &Args,
    device: ExecutionTarget,
    margin: &MarginSettings,
) -> CandidateConfig {
    CandidateConfig {
        initial_balance: args.initial_balance,
        max_position: args.max_position,
        margin_mode: margin.margin_mode,
        contract_multiplier: margin.contract_multiplier,
        margin_per_contract: margin.margin_per_contract,
        disable_margin: args.disable_margin,
        w_pnl: args.w_pnl,
        w_sortino: args.w_sortino,
        w_mdd: args.w_mdd,
        sortino_annualization: args.sortino_annualization,
        hidden: args.hidden,
        layers: args.layers,
        eval_windows: args.eval_windows,
        device,
        ignore_session: args.ignore_session,
        drawdown_penalty: args.drawdown_penalty,
        drawdown_penalty_growth: args.drawdown_penalty_growth,
        session_close_penalty: args.session_close_penalty,
        auto_close_minutes_before_close: args.auto_close_minutes_before_close,
        max_hold_bars_positive: args.max_hold_bars_positive,
        max_hold_bars_drawdown: args.max_hold_bars_drawdown,
        hold_duration_penalty: args.hold_duration_penalty,
        hold_duration_penalty_growth: args.hold_duration_penalty_growth,
        hold_duration_penalty_positive_scale: args.hold_duration_penalty_positive_scale,
        hold_duration_penalty_negative_scale: args.hold_duration_penalty_negative_scale,
        min_hold_bars: args.min_hold_bars,
        early_exit_penalty: args.early_exit_penalty,
        early_flip_penalty: args.early_flip_penalty,
        invalid_revert_penalty: args.invalid_revert_penalty,
        flat_hold_penalty: args.flat_hold_penalty,
        max_flat_hold_bars: args.max_flat_hold_bars,
        invalid_revert_penalty_growth: args.invalid_revert_penalty_growth,
        flat_hold_penalty_growth: args.flat_hold_penalty_growth,
    }
}

pub(crate) fn determine_batch_candidates(
    args: &Args,
    device: ExecutionTarget,
    pop_len: usize,
) -> usize {
    let mut batch_candidates = if args.batch_candidates > 0 {
        args.batch_candidates
    } else if device.is_accelerated() {
        pop_len
    } else {
        1
    };
    if batch_candidates == 0 {
        batch_candidates = 1;
    }
    batch_candidates.min(pop_len)
}

pub(crate) fn build_evolution_plan(args: &Args, target_pop_size: usize) -> EvolutionPlan {
    let elite_n = (args.elite_frac * target_pop_size as f64)
        .round()
        .clamp(1.0, target_pop_size as f64) as usize;
    let parent_pool_n = (args.parent_pool_frac * target_pop_size as f64)
        .round()
        .clamp(elite_n as f64, target_pop_size as f64) as usize;
    let immigrant_n = ((args.immigrant_frac * target_pop_size as f64).round() as usize)
        .min(target_pop_size.saturating_sub(elite_n));

    EvolutionPlan {
        target_pop_size,
        elite_n,
        parent_pool_n,
        immigrant_n,
    }
}

fn adjust_windows(
    label: &str,
    windows: Vec<(usize, usize)>,
    feature_warmup: usize,
    min_window_start: usize,
) -> Vec<(usize, usize)> {
    let before = windows.len();
    let adjusted = midas_env::sampler::enforce_min_start(&windows, min_window_start);
    let dropped = before.saturating_sub(adjusted.len());
    if dropped > 0 {
        println!(
            "info: dropped {} {} window(s) before feature warmup ({} bars)",
            dropped, label, feature_warmup
        );
    }
    adjusted
}

fn validate_split_paths(args: &Args, train: &PathBuf, val: &PathBuf, test: &PathBuf) -> Result<()> {
    let paths = [
        ("train", canonical_path(train)),
        ("validation", canonical_path(val)),
        ("test", canonical_path(test)),
    ];
    let mut overlaps = Vec::new();
    for left in 0..paths.len() {
        for right in (left + 1)..paths.len() {
            if paths[left].1 == paths[right].1 {
                overlaps.push(format!("{}={}", paths[left].0, paths[right].0));
            }
        }
    }

    if !overlaps.is_empty() {
        let message = format!(
            "{} split(s) resolve to the same parquet ({}); this is not an out-of-sample validation/test split",
            overlaps.len(),
            overlaps.join(", ")
        );
        if args.allow_overlapping_splits {
            println!(
                "warn: {message}; continuing because --allow-overlapping-splits was explicitly set"
            );
        } else {
            bail!(
                "{message}. Provide distinct chronological files, or use --allow-overlapping-splits only for an intentional smoke test"
            );
        }
    }

    Ok(())
}

fn validate_loaded_splits(
    args: &Args,
    train: (&PathBuf, &DataSet),
    val: (&PathBuf, &DataSet),
    test: (&PathBuf, &DataSet),
) -> Result<()> {
    let splits = [("train", train), ("validation", val), ("test", test)];
    let mut spans = Vec::with_capacity(splits.len());
    let mut obs_dim = None;
    let mut feature_count = None;
    let mut known_symbols = Vec::new();

    for (label, (path, dataset)) in splits {
        validate_dataset_shape(label, path, dataset)?;
        if let Some(expected) = obs_dim {
            if dataset.obs_dim != expected {
                bail!(
                    "{label} dataset {} has obs_dim {}, but the training dataset has obs_dim {}; feature schemas must match",
                    path.display(),
                    dataset.obs_dim,
                    expected
                );
            }
        } else {
            obs_dim = Some(dataset.obs_dim);
        }
        if let Some(expected) = feature_count {
            if dataset.feature_cols.len() != expected {
                bail!(
                    "{label} dataset {} has {} feature columns, but the training dataset has {}; feature schemas must match",
                    path.display(),
                    dataset.feature_cols.len(),
                    expected
                );
            }
        } else {
            feature_count = Some(dataset.feature_cols.len());
        }

        if dataset.symbol != "UNKNOWN" {
            known_symbols.push((label, dataset.symbol.as_str()));
        }
        spans.push((label, timestamp_span(label, path, dataset)?));
    }

    let symbols_match = known_symbols
        .first()
        .map(|(_, symbol)| known_symbols.iter().all(|(_, other)| other == symbol))
        .unwrap_or(true);
    if !symbols_match {
        println!(
            "warn: split symbols differ: {}; verify that this contract/instrument mix is intentional",
            known_symbols
                .iter()
                .map(|(label, symbol)| format!("{label}={symbol}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let timestamp_spans: Option<Vec<(i64, i64)>> = spans.iter().map(|(_, span)| *span).collect();
    if let Some(spans) = timestamp_spans {
        let chronological = spans[0].1 < spans[1].0 && spans[1].1 < spans[2].0;
        if chronological {
            println!(
                "info: chronological split confirmed | train {}..{} | val {}..{} | test {}..{}",
                spans[0].0, spans[0].1, spans[1].0, spans[1].1, spans[2].0, spans[2].1
            );
        } else if args.allow_overlapping_splits {
            println!(
                "warn: split timestamps overlap or are out of order; continuing because --allow-overlapping-splits was explicitly set"
            );
        } else {
            bail!(
                "split timestamps are not strictly chronological/non-overlapping (train {}..{}, val {}..{}, test {}..{}); provide clean walk-forward files or use --allow-overlapping-splits only for a smoke test",
                spans[0].0,
                spans[0].1,
                spans[1].0,
                spans[1].1,
                spans[2].0,
                spans[2].1
            );
        }
    } else {
        println!(
            "warn: one or more split datasets has no usable date column; chronological separation could not be verified"
        );
    }

    Ok(())
}

fn validate_dataset_shape(label: &str, path: &PathBuf, dataset: &DataSet) -> Result<()> {
    let bars = dataset.close.len();
    if bars < 2 {
        bail!(
            "{label} dataset {} has only {bars} bar(s); at least two are required for evaluation",
            path.display()
        );
    }

    for (name, len) in [
        ("open", dataset.open.len()),
        ("high", dataset._high.len()),
        ("low", dataset._low.len()),
        ("signal_open", dataset.signal_open.len()),
        ("signal_close", dataset.signal_close.len()),
        ("margin_ok", dataset.margin_ok.len()),
    ] {
        if len != bars {
            bail!(
                "{label} dataset {} has {name} length {len}, expected {bars}; refusing to train on misaligned rows",
                path.display()
            );
        }
    }
    if let Some(volume) = &dataset.volume {
        if volume.len() != bars {
            bail!(
                "{label} dataset {} has volume length {}, expected {bars}",
                path.display(),
                volume.len()
            );
        }
    }
    if let Some(datetime_ns) = &dataset.datetime_ns {
        if datetime_ns.len() != bars {
            bail!(
                "{label} dataset {} has date length {}, expected {bars}",
                path.display(),
                datetime_ns.len()
            );
        }
    }
    if let Some(session_open) = &dataset.session_open {
        if session_open.len() != bars {
            bail!(
                "{label} dataset {} has session_open length {}, expected {bars}",
                path.display(),
                session_open.len()
            );
        }
    }
    if let Some(minutes_to_close) = &dataset.minutes_to_close {
        if minutes_to_close.len() != bars {
            bail!(
                "{label} dataset {} has minutes_to_close length {}, expected {bars}",
                path.display(),
                minutes_to_close.len()
            );
        }
    }
    for (index, feature) in dataset.feature_cols.iter().enumerate() {
        if feature.len() != bars {
            bail!(
                "{label} dataset {} feature column {index} has length {}, expected {bars}",
                path.display(),
                feature.len()
            );
        }
    }
    if dataset.obs_dim == 0 {
        bail!(
            "{label} dataset {} produced an empty observation schema",
            path.display()
        );
    }
    if dataset.close.iter().any(|value| !value.is_finite()) {
        bail!(
            "{label} dataset {} contains a non-finite close value",
            path.display()
        );
    }
    Ok(())
}

fn timestamp_span(label: &str, path: &PathBuf, dataset: &DataSet) -> Result<Option<(i64, i64)>> {
    let Some(datetime_ns) = dataset.datetime_ns.as_ref() else {
        return Ok(None);
    };
    if datetime_ns.is_empty() {
        bail!(
            "{label} dataset {} has an empty date column",
            path.display()
        );
    }
    for pair in datetime_ns.windows(2) {
        if pair[1] <= pair[0] {
            bail!(
                "{label} dataset {} has non-increasing timestamps at {} and {}",
                path.display(),
                pair[0],
                pair[1]
            );
        }
    }
    Ok(Some((datetime_ns[0], *datetime_ns.last().unwrap())))
}

fn canonical_path(path: &PathBuf) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.clone())
}

fn require_windows(label: &str, windows: &[(usize, usize)], feature_warmup: usize) -> Result<()> {
    if windows.is_empty() {
        bail!("no {label} windows available after applying feature warmup ({feature_warmup} bars)");
    }
    Ok(())
}

use anyhow::{Context, Result};
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
    let train = data::load_dataset(&train_path, args.globex && !args.rth)?;
    let val = data::load_dataset(&val_path, args.globex && !args.rth)?;
    let test = data::load_dataset(&test_path, args.globex && !args.rth)?;
    if args.debug_data {
        data::dump_dataset_stats("train", &train);
        data::dump_dataset_stats("val", &val);
        data::dump_dataset_stats("test", &test);
    }

    let (margin_cfg, session_cfg) = util::load_symbol_config(&args.symbol_config, &train.symbol)?;
    let margin_mode = match args.margin_mode.as_str() {
        "per-contract" => MarginMode::PerContract,
        "price" => MarginMode::Price,
        _ => util::infer_margin_mode(&train.symbol, margin_cfg),
    };
    let contract_multiplier = if args.contract_multiplier > 0.0 {
        args.contract_multiplier
    } else {
        1.0
    };
    let margin_per_contract = args
        .margin_per_contract
        .or(margin_cfg)
        .unwrap_or_else(|| util::infer_margin(&train.symbol));
    let use_globex = if let Some(session) = session_cfg {
        match session.as_str() {
            "rth" => false,
            "globex" => true,
            _ => !args.rth,
        }
    } else {
        !args.rth
    };

    let train = train.with_session(use_globex);
    let val = val.with_session(use_globex);
    let test = test.with_session(use_globex);

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

    if windows.train.is_empty() {
        anyhow::bail!(
            "no training windows available after applying feature warmup ({} bars)",
            feature_warmup
        );
    }

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

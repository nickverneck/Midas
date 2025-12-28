use std::path::{Path, PathBuf};

#[cfg(not(feature = "tch"))]
fn main() {
    eprintln!("train_ga requires the 'tch' feature. Rebuild with --features tch and ensure LIBTORCH is set.");
    std::process::exit(1);
}

#[cfg(feature = "tch")]
use clap::Parser;

#[cfg(feature = "tch")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    run(args)
}

#[cfg(feature = "tch")]
#[derive(Parser, Debug, Clone)]
#[command(about = "Rust GA-only neuroevolution trainer (CUDA via libtorch)")]
struct Args {
    #[arg(long)]
    parquet: Option<PathBuf>,
    #[arg(long, default_value = "data/train")]
    train_parquet: PathBuf,
    #[arg(long, default_value = "data/val")]
    val_parquet: PathBuf,
    #[arg(long, default_value = "data/test")]
    test_parquet: PathBuf,
    #[arg(long)]
    full_file: bool,
    #[arg(long, default_value_t = 512)]
    window: usize,
    #[arg(long, default_value_t = 256)]
    step: usize,
    #[arg(long)]
    device: Option<String>,
    #[arg(long, default_value_t = true)]
    globex: bool,
    #[arg(long)]
    rth: bool,
    #[arg(long, default_value_t = 10000.0)]
    initial_balance: f64,
    #[arg(long)]
    margin_per_contract: Option<f64>,
    #[arg(long, default_value = "config/symbols.yaml")]
    symbol_config: PathBuf,
    #[arg(long, default_value = "runs_ga")]
    outdir: PathBuf,

    #[arg(long, default_value_t = 5)]
    generations: usize,
    #[arg(long, default_value_t = 6)]
    pop_size: usize,
    #[arg(long, default_value_t = 0)]
    workers: usize,
    #[arg(long, default_value_t = 0.33)]
    elite_frac: f64,
    #[arg(long, default_value_t = 0.05)]
    mutation_sigma: f64,
    #[arg(long, default_value_t = 0.5)]
    init_sigma: f64,
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,

    #[arg(long, default_value_t = 2)]
    eval_windows: usize,
    #[arg(long, default_value_t = 1.0)]
    w_pnl: f64,
    #[arg(long, default_value_t = 1.0)]
    w_sortino: f64,
    #[arg(long, default_value_t = 0.5)]
    w_mdd: f64,
    #[arg(long, default_value_t = 1.0)]
    sortino_annualization: f64,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    disable_margin: bool,
    #[arg(long)]
    skip_val_eval: bool,
}

#[cfg(feature = "tch")]
fn run(mut args: Args) -> anyhow::Result<()> {
    use anyhow::Context;
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use rayon::prelude::*;
    use std::io::Write;

    if let Some(seed) = args.seed {
        let mut rng = StdRng::seed_from_u64(seed);
        let _ = rng.r#gen::<u64>();
    }

    args.outdir = args.outdir.clone();
    std::fs::create_dir_all(&args.outdir)?;

    let device = resolve_device(args.device.as_deref());
    print_device(&device);
    if args.workers > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.workers)
            .build_global()
            .context("configure rayon thread pool")?;
        println!("🧵 Using {} worker threads", args.workers);
    }

    let (train_path, val_path, test_path) = resolve_paths(&args)?;
    let train = load_dataset(&train_path, args.globex && !args.rth)?;
    let val = load_dataset(&val_path, args.globex && !args.rth)?;
    let test = load_dataset(&test_path, args.globex && !args.rth)?;

    let (margin_cfg, session_cfg) = load_symbol_config(&args.symbol_config, &train.symbol)?;
    let margin_per_contract = args.margin_per_contract.or(margin_cfg).unwrap_or_else(|| infer_margin(&train.symbol));
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

    let windows_train = if args.full_file {
        vec![(0, train.close.len())]
    } else {
        midas_env::sampler::windows(train.close.len(), args.window, args.step)
    };
    let windows_val = if args.full_file {
        vec![(0, val.close.len())]
    } else {
        midas_env::sampler::windows(val.close.len(), args.window, args.step)
    };
    let windows_test = if args.full_file {
        vec![(0, test.close.len())]
    } else {
        midas_env::sampler::windows(test.close.len(), args.window, args.step)
    };

    if windows_train.is_empty() {
        anyhow::bail!("no training windows available");
    }

    let obs_dim = train.obs_dim;
    let genome_len = param_count(obs_dim, args.hidden, args.layers);

    let mut rng = StdRng::from_entropy();
    let normal = Normal::<f32>::new(0.0, args.init_sigma as f32)?;
    let mut pop: Vec<Vec<f32>> = (0..args.pop_size)
        .map(|_| (0..genome_len).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    let log_path = args.outdir.join("ga_log.csv");
    if !log_path.exists() {
        std::fs::write(
            &log_path,
            "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_pnl,eval_sortino,eval_drawdown,eval_ret_mean,train_pnl,train_sortino,train_drawdown,train_ret_mean\n",
        )?;
    }

    let mut best_overall_fitness = f64::NEG_INFINITY;
    let mut best_overall_genome: Option<Vec<f32>> = None;

    for generation in 0..args.generations {
        let gen_start = std::time::Instant::now();
        println!(
            "\n🚀 Generation {} | Evaluating {} candidates (device={:?})",
            generation,
            pop.len(),
            device
        );

        let eval_results: Vec<(usize, CandidateResult, Option<CandidateResult>)> = pop
            .par_iter()
            .enumerate()
            .map(|(idx, genome)| {
                let train_metrics = evaluate_candidate(
                    genome,
                    &train,
                    &windows_train,
                    &CandidateConfig {
                        initial_balance: args.initial_balance,
                        margin_per_contract,
                        disable_margin: args.disable_margin,
                        w_pnl: args.w_pnl,
                        w_sortino: args.w_sortino,
                        w_mdd: args.w_mdd,
                        sortino_annualization: args.sortino_annualization,
                        hidden: args.hidden,
                        layers: args.layers,
                        eval_windows: args.eval_windows,
                        device,
                    },
                    false,
                );
                let eval_metrics = if args.skip_val_eval {
                    None
                } else {
                    Some(evaluate_candidate(
                        genome,
                        &val,
                        &windows_val,
                        &CandidateConfig {
                            initial_balance: args.initial_balance,
                            margin_per_contract,
                            disable_margin: args.disable_margin,
                            w_pnl: args.w_pnl,
                            w_sortino: args.w_sortino,
                            w_mdd: args.w_mdd,
                            sortino_annualization: args.sortino_annualization,
                            hidden: args.hidden,
                            layers: args.layers,
                            eval_windows: args.eval_windows,
                            device,
                        },
                        true,
                    ))
                };
                (idx, train_metrics, eval_metrics)
            })
            .collect();

        let mut scored: Vec<(f64, Vec<f32>, Option<CandidateResult>, CandidateResult)> = Vec::with_capacity(pop.len());

        let mut results = eval_results;
        results.sort_by_key(|(idx, _, _)| *idx);

        for (idx, train_metrics, eval_metrics) in results.into_iter() {
            let genome = pop[idx].clone();
            scored.push((train_metrics.fitness, genome.clone(), eval_metrics.clone(), train_metrics.clone()));

            let eval_pnl = eval_metrics.as_ref().map(|m| format!("{:.4}", m.eval_pnl)).unwrap_or_default();
            let eval_sortino = eval_metrics.as_ref().map(|m| format!("{:.4}", m.eval_sortino)).unwrap_or_default();
            let eval_dd = eval_metrics.as_ref().map(|m| format!("{:.4}", m.eval_drawdown)).unwrap_or_default();
            let eval_ret = eval_metrics.as_ref().map(|m| format!("{:.4}", m.eval_ret_mean)).unwrap_or_default();

            let line = format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{:.4},{:.4},{:.4},{:.4}\n",
                generation,
                idx,
                args.w_pnl,
                args.w_sortino,
                args.w_mdd,
                train_metrics.fitness,
                eval_pnl,
                eval_sortino,
                eval_dd,
                eval_ret,
                train_metrics.eval_pnl,
                train_metrics.eval_sortino,
                train_metrics.eval_drawdown,
                train_metrics.eval_ret_mean
            );
            std::fs::OpenOptions::new()
                .append(true)
                .open(&log_path)?
                .write_all(line.as_bytes())
                .context("write ga_log")?;

            if let Some(eval) = eval_metrics {
                println!(
                    "  ✅ cand {}/{} | fitness {:.2} | train pnl {:.2} | val pnl {:.2}",
                    idx,
                    pop.len().saturating_sub(1),
                    train_metrics.fitness,
                    train_metrics.eval_pnl,
                    eval.eval_pnl
                );
            } else {
                println!(
                    "  ✅ cand {}/{} | fitness {:.2} | train pnl {:.2} | val skipped",
                    idx,
                    pop.len().saturating_sub(1),
                    train_metrics.fitness,
                    train_metrics.eval_pnl
                );
            }
        }

        println!(
            "⏱ Generation {} completed in {:.2?}",
            generation,
            gen_start.elapsed()
        );

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let top = scored.iter().take(5).collect::<Vec<_>>();
        for (rank, (_fit, genome, eval_metrics, train_metrics)) in top.iter().enumerate() {
            let metrics = eval_metrics.as_ref().unwrap_or(train_metrics);
            let policy_path = args
                .outdir
                .join(format!("policy_gen{}_rank{}.pt", generation, rank));
            save_policy(
                obs_dim,
                args.hidden,
                args.layers,
                device,
                genome,
                &policy_path,
            )?;
            if rank == 0 {
                println!(
                    "📈 Gen {} Top Performer: fitness {:.2}",
                    generation, metrics.fitness
                );
                if metrics.fitness > best_overall_fitness {
                    best_overall_fitness = metrics.fitness;
                    best_overall_genome = Some(genome.clone());
                }
            }
        }

        let elite_n = (args.elite_frac * args.pop_size as f64).round().max(1.0) as usize;
        let elites: Vec<Vec<f32>> = scored.iter().take(elite_n).map(|(_, g, _, _)| g.clone()).collect();
        let mut new_pop = elites.clone();

        let normal_mut = Normal::<f32>::new(0.0, args.mutation_sigma as f32)?;
        while new_pop.len() < args.pop_size {
            let parent_a = elites.choose(&mut rng).unwrap();
            let parent_b = elites.choose(&mut rng).unwrap();
            let mut child = crossover(parent_a, parent_b, &mut rng);
            for v in child.iter_mut() {
                *v += normal_mut.sample(&mut rng);
            }
            new_pop.push(child);
        }
        pop = new_pop;

        let ckpt_path = args.outdir.join(format!("checkpoint_gen{}.bin", generation));
        save_checkpoint(&ckpt_path, &pop)?;
    }

    if let Some(best_genome) = best_overall_genome {
        let metrics = evaluate_candidate(
            &best_genome,
            &test,
            &windows_test,
            &CandidateConfig {
                initial_balance: args.initial_balance,
                margin_per_contract,
                disable_margin: args.disable_margin,
                w_pnl: args.w_pnl,
                w_sortino: args.w_sortino,
                w_mdd: args.w_mdd,
                sortino_annualization: args.sortino_annualization,
                hidden: args.hidden,
                layers: args.layers,
                eval_windows: args.eval_windows,
                device,
            },
            false,
        );
        println!(
            "test | fitness {:.2} | pnl {:.2} | sortino {:.2} | mdd {:.2}",
            metrics.fitness, metrics.eval_pnl, metrics.eval_sortino, metrics.eval_drawdown
        );

        let best_path = args.outdir.join("best_overall_policy.pt");
        save_policy(obs_dim, args.hidden, args.layers, device, &best_genome, &best_path)?;
        println!("🏆 Saved best overall policy to {}", best_path.display());
    }

    Ok(())
}

#[cfg(feature = "tch")]
fn resolve_device(requested: Option<&str>) -> tch::Device {
    use tch::Device;
    match requested.unwrap_or("") {
        "cuda" | "cuda:0" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
        "mps" => {
            let mps_device = Device::Mps;
            let mps_ok = std::panic::catch_unwind(|| {
                let _t = tch::Tensor::zeros(&[1], (tch::Kind::Float, mps_device));
            })
            .is_ok();
            if mps_ok {
                mps_device
            } else {
                Device::Cpu
            }
        }
        "cpu" => Device::Cpu,
        _ => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
    }
}

#[cfg(feature = "tch")]
fn print_device(device: &tch::Device) {
    if let tch::Device::Cuda(idx) = device {
        println!("🟢 Using CUDA device: cuda:{}", idx);
    } else if let tch::Device::Mps = device {
        println!("🟢 Using MPS device");
    } else {
        println!("🟡 Using device: {:?} (CUDA available={})", device, tch::Cuda::is_available());
    }
}

#[cfg(feature = "tch")]
fn resolve_paths(args: &Args) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let resolve_path = |path: &Path, fallback: &Path| -> PathBuf {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = path
                .read_dir()
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
                .collect();
            entries.sort();
            return entries.first().cloned().unwrap_or_else(|| fallback.to_path_buf());
        }
        if path.exists() {
            path.to_path_buf()
        } else {
            fallback.to_path_buf()
        }
    };

    if let Some(p) = &args.parquet {
        Ok((p.clone(), p.clone(), p.clone()))
    } else {
        let train = resolve_path(&args.train_parquet, &args.train_parquet);
        let val = resolve_path(&args.val_parquet, &train);
        let test = resolve_path(&args.test_parquet, &val);
        Ok((train, val, test))
    }
}

#[cfg(feature = "tch")]
fn load_symbol_config(path: &Path, symbol: &str) -> anyhow::Result<(Option<f64>, Option<String>)> {
    if !path.exists() {
        return Ok((None, None));
    }
    let text = std::fs::read_to_string(path)?;
    let cfg: serde_yaml::Value = serde_yaml::from_str(&text)?;
    if let Some(entry) = cfg.get(symbol) {
        let margin = entry.get("margin_per_contract").and_then(|v| v.as_f64());
        let session = entry.get("session").and_then(|v| v.as_str()).map(|s| s.to_ascii_lowercase());
        Ok((margin, session))
    } else {
        Ok((None, None))
    }
}

#[cfg(feature = "tch")]
fn infer_margin(symbol: &str) -> f64 {
    let sym = symbol.to_ascii_uppercase();
    if sym.contains("MES") {
        return 50.0;
    }
    if sym == "ES" || sym.contains("ES@") || sym.ends_with("ES") {
        return 500.0;
    }
    100.0
}

#[cfg(feature = "tch")]
#[derive(Clone)]
struct DataSet {
    open: Vec<f64>,
    close: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    volume: Option<Vec<f64>>,
    datetime_ns: Option<Vec<i64>>,
    session_open: Option<Vec<bool>>,
    margin_ok: Vec<bool>,
    feature_cols: Vec<Vec<f64>>,
    obs_dim: usize,
    symbol: String,
}

#[cfg(feature = "tch")]
impl DataSet {
    fn with_session(mut self, globex: bool) -> Self {
        if let Some(dt) = &self.datetime_ns {
            self.session_open = Some(build_session_mask(dt, globex));
        }
        self
    }
}

#[cfg(feature = "tch")]
fn load_dataset(path: &Path, globex: bool) -> anyhow::Result<DataSet> {
    use polars::prelude::*;

    let file = std::fs::File::open(path)?;
    let df = polars::prelude::ParquetReader::new(file).finish()?;
    let open = if df.column("open").is_ok() {
        series_to_f64(df.column("open")?.as_materialized_series())?
    } else {
        series_to_f64(df.column("close")?.as_materialized_series())?
    };
    let close = series_to_f64(df.column("close")?.as_materialized_series())?;
    let high = series_to_f64(df.column("high")?.as_materialized_series())?;
    let low = series_to_f64(df.column("low")?.as_materialized_series())?;
    let volume = df
        .column("volume")
        .ok()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .transpose()?;
    let datetime_ns: Option<Vec<i64>> = df
        .column("date")
        .ok()
        .map(|c| series_to_i64(c.as_materialized_series()))
        .transpose()?;
    let symbol = match df.column("symbol")?.get(0)? {
        polars::prelude::AnyValue::String(s) => s.to_string(),
        _ => "UNKNOWN".to_string(),
    };

    let feats = midas_env::features::compute_features_ohlcv(
        &close,
        Some(&high),
        Some(&low),
        volume.as_deref(),
    );
    let feature_cols = ordered_feature_cols(feats)?;

    let session_open = datetime_ns.as_ref().map(|dt| build_session_mask(dt, globex));
    let margin_ok = vec![true; close.len()];

    let obs_dim = observation_len(&open, &close, volume.as_deref(), &feature_cols);

    Ok(DataSet {
        open,
        close,
        high,
        low,
        volume,
        datetime_ns,
        session_open,
        margin_ok,
        feature_cols,
        obs_dim,
        symbol,
    })
}

#[cfg(feature = "tch")]
fn ordered_feature_cols(mut feats: std::collections::HashMap<String, Vec<f64>>) -> anyhow::Result<Vec<Vec<f64>>> {
    let mut cols = Vec::new();
    for &p in midas_env::features::periods() {
        cols.push(feats.remove(&format!("sma_{p}")).unwrap());
        cols.push(feats.remove(&format!("ema_{p}")).unwrap());
        cols.push(feats.remove(&format!("hma_{p}")).unwrap());
    }
    for &p in midas_env::features::ATR_PERIODS.iter() {
        cols.push(feats.remove(&format!("atr_{p}")).unwrap());
    }
    Ok(cols)
}

#[cfg(feature = "tch")]
fn observation_len(open: &[f64], close: &[f64], volume: Option<&[f64]>, feature_cols: &[Vec<f64>]) -> usize {
    let mut len = 0;
    len += 1; // open
    len += 1; // close t-1
    if volume.is_some() {
        len += 1; // vol t-1
    }
    len += 1; // equity
    len += feature_cols.len();
    len += 2; // time sin/cos
    len += 1; // position
    len += 2; // session/margin
    if len == 0 || open.is_empty() || close.is_empty() {
        0
    } else {
        len
    }
}

#[cfg(feature = "tch")]
fn build_observation(
    data: &DataSet,
    idx: usize,
    position: i32,
    equity: f64,
) -> Vec<f32> {
    use chrono::Timelike;
    let mut obs = Vec::with_capacity(data.obs_dim);

    if idx < data.open.len() {
        obs.push(data.open[idx]);
    } else {
        obs.push(f64::NAN);
    }

    if idx > 0 {
        obs.push(data.close[idx - 1]);
        if let Some(vol) = data.volume.as_ref() {
            obs.push(vol[idx - 1]);
        }
    } else {
        obs.push(f64::NAN);
        if data.volume.is_some() {
            obs.push(f64::NAN);
        }
    }

    obs.push(equity);

    for col in data.feature_cols.iter() {
        obs.push(*col.get(idx.saturating_sub(1)).unwrap_or(&f64::NAN));
    }

    if let Some(dt) = data.datetime_ns.as_ref().and_then(|d| d.get(idx.saturating_sub(1))) {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*dt);
        let hour = dt.hour() as f64 + dt.minute() as f64 / 60.0;
        let angle = 2.0 * std::f64::consts::PI * (hour / 24.0);
        obs.push(angle.sin());
        obs.push(angle.cos());
    } else {
        obs.push(f64::NAN);
        obs.push(f64::NAN);
    }

    obs.push(position as f64);

    let session_val = data
        .session_open
        .as_ref()
        .and_then(|m| m.get(idx.saturating_sub(1)))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    let margin_val = data
        .margin_ok
        .get(idx.saturating_sub(1))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    obs.push(session_val);
    obs.push(margin_val);

    obs.into_iter()
        .map(|v| if v.is_finite() { v as f32 } else { 0.0 })
        .collect()
}

#[cfg(feature = "tch")]
fn build_session_mask(datetimes_ns: &[i64], globex: bool) -> Vec<bool> {
    use chrono::Timelike;
    use chrono_tz::America::New_York;

    datetimes_ns
        .iter()
        .map(|ns| {
            let dt_utc = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*ns);
            let dt_et = dt_utc.with_timezone(&New_York);
            let hour = dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0;
            if globex {
                !(hour >= 17.0)
            } else {
                (hour >= 9.5) && (hour <= 16.0)
            }
        })
        .collect()
}

#[cfg(feature = "tch")]
#[derive(Clone)]
struct CandidateConfig {
    initial_balance: f64,
    margin_per_contract: f64,
    disable_margin: bool,
    w_pnl: f64,
    w_sortino: f64,
    w_mdd: f64,
    sortino_annualization: f64,
    hidden: usize,
    layers: usize,
    eval_windows: usize,
    device: tch::Device,
}

#[cfg(feature = "tch")]
#[derive(Clone)]
struct CandidateResult {
    fitness: f64,
    eval_pnl: f64,
    eval_sortino: f64,
    eval_drawdown: f64,
    eval_ret_mean: f64,
}

#[cfg(feature = "tch")]
fn evaluate_candidate(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> CandidateResult {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
    use tch::{kind::Kind, no_grad, Tensor};
    use tch::nn::Module;

    let mut vs = tch::nn::VarStore::new(cfg.device);
    let policy = build_mlp(&vs.root(), data.obs_dim as i64, cfg.hidden as i64, cfg.layers);
    load_params_from_vec(&vs, genome);

    let env_cfg = EnvConfig {
        margin_per_contract: cfg.margin_per_contract,
        enforce_margin: !cfg.disable_margin,
        ..EnvConfig::default()
    };

    let mut eval_pnls = Vec::new();
    let mut eval_returns = Vec::new();
    let mut eval_equity = Vec::new();

    for &(start, end) in windows.iter().take(cfg.eval_windows) {
        if end <= start + 1 {
            continue;
        }
        let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
        let mut position = 0;
        let mut equity = cfg.initial_balance;
        let mut pnl_buf = Vec::with_capacity(end - start - 1);
        let mut eq_curve = Vec::with_capacity(end - start - 1);

        for t in (start + 1)..end {
            let obs = build_observation(data, t, position, equity);
            let obs_t = Tensor::f_from_slice(&obs)
                .expect("tensor from obs")
                .to_device(cfg.device)
                .reshape(&[1, obs.len() as i64]);

            let action_idx = no_grad(|| {
                let logits = policy.forward(&obs_t);
                let probs = logits.softmax(-1, Kind::Float);
                let sample = probs.multinomial(1, true);
                sample.int64_value(&[0, 0]) as i32
            });

            let action = match action_idx {
                0 => Action::Buy,
                1 => Action::Sell,
                2 => Action::Hold,
                _ => Action::Revert,
            };

            let session_open = data
                .session_open
                .as_ref()
                .and_then(|m| m.get(t))
                .copied()
                .unwrap_or(true);
            let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);

            let (_reward, info) = env.step(
                action,
                data.close[t],
                StepContext {
                    session_open,
                    margin_ok,
                },
            );
            position = env.state().position;
            equity = env.state().cash + env.state().unrealized_pnl;
            pnl_buf.push(info.pnl_change);
            eq_curve.push(equity);
        }

        let pnl_sum: f64 = pnl_buf.iter().sum();
        eval_pnls.push(pnl_sum);

        let mut prev_eq = cfg.initial_balance;
        for (i, &pnl) in pnl_buf.iter().enumerate() {
            let eq = eq_curve.get(i).copied().unwrap_or(prev_eq);
            let denom = if prev_eq.abs() < 1e-8 { 1e-8 } else { prev_eq };
            eval_returns.push(pnl / denom);
            prev_eq = eq;
        }

        eval_equity.push(eq_curve);

        if capture_history {
            let _ = capture_history;
        }
    }

    let eval_sortino = compute_sortino(&eval_returns, cfg.sortino_annualization, 0.0, 50.0);
    let eval_draw = eval_equity
        .iter()
        .map(|eq| max_drawdown(eq))
        .fold(0.0_f64, |a, b| a.max(b));
    let eval_pnl = if eval_pnls.is_empty() {
        0.0
    } else {
        eval_pnls.iter().sum::<f64>() / eval_pnls.len() as f64
    };

    let fitness = cfg.w_pnl * eval_pnl + cfg.w_sortino * eval_sortino - cfg.w_mdd * eval_draw;

    CandidateResult {
        fitness,
        eval_pnl,
        eval_sortino,
        eval_drawdown: eval_draw,
        eval_ret_mean: if eval_returns.is_empty() {
            0.0
        } else {
            eval_returns.iter().sum::<f64>() / eval_returns.len() as f64
        },
    }
}

#[cfg(feature = "tch")]
fn compute_sortino(returns: &[f64], annualization: f64, target: f64, cap: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for r in returns {
        sum += r - target;
    }
    let mean = sum / returns.len() as f64;
    let mut downside = 0.0;
    for r in returns {
        let ex = r - target;
        let d = if ex < 0.0 { ex } else { 0.0 };
        downside += d * d;
    }
    let downside_std = (downside / returns.len() as f64).sqrt();
    if downside_std < 1e-6 {
        return if mean > 0.0 { cap } else { 0.0 };
    }
    let mut ratio = mean / (downside_std + 1e-8);
    ratio *= annualization.sqrt();
    ratio.min(cap)
}

#[cfg(feature = "tch")]
fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0;
    for &eq in equity {
        if eq > peak {
            peak = eq;
        }
        let dd = peak - eq;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

#[cfg(feature = "tch")]
fn param_count(input_dim: usize, hidden: usize, layers: usize) -> usize {
    let mut count = 0;
    let mut in_dim = input_dim;
    for _ in 0..layers {
        count += in_dim * hidden + hidden;
        in_dim = hidden;
    }
    count += in_dim * 4 + 4;
    count
}

#[cfg(feature = "tch")]
fn build_mlp(p: &tch::nn::Path, input_dim: i64, hidden: i64, layers: usize) -> tch::nn::Sequential {
    use tch::nn;
    let mut seq = nn::seq();
    let mut in_dim = input_dim;
    for i in 0..layers {
        let linear = nn::linear(p / format!("layer_{i}"), in_dim, hidden, Default::default());
        seq = seq.add(linear).add_fn(|xs| xs.tanh());
        in_dim = hidden;
    }
    let out = nn::linear(p / "out", in_dim, 4, Default::default());
    seq.add(out)
}

#[cfg(feature = "tch")]
fn load_params_from_vec(vs: &tch::nn::VarStore, genome: &[f32]) {
    let vars = vs.trainable_variables();
    let mut offset = 0;
    for mut v in vars {
        let numel = v.numel();
        let slice = &genome[offset..offset + numel as usize];
        let t = tch::Tensor::f_from_slice(slice)
            .expect("tensor from genome")
            .reshape(&v.size())
            .to_device(v.device());
        v.copy_(&t);
        offset += numel as usize;
    }
}

#[cfg(feature = "tch")]
fn series_to_f64(series: &polars::prelude::Series) -> anyhow::Result<Vec<f64>> {
    use polars::prelude::AnyValue;
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Float64(v) => v,
            AnyValue::Float32(v) => v as f64,
            AnyValue::Int64(v) => v as f64,
            AnyValue::Int32(v) => v as f64,
            AnyValue::UInt32(v) => v as f64,
            AnyValue::UInt64(v) => v as f64,
            _ => f64::NAN,
        })
        .collect();
    Ok(out)
}

#[cfg(feature = "tch")]
fn series_to_i64(series: &polars::prelude::Series) -> anyhow::Result<Vec<i64>> {
    use polars::prelude::AnyValue;
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Int64(v) => v,
            AnyValue::Int32(v) => v as i64,
            AnyValue::UInt64(v) => v as i64,
            AnyValue::UInt32(v) => v as i64,
            _ => 0_i64,
        })
        .collect();
    Ok(out)
}

#[cfg(feature = "tch")]
fn crossover(a: &[f32], b: &[f32], rng: &mut impl rand::Rng) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if rng.gen_bool(0.5) {
            out.push(a[i]);
        } else {
            out.push(b[i]);
        }
    }
    out
}

#[cfg(feature = "tch")]
fn save_policy(
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    device: tch::Device,
    genome: &[f32],
    path: &Path,
) -> anyhow::Result<()> {
    let mut vs = tch::nn::VarStore::new(device);
    let _policy = build_mlp(&vs.root(), obs_dim as i64, hidden as i64, layers);
    load_params_from_vec(&vs, genome);
    vs.save(path)?;
    Ok(())
}

#[cfg(feature = "tch")]
fn save_checkpoint(path: &Path, pop: &[Vec<f32>]) -> anyhow::Result<()> {
    let data = bincode::serialize(pop)?;
    std::fs::write(path, data)?;
    Ok(())
}

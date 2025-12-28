use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(about = "Rust GA-only neuroevolution trainer (CUDA via libtorch)")]
pub struct Args {
    #[arg(long)]
    pub parquet: Option<PathBuf>,
    #[arg(long, default_value = "data/train")]
    pub train_parquet: PathBuf,
    #[arg(long, default_value = "data/val")]
    pub val_parquet: PathBuf,
    #[arg(long, default_value = "data/test")]
    pub test_parquet: PathBuf,
    #[arg(long)]
    pub full_file: bool,
    #[arg(long)]
    pub windowed: bool,
    #[arg(long, default_value_t = 512)]
    pub window: usize,
    #[arg(long, default_value_t = 256)]
    pub step: usize,
    #[arg(long)]
    pub device: Option<String>,
    #[arg(long, default_value_t = true)]
    pub globex: bool,
    #[arg(long)]
    pub rth: bool,
    #[arg(long, default_value_t = 10000.0)]
    pub initial_balance: f64,
    #[arg(long)]
    pub margin_per_contract: Option<f64>,
    #[arg(long, default_value = "config/symbols.yaml")]
    pub symbol_config: PathBuf,
    #[arg(long, default_value = "runs_ga")]
    pub outdir: PathBuf,

    #[arg(long, default_value_t = 5)]
    pub generations: usize,
    #[arg(long, default_value_t = 6)]
    pub pop_size: usize,
    #[arg(long, default_value_t = 0)]
    pub workers: usize,
    #[arg(long, default_value_t = 0.33)]
    pub elite_frac: f64,
    #[arg(long, default_value_t = 0.05)]
    pub mutation_sigma: f64,
    #[arg(long, default_value_t = 0.5)]
    pub init_sigma: f64,
    #[arg(long, default_value_t = 128)]
    pub hidden: usize,
    #[arg(long, default_value_t = 2)]
    pub layers: usize,

    #[arg(long, default_value_t = 2)]
    pub eval_windows: usize,
    #[arg(long, default_value_t = 1.0)]
    pub w_pnl: f64,
    #[arg(long, default_value_t = 1.0)]
    pub w_sortino: f64,
    #[arg(long, default_value_t = 0.5)]
    pub w_mdd: f64,
    #[arg(long, default_value_t = 1.0)]
    pub sortino_annualization: f64,
    #[arg(long, default_value_t = 0.0)]
    pub drawdown_penalty: f64,
    #[arg(long, default_value_t = 0.0)]
    pub drawdown_penalty_growth: f64,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub disable_margin: bool,
    #[arg(long)]
    pub skip_val_eval: bool,
    #[arg(long)]
    pub debug_data: bool,
    #[arg(long)]
    pub ignore_session: bool,
}

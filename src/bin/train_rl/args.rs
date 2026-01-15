use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(about = "Rust PPO trainer for midas_env")]
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
    #[arg(long, default_value_t = 0)]
    pub max_position: i32,
    #[arg(long, default_value = "auto", value_parser = ["auto", "per-contract", "price"]) ]
    pub margin_mode: String,
    #[arg(long, default_value_t = 1.0)]
    pub contract_multiplier: f64,
    #[arg(long)]
    pub margin_per_contract: Option<f64>,
    #[arg(long, default_value = "config/symbols.yaml")]
    pub symbol_config: PathBuf,
    #[arg(long, default_value = "runs_rl")]
    pub outdir: PathBuf,

    #[arg(long, default_value_t = 10)]
    pub epochs: usize,
    #[arg(long, default_value_t = 3)]
    pub train_windows: usize,
    #[arg(long, default_value_t = 4)]
    pub ppo_epochs: usize,
    #[arg(long, default_value_t = 0.0003)]
    pub lr: f64,
    #[arg(long, default_value_t = 0.99)]
    pub gamma: f64,
    #[arg(long, default_value_t = 0.95)]
    pub lam: f64,
    #[arg(long, default_value_t = 0.2)]
    pub clip: f64,
    #[arg(long, default_value_t = 0.5)]
    pub vf_coef: f64,
    #[arg(long, default_value_t = 0.01)]
    pub ent_coef: f64,
    #[arg(long, default_value_t = 128)]
    pub hidden: usize,
    #[arg(long, default_value_t = 2)]
    pub layers: usize,
    #[arg(long, default_value_t = 2)]
    pub eval_windows: usize,
    #[arg(long, default_value_t = 1)]
    pub log_interval: usize,
    #[arg(long, default_value_t = 1)]
    pub checkpoint_every: usize,
    #[arg(long)]
    pub load_checkpoint: Option<PathBuf>,
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
    #[arg(long, default_value_t = 0.0)]
    pub session_close_penalty: f64,
    #[arg(long, default_value_t = 15)]
    pub max_hold_bars_positive: usize,
    #[arg(long, default_value_t = 15)]
    pub max_hold_bars_drawdown: usize,
    #[arg(long, default_value_t = 1.0)]
    pub hold_duration_penalty: f64,
    #[arg(long, default_value_t = 0.05)]
    pub hold_duration_penalty_growth: f64,
    #[arg(long, default_value_t = 0.5)]
    pub hold_duration_penalty_positive_scale: f64,
    #[arg(long, default_value_t = 1.5)]
    pub hold_duration_penalty_negative_scale: f64,
    #[arg(long, default_value_t = 7.0)]
    pub invalid_revert_penalty: f64,
    #[arg(long, default_value_t = 0.5)]
    pub invalid_revert_penalty_growth: f64,
    #[arg(long, default_value_t = 2.20)]
    pub flat_hold_penalty: f64,
    #[arg(long, default_value_t = 0.05)]
    pub flat_hold_penalty_growth: f64,
    #[arg(long, default_value_t = 100)]
    pub max_flat_hold_bars: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub disable_margin: bool,
    #[arg(long)]
    pub ignore_session: bool,
}

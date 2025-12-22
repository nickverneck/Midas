mod parquet_loader;
mod backtesting;
mod env;
mod features;
mod sampler;

use anyhow::Result;
use clap::Parser;
use polars::prelude::DataFrame;
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
use crate::backtesting::{run_ema_crossover, EmaParams};
use crate::env::EnvConfig;
use crate::features::{compute_features, periods};
use chrono_tz::Tz;

/// Simple CLI for loading parquet files with Polars.
#[derive(Parser)]
#[command(name = "midas", version, about = "Load parquet data with Polars", long_about = None)]
struct Cli {
    /// Path to the parquet file to load.
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf,

    /// Mode: feature (default) or ema_rule.
    #[arg(long, default_value = "feature", value_parser = ["feature", "ema_rule"])]
    mode: String,

    #[arg(long, default_value_t = 5)]
    ema_fast: usize,

    #[arg(long, default_value_t = 21)]
    ema_slow: usize,

    /// Commission round-turn in USD.
    #[arg(long, default_value_t = 1.60)]
    commission: f64,

    /// Slippage per contract in USD.
    #[arg(long, default_value_t = 0.25)]
    slippage: f64,

    /// Max absolute position (contracts).
    #[arg(long, default_value_t = 1)]
    max_position: i32,

    /// Export equity/rewards to CSV (only for ema_rule).
    #[arg(long)]
    export: Option<PathBuf>,

    /// Symbol config YAML (margin/session defaults).
    #[arg(long, default_value = "config/symbols.yaml")]
    symbol_config: PathBuf,

    /// Session: auto (from config), rth, globex
    #[arg(long, default_value = "auto", value_parser = ["auto", "rth", "globex"])]
    session: String,
}

#[derive(Debug, Deserialize)]
struct SymbolCfg {
    margin_per_contract: Option<f64>,
    session: Option<String>,
    #[serde(default = "default_tz")]
    tz: String,
}

fn default_tz() -> String { "America/New_York".to_string() }

fn load_symbol_cfg(path: &PathBuf, symbol: &str) -> Option<SymbolCfg> {
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let cfg: serde_yaml::Value = serde_yaml::from_str(&text).ok()?;
    cfg.get(symbol)
        .and_then(|v| serde_yaml::from_value::<SymbolCfg>(v.clone()).ok())
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let timer = Instant::now();

    let df: DataFrame = parquet_loader::load_parquet(&args.file)?;
    let elapsed = timer.elapsed();

    println!("Loaded `{}`", args.file.display());
    println!("{df}");
    println!(
        "Finished in {:.9} seconds ({} µs) ({} rows × {} columns)",
        elapsed.as_secs_f64(),
        elapsed.as_micros(),
        df.height(),
        df.width()
    );

    // Common price vector
    let close = df.column("close")?.f64()?;
    let prices: Vec<f64> = close.into_no_null_iter().collect();
    if prices.len() < 2 {
        anyhow::bail!("need at least 2 price points for backtest");
    }

    // Symbol config (if present)
    let symbol = df
        .column("symbol")
        .ok()
        .and_then(|c| c.str().ok())
        .and_then(|s| s.get(0))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    let sym_cfg = load_symbol_cfg(&args.symbol_config, &symbol);

    let mut cfg = EnvConfig {
        commission_round_turn: args.commission,
        slippage_per_contract: args.slippage,
        max_position: args.max_position,
        margin_per_contract: sym_cfg
            .as_ref()
            .and_then(|c| c.margin_per_contract)
            .unwrap_or(50.0),
        ..Default::default()
    };
    // Session handling
    let session_choice = match args.session.as_str() {
        "rth" => Some("rth".to_string()),
        "globex" => Some("globex".to_string()),
        _ => sym_cfg.as_ref().and_then(|c| c.session.clone()),
    };
    if let Some(s) = session_choice {
        cfg.default_session_open = match s.to_lowercase().as_str() {
            "rth" => true,   // open flag used with external mask; here we just keep true
            "globex" => true,
            _ => true,
        };
    }

    match args.mode.as_str() {
        "feature" => {
            let feats = compute_features(&prices);
            println!("Feature-only mode: computed {} feature columns over {} rows", feats.len(), prices.len());
            println!("Periods: {:?}", periods());
            // Show a small preview of the last row to confirm values.
            let sample_col = format!("ema_{}", periods()[0]);
            if let Some(col) = feats.get(&sample_col) {
                if let Some(v) = col.last() {
                    println!("Sample {} last value: {}", sample_col, v);
                }
            }
        }
        "ema_rule" => {
            let params = EmaParams {
                fast: args.ema_fast,
                slow: args.ema_slow,
            };
            let res = run_ema_crossover(&prices, params, cfg);
            println!(
                "EMA crossover metrics (fast {}, slow {}): {:#?}",
                params.fast, params.slow, res.metrics
            );
            if let Some(path) = args.export {
                let mut wtr = csv::Writer::from_path(&path)?;
                wtr.write_record(["step", "equity", "reward"])?;
                for (i, (eq, r)) in res.equity_curve.iter().zip(res.rewards.iter()).enumerate() {
                    wtr.write_record([i.to_string(), eq.to_string(), r.to_string()])?;
                }
                wtr.flush()?;
                println!("Exported equity/rewards to {}", path.display());
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}

use anyhow::{bail, Context, Result};
use clap::Parser;
use polars::prelude::{AnyValue, DataFrame, ParquetReader, SerReader, Series};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

use midas_env::backtesting::compute_metrics;
use midas_env::env::{Action, EnvConfig, MarginMode, StepContext, TradingEnv};
use midas_env::features::{compute_features_ohlcv, periods, ATR_PERIODS};
use midas_env::script::{ScriptLimits, ScriptRunner};

#[derive(Parser, Debug)]
#[command(name = "backtest_script", about = "Run a Lua strategy script against a parquet dataset")]
struct Args {
    #[arg(long)]
    file: PathBuf,

    #[arg(long)]
    script: Option<PathBuf>,

    #[arg(long = "script-inline")]
    script_inline: Option<String>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long, default_value_t = 10_000.0)]
    initial_balance: f64,

    #[arg(long, default_value_t = 1)]
    max_position: i32,

    #[arg(long, default_value_t = 1.60)]
    commission_round_turn: f64,

    #[arg(long, default_value_t = 0.25)]
    slippage_per_contract: f64,

    #[arg(long, default_value_t = 50.0)]
    margin_per_contract: f64,

    #[arg(long, default_value_t = 1.0)]
    contract_multiplier: f64,

    #[arg(long, default_value = "per-contract")]
    margin_mode: String,

    #[arg(long, default_value_t = true)]
    enforce_margin: bool,

    #[arg(long, default_value_t = false)]
    globex: bool,

    #[arg(long, default_value_t = 64)]
    memory_limit_mb: usize,

    #[arg(long, default_value_t = 5_000_000)]
    instruction_limit: u64,

    #[arg(long, default_value_t = 10_000)]
    instruction_interval: u32,

    #[arg(long, default_value_t = false)]
    trace_actions: bool,
}

#[derive(Serialize)]
struct MetricsPayload {
    total_reward: f64,
    net_pnl: f64,
    ending_equity: f64,
    sharpe: f64,
    max_drawdown: f64,
    profit_factor: f64,
    win_rate: f64,
    max_consecutive_losses: usize,
    steps: usize,
}

#[derive(Serialize)]
struct ActionTrace {
    idx: usize,
    action: String,
}

#[derive(Serialize)]
struct BacktestPayload {
    metrics: MetricsPayload,
    equity_curve: Vec<f64>,
    actions: Option<Vec<ActionTrace>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    let script_text = match (args.script, args.script_inline) {
        (Some(path), None) => fs::read_to_string(&path)
            .with_context(|| format!("read script {}", path.display()))?,
        (None, Some(text)) => text,
        (Some(_), Some(_)) => bail!("Provide either --script or --script-inline, not both"),
        (None, None) => bail!("Provide --script or --script-inline"),
    };

    let df = load_parquet_slice(&args.file, args.offset, args.limit)?;
    let data = extract_data(&df, args.globex)?;

    if data.close.len() < 2 {
        bail!("Not enough bars to run backtest");
    }

    let limits = ScriptLimits {
        memory_bytes: if args.memory_limit_mb == 0 {
            None
        } else {
            Some(args.memory_limit_mb * 1024 * 1024)
        },
        instruction_limit: if args.instruction_limit == 0 {
            None
        } else {
            Some(args.instruction_limit)
        },
        instruction_check_interval: args.instruction_interval,
    };

    let runner = ScriptRunner::new(&script_text, limits)?;

    let env_cfg = EnvConfig {
        commission_round_turn: args.commission_round_turn,
        slippage_per_contract: args.slippage_per_contract,
        max_position: args.max_position,
        margin_per_contract: args.margin_per_contract,
        margin_mode: match args.margin_mode.as_str() {
            "price" => MarginMode::Price,
            _ => MarginMode::PerContract,
        },
        contract_multiplier: args.contract_multiplier,
        enforce_margin: args.enforce_margin,
        default_session_open: true,
        ..EnvConfig::default()
    };

    let lua = runner.lua();
    let mut init_ctx = lua.create_table()?;
    init_ctx.set("symbol", data.symbol.as_str())?;
    init_ctx.set("bars", data.close.len())?;
    init_ctx.set("start_ts", data.datetime_ns.first().copied().unwrap_or(0))?;
    init_ctx.set("end_ts", data.datetime_ns.last().copied().unwrap_or(0))?;
    runner.call_on_init(&init_ctx)?;

    let mut env = TradingEnv::new(data.close[0], args.initial_balance, env_cfg);
    let mut rewards = Vec::with_capacity(data.close.len().saturating_sub(1));
    let mut equity_curve = Vec::with_capacity(data.close.len().saturating_sub(1));
    let mut action_trace: Vec<ActionTrace> = Vec::new();

    for step in 1..data.close.len() {
        let bar_idx = step - 1;
        let state = env.state();
        let equity = state.cash + state.unrealized_pnl;

        let ctx = build_ctx_table(
            lua,
            bar_idx,
            state.position,
            state.cash,
            equity,
            state.unrealized_pnl,
            state.realized_pnl,
        )?;
        let bar = build_bar_table(lua, &data, bar_idx)?;

        let action = runner.call_on_bar(&ctx, &bar)?;
        let next_price = data.close[step];
        let step_ctx = StepContext {
            session_open: data.session_open.get(bar_idx).copied().unwrap_or(true),
            margin_ok: data.margin_ok.get(bar_idx).copied().unwrap_or(true),
            minutes_to_close: data.minutes_to_close.get(bar_idx).copied(),
        };

        let (reward, _info) = env.step(action, next_price, step_ctx);
        rewards.push(reward);
        let s = env.state();
        equity_curve.push(s.cash + s.unrealized_pnl);

        if args.trace_actions {
            action_trace.push(ActionTrace {
                idx: bar_idx,
                action: action_to_string(action).to_string(),
            });
        }
    }

    let metrics = compute_metrics(&rewards, &equity_curve);
    let ending_equity = equity_curve.last().copied().unwrap_or(args.initial_balance);
    let net_pnl = ending_equity - args.initial_balance;

    let payload = BacktestPayload {
        metrics: MetricsPayload {
            total_reward: metrics.total_reward,
            net_pnl,
            ending_equity,
            sharpe: metrics.sharpe_ratio,
            max_drawdown: metrics.max_drawdown,
            profit_factor: metrics.profit_factor,
            win_rate: metrics.win_rate,
            max_consecutive_losses: metrics.max_consecutive_losses,
            steps: rewards.len(),
        },
        equity_curve,
        actions: if args.trace_actions {
            Some(action_trace)
        } else {
            None
        },
    };

    let json = serde_json::to_string_pretty(&payload)?;
    println!("{json}");
    Ok(())
}

struct ScriptData {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Option<Vec<f64>>,
    datetime_ns: Vec<i64>,
    session_open: Vec<bool>,
    minutes_to_close: Vec<f64>,
    margin_ok: Vec<bool>,
    features: Vec<(String, Vec<f64>)>,
    symbol: String,
}

fn load_parquet_slice(path: &PathBuf, offset: usize, limit: Option<usize>) -> Result<DataFrame> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open parquet {}", path.display()))?;
    let mut df = ParquetReader::new(file).finish()?;
    let total = df.height();
    let start = offset.min(total);
    let len = limit.unwrap_or(total.saturating_sub(start)).min(total.saturating_sub(start));
    if start > 0 || len < total {
        df = df.slice(start as i64, len);
    }
    Ok(df)
}

fn extract_data(df: &DataFrame, globex: bool) -> Result<ScriptData> {
    let close = series_to_f64(df.column("close")?.as_materialized_series())?;
    let open = if let Ok(col) = df.column("open") {
        series_to_f64(col.as_materialized_series())?
    } else {
        close.clone()
    };
    let high = if let Ok(col) = df.column("high") {
        series_to_f64(col.as_materialized_series())?
    } else {
        close.clone()
    };
    let low = if let Ok(col) = df.column("low") {
        series_to_f64(col.as_materialized_series())?
    } else {
        close.clone()
    };
    let volume = df
        .column("volume")
        .ok()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .transpose()?;

    let datetime_ns = df
        .column("date")
        .ok()
        .map(|c| series_to_i64(c.as_materialized_series()))
        .transpose()?
        .unwrap_or_else(|| vec![0_i64; close.len()]);

    let session_open = df
        .column("session_open")
        .ok()
        .map(|c| series_to_bool(c.as_materialized_series()))
        .transpose()?
        .unwrap_or_else(|| build_session_mask(&datetime_ns, globex));

    let minutes_to_close = df
        .column("minutes_to_close")
        .ok()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .transpose()?
        .unwrap_or_else(|| build_minutes_to_close(&datetime_ns, globex));

    let margin_ok = df
        .column("margin_ok")
        .ok()
        .map(|c| series_to_bool(c.as_materialized_series()))
        .transpose()?
        .unwrap_or_else(|| vec![true; close.len()]);

    let symbol = df
        .column("symbol")
        .ok()
        .and_then(|c| c.get(0).ok())
        .and_then(|v| match v {
            AnyValue::String(s) => Some(s.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "UNKNOWN".to_string());

    let features = ordered_feature_cols(
        compute_features_ohlcv(&close, Some(&high), Some(&low), volume.as_deref()),
        close.len(),
    );

    Ok(ScriptData {
        open,
        high,
        low,
        close,
        volume,
        datetime_ns,
        session_open,
        minutes_to_close,
        margin_ok,
        features,
        symbol,
    })
}

fn ordered_feature_cols(
    mut feats: std::collections::HashMap<String, Vec<f64>>,
    len: usize,
) -> Vec<(String, Vec<f64>)> {
    let mut cols = Vec::new();
    for &p in periods() {
        let sma_key = format!("sma_{p}");
        let ema_key = format!("ema_{p}");
        let hma_key = format!("hma_{p}");
        cols.push((
            sma_key.clone(),
            feats.remove(&sma_key).unwrap_or_else(|| vec![f64::NAN; len]),
        ));
        cols.push((
            ema_key.clone(),
            feats.remove(&ema_key).unwrap_or_else(|| vec![f64::NAN; len]),
        ));
        cols.push((
            hma_key.clone(),
            feats.remove(&hma_key).unwrap_or_else(|| vec![f64::NAN; len]),
        ));
    }
    for &p in ATR_PERIODS.iter() {
        let key = format!("atr_{p}");
        cols.push((
            key.clone(),
            feats.remove(&key).unwrap_or_else(|| vec![f64::NAN; len]),
        ));
    }
    cols
}

fn build_ctx_table<'lua>(
    lua: &'lua mlua::Lua,
    step: usize,
    position: i32,
    cash: f64,
    equity: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
) -> Result<mlua::Table<'lua>> {
    let table = lua.create_table()?;
    table.set("step", step)?;
    table.set("position", position)?;
    table.set("cash", cash)?;
    table.set("equity", equity)?;
    table.set("unrealized_pnl", unrealized_pnl)?;
    table.set("realized_pnl", realized_pnl)?;
    Ok(table)
}

fn build_bar_table<'lua>(
    lua: &'lua mlua::Lua,
    data: &ScriptData,
    idx: usize,
) -> Result<mlua::Table<'lua>> {
    let table = lua.create_table()?;
    table.set("ts", data.datetime_ns.get(idx).copied().unwrap_or(0))?;
    table.set("open", data.open.get(idx).copied().unwrap_or(f64::NAN))?;
    table.set("high", data.high.get(idx).copied().unwrap_or(f64::NAN))?;
    table.set("low", data.low.get(idx).copied().unwrap_or(f64::NAN))?;
    table.set("close", data.close.get(idx).copied().unwrap_or(f64::NAN))?;
    if let Some(vol) = data.volume.as_ref().and_then(|v| v.get(idx).copied()) {
        table.set("volume", vol)?;
    }

    for (name, col) in data.features.iter() {
        if let Some(value) = col.get(idx).copied() {
            table.set(name.as_str(), value)?;
        }
    }

    Ok(table)
}

fn action_to_string(action: Action) -> &'static str {
    match action {
        Action::Buy => "buy",
        Action::Sell => "sell",
        Action::Hold => "hold",
        Action::Revert => "revert",
    }
}

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Float64(v) => v,
            AnyValue::Float32(v) => v as f64,
            AnyValue::Int64(v) => v as f64,
            AnyValue::Int32(v) => v as f64,
            AnyValue::UInt32(v) => v as f64,
            AnyValue::UInt64(v) => v as f64,
            AnyValue::Boolean(v) => if v { 1.0 } else { 0.0 },
            _ => f64::NAN,
        })
        .collect();
    Ok(out)
}

fn series_to_bool(series: &Series) -> Result<Vec<bool>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Boolean(v) => v,
            AnyValue::Int64(v) => v != 0,
            AnyValue::Int32(v) => v != 0,
            AnyValue::UInt32(v) => v != 0,
            AnyValue::UInt64(v) => v != 0,
            AnyValue::Float64(v) => v.abs() > f64::EPSILON,
            AnyValue::Float32(v) => (v as f64).abs() > f64::EPSILON,
            _ => false,
        })
        .collect();
    Ok(out)
}

fn series_to_i64(series: &Series) -> Result<Vec<i64>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Datetime(v, _, _) => v,
            AnyValue::DatetimeOwned(v, _, _) => v,
            AnyValue::Int64(v) => v,
            AnyValue::Int32(v) => v as i64,
            AnyValue::UInt64(v) => v as i64,
            AnyValue::UInt32(v) => v as i64,
            _ => 0_i64,
        })
        .collect();
    Ok(out)
}

fn build_session_mask(datetimes_ns: &[i64], globex: bool) -> Vec<bool> {
    use chrono::Timelike;
    use chrono_tz::America::New_York;

    if datetimes_ns.is_empty() || datetimes_ns[0] == 0 {
        return vec![true; datetimes_ns.len()];
    }

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

fn build_minutes_to_close(datetimes_ns: &[i64], globex: bool) -> Vec<f64> {
    use chrono::Timelike;
    use chrono_tz::America::New_York;

    if datetimes_ns.is_empty() || datetimes_ns[0] == 0 {
        return vec![0.0; datetimes_ns.len()];
    }

    datetimes_ns
        .iter()
        .map(|ns| {
            let dt_utc = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*ns);
            let dt_et = dt_utc.with_timezone(&New_York);
            let hour = dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0;
            let close_hour = if globex { 17.0 } else { 16.0 };
            let minutes = (close_hour - hour) * 60.0;
            if minutes.is_finite() { minutes.max(0.0) } else { 0.0 }
        })
        .collect()
}

use anyhow::{Context, Result, bail};
use clap::Parser;
use polars::prelude::{AnyValue, DataFrame, ParquetReader, SerReader, Series};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use midas_env::backtesting::{compute_metrics, equity_returns};
use midas_env::env::{Action, EnvConfig, MarginMode, StepContext, TradingEnv};
use midas_env::features::{alma, ema, hma, kama, sma, wma};

#[derive(Parser, Debug)]
#[command(
    name = "strategy_analyzer",
    about = "Sweep indicator ranges and return heatmap metrics"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzerConfig {
    file: String,
    offset: Option<usize>,
    limit: Option<usize>,
    initial_balance: f64,
    max_position: i32,
    commission_round_turn: f64,
    slippage_per_contract: f64,
    margin_per_contract: f64,
    contract_multiplier: f64,
    margin_mode: String,
    enforce_margin: bool,
    globex: Option<bool>,
    signal: SignalConfig,
    take_profit: Option<RangeF>,
    stop_loss: Option<RangeF>,
    fitness: Option<FitnessWeights>,
    max_combinations: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalConfig {
    indicator_a: IndicatorSpec,
    indicator_b: IndicatorSpec,
    buy_action: CrossAction,
    sell_action: CrossAction,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IndicatorSpec {
    kind: IndicatorKind,
    #[serde(default)]
    sweep_param: SweepParam,
    range: RangeSweep,
    params: Option<IndicatorParams>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IndicatorParams {
    period: Option<usize>,
    fast: Option<usize>,
    slow: Option<usize>,
    offset: Option<f64>,
    sigma: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum IndicatorKind {
    Sma,
    Ema,
    Hma,
    Wma,
    Kama,
    Alma,
    Price,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
enum SweepParam {
    #[default]
    Period,
    Fast,
    Slow,
    Offset,
    Sigma,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum CrossAction {
    Crossover,
    Crossunder,
}

#[derive(Debug, Deserialize)]
struct RangeSweep {
    start: f64,
    end: f64,
    step: f64,
}

#[derive(Debug, Deserialize)]
struct RangeF {
    start: f64,
    end: f64,
    step: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FitnessWeights {
    w_pnl: f64,
    w_sortino: f64,
    w_mdd: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzerOutput {
    axes: AxesInfo,
    results: Vec<AnalyzerCell>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AxesInfo {
    indicator_a: IndicatorAxis,
    indicator_b: IndicatorAxis,
    take_profit_values: Vec<f64>,
    stop_loss_values: Vec<f64>,
    total_combinations: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndicatorAxis {
    kind: IndicatorKind,
    sweep_param: SweepParam,
    periods: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalyzerCell {
    a_period: f64,
    b_period: f64,
    take_profit: Option<f64>,
    stop_loss: Option<f64>,
    metrics: MetricsPayload,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MetricsPayload {
    total_reward: f64,
    net_pnl: f64,
    ending_equity: f64,
    sharpe: f64,
    sortino: f64,
    max_drawdown: f64,
    profit_factor: f64,
    win_rate: f64,
    trades: usize,
    fitness: f64,
    steps: usize,
}

#[derive(Debug)]
struct MarketData {
    close: Vec<f64>,
    datetime_ns: Vec<i64>,
    session_open: Vec<bool>,
    minutes_to_close: Vec<f64>,
    margin_ok: Vec<bool>,
}

#[derive(Clone, Copy)]
struct Combo {
    a_value: f64,
    b_value: f64,
    take_profit: Option<f64>,
    stop_loss: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_text = fs::read_to_string(&args.config)
        .with_context(|| format!("read config {}", args.config.display()))?;
    let cfg: AnalyzerConfig =
        serde_json::from_str(&config_text).with_context(|| "parse analyzer config")?;
    run(cfg)
}

fn run(cfg: AnalyzerConfig) -> Result<()> {
    let file_path = PathBuf::from(&cfg.file);
    if !file_path.exists() {
        bail!("parquet file not found: {}", file_path.display());
    }

    let df = load_parquet_slice(&file_path, cfg.offset.unwrap_or(0), cfg.limit)?;
    let data = extract_data(&df, cfg.globex.unwrap_or(false))?;

    if data.close.len() < 2 {
        bail!("not enough bars to run analyzer");
    }

    let sweep_values_a = build_indicator_sweep_values(&cfg.signal.indicator_a, "indicatorA")?;
    let sweep_values_b = build_indicator_sweep_values(&cfg.signal.indicator_b, "indicatorB")?;

    let tp_values = match cfg.take_profit {
        Some(range) => build_range_f64(&range, "takeProfit")?,
        None => Vec::new(),
    };
    let sl_values = match cfg.stop_loss {
        Some(range) => build_range_f64(&range, "stopLoss")?,
        None => Vec::new(),
    };

    let tp_options: Vec<Option<f64>> = if tp_values.is_empty() {
        vec![None]
    } else {
        tp_values.iter().copied().map(Some).collect()
    };
    let sl_options: Vec<Option<f64>> = if sl_values.is_empty() {
        vec![None]
    } else {
        sl_values.iter().copied().map(Some).collect()
    };

    let total_combos =
        sweep_values_a.len() * sweep_values_b.len() * tp_options.len() * sl_options.len();
    let max_combos = cfg.max_combinations.unwrap_or(20_000);
    if total_combos > max_combos {
        bail!(
            "combination count {} exceeds limit {}",
            total_combos,
            max_combos
        );
    }

    let base_env = EnvConfig {
        commission_round_turn: cfg.commission_round_turn,
        slippage_per_contract: cfg.slippage_per_contract,
        max_position: cfg.max_position,
        margin_per_contract: cfg.margin_per_contract,
        margin_mode: match cfg.margin_mode.as_str() {
            "price" => MarginMode::Price,
            _ => MarginMode::PerContract,
        },
        contract_multiplier: cfg.contract_multiplier,
        enforce_margin: cfg.enforce_margin,
        default_session_open: true,
        ..EnvConfig::default()
    };

    let series_a = precompute_series(&cfg.signal.indicator_a, &sweep_values_a, &data.close)?;
    let series_b = precompute_series(&cfg.signal.indicator_b, &sweep_values_b, &data.close)?;

    let prices = Arc::new(data.close);
    let session_open = Arc::new(data.session_open);
    let margin_ok = Arc::new(data.margin_ok);
    let minutes_to_close = Arc::new(data.minutes_to_close);

    let combos = build_combos(&sweep_values_a, &sweep_values_b, &tp_options, &sl_options);

    let results: Vec<AnalyzerCell> = combos
        .par_iter()
        .map(|combo| {
            let series_a = series_a
                .get(&sweep_value_key(combo.a_value))
                .expect("missing indicator A series");
            let series_b = series_b
                .get(&sweep_value_key(combo.b_value))
                .expect("missing indicator B series");
            let metrics = run_strategy(
                &prices,
                series_a,
                series_b,
                &base_env,
                cfg.initial_balance,
                cfg.signal.buy_action,
                cfg.signal.sell_action,
                combo.take_profit,
                combo.stop_loss,
                &session_open,
                &margin_ok,
                &minutes_to_close,
                cfg.fitness.as_ref(),
            );
            AnalyzerCell {
                a_period: combo.a_value,
                b_period: combo.b_value,
                take_profit: combo.take_profit,
                stop_loss: combo.stop_loss,
                metrics,
            }
        })
        .collect();

    let output = AnalyzerOutput {
        axes: AxesInfo {
            indicator_a: IndicatorAxis {
                kind: cfg.signal.indicator_a.kind,
                sweep_param: cfg.signal.indicator_a.sweep_param,
                periods: sweep_values_a.clone(),
            },
            indicator_b: IndicatorAxis {
                kind: cfg.signal.indicator_b.kind,
                sweep_param: cfg.signal.indicator_b.sweep_param,
                periods: sweep_values_b.clone(),
            },
            take_profit_values: tp_values,
            stop_loss_values: sl_values,
            total_combinations: total_combos,
        },
        results,
    };

    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    Ok(())
}

fn build_indicator_sweep_values(spec: &IndicatorSpec, label: &str) -> Result<Vec<f64>> {
    if !sweep_param_allowed_for_kind(spec.kind, spec.sweep_param) {
        bail!(
            "{label} sweep parameter {:?} is not valid for {:?}",
            spec.sweep_param,
            spec.kind
        );
    }
    let values = build_range_any_f64(&spec.range, label)?;
    match spec.sweep_param {
        SweepParam::Period | SweepParam::Fast | SweepParam::Slow => {
            if values.is_empty() {
                bail!("{label} sweep range is empty");
            }
            for value in values.iter().copied() {
                if value <= 0.0 {
                    bail!("{label} sweep values must be > 0");
                }
                if (value - value.round()).abs() > 1e-9 {
                    bail!("{label} sweep values must be integers");
                }
            }
        }
        SweepParam::Offset => {
            for value in values.iter().copied() {
                if !(0.0..=1.0).contains(&value) {
                    bail!("{label} offset sweep values must be in [0, 1]");
                }
            }
        }
        SweepParam::Sigma => {
            for value in values.iter().copied() {
                if value <= 0.0 {
                    bail!("{label} sigma sweep values must be > 0");
                }
            }
        }
    }
    Ok(values)
}

fn sweep_param_allowed_for_kind(kind: IndicatorKind, sweep_param: SweepParam) -> bool {
    match kind {
        IndicatorKind::Kama => matches!(
            sweep_param,
            SweepParam::Period | SweepParam::Fast | SweepParam::Slow
        ),
        IndicatorKind::Alma => matches!(
            sweep_param,
            SweepParam::Period | SweepParam::Offset | SweepParam::Sigma
        ),
        _ => matches!(sweep_param, SweepParam::Period),
    }
}

fn build_range_any_f64(range: &RangeSweep, label: &str) -> Result<Vec<f64>> {
    if range.step <= 0.0 {
        bail!("{label} step must be greater than 0");
    }
    if range.end < range.start {
        bail!("{label} end must be >= start");
    }
    let mut out = Vec::new();
    let mut value = range.start;
    let mut guard = 0;
    while value <= range.end + 1e-9 {
        out.push(round_sweep_value(value));
        value += range.step;
        guard += 1;
        if guard > 10_000 {
            bail!("{label} range too large");
        }
    }
    Ok(out)
}

fn build_range_f64(range: &RangeF, label: &str) -> Result<Vec<f64>> {
    if range.step <= 0.0 {
        bail!("{label} step must be greater than 0");
    }
    if range.start < 0.0 {
        bail!("{label} start must be >= 0");
    }
    if range.end < range.start {
        bail!("{label} end must be >= start");
    }
    let mut out = Vec::new();
    let mut value = range.start;
    let mut guard = 0;
    while value <= range.end + 1e-9 {
        out.push(round_sweep_value(value));
        value += range.step;
        guard += 1;
        if guard > 10_000 {
            bail!("{label} range too large");
        }
    }
    Ok(out)
}

fn round_sweep_value(value: f64) -> f64 {
    let scaled = (value * 1_000_000.0).round();
    scaled / 1_000_000.0
}

fn precompute_series(
    spec: &IndicatorSpec,
    sweep_values: &[f64],
    prices: &[f64],
) -> Result<HashMap<i64, Arc<Vec<f64>>>> {
    let defaults = resolve_indicator_defaults(spec);
    let mut out = HashMap::new();

    for &sweep_value in sweep_values {
        let mut period = defaults.period;
        let mut fast = defaults.fast;
        let mut slow = defaults.slow;
        let mut offset = defaults.offset;
        let mut sigma = defaults.sigma;

        match spec.sweep_param {
            SweepParam::Period => period = to_usize_param(sweep_value, "period")?,
            SweepParam::Fast => fast = to_usize_param(sweep_value, "fast")?,
            SweepParam::Slow => slow = to_usize_param(sweep_value, "slow")?,
            SweepParam::Offset => offset = sweep_value,
            SweepParam::Sigma => sigma = sweep_value,
        }

        if period == 0 {
            bail!("indicator period must be > 0");
        }

        let series = match spec.kind {
            IndicatorKind::Sma => sma(prices, period),
            IndicatorKind::Ema => ema(prices, period),
            IndicatorKind::Hma => hma(prices, period),
            IndicatorKind::Wma => wma(prices, period),
            IndicatorKind::Kama => {
                if fast == 0 || slow == 0 {
                    bail!("kama params fast/slow must be > 0");
                }
                if slow <= fast {
                    bail!("kama param slow must be greater than fast");
                }
                kama(prices, period, fast, slow)
            }
            IndicatorKind::Alma => {
                if !(0.0..=1.0).contains(&offset) {
                    bail!("alma param offset must be in [0, 1]");
                }
                if sigma <= 0.0 {
                    bail!("alma param sigma must be greater than 0");
                }
                alma(prices, period, offset, sigma)
            }
            IndicatorKind::Price => {
                // Price is just the previous close (t-1)
                // First value is f64::NAN since there's no previous price
                let mut price_series = vec![f64::NAN];
                price_series.extend_from_slice(&prices[..prices.len().saturating_sub(1)]);
                price_series
            }
        };
        out.insert(sweep_value_key(sweep_value), Arc::new(series));
    }
    Ok(out)
}

#[derive(Clone, Copy)]
struct IndicatorDefaults {
    period: usize,
    fast: usize,
    slow: usize,
    offset: f64,
    sigma: f64,
}

fn resolve_indicator_defaults(spec: &IndicatorSpec) -> IndicatorDefaults {
    let params = spec.params.unwrap_or(IndicatorParams {
        period: None,
        fast: None,
        slow: None,
        offset: None,
        sigma: None,
    });
    IndicatorDefaults {
        period: params.period.unwrap_or(10),
        fast: params.fast.unwrap_or(2),
        slow: params.slow.unwrap_or(30),
        offset: params.offset.unwrap_or(0.85),
        sigma: params.sigma.unwrap_or(6.0),
    }
}

fn to_usize_param(value: f64, label: &str) -> Result<usize> {
    if !value.is_finite() {
        bail!("{label} must be finite");
    }
    if value <= 0.0 {
        bail!("{label} must be > 0");
    }
    if (value - value.round()).abs() > 1e-9 {
        bail!("{label} must be an integer");
    }
    Ok(value.round() as usize)
}

fn sweep_value_key(value: f64) -> i64 {
    (round_sweep_value(value) * 1_000_000.0).round() as i64
}

fn build_combos(
    values_a: &[f64],
    values_b: &[f64],
    tp_options: &[Option<f64>],
    sl_options: &[Option<f64>],
) -> Vec<Combo> {
    let mut combos =
        Vec::with_capacity(values_a.len() * values_b.len() * tp_options.len() * sl_options.len());
    for &a in values_a {
        for &b in values_b {
            for &tp in tp_options {
                for &sl in sl_options {
                    combos.push(Combo {
                        a_value: a,
                        b_value: b,
                        take_profit: tp,
                        stop_loss: sl,
                    });
                }
            }
        }
    }
    combos
}

fn run_strategy(
    prices: &[f64],
    series_a: &[f64],
    series_b: &[f64],
    base_env: &EnvConfig,
    initial_balance: f64,
    buy_action: CrossAction,
    sell_action: CrossAction,
    take_profit: Option<f64>,
    stop_loss: Option<f64>,
    session_open: &[bool],
    margin_ok: &[bool],
    minutes_to_close: &[f64],
    fitness_weights: Option<&FitnessWeights>,
) -> MetricsPayload {
    let mut env = TradingEnv::new(prices[0], initial_balance, base_env.clone());

    let mut rewards = Vec::with_capacity(prices.len().saturating_sub(1));
    let mut equity_curve = Vec::with_capacity(prices.len().saturating_sub(1));

    let mut position: i32 = 0;
    let mut entry_price: Option<f64> = None;
    let mut trades = 0usize;

    for t in 1..prices.len() {
        let prev_idx = t - 1;
        let curr_idx = t;

        let prev_a = series_a.get(prev_idx).copied().unwrap_or(f64::NAN);
        let prev_b = series_b.get(prev_idx).copied().unwrap_or(f64::NAN);
        let curr_a = series_a.get(curr_idx).copied().unwrap_or(f64::NAN);
        let curr_b = series_b.get(curr_idx).copied().unwrap_or(f64::NAN);

        let mut action = Action::Hold;

        if let Some(entry) = entry_price {
            let current_price = prices[curr_idx];
            if let Some(tp) = take_profit {
                if tp > 0.0 && hit_take_profit(position, entry, current_price, tp) {
                    action = exit_action(position);
                }
            }
            if matches!(action, Action::Hold) {
                if let Some(sl) = stop_loss {
                    if sl > 0.0 && hit_stop_loss(position, entry, current_price, sl) {
                        action = exit_action(position);
                    }
                }
            }
        }

        if matches!(action, Action::Hold) {
            let buy_signal = cross_signal(buy_action, prev_a, prev_b, curr_a, curr_b);
            let sell_signal = cross_signal(sell_action, prev_a, prev_b, curr_a, curr_b);
            action = decide_action(position, buy_signal, sell_signal);
        }

        let step_ctx = StepContext {
            session_open: session_open.get(prev_idx).copied().unwrap_or(true),
            margin_ok: margin_ok.get(prev_idx).copied().unwrap_or(true),
            minutes_to_close: minutes_to_close.get(prev_idx).copied(),
        };

        let prev_pos = position;
        let (reward, _info) = env.step(action, prices[curr_idx], step_ctx);
        let state = env.state();
        position = state.position;
        entry_price = update_entry_price(prev_pos, entry_price, position, prices[curr_idx]);
        if position != prev_pos {
            trades += (position - prev_pos).abs() as usize;
        }

        rewards.push(reward);
        equity_curve.push(state.cash + state.unrealized_pnl);
    }

    let metrics = compute_metrics(&rewards, &equity_curve, initial_balance);
    let ending_equity = equity_curve.last().copied().unwrap_or(initial_balance);
    let net_pnl = ending_equity - initial_balance;
    let returns = equity_returns(&equity_curve, initial_balance);
    let sortino = compute_sortino(&returns, 252.0, 0.0, 50.0);

    let (w_pnl, w_sortino, w_mdd) = if let Some(weights) = fitness_weights {
        (weights.w_pnl, weights.w_sortino, weights.w_mdd)
    } else {
        (1.0, 1.0, 1.0)
    };
    let fitness = w_pnl * net_pnl + w_sortino * sortino - w_mdd * metrics.max_drawdown;

    MetricsPayload {
        total_reward: metrics.total_reward,
        net_pnl,
        ending_equity,
        sharpe: metrics.sharpe_ratio,
        sortino,
        max_drawdown: metrics.max_drawdown,
        profit_factor: metrics.profit_factor,
        win_rate: metrics.win_rate,
        trades,
        fitness,
        steps: rewards.len(),
    }
}

fn cross_signal(action: CrossAction, prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    if prev_a.is_nan() || prev_b.is_nan() || curr_a.is_nan() || curr_b.is_nan() {
        return false;
    }
    match action {
        CrossAction::Crossover => prev_a <= prev_b && curr_a > curr_b,
        CrossAction::Crossunder => prev_a >= prev_b && curr_a < curr_b,
    }
}

fn decide_action(position: i32, buy_signal: bool, sell_signal: bool) -> Action {
    if buy_signal && !sell_signal {
        if position <= 0 {
            if position == 0 {
                Action::Buy
            } else {
                Action::Revert
            }
        } else {
            Action::Hold
        }
    } else if sell_signal && !buy_signal {
        if position >= 0 {
            if position == 0 {
                Action::Sell
            } else {
                Action::Revert
            }
        } else {
            Action::Hold
        }
    } else if buy_signal && sell_signal {
        if position > 0 {
            Action::Sell
        } else if position < 0 {
            Action::Buy
        } else {
            Action::Hold
        }
    } else {
        Action::Hold
    }
}

fn exit_action(position: i32) -> Action {
    if position > 0 {
        Action::Sell
    } else if position < 0 {
        Action::Buy
    } else {
        Action::Hold
    }
}

fn hit_take_profit(position: i32, entry: f64, price: f64, tp: f64) -> bool {
    if position > 0 {
        price >= entry * (1.0 + tp / 100.0)
    } else if position < 0 {
        price <= entry * (1.0 - tp / 100.0)
    } else {
        false
    }
}

fn hit_stop_loss(position: i32, entry: f64, price: f64, sl: f64) -> bool {
    if position > 0 {
        price <= entry * (1.0 - sl / 100.0)
    } else if position < 0 {
        price >= entry * (1.0 + sl / 100.0)
    } else {
        false
    }
}

fn update_entry_price(
    prev_pos: i32,
    prev_entry: Option<f64>,
    new_pos: i32,
    price: f64,
) -> Option<f64> {
    if new_pos == 0 {
        return None;
    }
    if prev_pos == 0 {
        return Some(price);
    }

    let prev_entry = prev_entry.unwrap_or(price);

    if prev_pos > 0 && new_pos > 0 {
        if new_pos > prev_pos {
            let added = (new_pos - prev_pos) as f64;
            let total = prev_entry * (prev_pos as f64) + price * added;
            Some(total / (new_pos as f64))
        } else {
            Some(prev_entry)
        }
    } else if prev_pos < 0 && new_pos < 0 {
        let prev_abs = prev_pos.abs() as f64;
        let new_abs = new_pos.abs() as f64;
        if new_abs > prev_abs {
            let added = new_abs - prev_abs;
            let total = prev_entry * prev_abs + price * added;
            Some(total / new_abs)
        } else {
            Some(prev_entry)
        }
    } else {
        Some(price)
    }
}

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

fn load_parquet_slice(path: &PathBuf, offset: usize, limit: Option<usize>) -> Result<DataFrame> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open parquet {}", path.display()))?;
    let mut df = ParquetReader::new(file).finish()?;
    let total = df.height();
    let start = offset.min(total);
    let len = limit
        .unwrap_or(total.saturating_sub(start))
        .min(total.saturating_sub(start));
    if start > 0 || len < total {
        df = df.slice(start as i64, len);
    }
    Ok(df)
}

fn extract_data(df: &DataFrame, globex: bool) -> Result<MarketData> {
    let close = series_to_f64(df.column("close")?.as_materialized_series())?;

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

    Ok(MarketData {
        close,
        datetime_ns,
        session_open,
        minutes_to_close,
        margin_ok,
    })
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
            AnyValue::Boolean(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
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
            if minutes.is_finite() {
                minutes.max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

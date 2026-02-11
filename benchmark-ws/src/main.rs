use anyhow::{bail, Context, Result};
use chrono::{DateTime, Timelike, Utc};
use chrono_tz::America::New_York;
use clap::Parser;
use dotenvy::dotenv;
use futures_util::{SinkExt, StreamExt};
use midas_env::features::{compute_features_ohlcv, wma};
use polars::prelude::{AnyValue, DataFrame, ParquetReader, SerReader, Series};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::time;
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_SIM_WS_URL: &str = "wss://md.tradovateapi.com/v1/websocket";
const DEFAULT_SIM_REST_URL: &str = "https://demo.tradovateapi.com/v1";
const DEFAULT_LIVE_WS_URL: &str = "wss://md.tradovateapi.com/v1/websocket";
const DEFAULT_LIVE_REST_URL: &str = "https://live.tradovateapi.com/v1";

#[derive(Parser, Debug)]
#[command(
    name = "benchmark-ws",
    about = "NinjaScript-inspired execution engine (parquet backtest runner)"
)]
struct Cli {
    /// Path to strategy TOML config.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Optional override for parquet input file.
    #[arg(long)]
    data_file: Option<PathBuf>,

    /// Optional override for ticker symbol.
    #[arg(long)]
    ticker: Option<String>,

    /// Required safety confirmation for live mode.
    #[arg(long, default_value_t = false)]
    confirm_live: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct EngineConfig {
    mode: String,
    confirm_live: bool,
    data_file: PathBuf,
    ticker: String,
    sim_contract: String,
    sim_ws_url: String,
    sim_rest_url: String,
    live_ws_url: String,
    live_rest_url: String,
    sim_token_path: PathBuf,
    sim_account_spec: Option<String>,
    sim_account_id: Option<i64>,
    sim_history_bars: usize,
    sim_max_runtime_seconds: Option<u64>,
    sim_max_realtime_bars: Option<usize>,
    sim_heartbeat_ms: u64,
    sim_order_time_in_force: String,
    sim_dry_run: bool,
    initial_balance: f64,
    quantity: i32,
    commission_round_turn: f64,
    slippage_per_contract: f64,
    point_value: f64,
    tick_size: f64,
    hma_length: usize,
    min_angle: f64,
    angle_lookback: usize,
    longs_only: bool,
    use_risk_management: bool,
    use_risk_management_short_only: bool,
    take_profit_dollars: f64,
    stop_loss_dollars: f64,
    use_trailing_stop: bool,
    trail_trigger_ticks: f64,
    trail_offset_ticks: f64,
    use_bar_trailing: bool,
    bar_trail_trigger_ticks: f64,
    bar_trail_ticks_per_bar: f64,
    use_ny_hours: bool,
    ny_start_hour: u32,
    ny_start_minute: u32,
    ny_end_hour: u32,
    ny_end_minute: u32,
    bars_required_to_trade: usize,
    offset: usize,
    limit: Option<usize>,
    export_trades_csv: Option<PathBuf>,
    export_equity_csv: Option<PathBuf>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            mode: "backtest".to_string(),
            confirm_live: false,
            data_file: PathBuf::from("data/train/SPY0.parquet"),
            ticker: "SPY".to_string(),
            sim_contract: "SPY".to_string(),
            sim_ws_url: DEFAULT_SIM_WS_URL.to_string(),
            sim_rest_url: DEFAULT_SIM_REST_URL.to_string(),
            live_ws_url: DEFAULT_LIVE_WS_URL.to_string(),
            live_rest_url: DEFAULT_LIVE_REST_URL.to_string(),
            sim_token_path: PathBuf::from(".auth/bearer-token.json"),
            sim_account_spec: None,
            sim_account_id: None,
            sim_history_bars: 500,
            sim_max_runtime_seconds: None,
            sim_max_realtime_bars: None,
            sim_heartbeat_ms: 2500,
            sim_order_time_in_force: "Day".to_string(),
            sim_dry_run: false,
            initial_balance: 10_000.0,
            quantity: 1,
            commission_round_turn: 1.60,
            slippage_per_contract: 0.25,
            point_value: 1.0,
            tick_size: 0.25,
            hma_length: 255,
            min_angle: 7.0,
            angle_lookback: 7,
            longs_only: false,
            use_risk_management: true,
            use_risk_management_short_only: false,
            take_profit_dollars: 30.0,
            stop_loss_dollars: 15.0,
            use_trailing_stop: false,
            trail_trigger_ticks: 10.0,
            trail_offset_ticks: 0.0,
            use_bar_trailing: false,
            bar_trail_trigger_ticks: 10.0,
            bar_trail_ticks_per_bar: 2.0,
            use_ny_hours: true,
            ny_start_hour: 9,
            ny_start_minute: 30,
            ny_end_hour: 16,
            ny_end_minute: 0,
            bars_required_to_trade: 50,
            offset: 0,
            limit: None,
            export_trades_csv: None,
            export_equity_csv: None,
        }
    }
}

#[derive(Debug)]
struct MarketData {
    symbol: String,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    datetime_ns: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Long,
    Short,
}

#[derive(Debug, Clone)]
struct OpenPosition {
    side: Side,
    qty: i32,
    entry_price: f64,
    entry_idx: usize,
    entry_ts_ns: i64,
    bars_since_entry: usize,
    trail_activated: bool,
    bar_trail_activated: bool,
    current_stop_price: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Trade {
    side: String,
    qty: i32,
    entry_idx: usize,
    exit_idx: usize,
    entry_ts_ns: i64,
    exit_ts_ns: i64,
    entry_price: f64,
    exit_price: f64,
    pnl: f64,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct EquityPoint {
    idx: usize,
    ts_ns: i64,
    close: f64,
    equity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignalAction {
    Hold,
    EnterLong,
    EnterShort,
    ExitLongOnShortSignal,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    ticker: String,
    dataset_symbol: String,
    bars_in_dataset: usize,
    bars_processed: usize,
    trades: usize,
    wins: usize,
    losses: usize,
    win_rate: f64,
    gross_profit: f64,
    gross_loss: f64,
    net_pnl: f64,
    commissions_paid: f64,
    ending_equity: f64,
    max_drawdown: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Backtest,
    Sim,
    Live,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenData {
    token: String,
}

#[derive(Debug, Deserialize)]
struct TradovateAccount {
    id: i64,
    name: String,
}

#[derive(Debug, Clone)]
struct Bar {
    ts_ns: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SimRunSummary {
    mode: String,
    ticker: String,
    contract: String,
    account_spec: String,
    account_id: i64,
    bars_warmup: usize,
    bars_realtime_processed: usize,
    orders_sent: usize,
    orders_failed: usize,
    net_pnl_local: f64,
    ending_equity_local: f64,
}

struct LiveSeries {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    datetime_ns: Vec<i64>,
    forming_bar: Option<Bar>,
}

impl LiveSeries {
    fn new() -> Self {
        Self {
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            datetime_ns: Vec::new(),
            forming_bar: None,
        }
    }

    fn push_closed_bar(&mut self, bar: &Bar) {
        if let Some(last_ts) = self.datetime_ns.last().copied() {
            if bar.ts_ns == last_ts {
                let i = self.datetime_ns.len() - 1;
                self.open[i] = bar.open;
                self.high[i] = bar.high;
                self.low[i] = bar.low;
                self.close[i] = bar.close;
                return;
            }
            if bar.ts_ns < last_ts {
                return;
            }
        }
        self.datetime_ns.push(bar.ts_ns);
        self.open.push(bar.open);
        self.high.push(bar.high);
        self.low.push(bar.low);
        self.close.push(bar.close);
    }
}

struct StrategyEngine {
    cfg: EngineConfig,
    symbol: String,
    position: Option<OpenPosition>,
    realized_pnl: f64,
    commissions_paid: f64,
    trades: Vec<Trade>,
    equity: Vec<EquityPoint>,
}

impl StrategyEngine {
    fn new(cfg: EngineConfig, symbol: String) -> Self {
        Self {
            cfg,
            symbol,
            position: None,
            realized_pnl: 0.0,
            commissions_paid: 0.0,
            trades: Vec::new(),
            equity: Vec::new(),
        }
    }

    fn position_side(&self) -> Option<Side> {
        self.position.as_ref().map(|p| p.side)
    }

    fn slippage_price_offset(&self) -> f64 {
        if self.cfg.point_value.abs() < f64::EPSILON {
            return 0.0;
        }
        self.cfg.slippage_per_contract / self.cfg.point_value
    }

    fn commission_per_side(&self, qty: i32) -> f64 {
        (self.cfg.commission_round_turn / 2.0) * qty.max(0) as f64
    }

    fn unrealized_pnl(&self, close: f64) -> f64 {
        let Some(pos) = self.position.as_ref() else {
            return 0.0;
        };

        match pos.side {
            Side::Long => (close - pos.entry_price) * self.cfg.point_value * pos.qty as f64,
            Side::Short => (pos.entry_price - close) * self.cfg.point_value * pos.qty as f64,
        }
    }

    fn ending_equity_from_close(&self, close: f64) -> f64 {
        self.cfg.initial_balance + self.realized_pnl + self.unrealized_pnl(close)
    }

    fn upsert_equity_point(&mut self, idx: usize, ts_ns: i64, close: f64) {
        let equity = self.ending_equity_from_close(close);
        if let Some(last) = self.equity.last_mut() {
            if last.idx == idx {
                last.ts_ns = ts_ns;
                last.close = close;
                last.equity = equity;
                return;
            }
        }
        self.equity.push(EquityPoint {
            idx,
            ts_ns,
            close,
            equity,
        });
    }

    fn open_position(&mut self, side: Side, idx: usize, ts_ns: i64, market_price: f64) {
        let slip = self.slippage_price_offset();
        let qty = self.cfg.quantity;
        let fill_price = match side {
            Side::Long => market_price + slip,
            Side::Short => market_price - slip,
        };

        let commission = self.commission_per_side(qty);
        self.realized_pnl -= commission;
        self.commissions_paid += commission;

        self.position = Some(OpenPosition {
            side,
            qty,
            entry_price: fill_price,
            entry_idx: idx,
            entry_ts_ns: ts_ns,
            bars_since_entry: 0,
            trail_activated: false,
            bar_trail_activated: false,
            current_stop_price: None,
        });
    }

    fn close_position(&mut self, idx: usize, ts_ns: i64, market_price: f64, reason: &str) {
        let Some(pos) = self.position.take() else {
            return;
        };

        let slip = self.slippage_price_offset();
        let exit_price = match pos.side {
            Side::Long => market_price - slip,
            Side::Short => market_price + slip,
        };

        let gross_pnl = match pos.side {
            Side::Long => (exit_price - pos.entry_price) * self.cfg.point_value * pos.qty as f64,
            Side::Short => (pos.entry_price - exit_price) * self.cfg.point_value * pos.qty as f64,
        };

        let exit_commission = self.commission_per_side(pos.qty);
        self.realized_pnl += gross_pnl - exit_commission;
        self.commissions_paid += exit_commission;

        let trade_net_pnl = gross_pnl
            - self.commission_per_side(pos.qty) // entry side
            - exit_commission;

        self.trades.push(Trade {
            side: match pos.side {
                Side::Long => "long".to_string(),
                Side::Short => "short".to_string(),
            },
            qty: pos.qty,
            entry_idx: pos.entry_idx,
            exit_idx: idx,
            entry_ts_ns: pos.entry_ts_ns,
            exit_ts_ns: ts_ns,
            entry_price: pos.entry_price,
            exit_price,
            pnl: trade_net_pnl,
            reason: reason.to_string(),
        });
    }

    fn risk_enabled_for_side(&self, side: Side) -> bool {
        if !self.cfg.use_risk_management {
            return false;
        }
        match side {
            Side::Long => !self.cfg.use_risk_management_short_only,
            Side::Short => true,
        }
    }

    fn dollars_to_price_distance(&self, dollars: f64, qty: i32) -> Option<f64> {
        if dollars <= 0.0 {
            return None;
        }
        let denom = self.cfg.point_value * qty as f64;
        if denom <= 0.0 {
            return None;
        }
        Some(dollars / denom)
    }

    fn fixed_take_profit_price(&self, pos: &OpenPosition) -> Option<f64> {
        if !self.risk_enabled_for_side(pos.side) {
            return None;
        }
        let distance = self.dollars_to_price_distance(self.cfg.take_profit_dollars, pos.qty)?;
        match pos.side {
            Side::Long => Some(pos.entry_price + distance),
            Side::Short => Some(pos.entry_price - distance),
        }
    }

    fn fixed_stop_loss_price(&self, pos: &OpenPosition) -> Option<f64> {
        if !self.risk_enabled_for_side(pos.side) {
            return None;
        }
        let distance = self.dollars_to_price_distance(self.cfg.stop_loss_dollars, pos.qty)?;
        match pos.side {
            Side::Long => Some(pos.entry_price - distance),
            Side::Short => Some(pos.entry_price + distance),
        }
    }

    fn effective_stop_price(&self, pos: &OpenPosition) -> (Option<f64>, Option<&'static str>) {
        let fixed = self.fixed_stop_loss_price(pos);
        let trailing = pos.current_stop_price;

        match (fixed, trailing, pos.side) {
            (None, None, _) => (None, None),
            (Some(price), None, _) => (Some(price), Some("stop_loss")),
            (None, Some(price), _) => (Some(price), Some("trail_stop")),
            (Some(fixed_price), Some(trail_price), Side::Long) => {
                if trail_price >= fixed_price {
                    (Some(trail_price), Some("trail_stop"))
                } else {
                    (Some(fixed_price), Some("stop_loss"))
                }
            }
            (Some(fixed_price), Some(trail_price), Side::Short) => {
                if trail_price <= fixed_price {
                    (Some(trail_price), Some("trail_stop"))
                } else {
                    (Some(fixed_price), Some("stop_loss"))
                }
            }
        }
    }

    fn manage_trailing_stop(&mut self, close: f64) {
        let Some(pos) = self.position.as_mut() else {
            return;
        };

        if !self.cfg.use_trailing_stop && !self.cfg.use_bar_trailing {
            return;
        }

        let tick_size = self.cfg.tick_size;
        let current_pnl_ticks = match pos.side {
            Side::Long => (close - pos.entry_price) / tick_size,
            Side::Short => (pos.entry_price - close) / tick_size,
        };

        if self.cfg.use_trailing_stop && current_pnl_ticks >= self.cfg.trail_trigger_ticks {
            let candidate = match pos.side {
                Side::Long => pos.entry_price + self.cfg.trail_offset_ticks * tick_size,
                Side::Short => pos.entry_price - self.cfg.trail_offset_ticks * tick_size,
            };

            match (pos.side, pos.current_stop_price) {
                (Side::Long, Some(current)) if candidate <= current => {}
                (Side::Short, Some(current)) if candidate >= current => {}
                _ => {
                    pos.current_stop_price = Some(candidate);
                    pos.trail_activated = true;
                }
            }
        }

        if self.cfg.use_bar_trailing && current_pnl_ticks >= self.cfg.bar_trail_trigger_ticks {
            if !pos.bar_trail_activated {
                pos.bar_trail_activated = true;
                let initial = match pos.side {
                    Side::Long => pos.entry_price + self.cfg.trail_offset_ticks * tick_size,
                    Side::Short => pos.entry_price - self.cfg.trail_offset_ticks * tick_size,
                };

                pos.current_stop_price = match (pos.side, pos.current_stop_price) {
                    (Side::Long, Some(current)) => Some(current.max(initial)),
                    (Side::Short, Some(current)) => Some(current.min(initial)),
                    (_, None) => Some(initial),
                };
            } else {
                let step = self.cfg.bar_trail_ticks_per_bar * tick_size;
                pos.current_stop_price = Some(match (pos.side, pos.current_stop_price) {
                    (Side::Long, Some(current)) => current + step,
                    (Side::Short, Some(current)) => current - step,
                    (Side::Long, None) => pos.entry_price + self.cfg.trail_offset_ticks * tick_size,
                    (Side::Short, None) => {
                        pos.entry_price - self.cfg.trail_offset_ticks * tick_size
                    }
                });
            }
        }
    }

    /// Returns true when a protective order exits the position on this bar.
    fn apply_intrabar_exits(
        &mut self,
        idx: usize,
        ts_ns: i64,
        high: f64,
        low: f64,
        close: f64,
    ) -> bool {
        if self.position.is_none() {
            return false;
        }

        if let Some(pos) = self.position.as_mut() {
            pos.bars_since_entry = pos.bars_since_entry.saturating_add(1);
        }

        self.manage_trailing_stop(close);

        let Some(pos) = self.position.as_ref() else {
            return false;
        };

        let take_profit_price = self.fixed_take_profit_price(pos);
        let (stop_price, stop_reason) = self.effective_stop_price(pos);

        let stop_hit = match (pos.side, stop_price) {
            (Side::Long, Some(price)) if low.is_finite() => low <= price,
            (Side::Short, Some(price)) if high.is_finite() => high >= price,
            _ => false,
        };
        let tp_hit = match (pos.side, take_profit_price) {
            (Side::Long, Some(price)) if high.is_finite() => high >= price,
            (Side::Short, Some(price)) if low.is_finite() => low <= price,
            _ => false,
        };

        // Conservative ordering: when both touch in same bar, prioritize stop.
        if stop_hit {
            if let Some(price) = stop_price {
                self.close_position(idx, ts_ns, price, stop_reason.unwrap_or("stop_loss"));
                return true;
            }
        }

        if tp_hit {
            if let Some(price) = take_profit_price {
                self.close_position(idx, ts_ns, price, "take_profit");
                return true;
            }
        }

        false
    }

    fn apply_signal(&mut self, signal: SignalAction, idx: usize, ts_ns: i64, close: f64) {
        match signal {
            SignalAction::Hold => {}
            SignalAction::EnterLong => match self.position_side() {
                Some(Side::Long) => {}
                Some(Side::Short) => {
                    self.close_position(idx, ts_ns, close, "signal_reverse_long");
                    self.open_position(Side::Long, idx, ts_ns, close);
                }
                None => self.open_position(Side::Long, idx, ts_ns, close),
            },
            SignalAction::EnterShort => match self.position_side() {
                Some(Side::Short) => {}
                Some(Side::Long) => {
                    self.close_position(idx, ts_ns, close, "signal_reverse_short");
                    self.open_position(Side::Short, idx, ts_ns, close);
                }
                None => self.open_position(Side::Short, idx, ts_ns, close),
            },
            SignalAction::ExitLongOnShortSignal => {
                if self.position_side() == Some(Side::Long) {
                    self.close_position(idx, ts_ns, close, "long_exit_short_signal");
                }
            }
        }
    }

    fn close_open_at_end(&mut self, idx: usize, ts_ns: i64, close: f64) {
        if self.position.is_some() {
            self.close_position(idx, ts_ns, close, "end_of_data");
        }
        self.upsert_equity_point(idx, ts_ns, close);
    }

    fn summary(&self, bars_in_dataset: usize, dataset_symbol: &str) -> RunSummary {
        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;

        for trade in self.trades.iter() {
            if trade.pnl > 0.0 {
                wins += 1;
                gross_profit += trade.pnl;
            } else if trade.pnl < 0.0 {
                losses += 1;
                gross_loss += -trade.pnl;
            }
        }

        let ending_equity = self
            .equity
            .last()
            .map(|p| p.equity)
            .unwrap_or(self.cfg.initial_balance + self.realized_pnl);
        let net_pnl = ending_equity - self.cfg.initial_balance;
        let win_rate = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64
        } else {
            0.0
        };

        let max_drawdown = max_drawdown(self.equity.iter().map(|p| p.equity));

        RunSummary {
            ticker: self.symbol.clone(),
            dataset_symbol: dataset_symbol.to_string(),
            bars_in_dataset,
            bars_processed: self.equity.len(),
            trades: self.trades.len(),
            wins,
            losses,
            win_rate,
            gross_profit,
            gross_loss,
            net_pnl,
            commissions_paid: self.commissions_paid,
            ending_equity,
            max_drawdown,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = load_config(&cli)?;
    match parse_mode(&cfg.mode)? {
        RunMode::Backtest => run_backtest(cfg),
        RunMode::Sim => run_realtime(cfg, RunMode::Sim).await,
        RunMode::Live => run_realtime(cfg, RunMode::Live).await,
    }
}

fn run_backtest(cfg: EngineConfig) -> Result<()> {
    let market = load_market_data(&cfg.data_file, cfg.offset, cfg.limit)?;
    if market.close.len() < 2 {
        bail!("not enough bars to execute strategy");
    }

    let mut features =
        compute_features_ohlcv(&market.close, Some(&market.high), Some(&market.low), None);
    let atr_14 = features
        .remove("atr_14")
        .unwrap_or_else(|| vec![f64::NAN; market.close.len()]);
    let zero_hma = compute_zero_lag_hma(&market.close, cfg.hma_length);

    if market.symbol != "UNKNOWN" && !cfg.ticker.eq_ignore_ascii_case(&market.symbol) {
        bail!(
            "ticker mismatch: config ticker `{}` but dataset symbol is `{}`",
            cfg.ticker,
            market.symbol
        );
    }

    let symbol = if cfg.ticker.trim().is_empty() {
        market.symbol.clone()
    } else {
        cfg.ticker.clone()
    };

    let mut engine = StrategyEngine::new(cfg.clone(), symbol);
    engine.upsert_equity_point(0, market.datetime_ns[0], market.close[0]);

    let sqrt_length = (cfg.hma_length as f64).sqrt().round() as usize;
    let warmup = (cfg.hma_length + sqrt_length).max(cfg.bars_required_to_trade);

    for idx in 1..market.close.len() {
        let ts_ns = market.datetime_ns[idx];
        let bar_open = market.open[idx];
        let bar_high = market.high[idx];
        let bar_low = market.low[idx];
        let bar_close = market.close[idx];

        // First apply protective stops/targets based on this bar's range.
        engine.apply_intrabar_exits(idx, ts_ns, bar_high, bar_low, bar_close);

        if idx >= warmup {
            let signal = evaluate_signal(
                idx,
                &cfg,
                &market.close,
                &zero_hma,
                &atr_14,
                ts_ns,
                engine.position_side(),
            );
            let _ = bar_open; // keep available for future intra-bar fill policies.
            engine.apply_signal(signal, idx, ts_ns, bar_close);
        }

        engine.upsert_equity_point(idx, ts_ns, bar_close);
    }

    let last_idx = market.close.len() - 1;
    engine.close_open_at_end(
        last_idx,
        market.datetime_ns[last_idx],
        market.close[last_idx],
    );

    if let Some(path) = cfg.export_trades_csv.as_deref() {
        write_trades_csv(path, &engine.trades)?;
        println!("Wrote trades CSV to {}", path.display());
    }

    if let Some(path) = cfg.export_equity_csv.as_deref() {
        write_equity_csv(path, &engine.equity)?;
        println!("Wrote equity CSV to {}", path.display());
    }

    let summary = engine.summary(market.close.len(), &market.symbol);
    print_summary(&summary)?;
    Ok(())
}

fn evaluate_signal(
    idx: usize,
    cfg: &EngineConfig,
    close: &[f64],
    zero_hma: &[f64],
    atr_14: &[f64],
    ts_ns: i64,
    current_side: Option<Side>,
) -> SignalAction {
    let prev_close = close
        .get(idx.saturating_sub(1))
        .copied()
        .unwrap_or(f64::NAN);
    let curr_close = close.get(idx).copied().unwrap_or(f64::NAN);
    let prev_hma = zero_hma
        .get(idx.saturating_sub(1))
        .copied()
        .unwrap_or(f64::NAN);
    let curr_hma = zero_hma.get(idx).copied().unwrap_or(f64::NAN);

    if !prev_close.is_finite()
        || !curr_close.is_finite()
        || !prev_hma.is_finite()
        || !curr_hma.is_finite()
    {
        return SignalAction::Hold;
    }

    if idx < cfg.angle_lookback {
        return SignalAction::Hold;
    }
    let lookback_hma = zero_hma[idx - cfg.angle_lookback];
    if !lookback_hma.is_finite() || lookback_hma.abs() < f64::EPSILON {
        return SignalAction::Hold;
    }

    let atr = atr_14.get(idx).copied().unwrap_or(f64::NAN);
    if !atr.is_finite() || atr.abs() < f64::EPSILON {
        return SignalAction::Hold;
    }

    let price_change = curr_hma - lookback_hma;
    let slope = price_change / (atr * cfg.angle_lookback as f64);
    let angle = slope.atan().to_degrees();
    let is_steep_enough = angle.abs() >= cfg.min_angle;

    let price_cross_above = cross_above(prev_close, prev_hma, curr_close, curr_hma);
    let price_cross_below = cross_below(prev_close, prev_hma, curr_close, curr_hma);

    let trading_allowed = !cfg.use_ny_hours || is_within_ny_hours(ts_ns, cfg);
    let short_signal = price_cross_below && is_steep_enough && angle < 0.0;

    if price_cross_above
        && is_steep_enough
        && angle > 0.0
        && current_side != Some(Side::Long)
        && trading_allowed
    {
        return SignalAction::EnterLong;
    }

    if short_signal && !cfg.longs_only && current_side != Some(Side::Short) && trading_allowed {
        return SignalAction::EnterShort;
    }

    if cfg.longs_only && short_signal && current_side == Some(Side::Long) {
        return SignalAction::ExitLongOnShortSignal;
    }

    SignalAction::Hold
}

fn cross_above(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a <= prev_b && curr_a > curr_b
}

fn cross_below(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a >= prev_b && curr_a < curr_b
}

fn is_within_ny_hours(ts_ns: i64, cfg: &EngineConfig) -> bool {
    if ts_ns == 0 {
        return true;
    }

    let dt_utc = DateTime::<Utc>::from_timestamp_nanos(ts_ns);
    let dt_ny = dt_utc.with_timezone(&New_York);
    let current_minutes = dt_ny.hour() * 60 + dt_ny.minute();
    let start_minutes = cfg.ny_start_hour * 60 + cfg.ny_start_minute;
    let end_minutes = cfg.ny_end_hour * 60 + cfg.ny_end_minute;

    current_minutes >= start_minutes && current_minutes <= end_minutes
}

fn compute_zero_lag_hma(close: &[f64], hma_length: usize) -> Vec<f64> {
    if close.is_empty() {
        return Vec::new();
    }

    let half_length = ((hma_length as f64) / 2.0).ceil() as usize;
    let sqrt_length = (hma_length as f64).sqrt().round() as usize;
    let wma_half = wma(close, half_length.max(1));
    let wma_full = wma(close, hma_length.max(1));

    let mut out = vec![f64::NAN; close.len()];
    for idx in 0..close.len() {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..sqrt_length {
            if idx < i {
                break;
            }
            let hist_idx = idx - i;
            let wma1 = wma_half[hist_idx];
            let wma2 = wma_full[hist_idx];
            if !wma1.is_finite() || !wma2.is_finite() {
                continue;
            }

            let weight = (sqrt_length - i) as f64;
            let diff = 2.0 * wma1 - wma2;
            sum += diff * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            out[idx] = sum / weight_sum;
        }
    }

    out
}

fn max_drawdown(values: impl Iterator<Item = f64>) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for value in values {
        if value > peak {
            peak = value;
        }
        let dd = peak - value;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn print_summary(summary: &RunSummary) -> Result<()> {
    println!("\nExecution Summary");
    println!("Ticker:            {}", summary.ticker);
    println!("Dataset Symbol:    {}", summary.dataset_symbol);
    println!("Bars (dataset):    {}", summary.bars_in_dataset);
    println!("Bars (processed):  {}", summary.bars_processed);
    println!("Trades:            {}", summary.trades);
    println!("Wins/Losses:       {}/{}", summary.wins, summary.losses);
    println!("Win Rate:          {:.2}%", summary.win_rate * 100.0);
    println!("Gross Profit:      {:.2}", summary.gross_profit);
    println!("Gross Loss:        {:.2}", summary.gross_loss);
    println!("Net PnL:           {:.2}", summary.net_pnl);
    println!("Commissions:       {:.2}", summary.commissions_paid);
    println!("Ending Equity:     {:.2}", summary.ending_equity);
    println!("Max Drawdown:      {:.2}", summary.max_drawdown);
    println!("\nJSON:");
    println!("{}", serde_json::to_string_pretty(summary)?);
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
    }
    Ok(())
}

fn write_trades_csv(path: &Path, trades: &[Trade]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("open trades csv {}", path.display()))?;
    for trade in trades.iter() {
        writer.serialize(trade)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_equity_csv(path: &Path, equity: &[EquityPoint]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = csv::Writer::from_path(path)
        .with_context(|| format!("open equity csv {}", path.display()))?;
    for point in equity.iter() {
        writer.serialize(point)?;
    }
    writer.flush()?;
    Ok(())
}

fn load_config(cli: &Cli) -> Result<EngineConfig> {
    dotenv().ok();

    let mut cfg = EngineConfig::default();

    let config_path = cli
        .config
        .clone()
        .or_else(|| env_string("MIDAS_EXEC_CONFIG").map(PathBuf::from));

    if let Some(path) = config_path {
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read config file {}", path.display()))?;
        cfg = toml::from_str(&text).with_context(|| format!("parse TOML {}", path.display()))?;
    }

    apply_env_overrides(&mut cfg)?;

    if let Some(path) = cli.data_file.as_ref() {
        cfg.data_file = path.clone();
    }
    if let Some(ticker) = cli.ticker.as_ref() {
        cfg.ticker = ticker.clone();
    }
    if cli.confirm_live {
        cfg.confirm_live = true;
    }
    if cfg.sim_account_id == Some(0) {
        cfg.sim_account_id = None;
    }
    if cfg.sim_max_runtime_seconds == Some(0) {
        cfg.sim_max_runtime_seconds = None;
    }
    if cfg.sim_max_realtime_bars == Some(0) {
        cfg.sim_max_realtime_bars = None;
    }
    if cfg.sim_contract.trim().is_empty() {
        cfg.sim_contract = cfg.ticker.clone();
    }

    validate_config(&cfg)?;
    Ok(cfg)
}

fn apply_env_overrides(cfg: &mut EngineConfig) -> Result<()> {
    if let Some(v) = env_string("MIDAS_EXEC_MODE") {
        cfg.mode = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_CONFIRM_LIVE")? {
        cfg.confirm_live = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_DATA_FILE") {
        cfg.data_file = PathBuf::from(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_TICKER") {
        cfg.ticker = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_CONTRACT") {
        cfg.sim_contract = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_WS_URL") {
        cfg.sim_ws_url = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_REST_URL") {
        cfg.sim_rest_url = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_LIVE_WS_URL") {
        cfg.live_ws_url = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_LIVE_REST_URL") {
        cfg.live_rest_url = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_TOKEN_PATH") {
        cfg.sim_token_path = PathBuf::from(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_TOKEN_PATH") {
        cfg.sim_token_path = PathBuf::from(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_ACCOUNT_SPEC") {
        cfg.sim_account_spec = Some(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_ACCOUNT_SPEC") {
        cfg.sim_account_spec = Some(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_ACCOUNT_ID") {
        cfg.sim_account_id = parse_optional_i64("MIDAS_EXEC_SIM_ACCOUNT_ID", &v)?;
    }
    if let Some(v) = env_string("MIDAS_EXEC_ACCOUNT_ID") {
        cfg.sim_account_id = parse_optional_i64("MIDAS_EXEC_ACCOUNT_ID", &v)?;
    }
    if let Some(v) = env_parse::<usize>("MIDAS_EXEC_SIM_HISTORY_BARS")? {
        cfg.sim_history_bars = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_MAX_RUNTIME_SECONDS") {
        cfg.sim_max_runtime_seconds = parse_optional_u64("MIDAS_EXEC_SIM_MAX_RUNTIME_SECONDS", &v)?;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_MAX_REALTIME_BARS") {
        cfg.sim_max_realtime_bars = parse_optional_usize("MIDAS_EXEC_SIM_MAX_REALTIME_BARS", &v)?;
    }
    if let Some(v) = env_parse::<u64>("MIDAS_EXEC_SIM_HEARTBEAT_MS")? {
        cfg.sim_heartbeat_ms = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_SIM_ORDER_TIF") {
        cfg.sim_order_time_in_force = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_SIM_DRY_RUN")? {
        cfg.sim_dry_run = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_INITIAL_BALANCE")? {
        cfg.initial_balance = v;
    }
    if let Some(v) = env_parse::<i32>("MIDAS_EXEC_QUANTITY")? {
        cfg.quantity = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_COMMISSION_ROUND_TURN")? {
        cfg.commission_round_turn = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_SLIPPAGE_PER_CONTRACT")? {
        cfg.slippage_per_contract = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_POINT_VALUE")? {
        cfg.point_value = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_TICK_SIZE")? {
        cfg.tick_size = v;
    }
    if let Some(v) = env_parse::<usize>("MIDAS_EXEC_HMA_LENGTH")? {
        cfg.hma_length = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_MIN_ANGLE")? {
        cfg.min_angle = v;
    }
    if let Some(v) = env_parse::<usize>("MIDAS_EXEC_ANGLE_LOOKBACK")? {
        cfg.angle_lookback = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_LONGS_ONLY")? {
        cfg.longs_only = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_USE_RISK_MANAGEMENT")? {
        cfg.use_risk_management = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_USE_RISK_MANAGEMENT_SHORT_ONLY")? {
        cfg.use_risk_management_short_only = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_TAKE_PROFIT_DOLLARS")? {
        cfg.take_profit_dollars = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_STOP_LOSS_DOLLARS")? {
        cfg.stop_loss_dollars = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_USE_TRAILING_STOP")? {
        cfg.use_trailing_stop = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_TRAIL_TRIGGER_TICKS")? {
        cfg.trail_trigger_ticks = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_TRAIL_OFFSET_TICKS")? {
        cfg.trail_offset_ticks = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_USE_BAR_TRAILING")? {
        cfg.use_bar_trailing = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_BAR_TRAIL_TRIGGER_TICKS")? {
        cfg.bar_trail_trigger_ticks = v;
    }
    if let Some(v) = env_parse::<f64>("MIDAS_EXEC_BAR_TRAIL_TICKS_PER_BAR")? {
        cfg.bar_trail_ticks_per_bar = v;
    }
    if let Some(v) = env_bool("MIDAS_EXEC_USE_NY_HOURS")? {
        cfg.use_ny_hours = v;
    }
    if let Some(v) = env_parse::<u32>("MIDAS_EXEC_NY_START_HOUR")? {
        cfg.ny_start_hour = v;
    }
    if let Some(v) = env_parse::<u32>("MIDAS_EXEC_NY_START_MINUTE")? {
        cfg.ny_start_minute = v;
    }
    if let Some(v) = env_parse::<u32>("MIDAS_EXEC_NY_END_HOUR")? {
        cfg.ny_end_hour = v;
    }
    if let Some(v) = env_parse::<u32>("MIDAS_EXEC_NY_END_MINUTE")? {
        cfg.ny_end_minute = v;
    }
    if let Some(v) = env_parse::<usize>("MIDAS_EXEC_BARS_REQUIRED_TO_TRADE")? {
        cfg.bars_required_to_trade = v;
    }
    if let Some(v) = env_parse::<usize>("MIDAS_EXEC_OFFSET")? {
        cfg.offset = v;
    }
    if let Some(v) = env_string("MIDAS_EXEC_LIMIT") {
        cfg.limit = parse_optional_usize("MIDAS_EXEC_LIMIT", &v)?;
    }
    if let Some(v) = env_string("MIDAS_EXEC_EXPORT_TRADES_CSV") {
        cfg.export_trades_csv = parse_optional_path(v);
    }
    if let Some(v) = env_string("MIDAS_EXEC_EXPORT_EQUITY_CSV") {
        cfg.export_equity_csv = parse_optional_path(v);
    }

    Ok(())
}

fn validate_config(cfg: &EngineConfig) -> Result<()> {
    let mode = parse_mode(&cfg.mode)?;
    if cfg.quantity <= 0 {
        bail!("quantity must be > 0");
    }
    if cfg.hma_length < 2 {
        bail!("hma_length must be >= 2");
    }
    if cfg.angle_lookback == 0 {
        bail!("angle_lookback must be >= 1");
    }
    if cfg.tick_size <= 0.0 {
        bail!("tick_size must be > 0");
    }
    if cfg.point_value <= 0.0 {
        bail!("point_value must be > 0");
    }
    if cfg.ny_start_hour > 23 || cfg.ny_end_hour > 23 {
        bail!("NY hour fields must be in [0, 23]");
    }
    if cfg.ny_start_minute > 59 || cfg.ny_end_minute > 59 {
        bail!("NY minute fields must be in [0, 59]");
    }

    if matches!(mode, RunMode::Sim | RunMode::Live) {
        if cfg.sim_history_bars == 0 {
            bail!("sim_history_bars must be > 0 in realtime modes");
        }
        if cfg.sim_heartbeat_ms == 0 {
            bail!("sim_heartbeat_ms must be > 0 in realtime modes");
        }
        let ws_url = if matches!(mode, RunMode::Live) {
            &cfg.live_ws_url
        } else {
            &cfg.sim_ws_url
        };
        let rest_url = if matches!(mode, RunMode::Live) {
            &cfg.live_rest_url
        } else {
            &cfg.sim_rest_url
        };
        if ws_url.trim().is_empty() {
            bail!("websocket URL must not be empty in realtime mode");
        }
        if rest_url.trim().is_empty() {
            bail!("REST URL must not be empty in realtime mode");
        }
        if cfg.sim_token_path.as_os_str().is_empty() {
            bail!("sim_token_path must not be empty in realtime mode");
        }
    }
    if matches!(mode, RunMode::Live) && !cfg.confirm_live {
        bail!(
            "live mode requires explicit confirmation: set `confirm_live = true`, pass `--confirm-live`, or set `MIDAS_EXEC_CONFIRM_LIVE=true`"
        );
    }
    Ok(())
}

fn env_string(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Err(_) => None,
    }
}

fn env_parse<T>(key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(raw) = env_string(key) else {
        return Ok(None);
    };
    let parsed = raw
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {key} value `{raw}`: {err}"))?;
    Ok(Some(parsed))
}

fn env_bool(key: &str) -> Result<Option<bool>> {
    let Some(raw) = env_string(key) else {
        return Ok(None);
    };
    let normalized = raw.to_ascii_lowercase();
    let parsed = match normalized.as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => {
            bail!("invalid {key} value `{raw}` (expected true/false/1/0)");
        }
    };
    Ok(Some(parsed))
}

fn parse_optional_usize(key: &str, raw: &str) -> Result<Option<usize>> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "none" || normalized == "null" {
        return Ok(None);
    }
    let value = normalized
        .parse::<usize>()
        .map_err(|err| anyhow::anyhow!("invalid {key} value `{raw}`: {err}"))?;
    Ok(Some(value))
}

fn parse_optional_i64(key: &str, raw: &str) -> Result<Option<i64>> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "none" || normalized == "null" {
        return Ok(None);
    }
    let value = normalized
        .parse::<i64>()
        .map_err(|err| anyhow::anyhow!("invalid {key} value `{raw}`: {err}"))?;
    Ok(Some(value))
}

fn parse_optional_u64(key: &str, raw: &str) -> Result<Option<u64>> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "none" || normalized == "null" {
        return Ok(None);
    }
    let value = normalized
        .parse::<u64>()
        .map_err(|err| anyhow::anyhow!("invalid {key} value `{raw}`: {err}"))?;
    Ok(Some(value))
}

fn parse_optional_path(raw: String) -> Option<PathBuf> {
    let normalized = raw.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "none" || normalized == "null" {
        None
    } else {
        Some(PathBuf::from(raw))
    }
}

fn load_market_data(path: &Path, offset: usize, limit: Option<usize>) -> Result<MarketData> {
    let file = fs::File::open(path).with_context(|| format!("open parquet {}", path.display()))?;
    let mut df = ParquetReader::new(file).finish()?;

    let total = df.height();
    let start = offset.min(total);
    let len = limit
        .unwrap_or(total.saturating_sub(start))
        .min(total.saturating_sub(start));
    if start > 0 || len < total {
        df = df.slice(start as i64, len);
    }

    extract_market_data(&df)
}

fn extract_market_data(df: &DataFrame) -> Result<MarketData> {
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
    let datetime_ns = df
        .column("date")
        .ok()
        .map(|c| series_to_i64(c.as_materialized_series()))
        .transpose()?
        .unwrap_or_else(|| vec![0_i64; close.len()]);

    let symbol = df
        .column("symbol")
        .ok()
        .and_then(|c| c.get(0).ok())
        .and_then(anyvalue_to_string)
        .unwrap_or_else(|| "UNKNOWN".to_string());

    Ok(MarketData {
        symbol,
        open,
        high,
        low,
        close,
        datetime_ns,
    })
}

fn anyvalue_to_string(value: AnyValue<'_>) -> Option<String> {
    match value {
        AnyValue::String(v) => Some(v.to_string()),
        AnyValue::StringOwned(v) => Some(v.to_string()),
        _ => None,
    }
}

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    let out = series
        .iter()
        .map(|value| match value {
            AnyValue::Float64(v) => v,
            AnyValue::Float32(v) => v as f64,
            AnyValue::Int64(v) => v as f64,
            AnyValue::Int32(v) => v as f64,
            AnyValue::UInt64(v) => v as f64,
            AnyValue::UInt32(v) => v as f64,
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

fn series_to_i64(series: &Series) -> Result<Vec<i64>> {
    let out = series
        .iter()
        .map(|value| match value {
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

fn parse_mode(mode: &str) -> Result<RunMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "backtest" => Ok(RunMode::Backtest),
        "sim" => Ok(RunMode::Sim),
        "live" => Ok(RunMode::Live),
        other => bail!("invalid mode `{other}` (expected `backtest`, `sim`, or `live`)"),
    }
}

fn load_bearer_token(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read bearer token file {}", path.display()))?;
    let data: TokenData =
        serde_json::from_str(&raw).with_context(|| "parse bearer token JSON (expected {token})")?;
    if data.token.trim().is_empty() {
        bail!("token file has empty token field: {}", path.display());
    }
    Ok(data.token)
}

fn parse_status_code(msg: &Value) -> Option<i64> {
    if let Some(code) = msg.get("s").and_then(|v| v.as_i64()) {
        return Some(code);
    }
    msg.get("s")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<i64>().ok())
}

fn parse_frame(raw: &str) -> (char, Option<Value>) {
    let mut chars = raw.chars();
    let frame_type = chars.next().unwrap_or('\0');
    let offset = frame_type.len_utf8();
    let payload = raw.get(offset..).unwrap_or("");
    let data = if payload.is_empty() {
        None
    } else {
        serde_json::from_str(payload).ok()
    };
    (frame_type, data)
}

fn create_message(endpoint: &str, id: u64, body: Option<&Value>) -> String {
    if let Some(body) = body {
        format!("{endpoint}\n{id}\n\n{}", body)
    } else {
        format!("{endpoint}\n{id}\n\n")
    }
}

fn normalize_rest_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn json_number(value: &Value, key: &str) -> Option<f64> {
    let raw = value.get(key)?;
    if let Some(v) = raw.as_f64() {
        return Some(v);
    }
    if let Some(v) = raw.as_i64() {
        return Some(v as f64);
    }
    if let Some(v) = raw.as_u64() {
        return Some(v as f64);
    }
    raw.as_str().and_then(|s| s.parse::<f64>().ok())
}

fn parse_bar(value: &Value) -> Option<Bar> {
    let ts_raw = value.get("timestamp")?.as_str()?;
    let ts_ns = chrono::DateTime::parse_from_rfc3339(ts_raw)
        .ok()?
        .with_timezone(&Utc)
        .timestamp_nanos_opt()?;

    Some(Bar {
        ts_ns,
        open: json_number(value, "open")?,
        high: json_number(value, "high")?,
        low: json_number(value, "low")?,
        close: json_number(value, "close")?,
    })
}

fn position_to_signed(side: Option<Side>, qty: i32) -> i32 {
    match side {
        Some(Side::Long) => qty,
        Some(Side::Short) => -qty,
        None => 0,
    }
}

async fn find_contract_id(client: &Client, rest_url: &str, token: &str, name: &str) -> Result<i64> {
    let url = format!(
        "{}/contract/find?name={}",
        normalize_rest_url(rest_url),
        name
    );
    let response = client.get(&url).bearer_auth(token).send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("contract lookup failed ({status}): {body}");
    }

    let payload: Value = response.json().await?;
    if let Some(id) = payload.get("id").and_then(|v| v.as_i64()) {
        return Ok(id);
    }
    if let Some(first) = payload.as_array().and_then(|arr| arr.first()) {
        if let Some(id) = first.get("id").and_then(|v| v.as_i64()) {
            return Ok(id);
        }
    }
    bail!("contract lookup response missing id for `{name}`")
}

async fn resolve_account(
    client: &Client,
    rest_url: &str,
    token: &str,
    mode: RunMode,
    preferred_spec: Option<String>,
    preferred_id: Option<i64>,
) -> Result<(String, i64)> {
    if let (Some(spec), Some(id)) = (preferred_spec.clone(), preferred_id) {
        return Ok((spec, id));
    }

    let url = format!("{}/account/list", normalize_rest_url(rest_url));
    let response = client.get(&url).bearer_auth(token).send().await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("account list failed ({status}): {body}");
    }

    let accounts: Vec<TradovateAccount> = response.json().await.unwrap_or_default();
    if accounts.is_empty() {
        bail!("no accounts returned by Tradovate account/list");
    }

    if let Some(id) = preferred_id {
        if let Some(acc) = accounts.iter().find(|a| a.id == id) {
            return Ok((acc.name.clone(), acc.id));
        }
        bail!("requested sim_account_id `{id}` not found in account/list");
    }

    if let Some(spec) = preferred_spec {
        if let Some(acc) = accounts.iter().find(|a| a.name.eq_ignore_ascii_case(&spec)) {
            return Ok((acc.name.clone(), acc.id));
        }
        bail!("requested sim_account_spec `{spec}` not found in account/list");
    }

    if matches!(mode, RunMode::Sim) {
        if let Some(acc) = accounts.iter().find(|a| {
            let up = a.name.to_ascii_uppercase();
            up.contains("SIM") || up.contains("DEMO")
        }) {
            return Ok((acc.name.clone(), acc.id));
        }
    } else if matches!(mode, RunMode::Live) {
        if let Some(acc) = accounts.iter().find(|a| {
            let up = a.name.to_ascii_uppercase();
            !up.contains("SIM") && !up.contains("DEMO")
        }) {
            return Ok((acc.name.clone(), acc.id));
        }
    }

    let first = accounts
        .into_iter()
        .next()
        .context("missing account entry")?;
    Ok((first.name, first.id))
}

struct SimBroker {
    client: Client,
    rest_url: String,
    token: String,
    account_spec: String,
    account_id: i64,
    symbol: String,
    time_in_force: String,
    dry_run: bool,
    orders_sent: usize,
    orders_failed: usize,
}

impl SimBroker {
    async fn apply_position_change(
        &mut self,
        previous: Option<Side>,
        current: Option<Side>,
        qty: i32,
        reason: &str,
    ) {
        let prev = position_to_signed(previous, qty);
        let next = position_to_signed(current, qty);
        let delta = next - prev;
        if delta == 0 {
            return;
        }

        let action = if delta > 0 { "Buy" } else { "Sell" };
        let order_qty = delta.unsigned_abs() as i32;
        if let Err(err) = self.place_market_order(action, order_qty, reason).await {
            self.orders_failed = self.orders_failed.saturating_add(1);
            eprintln!("order failed ({action} {order_qty}) reason={reason}: {err}");
        }
    }

    async fn place_market_order(
        &mut self,
        action: &str,
        order_qty: i32,
        reason: &str,
    ) -> Result<()> {
        if self.dry_run {
            self.orders_sent = self.orders_sent.saturating_add(1);
            println!(
                "[sim-dry-run] {action} {order_qty} {} ({reason})",
                self.symbol
            );
            return Ok(());
        }

        let url = format!("{}/order/placeorder", self.rest_url);
        let payload = json!({
            "accountSpec": self.account_spec,
            "accountId": self.account_id,
            "action": action,
            "symbol": self.symbol,
            "orderQty": order_qty,
            "orderType": "Market",
            "timeInForce": self.time_in_force,
            "isAutomated": true
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.token)
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("placeorder HTTP {}: {}", status, body);
        }

        let parsed: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!({ "raw": body }));
        if let Some(failure) = parsed.get("failureReason").and_then(|v| v.as_str()) {
            if !failure.trim().is_empty() {
                bail!("placeorder rejected: {}", failure);
            }
        }
        if let Some(err_text) = parsed.get("errorText").and_then(|v| v.as_str()) {
            if !err_text.trim().is_empty() {
                bail!("placeorder errorText: {}", err_text);
            }
        }

        self.orders_sent = self.orders_sent.saturating_add(1);
        println!("[sim] {action} {order_qty} {} ({reason})", self.symbol);
        Ok(())
    }
}

async fn run_realtime(cfg: EngineConfig, mode: RunMode) -> Result<()> {
    let token = load_bearer_token(&cfg.sim_token_path)?;
    let (rest_url, ws_url, mode_name) = match mode {
        RunMode::Sim => (
            normalize_rest_url(&cfg.sim_rest_url),
            cfg.sim_ws_url.clone(),
            "sim",
        ),
        RunMode::Live => (
            normalize_rest_url(&cfg.live_rest_url),
            cfg.live_ws_url.clone(),
            "live",
        ),
        RunMode::Backtest => unreachable!(),
    };
    let contract = if cfg.sim_contract.trim().is_empty() {
        cfg.ticker.clone()
    } else {
        cfg.sim_contract.clone()
    };

    let client = Client::new();
    let (account_spec, account_id) = resolve_account(
        &client,
        &rest_url,
        &token,
        mode,
        cfg.sim_account_spec.clone(),
        cfg.sim_account_id,
    )
    .await?;
    let contract_id = find_contract_id(&client, &rest_url, &token, &contract).await?;

    println!("Mode: {}", mode_name);
    println!("Ticker: {}", cfg.ticker);
    println!("Contract: {} (id {})", contract, contract_id);
    println!("Account: {} ({})", account_spec, account_id);
    println!("WebSocket: {}", ws_url);
    println!("REST: {}", rest_url);

    let mut broker = SimBroker {
        client: client.clone(),
        rest_url,
        token: token.clone(),
        account_spec: account_spec.clone(),
        account_id,
        symbol: contract.clone(),
        time_in_force: cfg.sim_order_time_in_force.clone(),
        dry_run: cfg.sim_dry_run,
        orders_sent: 0,
        orders_failed: 0,
    };

    let mut engine = StrategyEngine::new(cfg.clone(), cfg.ticker.clone());
    let mut series = LiveSeries::new();
    let sqrt_length = (cfg.hma_length as f64).sqrt().round() as usize;
    let warmup = (cfg.hma_length + sqrt_length).max(cfg.bars_required_to_trade);

    let (ws_stream, _) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .with_context(|| format!("connect websocket {}", ws_url))?;
    let (mut write, mut read) = ws_stream.split();
    let mut message_id: u64 = 0;
    let authorize_id: u64;
    let mut chart_req_id: Option<u64> = None;
    let mut historical_id: Option<i64> = None;
    let mut realtime_id: Option<i64> = None;
    let mut authorized = false;

    message_id += 1;
    authorize_id = message_id;
    write
        .send(Message::Text(format!(
            "authorize\n{}\n\n{}",
            message_id, token
        )))
        .await?;

    let mut heartbeat = time::interval(Duration::from_millis(cfg.sim_heartbeat_ms.max(250)));
    heartbeat.tick().await;

    let started = Instant::now();
    let runtime_limit = cfg.sim_max_runtime_seconds.map(Duration::from_secs);
    let mut realtime_bars_processed = 0usize;
    let mut warmup_bars = 0usize;

    loop {
        if let Some(limit) = runtime_limit {
            if started.elapsed() >= limit {
                println!("{mode_name} runtime limit reached");
                break;
            }
        }
        if let Some(max_bars) = cfg.sim_max_realtime_bars {
            if realtime_bars_processed >= max_bars {
                println!("{mode_name} realtime bar limit reached");
                break;
            }
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!("received ctrl-c, stopping {} loop", mode_name);
                break;
            }
            _ = heartbeat.tick(), if authorized => {
                let _ = write.send(Message::Text("[]".to_string())).await;
            }
            maybe_msg = read.next() => {
                let raw = match maybe_msg {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(Message::Binary(bytes))) => String::from_utf8_lossy(&bytes).to_string(),
                    Some(Ok(Message::Close(_))) => {
                        println!("websocket closed by server");
                        break;
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(err)) => {
                        bail!("websocket read error: {err}");
                    }
                    None => {
                        println!("websocket stream ended");
                        break;
                    }
                };

                let (frame_type, payload) = parse_frame(&raw);
                if frame_type != 'a' {
                    continue;
                }
                let Some(Value::Array(items)) = payload else {
                    continue;
                };

                for item in items {
                    let status = parse_status_code(&item);
                    let response_id = item.get("i").and_then(|v| v.as_u64());

                    if !authorized && status == Some(200) && response_id == Some(authorize_id) {
                        authorized = true;
                        message_id += 1;
                        chart_req_id = Some(message_id);
                        let req = json!({
                            "symbol": contract_id,
                            "chartDescription": {
                                "underlyingType": "MinuteBar",
                                "elementSize": 1,
                                "elementSizeUnit": "UnderlyingUnits",
                                "withHistogram": false
                            },
                            "timeRange": { "asMuchAsElements": cfg.sim_history_bars }
                        });
                        let request = create_message("md/getChart", message_id, Some(&req));
                        write.send(Message::Text(request)).await?;
                        continue;
                    }

                    if status == Some(200) && response_id == chart_req_id {
                        if let Some(d) = item.get("d") {
                            historical_id = d.get("historicalId").and_then(|v| v.as_i64()).or(historical_id);
                            realtime_id = d.get("realtimeId").and_then(|v| v.as_i64()).or(realtime_id);
                        }
                    }

                    let Some(charts) = item
                        .get("d")
                        .and_then(|d| d.get("charts"))
                        .and_then(|c| c.as_array())
                    else {
                        continue;
                    };

                    for chart in charts {
                        let chart_id = chart.get("id").and_then(|v| v.as_i64());
                        let Some(bars) = chart.get("bars").and_then(|b| b.as_array()) else {
                            continue;
                        };

                        for bar_json in bars {
                            let Some(bar) = parse_bar(bar_json) else {
                                continue;
                            };

                            let is_historical = chart_id.is_some() && historical_id.is_some() && chart_id == historical_id;
                            let is_realtime = chart_id.is_some() && realtime_id.is_some() && chart_id == realtime_id;

                            if is_historical || (historical_id.is_none() && realtime_id.is_none()) {
                                series.push_closed_bar(&bar);
                                warmup_bars = series.close.len();
                                continue;
                            }

                            if is_realtime {
                                if let Some(ref mut forming) = series.forming_bar {
                                    if bar.ts_ns == forming.ts_ns {
                                        *forming = bar;
                                        continue;
                                    }
                                    if bar.ts_ns < forming.ts_ns {
                                        continue;
                                    }

                                    let closed = forming.clone();
                                    *forming = bar;
                                    series.push_closed_bar(&closed);
                                    let idx = series.close.len().saturating_sub(1);

                                    if idx == 0 {
                                        engine.upsert_equity_point(
                                            0,
                                            series.datetime_ns[0],
                                            series.close[0],
                                        );
                                    } else {
                                        let prev_side = engine.position_side();
                                        engine.apply_intrabar_exits(
                                            idx,
                                            series.datetime_ns[idx],
                                            series.high[idx],
                                            series.low[idx],
                                            series.close[idx],
                                        );
                                        let post_exit_side = engine.position_side();
                                        broker
                                            .apply_position_change(prev_side, post_exit_side, cfg.quantity, "protective")
                                            .await;

                                        let mut feats = compute_features_ohlcv(
                                            &series.close,
                                            Some(&series.high),
                                            Some(&series.low),
                                            None,
                                        );
                                        let atr_14 = feats
                                            .remove("atr_14")
                                            .unwrap_or_else(|| vec![f64::NAN; series.close.len()]);
                                        let zero_hma = compute_zero_lag_hma(&series.close, cfg.hma_length);

                                        if idx >= warmup {
                                            let signal = evaluate_signal(
                                                idx,
                                                &cfg,
                                                &series.close,
                                                &zero_hma,
                                                &atr_14,
                                                series.datetime_ns[idx],
                                                engine.position_side(),
                                            );
                                            let before_signal_side = engine.position_side();
                                            engine.apply_signal(
                                                signal,
                                                idx,
                                                series.datetime_ns[idx],
                                                series.close[idx],
                                            );
                                            let after_signal_side = engine.position_side();
                                            broker
                                                .apply_position_change(before_signal_side, after_signal_side, cfg.quantity, "signal")
                                                .await;
                                        }

                                        engine.upsert_equity_point(
                                            idx,
                                            series.datetime_ns[idx],
                                            series.close[idx],
                                        );
                                    }

                                    realtime_bars_processed = realtime_bars_processed.saturating_add(1);
                                } else {
                                    series.forming_bar = Some(bar);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(last) = series.close.last().copied() {
        let idx = series.close.len().saturating_sub(1);
        let ts = series.datetime_ns.last().copied().unwrap_or(0);
        engine.close_open_at_end(idx, ts, last);
    }

    if let Some(path) = cfg.export_trades_csv.as_deref() {
        write_trades_csv(path, &engine.trades)?;
        println!("Wrote trades CSV to {}", path.display());
    }
    if let Some(path) = cfg.export_equity_csv.as_deref() {
        write_equity_csv(path, &engine.equity)?;
        println!("Wrote equity CSV to {}", path.display());
    }

    let ending_equity = engine
        .equity
        .last()
        .map(|p| p.equity)
        .unwrap_or(cfg.initial_balance + engine.realized_pnl);
    let sim_summary = SimRunSummary {
        mode: mode_name.to_string(),
        ticker: cfg.ticker.clone(),
        contract,
        account_spec,
        account_id,
        bars_warmup: warmup_bars,
        bars_realtime_processed: realtime_bars_processed,
        orders_sent: broker.orders_sent,
        orders_failed: broker.orders_failed,
        net_pnl_local: ending_equity - cfg.initial_balance,
        ending_equity_local: ending_equity,
    };

    println!(
        "\n{} Summary",
        if matches!(mode, RunMode::Live) {
            "Live"
        } else {
            "Sim"
        }
    );
    println!("{}", serde_json::to_string_pretty(&sim_summary)?);
    Ok(())
}

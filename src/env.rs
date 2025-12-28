//! Trading environment with discrete actions and observation builder.

use crate::features::{compute_features_ohlcv, periods, ATR_PERIODS};
use chrono::{DateTime, Timelike, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    Revert,
}

#[derive(Debug, Clone)]
pub struct EnvConfig {
    /// Round-turn commission per contract (USD).
    pub commission_round_turn: f64,
    /// Slippage per contract (USD).
    pub slippage_per_contract: f64,
    /// Maximum absolute position (contracts).
    pub max_position: i32,
    /// Margin required per contract (USD).
    pub margin_per_contract: f64,
    /// Enforce margin check internally (else rely on passed StepContext margin_ok).
    pub enforce_margin: bool,
    /// Session open flag default (used if StepContext not provided).
    pub default_session_open: bool,
    /// Penalty for holding inventory relative to max_position.
    pub risk_penalty: f64,
    /// Small penalty to discourage idling.
    pub idle_penalty: f64,
    /// Base penalty per step while in drawdown and holding a position.
    pub drawdown_penalty: f64,
    /// Linear growth multiplier for drawdown penalty per consecutive step.
    pub drawdown_penalty_growth: f64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            commission_round_turn: 1.60,
            slippage_per_contract: 0.25,
            max_position: 1,
            margin_per_contract: 50.0,
            enforce_margin: true,
            default_session_open: true,
            risk_penalty: 0.0,
            idle_penalty: 0.0,
            drawdown_penalty: 0.0,
            drawdown_penalty_growth: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepContext {
    /// Whether trading session is open (e.g., RTH). If false and action trades, apply hard penalty.
    pub session_open: bool,
    /// Whether margin is sufficient. If false, treat as margin call violation.
    pub margin_ok: bool,
}

impl Default for StepContext {
    fn default() -> Self {
        Self {
            session_open: true,
            margin_ok: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnvState {
    pub step: usize,
    pub position: i32,
    pub cash: f64,
    pub last_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub equity_peak: f64,
    pub drawdown_steps: usize,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct StepInfo {
    pub commission_paid: f64,
    pub slippage_paid: f64,
    pub pnl_change: f64,
    pub realized_pnl_change: f64,
    pub drawdown_penalty: f64,
    pub margin_call_violation: bool,
    pub position_limit_violation: bool,
    pub session_closed_violation: bool,
}

pub struct TradingEnv {
    cfg: EnvConfig,
    state: EnvState,
}

impl TradingEnv {
    pub fn new(initial_price: f64, initial_balance: f64, cfg: EnvConfig) -> Self {
        Self {
            cfg,
            state: EnvState {
                step: 0,
                position: 0,
                cash: initial_balance,
                last_price: initial_price,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
                equity_peak: initial_balance,
                drawdown_steps: 0,
                done: false,
            },
        }
    }

    pub fn reset(&mut self, initial_price: f64, initial_balance: f64) {
        self.state = EnvState {
            step: 0,
            position: 0,
            cash: initial_balance,
            last_price: initial_price,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            equity_peak: initial_balance,
            drawdown_steps: 0,
            done: false,
        };
    }

    pub fn state(&self) -> &EnvState {
        &self.state
    }

    /// Apply an action using the next market price and context, returning reward and info.
    pub fn step(&mut self, action: Action, next_price: f64, ctx: StepContext) -> (f64, StepInfo) {
        let session_open = ctx.session_open && self.cfg.default_session_open;
        if !session_open && !matches!(action, Action::Hold) {
            return (
                -1000.0,
                StepInfo {
                    commission_paid: 0.0,
                    slippage_paid: 0.0,
                    pnl_change: 0.0,
                    realized_pnl_change: 0.0,
                    drawdown_penalty: 0.0,
                    margin_call_violation: false,
                    position_limit_violation: false,
                    session_closed_violation: true,
                },
            );
        }

        if !ctx.margin_ok {
            return (
                -1000.0,
                StepInfo {
                    commission_paid: 0.0,
                    slippage_paid: 0.0,
                    pnl_change: 0.0,
                    realized_pnl_change: 0.0,
                    drawdown_penalty: 0.0,
                    margin_call_violation: true,
                    position_limit_violation: false,
                    session_closed_violation: false,
                },
            );
        }

        let target_position = match action {
            Action::Buy => (self.state.position + 1).clamp(-self.cfg.max_position, self.cfg.max_position),
            Action::Sell => (self.state.position - 1).clamp(-self.cfg.max_position, self.cfg.max_position),
            Action::Hold => self.state.position,
            Action::Revert => {
                if self.state.position == 0 {
                    0
                } else {
                    -self.state.position
                }
            }
        };

        let position_limit_violation = target_position.abs() > self.cfg.max_position;
        if position_limit_violation {
            return (
                -1000.0,
                StepInfo {
                    commission_paid: 0.0,
                    slippage_paid: 0.0,
                    pnl_change: 0.0,
                    realized_pnl_change: 0.0,
                    drawdown_penalty: 0.0,
                    margin_call_violation: false,
                    position_limit_violation,
                    session_closed_violation: false,
                },
            );
        }

        // Margin check if enabled.
        if self.cfg.enforce_margin {
            let required_margin = (target_position.abs() as f64) * self.cfg.margin_per_contract;
            let equity = self.state.cash + self.state.unrealized_pnl;
            if equity < required_margin {
                return (
                    -1000.0,
                    StepInfo {
                        commission_paid: 0.0,
                        slippage_paid: 0.0,
                        pnl_change: 0.0,
                        realized_pnl_change: 0.0,
                        drawdown_penalty: 0.0,
                        margin_call_violation: true,
                        position_limit_violation: false,
                        session_closed_violation: false,
                    },
                );
            }
        }

        let delta_pos = target_position - self.state.position;
        let commission_per_side = self.cfg.commission_round_turn / 2.0;
        let commission_paid = commission_per_side * (delta_pos.abs() as f64);
        let slippage_paid = self.cfg.slippage_per_contract * (delta_pos.abs() as f64);

        let price_change = next_price - self.state.last_price;
        let pnl_change = price_change * (self.state.position as f64);

        let trade_costs = commission_paid + slippage_paid;
        self.state.cash -= trade_costs;
        self.state.unrealized_pnl += pnl_change;
        let closing = self.state.position != 0
            && (target_position == 0
                || self.state.position.signum() != target_position.signum());
        let mut realized_pnl_change = 0.0;
        if closing {
            realized_pnl_change = self.state.unrealized_pnl;
            self.state.cash += self.state.unrealized_pnl;
            self.state.realized_pnl += self.state.unrealized_pnl;
            self.state.unrealized_pnl = 0.0;
        }
        self.state.position = target_position;
        self.state.last_price = next_price;
        self.state.step += 1;

        let equity = self.state.cash + self.state.unrealized_pnl;
        if self.state.position == 0 {
            self.state.drawdown_steps = 0;
            self.state.equity_peak = equity;
        } else if equity >= self.state.equity_peak {
            self.state.equity_peak = equity;
            self.state.drawdown_steps = 0;
        } else {
            self.state.drawdown_steps = self.state.drawdown_steps.saturating_add(1);
        }

        let drawdown_penalty = if self.state.position != 0 && self.state.drawdown_steps > 0 {
            let steps = self.state.drawdown_steps as f64;
            self.cfg.drawdown_penalty * steps * (1.0 + self.cfg.drawdown_penalty_growth * (steps - 1.0).max(0.0))
        } else {
            0.0
        };

        let mut reward = pnl_change
            - trade_costs
            - self.cfg.risk_penalty * (self.state.position.abs() as f64 / self.cfg.max_position as f64);

        if self.state.position == 0 && self.cfg.idle_penalty > 0.0 {
            reward -= self.cfg.idle_penalty;
        }
        if drawdown_penalty > 0.0 {
            reward -= drawdown_penalty;
        }

        (
            reward,
            StepInfo {
                commission_paid,
                slippage_paid,
                pnl_change,
                realized_pnl_change,
                drawdown_penalty,
                margin_call_violation: false,
                position_limit_violation: false,
                session_closed_violation: false,
            },
        )
    }
}

/// Build observation vector for step t (decision at start of bar t).
/// - Uses features up to t-1 (so pass slices ending at t-1).
pub fn build_observation(
    idx: usize,
    open: Option<&[f64]>,
    close: &[f64],
    high: &[f64],
    low: &[f64],
    volume: Option<&[f64]>,
    datetime_ns: Option<&[i64]>,
    session_open: Option<&[bool]>,
    margin_ok: Option<&[bool]>,
    position: i32,
    equity: f64,
) -> Vec<f64> {
    let feats = compute_features_ohlcv(close, Some(high), Some(low), volume);
    let mut obs = Vec::with_capacity(4 + feats.len() + 4);
    
    // Current bar's open (the "right now" price)
    if let Some(o) = open {
        obs.push(o[idx]);
    } else if idx > 0 {
        obs.push(close[idx - 1]);
    } else {
        obs.push(f64::NAN);
    }

    // price context t-1
    if idx > 0 {
        obs.push(close[idx - 1]);
        if let Some(vol) = volume {
            obs.push(vol[idx - 1]);
        }
    } else {
        obs.push(f64::NAN);
        obs.push(f64::NAN);
    }
    
    // Equity/Balance feature
    obs.push(equity);

    // indicators at t-1 (or NaN during warmup)
    for period in periods() {
        obs.push(*feats[&format!("sma_{period}")].get(idx - 1).unwrap_or(&f64::NAN));
        obs.push(*feats[&format!("ema_{period}")].get(idx - 1).unwrap_or(&f64::NAN));
        obs.push(*feats[&format!("hma_{period}")].get(idx - 1).unwrap_or(&f64::NAN));
    }
    for p in ATR_PERIODS {
        obs.push(*feats[&format!("atr_{p}")].get(idx - 1).unwrap_or(&f64::NAN));
    }

    // time encoding
    if let Some(ns) = datetime_ns.and_then(|d| d.get(idx - 1)) {
        let dt = DateTime::<Utc>::from_timestamp_nanos(*ns);
        let hour = dt.hour() as f64 + dt.minute() as f64 / 60.0;
        let angle = 2.0 * std::f64::consts::PI * (hour / 24.0);
        obs.push(angle.sin());
        obs.push(angle.cos());
    } else {
        obs.push(f64::NAN);
        obs.push(f64::NAN);
    }

    // position
    obs.push(position as f64);

    // masks
    let session_val = session_open
        .and_then(|m| m.get(idx - 1))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    let margin_val = margin_ok
        .and_then(|m| m.get(idx - 1))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    obs.push(session_val);
    obs.push(margin_val);

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revert_flips_and_charges_double_side_costs() {
        let mut env = TradingEnv::new(100.0, 1000.0, EnvConfig { max_position: 1, enforce_margin: false, ..Default::default() });
        let (_r1, _i1) = env.step(Action::Buy, 100.0, StepContext::default());
        assert_eq!(env.state.position, 1);

        let (reward, info) = env.step(Action::Revert, 100.0, StepContext::default());
        assert_eq!(env.state.position, -1);
        assert!(info.commission_paid > 0.0);
        assert!(reward < 0.0);
    }

    #[test]
    fn hold_incurs_no_trade_costs() {
        let mut env = TradingEnv::new(100.0, 1000.0, EnvConfig::default());
        let (_r, info) = env.step(Action::Hold, 101.0, StepContext::default());
        assert_eq!(info.commission_paid, 0.0);
        assert_eq!(env.state.position, 0);
    }

    #[test]
    fn session_closed_blocks_trading() {
        let mut env = TradingEnv::new(100.0, 1000.0, EnvConfig::default());
        let (r, info) = env.step(
            Action::Buy,
            100.0,
            StepContext {
                session_open: false,
                margin_ok: true,
            },
        );
        assert!(info.session_closed_violation);
        assert!(r < -999.0);
        assert_eq!(env.state.position, 0);
    }

    #[test]
    fn observation_includes_time_and_position() {
        let close = vec![1.0, 2.0, 3.0, 4.0];
        let high = close.clone();
        let low = close.clone();
        let now = Utc::now().timestamp_nanos_opt().unwrap();
        let dt = vec![now, now, now, now];
        let sess = vec![true, true, true, true];
        let margin = vec![true, true, true, true];
        let obs = build_observation(3, None, &close, &high, &low, None, Some(&dt), Some(&sess), Some(&margin), 1, 1000.0);
        assert!(obs.len() > 0);
        assert_eq!(obs.last().cloned().unwrap(), 1.0); // margin mask
    }
}

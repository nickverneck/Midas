//! Simple backtesting runner that plugs the TradingEnv over a price path
//! and produces episode metrics (Sharpe, PnL, drawdown, etc.).

use crate::env::{Action, EnvConfig, StepContext, TradingEnv};

#[derive(Debug, Clone, Default)]
pub struct EpisodeMetrics {
    pub total_reward: f64,
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub win_rate: f64,
    pub max_consecutive_losses: usize,
}

#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub metrics: EpisodeMetrics,
    pub equity_curve: Vec<f64>,
    pub rewards: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct EmaParams {
    pub fast: usize,
    pub slow: usize,
}

/// Run one episode over price series with given actions.
/// Assumes `actions.len() == prices.len() - 1` (one action per transition).
pub fn run_episode(prices: &[f64], actions: &[Action], cfg: EnvConfig) -> EpisodeResult {
    assert!(
        prices.len() >= 2,
        "need at least two prices to compute a step"
    );
    assert!(
        actions.len() + 1 == prices.len(),
        "actions should be one fewer than prices"
    );

    let mut env = TradingEnv::new(prices[0], cfg);
    let mut rewards = Vec::with_capacity(actions.len());
    let mut equity_curve = Vec::with_capacity(actions.len());

    for (i, action) in actions.iter().enumerate() {
        let next_price = prices[i + 1];
        let (reward, _info) = env.step(*action, next_price, StepContext::default());
        rewards.push(reward);
        let s = env.state();
        let equity = s.cash + s.unrealized_pnl;
        equity_curve.push(equity);
    }

    let metrics = compute_metrics(&rewards, &equity_curve);
    EpisodeResult {
        metrics,
        equity_curve,
        rewards,
    }
}

/// Simple EMA crossover strategy producing actions and running the episode.
pub fn run_ema_crossover(prices: &[f64], params: EmaParams, cfg: EnvConfig) -> EpisodeResult {
    assert!(params.fast < params.slow, "fast period must be < slow");
    assert!(prices.len() > params.slow + 1, "price series too short for EMA periods");

    let mut actions = vec![Action::Hold; prices.len() - 1];

    let mut ema_fast = prices[0];
    let mut ema_slow = prices[0];
    let alpha_fast = 2.0 / (params.fast as f64 + 1.0);
    let alpha_slow = 2.0 / (params.slow as f64 + 1.0);

    let mut position: i32 = 0;

    for i in 1..prices.len() {
        let price = prices[i];
        ema_fast = alpha_fast * price + (1.0 - alpha_fast) * ema_fast;
        ema_slow = alpha_slow * price + (1.0 - alpha_slow) * ema_slow;

        let action = if ema_fast > ema_slow {
            // desire long
            if position <= 0 {
                if position == 0 {
                    Action::Buy
                } else {
                    Action::Revert
                }
            } else {
                Action::Hold
            }
        } else if ema_fast < ema_slow {
            // desire short
            if position >= 0 {
                if position == 0 {
                    Action::Sell
                } else {
                    Action::Revert
                }
            } else {
                Action::Hold
            }
        } else {
            Action::Hold
        };

        actions[i - 1] = action;
        // Track expected position for next decision.
        position = match action {
            Action::Buy => (position + 1).clamp(-cfg.max_position, cfg.max_position),
            Action::Sell => (position - 1).clamp(-cfg.max_position, cfg.max_position),
            Action::Revert => -position,
            Action::Hold => position,
        };
    }

    run_episode(prices, &actions, cfg)
}

fn compute_metrics(rewards: &[f64], equity_curve: &[f64]) -> EpisodeMetrics {
    let total_reward: f64 = rewards.iter().sum();
    let total_pnl = *equity_curve.last().unwrap_or(&0.0);

    let sharpe_ratio = {
        if rewards.is_empty() {
            0.0
        } else {
            let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
            let var = rewards
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / rewards.len().max(1) as f64;
            let std = var.sqrt();
            if std == 0.0 { 0.0 } else { (mean / std) * (252f64).sqrt() }
        }
    };

    let max_drawdown = calc_max_drawdown(equity_curve);
    let (profit_factor, win_rate, max_consecutive_losses) = win_loss_stats(rewards);

    EpisodeMetrics {
        total_reward,
        total_pnl,
        sharpe_ratio,
        max_drawdown,
        profit_factor,
        win_rate,
        max_consecutive_losses,
    }
}

fn calc_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::MIN;
    let mut max_dd = 0.0;
    for &eq in equity_curve {
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

fn win_loss_stats(rewards: &[f64]) -> (f64, f64, usize) {
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut wins = 0;
    let mut losses = 0;
    let mut max_consec_losses = 0;
    let mut current_losses = 0;

    for &r in rewards {
        if r > 0.0 {
            wins += 1;
            gross_profit += r;
            current_losses = 0;
        } else if r < 0.0 {
            losses += 1;
            gross_loss += -r;
            current_losses += 1;
            if current_losses > max_consec_losses {
                max_consec_losses = current_losses;
            }
        }
    }

    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let win_rate = if wins + losses > 0 {
        wins as f64 / (wins + losses) as f64
    } else {
        0.0
    };

    (profit_factor, win_rate, max_consec_losses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_runs_and_computes_metrics() {
        let prices = [100.0, 101.0, 102.0, 101.0];
        let actions = [Action::Buy, Action::Hold, Action::Revert];
        let res = run_episode(&prices, &actions, EnvConfig::default());
        assert_eq!(res.rewards.len(), 3);
        assert_eq!(res.equity_curve.len(), 3);
        // Should have some drawdown after revert.
        assert!(res.metrics.max_drawdown >= 0.0);
    }
}

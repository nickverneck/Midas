use anyhow::Result;
use rand::Rng;
use tch::{kind::Kind, no_grad, Tensor};

use crate::data::{build_observation, DataSet};
use crate::metrics::{compute_sortino, max_drawdown};
use crate::model::{build_mlp, load_params_from_vec};

#[derive(Clone)]
pub struct CandidateConfig {
    pub initial_balance: f64,
    pub margin_per_contract: f64,
    pub disable_margin: bool,
    pub w_pnl: f64,
    pub w_sortino: f64,
    pub w_mdd: f64,
    pub sortino_annualization: f64,
    pub hidden: usize,
    pub layers: usize,
    pub eval_windows: usize,
    pub device: tch::Device,
    pub ignore_session: bool,
    pub drawdown_penalty: f64,
    pub drawdown_penalty_growth: f64,
}

#[derive(Clone)]
pub struct CandidateResult {
    pub fitness: f64,
    pub eval_pnl: f64,
    pub eval_pnl_realized: f64,
    pub eval_pnl_total: f64,
    pub eval_sortino: f64,
    pub eval_drawdown: f64,
    pub eval_ret_mean: f64,
    pub debug_non_hold: usize,
    pub debug_non_zero_pos: usize,
    pub debug_mean_abs_pnl: f64,
    pub debug_buy: usize,
    pub debug_sell: usize,
    pub debug_hold: usize,
    pub debug_revert: usize,
    pub debug_session_violations: usize,
    pub debug_margin_violations: usize,
    pub debug_position_violations: usize,
    pub debug_drawdown_penalty: f64,
}

pub fn evaluate_candidate(
    genome: &[f32],
    data: &DataSet,
    windows: &[(usize, usize)],
    cfg: &CandidateConfig,
    capture_history: bool,
) -> CandidateResult {
    use midas_env::env::{Action, EnvConfig, StepContext, TradingEnv};
    use tch::nn::Module;

    let vs = tch::nn::VarStore::new(cfg.device);
    let policy = build_mlp(&vs.root(), data.obs_dim as i64, cfg.hidden as i64, cfg.layers);
    load_params_from_vec(&vs, genome);

    let env_cfg = EnvConfig {
        margin_per_contract: cfg.margin_per_contract,
        enforce_margin: !cfg.disable_margin,
        drawdown_penalty: cfg.drawdown_penalty,
        drawdown_penalty_growth: cfg.drawdown_penalty_growth,
        ..EnvConfig::default()
    };

    let mut eval_pnls = Vec::new();
    let mut eval_pnls_realized = Vec::new();
    let mut eval_pnls_total = Vec::new();
    let mut eval_returns = Vec::new();
    let mut eval_equity: Vec<Vec<f64>> = Vec::new();
    let mut non_hold = 0usize;
    let mut non_zero_pos = 0usize;
    let mut abs_pnl_sum = 0.0f64;
    let mut pnl_steps = 0usize;
    let mut act_buy = 0usize;
    let mut act_sell = 0usize;
    let mut act_hold = 0usize;
    let mut act_revert = 0usize;
    let mut session_violations = 0usize;
    let mut margin_violations = 0usize;
    let mut position_violations = 0usize;
    let mut drawdown_penalty_sum = 0.0f64;

    for &(start, end) in windows.iter().take(cfg.eval_windows) {
        if end <= start + 1 {
            continue;
        }
        let mut env = TradingEnv::new(data.close[start], cfg.initial_balance, env_cfg.clone());
        let mut position = 0;
        let mut equity = cfg.initial_balance;
        let mut pnl_buf = Vec::with_capacity(end - start - 1);
        let mut eq_curve = Vec::with_capacity(end - start - 1);
        let mut window_drawdown_penalty = 0.0f64;

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
                0 => {
                    act_buy += 1;
                    Action::Buy
                }
                1 => {
                    act_sell += 1;
                    Action::Sell
                }
                2 => {
                    act_hold += 1;
                    Action::Hold
                }
                _ => {
                    act_revert += 1;
                    Action::Revert
                }
            };

            let session_open = if cfg.ignore_session {
                true
            } else {
                data.session_open
                    .as_ref()
                    .and_then(|m| m.get(t))
                    .copied()
                    .unwrap_or(true)
            };
            let margin_ok = *data.margin_ok.get(t).unwrap_or(&true);

            let (_reward, info) = env.step(
                action,
                data.close[t],
                StepContext {
                    session_open,
                    margin_ok,
                },
            );
            if info.session_closed_violation {
                session_violations += 1;
            }
            if info.margin_call_violation {
                margin_violations += 1;
            }
            if info.position_limit_violation {
                position_violations += 1;
            }
            position = env.state().position;
            equity = env.state().cash + env.state().unrealized_pnl;
            pnl_buf.push(info.pnl_change);
            eq_curve.push(equity);
            window_drawdown_penalty += info.drawdown_penalty;
            if !matches!(action, Action::Hold) {
                non_hold += 1;
            }
            if position != 0 {
                non_zero_pos += 1;
            }
            abs_pnl_sum += info.pnl_change.abs();
            pnl_steps += 1;
        }

        let realized_pnl = env.state().realized_pnl;
        let total_pnl = env.state().cash + env.state().unrealized_pnl - cfg.initial_balance;
        let pnl_sum = realized_pnl - window_drawdown_penalty;
        eval_pnls.push(pnl_sum);
        eval_pnls_realized.push(realized_pnl);
        eval_pnls_total.push(total_pnl);
        drawdown_penalty_sum += window_drawdown_penalty;

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
    let eval_pnl_realized = if eval_pnls_realized.is_empty() {
        0.0
    } else {
        eval_pnls_realized.iter().sum::<f64>() / eval_pnls_realized.len() as f64
    };
    let eval_pnl_total = if eval_pnls_total.is_empty() {
        0.0
    } else {
        eval_pnls_total.iter().sum::<f64>() / eval_pnls_total.len() as f64
    };

    let fitness = cfg.w_pnl * eval_pnl + cfg.w_sortino * eval_sortino - cfg.w_mdd * eval_draw;

    CandidateResult {
        fitness,
        eval_pnl,
        eval_pnl_realized,
        eval_pnl_total,
        eval_sortino,
        eval_drawdown: eval_draw,
        eval_ret_mean: if eval_returns.is_empty() {
            0.0
        } else {
            eval_returns.iter().sum::<f64>() / eval_returns.len() as f64
        },
        debug_non_hold: non_hold,
        debug_non_zero_pos: non_zero_pos,
        debug_mean_abs_pnl: if pnl_steps > 0 {
            abs_pnl_sum / pnl_steps as f64
        } else {
            0.0
        },
        debug_buy: act_buy,
        debug_sell: act_sell,
        debug_hold: act_hold,
        debug_revert: act_revert,
        debug_session_violations: session_violations,
        debug_margin_violations: margin_violations,
        debug_position_violations: position_violations,
        debug_drawdown_penalty: drawdown_penalty_sum,
    }
}

pub fn crossover(a: &[f32], b: &[f32], rng: &mut impl Rng) -> Vec<f32> {
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

pub fn save_checkpoint(path: &std::path::Path, pop: &[Vec<f32>]) -> Result<()> {
    let data = bincode::serialize(pop)?;
    std::fs::write(path, data)?;
    Ok(())
}

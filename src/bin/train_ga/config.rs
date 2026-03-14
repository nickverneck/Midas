use midas_env::env::MarginMode;
use midas_env::ml::ComputeRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ExecutionTarget {
    Cpu,
    Cuda(usize),
    Mps,
}

impl ExecutionTarget {
    pub fn effective_runtime(self) -> ComputeRuntime {
        match self {
            Self::Cpu => ComputeRuntime::Cpu,
            Self::Cuda(_) => ComputeRuntime::Cuda,
            Self::Mps => ComputeRuntime::Mps,
        }
    }

    pub fn is_accelerated(self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Mps)
    }

    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda(idx) => format!("cuda:{idx}"),
            Self::Mps => "mps".to_string(),
        }
    }

    #[cfg(feature = "torch")]
    pub fn as_tch_device(self) -> tch::Device {
        match self {
            Self::Cpu => tch::Device::Cpu,
            Self::Cuda(idx) => tch::Device::Cuda(idx),
            Self::Mps => tch::Device::Mps,
        }
    }
}

#[derive(Clone)]
pub struct CandidateConfig {
    pub initial_balance: f64,
    pub max_position: i32,
    pub margin_mode: MarginMode,
    pub contract_multiplier: f64,
    pub margin_per_contract: f64,
    pub disable_margin: bool,
    pub w_pnl: f64,
    pub w_sortino: f64,
    pub w_mdd: f64,
    pub sortino_annualization: f64,
    pub hidden: usize,
    pub layers: usize,
    pub eval_windows: usize,
    pub device: ExecutionTarget,
    pub ignore_session: bool,
    pub drawdown_penalty: f64,
    pub drawdown_penalty_growth: f64,
    pub session_close_penalty: f64,
    pub auto_close_minutes_before_close: f64,
    pub max_hold_bars_positive: usize,
    pub max_hold_bars_drawdown: usize,
    pub hold_duration_penalty: f64,
    pub hold_duration_penalty_growth: f64,
    pub hold_duration_penalty_positive_scale: f64,
    pub hold_duration_penalty_negative_scale: f64,
    pub invalid_revert_penalty: f64,
    pub invalid_revert_penalty_growth: f64,
    pub flat_hold_penalty: f64,
    pub flat_hold_penalty_growth: f64,
    pub max_flat_hold_bars: usize,
}

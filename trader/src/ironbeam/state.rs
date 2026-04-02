use crate::broker::{
    AccountInfo, AccountSnapshot, ContractSuggestion, LatencySnapshot, MarketSnapshot,
};
use crate::config::AppConfig;
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategy::{ExecutionRuntimeSnapshot, ExecutionStrategyConfig};
use reqwest::Client;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Instant;
use tokio::task::JoinHandle;

pub(super) enum InternalEvent {
    StreamPayload(String),
    StreamStatus(String),
    StreamError(String),
    RefreshAccountState { reason: Option<String> },
}

pub(super) struct IronbeamState {
    pub(super) client: Client,
    pub(super) session: Option<IronbeamSession>,
    pub(super) latency: LatencySnapshot,
}

pub(super) struct IronbeamSession {
    pub(super) cfg: AppConfig,
    pub(super) token: String,
    pub(super) user_name: Option<String>,
    pub(super) accounts: Vec<AccountInfo>,
    pub(super) selected_account_id: Option<i64>,
    pub(super) selected_contract: Option<ContractSuggestion>,
    pub(super) account_state: AccountState,
    pub(super) account_snapshots: Vec<AccountSnapshot>,
    pub(super) execution_config: ExecutionStrategyConfig,
    pub(super) execution_runtime: ExecutionRuntimeState,
    pub(super) managed_protection: BTreeMap<ProtectionKey, ManagedProtectionOrders>,
    pub(super) market: MarketSnapshot,
    pub(super) market_task: Option<JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub(super) struct PendingNativeReversalEntry {
    pub(super) target_qty: i32,
    pub(super) reason: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ExecutionRuntimeState {
    pub(super) armed: bool,
    pub(super) last_closed_bar_ts: Option<i64>,
    pub(super) pending_target_qty: Option<i32>,
    pub(super) pending_target_started_at: Option<Instant>,
    pub(super) pending_reversal_entry: Option<PendingNativeReversalEntry>,
    pub(super) last_summary: String,
    pub(super) hma_execution: HmaAngleExecutionState,
    pub(super) ema_execution: EmaCrossExecutionState,
}

impl ExecutionRuntimeState {
    pub(super) fn snapshot(&self) -> ExecutionRuntimeSnapshot {
        ExecutionRuntimeSnapshot {
            armed: self.armed,
            last_closed_bar_ts: self.last_closed_bar_ts,
            pending_target_qty: self.pending_target_qty,
            last_summary: self.last_summary.clone(),
        }
    }

    pub(super) fn reset_execution(&mut self) {
        self.pending_reversal_entry = None;
        self.hma_execution = HmaAngleExecutionState::default();
        self.ema_execution = EmaCrossExecutionState::default();
    }

    pub(super) fn set_pending_target(&mut self, target_qty: Option<i32>) {
        self.pending_target_qty = target_qty;
        self.pending_target_started_at = target_qty.map(|_| Instant::now());
    }

    pub(super) fn clear_pending_target(&mut self) {
        self.pending_target_qty = None;
        self.pending_target_started_at = None;
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct AccountState {
    pub(super) balances: BTreeMap<String, Value>,
    pub(super) positions: BTreeMap<String, BTreeMap<String, Value>>,
    pub(super) risks: BTreeMap<String, Value>,
    pub(super) orders: BTreeMap<String, BTreeMap<String, Value>>,
    pub(super) fills: BTreeMap<String, BTreeMap<String, Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct ProtectionKey {
    pub(super) account_id: i64,
    pub(super) contract_id: i64,
}

#[derive(Debug, Clone)]
pub(super) struct ManagedProtectionOrders {
    pub(super) signed_qty: i32,
    pub(super) take_profit_price: Option<f64>,
    pub(super) stop_price: Option<f64>,
    pub(super) last_requested_take_profit_price: Option<f64>,
    pub(super) last_requested_stop_price: Option<f64>,
    pub(super) take_profit_order_id: Option<String>,
    pub(super) stop_order_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct ProtectionOrderCandidate {
    pub(super) order_id: String,
    pub(super) price: Option<f64>,
    pub(super) quantity: Option<i32>,
}

pub(super) enum OrderDispatchOutcome {
    NoOp { message: String },
    Queued { target_qty: Option<i32> },
}

#[derive(Clone)]
pub(super) struct AccountRefresh {
    pub(super) balances: Vec<Value>,
    pub(super) positions: Vec<Value>,
    pub(super) risks: Vec<Value>,
    pub(super) orders: BTreeMap<String, Vec<Value>>,
    pub(super) fills: BTreeMap<String, Vec<Value>>,
}

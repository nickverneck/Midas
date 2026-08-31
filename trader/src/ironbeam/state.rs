use crate::broker::{
    AccountInfo, AccountSnapshot, ContractSuggestion, LatencySnapshot, MarketSnapshot,
};
use crate::config::AppConfig;
use crate::strategies::adx::AdxExecutionState;
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::hma_cross::HmaCrossExecutionState;
use crate::strategy::{ExecutionRuntimeSnapshot, ExecutionStrategyConfig};
use reqwest::Client;
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

const INTERNAL_EVENT_QUEUE_CAPACITY: usize = 256;
const INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY: usize = 128;
const MAX_INTERNAL_EVENT_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

pub(super) enum InternalEvent {
    StreamPayload(String),
    StreamStatus(String),
    StreamError(String),
    RefreshAccountState { reason: Option<String> },
}

#[derive(Clone, Debug)]
pub(super) struct InternalEventSender {
    normal: tokio::sync::mpsc::Sender<InternalEvent>,
    critical: tokio::sync::mpsc::Sender<InternalEvent>,
    overflow: Arc<InternalEventOverflow>,
}

#[derive(Debug, Default)]
struct InternalEventOverflow {
    count: AtomicU64,
    notify: Notify,
}

impl InternalEventSender {
    pub(super) fn send(
        &self,
        event: InternalEvent,
    ) -> Result<(), tokio::sync::mpsc::error::TrySendError<InternalEvent>> {
        let payload_bytes = match &event {
            InternalEvent::StreamPayload(payload)
            | InternalEvent::StreamStatus(payload)
            | InternalEvent::StreamError(payload) => payload.len(),
            InternalEvent::RefreshAccountState { .. } => 0,
        };
        if payload_bytes > MAX_INTERNAL_EVENT_PAYLOAD_BYTES {
            self.overflow.count.fetch_add(1, Ordering::Relaxed);
            self.overflow.notify.notify_one();
            return Err(tokio::sync::mpsc::error::TrySendError::Full(event));
        }
        let queue = if is_critical_event(&event) {
            &self.critical
        } else {
            &self.normal
        };
        match queue.try_send(event) {
            Ok(()) => Ok(()),
            Err(error @ tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.overflow.count.fetch_add(1, Ordering::Relaxed);
                self.overflow.notify.notify_one();
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    pub(super) async fn overflow_notified(&self) {
        self.overflow.notify.notified().await;
    }

    pub(super) fn take_overflow_count(&self) -> u64 {
        self.overflow.count.swap(0, Ordering::AcqRel)
    }
}

pub(super) struct InternalEventReceiver {
    critical: tokio::sync::mpsc::Receiver<InternalEvent>,
    normal: tokio::sync::mpsc::Receiver<InternalEvent>,
}

impl InternalEventReceiver {
    pub(super) async fn recv(&mut self) -> Option<InternalEvent> {
        tokio::select! {
            biased;
            event = self.critical.recv() => event,
            event = self.normal.recv() => event,
        }
    }
}

pub(super) fn internal_event_channel() -> (InternalEventSender, InternalEventReceiver) {
    let (critical, critical_receiver) =
        tokio::sync::mpsc::channel(INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY);
    let (normal, normal_receiver) = tokio::sync::mpsc::channel(INTERNAL_EVENT_QUEUE_CAPACITY);
    (
        InternalEventSender {
            normal,
            critical,
            overflow: Arc::new(InternalEventOverflow::default()),
        },
        InternalEventReceiver {
            critical: critical_receiver,
            normal: normal_receiver,
        },
    )
}

fn is_critical_event(event: &InternalEvent) -> bool {
    matches!(
        event,
        InternalEvent::StreamError(_) | InternalEvent::RefreshAccountState { .. }
    )
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
    pub(super) hma_cross_execution: HmaCrossExecutionState,
    pub(super) volume_hma_cross_execution: HmaCrossExecutionState,
    pub(super) volume_ema_cross_execution: EmaCrossExecutionState,
    pub(super) adx_execution: AdxExecutionState,
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
        self.hma_cross_execution = HmaCrossExecutionState::default();
        self.volume_hma_cross_execution = HmaCrossExecutionState::default();
        self.volume_ema_cross_execution = EmaCrossExecutionState::default();
        self.adx_execution = AdxExecutionState::default();
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

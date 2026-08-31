use super::*;
#[cfg(feature = "replay")]
use crate::broker::{
    ReplayBarProtectionPolicy, ReplayEngineMode, ReplayFillModel, ReplayLatencyConfig,
};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::sync::Notify;

/// Events produced by the broker/websocket workers are bounded at the service
/// boundary.  `send` is intentionally non-blocking: a full queue is a
/// service-health failure that the caller must propagate, never an invitation
/// to retain another event in an unbounded buffer.
#[derive(Clone, Debug)]
pub(crate) struct InternalEventSender {
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
    pub(crate) fn send(
        &self,
        event: InternalEvent,
    ) -> Result<(), tokio::sync::mpsc::error::TrySendError<InternalEvent>> {
        // Do this check before handing ownership to Tokio's queue.  A single
        // malformed/oversized broker batch must not consume a bounded slot or
        // become a retained allocation while the service is already unhealthy.
        if internal_event_payload_bytes(&event) > MAX_INTERNAL_EVENT_PAYLOAD_BYTES {
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
            Err(error @ tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                // A closed lane is just as fatal as a full lane for service
                // events: the supervisor must observe it and terminate the
                // affected session instead of leaving callers waiting for a
                // response that can no longer be delivered.
                self.overflow.count.fetch_add(1, Ordering::Relaxed);
                self.overflow.notify.notify_one();
                Err(error)
            }
        }
    }

    pub(crate) async fn overflow_notified(&self) {
        self.overflow.notify.notified().await;
    }

    pub(crate) fn take_overflow_count(&self) -> u64 {
        self.overflow.count.swap(0, Ordering::AcqRel)
    }
}

pub(crate) struct InternalEventReceiver {
    critical: tokio::sync::mpsc::Receiver<InternalEvent>,
    normal: tokio::sync::mpsc::Receiver<InternalEvent>,
}

impl InternalEventReceiver {
    pub(crate) async fn recv(&mut self) -> Option<InternalEvent> {
        tokio::select! {
            biased;
            event = self.critical.recv() => event,
            event = self.normal.recv() => event,
        }
    }
}

pub(crate) fn internal_event_channel(
    capacity: usize,
) -> (InternalEventSender, InternalEventReceiver) {
    let (critical, critical_receiver) =
        tokio::sync::mpsc::channel(INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY);
    let (normal, normal_receiver) = tokio::sync::mpsc::channel(capacity);
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

pub(crate) const INTERNAL_EVENT_QUEUE_CAPACITY: usize = 512;
const INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY: usize = 128;
pub(crate) const MAX_INTERNAL_EVENT_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_INTERNAL_ENTITY_PAYLOAD_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_INTERNAL_ENTITY_COUNT: usize = 4096;

/// The event channel itself owns the bound. No process-global permits are used
/// because permits can outlive an aborted service and poison a reconnect.
pub(crate) fn initialize_internal_backpressure() {}

pub(crate) async fn send_market_internal_event(
    tx: &InternalEventSender,
    event: InternalEvent,
) -> bool {
    tx.send(event).is_ok()
}

pub(crate) async fn send_user_entities(
    tx: &InternalEventSender,
    entities: Vec<EntityEnvelope>,
) -> bool {
    tx.send(InternalEvent::UserEntities(entities)).is_ok()
}

pub(crate) async fn send_snapshots_built(
    tx: &InternalEventSender,
    generation: u64,
    revision: u64,
    snapshots: Vec<AccountSnapshot>,
) -> Result<(), SnapshotDeliveryFailure> {
    match tx.send(InternalEvent::SnapshotsBuilt {
        generation,
        revision,
        snapshots,
    }) {
        Ok(()) => Ok(()),
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            Err(SnapshotDeliveryFailure::QueueFull)
        }
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
            Err(SnapshotDeliveryFailure::QueueClosed)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SnapshotDeliveryFailure {
    QueueFull,
    QueueClosed,
}

impl SnapshotDeliveryFailure {
    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::QueueFull => "internal event queue is full",
            Self::QueueClosed => "internal event queue is closed",
        }
    }
}

pub(crate) enum InternalEvent {
    UserEntities(Vec<EntityEnvelope>),
    SnapshotsBuilt {
        generation: u64,
        revision: u64,
        snapshots: Vec<AccountSnapshot>,
    },
    SnapshotsBuildFailed {
        generation: u64,
        revision: u64,
        failure: SnapshotDeliveryFailure,
    },
    RestLatencyMeasured(u64),
    UserSocketStatus(String),
    Market(MarketUpdate),
    #[cfg(feature = "replay")]
    ReplayMarket {
        update: MarketUpdate,
        response_tx: oneshot::Sender<Result<(), String>>,
    },
    #[cfg(feature = "replay")]
    ReplayBarrier(oneshot::Sender<()>),
    #[cfg(feature = "replay")]
    ReplayCompleted {
        error: Option<String>,
    },
    BrokerOrderAck(BrokerOrderAck),
    BrokerOrderFailed(BrokerOrderFailure),
    OrderStrategyAck(BrokerOrderStrategyAck),
    OrderStrategyFailed(BrokerOrderStrategyFailure),
    PendingTargetWatchdog,
    ProtectionSyncApplied(ProtectionSyncAck),
    ProtectionSyncFailed(ProtectionSyncFailure),
    Error(String),
}

fn is_critical_event(event: &InternalEvent) -> bool {
    match event {
        InternalEvent::UserEntities(entities)
            if entities
                .iter()
                .any(|entity| is_execution_critical_entity(&entity.entity_type)) =>
        {
            true
        }
        InternalEvent::SnapshotsBuilt { .. }
        | InternalEvent::SnapshotsBuildFailed { .. }
        | InternalEvent::BrokerOrderAck(_)
        | InternalEvent::BrokerOrderFailed(_)
        | InternalEvent::OrderStrategyAck(_)
        | InternalEvent::OrderStrategyFailed(_)
        | InternalEvent::ProtectionSyncApplied(_)
        | InternalEvent::ProtectionSyncFailed(_)
        | InternalEvent::Error(_) => true,
        #[cfg(feature = "replay")]
        InternalEvent::ReplayCompleted { .. } | InternalEvent::ReplayBarrier(_) => true,
        _ => false,
    }
}

fn is_execution_critical_entity(entity_type: &str) -> bool {
    matches!(
        entity_type.to_ascii_lowercase().as_str(),
        "fill"
            | "fillfee"
            | "commandreport"
            | "order"
            | "orderstrategy"
            | "orderstrategylink"
            | "position"
            | "account"
            | "accountriskstatus"
            | "cashbalance"
            | "executionreport"
    )
}

fn internal_event_payload_bytes(event: &InternalEvent) -> usize {
    match event {
        InternalEvent::UserEntities(entities) => {
            if entities.len() > MAX_INTERNAL_ENTITY_COUNT {
                return MAX_INTERNAL_EVENT_PAYLOAD_BYTES.saturating_add(1);
            }
            entities.iter().fold(0usize, |size, entity| {
                let entity_bytes = serde_json::to_vec(&entity.entity)
                    .map(|payload| payload.len())
                    .unwrap_or(MAX_INTERNAL_ENTITY_PAYLOAD_BYTES.saturating_add(1));
                if entity_bytes > MAX_INTERNAL_ENTITY_PAYLOAD_BYTES {
                    return MAX_INTERNAL_EVENT_PAYLOAD_BYTES.saturating_add(1);
                }
                size.saturating_add(entity.entity_type.len())
                    .saturating_add(32)
                    .saturating_add(entity_bytes)
            })
        }
        InternalEvent::SnapshotsBuilt { snapshots, .. } => serde_json::to_vec(snapshots)
            .map(|payload| payload.len())
            .unwrap_or(MAX_INTERNAL_EVENT_PAYLOAD_BYTES),
        InternalEvent::UserSocketStatus(message) | InternalEvent::Error(message) => message.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn closed_delivery_is_reported_without_global_permits() {
        let (tx, rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        drop(rx);
        assert!(!send_market_internal_event(&tx, InternalEvent::Error("closed".into())).await);
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[test]
    fn full_critical_lane_reports_overflow() {
        let (tx, _rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        for _ in 0..INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY {
            tx.send(InternalEvent::Error("full".into())).unwrap();
        }
        assert!(tx.send(InternalEvent::Error("overflow".into())).is_err());
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[tokio::test]
    async fn execution_entities_use_the_critical_lane() {
        let (tx, mut rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        for _ in 0..INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY {
            tx.send(InternalEvent::Error("full".into())).unwrap();
        }
        assert!(
            tx.send(InternalEvent::UserEntities(vec![EntityEnvelope {
                entity_type: "fill".into(),
                deleted: false,
                entity: Value::Null,
            }]))
            .is_err()
        );
        assert!(matches!(rx.recv().await, Some(InternalEvent::Error(_))));
    }

    #[test]
    fn oversized_entity_batch_is_rejected_before_queue_admission() {
        let (tx, _rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        let oversized = Value::String("x".repeat(MAX_INTERNAL_EVENT_PAYLOAD_BYTES));
        assert!(
            tx.send(InternalEvent::UserEntities(vec![EntityEnvelope {
                entity_type: "fill".into(),
                deleted: false,
                entity: oversized,
            }]))
            .is_err()
        );
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[test]
    fn execution_reports_use_the_critical_lane() {
        assert!(is_execution_critical_entity("executionReport"));
        let (tx, _rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        for _ in 0..INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY {
            tx.send(InternalEvent::Error("full".into())).unwrap();
        }
        assert!(
            tx.send(InternalEvent::UserEntities(vec![EntityEnvelope {
                entity_type: "executionReport".into(),
                deleted: false,
                entity: Value::Null,
            }]))
            .is_err()
        );
    }

    #[tokio::test]
    async fn snapshot_delivery_reports_closed_lane() {
        let (tx, rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        drop(rx);

        assert_eq!(
            send_snapshots_built(&tx, 7, 11, Vec::new()).await,
            Err(SnapshotDeliveryFailure::QueueClosed)
        );
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[tokio::test]
    async fn snapshot_delivery_reports_full_critical_lane() {
        let (tx, _rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        for _ in 0..INTERNAL_CRITICAL_EVENT_QUEUE_CAPACITY {
            tx.send(InternalEvent::Error("full".into())).unwrap();
        }

        assert_eq!(
            send_snapshots_built(&tx, 7, 11, Vec::new()).await,
            Err(SnapshotDeliveryFailure::QueueFull)
        );
        assert_eq!(tx.take_overflow_count(), 1);
    }

    #[test]
    fn snapshot_failure_completion_is_critical() {
        assert!(is_critical_event(&InternalEvent::SnapshotsBuildFailed {
            generation: 7,
            revision: 11,
            failure: SnapshotDeliveryFailure::QueueClosed,
        }));
    }

    #[tokio::test]
    async fn lossless_entity_events_retain_fifo_order() {
        let (tx, mut rx) = internal_event_channel(INTERNAL_EVENT_QUEUE_CAPACITY);
        for name in ["first", "second"] {
            assert!(
                send_user_entities(
                    &tx,
                    vec![EntityEnvelope {
                        entity_type: name.to_string(),
                        deleted: false,
                        entity: Value::Null,
                    }],
                )
                .await
            );
        }

        for expected in ["first", "second"] {
            let InternalEvent::UserEntities(entities) = rx.recv().await.unwrap() else {
                panic!("expected user entity event");
            };
            assert_eq!(entities[0].entity_type, expected);
        }
    }
}

pub(crate) struct UserSocketCommand {
    pub(crate) endpoint: String,
    pub(crate) query: Option<String>,
    pub(crate) body: Option<Value>,
    pub(crate) response_tx: oneshot::Sender<Result<Value, String>>,
}

pub(crate) struct OrderLatencyTracker {
    pub(crate) started_at: time::Instant,
    pub(crate) signal_started_at: Option<time::Instant>,
    pub(crate) signal_context: Option<String>,
    pub(crate) cl_ord_id: String,
    pub(crate) order_id: Option<i64>,
    pub(crate) order_strategy_id: Option<i64>,
    pub(crate) seen_recorded: bool,
    pub(crate) exec_report_recorded: bool,
    pub(crate) fill_recorded: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct EntityEnvelope {
    pub(crate) entity_type: String,
    pub(crate) deleted: bool,
    pub(crate) entity: Value,
}

pub(crate) enum BrokerCommand {
    MarketOrder {
        request_tx: UserSocketCommandSender,
        order: PendingMarketOrder,
    },
    #[cfg_attr(not(feature = "manual-orders"), allow(dead_code))]
    LiquidatePosition {
        request_tx: UserSocketCommandSender,
        liquidation: PendingLiquidation,
    },
    OrderStrategy {
        request_tx: UserSocketCommandSender,
        strategy: PendingOrderStrategyTransition,
    },
    LiquidateThenOrderStrategy {
        request_tx: UserSocketCommandSender,
        liquidation: PendingLiquidation,
        strategy: PendingOrderStrategyTransition,
    },
    NativeProtection {
        request_tx: UserSocketCommandSender,
        sync: PendingProtectionSync,
    },
    #[cfg(feature = "replay")]
    ReplayBar {
        bar: Bar,
        ticks: Vec<ReplayMarketTick>,
        dom_updates: Vec<ReplayMarketDom>,
        bar_index: u64,
        response_tx: oneshot::Sender<()>,
    },
    #[cfg(feature = "replay")]
    ConfigureReplay {
        mode: ReplayEngineMode,
        fill_model: ReplayFillModel,
        latency: ReplayLatencyConfig,
        bar_protection_policy: ReplayBarProtectionPolicy,
        response_tx: oneshot::Sender<()>,
    },
    #[cfg(feature = "replay")]
    ReplayDrain {
        market_ts_ns: Option<i64>,
        evaluation_id: Option<u64>,
        response_tx: oneshot::Sender<()>,
    },
}

pub(crate) struct PendingMarketOrder {
    pub(crate) simulate: bool,
    pub(crate) cl_ord_id: String,
    pub(crate) payload: Value,
    pub(crate) account_id: i64,
    pub(crate) contract_id: i64,
    pub(crate) interrupt_order_strategy_id: Option<i64>,
    pub(crate) cancel_order_ids: Vec<i64>,
    pub(crate) action_label: String,
    pub(crate) order_action: String,
    pub(crate) order_qty: i32,
    pub(crate) contract_name: String,
    pub(crate) account_name: String,
    pub(crate) reference_ts_ns: Option<i64>,
    pub(crate) reference_price: Option<f64>,
    pub(crate) simulated_next_qty: i32,
    pub(crate) reason_suffix: Option<String>,
    pub(crate) target_qty: Option<i32>,
}

pub(crate) struct PendingLiquidation {
    pub(crate) simulate: bool,
    pub(crate) request_id: String,
    pub(crate) payload: Value,
    pub(crate) account_id: i64,
    pub(crate) contract_id: i64,
    pub(crate) account_name: String,
    pub(crate) contract_name: String,
    pub(crate) reference_ts_ns: Option<i64>,
    pub(crate) reference_price: Option<f64>,
    pub(crate) target_qty: Option<i32>,
    pub(crate) interrupt_order_strategy_id: Option<i64>,
    pub(crate) cancel_order_ids: Vec<i64>,
}

pub(crate) struct PendingOrderStrategyTransition {
    pub(crate) simulate: bool,
    pub(crate) uuid: String,
    pub(crate) payload: Value,
    pub(crate) interrupt_order_strategy_id: Option<i64>,
    pub(crate) cancel_order_ids: Vec<i64>,
    pub(crate) order_action: String,
    pub(crate) entry_order_qty: i32,
    pub(crate) target_qty: i32,
    pub(crate) contract_name: String,
    pub(crate) account_name: String,
    pub(crate) reference_ts_ns: Option<i64>,
    pub(crate) reference_price: Option<f64>,
    pub(crate) take_profit_price: Option<f64>,
    pub(crate) stop_price: Option<f64>,
    pub(crate) replay_auto_trail: Option<ReplayAutoTrail>,
    pub(crate) reason_suffix: Option<String>,
    pub(crate) key: StrategyProtectionKey,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReplayAutoTrail {
    pub(crate) trigger_offset: f64,
    pub(crate) stop_offset: f64,
    pub(crate) frequency: f64,
}

pub(crate) struct BrokerOrderAck {
    pub(crate) endpoint: &'static str,
    pub(crate) cl_ord_id: String,
    pub(crate) order_id: Option<i64>,
    pub(crate) submit_rtt_ms: u64,
    pub(crate) message: String,
}

pub(crate) struct BrokerOrderFailure {
    pub(crate) endpoint: &'static str,
    pub(crate) cl_ord_id: String,
    pub(crate) message: String,
    pub(crate) target_qty: Option<i32>,
    pub(crate) stale_interrupt: bool,
}

pub(crate) struct BrokerOrderStrategyAck {
    pub(crate) endpoint: &'static str,
    pub(crate) uuid: String,
    pub(crate) order_strategy_id: Option<i64>,
    pub(crate) submit_rtt_ms: u64,
    pub(crate) message: String,
    pub(crate) target_qty: i32,
    pub(crate) key: StrategyProtectionKey,
}

pub(crate) struct BrokerOrderStrategyFailure {
    pub(crate) endpoint: &'static str,
    pub(crate) uuid: String,
    pub(crate) message: String,
    pub(crate) target_qty: i32,
    pub(crate) stale_interrupt: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DesiredNativeProtection {
    pub(crate) key: StrategyProtectionKey,
    pub(crate) account_name: String,
    pub(crate) contract_name: String,
    pub(crate) signed_qty: i32,
    pub(crate) take_profit_price: Option<f64>,
    pub(crate) stop_price: Option<f64>,
    pub(crate) reason: String,
}

pub(crate) struct PendingProtectionSync {
    pub(crate) simulate: bool,
    pub(crate) key: StrategyProtectionKey,
    pub(crate) account_name: String,
    pub(crate) contract_name: String,
    pub(crate) operation: ProtectionSyncOperation,
    pub(crate) message: Option<String>,
    pub(crate) next_state: Option<ManagedProtectionOrders>,
}

pub(crate) enum ProtectionSyncOperation {
    Clear { cancel_order_ids: Vec<i64> },
}

pub(crate) struct ProtectionSyncAck {
    pub(crate) endpoint: &'static str,
    pub(crate) key: StrategyProtectionKey,
    pub(crate) message: Option<String>,
    pub(crate) next_state: Option<ManagedProtectionOrders>,
}

pub(crate) struct ProtectionSyncFailure {
    pub(crate) endpoint: &'static str,
    pub(crate) message: String,
}

pub(crate) struct DetachedStrategyProtection {
    pub(crate) cancel_order_ids: Vec<i64>,
}

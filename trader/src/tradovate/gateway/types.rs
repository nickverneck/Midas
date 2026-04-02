use super::*;

pub(crate) enum InternalEvent {
    UserEntities(Vec<EntityEnvelope>),
    SnapshotsBuilt {
        revision: u64,
        snapshots: Vec<AccountSnapshot>,
    },
    RestLatencyMeasured(u64),
    UserSocketStatus(String),
    Market(MarketUpdate),
    BrokerOrderAck(BrokerOrderAck),
    BrokerOrderFailed(BrokerOrderFailure),
    OrderStrategyAck(BrokerOrderStrategyAck),
    OrderStrategyFailed(BrokerOrderStrategyFailure),
    PendingTargetWatchdog,
    ProtectionSyncApplied(ProtectionSyncAck),
    ProtectionSyncFailed(ProtectionSyncFailure),
    Error(String),
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
    pub(crate) strategy_owned_protection: bool,
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
        request_tx: UnboundedSender<UserSocketCommand>,
        order: PendingMarketOrder,
    },
    LiquidatePosition {
        request_tx: UnboundedSender<UserSocketCommand>,
        liquidation: PendingLiquidation,
    },
    OrderStrategy {
        request_tx: UnboundedSender<UserSocketCommand>,
        strategy: PendingOrderStrategyTransition,
    },
    LiquidateThenOrderStrategy {
        request_tx: UnboundedSender<UserSocketCommand>,
        liquidation: PendingLiquidation,
        strategy: PendingOrderStrategyTransition,
    },
    NativeProtection {
        request_tx: UnboundedSender<UserSocketCommand>,
        sync: PendingProtectionSync,
    },
    #[cfg(feature = "replay")]
    ReplayBar {
        bar: Bar,
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
    pub(crate) reason_suffix: Option<String>,
    pub(crate) key: StrategyProtectionKey,
}

pub(crate) struct BrokerOrderAck {
    pub(crate) cl_ord_id: String,
    pub(crate) order_id: Option<i64>,
    pub(crate) submit_rtt_ms: u64,
    pub(crate) message: String,
}

pub(crate) struct BrokerOrderFailure {
    pub(crate) cl_ord_id: String,
    pub(crate) message: String,
    pub(crate) target_qty: Option<i32>,
    pub(crate) stale_interrupt: bool,
}

pub(crate) struct BrokerOrderStrategyAck {
    pub(crate) uuid: String,
    pub(crate) order_strategy_id: Option<i64>,
    pub(crate) submit_rtt_ms: u64,
    pub(crate) message: String,
    pub(crate) target_qty: i32,
    pub(crate) key: StrategyProtectionKey,
}

pub(crate) struct BrokerOrderStrategyFailure {
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
    Clear {
        cancel_order_ids: Vec<i64>,
    },
    ModifyStop {
        payload: Value,
    },
    Replace {
        cancel_order_ids: Vec<i64>,
        request: ProtectionPlaceRequest,
    },
}

pub(crate) enum ProtectionPlaceRequest {
    TakeProfit { payload: Value },
    StopLoss { payload: Value },
    Oco { payload: Value },
}

pub(crate) struct ProtectionSyncAck {
    pub(crate) key: StrategyProtectionKey,
    pub(crate) message: Option<String>,
    pub(crate) next_state: Option<ManagedProtectionOrders>,
}

pub(crate) struct ProtectionSyncFailure {
    pub(crate) message: String,
}

pub(crate) struct DetachedStrategyProtection {
    pub(crate) cancel_order_ids: Vec<i64>,
}

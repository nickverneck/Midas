enum InternalEvent {
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
    ProtectionSyncApplied(ProtectionSyncAck),
    ProtectionSyncFailed(ProtectionSyncFailure),
    Error(String),
}

struct UserSocketCommand {
    endpoint: String,
    query: Option<String>,
    body: Option<Value>,
    response_tx: oneshot::Sender<Result<Value, String>>,
}

struct OrderLatencyTracker {
    started_at: time::Instant,
    cl_ord_id: String,
    order_id: Option<i64>,
    order_strategy_id: Option<i64>,
    seen_recorded: bool,
    exec_report_recorded: bool,
    fill_recorded: bool,
}

#[derive(Debug, Clone)]
struct EntityEnvelope {
    entity_type: String,
    deleted: bool,
    entity: Value,
}

struct LiveSeries {
    closed_bars: Vec<Bar>,
    forming_bar: Option<Bar>,
}

#[derive(Debug, Clone)]
struct MarketUpdate {
    contract_id: i64,
    contract_name: String,
    session_profile: Option<InstrumentSessionProfile>,
    value_per_point: Option<f64>,
    tick_size: Option<f64>,
    history_loaded: usize,
    live_bars: usize,
    status: String,
    bars: MarketBarsUpdate,
}

#[derive(Debug, Clone)]
enum MarketBarsUpdate {
    Snapshot {
        closed_bars: Vec<Bar>,
        forming_bar: Option<Bar>,
    },
    Forming {
        forming_bar: Bar,
    },
    Closed {
        closed_bar: Bar,
        forming_bar: Option<Bar>,
    },
}

enum BrokerCommand {
    MarketOrder {
        request_tx: UnboundedSender<UserSocketCommand>,
        order: PendingMarketOrder,
    },
    OrderStrategy {
        request_tx: UnboundedSender<UserSocketCommand>,
        strategy: PendingOrderStrategyTransition,
    },
    NativeProtection {
        request_tx: UnboundedSender<UserSocketCommand>,
        sync: PendingProtectionSync,
    },
}

struct PendingMarketOrder {
    cl_ord_id: String,
    payload: Value,
    interrupt_order_strategy_id: Option<i64>,
    cancel_order_ids: Vec<i64>,
    action_label: String,
    order_action: String,
    order_qty: i32,
    contract_name: String,
    account_name: String,
    reason_suffix: Option<String>,
    target_qty: Option<i32>,
}

struct PendingOrderStrategyTransition {
    uuid: String,
    payload: Value,
    interrupt_order_strategy_id: Option<i64>,
    order_action: String,
    entry_order_qty: i32,
    target_qty: i32,
    contract_name: String,
    account_name: String,
    reason_suffix: Option<String>,
    key: StrategyProtectionKey,
}

struct BrokerOrderAck {
    cl_ord_id: String,
    order_id: Option<i64>,
    submit_rtt_ms: u64,
    message: String,
}

struct BrokerOrderFailure {
    cl_ord_id: String,
    message: String,
    target_qty: Option<i32>,
}

struct BrokerOrderStrategyAck {
    uuid: String,
    order_strategy_id: Option<i64>,
    submit_rtt_ms: u64,
    message: String,
    target_qty: i32,
    key: StrategyProtectionKey,
}

struct BrokerOrderStrategyFailure {
    uuid: String,
    message: String,
    target_qty: i32,
}

#[derive(Debug, Clone, PartialEq)]
struct DesiredNativeProtection {
    key: StrategyProtectionKey,
    account_name: String,
    contract_name: String,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    reason: String,
}

struct PendingProtectionSync {
    key: StrategyProtectionKey,
    account_name: String,
    contract_name: String,
    operation: ProtectionSyncOperation,
    message: Option<String>,
    next_state: Option<ManagedProtectionOrders>,
}

enum ProtectionSyncOperation {
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

enum ProtectionPlaceRequest {
    TakeProfit { payload: Value },
    StopLoss { payload: Value },
    Oco { payload: Value },
}

struct ProtectionSyncAck {
    key: StrategyProtectionKey,
    message: Option<String>,
    next_state: Option<ManagedProtectionOrders>,
}

struct ProtectionSyncFailure {
    message: String,
}

struct DetachedStrategyProtection {
    cancel_order_ids: Vec<i64>,
}

impl LiveSeries {
    fn new() -> Self {
        Self {
            closed_bars: Vec::new(),
            forming_bar: None,
        }
    }

    fn push_closed_bar(&mut self, bar: &Bar) {
        if let Some(last) = self.closed_bars.last_mut() {
            if bar.ts_ns == last.ts_ns {
                *last = bar.clone();
                return;
            }
            if bar.ts_ns < last.ts_ns {
                return;
            }
        }
        self.closed_bars.push(bar.clone());
    }
}

fn spawn_user_sync_task(
    cfg: AppConfig,
    tokens: TokenBundle,
    account_ids: Vec<i64>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> (UnboundedSender<UserSocketCommand>, JoinHandle<()>) {
    let (request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(user_sync_worker(
        cfg,
        tokens,
        account_ids,
        request_rx,
        internal_tx,
    ));
    (request_tx, task)
}

fn spawn_broker_gateway_task(
    request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(broker_gateway_worker(request_rx, internal_tx))
}

async fn broker_gateway_worker(
    mut request_rx: UnboundedReceiver<BrokerCommand>,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    while let Some(command) = request_rx.recv().await {
        match command {
            BrokerCommand::MarketOrder { request_tx, order } => {
                match submit_market_order_via_gateway(&request_tx, order).await {
                    Ok(ack) => {
                        let _ = internal_tx.send(InternalEvent::BrokerOrderAck(ack));
                    }
                    Err(failure) => {
                        let _ = internal_tx.send(InternalEvent::BrokerOrderFailed(failure));
                    }
                }
            }
            BrokerCommand::OrderStrategy {
                request_tx,
                strategy,
            } => match submit_order_strategy_via_gateway(&request_tx, strategy).await {
                Ok(ack) => {
                    let _ = internal_tx.send(InternalEvent::OrderStrategyAck(ack));
                }
                Err(failure) => {
                    let _ = internal_tx.send(InternalEvent::OrderStrategyFailed(failure));
                }
            },
            BrokerCommand::NativeProtection { request_tx, sync } => {
                match submit_native_protection_via_gateway(&request_tx, sync).await {
                    Ok(ack) => {
                        let _ = internal_tx.send(InternalEvent::ProtectionSyncApplied(ack));
                    }
                    Err(failure) => {
                        let _ = internal_tx.send(InternalEvent::ProtectionSyncFailed(failure));
                    }
                }
            }
        }
    }
}

async fn submit_market_order_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order: PendingMarketOrder,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    if let Some(order_strategy_id) = order.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            return Err(BrokerOrderFailure {
                cl_ord_id: order.cl_ord_id,
                message: format!("failed to interrupt strategy {order_strategy_id}: {err}"),
                target_qty: order.target_qty,
            });
        }
    }

    for order_id in &order.cancel_order_ids {
        if let Err(err) = cancel_order_by_id(request_tx, *order_id).await {
            return Err(BrokerOrderFailure {
                cl_ord_id: order.cl_ord_id,
                message: format!("failed to clear strategy protection: {err}"),
                target_qty: order.target_qty,
            });
        }
    }

    let started_at = time::Instant::now();
    let parsed = match request_order_json(request_tx, "order/placeorder", &order.payload).await {
        Ok(parsed) => parsed,
        Err(err) => {
            return Err(BrokerOrderFailure {
                cl_ord_id: order.cl_ord_id,
                message: err.to_string(),
                target_qty: order.target_qty,
            });
        }
    };
    let submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    let order_id = json_i64(&parsed, "orderId").or_else(|| json_i64(&parsed, "id"));
    let mut message = format!(
        "{} submitted: {} {} {} on {}",
        order.action_label,
        order.order_action,
        order.order_qty,
        order.contract_name,
        order.account_name
    );
    if let Some(reason) = order.reason_suffix.as_deref() {
        message.push_str(&format!(" [{reason}]"));
    }
    if let Some(order_id) = order_id {
        message.push_str(&format!(" (order {order_id})"));
    }
    message.push_str(&format!(" [clOrdId {}]", order.cl_ord_id));
    Ok(BrokerOrderAck {
        cl_ord_id: order.cl_ord_id,
        order_id,
        submit_rtt_ms,
        message,
    })
}

async fn submit_order_strategy_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    if let Some(order_strategy_id) = strategy.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            return Err(BrokerOrderStrategyFailure {
                uuid: strategy.uuid,
                message: format!("failed to interrupt strategy {order_strategy_id}: {err}"),
                target_qty: strategy.target_qty,
            });
        }
    }

    let started_at = time::Instant::now();
    let parsed = match request_order_json(request_tx, "orderStrategy/startorderstrategy", &strategy.payload).await {
        Ok(parsed) => parsed,
        Err(err) => {
            return Err(BrokerOrderStrategyFailure {
                uuid: strategy.uuid,
                message: err.to_string(),
                target_qty: strategy.target_qty,
            });
        }
    };
    let submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    let strategy_entity = parsed.get("orderStrategy").unwrap_or(&parsed);
    let order_strategy_id = json_i64(strategy_entity, "id");
    let strategy_uuid = strategy_entity
        .get("uuid")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| Some(strategy.uuid.clone()))
        .unwrap_or_else(|| strategy.uuid.clone());
    let mut message = format!(
        "Strategy submitted: {} {} {} on {}",
        strategy.order_action,
        strategy.entry_order_qty,
        strategy.contract_name,
        strategy.account_name
    );
    if let Some(reason) = strategy.reason_suffix.as_deref() {
        message.push_str(&format!(" [{reason}]"));
    }
    if let Some(order_strategy_id) = order_strategy_id {
        message.push_str(&format!(" (strategy {order_strategy_id})"));
    }
    message.push_str(&format!(" [uuid {}]", strategy_uuid));
    Ok(BrokerOrderStrategyAck {
        uuid: strategy_uuid,
        order_strategy_id,
        submit_rtt_ms,
        message,
        target_qty: strategy.target_qty,
        key: strategy.key,
    })
}

async fn submit_native_protection_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    sync: PendingProtectionSync,
) -> Result<ProtectionSyncAck, ProtectionSyncFailure> {
    let mut next_state = sync.next_state;
    let failure_message = |err: anyhow::Error| ProtectionSyncFailure {
        message: format!(
            "native protection sync failed for {} on {}: {err}",
            sync.contract_name, sync.account_name
        ),
    };
    let outcome = match sync.operation {
        ProtectionSyncOperation::Clear { cancel_order_ids } => {
            let cleared = cancel_orders_by_id(request_tx, &cancel_order_ids).await;
            cleared.map(|cleared| ProtectionSyncAck {
                key: sync.key,
                message: if cleared { sync.message } else { None },
                next_state,
            })
        }
        ProtectionSyncOperation::ModifyStop { payload } => {
            request_order_json(request_tx, "order/modifyorder", &payload)
                .await
                .map(|_| ProtectionSyncAck {
                    key: sync.key,
                    message: sync.message,
                    next_state,
                })
        }
        ProtectionSyncOperation::Replace {
            cancel_order_ids,
            request,
        } => {
            let _ = match cancel_orders_by_id(request_tx, &cancel_order_ids).await {
                Ok(cancelled) => cancelled,
                Err(err) => return Err(failure_message(err)),
            };
            let (endpoint, payload, place_kind) = match &request {
                ProtectionPlaceRequest::TakeProfit { payload } => {
                    ("order/placeorder", payload, "tp")
                }
                ProtectionPlaceRequest::StopLoss { payload } => ("order/placeorder", payload, "sl"),
                ProtectionPlaceRequest::Oco { payload } => ("order/placeOCO", payload, "oco"),
            };
            let parsed = match request_order_json(request_tx, endpoint, payload).await {
                Ok(parsed) => parsed,
                Err(err) => return Err(failure_message(err)),
            };

            if let Some(state) = next_state.as_mut() {
                match place_kind {
                    "tp" => {
                        state.take_profit_order_id = first_known_order_id(&parsed);
                    }
                    "sl" => {
                        state.stop_order_id = first_known_order_id(&parsed);
                    }
                    _ => {
                        state.take_profit_order_id = first_known_order_id(&parsed);
                        state.stop_order_id = known_order_id(&parsed, &["otherId", "stopOrderId"]);
                    }
                }
            }

            Ok(ProtectionSyncAck {
                key: sync.key,
                message: sync.message,
                next_state,
            })
        }
    };

    outcome.map_err(failure_message)
}

fn spawn_rest_probe_task(
    client: Client,
    cfg: AppConfig,
    access_token: String,
    internal_tx: UnboundedSender<InternalEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        interval.tick().await;
        loop {
            interval.tick().await;
            if let Ok(rest_rtt_ms) = measure_rest_rtt_ms(&client, &cfg.env, &access_token).await {
                let _ = internal_tx.send(InternalEvent::RestLatencyMeasured(rest_rtt_ms));
            }
        }
    })
}

fn request_snapshot_refresh(
    state: &mut ServiceState,
    internal_tx: &UnboundedSender<InternalEvent>,
) {
    let Some(session) = state.session.as_ref() else {
        return;
    };
    state.snapshot_revision = state.snapshot_revision.saturating_add(1);
    let revision = state.snapshot_revision;
    let accounts = session.accounts.clone();
    let market = session.market.clone();
    let managed_protection = session.managed_protection.clone();
    let user_store = session.user_store.clone();
    let internal_tx = internal_tx.clone();
    tokio::spawn(async move {
        let snapshots = user_store.build_snapshots(&accounts, Some(&market), &managed_protection);
        let _ = internal_tx.send(InternalEvent::SnapshotsBuilt {
            revision,
            snapshots,
        });
    });
}

fn market_last_closed_ts(market: &MarketSnapshot) -> Option<i64> {
    let closed_len = market.history_loaded.min(market.bars.len());
    closed_len
        .checked_sub(1)
        .and_then(|idx| market.bars.get(idx))
        .map(|bar| bar.ts_ns)
}

fn build_market_update(
    contract: &ContractSuggestion,
    market_specs: Option<MarketSpecs>,
    history_loaded: usize,
    live_bars: usize,
    status: String,
    before_closed_len: usize,
    before_last_closed: Option<Bar>,
    before_forming: Option<Bar>,
    series: &LiveSeries,
) -> Option<MarketUpdate> {
    let bars = if before_closed_len == 0 && history_loaded > 0 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len + 1 {
        Some(MarketBarsUpdate::Snapshot {
            closed_bars: series.closed_bars.clone(),
            forming_bar: series.forming_bar.clone(),
        })
    } else if history_loaded > before_closed_len {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.closed_bars.last() != before_last_closed.as_ref() {
        series
            .closed_bars
            .last()
            .cloned()
            .map(|closed_bar| MarketBarsUpdate::Closed {
                closed_bar,
                forming_bar: series.forming_bar.clone(),
            })
    } else if series.forming_bar != before_forming {
        series
            .forming_bar
            .clone()
            .map(|forming_bar| MarketBarsUpdate::Forming { forming_bar })
    } else {
        None
    }?;

    Some(MarketUpdate {
        contract_id: contract.id,
        contract_name: contract.name.clone(),
        session_profile: market_specs.and_then(|specs| specs.session_profile),
        value_per_point: market_specs.and_then(|specs| specs.value_per_point),
        tick_size: market_specs.and_then(|specs| specs.tick_size),
        history_loaded,
        live_bars,
        status,
        bars,
    })
}

fn apply_market_update(market: &mut MarketSnapshot, update: MarketUpdate) -> bool {
    let prev_last_closed_ts = market_last_closed_ts(market);
    market.contract_id = Some(update.contract_id);
    market.contract_name = Some(update.contract_name);
    market.session_profile = update.session_profile;
    market.value_per_point = update.value_per_point;
    market.tick_size = update.tick_size;
    market.live_bars = update.live_bars;
    market.status = update.status;

    let closed_bar_advanced = match update.bars {
        MarketBarsUpdate::Snapshot {
            closed_bars,
            forming_bar,
        } => {
            let next_last_closed_ts = closed_bars.last().map(|bar| bar.ts_ns);
            market.history_loaded = update.history_loaded.min(closed_bars.len());
            market.bars = closed_bars;
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            next_last_closed_ts.is_some_and(|ts| prev_last_closed_ts.is_none_or(|prev| ts > prev))
        }
        MarketBarsUpdate::Forming { forming_bar } => {
            let closed_len = update.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            market.history_loaded = closed_len;
            market.bars.push(forming_bar);
            false
        }
        MarketBarsUpdate::Closed {
            closed_bar,
            forming_bar,
        } => {
            let closed_len = market.history_loaded.min(market.bars.len());
            market.bars.truncate(closed_len);
            match market.bars.last_mut() {
                Some(last) if last.ts_ns == closed_bar.ts_ns => {
                    *last = closed_bar.clone();
                }
                Some(last) if closed_bar.ts_ns > last.ts_ns => {
                    market.bars.push(closed_bar.clone());
                }
                None => market.bars.push(closed_bar.clone()),
                _ => {}
            }
            market.history_loaded = update.history_loaded.min(market.bars.len());
            market.bars.truncate(market.history_loaded);
            if let Some(forming_bar) = forming_bar {
                market.bars.push(forming_bar);
            }
            prev_last_closed_ts.is_none_or(|prev| closed_bar.ts_ns > prev)
        }
    };

    closed_bar_advanced
}

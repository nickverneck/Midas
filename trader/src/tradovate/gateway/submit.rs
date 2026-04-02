use super::*;

pub(crate) async fn submit_market_order_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    order: PendingMarketOrder,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    if let Some(order_strategy_id) = order.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderFailure {
                cl_ord_id: order.cl_ord_id,
                message: if stale_interrupt {
                    format!(
                        "strategy {order_strategy_id} was already inactive; waiting for broker sync before retrying the reversal"
                    )
                } else {
                    format!("failed to interrupt strategy {order_strategy_id}: {err}")
                },
                target_qty: order.target_qty,
                stale_interrupt,
            });
        }
    }

    for order_id in &order.cancel_order_ids {
        if let Err(err) = cancel_order_by_id(request_tx, *order_id).await {
            return Err(BrokerOrderFailure {
                cl_ord_id: order.cl_ord_id,
                message: format!("failed to clear strategy protection: {err}"),
                target_qty: order.target_qty,
                stale_interrupt: false,
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
                stale_interrupt: false,
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

pub(crate) async fn submit_liquidation_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    liquidation: PendingLiquidation,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    if let Some(order_strategy_id) = liquidation.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderFailure {
                cl_ord_id: liquidation.request_id,
                message: if stale_interrupt {
                    format!("strategy {order_strategy_id} was already inactive before close-all")
                } else {
                    format!("failed to interrupt strategy {order_strategy_id}: {err}")
                },
                target_qty: liquidation.target_qty,
                stale_interrupt,
            });
        }
    }

    for order_id in &liquidation.cancel_order_ids {
        if let Err(err) = cancel_order_by_id(request_tx, *order_id).await {
            return Err(BrokerOrderFailure {
                cl_ord_id: liquidation.request_id.clone(),
                message: format!("failed to clear strategy protection before close-all: {err}"),
                target_qty: liquidation.target_qty,
                stale_interrupt: false,
            });
        }
    }

    let started_at = time::Instant::now();
    let parsed =
        match request_order_json(request_tx, "order/liquidateposition", &liquidation.payload).await
        {
            Ok(parsed) => parsed,
            Err(err) => {
                return Err(BrokerOrderFailure {
                    cl_ord_id: liquidation.request_id,
                    message: err.to_string(),
                    target_qty: liquidation.target_qty,
                    stale_interrupt: false,
                });
            }
        };
    let submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    let order_id = json_i64(&parsed, "orderId").or_else(|| json_i64(&parsed, "id"));
    let mut message = format!(
        "Close submitted: liquidatePosition {} on {}",
        liquidation.contract_name, liquidation.account_name
    );
    if let Some(order_id) = order_id {
        message.push_str(&format!(" (order {order_id})"));
    }
    Ok(BrokerOrderAck {
        cl_ord_id: liquidation.request_id,
        order_id,
        submit_rtt_ms,
        message,
    })
}

pub(crate) async fn submit_liquidation_then_order_strategy_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    liquidation: PendingLiquidation,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    let strategy_uuid = strategy.uuid.clone();
    let target_qty = strategy.target_qty;
    let liquidation_ack = submit_liquidation_via_gateway(request_tx, liquidation)
        .await
        .map_err(|failure| BrokerOrderStrategyFailure {
            uuid: strategy_uuid,
            message: format!(
                "failed to submit close-all before immediate reversal: {}",
                failure.message
            ),
            target_qty,
            stale_interrupt: failure.stale_interrupt,
        })?;
    let mut strategy_ack = submit_order_strategy_via_gateway(request_tx, strategy).await?;
    strategy_ack.message = format!("{}; {}", liquidation_ack.message, strategy_ack.message);
    Ok(strategy_ack)
}

pub(crate) async fn submit_order_strategy_via_gateway(
    request_tx: &UnboundedSender<UserSocketCommand>,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    if let Some(order_strategy_id) = strategy.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderStrategyFailure {
                uuid: strategy.uuid,
                message: if stale_interrupt {
                    format!(
                        "strategy {order_strategy_id} was already inactive; waiting for broker sync before retrying the reversal"
                    )
                } else {
                    format!("failed to interrupt strategy {order_strategy_id}: {err}")
                },
                target_qty: strategy.target_qty,
                stale_interrupt,
            });
        }
    }

    for order_id in &strategy.cancel_order_ids {
        if let Err(err) = cancel_order_by_id(request_tx, *order_id).await {
            return Err(BrokerOrderStrategyFailure {
                uuid: strategy.uuid,
                message: format!("failed to clear strategy protection: {err}"),
                target_qty: strategy.target_qty,
                stale_interrupt: false,
            });
        }
    }

    let started_at = time::Instant::now();
    let parsed = match request_order_json(
        request_tx,
        "orderStrategy/startorderstrategy",
        &strategy.payload,
    )
    .await
    {
        Ok(parsed) => parsed,
        Err(err) => {
            return Err(BrokerOrderStrategyFailure {
                uuid: strategy.uuid,
                message: err.to_string(),
                target_qty: strategy.target_qty,
                stale_interrupt: false,
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

fn interrupt_error_is_stale(err: &anyhow::Error) -> bool {
    err.to_string()
        .to_ascii_lowercase()
        .contains("no active order strategy")
}

pub(crate) async fn submit_native_protection_via_gateway(
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

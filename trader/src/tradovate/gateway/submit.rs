use super::*;

pub(crate) async fn submit_market_order_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    order: PendingMarketOrder,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    if let Some(order_strategy_id) = order.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderFailure {
                endpoint: "orderStrategy/interruptorderstrategy",
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
                endpoint: "order/cancelorder",
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
                endpoint: "order/placeorder",
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
        endpoint: "order/placeorder",
        cl_ord_id: order.cl_ord_id,
        order_id,
        submit_rtt_ms,
        message,
    })
}

pub(crate) async fn submit_liquidation_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    liquidation: PendingLiquidation,
) -> Result<BrokerOrderAck, BrokerOrderFailure> {
    if let Some(order_strategy_id) = liquidation.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderFailure {
                endpoint: "orderStrategy/interruptorderstrategy",
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
                endpoint: "order/cancelorder",
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
                    endpoint: "order/liquidateposition",
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
        endpoint: "order/liquidateposition",
        cl_ord_id: liquidation.request_id,
        order_id,
        submit_rtt_ms,
        message,
    })
}

pub(crate) async fn submit_liquidation_then_order_strategy_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    liquidation: PendingLiquidation,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    let started_at = time::Instant::now();
    let strategy_uuid = strategy.uuid.clone();
    let target_qty = strategy.target_qty;
    let liquidation_ack = submit_liquidation_via_gateway(request_tx, liquidation)
        .await
        .map_err(|failure| BrokerOrderStrategyFailure {
            endpoint: failure.endpoint,
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
    strategy_ack.submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    Ok(strategy_ack)
}

pub(crate) async fn submit_flatten_then_order_strategy_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    flatten: PendingMarketOrder,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    let started_at = time::Instant::now();
    let strategy_uuid = strategy.uuid.clone();
    let target_qty = strategy.target_qty;
    let flatten_ack = submit_market_order_via_gateway(request_tx, flatten)
        .await
        .map_err(|failure| BrokerOrderStrategyFailure {
            endpoint: failure.endpoint,
            uuid: strategy_uuid,
            message: format!(
                "failed to submit market flatten before immediate reverse entry: {}",
                failure.message
            ),
            target_qty,
            stale_interrupt: failure.stale_interrupt,
        })?;
    let mut strategy_ack = match submit_order_strategy_via_gateway(request_tx, strategy).await {
        Ok(ack) => ack,
        Err(mut failure) => {
            failure.message = format!(
                "market flatten accepted ({}); reverse entry failed without retry: {}",
                flatten_ack.message, failure.message
            );
            return Err(failure);
        }
    };
    strategy_ack.message = format!("{}; {}", flatten_ack.message, strategy_ack.message);
    strategy_ack.submit_rtt_ms = started_at.elapsed().as_millis() as u64;
    Ok(strategy_ack)
}

pub(crate) async fn submit_order_strategy_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    strategy: PendingOrderStrategyTransition,
) -> Result<BrokerOrderStrategyAck, BrokerOrderStrategyFailure> {
    if let Some(order_strategy_id) = strategy.interrupt_order_strategy_id {
        if let Err(err) = interrupt_order_strategy_by_id(request_tx, order_strategy_id).await {
            let stale_interrupt = interrupt_error_is_stale(&err);
            return Err(BrokerOrderStrategyFailure {
                endpoint: "orderStrategy/interruptorderstrategy",
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
                endpoint: "order/cancelorder",
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
                endpoint: "orderStrategy/startorderstrategy",
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
        endpoint: "orderStrategy/startorderstrategy",
        uuid: strategy_uuid,
        order_strategy_id,
        submit_rtt_ms,
        message,
        target_qty: strategy.target_qty,
        key: strategy.key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::unbounded_channel;

    #[tokio::test]
    async fn no_confirm_reversal_posts_flatten_then_entry_without_fill_wait() {
        let (request_tx, mut request_rx) = unbounded_channel::<UserSocketCommand>();
        let response_task = tokio::spawn(async move {
            for (expected_endpoint, response) in [
                (
                    "orderStrategy/interruptorderstrategy",
                    json!({"success": true}),
                ),
                ("order/placeorder", json!({"orderId": 101})),
                (
                    "orderStrategy/startorderstrategy",
                    json!({"orderStrategy": {"id": 202, "uuid": "entry-uuid"}}),
                ),
            ] {
                let command = request_rx
                    .recv()
                    .await
                    .expect("expected the next broker request");
                assert_eq!(command.endpoint, expected_endpoint);
                command
                    .response_tx
                    .send(Ok(response))
                    .expect("gateway request should still be awaiting its response");
            }
            assert!(
                request_rx.try_recv().is_err(),
                "no fill or flat-position request is sent"
            );
        });

        let flatten = PendingMarketOrder {
            simulate: false,
            cl_ord_id: "flatten-uuid".to_string(),
            payload: json!({"accountId": 42, "action": "Sell", "orderQty": 1}),
            account_id: 42,
            contract_id: 3570918,
            interrupt_order_strategy_id: Some(77),
            cancel_order_ids: Vec::new(),
            action_label: "Strategy".to_string(),
            order_action: "Sell".to_string(),
            order_qty: 1,
            contract_name: "ESM6".to_string(),
            account_name: "SIM".to_string(),
            reference_ts_ns: None,
            reference_price: None,
            simulated_next_qty: 0,
            reason_suffix: Some("no-confirm flatten".to_string()),
            target_qty: Some(0),
        };
        let strategy = PendingOrderStrategyTransition {
            simulate: false,
            uuid: "entry-uuid".to_string(),
            payload: json!({"uuid": "entry-uuid", "action": "Sell"}),
            interrupt_order_strategy_id: None,
            cancel_order_ids: Vec::new(),
            order_action: "Sell".to_string(),
            entry_order_qty: 1,
            target_qty: -1,
            contract_name: "ESM6".to_string(),
            account_name: "SIM".to_string(),
            reference_ts_ns: None,
            reference_price: None,
            take_profit_price: None,
            stop_price: None,
            replay_auto_trail: None,
            reason_suffix: Some("reverse entry".to_string()),
            key: StrategyProtectionKey {
                account_id: 42,
                contract_id: 3570918,
            },
        };

        let ack =
            match submit_flatten_then_order_strategy_via_gateway(&request_tx, flatten, strategy)
                .await
            {
                Ok(ack) => ack,
                Err(failure) => panic!(
                    "flatten and entry should both be accepted: {}",
                    failure.message
                ),
            };
        response_task
            .await
            .expect("request recorder should complete");

        assert_eq!(ack.endpoint, "orderStrategy/startorderstrategy");
        assert_eq!(ack.uuid, "entry-uuid");
        assert!(ack.message.contains("order 101"));
        assert!(ack.message.contains("strategy 202"));
    }
}

fn interrupt_error_is_stale(err: &anyhow::Error) -> bool {
    err.to_string()
        .to_ascii_lowercase()
        .contains("no active order strategy")
}

pub(crate) async fn submit_native_protection_via_gateway(
    request_tx: &dyn UserSocketCommandSink,
    sync: PendingProtectionSync,
) -> Result<ProtectionSyncAck, ProtectionSyncFailure> {
    let next_state = sync.next_state;
    let failure_message = |endpoint: &'static str, err: anyhow::Error| ProtectionSyncFailure {
        endpoint,
        message: format!(
            "native protection sync failed for {} on {}: {err}",
            sync.contract_name, sync.account_name
        ),
    };
    let outcome = match sync.operation {
        ProtectionSyncOperation::Clear { cancel_order_ids } => {
            let cleared = match cancel_orders_by_id(request_tx, &cancel_order_ids).await {
                Ok(cleared) => cleared,
                Err(err) => return Err(failure_message("order/cancelorder", err)),
            };
            Ok(ProtectionSyncAck {
                endpoint: "order/cancelorder",
                key: sync.key,
                message: if cleared { sync.message } else { None },
                next_state,
            })
        }
    };

    outcome
}

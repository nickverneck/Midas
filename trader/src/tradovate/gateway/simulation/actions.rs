use super::{
    helpers::{exit_action_for_target, synthetic_ts_ns, working_order_entity},
    state::{ReplayBrokerState, SimActiveOrder, SimOrderStrategyState},
    *,
};

impl ReplayBrokerState {
    pub(crate) fn simulate_market_order(
        &mut self,
        order: PendingMarketOrder,
    ) -> std::result::Result<Vec<InternalEvent>, BrokerOrderFailure> {
        let fill_price = order.reference_price.ok_or_else(|| BrokerOrderFailure {
            cl_ord_id: order.cl_ord_id.clone(),
            message: format!(
                "replay order rejected: no market price is available for {}",
                order.contract_name
            ),
            target_qty: order.target_qty,
            stale_interrupt: false,
        })?;
        let ts_ns = synthetic_ts_ns(order.reference_ts_ns);
        let key = StrategyProtectionKey {
            account_id: order.account_id,
            contract_id: order.contract_id,
        };

        let mut envelopes = Vec::new();
        if let Some(strategy_id) = order.interrupt_order_strategy_id {
            envelopes.extend(self.clear_order_strategy(strategy_id));
        }
        envelopes.extend(self.cancel_orders(&order.cancel_order_ids));

        let order_id = self.next_order_id();
        let order_entity = json!({
            "id": order_id,
            "orderId": order_id,
            "accountId": order.account_id,
            "contractId": order.contract_id,
            "symbol": order.contract_name,
            "action": order.order_action,
            "orderQty": order.order_qty,
            "orderType": "Market",
            "price": fill_price,
            "ordStatus": "Filled",
            "clOrdId": order.cl_ord_id,
        });
        let exec_entity = json!({
            "id": self.next_exec_report_id(),
            "accountId": order.account_id,
            "contractId": order.contract_id,
            "orderId": order_id,
            "clOrdId": order.cl_ord_id,
            "status": "Filled",
            "price": fill_price,
            "timestamp": ts_ns,
        });
        let fill_entity = json!({
            "id": self.next_fill_id(),
            "accountId": order.account_id,
            "contractId": order.contract_id,
            "orderId": order_id,
            "source": "replay",
            "price": fill_price,
            "qty": order.order_qty,
            "buySell": order.order_action,
            "timestamp": ts_ns,
        });
        envelopes.push(EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: order_entity,
        });
        envelopes.push(EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: exec_entity,
        });
        envelopes.push(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: fill_entity,
        });
        envelopes.extend(self.update_position(
            key,
            &order.contract_name,
            order.simulated_next_qty,
            fill_price,
        ));

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
        message.push_str(&format!(
            " (order {order_id}) [clOrdId {}]",
            order.cl_ord_id
        ));

        Ok(vec![
            InternalEvent::BrokerOrderAck(BrokerOrderAck {
                cl_ord_id: order.cl_ord_id,
                order_id: Some(order_id),
                submit_rtt_ms: 0,
                message,
            }),
            InternalEvent::UserEntities(envelopes),
        ])
    }

    pub(crate) fn simulate_liquidation(
        &mut self,
        liquidation: PendingLiquidation,
    ) -> std::result::Result<Vec<InternalEvent>, BrokerOrderFailure> {
        let fill_price = liquidation
            .reference_price
            .ok_or_else(|| BrokerOrderFailure {
                cl_ord_id: liquidation.request_id.clone(),
                message: format!(
                    "replay close rejected: no market price is available for {}",
                    liquidation.contract_name
                ),
                target_qty: liquidation.target_qty,
                stale_interrupt: false,
            })?;
        let ts_ns = synthetic_ts_ns(liquidation.reference_ts_ns);
        let key = StrategyProtectionKey {
            account_id: liquidation.account_id,
            contract_id: liquidation.contract_id,
        };
        let current_qty = self
            .positions
            .get(&key)
            .map(|position| position.qty)
            .unwrap_or(0);
        let order_action = if current_qty >= 0 { "Sell" } else { "Buy" };
        let order_qty = current_qty.abs().max(1);

        let mut envelopes = self.clear_orders_for_contract(key);
        let order_id = self.next_order_id();
        envelopes.push(EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: json!({
                "id": order_id,
                "orderId": order_id,
                "accountId": liquidation.account_id,
                "contractId": liquidation.contract_id,
                "symbol": liquidation.contract_name,
                "action": order_action,
                "orderQty": order_qty,
                "orderType": "Market",
                "price": fill_price,
                "ordStatus": "Filled",
                "clOrdId": liquidation.request_id,
            }),
        });
        envelopes.push(EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: json!({
                "id": self.next_exec_report_id(),
                "accountId": liquidation.account_id,
                "contractId": liquidation.contract_id,
                "orderId": order_id,
                "clOrdId": liquidation.request_id,
                "status": "Filled",
                "price": fill_price,
                "timestamp": ts_ns,
            }),
        });
        envelopes.push(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({
                "id": self.next_fill_id(),
                "accountId": liquidation.account_id,
                "contractId": liquidation.contract_id,
                "orderId": order_id,
                "source": "replay",
                "price": fill_price,
                "qty": order_qty,
                "buySell": order_action,
                "timestamp": ts_ns,
            }),
        });
        envelopes.extend(self.update_position(key, &liquidation.contract_name, 0, fill_price));

        Ok(vec![
            InternalEvent::BrokerOrderAck(BrokerOrderAck {
                cl_ord_id: liquidation.request_id.clone(),
                order_id: Some(order_id),
                submit_rtt_ms: 0,
                message: format!(
                    "Close submitted: liquidatePosition {} on {} (order {order_id})",
                    liquidation.contract_name, liquidation.account_name
                ),
            }),
            InternalEvent::UserEntities(envelopes),
        ])
    }

    pub(crate) fn simulate_liquidation_then_order_strategy(
        &mut self,
        liquidation: PendingLiquidation,
        strategy: PendingOrderStrategyTransition,
    ) -> std::result::Result<Vec<InternalEvent>, BrokerOrderStrategyFailure> {
        let mut liquidation_entities = Vec::new();
        let mut liquidation_message = None;
        for event in self.simulate_liquidation(liquidation).map_err(|failure| {
            BrokerOrderStrategyFailure {
                uuid: strategy.uuid.clone(),
                message: failure.message,
                target_qty: strategy.target_qty,
                stale_interrupt: failure.stale_interrupt,
            }
        })? {
            match event {
                InternalEvent::BrokerOrderAck(ack) => {
                    liquidation_message = Some(ack.message);
                }
                InternalEvent::UserEntities(entities) => liquidation_entities.extend(entities),
                _ => {}
            }
        }

        let mut strategy_entities = Vec::new();
        let mut strategy_ack = None;
        for event in self.simulate_order_strategy(strategy)? {
            match event {
                InternalEvent::OrderStrategyAck(ack) => strategy_ack = Some(ack),
                InternalEvent::UserEntities(entities) => strategy_entities.extend(entities),
                _ => {}
            }
        }

        let mut ack = strategy_ack.expect("strategy replay should emit an ack");
        if let Some(liquidation_message) = liquidation_message {
            ack.message = format!("{liquidation_message}; {}", ack.message);
        }

        let mut events = vec![InternalEvent::OrderStrategyAck(ack)];
        if !liquidation_entities.is_empty() || !strategy_entities.is_empty() {
            liquidation_entities.extend(strategy_entities);
            events.push(InternalEvent::UserEntities(liquidation_entities));
        }
        Ok(events)
    }

    pub(crate) fn simulate_order_strategy(
        &mut self,
        strategy: PendingOrderStrategyTransition,
    ) -> std::result::Result<Vec<InternalEvent>, BrokerOrderStrategyFailure> {
        let fill_price = strategy
            .reference_price
            .ok_or_else(|| BrokerOrderStrategyFailure {
                uuid: strategy.uuid.clone(),
                message: format!(
                    "replay strategy rejected: no market price is available for {}",
                    strategy.contract_name
                ),
                target_qty: strategy.target_qty,
                stale_interrupt: false,
            })?;
        let ts_ns = synthetic_ts_ns(strategy.reference_ts_ns);

        let mut envelopes = Vec::new();
        if let Some(strategy_id) = strategy.interrupt_order_strategy_id {
            envelopes.extend(self.clear_order_strategy(strategy_id));
        }

        let order_strategy_id = self.next_strategy_id();
        let strategy_entity = json!({
            "id": order_strategy_id,
            "accountId": strategy.key.account_id,
            "contractId": strategy.key.contract_id,
            "symbol": strategy.contract_name,
            "uuid": strategy.uuid,
            "status": "Active",
        });
        self.order_strategies.insert(
            order_strategy_id,
            SimOrderStrategyState {
                entity: strategy_entity.clone(),
                order_ids: Vec::new(),
                link_ids: Vec::new(),
            },
        );
        envelopes.push(EntityEnvelope {
            entity_type: "orderStrategy".to_string(),
            deleted: false,
            entity: strategy_entity,
        });

        let entry_order_id = self.next_order_id();
        let entry_order = json!({
            "id": entry_order_id,
            "orderId": entry_order_id,
            "accountId": strategy.key.account_id,
            "contractId": strategy.key.contract_id,
            "symbol": strategy.contract_name,
            "action": strategy.order_action,
            "orderQty": strategy.entry_order_qty,
            "orderType": "Market",
            "price": fill_price,
            "ordStatus": "Filled",
            "clOrdId": strategy.uuid,
            "orderStrategyId": order_strategy_id,
        });
        envelopes.push(EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: entry_order,
        });
        envelopes.push(EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: json!({
                "id": self.next_exec_report_id(),
                "accountId": strategy.key.account_id,
                "contractId": strategy.key.contract_id,
                "orderId": entry_order_id,
                "clOrdId": strategy.uuid,
                "orderStrategyId": order_strategy_id,
                "status": "Filled",
                "price": fill_price,
                "timestamp": ts_ns,
            }),
        });
        envelopes.push(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: json!({
                "id": self.next_fill_id(),
                "accountId": strategy.key.account_id,
                "contractId": strategy.key.contract_id,
                "orderId": entry_order_id,
                "orderStrategyId": order_strategy_id,
                "source": "replay",
                "price": fill_price,
                "qty": strategy.entry_order_qty,
                "buySell": strategy.order_action,
                "timestamp": ts_ns,
            }),
        });

        let mut strategy_state = self
            .order_strategies
            .remove(&order_strategy_id)
            .expect("strategy state just inserted");
        if let Some(tp_price) = strategy.take_profit_price {
            let tp_order_id = self.next_order_id();
            let tp_link_id = self.next_link_id();
            let tp_order = json!({
                "id": tp_order_id,
                "orderId": tp_order_id,
                "accountId": strategy.key.account_id,
                "contractId": strategy.key.contract_id,
                "symbol": strategy.contract_name,
                "action": exit_action_for_target(strategy.target_qty),
                "orderQty": strategy.target_qty.abs().max(1),
                "orderType": "Limit",
                "price": tp_price,
                "ordStatus": "Working",
                "clOrdId": format!("{}-tp", strategy.uuid),
                "orderStrategyId": order_strategy_id,
            });
            self.active_orders.insert(
                tp_order_id,
                SimActiveOrder {
                    order: tp_order.clone(),
                    link_id: Some(tp_link_id),
                    strategy_id: Some(order_strategy_id),
                },
            );
            strategy_state.order_ids.push(tp_order_id);
            strategy_state.link_ids.push(tp_link_id);
            envelopes.push(EntityEnvelope {
                entity_type: "order".to_string(),
                deleted: false,
                entity: tp_order,
            });
            envelopes.push(EntityEnvelope {
                entity_type: "orderStrategyLink".to_string(),
                deleted: false,
                entity: json!({
                    "id": tp_link_id,
                    "orderStrategyId": order_strategy_id,
                    "orderId": tp_order_id
                }),
            });
        }
        if let Some(stop_price) = strategy.stop_price {
            let stop_order_id = self.next_order_id();
            let stop_link_id = self.next_link_id();
            let stop_order = json!({
                "id": stop_order_id,
                "orderId": stop_order_id,
                "accountId": strategy.key.account_id,
                "contractId": strategy.key.contract_id,
                "symbol": strategy.contract_name,
                "action": exit_action_for_target(strategy.target_qty),
                "orderQty": strategy.target_qty.abs().max(1),
                "orderType": "Stop",
                "stopPrice": stop_price,
                "ordStatus": "Working",
                "clOrdId": format!("{}-sl", strategy.uuid),
                "orderStrategyId": order_strategy_id,
            });
            self.active_orders.insert(
                stop_order_id,
                SimActiveOrder {
                    order: stop_order.clone(),
                    link_id: Some(stop_link_id),
                    strategy_id: Some(order_strategy_id),
                },
            );
            strategy_state.order_ids.push(stop_order_id);
            strategy_state.link_ids.push(stop_link_id);
            envelopes.push(EntityEnvelope {
                entity_type: "order".to_string(),
                deleted: false,
                entity: stop_order,
            });
            envelopes.push(EntityEnvelope {
                entity_type: "orderStrategyLink".to_string(),
                deleted: false,
                entity: json!({
                    "id": stop_link_id,
                    "orderStrategyId": order_strategy_id,
                    "orderId": stop_order_id
                }),
            });
        }
        self.order_strategies
            .insert(order_strategy_id, strategy_state);
        envelopes.extend(self.update_position(
            strategy.key,
            &strategy.contract_name,
            strategy.target_qty,
            fill_price,
        ));

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
        message.push_str(&format!(
            " (strategy {order_strategy_id}) [uuid {}]",
            strategy.uuid
        ));

        Ok(vec![
            InternalEvent::OrderStrategyAck(BrokerOrderStrategyAck {
                uuid: strategy.uuid,
                order_strategy_id: Some(order_strategy_id),
                submit_rtt_ms: 0,
                message,
                target_qty: strategy.target_qty,
                key: strategy.key,
            }),
            InternalEvent::UserEntities(envelopes),
        ])
    }

    pub(crate) fn simulate_native_protection(
        &mut self,
        sync: PendingProtectionSync,
    ) -> std::result::Result<Vec<InternalEvent>, ProtectionSyncFailure> {
        let mut next_state = sync.next_state;
        let mut envelopes = Vec::new();
        match sync.operation {
            ProtectionSyncOperation::Clear { cancel_order_ids } => {
                envelopes.extend(self.cancel_orders(&cancel_order_ids));
            }
            ProtectionSyncOperation::ModifyStop { payload } => {
                let Some(order_id) = json_i64(&payload, "orderId") else {
                    return Err(ProtectionSyncFailure {
                        message: format!(
                            "native protection sync failed for {} on {}: modify replay payload missing orderId",
                            sync.contract_name, sync.account_name
                        ),
                    });
                };
                let Some(active) = self.active_orders.get_mut(&order_id) else {
                    return Err(ProtectionSyncFailure {
                        message: format!(
                            "native protection sync failed for {} on {}: replay stop order {order_id} was not found",
                            sync.contract_name, sync.account_name
                        ),
                    });
                };
                if let Some(order) = active.order.as_object_mut() {
                    order.insert(
                        "stopPrice".to_string(),
                        payload.get("stopPrice").cloned().unwrap_or(Value::Null),
                    );
                }
                envelopes.push(EntityEnvelope {
                    entity_type: "order".to_string(),
                    deleted: false,
                    entity: active.order.clone(),
                });
                if let Some(state) = next_state.as_mut() {
                    state.stop_order_id = Some(order_id);
                }
            }
            ProtectionSyncOperation::Replace {
                cancel_order_ids,
                request,
            } => {
                envelopes.extend(self.cancel_orders(&cancel_order_ids));
                match request {
                    ProtectionPlaceRequest::TakeProfit { payload } => {
                        let order_id = self.next_order_id();
                        let entity =
                            working_order_entity(order_id, sync.key.contract_id, &payload, None);
                        self.active_orders.insert(
                            order_id,
                            SimActiveOrder {
                                order: entity.clone(),
                                link_id: None,
                                strategy_id: None,
                            },
                        );
                        if let Some(state) = next_state.as_mut() {
                            state.take_profit_order_id = Some(order_id);
                        }
                        envelopes.push(EntityEnvelope {
                            entity_type: "order".to_string(),
                            deleted: false,
                            entity,
                        });
                    }
                    ProtectionPlaceRequest::StopLoss { payload } => {
                        let order_id = self.next_order_id();
                        let entity =
                            working_order_entity(order_id, sync.key.contract_id, &payload, None);
                        self.active_orders.insert(
                            order_id,
                            SimActiveOrder {
                                order: entity.clone(),
                                link_id: None,
                                strategy_id: None,
                            },
                        );
                        if let Some(state) = next_state.as_mut() {
                            state.stop_order_id = Some(order_id);
                        }
                        envelopes.push(EntityEnvelope {
                            entity_type: "order".to_string(),
                            deleted: false,
                            entity,
                        });
                    }
                    ProtectionPlaceRequest::Oco { payload } => {
                        let tp_order_id = self.next_order_id();
                        let tp_entity =
                            working_order_entity(tp_order_id, sync.key.contract_id, &payload, None);
                        self.active_orders.insert(
                            tp_order_id,
                            SimActiveOrder {
                                order: tp_entity.clone(),
                                link_id: None,
                                strategy_id: None,
                            },
                        );
                        envelopes.push(EntityEnvelope {
                            entity_type: "order".to_string(),
                            deleted: false,
                            entity: tp_entity,
                        });

                        let stop_payload =
                            payload.get("other").cloned().unwrap_or_else(|| json!({}));
                        let stop_order_id = self.next_order_id();
                        let stop_entity = working_order_entity(
                            stop_order_id,
                            sync.key.contract_id,
                            &stop_payload,
                            None,
                        );
                        self.active_orders.insert(
                            stop_order_id,
                            SimActiveOrder {
                                order: stop_entity.clone(),
                                link_id: None,
                                strategy_id: None,
                            },
                        );
                        envelopes.push(EntityEnvelope {
                            entity_type: "order".to_string(),
                            deleted: false,
                            entity: stop_entity,
                        });

                        if let Some(state) = next_state.as_mut() {
                            state.take_profit_order_id = Some(tp_order_id);
                            state.stop_order_id = Some(stop_order_id);
                        }
                    }
                }
            }
        }

        Ok(vec![
            InternalEvent::ProtectionSyncApplied(ProtectionSyncAck {
                key: sync.key,
                message: sync.message,
                next_state,
            }),
            InternalEvent::UserEntities(envelopes),
        ])
    }
}

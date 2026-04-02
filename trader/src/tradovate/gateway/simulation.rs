use super::*;

#[derive(Debug, Clone)]
pub(crate) struct SimPosition {
    pub(crate) position_id: i64,
    pub(crate) qty: i32,
    pub(crate) avg_price: f64,
    pub(crate) symbol: String,
}

#[derive(Debug, Clone)]
pub(crate) struct SimActiveOrder {
    pub(crate) order: Value,
    pub(crate) link_id: Option<i64>,
    pub(crate) strategy_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub(crate) struct SimOrderStrategyState {
    pub(crate) entity: Value,
    pub(crate) order_ids: Vec<i64>,
    pub(crate) link_ids: Vec<i64>,
}

#[cfg(any(feature = "replay", test))]
#[derive(Debug, Clone)]
struct ReplayTriggeredOrder {
    order_id: i64,
    key: StrategyProtectionKey,
    distance_to_open: f64,
    stop_order: bool,
}

#[derive(Default)]
pub(crate) struct ReplayBrokerState {
    pub(crate) next_order_id: i64,
    pub(crate) next_fill_id: i64,
    pub(crate) next_exec_report_id: i64,
    pub(crate) next_position_id: i64,
    pub(crate) next_strategy_id: i64,
    pub(crate) next_link_id: i64,
    pub(crate) positions: HashMap<StrategyProtectionKey, SimPosition>,
    pub(crate) active_orders: HashMap<i64, SimActiveOrder>,
    pub(crate) order_strategies: HashMap<i64, SimOrderStrategyState>,
}

impl ReplayBrokerState {
    fn next_order_id(&mut self) -> i64 {
        self.next_order_id = self.next_order_id.saturating_add(1).max(1000);
        self.next_order_id
    }

    fn next_fill_id(&mut self) -> i64 {
        self.next_fill_id = self.next_fill_id.saturating_add(1).max(10_000);
        self.next_fill_id
    }

    fn next_exec_report_id(&mut self) -> i64 {
        self.next_exec_report_id = self.next_exec_report_id.saturating_add(1).max(20_000);
        self.next_exec_report_id
    }

    fn next_position_id(&mut self) -> i64 {
        self.next_position_id = self.next_position_id.saturating_add(1).max(30_000);
        self.next_position_id
    }

    fn next_strategy_id(&mut self) -> i64 {
        self.next_strategy_id = self.next_strategy_id.saturating_add(1).max(40_000);
        self.next_strategy_id
    }

    fn next_link_id(&mut self) -> i64 {
        self.next_link_id = self.next_link_id.saturating_add(1).max(50_000);
        self.next_link_id
    }

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

    #[cfg(any(feature = "replay", test))]
    pub(crate) fn simulate_replay_bar(&mut self, bar: &Bar) -> Vec<InternalEvent> {
        let triggered_order_ids = self
            .triggered_replay_orders(bar)
            .into_iter()
            .map(|triggered| triggered.order_id)
            .collect::<Vec<_>>();
        if triggered_order_ids.is_empty() {
            return Vec::new();
        }

        let mut envelopes = Vec::new();
        for order_id in triggered_order_ids {
            envelopes.extend(self.fill_replay_order(order_id, bar));
        }

        if envelopes.is_empty() {
            Vec::new()
        } else {
            vec![InternalEvent::UserEntities(envelopes)]
        }
    }

    #[cfg(any(feature = "replay", test))]
    fn triggered_replay_orders(&self, bar: &Bar) -> Vec<ReplayTriggeredOrder> {
        const EPSILON: f64 = 1e-9;

        let mut selected = HashMap::<StrategyProtectionKey, ReplayTriggeredOrder>::new();
        for (order_id, active) in &self.active_orders {
            let Some(key) = replay_order_key(&active.order) else {
                continue;
            };
            let Some(current_qty) = self.positions.get(&key).map(|position| position.qty) else {
                continue;
            };
            if current_qty == 0 || !replay_order_would_reduce_position(&active.order, current_qty) {
                continue;
            }
            if !replay_order_triggered(&active.order, bar) {
                continue;
            }

            let candidate = ReplayTriggeredOrder {
                order_id: *order_id,
                key,
                distance_to_open: replay_order_distance_to_open(&active.order, bar),
                stop_order: replay_order_is_stop(&active.order),
            };
            let replace = match selected.get(&key) {
                None => true,
                Some(current) => {
                    candidate.distance_to_open + EPSILON < current.distance_to_open
                        || ((candidate.distance_to_open - current.distance_to_open).abs()
                            <= EPSILON
                            && candidate.stop_order
                            && !current.stop_order)
                        || ((candidate.distance_to_open - current.distance_to_open).abs()
                            <= EPSILON
                            && candidate.stop_order == current.stop_order
                            && candidate.order_id < current.order_id)
                }
            };
            if replace {
                selected.insert(key, candidate);
            }
        }

        let mut triggered = selected.into_values().collect::<Vec<_>>();
        triggered.sort_by_key(|item| (item.key.account_id, item.key.contract_id, item.order_id));
        triggered
    }

    #[cfg(any(feature = "replay", test))]
    fn fill_replay_order(&mut self, order_id: i64, bar: &Bar) -> Vec<EntityEnvelope> {
        let Some(order) = self
            .active_orders
            .get(&order_id)
            .map(|active| active.order.clone())
        else {
            return Vec::new();
        };
        let Some(key) = replay_order_key(&order) else {
            return Vec::new();
        };
        let current_qty = self
            .positions
            .get(&key)
            .map(|position| position.qty)
            .unwrap_or_default();
        if current_qty == 0 || !replay_order_would_reduce_position(&order, current_qty) {
            return Vec::new();
        }
        let Some(active) = self.active_orders.remove(&order_id) else {
            return Vec::new();
        };

        let action = active
            .order
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let action_sign = replay_order_action_sign(&action).unwrap_or_default();
        let requested_qty = json_i64(&active.order, "orderQty")
            .and_then(|qty| i32::try_from(qty.abs()).ok())
            .unwrap_or(1)
            .max(1);
        let executed_qty = requested_qty.min(current_qty.abs());
        let fill_price =
            replay_order_fill_price(&active.order, bar).unwrap_or_else(|| bar.close.max(0.0));
        let next_qty = current_qty.saturating_add(action_sign.saturating_mul(executed_qty));
        let ts_ns = bar.ts_ns;
        let contract_name = order_symbol(&active.order).unwrap_or_default().to_string();
        let cl_ord_id = active
            .order
            .get("clOrdId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        if let Some(strategy_id) = active.strategy_id {
            if let Some(state) = self.order_strategies.get_mut(&strategy_id) {
                state.order_ids.retain(|id| *id != order_id);
                if let Some(link_id) = active.link_id {
                    state.link_ids.retain(|id| *id != link_id);
                }
            }
        }

        let mut filled_order = active.order.clone();
        if let Some(order) = filled_order.as_object_mut() {
            order.insert("ordStatus".to_string(), Value::String("Filled".to_string()));
            order.insert("price".to_string(), json!(fill_price));
            order.insert("filledQty".to_string(), json!(executed_qty));
        }

        let mut envelopes = vec![EntityEnvelope {
            entity_type: "order".to_string(),
            deleted: false,
            entity: filled_order,
        }];
        if let Some(link_id) = active.link_id {
            envelopes.push(EntityEnvelope {
                entity_type: "orderStrategyLink".to_string(),
                deleted: true,
                entity: json!({
                    "id": link_id,
                    "orderStrategyId": active.strategy_id,
                    "orderId": order_id,
                }),
            });
        }
        envelopes.push(EntityEnvelope {
            entity_type: "executionReport".to_string(),
            deleted: false,
            entity: json!({
                "id": self.next_exec_report_id(),
                "accountId": key.account_id,
                "contractId": key.contract_id,
                "orderId": order_id,
                "clOrdId": cl_ord_id,
                "orderStrategyId": active.strategy_id,
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
                "accountId": key.account_id,
                "contractId": key.contract_id,
                "orderId": order_id,
                "orderStrategyId": active.strategy_id,
                "source": "replay",
                "price": fill_price,
                "qty": executed_qty,
                "buySell": action,
                "timestamp": ts_ns,
                "symbol": contract_name,
            }),
        });
        envelopes.extend(self.update_position(key, &contract_name, next_qty, fill_price));

        if let Some(strategy_id) = active.strategy_id {
            envelopes.extend(self.clear_order_strategy(strategy_id));
        }
        if next_qty == 0 {
            envelopes.extend(self.cancel_orders_for_key_except(key, &[order_id]));
        }

        envelopes
    }

    fn cancel_orders(&mut self, order_ids: &[i64]) -> Vec<EntityEnvelope> {
        let mut envelopes = Vec::new();
        for order_id in order_ids {
            let Some(active) = self.active_orders.remove(order_id) else {
                continue;
            };
            envelopes.push(EntityEnvelope {
                entity_type: "order".to_string(),
                deleted: true,
                entity: active.order.clone(),
            });
            if let Some(link_id) = active.link_id {
                envelopes.push(EntityEnvelope {
                    entity_type: "orderStrategyLink".to_string(),
                    deleted: true,
                    entity: json!({
                        "id": link_id,
                        "orderStrategyId": active.strategy_id,
                        "orderId": order_id,
                    }),
                });
            }
            if let Some(strategy_id) = active.strategy_id {
                if let Some(state) = self.order_strategies.get_mut(&strategy_id) {
                    state.order_ids.retain(|id| *id != *order_id);
                    if let Some(link_id) = active.link_id {
                        state.link_ids.retain(|id| *id != link_id);
                    }
                }
            }
        }
        envelopes
    }

    #[cfg(any(feature = "replay", test))]
    fn cancel_orders_for_key_except(
        &mut self,
        key: StrategyProtectionKey,
        excluded_order_ids: &[i64],
    ) -> Vec<EntityEnvelope> {
        let order_ids = self
            .active_orders
            .iter()
            .filter_map(|(order_id, active)| {
                let matches_key = order_contract_id(&active.order) == Some(key.contract_id)
                    && extract_account_id("order", &active.order) == Some(key.account_id);
                if matches_key && !excluded_order_ids.contains(order_id) {
                    Some(*order_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        self.cancel_orders(&order_ids)
    }

    fn clear_order_strategy(&mut self, order_strategy_id: i64) -> Vec<EntityEnvelope> {
        let Some(state) = self.order_strategies.remove(&order_strategy_id) else {
            return Vec::new();
        };
        let mut envelopes = self.cancel_orders(&state.order_ids);
        for link_id in state.link_ids {
            envelopes.push(EntityEnvelope {
                entity_type: "orderStrategyLink".to_string(),
                deleted: true,
                entity: json!({
                    "id": link_id,
                    "orderStrategyId": order_strategy_id,
                }),
            });
        }
        envelopes.push(EntityEnvelope {
            entity_type: "orderStrategy".to_string(),
            deleted: true,
            entity: state.entity,
        });
        envelopes
    }

    fn clear_orders_for_contract(&mut self, key: StrategyProtectionKey) -> Vec<EntityEnvelope> {
        let order_ids = self
            .active_orders
            .iter()
            .filter_map(|(order_id, active)| {
                if order_contract_id(&active.order) == Some(key.contract_id)
                    && extract_account_id("order", &active.order) == Some(key.account_id)
                {
                    Some(*order_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let strategy_ids = self
            .order_strategies
            .iter()
            .filter_map(|(strategy_id, state)| {
                if strategy_contract_id(&state.entity) == Some(key.contract_id)
                    && strategy_account_id(&state.entity) == Some(key.account_id)
                {
                    Some(*strategy_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut envelopes = self.cancel_orders(&order_ids);
        for strategy_id in strategy_ids {
            envelopes.extend(self.clear_order_strategy(strategy_id));
        }
        envelopes
    }

    fn update_position(
        &mut self,
        key: StrategyProtectionKey,
        contract_name: &str,
        next_qty: i32,
        fill_price: f64,
    ) -> Vec<EntityEnvelope> {
        let current = self.positions.remove(&key);
        if next_qty == 0 {
            return current
                .map(|position| {
                    vec![EntityEnvelope {
                        entity_type: "position".to_string(),
                        deleted: true,
                        entity: json!({
                            "id": position.position_id,
                            "accountId": key.account_id,
                            "contractId": key.contract_id,
                            "symbol": position.symbol,
                            "netPos": position.qty,
                            "avgPrice": position.avg_price,
                        }),
                    }]
                })
                .unwrap_or_default();
        }

        let next_abs = next_qty.abs() as f64;
        let avg_price = match current.as_ref() {
            Some(position)
                if position.qty.signum() == next_qty.signum()
                    && position.qty.abs() < next_qty.abs() =>
            {
                let current_abs = position.qty.abs() as f64;
                ((position.avg_price * current_abs) + fill_price * (next_abs - current_abs))
                    / next_abs.max(1.0)
            }
            Some(position)
                if position.qty.signum() == next_qty.signum()
                    && position.qty.abs() >= next_qty.abs() =>
            {
                position.avg_price
            }
            _ => fill_price,
        };
        let position_id = current
            .as_ref()
            .map(|position| position.position_id)
            .unwrap_or_else(|| self.next_position_id());
        let entity = json!({
            "id": position_id,
            "accountId": key.account_id,
            "contractId": key.contract_id,
            "symbol": contract_name,
            "netPos": next_qty,
            "avgPrice": avg_price,
        });
        self.positions.insert(
            key,
            SimPosition {
                position_id,
                qty: next_qty,
                avg_price,
                symbol: contract_name.to_string(),
            },
        );
        vec![EntityEnvelope {
            entity_type: "position".to_string(),
            deleted: false,
            entity,
        }]
    }
}

fn exit_action_for_target(target_qty: i32) -> &'static str {
    if target_qty > 0 { "Sell" } else { "Buy" }
}

fn synthetic_ts_ns(reference_ts_ns: Option<i64>) -> i64 {
    reference_ts_ns
        .or_else(|| Utc::now().timestamp_nanos_opt())
        .unwrap_or_default()
}

fn working_order_entity(
    order_id: i64,
    contract_id: i64,
    payload: &Value,
    order_strategy_id: Option<i64>,
) -> Value {
    let mut entity = json!({
        "id": order_id,
        "orderId": order_id,
        "accountId": json_i64(payload, "accountId"),
        "contractId": contract_id,
        "symbol": payload.get("symbol").and_then(Value::as_str),
        "action": payload.get("action").and_then(Value::as_str),
        "orderQty": json_i64(payload, "orderQty"),
        "orderType": payload.get("orderType").and_then(Value::as_str).unwrap_or("Limit"),
        "ordStatus": "Working",
    });
    if let Some(cl_ord_id) = payload.get("clOrdId").and_then(Value::as_str) {
        entity["clOrdId"] = Value::String(cl_ord_id.to_string());
    }
    if let Some(price) = payload.get("price").and_then(Value::as_f64) {
        entity["price"] = json!(price);
    }
    if let Some(stop_price) = payload.get("stopPrice").and_then(Value::as_f64) {
        entity["stopPrice"] = json!(stop_price);
    }
    if let Some(order_strategy_id) = order_strategy_id {
        entity["orderStrategyId"] = json!(order_strategy_id);
    }
    entity
}

#[cfg(any(feature = "replay", test))]
fn replay_order_key(order: &Value) -> Option<StrategyProtectionKey> {
    Some(StrategyProtectionKey {
        account_id: extract_account_id("order", order)?,
        contract_id: order_contract_id(order)?,
    })
}

#[cfg(any(feature = "replay", test))]
fn replay_order_action_sign(action: &str) -> Option<i32> {
    match action.trim().to_ascii_lowercase().as_str() {
        "buy" => Some(1),
        "sell" => Some(-1),
        _ => None,
    }
}

#[cfg(any(feature = "replay", test))]
fn replay_order_trigger_price(order: &Value) -> Option<f64> {
    match order_type(order).as_deref().unwrap_or_default() {
        "limit" | "mit" => pick_number(order, &["price"]),
        "stop" | "stoplimit" | "trailingstop" | "trailingstoplimit" => {
            pick_number(order, &["stopPrice", "price"])
        }
        _ => pick_number(order, &["price", "stopPrice"]),
    }
}

#[cfg(any(feature = "replay", test))]
fn replay_order_fill_price(order: &Value, bar: &Bar) -> Option<f64> {
    replay_order_trigger_price(order).or_else(|| Some(bar.close))
}

#[cfg(any(feature = "replay", test))]
fn replay_order_is_stop(order: &Value) -> bool {
    matches!(
        order_type(order).as_deref().unwrap_or_default(),
        "stop" | "stoplimit" | "trailingstop" | "trailingstoplimit"
    )
}

#[cfg(any(feature = "replay", test))]
fn replay_order_triggered(order: &Value, bar: &Bar) -> bool {
    let Some(trigger_price) = replay_order_trigger_price(order) else {
        return false;
    };
    match (
        replay_order_is_stop(order),
        order
            .get("action")
            .and_then(Value::as_str)
            .map(|value| value.trim().to_ascii_lowercase()),
    ) {
        (false, Some(action)) if action == "buy" => bar.low <= trigger_price,
        (false, Some(action)) if action == "sell" => bar.high >= trigger_price,
        (true, Some(action)) if action == "buy" => bar.high >= trigger_price,
        (true, Some(action)) if action == "sell" => bar.low <= trigger_price,
        _ => false,
    }
}

#[cfg(any(feature = "replay", test))]
fn replay_order_would_reduce_position(order: &Value, current_qty: i32) -> bool {
    let Some(action) = order.get("action").and_then(Value::as_str) else {
        return false;
    };
    match replay_order_action_sign(action) {
        Some(sign) if current_qty > 0 => sign < 0,
        Some(sign) if current_qty < 0 => sign > 0,
        _ => false,
    }
}

#[cfg(any(feature = "replay", test))]
fn replay_order_distance_to_open(order: &Value, bar: &Bar) -> f64 {
    replay_order_trigger_price(order)
        .map(|price| (bar.open - price).abs())
        .unwrap_or(f64::INFINITY)
}

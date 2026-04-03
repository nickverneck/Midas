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
    pub(super) fn next_order_id(&mut self) -> i64 {
        self.next_order_id = self.next_order_id.saturating_add(1).max(1000);
        self.next_order_id
    }

    pub(super) fn next_fill_id(&mut self) -> i64 {
        self.next_fill_id = self.next_fill_id.saturating_add(1).max(10_000);
        self.next_fill_id
    }

    pub(super) fn next_exec_report_id(&mut self) -> i64 {
        self.next_exec_report_id = self.next_exec_report_id.saturating_add(1).max(20_000);
        self.next_exec_report_id
    }

    pub(super) fn next_position_id(&mut self) -> i64 {
        self.next_position_id = self.next_position_id.saturating_add(1).max(30_000);
        self.next_position_id
    }

    pub(super) fn next_strategy_id(&mut self) -> i64 {
        self.next_strategy_id = self.next_strategy_id.saturating_add(1).max(40_000);
        self.next_strategy_id
    }

    pub(super) fn next_link_id(&mut self) -> i64 {
        self.next_link_id = self.next_link_id.saturating_add(1).max(50_000);
        self.next_link_id
    }

    pub(super) fn cancel_orders(&mut self, order_ids: &[i64]) -> Vec<EntityEnvelope> {
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
    pub(super) fn cancel_orders_for_key_except(
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

    pub(super) fn clear_order_strategy(&mut self, order_strategy_id: i64) -> Vec<EntityEnvelope> {
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

    pub(super) fn clear_orders_for_contract(
        &mut self,
        key: StrategyProtectionKey,
    ) -> Vec<EntityEnvelope> {
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

    pub(super) fn update_position(
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

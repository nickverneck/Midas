use super::state::ReplayBrokerState;
use super::*;

#[cfg(any(feature = "replay", test))]
#[derive(Debug, Clone)]
struct ReplayTriggeredOrder {
    order_id: i64,
    key: StrategyProtectionKey,
    distance_to_open: f64,
    stop_order: bool,
}

impl ReplayBrokerState {
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

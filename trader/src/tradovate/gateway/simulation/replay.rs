use super::state::ReplayBrokerState;
#[cfg(any(feature = "replay", test))]
use super::*;
#[cfg(any(feature = "replay", test))]
use crate::broker::ReplayBarProtectionPolicy;

#[cfg(any(feature = "replay", test))]
#[derive(Debug, Clone)]
struct ReplayTriggeredOrder {
    order_id: i64,
    key: StrategyProtectionKey,
    distance_to_open: f64,
    stop_order: bool,
    ambiguous_bar: bool,
}

impl ReplayBrokerState {
    #[cfg(any(feature = "replay", test))]
    pub(crate) fn simulate_replay_bar(
        &mut self,
        bar: &Bar,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<InternalEvent> {
        let triggered_orders = self.triggered_replay_orders(bar, policy);
        let mut envelopes = Vec::new();
        for triggered in triggered_orders {
            envelopes.extend(self.fill_replay_order(triggered, bar, policy));
        }
        // Bar-only trailing changes are based on the completed bar and become
        // executable on the next raw bar. This prevents current-bar high/low
        // lookahead from manufacturing an intrabar stop path.
        envelopes.extend(self.update_replay_trailing_orders(bar));

        if envelopes.is_empty() {
            Vec::new()
        } else {
            vec![InternalEvent::UserEntities(envelopes)]
        }
    }

    #[cfg(any(feature = "replay", test))]
    pub(crate) fn simulate_replay_tick(
        &mut self,
        tick: &ReplayMarketTick,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<InternalEvent> {
        let triggered_orders = self.triggered_replay_tick_orders(tick, policy);
        let mut envelopes = Vec::new();
        for triggered in triggered_orders {
            envelopes.extend(self.fill_replay_tick_order(triggered, tick, policy));
        }
        envelopes.extend(self.update_replay_trailing_tick(tick));
        if envelopes.is_empty() {
            Vec::new()
        } else {
            vec![InternalEvent::UserEntities(envelopes)]
        }
    }

    #[cfg(any(feature = "replay", test))]
    pub(crate) fn simulate_replay_dom(
        &mut self,
        dom: &ReplayMarketDom,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<InternalEvent> {
        let triggered_orders = self.triggered_replay_dom_orders(dom, policy);
        let mut envelopes = Vec::new();
        for triggered in triggered_orders {
            envelopes.extend(self.fill_replay_dom_order(triggered, dom, policy));
        }
        if envelopes.is_empty() {
            Vec::new()
        } else {
            vec![InternalEvent::UserEntities(envelopes)]
        }
    }

    #[cfg(any(feature = "replay", test))]
    fn triggered_replay_orders(
        &self,
        bar: &Bar,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<ReplayTriggeredOrder> {
        const EPSILON: f64 = 1e-9;

        let mut candidates = HashMap::<StrategyProtectionKey, Vec<ReplayTriggeredOrder>>::new();
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
                ambiguous_bar: false,
            };
            candidates.entry(key).or_default().push(candidate);
        }

        let mut triggered = candidates
            .into_values()
            .filter_map(|group| {
                let ambiguous_bar = group.iter().any(|item| item.stop_order)
                    && group.iter().any(|item| !item.stop_order);
                let mut selected = group[0].clone();
                for candidate in group.into_iter().skip(1) {
                    let prefer_candidate = match policy {
                        ReplayBarProtectionPolicy::Conservative
                            if candidate.stop_order != selected.stop_order =>
                        {
                            candidate.stop_order
                        }
                        ReplayBarProtectionPolicy::Optimistic
                            if candidate.stop_order != selected.stop_order =>
                        {
                            !candidate.stop_order
                        }
                        _ => {
                            candidate.distance_to_open + EPSILON < selected.distance_to_open
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order
                                    && !selected.stop_order)
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order == selected.stop_order
                                    && candidate.order_id < selected.order_id)
                        }
                    };
                    if prefer_candidate {
                        selected = candidate;
                    }
                }
                selected.ambiguous_bar = ambiguous_bar;
                Some(selected)
            })
            .collect::<Vec<_>>();
        triggered.sort_by_key(|item| (item.key.account_id, item.key.contract_id, item.order_id));
        triggered
    }

    #[cfg(any(feature = "replay", test))]
    fn fill_replay_order(
        &mut self,
        triggered: ReplayTriggeredOrder,
        bar: &Bar,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<EntityEnvelope> {
        let order = self
            .active_orders
            .get(&triggered.order_id)
            .map(|active| active.order.clone());
        let fill_price = order
            .as_ref()
            .and_then(|order| replay_order_fill_price(order, bar))
            .unwrap_or_else(|| bar.close.max(0.0));
        self.fill_replay_order_at(
            triggered,
            fill_price,
            bar.ts_ns,
            "raw_bar_ohlc",
            ReplayExecutionPrecision::BarApproximate,
            policy,
        )
    }

    #[cfg(any(feature = "replay", test))]
    fn fill_replay_tick_order(
        &mut self,
        triggered: ReplayTriggeredOrder,
        tick: &ReplayMarketTick,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<EntityEnvelope> {
        let Some(order) = self
            .active_orders
            .get(&triggered.order_id)
            .map(|active| active.order.clone())
        else {
            return Vec::new();
        };
        let is_quote_exact = tick
            .bid_price
            .is_some_and(|price| price.is_finite() && price > 0.0)
            || tick
                .ask_price
                .is_some_and(|price| price.is_finite() && price > 0.0);
        let fill_price = replay_tick_protection_fill_price(&order, tick);
        self.fill_replay_order_at(
            triggered,
            fill_price,
            tick.ts_ns,
            if is_quote_exact {
                "tick_bid_ask"
            } else {
                "tick_trade_fallback"
            },
            if is_quote_exact {
                ReplayExecutionPrecision::QuoteExact
            } else {
                ReplayExecutionPrecision::TickExact
            },
            policy,
        )
    }

    #[cfg(any(feature = "replay", test))]
    fn fill_replay_dom_order(
        &mut self,
        triggered: ReplayTriggeredOrder,
        dom: &ReplayMarketDom,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<EntityEnvelope> {
        let Some(order) = self
            .active_orders
            .get(&triggered.order_id)
            .map(|active| active.order.clone())
        else {
            return Vec::new();
        };
        let Some(fill_price) = replay_dom_protection_fill_price(&order, dom) else {
            return Vec::new();
        };
        self.fill_replay_order_at(
            triggered,
            fill_price,
            dom.ts_ns,
            "dom_top_of_book",
            ReplayExecutionPrecision::DomAssisted,
            policy,
        )
    }

    #[cfg(any(feature = "replay", test))]
    fn fill_replay_order_at(
        &mut self,
        triggered: ReplayTriggeredOrder,
        fill_price: f64,
        ts_ns: i64,
        fill_source: &str,
        precision: ReplayExecutionPrecision,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<EntityEnvelope> {
        let order_id = triggered.order_id;
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
        let next_qty = current_qty.saturating_add(action_sign.saturating_mul(executed_qty));
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
        let trailing_exit = active
            .replay_auto_trail
            .is_some_and(|trail| trail.active && replay_order_is_stop(&active.order));
        let ambiguity_key = if fill_source.starts_with("tick_") {
            "replayAmbiguousTick"
        } else if fill_source.starts_with("dom_") {
            "replayAmbiguousDom"
        } else {
            "replayAmbiguousBar"
        };
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
        let mut fill_entity = json!({
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
            "replayFillSource": fill_source,
            "replayExecutionPrecision": match precision {
                ReplayExecutionPrecision::BarApproximate => "bar_approximate",
                ReplayExecutionPrecision::TickExact => "tick_exact",
                ReplayExecutionPrecision::QuoteExact => "quote_exact",
                ReplayExecutionPrecision::DomAssisted => "dom_assisted",
            },
            "replayFillTimestampNs": ts_ns,
            "replayProtectionOrderId": order_id,
            "replayExitReason": if trailing_exit {
                "trailing_stop"
            } else if replay_order_is_stop(&active.order) {
                "stop_loss"
            } else {
                "take_profit"
            },
            "replayBarProtectionPolicy": match policy {
                ReplayBarProtectionPolicy::Conservative => "conservative",
                ReplayBarProtectionPolicy::Optimistic => "optimistic",
                ReplayBarProtectionPolicy::NearestOpen => "nearest_open",
            },
        });
        if let Some(object) = fill_entity.as_object_mut() {
            object.insert(ambiguity_key.to_string(), json!(triggered.ambiguous_bar));
        }
        envelopes.push(EntityEnvelope {
            entity_type: "fill".to_string(),
            deleted: false,
            entity: fill_entity,
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

    #[cfg(any(feature = "replay", test))]
    fn update_replay_trailing_orders(&mut self, bar: &Bar) -> Vec<EntityEnvelope> {
        const EPSILON: f64 = 1e-9;
        let order_ids = self.active_orders.keys().copied().collect::<Vec<_>>();
        let mut envelopes = Vec::new();
        for order_id in order_ids {
            let Some(active) = self.active_orders.get_mut(&order_id) else {
                continue;
            };
            let Some(mut trail) = active.replay_auto_trail else {
                continue;
            };
            let Some(current_stop) = replay_order_trigger_price(&active.order) else {
                continue;
            };
            let exit_is_sell = active
                .order
                .get("action")
                .and_then(Value::as_str)
                .is_some_and(|action| action.eq_ignore_ascii_case("Sell"));
            let triggered = if exit_is_sell {
                bar.high + EPSILON >= trail.entry_price + trail.trigger_offset
            } else {
                bar.low - EPSILON <= trail.entry_price - trail.trigger_offset
            };
            if !trail.active && !triggered {
                continue;
            }
            trail.active = true;
            let raw_candidate = if exit_is_sell {
                bar.high - trail.stop_offset
            } else {
                bar.low + trail.stop_offset
            };
            let frequency = trail.frequency.max(EPSILON);
            let tightened = if exit_is_sell {
                let steps = ((raw_candidate - current_stop) / frequency)
                    .floor()
                    .max(0.0);
                current_stop + steps * frequency
            } else {
                let steps = ((current_stop - raw_candidate) / frequency)
                    .floor()
                    .max(0.0);
                current_stop - steps * frequency
            };
            let improves = if exit_is_sell {
                tightened > current_stop + EPSILON
            } else {
                tightened < current_stop - EPSILON
            };
            active.replay_auto_trail = Some(trail);
            let Some(order) = active.order.as_object_mut() else {
                continue;
            };
            order.insert("replayTrailingActive".to_string(), json!(true));
            if improves {
                order.insert("stopPrice".to_string(), json!(tightened));
            }
            envelopes.push(EntityEnvelope {
                entity_type: "order".to_string(),
                deleted: false,
                entity: active.order.clone(),
            });
        }
        envelopes
    }

    #[cfg(any(feature = "replay", test))]
    fn triggered_replay_tick_orders(
        &self,
        tick: &ReplayMarketTick,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<ReplayTriggeredOrder> {
        const EPSILON: f64 = 1e-9;
        let mut candidates = HashMap::<StrategyProtectionKey, Vec<ReplayTriggeredOrder>>::new();
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
            let Some(trigger_price) = replay_order_trigger_price(&active.order) else {
                continue;
            };
            let Some(action) = active.order.get("action").and_then(Value::as_str) else {
                continue;
            };
            let market_price = if action.eq_ignore_ascii_case("buy") {
                tick.ask_price
                    .filter(|price| price.is_finite() && *price > 0.0)
                    .unwrap_or(tick.last)
            } else {
                tick.bid_price
                    .filter(|price| price.is_finite() && *price > 0.0)
                    .unwrap_or(tick.last)
            };
            let stop_order = replay_order_is_stop(&active.order);
            let triggered = if stop_order {
                if action.eq_ignore_ascii_case("buy") {
                    market_price + EPSILON >= trigger_price
                } else {
                    market_price - EPSILON <= trigger_price
                }
            } else if action.eq_ignore_ascii_case("buy") {
                market_price - EPSILON <= trigger_price
            } else {
                market_price + EPSILON >= trigger_price
            };
            if !triggered {
                continue;
            }
            candidates
                .entry(key)
                .or_default()
                .push(ReplayTriggeredOrder {
                    order_id: *order_id,
                    key,
                    distance_to_open: (market_price - trigger_price).abs(),
                    stop_order,
                    ambiguous_bar: false,
                });
        }
        let mut triggered = candidates
            .into_values()
            .filter_map(|group| {
                let ambiguous = group.iter().any(|item| item.stop_order)
                    && group.iter().any(|item| !item.stop_order);
                let mut selected = group[0].clone();
                for candidate in group.into_iter().skip(1) {
                    let prefer = match policy {
                        ReplayBarProtectionPolicy::Conservative
                            if candidate.stop_order != selected.stop_order =>
                        {
                            candidate.stop_order
                        }
                        ReplayBarProtectionPolicy::Optimistic
                            if candidate.stop_order != selected.stop_order =>
                        {
                            !candidate.stop_order
                        }
                        _ => {
                            candidate.distance_to_open + EPSILON < selected.distance_to_open
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order
                                    && !selected.stop_order)
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order == selected.stop_order
                                    && candidate.order_id < selected.order_id)
                        }
                    };
                    if prefer {
                        selected = candidate;
                    }
                }
                selected.ambiguous_bar = ambiguous;
                Some(selected)
            })
            .collect::<Vec<_>>();
        triggered.sort_by_key(|item| (item.key.account_id, item.key.contract_id, item.order_id));
        triggered
    }

    #[cfg(any(feature = "replay", test))]
    fn triggered_replay_dom_orders(
        &self,
        dom: &ReplayMarketDom,
        policy: ReplayBarProtectionPolicy,
    ) -> Vec<ReplayTriggeredOrder> {
        const EPSILON: f64 = 1e-9;
        let mut candidates = HashMap::<StrategyProtectionKey, Vec<ReplayTriggeredOrder>>::new();
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
            let Some(trigger_price) = replay_order_trigger_price(&active.order) else {
                continue;
            };
            let Some(action) = active.order.get("action").and_then(Value::as_str) else {
                continue;
            };
            let Some(market_price) = replay_dom_executable_price(action, dom) else {
                continue;
            };
            let stop_order = replay_order_is_stop(&active.order);
            let triggered = if stop_order {
                if action.eq_ignore_ascii_case("buy") {
                    market_price + EPSILON >= trigger_price
                } else {
                    market_price - EPSILON <= trigger_price
                }
            } else if action.eq_ignore_ascii_case("buy") {
                market_price - EPSILON <= trigger_price
            } else {
                market_price + EPSILON >= trigger_price
            };
            if !triggered {
                continue;
            }
            candidates
                .entry(key)
                .or_default()
                .push(ReplayTriggeredOrder {
                    order_id: *order_id,
                    key,
                    distance_to_open: (market_price - trigger_price).abs(),
                    stop_order,
                    ambiguous_bar: false,
                });
        }
        let mut triggered = candidates
            .into_values()
            .filter_map(|group| {
                let ambiguous = group.iter().any(|item| item.stop_order)
                    && group.iter().any(|item| !item.stop_order);
                let mut selected = group[0].clone();
                for candidate in group.into_iter().skip(1) {
                    let prefer = match policy {
                        ReplayBarProtectionPolicy::Conservative
                            if candidate.stop_order != selected.stop_order =>
                        {
                            candidate.stop_order
                        }
                        ReplayBarProtectionPolicy::Optimistic
                            if candidate.stop_order != selected.stop_order =>
                        {
                            !candidate.stop_order
                        }
                        _ => {
                            candidate.distance_to_open + EPSILON < selected.distance_to_open
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order
                                    && !selected.stop_order)
                                || ((candidate.distance_to_open - selected.distance_to_open).abs()
                                    <= EPSILON
                                    && candidate.stop_order == selected.stop_order
                                    && candidate.order_id < selected.order_id)
                        }
                    };
                    if prefer {
                        selected = candidate;
                    }
                }
                selected.ambiguous_bar = ambiguous;
                Some(selected)
            })
            .collect::<Vec<_>>();
        triggered.sort_by_key(|item| (item.key.account_id, item.key.contract_id, item.order_id));
        triggered
    }

    #[cfg(any(feature = "replay", test))]
    fn update_replay_trailing_tick(&mut self, tick: &ReplayMarketTick) -> Vec<EntityEnvelope> {
        const EPSILON: f64 = 1e-9;
        let order_ids = self.active_orders.keys().copied().collect::<Vec<_>>();
        let mut envelopes = Vec::new();
        for order_id in order_ids {
            let Some(active) = self.active_orders.get_mut(&order_id) else {
                continue;
            };
            let Some(mut trail) = active.replay_auto_trail else {
                continue;
            };
            let Some(current_stop) = replay_order_trigger_price(&active.order) else {
                continue;
            };
            let exit_is_sell = active
                .order
                .get("action")
                .and_then(Value::as_str)
                .is_some_and(|action| action.eq_ignore_ascii_case("Sell"));
            let triggered = if exit_is_sell {
                tick.last + EPSILON >= trail.entry_price + trail.trigger_offset
            } else {
                tick.last - EPSILON <= trail.entry_price - trail.trigger_offset
            };
            if !trail.active && !triggered {
                continue;
            }
            trail.active = true;
            let raw_candidate = if exit_is_sell {
                tick.last - trail.stop_offset
            } else {
                tick.last + trail.stop_offset
            };
            let frequency = trail.frequency.max(EPSILON);
            let tightened = if exit_is_sell {
                current_stop
                    + (((raw_candidate - current_stop) / frequency)
                        .floor()
                        .max(0.0)
                        * frequency)
            } else {
                current_stop
                    - (((current_stop - raw_candidate) / frequency)
                        .floor()
                        .max(0.0)
                        * frequency)
            };
            let improves = if exit_is_sell {
                tightened > current_stop + EPSILON
            } else {
                tightened < current_stop - EPSILON
            };
            active.replay_auto_trail = Some(trail);
            let Some(order) = active.order.as_object_mut() else {
                continue;
            };
            order.insert("replayTrailingActive".to_string(), json!(true));
            if improves {
                order.insert("stopPrice".to_string(), json!(tightened));
            }
            envelopes.push(EntityEnvelope {
                entity_type: "order".to_string(),
                deleted: false,
                entity: active.order.clone(),
            });
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
fn replay_tick_protection_fill_price(order: &Value, tick: &ReplayMarketTick) -> f64 {
    if replay_order_is_stop(order) {
        match order.get("action").and_then(Value::as_str) {
            Some(action) if action.eq_ignore_ascii_case("buy") => tick
                .ask_price
                .filter(|price| price.is_finite() && *price > 0.0)
                .unwrap_or(tick.last),
            _ => tick
                .bid_price
                .filter(|price| price.is_finite() && *price > 0.0)
                .unwrap_or(tick.last),
        }
    } else {
        replay_order_trigger_price(order).unwrap_or(tick.last)
    }
}

#[cfg(any(feature = "replay", test))]
fn replay_dom_executable_price(action: &str, dom: &ReplayMarketDom) -> Option<f64> {
    let levels = if action.eq_ignore_ascii_case("buy") {
        &dom.asks
    } else if action.eq_ignore_ascii_case("sell") {
        &dom.bids
    } else {
        return None;
    };
    levels.iter().find_map(|level| {
        (level.price.is_finite() && level.price > 0.0 && level.size.is_finite() && level.size > 0.0)
            .then_some(level.price)
    })
}

#[cfg(any(feature = "replay", test))]
fn replay_dom_protection_fill_price(order: &Value, dom: &ReplayMarketDom) -> Option<f64> {
    let action = order.get("action").and_then(Value::as_str)?;
    if replay_order_is_stop(order) {
        replay_dom_executable_price(action, dom)
    } else {
        replay_order_trigger_price(order).or_else(|| replay_dom_executable_price(action, dom))
    }
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

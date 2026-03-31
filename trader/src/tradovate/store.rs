impl UserSyncStore {
    fn apply(&mut self, envelope: EntityEnvelope) {
        let entity_type = envelope.entity_type.to_ascii_lowercase();
        let Some(entity_id) = extract_entity_id(&envelope.entity) else {
            return;
        };

        match entity_type.as_str() {
            "account" => {
                if envelope.deleted {
                    self.accounts.remove(&entity_id);
                } else {
                    self.accounts.insert(entity_id, envelope.entity);
                }
            }
            "accountriskstatus" => {
                let Some(account_id) = extract_account_id("accountRiskStatus", &envelope.entity)
                else {
                    return;
                };
                if envelope.deleted {
                    self.risk.remove(&account_id);
                } else {
                    self.risk.insert(account_id, envelope.entity);
                }
            }
            "cashbalance" => {
                let Some(account_id) = extract_account_id("cashBalance", &envelope.entity) else {
                    return;
                };
                if envelope.deleted {
                    self.cash.remove(&account_id);
                } else {
                    self.cash.insert(account_id, envelope.entity);
                }
            }
            "position" => {
                let Some(account_id) = extract_account_id("position", &envelope.entity) else {
                    return;
                };
                let bucket = self.positions.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
                }
            }
            "order" => {
                let Some(account_id) = extract_account_id("order", &envelope.entity) else {
                    return;
                };
                let bucket = self.orders.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
                }
            }
            "orderstrategy" => {
                if envelope.deleted {
                    self.order_strategies.remove(&entity_id);
                } else {
                    self.order_strategies.insert(entity_id, envelope.entity);
                }
            }
            "orderstrategylink" => {
                if envelope.deleted {
                    self.order_strategy_links.remove(&entity_id);
                } else {
                    self.order_strategy_links.insert(entity_id, envelope.entity);
                }
            }
            "fill" => {
                let Some(account_id) = extract_account_id("fill", &envelope.entity) else {
                    return;
                };
                if envelope.deleted {
                    let remove_bucket = if let Some(bucket) = self.fills.get_mut(&account_id) {
                        bucket.remove(&entity_id);
                        bucket.is_empty()
                    } else {
                        false
                    };
                    if remove_bucket {
                        self.fills.remove(&account_id);
                    }
                } else if is_replay_entity(&envelope.entity) {
                    self.fills
                        .entry(account_id)
                        .or_default()
                        .insert(entity_id, envelope.entity);
                }
            }
            _ => {}
        }
    }

    fn build_snapshots(
        &self,
        accounts: &[AccountInfo],
        market: Option<&MarketSnapshot>,
        managed_protection: &BTreeMap<StrategyProtectionKey, ManagedProtectionOrders>,
    ) -> Vec<AccountSnapshot> {
        accounts
            .iter()
            .map(|account| {
                let raw_account = self
                    .accounts
                    .get(&account.id)
                    .cloned()
                    .or_else(|| Some(account.raw.clone()));
                let raw_risk = self.risk.get(&account.id).cloned();
                let raw_cash = self.cash.get(&account.id).cloned();
                let raw_positions = self
                    .positions
                    .get(&account.id)
                    .map(|items| items.values().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();
                let raw_fills = self
                    .fills
                    .get(&account.id)
                    .map(|items| items.values().cloned().collect::<Vec<_>>())
                    .unwrap_or_default();
                let replay_account = raw_account.as_ref().is_some_and(is_replay_entity)
                    || raw_risk.as_ref().is_some_and(is_replay_entity)
                    || raw_cash.as_ref().is_some_and(is_replay_entity);

                let mut balance = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "balance",
                                "netLiq",
                                "netLiquidationValue",
                                "netLiquidation",
                                "cashBalance",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_cash.as_ref().and_then(|value| {
                            pick_number(value, &["cashBalance", "totalCashValue", "amount"])
                        })
                    })
                    .or_else(|| {
                        raw_account
                            .as_ref()
                            .and_then(|value| pick_number(value, &["balance", "netLiq"]))
                    });
                let mut cash_balance = raw_cash.as_ref().and_then(|value| {
                    pick_number(
                        value,
                        &["cashBalance", "totalCashValue", "amount", "balance"],
                    )
                });
                let mut net_liq = raw_risk.as_ref().and_then(|value| {
                    pick_number(
                        value,
                        &[
                            "netLiq",
                            "netLiquidationValue",
                            "netLiquidation",
                            "balance",
                            "cashBalance",
                        ],
                    )
                });
                let mut realized_pnl = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "realizedPnL",
                                "realizedPnl",
                                "realizedProfitAndLoss",
                                "realizedProfitLoss",
                                "sessionRealizedPnL",
                                "sessionRealizedPnl",
                                "todayRealizedPnL",
                                "todayRealizedPnl",
                                "closedPnL",
                                "closedPnl",
                                "dayPnL",
                                "dayPnl",
                                "dailyPnL",
                                "dailyPnl",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_account.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "realizedPnL",
                                    "realizedPnl",
                                    "realizedProfitAndLoss",
                                    "realizedProfitLoss",
                                    "sessionRealizedPnL",
                                    "sessionRealizedPnl",
                                    "todayRealizedPnL",
                                    "todayRealizedPnl",
                                    "closedPnL",
                                    "closedPnl",
                                ],
                            )
                        })
                    })
                    .or_else(|| {
                        raw_cash.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "realizedPnL",
                                    "realizedPnl",
                                    "sessionRealizedPnL",
                                    "sessionRealizedPnl",
                                    "todayRealizedPnL",
                                    "todayRealizedPnl",
                                ],
                            )
                        })
                    });
                let intraday_margin = raw_risk
                    .as_ref()
                    .and_then(|value| {
                        pick_number(
                            value,
                            &[
                                "intradayMargin",
                                "dayMargin",
                                "dayTradeMargin",
                                "dayTradeMarginReq",
                                "marginRequirement",
                                "marginUsed",
                                "totalMargin",
                                "initialMarginReq",
                                "requiredIntradayMargin",
                                "initialMargin",
                                "maintenanceMargin",
                                "maintenanceMarginReq",
                                "marginReq",
                                "margin",
                            ],
                        )
                    })
                    .or_else(|| {
                        raw_account.as_ref().and_then(|value| {
                            pick_number(
                                value,
                                &[
                                    "intradayMargin",
                                    "dayTradeMargin",
                                    "dayTradeMarginReq",
                                    "initialMargin",
                                    "maintenanceMargin",
                                    "marginRequirement",
                                ],
                            )
                        })
                    });
                let mut unrealized_pnl = sum_position_metric(
                    &raw_positions,
                    &[
                        "unrealizedPnL",
                        "unrealizedPnl",
                        "floatingPnL",
                        "floatingPnl",
                        "openProfitAndLoss",
                        "netPnL",
                        "netPnl",
                        "openPnL",
                        "openPnl",
                    ],
                );
                unrealized_pnl = unrealized_pnl.or_else(|| {
                    market.and_then(|market| fallback_unrealized_pnl(&raw_positions, market))
                });
                net_liq = net_liq.or_else(|| match (balance, unrealized_pnl) {
                    (Some(balance), Some(unrealized)) => Some(balance + unrealized),
                    _ => None,
                });
                if replay_account {
                    let starting_balance = raw_account
                        .as_ref()
                        .and_then(replay_starting_balance)
                        .or_else(|| raw_risk.as_ref().and_then(replay_starting_balance))
                        .or_else(|| raw_cash.as_ref().and_then(replay_starting_balance))
                        .or(balance)
                        .or(cash_balance)
                        .or(net_liq)
                        .unwrap_or_default();
                    realized_pnl =
                        Some(replay_session_realized_pnl(&raw_fills, market).unwrap_or_default());
                    unrealized_pnl = Some(unrealized_pnl.unwrap_or_default());
                    balance = Some(starting_balance + realized_pnl.unwrap_or_default());
                    cash_balance = balance;
                    net_liq =
                        Some(balance.unwrap_or_default() + unrealized_pnl.unwrap_or_default());
                }
                let open_position_qty = sum_position_metric(
                    &raw_positions,
                    &["netPos", "netPosition", "qty", "quantity", "netQty"],
                );
                let market_position_qty = market.and_then(|market| {
                    let values = raw_positions
                        .iter()
                        .filter(|position| position_matches_market(position, market))
                        .filter_map(position_qty)
                        .collect::<Vec<_>>();
                    if values.is_empty() {
                        None
                    } else {
                        Some(values.iter().sum())
                    }
                });
                let market_entry_price =
                    market.and_then(|market| weighted_market_entry_price(&raw_positions, market));
                let (selected_contract_take_profit_price, selected_contract_stop_price) = market
                    .and_then(|market| {
                        market.contract_id.map(|contract_id| StrategyProtectionKey {
                            account_id: account.id,
                            contract_id,
                        })
                    })
                    .and_then(|key| managed_protection.get(&key))
                    .map(|orders| (orders.take_profit_price, orders.stop_price))
                    .unwrap_or((None, None));

                AccountSnapshot {
                    account_id: account.id,
                    account_name: account.name.clone(),
                    balance,
                    cash_balance,
                    net_liq,
                    realized_pnl,
                    unrealized_pnl,
                    intraday_margin,
                    open_position_qty,
                    market_position_qty,
                    market_entry_price,
                    selected_contract_take_profit_price,
                    selected_contract_stop_price,
                    raw_account,
                    raw_risk,
                    raw_cash,
                    raw_positions,
                }
            })
            .collect()
    }

    fn find_order(&self, account_id: i64, order_id: i64) -> Option<&Value> {
        self.orders.get(&account_id)?.get(&order_id)
    }

    fn contract_position_qty(&self, account_id: i64, contract: &ContractSuggestion) -> Option<f64> {
        let values = self
            .positions
            .get(&account_id)
            .into_iter()
            .flat_map(|positions| positions.values())
            .filter(|position| position_matches_contract(position, contract))
            .filter_map(position_qty)
            .collect::<Vec<_>>();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum())
        }
    }

    fn order_id_by_client_id(&self, account_id: i64, cl_ord_id: &str) -> Option<i64> {
        self.orders
            .get(&account_id)
            .into_iter()
            .flat_map(|orders| orders.values())
            .find(|order| {
                order
                    .get("clOrdId")
                    .and_then(Value::as_str)
                    .map(|value| value == cl_ord_id)
                    .unwrap_or(false)
            })
            .and_then(extract_entity_id)
    }

    fn find_active_order_strategy(&self, account_id: i64, contract_id: i64) -> Option<&Value> {
        self.order_strategies
            .values()
            .filter(|strategy| {
                strategy_is_owned_by_midas(strategy)
                    && order_strategy_is_active(strategy)
                    && self.order_strategy_matches_selected_contract(
                        strategy,
                        account_id,
                        contract_id,
                    )
            })
            .max_by_key(|strategy| extract_entity_id(strategy).unwrap_or_default())
    }

    fn linked_strategy_orders(&self, account_id: i64, order_strategy_id: i64) -> Vec<&Value> {
        let Some(orders) = self.orders.get(&account_id) else {
            return Vec::new();
        };
        self.order_strategy_links
            .values()
            .filter(|link| json_i64(link, "orderStrategyId") == Some(order_strategy_id))
            .filter_map(|link| json_i64(link, "orderId"))
            .filter_map(|order_id| orders.get(&order_id))
            .collect()
    }

    fn order_strategy_matches_selected_contract(
        &self,
        strategy: &Value,
        account_id: i64,
        contract_id: i64,
    ) -> bool {
        if strategy_account_id(strategy) == Some(account_id)
            && strategy_contract_id(strategy) == Some(contract_id)
        {
            return true;
        }

        let Some(order_strategy_id) = extract_entity_id(strategy) else {
            return false;
        };
        self.linked_strategy_orders(account_id, order_strategy_id)
            .into_iter()
            .any(|order| order_is_active(order) && order_contract_id(order) == Some(contract_id))
    }
}

fn strategy_is_owned_by_midas(strategy: &Value) -> bool {
    strategy
        .get("customTag50")
        .and_then(Value::as_str)
        .is_some_and(|tag| tag.starts_with("midas-"))
        || strategy
            .get("uuid")
            .and_then(Value::as_str)
            .is_some_and(|uuid| uuid.starts_with("midas-"))
}

fn strategy_account_id(strategy: &Value) -> Option<i64> {
    json_i64(strategy, "accountId")
        .or_else(|| {
            strategy
                .get("account")
                .and_then(|account| json_i64(account, "id"))
        })
        .or_else(|| strategy.get("account").and_then(Value::as_i64))
}

fn order_strategy_is_active(strategy: &Value) -> bool {
    let Some(status) = strategy
        .get("status")
        .and_then(Value::as_str)
        .or_else(|| strategy.get("strategyStatus").and_then(Value::as_str))
    else {
        return true;
    };

    let status = status.trim().to_ascii_lowercase();
    if status.is_empty() {
        return true;
    }

    !matches!(
        status.as_str(),
        "closed"
            | "closedstrategy"
            | "completed"
            | "completedstrategy"
            | "finished"
            | "finishedstrategy"
            | "inactive"
            | "inactivestrategy"
            | "interrupted"
            | "interruptedstrategy"
            | "rejected"
            | "rejectedstrategy"
            | "stopped"
            | "stoppedstrategy"
    )
}

fn pick_number(value: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|key| json_number(value, key))
}

fn sum_position_metric(positions: &[Value], keys: &[&str]) -> Option<f64> {
    let values = positions
        .iter()
        .filter_map(|position| pick_number(position, keys))
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum())
    }
}

fn is_replay_entity(value: &Value) -> bool {
    value.get("source")
        .and_then(Value::as_str)
        .is_some_and(|source| source.eq_ignore_ascii_case("replay"))
}

fn replay_starting_balance(value: &Value) -> Option<f64> {
    pick_number(
        value,
        &[
            "startingBalance",
            "initialBalance",
            "starting_balance",
            "balance",
            "cashBalance",
        ],
    )
}

fn replay_session_realized_pnl(fills: &[Value], market: Option<&MarketSnapshot>) -> Option<f64> {
    let market = market?;
    let value_per_point = market.value_per_point?;
    let mut ordered = fills
        .iter()
        .filter(|fill| fill_matches_market(fill, market))
        .cloned()
        .collect::<Vec<_>>();
    ordered.sort_by_key(|fill| {
        (
            json_i64(fill, "timestamp").unwrap_or_default(),
            extract_entity_id(fill).unwrap_or_default(),
        )
    });

    let mut position_qty = 0.0_f64;
    let mut avg_price = 0.0_f64;
    let mut realized = 0.0_f64;
    for fill in ordered {
        let Some(fill_qty) = replay_fill_signed_qty(&fill) else {
            continue;
        };
        let Some(fill_price) = pick_number(&fill, &["price"]) else {
            continue;
        };

        if position_qty.abs() <= f64::EPSILON || position_qty.signum() == fill_qty.signum() {
            let next_abs = position_qty.abs() + fill_qty.abs();
            avg_price = if position_qty.abs() <= f64::EPSILON {
                fill_price
            } else {
                ((avg_price * position_qty.abs()) + (fill_price * fill_qty.abs()))
                    / next_abs.max(1.0)
            };
            position_qty += fill_qty;
            continue;
        }

        let close_qty = position_qty.abs().min(fill_qty.abs());
        realized += match position_qty.signum() as i32 {
            1 => (fill_price - avg_price) * close_qty * value_per_point,
            -1 => (avg_price - fill_price) * close_qty * value_per_point,
            _ => 0.0,
        };

        let prior_abs = position_qty.abs();
        position_qty += fill_qty;
        if position_qty.abs() <= f64::EPSILON {
            position_qty = 0.0;
            avg_price = 0.0;
        } else if fill_qty.abs() > prior_abs {
            avg_price = fill_price;
        }
    }

    Some(realized)
}

fn fill_matches_market(fill: &Value, market: &MarketSnapshot) -> bool {
    let contract_id_match = market
        .contract_id
        .is_some_and(|contract_id| json_i64(fill, "contractId") == Some(contract_id));
    let symbol_match = market
        .contract_name
        .as_deref()
        .zip(
            fill.get("symbol")
                .and_then(Value::as_str)
                .or_else(|| fill.get("contractName").and_then(Value::as_str)),
        )
        .is_some_and(|(expected, actual)| actual.eq_ignore_ascii_case(expected));
    contract_id_match || symbol_match
}

fn replay_fill_signed_qty(fill: &Value) -> Option<f64> {
    let qty = pick_number(fill, &["qty", "quantity"])?.abs();
    let side = fill
        .get("buySell")
        .and_then(Value::as_str)
        .or_else(|| fill.get("action").and_then(Value::as_str))?;
    match side.to_ascii_lowercase().as_str() {
        "buy" => Some(qty),
        "sell" => Some(-qty),
        _ => None,
    }
}

fn fallback_unrealized_pnl(positions: &[Value], market: &MarketSnapshot) -> Option<f64> {
    let last_close = market.bars.last().map(|bar| bar.close)?;
    let value_per_point = market.value_per_point?;
    let values = positions
        .iter()
        .filter(|position| position_matches_market(position, market))
        .filter_map(|position| {
            let qty = position_qty(position)?;
            let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
            Some((last_close - entry_price) * qty * value_per_point)
        })
        .collect::<Vec<_>>();

    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum())
    }
}

fn weighted_market_entry_price(positions: &[Value], market: &MarketSnapshot) -> Option<f64> {
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in positions
        .iter()
        .filter(|position| position_matches_market(position, market))
    {
        let qty = position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = pick_number(position, &["netPrice", "avgPrice", "averagePrice"])?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }

    if total_qty <= f64::EPSILON {
        None
    } else {
        Some(weighted_sum / total_qty)
    }
}

fn position_matches_contract(position: &Value, contract: &ContractSuggestion) -> bool {
    let contract_id_match = position_contract_id(position) == Some(contract.id);
    let contract_maturity_match = json_i64(&contract.raw, "contractMaturityId")
        .zip(position_contract_maturity_id(position))
        .is_some_and(|(expected, actual)| expected == actual);
    let symbol_match =
        position_symbol(position).is_some_and(|symbol| symbol.eq_ignore_ascii_case(&contract.name));

    contract_id_match || contract_maturity_match || symbol_match
}

fn position_matches_market(position: &Value, market: &MarketSnapshot) -> bool {
    let contract_id_match = market
        .contract_id
        .is_some_and(|contract_id| position_contract_id(position) == Some(contract_id));
    let symbol_match = market
        .contract_name
        .as_deref()
        .zip(position_symbol(position))
        .is_some_and(|(expected, actual)| actual.eq_ignore_ascii_case(expected));
    contract_id_match || symbol_match
}

fn position_qty(position: &Value) -> Option<f64> {
    if let Some(net_qty) = pick_number(position, &["netPos", "netPosition", "netQty"]) {
        return Some(net_qty);
    }

    let raw_qty = pick_number(position, &["qty", "quantity"])?;
    let sign = position_side_sign(position).unwrap_or_else(|| {
        if raw_qty < 0.0 {
            -1.0
        } else {
            1.0
        }
    });
    Some(raw_qty.abs() * sign)
}

fn position_contract_id(position: &Value) -> Option<i64> {
    json_i64(position, "contractId").or_else(|| {
        position
            .get("contract")
            .and_then(|contract| json_i64(contract, "id"))
    })
}

fn position_contract_maturity_id(position: &Value) -> Option<i64> {
    json_i64(position, "contractMaturityId").or_else(|| {
        position
            .get("contract")
            .and_then(|contract| json_i64(contract, "contractMaturityId"))
    })
}

fn position_symbol(position: &Value) -> Option<&str> {
    position
        .get("symbol")
        .and_then(Value::as_str)
        .or_else(|| position.get("contractSymbol").and_then(Value::as_str))
        .or_else(|| position.get("name").and_then(Value::as_str))
        .or_else(|| {
            position
                .get("contract")
                .and_then(|contract| contract.get("name"))
                .and_then(Value::as_str)
        })
}

fn position_side_sign(position: &Value) -> Option<f64> {
    if let Some(value) = ["buySell", "side", "action", "positionSide"]
        .iter()
        .find_map(|key| position.get(*key).and_then(Value::as_str))
    {
        return match value.trim().to_ascii_lowercase().as_str() {
            "buy" | "bot" | "b" | "long" => Some(1.0),
            "sell" | "sld" | "s" | "short" => Some(-1.0),
            _ => None,
        };
    }

    position.get("isShort").and_then(Value::as_bool).map(|is_short| {
        if is_short {
            -1.0
        } else {
            1.0
        }
    })
}

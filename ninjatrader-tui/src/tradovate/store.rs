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
                let bucket = self.fills.entry(account_id).or_default();
                if envelope.deleted {
                    bucket.remove(&entity_id);
                } else {
                    bucket.insert(entity_id, envelope.entity);
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

                let balance = raw_risk
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
                let cash_balance = raw_cash.as_ref().and_then(|value| {
                    pick_number(
                        value,
                        &["cashBalance", "totalCashValue", "amount", "balance"],
                    )
                });
                let net_liq = raw_risk.as_ref().and_then(|value| {
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
                let realized_pnl = raw_risk
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
                let unrealized_pnl = sum_position_metric(
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
                let unrealized_pnl = unrealized_pnl.or_else(|| {
                    market.and_then(|market| fallback_unrealized_pnl(&raw_positions, market))
                });
                let net_liq = net_liq.or_else(|| match (balance, unrealized_pnl) {
                    (Some(balance), Some(unrealized)) => Some(balance + unrealized),
                    _ => None,
                });
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
                extract_account_id("orderStrategy", strategy) == Some(account_id)
                    && strategy_contract_id(strategy) == Some(contract_id)
                    && strategy
                        .get("status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| status.eq_ignore_ascii_case("ActiveStrategy"))
                    && strategy_is_owned_by_midas(strategy)
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
    pick_number(
        position,
        &["netPos", "netPosition", "qty", "quantity", "netQty"],
    )
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

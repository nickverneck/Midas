use super::*;

fn empty_history(run: &EngineRunState) -> EngineHistorySnapshot {
    EngineHistorySnapshot {
        run_id: run.run_id.clone(),
        started_at_utc: run.started_at_utc,
        updated_at_utc: None,
        account_id: run.account_id,
        account_name: run.account_name.clone(),
        contract_id: run.contract_id,
        contract_name: run.contract_name.clone(),
        position_qty: 0,
        average_entry_price: None,
        realized_pnl: 0.0,
        unrealized_pnl: 0.0,
        fees: 0.0,
        wins: 0,
        losses: 0,
        fills: Vec::new(),
    }
}

pub(super) fn start_engine_run(session: &mut SessionState) -> Result<()> {
    let account_id = session
        .selected_account_id
        .context("select an account before arming strategy history")?;
    let account_name = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .map(|account| account.name.clone())
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .as_ref()
        .context("select a contract before arming strategy history")?;
    let starting_position = session
        .user_store
        .contract_position_qty(account_id, contract)
        .unwrap_or_default()
        .round() as i32;
    if starting_position != 0 {
        bail!(
            "selected contract {} has pre-existing broker position {}; flatten it before starting an attributable engine run",
            contract.name,
            starting_position
        );
    }
    let started_at_utc = Utc::now();
    let run_id = if session.replay_enabled {
        session
            .cfg
            .replay_run_id
            .clone()
            .filter(|run_id| !run_id.trim().is_empty())
            .unwrap_or_else(|| {
                format!(
                    "{:x}{:x}",
                    started_at_utc.timestamp_millis(),
                    std::process::id()
                )
            })
    } else {
        format!(
            "{:x}{:x}",
            started_at_utc.timestamp_millis(),
            std::process::id()
        )
    };
    let mut run = EngineRunState {
        order_prefix: format!("midas-r{run_id}"),
        run_id,
        started_at_utc,
        account_id,
        account_name,
        contract_id: contract.id,
        contract_name: contract.name.clone(),
        owned_order_ids: BTreeSet::new(),
        history: EngineHistorySnapshot {
            run_id: String::new(),
            started_at_utc,
            updated_at_utc: None,
            account_id,
            account_name: String::new(),
            contract_id: contract.id,
            contract_name: String::new(),
            position_qty: 0,
            average_entry_price: None,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            fees: 0.0,
            wins: 0,
            losses: 0,
            fills: Vec::new(),
        },
    };
    run.history = empty_history(&run);
    session.engine_run = Some(run);
    refresh_engine_history(session);
    Ok(())
}

pub(super) fn emit_engine_history(
    event_tx: &UnboundedSender<ServiceEvent>,
    session: &SessionState,
) {
    if let Some(run) = session.engine_run.as_ref() {
        let _ = event_tx.send(ServiceEvent::EngineHistoryUpdated(run.history.clone()));
    }
}

pub(super) async fn refresh_engine_history_from_broker(
    client: &Client,
    session: &mut SessionState,
) {
    if session.replay_enabled || session.engine_run.is_none() {
        return;
    }
    for entity in [
        "order",
        "command",
        "orderStrategy",
        "orderStrategyLink",
        "fill",
        "fillFee",
    ] {
        let Ok(items) = fetch_entity_list(
            client,
            &session.cfg.env,
            &session.tokens.access_token,
            entity,
        )
        .await
        else {
            continue;
        };
        for item in items {
            session.user_store.apply(EntityEnvelope {
                entity_type: entity.to_string(),
                deleted: false,
                entity: item,
            });
        }
    }
    refresh_engine_history(session);
}

pub(super) fn refresh_engine_history(session: &mut SessionState) {
    let Some(run) = session.engine_run.as_ref() else {
        return;
    };
    let prefix = run.order_prefix.clone();
    let account_id = run.account_id;
    let contract_id = run.contract_id;
    let contract_name = run.contract_name.clone();
    let started_at_ns = run.started_at_utc.timestamp_nanos_opt().unwrap_or(i64::MIN);

    let mut owned_order_ids = run.owned_order_ids.clone();
    if let Some(orders) = session.user_store.orders.get(&account_id) {
        for (order_id, order) in orders {
            if entity_matches_contract(order, contract_id, &contract_name)
                && entity_client_id(order).is_some_and(|value| value.starts_with(&prefix))
            {
                owned_order_ids.insert(*order_id);
            }
        }
    }
    for command in session.user_store.commands.values() {
        if entity_client_id(command).is_some_and(|value| value.starts_with(&prefix)) {
            if let Some(order_id) = json_i64(command, "orderId") {
                owned_order_ids.insert(order_id);
            }
        }
    }

    let owned_strategy_ids = session
        .user_store
        .order_strategies
        .iter()
        .filter_map(|(strategy_id, strategy)| {
            (extract_account_id("orderStrategy", strategy) == Some(account_id)
                && entity_matches_contract(strategy, contract_id, &contract_name)
                && entity_client_id(strategy).is_some_and(|value| value.starts_with(&prefix)))
            .then_some(*strategy_id)
        })
        .collect::<BTreeSet<_>>();
    for link in session.user_store.order_strategy_links.values() {
        if json_i64(link, "orderStrategyId")
            .is_some_and(|strategy_id| owned_strategy_ids.contains(&strategy_id))
        {
            if let Some(order_id) = json_i64(link, "orderId") {
                owned_order_ids.insert(order_id);
            }
        }
    }

    let mut raw_fills = session
        .user_store
        .history_fills
        .values()
        .filter_map(|fill| {
            let order_id = json_i64(fill, "orderId")?;
            let order = session.user_store.find_order(account_id, order_id);
            let contract_matches = entity_matches_contract(fill, contract_id, &contract_name)
                || order.is_some_and(|order| {
                    entity_matches_contract(order, contract_id, &contract_name)
                });
            if !owned_order_ids.contains(&order_id) || !contract_matches {
                return None;
            }
            let ts_ns = json_timestamp_ns(fill, &["timestamp", "fillTime", "createdTime"])?;
            (ts_ns >= started_at_ns).then_some((ts_ns, order_id, fill))
        })
        .collect::<Vec<_>>();
    raw_fills.sort_by_key(|(ts_ns, _, fill)| (*ts_ns, extract_entity_id(fill).unwrap_or_default()));

    let value_per_point = session.market.value_per_point.unwrap_or(0.0);
    let mut position_qty = 0_i32;
    let mut average_entry_price = None;
    let mut realized_pnl = 0.0;
    let mut total_fees = 0.0;
    let mut wins = 0;
    let mut losses = 0;
    let mut fills = Vec::new();

    for (ts_ns, order_id, fill) in raw_fills {
        let Some(fill_id) = extract_entity_id(fill) else {
            continue;
        };
        let order = session.user_store.find_order(account_id, order_id);
        let Some(side) = order
            .and_then(trade_side_from_order)
            .or_else(|| trade_side_from_fill(fill))
        else {
            continue;
        };
        let Some(price) = pick_number(fill, &["price", "fillPrice", "lastPrice", "avgPrice"])
        else {
            continue;
        };
        let qty = pick_number(fill, &["qty", "fillQty", "lastQty", "quantity"])
            .unwrap_or_default()
            .abs()
            .round() as i32;
        if qty <= 0 {
            continue;
        }
        let signed_fill_qty = match side {
            TradeMarkerSide::Buy => qty,
            TradeMarkerSide::Sell => -qty,
        };
        let mut fill_realized = 0.0;
        let mut closed_position = false;
        if position_qty == 0 || position_qty.signum() == signed_fill_qty.signum() {
            let prior_abs = position_qty.abs() as f64;
            let next_abs = prior_abs + qty as f64;
            average_entry_price = Some(match average_entry_price {
                Some(entry) if prior_abs > 0.0 => {
                    (entry * prior_abs + price * qty as f64) / next_abs
                }
                _ => price,
            });
            position_qty += signed_fill_qty;
        } else {
            closed_position = true;
            let entry = average_entry_price.unwrap_or(price);
            let close_qty = position_qty.abs().min(qty) as f64;
            fill_realized =
                (price - entry) * close_qty * position_qty.signum() as f64 * value_per_point;
            let prior_sign = position_qty.signum();
            position_qty += signed_fill_qty;
            if position_qty == 0 {
                average_entry_price = None;
            } else if position_qty.signum() != prior_sign {
                average_entry_price = Some(price);
            }
        }
        let fee = fill_fee_total(&session.user_store, fill_id, fill);
        total_fees += fee;
        // Win/loss describes the trade direction before commissions. Fees
        // belong in net PnL, but must not turn a gross winning exit into a
        // losing trade and distort the F6 win rate.
        let gross_fill_realized = fill_realized;
        fill_realized -= fee;
        realized_pnl += fill_realized;
        if closed_position && gross_fill_realized > 0.005 {
            wins += 1;
        } else if closed_position && gross_fill_realized < -0.005 {
            losses += 1;
        }
        fills.push(EngineHistoryFill {
            fill_id,
            order_id,
            ts_ns,
            side,
            qty,
            price,
            realized_pnl: fill_realized,
        });
    }

    let unrealized_pnl = average_entry_price
        .zip(session.market.bars.last().map(|bar| bar.close))
        .map(|(entry, mark)| (mark - entry) * position_qty as f64 * value_per_point)
        .unwrap_or_default();
    let Some(run) = session.engine_run.as_mut() else {
        return;
    };
    run.history = EngineHistorySnapshot {
        run_id: run.run_id.clone(),
        started_at_utc: run.started_at_utc,
        updated_at_utc: Some(Utc::now()),
        account_id: run.account_id,
        account_name: run.account_name.clone(),
        contract_id: run.contract_id,
        contract_name: run.contract_name.clone(),
        position_qty,
        average_entry_price,
        realized_pnl,
        unrealized_pnl,
        fees: total_fees,
        wins,
        losses,
        fills,
    };
}

pub(super) fn refresh_engine_history_mark(session: &mut SessionState) {
    let Some(run) = session.engine_run.as_mut() else {
        return;
    };
    run.history.unrealized_pnl = run
        .history
        .average_entry_price
        .zip(session.market.bars.last().map(|bar| bar.close))
        .map(|(entry, mark)| {
            (mark - entry)
                * run.history.position_qty as f64
                * session.market.value_per_point.unwrap_or_default()
        })
        .unwrap_or_default();
    run.history.updated_at_utc = Some(Utc::now());
}

fn entity_client_id(entity: &Value) -> Option<&str> {
    entity
        .get("clOrdId")
        .and_then(Value::as_str)
        .or_else(|| entity.get("uuid").and_then(Value::as_str))
        .or_else(|| entity.get("customTag50").and_then(Value::as_str))
}

fn entity_matches_contract(entity: &Value, contract_id: i64, contract_name: &str) -> bool {
    json_i64(entity, "contractId") == Some(contract_id)
        || entity
            .get("symbol")
            .and_then(Value::as_str)
            .or_else(|| entity.get("contractSymbol").and_then(Value::as_str))
            .is_some_and(|symbol| symbol.eq_ignore_ascii_case(contract_name))
}

fn fill_fee_total(store: &UserSyncStore, fill_id: i64, fill: &Value) -> f64 {
    let fill_fee = store
        .fill_fees
        .values()
        .filter(|fee| json_i64(fee, "fillId") == Some(fill_id))
        .filter_map(|fee| {
            pick_number(
                fee,
                &["amount", "fee", "commission", "totalFee", "totalFees"],
            )
        })
        .map(f64::abs)
        .sum::<f64>();
    if fill_fee > f64::EPSILON {
        fill_fee
    } else {
        pick_number(
            fill,
            &["amount", "fee", "commission", "totalFee", "totalFees"],
        )
        .map(f64::abs)
        .filter(|fee| fee.is_finite())
        .unwrap_or_default()
    }
}

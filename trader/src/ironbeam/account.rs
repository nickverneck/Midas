use super::api::fetch_account_refresh;
use super::orders::refresh_managed_protection;
use super::state::{AccountRefresh, AccountState, IronbeamSession, ProtectionKey};
use super::support::{
    account_id_string, contract_symbol, fill_key, normalize_unix_timestamp_ns, order_is_active,
    order_symbol, pick_i64, pick_number, position_entry_price, position_key, position_symbol,
    signed_position_qty, stable_id,
};
use crate::broker::{ContractSuggestion, LatencySnapshot, TradeMarker, TradeMarkerSide};
use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use std::collections::BTreeMap;

const TRADE_MARKER_LIMIT: usize = 200;

pub(super) async fn refresh_account_state(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
) -> Result<()> {
    let refresh = fetch_account_refresh(client, session, latency).await?;
    session.account_state = build_account_state(&refresh);
    refresh_managed_protection(session);
    rebuild_account_snapshots(session);
    session.market.trade_markers = session
        .selected_contract
        .as_ref()
        .map(|contract| selected_trade_markers(&session.account_state, contract))
        .unwrap_or_default();
    Ok(())
}

fn build_account_state(refresh: &AccountRefresh) -> AccountState {
    let mut state = AccountState::default();

    for balance in &refresh.balances {
        if let Some(account_id) = account_id_string(balance) {
            state.balances.insert(account_id, balance.clone());
        }
    }

    for risk in &refresh.risks {
        if let Some(account_id) = account_id_string(risk) {
            state.risks.insert(account_id, risk.clone());
        }
    }

    for item in &refresh.positions {
        let Some(account_id) = account_id_string(item) else {
            continue;
        };
        let positions = item
            .get("positions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let entries = positions
            .into_iter()
            .filter_map(|position| Some((position_key(&position)?, position)))
            .collect::<BTreeMap<_, _>>();
        state.positions.insert(account_id, entries);
    }

    for (account_id, orders) in &refresh.orders {
        let entries = orders
            .iter()
            .filter_map(|order| Some((super::support::order_id_string(order)?, order.clone())))
            .collect::<BTreeMap<_, _>>();
        state.orders.insert(account_id.clone(), entries);
    }

    for (account_id, fills) in &refresh.fills {
        let entries = fills
            .iter()
            .filter_map(|fill| Some((fill_key(fill)?, fill.clone())))
            .collect::<BTreeMap<_, _>>();
        state.fills.insert(account_id.clone(), entries);
    }

    state
}

pub(super) fn rebuild_account_snapshots(session: &mut IronbeamSession) {
    let selected_symbol = session.selected_contract.as_ref().map(contract_symbol);
    let selected_contract_id = session
        .selected_contract
        .as_ref()
        .map(|contract| contract.id);
    session.account_snapshots = session
        .accounts
        .iter()
        .map(|account| {
            let balance = session.account_state.balances.get(&account.name).cloned();
            let risk = session.account_state.risks.get(&account.name).cloned();
            let raw_positions = session
                .account_state
                .positions
                .get(&account.name)
                .map(|positions| positions.values().cloned().collect::<Vec<_>>())
                .unwrap_or_default();

            let selected_position = selected_symbol.and_then(|symbol| {
                raw_positions
                    .iter()
                    .find(|position| position_symbol(position) == Some(symbol))
            });
            let open_position_qty = raw_positions
                .iter()
                .filter_map(signed_position_qty)
                .map(f64::abs)
                .sum::<f64>();
            let unrealized_pnl = selected_position
                .and_then(|position| pick_number(position, &["unrealizedPL", "unrealizedPnl"]))
                .or_else(|| {
                    balance
                        .as_ref()
                        .and_then(|item| pick_number(item, &["openTradeEquity"]))
                });

            let protection = selected_contract_id.and_then(|contract_id| {
                session.managed_protection.get(&ProtectionKey {
                    account_id: account.id,
                    contract_id,
                })
            });

            crate::broker::AccountSnapshot {
                account_id: account.id,
                account_name: account.name.clone(),
                balance: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["totalEquity", "balance"]))
                    .or_else(|| {
                        balance
                            .as_ref()
                            .and_then(|item| pick_number(item, &["netLiquidity"]))
                    })
                    .or_else(|| {
                        risk.as_ref()
                            .and_then(|item| pick_number(item, &["currentNetLiquidationValue"]))
                    }),
                cash_balance: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["cashBalance"])),
                net_liq: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["netLiquidity"]))
                    .or_else(|| {
                        risk.as_ref()
                            .and_then(|item| pick_number(item, &["currentNetLiquidationValue"]))
                    }),
                realized_pnl: balance
                    .as_ref()
                    .and_then(|item| pick_number(item, &["realizedPL", "realizedPnl"])),
                unrealized_pnl,
                intraday_margin: balance
                    .as_ref()
                    .and_then(|item| item.get("marginInfo"))
                    .and_then(|item| {
                        pick_number(item, &["initialTotalMargin", "maintenanceTotalMargin"])
                    }),
                open_position_qty: (open_position_qty > 0.0).then_some(open_position_qty),
                market_position_qty: selected_position.and_then(signed_position_qty),
                market_entry_price: selected_position.and_then(position_entry_price),
                selected_contract_take_profit_price: protection.and_then(|state| {
                    state
                        .take_profit_price
                        .or(state.last_requested_take_profit_price)
                }),
                selected_contract_stop_price: protection
                    .and_then(|state| state.stop_price.or(state.last_requested_stop_price)),
                raw_account: balance.clone(),
                raw_risk: risk.clone(),
                raw_cash: balance,
                raw_positions,
            }
        })
        .collect();
}

pub(super) fn active_orders_for_account<'a>(
    state: &'a AccountState,
    account_name: &str,
) -> Vec<&'a Value> {
    state
        .orders
        .get(account_name)
        .into_iter()
        .flat_map(|orders| orders.values())
        .filter(|order| order_is_active(order))
        .collect()
}

pub(super) fn selected_market_position_qty(session: &IronbeamSession) -> i32 {
    let Some(account_name) = session
        .selected_account_id
        .and_then(|account_id| super::support::account_name_by_id(session, account_id))
    else {
        return 0;
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return 0;
    };
    account_contract_position_qty(session, account_name, contract).round() as i32
}

pub(super) fn selected_market_entry_price(session: &IronbeamSession) -> Option<f64> {
    let account_name = session
        .selected_account_id
        .and_then(|account_id| super::support::account_name_by_id(session, account_id))?;
    let contract = session.selected_contract.as_ref()?;
    let symbol = contract_symbol(contract);
    let mut weighted_sum = 0.0;
    let mut total_qty = 0.0;
    for position in account_positions_for_symbol(&session.account_state, account_name, symbol) {
        let qty = signed_position_qty(position)?.abs();
        if qty <= f64::EPSILON {
            continue;
        }
        let entry_price = position_entry_price(position)?;
        weighted_sum += entry_price * qty;
        total_qty += qty;
    }
    (total_qty > f64::EPSILON).then_some(weighted_sum / total_qty)
}

pub(super) fn account_contract_position_qty(
    session: &IronbeamSession,
    account_name: &str,
    contract: &ContractSuggestion,
) -> f64 {
    account_positions_for_symbol(
        &session.account_state,
        account_name,
        contract_symbol(contract),
    )
    .into_iter()
    .filter_map(signed_position_qty)
    .sum::<f64>()
}

fn account_positions_for_symbol<'a>(
    state: &'a AccountState,
    account_name: &str,
    symbol: &str,
) -> Vec<&'a Value> {
    state
        .positions
        .get(account_name)
        .into_iter()
        .flat_map(|positions| positions.values())
        .filter(|position| position_symbol(position) == Some(symbol))
        .collect()
}

pub(super) fn selected_has_live_entry_path(session: &IronbeamSession) -> bool {
    let Some(account_name) = session
        .selected_account_id
        .and_then(|account_id| super::support::account_name_by_id(session, account_id))
    else {
        return false;
    };
    let Some(contract) = session.selected_contract.as_ref() else {
        return false;
    };
    let protection_ids = if let Some(account_id) = session.selected_account_id {
        let key = ProtectionKey {
            account_id,
            contract_id: contract.id,
        };
        session
            .managed_protection
            .get(&key)
            .map(|protection| {
                [
                    protection.take_profit_order_id.as_deref(),
                    protection.stop_order_id.as_deref(),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    active_orders_for_account(&session.account_state, account_name)
        .into_iter()
        .filter(|order| order_symbol(order) == Some(contract_symbol(contract)))
        .any(|order| {
            let Some(order_id) = super::support::order_id_string(order) else {
                return true;
            };
            !protection_ids.iter().any(|known| *known == order_id)
        })
}

pub(super) fn selected_trade_markers(
    account_state: &AccountState,
    contract: &ContractSuggestion,
) -> Vec<TradeMarker> {
    let symbol = contract_symbol(contract);
    let mut markers = account_state
        .fills
        .iter()
        .flat_map(|(account_name, fills)| {
            fills
                .values()
                .filter(move |fill| order_symbol(fill) == Some(symbol))
                .filter_map(move |fill| trade_marker_from_fill(account_name, contract, fill))
        })
        .collect::<Vec<_>>();
    markers.sort_by_key(|marker| marker.ts_ns);
    if markers.len() > TRADE_MARKER_LIMIT {
        let excess = markers.len() - TRADE_MARKER_LIMIT;
        markers.drain(..excess);
    }
    markers
}

fn trade_marker_from_fill(
    account_name: &str,
    contract: &ContractSuggestion,
    fill: &Value,
) -> Option<TradeMarker> {
    let side = match super::support::order_side(fill)?
        .to_ascii_uppercase()
        .as_str()
    {
        "BUY" => TradeMarkerSide::Buy,
        "SELL" => TradeMarkerSide::Sell,
        _ => return None,
    };
    let ts_ns = pick_i64(fill, &["timeOrderEvent"])
        .and_then(normalize_unix_timestamp_ns)
        .or_else(|| super::support::parse_timestamp_ns(fill.get("fillDate")?))?;
    let price = pick_number(fill, &["fillPrice", "avgFillPrice", "price"])?;
    let qty = pick_number(fill, &["fillQuantity", "quantity"])?.round() as i32;
    let fill_id = fill_key(fill).map(|value| stable_id(&value));
    Some(TradeMarker {
        fill_id: fill_id.filter(|value| *value > 0),
        account_id: Some(stable_id(account_name)),
        contract_id: Some(contract.id),
        contract_name: Some(contract.name.clone()),
        ts_ns,
        price,
        qty: qty.max(1),
        side,
    })
}

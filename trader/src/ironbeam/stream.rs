use super::account::{rebuild_account_snapshots, selected_trade_markers};
use super::orders::refresh_managed_protection;
use super::state::{AccountState, InternalEvent, IronbeamSession};
use super::support::{
    account_id_string, fill_key, order_id_string, parse_timestamp_ns, pick_number, position_key,
    position_update_groups, value_items,
};
use crate::broker::{Bar, ContractSuggestion};
use crate::config::TradingEnvironment;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_BAR_LIMIT: usize = 256;

pub(super) struct StreamEffect {
    pub(super) market_changed: bool,
    pub(super) account_changed: bool,
    pub(super) refresh_recommended: bool,
}

pub(super) async fn run_market_stream(
    env: TradingEnvironment,
    stream_id: String,
    token: String,
    contract: ContractSuggestion,
    internal_tx: UnboundedSender<InternalEvent>,
) {
    let ws_url = format!(
        "{}/stream/{stream_id}?token={token}",
        super::support::ironbeam_ws_base_url(env)
    );

    let Ok((mut socket, _)) = connect_async(&ws_url).await else {
        let _ = internal_tx.send(InternalEvent::StreamError(format!(
            "Failed to open Ironbeam stream for {}.",
            contract.name
        )));
        return;
    };

    let _ = internal_tx.send(InternalEvent::StreamStatus(format!(
        "Ironbeam market stream open for {}.",
        contract.name
    )));

    while let Some(frame) = socket.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let _ = internal_tx.send(InternalEvent::StreamPayload(text.to_string()));
            }
            Ok(Message::Binary(bytes)) => {
                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                    let _ = internal_tx.send(InternalEvent::StreamPayload(text));
                }
            }
            Ok(Message::Close(frame)) => {
                let reason = frame
                    .as_ref()
                    .map(|frame| frame.reason.to_string())
                    .filter(|reason| !reason.trim().is_empty())
                    .unwrap_or_else(|| "no reason provided".to_string());
                let _ = internal_tx.send(InternalEvent::StreamStatus(format!(
                    "Ironbeam market stream closed for {}: {}",
                    contract.name, reason
                )));
                break;
            }
            Ok(Message::Ping(_) | Message::Pong(_)) => {}
            Ok(Message::Frame(_)) => {}
            Err(err) => {
                let _ = internal_tx.send(InternalEvent::StreamError(format!(
                    "Ironbeam market stream error for {}: {}",
                    contract.name, err
                )));
                break;
            }
        }
    }
}

pub(super) fn apply_stream_payload(session: &mut IronbeamSession, parsed: &Value) -> StreamEffect {
    let mut market_changed = false;
    let mut account_changed = false;
    let mut refresh_recommended = false;

    if let Some(items) = parsed.get("ti").and_then(Value::as_array) {
        let incoming = parse_time_bars(items);
        if !incoming.is_empty() {
            let bar_limit = market_bar_limit(session);
            let appended = merge_bars(&mut session.market.bars, incoming, bar_limit);
            if session.market.history_loaded == 0 && session.market.live_bars == 0 {
                session.market.history_loaded = session.market.bars.len();
                session.market.live_bars = 0;
            } else if appended > 0 {
                session.market.live_bars = session.market.live_bars.saturating_add(appended);
            } else {
                session.market.live_bars = session.market.live_bars.saturating_add(1);
            }
            session.market.status = format!(
                "Streaming {} 1-minute bars from Ironbeam",
                session.market.contract_name.as_deref().unwrap_or("market")
            );
            market_changed = true;
        }
    }

    if apply_stream_balances(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_positions(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_risks(&mut session.account_state, parsed) {
        account_changed = true;
    }
    if apply_stream_orders(&mut session.account_state, parsed) {
        account_changed = true;
        refresh_recommended = true;
    }
    if apply_stream_fills(&mut session.account_state, parsed) {
        account_changed = true;
        refresh_recommended = true;
    }

    if account_changed {
        refresh_managed_protection(session);
        rebuild_account_snapshots(session);
        if let Some(contract) = session.selected_contract.as_ref() {
            session.market.trade_markers = selected_trade_markers(&session.account_state, contract);
        }
    }

    StreamEffect {
        market_changed,
        account_changed,
        refresh_recommended,
    }
}

pub(super) fn market_bar_limit(session: &IronbeamSession) -> usize {
    session
        .cfg
        .history_bars
        .saturating_add(512)
        .max(DEFAULT_BAR_LIMIT)
}

fn parse_time_bars(items: &[Value]) -> Vec<Bar> {
    let mut bars = items
        .iter()
        .filter_map(|item| {
            Some(Bar {
                ts_ns: parse_timestamp_ns(item.get("t")?)?,
                open: pick_number(item, &["o", "open"])?,
                high: pick_number(item, &["h", "high"])?,
                low: pick_number(item, &["l", "low"])?,
                close: pick_number(item, &["c", "close"])?,
            })
        })
        .collect::<Vec<_>>();
    bars.sort_by_key(|bar| bar.ts_ns);
    bars
}

fn merge_bars(existing: &mut Vec<Bar>, incoming: Vec<Bar>, bar_limit: usize) -> usize {
    let mut appended = 0;
    for bar in incoming {
        match existing.binary_search_by_key(&bar.ts_ns, |current| current.ts_ns) {
            Ok(index) => existing[index] = bar,
            Err(index) => {
                existing.insert(index, bar);
                appended += 1;
            }
        }
    }
    if existing.len() > bar_limit {
        let excess = existing.len() - bar_limit;
        existing.drain(..excess);
    }
    appended
}

fn apply_stream_balances(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("ba").and_then(Value::as_array) {
        for item in items {
            if let Some(account_id) = account_id_string(item) {
                state.balances.insert(account_id, item.clone());
                changed = true;
            }
        }
    }
    if let Some(item) = parsed.get("b") {
        for balance in value_items(item) {
            if let Some(account_id) = account_id_string(balance) {
                state.balances.insert(account_id, balance.clone());
                changed = true;
            }
        }
    }
    changed
}

fn apply_stream_positions(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(snapshot) = parsed.get("psa") {
        for (account_id, positions, replace) in position_update_groups(snapshot) {
            let entry = state.positions.entry(account_id).or_default();
            if replace {
                entry.clear();
            }
            for position in positions {
                if let Some(key) = position_key(&position) {
                    entry.insert(key, position);
                    changed = true;
                }
            }
        }
    }
    if let Some(update) = parsed.get("ps") {
        for (account_id, positions, _) in position_update_groups(update) {
            let entry = state.positions.entry(account_id).or_default();
            for position in positions {
                if let Some(key) = position_key(&position) {
                    if super::support::signed_position_qty(&position)
                        .map(|qty| qty.abs() <= f64::EPSILON)
                        .unwrap_or(false)
                    {
                        entry.remove(&key);
                    } else {
                        entry.insert(key, position);
                    }
                    changed = true;
                }
            }
        }
    }
    changed
}

fn apply_stream_risks(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("ria").and_then(Value::as_array) {
        for item in items {
            if let Some(account_id) = account_id_string(item) {
                state.risks.insert(account_id, item.clone());
                changed = true;
            }
        }
    }
    if let Some(item) = parsed.get("ri") {
        for risk in value_items(item) {
            if let Some(account_id) = account_id_string(risk) {
                state.risks.insert(account_id, risk.clone());
                changed = true;
            }
        }
    }
    changed
}

fn apply_stream_orders(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("o") {
        for order in value_items(items) {
            let Some(account_id) = account_id_string(order) else {
                continue;
            };
            let Some(order_id) = order_id_string(order) else {
                continue;
            };
            state
                .orders
                .entry(account_id)
                .or_default()
                .insert(order_id, order.clone());
            changed = true;
        }
    }
    changed
}

fn apply_stream_fills(state: &mut AccountState, parsed: &Value) -> bool {
    let mut changed = false;
    if let Some(items) = parsed.get("f") {
        for fill in value_items(items) {
            let Some(account_id) = account_id_string(fill) else {
                continue;
            };
            let Some(key) = fill_key(fill) else {
                continue;
            };
            state
                .fills
                .entry(account_id)
                .or_default()
                .insert(key, fill.clone());
            changed = true;
        }
    }
    changed
}

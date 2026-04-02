use super::account::{
    account_contract_position_qty, active_orders_for_account, rebuild_account_snapshots,
    selected_market_position_qty,
};
use super::api::{
    cancel_multiple_orders, cancel_order, place_protection_order, submit_market_order, update_order,
};
use super::state::{
    IronbeamSession, ManagedProtectionOrders, OrderDispatchOutcome, ProtectionKey,
    ProtectionOrderCandidate,
};
use super::support::{
    account_name_by_id, contract_symbol, is_protection_order_type, order_id_string, order_price,
    order_quantity, order_side, order_symbol, prices_match, sanitize_price,
    schedule_followup_refresh,
};
use crate::broker::{AccountInfo, ContractSuggestion, LatencySnapshot, ManualOrderAction};
use crate::strategy::NativeReversalMode;
use anyhow::{Context, Result};
use reqwest::Client;
use tokio::sync::mpsc::UnboundedSender;

struct OrderContext<'a> {
    account: &'a AccountInfo,
    contract: &'a ContractSuggestion,
}

enum ProtectionSyncAction {
    None,
    Replace {
        take_profit_price: Option<f64>,
        stop_price: Option<f64>,
    },
    Clear,
}

#[derive(Clone, Copy)]
enum ProtectionOrderKind {
    TakeProfit,
    StopLoss,
}

pub(super) async fn dispatch_manual_order(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
    action: ManualOrderAction,
) -> Result<OrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account_name = order_ctx.account.name.clone();
    let contract_name = order_ctx.contract.name.clone();
    let symbol = contract_symbol(order_ctx.contract).to_string();
    let current_qty =
        account_contract_position_qty(session, &account_name, order_ctx.contract).round() as i32;

    match action {
        ManualOrderAction::Buy => {
            if current_qty != 0 {
                cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            }
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                "BUY",
                session.cfg.order_qty.max(1),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued { target_qty: None })
        }
        ManualOrderAction::Sell => {
            if current_qty != 0 {
                cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            }
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                "SELL",
                session.cfg.order_qty.max(1),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued { target_qty: None })
        }
        ManualOrderAction::Close => {
            if current_qty == 0 {
                return Ok(OrderDispatchOutcome::NoOp {
                    message: format!(
                        "Close ignored: no open {} position on {}",
                        contract_name, account_name
                    ),
                });
            }
            cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
            let side = if current_qty > 0 { "SELL" } else { "BUY" };
            submit_market_order(
                client,
                &session.cfg,
                &session.token,
                &account_name,
                &symbol,
                side,
                current_qty.abs(),
                latency,
            )
            .await?;
            schedule_followup_refresh(internal_tx);
            Ok(OrderDispatchOutcome::Queued {
                target_qty: Some(0),
            })
        }
    }
}

pub(super) async fn dispatch_target_position_order(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
    target_qty: i32,
    automated: bool,
    reason: &str,
) -> Result<OrderDispatchOutcome> {
    let order_ctx = resolve_order_context(session)?;
    let account_name = order_ctx.account.name.clone();
    let contract_name = order_ctx.contract.name.clone();
    let symbol = contract_symbol(order_ctx.contract).to_string();
    let current_qty =
        account_contract_position_qty(session, &account_name, order_ctx.contract).round() as i32;
    let delta = target_qty.saturating_sub(current_qty);
    if delta == 0 {
        return Ok(OrderDispatchOutcome::NoOp {
            message: format!(
                "Target already satisfied: {} at {} on {} ({reason})",
                target_qty, contract_name, account_name
            ),
        });
    }

    let is_reversal =
        current_qty != 0 && target_qty != 0 && current_qty.signum() != target_qty.signum();
    if automated
        && is_reversal
        && session.execution_config.native_reversal_mode == NativeReversalMode::CloseAllEnter
    {
        cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
        let flatten_side = if current_qty > 0 { "SELL" } else { "BUY" };
        submit_market_order(
            client,
            &session.cfg,
            &session.token,
            &account_name,
            &symbol,
            flatten_side,
            current_qty.abs(),
            latency,
        )
        .await?;
        let entry_side = if target_qty > 0 { "BUY" } else { "SELL" };
        submit_market_order(
            client,
            &session.cfg,
            &session.token,
            &account_name,
            &symbol,
            entry_side,
            target_qty.abs(),
            latency,
        )
        .await?;
        schedule_followup_refresh(internal_tx);
        return Ok(OrderDispatchOutcome::Queued {
            target_qty: Some(target_qty),
        });
    }

    if automated
        && is_reversal
        && session.execution_config.native_reversal_mode == NativeReversalMode::FlattenConfirmEnter
    {
        cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
        let flatten_side = if current_qty > 0 { "SELL" } else { "BUY" };
        submit_market_order(
            client,
            &session.cfg,
            &session.token,
            &account_name,
            &symbol,
            flatten_side,
            current_qty.abs(),
            latency,
        )
        .await?;
        schedule_followup_refresh(internal_tx);
        session.execution_runtime.pending_reversal_entry =
            Some(super::state::PendingNativeReversalEntry {
                target_qty,
                reason: reason.to_string(),
            });
        return Ok(OrderDispatchOutcome::Queued {
            target_qty: Some(0),
        });
    }

    if current_qty != 0 {
        cancel_selected_protection(client, session, latency, internal_tx.clone()).await?;
    }

    let side = if delta > 0 { "BUY" } else { "SELL" };
    submit_market_order(
        client,
        &session.cfg,
        &session.token,
        &account_name,
        &symbol,
        side,
        delta.abs(),
        latency,
    )
    .await?;
    schedule_followup_refresh(internal_tx);
    Ok(OrderDispatchOutcome::Queued {
        target_qty: Some(target_qty),
    })
}

pub(super) async fn cancel_selected_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
) -> Result<()> {
    let Some(key) = selected_protection_key(session) else {
        return Ok(());
    };
    let Some(account) = session.selected_account_id.and_then(|account_id| {
        session
            .accounts
            .iter()
            .find(|account| account.id == account_id)
    }) else {
        return Ok(());
    };
    let ids = protection_order_ids(session, key);
    if ids.is_empty() {
        return Ok(());
    }
    cancel_multiple_orders(
        client,
        &session.cfg,
        &session.token,
        &account.name,
        &ids,
        latency,
    )
    .await?;
    session.managed_protection.remove(&key);
    schedule_followup_refresh(internal_tx);
    Ok(())
}

pub(super) async fn sync_native_protection(
    client: &Client,
    session: &mut IronbeamSession,
    latency: &mut LatencySnapshot,
    internal_tx: UnboundedSender<super::state::InternalEvent>,
    signed_qty: i32,
    take_profit_price: Option<f64>,
    stop_price: Option<f64>,
    _reason: &str,
) -> Result<()> {
    let Some(key) = selected_protection_key(session) else {
        return Ok(());
    };
    let order_ctx = resolve_order_context(session)?;
    let exit_side = if signed_qty > 0 { "SELL" } else { "BUY" };
    let quantity = signed_qty.abs().max(1);
    let desired_take_profit = sanitize_price(take_profit_price);
    let desired_stop = sanitize_price(stop_price);

    let existing = session.managed_protection.get(&key).cloned();
    let take_profit_candidate = resolve_take_profit_candidate(
        session,
        key,
        &order_ctx.account.name,
        order_ctx.contract,
        exit_side,
    );
    let stop_candidate = resolve_stop_candidate(
        session,
        key,
        &order_ctx.account.name,
        order_ctx.contract,
        exit_side,
    );

    let action = if signed_qty == 0 || (desired_take_profit.is_none() && desired_stop.is_none()) {
        ProtectionSyncAction::Clear
    } else if existing.as_ref().is_some_and(|current| {
        current.signed_qty == signed_qty
            && prices_match(
                current.last_requested_take_profit_price,
                desired_take_profit,
            )
            && prices_match(current.last_requested_stop_price, desired_stop)
            && (desired_take_profit.is_none()
                || current.take_profit_order_id.is_some()
                || take_profit_candidate.is_some())
            && (desired_stop.is_none()
                || current.stop_order_id.is_some()
                || stop_candidate.is_some())
    }) {
        ProtectionSyncAction::None
    } else {
        ProtectionSyncAction::Replace {
            take_profit_price: desired_take_profit,
            stop_price: desired_stop,
        }
    };

    match action {
        ProtectionSyncAction::None => {}
        ProtectionSyncAction::Clear => {
            let ids = protection_order_ids(session, key);
            if !ids.is_empty() {
                cancel_multiple_orders(
                    client,
                    &session.cfg,
                    &session.token,
                    &order_ctx.account.name,
                    &ids,
                    latency,
                )
                .await?;
            }
            session.managed_protection.remove(&key);
            schedule_followup_refresh(internal_tx);
        }
        ProtectionSyncAction::Replace {
            take_profit_price,
            stop_price,
        } => {
            let mut next_state = existing.unwrap_or(ManagedProtectionOrders {
                signed_qty,
                take_profit_price,
                stop_price,
                last_requested_take_profit_price: take_profit_price,
                last_requested_stop_price: stop_price,
                take_profit_order_id: None,
                stop_order_id: None,
            });
            next_state.signed_qty = signed_qty;
            next_state.take_profit_price =
                take_profit_candidate.as_ref().and_then(|item| item.price);
            next_state.stop_price = stop_candidate.as_ref().and_then(|item| item.price);
            next_state.last_requested_take_profit_price = take_profit_price;
            next_state.last_requested_stop_price = stop_price;

            sync_take_profit_order(
                client,
                session,
                latency,
                &order_ctx.account.name,
                order_ctx.contract,
                exit_side,
                quantity,
                take_profit_price,
                take_profit_candidate,
                &mut next_state,
            )
            .await?;
            sync_stop_order(
                client,
                session,
                latency,
                &order_ctx.account.name,
                order_ctx.contract,
                exit_side,
                quantity,
                stop_price,
                stop_candidate,
                &mut next_state,
            )
            .await?;
            session.managed_protection.insert(key, next_state);
            schedule_followup_refresh(internal_tx);
        }
    }

    rebuild_account_snapshots(session);
    Ok(())
}

pub(super) fn refresh_managed_protection(session: &mut IronbeamSession) {
    let Some(contract) = session.selected_contract.as_ref() else {
        session.managed_protection.clear();
        return;
    };
    let symbol = contract_symbol(contract).to_string();
    let contract_id = contract.id;

    for account in &session.accounts {
        let key = ProtectionKey {
            account_id: account.id,
            contract_id,
        };
        let signed_qty =
            account_contract_position_qty(session, &account.name, contract).round() as i32;
        if signed_qty == 0 {
            session.managed_protection.remove(&key);
            continue;
        }

        let exit_side = if signed_qty > 0 { "SELL" } else { "BUY" };
        let take_profit = find_protection_candidate(
            session,
            &account.name,
            &symbol,
            exit_side,
            ProtectionOrderKind::TakeProfit,
            session
                .managed_protection
                .get(&key)
                .and_then(|state| state.take_profit_order_id.as_deref()),
        );
        let stop = find_protection_candidate(
            session,
            &account.name,
            &symbol,
            exit_side,
            ProtectionOrderKind::StopLoss,
            session
                .managed_protection
                .get(&key)
                .and_then(|state| state.stop_order_id.as_deref()),
        );

        let entry = session
            .managed_protection
            .entry(key)
            .or_insert(ManagedProtectionOrders {
                signed_qty,
                take_profit_price: None,
                stop_price: None,
                last_requested_take_profit_price: None,
                last_requested_stop_price: None,
                take_profit_order_id: None,
                stop_order_id: None,
            });
        entry.signed_qty = signed_qty;
        if let Some(candidate) = take_profit {
            entry.take_profit_order_id = Some(candidate.order_id);
            entry.take_profit_price = candidate.price;
        } else {
            entry.take_profit_order_id = None;
            if entry.last_requested_take_profit_price.is_none() {
                entry.take_profit_price = None;
            }
        }
        if let Some(candidate) = stop {
            entry.stop_order_id = Some(candidate.order_id);
            entry.stop_price = candidate.price;
        } else {
            entry.stop_order_id = None;
            if entry.last_requested_stop_price.is_none() {
                entry.stop_price = None;
            }
        }
    }
}

pub(super) fn selected_protection_prices(session: &IronbeamSession) -> (Option<f64>, Option<f64>) {
    let Some(key) = selected_protection_key(session) else {
        return (None, None);
    };
    session
        .managed_protection
        .get(&key)
        .map(|state| {
            (
                state
                    .take_profit_price
                    .or(state.last_requested_take_profit_price),
                state.stop_price.or(state.last_requested_stop_price),
            )
        })
        .unwrap_or((None, None))
}

async fn sync_take_profit_order(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
    quantity: i32,
    desired_price: Option<f64>,
    existing: Option<ProtectionOrderCandidate>,
    state: &mut ManagedProtectionOrders,
) -> Result<()> {
    match (desired_price, existing) {
        (None, Some(existing)) => {
            cancel_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                &existing.order_id,
                latency,
            )
            .await?;
            state.take_profit_order_id = None;
            state.take_profit_price = None;
        }
        (None, None) => {
            state.take_profit_order_id = None;
            state.take_profit_price = None;
        }
        (Some(price), Some(existing)) => {
            if !prices_match(existing.price, Some(price)) || existing.quantity != Some(quantity) {
                update_order(
                    client,
                    &session.cfg,
                    &session.token,
                    account_name,
                    &existing.order_id,
                    quantity,
                    Some(price),
                    None,
                    latency,
                )
                .await?;
            }
            state.take_profit_order_id = Some(existing.order_id);
            state.take_profit_price = Some(price);
        }
        (Some(price), None) => {
            let order_id = place_protection_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                contract_symbol(contract),
                exit_side,
                quantity,
                "LIMIT",
                Some(price),
                None,
                latency,
            )
            .await?;
            state.take_profit_order_id = Some(order_id);
            state.take_profit_price = Some(price);
        }
    }
    Ok(())
}

async fn sync_stop_order(
    client: &Client,
    session: &IronbeamSession,
    latency: &mut LatencySnapshot,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
    quantity: i32,
    desired_price: Option<f64>,
    existing: Option<ProtectionOrderCandidate>,
    state: &mut ManagedProtectionOrders,
) -> Result<()> {
    match (desired_price, existing) {
        (None, Some(existing)) => {
            cancel_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                &existing.order_id,
                latency,
            )
            .await?;
            state.stop_order_id = None;
            state.stop_price = None;
        }
        (None, None) => {
            state.stop_order_id = None;
            state.stop_price = None;
        }
        (Some(price), Some(existing)) => {
            if !prices_match(existing.price, Some(price)) || existing.quantity != Some(quantity) {
                update_order(
                    client,
                    &session.cfg,
                    &session.token,
                    account_name,
                    &existing.order_id,
                    quantity,
                    None,
                    Some(price),
                    latency,
                )
                .await?;
            }
            state.stop_order_id = Some(existing.order_id);
            state.stop_price = Some(price);
        }
        (Some(price), None) => {
            let order_id = place_protection_order(
                client,
                &session.cfg,
                &session.token,
                account_name,
                contract_symbol(contract),
                exit_side,
                quantity,
                "STOP",
                None,
                Some(price),
                latency,
            )
            .await?;
            state.stop_order_id = Some(order_id);
            state.stop_price = Some(price);
        }
    }
    Ok(())
}

fn resolve_order_context(session: &IronbeamSession) -> Result<OrderContext<'_>> {
    let account_id = session
        .selected_account_id
        .context("select an account before sending orders")?;
    let account = session
        .accounts
        .iter()
        .find(|account| account.id == account_id)
        .context("selected account is no longer available")?;
    let contract = session
        .selected_contract
        .as_ref()
        .context("select a contract before sending orders")?;
    Ok(OrderContext { account, contract })
}

fn selected_protection_key(session: &IronbeamSession) -> Option<ProtectionKey> {
    let account_id = session.selected_account_id?;
    let contract_id = session.selected_contract.as_ref()?.id;
    Some(ProtectionKey {
        account_id,
        contract_id,
    })
}

fn protection_order_ids(session: &IronbeamSession, key: ProtectionKey) -> Vec<String> {
    let mut ids = Vec::new();
    if let Some(state) = session.managed_protection.get(&key) {
        if let Some(order_id) = state.take_profit_order_id.clone() {
            ids.push(order_id);
        }
        if let Some(order_id) = state.stop_order_id.clone()
            && !ids.contains(&order_id)
        {
            ids.push(order_id);
        }
    }

    if let Some(account_name) = account_name_by_id(session, key.account_id) {
        let symbol = session
            .selected_contract
            .as_ref()
            .map(contract_symbol)
            .unwrap_or_default();
        let exit_side = if selected_market_position_qty(session) > 0 {
            "SELL"
        } else {
            "BUY"
        };
        for order in active_orders_for_account(&session.account_state, account_name) {
            if order_symbol(order) != Some(symbol) {
                continue;
            }
            if order_side(order) != Some(exit_side) {
                continue;
            }
            if !is_protection_order_type(order) {
                continue;
            }
            if let Some(order_id) = order_id_string(order)
                && !ids.contains(&order_id)
            {
                ids.push(order_id);
            }
        }
    }

    ids
}

fn find_protection_candidate(
    session: &IronbeamSession,
    account_name: &str,
    symbol: &str,
    exit_side: &str,
    kind: ProtectionOrderKind,
    preferred_id: Option<&str>,
) -> Option<ProtectionOrderCandidate> {
    let mut candidates = active_orders_for_account(&session.account_state, account_name)
        .into_iter()
        .filter(|order| order_symbol(order) == Some(symbol))
        .filter(|order| order_side(order) == Some(exit_side))
        .filter_map(|order| {
            let order_type = super::support::normalized_order_type(order)?;
            let matches = match kind {
                ProtectionOrderKind::TakeProfit => order_type == "LIMIT",
                ProtectionOrderKind::StopLoss => order_type == "STOP" || order_type == "STOP_LIMIT",
            };
            if !matches {
                return None;
            }
            Some(ProtectionOrderCandidate {
                order_id: order_id_string(order)?,
                price: order_price(order),
                quantity: order_quantity(order),
            })
        })
        .collect::<Vec<_>>();
    if let Some(preferred_id) = preferred_id
        && let Some(index) = candidates
            .iter()
            .position(|candidate| candidate.order_id == preferred_id)
    {
        return Some(candidates.remove(index));
    }
    candidates.into_iter().next()
}

fn resolve_take_profit_candidate(
    session: &IronbeamSession,
    key: ProtectionKey,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
) -> Option<ProtectionOrderCandidate> {
    find_protection_candidate(
        session,
        account_name,
        contract_symbol(contract),
        exit_side,
        ProtectionOrderKind::TakeProfit,
        session
            .managed_protection
            .get(&key)
            .and_then(|state| state.take_profit_order_id.as_deref()),
    )
}

fn resolve_stop_candidate(
    session: &IronbeamSession,
    key: ProtectionKey,
    account_name: &str,
    contract: &ContractSuggestion,
    exit_side: &str,
) -> Option<ProtectionOrderCandidate> {
    find_protection_candidate(
        session,
        account_name,
        contract_symbol(contract),
        exit_side,
        ProtectionOrderKind::StopLoss,
        session
            .managed_protection
            .get(&key)
            .and_then(|state| state.stop_order_id.as_deref()),
    )
}

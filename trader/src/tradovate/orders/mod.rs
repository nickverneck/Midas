use super::*;

mod api;
mod builders;
mod dispatch;
mod protection;
mod queue;

use builders::{
    build_liquidation_request, build_market_order_request, build_order_strategy_request,
};
#[cfg(test)]
use builders::{price_offset_from_ticks, signed_profit_target_offset, signed_stop_loss_offset};
#[cfg(test)]
use dispatch::dispatch_native_order_strategy_target;
#[cfg(test)]
use protection::plan_native_protection_sync;
use protection::{detach_strategy_protection_for_selected, selected_active_order_strategy_id};
use queue::{
    enqueue_liquidation, enqueue_liquidation_then_order_strategy, enqueue_market_order,
    enqueue_order_strategy,
};

pub(super) use api::{
    cancel_order_by_id, cancel_orders_by_id, interrupt_order_strategy_by_id, request_order_json,
};
pub(super) use builders::native_order_strategy_enabled;
pub(super) use dispatch::{
    MarketOrderDispatchOutcome, dispatch_manual_order,
    dispatch_profile_legacy_order_strategy_target, dispatch_target_position_order,
};
pub(super) use protection::{
    collect_live_protection_orders, recover_live_protection_order,
    refresh_managed_protection_order_ids, selected_strategy_key, sync_native_protection,
    sync_native_protection_target,
};

struct OrderContext<'a> {
    account: &'a AccountInfo,
    contract: &'a ContractSuggestion,
}

fn resolve_order_context<'a>(session: &'a SessionState) -> Result<OrderContext<'a>> {
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

fn ensure_no_market_order_submit_in_flight(session: &SessionState) -> Result<()> {
    if session.order_submit_in_flight {
        bail!("order submission already in flight");
    }
    Ok(())
}

fn next_strategy_cl_ord_id(session: &mut SessionState, suffix: &str) -> String {
    let nonce = session.next_strategy_order_nonce;
    session.next_strategy_order_nonce = session.next_strategy_order_nonce.saturating_add(1);
    let ts = Utc::now().timestamp_millis();
    format!("midas-{ts}-{nonce}-{suffix}")
}

#[cfg(test)]
mod tests;

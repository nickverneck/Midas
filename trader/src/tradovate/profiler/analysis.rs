use super::report::{SwipeAttemptReport, SwipeDelaySummary};
use super::*;
use std::path::PathBuf;

pub(super) fn select_profile_account<'a>(
    accounts: &'a [AccountInfo],
    filter: &str,
) -> Option<&'a AccountInfo> {
    let needle = filter.trim().to_ascii_lowercase();
    accounts
        .iter()
        .find(|account| account.name.to_ascii_lowercase().contains(&needle))
        .or_else(|| {
            accounts
                .iter()
                .find(|account| account.name.to_ascii_lowercase().contains("demo"))
        })
        .or_else(|| accounts.first())
}

pub(super) fn select_profile_contract(
    contracts: &[ContractSuggestion],
    query: &str,
    exact: Option<&str>,
) -> Option<ContractSuggestion> {
    let exact = exact.map(|value| value.trim().to_ascii_uppercase());
    if let Some(exact) = exact.as_deref() {
        if let Some(contract) = contracts
            .iter()
            .find(|contract| contract.name.eq_ignore_ascii_case(exact))
        {
            return Some(contract.clone());
        }
    }

    let query = query.trim().to_ascii_uppercase();
    contracts
        .iter()
        .find(|contract| {
            let name = contract.name.to_ascii_uppercase();
            let description = contract.description.to_ascii_uppercase();
            name.starts_with(&query)
                && !name.starts_with("MES")
                && description.contains("E-MINI S&P")
        })
        .or_else(|| {
            contracts
                .iter()
                .find(|contract| contract.name.to_ascii_uppercase().starts_with(&query))
        })
        .cloned()
}

pub(super) fn profile_output_dir(requested: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = requested {
        fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
        return Ok(path.clone());
    }
    let dir = PathBuf::from(".run")
        .join("swipe-profiles")
        .join(Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

fn protective_orders(probe: &ExecutionProbeSnapshot) -> Vec<&ExecutionProbeOrder> {
    probe
        .selected_working_orders
        .iter()
        .filter(|order| {
            order
                .order_type
                .as_deref()
                .map(|order_type| {
                    matches!(
                        order_type.to_ascii_lowercase().as_str(),
                        "limit"
                            | "mit"
                            | "stop"
                            | "stoplimit"
                            | "trailingstop"
                            | "trailingstoplimit"
                    )
                })
                .unwrap_or(false)
        })
        .collect()
}

fn has_matching_strategy_owned_protection(
    probe: &ExecutionProbeSnapshot,
    expected_abs_qty: i32,
) -> bool {
    let bracket_qtys_match = !probe.broker_strategy_bracket_qtys.is_empty()
        && probe
            .broker_strategy_bracket_qtys
            .iter()
            .all(|qty| *qty == expected_abs_qty);
    bracket_qtys_match
        && (protective_orders(probe).len() >= 2 || probe.linked_active_orders.len() >= 2)
}

pub(super) fn has_any_visible_broker_path(probe: &ExecutionProbeSnapshot) -> bool {
    probe.order_submit_in_flight
        || probe.protection_sync_in_flight
        || probe.tracker_order_is_active
        || probe.tracker_strategy_has_live_orders
        || probe.tracker_within_strategy_grace
        || !probe.selected_working_orders.is_empty()
        || !probe.linked_active_orders.is_empty()
        || probe.managed_protection.is_some()
}

pub(super) fn probe_is_settled(
    probe: &ExecutionProbeSnapshot,
    expected_qty: i32,
    expected_abs_qty: i32,
) -> bool {
    if probe.execution_state.market_position_qty != expected_qty {
        return false;
    }
    if expected_qty == 0 {
        return !has_any_visible_broker_path(probe);
    }

    let protective_orders = protective_orders(probe);
    let has_matching_visible_protection =
        probe.managed_protection.as_ref().is_some_and(|protection| {
            protection.signed_qty.abs() == expected_abs_qty
                && protection.take_profit_price.is_some()
                && protection.stop_price.is_some()
        }) || (protective_orders.len() >= 2
            && protective_orders
                .iter()
                .all(|order| order.order_qty == Some(expected_abs_qty)))
            || has_matching_strategy_owned_protection(probe, expected_abs_qty);

    has_matching_visible_protection
}

pub(super) fn probe_findings(probe: &ExecutionProbeSnapshot, expected_abs_qty: i32) -> Vec<String> {
    let mut findings = Vec::new();
    let actual_qty = probe.execution_state.market_position_qty;
    if actual_qty.abs() > expected_abs_qty {
        findings.push(format!(
            "position {} exceeds configured qty {}",
            actual_qty, expected_abs_qty
        ));
    }

    let protective_orders = protective_orders(probe);
    let mismatched_leg_qtys = protective_orders
        .iter()
        .filter_map(|order| order.order_qty)
        .filter(|qty| *qty != expected_abs_qty)
        .collect::<Vec<_>>();
    if !mismatched_leg_qtys.is_empty() {
        findings.push(format!(
            "visible protection leg qty mismatch: {:?} (expected {})",
            mismatched_leg_qtys, expected_abs_qty
        ));
    }
    let mismatched_strategy_bracket_qtys = probe
        .broker_strategy_bracket_qtys
        .iter()
        .copied()
        .filter(|qty| *qty != expected_abs_qty)
        .collect::<Vec<_>>();
    if !mismatched_strategy_bracket_qtys.is_empty() {
        findings.push(format!(
            "broker strategy bracket qty mismatch: {:?} (expected {})",
            mismatched_strategy_bracket_qtys, expected_abs_qty
        ));
    }
    if protective_orders.len() > 2 {
        findings.push(format!(
            "visible protective order count is {} (expected at most 2)",
            protective_orders.len()
        ));
    }
    if probe.linked_active_orders.len() > 2 {
        findings.push(format!(
            "linked strategy order count is {} (expected at most 2)",
            probe.linked_active_orders.len()
        ));
    }
    if actual_qty != 0
        && probe
            .managed_protection
            .as_ref()
            .is_none_or(|protection| protection.signed_qty.abs() != expected_abs_qty)
        && protective_orders.is_empty()
        && probe.linked_active_orders.is_empty()
        && !has_any_visible_broker_path(probe)
    {
        findings.push("no visible protection on selected contract".to_string());
    }

    findings
}

pub(super) fn summarize_delay_reports(attempts: &[SwipeAttemptReport]) -> SwipeDelaySummary {
    SwipeDelaySummary {
        attempts_total: attempts.len(),
        submit_observed: attempts
            .iter()
            .filter(|attempt| attempt.submit.is_some())
            .count(),
        fill_observed: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .submit
                    .as_ref()
                    .and_then(|submit| submit.fill_ms)
                    .is_some()
            })
            .count(),
        delay_snapshots_with_mismatched_leg_qty: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("leg qty mismatch"))
            })
            .count(),
        delay_snapshots_with_no_visible_protection: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("no visible protection"))
            })
            .count(),
        delay_snapshots_with_oversized_position: attempts
            .iter()
            .filter(|attempt| {
                attempt
                    .delay_findings
                    .iter()
                    .any(|finding| finding.contains("exceeds configured qty"))
            })
            .count(),
        avg_submit_rtt_ms: avg_option_u64(attempts.iter().filter_map(|attempt| {
            attempt
                .submit
                .as_ref()
                .and_then(|submit| submit.broker_submit_ms)
        })),
        avg_seen_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| attempt.submit.as_ref().and_then(|submit| submit.seen_ms)),
        ),
        avg_exec_report_ms: avg_option_u64(attempts.iter().filter_map(|attempt| {
            attempt
                .submit
                .as_ref()
                .and_then(|submit| submit.exec_report_ms)
        })),
        avg_fill_ms: avg_option_u64(
            attempts
                .iter()
                .filter_map(|attempt| attempt.submit.as_ref().and_then(|submit| submit.fill_ms)),
        ),
    }
}

fn avg_option_u64(values: impl Iterator<Item = u64>) -> Option<u64> {
    let mut count = 0u64;
    let mut total = 0u64;
    for value in values {
        total = total.saturating_add(value);
        count = count.saturating_add(1);
    }
    if count == 0 {
        None
    } else {
        Some(total / count)
    }
}

use super::*;
use crate::broker::{ExecutionProbeOrder, LatencySnapshot};
use crate::strategy::ExecutionStateSnapshot;

fn base_probe() -> ExecutionProbeSnapshot {
    let mut execution_state = ExecutionStateSnapshot::default();
    execution_state.market_position_qty = 1;
    ExecutionProbeSnapshot {
        tag: "test".to_string(),
        captured_at_utc: Utc::now(),
        execution_state,
        latency: LatencySnapshot::default(),
        order_submit_in_flight: false,
        protection_sync_in_flight: false,
        tracker_order_id: None,
        tracker_order_is_active: false,
        tracker_order_strategy_id: None,
        tracker_strategy_has_live_orders: false,
        tracker_within_strategy_grace: false,
        tracked_order_strategy_id: None,
        broker_order_strategy_id: Some(77),
        broker_order_strategy_status: Some("Working".to_string()),
        broker_strategy_entry_order_qty: Some(1),
        broker_strategy_bracket_qtys: vec![1],
        selected_working_orders: Vec::new(),
        linked_active_orders: Vec::new(),
        managed_protection: None,
    }
}

#[test]
fn strategy_owned_bracket_with_matching_params_counts_as_settled() {
    let mut probe = base_probe();
    probe.linked_active_orders = vec![
        ExecutionProbeOrder {
            order_id: Some(1001),
            order_strategy_id: Some(77),
            cl_ord_id: Some("midas-tp".to_string()),
            order_type: Some("Limit".to_string()),
            action: Some("Sell".to_string()),
            order_qty: None,
            price: Some(5030.0),
            stop_price: None,
            status: Some("Working".to_string()),
        },
        ExecutionProbeOrder {
            order_id: Some(1002),
            order_strategy_id: Some(77),
            cl_ord_id: Some("midas-sl".to_string()),
            order_type: Some("Stop".to_string()),
            action: Some("Sell".to_string()),
            order_qty: None,
            price: None,
            stop_price: Some(4970.0),
            status: Some("Working".to_string()),
        },
    ];

    assert!(analysis::probe_is_settled(&probe, 1, 1));
}

#[test]
fn merge_probe_order_detail_backfills_qty_from_rest_version() {
    let mut probe = base_probe();
    probe.selected_working_orders = vec![ExecutionProbeOrder {
        order_id: Some(1001),
        order_strategy_id: Some(77),
        cl_ord_id: Some("midas-sl".to_string()),
        order_type: Some("Stop".to_string()),
        action: Some("Sell".to_string()),
        order_qty: None,
        price: None,
        stop_price: Some(4970.0),
        status: Some("Working".to_string()),
    }];

    merge_probe_order_detail(
        &mut probe,
        ExecutionProbeOrder {
            order_id: Some(1001),
            order_strategy_id: Some(77),
            cl_ord_id: Some("midas-sl".to_string()),
            order_type: Some("Stop".to_string()),
            action: Some("Sell".to_string()),
            order_qty: Some(1),
            price: None,
            stop_price: Some(4970.0),
            status: Some("Working".to_string()),
        },
    );

    assert_eq!(probe.selected_working_orders[0].order_qty, Some(1));
}

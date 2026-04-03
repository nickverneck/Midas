use super::*;

#[test]
fn execution_state_snapshot_includes_selected_protection_prices() {
    let mut session = test_session();
    session.market.contract_id = Some(3570918);
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": 1,
                "avgPrice": 6659.75
            }),
        )]),
    );
    session.managed_protection.insert(
        StrategyProtectionKey {
            account_id: 42,
            contract_id: 3570918,
        },
        ManagedProtectionOrders {
            signed_qty: 1,
            take_profit_price: Some(6662.5),
            stop_price: Some(6659.0),
            last_requested_take_profit_price: Some(6662.5),
            last_requested_stop_price: Some(6659.0),
            take_profit_cl_ord_id: None,
            stop_cl_ord_id: None,
            take_profit_order_id: None,
            stop_order_id: None,
        },
    );

    let snapshot = execution_state_snapshot(&session);

    assert_eq!(snapshot.market_entry_price, Some(6659.75));
    assert_eq!(snapshot.selected_contract_take_profit_price, Some(6662.5));
    assert_eq!(snapshot.selected_contract_stop_price, Some(6659.0));
}

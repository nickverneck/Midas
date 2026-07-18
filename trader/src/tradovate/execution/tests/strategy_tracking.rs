use super::*;

#[test]
fn reconcile_keeps_known_strategy_id_while_position_is_open() {
    let mut session = test_session();
    let key = StrategyProtectionKey {
        account_id: 42,
        contract_id: 3570918,
    };
    session.active_order_strategy = Some(TrackedOrderStrategy {
        key,
        order_strategy_id: 77,
        target_qty: -1,
    });
    session.user_store.positions.insert(
        42,
        BTreeMap::from([(
            1,
            json!({
                "id": 1,
                "accountId": 42,
                "contractId": 3570918,
                "netPos": -1
            }),
        )]),
    );

    reconcile_selected_active_order_strategy(&mut session);

    assert_eq!(
        session
            .active_order_strategy
            .as_ref()
            .map(|tracked| tracked.order_strategy_id),
        Some(77)
    );
}

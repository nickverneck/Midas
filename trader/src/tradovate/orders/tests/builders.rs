use super::*;

#[test]
fn price_offset_from_ticks_uses_tick_size() {
    assert_eq!(price_offset_from_ticks(8.0, Some(0.25)), Some(2.0));
    assert_eq!(price_offset_from_ticks(0.0, Some(0.25)), None);
    assert_eq!(price_offset_from_ticks(8.0, Some(0.0)), None);
}

#[test]
fn signed_bracket_offsets_match_order_side() {
    assert_eq!(signed_profit_target_offset("Buy", 2.0), 2.0);
    assert_eq!(signed_profit_target_offset("Sell", 2.0), -2.0);
    assert_eq!(signed_stop_loss_offset("Buy", 2.0), -2.0);
    assert_eq!(signed_stop_loss_offset("Sell", 2.0), 2.0);
}

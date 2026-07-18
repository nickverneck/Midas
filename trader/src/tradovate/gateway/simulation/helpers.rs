use super::*;

pub(super) fn exit_action_for_target(target_qty: i32) -> &'static str {
    if target_qty > 0 { "Sell" } else { "Buy" }
}

pub(super) fn synthetic_ts_ns(reference_ts_ns: Option<i64>) -> i64 {
    reference_ts_ns
        .or_else(|| Utc::now().timestamp_nanos_opt())
        .unwrap_or_default()
}

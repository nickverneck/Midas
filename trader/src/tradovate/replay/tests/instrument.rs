use super::super::instrument::{infer_contract_name, infer_tick_size, infer_value_per_point};
use std::path::Path;

#[test]
fn instrument_inference_preserves_known_contract_specs() {
    assert_eq!(
        infer_contract_name(Path::new("/tmp/GC 12-26.Last.txt")),
        "GC 12-26"
    );
    assert_eq!(infer_tick_size("GC 12-26"), 0.1);
    assert_eq!(infer_value_per_point("GC 12-26"), 100.0);
}

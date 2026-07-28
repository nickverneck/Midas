use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub(super) fn infer_contract_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("Replay");
    stem.trim_end_matches(".Last").to_string()
}

pub(super) fn replay_contract_id(path: &Path) -> i64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    (hasher.finish() & 0x3FFF_FFFF_FFFF_FFFF) as i64
}

pub(super) fn infer_tick_size(contract_name: &str) -> f64 {
    let symbol = contract_name
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    match symbol.as_str() {
        "ES" | "MES" | "NQ" | "MNQ" | "RTY" | "M2K" | "YM" | "MYM" => 0.25,
        "CL" | "MCL" => 0.01,
        "GC" | "MGC" => 0.1,
        _ => 0.25,
    }
}

pub(super) fn infer_value_per_point(contract_name: &str) -> f64 {
    let symbol = contract_name
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    match symbol.as_str() {
        "ES" => 50.0,
        "MES" => 5.0,
        "NQ" => 20.0,
        "MNQ" => 2.0,
        "RTY" => 50.0,
        "M2K" => 5.0,
        "YM" => 5.0,
        "MYM" => 0.5,
        "CL" => 1000.0,
        "MCL" => 100.0,
        "GC" => 100.0,
        "MGC" => 10.0,
        _ => 1.0,
    }
}

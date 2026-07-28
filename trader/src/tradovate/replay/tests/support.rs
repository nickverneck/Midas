use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn dt(raw: &str) -> chrono::DateTime<chrono::Utc> {
    raw.parse().expect("valid timestamp")
}

pub(super) fn temp_cache_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("trader-replay-state-{name}-{nonce}"))
}

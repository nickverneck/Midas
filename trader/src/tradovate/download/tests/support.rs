use chrono::{DateTime, Utc};

pub(super) fn dt(raw: &str) -> DateTime<Utc> {
    raw.parse().expect("valid timestamp")
}

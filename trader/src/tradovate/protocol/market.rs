use super::*;

pub(crate) fn parse_bar(value: &Value) -> Option<Bar> {
    let ts = value.get("timestamp")?.as_str()?;
    let ts_ns = parse_bar_timestamp_ns(ts)?;
    Some(Bar {
        ts_ns,
        open: json_number(value, "open")?,
        high: json_number(value, "high")?,
        low: json_number(value, "low")?,
        close: json_number(value, "close")?,
    })
}

pub(crate) fn parse_bar_timestamp_ns(ts: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|_| {
            chrono::DateTime::parse_from_str(ts, "%Y-%m-%dT%H:%M%:z")
                .map(|dt| dt.with_timezone(&Utc))
        })
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%dT%H:%MZ").map(|dt| dt.and_utc())
        })
        .ok()?
        .timestamp_nanos_opt()
}

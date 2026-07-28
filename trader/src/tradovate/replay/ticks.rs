use super::*;

use crate::replay_cache::ReplayCacheRawTickRow;
use anyhow::Context;
use chrono::{NaiveDate, NaiveTime, TimeZone, Timelike};

#[derive(Debug, Clone, Copy)]
pub(super) struct ReplayTick {
    pub(super) ts_ns: i64,
    pub(super) last: f64,
    pub(super) size: Option<f64>,
}

pub(super) fn replay_tick_from_cache_row(row: &ReplayCacheRawTickRow) -> ReplayTick {
    ReplayTick {
        ts_ns: row.ts_ns,
        last: row.price,
        size: Some(row.size),
    }
}

pub(super) fn parse_tick_line(line: &str) -> Result<ReplayTick> {
    let mut fields = line.split(';');
    let prefix = fields
        .next()
        .context("replay line missing timestamp prefix")?;
    let last = fields
        .next()
        .context("replay line missing last price")?
        .trim()
        .parse::<f64>()
        .context("parse last price")?;

    let mut prefix_parts = prefix.split_whitespace();
    let date_raw = prefix_parts.next().context("replay line missing date")?;
    let time_raw = prefix_parts.next().context("replay line missing time")?;
    let fraction_raw = prefix_parts
        .next()
        .context("replay line missing fractional time")?;

    let date = NaiveDate::parse_from_str(date_raw, "%Y%m%d").context("parse replay date")?;
    let time = NaiveTime::parse_from_str(time_raw, "%H%M%S").context("parse replay time")?;
    let fraction = fraction_raw
        .trim()
        .parse::<u32>()
        .context("parse replay fractional time")?;
    let nanos = fraction
        .checked_mul(100)
        .context("replay fractional time overflowed nanoseconds")?;
    let naive = date
        .and_hms_nano_opt(time.hour(), time.minute(), time.second(), nanos)
        .context("compose replay timestamp")?;
    let ts_ns = New_York
        .from_local_datetime(&naive)
        .single()
        .or_else(|| New_York.from_local_datetime(&naive).earliest())
        .or_else(|| New_York.from_local_datetime(&naive).latest())
        .context("resolve replay timestamp in America/New_York")?
        .with_timezone(&Utc)
        .timestamp_nanos_opt()
        .context("convert replay timestamp to nanoseconds")?;

    Ok(ReplayTick {
        ts_ns,
        last,
        size: None,
    })
}

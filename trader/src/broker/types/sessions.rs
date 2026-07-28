use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InstrumentSessionProfile {
    FuturesGlobex,
    EquityRth,
}

impl InstrumentSessionProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::FuturesGlobex => "Globex",
            Self::EquityRth => "RTH",
        }
    }

    pub fn evaluate_with_blockout(
        self,
        ts_ns: i64,
        blockout_minutes_before_close: f64,
    ) -> InstrumentSessionWindow {
        if ts_ns <= 0 {
            return InstrumentSessionWindow {
                session_open: true,
                minutes_to_close: None,
                hold_entries: false,
            };
        }

        let dt_et = DateTime::<Utc>::from_timestamp_nanos(ts_ns).with_timezone(&New_York);
        let blockout_minutes_before_close = blockout_minutes_before_close.max(0.0);
        match self {
            Self::FuturesGlobex => futures_globex_window(dt_et, blockout_minutes_before_close),
            Self::EquityRth => equity_rth_window(dt_et, blockout_minutes_before_close),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstrumentSessionWindow {
    pub session_open: bool,
    pub minutes_to_close: Option<f64>,
    pub hold_entries: bool,
}

pub fn infer_session_profile(product: &Value) -> InstrumentSessionProfile {
    match product
        .get("productType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "futures" => InstrumentSessionProfile::FuturesGlobex,
        _ => InstrumentSessionProfile::EquityRth,
    }
}

fn futures_globex_window(
    dt_et: DateTime<chrono_tz::Tz>,
    blockout_minutes_before_close: f64,
) -> InstrumentSessionWindow {
    let hour = fractional_hour(&dt_et);
    let session_open = match dt_et.weekday() {
        Weekday::Sun => hour >= 18.0,
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu => hour < 17.0 || hour >= 18.0,
        Weekday::Fri => hour < 17.0,
        Weekday::Sat => false,
    };

    if !session_open {
        return InstrumentSessionWindow {
            session_open: false,
            minutes_to_close: None,
            hold_entries: true,
        };
    }

    let close_date = if dt_et.weekday() == Weekday::Sun || hour >= 18.0 {
        dt_et
            .date_naive()
            .succ_opt()
            .unwrap_or_else(|| dt_et.date_naive())
    } else {
        dt_et.date_naive()
    };
    let close_at = new_york_close(close_date, 17, 0, 0);
    let minutes_to_close = close_at.map(|close| minutes_until(dt_et, close));
    let hold_entries = minutes_to_close
        .map(|minutes| minutes <= blockout_minutes_before_close)
        .unwrap_or(false);

    InstrumentSessionWindow {
        session_open: true,
        minutes_to_close,
        hold_entries,
    }
}

fn equity_rth_window(
    dt_et: DateTime<chrono_tz::Tz>,
    blockout_minutes_before_close: f64,
) -> InstrumentSessionWindow {
    let hour = fractional_hour(&dt_et);
    let session_open = matches!(
        dt_et.weekday(),
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu | Weekday::Fri
    ) && hour >= 9.5
        && hour < 16.0;

    if !session_open {
        return InstrumentSessionWindow {
            session_open: false,
            minutes_to_close: None,
            hold_entries: true,
        };
    }

    let close_at = new_york_close(dt_et.date_naive(), 16, 0, 0);
    let minutes_to_close = close_at.map(|close| minutes_until(dt_et, close));
    let hold_entries = minutes_to_close
        .map(|minutes| minutes <= blockout_minutes_before_close)
        .unwrap_or(false);

    InstrumentSessionWindow {
        session_open: true,
        minutes_to_close,
        hold_entries,
    }
}

fn fractional_hour(dt_et: &DateTime<chrono_tz::Tz>) -> f64 {
    dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0 + dt_et.second() as f64 / 3600.0
}

fn new_york_close(
    date: chrono::NaiveDate,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<DateTime<chrono_tz::Tz>> {
    let naive = date.and_hms_opt(hour, minute, second)?;
    New_York.from_local_datetime(&naive).single()
}

fn minutes_until(start: DateTime<chrono_tz::Tz>, end: DateTime<chrono_tz::Tz>) -> f64 {
    ((end - start).num_seconds() as f64 / 60.0).max(0.0)
}

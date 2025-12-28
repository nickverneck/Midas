use anyhow::Result;
use polars::prelude::{AnyValue, Series, SerReader};

#[derive(Clone)]
pub struct DataSet {
    pub open: Vec<f64>,
    pub close: Vec<f64>,
    pub _high: Vec<f64>,
    pub _low: Vec<f64>,
    pub volume: Option<Vec<f64>>,
    pub datetime_ns: Option<Vec<i64>>,
    pub session_open: Option<Vec<bool>>,
    pub margin_ok: Vec<bool>,
    pub feature_cols: Vec<Vec<f64>>,
    pub obs_dim: usize,
    pub symbol: String,
}

impl DataSet {
    pub fn with_session(mut self, globex: bool) -> Self {
        if let Some(dt) = &self.datetime_ns {
            self.session_open = Some(build_session_mask(dt, globex));
        }
        self
    }
}

pub fn load_dataset(path: &std::path::Path, globex: bool) -> Result<DataSet> {
    let file = std::fs::File::open(path)?;
    let df = polars::prelude::ParquetReader::new(file).finish()?;
    let open = if df.column("open").is_ok() {
        series_to_f64(df.column("open")?.as_materialized_series())?
    } else {
        series_to_f64(df.column("close")?.as_materialized_series())?
    };
    let close = series_to_f64(df.column("close")?.as_materialized_series())?;
    let high = series_to_f64(df.column("high")?.as_materialized_series())?;
    let low = series_to_f64(df.column("low")?.as_materialized_series())?;
    let volume: Option<Vec<f64>> = df
        .column("volume")
        .ok()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .transpose()?;
    let datetime_ns: Option<Vec<i64>> = df
        .column("date")
        .ok()
        .map(|c| series_to_i64(c.as_materialized_series()))
        .transpose()?;
    let symbol = match df.column("symbol")?.get(0)? {
        AnyValue::String(s) => s.to_string(),
        _ => "UNKNOWN".to_string(),
    };

    let feats = midas_env::features::compute_features_ohlcv(
        &close,
        Some(&high),
        Some(&low),
        volume.as_deref(),
    );
    let feature_cols = ordered_feature_cols(feats)?;

    let session_open = datetime_ns.as_ref().map(|dt| build_session_mask(dt, globex));
    let margin_ok = vec![true; close.len()];

    let obs_dim = observation_len(&open, &close, volume.as_deref(), &feature_cols);

    Ok(DataSet {
        open,
        close,
        _high: high,
        _low: low,
        volume,
        datetime_ns,
        session_open,
        margin_ok,
        feature_cols,
        obs_dim,
        symbol,
    })
}

pub fn dump_dataset_stats(label: &str, data: &DataSet) {
    let close = &data.close;
    if close.is_empty() {
        println!("info: {label}: empty dataset");
        return;
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    let mut zero_delta = 0usize;
    let mut total = 0usize;
    for i in 0..close.len() {
        let v = close[i];
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        if i > 0 {
            total += 1;
            if (close[i] - close[i - 1]).abs() < 1e-12 {
                zero_delta += 1;
            }
        }
    }
    println!(
        "info: {label}: close[min={:.6}, max={:.6}], zero_delta={}/{},", 
        min_v, max_v, zero_delta, total
    );
}

pub fn build_observation(data: &DataSet, idx: usize, position: i32, equity: f64) -> Vec<f32> {
    use chrono::Timelike;
    let mut obs = Vec::with_capacity(data.obs_dim);

    if idx < data.open.len() {
        obs.push(data.open[idx]);
    } else {
        obs.push(f64::NAN);
    }

    if idx > 0 {
        obs.push(data.close[idx - 1]);
        if let Some(vol) = data.volume.as_ref() {
            obs.push(vol[idx - 1]);
        }
    } else {
        obs.push(f64::NAN);
        if data.volume.is_some() {
            obs.push(f64::NAN);
        }
    }

    obs.push(equity);

    for col in data.feature_cols.iter() {
        obs.push(*col.get(idx.saturating_sub(1)).unwrap_or(&f64::NAN));
    }

    if let Some(dt) = data.datetime_ns.as_ref().and_then(|d| d.get(idx.saturating_sub(1))) {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*dt);
        let hour = dt.hour() as f64 + dt.minute() as f64 / 60.0;
        let angle = 2.0 * std::f64::consts::PI * (hour / 24.0);
        obs.push(angle.sin());
        obs.push(angle.cos());
    } else {
        obs.push(f64::NAN);
        obs.push(f64::NAN);
    }

    obs.push(position as f64);

    let session_val = data
        .session_open
        .as_ref()
        .and_then(|m| m.get(idx.saturating_sub(1)))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    let margin_val = data
        .margin_ok
        .get(idx.saturating_sub(1))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    obs.push(session_val);
    obs.push(margin_val);

    obs.into_iter()
        .map(|v| if v.is_finite() { v as f32 } else { 0.0 })
        .collect()
}

fn ordered_feature_cols(
    mut feats: std::collections::HashMap<String, Vec<f64>>,
) -> Result<Vec<Vec<f64>>> {
    let mut cols = Vec::new();
    for &p in midas_env::features::periods() {
        cols.push(feats.remove(&format!("sma_{p}")).unwrap());
        cols.push(feats.remove(&format!("ema_{p}")).unwrap());
        cols.push(feats.remove(&format!("hma_{p}")).unwrap());
    }
    for &p in midas_env::features::ATR_PERIODS.iter() {
        cols.push(feats.remove(&format!("atr_{p}")).unwrap());
    }
    Ok(cols)
}

fn observation_len(
    open: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    feature_cols: &[Vec<f64>],
) -> usize {
    let mut len = 0;
    len += 1;
    len += 1;
    if volume.is_some() {
        len += 1;
    }
    len += 1;
    len += feature_cols.len();
    len += 2;
    len += 1;
    len += 2;
    if len == 0 || open.is_empty() || close.is_empty() {
        0
    } else {
        len
    }
}

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Float64(v) => v,
            AnyValue::Float32(v) => v as f64,
            AnyValue::Int64(v) => v as f64,
            AnyValue::Int32(v) => v as f64,
            AnyValue::UInt32(v) => v as f64,
            AnyValue::UInt64(v) => v as f64,
            _ => f64::NAN,
        })
        .collect();
    Ok(out)
}

fn series_to_i64(series: &Series) -> Result<Vec<i64>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Datetime(v, _, _) => v,
            AnyValue::DatetimeOwned(v, _, _) => v,
            AnyValue::Int64(v) => v,
            AnyValue::Int32(v) => v as i64,
            AnyValue::UInt64(v) => v as i64,
            AnyValue::UInt32(v) => v as i64,
            _ => 0_i64,
        })
        .collect();
    Ok(out)
}

fn build_session_mask(datetimes_ns: &[i64], globex: bool) -> Vec<bool> {
    use chrono::Timelike;
    use chrono_tz::America::New_York;

    datetimes_ns
        .iter()
        .map(|ns| {
            let dt_utc = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*ns);
            let dt_et = dt_utc.with_timezone(&New_York);
            let hour = dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0;
            if globex {
                !(hour >= 17.0)
            } else {
                (hour >= 9.5) && (hour <= 16.0)
            }
        })
        .collect()
}

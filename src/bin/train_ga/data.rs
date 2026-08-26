use anyhow::{Context, Result};
use midas_env::bars::{BarInput, BarSelection, PreparedBars, prepare_bars};
use polars::prelude::{AnyValue, DataFrame, SerReader, Series, TimeUnit};

#[derive(Clone)]
pub struct DataSet {
    pub open: Vec<f64>,
    pub close: Vec<f64>,
    pub _high: Vec<f64>,
    pub _low: Vec<f64>,
    pub(crate) signal_open: Vec<f64>,
    pub(crate) signal_close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
    pub datetime_ns: Option<Vec<i64>>,
    pub session_open: Option<Vec<bool>>,
    pub minutes_to_close: Option<Vec<f64>>,
    pub margin_ok: Vec<bool>,
    pub feature_cols: Vec<Vec<f64>>,
    pub obs_dim: usize,
    #[allow(dead_code)]
    pub symbol: String,
    session_from_parquet: bool,
}

impl DataSet {
    pub fn with_session(mut self, globex: bool) -> Self {
        if self.session_from_parquet {
            return self;
        }
        if let Some(dt) = &self.datetime_ns {
            self.session_open = Some(build_session_mask(dt, globex));
            self.minutes_to_close = Some(build_minutes_to_close(dt, globex));
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_test(closes: &[f64]) -> Self {
        assert!(
            closes.len() >= 2,
            "synthetic GA data needs at least two bars"
        );
        let close = closes.to_vec();
        let open = close.clone();
        let high = close.clone();
        let low = close.clone();
        let feature_cols = Vec::new();
        let obs_dim = observation_len(&open, &close, None, &feature_cols);
        Self {
            open,
            close: close.clone(),
            _high: high,
            _low: low,
            signal_open: close.clone(),
            signal_close: close,
            volume: None,
            datetime_ns: None,
            session_open: None,
            minutes_to_close: None,
            margin_ok: vec![true; closes.len()],
            feature_cols,
            obs_dim,
            symbol: "TEST".to_string(),
            session_from_parquet: false,
        }
    }
}

#[allow(dead_code)]
pub fn load_dataset(path: &std::path::Path, globex: bool) -> Result<DataSet> {
    load_dataset_with_bars(path, globex, BarSelection::default())
}

pub fn read_symbol(path: &std::path::Path) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let df = polars::prelude::ParquetReader::new(file).finish()?;
    extract_symbol(&df)
}

pub fn load_dataset_with_bars(
    path: &std::path::Path,
    globex: bool,
    bar_selection: BarSelection,
) -> Result<DataSet> {
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
    let datetime_ns: Option<Vec<i64>> = ["ts_ns", "timestamp_ns", "date", "timestamp"]
        .iter()
        .find_map(|name| df.column(name).ok())
        .map(|column| series_to_i64(column.as_materialized_series()))
        .transpose()?;
    let session_open_from_df: Option<Vec<bool>> = df
        .column("session_open")
        .ok()
        .map(|c| series_to_bool(c.as_materialized_series()))
        .transpose()?;
    let minutes_to_close_from_df: Option<Vec<f64>> = df
        .column("minutes_to_close")
        .ok()
        .map(|c| series_to_f64(c.as_materialized_series()))
        .transpose()?;
    let margin_ok_from_df: Option<Vec<bool>> = df
        .column("margin_ok")
        .ok()
        .map(|c| series_to_bool(c.as_materialized_series()))
        .transpose()?;
    let session_from_parquet = session_open_from_df.is_some() && minutes_to_close_from_df.is_some();
    let symbol = extract_symbol(&df)?;
    let (raw_session_open, raw_minutes_to_close) = if session_from_parquet {
        (session_open_from_df, minutes_to_close_from_df)
    } else {
        (
            datetime_ns
                .as_ref()
                .map(|dt| build_session_mask(dt, globex)),
            datetime_ns
                .as_ref()
                .map(|dt| build_minutes_to_close(dt, globex)),
        )
    };

    let prepared = prepare_bars(
        BarInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
            volume: volume.as_deref(),
            datetime_ns: datetime_ns.as_deref(),
            session_open: raw_session_open.as_deref(),
            minutes_to_close: raw_minutes_to_close.as_deref(),
            margin_ok: margin_ok_from_df.as_deref(),
        },
        bar_selection,
    )?;

    let PreparedBars {
        execution_open,
        execution_high,
        execution_low,
        execution_close,
        signal_open,
        signal_high,
        signal_low,
        signal_close,
        volume,
        datetime_ns,
        session_open: prepared_session_open,
        minutes_to_close: prepared_minutes_to_close,
        margin_ok: prepared_margin_ok,
    } = prepared;

    let feature_cols = if bar_selection == BarSelection::default() {
        if let Some(precomputed) = precomputed_feature_cols(&df)? {
            precomputed
        } else {
            let feats = midas_env::features::compute_features_ohlcv(
                &signal_close,
                Some(&signal_high),
                Some(&signal_low),
                volume.as_deref(),
            );
            ordered_feature_cols(
                feats,
                &signal_open,
                &signal_close,
                &signal_high,
                &signal_low,
                volume.as_deref(),
            )?
        }
    } else {
        let feats = midas_env::features::compute_features_ohlcv(
            &signal_close,
            Some(&signal_high),
            Some(&signal_low),
            volume.as_deref(),
        );
        ordered_feature_cols(
            feats,
            &signal_open,
            &signal_close,
            &signal_high,
            &signal_low,
            volume.as_deref(),
        )?
    };

    let session_open = prepared_session_open.or_else(|| {
        datetime_ns
            .as_ref()
            .map(|dt| build_session_mask(dt, globex))
    });
    let minutes_to_close = prepared_minutes_to_close.or_else(|| {
        datetime_ns
            .as_ref()
            .map(|dt| build_minutes_to_close(dt, globex))
    });
    let margin_ok = prepared_margin_ok.unwrap_or_else(|| vec![true; execution_close.len()]);

    let obs_dim = observation_len(
        &signal_open,
        &signal_close,
        volume.as_deref(),
        &feature_cols,
    );

    Ok(DataSet {
        open: execution_open,
        close: execution_close,
        _high: execution_high,
        _low: execution_low,
        signal_open,
        signal_close,
        volume,
        datetime_ns,
        session_open,
        minutes_to_close,
        margin_ok,
        feature_cols,
        obs_dim,
        symbol,
        session_from_parquet,
    })
}

fn extract_symbol(df: &DataFrame) -> Result<String> {
    let symbol = match df.column("symbol").or_else(|_| df.column("instrument")) {
        Ok(column) => match column.get(0)? {
            AnyValue::String(s) => s.to_string(),
            _ => "UNKNOWN".to_string(),
        },
        Err(_) => "UNKNOWN".to_string(),
    };
    Ok(symbol)
}

/// Read the ordered causal feature registry emitted by the dense training
/// parquet.  A missing registry means this is a legacy OHLCV file and the GA
/// keeps its native feature bank.  Precomputed features are used only for the
/// default price-action/OHLC path; alternate bar transformations must rebuild
/// their features after the transformation.
fn precomputed_feature_cols(df: &DataFrame) -> Result<Option<Vec<Vec<f64>>>> {
    let Some(schema_column) = df.column("schema_version").ok() else {
        return Ok(None);
    };
    let schema = schema_column
        .as_materialized_series()
        .str()?
        .get(0)
        .unwrap_or_default();
    if schema != "training-bar-v1" {
        return Ok(None);
    }
    let feature_schema = df
        .column("feature_schema")
        .context("training bar dataset is missing feature_schema")?
        .as_materialized_series()
        .str()?
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("training bar feature_schema is empty"))?;
    let names = feature_schema
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    if names.is_empty() {
        anyhow::bail!("training bar feature_schema contains no features");
    }
    let forbidden = [
        "label_action",
        "label_name",
        "oracle_value",
        "oracle_position_before",
        "oracle_position_after",
        "action_value_normal",
        "action_value_skip",
        "action_value_invert",
    ];
    if names.iter().any(|name| forbidden.contains(name)) {
        anyhow::bail!("training bar feature_schema contains a label/future column");
    }
    names
        .iter()
        .map(|name| {
            let column = df
                .column(name)
                .with_context(|| format!("training bar feature `{name}` is missing"))?;
            series_to_f64(column.as_materialized_series())
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
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
        "info: {label}: close[min={:.6}, max={:.6}], zero_delta={}/{}, features={}, obs_dim={}",
        min_v,
        max_v,
        zero_delta,
        total,
        data.feature_cols.len(),
        data.obs_dim
    );
}

pub fn build_observation(
    data: &DataSet,
    idx: usize,
    position: i32,
    bars_in_position: usize,
    flat_steps: usize,
    max_position: i32,
    equity: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
    initial_balance: f64,
) -> Vec<f32> {
    use chrono::{Datelike, Timelike};
    use chrono_tz::America::New_York;
    let mut obs = Vec::with_capacity(data.obs_dim);

    let prev_idx = idx.saturating_sub(1);
    let prev_prev_idx = idx.saturating_sub(2);
    let prev_close = data.signal_close.get(prev_idx).copied().unwrap_or(f64::NAN);
    let prev_prev_close = data
        .signal_close
        .get(prev_prev_idx)
        .copied()
        .unwrap_or(f64::NAN);
    let open_t = data.signal_open.get(idx).copied().unwrap_or(f64::NAN);

    obs.push(relative_change(open_t, prev_close));
    obs.push(relative_change(prev_close, prev_prev_close));
    if idx > 0 {
        if let Some(vol) = data.volume.as_ref() {
            let current = vol.get(prev_idx).copied().unwrap_or(f64::NAN);
            let previous = vol.get(prev_prev_idx).copied().unwrap_or(f64::NAN);
            obs.push(log_volume_delta(current, previous));
        }
    } else if data.volume.is_some() {
        obs.push(f64::NAN);
    }

    let denom = if initial_balance.abs() < 1e-8 {
        1.0
    } else {
        initial_balance
    };
    obs.push((equity / denom) - 1.0);
    obs.push(unrealized_pnl / denom);
    obs.push(realized_pnl / denom);

    for col in data.feature_cols.iter() {
        obs.push(*col.get(idx.saturating_sub(1)).unwrap_or(&f64::NAN));
    }

    if let Some(dt) = data.datetime_ns.as_ref().and_then(|d| d.get(prev_idx)) {
        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*dt);
        let dt_et = dt.with_timezone(&New_York);
        let hour = dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0;
        let hour_angle = 2.0 * std::f64::consts::PI * (hour / 24.0);
        let weekday = dt_et.weekday().num_days_from_monday() as f64;
        let weekday_angle = 2.0 * std::f64::consts::PI * (weekday / 7.0);
        obs.push(hour_angle.sin());
        obs.push(hour_angle.cos());
        obs.push(weekday_angle.sin());
        obs.push(weekday_angle.cos());
    } else {
        obs.push(f64::NAN);
        obs.push(f64::NAN);
        obs.push(f64::NAN);
        obs.push(f64::NAN);
    }

    let minutes_to_close = data
        .minutes_to_close
        .as_ref()
        .and_then(|m| m.get(prev_idx))
        .copied()
        .unwrap_or(f64::NAN);
    obs.push(if minutes_to_close.is_finite() {
        (minutes_to_close / (24.0 * 60.0)).clamp(0.0, 1.0)
    } else {
        f64::NAN
    });

    let max_position = if max_position > 0 {
        max_position as f64
    } else {
        1.0
    };
    obs.push(position as f64 / max_position);
    obs.push((bars_in_position as f64 / 390.0).clamp(0.0, 4.0));
    obs.push((flat_steps as f64 / 390.0).clamp(0.0, 4.0));

    let session_val = data
        .session_open
        .as_ref()
        .and_then(|m| m.get(prev_idx))
        .map(|b| if *b { 1.0 } else { 0.0 })
        .unwrap_or(f64::NAN);
    let margin_val = data
        .margin_ok
        .get(prev_idx)
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
    open: &[f64],
    close: &[f64],
    high: &[f64],
    low: &[f64],
    volume: Option<&[f64]>,
) -> Result<Vec<Vec<f64>>> {
    let atr_14 = feats
        .get("atr_14")
        .expect("missing atr_14 for delta features")
        .clone();
    let ema_11 = feats
        .get("ema_11")
        .expect("missing ema_11 for delta features")
        .clone();
    let ema_53 = feats
        .get("ema_53")
        .expect("missing ema_53 for delta features")
        .clone();
    let hma_11 = feats
        .get("hma_11")
        .expect("missing hma_11 for delta features")
        .clone();
    let kama_19 = feats
        .get("kama_19")
        .expect("missing kama_19 for delta features")
        .clone();
    let vwap_dist_20 = feats
        .get(&format!("vwap_dist_{}", midas_env::features::VWAP_PERIOD))
        .expect("missing vwap distance for delta features")
        .clone();
    let rvol_20 = feats
        .get(&format!("rvol_{}", midas_env::features::RVOL_PERIOD))
        .expect("missing rvol for delta features")
        .clone();
    let obv = feats.remove("obv").expect("missing obv feature");
    let obv_impulse = normalized_obv_impulse(&obv, volume);

    let ret_1 = log_return(close, 1);
    let ret_3 = log_return(close, 3);
    let gap_open = gap_open_series(open, close, &atr_14);
    let body_1 = candle_body_series(open, close, &atr_14);
    let upper_wick_1 = upper_wick_series(open, close, high, &atr_14);
    let lower_wick_1 = lower_wick_series(open, close, low, &atr_14);
    let ema_11_slope = normalized_delta(&ema_11, &atr_14);
    let hma_11_slope = normalized_delta(&hma_11, &atr_14);
    let kama_19_slope = normalized_delta(&kama_19, &atr_14);
    let fast_slow_spread = normalized_spread(&ema_11, &ema_53, &atr_14);
    let fast_slow_spread_delta = first_difference(&fast_slow_spread);
    let vwap_dist_delta = first_difference(&vwap_dist_20);
    let rvol_delta = first_difference(&rvol_20);

    let mut cols = Vec::new();
    for &p in midas_env::features::periods() {
        cols.push(normalize_level_feature(
            &feats.remove(&format!("sma_{p}")).unwrap(),
            close,
            &atr_14,
        ));
        cols.push(normalize_level_feature(
            &feats.remove(&format!("ema_{p}")).unwrap(),
            close,
            &atr_14,
        ));
        cols.push(normalize_level_feature(
            &feats.remove(&format!("hma_{p}")).unwrap(),
            close,
            &atr_14,
        ));
        cols.push(normalize_level_feature(
            &feats.remove(&format!("kama_{p}")).unwrap(),
            close,
            &atr_14,
        ));
        cols.push(normalize_level_feature(
            &feats.remove(&format!("alma_{p}")).unwrap(),
            close,
            &atr_14,
        ));
    }
    for &p in midas_env::features::ATR_PERIODS.iter() {
        cols.push(normalize_range_feature(
            &feats.remove(&format!("atr_{p}")).unwrap(),
            close,
        ));
    }
    cols.push(
        feats
            .remove(&format!("rvol_{}", midas_env::features::RVOL_PERIOD))
            .unwrap(),
    );
    cols.push(
        feats
            .remove(&format!("cmf_{}", midas_env::features::CMF_PERIOD))
            .unwrap(),
    );
    cols.push(
        feats
            .remove(&format!("vwap_dist_{}", midas_env::features::VWAP_PERIOD))
            .unwrap(),
    );
    cols.push(obv_impulse);
    cols.push(ret_1);
    cols.push(ret_3);
    cols.push(gap_open);
    cols.push(body_1);
    cols.push(upper_wick_1);
    cols.push(lower_wick_1);
    cols.push(ema_11_slope);
    cols.push(hma_11_slope);
    cols.push(kama_19_slope);
    cols.push(fast_slow_spread);
    cols.push(fast_slow_spread_delta);
    cols.push(vwap_dist_delta);
    cols.push(rvol_delta);

    // The native GA feature bank above is intentionally broad, but it did not
    // contain the specific regime/context geometry used by the supervised
    // EMA-10/30 experiments.  Keep this extension causal and append it rather
    // than changing the existing feature order so older policies remain
    // diagnosable by their recorded observation dimension.
    let context_periods = [5usize, 10, 30, 60, 120, 210, 240];
    let context_lookbacks = [1usize, 3, 5, 10, 20];
    let context_emas: Vec<(usize, Vec<f64>)> = context_periods
        .iter()
        .map(|period| (*period, midas_env::features::ema(close, *period)))
        .collect();

    for (_, ema_values) in &context_emas {
        cols.push(normalize_level_feature(ema_values, close, &atr_14));
        for lookback in context_lookbacks {
            cols.push(normalized_delta_lag(ema_values, &atr_14, lookback));
        }
    }

    let ema_10 = &context_emas[1].1;
    let ema_30 = &context_emas[2].1;
    let ema_60 = &context_emas[3].1;
    let ema_120 = &context_emas[4].1;
    let ema_210 = &context_emas[5].1;
    let ema_240 = &context_emas[6].1;
    let trigger_spread = normalized_spread(ema_10, ema_30, &atr_14);
    let context_spread = normalized_spread(ema_210, ema_240, &atr_14);
    let medium_spread = normalized_spread(ema_30, ema_210, &atr_14);
    let fast_context_spread = normalized_spread(ema_60, ema_120, &atr_14);
    cols.push(trigger_spread.clone());
    cols.push(first_difference(&trigger_spread));
    cols.push(context_spread.clone());
    cols.push(first_difference(&context_spread));
    cols.push(medium_spread);
    cols.push(fast_context_spread);
    cols.push(raw_direction_series(&trigger_spread));

    let (has_previous_cross, bars_since_cross, displacement_since_cross, cross_counts) =
        crossover_history(ema_10, ema_30, close, &atr_14, &[5, 10, 30, 60]);
    cols.push(has_previous_cross);
    cols.push(bars_since_cross);
    cols.push(displacement_since_cross);
    cols.extend(cross_counts);

    let (range_width_short, range_position_short) =
        rolling_range_features(high, low, close, &atr_14, 20);
    let (range_width_long, range_position_long) =
        rolling_range_features(high, low, close, &atr_14, 60);
    cols.push(range_width_short);
    cols.push(range_position_short);
    cols.push(range_width_long);
    cols.push(range_position_long);
    cols.push(adx_series(high, low, close, 14));
    cols.push(efficiency_ratio_series(close, 14));
    Ok(cols)
}

fn observation_len(
    open: &[f64],
    close: &[f64],
    volume: Option<&[f64]>,
    feature_cols: &[Vec<f64>],
) -> usize {
    let mut len = 5; // open gap, prev return, equity ratio, unrealized ratio, realized ratio
    if volume.is_some() {
        len += 1; // log-volume delta
    }
    len += feature_cols.len();
    len += 4; // ET hour and weekday sin/cos
    len += 1; // minutes to close
    len += 1; // normalized position
    len += 1; // bars in position
    len += 1; // flat steps
    len += 2; // session + margin flags
    if len == 0 || open.is_empty() || close.is_empty() {
        0
    } else {
        len
    }
}

fn normalize_level_feature(values: &[f64], close: &[f64], atr: &[f64]) -> Vec<f64> {
    let len = values.len().min(close.len()).min(atr.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let value = values[i];
        let anchor = close[i];
        let scale = atr[i];
        if value.is_finite() && anchor.is_finite() && scale.is_finite() && scale.abs() > 1e-8 {
            out[i] = (value - anchor) / scale;
        } else if value.is_finite() && anchor.is_finite() && anchor.abs() > 1e-8 {
            out[i] = (value - anchor) / anchor;
        }
    }
    out
}

fn normalize_range_feature(values: &[f64], close: &[f64]) -> Vec<f64> {
    let len = values.len().min(close.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let value = values[i];
        let anchor = close[i];
        if value.is_finite() && anchor.is_finite() && anchor.abs() > 1e-8 {
            out[i] = value / anchor;
        }
    }
    out
}

fn normalized_obv_impulse(obv: &[f64], volume: Option<&[f64]>) -> Vec<f64> {
    let Some(volume) = volume else {
        return vec![f64::NAN; obv.len()];
    };
    let len = obv.len().min(volume.len());
    let diff = first_difference(obv);
    let vol_sma = midas_env::features::sma(volume, midas_env::features::RVOL_PERIOD);
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let flow = diff[i];
        let scale = vol_sma[i];
        if flow.is_finite() && scale.is_finite() && scale.abs() > 1e-8 {
            out[i] = flow / scale;
        }
    }
    out
}

fn relative_change(current: f64, previous: f64) -> f64 {
    if current.is_finite() && previous.is_finite() && previous.abs() > 1e-8 {
        (current - previous) / previous
    } else {
        f64::NAN
    }
}

fn log_volume_delta(current: f64, previous: f64) -> f64 {
    if current.is_finite() && previous.is_finite() && current >= 0.0 && previous >= 0.0 {
        (current + 1.0).ln() - (previous + 1.0).ln()
    } else {
        f64::NAN
    }
}

fn safe_div(num: f64, denom: f64) -> f64 {
    if !num.is_finite() || !denom.is_finite() || denom.abs() < 1e-8 {
        f64::NAN
    } else {
        num / denom
    }
}

fn log_return(values: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in lag..values.len() {
        let current = values[i];
        let prior = values[i - lag];
        if current.is_sign_positive() && prior.is_sign_positive() {
            out[i] = (current / prior).ln();
        }
    }
    out
}

fn gap_open_series(open: &[f64], close: &[f64], atr: &[f64]) -> Vec<f64> {
    let len = open.len().min(close.len()).min(atr.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len.saturating_sub(1) {
        out[i] = safe_div(open[i + 1] - close[i], atr[i]);
    }
    out
}

fn candle_body_series(open: &[f64], close: &[f64], atr: &[f64]) -> Vec<f64> {
    let len = open.len().min(close.len()).min(atr.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        out[i] = safe_div(close[i] - open[i], atr[i]);
    }
    out
}

fn upper_wick_series(open: &[f64], close: &[f64], high: &[f64], atr: &[f64]) -> Vec<f64> {
    let len = open.len().min(close.len()).min(high.len()).min(atr.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let candle_top = open[i].max(close[i]);
        out[i] = safe_div(high[i] - candle_top, atr[i]);
    }
    out
}

fn lower_wick_series(open: &[f64], close: &[f64], low: &[f64], atr: &[f64]) -> Vec<f64> {
    let len = open.len().min(close.len()).min(low.len()).min(atr.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        let candle_bottom = open[i].min(close[i]);
        out[i] = safe_div(candle_bottom - low[i], atr[i]);
    }
    out
}

fn normalized_delta(values: &[f64], norm: &[f64]) -> Vec<f64> {
    let len = values.len().min(norm.len());
    let mut out = vec![f64::NAN; len];
    for i in 1..len {
        out[i] = safe_div(values[i] - values[i - 1], norm[i]);
    }
    out
}

fn normalized_spread(fast: &[f64], slow: &[f64], norm: &[f64]) -> Vec<f64> {
    let len = fast.len().min(slow.len()).min(norm.len());
    let mut out = vec![f64::NAN; len];
    for i in 0..len {
        out[i] = safe_div(fast[i] - slow[i], norm[i]);
    }
    out
}

fn first_difference(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 1..values.len() {
        let current = values[i];
        let prior = values[i - 1];
        if current.is_finite() && prior.is_finite() {
            out[i] = current - prior;
        }
    }
    out
}

fn normalized_delta_lag(values: &[f64], norm: &[f64], lag: usize) -> Vec<f64> {
    let len = values.len().min(norm.len());
    let mut out = vec![f64::NAN; len];
    if lag == 0 {
        return out;
    }
    for i in lag..len {
        let current = values[i];
        let prior = values[i - lag];
        let scale = norm[i] * lag as f64;
        if current.is_finite() && prior.is_finite() && scale.is_finite() && scale.abs() > 1e-8 {
            out[i] = (current - prior) / scale;
        }
    }
    out
}

fn raw_direction_series(spread: &[f64]) -> Vec<f64> {
    spread
        .iter()
        .map(|value| {
            if value.is_finite() {
                value.signum()
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn crossover_history(
    fast: &[f64],
    slow: &[f64],
    close: &[f64],
    atr: &[f64],
    lookbacks: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let len = fast.len().min(slow.len()).min(close.len()).min(atr.len());
    let mut has_previous = vec![0.0; len];
    let mut bars_since = vec![0.0; len];
    let mut displacement = vec![0.0; len];
    let mut counts = lookbacks.iter().map(|_| vec![0.0; len]).collect::<Vec<_>>();
    let mut cross_indices = Vec::new();

    for i in 1..len {
        let previous_spread = fast[i - 1] - slow[i - 1];
        let spread = fast[i] - slow[i];
        let crossed = previous_spread.is_finite()
            && spread.is_finite()
            && ((previous_spread <= 0.0 && spread > 0.0)
                || (previous_spread >= 0.0 && spread < 0.0));
        if crossed {
            cross_indices.push(i);
        }

        if let Some(&last_cross) = cross_indices.last() {
            has_previous[i] = 1.0;
            bars_since[i] = (i - last_cross) as f64;
            if atr[i].is_finite() && atr[i].abs() > 1e-8 {
                displacement[i] = (close[i] - close[last_cross]) / atr[i];
            }
        }
        for (slot, lookback) in lookbacks.iter().enumerate() {
            counts[slot][i] = cross_indices
                .iter()
                .rev()
                .take_while(|&&cross| i.saturating_sub(cross) <= *lookback)
                .count() as f64;
        }
    }

    (has_previous, bars_since, displacement, counts)
}

fn rolling_range_features(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    atr: &[f64],
    lookback: usize,
) -> (Vec<f64>, Vec<f64>) {
    let len = high.len().min(low.len()).min(close.len()).min(atr.len());
    let mut width = vec![f64::NAN; len];
    let mut position = vec![f64::NAN; len];
    if lookback == 0 {
        return (width, position);
    }
    for i in 0..len {
        let start = i.saturating_add(1).saturating_sub(lookback);
        let rolling_high = high[start..=i]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let rolling_low = low[start..=i]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f64::INFINITY, f64::min);
        let range = rolling_high - rolling_low;
        if !range.is_finite() || !atr[i].is_finite() || atr[i].abs() <= 1e-8 {
            continue;
        }
        width[i] = range / atr[i];
        position[i] = if range > 1e-8 {
            ((close[i] - rolling_low) / range).clamp(0.0, 1.0)
        } else {
            0.5
        };
    }
    (width, position)
}

fn efficiency_ratio_series(close: &[f64], period: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; close.len()];
    if period == 0 {
        return out;
    }
    for i in period..close.len() {
        let net = (close[i] - close[i - period]).abs();
        let mut path = 0.0;
        for j in (i - period + 1)..=i {
            let current = close[j];
            let prior = close[j - 1];
            if current.is_finite() && prior.is_finite() {
                path += (current - prior).abs();
            }
        }
        if path > 1e-8 && net.is_finite() {
            out[i] = (net / path).clamp(0.0, 1.0);
        }
    }
    out
}

fn adx_series(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = high.len().min(low.len()).min(close.len());
    let mut out = vec![f64::NAN; len];
    if period == 0 || len < period * 2 {
        return out;
    }
    let mut true_ranges = vec![f64::NAN; len];
    for i in 0..len {
        true_ranges[i] = if i == 0 {
            high[i] - low[i]
        } else {
            (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs())
        };
    }
    let mut dx = vec![f64::NAN; len];
    for i in period..len {
        let start = i + 1 - period;
        let mut tr_sum = 0.0;
        let mut plus_sum = 0.0;
        let mut minus_sum = 0.0;
        for j in start..=i {
            tr_sum += true_ranges[j];
            if j == 0 {
                continue;
            }
            let up = high[j] - high[j - 1];
            let down = low[j - 1] - low[j];
            if up > down && up > 0.0 {
                plus_sum += up;
            } else if down > up && down > 0.0 {
                minus_sum += down;
            }
        }
        let denom = plus_sum + minus_sum;
        if tr_sum > 1e-8 && denom > 1e-8 {
            let plus_di = 100.0 * plus_sum / tr_sum;
            let minus_di = 100.0 * minus_sum / tr_sum;
            dx[i] = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di);
        }
    }
    for i in (period * 2 - 1)..len {
        let start = i + 1 - period;
        let values = &dx[start..=i];
        let finite = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite.len() == period {
            out[i] = finite.iter().sum::<f64>() / period as f64;
        }
    }
    out
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
    series
        .iter()
        .enumerate()
        .map(|(index, value)| match value {
            AnyValue::Datetime(value, unit, _) => Ok(datetime_to_ns(value, unit)),
            AnyValue::DatetimeOwned(value, unit, _) => Ok(datetime_to_ns(value, unit)),
            AnyValue::Int64(value) => Ok(value),
            AnyValue::Int32(value) => Ok(value as i64),
            AnyValue::UInt64(value) => i64::try_from(value)
                .with_context(|| format!("timestamp at row {index} does not fit in i64")),
            AnyValue::UInt32(value) => Ok(value as i64),
            other => anyhow::bail!(
                "unsupported timestamp value at row {index}: {other:?}; expected ts_ns/date/timestamp to be integer or datetime"
            ),
        })
        .collect()
}

fn datetime_to_ns(value: i64, unit: TimeUnit) -> i64 {
    match unit {
        TimeUnit::Nanoseconds => value,
        TimeUnit::Microseconds => value.saturating_mul(1_000),
        TimeUnit::Milliseconds => value.saturating_mul(1_000_000),
    }
}

fn series_to_bool(series: &Series) -> Result<Vec<bool>> {
    let out = series
        .iter()
        .map(|v| match v {
            AnyValue::Boolean(v) => v,
            AnyValue::UInt8(v) => v != 0,
            AnyValue::UInt16(v) => v != 0,
            AnyValue::UInt32(v) => v != 0,
            AnyValue::UInt64(v) => v != 0,
            AnyValue::Int8(v) => v != 0,
            AnyValue::Int16(v) => v != 0,
            AnyValue::Int32(v) => v != 0,
            AnyValue::Int64(v) => v != 0,
            AnyValue::Float32(v) => v != 0.0,
            AnyValue::Float64(v) => v != 0.0,
            _ => false,
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
                hour < 17.0 || hour >= 18.0
            } else {
                (hour >= 9.5) && (hour <= 16.0)
            }
        })
        .collect()
}

fn build_minutes_to_close(datetimes_ns: &[i64], globex: bool) -> Vec<f64> {
    use chrono::Timelike;
    use chrono_tz::America::New_York;

    datetimes_ns
        .iter()
        .map(|ns| {
            let dt_utc = chrono::DateTime::<chrono::Utc>::from_timestamp_nanos(*ns);
            let dt_et = dt_utc.with_timezone(&New_York);
            let hour = dt_et.hour() as f64 + dt_et.minute() as f64 / 60.0;
            let minutes = if globex {
                if hour < 17.0 {
                    (17.0 - hour) * 60.0
                } else if hour >= 18.0 {
                    ((24.0 - hour) + 17.0) * 60.0
                } else {
                    0.0
                }
            } else if hour <= 16.0 {
                (16.0 - hour) * 60.0
            } else {
                0.0
            };
            if minutes.is_finite() {
                minutes.max(0.0)
            } else {
                0.0
            }
        })
        .collect()
}

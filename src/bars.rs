use anyhow::{Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
pub enum BarKind {
    #[default]
    #[serde(rename = "price-action")]
    #[value(name = "price-action")]
    PriceAction,
    #[serde(rename = "volume")]
    #[value(name = "volume")]
    Volume,
}

impl fmt::Display for BarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PriceAction => "price-action",
            Self::Volume => "volume",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum, Default)]
pub enum PriceSource {
    #[default]
    #[serde(rename = "ohlc")]
    #[value(name = "ohlc")]
    Ohlc,
    #[serde(rename = "heikin-ashi")]
    #[value(name = "heikin-ashi")]
    HeikinAshi,
}

impl fmt::Display for PriceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ohlc => "ohlc",
            Self::HeikinAshi => "heikin-ashi",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarSelection {
    pub bar_kind: BarKind,
    pub volume_bar_size: Option<f64>,
    pub price_source: PriceSource,
}

impl Default for BarSelection {
    fn default() -> Self {
        Self {
            bar_kind: BarKind::PriceAction,
            volume_bar_size: None,
            price_source: PriceSource::Ohlc,
        }
    }
}

pub struct BarInput<'a> {
    pub open: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub close: &'a [f64],
    pub volume: Option<&'a [f64]>,
    pub datetime_ns: Option<&'a [i64]>,
    pub session_open: Option<&'a [bool]>,
    pub minutes_to_close: Option<&'a [f64]>,
    pub margin_ok: Option<&'a [bool]>,
}

pub struct PreparedBars {
    pub execution_open: Vec<f64>,
    pub execution_high: Vec<f64>,
    pub execution_low: Vec<f64>,
    pub execution_close: Vec<f64>,
    pub signal_open: Vec<f64>,
    pub signal_high: Vec<f64>,
    pub signal_low: Vec<f64>,
    pub signal_close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
    pub datetime_ns: Option<Vec<i64>>,
    pub session_open: Option<Vec<bool>>,
    pub minutes_to_close: Option<Vec<f64>>,
    pub margin_ok: Option<Vec<bool>>,
}

pub fn prepare_bars(input: BarInput<'_>, selection: BarSelection) -> Result<PreparedBars> {
    validate_input(&input)?;

    let execution = match selection.bar_kind {
        BarKind::PriceAction => copy_price_action_bars(&input),
        BarKind::Volume => {
            let target = selection.volume_bar_size.ok_or_else(|| {
                anyhow::anyhow!("--volume-bar-size / volumeBarSize is required for volume bars")
            })?;
            aggregate_volume_bars(&input, target)?
        }
    };

    let (signal_open, signal_high, signal_low, signal_close) = match selection.price_source {
        PriceSource::Ohlc => (
            execution.open.clone(),
            execution.high.clone(),
            execution.low.clone(),
            execution.close.clone(),
        ),
        PriceSource::HeikinAshi => heikin_ashi(
            &execution.open,
            &execution.high,
            &execution.low,
            &execution.close,
        ),
    };

    Ok(PreparedBars {
        execution_open: execution.open,
        execution_high: execution.high,
        execution_low: execution.low,
        execution_close: execution.close,
        signal_open,
        signal_high,
        signal_low,
        signal_close,
        volume: execution.volume,
        datetime_ns: execution.datetime_ns,
        session_open: execution.session_open,
        minutes_to_close: execution.minutes_to_close,
        margin_ok: execution.margin_ok,
    })
}

struct MutableBars {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Option<Vec<f64>>,
    datetime_ns: Option<Vec<i64>>,
    session_open: Option<Vec<bool>>,
    minutes_to_close: Option<Vec<f64>>,
    margin_ok: Option<Vec<bool>>,
}

fn validate_input(input: &BarInput<'_>) -> Result<()> {
    let len = input.close.len();
    if input.open.len() != len || input.high.len() != len || input.low.len() != len {
        bail!("open/high/low/close columns must have the same length");
    }
    validate_optional_len("volume", input.volume, len)?;
    validate_optional_len("date", input.datetime_ns, len)?;
    validate_optional_len("session_open", input.session_open, len)?;
    validate_optional_len("minutes_to_close", input.minutes_to_close, len)?;
    validate_optional_len("margin_ok", input.margin_ok, len)?;
    Ok(())
}

fn validate_optional_len<T>(name: &str, values: Option<&[T]>, expected: usize) -> Result<()> {
    if let Some(values) = values {
        if values.len() != expected {
            bail!(
                "{name} column length {} does not match OHLC length {}",
                values.len(),
                expected
            );
        }
    }
    Ok(())
}

fn copy_price_action_bars(input: &BarInput<'_>) -> MutableBars {
    MutableBars {
        open: input.open.to_vec(),
        high: input.high.to_vec(),
        low: input.low.to_vec(),
        close: input.close.to_vec(),
        volume: input.volume.map(|v| v.to_vec()),
        datetime_ns: input.datetime_ns.map(|v| v.to_vec()),
        session_open: input.session_open.map(|v| v.to_vec()),
        minutes_to_close: input.minutes_to_close.map(|v| v.to_vec()),
        margin_ok: input.margin_ok.map(|v| v.to_vec()),
    }
}

fn aggregate_volume_bars(input: &BarInput<'_>, target: f64) -> Result<MutableBars> {
    if !target.is_finite() || target <= 0.0 {
        bail!("volumeBarSize / --volume-bar-size must be > 0");
    }
    let volume = input
        .volume
        .ok_or_else(|| anyhow::anyhow!("volume bars require a volume column"))?;

    let mut out = MutableBars {
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        volume: Some(Vec::new()),
        datetime_ns: input.datetime_ns.map(|_| Vec::new()),
        session_open: input.session_open.map(|_| Vec::new()),
        minutes_to_close: input.minutes_to_close.map(|_| Vec::new()),
        margin_ok: input.margin_ok.map(|_| Vec::new()),
    };

    let mut bar_open = 0.0;
    let mut bar_high = 0.0;
    let mut bar_low = 0.0;
    let mut bar_close = 0.0;
    let mut bar_volume = 0.0;
    let mut bar_started = false;
    let mut last_idx = 0usize;

    for (idx, &current_volume) in volume.iter().enumerate() {
        if !current_volume.is_finite() || current_volume < 0.0 {
            bail!("volume bars require finite non-negative volume values");
        }

        if bar_started && metadata_boundary(input, last_idx, idx) {
            push_aggregated_bar(
                input, &mut out, last_idx, bar_open, bar_high, bar_low, bar_close, bar_volume,
            );
            bar_started = false;
        }

        if !bar_started {
            bar_open = input.open[idx];
            bar_high = input.high[idx];
            bar_low = input.low[idx];
            bar_volume = 0.0;
            bar_started = true;
        } else {
            bar_high = bar_high.max(input.high[idx]);
            bar_low = bar_low.min(input.low[idx]);
        }

        bar_close = input.close[idx];
        bar_volume += current_volume;
        last_idx = idx;

        if bar_volume >= target {
            push_aggregated_bar(
                input, &mut out, idx, bar_open, bar_high, bar_low, bar_close, bar_volume,
            );
            bar_started = false;
        }
    }

    if bar_started {
        push_aggregated_bar(
            input, &mut out, last_idx, bar_open, bar_high, bar_low, bar_close, bar_volume,
        );
    }

    Ok(out)
}

fn push_aggregated_bar(
    input: &BarInput<'_>,
    out: &mut MutableBars,
    source_idx: usize,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
) {
    out.open.push(open);
    out.high.push(high);
    out.low.push(low);
    out.close.push(close);
    if let Some(values) = out.volume.as_mut() {
        values.push(volume);
    }
    push_optional_value(&mut out.datetime_ns, input.datetime_ns, source_idx);
    push_optional_value(&mut out.session_open, input.session_open, source_idx);
    push_optional_value(
        &mut out.minutes_to_close,
        input.minutes_to_close,
        source_idx,
    );
    push_optional_value(&mut out.margin_ok, input.margin_ok, source_idx);
}

fn push_optional_value<T: Copy>(out: &mut Option<Vec<T>>, source: Option<&[T]>, idx: usize) {
    if let (Some(out), Some(source)) = (out.as_mut(), source) {
        if let Some(value) = source.get(idx).copied() {
            out.push(value);
        }
    }
}

fn metadata_boundary(input: &BarInput<'_>, previous_idx: usize, current_idx: usize) -> bool {
    if optional_value_changed(input.session_open, previous_idx, current_idx) {
        return true;
    }
    if optional_value_changed(input.margin_ok, previous_idx, current_idx) {
        return true;
    }
    if let Some(minutes) = input.minutes_to_close {
        let previous = minutes.get(previous_idx).copied().unwrap_or(f64::NAN);
        let current = minutes.get(current_idx).copied().unwrap_or(f64::NAN);
        if previous.is_finite() && current.is_finite() && current > previous + 1e-9 {
            return true;
        }
    }
    false
}

fn optional_value_changed<T: PartialEq + Copy>(
    values: Option<&[T]>,
    previous_idx: usize,
    current_idx: usize,
) -> bool {
    let Some(values) = values else {
        return false;
    };
    values.get(previous_idx).copied() != values.get(current_idx).copied()
}

fn heikin_ashi(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = close.len();
    let mut ha_open = Vec::with_capacity(len);
    let mut ha_high = Vec::with_capacity(len);
    let mut ha_low = Vec::with_capacity(len);
    let mut ha_close = Vec::with_capacity(len);

    for i in 0..len {
        let close_i = (open[i] + high[i] + low[i] + close[i]) / 4.0;
        let open_i = if i == 0 {
            (open[i] + close[i]) / 2.0
        } else {
            (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        };
        ha_open.push(open_i);
        ha_close.push(close_i);
        ha_high.push(high[i].max(open_i).max(close_i));
        ha_low.push(low[i].min(open_i).min(close_i));
    }

    (ha_open, ha_high, ha_low, ha_close)
}

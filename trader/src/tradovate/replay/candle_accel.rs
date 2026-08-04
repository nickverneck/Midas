//! Optional Candle acceleration for replay indicator batches.
//!
//! Candle is intentionally not part of the default build.  The replay
//! execution path remains the source of truth for fills, protection, account
//! state, and ordered strategy events.  This module is a small, side-effect
//! free accelerator boundary that a sweep planner can use to pre-compute
//! indicator state for many candidates.  It does **not** attempt to run the
//! execution loop on a GPU: order/fill decisions are path-dependent and must
//! remain serialized in market-timestamp order.
//!
//! The current Candle kernel computes only the previous/current EMA values for
//! each requested period.  That is enough for a crossover decision at the end
//! of a candidate window, and avoids pretending that a generic tensor backend
//! can preserve the complete event stream.  The CPU implementation is exact
//! (the same f64 recurrence as `strategies::ema_cross::ema_series`); Candle
//! uses f32 tensors and is therefore explicitly approximate.  A caller that
//! requires byte-for-byte replay parity should request [`ReplayAcceleration::Cpu`].

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Indicator acceleration requested by a sweep.
///
/// `Auto` selects CUDA first, then Metal, when the corresponding feature was
/// compiled and a device can be initialized.  If no optional device is
/// available it falls back to the exact CPU implementation.  Explicit device
/// requests fail closed instead of silently producing a CPU run that an
/// operator might mistake for a GPU benchmark.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayAcceleration {
    Cpu,
    Auto,
    CandleCuda,
    CandleMetal,
}

impl Default for ReplayAcceleration {
    fn default() -> Self {
        Self::Cpu
    }
}

impl ReplayAcceleration {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::CandleCuda => "candle_cuda",
            Self::CandleMetal => "candle_metal",
        }
    }

    pub(crate) fn is_candle(self) -> bool {
        matches!(self, Self::Auto | Self::CandleCuda | Self::CandleMetal)
    }
}

impl fmt::Display for ReplayAcceleration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for ReplayAcceleration {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "native" => Ok(Self::Cpu),
            "auto" | "candle" | "candle_auto" => Ok(Self::Auto),
            "cuda" | "candle_cuda" => Ok(Self::CandleCuda),
            "metal" | "candle_metal" => Ok(Self::CandleMetal),
            other => Err(format!(
                "unknown replay acceleration `{other}`; choose cpu, auto, candle_cuda, or candle_metal"
            )),
        }
    }
}

/// Device selected for an indicator call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayAccelerationDevice {
    Cpu,
    CandleCuda,
    CandleMetal,
}

impl ReplayAccelerationDevice {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CandleCuda => "candle_cuda",
            Self::CandleMetal => "candle_metal",
        }
    }
}

/// Result of probing the requested backend before executing a sweep.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ReplayAccelerationStatus {
    pub(crate) requested: ReplayAcceleration,
    pub(crate) selected: ReplayAccelerationDevice,
    pub(crate) candle_compiled: bool,
    pub(crate) device_available: bool,
    pub(crate) exact: bool,
    pub(crate) reason: String,
}

impl ReplayAccelerationStatus {
    pub(crate) fn cpu(requested: ReplayAcceleration, reason: impl Into<String>) -> Self {
        Self {
            requested,
            selected: ReplayAccelerationDevice::Cpu,
            candle_compiled: cfg!(feature = "candle"),
            device_available: true,
            exact: true,
            reason: reason.into(),
        }
    }
}

/// Previous/current EMA pair for one period.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayEmaLastPair {
    pub(crate) period: usize,
    pub(crate) previous: Option<f64>,
    pub(crate) current: Option<f64>,
}

/// Output from [`ema_last_batch`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ReplayEmaBatchResult {
    pub(crate) values: Vec<ReplayEmaLastPair>,
    pub(crate) backend: ReplayAccelerationDevice,
    /// `true` only for the native f64 implementation.  Candle currently uses
    /// f32 tensors for portable CUDA/Metal support, so its values are not
    /// suitable for byte-for-byte parity assertions.
    pub(crate) exact: bool,
    pub(crate) note: String,
}

/// Probe a backend without allocating a tensor or opening a broker connection.
pub(crate) fn probe_acceleration(requested: ReplayAcceleration) -> ReplayAccelerationStatus {
    if requested == ReplayAcceleration::Cpu {
        return ReplayAccelerationStatus::cpu(requested, "native f64 EMA evaluator selected");
    }

    #[cfg(feature = "candle")]
    {
        match select_candle_device(requested) {
            Ok((device, selected)) if !device.is_cpu() => ReplayAccelerationStatus {
                requested,
                selected,
                candle_compiled: true,
                device_available: true,
                exact: false,
                reason: format!("Candle {} device initialized", selected.label()),
            },
            Ok((_device, _selected)) => ReplayAccelerationStatus {
                requested,
                selected: ReplayAccelerationDevice::Cpu,
                candle_compiled: true,
                device_available: false,
                exact: true,
                reason:
                    "Candle compiled but no requested accelerator is available; using native CPU"
                        .to_string(),
            },
            Err(error) => ReplayAccelerationStatus {
                requested,
                selected: ReplayAccelerationDevice::Cpu,
                candle_compiled: true,
                device_available: false,
                exact: true,
                reason: error.to_string(),
            },
        }
    }

    #[cfg(not(feature = "candle"))]
    {
        ReplayAccelerationStatus {
            requested,
            selected: ReplayAccelerationDevice::Cpu,
            candle_compiled: false,
            device_available: false,
            exact: true,
            reason: format!(
                "Candle acceleration is not compiled; rebuild with `--features replay,candle` (or candle-cuda/candle-metal)"
            ),
        }
    }
}

/// Compute previous/current EMA values for several periods in one pass.
///
/// The CPU path is intentionally the reference implementation.  The Candle
/// path keeps all candidate states on the selected device while walking market
/// time and transfers only the final two rows back to the host.  This avoids
/// per-bar host/device synchronization, but it still has a sequential time
/// dependency: it is useful for many candidate periods, not for parallelizing
/// one period over time.  Fills and account transitions are outside this API.
pub(crate) fn ema_last_batch(
    values: &[f64],
    periods: &[usize],
    requested: ReplayAcceleration,
) -> Result<ReplayEmaBatchResult> {
    validate_ema_inputs(values, periods)?;
    if periods.is_empty() {
        return Ok(ReplayEmaBatchResult {
            values: Vec::new(),
            backend: ReplayAccelerationDevice::Cpu,
            exact: true,
            note: "no periods requested".to_string(),
        });
    }
    if values.is_empty() {
        return Ok(cpu_ema_last_batch(values, periods));
    }

    if requested == ReplayAcceleration::Cpu {
        return Ok(cpu_ema_last_batch(values, periods));
    }

    // Non-finite bars are legal in the generic indicator API and the native
    // EMA intentionally skips them.  The Candle tensor expression below does
    // not carry that branch without synchronizing each row, so preserve
    // semantics by falling back to the exact path for this input.
    if values.iter().any(|value| !value.is_finite()) {
        let mut output = cpu_ema_last_batch(values, periods);
        output.note = format!(
            "{}; input contained non-finite values, so the exact CPU path was used",
            output.note
        );
        return Ok(output);
    }

    #[cfg(feature = "candle")]
    {
        let (device, selected) = select_candle_device(requested)?;
        if device.is_cpu() {
            return Ok(cpu_ema_last_batch(values, periods));
        }
        return candle_ema_last_batch(values, periods, &device, selected);
    }

    #[cfg(not(feature = "candle"))]
    {
        // Explicit requests fail closed so the caller cannot accidentally
        // report a CPU benchmark as a GPU benchmark. `Auto` is intentionally
        // useful in a default build and falls back to the reference evaluator.
        if requested == ReplayAcceleration::Auto {
            let mut output = cpu_ema_last_batch(values, periods);
            output.note = "Candle not compiled; auto request fell back to native CPU".to_string();
            return Ok(output);
        }
        bail!(
            "{} requested but Candle is not compiled; rebuild with `--features replay,candle-{}`",
            requested,
            match requested {
                ReplayAcceleration::CandleCuda => "cuda",
                ReplayAcceleration::CandleMetal => "metal",
                ReplayAcceleration::Cpu | ReplayAcceleration::Auto => unreachable!(),
            }
        );
    }
}

fn validate_ema_inputs(values: &[f64], periods: &[usize]) -> Result<()> {
    if periods.iter().any(|period| *period == 0) {
        bail!("EMA periods must be greater than zero");
    }
    if values.is_empty() {
        return Ok(());
    }
    if values.iter().any(|value| value.is_nan()) {
        // NaN is handled by the exact fallback above, but infinities are not
        // valid input to the native recurrence either.  Keep this explicit so
        // future Candle kernels do not silently turn an invalid bar into data.
        return Ok(());
    }
    Ok(())
}

fn cpu_ema_last_batch(values: &[f64], periods: &[usize]) -> ReplayEmaBatchResult {
    let output = periods
        .iter()
        .copied()
        .map(|period| {
            let mut previous = None;
            let mut current = None;
            let Some(first) = values.first().copied() else {
                return ReplayEmaLastPair {
                    period,
                    previous,
                    current,
                };
            };
            let alpha = 2.0 / (period as f64 + 1.0);
            // Match strategies::ema_series exactly: a non-finite input bar
            // leaves the output at that index non-finite, but the recursive
            // state is retained for the next finite bar.  We only expose the
            // final two indexed outputs here, so skipping a NaN would report
            // the wrong previous/current pair.
            let mut state = first;
            if values.len() == 1 {
                current = state.is_finite().then_some(state);
            } else {
                if values.len() == 2 {
                    previous = state.is_finite().then_some(state);
                }
                for (index, value) in values.iter().copied().enumerate().skip(1) {
                    if value.is_finite() {
                        state = alpha * value + (1.0 - alpha) * state;
                    }
                    let output_at_index = value
                        .is_finite()
                        .then_some(state)
                        .filter(|value| value.is_finite());
                    if index + 1 == values.len() {
                        current = output_at_index;
                    } else if index + 2 == values.len() {
                        previous = output_at_index;
                    }
                }
            }
            ReplayEmaLastPair {
                period,
                previous,
                current,
            }
        })
        .collect();
    ReplayEmaBatchResult {
        values: output,
        backend: ReplayAccelerationDevice::Cpu,
        exact: true,
        note: "native f64 EMA recurrence".to_string(),
    }
}

#[cfg(feature = "candle")]
fn select_candle_device(
    requested: ReplayAcceleration,
) -> Result<(candle_core::Device, ReplayAccelerationDevice)> {
    use candle_core::Device;

    match requested {
        ReplayAcceleration::Cpu => Ok((Device::Cpu, ReplayAccelerationDevice::Cpu)),
        ReplayAcceleration::Auto => {
            #[cfg(feature = "candle-cuda")]
            if let Ok(device) = Device::new_cuda(0) {
                return Ok((device, ReplayAccelerationDevice::CandleCuda));
            }
            #[cfg(feature = "candle-metal")]
            if let Ok(device) = Device::new_metal(0) {
                return Ok((device, ReplayAccelerationDevice::CandleMetal));
            }
            Ok((Device::Cpu, ReplayAccelerationDevice::Cpu))
        }
        ReplayAcceleration::CandleCuda => {
            #[cfg(feature = "candle-cuda")]
            {
                let device = Device::new_cuda(0)
                    .map_err(|error| anyhow::anyhow!("initialize Candle CUDA device 0: {error}"))?;
                return Ok((device, ReplayAccelerationDevice::CandleCuda));
            }
            #[cfg(not(feature = "candle-cuda"))]
            bail!("Candle CUDA support is not compiled; use --features candle-cuda")
        }
        ReplayAcceleration::CandleMetal => {
            #[cfg(feature = "candle-metal")]
            {
                let device = Device::new_metal(0).map_err(|error| {
                    anyhow::anyhow!("initialize Candle Metal device 0: {error}")
                })?;
                return Ok((device, ReplayAccelerationDevice::CandleMetal));
            }
            #[cfg(not(feature = "candle-metal"))]
            bail!("Candle Metal support is not compiled; use --features candle-metal")
        }
    }
}

#[cfg(feature = "candle")]
fn candle_ema_last_batch(
    values: &[f64],
    periods: &[usize],
    device: &candle_core::Device,
    selected: ReplayAccelerationDevice,
) -> Result<ReplayEmaBatchResult> {
    use candle_core::Tensor;

    // Candle f32 is used on both CUDA and Metal. Keeping all candidate state
    // on-device avoids a device-to-host synchronization per input bar; each
    // scalar close is still uploaded as a tensor in this portable recurrence,
    // so this path is a correctness/profiling boundary rather than a promise
    // of speed for small candidate sets. A custom scan kernel can batch that
    // upload in a future accelerator implementation.
    let values_f32 = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let initial_state = periods.iter().map(|_| values_f32[0]).collect::<Vec<_>>();
    let alpha_values = periods
        .iter()
        .map(|period| 2.0f32 / (*period as f32 + 1.0))
        .collect::<Vec<_>>();
    let one_minus_alpha_values = periods
        .iter()
        .map(|period| 1.0f32 - 2.0f32 / (*period as f32 + 1.0))
        .collect::<Vec<_>>();
    let mut state = Tensor::new(initial_state.as_slice(), device)?;
    let alpha = Tensor::new(alpha_values.as_slice(), device)?;
    let one_minus_alpha = Tensor::new(one_minus_alpha_values.as_slice(), device)?;
    let mut previous = state.clone();
    for value in values_f32.iter().copied().skip(1) {
        previous = state;
        let value = Tensor::new(&[value], device)?.broadcast_as(periods.len())?;
        state = previous
            .broadcast_mul(&one_minus_alpha)?
            .broadcast_add(&value.broadcast_mul(&alpha)?)?;
    }

    // Ensure asynchronous CUDA/Metal work has completed before converting the
    // final tensors to host vectors. `synchronize` is a no-op on CPU.
    device.synchronize()?;
    let previous = previous.to_vec1::<f32>()?;
    let current = state.to_vec1::<f32>()?;
    let values = periods
        .iter()
        .enumerate()
        .map(|(index, period)| ReplayEmaLastPair {
            period: *period,
            previous: if values.len() > 1 {
                Some(previous[index] as f64)
            } else {
                None
            },
            current: Some(current[index] as f64),
        })
        .collect();
    Ok(ReplayEmaBatchResult {
        values,
        backend: selected,
        exact: false,
        note: "Candle f32 recurrence; indicator values may differ slightly from native f64; keep ordered fills on CPU"
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_acceleration_names() {
        assert_eq!(
            "cpu".parse::<ReplayAcceleration>(),
            Ok(ReplayAcceleration::Cpu)
        );
        assert_eq!(
            "candle".parse::<ReplayAcceleration>(),
            Ok(ReplayAcceleration::Auto)
        );
        assert_eq!(
            "cuda".parse::<ReplayAcceleration>(),
            Ok(ReplayAcceleration::CandleCuda)
        );
        assert!("vulkan".parse::<ReplayAcceleration>().is_err());
    }

    #[test]
    fn native_batch_matches_reference_recurrence() {
        let input = [1.0, 2.0, 5.0, 4.0, 8.0];
        let result = ema_last_batch(&input, &[2, 3], ReplayAcceleration::Cpu).unwrap();
        assert!(result.exact);
        assert_eq!(result.backend, ReplayAccelerationDevice::Cpu);
        let fast = result.values[0];
        let slow = result.values[1];
        assert_eq!(fast.period, 2);
        assert!((fast.current.unwrap() - 6.654_320_987_654_321).abs() < 1e-12);
        assert!((slow.current.unwrap() - 5.8125).abs() < 1e-12);
        assert!(fast.previous.unwrap() < fast.current.unwrap());
        assert!(slow.previous.unwrap() < slow.current.unwrap());
    }

    #[test]
    fn non_finite_values_use_exact_cpu_fallback() {
        let result = ema_last_batch(&[1.0, f64::NAN, 3.0], &[2], ReplayAcceleration::Auto).unwrap();
        assert!(result.exact);
        assert_eq!(result.backend, ReplayAccelerationDevice::Cpu);
        assert!(result.values[0].previous.is_none());
        assert!((result.values[0].current.unwrap() - 2.333_333_333_333_333).abs() < 1e-12);

        let trailing =
            ema_last_batch(&[1.0, 2.0, f64::NAN], &[2], ReplayAcceleration::Cpu).unwrap();
        assert!(trailing.values[0].previous.is_some());
        assert!(trailing.values[0].current.is_none());

        let first = ema_last_batch(&[f64::NAN, 2.0, 3.0], &[2], ReplayAcceleration::Cpu).unwrap();
        assert!(first.values[0].previous.is_none());
        assert!(first.values[0].current.is_none());
    }

    #[test]
    fn empty_input_is_a_valid_empty_indicator_window() {
        let result = ema_last_batch(&[], &[2, 10], ReplayAcceleration::Auto).unwrap();
        assert!(result.exact);
        assert_eq!(result.values.len(), 2);
        assert!(
            result
                .values
                .iter()
                .all(|value| { value.previous.is_none() && value.current.is_none() })
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_tensor_path_matches_with_f32_tolerance() {
        let device = candle_core::Device::Cpu;
        let result = candle_ema_last_batch(
            &[1.0, 2.0, 5.0, 4.0, 8.0],
            &[2, 3],
            &device,
            ReplayAccelerationDevice::CandleCuda,
        )
        .unwrap();
        assert!(!result.exact);
        assert_eq!(result.backend, ReplayAccelerationDevice::CandleCuda);
        let fast = result.values[0].current.unwrap();
        let slow = result.values[1].current.unwrap();
        assert!((fast - 6.654_321).abs() < 1e-5);
        assert!((slow - 5.8125).abs() < 1e-5);
    }

    #[cfg(not(feature = "candle"))]
    #[test]
    fn auto_falls_back_without_optional_feature() {
        let result = ema_last_batch(&[1.0, 2.0], &[2], ReplayAcceleration::Auto).unwrap();
        assert!(result.exact);
        assert!(result.note.contains("not compiled"));
        assert!(!probe_acceleration(ReplayAcceleration::Auto).candle_compiled);
    }
}

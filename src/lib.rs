//! Library entry for optional Python bindings.

pub mod env;
pub mod backtesting;
pub mod features;
pub mod sampler;

#[cfg(feature = "python")]
mod py_bindings {
    #![allow(unsafe_op_in_unsafe_fn)]

    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use numpy::{PyArray1, PyReadonlyArray1};

    use crate::env::{Action, EnvConfig, StepContext, TradingEnv};
    use crate::features::{compute_features_ohlcv, periods};
    use crate::env::build_observation;
    use crate::sampler::windows;

    #[pyclass]
    struct PyTradingEnv {
        inner: TradingEnv,
    }

    #[pymethods]
    impl PyTradingEnv {
        #[new]
        #[pyo3(signature = (initial_price, initial_balance=10000.0, max_position=1, commission_round_turn=1.60, slippage_per_contract=0.25, margin_per_contract=50.0, enforce_margin=true, risk_penalty=0.0, idle_penalty=0.0))]
        fn new(
            initial_price: f64,
            initial_balance: f64,
            max_position: i32,
            commission_round_turn: f64,
            slippage_per_contract: f64,
            margin_per_contract: f64,
            enforce_margin: bool,
            risk_penalty: f64,
            idle_penalty: f64,
        ) -> Self {
            let cfg = EnvConfig {
                commission_round_turn,
                slippage_per_contract,
                max_position,
                margin_per_contract,
                enforce_margin,
                default_session_open: true,
                risk_penalty,
                idle_penalty,
            };
            Self {
                inner: TradingEnv::new(initial_price, initial_balance, cfg),
            }
        }

        /// Reset environment state to a fresh price.
        #[pyo3(signature = (price, initial_balance=10000.0))]
        fn reset(&mut self, price: f64, initial_balance: f64) {
            self.inner.reset(price, initial_balance);
        }

        /// Step the environment.
        #[pyo3(signature = (action, next_price, session_open=true, margin_ok=true))]
        fn step(
            &mut self,
            action: &str,
            next_price: f64,
            session_open: bool,
            margin_ok: bool,
        ) -> PyResult<(f64, PyObject)> {
            let action = match action.to_ascii_lowercase().as_str() {
                "buy" => Action::Buy,
                "sell" => Action::Sell,
                "hold" => Action::Hold,
                "revert" | "flip" => Action::Revert,
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown action: {other}"))),
            };

            let (reward, info) = self.inner.step(
                action,
                next_price,
                StepContext {
                    session_open,
                    margin_ok,
                },
            );

            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                let s = self.inner.state();
                dict.set_item("commission_paid", info.commission_paid)?;
                dict.set_item("slippage_paid", info.slippage_paid)?;
                dict.set_item("pnl_change", info.pnl_change)?;
                dict.set_item("realized_pnl_change", info.realized_pnl_change)?;
                dict.set_item("drawdown_penalty", info.drawdown_penalty)?;
                dict.set_item("margin_call_violation", info.margin_call_violation)?;
                dict.set_item("position_limit_violation", info.position_limit_violation)?;
                dict.set_item("session_closed_violation", info.session_closed_violation)?;
                dict.set_item("unrealized_pnl", s.unrealized_pnl)?;
                dict.set_item("realized_pnl", s.realized_pnl)?;
                dict.set_item("position", s.position)?;
                dict.set_item("cash", s.cash)?;
                dict.set_item("last_price", s.last_price)?;
                dict.set_item("step", s.step)?;
                Ok((reward, dict.into()))
            })
        }

        /// Batch step: actions & prices arrays must align (len prices == len actions).
        #[pyo3(signature = (actions, prices, session_open=true, margin_ok=true))]
        fn step_batch(
            &mut self,
            actions: Vec<String>,
            prices: PyReadonlyArray1<'_, f64>,
            session_open: bool,
            margin_ok: bool,
        ) -> PyResult<(Vec<f64>, Vec<PyObject>)> {
            let prices = prices.as_slice()?;
            if prices.len() != actions.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "prices and actions must have same length",
                ));
            }
            let mut rewards = Vec::with_capacity(actions.len());
            let mut infos: Vec<PyObject> = Vec::with_capacity(actions.len());

            for (a, &p) in actions.iter().zip(prices.iter()) {
                let action = match a.to_ascii_lowercase().as_str() {
                    "buy" => Action::Buy,
                    "sell" => Action::Sell,
                    "hold" => Action::Hold,
                    "revert" | "flip" => Action::Revert,
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown action: {other}"
                        )))
                    }
                };

                let (reward, info) = self.inner.step(
                    action,
                    p,
                    StepContext {
                        session_open,
                        margin_ok,
                    },
                );
                rewards.push(reward);
                Python::with_gil(|py| {
                    let d = PyDict::new(py);
                    let s = self.inner.state();
                    d.set_item("commission_paid", info.commission_paid)?;
                    d.set_item("slippage_paid", info.slippage_paid)?;
                    d.set_item("pnl_change", info.pnl_change)?;
                    d.set_item("realized_pnl_change", info.realized_pnl_change)?;
                    d.set_item("drawdown_penalty", info.drawdown_penalty)?;
                    d.set_item("margin_call_violation", info.margin_call_violation)?;
                    d.set_item("position_limit_violation", info.position_limit_violation)?;
                    d.set_item("session_closed_violation", info.session_closed_violation)?;
                    d.set_item("unrealized_pnl", s.unrealized_pnl)?;
                    d.set_item("realized_pnl", s.realized_pnl)?;
                    d.set_item("position", s.position)?;
                    d.set_item("cash", s.cash)?;
                    d.set_item("last_price", s.last_price)?;
                    d.set_item("step", s.step)?;
                    infos.push(d.into());
                    Ok::<(), PyErr>(())
                })?;
            }

            Ok((rewards, infos))
        }
    }

    /// Compute feature columns (SMA/EMA/HMA over configured periods) from a 1-D float array of prices.
    #[pyfunction]
    #[pyo3(signature = (close, high=None, low=None, volume=None))]
    fn compute_features_py<'py>(
        py: Python<'py>,
        close: PyReadonlyArray1<'_, f64>,
        high: Option<PyReadonlyArray1<'_, f64>>,
        low: Option<PyReadonlyArray1<'_, f64>>,
        volume: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let close_slice = close.as_slice()?;

        let volume_owned: Option<Vec<f64>> = if let Some(v) = volume {
            Some(v.as_slice()?.to_vec())
        } else {
            None
        };
        let volume_slice = volume_owned.as_deref();

        let high_owned: Option<Vec<f64>> = if let Some(h) = high {
            Some(h.as_slice()?.to_vec())
        } else {
            None
        };
        let low_owned: Option<Vec<f64>> = if let Some(l) = low {
            Some(l.as_slice()?.to_vec())
        } else {
            None
        };

        let feats = compute_features_ohlcv(
            close_slice,
            high_owned.as_deref(),
            low_owned.as_deref(),
            volume_slice,
        );
        let dict = PyDict::new(py);
        for (k, v) in feats {
            let arr = PyArray1::from_vec(py, v);
            dict.set_item(k, arr)?;
        }
        Ok(dict)
    }

    /// Return the fixed indicator periods.
    #[pyfunction]
    fn indicator_periods() -> Vec<usize> {
        periods().to_vec()
    }

    /// Build observation for index `idx` using prior data (t-1) and position.
    #[pyfunction]
    #[pyo3(signature = (idx, close, high, low, open=None, volume=None, datetime_ns=None, session_open=None, margin_ok=None, position=0, equity=10000.0))]
    fn build_observation_py<'py>(
        py: Python<'py>,
        idx: usize,
        close: PyReadonlyArray1<'_, f64>,
        high: PyReadonlyArray1<'_, f64>,
        low: PyReadonlyArray1<'_, f64>,
        open: Option<PyReadonlyArray1<'_, f64>>,
        volume: Option<PyReadonlyArray1<'_, f64>>,
        datetime_ns: Option<PyReadonlyArray1<'_, i64>>,
        session_open: Option<PyReadonlyArray1<'_, bool>>,
        margin_ok: Option<PyReadonlyArray1<'_, bool>>,
        position: i32,
        equity: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let close_s = close.as_slice()?;
        let high_s = high.as_slice()?;
        let low_s = low.as_slice()?;
        let open_s: Option<Vec<f64>> = open.map(|o| o.as_slice().unwrap().to_vec());
        let vol_s: Option<Vec<f64>> = volume.map(|v| v.as_slice().unwrap().to_vec());
        let dt_s: Option<Vec<i64>> = datetime_ns.map(|d| d.as_slice().unwrap().to_vec());
        let sess_s: Option<Vec<bool>> = session_open.map(|s| s.as_slice().unwrap().to_vec());
        let margin_s: Option<Vec<bool>> = margin_ok.map(|s| s.as_slice().unwrap().to_vec());

        let obs = build_observation(
            idx,
            open_s.as_deref(),
            close_s,
            high_s,
            low_s,
            vol_s.as_deref(),
            dt_s.as_deref(),
            sess_s.as_deref(),
            margin_s.as_deref(),
            position,
            equity,
        );
        Ok(PyArray1::from_vec(py, obs))
    }

    /// Return chronological windows [start, end) over length `len`.
    #[pyfunction]
    fn list_windows(len: usize, window: usize, step: usize) -> Vec<(usize, usize)> {
        windows(len, window, step)
    }

    /// Python module definition.
    #[pymodule]
    fn midas_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyTradingEnv>()?;
        m.add_function(wrap_pyfunction!(compute_features_py, m)?)?;
        m.add_function(wrap_pyfunction!(indicator_periods, m)?)?;
        m.add_function(wrap_pyfunction!(build_observation_py, m)?)?;
        m.add_function(wrap_pyfunction!(list_windows, m)?)?;
        Ok(())
    }
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FillModelMode {
    Fixed,
    RandomAdverse,
}

impl Default for FillModelMode {
    fn default() -> Self {
        Self::Fixed
    }
}

impl FromStr for FillModelMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "fixed" | "off" => Ok(Self::Fixed),
            "random-adverse" | "random" | "stress" => Ok(Self::RandomAdverse),
            other => Err(format!(
                "unknown fill model {other:?}; expected fixed or random-adverse"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FillModelConfig {
    #[serde(default)]
    pub mode: FillModelMode,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_max_adverse_ticks")]
    pub max_adverse_ticks: u32,
    #[serde(default = "default_tick_value_usd")]
    pub tick_value_usd: f64,
}

impl Default for FillModelConfig {
    fn default() -> Self {
        Self {
            mode: FillModelMode::Fixed,
            seed: 0,
            max_adverse_ticks: default_max_adverse_ticks(),
            tick_value_usd: default_tick_value_usd(),
        }
    }
}

impl FillModelConfig {
    pub fn is_random_adverse(self) -> bool {
        matches!(self.mode, FillModelMode::RandomAdverse)
    }

    pub fn rng(self) -> Option<StdRng> {
        self.is_random_adverse()
            .then(|| StdRng::seed_from_u64(self.seed))
    }

    pub fn validate(self) -> Result<(), String> {
        if !self.tick_value_usd.is_finite() || self.tick_value_usd < 0.0 {
            return Err("fill tick value USD must be finite and >= 0".to_string());
        }
        Ok(())
    }

    pub fn sample_adverse_slippage(self, rng: &mut StdRng, contracts: u32) -> f64 {
        if !self.is_random_adverse() || contracts == 0 || self.tick_value_usd == 0.0 {
            return 0.0;
        }
        let ticks = rng.gen_range(0..=self.max_adverse_ticks);
        ticks as f64 * self.tick_value_usd * contracts as f64
    }

    pub fn with_seed(self, seed: u64) -> Self {
        Self { seed, ..self }
    }
}

pub fn derive_seed(base_seed: u64, components: &[u64]) -> u64 {
    let mut acc = splitmix64(base_seed);
    for &component in components {
        acc = splitmix64(acc ^ component);
    }
    acc
}

fn default_max_adverse_ticks() -> u32 {
    2
}

fn default_tick_value_usd() -> f64 {
    1.25
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

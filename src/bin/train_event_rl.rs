//! Causal two-action PPO/GRPO diagnostic for supervised crossover events.
//!
//! This runner deliberately uses the prepared event state and derives the
//! training reward from the event's own decision/interval prices. It never
//! reads action_value_* or label columns as observations. The environment is
//! a contextual normal/invert bandit: each crossover is one decision with an
//! isolated fixed-horizon reward. It is intentionally separate from the
//! native four-action bar-by-bar Trader environment.

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use midas_env::supervised::{
    LabelMode, SUPERVISED_DATASET_SCHEMA, SUPERVISED_LABEL_SCHEMA, SupervisedConfig,
};
use polars::prelude::{DataFrame, DataType, ParquetReader, SerReader};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Algorithm {
    Ppo,
    Grpo,
    Rlvr,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RewardNormalization {
    None,
    TrainMeanAbs,
}

#[derive(Debug, Parser)]
#[command(
    name = "train_event_rl",
    about = "Train causal two-action PPO/GRPO/RLVR normal-invert gates on supervised event states"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    outdir: PathBuf,
    #[arg(long, value_enum, default_value = "ppo")]
    algorithm: Algorithm,
    #[arg(long, default_value_t = 1000)]
    epochs: usize,
    #[arg(long, default_value_t = 4)]
    ppo_epochs: usize,
    #[arg(long, default_value_t = 4)]
    grpo_epochs: usize,
    #[arg(long, default_value_t = 8)]
    group_size: usize,
    #[arg(long, default_value_t = 32)]
    hidden: usize,
    #[arg(long, default_value_t = 1)]
    layers: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f64,
    #[arg(long, default_value_t = 1.0e-4)]
    l2: f64,
    #[arg(long, default_value_t = 0.2)]
    clip: f64,
    #[arg(long, default_value_t = 0.5)]
    value_coefficient: f64,
    #[arg(long, default_value_t = 0.01)]
    entropy_coefficient: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.6)]
    train_fraction: f64,
    #[arg(long, default_value_t = 0.2)]
    validation_fraction: f64,
    #[arg(long, default_value_t = 30)]
    purge_bars: usize,
    #[arg(long, value_enum, default_value = "train-mean-abs")]
    reward_normalization: RewardNormalization,
    #[arg(long, default_value_t = 250)]
    checkpoint_every: usize,
}

#[derive(Debug, Clone)]
struct Row {
    row_idx: usize,
    horizon_row_idx: usize,
    session_id: String,
    raw_features: Vec<f64>,
    rewards: [f64; 2],
}

#[derive(Debug, Clone)]
struct Dataset {
    feature_names: Vec<String>,
    rows: Vec<Row>,
    config_json: String,
    source_hash: String,
    fingerprint: String,
}

#[derive(Debug, Clone)]
struct Sample {
    features: Vec<f64>,
    rewards: [f64; 2],
}

#[derive(Debug, Clone)]
struct Splits {
    train: Range<usize>,
    validation: Range<usize>,
    holdout: Range<usize>,
    purge_bars: usize,
    sessions: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    event_count: usize,
    normal_count: usize,
    invert_count: usize,
    normal_fraction: f64,
    invert_fraction: f64,
    sum_pnl_usd: f64,
    mean_pnl_usd: f64,
    oracle_pnl_usd: f64,
    oracle_capture: f64,
    positive_event_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Checkpoint {
    epoch: usize,
    train: Summary,
    validation: Summary,
    policy_loss: f64,
    value_loss: Option<f64>,
    entropy: f64,
    approx_kl: f64,
    clip_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SplitReport {
    total: usize,
    train: usize,
    validation: usize,
    holdout: usize,
    purged: usize,
    sessions: usize,
    purge_bars: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Metrics {
    schema_version: &'static str,
    warning: &'static str,
    algorithm: &'static str,
    model_semantics: &'static str,
    input: String,
    seed: u64,
    feature_names: Vec<String>,
    config_json: String,
    source_hash_sha256: String,
    dataset_fingerprint_sha256: String,
    reward_normalization: &'static str,
    reward_scale_usd: f64,
    reward_definition: &'static str,
    training: TrainingMetadata,
    checkpoint_selection: &'static str,
    split: SplitReport,
    selected_epoch: usize,
    selected_train: Summary,
    selected_validation: Summary,
    fixed_normal: Summary,
    fixed_invert: Summary,
    learned: Summary,
    checkpoints: Vec<Checkpoint>,
    leakage_check: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct TrainingMetadata {
    epochs: usize,
    ppo_epochs: usize,
    grpo_epochs: usize,
    group_size: usize,
    hidden: usize,
    layers: usize,
    learning_rate: f64,
    l2: f64,
    clip: f64,
    value_coefficient: f64,
    entropy_coefficient: f64,
    train_fraction: f64,
    validation_fraction: f64,
    purge_bars: usize,
    checkpoint_every: usize,
}

const WARNING: &str =
    "causal event-level normal/invert PPO/GRPO/RLVR diagnostic; not sequential Trader replay";
const REWARD_DEFINITION: &str = "isolated event reward: raw_direction * (interval_end_price - decision_price) * contract_multiplier - round_trip_cost";

#[derive(Debug, Clone)]
struct Layer {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Mlp {
    layers: Vec<Layer>,
    input_dim: usize,
    output_dim: usize,
    hidden_layers: usize,
}

#[derive(Debug, Clone)]
struct Cache {
    inputs: Vec<Vec<f64>>,
    hidden_pre: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct Grad {
    weights: Vec<Vec<Vec<f64>>>,
    bias: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct Adam {
    step: u64,
    mw: Vec<Vec<Vec<f64>>>,
    vw: Vec<Vec<Vec<f64>>>,
    mb: Vec<Vec<f64>>,
    vb: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
struct PpoItem {
    features: Vec<f64>,
    action: usize,
    old_logp: f64,
    reward: f64,
    advantage: f64,
}

#[derive(Debug, Clone)]
struct GrpoItem {
    features: Vec<f64>,
    action: usize,
    old_logp: f64,
    advantage: f64,
}

#[derive(Debug, Clone)]
struct TrainResult {
    policy: Mlp,
    selected_epoch: usize,
    checkpoints: Vec<Checkpoint>,
}

impl Mlp {
    fn random(
        input_dim: usize,
        hidden: usize,
        hidden_layers: usize,
        output_dim: usize,
        rng: &mut StdRng,
    ) -> Self {
        let mut layers = Vec::with_capacity(hidden_layers + 1);
        let mut input = input_dim;
        for layer_index in 0..=hidden_layers {
            let output = if layer_index < hidden_layers {
                hidden
            } else {
                output_dim
            };
            let scale = (2.0 / input.max(1) as f64).sqrt();
            let weights = (0..output)
                .map(|_| {
                    (0..input)
                        .map(|_| rng.gen_range(-scale..scale))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            layers.push(Layer {
                weights,
                bias: vec![0.0; output],
            });
            input = hidden;
        }
        Self {
            layers,
            input_dim,
            output_dim,
            hidden_layers,
        }
    }

    fn forward(&self, input: &[f64]) -> (Vec<f64>, Cache) {
        debug_assert_eq!(input.len(), self.input_dim);
        let mut current = input.to_vec();
        let mut inputs = vec![current.clone()];
        let mut hidden_pre = Vec::with_capacity(self.hidden_layers);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut output = vec![0.0; layer.bias.len()];
            for (out_index, value) in output.iter_mut().enumerate() {
                *value = layer.bias[out_index]
                    + layer.weights[out_index]
                        .iter()
                        .zip(&current)
                        .map(|(weight, feature)| weight * feature)
                        .sum::<f64>();
            }
            if layer_index < self.hidden_layers {
                hidden_pre.push(output.clone());
                current = output.into_iter().map(f64::tanh).collect();
                inputs.push(current.clone());
            } else {
                return (output, Cache { inputs, hidden_pre });
            }
        }
        unreachable!("MLP always has an output layer")
    }

    fn zero_grad(&self) -> Grad {
        Grad {
            weights: self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .weights
                        .iter()
                        .map(|row| vec![0.0; row.len()])
                        .collect()
                })
                .collect(),
            bias: self
                .layers
                .iter()
                .map(|layer| vec![0.0; layer.bias.len()])
                .collect(),
        }
    }

    fn accumulate_backward(&self, cache: &Cache, output_grad: &[f64], grad: &mut Grad) {
        let mut delta = output_grad.to_vec();
        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let input = &cache.inputs[layer_index];
            for (out_index, delta_value) in delta.iter().enumerate() {
                grad.bias[layer_index][out_index] += *delta_value;
                for (input_index, feature) in input.iter().enumerate() {
                    grad.weights[layer_index][out_index][input_index] += delta_value * feature;
                }
            }
            if layer_index > 0 {
                let mut previous = vec![0.0; layer.weights[0].len()];
                for input_index in 0..previous.len() {
                    let propagated = delta
                        .iter()
                        .enumerate()
                        .map(|(out_index, value)| value * layer.weights[out_index][input_index])
                        .sum::<f64>();
                    let pre = cache.hidden_pre[layer_index - 1][input_index];
                    previous[input_index] = propagated * (1.0 - pre.tanh().powi(2));
                }
                delta = previous;
            }
        }
    }

    fn choose(&self, features: &[f64]) -> usize {
        let (logits, _) = self.forward(features);
        if logits.get(1).copied().unwrap_or(f64::NEG_INFINITY)
            > logits.first().copied().unwrap_or(f64::NEG_INFINITY)
        {
            1
        } else {
            0
        }
    }
}

impl Adam {
    fn new(model: &Mlp) -> Self {
        let zero = model.zero_grad();
        Self {
            step: 0,
            mw: zero.weights.clone(),
            vw: zero.weights,
            mb: zero.bias.clone(),
            vb: zero.bias,
        }
    }

    fn update(&mut self, model: &mut Mlp, grad: &Grad, learning_rate: f64, l2: f64) {
        self.step += 1;
        let step = self.step as f64;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let epsilon = 1.0e-8;
        let bias1 = 1.0 - beta1.powf(step);
        let bias2 = 1.0 - beta2.powf(step);
        for layer_index in 0..model.layers.len() {
            for out_index in 0..model.layers[layer_index].weights.len() {
                for input_index in 0..model.layers[layer_index].weights[out_index].len() {
                    let weight = &mut model.layers[layer_index].weights[out_index][input_index];
                    let gradient = grad.weights[layer_index][out_index][input_index] + l2 * *weight;
                    self.mw[layer_index][out_index][input_index] = beta1
                        * self.mw[layer_index][out_index][input_index]
                        + (1.0 - beta1) * gradient;
                    self.vw[layer_index][out_index][input_index] = beta2
                        * self.vw[layer_index][out_index][input_index]
                        + (1.0 - beta2) * gradient * gradient;
                    let m_hat = self.mw[layer_index][out_index][input_index] / bias1;
                    let v_hat = self.vw[layer_index][out_index][input_index] / bias2;
                    *weight -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
            for out_index in 0..model.layers[layer_index].bias.len() {
                let bias = &mut model.layers[layer_index].bias[out_index];
                let gradient = grad.bias[layer_index][out_index];
                self.mb[layer_index][out_index] =
                    beta1 * self.mb[layer_index][out_index] + (1.0 - beta1) * gradient;
                self.vb[layer_index][out_index] =
                    beta2 * self.vb[layer_index][out_index] + (1.0 - beta2) * gradient * gradient;
                let m_hat = self.mb[layer_index][out_index] / bias1;
                let v_hat = self.vb[layer_index][out_index] / bias2;
                *bias -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let dataset = load_dataset(&args.input)?;
    let splits = split_rows(
        &dataset.rows,
        args.train_fraction,
        args.validation_fraction,
        args.purge_bars,
    )?;
    let (means, scales) = fit_scaler(&dataset.rows, &splits.train);
    let samples = normalize_rows(&dataset.rows, &means, &scales);
    let reward_scale = reward_scale(&samples, &splits.train, args.reward_normalization);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let result = match args.algorithm {
        Algorithm::Ppo => train_ppo(&args, &samples, &splits, reward_scale, &mut rng)?,
        Algorithm::Grpo => train_grpo(&args, &samples, &splits, reward_scale, &mut rng)?,
        Algorithm::Rlvr => train_rlvr(&args, &samples, &splits, reward_scale, &mut rng)?,
    };
    write_artifacts(&args, &dataset, &splits, &samples, reward_scale, result)?;
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    if args.epochs == 0 || args.hidden == 0 || args.layers == 0 {
        bail!("epochs, hidden, and layers must be positive");
    }
    if args.ppo_epochs == 0 || args.grpo_epochs == 0 {
        bail!("ppo-epochs and grpo-epochs must be positive");
    }
    if args.group_size < 2 && matches!(args.algorithm, Algorithm::Grpo) {
        bail!("GRPO requires --group-size >= 2");
    }
    if !args.learning_rate.is_finite()
        || args.learning_rate <= 0.0
        || !args.l2.is_finite()
        || args.l2 < 0.0
        || !args.clip.is_finite()
        || !(0.0..1.0).contains(&args.clip)
        || !args.value_coefficient.is_finite()
        || args.value_coefficient < 0.0
        || !args.entropy_coefficient.is_finite()
        || args.entropy_coefficient < 0.0
    {
        bail!("learning rate/l2/clip values are invalid");
    }
    if !args.train_fraction.is_finite()
        || !args.validation_fraction.is_finite()
        || args.train_fraction <= 0.0
        || args.validation_fraction <= 0.0
        || args.train_fraction + args.validation_fraction >= 1.0
    {
        bail!("train and validation fractions must be positive and sum below one");
    }
    Ok(())
}

fn load_dataset(path: &Path) -> Result<Dataset> {
    let frame = ParquetReader::new(File::open(path)?).finish()?;
    if frame.height() == 0 {
        bail!("input event parquet is empty");
    }
    if consistent_string(&frame, "schema_version")? != SUPERVISED_DATASET_SCHEMA {
        bail!("input is not a supervised-event-v1 parquet");
    }
    if consistent_string(&frame, "label_schema")? != SUPERVISED_LABEL_SCHEMA {
        bail!("input label schema is not supported");
    }
    let config_json = consistent_string(&frame, "config_json")?;
    let config: SupervisedConfig = serde_json::from_str(&config_json)?;
    config.validate()?;
    if config.label_mode != LabelMode::NormalInvert {
        bail!("event gate training requires label_mode normal-invert");
    }
    let feature_names = consistent_string(&frame, "feature_schema")?
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if feature_names.is_empty() || feature_names != config.feature_names() {
        bail!("feature_schema does not match config feature registry");
    }
    reject_future_features(&feature_names)?;
    let timestamps = required_i64(&frame, "timestamp_ns")?;
    let row_indices = required_usize(&frame, "row_idx")?;
    let horizons = required_usize(&frame, "interval_end_row_idx")?;
    let sessions = required_strings(&frame, "session_id")?;
    let directions = required_i64(&frame, "raw_direction")?;
    let decision = required_f64(&frame, "decision_price")?;
    let interval_end = required_f64(&frame, "interval_end_price")?;
    let columns = feature_names
        .iter()
        .map(|name| required_f64(&frame, name))
        .collect::<Result<Vec<_>>>()?;
    let count = frame.height();
    if [
        timestamps.len(),
        row_indices.len(),
        horizons.len(),
        sessions.len(),
        directions.len(),
        decision.len(),
        interval_end.len(),
    ]
    .into_iter()
    .any(|length| length != count)
        || columns.iter().any(|column| column.len() != count)
    {
        bail!("input columns have inconsistent lengths");
    }
    if timestamps.windows(2).any(|window| window[1] <= window[0])
        || row_indices.windows(2).any(|window| window[1] <= window[0])
    {
        bail!("event rows must be strictly chronological");
    }
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        if horizons[index] <= row_indices[index] {
            bail!("event horizon must be after event row at {index}");
        }
        let direction = directions[index];
        if !matches!(direction, -1 | 1) {
            bail!("raw_direction must be -1 or 1 at row {index}");
        }
        let features = columns
            .iter()
            .map(|column| column[index])
            .collect::<Vec<_>>();
        if features.iter().any(|value| !value.is_finite()) {
            bail!("causal feature is non-finite at row {index}");
        }
        if !decision[index].is_finite() || !interval_end[index].is_finite() {
            bail!("decision or interval-end price is non-finite at row {index}");
        }
        if let Some(direction_index) = feature_names
            .iter()
            .position(|name| name == "raw_direction")
            && features[direction_index] != direction as f64
        {
            bail!("raw_direction feature disagrees with raw_direction column at row {index}");
        }
        let delta = interval_end[index] - decision[index];
        let normal = direction as f64 * delta * config.contract_multiplier - config.round_trip_cost;
        let invert =
            -(direction as f64) * delta * config.contract_multiplier - config.round_trip_cost;
        if !normal.is_finite() || !invert.is_finite() {
            bail!("derived action reward is non-finite at row {index}");
        }
        rows.push(Row {
            row_idx: row_indices[index],
            horizon_row_idx: horizons[index],
            session_id: sessions[index].clone(),
            raw_features: features,
            rewards: [normal, invert],
        });
    }
    Ok(Dataset {
        feature_names,
        rows,
        config_json,
        source_hash: consistent_string(&frame, "source_hash_sha256")?,
        fingerprint: consistent_string(&frame, "dataset_fingerprint_sha256")?,
    })
}

fn split_rows(
    rows: &[Row],
    train_fraction: f64,
    validation_fraction: f64,
    purge_bars: usize,
) -> Result<Splits> {
    let mut sessions = Vec::<(String, usize)>::new();
    for (index, row) in rows.iter().enumerate() {
        if sessions
            .last()
            .is_none_or(|(session, _)| session != &row.session_id)
        {
            sessions.push((row.session_id.clone(), index));
        }
    }
    if sessions.len() < 3 {
        bail!("at least three complete sessions are required");
    }
    let train_sessions = ((sessions.len() as f64 * train_fraction).floor() as usize)
        .max(1)
        .min(sessions.len() - 2);
    let validation_sessions = ((sessions.len() as f64 * validation_fraction).floor() as usize)
        .max(1)
        .min(sessions.len() - train_sessions - 1);
    let train_boundary = sessions[train_sessions].1;
    let validation_boundary = sessions[train_sessions + validation_sessions].1;
    let first_validation_row = rows[train_boundary].row_idx;
    let first_holdout_row = rows[validation_boundary].row_idx;
    let train_end = rows[..train_boundary]
        .iter()
        .rposition(|row| row.horizon_row_idx.saturating_add(purge_bars) < first_validation_row)
        .map(|index| index + 1)
        .unwrap_or(0);
    let validation_end = rows[train_boundary..validation_boundary]
        .iter()
        .rposition(|row| row.horizon_row_idx.saturating_add(purge_bars) < first_holdout_row)
        .map(|index| train_boundary + index + 1)
        .unwrap_or(train_boundary);
    if train_end == 0 || train_boundary >= validation_end || validation_boundary >= rows.len() {
        bail!("purge-bars leaves an empty split");
    }
    Ok(Splits {
        train: 0..train_end,
        validation: train_boundary..validation_end,
        holdout: validation_boundary..rows.len(),
        purge_bars,
        sessions: sessions.len(),
    })
}

fn fit_scaler(rows: &[Row], range: &Range<usize>) -> (Vec<f64>, Vec<f64>) {
    let feature_count = rows[0].raw_features.len();
    let count = (range.end - range.start).max(1) as f64;
    let mut means = vec![0.0; feature_count];
    for row in &rows[range.clone()] {
        for (mean, value) in means.iter_mut().zip(&row.raw_features) {
            *mean += *value;
        }
    }
    for mean in &mut means {
        *mean /= count;
    }
    let mut scales = vec![0.0; feature_count];
    for row in &rows[range.clone()] {
        for ((scale, mean), value) in scales.iter_mut().zip(&means).zip(&row.raw_features) {
            *scale += (value - mean).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / count).sqrt();
        if !scale.is_finite() || *scale < 1.0e-12 {
            *scale = 1.0;
        }
    }
    (means, scales)
}

fn normalize_rows(rows: &[Row], means: &[f64], scales: &[f64]) -> Vec<Sample> {
    rows.iter()
        .map(|row| Sample {
            features: row
                .raw_features
                .iter()
                .zip(means)
                .zip(scales)
                .map(|((value, mean), scale)| (value - mean) / scale)
                .collect(),
            rewards: row.rewards,
        })
        .collect()
}

fn reward_scale(samples: &[Sample], range: &Range<usize>, mode: RewardNormalization) -> f64 {
    if matches!(mode, RewardNormalization::None) {
        return 1.0;
    }
    let count = ((range.end - range.start) * 2).max(1) as f64;
    samples[range.clone()]
        .iter()
        .flat_map(|sample| sample.rewards)
        .map(f64::abs)
        .sum::<f64>()
        / count.max(1.0)
}

fn train_ppo(
    args: &Args,
    samples: &[Sample],
    splits: &Splits,
    reward_scale: f64,
    rng: &mut StdRng,
) -> Result<TrainResult> {
    let input_dim = samples[0].features.len();
    let mut policy = Mlp::random(input_dim, args.hidden, args.layers, 2, rng);
    let mut value = Mlp::random(input_dim, args.hidden, args.layers, 1, rng);
    let mut policy_optimizer = Adam::new(&policy);
    let mut value_optimizer = Adam::new(&value);
    let mut best_policy = policy.clone();
    let mut best_validation = f64::NEG_INFINITY;
    let mut selected_epoch = 0;
    let mut checkpoints = Vec::new();
    for epoch in 1..=args.epochs {
        let batch = collect_ppo_batch(samples, &splits.train, &policy, &value, reward_scale, rng);
        let losses = ppo_update(
            &mut policy,
            &mut value,
            &mut policy_optimizer,
            &mut value_optimizer,
            &batch,
            args,
        );
        let train = evaluate(samples, &splits.train, &policy);
        let validation = evaluate(samples, &splits.validation, &policy);
        if validation.sum_pnl_usd > best_validation {
            best_validation = validation.sum_pnl_usd;
            best_policy = policy.clone();
            selected_epoch = epoch;
        }
        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            checkpoints.push(Checkpoint {
                epoch,
                train,
                validation,
                policy_loss: losses.0,
                value_loss: Some(losses.1),
                entropy: losses.2,
                approx_kl: losses.3,
                clip_fraction: losses.4,
            });
        }
    }
    Ok(TrainResult {
        policy: best_policy,
        selected_epoch,
        checkpoints,
    })
}

fn collect_ppo_batch(
    samples: &[Sample],
    range: &Range<usize>,
    policy: &Mlp,
    value: &Mlp,
    reward_scale: f64,
    rng: &mut StdRng,
) -> Vec<PpoItem> {
    samples[range.clone()]
        .iter()
        .map(|sample| {
            let (logits, _) = policy.forward(&sample.features);
            let probs = softmax(&logits);
            let action = sample_action(&probs, rng);
            let old_logp = probs[action].max(1.0e-12).ln();
            let (value_logits, _) = value.forward(&sample.features);
            let reward = sample.rewards[action] / reward_scale;
            let value_estimate = value_logits[0];
            PpoItem {
                features: sample.features.clone(),
                action,
                old_logp,
                reward,
                advantage: reward - value_estimate,
            }
        })
        .collect()
}

fn ppo_update(
    policy: &mut Mlp,
    value: &mut Mlp,
    policy_optimizer: &mut Adam,
    value_optimizer: &mut Adam,
    batch: &[PpoItem],
    args: &Args,
) -> (f64, f64, f64, f64, f64) {
    let normalized =
        normalize_advantages(&batch.iter().map(|item| item.advantage).collect::<Vec<_>>());
    let mut last = (0.0, 0.0, 0.0, 0.0, 0.0);
    let count = batch.len().max(1) as f64;
    for _ in 0..args.ppo_epochs.max(1) {
        let mut policy_grad = policy.zero_grad();
        let mut value_grad = value.zero_grad();
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;
        let mut entropy_sum = 0.0;
        let mut kl_sum = 0.0;
        let mut clipped = 0usize;
        for (index, item) in batch.iter().enumerate() {
            let (logits, cache) = policy.forward(&item.features);
            let probs = softmax(&logits);
            let new_logp = probs[item.action].max(1.0e-12).ln();
            let log_ratio = (new_logp - item.old_logp).clamp(-20.0, 20.0);
            let ratio = log_ratio.exp();
            let advantage = normalized[index];
            let unclipped = ratio * advantage;
            let clipped_ratio = ratio.clamp(1.0 - args.clip, 1.0 + args.clip);
            let clipped_objective = clipped_ratio * advantage;
            policy_loss += -unclipped.min(clipped_objective) / count;
            let is_clipped = (advantage >= 0.0 && ratio > 1.0 + args.clip)
                || (advantage < 0.0 && ratio < 1.0 - args.clip);
            if is_clipped {
                clipped += 1;
            }
            let entropy = entropy(&probs);
            entropy_sum += entropy / count;
            let mut output_grad = vec![0.0; 2];
            if !is_clipped {
                let coefficient = -advantage * ratio / count;
                for (class, gradient) in output_grad.iter_mut().enumerate() {
                    *gradient += coefficient
                        * if class == item.action {
                            1.0 - probs[class]
                        } else {
                            -probs[class]
                        };
                }
            }
            for (class, gradient) in output_grad.iter_mut().enumerate() {
                *gradient += args.entropy_coefficient
                    * probs[class]
                    * (probs[class].max(1.0e-12).ln() + entropy)
                    / count;
            }
            policy.accumulate_backward(&cache, &output_grad, &mut policy_grad);

            let (value_logits, value_cache) = value.forward(&item.features);
            let difference = value_logits[0] - item.reward;
            value_loss += args.value_coefficient * difference * difference / count;
            value.accumulate_backward(
                &value_cache,
                &[2.0 * args.value_coefficient * difference / count],
                &mut value_grad,
            );
            kl_sum += ((ratio - 1.0) - log_ratio) / count;
        }
        policy_optimizer.update(policy, &policy_grad, args.learning_rate, args.l2);
        value_optimizer.update(value, &value_grad, args.learning_rate, args.l2);
        last = (
            policy_loss,
            value_loss,
            entropy_sum,
            kl_sum,
            clipped as f64 / count,
        );
    }
    last
}

/// Optimize the exact verifier reward for both available actions at every
/// event. Unlike PPO/GRPO, this does not sample one action and estimate an
/// advantage: the trainer knows both causal counterfactual rewards because the
/// event parquet contains the future interval only for training-time reward
/// calculation. Those rewards never enter the observation vector.
fn train_rlvr(
    args: &Args,
    samples: &[Sample],
    splits: &Splits,
    reward_scale: f64,
    rng: &mut StdRng,
) -> Result<TrainResult> {
    let input_dim = samples[0].features.len();
    let mut policy = Mlp::random(input_dim, args.hidden, args.layers, 2, rng);
    let mut optimizer = Adam::new(&policy);
    let mut best_policy = policy.clone();
    let mut best_validation = f64::NEG_INFINITY;
    let mut selected_epoch = 0;
    let mut checkpoints = Vec::new();
    for epoch in 1..=args.epochs {
        let losses = rlvr_update(
            &mut policy,
            &mut optimizer,
            samples,
            &splits.train,
            reward_scale,
            args,
        );
        let train = evaluate(samples, &splits.train, &policy);
        let validation = evaluate(samples, &splits.validation, &policy);
        if validation.sum_pnl_usd > best_validation {
            best_validation = validation.sum_pnl_usd;
            best_policy = policy.clone();
            selected_epoch = epoch;
        }
        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            checkpoints.push(Checkpoint {
                epoch,
                train,
                validation,
                policy_loss: losses.0,
                value_loss: None,
                entropy: losses.1,
                approx_kl: 0.0,
                clip_fraction: 0.0,
            });
        }
    }
    Ok(TrainResult {
        policy: best_policy,
        selected_epoch,
        checkpoints,
    })
}

fn rlvr_update(
    policy: &mut Mlp,
    optimizer: &mut Adam,
    samples: &[Sample],
    range: &Range<usize>,
    reward_scale: f64,
    args: &Args,
) -> (f64, f64) {
    let count = (range.end - range.start).max(1) as f64;
    let scale = reward_scale.max(1.0e-12);
    let mut grad = policy.zero_grad();
    let mut loss = 0.0;
    let mut entropy_sum = 0.0;
    for sample in &samples[range.clone()] {
        let (logits, cache) = policy.forward(&sample.features);
        let probabilities = softmax(&logits);
        let normalized_rewards = [sample.rewards[0] / scale, sample.rewards[1] / scale];
        let expected_reward = probabilities
            .iter()
            .zip(normalized_rewards)
            .map(|(probability, reward)| probability * reward)
            .sum::<f64>();
        let sample_entropy = entropy(&probabilities);
        // Keep the reported objective aligned with the gradient below:
        // minimize -E[r] - entropy_coefficient * H.
        loss -= (expected_reward + args.entropy_coefficient * sample_entropy) / count;
        entropy_sum += sample_entropy / count;

        // d[-sum(p_a r_a)] / d(logit_k) = -p_k (r_k - E[r]).
        let mut output_grad = vec![0.0; 2];
        for class in 0..2 {
            output_grad[class] =
                -probabilities[class] * (normalized_rewards[class] - expected_reward) / count;
            // The loss is -expected_reward - entropy_coefficient * H.
            output_grad[class] += args.entropy_coefficient
                * probabilities[class]
                * (probabilities[class].max(1.0e-12).ln() + sample_entropy)
                / count;
        }
        policy.accumulate_backward(&cache, &output_grad, &mut grad);
    }
    optimizer.update(policy, &grad, args.learning_rate, args.l2);
    (loss, entropy_sum)
}

fn train_grpo(
    args: &Args,
    samples: &[Sample],
    splits: &Splits,
    reward_scale: f64,
    rng: &mut StdRng,
) -> Result<TrainResult> {
    let input_dim = samples[0].features.len();
    let mut policy = Mlp::random(input_dim, args.hidden, args.layers, 2, rng);
    let mut optimizer = Adam::new(&policy);
    let mut best_policy = policy.clone();
    let mut best_validation = f64::NEG_INFINITY;
    let mut selected_epoch = 0;
    let mut checkpoints = Vec::new();
    for epoch in 1..=args.epochs {
        let batch = collect_grpo_batch(
            samples,
            &splits.train,
            &policy,
            reward_scale,
            args.group_size,
            rng,
        );
        let losses = grpo_update(&mut policy, &mut optimizer, &batch, args);
        let train = evaluate(samples, &splits.train, &policy);
        let validation = evaluate(samples, &splits.validation, &policy);
        if validation.sum_pnl_usd > best_validation {
            best_validation = validation.sum_pnl_usd;
            best_policy = policy.clone();
            selected_epoch = epoch;
        }
        if args.checkpoint_every > 0 && epoch % args.checkpoint_every == 0 {
            checkpoints.push(Checkpoint {
                epoch,
                train,
                validation,
                policy_loss: losses.0,
                value_loss: None,
                entropy: losses.1,
                approx_kl: losses.2,
                clip_fraction: losses.3,
            });
        }
    }
    Ok(TrainResult {
        policy: best_policy,
        selected_epoch,
        checkpoints,
    })
}

fn collect_grpo_batch(
    samples: &[Sample],
    range: &Range<usize>,
    policy: &Mlp,
    reward_scale: f64,
    group_size: usize,
    rng: &mut StdRng,
) -> Vec<GrpoItem> {
    let mut items = Vec::with_capacity((range.end - range.start) * group_size);
    for sample in &samples[range.clone()] {
        let (logits, _) = policy.forward(&sample.features);
        let probs = softmax(&logits);
        let mut group = Vec::with_capacity(group_size);
        for _ in 0..group_size {
            let action = sample_action(&probs, rng);
            group.push((
                action,
                probs[action].max(1.0e-12).ln(),
                sample.rewards[action] / reward_scale,
            ));
        }
        let mean = group.iter().map(|(_, _, reward)| reward).sum::<f64>() / group.len() as f64;
        let variance = group
            .iter()
            .map(|(_, _, reward)| (reward - mean).powi(2))
            .sum::<f64>()
            / group.len() as f64;
        let std = variance.sqrt().max(1.0e-8);
        for (action, old_logp, reward) in group {
            items.push(GrpoItem {
                features: sample.features.clone(),
                action,
                old_logp,
                advantage: (reward - mean) / std,
            });
        }
    }
    items
}

fn grpo_update(
    policy: &mut Mlp,
    optimizer: &mut Adam,
    batch: &[GrpoItem],
    args: &Args,
) -> (f64, f64, f64, f64) {
    let count = batch.len().max(1) as f64;
    let mut last = (0.0, 0.0, 0.0, 0.0);
    for _ in 0..args.grpo_epochs.max(1) {
        let mut grad = policy.zero_grad();
        let mut policy_loss = 0.0;
        let mut entropy_sum = 0.0;
        let mut kl_sum = 0.0;
        let mut clipped = 0usize;
        for item in batch {
            let (logits, cache) = policy.forward(&item.features);
            let probs = softmax(&logits);
            let new_logp = probs[item.action].max(1.0e-12).ln();
            let log_ratio = (new_logp - item.old_logp).clamp(-20.0, 20.0);
            let ratio = log_ratio.exp();
            let unclipped = ratio * item.advantage;
            let clipped_ratio = ratio.clamp(1.0 - args.clip, 1.0 + args.clip);
            policy_loss += -unclipped.min(clipped_ratio * item.advantage) / count;
            let is_clipped = (item.advantage >= 0.0 && ratio > 1.0 + args.clip)
                || (item.advantage < 0.0 && ratio < 1.0 - args.clip);
            if is_clipped {
                clipped += 1;
            }
            let entropy = entropy(&probs);
            entropy_sum += entropy / count;
            let mut output_grad = vec![0.0; 2];
            if !is_clipped {
                let coefficient = -item.advantage * ratio / count;
                for (class, gradient) in output_grad.iter_mut().enumerate() {
                    *gradient += coefficient
                        * if class == item.action {
                            1.0 - probs[class]
                        } else {
                            -probs[class]
                        };
                }
            }
            for (class, gradient) in output_grad.iter_mut().enumerate() {
                *gradient += args.entropy_coefficient
                    * probs[class]
                    * (probs[class].max(1.0e-12).ln() + entropy)
                    / count;
            }
            policy.accumulate_backward(&cache, &output_grad, &mut grad);
            kl_sum += ((ratio - 1.0) - log_ratio) / count;
        }
        optimizer.update(policy, &grad, args.learning_rate, args.l2);
        last = (policy_loss, entropy_sum, kl_sum, clipped as f64 / count);
    }
    last
}

fn evaluate(samples: &[Sample], range: &Range<usize>, policy: &Mlp) -> Summary {
    let mut normal_count = 0;
    let mut invert_count = 0;
    let mut pnl = 0.0;
    let mut oracle = 0.0;
    let mut positive = 0;
    for sample in &samples[range.clone()] {
        let action = policy.choose(&sample.features);
        if action == 0 {
            normal_count += 1;
        } else {
            invert_count += 1;
        }
        pnl += sample.rewards[action];
        oracle += sample.rewards[0].max(sample.rewards[1]);
        if sample.rewards[action] > 0.0 {
            positive += 1;
        }
    }
    let count = normal_count + invert_count;
    Summary {
        event_count: count,
        normal_count,
        invert_count,
        normal_fraction: normal_count as f64 / count.max(1) as f64,
        invert_fraction: invert_count as f64 / count.max(1) as f64,
        sum_pnl_usd: pnl,
        mean_pnl_usd: pnl / count.max(1) as f64,
        oracle_pnl_usd: oracle,
        oracle_capture: if oracle.abs() < 1.0e-12 {
            0.0
        } else {
            pnl / oracle
        },
        positive_event_fraction: positive as f64 / count.max(1) as f64,
    }
}

fn write_artifacts(
    args: &Args,
    dataset: &Dataset,
    splits: &Splits,
    samples: &[Sample],
    reward_scale: f64,
    result: TrainResult,
) -> Result<()> {
    std::fs::create_dir_all(&args.outdir)?;
    let algorithm = match args.algorithm {
        Algorithm::Ppo => "ppo",
        Algorithm::Grpo => "grpo",
        Algorithm::Rlvr => "rlvr",
    };
    let normalization = match args.reward_normalization {
        RewardNormalization::None => "none",
        RewardNormalization::TrainMeanAbs => "train-mean-abs",
    };
    let training = TrainingMetadata {
        epochs: args.epochs,
        ppo_epochs: args.ppo_epochs,
        grpo_epochs: args.grpo_epochs,
        group_size: args.group_size,
        hidden: args.hidden,
        layers: args.layers,
        learning_rate: args.learning_rate,
        l2: args.l2,
        clip: args.clip,
        value_coefficient: args.value_coefficient,
        entropy_coefficient: args.entropy_coefficient,
        train_fraction: args.train_fraction,
        validation_fraction: args.validation_fraction,
        purge_bars: args.purge_bars,
        checkpoint_every: args.checkpoint_every,
    };
    let (feature_means, feature_scales) = fit_scaler(&dataset.rows, &splits.train);
    let metrics = Metrics {
        schema_version: "event-gate-rl-metrics-v1",
        warning: WARNING,
        algorithm,
        model_semantics: "two-action-causal-contextual-bandit",
        input: args.input.display().to_string(),
        seed: args.seed,
        feature_names: dataset.feature_names.clone(),
        config_json: dataset.config_json.clone(),
        source_hash_sha256: dataset.source_hash.clone(),
        dataset_fingerprint_sha256: dataset.fingerprint.clone(),
        reward_normalization: normalization,
        reward_scale_usd: reward_scale,
        reward_definition: REWARD_DEFINITION,
        training: training.clone(),
        checkpoint_selection: "highest deterministic greedy validation sum_pnl_usd",
        split: SplitReport {
            total: dataset.rows.len(),
            train: splits.train.len(),
            validation: splits.validation.len(),
            holdout: splits.holdout.len(),
            purged: dataset.rows.len()
                - splits.train.len()
                - splits.validation.len()
                - splits.holdout.len(),
            sessions: splits.sessions,
            purge_bars: splits.purge_bars,
        },
        selected_epoch: result.selected_epoch,
        selected_train: evaluate(samples, &splits.train, &result.policy),
        selected_validation: evaluate(samples, &splits.validation, &result.policy),
        fixed_normal: fixed_summary(samples, &splits.holdout, 0),
        fixed_invert: fixed_summary(samples, &splits.holdout, 1),
        learned: evaluate(samples, &splits.holdout, &result.policy),
        checkpoints: result.checkpoints,
        leakage_check: "passed",
    };
    let policy = serde_json::json!({
        "schema_version": "event-gate-rl-policy-v1",
        "warning": WARNING,
        "algorithm": algorithm,
        "model_semantics": "two-action-causal-contextual-bandit",
        "seed": args.seed,
        "feature_names": dataset.feature_names,
        "feature_means": feature_means,
        "feature_scales": feature_scales,
        "config_json": dataset.config_json,
        "source_hash_sha256": dataset.source_hash,
        "dataset_fingerprint_sha256": dataset.fingerprint,
        "reward_scale_usd": reward_scale,
        "reward_normalization": normalization,
        "reward_definition": REWARD_DEFINITION,
        "training": training,
        "selected_epoch": result.selected_epoch,
        "checkpoint_selection": "highest deterministic greedy validation sum_pnl_usd",
        "leakage_check": "passed",
        "architecture": {"input": result.policy.input_dim, "hidden_layers": result.policy.hidden_layers, "output": result.policy.output_dim},
        "layers": result.policy.layers.iter().map(|layer| serde_json::json!({"weights":layer.weights,"bias":layer.bias})).collect::<Vec<_>>(),
    });
    write_json(&args.outdir, "metrics.json", &metrics)?;
    write_json(&args.outdir, "policy.json", &policy)?;
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}

fn fixed_summary(samples: &[Sample], range: &Range<usize>, action: usize) -> Summary {
    let mut policy = Mlp {
        layers: vec![Layer {
            weights: vec![vec![0.0; samples[0].features.len()]; 2],
            bias: vec![0.0, 0.0],
        }],
        input_dim: samples[0].features.len(),
        output_dim: 2,
        hidden_layers: 0,
    };
    policy.layers[0].bias[action] = 1.0;
    evaluate(samples, range, &policy)
}

fn normalize_advantages(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt().max(1.0e-8);
    values.iter().map(|value| (value - mean) / std).collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut values = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f64>().max(1.0e-12);
    for value in &mut values {
        *value /= sum;
    }
    values
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .map(|probability| {
            let probability = probability.max(1.0e-12);
            -probability * probability.ln()
        })
        .sum()
}

fn sample_action(probabilities: &[f64], rng: &mut StdRng) -> usize {
    WeightedIndex::new(probabilities.iter().map(|value| value.max(0.0)))
        .map(|distribution| distribution.sample(rng))
        .unwrap_or_else(|_| {
            probabilities
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .unwrap_or(0)
        })
}

fn reject_future_features(names: &[String]) -> Result<()> {
    for name in names {
        if name.contains("future")
            || name.contains("oracle")
            || name.contains("action_value")
            || matches!(
                name.as_str(),
                "label_action"
                    | "label_name"
                    | "decision_price"
                    | "interval_end_price"
                    | "interval_end_row_idx"
                    | "terminal_event"
            )
        {
            bail!("feature {name:?} is future/label-derived");
        }
    }
    Ok(())
}

fn consistent_string(frame: &DataFrame, name: &str) -> Result<String> {
    let values = required_strings(frame, name)?;
    let first = values
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("column {name} is empty"))?;
    if values.iter().any(|value| value != &first) {
        bail!("column {name} is not consistent across rows");
    }
    Ok(first)
}

fn required_f64(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let series = frame
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::Float64)?;
    Ok(series
        .f64()?
        .into_iter()
        .map(|value| value.ok_or_else(|| anyhow::anyhow!("null in column {name}")))
        .collect::<Result<Vec<_>>>()?)
}

fn required_i64(frame: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let series = frame
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::Int64)?;
    Ok(series
        .i64()?
        .into_iter()
        .map(|value| value.ok_or_else(|| anyhow::anyhow!("null in column {name}")))
        .collect::<Result<Vec<_>>>()?)
}

fn required_usize(frame: &DataFrame, name: &str) -> Result<Vec<usize>> {
    let series = frame
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::UInt64)?;
    Ok(series
        .u64()?
        .into_iter()
        .map(|value| {
            value
                .ok_or_else(|| anyhow::anyhow!("null in column {name}"))
                .and_then(|value| usize::try_from(value).context("usize conversion failed"))
        })
        .collect::<Result<Vec<_>>>()?)
}

fn required_strings(frame: &DataFrame, name: &str) -> Result<Vec<String>> {
    let series = frame.column(name)?.as_materialized_series();
    Ok(series
        .str()?
        .into_iter()
        .map(|value| {
            value
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow::anyhow!("null in column {name}"))
        })
        .collect::<Result<Vec<_>>>()?)
}

fn write_json<T: Serialize>(outdir: &Path, name: &str, value: &T) -> Result<()> {
    std::fs::create_dir_all(outdir)?;
    std::fs::write(
        outdir.join(name),
        format!("{}\n", serde_json::to_string_pretty(value)?),
    )?;
    Ok(())
}

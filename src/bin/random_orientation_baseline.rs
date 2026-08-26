//! Monte-Carlo baselines for a two-class normal/invert event dataset.
//!
//! This is deliberately separate from model training. It reads only the two
//! causal event path already stored in the supervised event parquet and
//! chooses normal or invert without reading features or labels.

use anyhow::{Context, Result, bail};
use clap::Parser;
use polars::prelude::{DataFrame, DataType, ParquetReader, SerReader};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    about = "Monte-Carlo normal/invert baseline for a supervised event parquet",
    long_about = "Choose normal or invert independently for every crossover event. The replay uses only causal event fields, then applies the configured multiplier and transition costs. No features, labels, action values, oracle actions, or future-derived columns are read for the random policies."
)]
struct Args {
    /// Two-class supervised event parquet.
    #[arg(long)]
    input: PathBuf,
    /// Seed for the deterministic coin-flip stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of random trials used for the Monte-Carlo distributions.
    #[arg(long, default_value_t = 10_000)]
    trials: usize,
    /// Step for the inverted-probability sweep, for example 0.05.
    #[arg(long, default_value_t = 0.05)]
    probability_step: f64,
    /// Optional JSON output path.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    Train,
    Validation,
    Holdout,
}

impl Window {
    const ALL: [Self; 3] = [Self::Train, Self::Validation, Self::Holdout];

    fn label(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Holdout => "holdout",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::Holdout => 2,
        }
    }
}

#[derive(Debug, Clone)]
struct Event {
    session_id: String,
    raw_direction: i8,
    decision_price: f64,
    interval_end_price: f64,
    terminal_event: bool,
    oracle_action: i8,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct WindowMetrics {
    events: usize,
    normal_actions: usize,
    invert_actions: usize,
    pnl: f64,
    mean_pnl: f64,
    wins: usize,
    win_rate: f64,
    max_drawdown: f64,
}

impl WindowMetrics {
    fn empty() -> Self {
        Self {
            events: 0,
            normal_actions: 0,
            invert_actions: 0,
            pnl: 0.0,
            mean_pnl: 0.0,
            wins: 0,
            win_rate: 0.0,
            max_drawdown: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct FixedPolicyMetrics {
    normal: [WindowMetrics; 3],
    invert: [WindowMetrics; 3],
    two_class_oracle_pnl: [f64; 3],
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ExpectedSweepRow {
    probability_invert: f64,
    train_pnl: f64,
    validation_pnl: f64,
    holdout_pnl: f64,
    all_pnl: f64,
    positive_windows: usize,
    all_windows_positive: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct Distribution {
    mean: f64,
    p05: f64,
    p50: f64,
    p95: f64,
    positive_fraction: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MonteCarloSweepRow {
    probability_invert: f64,
    train: Distribution,
    validation: Distribution,
    holdout: Distribution,
    all_windows_positive_fraction: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    input: String,
    seed: u64,
    trials: usize,
    probability_step: f64,
    event_count: usize,
    session_count: usize,
    contract_multiplier: f64,
    round_trip_cost: f64,
    train_sessions: Vec<String>,
    validation_sessions: Vec<String>,
    holdout_sessions: Vec<String>,
    fixed_policies: FixedPolicyMetrics,
    coin_flip_seed_result: [WindowMetrics; 3],
    expected_sweep: Vec<ExpectedSweepRow>,
    monte_carlo_sweep: Vec<MonteCarloSweepRow>,
}

#[derive(Debug, Clone, Copy)]
struct SplitAccumulator {
    metrics: WindowMetrics,
    cumulative: f64,
    peak: f64,
}

impl SplitAccumulator {
    fn empty() -> Self {
        Self {
            metrics: WindowMetrics::empty(),
            cumulative: 0.0,
            peak: 0.0,
        }
    }

    fn add(&mut self, invert: bool, pnl: f64) {
        self.metrics.events += 1;
        if invert {
            self.metrics.invert_actions += 1;
        } else {
            self.metrics.normal_actions += 1;
        }
        self.add_pnl_only(pnl);
        if pnl > 0.0 {
            self.metrics.wins += 1;
        }
    }

    fn add_pnl_only(&mut self, pnl: f64) {
        self.metrics.pnl += pnl;
        self.cumulative += pnl;
        self.peak = self.peak.max(self.cumulative);
        self.metrics.max_drawdown = self.metrics.max_drawdown.max(self.peak - self.cumulative);
    }

    fn finish(mut self) -> WindowMetrics {
        self.metrics.mean_pnl = if self.metrics.events == 0 {
            0.0
        } else {
            self.metrics.pnl / self.metrics.events as f64
        };
        self.metrics.win_rate = if self.metrics.events == 0 {
            0.0
        } else {
            self.metrics.wins as f64 / self.metrics.events as f64
        };
        self.metrics
    }
}

/// Small deterministic generator so a saved report is reproducible from seed.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.trials == 0 {
        bail!("--trials must be greater than zero");
    }
    if !args.probability_step.is_finite()
        || args.probability_step <= 0.0
        || args.probability_step > 1.0
    {
        bail!("--probability-step must be finite and in (0, 1]");
    }

    let (events, contract_multiplier, round_trip_cost) = load_events(&args.input)?;
    if events.is_empty() {
        bail!("input {} contains no events", args.input.display());
    }
    let (assignments, train_sessions, validation_sessions, holdout_sessions) =
        make_session_split(&events)?;
    let fixed_policies =
        fixed_policy_metrics(&events, &assignments, contract_multiplier, round_trip_cost);
    let coin_flip_seed_result = sampled_metrics(
        &events,
        &assignments,
        0.5,
        args.seed,
        contract_multiplier,
        round_trip_cost,
    );
    let probabilities = probability_grid(args.probability_step);
    let expected_sweep = probabilities
        .iter()
        .copied()
        .map(|probability_invert| {
            expected_row(
                probability_invert,
                &events,
                &assignments,
                contract_multiplier,
                round_trip_cost,
            )
        })
        .collect::<Vec<_>>();
    let monte_carlo_sweep = probabilities
        .iter()
        .copied()
        .map(|probability_invert| {
            monte_carlo_row(
                probability_invert,
                &events,
                &assignments,
                args.seed,
                args.trials,
                contract_multiplier,
                round_trip_cost,
            )
        })
        .collect::<Vec<_>>();

    print_report(
        &args,
        events.len(),
        train_sessions.as_slice(),
        validation_sessions.as_slice(),
        holdout_sessions.as_slice(),
        &fixed_policies,
        &coin_flip_seed_result,
        &expected_sweep,
        &monte_carlo_sweep,
    );

    if let Some(path) = args.json_out {
        let report = Report {
            input: args.input.display().to_string(),
            seed: args.seed,
            trials: args.trials,
            probability_step: args.probability_step,
            event_count: events.len(),
            session_count: train_sessions.len()
                + validation_sessions.len()
                + holdout_sessions.len(),
            contract_multiplier,
            round_trip_cost,
            train_sessions,
            validation_sessions,
            holdout_sessions,
            fixed_policies,
            coin_flip_seed_result,
            expected_sweep,
            monte_carlo_sweep,
        };
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        let bytes = serde_json::to_vec_pretty(&report).context("serialize random baseline")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("write random baseline report {}", path.display()))?;
        println!("\nJSON report: {}", path.display());
    }
    Ok(())
}

fn load_events(path: &PathBuf) -> Result<(Vec<Event>, f64, f64)> {
    let file = File::open(path).with_context(|| format!("open input {}", path.display()))?;
    let frame = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("read input {}", path.display()))?;
    let session_ids = string_column(&frame, "session_id")?;
    let raw_directions = integer_column(&frame, "raw_direction")?;
    let decision_prices = float_column(&frame, "decision_price")?;
    let interval_end_prices = float_column(&frame, "interval_end_price")?;
    let terminal_events = bool_column(&frame, "terminal_event")?;
    let oracle_actions = integer_column(&frame, "label_action")?;
    let timestamps = integer_column(&frame, "timestamp_ns")?;
    let contract_multiplier = consistent_float_column(&frame, "contract_multiplier")?;
    let round_trip_cost = consistent_float_column(&frame, "round_trip_cost")?;
    let len = frame.height();
    if [
        session_ids.len(),
        raw_directions.len(),
        decision_prices.len(),
        interval_end_prices.len(),
        terminal_events.len(),
        oracle_actions.len(),
        timestamps.len(),
    ]
    .iter()
    .any(|length| *length != len)
    {
        bail!("required event columns have inconsistent lengths");
    }
    if !contract_multiplier.is_finite() || contract_multiplier <= 0.0 {
        bail!("contract_multiplier must be finite and positive");
    }
    if !round_trip_cost.is_finite() || round_trip_cost < 0.0 {
        bail!("round_trip_cost must be finite and non-negative");
    }
    if timestamps.windows(2).any(|window| window[1] <= window[0]) {
        bail!("timestamp_ns must be strictly increasing");
    }
    let mut events = Vec::with_capacity(len);
    for index in 0..len {
        if !matches!(raw_directions[index], -1 | 1) {
            bail!(
                "raw_direction must be -1 or 1 at row {}, got {}",
                index,
                raw_directions[index]
            );
        }
        if !matches!(oracle_actions[index], -1 | 1) {
            bail!(
                "this baseline expects a normal/invert label_action at row {}, got {}",
                index,
                oracle_actions[index]
            );
        }
        events.push(Event {
            session_id: session_ids[index].clone(),
            raw_direction: raw_directions[index] as i8,
            decision_price: decision_prices[index],
            interval_end_price: interval_end_prices[index],
            terminal_event: terminal_events[index],
            oracle_action: oracle_actions[index] as i8,
        });
    }
    Ok((events, contract_multiplier, round_trip_cost))
}

fn string_column(frame: &DataFrame, name: &str) -> Result<Vec<String>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input is missing {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::String)
        .with_context(|| format!("cast {name} to String"))?;
    cast.str()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow::anyhow!("{name} is null at row {index}"))
        })
        .collect()
}

fn integer_column(frame: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input is missing {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Int64)
        .with_context(|| format!("cast {name} to Int64"))?;
    cast.i64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.ok_or_else(|| anyhow::anyhow!("{name} is null at row {index}")))
        .collect()
}

fn bool_column(frame: &DataFrame, name: &str) -> Result<Vec<bool>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input is missing {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Boolean)
        .with_context(|| format!("cast {name} to Boolean"))?;
    cast.bool()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| value.ok_or_else(|| anyhow::anyhow!("{name} is null at row {index}")))
        .collect()
}

fn float_column(frame: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let column = frame
        .column(name)
        .with_context(|| format!("input is missing {name}"))?;
    let cast = column
        .as_materialized_series()
        .cast(&DataType::Float64)
        .with_context(|| format!("cast {name} to Float64"))?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value.ok_or_else(|| anyhow::anyhow!("{name} is null at row {index}"))?;
            if !value.is_finite() {
                bail!("{name} is non-finite at row {index}");
            }
            Ok(value)
        })
        .collect()
}

fn consistent_float_column(frame: &DataFrame, name: &str) -> Result<f64> {
    let values = float_column(frame, name)?;
    let first = values
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("input has no rows while reading {name}"))?;
    if values.iter().any(|value| (*value - first).abs() > 1e-9) {
        bail!("input column {name} is not consistent across rows");
    }
    Ok(first)
}

fn make_session_split(
    events: &[Event],
) -> Result<(Vec<Window>, Vec<String>, Vec<String>, Vec<String>)> {
    let mut sessions = Vec::new();
    let mut seen = BTreeSet::new();
    for event in events {
        if seen.insert(event.session_id.clone()) {
            sessions.push(event.session_id.clone());
        }
    }
    if sessions.len() < 3 {
        bail!("need at least three sessions for train/validation/holdout");
    }
    let train_count = ((sessions.len() as f64 * 0.60).floor() as usize)
        .clamp(1, sessions.len().saturating_sub(2));
    let validation_count = ((sessions.len() as f64 * 0.20).floor() as usize)
        .clamp(1, sessions.len().saturating_sub(train_count + 1));
    let train_sessions = sessions[..train_count].to_vec();
    let validation_sessions = sessions[train_count..train_count + validation_count].to_vec();
    let holdout_sessions = sessions[train_count + validation_count..].to_vec();

    let mut session_window = HashMap::new();
    for session in &train_sessions {
        session_window.insert(session.clone(), Window::Train);
    }
    for session in &validation_sessions {
        session_window.insert(session.clone(), Window::Validation);
    }
    for session in &holdout_sessions {
        session_window.insert(session.clone(), Window::Holdout);
    }
    let assignments = events
        .iter()
        .map(|event| {
            session_window
                .get(&event.session_id)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("event session {} was not split", event.session_id))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        assignments,
        train_sessions,
        validation_sessions,
        holdout_sessions,
    ))
}

fn fixed_policy_metrics(
    events: &[Event],
    assignments: &[Window],
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> FixedPolicyMetrics {
    let normal = sampled_metrics(
        events,
        assignments,
        0.0,
        0,
        contract_multiplier,
        round_trip_cost,
    );
    let invert = sampled_metrics(
        events,
        assignments,
        1.0,
        0,
        contract_multiplier,
        round_trip_cost,
    );
    let oracle = oracle_metrics(events, assignments, contract_multiplier, round_trip_cost);
    FixedPolicyMetrics {
        normal,
        invert,
        two_class_oracle_pnl: oracle,
    }
}

fn sampled_metrics(
    events: &[Event],
    assignments: &[Window],
    probability_invert: f64,
    seed: u64,
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> [WindowMetrics; 3] {
    simulate_with_selector(
        events,
        assignments,
        seed,
        contract_multiplier,
        round_trip_cost,
        |rng, _event| rng.next_unit() < probability_invert,
    )
}

fn oracle_metrics(
    events: &[Event],
    assignments: &[Window],
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> [f64; 3] {
    let metrics = simulate_with_selector(
        events,
        assignments,
        0,
        contract_multiplier,
        round_trip_cost,
        |_rng, event| event.oracle_action < 0,
    );
    metrics.map(|metric| metric.pnl)
}

fn simulate_with_selector<F>(
    events: &[Event],
    assignments: &[Window],
    seed: u64,
    contract_multiplier: f64,
    round_trip_cost: f64,
    mut select_invert: F,
) -> [WindowMetrics; 3]
where
    F: FnMut(&mut SplitMix64, &Event) -> bool,
{
    let mut rng = SplitMix64::new(seed);
    let mut accumulators = [SplitAccumulator::empty(); 3];
    let mut current_session: Option<&str> = None;
    let mut position = 0_i8;
    for (event, window) in events.iter().zip(assignments) {
        if current_session != Some(event.session_id.as_str()) {
            current_session = Some(event.session_id.as_str());
            position = 0;
        }
        let invert = select_invert(&mut rng, event);
        let next_position = if invert {
            -event.raw_direction
        } else {
            event.raw_direction
        };
        let index = window.index();
        let pnl = interval_pnl(
            event,
            position,
            next_position,
            contract_multiplier,
            round_trip_cost,
        );
        accumulators[index].add(invert, pnl);
        position = next_position;
        if event.terminal_event {
            accumulators[index].add_pnl_only(transition_cost(position, 0, round_trip_cost));
            position = 0;
        }
    }
    [
        accumulators[0].finish(),
        accumulators[1].finish(),
        accumulators[2].finish(),
    ]
}

fn interval_pnl(
    event: &Event,
    from_position: i8,
    to_position: i8,
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> f64 {
    to_position as f64 * (event.interval_end_price - event.decision_price) * contract_multiplier
        + transition_cost(from_position, to_position, round_trip_cost)
}

fn transition_cost(from_position: i8, to_position: i8, round_trip_cost: f64) -> f64 {
    -((from_position - to_position).unsigned_abs() as f64 * round_trip_cost / 2.0)
}

fn position_index(position: i8) -> usize {
    match position {
        -1 => 0,
        0 => 1,
        1 => 2,
        _ => 1,
    }
}

fn probability_grid(step: f64) -> Vec<f64> {
    let count = (1.0 / step).ceil() as usize;
    (0..=count)
        .map(|index| (index as f64 * step).min(1.0))
        .collect()
}

fn expected_row(
    probability_invert: f64,
    events: &[Event],
    assignments: &[Window],
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> ExpectedSweepRow {
    let pnls = expected_pnl(
        probability_invert,
        events,
        assignments,
        contract_multiplier,
        round_trip_cost,
    );
    let positive_windows = pnls.iter().filter(|&&pnl| pnl > 0.0).count();
    ExpectedSweepRow {
        probability_invert,
        train_pnl: pnls[0],
        validation_pnl: pnls[1],
        holdout_pnl: pnls[2],
        all_pnl: pnls.iter().sum(),
        positive_windows,
        all_windows_positive: positive_windows == 3,
    }
}

/// Exact expected PnL of an independent normal/invert policy. This is not the
/// linear interpolation of always-normal and always-invert because each event
/// changes the current position and therefore future transition costs.
fn expected_pnl(
    probability_invert: f64,
    events: &[Event],
    assignments: &[Window],
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> [f64; 3] {
    let mut expected = [0.0; 3];
    let mut position_probabilities = [0.0, 1.0, 0.0]; // short, flat, long
    let positions = [-1_i8, 0_i8, 1_i8];
    let mut current_session: Option<&str> = None;
    for (event, window) in events.iter().zip(assignments) {
        if current_session != Some(event.session_id.as_str()) {
            current_session = Some(event.session_id.as_str());
            position_probabilities = [0.0, 1.0, 0.0];
        }
        let mut next_probabilities = [0.0; 3];
        let index = window.index();
        for (current_index, probability_position) in position_probabilities.into_iter().enumerate()
        {
            if probability_position == 0.0 {
                continue;
            }
            let current_position = positions[current_index];
            for (invert, choice_probability) in [
                (false, 1.0 - probability_invert),
                (true, probability_invert),
            ] {
                if choice_probability == 0.0 {
                    continue;
                }
                let next_position = if invert {
                    -event.raw_direction
                } else {
                    event.raw_direction
                };
                let probability = probability_position * choice_probability;
                expected[index] += probability
                    * interval_pnl(
                        event,
                        current_position,
                        next_position,
                        contract_multiplier,
                        round_trip_cost,
                    );
                next_probabilities[position_index(next_position)] += probability;
            }
        }
        position_probabilities = next_probabilities;
        if event.terminal_event {
            for (position_index, probability_position) in
                position_probabilities.into_iter().enumerate()
            {
                expected[index] += probability_position
                    * transition_cost(positions[position_index], 0, round_trip_cost);
            }
            position_probabilities = [0.0, 1.0, 0.0];
        }
    }
    expected
}

fn monte_carlo_row(
    probability_invert: f64,
    events: &[Event],
    assignments: &[Window],
    seed: u64,
    trials: usize,
    contract_multiplier: f64,
    round_trip_cost: f64,
) -> MonteCarloSweepRow {
    let mut values = [
        Vec::with_capacity(trials),
        Vec::with_capacity(trials),
        Vec::with_capacity(trials),
    ];
    let mut all_windows_positive = 0usize;
    for trial in 0..trials {
        let trial_seed = seed
            .wrapping_add((trial as u64).wrapping_mul(0xD1B54A32D192ED03))
            .wrapping_add(probability_invert.to_bits().rotate_left(17));
        let result = sampled_metrics(
            events,
            assignments,
            probability_invert,
            trial_seed,
            contract_multiplier,
            round_trip_cost,
        );
        if result.iter().all(|metrics| metrics.pnl > 0.0) {
            all_windows_positive += 1;
        }
        for index in 0..3 {
            values[index].push(result[index].pnl);
        }
    }
    MonteCarloSweepRow {
        probability_invert,
        train: distribution(&mut values[0]),
        validation: distribution(&mut values[1]),
        holdout: distribution(&mut values[2]),
        all_windows_positive_fraction: all_windows_positive as f64 / trials as f64,
    }
}

fn distribution(values: &mut [f64]) -> Distribution {
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Distribution {
        mean,
        p05: percentile(values, 0.05),
        p50: percentile(values, 0.50),
        p95: percentile(values, 0.95),
        positive_fraction: values.iter().filter(|&&value| value > 0.0).count() as f64
            / values.len() as f64,
    }
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        values[lower] + (values[upper] - values[lower]) * (position - lower as f64)
    }
}

fn print_report(
    args: &Args,
    event_count: usize,
    train_sessions: &[String],
    validation_sessions: &[String],
    holdout_sessions: &[String],
    fixed: &FixedPolicyMetrics,
    coin_flip: &[WindowMetrics; 3],
    expected: &[ExpectedSweepRow],
    monte_carlo: &[MonteCarloSweepRow],
) {
    println!("Input: {}", args.input.display());
    println!(
        "Events: {} | sessions: {} | split: {} train / {} validation / {} holdout",
        event_count,
        train_sessions.len() + validation_sessions.len() + holdout_sessions.len(),
        train_sessions.len(),
        validation_sessions.len(),
        holdout_sessions.len()
    );
    println!("Train sessions: {}", train_sessions.join(", "));
    println!("Validation sessions: {}", validation_sessions.join(", "));
    println!("Holdout sessions: {}", holdout_sessions.join(", "));

    println!(
        "\nFixed policies (sequential event replay with fees):\n{: <16} {:>14} {:>14} {:>14} {:>14}",
        "policy", "train", "validation", "holdout", "all"
    );
    for (label, metrics) in [
        ("always-normal", &fixed.normal),
        ("always-invert", &fixed.invert),
    ] {
        let all = metrics.iter().map(|metric| metric.pnl).sum::<f64>();
        println!(
            "{: <16} {:>14.2} {:>14.2} {:>14.2} {:>14.2}",
            label, metrics[0].pnl, metrics[1].pnl, metrics[2].pnl, all
        );
    }
    println!(
        "{: <16} {:>14.2} {:>14.2} {:>14.2} {:>14.2}",
        "2-class oracle",
        fixed.two_class_oracle_pnl[0],
        fixed.two_class_oracle_pnl[1],
        fixed.two_class_oracle_pnl[2],
        fixed.two_class_oracle_pnl.iter().sum::<f64>()
    );

    println!(
        "\nOne deterministic 50/50 coin flip (seed {}):\n{: <16} {:>8} {:>14} {:>12} {:>10}",
        args.seed, "window", "events", "pnl", "invert %", "win %"
    );
    for (window, metrics) in Window::ALL.into_iter().zip(coin_flip.iter()) {
        println!(
            "{: <16} {:>8} {:>14.2} {:>11.2}% {:>9.2}%",
            window.label(),
            metrics.events,
            metrics.pnl,
            metrics.invert_actions as f64 / metrics.events.max(1) as f64 * 100.0,
            metrics.win_rate * 100.0
        );
    }

    println!(
        "\nExpected weighted sweep (p = probability of invert; positive windows is out of 3):\n{:>8} {:>14} {:>14} {:>14} {:>14} {:>8} {:>8}",
        "p_inv", "train", "validation", "holdout", "all", "+win", "all+"
    );
    for row in expected {
        println!(
            "{:>7.0}% {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>8} {:>8}",
            row.probability_invert * 100.0,
            row.train_pnl,
            row.validation_pnl,
            row.holdout_pnl,
            row.all_pnl,
            row.positive_windows,
            if row.all_windows_positive {
                "yes"
            } else {
                "no"
            }
        );
    }

    println!(
        "\nMonte-Carlo distributions ({} trials per probability; p05 / median / p95; + is PnL > 0):",
        args.trials
    );
    println!(
        "{:>8} {:>39} {:>39} {:>39} {:>10}",
        "p_inv",
        "train mean | p05 / p50 / p95 | +",
        "validation mean | p05 / p50 / p95 | +",
        "holdout mean | p05 / p50 / p95 | +",
        "all+"
    );
    for row in monte_carlo {
        println!(
            "{:>7.0}% {:>8.0} | {:>8.0} / {:>8.0} / {:>8.0} | {:>5.1}% {:>8.0} | {:>8.0} / {:>8.0} / {:>8.0} | {:>5.1}% {:>8.0} | {:>8.0} / {:>8.0} / {:>8.0} | {:>5.1}% {:>9.2}%",
            row.probability_invert * 100.0,
            row.train.mean,
            row.train.p05,
            row.train.p50,
            row.train.p95,
            row.train.positive_fraction * 100.0,
            row.validation.mean,
            row.validation.p05,
            row.validation.p50,
            row.validation.p95,
            row.validation.positive_fraction * 100.0,
            row.holdout.mean,
            row.holdout.p05,
            row.holdout.p50,
            row.holdout.p95,
            row.holdout.positive_fraction * 100.0,
            row.all_windows_positive_fraction * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_grid_includes_endpoints() {
        assert_eq!(probability_grid(0.25), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.5), 2.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.25), 1.5);
    }
}

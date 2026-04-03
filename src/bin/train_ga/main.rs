#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod actions;
mod args;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod backends;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod behavior;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod config;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod data;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod evolution;
#[cfg(feature = "torch")]
mod ga;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod generation;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod logging;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod metrics;
#[cfg(feature = "torch")]
mod model;
mod paths;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod persistence;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod portable;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod runner;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod setup;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod types;
#[cfg(any(
    feature = "torch",
    feature = "backend-candle",
    feature = "backend-burn"
))]
mod util;

use clap::Parser;
use midas_env::ml::{self, MlBackend, TrainerKind};

use args::Args;

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();
    args.outdir = paths::resolve_outdir(args.outdir, "runs_ga");
    let stack = ml::resolve_training_stack(TrainerKind::Ga, &args.backend, &args.device)?;
    ml::print_training_stack(&stack);
    match stack.backend {
        MlBackend::Libtorch => run_libtorch(args, stack),
        MlBackend::Burn => run_burn(args, stack),
        MlBackend::Candle => run_candle(args, stack),
        MlBackend::Mlx => ml::ensure_backend_is_implemented(&stack),
    }
}

#[cfg(feature = "torch")]
fn run_libtorch(args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    runner::run(args, stack)
}

#[cfg(not(feature = "torch"))]
fn run_libtorch(_args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    anyhow::bail!(
        "backend '{}' was selected, but this build does not include the libtorch GA runner. Re-run with the 'torch' Cargo feature.",
        stack.backend
    )
}

#[cfg(feature = "backend-burn")]
fn run_burn(args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    runner::run(args, stack)
}

#[cfg(not(feature = "backend-burn"))]
fn run_burn(_args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    anyhow::bail!(
        "backend '{}' was selected, but this build does not include the Burn GA runner. Re-run with the 'backend-burn' Cargo feature.",
        stack.backend
    )
}

#[cfg(feature = "backend-candle")]
fn run_candle(args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    runner::run(args, stack)
}

#[cfg(not(feature = "backend-candle"))]
fn run_candle(_args: Args, stack: ml::ResolvedTrainingStack) -> anyhow::Result<()> {
    anyhow::bail!(
        "backend '{}' was selected, but this build does not include the Candle GA runner. Re-run with the 'backend-candle' Cargo feature.",
        stack.backend
    )
}

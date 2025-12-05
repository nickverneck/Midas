mod parquet_loader;

use anyhow::Result;
use clap::Parser;
use polars::prelude::DataFrame;
use std::{path::PathBuf, time::Instant};

/// Simple CLI for loading parquet files with Polars.
#[derive(Parser)]
#[command(name = "midas", version, about = "Load parquet data with Polars", long_about = None)]
struct Cli {
    /// Path to the parquet file to load.
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let timer = Instant::now();

    let df: DataFrame = parquet_loader::load_parquet(&args.file)?;
    let elapsed = timer.elapsed();

    println!("Loaded `{}`", args.file.display());
    println!("{df}");
    println!(
        "Finished in {:.9} seconds ({} µs) ({} rows × {} columns)",
        elapsed.as_secs_f64(),
        elapsed.as_micros(),
        df.height(),
        df.width()
    );

    Ok(())
}

//! Small schema-preserving parquet slicer for chronological train/validation/test splits.

use anyhow::{Context, Result, bail};
use clap::Parser;
use polars::prelude::{ParquetReader, ParquetWriter, SerReader};
use std::fs::File;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(about = "Write a timestamp-bounded slice of a parquet file")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Keep rows at or after this nanosecond timestamp.
    #[arg(long)]
    after_ts_ns: Option<i64>,
    /// Keep rows strictly before this nanosecond timestamp.
    #[arg(long)]
    before_ts_ns: Option<i64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.after_ts_ns.is_none() && args.before_ts_ns.is_none() {
        bail!("pass --after-ts-ns, --before-ts-ns, or both");
    }
    if let (Some(after), Some(before)) = (args.after_ts_ns, args.before_ts_ns) {
        if before <= after {
            bail!("--before-ts-ns must be greater than --after-ts-ns");
        }
    }
    if args.input == args.output {
        bail!("input and output must be different");
    }
    if args.output.exists() {
        bail!(
            "output {} already exists; refusing to clobber it",
            args.output.display()
        );
    }
    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }

    let input = File::open(&args.input)
        .with_context(|| format!("open input parquet {}", args.input.display()))?;
    let mut frame = ParquetReader::new(input)
        .finish()
        .with_context(|| format!("read input parquet {}", args.input.display()))?;
    let timestamp_name = if frame.get_column_names().iter().any(|name| *name == "ts_ns") {
        "ts_ns"
    } else if frame
        .get_column_names()
        .iter()
        .any(|name| *name == "timestamp_ns")
    {
        "timestamp_ns"
    } else {
        bail!("input must contain integer ts_ns or timestamp_ns timestamps");
    };
    let timestamps = frame
        .column(timestamp_name)
        .with_context(|| "input must contain integer ts_ns or timestamp_ns timestamps")?
        .as_materialized_series()
        .i64()
        .with_context(|| format!("input {timestamp_name} column must be Int64"))?;
    let start = args
        .after_ts_ns
        .and_then(|after| {
            timestamps
                .iter()
                .position(|value| value.is_some_and(|timestamp| timestamp >= after))
        })
        .unwrap_or(0);
    let end = args
        .before_ts_ns
        .and_then(|before| {
            timestamps
                .iter()
                .position(|value| value.is_some_and(|timestamp| timestamp >= before))
        })
        .unwrap_or(frame.height());
    if end <= start || end - start < 2 {
        bail!(
            "timestamp bounds leave fewer than two rows ({} rows)",
            end.saturating_sub(start)
        );
    }
    frame = frame.slice(start as i64, end - start);
    let output = File::create(&args.output)
        .with_context(|| format!("create output parquet {}", args.output.display()))?;
    ParquetWriter::new(output)
        .finish(&mut frame)
        .with_context(|| format!("write output parquet {}", args.output.display()))?;
    println!(
        "wrote {} rows from {} to {} between {:?} and {:?}",
        end - start,
        args.input.display(),
        args.output.display(),
        args.after_ts_ns,
        args.before_ts_ns
    );
    Ok(())
}

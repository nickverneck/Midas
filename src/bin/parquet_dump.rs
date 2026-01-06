use anyhow::Result;
use clap::Parser;
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Dump parquet rows to CSV for frontend visualization")]
struct Args {
    #[arg(long)]
    file: PathBuf,
    #[arg(long, default_value_t = 0)]
    limit: usize,
    #[arg(long, default_value_t = 0)]
    offset: usize,
    #[arg(long)]
    columns: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let file = std::fs::File::open(&args.file)?;
    let mut df = ParquetReader::new(file).finish()?;

    if df.column("row_idx").is_err() {
        let height = df.height();
        let row_idx: Vec<i64> = (0..height as i64).collect();
        df.with_column(Series::new("row_idx", row_idx))?;
    }

    if let Some(cols) = args.columns {
        let mut names: Vec<&str> = cols
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !names.iter().any(|name| *name == "row_idx") {
            names.insert(0, "row_idx");
        }
        df = df.select(names)?;
    }

    if args.offset > 0 || args.limit > 0 {
        let offset = args.offset.min(df.height());
        let remaining = df.height().saturating_sub(offset);
        let len = if args.limit == 0 {
            remaining
        } else {
            args.limit.min(remaining)
        };
        df = df.slice(offset as i64, len);
    }

    let mut stdout = std::io::stdout();
    CsvWriter::new(&mut stdout).has_header(true).finish(&mut df)?;
    Ok(())
}

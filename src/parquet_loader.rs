//! Utilities to read parquet files via Polars.

use polars::prelude::*;
use std::fs::File;
use std::path::Path;

/// Load the parquet file at the given path into a Polars `DataFrame`.
pub fn load_parquet(path: &Path) -> PolarsResult<DataFrame> {
    let file = File::open(path)?;
    ParquetReader::new(file).finish()
}

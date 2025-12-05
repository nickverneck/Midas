# Parquet + Technical Indicator Research
Updated: December 5, 2025

## Parquet ingestion
- **Pandas** already offers `read_parquet(engine='auto')`, which tries `pyarrow` first and falls back to `fastparquet`. Both engines understand directories, S3/GCS, and file-like objects, so data exported in partitioned Parquet is easy to load once the supporting engine is installed.
- **Polars** has a Rust-native reader exposed through `pl.read_parquet` (plus the ability to call `PolarsParquetReader` or `scan_parquet` for lazy pipelines). It mirrors Parquet’s columnar layout, so columnar filters or projections stay efficient without materializing the full dataset.
- **Fastparquet** is also available standalone and is already a recognized engine for Pandas; it exposes a `ParquetFile` abstraction if more control over metadata or selective row-group reads is needed.

## Technical indicators
- **PyIndicators** targets both Pandas and Polars frames. It ships a growing catalog of indicators (SMA, EMA, HMA, ATR, MACD, RSI, etc.) implemented in pure Python, so it can sit on top of whichever DataFrame backend you prefer without needing native extensions.
- **mintalib** focuses on performance via Cython kernels but exposes Pandas/Polars wrappers and expression builders for the likes of SMA, EMA, HMA, ATR, and others. It’s experimental, but useful if you want low-level control and the ability to compose expressions before materializing.

## Next steps
1. Decide whether to standardize on Pandas or Polars (or both) for signal creation, then pin the corresponding Parquet reader backend.
2. Evaluate PyIndicators and mintalib against a few sample datasets to see which API meshes best with the signal workflow and anticipated volume.

## Rust-native path
- **Polars** is written in Rust and mirrors Parquet’s columnar layout, so `pl.read_parquet` and `ParquetReader` keep parquet data in the same memory-friendly format as it leaves disk; the `parquet` feature exposes zero-copy readers, lazy execution, and parallelism that should feel familiar coming from pandas-style workflows while remaining idiomatic Rust. citeturn1search9
- **DataFusion** (Apache Arrow) exposes a `SessionContext::read_parquet` API that returns a DataFrame-like logical plan; it is similar to Pandas but lazy, so you can build transformations and then `collect` to execute queries on Arrow RecordBatches. citeturn1search1
- The lower-level **parquet** crate (via `parquet::arrow::ParquetFileArrowReader` and `ArrowWriter`) is still handy if you need more control over metadata, row-group filtering, or custom batching before feeding data into the analytic pipeline. citeturn1search2

- **YATA** supplies a broad set of moving averages (SMA, EMA family, HMA, DMA, etc.) implemented for Rust so you can keep all the signal computation in one language without hitting Python or C bindings. citeturn0search0turn0search2
- **velora_ta** focuses on streaming-safe and batch indicator calculations, covering SMA, EMA, HMA, ATR, and a long list of other trend/momentum/volatility tools that can be run point-by-point or over historical slices. citeturn0search1
- **ta-rs** includes SMA/EMA plus ATR, Bollinger Bands, CCI, RSI, MACD, ADL, Chandelier Exit, and serialization helpers to save computed indicator state back to disk if needed. citeturn0search7

### Immediate next steps
1. Pick the Rust data framework (Polars for eager/lazy DataFrame convenience or DataFusion/Arrow for SQL-style lazy plans and fine-grained batch control).
2. Prototype reading one of our Parquet datasets with that framework and then feed the column slice into YATA or velora_ta/ta-rs to validate the indicator APIs behave as expected for SMA, EMA, HMA, ATR.

## Benchmark snapshot
- **Polars** continues to lead the PDS-H benchmark in 2025 for both scale factors: the streaming engine clocked ~3.9s vs. DuckDB’s ~5.9s at SF-10, and while the in-memory engine slows at SF-100 (~152s vs. DuckDB’s 19.7s), the streaming mode remains within 1.2x of DuckDB and outpaces legacy engines by multiple orders of magnitude. citeturn1search0
- **DataFusion** has been hitting ClickBench records for Parquet workloads since the 43.0.0 release (Nov 18, 2024), with 45.0.0 (Feb 20, 2025) and 51.0.0 (Nov 25, 2025) each delivering incremental gains (e.g., faster CASE evaluation, remote Parquet defaults, metadata parsing) while still ranking at or near the top of the single-node Parquet leaderboard. citeturn1search1turn1search2turn1search8
- **YATA** publishes built-in benches for most moving averages and indicators; for example, SMA/EMA/CCI/HMA operations run in the single-digit nanoseconds per iteration range, with windowed variants (~10/100) staying below ~200ns. citeturn3search6
- **velora_ta** emphasises a streaming-first, zero-copy architecture with both real-time updates and batch mode support, but the docs do not expose explicit benchmark numbers yet, so you may need to profile it directly when comparing latency/throughput to YATA’s published ns/iter figures. citeturn4search0

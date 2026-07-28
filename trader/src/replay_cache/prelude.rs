use crate::broker::{Bar, BarKind, BarType, BrokerKind, CandleMode, ReplayDownloadCacheTarget};
use crate::config::TradingEnvironment;
use crate::replay_download::{DownloadWindow, HistoricalDownloadTelemetry, split_download_window};
use anyhow::{Context, Result, bail};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ConvertedType, Type as PhysicalType};
use parquet::errors::Result as ParquetResult;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{ChunkReader, Length};
use parquet::file::statistics::Statistics;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::fd::AsRawFd;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::FileExt;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const MANIFEST_FILE_NAME: &str = "manifest.json";
pub const MANIFEST_VERSION: u32 = 1;
pub const SERVER_BARS_SCHEMA_VERSION: u32 = 1;
pub const RAW_TICKS_SCHEMA_VERSION: u32 = 2;
const RAW_TICKS_LEGACY_SCHEMA_VERSION: u32 = 1;
/// Maximum rows retained by the Arrow writer before it flushes a Parquet row group.
pub const PARQUET_ROW_GROUP_ROWS: usize = 65_536;
/// Maximum source rows materialized into one Arrow `RecordBatch` while writing.
pub const PARQUET_WRITE_BATCH_ROWS: usize = 8_192;
/// Maximum rows decoded into one Arrow `RecordBatch` while streaming a cache file.
pub const PARQUET_READ_BATCH_ROWS: usize = 8_192;
const PARQUET_COMPRESSION_LABEL: &str = "snappy";
static CACHE_FILE_VERSION: AtomicU64 = AtomicU64::new(0);
const MANIFEST_LOCK_FILE_NAME: &str = ".manifest.lock";
const MANIFEST_LOCK_WAIT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawTickReadPolicy {
    LegacyV1SequenceIds,
    StrictV2ProviderIds,
}

use chrono::{DateTime, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[cfg(feature = "replay")]
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};

#[cfg(feature = "replay")]
static NEXT_REPLAY_DOWNLOAD_OPERATION_ID: OnceLock<AtomicU64> = OnceLock::new();

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReplayEngineMode {
    /// Original async bar loop retained during the deterministic-engine migration.
    #[default]
    Legacy,
    /// Market-time ordering is owned by the deterministic virtual event queue.
    Deterministic,
}

impl ReplayEngineMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Legacy => "Legacy compatibility",
            Self::Deterministic => "Deterministic virtual time",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayWindowSnapshot {
    pub preset: String,
    pub input_timezone: String,
    pub warmup_start: DateTime<Utc>,
    pub evaluation_start: DateTime<Utc>,
    pub evaluation_end: DateTime<Utc>,
    pub warmup_rows: usize,
    pub evaluation_rows_total: usize,
    pub evaluation_rows_processed: usize,
}

impl ReplayWindowSnapshot {
    pub fn local_range_label(&self) -> String {
        let timezone = self.input_timezone.parse::<Tz>().unwrap_or(chrono_tz::UTC);
        format!(
            "{} to {}",
            self.evaluation_start
                .with_timezone(&timezone)
                .format("%Y-%m-%d %H:%M:%S %Z"),
            self.evaluation_end
                .with_timezone(&timezone)
                .format("%Y-%m-%d %H:%M:%S %Z")
        )
    }

    pub fn evaluation_rows_remaining(&self) -> usize {
        self.evaluation_rows_total
            .saturating_sub(self.evaluation_rows_processed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDownloadOperationId(pub u64);

impl ReplayDownloadOperationId {
    #[cfg(feature = "replay")]
    pub fn next() -> Self {
        let counter = NEXT_REPLAY_DOWNLOAD_OPERATION_ID.get_or_init(|| {
            let epoch_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(1);
            let process_seed = u64::from(std::process::id()).rotate_left(32);
            AtomicU64::new((epoch_nanos ^ process_seed).max(1))
        });
        Self(counter.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayDownloadCacheTarget {
    pub dataset_dir: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplaySpeed {
    Realtime,
    X2,
    X5,
    X10,
    X25,
}

impl ReplaySpeed {
    pub fn label(self) -> &'static str {
        match self {
            Self::Realtime => "Realtime",
            Self::X2 => "2x",
            Self::X5 => "5x",
            Self::X10 => "10x",
            Self::X25 => "25x",
        }
    }

    #[cfg(feature = "replay")]
    pub fn multiplier(self) -> f64 {
        match self {
            Self::Realtime => 1.0,
            Self::X2 => 2.0,
            Self::X5 => 5.0,
            Self::X10 => 10.0,
            Self::X25 => 25.0,
        }
    }

    pub fn faster(self) -> Self {
        match self {
            Self::Realtime => Self::X2,
            Self::X2 => Self::X5,
            Self::X5 => Self::X10,
            Self::X10 => Self::X25,
            Self::X25 => Self::X25,
        }
    }

    pub fn slower(self) -> Self {
        match self {
            Self::Realtime => Self::Realtime,
            Self::X2 => Self::Realtime,
            Self::X5 => Self::X2,
            Self::X10 => Self::X5,
            Self::X25 => Self::X10,
        }
    }
}

impl Default for ReplaySpeed {
    fn default() -> Self {
        Self::Realtime
    }
}

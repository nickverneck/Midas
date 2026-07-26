use crate::broker::{BarType, BrokerKind};
use crate::config::TradingEnvironment;
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const PROBE_PLAN_VERSION: u32 = 1;
pub const PROBE_RESULT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadWindow {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl DownloadWindow {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Self> {
        if start >= end {
            bail!("download window requires start before end");
        }
        Ok(Self { start, end })
    }

    pub fn duration(self) -> ChronoDuration {
        self.end - self.start
    }

    pub fn contains(self, other: Self) -> bool {
        self.start <= other.start && self.end >= other.end
    }
}

pub fn plan_fixed_windows(window: DownloadWindow, max_duration: ChronoDuration) -> Result<Vec<DownloadWindow>> {
    if max_duration <= ChronoDuration::zero() {
        bail!("download chunk duration must be positive");
    }
    let mut windows = Vec::new();
    let mut start = window.start;
    while start < window.end {
        let end = start
            .checked_add_signed(max_duration)
            .unwrap_or(window.end)
            .min(window.end);
        windows.push(DownloadWindow { start, end });
        start = end;
    }
    Ok(windows)
}

pub fn split_download_window(
    window: DownloadWindow,
    session_boundaries: &[DateTime<Utc>],
    minimum: ChronoDuration,
) -> Option<(DownloadWindow, DownloadWindow)> {
    if minimum <= ChronoDuration::zero() || window.duration() < minimum * 2 {
        return None;
    }
    let midpoint = window.start + window.duration() / 2;
    let split = session_boundaries
        .iter()
        .copied()
        .filter(|candidate| {
            *candidate - window.start >= minimum && window.end - *candidate >= minimum
        })
        .min_by_key(|candidate| (*candidate - midpoint).num_milliseconds().abs())
        .unwrap_or(midpoint);
    if split - window.start < minimum || window.end - split < minimum {
        return None;
    }
    Some((
        DownloadWindow {
            start: window.start,
            end: split,
        },
        DownloadWindow {
            start: split,
            end: window.end,
        },
    ))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCapClassification {
    Yes,
    No,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadCompletionEvidence {
    EndOfHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoricalDownloadTelemetry {
    pub request: DownloadWindow,
    pub elapsed_ms: u64,
    pub wire_bytes: u64,
    pub packet_count: u64,
    pub provider_rows: u64,
    pub normalized_rows: u64,
    pub duplicate_rows: u64,
    pub dropped_rows: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_id: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub realtime_id: Option<i64>,
    pub end_of_history: bool,
    pub socket_closed: bool,
    pub timed_out: bool,
    pub cancellation_sent: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_evidence: Option<DownloadCompletionEvidence>,
    pub cap: DownloadCapClassification,
    pub cap_reason: String,
}

impl HistoricalDownloadTelemetry {
    pub fn new(request: DownloadWindow) -> Self {
        Self {
            request,
            elapsed_ms: 0,
            wire_bytes: 0,
            packet_count: 0,
            provider_rows: 0,
            normalized_rows: 0,
            duplicate_rows: 0,
            dropped_rows: 0,
            first_timestamp: None,
            last_timestamp: None,
            historical_id: None,
            realtime_id: None,
            end_of_history: false,
            socket_closed: false,
            timed_out: false,
            cancellation_sent: false,
            completion_evidence: None,
            cap: DownloadCapClassification::Unknown,
            cap_reason: "No independent provider-cap evidence was collected.".to_string(),
        }
    }

    pub fn completed_by_eoh(&self) -> bool {
        self.end_of_history
            && self.completion_evidence == Some(DownloadCompletionEvidence::EndOfHistory)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalDownloadFailureKind {
    Authentication,
    Authorization,
    RateLimited,
    Timeout,
    ClosedWithoutCompletion,
    SizeRejected,
    Transport,
    Provider,
    InvalidData,
    Cancelled,
}

impl HistoricalDownloadFailureKind {
    pub fn is_splittable(self) -> bool {
        matches!(
            self,
            Self::Timeout | Self::ClosedWithoutCompletion | Self::SizeRejected
        )
    }

    pub fn is_transient(self) -> bool {
        matches!(self, Self::RateLimited | Self::Timeout | Self::Transport)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoricalDownloadFailure {
    pub kind: HistoricalDownloadFailureKind,
    pub message: String,
    pub telemetry: HistoricalDownloadTelemetry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}

impl std::fmt::Display for HistoricalDownloadFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for HistoricalDownloadFailure {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeSource {
    RawTicks,
    ServerBars { bar_type: BarType },
}

impl ProbeSource {
    pub fn label(&self) -> String {
        match self {
            Self::RawTicks => "raw ticks".to_string(),
            Self::ServerBars { bar_type } => format!("{} server bars", bar_type.label()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadProbeSpec {
    pub probe_id: String,
    pub source: ProbeSource,
    pub window: DownloadWindow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_probe_id: Option<String>,
    #[serde(default)]
    pub split_depth: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadProbePlan {
    pub schema_version: u32,
    pub identity: String,
    pub provider: BrokerKind,
    pub env: TradingEnvironment,
    pub contract_symbol: String,
    pub contract_id: i64,
    pub created_at: DateTime<Utc>,
    pub cooldown_ms: u64,
    pub minimum_split_ms: i64,
    pub probes: Vec<DownloadProbeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadProbeResult {
    pub schema_version: u32,
    pub plan_identity: String,
    pub probe: DownloadProbeSpec,
    pub accepted: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<HistoricalDownloadFailureKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sanitized_error: Option<String>,
    pub telemetry: HistoricalDownloadTelemetry,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_read_rows: Option<u64>,
}

pub fn prepare_probe_directory(root: &Path, expected: &DownloadProbePlan) -> Result<DownloadProbePlan> {
    fs::create_dir_all(root).with_context(|| format!("create probe directory {}", root.display()))?;
    fs::create_dir_all(root.join("results"))
        .with_context(|| format!("create probe results directory {}", root.display()))?;
    let path = root.join("plan.json");
    if path.exists() {
        let stored: DownloadProbePlan = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        if stored.schema_version != PROBE_PLAN_VERSION || stored.identity != expected.identity {
            bail!(
                "probe plan identity mismatch in {}; use a new output directory",
                path.display()
            );
        }
        if stored.provider != expected.provider
            || stored.env != expected.env
            || stored.contract_symbol != expected.contract_symbol
            || stored.contract_id != expected.contract_id
        {
            bail!(
                "probe plan provider/environment/contract mismatch in {}; use a new output directory",
                path.display()
            );
        }
        return Ok(stored);
    }
    write_json_atomically(&path, expected)?;
    Ok(expected.clone())
}

pub fn verified_probe_result(root: &Path, plan: &DownloadProbePlan, probe_id: &str) -> Result<Option<DownloadProbeResult>> {
    let path = root.join("results").join(format!("{}.json", safe_id(probe_id)));
    if !path.exists() {
        return Ok(None);
    }
    let result: DownloadProbeResult = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    if result.schema_version != PROBE_RESULT_VERSION
        || result.plan_identity != plan.identity
        || result.probe.probe_id != probe_id
    {
        bail!("probe result identity mismatch in {}", path.display());
    }
    Ok(Some(result))
}

pub fn write_probe_result(root: &Path, result: &DownloadProbeResult) -> Result<PathBuf> {
    let path = root
        .join("results")
        .join(format!("{}.json", safe_id(&result.probe.probe_id)));
    write_json_atomically(&path, result)?;
    Ok(path)
}

pub fn write_probe_report(root: &Path, plan: &DownloadProbePlan) -> Result<PathBuf> {
    let mut report = String::from("# Tradovate Replay Download-Limit Probe Findings\n\n");
    report.push_str("This report is generated from sanitized, read-only market-data probe results. Missing rows are not interpreted as a provider cap without independent evidence.\n\n");
    report.push_str(&format!("- Plan identity: `{}`\n", plan.identity));
    report.push_str(&format!("- Environment: `{}`\n", plan.env.label()));
    report.push_str(&format!("- Contract: `{}` (`{}`)\n\n", plan.contract_symbol, plan.contract_id));
    report.push_str("| Probe | Source | UTC window `[start,end)` | Result | Rows | EOH | Cap | Elapsed | Bytes |\n");
    report.push_str("|---|---|---|---|---:|---|---|---:|---:|\n");
    for probe in &plan.probes {
        match verified_probe_result(root, plan, &probe.probe_id)? {
            Some(result) => report.push_str(&format!(
                "| `{}` | {} | `{}` to `{}` | {} | {} | {} | {:?} | {} ms | {} |\n",
                probe.probe_id,
                probe.source.label(),
                probe.window.start,
                probe.window.end,
                result.status,
                result.telemetry.normalized_rows,
                result.telemetry.end_of_history,
                result.telemetry.cap,
                result.telemetry.elapsed_ms,
                result.telemetry.wire_bytes,
            )),
            None => report.push_str(&format!(
                "| `{}` | {} | `{}` to `{}` | pending | - | - | unknown | - | - |\n",
                probe.probe_id,
                probe.source.label(),
                probe.window.start,
                probe.window.end,
            )),
        }
    }
    report.push_str("\n## Interpretation\n\nNo limit should be described as proven until a larger request fails or omits a range and bounded child requests explicitly complete that same range. Entitlement, environment, contract, and date remain part of every finding.\n");
    let path = root.join("report.md");
    write_atomically(&path, report.as_bytes())?;
    Ok(path)
}

fn safe_id(raw: &str) -> String {
    raw.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn write_json_atomically(path: &Path, value: &impl Serialize) -> Result<()> {
    write_atomically(path, &serde_json::to_vec_pretty(value)?)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("atomic output path has no parent")?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temp = parent.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().and_then(|value| value.to_str()).unwrap_or("output"),
        std::process::id(),
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)
        .with_context(|| format!("create {}", temp.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", temp.display()))?;
    file.sync_all()
        .with_context(|| format!("sync {}", temp.display()))?;
    fs::rename(&temp, path)
        .with_context(|| format!("replace {} from {}", path.display(), temp.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn dt(hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 23, hour, 0, 0)
            .single()
            .expect("timestamp")
    }

    #[test]
    fn fixed_windows_are_exact_half_open_and_cover_parent() {
        let parent = DownloadWindow::new(dt(0), dt(5)).expect("window");
        let windows = plan_fixed_windows(parent, ChronoDuration::hours(2)).expect("plan");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], DownloadWindow { start: dt(0), end: dt(2) });
        assert_eq!(windows[1].start, windows[0].end);
        assert_eq!(windows[2], DownloadWindow { start: dt(4), end: dt(5) });
    }

    #[test]
    fn split_prefers_nearest_valid_session_boundary() {
        let parent = DownloadWindow::new(dt(0), dt(4)).expect("window");
        let split = split_download_window(
            parent,
            &[dt(1), dt(3)],
            ChronoDuration::minutes(30),
        )
        .expect("split");
        assert_eq!(split.0.end, dt(1));
        assert_eq!(split.0.end, split.1.start);
        assert_eq!(split.1.end, parent.end);
    }

    #[test]
    fn split_refuses_children_below_minimum() {
        let parent = DownloadWindow::new(dt(0), dt(1)).expect("window");
        assert!(
            split_download_window(parent, &[], ChronoDuration::minutes(31)).is_none()
        );
    }
}

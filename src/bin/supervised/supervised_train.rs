use super::*;
use anyhow::{Context, Result, bail};
use polars::prelude::{DataFrame, DataType, ParquetReader};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "backend-candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "backend-candle")]
use candle_nn::{Module, VarBuilder, VarMap, linear, loss};

#[cfg(feature = "backend-burn")]
use burn::backend::Autodiff;
#[cfg(feature = "backend-burn")]
use burn::module::Param;
#[cfg(feature = "backend-burn")]
use burn::nn::loss::CrossEntropyLossConfig;
#[cfg(feature = "backend-burn")]
use burn::nn::{Linear, LinearConfig};
#[cfg(feature = "backend-burn")]
use burn::optim::GradientsParams;
#[cfg(feature = "backend-burn")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "backend-burn")]
use burn::tensor::{Int, Tensor as BurnTensor, TensorData};
#[cfg(feature = "backend-burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "backend-burn-mlx")]
use burn_mlx::{Mlx, MlxDevice};
#[cfg(feature = "backend-burn")]
use burn_ndarray::{NdArray, NdArrayDevice};

const POLICY_SCHEMA: &str = "supervised-policy-v2";
const LEGACY_POLICY_SCHEMA: &str = "supervised-policy-v1";
const SPLIT_PROVENANCE_SCHEMA: &str = "session-split-v1";
const RUN_COMPLETION_SCHEMA: &str = "supervised-training-completion-v1";
const RUN_COMPLETION_FILE: &str = "run.complete.json";
const ADAMW_BETA1: f64 = 0.9;
const ADAMW_BETA2: f64 = 0.999;
const ADAMW_EPSILON: f64 = 1e-8;

static ARTIFACT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
struct TrainingRow {
    session_id: String,
    timestamp_ns: i64,
    raw_direction: i8,
    label_action: i8,
    decision_price: f64,
    interval_end_price: f64,
    terminal_event: bool,
    features: Vec<f64>,
}

#[derive(Debug, Clone)]
struct TrainingDataset {
    feature_names: Vec<String>,
    config: SupervisedConfig,
    rows: Vec<TrainingRow>,
    source_hash: String,
    dataset_fingerprint: String,
}

/// The parquet writer emits a few source-bar and derived-index audit columns
/// in addition to `SupervisedEvent`.  Keep those columns in the integrity
/// payload too: hashing only the model feature map would allow an event row or
/// its source-bar snapshot to be edited while the claimed dataset fingerprint
/// remained unchanged.
#[derive(Debug, Clone, PartialEq)]
struct StoredEventPayload {
    event: SupervisedEvent,
    entry_row_idx: usize,
    event_open: f64,
    event_high: f64,
    event_low: f64,
    event_close: f64,
    event_volume: f64,
}

/// This is the exact provenance object emitted by the first-party dataset
/// preparer. `deny_unknown_fields` is intentional: training accepts only the
/// trusted event-dataset registry, rather than treating arbitrary JSON from an
/// externally supplied parquet as harmless metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct EventDatasetProvenance {
    #[serde(default)]
    source_path: Option<String>,
    timestamp_source: String,
    timestamp_unit: String,
    timestamp_timezone: String,
    index_timestamp_fallback: bool,
    raw_price_scale: Option<f64>,
    bar_kind: String,
    bar_value: f64,
    source_row_count: usize,
    source_hash_sha256: String,
    dataset_fingerprint_sha256: String,
}

/// `policy.json` and `metrics.json` are deliberately not the completion
/// signal. They are published separately because the existing run layout has
/// two files. Consumers must require this last-published marker and verify its
/// hashes before treating a run as complete; a crash or failed second publish
/// therefore cannot look like a valid pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct RunCompletionManifest {
    schema_version: String,
    status: String,
    policy_file: String,
    metrics_file: String,
    policy_sha256: String,
    metrics_sha256: String,
    dataset_fingerprint_sha256: String,
    total_epochs: usize,
}

/// An immutable, in-memory view of the three files that make a resumable run.
///
/// The files are first hard-linked into a private snapshot directory and read
/// from those snapshot names.  That makes the read independent of later
/// replacement of the caller-supplied paths; validation and deserialization
/// never go back to a mutable artifact path.
#[derive(Debug)]
struct ResumeSnapshot {
    policy_bytes: Vec<u8>,
}

struct PrivateResumeSnapshot {
    directory: PathBuf,
}

#[derive(Debug)]
struct ResolvedSource {
    path: PathBuf,
    metadata: std::fs::Metadata,
}

impl PrivateResumeSnapshot {
    fn create(parent: &Path) -> Result<Self> {
        for _ in 0..16 {
            let directory = parent.join(format!(
                ".midas-supervised-resume-{}-{}",
                std::process::id(),
                ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&directory) {
                Ok(()) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(
                            &directory,
                            std::fs::Permissions::from_mode(0o700),
                        )
                        .with_context(|| {
                            format!(
                                "restrict supervised resume snapshot {}",
                                directory.display()
                            )
                        })?;
                    }
                    return Ok(Self { directory });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "create private supervised resume snapshot {}",
                            directory.display()
                        )
                    });
                }
            }
        }
        bail!("could not allocate a private supervised resume snapshot directory")
    }

    fn read_artifact(&self, source: &Path, label: &str, slot: &str) -> Result<Vec<u8>> {
        let snapshot = self.directory.join(slot);

        // On Unix, link(2) does not dereference a source symlink.  If the
        // caller swaps a regular file for a symlink between the initial
        // inspection and this operation, the snapshot therefore contains the
        // symlink and require_regular_artifact rejects it.  Once linked, the
        // snapshot name is in a private directory and cannot be replaced by a
        // concurrent process during the read.
        #[cfg(unix)]
        std::fs::hard_link(source, &snapshot).with_context(|| {
            format!(
                "snapshot {label} {} without following a source symlink",
                source.display()
            )
        })?;

        #[cfg(not(unix))]
        std::fs::copy(source, &snapshot)
            .with_context(|| format!("snapshot {label} {}", source.display()))?;

        require_regular_artifact(&snapshot, label)?;
        let mut file = File::open(&snapshot)
            .with_context(|| format!("open snapshotted {label} {}", snapshot.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("read snapshotted {label} {}", snapshot.display()))?;
        Ok(bytes)
    }

    /// Copy a validated source through a no-follow file descriptor into this
    /// private directory.  The descriptor is opened only after the caller's
    /// root and symlink checks, and its identity is compared with the file
    /// that passed those checks.  This rejects a replacement of the source (or
    /// one of its parent directories) instead of silently parsing the new
    /// file.  Once copied, all source reads use the private snapshot path.
    fn snapshot_source(
        &self,
        source: &Path,
        validated_metadata: &std::fs::Metadata,
        label: &str,
        slot: &str,
    ) -> Result<PathBuf> {
        if validated_metadata.file_type().is_symlink() {
            bail!(
                "validated {label} {} became a symlink; refusing to follow it",
                source.display()
            );
        }
        if !validated_metadata.is_file() {
            bail!(
                "validated {label} {} is not a regular file",
                source.display()
            );
        }

        let mut source_file = open_no_follow(source, label)?;
        let opened_metadata = source_file
            .metadata()
            .with_context(|| format!("inspect opened {label} {}", source.display()))?;
        if !opened_metadata.is_file() {
            bail!("opened {label} {} is not a regular file", source.display());
        }
        if !same_file_identity(validated_metadata, &opened_metadata) {
            bail!(
                "validated {label} {} was replaced before its immutable snapshot could be created",
                source.display()
            );
        }

        let snapshot = self.directory.join(slot);
        let mut destination_options = OpenOptions::new();
        destination_options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            destination_options.mode(0o600);
        }
        let mut destination = destination_options
            .open(&snapshot)
            .with_context(|| format!("create private {label} snapshot {}", snapshot.display()))?;
        std::io::copy(&mut source_file, &mut destination)
            .with_context(|| format!("copy {label} into snapshot {}", snapshot.display()))?;
        destination
            .sync_all()
            .with_context(|| format!("sync private {label} snapshot {}", snapshot.display()))?;
        drop(destination);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&snapshot, std::fs::Permissions::from_mode(0o400))
                .with_context(|| format!("make private {label} snapshot read-only"))?;
        }
        require_regular_artifact(&snapshot, label)?;
        Ok(snapshot)
    }
}

#[cfg(unix)]
fn open_no_follow(path: &Path, label: &str) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    // O_NOFOLLOW is 00400000 on Linux and 0100 on the BSD/Darwin family.
    // Keep this local rather than adding a new crate dependency to the CLI.
    #[cfg(target_os = "linux")]
    const O_NOFOLLOW_FLAG: i32 = 0o400000;
    #[cfg(not(target_os = "linux"))]
    const O_NOFOLLOW_FLAG: i32 = 0o100;

    OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW_FLAG)
        .open(path)
        .with_context(|| format!("open {label} {} without following symlinks", path.display()))
}

#[cfg(not(unix))]
fn open_no_follow(path: &Path, label: &str) -> Result<File> {
    File::open(path)
        .with_context(|| format!("open {label} {} without following symlinks", path.display()))
}

#[cfg(unix)]
fn same_file_identity(first: &std::fs::Metadata, second: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    first.dev() == second.dev() && first.ino() == second.ino()
}

#[cfg(not(unix))]
fn same_file_identity(first: &std::fs::Metadata, second: &std::fs::Metadata) -> bool {
    first.len() == second.len() && first.modified().ok() == second.modified().ok()
}

impl Drop for PrivateResumeSnapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// The split is part of a policy's training provenance. Fractions alone are
/// not sufficient: rounding/clamping and a changed session ordering could
/// silently move events between train, validation, and holdout. Persisting
/// the exact session partitions makes resume reject that kind of drift.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SplitProvenance {
    schema_version: String,
    mode: String,
    train_fraction: f64,
    validation_fraction: f64,
    train_session_ids: Vec<String>,
    validation_session_ids: Vec<String>,
    holdout_session_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyArtifact {
    schema_version: String,
    dataset_schema: String,
    label_schema: String,
    trainer: String,
    backend: String,
    device: String,
    feature_names: Vec<String>,
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    class_names: Vec<String>,
    classes: usize,
    feature_count: usize,
    source_hash_sha256: String,
    dataset_fingerprint_sha256: String,
    config_json: String,
    seed: u64,
    #[serde(default)]
    training_epochs: usize,
    #[serde(default)]
    optimizer_state: Option<OptimizerStateArtifact>,
    /// Optional only so policies emitted by supervised-policy-v1 remain
    /// readable for evaluation. A resume always requires this field.
    #[serde(default)]
    split_provenance: Option<SplitProvenance>,
}

/// Backend-neutral AdamW continuation state. The moments use the policy's
/// canonical class-major layout: [class][feature], followed by one bias value
/// per class. Keeping this outside Burn/Candle tensor records means a policy
/// can be resumed on either backend without relying on backend-specific
/// parameter IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerStateArtifact {
    optimizer: String,
    step: usize,
    learning_rate: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_first_moment: Vec<f64>,
    weight_second_moment: Vec<f64>,
    bias_first_moment: Vec<f64>,
    bias_second_moment: Vec<f64>,
}

impl OptimizerStateArtifact {
    fn empty(feature_count: usize, learning_rate: f64, weight_decay: f64) -> Self {
        let weights = 3 * feature_count;
        Self {
            optimizer: "adamw-v1".to_string(),
            step: 0,
            learning_rate,
            weight_decay,
            beta1: ADAMW_BETA1,
            beta2: ADAMW_BETA2,
            epsilon: ADAMW_EPSILON,
            weight_first_moment: vec![0.0; weights],
            weight_second_moment: vec![0.0; weights],
            bias_first_moment: vec![0.0; 3],
            bias_second_moment: vec![0.0; 3],
        }
    }

    fn validate(&self, feature_count: usize) -> Result<()> {
        if self.optimizer != "adamw-v1"
            || self.weight_first_moment.len() != 3 * feature_count
            || self.weight_second_moment.len() != 3 * feature_count
            || self.bias_first_moment.len() != 3
            || self.bias_second_moment.len() != 3
            || self.step == 0
        {
            bail!("optimizer state shape or schema does not match the supervised policy");
        }
        if [
            self.learning_rate,
            self.weight_decay,
            self.beta1,
            self.beta2,
            self.epsilon,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
            || self.learning_rate == 0.0
            || self.beta1 >= 1.0
            || self.beta2 >= 1.0
            || self.epsilon == 0.0
        {
            bail!("optimizer state contains invalid hyperparameters");
        }
        if self
            .weight_first_moment
            .iter()
            .chain(self.weight_second_moment.iter())
            .chain(self.bias_first_moment.iter())
            .chain(self.bias_second_moment.iter())
            .any(|value| !value.is_finite())
        {
            bail!("optimizer state contains non-finite moment values");
        }
        Ok(())
    }
}

fn artifact_temp_path(destination: &Path) -> Result<PathBuf> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("artifact path has no valid file name"))?;
    let counter = ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(".{file_name}.tmp-{}-{counter}", std::process::id())))
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open artifact directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync artifact directory {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    // Directory fsync is not available through the portable std API on every
    // platform. The file itself is still flushed and synced below.
    Ok(())
}

/// Publish a JSON artifact without truncating an existing destination.
///
/// The temporary file is created beside the destination, written and synced,
/// then linked into place. `hard_link` is a no-replace operation on the
/// supported Unix targets, so an existing policy/checkpoint/metrics file
/// causes an error rather than being overwritten. The temp name is removed on
/// both success and failure paths.
fn publish_json_no_replace<T: Serialize>(destination: &Path, value: &T) -> Result<()> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create artifact directory {}", parent.display()))?;
    let temporary = artifact_temp_path(destination)?;
    let mut temporary_created = false;
    let result = (|| -> Result<()> {
        let bytes = serde_json::to_vec_pretty(value)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("create temporary artifact {}", temporary.display()))?;
        temporary_created = true;
        file.write_all(&bytes)
            .with_context(|| format!("write temporary artifact {}", temporary.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("terminate temporary artifact {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("sync temporary artifact {}", temporary.display()))?;
        drop(file);

        // Unlike rename, hard_link does not replace an existing destination.
        // Both names are in the same directory, making publication a single
        // directory-entry operation after the file contents are durable.
        std::fs::hard_link(&temporary, destination).with_context(|| {
            format!(
                "publish artifact {} without replacing an existing file",
                destination.display()
            )
        })?;
        sync_directory(parent)?;
        std::fs::remove_file(&temporary)
            .with_context(|| format!("remove temporary artifact {}", temporary.display()))?;
        sync_directory(parent)?;
        Ok(())
    })();

    if result.is_err() && temporary_created {
        // If linking succeeded but a later durability/cleanup step failed,
        // removing the temp name still leaves the published destination
        // intact. Ignore cleanup errors so the original failure is preserved.
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn publish_completed_run(
    policy_path: &Path,
    policy: &PolicyArtifact,
    metrics_path: &Path,
    metrics: &TrainingMetrics,
    completion_path: &Path,
    dataset_fingerprint: &str,
    total_epochs: usize,
) -> Result<()> {
    // These two files retain the existing layout and no-replace semantics.
    // The manifest is intentionally published last, after both files can be
    // read back and their top-level schemas have been checked.
    publish_json_no_replace(policy_path, policy)?;
    publish_json_no_replace(metrics_path, metrics)?;

    let policy_bytes = std::fs::read(policy_path)
        .with_context(|| format!("read published policy {}", policy_path.display()))?;
    let metrics_bytes = std::fs::read(metrics_path)
        .with_context(|| format!("read published metrics {}", metrics_path.display()))?;
    let policy_value: serde_json::Value =
        serde_json::from_slice(&policy_bytes).context("decode published supervised policy")?;
    let metrics_value: serde_json::Value =
        serde_json::from_slice(&metrics_bytes).context("decode published supervised metrics")?;
    if policy_value
        .get("schema_version")
        .and_then(|value| value.as_str())
        != Some(POLICY_SCHEMA)
    {
        bail!("published policy schema does not match {POLICY_SCHEMA}");
    }
    if metrics_value
        .get("policy_schema")
        .and_then(|value| value.as_str())
        != Some(POLICY_SCHEMA)
    {
        bail!("published metrics do not reference {POLICY_SCHEMA}");
    }

    let manifest = RunCompletionManifest {
        schema_version: RUN_COMPLETION_SCHEMA.to_string(),
        status: "complete".to_string(),
        policy_file: policy_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("policy path has no valid file name"))?
            .to_string(),
        metrics_file: metrics_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("metrics path has no valid file name"))?
            .to_string(),
        policy_sha256: super::sha256_file(policy_path)?,
        metrics_sha256: super::sha256_file(metrics_path)?,
        dataset_fingerprint_sha256: dataset_fingerprint.to_string(),
        total_epochs,
    };
    // If the process stops before this call, no completion marker exists and
    // consumers must treat the directory as incomplete.
    publish_json_no_replace(completion_path, &manifest)
}

fn ensure_artifact_absent(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => bail!(
            "refusing to overwrite existing supervised artifact {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("inspect supervised artifact {}", path.display()))
        }
    }
}

fn require_regular_artifact(path: &Path, label: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "{label} {} is a symlink; refusing to follow it",
            path.display()
        );
    }
    if !metadata.is_file() {
        bail!("{label} {} is not a regular file", path.display());
    }
    Ok(())
}

fn manifest_file_name(value: &str, label: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    let mut components = path.components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(path.to_path_buf()),
        _ => bail!("supervised completion manifest {label} `{value}` must be a single file name"),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

/// A policy is resumable only when it came from a fully published run.  The
/// manifest is the last publication step, so checking it here prevents a
/// direct CLI caller from resuming a policy left behind by a crashed or
/// partially copied run.  Keep this check in Rust as well as the HTTP API:
/// the CLI is a supported training entry point in its own right.
fn validate_resume_completion_manifest(policy_path: &Path) -> Result<ResumeSnapshot> {
    require_regular_artifact(policy_path, "resume policy")?;
    let requested_run_dir = policy_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let run_dir = std::fs::canonicalize(requested_run_dir).with_context(|| {
        format!(
            "canonicalize resume policy run directory {}",
            requested_run_dir.display()
        )
    })?;
    if !run_dir.is_dir() {
        bail!(
            "resume policy run directory {} is not a directory",
            run_dir.display()
        );
    }
    let policy_name = policy_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("resume policy has no valid file name"))?;
    // Resolve the final artifact name under the canonical directory before
    // snapshotting. A replacement of a symlinked parent path cannot redirect
    // the remainder of this validation to a different run.
    let stable_policy_path = run_dir.join(policy_name);
    require_regular_artifact(&stable_policy_path, "resume policy")?;
    let completion_path = run_dir.join(RUN_COMPLETION_FILE);
    require_regular_artifact(&completion_path, "supervised completion manifest")?;
    let snapshot = PrivateResumeSnapshot::create(&run_dir)?;
    let manifest_bytes = snapshot.read_artifact(
        &completion_path,
        "supervised completion manifest",
        "manifest.json",
    )?;
    let manifest: RunCompletionManifest =
        serde_json::from_slice(&manifest_bytes).context("decode supervised completion manifest")?;
    if manifest.schema_version != RUN_COMPLETION_SCHEMA || manifest.status != "complete" {
        bail!(
            "resume policy completion manifest is not a complete {RUN_COMPLETION_SCHEMA} artifact"
        );
    }

    let manifest_policy = manifest_file_name(&manifest.policy_file, "policy_file")?;
    if manifest_policy != Path::new(policy_name) {
        bail!(
            "completion manifest points to `{}`, not the requested resume policy `{policy_name}`",
            manifest.policy_file
        );
    }
    let metrics_name = manifest_file_name(&manifest.metrics_file, "metrics_file")?;
    let metrics_path = run_dir.join(metrics_name);
    require_regular_artifact(&metrics_path, "resume metrics")?;

    let policy_bytes =
        snapshot.read_artifact(&stable_policy_path, "resume policy", "policy.json")?;
    let metrics_bytes = snapshot.read_artifact(&metrics_path, "resume metrics", "metrics.json")?;
    let policy_hash = sha256_bytes(&policy_bytes);
    if policy_hash != manifest.policy_sha256 {
        bail!(
            "resume policy hash does not match the completion manifest; refusing an incomplete or modified run"
        );
    }
    let metrics_hash = sha256_bytes(&metrics_bytes);
    if metrics_hash != manifest.metrics_sha256 {
        bail!(
            "resume metrics hash does not match the completion manifest; refusing an incomplete or modified run"
        );
    }

    let policy_value: serde_json::Value =
        serde_json::from_slice(&policy_bytes).context("decode manifest-addressed resume policy")?;
    if policy_value
        .get("schema_version")
        .and_then(|value| value.as_str())
        != Some(POLICY_SCHEMA)
    {
        bail!("completion manifest addresses a policy with an unexpected schema");
    }
    if policy_value
        .get("dataset_fingerprint_sha256")
        .and_then(|value| value.as_str())
        != Some(manifest.dataset_fingerprint_sha256.as_str())
    {
        bail!("resume policy dataset fingerprint does not match the completion manifest");
    }
    let metrics_value: serde_json::Value = serde_json::from_slice(&metrics_bytes)
        .context("decode manifest-addressed resume metrics")?;
    if metrics_value
        .get("policy_schema")
        .and_then(|value| value.as_str())
        != Some(POLICY_SCHEMA)
    {
        bail!("completion manifest addresses metrics with an unexpected policy schema");
    }
    if metrics_value
        .get("dataset_fingerprint_sha256")
        .and_then(|value| value.as_str())
        != Some(manifest.dataset_fingerprint_sha256.as_str())
    {
        bail!("resume metrics dataset fingerprint does not match the completion manifest");
    }
    Ok(ResumeSnapshot { policy_bytes })
}

#[derive(Debug, Clone, Serialize)]
struct SplitMetrics {
    split: String,
    rows: usize,
    sessions: usize,
    accuracy: f64,
    cross_entropy: f64,
    macro_f1: f64,
    predicted_pnl: f64,
    oracle_pnl: f64,
    always_normal_pnl: f64,
    always_skip_pnl: f64,
    always_invert_pnl: f64,
    oracle_regret: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TrainingMetrics {
    schema_version: &'static str,
    policy_schema: &'static str,
    trainer: &'static str,
    backend: String,
    device: String,
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    seed: u64,
    feature_count: usize,
    feature_names: Vec<String>,
    source_hash_sha256: String,
    dataset_fingerprint_sha256: String,
    leakage_check: &'static str,
    split_mode: String,
    split_provenance: SplitProvenance,
    epochs_this_run: usize,
    start_epoch: usize,
    total_epochs: usize,
    checkpoint_every: usize,
    optimizer_state_saved: bool,
    checkpoints: Vec<CheckpointMetrics>,
    train: SplitMetrics,
    validation: SplitMetrics,
    holdout: SplitMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct CheckpointMetrics {
    epoch: usize,
    backend: String,
    device: String,
    policy_path: String,
    train: SplitMetrics,
    validation: SplitMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct EvaluationMetrics {
    schema_version: &'static str,
    policy_schema: &'static str,
    input: String,
    policy: String,
    feature_count: usize,
    leakage_check: &'static str,
    all: SplitMetrics,
}

fn build_policy(
    dataset: &TrainingDataset,
    split_provenance: &SplitProvenance,
    means: &[f64],
    scales: &[f64],
    weights: Vec<f64>,
    backend: &str,
    device: &str,
    seed: u64,
    training_epochs: usize,
    optimizer_state: Option<OptimizerStateArtifact>,
) -> Result<PolicyArtifact> {
    Ok(PolicyArtifact {
        schema_version: POLICY_SCHEMA.to_string(),
        dataset_schema: SUPERVISED_DATASET_SCHEMA.to_string(),
        label_schema: SUPERVISED_LABEL_SCHEMA.to_string(),
        trainer: "multiclass-logistic-regression".to_string(),
        backend: backend.to_string(),
        device: device.to_string(),
        feature_names: dataset.feature_names.clone(),
        means: means.to_vec(),
        scales: scales.to_vec(),
        weights,
        class_names: vec![
            "normal".to_string(),
            "skip".to_string(),
            "invert".to_string(),
        ],
        classes: 3,
        feature_count: dataset.feature_names.len(),
        source_hash_sha256: dataset.source_hash.clone(),
        dataset_fingerprint_sha256: dataset.dataset_fingerprint.clone(),
        config_json: serde_json::to_string(&dataset.config)?,
        seed,
        training_epochs,
        optimizer_state,
        split_provenance: Some(split_provenance.clone()),
    })
}

pub fn run_train(args: TrainArgs) -> Result<()> {
    let stack = midas_env::ml::resolve_training_stack(
        midas_env::ml::TrainerKind::Supervised,
        &args.backend,
        &args.device,
    )?;
    midas_env::ml::print_training_stack(&stack);
    midas_env::ml::ensure_backend_is_implemented(&stack)?;
    if args.epochs == 0 {
        bail!("epochs must be greater than zero");
    }
    // Snapshot and validate the source run before reading the training parquet
    // or using any policy bytes. This keeps the direct CLI contract identical
    // to the API: a policy without a complete, hash-matching run marker is
    // never a resumable input.
    let resume_snapshot = args
        .resume_policy
        .as_deref()
        .map(validate_resume_completion_manifest)
        .transpose()?;
    std::fs::create_dir_all(&args.outdir)?;
    let dataset = load_training_dataset(
        &args.input,
        args.source_root.as_deref(),
        args.allow_external_source,
    )?;
    let (train_indices, validation_indices, holdout_indices, split_mode, split_provenance) =
        session_splits(&dataset.rows, args.train_fraction, args.validation_fraction)?;
    let policy_path = args.outdir.join("policy.json");
    let metrics_path = args.outdir.join("metrics.json");
    let completion_path = args.outdir.join(RUN_COMPLETION_FILE);
    ensure_artifact_absent(&policy_path)?;
    ensure_artifact_absent(&metrics_path)?;
    ensure_artifact_absent(&completion_path)?;
    let resumed = if let Some(snapshot) = &resume_snapshot {
        let policy: PolicyArtifact =
            serde_json::from_slice(&snapshot.policy_bytes).context("decode resume policy")?;
        validate_policy(&policy, &dataset, true)?;
        validate_resume_compatibility(&policy, &args.backend, &split_provenance)?;
        Some(policy)
    } else {
        None
    };
    // A resumed run already has its initialization provenance. Keep that
    // seed in the artifact even if the caller supplies a different CLI seed;
    // loaded weights and optimizer state, rather than a new initializer, now
    // define the continuation.
    let training_seed = resumed
        .as_ref()
        .map(|policy| policy.seed)
        .unwrap_or(args.seed);
    let (means, scales) = resumed
        .as_ref()
        .map(|policy| (policy.means.clone(), policy.scales.clone()))
        .unwrap_or_else(|| fit_scaler(&dataset.rows, &train_indices, dataset.feature_names.len()));
    let feature_count = dataset.feature_names.len();
    let start_epoch = resumed
        .as_ref()
        .map(|policy| policy.training_epochs)
        .unwrap_or(0);
    let run_epochs = args.epochs.max(1);
    let end_epoch = start_epoch.saturating_add(run_epochs);
    let resumed_optimizer_state = resumed
        .as_ref()
        .and_then(|policy| policy.optimizer_state.clone());
    if let Some(state) = &resumed_optimizer_state {
        state.validate(feature_count)?;
        if (state.learning_rate - args.learning_rate).abs() > 1e-15
            || (state.weight_decay - args.l2).abs() > 1e-15
        {
            bail!(
                "resume optimizer state was created with learning_rate={} and l2={}, but this run requested learning_rate={} and l2={}; use the same optimizer configuration for exact continuation",
                state.learning_rate,
                state.weight_decay,
                args.learning_rate,
                args.l2
            );
        }
    }

    let mut checkpoints = Vec::new();
    let checkpoint_every = args.checkpoint_every;
    let backend_label = stack.backend.as_str().to_string();
    let mut checkpoint_callback = |epoch: usize,
                                   weights: &[f64],
                                   optimizer_state: Option<&OptimizerStateArtifact>,
                                   device: &'static str| {
        let checkpoint_path = format!("checkpoints/epoch-{epoch:08}/policy.json");
        let checkpoint_policy = build_policy(
            &dataset,
            &split_provenance,
            &means,
            &scales,
            weights.to_vec(),
            &backend_label,
            device,
            training_seed,
            epoch,
            optimizer_state.cloned(),
        )?;
        let checkpoint_file = args.outdir.join(&checkpoint_path);
        if let Some(parent) = checkpoint_file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        publish_json_no_replace(&checkpoint_file, &checkpoint_policy)?;
        checkpoints.push(CheckpointMetrics {
            epoch,
            backend: backend_label.clone(),
            device: device.to_string(),
            policy_path: checkpoint_path,
            train: metrics_for("train", &dataset, &train_indices, &checkpoint_policy),
            validation: metrics_for(
                "validation",
                &dataset,
                &validation_indices,
                &checkpoint_policy,
            ),
        });
        Ok::<(), anyhow::Error>(())
    };

    let (weights, actual_device, optimizer_state) = match stack.backend {
        midas_env::ml::MlBackend::CpuLinear => (
            train_linear_model(
                &dataset,
                &train_indices,
                &means,
                &scales,
                run_epochs,
                args.learning_rate,
                args.l2,
                resumed.as_ref().map(|policy| policy.weights.as_slice()),
                training_seed,
                start_epoch,
                end_epoch,
                checkpoint_every,
                &mut checkpoint_callback,
            )?,
            "cpu",
            None,
        ),
        midas_env::ml::MlBackend::Candle => {
            #[cfg(feature = "backend-candle")]
            {
                let (weights, device, state) = train_candle_model(
                    &dataset,
                    &train_indices,
                    &means,
                    &scales,
                    run_epochs,
                    args.learning_rate,
                    args.l2,
                    resumed.as_ref().map(|policy| policy.weights.as_slice()),
                    resumed_optimizer_state.as_ref(),
                    &args.device,
                    training_seed,
                    start_epoch,
                    end_epoch,
                    checkpoint_every,
                    &mut checkpoint_callback,
                )?;
                (weights, device, Some(state))
            }
            #[cfg(not(feature = "backend-candle"))]
            {
                bail!("Candle supervised training requires the `backend-candle` Cargo feature")
            }
        }
        midas_env::ml::MlBackend::Burn => {
            #[cfg(feature = "backend-burn")]
            {
                let (weights, device, state) = train_burn_model(
                    &dataset,
                    &train_indices,
                    &means,
                    &scales,
                    run_epochs,
                    args.learning_rate,
                    args.l2,
                    resumed.as_ref().map(|policy| policy.weights.as_slice()),
                    resumed_optimizer_state.as_ref(),
                    &args.device,
                    training_seed,
                    start_epoch,
                    end_epoch,
                    checkpoint_every,
                    &mut checkpoint_callback,
                )?;
                (weights, device, Some(state))
            }
            #[cfg(not(feature = "backend-burn"))]
            {
                bail!("Burn supervised training requires the `backend-burn` Cargo feature")
            }
        }
        backend => bail!("backend `{backend}` is not available for supervised training"),
    };
    let policy = build_policy(
        &dataset,
        &split_provenance,
        &means,
        &scales,
        weights,
        &backend_label,
        actual_device,
        training_seed,
        end_epoch,
        optimizer_state.clone(),
    )?;
    let metrics = TrainingMetrics {
        schema_version: "supervised-training-v1",
        policy_schema: POLICY_SCHEMA,
        trainer: "multiclass-logistic-regression",
        backend: policy.backend.clone(),
        device: policy.device.clone(),
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        l2: args.l2,
        seed: training_seed,
        feature_count: policy.feature_count,
        feature_names: policy.feature_names.clone(),
        source_hash_sha256: dataset.source_hash.clone(),
        dataset_fingerprint_sha256: dataset.dataset_fingerprint.clone(),
        leakage_check: "passed",
        split_mode,
        split_provenance: split_provenance.clone(),
        epochs_this_run: run_epochs,
        start_epoch,
        total_epochs: end_epoch,
        checkpoint_every,
        optimizer_state_saved: optimizer_state.is_some(),
        checkpoints,
        train: metrics_for("train", &dataset, &train_indices, &policy),
        validation: metrics_for("validation", &dataset, &validation_indices, &policy),
        holdout: metrics_for("holdout", &dataset, &holdout_indices, &policy),
    };
    publish_completed_run(
        &policy_path,
        &policy,
        &metrics_path,
        &metrics,
        &completion_path,
        &dataset.dataset_fingerprint,
        end_epoch,
    )?;
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}

pub fn run_evaluate(args: EvaluateArgs) -> Result<()> {
    let dataset = load_training_dataset(
        &args.input,
        args.source_root.as_deref(),
        args.allow_external_source,
    )?;
    let policy: PolicyArtifact = serde_json::from_slice(
        &std::fs::read(&args.policy)
            .with_context(|| format!("read policy {}", args.policy.display()))?,
    )
    .context("decode supervised policy")?;
    // Evaluation deliberately allows a new source/holdout artifact.  The
    // policy must still match the dataset schema, feature order, and causal
    // config, but provenance hashes are expected to differ across windows.
    validate_policy(&policy, &dataset, false)?;
    let indices = (0..dataset.rows.len()).collect::<Vec<_>>();
    let all = metrics_for("all", &dataset, &indices, &policy);
    let metrics = EvaluationMetrics {
        schema_version: "supervised-evaluation-v1",
        policy_schema: POLICY_SCHEMA,
        input: args.input.display().to_string(),
        policy: args.policy.display().to_string(),
        feature_count: dataset.feature_names.len(),
        leakage_check: "passed",
        all,
    };
    if let Some(out) = args.metrics.or(args.out) {
        publish_json_no_replace(&out, &metrics)?;
    }
    println!("{}", serde_json::to_string_pretty(&metrics)?);
    Ok(())
}

fn canonical_trusted_source_root(
    requested_root: Option<&Path>,
    allow_external_source: bool,
) -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("resolve supervised source working directory")?;
    let cwd = std::fs::canonicalize(&cwd).with_context(|| {
        format!(
            "canonicalize supervised source working directory {}",
            cwd.display()
        )
    })?;
    let root = requested_root.unwrap_or(cwd.as_path());
    let root = std::fs::canonicalize(root)
        .with_context(|| format!("canonicalize supervised source root {}", root.display()))?;
    if !root.is_dir() {
        bail!(
            "supervised source root {} is not a directory",
            root.display()
        );
    }
    if !root.starts_with(&cwd) && !allow_external_source {
        bail!(
            "supervised source root {} is outside the current working directory; pass --allow-external-source explicitly when an external trusted source root is required",
            root.display()
        );
    }
    Ok(root)
}

fn reject_parent_components(path: &Path) -> Result<()> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        bail!(
            "supervised source path `{}` contains `..`; use a contained path or pass a trusted --source-root and re-prepare the dataset",
            path.display()
        );
    }
    Ok(())
}

fn reject_symlink_components(path: &Path) -> Result<()> {
    let mut current = if path.is_absolute() {
        PathBuf::from(std::path::MAIN_SEPARATOR.to_string())
    } else {
        PathBuf::new()
    };
    for component in path.components() {
        match component {
            Component::RootDir => continue,
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::CurDir => continue,
            Component::ParentDir => {
                bail!("supervised source path `{}` contains `..`", path.display())
            }
            Component::Normal(part) => current.push(part),
        }
        if let Ok(metadata) = std::fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                bail!(
                    "supervised source path `{}` contains symlink component `{}`; refusing to follow source symlinks",
                    path.display(),
                    current.display()
                );
            }
        }
    }
    Ok(())
}

fn validate_contained_source(
    candidate: &Path,
    trusted_root: &Path,
) -> Result<Option<ResolvedSource>> {
    let metadata = match std::fs::symlink_metadata(candidate) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("inspect supervised source {}", candidate.display()));
        }
    };
    if metadata.file_type().is_symlink() {
        bail!(
            "supervised source {} is a symlink; refusing to follow source symlinks",
            candidate.display()
        );
    }
    if !metadata.is_file() {
        bail!(
            "supervised source {} is not a regular file",
            candidate.display()
        );
    }
    reject_parent_components(candidate)?;
    reject_symlink_components(candidate)?;
    let canonical = std::fs::canonicalize(candidate)
        .with_context(|| format!("canonicalize supervised source {}", candidate.display()))?;
    if !canonical.starts_with(trusted_root) {
        bail!(
            "supervised source {} resolves outside trusted source root {}; refusing provenance path",
            candidate.display(),
            trusted_root.display()
        );
    }
    let canonical_metadata = std::fs::symlink_metadata(&canonical).with_context(|| {
        format!(
            "inspect canonical supervised source {} after root validation",
            canonical.display()
        )
    })?;
    if canonical_metadata.file_type().is_symlink() || !canonical_metadata.is_file() {
        bail!(
            "canonical supervised source {} changed during root validation; refusing to follow it",
            canonical.display()
        );
    }
    Ok(Some(ResolvedSource {
        path: canonical,
        metadata: canonical_metadata,
    }))
}

fn resolve_source_path(
    dataset_path: &Path,
    recorded_path: &str,
    requested_root: Option<&Path>,
    allow_external_source: bool,
) -> Result<ResolvedSource> {
    let trusted_root = canonical_trusted_source_root(requested_root, allow_external_source)?;
    let recorded = PathBuf::from(recorded_path);
    reject_parent_components(&recorded)?;

    // The preparer records the CLI spelling.  Prefer the explicit trusted
    // root, then the dataset directory for portable relative artifacts, and
    // finally the process cwd for the documented workspace layout. Every
    // candidate is canonicalized and contained before it can be read.
    let mut candidates = Vec::with_capacity(3);
    if recorded.is_absolute() {
        candidates.push(recorded);
    } else {
        candidates.push(trusted_root.join(&recorded));
        if let Some(parent) = dataset_path.parent() {
            candidates.push(parent.join(&recorded));
        }
        candidates.push(recorded);
    }
    for candidate in candidates {
        if let Some(source) = validate_contained_source(&candidate, &trusted_root)? {
            return Ok(source);
        }
    }
    bail!(
        "supervised dataset source path `{recorded_path}` is unavailable under trusted source root {}; place the original Databento/NinjaTrader source under that root, or explicitly pass --source-root DIR --allow-external-source",
        trusted_root.display()
    )
}

fn timestamp_unit_hint(unit: &str) -> Option<&str> {
    match unit {
        "ns" | "us" | "ms" | "s" => Some(unit),
        // Datetime, text, date, and index sources carry their own parsing
        // semantics.  Passing their descriptive unit back to the numeric
        // parser would make an otherwise valid first-party source fail.
        _ => None,
    }
}

fn generated_event_payloads(
    events: &[SupervisedEvent],
    source: &LoadedSource,
) -> Result<Vec<StoredEventPayload>> {
    events
        .iter()
        .map(|event| {
            let row = event.row_idx;
            let bars = &source.bars;
            if row >= bars.close.len() {
                bail!(
                    "generated event {} refers to source row {} outside {} bars",
                    event.event_id,
                    row,
                    bars.close.len()
                );
            }
            Ok(StoredEventPayload {
                event: event.clone(),
                entry_row_idx: row.saturating_add(1),
                event_open: bars.open[row],
                event_high: bars.high[row],
                event_low: bars.low[row],
                event_close: bars.close[row],
                event_volume: bars.volume[row],
            })
        })
        .collect()
}

fn stored_event_payloads(
    df: &DataFrame,
    feature_names: &[String],
) -> Result<Vec<StoredEventPayload>> {
    let event_ids = usize_column(df, "event_id")?;
    let session_ids = required_string_column(df, "session_id")?;
    let session_event_indices = usize_column(df, "session_event_index")?;
    let row_indices = usize_column(df, "row_idx")?;
    let timestamps = timestamp_ns_column(df, "timestamp_ns")?;
    let entry_row_indices = usize_column(df, "entry_row_idx")?;
    let raw_directions = integer_column(df, "raw_direction")?;
    let decision_prices = float_column(df, "decision_price")?;
    let interval_end_prices = float_column(df, "interval_end_price")?;
    let interval_end_rows = usize_column(df, "interval_end_row_idx")?;
    let terminal_events = bool_column(df, "terminal_event")?;
    let event_opens = float_column(df, "event_open")?;
    let event_highs = float_column(df, "event_high")?;
    let event_lows = float_column(df, "event_low")?;
    let event_closes = float_column(df, "event_close")?;
    let event_volumes = float_column(df, "event_volume")?;
    let action_value_normals = float_column(df, "action_value_normal")?;
    let action_value_skips = float_column(df, "action_value_skip")?;
    let action_value_inverts = float_column(df, "action_value_invert")?;
    let label_actions = integer_column(df, "label_action")?;
    let label_names = required_string_column(df, "label_name")?;
    let oracle_positions_before = integer_column(df, "oracle_position_before")?;
    let oracle_positions_after = integer_column(df, "oracle_position_after")?;
    let oracle_values = float_column(df, "oracle_value")?;
    let feature_columns = feature_names
        .iter()
        .map(|name| float_column(df, name))
        .collect::<Result<Vec<_>>>()?;

    let row_count = event_ids.len();
    for (name, values) in [
        ("session_id", session_ids.len()),
        ("session_event_index", session_event_indices.len()),
        ("row_idx", row_indices.len()),
        ("timestamp_ns", timestamps.len()),
        ("entry_row_idx", entry_row_indices.len()),
        ("raw_direction", raw_directions.len()),
        ("decision_price", decision_prices.len()),
        ("interval_end_price", interval_end_prices.len()),
        ("interval_end_row_idx", interval_end_rows.len()),
        ("terminal_event", terminal_events.len()),
        ("event_open", event_opens.len()),
        ("event_high", event_highs.len()),
        ("event_low", event_lows.len()),
        ("event_close", event_closes.len()),
        ("event_volume", event_volumes.len()),
        ("action_value_normal", action_value_normals.len()),
        ("action_value_skip", action_value_skips.len()),
        ("action_value_invert", action_value_inverts.len()),
        ("label_action", label_actions.len()),
        ("label_name", label_names.len()),
        ("oracle_position_before", oracle_positions_before.len()),
        ("oracle_position_after", oracle_positions_after.len()),
        ("oracle_value", oracle_values.len()),
    ] {
        if values != row_count {
            bail!("dataset event column {name} length {values} does not match {row_count}");
        }
    }
    if feature_columns
        .iter()
        .any(|values| values.len() != row_count)
    {
        bail!("dataset feature column length does not match event rows");
    }

    let mut rows = Vec::with_capacity(row_count);
    for index in 0..row_count {
        let features = feature_names
            .iter()
            .zip(feature_columns.iter())
            .map(|(name, values)| (name.clone(), values[index]))
            .collect::<BTreeMap<_, _>>();
        rows.push(StoredEventPayload {
            event: SupervisedEvent {
                event_id: event_ids[index],
                session_id: session_ids[index].clone(),
                session_event_index: session_event_indices[index],
                row_idx: row_indices[index],
                timestamp_ns: timestamps[index],
                raw_direction: raw_directions[index],
                decision_price: decision_prices[index],
                interval_end_price: interval_end_prices[index],
                interval_end_row_idx: interval_end_rows[index],
                terminal_event: terminal_events[index],
                features,
                action_value_normal: action_value_normals[index],
                action_value_skip: action_value_skips[index],
                action_value_invert: action_value_inverts[index],
                label_action: label_actions[index],
                label_name: label_names[index].clone(),
                oracle_position_before: oracle_positions_before[index],
                oracle_position_after: oracle_positions_after[index],
                oracle_value: oracle_values[index],
            },
            entry_row_idx: entry_row_indices[index],
            event_open: event_opens[index],
            event_high: event_highs[index],
            event_low: event_lows[index],
            event_close: event_closes[index],
            event_volume: event_volumes[index],
        });
    }
    Ok(rows)
}

fn compare_event_payloads(
    stored: &[StoredEventPayload],
    regenerated: &[StoredEventPayload],
) -> Result<()> {
    if stored.len() != regenerated.len() {
        bail!(
            "supervised dataset event count {} does not match regenerated first-party event count {}; refusing modified event rows",
            stored.len(),
            regenerated.len()
        );
    }
    for (index, (stored, regenerated)) in stored.iter().zip(regenerated.iter()).enumerate() {
        if stored != regenerated {
            bail!(
                "supervised dataset event payload mismatch at row {} (event_id {}); a feature, value, label, index, or source-bar audit field was modified",
                index,
                stored.event.event_id
            );
        }
    }
    Ok(())
}

fn verify_source_hash(source_path: &Path, expected_hash: &str) -> Result<()> {
    let actual_hash = super::sha256_file(source_path)
        .with_context(|| format!("hash supervised dataset source {}", source_path.display()))?;
    if actual_hash != expected_hash {
        bail!(
            "supervised dataset source hash mismatch for {}; the original source was modified or the provenance is forged",
            source_path.display()
        );
    }
    Ok(())
}

fn source_snapshot_slot(source_path: &Path) -> String {
    source_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| format!("source.snapshot.{extension}"))
        .unwrap_or_else(|| "source.snapshot".to_string())
}

fn validate_recomputed_dataset_integrity(
    dataset_path: &Path,
    df: &DataFrame,
    config: &SupervisedConfig,
    feature_names: &[String],
    provenance: &EventDatasetProvenance,
    source_hash: &str,
    dataset_fingerprint: &str,
    recorded_source_path: &str,
    source_root: Option<&Path>,
    allow_external_source: bool,
) -> Result<()> {
    let resolved_source = resolve_source_path(
        dataset_path,
        recorded_source_path,
        source_root,
        allow_external_source,
    )?;
    let source_path = &resolved_source.path;

    // The path has passed the trusted-root and symlink-component checks.  Do
    // not reopen it for any integrity-sensitive operation, though: a writer
    // can replace a regular file or a parent directory immediately after
    // those checks.  The no-follow identity check in snapshot_source either
    // captures the validated inode or fails closed; all subsequent hashing,
    // parsing, and fingerprinting uses the private snapshot.
    let source_snapshot = PrivateResumeSnapshot::create(&std::env::temp_dir())?;
    let source_snapshot_path = source_snapshot.snapshot_source(
        source_path,
        &resolved_source.metadata,
        "supervised dataset source",
        &source_snapshot_slot(source_path),
    )?;
    verify_source_hash(&source_snapshot_path, source_hash)?;
    if provenance.source_hash_sha256 != source_hash {
        bail!("supervised provenance source hash does not match the dataset source hash");
    }
    if let Some(provenance_source_path) = provenance.source_path.as_deref()
        && provenance_source_path != recorded_source_path
    {
        bail!("supervised provenance source path disagrees with the source_path column");
    }

    let timestamp_unit = timestamp_unit_hint(&provenance.timestamp_unit);
    let regenerated_source = super::load_source(
        &source_snapshot_path,
        timestamp_unit,
        provenance.index_timestamp_fallback,
        &config.session_timezone,
        &config.bar_kind,
        config.bar_value,
        provenance.raw_price_scale,
    )
    .with_context(|| {
        format!(
            "re-read first-party supervised source {} for causal integrity validation",
            source_path.display()
        )
    })?;

    if regenerated_source.timestamp_source != provenance.timestamp_source
        || regenerated_source.timestamp_unit != provenance.timestamp_unit
        || regenerated_source.timestamp_timezone != provenance.timestamp_timezone
        || regenerated_source.index_timestamp_fallback != provenance.index_timestamp_fallback
        || regenerated_source.source_row_count != provenance.source_row_count
        || regenerated_source.raw_price_scale != provenance.raw_price_scale
    {
        bail!(
            "supervised source loader metadata differs from the stored provenance; re-prepare the dataset instead of trusting forged metadata"
        );
    }

    let regenerated_events = super::retain_complete_sessions(
        super::prepare_events(&regenerated_source.bars, config)
            .context("regenerate causal supervised events from the original source")?,
        &regenerated_source.bars,
        config,
    )?;
    if regenerated_events.is_empty() {
        bail!(
            "regenerated supervised source produced no complete-session events; refusing the dataset"
        );
    }
    let regenerated_fingerprint = super::dataset_fingerprint(
        &source_snapshot_path,
        &regenerated_source,
        config,
        &regenerated_events,
    )?;
    if regenerated_fingerprint != dataset_fingerprint
        || provenance.dataset_fingerprint_sha256 != regenerated_fingerprint
    {
        bail!(
            "supervised dataset fingerprint does not match events regenerated from the original source; refusing modified feature/value rows"
        );
    }

    let stored_payloads = stored_event_payloads(df, feature_names)?;
    let regenerated_payloads = generated_event_payloads(&regenerated_events, &regenerated_source)?;
    compare_event_payloads(&stored_payloads, &regenerated_payloads)
}

fn load_training_dataset(
    path: &Path,
    source_root: Option<&Path>,
    allow_external_source: bool,
) -> Result<TrainingDataset> {
    let df = ParquetReader::new(File::open(path)?).finish()?;
    let schema = required_string_column(&df, "schema_version")?;
    if schema
        .iter()
        .any(|value| value != SUPERVISED_DATASET_SCHEMA)
    {
        bail!("input is not a {SUPERVISED_DATASET_SCHEMA} parquet");
    }
    let label_schema = required_string_column(&df, "label_schema")?;
    if label_schema
        .iter()
        .any(|value| value != SUPERVISED_LABEL_SCHEMA)
    {
        bail!("input uses an unsupported label schema");
    }
    let feature_schema = required_consistent_string_column(&df, "feature_schema")?;
    let feature_names = feature_schema
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if feature_names.is_empty() {
        bail!("supervised dataset has no configured features");
    }
    reject_label_or_future_features(&feature_names)?;
    for name in &feature_names {
        if df.column(name).is_err() {
            bail!("dataset feature schema refers to missing column `{name}`");
        }
    }
    let config_json = required_consistent_string_column(&df, "config_json")?;
    let config_value: serde_json::Value =
        serde_json::from_str(&config_json).context("decode supervised config_json")?;
    let config: SupervisedConfig =
        serde_json::from_value(config_value.clone()).context("decode supervised config_json")?;
    config.validate()?;
    // The generated config is the feature registry. Re-serializing and
    // comparing the JSON value rejects omitted/defaulted or unknown fields
    // that serde would otherwise silently accept from an external parquet.
    if config_value != serde_json::to_value(&config)? {
        bail!(
            "config_json is not the exact first-party supervised feature/config registry; unknown or inconsistent metadata is not trusted"
        );
    }
    if config.feature_schema() != feature_schema {
        bail!("dataset feature_schema does not match its resolved config_json");
    }
    if config.feature_names() != feature_names {
        bail!("dataset feature_schema does not match the ordered feature registry");
    }
    let leakage_check = required_consistent_string_column(&df, "leakage_check")?;
    if leakage_check != "passed" {
        bail!(
            "dataset leakage_check must be `passed`; externally supplied or unverified event data is not trusted"
        );
    }
    let provenance_json = required_consistent_string_column(&df, "provenance_json")?;
    let provenance: EventDatasetProvenance = serde_json::from_str(&provenance_json)
        .context("decode first-party supervised provenance_json")?;
    let recorded_source_path = required_consistent_string_column(&df, "source_path")?;
    let source_hash_values = required_string_column(&df, "source_hash_sha256")?;
    let source_hash = source_hash_values.first().cloned().unwrap_or_default();
    if source_hash.is_empty() {
        bail!("dataset is missing source_hash_sha256");
    }
    if source_hash_values.iter().any(|value| value != &source_hash) {
        bail!("dataset contains mixed source hashes");
    }
    let dataset_fingerprint_values = required_string_column(&df, "dataset_fingerprint_sha256")?;
    let dataset_fingerprint = dataset_fingerprint_values
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("dataset is missing dataset_fingerprint_sha256"))?;
    if dataset_fingerprint.is_empty()
        || dataset_fingerprint_values
            .iter()
            .any(|value| value != &dataset_fingerprint)
    {
        bail!("dataset contains an invalid or mixed dataset fingerprint");
    }
    let timestamp_source = required_consistent_string_column(&df, "timestamp_source")?;
    let timestamp_unit = required_consistent_string_column(&df, "timestamp_unit")?;
    let timestamp_timezone = required_consistent_string_column(&df, "timestamp_timezone")?;
    let index_timestamp_fallback =
        required_consistent_bool_column(&df, "index_timestamp_fallback")?;
    let raw_price_scale = required_consistent_f64_column(&df, "raw_price_scale")?;
    let bar_kind = required_consistent_string_column(&df, "bar_kind")?;
    let bar_value = required_consistent_f64_column(&df, "bar_value")?;
    let source_row_count = required_consistent_i64_column(&df, "source_row_count")?;
    validate_event_dataset_provenance(
        &provenance,
        &config,
        &source_hash,
        &dataset_fingerprint,
        &timestamp_source,
        &timestamp_unit,
        &timestamp_timezone,
        index_timestamp_fallback,
        raw_price_scale,
        &bar_kind,
        bar_value,
        source_row_count,
    )?;
    validate_recomputed_dataset_integrity(
        path,
        &df,
        &config,
        &feature_names,
        &provenance,
        &source_hash,
        &dataset_fingerprint,
        &recorded_source_path,
        source_root,
        allow_external_source,
    )?;
    let session_ids = required_string_column(&df, "session_id")?;
    let raw_direction = integer_column(&df, "raw_direction")?;
    let label_action = integer_column(&df, "label_action")?;
    let decision_price = float_column(&df, "decision_price")?;
    let interval_end_price = float_column(&df, "interval_end_price")?;
    let terminal_event = bool_column(&df, "terminal_event")?;
    let timestamp_ns = timestamp_ns_column(&df, "timestamp_ns")?;
    let feature_columns = feature_names
        .iter()
        .map(|name| float_column(&df, name))
        .collect::<Result<Vec<_>>>()?;
    let row_count = session_ids.len();
    for (name, values) in [
        ("raw_direction", raw_direction.len()),
        ("label_action", label_action.len()),
        ("decision_price", decision_price.len()),
        ("interval_end_price", interval_end_price.len()),
        ("terminal_event", terminal_event.len()),
        ("timestamp_ns", timestamp_ns.len()),
    ] {
        if values != row_count {
            bail!("dataset column {name} length {values} does not match {row_count}");
        }
    }
    validate_event_timestamps(&timestamp_ns)?;
    let mut rows = Vec::with_capacity(row_count);
    for index in 0..row_count {
        let features = feature_columns
            .iter()
            .map(|values| values[index])
            .collect::<Vec<_>>();
        if features.iter().any(|value| !value.is_finite()) {
            bail!("non-finite training feature at row {index}");
        }
        if ![-1, 0, 1].contains(&label_action[index]) {
            bail!("label_action at row {index} must be -1, 0, or 1");
        }
        if ![-1, 1].contains(&raw_direction[index]) {
            bail!("raw_direction at row {index} must be -1 or 1");
        }
        rows.push(TrainingRow {
            session_id: session_ids[index].clone(),
            timestamp_ns: timestamp_ns[index],
            raw_direction: raw_direction[index],
            label_action: label_action[index],
            decision_price: decision_price[index],
            interval_end_price: interval_end_price[index],
            terminal_event: terminal_event[index],
            features,
        });
    }
    if rows.is_empty() {
        bail!("supervised dataset has zero events");
    }
    Ok(TrainingDataset {
        feature_names,
        config,
        rows,
        source_hash,
        dataset_fingerprint,
    })
}

fn validate_policy(
    policy: &PolicyArtifact,
    dataset: &TrainingDataset,
    require_same_provenance: bool,
) -> Result<()> {
    let legacy_schema = policy.schema_version == LEGACY_POLICY_SCHEMA;
    if policy.schema_version != POLICY_SCHEMA && !legacy_schema {
        bail!("unsupported policy schema {}", policy.schema_version);
    }
    if policy.dataset_schema != SUPERVISED_DATASET_SCHEMA {
        bail!("policy dataset schema does not match the supervised dataset");
    }
    if policy.label_schema != SUPERVISED_LABEL_SCHEMA {
        bail!("policy label schema does not match the supervised dataset");
    }
    let backend_supported = ["cpu-linear", "candle", "burn"].contains(&policy.backend.as_str());
    let device_supported = match policy.backend.as_str() {
        "burn" => ["cpu", "cuda", "mps"].contains(&policy.device.as_str()),
        _ => ["cpu", "cuda"].contains(&policy.device.as_str()),
    };
    if !backend_supported || !device_supported {
        bail!("policy backend/device is not supported by this trainer");
    }
    if policy.feature_names != dataset.feature_names {
        bail!("policy feature order does not match dataset feature_schema");
    }
    if policy.feature_count != dataset.feature_names.len() {
        bail!("policy feature_count does not match dataset feature_schema");
    }
    if policy.classes != 3
        || policy.class_names
            != vec![
                "normal".to_string(),
                "skip".to_string(),
                "invert".to_string(),
            ]
        || policy.means.len() != policy.feature_count
        || policy.scales.len() != policy.feature_count
        || policy.weights.len() != 3 * (policy.feature_count + 1)
    {
        bail!("policy class, scaler, or weight shape does not match dataset");
    }
    if policy
        .means
        .iter()
        .chain(policy.scales.iter())
        .chain(policy.weights.iter())
        .any(|value| !value.is_finite())
        || policy.scales.iter().any(|value| *value <= 0.0)
    {
        bail!("policy contains non-finite values or invalid scales");
    }
    if !legacy_schema && policy.split_provenance.is_none() {
        bail!("supervised-policy-v2 is missing split provenance");
    }
    if let Some(split_provenance) = &policy.split_provenance {
        validate_split_provenance(split_provenance)?;
    }
    if let Some(optimizer_state) = &policy.optimizer_state {
        if policy.backend == "cpu-linear" {
            bail!("cpu-linear policies cannot carry AdamW optimizer state");
        }
        optimizer_state.validate(policy.feature_count)?;
        if policy.training_epochs != optimizer_state.step {
            bail!(
                "policy training_epochs ({}) does not match optimizer step ({})",
                policy.training_epochs,
                optimizer_state.step
            );
        }
    }
    if require_same_provenance {
        if policy.source_hash_sha256 != dataset.source_hash {
            bail!("policy source hash does not match dataset");
        }
        if policy.dataset_fingerprint_sha256 != dataset.dataset_fingerprint {
            bail!("policy dataset fingerprint does not match dataset");
        }
    }
    let policy_config: SupervisedConfig =
        serde_json::from_str(&policy.config_json).context("decode policy config_json")?;
    policy_config.validate()?;
    if policy_config != dataset.config {
        bail!("policy config does not match dataset config");
    }
    Ok(())
}

fn validate_resume_compatibility(
    policy: &PolicyArtifact,
    requested_backend: &str,
    expected_split: &SplitProvenance,
) -> Result<()> {
    let requested_backend = midas_env::ml::MlBackend::parse(requested_backend)?.as_str();
    if policy.backend != requested_backend {
        bail!(
            "resume policy backend `{}` does not match requested backend `{requested_backend}`; backend changes are not supported for continuation",
            policy.backend
        );
    }
    if policy.schema_version != POLICY_SCHEMA {
        bail!(
            "legacy supervised policy schema {} is evaluation-only and cannot be resumed",
            policy.schema_version
        );
    }
    if policy.training_epochs == 0 {
        bail!("resume policy has no completed training epoch provenance; refusing to continue it");
    }
    let recorded_split = policy.split_provenance.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "resume policy has no split provenance; legacy policies can be evaluated but cannot be resumed"
        )
    })?;
    validate_split_provenance(recorded_split)?;
    if recorded_split != expected_split {
        bail!(
            "resume split provenance does not match requested train/validation fractions or exact session boundaries"
        );
    }

    match requested_backend {
        "cpu-linear" => {
            if policy.optimizer_state.is_some() {
                bail!("cpu-linear resume policy unexpectedly carries AdamW optimizer state");
            }
        }
        "burn" | "candle" => {
            let optimizer_state = policy.optimizer_state.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "{requested_backend} resume policy is missing AdamW optimizer state; refusing to reset optimizer provenance"
                )
            })?;
            optimizer_state.validate(policy.feature_count)?;
            if optimizer_state.step != policy.training_epochs {
                bail!(
                    "resume policy training_epochs ({}) does not match AdamW optimizer step ({})",
                    policy.training_epochs,
                    optimizer_state.step
                );
            }
        }
        other => bail!("backend `{other}` is not available for supervised training"),
    }
    // The recorded device is validated by `validate_policy`, while the
    // selected device is resolved by the concrete backend below. A same-
    // backend continuation may deliberately move tensors to that selected
    // device; all state crossing that boundary is host-serialized above.
    Ok(())
}

fn validate_split_provenance(provenance: &SplitProvenance) -> Result<()> {
    if provenance.schema_version != SPLIT_PROVENANCE_SCHEMA {
        bail!(
            "unsupported supervised split provenance schema {}",
            provenance.schema_version
        );
    }
    if provenance.mode != "session" {
        bail!(
            "unsupported supervised split mode {}; only session splits are resumable",
            provenance.mode
        );
    }
    if !(0.0..1.0).contains(&provenance.train_fraction)
        || !(0.0..1.0).contains(&provenance.validation_fraction)
        || provenance.train_fraction + provenance.validation_fraction >= 1.0
    {
        bail!("split provenance contains invalid train/validation fractions");
    }
    if provenance.train_session_ids.is_empty()
        || provenance.validation_session_ids.is_empty()
        || provenance.holdout_session_ids.is_empty()
    {
        bail!("split provenance must contain train, validation, and holdout sessions");
    }
    let mut sessions = BTreeSet::new();
    for session_id in provenance
        .train_session_ids
        .iter()
        .chain(provenance.validation_session_ids.iter())
        .chain(provenance.holdout_session_ids.iter())
    {
        if session_id.is_empty() || !sessions.insert(session_id) {
            bail!("split provenance contains an empty or overlapping session id");
        }
    }
    Ok(())
}

fn reject_label_or_future_features(names: &[String]) -> Result<()> {
    const FORBIDDEN_EXACT: [&str; 13] = [
        "label_action",
        "label_name",
        "oracle_value",
        "oracle_position_before",
        "oracle_position_after",
        "action_value_normal",
        "action_value_skip",
        "action_value_invert",
        "decision_price",
        "interval_end_price",
        "interval_end_row_idx",
        "terminal_event",
        "future_price",
    ];
    for name in names {
        if FORBIDDEN_EXACT.contains(&name.as_str())
            || name.contains("future")
            || name.contains("oracle")
            || name.contains("action_value")
        {
            bail!("feature `{name}` is a label/future-derived column and cannot be trained");
        }
    }
    Ok(())
}

fn validate_event_dataset_provenance(
    provenance: &EventDatasetProvenance,
    config: &SupervisedConfig,
    source_hash: &str,
    dataset_fingerprint: &str,
    timestamp_source: &str,
    timestamp_unit: &str,
    timestamp_timezone: &str,
    index_timestamp_fallback: bool,
    raw_price_scale: f64,
    bar_kind: &str,
    bar_value: f64,
    source_row_count: i64,
) -> Result<()> {
    if provenance.timestamp_source.trim().is_empty()
        || provenance.timestamp_unit.trim().is_empty()
        || provenance.timestamp_timezone.trim().is_empty()
    {
        bail!("supervised provenance contains an empty timestamp metadata field");
    }
    if provenance.timestamp_source != timestamp_source
        || provenance.timestamp_unit != timestamp_unit
        || provenance.timestamp_timezone != timestamp_timezone
        || provenance.index_timestamp_fallback != index_timestamp_fallback
    {
        bail!("supervised provenance_json disagrees with timestamp metadata columns");
    }
    if provenance.timestamp_timezone != config.session_timezone {
        bail!("supervised provenance timezone does not match config_json");
    }
    if provenance.bar_kind != bar_kind
        || provenance.bar_kind != config.bar_kind
        || (provenance.bar_value - bar_value).abs() > f64::EPSILON
        || (provenance.bar_value - config.bar_value).abs() > f64::EPSILON
    {
        bail!("supervised provenance bar metadata does not match the feature/config registry");
    }
    if provenance.source_row_count == 0
        || source_row_count <= 0
        || provenance.source_row_count != source_row_count as usize
    {
        bail!("supervised provenance source row count is inconsistent");
    }
    if provenance.source_hash_sha256 != source_hash
        || provenance.dataset_fingerprint_sha256 != dataset_fingerprint
    {
        bail!("supervised provenance hashes do not match the dataset metadata columns");
    }
    if !raw_price_scale.is_finite() || raw_price_scale <= 0.0 {
        bail!("raw_price_scale must be finite and positive");
    }
    let expected_raw_price_scale = provenance.raw_price_scale.unwrap_or(1.0);
    if !expected_raw_price_scale.is_finite()
        || expected_raw_price_scale <= 0.0
        || (expected_raw_price_scale - raw_price_scale).abs() > f64::EPSILON
    {
        bail!("supervised provenance raw price scale is inconsistent");
    }
    Ok(())
}

fn required_consistent_string_column(df: &DataFrame, name: &str) -> Result<String> {
    let values = required_string_column(df, name)?;
    let first = values
        .first()
        .ok_or_else(|| anyhow::anyhow!("supervised dataset is empty while reading {name}"))?;
    if first.trim().is_empty() || values.iter().any(|value| value != first) {
        bail!("dataset metadata column {name} is empty or inconsistent across event rows");
    }
    Ok(first.clone())
}

fn required_consistent_i64_column(df: &DataFrame, name: &str) -> Result<i64> {
    let values = integer_i64_column(df, name)?;
    let first = values
        .first()
        .ok_or_else(|| anyhow::anyhow!("supervised dataset is empty while reading {name}"))?;
    if values.iter().any(|value| value != first) {
        bail!("dataset metadata column {name} is inconsistent across event rows");
    }
    Ok(*first)
}

fn required_consistent_f64_column(df: &DataFrame, name: &str) -> Result<f64> {
    let values = float_column(df, name)?;
    let first = values
        .first()
        .ok_or_else(|| anyhow::anyhow!("supervised dataset is empty while reading {name}"))?;
    if values
        .iter()
        .any(|value| value.to_bits() != first.to_bits())
    {
        bail!("dataset metadata column {name} is inconsistent across event rows");
    }
    Ok(*first)
}

fn required_consistent_bool_column(df: &DataFrame, name: &str) -> Result<bool> {
    let values = bool_column(df, name)?;
    let first = values
        .first()
        .ok_or_else(|| anyhow::anyhow!("supervised dataset is empty while reading {name}"))?;
    if values.iter().any(|value| value != first) {
        bail!("dataset metadata column {name} is inconsistent across event rows");
    }
    Ok(*first)
}

fn required_string_column(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let column = df
        .column(name)
        .with_context(|| format!("dataset is missing column {name}"))?;
    column
        .as_materialized_series()
        .str()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .map(ToString::to_string)
                .ok_or_else(|| anyhow::anyhow!("column {name} has null at row {index}"))
        })
        .collect()
}

fn float_column(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let column = df
        .column(name)
        .with_context(|| format!("dataset is missing column {name}"))?;
    let cast = column.as_materialized_series().cast(&DataType::Float64)?;
    cast.f64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let value =
                value.ok_or_else(|| anyhow::anyhow!("column {name} has null at {index}"))?;
            if !value.is_finite() {
                bail!("column {name} has non-finite value at {index}");
            }
            Ok(value)
        })
        .collect()
}

fn integer_column(df: &DataFrame, name: &str) -> Result<Vec<i8>> {
    float_column(df, name)?
        .into_iter()
        .map(|value| {
            if value.fract() != 0.0 || value < i8::MIN as f64 || value > i8::MAX as f64 {
                bail!("column {name} has a non-integer value")
            }
            Ok(value as i8)
        })
        .collect()
}

fn integer_i64_column(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = df
        .column(name)
        .with_context(|| format!("dataset is missing column {name}"))?;
    let cast = column.as_materialized_series().cast(&DataType::Int64)?;
    cast.i64()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at {index}"))
        })
        .collect()
}

fn usize_column(df: &DataFrame, name: &str) -> Result<Vec<usize>> {
    integer_i64_column(df, name)?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            usize::try_from(value).with_context(|| {
                format!("column {name} has a negative or out-of-range index at row {index}")
            })
        })
        .collect()
}

fn timestamp_ns_column(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let column = df
        .column(name)
        .with_context(|| format!("dataset is missing generated event column {name}"))?;
    let series = column.as_materialized_series();
    if !matches!(
        series.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    ) {
        bail!("generated event column timestamp_ns must use an integer parquet type");
    }
    integer_i64_column(df, name)
}

fn validate_event_timestamps(timestamps: &[i64]) -> Result<()> {
    if timestamps.windows(2).any(|window| window[1] <= window[0]) {
        bail!(
            "timestamp_ns must be strictly increasing; refusing unsorted event rows before chronological session splitting"
        );
    }
    Ok(())
}

fn bool_column(df: &DataFrame, name: &str) -> Result<Vec<bool>> {
    let column = df
        .column(name)
        .with_context(|| format!("dataset is missing column {name}"))?;
    column
        .as_materialized_series()
        .bool()?
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            value.ok_or_else(|| anyhow::anyhow!("column {name} has null at {index}"))
        })
        .collect()
}

fn session_splits(
    rows: &[TrainingRow],
    train_fraction: f64,
    validation_fraction: f64,
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, String, SplitProvenance)> {
    if !(0.0..1.0).contains(&train_fraction)
        || !(0.0..1.0).contains(&validation_fraction)
        || train_fraction + validation_fraction >= 1.0
    {
        bail!("train/validation fractions must be positive and leave a holdout");
    }
    if rows.len() < 3 {
        bail!("supervised training requires at least three event rows");
    }
    if rows
        .windows(2)
        .any(|window| window[1].timestamp_ns <= window[0].timestamp_ns)
    {
        bail!("timestamp_ns must be strictly increasing before chronological session splitting");
    }
    let mut sessions = Vec::<String>::new();
    let mut seen = BTreeSet::new();
    for row in rows {
        if seen.insert(row.session_id.clone()) {
            sessions.push(row.session_id.clone());
        }
    }
    if sessions.len() < 3 {
        bail!(
            "supervised training requires at least three complete sessions; refusing overlapping event fallback"
        );
    }
    {
        let mut train_sessions = (sessions.len() as f64 * train_fraction).floor() as usize;
        let mut validation_sessions =
            (sessions.len() as f64 * validation_fraction).floor() as usize;
        train_sessions = train_sessions.clamp(1, sessions.len().saturating_sub(2));
        validation_sessions =
            validation_sessions.clamp(1, sessions.len().saturating_sub(train_sessions + 1));
        let train_set = sessions[..train_sessions].iter().collect::<BTreeSet<_>>();
        let validation_set = sessions[train_sessions..train_sessions + validation_sessions]
            .iter()
            .collect::<BTreeSet<_>>();
        let train_session_ids = sessions[..train_sessions].to_vec();
        let validation_session_ids =
            sessions[train_sessions..train_sessions + validation_sessions].to_vec();
        let holdout_session_ids = sessions[train_sessions + validation_sessions..].to_vec();
        let mut train = Vec::new();
        let mut validation = Vec::new();
        let mut holdout = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            if train_set.contains(&row.session_id) {
                train.push(index);
            } else if validation_set.contains(&row.session_id) {
                validation.push(index);
            } else {
                holdout.push(index);
            }
        }
        return Ok((
            train,
            validation,
            holdout,
            "session".to_string(),
            SplitProvenance {
                schema_version: SPLIT_PROVENANCE_SCHEMA.to_string(),
                mode: "session".to_string(),
                train_fraction,
                validation_fraction,
                train_session_ids,
                validation_session_ids,
                holdout_session_ids,
            },
        ));
    }
}

fn fit_scaler(
    rows: &[TrainingRow],
    indices: &[usize],
    feature_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; feature_count];
    for index in indices {
        for (feature, value) in rows[*index].features.iter().enumerate() {
            means[feature] += value;
        }
    }
    let denominator = indices.len().max(1) as f64;
    for mean in &mut means {
        *mean /= denominator;
    }
    let mut scales = vec![0.0; feature_count];
    for index in indices {
        for (feature, value) in rows[*index].features.iter().enumerate() {
            let delta = value - means[feature];
            scales[feature] += delta * delta;
        }
    }
    for scale in &mut scales {
        *scale = (*scale / denominator).sqrt().max(1e-9);
    }
    (means, scales)
}

fn class_index(label: i8) -> usize {
    match label {
        1 => 0,
        0 => 1,
        -1 => 2,
        _ => 1,
    }
}

fn scaled_features(row: &TrainingRow, means: &[f64], scales: &[f64]) -> Vec<f64> {
    row.features
        .iter()
        .enumerate()
        .map(|(index, value)| (value - means[index]) / scales[index])
        .collect()
}

fn logits(weights: &[f64], input: &[f64], classes: usize) -> Vec<f64> {
    let stride = input.len() + 1;
    (0..classes)
        .map(|class| {
            let offset = class * stride;
            weights[offset..offset + input.len()]
                .iter()
                .zip(input.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f64>()
                + weights[offset + input.len()]
        })
        .collect()
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = values
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f64>().max(1e-12);
    for value in &mut probabilities {
        *value /= sum;
    }
    probabilities
}

type CheckpointCallback<'a> =
    dyn FnMut(usize, &[f64], Option<&OptimizerStateArtifact>, &'static str) -> Result<()> + 'a;

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Generate the same small initial classifier for every backend. Candle's
/// CPU backend intentionally has no seedable RNG, so relying on backend
/// initialization would make `--seed` ineffective there. Explicit host-side
/// initialization keeps Burn, Candle, and the reference path comparable.
fn deterministic_initial_weights(feature_count: usize, seed: u64) -> Vec<f64> {
    let stride = feature_count + 1;
    let mut state = seed;
    let mut weights = vec![0.0; 3 * stride];
    for class in 0..3 {
        let offset = class * stride;
        for feature in 0..feature_count {
            let unit = splitmix64_next(&mut state) as f64 / u64::MAX as f64;
            weights[offset + feature] = (unit * 2.0 - 1.0) * 0.01;
        }
    }
    weights
}

fn checkpoint_due(checkpoint_every: usize, absolute_epoch: usize, final_epoch: usize) -> bool {
    checkpoint_every > 0
        && (absolute_epoch % checkpoint_every == 0 || absolute_epoch == final_epoch)
}

fn train_linear_model(
    dataset: &TrainingDataset,
    train_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    initial_weights: Option<&[f64]>,
    seed: u64,
    start_epoch: usize,
    final_epoch: usize,
    checkpoint_every: usize,
    checkpoint_callback: &mut CheckpointCallback<'_>,
) -> Result<Vec<f64>> {
    if train_indices.is_empty() {
        bail!("training split is empty");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
        bail!("learning_rate must be positive and l2 must be non-negative");
    }
    let classes = 3;
    let stride = dataset.feature_names.len() + 1;
    let mut weights = initial_weights
        .map(|weights| weights.to_vec())
        .unwrap_or_else(|| deterministic_initial_weights(dataset.feature_names.len(), seed));
    if weights.len() != classes * stride {
        bail!("resume policy weight shape does not match feature count");
    }
    for local_epoch in 0..epochs.max(1) {
        let mut gradients = vec![0.0; weights.len()];
        for index in train_indices {
            let row = &dataset.rows[*index];
            let input = scaled_features(row, means, scales);
            let probabilities = softmax(&logits(&weights, &input, classes));
            let target = class_index(row.label_action);
            for class in 0..classes {
                let error = probabilities[class] - if class == target { 1.0 } else { 0.0 };
                let offset = class * stride;
                for (feature, value) in input.iter().enumerate() {
                    gradients[offset + feature] += error * value;
                }
                gradients[offset + dataset.feature_names.len()] += error;
            }
        }
        let denominator = train_indices.len() as f64;
        for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
            *weight -= learning_rate * (gradient / denominator + l2 * *weight);
        }
        let absolute_epoch = start_epoch + local_epoch + 1;
        if checkpoint_due(checkpoint_every, absolute_epoch, final_epoch) {
            checkpoint_callback(absolute_epoch, &weights, None, "cpu")?;
        }
    }
    Ok(weights)
}

#[cfg(feature = "backend-candle")]
fn train_candle_model(
    dataset: &TrainingDataset,
    train_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    initial_weights: Option<&[f64]>,
    initial_optimizer_state: Option<&OptimizerStateArtifact>,
    requested_device: &str,
    seed: u64,
    start_epoch: usize,
    final_epoch: usize,
    checkpoint_every: usize,
    checkpoint_callback: &mut CheckpointCallback<'_>,
) -> Result<(Vec<f64>, &'static str, OptimizerStateArtifact)> {
    let feature_count = dataset.feature_names.len();
    if train_indices.is_empty() {
        bail!("training split is empty");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
        bail!("learning_rate must be positive and l2 must be non-negative");
    }

    let mut optimizer_state = initial_optimizer_state
        .cloned()
        .unwrap_or_else(|| OptimizerStateArtifact::empty(feature_count, learning_rate, l2));
    optimizer_state.validate(feature_count).or_else(|error| {
        if initial_optimizer_state.is_some() {
            Err(error)
        } else {
            Ok(())
        }
    })?;

    // Candle's CPU backend intentionally cannot seed its RNG. Seed before
    // construction where supported, then overwrite the initializer with the
    // same explicit host-generated weights used by every backend.
    let (device, device_label) = candle_training_device(requested_device)?;
    let _ = device.set_seed(seed);
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let classifier = linear(feature_count, 3, vb.pp("classifier"))?;

    let policy_weights = initial_weights
        .map(|weights| weights.to_vec())
        .unwrap_or_else(|| deterministic_initial_weights(feature_count, seed));
    let stride = feature_count + 1;
    if policy_weights.len() != 3 * stride {
        bail!("resume policy weight shape does not match feature count");
    }
    let mut weights = Vec::with_capacity(3 * feature_count);
    let mut biases = Vec::with_capacity(3);
    for class in 0..3 {
        let offset = class * stride;
        weights.extend(
            policy_weights[offset..offset + feature_count]
                .iter()
                .map(|value| *value as f32),
        );
        biases.push(policy_weights[offset + feature_count] as f32);
    }
    varmap.set_one(
        "classifier.weight",
        Tensor::from_vec(weights, (3, feature_count), &device)?,
    )?;
    varmap.set_one("classifier.bias", Tensor::from_vec(biases, 3, &device)?)?;

    let (weight_var, bias_var) = {
        let data = varmap.data().lock().unwrap();
        (
            data.get("classifier.weight")
                .cloned()
                .context("Candle classifier weight variable is missing")?,
            data.get("classifier.bias")
                .cloned()
                .context("Candle classifier bias variable is missing")?,
        )
    };
    let mut weight_first_moment = Tensor::from_vec(
        optimizer_state
            .weight_first_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        (3, feature_count),
        &device,
    )?;
    let mut weight_second_moment = Tensor::from_vec(
        optimizer_state
            .weight_second_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        (3, feature_count),
        &device,
    )?;
    let mut bias_first_moment = Tensor::from_vec(
        optimizer_state
            .bias_first_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        3,
        &device,
    )?;
    let mut bias_second_moment = Tensor::from_vec(
        optimizer_state
            .bias_second_moment
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>(),
        3,
        &device,
    )?;

    let mut features = Vec::with_capacity(train_indices.len() * feature_count);
    let mut targets = Vec::with_capacity(train_indices.len());
    for index in train_indices {
        let row = &dataset.rows[*index];
        features.extend(
            scaled_features(row, means, scales)
                .into_iter()
                .map(|value| value as f32),
        );
        targets.push(class_index(row.label_action) as u32);
    }
    let inputs = Tensor::from_vec(features, (train_indices.len(), feature_count), &device)?;
    let targets = Tensor::from_vec(targets, train_indices.len(), &device)?;
    for local_epoch in 0..epochs.max(1) {
        let logits = classifier.forward(&inputs)?;
        let objective = loss::cross_entropy(&logits, &targets)?;
        let gradients = objective.backward()?;
        let weight_gradient = gradients
            .get(weight_var.as_tensor())
            .context("Candle classifier weight gradient is missing")?;
        let bias_gradient = gradients
            .get(bias_var.as_tensor())
            .context("Candle classifier bias gradient is missing")?;

        optimizer_state.step += 1;
        let step = optimizer_state.step as i32;
        let factor_1 = 1.0 - optimizer_state.beta1;
        let factor_2 = 1.0 - optimizer_state.beta2;
        weight_first_moment =
            ((&weight_first_moment * optimizer_state.beta1)? + (weight_gradient * factor_1)?)?;
        weight_second_moment = ((&weight_second_moment * optimizer_state.beta2)?
            + (weight_gradient.sqr()? * factor_2)?)?;
        bias_first_moment =
            ((&bias_first_moment * optimizer_state.beta1)? + (bias_gradient * factor_1)?)?;
        bias_second_moment =
            ((&bias_second_moment * optimizer_state.beta2)? + (bias_gradient.sqr()? * factor_2)?)?;

        let scale_1 = 1.0 / (1.0 - optimizer_state.beta1.powi(step));
        let scale_2 = 1.0 / (1.0 - optimizer_state.beta2.powi(step));
        let weight_update = ((&weight_first_moment * scale_1)?
            / ((&weight_second_moment * scale_2)?.sqrt()? + optimizer_state.epsilon)?)?;
        let bias_update = ((&bias_first_moment * scale_1)?
            / ((&bias_second_moment * scale_2)?.sqrt()? + optimizer_state.epsilon)?)?;
        let next_weight = ((weight_var.as_tensor()
            * (1.0 - optimizer_state.learning_rate * optimizer_state.weight_decay))?
            - (&weight_update * optimizer_state.learning_rate)?)?;
        let next_bias = ((bias_var.as_tensor()
            * (1.0 - optimizer_state.learning_rate * optimizer_state.weight_decay))?
            - (&bias_update * optimizer_state.learning_rate)?)?;
        weight_var.set(&next_weight)?;
        bias_var.set(&next_bias)?;

        let absolute_epoch = start_epoch + local_epoch + 1;
        if checkpoint_due(checkpoint_every, absolute_epoch, final_epoch) {
            let packed = candle_policy_weights(&classifier, feature_count)?;
            let mut checkpoint_optimizer_state = optimizer_state.clone();
            checkpoint_optimizer_state.weight_first_moment = weight_first_moment
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .map(|value| value as f64)
                .collect();
            checkpoint_optimizer_state.weight_second_moment = weight_second_moment
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .map(|value| value as f64)
                .collect();
            checkpoint_optimizer_state.bias_first_moment = bias_first_moment
                .to_vec1::<f32>()?
                .into_iter()
                .map(|value| value as f64)
                .collect();
            checkpoint_optimizer_state.bias_second_moment = bias_second_moment
                .to_vec1::<f32>()?
                .into_iter()
                .map(|value| value as f64)
                .collect();
            checkpoint_callback(
                absolute_epoch,
                &packed,
                Some(&checkpoint_optimizer_state),
                device_label,
            )?;
        }
    }

    let packed = candle_policy_weights(&classifier, feature_count)?;
    optimizer_state.weight_first_moment = weight_first_moment
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .map(|value| value as f64)
        .collect();
    optimizer_state.weight_second_moment = weight_second_moment
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .map(|value| value as f64)
        .collect();
    optimizer_state.bias_first_moment = bias_first_moment
        .to_vec1::<f32>()?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    optimizer_state.bias_second_moment = bias_second_moment
        .to_vec1::<f32>()?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    Ok((packed, device_label, optimizer_state))
}

#[cfg(feature = "backend-candle")]
fn candle_policy_weights(classifier: &candle_nn::Linear, feature_count: usize) -> Result<Vec<f64>> {
    let weight = classifier
        .weight()
        .to_vec2::<f32>()
        .context("read Candle classifier weights")?;
    let bias = classifier
        .bias()
        .context("Candle classifier is missing a bias")?
        .to_vec1::<f32>()
        .context("read Candle classifier bias")?;
    if weight.len() != 3 || weight.iter().any(|row| row.len() != feature_count) {
        bail!("Candle classifier weight shape does not match feature count");
    }
    let mut packed = Vec::with_capacity(3 * (feature_count + 1));
    for class in 0..3 {
        packed.extend(weight[class].iter().map(|value| *value as f64));
        packed.push(bias[class] as f64);
    }
    Ok(packed)
}

#[cfg(feature = "backend-candle")]
fn candle_training_device(requested: &str) -> Result<(Device, &'static str)> {
    match requested.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => {
            #[cfg(feature = "backend-candle-cuda")]
            if midas_env::ml::candle_cuda::auto_is_allowed() {
                match Device::new_cuda(0) {
                    Ok(device) => return Ok((device, "cuda")),
                    Err(error) => {
                        eprintln!(
                            "warning: Candle supervised auto CUDA initialization failed; falling back to CPU: {error:#}"
                        );
                    }
                }
            }
            Ok((Device::Cpu, "cpu"))
        }
        "cpu" => Ok((Device::Cpu, "cpu")),
        "cuda" | "cuda:0" => {
            ensure_candle_cuda_policy()?;
            #[cfg(feature = "backend-candle-cuda")]
            {
                return Ok((
                    Device::new_cuda(0).context("initialize Candle CUDA device")?,
                    "cuda",
                ));
            }
            #[cfg(not(feature = "backend-candle-cuda"))]
            {
                bail!(
                    "Candle CUDA requires the `backend-candle-cuda` Cargo feature; rebuild with that feature"
                )
            }
        }
        "mps" | "metal" => bail!(
            "Candle Metal is not wired into supervised training; use Candle CPU/CUDA or Burn MLX"
        ),
        other => bail!("unsupported Candle device `{other}`"),
    }
}

#[cfg(feature = "backend-candle")]
fn ensure_candle_cuda_policy() -> Result<()> {
    if let Some(reason) = midas_env::ml::candle_cuda::explicit_block_reason() {
        bail!("{reason}");
    }
    Ok(())
}

#[cfg(feature = "backend-burn")]
// The supervised classifier is a tiny, dense CPU workload.  Burn's CubeCL
// CPU backend is optimized for larger fused kernels and can spend minutes
// compiling/dispatching a 1xN linear update.  NdArray is the same Burn
// autodiff API with a deterministic host implementation, and is also what
// the GA/RL CPU paths use for their small-matrix workloads.
type BurnCpuAutodiff = Autodiff<NdArray<f32, i32>>;

#[cfg(feature = "backend-burn-cuda")]
type BurnCudaAutodiff = Autodiff<Cuda<f32, i32>>;

#[cfg(feature = "backend-burn-mlx")]
type BurnMlxAutodiff = Autodiff<Mlx<f32>>;

#[cfg(feature = "backend-burn")]
fn train_burn_model(
    dataset: &TrainingDataset,
    train_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    initial_weights: Option<&[f64]>,
    initial_optimizer_state: Option<&OptimizerStateArtifact>,
    requested_device: &str,
    seed: u64,
    start_epoch: usize,
    final_epoch: usize,
    checkpoint_every: usize,
    checkpoint_callback: &mut CheckpointCallback<'_>,
) -> Result<(Vec<f64>, &'static str, OptimizerStateArtifact)> {
    let runtime = midas_env::ml::ComputeRuntime::parse(requested_device)?;
    match runtime {
        midas_env::ml::ComputeRuntime::Cpu => train_burn_model_on::<BurnCpuAutodiff>(
            dataset,
            train_indices,
            means,
            scales,
            epochs,
            learning_rate,
            l2,
            initial_weights,
            initial_optimizer_state,
            &NdArrayDevice::Cpu,
            "cpu",
            seed,
            start_epoch,
            final_epoch,
            checkpoint_every,
            checkpoint_callback,
        ),
        midas_env::ml::ComputeRuntime::Auto => {
            #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
            {
                return train_burn_model_on::<BurnMlxAutodiff>(
                    dataset,
                    train_indices,
                    means,
                    scales,
                    epochs,
                    learning_rate,
                    l2,
                    initial_weights,
                    initial_optimizer_state,
                    &MlxDevice::default(),
                    "mps",
                    seed,
                    start_epoch,
                    final_epoch,
                    checkpoint_every,
                    checkpoint_callback,
                );
            }
            #[cfg(all(
                not(all(target_os = "macos", feature = "backend-burn-mlx")),
                feature = "backend-burn-cuda"
            ))]
            {
                let device = CudaDevice::default();
                if burn_cuda_device_is_usable(&device) {
                    return train_burn_model_on::<BurnCudaAutodiff>(
                        dataset,
                        train_indices,
                        means,
                        scales,
                        epochs,
                        learning_rate,
                        l2,
                        initial_weights,
                        initial_optimizer_state,
                        &device,
                        "cuda",
                        seed,
                        start_epoch,
                        final_epoch,
                        checkpoint_every,
                        checkpoint_callback,
                    );
                }
                eprintln!(
                    "warning: Burn auto runtime could not verify a usable CUDA accelerator; falling back to CPU"
                );
            }
            {
                train_burn_model_on::<BurnCpuAutodiff>(
                    dataset,
                    train_indices,
                    means,
                    scales,
                    epochs,
                    learning_rate,
                    l2,
                    initial_weights,
                    initial_optimizer_state,
                    &NdArrayDevice::Cpu,
                    "cpu",
                    seed,
                    start_epoch,
                    final_epoch,
                    checkpoint_every,
                    checkpoint_callback,
                )
            }
        }
        midas_env::ml::ComputeRuntime::Cuda => {
            #[cfg(feature = "backend-burn-cuda")]
            {
                let device = CudaDevice::default();
                if !burn_cuda_device_is_usable(&device) {
                    bail!(
                        "Burn CUDA was explicitly requested, but no usable CUDA accelerator was detected on this host"
                    );
                }
                return train_burn_model_on::<BurnCudaAutodiff>(
                    dataset,
                    train_indices,
                    means,
                    scales,
                    epochs,
                    learning_rate,
                    l2,
                    initial_weights,
                    initial_optimizer_state,
                    &device,
                    "cuda",
                    seed,
                    start_epoch,
                    final_epoch,
                    checkpoint_every,
                    checkpoint_callback,
                );
            }
            #[cfg(not(feature = "backend-burn-cuda"))]
            {
                bail!(
                    "Burn CUDA requires the `backend-burn-cuda` Cargo feature; rebuild with that feature"
                )
            }
        }
        midas_env::ml::ComputeRuntime::Mps => {
            #[cfg(all(target_os = "macos", feature = "backend-burn-mlx"))]
            {
                return train_burn_model_on::<BurnMlxAutodiff>(
                    dataset,
                    train_indices,
                    means,
                    scales,
                    epochs,
                    learning_rate,
                    l2,
                    initial_weights,
                    initial_optimizer_state,
                    &MlxDevice::default(),
                    "mps",
                    seed,
                    start_epoch,
                    final_epoch,
                    checkpoint_every,
                    checkpoint_callback,
                );
            }
            #[cfg(not(all(target_os = "macos", feature = "backend-burn-mlx")))]
            {
                bail!("Burn Metal requires a macOS build with the `backend-burn-mlx` Cargo feature")
            }
        }
    }
}

#[cfg(feature = "backend-burn-cuda")]
fn burn_cuda_device_is_usable(device: &CudaDevice) -> bool {
    // `CudaDevice` is a lightweight descriptor in Burn/CubeCL; constructing it
    // does not prove that the driver and a physical device are usable. Force a
    // tiny allocation/readback and treat either a runtime error or panic as an
    // unavailable accelerator. The auto path can then safely choose NdArray.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        BurnTensor::<BurnCudaAutodiff, 1>::zeros([1], device)
            .to_data()
            .to_vec::<f32>()
            .is_ok()
    }))
    .unwrap_or(false)
}

#[cfg(feature = "backend-burn")]
fn train_burn_model_on<B: AutodiffBackend>(
    dataset: &TrainingDataset,
    train_indices: &[usize],
    means: &[f64],
    scales: &[f64],
    epochs: usize,
    learning_rate: f64,
    l2: f64,
    initial_weights: Option<&[f64]>,
    initial_optimizer_state: Option<&OptimizerStateArtifact>,
    device: &B::Device,
    device_label: &'static str,
    seed: u64,
    start_epoch: usize,
    final_epoch: usize,
    checkpoint_every: usize,
    checkpoint_callback: &mut CheckpointCallback<'_>,
) -> Result<(Vec<f64>, &'static str, OptimizerStateArtifact)> {
    let feature_count = dataset.feature_names.len();
    if train_indices.is_empty() {
        bail!("training split is empty");
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
        bail!("learning_rate must be positive and l2 must be non-negative");
    }

    let mut optimizer_state = initial_optimizer_state
        .cloned()
        .unwrap_or_else(|| OptimizerStateArtifact::empty(feature_count, learning_rate, l2));
    optimizer_state.validate(feature_count).or_else(|error| {
        if initial_optimizer_state.is_some() {
            Err(error)
        } else {
            Ok(())
        }
    })?;

    // Burn's Linear stores weights as [input, output], while the portable
    // supervised policy stores one [class, input, bias] row at a time. Keep
    // that conversion explicit so CPU, Candle, and Burn artifacts are
    // interchangeable and resume does not silently transpose a classifier.
    // Seed before construction, then use explicit host-generated weights so
    // Candle CPU (whose RNG cannot be seeded) follows the same initialization.
    B::seed(device, seed);
    let mut classifier: Linear<B> = LinearConfig::new(feature_count, 3).init(device);
    let policy_weights = initial_weights
        .map(|weights| weights.to_vec())
        .unwrap_or_else(|| deterministic_initial_weights(feature_count, seed));
    let stride = feature_count + 1;
    if policy_weights.len() != 3 * stride {
        bail!("resume policy weight shape does not match feature count");
    }
    let mut burn_weights = vec![0.0_f32; feature_count * 3];
    let mut biases = vec![0.0_f32; 3];
    for class in 0..3 {
        let offset = class * stride;
        biases[class] = policy_weights[offset + feature_count] as f32;
        for feature in 0..feature_count {
            burn_weights[feature * 3 + class] = policy_weights[offset + feature] as f32;
        }
    }
    classifier.weight = Param::from_data(TensorData::new(burn_weights, [feature_count, 3]), device);
    classifier.bias = Some(Param::from_data(TensorData::new(biases, [3]), device));

    let canonical_to_burn = |values: &[f64]| {
        let mut native = vec![0.0_f32; feature_count * 3];
        for class in 0..3 {
            for feature in 0..feature_count {
                native[feature * 3 + class] = values[class * feature_count + feature] as f32;
            }
        }
        native
    };
    let burn_to_canonical = |values: Vec<f32>| {
        let mut canonical = vec![0.0_f64; feature_count * 3];
        for class in 0..3 {
            for feature in 0..feature_count {
                canonical[class * feature_count + feature] = values[feature * 3 + class] as f64;
            }
        }
        canonical
    };
    let mut weight_first_moment = BurnTensor::<B, 2>::from_data(
        TensorData::new(
            canonical_to_burn(&optimizer_state.weight_first_moment),
            [feature_count, 3],
        ),
        device,
    );
    let mut weight_second_moment = BurnTensor::<B, 2>::from_data(
        TensorData::new(
            canonical_to_burn(&optimizer_state.weight_second_moment),
            [feature_count, 3],
        ),
        device,
    );
    let mut bias_first_moment = BurnTensor::<B, 1>::from_data(
        TensorData::new(
            optimizer_state
                .bias_first_moment
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            [3],
        ),
        device,
    );
    let mut bias_second_moment = BurnTensor::<B, 1>::from_data(
        TensorData::new(
            optimizer_state
                .bias_second_moment
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            [3],
        ),
        device,
    );

    let mut features = Vec::with_capacity(train_indices.len() * feature_count);
    let mut targets = Vec::with_capacity(train_indices.len());
    for index in train_indices {
        let row = &dataset.rows[*index];
        features.extend(
            scaled_features(row, means, scales)
                .into_iter()
                .map(|value| value as f32),
        );
        targets.push(class_index(row.label_action) as i32);
    }
    let inputs = BurnTensor::<B, 2>::from_data(
        TensorData::new(features, [train_indices.len(), feature_count]),
        device,
    );
    let targets =
        BurnTensor::<B, 1, Int>::from_data(TensorData::new(targets, [train_indices.len()]), device);
    let criterion = CrossEntropyLossConfig::new().init(device);
    for local_epoch in 0..epochs.max(1) {
        let logits = classifier.forward(inputs.clone());
        let objective = criterion.forward(logits, targets.clone());
        let mut gradients = GradientsParams::from_grads(objective.backward(), &classifier);
        let weight_gradient = gradients
            .remove::<B::InnerBackend, 2>(classifier.weight.id)
            .map(BurnTensor::<B, 2>::from_inner)
            .context("Burn classifier weight gradient is missing")?;
        let bias_id = classifier
            .bias
            .as_ref()
            .context("Burn classifier is missing a bias")?
            .id;
        let bias_gradient = gradients
            .remove::<B::InnerBackend, 1>(bias_id)
            .map(BurnTensor::<B, 1>::from_inner)
            .context("Burn classifier bias gradient is missing")?;

        optimizer_state.step += 1;
        let step = optimizer_state.step as i32;
        let beta1 = optimizer_state.beta1 as f32;
        let beta2 = optimizer_state.beta2 as f32;
        let factor_1 = 1.0 - beta1;
        let factor_2 = 1.0 - beta2;
        weight_first_moment = weight_first_moment
            .mul_scalar(beta1)
            .add(weight_gradient.clone().mul_scalar(factor_1));
        weight_second_moment = weight_second_moment
            .mul_scalar(beta2)
            .add(weight_gradient.square().mul_scalar(factor_2));
        bias_first_moment = bias_first_moment
            .mul_scalar(beta1)
            .add(bias_gradient.clone().mul_scalar(factor_1));
        bias_second_moment = bias_second_moment
            .mul_scalar(beta2)
            .add(bias_gradient.square().mul_scalar(factor_2));

        let scale_1 = 1.0 / (1.0 - beta1.powi(step));
        let scale_2 = 1.0 / (1.0 - beta2.powi(step));
        let epsilon = optimizer_state.epsilon as f32;
        let weight_update = weight_first_moment.clone().mul_scalar(scale_1).div(
            weight_second_moment
                .clone()
                .mul_scalar(scale_2)
                .sqrt()
                .add_scalar(epsilon),
        );
        let bias_update = bias_first_moment.clone().mul_scalar(scale_1).div(
            bias_second_moment
                .clone()
                .mul_scalar(scale_2)
                .sqrt()
                .add_scalar(epsilon),
        );
        let decay = 1.0 - (optimizer_state.learning_rate * optimizer_state.weight_decay) as f32;
        let next_weight = classifier
            .weight
            .val()
            .mul_scalar(decay)
            .sub(weight_update.mul_scalar(optimizer_state.learning_rate as f32));
        let next_bias = classifier
            .bias
            .as_ref()
            .context("Burn classifier is missing a bias")?
            .val()
            .mul_scalar(decay)
            .sub(bias_update.mul_scalar(optimizer_state.learning_rate as f32));
        classifier.weight = classifier
            .weight
            .clone()
            .map(|_| next_weight.detach().require_grad());
        let bias = classifier
            .bias
            .take()
            .context("Burn classifier is missing a bias")?;
        classifier.bias = Some(bias.map(|_| next_bias.detach().require_grad()));

        let absolute_epoch = start_epoch + local_epoch + 1;
        if checkpoint_due(checkpoint_every, absolute_epoch, final_epoch) {
            let packed = burn_policy_weights(&classifier, feature_count)?;
            let mut checkpoint_optimizer_state = optimizer_state.clone();
            checkpoint_optimizer_state.weight_first_moment = burn_to_canonical(
                weight_first_moment
                    .to_data()
                    .to_vec::<f32>()
                    .context("read Burn checkpoint weight first moment")?,
            );
            checkpoint_optimizer_state.weight_second_moment = burn_to_canonical(
                weight_second_moment
                    .to_data()
                    .to_vec::<f32>()
                    .context("read Burn checkpoint weight second moment")?,
            );
            checkpoint_optimizer_state.bias_first_moment = bias_first_moment
                .to_data()
                .to_vec::<f32>()
                .context("read Burn checkpoint bias first moment")?
                .into_iter()
                .map(|value| value as f64)
                .collect();
            checkpoint_optimizer_state.bias_second_moment = bias_second_moment
                .to_data()
                .to_vec::<f32>()
                .context("read Burn checkpoint bias second moment")?
                .into_iter()
                .map(|value| value as f64)
                .collect();
            checkpoint_callback(
                absolute_epoch,
                &packed,
                Some(&checkpoint_optimizer_state),
                device_label,
            )?;
        }
    }

    let packed = burn_policy_weights(&classifier, feature_count)?;
    optimizer_state.weight_first_moment = burn_to_canonical(
        weight_first_moment
            .to_data()
            .to_vec::<f32>()
            .context("read Burn weight first moment")?,
    );
    optimizer_state.weight_second_moment = burn_to_canonical(
        weight_second_moment
            .to_data()
            .to_vec::<f32>()
            .context("read Burn weight second moment")?,
    );
    optimizer_state.bias_first_moment = bias_first_moment
        .to_data()
        .to_vec::<f32>()
        .context("read Burn bias first moment")?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    optimizer_state.bias_second_moment = bias_second_moment
        .to_data()
        .to_vec::<f32>()
        .context("read Burn bias second moment")?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    Ok((packed, device_label, optimizer_state))
}

#[cfg(feature = "backend-burn")]
fn burn_policy_weights<B: AutodiffBackend>(
    classifier: &Linear<B>,
    feature_count: usize,
) -> Result<Vec<f64>> {
    let weight = classifier
        .weight
        .val()
        .to_data()
        .to_vec::<f32>()
        .context("read Burn classifier weights")?;
    let bias = classifier
        .bias
        .as_ref()
        .context("Burn classifier is missing a bias")?
        .val()
        .to_data()
        .to_vec::<f32>()
        .context("read Burn classifier bias")?;
    if weight.len() != feature_count * 3 || bias.len() != 3 {
        bail!("Burn classifier weight shape does not match feature count");
    }
    let mut packed = Vec::with_capacity(3 * (feature_count + 1));
    for class in 0..3 {
        for feature in 0..feature_count {
            packed.push(weight[feature * 3 + class] as f64);
        }
        packed.push(bias[class] as f64);
    }
    Ok(packed)
}

fn predicted_class(policy: &PolicyArtifact, row: &TrainingRow) -> usize {
    let input = scaled_features(row, &policy.means, &policy.scales);
    let values = logits(&policy.weights, &input, policy.classes);
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(1)
}

fn action_for_class(class: usize, raw_direction: i8, current: i8) -> i8 {
    match class {
        0 => raw_direction,
        1 => current,
        2 => -raw_direction,
        _ => current,
    }
}

fn transition_cost_for(from: i8, to: i8, config: &SupervisedConfig) -> f64 {
    -((from - to).unsigned_abs() as f64 * config.round_trip_cost / 2.0)
}

#[derive(Debug, Clone, Copy)]
struct SimulationResult {
    pnl: f64,
}

fn simulate(
    rows: &[TrainingRow],
    indices: &[usize],
    config: &SupervisedConfig,
    policy: Option<&PolicyArtifact>,
    fixed_class: Option<usize>,
) -> SimulationResult {
    let mut pnl = 0.0;
    let mut current_session = None::<&str>;
    let mut position = 0_i8;
    for index in indices {
        let row = &rows[*index];
        if current_session != Some(row.session_id.as_str()) {
            current_session = Some(row.session_id.as_str());
            position = 0;
        }
        let class =
            fixed_class.unwrap_or_else(|| predicted_class(policy.expect("policy required"), row));
        let next = action_for_class(class, row.raw_direction, position);
        pnl += next as f64
            * (row.interval_end_price - row.decision_price)
            * config.contract_multiplier;
        pnl += transition_cost_for(position, next, config);
        position = next;
        if row.terminal_event {
            pnl += transition_cost_for(position, 0, config);
            position = 0;
        }
    }
    SimulationResult { pnl }
}

fn metrics_for(
    name: &str,
    dataset: &TrainingDataset,
    indices: &[usize],
    policy: &PolicyArtifact,
) -> SplitMetrics {
    let mut confusion = [[0usize; 3]; 3];
    let mut cross_entropy = 0.0;
    for index in indices {
        let row = &dataset.rows[*index];
        let input = scaled_features(row, &policy.means, &policy.scales);
        let probabilities = softmax(&logits(&policy.weights, &input, 3));
        let target = class_index(row.label_action);
        let predicted = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(class, _)| class)
            .unwrap_or(1);
        confusion[target][predicted] += 1;
        cross_entropy -= probabilities[target].max(1e-12).ln();
    }
    let rows = indices.len();
    let accuracy = if rows == 0 {
        0.0
    } else {
        (0..3).map(|class| confusion[class][class]).sum::<usize>() as f64 / rows as f64
    };
    let macro_f1 = (0..3)
        .map(|class| {
            let tp = confusion[class][class] as f64;
            let fp = (0..3).map(|other| confusion[other][class]).sum::<usize>() as f64 - tp;
            let fn_ = (0..3).map(|other| confusion[class][other]).sum::<usize>() as f64 - tp;
            if tp == 0.0 || 2.0 * tp + fp + fn_ == 0.0 {
                0.0
            } else {
                2.0 * tp / (2.0 * tp + fp + fn_)
            }
        })
        .sum::<f64>()
        / 3.0;
    let predicted = simulate(&dataset.rows, indices, &dataset.config, Some(policy), None);
    let oracle = simulate_labels(&dataset.rows, indices, &dataset.config);
    let normal = simulate(&dataset.rows, indices, &dataset.config, None, Some(0));
    let skip = simulate(&dataset.rows, indices, &dataset.config, None, Some(1));
    let invert = simulate(&dataset.rows, indices, &dataset.config, None, Some(2));
    SplitMetrics {
        split: name.to_string(),
        rows,
        sessions: indices
            .iter()
            .map(|index| dataset.rows[*index].session_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        accuracy,
        cross_entropy: if rows == 0 {
            0.0
        } else {
            cross_entropy / rows as f64
        },
        macro_f1,
        predicted_pnl: predicted.pnl,
        oracle_pnl: oracle.pnl,
        always_normal_pnl: normal.pnl,
        always_skip_pnl: skip.pnl,
        always_invert_pnl: invert.pnl,
        oracle_regret: oracle.pnl - predicted.pnl,
    }
}

fn simulate_labels(
    rows: &[TrainingRow],
    indices: &[usize],
    config: &SupervisedConfig,
) -> SimulationResult {
    let mut pnl = 0.0;
    let mut current_session = None::<&str>;
    let mut position = 0_i8;
    for index in indices {
        let row = &rows[*index];
        if current_session != Some(row.session_id.as_str()) {
            current_session = Some(row.session_id.as_str());
            position = 0;
        }
        let next = action_for_class(class_index(row.label_action), row.raw_direction, position);
        pnl += next as f64
            * (row.interval_end_price - row.decision_price)
            * config.contract_multiplier;
        pnl += transition_cost_for(position, next, config);
        position = next;
        if row.terminal_event {
            pnl += transition_cost_for(position, 0, config);
            position = 0;
        }
    }
    SimulationResult { pnl }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resume_fixture(backend: &str) -> PolicyArtifact {
        let optimizer_state = if backend == "cpu-linear" {
            None
        } else {
            let mut state = OptimizerStateArtifact::empty(1, 0.001, 0.0001);
            state.step = 3;
            Some(state)
        };
        PolicyArtifact {
            schema_version: POLICY_SCHEMA.to_string(),
            dataset_schema: SUPERVISED_DATASET_SCHEMA.to_string(),
            label_schema: SUPERVISED_LABEL_SCHEMA.to_string(),
            trainer: "multiclass-logistic-regression".to_string(),
            backend: backend.to_string(),
            device: "cpu".to_string(),
            feature_names: vec!["feature".to_string()],
            means: vec![0.0],
            scales: vec![1.0],
            weights: vec![0.0; 6],
            class_names: vec![
                "normal".to_string(),
                "skip".to_string(),
                "invert".to_string(),
            ],
            classes: 3,
            feature_count: 1,
            source_hash_sha256: String::new(),
            dataset_fingerprint_sha256: String::new(),
            config_json: String::new(),
            seed: 42,
            training_epochs: if backend == "cpu-linear" { 3 } else { 3 },
            optimizer_state,
            split_provenance: Some(fixture_split()),
        }
    }

    fn row(session_id: &str) -> TrainingRow {
        TrainingRow {
            session_id: session_id.to_string(),
            timestamp_ns: match session_id {
                "one" => 1,
                "two" => 2,
                "three" => 3,
                _ => 1,
            },
            raw_direction: 1,
            label_action: 1,
            decision_price: 1.0,
            interval_end_price: 1.0,
            terminal_event: true,
            features: vec![0.0],
        }
    }

    fn fixture_split() -> SplitProvenance {
        session_splits(&[row("one"), row("two"), row("three")], 0.6, 0.2)
            .unwrap()
            .4
    }

    #[test]
    fn split_rejects_tiny_or_overlapping_fixtures() {
        assert!(session_splits(&[], 0.6, 0.2).is_err());
        assert!(session_splits(&[row("one")], 0.6, 0.2).is_err());
        assert!(session_splits(&[row("one"), row("one")], 0.6, 0.2).is_err());
        assert!(session_splits(&[row("one"), row("two"), row("three")], 0.6, 0.2).is_ok());
    }

    #[test]
    fn resume_rejects_every_backend_change_before_training() {
        for (source, requested) in [
            ("cpu-linear", "burn"),
            ("cpu-linear", "candle"),
            ("burn", "candle"),
            ("candle", "burn"),
        ] {
            let error =
                validate_resume_compatibility(&resume_fixture(source), requested, &fixture_split())
                    .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("does not match requested backend")
            );
        }
    }

    #[test]
    fn same_backend_resume_requires_adamw_epoch_provenance() {
        let mut policy = resume_fixture("burn");
        assert!(validate_resume_compatibility(&policy, "burn", &fixture_split()).is_ok());

        policy.optimizer_state = None;
        let error = validate_resume_compatibility(&policy, "burn", &fixture_split()).unwrap_err();
        assert!(error.to_string().contains("missing AdamW optimizer state"));

        let mut policy = resume_fixture("candle");
        policy.training_epochs = 4;
        let error = validate_resume_compatibility(&policy, "candle", &fixture_split()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not match AdamW optimizer step")
        );
    }

    #[test]
    fn resume_rejects_missing_or_changed_split_provenance() {
        let expected = fixture_split();
        let mut policy = resume_fixture("cpu-linear");
        policy.split_provenance = None;
        let error = validate_resume_compatibility(&policy, "cpu-linear", &expected).unwrap_err();
        assert!(error.to_string().contains("no split provenance"));

        let mut changed_fraction = expected.clone();
        changed_fraction.train_fraction = 0.5;
        let error = validate_resume_compatibility(
            &resume_fixture("cpu-linear"),
            "cpu-linear",
            &changed_fraction,
        )
        .unwrap_err();
        assert!(error.to_string().contains("split provenance"));

        let mut changed_boundary = expected.clone();
        changed_boundary.holdout_session_ids = vec!["different".to_string()];
        let error = validate_resume_compatibility(
            &resume_fixture("cpu-linear"),
            "cpu-linear",
            &changed_boundary,
        )
        .unwrap_err();
        assert!(error.to_string().contains("split provenance"));
    }

    #[test]
    fn legacy_policy_schema_is_evaluation_only() {
        let mut policy = resume_fixture("cpu-linear");
        policy.schema_version = LEGACY_POLICY_SCHEMA.to_string();
        let error =
            validate_resume_compatibility(&policy, "cpu-linear", &fixture_split()).unwrap_err();
        assert!(error.to_string().contains("evaluation-only"));
    }

    #[test]
    fn legacy_policy_json_still_deserializes_without_split_provenance() {
        let mut value = serde_json::to_value(resume_fixture("cpu-linear")).unwrap();
        value.as_object_mut().unwrap().remove("split_provenance");
        let decoded: PolicyArtifact = serde_json::from_value(value).unwrap();
        assert!(decoded.split_provenance.is_none());
    }

    #[test]
    fn event_timestamps_must_be_integer_and_strictly_chronological() {
        let unsorted = DataFrame::new(vec![
            Series::new("timestamp_ns".into(), &[2_i64, 1_i64]).into(),
        ])
        .unwrap();
        let timestamps = timestamp_ns_column(&unsorted, "timestamp_ns").unwrap();
        let error = validate_event_timestamps(&timestamps).unwrap_err();
        assert!(error.to_string().contains("strictly increasing"));

        let floating = DataFrame::new(vec![
            Series::new("timestamp_ns".into(), &[1.0_f64, 2.0_f64]).into(),
        ])
        .unwrap();
        let error = timestamp_ns_column(&floating, "timestamp_ns").unwrap_err();
        assert!(error.to_string().contains("integer parquet type"));
    }

    #[test]
    fn provenance_requires_the_first_party_schema_and_consistent_metadata() {
        let config = SupervisedConfig::default();
        let provenance = EventDatasetProvenance {
            source_path: None,
            timestamp_source: "timestamp_ns".to_string(),
            timestamp_unit: "ns".to_string(),
            timestamp_timezone: config.session_timezone.clone(),
            index_timestamp_fallback: false,
            raw_price_scale: None,
            bar_kind: config.bar_kind.clone(),
            bar_value: config.bar_value,
            source_row_count: 10,
            source_hash_sha256: "source".to_string(),
            dataset_fingerprint_sha256: "fingerprint".to_string(),
        };
        validate_event_dataset_provenance(
            &provenance,
            &config,
            "source",
            "fingerprint",
            "timestamp_ns",
            "ns",
            &config.session_timezone,
            false,
            1.0,
            &config.bar_kind,
            config.bar_value,
            10,
        )
        .unwrap();

        let mut changed = provenance.clone();
        changed.bar_kind = "range".to_string();
        let error = validate_event_dataset_provenance(
            &changed,
            &config,
            "source",
            "fingerprint",
            "timestamp_ns",
            "ns",
            &config.session_timezone,
            false,
            1.0,
            &config.bar_kind,
            config.bar_value,
            10,
        )
        .unwrap_err();
        assert!(error.to_string().contains("bar metadata"));

        let mut unknown = serde_json::to_value(provenance).unwrap();
        unknown
            .as_object_mut()
            .unwrap()
            .insert("untrusted_feature_value".to_string(), serde_json::json!(1));
        assert!(serde_json::from_value::<EventDatasetProvenance>(unknown).is_err());
    }

    fn integrity_fixture_payload() -> StoredEventPayload {
        let mut features = BTreeMap::new();
        features.insert("feature".to_string(), 1.25);
        let event = SupervisedEvent {
            event_id: 7,
            session_id: "session".to_string(),
            session_event_index: 2,
            row_idx: 3,
            timestamp_ns: 4,
            raw_direction: 1,
            decision_price: 101.0,
            interval_end_price: 102.0,
            interval_end_row_idx: 5,
            terminal_event: false,
            features,
            action_value_normal: 2.0,
            action_value_skip: 0.0,
            action_value_invert: -2.0,
            label_action: 1,
            label_name: "normal".to_string(),
            oracle_position_before: 0,
            oracle_position_after: 1,
            oracle_value: 2.0,
        };
        let source = LoadedSource {
            bars: SupervisedBars {
                open: vec![100.0, 100.0, 100.0, 101.0, 102.0, 103.0],
                high: vec![100.0, 100.0, 100.0, 101.5, 102.5, 103.5],
                low: vec![100.0, 100.0, 100.0, 100.5, 101.5, 102.5],
                close: vec![100.0, 100.0, 100.0, 101.0, 102.0, 103.0],
                volume: vec![1.0; 6],
                timestamp_ns: vec![1, 2, 3, 4, 5, 6],
            },
            timestamp_source: "timestamp_ns".to_string(),
            timestamp_unit: "ns".to_string(),
            timestamp_timezone: "UTC".to_string(),
            index_timestamp_fallback: false,
            raw_price_scale: None,
            volume_present: true,
            source_row_count: 6,
        };
        generated_event_payloads(&[event], &source)
            .unwrap()
            .pop()
            .unwrap()
    }

    #[test]
    fn event_integrity_rejects_tampered_feature_payload() {
        let expected = integrity_fixture_payload();
        let mut tampered = expected.clone();
        tampered.event.features.insert("feature".to_string(), 999.0);

        let error = compare_event_payloads(&[tampered], &[expected]).unwrap_err();
        assert!(error.to_string().contains("event payload mismatch"));
        assert!(error.to_string().contains("feature"));
    }

    #[test]
    fn source_integrity_rejects_modified_original_source() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-source-integrity-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("bars.csv");
        std::fs::write(&source, b"timestamp,open,high,low,close\n1,1,1,1,1\n").unwrap();
        let expected = super::sha256_file(&source).unwrap();
        std::fs::write(&source, b"timestamp,open,high,low,close\n1,2,2,2,2\n").unwrap();

        let error = verify_source_hash(&source, &expected).unwrap_err();
        assert!(error.to_string().contains("source hash mismatch"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn source_snapshot_survives_replacement_after_root_validation() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-source-snapshot-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("bars.csv");
        let original = b"timestamp,open,high,low,close\n1,1,1,1,1\n2,1.5,1.5,1.5,1.5\n";
        let replacement = b"timestamp,open,high,low,close\n1,999,999,999,999\n2,998,998,998,998\n";
        std::fs::write(&source, original).unwrap();

        let resolved = resolve_source_path(
            &directory.join("events.parquet"),
            "bars.csv",
            Some(&directory),
            true,
        )
        .unwrap();
        let source_hash = super::super::sha256_file(&source).unwrap();

        // Exercise the actual validation-to-open race boundary: a regular
        // replacement must fail closed because its inode differs from the
        // metadata captured by resolve_source_path.
        std::fs::remove_file(&source).unwrap();
        std::fs::write(&source, replacement).unwrap();
        let rejected_snapshot = PrivateResumeSnapshot::create(&std::env::temp_dir()).unwrap();
        let error = rejected_snapshot
            .snapshot_source(
                &resolved.path,
                &resolved.metadata,
                "supervised dataset source",
                &source_snapshot_slot(&resolved.path),
            )
            .unwrap_err();
        assert!(error.to_string().contains("replaced"));
        drop(rejected_snapshot);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let replacement_target = directory.join("replacement.csv");
            std::fs::write(&replacement_target, replacement).unwrap();
            std::fs::remove_file(&source).unwrap();
            symlink(&replacement_target, &source).unwrap();
            let symlink_snapshot = PrivateResumeSnapshot::create(&std::env::temp_dir()).unwrap();
            let error = symlink_snapshot
                .snapshot_source(
                    &resolved.path,
                    &resolved.metadata,
                    "supervised dataset source",
                    &source_snapshot_slot(&resolved.path),
                )
                .unwrap_err();
            assert!(error.to_string().contains("symlink"));
            drop(symlink_snapshot);
            std::fs::remove_file(&source).unwrap();
            std::fs::remove_file(replacement_target).unwrap();
        }

        if std::fs::symlink_metadata(&source).is_ok() {
            std::fs::remove_file(&source).unwrap();
        }
        std::fs::write(&source, original).unwrap();
        let resolved = resolve_source_path(
            &directory.join("events.parquet"),
            "bars.csv",
            Some(&directory),
            true,
        )
        .unwrap();
        let snapshot = PrivateResumeSnapshot::create(&std::env::temp_dir()).unwrap();
        let snapshot_path = snapshot
            .snapshot_source(
                &resolved.path,
                &resolved.metadata,
                "supervised dataset source",
                &source_snapshot_slot(&resolved.path),
            )
            .unwrap();

        // A replacement after root/symlink validation must not change the
        // bytes later hashed or parsed by the integrity validator.
        std::fs::remove_file(&source).unwrap();
        std::fs::write(&source, replacement).unwrap();
        verify_source_hash(&snapshot_path, &source_hash).unwrap();
        let config = SupervisedConfig::default();
        let loaded = super::super::load_source(
            &snapshot_path,
            Some("s"),
            false,
            &config.session_timezone,
            &config.bar_kind,
            config.bar_value,
            None,
        )
        .unwrap();
        assert_eq!(loaded.bars.close, vec![1.0, 1.5]);

        drop(snapshot);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn source_resolution_is_root_contained_and_rejects_traversal() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-source-root-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let trusted = directory.join("trusted");
        let external = directory.join("external");
        std::fs::create_dir_all(&trusted).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        let source = trusted.join("bars.csv");
        let outside = external.join("outside.csv");
        std::fs::write(&source, b"timestamp,open,high,low,close\n1,1,1,1,1\n").unwrap();
        std::fs::write(&outside, b"timestamp,open,high,low,close\n1,2,2,2,2\n").unwrap();
        let dataset = trusted.join("events.parquet");

        assert_eq!(
            resolve_source_path(&dataset, "bars.csv", Some(&trusted), true)
                .unwrap()
                .path,
            std::fs::canonicalize(&source).unwrap()
        );
        let traversal =
            resolve_source_path(&dataset, "../external/outside.csv", Some(&trusted), true)
                .unwrap_err();
        assert!(traversal.to_string().contains("contains `..`"));

        let outside_without_opt_in = resolve_source_path(
            &dataset,
            &outside.display().to_string(),
            Some(&trusted),
            true,
        )
        .unwrap_err();
        assert!(
            outside_without_opt_in
                .to_string()
                .contains("outside trusted source root")
        );

        let external_without_opt_in = resolve_source_path(
            &external.join("outside.csv"),
            "outside.csv",
            Some(&external),
            false,
        )
        .unwrap_err();
        assert!(
            external_without_opt_in
                .to_string()
                .contains("--allow-external-source")
        );
        assert_eq!(
            resolve_source_path(
                &external.join("events.parquet"),
                "outside.csv",
                Some(&external),
                true,
            )
            .unwrap()
            .path,
            std::fs::canonicalize(&outside).unwrap()
        );

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn source_resolution_rejects_symlinked_sources() {
        use std::os::unix::fs::symlink;

        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-source-symlink-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("bars.csv");
        let link = directory.join("bars-link.csv");
        std::fs::write(&source, b"timestamp,open,high,low,close\n1,1,1,1,1\n").unwrap();
        symlink(&source, &link).unwrap();

        let error = resolve_source_path(
            &directory.join("events.parquet"),
            "bars-link.csv",
            Some(&directory),
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("symlink"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resume_requires_a_complete_hash_matching_manifest() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-resume-manifest-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let policy_path = directory.join("policy.json");
        let metrics_path = directory.join("metrics.json");
        let policy = serde_json::json!({
            "schema_version": POLICY_SCHEMA,
            "dataset_fingerprint_sha256": "dataset-hash"
        });
        let metrics = serde_json::json!({
            "policy_schema": POLICY_SCHEMA,
            "dataset_fingerprint_sha256": "dataset-hash"
        });
        std::fs::write(&policy_path, serde_json::to_vec(&policy).unwrap()).unwrap();
        std::fs::write(&metrics_path, serde_json::to_vec(&metrics).unwrap()).unwrap();
        let manifest = RunCompletionManifest {
            schema_version: RUN_COMPLETION_SCHEMA.to_string(),
            status: "complete".to_string(),
            policy_file: "policy.json".to_string(),
            metrics_file: "metrics.json".to_string(),
            policy_sha256: super::super::sha256_file(&policy_path).unwrap(),
            metrics_sha256: super::super::sha256_file(&metrics_path).unwrap(),
            dataset_fingerprint_sha256: "dataset-hash".to_string(),
            total_epochs: 10,
        };
        std::fs::write(
            directory.join(RUN_COMPLETION_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        assert!(validate_resume_completion_manifest(&policy_path).is_ok());
        std::fs::write(&metrics_path, b"tampered").unwrap();
        let error = validate_resume_completion_manifest(&policy_path).unwrap_err();
        assert!(error.to_string().contains("metrics hash"));
        std::fs::write(&metrics_path, serde_json::to_vec(&metrics).unwrap()).unwrap();
        std::fs::write(&policy_path, b"tampered").unwrap();
        let error = validate_resume_completion_manifest(&policy_path).unwrap_err();
        assert!(error.to_string().contains("policy hash"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn resume_snapshot_survives_replacement_after_validation() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-resume-snapshot-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let policy_path = directory.join("policy.json");
        let metrics_path = directory.join("metrics.json");
        let policy = serde_json::json!({
            "schema_version": POLICY_SCHEMA,
            "dataset_fingerprint_sha256": "dataset-hash"
        });
        let metrics = serde_json::json!({
            "policy_schema": POLICY_SCHEMA,
            "dataset_fingerprint_sha256": "dataset-hash"
        });
        let policy_bytes = serde_json::to_vec(&policy).unwrap();
        let metrics_bytes = serde_json::to_vec(&metrics).unwrap();
        std::fs::write(&policy_path, &policy_bytes).unwrap();
        std::fs::write(&metrics_path, &metrics_bytes).unwrap();
        let manifest = RunCompletionManifest {
            schema_version: RUN_COMPLETION_SCHEMA.to_string(),
            status: "complete".to_string(),
            policy_file: "policy.json".to_string(),
            metrics_file: "metrics.json".to_string(),
            policy_sha256: sha256_bytes(&policy_bytes),
            metrics_sha256: sha256_bytes(&metrics_bytes),
            dataset_fingerprint_sha256: "dataset-hash".to_string(),
            total_epochs: 10,
        };
        std::fs::write(
            directory.join(RUN_COMPLETION_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let snapshot = validate_resume_completion_manifest(&policy_path).unwrap();

        // Replacing the caller paths after validation must not change what
        // run_train will deserialize from the returned snapshot.
        std::fs::remove_file(&policy_path).unwrap();
        std::fs::write(&policy_path, b"replacement").unwrap();
        std::fs::remove_file(&metrics_path).unwrap();
        std::fs::write(&metrics_path, b"replacement").unwrap();
        assert_eq!(snapshot.policy_bytes, policy_bytes);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn private_resume_snapshot_rejects_symlink_sources() {
        use std::os::unix::fs::symlink;

        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-resume-symlink-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("source.json");
        let link = directory.join("link.json");
        std::fs::write(&source, b"source").unwrap();
        symlink(&source, &link).unwrap();

        let snapshot = PrivateResumeSnapshot::create(&directory).unwrap();
        let error = snapshot
            .read_artifact(&link, "resume policy", "policy.json")
            .unwrap_err();
        assert!(error.to_string().contains("symlink"));
        drop(snapshot);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "backend-candle")]
    #[test]
    fn supervised_candle_device_selection_uses_shared_pascal_policy() {
        match midas_env::ml::candle_cuda::host_policy() {
            midas_env::ml::candle_cuda::HostPolicy::PascalBlocked
            | midas_env::ml::candle_cuda::HostPolicy::Unknown => {
                let (device, label) = candle_training_device("auto").unwrap();
                assert_eq!(label, "cpu");
                assert!(matches!(device, Device::Cpu));
                let error = candle_training_device("cuda").unwrap_err();
                let error = error.to_string();
                assert!(error.contains("Candle CUDA") || error.contains("backend-candle-cuda"));
            }
            midas_env::ml::candle_cuda::HostPolicy::Supported => {
                // On a supported host this test still exercises the shared
                // policy before the feature/device initialization branch.
                let _ = candle_training_device("auto");
            }
        }
    }

    #[test]
    fn completion_manifest_is_explicitly_complete_and_hash_addressed() {
        let manifest = RunCompletionManifest {
            schema_version: RUN_COMPLETION_SCHEMA.to_string(),
            status: "complete".to_string(),
            policy_file: "policy.json".to_string(),
            metrics_file: "metrics.json".to_string(),
            policy_sha256: "policy-hash".to_string(),
            metrics_sha256: "metrics-hash".to_string(),
            dataset_fingerprint_sha256: "dataset-hash".to_string(),
            total_epochs: 50,
        };
        let decoded: RunCompletionManifest =
            serde_json::from_value(serde_json::to_value(&manifest).unwrap()).unwrap();
        assert_eq!(decoded, manifest);
        assert_eq!(decoded.status, "complete");
        assert_eq!(decoded.policy_file, "policy.json");
        assert_eq!(decoded.metrics_file, "metrics.json");
    }

    #[test]
    fn json_publication_is_atomic_and_never_replaces_existing_artifacts() {
        let directory = std::env::temp_dir().join(format!(
            "midas-supervised-publish-test-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let destination = directory.join("policy.json");

        publish_json_no_replace(&destination, &serde_json::json!({"value": 1})).unwrap();
        assert_eq!(
            std::fs::read_to_string(&destination).unwrap(),
            "{\n  \"value\": 1\n}\n"
        );

        let error =
            publish_json_no_replace(&destination, &serde_json::json!({"value": 2})).unwrap_err();
        assert!(error.to_string().contains("without replacing"));
        assert_eq!(
            std::fs::read_to_string(&destination).unwrap(),
            "{\n  \"value\": 1\n}\n"
        );
        let temporary_files = std::fs::read_dir(&directory)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .contains(".policy.json.tmp-")
            })
            .count();
        assert_eq!(temporary_files, 0);

        std::fs::remove_dir_all(directory).unwrap();
    }
}

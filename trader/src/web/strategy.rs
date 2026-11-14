//! Local strategy API backed by the trader's existing replay CLI.
//!
//! The web UI remains the reference implementation for result, heatmap and
//! candidate artifact shaping. This module implements native spec discovery,
//! validation and replay job lifecycle without accepting caller supplied paths
//! or CLI arguments.
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    path::{Component, Path as FsPath, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs,
    io::AsyncReadExt,
    process::Command,
    sync::{Mutex, Notify},
};

const MAX_BODY: usize = 32 * 1024;
const MAX_SPEC: u64 = 2_000_000;
const MAX_OUTPUT: usize = 96 * 1024;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static JOB_GATE: AtomicBool = AtomicBool::new(false);
pub(super) struct ActiveGate;
impl Drop for ActiveGate {
    fn drop(&mut self) {
        JOB_GATE.store(false, Ordering::SeqCst);
    }
}
pub(super) fn reserve_job_gate() -> Option<ActiveGate> {
    JOB_GATE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .ok()
        .map(|_| ActiveGate)
}

#[derive(Clone)]
pub(super) struct ApiState {
    pub(super) trader_dir: PathBuf,
    pub(super) specs_root: PathBuf,
    pub(super) jobs_root: PathBuf,
    pub(super) jobs: Arc<Mutex<HashMap<String, Arc<Mutex<Job>>>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Job {
    pub(super) id: String,
    pub(super) operation: String,
    #[serde(rename = "sweepId")]
    pub(super) sweep_id: String,
    pub(super) status: String,
    #[serde(rename = "createdAt")]
    pub(super) created_at: String,
    #[serde(rename = "updatedAt")]
    pub(super) updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) progress: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) metric: Option<String>,
    #[serde(skip)]
    pub(super) spec_path: PathBuf,
    #[serde(skip)]
    pub(super) output_dir: PathBuf,
    #[serde(skip)]
    pub(super) cancelled: Arc<AtomicBool>,
    #[serde(skip)]
    pub(super) cancel_notify: Arc<Notify>,
    #[serde(skip)]
    source_job_id: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedJob {
    #[serde(flatten)]
    job: Job,
    spec_path: PathBuf,
    output_dir: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_job_id: Option<String>,
}

#[derive(Debug)]
pub(super) struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}
pub(super) type ApiResult<T> = Result<T, ApiError>;
pub(super) fn bad(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, message.into())
}
pub(super) fn missing(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, message.into())
}
pub(super) fn conflict(message: impl Into<String>) -> ApiError {
    ApiError(StatusCode::CONFLICT, message.into())
}
fn internal() -> ApiError {
    ApiError(
        StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected local strategy API error.".into(),
    )
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SweepInfo {
    id: String,
    name: String,
    #[serde(rename = "sweepId")]
    sweep_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dataset_view_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SubmitBody {
    operation: String,
    #[serde(rename = "sweepId")]
    sweep_id: String,
    #[serde(rename = "luaSource")]
    lua_source: Option<String>,
    #[serde(rename = "sourceJobId")]
    source_job_id: Option<String>,
    pub(super) metric: Option<String>,
}

/// Return a router mounted at the same paths as the SvelteKit strategy API.
pub(super) fn api_state() -> ApiState {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let trader_dir = if let Some(path) = std::env::var_os("MIDAS_TRADER_DIR") {
        PathBuf::from(path)
    } else if cwd.join("trader/Cargo.toml").is_file() {
        cwd.join("trader")
    } else if cwd.join("Cargo.toml").is_file() {
        cwd.clone()
    } else if cwd.join("../trader/Cargo.toml").is_file() {
        cwd.join("../trader")
    } else {
        cwd
    };
    let specs_root = trader_dir.join(".run");
    let jobs_root = specs_root.join("webui-strategy-jobs");
    ApiState {
        trader_dir,
        specs_root,
        jobs_root,
        jobs: Arc::new(Mutex::new(HashMap::new())),
    }
}
pub(super) fn router() -> Router {
    let state = api_state();
    Router::new()
        .route("/api/strategy", get(catalog).post(submit))
        .route("/api/strategy/lua", post(submit_lua))
        .route(
            "/api/strategy/jobs/{id}",
            get(job_status).delete(cancel_job),
        )
        .merge(super::strategy_artifacts::router(state.clone()))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(state)
}

pub(super) async fn ensure_initialized(state: &ApiState) -> ApiResult<()> {
    fs::create_dir_all(&state.jobs_root)
        .await
        .map_err(|_| internal())?;
    let mut jobs = state.jobs.lock().await;
    if !jobs.is_empty() {
        return Ok(());
    }
    let mut entries = fs::read_dir(&state.jobs_root)
        .await
        .map_err(|_| internal())?;
    while let Some(entry) = entries.next_entry().await.map_err(|_| internal())? {
        if !entry.file_type().await.map(|f| f.is_dir()).unwrap_or(false) {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        if !valid_id(&id) {
            continue;
        }
        let path = entry.path();
        let Ok(data) = fs::read(path.join("job.json")).await else {
            continue;
        };
        let Ok(saved) = serde_json::from_slice::<PersistedJob>(&data) else {
            continue;
        };
        let mut job = saved.job;
        if job.id != id {
            continue;
        }
        let Ok(canonical_root) = fs::canonicalize(&state.jobs_root).await else {
            continue;
        };
        let Ok(canonical_spec) = fs::canonicalize(&saved.spec_path).await else {
            continue;
        };
        let canonical_output = match fs::canonicalize(&saved.output_dir).await {
            Ok(output)
                if fs::metadata(&output)
                    .await
                    .map(|m| m.is_dir())
                    .unwrap_or(false) =>
            {
                output
            }
            Err(_) if saved.output_dir == path.join("output") => {
                let Ok(parent) = fs::canonicalize(&path).await else {
                    continue;
                };
                parent.join("output")
            }
            _ => continue,
        };
        if !under(&canonical_root, &canonical_spec)
            || !under(&canonical_root, &canonical_output)
            || !fs::metadata(&canonical_spec)
                .await
                .map(|m| m.is_file())
                .unwrap_or(false)
        {
            continue;
        }
        job.spec_path = canonical_spec;
        job.output_dir = canonical_output;
        job.source_job_id = saved.source_job_id;
        job.cancelled = Arc::new(AtomicBool::new(false));
        job.cancel_notify = Arc::new(Notify::new());
        if job.status == "running" || job.status == "queued" {
            job.status = "failed".into();
            job.error = Some("The web server restarted before this job completed.".into());
            job.updated_at = now();
            let _ = persist(&state.jobs_root, &job).await;
        }
        jobs.insert(id, Arc::new(Mutex::new(job)));
    }
    Ok(())
}

fn valid_id(id: &str) -> bool {
    id.len() == 36
        && id.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}
fn now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // RFC3339 UTC without a date dependency; chrono is already a trader dependency.
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs as i64, 0)
        .unwrap_or_default()
        .to_rfc3339()
}
fn new_id() -> String {
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "{:08x}-{:04x}-4{:03x}-a{:03x}-{:012x}",
        t as u32,
        (t >> 32) as u16,
        (t >> 48) as u16 & 0xfff,
        n as u16 & 0xfff,
        (t as u64 ^ n) & 0xffffffffffff
    )
}
fn encode_id(path: &str) -> String {
    base64::Engine::encode(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
        path.as_bytes(),
    )
}
fn decode_id(id: &str) -> ApiResult<String> {
    if id.len() > 2048 {
        return Err(bad("sweepId is required."));
    }
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, id)
        .map_err(|_| bad("Unknown sweepId."))?;
    if base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, &bytes) != id {
        return Err(bad("Unknown sweepId."));
    }
    let value = String::from_utf8(bytes).map_err(|_| bad("Unknown sweepId."))?;
    let path = FsPath::new(&value);
    if value.contains('\0')
        || path.is_absolute()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(bad("Unknown sweepId."));
    }
    Ok(value)
}
fn under(root: &FsPath, path: &FsPath) -> bool {
    path.strip_prefix(root).is_ok()
}
async fn spec_by_id(state: &ApiState, id: &str) -> ApiResult<(PathBuf, Value)> {
    let relative = decode_id(id)?;
    let path = state.specs_root.join(relative);
    if !under(&state.specs_root, &path)
        || path.extension().and_then(|e| e.to_str()) != Some("json")
        || !path
            .strip_prefix(&state.specs_root)
            .unwrap_or(&path)
            .components()
            .any(|c| c.as_os_str() == "specs")
    {
        return Err(bad("Unknown sweepId."));
    }
    let canonical_root = fs::canonicalize(&state.specs_root)
        .await
        .map_err(|_| internal())?;
    let canonical = fs::canonicalize(&path)
        .await
        .map_err(|_| missing("Sweep spec no longer exists."))?;
    if !under(&canonical_root, &canonical) {
        return Err(bad("Unknown sweepId."));
    }
    let meta = fs::metadata(&canonical).await.map_err(|_| internal())?;
    if !meta.is_file() || meta.len() > MAX_SPEC {
        return Err(bad("Sweep spec is too large or invalid."));
    }
    let value: Value = serde_json::from_slice(&fs::read(&canonical).await.map_err(|_| internal())?)
        .map_err(|_| bad("Sweep spec is not valid JSON."))?;
    if value.get("sweep_id").and_then(Value::as_str).is_none()
        || value.get("name").and_then(Value::as_str).is_none()
        || value.get("base_dataset_view").is_none()
    {
        return Err(bad("Sweep spec is not a supported native replay spec."));
    }
    Ok((canonical, value))
}

async fn catalog(State(state): State<ApiState>) -> ApiResult<Json<Value>> {
    ensure_initialized(&state).await?;
    let mut sweeps = Vec::new();
    let mut stack = vec![state.specs_root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(mut entries) = fs::read_dir(&dir).await else {
            continue;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(kind) = entry.file_type().await else {
                continue;
            };
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            if kind.is_dir() {
                if [
                    "runs",
                    "cache",
                    "outputs",
                    "results",
                    "node_modules",
                    "target",
                ]
                .contains(&entry.file_name().to_string_lossy().as_ref())
                    || entry.file_name().to_string_lossy().ends_with("-output")
                {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if !kind.is_file()
                || path.extension().and_then(|s| s.to_str()) != Some("json")
                || path.file_name().and_then(|s| s.to_str()) == Some("sweep.json")
            {
                continue;
            }
            let Ok(rel) = path.strip_prefix(&state.specs_root) else {
                continue;
            };
            if !rel.components().any(|c| c.as_os_str() == "specs") {
                continue;
            }
            let Ok(meta) = fs::metadata(&path).await else {
                continue;
            };
            if meta.len() > MAX_SPEC {
                continue;
            }
            let Ok(value) = fs::read(&path).await else {
                continue;
            };
            let Ok(doc) = serde_json::from_slice::<Value>(&value) else {
                continue;
            };
            let (Some(sweep_id), Some(name)) = (
                doc.get("sweep_id").and_then(Value::as_str),
                doc.get("name").and_then(Value::as_str),
            ) else {
                continue;
            };
            let Some(dataset) = doc.pointer("/base_dataset_view/id").and_then(Value::as_str) else {
                continue;
            };
            sweeps.push(SweepInfo {
                id: encode_id(&rel.to_string_lossy()),
                name: name.into(),
                sweep_id: sweep_id.into(),
                dataset_view_id: Some(dataset.into()),
            });
        }
    }
    sweeps.sort_by(|a, b| a.sweep_id.cmp(&b.sweep_id));
    // Serialize job snapshots outside the shared map's lock.
    let jobs = state.jobs.lock().await;
    let mut snapshots = Vec::with_capacity(jobs.len());
    for job in jobs.values() {
        snapshots.push(serde_json::to_value(&*job.lock().await).unwrap_or(Value::Null));
    }
    drop(jobs);
    Ok(Json(json!({"sweeps": sweeps, "jobs": snapshots})))
}

fn json_rejection(error: axum::extract::rejection::JsonRejection) -> ApiError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Strategy request body is too large.".into(),
        )
    } else {
        bad("Request body must be JSON.")
    }
}

async fn submit(
    State(state): State<ApiState>,
    body: Result<Json<SubmitBody>, axum::extract::rejection::JsonRejection>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    submit_inner(state, body.map_err(json_rejection)?.0, false).await
}
async fn submit_lua(
    State(state): State<ApiState>,
    body: Result<Json<SubmitBody>, axum::extract::rejection::JsonRejection>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    submit_inner(state, body.map_err(json_rejection)?.0, true).await
}
async fn submit_inner(
    state: ApiState,
    body: SubmitBody,
    lua_only: bool,
) -> ApiResult<(StatusCode, Json<Value>)> {
    ensure_initialized(&state).await?;
    if !["validate", "run", "rank", "lua-validate", "lua-run"].contains(&body.operation.as_str())
        || (lua_only != body.operation.starts_with("lua-"))
    {
        return Err(bad(if lua_only {
            "Lua prototype operation must be lua-validate or lua-run."
        } else {
            "Unsupported strategy operation."
        }));
    }
    if !lua_only && body.lua_source.is_some() {
        return Err(bad(
            "luaSource is only accepted for Lua prototype operations.",
        ));
    }
    if lua_only && (body.source_job_id.is_some() || body.metric.is_some()) {
        return Err(bad("Unsupported Lua prototype request field."));
    }
    if let Some(source) = &body.lua_source {
        if source.trim().is_empty() {
            return Err(bad("luaSource must contain Lua source text."));
        }
        if source.len() > 12 * 1024 {
            return Err(ApiError(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Lua source is limited to 12 KiB.".into(),
            ));
        }
    } else if body.operation.starts_with("lua-") {
        return Err(bad("luaSource must contain Lua source text."));
    }
    let (_source_path, mut spec) = spec_by_id(&state, &body.sweep_id).await?;
    if body.operation == "rank" {
        let source_id = body
            .source_job_id
            .as_deref()
            .filter(|id| valid_id(id))
            .ok_or_else(|| bad("rank requires a completed sourceJobId."))?;
        let source = state
            .jobs
            .lock()
            .await
            .get(source_id)
            .cloned()
            .ok_or_else(|| conflict("Ranking requires a completed replay job from this web UI."))?;
        let source = source.lock().await;
        if source.operation != "run" || source.status != "complete" {
            return Err(conflict(
                "Ranking requires a completed replay job from this web UI.",
            ));
        }
        if spec.get("sweep_id").and_then(Value::as_str) != Some(source.sweep_id.as_str()) {
            return Err(bad("sourceJobId does not belong to this sweep."));
        }
        if ![
            "robustness",
            "net-pnl",
            "gross-pnl",
            "fees",
            "max-drawdown",
            "required-account-size",
            "return-required",
            "profit-factor",
            "win-rate",
            "trade-count",
            "average-trade",
            "average-giveback",
            "largest-giveback",
            "mfe-capture",
        ]
        .contains(&body.metric.as_deref().unwrap_or(""))
        {
            return Err(bad("Unsupported ranking metric."));
        }
    }
    let _gate = reserve_job_gate().ok_or_else(|| conflict("A strategy job is already active."))?;
    let existing: Vec<_> = state.jobs.lock().await.values().cloned().collect();
    for existing in existing {
        if matches!(existing.lock().await.status.as_str(), "queued" | "running") {
            return Err(conflict("A strategy job is already active."));
        }
    }
    let id = new_id();
    let dir = state.jobs_root.join(&id);
    let mut job_dir = fs::DirBuilder::new();
    #[cfg(unix)]
    job_dir.mode(0o700);
    job_dir.create(&dir).await.map_err(|_| internal())?;
    let output_dir = if body.operation == "rank" {
        let source_id = body.source_job_id.as_deref().expect("validated above");
        let source = state
            .jobs
            .lock()
            .await
            .get(source_id)
            .cloned()
            .ok_or_else(|| conflict("Ranking source job disappeared."))?;
        source.lock().await.output_dir.clone()
    } else {
        dir.join("output")
    };
    fs::create_dir_all(&output_dir)
        .await
        .map_err(|_| internal())?;
    if body.operation == "rank" {
        fs::create_dir(output_dir.join(&id))
            .await
            .map_err(|_| internal())?;
    }
    if let Some(source) = body.lua_source {
        if let Err(message) = configure_lua_spec(&mut spec, &output_dir, &source) {
            return Err(bad(message));
        }
    } else if body.operation != "rank" {
        spec["output_dir"] = json!(output_dir);
    }
    let spec_path = dir.join("sweep.json");
    write_private_new(
        &spec_path,
        &serde_json::to_vec_pretty(&spec).map_err(|_| internal())?,
    )
    .await
    .map_err(|_| internal())?;
    let now = now();
    let job = Job {
        id: id.clone(),
        operation: body.operation.clone(),
        sweep_id: spec
            .get("sweep_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        status: "queued".into(),
        created_at: now.clone(),
        updated_at: now,
        progress: None,
        result: None,
        error: None,
        metric: body.metric.clone(),
        spec_path,
        output_dir,
        cancelled: Arc::new(AtomicBool::new(false)),
        cancel_notify: Arc::new(Notify::new()),
        source_job_id: body.source_job_id.clone(),
    };
    let job = job;
    persist(&state.jobs_root, &job)
        .await
        .map_err(|_| internal())?;
    let shared = Arc::new(Mutex::new(job));
    state.jobs.lock().await.insert(id, shared.clone());
    tokio::spawn(run_job(state.clone(), shared.clone(), _gate));
    let response = {
        let j = shared.lock().await;
        serde_json::to_value(&*j).map_err(|_| internal())?
    };
    Ok((StatusCode::ACCEPTED, Json(response)))
}

async fn run_job(state: ApiState, job: Arc<Mutex<Job>>, _gate: ActiveGate) {
    let (op, spec_path, output_dir) = {
        let mut j = job.lock().await;
        j.status = "running".into();
        j.updated_at = now();
        (
            j.operation.clone(),
            j.spec_path.clone(),
            j.output_dir.clone(),
        )
    };
    let _ = persist_locked(&state.jobs_root, &job).await;
    let mut output = String::new();
    let mut status = "complete";
    let mut result = Value::Null;
    let mut error = None;
    let operation = op.clone();
    let rank_args = if operation == "rank" {
        let j = job.lock().await;
        let summary = j.output_dir.join("sweep-summary.json");
        let canonical_root = fs::canonicalize(&state.jobs_root).await.ok();
        let canonical_output = fs::canonicalize(&j.output_dir).await.ok();
        let canonical = fs::canonicalize(&summary).await.ok();
        match (canonical_root, canonical_output, canonical) {
            (Some(root), Some(output), Some(path))
                if under(&root, &output) && under(&output, &path) =>
            {
                let ranking_path = output.join(&j.id).join("sweep-ranking.json");
                let mut args = vec![
                    "rank-replay-sweep".into(),
                    "--summary".into(),
                    path.display().to_string(),
                    "--metric".into(),
                    j.metric.clone().unwrap_or_default(),
                    "--limit".into(),
                    "100".into(),
                    "--output".into(),
                    ranking_path.display().to_string(),
                ];
                let plan = output.join("sweep-plan.json");
                if let Ok(canonical_plan) = fs::canonicalize(&plan).await {
                    if under(&output, &canonical_plan) {
                        args.extend(["--plan".into(), canonical_plan.display().to_string()]);
                    }
                }
                Some(args)
            }
            _ => None,
        }
    } else {
        None
    };
    if operation == "rank" {
        match rank_args {
            Some(args) => match run_cli(&state, &job, args, &mut output).await {
                Ok(0) => {
                    let j = job.lock().await;
                    let artifact = j.output_dir.join(&j.id).join("sweep-ranking.json");
                    let summary = j.output_dir.join("sweep-summary.json");
                    match fs::canonicalize(&artifact).await {
                        Ok(canonical)
                            if under(&j.output_dir, &canonical)
                                && fs::metadata(&canonical)
                                    .await
                                    .map(|m| m.is_file())
                                    .unwrap_or(false) =>
                        {
                            let plan = j.output_dir.join("sweep-plan.json");
                            let source_plan = if fs::canonicalize(&plan)
                                .await
                                .map(|p| under(&j.output_dir, &p))
                                .unwrap_or(false)
                            {
                                Some(plan)
                            } else {
                                None
                            };
                            result = json!({"ranked":true,"metric":j.metric,"output":output.trim(),"artifactPath":canonical,"sourceSummary":summary,"sourcePlan":source_plan,"sourceJobId":j.source_job_id});
                        }
                        _ => {
                            status = "failed";
                            error=Some("Native ranking exited successfully without a verified ranking artifact.".into());
                        }
                    }
                }
                Ok(code) => {
                    status = "failed";
                    error = Some(if output.is_empty() {
                        format!("Native ranking exited {code}.")
                    } else {
                        output.clone()
                    });
                }
                Err(e) => {
                    status = if e == "cancelled" {
                        "cancelled"
                    } else {
                        "failed"
                    };
                    error = Some(e);
                }
            },
            None => {
                status = "failed";
                error =
                    Some("Completed replay output has no verified sweep summary to rank.".into());
            }
        }
    } else {
        let validation = run_cli(
            &state,
            &job,
            vec![
                "validate-replay-sweep".into(),
                "--spec".into(),
                spec_path.display().to_string(),
            ],
            &mut output,
        )
        .await;
        match validation {
            Ok(0) if op == "validate" || op == "lua-validate" => {
                result = json!({"validated":true,"output":output.trim()})
            }
            Ok(0) if op == "run" || op == "lua-run" => {
                let replay = run_cli(
                    &state,
                    &job,
                    vec![
                        "run-replay-sweep".into(),
                        "--spec".into(),
                        spec_path.display().to_string(),
                    ],
                    &mut output,
                )
                .await;
                match replay {
                    Ok(0) if op == "lua-run" => match validate_lua_summary(&output_dir).await {
                        Ok(()) => result = json!({"completed":true,"output":output.trim()}),
                        Err(e) => {
                            status = "failed";
                            error = Some(e);
                        }
                    },
                    Ok(0) => result = json!({"completed":true,"output":output.trim()}),
                    Ok(code) => {
                        status = "failed";
                        error = Some(if output.is_empty() {
                            format!("Native replay exited {code}.")
                        } else {
                            output.clone()
                        });
                    }
                    Err(e) => {
                        status = if e == "cancelled" {
                            "cancelled"
                        } else {
                            "failed"
                        };
                        error = Some(e);
                    }
                }
            }
            Ok(code) => {
                status = "failed";
                error = Some(if output.is_empty() {
                    format!("Native validation exited {code}.")
                } else {
                    output.clone()
                });
            }
            Err(e) => {
                status = if e == "cancelled" {
                    "cancelled"
                } else {
                    "failed"
                };
                error = Some(e);
            }
        }
    }
    if let Ok(bytes) = fs::read(output_dir.join("sweep-status.json")).await {
        if let Ok(value) = serde_json::from_slice(&bytes) {
            let mut j = job.lock().await;
            j.progress = Some(value);
        }
    }
    {
        let mut j = job.lock().await;
        j.status = status.into();
        j.result = if status == "complete" {
            Some(result)
        } else {
            None
        };
        j.error = error.map(|e| e.chars().take(2000).collect());
        j.updated_at = now();
    }
    let _ = persist_locked(&state.jobs_root, &job).await;
}

fn configure_lua_spec(
    spec: &mut Value,
    output_dir: &FsPath,
    source: &str,
) -> Result<(), &'static str> {
    if !spec
        .get("base_strategy")
        .and_then(Value::as_object)
        .is_some()
    {
        return Err("Selected sweep has no supported base strategy.");
    }
    let name = spec
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("Replay sweep")
        .chars()
        .take(100)
        .collect::<String>();
    spec["name"] = json!(format!("{name} — Lua prototype"));
    spec["output_dir"] = json!(output_dir);
    spec["execution_mode"] = json!("isolated_services");
    spec["parameters"] = json!([]);
    spec["constraints"] = json!([]);
    spec["max_runs"] = json!(1);
    spec["parallelism"] = json!(1);
    let formats = spec
        .get("output_formats")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut formats: Vec<Value> = formats;
    if !formats.iter().any(|f| f.as_str() == Some("json_summary")) {
        formats.push(json!("json_summary"));
    }
    spec["output_formats"] = Value::Array(formats);
    let base = spec
        .get_mut("base_strategy")
        .and_then(Value::as_object_mut)
        .ok_or("Selected sweep has no supported base strategy.")?;
    base.insert("kind".into(), json!("Lua"));
    base.insert("native_signal_timing".into(), json!("ClosedBar"));
    let mut lua = base
        .get("lua")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    lua.insert("schema_version".into(), json!(1));
    lua.insert("source".into(), json!(source));
    lua.insert("memory_limit_bytes".into(), json!(8 * 1024 * 1024));
    lua.insert("instruction_limit".into(), json!(100_000));
    base.insert("lua".into(), Value::Object(lua));
    Ok(())
}
async fn read_capped(reader: Option<tokio::process::ChildStdout>, limit: usize) -> Vec<u8> {
    let Some(mut reader) = reader else {
        return Vec::new();
    };
    let mut kept = Vec::with_capacity(limit);
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let take = (limit - kept.len()).min(n);
                kept.extend_from_slice(&buffer[..take]);
            }
        }
    }
    kept
}
async fn read_capped_err(reader: Option<tokio::process::ChildStderr>, limit: usize) -> Vec<u8> {
    let Some(mut reader) = reader else {
        return Vec::new();
    };
    let mut kept = Vec::with_capacity(limit);
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let take = (limit - kept.len()).min(n);
                kept.extend_from_slice(&buffer[..take]);
            }
        }
    }
    kept
}
async fn validate_lua_summary(output_dir: &FsPath) -> Result<(), String> {
    let summary_path = output_dir.join("sweep-summary.json");
    let canonical = fs::canonicalize(&summary_path).await.map_err(|_| {
        "Lua replay exited successfully but did not produce a valid sweep summary.".to_string()
    })?;
    if !under(output_dir, &canonical) {
        return Err("Lua replay summary escaped the job output directory.".into());
    }
    if fs::metadata(&canonical)
        .await
        .map(|m| m.len() > 16 * 1024 * 1024)
        .unwrap_or(true)
    {
        return Err("Lua replay summary is too large or unavailable.".into());
    }
    let bytes = fs::read(&canonical)
        .await
        .map_err(|_| "Unable to read Lua replay summary.".to_string())?;
    let summary: Value = serde_json::from_slice(&bytes).map_err(|_| {
        "Lua replay exited successfully but did not produce a valid sweep summary.".to_string()
    })?;
    let runs = summary
        .get("runs")
        .and_then(Value::as_array)
        .ok_or("Lua replay exited successfully but did not produce a valid sweep summary.")?;
    if runs.len() != 1 {
        return Err(format!(
            "Lua prototype expected one replay run, but the summary contains {}.",
            runs.len()
        ));
    }
    let run = &runs[0];
    if run.get("status").and_then(Value::as_str) != Some("completed") {
        let id = run
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap_or("(unknown)");
        let status = run
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let detail = run
            .get("error")
            .and_then(Value::as_str)
            .filter(|e| !e.trim().is_empty())
            .map(|e| format!(": {}", e.trim()))
            .unwrap_or_default();
        return Err(
            format!("Lua replay run {id} finished with status {status}{detail}")
                .chars()
                .take(2000)
                .collect(),
        );
    }
    if summary.get("run_count").and_then(Value::as_u64) != Some(1)
        || summary.get("completed_count").and_then(Value::as_u64) != Some(1)
        || summary.get("failed_count").and_then(Value::as_u64) != Some(0)
        || summary.get("skipped_count").and_then(Value::as_u64) != Some(0)
    {
        return Err("Lua replay summary has inconsistent run counts; expected one completed, unfailed, unskipped run.".into());
    }
    let result = run
        .get("result_path")
        .and_then(Value::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or("Lua replay completed without a result_path artifact.")?;
    let path = if PathBuf::from(result).is_absolute() {
        PathBuf::from(result)
    } else {
        output_dir.join(result)
    };
    let resolved = fs::canonicalize(path)
        .await
        .map_err(|_| "Lua replay result_path does not resolve to an output file.".to_string())?;
    if !under(output_dir, &resolved)
        || !fs::metadata(&resolved)
            .await
            .map(|m| m.is_file())
            .unwrap_or(false)
    {
        return Err("Lua replay result_path escaped the output directory.".into());
    }
    Ok(())
}

async fn run_cli(
    state: &ApiState,
    job: &Arc<Mutex<Job>>,
    args: Vec<String>,
    output: &mut String,
) -> Result<i32, String> {
    if job.lock().await.cancelled.load(Ordering::SeqCst) {
        return Err("cancelled".into());
    }
    let binary = std::env::var_os("MIDAS_TRADER_BIN")
        .map(|v| super::standalone::absolute(PathBuf::from(v)))
        .unwrap_or_else(|| {
            std::env::current_exe().unwrap_or_else(|_| state.trader_dir.join("target/debug/trader"))
        });
    let mut command = Command::new(binary);
    if let Some(config) = std::env::var_os("MIDAS_CONFIG")
        .or_else(|| std::env::var_os("TRADER_CONFIG"))
        .or_else(|| std::env::var_os("MIDAS_TUI_CONFIG"))
    {
        let config = PathBuf::from(config);
        let config = super::standalone::absolute(config);
        command.arg("--config").arg(config);
    }
    command
        .args(args)
        .current_dir(&state.trader_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("Unable to start native strategy command: {e}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_task = tokio::spawn(async move { read_capped(stdout, MAX_OUTPUT / 2).await });
    let stderr_task = tokio::spawn(async move { read_capped_err(stderr, MAX_OUTPUT / 2).await });
    let cancel = job.lock().await.cancel_notify.clone();
    let notified = cancel.notified();
    tokio::pin!(notified);
    if job.lock().await.cancelled.load(Ordering::SeqCst) {
        let _ = child.kill().await;
        return Err("cancelled".into());
    }
    let code = tokio::select! { result=child.wait()=> result.map_err(|e|e.to_string())?.code().unwrap_or(1), _=notified=> {
        #[cfg(any(target_os="linux",target_os="macos"))] if let Some(pid)=child.id() { unsafe { libc::kill(-(pid as i32), libc::SIGTERM); } }
        let _=child.kill().await; let _=child.wait().await; return Err("cancelled".into());
    } };
    let mut bytes = stdout_task.await.unwrap_or_default();
    bytes.extend(stderr_task.await.unwrap_or_default());
    if bytes.len() > MAX_OUTPUT {
        bytes = bytes.split_off(bytes.len() - MAX_OUTPUT);
    }
    *output = String::from_utf8_lossy(&bytes).to_string();
    if job.lock().await.cancelled.load(Ordering::SeqCst) {
        return Err("cancelled".into());
    }
    Ok(code)
}

async fn job_status(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    ensure_initialized(&state).await?;
    let job = state
        .jobs
        .lock()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| missing("Strategy job not found."))?;
    let mut j = job.lock().await;
    if j.status == "running" {
        if let Ok(b) = fs::read(j.output_dir.join("sweep-status.json")).await {
            j.progress = serde_json::from_slice(&b).ok();
        }
    }
    Ok(Json(serde_json::to_value(&*j).map_err(|_| internal())?))
}
async fn cancel_job(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    ensure_initialized(&state).await?;
    let job = state
        .jobs
        .lock()
        .await
        .get(&id)
        .cloned()
        .ok_or_else(|| missing("Strategy job not found."))?;
    {
        let mut j = job.lock().await;
        if j.status == "queued" || j.status == "running" {
            j.cancelled.store(true, Ordering::SeqCst);
            j.cancel_notify.notify_one();
            if j.status == "queued" {
                j.status = "cancelled".into();
                j.updated_at = now();
            }
        }
    }
    persist_locked(&state.jobs_root, &job)
        .await
        .map_err(|_| internal())?;
    Ok(Json(
        serde_json::to_value(&*job.lock().await).map_err(|_| internal())?,
    ))
}
async fn write_private_new(path: &FsPath, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path).await?;
    file.write_all(bytes).await?;
    file.flush().await
}
async fn persist(root: &FsPath, job: &Job) -> std::io::Result<()> {
    let dir = root.join(&job.id);
    let record = PersistedJob {
        job: job.clone(),
        spec_path: job.spec_path.clone(),
        output_dir: job.output_dir.clone(),
        source_job_id: job.source_job_id.clone(),
    };
    let temp = dir.join(format!(".job-{}.tmp", new_id()));
    let bytes = serde_json::to_vec_pretty(&record).map_err(std::io::Error::other)?;
    write_private_new(&temp, &bytes).await?;
    if let Err(error) = fs::rename(&temp, dir.join("job.json")).await {
        let _ = fs::remove_file(temp).await;
        return Err(error);
    }
    Ok(())
}
async fn persist_locked(root: &FsPath, job: &Arc<Mutex<Job>>) -> std::io::Result<()> {
    let j = job.lock().await;
    persist(root, &j).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lua_sweep_preserves_formats_and_base_lua_options() {
        let mut spec = json!({"name":"Base","output_formats":["csv"],"base_strategy":{"kind":"EMA","lua":{"custom":"retained"}}});
        configure_lua_spec(&mut spec, FsPath::new("/tmp/out"), "return 1").unwrap();
        assert_eq!(spec["output_formats"], json!(["csv", "json_summary"]));
        assert_eq!(spec["base_strategy"]["lua"]["custom"], json!("retained"));
        assert_eq!(spec["base_strategy"]["lua"]["source"], json!("return 1"));
        assert_eq!(spec["execution_mode"], json!("isolated_services"));
    }
    #[tokio::test]
    async fn lua_summary_requires_exact_single_run_counts() {
        let dir = std::env::temp_dir().join(format!("midas-lua-summary-{}", new_id()));
        fs::create_dir_all(&dir).await.unwrap();
        fs::write(dir.join("result.json"), b"{}").await.unwrap();
        let valid = json!({"runs":[{"status":"completed","result_path":"result.json"}],"run_count":1,"completed_count":1,"failed_count":0,"skipped_count":0});
        fs::write(
            dir.join("sweep-summary.json"),
            serde_json::to_vec(&valid).unwrap(),
        )
        .await
        .unwrap();
        assert!(validate_lua_summary(&dir).await.is_ok());
        let invalid = json!({"runs":[{"status":"completed","result_path":"result.json"}],"run_count":1,"completed_count":1,"failed_count":1,"skipped_count":0});
        fs::write(
            dir.join("sweep-summary.json"),
            serde_json::to_vec(&invalid).unwrap(),
        )
        .await
        .unwrap();
        assert!(
            validate_lua_summary(&dir)
                .await
                .unwrap_err()
                .contains("inconsistent run counts")
        );
        fs::remove_dir_all(dir).await.unwrap();
    }
    #[tokio::test]
    async fn recovers_legacy_validation_job_without_output_directory() {
        let root = std::env::temp_dir().join(format!("midas-web-recovery-{}", new_id()));
        let jobs_root = root.join("webui-strategy-jobs");
        let id = "12345678-1234-4234-a234-123456789abc";
        let dir = jobs_root.join(id);
        fs::create_dir_all(&dir).await.unwrap();
        let spec_path = dir.join("sweep.json");
        fs::write(&spec_path, b"{}").await.unwrap();
        let record = json!({"id":id,"operation":"validate","sweepId":"sweep","status":"complete","createdAt":"2026-01-01T00:00:00Z","updatedAt":"2026-01-01T00:00:00Z","specPath":spec_path,"outputDir":dir.join("output")});
        fs::write(dir.join("job.json"), serde_json::to_vec(&record).unwrap())
            .await
            .unwrap();
        let state = ApiState {
            trader_dir: root.clone(),
            specs_root: root.clone(),
            jobs_root,
            jobs: Arc::new(Mutex::new(HashMap::new())),
        };
        ensure_initialized(&state).await.unwrap();
        assert!(state.jobs.lock().await.contains_key(id));
        fs::remove_dir_all(root).await.unwrap();
    }
    #[test]
    fn sweep_ids_reject_noncanonical_or_traversal_paths() {
        let valid = encode_id("specs/base.json");
        assert_eq!(decode_id(&valid).unwrap(), "specs/base.json");
        assert!(decode_id(&encode_id("specs/../secret.json")).is_err());
        assert!(decode_id(&format!("{valid}=")).is_err());
    }
}

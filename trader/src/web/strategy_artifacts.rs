//! Bounded, path-contained strategy result, heatmap and candidate artifacts.
//!
//! Response shapes intentionally mirror `webui/src/lib/server/strategy-jobs.ts`.
use super::strategy::{ApiState, Job};
use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::fs;

pub const MAX_JSON: usize = 16 * 1024 * 1024;
pub const MAX_RESULT_JSON: usize = 4 * 1024 * 1024;
pub const MAX_CSV: usize = 8 * 1024 * 1024;
pub const MAX_CANDIDATE: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactError(pub u16, pub String);
type AResult<T> = Result<T, ArtifactError>;
fn err(code: u16, message: impl Into<String>) -> ArtifactError {
    ArtifactError(code, message.into())
}

#[derive(Clone, Debug)]
pub struct ArtifactJob {
    pub id: String,
    pub operation: String,
    pub status: String,
    pub sweep_id: String,
    pub output_dir: PathBuf,
    pub result: Option<Value>,
}

fn contained(root: &Path, path: &Path) -> bool {
    path.strip_prefix(root).is_ok()
}
async fn canonical_file(root: &Path, path: &Path, limit: usize) -> AResult<PathBuf> {
    let cr = fs::canonicalize(root)
        .await
        .map_err(|_| err(500, "Unable to resolve managed strategy directory."))?;
    let cp = fs::canonicalize(path)
        .await
        .map_err(|_| err(404, "Strategy artifact is unavailable."))?;
    if !contained(&cr, &cp) {
        return Err(err(500, "Strategy artifact escaped its managed directory."));
    }
    let m = fs::metadata(&cp)
        .await
        .map_err(|_| err(500, "Unable to inspect strategy artifact."))?;
    if !m.is_file() {
        return Err(err(500, "Strategy artifact is not a regular file."));
    }
    if m.len() > limit as u64 {
        return Err(err(400, "Strategy artifact exceeds its byte limit."));
    }
    Ok(cp)
}
async fn canonical_directory(root: &Path, path: &Path) -> AResult<PathBuf> {
    let cr = fs::canonicalize(root)
        .await
        .map_err(|_| err(500, "Unable to resolve managed strategy directory."))?;
    let cp = fs::canonicalize(path)
        .await
        .map_err(|_| err(404, "Strategy output is unavailable."))?;
    if !contained(&cr, &cp) {
        return Err(err(
            400,
            "Strategy output is outside the managed job directory.",
        ));
    }
    if !fs::metadata(&cp)
        .await
        .map_err(|_| err(500, "Unable to inspect strategy output."))?
        .is_dir()
    {
        return Err(err(500, "Strategy output is not a directory."));
    }
    Ok(cp)
}
async fn read_bounded(path: &Path, limit: usize) -> AResult<Vec<u8>> {
    let mut f = fs::File::open(path)
        .await
        .map_err(|_| err(500, "Unable to read strategy artifact."))?;
    use tokio::io::AsyncReadExt;
    let mut b = Vec::with_capacity(limit.min(64 * 1024));
    f.take((limit as u64) + 1)
        .read_to_end(&mut b)
        .await
        .map_err(|_| err(500, "Unable to read strategy artifact."))?;
    if b.len() > limit {
        return Err(err(400, "Strategy artifact exceeds its byte limit."));
    }
    Ok(b)
}
async fn json_artifact(root: &Path, path: &Path, limit: usize) -> AResult<Value> {
    let p = canonical_file(root, path, limit).await?;
    serde_json::from_slice(&read_bounded(&p, limit).await?)
        .map_err(|_| err(400, "Strategy artifact is not valid JSON."))
}

/// Results for completed replay jobs. Result paths come from summary but must resolve under output.
pub async fn replay_results(job: &ArtifactJob) -> AResult<Value> {
    if !matches!(job.operation.as_str(), "run" | "lua-run") {
        return Err(err(404, "Replay job not found."));
    }
    if job.status != "complete" {
        return Err(err(
            409,
            "Replay artifacts are available after the job completes.",
        ));
    }
    let root = fs::canonicalize(&job.output_dir)
        .await
        .map_err(|_| err(500, "Replay output is unavailable."))?;
    let summary = json_artifact(&root, &root.join("sweep-summary.json"), MAX_JSON).await?;
    let source = summary
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| err(400, "Replay summary has an unsupported schema."))?;
    if source.len() > 10_000 {
        return Err(err(400, "Replay summary has excessive run rows."));
    }
    let mut runs = Vec::new();
    for run in source.iter().filter(|r| {
        r.get("status").and_then(Value::as_str) == Some("completed")
            && r.get("result_path").and_then(Value::as_str).is_some()
    }) {
        let raw = run["result_path"].as_str().unwrap();
        let rp = Path::new(raw);
        let result_path = if rp.is_absolute() {
            rp.to_path_buf()
        } else {
            root.join(rp)
        };
        let Ok(result_path) = canonical_file(&root, &result_path, MAX_RESULT_JSON).await else {
            continue;
        };
        let result: Value =
            serde_json::from_slice(&read_bounded(&result_path, MAX_RESULT_JSON).await?)
                .map_err(|_| err(400, "Replay result is not valid JSON."))?;
        let run_dir = result_path
            .parent()
            .ok_or_else(|| err(500, "Invalid replay result path."))?;
        let equity = read_csv_if_contained(&root, &run_dir.join("equity.csv")).await?;
        let trades = read_csv_if_contained(&root, &run_dir.join("trades.csv")).await?;
        let points:Vec<Value>=equity.into_iter().filter_map(|r| {
            let e=r.get("equity")?.as_str()?.parse::<f64>().ok()?;
            if !e.is_finite(){return None}
            Some(json!({"timestampNs":r.get("timestamp_ns"),"equity":e,"cumulativeNetPnl":r.get("cumulative_net_pnl").and_then(Value::as_str).and_then(|v|v.parse::<f64>().ok()).filter(|v|v.is_finite())}))
        }).collect();
        let mut metrics = Map::new();
        for (from, to) in [
            ("gross_pnl", "grossPnl"),
            ("net_pnl", "netPnl"),
            ("fees", "fees"),
            ("max_drawdown", "maxDrawdown"),
            ("trade_count", "tradeCount"),
        ] {
            metrics.insert(to.into(), run.get(from).cloned().unwrap_or(Value::Null));
        }
        runs.push(json!({"runId":run.get("run_id"),"metrics":metrics,"equity":points,"trades":trades,"result":result}));
    }
    let response = json!({"sweepId":job.sweep_id,"runs":runs});
    if serde_json::to_vec(&response)
        .map_err(|_| err(500, "Unable to encode replay results."))?
        .len()
        > 32 * 1024 * 1024
    {
        return Err(err(400, "Replay results exceed the response size limit."));
    }
    Ok(response)
}
async fn read_csv_if_contained(root: &Path, path: &Path) -> AResult<Vec<Map<String, Value>>> {
    let p = match canonical_file(root, path, MAX_CSV).await {
        Ok(p) => p,
        Err(ArtifactError(404, _)) => return Ok(vec![]),
        Err(ArtifactError(500, m)) if m.contains("escaped") => return Ok(vec![]),
        Err(e) => return Err(e),
    };
    parse_csv(
        std::str::from_utf8(&read_bounded(&p, MAX_CSV).await?)
            .map_err(|_| err(400, "Strategy CSV is not UTF-8."))?,
    )
}

/// Minimal RFC 4180 parser supporting quoted commas, quotes and newlines.
pub fn parse_csv(text: &str) -> AResult<Vec<Map<String, Value>>> {
    if text.trim().is_empty() {
        return Ok(vec![]);
    }
    let mut rows: Vec<Vec<String>> = vec![vec![]];
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"')
                } else {
                    quoted = false
                }
            } else {
                field.push(c)
            }
        } else {
            match c {
                '"' if field.is_empty() => quoted = true,
                ',' => rows.last_mut().unwrap().push(std::mem::take(&mut field)),
                '\n' => {
                    rows.last_mut().unwrap().push(std::mem::take(&mut field));
                    rows.push(vec![])
                }
                '\r' => {}
                _ => field.push(c),
            }
        }
    }
    if quoted {
        return Err(err(400, "Invalid strategy CSV: unterminated quoted field."));
    }
    rows.last_mut().unwrap().push(field);
    let rows: Vec<_> = rows
        .into_iter()
        .filter(|r| r.iter().any(|s| !s.trim().is_empty()))
        .collect();
    if rows.is_empty() {
        return Ok(vec![]);
    }
    let headers = &rows[0];
    let mut out = Vec::new();
    for row in rows.iter().skip(1) {
        if row.len() != headers.len() {
            return Err(err(400, "Invalid strategy CSV: inconsistent column count."));
        }
        let mut m = Map::new();
        for (h, v) in headers.iter().zip(row) {
            m.insert(h.clone(), Value::String(v.clone()));
        }
        out.push(m);
    }
    Ok(out)
}

/// Build a truthful grid; values and statuses are derived only from plan and summary records.
pub fn heatmap(
    plan: &Value,
    summary: &Value,
    x: Option<&str>,
    y: Option<&str>,
    metric: Option<&str>,
) -> AResult<Value> {
    let children = plan
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            err(
                400,
                "Native sweep plan or summary has an unsupported schema.",
            )
        })?;
    let runs = summary
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            err(
                400,
                "Native sweep plan or summary has an unsupported schema.",
            )
        })?;
    if children.len() > 10000 || runs.len() > 10000 {
        return Err(err(400, "Native sweep plan or summary has excessive rows."));
    }
    let scalar = |v: &Value| v.is_null() || v.is_string() || v.is_boolean() || v.as_f64().is_some();
    let mut axes: Vec<(String, Vec<Value>)> = Vec::new();
    for c in children {
        if let Some(p) = c.get("parameter_values").and_then(Value::as_object) {
            for (k, v) in p {
                if scalar(v) {
                    let a = axes.iter_mut().find(|a| a.0 == *k);
                    if let Some(a) = a {
                        if !a.1.contains(v) {
                            a.1.push(v.clone())
                        }
                    } else {
                        axes.push((k.clone(), vec![v.clone()]))
                    }
                }
            }
        }
    }
    let xp = x
        .map(str::to_owned)
        .or_else(|| axes.first().map(|a| a.0.clone()));
    let yp = y.map(str::to_owned).or_else(|| {
        axes.iter()
            .find(|a| Some(&a.0) != xp.as_ref())
            .map(|a| a.0.clone())
    });
    let (xp, yp) = match (xp, yp) {
        (Some(x), Some(y))
            if x != y && axes.iter().any(|a| a.0 == x) && axes.iter().any(|a| a.0 == y) =>
        {
            (x, y)
        }
        _ => return Err(err(400, "Choose two distinct sweep parameters.")),
    };
    let labels = [
        ("gross_pnl", "Gross P&L"),
        ("net_pnl", "Net P&L"),
        ("fees", "Fees"),
        ("max_drawdown", "Max drawdown"),
        ("trade_count", "Trades"),
        ("fill_count", "Fills"),
    ];
    let available: Vec<_> = labels
        .iter()
        .filter(|(k, _)| {
            runs.iter().any(|r| {
                r.get(*k)
                    .and_then(Value::as_f64)
                    .is_some_and(f64::is_finite)
            })
        })
        .collect();
    let mk = metric
        .filter(|m| available.iter().any(|(k, _)| k == m))
        .or_else(|| {
            available
                .iter()
                .find(|(k, _)| *k == "net_pnl")
                .map(|(k, _)| *k)
        })
        .or_else(|| available.first().map(|(k, _)| *k))
        .ok_or_else(|| err(400, "Sweep summary has no supported numeric metrics."))?;
    let xs = &axes.iter().find(|a| a.0 == xp).unwrap().1;
    let ys = &axes.iter().find(|a| a.0 == yp).unwrap().1;
    if xs.len().saturating_mul(ys.len()) > 10_000 {
        return Err(err(400, "Sweep heatmap has excessive cells."));
    }
    let mut cell_rows = Vec::new();
    for yy in ys {
        let mut cells = Vec::new();
        for xx in xs {
            let matches: Vec<_> = children
                .iter()
                .filter(|c| {
                    c.pointer(&format!("/parameter_values/{}", escape_pointer(&xp))) == Some(xx)
                        && c.pointer(&format!("/parameter_values/{}", escape_pointer(&yp)))
                            == Some(yy)
                })
                .collect();
            let mut candidates = Vec::new();
            for child in &matches {
                let idx = child.get("run_index");
                let run = runs.iter().find(|r| r.get("run_index") == idx);
                let status = if run.is_none() {
                    "missing_result"
                } else if run.unwrap().get("skipped").and_then(Value::as_bool) == Some(true) {
                    "skipped"
                } else if run.unwrap().get("status").and_then(Value::as_str) == Some("completed") {
                    "completed"
                } else {
                    run.unwrap()
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                };
                let value = if status == "completed" {
                    run.and_then(|r| r.get(mk))
                        .filter(|v| v.as_f64().is_some_and(f64::is_finite))
                        .cloned()
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                candidates.push(json!({"candidateId":child.get("run_id").or_else(||run.and_then(|r|r.get("run_id"))).unwrap_or(&Value::Null),"runIndex":idx,"status":status,"value":value,"error":run.and_then(|r|r.get("error")).unwrap_or(&Value::Null),"parameters":child.get("parameter_values")}));
            }
            cells
                .push(json!({"x":xx,"y":yy,"planned":!matches.is_empty(),"candidates":candidates}));
        }
        cell_rows.push(json!({"y":yy,"cells":cells}));
    }
    Ok(
        json!({"sweepId":summary.get("sweep_id").or_else(||plan.pointer("/spec/sweep_id")),"axes":axes.iter().map(|(path,values)|json!({"path":path,"values":values})).collect::<Vec<_>>(),"selectedAxes":{"x":xp,"y":yp},"metrics":available.iter().map(|(key,label)|json!({"key":key,"label":label})).collect::<Vec<_>>(),"selectedMetric":mk,"cells":cell_rows,"counts":{"planned":children.len(),"completed":summary.get("completed_count").cloned().unwrap_or_else(||json!(runs.iter().filter(|r|r.get("status").and_then(Value::as_str)==Some("completed")).count())),"failed":summary.get("failed_count").cloned().unwrap_or_else(||json!(runs.iter().filter(|r|r.get("status").and_then(Value::as_str)==Some("failed")).count())),"skipped":summary.get("skipped_count").cloned().unwrap_or_else(||json!(runs.iter().filter(|r|r.get("skipped").and_then(Value::as_bool)==Some(true)).count()))}}),
    )
}
fn escape_pointer(s: &str) -> String {
    s.replace('~', "~0").replace('/', "~1")
}

fn response_error(e: ArtifactError) -> Response {
    (
        StatusCode::from_u16(e.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(json!({"error":e.1})),
    )
        .into_response()
}
fn artifact_job(j: &Job) -> ArtifactJob {
    ArtifactJob {
        id: j.id.clone(),
        operation: j.operation.clone(),
        status: j.status.clone(),
        sweep_id: j.sweep_id.clone(),
        output_dir: j.output_dir.clone(),
        result: j.result.clone(),
    }
}
async fn fetch_job(
    state: &ApiState,
    id: &str,
) -> Result<std::sync::Arc<tokio::sync::Mutex<Job>>, Response> {
    if !valid_job_id(id) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error":"Strategy job not found."})),
        )
            .into_response());
    }
    if let Err(e) = super::strategy::ensure_initialized(state).await {
        return Err(e.into_response());
    }
    state.jobs.lock().await.get(id).cloned().ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"Strategy job not found."})),
        )
            .into_response()
    })
}

pub(super) async fn results_handler(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let job = match fetch_job(&state, &id).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let j = job.lock().await;
    match replay_results(&artifact_job(&j)).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => response_error(e),
    }
}
pub(super) async fn heatmap_handler(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let job = match fetch_job(&state, &id).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let j = job.lock().await;
    if j.operation != "run" {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"Replay job not found."})),
        )
            .into_response();
    }
    if j.status != "complete" {
        return response_error(err(
            409,
            "Sweep heatmap is available after the replay job completes.",
        ));
    }
    let root = match fs::canonicalize(&j.output_dir).await {
        Ok(v) => v,
        Err(_) => return response_error(err(500, "Replay output is unavailable.")),
    };
    let plan = match json_artifact(&root, &root.join("sweep-plan.json"), MAX_JSON).await {
        Ok(v) => v,
        Err(e) => return response_error(e),
    };
    let summary = match json_artifact(&root, &root.join("sweep-summary.json"), MAX_JSON).await {
        Ok(v) => v,
        Err(e) => return response_error(e),
    };
    match heatmap(
        &plan,
        &summary,
        q.get("x").map(String::as_str),
        q.get("y").map(String::as_str),
        q.get("metric").map(String::as_str),
    ) {
        Ok(v) => Json(v).into_response(),
        Err(e) => response_error(e),
    }
}
pub(super) async fn candidate_get_handler(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let job = match fetch_job(&state, &id).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let guard = job.lock().await;
    let j = artifact_job(&guard);
    drop(guard);
    if let Err(e) = verify_source_job(&state, &j).await {
        return response_error(e);
    }
    match candidate_preview(&state, &j).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => response_error(e),
    }
}
pub(super) async fn candidate_post_handler(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(v)) => v,
        Err(_) => return response_error(err(400, "Expected a JSON body with runId.")),
    };
    let Some(run_id) = body
        .get("runId")
        .and_then(Value::as_str)
        .filter(|s| s.len() <= 200)
    else {
        return response_error(err(400, "Expected a candidate runId."));
    };
    let job = match fetch_job(&state, &id).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let guard = job.lock().await;
    let j = artifact_job(&guard);
    drop(guard);
    if let Err(e) = verify_source_job(&state, &j).await {
        return response_error(e);
    }
    let preview = match candidate_preview(&state, &j).await {
        Ok(v) => v,
        Err(e) => return response_error(e),
    };
    let Some(candidate) = preview["candidates"]
        .as_array()
        .and_then(|a| {
            a.iter()
                .find(|c| c.pointer("/row/run_id").and_then(Value::as_str) == Some(run_id))
        })
        .and_then(|v| v.get("candidate"))
    else {
        return response_error(err(404, "Candidate runId was not found in this ranking."));
    };
    match save_candidate(&state.specs_root, candidate).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => response_error(e),
    }
}

async fn verify_source_job(state: &ApiState, job: &ArtifactJob) -> AResult<()> {
    let source_summary = fs::canonicalize(requested_source_summary(job)?)
        .await
        .map_err(|_| err(409, "Ranking job has no verified source summary."))?;
    let jobs_root = fs::canonicalize(&state.jobs_root)
        .await
        .map_err(|_| err(500, "Managed strategy jobs are unavailable."))?;
    let jobs = state.jobs.lock().await;
    for source in jobs.values() {
        let source = source.lock().await;
        if source.operation != "run"
            || source.status != "complete"
            || source.sweep_id != job.sweep_id
        {
            continue;
        }
        let Ok(out) = canonical_directory(&jobs_root, &source.output_dir).await else {
            continue;
        };
        let Ok(summary) =
            canonical_file(&jobs_root, &out.join("sweep-summary.json"), MAX_JSON).await
        else {
            continue;
        };
        if summary == source_summary {
            return Ok(());
        }
    }
    Err(err(
        409,
        "Completed source replay job for this ranking is unavailable.",
    ))
}

fn requested_source_summary(job: &ArtifactJob) -> AResult<PathBuf> {
    let requested = job
        .result
        .as_ref()
        .and_then(|v| v.get("sourceSummary"))
        .and_then(Value::as_str)
        .ok_or_else(|| err(409, "Ranking job has no verified source summary."))?;
    if !Path::new(requested).is_absolute() {
        return Err(err(409, "Ranking job has no verified source summary."));
    }
    Ok(PathBuf::from(requested))
}

async fn find_source_job(
    state: &ApiState,
    job: &ArtifactJob,
    summary: &Path,
) -> AResult<super::strategy::Job> {
    let jobs = state.jobs.lock().await;
    for candidate in jobs.values() {
        let candidate = candidate.lock().await;
        if candidate.operation != "run"
            || candidate.status != "complete"
            || candidate.sweep_id != job.sweep_id
        {
            continue;
        }
        let Ok(output) = canonical_directory(&state.jobs_root, &candidate.output_dir).await else {
            continue;
        };
        let Ok(candidate_summary) = canonical_file(
            &state.jobs_root,
            &output.join("sweep-summary.json"),
            MAX_JSON,
        )
        .await
        else {
            continue;
        };
        if candidate_summary == summary {
            return Ok(candidate.clone());
        }
    }
    Err(err(
        409,
        "Completed source replay job for this ranking is unavailable.",
    ))
}

static VALIDATION_SEQ: AtomicU64 = AtomicU64::new(1);
async fn create_validation_dir() -> AResult<PathBuf> {
    let base = std::env::temp_dir();
    for _ in 0..8 {
        let n = VALIDATION_SEQ.fetch_add(1, Ordering::Relaxed);
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = base.join(format!(
            "midas-candidate-{}-{time:x}-{n:x}",
            std::process::id()
        ));
        match fs::create_dir(&path).await {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))
                        .await
                        .map_err(|_| {
                            err(500, "Unable to secure candidate validation directory.")
                        })?;
                }
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(err(500, "Unable to create candidate validation directory.")),
        }
    }
    Err(err(500, "Unable to create candidate validation directory."))
}

async fn validate_native_spec(
    state: &ApiState,
    dir: &Path,
    spec_path: &Path,
    plan_path: &Path,
    spec: &[u8],
) -> AResult<Value> {
    use tokio::{
        io::AsyncReadExt,
        process::Command,
        time::{Duration, timeout},
    };
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(spec_path)
        .await
        .map_err(|_| err(500, "Unable to prepare candidate validation."))?;
    use tokio::io::AsyncWriteExt;
    f.write_all(spec)
        .await
        .map_err(|_| err(500, "Unable to prepare candidate validation."))?;
    f.flush()
        .await
        .map_err(|_| err(500, "Unable to prepare candidate validation."))?;
    drop(f);
    let binary = std::env::var_os("MIDAS_TRADER_BIN")
        .map(|p| super::standalone::absolute(PathBuf::from(p)))
        .unwrap_or_else(|| {
            std::env::current_exe().unwrap_or_else(|_| state.trader_dir.join("target/debug/trader"))
        });
    let mut cmd = Command::new(binary);
    if let Some(config) = std::env::var_os("MIDAS_CONFIG")
        .or_else(|| std::env::var_os("TRADER_CONFIG"))
        .or_else(|| std::env::var_os("MIDAS_TUI_CONFIG"))
    {
        let config = super::standalone::absolute(PathBuf::from(config));
        cmd.arg("--config").arg(config);
    }
    cmd.args(["validate-replay-sweep", "--spec"])
        .arg(spec_path)
        .arg("--output")
        .arg(plan_path)
        .current_dir(&state.trader_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|_| {
        err(
            400,
            "Native strategy configuration validation failed: unable to start validator",
        )
    })?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| err(400, "Native strategy configuration validation failed."))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| err(400, "Native strategy configuration validation failed."))?;
    let out_task = tokio::spawn(async move {
        let mut b = Vec::new();
        let _ = stdout.take(48 * 1024 + 1).read_to_end(&mut b).await;
        b
    });
    let err_task = tokio::spawn(async move {
        let mut b = Vec::new();
        let _ = stderr.take(48 * 1024 + 1).read_to_end(&mut b).await;
        b
    });
    let status = timeout(Duration::from_secs(15), child.wait())
        .await
        .map_err(|_| err(400, "Native strategy configuration validation timed out."))?
        .map_err(|_| err(400, "Native strategy configuration validation failed."))?;
    let mut output = out_task.await.unwrap_or_default();
    output.extend(err_task.await.unwrap_or_default());
    if !status.success() {
        let detail = String::from_utf8_lossy(&output);
        return Err(err(
            400,
            format!(
                "Native strategy configuration validation failed: {}",
                detail.chars().take(1000).collect::<String>()
            ),
        ));
    }
    let root = fs::canonicalize(dir)
        .await
        .map_err(|_| err(500, "Candidate validation output is unavailable."))?;
    json_artifact(&root, plan_path, MAX_CANDIDATE).await
}

fn valid_job_id(id: &str) -> bool {
    id.len() == 36
        && id.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}
pub(super) fn router(_state: ApiState) -> axum::Router<ApiState> {
    use axum::routing::get;
    axum::Router::new()
        .route("/api/strategy/jobs/{id}/results", get(results_handler))
        .route("/api/strategy/jobs/{id}/heatmap", get(heatmap_handler))
        .route(
            "/api/strategy/jobs/{id}/candidate",
            get(candidate_get_handler).post(candidate_post_handler),
        )
}

/// Verify exact ranking/plan correspondence and produce TUI candidate envelopes.
pub async fn candidate_preview(state: &ApiState, job: &ArtifactJob) -> AResult<Value> {
    if job.operation != "rank" {
        return Err(err(404, "Ranking job not found."));
    }
    if job.status != "complete" {
        return Err(err(
            409,
            "Candidate import is available after ranking completes.",
        ));
    }
    let jobs = fs::canonicalize(&state.jobs_root)
        .await
        .map_err(|_| err(500, "Managed strategy jobs are unavailable."))?;
    let output = canonical_directory(&jobs, &job.output_dir).await?;
    let rankpath = output.join(&job.id).join("sweep-ranking.json");
    let ranking = json_artifact(&output, &rankpath, MAX_CANDIDATE).await?;
    let rows = ranking
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            err(
                400,
                "Native ranking has malformed or excessive candidate rows.",
            )
        })?;
    if ranking.get("schema_version").and_then(Value::as_u64) != Some(1)
        || rows.len() > 100
        || rows.iter().any(|r| {
            !r.is_object()
                || r.get("run_id").and_then(Value::as_str).is_none()
                || r.get("run_index").and_then(Value::as_u64).is_none()
                || r.get("rank").and_then(Value::as_u64).is_none()
        })
    {
        return Err(err(
            400,
            "Native ranking has malformed or excessive candidate rows.",
        ));
    }
    let source = job
        .result
        .as_ref()
        .and_then(|v| v.get("sourceSummary"))
        .and_then(Value::as_str)
        .ok_or_else(|| err(409, "Ranking job has no verified source summary."))?;
    if !Path::new(source).is_absolute() {
        return Err(err(409, "Ranking job has no verified source summary."));
    }
    let summary_path = canonical_file(&jobs, Path::new(source), MAX_CANDIDATE).await?;
    if summary_path.file_name().and_then(|v| v.to_str()) != Some("sweep-summary.json") {
        return Err(err(
            400,
            "Source summary is outside the managed strategy job directory.",
        ));
    }
    let saved_plan_path = summary_path
        .parent()
        .ok_or_else(|| err(400, "Sweep plan is unavailable."))?
        .join("sweep-plan.json");
    let saved_plan_path = canonical_file(&jobs, &saved_plan_path, MAX_CANDIDATE).await?;
    let saved_plan = json_artifact(&jobs, &saved_plan_path, MAX_CANDIDATE).await?;
    if saved_plan.get("schema_version").and_then(Value::as_u64) != Some(1)
        || saved_plan
            .get("children")
            .and_then(Value::as_array)
            .is_none_or(|v| v.len() > 10000)
        || saved_plan.pointer("/spec/base_dataset_view").is_none()
    {
        return Err(err(400, "Native sweep plan has an unsupported schema."));
    }
    let source_prop = ranking
        .get("source_summary")
        .and_then(Value::as_str)
        .ok_or_else(|| err(400, "Native ranking has no source summary provenance."))?;
    if source_prop.len() > 4096 {
        return Err(err(400, "Native ranking has no source summary provenance."));
    }
    let ranked_summary = Path::new(source_prop);
    let ranked_summary = if ranked_summary.is_absolute() {
        ranked_summary.to_path_buf()
    } else {
        output.join(ranked_summary)
    };
    if fs::canonicalize(ranked_summary).await.ok().as_deref() != Some(summary_path.as_path()) {
        return Err(err(
            400,
            "Ranking source summary does not match its completed replay job.",
        ));
    }
    let source_job = find_source_job(state, job, &summary_path).await?;
    let spec_root = fs::canonicalize(&state.jobs_root)
        .await
        .map_err(|_| err(500, "Managed strategy jobs are unavailable."))?;
    let source_root = canonical_directory(&spec_root, &spec_root.join(&source_job.id)).await?;
    let spec_path = canonical_file(&source_root, &source_job.spec_path, 2_000_000).await?;
    let spec = read_bounded(&spec_path, 2_000_000).await?;
    let validation_dir = create_validation_dir().await?;
    let expanded_path = validation_dir.join("expanded-plan.json");
    let spec_temp = validation_dir.join("source-spec.json");
    let validation =
        validate_native_spec(state, &validation_dir, &spec_temp, &expanded_path, &spec).await;
    let plan = match validation {
        Ok(plan) => plan,
        Err(e) => {
            let _ = fs::remove_dir_all(&validation_dir).await;
            return Err(e);
        }
    };
    let _ = fs::remove_dir_all(&validation_dir).await;
    let children = plan
        .get("children")
        .and_then(Value::as_array)
        .filter(|v| v.len() <= 10000)
        .ok_or_else(|| err(400, "Native sweep plan has an unsupported schema."))?;
    let mut candidates = Vec::new();
    for row in rows {
        let run_id = row["run_id"].as_str().unwrap();
        let run_index = row["run_index"].as_u64().unwrap();
        let child = children
            .iter()
            .find(|c| c.get("run_id").and_then(Value::as_str) == Some(run_id))
            .ok_or_else(|| {
                err(
                    400,
                    format!("Candidate {run_id} is absent from the source plan."),
                )
            })?;
        if child["run_index"].as_u64() != Some(run_index)
            || child.get("parameter_values") != row.get("parameter_values")
            || child.get("overrides") != row.get("overrides")
        {
            return Err(err(400, "Candidate parameters do not match source plan."));
        }
        let config = child
            .get("resolved_strategy")
            .filter(|v| v.is_object())
            .ok_or_else(|| err(400, "Candidate has no resolved strategy configuration."))?;
        if config.get("kind").and_then(Value::as_str).is_none()
            || config
                .get("native_strategy")
                .and_then(Value::as_str)
                .is_none()
            || !config.get("native_hma").is_some_and(Value::is_object)
            || !config.get("native_ema").is_some_and(Value::is_object)
        {
            return Err(err(
                400,
                "Candidate execution configuration is missing required native strategy fields.",
            ));
        }
        if let Some(q) = config.get("order_qty") {
            if q.as_u64().is_none_or(|v| v == 0) {
                return Err(err(400, "Candidate order_qty must be a positive integer."));
            }
        }
        let rank = row["rank"].as_u64().unwrap();
        if rank == 0
            || row
                .get("metric_value")
                .is_some_and(|v| !v.is_null() && v.as_f64().is_none_or(|n| !n.is_finite()))
        {
            return Err(err(400, "Candidate ranking metadata is invalid."));
        }
        let envelope = json!({"schema_version":1,"provenance":{"ranking_artifact":rankpath.display().to_string(),"source_summary":summary_path.display().to_string(),"source_plan":saved_plan_path.display().to_string(),"sweep_id":ranking["sweep_id"],"run_id":run_id,"rank":rank,"metric":ranking["metric"],"metric_value":row.get("metric_value").cloned().unwrap_or(Value::Null),"fee_scenario":row["fee_scenario"]},"execution_config":config});
        candidates.push(json!({"row":row,"candidate":envelope}));
    }
    Ok(json!({"candidates":candidates}))
}

/// Atomically publish a candidate envelope without replacing an existing import.
pub async fn save_candidate(specs_root: &Path, candidate: &Value) -> AResult<Value> {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let dir = specs_root.join("strategy-candidates");
    let mut directory = fs::DirBuilder::new();
    directory.recursive(true);
    #[cfg(unix)]
    directory.mode(0o700);
    directory
        .create(&dir)
        .await
        .map_err(|_| err(500, "Unable to create strategy candidates directory."))?;
    let root = fs::canonicalize(specs_root)
        .await
        .map_err(|_| err(500, "Unable to resolve specs directory."))?;
    let canonical = fs::canonicalize(&dir)
        .await
        .map_err(|_| err(500, "Unable to resolve candidate directory."))?;
    if !contained(&root, &canonical)
        || !fs::metadata(&canonical)
            .await
            .map_err(|_| err(500, "Unable to inspect candidate directory."))?
            .is_dir()
    {
        return Err(err(
            400,
            "Candidate destination is outside the managed strategy directory.",
        ));
    }
    let safe = |s: &str| {
        let v: String = s
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(100)
            .collect();
        if v.is_empty() {
            "candidate".to_string()
        } else {
            v
        }
    };
    let prov = &candidate["provenance"];
    let stem = format!(
        "{}-{}",
        safe(prov["sweep_id"].as_str().unwrap_or("candidate")),
        safe(prov["run_id"].as_str().unwrap_or("candidate"))
    );
    let bytes = serde_json::to_vec_pretty(candidate)
        .map_err(|_| err(500, "Unable to encode candidate."))?;
    if bytes.len() > MAX_CANDIDATE {
        return Err(err(400, "Candidate artifact exceeds its byte limit."));
    }
    let nonce = SEQ.fetch_add(1, Ordering::Relaxed);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = canonical.join(format!(".{stem}.{}-{nonce}.tmp", time));
    use tokio::io::AsyncWriteExt;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(&temp)
        .await
        .map_err(|_| err(500, "Unable to reserve candidate temporary file."))?;
    let result = async {
        file.write_all(&bytes)
            .await
            .map_err(|_| err(500, "Unable to write candidate artifact."))?;
        file.sync_all()
            .await
            .map_err(|_| err(500, "Unable to sync candidate artifact."))?;
        drop(file);
        for suffix in 1..10000 {
            let name = if suffix == 1 {
                format!("{stem}.json")
            } else {
                format!("{stem}-import{suffix}.json")
            };
            let dest = canonical.join(name);
            match fs::hard_link(&temp, &dest).await {
                Ok(()) => return Ok(dest),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => return Err(err(500, "Unable to publish candidate artifact.")),
            }
        }
        Err(err(500, "Candidate import filename space is exhausted."))
    }
    .await;
    let _ = fs::remove_file(&temp).await;
    let path = result?;
    Ok(json!({"saved":true,"path":path,"candidate":candidate}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn candidate_saves_are_private_and_never_replace_an_existing_file() {
        let root = std::env::temp_dir().join(format!("midas-candidate-save-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir(&root).await.unwrap();
        let candidate = json!({"provenance":{"sweep_id":"sweep","run_id":"run"},"execution_config":{"order_qty":1}});
        let first = save_candidate(&root, &candidate).await.unwrap();
        let second = save_candidate(&root, &candidate).await.unwrap();
        let first_path = PathBuf::from(first["path"].as_str().unwrap());
        let second_path = PathBuf::from(second["path"].as_str().unwrap());
        assert_ne!(first_path, second_path);
        assert_eq!(fs::read(&first_path).await.unwrap(), fs::read(&second_path).await.unwrap());
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&first_path).await.unwrap().permissions().mode() & 0o777, 0o600);
        }
        fs::remove_dir_all(root).await.unwrap();
    }
    #[test]
    fn csv_quotes_and_multiline() {
        let rows = parse_csv("a,b\n1,\"x,y\"\n2,\"line1\nline2\"\n").unwrap();
        assert_eq!(rows[0]["b"], "x,y");
        assert_eq!(rows[1]["b"], "line1\nline2");
    }
    #[test]
    fn heatmap_uses_plan_values_and_summary_outcomes() {
        let plan = json!({"spec":{"sweep_id":"s"},"children":[{"run_id":"r1","run_index":0,"parameter_values":{"a":1,"b":"x"}},{"run_id":"r2","run_index":1,"parameter_values":{"a":2,"b":"x"}}]});
        let summary =
            json!({"runs":[{"run_id":"r1","run_index":0,"status":"completed","net_pnl":3.0}]});
        let m = heatmap(&plan, &summary, None, None, None).unwrap();
        assert_eq!(m["cells"][0]["cells"][0]["candidates"][0]["value"], 3.0);
        assert_eq!(
            m["cells"][0]["cells"][1]["candidates"][0]["status"],
            "missing_result"
        );
    }
    #[test]
    fn csv_bad_shape_rejected() {
        assert!(parse_csv("a,b\n1\n").is_err());
    }
}

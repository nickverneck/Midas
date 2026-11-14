//! Bounded native replay sweep editor endpoint.
//!
//! This module mirrors the SvelteKit `POST /api/strategy/specs` contract. It
//! resolves catalog identifiers on the server, clones only an approved native
//! replay spec, and asks the native CLI to validate the draft before publish.
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    path::{Component, Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs,
    io::AsyncReadExt,
    process::Command,
    time::{Duration, timeout},
};

const MAX_BODY: usize = 32 * 1024;
const MAX_SPEC: u64 = 2_000_000;
const MAX_NAME: usize = 100;
const MAX_DIMS: usize = 8;
const MAX_VALUES: usize = 16;
const MAX_COMBINATIONS: usize = 32;
const MAX_OUTPUT: usize = 96 * 1024;
static NEXT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct EditorState {
    trader: PathBuf,
    root: PathBuf,
    created: PathBuf,
}
#[derive(Debug)]
struct EditorError(StatusCode, String);
impl IntoResponse for EditorError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":self.1}))).into_response()
    }
}
type Result<T> = std::result::Result<T, EditorError>;
fn bad(s: impl Into<String>) -> EditorError {
    EditorError(StatusCode::BAD_REQUEST, s.into())
}
fn internal() -> EditorError {
    EditorError(
        StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected local strategy API error.".into(),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(rename = "sourceSweepId")]
    source_sweep_id: String,
    name: String,
    parameters: Vec<Parameter>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Parameter {
    path: String,
    values: Vec<Value>,
}

/// Router mounted by the strategy API at `/api/strategy/specs`.
pub(super) fn router() -> Router {
    let shared = super::strategy::api_state();
    let trader = shared.trader_dir;
    let root = shared.specs_root;
    let state = EditorState {
        trader,
        created: root.join("webui-created"),
        root,
    };
    Router::new()
        .route("/api/strategy/specs", post(create))
        .with_state(state)
}

async fn create(
    State(s): State<EditorState>,
    body: std::result::Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Result<(StatusCode, Json<Value>)> {
    let body = body.map_err(|_| bad("Request body must be JSON."))?;
    if body.len() > MAX_BODY {
        return Err(EditorError(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Strategy request body is too large.".into(),
        ));
    }
    let req: Request =
        serde_json::from_slice(&body).map_err(|_| bad("Request body must be JSON."))?;
    // Share the strategy job reservation for the full create/validate/publish
    // transaction so a replay cannot start against a partially published spec.
    let _gate = super::strategy::reserve_job_gate().ok_or_else(|| {
        EditorError(
            StatusCode::CONFLICT,
            "A strategy job is already active.".into(),
        )
    })?;
    if req.name.trim().is_empty() || req.name.trim().chars().count() > MAX_NAME {
        return Err(bad(format!(
            "name must contain 1 to {MAX_NAME} characters."
        )));
    }
    if req.parameters.len() > MAX_DIMS {
        return Err(bad(format!(
            "parameters must contain at most {MAX_DIMS} dimensions."
        )));
    }
    let mut seen = std::collections::HashSet::new();
    let mut combos = 1usize;
    for p in &req.parameters {
        if p.path.len() > 128 || !valid_path(&p.path) {
            return Err(bad(
                "Parameter paths must be dot-separated native scalar field names.",
            ));
        }
        if !seen.insert(&p.path) {
            return Err(bad(format!("Duplicate parameter path: {}", p.path)));
        }
        if p.values.is_empty() || p.values.len() > MAX_VALUES {
            return Err(bad(format!(
                "Each values list must contain 1 to {MAX_VALUES} scalar values."
            )));
        }
        let mut distinct = std::collections::HashSet::new();
        for v in &p.values {
            let valid = match v {
                Value::String(x) => x.chars().count() <= 128,
                Value::Bool(_) => true,
                Value::Number(n) => n.as_f64().map(f64::is_finite).unwrap_or(false),
                _ => false,
            };
            if !valid {
                return Err(bad(
                    "Parameter values must be bounded string, boolean, or finite number scalars.",
                ));
            }
            let enc = serde_json::to_string(v).map_err(|_| internal())?;
            if !distinct.insert(enc) {
                return Err(bad(format!(
                    "Parameter {} contains a duplicate value.",
                    p.path
                )));
            }
        }
        combos = combos.saturating_mul(p.values.len());
        if combos > MAX_COMBINATIONS {
            return Err(bad(format!(
                "Sweep grid may contain at most {MAX_COMBINATIONS} combinations."
            )));
        }
    }
    let (source_path, mut spec) = load_source(&s, &req.source_sweep_id).await?;
    if spec
        .get("base_strategy")
        .and_then(Value::as_object)
        .is_none()
    {
        return Err(bad("Selected sweep has no supported base strategy."));
    }
    spec.get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("Selected sweep is not a supported native replay specification."))?;
    let source_sweep = spec
        .get("sweep_id")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("Selected sweep is not a supported native replay specification."))?
        .to_owned();
    let dataset = spec
        .get("base_dataset_view")
        .and_then(|v| v.get("id"))
        .and_then(Value::as_str)
        .ok_or_else(|| bad("Selected sweep is not a supported native replay specification."))?
        .to_owned();
    let id = new_id();
    let sweep_id = format!("webui_{}", id.replace('-', ""));
    let dir = s.created.join(&id);
    let specs = dir.join("specs");
    let outputs = dir.join("outputs");
    let draft = specs.join("replay-sweep.json.tmp");
    let final_path = specs.join("replay-sweep.json");
    // Create an exclusive child under the canonical managed root, so a
    // pre-existing symlink can never redirect writes outside it.
    fs::create_dir_all(&s.created)
        .await
        .map_err(|_| internal())?;
    let canonical_root = fs::canonicalize(&s.created).await.map_err(|_| internal())?;
    fs::create_dir(&dir).await.map_err(|_| internal())?;
    let canonical_dir = fs::canonicalize(&dir).await.map_err(|_| internal())?;
    if !canonical_dir.starts_with(&canonical_root) {
        let _ = fs::remove_dir_all(&dir).await;
        return Err(bad(
            "Created spec path is outside the managed strategy directory.",
        ));
    }
    fs::create_dir(&specs).await.map_err(|_| internal())?;
    fs::create_dir(&outputs).await.map_err(|_| internal())?;
    let cleanup_dir = dir.clone();
    let result=async {
        spec["name"]=json!(req.name.trim()); spec["sweep_id"]=json!(sweep_id); spec["parameters"]=serde_json::to_value(&req.parameters).map_err(|_|internal())?; spec["output_dir"]=json!(outputs);
        let bytes=serde_json::to_vec_pretty(&spec).map_err(|_|internal())?;
        fs::write(&draft,bytes).await.map_err(|_|internal())?;
        let (code,output)=validate(&s,&draft).await.map_err(|e|EditorError(StatusCode::BAD_REQUEST,e))?;
        if code!=0 { return Err(bad(format!("Native sweep validation failed: {}",if output.is_empty(){format!("exit {code}")}else{output.clone()}))); }
        if let Some(line)=output.lines().find(|l|l.starts_with("Guardrail:")) { return Err(bad(format!("Native sweep exceeds a guardrail: {line}"))); }
        if output.to_lowercase().contains("large-sweep confirmation threshold") { return Err(bad("Native sweep requires large-sweep confirmation and cannot be created through this bounded editor.")); }
        fs::rename(&draft,&final_path).await.map_err(|_|internal())?;
        let rel=final_path.strip_prefix(&s.root).map_err(|_|internal())?.to_string_lossy().replace('\\',"/");
        Ok((StatusCode::CREATED,Json(json!({"id":encode(rel.as_bytes()),"name":spec["name"],"sweepId":spec["sweep_id"],"datasetViewId":dataset,"sourceSweepId":source_sweep,"parameters":spec["parameters"]}))))
    }.await;
    if result.is_err() {
        let _ = fs::remove_dir_all(cleanup_dir).await;
    }
    let _ = fs::remove_file(draft).await;
    let _ = source_path;
    result
}

async fn load_source(s: &EditorState, id: &str) -> Result<(PathBuf, Value)> {
    if id.len() > 2048 {
        return Err(bad("sweepId is required."));
    }
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, id)
        .map_err(|_| bad("Unknown sweepId."))?;
    if encode(&bytes) != id {
        return Err(bad("Unknown sweepId."));
    }
    let rel = std::str::from_utf8(&bytes).map_err(|_| bad("Unknown sweepId."))?;
    let relative = Path::new(rel);
    if rel.contains('\0')
        || relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        || relative.extension().and_then(|x| x.to_str()) != Some("json")
        || !relative.components().any(|c| c.as_os_str() == "specs")
    {
        return Err(bad("Unknown sweepId."));
    }
    let root = fs::canonicalize(&s.root).await.map_err(|_| internal())?;
    let mut current = s.root.clone();
    for component in relative.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current).await.map_err(|_| {
            EditorError(StatusCode::NOT_FOUND, "Sweep spec no longer exists.".into())
        })?;
        if metadata.file_type().is_symlink() {
            return Err(bad("Unknown sweepId."));
        }
    }
    let path = s.root.join(relative);
    let canonical = fs::canonicalize(&path)
        .await
        .map_err(|_| EditorError(StatusCode::NOT_FOUND, "Sweep spec no longer exists.".into()))?;
    if !canonical.starts_with(&root) {
        return Err(bad("Unknown sweepId."));
    }
    let meta = fs::metadata(&canonical).await.map_err(|_| internal())?;
    if !meta.is_file() || meta.len() > MAX_SPEC {
        return Err(bad("Sweep spec is too large or invalid."));
    }
    let doc: Value = serde_json::from_slice(&fs::read(&canonical).await.map_err(|_| internal())?)
        .map_err(|_| bad("Sweep spec is not valid JSON."))?;
    if doc.get("sweep_id").and_then(Value::as_str).is_none()
        || doc.get("name").and_then(Value::as_str).is_none()
        || doc.get("base_dataset_view").is_none()
    {
        return Err(bad("Sweep spec is not a supported native replay spec."));
    }
    Ok((canonical, doc))
}
fn valid_path(s: &str) -> bool {
    let mut parts = s.split('.');
    parts.all(|p| {
        !p.is_empty()
            && p.as_bytes()[0].is_ascii_lowercase()
            && p.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    })
}
fn encode(b: &[u8]) -> String {
    base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, b)
}
fn new_id() -> String {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
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
async fn validate(s: &EditorState, path: &Path) -> std::result::Result<(i32, String), String> {
    let binary = std::env::var_os("MIDAS_TRADER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe().unwrap_or_else(|_| s.trader.join("target/debug/trader"))
        });
    let mut cmd = Command::new(binary);
    if let Some(c) = std::env::var_os("MIDAS_CONFIG") {
        cmd.arg("--config")
            .arg(super::standalone::absolute(PathBuf::from(c)));
    }
    cmd.args(["validate-replay-sweep", "--spec"])
        .arg(path)
        .current_dir(&s.trader)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Unable to start native sweep validation: {e}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Native validation stdout unavailable.".to_string())?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Native validation stderr unavailable.".to_string())?;
    let stdout_task = tokio::spawn(async move { drain_bounded(&mut stdout, MAX_OUTPUT / 2).await });
    let stderr_task = tokio::spawn(async move { drain_bounded(&mut stderr, MAX_OUTPUT / 2).await });
    let status = timeout(Duration::from_secs(30), child.wait())
        .await
        .map_err(|_| "Native sweep validation timed out.".to_string())?
        .map_err(|e| e.to_string())?;
    let mut bytes = stdout_task
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    bytes.extend(
        stderr_task
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?,
    );
    if bytes.len() > MAX_OUTPUT {
        bytes = bytes.split_off(bytes.len() - MAX_OUTPUT);
    }
    Ok((
        status.code().unwrap_or(1),
        String::from_utf8_lossy(&bytes).trim().to_owned(),
    ))
}

async fn drain_bounded<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
) -> std::io::Result<Vec<u8>> {
    let mut kept = Vec::with_capacity(limit);
    let mut chunk = [0u8; 8192];
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        let remaining = limit.saturating_sub(kept.len());
        kept.extend_from_slice(&chunk[..n.min(remaining)]);
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::valid_path;

    #[test]
    fn parameter_paths_match_native_scalar_path_shape() {
        assert!(valid_path("base_strategy.native_ema.fast_length"));
        assert!(!valid_path("../output_dir"));
        assert!(!valid_path("base..field"));
        assert!(!valid_path("Field"));
    }
}

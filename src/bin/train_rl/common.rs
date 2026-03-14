use anyhow::Result;
use midas_env::env::MarginMode;
use std::path::{Path, PathBuf};

use crate::args::Args;

pub fn resolve_paths(args: &Args) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let resolve_path = |path: &Path, fallback: &Path| -> PathBuf {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = path
                .read_dir()
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "parquet").unwrap_or(false))
                .collect();
            entries.sort();
            return entries
                .first()
                .cloned()
                .unwrap_or_else(|| fallback.to_path_buf());
        }
        if path.exists() {
            path.to_path_buf()
        } else {
            fallback.to_path_buf()
        }
    };

    if let Some(p) = &args.parquet {
        Ok((p.clone(), p.clone(), p.clone()))
    } else {
        let train = resolve_path(&args.train_parquet, &args.train_parquet);
        let val = resolve_path(&args.val_parquet, &train);
        let test = resolve_path(&args.test_parquet, &val);
        Ok((train, val, test))
    }
}

pub fn load_symbol_config(path: &Path, symbol: &str) -> Result<(Option<f64>, Option<String>)> {
    if !path.exists() {
        return Ok((None, None));
    }
    let text = std::fs::read_to_string(path)?;
    let cfg: serde_yaml::Value = serde_yaml::from_str(&text)?;
    if let Some(entry) = cfg.get(symbol) {
        let margin = entry.get("margin_per_contract").and_then(|v| v.as_f64());
        let session = entry
            .get("session")
            .and_then(|v| v.as_str())
            .map(|s| s.to_ascii_lowercase());
        Ok((margin, session))
    } else {
        Ok((None, None))
    }
}

pub fn infer_margin(symbol: &str) -> f64 {
    if is_futures_symbol(symbol) {
        let sym = symbol.to_ascii_uppercase();
        if sym.contains("MES") {
            return 50.0;
        }
        return 500.0;
    }
    100.0
}

pub fn infer_margin_mode(symbol: &str, margin_cfg: Option<f64>) -> MarginMode {
    if margin_cfg.is_some() || is_futures_symbol(symbol) {
        MarginMode::PerContract
    } else {
        MarginMode::Price
    }
}

fn is_futures_symbol(symbol: &str) -> bool {
    let sym = symbol.to_ascii_uppercase();
    sym.contains("MES") || sym == "ES" || sym.contains("ES@") || sym.ends_with("ES")
}

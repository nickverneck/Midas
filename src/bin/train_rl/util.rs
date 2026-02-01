use anyhow::Result;
use midas_env::env::MarginMode;
use std::env;
use std::path::{Path, PathBuf};

use crate::args::Args;

pub fn resolve_device(requested: Option<&str>) -> tch::Device {
    preload_cuda_dlls();
    use tch::Device;
    match requested.unwrap_or("") {
        "cuda" | "cuda:0" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
        "mps" => {
            let mps_device = Device::Mps;
            let mps_ok = std::panic::catch_unwind(|| {
                let _t = tch::Tensor::zeros(&[1], (tch::Kind::Float, mps_device));
            })
            .is_ok();
            if mps_ok {
                mps_device
            } else {
                Device::Cpu
            }
        }
        "cpu" => Device::Cpu,
        _ => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
    }
}

fn preload_cuda_dlls() {
    if !cfg!(target_os = "windows") {
        return;
    }
    if env::var("MIDAS_PRELOAD_TORCH").as_deref() != Ok("1") {
        return;
    }
    let Ok(libtorch) = env::var("LIBTORCH") else {
        return;
    };
    let lib_dir = Path::new(&libtorch).join("lib");
    for dll in ["c10_cuda.dll", "torch_cuda.dll"] {
        let path = lib_dir.join(dll);
        if !path.exists() {
            continue;
        }
        unsafe {
            match libloading::Library::new(&path) {
                Ok(lib) => {
                    std::mem::forget(lib);
                    println!("info: preloaded {}", dll);
                }
                Err(err) => {
                    println!("warn: failed to preload {}: {}", dll, err);
                }
            }
        }
    }
}

pub fn print_device(device: &tch::Device) {
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        let cuda_dll = std::path::Path::new(&libtorch)
            .join("lib")
            .join("torch_cuda.dll");
        println!("info: LIBTORCH={}", libtorch);
        println!(
            "info: torch_cuda.dll exists: {}",
            if cuda_dll.exists() { "yes" } else { "no" }
        );
        let lib_dir = std::path::Path::new(&libtorch).join("lib");
        if let Ok(entries) = std::fs::read_dir(&lib_dir) {
            let mut has_cudart = false;
            let mut has_cublas = false;
            let mut has_cublas_lt = false;
            let mut has_cudnn = false;
            let mut has_nvrtc = false;
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("cudart64_") && name.ends_with(".dll") {
                        has_cudart = true;
                    }
                    if name.starts_with("cublas64_") && name.ends_with(".dll") {
                        has_cublas = true;
                    }
                    if name.starts_with("cublasLt64_") && name.ends_with(".dll") {
                        has_cublas_lt = true;
                    }
                    if name.starts_with("cudnn64_") && name.ends_with(".dll") {
                        has_cudnn = true;
                    }
                    if name.starts_with("nvrtc64_") && name.ends_with(".dll") {
                        has_nvrtc = true;
                    }
                }
            }
            println!(
                "info: libtorch cuda runtime: cudart={}, cublas={}",
                if has_cudart { "yes" } else { "no" },
                if has_cublas { "yes" } else { "no" }
            );
            println!(
                "info: libtorch cuda extras: cublasLt={}, cudnn={}, nvrtc={}",
                if has_cublas_lt { "yes" } else { "no" },
                if has_cudnn { "yes" } else { "no" },
                if has_nvrtc { "yes" } else { "no" }
            );
            if cfg!(target_os = "windows") {
                let system_root = std::env::var("SystemRoot")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("C:\\Windows"));
                let nvcuda = system_root.join("System32").join("nvcuda.dll");
                println!(
                    "info: nvcuda.dll in System32: {}",
                    if nvcuda.exists() { "yes" } else { "no" }
                );
            }
        }
    }
    if let tch::Device::Cuda(idx) = device {
        println!("info: using cuda:{}", idx);
    } else if let tch::Device::Mps = device {
        println!("info: using mps");
    } else {
        println!(
            "info: using {:?} (CUDA available={})",
            device,
            tch::Cuda::is_available()
        );
    }
}

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

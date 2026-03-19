use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct RunningEngine {
    pub id: u32,
    pub cwd: PathBuf,
    pub socket_path: PathBuf,
    pub socket_is_live: bool,
}

pub fn list_running_engines() -> Result<Vec<RunningEngine>> {
    imp::list_running_engines()
}

pub fn resolve_engine(id: u32) -> Result<RunningEngine> {
    imp::resolve_engine(id)
}

#[cfg(target_os = "linux")]
mod imp {
    use super::RunningEngine;
    use anyhow::{Context, Result, bail};
    use std::ffi::OsStr;
    use std::fs;
    use std::os::unix::fs::FileTypeExt;
    use std::path::{Path, PathBuf};

    const DEFAULT_ENGINE_SOCKET: &str = ".run/ninjatrader-engine.sock";

    pub fn list_running_engines() -> Result<Vec<RunningEngine>> {
        let mut engines = Vec::new();
        for entry in fs::read_dir("/proc").context("read /proc")? {
            let Ok(entry) = entry else {
                continue;
            };
            let file_name = entry.file_name();
            let Some(pid) = file_name.to_string_lossy().parse::<u32>().ok() else {
                continue;
            };
            if let Ok(engine) = resolve_engine(pid) {
                engines.push(engine);
            }
        }
        engines.sort_by_key(|engine| engine.id);
        Ok(engines)
    }

    pub fn resolve_engine(id: u32) -> Result<RunningEngine> {
        let proc_root = PathBuf::from(format!("/proc/{id}"));
        let cmdline = read_cmdline(proc_root.join("cmdline"))?;
        let cwd = fs::read_link(proc_root.join("cwd"))
            .with_context(|| format!("read cwd for engine {id}"))?;
        build_engine(id, &cmdline, &cwd)
    }

    fn build_engine(id: u32, cmdline: &[String], cwd: &Path) -> Result<RunningEngine> {
        if !is_ninjatrader_engine(cmdline) {
            bail!("process {id} is not a running ninjatrader-tui engine");
        }

        let socket = resolve_socket_path(cmdline, cwd);
        Ok(RunningEngine {
            id,
            cwd: cwd.to_path_buf(),
            socket_is_live: is_live_socket(&socket),
            socket_path: socket,
        })
    }

    fn read_cmdline(path: PathBuf) -> Result<Vec<String>> {
        let raw = fs::read(path).context("read process cmdline")?;
        Ok(raw
            .split(|byte| *byte == 0)
            .filter(|segment| !segment.is_empty())
            .map(|segment| String::from_utf8_lossy(segment).into_owned())
            .collect())
    }

    fn is_ninjatrader_engine(cmdline: &[String]) -> bool {
        let Some(executable) = cmdline.first() else {
            return false;
        };
        let Some(name) = Path::new(executable).file_name().and_then(OsStr::to_str) else {
            return false;
        };
        name == "ninjatrader-tui" && matches!(cmdline.last(), Some(arg) if arg == "engine")
    }

    fn resolve_socket_path(cmdline: &[String], cwd: &Path) -> PathBuf {
        let mut args = cmdline.iter();
        while let Some(arg) = args.next() {
            if arg == "--engine-socket" {
                if let Some(value) = args.next() {
                    return absolutize_socket(value, cwd);
                }
            }
            if let Some(value) = arg.strip_prefix("--engine-socket=") {
                return absolutize_socket(value, cwd);
            }
        }
        cwd.join(DEFAULT_ENGINE_SOCKET)
    }

    fn absolutize_socket(raw: &str, cwd: &Path) -> PathBuf {
        let path = PathBuf::from(raw);
        if path.is_absolute() {
            path
        } else {
            cwd.join(path)
        }
    }

    fn is_live_socket(path: &Path) -> bool {
        fs::metadata(path)
            .map(|meta| meta.file_type().is_socket())
            .unwrap_or(false)
    }

    #[cfg(test)]
    mod tests {
        use super::{DEFAULT_ENGINE_SOCKET, is_ninjatrader_engine, resolve_socket_path};
        use std::path::Path;

        #[test]
        fn detects_engine_process() {
            let args = vec![
                "/root/.cargo/bin/ninjatrader-tui".to_string(),
                "--engine-socket".to_string(),
                "/tmp/ninjatrader-tui".to_string(),
                "engine".to_string(),
            ];
            assert!(is_ninjatrader_engine(&args));
        }

        #[test]
        fn rejects_non_engine_process() {
            let args = vec![
                "/root/.cargo/bin/ninjatrader-tui".to_string(),
                "--config".to_string(),
                "config.toml".to_string(),
            ];
            assert!(!is_ninjatrader_engine(&args));
        }

        #[test]
        fn resolves_relative_socket_from_cwd() {
            let args = vec![
                "/root/.cargo/bin/ninjatrader-tui".to_string(),
                "--engine-socket".to_string(),
                ".run/ninjatrader-engine.sock".to_string(),
                "engine".to_string(),
            ];
            let resolved = resolve_socket_path(&args, Path::new("/srv/midas"));
            assert_eq!(
                resolved,
                Path::new("/srv/midas").join(DEFAULT_ENGINE_SOCKET)
            );
        }

        #[test]
        fn resolves_equals_socket_syntax() {
            let args = vec![
                "/root/.cargo/bin/ninjatrader-tui".to_string(),
                "--engine-socket=/tmp/custom.sock".to_string(),
                "engine".to_string(),
            ];
            let resolved = resolve_socket_path(&args, Path::new("/srv/midas"));
            assert_eq!(resolved, Path::new("/tmp/custom.sock"));
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::RunningEngine;
    use anyhow::{Result, bail};

    pub fn list_running_engines() -> Result<Vec<RunningEngine>> {
        bail!("`ninjatrader-tui list` is currently only supported on Linux");
    }

    pub fn resolve_engine(_id: u32) -> Result<RunningEngine> {
        bail!("`ninjatrader-tui attach` is currently only supported on Linux");
    }
}

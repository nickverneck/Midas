use anyhow::{Result, bail};
use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

const DEFAULT_ENGINE_SOCKET: &str = ".run/ninjatrader-engine.sock";

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

fn build_engine(
    id: u32,
    cmdline: &[String],
    cwd: &Path,
    socket_path: Option<PathBuf>,
) -> Result<RunningEngine> {
    if !is_ninjatrader_engine(cmdline) {
        bail!("process {id} is not a running ninjatrader-tui engine");
    }

    let socket_path = socket_path.unwrap_or_else(|| resolve_socket_path(cmdline, cwd));
    Ok(RunningEngine {
        id,
        cwd: cwd.to_path_buf(),
        socket_is_live: is_live_socket(&socket_path),
        socket_path,
    })
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

#[cfg(target_os = "linux")]
mod imp {
    use super::{RunningEngine, build_engine};
    use anyhow::{Context, Result};
    use std::fs;
    use std::path::PathBuf;

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
        build_engine(id, &cmdline, &cwd, None)
    }

    fn read_cmdline(path: PathBuf) -> Result<Vec<String>> {
        let raw = fs::read(path).context("read process cmdline")?;
        Ok(raw
            .split(|byte| *byte == 0)
            .filter(|segment| !segment.is_empty())
            .map(|segment| String::from_utf8_lossy(segment).into_owned())
            .collect())
    }

    #[cfg(test)]
    mod tests {
        use super::super::{DEFAULT_ENGINE_SOCKET, is_ninjatrader_engine, resolve_socket_path};
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

#[cfg(target_os = "macos")]
mod imp {
    use super::{
        DEFAULT_ENGINE_SOCKET, RunningEngine, build_engine, is_live_socket, resolve_socket_path,
    };
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn list_running_engines() -> Result<Vec<RunningEngine>> {
        let mut engines = Vec::new();
        for pid in list_candidate_pids()? {
            if let Ok(engine) = resolve_engine(pid) {
                engines.push(engine);
            }
        }
        engines.sort_by_key(|engine| engine.id);
        Ok(engines)
    }

    pub fn resolve_engine(id: u32) -> Result<RunningEngine> {
        let cmdline = read_cmdline(id)?;
        let cwd = read_cwd(id).unwrap_or_else(|| PathBuf::from("."));
        let socket_path = read_socket_path(id).or_else(|| {
            if cwd != Path::new(".") || cmdline.iter().any(|arg| arg.starts_with("--engine-socket"))
            {
                Some(resolve_socket_path(&cmdline, &cwd))
            } else {
                None
            }
        });
        build_engine(id, &cmdline, &cwd, socket_path)
    }

    fn list_candidate_pids() -> Result<Vec<u32>> {
        let output = Command::new("ps")
            .args(["-axo", "pid=,comm="])
            .output()
            .context("run ps to enumerate processes")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout
            .lines()
            .filter_map(parse_pid_comm_line)
            .filter(|(_, comm)| {
                Path::new(comm)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == "ninjatrader-tui")
            })
            .map(|(pid, _)| pid)
            .collect())
    }

    fn read_cmdline(id: u32) -> Result<Vec<String>> {
        let args_output = Command::new("ps")
            .args(["-p", &id.to_string(), "-ww", "-o", "args="])
            .output()
            .with_context(|| format!("run ps args for engine {id}"))?;
        let args = String::from_utf8_lossy(&args_output.stdout)
            .trim()
            .to_string();
        if args.is_empty() {
            return Ok(Vec::new());
        }
        Ok(args.split_whitespace().map(ToString::to_string).collect())
    }

    fn read_cwd(id: u32) -> Option<PathBuf> {
        read_named_lsof_path(id, &["-a", "-d", "cwd", "-Fn"])
    }

    fn read_socket_path(id: u32) -> Option<PathBuf> {
        let default_name = Path::new(DEFAULT_ENGINE_SOCKET)
            .file_name()
            .and_then(|name| name.to_str());
        let paths = read_lsof_paths(id, &["-a", "-U", "-Fn"]);
        paths
            .iter()
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .zip(default_name)
                    .is_some_and(|(actual, expected)| actual == expected)
                    && is_live_socket(path)
            })
            .cloned()
            .or_else(|| {
                paths.into_iter().find(|path| {
                    path.extension().and_then(|ext| ext.to_str()) == Some("sock")
                        || is_live_socket(path)
                })
            })
    }

    fn read_named_lsof_path(id: u32, prefix_args: &[&str]) -> Option<PathBuf> {
        read_lsof_paths(id, prefix_args).into_iter().next()
    }

    fn read_lsof_paths(id: u32, prefix_args: &[&str]) -> Vec<PathBuf> {
        let mut args = prefix_args.to_vec();
        args.push("-p");
        let pid = id.to_string();
        args.push(&pid);
        let output = match Command::new("lsof").args(&args).output() {
            Ok(output) => output,
            Err(_) => return Vec::new(),
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .filter_map(|line| line.strip_prefix('n'))
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .collect()
    }

    fn parse_pid_comm_line(line: &str) -> Option<(u32, String)> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        let (pid, comm) = trimmed.split_once(char::is_whitespace)?;
        Some((pid.parse().ok()?, comm.trim().to_string()))
    }

    #[cfg(test)]
    mod tests {
        use super::parse_pid_comm_line;

        #[test]
        fn parses_ps_pid_comm_output() {
            let parsed = parse_pid_comm_line(" 123 /Users/nick/.cargo/bin/ninjatrader-tui")
                .expect("ps line should parse");
            assert_eq!(parsed.0, 123);
            assert_eq!(parsed.1, "/Users/nick/.cargo/bin/ninjatrader-tui");
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod imp {
    use super::RunningEngine;
    use anyhow::{Result, bail};

    pub fn list_running_engines() -> Result<Vec<RunningEngine>> {
        bail!("`ninjatrader-tui list` is currently only supported on Linux and macOS");
    }

    pub fn resolve_engine(_id: u32) -> Result<RunningEngine> {
        bail!("`ninjatrader-tui attach` is currently only supported on Linux and macOS");
    }
}

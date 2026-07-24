use super::super::*;
use std::path::{Path, PathBuf};

impl App {
    pub(in crate::app) fn replay_dataset_library_lines(&self) -> Vec<Line<'static>> {
        let configured_path = &self.base_config.replay_file_path;
        let resolved = replay_dataset_file_metadata(configured_path);
        let mut lines = self.replay_cache_library_lines();
        lines.extend([
            Line::from(""),
            Line::from("Configured Local Replay File"),
            Line::from(format!(
                "Configured file: {}",
                display_path_name(configured_path)
            )),
            Line::from(format!("Configured path: {}", configured_path.display())),
        ]);

        match resolved {
            Some((path, bytes)) => {
                lines.extend([
                    Line::from("Status: ready"),
                    Line::from(format!("Resolved file: {}", display_path_name(&path))),
                    Line::from(format!("Resolved path: {}", path.display())),
                    Line::from(format!(
                        "Inferred contract: {}",
                        infer_replay_contract(&path)
                    )),
                    Line::from(format!("Size: {}", format_bytes(bytes))),
                    Line::from("Raw data: price-only ticks from local Last text file"),
                    Line::from("Available bars: seconds, minutes, tick-count, range"),
                    Line::from("Unavailable: volume needs per-trade size"),
                ]);
            }
            None => {
                lines.extend([
                    Line::from("Status: missing local file"),
                    Line::from("No replay dataset is available for the configured path."),
                    Line::from(
                        "Set replay_file_path or TRADER_REPLAY_FILE_PATH to a local tick file.",
                    ),
                    Line::from("Available after load: seconds, minutes, tick-count, range"),
                    Line::from("Unavailable: volume needs per-trade size"),
                ]);
            }
        }

        lines
    }

    pub(in crate::app) fn replay_market_control_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            styled_line(
                format!("Bar Type: {}", self.bar_type.kind().label()),
                self.focus == Focus::BarTypeToggle,
            ),
            styled_line(
                format!("{}: {}", self.replay_value_label(), self.bar_value_text()),
                self.focus == Focus::BarValue,
            ),
        ];

        if self.candle_mode_controls_visible() {
            lines.push(styled_line(
                format!("Candles: {}", self.candle_mode.label()),
                self.focus == Focus::CandleModeToggle,
            ));
        }

        if !self.replay_selected_bar_supported() {
            lines.push(Line::from(
                "Volume needs per-trade size; this Last file only has price.",
            ));
        } else if self.replay_cache_can_serve_selected_bar() {
            lines.push(Line::from(
                "Cached server bars provide this exact bar selection.",
            ));
        }
        if self.bar_type.kind() == BarKind::Tick {
            lines.push(Line::from(
                "Replay tick-count is local; live Tick Count needs Tradovate validation with a valid token.",
            ));
        }

        lines.extend([
            Line::from("Type digits to edit value; Left/Right changes bar type/candles."),
            Line::from(format!(
                "Effective chart: {}",
                self.bar_type.mode_label(self.effective_candle_mode())
            )),
            Line::from(format!(
                "Replay interval: {}ms between derived bars at 1x",
                self.base_config.replay_bar_interval_ms
            )),
        ]);
        lines
    }

    pub(in crate::app) fn replay_run_control_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        lines.push(styled_line(
            self.replay_start_action_label(),
            self.focus == Focus::ReplayMode,
        ));
        lines.push(Line::from(format!(
            "Replay source: {}",
            if self.replay_cache_can_serve_selected_bar() {
                "matching cached server bars"
            } else if self.local_replay_dataset_available() {
                "ready"
            } else {
                "missing"
            }
        )));
        lines.push(Line::from(format!(
            "Bar selection: {}",
            if self.replay_selected_bar_supported() {
                "supported"
            } else {
                "unsupported for local file"
            }
        )));
        lines.push(Line::from(
            "Enter uses the newest matching cached server-bar JSONL dataset, otherwise the local text file.",
        ));
        lines.push(Line::from(
            "Replay skips broker login and does not start live streams.",
        ));
        lines.push(Line::from(
            "Downloader: use `trader download-replay-data` to add server-bar caches.",
        ));
        lines
    }

    pub(in crate::app) fn replay_dataset_available(&self) -> bool {
        self.replay_cache_can_serve_selected_bar() || self.local_replay_dataset_available()
    }

    pub(in crate::app) fn replay_selected_bar_supported(&self) -> bool {
        self.replay_cache_can_serve_selected_bar() || self.bar_type.kind() != BarKind::Volume
    }

    fn local_replay_dataset_available(&self) -> bool {
        replay_dataset_file_metadata(&self.base_config.replay_file_path).is_some()
    }

    pub(in crate::app) fn replay_cache_can_serve_selected_bar(&self) -> bool {
        #[cfg(feature = "replay")]
        {
            self.replay_cache_library
                .first_server_bars_jsonl(self.bar_type, self.effective_candle_mode(), None)
                .is_some()
        }

        #[cfg(not(feature = "replay"))]
        {
            false
        }
    }

    fn replay_value_label(&self) -> &'static str {
        match self.bar_type.kind() {
            BarKind::Minute => "Minutes per bar",
            BarKind::Second => "Seconds per bar",
            BarKind::Tick => "Ticks per bar",
            BarKind::Volume => "Volume per bar",
            BarKind::Range => "Range ticks",
        }
    }

    fn replay_start_action_label(&self) -> String {
        if !self.replay_dataset_available() {
            "[Enter] Start Replay (missing dataset)".to_string()
        } else if !self.replay_selected_bar_supported() {
            "[Enter] Start Replay (volume unavailable)".to_string()
        } else if self.replay_cache_can_serve_selected_bar() {
            "[Enter] Start Cached Replay".to_string()
        } else {
            "[Enter] Start Local Replay".to_string()
        }
    }
}

impl App {
    #[cfg(feature = "replay")]
    fn replay_cache_library_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from("Owned Cached Datasets"),
            Line::from(format!(
                "Cache root: {}",
                self.base_config.replay_cache_dir.display()
            )),
        ];

        if self.replay_cache_library.datasets.is_empty() {
            lines.extend([
                Line::from("Status: no manifest.json datasets found"),
                Line::from(
                    "Download planner: trader download-replay-data --instrument MES --contract MESU6 --start YYYY-MM-DD --end YYYY-MM-DD",
                ),
                Line::from(
                    "Downloader uses market-data history only; it does not start user sync or order streams.",
                ),
            ]);
            if !self.replay_cache_library.warnings.is_empty() {
                lines.push(Line::from(format!(
                    "Manifest warnings: {}",
                    self.replay_cache_library.warnings.len()
                )));
            }
            return lines;
        }

        let selected_hit = self.replay_cache_library.first_server_bars_jsonl(
            self.bar_type,
            self.effective_candle_mode(),
            None,
        );
        lines.push(Line::from(format!(
            "Status: {} manifest(s), selected request {}",
            self.replay_cache_library.datasets.len(),
            if selected_hit.is_some() {
                "JSONL server-bar match"
            } else {
                "no JSONL server-bar match"
            }
        )));
        lines.push(Line::from(
            "Enter can start the newest matching JSONL server-bar cache; full dataset selection comes later.",
        ));

        for dataset in self.replay_cache_library.datasets.iter().take(5) {
            let manifest = &dataset.manifest;
            lines.extend([
                Line::from(format!("Dataset: {}", manifest.display_name)),
                Line::from(format!(
                    "  {} {} {} | {}",
                    manifest.provider.label(),
                    manifest.env.label(),
                    manifest.contract.symbol,
                    manifest.coverage.label()
                )),
                Line::from(format!(
                    "  Badges: {} | Rows: {}",
                    manifest.badges_label(),
                    manifest.row_count_total()
                )),
                Line::from(format!(
                    "  Shapes: {} | Modes: {}",
                    manifest.available_shapes_label(),
                    manifest.available_chart_modes_label()
                )),
                Line::from(format!("  Manifest: {}", dataset.manifest_path.display())),
            ]);
        }
        if self.replay_cache_library.datasets.len() > 5 {
            lines.push(Line::from(format!(
                "... {} more cached dataset(s)",
                self.replay_cache_library.datasets.len() - 5
            )));
        }
        for warning in self.replay_cache_library.warnings.iter().take(2) {
            lines.push(Line::from(format!("Warning: {warning}")));
        }
        lines
    }

    #[cfg(not(feature = "replay"))]
    fn replay_cache_library_lines(&self) -> Vec<Line<'static>> {
        Vec::new()
    }
}

fn replay_dataset_file_metadata(path: &Path) -> Option<(PathBuf, u64)> {
    replay_dataset_candidates(path)
        .into_iter()
        .find_map(|candidate| {
            let metadata = std::fs::metadata(&candidate).ok()?;
            metadata.is_file().then_some((candidate, metadata.len()))
        })
}

fn replay_dataset_candidates(path: &Path) -> Vec<PathBuf> {
    if path.is_absolute() {
        return vec![path.to_path_buf()];
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        push_unique_path(&mut candidates, cwd.join(path));
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    push_unique_path(&mut candidates, manifest_dir.join(path));
    if let Some(workspace_root) = manifest_dir.parent() {
        push_unique_path(&mut candidates, workspace_root.join(path));
    }
    candidates
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn infer_replay_contract(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("Replay")
        .trim_end_matches(".Last")
        .to_string()
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GB {
        format!("{:.1} GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes / KB)
    } else {
        format!("{bytes:.0} B")
    }
}

fn display_path_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| path.to_str().unwrap_or("-"))
        .to_string()
}

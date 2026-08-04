use super::super::*;
use crate::broker::{ReplayEngineMode, ReplayFillModel};
use std::path::{Path, PathBuf};

impl App {
    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_filtered_dataset_indices(&self) -> Vec<usize> {
        let query = self.replay_instrument_query.trim().to_ascii_lowercase();
        self.replay_cache_library
            .datasets
            .iter()
            .enumerate()
            .filter_map(|(index, dataset)| {
                if query.is_empty() {
                    return Some(index);
                }
                let manifest = &dataset.manifest;
                let instrument = manifest.instrument.symbol.to_ascii_lowercase();
                let contract = manifest.contract.symbol.to_ascii_lowercase();
                let name = manifest.display_name.to_ascii_lowercase();
                let instrument_name = manifest
                    .instrument
                    .name
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                (instrument.contains(&query)
                    || contract.contains(&query)
                    || name.contains(&query)
                    || instrument_name.contains(&query))
                .then_some(index)
            })
            .collect()
    }

    /// Return one picker option per exact cached server-bar shape.  A cache
    /// manifest is intentionally shared by several shapes, but the user must
    /// still be able to select `500 Vol` independently from `1000 Vol`.
    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_dataset_options(&self) -> Vec<(usize, Option<BarType>)> {
        let query = self.replay_instrument_query.trim().to_ascii_lowercase();
        let mut options = Vec::new();
        for (index, dataset) in self.replay_cache_library.datasets.iter().enumerate() {
            let manifest = &dataset.manifest;
            let instrument = manifest.instrument.symbol.to_ascii_lowercase();
            let contract = manifest.contract.symbol.to_ascii_lowercase();
            let name = manifest.display_name.to_ascii_lowercase();
            let instrument_name = manifest
                .instrument
                .name
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !query.is_empty()
                && !(instrument.contains(&query)
                    || contract.contains(&query)
                    || name.contains(&query)
                    || instrument_name.contains(&query))
            {
                continue;
            }

            if manifest.available_bar_shapes.is_empty() {
                // Raw-tick-only datasets have no fixed provider bar shape;
                // retain one option so they remain selectable.
                options.push((index, None));
            } else {
                options.extend(
                    manifest
                        .available_bar_shapes
                        .iter()
                        .copied()
                        .map(|bar_type| (index, Some(bar_type))),
                );
            }
        }
        options
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn select_replay_dataset_option(
        &mut self,
        dataset_index: usize,
        bar_type: Option<BarType>,
    ) {
        self.replay_dataset_index = Some(dataset_index);
        self.replay_dataset_bar_type = bar_type;
        self.replay_dataset_view_path = None;
        if let Some(bar_type) = bar_type {
            self.bar_type = bar_type;
            self.candle_mode = self.bar_type.effective_candle_mode(self.candle_mode);
        }
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_selected_option_is(
        &self,
        dataset_index: usize,
        bar_type: Option<BarType>,
    ) -> bool {
        self.replay_dataset_index == Some(dataset_index) && self.replay_dataset_bar_type == bar_type
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_selected_option_position(
        &self,
        options: &[(usize, Option<BarType>)],
    ) -> Option<usize> {
        options
            .iter()
            .position(|(index, bar_type)| self.replay_selected_option_is(*index, *bar_type))
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_selected_choice_label(&self) -> String {
        let Some(dataset) = self.replay_selected_dataset() else {
            return "none".to_string();
        };
        self.replay_dataset_bar_type
            .map(|bar_type| {
                format!(
                    "{} / {}",
                    dataset.manifest.contract.symbol,
                    bar_type.label()
                )
            })
            .unwrap_or_else(|| dataset.manifest.display_name.clone())
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn sync_replay_dataset_filter(&mut self) {
        let options = self.replay_dataset_options();
        if let Some((index, bar_type)) = options
            .iter()
            .copied()
            .find(|(index, bar_type)| self.replay_selected_option_is(*index, *bar_type))
        {
            self.select_replay_dataset_option(index, bar_type);
        } else if let Some((index, bar_type)) = options.first().copied() {
            self.select_replay_dataset_option(index, bar_type);
        } else {
            self.replay_dataset_index = None;
            self.replay_dataset_bar_type = None;
            self.replay_dataset_view_path = None;
        }
    }

    #[cfg(feature = "replay")]
    fn replay_dataset_row_line(
        &self,
        position: usize,
        index: usize,
        selected_bar_type: Option<BarType>,
    ) -> Line<'static> {
        let dataset = &self.replay_cache_library.datasets[index];
        let manifest = &dataset.manifest;
        let selected = self.replay_selected_option_is(index, selected_bar_type);
        let marker = if selected { ">" } else { " " };
        let coverage = if manifest.coverage.start.date_naive() == manifest.coverage.end.date_naive()
        {
            manifest.coverage.start.format("%Y-%m-%d").to_string()
        } else {
            format!(
                "{}..{}",
                manifest.coverage.start.format("%m-%d"),
                manifest.coverage.end.format("%m-%d")
            )
        };
        let row = match (manifest.available_bar_shapes.len() > 1, selected_bar_type) {
            (true, Some(bar_type)) => format!(
                "{marker} [{position}] {} | {} | {} | {coverage} | {} rows",
                manifest.contract.symbol,
                bar_type.label(),
                manifest.source_kind.label(),
                manifest
                    .files
                    .iter()
                    .find(|file| file.market_shape.bar_type == Some(bar_type))
                    .map(|file| file.row_count)
                    .unwrap_or_else(|| manifest.preferred_row_count_total()),
            ),
            _ => format!(
                "{marker} [{position}] {} | {} | {coverage} | {} rows",
                manifest.contract.symbol,
                manifest.source_kind.label(),
                manifest.preferred_row_count_total(),
            ),
        };
        if selected && self.focus == Focus::ReplayDataset {
            Line::from(Span::styled(
                row,
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if selected {
            Line::from(Span::styled(
                row,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))
        } else {
            Line::from(Span::styled(row, Style::default().fg(Color::White)))
        }
    }

    #[cfg(feature = "replay")]
    fn replay_dataset_compact_row_line(
        &self,
        position: usize,
        index: usize,
        selected_bar_type: Option<BarType>,
    ) -> Line<'static> {
        let dataset = &self.replay_cache_library.datasets[index];
        let manifest = &dataset.manifest;
        let selected = self.replay_selected_option_is(index, selected_bar_type);
        let marker = if selected { ">" } else { " " };
        let source = match manifest.source_kind {
            crate::replay_cache::ReplayCacheSourceKind::ServerBars => "bars",
            crate::replay_cache::ReplayCacheSourceKind::RawTicks => "raw",
            crate::replay_cache::ReplayCacheSourceKind::Mixed => "mixed",
            crate::replay_cache::ReplayCacheSourceKind::DerivedBars => "derived",
            crate::replay_cache::ReplayCacheSourceKind::DomStream => "L2",
            crate::replay_cache::ReplayCacheSourceKind::LocalText => "text",
        };
        let coverage = manifest.coverage.start.format("%m-%d");
        let row = match (manifest.available_bar_shapes.len() > 1, selected_bar_type) {
            (true, Some(bar_type)) => format!(
                "{marker} [{position}] {} | {} | {source} | {coverage} | {}",
                manifest.contract.symbol,
                bar_type.label(),
                manifest
                    .files
                    .iter()
                    .find(|file| file.market_shape.bar_type == Some(bar_type))
                    .map(|file| file.row_count)
                    .unwrap_or_else(|| manifest.preferred_row_count_total()),
            ),
            _ => format!(
                "{marker} [{position}] {} | {source} | {coverage} | {}",
                manifest.contract.symbol,
                manifest.preferred_row_count_total(),
            ),
        };
        if selected && self.focus == Focus::ReplayDataset {
            Line::from(Span::styled(
                row,
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if selected {
            Line::from(Span::styled(
                row,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))
        } else {
            Line::from(Span::styled(row, Style::default().fg(Color::White)))
        }
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_dataset_metadata_lines(&self) -> Vec<Line<'static>> {
        let Some(dataset) = self.replay_selected_dataset() else {
            let configured_path = &self.base_config.replay_file_path;
            return vec![
                Line::from("No cached contract selected."),
                Line::from("Select a cached row to inspect its metadata."),
                Line::from(""),
                replay_metadata_line(
                    "Local fallback",
                    if self.local_replay_dataset_available() {
                        "ready"
                    } else {
                        "missing"
                    },
                ),
                replay_metadata_line("File", display_path_name(configured_path)),
                Line::from("Enter on an empty search opens contract download."),
            ];
        };

        let manifest = &dataset.manifest;
        let raw_ticks =
            manifest.has_source_kind(crate::replay_cache::ReplayCacheSourceKind::RawTicks);
        let shapes = if raw_ticks && manifest.available_bar_shapes.is_empty() {
            "derived from raw ticks (time, tick-count, range)".to_string()
        } else {
            manifest.available_shapes_label()
        };
        let modes = if raw_ticks && manifest.available_chart_modes.is_empty() {
            "derived OHLC / Heikin Ashi".to_string()
        } else {
            manifest.available_chart_modes_label()
        };
        let mut lines = vec![
            replay_metadata_line("Instrument", manifest.instrument.symbol.clone()),
            replay_metadata_line("Contract", manifest.contract.symbol.clone()),
            replay_metadata_line(
                "Description",
                manifest
                    .instrument
                    .name
                    .clone()
                    .unwrap_or_else(|| manifest.display_name.clone()),
            ),
            replay_metadata_line("Dataset", manifest.display_name.clone()),
            replay_metadata_line(
                "Selected shape",
                self.replay_dataset_bar_type
                    .map(|bar_type| bar_type.label())
                    .unwrap_or_else(|| "raw ticks".to_string()),
            ),
            replay_metadata_line(
                "Provider",
                format!("{} / {}", manifest.provider.label(), manifest.env.label()),
            ),
            replay_metadata_line("Coverage", manifest.coverage.label()),
            replay_metadata_line("Source", manifest.source_kind.label()),
            replay_metadata_line("Rows", manifest.preferred_row_count_total().to_string()),
            replay_metadata_line("Granular", if raw_ticks { "yes" } else { "no" }),
            replay_metadata_line("Shapes", shapes),
            replay_metadata_line("Modes", modes),
            replay_metadata_line("Badges", manifest.badges_label()),
            replay_metadata_line("Manifest", dataset.manifest_path.display().to_string()),
        ];
        if !manifest.tags.is_empty() {
            lines.push(replay_metadata_line("Tags", manifest.tags.join(", ")));
        }
        if let Some(notes) = manifest.notes.as_deref() {
            lines.push(replay_metadata_line("Notes", notes.to_string()));
        }
        lines.push(Line::from(""));
        lines.push(Line::from("Enter/Right: replay setup"));
        lines.push(Line::from("Select row, then D: extend | V: saved views"));
        lines
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_dataset_metadata_compact_lines(&self) -> Vec<Line<'static>> {
        let Some(dataset) = self.replay_selected_dataset() else {
            return self.replay_dataset_metadata_lines();
        };
        let manifest = &dataset.manifest;
        let raw_ticks =
            manifest.has_source_kind(crate::replay_cache::ReplayCacheSourceKind::RawTicks);
        vec![
            replay_metadata_line("Contract", manifest.contract.symbol.clone()),
            replay_metadata_line("Source", manifest.source_kind.label()),
            replay_metadata_line(
                "Selected shape",
                self.replay_dataset_bar_type
                    .map(|bar_type| bar_type.label())
                    .unwrap_or_else(|| "raw ticks".to_string()),
            ),
            replay_metadata_line("Coverage", manifest.coverage.label()),
            replay_metadata_line("Rows", manifest.preferred_row_count_total().to_string()),
            replay_metadata_line("Granular", if raw_ticks { "yes" } else { "no" }),
            replay_metadata_line(
                "Shapes",
                if raw_ticks && manifest.available_bar_shapes.is_empty() {
                    "derived from raw ticks (time, tick-count, range)".to_string()
                } else {
                    manifest.available_shapes_label()
                },
            ),
            Line::from("Enter/Right: replay setup"),
            Line::from("Select row, then D/V for dataset actions"),
        ]
    }

    #[cfg(not(feature = "replay"))]
    pub(in crate::app) fn replay_dataset_metadata_lines(&self) -> Vec<Line<'static>> {
        vec![Line::from("Replay cache support is disabled.")]
    }

    #[cfg(not(feature = "replay"))]
    pub(in crate::app) fn replay_dataset_metadata_compact_lines(&self) -> Vec<Line<'static>> {
        vec![Line::from("Replay cache support is disabled.")]
    }

    pub(in crate::app) fn replay_dataset_library_compact_lines(&self) -> Vec<Line<'static>> {
        #[cfg(feature = "replay")]
        {
            let mut lines = vec![Line::from("Cached Contracts")];
            lines.push(Line::from(format!(
                "Search: {}{}",
                if self.replay_instrument_query.is_empty() {
                    "<all>".to_string()
                } else {
                    self.replay_instrument_query.clone()
                },
                if self.focus == Focus::ReplayInstrumentQuery {
                    " *"
                } else {
                    ""
                }
            )));
            let indices = self.replay_filtered_dataset_indices();
            let options = self.replay_dataset_options();
            if options.is_empty() {
                lines.extend([
                    Line::from("No cached contract matches."),
                    Line::from("Enter: search/download | Tab: run setup"),
                ]);
                return lines;
            }

            lines.push(Line::from(format!(
                "{} cached | selected {}",
                indices.len(),
                self.replay_selected_choice_label()
            )));
            lines.push(Line::from("Up/Down select shape | Enter/Right setup"));

            const VISIBLE_DATASETS: usize = 8;
            let selected_position = self.replay_selected_option_position(&options).unwrap_or(0);
            let visible_start = selected_position
                .saturating_sub(VISIBLE_DATASETS - 1)
                .min(options.len().saturating_sub(VISIBLE_DATASETS));
            for (position, (index, bar_type)) in options
                .iter()
                .copied()
                .enumerate()
                .skip(visible_start)
                .take(VISIBLE_DATASETS)
            {
                lines.push(self.replay_dataset_compact_row_line(position + 1, index, bar_type));
            }
            lines.push(Line::from(if self.focus == Focus::ReplayDataset {
                "N download | D extend | V views"
            } else {
                "Select row: N/D/V actions"
            }));
            lines
        }

        #[cfg(not(feature = "replay"))]
        {
            vec![Line::from("Replay cache support is disabled.")]
        }
    }

    pub(in crate::app) fn replay_market_control_compact_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            replay_edit_line(
                "Bar Type",
                format!("{} (Left/Right)", self.bar_type.kind().label()),
                self.focus == Focus::BarTypeToggle,
            ),
            replay_edit_line(
                self.replay_value_label(),
                format!("{} (digits)", self.bar_value_text()),
                self.focus == Focus::BarValue,
            ),
        ];
        if self.candle_mode_controls_visible() {
            lines.push(replay_edit_line(
                "Candles",
                format!("{} (Left/Right)", self.candle_mode.label()),
                self.focus == Focus::CandleModeToggle,
            ));
        }
        lines.push(replay_metadata_line(
            "Selection",
            self.bar_type.mode_label(self.effective_candle_mode()),
        ));
        lines.push(replay_metadata_line(
            "Data",
            if self.replay_dataset_available() {
                "ready"
            } else {
                "missing"
            },
        ));
        lines
    }

    pub(in crate::app) fn replay_run_control_compact_lines(&self) -> Vec<Line<'static>> {
        vec![
            replay_action_line(
                self.replay_start_action_compact_label(),
                self.focus == Focus::ReplayMode,
            ),
            replay_edit_line(
                format!("Capital ({})", self.base_config.replay_account_currency,),
                self.replay_initial_capital_text(),
                self.replay_initial_capital_focused(),
            ),
            replay_edit_line(
                "Margin",
                self.replay_margin_per_contract_text(),
                self.replay_margin_per_contract_focused(),
            ),
            replay_edit_line(
                "Safety",
                self.replay_safety_buffer_text(),
                self.replay_safety_buffer_focused(),
            ),
            replay_edit_line(
                "Safety buffer %",
                self.replay_safety_buffer_percent_text(),
                self.replay_safety_buffer_percent_focused(),
            ),
            replay_metadata_line(
                "Bar",
                if self.replay_selected_bar_supported() {
                    "supported"
                } else {
                    "unsupported"
                },
            ),
            replay_hint_line("Up/Down | digits | Enter"),
        ]
    }

    pub(in crate::app) fn replay_dataset_library_lines(&self) -> Vec<Line<'static>> {
        self.replay_cache_library_lines()
    }

    pub(in crate::app) fn replay_market_control_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            replay_edit_line(
                "Bar Type",
                format!("{} (Left/Right)", self.bar_type.kind().label()),
                self.focus == Focus::BarTypeToggle,
            ),
            replay_edit_line(
                self.replay_value_label(),
                format!("{} (digits)", self.bar_value_text()),
                self.focus == Focus::BarValue,
            ),
        ];

        if self.candle_mode_controls_visible() {
            lines.push(replay_edit_line(
                "Candles",
                format!("{} (Left/Right)", self.candle_mode.label()),
                self.focus == Focus::CandleModeToggle,
            ));
        }

        if !self.replay_selected_bar_supported() {
            lines.push(replay_metadata_line(
                "Readiness",
                "Volume needs per-trade size; this Last file only has price.",
            ));
        } else if self.replay_cache_can_serve_selected_bar() {
            lines.push(replay_metadata_line(
                "Readiness",
                self.replay_selected_source_readiness(),
            ));
        }
        if self.bar_type.kind() == BarKind::Tick {
            lines.push(replay_hint_line(
                "Replay tick-count is local; live Tick Count needs Tradovate validation with a valid token.",
            ));
        }

        lines.extend([
            replay_metadata_line(
                "Effective chart",
                self.bar_type.mode_label(self.effective_candle_mode()),
            ),
            replay_metadata_line(
                "Interval",
                format!(
                    "{}ms between derived bars at 1x",
                    self.base_config.replay_bar_interval_ms
                ),
            ),
            replay_hint_line("Up/Down navigates | Tab/Shift-Tab also moves focus"),
        ]);
        lines
    }

    pub(in crate::app) fn replay_run_control_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        lines.push(replay_action_line(
            self.replay_start_action_label(),
            self.focus == Focus::ReplayMode,
        ));
        lines.extend([
            replay_edit_line(
                format!(
                    "Initial capital ({})",
                    self.base_config.replay_account_currency
                ),
                self.replay_initial_capital_text(),
                self.replay_initial_capital_focused(),
            ),
            replay_edit_line(
                "Margin/contract",
                self.replay_margin_per_contract_text(),
                self.replay_margin_per_contract_focused(),
            ),
            replay_edit_line(
                "Safety buffer",
                self.replay_safety_buffer_text(),
                self.replay_safety_buffer_focused(),
            ),
            replay_edit_line(
                "Safety buffer %",
                self.replay_safety_buffer_percent_text(),
                self.replay_safety_buffer_percent_focused(),
            ),
            replay_metadata_line(
                "Margin model",
                format!(
                    "{} | analysis {}",
                    self.base_config.replay_margin_model,
                    if self.base_config.replay_margin_per_contract > 0.0 {
                        "enabled"
                    } else {
                        "disabled (set margin/contract > 0)"
                    }
                ),
            ),
        ]);
        lines.push(replay_metadata_line(
            "Replay source",
            self.replay_selected_source_readiness(),
        ));
        lines.push(replay_metadata_line(
            "Replay engine",
            self.base_config.replay_engine_mode.label(),
        ));
        lines.push(replay_metadata_line(
            "Fill model",
            match self.base_config.replay_engine_mode {
                ReplayEngineMode::Legacy => {
                    ReplayFillModel::LegacyReferencePrice.label().to_string()
                }
                ReplayEngineMode::Deterministic => {
                    self.base_config.replay_fill_model.label().to_string()
                }
            },
        ));
        lines.push(replay_metadata_line(
            "Latency model",
            match self.base_config.replay_engine_mode {
                ReplayEngineMode::Legacy => "ignored in Legacy".to_string(),
                ReplayEngineMode::Deterministic => {
                    let detail = match self.base_config.replay_latency_model {
                        ReplayLatencyModel::Fixed => {
                            format!("{}ms", self.base_config.replay_fixed_latency_ms)
                        }
                        ReplayLatencyModel::SeededObserved => format!(
                            "{} samples, seed {}",
                            self.base_config.replay_observed_latency_ms.len(),
                            self.base_config.replay_latency_seed
                        ),
                        _ => format!(
                            "{} samples",
                            self.base_config.replay_observed_latency_ms.len()
                        ),
                    };
                    format!(
                        "{} ({detail})",
                        self.base_config.replay_latency_model.label()
                    )
                }
            },
        ));
        lines.push(replay_metadata_line(
            "Bar protection",
            self.base_config.replay_bar_protection_policy.label(),
        ));
        #[cfg(feature = "replay")]
        lines.push(replay_metadata_line(
            "Dataset window",
            self.replay_selected_view_label(),
        ));
        lines.push(replay_metadata_line(
            "Bar selection",
            if self.replay_selected_bar_supported() {
                "supported"
            } else {
                "unsupported for selected source"
            },
        ));
        lines.push(replay_hint_line(
            "Up/Down navigates | digits edit | Enter starts",
        ));
        lines.push(replay_hint_line("Esc returns to cached contracts"));
        lines
    }

    pub(in crate::app) fn replay_dataset_available(&self) -> bool {
        if self.replay_dataset_index_is_selected() {
            self.replay_selected_cache_can_serve_bar()
        } else {
            self.replay_selected_cache_can_serve_bar() || self.local_replay_dataset_available()
        }
    }

    pub(in crate::app) fn replay_selected_bar_supported(&self) -> bool {
        if self.replay_dataset_index_is_selected() {
            self.replay_cache_can_serve_selected_bar()
        } else {
            self.replay_cache_can_serve_selected_bar() || self.bar_type.kind() != BarKind::Volume
        }
    }

    fn local_replay_dataset_available(&self) -> bool {
        replay_dataset_file_metadata(&self.base_config.replay_file_path).is_some()
    }

    pub(in crate::app) fn replay_cache_can_serve_selected_bar(&self) -> bool {
        self.replay_selected_cache_can_serve_bar()
    }

    #[cfg(feature = "replay")]
    fn replay_selected_source_readiness(&self) -> String {
        let Some(dataset) = self.replay_selected_dataset() else {
            return if self.local_replay_dataset_available() {
                "local text-file replay"
            } else {
                "missing replay data"
            }
            .to_string();
        };
        if dataset
            .server_bars_file_for(self.bar_type, self.effective_candle_mode(), None)
            .is_some()
        {
            "exact cached server bars".to_string()
        } else if dataset.raw_ticks_parquet_file_for(None).is_some() {
            "bars derived from cached raw ticks".to_string()
        } else {
            "selected cache does not support this bar".to_string()
        }
    }

    #[cfg(not(feature = "replay"))]
    fn replay_selected_source_readiness(&self) -> String {
        if self.local_replay_dataset_available() {
            "local text-file replay".to_string()
        } else {
            "missing replay data".to_string()
        }
    }

    #[cfg(feature = "replay")]
    fn replay_dataset_index_is_selected(&self) -> bool {
        self.replay_dataset_index.is_some()
    }

    #[cfg(not(feature = "replay"))]
    fn replay_dataset_index_is_selected(&self) -> bool {
        false
    }

    #[cfg(feature = "replay")]
    pub(in crate::app) fn replay_selected_dataset(
        &self,
    ) -> Option<&crate::replay_cache::ReplayCacheDataset> {
        self.replay_dataset_index
            .and_then(|index| self.replay_cache_library.datasets.get(index))
    }

    #[cfg(feature = "replay")]
    fn replay_selected_cache_can_serve_bar(&self) -> bool {
        if let Some(dataset) = self.replay_selected_dataset() {
            return dataset.can_serve(self.bar_type, self.effective_candle_mode(), None)
                || dataset.raw_ticks_parquet_file_for(None).is_some();
        }
        #[cfg(feature = "replay")]
        {
            self.replay_cache_library
                .first_server_bars(self.bar_type, self.effective_candle_mode(), None)
                .is_some()
                || self
                    .replay_cache_library
                    .raw_ticks_parquet_datasets(None)
                    .len()
                    == 1
        }

        #[cfg(not(feature = "replay"))]
        {
            false
        }
    }

    #[cfg(not(feature = "replay"))]
    fn replay_selected_cache_can_serve_bar(&self) -> bool {
        false
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
            if self
                .replay_selected_source_readiness()
                .contains("raw ticks")
            {
                "[Enter] Start Replay (derive from raw ticks)".to_string()
            } else {
                "[Enter] Start Replay (cached server bars)".to_string()
            }
        } else {
            "[Enter] Start Local Replay".to_string()
        }
    }

    fn replay_start_action_compact_label(&self) -> String {
        if !self.replay_dataset_available() {
            "Enter: replay (missing)".to_string()
        } else if !self.replay_selected_bar_supported() {
            "Enter: replay (unsupported)".to_string()
        } else if self.replay_cache_can_serve_selected_bar() {
            if self
                .replay_selected_source_readiness()
                .contains("raw ticks")
            {
                "Enter: replay (raw ticks)".to_string()
            } else {
                "Enter: replay (cached bars)".to_string()
            }
        } else {
            "Enter: local replay".to_string()
        }
    }

    #[cfg(feature = "replay")]
    fn replay_initial_capital_focused(&self) -> bool {
        self.focus == Focus::ReplayInitialCapital
    }

    #[cfg(not(feature = "replay"))]
    fn replay_initial_capital_focused(&self) -> bool {
        false
    }

    #[cfg(feature = "replay")]
    fn replay_margin_per_contract_focused(&self) -> bool {
        self.focus == Focus::ReplayMarginPerContract
    }

    #[cfg(not(feature = "replay"))]
    fn replay_margin_per_contract_focused(&self) -> bool {
        false
    }

    #[cfg(feature = "replay")]
    fn replay_safety_buffer_focused(&self) -> bool {
        self.focus == Focus::ReplaySafetyBuffer
    }

    #[cfg(not(feature = "replay"))]
    fn replay_safety_buffer_focused(&self) -> bool {
        false
    }

    #[cfg(feature = "replay")]
    fn replay_safety_buffer_percent_focused(&self) -> bool {
        self.focus == Focus::ReplaySafetyBufferPercent
    }

    #[cfg(not(feature = "replay"))]
    fn replay_safety_buffer_percent_focused(&self) -> bool {
        false
    }

    #[cfg(feature = "replay")]
    fn replay_initial_capital_text(&self) -> String {
        self.strategy_numeric_value(
            Focus::ReplayInitialCapital,
            format_float_input(self.base_config.replay_initial_capital),
        )
    }

    #[cfg(not(feature = "replay"))]
    fn replay_initial_capital_text(&self) -> String {
        format_float_input(self.base_config.replay_initial_capital)
    }

    #[cfg(feature = "replay")]
    fn replay_margin_per_contract_text(&self) -> String {
        self.strategy_numeric_value(
            Focus::ReplayMarginPerContract,
            format_float_input(self.base_config.replay_margin_per_contract),
        )
    }

    #[cfg(not(feature = "replay"))]
    fn replay_margin_per_contract_text(&self) -> String {
        format_float_input(self.base_config.replay_margin_per_contract)
    }

    #[cfg(feature = "replay")]
    fn replay_safety_buffer_text(&self) -> String {
        self.strategy_numeric_value(
            Focus::ReplaySafetyBuffer,
            format_float_input(self.base_config.replay_safety_buffer),
        )
    }

    #[cfg(not(feature = "replay"))]
    fn replay_safety_buffer_text(&self) -> String {
        format_float_input(self.base_config.replay_safety_buffer)
    }

    #[cfg(feature = "replay")]
    fn replay_safety_buffer_percent_text(&self) -> String {
        self.strategy_numeric_value(
            Focus::ReplaySafetyBufferPercent,
            format_float_input(self.base_config.replay_safety_buffer_percent),
        )
    }

    #[cfg(not(feature = "replay"))]
    fn replay_safety_buffer_percent_text(&self) -> String {
        format_float_input(self.base_config.replay_safety_buffer_percent)
    }

    #[cfg(feature = "replay")]
    fn replay_selected_view_label(&self) -> String {
        self.replay_dataset_view_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|value| value.to_str())
            .map(|id| format!("saved view `{id}`"))
            .unwrap_or_else(|| "full source coverage".to_string())
    }
}

impl App {
    #[cfg(feature = "replay")]
    fn replay_cache_library_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from("Cached Instruments / Contracts"),
            Line::from(format!(
                "Cache root: {}",
                self.base_config.replay_cache_dir.display()
            )),
            styled_line(
                format!(
                    "Search instruments: {}",
                    if self.replay_instrument_query.is_empty() {
                        "<all>"
                    } else {
                        &self.replay_instrument_query
                    }
                ),
                self.focus == Focus::ReplayInstrumentQuery,
            ),
        ];

        let indices = self.replay_filtered_dataset_indices();
        if indices.is_empty() {
            lines.extend([
                Line::from(if self.replay_cache_library.datasets.is_empty() {
                    "Status: no manifest.json datasets found"
                } else {
                    "Status: no cached contract matches the search"
                }),
                Line::from(
                    "Enter searches/downloads when empty; Tab reviews run setup.",
                ),
                Line::from(
                    "The Replay downloader uses read-only REST and market-data history without user or order streams.",
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

        let selected = self.replay_selected_dataset();
        let options = self.replay_dataset_options();
        let selection_hint = if options.len() == indices.len() {
            "Up/Down selects a cached contract; Enter/Right opens replay setup."
        } else {
            "Up/Down selects a cached contract shape; Enter/Right opens replay setup."
        };
        lines.push(Line::from(format!(
            "Status: {} cached contract(s), selected {}",
            indices.len(),
            if selected.is_some() {
                self.replay_selected_choice_label()
            } else {
                "none".to_string()
            }
        )));
        lines.push(Line::from(selection_hint));
        lines.push(Line::from(if self.focus == Focus::ReplayDataset {
            "N downloads from the list; D extends; V manages saved ranges."
        } else {
            "Select a row first; N/D/V manage the selected dataset."
        }));

        const VISIBLE_DATASETS: usize = 8;
        let selected_position = self.replay_selected_option_position(&options).unwrap_or(0);
        let visible_start = selected_position
            .saturating_sub(VISIBLE_DATASETS - 1)
            .min(options.len().saturating_sub(VISIBLE_DATASETS));
        let visible_end = (visible_start + VISIBLE_DATASETS).min(options.len());
        lines.push(Line::from(format!(
            "Showing cached shapes {}-{} of {}",
            visible_start + 1,
            visible_end,
            options.len()
        )));
        for (position, (index, bar_type)) in options
            .iter()
            .copied()
            .enumerate()
            .skip(visible_start)
            .take(VISIBLE_DATASETS)
        {
            lines.push(self.replay_dataset_row_line(position + 1, index, bar_type));
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

fn replay_metadata_line(label: impl Into<String>, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{}: ", label.into()),
            Style::default().fg(Color::Gray),
        ),
        Span::styled(value.into(), Style::default().fg(Color::White)),
    ])
}

fn replay_edit_line(
    label: impl Into<String>,
    value: impl Into<String>,
    focused: bool,
) -> Line<'static> {
    let value_style = if focused {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };
    Line::from(vec![
        Span::styled(
            format!("{}: ", label.into()),
            Style::default().fg(Color::Gray),
        ),
        Span::styled(value.into(), value_style),
    ])
}

fn replay_action_line(text: impl Into<String>, focused: bool) -> Line<'static> {
    let style = if focused {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    };
    Line::from(Span::styled(text.into(), style))
}

fn replay_hint_line(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        text.into(),
        Style::default().fg(Color::DarkGray),
    ))
}

fn display_path_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| path.to_str().unwrap_or("-"))
        .to_string()
}

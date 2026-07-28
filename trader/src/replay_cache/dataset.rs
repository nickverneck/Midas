use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayCacheDataset {
    pub manifest_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub manifest: ReplayCacheManifest,
}

impl ReplayCacheDataset {
    pub fn can_serve(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        self.manifest
            .supports_replay(bar_type, candle_mode, requested_coverage)
    }

    pub fn server_bars_jsonl_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty() {
            return None;
        }
        if requested_coverage.is_some_and(|requested| !self.manifest.coverage.contains(requested)) {
            return None;
        }
        self.manifest.files.iter().find(|file| {
            file.errors.is_empty()
                && file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.format == ReplayCacheFileFormat::Jsonl
                && file.market_shape.chart_mode != Some(CandleMode::HeikinAshi)
                && file
                    .schema_version
                    .is_none_or(|version| version == SERVER_BARS_SCHEMA_VERSION)
                && file.market_shape.supports(bar_type, candle_mode)
        })
    }

    pub fn server_bars_parquet_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty()
            || requested_coverage
                .is_some_and(|requested| !self.manifest.coverage.contains(requested))
        {
            return None;
        }
        self.manifest.files.iter().find(|file| {
            file.errors.is_empty()
                && file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.format == ReplayCacheFileFormat::Parquet
                && file.market_shape.chart_mode != Some(CandleMode::HeikinAshi)
                && file
                    .schema_version
                    .is_none_or(|version| version == SERVER_BARS_SCHEMA_VERSION)
                && file.market_shape.supports(bar_type, candle_mode)
        })
    }

    pub fn server_bars_file_for(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        self.server_bars_parquet_file_for(bar_type, candle_mode, requested_coverage)
            .or_else(|| self.server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage))
    }

    #[allow(dead_code)]
    pub fn resolve_server_bars_jsonl_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = self
            .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
            .with_context(|| {
                format!(
                    "no cached JSONL server bars for {} in {}",
                    bar_type.mode_label(candle_mode),
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }

    pub fn raw_ticks_parquet_file_for(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataFile> {
        self.raw_ticks_parquet_files_for(requested_coverage)
            .into_iter()
            .next()
    }

    pub fn raw_ticks_parquet_files_for(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Vec<&ReplayCacheDataFile> {
        if !self.manifest.errors.is_empty() {
            return Vec::new();
        }
        let effective_coverage = self
            .manifest
            .completed_raw_tick_coverage
            .as_ref()
            .unwrap_or(&self.manifest.coverage);
        if requested_coverage.is_some_and(|requested| !effective_coverage.contains(requested)) {
            return Vec::new();
        }
        let mut files = self
            .manifest
            .files
            .iter()
            .filter(|file| {
                file.errors.is_empty()
                    && file.source_kind == ReplayCacheSourceKind::RawTicks
                    && file.format == ReplayCacheFileFormat::Parquet
                    && raw_tick_manifest_entry_is_replayable(file)
            })
            .collect::<Vec<_>>();
        files.sort_by(|left, right| {
            left.request_start
                .cmp(&right.request_start)
                .then_with(|| left.relative_path.cmp(&right.relative_path))
        });
        if self.manifest.completed_raw_tick_windows.is_empty() {
            return (files.len() == 1
                && files[0].request_start.is_none()
                && files[0].request_end.is_none())
            .then_some(files)
            .unwrap_or_default();
        }
        let Some(completed) = self.manifest.completed_raw_tick_coverage.as_ref() else {
            return Vec::new();
        };
        let Ok(parent) = DownloadWindow::new(completed.start, completed.end) else {
            return Vec::new();
        };
        if validate_exact_window_coverage(parent, &self.manifest.completed_raw_tick_windows)
            .is_err()
        {
            return Vec::new();
        }
        let mut seen_windows = BTreeSet::new();
        if files.iter().any(|file| {
            let Some(start) = file.request_start else {
                return true;
            };
            let Some(end) = file.request_end else {
                return true;
            };
            let window = DownloadWindow { start, end };
            !self.manifest.completed_raw_tick_windows.contains(&window)
                || !seen_windows.insert((start, end))
        }) {
            return Vec::new();
        }
        files
    }

    pub fn resolve_raw_ticks_parquet_file(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedRawTicksFile> {
        let files = self.raw_ticks_parquet_files_for(requested_coverage);
        if files.len() != 1 {
            bail!(
                "raw-tick dataset {} resolves to {} Parquet files; use the multi-file resolver",
                self.manifest_path.display(),
                files.len()
            );
        }
        let file = files
            .into_iter()
            .next()
            .with_context(|| {
                format!(
                    "no cached Parquet raw ticks in {}",
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let data_file = Arc::new(
                File::open(&data_path)
                    .with_context(|| format!("open replay cache lease {}", data_path.display()))?,
            );
            let source = PositionIndependentFile::new(data_file.clone())?;
            validate_raw_tick_manifest_hash(&source, &file, &data_path)?;
            return Ok(ReplayCacheResolvedRawTicksFile {
                manifest_path: self.manifest_path.clone(),
                dataset_dir: self.dataset_dir.clone(),
                data_path,
                data_file,
                manifest: self.manifest.clone(),
                file,
            });
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = data_path;
            bail!(
                "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
            )
        }
    }

    pub fn resolve_raw_ticks_parquet_files(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedRawTicks> {
        let entries = self.raw_ticks_parquet_files_for(requested_coverage);
        if entries.is_empty() {
            bail!(
                "no complete cached Parquet raw-tick coverage in {}",
                self.manifest_path.display()
            );
        }
        let mut files = Vec::with_capacity(entries.len());
        for entry in entries {
            let file = entry.clone();
            let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                let data_file =
                    Arc::new(File::open(&data_path).with_context(|| {
                        format!("open replay cache lease {}", data_path.display())
                    })?);
                let source = PositionIndependentFile::new(data_file.clone())?;
                validate_raw_tick_manifest_hash(&source, &file, &data_path)?;
                files.push(ReplayCacheResolvedRawTicksFile {
                    manifest_path: self.manifest_path.clone(),
                    dataset_dir: self.dataset_dir.clone(),
                    data_path,
                    data_file,
                    manifest: self.manifest.clone(),
                    file,
                });
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            {
                let _ = data_path;
                bail!(
                    "position-independent leased replay cache reads are unsupported on this target; supported targets are Linux and macOS"
                );
            }
        }
        Ok(ReplayCacheResolvedRawTicks {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            manifest: self.manifest.clone(),
            files,
        })
    }

    #[allow(dead_code)]
    pub fn resolve_server_bars_parquet_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        self.resolve_server_bars_file_with_format(
            bar_type,
            candle_mode,
            requested_coverage,
            ReplayCacheFileFormat::Parquet,
        )
    }

    pub fn resolve_server_bars_file(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = self
            .server_bars_file_for(bar_type, candle_mode, requested_coverage)
            .with_context(|| {
                format!(
                    "no cached server bars for {} in {}",
                    bar_type.mode_label(candle_mode),
                    self.manifest_path.display()
                )
            })?
            .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }

    #[allow(dead_code)]
    fn resolve_server_bars_file_with_format(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
        format: ReplayCacheFileFormat,
    ) -> Result<ReplayCacheResolvedServerBarsFile> {
        let file = match format {
            ReplayCacheFileFormat::Parquet => {
                self.server_bars_parquet_file_for(bar_type, candle_mode, requested_coverage)
            }
            ReplayCacheFileFormat::Jsonl => {
                self.server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
            }
            _ => None,
        }
        .with_context(|| format!("no cached server bars in {}", self.manifest_path.display()))?
        .clone();
        let data_path = resolve_cache_data_path(&self.dataset_dir, &file.relative_path)?;
        Ok(ReplayCacheResolvedServerBarsFile {
            manifest_path: self.manifest_path.clone(),
            dataset_dir: self.dataset_dir.clone(),
            data_path,
            manifest: self.manifest.clone(),
            file,
        })
    }
}

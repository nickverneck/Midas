use super::*;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ReplayCacheLibrary {
    pub root: PathBuf,
    pub datasets: Vec<ReplayCacheDataset>,
    pub warnings: Vec<String>,
}

impl ReplayCacheLibrary {
    pub fn scan(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let mut library = Self {
            root: root.clone(),
            datasets: Vec::new(),
            warnings: Vec::new(),
        };
        scan_manifest_paths(
            &root,
            &mut |path| match ReplayCacheManifest::from_path(path) {
                Ok(manifest) => library.datasets.push(ReplayCacheDataset {
                    dataset_dir: path.parent().unwrap_or(root.as_path()).to_path_buf(),
                    manifest_path: path.to_path_buf(),
                    manifest,
                }),
                Err(err) => library
                    .warnings
                    .push(format!("{}: {err:#}", path.display())),
            },
        );
        library.datasets.sort_by(|left, right| {
            right
                .manifest
                .coverage
                .end
                .cmp(&left.manifest.coverage.end)
                .then_with(|| left.manifest.display_name.cmp(&right.manifest.display_name))
        });
        library
    }

    #[allow(dead_code)]
    pub fn first_serving(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets
            .iter()
            .find(|dataset| dataset.can_serve(bar_type, candle_mode, requested_coverage))
    }

    pub fn first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets.iter().find(|dataset| {
            dataset.can_serve(bar_type, candle_mode, requested_coverage)
                && dataset
                    .server_bars_jsonl_file_for(bar_type, candle_mode, requested_coverage)
                    .is_some()
        })
    }

    pub fn first_server_bars(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Option<&ReplayCacheDataset> {
        self.datasets.iter().find(|dataset| {
            dataset.can_serve(bar_type, candle_mode, requested_coverage)
                && dataset
                    .server_bars_file_for(bar_type, candle_mode, requested_coverage)
                    .is_some()
        })
    }

    #[allow(dead_code)]
    pub fn load_first_server_bars_jsonl(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedServerBars>> {
        let Some(dataset) = self.first_server_bars_jsonl(bar_type, candle_mode, requested_coverage)
        else {
            return Ok(None);
        };
        load_server_bars_jsonl_cache_file(dataset, bar_type, candle_mode, requested_coverage)
            .map(Some)
    }

    pub fn load_first_server_bars(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedServerBars>> {
        let Some(dataset) = self.first_server_bars(bar_type, candle_mode, requested_coverage)
        else {
            return Ok(None);
        };
        load_server_bars_cache_file(dataset, bar_type, candle_mode, requested_coverage).map(Some)
    }

    pub fn raw_ticks_parquet_datasets(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Vec<&ReplayCacheDataset> {
        self.datasets
            .iter()
            .filter(|dataset| {
                dataset
                    .raw_ticks_parquet_file_for(requested_coverage)
                    .is_some()
            })
            .collect()
    }

    pub fn load_unique_raw_ticks_parquet(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheLoadedRawTicks>> {
        self.load_unique_raw_ticks_parquet_range(requested_coverage, None)
    }

    pub fn resolve_unique_raw_ticks_parquet(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheResolvedRawTicksFile>> {
        let datasets = self.raw_ticks_parquet_datasets(requested_coverage);
        if datasets.len() > 1 {
            let names = datasets
                .iter()
                .map(|dataset| dataset.manifest.display_name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "multiple cached raw-tick datasets match replay: {names}; select one dataset before starting replay"
            );
        }
        let Some(dataset) = datasets.first().copied() else {
            return Ok(None);
        };
        dataset
            .resolve_raw_ticks_parquet_file(requested_coverage)
            .map(Some)
    }

    pub fn resolve_unique_raw_ticks_parquet_files(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<Option<ReplayCacheResolvedRawTicks>> {
        let datasets = self.raw_ticks_parquet_datasets(requested_coverage);
        if datasets.len() > 1 {
            let names = datasets
                .iter()
                .map(|dataset| dataset.manifest.display_name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "multiple cached raw-tick datasets match replay: {names}; select one dataset before starting replay"
            );
        }
        let Some(dataset) = datasets.first().copied() else {
            return Ok(None);
        };
        dataset
            .resolve_raw_ticks_parquet_files(requested_coverage)
            .map(Some)
    }

    pub fn load_unique_raw_ticks_parquet_range(
        &self,
        requested_coverage: Option<&ReplayCacheCoverage>,
        timestamp_range: Option<&ReplayCacheTimeRange>,
    ) -> Result<Option<ReplayCacheLoadedRawTicks>> {
        let Some(resolved) = self.resolve_unique_raw_ticks_parquet(requested_coverage)? else {
            return Ok(None);
        };
        let mut ticks = Vec::new();
        stream_resolved_raw_ticks_parquet(&resolved, timestamp_range, |row| {
            ticks.push(row);
            Ok(())
        })?;
        Ok(Some(ReplayCacheLoadedRawTicks {
            manifest_path: resolved.manifest_path,
            dataset_dir: resolved.dataset_dir,
            data_path: resolved.data_path,
            manifest: resolved.manifest,
            file: resolved.file,
            ticks,
        }))
    }

    pub fn load_raw_ticks_parquet_dataset(
        &self,
        dataset: &ReplayCacheDataset,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> Result<ReplayCacheLoadedRawTicks> {
        self.load_raw_ticks_parquet_dataset_range(dataset, requested_coverage, None)
    }

    pub fn load_raw_ticks_parquet_dataset_range(
        &self,
        dataset: &ReplayCacheDataset,
        requested_coverage: Option<&ReplayCacheCoverage>,
        timestamp_range: Option<&ReplayCacheTimeRange>,
    ) -> Result<ReplayCacheLoadedRawTicks> {
        let resolved = dataset.resolve_raw_ticks_parquet_file(requested_coverage)?;
        let mut ticks = Vec::new();
        stream_resolved_raw_ticks_parquet(&resolved, timestamp_range, |row| {
            ticks.push(row);
            Ok(())
        })?;
        Ok(ReplayCacheLoadedRawTicks {
            manifest_path: resolved.manifest_path,
            dataset_dir: resolved.dataset_dir,
            data_path: resolved.data_path,
            manifest: resolved.manifest,
            file: resolved.file,
            ticks,
        })
    }
}

pub(super) fn scan_manifest_paths(root: &Path, on_manifest: &mut impl FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|value| value.to_str()) == Some(MANIFEST_FILE_NAME) {
            on_manifest(&path);
            continue;
        }
        if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            scan_manifest_paths(&path, on_manifest);
        }
    }
}

pub(super) fn resolve_cache_data_path(dataset_dir: &Path, relative_path: &Path) -> Result<PathBuf> {
    if relative_path.is_absolute() {
        bail!(
            "cache data path {} must be relative to its manifest",
            relative_path.display()
        );
    }
    for component in relative_path.components() {
        match component {
            Component::Normal(_) => {}
            _ => bail!(
                "cache data path {} is not a safe manifest-relative path",
                relative_path.display()
            ),
        }
    }
    let data_path = dataset_dir.join(relative_path);
    if !data_path.is_file() {
        bail!("cache data file {} was not found", data_path.display());
    }
    Ok(data_path)
}

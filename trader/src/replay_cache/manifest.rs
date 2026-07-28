use super::*;

pub(super) struct ReplayManifestLock {
    file: File,
}

impl ReplayManifestLock {
    pub(super) fn acquire(dataset_dir: &Path) -> Result<Self> {
        Self::acquire_with_timeout(dataset_dir, MANIFEST_LOCK_WAIT)
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub(super) fn acquire_with_timeout(dataset_dir: &Path, timeout: Duration) -> Result<Self> {
        fs::create_dir_all(dataset_dir)
            .with_context(|| format!("create {}", dataset_dir.display()))?;
        let lock_path = dataset_dir.join(MANIFEST_LOCK_FILE_NAME);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&lock_path)
            .with_context(|| format!("open replay cache manifest lock {}", lock_path.display()))?;
        let started = Instant::now();

        loop {
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                return Ok(Self { file });
            }

            let err = std::io::Error::last_os_error();
            let raw_error = err.raw_os_error();
            if raw_error != Some(libc::EWOULDBLOCK)
                && raw_error != Some(libc::EAGAIN)
                && raw_error != Some(libc::EINTR)
            {
                return Err(err).with_context(|| {
                    format!("lock replay cache manifest {}", lock_path.display())
                });
            }
            if started.elapsed() >= timeout {
                bail!(
                    "timed out waiting for replay cache manifest lock {} after {} ms",
                    lock_path.display(),
                    timeout.as_millis()
                );
            }
            let retry_delay =
                Duration::from_millis(20).min(timeout.saturating_sub(started.elapsed()));
            std::thread::sleep(retry_delay);
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn acquire_with_timeout(dataset_dir: &Path, timeout: Duration) -> Result<Self> {
        let _ = (dataset_dir, timeout);
        bail!(
            "replay cache manifest locking is unsupported on this target; supported targets are Linux and macOS"
        )
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Drop for ReplayManifestLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ReplayCacheMetadataIdentity {
    provider: BrokerKind,
    env: TradingEnvironment,
    user_id: i64,
    account_ids: Vec<i64>,
    contract_id: i64,
    product_id: i64,
}

fn contract_metadata_identity(
    metadata: &ReplayCacheContractMetadata,
) -> Option<ReplayCacheMetadataIdentity> {
    let user_id = metadata.context.user_id?;
    let mut account_ids = metadata
        .context
        .accounts
        .iter()
        .map(|account| account.id)
        .collect::<Vec<_>>();
    account_ids.sort_unstable();
    account_ids.dedup();
    if account_ids.is_empty() {
        return None;
    }
    let contract_id = metadata.contract.payload.get("id")?.as_i64()?;
    let product_id = metadata
        .product
        .as_ref()
        .and_then(|snapshot| snapshot.payload.get("id"))
        .and_then(Value::as_i64)
        .or_else(|| {
            metadata
                .maturity
                .as_ref()
                .and_then(|snapshot| snapshot.payload.get("productId"))
                .and_then(Value::as_i64)
        })?;
    Some(ReplayCacheMetadataIdentity {
        provider: metadata.context.provider,
        env: metadata.context.env,
        user_id,
        account_ids,
        contract_id,
        product_id,
    })
}

pub(super) fn merge_contract_metadata(
    existing: Option<ReplayCacheContractMetadata>,
    incoming: Option<ReplayCacheContractMetadata>,
) -> Option<ReplayCacheContractMetadata> {
    let Some(mut incoming) = incoming else {
        return None;
    };
    let Some(existing) = existing else {
        return Some(incoming);
    };
    let identities_match = contract_metadata_identity(&existing)
        .zip(contract_metadata_identity(&incoming))
        .is_some_and(|(existing, incoming)| existing == incoming);
    if !identities_match {
        return Some(incoming);
    }
    incoming.maturity = incoming.maturity.or(existing.maturity);
    incoming.maturity_chain = incoming.maturity_chain.or(existing.maturity_chain);
    incoming.product = incoming.product.or(existing.product);
    incoming.product_sessions = incoming.product_sessions.or(existing.product_sessions);
    incoming.product_margins = incoming.product_margins.or(existing.product_margins);
    incoming.contract_margins = incoming.contract_margins.or(existing.contract_margins);
    incoming.fee_params = incoming.fee_params.or(existing.fee_params);
    incoming.suggested_coverage = incoming.suggested_coverage.or(existing.suggested_coverage);
    Some(incoming)
}

impl ReplayCacheManifest {
    pub fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut manifest: Self =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        manifest.normalize_derived_fields();
        Ok(manifest)
    }

    pub fn supports_replay(
        &self,
        bar_type: BarType,
        candle_mode: CandleMode,
        requested_coverage: Option<&ReplayCacheCoverage>,
    ) -> bool {
        if !self.errors.is_empty() {
            return false;
        }
        if requested_coverage.is_some_and(|requested| !self.coverage.contains(requested)) {
            return false;
        }
        self.files
            .iter()
            .any(|file| replay_file_can_serve(file, bar_type, candle_mode))
    }

    pub fn normalize_derived_fields(&mut self) {
        self.source_kind = manifest_source_kind_from_files(&self.files, self.source_kind);
        if self.available_bar_shapes.is_empty() {
            let mut shapes = Vec::new();
            for file in &self.files {
                if let Some(bar_type) = file.market_shape.bar_type
                    && !shapes.contains(&bar_type)
                {
                    shapes.push(bar_type);
                }
            }
            self.available_bar_shapes = shapes;
        }

        if self.available_chart_modes.is_empty() {
            let mut modes = Vec::new();
            for file in &self.files {
                if let Some(mode) = file.market_shape.chart_mode
                    && !modes.contains(&mode)
                {
                    modes.push(mode);
                }
            }
            self.available_chart_modes = modes;
        }

        if self.badges.is_empty() {
            self.badges = self.derived_badges();
        }
    }

    pub fn derived_badges(&self) -> Vec<String> {
        let mut badges = BTreeSet::new();
        if self.source_kind != ReplayCacheSourceKind::Mixed {
            badges.insert(self.source_kind.badge().to_string());
        }
        for file in &self.files {
            badges.insert(file.source_kind.badge().to_string());
            if file.market_shape.bar_type.is_some_and(|bar_type| {
                matches!(
                    bar_type.kind(),
                    BarKind::Tick | BarKind::Volume | BarKind::Range
                )
            }) {
                badges.insert(file.market_shape.label().to_ascii_lowercase());
            }
            if let Some(chart_mode) = file.market_shape.chart_mode {
                badges.insert(chart_mode.label().to_ascii_lowercase());
            }
        }
        badges.into_iter().collect()
    }

    #[cfg(test)]
    pub(super) fn row_count_total(&self) -> u64 {
        self.files.iter().map(|file| file.row_count).sum()
    }

    pub fn preferred_row_count_total(&self) -> u64 {
        self.files
            .iter()
            .filter(|file| {
                !(file.format == ReplayCacheFileFormat::Jsonl
                    && self.files.iter().any(|candidate| {
                        candidate.source_kind == file.source_kind
                            && candidate.format == ReplayCacheFileFormat::Parquet
                            && candidate.market_shape == file.market_shape
                    }))
            })
            .map(|file| file.row_count)
            .sum()
    }

    pub fn has_source_kind(&self, source_kind: ReplayCacheSourceKind) -> bool {
        self.files
            .iter()
            .any(|file| file.source_kind == source_kind)
    }

    pub fn downloadable_source_kinds(&self) -> Vec<ReplayCacheSourceKind> {
        [
            ReplayCacheSourceKind::ServerBars,
            ReplayCacheSourceKind::RawTicks,
        ]
        .into_iter()
        .filter(|source_kind| self.has_source_kind(*source_kind))
        .collect()
    }

    pub fn available_shapes_label(&self) -> String {
        if self.available_bar_shapes.is_empty() {
            return "none listed".to_string();
        }
        self.available_bar_shapes
            .iter()
            .map(|bar_type| bar_type.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn available_chart_modes_label(&self) -> String {
        if self.available_chart_modes.is_empty() {
            return "none listed".to_string();
        }
        self.available_chart_modes
            .iter()
            .map(|mode| mode.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn badges_label(&self) -> String {
        if self.badges.is_empty() {
            return "none".to_string();
        }
        self.badges.join(", ")
    }
}

pub(super) fn replay_file_can_serve(
    file: &ReplayCacheDataFile,
    bar_type: BarType,
    candle_mode: CandleMode,
) -> bool {
    file.errors.is_empty()
        && matches!(
            file.source_kind,
            ReplayCacheSourceKind::ServerBars
                | ReplayCacheSourceKind::DerivedBars
                | ReplayCacheSourceKind::LocalText
        )
        && file.market_shape.supports(bar_type, candle_mode)
}

pub(super) fn manifest_source_kind_from_files(
    files: &[ReplayCacheDataFile],
    fallback: ReplayCacheSourceKind,
) -> ReplayCacheSourceKind {
    let Some(first) = files.first().map(|file| file.source_kind) else {
        return fallback;
    };
    if files.iter().all(|file| file.source_kind == first) {
        first
    } else {
        ReplayCacheSourceKind::Mixed
    }
}

pub(super) fn upsert_server_bars_manifest(
    write: &ReplayCacheServerBarsWrite,
    dataset_dir: &Path,
    data_file: ReplayCacheDataFile,
    first_timestamp: DateTime<Utc>,
    last_timestamp: DateTime<Utc>,
) -> Result<(PathBuf, ReplayCacheManifest)> {
    let manifest_path = dataset_dir.join(MANIFEST_FILE_NAME);
    let mut manifest = if manifest_path.exists() {
        ReplayCacheManifest::from_path(&manifest_path)
            .with_context(|| format!("load existing {}", manifest_path.display()))?
    } else {
        ReplayCacheManifest {
            manifest_version: MANIFEST_VERSION,
            provider: write.provider,
            env: write.env,
            instrument: write.instrument.clone(),
            contract: write.contract.clone(),
            display_name: write.display_name.clone().unwrap_or_else(|| {
                replay_cache_display_name(
                    &write.contract.symbol,
                    write.request_start,
                    write.request_end,
                    write.bar_type,
                )
            }),
            coverage: ReplayCacheCoverage {
                start: first_timestamp,
                end: last_timestamp,
                trading_date: Some(write.request_start.date_naive()),
            },
            completed_raw_tick_coverage: None,
            completed_raw_tick_windows: Vec::new(),
            source_kind: ReplayCacheSourceKind::ServerBars,
            download_request: Value::Null,
            tick_specs: write.tick_specs.clone(),
            contract_metadata: write.contract_metadata.clone(),
            files: Vec::new(),
            app: None,
            warnings: Vec::new(),
            errors: Vec::new(),
            badges: Vec::new(),
            available_bar_shapes: Vec::new(),
            available_chart_modes: Vec::new(),
            tags: write.tags.clone().unwrap_or_default(),
            notes: write.notes.clone(),
        }
    };
    manifest.provider = write.provider;
    manifest.env = write.env;
    manifest.instrument = write.instrument.clone();
    manifest.contract = write.contract.clone();
    manifest.source_kind = ReplayCacheSourceKind::ServerBars;
    manifest.download_request = write.download_request.clone();
    manifest.tick_specs = write.tick_specs.clone();
    manifest.contract_metadata = merge_contract_metadata(
        manifest.contract_metadata.take(),
        write.contract_metadata.clone(),
    );
    manifest.app = Some(ReplayCacheAppMetadata {
        app_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        git_commit: option_env!("VERGEN_GIT_SHA").map(ToString::to_string),
        generated_at: Some(Utc::now()),
    });
    manifest.warnings = write.warnings.clone();
    manifest.errors.clear();
    if let Some(display_name) = write
        .display_name
        .clone()
        .filter(|name| !name.trim().is_empty())
    {
        manifest.display_name = display_name;
    }
    if let Some(tags) = write.tags.clone() {
        manifest.tags = normalize_cache_tags(tags);
    }
    manifest.notes = write.notes.clone();
    let shape = data_file.market_shape.bar_type;
    let format = data_file.format.clone();
    let superseded_files = manifest
        .files
        .iter()
        .filter(|file| {
            file.source_kind == ReplayCacheSourceKind::ServerBars
                && file.market_shape.bar_type == shape
                && file.format == format
        })
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    manifest.files.retain(|file| {
        !(file.source_kind == ReplayCacheSourceKind::ServerBars
            && file.market_shape.bar_type == shape
            && file.format == format)
    });
    manifest.files.push(data_file);
    manifest.files.sort_by(|left, right| {
        left.relative_path
            .to_string_lossy()
            .cmp(&right.relative_path.to_string_lossy())
    });
    manifest.source_kind =
        manifest_source_kind_from_files(&manifest.files, ReplayCacheSourceKind::ServerBars);
    manifest.coverage = manifest_coverage_from_files(&manifest.files, write.request_start);
    manifest.available_bar_shapes = manifest
        .files
        .iter()
        .filter_map(|file| file.market_shape.bar_type)
        .collect();
    manifest
        .available_bar_shapes
        .sort_by_key(|bar_type| bar_type.value());
    manifest.available_bar_shapes.dedup();
    manifest.available_chart_modes = if write.bar_type.supports_candle_mode() {
        vec![CandleMode::Standard, CandleMode::HeikinAshi]
    } else {
        vec![CandleMode::Standard]
    };
    manifest.badges = manifest.derived_badges();
    fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    write_bytes_atomically(
        &manifest_path,
        &serde_json::to_vec_pretty(&manifest).context("serialize replay cache manifest")?,
    )?;
    remove_superseded_cache_files(&dataset_dir, &superseded_files);
    Ok((manifest_path, manifest))
}

pub(super) fn write_bytes_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    write_file_atomically(path, |file| {
        file.write_all(bytes)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    })
}

pub(super) fn write_file_atomically(
    path: &Path,
    write: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cache-file");
    let nonce = Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let mut write = Some(write);

    for attempt in 0..16_u8 {
        let temp_path = parent.join(format!(
            ".{file_name}.tmp-{}-{nonce}-{attempt}",
            std::process::id()
        ));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => file,
            Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("create {}", temp_path.display()));
            }
        };

        let result = (|| {
            write.take().expect("atomic writer runs once")(&mut file)?;
            file.sync_all()
                .with_context(|| format!("sync {}", temp_path.display()))?;
            drop(file);
            fs::rename(&temp_path, path).with_context(|| {
                format!("replace {} with {}", path.display(), temp_path.display())
            })?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        return result;
    }

    bail!(
        "could not allocate temporary file beside {}",
        path.display()
    )
}

pub(super) fn validate_loaded_server_bars_metadata(
    resolved: &ReplayCacheResolvedServerBarsFile,
    bars: &[Bar],
    timestamp_range: Option<&ReplayCacheTimeRange>,
) -> Result<()> {
    if timestamp_range.is_none() && bars.is_empty() {
        bail!(
            "server-bar cache file {} contained no bars",
            resolved.data_path.display()
        );
    }
    if let Some(range) = timestamp_range {
        for bar in bars {
            if !range.contains_ns(bar.ts_ns)? {
                bail!(
                    "server-bar cache file {} returned timestamp {} outside requested range",
                    resolved.data_path.display(),
                    bar.ts_ns
                );
            }
        }
        return Ok(());
    }
    if bars.len() as u64 != resolved.file.row_count {
        bail!(
            "server-bar cache row count mismatch for {}: manifest={} actual={}",
            resolved.data_path.display(),
            resolved.file.row_count,
            bars.len()
        );
    }
    let first_timestamp = DateTime::<Utc>::from_timestamp_nanos(bars[0].ts_ns);
    let last_timestamp =
        DateTime::<Utc>::from_timestamp_nanos(bars.last().expect("bars are non-empty").ts_ns);
    if first_timestamp != resolved.file.first_timestamp
        || last_timestamp != resolved.file.last_timestamp
    {
        bail!(
            "server-bar cache timestamp range mismatch for {}: manifest={}..{} actual={}..{}",
            resolved.data_path.display(),
            resolved.file.first_timestamp,
            resolved.file.last_timestamp,
            first_timestamp,
            last_timestamp
        );
    }
    Ok(())
}

pub(super) fn manifest_coverage_from_files(
    files: &[ReplayCacheDataFile],
    fallback_start: DateTime<Utc>,
) -> ReplayCacheCoverage {
    let start = files
        .iter()
        .map(|file| file.first_timestamp)
        .min()
        .unwrap_or(fallback_start);
    let end = files
        .iter()
        .map(|file| file.last_timestamp)
        .max()
        .unwrap_or(start);
    ReplayCacheCoverage {
        start,
        end,
        trading_date: Some(start.date_naive()),
    }
}

pub(super) fn normalize_cache_tags(tags: Vec<String>) -> Vec<String> {
    let mut tags = tags
        .into_iter()
        .map(|tag| tag.trim().to_string())
        .filter(|tag| !tag.is_empty())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

pub(super) fn replay_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> String {
    format!(
        "{} {} to {} {} server bars",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type.label()
    )
}

pub(super) fn raw_ticks_cache_display_name(
    contract: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> String {
    format!(
        "{} raw ticks {} to {}",
        contract,
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    )
}

pub(super) fn bar_type_file_label(bar_type: BarType) -> String {
    let kind = match bar_type.kind() {
        BarKind::Minute => "minute",
        BarKind::Second => "second",
        BarKind::Tick => "tick",
        BarKind::Volume => "volume",
        BarKind::Range => "range",
    };
    format!("{}{}", bar_type.value(), kind)
}

pub(super) fn safe_cache_segment(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

pub(super) fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

pub(super) fn fnv1a64_file_hex(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    fnv1a64_reader_hex(file, path)
}

pub(super) fn fnv1a64_reader_hex<R: Read>(reader: R, path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(64 * 1024, reader);
    let mut buffer = [0_u8; 64 * 1024];
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

pub(super) fn raw_tick_manifest_entry_is_replayable(file: &ReplayCacheDataFile) -> bool {
    matches!(
        file.schema_version,
        Some(RAW_TICKS_LEGACY_SCHEMA_VERSION | RAW_TICKS_SCHEMA_VERSION)
    )
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) fn validate_raw_tick_manifest_hash(
    source: &PositionIndependentFile,
    file: &ReplayCacheDataFile,
    path: &Path,
) -> Result<()> {
    let expected = file
        .data_hash
        .as_ref()
        .with_context(|| {
            format!(
                "raw tick schema v{} cache {} has no writer data hash; re-download the dataset before replay",
                file.schema_version.unwrap_or_default(),
                path.display()
            )
        })?;
    if expected.algorithm != "fnv1a64" {
        bail!(
            "raw tick schema v{} cache {} uses unsupported hash algorithm {}; re-download the dataset before replay",
            file.schema_version.unwrap_or_default(),
            path.display(),
            expected.algorithm
        );
    }
    let actual = fnv1a64_reader_hex(
        source
            .get_read(0)
            .with_context(|| format!("open {} for position-independent hashing", path.display()))?,
        path,
    )?;
    if actual != expected.value {
        bail!(
            "raw tick schema v{} cache hash mismatch for {}: manifest={} actual={}; re-download the dataset before replay",
            file.schema_version.unwrap_or_default(),
            path.display(),
            expected.value,
            actual
        );
    }
    Ok(())
}

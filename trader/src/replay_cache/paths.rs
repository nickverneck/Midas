use super::*;

pub fn replay_cache_dataset_dir(
    root: &Path,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument: &str,
    contract: &str,
    start_date: NaiveDate,
) -> PathBuf {
    root.join(provider.label().to_ascii_lowercase())
        .join(match env {
            TradingEnvironment::Sim => "sim",
            TradingEnvironment::Live => "live",
        })
        .join(safe_cache_segment(instrument))
        .join(safe_cache_segment(contract))
        .join(start_date.to_string())
}

pub(super) fn replay_cache_write_dataset_dir(
    root: &Path,
    provider: BrokerKind,
    env: TradingEnvironment,
    instrument: &str,
    contract: &ReplayCacheContract,
    start_date: NaiveDate,
    target: Option<&ReplayDownloadCacheTarget>,
) -> Result<PathBuf> {
    let Some(target) = target else {
        fs::create_dir_all(root)
            .with_context(|| format!("create replay cache root {}", root.display()))?;
        let canonical_root = fs::canonicalize(root)
            .with_context(|| format!("resolve replay cache root {}", root.display()))?;
        let dataset_dir = replay_cache_dataset_dir(
            &canonical_root,
            provider,
            env,
            instrument,
            &contract.symbol,
            start_date,
        );
        create_cache_dataset_without_symlinks(&canonical_root, &dataset_dir)?;
        let canonical_dataset = fs::canonicalize(&dataset_dir)
            .with_context(|| format!("resolve replay cache dataset {}", dataset_dir.display()))?;
        if !canonical_dataset.starts_with(&canonical_root) {
            bail!(
                "replay cache dataset {} escapes cache root {}",
                canonical_dataset.display(),
                canonical_root.display()
            );
        }
        return Ok(canonical_dataset);
    };

    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("resolve replay cache root {}", root.display()))?;
    let canonical_dataset = fs::canonicalize(&target.dataset_dir).with_context(|| {
        format!(
            "resolve replay cache target directory {}",
            target.dataset_dir.display()
        )
    })?;
    if !canonical_dataset.starts_with(&canonical_root) {
        bail!(
            "replay cache target {} is outside cache root {}",
            canonical_dataset.display(),
            canonical_root.display()
        );
    }

    let canonical_manifest = fs::canonicalize(&target.manifest_path).with_context(|| {
        format!(
            "resolve replay cache target manifest {}",
            target.manifest_path.display()
        )
    })?;
    let expected_manifest = canonical_dataset.join(MANIFEST_FILE_NAME);
    if canonical_manifest != expected_manifest {
        bail!(
            "replay cache target manifest {} does not identify dataset {}",
            canonical_manifest.display(),
            canonical_dataset.display()
        );
    }

    let manifest = ReplayCacheManifest::from_path(&canonical_manifest)
        .with_context(|| format!("load replay cache target {}", canonical_manifest.display()))?;
    if manifest.provider != provider
        || manifest.env != env
        || !manifest.instrument.symbol.eq_ignore_ascii_case(instrument)
        || !manifest
            .contract
            .symbol
            .eq_ignore_ascii_case(&contract.symbol)
        || (manifest.contract.id.is_some()
            && contract.id.is_some()
            && manifest.contract.id != contract.id)
    {
        bail!(
            "replay cache target identity does not match the requested provider, environment, instrument, and contract"
        );
    }

    Ok(canonical_dataset)
}

pub(super) fn create_cache_dataset_without_symlinks(root: &Path, dataset_dir: &Path) -> Result<()> {
    let relative = dataset_dir.strip_prefix(root).with_context(|| {
        format!(
            "replay cache dataset {} is not under root {}",
            dataset_dir.display(),
            root.display()
        )
    })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(segment) = component else {
            bail!("replay cache dataset contains an unsafe path component");
        };
        current.push(segment);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!(
                    "replay cache dataset path contains symlink {}",
                    current.display()
                );
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => bail!(
                "replay cache dataset component is not a directory: {}",
                current.display()
            ),
            Err(err) if err.kind() == ErrorKind::NotFound => match fs::create_dir(&current) {
                Ok(()) => {}
                Err(create_err) if create_err.kind() == ErrorKind::AlreadyExists => {
                    let metadata = fs::symlink_metadata(&current).with_context(|| {
                        format!("inspect replay cache dataset {}", current.display())
                    })?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        bail!(
                            "replay cache dataset component is unsafe: {}",
                            current.display()
                        );
                    }
                }
                Err(create_err) => {
                    return Err(create_err).with_context(|| {
                        format!("create replay cache dataset {}", current.display())
                    });
                }
            },
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("inspect replay cache dataset {}", current.display())
                });
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn server_bars_relative_path(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> PathBuf {
    PathBuf::from("server-bars").join(format!(
        "{}_to_{}_{}.jsonl",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type_file_label(bar_type)
    ))
}

pub fn server_bars_parquet_relative_path(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    bar_type: BarType,
) -> PathBuf {
    PathBuf::from("server-bars").join(format!(
        "{}_to_{}_{}.parquet",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d"),
        bar_type_file_label(bar_type)
    ))
}

pub fn raw_ticks_relative_path(start: DateTime<Utc>, end: DateTime<Utc>) -> PathBuf {
    PathBuf::from("raw-ticks").join(format!(
        "{}_to_{}_ticks.parquet",
        start.format("%Y-%m-%d"),
        end.format("%Y-%m-%d")
    ))
}

pub(super) fn versioned_cache_relative_path(base: &Path) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("cache");
    let extension = base.extension().and_then(|value| value.to_str());
    let nonce = Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let sequence = CACHE_FILE_VERSION.fetch_add(1, Ordering::Relaxed);
    let file_name = match extension {
        Some(extension) => format!(
            "{stem}_v{nonce}-{}-{sequence}.{extension}",
            std::process::id()
        ),
        None => format!("{stem}_v{nonce}-{}-{sequence}", std::process::id()),
    };
    parent.join(file_name)
}

pub(super) fn remove_superseded_cache_files(dataset_dir: &Path, relative_paths: &[PathBuf]) {
    for relative_path in relative_paths {
        let Ok(data_path) = resolve_cache_data_path(dataset_dir, relative_path) else {
            continue;
        };
        let _ = fs::remove_file(data_path);
    }
}

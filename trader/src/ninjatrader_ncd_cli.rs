//! CSV export for NinjaTrader's local .ncd market-data files.

use crate::cli::ExportNinjaTraderNcdArgs;
use crate::ninjatrader_ncd::{NcdKind, decode_file};
use anyhow::{Context, Result, bail, ensure};
use chrono::{NaiveDate, TimeZone};
use chrono_tz::Tz;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const DOTNET_TICKS_PER_SECOND: i64 = 10_000_000;
pub(crate) fn export_ninjatrader_ncd(args: ExportNinjaTraderNcdArgs) -> Result<()> {
    let timezone: Tz = args
        .timezone
        .parse()
        .with_context(|| format!("invalid IANA timezone `{}`", args.timezone))?;
    let kind = match args.kind.as_str() {
        "minute" => NcdKind::Minute,
        "tick" => NcdKind::Tick,
        _ => bail!("kind must be minute or tick"),
    };
    let mut files = Vec::new();
    collect_ncd_files(&args.input, &mut files)?;
    files.sort();
    ensure!(
        !files.is_empty(),
        "no .ncd files found under {}",
        args.input.display()
    );
    let expected_digits = if kind == NcdKind::Minute { 8 } else { 12 };
    let mut data_type: Option<String> = None;
    for path in &files {
        let current = validate_ncd_name(path, expected_digits)?;
        if let Some(expected) = &data_type {
            ensure!(
                &current == expected,
                "directory contains mixed NCD data types ({expected} and {current})"
            );
        } else {
            data_type = Some(current);
        }
    }
    ensure!(
        !args.output.exists(),
        "output already exists: {}",
        args.output.display()
    );

    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    let temp_path = args.output.with_extension(format!(
        "{}.{}.tmp",
        args.output
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("csv"),
        std::process::id()
    ));
    ensure!(
        !temp_path.exists(),
        "temporary output already exists: {}",
        temp_path.display()
    );
    let mut rows = 0_u64;
    let temp_file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .with_context(|| format!("create temporary CSV {}", temp_path.display()))?;
    let write_result = (|| -> Result<()> {
        let mut writer = BufWriter::new(temp_file);
        writeln!(writer, "timestamp,open,high,low,close,volume,source")?;
        for path in &files {
            let records =
                decode_file(path, kind).with_context(|| format!("decode {}", path.display()))?;
            let source = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            for record in records {
                let timestamp = local_ticks_to_timestamp(record.local_ticks, timezone)
                    .with_context(|| format!("convert timestamp in {}", path.display()))?;
                writeln!(
                    writer,
                    "{timestamp},{},{},{},{},{},{}",
                    record.open,
                    record.high,
                    record.low,
                    record.close,
                    record.volume,
                    csv_field(source)
                )?;
                rows += 1;
            }
        }
        writer.flush()?;
        fs::hard_link(&temp_path, &args.output)
            .with_context(|| format!("commit CSV to {}", args.output.display()))?;
        let _ = fs::remove_file(&temp_path);
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result?;
    println!(
        "Exported {rows} {kind_name} rows from {} file(s) to {}",
        files.len(),
        args.output.display(),
        kind_name = args.kind
    );
    Ok(())
}

fn collect_ncd_files(path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("inspect input {}", path.display()))?;
    if metadata.is_file() {
        ensure!(
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ncd")),
            "input file must have a .ncd extension: {}",
            path.display()
        );
        output.push(path.to_path_buf());
        return Ok(());
    }
    ensure!(
        metadata.is_dir(),
        "input is not a file or directory: {}",
        path.display()
    );
    for entry in fs::read_dir(path).with_context(|| format!("read directory {}", path.display()))? {
        let entry = entry?;
        let child = entry.path();
        if entry.file_type()?.is_file()
            && child
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ncd"))
        {
            output.push(child);
        }
    }
    Ok(())
}

fn validate_ncd_name(path: &Path, digits: usize) -> Result<String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("non-UTF8 NCD filename {}", path.display()))?;
    let lower = name.to_ascii_lowercase();
    let extension = [".last.ncd", ".bid.ncd", ".ask.ncd"]
        .into_iter()
        .find(|suffix| lower.ends_with(suffix))
        .with_context(|| format!("unsupported NCD filename `{name}`; expected timestamp.Last.ncd, .Bid.ncd, or .Ask.ncd"))?;
    let prefix = &name[..name.len() - extension.len()];
    ensure!(
        prefix.len() == digits && prefix.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid NCD filename `{name}`; expected {digits} timestamp digits followed by .Last.ncd, .Bid.ncd, or .Ask.ncd"
    );
    Ok(extension
        .trim_start_matches('.')
        .trim_end_matches(".ncd")
        .to_ascii_lowercase())
}

fn local_ticks_to_timestamp(ticks: i64, timezone: Tz) -> Result<String> {
    ensure!(ticks >= 0, "negative .NET local tick value {ticks}");
    let seconds = ticks / DOTNET_TICKS_PER_SECOND;
    let subsecond_ticks = ticks % DOTNET_TICKS_PER_SECOND;
    let date = NaiveDate::from_ymd_opt(1, 1, 1).context("construct .NET epoch")?;
    let dotnet_epoch = date.and_hms_opt(0, 0, 0).context("construct .NET epoch")?;
    let local_naive = dotnet_epoch
        .checked_add_signed(chrono::Duration::seconds(seconds))
        .and_then(|value| {
            value.checked_add_signed(chrono::Duration::nanoseconds(subsecond_ticks * 100))
        })
        .context(".NET local timestamp is outside chrono's supported range")?;
    let local = match timezone.from_local_datetime(&local_naive) {
        chrono::LocalResult::Single(value) => value,
        chrono::LocalResult::Ambiguous(_, _) => {
            bail!("ambiguous local timestamp {local_naive} in {timezone}")
        }
        chrono::LocalResult::None => {
            bail!("nonexistent local timestamp {local_naive} in {timezone}")
        }
    };
    // Use nanosecond formatting, retaining the decoder's 100ns precision while
    // producing an RFC3339 timestamp with an explicit offset.
    Ok(local.to_rfc3339_opts(chrono::SecondsFormat::Nanos, false))
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, NaiveDateTime};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ncd-cli-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn local_ticks(value: NaiveDateTime) -> i64 {
        let epoch = NaiveDate::from_ymd_opt(1, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let duration = value.signed_duration_since(epoch);
        duration.num_seconds() * DOTNET_TICKS_PER_SECOND + i64::from(duration.subsec_nanos() / 100)
    }

    #[test]
    fn modern_timestamp_keeps_100ns_precision_and_offset() {
        let local = NaiveDate::from_ymd_opt(2026, 11, 14)
            .unwrap()
            .and_hms_nano_opt(12, 34, 56, 123_456_700)
            .unwrap();
        assert_eq!(
            local_ticks_to_timestamp(local_ticks(local), chrono_tz::America::New_York).unwrap(),
            "2026-11-14T12:34:56.123456700-05:00"
        );
    }

    #[test]
    fn rejects_ambiguous_and_nonexistent_local_times() {
        let timezone = chrono_tz::America::New_York;
        let ambiguous = NaiveDate::from_ymd_opt(2024, 11, 3)
            .unwrap()
            .and_hms_opt(1, 30, 0)
            .unwrap();
        let nonexistent = NaiveDate::from_ymd_opt(2024, 3, 10)
            .unwrap()
            .and_hms_opt(2, 30, 0)
            .unwrap();
        assert!(local_ticks_to_timestamp(local_ticks(ambiguous), timezone).is_err());
        assert!(local_ticks_to_timestamp(local_ticks(nonexistent), timezone).is_err());
    }

    #[test]
    fn validates_kind_timestamp_width_and_data_type() {
        assert_eq!(
            validate_ncd_name(Path::new("20261114.Last.ncd"), 8).unwrap(),
            "last"
        );
        assert_eq!(
            validate_ncd_name(Path::new("202611141234.Bid.NCD"), 12).unwrap(),
            "bid"
        );
        assert!(validate_ncd_name(Path::new("20261114.Last.ncd"), 12).is_err());
        assert!(validate_ncd_name(Path::new("20261114.Other.ncd"), 8).is_err());
    }

    #[test]
    fn failed_export_does_not_create_output_or_leave_temp_file() {
        let dir = temp_dir();
        let input = dir.join("20261114.Last.ncd");
        fs::write(&input, b"bad").unwrap();
        let output = dir.join("out.csv");
        let error = export_ninjatrader_ncd(ExportNinjaTraderNcdArgs {
            input: input.clone(),
            output: output.clone(),
            kind: "minute".into(),
            timezone: "UTC".into(),
        });
        assert!(error.is_err());
        assert!(!output.exists());
        assert!(
            !dir.join(format!("out.csv.{}.tmp", std::process::id()))
                .exists()
        );

        fs::write(&output, b"keep").unwrap();
        assert!(
            export_ninjatrader_ncd(ExportNinjaTraderNcdArgs {
                input,
                output: output.clone(),
                kind: "minute".into(),
                timezone: "UTC".into(),
            })
            .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"keep");
        fs::remove_dir_all(dir).unwrap();
    }
}

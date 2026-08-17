use anyhow::{Context, Result, bail};
use csv::StringRecord;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{args::Args, generation::EvaluatedCandidate};

const GA_LOG_HEADER: &str = "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,selection_fitness,eval_net_objective_pnl,eval_gross_realized_pnl,eval_net_realized_pnl_after_costs_and_penalties,eval_total_net_equity_delta,eval_execution_costs,eval_shaping_penalties,eval_terminal_liquidation_cost,eval_sortino,eval_drawdown,eval_ret_mean,train_net_objective_pnl,train_gross_realized_pnl,train_net_realized_pnl_after_costs_and_penalties,train_total_net_equity_delta,train_execution_costs,train_shaping_penalties,train_terminal_liquidation_cost,train_sortino,train_drawdown,train_ret_mean\n";

const LEGACY_GA_LOG_HEADER_COMPLETE: &str = "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,selection_fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n";
const LEGACY_GA_LOG_HEADER_EVAL_ONLY: &str = "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n";
const LEGACY_GA_LOG_HEADER_BASIC: &str = "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GaLogSchema {
    Current,
    LegacyComplete,
    LegacyEvalOnly,
    LegacyBasic,
}

impl GaLogSchema {
    fn field_count(self) -> usize {
        match self {
            Self::Current => 28,
            Self::LegacyComplete => 20,
            Self::LegacyEvalOnly => 19,
            Self::LegacyBasic => 18,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::LegacyComplete => "legacy complete",
            Self::LegacyEvalOnly => "legacy eval-only",
            Self::LegacyBasic => "legacy basic",
        }
    }

    fn optional_field(self, index: usize) -> bool {
        match self {
            Self::Current => {
                matches!(index, 6 | 7 | 19 | 22 | 23 | 24) || (8..=17).contains(&index)
            }
            Self::LegacyComplete => index == 6 || (8..=13).contains(&index),
            Self::LegacyEvalOnly => index == 6 || (7..=12).contains(&index),
            Self::LegacyBasic => (6..=11).contains(&index),
        }
    }
}

fn schema_from_header(header: &StringRecord) -> Result<GaLogSchema> {
    let actual: Vec<&str> = header.iter().collect();
    let expected_current: Vec<&str> = GA_LOG_HEADER.trim_end().split(',').collect();
    let expected_legacy_complete: Vec<&str> = LEGACY_GA_LOG_HEADER_COMPLETE
        .trim_end()
        .split(',')
        .collect();
    let expected_legacy_eval_only: Vec<&str> = LEGACY_GA_LOG_HEADER_EVAL_ONLY
        .trim_end()
        .split(',')
        .collect();
    let expected_legacy_basic: Vec<&str> =
        LEGACY_GA_LOG_HEADER_BASIC.trim_end().split(',').collect();

    if actual == expected_current {
        Ok(GaLogSchema::Current)
    } else if actual == expected_legacy_complete {
        Ok(GaLogSchema::LegacyComplete)
    } else if actual == expected_legacy_eval_only {
        Ok(GaLogSchema::LegacyEvalOnly)
    } else if actual == expected_legacy_basic {
        Ok(GaLogSchema::LegacyBasic)
    } else {
        bail!(
            "ga_log.csv schema mismatch: expected the exact current header or one of the known legacy headers, found {} columns in an unknown or reordered layout; move or delete the incompatible file before resuming",
            actual.len()
        );
    }
}

fn validate_numeric_fields(
    row: &StringRecord,
    schema: GaLogSchema,
    row_number: usize,
) -> Result<()> {
    if row.len() != schema.field_count() {
        bail!(
            "ga_log.csv {} row {} has {} columns; expected {}; refusing to resume",
            schema.name(),
            row_number,
            row.len(),
            schema.field_count()
        );
    }

    for (index, value) in row.iter().enumerate() {
        if value.is_empty() {
            if schema.optional_field(index) {
                continue;
            }
            bail!(
                "ga_log.csv {} row {} has an empty required field at column {}; refusing to resume",
                schema.name(),
                row_number,
                index + 1
            );
        }

        let valid = if index < 2 {
            value.parse::<usize>().is_ok()
        } else {
            value
                .parse::<f64>()
                .map(|number| number.is_finite())
                .unwrap_or(false)
        };
        if !valid {
            bail!(
                "ga_log.csv {} row {} has an invalid numeric value at column {} ({value:?}); refusing to resume",
                schema.name(),
                row_number,
                index + 1
            );
        }
    }
    Ok(())
}

fn normalized_legacy_row(row: &StringRecord, schema: GaLogSchema) -> String {
    let mut normalized = vec![String::new(); GaLogSchema::Current.field_count()];

    for index in 0..6 {
        normalized[index] = row[index].to_owned();
    }

    // These are known aliases from the old logger. The new gross, cost,
    // shaping, terminal, and (where absent) selection fields stay empty
    // because the legacy rows do not contain those measurements.
    match schema {
        GaLogSchema::LegacyComplete => {
            normalized[6] = row[6].to_owned();
            normalized[7] = row[7].to_owned();
            normalized[8] = row[8].to_owned();
            normalized[10] = row[9].to_owned();
            normalized[11] = row[10].to_owned();
            normalized[15] = row[11].to_owned();
            normalized[16] = row[12].to_owned();
            normalized[17] = row[13].to_owned();
            normalized[18] = row[14].to_owned();
            normalized[20] = row[15].to_owned();
            normalized[21] = row[16].to_owned();
            normalized[25] = row[17].to_owned();
            normalized[26] = row[18].to_owned();
            normalized[27] = row[19].to_owned();
        }
        GaLogSchema::LegacyEvalOnly => {
            normalized[6] = row[6].to_owned();
            normalized[8] = row[7].to_owned();
            normalized[10] = row[8].to_owned();
            normalized[11] = row[9].to_owned();
            normalized[15] = row[10].to_owned();
            normalized[16] = row[11].to_owned();
            normalized[17] = row[12].to_owned();
            normalized[18] = row[13].to_owned();
            normalized[20] = row[14].to_owned();
            normalized[21] = row[15].to_owned();
            normalized[25] = row[16].to_owned();
            normalized[26] = row[17].to_owned();
            normalized[27] = row[18].to_owned();
        }
        GaLogSchema::LegacyBasic => {
            normalized[8] = row[6].to_owned();
            normalized[10] = row[7].to_owned();
            normalized[11] = row[8].to_owned();
            normalized[15] = row[9].to_owned();
            normalized[16] = row[10].to_owned();
            normalized[17] = row[11].to_owned();
            normalized[18] = row[12].to_owned();
            normalized[20] = row[13].to_owned();
            normalized[21] = row[14].to_owned();
            normalized[25] = row[15].to_owned();
            normalized[26] = row[16].to_owned();
            normalized[27] = row[17].to_owned();
        }
        GaLogSchema::Current => unreachable!("current rows are not normalized"),
    }

    format!("{}\n", normalized.join(","))
}

fn validate_and_normalize(contents: &str) -> Result<(GaLogSchema, String)> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(contents.as_bytes());
    let mut records = reader.records();
    let header = records
        .next()
        .transpose()
        .context("read ga_log.csv header")?
        .ok_or_else(|| anyhow::anyhow!("ga_log.csv is empty; refusing to resume"))?;
    let schema = schema_from_header(&header)?;
    let mut normalized = String::from(GA_LOG_HEADER);

    for (offset, record) in records.enumerate() {
        let row_number = offset + 2;
        let record = record.with_context(|| format!("read ga_log.csv row {row_number}"))?;
        validate_numeric_fields(&record, schema, row_number)?;
        if schema == GaLogSchema::Current {
            normalized.push_str(&record.iter().collect::<Vec<_>>().join(","));
            normalized.push('\n');
        } else {
            normalized.push_str(&normalized_legacy_row(&record, schema));
        }
    }

    Ok((schema, normalized))
}

fn migrate_legacy_log(log_path: &Path, normalized: &str) -> Result<()> {
    let temp_path = log_path.with_file_name(format!(
        ".{}.ga_log.migrating.{}.tmp",
        log_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("ga_log.csv"),
        std::process::id()
    ));
    let result = (|| -> Result<()> {
        let mut temp = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| format!("create temporary GA log {}", temp_path.display()))?;
        temp.write_all(normalized.as_bytes())
            .context("write normalized GA log")?;
        temp.sync_all().context("flush normalized GA log")?;
        std::fs::rename(&temp_path, log_path).with_context(|| {
            format!(
                "replace legacy GA log {} with its normalized schema",
                log_path.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
fn validate_ga_log_header(header: &str) -> Result<()> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(header.as_bytes());
    let record = reader
        .records()
        .next()
        .transpose()
        .context("read GA log header")?
        .ok_or_else(|| anyhow::anyhow!("GA log header is empty"))?;
    schema_from_header(&record).map(|_| ())
}

pub(crate) struct GaLogState {
    path: PathBuf,
}

pub(crate) fn initialize_ga_log(outdir: &Path) -> Result<GaLogState> {
    let log_path = outdir.join("ga_log.csv");
    if log_path.exists() {
        let meta = std::fs::metadata(&log_path)?;
        if meta.len() == 0 {
            std::fs::write(&log_path, GA_LOG_HEADER)?;
        } else {
            let contents = std::fs::read_to_string(&log_path).with_context(|| {
                format!(
                    "read ga_log.csv at {}; move or delete it before resuming",
                    log_path.display()
                )
            })?;
            let (schema, normalized) = validate_and_normalize(&contents).with_context(|| {
                format!(
                    "incompatible or corrupt ga_log.csv at {}; move or delete it before resuming",
                    log_path.display()
                )
            })?;
            if schema != GaLogSchema::Current {
                migrate_legacy_log(&log_path, &normalized)?;
            }
        }
    } else {
        std::fs::write(&log_path, GA_LOG_HEADER)?;
    }

    Ok(GaLogState { path: log_path })
}

pub(crate) fn append_generation_log(
    log_state: &GaLogState,
    args: &Args,
    generation: usize,
    candidates: &[EvaluatedCandidate],
) -> Result<()> {
    if candidates.is_empty() {
        return Ok(());
    }

    let mut log_buffer = String::new();
    for candidate in candidates {
        let eval_net_objective_pnl = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_pnl))
            .unwrap_or_default();
        let eval_fitness = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.fitness))
            .unwrap_or_default();
        let selection_fitness = format!("{:.4}", candidate.selection_score);
        let eval_gross_realized_pnl = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_gross_realized_pnl))
            .unwrap_or_default();
        let eval_net_realized_pnl = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_pnl_realized))
            .unwrap_or_default();
        let eval_total_net_equity_delta = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_pnl_total))
            .unwrap_or_default();
        let eval_execution_costs = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_execution_costs))
            .unwrap_or_default();
        let eval_shaping_penalties = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_shaping_penalties))
            .unwrap_or_default();
        let eval_terminal_liquidation_cost = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.terminal_liquidation_cost))
            .unwrap_or_default();
        let eval_sortino = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_sortino))
            .unwrap_or_default();
        let eval_drawdown = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_drawdown))
            .unwrap_or_default();
        let eval_ret_mean = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.8}", metrics.eval_ret_mean))
            .unwrap_or_default();

        let train_metrics = &candidate.train_metrics;
        let train_terminal_liquidation_cost =
            format!("{:.4}", train_metrics.terminal_liquidation_cost);
        let line = format!(
            "{generation},{idx},{w_pnl:.4},{w_sortino:.4},{w_mdd:.4},{fitness:.4},{eval_fitness},{selection_fitness},{eval_net_objective_pnl},{eval_gross_realized_pnl},{eval_net_realized_pnl},{eval_total_net_equity_delta},{eval_execution_costs},{eval_shaping_penalties},{eval_terminal_liquidation_cost},{eval_sortino},{eval_drawdown},{eval_ret_mean},{train_net_objective_pnl:.4},{train_gross_realized_pnl:.4},{train_net_realized_pnl:.4},{train_total_net_equity_delta:.4},{train_execution_costs:.4},{train_shaping_penalties:.4},{train_terminal_liquidation_cost},{train_sortino:.4},{train_drawdown:.4},{train_ret_mean:.8}\n",
            generation = generation,
            idx = candidate.idx,
            w_pnl = args.w_pnl,
            w_sortino = args.w_sortino,
            w_mdd = args.w_mdd,
            fitness = train_metrics.fitness,
            eval_fitness = eval_fitness,
            selection_fitness = selection_fitness,
            eval_net_objective_pnl = eval_net_objective_pnl,
            eval_gross_realized_pnl = eval_gross_realized_pnl,
            eval_net_realized_pnl = eval_net_realized_pnl,
            eval_total_net_equity_delta = eval_total_net_equity_delta,
            eval_execution_costs = eval_execution_costs,
            eval_shaping_penalties = eval_shaping_penalties,
            eval_terminal_liquidation_cost = eval_terminal_liquidation_cost,
            eval_sortino = eval_sortino,
            eval_drawdown = eval_drawdown,
            eval_ret_mean = eval_ret_mean,
            train_net_objective_pnl = train_metrics.eval_pnl,
            train_gross_realized_pnl = train_metrics.eval_gross_realized_pnl,
            train_net_realized_pnl = train_metrics.eval_pnl_realized,
            train_total_net_equity_delta = train_metrics.eval_pnl_total,
            train_execution_costs = train_metrics.eval_execution_costs,
            train_shaping_penalties = train_metrics.eval_shaping_penalties,
            train_terminal_liquidation_cost = train_terminal_liquidation_cost,
            train_sortino = train_metrics.eval_sortino,
            train_drawdown = train_metrics.eval_drawdown,
            train_ret_mean = train_metrics.eval_ret_mean
        );
        log_buffer.push_str(&line);
    }

    std::fs::OpenOptions::new()
        .append(true)
        .open(&log_state.path)?
        .write_all(log_buffer.as_bytes())
        .context("write ga_log")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        GA_LOG_HEADER, LEGACY_GA_LOG_HEADER_BASIC, LEGACY_GA_LOG_HEADER_COMPLETE,
        LEGACY_GA_LOG_HEADER_EVAL_ONLY, initialize_ga_log, validate_ga_log_header,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_outdir(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "midas-train-ga-logging-{label}-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create test output directory");
        path
    }

    #[test]
    fn accepts_current_and_known_legacy_headers() {
        validate_ga_log_header(GA_LOG_HEADER).expect("current header is valid");
        validate_ga_log_header(LEGACY_GA_LOG_HEADER_COMPLETE).expect("legacy header is valid");
        validate_ga_log_header(LEGACY_GA_LOG_HEADER_EVAL_ONLY).expect("legacy header is valid");
        validate_ga_log_header(LEGACY_GA_LOG_HEADER_BASIC).expect("legacy header is valid");
    }

    #[test]
    fn migrates_legacy_resume_rows_without_inventing_new_metrics() {
        let outdir = test_outdir("legacy-resume");
        let log_path = outdir.join("ga_log.csv");
        fs::write(
            &log_path,
            format!(
                "{LEGACY_GA_LOG_HEADER_COMPLETE}7,3,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0\n"
            ),
        )
        .expect("write legacy log");

        initialize_ga_log(&outdir).expect("legacy log should resume");
        let normalized = fs::read_to_string(&log_path).expect("read normalized log");
        assert!(normalized.starts_with(GA_LOG_HEADER));
        assert_eq!(
            normalized.lines().nth(1).expect("normalized data row"),
            "7,3,1.0,2.0,3.0,4.0,5.0,6.0,7.0,,8.0,9.0,,,,10.0,11.0,12.0,13.0,,14.0,15.0,,,,16.0,17.0,18.0",
        );
        fs::remove_dir_all(outdir).expect("remove test output directory");
    }

    #[test]
    fn migrates_legacy_headers_without_eval_or_selection_columns() {
        for (label, header, row) in [
            (
                "eval-only",
                LEGACY_GA_LOG_HEADER_EVAL_ONLY,
                "7,3,1.0,2.0,3.0,4.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0",
            ),
            (
                "basic",
                LEGACY_GA_LOG_HEADER_BASIC,
                "7,3,1.0,2.0,3.0,4.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0",
            ),
        ] {
            let outdir = test_outdir(label);
            let log_path = outdir.join("ga_log.csv");
            fs::write(&log_path, format!("{header}{row}\n")).expect("write legacy log");
            initialize_ga_log(&outdir).expect("legacy log should resume");
            initialize_ga_log(&outdir).expect("normalized log should resume again");
            let normalized = fs::read_to_string(&log_path).expect("read normalized log");
            assert!(normalized.starts_with(GA_LOG_HEADER));
            assert_eq!(normalized.lines().count(), 2);
            fs::remove_dir_all(outdir).expect("remove test output directory");
        }
    }

    #[test]
    fn rejects_unknown_reordered_and_corrupt_logs_without_modifying_them() {
        let unknown = GA_LOG_HEADER.replace("eval_net_objective_pnl", "eval_pnl");
        assert!(validate_ga_log_header(&unknown).is_err());

        let mut reordered = GA_LOG_HEADER.trim_end().split(',').collect::<Vec<_>>();
        reordered.swap(0, 1);
        let reordered = format!("{}\n", reordered.join(","));
        assert!(validate_ga_log_header(&reordered).is_err());

        for (label, contents) in [("unknown", unknown), ("reordered", reordered)] {
            let outdir = test_outdir(label);
            let log_path = outdir.join("ga_log.csv");
            fs::write(&log_path, &contents).expect("write incompatible log");
            assert!(initialize_ga_log(&outdir).is_err());
            assert_eq!(
                fs::read_to_string(&log_path).expect("read incompatible log"),
                contents
            );
            fs::remove_dir_all(outdir).expect("remove test output directory");
        }

        let outdir = test_outdir("reject-corrupt");
        let log_path = outdir.join("ga_log.csv");
        let corrupt = format!("{LEGACY_GA_LOG_HEADER_COMPLETE}1,2,3\n");
        fs::write(&log_path, &corrupt).expect("write corrupt log");
        assert!(initialize_ga_log(&outdir).is_err());
        assert_eq!(
            fs::read_to_string(&log_path).expect("read corrupt log"),
            corrupt
        );
        fs::remove_dir_all(outdir).expect("remove test output directory");
    }
}

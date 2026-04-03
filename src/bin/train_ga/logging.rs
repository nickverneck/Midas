use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::{args::Args, generation::EvaluatedCandidate};

const GA_LOG_HEADER: &str = "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_fitness,selection_fitness,eval_fitness_pnl,eval_pnl_realized,eval_pnl_total,eval_sortino,eval_drawdown,eval_ret_mean,train_fitness_pnl,train_pnl_realized,train_pnl_total,train_sortino,train_drawdown,train_ret_mean\n";

pub(crate) struct GaLogState {
    path: PathBuf,
    has_eval_fitness: bool,
    has_selection_fitness: bool,
}

pub(crate) fn initialize_ga_log(outdir: &Path) -> Result<GaLogState> {
    let log_path = outdir.join("ga_log.csv");
    let (has_eval_fitness, has_selection_fitness) = if log_path.exists() {
        let meta = std::fs::metadata(&log_path)?;
        if meta.len() == 0 {
            std::fs::write(&log_path, GA_LOG_HEADER)?;
            (true, true)
        } else {
            let file = std::fs::File::open(&log_path)?;
            let mut reader = BufReader::new(file);
            let mut header = String::new();
            let _ = reader.read_line(&mut header)?;
            let has_eval_fitness = header.split(',').any(|col| col.trim() == "eval_fitness");
            let has_selection_fitness = header
                .split(',')
                .any(|col| col.trim() == "selection_fitness");
            if !has_eval_fitness || !has_selection_fitness {
                println!(
                    "warn: ga_log.csv missing selection columns; delete the log to enable eval/selection fitness tracking"
                );
            }
            (has_eval_fitness, has_selection_fitness)
        }
    } else {
        std::fs::write(&log_path, GA_LOG_HEADER)?;
        (true, true)
    };

    Ok(GaLogState {
        path: log_path,
        has_eval_fitness,
        has_selection_fitness,
    })
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
        let eval_pnl = candidate
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
        let eval_pnl_realized = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_pnl_realized))
            .unwrap_or_default();
        let eval_pnl_total = candidate
            .eval_metrics
            .as_ref()
            .map(|metrics| format!("{:.4}", metrics.eval_pnl_total))
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
        let line = if log_state.has_eval_fitness && log_state.has_selection_fitness {
            format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                generation,
                candidate.idx,
                args.w_pnl,
                args.w_sortino,
                args.w_mdd,
                train_metrics.fitness,
                eval_fitness,
                selection_fitness,
                eval_pnl,
                eval_pnl_realized,
                eval_pnl_total,
                eval_sortino,
                eval_drawdown,
                eval_ret_mean,
                train_metrics.eval_pnl,
                train_metrics.eval_pnl_realized,
                train_metrics.eval_pnl_total,
                train_metrics.eval_sortino,
                train_metrics.eval_drawdown,
                train_metrics.eval_ret_mean
            )
        } else if log_state.has_eval_fitness {
            format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                generation,
                candidate.idx,
                args.w_pnl,
                args.w_sortino,
                args.w_mdd,
                train_metrics.fitness,
                eval_fitness,
                eval_pnl,
                eval_pnl_realized,
                eval_pnl_total,
                eval_sortino,
                eval_drawdown,
                eval_ret_mean,
                train_metrics.eval_pnl,
                train_metrics.eval_pnl_realized,
                train_metrics.eval_pnl_total,
                train_metrics.eval_sortino,
                train_metrics.eval_drawdown,
                train_metrics.eval_ret_mean
            )
        } else {
            format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.8}\n",
                generation,
                candidate.idx,
                args.w_pnl,
                args.w_sortino,
                args.w_mdd,
                train_metrics.fitness,
                eval_pnl,
                eval_pnl_realized,
                eval_pnl_total,
                eval_sortino,
                eval_drawdown,
                eval_ret_mean,
                train_metrics.eval_pnl,
                train_metrics.eval_pnl_realized,
                train_metrics.eval_pnl_total,
                train_metrics.eval_sortino,
                train_metrics.eval_drawdown,
                train_metrics.eval_ret_mean
            )
        };
        log_buffer.push_str(&line);
    }

    std::fs::OpenOptions::new()
        .append(true)
        .open(&log_state.path)?
        .write_all(log_buffer.as_bytes())
        .context("write ga_log")?;
    Ok(())
}

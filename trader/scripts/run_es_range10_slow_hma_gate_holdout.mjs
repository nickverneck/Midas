#!/usr/bin/env node

/**
 * Holdout replay for the fixed slow-HMA gate configurations selected from
 * the earlier ES 10-range Jul 13-Aug 7 study.
 *
 * This intentionally runs fixed candidates rather than choosing a new
 * parameter per holdout week. The new Databento catch-up cache starts on
 * Aug 9, which gives a full seven-day warmup for the Aug 17 and Aug 24
 * evaluation windows.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const cacheRoot = path.join(root, '.run', 'databento-replay-cache-esu6-catchup-20260827');
const templatePath = path.join(
	root,
	'.run',
	'replay-sweeps',
	'hma210-240-protection-20260820',
	'gc-aug10-tpsl',
	'sweep.json'
);
const outputRoot = path.join(root, '.run', 'replay-sweeps', 'es-range10-slow-hma-gate-holdout-20260827');

const option = (name) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
const parallelism = Number(option('parallelism') ?? 4);
const windowFilter = option('window');

if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
for (const required of [binary, cacheConfig, cacheRoot, templatePath]) {
	if (!fs.existsSync(required)) throw new Error(`missing ${required}`);
}

const windows = [
	{
		slug: '2026-08-17',
		label: '2026-08-17 through 2026-08-21',
		evaluationStart: '2026-08-17T00:00:00Z',
		evaluationEnd: '2026-08-21T20:59:59.999Z'
	},
	{
		slug: '2026-08-24',
		label: '2026-08-24 through 2026-08-27 partial',
		evaluationStart: '2026-08-24T00:00:00Z',
		// Keep the requested end inside the imported manifest coverage. The
		// Databento job was requested through 13:38:45Z, but its final trade
		// timestamp is 2026-08-27T13:38:44.831778733Z.
		evaluationEnd: '2026-08-27T13:38:44.831778733Z'
	}
];
const selectedWindows = windowFilter
	? windows.filter((window) => window.slug === windowFilter)
	: windows;
if (!selectedWindows.length) throw new Error(`no window selected for ${windowFilter}`);

const sourceManifest = 'tradovate/sim/ES/ESU6/2026-08-07/manifest.json';
const pairs = {
	'240/480': { slug: 'hma240_480', fast: 240, slow: 480 },
	'300/600': { slug: 'hma300_600', fast: 300, slow: 600 }
};

// These values were fixed before looking at the new Aug 17/Aug 24 data.
const candidates = [
	{
		slug: 'hma240_480_normal',
		pair: '240/480',
		label: 'HMA 240/480 normal control',
		gate: 'none'
	},
	{
		slug: 'hma240_480_decay_h2_m300_d05',
		pair: '240/480',
		label: 'HMA 240/480 decay h2/m300/d0.50',
		gate: 'markov',
		markov: { history: 2, margin: 300, decay: 0.5 }
	},
	{
		slug: 'hma240_480_adx_l3_t30',
		pair: '240/480',
		label: 'HMA 240/480 ADX length3/below30',
		gate: 'adx',
		adx: { length: 3, threshold: 30 }
	},
	{
		slug: 'hma240_480_adx_l8_t20',
		pair: '240/480',
		label: 'HMA 240/480 ADX length8/below20 lower-DD alternative',
		gate: 'adx',
		adx: { length: 8, threshold: 20 }
	},
	{
		slug: 'hma300_600_normal',
		pair: '300/600',
		label: 'HMA 300/600 normal control',
		gate: 'none'
	},
	{
		slug: 'hma300_600_decay_h2_m600_d05',
		pair: '300/600',
		label: 'HMA 300/600 decay h2/m600/d0.50',
		gate: 'markov',
		markov: { history: 2, margin: 600, decay: 0.5 }
	},
	{
		slug: 'hma300_600_adx_l3_t15',
		pair: '300/600',
		label: 'HMA 300/600 ADX length3/below15',
		gate: 'adx',
		adx: { length: 3, threshold: 15 }
	}
];

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function adaptiveAdxGate(length, threshold) {
	return {
		enabled: true,
		combine: 'any',
		minimum_feature_votes: 1,
		invert_confirmation_bars: 1,
		normal_confirmation_bars: 1,
		hold_on_missing_features: false,
		hold_during_dwell: false,
		reset_on_missing_features: true,
		use_relative_volume: false,
		volume_lookback_bars: 30,
		invert_below_relative_volume: 0,
		invert_above_relative_volume: null,
		use_atr_ratio: false,
		atr_length: 14,
		atr_lookback_bars: 30,
		invert_below_atr_ratio: 0,
		invert_above_atr_ratio: null,
		use_choppiness: false,
		choppiness_length: 30,
		invert_below_choppiness: 0,
		invert_above_choppiness: null,
		use_directional_return: false,
		directional_return_length: 60,
		invert_when_aligned_return_below: null,
		invert_when_aligned_return_above: null,
		use_adx: true,
		adx_length: length,
		invert_below_adx: threshold,
		invert_above_adx: null,
		use_di_imbalance: false,
		invert_when_di_imbalance_below: 0.1,
		invert_when_di_imbalance_above: null,
		use_ema_spread: false,
		invert_when_normalized_spread_below: 0,
		invert_when_normalized_spread_above: null,
		use_ema_slope: false,
		ema_slope_lookback: 3,
		invert_when_normalized_slope_below: 0,
		invert_when_normalized_slope_above: null,
		use_directional_ema_gap: false,
		directional_ema_length: 120,
		directional_gap_lookback_bars: 1,
		invert_when_bullish_gap_below: null,
		invert_when_bullish_gap_above: null,
		invert_when_bearish_gap_below: null,
		invert_when_bearish_gap_above: null,
		use_session_window: false,
		session_start_minute_et: 0,
		session_end_minute_et: 0,
		invert_inside_session_window: true,
		session_weekdays_mask: 31,
		use_higher_timeframe_context: false,
		higher_timeframe_average_kind: 'ema',
		higher_timeframe_minutes: 15,
		higher_timeframe_fast_length: 8,
		higher_timeframe_slow_length: 21,
		higher_timeframe_atr_length: 14,
		higher_timeframe_slope_lookback: 3,
		higher_timeframe_persistence_bars: 2,
		higher_timeframe_spread_atr_floor: 0,
		higher_timeframe_spread_atr_ceiling: null,
		higher_timeframe_slope_atr_floor: 0,
		higher_timeframe_conflict_action: 'invert',
		neutral_action: 'normal_fallback'
	};
}

function setCommon(spec, window, candidate, pair, outputDir) {
	spec.schema_version = 1;
	spec.sweep_id = `esu6_range10_${candidate.slug}_${window.slug}_holdout_20260827`;
	spec.name = `ESU6 10-range ${candidate.label} — ${window.label}`;
	spec.notes = 'Exact-contract ESU6 Databento catch-up trades; fixed candidate selected from the prior Jul 13-Aug 7 study; causal 10-range bars; closed-bar raw-bar-open execution; TP 120 ticks; no stop; ES fee included.';
	spec.base_dataset_view = {
		view_version: 1,
		id: `esu6_${window.slug}_${candidate.slug}_holdout_20260827`,
		source: {
			manifest_id: sourceManifest,
			provider: 'tradovate',
			env: 'sim',
			instrument: 'ES',
			contract: 'ESU6'
		},
		evaluation_start: window.evaluationStart,
		evaluation_end: window.evaluationEnd,
		input_timezone: 'UTC',
		session_preset: 'full_source',
		warmup: { duration_seconds: 604800, trading: 'flat_until_evaluation' }
	};
	spec.bar_type = { kind: 'range', value: 10 };
	spec.candle_mode = 'standard';
	spec.fill_model = 'raw_bar_open';
	spec.engine_mode = 'deterministic';
	spec.execution_mode = 'prepared_cpu';
	spec.evaluator_mode = 'streaming';
	spec.parallelism = parallelism;
	spec.output_formats = ['json_summary'];
	spec.output_dir = outputDir;
	spec.primary_fee_schedule = {
		name: 'esu6_tradovate_2026-08-01_per_side',
		currency: 'USD',
		commission_per_contract: 2.88,
		exchange_per_contract: 0,
		clearing_per_contract: 0,
		regulatory_per_contract: 0,
		misc_per_contract: 0
	};
	spec.fee_scenarios = [];
	spec.initial_capital = 100000;
	spec.parameters = [];
	spec.max_runs = 1;
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: 1,
		max_parallel_jobs: parallelism,
		max_cache_read_rows_per_worker: 10000000,
		large_sweep_threshold: 100,
		max_estimated_memory_bytes: 8589934592,
		max_estimated_output_bytes: 1073741824
	};
	spec.base_strategy.native_signal_timing = 'ClosedBar';
	spec.base_strategy.native_signal_delay_bars = 0;
	spec.base_strategy.native_execution_path = 'Guarded';
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.blockout_enabled = true;
	spec.base_strategy.blockout_minutes_before_close = 45;
	spec.base_strategy.order_qty = 1;
	spec.base_strategy.native_hma_cross = {
		...spec.base_strategy.native_hma_cross,
		fast_length: pair.fast,
		slow_length: pair.slow,
		calculation_mode: 'incremental',
		inverted: false,
		take_profit_ticks: 120,
		stop_loss_ticks: 0,
		use_trailing_stop: false,
		trail_trigger_ticks: 40,
		trail_offset_ticks: 10
	};
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
}

function makeSpec(window, candidate) {
	const spec = clone(JSON.parse(fs.readFileSync(templatePath, 'utf8')));
	const pair = pairs[candidate.pair];
	const outputDir = path.join(outputRoot, window.slug, candidate.slug);
	setCommon(spec, window, candidate, pair, outputDir);

	if (candidate.gate === 'markov') {
		spec.base_strategy.native_strategy = 'HmaCross';
		spec.replay_markov_orientation_gate = {
		...spec.replay_markov_orientation_gate,
		enabled: true,
		score_window_outcomes: candidate.markov.history,
		minimum_completed_outcomes: 1,
		score_margin_ticks: candidate.markov.margin,
		max_abs_outcome_ticks: 1200,
		outcome_decay: candidate.markov.decay,
		confirmation_events: 1,
		minimum_dwell_events: 0,
		reset_after_gap_minutes: 120,
		efficiency_lookback_bars: null,
		neutral_below_efficiency_ratio: null,
		normal_override_above_efficiency_ratio: null,
		neutral_action: 'normal_fallback'
		};
	} else if (candidate.gate === 'adx') {
		spec.base_strategy.native_strategy = 'VolumeAdaptiveHmaCross';
		spec.base_strategy.native_volume_hma_cross = {
			...spec.base_strategy.native_volume_hma_cross,
			fast_length: pair.fast,
			slow_length: pair.slow,
			calculation_mode: 'incremental',
			inverted: false,
			take_profit_ticks: 120,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 40,
			trail_offset_ticks: 10,
			volume_regime: {
				...(spec.base_strategy.native_volume_hma_cross.volume_regime ?? {}),
				lookback_bars: 30,
				invert_below_relative_volume: 0
			},
			ema_gate: {
				...(spec.base_strategy.native_volume_hma_cross.ema_gate ?? {}),
				enabled: false
			},
			adaptive_gate: adaptiveAdxGate(candidate.adx.length, candidate.adx.threshold)
		};
		spec.replay_markov_orientation_gate.enabled = false;
	} else {
		spec.base_strategy.native_strategy = 'HmaCross';
	}
	return spec;
}

function runSpec(spec) {
	const specPath = path.join(outputRoot, 'specs', `${spec.sweep_id}.json`);
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${spec.name} ===`);
	const result = spawnSync(
		binary,
		['--config', cacheConfig, 'run-replay-sweep', '--spec', specPath, '--no-resume'],
		{
			cwd: root,
			env: { ...process.env, TRADER_DATA_CACHE_DIR: cacheRoot },
			encoding: 'utf8',
			stdio: 'inherit'
		}
	);
	if (result.status !== 0) throw new Error(`holdout replay failed with status ${result.status}`);
	const outputDir = path.isAbsolute(spec.output_dir) ? spec.output_dir : path.join(root, spec.output_dir);
	const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'sweep-summary.json'), 'utf8'));
	const plan = JSON.parse(fs.readFileSync(path.join(outputDir, 'sweep-plan.json'), 'utf8'));
	const parameters = plan.children[0]?.parameter_values ?? {};
	const run = summary.runs[0];
	if (!run) throw new Error(`holdout replay produced no run for ${spec.sweep_id}`);
	if (run.result_path) {
		const resultPath = path.isAbsolute(run.result_path) ? run.result_path : path.join(root, run.result_path);
		fs.rmSync(resultPath, { force: true });
	}
	return {
		window: spec.base_dataset_view.evaluation_start.slice(0, 10),
		candidate: spec.sweep_id,
		status: run.status,
		gross_pnl: run.gross_pnl,
		net_pnl: run.net_pnl,
		fees: run.fees,
		trade_count: run.trade_count,
		fill_count: run.fill_count,
		max_drawdown: run.max_drawdown,
		execution_backend: run.execution_backend,
		parameters
	};
}

const rows = [];
for (const window of selectedWindows) {
	for (const candidate of candidates) rows.push({ ...runSpec(makeSpec(window, candidate)), candidate_slug: candidate.slug, gate: candidate.gate, pair: candidate.pair, label: candidate.label });
}

const report = {
	schema_version: 'es-range10-slow-hma-gate-holdout-v1',
	generated_at_utc: new Date().toISOString(),
	contract: 'ESU6',
	bar_type: '10-range standard OHLC',
	fee_per_contract_side: 2.88,
	cache: cacheRoot,
	selection_period: '2026-07-13 through 2026-08-07',
	holdout_windows: selectedWindows,
	method: 'Fixed configurations selected before the holdout; no parameter re-selection on Aug 17 or Aug 24 data.',
	rows
};
fs.mkdirSync(outputRoot, { recursive: true });
fs.writeFileSync(path.join(outputRoot, 'rows.json'), `${JSON.stringify(rows, null, 2)}\n`);
fs.writeFileSync(path.join(outputRoot, 'summary.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ rows: rows.length, output: path.relative(root, path.join(outputRoot, 'summary.json')) }, null, 2));

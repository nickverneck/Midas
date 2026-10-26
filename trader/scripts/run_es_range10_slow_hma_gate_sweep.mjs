#!/usr/bin/env node

/**
 * ESU6 10-range slow-HMA orientation-gate experiment.
 *
 * The two fixed HMA triggers are the recent four-week protection candidates:
 *   - HMA 240/480, TP 120 ticks, no stop
 *   - HMA 300/600, TP 120 ticks, no stop
 *
 * Each candidate is tested with one causal orientation selector at a time:
 *   - markov: prior completed-cross outcomes with history/margin/decay sweep
 *   - adx: ADX-below-threshold inversion with length/threshold sweep
 *
 * The normal fixed controls are already recorded in the companion protection
 * sweep.  This runner emits compact gate rows and removes per-run result
 * ledgers after extracting them, so the experiment does not consume the disk
 * with duplicate replay output.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const templatePath = path.join(
	root,
	'.run',
	'replay-sweeps',
	'hma210-240-protection-20260820',
	'gc-aug10-tpsl',
	'sweep.json'
);
const outputRoot = path.join(root, '.run', 'replay-sweeps', 'es-range10-slow-hma-gates-20260827');

const option = (name) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
const gateFilter = option('gate') ?? 'all';
const parallelism = Number(option('parallelism') ?? 4);
const windowFilter = option('window');

if (!['markov', 'adx', 'all'].includes(gateFilter)) {
	throw new Error(`unsupported --gate=${gateFilter}; expected markov, adx, or all`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
for (const required of [binary, cacheConfig, templatePath]) {
	if (!fs.existsSync(required)) throw new Error(`missing ${required}`);
}

const pairs = [
	{ slug: 'hma240_480', fast: 240, slow: 480, label: 'HMA 240/480' },
	{ slug: 'hma300_600', fast: 300, slow: 600, label: 'HMA 300/600' }
];

// These are exactly the recent4 windows used by the ES protection comparison.
const windows = [
	{
		slug: '2026-07-13',
		label: '2026-07-13 through 2026-07-17',
		evaluationStart: '2026-07-13T00:00:00Z',
		evaluationEnd: '2026-07-17T20:59:59.840Z'
	},
	{
		slug: '2026-07-20',
		label: '2026-07-20 through 2026-07-24',
		evaluationStart: '2026-07-20T00:00:00Z',
		evaluationEnd: '2026-07-24T20:59:59.840Z'
	},
	{
		slug: '2026-07-27',
		label: '2026-07-27 through 2026-07-31',
		evaluationStart: '2026-07-27T00:00:00Z',
		evaluationEnd: '2026-07-31T20:59:59.840Z'
	},
	{
		slug: '2026-08-03',
		label: '2026-08-03 through 2026-08-07',
		evaluationStart: '2026-08-03T00:00:00Z',
		evaluationEnd: '2026-08-07T20:59:59.840Z'
	}
];

const selectedWindows = windowFilter
	? windows.filter((window) => window.slug === windowFilter)
	: windows;
if (!selectedWindows.length) throw new Error(`no window selected for ${windowFilter}`);

const sourceManifest = 'tradovate/sim/ES/ESU6/2026-02-09/manifest.json';
const markovGrid = {
	history: [2, 3, 6, 8, 12],
	margin: [300, 600],
	decay: [0.5, 0.75, 1]
};
const adxGrid = {
	length: [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28],
	threshold: [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
};

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function adaptiveAdxGate(length = 14, threshold = 45) {
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

function datasetView(window, gate, pair) {
	return {
		view_version: 1,
		id: `esu6_${window.slug}_${pair.slug}_${gate}_range10_20260827`,
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
		warmup: {
			duration_seconds: 604800,
			trading: 'flat_until_evaluation'
		}
	};
}

function commonSpec(window, gate, pair, outputDir) {
	const template = JSON.parse(fs.readFileSync(templatePath, 'utf8'));
	const spec = clone(template);
	spec.schema_version = 1;
	spec.sweep_id = `esu6_range10_${pair.slug}_${gate}_${window.slug}_20260827`;
	spec.name = `ESU6 10-range ${pair.label} ${gate} orientation gate — ${window.label}`;
	spec.notes = 'Exact-contract ESU6 Databento trades; causal 10-range bars; closed-bar raw-bar-open execution; TP 120 ticks; no stop; ES fee included; gate chooses normal or inverted orientation from information available at the crossover.';
	spec.base_dataset_view = datasetView(window, gate, pair);
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
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: spec.max_runs,
		max_parallel_jobs: parallelism,
		max_cache_read_rows_per_worker: 20000000,
		large_sweep_threshold: 100,
		max_estimated_memory_bytes: 8589934592,
		max_estimated_output_bytes: 137438953472
	};
	return spec;
}

function setHmaProtection(config, pair) {
	return {
		...config,
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
}

function makeMarkovSpec(window, pair) {
	const outputDir = path.join(outputRoot, 'markov', pair.slug, window.slug);
	const spec = commonSpec(window, 'markov', pair, outputDir);
	spec.base_strategy.native_strategy = 'HmaCross';
	spec.base_strategy.native_signal_timing = 'ClosedBar';
	spec.base_strategy.native_signal_delay_bars = 0;
	spec.base_strategy.native_execution_path = 'Guarded';
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.blockout_enabled = true;
	spec.base_strategy.blockout_minutes_before_close = 45;
	spec.base_strategy.order_qty = 1;
	spec.base_strategy.native_hma_cross = setHmaProtection(spec.base_strategy.native_hma_cross, pair);
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: true,
		minimum_completed_outcomes: 1,
		max_abs_outcome_ticks: 1200,
		confirmation_events: 1,
		minimum_dwell_events: 0,
		reset_after_gap_minutes: 120,
		efficiency_lookback_bars: null,
		neutral_below_efficiency_ratio: null,
		normal_override_above_efficiency_ratio: null,
		neutral_action: 'normal_fallback'
	};
	spec.parameters = [
		{ path: 'replay_markov_orientation_gate.score_window_outcomes', values: markovGrid.history },
		{ path: 'replay_markov_orientation_gate.score_margin_ticks', values: markovGrid.margin },
		{ path: 'replay_markov_orientation_gate.outcome_decay', values: markovGrid.decay }
	];
	spec.max_runs = markovGrid.history.length * markovGrid.margin.length * markovGrid.decay.length;
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function makeAdxSpec(window, pair) {
	const outputDir = path.join(outputRoot, 'adx', pair.slug, window.slug);
	const spec = commonSpec(window, 'adx', pair, outputDir);
	spec.base_strategy.native_strategy = 'VolumeAdaptiveHmaCross';
	spec.base_strategy.native_signal_timing = 'ClosedBar';
	spec.base_strategy.native_signal_delay_bars = 0;
	spec.base_strategy.native_execution_path = 'Guarded';
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.blockout_enabled = true;
	spec.base_strategy.blockout_minutes_before_close = 45;
	spec.base_strategy.order_qty = 1;
	spec.base_strategy.native_volume_hma_cross = {
		...spec.base_strategy.native_volume_hma_cross,
		...setHmaProtection({}, pair),
		volume_regime: {
			...(spec.base_strategy.native_volume_hma_cross.volume_regime ?? {}),
			lookback_bars: 30,
			invert_below_relative_volume: 0
		},
		ema_gate: {
			...(spec.base_strategy.native_volume_hma_cross.ema_gate ?? {}),
			enabled: false
		},
		adaptive_gate: adaptiveAdxGate()
	};
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
	spec.parameters = [
		{ path: 'native_volume_hma_cross.adaptive_gate.adx_length', values: adxGrid.length },
		{ path: 'native_volume_hma_cross.adaptive_gate.invert_below_adx', values: adxGrid.threshold }
	];
	spec.max_runs = adxGrid.length.length * adxGrid.threshold.length;
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function compactRows(spec, summary, plan) {
	const parametersByRun = new Map(plan.children.map((child) => [child.run_id, child.parameter_values ?? {}]));
	const rows = summary.runs.map((run) => ({
		run_id: run.run_id,
		status: run.status,
		gross_pnl: run.gross_pnl,
		net_pnl: run.net_pnl,
		fees: run.fees,
		trade_count: run.trade_count,
		fill_count: run.fill_count,
		max_drawdown: run.max_drawdown,
		execution_backend: run.execution_backend,
		parameters: parametersByRun.get(run.run_id) ?? {}
	}));
	for (const run of summary.runs) {
		if (!run.result_path) continue;
		const resultPath = path.isAbsolute(run.result_path) ? run.result_path : path.join(root, run.result_path);
		fs.rmSync(resultPath, { force: true });
	}
	return rows;
}

function runSpec(spec, label) {
	const specPath = path.join(outputRoot, 'specs', `${label}-${spec.sweep_id}.json`);
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${label}: ${spec.sweep_id} (${spec.max_runs} runs) ===`);
	const result = spawnSync(
		binary,
		['--config', cacheConfig, 'run-replay-sweep', '--spec', specPath, '--no-resume', '--allow-large'],
		{
			cwd: root,
			env: { ...process.env, TRADER_DATA_CACHE_DIR: path.join(root, '.run', 'databento-replay-cache') },
			encoding: 'utf8',
			stdio: 'inherit'
		}
	);
	if (result.status !== 0) throw new Error(`${label} failed with status ${result.status}`);
	const outputDir = path.isAbsolute(spec.output_dir) ? spec.output_dir : path.join(root, spec.output_dir);
	const summaryPath = path.join(outputDir, 'sweep-summary.json');
	const planPath = path.join(outputDir, 'sweep-plan.json');
	return compactRows(spec, JSON.parse(fs.readFileSync(summaryPath, 'utf8')), JSON.parse(fs.readFileSync(planPath, 'utf8')));
}

const rows = [];
for (const window of selectedWindows) {
	for (const pair of pairs) {
		if (gateFilter === 'markov' || gateFilter === 'all') {
			for (const row of runSpec(makeMarkovSpec(window, pair), `markov-${pair.slug}-${window.slug}`)) {
				rows.push({ gate: 'markov', window: window.slug, pair: `${pair.fast}/${pair.slow}`, ...row });
			}
		}
		if (gateFilter === 'adx' || gateFilter === 'all') {
			for (const row of runSpec(makeAdxSpec(window, pair), `adx-${pair.slug}-${window.slug}`)) {
				rows.push({ gate: 'adx', window: window.slug, pair: `${pair.fast}/${pair.slow}`, ...row });
			}
		}
	}
}

const report = {
	schema_version: 'es-range10-slow-hma-gates-v1',
	generated_at_utc: new Date().toISOString(),
	contract: 'ESU6',
	bar_type: '10-range standard OHLC',
	fee_per_contract_side: 2.88,
	protection: 'TP 120 ticks, SL 0, trailing off',
	causal: {
		markov: 'At each HMA cross, settle only the previous cross using its completed close-to-close signed return; no current-cross future outcome is used.',
		adx: 'At each HMA cross, invert only when the completed-bar ADX is below the candidate threshold; no future bars are used.'
	},
	windows: selectedWindows,
	pairs,
	markov_grid: markovGrid,
	adx_grid: adxGrid,
	rows
};
fs.mkdirSync(outputRoot, { recursive: true });
fs.writeFileSync(path.join(outputRoot, 'rows.json'), `${JSON.stringify(rows, null, 2)}\n`);
fs.writeFileSync(path.join(outputRoot, 'summary.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
	gate: gateFilter,
	windows: selectedWindows.map(({ slug }) => slug),
	pairs: pairs.map(({ fast, slow }) => `${fast}/${slow}`),
	rows: rows.length,
	output: path.relative(root, path.join(outputRoot, 'summary.json'))
}, null, 2));

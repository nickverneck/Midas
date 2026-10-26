#!/usr/bin/env node

/**
 * GCZ6 10-range EMA 420/480 ADX orientation-gate sweep.
 *
 * The gate is evaluated on completed range bars at EMA crossover decisions.
 * It inverts when ADX is below the swept threshold; all other adaptive-gate
 * features and the volume/secondary-EMA gates are disabled.  Both fixed
 * trailing profiles found in the preceding experiment are tested:
 *   - normal-base: trigger 80 / offset 5
 *   - inverted-base: trigger 40 / offset 5
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const phase = process.argv.find((arg) => arg.startsWith('--phase='))?.split('=')[1] ?? 'all';
const parallelism = Number(process.argv.find((arg) => arg.startsWith('--parallelism='))?.split('=')[1] ?? 4);

if (!['controls', 'adx', 'all'].includes(phase)) {
	throw new Error(`unsupported --phase=${phase}; expected controls, adx, or all`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --release --features replay --bin trader`);
}

const sourceManifest = 'tradovate/sim/GC/GCZ6/2025-08-19/manifest.json';
const outputRoot = '.run/replay-sweeps/gc-range10-ema420-480-adx-gate-20260827';
const templatePath = '.run/replay-sweeps/hma210-240-protection-20260820/gc-aug10-tpsl/sweep.json';

const windows = [
	{
		slug: 'jul27',
		label: '2026-07-27 through 2026-08-02',
		evaluationStart: '2026-07-27T00:00:00Z',
		evaluationEnd: '2026-08-02T23:59:59.999Z'
	},
	{
		slug: 'aug03',
		label: '2026-08-03 through 2026-08-07',
		evaluationStart: '2026-08-03T00:00:00Z',
		evaluationEnd: '2026-08-07T20:59:55.632000437Z'
	},
	{
		slug: 'aug10',
		label: '2026-08-10 through 2026-08-14',
		evaluationStart: '2026-08-10T00:00:00Z',
		evaluationEnd: '2026-08-14T20:59:00Z'
	}
];

const profiles = [
	{
		slug: 'normal_base',
		label: 'normal-base trailing 80/5',
		inverted: false,
		trailTrigger: 80,
		trailOffset: 5
	},
	{
		slug: 'inverted_base',
		label: 'inverted-base trailing 40/5',
		inverted: true,
		trailTrigger: 40,
		trailOffset: 5
	}
];

const adxLengths = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28];
const adxThresholds = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70];

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function adaptiveAdxGate(enabled, length = 14, threshold = 45) {
	return {
		enabled,
		combine: 'any',
		minimum_feature_votes: 1,
		invert_confirmation_bars: 1,
		normal_confirmation_bars: 1,
		hold_on_missing_features: false,
		hold_during_dwell: false,
		reset_on_missing_features: true,
		use_relative_volume: false,
		volume_lookback_bars: 30,
		invert_below_relative_volume: 0.0,
		invert_above_relative_volume: null,
		use_atr_ratio: false,
		atr_length: 14,
		atr_lookback_bars: 30,
		invert_below_atr_ratio: 0.0,
		invert_above_atr_ratio: null,
		use_choppiness: false,
		choppiness_length: 30,
		invert_below_choppiness: 0.0,
		invert_above_choppiness: null,
		use_directional_return: false,
		directional_return_length: 60,
		invert_when_aligned_return_below: null,
		invert_when_aligned_return_above: null,
		use_adx: enabled,
		adx_length: length,
		invert_below_adx: threshold,
		invert_above_adx: null,
		use_di_imbalance: false,
		invert_when_di_imbalance_below: 0.1,
		invert_when_di_imbalance_above: null,
		use_ema_spread: false,
		invert_when_normalized_spread_below: 0.0,
		invert_when_normalized_spread_above: null,
		use_ema_slope: false,
		ema_slope_lookback: 3,
		invert_when_normalized_slope_below: 0.0,
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
		higher_timeframe_spread_atr_floor: 0.0,
		higher_timeframe_spread_atr_ceiling: null,
		higher_timeframe_slope_atr_floor: 0.0,
		higher_timeframe_conflict_action: 'invert',
		neutral_action: 'normal_fallback'
	};
}

function baseSpec(window, profile, id, name, outputDir) {
	const spec = clone(JSON.parse(fs.readFileSync(path.join(root, templatePath), 'utf8')));
	spec.schema_version = 1;
	spec.sweep_id = id;
	spec.name = name;
	spec.notes = 'Exact-contract GCZ6 Databento trades; causal range10 bars; EMA fast=420 slow=480; fixed trailing profile; ADX-only orientation gate; fees included.';
	spec.base_dataset_view = {
		view_version: 1,
		id: `gcz6_${window.slug}_range10_ohlc_ema420_480_adx_gate_20260827`,
		source: {
			manifest_id: sourceManifest,
			provider: 'tradovate',
			env: 'sim',
			instrument: 'GC',
			contract: 'GCZ6'
		},
		evaluation_start: window.evaluationStart,
		evaluation_end: window.evaluationEnd,
		input_timezone: 'UTC',
		session_preset: 'full_source',
		warmup: { duration_seconds: 172800, trading: 'flat_until_evaluation' }
	};
	spec.bar_type = { kind: 'range', value: 10 };
	spec.candle_mode = 'standard';
	spec.fill_model = 'raw_bar_open';
	spec.execution_mode = 'prepared_cpu';
	spec.evaluator_mode = 'streaming';
	spec.parallelism = parallelism;
	spec.output_formats = ['json_summary'];
	spec.output_dir = outputDir;
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: spec.max_runs,
		max_parallel_jobs: parallelism
	};
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.native_strategy = 'VolumeAdaptiveEmaCross';
	spec.base_strategy.native_volume_ema_cross = {
		...spec.base_strategy.native_volume_ema_cross,
		fast_length: 420,
		slow_length: 480,
		inverted: profile.inverted,
		take_profit_ticks: 0,
		stop_loss_ticks: 0,
		use_trailing_stop: true,
		trail_trigger_ticks: profile.trailTrigger,
		trail_offset_ticks: profile.trailOffset,
		volume_regime: {
			...spec.base_strategy.native_volume_ema_cross.volume_regime,
			lookback_bars: 30,
			invert_below_relative_volume: 0.0
		},
		ema_gate: {
		...spec.base_strategy.native_volume_ema_cross.ema_gate,
			enabled: false
		},
		adaptive_gate: adaptiveAdxGate(false)
	};
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
	return spec;
}

function controlsSpec(window, profile) {
	const id = `gcz6_ema420_480_range10_adx_controls_${profile.slug}_${window.slug}_20260827`;
	const spec = baseSpec(
		window,
		profile,
		id,
		`GCZ6 10-range EMA 420/480 ${profile.label} control — ${window.label}`,
		`${outputRoot}/controls/${profile.slug}/${window.slug}`
	);
	spec.parameters = [];
	spec.max_runs = 1;
	spec.guardrails.max_combinations = 1;
	return spec;
}

function adxSpec(window, profile) {
	const id = `gcz6_ema420_480_range10_adx_${profile.slug}_${window.slug}_20260827`;
	const spec = baseSpec(
		window,
		profile,
		id,
		`GCZ6 10-range EMA 420/480 ${profile.label} ADX gate sweep — ${window.label}`,
		`${outputRoot}/adx/${profile.slug}/${window.slug}`
	);
	spec.base_strategy.native_volume_ema_cross.adaptive_gate = adaptiveAdxGate(true);
	spec.parameters = [
		{ path: 'native_volume_ema_cross.adaptive_gate.adx_length', values: adxLengths },
		{ path: 'native_volume_ema_cross.adaptive_gate.invert_below_adx', values: adxThresholds }
	];
	spec.max_runs = adxLengths.length * adxThresholds.length;
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function runSpec(spec, label) {
	const specPath = path.join(root, outputRoot, `${label}-${spec.base_dataset_view.id}.json`);
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${label} ${spec.base_dataset_view.id}: ${spec.max_runs} runs ===`);
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
	if (result.status !== 0) {
		throw new Error(`${label} sweep failed for ${spec.base_dataset_view.id} with status ${result.status}`);
	}
	const summaryPath = path.join(root, spec.output_dir, 'sweep-summary.json');
	const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
	for (const run of summary.runs) {
		if (!run.result_path) continue;
		const resultPath = path.isAbsolute(run.result_path) ? run.result_path : path.join(root, run.result_path);
		fs.rmSync(resultPath, { force: true });
	}
	return summary;
}

for (const window of windows) {
	for (const profile of profiles) {
		if (phase === 'controls' || phase === 'all') runSpec(controlsSpec(window, profile), 'control');
		if (phase === 'adx' || phase === 'all') runSpec(adxSpec(window, profile), 'adx');
	}
}

console.log(JSON.stringify({
	schema_version: 'gcz6-ema420-480-range10-adx-gate-v1',
	phase,
	windows: windows.map(({ slug }) => slug),
	profiles: profiles.map(({ slug }) => slug),
	contract: 'GCZ6',
	bar_type: '10-range standard OHLC',
	warmup_seconds: 172800,
	gate: 'invert when completed-bar ADX is below threshold',
	adx_lengths: adxLengths,
	adx_thresholds: adxThresholds,
	runs_per_profile_window: { controls: 1, adx: adxLengths.length * adxThresholds.length }
}, null, 2));

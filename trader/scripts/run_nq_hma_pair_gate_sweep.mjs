#!/usr/bin/env node

/**
 * NQ 10-range HMA-210 pair follow-up.
 *
 * This runner keeps the two orientation selectors separate:
 *   - markov: prior completed-cross outcomes with an outcome-decay sweep
 *   - adx: causal ADX-only adaptive gate with the previously explored upper
 *     threshold region
 *
 * The fixed TP60/SL0 reference is used for both selectors.  Fixed normal,
 * fixed inverted, no-protection, and exact-$200 (TP40) controls are recorded
 * by the companion HMA-pair sweep and are not reselected here.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const gateFilter = process.argv.find((arg) => arg.startsWith('--gate='))?.split('=')[1] ?? 'all';
const parallelism = Number(process.argv.find((arg) => arg.startsWith('--parallelism='))?.split('=')[1] ?? 4);

if (!['markov', 'adx', 'all'].includes(gateFilter)) {
	throw new Error(`unsupported --gate=${gateFilter}; expected markov, adx, or all`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --release --features replay --bin trader`);
}

const windows = [
	{
		slug: 'current',
		template: '.run/replay-sweeps/nq-hma210-pairs-range10-databento-20260827/base2/current/sweep.json',
		start: '2026-08-23T22:00:00Z',
		end: '2026-08-26T13:24:59.961621275Z',
		manifest: 'tradovate/sim/NQ/NQU6/2026-08-23/manifest.json'
	},
	{
		slug: 'last',
		template: '.run/replay-sweeps/nq-hma210-pairs-range10-databento-20260827/base2/last/sweep.json',
		start: '2026-08-16T22:00:00Z',
		end: '2026-08-21T20:59:59.682552825Z',
		manifest: 'tradovate/sim/NQ/NQU6/2026-08-16/manifest.json'
	},
	{
		slug: 'previous',
		template: '.run/replay-sweeps/nq-hma210-pairs-range10-databento-20260827/base2/previous/sweep.json',
		start: '2026-08-09T22:00:00Z',
		end: '2026-08-14T20:59:58.819532871Z',
		manifest: 'tradovate/sim/NQ/NQU6/2026-08-09/manifest.json'
	}
];

const markovGrid = {
	history: [2, 3, 6, 8, 12],
	margin: [300, 600],
	decay: [0.5, 0.75, 1]
};
const adxGrid = {
	length: [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 28],
	threshold: [45, 48, 50, 52, 55, 60, 65, 70]
};

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function baseMetadata(spec, window, id, name, outputDir) {
	spec.schema_version = 1;
	spec.sweep_id = id;
	spec.name = name;
	spec.notes = `${spec.notes ?? ''}; NQ NQU6 10-range Databento trades; HMA fast=210; TP60/SL0; causal gate only; fixed controls in hma-pair base2 sweep.`;
	spec.base_dataset_view = {
		view_version: 1,
		id: `nqu6_${window.slug}_range10_hma210_pair_gate_20260827`,
		source: {
			manifest_id: window.manifest,
			provider: 'tradovate',
			env: 'sim',
			instrument: 'NQ',
			contract: 'NQU6'
		},
		evaluation_start: window.start,
		evaluation_end: window.end,
		input_timezone: 'UTC',
		session_preset: 'full_source',
		warmup: { duration_seconds: 0, trading: 'flat_until_evaluation' }
	};
	spec.parallelism = parallelism;
	spec.execution_mode = 'prepared_cpu';
	spec.evaluator_mode = 'streaming';
	spec.output_formats = ['json_summary'];
	spec.output_dir = outputDir;
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: spec.max_runs,
		max_parallel_jobs: parallelism
	};
	return spec;
}

function makeMarkovSpec(window) {
	const template = JSON.parse(fs.readFileSync(path.join(root, window.template), 'utf8'));
	const spec = clone(template);
	spec.base_strategy.native_strategy = 'HmaCross';
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.native_hma_cross = {
		...spec.base_strategy.native_hma_cross,
		fast_length: 210,
		slow_length: 240,
		calculation_mode: 'incremental',
		inverted: false,
		take_profit_ticks: 60,
		stop_loss_ticks: 0,
		use_trailing_stop: false,
		trail_trigger_ticks: 10,
		trail_offset_ticks: 5
	};
	spec.parameters = [
		{ path: 'native_hma_cross.slow_length', values: [280, 480] },
		{ path: 'replay_markov_orientation_gate.score_window_outcomes', values: markovGrid.history },
		{ path: 'replay_markov_orientation_gate.score_margin_ticks', values: markovGrid.margin },
		{ path: 'replay_markov_orientation_gate.outcome_decay', values: markovGrid.decay }
	];
	spec.max_runs = 2 * markovGrid.history.length * markovGrid.margin.length * markovGrid.decay.length;
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
	return baseMetadata(
		spec,
		window,
		`nqu6_hma210_pair_markov_${window.slug}_20260827`,
		`NQ ${window.slug} HMA 210/280-or-480 causal decay gate`,
		`.run/replay-sweeps/nq-hma210-pairs-range10-gates-20260827/markov/${window.slug}`
	);
}

function makeAdxSpec(window) {
	const templatePath = '.run/replay-sweeps/nq-hma210-240-range10-adx-gate-databento-20260827-upper/current/sweep.json';
	const spec = JSON.parse(fs.readFileSync(path.join(root, templatePath), 'utf8'));
	spec.base_strategy.native_strategy = 'VolumeAdaptiveHmaCross';
	spec.base_strategy.native_volume_hma_cross = {
		...spec.base_strategy.native_volume_hma_cross,
		fast_length: 210,
		slow_length: 240,
		calculation_mode: 'incremental',
		inverted: false,
		take_profit_ticks: 60,
		stop_loss_ticks: 0,
		use_trailing_stop: false,
		trail_trigger_ticks: 10,
		trail_offset_ticks: 5,
		volume_regime: { ...(spec.base_strategy.native_volume_hma_cross.volume_regime ?? {}), lookback_bars: 30, invert_below_relative_volume: 0 },
		ema_gate: { ...(spec.base_strategy.native_volume_hma_cross.ema_gate ?? {}), enabled: false },
		adaptive_gate: {
			...spec.base_strategy.native_volume_hma_cross.adaptive_gate,
			enabled: true,
			combine: 'any',
			minimum_feature_votes: 1,
			invert_confirmation_bars: 1,
			normal_confirmation_bars: 1,
			hold_on_missing_features: false,
			hold_during_dwell: false,
			reset_on_missing_features: true,
			use_relative_volume: false,
			use_atr_ratio: false,
			use_choppiness: false,
			use_directional_return: false,
			use_adx: true,
			adx_length: 14,
			invert_below_adx: 45,
			invert_above_adx: null,
			use_di_imbalance: false,
			use_ema_spread: false,
			use_ema_slope: false,
			use_directional_ema_gap: false,
			use_session_window: false,
			use_higher_timeframe_context: false
		}
	};
	spec.parameters = [
		{ path: 'native_volume_hma_cross.slow_length', values: [280, 480] },
		{ path: 'native_volume_hma_cross.adaptive_gate.adx_length', values: adxGrid.length },
		{ path: 'native_volume_hma_cross.adaptive_gate.invert_below_adx', values: adxGrid.threshold }
	];
	spec.max_runs = 2 * adxGrid.length.length * adxGrid.threshold.length;
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
	return baseMetadata(
		spec,
		window,
		`nqu6_hma210_pair_adx_${window.slug}_20260827`,
		`NQ ${window.slug} HMA 210/280-or-480 ADX gate`,
		`.run/replay-sweeps/nq-hma210-pairs-range10-gates-20260827/adx/${window.slug}`
	);
}

function runSpec(spec, label) {
	const specPath = path.join(root, '.run', 'replay-sweeps', 'nq-hma210-pairs-range10-gates-20260827', `${label}-${spec.base_dataset_view.id}.json`);
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${label} ${spec.base_dataset_view.id}: ${spec.max_runs} runs ===`);
	const result = spawnSync(
		binary,
		[
			'--config',
			cacheConfig,
			'run-replay-sweep',
			'--spec',
			specPath,
			'--no-resume',
			'--allow-large'
		],
		{ cwd: root, env: { ...process.env, TRADER_DATA_CACHE_DIR: path.join(root, '.run', 'databento-replay-cache') }, encoding: 'utf8', stdio: 'inherit' }
	);
	if (result.status !== 0) {
		throw new Error(`${label} sweep failed for ${spec.base_dataset_view.id} with status ${result.status}`);
	}
}

for (const window of windows) {
	if (gateFilter === 'markov' || gateFilter === 'all') runSpec(makeMarkovSpec(window), 'markov');
	if (gateFilter === 'adx' || gateFilter === 'all') runSpec(makeAdxSpec(window), 'adx');
}

console.log(JSON.stringify({
	schema_version: 'nq-hma-pair-gate-sweep-v1',
	gate: gateFilter,
	windows: windows.map(({ slug }) => slug),
	markov_runs_per_window: 2 * markovGrid.history.length * markovGrid.margin.length * markovGrid.decay.length,
	adx_runs_per_window: 2 * adxGrid.length.length * adxGrid.threshold.length
}, null, 2));

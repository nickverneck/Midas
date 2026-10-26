#!/usr/bin/env node

/**
 * GCZ6 one-minute EMA 420/480 protection and orientation experiment.
 *
 * Controls are run before protection sweeps, and every protection candidate
 * includes both normal and inverted orientations.  The replay kernel is
 * causal and uses the exact-contract Databento trade tapes already cached in
 * the repository.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const phase = process.argv.find((arg) => arg.startsWith('--phase='))?.split('=')[1] ?? 'all';
const parallelism = Number(process.argv.find((arg) => arg.startsWith('--parallelism='))?.split('=')[1] ?? 4);

if (!['controls', 'bracket', 'trail', 'all'].includes(phase)) {
	throw new Error(`unsupported --phase=${phase}; expected controls, bracket, trail, or all`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --release --features replay --bin trader`);
}

const windows = [
	{
		slug: 'jul27',
		label: '2026-07-27 through 2026-08-02',
		manifest: 'tradovate/sim/GC/GCZ6/2026-07-27/manifest.json',
		start: '2026-07-27T00:00:00.168550921Z',
		end: '2026-08-02T23:59:41.813002085Z'
	},
	{
		slug: 'aug03',
		label: '2026-08-03 through 2026-08-07',
		manifest: 'tradovate/sim/GC/GCZ6/2026-08-03/manifest.json',
		start: '2026-08-03T00:00:00.354760733Z',
		end: '2026-08-07T20:59:55.632000437Z'
	},
	{
		slug: 'aug10',
		label: '2026-08-10 through 2026-08-14',
		manifest: 'tradovate/sim/GC/GCZ6/2026-08-10/manifest.json',
		start: '2026-08-10T00:00:00.048986137Z',
		end: '2026-08-14T20:59:00Z'
	}
];

const templatePath = '.run/replay-sweeps/hma210-240-protection-20260820/gc-aug10-tpsl/sweep.json';
const outputRoot = '.run/replay-sweeps/gc-ema420-480-20260827';

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function baseSpec(window, id, name, outputDir, protection) {
	const spec = JSON.parse(fs.readFileSync(path.join(root, templatePath), 'utf8'));
	spec.schema_version = 1;
	spec.sweep_id = id;
	spec.name = name;
	spec.notes = 'Exact-contract GCZ6 Databento trades; one-minute standard OHLC derived causally; EMA fast=420 slow=480; normal and inverted controls; TP/SL values are ticks; one contract.';
	spec.base_dataset_view = {
		view_version: 1,
		id: `gcz6_${window.slug}_1m_ohlc_ema420_480_20260827`,
		source: {
			manifest_id: window.manifest,
			provider: 'tradovate',
			env: 'sim',
			instrument: 'GC',
			contract: 'GCZ6'
		},
		evaluation_start: window.start,
		evaluation_end: window.end,
		input_timezone: 'UTC',
		session_preset: 'full_source',
		warmup: { duration_seconds: 0, trading: 'flat_until_evaluation' }
	};
	spec.bar_type = { kind: 'minute', value: 1 };
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
	spec.base_strategy.native_strategy = 'EmaCross';
	spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
	spec.base_strategy.native_ema = {
		...spec.base_strategy.native_ema,
		fast_length: 420,
		slow_length: 480,
		inverted: false,
		take_profit_ticks: protection.take_profit_ticks,
		stop_loss_ticks: protection.stop_loss_ticks,
		use_trailing_stop: protection.use_trailing_stop,
		trail_trigger_ticks: protection.trail_trigger_ticks,
		trail_offset_ticks: protection.trail_offset_ticks
	};
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
	return spec;
}

function controlsSpec(window, kind, protection) {
	const id = `gcz6_ema420_480_${kind}_${window.slug}_20260827`;
	const spec = baseSpec(
		window,
		id,
		`GCZ6 EMA 420/480 ${kind} — ${window.label}`,
		`${outputRoot}/${kind}/${window.slug}`,
		protection
	);
	spec.parameters = [
		{ path: 'native_ema.inverted', values: [false, true] }
	];
	spec.max_runs = 2;
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function bracketSpec(window) {
	const spec = baseSpec(
		window,
		`gcz6_ema420_480_bracket_${window.slug}_20260827`,
		`GCZ6 EMA 420/480 bracket protection sweep — ${window.label}`,
		`${outputRoot}/bracket/${window.slug}`,
		{ take_profit_ticks: 80, stop_loss_ticks: 70, use_trailing_stop: false, trail_trigger_ticks: 40, trail_offset_ticks: 10 }
	);
	spec.parameters = [
		{ path: 'native_ema.inverted', values: [false, true] },
		{ path: 'native_ema.take_profit_ticks', values: [0, 40, 60, 80, 100, 120] },
		{ path: 'native_ema.stop_loss_ticks', values: [0, 35, 50, 70, 90, 120] }
	];
	spec.max_runs = 2 * 6 * 6;
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function trailSpec(window) {
	const spec = baseSpec(
		window,
		`gcz6_ema420_480_trail_${window.slug}_20260827`,
		`GCZ6 EMA 420/480 trailing protection sweep — ${window.label}`,
		`${outputRoot}/trail/${window.slug}`,
		{ take_profit_ticks: 80, stop_loss_ticks: 70, use_trailing_stop: true, trail_trigger_ticks: 40, trail_offset_ticks: 10 }
	);
	spec.parameters = [
		{ path: 'native_ema.inverted', values: [false, true] },
		{ path: 'native_ema.take_profit_ticks', values: [0, 80] },
		{ path: 'native_ema.stop_loss_ticks', values: [0, 70] },
		{ path: 'native_ema.trail_trigger_ticks', values: [20, 40, 60, 80] },
		{ path: 'native_ema.trail_offset_ticks', values: [5, 10, 20, 30] }
	];
	spec.max_runs = 2 * 2 * 2 * 4 * 4;
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
}

for (const window of windows) {
	if (phase === 'controls' || phase === 'all') {
		runSpec(controlsSpec(window, 'baseline', { take_profit_ticks: 0, stop_loss_ticks: 0, use_trailing_stop: false, trail_trigger_ticks: 40, trail_offset_ticks: 10 }), 'baseline');
		runSpec(controlsSpec(window, 'fixed_protection', { take_profit_ticks: 80, stop_loss_ticks: 70, use_trailing_stop: false, trail_trigger_ticks: 40, trail_offset_ticks: 10 }), 'fixed');
	}
	if (phase === 'bracket' || phase === 'all') runSpec(bracketSpec(window), 'bracket');
	if (phase === 'trail' || phase === 'all') runSpec(trailSpec(window), 'trail');
}

console.log(JSON.stringify({
	schema_version: 'gcz6-ema420-480-experiment-v1',
	phase,
	windows: windows.map(({ slug }) => slug),
		contract: 'GCZ6',
		bar_type: '1-minute standard OHLC',
		tick_conversion: { tick_size: 0.1, dollars_per_tick: 10, tp_800_usd_ticks: 80, sl_700_usd_ticks: 70 },
		counts: { baseline: 2, fixed: 2, bracket: 72, trail: 128 }
}, null, 2));

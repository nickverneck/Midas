#!/usr/bin/env node

/**
 * Compare fixed-orientation EMA/HMA controls with the causal historical-cross
 * gate on exact-contract GCZ6 Databento trades.
 *
 * The range bars are derived by the replay engine from the imported trade
 * tape.  The gate parameters are frozen research candidates; this runner does
 * not select a gate from the result of an individual window.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const option = (name) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
const phase = option('phase') ?? 'all';
const parallelism = Number(option('parallelism') ?? 2);
const tag = option('tag') ?? '20260821';
const windowFilter = option('window');
const barFilter = option('bar');
const triggerFilter = option('trigger');
const orientationFilter = option('orientation');
const executionMode = option('execution-mode') ?? 'prepared_cpu';
const gateFilter = option('gate') ?? 'n2-m300-d1';
const protectionGateFilter = option('protection-gates') ?? 'selected';
const outputRoot = path.join(root, '.run', 'replay-sweeps', `range-gate-${tag}`);

if (!['baseline', 'gate', 'protection', 'all'].includes(phase)) {
	throw new Error(`unsupported --phase=${phase}; expected baseline, gate, protection, or all`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 4) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected 1..4`);
}
if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --features replay --bin trader`);
}

const windows = [
	{
		slug: 'jul27',
		manifestId: 'tradovate/sim/GC/GCZ6/2025-08-19/manifest.json',
		evaluationStart: '2026-07-27T00:00:00.000Z',
		evaluationEnd: '2026-08-02T23:59:59.999Z'
	},
	{
		slug: 'aug03',
		manifestId: 'tradovate/sim/GC/GCZ6/2025-08-19/manifest.json',
		evaluationStart: '2026-08-03T00:00:00.000Z',
		evaluationEnd: '2026-08-07T20:59:30.157Z'
	},
	{
		slug: 'aug10',
		manifestId: 'tradovate/sim/GC/GCZ6/2025-08-19/manifest.json',
		evaluationStart: '2026-08-10T00:00:00.000Z',
		evaluationEnd: '2026-08-14T20:59:00.000Z'
	}
];

const bars = [
	{ slug: 'range10', value: 10 },
	{ slug: 'range1', value: 1 }
];

const triggers = [
	{ slug: 'ema1030', strategy: 'EmaCross', path: 'native_ema.inverted', fast: 10, slow: 30 },
	{ slug: 'hma210240', strategy: 'HmaCross', path: 'native_hma_cross.inverted', fast: 210, slow: 240 },
	{ slug: 'hma1030', strategy: 'HmaCross', path: 'native_hma_cross.inverted', fast: 10, slow: 30 },
	{ slug: 'hma330', strategy: 'HmaCross', path: 'native_hma_cross.inverted', fast: 3, slow: 30 }
];

const gateCandidates = [
	{ slug: 'n2-m300-d1', history: 2, margin: 300, decay: 1 },
	{ slug: 'n6-m600-d05', history: 6, margin: 600, decay: 0.5 },
	{ slug: 'n12-m600-d05', history: 12, margin: 600, decay: 0.5 }
];
const selectedGate = gateCandidates.find((candidate) => candidate.slug === gateFilter);
if (!selectedGate) {
	throw new Error(`unsupported --gate=${gateFilter}; expected one of ${gateCandidates.map((candidate) => candidate.slug).join(', ')}`);
}

const protectionGates = protectionGateFilter === 'all' ? gateCandidates : [selectedGate];
if (protectionGateFilter !== 'all' && protectionGateFilter !== 'selected') {
	throw new Error(`unsupported --protection-gates=${protectionGateFilter}; expected selected or all`);
}

const selectedWindows = windows.filter((window) => !windowFilter || window.slug === windowFilter);
const selectedBars = bars.filter((bar) => !barFilter || bar.slug === barFilter);
const triggerFilters = triggerFilter?.split(',').map((value) => value.trim()).filter(Boolean) ?? [];
const selectedTriggers = triggers.filter((trigger) => !triggerFilters.length || triggerFilters.includes(trigger.slug));
if (!selectedWindows.length || !selectedBars.length || !selectedTriggers.length) {
	throw new Error(`filter selected no cases: window=${windowFilter ?? '*'} bar=${barFilter ?? '*'} trigger=${triggerFilter ?? '*'}`);
}
const baselineOrientations = orientationFilter === 'normal'
	? [false]
	: orientationFilter === 'inverted'
		? [true]
		: [false, true];
if (!['prepared_cpu', 'batch_cpu'].includes(executionMode)) {
	throw new Error(`unsupported --execution-mode=${executionMode}; expected prepared_cpu or batch_cpu`);
}
if (phase !== 'baseline' && executionMode !== 'prepared_cpu') {
	throw new Error('the historical gate requires --execution-mode=prepared_cpu');
}

function strategy(trigger, inverted = false) {
	return {
		kind: 'Native',
		native_strategy: trigger.strategy,
		native_signal_timing: 'ClosedBar',
		native_signal_delay_bars: 0,
		native_execution_path: 'Guarded',
		native_reversal_mode: 'CloseAllEnter',
		blockout_enabled: true,
		blockout_minutes_before_close: 45,
		native_hma: {
			hma_length: 255,
			min_angle: 7,
			angle_lookback: 7,
			bars_required_to_trade: 50,
			longs_only: false,
			inverted: false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 12,
			trail_offset_ticks: 8
		},
		native_ema: {
			fast_length: 10,
			slow_length: 30,
			inverted: trigger.strategy === 'EmaCross' ? inverted : false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 12,
			trail_offset_ticks: 8
		},
		native_hma_cross: {
			fast_length: trigger.strategy === 'HmaCross' ? trigger.fast : 210,
			slow_length: trigger.strategy === 'HmaCross' ? trigger.slow : 240,
			calculation_mode: 'incremental',
			inverted: trigger.strategy === 'HmaCross' ? inverted : false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 10,
			trail_offset_ticks: 5
		},
		native_volume_hma_cross: {
			fast_length: 210,
			slow_length: 240,
			calculation_mode: 'incremental',
			inverted: false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 10,
			trail_offset_ticks: 5,
			volume_regime: { lookback_bars: 30, invert_below_relative_volume: 0.35 },
			ema_gate: { enabled: false, ema_length: 500, invert_when_above: true }
		},
		native_volume_ema_cross: {
			fast_length: 21,
			slow_length: 55,
			inverted: false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 12,
			trail_offset_ticks: 8,
			volume_regime: { lookback_bars: 30, invert_below_relative_volume: 0.6 },
			ema_gate: { enabled: false, ema_length: 500, invert_when_above: true }
		},
		native_adx: {
			adx_length: 14,
			adx_entry_threshold: 25,
			adx_exit_threshold: 20,
			di_imbalance_threshold: 0.1,
			slope_lookback: 3,
			dominance_bars: 2,
			breakout_lookback: 20,
			inverted: false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false,
			trail_trigger_ticks: 12,
			trail_offset_ticks: 8
		},
		order_qty: 1
	};
}

function gateConfig(candidate) {
	return {
		enabled: true,
		score_window_outcomes: candidate.history,
		minimum_completed_outcomes: 1,
		score_margin_ticks: candidate.margin,
		max_abs_outcome_ticks: 1200,
		outcome_decay: candidate.decay,
		confirmation_events: 1,
		minimum_dwell_events: 0,
		reset_after_gap_minutes: 120,
		efficiency_lookback_bars: null,
		neutral_below_efficiency_ratio: null,
		normal_override_above_efficiency_ratio: null,
		neutral_action: 'normal_fallback'
	};
}

const bracketProtectionGrid = {
	slug: 'bracket',
	name: 'TP/SL bracket',
	paths: (trigger) => {
		const prefix = trigger.strategy === 'EmaCross' ? 'native_ema' : 'native_hma_cross';
		return [
			{ path: `${prefix}.take_profit_ticks`, values: [0, 20, 40, 60] },
			{ path: `${prefix}.stop_loss_ticks`, values: [0, 20, 40] },
			{ path: `${prefix}.use_trailing_stop`, values: [false] }
		];
	}
};

const trailingProtectionGrid = {
	slug: 'trailing',
	name: 'trailing stop',
	paths: (trigger) => {
		const prefix = trigger.strategy === 'EmaCross' ? 'native_ema' : 'native_hma_cross';
		return [
			{ path: `${prefix}.take_profit_ticks`, values: [0] },
			{ path: `${prefix}.stop_loss_ticks`, values: [20] },
			{ path: `${prefix}.use_trailing_stop`, values: [true] },
			{ path: `${prefix}.trail_trigger_ticks`, values: [20, 40, 80] },
			{ path: `${prefix}.trail_offset_ticks`, values: [5, 10, 20] }
		];
	}
};
const protectionGrids = [bracketProtectionGrid, trailingProtectionGrid];

function baseSpec(window, bar, trigger, outputDir) {
	return {
		schema_version: 1,
		sweep_id: `range_gate_${window.slug}_${bar.slug}_${trigger.slug}`,
		name: `GCZ6 ${bar.slug} ${trigger.slug} baseline/gate research`,
		notes: 'Exact-contract Databento trades; range bars are derived causally by the replay engine; no protection; fixed gate candidates are not selected per window.',
		base_dataset_view: {
			view_version: 1,
			id: `gcz6_databento_${window.slug}_${bar.slug}_view`,
			source: {
				manifest_id: window.manifestId,
				provider: 'tradovate',
				env: 'sim',
				instrument: 'GC',
				contract: 'GCZ6'
			},
			evaluation_start: window.evaluationStart,
			evaluation_end: window.evaluationEnd,
			input_timezone: 'UTC',
			session_preset: 'full_source',
			warmup: { duration_seconds: 0, trading: 'flat_until_evaluation' }
		},
		bar_type: { kind: 'range', value: bar.value },
		candle_mode: 'standard',
		base_strategy: strategy(trigger),
		parameters: [],
		constraints: [],
		engine_mode: 'deterministic',
		evaluator_mode: 'streaming',
		fill_model: 'raw_bar_open',
		latency: { model: 'fixed', fixed_latency_ms: 0, observed_samples_ms: [], seed: 1 },
		bar_protection_policy: 'conservative',
		primary_fee_schedule: {
			name: 'gcz6_tradovate_2026-08-01_per_side',
			currency: 'USD',
			commission_per_contract: 1.29,
			exchange_per_contract: 1.6,
			clearing_per_contract: 0.19,
			regulatory_per_contract: 0.02,
			misc_per_contract: 0
		},
		fee_scenarios: [],
		initial_capital: 100000,
		margin: null,
		max_runs: 1,
		parallelism,
		execution_mode: executionMode,
		guardrails: {
			max_combinations: 2,
			max_parallel_jobs: parallelism,
			max_cache_read_rows_per_worker: 20000000,
			large_sweep_threshold: 100,
			max_estimated_memory_bytes: 8589934592,
			max_estimated_output_bytes: 68719476736
		},
		output_dir: outputDir,
		output_formats: ['json_summary']
	};
}

function protectionSpec(window, bar, trigger, gate, grid, outputDir) {
	const spec = baseSpec(window, bar, trigger, outputDir);
	spec.sweep_id = `range_gate_protection_${window.slug}_${bar.slug}_${trigger.slug}_${grid.slug}`;
	spec.sweep_id += `_${gate.slug}`;
	spec.name = `GCZ6 ${bar.slug} ${trigger.slug} ${grid.name} with ${gate.slug} historical gate`;
	spec.notes = 'Exact-contract Databento trades; causal historical-cross gate frozen from the prior study; focused protection grid; fees included.';
	spec.base_strategy = strategy(trigger);
	spec.replay_markov_orientation_gate = gateConfig(gate);
	spec.parameters = grid.paths(trigger);
	spec.max_runs = spec.parameters.reduce((count, parameter) => count * parameter.values.length, 1);
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

function runSpec(spec, specPath) {
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	const result = spawnSync(
		binary,
		['--config', cacheConfig, 'run-replay-sweep', '--spec', specPath, '--no-resume'],
		{
			cwd: root,
			env: { ...process.env, TRADER_DATA_CACHE_DIR: path.join(root, '.run', 'databento-replay-cache') },
			encoding: 'utf8',
			stdio: 'inherit'
		}
	);
	if (result.status !== 0) {
		const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`.trim();
		throw new Error(`replay failed for ${specPath}:\n${output.slice(-6000)}`);
	}
	const outputDir = path.join(root, spec.output_dir);
	const summary = JSON.parse(fs.readFileSync(path.join(outputDir, 'sweep-summary.json'), 'utf8'));
	const plan = JSON.parse(fs.readFileSync(path.join(outputDir, 'sweep-plan.json'), 'utf8'));
	const params = new Map(plan.children.map((child) => [child.run_id, child.parameter_values ?? {}]));
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
		parameters: params.get(run.run_id) ?? {}
	}));
	// The research report only needs the compact sweep summary.  The replay
	// engine always writes a detailed per-bar result.json, which can be hundreds
	// of MB for one-tick bars.  Remove those artifacts after extracting the
	// summary so a large matrix does not consume the workstation disk.
	for (const run of summary.runs) {
		if (!run.result_path) continue;
		const resultPath = path.isAbsolute(run.result_path)
			? run.result_path
			: path.join(root, run.result_path);
		fs.rmSync(resultPath, { force: true });
	}
	return rows;
}

const rows = [];
for (const window of selectedWindows) {
	for (const bar of selectedBars) {
		for (const trigger of selectedTriggers) {
			const common = path.join(outputRoot, window.slug, bar.slug, trigger.slug);
			if (phase === 'baseline' || phase === 'all') {
				const spec = baseSpec(window, bar, trigger, path.relative(root, path.join(common, 'baseline')));
				spec.sweep_id += '_baseline';
				spec.name += ' baseline orientation sweep';
				spec.base_strategy = strategy(trigger);
				spec.parameters = [{ path: trigger.path, values: baselineOrientations }];
				spec.max_runs = baselineOrientations.length;
				spec.guardrails.max_combinations = baselineOrientations.length;
				for (const row of runSpec(spec, path.join(root, spec.output_dir, 'spec.json'))) {
					rows.push({ phase: 'baseline', window: window.slug, bar: bar.slug, trigger: trigger.slug, gate: 'none', ...row });
				}
			}
			if (phase === 'gate' || phase === 'all') {
				for (const candidate of gateCandidates) {
					const spec = baseSpec(window, bar, trigger, path.relative(root, path.join(common, `gate-${candidate.slug}`)));
					spec.sweep_id += `_gate_${candidate.slug}`;
					spec.name += ` historical gate ${candidate.slug}`;
					spec.replay_markov_orientation_gate = gateConfig(candidate);
					for (const row of runSpec(spec, path.join(root, spec.output_dir, 'spec.json'))) {
						rows.push({ phase: 'gate', window: window.slug, bar: bar.slug, trigger: trigger.slug, gate: candidate.slug, ...row });
					}
				}
			}
			if (phase === 'protection' || phase === 'all') {
				for (const gate of protectionGates) {
					for (const grid of protectionGrids) {
						const spec = protectionSpec(window, bar, trigger, gate, grid, path.relative(root, path.join(common, `protection-${grid.slug}-${gate.slug}`)));
						for (const row of runSpec(spec, path.join(root, spec.output_dir, 'spec.json'))) {
							rows.push({ phase: 'protection', window: window.slug, bar: bar.slug, trigger: trigger.slug, gate: gate.slug, protection: grid.slug, ...row });
						}
					}
				}
			}
		}
	}
}

const report = {
	schema_version: 'range-gate-experiment-v1',
	generated_at_utc: new Date().toISOString(),
	cache_config: path.relative(root, cacheConfig),
	definition: 'Baseline compares normal/inverted fixed orientation. Gate candidates use only outcomes from prior completed crosses and default to normal fallback until evidence exists.',
	windows: selectedWindows,
	bars: selectedBars,
		triggers: selectedTriggers,
		gate_candidates: gateCandidates,
		selected_gate: selectedGate,
		protection_gates: protectionGates,
		protection_grids: protectionGrids.map((grid) => ({ slug: grid.slug, name: grid.name })),
	rows
};
fs.mkdirSync(outputRoot, { recursive: true });
fs.writeFileSync(path.join(outputRoot, 'summary.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ phase, rows: rows.length, output: path.relative(root, path.join(outputRoot, 'summary.json')) }, null, 2));

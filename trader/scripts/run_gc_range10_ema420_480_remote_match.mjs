#!/usr/bin/env node

/**
 * Match the current remote GCZ6 10-range EMA 420/480 strategy in replay.
 *
 * The baseline controls use the live configuration: closed-bar signals,
 * one-bar delay, guarded execution, Flatten > Confirm > Enter, and the
 * 45-minute session blockout.  The protection variants are deliberately
 * kept separate from the orientation/gate experiments so the comparisons
 * remain interpretable.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const option = (name, fallback) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split('=').slice(1).join('=') ?? fallback;
const phase = option('phase', 'baseline');
const parallelism = Number(option('parallelism', '4'));
const protectionFilter = option('protection', 'all');
const signalDelayBars = Number(option('delay', '1'));
const reversalMode = option('reversal', 'FlattenConfirmEnter');

if (!['baseline', 'history', 'adx'].includes(phase)) {
	throw new Error(`unsupported --phase=${phase}; expected baseline, history, or adx`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected 1..8`);
}
if (!Number.isInteger(signalDelayBars) || signalDelayBars < 0 || signalDelayBars > 1) {
	throw new Error(`unsupported --delay=${signalDelayBars}; expected 0 or 1`);
}
if (!['FlattenConfirmEnter', 'CloseAllEnter'].includes(reversalMode)) {
	throw new Error(`unsupported --reversal=${reversalMode}; expected FlattenConfirmEnter or CloseAllEnter`);
}
if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --release --features replay --bin trader`);
}

const windows = [
	{
		slug: 'jul27',
		label: '2026-07-27 through 2026-08-02',
		evaluationStart: '2026-07-27T00:00:00.000Z',
		evaluationEnd: '2026-08-02T23:59:59.999Z'
	},
	{
		slug: 'aug03',
		label: '2026-08-03 through 2026-08-07',
		evaluationStart: '2026-08-03T00:00:00.000Z',
		evaluationEnd: '2026-08-07T20:59:55.632000437Z'
	},
	{
		slug: 'aug10',
		label: '2026-08-10 through 2026-08-14',
		evaluationStart: '2026-08-10T00:00:00.000Z',
		evaluationEnd: '2026-08-14T20:59:00.000Z'
	}
];

const protections = [
	{
		slug: 'tp90_sl1000',
		label: 'matched remote TP90 / SL1000',
		take_profit_ticks: 90,
		stop_loss_ticks: 1000,
		use_trailing_stop: false,
		trail_trigger_ticks: 80,
		trail_offset_ticks: 5
	},
	{
		slug: 'no_tp_sl1000',
		label: 'no TP / SL1000',
		take_profit_ticks: 0,
		stop_loss_ticks: 1000,
		use_trailing_stop: false,
		trail_trigger_ticks: 80,
		trail_offset_ticks: 5
	},
	{
		slug: 'trail80_5_sl1000',
		label: 'no TP / SL1000 / trail80-5',
		take_profit_ticks: 0,
		stop_loss_ticks: 1000,
		use_trailing_stop: true,
		trail_trigger_ticks: 80,
		trail_offset_ticks: 5
	},
	{
		slug: 'tp90_sl1000_trail80_5',
		label: 'TP90 / SL1000 / trail80-5',
		take_profit_ticks: 90,
		stop_loss_ticks: 1000,
		use_trailing_stop: true,
		trail_trigger_ticks: 80,
		trail_offset_ticks: 5
	}
];

const selectedProtections = protectionFilter === 'all'
	? protections
	: protections.filter((item) => item.slug === protectionFilter);
if (!selectedProtections.length) {
	throw new Error(`unknown --protection=${protectionFilter}; expected all or ${protections.map((item) => item.slug).join(', ')}`);
}

const historyGates = [
	{ slug: 'h2_m300_d1', history: 2, margin: 300, decay: 1 },
	{ slug: 'h6_m600_d05', history: 6, margin: 600, decay: 0.5 },
	{ slug: 'h12_m600_d05', history: 12, margin: 600, decay: 0.5 }
];

const adxLengths = [3, 5, 7, 10, 14, 20, 28];
const adxThresholds = [20, 25, 30, 35, 40, 45, 50, 55, 60];

const templatePath = path.join(root, '.run', 'replay-sweeps', 'hma210-240-protection-20260820', 'gc-aug10-tpsl', 'sweep.json');
const template = JSON.parse(fs.readFileSync(templatePath, 'utf8'));
const outputRoot = path.join(root, '.run', 'replay-sweeps', `gc-range10-ema420-480-remote-match-20260918${signalDelayBars === 0 ? '-delay0' : ''}${reversalMode === 'CloseAllEnter' ? '-closeall' : ''}`);

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function markovGate(gate) {
	return {
		enabled: true,
		score_window_outcomes: gate.history,
		minimum_completed_outcomes: 1,
		score_margin_ticks: gate.margin,
		max_abs_outcome_ticks: 1200,
		confirmation_events: 1,
		minimum_dwell_events: 0,
		reset_after_gap_minutes: 120,
		efficiency_lookback_bars: null,
		neutral_below_efficiency_ratio: null,
		neutral_action: 'normal_fallback',
		outcome_decay: gate.decay,
		normal_override_above_efficiency_ratio: null
	};
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

function baseSpec(window, protection, id, outputDir) {
	const spec = clone(template);
	spec.schema_version = 1;
	spec.sweep_id = id;
	spec.name = `GCZ6 10-range EMA 420/480 ${protection.label} — ${window.label}`;
	spec.notes = 'Exact-contract GCZ6 Databento trades; causal range10 bars; matched remote EMA 420/480 settings; closed bar + one-bar delay; Guarded Flatten > Confirm > Enter; fees included.';
	spec.base_dataset_view = {
		view_version: 1,
		id: `gcz6_${window.slug}_range10_ema420_480_remote_match_20260918`,
		source: {
			manifest_id: 'tradovate/sim/GC/GCZ6/2025-08-19/manifest.json',
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
		max_parallel_jobs: parallelism,
		max_cache_read_rows_per_worker: 20000000
	};
	spec.base_strategy.kind = 'Native';
	spec.base_strategy.native_strategy = 'EmaCross';
	spec.base_strategy.native_signal_timing = 'ClosedBar';
	spec.base_strategy.native_signal_delay_bars = signalDelayBars;
	spec.base_strategy.native_execution_path = 'Guarded';
	spec.base_strategy.native_reversal_mode = reversalMode;
	spec.base_strategy.blockout_enabled = true;
	spec.base_strategy.blockout_minutes_before_close = 45;
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

function runSpec(spec, specPath) {
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${spec.name} (${spec.max_runs} runs) ===`);
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
	if (result.status !== 0) throw new Error(`replay sweep failed: ${specPath} (status ${result.status})`);
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
	for (const run of summary.runs) {
		if (!run.result_path) continue;
		const resultPath = path.isAbsolute(run.result_path) ? run.result_path : path.join(root, run.result_path);
		fs.rmSync(resultPath, { force: true });
	}
	return rows;
}

const rows = [];
for (const window of windows) {
	for (const protection of selectedProtections) {
		const baseDir = path.join(outputRoot, phase, window.slug, protection.slug);
		if (phase === 'baseline') {
			const spec = baseSpec(window, protection, `gc_ema420_480_range10_remote_${protection.slug}_${window.slug}`, path.relative(root, baseDir));
			spec.parameters = [{ path: 'native_ema.inverted', values: [false, true] }];
			spec.max_runs = 2;
			spec.guardrails.max_combinations = 2;
			for (const row of runSpec(spec, path.join(baseDir, 'spec.json'))) {
				rows.push({ phase, window: window.slug, protection: protection.slug, gate: 'none', ...row });
			}
		}
		if (phase === 'history') {
			for (const gate of historyGates) {
				const spec = baseSpec(window, protection, `gc_ema420_480_range10_history_${gate.slug}_${protection.slug}_${window.slug}`, path.relative(root, path.join(baseDir, gate.slug)));
				spec.replay_markov_orientation_gate = markovGate(gate);
				spec.parameters = [];
				spec.max_runs = 1;
			spec.guardrails.max_combinations = 1;
				for (const row of runSpec(spec, path.join(baseDir, gate.slug, 'spec.json'))) {
					rows.push({ phase, window: window.slug, protection: protection.slug, gate: gate.slug, ...row });
				}
			}
		}
		if (phase === 'adx') {
			const spec = baseSpec(window, protection, `gc_ema420_480_range10_adx_${protection.slug}_${window.slug}`, path.relative(root, baseDir));
			spec.base_strategy.native_strategy = 'VolumeAdaptiveEmaCross';
			spec.base_strategy.native_volume_ema_cross = {
			...spec.base_strategy.native_volume_ema_cross,
			fast_length: 420,
			slow_length: 480,
			inverted: false,
			take_profit_ticks: protection.take_profit_ticks,
			stop_loss_ticks: protection.stop_loss_ticks,
			use_trailing_stop: protection.use_trailing_stop,
			trail_trigger_ticks: protection.trail_trigger_ticks,
			trail_offset_ticks: protection.trail_offset_ticks,
			volume_regime: { lookback_bars: 30, invert_below_relative_volume: 0 },
			ema_gate: { enabled: false, ema_length: 500, invert_when_above: true },
			adaptive_gate: adaptiveAdxGate(14, 45)
		};
		spec.parameters = [
			{ path: 'native_volume_ema_cross.adaptive_gate.adx_length', values: adxLengths },
			{ path: 'native_volume_ema_cross.adaptive_gate.invert_below_adx', values: adxThresholds }
		];
		spec.max_runs = adxLengths.length * adxThresholds.length;
		spec.guardrails.max_combinations = spec.max_runs;
		for (const row of runSpec(spec, path.join(baseDir, 'spec.json'))) {
			rows.push({ phase, window: window.slug, protection: protection.slug, gate: 'adx', ...row });
		}
		}
	}
}

const report = {
	schema_version: 'gc-range10-ema420-480-remote-match-v1',
	generated_at_utc: new Date().toISOString(),
	phase,
	contract: 'GCZ6',
	bar_type: '10-range standard OHLC',
	strategy: `EMA 420/480, ClosedBar, ${signalDelayBars}-bar delay, Guarded, ${reversalMode}, 45m blockout`,
	signal_delay_bars: signalDelayBars,
	windows,
	protections: selectedProtections,
	history_gates: historyGates,
	adx_grid: { lengths: adxLengths, thresholds: adxThresholds },
	rows
};
fs.mkdirSync(outputRoot, { recursive: true });
const reportPath = path.join(outputRoot, `${phase}-summary.json`);
fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ phase, rows: rows.length, report: path.relative(root, reportPath) }, null, 2));

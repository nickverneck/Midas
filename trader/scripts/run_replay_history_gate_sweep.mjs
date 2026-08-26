#!/usr/bin/env node

/**
 * Reproducible replay sweep for the causal historical-cross orientation gate.
 *
 * The gate itself is implemented in Rust. This runner only expands the
 * research grid over already-completed crossover outcomes and launches the
 * normal prepared replay kernel for each GC window. It deliberately does not
 * consume event labels or future/oracle columns.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'debug', 'trader');
const stage = process.argv.find((arg) => arg.startsWith('--stage='))?.split('=')[1] ?? 'history';
const trigger = process.argv.find((arg) => arg.startsWith('--trigger='))?.split('=')[1] ?? 'hma';
const baseline = stage === 'baseline';
const parallelism = Number(process.argv.find((arg) => arg.startsWith('--parallelism='))?.split('=')[1] ?? 1);
const tag = process.argv.find((arg) => arg.startsWith('--tag='))?.split('=')[1];
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
if (!['hma', 'ema'].includes(trigger)) {
	throw new Error(`unsupported --trigger=${trigger}; expected hma or ema`);
}
const outputStageBase = trigger === 'hma' ? stage : `${trigger}-${stage}`;
const outputStage = tag ? `${outputStageBase}-${tag}` : outputStageBase;
const outputRoot = path.join(root, '.run', 'replay-sweeps', 'historical-cross-gate-20260820', outputStage);

const windows = [
	{
		slug: 'gcz6-aug03-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/gc-aug03-markov-m300.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcz6-aug10-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/gc-aug10-markov-m300.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcz6-jul06-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/older/gc-jul06-markov600.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcz6-jul13-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/older/gc-jul13-markov600.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcz6-jul20-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/older/gc-jul20-markov600.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcz6-jul27-hma210-240',
		template: '.run/replay-sweeps/hma210-240-gates-20260820/older/gc-jul27-markov600.json',
		cache: '.run/replay-cache'
	},
	{
		slug: 'gcq6-jun22-ema10-30',
		template: '.run/replay-sweeps/gc-regime-gate-research-20260810/markov/gcq6-jun22-1m-ema10-30/markov.json',
		cache: '.run/replay-gcq6-random-week-cache-20260622'
	}
];

const stageConfig = stage === 'trend-veto'
	? {
		name: 'historical outcome + strong-efficiency normal veto',
		history: [1, 2, 3, 6, 8, 12],
		minimum: [1],
		margin: [300, 600],
		decay: [0.5, 0.75, 1],
		normalOverride: [null, 0.5, 0.75, 0.9],
		reset: [120],
		efficiencyLookback: 30,
		neutralBelow: null
	}
	: stage === 'fixed'
	? {
		name: 'frozen recency-weighted historical outcome gate',
		history: [6, 8, 12],
		minimum: [1],
		margin: [600],
		decay: [0.5],
		normalOverride: [null],
		reset: [120],
		efficiencyLookback: null,
		neutralBelow: null
	}
	: stage === 'session'
	? {
		name: 'historical outcome + session-gap reset comparison',
		history: [1, 2, 3, 6, 8, 12],
		minimum: [1],
		margin: [300, 600],
		decay: [0.5, 1],
		normalOverride: [null],
		reset: [60, 120, null],
		efficiencyLookback: null,
		neutralBelow: null
	}
	: stage === 'regime'
	? {
		name: 'historical outcome + efficiency regime override',
		history: [1, 2, 3, 4, 6, 8],
		minimum: [1],
		margin: [100, 300, 600],
		decay: [0.5, 0.75, 1],
		normalOverride: [null, 0.5, 0.75],
		reset: [120],
		efficiencyLookback: 30,
		neutralBelow: 0.2
	}
	: {
		name: 'historical crossover outcome window and margin',
		history: [1, 2, 3, 4, 6, 8, 12, 16, 24],
		minimum: [1],
		margin: [100, 300, 600],
		decay: [1],
		normalOverride: [null],
		reset: [120],
		efficiencyLookback: null,
		neutralBelow: null
	};

function values(pathName, valuesList) {
	return { path: pathName, values: valuesList };
}

function makeSpec(window) {
	const templatePath = path.join(root, window.template);
	const spec = JSON.parse(fs.readFileSync(templatePath, 'utf8'));
	const combinationCount = stageConfig.history.length
		* stageConfig.minimum.length
		* stageConfig.margin.length
		* stageConfig.decay.length
		* stageConfig.normalOverride.length
		* stageConfig.reset.length;

	spec.schema_version = 1;
	spec.sweep_id = `history_gate_${trigger}_${window.slug}_${stage}`;
	spec.name = `GC ${trigger.toUpperCase()} historical crossover gate — ${window.slug} — ${stageConfig.name}`;
	spec.notes = `${spec.notes}; causal prior-cross shadow outcomes only; no future labels; trigger=${trigger}; stage=${stage}`;
	if (trigger === 'ema') {
		spec.base_strategy.native_strategy = 'EmaCross';
		spec.base_strategy.native_reversal_mode = 'CloseAllEnter';
		spec.base_strategy.native_ema = {
			...(spec.base_strategy.native_ema ?? {}),
			fast_length: 10,
			slow_length: 30,
			inverted: false,
			take_profit_ticks: 0,
			stop_loss_ticks: 0,
			use_trailing_stop: false
		};
	}
	spec.parameters = baseline ? [] : [
		values('replay_markov_orientation_gate.score_window_outcomes', stageConfig.history),
		values('replay_markov_orientation_gate.minimum_completed_outcomes', stageConfig.minimum),
		values('replay_markov_orientation_gate.score_margin_ticks', stageConfig.margin),
		values('replay_markov_orientation_gate.outcome_decay', stageConfig.decay),
		values('replay_markov_orientation_gate.normal_override_above_efficiency_ratio', stageConfig.normalOverride),
		values('replay_markov_orientation_gate.reset_after_gap_minutes', stageConfig.reset)
	];
	spec.max_runs = baseline ? 1 : combinationCount;
	spec.parallelism = parallelism;
	spec.execution_mode = 'prepared_cpu';
	spec.evaluator_mode = 'streaming';
	spec.output_formats = ['json_summary'];
	spec.output_dir = path.relative(root, path.join(outputRoot, window.slug));
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: baseline ? 1 : combinationCount,
		max_parallel_jobs: parallelism
	};
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: !baseline,
		minimum_completed_outcomes: 1,
		confirmation_events: 1,
		minimum_dwell_events: 0,
		efficiency_lookback_bars: stageConfig.efficiencyLookback,
		neutral_below_efficiency_ratio: stageConfig.neutralBelow,
		normal_override_above_efficiency_ratio: null,
		outcome_decay: 1,
		reset_after_gap_minutes: stageConfig.reset[0]
	};
	return spec;
}

function runWindow(window) {
	const spec = makeSpec(window);
	const specPath = path.join(outputRoot, `${window.slug}.json`);
	fs.mkdirSync(outputRoot, { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);

	const env = { ...process.env, TRADER_DATA_CACHE_DIR: path.join(root, window.cache) };
	const result = spawnSync(
		binary,
		[
			'run-replay-sweep',
			'--spec',
			specPath,
			'--no-resume',
			...(spec.max_runs > (spec.guardrails.large_sweep_threshold ?? 100) ? ['--allow-large'] : [])
		],
		{ cwd: root, env, encoding: 'utf8', stdio: 'inherit' }
	);
	if (result.status !== 0) {
		throw new Error(`replay sweep failed for ${window.slug} with status ${result.status}`);
	}

	const planPath = path.join(root, spec.output_dir, 'sweep-plan.json');
	const summaryPath = path.join(root, spec.output_dir, 'sweep-summary.json');
	const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
	const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
	const parametersByRun = new Map(plan.children.map((child) => [child.run_id, child.parameter_values]));
	return summary.runs.map((run) => ({
		window: window.slug,
		trigger: spec.base_strategy.native_strategy,
		parameters: parametersByRun.get(run.run_id) ?? {},
		run_id: run.run_id,
		status: run.status,
		gross_pnl: run.gross_pnl,
		net_pnl: run.net_pnl,
		fees: run.fees,
		trade_count: run.trade_count,
		max_drawdown: run.max_drawdown
	}));
}

if (!fs.existsSync(binary)) {
	throw new Error(`missing ${binary}; build first with cargo build --features replay --bin trader`);
}

const rows = windows.flatMap(runWindow);
const report = {
	schema_version: 'historical-cross-gate-sweep-v1',
	trigger,
	stage,
	causal_definition: 'At cross i, settle only cross i-1 using close-to-close signed raw-direction return; select orientation for cross i.',
	windows: windows.map(({ slug, template, cache }) => ({ slug, template, cache })),
	grid: stageConfig,
	rows
};
fs.writeFileSync(path.join(outputRoot, 'summary.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ stage, rows: rows.length, output: path.relative(root, path.join(outputRoot, 'summary.json')) }, null, 2));

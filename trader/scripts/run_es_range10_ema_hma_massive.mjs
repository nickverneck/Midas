#!/usr/bin/env node

/**
 * ESU6 10-range EMA/HMA research matrix.
 *
 * Phase 1 deliberately runs a broad, fee-inclusive normal/inverted grid over
 * every complete ESU6 week available in the imported Databento trade tape.
 * Phase 2 applies protection only to candidates that survive that cross-week
 * baseline, plus explicit 10/30 and slow-pair controls.  This keeps the
 * protection search from selecting a one-week artifact.
 *
 * The range bars are built causally by the replay engine from the exact ESU6
 * trades.  Signals are closed-bar, fills are raw-bar-open, and the ES fee is
 * $2.88 per contract side.
 */

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '..');
const binary = path.join(root, 'target', 'release', 'trader');
const cacheConfig = path.join(root, '.run', 'databento-replay-cache.toml');
const templatePath = path.join(root, '.run', 'replay-sweeps', 'hma210-240-protection-20260820', 'gc-aug10-tpsl', 'sweep.json');
const outputRoot = path.join(root, '.run', 'replay-sweeps', 'es-range10-ema-hma-massive-20260827');
const sourceManifest = 'tradovate/sim/ES/ESU6/2026-02-09/manifest.json';

const option = (name) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
const phase = option('phase') ?? 'baseline';
const scope = option('scope') ?? (phase === 'protection' ? 'recent4' : 'all');
const parallelism = Number(option('parallelism') ?? 4);
const maxProtectionPairs = Number(option('max-protection-pairs') ?? 8);
const windowFilter = option('window');

if (!['baseline', 'protection', 'all'].includes(phase)) {
	throw new Error(`unsupported --phase=${phase}; expected baseline, protection, or all`);
}
if (!['all', 'recent4', 'recent8'].includes(scope)) {
	throw new Error(`unsupported --scope=${scope}; expected all, recent4, or recent8`);
}
if (!Number.isInteger(parallelism) || parallelism < 1 || parallelism > 8) {
	throw new Error(`unsupported --parallelism=${parallelism}; expected an integer from 1 to 8`);
}
if (!Number.isInteger(maxProtectionPairs) || maxProtectionPairs < 1 || maxProtectionPairs > 24) {
	throw new Error(`unsupported --max-protection-pairs=${maxProtectionPairs}; expected an integer from 1 to 24`);
}
for (const required of [binary, cacheConfig, templatePath]) {
	if (!fs.existsSync(required)) throw new Error(`missing ${required}`);
}

const fastValues = [10, 20, 30, 40, 60, 80, 100, 120, 150, 180, 200, 210, 240, 250, 300, 360, 420, 480];
const slowValues = [30, 60, 90, 120, 150, 180, 240, 280, 300, 360, 400, 420, 480, 500, 560, 600, 700, 720, 800];

const triggerKinds = [
	{ slug: 'ema', strategy: 'EmaCross', prefix: 'native_ema' },
	{ slug: 'hma', strategy: 'HmaCross', prefix: 'native_hma_cross' }
];

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function utcDate(date) {
	return date.toISOString().replace('.000Z', 'Z');
}

function makeWindows() {
	const result = [];
	const first = Date.parse('2026-02-16T00:00:00Z');
	const last = Date.parse('2026-08-03T00:00:00Z');
	for (let start = first; start <= last; start += 7 * 24 * 60 * 60 * 1000) {
		// ESU6 coverage ends at the Friday Globex close.  Do not extend the
		// evaluation view through the weekend, where no source rows exist.
		// Leave a sub-millisecond margin under the archive's final timestamp;
		// the source ends at 20:59:59.841574399Z.
		const end = start + 4 * 24 * 60 * 60 * 1000 + 21 * 60 * 60 * 1000 - 160;
		const startDate = new Date(start);
		const endDate = new Date(end);
		const slug = utcDate(startDate).slice(0, 10);
		result.push({
			slug,
			label: `${utcDate(startDate).slice(0, 10)} through ${utcDate(endDate).slice(0, 10)}`,
			evaluationStart: utcDate(startDate),
			evaluationEnd: utcDate(endDate)
		});
	}
	return result;
}

const allWindows = makeWindows();
let selectedWindows = allWindows;
if (scope === 'recent4') selectedWindows = allWindows.slice(-4);
if (scope === 'recent8') selectedWindows = allWindows.slice(-8);
if (windowFilter) selectedWindows = selectedWindows.filter((window) => window.slug === windowFilter);
if (!selectedWindows.length) throw new Error(`no windows selected for scope=${scope}, window=${windowFilter ?? '*'}`);

const template = JSON.parse(fs.readFileSync(templatePath, 'utf8'));

function strategyFor(kind, fast = 10, slow = 30, inverted = false, protection = {}) {
	const strategy = clone(template.base_strategy);
	strategy.native_strategy = kind.strategy;
	strategy.native_signal_timing = 'ClosedBar';
	strategy.native_signal_delay_bars = 0;
	strategy.native_execution_path = 'Guarded';
	strategy.native_reversal_mode = 'CloseAllEnter';
	strategy.blockout_enabled = true;
	strategy.blockout_minutes_before_close = 45;
	strategy.order_qty = 1;

	strategy.native_ema = {
		...strategy.native_ema,
		fast_length: kind.slug === 'ema' ? fast : 10,
		slow_length: kind.slug === 'ema' ? slow : 30,
		inverted: kind.slug === 'ema' ? inverted : false,
		take_profit_ticks: 0,
		stop_loss_ticks: 0,
		use_trailing_stop: false,
		trail_trigger_ticks: 40,
		trail_offset_ticks: 10
	};
	strategy.native_hma_cross = {
		...strategy.native_hma_cross,
		fast_length: kind.slug === 'hma' ? fast : 210,
		slow_length: kind.slug === 'hma' ? slow : 240,
		calculation_mode: 'incremental',
		inverted: kind.slug === 'hma' ? inverted : false,
		take_profit_ticks: 0,
		stop_loss_ticks: 0,
		use_trailing_stop: false,
		trail_trigger_ticks: 40,
		trail_offset_ticks: 10
	};

	const target = kind.prefix;
	strategy[target] = {
		...strategy[target],
		...protection
	};
	return strategy;
}

function baseSpec(window, kind, outputDir) {
	const spec = clone(template);
	const id = `esu6_range10_${kind.slug}_${window.slug}_20260827`;
	spec.schema_version = 1;
	spec.sweep_id = id;
	spec.name = `ESU6 10-range ${kind.strategy} broad orientation matrix — ${window.label}`;
	spec.notes = 'Exact-contract ESU6 Databento trades; causal 10-range bars; closed-bar raw-bar-open execution; normal and inverted orientations; explicit ES fee included; 7-day indicator warmup trades flat until evaluation.';
	spec.base_dataset_view = {
		view_version: 1,
		id: `esu6_${window.slug}_range10_standard_20260827`,
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
			// The archive starts on Feb 9 at 02:21:43Z.  Keep the first
			// evaluation view inside that exact coverage while retaining
			// essentially a full week of pre-roll.
			duration_seconds: window.slug === '2026-02-16' ? 596000 : 604800,
			trading: 'flat_until_evaluation'
		}
	};
	spec.bar_type = { kind: 'range', value: 10 };
	spec.candle_mode = 'standard';
	spec.fill_model = 'raw_bar_open';
	spec.execution_mode = 'prepared_cpu';
	spec.evaluator_mode = 'streaming';
	spec.parallelism = parallelism;
	spec.output_formats = ['json_summary'];
	spec.output_dir = path.relative(root, outputDir);
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
	spec.base_strategy = strategyFor(kind);
	spec.replay_markov_orientation_gate = {
		...(spec.replay_markov_orientation_gate ?? {}),
		enabled: false
	};
	spec.guardrails = {
		...(spec.guardrails ?? {}),
		max_combinations: 10000,
		max_parallel_jobs: parallelism,
		max_cache_read_rows_per_worker: 20000000,
		large_sweep_threshold: 100,
		max_estimated_memory_bytes: 8589934592,
		// The runner removes per-candidate result ledgers after extracting the
		// compact summary.  Keep a finite but higher launch ceiling for this
		// intentionally large matrix; the default 64 GiB estimate is based on
		// raw-tick rows and is conservative for prepared range-bar summaries.
		max_estimated_output_bytes: 137438953472
	};
	return spec;
}

function baselineSpec(window, kind) {
	const outputDir = path.join(outputRoot, 'baseline', window.slug, kind.slug);
	const spec = baseSpec(window, kind, outputDir);
	const prefix = kind.prefix;
	spec.parameters = [
		{ path: `${prefix}.fast_length`, values: fastValues },
		{ path: `${prefix}.slow_length`, values: slowValues },
		{ path: `${prefix}.inverted`, values: [false, true] }
	];
	spec.constraints = [{ kind: 'less', left: `${prefix}.fast_length`, right: `${prefix}.slow_length` }];
	spec.max_runs = fastValues.length * slowValues.length * 2;
	spec.guardrails.max_combinations = spec.max_runs;
	spec.sweep_id += '_baseline';
	spec.name += ' baseline';
	return spec;
}

function protectionSpec(window, kind, pair, grid) {
	const outputDir = path.join(outputRoot, 'protection', scope, window.slug, kind.slug, `${pair.fast}_${pair.slow}`, grid.slug);
	const spec = baseSpec(window, kind, outputDir);
	const prefix = kind.prefix;
	spec.base_strategy = strategyFor(kind, pair.fast, pair.slow);
	spec.sweep_id = `esu6_range10_${kind.slug}_${pair.fast}_${pair.slow}_${grid.slug}_${window.slug}_20260827`;
	spec.name = `ESU6 10-range ${kind.strategy} ${pair.fast}/${pair.slow} ${grid.name} — ${window.label}`;
	spec.notes = 'Exact-contract ESU6 Databento trades; causal 10-range bars; protection applied only after cross-week baseline selection; normal and inverted orientation included; ES fee included.';
	spec.parameters = grid.parameters(prefix);
	spec.max_runs = spec.parameters.reduce((count, parameter) => count * parameter.values.length, 1);
	spec.guardrails.max_combinations = spec.max_runs;
	return spec;
}

const protectionGrids = [
	{
		slug: 'bracket',
		name: 'bracket protection',
		parameters: (prefix) => [
			{ path: `${prefix}.inverted`, values: [false, true] },
			{ path: `${prefix}.take_profit_ticks`, values: [0, 20, 40, 60, 80, 120, 160] },
			{ path: `${prefix}.stop_loss_ticks`, values: [0, 40, 80, 120, 160] },
			{ path: `${prefix}.use_trailing_stop`, values: [false] }
		]
	},
	{
		slug: 'trailing',
		name: 'trailing protection',
		parameters: (prefix) => [
			{ path: `${prefix}.inverted`, values: [false, true] },
			{ path: `${prefix}.take_profit_ticks`, values: [0, 80] },
			{ path: `${prefix}.stop_loss_ticks`, values: [0, 80] },
			{ path: `${prefix}.use_trailing_stop`, values: [true] },
			{ path: `${prefix}.trail_trigger_ticks`, values: [20, 40, 60, 80] },
			{ path: `${prefix}.trail_offset_ticks`, values: [5, 10, 20] }
		]
	}
];

function runSpec(spec, label) {
	const specPath = path.join(outputRoot, 'specs', `${label}-${spec.sweep_id}.json`);
	fs.mkdirSync(path.dirname(specPath), { recursive: true });
	fs.writeFileSync(specPath, `${JSON.stringify(spec, null, 2)}\n`);
	console.log(`\n=== ${label}: ${spec.sweep_id} (${spec.max_runs} planned combinations) ===`);
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
		const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`.trim();
		throw new Error(`replay failed for ${specPath} with status ${result.status}:\n${output.slice(-6000)}`);
	}
	const outputDir = path.join(root, spec.output_dir);
	const summaryPath = path.join(outputDir, 'sweep-summary.json');
	const planPath = path.join(outputDir, 'sweep-plan.json');
	const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
	const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
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

function numberAt(parameters, pathName) {
	const value = parameters[pathName];
	return typeof value === 'number' ? value : Number(value);
}

function boolAt(parameters, pathName) {
	return parameters[pathName] === true || parameters[pathName] === 'true';
}

function aggregateBaseline(rows) {
	const groups = new Map();
	for (const row of rows) {
		const fast = numberAt(row.parameters, `${row.kindPrefix}.fast_length`);
		const slow = numberAt(row.parameters, `${row.kindPrefix}.slow_length`);
		const inverted = boolAt(row.parameters, `${row.kindPrefix}.inverted`);
		const key = `${row.kind}:${fast}/${slow}:${inverted ? 'inverted' : 'normal'}`;
		if (!groups.has(key)) groups.set(key, { key, kind: row.kind, fast, slow, inverted, weeks: [] });
		groups.get(key).weeks.push(row);
	}
	const aggregates = [...groups.values()].map((group) => {
		const nets = group.weeks.map((row) => row.net_pnl).filter(Number.isFinite).sort((a, b) => a - b);
		const positiveWeeks = nets.filter((value) => value > 0).length;
		const median = nets.length ? nets[Math.floor(nets.length / 2)] : null;
		return {
			key: group.key,
			kind: group.kind,
			fast: group.fast,
			slow: group.slow,
			orientation: group.inverted ? 'inverted' : 'normal',
			weeks: group.weeks.length,
			positive_weeks: positiveWeeks,
			positive_week_ratio: group.weeks.length ? positiveWeeks / group.weeks.length : 0,
			net_pnl_sum: group.weeks.reduce((sum, row) => sum + (row.net_pnl ?? 0), 0),
			gross_pnl_sum: group.weeks.reduce((sum, row) => sum + (row.gross_pnl ?? 0), 0),
			fees_sum: group.weeks.reduce((sum, row) => sum + (row.fees ?? 0), 0),
			median_weekly_net_pnl: median,
			max_weekly_drawdown: Math.max(...group.weeks.map((row) => row.max_drawdown ?? 0)),
			trade_count_sum: group.weeks.reduce((sum, row) => sum + (row.trade_count ?? 0), 0)
		};
	});
	const byNet = [...aggregates].sort((a, b) => b.net_pnl_sum - a.net_pnl_sum);
	const byRobustness = [...aggregates].sort((a, b) => {
		if (b.positive_week_ratio !== a.positive_week_ratio) return b.positive_week_ratio - a.positive_week_ratio;
		if ((b.median_weekly_net_pnl ?? -Infinity) !== (a.median_weekly_net_pnl ?? -Infinity)) return (b.median_weekly_net_pnl ?? -Infinity) - (a.median_weekly_net_pnl ?? -Infinity);
		return b.net_pnl_sum - a.net_pnl_sum;
	});
	return { by_net_pnl: byNet, by_robustness: byRobustness };
}

function pairKey(kind, fast, slow) {
	return `${kind.slug}:${fast}/${slow}`;
}

function selectProtectionPairs(baselineSummary) {
	const pairScores = new Map();
	for (const row of baselineSummary.by_net_pnl) {
		const key = `${row.kind}:${row.fast}/${row.slow}`;
		const existing = pairScores.get(key);
		if (!existing || row.net_pnl_sum > existing.best_orientation_net_pnl) {
			pairScores.set(key, {
				key,
				kind: triggerKinds.find((candidate) => candidate.slug === row.kind),
				fast: row.fast,
				slow: row.slow,
				best_orientation_net_pnl: row.net_pnl_sum
			});
		}
	}
	const explicit = [
		[10, 30],
		[210, 240],
		[200, 400],
		[240, 480],
		[300, 600],
		[360, 720]
	];
	const selected = [];
	for (const kind of triggerKinds) {
		const candidates = [...pairScores.values()]
			.filter((candidate) => candidate.kind?.slug === kind.slug)
			.sort((a, b) => b.best_orientation_net_pnl - a.best_orientation_net_pnl);
		const ordered = [];
		for (const [fast, slow] of explicit) {
			const candidate = pairScores.get(pairKey(kind, fast, slow));
			if (candidate) ordered.push(candidate);
		}
		ordered.push(...candidates);
		for (const candidate of ordered) {
			if (selected.some((item) => item.key === candidate.key)) continue;
			selected.push(candidate);
			if (selected.filter((item) => item.kind.slug === kind.slug).length >= Math.ceil(maxProtectionPairs / 2)) break;
		}
	}
	return selected.slice(0, maxProtectionPairs);
}

const baselineRows = [];
const protectionRows = [];

if (phase === 'baseline' || phase === 'all') {
	for (const window of selectedWindows) {
		for (const kind of triggerKinds) {
			const spec = baselineSpec(window, kind);
			const rows = runSpec(spec, `baseline-${window.slug}-${kind.slug}`);
			for (const row of rows) baselineRows.push({
				phase: 'baseline',
				window: window.slug,
				kind: kind.slug,
				kindPrefix: kind.prefix,
				...row
			});
		}
	}
	fs.mkdirSync(outputRoot, { recursive: true });
	const summary = {
		schema_version: 'es-range10-ema-hma-massive-baseline-v1',
		generated_at_utc: new Date().toISOString(),
		contract: 'ESU6',
		bar_type: '10-range standard OHLC',
		fee_per_contract_side: 2.88,
		warmup_seconds: 604800,
		windows: selectedWindows,
		fast_values: fastValues,
		slow_values: slowValues,
		rows: baselineRows,
		aggregates: aggregateBaseline(baselineRows)
	};
	fs.writeFileSync(path.join(outputRoot, 'baseline-rows.json'), `${JSON.stringify(baselineRows, null, 2)}\n`);
	fs.writeFileSync(path.join(outputRoot, 'baseline-summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
	console.log(JSON.stringify({ phase: 'baseline', rows: baselineRows.length, output: path.relative(root, path.join(outputRoot, 'baseline-summary.json')), top_net: summary.aggregates.by_net_pnl.slice(0, 12), top_robustness: summary.aggregates.by_robustness.slice(0, 12) }, null, 2));
}

if (phase === 'protection' || phase === 'all') {
	const baselinePath = path.join(outputRoot, 'baseline-summary.json');
	if (!fs.existsSync(baselinePath)) throw new Error(`missing ${baselinePath}; run --phase=baseline first`);
	const baselineSummary = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
	const pairs = selectProtectionPairs(baselineSummary.aggregates);
	console.log(`Selected ${pairs.length} protection pairs: ${pairs.map((pair) => pair.key).join(', ')}`);
	for (const window of selectedWindows) {
		for (const pair of pairs) {
			for (const grid of protectionGrids) {
				const spec = protectionSpec(window, pair.kind, pair, grid);
				const rows = runSpec(spec, `protection-${window.slug}-${pair.kind.slug}-${pair.fast}-${pair.slow}-${grid.slug}`);
				for (const row of rows) protectionRows.push({
					phase: 'protection',
					window: window.slug,
					kind: pair.kind.slug,
					kindPrefix: pair.kind.prefix,
					pair: `${pair.fast}/${pair.slow}`,
					grid: grid.slug,
					...row
				});
			}
		}
	}
	fs.mkdirSync(outputRoot, { recursive: true });
	fs.writeFileSync(path.join(outputRoot, 'protection-rows.json'), `${JSON.stringify(protectionRows, null, 2)}\n`);
	console.log(JSON.stringify({ phase: 'protection', rows: protectionRows.length, selected_pairs: pairs.map((pair) => pair.key), output: path.relative(root, path.join(outputRoot, 'protection-rows.json')) }, null, 2));
}

#!/usr/bin/env node

/**
 * Expanding, chronological walk-forward evaluation for train_event_gate.
 *
 * Each fold trains on the first N sessions, selects the best epoch on the
 * following validation sessions, and reports the next untouched test block.
 * The test block is represented by train_event_gate's holdout split after a
 * timestamp-bounded prefix is created; it is never used for selection.
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const root = process.cwd();

const defaults = {
	algorithm: 'advantage',
	epochs: 2000,
	checkpointEvery: 250,
	generations: 500,
	population: 12,
	seed: null,
	learningRate: 0.005,
	l2: 0.01,
	purgeBars: 30,
	trainSessions: 8,
	validationSessions: 2,
	testSessions: 2,
	step: 1,
	maxFolds: 10
};

const usage = () => {
	console.error(`Usage: node scripts/run_event_gate_walkforward.mjs --input EVENTS --outdir OUTDIR [options]

Options:
  --algorithm advantage|value|mlp|ga
  --feature-filter comma,separated,features
  --epochs N                 default ${defaults.epochs}
  --generations N            default ${defaults.generations} (GA)
  --population N             default ${defaults.population} (GA)
  --seed N                   optional deterministic seed
  --checkpoint-every N       default ${defaults.checkpointEvery}
  --learning-rate N          default ${defaults.learningRate}
  --l2 N                     default ${defaults.l2}
  --purge-bars N             default ${defaults.purgeBars}
  --train-sessions N         default ${defaults.trainSessions}
  --validation-sessions N    default ${defaults.validationSessions}
  --test-sessions N          default ${defaults.testSessions}
  --step N                   default ${defaults.step}
  --max-folds N              default ${defaults.maxFolds}`);
};

const parseArgs = (argv) => {
	const options = { ...defaults };
	const required = {};
	const numeric = new Set([
		'epochs',
		'checkpoint-every',
		'generations',
		'population',
		'learning-rate',
		'l2',
		'purge-bars',
		'train-sessions',
		'validation-sessions',
		'test-sessions',
		'step',
		'max-folds'
	]);
	const aliases = {
		'checkpoint-every': 'checkpointEvery',
		'generations': 'generations',
		'population': 'population',
		'learning-rate': 'learningRate',
		'train-sessions': 'trainSessions',
		'validation-sessions': 'validationSessions',
		'test-sessions': 'testSessions',
		'purge-bars': 'purgeBars',
		'max-folds': 'maxFolds'
	};
	for (let index = 0; index < argv.length; index += 1) {
		const token = argv[index];
		if (token === '--help' || token === '-h') {
			usage();
			process.exit(0);
		}
		if (!token.startsWith('--')) throw new Error(`unexpected argument ${token}`);
		const name = token.slice(2);
		const value = argv[++index];
		if (value === undefined || value.startsWith('--')) throw new Error(`${token} requires a value`);
		if (name === 'input' || name === 'outdir') {
			required[name] = value;
			continue;
		}
		if (name === 'algorithm' || name === 'feature-filter') {
			options[name === 'algorithm' ? 'algorithm' : 'featureFilter'] = value;
			continue;
		}
		if (name === 'seed') {
			const parsed = Number(value);
			if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error('--seed must be a non-negative safe integer');
			options.seed = parsed;
			continue;
		}
		if (!numeric.has(name)) throw new Error(`unknown option ${token}`);
		const parsed = Number(value);
		if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${token} must be a finite non-negative number`);
		options[aliases[name] ?? name] = parsed;
	}
	if (!required.input || !required.outdir) {
		usage();
		throw new Error('--input and --outdir are required');
	}
	if (!['advantage', 'value', 'mlp', 'ga'].includes(options.algorithm)) {
		throw new Error('--algorithm must be advantage, value, mlp, or ga');
	}
	for (const name of ['trainSessions', 'validationSessions', 'testSessions', 'step', 'maxFolds']) {
		if (!Number.isSafeInteger(options[name]) || options[name] < 1) throw new Error(`${name} must be a positive integer`);
	}
	if (!Number.isSafeInteger(options.epochs) || options.epochs < 1) throw new Error('--epochs must be a positive integer');
	if (!Number.isSafeInteger(options.generations) || options.generations < 1) throw new Error('--generations must be a positive integer');
	if (!Number.isSafeInteger(options.population) || options.population < 2) throw new Error('--population must be at least 2');
	if (!Number.isSafeInteger(options.checkpointEvery) || options.checkpointEvery < 0) throw new Error('--checkpoint-every must be a non-negative integer');
	if (options.purgeBars < 0 || options.l2 < 0 || options.learningRate <= 0) throw new Error('purge/l2 must be non-negative and learning rate positive');
	return { ...required, ...options };
};

const run = (program, args, { quiet = false } = {}) =>
	execFileSync(program, args, {
		cwd: root,
		encoding: 'utf8',
		stdio: quiet ? ['ignore', 'ignore', 'inherit'] : ['ignore', 'pipe', 'inherit']
	});

const readSessionRows = (input) => {
	const parquetDump = path.join(root, 'target', 'debug', 'parquet_dump');
	const output = run(parquetDump, ['--file', input, '--columns', 'timestamp_ns,session_id', '--limit', '0']);
	const lines = output.trim().split(/\r?\n/);
	if (lines.length < 2) throw new Error('event parquet has no timestamp/session rows');
	const header = lines.shift().split(',');
	const timestampIndex = header.indexOf('timestamp_ns');
	const sessionIndex = header.indexOf('session_id');
	if (timestampIndex < 0 || sessionIndex < 0) throw new Error('parquet_dump did not return timestamp_ns and session_id');
	const starts = [];
	let previousSession = null;
	let lastTimestamp = null;
	for (const line of lines) {
		if (!line.trim()) continue;
		const fields = line.split(',');
		const session = fields[sessionIndex];
		const timestamp = BigInt(fields[timestampIndex]);
		if (lastTimestamp !== null && timestamp <= lastTimestamp) {
			throw new Error('event parquet timestamps must be strictly increasing');
		}
		if (session !== previousSession) {
			starts.push({ session, timestamp });
			previousSession = session;
		}
		lastTimestamp = timestamp;
	}
	if (lastTimestamp === null || lastTimestamp === (1n << 63n) - 1n) {
		throw new Error('event parquet has no usable timestamp cutoff');
	}
	return { starts, endExclusive: lastTimestamp + 1n };
};

const summarize = (folds) => {
	const sum = (field) => folds.reduce((total, fold) => total + fold[field], 0);
	return {
		fold_count: folds.length,
		test_pnl_usd: sum('test_pnl_usd'),
		fixed_normal_pnl_usd: sum('fixed_normal_pnl_usd'),
		incremental_vs_normal_usd: sum('incremental_vs_normal_usd'),
		mean_test_pnl_usd: sum('test_pnl_usd') / folds.length,
		mean_incremental_vs_normal_usd: sum('incremental_vs_normal_usd') / folds.length,
		positive_incremental_folds: folds.filter((fold) => fold.incremental_vs_normal_usd > 0).length
	};
};

const main = () => {
	let options;
	try {
		options = parseArgs(process.argv.slice(2));
	} catch (error) {
		usage();
		throw error;
	}

	const input = path.resolve(root, options.input);
	const output = path.resolve(root, options.outdir);
	if (!fs.existsSync(input)) throw new Error(`input does not exist: ${input}`);
	if (fs.existsSync(output)) throw new Error(`outdir already exists; refusing to clobber it: ${output}`);
	fs.mkdirSync(output, { recursive: true });

	const { starts: sessions, endExclusive } = readSessionRows(input);
	const minimumSessions = options.trainSessions + options.validationSessions + options.testSessions;
	const maxAvailableFolds = Math.floor((sessions.length - minimumSessions) / options.step) + 1;
	const foldCount = Math.min(options.maxFolds, maxAvailableFolds);
	if (foldCount < 1) throw new Error(`only ${sessions.length} sessions; need at least ${minimumSessions}`);

	const parquetSlice = path.join(root, 'target', 'debug', 'parquet_slice');
	const trainGate = path.join(root, 'target', 'debug', 'train_event_gate');
	const folds = [];
	for (let fold = 0; fold < foldCount; fold += 1) {
		const trainSessions = options.trainSessions + fold * options.step;
		const firstTestSession = trainSessions + options.validationSessions;
		const prefixSessionCount = trainSessions + options.validationSessions + options.testSessions;
		const foldDir = path.join(output, `fold-${String(fold + 1).padStart(2, '0')}`);
		const prefix = path.join(foldDir, 'prefix.parquet');
		const modelOut = path.join(foldDir, 'model');
		fs.mkdirSync(foldDir, { recursive: true });
		// For every non-final fold, the next session's first event is an exact
		// exclusive cutoff.  A final fold needs the timestamp after the final
		// event, not the first timestamp of that final session; otherwise it
		// silently truncates its last test session to one event.
		const cutoff = sessions[prefixSessionCount]?.timestamp ?? endExclusive;
		run(parquetSlice, ['--input', input, '--output', prefix, '--before-ts-ns', cutoff.toString()], { quiet: true });

		const totalSessions = prefixSessionCount;
		const trainFraction = trainSessions / totalSessions;
		const validationFraction = options.validationSessions / totalSessions;
		const args = [
			'--input', prefix,
			'--outdir', modelOut,
			'--algorithm', options.algorithm,
			'--epochs', String(options.epochs),
			'--generations', String(options.generations),
			'--population', String(options.population),
			'--checkpoint-every', String(options.checkpointEvery),
			'--learning-rate', String(options.learningRate),
			'--l2', String(options.l2),
			'--purge-bars', String(options.purgeBars),
			'--train-fraction', String(trainFraction),
			'--validation-fraction', String(validationFraction)
		];
		if (options.featureFilter) args.push('--feature-filter', options.featureFilter);
		if (options.seed !== null) args.push('--seed', String(options.seed));
		run(trainGate, args, { quiet: true });
		const metricsPath = path.join(modelOut, 'metrics.json');
		const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
		const calibrated = options.algorithm === 'value' && metrics.calibration?.selected_holdout;
		const test = calibrated ? metrics.calibration.selected_holdout : metrics.learned;
		const normal = metrics.fixed_normal;
		const validation = calibrated ? metrics.calibration.selected_validation : metrics.selected_validation;
		folds.push({
			fold: fold + 1,
			train_sessions: trainSessions,
			validation_sessions: options.validationSessions,
			test_sessions: options.testSessions,
			train_start_session: sessions[0].session,
			test_start_session: sessions[firstTestSession].session,
			test_end_session: sessions[prefixSessionCount - 1].session,
			selected_iteration: metrics.selected_iteration,
			calibrated,
			validation_pnl_usd: validation.sum_pnl_usd,
			test_pnl_usd: test.sum_pnl_usd,
			fixed_normal_pnl_usd: normal.sum_pnl_usd,
			incremental_vs_normal_usd: test.sum_pnl_usd - normal.sum_pnl_usd,
			test_invert_count: test.invert_count,
			test_event_count: test.event_count,
			model_dir: path.relative(root, modelOut)
		});
	}

	const report = {
		schema_version: 'event-gate-walk-forward-v1',
		input: path.relative(root, input),
		algorithm: options.algorithm,
		feature_filter: options.featureFilter ?? null,
		train_sessions: options.trainSessions,
		validation_sessions: options.validationSessions,
		test_sessions: options.testSessions,
		step: options.step,
		purge_bars: options.purgeBars,
		epochs: options.epochs,
		generations: options.generations,
		population: options.population,
		seed: options.seed,
		checkpoint_every: options.checkpointEvery,
		learning_rate: options.learningRate,
		l2: options.l2,
		folds,
		summary: summarize(folds)
	};
	fs.writeFileSync(path.join(output, 'walkforward.json'), `${JSON.stringify(report, null, 2)}\n`);
	console.log(JSON.stringify(report, null, 2));
};

try {
	main();
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
}

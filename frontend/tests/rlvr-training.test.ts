// @ts-nocheck
import { expect, test } from 'bun:test';
import { buildRlvrCliArgs } from '../src/lib/server/rlvr';

test('builds the RLVR event command with the fixed two-action algorithm', () => {
	const args = buildRlvrCliArgs({
		input: '.run/training/gcz6-events.parquet',
		outdir: '.run/event-rlvr/runs/smoke',
		epochs: 1000,
		checkpointEvery: 250,
		hidden: 32,
		layers: 2,
		learningRate: 0.001,
		l2: 0.0001,
		entropyCoefficient: 0.01,
		seed: 123,
		trainFraction: 0.6,
		validationFraction: 0.2,
		purgeBars: 30,
		rewardNormalization: 'train-mean-abs'
	});

	expect(args.slice(0, 4)).toEqual(['run', '--bin', 'train_event_rl', '--']);
	expect(args).toContain('--algorithm');
	expect(args[args.indexOf('--algorithm') + 1]).toBe('rlvr');
	expect(args[args.indexOf('--input') + 1]).toBe('.run/training/gcz6-events.parquet');
	expect(args[args.indexOf('--outdir') + 1]).toBe('.run/event-rlvr/runs/smoke');
	expect(args[args.indexOf('--checkpoint-every') + 1]).toBe('250');
	expect(args[args.indexOf('--purge-bars') + 1]).toBe('30');
	expect(args[args.indexOf('--reward-normalization') + 1]).toBe('train-mean-abs');
});

test('does not put a backend or four-action Trader flag on the RLVR command', () => {
	const args = buildRlvrCliArgs({
		input: 'data/events.parquet',
		outdir: '.run/event-rlvr/runs/test',
		epochs: 1,
		checkpointEvery: 0,
		hidden: 8,
		layers: 1,
		learningRate: 0.01,
		l2: 0,
		entropyCoefficient: 0,
		seed: 42,
		trainFraction: 0.5,
		validationFraction: 0.25,
		purgeBars: 0,
		rewardNormalization: 'none'
	});

	expect(args).not.toContain('--backend');
	expect(args).not.toContain('--device');
	expect(args).not.toContain('--action-space');
	expect(args[args.indexOf('--algorithm') + 1]).toBe('rlvr');
});

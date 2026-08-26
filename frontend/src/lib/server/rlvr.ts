export type RlvrCliOptions = {
	input: string;
	outdir: string;
	epochs: number;
	checkpointEvery: number;
	hidden: number;
	layers: number;
	learningRate: number;
	l2: number;
	entropyCoefficient: number;
	seed: number;
	trainFraction: number;
	validationFraction: number;
	purgeBars: number;
	rewardNormalization: 'none' | 'train-mean-abs';
};

export const buildRlvrCliArgs = (options: RlvrCliOptions) => [
	'run',
	'--bin',
	'train_event_rl',
	'--',
	'--input',
	options.input,
	'--outdir',
	options.outdir,
	'--algorithm',
	'rlvr',
	'--epochs',
	String(options.epochs),
	'--checkpoint-every',
	String(options.checkpointEvery),
	'--hidden',
	String(options.hidden),
	'--layers',
	String(options.layers),
	'--learning-rate',
	String(options.learningRate),
	'--l2',
	String(options.l2),
	'--entropy-coefficient',
	String(options.entropyCoefficient),
	'--seed',
	String(options.seed),
	'--train-fraction',
	String(options.trainFraction),
	'--validation-fraction',
	String(options.validationFraction),
	'--purge-bars',
	String(options.purgeBars),
	'--reward-normalization',
	options.rewardNormalization
];

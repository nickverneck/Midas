export type StartMode = 'new' | 'continue';
export type TrainingAlgorithm = 'supervised' | 'ga' | 'rl';
export type BackendId = 'torch' | 'burn' | 'candle' | 'cpu-linear';
export type DeviceId = 'auto' | 'cpu' | 'cuda' | 'mps';
export type IndicatorKind = 'ema' | 'sma' | 'hma' | 'kama' | 'alma' | 'atr' | 'adx' | 'rvol' | 'er';
export type BarKind = 'minute' | 'second' | 'tick' | 'volume' | 'range';

export type BackendCapability = {
	ga: boolean;
	rl: boolean;
	supervised: boolean;
	devices: Record<DeviceId, boolean>;
	note: string;
};

export type TrainingCapabilities = {
	platform: string;
	backends: Record<BackendId, BackendCapability>;
};

export const fallbackTrainingCapabilities: TrainingCapabilities = {
	platform: 'unknown',
	backends: {
		torch: {
			ga: true,
			rl: true,
			supervised: false,
			devices: { auto: true, cpu: true, cuda: false, mps: false },
			note: 'Libtorch legacy GA/RL runner.'
		},
		burn: {
			ga: true,
			rl: true,
			supervised: true,
			devices: { auto: true, cpu: true, cuda: false, mps: false },
			note: 'GA, RL, and supervised Burn paths are implemented; CUDA depends on the server build.'
		},
		candle: {
			ga: true,
			rl: true,
			supervised: true,
			devices: { auto: true, cpu: true, cuda: false, mps: false },
			note: 'GA/RL and supervised CPU paths are implemented; CUDA requires the compiled CUDA feature.'
		},
		'cpu-linear': {
			ga: false,
			rl: false,
			supervised: true,
			devices: { auto: false, cpu: true, cuda: false, mps: false },
			note: 'Working supervised reference implementation on CPU.'
		}
	}
};

export type FeatureSpec = {
	indicator: IndicatorKind;
	period: number;
	lookbacks: number[];
	include_value: boolean;
	include_delta: boolean;
	normalize_by_atr: boolean;
};

export type TrainingDefinition = {
	sourcePath: string;
	instrument: string;
	contract: string;
	triggerFast: number;
	triggerSlow: number;
	contextFast: number;
	contextSlow: number;
	barKind: BarKind;
	barValue: number;
	features: FeatureSpec[];
	atrPeriod: number;
	slopeLookback: number;
	contractMultiplier: number;
	roundTripCost: number;
	timestampUnit: '' | 'ns' | 'us' | 'ms' | 's';
	allowIndexTimestamps: boolean;
};

export type PreparedSummary = {
	schema_version?: string;
	label_schema?: string;
	source_rows?: number;
	event_rows?: number;
	session_count?: number;
	feature_count?: number;
	feature_schema?: string;
	timestamp_source?: string;
	bar_kind?: string;
	bar_value?: number;
	leakage_check?: string;
};

export type PreparedArtifact = {
	path: string;
	summary: PreparedSummary | null;
};

export type TrainOptions = {
	epochs: number;
	checkpointEvery: number;
	learningRate: number;
	l2: number;
	seed: number;
	trainFraction: number;
	validationFraction: number;
	resumePolicy: string;
	datasetPath: string;
};

export type TrainingResult = {
	runDir: string;
	datasetPath?: string;
	policyPath: string;
	metricsPath: string;
	summary: Record<string, unknown> | null;
};

export type SupervisedSplitMetrics = {
	split?: string;
	rows?: number;
	sessions?: number;
	accuracy?: number;
	cross_entropy?: number;
	macro_f1?: number;
	predicted_pnl?: number;
	oracle_pnl?: number;
	always_normal_pnl?: number;
	always_skip_pnl?: number;
	always_invert_pnl?: number;
	oracle_regret?: number;
};

export type SupervisedCheckpointMetrics = {
	epoch?: number;
	backend?: string;
	device?: string;
	policy_path?: string;
	train?: SupervisedSplitMetrics;
	validation?: SupervisedSplitMetrics;
};

export type SupervisedTrainingSummary = {
	backend?: string;
	device?: string;
	epochs?: number;
	epochs_this_run?: number;
	start_epoch?: number;
	total_epochs?: number;
	checkpoint_every?: number;
	optimizer_state_saved?: boolean;
	checkpoints?: SupervisedCheckpointMetrics[];
	train?: SupervisedSplitMetrics;
	validation?: SupervisedSplitMetrics;
	holdout?: SupervisedSplitMetrics;
	leakage_check?: string;
};

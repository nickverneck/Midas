export type TrainMode = "ga" | "rl";
export type RlAlgorithm = "ppo" | "grpo";
export type MlBackend = "libtorch" | "burn" | "candle" | "mlx";
export type BuildProfile = "debug" | "release";
export type ConsoleLineType = "stdout" | "stderr" | "system" | "error";
export type ConsoleLine = { type: ConsoleLineType; text: string };
export type DataMode = "full" | "windowed";
export type StartChoice = "new" | "resume";
export type ParquetKey = "train-parquet" | "val-parquet" | "test-parquet";
export type FuturesPresetKey = "mes-micro" | "es-mini";

export type FileEntry = {
	name: string;
	path: string;
	kind: "dir" | "file";
};

export type FilePickerTarget =
	| { kind: "parquet"; mode: TrainMode; key: ParquetKey }
	| { kind: "checkpoint"; mode: TrainMode };

export type FitnessPoint = {
	gen: number;
	fitness: number;
};

export type FuturesPreset = {
	label: string;
	marginPerContract: number;
	contractMultiplier: number;
	note: string;
};

export type GaParams = {
	backend: MlBackend;
	outdir: string;
	"train-parquet": string;
	"val-parquet": string;
	"test-parquet": string;
	device: string;
	"batch-candidates": number;
	generations: number;
	"pop-size": number;
	workers: number;
	window: number;
	step: number;
	"initial-balance": number;
	"max-position": number;
	"margin-mode": string;
	"contract-multiplier": number;
	"margin-per-contract": string | number;
	"auto-close-minutes-before-close": number;
	"max-hold-bars-positive": number;
	"max-hold-bars-drawdown": number;
	"hold-duration-penalty": number;
	"hold-duration-penalty-growth": number;
	"hold-duration-penalty-positive-scale": number;
	"hold-duration-penalty-negative-scale": number;
	"selection-train-weight": number;
	"selection-eval-weight": number;
	"selection-gap-penalty": number;
	"elite-frac": number;
	"mutation-sigma": number;
	"init-sigma": number;
	hidden: number;
	layers: number;
	"eval-windows": number;
	"w-pnl": number;
	"w-sortino": number;
	"w-mdd": number;
	"drawdown-penalty": number;
	"drawdown-penalty-growth": number;
	"session-close-penalty": number;
	"flat-hold-penalty": number;
	"flat-hold-penalty-growth": number;
	"max-flat-hold-bars": number;
	"save-top-n": number;
	"save-every": number;
	"checkpoint-every": number;
};

export type RlParams = {
	algorithm: RlAlgorithm;
	backend: MlBackend;
	outdir: string;
	"train-parquet": string;
	"val-parquet": string;
	"test-parquet": string;
	device: string;
	window: number;
	step: number;
	epochs: number;
	"train-windows": number;
	"ppo-epochs": number;
	"group-size": number;
	"grpo-epochs": number;
	lr: number;
	gamma: number;
	lam: number;
	clip: number;
	"vf-coef": number;
	"ent-coef": number;
	dropout: number;
	hidden: number;
	layers: number;
	"eval-windows": number;
	"initial-balance": number;
	"max-position": number;
	"margin-mode": string;
	"contract-multiplier": number;
	"margin-per-contract": string | number;
	"auto-close-minutes-before-close": number;
	"max-hold-bars-positive": number;
	"max-hold-bars-drawdown": number;
	"hold-duration-penalty": number;
	"hold-duration-penalty-growth": number;
	"hold-duration-penalty-positive-scale": number;
	"hold-duration-penalty-negative-scale": number;
	"w-pnl": number;
	"w-sortino": number;
	"w-mdd": number;
	"log-interval": number;
	"checkpoint-every": number;
};

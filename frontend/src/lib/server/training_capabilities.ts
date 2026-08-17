import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';
import type {
	BackendCapability,
	BackendId,
	TrainingCapabilities
} from '$lib/components/training/types';
import {
    type CandleCudaProbeHooks,
	detectNvidiaComputeCapabilities,
	getCandleCudaPolicy,
	hasUsableMetalToolchain
} from './ml_env';

const hasTorchCudaSignal = (env: NodeJS.ProcessEnv) => {
	if (env.TORCH_CUDA_VERSION?.trim()) return true;

	const libtorchRoot = env.LIBTORCH?.trim();
	if (!libtorchRoot) return false;

	const libtorchLib = path.join(libtorchRoot, 'lib');
	const cudaLibraries =
		process.platform === 'win32'
			? ['torch_cuda.dll']
			: process.platform === 'darwin'
				? ['libtorch_cuda.dylib']
				: ['libtorch_cuda.so'];
	return cudaLibraries.some((library) => fs.existsSync(path.join(libtorchLib, library)));
};

const hasTorchMpsRuntime = (env: NodeJS.ProcessEnv) => {
	if (process.platform !== 'darwin') return false;

	const python = env.PYTHON?.trim() || 'python3';
	try {
		const output = execFileSync(
			python,
			[
				'-c',
				[
					'import torch',
					'mps = getattr(torch.backends, "mps", None)',
					'ok = bool(mps and mps.is_built() and mps.is_available())',
					'if ok: torch.zeros((1,), device="mps").item()',
					'print("1" if ok else "0")'
				].join('\n')
			],
			{ env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
		)
		return output.toString().trim() === '1';
	} catch {
		return false;
	}
};

export const getTrainingCapabilities = (
	env: NodeJS.ProcessEnv = process.env,
	probeHooks: CandleCudaProbeHooks = {}
): TrainingCapabilities => {
	const mac = process.platform === 'darwin';
	const computeCapabilities = detectNvidiaComputeCapabilities(env, probeHooks);
	const cudaAvailable = computeCapabilities.length > 0;
	const candleCudaPolicy = getCandleCudaPolicy(env, probeHooks);
	const burnCuda = cudaAvailable;
	const candleCuda = candleCudaPolicy.usable;
    const burnMps = mac && hasUsableMetalToolchain(env) && env.MIDAS_BURN_MLX !== '0';
    const torchCuda = cudaAvailable && hasTorchCudaSignal(env);
    const torchMps = mac && hasUsableMetalToolchain(env) && hasTorchMpsRuntime(env);

	const capability = (
		ga: boolean,
		rl: boolean,
		supervised: boolean,
		devices: BackendCapability['devices'],
		note: string
	): BackendCapability => ({ ga, rl, supervised, devices, note });

	const backends: Record<BackendId, BackendCapability> = {
		torch: capability(
			true,
			true,
			false,
			{ auto: true, cpu: true, cuda: torchCuda, mps: torchMps },
			'Libtorch legacy GA/RL runner; CUDA requires a usable NVIDIA GPU and a CUDA-enabled Torch/libtorch signal.'
		),
		burn: capability(
			true,
			true,
			true,
			{ auto: true, cpu: true, cuda: burnCuda, mps: burnMps },
			burnMps
				? 'GA, PPO/GRPO RL, and supervised learning are implemented; Burn MLX is available on this Mac.'
				: 'GA, PPO/GRPO RL, and supervised learning are implemented. Auto probes CUDA and safely falls back to deterministic burn-ndarray CPU; explicit CUDA requires a usable NVIDIA device.'
		),
		candle: capability(
			true,
			true,
			true,
			{ auto: true, cpu: true, cuda: candleCuda, mps: false },
			candleCudaPolicy.reason ??
				'Candle GA/RL and supervised CPU paths are implemented; Candle Metal is not wired.'
		),
		'cpu-linear': capability(
			false,
			false,
			true,
			{ auto: false, cpu: true, cuda: false, mps: false },
			'Working supervised reference implementation on CPU only.'
		)
	};

	return { platform: process.platform, backends };
};

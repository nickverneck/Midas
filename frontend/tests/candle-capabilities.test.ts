// @ts-nocheck
import { expect, test } from 'bun:test';
import {
	getCandleCudaPolicy,
	resolveBackendFeatures,
	type CandleCudaProbeHooks
} from '../src/lib/server/ml_env';
import { getTrainingCapabilities } from '../src/lib/server/training_capabilities';

const makeProbe = ({
	compute = '8.6',
	nvcc = 'Cuda compilation tools, release 12.8, V12.8.61',
	driver = 'NVIDIA-SMI 580.142    CUDA Version: 13.0',
	headers = true,
	resolveNvcc = true
} = {}) => {
	const calls: Array<{ file: string; args: string[] }> = [];
	const hooks: CandleCudaProbeHooks = {
		platform: 'linux',
		resolveExecutable: () => (resolveNvcc ? '/opt/cuda/bin/nvcc' : null),
		fileExists: (candidate) => headers && candidate === '/opt/cuda/include/cuda.h',
		execFileSync: (file, args) => {
			calls.push({ file, args });
			if (file === 'nvidia-smi' && args[0] === '--query-gpu=compute_cap') return `${compute}\n`;
			if (file === 'nvidia-smi' && args.length === 0) return driver;
			if (file === '/opt/cuda/bin/nvcc' && args[0] === '--version') return nvcc;
			throw new Error(`unexpected probe command: ${file} ${args.join(' ')}`);
		}
	};
	return { hooks, calls };
};

test('requires a verified nvcc toolkit and headers before enabling Candle CUDA', () => {
	const { hooks, calls } = makeProbe();
	const env = { PATH: '', CUDA_HOME: '/opt/cuda' };

	const policy = getCandleCudaPolicy(env, hooks);

	expect(policy.usable).toBe(true);
	expect(policy.toolchainVerified).toBe(true);
	expect(policy.toolkitVersion).toEqual({ major: 12, minor: 8 });
	expect(policy.driverCudaVersion).toEqual({ major: 13, minor: 0 });
	calls.length = 0;
	expect(resolveBackendFeatures('candle', env, 'auto', hooks)).toContain('backend-candle-cuda');
	expect(calls.map(({ file, args }) => `${file} ${args.join(' ')}`)).toEqual([
		'nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits',
		'/opt/cuda/bin/nvcc --version',
		'nvidia-smi '
	]);
});

test('does not advertise Candle CUDA from nvidia-smi alone', () => {
	const { hooks } = makeProbe({ resolveNvcc: false });
	const policy = getCandleCudaPolicy({ PATH: '' }, hooks);

	expect(policy.usable).toBe(false);
	expect(policy.toolchainVerified).toBe(false);
	expect(policy.reason).toMatch(/nvcc --version/);
	expect(resolveBackendFeatures('candle', {}, 'auto', hooks)).not.toContain('backend-candle-cuda');
	const capabilities = getTrainingCapabilities({}, hooks);
	expect(capabilities.backends.candle.devices.cuda).toBe(false);
});

test('rejects an incomplete CUDA toolkit with no headers', () => {
	const { hooks } = makeProbe({ headers: false });
	const policy = getCandleCudaPolicy({ CUDA_HOME: '/opt/cuda' }, hooks);

	expect(policy.usable).toBe(false);
	expect(policy.reason).toMatch(/CUDA headers/);
});

test('rejects an old or driver-incompatible toolkit', () => {
	const old = getCandleCudaPolicy(
		{ CUDA_HOME: '/opt/cuda' },
		makeProbe({ nvcc: 'Cuda compilation tools, release 10.2, V10.2.89' }).hooks
	);
	const newerThanDriver = getCandleCudaPolicy(
		{ CUDA_HOME: '/opt/cuda' },
		makeProbe({
			nvcc: 'Cuda compilation tools, release 13.0, V13.0.10',
			driver: 'NVIDIA-SMI 535.0    CUDA Version: 12.2'
		}).hooks
	);

	expect(old.reason).toMatch(/CUDA 11 or newer/);
	expect(newerThanDriver.reason).toMatch(/newer than the NVIDIA driver's CUDA/);
});

test('keeps the Pascal block ahead of toolchain probing', () => {
	const { hooks, calls } = makeProbe({ compute: '6.1' });
	const policy = getCandleCudaPolicy({ CUDA_HOME: '/opt/cuda' }, hooks);

	expect(policy.usable).toBe(false);
	expect(policy.pascalBlocked).toBe(true);
	expect(policy.reason).toMatch(/Pascal/);
	expect(calls).toHaveLength(1);
});

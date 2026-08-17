import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

export type MlBackend = 'libtorch' | 'burn' | 'candle' | 'mlx';

type CommandProbe = (
    file: string,
    args: string[],
    options: {
        env: NodeJS.ProcessEnv;
        encoding: 'utf8';
        stdio: ['ignore', 'pipe', 'ignore'];
    }
) => string | Buffer;

export type CandleCudaProbeHooks = {
    execFileSync?: CommandProbe;
    fileExists?: (candidate: string) => boolean;
    resolveExecutable?: (name: string, env: NodeJS.ProcessEnv) => string | null;
    platform?: NodeJS.Platform;
};

const resolveExecutableOnPath = (name: string, env: NodeJS.ProcessEnv) => {
    const executable = process.platform === 'win32' && !name.endsWith('.exe') ? `${name}.exe` : name;
    if (path.isAbsolute(executable)) return fs.existsSync(executable) ? executable : null;
    for (const root of [env.CUDA_HOME, env.CUDA_PATH].filter(Boolean) as string[]) {
        const candidate = path.join(root, 'bin', executable);
        if (fs.existsSync(candidate)) return candidate;
    }
    for (const directory of (env.PATH ?? '').split(path.delimiter).filter(Boolean)) {
        const candidate = path.join(directory, executable);
        if (fs.existsSync(candidate)) return candidate;
    }
    return null;
};

const defaultCandleCudaProbeHooks: Required<CandleCudaProbeHooks> = {
    execFileSync: execFileSync as unknown as CommandProbe,
    fileExists: fs.existsSync,
    resolveExecutable: (name, env) => resolveExecutableOnPath(name, env),
    platform: process.platform
};

const mergeCandleCudaProbeHooks = (hooks: CandleCudaProbeHooks = {}): Required<CandleCudaProbeHooks> => ({
    execFileSync: hooks.execFileSync ?? defaultCandleCudaProbeHooks.execFileSync,
    fileExists: hooks.fileExists ?? defaultCandleCudaProbeHooks.fileExists,
    resolveExecutable: hooks.resolveExecutable ?? defaultCandleCudaProbeHooks.resolveExecutable,
    platform: hooks.platform ?? defaultCandleCudaProbeHooks.platform
});

const uniqueNonEmpty = (values: Array<string | undefined | null>) =>
    Array.from(
        new Set(
            values
                .filter((value): value is string => Boolean(value && value.trim()))
                .map((value) => path.resolve(value))
        )
    );

const cudaToolkitRoots = (
    env: NodeJS.ProcessEnv,
    nvccPath: string | null,
    platform: NodeJS.Platform
) => {
    const roots = uniqueNonEmpty([
        nvccPath && path.dirname(path.dirname(nvccPath)),
        env.CUDA_HOME,
        env.CUDA_PATH,
        platform === 'win32' ? env.ProgramFiles && path.join(env.ProgramFiles, 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v12.0') : '/usr/local/cuda'
    ]);
    return roots;
};

const hasCudaHeader = (root: string, fileExists: (candidate: string) => boolean) =>
    fileExists(path.join(root, 'include', 'cuda.h')) ||
    fileExists(path.join(root, 'include', 'cuda_runtime_api.h'));

const parseCudaVersion = (output: string) => {
    const match = output.match(/(?:release|CUDA Version:|Version:|V)\s*(\d+)\.(\d+)/i);
    if (!match) return null;
    const major = Number(match[1]);
    const minor = Number(match[2]);
    return Number.isInteger(major) && Number.isInteger(minor) ? { major, minor } : null;
};

const compareCudaVersions = (left: { major: number; minor: number }, right: { major: number; minor: number }) =>
    left.major - right.major || left.minor - right.minor;

const detectNvidiaDriverCudaVersion = (
    env: NodeJS.ProcessEnv,
    hooks: Required<CandleCudaProbeHooks>
) => {
    try {
        const output = hooks.execFileSync('nvidia-smi', [], {
            env,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        });
        return parseCudaVersion(output.toString());
    } catch {
        return null;
    }
};

export const detectNvidiaComputeCapabilities = (
    env: NodeJS.ProcessEnv = process.env,
    hooks: CandleCudaProbeHooks = {}
) => {
    const probe = mergeCandleCudaProbeHooks(hooks);
    try {
        return probe.execFileSync(
            'nvidia-smi',
            ['--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
            { env, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
        )
            .toString()
            .split(/\r?\n/)
            .map((value) => value.trim())
            .filter(Boolean);
    } catch {
        return [] as string[];
    }
};

export const hasUsableNvidiaGpu = (env: NodeJS.ProcessEnv = process.env) =>
    detectNvidiaComputeCapabilities(env).length > 0;

export const isPascalComputeCapability = (value: string) => {
    const normalized = value.trim().toLowerCase();
    return normalized === '6.1' || normalized === 'sm_61' || normalized === 'sm61' || normalized === 'sm-61';
};

// Candle's CUDA kernels are not enabled for Pascal (sm_61) in this project.
// Keep this policy in one helper so capability reporting, env overrides, and
// Cargo feature selection cannot advertise different answers for one host.
// The probe deliberately stops at metadata/toolchain checks; it never builds
// Candle or starts a training process from the capabilities endpoint.
export const getCandleCudaPolicy = (
    env: NodeJS.ProcessEnv = process.env,
    hooks: CandleCudaProbeHooks = {}
) => {
    const probe = mergeCandleCudaProbeHooks(hooks);
    const computeCapabilities = detectNvidiaComputeCapabilities(env, probe);
    if (computeCapabilities.length === 0) {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason:
                'Candle CUDA requires a verified NVIDIA compute capability, but nvidia-smi could not report one; use CPU or install/enable nvidia-smi.'
        } as const;
    }
    if (computeCapabilities.some(isPascalComputeCapability)) {
        return {
            usable: false,
            pascalBlocked: true,
            toolchainVerified: false,
            reason:
                'Candle CUDA is disabled for NVIDIA Pascal (sm_61, including GTX 1080 Ti) because the current Candle CUDA kernels are not compatible; choose CPU or Burn/libtorch for GPU training.'
        } as const;
    }

    const nvccName = probe.platform === 'win32' ? 'nvcc.exe' : 'nvcc';
    const nvccPath = probe.resolveExecutable(nvccName, env);
    const nvccCommand = nvccPath ?? nvccName;
    let nvccOutput: string;
    try {
        nvccOutput = probe
            .execFileSync(nvccCommand, ['--version'], {
                env,
                encoding: 'utf8',
                stdio: ['ignore', 'pipe', 'ignore']
            })
            .toString();
    } catch {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason:
                'Candle CUDA is unavailable because nvcc --version could not run. Install a CUDA toolkit, put its bin directory on PATH, or set CUDA_HOME/CUDA_PATH; use CPU if only the NVIDIA driver is installed.'
        } as const;
    }

    const toolkitVersion = parseCudaVersion(nvccOutput);
    if (!toolkitVersion) {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason:
                'Candle CUDA is unavailable because nvcc returned no recognizable CUDA toolkit version; verify that CUDA_HOME/CUDA_PATH and PATH point to a complete CUDA toolkit.'
        } as const;
    }
    if (toolkitVersion.major < 11) {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason: `Candle CUDA requires CUDA 11 or newer; nvcc reports ${toolkitVersion.major}.${toolkitVersion.minor}. Install a newer toolkit or choose CPU.`
        } as const;
    }

    const toolkitRoots = cudaToolkitRoots(env, nvccPath, probe.platform);
    const nvccRoot = nvccPath ? path.dirname(path.dirname(nvccPath)) : null;
    const headerRoot = nvccRoot
        ? hasCudaHeader(nvccRoot, probe.fileExists)
            ? nvccRoot
            : null
        : toolkitRoots.find((root) => hasCudaHeader(root, probe.fileExists));
    if (!headerRoot) {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason:
                'Candle CUDA is unavailable because CUDA headers (cuda.h or cuda_runtime_api.h) were not found beside nvcc. Install the full CUDA toolkit and set CUDA_HOME/CUDA_PATH to its root; use CPU for a driver-only installation.'
        } as const;
    }

    const driverCudaVersion = detectNvidiaDriverCudaVersion(env, probe);
    if (driverCudaVersion && compareCudaVersions(toolkitVersion, driverCudaVersion) > 0) {
        return {
            usable: false,
            pascalBlocked: false,
            toolchainVerified: false,
            reason: `Candle CUDA is unavailable because nvcc ${toolkitVersion.major}.${toolkitVersion.minor} is newer than the NVIDIA driver's CUDA ${driverCudaVersion.major}.${driverCudaVersion.minor} support. Upgrade the driver or select CPU.`
        } as const;
    }

    return {
        usable: true,
        pascalBlocked: false,
        toolchainVerified: true,
        toolkitRoot: headerRoot,
        toolkitVersion,
        driverCudaVersion,
        reason: null
    } as const;
};

export const hasUsableCandleNvidiaGpu = (
    env: NodeJS.ProcessEnv = process.env,
    hooks: CandleCudaProbeHooks = {}
) => getCandleCudaPolicy(env, hooks).usable;

export const resolveProjectRoot = () => {
    const cwd = process.cwd();
    if (fs.existsSync(path.join(cwd, 'Cargo.toml'))) {
        return cwd;
    }
    const parent = path.resolve(cwd, '..');
    if (fs.existsSync(path.join(parent, 'Cargo.toml'))) {
        return parent;
    }
    return cwd;
};

export const loadDotEnv = (root: string) => {
    const envPath = path.join(root, '.env');
    if (!fs.existsSync(envPath)) return {};
    const out: Record<string, string> = {};
    const raw = fs.readFileSync(envPath, 'utf8');
    for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
        const [key, ...rest] = trimmed.split('=');
        const value = rest.join('=').trim().replace(/^['"]|['"]$/g, '');
        if (key) out[key.trim()] = value;
    }
    return out;
};

const findVenvTorchRoot = (venvDir: string) => {
    const isWindows = process.platform === 'win32';
    if (isWindows) {
        const torchRoot = path.join(venvDir, 'Lib', 'site-packages', 'torch');
        return fs.existsSync(torchRoot) ? torchRoot : null;
    }
    const libDir = path.join(venvDir, 'lib');
    if (!fs.existsSync(libDir)) return null;
    const entries = fs.readdirSync(libDir, { withFileTypes: true });
    for (const entry of entries) {
        if (!entry.isDirectory() || !entry.name.startsWith('python')) continue;
        const torchRoot = path.join(libDir, entry.name, 'site-packages', 'torch');
        if (fs.existsSync(torchRoot)) return torchRoot;
    }
    return null;
};

const attachVirtualEnv = (root: string, baseEnv: NodeJS.ProcessEnv) => {
    const env = { ...baseEnv };
    const isWindows = process.platform === 'win32';
    const venvDir = path.join(root, '.venv');
    const venvBin = path.join(venvDir, isWindows ? 'Scripts' : 'bin');
    const venvPython = path.join(venvBin, isWindows ? 'python.exe' : 'python');
    if (!fs.existsSync(venvPython)) {
        return env;
    }

    env.VIRTUAL_ENV = env.VIRTUAL_ENV ?? venvDir;
    env.PATH = `${venvBin}${path.delimiter}${env.PATH ?? ''}`;
    env.PYTHON = env.PYTHON ?? venvPython;
    return env;
};

const attachCudaEnv = (baseEnv: NodeJS.ProcessEnv) => {
    const env = { ...baseEnv };
    const nvccName = process.platform === 'win32' ? 'nvcc.exe' : 'nvcc';
    const cudaRoot = [env.CUDA_HOME, env.CUDA_PATH, '/usr/local/cuda'].find(
        (candidate) => candidate && fs.existsSync(path.join(candidate, 'bin', nvccName))
    );
    if (!cudaRoot) {
        return env;
    }
    const cudaBin = path.join(cudaRoot, 'bin');
    env.CUDA_HOME = env.CUDA_HOME ?? cudaRoot;
    env.CUDA_PATH = env.CUDA_PATH ?? cudaRoot;
    env.PATH = `${cudaBin}${path.delimiter}${env.PATH ?? ''}`;
    if (process.platform === 'linux' && !env.NVCC_CCBIN) {
        const supportedHostCompiler = ['/usr/bin/gcc-14', '/usr/local/bin/gcc-14'].find((candidate) =>
            fs.existsSync(candidate)
        );
        if (supportedHostCompiler) env.NVCC_CCBIN = supportedHostCompiler;
    }
    return env;
};

// Cudarc falls back to the newest CUDA API when nvcc is unavailable. That can
// ask an older-but-compatible NVIDIA driver for symbols it does not export.
// Prefer the API level advertised by the installed driver for Burn's dynamic
// CUDA backend; an explicit user setting always wins.
const attachBurnCudaApiVersion = (baseEnv: NodeJS.ProcessEnv) => {
    const env = { ...baseEnv };
    if (env.CUDARC_CUDA_VERSION || process.platform !== 'linux') return env;
    try {
        const output = execFileSync('nvidia-smi', [], {
            env,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        }).toString();
        const match = output.match(/CUDA Version:\s*(\d+)\.(\d+)/i);
        if (match) {
            const major = Number(match[1]);
            const minor = Number(match[2]);
            if (Number.isInteger(major) && Number.isInteger(minor)) {
                env.CUDARC_CUDA_VERSION = normalizeCudarcCudaVersion(major, minor);
            }
        }
    } catch {
        // Let Cudarc's normal build-time detection handle hosts without nvidia-smi.
    }
    return env;
};

/**
 * A Burn MLX build is only useful when the host can actually find Apple's
 * Metal compiler. Keep this probe in the shared environment helper so the
 * capability endpoint and Cargo feature selection make the same decision.
 */
export const hasUsableMetalToolchain = (env: NodeJS.ProcessEnv = process.env) => {
    if (process.platform !== 'darwin') return false;
    try {
        execFileSync('xcrun', ['-sdk', 'macosx', '--find', 'metal'], {
            env,
            stdio: ['ignore', 'ignore', 'ignore']
        });
        return true;
    } catch {
        return false;
    }
};

// Cudarc encodes CUDA 12.8 as 12080 (major * 1000 + minor * 10), rather than
// concatenating the minor digits after a fixed zero.
export const normalizeCudarcCudaVersion = (major: number, minor: number) =>
    String(major * 1000 + minor * 10);

const attachLibtorchEnv = (root: string, baseEnv: NodeJS.ProcessEnv) => {
    const env = { ...baseEnv };
    const isWindows = process.platform === 'win32';
    const venvDir = path.join(root, '.venv');
    const venvBin = path.join(venvDir, isWindows ? 'Scripts' : 'bin');
    const venvPython = path.join(venvBin, isWindows ? 'python.exe' : 'python');
    if (!fs.existsSync(venvPython)) {
        return env;
    }

    env.LIBTORCH_USE_PYTORCH = env.LIBTORCH_USE_PYTORCH ?? '1';
    env.LIBTORCH_BYPASS_VERSION_CHECK = env.LIBTORCH_BYPASS_VERSION_CHECK ?? '1';
    if (process.platform === 'win32') {
        env.MIDAS_PRELOAD_TORCH = env.MIDAS_PRELOAD_TORCH ?? '1';
    }

    const python = env.PYTHON ?? venvPython;
    try {
        const torchRoot = require('child_process')
            .execFileSync(
                python,
                ['-c', 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)'],
                { env, stdio: ['ignore', 'pipe', 'ignore'] }
            )
            .toString()
            .trim();
        if (torchRoot) {
            if (env.LIBTORCH_USE_PYTORCH === '1') {
                env.LIBTORCH = torchRoot;
            } else {
                env.LIBTORCH = env.LIBTORCH ?? torchRoot;
            }
        }
    } catch {
        // Fall back to default env when torch isn't available in the venv.
    }

    if (!env.LIBTORCH || env.LIBTORCH_USE_PYTORCH === '1') {
        const torchRoot = findVenvTorchRoot(venvDir);
        if (torchRoot) {
            env.LIBTORCH = torchRoot;
        }
    }

    if (env.LIBTORCH) {
        const torchLib = path.join(env.LIBTORCH, 'lib');
        if (fs.existsSync(torchLib)) {
            const libKey =
                process.platform === 'darwin'
                    ? 'DYLD_LIBRARY_PATH'
                    : process.platform === 'linux'
                        ? 'LD_LIBRARY_PATH'
                        : 'PATH';
            env[libKey] = env[libKey]
                ? `${torchLib}${path.delimiter}${env[libKey]}`
                : torchLib;
            if (process.platform === 'darwin') {
                env.DYLD_FALLBACK_LIBRARY_PATH = env.DYLD_FALLBACK_LIBRARY_PATH
                    ? `${torchLib}${path.delimiter}${env.DYLD_FALLBACK_LIBRARY_PATH}`
                    : torchLib;
            }
        }
    }

    return env;
};

export const resolveTrainerEnv = (
    root: string,
    baseEnv: NodeJS.ProcessEnv,
    backend: MlBackend
) => {
    let env = attachCudaEnv(attachVirtualEnv(root, baseEnv));
    if (backend === 'burn') env = attachBurnCudaApiVersion(env);
    if (backend === 'libtorch') {
        return attachLibtorchEnv(root, env);
    }
    return env;
};

export const resolveBackendFeatures = (
    backend: MlBackend,
    env: NodeJS.ProcessEnv = process.env,
    runtime: string | undefined = 'auto',
    hooks: CandleCudaProbeHooks = {}
) => {
    const features: string[] = [];
    const normalizedRuntime = (runtime ?? 'auto').toLowerCase();
    switch (backend) {
        case 'libtorch':
            features.push('torch');
            break;
        case 'burn':
            features.push('backend-burn');
            if (
                process.platform === 'darwin' &&
                hasUsableMetalToolchain(env) &&
                env.MIDAS_BURN_MLX !== '0' &&
                (normalizedRuntime === 'mps' || normalizedRuntime === 'auto')
            ) {
                features.push('backend-burn-mlx');
            }
            if (
                env.MIDAS_BURN_CUDA === '1' ||
                normalizedRuntime === 'cuda' ||
                normalizedRuntime === 'cuda:0' ||
                (normalizedRuntime === 'auto' && hasUsableNvidiaGpu(env))
            ) {
                features.push('backend-burn-cuda');
            }
            break;
        case 'candle':
            features.push('backend-candle');
            if (process.platform === 'darwin' && env.MIDAS_CANDLE_ACCELERATE !== '0') {
                features.push('backend-candle-accelerate');
            }
            // An env override is only a request to opt into CUDA; it cannot
            // override the sm_61 safety policy.  Explicit runtime requests
            // are validated by the API before Cargo is spawned, while this
            // helper stays safe when called directly by build orchestration.
            if (
                hasUsableCandleNvidiaGpu(env, hooks) &&
                (env.MIDAS_CANDLE_CUDA === '1' ||
                    normalizedRuntime === 'cuda' ||
                    normalizedRuntime === 'cuda:0' ||
                    normalizedRuntime === 'auto')
            ) {
                features.push('backend-candle-cuda');
            }
            break;
        case 'mlx':
            features.push('backend-mlx');
            break;
    }
    return Array.from(new Set(features));
};

export const resolveCargoBin = (env: NodeJS.ProcessEnv) => {
    if (env.CARGO_BIN) return env.CARGO_BIN;
    if (process.platform === 'win32') {
        const cargoHome =
            env.CARGO_HOME ??
            (env.USERPROFILE ? path.join(env.USERPROFILE, '.cargo') : null);
        if (cargoHome) {
            const cargoPath = path.join(cargoHome, 'bin', 'cargo.exe');
            if (fs.existsSync(cargoPath)) return cargoPath;
        }
    }
    return 'cargo';
};

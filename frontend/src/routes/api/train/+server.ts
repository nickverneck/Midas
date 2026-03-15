import { spawn } from 'child_process';
import {
    loadDotEnv,
    resolveBackendFeatures,
    resolveCargoBin,
    resolveProjectRoot,
    resolveTrainerEnv,
    type MlBackend
} from '$lib/server/ml_env';

type TrainEngine = 'ga' | 'rl';
type BuildProfile = 'debug' | 'release';
const coerceBackend = (value: unknown): MlBackend => {
    switch (value) {
        case 'burn':
        case 'candle':
        case 'mlx':
        case 'libtorch':
            return value;
        default:
            return 'libtorch';
	}
};

const coerceProfile = (value: unknown): BuildProfile => (
	value === 'release' ? 'release' : 'debug'
);

const buildCliArgs = (params: Record<string, unknown>) => {
    const args: string[] = [];
    for (const [key, value] of Object.entries(params)) {
        if (value === true) {
            args.push(`--${key}`);
        } else if (value !== false && value !== null && value !== undefined && value !== '') {
            args.push(`--${key}`, String(value));
        }
    }
    return args;
};

import type { RequestEvent } from "@sveltejs/kit";

export const POST = async ({ request }: RequestEvent) => {
    const payload = await request.json();
    const engine = (payload?.engine ?? 'ga') as TrainEngine;
    const params = (payload?.params ?? payload ?? {}) as Record<string, unknown>;
    const profile = coerceProfile(payload?.profile);
    const backend = coerceBackend(params.backend);
    const { signal } = request;

    const root = resolveProjectRoot();
    const cliArgs = buildCliArgs(params);
    const dotenv = loadDotEnv(root);
    const env = resolveTrainerEnv(root, { ...process.env, ...dotenv }, backend);
    const runtime = typeof params.device === 'string' ? params.device : 'auto';
    const features = resolveBackendFeatures(backend, env, runtime);

    const command = resolveCargoBin(env);
    const cargoArgs = ['run'];
    if (profile === 'release') {
        cargoArgs.push('--release');
    }
    cargoArgs.push('--features', features.join(','));
    if (engine === 'rl') {
        cargoArgs.push('--bin', 'train_rl', '--', ...cliArgs);
    } else {
        cargoArgs.push('--bin', 'train_ga', '--', ...cliArgs);
    }

    const stream = new ReadableStream({
        start(controller) {
            const child = spawn(command, cargoArgs, {
                cwd: root,
                env
            });

            const onAbort = () => {
                if (!child.killed) {
                    child.kill('SIGTERM');
                }
            };

            signal.addEventListener('abort', onAbort);

            child.stdout.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stdout', content: data.toString() })}\n\n`);
            });

            child.stderr.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stderr', content: data.toString() })}\n\n`);
            });

            child.on('close', (code) => {
                signal.removeEventListener('abort', onAbort);
                controller.enqueue(`data: ${JSON.stringify({ type: 'exit', code })}\n\n`);
                controller.close();
            });

            child.on('error', (err) => {
                signal.removeEventListener('abort', onAbort);
                controller.enqueue(`data: ${JSON.stringify({ type: 'error', content: err.message })}\n\n`);
                controller.close();
            });
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    });
};

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

type TrainEngine = 'rust' | 'python';

const resolveProjectRoot = () => {
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

export const POST = async ({ request }) => {
    const payload = await request.json();
    const engine = (payload?.engine ?? 'rust') as TrainEngine;
    const params = (payload?.params ?? payload ?? {}) as Record<string, unknown>;
    const { signal } = request;

    const root = resolveProjectRoot();
    const cliArgs = buildCliArgs(params);

    let command = '';
    let args: string[] = [];
    if (engine === 'python') {
        command = 'python3';
        args = ['python/examples/train_hybrid.py', ...cliArgs];
    } else {
        command = 'cargo';
        args = ['run', '--bin', 'train_ga', '--features', 'tch', '--', ...cliArgs];
    }

    const stream = new ReadableStream({
        start(controller) {
            const child = spawn(command, args, {
                cwd: root,
                env: process.env
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

import { execFileSync } from 'child_process';
import path from 'path';
import {
    loadDotEnv,
    resolveCargoBin,
    resolveProjectRoot,
    resolveTrainerEnv
} from '$lib/server/ml_env';

export const GET = async () => {
    const root = resolveProjectRoot();
    const dotenv = loadDotEnv(root);
    const env = resolveTrainerEnv(root, { ...process.env, ...dotenv }, 'libtorch');
    const cargoCommand = resolveCargoBin(env);
    const torchArgs = ['run', '--bin', 'tch_mps_check', '--features', 'torch'];
    const mlxScript = path.join(root, 'python', 'examples', 'mlx_probe.py');
    const pythonCommand = env.PYTHON ?? 'python3';
    const xcrunCommand = 'xcrun';

    const torchLibDir = env.LIBTORCH ? path.join(env.LIBTORCH, 'lib') : null;
    const pathValue = env.PATH ?? '';
    const pathHasTorchLib = torchLibDir
        ? pathValue.toLowerCase().includes(torchLibDir.toLowerCase())
        : false;
    const envSnapshot = {
        LIBTORCH: env.LIBTORCH ?? null,
        LIBTORCH_USE_PYTORCH: env.LIBTORCH_USE_PYTORCH ?? null,
        LIBTORCH_BYPASS_VERSION_CHECK: env.LIBTORCH_BYPASS_VERSION_CHECK ?? null,
        TORCH_CUDA_VERSION: env.TORCH_CUDA_VERSION ?? null,
        PYTHON: env.PYTHON ?? null,
        VIRTUAL_ENV: env.VIRTUAL_ENV ?? null,
        TORCH_LIB_DIR: torchLibDir,
        PATH_HAS_TORCH_LIB: pathHasTorchLib ? '1' : '0',
        MLX_PROBE_SCRIPT: mlxScript,
        MLX_PYTHON: pythonCommand
    };

    let torchOk = false;
    let torchOutput = '';
    let torchError = '';

    try {
        torchOutput = execFileSync(cargoCommand, torchArgs, {
            cwd: root,
            env,
            encoding: 'utf8'
        });
        torchOk = true;
    } catch (err) {
        const error = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string };
        torchOutput = error.stdout ?? '';
        torchError = error.stderr ?? error.message;
    }

    let mlxOk = false;
    let mlxOutput = '';
    let mlxError = '';

    try {
        mlxOutput = execFileSync(pythonCommand, [mlxScript], {
            cwd: root,
            env,
            encoding: 'utf8'
        });
        mlxOk = true;
    } catch (err) {
        const error = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string };
        mlxOutput = error.stdout ?? '';
        mlxError = error.stderr ?? error.message;
    }

    let burnMlxToolchainOk = false;
    let burnMlxToolchainOutput = '';
    let burnMlxToolchainError = '';

    try {
        const cmakePath = execFileSync('which', ['cmake'], {
            cwd: root,
            env,
            encoding: 'utf8'
        }).trim();
        const metalPath = execFileSync(xcrunCommand, ['-sdk', 'macosx', '--find', 'metal'], {
            cwd: root,
            env,
            encoding: 'utf8'
        }).trim();
        const metalVersion = execFileSync(xcrunCommand, ['-sdk', 'macosx', 'metal', '-v'], {
            cwd: root,
            env,
            encoding: 'utf8'
        }).trim();
        burnMlxToolchainOutput = [
            `cmake: ${cmakePath || '(not found)'}`,
            `metal: ${metalPath || '(not found)'}`,
            metalVersion || '(no metal version output)'
        ].join('\n');
        burnMlxToolchainOk = true;
    } catch (err) {
        const error = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string };
        burnMlxToolchainOutput = error.stdout ?? '';
        burnMlxToolchainError = error.stderr ?? error.message;
    }

    const sections = [
        '=== libtorch ===',
        torchOutput.trim() || '(no output)',
        '=== mlx ===',
        mlxOutput.trim() || '(no output)',
        '=== burn-mlx toolchain ===',
        burnMlxToolchainOutput.trim() || '(no output)'
    ];
    const combinedOutput = sections.join('\n\n');
    const combinedError = [torchError.trim(), mlxError.trim(), burnMlxToolchainError.trim()]
        .filter(Boolean)
        .join('\n\n');

    if (torchOk) {
        return new Response(
            JSON.stringify({
                ok: true,
                command: cargoCommand,
                args: torchArgs,
                output: combinedOutput,
                env: envSnapshot,
                probes: {
                    libtorch: { ok: torchOk, output: torchOutput },
                    mlx: { ok: mlxOk, output: mlxOutput, error: mlxError },
                    burnMlxToolchain: {
                        ok: burnMlxToolchainOk,
                        output: burnMlxToolchainOutput,
                        error: burnMlxToolchainError
                    }
                }
            }),
            { headers: { 'Content-Type': 'application/json' } }
        );
    }

    return new Response(
        JSON.stringify({
            ok: false,
            command: cargoCommand,
            args: torchArgs,
            env: envSnapshot,
            stdout: combinedOutput,
            stderr: combinedError,
            error: torchError || 'Diagnostics failed',
            probes: {
                libtorch: { ok: torchOk, output: torchOutput, error: torchError },
                mlx: { ok: mlxOk, output: mlxOutput, error: mlxError },
                burnMlxToolchain: {
                    ok: burnMlxToolchainOk,
                    output: burnMlxToolchainOutput,
                    error: burnMlxToolchainError
                }
            }
        }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
};

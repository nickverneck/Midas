import { execFileSync } from 'child_process';
import {
    loadDotEnv,
    resolveCargoBin,
    resolveProjectRoot,
    resolveTorchEnv
} from '$lib/server/torch_env';

export const GET = async () => {
    const root = resolveProjectRoot();
    const dotenv = loadDotEnv(root);
    const env = resolveTorchEnv(root, { ...process.env, ...dotenv });
    const command = resolveCargoBin(env);
    const args = ['run', '--bin', 'tch_mps_check', '--features', 'torch'];

    const torchLibDir = env.LIBTORCH ? `${env.LIBTORCH}\\lib` : null;
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
        PATH_HAS_TORCH_LIB: pathHasTorchLib ? '1' : '0'
    };

    try {
        const output = execFileSync(command, args, { cwd: root, env, encoding: 'utf8' });
        return new Response(
            JSON.stringify({ ok: true, command, args, output, env: envSnapshot }),
            { headers: { 'Content-Type': 'application/json' } }
        );
    } catch (err) {
        const error = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string };
        return new Response(
            JSON.stringify({
                ok: false,
                command,
                args,
                env: envSnapshot,
                stdout: error.stdout ?? '',
                stderr: error.stderr ?? String(error),
                error: error.message
            }),
            { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
    }
};

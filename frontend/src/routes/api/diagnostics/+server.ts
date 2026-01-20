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

    try {
        const output = execFileSync(command, args, { cwd: root, env, encoding: 'utf8' });
        return new Response(
            JSON.stringify({ ok: true, command, args, output }),
            { headers: { 'Content-Type': 'application/json' } }
        );
    } catch (err) {
        const error = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string };
        return new Response(
            JSON.stringify({
                ok: false,
                command,
                args,
                stdout: error.stdout ?? '',
                stderr: error.stderr ?? String(error),
                error: error.message
            }),
            { status: 500, headers: { 'Content-Type': 'application/json' } }
        );
    }
};

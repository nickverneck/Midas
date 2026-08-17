import { getTrainingCapabilities } from '$lib/server/training_capabilities';
import { loadDotEnv, resolveProjectRoot, resolveTrainerEnv } from '$lib/server/ml_env';

export const GET = () => {
	const root = resolveProjectRoot();
	const env = resolveTrainerEnv(root, { ...process.env, ...loadDotEnv(root) }, 'libtorch');
	return new Response(JSON.stringify({ ok: true, capabilities: getTrainingCapabilities(env) }), {
		headers: {
			'Content-Type': 'application/json',
			'Cache-Control': 'no-store'
		}
	});
};

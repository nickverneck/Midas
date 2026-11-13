/** @param {unknown} summary */
export function luaReplayFailureMessage(summary) {
	if (!summary || typeof summary !== 'object') {
		return 'Lua replay exited successfully but did not produce a valid sweep summary.';
	}
	const candidate = /** @type {Record<string, any>} */ (summary);
	if (!Array.isArray(candidate.runs)) return 'Lua replay exited successfully but did not produce a valid sweep summary.';
	const runs = /** @type {any[]} */ (candidate.runs);
	if (runs.length !== 1) return `Lua prototype expected one replay run, but the summary contains ${runs.length}.`;
	const run = runs[0];
	if (run?.status === 'completed') {
		if (candidate.run_count !== 1 || candidate.completed_count !== 1 || candidate.failed_count !== 0 || candidate.skipped_count !== 0) {
			return 'Lua replay summary has inconsistent run counts; expected one completed, unfailed, unskipped run.';
		}
		if (typeof run.result_path !== 'string' || !run.result_path.trim()) return 'Lua replay completed without a result_path artifact.';
		return undefined;
	}
	const detail = typeof run?.error === 'string' && run.error.trim() ? `: ${run.error.trim()}` : '';
	return `Lua replay run ${typeof run?.run_id === 'string' ? run.run_id : '(unknown)'} finished with status ${typeof run?.status === 'string' ? run.status : 'unknown'}${detail}`.slice(0, 2000);
}

/** @param {string} operation */
export function supportsReplayResults(operation) {
	return operation === 'run' || operation === 'lua-run';
}

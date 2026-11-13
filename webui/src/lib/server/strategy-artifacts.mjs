import { resolve, relative, sep, isAbsolute } from 'node:path';

const jobIdPattern = /^[a-f0-9-]{36}$/;

/** Keep every native ranking discoverable as sweep-ranking.json while making
 * each result immutable and unique to its ranking job. */
/** @param {string} outputDir @param {string} rankingJobId */
export function rankingArtifactPath(outputDir, rankingJobId) {
	if (typeof outputDir !== 'string' || !jobIdPattern.test(rankingJobId)) throw new Error('Invalid ranking artifact location.');
	// The TUI scans from the trader root with a bounded depth of five. A
	// `sweep-rankings/<id>` subtree under outputDir is one level too deep once
	// outputDir is inside `.run/webui-strategy-jobs/<job>/output`.
	return resolve(outputDir, rankingJobId, 'sweep-ranking.json');
}

/** @param {string} root @param {string} candidate */
export function isWithinDirectory(root, candidate) {
	const rel = relative(resolve(root), resolve(candidate));
	return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel));
}

/** @param {Record<string, any>} saved @param {string} directoryName @param {string} jobsRoot */
export function isRecoverableJobRecord(saved, directoryName, jobsRoot) {
	return Boolean(saved && saved.id === directoryName && jobIdPattern.test(directoryName)
		&& typeof saved.specPath === 'string' && typeof saved.outputDir === 'string'
		&& isWithinDirectory(jobsRoot, saved.specPath) && isWithinDirectory(jobsRoot, saved.outputDir));
}

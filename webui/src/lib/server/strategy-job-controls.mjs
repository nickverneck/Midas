/** Reserve the single strategy-job slot across asynchronous job setup. */
/** @returns {{ reserve: () => (() => void) | undefined }} */
export function createActiveJobGate() {
	let occupied = false;
	return {
		reserve() {
			if (occupied) return undefined;
			occupied = true;
			let released = false;
			return () => {
				if (released) return;
				released = true;
				occupied = false;
			};
		}
	};
}

/** Keep cancellation checks adjacent to every child-process spawn. */
/** @param {{ cancelled?: boolean }} job @param {() => import('node:child_process').ChildProcess} start */
export function startUnlessCancelled(job, start) {
	if (job.cancelled) throw new Error('Strategy job was cancelled.');
	return start();
}

/** Keep a child's successful exit from hiding a cancellation received while it ran. */
/** @template T @param {{ cancelled?: boolean }} job @param {() => Promise<T>} run */
export async function runUnlessCancelled(job, run) {
	const result = await run();
	if (job.cancelled) throw new Error('Strategy job was cancelled.');
	return result;
}

/** @param {{ cancelled?: boolean }} job */
export function assertNotCancelled(job) {
	if (job.cancelled) throw new Error('Strategy job was cancelled.');
}

const BUSY_MESSAGE = 'The chart cache reader is busy. Retry shortly.';

/** Retry only the bridge's single-flight response, while this selection remains current. */
/** @param {typeof fetch} fetchImpl @param {string} url @param {() => boolean} isCurrent @param {(ms: number) => Promise<void>} [wait] */
export async function fetchStoredCachePage(fetchImpl, url, isCurrent, wait = ms => new Promise(resolve => setTimeout(resolve, ms))) {
	let delay = 100;
	for (let attempt = 0; ; attempt++) {
		if (!isCurrent()) return undefined;
		const response = await fetchImpl(url, { cache: 'no-store' });
		if (response.status !== 503 || attempt >= 20) return response;
		let body;
		try { body = await response.clone().json(); } catch { return response; }
		if (body?.error !== BUSY_MESSAGE) return response;
		if (!isCurrent()) return undefined;
		await wait(delay);
		delay = Math.min(delay * 2, 1000);
	}
}

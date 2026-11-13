export function fetchStoredCachePage(
	fetchImpl: typeof fetch,
	url: string,
	isCurrent: () => boolean,
	wait?: (ms: number) => Promise<void>
): Promise<Response | undefined>;

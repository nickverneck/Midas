import type { RequestHandler } from './$types';
import { apiError, assertLocalRequest } from '../../../lib/server/chart-cache';
import { LiveMarketError, liveMarketError, readLiveMarket } from '../../../lib/server/live-market';

export const GET: RequestHandler = async (event) => {
	try {
		assertLocalRequest(event);
		return Response.json(await readLiveMarket(), { headers: { 'cache-control': 'no-store' } });
	} catch (error) {
		return error instanceof LiveMarketError ? liveMarketError(error) : apiError(error);
	}
};

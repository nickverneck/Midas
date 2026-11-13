import type { RequestHandler } from './$types';
import { apiError, assertLocalRequest, runChartCache } from '../../../lib/server/chart-cache';

export const GET: RequestHandler = async (event) => {
	try {
		assertLocalRequest(event);
		return Response.json(await runChartCache([]), { headers: { 'cache-control': 'no-store' } });
	} catch (error) { return apiError(error); }
};

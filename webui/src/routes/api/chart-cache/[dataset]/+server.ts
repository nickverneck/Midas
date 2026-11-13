import type { RequestHandler } from './$types';
import { apiError, assertLocalRequest, BridgeError, runChartCache, validatePageQuery } from '../../../../lib/server/chart-cache';

export const GET: RequestHandler = async (event) => {
	try {
		assertLocalRequest(event);
		const { dataset } = event.params;
		if (!/^[a-f0-9]{64}$/.test(dataset)) throw new BridgeError(400, 'dataset must be a 64 character chart-cache dataset id.');
		const args = validatePageQuery(event.url);
		args.push('--dataset', dataset);
		return Response.json(await runChartCache(args), { headers: { 'cache-control': 'no-store' } });
	} catch (error) { return apiError(error); }
};

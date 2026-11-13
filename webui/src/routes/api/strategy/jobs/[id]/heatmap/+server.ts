import type { RequestHandler } from './$types';
import { getStrategyHeatmap, strategyApiError } from '../../../../../../lib/server/strategy-jobs';

export const GET: RequestHandler = async (event) => {
	try { return await getStrategyHeatmap(event); }
	catch (error) { return strategyApiError(error); }
};

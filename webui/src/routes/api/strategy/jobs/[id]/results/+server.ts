import type { RequestHandler } from './$types';
import { getStrategyReplayResults, strategyApiError } from '../../../../../../lib/server/strategy-jobs';

export const GET: RequestHandler = async (event) => {
	try { return await getStrategyReplayResults(event); }
	catch (error) { return strategyApiError(error); }
};

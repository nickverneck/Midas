import type { RequestHandler } from './$types';
import { cancelStrategyJob, getStrategyJob, strategyApiError } from '../../../../../lib/server/strategy-jobs';

export const GET: RequestHandler = async (event) => {
	try { return await getStrategyJob(event); }
	catch (error) { return strategyApiError(error); }
};

export const DELETE: RequestHandler = async (event) => {
	try { return await cancelStrategyJob(event); }
	catch (error) { return strategyApiError(error); }
};

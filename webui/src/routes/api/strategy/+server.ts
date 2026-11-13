import type { RequestHandler } from './$types';
import { getStrategyCatalog, submitStrategyJob, strategyApiError } from '../../../lib/server/strategy-jobs';

export const GET: RequestHandler = async (event) => {
	try { return await getStrategyCatalog(event); }
	catch (error) { return strategyApiError(error); }
};

export const POST: RequestHandler = async (event) => {
	try { return await submitStrategyJob(event); }
	catch (error) { return strategyApiError(error); }
};

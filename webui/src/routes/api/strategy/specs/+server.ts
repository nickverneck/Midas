import type { RequestHandler } from './$types';
import { createStrategySweep, strategyApiError } from '../../../../lib/server/strategy-jobs';

export const POST: RequestHandler = async (event) => {
	try { return await createStrategySweep(event); }
	catch (error) { return strategyApiError(error); }
};

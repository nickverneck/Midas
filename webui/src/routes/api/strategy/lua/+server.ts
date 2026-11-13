import type { RequestHandler } from './$types';
import { submitLuaPrototypeJob, strategyApiError } from '../../../../lib/server/strategy-jobs';

export const POST: RequestHandler = async (event) => {
	try { return await submitLuaPrototypeJob(event); }
	catch (error) { return strategyApiError(error); }
};

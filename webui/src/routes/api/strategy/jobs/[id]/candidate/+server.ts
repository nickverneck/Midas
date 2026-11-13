import type { RequestEvent } from '@sveltejs/kit';
import { getStrategyCandidates, saveStrategyCandidate, strategyApiError } from '../../../../../../lib/server/strategy-jobs';

export const GET = async (event: RequestEvent) => {
	try { return await getStrategyCandidates(event); }
	catch (error) { return strategyApiError(error); }
};

export const POST = async (event: RequestEvent) => {
	try { return await saveStrategyCandidate(event); }
	catch (error) { return strategyApiError(error); }
};

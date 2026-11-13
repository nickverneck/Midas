import type { RequestHandler } from '@sveltejs/kit';
import { handleStandaloneMarketStream, standaloneMarketError } from '../../../../lib/server/standalone-market';

export const POST: RequestHandler = async (event) => {
	try { return await handleStandaloneMarketStream(event); }
	catch (error) { return standaloneMarketError(error); }
};

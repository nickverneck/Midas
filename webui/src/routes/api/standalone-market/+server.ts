import type { RequestHandler } from './$types';
import { handleStandaloneMarket, standaloneMarketError } from '../../../lib/server/standalone-market';

export const POST: RequestHandler = async (event) => {
	try {
		const payload = await handleStandaloneMarket(event);
		return Response.json(payload, { headers: { 'cache-control': 'no-store' } });
	} catch (error) { return standaloneMarketError(error); }
};

import type { FuturesPreset, FuturesPresetKey } from "./types";

export const defaultWindowBars = 7 * 24 * 60;
export const defaultStepBars = 128;

export const nativeSelectClass =
	"border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]";

export const detailsCardClass = "rounded-lg border bg-background/60 p-4";
export const detailsGridClass = "mt-4 grid gap-4 md:grid-cols-2";
export const futuresPresetSummary =
	"MES uses $50 margin and 5x multiplier; ES uses $500 margin and 50x multiplier. Auto-close stays at 5 minutes before close.";

export const futuresPresets: Record<FuturesPresetKey, FuturesPreset> = {
	"mes-micro": {
		label: "MES Micro",
		marginPerContract: 50,
		contractMultiplier: 5,
		note: "NinjaTrader MES intraday"
	},
	"es-mini": {
		label: "ES Mini",
		marginPerContract: 500,
		contractMultiplier: 50,
		note: "NinjaTrader ES intraday"
	}
};

export type ProviderBar = { ts_ns: string; open: number; high: number; low: number; close: number; volume?: number };

/** Stream updates contain complete snapshots; preserve their order and duplicate timestamps. */
export function providerSnapshotBars(bars: ProviderBar[]): ProviderBar[] {
	return bars.map((bar) => ({ ...bar }));
}

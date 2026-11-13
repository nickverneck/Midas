/** Invalidates async UI results when their owning view or selection changes. */
export class RequestEpoch {
	private current = 0;

	capture(): number { return this.current; }
	invalidate(): void { this.current += 1; }
	isCurrent(epoch: number): boolean { return epoch === this.current; }
}

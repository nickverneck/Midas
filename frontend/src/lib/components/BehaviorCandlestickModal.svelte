<script lang="ts">
	import { onDestroy, onMount } from 'svelte';

	type BehaviorRow = Record<string, any>;
	type Candle = {
		index: number;
		open: number;
		high: number;
		low: number;
		close: number;
		action: string;
		row: BehaviorRow;
	};
	type Props = {
		open?: boolean;
		title?: string;
		rows?: BehaviorRow[];
		onClose?: () => void;
	};

	let { open = false, title = 'Behavior Candlesticks', rows = [], onClose }: Props = $props();

	let container: HTMLDivElement;
	let canvas: HTMLCanvasElement;
	let ctx: CanvasRenderingContext2D | null = null;
	let resizeObserver: ResizeObserver | null = null;
	let raf = 0;

	let playing = $state(false);
	let speedMs = $state(250);
	let currentIndex = $state(0);
	let lastCount = 0;

	const toNumber = (value: unknown): number | null => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};

	const normalizeAction = (value: unknown) =>
		typeof value === 'string' ? value.trim().toLowerCase() : '';

	const formatNumber = (value: unknown, digits = 4) => {
		const num = toNumber(value);
		return num === null ? '—' : num.toFixed(digits);
	};

	const formatTimestamp = (row: BehaviorRow) => {
		const raw =
			row.datetime_ns ??
			row.datetime ??
			row.date ??
			row.timestamp ??
			row.datetime_ns ??
			null;
		if (raw === null || raw === undefined) return '—';
		if (typeof raw === 'string' && raw.trim() !== '') return raw;
		const num = toNumber(raw);
		if (num === null) return '—';
		let ms = num;
		if (num > 1e14) {
			ms = Math.round(num / 1e6);
		} else if (num > 1e12) {
			ms = num;
		} else if (num > 1e9) {
			ms = num * 1000;
		}
		const date = new Date(ms);
		if (Number.isNaN(date.getTime())) return String(raw);
		return date.toISOString().replace('T', ' ').replace('Z', '');
	};

	let candles = $derived.by(() => {
		const parsed = rows
			.map((row, index) => {
				const openVal = toNumber(row.open);
				const highVal = toNumber(row.high);
				const lowVal = toNumber(row.low);
				const closeVal = toNumber(row.close);
				if (openVal === null || highVal === null || lowVal === null || closeVal === null) {
					return null;
				}
				return {
					index,
					open: openVal,
					high: highVal,
					low: lowVal,
					close: closeVal,
					action: normalizeAction(row.action),
					row
				} as Candle;
			})
			.filter((item): item is Candle => item !== null);
		return parsed;
	});

	let currentCandle = $derived.by(() => (candles[currentIndex] ? candles[currentIndex] : null));

	const syncCanvasSize = () => {
		if (!canvas || !container) return { width: 0, height: 0, ratio: 1 };
		const rect = container.getBoundingClientRect();
		const ratio = window.devicePixelRatio || 1;
		const width = Math.max(1, Math.floor(rect.width));
		const height = Math.max(1, Math.floor(rect.height));
		if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
			canvas.width = Math.floor(width * ratio);
			canvas.height = Math.floor(height * ratio);
			canvas.style.width = `${width}px`;
			canvas.style.height = `${height}px`;
		}
		if (!ctx) {
			ctx = canvas.getContext('2d');
		}
		if (ctx) {
			ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
		}
		return { width, height, ratio };
	};

	const drawChart = () => {
		if (!open || !canvas || !container) return;
		const { width, height } = syncCanvasSize();
		if (!ctx || width === 0 || height === 0) return;

		ctx.clearRect(0, 0, width, height);
		const padding = { top: 16, right: 20, bottom: 28, left: 54 };
		const plotWidth = width - padding.left - padding.right;
		const plotHeight = height - padding.top - padding.bottom;

		if (candles.length === 0) {
			ctx.fillStyle = '#64748b';
			ctx.font = '12px sans-serif';
			ctx.fillText('No candlestick data available.', padding.left, padding.top + 20);
			return;
		}

		const highs = candles.map((c) => c.high);
		const lows = candles.map((c) => c.low);
		let maxVal = Math.max(...highs);
		let minVal = Math.min(...lows);
		if (maxVal - minVal < 1e-6) {
			maxVal += 1;
			minVal -= 1;
		}
		const range = maxVal - minVal;
		const scaleY = (value: number) =>
			padding.top + ((maxVal - value) / range) * plotHeight;

		ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
		ctx.lineWidth = 1;
		const gridLines = 4;
		for (let i = 0; i <= gridLines; i += 1) {
			const y = padding.top + (plotHeight / gridLines) * i;
			ctx.beginPath();
			ctx.moveTo(padding.left, y);
			ctx.lineTo(padding.left + plotWidth, y);
			ctx.stroke();
		}

		ctx.fillStyle = '#64748b';
		ctx.font = '11px sans-serif';
		for (let i = 0; i <= gridLines; i += 1) {
			const value = maxVal - (range / gridLines) * i;
			const y = padding.top + (plotHeight / gridLines) * i + 4;
			ctx.fillText(value.toFixed(2), 6, y);
		}

		const step = plotWidth / candles.length;
		const candleWidth = Math.max(2, Math.min(step * 0.7, 16));

		for (let i = 0; i < candles.length; i += 1) {
			const candle = candles[i];
			const x = padding.left + i * step + step / 2;
			const openY = scaleY(candle.open);
			const closeY = scaleY(candle.close);
			const highY = scaleY(candle.high);
			const lowY = scaleY(candle.low);
			const up = candle.close >= candle.open;
			const color = up ? '#22c55e' : '#ef4444';

			ctx.strokeStyle = color;
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(x, highY);
			ctx.lineTo(x, lowY);
			ctx.stroke();

			ctx.fillStyle = color;
			const bodyTop = Math.min(openY, closeY);
			const bodyHeight = Math.max(1, Math.abs(openY - closeY));
			ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);

			if (candle.action === 'buy' || candle.action === 'sell' || candle.action === 'revert') {
				const markerY = closeY;
				const markerColor =
					candle.action === 'buy'
						? '#22c55e'
						: candle.action === 'sell'
							? '#ef4444'
							: '#f97316';
				ctx.fillStyle = markerColor;
				ctx.strokeStyle = 'rgba(15, 23, 42, 0.5)';
				if (candle.action === 'buy') {
					ctx.beginPath();
					ctx.moveTo(x, markerY - 8);
					ctx.lineTo(x - 6, markerY + 5);
					ctx.lineTo(x + 6, markerY + 5);
					ctx.closePath();
					ctx.fill();
				} else if (candle.action === 'sell') {
					ctx.beginPath();
					ctx.moveTo(x, markerY + 8);
					ctx.lineTo(x - 6, markerY - 5);
					ctx.lineTo(x + 6, markerY - 5);
					ctx.closePath();
					ctx.fill();
				} else {
					ctx.fillRect(x - 5, markerY - 5, 10, 10);
				}
			}
		}

		if (currentIndex >= 0 && currentIndex < candles.length) {
			const x = padding.left + currentIndex * step + step / 2;
			ctx.strokeStyle = 'rgba(59, 130, 246, 0.7)';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(x, padding.top);
			ctx.lineTo(x, padding.top + plotHeight);
			ctx.stroke();
		}
	};

	const scheduleDraw = () => {
		if (!open) return;
		if (raf) cancelAnimationFrame(raf);
		raf = requestAnimationFrame(drawChart);
	};

	const stepPrev = () => {
		currentIndex = Math.max(0, currentIndex - 1);
	};

	const stepNext = () => {
		currentIndex = Math.min(candles.length - 1, currentIndex + 1);
	};

	const togglePlay = () => {
		if (!candles.length) return;
		playing = !playing;
	};

	$effect(() => {
		if (!open) {
			playing = false;
			return;
		}
		if (candles.length !== lastCount) {
			lastCount = candles.length;
			currentIndex = 0;
		} else if (currentIndex >= candles.length) {
			currentIndex = Math.max(0, candles.length - 1);
		}
	});

	$effect(() => {
		if (!open) return;
		scheduleDraw();
	});

	$effect(() => {
		if (!open || !playing) return;
		const interval = setInterval(() => {
			if (currentIndex >= candles.length - 1) {
				playing = false;
				return;
			}
			currentIndex += 1;
		}, speedMs);
		return () => clearInterval(interval);
	});

	onMount(() => {
		return () => {
			if (resizeObserver) resizeObserver.disconnect();
		};
	});

	$effect(() => {
		if (!open || !container) return;
		if (resizeObserver) resizeObserver.disconnect();
		resizeObserver = new ResizeObserver(() => scheduleDraw());
		resizeObserver.observe(container);
		return () => resizeObserver?.disconnect();
	});

	onDestroy(() => {
		if (raf) cancelAnimationFrame(raf);
	});
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex bg-black/60"
		role="dialog"
		aria-modal="true"
		aria-label={title}
	>
		<button
			type="button"
			class="absolute inset-0"
			onclick={() => onClose?.()}
			aria-label="Close candlestick viewer"
		></button>
		<div class="relative z-10 flex h-full w-full flex-col gap-4 overflow-hidden rounded-none border bg-background p-4 shadow-xl">
			<div class="flex flex-wrap items-start justify-between gap-3">
				<div>
					<div class="text-xs uppercase tracking-wide text-muted-foreground">Behavior Candlesticks</div>
					<div class="text-lg font-semibold">{title}</div>
					<div class="text-xs text-muted-foreground">{candles.length.toLocaleString()} bars</div>
				</div>
				<button
					type="button"
					class="rounded-md border px-3 py-1 text-xs hover:bg-muted"
					onclick={() => onClose?.()}
				>
					Close
				</button>
			</div>

			<div class="flex flex-wrap items-center gap-2 text-xs">
				<button
					type="button"
					class="rounded-md border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50"
					onclick={togglePlay}
					disabled={candles.length === 0}
				>
					{playing ? 'Pause' : 'Play'}
				</button>
				<button
					type="button"
					class="rounded-md border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50"
					onclick={stepPrev}
					disabled={candles.length === 0 || currentIndex <= 0}
				>
					Prev
				</button>
				<button
					type="button"
					class="rounded-md border px-3 py-1 text-xs hover:bg-muted disabled:opacity-50"
					onclick={stepNext}
					disabled={candles.length === 0 || currentIndex >= candles.length - 1}
				>
					Next
				</button>
				<div class="flex items-center gap-2 text-xs">
					<span class="text-muted-foreground">Speed</span>
					<select
						class="rounded-md border bg-background px-2 py-1 text-xs"
						bind:value={speedMs}
					>
						<option value={800}>Slow</option>
						<option value={400}>Normal</option>
						<option value={200}>Fast</option>
						<option value={80}>Turbo</option>
					</select>
				</div>
				<div class="text-xs text-muted-foreground">
					Step {Math.min(currentIndex + 1, candles.length)} / {candles.length}
				</div>
			</div>

			<div class="flex min-h-0 flex-1 flex-col gap-4 xl:flex-row">
				<div class="flex min-h-0 flex-1 flex-col gap-3">
					<div class="flex items-center gap-3">
						<input
							class="w-full"
							type="range"
							min="0"
							max={Math.max(candles.length - 1, 0)}
							step="1"
							value={currentIndex}
							oninput={(event) => {
								const target = event.currentTarget as HTMLInputElement;
								currentIndex = Number(target.value);
							}}
							disabled={candles.length === 0}
						/>
					</div>
					<div class="flex-1 overflow-hidden rounded-lg border bg-background">
						<div class="h-full w-full min-h-0 p-2" bind:this={container}>
							<canvas class="h-full w-full" bind:this={canvas}></canvas>
						</div>
					</div>
					<div class="flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
						<div class="flex items-center gap-2">
							<span class="h-2 w-2 rounded-full bg-emerald-500"></span>
							<span>Buy</span>
						</div>
						<div class="flex items-center gap-2">
							<span class="h-2 w-2 rounded-full bg-rose-500"></span>
							<span>Sell</span>
						</div>
						<div class="flex items-center gap-2">
							<span class="h-2 w-2 rounded-full bg-orange-500"></span>
							<span>Revert</span>
						</div>
						<div class="flex items-center gap-2">
							<span class="h-2 w-2 rounded-full bg-slate-300"></span>
							<span>Hold hidden</span>
						</div>
					</div>
				</div>

				<div class="w-full max-w-sm min-h-0 space-y-4 overflow-auto">
					<div class="rounded-lg border p-3">
						<div class="text-xs uppercase text-muted-foreground">Current Step</div>
						{#if currentCandle}
							<div class="mt-3 space-y-2 text-xs">
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Action</span>
									<span class="font-semibold">{currentCandle.action || 'hold'}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Data idx</span>
									<span>{currentCandle.row.data_idx ?? currentCandle.row.idx ?? '—'}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Window</span>
									<span>{currentCandle.row.window ?? currentCandle.row.window_idx ?? '—'}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Time</span>
									<span class="font-mono text-[11px]">{formatTimestamp(currentCandle.row)}</span>
								</div>
							</div>
						{:else}
							<div class="mt-3 text-xs text-muted-foreground">No data loaded.</div>
						{/if}
					</div>

					<div class="rounded-lg border p-3">
						<div class="text-xs uppercase text-muted-foreground">Price</div>
						{#if currentCandle}
							<div class="mt-3 space-y-2 text-xs">
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Open</span>
									<span>{formatNumber(currentCandle.open)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">High</span>
									<span>{formatNumber(currentCandle.high)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Low</span>
									<span>{formatNumber(currentCandle.low)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Close</span>
									<span>{formatNumber(currentCandle.close)}</span>
								</div>
							</div>
						{:else}
							<div class="mt-3 text-xs text-muted-foreground">—</div>
						{/if}
					</div>

					<div class="rounded-lg border p-3">
						<div class="text-xs uppercase text-muted-foreground">Account</div>
						{#if currentCandle}
							<div class="mt-3 space-y-2 text-xs">
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Equity</span>
									<span>{formatNumber(currentCandle.row.equity_after)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Cash</span>
									<span>{formatNumber(currentCandle.row.cash)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Unrealized</span>
									<span>{formatNumber(currentCandle.row.unrealized_pnl)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Realized</span>
									<span>{formatNumber(currentCandle.row.realized_pnl)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">PnL Change</span>
									<span>{formatNumber(currentCandle.row.pnl_change)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Realized Change</span>
									<span>{formatNumber(currentCandle.row.realized_pnl_change)}</span>
								</div>
							</div>
						{:else}
							<div class="mt-3 text-xs text-muted-foreground">—</div>
						{/if}
					</div>

					<div class="rounded-lg border p-3">
						<div class="text-xs uppercase text-muted-foreground">Totals</div>
						{#if currentCandle}
							<div class="mt-3 space-y-2 text-xs">
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">PnL</span>
									<span>{formatNumber(currentCandle.row.pnl)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Realized PnL</span>
									<span>{formatNumber(currentCandle.row.pnl_realized)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Total PnL</span>
									<span>{formatNumber(currentCandle.row.pnl_total)}</span>
								</div>
								<div class="flex items-center justify-between">
									<span class="text-muted-foreground">Reward</span>
									<span>{formatNumber(currentCandle.row.reward)}</span>
								</div>
							</div>
						{:else}
							<div class="mt-3 text-xs text-muted-foreground">—</div>
						{/if}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

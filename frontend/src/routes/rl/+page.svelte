<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from "$lib/components/ui/card";
	import * as Tabs from "$lib/components/ui/tabs";
	import { Input } from "$lib/components/ui/input";
	import { Button } from "$lib/components/ui/button";
	import { Badge } from "$lib/components/ui/badge";
	import GaChart from "$lib/components/GaChart.svelte";

	type ChartTab = 'fitness' | 'performance' | 'drawdown' | 'loss';
	
	type RlPoint = {
		epoch: number;
		fitness: number | null;
		trainRet: number | null;
		trainPnl: number | null;
		trainSortino: number | null;
		trainDrawdown: number | null;
		evalRet: number | null;
		evalPnl: number | null;
		evalSortino: number | null;
		evalDrawdown: number | null;
		policyLoss: number | null;
		valueLoss: number | null;
		entropy: number | null;
		totalLoss: number | null;
	};

	const logChunk = 1000;

	let chartTab = $state<ChartTab>('fitness');
	let logDir = $state('runs_rl');
	let activeLogDir = $state('runs_rl');
	let logMap = $state(new Map<number, RlPoint>());
	let loading = $state(false);
	let loadingMore = $state(false);
	let doneLoading = $state(false);
	let nextOffset = $state(0);
	let error = $state('');
	let loadToken = 0;

	const toNumber = (value: unknown): number | null => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};

	const normalizeLogDir = (value: string) => value.trim() || 'runs_rl';
	const buildLogsUrl = (offset: number, dir: string) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir,
			log: 'rl'
		});
		return `/api/logs?${params.toString()}`;
	};

	const updateFromChunk = (rows: Array<Record<string, unknown>>) => {
		if (!rows || rows.length === 0) return;
		const next = new Map(logMap);
		for (const row of rows) {
			if (!row || typeof row !== 'object') continue;
			const epoch = toNumber(row.epoch);
			if (epoch === null) continue;
			next.set(epoch, {
				epoch,
				fitness: toNumber(row.fitness),
				trainRet: toNumber(row.train_ret_mean),
				trainPnl: toNumber(row.train_pnl),
				trainSortino: toNumber(row.train_sortino),
				trainDrawdown: toNumber(row.train_drawdown),
				evalRet: toNumber(row.eval_ret_mean),
				evalPnl: toNumber(row.eval_pnl),
				evalSortino: toNumber(row.eval_sortino),
				evalDrawdown: toNumber(row.eval_drawdown),
				policyLoss: toNumber(row.policy_loss),
				valueLoss: toNumber(row.value_loss),
				entropy: toNumber(row.entropy),
				totalLoss: toNumber(row.total_loss)
			});
		}
		logMap = next;
	};

	function resetState() {
		logMap = new Map();
		loading = true;
		error = '';
		nextOffset = 0;
		doneLoading = false;
		loadingMore = false;
	}

	async function fetchLogs() {
		const token = ++loadToken;
		resetState();
		try {
			const dir = normalizeLogDir(logDir);
			activeLogDir = dir;
			const res = await fetch(buildLogsUrl(0, dir));
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to fetch logs (${res.status})`);
			}
			const payload = await res.json();
			if (token !== loadToken) return;
			updateFromChunk(Array.isArray(payload.data) ? payload.data : []);
			nextOffset = payload.nextOffset || 0;
			doneLoading = Boolean(payload.done);
			if (!doneLoading) scheduleLoadMore(token, dir);
		} catch (e) {
			console.error('Failed to fetch logs', e);
			error = e instanceof Error ? e.message : String(e);
		} finally {
			if (token === loadToken) loading = false;
		}
	}

	async function scheduleLoadMore(token: number, dir: string) {
		if (doneLoading || loadingMore || token !== loadToken || dir !== activeLogDir) return;
		loadingMore = true;
		setTimeout(async () => {
			if (token !== loadToken || dir !== activeLogDir) {
				loadingMore = false;
				return;
			}
			try {
				const res = await fetch(buildLogsUrl(nextOffset, dir));
				if (!res.ok) throw new Error(`Failed to fetch logs (${res.status})`);
				const payload = await res.json();
				if (token !== loadToken || dir !== activeLogDir) return;
				updateFromChunk(Array.isArray(payload.data) ? payload.data : []);
				if (typeof payload.nextOffset === 'number') {
					nextOffset = payload.nextOffset;
				}
				doneLoading = Boolean(payload.done);
			} catch (e) {
				if (token === loadToken) {
					error = e instanceof Error ? e.message : String(e);
				}
			} finally {
				loadingMore = false;
				if (!doneLoading) scheduleLoadMore(token, dir);
			}
		}, 200);
	}

	onMount(fetchLogs);

	let epochData = $derived.by(() =>
		Array.from(logMap.values()).sort((a, b) => a.epoch - b.epoch)
	);
	let latest = $derived.by(() => (epochData.length > 0 ? epochData[epochData.length - 1] : null));

	const toSeries = (picker: (row: RlPoint) => number | null) =>
		epochData.map((row) => ({ x: row.epoch, y: picker(row) }));

	const chartOptions = {
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			}
		}
	};

	let fitnessChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Fitness (Eval)',
				data: toSeries((row) => row.fitness),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.35)',
				tension: 0.1
			},
			{
				label: 'Eval PnL',
				data: toSeries((row) => row.evalPnl),
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.3)',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	}));

	let performanceChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Train PnL',
				data: toSeries((row) => row.trainPnl),
				borderColor: 'rgb(148, 163, 184)',
				backgroundColor: 'rgba(148, 163, 184, 0.25)',
				tension: 0.1
			},
			{
				label: 'Eval PnL',
				data: toSeries((row) => row.evalPnl),
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.3)',
				tension: 0.1
			},
			{
				label: 'Eval Sortino',
				data: toSeries((row) => row.evalSortino),
				borderColor: 'rgb(168, 85, 247)',
				backgroundColor: 'rgba(168, 85, 247, 0.2)',
				borderDash: [4, 4],
				yAxisID: 'y1',
				tension: 0.1
			}
		]
	}));

	let drawdownChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Train Drawdown',
				data: toSeries((row) => row.trainDrawdown),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.3)',
				tension: 0.1
			},
			{
				label: 'Eval Drawdown',
				data: toSeries((row) => row.evalDrawdown),
				borderColor: 'rgb(248, 113, 113)',
				backgroundColor: 'rgba(248, 113, 113, 0.2)',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	}));

	let lossChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Policy Loss',
				data: toSeries((row) => row.policyLoss),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.3)',
				tension: 0.1
			},
			{
				label: 'Value Loss',
				data: toSeries((row) => row.valueLoss),
				borderColor: 'rgb(16, 185, 129)',
				backgroundColor: 'rgba(16, 185, 129, 0.25)',
				tension: 0.1
			},
			{
				label: 'Entropy',
				data: toSeries((row) => row.entropy),
				borderColor: 'rgb(251, 191, 36)',
				backgroundColor: 'rgba(251, 191, 36, 0.2)',
				borderDash: [4, 4],
				yAxisID: 'y1',
				tension: 0.1
			}
		]
	}));
</script>

<main class="p-8 space-y-8">
	<div class="flex flex-wrap items-center justify-between gap-4">
		<div>
			<h1 class="text-4xl font-bold tracking-tight">RL Analytics</h1>
			<p class="text-sm text-muted-foreground">PPO training metrics from Rust runs.</p>
		</div>
		<div class="flex items-center gap-2">
			<Input
				class="w-56"
				placeholder="runs_rl"
				bind:value={logDir}
			/>
			<Button onclick={fetchLogs} disabled={loading}>
				{loading ? 'Loading...' : 'Reload'}
			</Button>
		</div>
	</div>

	{#if error}
		<div class="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
			{error}
		</div>
	{/if}

	<div class="grid gap-6 xl:grid-cols-3">
		<Card.Root class="xl:col-span-2">
			<Card.Header>
				<Card.Title>Training Curves</Card.Title>
				<Card.Description>
					{activeLogDir} · {epochData.length.toLocaleString()} epochs
				</Card.Description>
			</Card.Header>
			<Card.Content>
				<Tabs.Root bind:value={chartTab} class="w-full">
					<Tabs.List class="grid w-full grid-cols-2 lg:grid-cols-4 lg:w-[640px] mb-4">
						<Tabs.Trigger value="fitness">Fitness</Tabs.Trigger>
						<Tabs.Trigger value="performance">Performance</Tabs.Trigger>
						<Tabs.Trigger value="drawdown">Drawdown</Tabs.Trigger>
						<Tabs.Trigger value="loss">Loss</Tabs.Trigger>
					</Tabs.List>
					<Tabs.Content value="fitness">
						<div class="h-[320px]">
							<GaChart data={fitnessChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="performance">
						<div class="h-[320px]">
							<GaChart data={performanceChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="drawdown">
						<div class="h-[320px]">
							<GaChart data={drawdownChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="loss">
						<div class="h-[320px]">
							<GaChart data={lossChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
				</Tabs.Root>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header>
				<Card.Title>Latest Snapshot</Card.Title>
				<Card.Description>Most recent epoch summary.</Card.Description>
			</Card.Header>
			<Card.Content>
				{#if latest}
					<div class="space-y-3 text-sm">
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Epoch</span>
							<span class="font-semibold">{latest.epoch}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Fitness</span>
							<span class="font-semibold">{latest.fitness ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval PnL</span>
							<span>{latest.evalPnl ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Sortino</span>
							<span>{latest.evalSortino ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Drawdown</span>
							<span>{latest.evalDrawdown ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Policy Loss</span>
							<span>{latest.policyLoss ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Value Loss</span>
							<span>{latest.valueLoss ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Entropy</span>
							<span>{latest.entropy ?? '—'}</span>
						</div>
					</div>
				{:else}
					<div class="text-sm text-muted-foreground">No RL logs yet.</div>
				{/if}
			</Card.Content>
			{#if loading}
				<Card.Footer>
					<Badge variant="outline" class="animate-pulse">Loading</Badge>
				</Card.Footer>
			{/if}
		</Card.Root>
	</div>
</main>

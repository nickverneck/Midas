<script lang="ts">
	import { CheckCircle2, CircleDashed, LoaderCircle, Octagon, Play, Square } from 'lucide-svelte';
	import type { BackendId, DeviceId, PreparedArtifact, StartMode, TrainingResult } from './types';

	type Props = {
		artifact: PreparedArtifact | null;
		result: TrainingResult | null;
		backend: BackendId;
		device: DeviceId;
		startMode: StartMode;
		status: 'idle' | 'preparing' | 'prepared' | 'training' | 'trained' | 'error';
		error: string;
		canPrepare: boolean;
		canTrain: boolean;
		onPrepare: () => void;
		onTrain: () => void;
		onStop: () => void;
	};

	let { artifact, result, backend, device, startMode, status, error, canPrepare, canTrain, onPrepare, onTrain, onStop }: Props = $props();
	let schemaOpen = $state(false);
	const busy = $derived(status === 'preparing' || status === 'training');
	const number = new Intl.NumberFormat('en-US');
	const pnlNumber = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2, signDisplay: 'exceptZero' });

	const asRecord = (value: unknown): Record<string, unknown> | null =>
		typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
	const holdout = $derived(asRecord(result?.summary?.holdout));
	const leakage = $derived(
		typeof result?.summary?.leakage_check === 'string'
			? result.summary.leakage_check
			: artifact?.summary?.leakage_check ?? 'unknown'
	);

	const formatPnl = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? pnlNumber.format(value) : '—');
	const formatAccuracy = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—');
	const backendLabel = (value: unknown) => ({ 'cpu-linear': 'CPU reference', burn: 'Burn', candle: 'Candle', torch: 'Libtorch' }[String(value)] ?? String(value));
	const deviceLabel = (value: unknown) => ({ auto: 'Auto', cpu: 'CPU', cuda: 'CUDA', 'cuda:0': 'CUDA', mps: 'MPS' }[String(value)] ?? String(value));
	const selectedRuntime = $derived({
		backend: typeof result?.summary?.backend === 'string' ? result.summary.backend : backend,
		device: typeof result?.summary?.device === 'string' ? result.summary.device : device
	});
</script>

<aside class="flex min-h-full flex-col rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="artifact-title">
	<div class="flex items-start justify-between gap-3">
		<div>
			<h2 id="artifact-title" class="text-sm font-semibold">Dataset & results</h2>
			<p class="text-[11px] text-muted-foreground">Prepared artifact and holdout checks</p>
		</div>
		<span class="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-1 text-[10px] font-medium uppercase tracking-wide">
			{#if status === 'preparing' || status === 'training'}<LoaderCircle class="animate-spin" size={12} /> Running
			{:else if status === 'prepared' || status === 'trained'}<CheckCircle2 class="text-emerald-600" size={12} /> Ready
			{:else if status === 'error'}<Octagon class="text-destructive" size={12} /> Error
			{:else}<CircleDashed size={12} /> Waiting{/if}
		</span>
	</div>

	{#if artifact?.summary}
		<div class="mt-3 grid grid-cols-2 gap-1.5">
			<div class="rounded-lg border p-2"><span class="block text-[10px] uppercase text-muted-foreground">Event rows</span><b class="text-base">{number.format(artifact.summary.event_rows ?? 0)}</b></div>
			<div class="rounded-lg border p-2"><span class="block text-[10px] uppercase text-muted-foreground">Sessions</span><b class="text-base">{number.format(artifact.summary.session_count ?? 0)}</b></div>
			<div class="rounded-lg border p-2"><span class="block text-[10px] uppercase text-muted-foreground">Features</span><b class="text-base">{number.format(artifact.summary.feature_count ?? 0)}</b></div>
			<div class="rounded-lg border p-2"><span class="block text-[10px] uppercase text-muted-foreground">Leakage</span><b class="text-xs text-emerald-700">{artifact.summary.leakage_check ?? 'unknown'}</b></div>
		</div>
		<div class="mt-2 rounded-lg bg-muted/50 p-2.5 text-[11px]">
			<span class="block text-[10px] uppercase text-muted-foreground">Dataset artifact</span>
			<code class="mt-1 block break-all">{artifact.path}</code>
			<button type="button" onclick={() => (schemaOpen = !schemaOpen)} class="mt-1.5 text-[10px] font-medium underline underline-offset-2">{schemaOpen ? 'Hide' : 'Show'} feature schema</button>
			{#if schemaOpen}<p class="mt-1.5 max-h-24 overflow-auto break-all text-[10px] leading-4 text-muted-foreground">{artifact.summary.feature_schema}</p>{/if}
		</div>
	{:else if startMode === 'continue'}
		<div class="mt-3 rounded-lg border border-dashed p-3 text-center"><CircleDashed class="mx-auto text-muted-foreground" size={22} /><p class="mt-1 text-xs font-medium">{canTrain ? 'Ready to continue' : 'Select artifacts to continue'}</p><p class="mt-1 text-[10px] leading-4 text-muted-foreground">{canTrain ? 'The saved dataset and policy are selected.' : 'Choose the dataset and policy in the Continue panel.'}</p></div>
	{:else}
		<div class="mt-3 rounded-lg border border-dashed p-4 text-center"><CircleDashed class="mx-auto text-muted-foreground" size={24} /><p class="mt-1 text-xs font-medium">No prepared dataset</p><p class="mt-1 text-[10px] leading-4 text-muted-foreground">Prepare the source to create causal event rows.</p></div>
	{/if}

	{#if result}
		<div class="mt-2 rounded-lg border border-emerald-200 bg-emerald-50 p-2.5 text-emerald-950 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-50">
			<div class="flex items-center justify-between gap-2"><b class="text-xs">Training complete</b><span class="text-[10px]">{leakage === 'passed' ? 'Leakage passed' : `Leakage: ${leakage}`}</span></div>
			<div class="mt-2 grid grid-cols-2 gap-1.5 text-[10px]">
				<div class="rounded bg-white/60 p-1.5 dark:bg-black/20"><span class="block opacity-70">Holdout accuracy</span><b class="text-sm">{formatAccuracy(holdout?.accuracy)}</b></div>
				<div class="rounded bg-white/60 p-1.5 dark:bg-black/20"><span class="block opacity-70">Predicted PnL</span><b class="text-sm">{formatPnl(holdout?.predicted_pnl)}</b></div>
				<div class="rounded bg-white/60 p-1.5 dark:bg-black/20"><span class="block opacity-70">Always normal</span><b class="text-sm">{formatPnl(holdout?.always_normal_pnl)}</b></div>
				<div class="rounded bg-white/60 p-1.5 dark:bg-black/20"><span class="block opacity-70">Oracle PnL</span><b class="text-sm">{formatPnl(holdout?.oracle_pnl)}</b></div>
			</div>
			<div class="mt-2 space-y-1 text-[10px]">
				{#if result.datasetPath}<div><span class="opacity-70">Dataset </span><code class="break-all">{result.datasetPath}</code></div>{/if}
				<div><span class="opacity-70">Policy </span><code class="break-all">{result.policyPath}</code></div>
				<div><span class="opacity-70">Metrics </span><code class="break-all">{result.metricsPath}</code></div>
			</div>
		</div>
	{/if}
	{#if error}<p role="alert" class="mt-2 rounded-lg border border-destructive/30 bg-destructive/10 p-2.5 text-xs text-destructive">{error}</p>{/if}

	<div class="mt-3">
		{#if busy}
			<button type="button" onclick={onStop} class="inline-flex h-9 w-full items-center justify-center gap-2 rounded-md border border-destructive text-xs font-semibold text-destructive hover:bg-destructive/10"><Square size={13} fill="currentColor" /> Stop</button>
		{:else if artifact || startMode === 'continue'}
			<button type="button" onclick={onTrain} disabled={!canTrain} class="inline-flex h-9 w-full items-center justify-center gap-2 rounded-md bg-foreground text-xs font-semibold text-background disabled:opacity-40"><Play size={14} fill="currentColor" /> {startMode === 'continue' ? 'Continue training' : 'Train policy'}</button>
		{:else}
			<button type="button" onclick={onPrepare} disabled={!canPrepare} class="inline-flex h-9 w-full items-center justify-center gap-2 rounded-md bg-foreground text-xs font-semibold text-background disabled:opacity-40"><Play size={14} /> Prepare dataset</button>
		{/if}
		<p class="mt-1.5 text-center text-[10px] text-muted-foreground">{backendLabel(selectedRuntime.backend)} · {deviceLabel(selectedRuntime.device)} · bounded process · project-safe paths</p>
	</div>
</aside>

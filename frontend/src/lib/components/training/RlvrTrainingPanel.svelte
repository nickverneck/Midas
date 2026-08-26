<script lang="ts">
	import { CheckCircle2, Info, LoaderCircle, Play, ShieldCheck, Sparkles, Square, Terminal } from 'lucide-svelte';
	import SourceFilePicker from './SourceFilePicker.svelte';

	type LogKind = 'system' | 'stdout' | 'stderr' | 'error';
	type LogLine = { kind: LogKind; text: string };
	type SplitSummary = {
		event_count?: number;
		sum_pnl_usd?: number;
		oracle_pnl_usd?: number;
		oracle_capture?: number;
		normal_fraction?: number;
		invert_fraction?: number;
	};
	type RlvrMetrics = {
		selected_epoch?: number;
		selected_train?: SplitSummary;
		selected_validation?: SplitSummary;
		fixed_normal?: SplitSummary;
		fixed_invert?: SplitSummary;
		learned?: SplitSummary;
		split?: { train?: number; validation?: number; holdout?: number; purged?: number; sessions?: number };
		reward_scale_usd?: number;
		reward_normalization?: string;
		leakage_check?: string;
	};
	type SseMessage = {
		type?: string;
		content?: string;
		code?: number | null;
		metrics?: RlvrMetrics;
		input?: string;
		runDir?: string;
		metricsPath?: string;
		policyPath?: string;
	};

	let input = $state('.run/training/gcz6-1m-regime-persistence-events.parquet');
	let outdir = $state('.run/event-rlvr/runs/gcz6-rlvr');
	let epochs = $state(1000);
	let checkpointEvery = $state(250);
	let hidden = $state(32);
	let layers = $state(1);
	let learningRate = $state(0.001);
	let l2 = $state(0.0001);
	let entropyCoefficient = $state(0.01);
	let seed = $state(42);
	let trainFraction = $state(0.6);
	let validationFraction = $state(0.2);
	let purgeBars = $state(30);
	let rewardNormalization = $state<'none' | 'train-mean-abs'>('train-mean-abs');
	let running = $state(false);
	let error = $state('');
	let exitCode = $state<number | null>(null);
	let logs = $state<LogLine[]>([]);
	let metrics = $state<RlvrMetrics | null>(null);
	let artifactPaths = $state<{ runDir?: string; metricsPath?: string; policyPath?: string }>({});
	let controller: AbortController | null = null;

	const inputClass = 'h-9 min-w-0 w-full rounded-md border bg-background px-2.5 text-xs sm:text-sm';
	const selectClass = `${inputClass} pr-7`;
	const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2, signDisplay: 'exceptZero' });
	const integer = new Intl.NumberFormat('en-US');

	const canRun = $derived(!running && input.trim().length > 0 && outdir.trim().length > 0);
	const statusLabel = $derived(running ? 'Running' : metrics ? 'Complete' : error ? 'Needs attention' : 'Ready');

	const numberValue = (event: Event) => Number((event.currentTarget as HTMLInputElement).value);

	const appendLog = (kind: LogKind, text: string) => {
		const parts = text.split(/\r?\n/).filter((part) => part.length > 0);
		if (parts.length === 0) return;
		logs = [...logs, ...parts.map((part) => ({ kind, text: part }))].slice(-180);
	};

	const readError = async (response: Response) => {
		const body = await response.text().catch(() => '');
		try {
			const parsed = JSON.parse(body) as { error?: string };
			return parsed.error || body || `RLVR request failed (${response.status})`;
		} catch {
			return body || `RLVR request failed (${response.status})`;
		}
	};

	const parseEvent = (payload: string) => {
		let event: SseMessage;
		try {
			event = JSON.parse(payload) as SseMessage;
		} catch {
			appendLog('error', 'Received malformed RLVR output.');
			return;
		}
		if (event.type === 'stdout' || event.type === 'stderr') {
			appendLog(event.type, event.content ?? '');
		} else if (event.type === 'start') {
			appendLog('system', `Started RLVR for ${event.input ?? input}.`);
		} else if (event.type === 'complete') {
			metrics = event.metrics ?? null;
			artifactPaths = { runDir: event.runDir, metricsPath: event.metricsPath, policyPath: event.policyPath };
			appendLog('system', `Artifacts written to ${event.runDir ?? outdir}.`);
		} else if (event.type === 'error') {
			error = event.content ?? 'RLVR runner error';
			appendLog('error', error);
		} else if (event.type === 'exit') {
			exitCode = typeof event.code === 'number' ? event.code : null;
			if (event.code !== 0) error = `RLVR exited with code ${event.code ?? 'unknown'}.`;
			appendLog('system', `Process exited with code ${event.code ?? 'unknown'}.`);
			running = false;
		}
	};

	const start = async () => {
		if (running) {
			controller?.abort();
			return;
		}
		if (!canRun) {
			error = 'Choose an event parquet and an output directory.';
			return;
		}

		error = '';
		exitCode = null;
		metrics = null;
		artifactPaths = {};
		logs = [];
		running = true;
		controller = new AbortController();

		try {
			const response = await fetch('/api/rlvr/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				signal: controller.signal,
				body: JSON.stringify({
					input: input.trim(),
					outdir: outdir.trim(),
					epochs,
					checkpoint_every: checkpointEvery,
					hidden,
					layers,
					learning_rate: learningRate,
					l2,
					entropy_coefficient: entropyCoefficient,
					seed,
					train_fraction: trainFraction,
					validation_fraction: validationFraction,
					purge_bars: purgeBars,
					reward_normalization: rewardNormalization
				})
			});
			if (!response.ok || !response.body) throw new Error(await readError(response));

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			while (true) {
				const { value, done } = await reader.read();
				buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
				const messages = buffer.split('\n\n');
				buffer = messages.pop() ?? '';
				for (const message of messages) {
					const data = message.split('\n').find((line) => line.startsWith('data: '));
					if (data) parseEvent(data.slice(6));
				}
				if (done) break;
			}
			if (buffer.startsWith('data: ')) parseEvent(buffer.slice(6));
		} catch (reason) {
			if (reason instanceof DOMException && reason.name === 'AbortError') {
				appendLog('system', 'RLVR stopped by user.');
			} else {
				error = reason instanceof Error ? reason.message : String(reason);
				appendLog('error', error);
			}
		} finally {
			running = false;
			controller = null;
		}
	};

	const formatPnl = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? number.format(value) : '—');
	const formatCount = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? integer.format(value) : '—');
	const formatCapture = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—');
</script>

<main class="training-workspace mx-auto w-full max-w-7xl px-3 py-2 sm:px-5 sm:py-3">
	<header class="mb-3 flex flex-wrap items-start justify-between gap-3">
		<div class="flex items-start gap-2">
			<div class="grid size-9 shrink-0 place-items-center rounded-lg bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-200"><Sparkles size={17} /></div>
			<div>
				<div class="mb-0.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Event reinforcement</div>
				<h1 class="text-xl font-semibold tracking-tight sm:text-2xl">RLVR normal / invert gate</h1>
				<p class="mt-0.5 max-w-2xl text-xs leading-5 text-muted-foreground">Optimize the exact verifier reward for both choices at every EMA event, then select the best validation checkpoint before touching holdout.</p>
			</div>
		</div>
		<div class="inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-[11px] text-muted-foreground"><ShieldCheck size={13} class="text-emerald-600" /> Causal inputs only</div>
	</header>

	<div class="mb-3 grid gap-3 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.75fr)]">
		<section class="space-y-3" aria-labelledby="rlvr-config-title">
			<div class="rounded-xl border border-violet-200 bg-violet-50 p-3 text-violet-950 dark:border-violet-900 dark:bg-violet-950/40 dark:text-violet-50 sm:p-4">
				<div class="flex items-start gap-2"><Info size={15} class="mt-0.5 shrink-0" /><div><h2 class="text-sm font-semibold">What this trainer does</h2><p class="mt-1 text-[11px] leading-5">RLVR is a two-action event contextual bandit: <b>normal</b> follows the crossover and <b>invert</b> reverses it. It is not the native four-action, sequential Trader RL environment. The future interval is used only by the verifier reward during training; it is never sent as a feature.</p></div></div>
			</div>

			<div class="rounded-xl border bg-card p-3 shadow-xs sm:p-4">
				<div class="mb-3 flex items-center justify-between gap-2"><div><h2 id="rlvr-config-title" class="text-sm font-semibold">Dataset and run</h2><p class="text-[11px] text-muted-foreground">Use a supervised-event-v1 parquet with normal-invert labels.</p></div><span class="rounded-full bg-muted px-2 py-1 text-[10px] font-medium">RLVR</span></div>
				<div class="grid gap-3 sm:grid-cols-2">
					<div class="sm:col-span-2"><label class="mb-1 block text-[11px] font-medium" for="rlvr-input">Causal event parquet</label><SourceFilePicker id="rlvr-input" value={input} onChange={(value) => (input = value)} disabled={running} initialDir=".run/training" extensions={['parquet']} label="Choose causal event parquet" buttonLabel="Browse events" placeholder=".run/training/*-events.parquet" emptyText="No event parquet files in this folder." /><p class="mt-1 text-[10px] leading-4 text-muted-foreground">The server accepts any project-confined parquet, then the Rust runner verifies its schema and rejects future/label-derived features.</p></div>
					<label class="text-[11px] font-medium sm:col-span-2" for="rlvr-outdir">Output directory<input id="rlvr-outdir" class={`${inputClass} mt-1`} value={outdir} oninput={(event) => (outdir = (event.currentTarget as HTMLInputElement).value)} disabled={running} placeholder=".run/event-rlvr/runs/gcz6-rlvr" /><span class="mt-1 block text-[10px] font-normal text-muted-foreground">Must stay under <code>.run/event-rlvr/runs</code>.</span></label>
				</div>
			</div>

			<div class="rounded-xl border bg-card p-3 shadow-xs sm:p-4">
				<div class="mb-3"><h2 class="text-sm font-semibold">Optimization controls</h2><p class="text-[11px] text-muted-foreground">The defaults match the GC diagnostic runs.</p></div>
				<div class="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
					<label class="text-[11px] font-medium" for="rlvr-epochs">Epochs<input id="rlvr-epochs" class={`${inputClass} mt-1`} type="number" min="1" max="100000" value={epochs} oninput={(event) => (epochs = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-checkpoint">Checkpoint every<input id="rlvr-checkpoint" class={`${inputClass} mt-1`} type="number" min="0" max="100000" value={checkpointEvery} oninput={(event) => (checkpointEvery = numberValue(event))} disabled={running} /><span class="mt-1 block text-[10px] font-normal text-muted-foreground">0 disables curves</span></label>
					<label class="text-[11px] font-medium" for="rlvr-hidden">Hidden units<input id="rlvr-hidden" class={`${inputClass} mt-1`} type="number" min="1" max="4096" value={hidden} oninput={(event) => (hidden = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-layers">Hidden layers<input id="rlvr-layers" class={`${inputClass} mt-1`} type="number" min="1" max="32" value={layers} oninput={(event) => (layers = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-learning-rate">Learning rate<input id="rlvr-learning-rate" class={`${inputClass} mt-1`} type="number" min="0.00000001" max="1" step="any" value={learningRate} oninput={(event) => (learningRate = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-l2">L2<input id="rlvr-l2" class={`${inputClass} mt-1`} type="number" min="0" max="1" step="any" value={l2} oninput={(event) => (l2 = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-entropy">Entropy coefficient<input id="rlvr-entropy" class={`${inputClass} mt-1`} type="number" min="0" max="1" step="any" value={entropyCoefficient} oninput={(event) => (entropyCoefficient = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-seed">Seed<input id="rlvr-seed" class={`${inputClass} mt-1`} type="number" min="0" value={seed} oninput={(event) => (seed = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-train-fraction">Train fraction<input id="rlvr-train-fraction" class={`${inputClass} mt-1`} type="number" min="0.05" max="0.9" step="0.05" value={trainFraction} oninput={(event) => (trainFraction = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-validation-fraction">Validation fraction<input id="rlvr-validation-fraction" class={`${inputClass} mt-1`} type="number" min="0.05" max="0.9" step="0.05" value={validationFraction} oninput={(event) => (validationFraction = numberValue(event))} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="rlvr-purge-bars">Purge bars<input id="rlvr-purge-bars" class={`${inputClass} mt-1`} type="number" min="0" max="100000" value={purgeBars} oninput={(event) => (purgeBars = numberValue(event))} disabled={running} /><span class="mt-1 block text-[10px] font-normal text-muted-foreground">Removes overlap at split edges</span></label>
					<label class="text-[11px] font-medium" for="rlvr-reward-normalization">Reward normalization<select id="rlvr-reward-normalization" class={`${selectClass} mt-1`} bind:value={rewardNormalization} disabled={running}><option value="train-mean-abs">Train mean absolute reward</option><option value="none">None</option></select></label>
				</div>
			</div>
		</section>

		<aside class="flex min-h-full flex-col gap-3 rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="rlvr-run-title">
			<div class="flex items-start justify-between gap-2"><div><h2 id="rlvr-run-title" class="text-sm font-semibold">Run RLVR</h2><p class="text-[11px] text-muted-foreground">SSE output stays visible while the process runs.</p></div><span class="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-1 text-[10px] font-medium">{#if running}<LoaderCircle size={11} class="animate-spin" />{:else if metrics}<CheckCircle2 size={11} class="text-emerald-600" />{:else}<Sparkles size={11} />{/if} {statusLabel}</span></div>
			<div class="rounded-lg border border-dashed bg-muted/20 p-3 text-[11px] leading-5"><b class="block text-xs">Split discipline</b><span class="text-muted-foreground">Train-only scaling and reward normalization. Validation selects the epoch. Holdout is reported once from that selected policy.</span></div>
			<button type="button" onclick={start} disabled={!canRun} class={`inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md text-xs font-semibold ${running ? 'border border-destructive text-destructive hover:bg-destructive/10' : 'bg-foreground text-background disabled:opacity-40'}`}>{#if running}<Square size={14} fill="currentColor" /> Stop RLVR{:else}<Play size={14} fill="currentColor" /> Start RLVR{/if}</button>
			{#if exitCode !== null && exitCode !== 0}<p role="alert" class="rounded-lg border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">The runner exited with code {exitCode}. Check the console below.</p>{/if}
			{#if error}<p role="alert" class="rounded-lg border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">{error}</p>{/if}
			<p class="mt-auto text-[10px] leading-4 text-muted-foreground"><Terminal size={12} class="mr-1 inline" />Command: <code>cargo run --bin train_event_rl -- --algorithm rlvr</code></p>
		</aside>
	</div>

	{#if metrics}
		<section class="mb-3 rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="rlvr-results-title">
			<div class="flex flex-wrap items-center justify-between gap-2"><div><h2 id="rlvr-results-title" class="text-sm font-semibold">Selected-policy results</h2><p class="text-[11px] text-muted-foreground">Validation-selected epoch {metrics.selected_epoch ?? '—'} · leakage {metrics.leakage_check ?? 'unknown'}</p></div><span class="rounded-full bg-emerald-100 px-2 py-1 text-[10px] font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">Verifier complete</span></div>
			<div class="mt-3 grid grid-cols-2 gap-2 lg:grid-cols-5">
				<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase text-muted-foreground">Holdout learned</span><b class="text-base">{formatPnl(metrics.learned?.sum_pnl_usd)}</b><span class="block text-[10px] text-muted-foreground">{formatCapture(metrics.learned?.oracle_capture)} of oracle</span></div>
				<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase text-muted-foreground">Holdout normal</span><b class="text-base">{formatPnl(metrics.fixed_normal?.sum_pnl_usd)}</b></div>
				<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase text-muted-foreground">Holdout invert</span><b class="text-base">{formatPnl(metrics.fixed_invert?.sum_pnl_usd)}</b></div>
				<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase text-muted-foreground">Holdout oracle</span><b class="text-base">{formatPnl(metrics.learned?.oracle_pnl_usd)}</b></div>
				<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase text-muted-foreground">Holdout events</span><b class="text-base">{formatCount(metrics.learned?.event_count)}</b><span class="block text-[10px] text-muted-foreground">{formatCount(metrics.split?.purged)} purged</span></div>
			</div>
			<div class="mt-3 grid gap-2 text-[10px] text-muted-foreground sm:grid-cols-3"><div>Run <code class="break-all text-foreground">{artifactPaths.runDir ?? '—'}</code></div><div>Metrics <code class="break-all text-foreground">{artifactPaths.metricsPath ?? '—'}</code></div><div>Policy <code class="break-all text-foreground">{artifactPaths.policyPath ?? '—'}</code></div></div>
		</section>
	{/if}

	<section class="rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="rlvr-console-title">
		<div class="mb-2 flex items-center justify-between gap-2"><div><h2 id="rlvr-console-title" class="text-sm font-semibold">Runner console</h2><p class="text-[11px] text-muted-foreground">Build, progress, and verifier output.</p></div><span class="text-[10px] text-muted-foreground">{logs.length} lines</span></div>
		<pre class="max-h-72 min-h-24 overflow-auto rounded-lg bg-zinc-950 p-3 font-mono text-[10px] leading-4 text-zinc-200">{#if logs.length === 0}<span class="text-zinc-500">No output yet. Start RLVR to see live logs.</span>{:else}{#each logs as log}<span class={log.kind === 'stderr' || log.kind === 'error' ? 'text-rose-300' : log.kind === 'system' ? 'text-cyan-300' : ''}>[{log.kind}] {log.text}{'\n'}</span>{/each}{/if}</pre>
	</section>
</main>

<script lang="ts">
	import { BarChart3, CircleAlert, ExternalLink, LoaderCircle, Play, RefreshCcw, Square, Terminal } from 'lucide-svelte';
	import SourceFilePicker from './SourceFilePicker.svelte';
	import type { BackendId, DeviceId, StartMode } from './types';

	type Props = {
		algorithm: 'ga' | 'rl';
		startMode: StartMode;
		backend: BackendId;
		device: DeviceId;
	};

	let { algorithm, startMode, backend, device }: Props = $props();

	type LogLine = { kind: 'system' | 'stdout' | 'stderr' | 'error'; text: string };
	type BuildProfile = 'debug' | 'release';

	let trainPath = $state('data/train');
	let valPath = $state('data/val');
	let testPath = $state('data/test');
	let checkpointPath = $state('');
	let outdir = $state('runs_ga');
	let profile = $state<BuildProfile>('release');
	let dataMode = $state<'windowed' | 'full'>('windowed');
	let window = $state(512);
	let step = $state(256);
	let generations = $state(5);
	let population = $state(6);
	let workers = $state(2);
	let epochs = $state(10);
	let trainWindows = $state(3);
	let rlAlgorithm = $state<'ppo' | 'grpo'>('ppo');
	let logs = $state<LogLine[]>([]);
	let running = $state(false);
	let exitCode = $state<number | null>(null);
	let error = $state('');
	let controller: AbortController | null = null;

	const inputClass = 'h-9 min-w-0 w-full rounded-md border bg-background px-2.5 text-xs sm:text-sm';
	const selectClass = `${inputClass} pr-7`;
	const isGa = $derived(algorithm === 'ga');
	const legacyBackend = $derived(backend === 'burn' || backend === 'candle' ? backend : 'libtorch');
	const checkpointExtensions = $derived(isGa ? ['bin', 'json'] : ['pt', 'safetensors', 'bin']);
	const canRun = $derived(
		!running && trainPath.trim().length > 0 && outdir.trim().length > 0 &&
		(startMode === 'new' || checkpointPath.trim().length > 0)
	);
	const runLabel = $derived(running ? 'Stop run' : startMode === 'continue' ? `Resume ${isGa ? 'GA' : 'RL'}` : `Start ${isGa ? 'GA' : 'RL'}`);

	const appendLog = (kind: LogLine['kind'], text: string) => {
		const parts = text.split(/\r?\n/).filter((part) => part.length > 0);
		if (parts.length === 0) return;
		logs = [...logs, ...parts.map((part) => ({ kind, text: part }))].slice(-160);
	};

	const setDefaultOutput = () => {
		const expected = isGa ? 'runs_ga' : 'runs_rl';
		if (outdir === 'runs_ga' || outdir === 'runs_rl') outdir = expected;
	};

	$effect(() => {
		algorithm;
		setDefaultOutput();
	});

	const buildParams = (): Record<string, unknown> => {
		const params: Record<string, unknown> = {
			backend: legacyBackend,
			device,
			outdir: outdir.trim(),
			'train-parquet': trainPath.trim(),
			'val-parquet': valPath.trim(),
			'test-parquet': testPath.trim(),
			window,
			step,
			windowed: dataMode === 'windowed',
			'full-file': dataMode === 'full',
			'load-checkpoint': startMode === 'continue' ? checkpointPath.trim() : undefined,
			'checkpoint-every': 1,
			...(isGa ? { 'selection-use-eval': true, 'eval-windows': 0 } : {})
		};

		if (isGa) {
			Object.assign(params, {
				generations,
				'pop-size': population,
				workers
			});
		} else {
			Object.assign(params, {
				algorithm: rlAlgorithm,
				epochs,
				'train-windows': trainWindows,
				'ppo-epochs': 4,
				'group-size': 8
			});
		}
		return params;
	};

	const parseEvent = (payload: string) => {
		try {
			const event = JSON.parse(payload) as { type?: string; content?: string; code?: number | null };
			if (event.type === 'stdout' || event.type === 'stderr') {
				appendLog(event.type, event.content ?? '');
			} else if (event.type === 'error') {
				appendLog('error', event.content ?? 'Runner error');
				error = event.content ?? 'Runner error';
			} else if (event.type === 'exit') {
				exitCode = typeof event.code === 'number' ? event.code : null;
				appendLog('system', `Process exited with code ${event.code ?? 'unknown'}.`);
				running = false;
			}
		} catch {
			appendLog('error', 'Received malformed training output.');
		}
	};

	const start = async () => {
		if (running) {
			controller?.abort();
			return;
		}
		if (!canRun) {
			error = startMode === 'continue' ? 'Choose a checkpoint before resuming.' : 'Choose a training dataset and output directory.';
			return;
		}

		error = '';
		exitCode = null;
		logs = [];
		running = true;
		controller = new AbortController();
		appendLog('system', `${startMode === 'continue' ? 'Resuming' : 'Starting'} ${isGa ? 'GA' : `RL (${rlAlgorithm.toUpperCase()})`} with ${legacyBackend} / ${profile}.`);

		try {
			const response = await fetch('/api/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				signal: controller.signal,
				body: JSON.stringify({ engine: algorithm, params: buildParams(), profile })
			});
			if (!response.ok || !response.body) {
				const body = await response.text().catch(() => '');
				throw new Error(body || `Training request failed (${response.status})`);
			}

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
				appendLog('system', 'Training stopped by user.');
			} else {
				error = reason instanceof Error ? reason.message : String(reason);
				appendLog('error', error);
			}
		} finally {
			running = false;
			controller = null;
		}
	};
</script>

<section class="legacy-runner rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="legacy-runner-title">
	<div class="flex flex-wrap items-start justify-between gap-2">
		<div class="flex items-start gap-2">
			<div class="grid size-9 shrink-0 place-items-center rounded-lg bg-muted"><Terminal size={17} /></div>
			<div>
				<h2 id="legacy-runner-title" class="text-sm font-semibold">{isGa ? 'Genetic algorithm' : 'Reinforcement learning'} runner</h2>
				<p class="text-[11px] text-muted-foreground">Compact controls for the existing Rust {isGa ? 'GA' : 'RL'} CLI and live SSE output.</p>
			</div>
		</div>
		<a href={isGa ? '/ga' : '/rl'} class="inline-flex min-h-11 items-center gap-1 rounded-md border px-3 text-xs font-medium hover:bg-muted"><BarChart3 size={14} /> Charts <ExternalLink size={12} /></a>
	</div>

	<div class="legacy-runner-grid mt-3 grid gap-3 lg:grid-cols-[minmax(0,1.35fr)_minmax(18rem,0.8fr)]">
		<div class="space-y-3">
			<div class="grid gap-2 rounded-lg border bg-muted/15 p-2.5 sm:grid-cols-2">
				<div class="sm:col-span-2 flex items-center justify-between gap-2">
					<div class="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Dataset paths</div>
					<span class="text-[10px] text-muted-foreground">{dataMode === 'full' ? 'Full files' : 'Windowed samples'}</span>
				</div>
				<div class="sm:col-span-2"><label class="mb-1 block text-[11px] font-medium" for="legacy-train-path">Train</label><SourceFilePicker value={trainPath} onChange={(value) => (trainPath = value)} disabled={running} initialDir="data" extensions={['parquet']} label="Choose train parquet" placeholder="data/train or train.parquet" /></div>
				<label class="text-[11px] font-medium" for="legacy-val-path">Validation<input id="legacy-val-path" class={inputClass + ' mt-1'} value={valPath} oninput={(event) => (valPath = event.currentTarget.value)} disabled={running} /></label>
				<label class="text-[11px] font-medium" for="legacy-test-path">Test<input id="legacy-test-path" class={inputClass + ' mt-1'} value={testPath} oninput={(event) => (testPath = event.currentTarget.value)} disabled={running} /></label>
			</div>

			<div class="grid gap-2 rounded-lg border p-2.5 sm:grid-cols-3">
				<label class="text-[11px] font-medium sm:col-span-2" for="legacy-output">Output directory<input id="legacy-output" class={inputClass + ' mt-1'} value={outdir} oninput={(event) => (outdir = event.currentTarget.value)} disabled={running} /></label>
				<label class="text-[11px] font-medium" for="legacy-profile">Build<select id="legacy-profile" class={selectClass + ' mt-1'} bind:value={profile} disabled={running}><option value="release">Release</option><option value="debug">Debug</option></select></label>
			</div>
			<p class="px-1 text-[10px] text-muted-foreground">Runtime: <span class="font-medium text-foreground">{device.toUpperCase()}</span> · selected in the setup bar above.</p>

			<div class="grid gap-2 rounded-lg border p-2.5 sm:grid-cols-4">
				<label class="text-[11px] font-medium" for="legacy-data-mode">Data mode<select id="legacy-data-mode" class={selectClass + ' mt-1'} bind:value={dataMode} disabled={running}><option value="windowed">Windowed</option><option value="full">Full file</option></select></label>
				<label class="text-[11px] font-medium" for="legacy-window">Window<input id="legacy-window" class={inputClass + ' mt-1'} type="number" min="1" bind:value={window} disabled={running} /></label>
				<label class="text-[11px] font-medium" for="legacy-step">Step<input id="legacy-step" class={inputClass + ' mt-1'} type="number" min="1" bind:value={step} disabled={running} /></label>
				{#if isGa}
					<label class="text-[11px] font-medium" for="legacy-iterations">Generations<input id="legacy-iterations" class={inputClass + ' mt-1'} type="number" min="1" bind:value={generations} disabled={running} /></label>
				{:else}
					<label class="text-[11px] font-medium" for="legacy-iterations">Epochs<input id="legacy-iterations" class={inputClass + ' mt-1'} type="number" min="1" bind:value={epochs} disabled={running} /></label>
				{/if}
			</div>

			<div class="grid gap-2 rounded-lg border p-2.5 sm:grid-cols-3">
				{#if isGa}
					<label class="text-[11px] font-medium" for="legacy-population">Population<input id="legacy-population" class={inputClass + ' mt-1'} type="number" min="2" bind:value={population} disabled={running} /></label>
					<label class="text-[11px] font-medium" for="legacy-workers">Workers<input id="legacy-workers" class={inputClass + ' mt-1'} type="number" min="0" bind:value={workers} disabled={running} /></label>
					<div class="flex items-end text-[10px] leading-4 text-muted-foreground">Checkpoints are saved every iteration under the output directory.</div>
				{:else}
					<label class="text-[11px] font-medium" for="legacy-rl-algorithm">Algorithm<select id="legacy-rl-algorithm" class={selectClass + ' mt-1'} bind:value={rlAlgorithm} disabled={running}><option value="ppo">PPO</option><option value="grpo">GRPO</option></select></label>
					<label class="text-[11px] font-medium" for="legacy-train-windows">Train windows<input id="legacy-train-windows" class={inputClass + ' mt-1'} type="number" min="1" bind:value={trainWindows} disabled={running} /></label>
					<div class="flex items-end text-[10px] leading-4 text-muted-foreground">Checkpoints are saved every epoch under the output directory.</div>
				{/if}
			</div>

			{#if startMode === 'continue'}
				<div class="rounded-lg border border-dashed bg-muted/15 p-2.5">
					<div class="mb-1.5 flex items-center gap-1.5 text-xs font-semibold"><RefreshCcw size={13} /> Continue from checkpoint</div>
					<SourceFilePicker value={checkpointPath} onChange={(value) => (checkpointPath = value)} disabled={running} initialDir={isGa ? 'runs_ga' : 'runs_rl'} extensions={checkpointExtensions} label={`Choose ${isGa ? 'GA' : 'RL'} checkpoint`} placeholder={isGa ? 'runs_ga/checkpoint_gen4.bin' : 'runs_rl/checkpoint_epoch4.pt'} emptyText="No compatible checkpoints found." />
				</div>
			{/if}
		</div>

		<aside class="flex min-h-0 flex-col rounded-lg border bg-zinc-950 p-2.5 text-zinc-100" aria-label="Training output">
			<div class="flex items-center justify-between gap-2 border-b border-zinc-800 pb-2">
				<div class="flex items-center gap-1.5 text-xs font-semibold"><Terminal size={13} /> Live output</div>
				{#if running}<span class="flex items-center gap-1 text-[10px] text-amber-300"><LoaderCircle class="animate-spin" size={11} /> Running</span>{:else if exitCode !== null}<span class={`text-[10px] ${exitCode === 0 ? 'text-emerald-300' : 'text-rose-300'}`}>exit {exitCode}</span>{/if}
			</div>
			<div class="mt-2 min-h-36 flex-1 overflow-auto rounded bg-black/30 p-2 font-mono text-[10px] leading-4" aria-live="polite">
				{#if logs.length === 0}<span class="text-zinc-500">Start a run to stream Rust stdout and stderr here.</span>{:else}{#each logs as line}<div class={line.kind === 'stderr' || line.kind === 'error' ? 'text-rose-300' : line.kind === 'system' ? 'text-cyan-300' : 'text-zinc-300'}>{line.text}</div>{/each}{/if}
			</div>
			{#if error}<div role="alert" class="mt-2 flex gap-1.5 rounded bg-rose-950/70 p-2 text-[10px] leading-4 text-rose-200"><CircleAlert size={13} class="mt-0.5 shrink-0" />{error}</div>{/if}
			<button type="button" onclick={() => void start()} disabled={!running && !canRun} class={`mt-2 inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-md px-3 text-xs font-semibold ${running ? 'border border-rose-400 text-rose-200 hover:bg-rose-950' : 'bg-white text-zinc-950 hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-40'}`}>
				{#if running}<Square size={13} fill="currentColor" />{:else}<Play size={14} fill="currentColor" />{/if}{runLabel}
			</button>
			<p class="mt-1.5 text-center text-[10px] text-zinc-500">Uses <code>/api/train</code> · {isGa ? 'GA' : 'RL'} charts stay separate</p>
		</aside>
	</div>
</section>

<style>
	@media (min-width: 1024px) and (max-height: 800px) {
		.legacy-runner {
			max-height: calc(100vh - 14.25rem);
			overflow: hidden;
		}

		.legacy-runner-grid {
			min-height: 0;
			height: 100%;
			max-height: calc(100vh - 20rem);
			grid-template-rows: minmax(0, 1fr);
		}

		.legacy-runner-grid > div,
		.legacy-runner-grid > aside {
			min-height: 0;
			height: 100%;
			max-height: 100%;
			overflow-y: auto;
		}
	}
</style>

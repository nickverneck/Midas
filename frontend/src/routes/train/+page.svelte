<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import { Badge } from "$lib/components/ui/badge";
	import * as Tabs from "$lib/components/ui/tabs";
	import GaChart from "$lib/components/GaChart.svelte";

	type TrainMode = 'ga' | 'rl';
	type ConsoleLine = { type: string; text: string };
	type DataMode = 'full' | 'windowed';
	type StartChoice = 'new' | 'resume';

	let trainMode = $state<TrainMode>('ga');
	let consoleOutput = $state<ConsoleLine[]>([]);
	let training = $state(false);
	let paramsCollapsed = $state(false);
	let startChoice = $state<StartChoice | null>(null);
	let checkpointPath = $state('');
	let gaDataMode = $state<DataMode>('windowed');
	let rlDataMode = $state<DataMode>('windowed');
	let abortController: AbortController | null = null;
	let fitnessByGen = $state(new Map<number, number>());
	let logOffset = $state(0);
	let logPolling = false;
	let logTimer: ReturnType<typeof setTimeout> | null = null;
	let activeLogDir = $state("runs_ga");
	const logChunk = 500;
	const logPollInterval = 1500;
	const logPollMaxInterval = 6000;
	let logPollDelay = logPollInterval;
	let liveLogUpdates = $state(true);
	let diagnosticsOutput = $state<string | null>(null);
	let diagnosticsError = $state<string | null>(null);
	let diagnosticsLoading = $state(false);
	
	let gaParams = $state({
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "cpu",
		"batch-candidates": 0,
		generations: 5,
		"pop-size": 6,
		workers: 2,
		window: 512,
		step: 256,
		"initial-balance": 10000,
		"max-position": 0,
		"margin-mode": "auto",
		"contract-multiplier": 1.0,
		"margin-per-contract": "",
		"max-hold-bars-positive": 15,
		"max-hold-bars-drawdown": 15,
		"hold-duration-penalty": 1.0,
		"hold-duration-penalty-growth": 0.05,
		"hold-duration-penalty-positive-scale": 0.5,
		"hold-duration-penalty-negative-scale": 1.5,
		"selection-train-weight": 0.3,
		"selection-eval-weight": 0.7,
		"selection-gap-penalty": 0.2,
		"elite-frac": 0.33,
		"mutation-sigma": 0.05,
		"init-sigma": 0.5,
		hidden: 128,
		layers: 2,
		"eval-windows": 2,
		"w-pnl": 1.0,
		"w-sortino": 1.0,
		"w-mdd": 0.5,
		"save-top-n": 5,
		"save-every": 1,
		"checkpoint-every": 1
	});

	let rlParams = $state({
		outdir: "runs_rl",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "cpu",
		window: 512,
		step: 256,
		epochs: 10,
		"train-windows": 3,
		"ppo-epochs": 4,
		lr: 0.0003,
		gamma: 0.99,
		lam: 0.95,
		clip: 0.2,
		"vf-coef": 0.5,
		"ent-coef": 0.01,
		hidden: 128,
		layers: 2,
		"eval-windows": 2,
		"initial-balance": 10000,
		"max-position": 0,
		"margin-mode": "auto",
		"contract-multiplier": 1.0,
		"margin-per-contract": "",
		"max-hold-bars-positive": 15,
		"max-hold-bars-drawdown": 15,
		"hold-duration-penalty": 1.0,
		"hold-duration-penalty-growth": 0.05,
		"hold-duration-penalty-positive-scale": 0.5,
		"hold-duration-penalty-negative-scale": 1.5,
		"w-pnl": 1.0,
		"w-sortino": 1.0,
		"w-mdd": 0.5,
		"log-interval": 1,
		"checkpoint-every": 1
	});

	type ParquetKey = "train-parquet" | "val-parquet" | "test-parquet";
	type FileEntry = { name: string; path: string; kind: "dir" | "file" };
	type FilePickerTarget =
		| { kind: "parquet"; mode: TrainMode; key: ParquetKey }
		| { kind: "checkpoint"; mode: TrainMode };

	let filePickerTarget: FilePickerTarget | null = null;
	let fileBrowserOpen = $state(false);
	let fileBrowserDir = $state("");
	let fileBrowserParent = $state<string | null>(null);
	let fileBrowserEntries = $state<FileEntry[]>([]);
	let fileBrowserLoading = $state(false);
	let fileBrowserError = $state("");
	let fileBrowserToken = 0;
	let fileBrowserTitle = $state("Select File");
	let fileBrowserExtensions = $state<string[]>(["parquet"]);

	const setParquetPath = (mode: TrainMode, key: ParquetKey, value: string) => {
		if (mode === "ga") {
			gaParams[key] = value;
		} else {
			rlParams[key] = value;
		}
	};

	const normalizeExtensions = (extensions: string[]) =>
		extensions.map((ext) => ext.replace(/^\./, "").toLowerCase());

	const runDiagnostics = async () => {
		if (diagnosticsLoading) return;
		diagnosticsLoading = true;
		diagnosticsError = null;
		try {
			const response = await fetch('/api/diagnostics');
			const data = await response.json();
			if (!response.ok || !data.ok) {
				diagnosticsOutput = (data.stdout || data.output || '').trim();
				diagnosticsError = (data.stderr || data.error || 'Diagnostics failed').trim();
				return;
			}
			diagnosticsOutput = (data.output || '').trim();
		} catch (err) {
			diagnosticsError = err instanceof Error ? err.message : String(err);
		} finally {
			diagnosticsLoading = false;
		}
	};

	const buildFileBrowserUrl = (dir: string, extensions: string[]) => {
		const params = new URLSearchParams();
		if (dir.trim() !== "") {
			params.set("dir", dir);
		}
		const normalized = normalizeExtensions(extensions);
		if (normalized.length > 0) {
			params.set("ext", normalized.join(","));
		}
		const query = params.toString();
		return query ? `/api/files?${query}` : "/api/files";
	};

	const loadFileBrowser = async (dir: string, extensions: string[]) => {
		const token = ++fileBrowserToken;
		fileBrowserLoading = true;
		fileBrowserError = "";
		try {
			const res = await fetch(buildFileBrowserUrl(dir, extensions));
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to load files (${res.status})`);
			}
			const payload = await res.json();
			if (token !== fileBrowserToken) return false;
			fileBrowserDir = typeof payload.dir === "string" ? payload.dir : dir;
			fileBrowserParent = typeof payload.parent === "string" ? payload.parent : null;
			fileBrowserEntries = Array.isArray(payload.entries) ? payload.entries : [];
			return true;
		} catch (err) {
			if (token === fileBrowserToken) {
				fileBrowserError = err instanceof Error ? err.message : String(err);
				fileBrowserEntries = [];
				fileBrowserParent = null;
			}
			return false;
		} finally {
			if (token === fileBrowserToken) {
				fileBrowserLoading = false;
			}
		}
	};

	const openFileBrowser = async (
		target: FilePickerTarget,
		title: string,
		extensions: string[],
		preferredDir: string
	) => {
		filePickerTarget = target;
		fileBrowserTitle = title;
		fileBrowserExtensions = normalizeExtensions(extensions);
		fileBrowserOpen = true;
		fileBrowserDir = "";
		fileBrowserParent = null;
		fileBrowserEntries = [];
		const ok = await loadFileBrowser(preferredDir, fileBrowserExtensions);
		if (!ok && preferredDir !== "") {
			await loadFileBrowser("", fileBrowserExtensions);
		}
	};

	const openParquetPicker = async (mode: TrainMode, key: ParquetKey) => {
		await openFileBrowser({ kind: "parquet", mode, key }, "Select Parquet File", ["parquet"], "data");
	};

	const openCheckpointPicker = async (mode: TrainMode) => {
		const ext = mode === "ga" ? "bin" : "pt";
		const title = mode === "ga" ? "Select GA Checkpoint" : "Select RL Checkpoint";
		const preferredDir = mode === "ga" ? "runs_ga" : "runs_rl";
		await openFileBrowser({ kind: "checkpoint", mode }, title, [ext], preferredDir);
	};

	const closeFileBrowser = () => {
		fileBrowserOpen = false;
		fileBrowserError = "";
		filePickerTarget = null;
	};

	const handleFileEntry = (entry: FileEntry) => {
		if (entry.kind === "dir") {
			void loadFileBrowser(entry.path, fileBrowserExtensions);
			return;
		}
		if (!filePickerTarget) return;
		if (filePickerTarget.kind === "parquet") {
			setParquetPath(filePickerTarget.mode, filePickerTarget.key, entry.path);
		} else {
			checkpointPath = entry.path;
		}
		closeFileBrowser();
	};

	const handleFileBrowserUp = () => {
		if (fileBrowserParent === null) return;
		void loadFileBrowser(fileBrowserParent, fileBrowserExtensions);
	};

	const toNumber = (value: unknown): number | null => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};
	const trimmedCheckpoint = $derived.by(() => checkpointPath.trim());
	const canStartTraining = $derived.by(() => {
		if (!startChoice) return false;
		if (startChoice === 'resume') return trimmedCheckpoint.length > 0;
		return true;
	});
	const runLabel = $derived.by(() => {
		if (training) return 'Stop Training';
		if (startChoice === 'resume') {
			return trainMode === 'ga' ? 'Resume GA Training' : 'Resume RL Training';
		}
		if (startChoice === 'new') {
			return trainMode === 'ga' ? 'Start GA Training' : 'Start RL Training';
		}
		return 'Start Training';
	});
	const runTitle = $derived.by(() => {
		if (training) return 'Stop Training';
		if (startChoice === 'resume') return 'Resume Training';
		if (startChoice === 'new') return 'Start Training';
		return 'Start Training';
	});
	const collapsedLabel = $derived.by(() => (training ? 'Stop' : 'Setup'));
	const collapsedTitle = $derived.by(() => (training ? 'Stop Training' : 'Open Setup'));
	const logKey = $derived.by(() => (trainMode === 'rl' ? 'epoch' : 'gen'));
	const logLabel = $derived.by(() => (trainMode === 'rl' ? 'Epoch' : 'Gen'));
	const logType = $derived.by(() => (trainMode === 'rl' ? 'rl' : 'ga'));
	const fitnessSeries = $derived.by(() =>
		Array.from(fitnessByGen.entries())
			.sort(([a], [b]) => a - b)
			.map(([gen, fitness]) => ({ gen, fitness }))
	);
	const fitnessChartData = $derived.by(() => ({
		labels: fitnessSeries.map((point) => `${logLabel} ${point.gen}`),
		datasets: [
			{
				label: 'Best Fitness',
				data: fitnessSeries.map((point) => point.fitness),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.35)',
				tension: 0.2,
				fill: false
			}
		]
	}));
	const fitnessChartOptions = {
		scales: {
			y: {
				title: {
					display: true,
					text: 'Fitness'
				}
			}
		}
	};

	const updateFitness = (gen: number, fitness: number) => {
		if (!Number.isFinite(gen) || !Number.isFinite(fitness)) return;
		const next = new Map(fitnessByGen);
		const existing = next.get(gen);
		if (existing === undefined || fitness > existing) {
			next.set(gen, fitness);
			fitnessByGen = next;
		}
	};

	const updateFitnessFromRows = (rows: Array<Record<string, unknown>>) => {
		for (const row of rows) {
			if (!row || typeof row !== 'object') continue;
			const gen = toNumber(row[logKey]);
			const fitness = toNumber(row.fitness);
			if (gen !== null && fitness !== null) {
				updateFitness(gen, fitness);
			}
		}
	};

	const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

	const buildLogUrl = (dir: string, offset: number) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir,
			log: logType
		});
		return `/api/logs?${params.toString()}`;
	};

	const fetchLogPayload = async (dir: string, offset: number) => {
		const res = await fetch(buildLogUrl(dir, offset));
		if (res.status === 404) {
			return { rows: [], nextOffset: offset, done: false, notFound: true };
		}
		if (!res.ok) throw new Error(`Log fetch failed (${res.status})`);
		const payload = await res.json();
		const rows = Array.isArray(payload.data) ? payload.data : [];
		const nextOffset = typeof payload.nextOffset === 'number' ? payload.nextOffset : offset + rows.length;
		const done = Boolean(payload.done);
		return {
			rows: rows as Array<Record<string, unknown>>,
			nextOffset,
			done,
			notFound: false
		};
	};

	const readLogChunk = async (dir: string) => {
		const payload = await fetchLogPayload(dir, logOffset);
		if (payload.rows.length > 0) {
			logOffset = payload.nextOffset;
		}
		return payload;
	};

	const drainLogs = async (dir: string) => {
		let emptyReads = 0;
		while (emptyReads < 5) {
			const { rows } = await readLogChunk(dir);
			if (rows.length > 0) {
				updateFitnessFromRows(rows);
				emptyReads = 0;
				continue;
			}
			emptyReads += 1;
			await sleep(200);
		}
	};

	const loadAllLogs = async (dir: string) => {
		let offset = 0;
		let done = false;
		let emptyReads = 0;
		while (!done && emptyReads < 20) {
			const payload = await fetchLogPayload(dir, offset);
			if (payload.rows.length > 0) {
				updateFitnessFromRows(payload.rows);
				offset = payload.nextOffset;
				emptyReads = 0;
			} else {
				emptyReads += 1;
			}
			done = payload.done;
			if (!done) {
				await sleep(200);
			}
		}
		logOffset = offset;
	};

	const rebuildFitnessFromLogs = async (dir: string) => {
		fitnessByGen = new Map();
		logOffset = 0;
		try {
			const params = new URLSearchParams({ dir, mode: "summary", log: logType, key: logKey });
			const res = await fetch(`/api/logs?${params.toString()}`);
			if (!res.ok) throw new Error(`Log summary failed (${res.status})`);
			const payload = await res.json();
			const rows = Array.isArray(payload.data) ? payload.data : [];
			updateFitnessFromRows(rows);
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: 'error', text: message }];
		}
	};

	const pollLogs = async () => {
		if (!logPolling) return;
		try {
			const { rows, notFound } = await readLogChunk(activeLogDir);
			if (rows.length > 0) {
				updateFitnessFromRows(rows);
				logPollDelay = logPollInterval;
			} else {
				logPollDelay = Math.min(
					logPollDelay + (notFound ? 1000 : 500),
					logPollMaxInterval
				);
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: 'error', text: message }];
			stopLogPolling();
		} finally {
			if (logPolling) {
				logTimer = setTimeout(pollLogs, logPollDelay);
			}
		}
	};

	const startLogPolling = (dir: string) => {
		const fallbackDir = trainMode === 'rl' ? "runs_rl" : "runs_ga";
		activeLogDir = dir.trim() || fallbackDir;
		logOffset = 0;
		logPolling = true;
		logPollDelay = logPollInterval;
		if (logTimer) clearTimeout(logTimer);
		void pollLogs();
	};

	const stopLogPolling = () => {
		logPolling = false;
		if (logTimer) clearTimeout(logTimer);
		logTimer = null;
	};

	const selectStartChoice = (choice: StartChoice) => {
		if (training) return;
		startChoice = choice;
		if (choice === 'new') {
			checkpointPath = '';
		}
	};

	const resetTrainingSetup = () => {
		if (training) return;
		startChoice = null;
		checkpointPath = '';
	};

	const attachCheckpoint = (params: Record<string, unknown>) => {
		if (startChoice === 'resume') {
			const trimmed = checkpointPath.trim();
			if (trimmed) {
				params["load-checkpoint"] = trimmed;
			}
		}
		return params;
	};

	const buildGaParams = () => {
		const params = { ...gaParams } as Record<string, unknown>;
		if (gaDataMode === 'full') {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return attachCheckpoint(params);
	};

	const buildRlParams = () => {
		const params = { ...rlParams } as Record<string, unknown>;
		if (rlDataMode === 'full') {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return attachCheckpoint(params);
	};

	const toggleTraining = () => {
		if (training) {
			stopTraining();
		} else {
			if (!canStartTraining) {
				const message = startChoice
					? 'Add a checkpoint path to resume training.'
					: 'Choose new training or resume from a checkpoint first.';
				consoleOutput = [...consoleOutput, { type: 'error', text: message }];
				return;
			}
			startTraining(trainMode);
		}
	};

	async function startTraining(mode: TrainMode) {
		if (training) return;
		if (!startChoice) {
			consoleOutput = [
				...consoleOutput,
				{ type: 'error', text: 'Choose new training or resume from a checkpoint first.' }
			];
			return;
		}
		if (startChoice === 'resume' && !checkpointPath.trim()) {
			consoleOutput = [...consoleOutput, { type: 'error', text: 'Checkpoint path is required to resume.' }];
			return;
		}
		const params = mode === 'ga' ? buildGaParams() : buildRlParams();
		const fallbackDir = trainMode === 'rl' ? "runs_rl" : "runs_ga";
		const outdir = typeof params.outdir === 'string' ? params.outdir : fallbackDir;
		abortController?.abort();
		abortController = new AbortController();
		training = true;
		const verb = startChoice === 'resume' ? 'Resuming' : 'Starting';
		const suffix = startChoice === 'resume' ? ' from checkpoint' : '';
		consoleOutput = [{ type: 'system', text: `${verb} ${mode.toUpperCase()} training${suffix}...` }];
		fitnessByGen = new Map();
		if (liveLogUpdates) {
			startLogPolling(outdir);
		} else {
			activeLogDir = outdir;
			logOffset = 0;
		}
		
		try {
			const response = await fetch('/api/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ engine: mode, params }),
				signal: abortController.signal
			});

			if (!response.ok || !response.body) {
				throw new Error(`Training request failed (${response.status})`);
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				const chunk = decoder.decode(value);
				const lines = chunk.split('\n\n');
				
				for (const line of lines) {
					if (line.startsWith('data: ')) {
						const data = JSON.parse(line.substring(6));
						if (data.type === 'stdout' || data.type === 'stderr') {
							consoleOutput = [...consoleOutput, { type: data.type, text: data.content }];
						} else if (data.type === 'error') {
							consoleOutput = [...consoleOutput, { type: 'error', text: data.content }];
						} else if (data.type === 'exit') {
							training = false;
							abortController = null;
							stopLogPolling();
							if (liveLogUpdates) {
								await drainLogs(activeLogDir);
							}
							await sleep(300);
							await rebuildFitnessFromLogs(activeLogDir);
							consoleOutput = [...consoleOutput, { type: 'system', text: `Process exited with code ${data.code}` }];
						}
					}
				}
			}
		} catch (e) {
			if (e instanceof DOMException && e.name === 'AbortError') {
				consoleOutput = [...consoleOutput, { type: 'system', text: 'Training stopped by user.' }];
			} else {
				const message = e instanceof Error ? e.message : String(e);
				consoleOutput = [...consoleOutput, { type: 'error', text: message }];
			}
			training = false;
			abortController = null;
			stopLogPolling();
		}
	}

	const stopTraining = () => {
		if (!training) return;
		abortController?.abort();
		stopLogPolling();
	};

	const handleCollapsedAction = () => {
		if (training) {
			stopTraining();
		} else {
			paramsCollapsed = false;
		}
	};
</script>

<div class="min-h-screen bg-background">
	<aside
		class={`bg-card border-r shadow-sm transition-[width] duration-200 ease-out w-full lg:fixed lg:inset-y-0 lg:left-0 lg:pt-14 relative ${paramsCollapsed ? 'lg:w-[120px]' : 'lg:w-[360px]'}`}
	>
		{#if paramsCollapsed}
			<button
				type="button"
				class="absolute inset-0 cursor-pointer bg-transparent border-0 p-0 appearance-none z-0"
				aria-label="Expand training parameters"
				title="Expand parameters"
				onclick={() => (paramsCollapsed = false)}
			></button>
		{/if}
		<div class="relative z-10 flex h-full flex-col">
			{#if !paramsCollapsed}
				<div class="flex items-center justify-between px-6 py-4 border-b">
					<div class="text-sm font-semibold tracking-tight">Training Controls</div>
					<Button variant="ghost" size="sm" onclick={() => (paramsCollapsed = true)}>
						Collapse
					</Button>
				</div>
			{/if}
			{#if paramsCollapsed}
				<div class="px-4 py-4">
					<Button
						variant={training ? "destructive" : "secondary"}
						onclick={handleCollapsedAction}
						class="w-full"
						title={collapsedTitle}
					>
						{collapsedLabel}
					</Button>
				</div>
			{:else}
				<div class="flex-1 overflow-y-auto px-6 pb-6">
					<div class="space-y-6">
						<div class="rounded-lg border bg-card/50 p-4 space-y-3">
							<div class="flex items-start justify-between gap-3">
								<div>
									<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 1</div>
									<div class="text-sm font-semibold">Choose a starting point</div>
								</div>
								{#if startChoice}
									<Button variant="ghost" size="sm" onclick={resetTrainingSetup} disabled={training}>
										New Training
									</Button>
								{/if}
							</div>
							{#if !startChoice}
								<p class="text-xs text-muted-foreground">
									Do you want to start fresh or continue from a checkpoint?
								</p>
								<div class="grid gap-2">
									<Button onclick={() => selectStartChoice('new')} disabled={training}>
										New Training
									</Button>
									<Button variant="outline" onclick={() => selectStartChoice('resume')} disabled={training}>
										Continue from Checkpoint
									</Button>
								</div>
							{:else}
								<div class="flex items-center gap-2 text-sm">
									<span class="text-muted-foreground">Selected:</span>
									<Badge variant="secondary">
										{startChoice === 'resume' ? 'Resume from Checkpoint' : 'New Training'}
									</Badge>
								</div>
								{#if startChoice === 'resume'}
									<div class="grid gap-2">
										<Label for="checkpoint-path">Checkpoint Path</Label>
										<div class="flex flex-col gap-2">
											<Input
												id="checkpoint-path"
												type="text"
												bind:value={checkpointPath}
												placeholder={trainMode === 'ga' ? "runs_ga/checkpoint_gen4.bin" : "runs_rl/checkpoint_epoch4.pt"}
											/>
											<Button type="button" variant="outline" onclick={() => openCheckpointPicker(trainMode)}>
												Browse
											</Button>
										</div>
										<div class="text-xs text-muted-foreground">
											GA checkpoints use .bin; RL checkpoints use .pt.
										</div>
									</div>
								{/if}
							{/if}
						</div>

						{#if startChoice}
							<div class="rounded-lg border bg-card/50 p-4 space-y-4">
								<div class="flex items-center justify-between">
									<div>
										<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 2</div>
										<div class="text-sm font-semibold">Configure your run</div>
									</div>
									<Badge variant="outline">{trainMode === 'ga' ? 'GA' : 'RL'}</Badge>
								</div>

								<Tabs.Root bind:value={trainMode} class="w-full">
									<Tabs.List class="grid w-full grid-cols-2 mb-4">
										<Tabs.Trigger value="ga">GA (Primary)</Tabs.Trigger>
										<Tabs.Trigger value="rl">RL (PPO)</Tabs.Trigger>
									</Tabs.List>

									<Tabs.Content value="ga">
										<form
											class="space-y-4"
											onsubmit={(event) => {
												event.preventDefault();
												startTraining('ga');
											}}
										>
											<div class="space-y-4">
												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-train-parquet">Train Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="ga-train-parquet"
																	type="text"
																	bind:value={gaParams["train-parquet"]}
																	placeholder="data/train"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("ga", "train-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="ga-val-parquet">Val Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="ga-val-parquet"
																	type="text"
																	bind:value={gaParams["val-parquet"]}
																	placeholder="data/val"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("ga", "val-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="ga-test-parquet">Test Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="ga-test-parquet"
																	type="text"
																	bind:value={gaParams["test-parquet"]}
																	placeholder="data/test"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("ga", "test-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="ga-data-mode">Data Mode</Label>
															<select
																id="ga-data-mode"
																bind:value={gaDataMode}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="windowed">Windowed</option>
																<option value="full">Full file</option>
															</select>
														</div>
														{#if gaDataMode === 'windowed'}
															<div class="grid gap-2">
																<Label for="ga-window">Window Size</Label>
																<Input id="ga-window" type="number" min="1" bind:value={gaParams.window} />
															</div>
															<div class="grid gap-2">
																<Label for="ga-step">Step Size</Label>
																<Input id="ga-step" type="number" min="1" bind:value={gaParams.step} />
															</div>
														{/if}
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Hardware</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-device">Device</Label>
															<select
																id="ga-device"
																bind:value={gaParams.device}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="cpu">CPU</option>
																<option value="mps">MPS</option>
																<option value="cuda">CUDA</option>
															</select>
														</div>
														<div class="grid gap-2">
															<Label for="ga-batch-candidates">Batch Candidates</Label>
															<Input
																id="ga-batch-candidates"
																type="number"
																min="0"
																bind:value={gaParams["batch-candidates"]}
															/>
															<p class="text-xs text-muted-foreground">0 uses auto batching.</p>
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Evolution Settings</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-generations">Generations</Label>
															<Input id="ga-generations" type="number" min="1" bind:value={gaParams.generations} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-pop-size">Population Size</Label>
															<Input id="ga-pop-size" type="number" min="1" bind:value={gaParams["pop-size"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-workers">Workers</Label>
															<Input id="ga-workers" type="number" min="0" bind:value={gaParams.workers} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-elite-frac">Elite Fraction</Label>
															<Input id="ga-elite-frac" type="number" step="0.01" min="0" max="1" bind:value={gaParams["elite-frac"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-mutation-sigma">Mutation Sigma</Label>
															<Input id="ga-mutation-sigma" type="number" step="0.01" min="0" bind:value={gaParams["mutation-sigma"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-init-sigma">Init Sigma</Label>
															<Input id="ga-init-sigma" type="number" step="0.01" min="0" bind:value={gaParams["init-sigma"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-eval-windows">Eval Windows</Label>
															<Input id="ga-eval-windows" type="number" min="1" bind:value={gaParams["eval-windows"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Model &amp; Fitness</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-hidden">Hidden Units</Label>
															<Input id="ga-hidden" type="number" min="1" bind:value={gaParams.hidden} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-layers">Layers</Label>
															<Input id="ga-layers" type="number" min="1" bind:value={gaParams.layers} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-initial-balance">Initial Balance</Label>
															<Input id="ga-initial-balance" type="number" step="0.01" bind:value={gaParams["initial-balance"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-max-position">Max Position (0 = no cap)</Label>
															<Input id="ga-max-position" type="number" min="0" bind:value={gaParams["max-position"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-margin-mode">Margin Mode</Label>
															<select
																id="ga-margin-mode"
																bind:value={gaParams["margin-mode"]}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="auto">Auto</option>
																<option value="per-contract">Per-contract</option>
																<option value="price">Price-based</option>
															</select>
														</div>
														<div class="grid gap-2">
															<Label for="ga-contract-multiplier">Contract Multiplier</Label>
															<Input id="ga-contract-multiplier" type="number" step="0.01" min="0" bind:value={gaParams["contract-multiplier"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-margin-per-contract">Margin Per Contract</Label>
															<Input id="ga-margin-per-contract" type="number" step="0.01" min="0" bind:value={gaParams["margin-per-contract"]} placeholder="auto from config" />
														</div>
														<div class="grid gap-2">
															<Label for="ga-w-pnl">Fitness Weight (PNL)</Label>
															<Input id="ga-w-pnl" type="number" step="0.01" bind:value={gaParams["w-pnl"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-w-sortino">Fitness Weight (Sortino)</Label>
															<Input id="ga-w-sortino" type="number" step="0.01" bind:value={gaParams["w-sortino"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-w-mdd">Fitness Weight (Max DD)</Label>
															<Input id="ga-w-mdd" type="number" step="0.01" bind:value={gaParams["w-mdd"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-selection-train-weight">Selection Weight (Train)</Label>
															<Input id="ga-selection-train-weight" type="number" step="0.01" min="0" bind:value={gaParams["selection-train-weight"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-selection-eval-weight">Selection Weight (Eval)</Label>
															<Input id="ga-selection-eval-weight" type="number" step="0.01" min="0" bind:value={gaParams["selection-eval-weight"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-selection-gap-penalty">Selection Gap Penalty</Label>
															<Input id="ga-selection-gap-penalty" type="number" step="0.01" min="0" bind:value={gaParams["selection-gap-penalty"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Position Hold Penalties</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-max-hold-bars-positive">Max Hold Bars (Profit)</Label>
															<Input id="ga-max-hold-bars-positive" type="number" min="0" bind:value={gaParams["max-hold-bars-positive"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-max-hold-bars-drawdown">Max Hold Bars (Drawdown)</Label>
															<Input id="ga-max-hold-bars-drawdown" type="number" min="0" bind:value={gaParams["max-hold-bars-drawdown"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-hold-duration-penalty">Hold Penalty</Label>
															<Input id="ga-hold-duration-penalty" type="number" step="0.01" min="0" bind:value={gaParams["hold-duration-penalty"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-hold-duration-penalty-growth">Hold Penalty Growth</Label>
															<Input id="ga-hold-duration-penalty-growth" type="number" step="0.01" min="0" bind:value={gaParams["hold-duration-penalty-growth"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-hold-duration-penalty-positive-scale">Hold Penalty Scale (Profit)</Label>
															<Input id="ga-hold-duration-penalty-positive-scale" type="number" step="0.01" min="0" bind:value={gaParams["hold-duration-penalty-positive-scale"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-hold-duration-penalty-negative-scale">Hold Penalty Scale (Loss)</Label>
															<Input id="ga-hold-duration-penalty-negative-scale" type="number" step="0.01" min="0" bind:value={gaParams["hold-duration-penalty-negative-scale"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Output &amp; Checkpoints</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="ga-outdir">Output Folder</Label>
															<Input id="ga-outdir" type="text" bind:value={gaParams.outdir} placeholder="runs_ga" />
														</div>
														<div class="grid gap-2">
															<Label for="ga-save-top-n">Save Top N</Label>
															<Input id="ga-save-top-n" type="number" min="0" bind:value={gaParams["save-top-n"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-save-every">Save Every (gens)</Label>
															<Input id="ga-save-every" type="number" min="0" bind:value={gaParams["save-every"]} />
														</div>
														<div class="grid gap-2">
															<Label for="ga-checkpoint-every">Checkpoint Every (gens)</Label>
															<Input id="ga-checkpoint-every" type="number" min="0" bind:value={gaParams["checkpoint-every"]} />
														</div>
													</div>
												</details>
											</div>
										</form>
									</Tabs.Content>

									<Tabs.Content value="rl">
										<form
											class="space-y-4"
											onsubmit={(event) => {
												event.preventDefault();
												startTraining('rl');
											}}
										>
											<div class="space-y-4">
												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-train-parquet">Train Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="rl-train-parquet"
																	type="text"
																	bind:value={rlParams["train-parquet"]}
																	placeholder="data/train"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("rl", "train-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="rl-val-parquet">Val Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="rl-val-parquet"
																	type="text"
																	bind:value={rlParams["val-parquet"]}
																	placeholder="data/val"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("rl", "val-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="rl-test-parquet">Test Parquet</Label>
															<div class="flex flex-col gap-2">
																<Input
																	id="rl-test-parquet"
																	type="text"
																	bind:value={rlParams["test-parquet"]}
																	placeholder="data/test"
																/>
																<Button type="button" variant="outline" onclick={() => openParquetPicker("rl", "test-parquet")}>
																	Browse
																</Button>
															</div>
														</div>
														<div class="grid gap-2">
															<Label for="rl-data-mode">Data Mode</Label>
															<select
																id="rl-data-mode"
																bind:value={rlDataMode}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="windowed">Windowed</option>
																<option value="full">Full file</option>
															</select>
														</div>
														{#if rlDataMode === 'windowed'}
															<div class="grid gap-2">
																<Label for="rl-window">Window Size</Label>
																<Input id="rl-window" type="number" min="1" bind:value={rlParams.window} />
															</div>
															<div class="grid gap-2">
																<Label for="rl-step">Step Size</Label>
																<Input id="rl-step" type="number" min="1" bind:value={rlParams.step} />
															</div>
														{/if}
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Hardware</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-device">Device</Label>
															<select
																id="rl-device"
																bind:value={rlParams.device}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="cpu">CPU</option>
																<option value="mps">MPS</option>
																<option value="cuda">CUDA</option>
															</select>
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">PPO Training</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-epochs">Epochs</Label>
															<Input id="rl-epochs" type="number" min="1" bind:value={rlParams.epochs} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-train-windows">Train Windows</Label>
															<Input id="rl-train-windows" type="number" min="0" bind:value={rlParams["train-windows"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-ppo-epochs">PPO Epochs</Label>
															<Input id="rl-ppo-epochs" type="number" min="1" bind:value={rlParams["ppo-epochs"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-eval-windows">Eval Windows</Label>
															<Input id="rl-eval-windows" type="number" min="1" bind:value={rlParams["eval-windows"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Optimization</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-lr">Learning Rate</Label>
															<Input id="rl-lr" type="number" step="0.0001" bind:value={rlParams.lr} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-gamma">Gamma</Label>
															<Input id="rl-gamma" type="number" step="0.01" min="0" max="1" bind:value={rlParams.gamma} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-lam">Lambda</Label>
															<Input id="rl-lam" type="number" step="0.01" min="0" max="1" bind:value={rlParams.lam} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-clip">Clip Range</Label>
															<Input id="rl-clip" type="number" step="0.01" min="0" bind:value={rlParams.clip} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-vf-coef">Value Coef</Label>
															<Input id="rl-vf-coef" type="number" step="0.01" min="0" bind:value={rlParams["vf-coef"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-ent-coef">Entropy Coef</Label>
															<Input id="rl-ent-coef" type="number" step="0.01" min="0" bind:value={rlParams["ent-coef"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Model &amp; Fitness</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-hidden">Hidden Units</Label>
															<Input id="rl-hidden" type="number" min="1" bind:value={rlParams.hidden} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-layers">Layers</Label>
															<Input id="rl-layers" type="number" min="1" bind:value={rlParams.layers} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-initial-balance">Initial Balance</Label>
															<Input id="rl-initial-balance" type="number" step="0.01" bind:value={rlParams["initial-balance"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-max-position">Max Position (0 = no cap)</Label>
															<Input id="rl-max-position" type="number" min="0" bind:value={rlParams["max-position"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-margin-mode">Margin Mode</Label>
															<select
																id="rl-margin-mode"
																bind:value={rlParams["margin-mode"]}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="auto">Auto</option>
																<option value="per-contract">Per-contract</option>
																<option value="price">Price-based</option>
															</select>
														</div>
														<div class="grid gap-2">
															<Label for="rl-contract-multiplier">Contract Multiplier</Label>
															<Input id="rl-contract-multiplier" type="number" step="0.01" min="0" bind:value={rlParams["contract-multiplier"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-margin-per-contract">Margin Per Contract</Label>
															<Input id="rl-margin-per-contract" type="number" step="0.01" min="0" bind:value={rlParams["margin-per-contract"]} placeholder="auto from config" />
														</div>
														<div class="grid gap-2">
															<Label for="rl-w-pnl">Fitness Weight (PNL)</Label>
															<Input id="rl-w-pnl" type="number" step="0.01" bind:value={rlParams["w-pnl"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-w-sortino">Fitness Weight (Sortino)</Label>
															<Input id="rl-w-sortino" type="number" step="0.01" bind:value={rlParams["w-sortino"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-w-mdd">Fitness Weight (Max DD)</Label>
															<Input id="rl-w-mdd" type="number" step="0.01" bind:value={rlParams["w-mdd"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Position Hold Penalties</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-max-hold-bars-positive">Max Hold Bars (Profit)</Label>
															<Input id="rl-max-hold-bars-positive" type="number" min="0" bind:value={rlParams["max-hold-bars-positive"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-max-hold-bars-drawdown">Max Hold Bars (Drawdown)</Label>
															<Input id="rl-max-hold-bars-drawdown" type="number" min="0" bind:value={rlParams["max-hold-bars-drawdown"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-hold-duration-penalty">Hold Penalty</Label>
															<Input id="rl-hold-duration-penalty" type="number" step="0.01" min="0" bind:value={rlParams["hold-duration-penalty"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-hold-duration-penalty-growth">Hold Penalty Growth</Label>
															<Input id="rl-hold-duration-penalty-growth" type="number" step="0.01" min="0" bind:value={rlParams["hold-duration-penalty-growth"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-hold-duration-penalty-positive-scale">Hold Penalty Scale (Profit)</Label>
															<Input id="rl-hold-duration-penalty-positive-scale" type="number" step="0.01" min="0" bind:value={rlParams["hold-duration-penalty-positive-scale"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-hold-duration-penalty-negative-scale">Hold Penalty Scale (Loss)</Label>
															<Input id="rl-hold-duration-penalty-negative-scale" type="number" step="0.01" min="0" bind:value={rlParams["hold-duration-penalty-negative-scale"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Output &amp; Checkpoints</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rl-outdir">Output Folder</Label>
															<Input id="rl-outdir" type="text" bind:value={rlParams.outdir} placeholder="runs_rl" />
														</div>
														<div class="grid gap-2">
															<Label for="rl-log-interval">Log Interval (epochs)</Label>
															<Input id="rl-log-interval" type="number" min="1" bind:value={rlParams["log-interval"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rl-checkpoint-every">Checkpoint Every (epochs)</Label>
															<Input id="rl-checkpoint-every" type="number" min="0" bind:value={rlParams["checkpoint-every"]} />
														</div>
													</div>
												</details>
											</div>
										</form>
									</Tabs.Content>
								</Tabs.Root>
							</div>

							<div class="rounded-lg border bg-card/50 p-4 space-y-3">
								<div class="flex items-center justify-between">
									<div>
										<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 3</div>
										<div class="text-sm font-semibold">Run training</div>
									</div>
									{#if training}
										<Badge variant="outline" class="animate-pulse">Active</Badge>
									{/if}
								</div>
								<div class="flex items-center gap-2 text-sm">
									<input
										id="live-log-updates"
										type="checkbox"
										class="h-4 w-4 rounded border-input"
										bind:checked={liveLogUpdates}
									/>
									<Label for="live-log-updates">Live log updates</Label>
								</div>
								<p class="text-xs text-muted-foreground">
									Disable to load logs only after training completes.
								</p>
								<Button
									variant={training ? "destructive" : "default"}
									onclick={toggleTraining}
									class="w-full"
									title={runTitle}
									disabled={!training && !canStartTraining}
								>
									{runLabel}
								</Button>
								<Button
									variant="outline"
									class="w-full"
									onclick={runDiagnostics}
									disabled={diagnosticsLoading}
								>
									{diagnosticsLoading ? 'Running diagnostics...' : 'Run CUDA diagnostics'}
								</Button>
								{#if startChoice === 'resume' && !trimmedCheckpoint}
									<div class="text-xs text-muted-foreground">
										Add a checkpoint path to resume training.
									</div>
								{/if}
							</div>
						{/if}
					</div>
				</div>
			{/if}
		</div>
	</aside>

	<main class={`p-8 space-y-8 ${paramsCollapsed ? 'lg:ml-[120px]' : 'lg:ml-[360px]'}`}>
		<div class="flex items-center gap-4">
			<a href="/ga" class="text-sm text-muted-foreground hover:text-foreground">← Back to GA Analytics</a>
			<h1 class="text-4xl font-bold tracking-tight">
				Train {trainMode === 'ga' ? 'GA' : 'RL'} Model
			</h1>
		</div>

		<div class="space-y-6">
			<Card.Root>
				<Card.Header class="flex flex-row items-center justify-between">
					<Card.Title>Best Fitness by {logLabel}</Card.Title>
					{#if training}
						<Badge variant="outline" class="animate-pulse">Active</Badge>
					{/if}
				</Card.Header>
				<Card.Content>
					{#if fitnessSeries.length > 0}
						<div class="h-[260px] min-h-[220px]">
							<GaChart data={fitnessChartData} options={fitnessChartOptions} />
						</div>
					{:else}
						<div class="h-[220px] flex items-center justify-center text-muted-foreground">
							Waiting for the first {logLabel.toLowerCase()} results...
						</div>
					{/if}
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="flex flex-row items-center justify-between">
					<Card.Title>Console Output</Card.Title>
					{#if training}
						<Badge variant="outline" class="animate-pulse">Active</Badge>
					{/if}
				</Card.Header>
				<Card.Content>
					<ScrollArea class="h-[260px] w-full bg-zinc-950 p-4 rounded-md border border-zinc-800 font-mono text-sm">
						{#each consoleOutput as line}
							<div class={line.type === 'stderr' || line.type === 'error' ? 'text-red-400' : line.type === 'system' ? 'text-blue-400' : 'text-zinc-300'}>
								<span class="opacity-50 mr-2">[{new Date().toLocaleTimeString()}]</span>
								{line.text}
							</div>
						{/each}
						{#if consoleOutput.length === 0}
							<div class="text-zinc-600 italic">No output yet. Start training to see logs.</div>
						{/if}
					</ScrollArea>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="flex flex-row items-center justify-between">
					<Card.Title>Diagnostics Output</Card.Title>
					{#if diagnosticsLoading}
						<Badge variant="outline" class="animate-pulse">Running</Badge>
					{/if}
				</Card.Header>
				<Card.Content>
					<ScrollArea class="h-[220px] w-full bg-zinc-950 p-4 rounded-md border border-zinc-800 font-mono text-sm">
						{#if diagnosticsError}
							<div class="text-red-400 whitespace-pre-wrap">{diagnosticsError}</div>
						{/if}
						{#if diagnosticsOutput}
							<div class="text-zinc-300 whitespace-pre-wrap">{diagnosticsOutput}</div>
						{:else}
							<div class="text-zinc-600 italic">Run diagnostics to see CUDA/tch info.</div>
						{/if}
					</ScrollArea>
				</Card.Content>
			</Card.Root>
		</div>
	</main>
	{#if fileBrowserOpen}
		<div
			class="fixed inset-0 z-50 flex items-center justify-center p-4"
			role="dialog"
			aria-modal="true"
			aria-label="Select parquet file"
		>
			<button
				type="button"
				class="absolute inset-0 bg-black/40"
				onclick={closeFileBrowser}
				aria-label="Close file picker"
			></button>
			<div class="relative z-10 w-full max-w-2xl rounded-xl border bg-background p-4 shadow-lg">
				<div class="flex items-start justify-between gap-4">
					<div>
						<div class="text-xs uppercase tracking-wide text-muted-foreground">{fileBrowserTitle}</div>
						<div class="text-sm font-semibold">{fileBrowserDir || "Project root"}</div>
						<div class="text-xs text-muted-foreground">
							Allowed: {fileBrowserExtensions.map((ext) => `.${ext}`).join(", ")}
						</div>
					</div>
					<Button type="button" variant="ghost" size="sm" onclick={closeFileBrowser}>
						Close
					</Button>
				</div>
				<div class="mt-3 flex items-center gap-2">
					<Button type="button" variant="outline" size="sm" onclick={handleFileBrowserUp} disabled={fileBrowserParent === null}>
						Up
					</Button>
					<div class="text-xs text-muted-foreground truncate">
						{fileBrowserDir ? `/${fileBrowserDir}` : "/"}
					</div>
				</div>

				{#if fileBrowserError}
					<div class="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
						{fileBrowserError}
					</div>
				{/if}

				<div class="mt-3 max-h-[360px] overflow-auto rounded-lg border">
					{#if fileBrowserLoading}
						<div class="px-4 py-6 text-sm text-muted-foreground">Loading files...</div>
					{:else if fileBrowserEntries.length === 0}
						<div class="px-4 py-6 text-sm text-muted-foreground">No parquet files found here.</div>
					{:else}
						<div class="divide-y">
							{#each fileBrowserEntries as entry}
								<button
									type="button"
									class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-muted/50"
									onclick={() => handleFileEntry(entry)}
								>
									<span class="text-[11px] font-semibold uppercase text-muted-foreground">
										{entry.kind === "dir" ? "Dir" : "File"}
									</span>
									<span class="text-sm">{entry.name}</span>
								</button>
							{/each}
						</div>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>

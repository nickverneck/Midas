<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import { Badge } from "$lib/components/ui/badge";
	import * as Tabs from "$lib/components/ui/tabs";
	import GaChart from "$lib/components/GaChart.svelte";

	type TrainMode = 'rust' | 'python';
	type ConsoleLine = { type: string; text: string };
	type DataMode = 'full' | 'windowed';
	type StartChoice = 'new' | 'resume';

	let trainMode = $state<TrainMode>('rust');
	let consoleOutput = $state<ConsoleLine[]>([]);
	let training = $state(false);
	let paramsCollapsed = $state(false);
	let startChoice = $state<StartChoice | null>(null);
	let checkpointPath = $state('');
	let rustDataMode = $state<DataMode>('windowed');
	let pythonDataMode = $state<DataMode>('windowed');
	let abortController: AbortController | null = null;
	let fitnessByGen = $state(new Map<number, number>());
	let logOffset = $state(0);
	let logPolling = false;
	let logTimer: ReturnType<typeof setTimeout> | null = null;
	let activeLogDir = $state("runs_ga");
	const logChunk = 500;
	const logPollInterval = 1500;
	
	let rustParams = $state({
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "cpu",
		generations: 5,
		"pop-size": 6,
		workers: 2,
		window: 512,
		step: 256,
		"initial-balance": 10000,
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

	let pythonParams = $state({
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "cpu",
		generations: 5,
		"pop-size": 6,
		workers: 2,
		window: 512,
		step: 256,
		"train-epochs": 2,
		"train-windows": 3,
		"eval-windows": 2,
		lr: 0.0003,
		gamma: 0.99,
		lam: 0.95,
		"initial-balance": 10000,
		"mutation-sigma": 0.25,
		"save-top-n": 5,
		"save-every": 1,
		"checkpoint-every": 1
	});

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
			return trainMode === 'rust' ? 'Resume Rust Training' : 'Resume Python Training';
		}
		if (startChoice === 'new') {
			return trainMode === 'rust' ? 'Start Rust Training' : 'Start Python Training';
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
	const fitnessSeries = $derived.by(() =>
		Array.from(fitnessByGen.entries())
			.sort(([a], [b]) => a - b)
			.map(([gen, fitness]) => ({ gen, fitness }))
	);
	const fitnessChartData = $derived.by(() => ({
		labels: fitnessSeries.map((point) => `Gen ${point.gen}`),
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
			const gen = toNumber(row.gen);
			const fitness = toNumber(row.fitness);
			if (gen !== null && fitness !== null) {
				updateFitness(gen, fitness);
			}
		}
	};

	const buildLogUrl = (dir: string, offset: number) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir
		});
		return `/api/logs?${params.toString()}`;
	};

	const readLogChunk = async (dir: string) => {
		const res = await fetch(buildLogUrl(dir, logOffset));
		if (res.status === 404) return [];
		if (!res.ok) throw new Error(`Log fetch failed (${res.status})`);
		const payload = await res.json();
		const rows = Array.isArray(payload.data) ? payload.data : [];
		const nextOffset = typeof payload.nextOffset === 'number' ? payload.nextOffset : logOffset + rows.length;
		logOffset = nextOffset;
		return rows as Array<Record<string, unknown>>;
	};

	const pollLogs = async () => {
		if (!logPolling) return;
		try {
			const rows = await readLogChunk(activeLogDir);
			if (rows.length > 0) {
				updateFitnessFromRows(rows);
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: 'error', text: message }];
			stopLogPolling();
		} finally {
			if (logPolling) {
				logTimer = setTimeout(pollLogs, logPollInterval);
			}
		}
	};

	const startLogPolling = (dir: string) => {
		activeLogDir = dir.trim() || "runs_ga";
		logOffset = 0;
		logPolling = true;
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

	const buildRustParams = () => {
		const params = { ...rustParams } as Record<string, unknown>;
		if (rustDataMode === 'full') {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return attachCheckpoint(params);
	};

	const buildPythonParams = () => {
		const params = { ...pythonParams } as Record<string, unknown>;
		if (pythonDataMode === 'full') {
			params["full-file"] = true;
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
		const params = mode === 'rust' ? buildRustParams() : buildPythonParams();
		const outdir = typeof params.outdir === 'string' ? params.outdir : "runs_ga";
		abortController?.abort();
		abortController = new AbortController();
		training = true;
		const verb = startChoice === 'resume' ? 'Resuming' : 'Starting';
		const suffix = startChoice === 'resume' ? ' from checkpoint' : '';
		consoleOutput = [{ type: 'system', text: `${verb} ${mode.toUpperCase()} training${suffix}...` }];
		fitnessByGen = new Map();
		startLogPolling(outdir);
		
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
		class={`bg-card border-r shadow-sm transition-[width] duration-200 ease-out w-full lg:fixed lg:inset-y-0 lg:left-0 relative ${paramsCollapsed ? 'lg:w-[120px]' : 'lg:w-[360px]'}`}
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
										<Input
											id="checkpoint-path"
											type="text"
											bind:value={checkpointPath}
											placeholder={trainMode === 'rust' ? "runs_ga/checkpoint_gen4.bin" : "runs_ga/checkpoint_gen4.pt"}
										/>
										<div class="text-xs text-muted-foreground">
											Rust checkpoints use .bin; Python checkpoints use .pt.
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
									<Badge variant="outline">{trainMode === 'rust' ? 'Rust' : 'Python'}</Badge>
								</div>

								<Tabs.Root bind:value={trainMode} class="w-full">
									<Tabs.List class="grid w-full grid-cols-2 mb-4">
										<Tabs.Trigger value="rust">Rust (Primary)</Tabs.Trigger>
										<Tabs.Trigger value="python">Python (Legacy)</Tabs.Trigger>
									</Tabs.List>

									<Tabs.Content value="rust">
										<form
											class="space-y-4"
											onsubmit={(event) => {
												event.preventDefault();
												startTraining('rust');
											}}
										>
											<div class="space-y-4">
												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rust-train-parquet">Train Parquet</Label>
															<Input id="rust-train-parquet" type="text" bind:value={rustParams["train-parquet"]} placeholder="data/train" />
														</div>
														<div class="grid gap-2">
															<Label for="rust-val-parquet">Val Parquet</Label>
															<Input id="rust-val-parquet" type="text" bind:value={rustParams["val-parquet"]} placeholder="data/val" />
														</div>
														<div class="grid gap-2">
															<Label for="rust-test-parquet">Test Parquet</Label>
															<Input id="rust-test-parquet" type="text" bind:value={rustParams["test-parquet"]} placeholder="data/test" />
														</div>
														<div class="grid gap-2">
															<Label for="rust-data-mode">Data Mode</Label>
															<select
																id="rust-data-mode"
																bind:value={rustDataMode}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="windowed">Windowed</option>
																<option value="full">Full file</option>
															</select>
														</div>
														{#if rustDataMode === 'windowed'}
															<div class="grid gap-2">
																<Label for="rust-window">Window Size</Label>
																<Input id="rust-window" type="number" min="1" bind:value={rustParams.window} />
															</div>
															<div class="grid gap-2">
																<Label for="rust-step">Step Size</Label>
																<Input id="rust-step" type="number" min="1" bind:value={rustParams.step} />
															</div>
														{/if}
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Evolution Settings</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rust-generations">Generations</Label>
															<Input id="rust-generations" type="number" min="1" bind:value={rustParams.generations} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-pop-size">Population Size</Label>
															<Input id="rust-pop-size" type="number" min="1" bind:value={rustParams["pop-size"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-workers">Workers</Label>
															<Input id="rust-workers" type="number" min="0" bind:value={rustParams.workers} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-elite-frac">Elite Fraction</Label>
															<Input id="rust-elite-frac" type="number" step="0.01" min="0" max="1" bind:value={rustParams["elite-frac"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-mutation-sigma">Mutation Sigma</Label>
															<Input id="rust-mutation-sigma" type="number" step="0.01" min="0" bind:value={rustParams["mutation-sigma"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-init-sigma">Init Sigma</Label>
															<Input id="rust-init-sigma" type="number" step="0.01" min="0" bind:value={rustParams["init-sigma"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-eval-windows">Eval Windows</Label>
															<Input id="rust-eval-windows" type="number" min="1" bind:value={rustParams["eval-windows"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Model &amp; Fitness</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rust-device">Device</Label>
															<select
																id="rust-device"
																bind:value={rustParams.device}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="">Auto</option>
																<option value="cpu">CPU</option>
																<option value="mps">MPS</option>
																<option value="cuda">CUDA</option>
															</select>
														</div>
														<div class="grid gap-2">
															<Label for="rust-hidden">Hidden Units</Label>
															<Input id="rust-hidden" type="number" min="1" bind:value={rustParams.hidden} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-layers">Layers</Label>
															<Input id="rust-layers" type="number" min="1" bind:value={rustParams.layers} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-initial-balance">Initial Balance</Label>
															<Input id="rust-initial-balance" type="number" step="0.01" bind:value={rustParams["initial-balance"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-w-pnl">Fitness Weight (PNL)</Label>
															<Input id="rust-w-pnl" type="number" step="0.01" bind:value={rustParams["w-pnl"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-w-sortino">Fitness Weight (Sortino)</Label>
															<Input id="rust-w-sortino" type="number" step="0.01" bind:value={rustParams["w-sortino"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-w-mdd">Fitness Weight (Max DD)</Label>
															<Input id="rust-w-mdd" type="number" step="0.01" bind:value={rustParams["w-mdd"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Output &amp; Checkpoints</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="rust-outdir">Output Folder</Label>
															<Input id="rust-outdir" type="text" bind:value={rustParams.outdir} placeholder="runs_ga" />
														</div>
														<div class="grid gap-2">
															<Label for="rust-save-top-n">Save Top N</Label>
															<Input id="rust-save-top-n" type="number" min="0" bind:value={rustParams["save-top-n"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-save-every">Save Every (gens)</Label>
															<Input id="rust-save-every" type="number" min="0" bind:value={rustParams["save-every"]} />
														</div>
														<div class="grid gap-2">
															<Label for="rust-checkpoint-every">Checkpoint Every (gens)</Label>
															<Input id="rust-checkpoint-every" type="number" min="0" bind:value={rustParams["checkpoint-every"]} />
														</div>
													</div>
												</details>
											</div>
										</form>
									</Tabs.Content>

									<Tabs.Content value="python">
										<form
											class="space-y-4"
											onsubmit={(event) => {
												event.preventDefault();
												startTraining('python');
											}}
										>
											<div class="space-y-4">
												<details class="rounded-lg border bg-background/60 p-4" open>
													<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="py-train-parquet">Train Parquet</Label>
															<Input id="py-train-parquet" type="text" bind:value={pythonParams["train-parquet"]} placeholder="data/train" />
														</div>
														<div class="grid gap-2">
															<Label for="py-val-parquet">Val Parquet</Label>
															<Input id="py-val-parquet" type="text" bind:value={pythonParams["val-parquet"]} placeholder="data/val" />
														</div>
														<div class="grid gap-2">
															<Label for="py-test-parquet">Test Parquet</Label>
															<Input id="py-test-parquet" type="text" bind:value={pythonParams["test-parquet"]} placeholder="data/test" />
														</div>
														<div class="grid gap-2">
															<Label for="py-data-mode">Data Mode</Label>
															<select
																id="py-data-mode"
																bind:value={pythonDataMode}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="windowed">Windowed</option>
																<option value="full">Full file</option>
															</select>
														</div>
														{#if pythonDataMode === 'windowed'}
															<div class="grid gap-2">
																<Label for="py-window">Window Size</Label>
																<Input id="py-window" type="number" min="1" bind:value={pythonParams.window} />
															</div>
															<div class="grid gap-2">
																<Label for="py-step">Step Size</Label>
																<Input id="py-step" type="number" min="1" bind:value={pythonParams.step} />
															</div>
														{/if}
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">GA Settings</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="py-generations">Generations</Label>
															<Input id="py-generations" type="number" min="1" bind:value={pythonParams.generations} />
														</div>
														<div class="grid gap-2">
															<Label for="py-pop-size">Population Size</Label>
															<Input id="py-pop-size" type="number" min="1" bind:value={pythonParams["pop-size"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-workers">Workers</Label>
															<Input id="py-workers" type="number" min="0" bind:value={pythonParams.workers} />
														</div>
														<div class="grid gap-2">
															<Label for="py-mutation-sigma">Mutation Sigma</Label>
															<Input id="py-mutation-sigma" type="number" step="0.01" min="0" bind:value={pythonParams["mutation-sigma"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-eval-windows">Eval Windows</Label>
															<Input id="py-eval-windows" type="number" min="1" bind:value={pythonParams["eval-windows"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">PPO Settings</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="py-train-epochs">Train Epochs</Label>
															<Input id="py-train-epochs" type="number" min="1" bind:value={pythonParams["train-epochs"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-train-windows">Train Windows</Label>
															<Input id="py-train-windows" type="number" min="1" bind:value={pythonParams["train-windows"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-lr">Learning Rate</Label>
															<Input id="py-lr" type="number" step="0.0001" bind:value={pythonParams.lr} />
														</div>
														<div class="grid gap-2">
															<Label for="py-gamma">Gamma</Label>
															<Input id="py-gamma" type="number" step="0.01" min="0" max="1" bind:value={pythonParams.gamma} />
														</div>
														<div class="grid gap-2">
															<Label for="py-lam">Lambda</Label>
															<Input id="py-lam" type="number" step="0.01" bind:value={pythonParams.lam} />
														</div>
														<div class="grid gap-2">
															<Label for="py-initial-balance">Initial Balance</Label>
															<Input id="py-initial-balance" type="number" step="0.01" bind:value={pythonParams["initial-balance"]} />
														</div>
													</div>
												</details>

												<details class="rounded-lg border bg-background/60 p-4">
													<summary class="cursor-pointer text-sm font-semibold">Output &amp; Runtime</summary>
													<div class="mt-4 grid gap-4 md:grid-cols-2">
														<div class="grid gap-2">
															<Label for="py-outdir">Output Folder</Label>
															<Input id="py-outdir" type="text" bind:value={pythonParams.outdir} placeholder="runs_ga" />
														</div>
														<div class="grid gap-2">
															<Label for="py-device">Device</Label>
															<select
																id="py-device"
																bind:value={pythonParams.device}
																class="border-input bg-background ring-offset-background placeholder:text-muted-foreground flex h-9 w-full min-w-0 rounded-md border px-3 py-1 text-sm shadow-xs transition-[color,box-shadow] outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
															>
																<option value="">Auto</option>
																<option value="cpu">CPU</option>
																<option value="mps">MPS</option>
																<option value="cuda">CUDA</option>
															</select>
														</div>
														<div class="grid gap-2">
															<Label for="py-save-top-n">Save Top N</Label>
															<Input id="py-save-top-n" type="number" min="0" bind:value={pythonParams["save-top-n"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-save-every">Save Every (gens)</Label>
															<Input id="py-save-every" type="number" min="0" bind:value={pythonParams["save-every"]} />
														</div>
														<div class="grid gap-2">
															<Label for="py-checkpoint-every">Checkpoint Every (gens)</Label>
															<Input id="py-checkpoint-every" type="number" min="0" bind:value={pythonParams["checkpoint-every"]} />
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
								<Button
									variant={training ? "destructive" : "default"}
									onclick={toggleTraining}
									class="w-full"
									title={runTitle}
									disabled={!training && !canStartTraining}
								>
									{runLabel}
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
			<a href="/" class="text-sm text-muted-foreground hover:text-foreground">← Back to Dashboard</a>
			<h1 class="text-4xl font-bold tracking-tight">Train GA Model</h1>
		</div>

		<div class="space-y-6">
			<Card.Root>
				<Card.Header class="flex flex-row items-center justify-between">
					<Card.Title>Best Fitness by Generation</Card.Title>
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
							Waiting for the first generation results...
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
		</div>
	</main>
</div>

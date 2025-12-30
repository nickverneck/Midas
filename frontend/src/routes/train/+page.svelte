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

	let trainMode = $state<TrainMode>('rust');
	let consoleOutput = $state<ConsoleLine[]>([]);
	let training = $state(false);
	let paramsCollapsed = $state(false);
	let rustDataMode = $state<DataMode>('windowed');
	let pythonDataMode = $state<DataMode>('windowed');
	let abortController: AbortController | null = null;
	let stdoutBuffer = '';
	let fitnessByGen = $state(new Map<number, number>());
	
	let rustParams = $state({
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "",
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
		device: "",
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

	const fitnessRegex = /Gen\s+(\d+)\s+Top Performer:\s+(?:train\s+)?fitness\s+([-\d.]+)/i;
	const startLabel = $derived.by(() =>
		training ? 'Training...' : trainMode === 'rust' ? 'Start Rust Training' : 'Start Python Training'
	);
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

	const ingestStdout = (text: string) => {
		stdoutBuffer += text;
		const lines = stdoutBuffer.split(/\r?\n/);
		stdoutBuffer = lines.pop() ?? '';
		for (const line of lines) {
			const match = line.match(fitnessRegex);
			if (match) {
				updateFitness(Number(match[1]), Number(match[2]));
			}
		}
	};

	const buildRustParams = () => {
		const params = { ...rustParams } as Record<string, unknown>;
		if (rustDataMode === 'full') {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return params;
	};

	const buildPythonParams = () => {
		const params = { ...pythonParams } as Record<string, unknown>;
		if (pythonDataMode === 'full') {
			params["full-file"] = true;
		}
		return params;
	};

	async function startTraining(mode: TrainMode) {
		if (training) return;
		const params = mode === 'rust' ? buildRustParams() : buildPythonParams();
		abortController?.abort();
		abortController = new AbortController();
		training = true;
		consoleOutput = [{ type: 'system', text: `Starting ${mode.toUpperCase()} training...` }];
		fitnessByGen = new Map();
		stdoutBuffer = '';
		
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
							if (data.type === 'stdout') {
								ingestStdout(data.content);
							}
							consoleOutput = [...consoleOutput, { type: data.type, text: data.content }];
						} else if (data.type === 'error') {
							consoleOutput = [...consoleOutput, { type: 'error', text: data.content }];
						} else if (data.type === 'exit') {
							training = false;
							abortController = null;
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
		}
	}

	const stopTraining = () => {
		if (!training) return;
		abortController?.abort();
	};
</script>

<div class="p-8 space-y-8 max-w-[1600px] mx-auto">
	<div class="flex items-center gap-4">
        <a href="/" class="text-sm text-muted-foreground hover:text-foreground">← Back to Dashboard</a>
		<h1 class="text-4xl font-bold tracking-tight">Train GA Model</h1>
	</div>

	<div class="grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-8">
		<Card.Root class="self-start lg:sticky lg:top-6">
			{#if !paramsCollapsed}
				<Card.Header class="flex items-center justify-between">
					<Card.Title>Training Controls</Card.Title>
					<Button variant="ghost" size="sm" onclick={() => (paramsCollapsed = true)}>
						Collapse
					</Button>
				</Card.Header>
			{/if}
			<Card.Content
				class={paramsCollapsed ? "cursor-pointer" : ""}
				onclick={() => {
					if (paramsCollapsed) paramsCollapsed = false;
				}}
				title={paramsCollapsed ? "Click to expand parameters" : undefined}
			>
				<div class="space-y-2" onclick={(event) => event.stopPropagation()}>
					<Button onclick={() => startTraining(trainMode)} disabled={training} class="w-full">
						{startLabel}
					</Button>
					<Button variant="destructive" onclick={stopTraining} disabled={!training} class="w-full">
						Stop Training
					</Button>
				</div>

				{#if !paramsCollapsed}
					<div class="mt-4 max-h-[calc(100vh-260px)] overflow-y-auto pr-2">
						<Tabs.Root bind:value={trainMode} class="w-full">
							<Tabs.List class="grid w-full grid-cols-2 mb-4">
								<Tabs.Trigger value="rust">Rust (Primary)</Tabs.Trigger>
								<Tabs.Trigger value="python">Python (Legacy)</Tabs.Trigger>
							</Tabs.List>

							<Tabs.Content value="rust">
								<form class="space-y-4" onsubmit={(e) => { e.preventDefault(); startTraining('rust'); }}>
									<div class="grid gap-4 md:grid-cols-2">
										<div class="grid gap-2">
											<Label for="rust-outdir">Output Folder</Label>
											<Input id="rust-outdir" type="text" bind:value={rustParams.outdir} placeholder="runs_ga" />
										</div>
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
										<div class="grid gap-2">
											<Label for="rust-initial-balance">Initial Balance</Label>
											<Input id="rust-initial-balance" type="number" step="0.01" bind:value={rustParams["initial-balance"]} />
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
											<Label for="rust-hidden">Hidden Units</Label>
											<Input id="rust-hidden" type="number" min="1" bind:value={rustParams.hidden} />
										</div>
										<div class="grid gap-2">
											<Label for="rust-layers">Layers</Label>
											<Input id="rust-layers" type="number" min="1" bind:value={rustParams.layers} />
										</div>
										<div class="grid gap-2">
											<Label for="rust-eval-windows">Eval Windows</Label>
											<Input id="rust-eval-windows" type="number" min="1" bind:value={rustParams["eval-windows"]} />
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
								</form>
							</Tabs.Content>

							<Tabs.Content value="python">
								<form class="space-y-4" onsubmit={(e) => { e.preventDefault(); startTraining('python'); }}>
									<div class="grid gap-4 md:grid-cols-2">
										<div class="grid gap-2">
											<Label for="py-outdir">Output Folder</Label>
											<Input id="py-outdir" type="text" bind:value={pythonParams.outdir} placeholder="runs_ga" />
										</div>
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
										<div class="grid gap-2">
											<Label for="py-train-epochs">Train Epochs</Label>
											<Input id="py-train-epochs" type="number" min="1" bind:value={pythonParams["train-epochs"]} />
										</div>
										<div class="grid gap-2">
											<Label for="py-train-windows">Train Windows</Label>
											<Input id="py-train-windows" type="number" min="1" bind:value={pythonParams["train-windows"]} />
										</div>
										<div class="grid gap-2">
											<Label for="py-eval-windows">Eval Windows</Label>
											<Input id="py-eval-windows" type="number" min="1" bind:value={pythonParams["eval-windows"]} />
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
											<Input id="py-lam" type="number" step="0.01" min="0" max="1" bind:value={pythonParams.lam} />
										</div>
										<div class="grid gap-2">
											<Label for="py-initial-balance">Initial Balance</Label>
											<Input id="py-initial-balance" type="number" step="0.01" bind:value={pythonParams["initial-balance"]} />
										</div>
										<div class="grid gap-2">
											<Label for="py-mutation-sigma">Mutation Sigma</Label>
											<Input id="py-mutation-sigma" type="number" step="0.01" min="0" bind:value={pythonParams["mutation-sigma"]} />
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
								</form>
							</Tabs.Content>
						</Tabs.Root>
					</div>
				{/if}
			</Card.Content>
		</Card.Root>

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
					<ScrollArea class="h-[320px] w-full bg-zinc-950 p-4 rounded-md border border-zinc-800 font-mono text-sm">
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
	</div>
</div>

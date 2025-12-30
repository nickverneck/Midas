<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import { Badge } from "$lib/components/ui/badge";
	import * as Tabs from "$lib/components/ui/tabs";

	type TrainMode = 'rust' | 'python';
	type ConsoleLine = { type: string; text: string };

	let trainMode = $state<TrainMode>('rust');
	let consoleOutput = $state<ConsoleLine[]>([]);
	let training = $state(false);
	
	let rustParams = $state({
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
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

	async function startTraining(mode: TrainMode) {
		const params = mode === 'rust' ? rustParams : pythonParams;
		training = true;
		consoleOutput = [{ type: 'system', text: `Starting ${mode.toUpperCase()} training...` }];
		
		try {
			const response = await fetch('/api/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ engine: mode, params })
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
							consoleOutput = [...consoleOutput, { type: 'system', text: `Process exited with code ${data.code}` }];
						}
					}
				}
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: 'error', text: message }];
			training = false;
		}
	}
</script>

<div class="p-8 space-y-8 max-w-6xl mx-auto">
	<div class="flex items-center gap-4">
        <a href="/" class="text-sm text-muted-foreground hover:text-foreground">← Back to Dashboard</a>
		<h1 class="text-4xl font-bold tracking-tight">Train GA Model</h1>
	</div>

	<div class="grid grid-cols-1 md:grid-cols-3 gap-8">
		<Card.Root class="md:col-span-1">
			<Card.Header>
				<Card.Title>Parameters</Card.Title>
			</Card.Header>
			<Card.Content>
				<Tabs.Root bind:value={trainMode} class="w-full">
					<Tabs.List class="grid w-full grid-cols-2 mb-4">
						<Tabs.Trigger value="rust">Rust (Primary)</Tabs.Trigger>
						<Tabs.Trigger value="python">Python (Legacy)</Tabs.Trigger>
					</Tabs.List>

					<Tabs.Content value="rust">
						<form class="space-y-4" onsubmit={(e) => { e.preventDefault(); startTraining('rust'); }}>
							<Button type="submit" disabled={training} class="w-full">
								{training ? 'Training...' : 'Start Rust Training'}
							</Button>
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
									<Label for="rust-window">Window Size</Label>
									<Input id="rust-window" type="number" min="1" bind:value={rustParams.window} />
								</div>
								<div class="grid gap-2">
									<Label for="rust-step">Step Size</Label>
									<Input id="rust-step" type="number" min="1" bind:value={rustParams.step} />
								</div>
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
							<Button type="submit" disabled={training} class="w-full">
								{training ? 'Training...' : 'Start Python Training'}
							</Button>
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
									<Label for="py-window">Window Size</Label>
									<Input id="py-window" type="number" min="1" bind:value={pythonParams.window} />
								</div>
								<div class="grid gap-2">
									<Label for="py-step">Step Size</Label>
									<Input id="py-step" type="number" min="1" bind:value={pythonParams.step} />
								</div>
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
			</Card.Content>
		</Card.Root>

		<Card.Root class="md:col-span-2">
			<Card.Header class="flex flex-row items-center justify-between">
				<Card.Title>Console Output</Card.Title>
				{#if training}
					<Badge variant="outline" class="animate-pulse">Active</Badge>
				{/if}
			</Card.Header>
			<Card.Content>
				<ScrollArea class="h-[600px] w-full bg-zinc-950 p-4 rounded-md border border-zinc-800 font-mono text-sm">
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

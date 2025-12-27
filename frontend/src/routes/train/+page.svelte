<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import { Badge } from "$lib/components/ui/badge";

	let consoleOutput = $state([]);
	let training = $state(false);
	
	let params = $state({
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
        "mutation-sigma": 0.25
	});

	async function startTraining() {
		training = true;
		consoleOutput = [];
		
		try {
			const response = await fetch('/api/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(params)
			});

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
						} else if (data.type === 'exit') {
							training = false;
							consoleOutput = [...consoleOutput, { type: 'system', text: `Process exited with code ${data.code}` }];
						}
					}
				}
			}
		} catch (e) {
			consoleOutput = [...consoleOutput, { type: 'error', text: e.message }];
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
				<form class="space-y-4" onsubmit={(e) => { e.preventDefault(); startTraining(); }}>
					<div class="grid gap-2">
                        <Label for="generations">Generations</Label>
                        <Input id="generations" type="number" bind:value={params.generations} />
                    </div>
                    <div class="grid gap-2">
                        <Label for="pop-size">Population Size</Label>
                        <Input id="pop-size" type="number" bind:value={params["pop-size"]} />
                    </div>
                    <div class="grid gap-2">
                        <Label for="workers">Workers</Label>
                        <Input id="workers" type="number" bind:value={params.workers} />
                    </div>
                    <div class="grid gap-2">
                        <Label for="lr">Learning Rate</Label>
                        <Input id="lr" type="number" step="0.0001" bind:value={params.lr} />
                    </div>
                    <div class="grid gap-2">
                        <Label for="window">Window Size</Label>
                        <Input id="window" type="number" bind:value={params.window} />
                    </div>
                    <div class="grid gap-2">
                        <Label for="step">Step Size</Label>
                        <Input id="step" type="number" bind:value={params.step} />
                    </div>
                    
					<Button type="submit" disabled={training} class="w-full">
						{training ? 'Training...' : 'Start Training'}
					</Button>
				</form>
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
						<div class={line.type === 'stderr' ? 'text-red-400' : line.type === 'system' ? 'text-blue-400' : 'text-zinc-300'}>
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

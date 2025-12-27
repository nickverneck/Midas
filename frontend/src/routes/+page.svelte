<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
    import GaChart from "$lib/components/GaChart.svelte";

	let logs = $state([]);
	let loading = $state(true);

	async function fetchLogs() {
		try {
			const res = await fetch('/api/logs');
			logs = await res.json();
		} catch (e) {
			console.error('Failed to fetch logs', e);
		} finally {
			loading = false;
		}
	}

	onMount(fetchLogs);

    let chartData = $derived({
        labels: logs.map(l => `G${l.gen} I${l.idx}`),
        datasets: [
            {
                label: 'Fitness',
                data: logs.map(l => l.fitness),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                tension: 0.1
            },
            {
                label: 'Eval PNL',
                data: logs.map(l => l.eval_pnl),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                tension: 0.1
            }
        ]
    });
</script>

<div class="p-8 space-y-8">
	<div class="flex justify-between items-center">
		<h1 class="text-4xl font-bold tracking-tight">Midas GA Dashboard</h1>
        <div class="flex gap-4">
            <a href="/train" class="px-4 py-2 bg-primary text-primary-foreground rounded-md font-medium hover:bg-primary/90 transition-colors">
                New Training Run
            </a>
		    <button onclick={fetchLogs} class="px-4 py-2 bg-secondary text-secondary-foreground rounded-md font-medium hover:bg-secondary/80 transition-colors">
                Refresh Logs
            </button>
        </div>
	</div>

	{#if loading}
		<div class="flex justify-center items-center h-64">
			<Badge variant="outline" class="animate-pulse">Loading logs...</Badge>
		</div>
	{:else}
		<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
			<Card.Root>
				<Card.Header>
					<Card.Title>Total Generations</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{Math.max(...logs.map(l => l.gen)) + 1}</p>
				</Card.Content>
			</Card.Root>
			<Card.Root>
				<Card.Header>
					<Card.Title>Best Fitness</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold text-green-500">{Math.max(...logs.map(l => l.fitness)).toFixed(2)}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root>
				<Card.Header>
					<Card.Title>Avg Sortino</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{(logs.reduce((acc, l) => acc + l.eval_sortino, 0) / logs.length).toFixed(2)}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root>
				<Card.Header>
					<Card.Title>Total Candidates</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{logs.length}</p>
				</Card.Content>
			</Card.Root>
		</div>

		<Card.Root>
			<Card.Header>
				<Card.Title>Training Progress</Card.Title>
			</Card.Header>
			<Card.Content class="h-[400px]">
                <GaChart data={chartData} />
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header>
				<Card.Title>Candidate Metrics</Card.Title>
			</Card.Header>
			<Card.Content>
                <ScrollArea class="h-[500px]">
                    <Table.Root>
                        <Table.Header>
                            <Table.Row>
                                <Table.Head>Gen</Table.Head>
                                <Table.Head>Idx</Table.Head>
                                <Table.Head>Fitness</Table.Head>
                                <Table.Head>PNL</Table.Head>
                                <Table.Head>Sortino</Table.Head>
                                <Table.Head>Drawdown</Table.Head>
                                <Table.Head>Weights (PNL/Sortino/MDD)</Table.Head>
                            </Table.Row>
                        </Table.Header>
                        <Table.Body>
                            {#each logs.slice().reverse() as log}
                                <Table.Row>
                                    <Table.Cell>{log.gen}</Table.Cell>
                                    <Table.Cell>{log.idx}</Table.Cell>
                                    <Table.Cell class="font-medium text-blue-500">{log.fitness.toFixed(4)}</Table.Cell>
                                    <Table.Cell class={log.eval_pnl >= 0 ? 'text-green-500' : 'text-red-500'}>{log.eval_pnl.toFixed(4)}</Table.Cell>
                                    <Table.Cell>{log.eval_sortino.toFixed(4)}</Table.Cell>
                                    <Table.Cell class="text-red-400">{log.eval_drawdown.toFixed(4)}</Table.Cell>
                                    <Table.Cell class="text-xs text-muted-foreground">{log.w_pnl.toFixed(2)} / {log.w_sortino.toFixed(2)} / {log.w_mdd.toFixed(2)}</Table.Cell>
                                </Table.Row>
                            {/each}
                        </Table.Body>
                    </Table.Root>
                </ScrollArea>
			</Card.Content>
		</Card.Root>
	{/if}
</div>

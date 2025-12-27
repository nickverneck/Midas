<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
    import * as Tabs from "$lib/components/ui/tabs";
    import * as Alert from "$lib/components/ui/alert";
    import { Separator } from "$lib/components/ui/separator";
    import GaChart from "$lib/components/GaChart.svelte";
    import { AlertCircle, TrendingUp, Info, ListFilter, Bug } from "lucide-svelte";

	let logs = $state([]);
	let loading = $state(true);
    let error = $state("");

	async function fetchLogs() {
        loading = true;
		try {
			const res = await fetch('/api/logs');
            if (!res.ok) throw new Error("Failed to fetch logs");
			logs = await res.json();
		} catch (e) {
			console.error('Failed to fetch logs', e);
            error = e.message;
		} finally {
			loading = false;
		}
	}

	onMount(fetchLogs);

    // Dynamic key detection
    let metricKey = $derived('eval_sortino');
    let wMetricKey = $derived('w_sortino');
    let metricLabel = $derived('SORTINO');

    // Data Aggregation by Generation
    let genData = $derived.by(() => {
        if (logs.length === 0) return [];
        
        const gens = new Map();
        logs.forEach(l => {
            if (!gens.has(l.gen)) gens.set(l.gen, []);
            gens.get(l.gen).push(l);
        });

        return Array.from(gens.entries()).map(([gen, individuals]) => {
            const fitnesses = individuals.map(i => i.fitness);
            const pnls = individuals.map(i => i.eval_pnl);
            const metrics = individuals.map(i => i[metricKey]);
            
            return {
                gen,
                count: individuals.length,
                bestFitness: Math.max(...fitnesses),
                avgFitness: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
                bestPnl: Math.max(...pnls),
                avgPnl: pnls.reduce((a, b) => a + b, 0) / pnls.length,
                bestMetric: Math.max(...metrics),
                avgMetric: metrics.reduce((a, b) => a + b, 0) / metrics.length,
            };
        }).sort((a, b) => a.gen - b.gen);
    });

    // Chart: Fitness Evolution
    let fitnessChartData = $derived({
        labels: genData.map(g => `Gen ${g.gen}`),
        datasets: [
            {
                label: 'Best Fitness',
                data: genData.map(g => g.bestFitness),
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                tension: 0.1
            },
            {
                label: 'Avg Fitness',
                data: genData.map(g => g.avgFitness),
                borderColor: 'rgb(147, 197, 253)',
                backgroundColor: 'rgba(147, 197, 253, 0.2)',
                borderDash: [5, 5],
                tension: 0.1
            }
        ]
    });

    // Chart: PNL Evolution
    let pnlChartData = $derived({
        labels: genData.map(g => `Gen ${g.gen}`),
        datasets: [
            {
                label: `Best Eval PNL`,
                data: genData.map(g => g.bestPnl),
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.5)',
                tension: 0.1
            },
            {
                label: `Best ${metricLabel}`,
                data: genData.map(g => g.bestMetric),
                borderColor: 'rgb(168, 85, 247)',
                backgroundColor: 'rgba(168, 85, 247, 0.5)',
                yAxisID: 'y1',
                tension: 0.1
            }
        ]
    });

    // Issues Detection
    let issues = $derived.by(() => {
        const list = [];
        if (logs.length === 0) return list;

        // 1. Fitness Outliers
        const highFitness = logs.filter(l => l.fitness > 1000);
        if (highFitness.length > 0) {
            list.push({
                type: 'warning',
                title: 'Fitness Outliers Detected',
                message: `${highFitness.length} individuals have fitness > 1000. This may indicate logic errors or extreme lucky outliers corrupting selection.`,
                items: highFitness.slice(0, 5).map(i => `Gen ${i.gen}, Idx ${i.idx}: ${i.fitness.toFixed(2)}`)
            });
        }

        // 2. Metric Capping
        const capped = logs.filter(l => l[metricKey] >= 50.0);
        if (capped.length > 0) {
            list.push({
                type: 'info',
                title: `${metricLabel} Cap Hit`,
                message: `${capped.length} individuals hit the cap of 50.0 for ${metricLabel}. Consider raising the cap if they are consistently flatlining at max.`,
                items: capped.slice(0, 5).map(i => `Gen ${i.gen}, Idx ${i.idx}`)
            });
        }

        // 3. Zero Drawdown
        const zeroDD = logs.filter(l => l.eval_drawdown === 0 && l.eval_pnl !== 0);
        if (zeroDD.length > 0) {
            list.push({
                type: 'warning',
                title: 'Zero Drawdown (Possible Fake Results)',
                message: `${zeroDD.length} individuals have 0% drawdown but non-zero PNL. This often indicates insufficient evaluation data or "one-hit wonder" trades.`,
                items: zeroDD.slice(0, 5).map(i => `Gen ${i.gen}, Idx ${i.idx}`)
            });
        }

        // 4. Negative Returns
        const negativeRet = logs.filter(l => l.eval_ret_mean < 0);
        if (negativeRet.length === logs.length) {
            list.push({
                type: 'destructive',
                title: 'Universal Negative Returns',
                message: "Every single individual in the log has a negative return mean. The strategy logic or features might be fundamentally flawed."
            });
        }

        // 5. Inconsistent Population
        const counts = genData.map(g => g.count);
        const uniqueCounts = new Set(counts);
        if (uniqueCounts.size > 1) {
            list.push({
                type: 'info',
                title: 'Inconsistent Population Sizes',
                message: `Population sizes vary across generations: ${Array.from(uniqueCounts).join(', ')}. Ensure this is intentional (e.g. elite pruning).`
            });
        }

        return list;
    });

</script>

<div class="p-8 space-y-8 max-w-[1600px] mx-auto">
	<div class="flex justify-between items-center bg-card p-6 rounded-xl border shadow-sm">
		<div>
            <h1 class="text-4xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">Midas Dashboard</h1>
            <p class="text-muted-foreground mt-1">Analyzing Training Performance & Evolution</p>
        </div>
        <div class="flex gap-4">
            <a href="/train" class="px-5 py-2.5 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-all flex items-center gap-2 shadow-sm">
                <TrendingUp size={18} /> New Run
            </a>
		    <button onclick={fetchLogs} class="px-5 py-2.5 bg-secondary text-secondary-foreground rounded-lg font-semibold hover:bg-secondary/80 transition-all flex items-center gap-2 border">
                <ListFilter size={18} /> Refresh
            </button>
        </div>
	</div>

    {#if error}
        <Alert.Root variant="destructive">
            <AlertCircle class="h-4 w-4" />
            <Alert.Title>Error Loading Logs</Alert.Title>
            <Alert.Description>{error}</Alert.Description>
        </Alert.Root>
    {/if}

	{#if loading}
		<div class="flex flex-col justify-center items-center h-[60vh] gap-4">
			<div class="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
			<p class="text-muted-foreground font-medium animate-pulse">Processing {logs.length || "thousands of"} individuals...</p>
		</div>
	{:else}
		<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
			<Card.Root class="border-l-4 border-l-blue-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Generations</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{genData.length}</p>
				</Card.Content>
			</Card.Root>
			<Card.Root class="border-l-4 border-l-green-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Global Best Fitness</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold text-green-500">{Math.max(...logs.map(l => l.fitness)).toFixed(2)}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root class="border-l-4 border-l-purple-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Avg {metricLabel}</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{(logs.reduce((acc, l) => acc + (l[metricKey] || 0), 0) / (logs.length || 1)).toFixed(2)}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root class="border-l-4 border-l-orange-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Total Processed</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{logs.length.toLocaleString()}</p>
				</Card.Content>
			</Card.Root>
		</div>

        <Tabs.Root value="overview" class="w-full">
            <Tabs.List class="grid w-full grid-cols-4 lg:w-[600px] mb-6">
                <Tabs.Trigger value="overview">Overview</Tabs.Trigger>
                <Tabs.Trigger value="evolution">Evolution</Tabs.Trigger>
                <Tabs.Trigger value="issues" class="flex gap-2 items-center">
                    Issues {#if issues.length > 0}<Badge variant="destructive" class="ml-1 px-1.5 py-0 text-[10px]">{issues.length}</Badge>{/if}
                </Tabs.Trigger>
                <Tabs.Trigger value="data">Full Log</Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="overview" class="space-y-6">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card.Root>
                        <Card.Header>
                            <Card.Title class="flex items-center gap-2"><TrendingUp size={20} class="text-blue-500"/> Fitness Evolution</Card.Title>
                            <Card.Description>Maximum and average fitness progress per generation</Card.Description>
                        </Card.Header>
                        <Card.Content class="h-[350px]">
                            <GaChart data={fitnessChartData} />
                        </Card.Content>
                    </Card.Root>

                    <Card.Root>
                        <Card.Header>
                            <Card.Title class="flex items-center gap-2"><TrendingUp size={20} class="text-green-500"/> Performance Evolution</Card.Title>
                            <Card.Description>Best PNL and {metricLabel} across training windows</Card.Description>
                        </Card.Header>
                        <Card.Content class="h-[350px]">
                            <GaChart data={pnlChartData} />
                        </Card.Content>
                    </Card.Root>
                </div>
                
                <Card.Root>
                    <Card.Header>
                        <Card.Title>Recent Highlights</Card.Title>
                    </Card.Header>
                    <Card.Content>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {#each genData.slice(-3).reverse() as g}
                                <div class="p-4 bg-muted/50 rounded-lg border">
                                    <h4 class="font-bold text-lg">Generation {g.gen}</h4>
                                    <div class="mt-2 space-y-1 text-sm">
                                        <div class="flex justify-between"><span>Max Fitness:</span> <span class="font-mono text-blue-500">{g.bestFitness.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max PNL:</span> <span class="font-mono text-green-500">{g.bestPnl.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max {metricLabel}:</span> <span class="font-mono text-purple-500">{g.bestMetric.toFixed(2)}</span></div>
                                    </div>
                                </div>
                            {/each}
                        </div>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>

            <Tabs.Content value="evolution">
                <Card.Root>
                    <Card.Header>
                        <Card.Title>Detailed Generation Analysis</Card.Title>
                    </Card.Header>
                    <Card.Content>
                        <Table.Root>
                            <Table.Header>
                                <Table.Row>
                                    <Table.Head>Gen</Table.Head>
                                    <Table.Head>Population</Table.Head>
                                    <Table.Head>Best Fitness</Table.Head>
                                    <Table.Head>Avg Fitness</Table.Head>
                                    <Table.Head>Best PNL</Table.Head>
                                    <Table.Head>Best {metricLabel}</Table.Head>
                                </Table.Row>
                            </Table.Header>
                            <Table.Body>
                                {#each genData.slice().reverse() as g}
                                    <Table.Row>
                                        <Table.Cell class="font-bold">{g.gen}</Table.Cell>
                                        <Table.Cell>{g.count}</Table.Cell>
                                        <Table.Cell class="text-blue-500 font-medium">{g.bestFitness.toFixed(4)}</Table.Cell>
                                        <Table.Cell class="text-muted-foreground">{g.avgFitness.toFixed(4)}</Table.Cell>
                                        <Table.Cell class={g.bestPnl >= 0 ? 'text-green-500' : 'text-red-500'}>{g.bestPnl.toFixed(4)}</Table.Cell>
                                        <Table.Cell class="text-purple-500">{g.bestMetric.toFixed(4)}</Table.Cell>
                                    </Table.Row>
                                {/each}
                            </Table.Body>
                        </Table.Root>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>

            <Tabs.Content value="issues" class="space-y-4">
                {#if issues.length === 0}
                    <div class="flex flex-col items-center justify-center py-20 text-muted-foreground">
                        <Info size={48} class="mb-4 opacity-20"/>
                        <p class="text-xl font-medium">No major issues detected</p>
                        <p class="text-sm">Training data looks within expected heuristic bounds.</p>
                    </div>
                {:else}
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {#each issues as issue}
                            <Card.Root class={issue.type === 'destructive' ? 'border-red-500' : issue.type === 'warning' ? 'border-amber-500' : 'border-blue-500'}>
                                <Card.Header class="flex flex-row items-center gap-4 py-4">
                                    <div class={`p-2 rounded-full ${issue.type === 'destructive' ? 'bg-red-100 text-red-600' : issue.type === 'warning' ? 'bg-amber-100 text-amber-600' : 'bg-blue-100 text-blue-600'}`}>
                                        <Bug size={24} />
                                    </div>
                                    <div>
                                        <Card.Title class="text-lg">{issue.title}</Card.Title>
                                    </div>
                                </Card.Header>
                                <Card.Content class="pb-4">
                                    <p class="text-sm text-muted-foreground">{issue.message}</p>
                                    {#if issue.items}
                                        <div class="mt-3 p-3 bg-muted rounded-md text-xs font-mono space-y-1">
                                            {#each issue.items as item}
                                                <div>• {item}</div>
                                            {/each}
                                        </div>
                                    {/if}
                                </Card.Content>
                            </Card.Root>
                        {/each}
                    </div>
                {/if}
            </Tabs.Content>

            <Tabs.Content value="data">
                <Card.Root>
                    <Card.Header class="flex flex-row items-center justify-between pb-4">
                        <div>
                            <Card.Title>Raw Training Logs</Card.Title>
                            <Card.Description>Scroll to view all {logs.length} data points</Card.Description>
                        </div>
                    </Card.Header>
                    <Card.Content>
                        <ScrollArea class="h-[600px] rounded-md border">
                            <Table.Root>
                                <Table.Header class="sticky top-0 bg-background z-10 shadow-sm border-b">
                                    <Table.Row>
                                        <Table.Head class="w-[80px]">Gen</Table.Head>
                                        <Table.Head class="w-[80px]">Idx</Table.Head>
                                        <Table.Head>Fitness</Table.Head>
                                        <Table.Head>PNL</Table.Head>
                                        <Table.Head>{metricLabel}</Table.Head>
                                        <Table.Head>DD</Table.Head>
                                        <Table.Head class="hidden md:table-cell text-right">Weights (PNL/{metricLabel}/MDD)</Table.Head>
                                    </Table.Row>
                                </Table.Header>
                                <Table.Body>
                                    {#each logs.slice().reverse() as log}
                                        <Table.Row class="hover:bg-muted/30 transition-colors">
                                            <Table.Cell>{log.gen}</Table.Cell>
                                            <Table.Cell>{log.idx}</Table.Cell>
                                            <Table.Cell class="font-medium text-blue-500">{log.fitness.toFixed(4)}</Table.Cell>
                                            <Table.Cell class={log.eval_pnl >= 0 ? 'text-green-500 font-medium' : 'text-red-500'}>{log.eval_pnl.toFixed(4)}</Table.Cell>
                                            <Table.Cell>{log[metricKey]?.toFixed(4)}</Table.Cell>
                                            <Table.Cell class="text-red-400">{log.eval_drawdown.toFixed(4)}%</Table.Cell>
                                            <Table.Cell class="text-right hidden md:table-cell text-[10px] text-muted-foreground font-mono">
                                                {log.w_pnl.toFixed(2)} / {log[wMetricKey]?.toFixed(2)} / {log.w_mdd.toFixed(2)}
                                            </Table.Cell>
                                        </Table.Row>
                                    {/each}
                                </Table.Body>
                            </Table.Root>
                        </ScrollArea>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>
        </Tabs.Root>
	{/if}
</div>

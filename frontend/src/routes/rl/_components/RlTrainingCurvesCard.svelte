<script lang="ts">
	import GaChart from "$lib/components/GaChart.svelte";
	import * as Card from "$lib/components/ui/card";
	import * as Tabs from "$lib/components/ui/tabs";
	import type { ChartTab, RlChartsViewModel } from "../types";

	type Props = {
		activeLogDir: string;
		epochCount: number;
		chartTab: ChartTab;
		charts: RlChartsViewModel;
	};

	let { activeLogDir, epochCount, chartTab = $bindable(), charts }: Props = $props();
</script>

<Card.Root class="xl:col-span-2">
	<Card.Header>
		<Card.Title>Training Curves</Card.Title>
		<Card.Description>{activeLogDir} · {epochCount.toLocaleString()} epochs</Card.Description>
	</Card.Header>
	<Card.Content>
		<Tabs.Root bind:value={chartTab} class="w-full">
			<Tabs.List class="mb-4 grid w-full grid-cols-2 lg:w-[1240px] lg:grid-cols-8">
				<Tabs.Trigger value="fitness">Fitness</Tabs.Trigger>
				<Tabs.Trigger value="performance">Performance</Tabs.Trigger>
				<Tabs.Trigger value="drawdown">Drawdown</Tabs.Trigger>
				<Tabs.Trigger value="frontier">Frontier</Tabs.Trigger>
				<Tabs.Trigger value="loss">Loss</Tabs.Trigger>
				<Tabs.Trigger value="gradients">Gradients</Tabs.Trigger>
				<Tabs.Trigger value="policy">Policy</Tabs.Trigger>
				<Tabs.Trigger value="probe">Probe</Tabs.Trigger>
			</Tabs.List>

			<Tabs.Content value="fitness">
				<div class="h-[320px]">
					<GaChart data={charts.fitness.data} options={charts.fitness.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="performance">
				<div class="h-[320px]">
					<GaChart data={charts.performance.data} options={charts.performance.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="drawdown">
				<div class="h-[320px]">
					<GaChart data={charts.drawdown.data} options={charts.drawdown.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="frontier">
				{#if charts.frontier.data.datasets.length > 0}
					<div class="space-y-3">
						<div class="h-[320px]">
							<GaChart
								data={charts.frontier.data}
								options={charts.frontier.options}
								type={charts.frontier.type}
							/>
						</div>
						{#if charts.frontier.note}
							<p class="text-xs text-muted-foreground">{charts.frontier.note}</p>
						{/if}
					</div>
				{:else}
					<div class="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
						{charts.frontier.emptyMessage}
					</div>
				{/if}
			</Tabs.Content>

			<Tabs.Content value="loss">
				<div class="h-[320px]">
					<GaChart data={charts.loss.data} options={charts.loss.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="gradients">
				<div class="h-[320px]">
					<GaChart data={charts.gradients.data} options={charts.gradients.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="policy">
				<div class="h-[320px]">
					<GaChart data={charts.policy.data} options={charts.policy.options} />
				</div>
			</Tabs.Content>

			<Tabs.Content value="probe">
				<div class="h-[320px]">
					<GaChart data={charts.probe.data} options={charts.probe.options} />
				</div>
			</Tabs.Content>
		</Tabs.Root>
	</Card.Content>
</Card.Root>

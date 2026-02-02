<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { Separator } from "$lib/components/ui/separator";
	import { LineChart } from "lucide-svelte";
	import type { AnalyzerConfig } from "../types";

	type Props = {
		analyzer: AnalyzerConfig;
	};

	let { analyzer }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<LineChart size={18} />
			Signal Builder
		</Card.Title>
		<Card.Description>Define the indicator sweep and crossover logic.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-5">
		<div class="grid gap-4 lg:grid-cols-2">
			<div class="space-y-3 rounded-lg border bg-muted/20 p-4">
				<p class="text-xs font-semibold uppercase text-muted-foreground">Indicator A</p>
				<div class="space-y-2">
					<Label>Type</Label>
					<Select.Root bind:value={analyzer.indicatorA.kind}>
						<Select.Trigger class="w-full">{analyzer.indicatorA.kind}</Select.Trigger>
					<Select.Content>
					<Select.Item value="sma">sma</Select.Item>
					<Select.Item value="ema">ema</Select.Item>
					<Select.Item value="hma">hma</Select.Item>
					<Select.Item value="wma">wma</Select.Item>
					<Select.Item value="price">price</Select.Item>
				</Select.Content>
			</Select.Root>
		</div>
		<div class="grid gap-2 sm:grid-cols-3">
				<div class="space-y-1">
					<Label class="text-xs">Start</Label>
					<Input type="number" bind:value={analyzer.indicatorA.start} min="1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End</Label>
						<Input type="number" bind:value={analyzer.indicatorA.end} min="1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step</Label>
						<Input type="number" bind:value={analyzer.indicatorA.step} min="1" />
					</div>
				</div>
			</div>
			<div class="space-y-3 rounded-lg border bg-muted/20 p-4">
				<p class="text-xs font-semibold uppercase text-muted-foreground">Indicator B</p>
				<div class="space-y-2">
					<Label>Type</Label>
					<Select.Root bind:value={analyzer.indicatorB.kind}>
						<Select.Trigger class="w-full">{analyzer.indicatorB.kind}</Select.Trigger>
						<Select.Content>
						<Select.Item value="sma">sma</Select.Item>
						<Select.Item value="ema">ema</Select.Item>
						<Select.Item value="hma">hma</Select.Item>
						<Select.Item value="wma">wma</Select.Item>
						<Select.Item value="price">price</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
				<div class="grid gap-2 sm:grid-cols-3">
					<div class="space-y-1">
						<Label class="text-xs">Start</Label>
						<Input type="number" bind:value={analyzer.indicatorB.start} min="1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End</Label>
						<Input type="number" bind:value={analyzer.indicatorB.end} min="1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step</Label>
						<Input type="number" bind:value={analyzer.indicatorB.step} min="1" />
					</div>
				</div>
			</div>
		</div>
		<Separator />
		<div class="grid gap-4 sm:grid-cols-2">
			<div class="space-y-2">
				<Label>Buy Action</Label>
				<Select.Root bind:value={analyzer.buyAction}>
					<Select.Trigger class="w-full">{analyzer.buyAction}</Select.Trigger>
					<Select.Content>
						<Select.Item value="crossover">crossover</Select.Item>
						<Select.Item value="crossunder">crossunder</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
			<div class="space-y-2">
				<Label>Sell Action</Label>
				<Select.Root bind:value={analyzer.sellAction}>
					<Select.Trigger class="w-full">{analyzer.sellAction}</Select.Trigger>
					<Select.Content>
						<Select.Item value="crossover">crossover</Select.Item>
						<Select.Item value="crossunder">crossunder</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
		</div>
		<p class="text-xs text-muted-foreground">
			Signals run on the same indicator pair; buy and sell actions can differ.
		</p>
	</Card.Content>
</Card.Root>

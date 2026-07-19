<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import * as Select from "$lib/components/ui/select";
	import { BarChart3 } from "lucide-svelte";
	import type { AnalyzerConfig } from "../types";

	type Props = {
		analyzer: AnalyzerConfig;
		invalidVolumeBarSize: boolean;
	};

	let { analyzer, invalidVolumeBarSize }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<BarChart3 size={18} />
			Bar Settings
		</Card.Title>
		<Card.Description>Choose the bar stream and signal price source.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		<div class="grid gap-4 sm:grid-cols-2">
			<div class="space-y-2">
				<Label>Bar Type</Label>
				<Select.Root bind:value={analyzer.barKind}>
					<Select.Trigger class="w-full">
						{analyzer.barKind === "volume" ? "Volume" : "Price-action"}
					</Select.Trigger>
					<Select.Content>
						<Select.Item value="price-action">Price-action</Select.Item>
						<Select.Item value="volume">Volume</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>

			<div class="space-y-2">
				<Label>Price Source</Label>
				<Select.Root bind:value={analyzer.priceSource}>
					<Select.Trigger class="w-full">
						{analyzer.priceSource === "heikin-ashi" ? "Heikin-Ashi" : "OHLC"}
					</Select.Trigger>
					<Select.Content>
						<Select.Item value="ohlc">OHLC</Select.Item>
						<Select.Item value="heikin-ashi">Heikin-Ashi</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>

			{#if analyzer.barKind === "volume"}
				<div class="space-y-2 sm:col-span-2">
					<Label for="analyzer-volume-bar-size">Volume Bar Size</Label>
					<Input
						id="analyzer-volume-bar-size"
						type="number"
						min="1"
						step="1"
						bind:value={analyzer.volumeBarSize}
						aria-invalid={invalidVolumeBarSize}
						placeholder="Contracts per bar"
					/>
					{#if invalidVolumeBarSize}
						<p class="text-xs text-destructive">Enter a volume bar size greater than 0.</p>
					{:else}
						<p class="text-xs text-muted-foreground">Required for volume bars.</p>
					{/if}
				</div>
			{/if}
		</div>

		<p class="text-xs text-muted-foreground">
			Heikin-Ashi affects signals only; fills remain raw OHLC.
		</p>
	</Card.Content>
</Card.Root>

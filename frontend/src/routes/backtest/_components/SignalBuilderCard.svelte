<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import * as Select from "$lib/components/ui/select";
	import { Separator } from "$lib/components/ui/separator";
	import { LineChart } from "lucide-svelte";
	import type { AnalyzerConfig, IndicatorKind, IndicatorSweepParam } from "../types";

	type Props = {
		analyzer: AnalyzerConfig;
	};

	const sweepParamLabel: Record<IndicatorSweepParam, string> = {
		period: "Period",
		fast: "Fast",
		slow: "Slow",
		offset: "Offset",
		sigma: "Sigma"
	};

	const sweepOptionsForKind = (kind: IndicatorKind): IndicatorSweepParam[] => {
		switch (kind) {
			case "kama":
				return ["period", "fast", "slow"];
			case "alma":
				return ["period", "offset", "sigma"];
			default:
				return ["period"];
		}
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
							<Select.Item value="kama">kama (Kaufman)</Select.Item>
							<Select.Item value="alma">alma (Arnaud Legoux)</Select.Item>
							<Select.Item value="price">price</Select.Item>
						</Select.Content>
					</Select.Root>
				</div>
				<div class="space-y-2">
					<Label>Sweep Parameter</Label>
					<div class="flex flex-wrap gap-x-4 gap-y-2 text-xs">
						{#each sweepOptionsForKind(analyzer.indicatorA.kind) as option}
							<label class="flex items-center gap-2">
								<input
									type="radio"
									name="indicator-a-sweep"
									value={option}
									bind:group={analyzer.indicatorA.sweepParam}
								/>
								<span>{sweepParamLabel[option]}</span>
							</label>
						{/each}
					</div>
				</div>
				<div class="grid gap-2 sm:grid-cols-3">
					<div class="space-y-1">
						<Label class="text-xs">Start ({sweepParamLabel[analyzer.indicatorA.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorA.start} />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End ({sweepParamLabel[analyzer.indicatorA.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorA.end} />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step ({sweepParamLabel[analyzer.indicatorA.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorA.step} />
					</div>
				</div>
				{#if analyzer.indicatorA.kind !== "price"}
					<div class="space-y-1">
						<Label class="text-xs">Period {analyzer.indicatorA.sweepParam === "period" ? "(swept)" : "(fixed)"}</Label>
						<Input
							type="number"
							min="1"
							step="1"
							bind:value={analyzer.indicatorA.period}
							disabled={analyzer.indicatorA.sweepParam === "period"}
						/>
					</div>
				{/if}
				{#if analyzer.indicatorA.kind === "kama"}
					<div class="grid gap-2 sm:grid-cols-2">
						<div class="space-y-1">
							<Label class="text-xs">Fast {analyzer.indicatorA.sweepParam === "fast" ? "(swept)" : "(fixed)"}</Label>
							<Input
								type="number"
								min="1"
								step="1"
								bind:value={analyzer.indicatorA.kamaFast}
								disabled={analyzer.indicatorA.sweepParam === "fast"}
							/>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">Slow {analyzer.indicatorA.sweepParam === "slow" ? "(swept)" : "(fixed)"}</Label>
							<Input
								type="number"
								min="1"
								step="1"
								bind:value={analyzer.indicatorA.kamaSlow}
								disabled={analyzer.indicatorA.sweepParam === "slow"}
							/>
						</div>
					</div>
				{/if}
				{#if analyzer.indicatorA.kind === "alma"}
					<div class="grid gap-2 sm:grid-cols-2">
						<div class="space-y-1">
							<Label class="text-xs">
								Offset {analyzer.indicatorA.sweepParam === "offset" ? "(swept)" : "(fixed)"}
							</Label>
							<Input
								type="number"
								step="0.01"
								min="0"
								max="1"
								bind:value={analyzer.indicatorA.almaOffset}
								disabled={analyzer.indicatorA.sweepParam === "offset"}
							/>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">
								Sigma {analyzer.indicatorA.sweepParam === "sigma" ? "(swept)" : "(fixed)"}
							</Label>
							<Input
								type="number"
								step="0.1"
								min="0.000001"
								bind:value={analyzer.indicatorA.almaSigma}
								disabled={analyzer.indicatorA.sweepParam === "sigma"}
							/>
						</div>
					</div>
				{/if}
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
							<Select.Item value="kama">kama (Kaufman)</Select.Item>
							<Select.Item value="alma">alma (Arnaud Legoux)</Select.Item>
							<Select.Item value="price">price</Select.Item>
						</Select.Content>
					</Select.Root>
				</div>
				<div class="space-y-2">
					<Label>Sweep Parameter</Label>
					<div class="flex flex-wrap gap-x-4 gap-y-2 text-xs">
						{#each sweepOptionsForKind(analyzer.indicatorB.kind) as option}
							<label class="flex items-center gap-2">
								<input
									type="radio"
									name="indicator-b-sweep"
									value={option}
									bind:group={analyzer.indicatorB.sweepParam}
								/>
								<span>{sweepParamLabel[option]}</span>
							</label>
						{/each}
					</div>
				</div>
				<div class="grid gap-2 sm:grid-cols-3">
					<div class="space-y-1">
						<Label class="text-xs">Start ({sweepParamLabel[analyzer.indicatorB.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorB.start} />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End ({sweepParamLabel[analyzer.indicatorB.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorB.end} />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step ({sweepParamLabel[analyzer.indicatorB.sweepParam]})</Label>
						<Input type="number" step="any" bind:value={analyzer.indicatorB.step} />
					</div>
				</div>
				{#if analyzer.indicatorB.kind !== "price"}
					<div class="space-y-1">
						<Label class="text-xs">Period {analyzer.indicatorB.sweepParam === "period" ? "(swept)" : "(fixed)"}</Label>
						<Input
							type="number"
							min="1"
							step="1"
							bind:value={analyzer.indicatorB.period}
							disabled={analyzer.indicatorB.sweepParam === "period"}
						/>
					</div>
				{/if}
				{#if analyzer.indicatorB.kind === "kama"}
					<div class="grid gap-2 sm:grid-cols-2">
						<div class="space-y-1">
							<Label class="text-xs">Fast {analyzer.indicatorB.sweepParam === "fast" ? "(swept)" : "(fixed)"}</Label>
							<Input
								type="number"
								min="1"
								step="1"
								bind:value={analyzer.indicatorB.kamaFast}
								disabled={analyzer.indicatorB.sweepParam === "fast"}
							/>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">Slow {analyzer.indicatorB.sweepParam === "slow" ? "(swept)" : "(fixed)"}</Label>
							<Input
								type="number"
								min="1"
								step="1"
								bind:value={analyzer.indicatorB.kamaSlow}
								disabled={analyzer.indicatorB.sweepParam === "slow"}
							/>
						</div>
					</div>
				{/if}
				{#if analyzer.indicatorB.kind === "alma"}
					<div class="grid gap-2 sm:grid-cols-2">
						<div class="space-y-1">
							<Label class="text-xs">
								Offset {analyzer.indicatorB.sweepParam === "offset" ? "(swept)" : "(fixed)"}
							</Label>
							<Input
								type="number"
								step="0.01"
								min="0"
								max="1"
								bind:value={analyzer.indicatorB.almaOffset}
								disabled={analyzer.indicatorB.sweepParam === "offset"}
							/>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">
								Sigma {analyzer.indicatorB.sweepParam === "sigma" ? "(swept)" : "(fixed)"}
							</Label>
							<Input
								type="number"
								step="0.1"
								min="0.000001"
								bind:value={analyzer.indicatorB.almaSigma}
								disabled={analyzer.indicatorB.sweepParam === "sigma"}
							/>
						</div>
					</div>
				{/if}
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

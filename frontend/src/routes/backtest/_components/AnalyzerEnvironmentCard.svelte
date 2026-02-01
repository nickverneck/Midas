<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { Gauge } from "lucide-svelte";
	import type { AnalyzerEnv } from "../types";

	type Props = {
		analyzerEnv: AnalyzerEnv;
		invalidAnalyzerInitialBalance: boolean;
		invalidAnalyzerMaxPosition: boolean;
		invalidAnalyzerCommission: boolean;
		invalidAnalyzerSlippage: boolean;
		invalidAnalyzerMargin: boolean;
		invalidAnalyzerContractMultiplier: boolean;
	};

	let {
		analyzerEnv,
		invalidAnalyzerInitialBalance,
		invalidAnalyzerMaxPosition,
		invalidAnalyzerCommission,
		invalidAnalyzerSlippage,
		invalidAnalyzerMargin,
		invalidAnalyzerContractMultiplier
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<Gauge size={18} />
			Analyzer Environment
		</Card.Title>
		<Card.Description>Balance, leverage, and costs used for the sweep.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		<div class="grid gap-4 sm:grid-cols-2">
			<div class="space-y-2">
				<Label>Initial Balance</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.initialBalance}
					aria-invalid={invalidAnalyzerInitialBalance}
				/>
			</div>
			<div class="space-y-2">
				<Label>Max Position</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.maxPosition}
					aria-invalid={invalidAnalyzerMaxPosition}
				/>
			</div>
			<div class="space-y-2">
				<Label>Commission (round trip)</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.commission}
					aria-invalid={invalidAnalyzerCommission}
				/>
			</div>
			<div class="space-y-2">
				<Label>Slippage / Contract</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.slippage}
					aria-invalid={invalidAnalyzerSlippage}
				/>
			</div>
			<div class="space-y-2">
				<Label>Margin / Contract</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.marginPerContract}
					aria-invalid={invalidAnalyzerMargin}
				/>
			</div>
			<div class="space-y-2">
				<Label>Contract Multiplier</Label>
				<Input
					type="number"
					bind:value={analyzerEnv.contractMultiplier}
					aria-invalid={invalidAnalyzerContractMultiplier}
				/>
			</div>
			<div class="space-y-2">
				<Label>Margin Mode</Label>
				<Select.Root bind:value={analyzerEnv.marginMode}>
					<Select.Trigger class="w-full">{analyzerEnv.marginMode}</Select.Trigger>
					<Select.Content>
						<Select.Item value="per-contract">per-contract</Select.Item>
						<Select.Item value="price">price</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
			<div class="flex items-center gap-2 pt-6">
				<input
					type="checkbox"
					class="h-4 w-4 rounded border-muted-foreground"
					bind:checked={analyzerEnv.enforceMargin}
				/>
				<Label>Enforce Margin</Label>
			</div>
		</div>
	</Card.Content>
</Card.Root>

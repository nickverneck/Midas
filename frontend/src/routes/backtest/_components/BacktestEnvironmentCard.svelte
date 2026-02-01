<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { Separator } from "$lib/components/ui/separator";
	import { Gauge } from "lucide-svelte";
	import type { BacktestEnv, BacktestLimits } from "../types";

	type Props = {
		env: BacktestEnv;
		limits: BacktestLimits;
		invalidInitialBalance: boolean;
		invalidMaxPosition: boolean;
		invalidCommission: boolean;
		invalidSlippage: boolean;
		invalidMargin: boolean;
		invalidContractMultiplier: boolean;
		invalidMemory: boolean;
		invalidInstructionLimit: boolean;
		invalidInstructionInterval: boolean;
	};

	let {
		env,
		limits,
		invalidInitialBalance,
		invalidMaxPosition,
		invalidCommission,
		invalidSlippage,
		invalidMargin,
		invalidContractMultiplier,
		invalidMemory,
		invalidInstructionLimit,
		invalidInstructionInterval
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<Gauge size={18} />
			Environment
		</Card.Title>
		<Card.Description>Match the training constraints for fair comparisons.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		<div class="grid gap-4 sm:grid-cols-2">
			<div class="space-y-2">
				<Label>Initial Balance</Label>
				<Input type="number" bind:value={env.initialBalance} aria-invalid={invalidInitialBalance} />
			</div>
			<div class="space-y-2">
				<Label>Max Position</Label>
				<Input type="number" bind:value={env.maxPosition} aria-invalid={invalidMaxPosition} />
			</div>
			<div class="space-y-2">
				<Label>Commission (round trip)</Label>
				<Input type="number" bind:value={env.commission} aria-invalid={invalidCommission} />
			</div>
			<div class="space-y-2">
				<Label>Slippage / Contract</Label>
				<Input type="number" bind:value={env.slippage} aria-invalid={invalidSlippage} />
			</div>
			<div class="space-y-2">
				<Label>Margin / Contract</Label>
				<Input type="number" bind:value={env.marginPerContract} aria-invalid={invalidMargin} />
			</div>
			<div class="space-y-2">
				<Label>Contract Multiplier</Label>
				<Input
					type="number"
					bind:value={env.contractMultiplier}
					aria-invalid={invalidContractMultiplier}
				/>
			</div>
			<div class="space-y-2">
				<Label>Margin Mode</Label>
				<Select.Root bind:value={env.marginMode}>
					<Select.Trigger class="w-full">{env.marginMode}</Select.Trigger>
					<Select.Content>
						<Select.Item value="per-contract">per-contract</Select.Item>
						<Select.Item value="price">price</Select.Item>
					</Select.Content>
				</Select.Root>
			</div>
		</div>
		<Separator />
		<div class="space-y-2">
			<Label>Safety Limits</Label>
			<div class="grid gap-3 sm:grid-cols-3">
				<Input
					type="number"
					bind:value={limits.memoryMb}
					placeholder="MB"
					aria-invalid={invalidMemory}
				/>
				<Input
					type="number"
					bind:value={limits.instructionLimit}
					placeholder="Instructions"
					aria-invalid={invalidInstructionLimit}
				/>
				<Input
					type="number"
					bind:value={limits.instructionInterval}
					placeholder="Check interval"
					aria-invalid={invalidInstructionInterval}
				/>
			</div>
			<p class="text-xs text-muted-foreground">Lua runtime caps for runaway scripts.</p>
		</div>
	</Card.Content>
</Card.Root>

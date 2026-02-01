<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { LineChart } from "lucide-svelte";
	import type { DatasetMode } from "../types";

	type Props = {
		title: string;
		description: string;
		datasetMode: DatasetMode;
		datasetPath: string;
		datasetPathInvalid: boolean;
		placeholder?: string;
		onBrowse?: () => void;
	};

	let {
		title,
		description,
		datasetMode = $bindable(),
		datasetPath = $bindable(),
		datasetPathInvalid,
		placeholder = "data/train/SPY0.parquet",
		onBrowse
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<LineChart size={18} />
			{title}
		</Card.Title>
		<Card.Description>{description}</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		<div class="space-y-2">
			<Label>Dataset</Label>
			<Select.Root bind:value={datasetMode}>
				<Select.Trigger class="w-full">{datasetMode}</Select.Trigger>
				<Select.Content>
					<Select.Item value="train">train</Select.Item>
					<Select.Item value="val">val</Select.Item>
					<Select.Item value="custom">custom</Select.Item>
				</Select.Content>
			</Select.Root>
		</div>
		<div class="space-y-2">
			<Label>Path</Label>
			<div class="flex items-end gap-2">
				<div class="flex-1">
					<Input
						bind:value={datasetPath}
						placeholder={placeholder}
						disabled={datasetMode !== "custom"}
						aria-invalid={datasetPathInvalid}
					/>
				</div>
				{#if onBrowse}
					<Button type="button" variant="outline" size="sm" onclick={onBrowse}>
						Browse
					</Button>
				{/if}
			</div>
			{#if datasetMode === "custom"}
				<p class="text-xs text-muted-foreground">Provide a .parquet file path.</p>
			{/if}
		</div>
	</Card.Content>
</Card.Root>

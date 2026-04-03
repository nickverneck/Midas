<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";

	type Props = {
		loading: boolean;
		output: string | null;
		error: string | null;
		env: Record<string, string | null> | null;
	};

	let { loading, output, error, env }: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-center justify-between">
		<Card.Title>Diagnostics Output</Card.Title>
		{#if loading}
			<Badge variant="outline" class="animate-pulse">Running</Badge>
		{/if}
	</Card.Header>
	<Card.Content>
		<ScrollArea class="h-[220px] w-full rounded-md border border-zinc-800 bg-zinc-950 p-4 font-mono text-sm">
			{#if env}
				<div class="mb-3 whitespace-pre-wrap text-zinc-400">
					{Object.entries(env)
						.map(([key, value]) => `${key}=${value ?? ""}`)
						.join("\n")}
				</div>
			{/if}
			{#if error}
				<div class="whitespace-pre-wrap text-red-400">{error}</div>
			{/if}
			{#if output}
				<div class="whitespace-pre-wrap text-zinc-300">{output}</div>
			{:else}
				<div class="italic text-zinc-600">
					Run diagnostics to see libtorch and MLX viability info. Candle GA and RL are already runnable from the main form.
				</div>
			{/if}
		</ScrollArea>
	</Card.Content>
</Card.Root>

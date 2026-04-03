<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import type { ConsoleLine } from "../types";

	type Props = {
		training: boolean;
		consoleOutput: ConsoleLine[];
	};

	let { training, consoleOutput }: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-center justify-between">
		<Card.Title>Console Output</Card.Title>
		{#if training}
			<Badge variant="outline" class="animate-pulse">Active</Badge>
		{/if}
	</Card.Header>
	<Card.Content>
		<ScrollArea class="h-[260px] w-full rounded-md border border-zinc-800 bg-zinc-950 p-4 font-mono text-sm">
			{#each consoleOutput as line}
				<div
					class={line.type === "stderr" || line.type === "error"
						? "text-red-400"
						: line.type === "system"
							? "text-blue-400"
							: "text-zinc-300"}
				>
					<span class="mr-2 opacity-50">[{new Date().toLocaleTimeString()}]</span>
					{line.text}
				</div>
			{/each}
			{#if consoleOutput.length === 0}
				<div class="italic text-zinc-600">No output yet. Start training to see logs.</div>
			{/if}
		</ScrollArea>
	</Card.Content>
</Card.Root>

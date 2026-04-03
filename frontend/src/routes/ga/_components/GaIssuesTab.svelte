<script lang="ts">
	import { Bug, Info } from "lucide-svelte";
	import * as Card from "$lib/components/ui/card";
	import type { IssueItem } from "../types";

	type Props = {
		issues: IssueItem[];
	};

	let { issues }: Props = $props();
</script>

{#if issues.length === 0}
	<div class="flex flex-col items-center justify-center py-20 text-muted-foreground">
		<Info size={48} class="mb-4 opacity-20" />
		<p class="text-xl font-medium">No major issues detected</p>
		<p class="text-sm">Training data looks within expected heuristic bounds.</p>
	</div>
{:else}
	<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
		{#each issues as issue}
			<Card.Root
				class={issue.type === "destructive"
					? "border-red-500"
					: issue.type === "warning"
						? "border-amber-500"
						: "border-blue-500"}
			>
				<Card.Header class="flex flex-row items-center gap-4 py-4">
					<div
						class={`rounded-full p-2 ${issue.type === "destructive"
							? "bg-red-100 text-red-600"
							: issue.type === "warning"
								? "bg-amber-100 text-amber-600"
								: "bg-blue-100 text-blue-600"}`}
					>
						<Bug size={24} />
					</div>
					<div>
						<Card.Title class="text-lg">{issue.title}</Card.Title>
					</div>
				</Card.Header>
				<Card.Content class="pb-4">
					<p class="text-sm text-muted-foreground">{issue.message}</p>
					{#if issue.items}
						<div class="mt-3 space-y-1 rounded-md bg-muted p-3 font-mono text-xs">
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

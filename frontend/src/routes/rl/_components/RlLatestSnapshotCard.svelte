<script lang="ts">
	import { Badge } from "$lib/components/ui/badge";
	import * as Card from "$lib/components/ui/card";
	import type { RlPoint, RlSnapshotRow } from "../types";

	type Props = {
		latest: RlPoint | null;
		snapshotRows: RlSnapshotRow[];
		loading: boolean;
	};

	let { latest, snapshotRows, loading }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Latest Snapshot</Card.Title>
		<Card.Description>Most recent epoch summary.</Card.Description>
	</Card.Header>
	<Card.Content>
		{#if latest}
			<div class="space-y-3 text-sm">
				{#each snapshotRows as row}
					<div class="flex items-center justify-between gap-4">
						<span class="text-muted-foreground">{row.label}</span>
						<span class={row.emphasis ? "font-semibold uppercase" : ""}>{row.value ?? "—"}</span>
					</div>
				{/each}
			</div>
		{:else}
			<div class="text-sm text-muted-foreground">No RL logs yet.</div>
		{/if}
	</Card.Content>
	{#if loading}
		<Card.Footer>
			<Badge variant="outline" class="animate-pulse">Loading</Badge>
		</Card.Footer>
	{/if}
</Card.Root>

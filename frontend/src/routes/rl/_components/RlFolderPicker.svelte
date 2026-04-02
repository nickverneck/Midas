<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import type { FolderEntry } from "../types";

	type Props = {
		open: boolean;
		loading: boolean;
		error: string;
		entries: FolderEntry[];
		onClose: () => void;
		onSelect: (folder: FolderEntry) => void;
	};

	let { open, loading, error, entries, onClose, onSelect }: Props = $props();
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center p-4"
		role="dialog"
		aria-modal="true"
		aria-label="Select RL run folder"
	>
		<button
			type="button"
			class="absolute inset-0 bg-black/40"
			onclick={onClose}
			aria-label="Close folder picker"
		></button>
		<div class="relative z-10 w-full max-w-2xl rounded-xl border bg-background p-4 shadow-lg">
			<div class="flex items-start justify-between gap-4">
				<div>
					<div class="text-xs uppercase tracking-wide text-muted-foreground">Select RL Run Folder</div>
					<div class="text-sm font-semibold">runs_rl</div>
					<div class="text-xs text-muted-foreground">Sorted by modification time (newest first)</div>
				</div>
				<Button type="button" variant="ghost" size="sm" onclick={onClose}>Close</Button>
			</div>

			{#if error}
				<div class="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
					{error}
				</div>
			{/if}

			<ScrollArea class="mt-3 max-h-[360px] rounded-lg border">
				{#if loading}
					<div class="px-4 py-6 text-sm text-muted-foreground">Loading folders...</div>
				{:else if entries.length === 0}
					<div class="px-4 py-6 text-sm text-muted-foreground">No run folders found in runs_rl.</div>
				{:else}
					<div class="divide-y">
						{#each entries as folder}
							<button
								type="button"
								class="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-muted/50"
								onclick={() => onSelect(folder)}
							>
								<span class="text-[11px] font-semibold uppercase text-muted-foreground">
									Folder
								</span>
								<span class="text-sm font-medium">{folder.name}</span>
								{#if folder.mtime}
									<span class="ml-auto text-xs text-muted-foreground">
										{new Date(folder.mtime * 1000).toLocaleString()}
									</span>
								{/if}
							</button>
						{/each}
					</div>
				{/if}
			</ScrollArea>
		</div>
	</div>
{/if}

<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import type { FileEntry } from "../types";

	type Props = {
		open: boolean;
		title: string;
		dir: string;
		extensions: string[];
		parent: string | null;
		entries: FileEntry[];
		loading: boolean;
		error: string;
		emptyMessage: string;
		onClose: () => void;
		onUp: () => void;
		onSelect: (entry: FileEntry) => void;
	};

	let {
		open,
		title,
		dir,
		extensions,
		parent,
		entries,
		loading,
		error,
		emptyMessage,
		onClose,
		onUp,
		onSelect
	}: Props = $props();
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center p-4"
		role="dialog"
		aria-modal="true"
		aria-label={title}
	>
		<button
			type="button"
			class="absolute inset-0 bg-black/40"
			onclick={onClose}
			aria-label="Close file picker"
		></button>
		<div class="relative z-10 w-full max-w-2xl rounded-xl border bg-background p-4 shadow-lg">
			<div class="flex items-start justify-between gap-4">
				<div>
					<div class="text-xs uppercase tracking-wide text-muted-foreground">{title}</div>
					<div class="text-sm font-semibold">{dir || "Project root"}</div>
					<div class="text-xs text-muted-foreground">
						Allowed: {extensions.map((ext) => `.${ext}`).join(", ")}
					</div>
				</div>
				<Button type="button" variant="ghost" size="sm" onclick={onClose}>Close</Button>
			</div>
			<div class="mt-3 flex items-center gap-2">
				<Button type="button" variant="outline" size="sm" onclick={onUp} disabled={parent === null}>
					Up
				</Button>
				<div class="truncate text-xs text-muted-foreground">{dir ? `/${dir}` : "/"}</div>
			</div>

			{#if error}
				<div class="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
					{error}
				</div>
			{/if}

			<div class="mt-3 max-h-[360px] overflow-auto rounded-lg border">
				{#if loading}
					<div class="px-4 py-6 text-sm text-muted-foreground">Loading files...</div>
				{:else if entries.length === 0}
					<div class="px-4 py-6 text-sm text-muted-foreground">{emptyMessage}</div>
				{:else}
					<div class="divide-y">
						{#each entries as entry}
							<button
								type="button"
								class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-muted/50"
								onclick={() => onSelect(entry)}
							>
								<span class="text-[11px] font-semibold uppercase text-muted-foreground">
									{entry.kind === "dir" ? "Dir" : "File"}
								</span>
								<span class="text-sm">{entry.name}</span>
							</button>
						{/each}
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

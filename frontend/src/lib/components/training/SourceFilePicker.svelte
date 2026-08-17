<script lang="ts">
	import { FolderOpen, X } from 'lucide-svelte';

	type Entry = { name: string; path: string; kind: 'dir' | 'file' };
	type Props = {
		value: string;
		onChange: (value: string) => void;
		id?: string;
		disabled?: boolean;
		initialDir?: string;
		extensions?: string[];
		label?: string;
		buttonLabel?: string;
		placeholder?: string;
		emptyText?: string;
	};

	let {
		value,
		onChange,
		id,
		disabled = false,
		initialDir = 'data',
		extensions = ['parquet', 'csv', 'txt'],
		label = 'Browse files',
		buttonLabel = label,
		placeholder = 'data/bars.parquet, .csv, or Ninja .Last.txt',
		emptyText = 'No matching files here.'
	}: Props = $props();
	let open = $state(false);
	let dir = $state('');
	let parent = $state<string | null>(null);
	let entries = $state<Entry[]>([]);
	let loading = $state(false);
	let error = $state('');

	const load = async (nextDir: string) => {
		loading = true;
		error = '';
		try {
			const params = new URLSearchParams({ ext: extensions.join(',') });
			if (nextDir) params.set('dir', nextDir);
			const response = await fetch(`/api/files?${params}`);
			const body = await response.json();
			if (!response.ok) throw new Error(body.error || 'Unable to browse files');
			dir = body.dir ?? '';
			parent = typeof body.parent === 'string' ? body.parent : null;
			entries = Array.isArray(body.entries) ? body.entries : [];
		} catch (reason) {
			if (nextDir !== '') {
				await load('');
				return;
			}
			error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			loading = false;
		}
	};

	const show = () => {
		if (disabled) return;
		open = true;
		void load(initialDir);
	};

	const choose = (entry: Entry) => {
		if (disabled) return;
		if (entry.kind === 'dir') {
			void load(entry.path);
			return;
		}
		onChange(entry.path);
		open = false;
	};
</script>

	<div class="min-w-0">
		<div class="flex min-w-0 gap-2">
	<input
		{id}
			value={value}
		oninput={(event) => onChange((event.currentTarget as HTMLInputElement).value)}
		disabled={disabled}
		placeholder={placeholder}
		title={value || placeholder}
		aria-label={value ? `${label}: ${value}` : label}
		class="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-xs disabled:opacity-60 sm:h-9 sm:px-3 sm:text-sm"
	/>
	<button type="button" onclick={show} disabled={disabled} class="inline-flex min-h-11 shrink-0 items-center gap-1 rounded-md border bg-background px-3 text-[11px] font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50 sm:text-xs">
		<FolderOpen size={13} /> <span class="hidden sm:inline">{buttonLabel}</span><span class="sm:hidden">Browse</span>
	</button>
		</div>
		{#if value}
			<details class="mt-1 sm:hidden">
				<summary class="cursor-pointer text-[10px] font-medium text-muted-foreground underline underline-offset-2">View full path</summary>
				<code class="mt-1 block max-h-16 overflow-auto break-all rounded bg-muted/60 px-1.5 py-1 text-[10px] leading-4">{value}</code>
			</details>
		{/if}
	</div>

{#if open}
	<div class="fixed inset-0 z-50 grid place-items-center p-3 sm:p-4" role="dialog" aria-modal="true" aria-label={label}>
		<button type="button" class="absolute inset-0 bg-black/45" onclick={() => (open = false)} disabled={disabled} aria-label="Close file browser"></button>
		<div class="relative z-10 w-full max-w-xl overflow-hidden rounded-xl border bg-background shadow-xl">
			<header class="flex items-center justify-between border-b p-3">
				<div class="min-w-0"><b class="block text-sm">{label}</b><span class="block truncate text-[11px] text-muted-foreground">/{dir} · {extensions.map((extension) => `.${extension}`).join(', ')}</span></div>
				<button type="button" onclick={() => (open = false)} disabled={disabled} class="grid size-11 place-items-center rounded hover:bg-muted disabled:opacity-50" aria-label="Close"><X size={16} /></button>
			</header>
			<div class="flex items-center gap-2 border-b px-3 py-2">
				<button type="button" onclick={() => parent !== null && void load(parent)} disabled={disabled || parent === null || loading} class="min-h-11 rounded border px-3 py-2 text-xs disabled:opacity-40">Up</button>
				{#if error}<span class="truncate text-xs text-destructive">{error}</span>{/if}
			</div>
			<div class="max-h-[60vh] overflow-auto p-2">
				{#if loading}<p class="p-4 text-sm text-muted-foreground">Loading…</p>
				{:else if entries.length === 0}<p class="p-4 text-sm text-muted-foreground">{emptyText}</p>
				{:else}
					{#each entries as entry}
						<button type="button" onclick={() => choose(entry)} disabled={disabled} class="flex min-h-11 w-full items-center gap-3 rounded px-3 py-2 text-left hover:bg-muted disabled:opacity-50">
							<span class="w-8 shrink-0 text-[10px] font-semibold uppercase text-muted-foreground">{entry.kind === 'dir' ? 'Dir' : 'File'}</span><span class="min-w-0 break-all text-sm leading-5">{entry.name}</span>
						</button>
					{/each}
				{/if}
			</div>
		</div>
	</div>
{/if}

<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { ChevronLeft, ChevronRight, FileCode, Gauge } from "lucide-svelte";
	import type { BacktestView } from "../types";

	type NavItem = {
		id: BacktestView;
		label: string;
		icon: typeof FileCode;
	};

	type Props = {
		activeView: BacktestView;
		collapsed: boolean;
	};

	let { activeView = $bindable(), collapsed = $bindable() }: Props = $props();

	const navItems: NavItem[] = [
		{ id: "script", label: "Script Runner", icon: FileCode },
		{ id: "analyzer", label: "Strategy Analyzer", icon: Gauge }
	];
</script>

<aside
	class={`bg-card border-r shadow-sm transition-[width] duration-200 ease-out w-full lg:fixed lg:inset-y-0 lg:left-0 lg:pt-14 relative ${
		collapsed ? "lg:w-[120px]" : "lg:w-[320px]"
	}`}
>
	<div class="relative z-10 flex h-full flex-col">
		<div class={`flex items-center justify-between border-b px-4 py-4 ${collapsed ? "lg:px-3" : ""}`}>
			<div class={`flex items-center gap-2 ${collapsed ? "justify-center w-full" : ""}`}>
				<div class="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-sm font-semibold text-primary">
					BT
				</div>
				{#if !collapsed}
					<div>
						<div class="text-sm font-semibold text-foreground">Backtest</div>
						<div class="text-xs text-muted-foreground">Dashboards</div>
					</div>
				{/if}
			</div>
			{#if !collapsed}
				<Button
					variant="ghost"
					size="icon"
					class="h-8 w-8"
					onclick={() => (collapsed = true)}
					aria-label="Collapse menu"
				>
					<ChevronLeft size={16} />
				</Button>
			{/if}
		</div>

		{#if collapsed}
			<div class="px-3 py-3">
				<Button
					variant="ghost"
					size="icon"
					class="h-8 w-8"
					onclick={() => (collapsed = false)}
					aria-label="Expand menu"
				>
					<ChevronRight size={16} />
				</Button>
			</div>
		{/if}

		<nav class={`flex flex-col gap-1 px-3 py-2 ${collapsed ? "items-center" : ""}`}>
			{#each navItems as item}
				<button
					type="button"
					class={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition ${
						activeView === item.id
							? "bg-muted text-foreground"
							: "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
					}`}
					onclick={() => (activeView = item.id)}
					aria-label={item.label}
					aria-current={activeView === item.id ? "page" : undefined}
					title={collapsed ? item.label : undefined}
				>
					<item.icon size={18} />
					{#if !collapsed}
						<span>{item.label}</span>
					{/if}
				</button>
			{/each}
		</nav>
	</div>
</aside>

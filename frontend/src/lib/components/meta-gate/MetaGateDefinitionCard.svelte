<script lang="ts">
	type Action = "normal" | "skip" | "invert";

	type Props = {
		triggerFast?: number;
		triggerSlow?: number;
		contextFast?: number;
		contextSlow?: number;
		enabledActions?: Action[];
	};

	let {
		triggerFast = $bindable(10),
		triggerSlow = $bindable(30),
		contextFast = $bindable(210),
		contextSlow = $bindable(240),
		enabledActions = $bindable<Action[]>(["normal", "skip"])
	}: Props = $props();

	const actionDefinitions: Array<{
		id: Action;
		label: string;
		description: string;
		accent: string;
	}> = [
		{
			id: "normal",
			label: "Normal",
			description: "Follow the direction of the raw trigger cross.",
			accent: "border-emerald-500/40 bg-emerald-500/5"
		},
		{
			id: "skip",
			label: "Skip",
			description:
				"Fixed-horizon diagnostic only: assign no new trade PnL to this event; it does not model holding a live position.",
			accent: "border-slate-400/50 bg-slate-500/5"
		},
		{
			id: "invert",
			label: "Invert",
			description: "Target the direction opposite to the raw trigger cross.",
			accent: "border-violet-500/40 bg-violet-500/5"
		}
	];
	const requiredActions: Action[] = ["normal", "skip"];

	function isRequiredAction(action: Action) {
		return requiredActions.includes(action);
	}

	if (requiredActions.some((action) => !enabledActions.includes(action))) {
		enabledActions = [
			...requiredActions,
			...enabledActions.filter((action) => !requiredActions.includes(action))
		];
	}

	const causalFeatures = [
		"raw_direction",
		"trigger_spread_atr",
		"context_spread_atr",
		"trigger_fast_slope_atr",
		"trigger_slow_slope_atr",
		"context_fast_slope_atr",
		"context_slow_slope_atr",
		"efficiency_ratio",
		"rvol_20",
		"return_1",
		"return_5"
	];

	const plannedFeatures = ["position_state", "session_state"];

	const futureLabels = [
		"normal_return_horizon",
		"invert_return_horizon",
		"best_action_horizon",
		"forward_exit_timestamp"
	];

	function setActionEnabled(action: Action, checked: boolean) {
		if (isRequiredAction(action) && !checked) {
			return;
		}

		if (checked) {
			if (!enabledActions.includes(action)) {
				enabledActions = [...enabledActions, action];
			}
			return;
		}

		enabledActions = enabledActions.filter((enabledAction) => enabledAction !== action);
	}
</script>

<section class="space-y-5 rounded-xl border bg-card/50 p-5 shadow-sm" aria-labelledby="meta-gate-title">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div class="space-y-1">
			<div class="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
				Problem definition
			</div>
			<h2 id="meta-gate-title" class="text-base font-semibold tracking-tight">
				Cross-conditioned meta gate
			</h2>
			<p class="max-w-2xl text-sm leading-relaxed text-muted-foreground">
				The configured EMA {triggerFast}/{triggerSlow} cross proposes a raw direction; the gate emits a
				fixed-horizon diagnostic action: normal, skip, or invert.
			</p>
		</div>
		<span class="rounded-full border border-primary/30 bg-primary/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
			Event-level prototype
		</span>
	</div>

	<div class="grid gap-3 md:grid-cols-2">
		<fieldset class="space-y-3 rounded-lg border p-4">
			<legend class="px-1 text-sm font-semibold">Trigger EMA pair</legend>
			<p class="text-xs leading-relaxed text-muted-foreground">
				The raw cross that creates a decision event.
			</p>
			<div class="grid grid-cols-2 gap-3">
				<label class="space-y-1.5 text-xs font-medium">
					<span class="text-muted-foreground">Fast period</span>
					<input
						class="h-9 w-full rounded-md border bg-background px-3 text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
						type="number"
						min="1"
						step="1"
						aria-label="Trigger fast EMA period"
						bind:value={triggerFast}
					/>
				</label>
				<label class="space-y-1.5 text-xs font-medium">
					<span class="text-muted-foreground">Slow period</span>
					<input
						class="h-9 w-full rounded-md border bg-background px-3 text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
						type="number"
						min="2"
						step="1"
						aria-label="Trigger slow EMA period"
						bind:value={triggerSlow}
					/>
				</label>
			</div>
			<div class="rounded-md bg-muted/50 px-3 py-2 font-mono text-xs text-muted-foreground">
				EMA {triggerFast}/{triggerSlow} → raw buy or sell direction
			</div>
		</fieldset>

		<fieldset class="space-y-3 rounded-lg border p-4">
			<legend class="px-1 text-sm font-semibold">Context EMA pair</legend>
			<p class="text-xs leading-relaxed text-muted-foreground">
				The slower regime context supplied to the gate at the event.
			</p>
			<div class="grid grid-cols-2 gap-3">
				<label class="space-y-1.5 text-xs font-medium">
					<span class="text-muted-foreground">Fast period</span>
					<input
						class="h-9 w-full rounded-md border bg-background px-3 text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
						type="number"
						min="2"
						step="1"
						aria-label="Context fast EMA period"
						bind:value={contextFast}
					/>
				</label>
				<label class="space-y-1.5 text-xs font-medium">
					<span class="text-muted-foreground">Slow period</span>
					<input
						class="h-9 w-full rounded-md border bg-background px-3 text-sm font-semibold outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
						type="number"
						min="3"
						step="1"
						aria-label="Context slow EMA period"
						bind:value={contextSlow}
					/>
				</label>
			</div>
			<div class="rounded-md bg-muted/50 px-3 py-2 font-mono text-xs text-muted-foreground">
				EMA {contextFast}/{contextSlow} → regime context, never future data
			</div>
		</fieldset>
	</div>

	<fieldset
		class="space-y-3"
		aria-labelledby="allowed-gate-actions-title"
		aria-describedby="allowed-gate-actions-help"
	>
		<legend id="allowed-gate-actions-title" class="text-sm font-semibold">Allowed gate actions</legend>
		<div class="flex flex-wrap items-end justify-between gap-2">
			<div>
				<p class="mt-1 text-xs text-muted-foreground">
					Choose which decisions the model may emit. Normal and skip are enabled by default.
				</p>
				<p id="allowed-gate-actions-help" class="mt-1 text-xs text-muted-foreground">
					Normal and skip are required by the current API and trainer; invert is optional.
				</p>
			</div>
			<span class="text-xs tabular-nums text-muted-foreground">
				{enabledActions.length} of {actionDefinitions.length} enabled
			</span>
		</div>

		<div class="grid gap-2 lg:grid-cols-3">
			{#each actionDefinitions as action}
				<label
					class={`flex cursor-pointer gap-3 rounded-lg border p-3 transition hover:border-foreground/30 ${action.accent} ${enabledActions.includes(action.id) ? "ring-1 ring-ring/40" : "opacity-70"}`}
				>
					<input
						class="mt-0.5 size-4 shrink-0 accent-primary"
						type="checkbox"
						checked={isRequiredAction(action.id) || enabledActions.includes(action.id)}
						disabled={isRequiredAction(action.id)}
						aria-label={`Enable ${action.label} action`}
						onchange={(event) =>
							setActionEnabled(action.id, (event.currentTarget as HTMLInputElement).checked)}
					/>
					<span class="min-w-0 space-y-1">
						<span class="block text-sm font-semibold">{action.label}</span>
						<span class="block text-xs leading-relaxed text-muted-foreground">
							{action.description}
						</span>
					</span>
				</label>
			{/each}
		</div>
	</fieldset>

	<div class="grid gap-3 rounded-lg border border-dashed p-4 md:grid-cols-4">
		<div class="space-y-1">
			<div class="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Lifecycle</div>
			<p class="text-xs leading-relaxed text-muted-foreground">
				This slice evaluates action labels over a fixed horizon; it does not simulate live orders or
				position state.
			</p>
		</div>
		<div class="space-y-1">
			<div class="text-xs font-semibold text-emerald-700 dark:text-emerald-300">Normal</div>
			<p class="text-xs leading-relaxed text-muted-foreground">
				Evaluates the raw cross direction over the configured horizon. Fills, protection, and reversals
				are not simulated here.
			</p>
		</div>
		<div class="space-y-1">
			<div class="text-xs font-semibold text-slate-700 dark:text-slate-300">Skip</div>
			<p class="text-xs leading-relaxed text-muted-foreground">
				Assigns no new trade PnL to this event. It is not a live-position hold, flatten, or reverse
				decision because current position state is not modeled.
			</p>
		</div>
		<div class="space-y-1">
			<div class="text-xs font-semibold text-violet-700 dark:text-violet-300">Invert</div>
			<p class="text-xs leading-relaxed text-muted-foreground">
				Evaluates the direction opposite the raw cross over the configured horizon. Fills and position
				transitions are not simulated here.
			</p>
		</div>
	</div>

	<div class="space-y-3 rounded-lg border p-4">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<div>
				<div class="text-sm font-semibold">Event row contract</div>
				<p class="mt-1 text-xs text-muted-foreground">
					The model sees only values available when the cross closes.
				</p>
			</div>
			<div class="flex flex-wrap gap-2 text-[11px] font-semibold uppercase tracking-wide">
				<span class="rounded-full bg-sky-500/10 px-2 py-1 text-sky-700 dark:text-sky-300">
					Causal at event
				</span>
				<span class="rounded-full bg-amber-500/10 px-2 py-1 text-amber-700 dark:text-amber-300">
					Future outcome = label
				</span>
			</div>
		</div>

		<div class="grid gap-3 md:grid-cols-2">
			<div class="rounded-md bg-sky-500/5 p-3 ring-1 ring-sky-500/15">
				<div class="mb-2 text-xs font-semibold uppercase tracking-wide text-sky-700 dark:text-sky-300">
					Feature values · causal at event
				</div>
				<ul class="space-y-1.5 font-mono text-[11px] text-muted-foreground">
					{#each causalFeatures as feature}
						<li class="flex gap-2"><span class="text-sky-600 dark:text-sky-400">•</span>{feature}</li>
					{/each}
				</ul>
			</div>
			<div class="rounded-md bg-amber-500/5 p-3 ring-1 ring-amber-500/15">
				<div class="mb-2 text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
					Future outcome columns · labels
				</div>
				<ul class="space-y-1.5 font-mono text-[11px] text-muted-foreground">
					{#each futureLabels as label}
						<li class="flex gap-2"><span class="text-amber-600 dark:text-amber-400">•</span>{label}</li>
					{/each}
				</ul>
			</div>
		</div>
		<div class="rounded-md border border-dashed border-muted-foreground/30 bg-muted/20 p-3 text-xs text-muted-foreground">
			<span class="font-semibold text-foreground">Planned features:</span>
			{plannedFeatures.join(" and ")} are not part of the current 11-feature trainer vector yet;
			they require a sequential position/session-state implementation.
		</div>
	</div>

	<div class="rounded-lg border border-dashed bg-muted/20 p-4 text-xs leading-relaxed text-muted-foreground" role="note">
		<span class="font-semibold text-foreground">Unsupported in this slice:</span>
		HMA/ADX selection, selectable feature subsets, and date, session, or contract-transition split
		controls. The current prototype uses EMA/EMA, the fixed 11-feature vector above, and chronological
		fraction splits.
	</div>

	<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm" role="note">
		<div class="flex gap-3">
			<div class="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-amber-500/20 text-xs font-bold text-amber-700 dark:text-amber-300" aria-hidden="true">
				!
			</div>
			<div class="space-y-1">
				<div class="font-semibold text-amber-900 dark:text-amber-100">Prototype validation boundary</div>
				<p class="text-xs leading-relaxed text-amber-900/80 dark:text-amber-100/80">
					The current prototype uses fixed-horizon diagnostics for future outcomes. These are labels,
					not Trader replay fills; validate the selected gate with Trader replay before treating a result
					as executable.
				</p>
			</div>
		</div>
	</div>
</section>

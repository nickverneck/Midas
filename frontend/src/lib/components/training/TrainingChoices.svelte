<script lang="ts">
	import { BrainCircuit, Cpu, Flame, Network, RefreshCcw, Rocket, Sparkles, Workflow } from 'lucide-svelte';
	import type { BackendId, DeviceId, StartMode, TrainingAlgorithm, TrainingCapabilities } from './types';

	type Props = {
		startMode: StartMode;
		algorithm: TrainingAlgorithm;
		backend: BackendId;
		device: DeviceId;
		capabilities: TrainingCapabilities;
		showAlgorithmChoices?: boolean;
		onStartMode: (value: StartMode) => void;
		onAlgorithm: (value: TrainingAlgorithm) => void;
		onBackend: (value: BackendId) => void;
		onDevice: (value: DeviceId) => void;
	};

	let { startMode, algorithm, backend, device, capabilities, showAlgorithmChoices = true, onStartMode, onAlgorithm, onBackend, onDevice }: Props = $props();

	const algorithms = [
		{ id: 'supervised' as const, label: 'Supervised', note: 'Event classifier', icon: BrainCircuit },
		{ id: 'ga' as const, label: 'GA', note: 'Legacy optimizer', icon: Network },
		{ id: 'rl' as const, label: 'RL', note: 'Legacy policy', icon: Workflow }
	];
	const backends = [
		{ id: 'torch' as const, label: 'Libtorch', icon: Flame },
		{ id: 'burn' as const, label: 'Burn', icon: Sparkles },
		{ id: 'candle' as const, label: 'Candle', icon: Rocket },
		{ id: 'cpu-linear' as const, label: 'CPU reference', icon: Cpu }
	];
	const deviceItems = [
		{ id: 'auto' as const, label: 'Auto', note: 'Use backend default' },
		{ id: 'cpu' as const, label: 'CPU', note: 'Always available' },
		{ id: 'cuda' as const, label: 'CUDA', note: 'Compiled GPU path' }
	];

	const capabilityFor = (id: BackendId) => capabilities.backends[id];
	const backendSupported = (id: BackendId) => capabilityFor(id)[algorithm];
	const deviceSupported = (id: DeviceId) => backendSupported(backend) && capabilityFor(backend).devices[id];
	const visibleDevices = $derived(capabilityFor(backend).devices.mps ? [...deviceItems, { id: 'mps' as const, label: 'MPS', note: 'Apple GPU / Burn MLX' }] : deviceItems);
</script>

<section class="grid gap-3 lg:grid-cols-[minmax(12rem,0.75fr)_minmax(18rem,1fr)_minmax(22rem,1.65fr)]" aria-label="Training setup">
	<div class="grid min-w-0 grid-cols-2 gap-2 rounded-xl border bg-card p-2 shadow-xs">
		<button
			type="button"
			onclick={() => onStartMode('new')}
			class:bg-foreground={startMode === 'new'}
			class:text-background={startMode === 'new'}
			class="flex min-h-14 min-w-0 items-center gap-2 rounded-lg px-2.5 text-left transition-colors hover:bg-muted sm:px-3"
		>
			<Rocket size={17} class="shrink-0" /><span class="min-w-0"><b class="block whitespace-nowrap text-sm">New</b><small class="block whitespace-nowrap text-[11px] opacity-70">Fresh policy</small></span>
		</button>
		<button
			type="button"
			onclick={() => onStartMode('continue')}
			class:bg-foreground={startMode === 'continue'}
			class:text-background={startMode === 'continue'}
			class="flex min-h-14 min-w-0 items-center gap-2 rounded-lg px-2.5 text-left transition-colors hover:bg-muted sm:px-3"
		>
			<RefreshCcw size={17} class="shrink-0" /><span class="min-w-0"><b class="block whitespace-nowrap text-sm">Continue</b><small class="block whitespace-nowrap text-[11px] opacity-70">Resume policy</small></span>
		</button>
	</div>

		{#if showAlgorithmChoices}
		<div class="grid grid-cols-3 gap-2 rounded-xl border bg-card p-2 shadow-xs">
		{#each algorithms as item}
			<button
				type="button"
				onclick={() => onAlgorithm(item.id)}
				aria-pressed={algorithm === item.id}
				class:border-foreground={algorithm === item.id}
				class:bg-muted={algorithm === item.id}
				class="min-w-0 rounded-lg border border-transparent px-1.5 py-2 text-center hover:bg-muted"
			>
				<item.icon class="mx-auto mb-1" size={17} />
				<b class="block whitespace-nowrap text-xs sm:text-sm lg:text-xs 2xl:text-sm">{item.label}</b>
				<small class="hidden truncate text-[10px] text-muted-foreground sm:block">{item.note}</small>
			</button>
			{/each}
		</div>
		{/if}

	<div class="rounded-xl border bg-card p-2 shadow-xs">
		<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
		{#each backends as item}
			{@const capability = capabilityFor(item.id)}
			{@const enabled = backendSupported(item.id)}
			<button
				type="button"
				onclick={() => enabled && onBackend(item.id)}
				disabled={!enabled}
				aria-pressed={backend === item.id}
				class:border-emerald-600={backend === item.id && enabled}
				class:bg-emerald-50={backend === item.id && enabled}
				class="grid min-w-0 grid-cols-[auto_1fr] items-center gap-2 rounded-lg border border-transparent px-2 py-2 text-left disabled:cursor-not-allowed disabled:opacity-55 sm:grid-cols-1 sm:gap-1 sm:text-center"
				title={enabled ? capability.note : `${item.label} is not enabled for ${algorithm === 'supervised' ? 'supervised' : algorithm.toUpperCase()} training`}
			>
				<item.icon size={17} class="mx-auto shrink-0" />
				<span class="min-w-0"><b class="block whitespace-nowrap text-[11px]">{item.label}</b><small class="block truncate text-[10px] text-muted-foreground">{enabled ? (item.id === 'cpu-linear' ? 'Supervised only' : 'Available') : 'Unavailable here'}</small></span>
			</button>
		{/each}
		</div>
		<div class="mt-2 flex flex-wrap items-center gap-1.5 border-t pt-2">
			<span class="mr-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Runtime</span>
			{#each visibleDevices as item}
				<button
					type="button"
					onclick={() => deviceSupported(item.id) && onDevice(item.id)}
					disabled={!deviceSupported(item.id)}
					aria-pressed={device === item.id}
					class:border-foreground={device === item.id && deviceSupported(item.id)}
					class:bg-muted={device === item.id && deviceSupported(item.id)}
					class="min-h-8 rounded-md border border-transparent px-2.5 text-[10px] font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-45"
					title={deviceSupported(item.id) ? item.note : `${item.label} is unavailable for ${backend}`}
				>{item.label}</button>
			{/each}
			<span class="ml-auto text-[10px] text-muted-foreground">{capabilityFor(backend).note}</span>
		</div>
	</div>
</section>

<p class="mt-1 px-1 text-[10px] leading-4 text-muted-foreground">
	{#if algorithm === 'supervised'}
		Supervised training is available on the CPU reference, Burn, and Candle. Accelerator choices appear only when the server build exposes them.
	{:else}
		{algorithm.toUpperCase()} uses the existing Rust runner. CUDA is enabled only when the corresponding backend feature is configured on the server.
	{/if}
</p>

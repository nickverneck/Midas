<script lang="ts">
	import { onMount } from 'svelte';
	import { Activity, ShieldCheck } from 'lucide-svelte';
	import { fallbackTrainingCapabilities } from '$lib/components/training/types';
	import ArtifactPanel from '$lib/components/training/ArtifactPanel.svelte';
	import DefinitionPanel from '$lib/components/training/DefinitionPanel.svelte';
	import SupervisedDiagnostics from '$lib/components/training/SupervisedDiagnostics.svelte';
	import StepIndicator from '$lib/components/training/StepIndicator.svelte';
	import TrainingChoices from '$lib/components/training/TrainingChoices.svelte';
	import type {
		BackendId,
		DeviceId,
		PreparedArtifact,
		PreparedSummary,
		StartMode,
		TrainingAlgorithm,
		TrainingDefinition,
		TrainingResult,
		TrainOptions,
		TrainingCapabilities
	} from '$lib/components/training/types';

	type FlowStatus = 'idle' | 'preparing' | 'prepared' | 'training' | 'trained' | 'error';
	type ApiBody = Record<string, unknown> & { ok?: boolean; error?: string };

	let startMode = $state<StartMode>('new');
	let algorithm = $state<TrainingAlgorithm>('supervised');
	let backend = $state<BackendId>('cpu-linear');
	let device = $state<DeviceId>('cpu');
	let capabilities = $state<TrainingCapabilities>(fallbackTrainingCapabilities);
	let status = $state<FlowStatus>('idle');
	let error = $state('');
	let artifact = $state<PreparedArtifact | null>(null);
	let result = $state<TrainingResult | null>(null);
	let activeRequest: AbortController | null = null;

	let definition = $state<TrainingDefinition>({
		sourcePath: '',
		instrument: 'GC',
		contract: 'GCZ6',
		triggerFast: 10,
		triggerSlow: 30,
		contextFast: 210,
		contextSlow: 240,
		barKind: 'minute',
		barValue: 1,
		features: [
			{ indicator: 'ema', period: 10, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'ema', period: 30, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'ema', period: 210, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'ema', period: 240, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'atr', period: 14, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'adx', period: 14, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'rvol', period: 20, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true },
			{ indicator: 'er', period: 14, lookbacks: [1, 3, 5], include_value: true, include_delta: true, normalize_by_atr: true }
		],
		atrPeriod: 14,
		slopeLookback: 5,
		contractMultiplier: 100,
		roundTripCost: 0,
		timestampUnit: '',
		allowIndexTimestamps: false
	});

	let options = $state<TrainOptions>({
		epochs: 50,
		checkpointEvery: 10,
		learningRate: 0.001,
		l2: 0.0001,
		seed: 42,
		trainFraction: 0.6,
		validationFraction: 0.2,
		resumePolicy: '',
		datasetPath: ''
	});

	const busy = $derived(status === 'preparing' || status === 'training');
	const resetAfterInputChange = () => {
		result = null;
		error = '';
		if (!busy) status = startMode === 'new' && artifact ? 'prepared' : 'idle';
	};
	const invalidatePreparedState = (clearDataset: boolean) => {
		artifact = null;
		result = null;
		error = '';
		if (clearDataset) options = { ...options, datasetPath: '' };
		if (!busy) status = 'idle';
	};
	const updateDefinition = (patch: Partial<TrainingDefinition>) => {
		definition = { ...definition, ...patch };
		// A prepared dataset is defined by every field in this object. Never
		// leave the previous artifact available after a definition edit.
		invalidatePreparedState(startMode === 'new');
	};
	const updateOptions = (patch: Partial<TrainOptions>) => {
		options = { ...options, ...patch };
		if ('datasetPath' in patch || 'resumePolicy' in patch) artifact = null;
		resetAfterInputChange();
	};
	const canPrepare = $derived(
		startMode === 'new' &&
		definition.sourcePath.trim().length > 0 &&
		definition.features.length > 0 &&
		definition.triggerFast < definition.triggerSlow &&
		definition.contextFast < definition.contextSlow &&
		!busy
	);
	const canTrain = $derived(
		capabilities.backends[backend][algorithm] &&
		capabilities.backends[backend].devices[device] &&
		!busy &&
		(startMode === 'new'
			? Boolean(artifact?.path)
			: options.datasetPath.trim().length > 0 && options.resumePolicy.trim().length > 0)
	);

	const chooseBackend = (value: TrainingAlgorithm, caps = capabilities): BackendId => {
		const preferred: BackendId = value === 'supervised' ? 'cpu-linear' : value === 'ga' ? 'burn' : 'candle';
		if (caps.backends[preferred][value]) return preferred;
		return (['candle', 'burn', 'torch'] as BackendId[]).find((id) => caps.backends[id][value]) ?? 'cpu-linear';
	};

	const ensureRuntime = () => {
		if (!capabilities.backends[backend][algorithm]) backend = chooseBackend(algorithm);
		if (!capabilities.backends[backend].devices[device]) device = 'cpu';
	};

	onMount(() => {
		let active = true;
		void fetch('/api/train/capabilities')
			.then(async (response) => {
				if (!response.ok) throw new Error(`capability request failed (${response.status})`);
				const body = (await response.json()) as { capabilities?: TrainingCapabilities };
				if (active && body.capabilities) {
					capabilities = body.capabilities;
					ensureRuntime();
				}
			})
			.catch(() => {
				// Keep the conservative fallback matrix when the status endpoint is unavailable.
			});
		return () => {
			active = false;
		};
	});
	const currentStep = $derived<1 | 2 | 3>(
		status === 'preparing' ? 2 : artifact || status === 'training' || status === 'trained' ? 3 : 1
	);

	const setStartMode = (value: StartMode) => {
		if (busy) return;
		startMode = value;
		error = '';
		result = null;
		if (value === 'continue') {
			// Continue uses the explicitly selected saved artifacts. A freshly
			// prepared artifact must not be mistaken for a continuation target.
			artifact = null;
			status = 'idle';
		} else {
			status = artifact ? 'prepared' : 'idle';
		}
	};

	const setAlgorithm = (value: TrainingAlgorithm) => {
		if (busy || value !== 'supervised') return;
		algorithm = value;
		backend = chooseBackend(value);
		device = capabilities.backends[backend].devices[device] ? device : 'cpu';
		resetAfterInputChange();
	};

	const setBackend = (value: BackendId) => {
		if (busy || !capabilities.backends[value][algorithm]) return;
		backend = value;
		if (!capabilities.backends[backend].devices[device]) device = 'cpu';
		resetAfterInputChange();
	};

	const setDevice = (value: DeviceId) => {
		if (busy || !capabilities.backends[backend].devices[value]) return;
		device = value;
		resetAfterInputChange();
	};

	const readApiBody = async (response: Response): Promise<ApiBody> => {
		const body: unknown = await response.json().catch(() => ({}));
		if (typeof body !== 'object' || body === null || Array.isArray(body)) {
			throw new Error(`Server returned an invalid response (${response.status})`);
		}
		return body as ApiBody;
	};

	const prepare = async () => {
		if (!canPrepare) return;
		const requestController = new AbortController();
		activeRequest = requestController;
		// Invalidate before the request starts as well as on failure. This
		// prevents a failed re-prepare from falling back to the old dataset.
		artifact = null;
		result = null;
		options = { ...options, datasetPath: '' };
		status = 'preparing';
		error = '';
		try {
			const response = await fetch('/api/supervised/prepare', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				signal: requestController.signal,
				body: JSON.stringify({
					input: definition.sourcePath,
					instrument: definition.instrument,
					contract: definition.contract,
					trigger_fast: definition.triggerFast,
					trigger_slow: definition.triggerSlow,
					context_fast: definition.contextFast,
					context_slow: definition.contextSlow,
					features: definition.features,
					bar_kind: definition.barKind,
					bar_value: definition.barValue,
					atr_period: definition.atrPeriod,
					slope_lookback: definition.slopeLookback,
					contract_multiplier: definition.contractMultiplier,
					round_trip_cost: definition.roundTripCost,
					timestamp_unit: definition.timestampUnit || undefined,
					allow_index_timestamps: definition.allowIndexTimestamps
				})
			});
			const body = await readApiBody(response);
			if (!response.ok || body.ok !== true) throw new Error(body.error || 'Preparation failed');
			const artifactPath = typeof body.artifactPath === 'string' ? body.artifactPath.trim() : '';
			if (!artifactPath) throw new Error('Preparation succeeded without a dataset artifact');
			artifact = {
				path: artifactPath,
				summary: (body.summary as PreparedSummary | undefined) ?? null
			};
			options = { ...options, datasetPath: artifact.path };
			status = 'prepared';
		} catch (reason) {
			artifact = null;
			result = null;
			options = { ...options, datasetPath: '' };
			if (reason instanceof DOMException && reason.name === 'AbortError') {
				error = 'Preparation stopped.';
				status = 'idle';
			} else {
				error = reason instanceof Error ? reason.message : String(reason);
				status = 'error';
			}
		} finally {
			if (activeRequest === requestController) activeRequest = null;
		}
	};

	const train = async () => {
		if (!canTrain) return;
		const datasetPath = (startMode === 'new' ? artifact?.path : options.datasetPath)?.trim() ?? '';
		if (!datasetPath) return;
		const requestController = new AbortController();
		activeRequest = requestController;
		status = 'training';
		error = '';
		result = null;
		try {
			const response = await fetch('/api/supervised/train', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				signal: requestController.signal,
				body: JSON.stringify({
					mode: startMode,
					input: datasetPath,
					backend,
					device,
					resume_policy: startMode === 'continue' ? options.resumePolicy : undefined,
					epochs: options.epochs,
					learning_rate: options.learningRate,
					l2: options.l2,
					seed: options.seed,
					train_fraction: options.trainFraction,
					validation_fraction: options.validationFraction,
					checkpoint_every: options.checkpointEvery
				})
			});
			const body = await readApiBody(response);
			if (!response.ok || body.ok !== true) throw new Error(body.error || 'Training failed');
			result = {
				runDir: String(body.runDir ?? ''),
				datasetPath: String(body.datasetPath ?? ''),
				policyPath: String(body.policyPath ?? ''),
				metricsPath: String(body.metricsPath ?? ''),
				summary: (body.summary as Record<string, unknown> | undefined) ?? null
			};
			options = { ...options, resumePolicy: result.policyPath };
			status = 'trained';
		} catch (reason) {
			if (reason instanceof DOMException && reason.name === 'AbortError') {
				error = 'Training stopped.';
				status = artifact ? 'prepared' : 'idle';
			} else {
				error = reason instanceof Error ? reason.message : String(reason);
				status = 'error';
			}
		} finally {
			if (activeRequest === requestController) activeRequest = null;
		}
	};

	const stop = () => activeRequest?.abort();
</script>

<main class="training-workspace mx-auto w-full max-w-7xl px-3 py-2 sm:px-5 sm:py-3">
	<header class="training-header mb-2 flex flex-wrap items-center justify-between gap-2">
		<div>
			<div class="mb-0.5 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground"><Activity size={13} /> Training workspace</div>
			<h1 class="text-xl font-semibold tracking-tight sm:text-2xl">From bars to a causal policy</h1>
			<p class="mt-0.5 hidden text-xs text-muted-foreground sm:block">Choose a run, define the event state, then prepare and train.</p>
		</div>
		<div class="hidden items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-[11px] text-muted-foreground sm:inline-flex"><ShieldCheck size={13} class="text-emerald-600" /> Project-confined artifacts</div>
	</header>

	<TrainingChoices
		{startMode}
		{algorithm}
		{backend}
		{device}
		{capabilities}
		showAlgorithmChoices={false}
		onStartMode={setStartMode}
		onAlgorithm={setAlgorithm}
		onBackend={setBackend}
		onDevice={setDevice}
	/>

	{#if algorithm === 'supervised'}
		<div class="step-card my-2 rounded-xl border bg-card px-3 py-2 shadow-xs"><StepIndicator step={currentStep} /></div>
		<div class="event-grid grid gap-3 lg:grid-cols-[minmax(0,1.65fr)_minmax(18rem,0.75fr)]">
			<DefinitionPanel
				{definition}
				{options}
				{startMode}
				disabled={busy}
				onDefinitionChange={updateDefinition}
				onOptionsChange={updateOptions}
			/>
			<ArtifactPanel
				{artifact}
				{result}
				{backend}
				{device}
				{startMode}
				{status}
				{error}
				{canPrepare}
				{canTrain}
				onPrepare={() => void prepare()}
				onTrain={() => void train()}
				onStop={stop}
			/>
		</div>
		{#if result?.summary}
			<SupervisedDiagnostics summary={result.summary} />
		{/if}
	{/if}
</main>

<style>
	:global(.training-workspace .event-grid > section),
	:global(.training-workspace .event-grid > aside) {
		min-width: 0;
	}

	@media (min-width: 1024px) and (max-height: 800px) {
		:global(.training-workspace) {
			height: calc(100vh - 3.5625rem);
			max-height: calc(100vh - 3.5625rem);
			overflow: hidden;
		}

		:global(.training-workspace .training-header) {
			margin-bottom: 0.25rem;
		}

		:global(.training-workspace .training-header p) {
			display: none;
		}

		:global(.training-workspace .step-card) {
			margin-block: 0.35rem;
			padding-block: 0.35rem;
		}

		:global(.training-workspace .event-grid) {
			min-height: 0;
			max-height: calc(100vh - 11.5rem);
		}

		:global(.training-workspace .event-grid > section),
		:global(.training-workspace .event-grid > aside) {
			min-height: 0;
			overflow-y: auto;
		}
	}
</style>

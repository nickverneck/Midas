<script lang="ts">
	import { AlertTriangle, Info, TrendingDown, TrendingUp } from 'lucide-svelte';
	import type { ChartConfiguration } from 'chart.js';
	import GaChart from '$lib/components/GaChart.svelte';
	import type {
		SupervisedCheckpointMetrics,
		SupervisedSplitMetrics,
		SupervisedTrainingSummary
	} from './types';

	type Props = { summary: Record<string, unknown> | null };
	type DiagnosticPoint = {
		epoch: number;
		train: SupervisedSplitMetrics;
		validation: SupervisedSplitMetrics;
	};
	type BestCheckpoint = {
		point: DiagnosticPoint;
		key: 'predicted_pnl' | 'cross_entropy';
	};
	type ChartData = ChartConfiguration['data'];
	type ChartOptions = ChartConfiguration['options'];
	type ValidationQuality = {
		label: 'Insufficient quality data' | 'Weak validation quality' | 'Mixed validation quality' | 'Promising validation quality';
		tone: 'insufficient' | 'weak' | 'mixed' | 'promising';
	};

	const TRAIN_VALIDATION_GAP_THRESHOLD = 0.1;
	const CROSS_ENTROPY_GAP_THRESHOLD = 0.15;

	let { summary }: Props = $props();

	const asRecord = (value: unknown): Record<string, unknown> | null =>
		typeof value === 'object' && value !== null && !Array.isArray(value)
			? (value as Record<string, unknown>)
			: null;
	const asNumber = (value: unknown) => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};
	const asMetrics = (value: unknown): SupervisedSplitMetrics | null => {
		const record = asRecord(value);
		return record ? (record as SupervisedSplitMetrics) : null;
	};
	const asSummary = $derived((asRecord(summary) ?? {}) as SupervisedTrainingSummary);
	const metric = (metrics: SupervisedSplitMetrics | null | undefined, key: keyof SupervisedSplitMetrics) => {
		const observations = asNumber(metrics?.rows) ?? asNumber(metrics?.sessions);
		// Empty splits are serialized with zero-valued metrics by the trainer;
		// those values are not evidence for a curve or checkpoint choice.
		if (observations !== null && observations <= 0) return null;
		return asNumber(metrics?.[key]);
	};
	const meaningfulMetricKeys = ['accuracy', 'cross_entropy', 'macro_f1'] as const;
	const hasMeaningfulMetrics = (metrics: SupervisedSplitMetrics | null | undefined) => {
		if (meaningfulMetricKeys.some((key) => metric(metrics, key) !== null)) return true;
		const predictedPnl = metric(metrics, 'predicted_pnl');
		// A zero PnL without a loss or quality metric is commonly an empty or
		// placeholder split, not evidence for selecting a checkpoint.
		return predictedPnl !== null && predictedPnl !== 0;
	};
	const hasObviousTrainValidationDivergence = (
		train: SupervisedSplitMetrics | null | undefined,
		validation: SupervisedSplitMetrics | null | undefined
	) => {
		const trainF1 = metric(train, 'macro_f1');
		const validationF1 = metric(validation, 'macro_f1');
		const trainAccuracy = metric(train, 'accuracy');
		const validationAccuracy = metric(validation, 'accuracy');
		const trainLoss = metric(train, 'cross_entropy');
		const validationLoss = metric(validation, 'cross_entropy');
		return (
			(trainF1 !== null && validationF1 !== null && trainF1 - validationF1 >= TRAIN_VALIDATION_GAP_THRESHOLD) ||
			(trainAccuracy !== null && validationAccuracy !== null && trainAccuracy - validationAccuracy >= TRAIN_VALIDATION_GAP_THRESHOLD) ||
			(trainLoss !== null && validationLoss !== null && validationLoss - trainLoss >= CROSS_ENTROPY_GAP_THRESHOLD)
		);
	};
	const percent = (value: number | null) => (value === null ? '—' : `${(value * 100).toFixed(1)}%`);
	const signed = (value: number | null, digits = 2) =>
		value === null
			? '—'
			: `${value >= 0 ? '+' : ''}${value.toLocaleString('en-US', { maximumFractionDigits: digits })}`;
	const titleCase = (value: string) => value.replace(/[-_]/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

	const checkpoints = $derived.by<DiagnosticPoint[]>(() => {
		const raw = asSummary.checkpoints;
		if (!Array.isArray(raw)) return [];
		return raw
			.map((value) => {
				const checkpoint = asRecord(value) as SupervisedCheckpointMetrics | null;
				const epoch = asNumber(checkpoint?.epoch);
				const train = asMetrics(checkpoint?.train) ?? {};
				const validation = asMetrics(checkpoint?.validation) ?? {};
				return epoch !== null && (hasMeaningfulMetrics(train) || hasMeaningfulMetrics(validation))
					? { epoch, train, validation }
					: null;
			})
			.filter((value): value is DiagnosticPoint => value !== null)
			.sort((left, right) => left.epoch - right.epoch);
	});

	const finalTrain = $derived.by(() => {
		const summaryTrain = asMetrics(asSummary.train);
		if (hasMeaningfulMetrics(summaryTrain)) return summaryTrain;
		return [...checkpoints].reverse().find((point) => hasMeaningfulMetrics(point.train))?.train ?? null;
	});
	const finalValidation = $derived.by(() => {
		const summaryValidation = asMetrics(asSummary.validation);
		if (hasMeaningfulMetrics(summaryValidation)) return summaryValidation;
		return [...checkpoints].reverse().find((point) => hasMeaningfulMetrics(point.validation))?.validation ?? null;
	});
	const holdout = $derived(asMetrics(asSummary.holdout));
	const validationQuality = $derived.by<ValidationQuality>(() => {
		const validationF1 = metric(finalValidation, 'macro_f1');
		const validationAccuracy = metric(finalValidation, 'accuracy');
		if (validationF1 === null && validationAccuracy === null) {
			return { label: 'Insufficient quality data', tone: 'insufficient' };
		}
		if ((validationF1 !== null && validationF1 < 0.4) || (validationAccuracy !== null && validationAccuracy < 0.5)) {
			return { label: 'Weak validation quality', tone: 'weak' };
		}
		if ((validationF1 !== null && validationF1 >= 0.6) || (validationAccuracy !== null && validationAccuracy >= 0.65)) {
			return { label: 'Promising validation quality', tone: 'promising' };
		}
		return { label: 'Mixed validation quality', tone: 'mixed' };
	});
	const qualityClass = $derived(
		validationQuality.tone === 'weak'
			? 'border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-50'
			: validationQuality.tone === 'promising'
				? 'border-emerald-200 bg-emerald-50 text-emerald-950 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-50'
				: 'border-slate-200 bg-slate-50 text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200'
	);
	const bestCheckpoint = $derived.by<BestCheckpoint | null>(() => {
		const withPnl = checkpoints.filter((point) => {
			const predictedPnl = metric(point.validation, 'predicted_pnl');
			return predictedPnl !== null && predictedPnl !== 0;
		});
		if (withPnl.length > 0) {
			return {
				point: withPnl.reduce((best, point) =>
				(metric(point.validation, 'predicted_pnl') ?? -Infinity) >
				(metric(best.validation, 'predicted_pnl') ?? -Infinity)
					? point
					: best
			),
				key: 'predicted_pnl'
			};
		}
		const withLoss = checkpoints.filter((point) => metric(point.validation, 'cross_entropy') !== null);
		if (withLoss.length === 0) return null;
		return {
			point: withLoss.reduce((best, point) =>
				(metric(point.validation, 'cross_entropy') ?? Infinity) <
				(metric(best.validation, 'cross_entropy') ?? Infinity)
					? point
					: best
			),
			key: 'cross_entropy'
		};
	});

	const chartOptions: ChartOptions = {
		animation: false,
		plugins: {
			legend: {
				display: true,
				position: 'bottom',
				labels: { boxWidth: 10, usePointStyle: true, font: { size: 10 } }
			}
		},
		scales: {
			x: {
				grid: { display: false },
				ticks: { maxTicksLimit: 8, font: { size: 10 } }
			},
			y: { ticks: { font: { size: 10 } } }
		}
	};

	const chartData = $derived.by(() => {
		const series = (key: keyof SupervisedSplitMetrics): ChartData => {
			const points = checkpoints.filter(
				(point) => metric(point.train, key) !== null || metric(point.validation, key) !== null
			);
			const labels = points.map((point) => point.epoch.toLocaleString('en-US'));
			return {
				labels,
				datasets: [
					{
						label: 'Train',
						data: points.map((point) => metric(point.train, key)),
						borderColor: '#2563eb',
						backgroundColor: 'transparent',
						borderWidth: 2,
						tension: 0.25,
						pointRadius: points.length > 40 ? 0 : 2
					},
					{
						label: 'Validation',
						data: points.map((point) => metric(point.validation, key)),
						borderColor: '#0f766e',
						backgroundColor: 'transparent',
						borderWidth: 2,
						tension: 0.25,
						pointRadius: points.length > 40 ? 0 : 2
					}
				]
			};
		};

		const pnlBaselines = [
			{ key: 'always_normal_pnl' as const, label: 'Always normal', color: '#64748b' },
			{ key: 'always_invert_pnl' as const, label: 'Always invert', color: '#d97706' },
			{ key: 'always_skip_pnl' as const, label: 'Always skip', color: '#94a3b8' }
			];
			const hasPnlSignal = checkpoints.some((point) =>
				[metric(point.train, 'predicted_pnl'), metric(point.validation, 'predicted_pnl')].some(
					(value) => value !== null && value !== 0
				)
			);
			const pnlPoints = hasPnlSignal
				? checkpoints.filter(
						(point) => metric(point.train, 'predicted_pnl') !== null || metric(point.validation, 'predicted_pnl') !== null
					)
				: [];
			const pnl = series('predicted_pnl');
			for (const baseline of pnlBaselines) {
				pnl.datasets.push({
					label: baseline.label,
					data: pnlPoints.map((point) => metric(point.validation, baseline.key)),
				borderColor: baseline.color,
				backgroundColor: 'transparent',
				borderDash: [5, 4],
				borderWidth: 1.5,
				pointRadius: 0,
				tension: 0
			});
		}
		return {
			crossEntropy: series('cross_entropy'),
			quality: series('macro_f1'),
			pnl
		};
	});
	const chartAvailability = $derived({
		crossEntropy: checkpoints.some((point) => metric(point.train, 'cross_entropy') !== null || metric(point.validation, 'cross_entropy') !== null),
		quality: checkpoints.some((point) => metric(point.train, 'macro_f1') !== null || metric(point.validation, 'macro_f1') !== null),
		pnl: checkpoints.some((point) =>
			[metric(point.train, 'predicted_pnl'), metric(point.validation, 'predicted_pnl')].some(
				(value) => value !== null && value !== 0
			)
		)
	});

	const diagnosis = $derived.by(() => {
		const rawValidationPnls = checkpoints.map((point) => metric(point.validation, 'predicted_pnl')).filter((value): value is number => value !== null);
		const validationPnls = rawValidationPnls.some((value) => value !== 0) ? rawValidationPnls : [];
		const validationLosses = checkpoints.map((point) => metric(point.validation, 'cross_entropy')).filter((value): value is number => value !== null);
		const finalPnl = metric(finalValidation, 'predicted_pnl');
		const bestPnl = Math.max(...validationPnls, -Infinity);
		const minLoss = Math.min(...validationLosses, Infinity);
		const finalLoss = metric(finalValidation, 'cross_entropy');
		const trainAccuracy = metric(finalTrain, 'accuracy');
		const validationAccuracy = metric(finalValidation, 'accuracy');
		const validationBaseline = metric(finalValidation, 'always_normal_pnl');
		const range = validationPnls.length > 1 ? Math.max(...validationPnls) - Math.min(...validationPnls) : 0;
		const signChanges = validationPnls.slice(1).reduce((count, value, index) => {
			const previous = validationPnls[index];
			return count + (previous !== 0 && value !== 0 && Math.sign(previous) !== Math.sign(value) ? 1 : 0);
		}, 0);
		const obviousDivergence =
			hasObviousTrainValidationDivergence(finalTrain, finalValidation) ||
			checkpoints.some((point) => hasObviousTrainValidationDivergence(point.train, point.validation));
		const possibleOverfit =
			obviousDivergence ||
			(finalPnl !== null && bestPnl !== -Infinity && finalPnl < bestPnl - Math.max(1, Math.abs(bestPnl) * 0.1)) ||
			(finalLoss !== null && minLoss !== Infinity && finalLoss > minLoss + 0.1);
		const unstable = signChanges >= 2 || range > Math.max(10, Math.abs(bestPnl) * 0.75);
		const underfit =
			trainAccuracy !== null &&
			validationAccuracy !== null &&
			trainAccuracy < 0.5 &&
			validationAccuracy < 0.5 &&
			(validationBaseline === null || (finalPnl ?? -Infinity) <= validationBaseline);

		const hasValidationSignal =
			hasMeaningfulMetrics(finalValidation) ||
			checkpoints.some((point) => hasMeaningfulMetrics(point.validation));
		if (!hasValidationSignal) {
			return {
				kind: 'insufficient',
				label: 'Insufficient validation data',
				message: 'The result does not contain a finite validation metric yet. Keep the run output and metrics artifact, then run with a populated validation split or checkpoint metrics.'
			};
		}
		if (possibleOverfit) {
			return {
				kind: 'overfit',
				label: 'Possible overfit',
				message: obviousDivergence ? 'Training quality is materially ahead of validation on the available validation-safe metrics. Inspect the best validation checkpoint and consider stronger regularization or fewer epochs.' : 'Validation peaked before the final epoch. Resume from the best validation checkpoint or reduce training duration.'
			};
		}
		if (checkpoints.length < 2) {
			return {
				kind: 'limited',
				label: 'More checkpoints needed',
				message: 'Set a positive checkpoint interval and run enough epochs to see whether validation improves or starts to degrade.'
			};
		}
		if (unstable) {
			return {
				kind: 'unstable',
				label: 'Unstable validation',
				message: 'Validation moves sharply between checkpoints. Try a lower learning rate, a different seed, or more representative sessions.'
			};
		}
		if (underfit) {
			return {
				kind: 'underfit',
				label: 'Likely underfit',
				message: 'Train and validation quality remain weak. Add useful state features, revisit labels, or train longer before judging the model.'
			};
		}
		return {
			kind: 'neutral',
			label: 'No obvious overfit',
			message: 'The available validation curves show no clear overfit signal. This is a fit assessment, not a claim that validation quality or profitability is good.'
		};
	});

	const statusClass = $derived(
		diagnosis.kind === 'neutral'
			? 'border-slate-200 bg-slate-50 text-slate-800 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200'
			: diagnosis.kind === 'limited' || diagnosis.kind === 'insufficient'
				? 'border-sky-200 bg-sky-50 text-sky-950 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-50'
				: 'border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-50'
	);
	const trainValidationGap = $derived.by(() => {
		const trainF1 = metric(finalTrain, 'macro_f1');
		const validationF1 = metric(finalValidation, 'macro_f1');
		return trainF1 !== null && validationF1 !== null ? trainF1 - validationF1 : null;
	});
	const holdoutVsValidation = $derived.by(() => {
		const holdoutPnl = metric(holdout, 'predicted_pnl');
		const validationPnl = metric(finalValidation, 'predicted_pnl');
		return holdoutPnl !== null && validationPnl !== null ? holdoutPnl - validationPnl : null;
	});
	const runtime = $derived(
		`${titleCase(String(asSummary.backend ?? 'unknown'))} · ${titleCase(String(asSummary.device ?? 'unknown'))}`
	);
	const splitRows = $derived([
		{ label: 'Train', metrics: finalTrain },
		{ label: 'Validation', metrics: finalValidation },
		{ label: 'Holdout', metrics: holdout }
	]);

	const qualityChart = $derived.by(() => ({
		...chartData.quality,
		datasets: chartData.quality.datasets.map((dataset) => ({
			...dataset,
			label: dataset.label === 'Train' ? 'Train macro-F1' : 'Validation macro-F1'
		}))
	} satisfies ChartData));
</script>

<section class="rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="supervised-diagnostics-title">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<div class="flex items-center gap-2">
				<h2 id="supervised-diagnostics-title" class="text-sm font-semibold">Supervised diagnostics</h2>
				<span class={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${statusClass}`}>{diagnosis.label}</span>
				<span class={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${qualityClass}`}>Validation: {validationQuality.label}</span>
			</div>
			<p class="mt-1 text-[11px] text-muted-foreground">{runtime} · validation is used for diagnosis; holdout is shown only as a final comparison.</p>
		</div>
		<div class="flex items-center gap-2 text-[10px] text-muted-foreground">
			<span>{checkpoints.length.toLocaleString('en-US')} checkpoints</span>
			{#if asSummary.optimizer_state_saved}<span class="rounded-full bg-muted px-2 py-1">AdamW state saved</span>{/if}
		</div>
	</div>

	<div class={`mt-3 flex items-start gap-2 rounded-lg border p-2.5 text-[11px] leading-4 ${statusClass}`}>
		{#if diagnosis.kind === 'neutral'}<Info class="mt-0.5 shrink-0" size={15} />
		{:else if diagnosis.kind === 'limited' || diagnosis.kind === 'insufficient'}<Info class="mt-0.5 shrink-0" size={15} />
		{:else}<AlertTriangle class="mt-0.5 shrink-0" size={15} />{/if}
		<span>{diagnosis.message}</span>
	</div>

		<div class="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
			<div class="rounded-lg border p-2.5">
				<span class="block text-[10px] uppercase tracking-wide text-muted-foreground">Best validation checkpoint</span>
				{#if bestCheckpoint}
					<b class="mt-1 block text-base">Epoch {bestCheckpoint.point.epoch.toLocaleString('en-US')}</b>
					<span class="text-[10px] text-muted-foreground">{bestCheckpoint.key === 'predicted_pnl' ? 'Validation PnL' : 'Validation cross-entropy'} {signed(metric(bestCheckpoint.point.validation, bestCheckpoint.key))}</span>
				{:else}
					<b class="mt-1 block text-base">Insufficient data</b>
					<span class="text-[10px] text-muted-foreground">No finite validation selection metric</span>
				{/if}
			</div>
		<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase tracking-wide text-muted-foreground">Train → validation F1 gap</span><b class="mt-1 block text-base">{trainValidationGap === null ? '—' : `${(trainValidationGap * 100).toFixed(1)} pts`}</b><span class="text-[10px] text-muted-foreground">Overfit flag at ≥{(TRAIN_VALIDATION_GAP_THRESHOLD * 100).toFixed(1)} pts</span></div>
		<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase tracking-wide text-muted-foreground">Holdout vs validation</span><b class="mt-1 block text-base">{signed(holdoutVsValidation)}</b><span class="text-[10px] text-muted-foreground">Predicted PnL delta</span></div>
		<div class="rounded-lg border p-2.5"><span class="block text-[10px] uppercase tracking-wide text-muted-foreground">Holdout edge</span><b class="mt-1 block text-base">{signed(metric(holdout, 'predicted_pnl') === null || metric(holdout, 'always_normal_pnl') === null ? null : metric(holdout, 'predicted_pnl')! - metric(holdout, 'always_normal_pnl')!)}</b><span class="text-[10px] text-muted-foreground">Against always normal</span></div>
	</div>

	{#if checkpoints.length > 0}
		<div class="mt-3 grid gap-3 xl:grid-cols-3">
			{#if chartAvailability.crossEntropy}
				<div class="min-w-0 rounded-lg border p-2.5">
					<div class="mb-1 flex items-baseline justify-between gap-2"><div><h3 class="text-xs font-semibold">Cross-entropy</h3><p class="text-[10px] text-muted-foreground">Lower is better</p></div><TrendingDown size={14} class="text-muted-foreground" /></div>
					<div class="h-56"><GaChart data={chartData.crossEntropy} options={chartOptions} /></div>
				</div>
			{:else}
				<div class="min-w-0 rounded-lg border border-dashed p-2.5 text-[11px] text-muted-foreground"><h3 class="text-xs font-semibold text-foreground">Cross-entropy</h3><div class="mt-3 flex items-start gap-2"><Info size={14} class="mt-0.5 shrink-0" /><span>Insufficient checkpoint metrics for this chart.</span></div></div>
			{/if}
			{#if chartAvailability.quality}
				<div class="min-w-0 rounded-lg border p-2.5">
					<div class="mb-1 flex items-baseline justify-between gap-2"><div><h3 class="text-xs font-semibold">Macro-F1</h3><p class="text-[10px] text-muted-foreground">Train vs validation quality</p></div><TrendingUp size={14} class="text-muted-foreground" /></div>
					<div class="h-56"><GaChart data={qualityChart} options={chartOptions} /></div>
				</div>
			{:else}
				<div class="min-w-0 rounded-lg border border-dashed p-2.5 text-[11px] text-muted-foreground"><h3 class="text-xs font-semibold text-foreground">Macro-F1</h3><div class="mt-3 flex items-start gap-2"><Info size={14} class="mt-0.5 shrink-0" /><span>Insufficient checkpoint metrics for this chart.</span></div></div>
			{/if}
			{#if chartAvailability.pnl}
				<div class="min-w-0 rounded-lg border p-2.5">
					<div class="mb-1 flex items-baseline justify-between gap-2"><div><h3 class="text-xs font-semibold">Predicted PnL</h3><p class="text-[10px] text-muted-foreground">Dashed lines are validation baselines</p></div><TrendingUp size={14} class="text-muted-foreground" /></div>
					<div class="h-56"><GaChart data={chartData.pnl} options={chartOptions} /></div>
				</div>
			{:else}
				<div class="min-w-0 rounded-lg border border-dashed p-2.5 text-[11px] text-muted-foreground"><h3 class="text-xs font-semibold text-foreground">Predicted PnL</h3><div class="mt-3 flex items-start gap-2"><Info size={14} class="mt-0.5 shrink-0" /><span>Insufficient checkpoint metrics for this chart.</span></div></div>
			{/if}
		</div>
	{:else}
		<div class="mt-3 flex items-start gap-2 rounded-lg border border-dashed p-3 text-[11px] text-muted-foreground"><Info class="mt-0.5 shrink-0" size={14} /><span>No checkpoint curves were returned. Set <b>Checkpoint every</b> above zero for the next run; final train, validation, and holdout metrics remain available in the result artifact.</span></div>
	{/if}

	<div class="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.7fr)]">
		<div class="rounded-lg border p-2.5">
			<h3 class="text-xs font-semibold">Final split comparison</h3>
			<div class="mt-2 space-y-2 sm:hidden">
				{#each splitRows as row}
					<div class="rounded-md border bg-muted/20 p-2">
						<div class="mb-1.5 text-[11px] font-semibold">{row.label}</div>
						<dl class="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
							<div><dt class="text-muted-foreground">Accuracy</dt><dd class="font-medium">{percent(metric(row.metrics, 'accuracy'))}</dd></div>
							<div><dt class="text-muted-foreground">Macro-F1</dt><dd class="font-medium">{percent(metric(row.metrics, 'macro_f1'))}</dd></div>
							<div><dt class="text-muted-foreground">Predicted PnL</dt><dd class="font-medium">{signed(metric(row.metrics, 'predicted_pnl'))}</dd></div>
							<div><dt class="text-muted-foreground">Always normal</dt><dd class="font-medium">{signed(metric(row.metrics, 'always_normal_pnl'))}</dd></div>
						</dl>
					</div>
				{/each}
			</div>
			<div class="mt-2 hidden overflow-x-auto sm:block">
				<table class="w-full min-w-[32rem] text-left text-[10px]">
					<thead class="border-b text-muted-foreground"><tr><th class="py-1.5 pr-3 font-medium">Split</th><th class="py-1.5 pr-3 font-medium">Accuracy</th><th class="py-1.5 pr-3 font-medium">Macro-F1</th><th class="py-1.5 pr-3 font-medium">Predicted PnL</th><th class="py-1.5 font-medium">Always normal</th></tr></thead>
					<tbody>
						<tr class="border-b"><td class="py-1.5 pr-3 font-medium">Train</td><td class="py-1.5 pr-3">{percent(metric(finalTrain, 'accuracy'))}</td><td class="py-1.5 pr-3">{percent(metric(finalTrain, 'macro_f1'))}</td><td class="py-1.5 pr-3">{signed(metric(finalTrain, 'predicted_pnl'))}</td><td class="py-1.5">{signed(metric(finalTrain, 'always_normal_pnl'))}</td></tr>
						<tr class="border-b"><td class="py-1.5 pr-3 font-medium">Validation</td><td class="py-1.5 pr-3">{percent(metric(finalValidation, 'accuracy'))}</td><td class="py-1.5 pr-3">{percent(metric(finalValidation, 'macro_f1'))}</td><td class="py-1.5 pr-3">{signed(metric(finalValidation, 'predicted_pnl'))}</td><td class="py-1.5">{signed(metric(finalValidation, 'always_normal_pnl'))}</td></tr>
						<tr><td class="py-1.5 pr-3 font-medium">Holdout</td><td class="py-1.5 pr-3">{percent(metric(holdout, 'accuracy'))}</td><td class="py-1.5 pr-3">{percent(metric(holdout, 'macro_f1'))}</td><td class="py-1.5 pr-3">{signed(metric(holdout, 'predicted_pnl'))}</td><td class="py-1.5">{signed(metric(holdout, 'always_normal_pnl'))}</td></tr>
					</tbody>
				</table>
			</div>
		</div>
		<div class="rounded-lg bg-muted/45 p-2.5 text-[10px] leading-4 text-muted-foreground">
			<div class="flex items-center gap-1.5 font-semibold text-foreground"><Info size={13} /> How to read this</div>
			<ul class="mt-1.5 space-y-1.5">
					<li>No obvious overfit means validation follows training; the separate quality badge is not proof of profitability.</li>
				<li>Use the best validation epoch as the candidate checkpoint, then test that policy on older weeks.</li>
					<li>Holdout is not used to choose the epoch; use it only as a post-hoc comparison against validation.</li>
			</ul>
		</div>
	</div>
</section>

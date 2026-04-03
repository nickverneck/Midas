<script lang="ts">
	import { onMount } from "svelte";
	import { AlertCircle } from "lucide-svelte";
	import BehaviorCandlestickModal from "$lib/components/BehaviorCandlestickModal.svelte";
	import * as Alert from "$lib/components/ui/alert";
	import { Badge } from "$lib/components/ui/badge";
	import * as Tabs from "$lib/components/ui/tabs";
	import {
		DEFAULT_LOG_DIR,
		MIN_ZOOM_WINDOW,
		PAGE_SIZE,
		ZOOM_STEP,
		applyLogChunk,
		avgOrNull,
		buildActiveChartMeta,
		buildBehaviorDisplay,
		buildDrawdownChartData,
		buildFilteredSummary,
		buildFitnessChartData,
		buildFrontierChartData,
		buildFrontierOptions,
		buildGenData,
		buildIssues,
		buildLogsUrl,
		buildPagedLogs,
		buildPlateauMetricLabel,
		buildPlateauResult,
		buildPnlChartData,
		buildRealizedChartData,
		buildZoomRangeLabel,
		createEmptyIssueState,
		detectEval,
		formatBehaviorLabel,
		normalizeLogDir,
		parseBehaviorFile,
		parseCsvRows,
		toIndex
	} from "./analytics";
	import type {
		BehaviorFile,
		BehaviorFilter,
		BehaviorRow,
		ChartTab,
		GenMember,
		LogRow,
		MainTab,
		ParquetRow,
		PopulationFilter
	} from "./types";
	import GaBehaviorTab from "./_components/GaBehaviorTab.svelte";
	import GaDataTab from "./_components/GaDataTab.svelte";
	import GaEvolutionTab from "./_components/GaEvolutionTab.svelte";
	import GaIssuesTab from "./_components/GaIssuesTab.svelte";
	import GaOverviewTab from "./_components/GaOverviewTab.svelte";
	import GaPageHeader from "./_components/GaPageHeader.svelte";
	import GaSummaryCards from "./_components/GaSummaryCards.svelte";

	let logs: LogRow[] = $state([]);
	let loading = $state(true);
	let error = $state("");
	let loadingMore = $state(false);
	let doneLoading = $state(false);
	let nextOffset = $state(0);
	let logPage = $state(1);
	let logDir = $state(DEFAULT_LOG_DIR);
	let activeLogDir = $state(DEFAULT_LOG_DIR);
	let chartTab = $state<ChartTab>("fitness");
	let populationFilter = $state<PopulationFilter>("all");
	let mainTab = $state<MainTab>("overview");
	let zoomWindow = $state(0);
	let zoomStart = $state(0);
	let plateauWindow = $state(300);
	let plateauMinDelta = $state(0.5);
	let hasEval = $state(false);
	let bestFitness = $state(Number.NEGATIVE_INFINITY);
	let genMembers: Map<number, GenMember[]> = $state(new Map());
	let issueState = $state(createEmptyIssueState());

	let behaviorFiles: BehaviorFile[] = $state([]);
	let behaviorListLoading = $state(false);
	let behaviorError = $state("");
	let behaviorListDir = $state("");
	let behaviorFilter = $state<BehaviorFilter>("all");
	let behaviorRowLimit = $state(500);

	let selectedTrainBehavior = $state("");
	let selectedValBehavior = $state("");
	let lastTrainBehavior = $state("");
	let lastValBehavior = $state("");
	let trainBehaviorRows: BehaviorRow[] = $state([]);
	let valBehaviorRows: BehaviorRow[] = $state([]);
	let trainBehaviorLoading = $state(false);
	let valBehaviorLoading = $state(false);

	let trainDataRows: ParquetRow[] = $state([]);
	let valDataRows: ParquetRow[] = $state([]);
	let trainDataMap: Map<number, ParquetRow> = $state(new Map());
	let valDataMap: Map<number, ParquetRow> = $state(new Map());
	let trainDataLoading = $state(false);
	let valDataLoading = $state(false);

	let trainChartOpen = $state(false);
	let valChartOpen = $state(false);

	let loadToken = 0;
	let hasEvalDetermined = false;

	function resetState() {
		logs = [];
		genMembers = new Map();
		issueState = createEmptyIssueState();
		bestFitness = Number.NEGATIVE_INFINITY;
		logPage = 1;
		hasEval = false;
		hasEvalDetermined = false;
		zoomWindow = 0;
		zoomStart = 0;
	}

	function updateFromChunk(rows: LogRow[]) {
		if (!rows.length) return;

		if (!hasEvalDetermined) {
			hasEval = detectEval(rows);
			hasEvalDetermined = true;
		}

		const next = applyLogChunk(genMembers, bestFitness, issueState, rows, hasEval);
		logs.push(...rows);
		genMembers = next.genMembers;
		bestFitness = next.bestFitness;
		issueState = next.issueState;
	}

	async function fetchLogs() {
		const token = ++loadToken;
		loading = true;
		error = "";
		doneLoading = false;
		nextOffset = 0;
		loadingMore = false;
		resetState();

		try {
			const dir = normalizeLogDir(logDir);
			activeLogDir = dir;

			const res = await fetch(buildLogsUrl(0, dir));
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to fetch logs (${res.status})`);
			}

			const payload = await res.json();
			if (token !== loadToken) return;

			updateFromChunk(payload.data || []);
			nextOffset = payload.nextOffset || logs.length;
			doneLoading = payload.done || false;
			if (!doneLoading) {
				scheduleLoadMore(token, dir);
			}
		} catch (err) {
			console.error("Failed to fetch logs", err);
			error = err instanceof Error ? err.message : String(err);
		} finally {
			if (token === loadToken) {
				loading = false;
			}
		}
	}

	function scheduleLoadMore(token: number, dir: string) {
		if (doneLoading || loadingMore || token !== loadToken || dir !== activeLogDir) return;

		loadingMore = true;
		setTimeout(async () => {
			if (token !== loadToken || dir !== activeLogDir) {
				loadingMore = false;
				return;
			}

			try {
				const res = await fetch(buildLogsUrl(nextOffset, dir));
				if (!res.ok) throw new Error("Failed to fetch logs");

				const payload = await res.json();
				if (token !== loadToken || dir !== activeLogDir) return;

				if (payload.data && payload.data.length > 0) {
					updateFromChunk(payload.data);
					nextOffset = payload.nextOffset || nextOffset + payload.data.length;
				} else if (typeof payload.nextOffset === "number" && payload.nextOffset !== nextOffset) {
					nextOffset = payload.nextOffset;
				} else if (payload.done) {
					doneLoading = true;
				}

				if (payload.done) {
					doneLoading = true;
				}
			} catch (err) {
				console.error("Failed to fetch more logs", err);
				doneLoading = true;
			} finally {
				loadingMore = false;
				if (!doneLoading) {
					scheduleLoadMore(token, dir);
				}
			}
		}, 60);
	}

	async function fetchBehaviorList() {
		behaviorListLoading = true;
		behaviorError = "";

		try {
			const params = new URLSearchParams({
				mode: "list",
				dir: activeLogDir
			});

			const res = await fetch(`/api/behavior?${params.toString()}`);
			if (!res.ok) {
				const payload = await res.json().catch(() => ({}));
				throw new Error(payload?.error || `Failed to list behavior CSVs (${res.status})`);
			}

			const payload = await res.json();
			const files = Array.isArray(payload.files) ? payload.files : [];
			behaviorFiles = files
				.map((name: string) => parseBehaviorFile(name))
				.sort((a: BehaviorFile, b: BehaviorFile) => {
					const genA = a.gen ?? -1;
					const genB = b.gen ?? -1;
					if (genA !== genB) return genB - genA;
					return a.name.localeCompare(b.name);
				});

			if (!behaviorFiles.some((file) => file.name === selectedTrainBehavior)) {
				const trainFile = behaviorFiles.find((file) => file.split === "train");
				selectedTrainBehavior = trainFile?.name ?? "";
			}

			if (!behaviorFiles.some((file) => file.name === selectedValBehavior)) {
				const valFile = behaviorFiles.find((file) => file.split === "val");
				selectedValBehavior = valFile?.name ?? "";
			}
		} catch (err) {
			behaviorError = err instanceof Error ? err.message : String(err);
		} finally {
			behaviorListLoading = false;
		}
	}

	async function fetchBehaviorCsv(split: "train" | "val", file: string) {
		if (!file) return;

		if (split === "train") {
			trainBehaviorLoading = true;
		} else {
			valBehaviorLoading = true;
		}

		try {
			const params = new URLSearchParams({
				dir: activeLogDir,
				file
			});
			const res = await fetch(`/api/behavior?${params.toString()}`);
			if (!res.ok) {
				throw new Error(`Failed to load ${split} behavior CSV (${res.status})`);
			}

			const text = await res.text();
			const rows = parseCsvRows<BehaviorRow>(text);
			if (split === "train") {
				trainBehaviorRows = rows;
			} else {
				valBehaviorRows = rows;
			}
		} catch (err) {
			behaviorError = err instanceof Error ? err.message : String(err);
		} finally {
			if (split === "train") {
				trainBehaviorLoading = false;
			} else {
				valBehaviorLoading = false;
			}
		}
	}

	async function fetchParquetData(split: "train" | "val") {
		if (split === "train") {
			trainDataLoading = true;
		} else {
			valDataLoading = true;
		}

		try {
			const params = new URLSearchParams({ dataset: split });
			const res = await fetch(`/api/parquet?${params.toString()}`);
			if (!res.ok) {
				throw new Error(`Failed to load ${split} parquet (${res.status})`);
			}

			const text = await res.text();
			const rows = parseCsvRows<ParquetRow>(text);
			const map = new Map<number, ParquetRow>();
			for (const row of rows) {
				const idx = toIndex(row.row_idx ?? row.index ?? row.idx);
				if (idx === null) continue;
				map.set(idx, row);
			}

			if (split === "train") {
				trainDataRows = rows;
				trainDataMap = map;
			} else {
				valDataRows = rows;
				valDataMap = map;
			}
		} catch (err) {
			behaviorError = err instanceof Error ? err.message : String(err);
		} finally {
			if (split === "train") {
				trainDataLoading = false;
			} else {
				valDataLoading = false;
			}
		}
	}

	function zoomIn() {
		if (genData.length === 0) return;
		const base = zoomWindowSize || genData.length;
		const next = Math.max(minZoomWindow, Math.floor(base / ZOOM_STEP));
		zoomWindow = next;
		zoomStart = Math.max(0, genData.length - next);
	}

	function zoomOut() {
		if (genData.length === 0) return;
		const base = zoomWindowSize || genData.length;
		const next = Math.min(genData.length, Math.ceil(base * ZOOM_STEP));
		zoomWindow = next >= genData.length ? 0 : next;
		zoomStart = next >= genData.length ? 0 : Math.max(0, genData.length - next);
	}

	function zoomReset() {
		zoomWindow = 0;
		zoomStart = 0;
	}

	onMount(fetchLogs);

	$effect(() => {
		if (mainTab !== "behavior") return;

		if (behaviorListDir !== activeLogDir) {
			behaviorListDir = activeLogDir;
			trainBehaviorRows = [];
			valBehaviorRows = [];
			lastTrainBehavior = "";
			lastValBehavior = "";
			fetchBehaviorList();
		}

		if (trainDataRows.length === 0 && !trainDataLoading) {
			fetchParquetData("train");
		}

		if (valDataRows.length === 0 && !valDataLoading) {
			fetchParquetData("val");
		}
	});

	$effect(() => {
		if (mainTab !== "behavior") return;
		if (!selectedTrainBehavior || selectedTrainBehavior === lastTrainBehavior) return;
		lastTrainBehavior = selectedTrainBehavior;
		fetchBehaviorCsv("train", selectedTrainBehavior);
	});

	$effect(() => {
		if (mainTab !== "behavior") return;
		if (!selectedValBehavior || selectedValBehavior === lastValBehavior) return;
		lastValBehavior = selectedValBehavior;
		fetchBehaviorCsv("val", selectedValBehavior);
	});

	let sourceLabel = $derived(hasEval ? "Eval" : "Train");
	let pnlKey = $derived(hasEval ? "eval_fitness_pnl" : "train_fitness_pnl");
	let pnlRealizedKey = $derived(hasEval ? "eval_pnl_realized" : "train_pnl_realized");
	let pnlTotalKey = $derived(hasEval ? "eval_pnl_total" : "train_pnl_total");
	let metricKey = $derived(hasEval ? "eval_sortino" : "train_sortino");
	let drawdownKey = $derived(hasEval ? "eval_drawdown" : "train_drawdown");
	let wMetricKey = $derived("w_sortino");
	let metricLabel = $derived("SORTINO");

	let trainBehaviorFiles = $derived(behaviorFiles.filter((file) => file.split === "train"));
	let valBehaviorFiles = $derived(behaviorFiles.filter((file) => file.split === "val"));
	let trainBehaviorDisplay = $derived.by(() =>
		buildBehaviorDisplay(trainBehaviorRows, trainDataMap, behaviorFilter, behaviorRowLimit)
	);
	let valBehaviorDisplay = $derived.by(() =>
		buildBehaviorDisplay(valBehaviorRows, valDataMap, behaviorFilter, behaviorRowLimit)
	);
	let trainChartRows = $derived.by(() =>
		behaviorRowLimit > 0 ? trainBehaviorRows.slice(0, behaviorRowLimit) : trainBehaviorRows
	);
	let valChartRows = $derived.by(() =>
		behaviorRowLimit > 0 ? valBehaviorRows.slice(0, behaviorRowLimit) : valBehaviorRows
	);
	let trainChartTitle = $derived.by(() => {
		const file = trainBehaviorFiles.find((entry) => entry.name === selectedTrainBehavior);
		return file ? `Train · ${formatBehaviorLabel(file)}` : "Train Behavior";
	});
	let valChartTitle = $derived.by(() => {
		const file = valBehaviorFiles.find((entry) => entry.name === selectedValBehavior);
		return file ? `Eval · ${formatBehaviorLabel(file)}` : "Eval Behavior";
	});

	let genData = $derived.by(() => buildGenData(genMembers, populationFilter));
	let filteredSummary = $derived.by(() => buildFilteredSummary(genData));
	let avgMetric = $derived(avgOrNull(filteredSummary.metricSum, filteredSummary.metricCount));
	let trainRealizedAvg = $derived(avgOrNull(filteredSummary.trainSum, filteredSummary.trainCount));
	let evalRealizedAvg = $derived(avgOrNull(filteredSummary.evalSum, filteredSummary.evalCount));
	let trainRealizedBest = $derived(filteredSummary.trainBest);
	let evalRealizedBest = $derived(filteredSummary.evalBest);
	let hasTrainRealized = $derived(filteredSummary.trainCount > 0);
	let hasEvalRealized = $derived(filteredSummary.evalCount > 0);

	let zoomWindowSize = $derived(
		genData.length === 0
			? 0
			: zoomWindow <= 0
				? genData.length
				: Math.min(zoomWindow, genData.length)
	);
	let minZoomWindow = $derived(
		genData.length === 0 ? 0 : Math.min(MIN_ZOOM_WINDOW, genData.length)
	);
	let maxZoomStart = $derived(
		genData.length === 0 ? 0 : Math.max(0, genData.length - zoomWindowSize)
	);
	let zoomRangeLabel = $derived.by(() => buildZoomRangeLabel(genData, zoomWindowSize, zoomStart));
	let canZoomIn = $derived(genData.length > 0 && zoomWindowSize > minZoomWindow);
	let canZoomOut = $derived(genData.length > 0 && zoomWindowSize < genData.length);
	let visibleGenData = $derived.by(() => {
		if (genData.length === 0) return [];
		if (zoomWindowSize >= genData.length) return genData;
		const startIndex = Math.min(zoomStart, maxZoomStart);
		return genData.slice(startIndex, startIndex + zoomWindowSize);
	});

	let activeChartMeta = $derived.by(() =>
		buildActiveChartMeta(chartTab, sourceLabel, metricLabel)
	);
	let plateauMetricLabel = $derived.by(() =>
		buildPlateauMetricLabel(hasEvalRealized, hasTrainRealized)
	);
	let plateauResult = $derived.by(() =>
		buildPlateauResult(
			genData,
			plateauWindow,
			plateauMinDelta,
			hasEvalRealized,
			hasTrainRealized
		)
	);

	let fitnessChartData = $derived.by(() => buildFitnessChartData(visibleGenData));
	let pnlChartData = $derived.by(() =>
		buildPnlChartData(visibleGenData, sourceLabel, metricLabel)
	);
	let drawdownChartData = $derived.by(() =>
		buildDrawdownChartData(visibleGenData, sourceLabel)
	);
	let frontierChartData = $derived.by(() =>
		buildFrontierChartData(visibleGenData, genMembers, populationFilter)
	);
	let frontierOptions = $derived.by(() => buildFrontierOptions(sourceLabel));
	let realizedChartData = $derived.by(() =>
		buildRealizedChartData(visibleGenData, hasTrainRealized, hasEvalRealized)
	);
	let issues = $derived.by(() =>
		buildIssues(logs.length, issueState, metricLabel, sourceLabel, genData)
	);
	let recentGenerations = $derived.by(() => genData.slice(-3).reverse());

	let maxLogPage = $derived(Math.max(1, Math.ceil(logs.length / PAGE_SIZE)));
	let pagedLogs = $derived.by(() => buildPagedLogs(logs, logPage));

	$effect(() => {
		if (logPage > maxLogPage) {
			logPage = maxLogPage;
		}
	});

	$effect(() => {
		if (genData.length === 0) {
			zoomStart = 0;
			return;
		}

		if (zoomWindow > genData.length) {
			zoomWindow = genData.length;
		}

		const maxStart = Math.max(0, genData.length - zoomWindowSize);
		if (zoomWindowSize >= genData.length) {
			zoomStart = 0;
		} else if (zoomStart > maxStart) {
			zoomStart = maxStart;
		} else if (zoomStart < 0) {
			zoomStart = 0;
		}
	});
</script>

<div class="mx-auto max-w-[1600px] space-y-8 p-8">
	<GaPageHeader bind:logDir {loading} onReload={fetchLogs} />

	{#if error}
		<Alert.Root variant="destructive">
			<AlertCircle class="h-4 w-4" />
			<Alert.Title>Error Loading Logs</Alert.Title>
			<Alert.Description>{error}</Alert.Description>
		</Alert.Root>
	{/if}

	{#if loading}
		<div class="flex h-[60vh] flex-col items-center justify-center gap-4">
			<div class="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
			<p class="font-medium text-muted-foreground animate-pulse">
				Processing {logs.length || "thousands of"} individuals...
			</p>
		</div>
	{:else}
		<GaSummaryCards
			generationCount={genData.length}
			{bestFitness}
			{metricLabel}
			{avgMetric}
			logsCount={logs.length}
			{plateauResult}
			{plateauMetricLabel}
			bind:plateauWindow
			bind:plateauMinDelta
		/>

		<Tabs.Root bind:value={mainTab} class="w-full">
			<Tabs.List class="mb-6 grid w-full grid-cols-5 lg:w-[760px]">
				<Tabs.Trigger value="overview">Overview</Tabs.Trigger>
				<Tabs.Trigger value="evolution">Evolution</Tabs.Trigger>
				<Tabs.Trigger value="issues" class="flex items-center gap-2">
					Issues
					{#if issues.length > 0}
						<Badge variant="destructive" class="ml-1 px-1.5 py-0 text-[10px]">
							{issues.length}
						</Badge>
					{/if}
				</Tabs.Trigger>
				<Tabs.Trigger value="behavior">Behavior</Tabs.Trigger>
				<Tabs.Trigger value="data">Full Log</Tabs.Trigger>
			</Tabs.List>

			<Tabs.Content value="overview" class="space-y-6">
				<GaOverviewTab
					{activeChartMeta}
					{sourceLabel}
					{metricLabel}
					bind:chartTab
					bind:populationFilter
					generationCount={genData.length}
					{zoomRangeLabel}
					{zoomWindowSize}
					{maxZoomStart}
					bind:zoomStart
					{canZoomIn}
					{canZoomOut}
					{fitnessChartData}
					{pnlChartData}
					{realizedChartData}
					{drawdownChartData}
					{frontierChartData}
					{frontierOptions}
					{trainRealizedAvg}
					{trainRealizedBest}
					{evalRealizedAvg}
					{evalRealizedBest}
					{recentGenerations}
					onZoomIn={zoomIn}
					onZoomOut={zoomOut}
					onZoomReset={zoomReset}
				/>
			</Tabs.Content>

			<Tabs.Content value="evolution">
				<GaEvolutionTab {genData} {sourceLabel} {metricLabel} />
			</Tabs.Content>

			<Tabs.Content value="issues" class="space-y-4">
				<GaIssuesTab {issues} />
			</Tabs.Content>

			<Tabs.Content value="behavior" class="space-y-6">
				<GaBehaviorTab
					{activeLogDir}
					{behaviorListLoading}
					{behaviorError}
					{trainBehaviorFiles}
					{valBehaviorFiles}
					bind:selectedTrainBehavior
					bind:selectedValBehavior
					{trainBehaviorLoading}
					{valBehaviorLoading}
					{trainDataLoading}
					{valDataLoading}
					trainBehaviorRowCount={trainBehaviorRows.length}
					valBehaviorRowCount={valBehaviorRows.length}
					trainDataRowCount={trainDataRows.length}
					valDataRowCount={valDataRows.length}
					bind:behaviorFilter
					bind:behaviorRowLimit
					{trainBehaviorDisplay}
					{valBehaviorDisplay}
					onRefresh={fetchBehaviorList}
					onLoadTrain={() => fetchBehaviorCsv("train", selectedTrainBehavior)}
					onLoadVal={() => fetchBehaviorCsv("val", selectedValBehavior)}
					onReloadParquet={() => {
						fetchParquetData("train");
						fetchParquetData("val");
					}}
					onOpenTrainChart={() => (trainChartOpen = true)}
					onOpenValChart={() => (valChartOpen = true)}
				/>
			</Tabs.Content>

			<Tabs.Content value="data">
				<GaDataTab
					{logs}
					{pagedLogs}
					{sourceLabel}
					{metricLabel}
					{pnlKey}
					{pnlRealizedKey}
					{pnlTotalKey}
					{metricKey}
					{drawdownKey}
					{wMetricKey}
					bind:logPage
					{maxLogPage}
				/>
			</Tabs.Content>
		</Tabs.Root>
	{/if}
</div>

<BehaviorCandlestickModal
	open={trainChartOpen}
	title={trainChartTitle}
	rows={trainChartRows}
	onClose={() => (trainChartOpen = false)}
/>
<BehaviorCandlestickModal
	open={valChartOpen}
	title={valChartTitle}
	rows={valChartRows}
	onClose={() => (valChartOpen = false)}
/>

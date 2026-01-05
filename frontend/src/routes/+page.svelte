<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import { Input } from "$lib/components/ui/input";
    import * as Tabs from "$lib/components/ui/tabs";
    import * as Alert from "$lib/components/ui/alert";
    import GaChart from "$lib/components/GaChart.svelte";
    import { AlertCircle, TrendingUp, Info, ListFilter, Bug } from "lucide-svelte";

	type LogRow = Record<string, any>;

	type GenMember = {
		fitness: number | null;
		pnl: number | null;
		realized: number | null;
		total: number | null;
		metric: number | null;
		drawdown: number | null;
		trainRealized: number | null;
		evalRealized: number | null;
	};

	type IssueState = {
		highFitnessCount: number;
		highFitnessItems: string[];
		cappedCount: number;
		cappedItems: string[];
		zeroDdCount: number;
		zeroDdItems: string[];
		retCount: number;
		retNegativeCount: number;
	};

	type KeySet = {
		pnlKey: string;
		pnlRealizedKey: string;
		pnlTotalKey: string;
		metricKey: string;
		drawdownKey: string;
		retKey: string;
	};

	const TRAIN_KEYS: KeySet = {
		pnlKey: 'train_fitness_pnl',
		pnlRealizedKey: 'train_pnl_realized',
		pnlTotalKey: 'train_pnl_total',
		metricKey: 'train_sortino',
		drawdownKey: 'train_drawdown',
		retKey: 'train_ret_mean'
	};

	const EVAL_KEYS: KeySet = {
		pnlKey: 'eval_fitness_pnl',
		pnlRealizedKey: 'eval_pnl_realized',
		pnlTotalKey: 'eval_pnl_total',
		metricKey: 'eval_sortino',
		drawdownKey: 'eval_drawdown',
		retKey: 'eval_ret_mean'
	};

	const EVAL_PROBE_KEYS = [
		'eval_fitness_pnl',
		'eval_pnl_realized',
		'eval_pnl_total',
		'eval_sortino',
		'eval_drawdown',
		'eval_ret_mean'
	] as const;

	const logChunk = 1000;
	const pageSize = 200;
	const MIN_ZOOM_WINDOW = 50;
	const ZOOM_STEP = 1.5;
	type ChartTab = 'fitness' | 'performance' | 'realized' | 'drawdown' | 'frontier';
	type PopulationFilter = 'all' | 'top5' | 'top10' | 'top20p';

	let logs: LogRow[] = $state([]);
	let loading = $state(true);
    let error = $state("");
    let loadingMore = $state(false);
    let doneLoading = $state(false);
    let nextOffset = $state(0);
    let logPage = $state(1);
	let logDir = $state("runs_ga");
	let activeLogDir = $state("runs_ga");
	let chartTab = $state<ChartTab>('fitness');
	let populationFilter = $state<PopulationFilter>('all');
	let zoomWindow = $state(0);
	let zoomStart = $state(0);
	let plateauWindow = $state(300);
	let plateauMinDelta = $state(0.5);
	let hasEval = $state(false);
	let bestFitness = $state(Number.NEGATIVE_INFINITY);
	let genMembers: Map<number, GenMember[]> = $state(new Map());
	let issueState: IssueState = $state({
		highFitnessCount: 0,
		highFitnessItems: [],
		cappedCount: 0,
		cappedItems: [],
		zeroDdCount: 0,
		zeroDdItems: [],
		retCount: 0,
		retNegativeCount: 0
	});

	let loadToken = 0;
	let hasEvalDetermined = false;

    const toNumber = (value: unknown): number | null => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};
	const formatNum = (value: unknown, digits = 4) => {
		const num = toNumber(value);
		return num === null ? '—' : num.toFixed(digits);
	};
	const avgOrNull = (sum: number, count: number) => count > 0 ? sum / count : null;
	const finiteValues = (values: Array<number | null>) =>
		values.filter((value): value is number => value !== null && Number.isFinite(value));
	const meanValue = (values: number[]) => values.length ? values.reduce((acc, v) => acc + v, 0) / values.length : 0;
	const maxValue = (values: number[]) => values.length ? Math.max(...values) : 0;
	const minValue = (values: number[]) => values.length ? Math.min(...values) : 0;
	const normalizeLogDir = (value: string) => value.trim() || "runs_ga";
	const buildLogsUrl = (offset: number, dir: string) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir
		});
		return `/api/logs?${params.toString()}`;
	};
	const percentileValue = (sortedValues: number[], percentile: number) => {
		if (sortedValues.length === 0) return null;
		const index = (sortedValues.length - 1) * percentile;
		const lower = Math.floor(index);
		const upper = Math.ceil(index);
		if (lower === upper) return sortedValues[lower];
		const weight = index - lower;
		return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
	};
	const selectMembers = (members: GenMember[]) => {
		if (populationFilter === 'all') return members;
		const ranked = members
			.filter((member) => member.fitness !== null)
			.sort((a, b) => (b.fitness ?? Number.NEGATIVE_INFINITY) - (a.fitness ?? Number.NEGATIVE_INFINITY));
		if (ranked.length === 0) return members;
		let limit = ranked.length;
		if (populationFilter === 'top5') limit = Math.min(5, ranked.length);
		if (populationFilter === 'top10') limit = Math.min(10, ranked.length);
		if (populationFilter === 'top20p') limit = Math.max(1, Math.round(ranked.length * 0.2));
		return ranked.slice(0, limit);
	};

	function resetState() {
		logs = [];
		genMembers = new Map();
		issueState = {
			highFitnessCount: 0,
			highFitnessItems: [],
			cappedCount: 0,
			cappedItems: [],
			zeroDdCount: 0,
			zeroDdItems: [],
			retCount: 0,
			retNegativeCount: 0
		};
		bestFitness = Number.NEGATIVE_INFINITY;
		logPage = 1;
		hasEval = false;
		hasEvalDetermined = false;
		zoomWindow = 0;
		zoomStart = 0;
	}

	function detectEval(rows: LogRow[]) {
		if (hasEvalDetermined || rows.length === 0) return;
		const first = rows[0];
		if (first && typeof first === 'object') {
			hasEval = EVAL_PROBE_KEYS.some((key) => key in first);
		} else {
			hasEval = false;
		}
		hasEvalDetermined = true;
	}

	const formatIssueId = (row: LogRow) => {
		const genLabel = row.gen ?? '—';
		const idxLabel = row.idx ?? '—';
		return `Gen ${genLabel}, Idx ${idxLabel}`;
	};

	function updateFromChunk(rows: LogRow[]) {
		if (!rows || rows.length === 0) return;

		detectEval(rows);
		const keys = hasEval ? EVAL_KEYS : TRAIN_KEYS;
		const membersMap = new Map(genMembers);

		let chunkBestFitness = bestFitness;
		const nextIssue: IssueState = {
			highFitnessCount: issueState.highFitnessCount,
			highFitnessItems: [...issueState.highFitnessItems],
			cappedCount: issueState.cappedCount,
			cappedItems: [...issueState.cappedItems],
			zeroDdCount: issueState.zeroDdCount,
			zeroDdItems: [...issueState.zeroDdItems],
			retCount: issueState.retCount,
			retNegativeCount: issueState.retNegativeCount
		};

		for (const row of rows) {
			if (!row || typeof row !== 'object') continue;

			const fitness = toNumber(row.fitness);
			const pnl = toNumber(row[keys.pnlKey]);
			const realized = toNumber(row[keys.pnlRealizedKey]);
			const trainRealized = toNumber(row.train_pnl_realized);
			const evalRealized = toNumber(row.eval_pnl_realized);
			const total = toNumber(row[keys.pnlTotalKey]);
			const metric = toNumber(row[keys.metricKey]);
			const drawdown = toNumber(row[keys.drawdownKey]);
			const ret = toNumber(row[keys.retKey]);
			const genValue = toNumber(row.gen);

			if (genValue !== null) {
				const member: GenMember = {
					fitness,
					pnl,
					realized,
					total,
					metric,
					drawdown,
					trainRealized,
					evalRealized
				};
				const list = membersMap.get(genValue);
				if (list) {
					list.push(member);
				} else {
					membersMap.set(genValue, [member]);
				}
			}

			if (fitness !== null) {
				chunkBestFitness = Math.max(chunkBestFitness, fitness);
			}

			const issueId = formatIssueId(row);
			if (fitness !== null && fitness > 1000) {
				nextIssue.highFitnessCount += 1;
				if (nextIssue.highFitnessItems.length < 5) {
					nextIssue.highFitnessItems.push(`${issueId}: ${fitness.toFixed(2)}`);
				}
			}
			if (metric !== null && metric >= 50.0) {
				nextIssue.cappedCount += 1;
				if (nextIssue.cappedItems.length < 5) {
					nextIssue.cappedItems.push(issueId);
				}
			}
			if (drawdown !== null && pnl !== null && drawdown === 0 && pnl !== 0) {
				nextIssue.zeroDdCount += 1;
				if (nextIssue.zeroDdItems.length < 5) {
					nextIssue.zeroDdItems.push(issueId);
				}
			}
			if (ret !== null) {
				nextIssue.retCount += 1;
				if (ret < 0) nextIssue.retNegativeCount += 1;
			}
		}

		logs.push(...rows);
		genMembers = membersMap;
		bestFitness = chunkBestFitness;
		issueState = nextIssue;
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
		} catch (e) {
			console.error('Failed to fetch logs', e);
            error = e instanceof Error ? e.message : String(e);
		} finally {
			if (token === loadToken) {
				loading = false;
			}
		}
	}

	onMount(fetchLogs);

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
                    nextOffset = payload.nextOffset || (nextOffset + payload.data.length);
                } else if (typeof payload.nextOffset === 'number' && payload.nextOffset !== nextOffset) {
                    nextOffset = payload.nextOffset;
                } else if (payload.done) {
                    doneLoading = true;
                }
                if (payload.done) {
                    doneLoading = true;
                }
            } catch (e) {
                console.error('Failed to fetch more logs', e);
                doneLoading = true;
            } finally {
                loadingMore = false;
                if (!doneLoading) {
                    scheduleLoadMore(token, dir);
                }
            }
        }, 60);
    }

    // Dynamic key detection
    let sourceLabel = $derived(hasEval ? 'Eval' : 'Train');
    let pnlKey = $derived(hasEval ? 'eval_fitness_pnl' : 'train_fitness_pnl');
    let pnlRealizedKey = $derived(hasEval ? 'eval_pnl_realized' : 'train_pnl_realized');
    let pnlTotalKey = $derived(hasEval ? 'eval_pnl_total' : 'train_pnl_total');
    let metricKey = $derived(hasEval ? 'eval_sortino' : 'train_sortino');
    let drawdownKey = $derived(hasEval ? 'eval_drawdown' : 'train_drawdown');
    let retKey = $derived(hasEval ? 'eval_ret_mean' : 'train_ret_mean');
    let wMetricKey = $derived('w_sortino');
    let metricLabel = $derived('SORTINO');

    // Data Aggregation by Generation
    let genData = $derived.by(() => {
        if (genMembers.size === 0) return [];

        return Array.from(genMembers.entries())
			.sort(([genA], [genB]) => genA - genB)
			.map(([gen, members]) => {
				const selected = selectMembers(members);
				const fitnessValues = finiteValues(selected.map((m) => m.fitness));
				const pnlValues = finiteValues(selected.map((m) => m.pnl));
				const realizedValues = finiteValues(selected.map((m) => m.realized));
				const totalValues = finiteValues(selected.map((m) => m.total));
				const metricValues = finiteValues(selected.map((m) => m.metric));
				const drawdownValues = finiteValues(selected.map((m) => m.drawdown));
				const trainRealizedValues = finiteValues(selected.map((m) => m.trainRealized));
				const evalRealizedValues = finiteValues(selected.map((m) => m.evalRealized));
				const sortedFitness = [...fitnessValues].sort((a, b) => a - b);
				const metricSum = metricValues.reduce((acc, v) => acc + v, 0);
				const trainRealizedSum = trainRealizedValues.reduce((acc, v) => acc + v, 0);
				const evalRealizedSum = evalRealizedValues.reduce((acc, v) => acc + v, 0);

				return {
					gen,
					population: members.length,
					count: selected.length,
					bestFitness: maxValue(fitnessValues),
					avgFitness: meanValue(fitnessValues),
					p50Fitness: percentileValue(sortedFitness, 0.5),
					p90Fitness: percentileValue(sortedFitness, 0.9),
					bestPnl: maxValue(pnlValues),
					avgPnl: meanValue(pnlValues),
					bestRealizedPnl: maxValue(realizedValues),
					avgRealizedPnl: meanValue(realizedValues),
					bestTrainRealized: maxValue(trainRealizedValues),
					avgTrainRealized: meanValue(trainRealizedValues),
					bestEvalRealized: maxValue(evalRealizedValues),
					avgEvalRealized: meanValue(evalRealizedValues),
					bestTotalPnl: maxValue(totalValues),
					avgTotalPnl: meanValue(totalValues),
					bestMetric: maxValue(metricValues),
					avgMetric: meanValue(metricValues),
					metricSum,
					metricCount: metricValues.length,
					trainRealizedSum,
					trainRealizedCount: trainRealizedValues.length,
					evalRealizedSum,
					evalRealizedCount: evalRealizedValues.length,
					maxDrawdown: maxValue(drawdownValues),
					minDrawdown: minValue(drawdownValues),
					avgDrawdown: meanValue(drawdownValues)
				};
			});
    });

	let filteredSummary = $derived.by(() => {
		let metricSum = 0;
		let metricCount = 0;
		let trainSum = 0;
		let trainCount = 0;
		let trainBest = Number.NEGATIVE_INFINITY;
		let evalSum = 0;
		let evalCount = 0;
		let evalBest = Number.NEGATIVE_INFINITY;

		for (const g of genData) {
			metricSum += g.metricSum;
			metricCount += g.metricCount;
			trainSum += g.trainRealizedSum;
			trainCount += g.trainRealizedCount;
			trainBest = Math.max(trainBest, g.bestTrainRealized);
			evalSum += g.evalRealizedSum;
			evalCount += g.evalRealizedCount;
			evalBest = Math.max(evalBest, g.bestEvalRealized);
		}

		return {
			metricSum,
			metricCount,
			trainSum,
			trainCount,
			trainBest,
			evalSum,
			evalCount,
			evalBest
		};
	});

	let avgMetric = $derived(avgOrNull(filteredSummary.metricSum, filteredSummary.metricCount));
	let trainRealizedAvg = $derived(avgOrNull(filteredSummary.trainSum, filteredSummary.trainCount));
	let evalRealizedAvg = $derived(avgOrNull(filteredSummary.evalSum, filteredSummary.evalCount));
	let trainRealizedBest = $derived(filteredSummary.trainBest);
	let evalRealizedBest = $derived(filteredSummary.evalBest);
	let hasTrainRealized = $derived(filteredSummary.trainCount > 0);
	let hasEvalRealized = $derived(filteredSummary.evalCount > 0);

	let zoomWindowSize = $derived(genData.length === 0 ? 0 : (zoomWindow <= 0 ? genData.length : Math.min(zoomWindow, genData.length)));
	let minZoomWindow = $derived(genData.length === 0 ? 0 : Math.min(MIN_ZOOM_WINDOW, genData.length));
	let maxZoomStart = $derived(genData.length === 0 ? 0 : Math.max(0, genData.length - zoomWindowSize));
	let zoomRangeLabel = $derived.by(() => {
		if (genData.length === 0) return 'No data';
		if (zoomWindowSize >= genData.length) return `All ${genData.length} generations`;
		const startIndex = Math.min(zoomStart, maxZoomStart);
		const endIndex = Math.min(startIndex + zoomWindowSize - 1, genData.length - 1);
		const startGen = genData[startIndex]?.gen ?? '—';
		const endGen = genData[endIndex]?.gen ?? '—';
		return `Gen ${startGen} to Gen ${endGen}`;
	});
	let canZoomIn = $derived(genData.length > 0 && zoomWindowSize > minZoomWindow);
	let canZoomOut = $derived(genData.length > 0 && zoomWindowSize < genData.length);

	let visibleGenData = $derived.by(() => {
		if (genData.length === 0) return [];
		if (zoomWindowSize >= genData.length) return genData;
		const startIndex = Math.min(zoomStart, maxZoomStart);
		return genData.slice(startIndex, startIndex + zoomWindowSize);
	});

	const preferredRealized = (member: GenMember) => {
		if (member.evalRealized !== null) return member.evalRealized;
		if (member.trainRealized !== null) return member.trainRealized;
		return member.realized;
	};

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

	let activeChartMeta = $derived.by(() => {
		if (chartTab === 'fitness') {
			return {
				title: 'Fitness Evolution',
				description: 'Maximum and average fitness progress per generation'
			};
		}
		if (chartTab === 'performance') {
			return {
				title: 'Performance Evolution',
				description: `Best ${sourceLabel} PNL, realized PNL, and ${metricLabel} across windows`
			};
		}
		if (chartTab === 'drawdown') {
			return {
				title: 'Drawdown Monitoring',
				description: 'Worst and average drawdown per generation'
			};
		}
		if (chartTab === 'frontier') {
			return {
				title: 'Risk/Return Frontier',
				description: 'Best realized PNL plotted against max drawdown'
			};
		}
		return {
			title: 'Realized PNL Focus',
			description: 'Best realized PNL for training and eval tracked independently'
		};
	});

	let plateauMetricLabel = $derived.by(() => {
		if (hasEvalRealized) return 'Best Eval Realized PNL';
		if (hasTrainRealized) return 'Best Train Realized PNL';
		return 'Best Fitness';
	});

	let plateauResult = $derived.by(() => {
		if (genData.length === 0) {
			return { plateauGen: null, lastImprovementGen: null };
		}
		const windowSize = Math.max(1, Math.round(plateauWindow));
		const minDelta = Math.max(0, plateauMinDelta);
		let best = Number.NEGATIVE_INFINITY;
		let lastImprovementIndex: number | null = null;
		let plateauIndex: number | null = null;

		const pickValue = (g) => {
			if (hasEvalRealized) return g.bestEvalRealized;
			if (hasTrainRealized) return g.bestTrainRealized;
			return g.bestFitness;
		};

		for (let i = 0; i < genData.length; i += 1) {
			const value = pickValue(genData[i]);
			if (!Number.isFinite(value)) continue;
			if (best === Number.NEGATIVE_INFINITY) {
				best = value;
				lastImprovementIndex = i;
				continue;
			}
			if (value > best + minDelta) {
				best = value;
				lastImprovementIndex = i;
			}
			if (lastImprovementIndex !== null && i - lastImprovementIndex >= windowSize) {
				plateauIndex = i;
				break;
			}
		}

		return {
			plateauGen: plateauIndex !== null ? genData[plateauIndex]?.gen ?? null : null,
			lastImprovementGen: lastImprovementIndex !== null ? genData[lastImprovementIndex]?.gen ?? null : null
		};
	});

    // Chart: Fitness Evolution
    let fitnessChartData = $derived({
        labels: visibleGenData.map(g => `Gen ${g.gen}`),
        datasets: [
            {
                label: 'Best Fitness',
                data: visibleGenData.map(g => g.bestFitness),
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                tension: 0.1
            },
            {
                label: 'Avg Fitness',
                data: visibleGenData.map(g => g.avgFitness),
                borderColor: 'rgb(147, 197, 253)',
                backgroundColor: 'rgba(147, 197, 253, 0.2)',
                borderDash: [5, 5],
                tension: 0.1
            },
            {
                label: 'P90 Fitness',
                data: visibleGenData.map(g => g.p90Fitness),
                borderColor: 'rgb(14, 116, 144)',
                backgroundColor: 'rgba(14, 116, 144, 0.15)',
                borderDash: [4, 4],
                tension: 0.1
            },
            {
                label: 'Median Fitness',
                data: visibleGenData.map(g => g.p50Fitness),
                borderColor: 'rgb(56, 189, 248)',
                backgroundColor: 'rgba(56, 189, 248, 0.12)',
                borderDash: [2, 6],
                tension: 0.1
            }
        ]
    });

    // Chart: PNL Evolution
    let pnlChartData = $derived({
        labels: visibleGenData.map(g => `Gen ${g.gen}`),
        datasets: [
            {
                label: `Best ${sourceLabel} Fitness PNL`,
                data: visibleGenData.map(g => g.bestPnl),
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.5)',
                tension: 0.1
            },
            {
                label: `Best ${sourceLabel} Realized PNL`,
                data: visibleGenData.map(g => g.bestRealizedPnl),
                borderColor: 'rgb(16, 185, 129)',
                backgroundColor: 'rgba(16, 185, 129, 0.35)',
                borderDash: [4, 4],
                tension: 0.1
            },
            {
                label: `Best ${metricLabel}`,
                data: visibleGenData.map(g => g.bestMetric),
                borderColor: 'rgb(168, 85, 247)',
                backgroundColor: 'rgba(168, 85, 247, 0.5)',
                yAxisID: 'y1',
                tension: 0.1
            }
        ]
    });

	let drawdownChartData = $derived({
		labels: visibleGenData.map(g => `Gen ${g.gen}`),
		datasets: [
			{
				label: `Worst ${sourceLabel} Drawdown`,
				data: visibleGenData.map(g => g.maxDrawdown),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.3)',
				tension: 0.1
			},
			{
				label: `Avg ${sourceLabel} Drawdown`,
				data: visibleGenData.map(g => g.avgDrawdown),
				borderColor: 'rgb(251, 146, 60)',
				backgroundColor: 'rgba(251, 146, 60, 0.2)',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	});

	let frontierChartData = $derived.by(() => {
		if (visibleGenData.length === 0) return { labels: [], datasets: [] };
		const points = [];
		for (const g of visibleGenData) {
			const members = genMembers.get(g.gen);
			if (!members) continue;
			const selected = selectMembers(members);
			let bestPoint = null;
			for (const member of selected) {
				const realized = preferredRealized(member);
				const drawdown = member.drawdown;
				if (realized === null || drawdown === null) continue;
				if (!bestPoint || realized > bestPoint.y) {
					bestPoint = { x: drawdown, y: realized, gen: g.gen };
				}
			}
			if (bestPoint) points.push(bestPoint);
		}

		return {
			labels: [],
			datasets: [
				{
					label: 'Best Realized vs Drawdown',
					data: points,
					borderColor: 'rgb(59, 130, 246)',
					backgroundColor: 'rgba(59, 130, 246, 0.5)',
					showLine: false,
					pointRadius: 3
				}
			]
		};
	});

	const frontierOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear',
				title: {
					display: true,
					text: `Max ${sourceLabel} Drawdown (%)`
				}
			},
			y: {
				title: {
					display: true,
					text: 'Realized PNL'
				}
			}
		}
	}));

	let realizedChartData = $derived.by(() => {
		const labels = visibleGenData.map(g => `Gen ${g.gen}`);
		const datasets = [];

		if (hasTrainRealized) {
			datasets.push({
				label: 'Best Train Realized PNL',
				data: visibleGenData.map(g => g.bestTrainRealized),
				borderColor: 'rgb(14, 116, 144)',
				backgroundColor: 'rgba(14, 116, 144, 0.35)',
				tension: 0.1
			});
		}

		if (hasEvalRealized) {
			datasets.push({
				label: 'Best Eval Realized PNL',
				data: visibleGenData.map(g => g.bestEvalRealized),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.3)',
				tension: 0.1
			});
		}

		return { labels, datasets };
	});

    // Issues Detection
    let issues = $derived.by(() => {
        const list = [];
        if (logs.length === 0) return list;

        // 1. Fitness Outliers
        if (issueState.highFitnessCount > 0) {
            list.push({
                type: 'warning',
                title: 'Fitness Outliers Detected',
                message: `${issueState.highFitnessCount} individuals have fitness > 1000. This may indicate logic errors or extreme lucky outliers corrupting selection.`,
                items: issueState.highFitnessItems
            });
        }

        // 2. Metric Capping
        if (issueState.cappedCount > 0) {
            list.push({
                type: 'info',
                title: `${metricLabel} Cap Hit`,
                message: `${issueState.cappedCount} individuals hit the cap of 50.0 for ${metricLabel} (${sourceLabel}). Consider raising the cap if they are consistently flatlining at max.`,
                items: issueState.cappedItems
            });
        }

        // 3. Zero Drawdown
        if (issueState.zeroDdCount > 0) {
            list.push({
                type: 'warning',
                title: 'Zero Drawdown (Possible Fake Results)',
                message: `${issueState.zeroDdCount} individuals have 0% drawdown but non-zero PNL. This often indicates insufficient evaluation data or "one-hit wonder" trades.`,
                items: issueState.zeroDdItems
            });
        }

        // 4. Negative Returns
        if (issueState.retCount > 0 && issueState.retNegativeCount === issueState.retCount) {
            list.push({
                type: 'destructive',
                title: 'Universal Negative Returns',
                message: `Every single individual in the log has a negative return mean (${sourceLabel}). The strategy logic or features might be fundamentally flawed.`
            });
        }

        // 5. Inconsistent Population
        const counts = genData.map(g => g.population);
        const uniqueCounts = new Set(counts);
        if (uniqueCounts.size > 1) {
            list.push({
                type: 'info',
                title: 'Inconsistent Population Sizes',
                message: `Population sizes vary across generations: ${Array.from(uniqueCounts).join(', ')}. Ensure this is intentional (e.g. elite pruning).`
            });
        }

        return list;
    });

	let maxLogPage = $derived(Math.max(1, Math.ceil(logs.length / pageSize)));

	let pagedLogs = $derived.by(() => {
		if (logs.length === 0) return [];
		const total = logs.length;
		const end = total - (logPage - 1) * pageSize;
		const start = Math.max(0, end - pageSize);
		return logs.slice(start, end).reverse();
	});

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

<div class="p-8 space-y-8 max-w-[1600px] mx-auto">
	<div class="flex flex-wrap justify-between items-center gap-4 bg-card p-6 rounded-xl border shadow-sm">
		<div>
            <h1 class="text-4xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">Midas Dashboard</h1>
            <p class="text-muted-foreground mt-1">Analyzing Training Performance & Evolution</p>
        </div>
        <div class="flex flex-wrap items-center gap-3 justify-end">
            <div class="flex items-center gap-2">
                <span class="text-xs text-muted-foreground">Log folder</span>
                <Input
                    class="h-9 w-48"
                    placeholder="runs_ga"
                    bind:value={logDir}
                    aria-label="Log folder for charts"
                />
            </div>
		    <button onclick={fetchLogs} class="px-5 py-2.5 bg-secondary text-secondary-foreground rounded-lg font-semibold hover:bg-secondary/80 transition-all flex items-center gap-2 border">
                <ListFilter size={18} /> Load Logs
            </button>
            <a href="/train" class="px-5 py-2.5 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-all flex items-center gap-2 shadow-sm">
                <TrendingUp size={18} /> New Run
            </a>
        </div>
	</div>

    {#if error}
        <Alert.Root variant="destructive">
            <AlertCircle class="h-4 w-4" />
            <Alert.Title>Error Loading Logs</Alert.Title>
            <Alert.Description>{error}</Alert.Description>
        </Alert.Root>
    {/if}

	{#if loading}
		<div class="flex flex-col justify-center items-center h-[60vh] gap-4">
			<div class="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
			<p class="text-muted-foreground font-medium animate-pulse">Processing {logs.length || "thousands of"} individuals...</p>
		</div>
	{:else}
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
			<Card.Root class="border-l-4 border-l-blue-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Generations</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{genData.length}</p>
				</Card.Content>
			</Card.Root>
			<Card.Root class="border-l-4 border-l-green-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Global Best Fitness</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold text-green-500">{formatNum(bestFitness, 2)}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root class="border-l-4 border-l-purple-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Avg {metricLabel}</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{formatNum(avgMetric, 2)}</p>
				</Card.Content>
			</Card.Root>
			<Card.Root class="border-l-4 border-l-orange-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Total Processed</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">{logs.length.toLocaleString()}</p>
				</Card.Content>
			</Card.Root>
            <Card.Root class="border-l-4 border-l-slate-500">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Plateau Gen</Card.Title>
				</Card.Header>
				<Card.Content>
					<p class="text-3xl font-bold">
						{plateauResult.plateauGen ? `Gen ${plateauResult.plateauGen}` : 'Not reached'}
					</p>
					<p class="text-xs text-muted-foreground mt-1">
						Metric: {plateauMetricLabel}
					</p>
					<p class="text-xs text-muted-foreground">
						Last improvement: {plateauResult.lastImprovementGen ? `Gen ${plateauResult.lastImprovementGen}` : '—'}
					</p>
					<div class="flex items-center gap-2 mt-2 text-[10px] text-muted-foreground">
						<span>Window</span>
						<input
							type="number"
							min="50"
							step="50"
							bind:value={plateauWindow}
							class="w-16 px-1 py-0.5 border rounded-md bg-background text-xs"
						/>
						<span>Delta</span>
						<input
							type="number"
							min="0"
							step="0.1"
							bind:value={plateauMinDelta}
							class="w-16 px-1 py-0.5 border rounded-md bg-background text-xs"
						/>
					</div>
				</Card.Content>
			</Card.Root>
		</div>

        <Tabs.Root value="overview" class="w-full">
            <Tabs.List class="grid w-full grid-cols-4 lg:w-[600px] mb-6">
                <Tabs.Trigger value="overview">Overview</Tabs.Trigger>
                <Tabs.Trigger value="evolution">Evolution</Tabs.Trigger>
                <Tabs.Trigger value="issues" class="flex gap-2 items-center">
                    Issues {#if issues.length > 0}<Badge variant="destructive" class="ml-1 px-1.5 py-0 text-[10px]">{issues.length}</Badge>{/if}
                </Tabs.Trigger>
                <Tabs.Trigger value="data">Full Log</Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="overview" class="space-y-6">
                <Card.Root>
                    <Card.Header class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <Card.Title class="flex items-center gap-2"><TrendingUp size={20} class="text-blue-500"/> {activeChartMeta.title}</Card.Title>
                            <Card.Description>{activeChartMeta.description}</Card.Description>
                        </div>
                        <div class="flex flex-col gap-2 text-xs w-full lg:w-auto lg:items-end">
                            <div class="flex flex-wrap items-center gap-2">
                                <span class="text-muted-foreground">Population</span>
                                <select
                                    bind:value={populationFilter}
                                    class="px-2 py-1 border rounded-md bg-background text-xs"
                                >
                                    <option value="all">All individuals</option>
                                    <option value="top5">Top 5 by fitness</option>
                                    <option value="top10">Top 10 by fitness</option>
                                    <option value="top20p">Top 20% by fitness</option>
                                </select>
                            </div>
                            <div class="flex flex-wrap items-center gap-2">
                                <span class="text-muted-foreground">{zoomRangeLabel}</span>
                                <button
                                    class="px-2.5 py-1 border rounded-md font-medium disabled:opacity-40"
                                    onclick={zoomIn}
                                    disabled={!canZoomIn}
                                >
                                    Zoom In
                                </button>
                                <button
                                    class="px-2.5 py-1 border rounded-md font-medium disabled:opacity-40"
                                    onclick={zoomOut}
                                    disabled={!canZoomOut}
                                >
                                    Zoom Out
                                </button>
                                <button
                                    class="px-2.5 py-1 border rounded-md font-medium disabled:opacity-40"
                                    onclick={zoomReset}
                                    disabled={genData.length === 0 || (zoomWindowSize >= genData.length && zoomStart === 0)}
                                >
                                    Reset Window
                                </button>
                            </div>
                            <div class="flex items-center gap-2 w-full lg:w-auto">
                                <span class="text-muted-foreground">Window</span>
                                <input
                                    type="range"
                                    min="0"
                                    max={maxZoomStart}
                                    step="1"
                                    bind:value={zoomStart}
                                    disabled={zoomWindowSize >= genData.length || maxZoomStart === 0}
                                    class="w-full lg:w-48 accent-primary disabled:opacity-40"
                                    aria-label="Pan window across generations"
                                />
                            </div>
                            <div class="text-[10px] text-muted-foreground">
                                Ctrl + scroll or pinch to zoom. Shift + drag to pan.
                            </div>
                        </div>
                    </Card.Header>
                    <Card.Content class="pt-0">
                        <Tabs.Root bind:value={chartTab} class="w-full">
                            <Tabs.List class="grid w-full grid-cols-2 lg:grid-cols-5 lg:w-[900px] mb-4">
                                <Tabs.Trigger value="fitness">Fitness</Tabs.Trigger>
                                <Tabs.Trigger value="performance">Performance</Tabs.Trigger>
                                <Tabs.Trigger value="realized">Realized</Tabs.Trigger>
                                <Tabs.Trigger value="drawdown">Drawdown</Tabs.Trigger>
                                <Tabs.Trigger value="frontier">Frontier</Tabs.Trigger>
                            </Tabs.List>

                            <Tabs.Content value="fitness">
                                {#if chartTab === 'fitness'}
                                    <div class="h-[65vh] min-h-[420px]">
                                        <GaChart data={fitnessChartData} />
                                    </div>
                                {/if}
                            </Tabs.Content>

                            <Tabs.Content value="performance">
                                {#if chartTab === 'performance'}
                                    <div class="h-[65vh] min-h-[420px]">
                                        <GaChart data={pnlChartData} />
                                    </div>
                                {/if}
                            </Tabs.Content>

                            <Tabs.Content value="realized">
                                {#if chartTab === 'realized'}
                                    {#if realizedChartData.datasets.length > 0}
                                        <div class="h-[65vh] min-h-[420px]">
                                            <GaChart data={realizedChartData} />
                                        </div>
                                    {:else}
                                        <div class="h-[50vh] min-h-[320px] flex items-center justify-center text-muted-foreground">
                                            No realized PNL columns found in this run.
                                        </div>
                                    {/if}
                                {/if}
                            </Tabs.Content>

                            <Tabs.Content value="drawdown">
                                {#if chartTab === 'drawdown'}
                                    <div class="h-[65vh] min-h-[420px]">
                                        <GaChart data={drawdownChartData} />
                                    </div>
                                {/if}
                            </Tabs.Content>

                            <Tabs.Content value="frontier">
                                {#if chartTab === 'frontier'}
                                    {#if frontierChartData.datasets.length > 0}
                                        <div class="h-[65vh] min-h-[420px]">
                                            <GaChart data={frontierChartData} options={frontierOptions} type="scatter" />
                                        </div>
                                    {:else}
                                        <div class="h-[50vh] min-h-[320px] flex items-center justify-center text-muted-foreground">
                                            Not enough data to build the frontier yet.
                                        </div>
                                    {/if}
                                {/if}
                            </Tabs.Content>
                        </Tabs.Root>
                    </Card.Content>
                </Card.Root>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card.Root class="border-l-4 border-l-emerald-500">
                        <Card.Header class="pb-2">
                            <Card.Title class="text-sm font-medium text-muted-foreground">Train Realized PNL (Avg)</Card.Title>
                        </Card.Header>
                        <Card.Content>
                            <p class="text-3xl font-bold text-emerald-500">{formatNum(trainRealizedAvg, 2)}</p>
                            <p class="text-xs text-muted-foreground mt-1">Best: {formatNum(trainRealizedBest, 2)}</p>
                        </Card.Content>
                    </Card.Root>
                    <Card.Root class="border-l-4 border-l-rose-500">
                        <Card.Header class="pb-2">
                            <Card.Title class="text-sm font-medium text-muted-foreground">Eval Realized PNL (Avg)</Card.Title>
                        </Card.Header>
                        <Card.Content>
                            <p class="text-3xl font-bold text-rose-500">{formatNum(evalRealizedAvg, 2)}</p>
                            <p class="text-xs text-muted-foreground mt-1">Best: {formatNum(evalRealizedBest, 2)}</p>
                        </Card.Content>
                    </Card.Root>
                </div>
                
                <Card.Root>
                    <Card.Header>
                        <Card.Title>Recent Highlights</Card.Title>
                    </Card.Header>
                    <Card.Content>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {#each genData.slice(-3).reverse() as g}
                                <div class="p-4 bg-muted/50 rounded-lg border">
                                    <h4 class="font-bold text-lg">Generation {g.gen}</h4>
                                        <div class="mt-2 space-y-1 text-sm">
                                            <div class="flex justify-between"><span>Max Fitness:</span> <span class="font-mono text-blue-500">{g.bestFitness.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max {sourceLabel} Fitness PNL:</span> <span class="font-mono text-green-500">{g.bestPnl.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max Realized PNL:</span> <span class="font-mono text-emerald-500">{g.bestRealizedPnl.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max Total PNL:</span> <span class="font-mono text-teal-500">{g.bestTotalPnl.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Max {metricLabel}:</span> <span class="font-mono text-purple-500">{g.bestMetric.toFixed(2)}</span></div>
                                        <div class="flex justify-between"><span>Worst Drawdown:</span> <span class="font-mono text-red-500">{formatNum(g.maxDrawdown, 2)}%</span></div>
                                        </div>
                                    </div>
                            {/each}
                        </div>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>

            <Tabs.Content value="evolution">
                <Card.Root>
                    <Card.Header>
                        <Card.Title>Detailed Generation Analysis</Card.Title>
                    </Card.Header>
                    <Card.Content>
                        <Table.Root>
                            <Table.Header>
                                <Table.Row>
                                    <Table.Head>Gen</Table.Head>
                                    <Table.Head>Selected</Table.Head>
                                    <Table.Head>Best Fitness</Table.Head>
                                    <Table.Head>Avg Fitness</Table.Head>
                                    <Table.Head>Best {sourceLabel} Fitness PNL</Table.Head>
                                    <Table.Head>Best Realized</Table.Head>
                                    <Table.Head>Best Total</Table.Head>
                                    <Table.Head>Worst DD</Table.Head>
                                    <Table.Head>Avg DD</Table.Head>
                                    <Table.Head>Best {metricLabel}</Table.Head>
                                </Table.Row>
                            </Table.Header>
                            <Table.Body>
                                {#each genData.slice().reverse() as g}
                                    <Table.Row>
                                        <Table.Cell class="font-bold">{g.gen}</Table.Cell>
                                        <Table.Cell>{g.count === g.population ? g.population : `${g.count}/${g.population}`}</Table.Cell>
                                        <Table.Cell class="text-blue-500 font-medium">{g.bestFitness.toFixed(4)}</Table.Cell>
                                        <Table.Cell class="text-muted-foreground">{g.avgFitness.toFixed(4)}</Table.Cell>
                                        <Table.Cell class={g.bestPnl >= 0 ? 'text-green-500' : 'text-red-500'}>{g.bestPnl.toFixed(4)}</Table.Cell>
                                        <Table.Cell class={g.bestRealizedPnl >= 0 ? 'text-emerald-500' : 'text-red-500'}>{formatNum(g.bestRealizedPnl)}</Table.Cell>
                                        <Table.Cell class={g.bestTotalPnl >= 0 ? 'text-teal-500' : 'text-red-500'}>{formatNum(g.bestTotalPnl)}</Table.Cell>
                                        <Table.Cell class="text-red-500">{formatNum(g.maxDrawdown)}%</Table.Cell>
                                        <Table.Cell class="text-muted-foreground">{formatNum(g.avgDrawdown)}%</Table.Cell>
                                        <Table.Cell class="text-purple-500">{g.bestMetric.toFixed(4)}</Table.Cell>
                                    </Table.Row>
                                {/each}
                            </Table.Body>
                        </Table.Root>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>

            <Tabs.Content value="issues" class="space-y-4">
                {#if issues.length === 0}
                    <div class="flex flex-col items-center justify-center py-20 text-muted-foreground">
                        <Info size={48} class="mb-4 opacity-20"/>
                        <p class="text-xl font-medium">No major issues detected</p>
                        <p class="text-sm">Training data looks within expected heuristic bounds.</p>
                    </div>
                {:else}
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {#each issues as issue}
                            <Card.Root class={issue.type === 'destructive' ? 'border-red-500' : issue.type === 'warning' ? 'border-amber-500' : 'border-blue-500'}>
                                <Card.Header class="flex flex-row items-center gap-4 py-4">
                                    <div class={`p-2 rounded-full ${issue.type === 'destructive' ? 'bg-red-100 text-red-600' : issue.type === 'warning' ? 'bg-amber-100 text-amber-600' : 'bg-blue-100 text-blue-600'}`}>
                                        <Bug size={24} />
                                    </div>
                                    <div>
                                        <Card.Title class="text-lg">{issue.title}</Card.Title>
                                    </div>
                                </Card.Header>
                                <Card.Content class="pb-4">
                                    <p class="text-sm text-muted-foreground">{issue.message}</p>
                                    {#if issue.items}
                                        <div class="mt-3 p-3 bg-muted rounded-md text-xs font-mono space-y-1">
                                            {#each issue.items as item}
                                                <div>• {item}</div>
                                            {/each}
                                        </div>
                                    {/if}
                                </Card.Content>
                            </Card.Root>
                        {/each}
                    </div>
                {/if}
            </Tabs.Content>

            <Tabs.Content value="data">
                <Card.Root>
                    <Card.Header class="flex flex-row items-center justify-between pb-4">
                        <div>
                            <Card.Title>Raw Training Logs</Card.Title>
                            <Card.Description>Scroll to view all {logs.length} data points</Card.Description>
                        </div>
                    </Card.Header>
                    <Card.Content>
                        <ScrollArea class="h-[600px] rounded-md border">
                            <Table.Root>
                                <Table.Header class="sticky top-0 bg-background z-10 shadow-sm border-b">
                                    <Table.Row>
                                        <Table.Head class="w-[80px]">Gen</Table.Head>
                                        <Table.Head class="w-[80px]">Idx</Table.Head>
                                        <Table.Head>Fitness</Table.Head>
                                    <Table.Head>{sourceLabel} Fitness PNL</Table.Head>
                                    <Table.Head>Realized PNL</Table.Head>
                                    <Table.Head>Total PNL</Table.Head>
                                    <Table.Head>{metricLabel}</Table.Head>
                                    <Table.Head>{sourceLabel} DD</Table.Head>
                                        <Table.Head class="hidden md:table-cell text-right">Weights (PNL/{metricLabel}/MDD)</Table.Head>
                                    </Table.Row>
                                </Table.Header>
                                <Table.Body>
                                {#each pagedLogs as log}
                                        <Table.Row class="hover:bg-muted/30 transition-colors">
                                            <Table.Cell>{log.gen}</Table.Cell>
                                            <Table.Cell>{log.idx}</Table.Cell>
                                            <Table.Cell class="font-medium text-blue-500">{log.fitness.toFixed(4)}</Table.Cell>
                                            <Table.Cell class={log[pnlKey] >= 0 ? 'text-green-500 font-medium' : 'text-red-500'}>{formatNum(log[pnlKey])}</Table.Cell>
                                            <Table.Cell class={log[pnlRealizedKey] >= 0 ? 'text-emerald-500 font-medium' : 'text-red-500'}>{formatNum(log[pnlRealizedKey])}</Table.Cell>
                                            <Table.Cell class={log[pnlTotalKey] >= 0 ? 'text-teal-500 font-medium' : 'text-red-500'}>{formatNum(log[pnlTotalKey])}</Table.Cell>
                                            <Table.Cell>{formatNum(log[metricKey])}</Table.Cell>
                                            <Table.Cell class="text-red-400">{formatNum(log[drawdownKey])}%</Table.Cell>
                                            <Table.Cell class="text-right hidden md:table-cell text-[10px] text-muted-foreground font-mono">
                                                {log.w_pnl.toFixed(2)} / {log[wMetricKey]?.toFixed(2)} / {log.w_mdd.toFixed(2)}
                                            </Table.Cell>
                                        </Table.Row>
                                    {/each}
                                </Table.Body>
                            </Table.Root>
                        </ScrollArea>
                        <div class="flex items-center justify-between mt-4">
                            <div class="text-xs text-muted-foreground">
                                {logs.length.toLocaleString()} rows loaded
                            </div>
                            <div class="flex items-center gap-2">
                                <button
                                    class="px-3 py-1 border rounded-md text-xs disabled:opacity-40"
                                    onclick={() => (logPage = Math.max(1, logPage - 1))}
                                    disabled={logPage === 1}
                                >
                                    Prev
                                </button>
                                <div class="text-xs">
                                    Page {logPage} / {maxLogPage}
                                </div>
                                <button
                                    class="px-3 py-1 border rounded-md text-xs disabled:opacity-40"
                                    onclick={() => (logPage = Math.min(maxLogPage, logPage + 1))}
                                    disabled={logPage >= maxLogPage}
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    </Card.Content>
                </Card.Root>
            </Tabs.Content>
        </Tabs.Root>
	{/if}
</div>

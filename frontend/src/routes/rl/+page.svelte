<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from "$lib/components/ui/card";
	import * as Tabs from "$lib/components/ui/tabs";
	import { Input } from "$lib/components/ui/input";
	import { Button } from "$lib/components/ui/button";
	import { Badge } from "$lib/components/ui/badge";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import GaChart from "$lib/components/GaChart.svelte";

	type ChartTab = 'fitness' | 'performance' | 'drawdown' | 'frontier' | 'loss' | 'gradients' | 'policy' | 'probe';
	type FolderEntry = { name: string; path: string; kind: "dir" | "file"; mtime?: number };
	
	type RlPoint = {
		epoch: number;
		fitness: number | null;
		trainRet: number | null;
		trainPnl: number | null;
		trainRealizedPnl: number | null;
		trainSortino: number | null;
		trainDrawdown: number | null;
		trainCommission: number | null;
		trainSlippage: number | null;
		trainBuyFrac: number | null;
		trainSellFrac: number | null;
		trainHoldFrac: number | null;
		trainRevertFrac: number | null;
		trainMeanMaxProb: number | null;
		trainEntries: number | null;
		trainExits: number | null;
		trainFlips: number | null;
		trainAvgHold: number | null;
		evalRet: number | null;
		evalPnl: number | null;
		evalRealizedPnl: number | null;
		evalSortino: number | null;
		evalDrawdown: number | null;
		evalCommission: number | null;
		evalSlippage: number | null;
		evalBuyFrac: number | null;
		evalSellFrac: number | null;
		evalHoldFrac: number | null;
		evalRevertFrac: number | null;
		evalMeanMaxProb: number | null;
		evalEntries: number | null;
		evalExits: number | null;
		evalFlips: number | null;
		evalAvgHold: number | null;
		probeRet: number | null;
		probePnl: number | null;
		probeRealizedPnl: number | null;
		probeSortino: number | null;
		probeDrawdown: number | null;
		probeCommission: number | null;
		probeSlippage: number | null;
		probeBuyFrac: number | null;
		probeSellFrac: number | null;
		probeHoldFrac: number | null;
		probeRevertFrac: number | null;
		probeMeanMaxProb: number | null;
		probeEntries: number | null;
		probeExits: number | null;
		probeFlips: number | null;
		probeAvgHold: number | null;
		policyLoss: number | null;
		valueLoss: number | null;
		policyGradNorm: number | null;
		valueGradNorm: number | null;
		approxKl: number | null;
		klDiv: number | null;
		clipFrac: number | null;
		entropy: number | null;
		perplexity: number | null;
		totalLoss: number | null;
		algorithm: 'ppo' | 'grpo' | null;
	};

	const logChunk = 1000;

	let chartTab = $state<ChartTab>('fitness');
	let logDir = $state('runs_rl');
	let activeLogDir = $state('runs_rl');
	let logMap = $state(new Map<number, RlPoint>());
	let fitnessWeights = $state({ pnl: 1.0, sortino: 1.0, mdd: 0.5 });
	let loading = $state(false);
	let loadingMore = $state(false);
	let doneLoading = $state(false);
	let nextOffset = $state(0);
	let error = $state('');
	let loadToken = 0;

	// Folder picker state
	let folderPickerOpen = $state(false);
	let folderPickerLoading = $state(false);
	let folderPickerError = $state('');
	let folderEntries = $state<FolderEntry[]>([]);
	let folderPickerToken = 0;

	const TRAIN_REALIZED_KEYS = ['train_realized_pnl', 'train_pnl_realized', 'train_realized'];
	const EVAL_REALIZED_KEYS = ['eval_realized_pnl', 'eval_pnl_realized', 'eval_realized'];

	const toNumber = (value: unknown): number | null => {
		if (typeof value === 'number') return Number.isFinite(value) ? value : null;
		if (typeof value === 'string' && value.trim() !== '') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};

	const firstNumber = (row: Record<string, unknown>, keys: string[]) => {
		for (const key of keys) {
			const value = toNumber(row[key]);
			if (value !== null) return value;
		}
		return null;
	};

	const normalizeLogDir = (value: string) => value.trim() || 'runs_rl';

	// Folder picker functions
	const loadFolders = async () => {
		const token = ++folderPickerToken;
		folderPickerLoading = true;
		folderPickerError = '';
		try {
			const res = await fetch('/api/files?dir=runs_rl');
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to load folders (${res.status})`);
			}
			const payload = await res.json();
			if (token !== folderPickerToken) return;
			
			// Filter only directories and sort by mtime (newest first)
			const entries = Array.isArray(payload.entries) ? payload.entries : [];
			const folders = entries
				.filter((e: FolderEntry) => e.kind === 'dir')
				.sort((a: FolderEntry, b: FolderEntry) => (b.mtime || 0) - (a.mtime || 0));
			folderEntries = folders;
		} catch (err) {
			if (token === folderPickerToken) {
				folderPickerError = err instanceof Error ? err.message : String(err);
				folderEntries = [];
			}
		} finally {
			if (token === folderPickerToken) {
				folderPickerLoading = false;
			}
		}
	};

	const openFolderPicker = async () => {
		folderPickerOpen = true;
		await loadFolders();
	};

	const closeFolderPicker = () => {
		folderPickerOpen = false;
		folderPickerError = '';
	};

	const selectFolder = (folder: FolderEntry) => {
		logDir = folder.path;
		closeFolderPicker();
		fetchLogs();
	};

	const buildLogsUrl = (offset: number, dir: string) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir,
			log: 'rl'
		});
		return `/api/logs?${params.toString()}`;
	};

	const updateFromChunk = (rows: Array<Record<string, unknown>>) => {
		if (!rows || rows.length === 0) return;
		const next = new Map(logMap);
		for (const row of rows) {
			if (!row || typeof row !== 'object') continue;
			const epoch = toNumber(row.epoch);
			if (epoch === null) continue;
			const valueLoss = toNumber(row.value_loss);
			const approxKl = toNumber(row.approx_kl);
			const klDiv = toNumber(row.kl_div);
			const clipFrac = toNumber(row.clip_frac);
			const trainRealizedPnl = firstNumber(row, TRAIN_REALIZED_KEYS);
			const evalRealizedPnl = firstNumber(row, EVAL_REALIZED_KEYS);
			const entropy = toNumber(row.entropy);
			const algorithmName =
				typeof row.algorithm === 'string' ? row.algorithm.trim().toLowerCase() : '';
			let algorithm: 'ppo' | 'grpo' | null = null;
			if (algorithmName === 'ppo' || algorithmName === 'grpo') {
				algorithm = algorithmName;
			} else if (valueLoss !== null) {
				algorithm = 'ppo';
			} else if (klDiv !== null) {
				algorithm = 'grpo';
			}
			const perplexity = toNumber(row.perplexity) ?? (entropy !== null ? Math.exp(entropy) : null);
			next.set(epoch, {
				epoch,
				fitness: toNumber(row.fitness),
				trainRet: toNumber(row.train_ret_mean),
				trainPnl: toNumber(row.train_pnl),
				trainRealizedPnl,
				trainSortino: toNumber(row.train_sortino),
				trainDrawdown: toNumber(row.train_drawdown),
				trainCommission: toNumber(row.train_commission),
				trainSlippage: toNumber(row.train_slippage),
				trainBuyFrac: toNumber(row.train_buy_frac),
				trainSellFrac: toNumber(row.train_sell_frac),
				trainHoldFrac: toNumber(row.train_hold_frac),
				trainRevertFrac: toNumber(row.train_revert_frac),
				trainMeanMaxProb: toNumber(row.train_mean_max_prob),
				trainEntries: toNumber(row.train_entries),
				trainExits: toNumber(row.train_exits),
				trainFlips: toNumber(row.train_flips),
				trainAvgHold: toNumber(row.train_avg_hold),
				evalRet: toNumber(row.eval_ret_mean),
				evalPnl: toNumber(row.eval_pnl),
				evalRealizedPnl,
				evalSortino: toNumber(row.eval_sortino),
				evalDrawdown: toNumber(row.eval_drawdown),
				evalCommission: toNumber(row.eval_commission),
				evalSlippage: toNumber(row.eval_slippage),
				evalBuyFrac: toNumber(row.eval_buy_frac),
				evalSellFrac: toNumber(row.eval_sell_frac),
				evalHoldFrac: toNumber(row.eval_hold_frac),
				evalRevertFrac: toNumber(row.eval_revert_frac),
				evalMeanMaxProb: toNumber(row.eval_mean_max_prob),
				evalEntries: toNumber(row.eval_entries),
				evalExits: toNumber(row.eval_exits),
				evalFlips: toNumber(row.eval_flips),
				evalAvgHold: toNumber(row.eval_avg_hold),
				probeRet: toNumber(row.probe_ret_mean),
				probePnl: toNumber(row.probe_pnl),
				probeRealizedPnl: toNumber(row.probe_realized_pnl),
				probeSortino: toNumber(row.probe_sortino),
				probeDrawdown: toNumber(row.probe_drawdown),
				probeCommission: toNumber(row.probe_commission),
				probeSlippage: toNumber(row.probe_slippage),
				probeBuyFrac: toNumber(row.probe_buy_frac),
				probeSellFrac: toNumber(row.probe_sell_frac),
				probeHoldFrac: toNumber(row.probe_hold_frac),
				probeRevertFrac: toNumber(row.probe_revert_frac),
				probeMeanMaxProb: toNumber(row.probe_mean_max_prob),
				probeEntries: toNumber(row.probe_entries),
				probeExits: toNumber(row.probe_exits),
				probeFlips: toNumber(row.probe_flips),
				probeAvgHold: toNumber(row.probe_avg_hold),
				policyLoss: toNumber(row.policy_loss),
				valueLoss,
				policyGradNorm: toNumber(row.policy_grad_norm),
				valueGradNorm: toNumber(row.value_grad_norm),
				approxKl,
				klDiv,
				clipFrac,
				entropy,
				perplexity,
				totalLoss: toNumber(row.total_loss),
				algorithm
			});
		}
		logMap = next;
	};

	function resetState() {
		logMap = new Map();
		loading = true;
		error = '';
		nextOffset = 0;
		doneLoading = false;
		loadingMore = false;
	}

	async function fetchLogs() {
		const token = ++loadToken;
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
			updateFromChunk(Array.isArray(payload.data) ? payload.data : []);
			nextOffset = payload.nextOffset || 0;
			doneLoading = Boolean(payload.done);
			if (!doneLoading) scheduleLoadMore(token, dir);
		} catch (e) {
			console.error('Failed to fetch logs', e);
			error = e instanceof Error ? e.message : String(e);
		} finally {
			if (token === loadToken) loading = false;
		}
	}

	async function scheduleLoadMore(token: number, dir: string) {
		if (doneLoading || loadingMore || token !== loadToken || dir !== activeLogDir) return;
		loadingMore = true;
		setTimeout(async () => {
			if (token !== loadToken || dir !== activeLogDir) {
				loadingMore = false;
				return;
			}
			try {
				const res = await fetch(buildLogsUrl(nextOffset, dir));
				if (!res.ok) throw new Error(`Failed to fetch logs (${res.status})`);
				const payload = await res.json();
				if (token !== loadToken || dir !== activeLogDir) return;
				updateFromChunk(Array.isArray(payload.data) ? payload.data : []);
				if (typeof payload.nextOffset === 'number') {
					nextOffset = payload.nextOffset;
				}
				doneLoading = Boolean(payload.done);
			} catch (e) {
				if (token === loadToken) {
					error = e instanceof Error ? e.message : String(e);
				}
			} finally {
				loadingMore = false;
				if (!doneLoading) scheduleLoadMore(token, dir);
			}
		}, 200);
	}

	onMount(fetchLogs);

	let epochData = $derived.by(() =>
		Array.from(logMap.values()).sort((a, b) => a.epoch - b.epoch)
	);
	let latest = $derived.by(() => (epochData.length > 0 ? epochData[epochData.length - 1] : null));

	const toSeries = (picker: (row: RlPoint) => number | null) =>
		epochData.map((row) => ({ x: row.epoch, y: picker(row) }));

	const toFrontierSeries = (
		drawdownPicker: (row: RlPoint) => number | null,
		realizedPicker: (row: RlPoint) => number | null,
		pnlPicker: (row: RlPoint) => number | null
	) =>
		epochData.flatMap((row) => {
			const drawdown = drawdownPicker(row);
			const realizedPnl = realizedPicker(row);
			const pnl = pnlPicker(row);
			const value = realizedPnl ?? pnl;
			if (drawdown === null || value === null) return [];
			return [
				{
					x: drawdown,
					y: value,
					epoch: row.epoch,
					metricLabel: realizedPnl !== null ? 'Realized PnL' : 'PnL'
				}
			];
		});

	let resolvedWeights = $derived.by(() => ({
		pnl: toNumber(fitnessWeights.pnl) ?? 1,
		sortino: toNumber(fitnessWeights.sortino) ?? 1,
		mdd: toNumber(fitnessWeights.mdd) ?? 0.5
	}));

	const calcFitness = (pnl: number | null, sortino: number | null, drawdown: number | null) => {
		if (pnl === null || sortino === null || drawdown === null) return null;
		return (
			resolvedWeights.pnl * pnl +
			resolvedWeights.sortino * sortino -
			resolvedWeights.mdd * drawdown
		);
	};

	const chartOptions = {
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			}
		}
	};

	let hasTrainRealized = $derived.by(() =>
		epochData.some((row) => row.trainRealizedPnl !== null)
	);
	let hasEvalRealized = $derived.by(() =>
		epochData.some((row) => row.evalRealizedPnl !== null)
	);
	let frontierAxisLabel = $derived.by(() => {
		const realizedSeriesCount = Number(hasTrainRealized) + Number(hasEvalRealized);
		if (realizedSeriesCount === 2) return 'Realized PnL';
		if (realizedSeriesCount === 1) return 'Realized PnL / PnL';
		return 'PnL';
	});
	let frontierNote = $derived.by(() => {
		const hasTrainPnl = epochData.some((row) => row.trainPnl !== null);
		const hasEvalPnl = epochData.some((row) => row.evalPnl !== null);
		const trainFallback = hasTrainPnl && !hasTrainRealized;
		const evalFallback = hasEvalPnl && !hasEvalRealized;
		if (trainFallback && evalFallback) {
			return 'This RL run does not expose realized-PnL columns, so the frontier uses the logged train and eval PnL values.';
		}
		if (trainFallback) {
			return 'Train frontier points use logged PnL because this run does not include train realized-PnL columns.';
		}
		if (evalFallback) {
			return 'Eval frontier points use logged PnL because this run does not include eval realized-PnL columns.';
		}
		return '';
	});

	let fitnessChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Train Fitness',
				data: toSeries((row) => calcFitness(row.trainPnl, row.trainSortino, row.trainDrawdown)),
				borderColor: 'rgb(148, 163, 184)',
				backgroundColor: 'rgba(148, 163, 184, 0.25)',
				tension: 0.1
			},
			{
				label: 'Eval Fitness',
				data: toSeries((row) => calcFitness(row.evalPnl, row.evalSortino, row.evalDrawdown)),
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.3)',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	}));

	let performanceChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Train PnL',
				data: toSeries((row) => row.trainPnl),
				borderColor: 'rgb(148, 163, 184)',
				backgroundColor: 'rgba(148, 163, 184, 0.25)',
				tension: 0.1
			},
			{
				label: 'Eval PnL',
				data: toSeries((row) => row.evalPnl),
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.3)',
				tension: 0.1
			},
			{
				label: 'Eval Sortino',
				data: toSeries((row) => row.evalSortino),
				borderColor: 'rgb(168, 85, 247)',
				backgroundColor: 'rgba(168, 85, 247, 0.2)',
				borderDash: [4, 4],
				yAxisID: 'y1',
				tension: 0.1
			}
		]
	}));

	let drawdownChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Train Drawdown',
				data: toSeries((row) => row.trainDrawdown),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.3)',
				tension: 0.1
			},
			{
				label: 'Eval Drawdown',
				data: toSeries((row) => row.evalDrawdown),
				borderColor: 'rgb(248, 113, 113)',
				backgroundColor: 'rgba(248, 113, 113, 0.2)',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	}));

	let frontierChartData = $derived.by(() => {
		const datasets = [];
		const trainData = toFrontierSeries(
			(row) => row.trainDrawdown,
			(row) => row.trainRealizedPnl,
			(row) => row.trainPnl
		);
		const evalData = toFrontierSeries(
			(row) => row.evalDrawdown,
			(row) => row.evalRealizedPnl,
			(row) => row.evalPnl
		);

		if (trainData.length > 0) {
			datasets.push({
				label: hasTrainRealized ? 'Train Realized PnL vs Drawdown' : 'Train PnL vs Drawdown',
				data: trainData,
				borderColor: 'rgb(148, 163, 184)',
				backgroundColor: 'rgba(148, 163, 184, 0.55)',
				showLine: false,
				pointRadius: 3
			});
		}

		if (evalData.length > 0) {
			datasets.push({
				label: hasEvalRealized ? 'Eval Realized PnL vs Drawdown' : 'Eval PnL vs Drawdown',
				data: evalData,
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.45)',
				showLine: false,
				pointRadius: 3
			});
		}

		return { labels: [], datasets };
	});

	let frontierOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Max Drawdown (%)' }
			},
			y: {
				title: { display: true, text: frontierAxisLabel }
			}
		},
		plugins: {
			tooltip: {
				callbacks: {
					label: (ctx: any) => {
						const raw = ctx.raw as {
							x?: number;
							y?: number;
							epoch?: number;
							metricLabel?: string;
						};
						const drawdown = raw?.x ?? ctx.parsed?.x;
						const pnl = raw?.y ?? ctx.parsed?.y;
						return [
							ctx.dataset.label ?? 'Frontier Point',
							`Epoch: ${raw?.epoch ?? '—'}`,
							`Drawdown: ${drawdown ?? '—'}`,
							`${raw?.metricLabel ?? frontierAxisLabel}: ${pnl ?? '—'}`
						];
					}
				}
			}
		}
	}));

	let lossChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Policy Loss',
				data: toSeries((row) => row.policyLoss),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.3)',
				tension: 0.1
			},
			{
				label: 'Value Loss',
				data: toSeries((row) => row.valueLoss),
				borderColor: 'rgb(16, 185, 129)',
				backgroundColor: 'rgba(16, 185, 129, 0.25)',
				tension: 0.1
			},
			{
				label: 'Entropy',
				data: toSeries((row) => row.entropy),
				borderColor: 'rgb(251, 191, 36)',
				backgroundColor: 'rgba(251, 191, 36, 0.2)',
				borderDash: [4, 4],
				yAxisID: 'y1',
				tension: 0.1
			},
			{
				label: 'Perplexity',
				data: toSeries((row) => row.perplexity),
				borderColor: 'rgb(249, 115, 22)',
				backgroundColor: 'rgba(249, 115, 22, 0.18)',
				yAxisID: 'y1',
				tension: 0.1
			},
			{
				label: 'Approx KL',
				data: toSeries((row) => row.approxKl),
				borderColor: 'rgb(244, 63, 94)',
				backgroundColor: 'rgba(244, 63, 94, 0.18)',
				yAxisID: 'y2',
				tension: 0.1
			},
			{
				label: 'KL Div',
				data: toSeries((row) => row.klDiv),
				borderColor: 'rgb(236, 72, 153)',
				backgroundColor: 'rgba(236, 72, 153, 0.18)',
				borderDash: [4, 4],
				yAxisID: 'y2',
				tension: 0.1
			},
			{
				label: 'Clip Frac',
				data: toSeries((row) => row.clipFrac),
				borderColor: 'rgb(14, 165, 233)',
				backgroundColor: 'rgba(14, 165, 233, 0.18)',
				borderDash: [2, 3],
				yAxisID: 'y2',
				tension: 0.1
			}
		]
	}));

	let lossOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			},
			y: {
				title: { display: true, text: 'Loss' }
			},
			y1: {
				position: 'right' as const,
				grid: { drawOnChartArea: false },
				title: { display: true, text: 'Entropy / Perplexity' }
			},
			y2: {
				position: 'right' as const,
				grid: { drawOnChartArea: false },
				title: { display: true, text: 'KL / Clip' }
			}
		}
	}));

	let gradientChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Policy Grad Norm',
				data: toSeries((row) => row.policyGradNorm),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.25)',
				tension: 0.1
			},
			{
				label: 'Value Grad Norm',
				data: toSeries((row) => row.valueGradNorm),
				borderColor: 'rgb(16, 185, 129)',
				backgroundColor: 'rgba(16, 185, 129, 0.22)',
				tension: 0.1
			}
		]
	}));

	let gradientOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			},
			y: {
				title: { display: true, text: 'L2 Gradient Norm' }
			}
		}
	}));

	let policyChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Eval Buy %',
				data: toSeries((row) => row.evalBuyFrac),
				borderColor: 'rgb(34, 197, 94)',
				backgroundColor: 'rgba(34, 197, 94, 0.2)',
				tension: 0.1
			},
			{
				label: 'Eval Sell %',
				data: toSeries((row) => row.evalSellFrac),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.2)',
				tension: 0.1
			},
			{
				label: 'Eval Hold %',
				data: toSeries((row) => row.evalHoldFrac),
				borderColor: 'rgb(148, 163, 184)',
				backgroundColor: 'rgba(148, 163, 184, 0.18)',
				tension: 0.1
			},
			{
				label: 'Eval Revert %',
				data: toSeries((row) => row.evalRevertFrac),
				borderColor: 'rgb(168, 85, 247)',
				backgroundColor: 'rgba(168, 85, 247, 0.16)',
				tension: 0.1
			},
			{
				label: 'Eval Max Prob',
				data: toSeries((row) => row.evalMeanMaxProb),
				borderColor: 'rgb(245, 158, 11)',
				backgroundColor: 'rgba(245, 158, 11, 0.18)',
				borderDash: [4, 4],
				tension: 0.1
			},
			{
				label: 'Probe Max Prob',
				data: toSeries((row) => row.probeMeanMaxProb),
				borderColor: 'rgb(14, 165, 233)',
				backgroundColor: 'rgba(14, 165, 233, 0.18)',
				borderDash: [2, 3],
				tension: 0.1
			}
		]
	}));

	let policyOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			},
			y: {
				min: 0,
				max: 1,
				title: { display: true, text: 'Fraction / Probability' }
			}
		}
	}));

	let probeChartData = $derived.by(() => ({
		datasets: [
			{
				label: 'Probe PnL',
				data: toSeries((row) => row.probePnl),
				borderColor: 'rgb(59, 130, 246)',
				backgroundColor: 'rgba(59, 130, 246, 0.22)',
				tension: 0.1
			},
			{
				label: 'Probe Realized PnL',
				data: toSeries((row) => row.probeRealizedPnl),
				borderColor: 'rgb(16, 185, 129)',
				backgroundColor: 'rgba(16, 185, 129, 0.2)',
				tension: 0.1
			},
			{
				label: 'Probe Drawdown',
				data: toSeries((row) => row.probeDrawdown),
				borderColor: 'rgb(239, 68, 68)',
				backgroundColor: 'rgba(239, 68, 68, 0.18)',
				yAxisID: 'y1',
				tension: 0.1
			},
			{
				label: 'Probe Max Prob',
				data: toSeries((row) => row.probeMeanMaxProb),
				borderColor: 'rgb(245, 158, 11)',
				backgroundColor: 'rgba(245, 158, 11, 0.18)',
				yAxisID: 'y2',
				borderDash: [4, 4],
				tension: 0.1
			}
		]
	}));

	let probeOptions = $derived.by(() => ({
		scales: {
			x: {
				type: 'linear' as const,
				title: { display: true, text: 'Epoch' },
				ticks: { precision: 0 }
			},
			y: {
				title: { display: true, text: 'PnL' }
			},
			y1: {
				position: 'right' as const,
				grid: { drawOnChartArea: false },
				title: { display: true, text: 'Drawdown (%)' }
			},
			y2: {
				position: 'right' as const,
				grid: { drawOnChartArea: false },
				min: 0,
				max: 1,
				title: { display: true, text: 'Confidence' }
			}
		}
	}));
</script>

<main class="p-8 space-y-8">
	<div class="flex flex-wrap items-center justify-between gap-4">
		<div>
			<h1 class="text-4xl font-bold tracking-tight">RL Analytics</h1>
			<p class="text-sm text-muted-foreground">RL training metrics from Rust runs.</p>
		</div>
		<div class="flex flex-wrap items-center gap-3">
			<div class="flex items-center gap-2">
				<Input
					class="w-56"
					placeholder="runs_rl"
					bind:value={logDir}
				/>
				<Button onclick={fetchLogs} disabled={loading}>
					{loading ? 'Loading...' : 'Reload'}
				</Button>
				<Button variant="outline" onclick={openFolderPicker}>
					Browse
				</Button>
			</div>
			<div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
				<span class="font-medium uppercase tracking-wide">Fitness weights</span>
				<div class="flex items-center gap-1">
					<span>w_pnl</span>
					<Input
						class="h-8 w-20 text-xs"
						type="number"
						step="0.01"
						aria-label="Fitness weight PnL"
						bind:value={fitnessWeights.pnl}
					/>
				</div>
				<div class="flex items-center gap-1">
					<span>w_sortino</span>
					<Input
						class="h-8 w-20 text-xs"
						type="number"
						step="0.01"
						aria-label="Fitness weight Sortino"
						bind:value={fitnessWeights.sortino}
					/>
				</div>
				<div class="flex items-center gap-1">
					<span>w_mdd</span>
					<Input
						class="h-8 w-20 text-xs"
						type="number"
						step="0.01"
						aria-label="Fitness weight MDD"
						bind:value={fitnessWeights.mdd}
					/>
				</div>
			</div>
		</div>
	</div>

	{#if error}
		<div class="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
			{error}
		</div>
	{/if}

	<div class="grid gap-6 xl:grid-cols-3">
		<Card.Root class="xl:col-span-2">
			<Card.Header>
				<Card.Title>Training Curves</Card.Title>
				<Card.Description>
					{activeLogDir} · {epochData.length.toLocaleString()} epochs
				</Card.Description>
			</Card.Header>
			<Card.Content>
				<Tabs.Root bind:value={chartTab} class="w-full">
					<Tabs.List class="mb-4 grid w-full grid-cols-2 lg:w-[1240px] lg:grid-cols-8">
						<Tabs.Trigger value="fitness">Fitness</Tabs.Trigger>
						<Tabs.Trigger value="performance">Performance</Tabs.Trigger>
						<Tabs.Trigger value="drawdown">Drawdown</Tabs.Trigger>
						<Tabs.Trigger value="frontier">Frontier</Tabs.Trigger>
						<Tabs.Trigger value="loss">Loss</Tabs.Trigger>
						<Tabs.Trigger value="gradients">Gradients</Tabs.Trigger>
						<Tabs.Trigger value="policy">Policy</Tabs.Trigger>
						<Tabs.Trigger value="probe">Probe</Tabs.Trigger>
					</Tabs.List>
					<Tabs.Content value="fitness">
						<div class="h-[320px]">
							<GaChart data={fitnessChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="performance">
						<div class="h-[320px]">
							<GaChart data={performanceChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="drawdown">
						<div class="h-[320px]">
							<GaChart data={drawdownChartData} options={chartOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="frontier">
						{#if frontierChartData.datasets.length > 0}
							<div class="space-y-3">
								<div class="h-[320px]">
									<GaChart data={frontierChartData} options={frontierOptions} type="scatter" />
								</div>
								{#if frontierNote}
									<p class="text-xs text-muted-foreground">
										{frontierNote}
									</p>
								{/if}
							</div>
						{:else}
							<div class="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
								Not enough drawdown and PnL data to build the frontier yet.
							</div>
						{/if}
					</Tabs.Content>
					<Tabs.Content value="loss">
						<div class="h-[320px]">
							<GaChart data={lossChartData} options={lossOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="gradients">
						<div class="h-[320px]">
							<GaChart data={gradientChartData} options={gradientOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="policy">
						<div class="h-[320px]">
							<GaChart data={policyChartData} options={policyOptions} />
						</div>
					</Tabs.Content>
					<Tabs.Content value="probe">
						<div class="h-[320px]">
							<GaChart data={probeChartData} options={probeOptions} />
						</div>
					</Tabs.Content>
				</Tabs.Root>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header>
				<Card.Title>Latest Snapshot</Card.Title>
				<Card.Description>Most recent epoch summary.</Card.Description>
			</Card.Header>
			<Card.Content>
				{#if latest}
					<div class="space-y-3 text-sm">
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Algorithm</span>
							<span class="font-semibold uppercase">{latest.algorithm ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Epoch</span>
							<span class="font-semibold">{latest.epoch}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Train Fitness</span>
							<span class="font-semibold">{calcFitness(latest.trainPnl, latest.trainSortino, latest.trainDrawdown) ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Fitness</span>
							<span class="font-semibold">{calcFitness(latest.evalPnl, latest.evalSortino, latest.evalDrawdown) ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Logged Fitness</span>
							<span class="font-semibold">{latest.fitness ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval PnL</span>
							<span>{latest.evalPnl ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Realized PnL</span>
							<span>{latest.evalRealizedPnl ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Sortino</span>
							<span>{latest.evalSortino ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Drawdown</span>
							<span>{latest.evalDrawdown ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Eval Max Prob</span>
							<span>{latest.evalMeanMaxProb ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Probe PnL</span>
							<span>{latest.probePnl ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Probe Drawdown</span>
							<span>{latest.probeDrawdown ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Policy Loss</span>
							<span>{latest.policyLoss ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Value Loss</span>
							<span>{latest.valueLoss ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Policy Grad Norm</span>
							<span>{latest.policyGradNorm ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Value Grad Norm</span>
							<span>{latest.valueGradNorm ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Entropy</span>
							<span>{latest.entropy ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Perplexity</span>
							<span>{latest.perplexity ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Approx KL</span>
							<span>{latest.approxKl ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">KL Div</span>
							<span>{latest.klDiv ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Clip Frac</span>
							<span>{latest.clipFrac ?? '—'}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-muted-foreground">Probe Entries / Flips</span>
							<span>{latest.probeEntries ?? '—'} / {latest.probeFlips ?? '—'}</span>
						</div>
					</div>
				{:else}
					<div class="text-sm text-muted-foreground">No RL logs yet.</div>
				{/if}
			</Card.Content>
			{#if loading}
				<Card.Footer>
					<Badge variant="outline" class="animate-pulse">Loading</Badge>
				</Card.Footer>
			{/if}
		</Card.Root>
	</div>

	<!-- Folder Picker Modal -->
	{#if folderPickerOpen}
		<div
			class="fixed inset-0 z-50 flex items-center justify-center p-4"
			role="dialog"
			aria-modal="true"
			aria-label="Select RL run folder"
		>
			<button
				type="button"
				class="absolute inset-0 bg-black/40"
				onclick={closeFolderPicker}
				aria-label="Close folder picker"
			></button>
			<div class="relative z-10 w-full max-w-2xl rounded-xl border bg-background p-4 shadow-lg">
				<div class="flex items-start justify-between gap-4">
					<div>
						<div class="text-xs uppercase tracking-wide text-muted-foreground">Select RL Run Folder</div>
						<div class="text-sm font-semibold">runs_rl</div>
						<div class="text-xs text-muted-foreground">
							Sorted by modification time (newest first)
						</div>
					</div>
					<Button type="button" variant="ghost" size="sm" onclick={closeFolderPicker}>
						Close
					</Button>
				</div>

				{#if folderPickerError}
					<div class="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
						{folderPickerError}
					</div>
				{/if}

				<ScrollArea class="mt-3 max-h-[360px] rounded-lg border">
					{#if folderPickerLoading}
						<div class="px-4 py-6 text-sm text-muted-foreground">Loading folders...</div>
					{:else if folderEntries.length === 0}
						<div class="px-4 py-6 text-sm text-muted-foreground">No run folders found in runs_rl.</div>
					{:else}
						<div class="divide-y">
							{#each folderEntries as folder}
								<button
									type="button"
									class="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-muted/50"
									onclick={() => selectFolder(folder)}
								>
									<span class="text-[11px] font-semibold uppercase text-muted-foreground">Folder</span>
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
</main>

<script lang="ts">
	import { onMount } from "svelte";
	import RlFolderPicker from "./_components/RlFolderPicker.svelte";
	import RlLatestSnapshotCard from "./_components/RlLatestSnapshotCard.svelte";
	import RlPageHeader from "./_components/RlPageHeader.svelte";
	import RlTrainingCurvesCard from "./_components/RlTrainingCurvesCard.svelte";
	import {
		DEFAULT_FITNESS_WEIGHTS,
		DEFAULT_LOG_DIR,
		buildRlCharts,
		buildSnapshotRows,
		getLatestPoint,
		mergeRlRows,
		normalizeLogDir,
		resolveFitnessWeights,
		sortEpochData
	} from "./analytics";
	import type { ChartTab, FitnessWeights, FolderEntry, RlPoint } from "./types";

	const logChunk = 1000;

	let chartTab = $state<ChartTab>("fitness");
	let logDir = $state(DEFAULT_LOG_DIR);
	let activeLogDir = $state(DEFAULT_LOG_DIR);
	let logMap = $state(new Map<number, RlPoint>());
	let fitnessWeights = $state<FitnessWeights>({ ...DEFAULT_FITNESS_WEIGHTS });
	let loading = $state(false);
	let loadingMore = $state(false);
	let doneLoading = $state(false);
	let nextOffset = $state(0);
	let error = $state("");
	let loadToken = 0;

	let folderPickerOpen = $state(false);
	let folderPickerLoading = $state(false);
	let folderPickerError = $state("");
	let folderEntries = $state<FolderEntry[]>([]);
	let folderPickerToken = 0;

	const buildLogsUrl = (offset: number, dir: string) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir,
			log: "rl"
		});
		return `/api/logs?${params.toString()}`;
	};

	const loadFolders = async () => {
		const token = ++folderPickerToken;
		folderPickerLoading = true;
		folderPickerError = "";
		try {
			const res = await fetch(`/api/files?dir=${DEFAULT_LOG_DIR}`);
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to load folders (${res.status})`);
			}
			const payload = await res.json();
			if (token !== folderPickerToken) return;

			const entries = Array.isArray(payload.entries) ? payload.entries : [];
			folderEntries = entries
				.filter((entry: FolderEntry) => entry.kind === "dir")
				.sort((a: FolderEntry, b: FolderEntry) => (b.mtime || 0) - (a.mtime || 0));
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
		folderPickerError = "";
	};

	const selectFolder = (folder: FolderEntry) => {
		logDir = folder.path;
		closeFolderPicker();
		void fetchLogs();
	};

	const resetState = () => {
		logMap = new Map();
		loading = true;
		error = "";
		nextOffset = 0;
		doneLoading = false;
		loadingMore = false;
	};

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

			logMap = mergeRlRows(new Map(), Array.isArray(payload.data) ? payload.data : []);
			nextOffset = typeof payload.nextOffset === "number" ? payload.nextOffset : 0;
			doneLoading = Boolean(payload.done);

			if (!doneLoading) {
				void scheduleLoadMore(token, dir);
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

				logMap = mergeRlRows(logMap, Array.isArray(payload.data) ? payload.data : []);
				if (typeof payload.nextOffset === "number") {
					nextOffset = payload.nextOffset;
				}
				doneLoading = Boolean(payload.done);
			} catch (err) {
				if (token === loadToken) {
					error = err instanceof Error ? err.message : String(err);
				}
			} finally {
				loadingMore = false;
				if (!doneLoading) {
					void scheduleLoadMore(token, dir);
				}
			}
		}, 200);
	}

	onMount(fetchLogs);

	let epochData = $derived.by(() => sortEpochData(logMap));
	let latest = $derived.by(() => getLatestPoint(epochData));
	let resolvedWeights = $derived.by(() => resolveFitnessWeights(fitnessWeights));
	let charts = $derived.by(() => buildRlCharts(epochData, resolvedWeights));
	let snapshotRows = $derived.by(() => buildSnapshotRows(latest, resolvedWeights));
</script>

<main class="space-y-8 p-8">
	<RlPageHeader
		bind:logDir
		bind:fitnessWeights
		{loading}
		onReload={fetchLogs}
		onBrowse={openFolderPicker}
	/>

	{#if error}
		<div class="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">
			{error}
		</div>
	{/if}

	<div class="grid gap-6 xl:grid-cols-3">
		<RlTrainingCurvesCard
			bind:chartTab
			{activeLogDir}
			epochCount={epochData.length}
			{charts}
		/>
		<RlLatestSnapshotCard {latest} {snapshotRows} {loading} />
	</div>

	<RlFolderPicker
		open={folderPickerOpen}
		loading={folderPickerLoading}
		error={folderPickerError}
		entries={folderEntries}
		onClose={closeFolderPicker}
		onSelect={selectFolder}
	/>
</main>

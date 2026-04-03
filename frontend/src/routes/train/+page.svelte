<script lang="ts">
	import {
		defaultStepBars,
		defaultWindowBars,
		futuresPresets
	} from "./constants";
	import TrainConsoleCard from "./_components/TrainConsoleCard.svelte";
	import TrainDiagnosticsCard from "./_components/TrainDiagnosticsCard.svelte";
	import TrainFileBrowserModal from "./_components/TrainFileBrowserModal.svelte";
	import TrainFitnessCard from "./_components/TrainFitnessCard.svelte";
	import TrainPageHeader from "./_components/TrainPageHeader.svelte";
	import TrainSidebar from "./_components/TrainSidebar.svelte";
	import type {
		BuildProfile,
		ConsoleLine,
		DataMode,
		FileEntry,
		FilePickerTarget,
		FitnessPoint,
		FuturesPresetKey,
		GaParams,
		MlBackend,
		ParquetKey,
		RlAlgorithm,
		RlParams,
		StartChoice,
		TrainMode
	} from "./types";

	let trainMode = $state<TrainMode>("ga");
	let rlAlgorithm = $state<RlAlgorithm>("ppo");
	let consoleOutput = $state<ConsoleLine[]>([]);
	let training = $state(false);
	let paramsCollapsed = $state(false);
	let startChoice = $state<StartChoice | null>(null);
	let checkpointPath = $state("");
	let gaDataMode = $state<DataMode>("windowed");
	let rlDataMode = $state<DataMode>("windowed");
	let abortController: AbortController | null = null;
	let fitnessByGen = $state(new Map<number, number>());
	let logOffset = $state(0);
	let logPolling = false;
	let logTimer: ReturnType<typeof setTimeout> | null = null;
	let activeLogDir = $state("runs_ga");
	const logChunk = 500;
	const logPollInterval = 1500;
	const logPollMaxInterval = 6000;
	let logPollDelay = logPollInterval;
	let liveLogUpdates = $state(true);
	let diagnosticsOutput = $state<string | null>(null);
	let diagnosticsError = $state<string | null>(null);
	let diagnosticsLoading = $state(false);
	let diagnosticsEnv = $state<Record<string, string | null> | null>(null);
	let buildProfile = $state<BuildProfile>("debug");

	let gaParams = $state<GaParams>({
		backend: "libtorch",
		outdir: "runs_ga",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "auto",
		"batch-candidates": 0,
		generations: 5,
		"pop-size": 6,
		workers: 2,
		window: defaultWindowBars,
		step: defaultStepBars,
		"initial-balance": 10000,
		"max-position": 1,
		"margin-mode": "auto",
		"contract-multiplier": 1.0,
		"margin-per-contract": "",
		"auto-close-minutes-before-close": 5,
		"max-hold-bars-positive": 15,
		"max-hold-bars-drawdown": 15,
		"hold-duration-penalty": 1.0,
		"hold-duration-penalty-growth": 0.05,
		"hold-duration-penalty-positive-scale": 0.5,
		"hold-duration-penalty-negative-scale": 1.5,
		"selection-train-weight": 0.3,
		"selection-eval-weight": 0.7,
		"selection-gap-penalty": 0.2,
		"elite-frac": 0.33,
		"mutation-sigma": 0.05,
		"init-sigma": 0.5,
		hidden: 64,
		layers: 2,
		"eval-windows": 2,
		"w-pnl": 1.0,
		"w-sortino": 1.0,
		"w-mdd": 0.5,
		"drawdown-penalty": 0.0,
		"drawdown-penalty-growth": 0.0,
		"session-close-penalty": 0.0,
		"flat-hold-penalty": 2.2,
		"flat-hold-penalty-growth": 0.05,
		"max-flat-hold-bars": 100,
		"save-top-n": 5,
		"save-every": 1,
		"checkpoint-every": 1
	});

	let rlParams = $state<RlParams>({
		algorithm: "ppo",
		backend: "libtorch",
		outdir: "runs_rl",
		"train-parquet": "data/train",
		"val-parquet": "data/val",
		"test-parquet": "data/test",
		device: "auto",
		window: defaultWindowBars,
		step: defaultStepBars,
		epochs: 10,
		"train-windows": 3,
		"ppo-epochs": 4,
		"group-size": 8,
		"grpo-epochs": 4,
		lr: 0.0003,
		gamma: 0.99,
		lam: 0.95,
		clip: 0.2,
		"vf-coef": 0.5,
		"ent-coef": 0.01,
		dropout: 0.0,
		hidden: 64,
		layers: 2,
		"eval-windows": 2,
		"initial-balance": 10000,
		"max-position": 1,
		"margin-mode": "auto",
		"contract-multiplier": 1.0,
		"margin-per-contract": "",
		"auto-close-minutes-before-close": 5,
		"max-hold-bars-positive": 15,
		"max-hold-bars-drawdown": 15,
		"hold-duration-penalty": 1.0,
		"hold-duration-penalty-growth": 0.05,
		"hold-duration-penalty-positive-scale": 0.5,
		"hold-duration-penalty-negative-scale": 1.5,
		"w-pnl": 1.0,
		"w-sortino": 1.0,
		"w-mdd": 0.5,
		"log-interval": 1,
		"checkpoint-every": 1
	});

	let filePickerTarget: FilePickerTarget | null = null;
	let fileBrowserOpen = $state(false);
	let fileBrowserDir = $state("");
	let fileBrowserParent = $state<string | null>(null);
	let fileBrowserEntries = $state<FileEntry[]>([]);
	let fileBrowserLoading = $state(false);
	let fileBrowserError = $state("");
	let fileBrowserToken = 0;
	let fileBrowserTitle = $state("Select File");
	let fileBrowserExtensions = $state<string[]>(["parquet"]);

	const setParquetPath = (mode: TrainMode, key: ParquetKey, value: string) => {
		if (mode === "ga") {
			gaParams[key] = value;
		} else {
			rlParams[key] = value;
		}
	};

	const applyFuturesPreset = (mode: TrainMode, presetKey: FuturesPresetKey) => {
		const preset = futuresPresets[presetKey];
		const target = mode === "ga" ? gaParams : rlParams;
		target["margin-mode"] = "per-contract";
		target["margin-per-contract"] = String(preset.marginPerContract);
		target["contract-multiplier"] = preset.contractMultiplier;
		target["auto-close-minutes-before-close"] = 5;
		consoleOutput = [
			...consoleOutput,
			{
				type: "system",
				text: `Applied ${preset.label} preset (${preset.note}): margin ${preset.marginPerContract}, multiplier ${preset.contractMultiplier}, auto-close 5m.`
			}
		];
	};

	const normalizeExtensions = (extensions: string[]) =>
		extensions.map((ext) => ext.replace(/^\./, "").toLowerCase());

	const runDiagnostics = async () => {
		if (diagnosticsLoading) return;
		diagnosticsLoading = true;
		diagnosticsError = null;
		diagnosticsEnv = null;
		try {
			const response = await fetch("/api/diagnostics");
			const data = await response.json();
			if (!response.ok || !data.ok) {
				diagnosticsOutput = (data.stdout || data.output || "").trim();
				diagnosticsError = (data.stderr || data.error || "Diagnostics failed").trim();
				diagnosticsEnv = data.env ?? null;
				return;
			}
			diagnosticsOutput = (data.output || "").trim();
			diagnosticsEnv = data.env ?? null;
		} catch (err) {
			diagnosticsError = err instanceof Error ? err.message : String(err);
		} finally {
			diagnosticsLoading = false;
		}
	};

	const buildFileBrowserUrl = (dir: string, extensions: string[]) => {
		const params = new URLSearchParams();
		if (dir.trim() !== "") {
			params.set("dir", dir);
		}
		const normalized = normalizeExtensions(extensions);
		if (normalized.length > 0) {
			params.set("ext", normalized.join(","));
		}
		const query = params.toString();
		return query ? `/api/files?${query}` : "/api/files";
	};

	const loadFileBrowser = async (dir: string, extensions: string[]) => {
		const token = ++fileBrowserToken;
		fileBrowserLoading = true;
		fileBrowserError = "";
		try {
			const res = await fetch(buildFileBrowserUrl(dir, extensions));
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to load files (${res.status})`);
			}
			const payload = await res.json();
			if (token !== fileBrowserToken) return false;
			fileBrowserDir = typeof payload.dir === "string" ? payload.dir : dir;
			fileBrowserParent = typeof payload.parent === "string" ? payload.parent : null;
			fileBrowserEntries = Array.isArray(payload.entries) ? payload.entries : [];
			return true;
		} catch (err) {
			if (token === fileBrowserToken) {
				fileBrowserError = err instanceof Error ? err.message : String(err);
				fileBrowserEntries = [];
				fileBrowserParent = null;
			}
			return false;
		} finally {
			if (token === fileBrowserToken) {
				fileBrowserLoading = false;
			}
		}
	};

	const openFileBrowser = async (
		target: FilePickerTarget,
		title: string,
		extensions: string[],
		preferredDir: string
	) => {
		filePickerTarget = target;
		fileBrowserTitle = title;
		fileBrowserExtensions = normalizeExtensions(extensions);
		fileBrowserOpen = true;
		fileBrowserDir = "";
		fileBrowserParent = null;
		fileBrowserEntries = [];
		const ok = await loadFileBrowser(preferredDir, fileBrowserExtensions);
		if (!ok && preferredDir !== "") {
			await loadFileBrowser("", fileBrowserExtensions);
		}
	};

	const openParquetPicker = async (mode: TrainMode, key: ParquetKey) => {
		await openFileBrowser({ kind: "parquet", mode, key }, "Select Parquet File", ["parquet"], "data");
	};

	const openCheckpointPicker = async (mode: TrainMode) => {
		const ext = mode === "ga" ? "bin" : "pt";
		const title = mode === "ga" ? "Select GA Checkpoint" : "Select RL Checkpoint";
		const preferredDir = mode === "ga" ? "runs_ga" : "runs_rl";
		await openFileBrowser({ kind: "checkpoint", mode }, title, [ext], preferredDir);
	};

	const closeFileBrowser = () => {
		fileBrowserOpen = false;
		fileBrowserError = "";
		filePickerTarget = null;
	};

	const handleFileEntry = (entry: FileEntry) => {
		if (entry.kind === "dir") {
			void loadFileBrowser(entry.path, fileBrowserExtensions);
			return;
		}
		if (!filePickerTarget) return;
		if (filePickerTarget.kind === "parquet") {
			setParquetPath(filePickerTarget.mode, filePickerTarget.key, entry.path);
		} else {
			checkpointPath = entry.path;
		}
		closeFileBrowser();
	};

	const handleFileBrowserUp = () => {
		if (fileBrowserParent === null) return;
		void loadFileBrowser(fileBrowserParent, fileBrowserExtensions);
	};

	const toNumber = (value: unknown): number | null => {
		if (typeof value === "number") return Number.isFinite(value) ? value : null;
		if (typeof value === "string" && value.trim() !== "") {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}
		return null;
	};

	const trimmedCheckpoint = $derived.by(() => checkpointPath.trim());
	const canStartTraining = $derived.by(() => {
		if (!startChoice) return false;
		if (startChoice === "resume") return trimmedCheckpoint.length > 0;
		return true;
	});
	const runLabel = $derived.by(() => {
		if (training) return "Stop Training";
		if (startChoice === "resume") {
			return trainMode === "ga" ? "Resume GA Training" : "Resume RL Training";
		}
		if (startChoice === "new") {
			return trainMode === "ga" ? "Start GA Training" : "Start RL Training";
		}
		return "Start Training";
	});
	const runTitle = $derived.by(() => {
		if (training) return "Stop Training";
		if (startChoice === "resume") return "Resume Training";
		if (startChoice === "new") return "Start Training";
		return "Start Training";
	});
	const collapsedLabel = $derived.by(() => (training ? "Stop" : "Setup"));
	const collapsedTitle = $derived.by(() => (training ? "Stop Training" : "Open Setup"));
	const logKey = $derived.by(() => (trainMode === "rl" ? "epoch" : "gen"));
	const logLabel = $derived.by(() => (trainMode === "rl" ? "Epoch" : "Gen"));
	const logType = $derived.by(() => (trainMode === "rl" ? "rl" : "ga"));
	const fitnessSeries = $derived.by<FitnessPoint[]>(() =>
		Array.from(fitnessByGen.entries())
			.sort(([a], [b]) => a - b)
			.map(([gen, fitness]) => ({ gen, fitness }))
	);
	const fitnessChartData = $derived.by(() => ({
		labels: fitnessSeries.map((point) => `${logLabel} ${point.gen}`),
		datasets: [
			{
				label: "Best Fitness",
				data: fitnessSeries.map((point) => point.fitness),
				borderColor: "rgb(59, 130, 246)",
				backgroundColor: "rgba(59, 130, 246, 0.35)",
				tension: 0.2,
				fill: false
			}
		]
	}));
	const fileBrowserEmptyMessage = $derived.by(() =>
		fileBrowserExtensions.includes("parquet") ? "No parquet files found here." : "No matching files found here."
	);

	const fitnessChartOptions = {
		scales: {
			y: {
				title: {
					display: true,
					text: "Fitness"
				}
			}
		}
	};

	const updateFitness = (gen: number, fitness: number) => {
		if (!Number.isFinite(gen) || !Number.isFinite(fitness)) return;
		const next = new Map(fitnessByGen);
		const existing = next.get(gen);
		if (existing === undefined || fitness > existing) {
			next.set(gen, fitness);
			fitnessByGen = next;
		}
	};

	const updateFitnessFromRows = (rows: Array<Record<string, unknown>>) => {
		for (const row of rows) {
			if (!row || typeof row !== "object") continue;
			const gen = toNumber(row[logKey]);
			const fitness = toNumber(row.fitness);
			if (gen !== null && fitness !== null) {
				updateFitness(gen, fitness);
			}
		}
	};

	const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

	const buildLogUrl = (dir: string, offset: number) => {
		const params = new URLSearchParams({
			limit: String(logChunk),
			offset: String(offset),
			dir,
			log: logType
		});
		return `/api/logs?${params.toString()}`;
	};

	const fetchLogPayload = async (dir: string, offset: number) => {
		const res = await fetch(buildLogUrl(dir, offset));
		if (res.status === 404) {
			return { rows: [], nextOffset: offset, done: false, notFound: true };
		}
		if (!res.ok) throw new Error(`Log fetch failed (${res.status})`);
		const payload = await res.json();
		const rows = Array.isArray(payload.data) ? payload.data : [];
		const nextOffset =
			typeof payload.nextOffset === "number" ? payload.nextOffset : offset + rows.length;
		const done = Boolean(payload.done);
		return {
			rows: rows as Array<Record<string, unknown>>,
			nextOffset,
			done,
			notFound: false
		};
	};

	const readLogChunk = async (dir: string) => {
		const payload = await fetchLogPayload(dir, logOffset);
		if (payload.rows.length > 0) {
			logOffset = payload.nextOffset;
		}
		return payload;
	};

	const drainLogs = async (dir: string) => {
		let emptyReads = 0;
		while (emptyReads < 5) {
			const { rows } = await readLogChunk(dir);
			if (rows.length > 0) {
				updateFitnessFromRows(rows);
				emptyReads = 0;
				continue;
			}
			emptyReads += 1;
			await sleep(200);
		}
	};

	const rebuildFitnessFromLogs = async (dir: string) => {
		fitnessByGen = new Map();
		logOffset = 0;
		try {
			const params = new URLSearchParams({ dir, mode: "summary", log: logType, key: logKey });
			const res = await fetch(`/api/logs?${params.toString()}`);
			if (!res.ok) throw new Error(`Log summary failed (${res.status})`);
			const payload = await res.json();
			const rows = Array.isArray(payload.data) ? payload.data : [];
			updateFitnessFromRows(rows);
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: "error", text: message }];
		}
	};

	const pollLogs = async () => {
		if (!logPolling) return;
		try {
			const { rows, notFound } = await readLogChunk(activeLogDir);
			if (rows.length > 0) {
				updateFitnessFromRows(rows);
				logPollDelay = logPollInterval;
			} else {
				logPollDelay = Math.min(logPollDelay + (notFound ? 1000 : 500), logPollMaxInterval);
			}
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			consoleOutput = [...consoleOutput, { type: "error", text: message }];
			stopLogPolling();
		} finally {
			if (logPolling) {
				logTimer = setTimeout(pollLogs, logPollDelay);
			}
		}
	};

	const startLogPolling = (dir: string) => {
		const fallbackDir = trainMode === "rl" ? "runs_rl" : "runs_ga";
		activeLogDir = dir.trim() || fallbackDir;
		logOffset = 0;
		logPolling = true;
		logPollDelay = logPollInterval;
		if (logTimer) clearTimeout(logTimer);
		void pollLogs();
	};

	const stopLogPolling = () => {
		logPolling = false;
		if (logTimer) clearTimeout(logTimer);
		logTimer = null;
	};

	const selectStartChoice = (choice: StartChoice) => {
		if (training) return;
		startChoice = choice;
		if (choice === "new") {
			checkpointPath = "";
		}
	};

	const resetTrainingSetup = () => {
		if (training) return;
		startChoice = null;
		checkpointPath = "";
	};

	const attachCheckpoint = (params: Record<string, unknown>) => {
		if (startChoice === "resume") {
			const trimmed = checkpointPath.trim();
			if (trimmed) {
				params["load-checkpoint"] = trimmed;
			}
		}
		return params;
	};

	const buildGaParams = () => {
		const params = { ...gaParams } as Record<string, unknown>;
		if (gaDataMode === "full") {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return attachCheckpoint(params);
	};

	const buildRlParams = () => {
		const params = { ...rlParams } as Record<string, unknown>;
		params.algorithm = rlAlgorithm;
		if (params.backend !== "candle" && params.backend !== "libtorch") {
			delete params.dropout;
		}
		if (rlDataMode === "full") {
			params["full-file"] = true;
		} else {
			params.windowed = true;
		}
		return attachCheckpoint(params);
	};

	const toggleTraining = () => {
		if (training) {
			stopTraining();
		} else {
			if (!canStartTraining) {
				const message = startChoice
					? "Add a checkpoint path to resume training."
					: "Choose new training or resume from a checkpoint first.";
				consoleOutput = [...consoleOutput, { type: "error", text: message }];
				return;
			}
			void startTraining(trainMode);
		}
	};

	async function startTraining(mode: TrainMode) {
		if (training) return;
		if (!startChoice) {
			consoleOutput = [
				...consoleOutput,
				{ type: "error", text: "Choose new training or resume from a checkpoint first." }
			];
			return;
		}
		if (startChoice === "resume" && !checkpointPath.trim()) {
			consoleOutput = [
				...consoleOutput,
				{ type: "error", text: "Checkpoint path is required to resume." }
			];
			return;
		}
		const params = mode === "ga" ? buildGaParams() : buildRlParams();
		const backend = (typeof params.backend === "string" ? params.backend : "libtorch") as MlBackend;
		const fallbackDir = trainMode === "rl" ? "runs_rl" : "runs_ga";
		const outdir = typeof params.outdir === "string" ? params.outdir : fallbackDir;
		abortController?.abort();
		abortController = new AbortController();
		training = true;
		const verb = startChoice === "resume" ? "Resuming" : "Starting";
		const suffix = startChoice === "resume" ? " from checkpoint" : "";
		const algoInfo = mode === "rl" ? ` (${rlAlgorithm.toUpperCase()})` : "";
		consoleOutput = [
			{
				type: "system",
				text: `${verb} ${mode.toUpperCase()}${algoInfo} training on ${String(backend).toUpperCase()} using ${buildProfile.toUpperCase()} build${suffix}...`
			}
		];
		fitnessByGen = new Map();
		if (liveLogUpdates) {
			startLogPolling(outdir);
		} else {
			activeLogDir = outdir;
			logOffset = 0;
		}

		try {
			const response = await fetch("/api/train", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ engine: mode, params, profile: buildProfile }),
				signal: abortController.signal
			});

			if (!response.ok || !response.body) {
				throw new Error(`Training request failed (${response.status})`);
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();

			while (true) {
				const { value, done } = await reader.read();
				if (done) break;

				const chunk = decoder.decode(value);
				const lines = chunk.split("\n\n");

				for (const line of lines) {
					if (!line.startsWith("data: ")) continue;
					const data = JSON.parse(line.substring(6));
					if (data.type === "stdout" || data.type === "stderr") {
						consoleOutput = [...consoleOutput, { type: data.type, text: data.content }];
						const runDirMatch = data.content.match(/info: run directory (.+)/);
						if (runDirMatch) {
							const actualDir = runDirMatch[1].trim();
							activeLogDir = actualDir;
							if (liveLogUpdates) {
								stopLogPolling();
								startLogPolling(actualDir);
							}
						}
					} else if (data.type === "error") {
						consoleOutput = [...consoleOutput, { type: "error", text: data.content }];
					} else if (data.type === "exit") {
						training = false;
						abortController = null;
						stopLogPolling();
						if (liveLogUpdates) {
							await drainLogs(activeLogDir);
						}
						await sleep(300);
						await rebuildFitnessFromLogs(activeLogDir);
						consoleOutput = [
							...consoleOutput,
							{ type: "system", text: `Process exited with code ${data.code}` }
						];
					}
				}
			}
		} catch (e) {
			if (e instanceof DOMException && e.name === "AbortError") {
				consoleOutput = [...consoleOutput, { type: "system", text: "Training stopped by user." }];
			} else {
				const message = e instanceof Error ? e.message : String(e);
				consoleOutput = [...consoleOutput, { type: "error", text: message }];
			}
			training = false;
			abortController = null;
			stopLogPolling();
		}
	}

	const stopTraining = () => {
		if (!training) return;
		abortController?.abort();
		stopLogPolling();
	};

	const handleCollapsedAction = () => {
		if (training) {
			stopTraining();
		} else {
			paramsCollapsed = false;
		}
	};

	const handleBrowseCheckpoint = () => openCheckpointPicker(trainMode);
</script>

<div class="min-h-screen bg-background">
	<TrainSidebar
		bind:paramsCollapsed
		{training}
		{startChoice}
		bind:buildProfile
		bind:checkpointPath
		bind:trainMode
		bind:rlAlgorithm
		bind:gaDataMode
		bind:rlDataMode
		{gaParams}
		{rlParams}
		{diagnosticsLoading}
		bind:liveLogUpdates
		{canStartTraining}
		{runLabel}
		{runTitle}
		{trimmedCheckpoint}
		{collapsedLabel}
		{collapsedTitle}
		onResetTrainingSetup={resetTrainingSetup}
		onSelectStartChoice={selectStartChoice}
		onRunDiagnostics={runDiagnostics}
		onBrowseCheckpoint={handleBrowseCheckpoint}
		onBrowseParquet={openParquetPicker}
		onApplyFuturesPreset={applyFuturesPreset}
		onToggleTraining={toggleTraining}
		onCollapsedAction={handleCollapsedAction}
		onSubmitGa={() => void startTraining("ga")}
		onSubmitRl={() => void startTraining("rl")}
	/>

	<main class={`p-8 space-y-8 ${paramsCollapsed ? "lg:ml-[120px]" : "lg:ml-[360px]"}`}>
		<TrainPageHeader {trainMode} />

		<div class="space-y-6">
			<TrainFitnessCard
				{training}
				{logLabel}
				{fitnessSeries}
				chartData={fitnessChartData}
				chartOptions={fitnessChartOptions}
			/>
			<TrainConsoleCard {training} {consoleOutput} />
			<TrainDiagnosticsCard
				loading={diagnosticsLoading}
				output={diagnosticsOutput}
				error={diagnosticsError}
				env={diagnosticsEnv}
			/>
		</div>
	</main>

	<TrainFileBrowserModal
		open={fileBrowserOpen}
		title={fileBrowserTitle}
		dir={fileBrowserDir}
		extensions={fileBrowserExtensions}
		parent={fileBrowserParent}
		entries={fileBrowserEntries}
		loading={fileBrowserLoading}
		error={fileBrowserError}
		emptyMessage={fileBrowserEmptyMessage}
		onClose={closeFileBrowser}
		onUp={handleFileBrowserUp}
		onSelect={handleFileEntry}
	/>
</div>

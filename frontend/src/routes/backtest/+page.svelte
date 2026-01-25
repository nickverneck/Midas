<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import * as Select from "$lib/components/ui/select";
	import { Badge } from "$lib/components/ui/badge";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { Separator } from "$lib/components/ui/separator";
	import GaChart from "$lib/components/GaChart.svelte";
	import { FileCode, Gauge, LineChart, Play, Shield } from "lucide-svelte";

	const sampleScript = `-- Example: EMA cross strategy
-- Return one of: "buy", "sell", "revert", "hold"

local fast = 10
local slow = 30

function on_bar(ctx, bar)
  if bar["ema_" .. fast] ~= bar["ema_" .. fast] then
    return "hold"
  end

  if bar["ema_" .. fast] > bar["ema_" .. slow] then
    if ctx.position <= 0 then
      return ctx.position == 0 and "buy" or "revert"
    end
  else
    if ctx.position >= 0 then
      return ctx.position == 0 and "sell" or "revert"
    end
  end

  return "hold"
end
`;

	type DatasetMode = "train" | "val" | "custom";
	let datasetMode = $state<DatasetMode>("train");
	let datasetPath = $state("data/train/SPY0.parquet");

	let scriptText = $state(sampleScript);

	type BacktestMetrics = {
		total_reward: number;
		net_pnl: number;
		ending_equity: number;
		sharpe: number;
		max_drawdown: number;
		profit_factor: number;
		win_rate: number;
		max_consecutive_losses: number;
		steps: number;
	};

	type BacktestResult = {
		metrics: BacktestMetrics;
		equity_curve: number[];
		actions?: { idx: number; action: string }[];
	};

	let runResult = $state<BacktestResult | null>(null);
	let running = $state(false);
	let runError = $state("");

	const toNumber = (value: unknown) => {
		if (value === null || value === undefined || value === "") return null;
		const num = Number(value);
		return Number.isFinite(num) ? num : null;
	};

	let env = $state({
		initialBalance: 10_000,
		maxPosition: 1,
		commission: 1.6,
		slippage: 0.25,
		marginPerContract: 50,
		marginMode: "per-contract",
		contractMultiplier: 1.0,
		enforceMargin: true
	});

	let limits = $state({
		memoryMb: 64,
		instructionLimit: 5_000_000,
		instructionInterval: 10_000
	});

	let numericEnv = $derived.by(() => ({
		initialBalance: toNumber(env.initialBalance),
		maxPosition: toNumber(env.maxPosition),
		commission: toNumber(env.commission),
		slippage: toNumber(env.slippage),
		marginPerContract: toNumber(env.marginPerContract),
		contractMultiplier: toNumber(env.contractMultiplier)
	}));

	let numericLimits = $derived.by(() => ({
		memoryMb: toNumber(limits.memoryMb),
		instructionLimit: toNumber(limits.instructionLimit),
		instructionInterval: toNumber(limits.instructionInterval)
	}));

	let scriptInvalid = $derived.by(() => scriptText.trim().length === 0);
	let datasetPathInvalid = $derived.by(() => {
		if (datasetMode !== "custom") return false;
		const trimmed = datasetPath.trim().toLowerCase();
		if (!trimmed) return true;
		return !trimmed.endsWith(".parquet");
	});

	let invalidInitialBalance = $derived.by(
		() => numericEnv.initialBalance === null || numericEnv.initialBalance <= 0
	);
	let invalidMaxPosition = $derived.by(
		() => numericEnv.maxPosition === null || numericEnv.maxPosition < 0
	);
	let invalidCommission = $derived.by(
		() => numericEnv.commission === null || numericEnv.commission < 0
	);
	let invalidSlippage = $derived.by(
		() => numericEnv.slippage === null || numericEnv.slippage < 0
	);
	let invalidMargin = $derived.by(
		() => numericEnv.marginPerContract === null || numericEnv.marginPerContract < 0
	);
	let invalidContractMultiplier = $derived.by(
		() => numericEnv.contractMultiplier === null || numericEnv.contractMultiplier <= 0
	);
	let invalidMemory = $derived.by(
		() => numericLimits.memoryMb === null || numericLimits.memoryMb < 0
	);
	let invalidInstructionLimit = $derived.by(
		() => numericLimits.instructionLimit === null || numericLimits.instructionLimit < 0
	);
	let invalidInstructionInterval = $derived.by(
		() => numericLimits.instructionInterval === null || numericLimits.instructionInterval <= 0
	);

	let validationErrors = $derived.by(() => {
		const errors: string[] = [];
		if (scriptInvalid) errors.push("Script is required.");
		if (datasetPathInvalid) {
			errors.push("Custom dataset path must be a .parquet file.");
		}
		if (invalidInitialBalance) errors.push("Initial balance must be greater than 0.");
		if (invalidMaxPosition) errors.push("Max position must be 0 or higher.");
		if (invalidCommission) errors.push("Commission must be 0 or higher.");
		if (invalidSlippage) errors.push("Slippage must be 0 or higher.");
		if (invalidMargin) errors.push("Margin per contract must be 0 or higher.");
		if (invalidContractMultiplier) errors.push("Contract multiplier must be greater than 0.");
		if (invalidMemory) errors.push("Memory limit must be 0 or higher.");
		if (invalidInstructionLimit) errors.push("Instruction limit must be 0 or higher.");
		if (invalidInstructionInterval) errors.push("Instruction interval must be greater than 0.");
		return errors;
	});

	let canRun = $derived.by(() => !running && validationErrors.length === 0);

	$effect(() => {
		if (datasetMode === "train") {
			datasetPath = "data/train/SPY0.parquet";
		} else if (datasetMode === "val") {
			datasetPath = "data/val/SPY.parquet";
		}
	});

	const demoEquitySeries = Array.from({ length: 140 }, (_, i) => {
		const drift = i * 3.2;
		const wave = Math.sin(i / 9) * 40;
		return 10_000 + drift + wave;
	});

	const demoBaselineSeries = demoEquitySeries.map((v, i) => v - 120 + Math.cos(i / 11) * 25);

	const demoMetrics = {
		total_reward: 0,
		net_pnl: 412.36,
		ending_equity: 10_412.36,
		sharpe: 1.48,
		max_drawdown: 132.4,
		profit_factor: 1.32,
		win_rate: 0.54,
		max_consecutive_losses: 4,
		steps: 13_892
	};

	let equitySeries = $derived.by(() => runResult?.equity_curve ?? demoEquitySeries);
	let activeMetrics = $derived.by(() => runResult?.metrics ?? demoMetrics);

	let equityChartData = $derived.by(() => {
		const datasets = [
			{
				label: "Script",
				data: equitySeries,
				borderColor: "var(--color-chart-2)",
				backgroundColor: "rgba(59, 130, 246, 0.12)",
				fill: true,
				borderWidth: 2,
				tension: 0.25
			}
		];

		if (!runResult) {
			datasets.push({
				label: "Model",
				data: demoBaselineSeries,
				borderColor: "var(--color-chart-4)",
				backgroundColor: "transparent",
				borderDash: [6, 4],
				borderWidth: 2,
				tension: 0.2
			});
		}

		return {
			labels: equitySeries.map((_, i) => i + 1),
			datasets
		};
	});

	const chartOptions = {
		plugins: {
			legend: {
				display: true,
				position: "bottom"
			}
		},
		scales: {
			x: {
				display: false
			},
			y: {
				ticks: {
					callback: (value: number) => `$${value.toLocaleString()}`
				}
			}
		}
	};

	const runBacktest = async () => {
		if (validationErrors.length > 0) {
			runError = "Fix validation errors before running.";
			return;
		}
		running = true;
		runError = "";
		try {
			const payload = {
				dataset: datasetMode === "custom" ? null : datasetMode,
				path: datasetMode === "custom" ? datasetPath : null,
				script: scriptText,
				env: {
					initialBalance: numericEnv.initialBalance,
					maxPosition: numericEnv.maxPosition,
					commission: numericEnv.commission,
					slippage: numericEnv.slippage,
					marginPerContract: numericEnv.marginPerContract,
					marginMode: env.marginMode,
					contractMultiplier: numericEnv.contractMultiplier,
					enforceMargin: env.enforceMargin
				},
				limits: {
					memoryMb: numericLimits.memoryMb,
					instructionLimit: numericLimits.instructionLimit,
					instructionInterval: numericLimits.instructionInterval
				}
			};

			const res = await fetch("/api/backtest", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data?.error || `Backtest failed (${res.status})`);
			}
			runResult = data as BacktestResult;
		} catch (err) {
			runError = err instanceof Error ? err.message : String(err);
		} finally {
			running = false;
		}
	};
</script>

<main class="mx-auto max-w-6xl space-y-8 px-6 py-8">
	<section class="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-slate-50 via-white to-slate-50 p-8">
		<div class="flex flex-wrap items-center gap-2">
			<Badge variant="secondary">Lua 5.4</Badge>
			<Badge variant="secondary">Local Parquet</Badge>
			<Badge variant="secondary">Safe Runtime</Badge>
		</div>
		<div class="mt-4 space-y-2">
			<h1 class="text-3xl font-semibold tracking-tight">Scripted Backtests</h1>
			<p class="max-w-2xl text-sm text-muted-foreground">
				Design and iterate on Lua strategies with full transparency into signals, trades, and
				performance. Compare scripts against trained models on the same dataset and environment.
			</p>
		</div>
		<div class="absolute -right-16 -top-16 h-52 w-52 rounded-full bg-primary/10 blur-3xl"></div>
		<div class="absolute -bottom-20 -left-8 h-48 w-48 rounded-full bg-secondary/40 blur-3xl"></div>
	</section>

	<section class="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
		<div class="flex flex-col gap-6">
			<Card.Root>
				<Card.Header class="flex flex-row items-start justify-between">
					<div>
						<Card.Title class="flex items-center gap-2">
							<FileCode size={18} />
							Strategy Script
						</Card.Title>
						<Card.Description>Write or paste Lua here. The script runs bar-by-bar.</Card.Description>
					</div>
					<div class="flex items-center gap-2">
						<Button variant="secondary" size="sm" on:click={() => (scriptText = sampleScript)}>
							Reset Sample
						</Button>
						<Button variant="outline" size="sm" disabled>
							Load .lua
						</Button>
					</div>
				</Card.Header>
				<Card.Content>
					<textarea
						class={`min-h-[320px] w-full resize-none rounded-lg border bg-muted/30 px-4 py-3 font-mono text-xs leading-relaxed shadow-inner focus:outline-none focus:ring-2 ${
							scriptInvalid ? "border-destructive focus:ring-destructive/40" : "focus:ring-ring"
						}`}
						aria-invalid={scriptInvalid}
						bind:value={scriptText}
					></textarea>
					<div class="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
						<span>Return: buy · sell · revert · hold</span>
						<span class="h-1 w-1 rounded-full bg-muted-foreground"></span>
						<span>Indicators: EMA / SMA / HMA / ATR</span>
						<span class="h-1 w-1 rounded-full bg-muted-foreground"></span>
						<span>Context: position, cash, equity</span>
					</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title class="flex items-center gap-2">
						<Shield size={18} />
						Script API Surface
					</Card.Title>
					<Card.Description>Keep strategies deterministic and fast.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="grid gap-4 sm:grid-cols-2">
						<div class="rounded-lg border bg-muted/20 p-4">
							<p class="text-xs font-semibold uppercase text-muted-foreground">ctx</p>
							<ul class="mt-2 space-y-1 text-sm">
								<li>step</li>
								<li>position</li>
								<li>cash / equity</li>
								<li>unrealized / realized PnL</li>
							</ul>
						</div>
						<div class="rounded-lg border bg-muted/20 p-4">
							<p class="text-xs font-semibold uppercase text-muted-foreground">bar</p>
							<ul class="mt-2 space-y-1 text-sm">
								<li>ts / open / high / low / close</li>
								<li>volume (if present)</li>
								<li>sma_*, ema_*, hma_*, atr_*</li>
							</ul>
						</div>
					</div>
					<pre class="rounded-lg border bg-background px-4 py-3 text-xs text-muted-foreground">on_bar(ctx, bar) -&gt; "buy" | "sell" | "revert" | "hold"</pre>
				</Card.Content>
			</Card.Root>
		</div>

		<div class="flex flex-col gap-6">
			<Card.Root>
				<Card.Header>
					<Card.Title class="flex items-center gap-2">
						<LineChart size={18} />
						Dataset
					</Card.Title>
					<Card.Description>Select the parquet slice to evaluate.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="space-y-2">
						<Label>Dataset</Label>
						<Select.Root bind:value={datasetMode}>
							<Select.Trigger class="w-full">{datasetMode}</Select.Trigger>
							<Select.Content>
								<Select.Item value="train">train</Select.Item>
								<Select.Item value="val">val</Select.Item>
								<Select.Item value="custom">custom</Select.Item>
							</Select.Content>
						</Select.Root>
					</div>
					<div class="space-y-2">
						<Label>Path</Label>
						<Input
							bind:value={datasetPath}
							placeholder="data/train/SPY0.parquet"
							disabled={datasetMode !== "custom"}
							aria-invalid={datasetPathInvalid}
						/>
						{#if datasetMode === "custom"}
							<p class="text-xs text-muted-foreground">Provide a .parquet file path.</p>
						{/if}
					</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title class="flex items-center gap-2">
						<Gauge size={18} />
						Environment
					</Card.Title>
					<Card.Description>Match the training constraints for fair comparisons.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="grid gap-4 sm:grid-cols-2">
						<div class="space-y-2">
							<Label>Initial Balance</Label>
							<Input type="number" bind:value={env.initialBalance} aria-invalid={invalidInitialBalance} />
						</div>
						<div class="space-y-2">
							<Label>Max Position</Label>
							<Input type="number" bind:value={env.maxPosition} aria-invalid={invalidMaxPosition} />
						</div>
						<div class="space-y-2">
							<Label>Commission (round trip)</Label>
							<Input type="number" bind:value={env.commission} aria-invalid={invalidCommission} />
						</div>
						<div class="space-y-2">
							<Label>Slippage / Contract</Label>
							<Input type="number" bind:value={env.slippage} aria-invalid={invalidSlippage} />
						</div>
						<div class="space-y-2">
							<Label>Margin / Contract</Label>
							<Input type="number" bind:value={env.marginPerContract} aria-invalid={invalidMargin} />
						</div>
						<div class="space-y-2">
							<Label>Contract Multiplier</Label>
							<Input
								type="number"
								bind:value={env.contractMultiplier}
								aria-invalid={invalidContractMultiplier}
							/>
						</div>
						<div class="space-y-2">
							<Label>Margin Mode</Label>
							<Select.Root bind:value={env.marginMode}>
								<Select.Trigger class="w-full">{env.marginMode}</Select.Trigger>
								<Select.Content>
									<Select.Item value="per-contract">per-contract</Select.Item>
									<Select.Item value="price">price</Select.Item>
								</Select.Content>
							</Select.Root>
						</div>
					</div>
					<Separator />
					<div class="space-y-2">
						<Label>Safety Limits</Label>
						<div class="grid gap-3 sm:grid-cols-3">
							<Input
								type="number"
								bind:value={limits.memoryMb}
								placeholder="MB"
								aria-invalid={invalidMemory}
							/>
							<Input
								type="number"
								bind:value={limits.instructionLimit}
								placeholder="Instructions"
								aria-invalid={invalidInstructionLimit}
							/>
							<Input
								type="number"
								bind:value={limits.instructionInterval}
								placeholder="Check interval"
								aria-invalid={invalidInstructionInterval}
							/>
						</div>
						<p class="text-xs text-muted-foreground">Lua runtime caps for runaway scripts.</p>
					</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header>
					<Card.Title class="flex items-center gap-2">
						<Play size={18} />
						Run Backtest
					</Card.Title>
					<Card.Description>Execute the Lua script against the selected dataset.</Card.Description>
				</Card.Header>
				<Card.Content class="space-y-3">
					<Button class="w-full" on:click={runBacktest} disabled={!canRun}>
						{running ? "Running..." : "Run Script"}
					</Button>
					{#if validationErrors.length > 0}
						<div class="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
							<p class="font-medium">Fix before running:</p>
							<ul class="mt-1 space-y-1">
								{#each validationErrors as err}
									<li>• {err}</li>
								{/each}
							</ul>
						</div>
					{/if}
					{#if runError}
						<p class="text-xs text-destructive">{runError}</p>
					{/if}
					<p class="text-xs text-muted-foreground">
						{#if runResult}
							Latest run loaded. Adjust inputs and rerun to compare changes.
						{:else}
							Run a backtest to populate metrics and charts.
						{/if}
					</p>
				</Card.Content>
			</Card.Root>
		</div>
	</section>

	<section class="grid gap-6 lg:grid-cols-[1.35fr,0.65fr]">
		<Card.Root>
			<Card.Header>
				<Card.Title>Equity Curve</Card.Title>
				<Card.Description>
					{#if runResult}
						Latest script run.
					{:else}
						Demo preview (run a backtest to update).
					{/if}
				</Card.Description>
			</Card.Header>
			<Card.Content>
				<GaChart data={equityChartData} options={chartOptions} />
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="flex flex-row items-start justify-between">
				<div>
					<Card.Title>Performance Snapshot</Card.Title>
					<Card.Description>High-level stats for the latest run.</Card.Description>
				</div>
				{#if !runResult}
					<Badge variant="outline">Demo</Badge>
				{/if}
			</Card.Header>
			<Card.Content>
				<Table.Root>
					<Table.Body>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Net PnL</Table.Cell>
							<Table.Cell class="text-right font-medium">${activeMetrics.net_pnl.toFixed(2)}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Ending Equity</Table.Cell>
							<Table.Cell class="text-right font-medium">${activeMetrics.ending_equity.toFixed(2)}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Sharpe</Table.Cell>
							<Table.Cell class="text-right">{activeMetrics.sharpe.toFixed(2)}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Max Drawdown</Table.Cell>
							<Table.Cell class="text-right">${activeMetrics.max_drawdown.toFixed(2)}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Profit Factor</Table.Cell>
							<Table.Cell class="text-right">{activeMetrics.profit_factor.toFixed(2)}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Win Rate</Table.Cell>
							<Table.Cell class="text-right">{(activeMetrics.win_rate * 100).toFixed(1)}%</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Max Consec Losses</Table.Cell>
							<Table.Cell class="text-right">{activeMetrics.max_consecutive_losses}</Table.Cell>
						</Table.Row>
						<Table.Row>
							<Table.Cell class="text-muted-foreground">Steps</Table.Cell>
							<Table.Cell class="text-right">{activeMetrics.steps.toLocaleString()}</Table.Cell>
						</Table.Row>
					</Table.Body>
				</Table.Root>
				<Separator class="my-4" />
				<div class="space-y-2 text-xs text-muted-foreground">
					<p>Baseline comparison (GA/RL) will appear once metrics are wired in.</p>
					<div class="flex gap-2">
						<Badge variant="outline">GA latest</Badge>
						<Badge variant="outline">RL latest</Badge>
					</div>
				</div>
			</Card.Content>
		</Card.Root>
	</section>
</main>

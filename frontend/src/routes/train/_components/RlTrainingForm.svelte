<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import {
		detailsCardClass,
		detailsGridClass,
		futuresPresetSummary,
		nativeSelectClass
	} from "../constants";
	import type {
		DataMode,
		FuturesPresetKey,
		ParquetKey,
		RlAlgorithm,
		RlParams
	} from "../types";

	type Props = {
		params: RlParams;
		dataMode: DataMode;
		rlAlgorithm: RlAlgorithm;
		onBrowseParquet: (key: ParquetKey) => void;
		onApplyFuturesPreset: (presetKey: FuturesPresetKey) => void;
	};

	let {
		params,
		dataMode = $bindable(),
		rlAlgorithm = $bindable(),
		onBrowseParquet,
		onApplyFuturesPreset
	}: Props = $props();
</script>

<div class="space-y-4">
	<details class={detailsCardClass} open>
		<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-train-parquet">Train Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="rl-train-parquet"
						type="text"
						bind:value={params["train-parquet"]}
						placeholder="data/train"
					/>
					<Button type="button" variant="outline" onclick={() => onBrowseParquet("train-parquet")}>
						Browse
					</Button>
				</div>
			</div>
			<div class="grid gap-2">
				<Label for="rl-val-parquet">Val Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="rl-val-parquet"
						type="text"
						bind:value={params["val-parquet"]}
						placeholder="data/val"
					/>
					<Button type="button" variant="outline" onclick={() => onBrowseParquet("val-parquet")}>
						Browse
					</Button>
				</div>
			</div>
			<div class="grid gap-2">
				<Label for="rl-test-parquet">Test Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="rl-test-parquet"
						type="text"
						bind:value={params["test-parquet"]}
						placeholder="data/test"
					/>
					<Button type="button" variant="outline" onclick={() => onBrowseParquet("test-parquet")}>
						Browse
					</Button>
				</div>
			</div>
			<div class="grid gap-2">
				<Label for="rl-data-mode">Data Mode</Label>
				<select id="rl-data-mode" bind:value={dataMode} class={nativeSelectClass}>
					<option value="windowed">Windowed</option>
					<option value="full">Full file</option>
				</select>
			</div>
			{#if dataMode === "windowed"}
				<div class="grid gap-2">
					<Label for="rl-window">Window Size</Label>
					<Input id="rl-window" type="number" min="1" bind:value={params.window} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-step">Step Size</Label>
					<Input id="rl-step" type="number" min="1" bind:value={params.step} />
				</div>
			{/if}
		</div>
	</details>

	<details class={detailsCardClass} open>
		<summary class="cursor-pointer text-sm font-semibold">Hardware</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-backend">Backend</Label>
				<select id="rl-backend" bind:value={params.backend} class={nativeSelectClass}>
					<option value="libtorch">libtorch</option>
					<option value="burn">burn</option>
					<option value="candle">candle</option>
					<option value="mlx">mlx</option>
				</select>
			</div>
			<div class="grid gap-2">
				<Label for="rl-device">Device</Label>
				<select id="rl-device" bind:value={params.device} class={nativeSelectClass}>
					<option value="auto">Auto</option>
					<option value="cpu">CPU</option>
					<option value="mps">MPS</option>
					<option value="cuda">CUDA</option>
				</select>
			</div>
			<div class="grid gap-2 rounded-md border border-dashed p-3 text-xs text-muted-foreground md:col-span-2">
				<div class="font-medium uppercase tracking-wide text-foreground">Backend rollout status</div>
				{#if params.backend === "libtorch"}
					<div>`libtorch` is the active implementation. `Auto` prefers CUDA, then MPS, then CPU.</div>
				{:else if params.backend === "burn"}
					<div>`burn` is implemented for GA in this branch, but the RL runner is still not implemented yet.</div>
				{:else if params.backend === "candle"}
					<div>`candle` is implemented for RL PPO and GRPO in this branch. Use `cpu` now, enable the Candle CUDA Cargo feature on the Linux box when you want to benchmark GPU, and keep MLX for Apple GPU viability.</div>
				{:else}
					<div>`mlx` is reserved for a separate runner path so Apple-dev and Linux-CUDA benchmarking can slot into the same workflow later.</div>
				{/if}
			</div>
		</div>
	</details>

	<details class={detailsCardClass} open>
		<summary class="cursor-pointer text-sm font-semibold">Algorithm</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-algorithm">RL Algorithm</Label>
				<select id="rl-algorithm" bind:value={rlAlgorithm} class={nativeSelectClass}>
					<option value="ppo">PPO (Proximal Policy Optimization)</option>
					<option value="grpo">GRPO (Group Relative Policy Optimization)</option>
				</select>
				<p class="text-xs text-muted-foreground">
					{rlAlgorithm === "ppo"
						? "Uses actor-critic with value network"
						: "Group-based, no value network required"}
				</p>
			</div>
		</div>
	</details>

	{#if rlAlgorithm === "ppo"}
		<details class={detailsCardClass}>
			<summary class="cursor-pointer text-sm font-semibold">PPO Training</summary>
			<div class={detailsGridClass}>
				<div class="grid gap-2">
					<Label for="rl-epochs">Epochs</Label>
					<Input id="rl-epochs" type="number" min="1" bind:value={params.epochs} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-train-windows">Train Windows</Label>
					<Input id="rl-train-windows" type="number" min="0" bind:value={params["train-windows"]} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-ppo-epochs">PPO Epochs</Label>
					<Input id="rl-ppo-epochs" type="number" min="1" bind:value={params["ppo-epochs"]} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-eval-windows">Eval Windows</Label>
					<Input id="rl-eval-windows" type="number" min="1" bind:value={params["eval-windows"]} />
				</div>
			</div>
		</details>
	{/if}

	{#if rlAlgorithm === "grpo"}
		<details class={detailsCardClass}>
			<summary class="cursor-pointer text-sm font-semibold">GRPO Training</summary>
			<div class={detailsGridClass}>
				<div class="grid gap-2">
					<Label for="rl-epochs">Epochs</Label>
					<Input id="rl-epochs" type="number" min="1" bind:value={params.epochs} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-train-windows">Train Windows</Label>
					<Input id="rl-train-windows" type="number" min="0" bind:value={params["train-windows"]} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-group-size">Group Size</Label>
					<Input id="rl-group-size" type="number" min="2" bind:value={params["group-size"]} />
				</div>
				<div class="grid gap-2">
					<Label for="rl-eval-windows">Eval Windows</Label>
					<Input id="rl-eval-windows" type="number" min="1" bind:value={params["eval-windows"]} />
				</div>
			</div>
		</details>
	{/if}

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Optimization</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-lr">Learning Rate</Label>
				<Input id="rl-lr" type="number" step="0.0001" bind:value={params.lr} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-gamma">Gamma</Label>
				<Input id="rl-gamma" type="number" step="0.01" min="0" max="1" bind:value={params.gamma} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-lam">Lambda</Label>
				<Input id="rl-lam" type="number" step="0.01" min="0" max="1" bind:value={params.lam} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-clip">Clip Range</Label>
				<Input id="rl-clip" type="number" step="0.01" min="0" bind:value={params.clip} />
			</div>
			{#if rlAlgorithm === "ppo"}
				<div class="grid gap-2">
					<Label for="rl-vf-coef">Value Coef</Label>
					<Input id="rl-vf-coef" type="number" step="0.01" min="0" bind:value={params["vf-coef"]} />
				</div>
			{/if}
			<div class="grid gap-2">
				<Label for="rl-ent-coef">Entropy Coef</Label>
				<Input id="rl-ent-coef" type="number" step="0.01" min="0" bind:value={params["ent-coef"]} />
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Model &amp; Fitness</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-hidden">Hidden Units</Label>
				<Input id="rl-hidden" type="number" min="1" bind:value={params.hidden} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-layers">Layers</Label>
				<Input id="rl-layers" type="number" min="1" bind:value={params.layers} />
			</div>
			{#if params.backend === "candle" || params.backend === "libtorch"}
				<div class="grid gap-2">
					<Label for="rl-dropout">Dropout</Label>
					<Input id="rl-dropout" type="number" min="0" max="0.95" step="0.05" bind:value={params.dropout} />
					<div class="text-xs text-muted-foreground">
						Libtorch and Candle only. Applied after each hidden-layer activation during training; eval and test disable it.
					</div>
				</div>
			{/if}
			<div class="grid gap-2">
				<Label for="rl-initial-balance">Initial Balance</Label>
				<Input
					id="rl-initial-balance"
					type="number"
					step="0.01"
					bind:value={params["initial-balance"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-max-position">Max Position (0 = no cap)</Label>
				<Input id="rl-max-position" type="number" min="0" bind:value={params["max-position"]} />
			</div>
			<div class="grid gap-2 rounded-md border border-dashed p-3 md:col-span-2">
				<div class="text-xs font-medium uppercase tracking-wide text-muted-foreground">
					Futures Presets (NinjaTrader)
				</div>
				<div class="flex flex-wrap gap-2">
					<Button
						type="button"
						size="sm"
						variant="outline"
						onclick={() => onApplyFuturesPreset("mes-micro")}
					>
						MES Micro
					</Button>
					<Button
						type="button"
						size="sm"
						variant="outline"
						onclick={() => onApplyFuturesPreset("es-mini")}
					>
						ES Mini
					</Button>
				</div>
				<p class="text-xs text-muted-foreground">{futuresPresetSummary}</p>
			</div>
			<div class="grid gap-2">
				<Label for="rl-margin-mode">Margin Mode</Label>
				<select id="rl-margin-mode" bind:value={params["margin-mode"]} class={nativeSelectClass}>
					<option value="auto">Auto</option>
					<option value="per-contract">Per-contract</option>
					<option value="price">Price-based</option>
				</select>
			</div>
			<div class="grid gap-2">
				<Label for="rl-contract-multiplier">Contract Multiplier</Label>
				<Input
					id="rl-contract-multiplier"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["contract-multiplier"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-margin-per-contract">Margin Per Contract</Label>
				<Input
					id="rl-margin-per-contract"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["margin-per-contract"]}
					placeholder="auto from config"
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-w-pnl">Fitness Weight (PNL)</Label>
				<Input id="rl-w-pnl" type="number" step="0.01" bind:value={params["w-pnl"]} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-w-sortino">Fitness Weight (Sortino)</Label>
				<Input id="rl-w-sortino" type="number" step="0.01" bind:value={params["w-sortino"]} />
			</div>
			<div class="grid gap-2">
				<Label for="rl-w-mdd">Fitness Weight (Max DD)</Label>
				<Input id="rl-w-mdd" type="number" step="0.01" bind:value={params["w-mdd"]} />
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Position Hold Penalties</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-max-hold-bars-positive">Max Hold Bars (Profit)</Label>
				<Input
					id="rl-max-hold-bars-positive"
					type="number"
					min="0"
					bind:value={params["max-hold-bars-positive"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-max-hold-bars-drawdown">Max Hold Bars (Drawdown)</Label>
				<Input
					id="rl-max-hold-bars-drawdown"
					type="number"
					min="0"
					bind:value={params["max-hold-bars-drawdown"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-hold-duration-penalty">Hold Penalty</Label>
				<Input
					id="rl-hold-duration-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-hold-duration-penalty-growth">Hold Penalty Growth</Label>
				<Input
					id="rl-hold-duration-penalty-growth"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-growth"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-hold-duration-penalty-positive-scale">Hold Penalty Scale (Profit)</Label>
				<Input
					id="rl-hold-duration-penalty-positive-scale"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-positive-scale"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-hold-duration-penalty-negative-scale">Hold Penalty Scale (Loss)</Label>
				<Input
					id="rl-hold-duration-penalty-negative-scale"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-negative-scale"]}
				/>
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Output &amp; Checkpoints</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="rl-outdir">Output Folder</Label>
				<Input id="rl-outdir" type="text" bind:value={params.outdir} placeholder="runs_rl" />
			</div>
			<div class="grid gap-2">
				<Label for="rl-log-interval">Log Interval (epochs)</Label>
				<Input
					id="rl-log-interval"
					type="number"
					min="1"
					bind:value={params["log-interval"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="rl-checkpoint-every">Checkpoint Every (epochs)</Label>
				<Input
					id="rl-checkpoint-every"
					type="number"
					min="0"
					bind:value={params["checkpoint-every"]}
				/>
			</div>
		</div>
	</details>
</div>

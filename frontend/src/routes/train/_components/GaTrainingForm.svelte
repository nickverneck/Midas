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
	import type { DataMode, FuturesPresetKey, GaParams, ParquetKey } from "../types";

	type Props = {
		params: GaParams;
		dataMode: DataMode;
		onBrowseParquet: (key: ParquetKey) => void;
		onApplyFuturesPreset: (presetKey: FuturesPresetKey) => void;
	};

	let { params, dataMode = $bindable(), onBrowseParquet, onApplyFuturesPreset }: Props = $props();
</script>

<div class="space-y-4">
	<details class={detailsCardClass} open>
		<summary class="cursor-pointer text-sm font-semibold">Train Data</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-train-parquet">Train Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="ga-train-parquet"
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
				<Label for="ga-val-parquet">Val Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="ga-val-parquet"
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
				<Label for="ga-test-parquet">Test Parquet</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="ga-test-parquet"
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
				<Label for="ga-data-mode">Data Mode</Label>
				<select id="ga-data-mode" bind:value={dataMode} class={nativeSelectClass}>
					<option value="windowed">Windowed</option>
					<option value="full">Full file</option>
				</select>
			</div>
			{#if dataMode === "windowed"}
				<div class="grid gap-2">
					<Label for="ga-window">Window Size</Label>
					<Input id="ga-window" type="number" min="1" bind:value={params.window} />
				</div>
				<div class="grid gap-2">
					<Label for="ga-step">Step Size</Label>
					<Input id="ga-step" type="number" min="1" bind:value={params.step} />
				</div>
			{/if}
		</div>
	</details>

	<details class={detailsCardClass} open>
		<summary class="cursor-pointer text-sm font-semibold">Hardware</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-backend">Backend</Label>
				<select id="ga-backend" bind:value={params.backend} class={nativeSelectClass}>
					<option value="libtorch">libtorch</option>
					<option value="burn">burn</option>
					<option value="candle">candle</option>
					<option value="mlx">mlx</option>
				</select>
			</div>
			<div class="grid gap-2">
				<Label for="ga-device">Device</Label>
				<select id="ga-device" bind:value={params.device} class={nativeSelectClass}>
					<option value="auto">Auto</option>
					<option value="cpu">CPU</option>
					<option value="mps">MPS</option>
					<option value="cuda">CUDA</option>
				</select>
			</div>
			<div class="grid gap-2">
				<Label for="ga-batch-candidates">Batch Candidates</Label>
				<Input
					id="ga-batch-candidates"
					type="number"
					min="0"
					bind:value={params["batch-candidates"]}
				/>
				<p class="text-xs text-muted-foreground">0 uses auto batching.</p>
			</div>
			<div class="grid gap-2 rounded-md border border-dashed p-3 text-xs text-muted-foreground md:col-span-2">
				<div class="font-medium uppercase tracking-wide text-foreground">Backend rollout status</div>
				{#if params.backend === "libtorch"}
					<div>`libtorch` is the active implementation. `Auto` prefers CUDA, then MPS, then CPU.</div>
				{:else if params.backend === "burn"}
					<div>`burn` is implemented for GA in this branch. Use `cpu` for the Burn CPU backend, enable the Burn CUDA Cargo feature on the Linux box when you want native CUDA, select `mps` on macOS when the burn-mlx toolchain is installed, and set `MIDAS_BURN_CPU_BACKEND=ndarray` only when you want the legacy CPU path.</div>
				{:else if params.backend === "candle"}
					<div>`candle` is implemented for GA in this branch. Use `cpu` now, enable the Candle CUDA Cargo feature on the Linux box when you want to benchmark GPU, and keep MLX for Apple GPU viability.</div>
				{:else}
					<div>`mlx` is reserved for a separate runner path so Apple-dev and Linux-CUDA benchmarking can slot into the same workflow later.</div>
				{/if}
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Evolution Settings</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-generations">Generations</Label>
				<Input id="ga-generations" type="number" min="1" bind:value={params.generations} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-pop-size">Population Size</Label>
				<Input id="ga-pop-size" type="number" min="1" bind:value={params["pop-size"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-workers">Workers</Label>
				<Input id="ga-workers" type="number" min="0" bind:value={params.workers} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-elite-frac">Elite Fraction</Label>
				<Input
					id="ga-elite-frac"
					type="number"
					step="0.01"
					min="0"
					max="1"
					bind:value={params["elite-frac"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-mutation-sigma">Mutation Sigma</Label>
				<Input
					id="ga-mutation-sigma"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["mutation-sigma"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-init-sigma">Init Sigma</Label>
				<Input id="ga-init-sigma" type="number" step="0.01" min="0" bind:value={params["init-sigma"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-eval-windows">Eval Windows</Label>
				<Input id="ga-eval-windows" type="number" min="1" bind:value={params["eval-windows"]} />
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Model &amp; Fitness</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-hidden">Hidden Units</Label>
				<Input id="ga-hidden" type="number" min="1" bind:value={params.hidden} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-layers">Layers</Label>
				<Input id="ga-layers" type="number" min="1" bind:value={params.layers} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-initial-balance">Initial Balance</Label>
				<Input
					id="ga-initial-balance"
					type="number"
					step="0.01"
					bind:value={params["initial-balance"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-max-position">Max Position (0 = no cap)</Label>
				<Input id="ga-max-position" type="number" min="0" bind:value={params["max-position"]} />
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
				<Label for="ga-margin-mode">Margin Mode</Label>
				<select id="ga-margin-mode" bind:value={params["margin-mode"]} class={nativeSelectClass}>
					<option value="auto">Auto</option>
					<option value="per-contract">Per-contract</option>
					<option value="price">Price-based</option>
				</select>
			</div>
			<div class="grid gap-2">
				<Label for="ga-contract-multiplier">Contract Multiplier</Label>
				<Input
					id="ga-contract-multiplier"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["contract-multiplier"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-margin-per-contract">Margin Per Contract</Label>
				<Input
					id="ga-margin-per-contract"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["margin-per-contract"]}
					placeholder="auto from config"
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-w-pnl">Fitness Weight (PNL)</Label>
				<Input id="ga-w-pnl" type="number" step="0.01" bind:value={params["w-pnl"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-w-sortino">Fitness Weight (Sortino)</Label>
				<Input id="ga-w-sortino" type="number" step="0.01" bind:value={params["w-sortino"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-w-mdd">Fitness Weight (Max DD)</Label>
				<Input id="ga-w-mdd" type="number" step="0.01" bind:value={params["w-mdd"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-drawdown-penalty">Drawdown Penalty</Label>
				<Input
					id="ga-drawdown-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["drawdown-penalty"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-drawdown-penalty-growth">Drawdown Penalty Growth</Label>
				<Input
					id="ga-drawdown-penalty-growth"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["drawdown-penalty-growth"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-session-close-penalty">Session Close Penalty</Label>
				<Input
					id="ga-session-close-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["session-close-penalty"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-selection-train-weight">Selection Weight (Train)</Label>
				<Input
					id="ga-selection-train-weight"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["selection-train-weight"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-selection-eval-weight">Selection Weight (Eval)</Label>
				<Input
					id="ga-selection-eval-weight"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["selection-eval-weight"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-selection-gap-penalty">Selection Gap Penalty</Label>
				<Input
					id="ga-selection-gap-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["selection-gap-penalty"]}
				/>
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Position Hold Penalties</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-max-hold-bars-positive">Max Hold Bars (Profit)</Label>
				<Input
					id="ga-max-hold-bars-positive"
					type="number"
					min="0"
					bind:value={params["max-hold-bars-positive"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-max-hold-bars-drawdown">Max Hold Bars (Drawdown)</Label>
				<Input
					id="ga-max-hold-bars-drawdown"
					type="number"
					min="0"
					bind:value={params["max-hold-bars-drawdown"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-hold-duration-penalty">Hold Penalty</Label>
				<Input
					id="ga-hold-duration-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-hold-duration-penalty-growth">Hold Penalty Growth</Label>
				<Input
					id="ga-hold-duration-penalty-growth"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-growth"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-hold-duration-penalty-positive-scale">Hold Penalty Scale (Profit)</Label>
				<Input
					id="ga-hold-duration-penalty-positive-scale"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-positive-scale"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-hold-duration-penalty-negative-scale">Hold Penalty Scale (Loss)</Label>
				<Input
					id="ga-hold-duration-penalty-negative-scale"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["hold-duration-penalty-negative-scale"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-flat-hold-penalty">Flat Hold Penalty</Label>
				<Input
					id="ga-flat-hold-penalty"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["flat-hold-penalty"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-flat-hold-penalty-growth">Flat Hold Penalty Growth</Label>
				<Input
					id="ga-flat-hold-penalty-growth"
					type="number"
					step="0.01"
					min="0"
					bind:value={params["flat-hold-penalty-growth"]}
				/>
			</div>
			<div class="grid gap-2">
				<Label for="ga-max-flat-hold-bars">Max Flat Hold Bars</Label>
				<Input
					id="ga-max-flat-hold-bars"
					type="number"
					min="0"
					bind:value={params["max-flat-hold-bars"]}
				/>
			</div>
		</div>
	</details>

	<details class={detailsCardClass}>
		<summary class="cursor-pointer text-sm font-semibold">Output &amp; Checkpoints</summary>
		<div class={detailsGridClass}>
			<div class="grid gap-2">
				<Label for="ga-outdir">Output Folder</Label>
				<Input id="ga-outdir" type="text" bind:value={params.outdir} placeholder="runs_ga" />
			</div>
			<div class="grid gap-2">
				<Label for="ga-save-top-n">Save Top N</Label>
				<Input id="ga-save-top-n" type="number" min="0" bind:value={params["save-top-n"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-save-every">Save Every (gens)</Label>
				<Input id="ga-save-every" type="number" min="0" bind:value={params["save-every"]} />
			</div>
			<div class="grid gap-2">
				<Label for="ga-checkpoint-every">Checkpoint Every (gens)</Label>
				<Input
					id="ga-checkpoint-every"
					type="number"
					min="0"
					bind:value={params["checkpoint-every"]}
				/>
			</div>
		</div>
	</details>
</div>

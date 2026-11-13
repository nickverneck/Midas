<script lang="ts">
	type Axis = { path: string; values: Array<string | number | boolean | null> };
	type Candidate = { candidateId: string; runIndex?: number; status: string; value: number | null; error?: string | null; parameters: Record<string, unknown> };
	type Cell = { x: Axis['values'][number]; y: Axis['values'][number]; planned: boolean; candidates: Candidate[] };
	type Model = { sweepId: string | null; axes: Axis[]; selectedAxes: { x: string; y: string }; metrics: Array<{ key: string; label: string }>; selectedMetric: string; cells: Array<{ y: Cell['y']; cells: Cell[] }>; counts: { planned: number; completed: number; failed: number; skipped: number } };
	export let jobId = '';
	let model: Model | undefined;
	let error = '';
	let loading = false;
	let selected: Cell | undefined;
	let requestVersion = 0;

	async function load(x?: string, y?: string, metric?: string) {
		const version = ++requestVersion;
		loading = true; error = '';
		try {
			const query = new URLSearchParams(); if (x) query.set('x', x); if (y) query.set('y', y); if (metric) query.set('metric', metric);
			const response = await fetch(`/api/strategy/jobs/${encodeURIComponent(jobId)}/heatmap?${query}`, { cache: 'no-store' });
			const value = await response.json();
			if (!response.ok) throw new Error(value.error ?? `Heatmap request failed (${response.status}).`);
			if (version === requestVersion) { model = value; selected = undefined; }
		} catch (cause) { if (version === requestVersion) error = cause instanceof Error ? cause.message : 'Unable to load the native sweep heatmap.'; }
		finally { if (version === requestVersion) loading = false; }
	}

	$: if (jobId) load();
	function changeAxis(which: 'x' | 'y', value: string) {
		if (!model) return;
		const x = which === 'x' ? value : model.selectedAxes.x;
		const y = which === 'y' ? value : model.selectedAxes.y;
		if (x === y) { error = 'Choose two distinct parameter axes.'; return; }
		void load(x, y, model.selectedMetric);
	}
	function changeMetric(value: string) { if (model) void load(model.selectedAxes.x, model.selectedAxes.y, value); }
	function showValue(value: unknown) { return value === null ? 'null' : String(value); }
	function format(value: number | null) { return value === null ? '—' : new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value); }
	function cellState(cell: Cell) {
		if (!cell.planned) return 'not-planned';
		if (!cell.candidates.length) return 'missing';
		const statuses = new Set(cell.candidates.map(candidate => candidate.status));
		if (statuses.has('completed') && statuses.size > 1) return 'mixed';
		if (statuses.has('completed')) return 'completed';
		if (cell.candidates.some(candidate => candidate.status === 'failed')) return 'failed';
		if (cell.candidates.some(candidate => candidate.status === 'missing_result')) return 'missing';
		return 'skipped';
	}
	function cellValue(cell: Cell) {
		const completed = cell.candidates.filter(candidate => candidate.status === 'completed' && candidate.value !== null);
		if (cellState(cell) === 'not-planned') return '';
		if (cell.candidates.length > 1) return `${completed.length}/${cell.candidates.length} values`;
		if (completed.length === 1) return format(completed[0].value);
		return cell.candidates.some(candidate => candidate.status === 'completed') ? 'no metric' : cellState(cell);
	}
	function color(cell: Cell) {
		if (cellState(cell) !== 'completed' || cell.candidates.length !== 1 || cell.candidates[0].value === null) return '';
		const values = model?.cells.flatMap(row => row.cells.flatMap(item => item.candidates.filter(candidate => candidate.status === 'completed' && candidate.value !== null).map(candidate => candidate.value as number))) ?? [];
		const min = Math.min(...values), max = Math.max(...values);
		const ratio = max === min ? 0.5 : (cell.candidates.find(candidate => candidate.status === 'completed' && candidate.value !== null)?.value as number - min) / (max - min);
		return `--heat: ${Math.max(0, Math.min(1, ratio))}`;
	}
	function choose(cell: Cell) { if (cell.planned) selected = cell; }
</script>

<section class="heatmap" aria-label="Native sweep heatmap">
	<header><div><h2>Sweep heatmap</h2><p>Native candidates from the completed sweep plan and observed summary.</p></div>{#if model}<span>{model.counts.completed} completed · {model.counts.failed} failed · {model.counts.skipped} skipped of {model.counts.planned}</span>{/if}</header>
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	{#if loading && !model}<p>Loading native sweep artifacts…</p>{:else if model}
		<div class="controls">
			<label>X axis <select value={model.selectedAxes.x} on:change={(event) => changeAxis('x', event.currentTarget.value)}>{#each model.axes as axis}<option value={axis.path}>{axis.path}</option>{/each}</select></label>
			<label>Y axis <select value={model.selectedAxes.y} on:change={(event) => changeAxis('y', event.currentTarget.value)}>{#each model.axes as axis}<option value={axis.path} disabled={axis.path === model.selectedAxes.x}>{axis.path}</option>{/each}</select></label>
			<label>Metric <select value={model.selectedMetric} on:change={(event) => changeMetric(event.currentTarget.value)}>{#each model.metrics as metric}<option value={metric.key}>{metric.label}</option>{/each}</select></label>
		</div>
		<div class="legend"><span class="swatch done"></span> completed metric <span class="swatch fail"></span> failed <span class="swatch missing"></span> missing result <span class="swatch skip"></span> skipped <span class="swatch none"></span> not planned</div>
		<div class="grid-scroll"><table><thead><tr><th>{model.selectedAxes.y} ↓ / {model.selectedAxes.x} →</th>{#each model.cells[0]?.cells ?? [] as cell}<th>{showValue(cell.x)}</th>{/each}</tr></thead><tbody>{#each model.cells as row}<tr><th>{showValue(row.y)}</th>{#each row.cells as cell}<td><button class={cellState(cell)} style={color(cell)} disabled={!cell.planned} title={cell.planned ? `${cell.candidates.length} planned candidate(s); click for exact candidate IDs and parameter values` : 'No candidate in this coordinate'} on:click={() => choose(cell)}>{cellValue(cell)}</button></td>{/each}</tr>{/each}</tbody></table></div>
		{#if selected}
			<aside><button class="close" on:click={() => selected = undefined} aria-label="Close candidate details">×</button><h3>{model.selectedAxes.x} = {showValue(selected.x)} · {model.selectedAxes.y} = {showValue(selected.y)}</h3>
				{#each selected.candidates as candidate}<article><strong>{candidate.candidateId}</strong><span class="status {candidate.status}">{candidate.status}</span><p>{model.metrics.find(metric => metric.key === model?.selectedMetric)?.label}: {candidate.value === null ? 'unavailable' : format(candidate.value)}</p>{#if candidate.error}<p class="error">{candidate.error}</p>{/if}<details><summary>Exact parameters</summary><pre>{JSON.stringify(candidate.parameters, null, 2)}</pre></details></article>{/each}
			</aside>
		{/if}
	{:else}<p>Heatmap needs at least two parameter axes in the native sweep plan.</p>{/if}
</section>

<style>
	.heatmap{background:#111a2b;border:1px solid #26344b;border-radius:12px;padding:18px;color:#e8eef8}.heatmap header{display:flex;justify-content:space-between;align-items:start;gap:18px}.heatmap h2{margin:0}.heatmap header p{margin:5px 0 14px;color:#9eacc0}.heatmap header span,.legend{font-size:.82rem;color:#aebbd0}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.controls label{display:grid;gap:5px;color:#aebbd0;font-size:.82rem}.controls select{min-width:190px;max-width:min(38vw,460px);background:#0b1220;color:#e8eef8;border:1px solid #35455f;border-radius:6px;padding:7px}.legend{display:flex;align-items:center;gap:7px;margin:12px 0}.swatch{width:13px;height:13px;border-radius:3px}.done{background:linear-gradient(90deg,#275e58,#70bf79)}.fail{background:#75343a}.missing{background:#795f2b}.skip{background:#4e5870}.none{background:#1a2638}.grid-scroll{max-height:550px;overflow:auto;border:1px solid #26344b;border-radius:7px}table{border-collapse:separate;border-spacing:1px;min-width:100%;font-size:.82rem}th,td{padding:3px;text-align:center;white-space:nowrap}thead th,tbody th{position:sticky;background:#111a2b;z-index:1}thead th{top:0}tbody th{left:0;text-align:right}.grid-scroll td button{min-width:68px;min-height:38px;border:0;border-radius:4px;background:#1a2638;color:#cbd6e5;cursor:pointer}button.completed{background:color-mix(in srgb,#275e58 calc((var(--heat, .5) * 55%) + 25%),#202b3d);color:#fff}button.failed{background:#75343a;color:#fff}button.missing{background:#795f2b;color:#fff}button.skipped{background:#4e5870;color:#fff}button.not-planned{background:#131d2c;cursor:default}button:hover:not(:disabled){outline:2px solid #e2edf8}aside{position:relative;margin-top:14px;padding:16px;background:#0b1220;border-radius:8px}aside h3{padding-right:30px}article{border-top:1px solid #26344b;padding:12px 0}article strong{font-family:monospace;word-break:break-all}.status{margin-left:12px;color:#b9c6d9}.status.failed,.error{color:#ff9999}.status.missing_result{color:#ffd276}article p{margin:7px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#cbd6e5}.close{float:right;background:transparent;border:0;color:#dce8f5;font-size:1.5rem;cursor:pointer}.error{padding:8px}
</style>

<script lang="ts">
	export let jobId: string;
	type Candidate = { row: { run_id: string; run_index: number; rank: number; fee_scenario: string; metric_value?: number | null }; candidate: { schema_version: number; provenance: Record<string, unknown>; execution_config: Record<string, unknown> } };
	let candidates: Candidate[] = [];
	let selectedRunId = '';
	let loading = false;
	let saving = false;
	let error = '';
	let savedPath = '';
	let requestSequence = 0;
	$: selected = candidates.find(item => item.row.run_id === selectedRunId) ?? candidates[0];
	async function load() {
		const sequence = ++requestSequence;
		candidates = []; selectedRunId = ''; savedPath = ''; error = '';
		if (!jobId) return;
		loading = true;
		try {
			const response = await fetch(`/api/strategy/jobs/${encodeURIComponent(jobId)}/candidate`, { cache: 'no-store' });
			const body = await response.json();
			if (sequence !== requestSequence) return;
			if (!response.ok) throw new Error(body.error ?? 'Unable to load ranked candidates.');
			if (!Array.isArray(body?.candidates)) throw new Error('Candidate API returned invalid data.');
			candidates = body.candidates;
			selectedRunId = candidates[0]?.row.run_id ?? '';
		} catch (cause) { if (sequence === requestSequence) error = cause instanceof Error ? cause.message : 'Unable to load ranked candidates.'; }
		finally { if (sequence === requestSequence) loading = false; }
	}
	async function save() {
		if (!selected) return;
		saving = true; error = ''; savedPath = '';
		try {
			const response = await fetch(`/api/strategy/jobs/${encodeURIComponent(jobId)}/candidate`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ runId: selected.row.run_id }) });
			const body = await response.json();
			if (!response.ok) throw new Error(body.error ?? 'Unable to save candidate.');
			savedPath = body.path;
		} catch (cause) { error = cause instanceof Error ? cause.message : 'Unable to save candidate.'; }
		finally { saving = false; }
	}
	$: if (jobId) void load();
	$: if (!jobId) { requestSequence++; candidates = []; selectedRunId = ''; loading = false; }
</script>

<section class="candidate-import" aria-label="Save ranked strategy candidate">
	<header><div><div class="eyebrow">TUI STRATEGY SETUP</div><h3>Save candidate configuration</h3><p>Review the validated native configuration and provenance, then save a schema-compatible candidate record.</p></div><button type="button" onclick={load} disabled={loading || saving}>Refresh</button></header>
	{#if loading}<p class="muted">Checking native ranking and sweep plan…</p>
	{:else if error}<p class="error" role="alert">{error}</p>
	{:else if candidates.length}
		<label class="choose">Ranked candidate<select bind:value={selectedRunId}>{#each candidates as item (item.row.run_id)}<option value={item.row.run_id}>#{item.row.rank} · {item.row.run_id} · {item.row.fee_scenario}</option>{/each}</select></label>
		{#if selected}
			<div class="meta"><span>Rank <b>{selected.row.rank}</b></span><span>Run index <b>{selected.row.run_index}</b></span><span>Scenario <b>{selected.row.fee_scenario}</b></span><span>Metric <b>{String(selected.candidate.provenance.metric)}: {selected.row.metric_value ?? '—'}</b></span></div>
			<h4>Exact native execution configuration</h4><pre>{JSON.stringify(selected.candidate.execution_config, null, 2)}</pre>
			<h4>Import provenance</h4><pre>{JSON.stringify(selected.candidate.provenance, null, 2)}</pre>
			<button class="save" type="button" onclick={save} disabled={saving}>{saving ? 'Saving…' : 'Save validated candidate'}</button>
		{/if}
	{:else}<p class="muted">This completed ranking has no validated candidate rows.</p>{/if}
	{#if savedPath}<p class="saved" role="status">Saved candidate record: <code>{savedPath}</code></p>{/if}
	<footer>To apply this candidate in TUI, open Analytics → Sweeps, select this ranking artifact, press I, then Enter. The TUI revalidates the plan and saves its import record. Saving here does not change strategy setup, arm a strategy, or place orders.</footer>
</section>

<style>
	.candidate-import{margin:14px 0;padding:15px;border:1px solid #e7e2d8;border-radius:10px;background:#fbfaf7;color:#555449}.candidate-import header{display:flex;justify-content:space-between;gap:12px;align-items:start}.eyebrow{font-size:9px;letter-spacing:.1em;color:#898477}.candidate-import h3{font-size:14px;margin:4px 0}.candidate-import header p,.muted,.candidate-import footer{font-size:10px;color:#888477}.candidate-import header p{margin:5px 0 12px}.candidate-import button,.choose select{border:1px solid #ded9ce;border-radius:6px;background:#fff;color:#514f46;padding:7px 9px;font-size:10px}.choose{display:grid;gap:5px;font-size:10px}.choose select{width:100%}.meta{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.meta span{border:1px solid #e7e2d8;border-radius:6px;padding:6px 8px;font-size:9px;color:#817d72}.meta b{color:#4b4c43}.candidate-import h4{font-size:10px;margin:11px 0 5px;color:#6d695f}.candidate-import pre{max-height:190px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:#f1efe9;border-radius:6px;padding:9px;font:9px/1.45 ui-monospace,monospace;color:#555449}.save{margin:5px 0;background:#596f5b!important;color:white!important;border-color:#596f5b!important}.save:disabled,.candidate-import button:disabled{opacity:.55}.error{color:#a74f42;font-size:10px}.saved{font-size:10px;color:#55725e;overflow-wrap:anywhere}.candidate-import code{font-family:ui-monospace,monospace}.candidate-import footer{border-top:1px solid #ebe8df;padding-top:8px;margin-top:10px}
</style>

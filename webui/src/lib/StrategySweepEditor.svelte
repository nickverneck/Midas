<script lang="ts">
	export let sweeps: Array<{ id: string; name: string; sweepId: string }> = [];
	export let onCreated: (sweep: { id: string; name: string; sweepId: string; datasetViewId?: string }) => void = () => {};
	let sourceSweepId = '';
	let search = '';
	let name = '';
	let path = '';
	let valuesText = '[1]';
	let parameters: Array<{ path: string; values: Array<string | number | boolean> }> = [];
	let busy = false;
	let error = '';
	let created = '';
	$: source = sweeps.find((sweep) => sweep.id === sourceSweepId);
	$: matches = sweeps.filter((sweep) => `${sweep.name} ${sweep.sweepId}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
	$: firstMatches = matches.slice(0, 100);
	$: visibleSweeps = source && !firstMatches.some((sweep) => sweep.id === source.id) ? [source, ...firstMatches] : firstMatches;

	function addParameter() {
		error = '';
		try {
			if (!/^[a-z][a-z0-9_.]{0,127}$/.test(path)) throw new Error('Use a dot-separated native scalar path, such as order_qty.');
			const values: unknown = JSON.parse(valuesText);
			if (!Array.isArray(values) || !values.length || values.length > 16) throw new Error('Enter a JSON array with 1 to 16 values.');
			if (parameters.some((item) => item.path === path)) throw new Error('That path is already in the parameter list.');
			if (parameters.length >= 8) throw new Error('At most 8 parameter dimensions are allowed.');
			parameters = [...parameters, { path, values: values as Array<string | number | boolean> }];
			path = '';
		} catch (e) { error = e instanceof Error ? e.message : 'Invalid parameter values.'; }
	}

	async function create() {
		error = ''; created = '';
		if (!sourceSweepId) { error = 'Select a catalog sweep to clone.'; return; }
		if (!name.trim()) { error = 'Name the new sweep.'; return; }
		busy = true;
		try {
			const response = await fetch('/api/strategy/specs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sourceSweepId, name, parameters }) });
			const result = await response.json();
			if (!response.ok) throw new Error(typeof result?.error === 'string' ? result.error : `Request failed (${response.status}).`);
			created = result.sweepId;
			onCreated(result);
		} catch (e) { error = e instanceof Error ? e.message : 'Unable to create native sweep.'; }
		finally { busy = false; }
	}
</script>

<section class="editor" aria-labelledby="sweep-editor-title">
	<div class="head"><div><span class="eyebrow">NATIVE REPLAY SPEC</span><h3 id="sweep-editor-title">Create a bounded sweep</h3></div><span class="limit">32 combinations max</span></div>
	<p>Clone a catalog replay spec, then set parameter value lists. Dataset, strategy, replay, fees, account settings, and guardrails come from the selected spec.</p>
	<label for="sweep-source">Base catalog sweep</label>
	<input type="search" bind:value={search} aria-label="Search base sweeps" placeholder="Search sweep name or ID" />
	<small>{matches.length.toLocaleString()} matches{matches.length > 100 ? ' · showing first 100; refine search' : ''}</small>
	<select id="sweep-source" bind:value={sourceSweepId} on:change={() => { if (source) name = `${source.name} — edited`; }}>
		<option value="">Select a native sweep</option>
		{#each visibleSweeps as sweep}<option value={sweep.id}>{sweep.name} · {sweep.sweepId}</option>{/each}
	</select>
	<label for="sweep-name">New sweep name</label><input id="sweep-name" bind:value={name} maxlength="100" />
	<div class="parameter-head"><strong>Parameter dimensions</strong><span>{parameters.length} / 8</span></div>
	<div class="parameter-inputs"><input aria-label="Native scalar parameter path" placeholder="order_qty" bind:value={path} maxlength="128" /><input aria-label="Parameter values as JSON array" placeholder="[1, 2]" bind:value={valuesText} /><button type="button" disabled={parameters.length >= 8} on:click={addParameter}>Add</button></div>
	<small>Each list allows up to 16 finite numbers, booleans, or short strings. The native validator checks every path and value against the replay schema.</small>
	{#if parameters.length}<ul>{#each parameters as parameter, index}<li><code>{parameter.path}</code><span>{JSON.stringify(parameter.values)}</span><button type="button" aria-label={`Remove ${parameter.path}`} on:click={() => parameters = parameters.filter((_, i) => i !== index)}>Remove</button></li>{/each}</ul>{/if}
	<div class="actions"><button class="primary" disabled={busy || !sourceSweepId || !name.trim()} on:click={create}>{busy ? 'Validating…' : 'Create spec'}</button></div>
	{#if error}<div class="error" role="alert">{error}</div>{/if}
	{#if created}<div class="success" role="status">Created {created}. Refresh the catalog to select it for native validation or replay.</div>{/if}
</section>

<style>
	.editor{border:1px solid #e7e2d8;border-radius:10px;background:#fbfaf7;padding:13px;margin:12px 0;color:#615e54}.head,.parameter-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.eyebrow{font-size:8px;letter-spacing:.12em;color:#9a895c}.head h3{font:700 14px Manrope;margin:4px 0 0;color:#48483f}.limit,small{font-size:9px;color:#918c80}.editor p{font-size:10px;line-height:1.5;color:#868174;margin:10px 0}.editor label{display:block;font-size:9px;font-weight:600;margin:9px 0 5px}.editor select,.editor input{box-sizing:border-box;min-width:0;border:1px solid #e3dfd5;border-radius:6px;background:#fff;padding:7px;color:#555146;font:10px Manrope}.editor select,.editor>input{width:100%}.parameter-head{margin:12px 0 6px;font-size:10px}.parameter-inputs{display:grid;grid-template-columns:1fr 1fr auto;gap:6px}.parameter-inputs input:nth-child(2){font-family:ui-monospace,monospace}.parameter-inputs button,.editor li button,.actions button{border:1px solid #e1dccf;border-radius:6px;background:#fff;color:#696355;font-size:9px;padding:6px 8px;cursor:pointer}.editor ul{list-style:none;margin:8px 0;padding:0}.editor li{display:flex;gap:8px;align-items:center;border-top:1px solid #ebe8df;padding:6px 0;font-size:9px}.editor li code{color:#476550}.editor li span{flex:1;overflow-wrap:anywhere}.actions{margin-top:10px}.actions .primary{background:#e9ede7;border-color:#dce4d8;color:#54705c}.editor button:disabled{opacity:.5;cursor:not-allowed}.error,.success{font-size:10px;margin-top:8px}.error{color:#a74f42}.success{color:#54705c}
</style>

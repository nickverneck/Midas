<script lang="ts">
	import ReplayResults from './ReplayResults.svelte';
	import { onDestroy } from 'svelte';
	export let sweepId = '';
	type Job = { id: string; operation: 'lua-validate' | 'lua-run'; sweepId: string; status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'; result?: unknown; error?: string; updatedAt: string };
	type ReplayRun = { runId: string; metrics: Record<string, unknown>; trades: unknown[] };
	let source = `function on_bar(ctx, bar)\n  return "hold"\nend`;
	let job: Job | undefined;
	let replayRuns: ReplayRun[] = [];
	let resultsLoading = false;
	let resultsLoadedFor = '';
	let busy = false;
	let error = '';
	let poller: ReturnType<typeof setInterval> | undefined;
	let pollInFlight = false;
	const maxBytes = 12_288;
	$: sourceBytes = new TextEncoder().encode(source).length;
	$: terminal = !!job && ['complete', 'failed', 'cancelled'].includes(job.status);
	onDestroy(() => { if (poller) clearInterval(poller); });

	async function json<T>(response: Response): Promise<T> {
		const value = await response.json();
		if (!response.ok) throw new Error(typeof value?.error === 'string' ? value.error : `Request failed (${response.status}).`);
		return value as T;
	}
	async function submit(operation: 'lua-validate' | 'lua-run') {
		error = '';
		replayRuns = []; resultsLoadedFor = '';
		if (!sweepId) { error = 'Select an existing native sweep first.'; return; }
		if (!source.trim()) { error = 'Enter Lua source to continue.'; return; }
		if (sourceBytes > maxBytes) { error = 'Lua source exceeds the 12 KiB limit.'; return; }
		busy = true;
		try {
			job = await json<Job>(await fetch('/api/strategy/lua', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ operation, sweepId, luaSource: source }) }));
			if (poller) clearInterval(poller);
			poller = setInterval(() => void poll(), 1000);
		} catch (e) { error = e instanceof Error ? e.message : 'Unable to submit Lua prototype.'; }
		finally { busy = false; }
	}
	async function poll() {
		if (!job || terminal) { if (poller) clearInterval(poller); return; }
		if (pollInFlight) return;
		const requestedJobId = job.id;
		pollInFlight = true;
		try {
			const updated = await json<Job>(await fetch(`/api/strategy/jobs/${encodeURIComponent(requestedJobId)}`, { cache: 'no-store' }));
			if (job?.id !== requestedJobId) return;
			job = updated;
			if (job.status === 'complete' && job.operation === 'lua-run') await loadResults();
			if (terminal && poller) clearInterval(poller);
		}
		catch (e) { error = e instanceof Error ? e.message : 'Unable to read Lua job status.'; }
		finally { pollInFlight = false; }
	}
	async function loadResults() {
		if (!job || job.operation !== 'lua-run' || resultsLoadedFor === job.id || resultsLoading) return;
		const requestedJobId = job.id;
		resultsLoading = true;
		try {
			const value = await json<{ runs: ReplayRun[] }>(await fetch(`/api/strategy/jobs/${encodeURIComponent(requestedJobId)}/results`, { cache: 'no-store' }));
			if (job?.id !== requestedJobId) return;
			replayRuns = value.runs ?? [];
			resultsLoadedFor = requestedJobId;
		} catch (e) { error = e instanceof Error ? e.message : 'Unable to read Lua replay results.'; }
		finally { resultsLoading = false; }
	}
	async function cancel() {
		if (!job) return;
		busy = true;
		try { job = await json<Job>(await fetch(`/api/strategy/jobs/${encodeURIComponent(job.id)}`, { method: 'DELETE' })); }
		catch (e) { error = e instanceof Error ? e.message : 'Unable to cancel Lua job.'; }
		finally { busy = false; }
	}
</script>

<section class="lua-panel" aria-labelledby="lua-title">
	<div class="head"><div><span class="eyebrow">OFFLINE REPLAY</span><h3 id="lua-title">Lua prototype</h3></div><span class="limit">{sourceBytes.toLocaleString()} / {maxBytes.toLocaleString()} bytes</span></div>
	<p>Uses the selected sweep’s dataset, replay, fee, and account settings. Runs one Lua strategy in isolated replay services.</p>
	<label for="lua-source">Lua source</label>
	<textarea id="lua-source" bind:value={source} spellcheck="false" maxlength="12288" aria-describedby="lua-limit"></textarea>
	<small id="lua-limit">Return “buy”, “sell”, or “hold”. Read bar.open, bar.high, bar.low, bar.close, bar.volume, bar.timestamp_ns, ctx.bar_index and ctx.position_qty. Globals persist within each isolated replay. Runs use closed bars; ctx.is_warmup is true while rebuilding history, where position_qty is unavailable. Limits: 8 MiB and 100,000 instructions per evaluation.</small>
	<div class="actions"><button disabled={busy || (job !== undefined && !terminal) || !sweepId || sourceBytes > maxBytes} on:click={() => submit('lua-validate')}>Validate</button><button class="primary" disabled={busy || (job !== undefined && !terminal) || !sweepId || sourceBytes > maxBytes} on:click={() => submit('lua-run')}>Run replay</button>{#if job && !terminal}<button class="cancel" disabled={busy} on:click={cancel}>Cancel</button>{/if}</div>
	{#if error}<div class="error" role="alert">{error}</div>{/if}
	{#if job}<div class="status"><div><strong>{job.operation === 'lua-run' ? 'Replay' : 'Validation'}</strong><span class="state">{job.status}</span></div>{#if job.error}<pre>{job.error}</pre>{:else if job.result}<pre>{JSON.stringify(job.result, null, 2)}</pre>{/if}</div>{/if}
	{#if job?.operation === 'lua-run' && job.status === 'complete'}<div class="results"><strong>Replay results</strong>{#if resultsLoading}<small>Loading native replay artifacts…</small>{:else if replayRuns.length}{#each replayRuns as run}<div class="run"><span>{run.runId} · {run.trades.length} trades</span><pre>{JSON.stringify(run.metrics, null, 2)}</pre></div>{/each}{:else if resultsLoadedFor === job.id}<small>No completed replay artifacts were returned.</small>{/if}</div>{/if}
{#if job?.operation === 'lua-run' && job.status === 'complete'}<ReplayResults jobId={job.id}/>{/if}
</section>

<style>
	.lua-panel{border:1px solid #e7e2d8;border-radius:10px;background:#fbfaf7;padding:13px;margin:12px 0;color:#615e54}.head,.head>div,.status>div{display:flex;align-items:center;justify-content:space-between;gap:8px}.eyebrow{font-size:8px;letter-spacing:.12em;color:#9a895c}.head h3{font:700 14px Manrope;margin:4px 0 0;color:#48483f}.limit,small{font-size:9px;color:#918c80}.lua-panel p{font-size:10px;line-height:1.5;color:#868174;margin:10px 0}.lua-panel label{display:block;font-size:9px;font-weight:600;margin:9px 0 5px}textarea{box-sizing:border-box;width:100%;min-height:190px;resize:vertical;border:1px solid #e3dfd5;border-radius:7px;background:#fff;padding:10px;color:#42453f;font:11px/1.55 ui-monospace,SFMono-Regular,monospace;tab-size:2}.actions{display:flex;gap:7px;margin-top:10px}button{border:1px solid #e1dccf;border-radius:6px;background:#fff;color:#696355;font-size:10px;padding:7px 10px;cursor:pointer}button.primary{background:#e9ede7;border-color:#dce4d8;color:#54705c}button.cancel{color:#a34b3e;background:#fbefeb;border-color:#eed3cc}button:disabled{opacity:.5;cursor:not-allowed}.error{font-size:10px;color:#a74f42;margin-top:9px}.status{border-top:1px solid #ebe8df;margin-top:11px;padding-top:10px;font-size:10px}.state{text-transform:uppercase;letter-spacing:.06em;color:#8b6c2a;font-size:9px}.status pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:180px;overflow:auto;font:10px/1.5 ui-monospace,monospace;background:#f2f0e9;border-radius:6px;padding:8px;margin:8px 0 0}
	.results{border-top:1px solid #ebe8df;margin-top:10px;padding-top:9px;font-size:10px}.run{margin-top:8px}.run>span{font-size:9px;color:#777}.run pre{white-space:pre-wrap;background:#f2f0e9;padding:8px;border-radius:6px;font:10px/1.5 ui-monospace,monospace;margin:5px 0}
</style>

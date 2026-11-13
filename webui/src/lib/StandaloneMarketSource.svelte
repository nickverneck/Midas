<script lang="ts">
	import { createEventDispatcher, onDestroy } from 'svelte';
	import { SseParser } from '$lib/sse-parser';
	type Bar = { ts_ns: string; open: number; high: number; low: number; close: number; volume?: number };
	type Snapshot = { connection_status: string; chart_freshness: string; source: string; broker_source: string; fidelity: string; contract: string; bar_kind: string; bar_value: number; bars: Bar[]; latest_bar_ts_ns: string | null; observed_at: string | null };
	const dispatch = createEventDispatcher<{ snapshot: Snapshot; state: { status: string; error?: string } }>();
	let contract = 'ESZ6'; let barKind = 'minute'; let barValue = 1;
	let busy = false; let status = 'ended'; let error = ''; let snapshot: Snapshot | undefined; let controller: AbortController | undefined; let generation = 0;
	const requestBody = () => ({ contract, bar_kind: barKind, bar_value: Number(barValue) });
	function setStatus(next: string, message = '') { status = next; error = message; dispatch('state', { status, ...(message ? { error: message } : {}) }); }
	function consumeSnapshot(value: Snapshot) { snapshot = value; dispatch('snapshot', value); }
	async function start() {
		if (busy) return;
		const current = ++generation; const aborter = new AbortController(); controller = aborter; busy = true; setStatus('connecting');
		try {
			const response = await fetch('/api/standalone-market/stream', { method: 'POST', cache: 'no-store', headers: { 'content-type': 'application/json', accept: 'text/event-stream' }, body: JSON.stringify(requestBody()), signal: aborter.signal });
			if (!response.ok) { let message = `Market stream failed (${response.status}).`; try { message = (await response.json()).error ?? message; } catch {} throw new Error(message); }
			if (!response.body) throw new Error('Market stream response has no body.');
			const reader = response.body.getReader(); const decoder = new TextDecoder();
			const parser = new SseParser(({ event, data }) => {
				let payload: any; try { payload = JSON.parse(data); } catch { throw new Error('Market stream sent invalid event data.'); }
				if (event === 'snapshot') { consumeSnapshot(payload as Snapshot); setStatus('connected'); }
				else if (event === 'status') { if (payload.connection_status === 'disconnected') throw new Error(payload.error ?? 'Market stream failed.'); setStatus('ended'); }
			});
			while (current === generation) {
				const { value, done } = await reader.read();
				if (done) { parser.push(decoder.decode()); parser.finish(); break; }
				parser.push(decoder.decode(value, { stream: true }));
			}
			if (current === generation) setStatus('ended');
		} catch (cause) { if (current === generation && !aborter.signal.aborted) { aborter.abort(); setStatus('error', cause instanceof Error ? cause.message : 'Unable to read provider market stream.'); } }
		finally { if (current === generation) { busy = false; controller = undefined; } }
	}
	function stop() { if (!busy) return; generation++; controller?.abort(); controller = undefined; busy = false; setStatus('ended'); }
	async function refresh() {
		if (busy) return; const current = ++generation; busy = true; setStatus('connecting');
		try { const response = await fetch('/api/standalone-market', { method: 'POST', cache: 'no-store', headers: { 'content-type': 'application/json' }, body: JSON.stringify(requestBody()) }); const value = await response.json(); if (!response.ok) throw new Error(value.error ?? `Market request failed (${response.status}).`); if (current === generation) { consumeSnapshot(value as Snapshot); setStatus('ended'); } }
		catch (cause) { if (current === generation) setStatus('error', cause instanceof Error ? cause.message : 'Unable to request provider chart bars.'); }
		finally { if (current === generation) busy = false; }
	}
	onDestroy(() => { generation++; controller?.abort(); });
</script>

<section class="source" aria-label="Standalone market source">
	<header><div><span class="eyebrow">STANDALONE PROVIDER STREAM</span><h2>Standalone market source</h2><p>Starts only when requested. Observes provider market data without starting the trader account or order stream. No orders are sent.</p></div><span class="status">{status}</span></header>
	<div class="controls">
		<label>Exact contract symbol<input bind:value={contract} maxlength="16" pattern="[A-Za-z0-9]+" autocomplete="off" spellcheck="false" disabled={busy} /></label>
		<label>Bar type<select bind:value={barKind} disabled={busy}><option value="minute">Minute</option><option value="second">Second</option><option value="tick">Tick</option><option value="volume">Volume</option><option value="range">Range</option></select></label><label>Bar size<input type="number" min="1" max="100000" step="1" bind:value={barValue} disabled={busy} /></label>
		<button class="refresh" disabled={busy} on:click={start}>Start stream</button><button class="stop" disabled={!busy} on:click={stop}>Stop</button><button class="once" disabled={busy} on:click={refresh}>Refresh once</button>
	</div>
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	{#if snapshot}<div class="metadata"><span>{snapshot.contract} · {snapshot.bar_value} {snapshot.bar_kind}</span><span>{snapshot.bars.length.toLocaleString()} bars</span><span>Freshness: unknown</span><span>Latest ns: {snapshot.latest_bar_ts_ns ?? snapshot.bars.at(-1)?.ts_ns ?? 'not reported'}</span></div><p class="fidelity">{snapshot.broker_source} · {snapshot.fidelity} · Observed {snapshot.observed_at ?? 'time not reported'}.</p>{/if}
</section>

<style>
	.source{background:#fbfaf7;border:1px solid #e5dfd2;border-radius:12px;padding:18px;color:#39372f}.source header{display:flex;justify-content:space-between;gap:18px;align-items:start}.source h2{margin:4px 0}.source header p{margin:5px 0 14px;max-width:780px;color:#817b6f}.eyebrow{font-size:.72rem;letter-spacing:.08em;color:#927744}.status,.metadata{font-size:.82rem;color:#746b59;text-transform:capitalize}.controls{display:flex;align-items:end;gap:8px;flex-wrap:wrap}.controls label{display:grid;gap:5px;font-size:.82rem;color:#746b59}.controls select,.controls input,.refresh,.stop,.once{background:#fff;color:#39372f;border:1px solid #d9d1c2;border-radius:6px;padding:8px}.controls select{min-width:130px}.refresh,.stop,.once{cursor:pointer}.refresh{background:#67543b;color:white}.stop{background:#f7ebe6;color:#914b3c}.refresh:disabled,.stop:disabled,.once:disabled{opacity:.6;cursor:default}.metadata{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px}.fidelity{color:#817b6f;font-size:.82rem}.error{color:#a34839}
</style>

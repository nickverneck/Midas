<script lang="ts">
	export let jobId: string;
	type Point = { timestampNs: string; equity: number; cumulativeNetPnl: number };
	type Summary = { initial_capital?: number; ending_equity?: number; return_on_initial_capital_pct?: number | null; closed_trade_count?: number; win_count?: number; loss_count?: number; win_rate_pct?: number | null; profit_factor?: number | null; max_drawdown_pct?: number | null };
	type Run = { runId: string; metrics: { grossPnl: number; netPnl: number; fees: number; maxDrawdown: number; tradeCount: number }; equity: Point[]; trades: Record<string, string>[]; result?: { summary?: Summary; metadata?: { account_currency?: string } } };
	let data: { sweepId: string; runs: Run[] } | undefined;
	let error = '';
	let loading = false;
	let activeRun = '';
	let requestId = 0;
	$: run = data?.runs.find(item => item.runId === activeRun) ?? data?.runs[0];
	$: points = run?.equity ?? [];
	$: peakValues = points.reduce((state: number[], point, index) => { state.push(Math.max(index ? state[index - 1] : point.equity, point.equity)); return state; }, []);
	$: drawdowns = points.map((point, index) => peakValues[index] ? (point.equity - peakValues[index]) / peakValues[index] * 100 : 0);
	$: line = chartLine(points.map(point => point.equity));
	$: ddLine = chartLine(drawdowns);
	function chartLine(values: number[]) {
		if (!values.length) return '';
		const min = values.reduce((a, b) => Math.min(a, b), Infinity), max = values.reduce((a, b) => Math.max(a, b), -Infinity), span = max - min || 1;
		return values.map((value, i) => `${i ? 'L' : 'M'} ${(i / Math.max(values.length - 1, 1) * 1000).toFixed(1)} ${(190 - (value - min) / span * 170).toFixed(1)}`).join(' ');
	}
	function money(value: number | null | undefined) {
		if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
		const currency = run?.result?.metadata?.account_currency;
		if (currency && /^[A-Z]{3}$/.test(currency)) {
			try { return value.toLocaleString(undefined, { style: 'currency', currency, maximumFractionDigits: 2 }); } catch { /* Display the recorded code below. */ }
		}
		return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${currency ? ` ${currency}` : ''}`;
	}
	function number(value: number | null | undefined, suffix = '') { return typeof value === 'number' && Number.isFinite(value) ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}` : '—'; }
	async function load() {
		if (!jobId) return;
		const currentRequest = ++requestId;
		const requestedJobId = jobId;
		data = undefined;
		activeRun = '';
		loading = true; error = '';
		try {
			const response = await fetch(`/api/strategy/jobs/${encodeURIComponent(requestedJobId)}/results`, { cache: 'no-store' });
			const body = await response.json();
			if (currentRequest !== requestId || jobId !== requestedJobId) return;
			if (!response.ok) throw new Error(body.error ?? 'Unable to load replay artifacts.');
			if (!body || !Array.isArray(body.runs)) throw new Error('The replay results API returned invalid data.');
			data = body;
			if (!activeRun) activeRun = body.runs?.[0]?.runId ?? '';
		} catch (e) {
			if (currentRequest === requestId && jobId === requestedJobId) error = e instanceof Error ? e.message : 'Unable to load replay artifacts.';
		} finally {
			if (currentRequest === requestId && jobId === requestedJobId) loading = false;
		}
	}
	$: if (jobId) void load();
	$: if (!jobId) { requestId++; data = undefined; activeRun = ''; loading = false; error = ''; }
</script>

{#if loading}<p class="hint">Loading native replay artifacts…</p>{:else if error}<p class="error">{error}</p>{:else if run}
	<section class="results" aria-label="Native replay results">
		{#if data && data.runs.length > 1}<label>Replay run <select bind:value={activeRun}>{#each data.runs as item}<option value={item.runId}>{item.runId}</option>{/each}</select></label>{/if}
		<div class="metrics"><div><small>Net P&amp;L</small><b>{money(run.metrics.netPnl)}</b></div><div><small>Gross P&amp;L</small><b>{money(run.metrics.grossPnl)}</b></div><div><small>Fees</small><b>{money(run.metrics.fees)}</b></div><div><small>Return</small><b>{number(run.result?.summary?.return_on_initial_capital_pct, '%')}</b></div><div><small>Starting capital</small><b>{money(run.result?.summary?.initial_capital)}</b></div><div><small>Ending equity</small><b>{money(run.result?.summary?.ending_equity)}</b></div><div><small>Max drawdown</small><b>{money(run.metrics.maxDrawdown)}</b></div><div><small>Max drawdown %</small><b>{number(run.result?.summary?.max_drawdown_pct, '%')}</b></div><div><small>Trades</small><b>{number(run.metrics.tradeCount)}</b></div><div><small>Closed trades</small><b>{number(run.result?.summary?.closed_trade_count)}</b></div><div><small>Win rate</small><b>{number(run.result?.summary?.win_rate_pct, '%')}</b></div><div><small>Profit factor</small><b>{number(run.result?.summary?.profit_factor)}</b></div><div><small>Wins / losses</small><b>{number(run.result?.summary?.win_count)} / {number(run.result?.summary?.loss_count)}</b></div></div>
		<div class="chart"><h4>Equity · native equity.csv</h4>{#if points.length}<svg viewBox="0 0 1000 210" role="img" aria-label="Replay equity curve"><path d={line} /></svg><div class="ends"><span>{new Date(Number(points[0].timestampNs) / 1e6).toLocaleString()}</span><span>{new Date(Number(points.at(-1)?.timestampNs ?? 0) / 1e6).toLocaleString()}</span></div>{:else}<p class="hint">No equity.csv points were emitted for this run.</p>{/if}</div>
		<div class="chart drawdown"><h4>Drawdown · calculated from native equity points</h4>{#if points.length}<svg viewBox="0 0 1000 210" role="img" aria-label="Replay drawdown curve"><path d={ddLine} /></svg><small>Worst observed: {drawdowns.reduce((a, b) => Math.min(a, b), 0).toFixed(3)}%</small>{:else}<p class="hint">No equity points available.</p>{/if}</div>
		{#if run.trades.length}<div class="trade-list"><h4>Native trades.csv · {run.trades.length} rows</h4><div class="table-scroll"><table><thead><tr>{#each Object.keys(run.trades[0]) as key}<th>{key}</th>{/each}</tr></thead><tbody>{#each run.trades.slice(0, 100) as trade}<tr>{#each Object.values(trade) as value}<td>{value}</td>{/each}</tr>{/each}</tbody></table></div></div>{/if}
	</section>
{:else if data}<p class="hint">No completed replay runs were emitted for this job.</p>
{/if}

<style>
	.results{display:grid;gap:14px;margin-top:16px;color:#57564d}.results>label{font-size:10px;color:#716e63}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}.metrics div{padding:10px;border:1px solid #e8e5dc;border-radius:8px;background:#fbfaf7}.metrics small,.metrics b{display:block}.metrics small,.hint{color:#918c80;font-size:11px}.metrics b{margin-top:5px;color:#44453c;font-size:14px}.chart{padding:12px;border:1px solid #e8e5dc;border-radius:8px;background:#fbfaf7}.chart h4,.trade-list h4{margin:0 0 8px;color:#5e5c52;font-size:12px}.chart svg{display:block;width:100%;height:130px}.chart path{fill:none;stroke:#718a72;stroke-width:3;vector-effect:non-scaling-stroke}.drawdown path{stroke:#b49355}.drawdown>small{color:#918c80;font-size:10px}.ends{display:flex;justify-content:space-between;color:#918c80;font-size:10px}.trade-list{min-width:0;padding:12px;border:1px solid #e8e5dc;border-radius:8px;background:#fbfaf7}.table-scroll{overflow:auto;max-height:260px}table{border-collapse:collapse;font-size:11px;white-space:nowrap}th,td{padding:6px 8px;border-bottom:1px solid #eeece5;text-align:left}th{color:#777468;background:#f3f1eb}td{color:#57564d}.error{color:#a74f42;font-size:11px}select{margin-left:8px;border:1px solid #e8e4da;border-radius:6px;background:#fff;color:#57564d;padding:5px 7px;font-size:10px}
</style>

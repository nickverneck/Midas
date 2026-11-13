<script lang="ts">
	export let progress: unknown = undefined;
	export let jobStatus = 'running';
	export let jobError: string | undefined = undefined;

	type Run = { run_id?: unknown; run_index?: unknown; status?: unknown; error?: unknown; skipped?: unknown };
	type NativeStatus = {
		 schema_version?: unknown; sweep_id?: unknown; name?: unknown; status?: unknown;
		started_at_utc?: unknown; updated_at_utc?: unknown;
		 run_count?: unknown; completed_count?: unknown; failed_count?: unknown; skipped_count?: unknown;
		 active_count?: unknown; pending_count?: unknown; cancellation_requested?: unknown;
		 last_completed_run_id?: unknown; error?: unknown; runs?: unknown; warnings?: unknown;
	};

	$: doc = isRecord(progress) ? progress as NativeStatus : undefined;
	$: runs = Array.isArray(doc?.runs) ? doc.runs.filter(isRecord) as Run[] : [];
	$: total = count(doc?.run_count);
	$: completed = count(doc?.completed_count);
	$: failed = count(doc?.failed_count);
	$: skipped = count(doc?.skipped_count);
	$: active = count(doc?.active_count);
	$: pending = count(doc?.pending_count);
	$: percent = total !== undefined && total > 0 && completed !== undefined && failed !== undefined && skipped !== undefined
		? Math.min(100, Math.round((completed + failed + skipped) / total * 100)) : undefined;
	$: sweepErrors = runs.filter(run => typeof run.error === 'string' && run.error.trim());
	$: warnings = Array.isArray(doc?.warnings) ? doc.warnings.filter((item): item is string => typeof item === 'string' && !!item.trim()) : [];
	$: nativeError = typeof doc?.error === 'string' && doc.error.trim() ? doc.error : undefined;
	$: hasFailures = failed !== undefined && failed > 0 || sweepErrors.length > 0 || !!nativeError || !!jobError;

	function isRecord(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value); }
	function count(value: unknown): number | undefined { return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined; }
	function display(value: number | undefined): string { return value === undefined ? 'Unknown' : value.toLocaleString(); }
	function text(value: unknown): string { return typeof value === 'string' && value.trim() ? value : 'Unknown'; }
	function formatDate(value: unknown): string {
		if (typeof value !== 'string') return 'Unknown';
		const date = new Date(value);
		return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString();
	}
</script>

<section class="replay-progress" aria-label="Native replay progress">
	<header>
		<div><small>Native replay sweep</small><h3>{text(doc?.name)}</h3></div>
		<span class="state" class:failed={hasFailures || doc?.status === 'failed' || jobStatus === 'failed'} class:done={jobStatus === 'complete' || doc?.status === 'completed'} aria-live="polite">{text(doc?.status ?? jobStatus)}</span>
	</header>
	{#if percent !== undefined}
		<div class="progress-track" role="progressbar" aria-label="Replay runs finished" aria-valuemin="0" aria-valuemax="100" aria-valuenow={percent}><span style={`width:${percent}%`}></span></div>
		<div class="progress-caption" aria-live="polite">{percent}% of runs finished (completed, failed, or skipped)</div>
	{:else}
		<div class="progress-track unknown" aria-hidden="true"><span></span></div><div class="progress-caption">Progress: Unknown</div>
	{/if}
	<div class="counts">
		<div><small>Completed</small><b>{display(completed)}</b></div><div><small>Failed</small><b class:bad={failed !== undefined && failed > 0}>{display(failed)}</b></div>
		<div><small>Skipped</small><b>{display(skipped)}</b></div><div><small>Running</small><b>{display(active)}</b></div>
		<div><small>Queued</small><b>{display(pending)}</b></div><div><small>Total</small><b>{display(total)}</b></div>
	</div>
	{#if doc?.last_completed_run_id || doc?.started_at_utc || doc?.updated_at_utc}
		<div class="details">
			{#if doc?.last_completed_run_id}<span>Last completed: <b>{text(doc.last_completed_run_id)}</b></span>{/if}
			{#if doc?.started_at_utc}<span>Started: <b>{formatDate(doc.started_at_utc)}</b></span>{/if}
			{#if doc?.updated_at_utc}<span>Updated: <b>{formatDate(doc.updated_at_utc)}</b></span>{/if}
		</div>
	{/if}
	{#if jobError || nativeError || sweepErrors.length}
		<div class="messages errors" role="alert"><b>Errors</b>
			{#if jobError}<p>{jobError}</p>{/if}{#if nativeError}<p>{nativeError}</p>{/if}
			{#each sweepErrors as run}<p><strong>Run {text(run.run_id)}:</strong> {text(run.error)}</p>{/each}
		</div>
	{/if}
	{#if warnings.length}
		<div class="messages warnings"><b>Warnings</b>{#each warnings as warning}<p>{warning}</p>{/each}</div>
	{/if}
	{#if doc?.cancellation_requested === true}<p class="cancel-note">Cancellation requested; the runner is finishing its current work.</p>{/if}
	{#if !doc}<p class="loading">Native sweep status: Unknown. Waiting for the first status update.</p>{/if}
</section>

<style>
	.replay-progress{display:grid;gap:10px;padding:13px;border:1px solid #e8e5dc;border-radius:9px;background:#fbfaf7;color:#57564d}
	header{display:flex;align-items:center;justify-content:space-between;gap:10px}header small,.counts small{color:#918c80;font-size:10px;text-transform:uppercase;letter-spacing:.06em}h3{margin:3px 0 0;color:#44453c;font-size:13px;font-weight:650}
	.state{padding:4px 8px;border-radius:20px;background:#efeee7;color:#716e63;font-size:10px;text-transform:capitalize}.state.done{background:#e7efe5;color:#567451}.state.failed{background:#f5e8e3;color:#a74f42}
	.progress-track{height:7px;overflow:hidden;border-radius:8px;background:#e9e7df}.progress-track span{display:block;height:100%;border-radius:inherit;background:#718a72;transition:width .25s}.progress-track.unknown span{width:30%;background:#c9c6ba;animation:indeterminate 1.4s ease-in-out infinite}.progress-caption{margin-top:-6px;color:#858174;font-size:10px}
	.counts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.counts>div{padding:8px;border:1px solid #ece9e0;border-radius:7px;background:#fff}.counts small,.counts b{display:block}.counts b{margin-top:3px;color:#44453c;font-size:13px}.counts b.bad{color:#a74f42}.details{display:flex;flex-wrap:wrap;gap:5px 14px;color:#918c80;font-size:10px}.details b{color:#716e63;font-weight:550}
	.messages{padding:9px 10px;border-radius:7px;font-size:11px}.messages b{font-size:10px;text-transform:uppercase;letter-spacing:.05em}.messages p{margin:5px 0 0;overflow-wrap:anywhere}.errors{border:1px solid #ead0c9;background:#fbf1ee;color:#91493f}.warnings{border:1px solid #eadfc5;background:#fbf7eb;color:#806b3b}.loading,.cancel-note{margin:0;color:#918c80;font-size:10px}.cancel-note{color:#806b3b}
	@keyframes indeterminate{0%{transform:translateX(-100%)}100%{transform:translateX(340%)}}
	@media(prefers-reduced-motion:reduce){.progress-track span{transition:none}.progress-track.unknown span{animation:none;width:100%}}
	@media(max-width:420px){.counts{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>

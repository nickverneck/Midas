<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Shield } from "lucide-svelte";
</script>

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

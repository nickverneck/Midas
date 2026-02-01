<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { FileCode } from "lucide-svelte";

	type Props = {
		scriptText: string;
		sampleScript: string;
		scriptInvalid: boolean;
	};

	let { scriptText = $bindable(), sampleScript, scriptInvalid }: Props = $props();

	const resetSample = () => {
		scriptText = sampleScript;
	};
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-start justify-between">
		<div>
			<Card.Title class="flex items-center gap-2">
				<FileCode size={18} />
				Strategy Script
			</Card.Title>
			<Card.Description>Write or paste Lua here. The script runs bar-by-bar.</Card.Description>
		</div>
		<div class="flex items-center gap-2">
			<Button variant="secondary" size="sm" onclick={resetSample}>Reset Sample</Button>
			<Button variant="outline" size="sm" disabled>
				Load .lua
			</Button>
		</div>
	</Card.Header>
	<Card.Content>
		<textarea
			class={`min-h-[320px] w-full resize-none rounded-lg border bg-muted/30 px-4 py-3 font-mono text-xs leading-relaxed shadow-inner focus:outline-none focus:ring-2 ${
				scriptInvalid ? "border-destructive focus:ring-destructive/40" : "focus:ring-ring"
			}`}
			aria-invalid={scriptInvalid}
			bind:value={scriptText}
		></textarea>
		<div class="mt-4 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
			<span>Return: buy · sell · revert · hold</span>
			<span class="h-1 w-1 rounded-full bg-muted-foreground"></span>
			<span>Indicators: EMA / SMA / HMA / ATR</span>
			<span class="h-1 w-1 rounded-full bg-muted-foreground"></span>
			<span>Context: position, cash, equity</span>
		</div>
	</Card.Content>
</Card.Root>

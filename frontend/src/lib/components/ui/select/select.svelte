<script lang="ts">
	import { Select as SelectPrimitive } from "bits-ui";

	type SingleSelectRootProps = Omit<SelectPrimitive.RootProps, "type" | "value" | "onValueChange"> & {
		type?: "single";
		value?: string;
		onValueChange?: (value: string) => void;
	};

	let {
		open = $bindable(false),
		value = $bindable(),
		...restProps
	}: SingleSelectRootProps = $props();
</script>

<SelectPrimitive.Root type="single" bind:open bind:value={value as never} {...restProps} />

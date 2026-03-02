<script lang="ts">
	import { onMount } from "svelte";
	import type { AnalyzerCell, AnalyzerMetrics } from "../types";

	export type SweepDimension = "aPeriod" | "bPeriod" | "takeProfit" | "stopLoss";

	type Props = {
		cells: AnalyzerCell[];
		xDimension: SweepDimension;
		yDimension: SweepDimension;
		zMetric: keyof AnalyzerMetrics;
		selectedCell: AnalyzerCell | null;
	};

	let { cells, xDimension, yDimension, zMetric, selectedCell = $bindable() }: Props = $props();

	let containerEl: HTMLDivElement;
	let canvasEl: HTMLCanvasElement;

	let three: any = null;
	let scene: any = null;
	let camera: any = null;
	let renderer: any = null;
	let controls: any = null;
	let raycaster: any = null;
	let mouse: any = null;
	let points: any = null;
	let pointCells: AnalyzerCell[] = [];

	let disposed = false;
	let rafId: number | null = null;
	let resizeObserver: ResizeObserver | null = null;

	const metricValue = (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => {
		switch (metric) {
			case "netPnl":
				return metrics.netPnl;
			case "endingEquity":
				return metrics.endingEquity;
			case "sharpe":
				return metrics.sharpe;
			case "sortino":
				return metrics.sortino;
			case "maxDrawdown":
				return metrics.maxDrawdown;
			case "profitFactor":
				return metrics.profitFactor;
			case "winRate":
				return metrics.winRate;
			case "trades":
				return metrics.trades;
			case "fitness":
				return metrics.fitness;
			case "totalReward":
				return metrics.totalReward;
			case "steps":
				return metrics.steps;
			default:
				return metrics.fitness;
		}
	};

	const dimValue = (cell: AnalyzerCell, dim: SweepDimension) => {
		switch (dim) {
			case "aPeriod":
				return cell.aPeriod;
			case "bPeriod":
				return cell.bPeriod;
			case "takeProfit":
				return cell.takeProfit;
			case "stopLoss":
				return cell.stopLoss;
			default:
				return null;
		}
	};

	const toPlotValue = (value: number | null, min: number, max: number) => {
		if (value === null || !Number.isFinite(value)) {
			const span = max > min ? max - min : 1;
			return min - span * 0.15;
		}
		return value;
	};

	const normalize = (value: number, min: number, max: number, span = 120) => {
		if (!Number.isFinite(value)) return 0;
		if (max <= min) return 0;
		const t = (value - min) / (max - min);
		return (t - 0.5) * span;
	};

	const disposePoints = () => {
		if (!points || !scene) return;
		scene.remove(points);
		if (points.geometry) points.geometry.dispose();
		if (points.material) points.material.dispose();
		points = null;
		pointCells = [];
	};

	const rebuildPoints = () => {
		if (!three || !scene) return;
		disposePoints();
		if (!cells || cells.length === 0) return;

		const candidates = cells
			.map((cell) => {
				const xRaw = dimValue(cell, xDimension);
				const yRaw = dimValue(cell, yDimension);
				const zRaw = metricValue(cell.metrics, zMetric);
				if (!Number.isFinite(zRaw)) return null;
				return { cell, xRaw, yRaw, zRaw };
			})
			.filter((item): item is { cell: AnalyzerCell; xRaw: number | null; yRaw: number | null; zRaw: number } => item !== null);

		if (candidates.length === 0) return;

		const xFinite = candidates
			.map((entry) => entry.xRaw)
			.filter((v): v is number => v !== null && Number.isFinite(v));
		const yFinite = candidates
			.map((entry) => entry.yRaw)
			.filter((v): v is number => v !== null && Number.isFinite(v));
		const zFinite = candidates.map((entry) => entry.zRaw).filter((v) => Number.isFinite(v));

		const xMin = xFinite.length > 0 ? Math.min(...xFinite) : 0;
		const xMax = xFinite.length > 0 ? Math.max(...xFinite) : 1;
		const yMin = yFinite.length > 0 ? Math.min(...yFinite) : 0;
		const yMax = yFinite.length > 0 ? Math.max(...yFinite) : 1;
		const zMin = zFinite.length > 0 ? Math.min(...zFinite) : 0;
		const zMax = zFinite.length > 0 ? Math.max(...zFinite) : 1;

		const positions = new Float32Array(candidates.length * 3);
		const colors = new Float32Array(candidates.length * 3);
		const tempColor = new three.Color();
		pointCells = new Array(candidates.length);

		for (let i = 0; i < candidates.length; i++) {
			const entry = candidates[i];
			const xVal = toPlotValue(entry.xRaw, xMin, xMax);
			const yVal = toPlotValue(entry.yRaw, yMin, yMax);

			positions[i * 3] = normalize(xVal, xMin, xMax);
			positions[i * 3 + 1] = normalize(yVal, yMin, yMax);
			positions[i * 3 + 2] = normalize(entry.zRaw, zMin, zMax);

			const t = zMax > zMin ? (entry.zRaw - zMin) / (zMax - zMin) : 0.5;
			tempColor.setHSL(0.66 - 0.66 * t, 0.8, 0.52);
			colors[i * 3] = tempColor.r;
			colors[i * 3 + 1] = tempColor.g;
			colors[i * 3 + 2] = tempColor.b;

			pointCells[i] = entry.cell;
		}

		const geometry = new three.BufferGeometry();
		geometry.setAttribute("position", new three.BufferAttribute(positions, 3));
		geometry.setAttribute("color", new three.BufferAttribute(colors, 3));

		const material = new three.PointsMaterial({
			size: 3.8,
			vertexColors: true,
			sizeAttenuation: true,
			opacity: 0.95,
			transparent: true
		});

		points = new three.Points(geometry, material);
		scene.add(points);
	};

	const handlePointSelect = (event: PointerEvent) => {
		if (!raycaster || !mouse || !renderer || !camera || !points) return;
		const rect = renderer.domElement.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return;

		mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
		raycaster.setFromCamera(mouse, camera);
		const hits = raycaster.intersectObject(points, false);
		if (hits.length === 0) return;
		const index = hits[0]?.index;
		if (index === undefined || index === null) return;
		selectedCell = pointCells[index] ?? null;
	};

	const resize = () => {
		if (!renderer || !camera || !containerEl) return;
		const width = Math.max(1, containerEl.clientWidth);
		const height = Math.max(320, containerEl.clientHeight);
		renderer.setSize(width, height, false);
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
	};

onMount(() => {
	void (async () => {
		const THREE = await import("three");
		const controlsModule = await import("three/examples/jsm/controls/OrbitControls.js");
		if (disposed) return;

		three = THREE;
		scene = new THREE.Scene();
		scene.background = new THREE.Color("#0b1020");

		camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
		camera.position.set(90, 90, 120);

		renderer = new THREE.WebGLRenderer({
			canvas: canvasEl,
			antialias: true,
			alpha: false
		});
		renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

		controls = new controlsModule.OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.06;
		controls.target.set(0, 0, 0);

		raycaster = new THREE.Raycaster();
		raycaster.params.Points.threshold = 3.2;
		mouse = new THREE.Vector2();

		const gridHelper = new THREE.GridHelper(120, 12, 0x334155, 0x1e293b);
		scene.add(gridHelper);
		scene.add(new THREE.AxesHelper(72));

		renderer.domElement.addEventListener("pointerdown", handlePointSelect);

		resizeObserver = new ResizeObserver(() => resize());
		resizeObserver.observe(containerEl);
		resize();
		rebuildPoints();

		const animate = () => {
			if (disposed) return;
			rafId = window.requestAnimationFrame(animate);
			controls.update();
			renderer.render(scene, camera);
		};
		animate();
	})();

	return () => {
		disposed = true;
		if (rafId !== null) {
			window.cancelAnimationFrame(rafId);
			rafId = null;
		}
		if (resizeObserver) {
			resizeObserver.disconnect();
			resizeObserver = null;
		}
		renderer?.domElement?.removeEventListener("pointerdown", handlePointSelect);
		disposePoints();
		controls?.dispose?.();
		renderer?.dispose?.();
		three = null;
		scene = null;
		camera = null;
		renderer = null;
		controls = null;
		raycaster = null;
		mouse = null;
	};
});

	$effect(() => {
		cells;
		xDimension;
		yDimension;
		zMetric;
		rebuildPoints();
	});
</script>

<div bind:this={containerEl} class="relative h-[420px] w-full overflow-hidden rounded-lg border bg-slate-950">
	<canvas bind:this={canvasEl} class="h-full w-full"></canvas>
	{#if cells.length === 0}
		<div class="pointer-events-none absolute inset-0 grid place-items-center text-sm text-slate-300/90">
			No points to display for this slice.
		</div>
	{/if}
</div>

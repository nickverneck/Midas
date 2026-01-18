<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import {
        Chart,
        Title,
        Tooltip,
        Legend,
        LineElement,
        LinearScale,
        PointElement,
        CategoryScale,
        LineController,
        ScatterController,
        Decimation,
        type ChartConfiguration,
        type ChartType
    } from 'chart.js';

    Chart.register(
        Title,
        Tooltip,
        Legend,
        LineElement,
        LinearScale,
        PointElement,
        CategoryScale,
        LineController,
        ScatterController,
        Decimation
    );

    type ChartProps = {
        data?: ChartConfiguration['data'];
        options?: ChartConfiguration['options'];
        type?: ChartType;
    };

    let { data, options, type }: ChartProps = $props();
    let canvas: HTMLCanvasElement;
    let chart: Chart | null = null;
    let destroyed = false;
    let mounted = $state(false);
    let zoomRegistered = false;
    let zoomRegisterPromise: Promise<void> | null = null;

    const ensureZoomPlugin = async () => {
        if (zoomRegistered || typeof window === 'undefined') return;
        if (!zoomRegisterPromise) {
            zoomRegisterPromise = import('chartjs-plugin-zoom')
                .then((mod) => {
                    Chart.register(mod.default ?? mod);
                    zoomRegistered = true;
                })
                .catch(() => {
                    // Zoom is optional; charts still render without it.
                });
        }
        await zoomRegisterPromise;
    };

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        normalized: true,
        interaction: {
            mode: 'index' as const,
            intersect: false,
        },
        elements: {
            point: {
                radius: 2
            }
        },
        plugins: {
            decimation: {
                enabled: false,
                algorithm: 'min-max' as const,
                samples: 400
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x' as const,
                    modifierKey: 'shift' as const
                },
                zoom: {
                    wheel: {
                        enabled: true,
                        modifierKey: 'ctrl' as const
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'x' as const
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                type: 'linear' as const,
                display: true,
                position: 'left' as const,
            },
            y1: {
                type: 'linear' as const,
                display: false,
                position: 'right' as const,
                grid: {
                    drawOnChartArea: false,
                },
            },
        }
    };

    const mergeOptions = (customOptions?: ChartConfiguration['options']) => ({
        ...defaultOptions,
        ...(customOptions ?? {}),
        interaction: {
            ...defaultOptions.interaction,
            ...(customOptions?.interaction ?? {})
        },
        elements: {
            ...defaultOptions.elements,
            ...(customOptions?.elements ?? {})
        },
        plugins: {
            ...defaultOptions.plugins,
            ...(customOptions?.plugins ?? {})
        },
        scales: {
            ...defaultOptions.scales,
            ...(customOptions?.scales ?? {})
        }
    });

    const snapshotData = (raw: any) => {
        const labels = Array.isArray(raw?.labels) ? [...raw.labels] : [];
        const datasets = Array.isArray(raw?.datasets)
            ? raw.datasets.map((ds: any) => ({
                ...ds,
                data: Array.isArray(ds.data) ? [...ds.data] : []
            }))
            : [];
        const seriesCount = datasets.reduce((max: number, ds: any) => {
            const len = Array.isArray(ds.data) ? ds.data.length : 0;
            return Math.max(max, len);
        }, 0);
        const usesXY = datasets.some((ds: any) =>
            Array.isArray(ds.data) &&
            ds.data.some((point: any) => point && typeof point === 'object' && ('x' in point || 'y' in point))
        );

        return { labels, datasets, seriesCount, usesXY };
    };

    $effect(() => {
        if (!mounted || !canvas) return;
        let cancelled = false;

        const setup = async () => {
            await ensureZoomPlugin();
            if (cancelled || destroyed || !mounted || !canvas) return;

            const { labels, datasets, seriesCount, usesXY } = snapshotData(data);
            const nextOptions = mergeOptions(options);
            const nextType = (type ?? 'line') as ChartType;

            if (!chart || (chart.config as ChartConfiguration).type !== nextType) {
                if (chart) chart.destroy();
                chart = new Chart(canvas, {
                    type: nextType,
                    data: { labels, datasets },
                    options: nextOptions
                });
            } else {
                chart.data = { labels, datasets };
                chart.options = nextOptions;
            }

            if (!chart) return;
            if (chart.options.scales?.y1) {
                chart.options.scales.y1.display = datasets.some((d: any) => d.yAxisID === 'y1');
            }
            chart.options.interaction = {
                ...(chart.options.interaction ?? {}),
                mode: nextType === 'scatter' ? 'nearest' : 'index'
            };
            chart.options.parsing = usesXY ? false : undefined;
            chart.options.animation = seriesCount > 2000 ? false : undefined;
            if (chart.options.elements?.point) {
                chart.options.elements.point.radius = seriesCount > 1000 ? 0 : 2;
            }
            if (chart.options.plugins?.decimation) {
                chart.options.plugins.decimation.enabled = usesXY && seriesCount > 2000;
            }
            if (chart.options.plugins?.zoom) {
                const zoomMode = nextType === 'scatter' ? 'xy' : 'x';
				const zoomPlugin = chart.options.plugins.zoom as any;
				if (zoomPlugin.pan) zoomPlugin.pan.mode = zoomMode;
				if (zoomPlugin.zoom) zoomPlugin.zoom.mode = zoomMode;
            }
            chart.update();
            chart.resize();
        };

        void setup();
        return () => {
            cancelled = true;
        };
    });

    onMount(() => {
        mounted = true;
        return () => {
            mounted = false;
        };
    });

    onDestroy(() => {
        destroyed = true;
        if (chart) chart.destroy();
        chart = null;
    });
</script>

<div class="w-full h-full">
    <canvas bind:this={canvas}></canvas>
</div>

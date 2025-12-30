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

    let { data, options = {}, type = 'line' } = $props();
    let canvas: HTMLCanvasElement;
    let chart: Chart;
    let destroyed = false;
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
            mode: 'index',
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
                algorithm: 'min-max',
                samples: 400
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: 'x',
                    modifierKey: 'shift'
                },
                zoom: {
                    wheel: {
                        enabled: true,
                        modifierKey: 'ctrl'
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'x'
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                type: 'linear',
                display: true,
                position: 'left',
            },
            y1: {
                type: 'linear',
                display: false,
                position: 'right',
                grid: {
                    drawOnChartArea: false,
                },
            },
        }
    };

    const mergeOptions = () => ({
        ...defaultOptions,
        ...options,
        interaction: {
            ...defaultOptions.interaction,
            ...(options?.interaction ?? {})
        },
        elements: {
            ...defaultOptions.elements,
            ...(options?.elements ?? {})
        },
        plugins: {
            ...defaultOptions.plugins,
            ...(options?.plugins ?? {})
        },
        scales: {
            ...defaultOptions.scales,
            ...(options?.scales ?? {})
        }
    });

    $effect(() => {
        if (!chart) return;
        const labels = Array.isArray(data?.labels) ? [...data.labels] : [];
        const datasets = Array.isArray(data?.datasets)
            ? data.datasets.map((ds) => ({
                ...ds,
                data: Array.isArray(ds.data) ? [...ds.data] : []
            }))
            : [];
        const seriesCount = datasets.reduce((max, ds) => {
            const len = Array.isArray(ds.data) ? ds.data.length : 0;
            return Math.max(max, len);
        }, 0);

        chart.data.labels = labels;
        chart.data.datasets = datasets;
        chart.options = mergeOptions();
        if (chart.options.scales?.y1) {
            chart.options.scales.y1.display = datasets.some(d => d.yAxisID === 'y1');
        }
        chart.options.interaction = {
            ...(chart.options.interaction ?? {}),
            mode: type === 'scatter' ? 'nearest' : 'index'
        };
        const usesXY = datasets.some((ds) =>
            Array.isArray(ds.data) &&
            ds.data.some((point) => point && typeof point === 'object' && ('x' in point || 'y' in point))
        );
        chart.options.parsing = usesXY ? false : undefined;
        chart.options.animation = seriesCount > 2000 ? false : undefined;
        if (chart.options.elements?.point) {
            chart.options.elements.point.radius = seriesCount > 1000 ? 0 : 2;
        }
        if (chart.options.plugins?.decimation) {
            chart.options.plugins.decimation.enabled = usesXY && seriesCount > 2000;
        }
        if (chart.options.plugins?.zoom) {
            const zoomMode = type === 'scatter' ? 'xy' : 'x';
            chart.options.plugins.zoom.pan.mode = zoomMode;
            chart.options.plugins.zoom.zoom.mode = zoomMode;
        }
        chart.update('none');
    });

    onMount(() => {
        const setup = async () => {
            await ensureZoomPlugin();
            if (destroyed) return;
            const labels = Array.isArray(data?.labels) ? [...data.labels] : [];
            const datasets = Array.isArray(data?.datasets)
                ? data.datasets.map((ds) => ({
                    ...ds,
                    data: Array.isArray(ds.data) ? [...ds.data] : []
                }))
                : [];
            const config: ChartConfiguration = {
                type: type as ChartType,
                data: { labels, datasets },
                options: mergeOptions()
            };
            chart = new Chart(canvas, config);
        };
        void setup();
    });

    onDestroy(() => {
        destroyed = true;
        if (chart) chart.destroy();
    });
</script>

<div class="w-full h-full">
    <canvas bind:this={canvas}></canvas>
</div>

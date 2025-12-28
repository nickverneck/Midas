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
        Decimation,
        type ChartConfiguration
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
        Decimation
    );

    let { data, options = {} } = $props();
    let canvas: HTMLCanvasElement;
    let chart: Chart;

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        normalized: true,
        parsing: false,
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
        chart.options.scales.y1.display = datasets.some(d => d.yAxisID === 'y1');
        chart.options.animation = seriesCount > 2000 ? false : undefined;
        if (chart.options.elements?.point) {
            chart.options.elements.point.radius = seriesCount > 1000 ? 0 : 2;
        }
        if (chart.options.plugins?.decimation) {
            chart.options.plugins.decimation.enabled = seriesCount > 2000;
        }
        chart.update('none');
    });

    onMount(() => {
        const labels = Array.isArray(data?.labels) ? [...data.labels] : [];
        const datasets = Array.isArray(data?.datasets)
            ? data.datasets.map((ds) => ({
                ...ds,
                data: Array.isArray(ds.data) ? [...ds.data] : []
            }))
            : [];
        const config: ChartConfiguration = {
            type: 'line',
            data: { labels, datasets },
            options: { ...defaultOptions, ...options }
        };
        chart = new Chart(canvas, config);
    });

    onDestroy(() => {
        if (chart) chart.destroy();
    });
</script>

<div class="w-full h-full">
    <canvas bind:this={canvas}></canvas>
</div>

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
        LineController
    );

    let { data, options = {} } = $props();
    let canvas: HTMLCanvasElement;
    let chart: Chart;

    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
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
                display: data.datasets.some(d => d.yAxisID === 'y1'),
                position: 'right',
                grid: {
                    drawOnChartArea: false,
                },
            },
        }
    };

    $effect(() => {
        if (chart) {
            chart.data = data;
            // Update y1 visibility based on new data
            chart.options.scales.y1.display = data.datasets.some(d => d.yAxisID === 'y1');
            chart.update('none');
        }
    });

    $effect(() => {
        const seriesCount = data?.datasets?.[0]?.data?.length || 0;
        if (chart) {
            chart.options.animation = seriesCount > 2000 ? false : undefined;
            chart.update('none');
        }
    });

    onMount(() => {
        const config: ChartConfiguration = {
            type: 'line',
            data: data,
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

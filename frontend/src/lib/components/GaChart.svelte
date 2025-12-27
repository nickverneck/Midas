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

    let { data } = $props();
    let canvas: HTMLCanvasElement;
    let chart: Chart;

    $effect(() => {
        if (chart) {
            chart.data = data;
            chart.update();
        }
    });

    onMount(() => {
        const config: ChartConfiguration = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
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

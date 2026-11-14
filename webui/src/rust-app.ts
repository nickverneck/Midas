import { mount } from 'svelte';
import App from './routes/+page.svelte';

const target = document.getElementById('app');
if (!target) throw new Error('Missing application root');
mount(App, { target });

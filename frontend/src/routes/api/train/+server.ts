import { json } from '@sveltejs/kit';
import { spawn } from 'child_process';
import { Readable } from 'stream';

export const POST = async ({ request }) => {
    const params = await request.json();

    // Construct arguments for the python script
    const args = ['python/examples/train_ga.py'];

    for (const [key, value] of Object.entries(params)) {
        if (value === true) {
            args.push(`--${key}`);
        } else if (value !== false && value !== null && value !== undefined && value !== '') {
            args.push(`--${key}`, String(value));
        }
    }

    const stream = new ReadableStream({
        start(controller) {
            const process = spawn('python3', args, {
                cwd: '../../' // Root of the project
            });

            process.stdout.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stdout', content: data.toString() })}\n\n`);
            });

            process.stderr.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stderr', content: data.toString() })}\n\n`);
            });

            process.on('close', (code) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'exit', code })}\n\n`);
                controller.close();
            });

            process.on('error', (err) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'error', content: err.message })}\n\n`);
                controller.close();
            });
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    });
};

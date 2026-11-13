import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { mkdtemp, readFile, rm, writeFile, chmod } from 'node:fs/promises';

const webuiDir = path.resolve(import.meta.dirname, '..');

async function availablePort() {
	const server = createServer();
	await new Promise((resolve, reject) => server.once('error', reject).listen(0, '127.0.0.1', resolve));
	const { port } = server.address();
	await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
	return port;
}

test('built preview invokes the default CLI from trader cwd with launch-root config resolution', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-preview-'));
	const traderDir = path.join(dir, 'trader-root');
	const fakeCli = path.join(traderDir, 'target', 'debug', 'trader');
	const capturePath = path.join(dir, 'invocation.json');
	const configPath = './preview-provider-config.toml';
	const port = await availablePort();
	const fakePayload = JSON.stringify({ bars: [{ ts_ns: '1735689600123456789', open: 1, high: 2, low: 0, close: 1 }] });
	await (await import('node:fs/promises')).mkdir(path.dirname(fakeCli), { recursive: true });
	await writeFile(fakeCli, `#!/usr/bin/env node\nconst fs=require('node:fs');fs.writeFileSync(${JSON.stringify(capturePath)},JSON.stringify({args:process.argv.slice(2),cwd:process.cwd()}));process.stdout.write(${JSON.stringify(fakePayload)});\n`);
	await chmod(fakeCli, 0o700);
	const env = { ...process.env, MIDAS_TRADER_DIR: traderDir, MIDAS_CONFIG: configPath };
	delete env.MIDAS_TRADER_BIN;
	const preview = spawn('npm', ['run', 'preview', '--', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
		cwd: webuiDir,
		env,
		stdio: 'ignore',
		windowsHide: true
	});
	try {
		const url = `http://127.0.0.1:${port}/api/standalone-market`;
		let response;
		const deadline = Date.now() + 15_000;
		while (Date.now() < deadline) {
			if (preview.exitCode !== null) throw new Error(`npm run preview exited with status ${preview.exitCode}`);
			try {
				response = await fetch(url, {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 })
				});
				break;
			} catch { await new Promise((resolve) => setTimeout(resolve, 100)); }
		}
		assert.ok(response, 'preview server became ready');
		assert.equal(response.status, 200, await response.text());
		const invocation = JSON.parse(await readFile(capturePath, 'utf8'));
		assert.equal(invocation.cwd, traderDir);
		assert.deepEqual(invocation.args.slice(0, 2), ['--config', path.resolve(webuiDir, configPath)]);
		assert.deepEqual(invocation.args.slice(2), ['standalone-market', '--contract', 'ESZ6', '--bar-kind', 'minute', '--bar-value', '1', '--timeout-seconds', '15']);
	} finally {
		preview.kill('SIGTERM');
		await new Promise((resolve) => preview.once('close', resolve));
		await rm(dir, { recursive: true, force: true });
	}
});

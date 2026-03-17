#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const usage = `Usage:
  node scripts/run_train_via_api.mjs --engine ga|rl --params-file <file.json> [options]

Options:
  --host <url>          Frontend base URL. Default: http://127.0.0.1:4173
  --profile <profile>   Cargo profile: debug|release. Default: release
  --summary-out <file>  Optional path to write the run summary JSON
  --log-limit <n>       Rows per /api/logs fetch. Default: 500
`;

const parseArgs = (argv) => {
  const out = {
    host: 'http://127.0.0.1:4173',
    profile: 'release',
    logLimit: 500,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--engine') {
      out.engine = argv[++i];
    } else if (arg === '--params-file') {
      out.paramsFile = argv[++i];
    } else if (arg === '--host') {
      out.host = argv[++i];
    } else if (arg === '--profile') {
      out.profile = argv[++i];
    } else if (arg === '--summary-out') {
      out.summaryOut = argv[++i];
    } else if (arg === '--log-limit') {
      out.logLimit = Number(argv[++i]);
    } else if (arg === '--help' || arg === '-h') {
      out.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return out;
};

const readJsonFile = (filePath) => JSON.parse(fs.readFileSync(filePath, 'utf8'));

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const fetchJson = async (url) => {
  const response = await fetch(url, {
    headers: { accept: 'application/json' },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error || `Request failed: ${response.status}`);
  }
  return data;
};

const toNumber = (value) => {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const chooseMetricKey = (engine, rows) => {
  if (engine === 'ga') {
    const gaKeys = ['selection_fitness', 'eval_fitness', 'fitness'];
    return gaKeys.find((key) => rows.some((row) => toNumber(row[key]) !== null)) || 'fitness';
  }
  return 'fitness';
};

const bestRowBy = (rows, metricKey) => {
  let best = null;
  for (const row of rows) {
    const metric = toNumber(row[metricKey]);
    if (metric === null) continue;
    if (!best || metric > best.metric) {
      best = { metric, row };
    }
  }
  return best?.row ?? null;
};

const bestRowByAnyMetric = (rows, metricKeys) => {
  for (const metricKey of metricKeys) {
    const best = bestRowBy(rows, metricKey);
    if (best) {
      return { metricKey, row: best };
    }
  }
  return { metricKey: null, row: null };
};

const collectLogs = async ({ host, runDir, logType, key, limit }) => {
  let offset = 0;
  const rows = [];
  while (true) {
    const url = new URL('/api/logs', host);
    url.searchParams.set('dir', runDir);
    url.searchParams.set('log', logType);
    url.searchParams.set('offset', String(offset));
    url.searchParams.set('limit', String(limit));
    const payload = await fetchJson(url);
    rows.push(...(payload.data || []));
    offset = payload.nextOffset ?? offset + (payload.data?.length || 0);
    if (payload.done) {
      break;
    }
    if (!payload.data || payload.data.length === 0) {
      await sleep(300);
    }
  }

  const summaryUrl = new URL('/api/logs', host);
  summaryUrl.searchParams.set('dir', runDir);
  summaryUrl.searchParams.set('log', logType);
  summaryUrl.searchParams.set('mode', 'summary');
  summaryUrl.searchParams.set('key', key);
  const summary = await fetchJson(summaryUrl);
  return { rows, summary: summary.data || [] };
};

const parseStdoutInsights = (combinedOutput) => {
  const testLine = combinedOutput
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line.startsWith('test |'));
  const saveLines = combinedOutput
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('Saved final') || line.startsWith('Saved checkpoint'));
  return {
    testLine: testLine || null,
    saveLines,
  };
};

const loadRunArtifacts = (runDir) => {
  const root = process.cwd();
  const resolved = path.resolve(root, runDir);
  const stackPath = path.join(resolved, 'training_stack.json');
  const summaryPath = path.join(resolved, 'run_summary.json');
  const stack = fs.existsSync(stackPath)
    ? JSON.parse(fs.readFileSync(stackPath, 'utf8'))
    : null;
  return { resolved, stackPath, summaryPath, stack };
};

const streamTraining = async ({ host, engine, params, profile }) => {
  const response = await fetch(new URL('/api/train', host), {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'text/event-stream',
    },
    body: JSON.stringify({ engine, params, profile }),
  });

  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(text || `Training request failed: ${response.status}`);
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = '';
  let runDir = null;
  let exitCode = null;
  let combinedOutput = '';

  const processEvent = (rawEvent) => {
    const trimmed = rawEvent.trim();
    if (!trimmed) return;
    const dataLine = trimmed
      .split('\n')
      .find((line) => line.startsWith('data: '));
    if (!dataLine) return;
    const payload = JSON.parse(dataLine.slice(6));
    const content = typeof payload.content === 'string' ? payload.content : '';

    if (payload.type === 'stdout' || payload.type === 'stderr') {
      combinedOutput += content;
      const matcher = content.match(/info: run directory ([^\n\r]+)/);
      if (matcher) {
        runDir = matcher[1].trim();
      }
      const stream = payload.type === 'stderr' ? process.stderr : process.stdout;
      stream.write(content);
      return;
    }

    if (payload.type === 'exit') {
      exitCode = Number(payload.code ?? 0);
      return;
    }

    if (payload.type === 'error') {
      throw new Error(content || 'Training stream error');
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() ?? '';
    for (const event of events) {
      processEvent(event);
    }
  }

  if (buffer.trim() !== '') {
    processEvent(buffer);
  }

  return { runDir, exitCode, combinedOutput };
};

const main = async () => {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage);
    return;
  }
  if (!args.engine || !args.paramsFile) {
    throw new Error(`Missing required arguments.\n\n${usage}`);
  }

  const params = readJsonFile(args.paramsFile);
  const { runDir, exitCode, combinedOutput } = await streamTraining({
    host: args.host,
    engine: args.engine,
    params,
    profile: args.profile,
  });

  if (!runDir) {
    throw new Error('Training completed without emitting a run directory.');
  }

  const logType = args.engine === 'ga' ? 'ga' : 'rl';
  const summaryKey = args.engine === 'ga' ? 'gen' : 'epoch';
  const { rows, summary } = await collectLogs({
    host: args.host,
    runDir,
    logType,
    key: summaryKey,
    limit: args.logLimit,
  });
  const metricKey = chooseMetricKey(args.engine, rows);
  const finalRow = rows.length > 0 ? rows[rows.length - 1] : null;
  const bestRow = bestRowBy(rows, metricKey);
  const bestEvalPnl = bestRowByAnyMetric(rows, ['eval_pnl', 'eval_pnl_total', 'eval_pnl_realized']);
  const artifacts = loadRunArtifacts(runDir);
  const stdoutInsights = parseStdoutInsights(combinedOutput);

  const runSummary = {
    engine: args.engine,
    profile: args.profile,
    host: args.host,
    params,
    runDir,
    exitCode,
    metricKey,
    rowCount: rows.length,
    fitnessSummary: summary,
    finalRow,
    bestRow,
    bestEvalPnlMetricKey: bestEvalPnl.metricKey,
    bestEvalPnlRow: bestEvalPnl.row,
    stdoutInsights,
    trainingStack: artifacts.stack,
    generatedAt: new Date().toISOString(),
  };

  if (args.summaryOut) {
    fs.mkdirSync(path.dirname(args.summaryOut), { recursive: true });
    fs.writeFileSync(args.summaryOut, `${JSON.stringify(runSummary, null, 2)}\n`);
  }

  if (artifacts.summaryPath.startsWith(artifacts.resolved)) {
    fs.writeFileSync(artifacts.summaryPath, `${JSON.stringify(runSummary, null, 2)}\n`);
  }

  console.log('\n=== Run Summary ===');
  console.log(JSON.stringify(runSummary, null, 2));

  if (exitCode !== 0) {
    process.exitCode = exitCode || 1;
  }
};

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

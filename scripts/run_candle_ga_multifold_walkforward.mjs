#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const usage = `Usage:
  node scripts/run_candle_ga_multifold_walkforward.mjs [options]

Options:
  --host <url>         Frontend base URL. Default: http://127.0.0.1:4173
  --profile <profile>  Cargo profile: debug|release. Default: release
  --data-root <path>   Root folder containing dated train folders. Default: data/train
  --symbol <file>      Instrument parquet file name. Default: ES_F.parquet
  --window <n>         Window size. Default: 700
  --step <n>           Step size. Default: 128
  --generations <n>    GA generations per fold. Default: 120
  --pop-size <n>       GA population size. Default: 12
  --workers <n>        GA worker threads. Default: 2
  --eval-windows <n>   GA eval/test window cap. Default: 999
  --seed <n>           GA RNG seed. Default: 42
  --max-hold-bars-positive <n>      Soft max hold bars while in profit. Default: 100
  --max-hold-bars-drawdown <n>      Soft max hold bars while in drawdown. Default: 50
  --hold-duration-penalty <x>       Hold penalty after soft max. Default: 0.1
  --hold-duration-penalty-growth <x> Hold penalty growth after soft max. Default: 0
  --min-hold-bars <n>               Minimum bars before exit/flip. Default: 25
  --early-exit-penalty <x>          Penalty per missing bar on early exit. Default: 1
  --early-flip-penalty <x>          Penalty per missing bar on early flip. Default: 2
  --flat-hold-penalty <x>           Flat hold penalty after max flat bars. Default: 0.25
  --flat-hold-penalty-growth <x>    Flat hold penalty growth. Default: 0
  --max-flat-hold-bars <n>          Flat hold grace bars. Default: 200
  --help               Show this message
`;

const parseArgs = (argv) => {
  const out = {
    host: 'http://127.0.0.1:4173',
    profile: 'release',
    dataRoot: 'data/train',
    symbol: 'ES_F.parquet',
    window: 700,
    step: 128,
    generations: 120,
    popSize: 12,
    workers: 2,
    evalWindows: 999,
    seed: 42,
    maxHoldBarsPositive: 100,
    maxHoldBarsDrawdown: 50,
    holdDurationPenalty: 0.1,
    holdDurationPenaltyGrowth: 0,
    minHoldBars: 25,
    earlyExitPenalty: 1,
    earlyFlipPenalty: 2,
    flatHoldPenalty: 0.25,
    flatHoldPenaltyGrowth: 0,
    maxFlatHoldBars: 200,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--host') {
      out.host = argv[++i];
    } else if (arg === '--profile') {
      out.profile = argv[++i];
    } else if (arg === '--data-root') {
      out.dataRoot = argv[++i];
    } else if (arg === '--symbol') {
      out.symbol = argv[++i];
    } else if (arg === '--window') {
      out.window = Number(argv[++i]);
    } else if (arg === '--step') {
      out.step = Number(argv[++i]);
    } else if (arg === '--generations') {
      out.generations = Number(argv[++i]);
    } else if (arg === '--pop-size') {
      out.popSize = Number(argv[++i]);
    } else if (arg === '--workers') {
      out.workers = Number(argv[++i]);
    } else if (arg === '--eval-windows') {
      out.evalWindows = Number(argv[++i]);
    } else if (arg === '--seed') {
      out.seed = Number(argv[++i]);
    } else if (arg === '--max-hold-bars-positive') {
      out.maxHoldBarsPositive = Number(argv[++i]);
    } else if (arg === '--max-hold-bars-drawdown') {
      out.maxHoldBarsDrawdown = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty') {
      out.holdDurationPenalty = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty-growth') {
      out.holdDurationPenaltyGrowth = Number(argv[++i]);
    } else if (arg === '--min-hold-bars') {
      out.minHoldBars = Number(argv[++i]);
    } else if (arg === '--early-exit-penalty') {
      out.earlyExitPenalty = Number(argv[++i]);
    } else if (arg === '--early-flip-penalty') {
      out.earlyFlipPenalty = Number(argv[++i]);
    } else if (arg === '--flat-hold-penalty') {
      out.flatHoldPenalty = Number(argv[++i]);
    } else if (arg === '--flat-hold-penalty-growth') {
      out.flatHoldPenaltyGrowth = Number(argv[++i]);
    } else if (arg === '--max-flat-hold-bars') {
      out.maxFlatHoldBars = Number(argv[++i]);
    } else if (arg === '--help' || arg === '-h') {
      out.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return out;
};

const repoRoot = process.cwd();
const runViaApiScript = path.join(repoRoot, 'scripts', 'run_train_via_api.mjs');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');

const writeJson = (filePath, value) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
};

const readJson = (filePath) => JSON.parse(fs.readFileSync(filePath, 'utf8'));

const parseFolderDate = (folderName) => {
  const match = folderName.match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (!match) return null;
  const [, mm, dd, yyyy] = match;
  return new Date(`${yyyy}-${mm}-${dd}T00:00:00Z`);
};

const discoverWeeks = ({ dataRoot, symbol }) => {
  return fs
    .readdirSync(dataRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const asDate = parseFolderDate(entry.name);
      if (!asDate) return null;
      const parquetPath = path.join(dataRoot, entry.name, symbol);
      if (!fs.existsSync(parquetPath)) return null;
      return {
        folder: entry.name,
        parquetPath,
        sortKey: asDate.getTime(),
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.sortKey - b.sortKey);
};

const parseTestLine = (line) => {
  if (typeof line !== 'string' || !line.startsWith('test |')) {
    return null;
  }
  const match = line.match(/fitness ([^|]+) \| pnl ([^|]+) \| sortino ([^|]+) \| mdd ([^|]+)/);
  if (!match) {
    return { raw: line };
  }
  return {
    raw: line,
    fitness: Number(match[1].trim()),
    pnl: Number(match[2].trim()),
    sortino: Number(match[3].trim()),
    mdd: Number(match[4].trim()),
  };
};

const runStage = ({ engine, params, summaryPath, host, profile }) => {
  const paramsPath = summaryPath.replace(/\.summary\.json$/, '.params.json');
  writeJson(paramsPath, params);
  const child = spawnSync(
    process.execPath,
    [
      runViaApiScript,
      '--engine',
      engine,
      '--params-file',
      paramsPath,
      '--host',
      host,
      '--profile',
      profile,
      '--summary-out',
      summaryPath,
    ],
    {
      cwd: repoRoot,
      stdio: 'inherit',
    }
  );
  if (child.status !== 0) {
    throw new Error(`${engine} stage failed with exit code ${child.status ?? 1}`);
  }
  return readJson(summaryPath);
};

const runReplay = ({ summaryPath, replayDir, profile }) => {
  const cargoArgs =
    profile === 'release'
      ? ['run', '--release', '--features', 'backend-candle', '--bin', 'inspect_ga_policy', '--']
      : ['run', '--features', 'backend-candle', '--bin', 'inspect_ga_policy', '--'];
  const child = spawnSync(
    'cargo',
    [...cargoArgs, '--run-summary', summaryPath, '--outdir', replayDir],
    {
      cwd: repoRoot,
      stdio: 'inherit',
    }
  );
  if (child.status !== 0) {
    throw new Error(`policy replay failed with exit code ${child.status ?? 1}`);
  }
  return readJson(path.join(replayDir, 'replay_summary.json'));
};

const makeGaParams = ({ fold, outdir, args }) => ({
  backend: 'candle',
  device: 'cpu',
  outdir,
  'train-parquet': fold.train.parquetPath,
  'val-parquet': fold.val.parquetPath,
  'test-parquet': fold.test.parquetPath,
  windowed: true,
  window: args.window,
  step: args.step,
  generations: args.generations,
  'pop-size': args.popSize,
  workers: args.workers,
  'batch-candidates': 0,
  'elite-frac': 0.17,
  'parent-pool-frac': 0.67,
  'immigrant-frac': 0.17,
  'mutation-sigma': 0.05,
  'init-sigma': 0.5,
  hidden: 64,
  layers: 2,
  'eval-windows': args.evalWindows,
  'save-top-n': 0,
  'save-every': 0,
  'checkpoint-every': 0,
  'behavior-every': 0,
  'selection-use-eval': true,
  'selection-train-weight': 0.2,
  'selection-eval-weight': 0.8,
  'selection-gap-penalty': 0.1,
  'w-pnl': 1,
  'w-sortino': 1,
  'w-mdd': 0.5,
  'initial-balance': 10000,
  'max-position': 1,
  'margin-mode': 'per-contract',
  'margin-per-contract': 500,
  'contract-multiplier': 50,
  'auto-close-minutes-before-close': 5,
  'drawdown-penalty': 0,
  'drawdown-penalty-growth': 0,
  'session-close-penalty': 0,
  'max-hold-bars-positive': args.maxHoldBarsPositive,
  'max-hold-bars-drawdown': args.maxHoldBarsDrawdown,
  'hold-duration-penalty': args.holdDurationPenalty,
  'hold-duration-penalty-growth': args.holdDurationPenaltyGrowth,
  'hold-duration-penalty-positive-scale': 0.5,
  'hold-duration-penalty-negative-scale': 1.5,
  'min-hold-bars': args.minHoldBars,
  'early-exit-penalty': args.earlyExitPenalty,
  'early-flip-penalty': args.earlyFlipPenalty,
  'flat-hold-penalty': args.flatHoldPenalty,
  'flat-hold-penalty-growth': args.flatHoldPenaltyGrowth,
  'max-flat-hold-bars': args.maxFlatHoldBars,
  seed: args.seed,
});

const buildFolds = (weeks) => {
  const folds = [];
  for (let i = 0; i + 2 < weeks.length; i += 1) {
    folds.push({
      index: i + 1,
      train: weeks[i],
      val: weeks[i + 1],
      test: weeks[i + 2],
    });
  }
  return folds;
};

const main = () => {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage);
    return;
  }

  const weeks = discoverWeeks({ dataRoot: args.dataRoot, symbol: args.symbol });
  if (weeks.length < 3) {
    throw new Error(
      `Need at least 3 dated folders containing ${args.symbol} under ${args.dataRoot}; found ${weeks.length}`
    );
  }

  const folds = buildFolds(weeks);
  const symbolStem = path.basename(args.symbol, '.parquet').toLowerCase();
  const reportDir = path.join(
    repoRoot,
    'runs',
    'pipeline_reports',
    `${symbolStem}_ga_multifold_${stamp}`
  );
  fs.mkdirSync(reportDir, { recursive: true });

  const results = [];
  for (const fold of folds) {
    const label = `${symbolStem}_fold${fold.index}_${stamp}`;
    const outdir = path.join('runs_ga', label);
    const params = makeGaParams({ fold, outdir, args });
    const summaryPath = path.join(reportDir, `fold_${fold.index}.summary.json`);
    const replayDir = path.join(reportDir, `fold_${fold.index}_test_replay`);

    console.log(
      `\n=== Fold ${fold.index}: train=${fold.train.folder} val=${fold.val.folder} test=${fold.test.folder} ===`
    );
    const training = runStage({
      engine: 'ga',
      params,
      summaryPath,
      host: args.host,
      profile: args.profile,
    });
    const replay = runReplay({
      summaryPath,
      replayDir,
      profile: args.profile,
    });

    results.push({
      fold: fold.index,
      trainFolder: fold.train.folder,
      valFolder: fold.val.folder,
      testFolder: fold.test.folder,
      training,
      trainingTest: parseTestLine(training?.stdoutInsights?.testLine),
      replay,
    });
  }

  const replayPnls = results
    .map((result) => result?.replay?.metrics?.eval_pnl_total)
    .filter((value) => Number.isFinite(value));
  const replayMdds = results
    .map((result) => result?.replay?.metrics?.eval_drawdown)
    .filter((value) => Number.isFinite(value));

  const summary = {
    generatedAt: new Date().toISOString(),
    host: args.host,
    profile: args.profile,
    dataRoot: args.dataRoot,
    symbol: args.symbol,
    window: args.window,
    step: args.step,
    generations: args.generations,
    popSize: args.popSize,
    workers: args.workers,
    evalWindows: args.evalWindows,
    seed: args.seed,
    maxHoldBarsPositive: args.maxHoldBarsPositive,
    maxHoldBarsDrawdown: args.maxHoldBarsDrawdown,
    holdDurationPenalty: args.holdDurationPenalty,
    holdDurationPenaltyGrowth: args.holdDurationPenaltyGrowth,
    minHoldBars: args.minHoldBars,
    earlyExitPenalty: args.earlyExitPenalty,
    earlyFlipPenalty: args.earlyFlipPenalty,
    flatHoldPenalty: args.flatHoldPenalty,
    flatHoldPenaltyGrowth: args.flatHoldPenaltyGrowth,
    maxFlatHoldBars: args.maxFlatHoldBars,
    reportDir,
    foldCount: results.length,
    replayAvgPnl:
      replayPnls.length > 0
        ? replayPnls.reduce((sum, value) => sum + value, 0) / replayPnls.length
        : null,
    replayAvgMdd:
      replayMdds.length > 0
        ? replayMdds.reduce((sum, value) => sum + value, 0) / replayMdds.length
        : null,
    results,
  };

  writeJson(path.join(reportDir, 'pipeline_summary.json'), summary);
  console.log('\n=== Multifold Summary ===');
  console.log(JSON.stringify(summary, null, 2));
};

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}

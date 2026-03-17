#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const usage = `Usage:
  node scripts/run_candle_rl_multifold_walkforward.mjs [options]

Options:
  --host <url>              Frontend base URL. Default: http://127.0.0.1:4173
  --profile <profile>       Cargo profile: debug|release. Default: release
  --data-root <path>        Root folder containing dated train folders. Default: data/train
  --symbol <file>           Instrument parquet file name. Default: ES_F.parquet
  --algorithm <name>        RL algorithm: ppo|grpo. Default: ppo
  --window <n>              Window size. Default: 700
  --step <n>                Step size. Default: 128
  --epochs <n>              RL epochs per fold. Default: 24
  --train-windows <n>       RL train windows per epoch. Default: 24
  --ppo-epochs <n>          PPO update epochs. Default: 4
  --grpo-epochs <n>         GRPO update epochs. Default: 4
  --group-size <n>          GRPO rollout group size. Default: 8
  --eval-windows <n>        Validation/test window cap during training. Default: 999
  --hidden <n>              Hidden units. Default: 64
  --layers <n>              Hidden layers. Default: 2
  --max-position <n>        Max position. Default: 1
  --lr <n>                  Learning rate. Default: 0.0003
  --gamma <n>               Discount factor. Default: 0.99
  --lam <n>                 GAE lambda. Default: 0.95
  --clip <n>                PPO clip ratio. Default: 0.2
  --vf-coef <n>             Value loss coefficient. Default: 0.5
  --ent-coef <n>            Entropy coefficient. Default: 0.01
  --checkpoint-metric <m>   Replay checkpoint choice: fitness|eval-pnl|final. Default: fitness
  --sample                  Replay stochastically instead of argmax
  --seed <n>                Optional replay seed when --sample is set
  --max-hold-bars-positive <n>      Soft max hold bars while in profit. Default: 15
  --max-hold-bars-drawdown <n>      Soft max hold bars while in drawdown. Default: 15
  --hold-duration-penalty <x>       Hold penalty after soft max. Default: 1
  --hold-duration-penalty-growth <x> Hold penalty growth after soft max. Default: 0.05
  --hold-duration-penalty-positive-scale <x>  Hold penalty scale in profit. Default: 0.5
  --hold-duration-penalty-negative-scale <x>  Hold penalty scale in drawdown. Default: 1.5
  --min-hold-bars <n>               Minimum bars before exit/flip. Default: 0
  --early-exit-penalty <x>          Penalty per missing bar on early exit. Default: 0
  --early-flip-penalty <x>          Penalty per missing bar on early flip. Default: 0
  --flat-hold-penalty <x>           Flat hold penalty after max flat bars. Default: 2.2
  --flat-hold-penalty-growth <x>    Flat hold penalty growth. Default: 0.05
  --max-flat-hold-bars <n>          Flat hold grace bars. Default: 100
  --help                    Show this message
`;

const parseArgs = (argv) => {
  const out = {
    host: 'http://127.0.0.1:4173',
    profile: 'release',
    dataRoot: 'data/train',
    symbol: 'ES_F.parquet',
    algorithm: 'ppo',
    window: 700,
    step: 128,
    epochs: 24,
    trainWindows: 24,
    ppoEpochs: 4,
    grpoEpochs: 4,
    groupSize: 8,
    evalWindows: 999,
    hidden: 64,
    layers: 2,
    maxPosition: 1,
    lr: 0.0003,
    gamma: 0.99,
    lam: 0.95,
    clip: 0.2,
    vfCoef: 0.5,
    entCoef: 0.01,
    checkpointMetric: 'fitness',
    sample: false,
    seed: null,
    maxHoldBarsPositive: 15,
    maxHoldBarsDrawdown: 15,
    holdDurationPenalty: 1,
    holdDurationPenaltyGrowth: 0.05,
    holdDurationPenaltyPositiveScale: 0.5,
    holdDurationPenaltyNegativeScale: 1.5,
    minHoldBars: 0,
    earlyExitPenalty: 0,
    earlyFlipPenalty: 0,
    flatHoldPenalty: 2.2,
    flatHoldPenaltyGrowth: 0.05,
    maxFlatHoldBars: 100,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--host') out.host = argv[++i];
    else if (arg === '--profile') out.profile = argv[++i];
    else if (arg === '--data-root') out.dataRoot = argv[++i];
    else if (arg === '--symbol') out.symbol = argv[++i];
    else if (arg === '--algorithm') out.algorithm = argv[++i];
    else if (arg === '--window') out.window = Number(argv[++i]);
    else if (arg === '--step') out.step = Number(argv[++i]);
    else if (arg === '--epochs') out.epochs = Number(argv[++i]);
    else if (arg === '--train-windows') out.trainWindows = Number(argv[++i]);
    else if (arg === '--ppo-epochs') out.ppoEpochs = Number(argv[++i]);
    else if (arg === '--grpo-epochs') out.grpoEpochs = Number(argv[++i]);
    else if (arg === '--group-size') out.groupSize = Number(argv[++i]);
    else if (arg === '--eval-windows') out.evalWindows = Number(argv[++i]);
    else if (arg === '--hidden') out.hidden = Number(argv[++i]);
    else if (arg === '--layers') out.layers = Number(argv[++i]);
    else if (arg === '--max-position') out.maxPosition = Number(argv[++i]);
    else if (arg === '--lr') out.lr = Number(argv[++i]);
    else if (arg === '--gamma') out.gamma = Number(argv[++i]);
    else if (arg === '--lam') out.lam = Number(argv[++i]);
    else if (arg === '--clip') out.clip = Number(argv[++i]);
    else if (arg === '--vf-coef') out.vfCoef = Number(argv[++i]);
    else if (arg === '--ent-coef') out.entCoef = Number(argv[++i]);
    else if (arg === '--checkpoint-metric') out.checkpointMetric = argv[++i];
    else if (arg === '--sample') out.sample = true;
    else if (arg === '--seed') out.seed = Number(argv[++i]);
    else if (arg === '--max-hold-bars-positive') out.maxHoldBarsPositive = Number(argv[++i]);
    else if (arg === '--max-hold-bars-drawdown') out.maxHoldBarsDrawdown = Number(argv[++i]);
    else if (arg === '--hold-duration-penalty') out.holdDurationPenalty = Number(argv[++i]);
    else if (arg === '--hold-duration-penalty-growth') out.holdDurationPenaltyGrowth = Number(argv[++i]);
    else if (arg === '--hold-duration-penalty-positive-scale') out.holdDurationPenaltyPositiveScale = Number(argv[++i]);
    else if (arg === '--hold-duration-penalty-negative-scale') out.holdDurationPenaltyNegativeScale = Number(argv[++i]);
    else if (arg === '--min-hold-bars') out.minHoldBars = Number(argv[++i]);
    else if (arg === '--early-exit-penalty') out.earlyExitPenalty = Number(argv[++i]);
    else if (arg === '--early-flip-penalty') out.earlyFlipPenalty = Number(argv[++i]);
    else if (arg === '--flat-hold-penalty') out.flatHoldPenalty = Number(argv[++i]);
    else if (arg === '--flat-hold-penalty-growth') out.flatHoldPenaltyGrowth = Number(argv[++i]);
    else if (arg === '--max-flat-hold-bars') out.maxFlatHoldBars = Number(argv[++i]);
    else if (arg === '--help' || arg === '-h') out.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
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

const discoverWeeks = ({ dataRoot, symbol }) =>
  fs
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

const parseTestLine = (line) => {
  if (typeof line !== 'string' || !line.startsWith('test |')) return null;
  const match = line.match(/pnl ([^|]+) \| sortino ([^|]+) \| mdd ([^|]+)/);
  if (!match) return { raw: line };
  return {
    raw: line,
    pnl: Number(match[1].trim()),
    sortino: Number(match[2].trim()),
    mdd: Number(match[3].trim()),
  };
};

const runStage = ({ params, summaryPath, host, profile }) => {
  const paramsPath = summaryPath.replace(/\.summary\.json$/, '.params.json');
  writeJson(paramsPath, params);
  const child = spawnSync(
    process.execPath,
    [
      runViaApiScript,
      '--engine',
      'rl',
      '--params-file',
      paramsPath,
      '--host',
      host,
      '--profile',
      profile,
      '--summary-out',
      summaryPath,
    ],
    { cwd: repoRoot, stdio: 'inherit' }
  );
  if (child.status !== 0) {
    throw new Error(`rl stage failed with exit code ${child.status ?? 1}`);
  }
  return readJson(summaryPath);
};

const runReplay = ({ summaryPath, replayDir, profile, checkpointMetric, sample, seed }) => {
  const cargoArgs =
    profile === 'release'
      ? ['run', '--release', '--features', 'backend-candle', '--bin', 'inspect_rl_policy', '--']
      : ['run', '--features', 'backend-candle', '--bin', 'inspect_rl_policy', '--'];
  const args = [
    ...cargoArgs,
    '--run-summary',
    summaryPath,
    '--outdir',
    replayDir,
    '--checkpoint-metric',
    checkpointMetric,
  ];
  if (sample) {
    args.push('--sample');
    if (Number.isFinite(seed)) {
      args.push('--seed', String(seed));
    }
  }
  const child = spawnSync('cargo', args, {
    cwd: repoRoot,
    stdio: 'inherit',
  });
  if (child.status !== 0) {
    throw new Error(`policy replay failed with exit code ${child.status ?? 1}`);
  }
  return readJson(path.join(replayDir, 'replay_summary.json'));
};

const makeRlParams = ({ fold, outdir, args }) => ({
  algorithm: args.algorithm,
  backend: 'candle',
  device: 'cpu',
  outdir,
  'train-parquet': fold.train.parquetPath,
  'val-parquet': fold.val.parquetPath,
  'test-parquet': fold.test.parquetPath,
  windowed: true,
  window: args.window,
  step: args.step,
  epochs: args.epochs,
  'train-windows': args.trainWindows,
  'ppo-epochs': args.ppoEpochs,
  'grpo-epochs': args.grpoEpochs,
  'group-size': args.groupSize,
  'eval-windows': args.evalWindows,
  lr: args.lr,
  gamma: args.gamma,
  lam: args.lam,
  clip: args.clip,
  'vf-coef': args.vfCoef,
  'ent-coef': args.entCoef,
  hidden: args.hidden,
  layers: args.layers,
  'log-interval': 1,
  'checkpoint-every': 1,
  'initial-balance': 10000,
  'max-position': args.maxPosition,
  'margin-mode': 'per-contract',
  'margin-per-contract': 500,
  'contract-multiplier': 50,
  'auto-close-minutes-before-close': 5,
  'w-pnl': 1,
  'w-sortino': 1,
  'w-mdd': 0.5,
  'max-hold-bars-positive': args.maxHoldBarsPositive,
  'max-hold-bars-drawdown': args.maxHoldBarsDrawdown,
  'hold-duration-penalty': args.holdDurationPenalty,
  'hold-duration-penalty-growth': args.holdDurationPenaltyGrowth,
  'hold-duration-penalty-positive-scale': args.holdDurationPenaltyPositiveScale,
  'hold-duration-penalty-negative-scale': args.holdDurationPenaltyNegativeScale,
  'min-hold-bars': args.minHoldBars,
  'early-exit-penalty': args.earlyExitPenalty,
  'early-flip-penalty': args.earlyFlipPenalty,
  'flat-hold-penalty': args.flatHoldPenalty,
  'flat-hold-penalty-growth': args.flatHoldPenaltyGrowth,
  'max-flat-hold-bars': args.maxFlatHoldBars,
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
  if (!['ppo', 'grpo'].includes(args.algorithm)) {
    throw new Error(`Unsupported --algorithm ${args.algorithm}; expected ppo or grpo`);
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
    `${symbolStem}_${args.algorithm}_multifold_${stamp}`
  );
  fs.mkdirSync(reportDir, { recursive: true });

  const results = [];
  for (const fold of folds) {
    const label = `${symbolStem}_${args.algorithm}_fold${fold.index}_${stamp}`;
    const outdir = path.join('runs_rl', label);
    const params = makeRlParams({ fold, outdir, args });
    const summaryPath = path.join(reportDir, `fold_${fold.index}.summary.json`);
    const replayDir = path.join(reportDir, `fold_${fold.index}_test_replay`);

    console.log(
      `\n=== Fold ${fold.index}: train=${fold.train.folder} val=${fold.val.folder} test=${fold.test.folder} ===`
    );
    const training = runStage({
      params,
      summaryPath,
      host: args.host,
      profile: args.profile,
    });
    const replay = runReplay({
      summaryPath,
      replayDir,
      profile: args.profile,
      checkpointMetric: args.checkpointMetric,
      sample: args.sample,
      seed: args.seed,
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
    .map((result) => result?.replay?.metrics?.pnl)
    .filter((value) => Number.isFinite(value));
  const replayMdds = results
    .map((result) => result?.replay?.metrics?.drawdown)
    .filter((value) => Number.isFinite(value));

  const summary = {
    generatedAt: new Date().toISOString(),
    host: args.host,
    profile: args.profile,
    dataRoot: args.dataRoot,
    symbol: args.symbol,
    algorithm: args.algorithm,
    window: args.window,
    step: args.step,
    epochs: args.epochs,
    trainWindows: args.trainWindows,
    ppoEpochs: args.ppoEpochs,
    grpoEpochs: args.grpoEpochs,
    groupSize: args.groupSize,
    evalWindows: args.evalWindows,
    hidden: args.hidden,
    layers: args.layers,
    maxPosition: args.maxPosition,
    checkpointMetric: args.checkpointMetric,
    sample: args.sample,
    seed: args.seed,
    maxHoldBarsPositive: args.maxHoldBarsPositive,
    maxHoldBarsDrawdown: args.maxHoldBarsDrawdown,
    holdDurationPenalty: args.holdDurationPenalty,
    holdDurationPenaltyGrowth: args.holdDurationPenaltyGrowth,
    holdDurationPenaltyPositiveScale: args.holdDurationPenaltyPositiveScale,
    holdDurationPenaltyNegativeScale: args.holdDurationPenaltyNegativeScale,
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

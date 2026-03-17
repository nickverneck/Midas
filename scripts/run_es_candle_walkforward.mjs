#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const usage = `Usage:
  node scripts/run_es_candle_walkforward.mjs [options]

Options:
  --host <url>         Frontend base URL. Default: http://127.0.0.1:4173
  --profile <profile>  Build profile: debug|release. Default: release
  --window <n>         Window size. Default: 700
  --step <n>           Step size. Default: 128
  --mode <mode>        ga|rl|both. Default: both
  --ga-generations <n> GA generations. Default: 4
  --ga-pop-size <n>    GA population size. Default: 8
  --ga-workers <n>     GA worker threads. Default: 2
  --ga-elite-frac <n>  GA elite fraction. Default: 0.33
  --ga-parent-pool-frac <n> GA parent pool fraction. Default: 0.33
  --ga-immigrant-frac <n> GA immigrant fraction. Default: 0
  --ga-mutation-sigma <n> GA mutation sigma. Default: 0.05
  --ga-init-sigma <n>  GA init sigma. Default: 0.5
  --ga-hidden <n>      GA hidden units. Default: 64
  --ga-layers <n>      GA hidden layers. Default: 2
  --ga-max-position <n> GA max position. Default: 1
  --ga-eval-windows <n> GA eval/test windows cap. Default: 4
  --ga-seed <n>        Optional GA RNG seed
  --ga-save-top-n <n>  Intermediate policies to save per save step. Default: 3
  --ga-save-every <n>  Save cadence in generations. Default: 1
  --ga-checkpoint-every <n> Checkpoint cadence in generations. Default: 1
  --ga-behavior-every <n> Behavior CSV cadence in generations. Default: 1
  --w-pnl <n>          Fitness pnl weight. Default: 1
  --w-sortino <n>      Fitness sortino weight. Default: 1
  --w-mdd <n>          Fitness max-drawdown weight. Default: 0.5
  --drawdown-penalty <n>       GA drawdown penalty. Default: 0
  --drawdown-penalty-growth <n> GA drawdown penalty growth. Default: 0
  --session-close-penalty <n>  GA session close penalty. Default: 0
  --max-hold-bars-positive <n> GA max hold bars in profit. Default: 15
  --max-hold-bars-drawdown <n> GA max hold bars in drawdown. Default: 15
  --hold-duration-penalty <n>  GA hold duration penalty. Default: 1
  --hold-duration-penalty-growth <n> GA hold penalty growth. Default: 0.05
  --hold-duration-penalty-positive-scale <n> GA hold penalty scale in profit. Default: 0.5
  --hold-duration-penalty-negative-scale <n> GA hold penalty scale in drawdown. Default: 1.5
  --flat-hold-penalty <n>      GA flat hold penalty. Default: 2.2
  --flat-hold-penalty-growth <n> GA flat hold penalty growth. Default: 0.05
  --max-flat-hold-bars <n>     GA max flat hold bars before penalty. Default: 100
  --selection-train-weight <n> GA train fitness weight in selection. Default: 0.3
  --selection-eval-weight <n>  GA eval fitness weight in selection. Default: 0.7
  --selection-gap-penalty <n>  GA train/eval gap penalty. Default: 0.2
  --selection-use-eval         Enable GA selection on eval-aware score
  --rl-epochs <n>      RL epochs. Default: 6
  --rl-train-windows <n> RL training windows per epoch. Default: 12
  --rl-ppo-epochs <n>  PPO update epochs. Default: 4
  --rl-hidden <n>      RL hidden units. Default: 64
  --rl-layers <n>      RL hidden layers. Default: 2
  --rl-max-position <n> RL max position. Default: 1
  --rl-eval-windows <n> RL eval/test windows cap. Default: 4
  --help               Show this message
`;

const parseArgs = (argv) => {
  const args = {
    host: 'http://127.0.0.1:4173',
    profile: 'release',
    window: 700,
    step: 128,
    mode: 'both',
    gaGenerations: 4,
    gaPopSize: 8,
    gaWorkers: 2,
    gaEliteFrac: 0.33,
    gaParentPoolFrac: 0.33,
    gaImmigrantFrac: 0,
    gaMutationSigma: 0.05,
    gaInitSigma: 0.5,
    gaHidden: 64,
    gaLayers: 2,
    gaMaxPosition: 1,
    gaEvalWindows: 4,
    gaSeed: null,
    gaSaveTopN: 3,
    gaSaveEvery: 1,
    gaCheckpointEvery: 1,
    gaBehaviorEvery: 1,
    wPnl: 1,
    wSortino: 1,
    wMdd: 0.5,
    drawdownPenalty: 0,
    drawdownPenaltyGrowth: 0,
    sessionClosePenalty: 0,
    maxHoldBarsPositive: 15,
    maxHoldBarsDrawdown: 15,
    holdDurationPenalty: 1,
    holdDurationPenaltyGrowth: 0.05,
    holdDurationPenaltyPositiveScale: 0.5,
    holdDurationPenaltyNegativeScale: 1.5,
    flatHoldPenalty: 2.2,
    flatHoldPenaltyGrowth: 0.05,
    maxFlatHoldBars: 100,
    selectionTrainWeight: 0.3,
    selectionEvalWeight: 0.7,
    selectionGapPenalty: 0.2,
    selectionUseEval: false,
    rlEpochs: 6,
    rlTrainWindows: 12,
    rlPpoEpochs: 4,
    rlHidden: 64,
    rlLayers: 2,
    rlMaxPosition: 1,
    rlEvalWindows: 4,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--host') {
      args.host = argv[++i];
    } else if (arg === '--profile') {
      args.profile = argv[++i];
    } else if (arg === '--window') {
      args.window = Number(argv[++i]);
    } else if (arg === '--step') {
      args.step = Number(argv[++i]);
    } else if (arg === '--mode') {
      args.mode = argv[++i];
    } else if (arg === '--ga-generations') {
      args.gaGenerations = Number(argv[++i]);
    } else if (arg === '--ga-pop-size') {
      args.gaPopSize = Number(argv[++i]);
    } else if (arg === '--ga-workers') {
      args.gaWorkers = Number(argv[++i]);
    } else if (arg === '--ga-elite-frac') {
      args.gaEliteFrac = Number(argv[++i]);
    } else if (arg === '--ga-parent-pool-frac') {
      args.gaParentPoolFrac = Number(argv[++i]);
    } else if (arg === '--ga-immigrant-frac') {
      args.gaImmigrantFrac = Number(argv[++i]);
    } else if (arg === '--ga-mutation-sigma') {
      args.gaMutationSigma = Number(argv[++i]);
    } else if (arg === '--ga-init-sigma') {
      args.gaInitSigma = Number(argv[++i]);
    } else if (arg === '--ga-hidden') {
      args.gaHidden = Number(argv[++i]);
    } else if (arg === '--ga-layers') {
      args.gaLayers = Number(argv[++i]);
    } else if (arg === '--ga-max-position') {
      args.gaMaxPosition = Number(argv[++i]);
    } else if (arg === '--ga-eval-windows') {
      args.gaEvalWindows = Number(argv[++i]);
    } else if (arg === '--ga-seed') {
      args.gaSeed = Number(argv[++i]);
    } else if (arg === '--ga-save-top-n') {
      args.gaSaveTopN = Number(argv[++i]);
    } else if (arg === '--ga-save-every') {
      args.gaSaveEvery = Number(argv[++i]);
    } else if (arg === '--ga-checkpoint-every') {
      args.gaCheckpointEvery = Number(argv[++i]);
    } else if (arg === '--ga-behavior-every') {
      args.gaBehaviorEvery = Number(argv[++i]);
    } else if (arg === '--w-pnl') {
      args.wPnl = Number(argv[++i]);
    } else if (arg === '--w-sortino') {
      args.wSortino = Number(argv[++i]);
    } else if (arg === '--w-mdd') {
      args.wMdd = Number(argv[++i]);
    } else if (arg === '--drawdown-penalty') {
      args.drawdownPenalty = Number(argv[++i]);
    } else if (arg === '--drawdown-penalty-growth') {
      args.drawdownPenaltyGrowth = Number(argv[++i]);
    } else if (arg === '--session-close-penalty') {
      args.sessionClosePenalty = Number(argv[++i]);
    } else if (arg === '--max-hold-bars-positive') {
      args.maxHoldBarsPositive = Number(argv[++i]);
    } else if (arg === '--max-hold-bars-drawdown') {
      args.maxHoldBarsDrawdown = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty') {
      args.holdDurationPenalty = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty-growth') {
      args.holdDurationPenaltyGrowth = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty-positive-scale') {
      args.holdDurationPenaltyPositiveScale = Number(argv[++i]);
    } else if (arg === '--hold-duration-penalty-negative-scale') {
      args.holdDurationPenaltyNegativeScale = Number(argv[++i]);
    } else if (arg === '--flat-hold-penalty') {
      args.flatHoldPenalty = Number(argv[++i]);
    } else if (arg === '--flat-hold-penalty-growth') {
      args.flatHoldPenaltyGrowth = Number(argv[++i]);
    } else if (arg === '--max-flat-hold-bars') {
      args.maxFlatHoldBars = Number(argv[++i]);
    } else if (arg === '--selection-train-weight') {
      args.selectionTrainWeight = Number(argv[++i]);
    } else if (arg === '--selection-eval-weight') {
      args.selectionEvalWeight = Number(argv[++i]);
    } else if (arg === '--selection-gap-penalty') {
      args.selectionGapPenalty = Number(argv[++i]);
    } else if (arg === '--selection-use-eval') {
      args.selectionUseEval = true;
    } else if (arg === '--rl-epochs') {
      args.rlEpochs = Number(argv[++i]);
    } else if (arg === '--rl-train-windows') {
      args.rlTrainWindows = Number(argv[++i]);
    } else if (arg === '--rl-ppo-epochs') {
      args.rlPpoEpochs = Number(argv[++i]);
    } else if (arg === '--rl-hidden') {
      args.rlHidden = Number(argv[++i]);
    } else if (arg === '--rl-layers') {
      args.rlLayers = Number(argv[++i]);
    } else if (arg === '--rl-max-position') {
      args.rlMaxPosition = Number(argv[++i]);
    } else if (arg === '--rl-eval-windows') {
      args.rlEvalWindows = Number(argv[++i]);
    } else if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
};

const repoRoot = process.cwd();
const pipelineStamp = new Date().toISOString().replace(/[:.]/g, '-');
const reportDir = path.join(repoRoot, 'runs', 'pipeline_reports', `es_candle_${pipelineStamp}`);
const runViaApiScript = path.join(repoRoot, 'scripts', 'run_train_via_api.mjs');
const gaOutdir = path.join('runs_ga', `es_candle_fold1_${pipelineStamp}`);
const rlOutdir = path.join('runs_rl', `es_candle_fold2_${pipelineStamp}`);

const esPreset = {
  'margin-mode': 'per-contract',
  'margin-per-contract': 500,
  'contract-multiplier': 50,
  'auto-close-minutes-before-close': 5,
};

const week = (folder) => path.join('data', 'train', folder, 'ES_F.parquet');

const gaFold = {
  name: 'ga_fold_1',
  engine: 'ga',
  params: {
    backend: 'candle',
    device: 'cpu',
    outdir: gaOutdir,
    'train-parquet': week('12-29-2025'),
    'val-parquet': week('01-04-2026'),
    'test-parquet': week('01-12-2026'),
    windowed: true,
    window: 700,
    step: 128,
    generations: 4,
    'pop-size': 8,
    workers: 2,
    'batch-candidates': 0,
    'elite-frac': 0.33,
    'parent-pool-frac': 0.33,
    'immigrant-frac': 0,
    'mutation-sigma': 0.05,
    'init-sigma': 0.5,
    hidden: 64,
    layers: 2,
    'eval-windows': 4,
    'drawdown-penalty': 0,
    'drawdown-penalty-growth': 0,
    'session-close-penalty': 0,
    'max-hold-bars-positive': 15,
    'max-hold-bars-drawdown': 15,
    'hold-duration-penalty': 1.0,
    'hold-duration-penalty-growth': 0.05,
    'hold-duration-penalty-positive-scale': 0.5,
    'hold-duration-penalty-negative-scale': 1.5,
    'flat-hold-penalty': 2.2,
    'flat-hold-penalty-growth': 0.05,
    'max-flat-hold-bars': 100,
    'save-top-n': 3,
    'save-every': 1,
    'checkpoint-every': 1,
    'behavior-every': 1,
    'selection-train-weight': 0.3,
    'selection-eval-weight': 0.7,
    'selection-gap-penalty': 0.2,
    'w-pnl': 1.0,
    'w-sortino': 1.0,
    'w-mdd': 0.5,
    'initial-balance': 10000,
    'max-position': 1,
    ...esPreset,
  },
};

const rlFold = {
  name: 'rl_fold_2',
  engine: 'rl',
  params: {
    algorithm: 'ppo',
    backend: 'candle',
    device: 'cpu',
    outdir: rlOutdir,
    'train-parquet': week('01-04-2026'),
    'val-parquet': week('01-12-2026'),
    'test-parquet': week('01-20-2026'),
    windowed: true,
    window: 700,
    step: 128,
    epochs: 6,
    'train-windows': 12,
    'ppo-epochs': 4,
    'eval-windows': 4,
    lr: 0.0003,
    gamma: 0.99,
    lam: 0.95,
    clip: 0.2,
    'vf-coef': 0.5,
    'ent-coef': 0.01,
    hidden: 64,
    layers: 2,
    'log-interval': 1,
    'checkpoint-every': 1,
    'w-pnl': 1.0,
    'w-sortino': 1.0,
    'w-mdd': 0.5,
    'initial-balance': 10000,
    'max-position': 1,
    ...esPreset,
  },
};

const writeJson = (filePath, value) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
};

const runStage = ({ stage, host, profile }) => {
  const paramsPath = path.join(reportDir, `${stage.name}.params.json`);
  const summaryPath = path.join(reportDir, `${stage.name}.summary.json`);
  writeJson(paramsPath, stage.params);

  const child = spawnSync(
    process.execPath,
    [
      runViaApiScript,
      '--engine',
      stage.engine,
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
    throw new Error(`${stage.name} failed with exit code ${child.status ?? 1}`);
  }

  return JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
};

const main = () => {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage);
    return;
  }

  gaFold.params.window = args.window;
  gaFold.params.step = args.step;
  gaFold.params.generations = args.gaGenerations;
  gaFold.params['pop-size'] = args.gaPopSize;
  gaFold.params.workers = args.gaWorkers;
  gaFold.params['elite-frac'] = args.gaEliteFrac;
  gaFold.params['parent-pool-frac'] = args.gaParentPoolFrac;
  gaFold.params['immigrant-frac'] = args.gaImmigrantFrac;
  gaFold.params['mutation-sigma'] = args.gaMutationSigma;
  gaFold.params['init-sigma'] = args.gaInitSigma;
  gaFold.params.hidden = args.gaHidden;
  gaFold.params.layers = args.gaLayers;
  gaFold.params['max-position'] = args.gaMaxPosition;
  gaFold.params['eval-windows'] = args.gaEvalWindows;
  gaFold.params['save-top-n'] = args.gaSaveTopN;
  gaFold.params['save-every'] = args.gaSaveEvery;
  gaFold.params['checkpoint-every'] = args.gaCheckpointEvery;
  gaFold.params['behavior-every'] = args.gaBehaviorEvery;
  if (args.gaSeed !== null && Number.isFinite(args.gaSeed)) {
    gaFold.params.seed = args.gaSeed;
  } else {
    delete gaFold.params.seed;
  }
  gaFold.params['w-pnl'] = args.wPnl;
  gaFold.params['w-sortino'] = args.wSortino;
  gaFold.params['w-mdd'] = args.wMdd;
  gaFold.params['drawdown-penalty'] = args.drawdownPenalty;
  gaFold.params['drawdown-penalty-growth'] = args.drawdownPenaltyGrowth;
  gaFold.params['session-close-penalty'] = args.sessionClosePenalty;
  gaFold.params['max-hold-bars-positive'] = args.maxHoldBarsPositive;
  gaFold.params['max-hold-bars-drawdown'] = args.maxHoldBarsDrawdown;
  gaFold.params['hold-duration-penalty'] = args.holdDurationPenalty;
  gaFold.params['hold-duration-penalty-growth'] = args.holdDurationPenaltyGrowth;
  gaFold.params['hold-duration-penalty-positive-scale'] = args.holdDurationPenaltyPositiveScale;
  gaFold.params['hold-duration-penalty-negative-scale'] = args.holdDurationPenaltyNegativeScale;
  gaFold.params['flat-hold-penalty'] = args.flatHoldPenalty;
  gaFold.params['flat-hold-penalty-growth'] = args.flatHoldPenaltyGrowth;
  gaFold.params['max-flat-hold-bars'] = args.maxFlatHoldBars;
  gaFold.params['selection-train-weight'] = args.selectionTrainWeight;
  gaFold.params['selection-eval-weight'] = args.selectionEvalWeight;
  gaFold.params['selection-gap-penalty'] = args.selectionGapPenalty;
  if (args.selectionUseEval) {
    gaFold.params['selection-use-eval'] = true;
  } else {
    delete gaFold.params['selection-use-eval'];
  }
  rlFold.params.window = args.window;
  rlFold.params.step = args.step;
  rlFold.params.epochs = args.rlEpochs;
  rlFold.params['train-windows'] = args.rlTrainWindows;
  rlFold.params['ppo-epochs'] = args.rlPpoEpochs;
  rlFold.params.hidden = args.rlHidden;
  rlFold.params.layers = args.rlLayers;
  rlFold.params['max-position'] = args.rlMaxPosition;
  rlFold.params['eval-windows'] = args.rlEvalWindows;
  rlFold.params['w-pnl'] = args.wPnl;
  rlFold.params['w-sortino'] = args.wSortino;
  rlFold.params['w-mdd'] = args.wMdd;

  const stages = [];
  if (args.mode === 'ga') {
    stages.push(gaFold);
  } else if (args.mode === 'rl') {
    stages.push(rlFold);
  } else {
    stages.push(gaFold, rlFold);
  }

  fs.mkdirSync(reportDir, { recursive: true });
  const results = stages.map((stage) => runStage({ stage, host: args.host, profile: args.profile }));

  const pipelineSummary = {
    generatedAt: new Date().toISOString(),
    host: args.host,
    profile: args.profile,
    step: args.step,
    window: args.window,
    mode: args.mode,
    reportDir,
    results,
  };

  writeJson(path.join(reportDir, 'pipeline_summary.json'), pipelineSummary);
  console.log('\n=== Pipeline Summary ===');
  console.log(JSON.stringify(pipelineSummary, null, 2));
};

main();

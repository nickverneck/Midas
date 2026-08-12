# GC EMA 10/30: new causal normal/inverted gates

Date: 2026-08-11  
Status: replay research only; no live configuration change.

## Protocol

The exact replay used the existing GC EMA 10/30 baseline: standard one-minute
bars from exact-contract raw ticks, closed-bar signal evaluation, next-bar-open
fills, guarded execution, 45-minute session blockout, no protection, and the
existing fee schedule. The seven frozen weeks are:

- Development: May 11, June 8, June 22, July 27, and August 3.
- Holdout: June 15 and July 14.

The holdouts were scored after the gate families and thresholds were screened.
They were not used to choose the candidate.

## Gates tested

1. `CHOP30 > threshold`: invert when the Choppiness Index of the preceding 30
   completed bars is above the threshold; otherwise stay normal.
2. `RET60 < threshold`: compute
   `raw_cross_direction * (close[t-1] - close[t-61]) / ATR[t-1]`; invert when
   that prior move is below the threshold, otherwise stay normal.
3. `RET60 > threshold`: the opposite-tail control, inverting after a strongly
   aligned prior move.

All features exclude the decision bar. Missing values are neutral. The gate
starts normal and changes only at the causal strategy decision points.

## Provenance

`RET60 < 2 ATR` is not a Reddit-reported rule. The Reddit memo supplied broad
experiment ideas—ATR/volatility context, ADX/DI, higher-timeframe persistence,
session/VWAP, and neutral/abstain states. The 60-bar aligned-return feature,
the inversion mapping, and the 2-ATR threshold were local hypotheses selected
from this repository's replay results.

## Exact replay aggregate

Net P&L is USD. `dev+` and `hold+` count positive weeks; `max DD` is the
largest weekly drawdown in the seven-window run.

| Gate | Dev net | Dev+ | Holdout net | Hold+ | All net | All+ | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed normal control | -18,133.2 | 2/5 | -3,297.3 | 1/2 | -21,430.5 | 3/7 | 23,676.4 |
| Fixed inverted control | 5,286.8 | 3/5 | 922.7 | 1/2 | 6,209.5 | 4/7 | 29,928.2 |
| CHOP30 > 45 | 33,969.2 | 4/5 | 5,656.7 | 2/2 | 39,625.9 | 6/7 | 24,913.2 |
| CHOP30 > 55 | 30,547.4 | 4/5 | 1,361.9 | 2/2 | 31,909.3 | 6/7 | 23,828.0 |
| CHOP30 > 65 | -31,962.8 | 2/5 | -2,704.9 | 1/2 | -34,667.7 | 3/7 | 29,464.0 |
| RET60 < 1 ATR | 54,115.6 | 3/5 | 28,618.1 | 2/2 | 82,733.7 | 5/7 | 11,003.8 |
| RET60 < 2 ATR | 55,827.6 | 4/5 | 15,888.5 | 2/2 | 71,716.1 | 6/7 | 14,793.4 |
| RET60 < 3 ATR | 56,691.2 | 4/5 | 17,051.3 | 2/2 | 73,742.5 | 6/7 | 30,240.2 |

The `RET60 > 1/2/3 ATR` controls were rejected: development net was
`-62,150.6 / -65,598.6 / -67,975.0`, with zero or one positive development
week. This is evidence against blindly inverting after a large move aligned
with the raw cross.

The conservative selection is `RET60 < 2 ATR`: it keeps the 6/7 transfer rate
of the other strong candidates while avoiding the weaker 3/5 development
consistency of the 1-ATR threshold and the much larger worst-week loss and
drawdown of the 3-ATR threshold.

Selected-candidate weekly net P&L:

| Week | RET60 < 2 ATR |
|---|---:|
| May 11 | +11,492.6 |
| June 8 | +18,365.8 |
| June 15 holdout | +13,978.5 |
| June 22 | +21,503.4 |
| July 14 holdout | +1,910.0 |
| July 27 | +10,312.2 |
| August 3 | -5,846.4 |

## Current conclusion

The best causal rule found in this pass is:

> At a raw EMA 10/30 cross, invert only when the preceding 60-bar move in the
> cross direction is less than 2 ATR; otherwise use normal orientation.

This is not a daily hindsight picker and does not select a whole week in
advance. It is an event-level gate that can change orientation as new crosses
arrive. It remains research-only: the sample is seven overlapping-window
experiments, the threshold was selected from a small frozen family, and the
August 3 miss shows that a fixed normal trend week can still be damaged.
The next safety test should add confirmation/dwell or a Neutral state and
validate those settings on new weeks before any deployment discussion.

## Reproducibility artifacts

- Gate implementation: `src/strategies/adaptive_gate.rs`.
- Choppiness spec generator: `.run/replay-sweeps/gc-regime-gate-research-20260810/make_choppiness_specs.py`.
- Directional-return spec generator: `.run/replay-sweeps/gc-regime-gate-research-20260810/make_directional_return_specs.py`.
- Exact Choppiness summaries: `.run/replay-sweeps/gc-regime-gate-research-20260810/results/adaptive_*_10_30_choppiness30_grid/sweep-summary.csv`.
- Exact return summaries: `.run/replay-sweeps/gc-regime-gate-research-20260810/results/adaptive_*_10_30_directional_return60_*_grid/sweep-summary.csv`.

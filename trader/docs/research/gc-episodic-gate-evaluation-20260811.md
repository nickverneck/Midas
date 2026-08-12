# GC EMA 10/30 episodic gate evaluation

Date: 2026-08-11  
Status: replay research only; no live configuration change.

## Result in plain English

The evaluation is now episodic:

1. The latest completed week, August 3–7, is evaluated day by day.
2. July 27–31 is retained as the prior contract-transition episode.
3. July 20–24 and older weekly episodes are evaluated backward.
4. A candidate is judged by both weekly persistence and daily behavior, rather
   than by one profitable aggregate.

There is no single gate that maximizes both the latest daily P&L and the
backward weekly P&L. The best current compromise is the RVOL/DI gate with:

> invert when relative volume is below approximately `0.95` and DI imbalance
> is below `0` (XOR, minimum two votes).

This is a resilience candidate, not a claim of a deployable edge. It was
positive on 10 of 11 weekly episodes and produced `+$8,180` over the five
latest daily slices, but it trailed fixed-normal on the current week
(`+$8,180` versus `+$14,253`). Its value is strongest in episodes where the
inverted orientation is favored; it gives back performance in strong normal
trend episodes.

## Replay protocol

All new runs used the causal 1-minute standard-candle protocol: closed-bar
signals, raw-bar-open fills, guarded execution, 45-minute blockout, the same
fee schedule, and no protection orders. Daily slices carried prior bars as
indicator warmup but restarted trading flat, so daily P&L is an episode
diagnostic and must not be added to the full-week aggregate.

| Episode | Contract | Treatment |
|---|---|---|
| Feb 16–20 | GCZ6 | backward weekly |
| Apr 20–24 | GCZ6 | backward weekly |
| May 11–15 | GCQ6 | backward weekly; isolated Q6 cache |
| May 25–29 | GCZ6 | backward weekly |
| Jun 8–12 | GCQ6 | backward weekly; isolated Q6 cache |
| Jun 15–19 | GCZ6 | backward weekly; contract-era change flagged |
| Jun 22–26 | GCQ6 | backward weekly; isolated Q6 cache |
| Jul 14–17 | GCZ6 | backward weekly |
| Jul 20–24 | GCZ6 | one-week-before episode |
| Jul 27–31 | GCZ6 | prior week; contract-transition context |
| Aug 3–7 | GCZ6 | latest week; also evaluated as five daily episodes |

The full-week August 3–7 result and the five daily results describe the same
market week at different resolutions; they are intentionally not combined in
one P&L total.

## Latest week, day by day

The two controls show that orientation changed during the week:

| Day | Normal | Inverted | Better fixed orientation |
|---|---:|---:|---|
| Aug 3 | -$208 | -$288 | normal |
| Aug 4 | -$900 | +$380 | inverted |
| Aug 5 | +$16,688 | -$17,072 | normal |
| Aug 6 | +$3,942 | -$4,338 | normal |
| Aug 7 | -$5,268 | +$4,772 | inverted |
| **Total** | **+$14,253** | **-$16,546** | hindsight daily switch: **+$25,573** |

The hindsight switch is an upper-bound diagnostic, not a live policy. It
shows why selecting one orientation for the whole week is insufficient.

## RVOL interpolation sweep

DI was fixed at `< 0`; only the RVOL threshold was interpolated between the
previous `0.8` and `1.0` tests. All 16 episodes and all 62 interpolation/
endpoint runs completed successfully.

### Same 11-episode weekly sample

| Gate | Positive weeks | Sum net P&L | Median week | Worst week |
|---|---:|---:|---:|---:|
| RVOL `< 0.80` | 8/11 | +$37,081 | +$4,066 | -$17,361 |
| RVOL `< 0.85` | 7/11 | +$35,104 | +$2,338 | -$17,488 |
| RVOL `< 0.90` | 9/11 | +$46,532 | +$2,931 | -$20,196 |
| RVOL `< 0.95` | **10/11** | **+$56,768** | **+$4,066** | **-$13,084** |
| RVOL `< 1.00` | 9/11 | **+$73,643** | **+$5,010** | **-$13,084** |

The `1.00` setting has the largest sum, while `0.95` has the better positive-
episode rate. That is why `0.95` is the conservative frozen candidate for the
next prospective test; the choice should be made before looking at that test's
P&L.

### Latest five daily sample

| Gate | Positive days | Sum net P&L | Worst day |
|---|---:|---:|---:|
| RVOL `< 0.80` | 4/5 | +$14,003 | -$6,947 |
| RVOL `< 0.85` | **4/5** | **+$16,631** | **-$5,987** |
| RVOL `< 0.90` | 4/5 | +$11,360 | -$5,987 |
| RVOL `< 0.95` | 4/5 | +$8,180 | -$6,667 |
| RVOL `< 1.00` | 4/5 | +$8,634 | -$6,667 |
| RVOL `< 1.20` | 3/5 | +$2,779 | -$7,127 |

The daily result favors the lower/middle band, but none of the thresholds
avoids the Aug 5 loss. This is evidence that the gate is reducing sensitivity,
not that it has solved orientation selection.

## Other gate families

The weekly and daily rankings disagree, which is exactly why the episodes must
be kept separate:

| Gate | Weekly evidence | Latest daily evidence |
|---|---|---|
| RET60 `< 1 ATR` | +$88,302 across 8 weeks; 6/8 positive; worst about -$965 | +$1,953 across 5 days; only 2/5 positive |
| RET60 `< 2 ATR` | +$66,760 across 8 weeks; 6/8 positive; worst about -$5,846 | not stable day to day |
| CHOP30 `> 55` | +$39,570 across 8 weeks; 7/8 positive | +$7,259; 4/5 positive |
| RVOL `< 0.95` + DI `< 0` | +$56,768 across 11 weeks; 10/11 positive | +$8,180; 4/5 positive |

The RET60 results are a useful weekly correlation candidate, but the latest
daily holdout weakens the idea that its weekly success can be used as a
day-level selector. `RET60 < 2 ATR` is a repository hypothesis, not a numeric
rule taken from Reddit; the Reddit research supplied broad feature ideas only.

## Regime correlation

For analysis, define the weekly orientation spread as:

`normal-control P&L - inverted-control P&L`

Positive values mean the normal orientation won that episode; negative values
mean the inverted orientation won. This is an offline regime label, not a
live feature.

Against the hindsight baseline `max(normal, inverted)`:

| Gate | Median delta | Positive deltas | Pearson correlation with orientation spread |
|---|---:|---:|---:|
| RVOL `< 0.95` + DI `< 0` | -$2,121 | 5/11 | -0.47 |
| RVOL `< 1.00` + DI `< 0` | +$861 | 6/11 | -0.65 |

The negative correlation is directionally useful: these gates tend to help
more when the inverted regime is favored and give back more when a strong
normal trend dominates. It is not proof of predictive power because the
baseline is an oracle and there are only 11 weekly observations.

## Prospective episodic rule

For the next evaluation cycle:

1. Freeze the candidate family and threshold using only completed episodes.
2. Give recent episodes more weight, but require a minimum positive-episode
   rate and inspect the worst episode separately.
3. Test the frozen candidate on the next week day by day; do not retune inside
   that week from its realized P&L.
4. Append the completed week to the history, re-score the candidates, and only
   then choose the next frozen setting.
5. Keep contract-transition episodes labeled rather than silently mixing them
   with ordinary weeks.

The current frozen research candidate is `RVOL < 0.95`, `DI < 0`, XOR,
minimum two votes, on EMA 10/30. It should remain research-only until it is
tested on a new unseen week under this rule.

## Reproducibility artifacts

- Episodic matrix and primary results:
  `.run/replay-sweeps/gc-episodic-20260811/`
- Interpolation specs:
  `.run/replay-sweeps/gc-episodic-20260811/intermediate_rvol_specs/`
- Endpoint specs:
  `.run/replay-sweeps/gc-episodic-20260811/endpoint_rvol_specs/`
- Earlier gate research: `docs/research/gc-regime-gate-new-gates-20260811.md`
- Reddit feature memo: `docs/research/reddit-algotrading-regime-gates-20260810.md`

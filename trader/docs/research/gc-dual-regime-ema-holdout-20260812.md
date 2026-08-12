# GC two-layer EMA regime holdout

Date: 2026-08-12  
Status: replay research only; no live configuration change.

## Question

Keep EMA 10/30 as the only entry trigger, then use a separate moving-average
pair to decide whether that raw cross is traded normally or inverted. The
orientation decision is evaluated only at a raw 10/30 cross; it is not a
whole-day label.

The first pass used a separate one-minute EMA pair. At each 10/30 cross, the
secondary pair was read from the prior completed one-minute bar. If its
direction agreed with the raw cross, the cross stayed normal; if it disagreed,
the cross was inverted. This is causal and cross-by-cross, but it is still a
research implementation of the two-layer idea rather than a new live strategy
kind.

## Replay protocol

- one-minute standard bars, full source sessions;
- exact GCZ6 and GCQ6 contract data;
- closed-bar signal evaluation and raw-bar-open fills;
- one contract, current GC fee schedule, no TP/SL/trailing protection;
- daily episodes with execution flat until the evaluation day;
- seven calendar days of indicator warmup before every daily episode.

The August 3–7 GCZ6 week is the reference window. The June 22–26 GCQ6 week
is a separate older-contract holdout, not a future chronological holdout. The
GCQ6 cache was rebuilt from June 15 through June 26 so all five daily episodes
have real prior bars rather than starting their indicators at the first test
bar.

## Results

Net P&L is USD. The five daily values are summed only for the episode total;
the positive-day count and worst daily drawdown are shown to expose whether a
weekly total depends on one day.

| Orientation selector | GCZ6 Aug 3–7 | Positive days | Worst DD | GCQ6 Jun 22–26 | Positive days | Worst DD |
|---|---:|---:|---:|---:|---:|---:|
| Fixed normal 10/30 | +$17,139.60 | 3/5 | $8,117.80 | -$784.40 | 3/5 | $16,165.60 |
| Frozen RVOL < 0.90 + DI XOR | +$11,719.00 | 4/5 | $13,935.80 | +$9,370.80 | 4/5 | $5,285.50 |
| 60-minute context, spread cap 0.50 ATR | +$18,731.00 | 3/5 | $8,117.80 | +$11,373.80 | 2/5 | $5,075.50 |
| Secondary EMA 2/6 | +$18,859.40 | 3/5 | $6,725.40 | +$4,834.00 | 3/5 | $12,885.00 |
| Secondary EMA 3/8 | +$18,311.80 | 3/5 | $6,725.40 | -$401.20 | 3/5 | $12,885.00 |
| Secondary EMA 5/15 | +$19,104.20 | 3/5 | $7,393.00 | +$5,006.40 | 3/5 | $13,237.40 |
| Secondary EMA 8/21 | +$19,584.20 | 3/5 | $7,393.00 | +$1,711.20 | 3/5 | $13,237.40 |
| Secondary EMA 15/30 | +$22,848.40 | 4/5 | $6,725.40 | -$839.80 | 3/5 | $13,069.80 |
| Secondary EMA 30/60 | +$6,979.60 | 2/5 | $8,976.10 | +$974.20 | 3/5 | $11,237.50 |
| Secondary EMA 60/120 | +$4,463.40 | 2/5 | $7,217.50 | +$9,912.40 | 4/5 | $13,925.10 |
| Secondary EMA 120/240 | +$2,517.80 | 2/5 | $8,429.60 | +$5,000.60 | 3/5 | $9,862.70 |

The best two-window sum among the secondary EMA pairs was EMA 5/15 at
$24,110.60, followed closely by EMA 2/6 at $23,693.40. Neither was positive
on more than three days in either window. EMA 60/120 had the best old-contract
day coverage at 4/5, but it was weak on the reference August window.

## HMA follow-up

The native gate now supports `higher_timeframe_average_kind: "hma"` as an
opt-in secondary selector. The trigger remained EMA10/30, the HMA pair was
read from the prior completed one-minute bar, and each day retained the same
seven-calendar-day warmup.

| Secondary HMA pair | GCZ6 Aug total | GCZ6 positive days | GCQ6 Jun total | GCQ6 positive days | Two-window sum |
|---|---:|---:|---:|---:|---:|
| HMA 2/6 | +$18,783.40 | 3/5 | -$11,611.60 | 1/5 | +$7,171.80 |
| HMA 3/8 | +$19,004.80 | 3/5 | -$18,608.20 | 1/5 | +$396.60 |
| HMA 5/15 | +$18,730.40 | 3/5 | +$2,394.00 | 3/5 | +$21,124.40 |
| HMA 8/21 | +$15,876.60 | 3/5 | +$2,202.20 | 3/5 | +$18,078.80 |
| HMA 15/30 | +$3,173.00 | 2/5 | +$5,172.40 | 3/5 | +$8,345.40 |
| HMA 30/60 | +$350.00 | 3/5 | +$3,785.20 | 3/5 | +$4,135.20 |
| HMA 60/120 | -$8,073.60 | 1/5 | -$17,765.40 | 1/5 | -$25,839.00 |
| HMA 120/240 | +$2,285.00 | 3/5 | +$3,919.40 | 3/5 | +$6,204.40 |

HMA 5/15 was the best HMA pair across the two windows, but it remained below
the corresponding EMA 5/15 result of $24,110.60 and stayed positive on only
3/5 days in each window. HMA did not provide a resilient improvement over the
EMA secondary selector; the faster HMA pairs were especially state-sensitive,
with HMA 3/8 losing $18,608.20 on GCQ6 after making $19,004.80 on GCZ6.

The 60-minute spread-cap result is attractive on total dollars but sparse on
the GCQ6 holdout: two of its positive days contain only one or four trades.
That is not enough evidence to prefer it over the four-positive-day RVOL/DI
candidate.

## Interpretation

The two-layer idea is plausible but the first EMA-only version does not yet
survive as a stable selector. Faster pairs tend to behave similarly to the
10/30 cross and leave the June 25–26 losses largely intact. Slower pairs reduce
trades and can catch a favorable orientation on one episode, but their result
depends heavily on the market state; EMA 60/120 is the clearest example.

For now, the frozen RVOL/DI XOR gate is the more resilient cross-by-cross
baseline on the older-contract holdout: positive on 4/5 days with a much
smaller worst daily drawdown than fixed normal. HMA is worth keeping as a
native option for future testing, but these two windows do not justify
choosing HMA over EMA or enabling either live.

None of these numerical thresholds came from Reddit. The Reddit research
motivated testing causal regime/context features and separate normal,
neutral, or inverted branches; the pair lengths and the 0.50 ATR cap are local
replay experiments.

## Artifacts

- GCQ6 full-warmup matrix: `.run/replay-sweeps/gcq6-dual-regime-20260812/`
- GCZ6 reference matrix: `.run/replay-sweeps/gcz6-dual-regime-20260812/`
- GCQ6 cache/config: `.run/replay-gcq6-holdout-cache-20260812/` and
  `.run/gcq6-dual-regime-20260812.toml`
- Generator scripts: `.run/replay-sweeps/make_gcq6_dual_regime_specs_20260812.py`
  and `.run/replay-sweeps/make_gcz6_dual_regime_specs_20260812.py`
- HMA matrices: `.run/replay-sweeps/gcq6-dual-regime-hma-20260812/` and
  `.run/replay-sweeps/gcz6-dual-regime-hma-20260812/`
- HMA generators: `.run/replay-sweeps/make_gcq6_dual_regime_hma_specs_20260812.py`
  and `.run/replay-sweeps/make_gcz6_dual_regime_hma_specs_20260812.py`

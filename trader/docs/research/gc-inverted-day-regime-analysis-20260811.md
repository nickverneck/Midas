# GC inverted-day regime analysis

Date: 2026-08-11  
Status: replay research only; no live configuration change.

## Question

The daily controls create a hindsight label by comparing a whole-day normal
lane with a whole-day inverted lane. This analysis checks whether the two
inverted labels share a market regime, and checks what the adaptive gate did at
each EMA 10/30 crossover.

The diagnostic replay used the frozen research candidate:

`RVOL < 0.95` and `DI imbalance < 0`, `XOR`, minimum two available features,
one confirmation bar. With exactly two features, XOR means inversion occurs
when exactly one of the two feature votes is true; it does not mean both
conditions must be true.

## Whole-day labels and market state

| Day | Normal | Inverted | Label | Day net points | Range | Day efficiency | Mean ER60 | Mean ADX14 | Median CHOP30 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Aug 3 | -$208 | -$288 | normal, near tie | -23.1 | 60.3 | 0.021 | 0.117 | 25.1 | 48.2 |
| Aug 4 | -$900 | +$380 | **inverted** | +21.4 | 65.6 | 0.019 | 0.133 | 27.4 | 45.2 |
| Aug 5 | +$16,688 | -$17,072 | normal | +180.9 | 201.7 | 0.128 | 0.188 | 30.7 | 48.0 |
| Aug 6 | +$3,942 | -$4,338 | normal | -32.1 | 82.5 | 0.020 | 0.125 | 26.5 | 46.4 |
| Aug 7 | -$5,268 | +$4,772 | **inverted** | +94.7 | 144.3 | 0.060 | 0.126 | 25.2 | 47.8 |

Interpretation:

- The inverted days were both net up, but they were not clean trends. Their
  directional efficiency was low/moderate and their range was large relative
  to the net move.
- Aug 5 was also up, but it had the clearest directional behavior: much larger
  net movement, higher ER60, higher ADX, and it strongly favored normal.
- Aug 3 and Aug 6 were down and similarly choppy, but favored normal. So
  direction alone, and chop alone, do not identify an inverted day.
- Both inverted days had positive overnight/Europe/NY-AM session progression
  followed by weaker NY-PM behavior. Aug 5 had the same broad direction but a
  much stronger, more persistent move. This suggests the useful distinction is
  closer to “up impulse with unstable/reversing follow-through” versus “clean
  trend,” not simply uptrend versus downtrend.

The sample is only two inverted days, so this is a hypothesis rather than a
classifier.

## What the gate actually did

The diagnostics recorded 1,259 or fewer one-minute evaluation rows per day,
but only rows with a raw EMA cross are adaptive-gate decisions. Non-crossing
rows are emitted for observability and show the no-cross wrapper; they should
not be counted as orientation decisions.

| Day | Raw crosses | Inverted decisions | Normal decisions | Adaptive candidate net |
|---|---:|---:|---:|---:|
| Aug 3 | 42 | 25 | 17 | +$6,153 |
| Aug 4 | 47 | 27 | 20 | +$2,664 |
| Aug 5 | 33 | 17 | 16 | -$6,667 |
| Aug 6 | 33 | 19 | 14 | +$3,547 |
| Aug 7 | 41 | 18 | 23 | +$2,484 |

The count is not a day selector: Aug 3 and Aug 6 had more inverted decisions
than normal decisions while their fixed whole-day labels were normal. The
orientation is mechanical at each cross. For example:

- Aug 4 inverted decisions averaged RVOL `0.827x`, while normal decisions
  averaged `2.112x`; this day mostly triggered the low-volume/positive-DI XOR
  branch.
- Aug 7 inverted decisions averaged DI imbalance `-0.066`, while normal
  decisions averaged `+0.065`; this day mostly triggered the negative-DI/high-
  volume branch.

Within the adaptive candidate, entry trades grouped by the orientation active
at their triggering cross were:

| Day | Normal-entry trades / net | Inverted-entry trades / net |
|---|---:|---:|
| Aug 3 | 10 / +$3,558 | 17 / +$2,595 |
| Aug 4 | 7 / -$623 | 15 / +$3,287 |
| Aug 5 | 6 / +$3,163 | 8 / -$9,830 |
| Aug 6 | 5 / +$2,789 | 10 / +$758 |
| Aug 7 | 14 / +$4,743 | 8 / -$2,260 |

This reinforces that a per-cross adaptive gate and a whole-day inverted lane
are different experiments. On Aug 7, the whole-day inverted control won, but
the adaptive candidate's normal-selected entries contributed more than its
inverted-selected entries.

## Granularity of the current implementation

The current adaptive layer is not a persistent whole-day switch and it is not
evaluated on every bar. In the guarded EMA path, the base EMA evaluation is
checked first; when there is no raw cross, the adaptive computation is skipped.
At a raw buy/sell cross, the causal feature prefix is reconstructed, the
orientation is confirmed, and that orientation is XORed into the effective EMA
signal. If the resulting target already matches the position, the row can be
`no_target`; therefore a gate decision does not necessarily create a trade.

So the precise answer is: the gate is cross-by-cross / signal-opportunity
based. The fixed normal/inverted control runs were whole-day lanes used only to
create the hindsight labels.

## Better next selector

Keep the hindsight daily label for research, but test a separate causal
day-level selector. At a fixed checkpoint, such as the NY open or the first
120–240 evaluation bars, calculate only information available up to that
point:

1. Signed overnight/Europe return normalized by ATR.
2. ER60 and CHOP30 to distinguish persistence from back-and-forth travel.
3. ADX level/slope and DI imbalance.
4. EMA spread/slope and whether the raw cross agrees with the prior directional
   regime.
5. Session return reversal after the checkpoint.

The first rule worth testing is a simple “unstable up impulse” hypothesis:
choose inverted only when the early move is directionally positive but its
efficiency is low/moderate and trend strength is not decisively high; otherwise
choose normal. Do not freeze numeric thresholds from these five days. Sweep
the rule on completed historical episodes, then test it unchanged on a new
holdout week.

For the trade-level path, the more direct feature is aligned context:
`raw_cross_direction × DI_imbalance` (or the existing aligned-return/EMA-gap
features). A bullish cross against a bearish prior regime is a candidate for
inversion; a bullish cross aligned with the prior regime should stay normal.
That is semantically cleaner than using unaligned DI with an XOR rule.

## Artifacts

- Daily control and gate sweeps: `.run/replay-sweeps/gc-episodic-20260811/`
- Cross diagnostics: `.run/replay-sweeps/gc-inverted-day-diagnostics-20260811/`
- Primary episodic report: `docs/research/gc-episodic-gate-evaluation-20260811.md`

# GC regime-gate research — 2026-08-10

This is the tracked summary of the GC inversion/regime-gate experiment. The
full exact replay specs, per-run artifacts, CSV, and reactive dashboard are in
the local working artifact directory:

`.run/replay-sweeps/gc-regime-gate-research-20260810/`

Open its `index.html` through a local HTTP server to watch
`gate-results.csv` refresh every five seconds. The working directory is
runtime output and is intentionally ignored by Git; this document preserves
the conclusions and the acceptance rules.

## Question and protocol

The observed “inverted day” may actually be an event-level flow regime: a
normal EMA cross can be useful in one state and the opposite cross in another.
The gate must decide from the completed decision bar and prior state only. A
hindsight daily/crossover winner is a diagnostic label, never an input.

The exact screen used GC EMA 10/30 on one-minute standard OHLC, closed-bar
signals, next-bar-open fills, guarded replay, 45-minute blockout, one contract,
and no TP/SL/trailing protection. Development windows were May 11, June 8,
June 22, July 27, and August 3. June 15 and July 14 were held out; three
additional sparse GCZ6 Databento weeks (February 16, April 20, May 25) were
then run as extra checks.

## Baseline orientation

The five development windows had fixed-orientation results:

| Window | Normal | Inverted | Better |
| --- | ---: | ---: | --- |
| June 22 | -$9,077.0 | +$6,163.0 | inverted |
| July 27 | +$2,492.8 | -$5,047.2 | normal |
| August 3 | +$18,150.4 | -$20,729.6 | normal |
| June 8 | -$14,310.8 | +$11,409.2 | inverted |
| May 11 | -$15,388.6 | +$13,491.4 | inverted |

Choosing the better fixed orientation after seeing each window gives
+$51,706.8. That is a hindsight diagnostic, not a deployable strategy. The
state changes within dates, and every ET session contains both orientations;
there is no stable Thursday, lunch, after-hours, or power-hour switch.

## Gate results

### Relative volume + DI

`relative_volume < 0.90 XOR DI imbalance < 0` produced +$66,174.2 on the five
development windows, positive on all five. It lost -$21,763.5 on the unseen
June 15 window and -$802.4 on July 14. Running the same feature rule with the
fixed base inverted was profitable on June 15, showing that the mapping from
feature to orientation itself changes by regime.

### ADX + directional EMA(240)

The best development rows used `ADX(7) <= 25 XOR` a direction-aware EMA(240)
gap at the current closed cross. Depending on the frozen gap thresholds, the
first five plus June 15 reached approximately +$77.6k–+$96.8k. The same
parameters lost approximately $10.2k–$13.3k on July 14 and lost $7,883.6 on
the additional May 25 holdout. This is a useful explanatory feature, not a
generic gate.

### Symmetric EMA(240) counter-trend context

The closest simple candidate was:

```text
invert when a bullish raw cross is at least 2 ATR below EMA(240),
or a bearish raw cross is at least 2 ATR above EMA(240)
```

Exact seven-window results were:

```text
June22 +8,169.2   July27 +1,693.4   Aug03 +10,345.8
June08 +1,623.4   May11  -3,517.4   June15 +8,663.2   July14 +5,932.2
```

That is +$32,909.8, positive on six of seven, with maximum observed window DD
about $11,393. It remained positive on the three extra sparse weeks, but with
only two trades each in that candidate, so the apparent stability is not
strong evidence. It remains replay-only.

Across all ten rows currently in the dashboard, the frozen EMA240 candidate
sums to +$59,515.7 (9/10 positive) versus +$2,624.5 for constant normal and
-$18,775.5 for constant inverted. This aggregate is descriptive only: the
three extra archive weeks have very few baseline/candidate trades, and the
candidate was selected from an earlier grid.

### Other event-level tests

Standalone ATR, ADX/DI, relative volume, EMA distance/slope, session, weekday,
and time-of-day rules did not separate normal from inverted outcomes robustly.
Previous-day/previous-crossover repeat/flip rules changed sign across weeks.
An offline exploratory rule based on `raw_direction × 15-bar return > 1.2`
was positive on both June 15 and July 14 but lost June 22 in development; it
is not accepted. Stateful shadow normal/inverted selectors likewise did not
produce a positive all-window result without lookahead.

## Interpretation and safeguards

The evidence points to a flow-dependent, event-level state rather than a
calendar gate. Contract roll/liquidity and missing L2/economic-calendar data
are plausible causes, but this dataset cannot attribute a move to a Fed event
or depth imbalance.

`src/strategies/adaptive_gate.rs` is opt-in and disabled by default. It supports
direction-aware EMA-gap votes and an explicit zero lookback that samples the
current closed bar; the default remains one bar behind the cross. Prefix tests
verify that future bars cannot alter a prior snapshot. It reconstructs replay
prefix state and is not yet an incremental live state machine.

## Acceptance decision

No gate has matched the hindsight winners across the unseen windows. Do not
enable RV/DI, ADX/gap, or EMA240 gates in live execution. The next research
implementation should be replay-only and use `Normal / Inverted / Neutral`
states, two isolated shadow experts, lagged scoring, hysteresis, minimum
samples, dwell, explicit missing-data resets, and an audit log. Parameters must
be frozen on training weeks and tested on additional GC contracts/weeks before
trying ES/MES. Abstaining is safer than forcing an orientation when confidence
is low.

Validation completed after the causal source changes:

- `cargo fmt --check`
- `cargo test --features replay strategies::adaptive_gate -- --nocapture` — 12 passed
- `cargo test --features replay strategies::volume_regime -- --nocapture` — 4 passed
- `cargo test --features replay aggregation -- --nocapture` — 10 passed
- `cargo build --features replay`

## Novel detector and timeframe follow-up

The exploratory review in
`.run/replay-sweeps/gc-regime-gate-research-20260810/novel-regime-research.md`
recommends treating efficiency ratio, variance ratio/autocorrelation,
choppiness, and directional entropy as causal confidence/neutral features,
not as a permanent normal-versus-inverted mapping. A high-efficiency,
persistent move should increase confidence in retaining the current
orientation; low-efficiency or anti-persistent conditions should move the
selector toward `Neutral` before they are allowed to propose inversion.

The Aug 03 control is a required safeguard: constant normal was +$18,150.4
with $8,117.8 drawdown, while constant inverted was -$20,729.6 with
$29,928.2 drawdown. The RV/DI gate (+$9,973.8) and EMA240 countertrend gate
(+$10,345.8) were profitable but materially worse than the normal control;
ADX+gap (+$18,514.6) approximately preserved it but is not generic on later
holdouts. A gate is therefore not successful merely because it has positive
P&L: it must preserve a known-normal week while improving genuinely unseen
weeks.

The next comparison should run both EMA 10/30 and 210/240 with no protection
first, then repeat only frozen candidates, across the same chronological
windows in: (1) one-minute OHLC, (2) five-minute bars derived causally from
the one-minute/tick source, (3) 500-volume bars, and (4) 10-range bars. The
five-minute test must include both same bar lengths and lengths scaled to the
same clock horizon; otherwise it confounds aggregation with strategy speed.
For every view record normal/inverted P&L, drawdown, trade count, average bar
duration, cross count, inversion rate, and regret versus each fixed
orientation. Volume/range bars must be checked for source-volume and timestamp
parity. Selection remains week-forward: thresholds are frozen on earlier
windows and tested on a later week/contract.

### First cross-timeframe controls

The 5-minute raw-tick controls were run as two-child normal/inverted sweeps,
with identical fill, fee, reversal, and blockout settings. Same-bar-length
EMA 10/30 did **not** preserve the one-minute orientation: it favored normal
on Aug 03 (+$20,965.8 / -$21,474.2 inverted), Jun 08 (+$10,111.6 /
-$10,508.4), and Jun 22 (+$13,155.4 / -$13,564.6), but favored inverted on
Jun 15 (-$3,619.1 / +$3,240.9) and Jul 14 (-$5,756.0 / +$5,384.0). Thus a
coarser time bar can remove some short-horizon inversion behavior, but it is
not a generic normal-orientation gate.

To separate aggregation from a five-times-slower strategy, 5-minute EMA 2/6
was also tested as the approximate clock-horizon equivalent of one-minute
10/30. It favored normal on Aug 03 (+$16,977.6 / -$19,482.4) and Jun 15
(+$20,509.5 / -$22,710.5), inverted on Jun 08 (-$16,543.2 / +$13,616.8),
and produced losses in both directions on Jun 22 and Jul 14. The analogous
5-minute 42/48 slow pair has only 12--21 trades per window; it is diagnostic
only. These controls show that bar construction and physical horizon are both
state variables, not merely a solution to the inversion issue.

The current Aug 03 exact-server-bar controls remain strongly normal across
representations: volume-500 EMA 10/30 +$5,623.8 versus -$6,256.2 inverted
(51 trades), range-10 EMA 10/30 +$18,230.4 versus -$27,009.6 (708 trades),
and range-10 EMA 210/240 +$13,323.8 versus -$13,956.2 (51 trades). The
volume-500 210/240 comparison has only three trades and is not meaningful.

An attempted raw-tick-derived range-10 replay must be excluded: all five
windows produced repeated virtual-clock ordering errors (`event ... behind
virtual clock`) and zero/near-zero trades despite completing with a nominal
status. Existing server-range controls are usable; raw-derived range results
are not a valid research input until their timestamp/order path is fixed and
parity-tested.

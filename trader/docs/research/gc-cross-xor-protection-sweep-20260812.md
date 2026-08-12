# GC EMA 10/30 cross-by-cross protection sweep — 2026-08-12

## Result

The frozen strategy was tested with protection added:

```text
EMA 10/30
RVOL < 0.95 XOR DI imbalance < 0
one contract, closed-bar signal, raw-bar-open fill
```

The gate and cross-by-cross orientation logic were held fixed. Only native
EMA protection changed. All 836 weekly sweep combinations completed without
replay failures under the conservative bar-protection policy.

The highest P&L remains the no-protection control. The most useful risk-adjusted
shortlist is a trailing stop with:

```text
stop_loss_ticks   = 20
trail_trigger     = 20 ticks
trail_offset      = 10 ticks
take_profit       = disabled
```

This is a protection candidate, not a claim that the gate itself is
deployable.

## Weekly sweep design

Eleven weekly episodes were tested: Feb 16, Apr 20, May 11, May 25, Jun 8,
Jun 15, Jun 22, Jul 14, Jul 20, Jul 27, and Aug 3–7. The first seven were
treated as the development history; Jul 14–31 is the later holdout block; Aug
3–7 is the latest reference week.

Fixed bracket grid:

- take profit: `0, 40, 80, 120` ticks;
- stop loss: `0, 20, 50, 80` ticks.

Trailing grid:

- initial stop loss: `0, 20, 50, 80, 100` ticks;
- trigger: `20, 40, 60, 80` ticks;
- offset: `5, 10, 20` ticks.

The `(TP=0, SL=0)` fixed row is the exact no-protection control. A trailing
configuration with `stop_loss=0` still has the engine’s initial protective
stop based on trigger plus offset before the trailing state activates.

## Weekly comparison

| Protection | Development 7 weeks | Jul 14–31, 3 weeks | Aug 3–7 | All 11 weeks |
|---|---:|---:|---:|---:|
| No protection | `+$44,147.9`, 6/7 | `+$12,197.4`, 2/3 | `+$6,596.2` | **`+$62,941.5`, 9/11** |
| Fixed SL 80 | `+$42,967.2`, 3/7 | `-$902.6`, 1/3 | `+$2,396.2` | `+$44,460.8`, 5/11 |
| Trail SL80 / trigger40 / offset5 | `+$40,378.6`, 7/7 | `-$3,412.6`, 1/3 | `+$1,556.2` | `+$38,522.2`, 9/11 |
| Trail SL20 / trigger20 / offset10 | `+$33,158.6`, 7/7 | `+$6,137.4`, 3/3 | `+$1,626.2` | `+$40,922.2`, 11/11 |
| Trail SL0 / trigger20 / offset10 | `+$35,238.6`, 7/7 | `+$5,057.4`, 2/3 | `+$2,756.2` | `+$43,052.2`, 10/11 |

The development-maximizing trailing row was SL80/trigger40/offset5, but it
lost money in two of the three later July holdout weeks. The SL20/20/10 row
was less profitable in development but positive in every one of the 11 weekly
episodes, with a much smaller worst loss.

## Daily July/August validation

The shortlist was rerun day by day on Jul 14–31 and Aug 3–7, with trading
reset flat for each daily episode. This preserves the earlier cross-by-cross
diagnostic protocol.

| Protection | Jul 14–31 | Aug 3–7 | Combined 19 days | Worst daily P&L | Max daily DD |
|---|---:|---:|---:|---:|---:|
| No protection | `+$13,081.8`, 7/14 | `+$8,180.0`, 4/5 | `+$21,261.8`, 11/19 | `-$6,666.8` | `$14,275.8` |
| Fixed SL80 | `-$248.2`, 5/14 | `+$5,660.0`, 3/5 | `+$5,411.8`, 8/19 | `-$5,765.4` | `$6,436.8` |
| Trail SL80/40/5 | `-$2,758.2`, 6/14 | `+$4,230.0`, 3/5 | `+$1,471.8`, 9/19 | `-$4,145.4` | `$5,218.9` |
| Trail SL20/20/10 | **`+$7,111.8`, 10/14** | `+$1,640.0`, 2/5 | **`+$8,751.8`, 12/19** | `-$1,176.8` | `$1,755.1` |
| Trail SL20/20/5 | `+$6,211.8`, 10/14 | `-$200.0`, 2/5 | `+$6,011.8`, 12/19 | `-$1,273.0` | `$1,705.1` |

The 10-tick offset was better than the adjacent 5-tick offset on daily P&L,
while both had similar low drawdown. Both protection rows necessarily give up
some of the large trend capture that makes the unprotected strategy profitable.

## Interpretation

Protection did not improve the raw total P&L. It changed the distribution:

- no protection captured the large winners, but retained a roughly `$6.7k`
  worst daily loss and a roughly `$14.3k` maximum daily drawdown in the latest
  daily sample;
- the selected SL20/trigger20/offset10 trail reduced the worst daily loss to
  roughly `$1.2k` and maximum daily drawdown to roughly `$1.8k`;
- the same row was positive in all 11 weekly episodes in the weekly sweep;
- the exact same row still lost on Aug 5–6 in the daily split, so protection
  does not solve the cross-orientation problem.

Protection exits also increase fills and fees. The test used the same
conservative intrabar ambiguity rule throughout, so a bar touching both a
stop and target is not treated optimistically.

## Decision

Keep no protection as the P&L control. If the objective is to reduce tail risk,
carry `SL20 / trigger20 / offset10` as the research protection candidate for a
fresh untouched week. Do not select the SL80/40/5 trailing row merely because
it was the best development sum; its later holdout behavior was weaker.

The next prospective test should freeze both the XOR gate and the protection
configuration before the week begins, then report P&L, drawdown, trade count,
fees, protection-exit count, and cross-by-cross orientation counts separately.

## Reproducibility artifacts

- Weekly fixed/trailing sweeps: `.run/replay-sweeps/gc-cross-xor-protection-20260812/`
- Daily shortlist validation: `.run/replay-sweeps/gc-cross-xor-protection-daily-20260812/`
- Frozen gate: `RVOL < 0.95`, `DI imbalance < 0`, `combine=xor`, minimum two votes.

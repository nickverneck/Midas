# GC HTF cross-by-cross replay — corrected warm-up pass

Date: 2026-08-12  
Evaluation window: Aug 3–7, 2026, one day per replay  
Instrument: GCZ6, 1-minute standard bars  
Signal: EMA 10/30, closed-bar crosses  
Protection: disabled (`TP=0`, `SL=0`, trailing disabled) so this isolates direction/gating.

## Warm-up correction

Each daily view included the preceding seven calendar days and was flat until that day's evaluation boundary. The replay worker now sends the warm-up snapshot into the strategy series without executing it. The five runs loaded 6,585–6,887 warm-up bars and processed 1,257–1,260 evaluation bars per day.

## Day-by-day net P&L

| Day | Fixed normal | RVOL+DI XOR control | HTF-60 neutral normal / conflict invert |
|---|---:|---:|---:|
| Aug 3 | $1,585.80 | $8,678.80 | -$402.60 |
| Aug 4 | -$349.00 | $4,701.20 | $2,615.60 |
| Aug 5 | $16,687.80 | -$5,986.80 | $17,213.80 |
| Aug 6 | $4,029.20 | $2,308.40 | -$4,571.00 |
| Aug 7 | -$4,814.20 | $2,017.40 | $6,102.80 |
| **Week** | **$17,139.60** | **$11,719.00** | **$20,958.60** |

The HTF-60 candidate improved the fixed-normal control by $3,819.00 (+22.3%), with 47 trades versus 192 and a maximum daily drawdown of $4,571 versus $8,117.80. It was profitable on 3 of 5 days, so this is promising but not yet resilient evidence.

Strict HTF variants were more selective and are not comparable as fair improvements: HTF-60 skip totaled $23,599.00 on only 5 trades, and HTF-60 invert totaled $23,112.80 on 6 trades.

## Aug 3 / Aug 6 failure investigation

The UTC-labeled Aug 3 session begins at about 20:14 ET on Aug 2. The HTF context is unavailable until 23:35 ET because only completed 60-minute buckets count. At 23:53 ET, a bullish 1-minute cross conflicts with a weak bearish HTF state (spread about -0.107 ATR, slope about -0.008 ATR/bar), so the gate inverts it while already short and suppresses the normal long reversal. This is consistent with a contract-transition / low-liquidity failure, not a clean trend example.

The UTC-labeled Aug 6 session begins at about 20:03 ET on Aug 5. Its first bearish 1-minute cross occurred while the HTF state looked strongly bullish: spread +2.14 ATR and slope +0.30 ATR/bar. The gate inverted the cross long, then kept every later bearish cross inverted while the HTF state lagged the actual downside transition. That long stayed open until 14:33 ET and lost $2,336.20; the full HTF candidate ended the day at -$4,571.00 while fixed-normal made +$4,029.20.

Focused transition tests:

| Guard | Aug 6 net | Aug 3–7 week net |
|---|---:|---:|
| HTF conflict invert, no guard | -$4,571.00 | $20,958.60 |
| Require negative 60-minute aligned return | $1,474.60 | $17,346.60 |
| HTF spread ceiling 0.25 ATR | $4,029.20 | $17,402.80 |
| HTF spread ceiling 0.50 ATR | $4,029.20 | $18,731.00 |

The spread ceiling is the cleanest protection against the Aug 6 failure, but it gives back part of the original gate's weekly gain. The key improvement still needed is a causal transition detector that recognizes a strong lower-timeframe break before the lagging HTF EMA state catches up.

A combined exploratory rule—0.5-ATR HTF spread ceiling plus a 240-minute countertrend price-location check—fixed Aug 6 at +$4,029.20 and totaled $24,101.40 for the week. It also turned Aug 7 from +$6,228.40 to -$2,357.00, so it is not accepted as a final gate; it demonstrates that fixing Aug 6 with static thresholds can simply move the episode failure elsewhere.

The candidate specs and per-day result artifacts are under `.run/replay-sweeps/gc-htf-20260812/`.

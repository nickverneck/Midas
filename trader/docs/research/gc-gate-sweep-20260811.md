# GC causal gate sweep: August 3–7 and Feb 9–13 validation

Date: 2026-08-11  
Instrument: GC, exact cached contract `GCZ6`  
Bars: standard 1-minute source bars; closed-bar signals; raw-bar-open fills; guarded execution; 45-minute blockout; no protection

## Sweep

I ran 40 predeclared causal candidates on the August 3–7 window:

- RET60 below thresholds `0.5, 1, 1.5, 2, 2.5, 3 ATR` on EMA 10/30 and 210/240;
- CHOP30 above thresholds `45, 50, 55, 60, 65` on EMA 10/30 and 210/240;
- directional EMA-gap ANY combinations with bullish-below and bearish-above thresholds `0, 1, 2` on both EMA pairs.

No candidate was selected using Feb data. The August results were compared with
the fixed-normal controls from the same replay protocol:

| Pair / candidate | Aug net P/L | Max DD | Trades | Fixed-normal comparison |
|---|---:|---:|---:|---:|
| EMA 10/30, directional gap (`bullish below 0`, `bearish above 1`) | +$20,027.80 | $4,166.20 | 31 | +$18,150.40 |
| EMA 210/240, CHOP30 > 50 | +$22,734.20 | $2,749.30 | 9 | +$15,387.00 |
| EMA 210/240, RET60 < 1.5 ATR | +$18,605.60 | $4,014.10 | 12 | +$15,387.00 |

The 210/240 CHOP result has only nine trades, so its apparent improvement is
especially fragile.

## Frozen validation on Feb 9–13

The top August candidates were frozen without retuning and run on the separate
Feb 9–13 window:

| Pair / candidate | Feb net P/L | Max DD | Trades | Feb fixed-normal | Feb fixed-inverted |
|---|---:|---:|---:|---:|---:|
| EMA 10/30, directional gap | -$21,420.20 | $36,407.10 | 21 | -$25,902.50 | +$25,437.50 |
| EMA 210/240, CHOP30 > 50 | +$1,193.80 | $3.10 | 1 | +$1,193.80 | -$1,206.20 |
| EMA 210/240, RET60 < 1.5 ATR | +$1,193.80 | $3.10 | 1 | +$1,193.80 | -$1,206.20 |

The directional-gap gate improved EMA 10/30 relative to fixed-normal by
$4,482.30 on Feb, but it remained far below fixed-inverted. The 210/240
validation is too sparse to establish an edge: all three strategies made only
one trade.

## Conclusion

The sweep found August-specific improvements, but no gate demonstrated a
robust normal/inverted selection advantage across both windows. The best
current statement is:

> Gates can improve a fixed-normal result in some windows, but the tested
> candidates do not yet beat the correct fixed orientation out of sample.

These results remain research-only. The exact specs and summaries are under
`.run/replay-sweeps/gc-gate-sweep-20260811/`.

# GC Heikin-Ashi color strategy and fresh holdout

Date: 2026-08-11  
Instrument: GC, exact cached contract `GCZ6`  
Holdout window: 2026-02-09 00:10 UTC through 2026-02-13 20:59:59 UTC  
Bar/replay: 1-minute standard source bars, closed-bar signals, raw-bar-open fills, one contract, no protection orders, current fee schedule

## Native strategy

The delegated implementation adds `NativeStrategyKind::HeikinAshiColor` and the
`HeikinAshiConfig` native strategy. It derives Heikin-Ashi bars recursively from
completed standard OHLC bars, then acts only on color transitions:

- non-green to green: buy;
- non-red to red: sell;
- doji/neutral or an unchanged color: hold;
- `inverted = true`: swap buy and sell transition directions.

The transition edge is intentional: it prevents sending a new order on every
bar while the color remains unchanged. The evaluator suppresses a signal that
would simply repeat the current position.

Implementation: `src/strategies/heikin_ashi.rs`, integrated as a native strategy
through `src/strategy.rs` and the replay/live evaluation paths.

## Reddit provenance

The `RET60 < 2 ATR` rule is **not** a Reddit-reported rule. The Reddit research
memo supplied broad experiment ideas—ATR/volatility context, ADX/DI,
higher-timeframe persistence, session/VWAP filters, and neutral/abstain states.
The 60-bar return feature, its alignment to the EMA cross, the inversion mapping,
and the 2-ATR threshold were local hypotheses selected from this repository's
causal replay results.

The earlier seven-window research sweep selected `RET60 < 2 ATR` because it was
positive on 6/7 windows and had the best conservative development/holdout
tradeoff among the tested return thresholds. That selection was still a local
backtest result, not evidence that Reddit established the rule.

## Fresh holdout results

### Heikin-Ashi color strategy

| Orientation | Gross P/L | Net P/L | Fees | Max drawdown | Trades | Fills |
|---|---:|---:|---:|---:|---:|---:|
| Normal | -$31,240.00 | -$32,786.90 | $1,546.90 | $48,303.20 | 250 | 253 |
| Inverted | +$31,240.00 | +$29,693.10 | $1,546.90 | $12,789.20 | 250 | 253 |

This unseen week favored the inverted Heikin-Ashi orientation. It is one
holdout observation, so it does not establish a deployable orientation selector.

### EMA 10/30 controls and `RET60 < 2 ATR`

These runs use the same fresh window and replay assumptions, but the EMA
baseline has 38 trades while the adaptive gate has 16 trades.

| Rule | Gross P/L | Net P/L | Fees | Max drawdown | Trades | Fills |
|---|---:|---:|---:|---:|---:|---:|
| Fixed normal | -$25,670.00 | -$25,902.50 | $232.50 | $31,431.50 | 38 | 41 |
| Fixed inverted | +$25,670.00 | +$25,437.50 | $232.50 | $14,996.20 | 38 | 41 |
| `RET60 < 2 ATR` gate | -$8,870.00 | -$8,966.10 | $96.10 | $14,996.20 | 16 | 19 |

On this additional holdout, the gate improved on fixed normal but failed to
beat fixed inverted. The result therefore does not validate `RET60 < 2 ATR` as
a general normal/inverted selector; it is a candidate requiring more untouched
windows and preferably an abstain/risk rule.

## Artifacts and tests

Specs and summaries are under:

`.run/replay-sweeps/gc-heikin-ashi-color-20260811/`

The native strategy unit tests pass (5 tests), the adaptive-gate tests pass (16
tests), and the Rust workspace formats cleanly with `cargo fmt --all -- --check`.

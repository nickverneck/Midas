# Research Findings (Dec 16, 2025)

## NinjaTrader commissions (MES micro futures)
- Published plan rates (per side): Free $0.39, Monthly $0.29, Lifetime $0.09; exchange/clearing/NFA fees extra. citeturn0search0  
- All-in per-side cost (includes exchange $0.37, clearing $0.19, NFA) from Feb 28, 2025 PDF: Lifetime ≈ $0.65, Monthly ≈ $0.81, Free ≈ $0.91 → round‑turn ≈ $1.30 / $1.62 / $1.82 respectively. citeturn0search5  
- Technology/routing fees can add $0.05–$0.25 per contract if using CQG/TT/third‑party bridges. citeturn0search1  
- For backtests, model commission as all‑in round‑turn: choose $1.60 baseline (Monthly) and make it configurable.

## Time encoding: cyclic sin/cos vs. adding an LSTM
- Cyclic encoding (sin/cos of hour) preserves adjacency (23:00 near 00:00), is model‑agnostic, cheap, and works well for tree/linear models. citeturn2search0turn2search5  
- LSTM advantages: captures longer, non‑periodic dependencies and regime shifts; multiple 2025 studies show LSTM variants outperform classical baselines on financial series. citeturn3academia12turn3search6  
- Costs/risks of LSTM: heavier training time (hours vs minutes), higher overfit risk on intraday noise, requires sequence windows and GPU; inference latency higher than simple MLP on cyclic features.  
- Recommended path: keep cyclic hour sin/cos as default features; add an optional temporal block (small LSTM/GRU or temporal conv) only if offline experiments show material uplift in Sharpe/IC; gate with a feature flag in Python training.

## Action space granularity for RL trading
- Discrete (flat/long/short) is stable and easy to constrain; limited sizing expressiveness.  
- Continuous sizing enables smoother risk and inventory control but must be clipped to margin/position limits and mapped to tick sizes; can be sensitive to slippage model. citeturn1search2  
- Hybrid approaches (“learn continuously, act discretely”) scope a continuous proposal then snap to discrete price/size grid, improving sample efficiency and stability in order‑execution tasks. citeturn1search0turn1search1  
- Guidance for this project: start with small discrete actions {flat, +1, –1 micro} for robustness; add a hybrid head that outputs desired size (continuous) then discretizes to tick/lot for limit orders once baseline is stable; keep slippage/commission in reward.

## ADX strategy research (Jul 24, 2026)

ADX is best treated as a trend-regime filter, not a standalone entry signal. It measures trend strength without direction. Direction should come from `+DI/-DI` plus a separate price-confirmation rule such as a breakout, moving-average trend filter, market-structure condition, or pullback-continuation trigger.

Calculation summary:
- `+DM` captures bullish directional range expansion when the current high exceeds the prior high more than the prior low exceeds the current low.
- `-DM` captures bearish directional range expansion when the prior low exceeds the current low more than the current high exceeds the prior high.
- True range normalizes directional movement by volatility.
- `+DI = 100 * smoothed(+DM) / smoothed(TR)`.
- `-DI = 100 * smoothed(-DM) / smoothed(TR)`.
- `DX = 100 * abs(+DI - -DI) / (+DI + -DI)`.
- `ADX` is the smoothed `DX`, usually with Wilder smoothing.

Simple `+DI/-DI` crosses are weak as trading signals because they can flip repeatedly in choppy markets and they measure directional range expansion rather than tradable drift. A better systematic representation is a continuous directional-strength score:

```text
di_imbalance = (+DI - -DI) / (+DI + -DI)
signed_trend_score = ADX * di_imbalance
```

This is preferable to `ADX * sign(+DI - -DI)` because it penalizes weak DI separation. A tiny `+DI > -DI` should not produce a full bullish trend reading.

Candidate regime logic:
- Turn trend mode on only after ADX crosses above an upper threshold.
- Turn trend mode off only after ADX falls below a lower threshold.
- Require directional imbalance above a minimum threshold.
- Require DI dominance for `N` completed bars.
- Require ADX slope to be positive over several bars, not just one uptick.
- Avoid new entries when ADX is falling.
- Avoid entries inside congestion boxes, narrow ranges, or VWAP bands.
- Add cooldown after failed breakouts or repeated losses in the same regime.

Candidate long permission:
- Trend mode is active.
- `signed_trend_score` is above the long threshold.
- ADX slope is positive over completed bars.
- Price confirms with a breakout, trend MA, market structure, or pullback continuation.

Candidate short permission:
- Trend mode is active.
- `signed_trend_score` is below the short threshold.
- ADX slope is positive over completed bars.
- Price confirms with downside structure, breakdown, trend MA, or pullback continuation.

Timeframe guidance:
- ADX is generally more reliable on higher timeframes for regime detection.
- `1m` ADX is fragile because spread, microstructure noise, and session artifacts can dominate the signal.
- A practical design is to use `15m` or `1h` ADX for regime and direction, then use `1m` or `5m` data for execution and entry timing.
- Multi-timeframe backtests must only use completed higher-timeframe bars. Using an unfinished `15m` or `1h` ADX bar while trading `1m` data creates lookahead bias.

Visualization guidance:
- Separate bull/bear ADX lines are intuitive for discretionary chart reading: bull ADX when `+DI > -DI`, bear ADX when `-DI > +DI`.
- For sweeps and automated research, prefer the signed continuous score because it is easier to threshold, compare, and validate.

Main failure risks:
- `ADX > 25` is not universal; thresholds are asset-, session-, volatility-, and timeframe-dependent.
- ADX is smoothed and can be late.
- Very high ADX can indicate extension rather than fresh opportunity.
- ADX can rise during violent two-sided volatility, not only clean directional trends.
- Costs, spread, slippage, and latency can erase most `1m` edges.
- Shorts may not mirror longs because of squeezes, borrow/funding constraints, and asymmetric volatility.
- Too many filters can create overfit backtests with low trade count.

Backtest and sweep requirements:
- Test the ADX filter against the same entry system without ADX.
- Include realistic commission, spread, slippage, latency, and adverse-fill stress.
- Sweep thresholds, ADX length, DI imbalance, slope lookback, cooldown, and timeframe, then require broad stable regions instead of one isolated optimum.
- Use walk-forward or anchored out-of-sample validation.
- Report long and short performance separately.
- Break results down by asset, year, session, and volatility regime.
- Stress costs at `2x` to `5x` baseline.
- Compare against simple baselines: breakout alone, moving-average trend alone, ADX filter only, and random entries with the same holding period.

Acceptance conditions:
- Net Sharpe/Sortino remains positive after conservative costs.
- Drawdown is tolerable and not concentrated in one market window.
- Edge persists across nearby ADX lengths and thresholds.
- ADX improves the base directional entry system instead of merely reducing both losses and winners.
- Out-of-sample degradation is modest.

Rejection conditions:
- Edge disappears after fees, spread, and slippage.
- Best parameters are isolated spikes.
- Most profits come from one asset, year, or volatility event.
- Trade count is too low relative to the optimization space.
- ADX removes so many trades that the remaining system has no meaningful net improvement.

## L2 depth and Bookmap-style order-flow signals (Jul 24, 2026)

L2 depth data should be treated as short-horizon confirmation and execution timing, not as a complete strategy by itself. Resting liquidity in a heatmap shows available orders, not guaranteed intent. A large bid or ask wall can be real support/resistance, a price magnet, or liquidity that disappears before price reaches it. The more useful signal is how price, market orders, cancellations, and replenishment behave when price interacts with that liquidity.

Minimum useful data:
- L2 depth snapshots or incremental depth updates.
- Best bid/ask and at least top `5` to `10` book levels.
- Trade prints with timestamp, price, size, and aggressor side if available.
- Event ordering or exchange sequence numbers.
- Stable timestamps precise enough for the intended horizon.

Better data:
- MBO/order-by-order events instead of only aggregated MBP depth.
- Add, cancel, modify, and execution events.
- Queue position estimates.
- Hidden/iceberg or replenishment inference.
- Venue/exchange timestamps rather than only vendor receive timestamps.

Core signals:

1. Book imbalance:

```text
book_imbalance_N =
  (sum_bid_size_N - sum_ask_size_N) /
  (sum_bid_size_N + sum_ask_size_N)
```

Positive imbalance means bid depth dominates the selected levels. Negative imbalance means ask depth dominates. This is useful as context, but it is weak alone because displayed liquidity can be pulled.

2. Microprice:

```text
microprice =
  (ask_price * bid_size + bid_price * ask_size) /
  (bid_size + ask_size)
```

If microprice is above midprice, the top of book leans bullish. If it is below midprice, it leans bearish. This is most useful for very short-term entry timing, exit timing, and limit-order placement.

3. Order flow imbalance:

```text
bullish pressure = bids added + asks removed + buyer-initiated trades
bearish pressure = asks added + bids removed + seller-initiated trades
```

Order flow imbalance is usually more useful than static book depth because it captures whether demand/supply is strengthening or disappearing. Positive OFI suggests demand is adding pressure or supply is being removed. Negative OFI suggests supply is adding pressure or demand is being removed.

4. Absorption:

Bullish absorption:
- Large seller-initiated volume hits the bid.
- Price fails to break lower.
- Bid liquidity replenishes.
- OFI flips positive or sellers stop making progress.
- Buy trigger is the failure/reclaim, not the initial bid wall.

Bearish absorption:
- Large buyer-initiated volume hits the ask.
- Price fails to break higher.
- Ask liquidity replenishes.
- OFI flips negative or buyers stop making progress.
- Sell trigger is the rejection/loss of level, not the initial ask wall.

5. Liquidity pulling and stacking:

Bullish context:
- Bids stack below price.
- Asks pull above price.
- Aggressive buy flow appears.
- Bid refills behind price after upward movement.

Bearish context:
- Asks stack above price.
- Bids pull below price.
- Aggressive sell flow appears.
- Ask refills above price after downward movement.

This signal is fragile and should require persistence. Single-frame pulling/stacking can be spoofing, quote refresh noise, or ordinary market-maker inventory adjustment.

6. Wall rejection vs. wall break:

Rejection long:
- Price trades into a large bid wall.
- Sell flow is absorbed.
- Price reclaims the level.
- Bid remains or replenishes.
- Stop belongs behind the absorbed liquidity.

Breakout long:
- Large ask wall is consumed or pulled.
- Price accepts above the wall.
- Bid liquidity follows behind price.
- Continuation trigger occurs after acceptance, not just the first touch.

Short logic is symmetric around large ask rejection or bid wall break.

7. Sweep and liquidity vacuum:

Bullish continuation:
- Buyer-initiated trades sweep multiple ask levels.
- Ask depth above is thin.
- Bid liquidity refills behind price.
- Price holds above the swept area.

Bearish continuation:
- Seller-initiated trades sweep multiple bid levels.
- Bid depth below is thin.
- Ask liquidity refills above price.
- Price holds below the swept area.

Failed sweeps can become reversal signals. If price sweeps up and cannot hold above the swept liquidity, it may indicate exhaustion rather than continuation.

Candidate buy trigger:
- Higher-timeframe or slower strategy permits longs.
- Spread is normal.
- Top-`N` book imbalance is positive or improving.
- OFI is positive over recent seconds/events.
- Asks are being consumed or pulled.
- Aggressive buy volume confirms.
- Price breaks, reclaims, or holds a local level.

Candidate reversal buy trigger:
- Price flushes into visible bid liquidity.
- Large seller-initiated volume prints.
- Price fails to continue lower.
- Bid replenishes.
- OFI flips positive.
- Buy on reclaim, with stop behind the absorbed level.

Candidate sell triggers mirror the same structure with ask-side absorption, negative OFI, bid pulling, or failed upward sweeps.

Chop and false-signal filters:
- Do not trade when imbalance flips sign repeatedly.
- Avoid wide or unstable spreads.
- Avoid thin books where one order can distort the signal.
- Ignore walls that appear and disappear quickly unless they produce a clear price reaction.
- Avoid signals when expected move is smaller than spread, fees, slippage, and latency cost.
- Require the order-book signal to align with a slower regime filter when possible.
- Require persistence across several events or seconds instead of acting on one snapshot.

Preferred project use:
- Use L2 as a confirmation/execution layer on top of a slower directional strategy such as ADX trend regime, breakout, or moving-average trend.
- Example: ADX permits only longs, then L2 decides whether a pullback has real bid absorption or whether buyers are absent.
- For `1m` OHLCV strategies, L2 can improve entries/fills but should not be retrofitted into old bars without granular replay data.

Backtest and sweep requirements:
- Use event-time replay when possible, not only sampled snapshots.
- Preserve event ordering between trades, depth updates, cancels, and quote changes.
- Use completed slower-timeframe signals if combining L2 with `1m`, `5m`, or higher timeframe bars.
- Include realistic spread, commissions, latency, queue position, partial fills, and adverse selection.
- Test whether L2 improves the same strategy without L2.
- Report performance by session, volatility regime, spread regime, and liquidity regime.
- Stress test with increased latency and worse queue assumptions.

Acceptance conditions:
- Signal survives realistic costs and latency.
- Performance improves versus the same base strategy without L2.
- Edge is not concentrated around a few news events or low-liquidity anomalies.
- Parameters have broad stable regions across depth levels, lookback windows, and imbalance thresholds.
- Long and short behavior is reviewed separately.

Rejection conditions:
- Signal only works with perfect fills or zero latency.
- Static walls alone appear predictive, but the edge disappears after cancellation/spoof filters.
- Best result depends on one depth level or one exact window length.
- Expected move is smaller than transaction costs.
- The backtest uses sampled snapshots that cannot reconstruct event order.

Research references:
- Cont, Kukanov, and Stoikov, "The Price Impact of Order Book Events": https://axi.lims.ac.uk/paper/1011.6402
- Gould and Bonart, "Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book": https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2702117
- Databento, "Microprice and book imbalance": https://databento.com/docs/examples/order-book/microprice

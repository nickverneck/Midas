# Reddit practitioner research: causal regime gates for intraday EMA/HMA futures

**Research date:** 2026-08-10  
**Scope:** r/algotrading first, with r/mltraders and futures/order-flow threads as
secondary sources.  This is a research memo, not an acceptance report and not
investment advice.

## Executive summary

Reddit reports do not establish a robust orientation-flip rule.  The strongest
implementable pattern is more modest: use a small number of causal features to
identify when a trend-following EMA/HMA cross is likely to be low quality, then
abstain, reduce size, or keep the established orientation.  The most repeated
features are:

1. volatility relative to its own recent distribution (ATR or realized-vol
   percentile),
2. trend strength and persistence (ADX/DI, moving-average slope/separation,
   efficiency/choppiness), and
3. futures session context (opening range, VWAP, time of day, and
   session-normalized volume).

HMM/GMM and supervised meta-label models are plausible later experiments, but
the reports emphasize lag, state-label instability, small samples, and leakage
from smoothed labels or post-hoc strategy outcomes.  Order-book imbalance and
OFI require a real timestamped L2/MBO feed and an execution model; the Reddit
examples offer a useful feature definition but no convincing intraday futures
P&L evidence.

The recommended first replay is therefore a two-axis **ATR-percentile ×
ADX/EMA-orientation** matrix with a `Normal / Neutral` action.  A state should
not invert a raw cross merely because it is low volume, low ADX, or an HMM state
has a name that looked good in hindsight.  Treat inversion as a separately
trained hypothesis and require it to beat both constant-normal and
constant-inverted controls on untouched chronological windows.

## Evidence quality and how to read the links

All sources below are self-reports or comments.  Upvotes, asserted live use,
and screenshots are not independently audited.  “Claim” means the author
reported that result; “reproducible detail” means the post gives enough setup
to implement a test, not that the result is verified.  A result is not accepted
as fact without independent replay.

### Practitioner reports

| Topic | Reported setup or result | What is useful / what remains unproven |
| --- | --- | --- |
| Simple ATR allocation | In [“Do you use regime filters?”](https://www.reddit.com/r/algotrading/comments/1se1rof/do_you_use_regime_filters/), an author says a compatibility matrix routes trend/momentum/breakout systems to trend states, mean reversion/VWAP systems to range states, and breakout/momentum systems to high-volatility states.  The detector is described as ATR-based; after 2,018 real trades the author says 93% of P&L came from three agents. | Supports per-strategy compatibility rather than one global on/off switch.  The P&L concentration is a warning about correlated agents and is not evidence that ATR caused the result.  No asset, threshold, fills, or trade log is provided. |
| State filter / abstention | In [the companion regime-filter discussion](https://www.reddit.com/r/algotrading/comments/1s7dvbu/do_you_use_regime_filters/), one practitioner reports a basic state filter almost halved trades while improving net P&L and reducing equity volatility.  Other comments say simple ATR plus moving-average slope is more robust than complex HMMs, while another reports that filters reduced losses but also removed winners. | A useful acceptance framing: measure rejected-trade expectancy and compare against equal-exposure sizing, not just filtered versus unfiltered P&L.  No independently verified numbers. |
| ATR percentile and sizing | In [“How are you guys adapting trend following algos…”](https://www.reddit.com/r/algotrading/comments/1s9wrmb/how-are-you-guys-adapting-trend-following-algos/), a commenter claims 30-day realized volatility versus a one-year rolling percentile held up out of sample for a trend system; below the 25th percentile was described as hostile, with a separate mean-reversion system used there.  Another commenter recommends low-volatility abstention or confidence-based sizing instead of forcing a trend strategy. | Test percentile/relative ATR rather than absolute ATR, and treat the low-vol state as neutral or reduced exposure.  “Held up OOS” is an unverified claim; the thread gives no folds, symbols, or costs. |
| ADX/DI | In [“Backtest results for an ADX trading strategy”](https://www.reddit.com/r/algotrading/comments/1irhrcw/backtest_results_for_an_adx_trading_strategy/), a two-year S&P 500 hourly test initially used ADX > 25, +DI/−DI cross, next-bar entry, and a 1× ATR stop/2R target.  The author says removing the DI cross and using an ADX threshold plus 1.5× ATR/3.5R improved the result; a 200 EMA reduced trades and drawdown. | It is a concrete ADX/ATR/EMA experiment, but the author did not include fees or slippage and did not run a clean OOS test.  Comments explicitly flag parameter selection and the short sample.  Do not copy the 25, 1.5×, or 3.5R values as defaults. |
| Futures EMA + chop/ATR | In [“Is this Trading Strategy Tradable?”](https://www.reddit.com/r/algotrading/comments/1q7j6zh/is-this-trading-strategy-tradable/), an author describes a 5-minute NQ EMA crossover with a choppiness filter, ATR volatility filter, 10-bar/time exits, commissions, and slippage; only 16 months were tested and the author says roughly 60% of profit came from a New York-open swing.  The author also says the system did not work before the 16-month window. | Directly relevant to EMA futures gating and a useful failure-mode report: session concentration and regime dependence can dominate aggregate P&L.  The short, selected window makes the result exploratory only. |
| Opening range and session | In [a five-year ORB report](https://www.reddit.com/r/algotrading/comments/1j9pxsr/backtest_results_for_the_opening-range-breakout/), the claimed rule uses the first 15-minute New York candle, waits for a close outside the range, enters the next candle before noon, and uses a range stop with a 1.5R target.  The author says most profits occurred in the first two hours and reports positive tests on S&P 500 CFD, BTC, and GBP/USD.  Lower-range shorts were later described as mixed or unprofitable on S&P 500. | A clean session-gate control: compare 09:45–12:00 ET with all-hours and use next-bar fills.  The author used CFD data with acknowledged data-cleaning issues, and comments warn about bull-market bias and possible entry/stop lookahead. |
| Futures ORB/VWAP/volume | In [“MNQ Futures — 5-Year Backtest Results…”](https://www.reddit.com/r/algotrading/comments/1saburp/mnq_futures_5year_backtest_results_across_4/), the author reports 1-second/1-minute data, walk-forward next-bar execution, commission, and Monte Carlo checks.  The reported table shows ORB + VWAP (first New York hour, one trade/day, 1:2 R:R) at +$6,092/PF 1.18 over 943 trades; a 15-minute opening-range displacement setup is reported at +$6,485/PF 1.20 over 524 trades.  A volume-filtered FVG setup is reported at +$2,066/PF 1.14 over 2,817 trades. | This is the most concrete futures/session example found, but it is still a single self-reported backtest and the post does not publish raw data or code.  Re-test with the project’s exact fill model, fees, contract rolls, and held-out contracts/weeks. |
| Relative volume | In [a paper-replication discussion](https://www.reddit.com/r/algotrading/comments/1dhs545/has_anyone_reviewed_this_paper_on_an_opening-breakout-strategy/), the proposed equity strategy ranks the top 20 of 7,000 stocks by opening-five-minute relative volume, trades the opening direction, and uses an ATR stop.  The Reddit author could not reproduce the claimed results on selected stocks and notes the source was a non-peer-reviewed preprint.  Separately, [a 9/21 EMA paper-trading report](https://www.reddit.com/r/algotrading/comments/1u2g8ye/59_days_of_paper_trading_a_921_ema_crossover/) adds a previous-day-volume > 80% of 10-day-average confirmation and ATR stops, but only 14 trades were closed and two winners supplied most of the P&L. | RVOL is worth testing as a participation/quality gate, especially normalized by session minute, but not as proof that low volume implies inversion.  Opening volume is U-shaped and contract-specific; same-minute history and roll-aware data are required. |
| Higher-timeframe EMA geometry | In [“Using a trend filter”](https://www.reddit.com/r/algotrading/comments/1uz2fpi/using-a-trend-filter/), the proposed rule allows 5-minute longs only when the 1-hour price is above EMA(50) and EMA(50) is above EMA(200).  Comments recommend a higher-timeframe EMA slope, an ATR-normalized separation floor, and two or three closed-bar persistence; they also recommend comparing long/short/both/none buckets OOS. | This maps directly to EMA/HMA orientation: preserve a raw cross when higher-timeframe direction agrees, and abstain when averages are stacked but flat.  The thread is advice, not a performance report. |
| HMM/GMM/state models | In [“regime detection actually moved my live results”](https://www.reddit.com/r/mltraders/comments/1uprq3t/regime_detection_actually_moved_my_live_results/), an author claims a simple HMM+GMM changed only position size and made drawdowns shallower; a reply says it behaved poorly at <=15-minute horizons but better at 30 minutes–2 hours in that author’s forex tests.  In [“Let’s talk about regime detection”](https://www.reddit.com/r/algotrading/comments/1razsuv/lets_talk_about_regime_detection/), another practitioner reports a Markov-switching macro FX module with multiple walk-forward tests and promising medium-term bias, while explicitly saying it is too slow for day trading. Other comments recommend volatility clustering/GMM over a complex HMM because of lag and overfit risk. | Use only forward-filtered probabilities, stationary features, a small state count, and a neutral state.  Never use a full-sample smoothed/Viterbi path to label historical bars.  These reports support an experiment, not an intraday orientation flip. |
| Meta-labeling / classifier | The long [meta-labeling write-up](https://www.reddit.com/r/algotrading/comments/1lnm48w/meta-labeling-for-algorithmic-trading-how-to/) recommends a base strategy with an existing edge, labels every candidate signal (including signals that would later be skipped), and trains a secondary classifier on entry-time features.  The author claims a typical 1–3 percentage-point win-rate improvement and larger drawdown benefit.  A follow-up claims six base models, Databento MBO/FRED features, and two production ES/NQ systems around PF 1.15 and Calmar 2.5, with >2,500 trades and weekly retraining. | This is the best blueprint for an ML gate, not evidence of the claimed edge.  The post correctly stresses exact signal-time features, nested/purged CV, calibration, costs, and comparison to the raw strategy.  Use a small logistic/linear model first; do not train a direction model on future outcomes. |
| ML leakage and sample-size warning | In [“Seeking Sanity Check on Order Flow Strategy”](https://www.reddit.com/r/algotrading/comments/1mic9r0/seeking-sanity-check-on-order-flow-strategy/), a three-stage footprint ML system reports 75% win rate/PF 21.15 after removing a feature that gave AUC 1.0, but only four trades.  In [“Has anyone had success with ML?”](https://www.reddit.com/r/algotrading/comments/1progoo/has-anyone-had-success-with-ml/), commenters distinguish “good/bad trade” labels from buy/sell labels and suggest roughly 1,000+ prior trades before attempting meta-labeling. | Treat tiny-trade high-PF results as a leakage/sample-size warning.  Preserve all candidate events, including rejected and overlapping events, and embargo the label horizon in walk-forward training. |
| Order-book / OFI | In [“Algo only based on Orderbook Imbalance”](https://www.reddit.com/r/algotrading/comments/1pgsphr/algo-only-based-on-orderbook-imbalance-could-it/), the author describes buy/sell/neutral regime selection from OBI, OFI to confirm/persist the state, and limit-only execution with 2–3 ms latency; no P&L is reported.  In [the feature discussion](https://www.reddit.com/r/algotrading/comments/11rvte5/machine_learning_metalabelling/), trade pressure is described as aggressor-side imbalance and book pressure as normalized bid/ask depth imbalance. | A useful microstructure feature specification, but not usable with OHLCV alone.  The project would need timestamped L1/L2/MBO, spoofing/persistence checks, latency, queue/partial-fill assumptions, and the deterministic DOM/tick replay path.  Test it as a confidence/size veto before allowing it to flip orientation. |
| Walk-forward discipline | In [“My Walkforward Optimization Backtesting System…”](https://www.reddit.com/r/algotrading/comments/1j187b3/my_walkforward_optimization_backtesting_system/), the author describes 2016–2024 rolling IS/OOS optimization of EMA lengths, ATR multipliers, and ADX thresholds with slippage.  Comments warn that many optimized variables can overfit.  The more detailed [meta-labeling post](https://www.reddit.com/r/algotrading/comments/1lnm48w/meta-labeling-for-algorithmic-trading-how-to/) recommends outer-fold predictions, purged CV for overlapping trades, and explicit leakage checks. | This is validation methodology rather than an alpha claim.  It should be mandatory for every candidate below. |

## What these reports imply for EMA/HMA orientation

The Reddit evidence mostly concerns **trade permission or sizing**, not a
causal claim that a measurable regime should reverse a crossover.  In
particular:

- low ADX, low ATR, and low volume are commonly described as “do not trade” or
  “reduce size,” not “take the opposite side”;
- high volatility can mean either a clean trend or directionless whipsaw, so
  ATR without a direction/persistence feature is ambiguous;
- a higher-timeframe EMA/HMA slope and normalized separation can define a
  directional context, but the mapping from context to *normal versus
  inverted* must be learned/frozen on earlier windows; and
- opening/session effects are strong enough to confound an orientation test.
  A gate that only appears profitable because it removes overnight or lunch
  trades is a time filter, not evidence of inversion.

The conservative output states should be `Normal`, `Inverted`, and `Neutral`,
where `Neutral` either abstains or falls back to the base orientation.  Require
persistence/hysteresis for state changes.  Do not switch on a single feature
cross, and do not let a gate close or reverse an existing position solely
because the state changed.

## Prioritized experiment queue

Ranks are based on causal implementability with the existing replay system,
amount of independent practitioner support, and data/overfit risk.  Thresholds
below are **search bounds or starting hypotheses**, not selected parameters.
Freeze any chosen value using only earlier windows.

| Rank | Candidate and hypothesis | Feature inputs (all known at the closed decision bar) | Minimal decision logic | Data / rejection criteria |
| ---: | --- | --- | --- | --- |
| 1 | **Higher-timeframe EMA/HMA geometry + persistence.** A raw fast/slow cross is cleaner when the slower context is directional rather than stacked-and-flat. | Completed 5-minute/15-minute/1-hour bars; HTF fast/slow EMA or HMA spread; spread/ATR; slope over 2–3 completed HTF bars; optional Kaufman efficiency ratio. | `Normal` when raw cross direction agrees with HTF slope and normalized spread exceeds a small floor; `Neutral` when slope/spread is weak or disagreement is not proven. Test `Inverted` only as a separate, predeclared branch when prior-window labels show persistent opposite expectancy. Require 2 confirmations and a dwell period. | OHLCV only, causal HTF aggregation. First test 5m/15m/1m source views and contract roll boundaries. Reject if gains come only from reducing trades, if rejected events are not negative OOS, or if direction flips at every small slope change. Existing `RegimeAdaptiveGateConfig` already exposes EMA slope, normalized spread, and session features. |
| 2 | **ATR percentile × ADX/DI (trend/chop matrix).** Low-vol chop is hostile to trend crosses; high volatility needs a directional confirmation. | ATR(14) or realized-vol ratio/percentile over 30–60 prior bars or sessions; ADX(14), ADX slope, signed DI imbalance; optional efficiency/choppiness. | Example frozen matrix: ADX below a floor or efficiency below a floor → `Neutral`; ADX above floor and DI/slope agrees → `Normal`; high-vol but weak directional evidence → `Neutral`/reduced size. Evaluate 20/25 ADX and 25/50/75 ATR-percentile neighborhoods only on training folds. Do not infer inversion from chop. | OHLCV. Compare a hard gate, a soft size multiplier, and an equal-exposure control. Record expectancy of removed trades. The ADX report omitted costs/OOS, and another futures report says ADX can be lagging, so require robust results across prior/unseen windows and latency-sensitive fills. |
| 3 | **Session/opening-range + VWAP gate.** The first NY hours may have different continuation/whipsaw behavior from overnight and lunch. | Exchange-local timestamp; completed 15-minute or 1-hour opening range; session VWAP and distance/side; time since open; prior session range; optional OR width / ATR. | Permit the EMA/HMA cross only in a predeclared window (e.g. 09:45–12:00 ET) and only when breakout direction agrees with VWAP. Add an `OR width / ATR` bucket as a volatility veto, not a post-hoc best-hour selector. Run all-hours, RTH-only, and window-only controls. | Standard OHLCV plus reliable timezone/session calendar; no external data. Use next-bar entry after the confirming close. Reject if the window was selected by inspecting the test P&L, if a single open trade supplies most P&L, or if CFD/continuous-contract behavior differs from the traded future. |
| 4 | **Session-normalized RVOL / liquidity participation.** Low participation may make crosses noisier; unusually high participation may confirm breakouts, but direction is not implied. | Current completed-bar volume; same-minute-of-session median/mean from prior sessions; rolling RVOL; bar count/quote availability; spread/depth if available. | `Neutral` or smaller size below a low RVOL percentile; permit only if RVOL is in a predeclared middle/high band and price/EMA direction agrees. Test low-tail veto and high-tail confirmation separately. Missing/non-positive volume must be neutral, not zero. | OHLCV volume is enough for the first pass; same-minute history is needed to avoid the opening U-shape. Roll-adjust volume and compare standard, volume, and range bars. The opening-volume paper was not reproducible and the 14-trade EMA/RVOL report is underpowered. Existing `VolumeRegimeConfig` compares current volume only with preceding bars; a same-session variant must not use future bars. |
| 5 | **Causal meta-label logistic/ensemble.** A base EMA/HMA cross may have edge, while entry-time context can identify a subset of bad crosses. | Event-time raw direction, EMA/HMA spread/slope, ATR percentile, ADX/DI, efficiency/choppiness, RVOL, session/OR/VWAP, prior gate state, data-quality flags; add L1/L2 only in a separate study. | Log every raw candidate, including signals that overlap or would be skipped. Label only after a fixed outcome horizon or actual replay exit. Train a simple calibrated logistic/regularized tree on prior folds; `Neutral/abstain` below a confidence threshold; size smoothly with calibrated probability before testing direction flips. | Needs many events (practitioner guidance suggests 1,000+; more is better), exact feature snapshots, purged/embargoed walk-forward CV, and costs. Compare OOS meta strategy to the raw strategy and equal-trade-count sizing. Never use current/future fill price, next-bar outcome, or full-sample normalization in features. Rust replay can consume a frozen per-event schedule; training can remain offline. |
| 6 | **Forward-filtered HMM/GMM or small Markov state model.** Latent volatility/trend states may smooth noisy gates, but state confirmation is delayed. | Stationary log returns, rolling realized volatility, ATR ratio, EMA/HMA slope/spread, efficiency ratio; 2–3 states maximum. | Fit/refit only on the training prefix. At each bar use forward-filtered state probabilities, not smoothed/Viterbi labels. If max probability is below a confidence floor or state just changed, `Neutral`; otherwise use a state-to-action map frozen from training. Add hysteresis and minimum dwell. Test sizing/permission before orientation inversion. | Requires offline model fitting and careful serialization; 1-minute behavior may be pathological. Use walk-forward refits and state interpretability checks (return/volatility signature), not state names. Existing `ReplayMarkovOrientationGate` provides a causal score/neutral/dwell pattern but is not an HMM and must not be treated as one. |
| 7 | **OFI/OBI/DOM microstructure veto.** Persistent order-flow/depth imbalance may distinguish a genuine cross from a thin or adverse-selection cross. | Timestamped bid/ask or MBO snapshots; top-k depth imbalance; OFI over a short event/bar window; aggressor trade pressure; spread, depth, quote age, cancellations, and data-quality flags. | At bar close, require a minimum depth/quote-quality floor and same-sign OFI/OBI for K observations before permitting the raw orientation; disagreement → `Neutral`/smaller size. Do not flip direction from a single snapshot. | Requires L1/L2/MBO sidecar and deterministic tick/DOM fills, latency, queue and partial-fill assumptions. OHLCV replay cannot validate it. First compare no-book, quote-aware, and DOM fill models; reject if the apparent edge disappears under spread/latency or if snapshots are stale/spoofable. |

### Mandatory control matrix for every rank

For each candidate, run the same raw EMA and HMA baselines under identical
windows, contract, bar construction, blockout, protection, fee, and fill
settings:

1. no gate;
2. constant normal orientation;
3. constant inverted orientation;
4. gate with normal/neutral only;
5. gate with any inversion branch;
6. equal-exposure sizing or random/blocked-event control when the gate removes
   trades.

Report aggregate and per-window P&L after costs, max drawdown, number of
signals/trades, rejected-signal expectancy, state occupancy, inversion rate,
orientation regret versus each fixed baseline, latency/fill model, and
session/contract splits. A positive aggregate is not enough: require sign and
drawdown stability on unseen chronological windows and at least one additional
contract or data representation.

## Minimal causal replay formulation

The implementation can be replay-only and need not alter live strategy
defaults.  A valid decision event is:

```text
for each completed decision bar b in timestamp order:
    history := bars with timestamp <= b.timestamp
    raw := EMA/HMA cross evaluated from history
    features := gate features from history only
    state := gate(features, prior gate state)
    if raw signal and state permits it:
        submit at b close + configured latency
        fill at the first eligible later raw event/bar under the selected model
    only after an outcome is observable:
        update any online score / shadow expert / training prefix
```

Specific safeguards:

- “Current” means a completed decision bar.  A signal based on bar `b` may not
  fill at `b.open`; use next-bar-open or the deterministic fixed-latency event
  path.  If the feed has only coarse bars, label the fill as bar-approximate.
- Build higher-timeframe bars incrementally.  A 1-hour EMA/HMA feature at a
  5-minute decision may use only the last completed 1-hour bar; never use the
  still-forming hour’s eventual close.
- Compute RVOL against preceding same-session observations.  Do not let the
  current partial bar raise its own reference, and reset or mark missing data
  at session gaps, rolls, and feed holes.
- Mark a shadow cross outcome only when its chosen horizon/next-cross/fill is
  already observable.  A Markov or expert selector may then choose the *next*
  signal; it may not relabel a past signal after seeing its outcome.
- Fit thresholds, state-to-action maps, feature scaling, and classifiers only
  on the training prefix of each chronological fold.  Use a purge/embargo at
  least as long as the label/holding horizon when candidate events overlap.
  Do not use full-history quantiles, full-sample HMM smoothing, or a test
  window’s “best orientation” as an input.
- For a meta-label model, record every raw candidate before applying the model,
  including candidates that overlap an existing position.  Otherwise the model
  learns from a selected sample and its skip rate is not causal.
- If a required input is absent (volume, quote, DOM, higher-timeframe bar),
  return `Neutral`/hold according to the experiment specification.  Do not
  silently convert missing data to zero, normal, or inverted.

The repository already has useful replay-only building blocks: the adaptive
gate exposes causal relative-volume, ATR, ADX/DI, EMA spread/slope, and session
features; `VolumeRegimeConfig` compares a current bar with preceding volume; and
`ReplayMarkovOrientationGate` implements bounded shadow outcomes, gap reset,
neutral handling, confirmation, dwell, and a prefix-causality test pattern.
These are implementation references only; this memo does not enable or modify
any production strategy.

## Acceptance / stop rules before considering live use

Do not enable a candidate because it has positive P&L or a smoother chart.  A
candidate must first:

- beat the no-gate and fixed-orientation controls on untouched chronological
  windows after fees, slippage, and the selected deterministic fill model;
- remain directionally and economically interpretable when thresholds move
  over a small predeclared neighborhood;
- show that rejected trades have worse expectancy than retained trades and that
  equal-exposure sizing alone does not explain the improvement;
- preserve trade/event counts sufficient for uncertainty estimates (four trades
  and a 20-trade paper sample are not evidence of an edge);
- survive an additional contract, week, session representation, and at least
  one adverse volatility/chop period; and
- pass a prefix test: truncating future bars must not change an earlier feature,
  state, decision, or scheduled fill.

The default deployment posture should remain **no orientation inversion** until
one of the above experiments demonstrates a stable, causal, out-of-sample
mapping.  A neutral/abstain outcome is preferable to forcing a normal or
inverted trade when the regime classifier is uncertain.

## Source index (direct Reddit links)

The detailed links are embedded above.  The main threads used were:

- [Regime filters: simple ATR compatibility matrix](https://www.reddit.com/r/algotrading/comments/1se1rof/do_you_use_regime_filters/)
- [Regime filters: state gating, soft versus hard filters](https://www.reddit.com/r/algotrading/comments/1s7dvbu/do_you_use_regime_filters/)
- [How to establish a successful market regime filter?](https://www.reddit.com/r/algotrading/comments/1rvfy12/how_to_establish_a_successful_market_regime_filter/)
- [Let's talk about regime detection](https://www.reddit.com/r/algotrading/comments/1razsuv/lets_talk_about_regime_detection/)
- [HMM + GMM sizing report](https://www.reddit.com/r/mltraders/comments/1uprq3t/regime_detection_actually_moved_my_live_results/)
- [ADX strategy backtest](https://www.reddit.com/r/algotrading/comments/1irhrcw/backtest_results_for_an_adx_trading_strategy/)
- [Opening-range breakout backtest](https://www.reddit.com/r/algotrading/comments/1j9pxsr/backtest_results_for_the_opening_range_breakout/)
- [MNQ five-year strategy comparison](https://www.reddit.com/r/algotrading/comments/1saburp/mnq_futures_5year_backtest_results_across_4/)
- [Opening-breakout relative-volume replication check](https://www.reddit.com/r/algotrading/comments/1dhs545/has_anyone_reviewed_this_paper_on_an_opening-breakout-strategy/)
- [EMA/RVOL paper-trading sample-size warning](https://www.reddit.com/r/algotrading/comments/1u2g8ye/59_days_of_paper_trading_a_921_ema_crossover/)
- [Higher-timeframe EMA trend filter](https://www.reddit.com/r/algotrading/comments/1uz2fpi/using_a_trend_filter/)
- [Meta-labeling workflow and safeguards](https://www.reddit.com/r/algotrading/comments/1lnm48w/meta-labeling-for-algorithmic-trading-how-to/)
- [ML success/sample-size discussion](https://www.reddit.com/r/algotrading/comments/1progoo/has_anyone_had_success_with_ml/)
- [Order-book imbalance and OFI](https://www.reddit.com/r/algotrading/comments/1pgsphr/algo-only-based-on-orderbook-imbalance-could-it/)
- [Walk-forward optimization](https://www.reddit.com/r/algotrading/comments/1j187b3/my_walkforward_optimization_backtesting_system/)
- [NQ EMA + chop/ATR filter](https://www.reddit.com/r/algotrading/comments/1q7j6zh/is_this_trading_strategy_tradable/)

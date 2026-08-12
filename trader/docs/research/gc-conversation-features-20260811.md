# GC EMA 10/30 conversation-feature experiments

Date: 2026-08-11  
Status: replay research only; no live configuration change.

## What was tested

The conversation suggested adding regime/context to the EMA cross. I tested
three of its most direct ideas on exact-contract GCZ6 data:

- Kaufman Efficiency Ratio (ER) as a causal low-efficiency neutral veto;
- ADX as a low-strength inversion proxy;
- price distance from the slow EMA, normalized by ATR, as a stretch gate.

The replay protocol was standard one-minute bars, closed-bar signals,
raw-bar-open fills, full source sessions, guarded execution, a 45-minute
blockout, no TP/SL/trailing protection, and the existing GC fee schedule.
The August 3–7 window was used for candidate screening. February 9–13 was a
frozen validation window. June 15–19 and July 14–17 were additional frozen
checks for the selected stretch threshold.

## Results

Net P/L is USD.

| Candidate | Aug 3–7 screen | Feb 9–13 frozen validation | June 15 check | July 14 check |
|---|---:|---:|---:|---:|
| ER < 0.05 neutral fallback | +$19,141.40 | -$25,902.50 | — | — |
| ADX < 18 inversion proxy | +$17,615.80 | -$6,882.90 | — | — |
| Slow-EMA stretch > 1.5 ATR | +$22,883.00 | +$44,029.10 | -$7,890.30 | -$7,616.80 |

The February stretch result is not an execution-path artifact. A same-path
`VolumeAdaptiveEmaCross` control with the adaptive gate disabled made
`-$25,902.50`; enabling only the stretch gate made `+$44,029.10`. The gain was
mostly one large short trade during the February selloff, so it is a
window-specific outcome until more independent windows confirm it.

For reference, the fixed EMA 10/30 controls on February were `-$25,902.50`
normal and `+$25,437.50` inverted. The ADX proxy improved on fixed normal but
still lost to fixed inverted. The stretch gate lost on both additional checks
against the same-path normal controls: June normal `-$3,448.70`, July normal
`+$151.40`.

## Interpretation

The conversation produced useful features, but not a robust selector yet:

- Stretch is the most interesting feature, but 2/4 windows positive after
  freezing the 1.5-ATR threshold is not enough to deploy.
- ADX behaved plausibly as a regime filter, but the current implementation is
  only an inversion proxy. It does not implement the conversation's proposed
  neutral/permission state.
- ER was implemented causally, but the tested `ER < 0.05` veto did nothing on
  February and matched fixed normal. That threshold is likely too restrictive,
  or ER needs to be combined with a true abstain/permission decision.
- EMA spread and normalized slope are already available in the native adaptive
  gate, but were not selected by this focused pass. KAMA and ALMA were not
  added: the conversation itself suggests testing ER before replacing the
  existing EMA pair, and KAMA would mostly compress ER into another smoother.

None of the numerical thresholds above came from Reddit. The Reddit memo
provided broad hypotheses—ATR/volatility context, ADX/DI, persistence,
session/VWAP, and neutral states. The ER cutoff, ADX proxy behavior, stretch
definition, and thresholds were local replay experiments.

## Artifacts

- August feature sweep: `.run/replay-sweeps/gc-convo-features-20260811/aug03/`
- February validation and same-path control:
  `.run/replay-sweeps/gc-convo-features-20260811/controls_validation/`
- Additional June/July frozen checks:
  `.run/replay-sweeps/gc-convo-features-20260811/extra_validation/`
- Existing causal adaptive-gate implementation: `src/strategies/adaptive_gate.rs`
- Existing causal replay ER implementation: `src/strategies/markov_orientation_gate.rs`

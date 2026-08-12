# GC daily orientation prior with cross-by-cross execution — 2026-08-12

## Result

Among the cross-by-cross day-prior variants tested here, the original `RVOL < 0.95` + `DI imbalance < 0`, `XOR` gate remains strongest. A day label contains useful descriptive information, but using it to switch the cross gate to `any` or `all` did not improve the July holdout.

The frozen candidate was replayed on 49 prior trading days, then the day-prior overlays were tested on the July 14–31 holdout and the latest August 3–7 reference week. All runs use 1-minute standard bars, EMA 10/30, raw-bar-open fills, zero replay latency, and the existing GC fee schedule.

## What was tested

Each prior day was replayed twice from flat:

- normal EMA orientation;
- fully inverted EMA orientation.

The hindsight label is whichever lane had higher net P&L; differences under $10 are marked `tie`. This label is for research only and is not available before the day begins.

The cross candidate remains event-level. At every raw EMA cross, RVOL and DI each produce an inversion vote. The day-prior overlay adds a third vote:

- day prior `inverted` → `combine=any`;
- day prior `normal` → `combine=all`;
- uncertain prior → original `combine=xor`.

This keeps individual crosses free to change orientation. It does not force one direction for the whole day.

## Hindsight labels across the prior 49 days

| Episode | Daily labels, in order | Contract note |
|---|---|---|
| Feb 16–20 | tie, inverted, tie, tie, tie | GCZ6 |
| Apr 20–24 | inverted, tie, tie, inverted, tie | GCZ6 |
| May 11–15 | inverted, inverted, inverted, inverted, inverted | GCQ6 transition-era data |
| May 25–29 | inverted, inverted, normal, inverted, normal | GCZ6 |
| Jun 8–12 | inverted, inverted, inverted, normal, inverted | GCQ6 |
| Jun 15–19 | normal, inverted, inverted, inverted, normal | GCZ6 |
| Jun 22–26 | normal, normal, normal, inverted, inverted | GCQ6 |
| Jul 14–17 | normal, normal, inverted, inverted | GCZ6 |
| Jul 20–24 | inverted, normal, inverted, normal, inverted | GCZ6 |
| Jul 27–31 | normal, inverted, normal, inverted, inverted | GCZ6 |

Counts: 28 inverted, 14 normal, and 7 ties.

The corresponding 49-day control totals were normal `-$32,857.6`, inverted `+$18,622.4`, and hindsight daily selection `+$142,922.4`. That last number is an oracle comparison, not a tradable result.

The day-level descriptive relationship was directionally sensible but not decisive:

| Hindsight label | Mean full-day net move | Mean full-day efficiency ratio |
|---|---:|---:|
| Inverted, n=28 | `-21.3` | `0.114` |
| Normal, n=14 | `+2.6` | `0.141` |
| Tie, n=7 | `+12.0` | `0.149`* |

\* ER was available for only four tie days. Inverted days were generally less efficient and more often down, but the overlap is too large for a fixed day classifier.

## Latest week: label versus cross behavior

The latest Aug 3–7 controls were normal, inverted, normal, normal, inverted. The frozen XOR gate’s cross decisions were:

| Day | Hindsight day label | Inverted crosses | Normal crosses | XOR candidate net |
|---|---|---:|---:|---:|
| Aug 3 | normal | 25 | 17 | `+$6,152.6` |
| Aug 4 | inverted | 27 | 20 | `+$2,663.6` |
| Aug 5 | normal | 17 | 16 | `-$6,666.8` |
| Aug 6 | normal | 19 | 14 | `+$3,547.0` |
| Aug 7 | inverted | 18 | 23 | `+$2,483.6` |
| **Total** | — | **106** | **90** | **`+$8,180.0`** |

Aug 3 and Aug 6 are the important counterexamples: the majority of their crosses were inverted even though the hindsight whole-day lane favored normal. Cross count cannot be used as a day selector.

## Overlay replay results

| Overlay | July 14–31 | Aug 3–7 | Combined |
|---|---:|---:|---:|
| Cross-only XOR baseline | `+$13,081.8` | `+$8,180.0` | **`+$21,261.8`** |
| Hindsight day label → `any`/`all` | `+$501.2` | `+$13,872.4` | `+$14,373.6` |
| Causal lag-1 ER, rolling 10-day median | `-$25,049.8` | `-$3,838.6` | `-$28,888.4` |
| Causal lag-1 ER, 0.75/1.25 neutral band | `-$4,758.2` | `+$8,180.0` | `+$3,421.8` |

The causal prior used only the previous completed day’s full-day ER60. The neutral-band version selected `XOR` on 18 of 19 test days, so it mostly declined to make a day-level claim. The unbanded version made a day claim on every test day and was materially worse.

The hindsight overlay helped the current August week, but hurt the July holdout. This is evidence that a day label can explain a finished episode without being a stable causal predictor for the next episode.

## Frozen candidate over the prior diagnostic history

The cross-only RVOL/DI XOR candidate produced `+$36,285.6` over the 49 prior daily diagnostic episodes: 24 positive, 21 negative, and 4 approximately flat. Adding the latest Aug week brings the same `.95` candidate to `+$44,465.6` across those 54 daily episodes, but the August week is still the latest reference rather than an independent future holdout.

## Decision

Do not promote a whole-day normal/inverted selector. Keep orientation cross-by-cross with XOR as the baseline. A regime prior may be retained as a soft diagnostic or a tie-breaker only after it survives a new untouched week; the current rolling ER prior does not qualify.

## Reproducibility artifacts

- Daily normal/inverted controls: `.run/replay-sweeps/gc-daily-orientation-labels-20260812/`
- 49-day cross diagnostics: `.run/replay-sweeps/gc-daily-cross-diagnostics-20260812/`
- July/Aug overlay replays: `.run/replay-sweeps/gc-day-prior-cross-overlay-20260812/`
- Current research candidate definition: `RVOL < 0.95`, `DI imbalance < 0`, `combine=xor`, minimum two feature votes, one confirmation bar.

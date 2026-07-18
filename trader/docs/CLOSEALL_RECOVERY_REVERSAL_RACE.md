# CloseAll Reversal Recovery Race Findings

## Context

The issue was observed with Native Rust HMA Crossover using broker-owned Tradovate protection:

- Strategy: HMA Crossover
- Timing: Closed Bar
- Execution path: Guarded
- Reversal mode: CloseAll > Enter
- TP/SL broker-owned through `orderStrategy/startorderstrategy`

The symptom was that long reversals appeared to fire, but the account returned flat almost immediately. In session stats, this looked like many consecutive shorts because the long side was not allowed to remain open long enough to show as a normal running long.

## Relevant Code Path

For `CloseAll > Enter`, the software sends a Tradovate close request first, then starts the new broker-owned order strategy:

```text
order/liquidateposition
orderStrategy/startorderstrategy
```

The submit sequence is in `src/tradovate/gateway/submit.rs`:

```rust
let liquidation_ack = submit_liquidation_via_gateway(request_tx, liquidation).await?;
let mut strategy_ack = submit_order_strategy_via_gateway(request_tx, strategy).await?;
```

The important detail is that the liquidation ack means the broker accepted the close request. It does not prove that all position/order-strategy side effects from the close request are finished.

## Recent Change Under Suspicion

Commit:

```text
4da7920 Fix Tradovate broker-owned reversal handling
Date: 2026-07-13 13:56:35 -0400
```

This added CloseAll recovery state and logic:

- `pending_closeall_reversal_entry` in `src/tradovate/types.rs`
- setting that pending recovery state in `src/tradovate/orders/dispatch.rs`
- `recover_closeall_reversal_if_flat` in `src/tradovate/execution/strategy.rs`

That recovery function is what produces this log text:

```text
closeall reversal recovered from flat after broker path cleared
```

## What The Logs Show

The clearest July 14 sequence was:

```text
15:14:00.392 Close submitted; Strategy submitted: Buy 1
15:14:00.449 initial Buy fill
15:14:01.048 recovered Buy submitted
15:14:01.133 recovered Buy fill
15:14:01.192 Position confirmed at target 1
15:14:01.933 flat broker sync wait
```

This means the recovered long was actually confirmed at target `1`. After that, the account sync reported flat while the just-submitted order path was disappearing. There was no new sell signal causing that flatten.

Other July 14 examples showed the same pattern:

```text
13:10:01.083 recovered Buy submitted
13:10:01.339 Position confirmed at target 1
13:10:01.416 flat broker sync wait
```

```text
13:21:01.651 recovered Buy submitted
13:21:01.883 Position confirmed at target 1
13:21:01.908 flat broker sync wait
```

```text
13:41:01.356 recovered Buy submitted
13:41:01.504 Position confirmed at target 1
13:41:01.998 flat broker sync wait
```

The confirmed-to-flat delay ranged from tens of milliseconds to hundreds of milliseconds.

## Yesterday Versus Today

This was not completely new on July 14.

The July 13 logs already showed recovered entries confirming and then going flat:

```text
09:50:17.413 recovered Sell submitted
09:50:17.530 Position confirmed at target -1
09:50:18.088 flat broker sync wait
```

However, it was less frequent and mostly visible on recovered sells. On July 14, it was frequent and visible on recovered buys, which made it look like the long side was broken and the strategy kept ending up short or flat.

Counts from inspected logs:

- `session-20260713T132425Z.txt`: 2 recovered entries, 2 flat broker sync waits
- `session-20260713T135117Z.txt`: 8 recovered entries, 2 flat broker sync waits
- `session-20260714T184105Z.txt`: 10 recovered entries, 10 flat broker sync waits
- `session-20260714T192004Z.txt`: 11 recovered entries, 11 flat broker sync waits

The July 14 files overlap, so the exact unique count is lower than the raw sum, but the frequency is still much higher than July 13.

## Why Recovery Is A Plausible Culprit

After `startorderstrategy` succeeds, the software does not intentionally send another close for the recovered entry. It mostly does bookkeeping:

- clears `order_submit_in_flight`
- records the broker `order_strategy_id`
- sets `active_order_strategy`
- waits for account sync
- clears `pending_target_qty` after the target is reached

Positive TP/SL protection sync is intentionally ignored for broker-owned Tradovate entries, so the app is not creating separate software TP/SL orders after entry.

The suspicious behavior is specifically this:

1. CloseAll reversal starts.
2. Initial close/order-strategy sequence gets the account flat instead of safely reversed.
3. Recovery sees flat and submits the intended entry again.
4. The recovered entry fills and confirms the target.
5. Broker/account state shortly after reports flat again.

That suggests the recovery entry is being submitted while the broker-side close-all/order-strategy cleanup is still unwinding. In other words, the recovery can still be inside the same race window it was trying to repair.

## What Reverting The Recovery Tests

Reverting the `pending_closeall_reversal_entry` / `recover_closeall_reversal_if_flat` change would test whether the new recovery submit is what causes the confirmed-then-flat churn.

If reverting removes this pattern:

```text
recovered entry -> target confirmed -> flat broker sync wait
```

then the recovery path is very likely the immediate culprit.

This test does not necessarily prove that old `CloseAll > Enter` is fully correct. Without recovery, the first immediate `startorderstrategy` can still be consumed by the close-all transition and leave the account flat. The revert test is useful because it separates two problems:

- the older CloseAll timing risk, where liquidation ack is not the same as confirmed flat
- the newer recovery behavior, where a second broker-owned entry may be submitted before the close-all cleanup is truly done

## Current Working Hypothesis

The likely failure is not an intentional software close after entry. The likely failure is a broker-state timing race:

```text
liquidate accepted
new strategy submitted
initial entry consumed by close-all transition
recovery entry submitted
recovery entry confirmed
delayed close-all/order-strategy cleanup still flattens or reports flat
```

The `Flatten > Confirm > Enter` path was added as the cleaner comparison because it waits for a broker-confirmed flat state and cleared broker path before submitting the new broker-owned strategy. That path should avoid the recovery race instead of trying to repair it after the fact.

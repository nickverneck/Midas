# Tradovate Broker-Owned Protection Policy

## Summary

For native Tradovate strategy automation, protection should be broker-owned.

The app should not synthesize software/app-managed TP, SL, or trailing-stop
orders after an entry has already been submitted through Tradovate
`orderStrategy/startorderstrategy`.

## Why

The simple intended flow is:

```text
signal -> entry -> broker-owned TP/SL/trail
```

But with Tradovate, a `startorderstrategy` entry is represented by several async
broker objects:

```text
orderStrategy
entry order
fill
position
TP order
SL order
orderStrategyLink rows
strategy status
```

Those updates do not always arrive together. If the app tries to "repair"
protection from a partial snapshot, it can create a second OCO after the original
broker-owned bracket has already exited or is still cleaning up.

That is what showed up in the logs:

```text
position flattened
old strategy linked orders still visible
app placed order/placeOCO
app immediately cleared it
the temporary stop could fill and flip the account
```

So the right fix is not a delay or cooldown. The right fix is ownership.

## Policy

### Guarded Native Entries

When native automation uses `orderStrategy/startorderstrategy`:

```text
entry: broker-owned
take profit: broker-owned
stop loss: broker-owned
trailing stop: broker-owned
```

The app may observe and display broker protection, but it should not create
separate fallback protection with:

```text
order/placeOCO
order/placeorder
```

### CloseAll > Enter

For `CloseAll > Enter` reversals:

```text
1. order/liquidateposition
2. orderStrategy/startorderstrategy
```

The new entry and its TP/SL/trail are broker-owned through the new order
strategy.

The app should not add separate native OCO protection after the new strategy
entry.

### Flatten > Confirm > Enter

For `Flatten > Confirm > Enter` reversals:

```text
1. flatten to zero
2. wait until flat is confirmed
3. orderStrategy/startorderstrategy
```

The new entry and protection are still broker-owned.

### Direct Reversal / Direct Market Orders

Direct mode uses `order/placeorder`, not `startorderstrategy`.

Because direct mode does not create a broker-owned bracket, TP/SL/trailing
controls should be disabled or hidden for direct mode.

Direct mode should not create software-defined TP/SL/trailing protection.

If the user wants TP/SL/trailing protection, they should use an execution mode
that routes through `orderStrategy/startorderstrategy`.

## UI Rules

Protection fields should appear only when the selected execution mode can route
broker-owned protection.

Show TP/SL/trailing controls for:

```text
Guarded + CloseAll > Enter
Guarded + Flatten > Confirm > Enter
Guarded non-reversal entries
```

Hide or disable TP/SL/trailing controls for:

```text
Direct reversal
Direct/placeOrder paths
Simple diagnostic paths
HMA direct paths
manual placeOrder-only paths
```

## Engine Rules

The engine should enforce these rules even if the UI is wrong:

```text
if selected position is flat:
    never place TP/SL/OCO
    only clear known app-managed protection if it exists

if entry was startorderstrategy:
    trust the broker-owned bracket
    do not synthesize app-managed OCO

if entry was placeOrder:
    do not create software TP/SL/trailing protection
    protection options should have been disabled
```

## Terminology

Broker-owned protection:

```text
Protection created by Tradovate as part of orderStrategy/startorderstrategy.
```

App-managed protection:

```text
Protection the app creates separately with order/placeOCO or order/placeorder.
```

The desired direction is to eliminate app-managed protection for native
automation and rely on broker-owned strategy protection only.

## Related Investigation

The longer incident/design note is:

```text
/home/nick/dev/Midas/docs/tradovate_protection_sync_engine_plan.md
```

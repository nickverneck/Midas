# Tradovate Protection Sync Engine Plan

## Status

Living investigation and design plan.

No runtime fix has been applied from this document yet. The goal is to design
the right engine behavior before changing order routing or protection sync.

## Problem Statement

The Tradovate native strategy path can briefly place or re-place TP/SL
protection after the actual trade has already exited. In the latest saved log,
the entry filled correctly, the TP likely hit, then the app briefly created a
new native OCO bracket and immediately cancelled it.

This is unacceptable because even a short-lived orphan stop or orphan bracket can
be dangerous:

- it can leave a stop order working after the intended position is gone,
- it can create a false reverse position if the stop fills after flat,
- it can consume order limits,
- it can make the UI look like protection is blinking,
- it can hide the real state transition that the engine should be modeling.

The fix must not add arbitrary execution delay. We should not solve this with a
cooldown, sleep, debounce, or global lock. The engine needs state correctness,
not timing luck.

## Latest Observed Sequence

Log file:

```text
.run/trader-logs/session-20260709T020452Z.txt
```

Configuration:

```text
Strategy: Native Rust / HMA Crossover
Timing: Closed Bar
Delay: 0
Execution Path: Guarded
Reversal: CloseAll > Enter
Fast: 4
Slow: 30
TP: 5 ticks
SL: 10 ticks
Trailing: off
Contract: MESU6
```

Timeline:

```text
22:04:00.313  HMA cross Buy signal, qty 0 -> 1
22:04:00.409  Strategy submitted via orderStrategy/startorderstrategy
22:04:00.904  Fill detail: Buy 1 @ 7530.00
22:04:01.166  Account sync actual position becomes 1
22:04:01.181  Position confirmed at target 1
22:04:32.494  Balance delta +5.34, consistent with TP exit
22:04:32.153  Engine logs flat broker sync wait, but old strategy has 2 active linked orders
22:04:32.550  Engine places Native TP/SL via order/placeOCO
22:04:32.625  Engine clears that native protection
22:04:32.644  Clear applied via order/cancelorder
```

The entry was not the problem:

```text
endpoint orderStrategy/startorderstrategy
fill detail | Buy 1 @ 7530.00
```

The problem is after the exit. The protection reconciliation path believed there
was still a position that needed fallback native OCO protection, even though the
trade was exiting or already flat.

## Current Behavior In Code

Main account-sync path:

```text
handle_execution_account_sync
  selected_market_position_qty
  sync_active_execution_position
  reconcile_selected_active_order_strategy
  hydrate_selected_order_strategy_protection
  pending-target handling
  selected_managed_protection_waiting_for_position_sync
  sync_execution_protection
```

Main strategy-loop path:

```text
maybe_run_execution_strategy
  selected_market_position_qty
  sync_active_execution_position
  flat broker path gate
  pending-target handling
  selected_managed_protection_waiting_for_position_sync
  strategy evaluation
  sync_execution_protection on hold/no-op paths
```

Protection sync:

```text
sync_execution_protection
  if not armed/native/protection-enabled: no-op
  if pending target exists: no-op
  if strategy-owned protection should still hydrate: no-op
  signed_qty = selected_market_position_qty
  entry_price = selected_market_entry_price
  sync_active_execution_position(signed_qty, entry_price)

  if signed_qty == 0:
      desired protection = none
      sync_native_protection(... signed_qty 0 ...)
  else:
      desired TP/SL = derived from selected entry price
      sync_native_protection(... signed_qty nonzero ...)
```

Planner:

```text
plan_native_protection_sync
  if signed_qty == 0 or no desired TP/SL:
      detach/cancel known protection
  else:
      if existing protection matches:
          no-op
      if stop-only modify is possible:
          modify stop
      else:
          detach old protection
          place TP, SL, or OCO
```

## Root-Cause Shape

This is a state ordering problem between these broker facts:

- selected contract position quantity,
- selected contract entry price,
- order strategy status,
- order strategy linked child orders,
- fill records,
- balance movement,
- local managed protection state.

Tradovate sends those facts through separate streams/entities. The local store
can temporarily contain a mixed state.

Observed mixed state after TP:

```text
position: flat, or moving toward flat
old order strategy: still active
old linked orders: still visible
managed protection: none
pending target: none
latency tracker: old strategy, no longer in grace
```

Then a later sync appears to have exposed enough nonzero position/entry data for
`sync_execution_protection` to synthesize new TP/SL protection. The next sync saw
flat and cleared it.

The engine reacted to each individual snapshot instead of treating the
position/order/protection combination as a small state machine.

## Requirements

### Functional

The engine must:

- enter immediately when a valid signal is accepted,
- keep using `orderStrategy/startorderstrategy` for guarded native entries with
  broker-owned TP/SL,
- support direct market-order paths that need app-managed protection,
- keep trailing-stop updates responsive,
- prevent duplicate same-bar closed-bar entries,
- prevent orphan native TP/SL after flat,
- avoid placing fallback native OCO when a broker-owned strategy bracket is still
  expected to protect the position,
- recover if Tradovate fails to expose strategy-owned bracket children,
- recover if a true position remains open but protection is missing,
- keep per-contract/account state isolated.

### Non-Functional

The engine must not:

- add sleeps,
- add cooldowns,
- delay entry execution by default,
- globally serialize all strategy processing,
- block new opposite signals after the current order path is known settled,
- depend on balance deltas for order safety,
- rely on UI chart labels as authority.

## Engine Invariants

These are the rules the code should enforce.

### Invariant 1: Flat Means No New Protection

If the selected account/contract position is explicitly flat, the protection
planner may clear known protection but must not place new TP/SL or OCO orders.

This is a hard safety rule, not a timing rule.

### Invariant 2: Entry Ownership Determines Protection Source

For entries submitted through `orderStrategy/startorderstrategy`, the first
source of protection is the Tradovate order strategy bracket. The app should not
immediately synthesize separate native OCO protection unless the strategy-owned
bracket is proven missing while the position is still open.

For entries submitted through `order/placeorder`, the app may need to synthesize
native OCO protection because there is no broker-owned strategy bracket.

### Invariant 3: Never Infer Open Position From Orders Alone

Active linked orders can imply an order path is still settling, but they must not
be enough to create new protection. New protection requires an authoritative
open selected position.

### Invariant 4: Pending Order Path Is Per Contract

Only the selected account/contract should be gated. A waiting MESU6 order path
must not block unrelated symbols or accounts.

### Invariant 5: Reconciliation Is Event Driven

The engine should re-evaluate on account/order/fill/strategy updates. It should
not wait a fixed number of milliseconds to guess that Tradovate is done.

### Invariant 6: Protection Placement Needs A Fresh Position Context

Placing or modifying native protection should require:

- selected position quantity is nonzero,
- selected entry price exists,
- position side matches desired protection side,
- position context is not contradicted by a newer flat/exit event,
- for strategy-owned entries, broker-owned bracket has been classified missing
  or unusable.

## Candidate Solutions

### Option A: Hard Flat Guard In Protection Sync

Add a guard near `sync_execution_protection` or `plan_native_protection_sync`:

```text
if selected position is explicitly flat:
    only clear protection
    never place TP/SL/OCO
```

Pros:

- simple,
- no delay,
- directly prevents the latest ghost OCO case,
- easy to test.

Cons:

- does not fully solve stale nonzero snapshots,
- does not distinguish strategy-owned protection from app-owned protection,
- may still place fallback OCO if a stale nonzero position appears after a flat
  event.

Verdict:

Useful as a safety invariant, but not sufficient by itself.

### Option B: Position Epoch / Exit Epoch

Track a monotonically increasing local epoch for selected account/contract
position state.

Example:

```text
position_epoch += 1 whenever selected position qty or entry changes
flat_epoch = position_epoch when qty becomes 0
entry_epoch = position_epoch when qty becomes nonzero
protection requests carry the epoch they were derived from
```

Before placing protection:

```text
if desired.epoch != current_position_epoch:
    drop stale protection request
if current qty == 0:
    clear only
```

Pros:

- no time delay,
- prevents stale desired protection from applying after a newer flat event,
- gives a clean model for logs/tests,
- per contract/account.

Cons:

- requires a small new runtime state map,
- tests must simulate position updates in order,
- must be careful with startup hydration where the first known position is open.

Verdict:

Strong candidate. This addresses state ordering without bottlenecks.

### Option C: Protection Ownership State Machine

Represent protection state explicitly:

```text
NoPosition
EntryPending
OpenWithStrategyBracketHydrating
OpenWithStrategyBracketLive
OpenNeedsNativeProtection
OpenWithNativeProtectionLive
FlatClearingProtection
FlatNoProtection
```

Allowed transitions:

```text
NoPosition -> EntryPending
EntryPending -> OpenWithStrategyBracketHydrating
OpenWithStrategyBracketHydrating -> OpenWithStrategyBracketLive
OpenWithStrategyBracketHydrating -> OpenNeedsNativeProtection
OpenNeedsNativeProtection -> OpenWithNativeProtectionLive
Open* -> FlatClearingProtection
FlatClearingProtection -> FlatNoProtection
```

Forbidden transitions:

```text
FlatNoProtection -> OpenWithNativeProtectionLive without a nonzero position
FlatClearingProtection -> OpenWithNativeProtectionLive from old strategy links
OpenWithStrategyBracketHydrating -> OpenWithNativeProtectionLive unless bracket is proven missing
```

Pros:

- most robust long-term model,
- explains behavior in logs,
- cleanly separates entry, broker-owned bracket, native fallback, and flat clear,
- avoids reactionary cancel/re-place churn.

Cons:

- larger implementation,
- must keep state compact enough to avoid overengineering,
- needs focused test coverage.

Verdict:

Best long-term direction. Implementable incrementally with Option A and B.

### Option D: Cooldown/Debounce

Suppress protection replacement for N milliseconds after a sync.

Pros:

- easy,
- hides some flicker.

Cons:

- adds timing delay,
- can delay real fixes,
- can leave stale protection alive during a real position change,
- does not prove the state is correct.

Verdict:

Rejected. The user explicitly does not want cooldowns, and the engine should not
depend on timing luck.

### Option E: Use Strategy-Owned Brackets Only For Native Guarded Entries

For `orderStrategy/startorderstrategy` entries, do not synthesize native OCO at
all. Only observe the broker-owned bracket and display it.

Pros:

- eliminates duplicate fallback OCO for strategy entries,
- simpler runtime behavior.

Cons:

- if Tradovate accepts/fills entry but fails to expose or attach bracket, the
  app has no recovery path,
- direct paths still need native protection,
- weaker safety if broker strategy starts partially.

Verdict:

Too strict as a complete solution. Better rule: prefer strategy-owned bracket,
fallback to native protection only after the open position is confirmed and the
strategy bracket is proven unavailable.

## Recommended Design

Use a small per-account/contract execution state reducer with three core ideas:

1. hard flat guard,
2. position epoch,
3. protection ownership classification.

This gives deterministic behavior without execution delay.

### State To Track

Per `StrategyProtectionKey`:

```text
position_epoch: u64
last_known_qty: i32
last_known_entry_price: Option<f64>
last_flat_epoch: Option<u64>
last_open_epoch: Option<u64>
entry_owner: Unknown | OrderStrategy | MarketOrder
active_strategy_id: Option<i64>
native_protection_state: None | Syncing | Live | Clearing
strategy_bracket_state: Unknown | Hydrating | Live | Missing | Terminal
```

This does not need to block signal dispatch. It is reconciliation metadata for
protection safety.

### Protection Placement Rules

Native OCO may be placed only if all are true:

```text
selected qty != 0
entry price exists
desired protection epoch == current position epoch
no newer flat epoch exists after desired epoch
pending target is none
entry owner allows native protection
strategy-owned bracket is not live/hydrating
```

For `orderStrategy/startorderstrategy` entries:

```text
entry_owner = OrderStrategy
strategy_bracket_state = Hydrating
native fallback is blocked while bracket is Hydrating or Live
native fallback allowed only when:
    position is still nonzero
    entry epoch is current
    strategy record is terminal or missing
    no active protective child orders exist
```

For `order/placeorder` entries:

```text
entry_owner = MarketOrder
native protection may be placed as soon as position is confirmed nonzero
```

For flat:

```text
qty == 0:
    update flat epoch
    clear native protection if known
    never place new protection
    do not infer open position from active linked orders
```

### Signal Throughput

This design does not add a bottleneck for consecutive signals:

- Entry signals still route immediately when the strategy accepts them.
- Existing `pending_target_qty` and same-bar guards remain the dispatch safety
  boundaries.
- Protection reconciliation is per account/contract and event-driven.
- No global queue is introduced.
- No sleeps or cooldowns are introduced.

Consecutive signals on the same contract should be handled by target state:

```text
if target pending:
    do not stack new orders
if target reached:
    next signal can route
if opposite signal arrives before target reached:
    decide using effective position/pending target, not stale broker qty
```

## Implementation Phases

### Phase 1: Document And Add Observability

Add debug fields to protection sync logs:

```text
selected_qty
selected_entry_price
position_epoch
flat_epoch
entry_owner
strategy_bracket_state
native_protection_state
active_strategy_id
linked_active_count
reason protection was allowed/blocked
```

Goal:

Make the next live log explain exactly why protection was placed or skipped.

### Phase 2: Hard Flat Guard

Add a no-place rule when selected position is explicitly flat.

Test:

```text
position qty = 0
old strategy has active linked TP/SL orders
sync_execution_protection runs
expect: no order/placeOCO
expect: clear only if managed/native protection exists
```

### Phase 3: Position Epoch

Track position changes per selected account/contract.

Tests:

```text
open epoch creates desired protection
flat epoch arrives before protection sync applies
expect: stale replace request is dropped or converted to clear

flat epoch arrives, then stale nonzero position snapshot appears with older data
expect: no native OCO unless a newer open epoch is established
```

### Phase 4: Protection Ownership State

Classify strategy-owned bracket vs app-owned native protection.

Tests:

```text
startorderstrategy entry filled, bracket hydrating
expect: no fallback OCO

startorderstrategy entry filled, bracket live
expect: no fallback OCO

startorderstrategy entry filled, bracket missing, position still open
expect: fallback native OCO allowed

market-order direct entry filled, position open
expect: native OCO allowed
```

### Phase 5: Regression From Latest Log

Simulate:

```text
Buy startorderstrategy fills at 7530.00
position confirms +1
TP fills / balance moves
position becomes 0
old strategy still has 2 active linked orders
account sync and strategy loop run
```

Expected:

```text
no order/placeOCO after flat
native protection state remains none or clearing
old linked orders may cause broker sync wait, but never protection placement
```

## Test Matrix

### Entry Path

- guarded native flat entry uses `orderStrategy/startorderstrategy`,
- `CloseAll > Enter` uses `order/liquidateposition` then
  `orderStrategy/startorderstrategy`,
- direct reversal uses `order/placeorder`,
- manual orders use `order/placeorder`.

### Protection Path

- native OCO is not placed when qty is flat,
- native OCO is placed for market-order entry after open position confirmation,
- native OCO is not placed for strategy-owned bracket while hydrating,
- native OCO is not placed for strategy-owned bracket while linked orders are
  active,
- native OCO fallback is placed if strategy bracket is proven missing and
  position is open,
- clear does not cancel unrelated contract/account orders.

### Race Cases

- fill arrives before position,
- position arrives before linked orders,
- TP fill/balance update arrives before strategy terminal status,
- flat position arrives while old linked orders remain active,
- stale nonzero position appears after a flat update,
- order strategy remains active with no linked orders,
- linked child orders remain active while order strategy status is terminal.

### Throughput Cases

- repeated closed-bar fingerprint updates do not stack entries,
- new opposite signal after target reached can route,
- new same-side signal while flat is blocked until a new cross,
- consecutive opposite signals use effective pending target, not stale actual
  quantity,
- unrelated contracts/accounts are not blocked by a selected contract waiting
  for protection sync.

## Open Questions

1. Does Tradovate ever emit a position update with a stale nonzero `netPos` after
   a newer flat update for the same position id?
2. Are order strategy child orders always linked through `orderStrategyLink`, or
   can they be visible only as plain active orders with `orderStrategyId`?
3. When TP fills inside a strategy bracket, what exact order statuses arrive for
   the sibling stop and strategy record?
4. Does `orderStrategy/startorderstrategy` ever acknowledge strategy creation but
   fail to attach bracket children after entry fill?
5. Should fallback native OCO for strategy-owned entries require an explicit
   "bracket missing" observation, or should it require a manual emergency mode?

## Current Recommendation

Do not add a cooldown.

Implement the safety in this order:

1. hard flat no-place guard,
2. richer protection decision logging,
3. position epoch to reject stale protection intents,
4. strategy-bracket ownership state to decide when fallback native protection is
   allowed.

This keeps execution fast while making protection placement depend on confirmed
state, not on arbitrary elapsed time.

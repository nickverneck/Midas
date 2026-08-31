//! Deterministic, broker-owned bracket protection.
//!
//! This module deliberately knows nothing about Tradovate, serde, sockets, or
//! replay fixtures.  A parent broker adapter can translate [`FillIntent`] and
//! [`ProtectionEvent`] into its wire protocol.
//!
//! Prices are integer ticks rather than floating point values.  The caller is
//! responsible for converting an instrument price to ticks using the
//! instrument's tick size.  This makes level comparisons and overflow checks
//! deterministic across machines.
//!
//! # Market-data assumptions
//!
//! A [`MarketTick`] is an executable two-sided quote.  A long position exits
//! by selling at the bid; a short position exits by buying at the ask.  The
//! state machine therefore uses the bid for all long-side trigger and trail
//! calculations, and the ask for all short-side calculations.  It does not
//! use a last-trade price, because a last trade is not necessarily executable
//! for an exit.
//!
//! A single quote can satisfy both a take-profit and a stop condition when a
//! dynamic trailing stop has ratcheted through the take-profit level.
//! [`ExitPrecedence`] makes that case explicit. Static positive TP/SL offsets
//! cannot overlap on their own, and crossed quotes are rejected.
//!
//! Trailing protection replaces the static stop leg when it activates; it is
//! not a second stop order.  Until activation, the static stop remains the
//! protective stop.  Once an exit intent is emitted, the bracket is terminal
//! and subsequent ticks cannot produce another intent.  The parent should
//! treat the intent as a broker-owned fill and publish the resulting fill and
//! position update atomically with the sibling cancellation events.

use std::convert::TryFrom;

/// An instrument price expressed in the instrument's minimum tick units.
pub type PriceTicks = i64;

/// Monotonic sequence supplied by the replay or market-data adapter.
pub type TickSequence = u64;

/// Direction of the protected position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

impl PositionSide {
    /// The order side required to flatten this position.
    pub const fn exit_order_side(self) -> OrderSide {
        match self {
            Self::Long => OrderSide::Sell,
            Self::Short => OrderSide::Buy,
        }
    }
}

/// Direction of the broker-owned exit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// The protection leg that generated an exit or was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionLeg {
    TakeProfit,
    StopLoss,
    TrailingStop,
}

/// Which exit wins when the same quote satisfies both profit and stop levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitPrecedence {
    /// Prefer the take-profit leg.  This is useful when the replay fixture
    /// intentionally models a favorable gap through both levels.
    TakeProfitFirst,
    /// Prefer the protective stop.  This is the conservative default for an
    /// ambiguous quote.
    StopFirst,
}

/// A position for which a bracket is armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub side: PositionSide,
    /// Always positive.  A short position is represented by `side`, not a
    /// negative quantity.
    pub quantity: u64,
    pub entry_price_ticks: PriceTicks,
}

impl Position {
    pub const fn new(
        side: PositionSide,
        quantity: u64,
        entry_price_ticks: PriceTicks,
    ) -> Result<Self, ConfigError> {
        if quantity == 0 {
            return Err(ConfigError::ZeroQuantity);
        }
        Ok(Self {
            side,
            quantity,
            entry_price_ticks,
        })
    }
}

/// Optional trailing-stop configuration, in ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrailingConfig {
    /// Favorable movement from entry required before the static stop is
    /// replaced by the trailing stop.
    pub activation_ticks: u64,
    /// Distance from the favorable executable quote to the trailing stop.
    pub offset_ticks: u64,
    /// Minimum tightening increment. Tradovate calls this `freq`; one tick is
    /// the most precise setting and is the default chosen by the adapter when
    /// the wire payload omits it.
    pub frequency_ticks: u64,
}

/// Bracket parameters. `None` disables that leg; at least one leg must be
/// enabled. Zero-valued offsets are rejected instead of being interpreted
/// ambiguously as either an immediate order or a disabled order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtectionParams {
    pub take_profit_ticks: Option<u64>,
    pub stop_loss_ticks: Option<u64>,
    pub trailing: Option<TrailingConfig>,
    pub exit_precedence: ExitPrecedence,
}

/// A quote observed at one deterministic replay point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketTick {
    pub sequence: TickSequence,
    pub bid_ticks: PriceTicks,
    pub ask_ticks: PriceTicks,
}

impl MarketTick {
    pub const fn new(sequence: TickSequence, bid_ticks: PriceTicks, ask_ticks: PriceTicks) -> Self {
        Self {
            sequence,
            bid_ticks,
            ask_ticks,
        }
    }
}

/// The broker-owned exit order requested by a triggered bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillIntent {
    pub leg: ProtectionLeg,
    pub side: OrderSide,
    pub quantity: u64,
    /// The executable side of the quote at the trigger sequence.
    pub fill_price_ticks: PriceTicks,
    pub trigger_sequence: TickSequence,
}

/// Events emitted in deterministic order while processing a quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionEvent {
    /// The static stop was replaced by a trailing stop.
    TrailingActivated {
        previous_stop_ticks: Option<PriceTicks>,
        new_stop_ticks: PriceTicks,
    },
    /// The active trailing stop was amended in the favorable direction.
    TrailingRatchet {
        previous_stop_ticks: PriceTicks,
        new_stop_ticks: PriceTicks,
    },
    /// A broker-owned marketable exit should be filled at the executable
    /// quote and the position flattened by the parent.
    FillIntent(FillIntent),
    /// The sibling leg was cancelled as a consequence of the exit intent.
    SiblingCancelled { leg: ProtectionLeg },
}

/// Whether the bracket is still able to emit a fill intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketStatus {
    Armed,
    ExitTriggered {
        leg: ProtectionLeg,
        trigger_sequence: TickSequence,
    },
}

/// Configuration errors caught before any bracket is armed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    ZeroQuantity,
    NoProtectionLeg,
    ZeroOffset(&'static str),
    PriceOverflow(&'static str),
}

/// Errors in the input market stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickError {
    CrossedQuote {
        bid_ticks: PriceTicks,
        ask_ticks: PriceTicks,
    },
    OutOfOrder {
        previous_sequence: TickSequence,
        received_sequence: TickSequence,
    },
}

/// The pure state machine for one broker-owned bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtectionBracket {
    position: Position,
    params: ProtectionParams,
    take_profit_ticks: Option<PriceTicks>,
    static_stop_ticks: Option<PriceTicks>,
    trailing_stop_ticks: Option<PriceTicks>,
    last_sequence: Option<TickSequence>,
    status: BracketStatus,
}

impl ProtectionBracket {
    /// Arms a bracket around one already-filled position.
    pub fn new(position: Position, params: ProtectionParams) -> Result<Self, ConfigError> {
        if params.take_profit_ticks.is_none()
            && params.stop_loss_ticks.is_none()
            && params.trailing.is_none()
        {
            return Err(ConfigError::NoProtectionLeg);
        }

        validate_offset(params.take_profit_ticks, "take_profit_ticks")?;
        validate_offset(params.stop_loss_ticks, "stop_loss_ticks")?;
        if let Some(trailing) = params.trailing {
            if trailing.activation_ticks == 0 {
                return Err(ConfigError::ZeroOffset("trailing.activation_ticks"));
            }
            if trailing.offset_ticks == 0 {
                return Err(ConfigError::ZeroOffset("trailing.offset_ticks"));
            }
            if trailing.frequency_ticks == 0 {
                return Err(ConfigError::ZeroOffset("trailing.frequency_ticks"));
            }
            if i64::try_from(trailing.activation_ticks).is_err() {
                return Err(ConfigError::PriceOverflow("trailing.activation_ticks"));
            }
            if i64::try_from(trailing.offset_ticks).is_err() {
                return Err(ConfigError::PriceOverflow("trailing.offset_ticks"));
            }
            if i64::try_from(trailing.frequency_ticks).is_err() {
                return Err(ConfigError::PriceOverflow("trailing.frequency_ticks"));
            }
        }

        let take_profit_ticks = match params.take_profit_ticks {
            Some(offset) => Some(offset_price(
                position.side,
                position.entry_price_ticks,
                offset,
                ProtectionLeg::TakeProfit,
            )?),
            None => None,
        };
        let static_stop_ticks = match params.stop_loss_ticks {
            Some(offset) => Some(offset_price(
                position.side,
                position.entry_price_ticks,
                offset,
                ProtectionLeg::StopLoss,
            )?),
            None => None,
        };

        Ok(Self {
            position,
            params,
            take_profit_ticks,
            static_stop_ticks,
            trailing_stop_ticks: None,
            last_sequence: None,
            status: BracketStatus::Armed,
        })
    }

    pub const fn position(&self) -> Position {
        self.position
    }

    pub const fn status(&self) -> BracketStatus {
        self.status
    }

    pub const fn take_profit_ticks(&self) -> Option<PriceTicks> {
        self.take_profit_ticks
    }

    /// Returns the currently active stop level. After trailing activation,
    /// this is the trailing level and the static stop is no longer active.
    pub const fn active_stop_ticks(&self) -> Option<PriceTicks> {
        match self.trailing_stop_ticks {
            Some(stop) => Some(stop),
            None => self.static_stop_ticks,
        }
    }

    /// Process exactly one quote. Equal sequence numbers are idempotent and
    /// produce no events; lower sequence numbers are rejected. The state is
    /// changed before a fill intent is returned, preventing duplicate exits
    /// if a parent adapter retries delivery of the event.
    pub fn on_tick(&mut self, tick: MarketTick) -> Result<Vec<ProtectionEvent>, TickError> {
        if tick.bid_ticks > tick.ask_ticks {
            return Err(TickError::CrossedQuote {
                bid_ticks: tick.bid_ticks,
                ask_ticks: tick.ask_ticks,
            });
        }
        if let Some(previous_sequence) = self.last_sequence {
            if tick.sequence < previous_sequence {
                return Err(TickError::OutOfOrder {
                    previous_sequence,
                    received_sequence: tick.sequence,
                });
            }
            if tick.sequence == previous_sequence {
                return Ok(Vec::new());
            }
        }
        self.last_sequence = Some(tick.sequence);

        if self.status != BracketStatus::Armed {
            return Ok(Vec::new());
        }

        let executable_price = match self.position.side {
            PositionSide::Long => tick.bid_ticks,
            PositionSide::Short => tick.ask_ticks,
        };
        let mut events = Vec::new();
        self.update_trailing(executable_price, &mut events);

        let take_profit_hit =
            self.take_profit_ticks
                .is_some_and(|level| match self.position.side {
                    PositionSide::Long => executable_price >= level,
                    PositionSide::Short => executable_price <= level,
                });
        let stop_level = self.active_stop_ticks();
        let stop_hit = stop_level.is_some_and(|level| match self.position.side {
            PositionSide::Long => executable_price <= level,
            PositionSide::Short => executable_price >= level,
        });

        let stop_leg = self
            .trailing_stop_ticks
            .map(|_| ProtectionLeg::TrailingStop)
            .or_else(|| self.static_stop_ticks.map(|_| ProtectionLeg::StopLoss));
        let selected_leg = select_exit_leg(
            take_profit_hit,
            stop_hit,
            stop_leg,
            self.params.exit_precedence,
        );

        let Some(leg) = selected_leg else {
            return Ok(events);
        };

        // Terminal before emitting anything: an adapter that processes the
        // returned vector asynchronously cannot cause a second exit.
        self.status = BracketStatus::ExitTriggered {
            leg,
            trigger_sequence: tick.sequence,
        };

        events.push(ProtectionEvent::FillIntent(FillIntent {
            leg,
            side: self.position.side.exit_order_side(),
            quantity: self.position.quantity,
            fill_price_ticks: executable_price,
            trigger_sequence: tick.sequence,
        }));

        if leg != ProtectionLeg::TakeProfit && self.take_profit_ticks.is_some() {
            events.push(ProtectionEvent::SiblingCancelled {
                leg: ProtectionLeg::TakeProfit,
            });
        } else if leg == ProtectionLeg::TakeProfit && stop_leg.is_some() {
            events.push(ProtectionEvent::SiblingCancelled {
                leg: stop_leg.expect("checked above"),
            });
        }

        Ok(events)
    }

    fn update_trailing(&mut self, executable_price: PriceTicks, events: &mut Vec<ProtectionEvent>) {
        let Some(trailing) = self.params.trailing else {
            return;
        };

        let favorable = match self.position.side {
            PositionSide::Long => executable_price.saturating_sub(self.position.entry_price_ticks),
            PositionSide::Short => self
                .position
                .entry_price_ticks
                .saturating_sub(executable_price),
        };
        let activation = i64::try_from(trailing.activation_ticks).unwrap_or(i64::MAX);
        if self.trailing_stop_ticks.is_none() && favorable >= activation {
            let Ok(candidate) =
                trailing_price(self.position.side, executable_price, trailing.offset_ticks)
            else {
                return;
            };
            let previous_stop = self.static_stop_ticks;
            let new_stop = previous_stop
                .map(|previous| {
                    ratchet_price(
                        self.position.side,
                        previous,
                        candidate,
                        trailing.frequency_ticks,
                    )
                })
                .unwrap_or(candidate);
            self.trailing_stop_ticks = Some(new_stop);
            events.push(ProtectionEvent::TrailingActivated {
                previous_stop_ticks: previous_stop,
                new_stop_ticks: new_stop,
            });
            return;
        }

        let Some(previous_stop) = self.trailing_stop_ticks else {
            return;
        };
        let Ok(candidate) =
            trailing_price(self.position.side, executable_price, trailing.offset_ticks)
        else {
            return;
        };
        let candidate = ratchet_price(
            self.position.side,
            previous_stop,
            candidate,
            trailing.frequency_ticks,
        );
        let improved = match self.position.side {
            PositionSide::Long => candidate > previous_stop,
            PositionSide::Short => candidate < previous_stop,
        };
        if improved {
            self.trailing_stop_ticks = Some(candidate);
            events.push(ProtectionEvent::TrailingRatchet {
                previous_stop_ticks: previous_stop,
                new_stop_ticks: candidate,
            });
        }
    }
}

fn ratchet_price(
    side: PositionSide,
    current_stop: PriceTicks,
    candidate: PriceTicks,
    frequency_ticks: u64,
) -> PriceTicks {
    let Ok(frequency) = i64::try_from(frequency_ticks) else {
        return current_stop;
    };
    if frequency <= 0 {
        return current_stop;
    }
    match side {
        PositionSide::Long => {
            let distance = candidate.saturating_sub(current_stop);
            if distance < 0 {
                current_stop
            } else {
                current_stop.saturating_add((distance / frequency).saturating_mul(frequency))
            }
        }
        PositionSide::Short => {
            let distance = current_stop.saturating_sub(candidate);
            if distance < 0 {
                current_stop
            } else {
                current_stop.saturating_sub((distance / frequency).saturating_mul(frequency))
            }
        }
    }
}

fn validate_offset(offset: Option<u64>, name: &'static str) -> Result<(), ConfigError> {
    if offset == Some(0) {
        Err(ConfigError::ZeroOffset(name))
    } else {
        Ok(())
    }
}

fn select_exit_leg(
    take_profit_hit: bool,
    stop_hit: bool,
    stop_leg: Option<ProtectionLeg>,
    precedence: ExitPrecedence,
) -> Option<ProtectionLeg> {
    match (take_profit_hit, stop_hit, precedence) {
        (true, true, ExitPrecedence::TakeProfitFirst) => Some(ProtectionLeg::TakeProfit),
        (true, true, ExitPrecedence::StopFirst) => stop_leg,
        (true, false, _) => Some(ProtectionLeg::TakeProfit),
        (false, true, _) => stop_leg,
        (false, false, _) => None,
    }
}

fn offset_price(
    side: PositionSide,
    entry: PriceTicks,
    offset: u64,
    leg: ProtectionLeg,
) -> Result<PriceTicks, ConfigError> {
    let offset = i64::try_from(offset).map_err(|_| ConfigError::PriceOverflow(leg_name(leg)))?;
    match (side, leg) {
        (PositionSide::Long, ProtectionLeg::TakeProfit)
        | (PositionSide::Short, ProtectionLeg::StopLoss) => entry
            .checked_add(offset)
            .ok_or(ConfigError::PriceOverflow(leg_name(leg))),
        (PositionSide::Long, ProtectionLeg::StopLoss)
        | (PositionSide::Short, ProtectionLeg::TakeProfit) => entry
            .checked_sub(offset)
            .ok_or(ConfigError::PriceOverflow(leg_name(leg))),
        (_, ProtectionLeg::TrailingStop) => Err(ConfigError::PriceOverflow("trailing")),
    }
}

fn trailing_price(
    side: PositionSide,
    executable_price: PriceTicks,
    offset: u64,
) -> Result<PriceTicks, ConfigError> {
    let offset = i64::try_from(offset).map_err(|_| ConfigError::PriceOverflow("trailing"))?;
    match side {
        PositionSide::Long => executable_price
            .checked_sub(offset)
            .ok_or(ConfigError::PriceOverflow("trailing")),
        PositionSide::Short => executable_price
            .checked_add(offset)
            .ok_or(ConfigError::PriceOverflow("trailing")),
    }
}

const fn leg_name(leg: ProtectionLeg) -> &'static str {
    match leg {
        ProtectionLeg::TakeProfit => "take_profit_ticks",
        ProtectionLeg::StopLoss => "stop_loss_ticks",
        ProtectionLeg::TrailingStop => "trailing",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(side: PositionSide, quantity: u64, entry: PriceTicks) -> Position {
        Position::new(side, quantity, entry).expect("valid position")
    }

    fn params(tp: Option<u64>, sl: Option<u64>) -> ProtectionParams {
        ProtectionParams {
            take_profit_ticks: tp,
            stop_loss_ticks: sl,
            trailing: None,
            exit_precedence: ExitPrecedence::StopFirst,
        }
    }

    fn fill(events: &[ProtectionEvent]) -> FillIntent {
        events
            .iter()
            .find_map(|event| match event {
                ProtectionEvent::FillIntent(intent) => Some(*intent),
                _ => None,
            })
            .expect("fill intent")
    }

    #[test]
    fn long_tp_uses_bid_and_cancels_stop() {
        let mut bracket = ProtectionBracket::new(
            position(PositionSide::Long, 7, 100),
            params(Some(10), Some(5)),
        )
        .unwrap();

        let events = bracket.on_tick(MarketTick::new(1, 110, 111)).unwrap();
        assert_eq!(
            fill(&events),
            FillIntent {
                leg: ProtectionLeg::TakeProfit,
                side: OrderSide::Sell,
                quantity: 7,
                fill_price_ticks: 110,
                trigger_sequence: 1,
            }
        );
        assert!(events.contains(&ProtectionEvent::SiblingCancelled {
            leg: ProtectionLeg::StopLoss
        }));
        assert!(matches!(
            bracket.status(),
            BracketStatus::ExitTriggered { .. }
        ));
    }

    #[test]
    fn short_stop_uses_ask_and_cancels_profit() {
        let mut bracket = ProtectionBracket::new(
            position(PositionSide::Short, 3, 100),
            params(Some(8), Some(4)),
        )
        .unwrap();

        let events = bracket.on_tick(MarketTick::new(1, 103, 104)).unwrap();
        let intent = fill(&events);
        assert_eq!(intent.leg, ProtectionLeg::StopLoss);
        assert_eq!(intent.side, OrderSide::Buy);
        assert_eq!(intent.quantity, 3);
        assert_eq!(intent.fill_price_ticks, 104);
        assert!(events.contains(&ProtectionEvent::SiblingCancelled {
            leg: ProtectionLeg::TakeProfit
        }));
    }

    #[test]
    fn stop_precedence_wins_when_trail_overlaps_tp() {
        // Valid positive TP/SL levels cannot normally overlap: a quote high
        // enough to ratchet a trail through TP would trigger TP first. Keep
        // the shared selector tested anyway so any future trigger source
        // (for example an exchange-reported ambiguous touch) remains explicit.
        assert_eq!(
            select_exit_leg(
                true,
                true,
                Some(ProtectionLeg::TrailingStop),
                ExitPrecedence::StopFirst,
            ),
            Some(ProtectionLeg::TrailingStop)
        );
    }

    #[test]
    fn take_profit_precedence_is_explicit_for_overlap() {
        assert_eq!(
            select_exit_leg(
                true,
                true,
                Some(ProtectionLeg::StopLoss),
                ExitPrecedence::TakeProfitFirst,
            ),
            Some(ProtectionLeg::TakeProfit)
        );
    }

    #[test]
    fn trailing_activates_and_only_ratchets_favorably() {
        let mut bracket = ProtectionBracket::new(
            position(PositionSide::Long, 2, 100),
            ProtectionParams {
                take_profit_ticks: None,
                stop_loss_ticks: Some(5),
                trailing: Some(TrailingConfig {
                    activation_ticks: 10,
                    offset_ticks: 3,
                    frequency_ticks: 1,
                }),
                exit_precedence: ExitPrecedence::StopFirst,
            },
        )
        .unwrap();

        let activated = bracket.on_tick(MarketTick::new(1, 110, 111)).unwrap();
        assert!(activated.contains(&ProtectionEvent::TrailingActivated {
            previous_stop_ticks: Some(95),
            new_stop_ticks: 107,
        }));
        assert_eq!(bracket.active_stop_ticks(), Some(107));

        let unchanged = bracket.on_tick(MarketTick::new(2, 108, 109)).unwrap();
        assert!(unchanged.is_empty());
        assert_eq!(bracket.active_stop_ticks(), Some(107));

        let ratcheted = bracket.on_tick(MarketTick::new(3, 115, 116)).unwrap();
        assert!(ratcheted.contains(&ProtectionEvent::TrailingRatchet {
            previous_stop_ticks: 107,
            new_stop_ticks: 112,
        }));
        assert_eq!(bracket.active_stop_ticks(), Some(112));
    }

    #[test]
    fn short_trailing_uses_ask_and_fires_at_ask() {
        let mut bracket = ProtectionBracket::new(
            position(PositionSide::Short, 4, 100),
            ProtectionParams {
                take_profit_ticks: None,
                stop_loss_ticks: Some(8),
                trailing: Some(TrailingConfig {
                    activation_ticks: 5,
                    offset_ticks: 2,
                    frequency_ticks: 1,
                }),
                exit_precedence: ExitPrecedence::StopFirst,
            },
        )
        .unwrap();

        bracket.on_tick(MarketTick::new(1, 94, 95)).unwrap();
        assert_eq!(bracket.active_stop_ticks(), Some(97));
        let events = bracket.on_tick(MarketTick::new(2, 97, 98)).unwrap();
        let intent = fill(&events);
        assert_eq!(intent.leg, ProtectionLeg::TrailingStop);
        assert_eq!(intent.fill_price_ticks, 98);
        assert_eq!(intent.side, OrderSide::Buy);
    }

    #[test]
    fn duplicate_is_idempotent_and_terminal_bracket_cannot_double_fill() {
        let mut bracket = ProtectionBracket::new(
            position(PositionSide::Long, 1, 100),
            params(Some(5), Some(5)),
        )
        .unwrap();
        let first = bracket.on_tick(MarketTick::new(9, 105, 106)).unwrap();
        assert!(!first.is_empty());
        assert!(
            bracket
                .on_tick(MarketTick::new(9, 105, 106))
                .unwrap()
                .is_empty()
        );
        assert!(
            bracket
                .on_tick(MarketTick::new(10, 90, 91))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn out_of_order_and_crossed_quotes_are_rejected_without_mutation() {
        let mut bracket =
            ProtectionBracket::new(position(PositionSide::Long, 1, 100), params(None, Some(5)))
                .unwrap();
        bracket.on_tick(MarketTick::new(2, 101, 102)).unwrap();
        assert_eq!(
            bracket.on_tick(MarketTick::new(1, 90, 91)),
            Err(TickError::OutOfOrder {
                previous_sequence: 2,
                received_sequence: 1,
            })
        );
        assert_eq!(
            bracket.on_tick(MarketTick::new(3, 103, 102)),
            Err(TickError::CrossedQuote {
                bid_ticks: 103,
                ask_ticks: 102,
            })
        );
        assert_eq!(bracket.status(), BracketStatus::Armed);
    }

    #[test]
    fn zero_quantity_zero_offset_and_no_legs_are_rejected() {
        assert_eq!(
            Position::new(PositionSide::Long, 0, 100),
            Err(ConfigError::ZeroQuantity)
        );
        assert_eq!(
            ProtectionBracket::new(position(PositionSide::Long, 1, 100), params(Some(0), None)),
            Err(ConfigError::ZeroOffset("take_profit_ticks"))
        );
        assert_eq!(
            ProtectionBracket::new(
                position(PositionSide::Long, 1, 100),
                ProtectionParams {
                    take_profit_ticks: None,
                    stop_loss_ticks: None,
                    trailing: None,
                    exit_precedence: ExitPrecedence::StopFirst,
                }
            ),
            Err(ConfigError::NoProtectionLeg)
        );
    }

    #[test]
    fn price_overflow_is_rejected_before_arming() {
        assert_eq!(
            ProtectionBracket::new(
                position(PositionSide::Long, 1, i64::MAX),
                params(Some(1), None)
            ),
            Err(ConfigError::PriceOverflow("take_profit_ticks"))
        );
        assert_eq!(
            ProtectionBracket::new(
                position(PositionSide::Short, 1, i64::MIN),
                params(Some(1), None)
            ),
            Err(ConfigError::PriceOverflow("take_profit_ticks"))
        );
    }
}

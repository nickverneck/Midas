//! Replay-only, causal orientation scheduler for EMA crossover research.
//!
//! This is intentionally **not** wired into the live strategy dispatcher.  A
//! replay sweep builds a schedule from raw EMA crosses, then the prepared
//! replay kernel consumes that immutable schedule.  The score at a cross uses
//! only the preceding cross's *already observed* close-to-close excursion;
//! it never reads the next bar's fill or the current cross's future outcome.

use crate::broker::Bar;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// The orientation selected for a raw EMA cross.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarkovOrientationState {
    Normal,
    Inverted,
    /// Insufficient evidence or a context veto.  The configured neutral
    /// action determines whether this means normal fallback or no entry.
    Neutral,
}

impl Default for MarkovOrientationState {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MarkovNeutralAction {
    /// Keep the established raw/normal strategy active while reporting the
    /// selector as neutral.  This is the conservative research default.
    NormalFallback,
    /// Suppress an entry while neutral.  Useful only as an explicit replay
    /// comparison because it changes trade count.
    Abstain,
}

impl Default for MarkovNeutralAction {
    fn default() -> Self {
        Self::NormalFallback
    }
}

/// Configuration for the replay-only Markov orientation scheduler.
///
/// A "shadow outcome" is the prior raw cross's signed close-to-close return
/// measured when the next raw cross closes.  It is a conservative causal mark,
/// not a broker fill and not an oracle daily label.  Normal and inverted lanes
/// receive the same observation with opposite signs, then a rolling score,
/// confirmation and dwell rule selects an orientation for the *next* entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ReplayMarkovOrientationGateConfig {
    /// Strict opt-in. Disabled is a no-op and leaves all existing sweeps and
    /// every live execution path unchanged.
    pub enabled: bool,
    /// Number of completed paired shadow outcomes kept per orientation.
    pub score_window_outcomes: usize,
    /// No orientation switch is proposed before this many completed outcomes.
    pub minimum_completed_outcomes: usize,
    /// Required normal-minus-inverted rolling-score separation, in ticks.
    pub score_margin_ticks: f64,
    /// Cap one marked excursion so one exceptional trend cannot make the
    /// selector chase an already-ended move.
    pub max_abs_outcome_ticks: f64,
    /// Consecutive identical proposals required before a state transition.
    pub confirmation_events: usize,
    /// Number of raw-cross decisions the current normal/inverted state must
    /// retain before it may switch again.
    pub minimum_dwell_events: usize,
    /// A gap larger than this explicitly clears scores and returns to normal.
    /// `None` disables gap resets.
    pub reset_after_gap_minutes: Option<u32>,
    /// If set, compute the preceding-bar Kaufman efficiency ratio over this
    /// many bars. A value below the cutoff moves the state to Neutral rather
    /// than treating chop as proof of inversion.
    pub efficiency_lookback_bars: Option<usize>,
    pub neutral_below_efficiency_ratio: Option<f64>,
    #[serde(default)]
    pub neutral_action: MarkovNeutralAction,
}

impl Default for ReplayMarkovOrientationGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            score_window_outcomes: 16,
            minimum_completed_outcomes: 5,
            score_margin_ticks: 300.0,
            max_abs_outcome_ticks: 1_200.0,
            confirmation_events: 3,
            minimum_dwell_events: 10,
            reset_after_gap_minutes: Some(120),
            efficiency_lookback_bars: Some(30),
            neutral_below_efficiency_ratio: Some(0.20),
            neutral_action: MarkovNeutralAction::NormalFallback,
        }
    }
}

impl ReplayMarkovOrientationGateConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.score_window_outcomes == 0 {
            return Err("Markov score_window_outcomes must be greater than zero".into());
        }
        if self.minimum_completed_outcomes == 0
            || self.minimum_completed_outcomes > self.score_window_outcomes
        {
            return Err(
                "Markov minimum_completed_outcomes must be in 1..=score_window_outcomes".into(),
            );
        }
        if self.confirmation_events == 0 {
            return Err("Markov confirmation_events must be greater than zero".into());
        }
        if !self.score_margin_ticks.is_finite() || self.score_margin_ticks < 0.0 {
            return Err("Markov score_margin_ticks must be finite and non-negative".into());
        }
        if !self.max_abs_outcome_ticks.is_finite() || self.max_abs_outcome_ticks <= 0.0 {
            return Err("Markov max_abs_outcome_ticks must be finite and greater than zero".into());
        }
        match (
            self.efficiency_lookback_bars,
            self.neutral_below_efficiency_ratio,
        ) {
            (None, None) => {}
            (Some(lookback), Some(cutoff))
                if lookback >= 2 && cutoff.is_finite() && (0.0..=1.0).contains(&cutoff) => {}
            _ => {
                return Err(
                    "Markov efficiency settings require lookback >= 2 and cutoff in [0, 1]".into(),
                );
            }
        }
        Ok(())
    }
}

/// One complete, auditable decision. `effective_inverted` is what a replay
/// kernel applies; it is false for neutral normal-fallback and `None` for a
/// neutral abstention.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovOrientationDecision {
    pub state: MarkovOrientationState,
    pub effective_inverted: Option<bool>,
    pub proposal: MarkovOrientationState,
    pub normal_score_ticks: f64,
    pub inverted_score_ticks: f64,
    pub completed_outcomes: usize,
    pub confirmation_count: usize,
    pub dwell_events: usize,
    pub efficiency_ratio: Option<f64>,
    pub reset_for_gap: bool,
    pub audit_reason: &'static str,
}

impl MarkovOrientationDecision {
    fn disabled() -> Self {
        Self {
            state: MarkovOrientationState::Normal,
            effective_inverted: Some(false),
            proposal: MarkovOrientationState::Normal,
            normal_score_ticks: 0.0,
            inverted_score_ticks: 0.0,
            completed_outcomes: 0,
            confirmation_count: 0,
            dwell_events: 0,
            efficiency_ratio: None,
            reset_for_gap: false,
            audit_reason: "markov_disabled",
        }
    }
}

/// Stateful causal scheduler. It is deliberately small and allocation-free
/// per decision except for the bounded outcome queue.
#[derive(Debug, Clone)]
pub struct ReplayMarkovOrientationGate {
    config: ReplayMarkovOrientationGateConfig,
    state: MarkovOrientationState,
    proposed: Option<MarkovOrientationState>,
    proposal_count: usize,
    dwell_events: usize,
    outcomes: VecDeque<f64>,
    previous_event: Option<(i64, f64, i8)>,
}

impl ReplayMarkovOrientationGate {
    pub fn new(config: ReplayMarkovOrientationGateConfig) -> Self {
        Self {
            config,
            state: MarkovOrientationState::Normal,
            proposed: None,
            proposal_count: 0,
            dwell_events: 0,
            outcomes: VecDeque::new(),
            previous_event: None,
        }
    }

    /// Observe a raw EMA cross at the current completed bar. `raw_direction`
    /// is +1 for raw buy and -1 for raw sell. All price/context data must end
    /// at this bar. The returned decision applies to this cross's next-bar
    /// entry and cannot be changed by future bars.
    pub fn observe_cross(
        &mut self,
        bars: &[Bar],
        raw_direction: i8,
        tick_size: f64,
    ) -> MarkovOrientationDecision {
        if !self.config.enabled {
            return MarkovOrientationDecision::disabled();
        }
        let Some(current) = bars.last() else {
            return self.neutral_decision(None, false, "markov_missing_bar");
        };
        if raw_direction != 1 && raw_direction != -1 || !tick_size.is_finite() || tick_size <= 0.0 {
            return self.neutral_decision(None, false, "markov_invalid_input");
        }

        let mut reset_for_gap = false;
        if let Some((previous_ts, previous_close, previous_direction)) = self.previous_event {
            let gap_exceeded = self.config.reset_after_gap_minutes.is_some_and(|minutes| {
                current.ts_ns.saturating_sub(previous_ts)
                    > (minutes as i64).saturating_mul(60 * 1_000_000_000)
            });
            if gap_exceeded {
                self.reset();
                reset_for_gap = true;
            } else if current.close.is_finite() && previous_close.is_finite() {
                // The prior signal's mark closes *at this already completed
                // bar*. This is intentionally before the current next-bar
                // execution and therefore has no same-event outcome leakage.
                let outcome =
                    previous_direction as f64 * (current.close - previous_close) / tick_size;
                if outcome.is_finite() {
                    self.outcomes.push_back(outcome.clamp(
                        -self.config.max_abs_outcome_ticks,
                        self.config.max_abs_outcome_ticks,
                    ));
                    while self.outcomes.len() > self.config.score_window_outcomes {
                        self.outcomes.pop_front();
                    }
                }
            }
        }

        let efficiency_ratio =
            efficiency_ratio_before_current(bars, self.config.efficiency_lookback_bars);
        let proposal = self.proposal(efficiency_ratio);
        self.apply_proposal(proposal);
        self.previous_event = Some((current.ts_ns, current.close, raw_direction));
        let (normal_score_ticks, inverted_score_ticks) = self.scores();
        let reason = if reset_for_gap {
            "markov_gap_reset"
        } else if matches!(proposal, MarkovOrientationState::Neutral) {
            "markov_neutral_context_or_warmup"
        } else if self.state != proposal {
            "markov_confirmation_or_dwell"
        } else {
            "markov_confirmed"
        };
        self.decision(
            proposal,
            normal_score_ticks,
            inverted_score_ticks,
            efficiency_ratio,
            reset_for_gap,
            reason,
        )
    }

    fn reset(&mut self) {
        self.state = MarkovOrientationState::Normal;
        self.proposed = None;
        self.proposal_count = 0;
        self.dwell_events = 0;
        self.outcomes.clear();
        self.previous_event = None;
    }

    fn scores(&self) -> (f64, f64) {
        let normal = self.outcomes.iter().copied().sum::<f64>();
        (normal, -normal)
    }

    fn proposal(&self, efficiency_ratio: Option<f64>) -> MarkovOrientationState {
        if self.outcomes.len() < self.config.minimum_completed_outcomes {
            return MarkovOrientationState::Neutral;
        }
        if self
            .config
            .neutral_below_efficiency_ratio
            .is_some_and(|cutoff| efficiency_ratio.is_none_or(|ratio| ratio < cutoff))
        {
            return MarkovOrientationState::Neutral;
        }
        let (normal, inverted) = self.scores();
        let difference = normal - inverted;
        if difference >= self.config.score_margin_ticks {
            MarkovOrientationState::Normal
        } else if difference <= -self.config.score_margin_ticks {
            MarkovOrientationState::Inverted
        } else {
            MarkovOrientationState::Neutral
        }
    }

    fn apply_proposal(&mut self, proposal: MarkovOrientationState) {
        if self.proposed == Some(proposal) {
            self.proposal_count = self.proposal_count.saturating_add(1);
        } else {
            self.proposed = Some(proposal);
            self.proposal_count = 1;
        }

        if proposal == MarkovOrientationState::Neutral {
            self.state = MarkovOrientationState::Neutral;
            self.dwell_events = 0;
            return;
        }
        if self.state == proposal {
            self.dwell_events = self.dwell_events.saturating_add(1);
            return;
        }
        // Neutral is deliberately an uncertainty state, not a sticky third
        // orientation. Once enough causal evidence accumulates it may leave
        // Neutral after confirmation; dwell only protects a switch *between*
        // established normal and inverted states.
        let dwell_blocks = self.state != MarkovOrientationState::Neutral
            && self.dwell_events < self.config.minimum_dwell_events;
        if dwell_blocks || self.proposal_count < self.config.confirmation_events {
            self.dwell_events = self.dwell_events.saturating_add(1);
            return;
        }
        self.state = proposal;
        self.dwell_events = 0;
    }

    fn decision(
        &self,
        proposal: MarkovOrientationState,
        normal_score_ticks: f64,
        inverted_score_ticks: f64,
        efficiency_ratio: Option<f64>,
        reset_for_gap: bool,
        audit_reason: &'static str,
    ) -> MarkovOrientationDecision {
        let effective_inverted = match self.state {
            MarkovOrientationState::Normal => Some(false),
            MarkovOrientationState::Inverted => Some(true),
            MarkovOrientationState::Neutral => match self.config.neutral_action {
                MarkovNeutralAction::NormalFallback => Some(false),
                MarkovNeutralAction::Abstain => None,
            },
        };
        MarkovOrientationDecision {
            state: self.state,
            effective_inverted,
            proposal,
            normal_score_ticks,
            inverted_score_ticks,
            completed_outcomes: self.outcomes.len(),
            confirmation_count: self.proposal_count,
            dwell_events: self.dwell_events,
            efficiency_ratio,
            reset_for_gap,
            audit_reason,
        }
    }

    fn neutral_decision(
        &self,
        efficiency_ratio: Option<f64>,
        reset_for_gap: bool,
        reason: &'static str,
    ) -> MarkovOrientationDecision {
        let (normal, inverted) = self.scores();
        MarkovOrientationDecision {
            state: MarkovOrientationState::Neutral,
            effective_inverted: match self.config.neutral_action {
                MarkovNeutralAction::NormalFallback => Some(false),
                MarkovNeutralAction::Abstain => None,
            },
            proposal: MarkovOrientationState::Neutral,
            normal_score_ticks: normal,
            inverted_score_ticks: inverted,
            completed_outcomes: self.outcomes.len(),
            confirmation_count: self.proposal_count,
            dwell_events: self.dwell_events,
            efficiency_ratio,
            reset_for_gap,
            audit_reason: reason,
        }
    }
}

/// Kaufman efficiency ratio using only bars before the current decision bar.
fn efficiency_ratio_before_current(bars: &[Bar], lookback: Option<usize>) -> Option<f64> {
    let lookback = lookback?;
    if lookback < 2 || bars.len() <= lookback {
        return None;
    }
    let end = bars.len() - 1; // exclude current decision bar.
    let start = end.checked_sub(lookback)?;
    let closes = bars
        .get(start..=end)?
        .iter()
        .map(|bar| bar.close)
        .collect::<Vec<_>>();
    if closes.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let path = closes
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .sum::<f64>();
    (path > f64::EPSILON).then_some((closes.last()? - closes.first()?).abs() / path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(ts_ns: i64, close: f64) -> Bar {
        Bar {
            ts_ns,
            open: close,
            high: close,
            low: close,
            close,
            volume: Some(1.0),
        }
    }

    fn config() -> ReplayMarkovOrientationGateConfig {
        ReplayMarkovOrientationGateConfig {
            enabled: true,
            score_window_outcomes: 3,
            minimum_completed_outcomes: 1,
            score_margin_ticks: 0.5,
            max_abs_outcome_ticks: 100.0,
            confirmation_events: 1,
            minimum_dwell_events: 0,
            reset_after_gap_minutes: None,
            efficiency_lookback_bars: None,
            neutral_below_efficiency_ratio: None,
            neutral_action: MarkovNeutralAction::NormalFallback,
        }
    }

    #[test]
    fn disabled_is_strict_normal_noop() {
        let mut gate =
            ReplayMarkovOrientationGate::new(ReplayMarkovOrientationGateConfig::default());
        let decision = gate.observe_cross(&[bar(0, 100.0)], 1, 1.0);
        assert_eq!(decision.state, MarkovOrientationState::Normal);
        assert_eq!(decision.effective_inverted, Some(false));
        assert_eq!(decision.audit_reason, "markov_disabled");
    }

    #[test]
    fn only_completed_prior_cross_can_change_orientation() {
        let mut gate = ReplayMarkovOrientationGate::new(config());
        let first = gate.observe_cross(&[bar(0, 100.0)], 1, 1.0);
        assert_eq!(first.state, MarkovOrientationState::Neutral);
        // The prior raw-long outcome is negative at this closed bar, so
        // inverted wins. No future bar is needed or read.
        let second = gate.observe_cross(&[bar(0, 100.0), bar(60, 98.0)], -1, 1.0);
        assert_eq!(second.state, MarkovOrientationState::Inverted);
        assert_eq!(second.effective_inverted, Some(true));
        assert_eq!(second.completed_outcomes, 1);
    }

    #[test]
    fn prefix_decisions_do_not_change_when_future_bars_are_appended() {
        let bars = vec![
            bar(0, 100.0),
            bar(60, 98.0),
            bar(120, 97.0),
            bar(180, 101.0),
        ];
        let mut prefix_gate = ReplayMarkovOrientationGate::new(config());
        let first = prefix_gate.observe_cross(&bars[..1], 1, 1.0);
        let second = prefix_gate.observe_cross(&bars[..2], -1, 1.0);

        let mut replay_gate = ReplayMarkovOrientationGate::new(config());
        let first_again = replay_gate.observe_cross(&bars[..1], 1, 1.0);
        let second_again = replay_gate.observe_cross(&bars[..2], -1, 1.0);
        assert_eq!(first, first_again);
        assert_eq!(second, second_again);
    }

    #[test]
    fn gap_reset_discards_outcomes_and_returns_neutral_fallback() {
        let mut config = config();
        config.reset_after_gap_minutes = Some(1);
        let mut gate = ReplayMarkovOrientationGate::new(config);
        let _ = gate.observe_cross(&[bar(0, 100.0)], 1, 1.0);
        let decision = gate.observe_cross(&[bar(0, 100.0), bar(121_000_000_000, 90.0)], -1, 1.0);
        assert!(decision.reset_for_gap);
        assert_eq!(decision.completed_outcomes, 0);
        assert_eq!(decision.state, MarkovOrientationState::Neutral);
        assert_eq!(decision.effective_inverted, Some(false));
    }

    #[test]
    fn validation_rejects_invalid_settings() {
        let mut invalid = config();
        invalid.minimum_completed_outcomes = 4;
        assert!(invalid.validate().is_err());
        invalid.minimum_completed_outcomes = 1;
        invalid.efficiency_lookback_bars = Some(1);
        invalid.neutral_below_efficiency_ratio = Some(0.2);
        assert!(invalid.validate().is_err());
    }
}

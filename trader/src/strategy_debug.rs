#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StrategyDecisionDebug {
    pub strategy: Option<String>,
    pub broker: Option<String>,
    pub path: Option<String>,
    pub decision: Option<String>,
    pub timing: Option<String>,
    pub signal_delay_bars: Option<String>,
    pub bar_ts: Option<String>,
    pub bar_index: Option<String>,
    pub bar_count: Option<String>,
    pub fingerprint: Option<String>,
    pub actual_qty: Option<String>,
    pub effective_qty: Option<String>,
    pub pending_target: Option<String>,
    pub target_qty: Option<String>,
    pub signal: Option<String>,
    pub reason: Option<String>,
    pub strategy_detail: Option<String>,
    pub gate_detail: Option<String>,
}

impl StrategyDecisionDebug {
    pub fn format(&self) -> String {
        let mut parts = vec!["strategy decision".to_string()];
        push_field(&mut parts, "strategy", self.strategy.as_deref());
        push_field(&mut parts, "broker", self.broker.as_deref());
        push_field(&mut parts, "path", self.path.as_deref());
        push_field(&mut parts, "decision", self.decision.as_deref());
        push_field(&mut parts, "timing", self.timing.as_deref());
        push_field(
            &mut parts,
            "signal_delay_bars",
            self.signal_delay_bars.as_deref(),
        );
        push_field(&mut parts, "bar_ts", self.bar_ts.as_deref());
        push_field(&mut parts, "bar_index", self.bar_index.as_deref());
        push_field(&mut parts, "bar_count", self.bar_count.as_deref());
        push_field(&mut parts, "fingerprint", self.fingerprint.as_deref());
        push_field(&mut parts, "actual_qty", self.actual_qty.as_deref());
        push_field(&mut parts, "effective_qty", self.effective_qty.as_deref());
        push_field(&mut parts, "pending_target", self.pending_target.as_deref());
        push_field(&mut parts, "target_qty", self.target_qty.as_deref());
        push_field(&mut parts, "signal", self.signal.as_deref());
        push_field(&mut parts, "reason", self.reason.as_deref());
        push_field(
            &mut parts,
            "strategy_detail",
            self.strategy_detail.as_deref(),
        );
        push_field(&mut parts, "gate_detail", self.gate_detail.as_deref());
        parts.join(" | ")
    }
}

pub fn format_strategy_decision(payload: &StrategyDecisionDebug) -> String {
    payload.format()
}

fn push_field(parts: &mut Vec<String>, name: &str, value: Option<&str>) {
    parts.push(format!("{name}={}", value.unwrap_or("n/a")));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatter_uses_stable_prefix_and_field_order() {
        let line = format_strategy_decision(&StrategyDecisionDebug {
            strategy: Some("ema_cross".to_string()),
            broker: Some("tradovate".to_string()),
            path: Some("guarded".to_string()),
            decision: Some("dispatching".to_string()),
            timing: Some("closed bar".to_string()),
            signal_delay_bars: Some("1".to_string()),
            bar_ts: Some("42".to_string()),
            bar_index: Some("7".to_string()),
            bar_count: Some("9".to_string()),
            fingerprint: Some("Some(99)".to_string()),
            actual_qty: Some("0".to_string()),
            effective_qty: Some("0".to_string()),
            pending_target: Some("none".to_string()),
            target_qty: Some("1".to_string()),
            signal: Some("Buy".to_string()),
            reason: Some("signal_ready".to_string()),
            strategy_detail: Some("delta=-1.00->1.00".to_string()),
            gate_detail: Some("submit_in_flight=false".to_string()),
        });

        assert!(line.starts_with("strategy decision |"));
        assert!(line.contains("| strategy=ema_cross | broker=tradovate | path=guarded |"));
        assert!(line.contains("| signal_delay_bars=1 | bar_ts=42 |"));
        assert!(line.contains("| strategy_detail=delta=-1.00->1.00 |"));
    }

    #[test]
    fn formatter_renders_missing_values_as_na() {
        let line = format_strategy_decision(&StrategyDecisionDebug {
            strategy: Some("hma_angle".to_string()),
            ..StrategyDecisionDebug::default()
        });

        assert_eq!(
            line,
            "strategy decision | strategy=hma_angle | broker=n/a | path=n/a | decision=n/a | timing=n/a | signal_delay_bars=n/a | bar_ts=n/a | bar_index=n/a | bar_count=n/a | fingerprint=n/a | actual_qty=n/a | effective_qty=n/a | pending_target=n/a | target_qty=n/a | signal=n/a | reason=n/a | strategy_detail=n/a | gate_detail=n/a"
        );
    }
}

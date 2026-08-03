use super::super::*;

impl App {
    pub(in crate::app) fn replay_build_available(&self) -> bool {
        cfg!(feature = "replay")
            && self
                .available_brokers
                .iter()
                .any(|broker| *broker == BrokerKind::Tradovate)
    }

    pub(in crate::app) fn replay_navigation_active(&self) -> bool {
        self.replay_build_available() && self.session_mode == EngineCreateMode::Replay
    }

    pub(in crate::app) fn broker_supports_bar_type_selection(&self) -> bool {
        self.selected_broker == BrokerKind::Tradovate
    }

    pub(in crate::app) fn broker_supports_heikin_ashi(&self) -> bool {
        self.selected_broker == BrokerKind::Tradovate
    }

    pub(in crate::app) fn replay_affordance_visible(&self) -> bool {
        self.replay_build_available()
    }

    pub(in crate::app) fn analytics_affordance_visible(&self) -> bool {
        self.replay_navigation_active()
    }

    pub(in crate::app) fn session_stats_affordance_visible(&self) -> bool {
        self.session_stats.enabled && !self.replay_navigation_active()
    }

    pub(in crate::app) fn manual_order_affordance_visible(&self) -> bool {
        cfg!(feature = "manual-orders")
            && !self.replay_navigation_active()
            && self.capabilities.manual_orders
    }

    pub(in crate::app) fn automated_strategy_affordance_visible(&self) -> bool {
        self.capabilities.automated_orders
    }

    pub(in crate::app) fn engine_create_affordance_visible(&self) -> bool {
        self.engine_creation_enabled
    }

    pub(in crate::app) fn engine_close_and_kill_affordance_visible(&self) -> bool {
        cfg!(feature = "manual-orders")
    }

    pub(in crate::app) fn engine_select_item_count(&self) -> usize {
        self.engine_summaries.len() + usize::from(self.engine_create_affordance_visible())
    }

    pub(in crate::app) fn clamp_selected_engine(&mut self) {
        let item_count = self.engine_select_item_count();
        if item_count == 0 {
            self.selected_engine = 0;
        } else if self.selected_engine >= item_count {
            self.selected_engine = item_count - 1;
        }
    }

    pub(in crate::app) fn bar_type_controls_visible(&self) -> bool {
        self.broker_supports_bar_type_selection()
    }

    pub(in crate::app) fn candle_mode_controls_visible(&self) -> bool {
        self.broker_supports_heikin_ashi() && self.bar_type.supports_candle_mode()
    }

    pub(in crate::app) fn visible_strategy_kinds(&self) -> &'static [StrategyKind] {
        &[StrategyKind::Native]
    }

    pub(in crate::app) fn next_visible_strategy_kind(&self) -> StrategyKind {
        let kinds = self.visible_strategy_kinds();
        let index = kinds
            .iter()
            .position(|kind| *kind == self.strategy.kind)
            .unwrap_or(0);
        kinds[(index + 1) % kinds.len()]
    }

    pub(in crate::app) fn prev_visible_strategy_kind(&self) -> StrategyKind {
        let kinds = self.visible_strategy_kinds();
        let index = kinds
            .iter()
            .position(|kind| *kind == self.strategy.kind)
            .unwrap_or(0);
        kinds[(index + kinds.len() - 1) % kinds.len()]
    }

    pub(in crate::app) fn normalize_market_controls_for_broker(&mut self) {
        if !self.broker_supports_bar_type_selection() {
            self.bar_type = BarType::default();
        }
        if !self.broker_supports_heikin_ashi() || !self.bar_type.supports_candle_mode() {
            self.candle_mode = CandleMode::Standard;
        }
    }
}

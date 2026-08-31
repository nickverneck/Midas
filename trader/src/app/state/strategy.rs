use super::super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::app) enum StrategyReadinessStatus {
    ReadyToArm,
    MonitorOnly,
    PreviewOnly,
    NeedsAttention,
}

impl StrategyReadinessStatus {
    pub(in crate::app) fn label(self) -> &'static str {
        match self {
            Self::ReadyToArm => "Ready to arm",
            Self::MonitorOnly => "Monitor only",
            Self::PreviewOnly => "Preview only",
            Self::NeedsAttention => "Needs attention",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::app) struct StrategyReadiness {
    pub(in crate::app) status: StrategyReadinessStatus,
    pub(in crate::app) blockers: Vec<String>,
    pub(in crate::app) warnings: Vec<String>,
    pub(in crate::app) adjustments: Vec<String>,
}

impl StrategyReadiness {
    pub(in crate::app) fn can_arm(&self) -> bool {
        self.status == StrategyReadinessStatus::ReadyToArm
    }
}

impl App {
    pub(in crate::app) fn clear_strategy_numeric_input(&mut self) {
        self.strategy_numeric_input = None;
    }

    pub(in crate::app) fn bar_value_text(&self) -> String {
        self.strategy_numeric_value(Focus::BarValue, self.bar_type.value().to_string())
    }

    pub(in crate::app) fn effective_candle_mode(&self) -> CandleMode {
        self.bar_type.effective_candle_mode(self.candle_mode)
    }

    pub(in crate::app) fn market_data_title_prefix(&self) -> String {
        self.bar_type.mode_label(self.effective_candle_mode())
    }

    pub(in crate::app) fn strategy_numeric_value(&self, focus: Focus, fallback: String) -> String {
        self.strategy_numeric_input
            .as_ref()
            .filter(|draft| draft.focus == focus)
            .map(|draft| draft.value.clone())
            .unwrap_or(fallback)
    }

    pub(in crate::app) fn market_update_age_ms(&self) -> Option<u64> {
        self.last_market_update_at
            .map(|instant| instant.elapsed().as_millis() as u64)
    }

    pub(in crate::app) fn latency_summary(&self) -> String {
        format!(
            "REST {} | Submit {} | Seen {} | Ack {} | Fill {} | Market {}",
            format_latency_ms(self.latency.rest_rtt_ms),
            format_latency_ms(self.latency.last_order_ack_ms),
            format_latency_ms(self.latency.last_order_seen_ms),
            format_latency_ms(self.latency.last_exec_report_ms),
            format_latency_ms(self.latency.last_fill_ms),
            format_age_ms(self.market_update_age_ms()),
        )
    }

    pub(in crate::app) fn strategy_readiness(&self) -> StrategyReadiness {
        let mut blockers = Vec::new();
        let mut warnings = Vec::new();
        let mut adjustments = Vec::new();

        if self.strategy.kind != StrategyKind::Native {
            return StrategyReadiness {
                status: StrategyReadinessStatus::PreviewOnly,
                blockers: vec![format!(
                    "{} is not executable from the TUI yet.",
                    self.strategy.kind.label()
                )],
                warnings,
                adjustments,
            };
        }

        if self.strategy.order_qty <= 0 {
            blockers.push("Order Qty must be at least 1.".to_string());
        }

        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                if self.strategy.native_hma.hma_length < 2 {
                    blockers.push("HMA Length must be at least 2.".to_string());
                }
                if self.strategy.native_hma.angle_lookback == 0 {
                    blockers.push("Angle Lookback must be at least 1.".to_string());
                }
                if self.strategy.native_hma.bars_required_to_trade == 0 {
                    blockers.push("Bars Required must be at least 1.".to_string());
                }
            }
            NativeStrategyKind::EmaCross => {
                let fast = self.strategy.native_ema.fast_length;
                let slow = self.strategy.native_ema.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast EMA Length ({fast}) must be less than Slow EMA Length ({slow})."
                    ));
                }
            }
            NativeStrategyKind::HmaCross => {
                let fast = self.strategy.native_hma_cross.fast_length;
                let slow = self.strategy.native_hma_cross.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast HMA Length ({fast}) must be less than Slow HMA Length ({slow})."
                    ));
                }
            }
            NativeStrategyKind::HeikinAshiColor => {}
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                let config = &self.strategy.native_volume_hma_cross;
                let fast = config.hma_cross.fast_length;
                let slow = config.hma_cross.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast HMA Length ({fast}) must be less than Slow HMA Length ({slow})."
                    ));
                }
                if let Err(error) = config.volume_regime.validate() {
                    blockers.push(error);
                }
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                let config = &self.strategy.native_volume_ema_cross;
                let fast = config.ema_cross.fast_length;
                let slow = config.ema_cross.slow_length;
                if fast >= slow {
                    blockers.push(format!(
                        "Fast EMA Length ({fast}) must be less than Slow EMA Length ({slow})."
                    ));
                }
                if let Err(error) = config.volume_regime.validate() {
                    blockers.push(error);
                }
                if let Err(error) = config.ema_gate.validate() {
                    blockers.push(error);
                }
            }
            NativeStrategyKind::Adx => {
                let config = &self.strategy.native_adx;
                if config.adx_length == 0 {
                    blockers.push("ADX Length must be at least 1.".to_string());
                }
                if !config.adx_entry_threshold.is_finite()
                    || !(0.0..=100.0).contains(&config.adx_entry_threshold)
                {
                    blockers.push(
                        "ADX Entry Threshold must be finite and between 0 and 100.".to_string(),
                    );
                }
                if !config.adx_exit_threshold.is_finite()
                    || !(0.0..=100.0).contains(&config.adx_exit_threshold)
                {
                    blockers.push(
                        "ADX Exit Threshold must be finite and between 0 and 100.".to_string(),
                    );
                }
                if config.adx_entry_threshold.is_finite()
                    && config.adx_exit_threshold.is_finite()
                    && config.adx_exit_threshold >= config.adx_entry_threshold
                {
                    blockers.push("ADX Exit Threshold must be below Entry Threshold.".to_string());
                }
                if !config.di_imbalance_threshold.is_finite()
                    || !(0.0..=1.0).contains(&config.di_imbalance_threshold)
                {
                    blockers.push("DI Imbalance Threshold must be between 0 and 1.".to_string());
                }
                if config.slope_lookback == 0 {
                    blockers.push("ADX Slope Lookback must be at least 1.".to_string());
                }
                if config.dominance_bars == 0 {
                    blockers.push("ADX Dominance Bars must be at least 1.".to_string());
                }
                for (label, value) in [
                    ("ADX Take Profit Ticks", config.take_profit_ticks),
                    ("ADX Stop Loss Ticks", config.stop_loss_ticks),
                    ("ADX Trail Trigger Ticks", config.trail_trigger_ticks),
                    ("ADX Trail Offset Ticks", config.trail_offset_ticks),
                ] {
                    if !value.is_finite() || value < 0.0 {
                        blockers.push(format!("{label} must be finite and nonnegative."));
                    }
                }
            }
        }

        if self.strategy.native_signal_timing == NativeSignalTiming::ClosedBar
            && self.market.bars.is_empty()
        {
            warnings.push("Closed-bar timing will wait for completed bars.".to_string());
        }
        if self.native_protection_controls_visible()
            && !self.active_native_uses_broker_owned_protection()
        {
            warnings.push("No TP/SL/trailing protection is configured.".to_string());
        }

        if self.active_native_uses_broker_owned_protection() && !self.capabilities.native_protection
        {
            blockers.push("Native protection is unavailable for this engine.".to_string());
        } else {
            if self.active_native_uses_broker_owned_protection()
                && self.strategy.native_reversal_mode == NativeReversalMode::Direct
            {
                adjustments.push(
                    "Direct reversal will switch to CloseAll > Enter for broker-owned protection."
                        .to_string(),
                );
            }
            if self.active_native_requires_guarded_path()
                && self.strategy.native_execution_path != NativeExecutionPath::Guarded
            {
                adjustments.push(
                    "Execution Path will switch to Guarded for this reversal/protection setup."
                        .to_string(),
                );
            }
        }

        if !blockers.is_empty() {
            return StrategyReadiness {
                status: StrategyReadinessStatus::NeedsAttention,
                blockers,
                warnings,
                adjustments,
            };
        }

        if !self.automated_strategy_affordance_visible() {
            blockers.push(format!(
                "{} automation is unavailable.",
                self.selected_broker.label()
            ));
        }
        if self.accounts.get(self.selected_account).is_none() {
            blockers.push("Select an account before arming.".to_string());
        }
        if !self.selected_contract_ready() {
            blockers.push("Select a contract before arming.".to_string());
        }

        StrategyReadiness {
            status: if blockers.is_empty() {
                StrategyReadinessStatus::ReadyToArm
            } else {
                StrategyReadinessStatus::MonitorOnly
            },
            blockers,
            warnings,
            adjustments,
        }
    }

    pub(in crate::app) fn selected_contract_ready(&self) -> bool {
        self.contract_results.get(self.selected_contract).is_some()
            || self.market.contract_id.is_some()
            || self.market.contract_name.is_some()
    }

    pub(in crate::app) fn strategy_continue_label(&self) -> String {
        match self.strategy_readiness().status {
            StrategyReadinessStatus::ReadyToArm => {
                "[Enter] Continue / Arm Native Strategy".to_string()
            }
            StrategyReadinessStatus::MonitorOnly => "[Enter] Continue / Monitor Only".to_string(),
            StrategyReadinessStatus::PreviewOnly => "[Enter] Continue / Preview Only".to_string(),
            StrategyReadinessStatus::NeedsAttention => "[Enter] Review Strategy Setup".to_string(),
        }
    }

    pub(in crate::app) fn sync_execution_strategy_config(&self, cmd_tx: &ServiceCommandSender) {
        if !self.capabilities.automated_orders {
            return;
        }
        let _ = cmd_tx.send(ServiceCommand::SetExecutionStrategyConfig(
            self.strategy.execution_config(),
        ));
    }

    pub(in crate::app) fn native_protection_controls_visible(&self) -> bool {
        if self.strategy.kind != StrategyKind::Native {
            return false;
        }
        if !self.capabilities.native_protection {
            return false;
        }
        if self.selected_broker != BrokerKind::Tradovate {
            return true;
        }
        self.strategy.native_execution_path == NativeExecutionPath::Guarded
            && self.strategy.native_reversal_mode != NativeReversalMode::Direct
    }

    pub(in crate::app) fn native_summary_for_display(&self) -> String {
        if self.native_protection_controls_visible() {
            self.strategy.native_summary()
        } else {
            self.strategy.native_summary_without_protection()
        }
    }

    pub(in crate::app) fn active_native_uses_broker_owned_protection(&self) -> bool {
        if self.strategy.kind != StrategyKind::Native {
            return false;
        }
        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self.strategy.native_hma.uses_native_protection(),
            NativeStrategyKind::EmaCross => self.strategy.native_ema.uses_native_protection(),
            NativeStrategyKind::HmaCross => self.strategy.native_hma_cross.uses_native_protection(),
            NativeStrategyKind::HeikinAshiColor => false,
            NativeStrategyKind::VolumeAdaptiveHmaCross => self
                .strategy
                .native_volume_hma_cross
                .uses_native_protection(),
            NativeStrategyKind::VolumeAdaptiveEmaCross => self
                .strategy
                .native_volume_ema_cross
                .uses_native_protection(),
            NativeStrategyKind::Adx => self.strategy.native_adx.uses_native_protection(),
        }
    }

    pub(in crate::app) fn active_native_requires_guarded_path(&self) -> bool {
        self.strategy.kind == StrategyKind::Native
            && (self.strategy.native_reversal_mode != NativeReversalMode::Direct
                || self.active_native_uses_broker_owned_protection()
                // HMA Direct is implemented only for HMA Crossover.  Other
                // native strategies must be normalized to Guarded so an
                // armed configuration cannot silently become a no-op.
                || (self.strategy.native_execution_path == NativeExecutionPath::HmaDirect
                    && !matches!(
                        self.strategy.native_strategy,
                        NativeStrategyKind::HmaCross | NativeStrategyKind::VolumeAdaptiveHmaCross
                    )))
    }

    pub(in crate::app) fn normalize_native_reversal_mode_before_arm(
        &mut self,
    ) -> Option<&'static str> {
        if !self.active_native_uses_broker_owned_protection()
            || self.strategy.native_reversal_mode != NativeReversalMode::Direct
        {
            return None;
        }
        let previous = self.strategy.native_reversal_mode.label();
        self.strategy.native_reversal_mode = NativeReversalMode::CloseAllEnter;
        Some(previous)
    }

    pub(in crate::app) fn normalize_native_execution_path_before_arm(
        &mut self,
    ) -> Option<&'static str> {
        if !self.active_native_requires_guarded_path()
            || self.strategy.native_execution_path == NativeExecutionPath::Guarded
        {
            return None;
        }
        let previous = self.strategy.native_execution_path.label();
        self.strategy.native_execution_path = NativeExecutionPath::Guarded;
        Some(previous)
    }

    pub(in crate::app) fn manual_disarm_native_strategy(&mut self, cmd_tx: &ServiceCommandSender) {
        if !self.capabilities.automated_orders {
            return;
        }
        let _ = cmd_tx.send(ServiceCommand::DisarmExecutionStrategy {
            reason: "Manual strategy disarm requested.".to_string(),
        });
        self.push_log("Manual strategy disarm requested.".to_string());
    }

    pub(in crate::app) fn arm_native_strategy(&mut self, cmd_tx: &ServiceCommandSender) {
        if !self.capabilities.automated_orders {
            return;
        }
        if let Some(previous_mode) = self.normalize_native_reversal_mode_before_arm() {
            self.push_log(format!(
                "Reversal Mode switched to CloseAll > Enter before arming; {previous_mode} cannot attach broker-owned TP/SL/trailing protection."
            ));
        }
        if let Some(previous_path) = self.normalize_native_execution_path_before_arm() {
            self.push_log(format!(
                "Execution Path switched to Guarded before arming; {previous_path} ignores broker protection or non-direct reversal settings."
            ));
        }
        self.sync_execution_strategy_config(cmd_tx);
        let _ = cmd_tx.send(ServiceCommand::ArmExecutionStrategy);
    }
}

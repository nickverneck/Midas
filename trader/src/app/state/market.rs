use super::super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(in crate::app) struct DisplayedTradeLevels {
    pub(in crate::app) entry_price: Option<f64>,
    pub(in crate::app) take_profit_price: Option<f64>,
    pub(in crate::app) stop_price: Option<f64>,
    pub(in crate::app) take_profit_projected: bool,
    pub(in crate::app) stop_price_projected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::app) struct DisplayedAutoTrail {
    pub(in crate::app) trigger_ticks: f64,
    pub(in crate::app) offset_ticks: f64,
    pub(in crate::app) initial_stop_ticks_from_entry: f64,
    pub(in crate::app) first_stop_ticks_from_entry: f64,
    pub(in crate::app) has_fixed_stop: bool,
    pub(in crate::app) initial_stop_price: Option<f64>,
    pub(in crate::app) trigger_price: Option<f64>,
    pub(in crate::app) first_stop_price: Option<f64>,
}

impl App {
    pub(in crate::app) fn set_replay_speed(
        &mut self,
        cmd_tx: &UnboundedSender<ServiceCommand>,
        speed: ReplaySpeed,
    ) {
        if self.session_kind != SessionKind::Replay || self.replay_speed == speed {
            return;
        }
        self.replay_speed = speed;
        let _ = cmd_tx.send(ServiceCommand::SetReplaySpeed { speed });
    }

    pub(in crate::app) fn sync_selected_account(&self, cmd_tx: &UnboundedSender<ServiceCommand>) {
        if let Some(account) = self.accounts.get(self.selected_account) {
            let _ = cmd_tx.send(ServiceCommand::SelectAccount {
                account_id: account.id,
            });
        }
    }

    pub(in crate::app) fn selected_snapshot(&self) -> Option<&AccountSnapshot> {
        let account = self.accounts.get(self.selected_account)?;
        self.snapshot_for_account(account.id)
    }

    pub(in crate::app) fn selected_contract_unrealized_pnl(
        &self,
        snapshot: &AccountSnapshot,
    ) -> Option<f64> {
        let marked = snapshot
            .market_position_qty
            .filter(|qty| qty.is_finite())
            .zip(
                snapshot
                    .market_entry_price
                    .filter(|price| price.is_finite()),
            )
            .zip(self.market.bars.last().map(|bar| bar.close))
            .zip(
                self.market
                    .value_per_point
                    .filter(|value| value.is_finite() && *value > 0.0),
            )
            .map(|(((qty, entry), mark), value_per_point)| (mark - entry) * qty * value_per_point);
        marked.or(snapshot.unrealized_pnl)
    }

    pub(in crate::app) fn snapshot_for_account(&self, account_id: i64) -> Option<&AccountSnapshot> {
        self.account_snapshots
            .iter()
            .find(|snapshot| snapshot.account_id == account_id)
    }

    pub(in crate::app) fn closed_bars(&self) -> &[crate::broker::Bar] {
        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        &self.market.bars[..closed_len]
    }

    pub(in crate::app) fn latest_closed_bar_ts(&self) -> Option<i64> {
        self.closed_bars().last().map(|bar| bar.ts_ns)
    }

    pub(in crate::app) fn session_window_at(&self, ts_ns: i64) -> Option<InstrumentSessionWindow> {
        self.market.session_profile.map(|profile| {
            profile.evaluate_with_blockout(ts_ns, self.strategy.blockout_minutes_before_close)
        })
    }

    pub(in crate::app) fn latest_session_window(&self) -> Option<InstrumentSessionWindow> {
        self.latest_closed_bar_ts()
            .and_then(|ts_ns| self.session_window_at(ts_ns))
    }

    pub(in crate::app) fn session_gate_summary(&self) -> String {
        if !self.strategy.blockout_enabled {
            return "blockout off".to_string();
        }

        let Some(profile) = self.market.session_profile else {
            return "n/a".to_string();
        };
        let Some(window) = self.latest_session_window() else {
            return format!("{} awaiting bars", profile.label());
        };

        if !window.session_open {
            return format!("{} closed; holding until reopen", profile.label());
        }

        let minutes_to_close = window
            .minutes_to_close
            .map(|minutes| format!("{minutes:.0}m"))
            .unwrap_or_else(|| "n/a".to_string());
        if self.strategy.blockout_enabled && window.hold_entries {
            format!(
                "{} hold active; flattening before close ({} left)",
                profile.label(),
                minutes_to_close
            )
        } else {
            format!("{} open; {} to close", profile.label(), minutes_to_close)
        }
    }

    pub(in crate::app) fn strategy_runtime_summary(&self) -> String {
        if !self.capabilities.automated_orders {
            return format!("{} monitor-only", self.selected_broker.label());
        }
        if !self.strategy_runtime.last_summary.is_empty() {
            return self.strategy_runtime.last_summary.clone();
        }
        if self.strategy_runtime.armed {
            "Armed".to_string()
        } else {
            "Disarmed".to_string()
        }
    }

    pub(in crate::app) fn effective_market_position_qty(&self) -> i32 {
        self.strategy_runtime.pending_target_qty.unwrap_or_else(|| {
            self.selected_snapshot()
                .and_then(|snapshot| snapshot.market_position_qty)
                .unwrap_or(0.0)
                .round() as i32
        })
    }

    pub(in crate::app) fn displayed_trade_levels(&self) -> DisplayedTradeLevels {
        let entry_price = self
            .selected_snapshot()
            .and_then(|snapshot| snapshot.market_entry_price)
            .filter(|price| price.is_finite());
        let signed_qty = self
            .selected_snapshot()
            .and_then(|snapshot| snapshot.market_position_qty)
            .map(|qty| qty.round() as i32)
            .filter(|qty| *qty != 0);
        let mut levels = DisplayedTradeLevels {
            entry_price,
            take_profit_price: self
                .selected_snapshot()
                .and_then(|snapshot| snapshot.selected_contract_take_profit_price)
                .filter(|price| price.is_finite()),
            stop_price: self
                .selected_snapshot()
                .and_then(|snapshot| snapshot.selected_contract_stop_price)
                .filter(|price| price.is_finite()),
            ..DisplayedTradeLevels::default()
        };

        let (Some(entry_price), Some(signed_qty)) = (entry_price, signed_qty) else {
            return levels;
        };

        if levels.take_profit_price.is_none() {
            levels.take_profit_price =
                self.projected_native_take_profit_price(entry_price, signed_qty);
            levels.take_profit_projected = levels.take_profit_price.is_some();
        }
        if levels.stop_price.is_none() {
            levels.stop_price = self.projected_native_stop_price(entry_price, signed_qty);
            levels.stop_price_projected = levels.stop_price.is_some();
        }

        levels
    }

    pub(in crate::app) fn displayed_auto_trail(&self) -> Option<DisplayedAutoTrail> {
        if self.strategy.kind != StrategyKind::Native {
            return None;
        }
        if !self.native_protection_controls_visible() {
            return None;
        }

        let (use_trailing_stop, trigger_ticks, offset_ticks, stop_loss_ticks) =
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => (
                    self.strategy.native_hma.use_trailing_stop,
                    self.strategy.native_hma.trail_trigger_ticks,
                    self.strategy.native_hma.trail_offset_ticks,
                    self.strategy.native_hma.stop_loss_ticks,
                ),
                NativeStrategyKind::EmaCross => (
                    self.strategy.native_ema.use_trailing_stop,
                    self.strategy.native_ema.trail_trigger_ticks,
                    self.strategy.native_ema.trail_offset_ticks,
                    self.strategy.native_ema.stop_loss_ticks,
                ),
                NativeStrategyKind::HmaCross => (
                    self.strategy.native_hma_cross.use_trailing_stop,
                    self.strategy.native_hma_cross.trail_trigger_ticks,
                    self.strategy.native_hma_cross.trail_offset_ticks,
                    self.strategy.native_hma_cross.stop_loss_ticks,
                ),
                NativeStrategyKind::HeikinAshiColor => (false, 0.0, 0.0, 0.0),
                NativeStrategyKind::VolumeAdaptiveHmaCross => (
                    self.strategy
                        .native_volume_hma_cross
                        .hma_cross
                        .use_trailing_stop,
                    self.strategy
                        .native_volume_hma_cross
                        .hma_cross
                        .trail_trigger_ticks,
                    self.strategy
                        .native_volume_hma_cross
                        .hma_cross
                        .trail_offset_ticks,
                    self.strategy
                        .native_volume_hma_cross
                        .hma_cross
                        .stop_loss_ticks,
                ),
                NativeStrategyKind::VolumeAdaptiveEmaCross => (
                    self.strategy
                        .native_volume_ema_cross
                        .ema_cross
                        .use_trailing_stop,
                    self.strategy
                        .native_volume_ema_cross
                        .ema_cross
                        .trail_trigger_ticks,
                    self.strategy
                        .native_volume_ema_cross
                        .ema_cross
                        .trail_offset_ticks,
                    self.strategy
                        .native_volume_ema_cross
                        .ema_cross
                        .stop_loss_ticks,
                ),
                NativeStrategyKind::Adx => (
                    self.strategy.native_adx.use_trailing_stop,
                    self.strategy.native_adx.trail_trigger_ticks,
                    self.strategy.native_adx.trail_offset_ticks,
                    self.strategy.native_adx.stop_loss_ticks,
                ),
            };
        if !use_trailing_stop || trigger_ticks <= 0.0 || offset_ticks <= 0.0 {
            return None;
        }

        let has_fixed_stop = stop_loss_ticks > 0.0;
        let initial_stop_distance_ticks = if has_fixed_stop {
            stop_loss_ticks
        } else {
            trigger_ticks + offset_ticks
        };
        let initial_stop_ticks_from_entry = -initial_stop_distance_ticks;
        let first_stop_ticks_from_entry = trigger_ticks - offset_ticks;
        let (initial_stop_price, trigger_price, first_stop_price) = self
            .market
            .tick_size
            .filter(|tick| tick.is_finite() && *tick > 0.0)
            .and_then(|tick_size| {
                let entry_price = self
                    .selected_snapshot()
                    .and_then(|snapshot| snapshot.market_entry_price)
                    .filter(|price| price.is_finite())?;
                let signed_qty = self
                    .selected_snapshot()
                    .and_then(|snapshot| snapshot.market_position_qty)
                    .map(|qty| qty.round() as i32)
                    .filter(|qty| *qty != 0)?;
                let direction = if signed_qty > 0 { 1.0 } else { -1.0 };
                Some((
                    entry_price + direction * initial_stop_ticks_from_entry * tick_size,
                    entry_price + direction * trigger_ticks * tick_size,
                    entry_price + direction * first_stop_ticks_from_entry * tick_size,
                ))
            })
            .map_or(
                (None, None, None),
                |(initial_stop_price, trigger_price, first_stop_price)| {
                    (
                        Some(initial_stop_price),
                        Some(trigger_price),
                        Some(first_stop_price),
                    )
                },
            );

        Some(DisplayedAutoTrail {
            trigger_ticks,
            offset_ticks,
            initial_stop_ticks_from_entry,
            first_stop_ticks_from_entry,
            has_fixed_stop,
            initial_stop_price,
            trigger_price,
            first_stop_price,
        })
    }

    pub(in crate::app) fn projected_native_take_profit_price(
        &self,
        entry_price: f64,
        signed_qty: i32,
    ) -> Option<f64> {
        if self.strategy.kind != StrategyKind::Native || !entry_price.is_finite() || signed_qty == 0
        {
            return None;
        }

        let offset = match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => self
                .strategy
                .native_hma
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::EmaCross => self
                .strategy
                .native_ema
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::HmaCross => self
                .strategy
                .native_hma_cross
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::HeikinAshiColor => return None,
            NativeStrategyKind::VolumeAdaptiveHmaCross => self
                .strategy
                .native_volume_hma_cross
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::VolumeAdaptiveEmaCross => self
                .strategy
                .native_volume_ema_cross
                .take_profit_offset(self.market.tick_size)?,
            NativeStrategyKind::Adx => self
                .strategy
                .native_adx
                .take_profit_offset(self.market.tick_size)?,
        };

        Some(if signed_qty > 0 {
            entry_price + offset
        } else {
            entry_price - offset
        })
    }

    pub(in crate::app) fn projected_native_stop_price(
        &self,
        entry_price: f64,
        signed_qty: i32,
    ) -> Option<f64> {
        if self.strategy.kind != StrategyKind::Native || !entry_price.is_finite() || signed_qty == 0
        {
            return None;
        }

        match self.strategy.native_strategy {
            NativeStrategyKind::HmaAngle => {
                let mut runtime = crate::strategies::hma_angle::HmaAngleExecutionState::default();
                self.strategy
                    .native_hma
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_hma
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::EmaCross => {
                let mut runtime = crate::strategies::ema_cross::EmaCrossExecutionState::default();
                self.strategy
                    .native_ema
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_ema
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::HmaCross => {
                let mut runtime = crate::strategies::hma_cross::HmaCrossExecutionState::default();
                self.strategy.native_hma_cross.sync_position(
                    &mut runtime,
                    signed_qty,
                    Some(entry_price),
                );
                self.strategy
                    .native_hma_cross
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::HeikinAshiColor => None,
            NativeStrategyKind::VolumeAdaptiveHmaCross => {
                let mut runtime = crate::strategies::hma_cross::HmaCrossExecutionState::default();
                self.strategy.native_volume_hma_cross.sync_position(
                    &mut runtime,
                    signed_qty,
                    Some(entry_price),
                );
                self.strategy
                    .native_volume_hma_cross
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::VolumeAdaptiveEmaCross => {
                let mut runtime = crate::strategies::ema_cross::EmaCrossExecutionState::default();
                self.strategy.native_volume_ema_cross.sync_position(
                    &mut runtime,
                    signed_qty,
                    Some(entry_price),
                );
                self.strategy
                    .native_volume_ema_cross
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
            NativeStrategyKind::Adx => {
                let mut runtime = crate::strategies::adx::AdxExecutionState::default();
                self.strategy
                    .native_adx
                    .sync_position(&mut runtime, signed_qty, Some(entry_price));
                self.strategy
                    .native_adx
                    .current_effective_stop_price(&runtime, self.market.tick_size)
            }
        }
        .filter(|price| price.is_finite())
    }
}

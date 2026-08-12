use super::super::*;

impl App {
    pub(in crate::app) fn login_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![
            Focus::Env,
            Focus::AuthMode,
            Focus::LogMode,
            Focus::TokenPath,
            Focus::TokenOverride,
            Focus::Username,
            Focus::Password,
            Focus::Connect,
        ];
        match self.selected_broker {
            BrokerKind::Ironbeam => {
                order.insert(7, Focus::ApiKey);
            }
            BrokerKind::Tradovate => {
                order.splice(
                    7..7,
                    [Focus::AppId, Focus::AppVersion, Focus::Cid, Focus::Secret],
                );
            }
        }
        order
    }

    pub(in crate::app) fn strategy_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::StrategyKind, Focus::OrderQty];
        if self.strategy.kind == StrategyKind::Native {
            let show_protection_controls = self.native_protection_controls_visible();
            order.push(Focus::NativeStrategy);
            order.push(Focus::NativeSignalTiming);
            order.push(Focus::NativeSignalDelayBars);
            order.push(Focus::NativeExecutionPath);
            order.push(Focus::NativeReversalMode);
            order.push(Focus::NativeBlockoutEnabled);
            order.push(Focus::NativeBlockoutMinutes);
            match self.strategy.native_strategy {
                NativeStrategyKind::HmaAngle => {
                    order.extend([
                        Focus::HmaLength,
                        Focus::HmaMinAngle,
                        Focus::HmaAngleLookback,
                        Focus::HmaBarsRequired,
                        Focus::HmaLongsOnly,
                        Focus::HmaInverted,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::HmaTakeProfitTicks,
                            Focus::HmaStopLossTicks,
                            Focus::HmaTrailingStop,
                        ]);
                        if self.strategy.native_hma.use_trailing_stop {
                            order.extend([Focus::HmaTrailTriggerTicks, Focus::HmaTrailOffsetTicks]);
                        }
                    }
                }
                NativeStrategyKind::EmaCross | NativeStrategyKind::HmaCross => {
                    order.extend([
                        Focus::EmaFastLength,
                        Focus::EmaSlowLength,
                        Focus::EmaInverted,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::EmaTakeProfitTicks,
                            Focus::EmaStopLossTicks,
                            Focus::EmaTrailingStop,
                        ]);
                        let use_trailing_stop = match self.strategy.native_strategy {
                            NativeStrategyKind::HmaCross => {
                                self.strategy.native_hma_cross.use_trailing_stop
                            }
                            _ => self.strategy.native_ema.use_trailing_stop,
                        };
                        if use_trailing_stop {
                            order.extend([Focus::EmaTrailTriggerTicks, Focus::EmaTrailOffsetTicks]);
                        }
                    }
                }
                NativeStrategyKind::HeikinAshiColor => {
                    order.push(Focus::EmaInverted);
                }
                NativeStrategyKind::VolumeAdaptiveHmaCross => {
                    order.extend([
                        Focus::EmaFastLength,
                        Focus::EmaSlowLength,
                        Focus::EmaInverted,
                        Focus::VolumeHmaLookbackBars,
                        Focus::VolumeHmaInvertBelowRatio,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::EmaTakeProfitTicks,
                            Focus::EmaStopLossTicks,
                            Focus::EmaTrailingStop,
                        ]);
                        if self
                            .strategy
                            .native_volume_hma_cross
                            .hma_cross
                            .use_trailing_stop
                        {
                            order.extend([Focus::EmaTrailTriggerTicks, Focus::EmaTrailOffsetTicks]);
                        }
                    }
                }
                NativeStrategyKind::VolumeAdaptiveEmaCross => {
                    order.extend([
                        Focus::EmaFastLength,
                        Focus::EmaSlowLength,
                        Focus::EmaInverted,
                        Focus::VolumeHmaLookbackBars,
                        Focus::VolumeHmaInvertBelowRatio,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::EmaTakeProfitTicks,
                            Focus::EmaStopLossTicks,
                            Focus::EmaTrailingStop,
                        ]);
                        if self
                            .strategy
                            .native_volume_ema_cross
                            .ema_cross
                            .use_trailing_stop
                        {
                            order.extend([Focus::EmaTrailTriggerTicks, Focus::EmaTrailOffsetTicks]);
                        }
                    }
                }
                NativeStrategyKind::Adx => {
                    order.extend([
                        Focus::AdxLength,
                        Focus::AdxEntryThreshold,
                        Focus::AdxExitThreshold,
                        Focus::AdxDiImbalance,
                        Focus::AdxSlopeLookback,
                        Focus::AdxDominanceBars,
                        Focus::AdxBreakoutLookback,
                        Focus::AdxInverted,
                    ]);
                    if show_protection_controls {
                        order.extend([
                            Focus::AdxTakeProfitTicks,
                            Focus::AdxStopLossTicks,
                            Focus::AdxTrailingStop,
                        ]);
                        if self.strategy.native_adx.use_trailing_stop {
                            order.extend([Focus::AdxTrailTriggerTicks, Focus::AdxTrailOffsetTicks]);
                        }
                    }
                }
            }
        } else if self.strategy.kind == StrategyKind::Lua {
            order.push(Focus::LuaSourceMode);
            match self.strategy.lua_source_mode {
                LuaSourceMode::File => order.push(Focus::LuaFilePath),
                LuaSourceMode::Editor => order.push(Focus::LuaEditor),
            }
        }
        order.push(Focus::StrategyContinue);
        order
    }

    pub(in crate::app) fn selection_focus_order(&self) -> Vec<Focus> {
        let mut order = vec![Focus::AccountList];
        if self.bar_type_controls_visible() {
            order.push(Focus::BarTypeToggle);
            order.push(Focus::BarValue);
        }
        if self.candle_mode_controls_visible() {
            order.push(Focus::CandleModeToggle);
        }
        order.extend([Focus::InstrumentQuery, Focus::ContractList]);
        order
    }

    pub(in crate::app) fn replay_focus_order(&self) -> Vec<Focus> {
        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Downloader {
            let mut order = vec![
                Focus::ReplayDownloadProvider,
                Focus::ReplayDownloadEnv,
                Focus::ReplayDownloadInstrument,
                Focus::ReplayDownloadContract,
                Focus::ReplayDownloadStart,
                Focus::ReplayDownloadEnd,
                Focus::ReplayDownloadSource,
            ];
            if self.replay_downloader.source_kind == Some(ReplayCacheSourceKind::ServerBars) {
                order.extend([
                    Focus::ReplayDownloadBarType,
                    Focus::ReplayDownloadBarValue,
                    Focus::ReplayDownloadCandleMode,
                ]);
            }
            order.extend([
                Focus::ReplayDownloadName,
                Focus::ReplayDownloadTags,
                Focus::ReplayDownloadCacheRoot,
                Focus::ReplayDownloadSubmit,
            ]);
            return order;
        }

        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::DatasetViews {
            let Some(editor) = self.replay_dataset_views.editor.as_ref() else {
                return vec![Focus::ReplayViewList];
            };
            let mut order = vec![Focus::ReplayViewId, Focus::ReplayViewPreset];
            match editor.preset {
                ReplayDatasetSessionPreset::FullSource => {}
                ReplayDatasetSessionPreset::FuturesGlobex
                | ReplayDatasetSessionPreset::FuturesRthNewYork
                | ReplayDatasetSessionPreset::FuturesRthChicago => {
                    order.push(Focus::ReplayViewTradingDate);
                }
                ReplayDatasetSessionPreset::CustomLocal => {
                    order.extend([
                        Focus::ReplayViewStart,
                        Focus::ReplayViewEnd,
                        Focus::ReplayViewTimezone,
                    ]);
                }
                ReplayDatasetSessionPreset::CustomUtc => {
                    order.extend([Focus::ReplayViewStart, Focus::ReplayViewEnd]);
                }
            }
            order.extend([Focus::ReplayViewWarmupMinutes, Focus::ReplayViewSave]);
            return order;
        }

        #[cfg(feature = "replay")]
        if self.replay_view == ReplayView::Library {
            return vec![Focus::ReplayInstrumentQuery, Focus::ReplayDataset];
        }

        let mut order = vec![Focus::BarTypeToggle, Focus::BarValue];
        if self.candle_mode_controls_visible() {
            order.push(Focus::CandleModeToggle);
        }
        #[cfg(feature = "replay")]
        if self.replay_view != ReplayView::Setup {
            order.splice(0..0, [Focus::ReplayInstrumentQuery, Focus::ReplayDataset]);
        }
        #[cfg(feature = "replay")]
        order.extend([
            Focus::ReplayInitialCapital,
            Focus::ReplayMarginPerContract,
            Focus::ReplaySafetyBuffer,
            Focus::ReplaySafetyBufferPercent,
        ]);
        order.push(Focus::ReplayMode);
        order
    }

    pub(in crate::app) fn next_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    pub(in crate::app) fn prev_login_focus(&self) -> Focus {
        let order = self.login_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    pub(in crate::app) fn next_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    pub(in crate::app) fn prev_strategy_focus(&self) -> Focus {
        let order = self.strategy_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    pub(in crate::app) fn next_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    pub(in crate::app) fn prev_selection_focus(&self) -> Focus {
        let order = self.selection_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    pub(in crate::app) fn next_replay_focus(&self) -> Focus {
        let order = self.replay_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + 1) % order.len()]
    }

    pub(in crate::app) fn prev_replay_focus(&self) -> Focus {
        let order = self.replay_focus_order();
        let index = order
            .iter()
            .position(|focus| *focus == self.focus)
            .unwrap_or(0);
        order[(index + order.len() - 1) % order.len()]
    }

    pub(in crate::app) fn is_text_focus(&self) -> bool {
        let standard_focus = matches!(
            self.focus,
            Focus::TokenOverride
                | Focus::Username
                | Focus::Password
                | Focus::ApiKey
                | Focus::AppId
                | Focus::AppVersion
                | Focus::Cid
                | Focus::Secret
                | Focus::TokenPath
                | Focus::OrderQty
                | Focus::HmaLength
                | Focus::HmaMinAngle
                | Focus::HmaAngleLookback
                | Focus::HmaBarsRequired
                | Focus::HmaTakeProfitTicks
                | Focus::HmaStopLossTicks
                | Focus::HmaTrailTriggerTicks
                | Focus::HmaTrailOffsetTicks
                | Focus::EmaFastLength
                | Focus::EmaSlowLength
                | Focus::EmaTakeProfitTicks
                | Focus::EmaStopLossTicks
                | Focus::EmaTrailTriggerTicks
                | Focus::EmaTrailOffsetTicks
                | Focus::VolumeHmaLookbackBars
                | Focus::VolumeHmaInvertBelowRatio
                | Focus::AdxLength
                | Focus::AdxEntryThreshold
                | Focus::AdxExitThreshold
                | Focus::AdxDiImbalance
                | Focus::AdxSlopeLookback
                | Focus::AdxDominanceBars
                | Focus::AdxBreakoutLookback
                | Focus::AdxTakeProfitTicks
                | Focus::AdxStopLossTicks
                | Focus::AdxTrailTriggerTicks
                | Focus::AdxTrailOffsetTicks
                | Focus::NativeBlockoutMinutes
                | Focus::NativeSignalDelayBars
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::BarValue
                | Focus::InstrumentQuery
        );
        #[cfg(feature = "replay")]
        let replay_download_focus = matches!(
            self.focus,
            Focus::ReplayDownloadInstrument
                | Focus::ReplayDownloadStart
                | Focus::ReplayDownloadEnd
                | Focus::ReplayDownloadName
                | Focus::ReplayDownloadTags
                | Focus::ReplayDownloadCacheRoot
        );
        #[cfg(feature = "replay")]
        let replay_instrument_focus = matches!(self.focus, Focus::ReplayInstrumentQuery);
        #[cfg(feature = "replay")]
        let replay_setup_numeric_focus = matches!(
            self.focus,
            Focus::ReplayInitialCapital
                | Focus::ReplayMarginPerContract
                | Focus::ReplaySafetyBuffer
                | Focus::ReplaySafetyBufferPercent
        );
        #[cfg(feature = "replay")]
        let replay_view_focus = matches!(
            self.focus,
            Focus::ReplayViewId
                | Focus::ReplayViewTradingDate
                | Focus::ReplayViewStart
                | Focus::ReplayViewEnd
                | Focus::ReplayViewTimezone
                | Focus::ReplayViewWarmupMinutes
        );
        #[cfg(not(feature = "replay"))]
        let replay_download_focus = false;
        #[cfg(not(feature = "replay"))]
        let replay_instrument_focus = false;
        #[cfg(not(feature = "replay"))]
        let replay_setup_numeric_focus = false;
        #[cfg(not(feature = "replay"))]
        let replay_view_focus = false;
        standard_focus
            || replay_download_focus
            || replay_instrument_focus
            || replay_setup_numeric_focus
            || replay_view_focus
    }

    pub(in crate::app) fn is_free_text_focus(&self) -> bool {
        let standard_focus = matches!(
            self.focus,
            Focus::TokenOverride
                | Focus::Username
                | Focus::Password
                | Focus::ApiKey
                | Focus::AppId
                | Focus::AppVersion
                | Focus::Cid
                | Focus::Secret
                | Focus::TokenPath
                | Focus::LuaFilePath
                | Focus::LuaEditor
                | Focus::InstrumentQuery
        );
        #[cfg(feature = "replay")]
        let replay_download_focus = matches!(
            self.focus,
            Focus::ReplayDownloadInstrument
                | Focus::ReplayDownloadStart
                | Focus::ReplayDownloadEnd
                | Focus::ReplayDownloadName
                | Focus::ReplayDownloadTags
                | Focus::ReplayDownloadCacheRoot
        );
        #[cfg(feature = "replay")]
        let replay_instrument_focus = matches!(self.focus, Focus::ReplayInstrumentQuery);
        #[cfg(feature = "replay")]
        let replay_view_focus = matches!(
            self.focus,
            Focus::ReplayViewId
                | Focus::ReplayViewTradingDate
                | Focus::ReplayViewStart
                | Focus::ReplayViewEnd
                | Focus::ReplayViewTimezone
                | Focus::ReplayViewWarmupMinutes
        );
        #[cfg(not(feature = "replay"))]
        let replay_download_focus = false;
        #[cfg(not(feature = "replay"))]
        let replay_instrument_focus = false;
        #[cfg(not(feature = "replay"))]
        let replay_view_focus = false;
        standard_focus || replay_download_focus || replay_instrument_focus || replay_view_focus
    }
}

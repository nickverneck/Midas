use super::*;

impl App {
    pub(in crate::app) fn handle_dashboard_key(
        &mut self,
        key: KeyEvent,
        cmd_tx: &ServiceCommandSender,
    ) {
        #[cfg(feature = "replay")]
        if self.session_kind.is_ninjatrader_replay() {
            match key.code {
                KeyCode::Char('[') => {
                    self.adjust_ninjatrader_replay_speed(cmd_tx, false);
                    return;
                }
                KeyCode::Char(']') => {
                    self.adjust_ninjatrader_replay_speed(cmd_tx, true);
                    return;
                }
                KeyCode::Char('0') => {
                    self.send_ninjatrader_replay_speed(cmd_tx, 0);
                    return;
                }
                KeyCode::Char('x') | KeyCode::Char('X') => {
                    let _ = cmd_tx.send(ServiceCommand::StopNinjaTraderReplay);
                    self.status = "Stopping NinjaTrader hosted Replay...".to_string();
                    self.push_log(self.status.clone());
                    return;
                }
                _ => {}
            }
        }
        let action = match key.code {
            KeyCode::Char('[') if self.session_kind == SessionKind::Replay => {
                self.set_replay_speed(cmd_tx, self.replay_speed.slower());
                return;
            }
            KeyCode::Char(']') if self.session_kind == SessionKind::Replay => {
                self.set_replay_speed(cmd_tx, self.replay_speed.faster());
                return;
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                match ch.to_ascii_lowercase() {
                    'v' => {
                        self.dashboard_visuals_enabled = !self.dashboard_visuals_enabled;
                        self.push_log(format!(
                            "Dashboard visuals {}",
                            if self.dashboard_visuals_enabled {
                                "enabled"
                            } else {
                                "disabled"
                            }
                        ));
                        return;
                    }
                    '0' if self.session_kind == SessionKind::Replay => {
                        self.set_replay_speed(cmd_tx, ReplaySpeed::Realtime);
                        return;
                    }
                    #[cfg(feature = "manual-orders")]
                    'b' => Some(ManualOrderAction::Buy),
                    #[cfg(feature = "manual-orders")]
                    's' => Some(ManualOrderAction::Sell),
                    #[cfg(feature = "manual-orders")]
                    'c' => Some(ManualOrderAction::Close),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(action) = action {
            if !self.manual_order_affordance_visible() {
                return;
            }
            self.sync_selected_account(cmd_tx);
            let _ = cmd_tx.send(ServiceCommand::ManualOrder { action });
        }
    }

    pub(in crate::app) fn render_dashboard(&self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(area);

        let session_lines = self.dashboard_summary_lines();
        let stats_lines = self.stats_lines();
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(12),
                Constraint::Length(stats_lines.len().saturating_add(2) as u16),
            ])
            .split(columns[0]);

        let session = Paragraph::new(session_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Session + Selection"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(session, left[0]);

        let stats = Paragraph::new(stats_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Account Stats"),
            )
            .wrap(Wrap { trim: true });
        frame.render_widget(stats, left[1]);

        self.render_chart(frame, columns[1]);
    }

    #[cfg(feature = "replay")]
    fn adjust_ninjatrader_replay_speed(&mut self, cmd_tx: &ServiceCommandSender, faster: bool) {
        const SPEEDS: [u16; 6] = [0, 25, 50, 100, 200, 400];
        let current = self
            .hosted_replay_speed
            .map(NinjaTraderReplaySpeed::percent)
            .unwrap_or(0);
        let position = SPEEDS
            .iter()
            .position(|speed| *speed == current)
            .unwrap_or(0);
        let next = if faster {
            (position + 1) % SPEEDS.len()
        } else {
            (position + SPEEDS.len() - 1) % SPEEDS.len()
        };
        self.send_ninjatrader_replay_speed(cmd_tx, SPEEDS[next]);
    }

    #[cfg(feature = "replay")]
    fn send_ninjatrader_replay_speed(&mut self, cmd_tx: &ServiceCommandSender, percent: u16) {
        let Ok(speed) = NinjaTraderReplaySpeed::new(percent) else {
            return;
        };
        if percent > 0 && !self.strategy_runtime.armed {
            self.status = "Replay is paused. Choose an executable strategy with F3, continue to arm it, then change speed.".to_string();
            self.push_log(format!("INFO: {}", self.status));
            return;
        }
        let _ = cmd_tx.send(ServiceCommand::SetNinjaTraderReplaySpeed { speed });
    }

    pub(in crate::app) fn render_chart(&self, frame: &mut Frame<'_>, area: Rect) {
        if self.market.bars.is_empty() {
            let empty_state =
                if self.session_kind.is_ninjatrader_replay() && self.selected_contract_ready() {
                    "Waiting for Replay quotes."
                } else {
                    "Select a contract to load history + live bars."
                };
            let empty = Paragraph::new(vec![
                Line::from(self.market.status.clone()),
                Line::from(empty_state),
            ])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!("{} Market Data", self.market_data_title_prefix())),
            );
            frame.render_widget(empty, area);
            return;
        }

        let closed_len = self.market.history_loaded.min(self.market.bars.len());
        let source_bars = if self.strategy.kind == StrategyKind::Native
            && self.strategy.native_signal_timing == NativeSignalTiming::ClosedBar
        {
            &self.market.bars[..closed_len]
        } else {
            &self.market.bars[..]
        };
        let bars = source_bars
            .iter()
            .rev()
            .take(180)
            .cloned()
            .collect::<Vec<_>>();
        let mut bars = bars.into_iter().rev().collect::<Vec<_>>();
        if bars.is_empty() {
            bars = source_bars.to_vec();
        }
        let visible_start = source_bars.len().saturating_sub(bars.len());

        let points = bars
            .iter()
            .enumerate()
            .map(|(idx, bar)| (idx as f64, bar.close))
            .collect::<Vec<_>>();
        let selected_snapshot = self.selected_snapshot();
        let selected_account_id = selected_snapshot.map(|snapshot| snapshot.account_id);
        let trade_levels = self.displayed_trade_levels();
        let entry_price = trade_levels.entry_price;
        let take_profit_price = trade_levels.take_profit_price;
        let stop_price = trade_levels.stop_price;
        let mut chart_prices = bars
            .iter()
            .flat_map(|bar| [bar.low, bar.close, bar.high])
            .filter(|price| price.is_finite())
            .collect::<Vec<_>>();
        let market_label = self.market_data_title_prefix();
        let mut title = match &self.market.contract_name {
            Some(name) => format!(
                "{market_label} Market Data [{}] hist={} live={}",
                name, self.market.history_loaded, self.market.live_bars
            ),
            None => format!("{market_label} Market Data"),
        };
        let mut overlay_labels = Vec::new();
        if let Some(price) = entry_price {
            overlay_labels.push(format!("EP {price:.2}"));
            chart_prices.push(price);
        }
        if let Some(price) = take_profit_price {
            let label = if trade_levels.take_profit_projected {
                "TP*"
            } else {
                "TP"
            };
            overlay_labels.push(format!("{label} {price:.2}"));
            chart_prices.push(price);
        }
        if let Some(price) = stop_price {
            let label = if trade_levels.stop_price_projected {
                "SL*"
            } else {
                "SL"
            };
            overlay_labels.push(format!("{label} {price:.2}"));
            chart_prices.push(price);
        }
        if !overlay_labels.is_empty() {
            title.push_str(" | ");
            title.push_str(&overlay_labels.join(" "));
        }
        let first_ts = bars.first().map(|bar| bar.ts_ns).unwrap_or_default();
        let last_ts = bars.last().map(|bar| bar.ts_ns).unwrap_or_default();
        let mut buy_marker_points = Vec::new();
        let mut sell_marker_points = Vec::new();
        for marker in &self.market.trade_markers {
            if marker.ts_ns < first_ts || marker.ts_ns > last_ts {
                continue;
            }
            if !trade_marker_matches_selection(marker, selected_account_id, &self.market) {
                continue;
            }
            let Some((idx, _)) = bars
                .iter()
                .enumerate()
                .min_by_key(|(_, bar)| bar.ts_ns.abs_diff(marker.ts_ns))
            else {
                continue;
            };
            chart_prices.push(marker.price);
            let point = (idx as f64, marker.price);
            match marker.side {
                TradeMarkerSide::Buy => buy_marker_points.push(point),
                TradeMarkerSide::Sell => sell_marker_points.push(point),
            }
        }
        let overlay = self.dashboard_visuals_enabled.then(|| {
            self.build_dashboard_visual_overlay(
                source_bars,
                visible_start,
                &buy_marker_points,
                &sell_marker_points,
            )
        });
        if let Some(overlay) = overlay.as_ref() {
            title.push_str(" | Visuals ");
            title.push_str(&overlay.label.to_uppercase());
        }
        let (min_close, max_close) = chart_prices
            .iter()
            .copied()
            .filter(|price| price.is_finite())
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(min_v, max_v), price| (min_v.min(price), max_v.max(price)),
            );
        let y_bounds = if min_close.is_finite() && max_close.is_finite() && min_close < max_close {
            let padding =
                ((max_close - min_close).abs() * 0.05).max(self.market.tick_size.unwrap_or(0.25));
            [min_close - padding, max_close + padding]
        } else if min_close.is_finite() {
            [min_close - 1.0, min_close + 1.0]
        } else {
            [0.0, 1.0]
        };
        let mut segment_points = Vec::with_capacity(points.len().saturating_sub(1));
        let mut segment_colors = Vec::with_capacity(points.len().saturating_sub(1));
        for window in points.windows(2) {
            let start = window[0];
            let end = window[1];
            segment_points.push(vec![start, end]);
            segment_colors.push(if end.1 >= start.1 {
                Color::Green
            } else {
                Color::Red
            });
        }

        let mut datasets = segment_points
            .iter()
            .zip(segment_colors.iter())
            .map(|(segment, color)| {
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(*color))
                    .data(segment.as_slice())
            })
            .collect::<Vec<_>>();

        let last_point_data = points.last().copied().map(|point| vec![point]);
        let line_end = points.len().max(2) as f64 - 1.0;
        let entry_line = entry_price.map(|price| vec![(0.0, price), (line_end, price)]);
        let take_profit_line = take_profit_price.map(|price| vec![(0.0, price), (line_end, price)]);
        let stop_line = stop_price.map(|price| vec![(0.0, price), (line_end, price)]);
        if let Some(last_point) = points.last().copied() {
            let last_point_color = if points.len() >= 2 && last_point.1 < points[points.len() - 2].1
            {
                Color::Red
            } else {
                Color::Green
            };
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(last_point_color))
                    .data(last_point_data.as_deref().unwrap_or(&[])),
            );
        }
        if let Some(line) = entry_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(line),
            );
        }
        if let Some(line) = take_profit_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Green))
                    .data(line),
            );
        }
        if let Some(line) = stop_line.as_deref() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Red))
                    .data(line),
            );
        }
        if !self.dashboard_visuals_enabled && !buy_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&buy_marker_points),
            );
        }
        if !self.dashboard_visuals_enabled && !sell_marker_points.is_empty() {
            datasets.push(
                Dataset::default()
                    .marker(symbols::Marker::Block)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(Color::Magenta))
                    .data(&sell_marker_points),
            );
        }
        let chart_block = Block::default().borders(Borders::ALL).title(title);
        let plot_area = chart_block.inner(area);
        let x_bounds = [0.0, points.len().max(1) as f64];
        let time_labels = self.dashboard_chart_time_labels(&bars);
        let price_labels = self.dashboard_chart_price_labels(y_bounds);
        let chart = Chart::new(datasets)
            .block(chart_block)
            .x_axis(
                Axis::default()
                    .title("Time local")
                    .bounds(x_bounds)
                    .labels(time_labels),
            )
            .y_axis(
                Axis::default()
                    .title("Price")
                    .bounds(y_bounds)
                    .labels(price_labels),
            );
        frame.render_widget(chart, area);
        if let Some(overlay) = overlay.as_ref() {
            self.render_dashboard_canvas_overlay(frame, plot_area, x_bounds, y_bounds, overlay);
        }
    }

    fn dashboard_chart_price_labels(&self, y_bounds: [f64; 2]) -> [String; 3] {
        let middle = y_bounds[0] + (y_bounds[1] - y_bounds[0]) / 2.0;
        let precision = chart_price_precision(self.market.tick_size);
        [y_bounds[0], middle, y_bounds[1]].map(|price| format!("{price:.precision$}"))
    }

    pub(in crate::app) fn dashboard_chart_time_labels(
        &self,
        bars: &[crate::broker::Bar],
    ) -> Vec<String> {
        if bars.is_empty() {
            return vec![String::new(), String::new()];
        }

        // These labels sit at fixed positions on the chart, so each one must
        // describe the bar at that position. In particular, don't interpolate
        // a timestamp between middle bars: replay data can contain equal or
        // non-monotonic timestamps, and interpolation can invent a time that
        // no displayed bar has.
        let label_indices = if bars.len() == 1 {
            vec![0]
        } else if bars.len() == 2 {
            vec![0, 1]
        } else {
            vec![0, bars.len() / 2, bars.len() - 1]
        };
        let timestamps = label_indices
            .iter()
            .map(|index| bars[*index].ts_ns)
            .collect::<Vec<_>>();
        let first_date = bars.first().and_then(|bar| local_bar_date(bar.ts_ns));
        let spans_multiple_days = bars
            .iter()
            .filter_map(|bar| local_bar_date(bar.ts_ns))
            .any(|date| Some(date) != first_date);
        let mut labels = timestamps
            .iter()
            .map(|timestamp| chart_time_label(*timestamp, spans_multiple_days, false))
            .collect::<Vec<_>>();

        if has_duplicate_labels(&labels) {
            labels = timestamps
                .iter()
                .map(|timestamp| chart_time_label(*timestamp, spans_multiple_days, true))
                .collect();
        }

        // Identical timestamps (or timestamps outside Chrono's supported
        // range) still need useful, distinct endpoint labels. The suffix is
        // the one-based position in the visible bar sequence.
        if has_duplicate_labels(&labels) {
            for (label, index) in labels.iter_mut().zip(label_indices) {
                label.push_str(&format!("#{}", index + 1));
            }
        }
        if labels.len() == 1 {
            labels.push(String::new());
        }
        labels
    }

    pub(in crate::app) fn render_dashboard_canvas_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        x_bounds: [f64; 2],
        y_bounds: [f64; 2],
        overlay: &DashboardVisualOverlay,
    ) {
        if area.is_empty() {
            return;
        }

        if !overlay.indicator_segments.is_empty() {
            let indicator_canvas = Canvas::default()
                .marker(symbols::Marker::Braille)
                .x_bounds(x_bounds)
                .y_bounds(y_bounds)
                .paint(|ctx| {
                    for segment in &overlay.indicator_segments {
                        ctx.draw(&CanvasLine {
                            x1: segment.start.0,
                            y1: segment.start.1,
                            x2: segment.end.0,
                            y2: segment.end.1,
                            color: segment.color,
                        });
                    }
                });
            frame.render_widget(indicator_canvas, area);
        }

        if overlay.glyphs.is_empty() {
            return;
        }

        let x_span = (x_bounds[1] - x_bounds[0]).abs().max(1.0);
        let y_span = (y_bounds[1] - y_bounds[0]).abs().max(0.0001);
        let dx = (x_span / 95.0).clamp(0.35, 1.35);
        let dy = (y_span / 35.0)
            .max(
                self.market
                    .tick_size
                    .filter(|tick| tick.is_finite() && *tick > 0.0)
                    .map(|tick| tick * 0.75)
                    .unwrap_or(y_span / 140.0),
            )
            .min(y_span / 16.0)
            .max(y_span / 240.0);
        let glyph_canvas = Canvas::default()
            .marker(symbols::Marker::HalfBlock)
            .x_bounds(x_bounds)
            .y_bounds(y_bounds)
            .paint(|ctx| {
                for glyph in overlay.glyphs.iter().copied() {
                    self.draw_overlay_glyph(ctx, glyph, dx, dy);
                }
            });
        frame.render_widget(glyph_canvas, area);
    }
}

fn chart_price_precision(tick_size: Option<f64>) -> usize {
    let Some(tick_size) = tick_size.filter(|tick| tick.is_finite() && *tick > 0.0) else {
        return 2;
    };
    (0..=4)
        .find(|precision| {
            let scale = 10_f64.powi(*precision as i32);
            ((tick_size * scale) - (tick_size * scale).round()).abs() < 1e-7
        })
        .unwrap_or(4)
}

fn has_duplicate_labels(labels: &[String]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(index, label)| labels[..index].contains(label))
}

fn local_bar_datetime(timestamp_ns: i64) -> Option<chrono::DateTime<chrono::Local>> {
    chrono::DateTime::<chrono::Utc>::from_timestamp(
        timestamp_ns.div_euclid(1_000_000_000),
        timestamp_ns.rem_euclid(1_000_000_000) as u32,
    )
    .map(|timestamp| timestamp.with_timezone(&chrono::Local))
}

fn local_bar_date(timestamp_ns: i64) -> Option<chrono::NaiveDate> {
    local_bar_datetime(timestamp_ns).map(|timestamp| timestamp.date_naive())
}

fn chart_time_label(timestamp_ns: i64, include_date: bool, include_seconds: bool) -> String {
    let Some(timestamp) = local_bar_datetime(timestamp_ns) else {
        return "--:--".to_string();
    };
    let format = match (include_date, include_seconds) {
        (true, true) => "%m-%d %H:%M:%S",
        (true, false) => "%m-%d %H:%M",
        (false, true) => "%H:%M:%S",
        (false, false) => "%H:%M",
    };
    timestamp.format(format).to_string()
}

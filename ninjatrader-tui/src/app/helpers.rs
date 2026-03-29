fn edit_string(target: &mut String, key: KeyEvent) {
    match key.code {
        KeyCode::Backspace => {
            target.pop();
        }
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            target.push(ch);
        }
        _ => {}
    }
}

fn styled_line(text: String, focused: bool) -> Line<'static> {
    if focused {
        Line::from(Span::styled(
            text,
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
    } else {
        Line::from(text)
    }
}

fn pnl_style(value: Option<f64>) -> Style {
    match value {
        Some(value) if value > 0.0 => Style::default().fg(Color::Green),
        Some(value) if value < 0.0 => Style::default().fg(Color::Red),
        _ => Style::default(),
    }
}

fn format_money(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_signed_money(value: Option<f64>) -> String {
    match value {
        Some(value) if value > 0.0 => format!("+{value:.2}"),
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_quantity(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "n/a".to_string(),
    }
}

fn format_latency_ms(value: Option<u64>) -> String {
    match value {
        Some(value) => format!("{value}ms"),
        None => "n/a".to_string(),
    }
}

fn format_age_ms(value: Option<u64>) -> String {
    match value {
        Some(value) if value >= 1_000 => format!("{:.1}s", value as f64 / 1_000.0),
        Some(value) => format!("{value}ms"),
        None => "n/a".to_string(),
    }
}

fn format_latency_group(
    submit: Option<u64>,
    seen: Option<u64>,
    ack: Option<u64>,
    fill: Option<u64>,
) -> String {
    format!(
        "sub {} | seen {} | ack {} | fill {}",
        format_latency_ms(submit),
        format_latency_ms(seen),
        format_latency_ms(ack),
        format_latency_ms(fill),
    )
}

#[derive(Debug, Clone, Copy)]
struct OverlaySegment {
    start: (f64, f64),
    end: (f64, f64),
    color: Color,
}

#[derive(Debug, Clone, Copy)]
enum OverlayGlyphKind {
    BuyMarker,
    SellMarker,
    BullishCross,
    BearishCross,
}

#[derive(Debug, Clone, Copy)]
struct OverlayGlyph {
    center: (f64, f64),
    color: Color,
    kind: OverlayGlyphKind,
}

#[derive(Debug, Clone, Default)]
struct DashboardVisualOverlay {
    label: String,
    indicator_segments: Vec<OverlaySegment>,
    glyphs: Vec<OverlayGlyph>,
}

fn trade_marker_matches_selection(
    marker: &TradeMarker,
    selected_account_id: Option<i64>,
    market: &MarketSnapshot,
) -> bool {
    let account_matches = selected_account_id
        .zip(marker.account_id)
        .map(|(selected, marker_account)| selected == marker_account)
        .unwrap_or(true);
    if !account_matches {
        return false;
    }

    if marker.contract_id.is_none() && marker.contract_name.is_none() {
        return true;
    }

    let contract_id_matches = marker
        .contract_id
        .zip(market.contract_id)
        .map(|(marker_contract_id, market_contract_id)| marker_contract_id == market_contract_id)
        .unwrap_or(false);
    let contract_name_matches = marker
        .contract_name
        .as_deref()
        .zip(market.contract_name.as_deref())
        .map(|(marker_contract_name, market_contract_name)| {
            marker_contract_name.eq_ignore_ascii_case(market_contract_name)
        })
        .unwrap_or(false);

    contract_id_matches || contract_name_matches
}

fn bool_label(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

fn append_overlay_series_segments(
    values: &[f64],
    color: Color,
    segments: &mut Vec<OverlaySegment>,
) {
    let mut previous = None;
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            previous = None;
            continue;
        }

        let current = (idx as f64, value);
        if let Some(start) = previous {
            segments.push(OverlaySegment {
                start,
                end: current,
                color,
            });
        }
        previous = Some(current);
    }
}

fn crossed_above(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a <= prev_b && curr_a > curr_b
}

fn crossed_below(prev_a: f64, prev_b: f64, curr_a: f64, curr_b: f64) -> bool {
    prev_a >= prev_b && curr_a < curr_b
}

fn mask(value: &str) -> String {
    if value.is_empty() {
        String::new()
    } else {
        "*".repeat(value.len().min(16))
    }
}

fn display_token_override(focused: bool, value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    if focused || value.len() <= 18 {
        return value.to_string();
    }
    format!("{}...{}", &value[..8], &value[value.len() - 6..])
}

fn toggle_bool(target: &mut bool, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = !*target;
            true
        }
        _ => false,
    }
}

fn adjust_usize(target: &mut usize, key: KeyEvent, min: usize, step: usize) -> bool {
    match key.code {
        KeyCode::Left => {
            *target = target.saturating_sub(step).max(min);
            true
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = target.saturating_add(step).max(min);
            true
        }
        _ => false,
    }
}

fn adjust_float(target: &mut f64, key: KeyEvent, min: f64, step: f64) -> bool {
    match key.code {
        KeyCode::Left => {
            *target = (*target - step).max(min);
            true
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
            *target = (*target + step).max(min);
            true
        }
        _ => false,
    }
}

fn edit_strategy_usize(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut usize,
    key: KeyEvent,
    min: usize,
    step: usize,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        return adjust_usize(target, key, min, step);
    }

    match key.code {
        KeyCode::Backspace => {
            let next = numeric_backspace(draft, focus, &target.to_string());
            if let Some(value) = next.and_then(|value| value.parse::<usize>().ok()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if ch.is_ascii_digit()
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            let next = numeric_append(draft, focus, ch, false);
            if let Ok(value) = next.parse::<usize>() {
                *target = value.max(min);
                return true;
            }
        }
        _ => {}
    }
    false
}

fn edit_strategy_i32(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut i32,
    key: KeyEvent,
    min: i32,
    step: i32,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        match key.code {
            KeyCode::Left => {
                *target = target.saturating_sub(step).max(min);
                return true;
            }
            KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ') => {
                *target = target.saturating_add(step).max(min);
                return true;
            }
            _ => return false,
        }
    }

    match key.code {
        KeyCode::Backspace => {
            let next = numeric_backspace(draft, focus, &target.to_string());
            if let Some(value) = next.and_then(|value| value.parse::<i32>().ok()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if ch.is_ascii_digit()
                && !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            let next = numeric_append(draft, focus, ch, false);
            if let Ok(value) = next.parse::<i32>() {
                *target = value.max(min);
                return true;
            }
        }
        _ => {}
    }
    false
}

fn edit_strategy_float(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    target: &mut f64,
    key: KeyEvent,
    min: f64,
    step: f64,
) -> bool {
    if matches!(
        key.code,
        KeyCode::Left | KeyCode::Right | KeyCode::Enter | KeyCode::Char(' ')
    ) {
        *draft = None;
        return adjust_float(target, key, min, step);
    }

    match key.code {
        KeyCode::Backspace => {
            let current = format_float_input(*target);
            let next = numeric_backspace(draft, focus, &current);
            if let Some(value) = parse_float_input(next.as_deref()) {
                *target = value.max(min);
                return true;
            }
        }
        KeyCode::Char(ch)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            if ch.is_ascii_digit() || ch == '.' {
                let next = numeric_append(draft, focus, ch, true);
                if let Some(value) = parse_float_input(Some(next.as_str())) {
                    *target = value.max(min);
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}

fn numeric_append(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    ch: char,
    allow_decimal: bool,
) -> String {
    let entry = match draft {
        Some(entry) if entry.focus == focus => entry,
        _ => {
            *draft = Some(NumericInputState {
                focus,
                value: String::new(),
            });
            draft.as_mut().expect("draft just inserted")
        }
    };

    if ch == '.' {
        if !allow_decimal || entry.value.contains('.') {
            return entry.value.clone();
        }
        if entry.value.is_empty() {
            entry.value.push('0');
        }
    }
    entry.value.push(ch);
    entry.value.clone()
}

fn numeric_backspace(
    draft: &mut Option<NumericInputState>,
    focus: Focus,
    current: &str,
) -> Option<String> {
    let entry = match draft {
        Some(entry) if entry.focus == focus => entry,
        _ => {
            *draft = Some(NumericInputState {
                focus,
                value: current.to_string(),
            });
            draft.as_mut().expect("draft just inserted")
        }
    };
    entry.value.pop();
    Some(entry.value.clone())
}

fn format_float_input(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        let mut text = value.to_string();
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.push('0');
        }
        text
    }
}

fn parse_float_input(value: Option<&str>) -> Option<f64> {
    let raw = value?;
    if raw.is_empty() {
        return None;
    }
    if raw == "." {
        return Some(0.0);
    }
    raw.parse::<f64>().ok()
}

impl App {
    fn dashboard_visual_overlay_label(&self) -> String {
        if !self.dashboard_visuals_enabled {
            return "off [v toggles]".to_string();
        }

        match self.strategy.kind {
            StrategyKind::Native => match self.strategy.native_strategy {
                NativeStrategyKind::EmaCross => "on | EMA fast/slow + fills".to_string(),
                NativeStrategyKind::HmaAngle => "on | HMA cross map + fills".to_string(),
            },
            _ => "on | fills only".to_string(),
        }
    }

    fn build_dashboard_visual_overlay(
        &self,
        bars: &[crate::tradovate::Bar],
        buy_marker_points: &[(f64, f64)],
        sell_marker_points: &[(f64, f64)],
    ) -> DashboardVisualOverlay {
        let mut overlay = DashboardVisualOverlay::default();
        overlay
            .glyphs
            .extend(buy_marker_points.iter().copied().map(|center| OverlayGlyph {
                center,
                color: Color::Cyan,
                kind: OverlayGlyphKind::BuyMarker,
            }));
        overlay
            .glyphs
            .extend(sell_marker_points.iter().copied().map(|center| OverlayGlyph {
                center,
                color: Color::Magenta,
                kind: OverlayGlyphKind::SellMarker,
            }));

        if self.strategy.kind != StrategyKind::Native || bars.len() < 2 {
            overlay.label = "fills".to_string();
            return overlay;
        }

        let close = bars.iter().map(|bar| bar.close).collect::<Vec<_>>();
        match self.strategy.native_strategy {
            NativeStrategyKind::EmaCross => {
                let fast = ema_series(&close, self.strategy.native_ema.fast_length.max(1));
                let slow = ema_series(&close, self.strategy.native_ema.slow_length.max(1));
                append_overlay_series_segments(
                    &fast,
                    Color::Cyan,
                    &mut overlay.indicator_segments,
                );
                append_overlay_series_segments(
                    &slow,
                    Color::Yellow,
                    &mut overlay.indicator_segments,
                );

                for idx in 1..close.len() {
                    let prev_fast = fast[idx - 1];
                    let curr_fast = fast[idx];
                    let prev_slow = slow[idx - 1];
                    let curr_slow = slow[idx];
                    if !prev_fast.is_finite()
                        || !curr_fast.is_finite()
                        || !prev_slow.is_finite()
                        || !curr_slow.is_finite()
                    {
                        continue;
                    }

                    let center = (idx as f64, (curr_fast + curr_slow) / 2.0);
                    if crossed_above(prev_fast, prev_slow, curr_fast, curr_slow) {
                        overlay.glyphs.push(OverlayGlyph {
                            center,
                            color: Color::Green,
                            kind: OverlayGlyphKind::BullishCross,
                        });
                    } else if crossed_below(prev_fast, prev_slow, curr_fast, curr_slow) {
                        overlay.glyphs.push(OverlayGlyph {
                            center,
                            color: Color::Red,
                            kind: OverlayGlyphKind::BearishCross,
                        });
                    }
                }

                overlay.label = "ema".to_string();
            }
            NativeStrategyKind::HmaAngle => {
                let hma = zero_lag_hma_series(&close, self.strategy.native_hma.hma_length.max(1));
                append_overlay_series_segments(
                    &hma,
                    Color::Yellow,
                    &mut overlay.indicator_segments,
                );

                for idx in 1..close.len() {
                    let prev_close = close[idx - 1];
                    let curr_close = close[idx];
                    let prev_hma = hma[idx - 1];
                    let curr_hma = hma[idx];
                    if !prev_close.is_finite()
                        || !curr_close.is_finite()
                        || !prev_hma.is_finite()
                        || !curr_hma.is_finite()
                    {
                        continue;
                    }

                    let center = (idx as f64, (curr_close + curr_hma) / 2.0);
                    if crossed_above(prev_close, prev_hma, curr_close, curr_hma) {
                        overlay.glyphs.push(OverlayGlyph {
                            center,
                            color: Color::Green,
                            kind: OverlayGlyphKind::BullishCross,
                        });
                    } else if crossed_below(prev_close, prev_hma, curr_close, curr_hma) {
                        overlay.glyphs.push(OverlayGlyph {
                            center,
                            color: Color::Red,
                            kind: OverlayGlyphKind::BearishCross,
                        });
                    }
                }

                overlay.label = "hma".to_string();
            }
        }

        overlay
    }

    fn draw_overlay_glyph(
        &self,
        ctx: &mut ratatui::widgets::canvas::Context<'_>,
        glyph: OverlayGlyph,
        dx: f64,
        dy: f64,
    ) {
        let (x, y) = glyph.center;
        match glyph.kind {
            OverlayGlyphKind::BuyMarker => {
                ctx.draw(&CanvasLine {
                    x1: x - dx,
                    y1: y - dy,
                    x2: x,
                    y2: y,
                    color: glyph.color,
                });
                ctx.draw(&CanvasLine {
                    x1: x + dx,
                    y1: y - dy,
                    x2: x,
                    y2: y,
                    color: glyph.color,
                });
                ctx.draw(&CanvasLine {
                    x1: x,
                    y1: y,
                    x2: x,
                    y2: y + dy * 0.7,
                    color: glyph.color,
                });
            }
            OverlayGlyphKind::SellMarker => {
                ctx.draw(&CanvasLine {
                    x1: x - dx,
                    y1: y + dy,
                    x2: x,
                    y2: y,
                    color: glyph.color,
                });
                ctx.draw(&CanvasLine {
                    x1: x + dx,
                    y1: y + dy,
                    x2: x,
                    y2: y,
                    color: glyph.color,
                });
                ctx.draw(&CanvasLine {
                    x1: x,
                    y1: y,
                    x2: x,
                    y2: y - dy * 0.7,
                    color: glyph.color,
                });
            }
            OverlayGlyphKind::BullishCross | OverlayGlyphKind::BearishCross => {
                ctx.draw(&CanvasLine {
                    x1: x - dx,
                    y1: y - dy,
                    x2: x + dx,
                    y2: y + dy,
                    color: glyph.color,
                });
                ctx.draw(&CanvasLine {
                    x1: x - dx,
                    y1: y + dy,
                    x2: x + dx,
                    y2: y - dy,
                    color: glyph.color,
                });
            }
        }
    }
}

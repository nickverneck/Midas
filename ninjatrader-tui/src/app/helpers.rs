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

fn pnl_line(label: &str, value: Option<f64>) -> Line<'static> {
    Line::from(vec![
        Span::raw(format!("{label}: ")),
        Span::styled(format_signed_money(value), pnl_style(value)),
    ])
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

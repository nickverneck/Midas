use super::*;

impl App {
    #[cfg(feature = "replay")]
    pub(super) fn render_replay_downloader_screen(&self, frame: &mut Frame<'_>, area: Rect) {
        let compact = area.width < 120 || area.height < 28;
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(if compact {
                [Constraint::Percentage(58), Constraint::Percentage(42)]
            } else {
                [Constraint::Percentage(52), Constraint::Percentage(48)]
            })
            .split(area);
        let downloader = &self.replay_downloader;
        let source_label = downloader
            .source_kind
            .map(|source| source.label())
            .unwrap_or("choose explicitly");
        let exact_label = downloader
            .exact_contract
            .as_ref()
            .map(|contract| format!("{} (#{} exact)", contract.name, contract.id))
            .unwrap_or_else(|| "not selected".to_string());
        let mut form_lines = vec![
            Line::from(format!(
                "Workflow: {} | Esc returns to library",
                match downloader.workflow {
                    ReplayDownloadWorkflow::New => "New dataset",
                    ReplayDownloadWorkflow::Extend => "Extend / refresh",
                }
            )),
            styled_line(
                format!("Provider: {} (only supported)", downloader.provider.label()),
                self.focus == Focus::ReplayDownloadProvider,
            ),
            styled_line(
                format!("Environment: {}", downloader.env.label()),
                self.focus == Focus::ReplayDownloadEnv,
            ),
            styled_line(
                format!("Instrument query: {}", downloader.instrument_query),
                self.focus == Focus::ReplayDownloadInstrument,
            ),
            styled_line(
                format!("Exact contract: {exact_label}"),
                self.focus == Focus::ReplayDownloadContract,
            ),
            styled_line(
                format!("Coverage start: {}", downloader.start_date),
                self.focus == Focus::ReplayDownloadStart,
            ),
            styled_line(
                format!("Coverage end: {} (inclusive)", downloader.end_date),
                self.focus == Focus::ReplayDownloadEnd,
            ),
            styled_line(
                format!("Source: {source_label} (Left bars / Right ticks)"),
                self.focus == Focus::ReplayDownloadSource,
            ),
        ];
        if downloader.source_kind == Some(ReplayCacheSourceKind::ServerBars) {
            form_lines.extend([
                styled_line(
                    format!("Server bar: {}", downloader.bar_type.kind().label()),
                    self.focus == Focus::ReplayDownloadBarType,
                ),
                styled_line(
                    format!("Bar value: {}", downloader.bar_type.value()),
                    self.focus == Focus::ReplayDownloadBarValue,
                ),
                styled_line(
                    format!("Candles: {}", downloader.candle_mode.label()),
                    self.focus == Focus::ReplayDownloadCandleMode,
                ),
            ]);
        }
        form_lines.extend([
            styled_line(
                format!("Dataset name: {}", downloader.display_name),
                self.focus == Focus::ReplayDownloadName,
            ),
            styled_line(
                format!("Tags (comma-separated): {}", downloader.tags),
                self.focus == Focus::ReplayDownloadTags,
            ),
            styled_line(
                format!("Cache root: {}", downloader.cache_root),
                self.focus == Focus::ReplayDownloadCacheRoot,
            ),
            styled_line(
                "[Enter] Start read-only download".to_string(),
                self.focus == Focus::ReplayDownloadSubmit,
            ),
        ]);
        let form_scroll = focused_paragraph_scroll_offset(&form_lines, columns[0]);
        let form = Paragraph::new(form_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Dataset Downloader"),
            )
            .scroll((form_scroll, 0))
            .wrap(Wrap { trim: true });
        frame.render_widget(form, columns[0]);

        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if compact {
                [Constraint::Length(6), Constraint::Min(5)]
            } else {
                [Constraint::Percentage(52), Constraint::Percentage(48)]
            })
            .split(columns[1]);
        let results = if downloader.contract_results.is_empty() {
            vec![ListItem::new(Line::from("Enter on Instrument to search"))]
        } else {
            downloader
                .contract_results
                .iter()
                .enumerate()
                .map(|(index, contract)| {
                    ListItem::new(styled_line(
                        format!(
                            "{} #{} | {}",
                            contract.name, contract.id, contract.description
                        ),
                        self.focus == Focus::ReplayDownloadContract
                            && index == downloader.selected_contract,
                    ))
                })
                .collect()
        };
        let contracts = List::new(results)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Exact Contract"),
            )
            .scroll_padding(1);
        let mut contract_state = focused_list_state(
            self.focus == Focus::ReplayDownloadContract,
            downloader.selected_contract,
            downloader.contract_results.len(),
        );
        frame.render_stateful_widget(contracts, right[0], &mut contract_state);

        let estimate = match (downloader.estimated_rows, downloader.estimated_bytes) {
            (Some(rows), Some(bytes)) => format!("{rows} rows / {bytes} bytes"),
            (Some(rows), None) => format!("{rows} rows / bytes unavailable"),
            (None, Some(bytes)) => format!("rows unavailable / {bytes} bytes"),
            (None, None) => "unavailable until provider responds".to_string(),
        };
        let actual = match (downloader.actual_rows, downloader.actual_bytes) {
            (Some(rows), Some(bytes)) => format!("{rows} rows / {bytes} bytes"),
            _ => "pending".to_string(),
        };
        let mut status_lines = vec![
            Line::from(format!("Phase: {}", downloader.phase.label())),
            Line::from(downloader.phase_message.clone()),
            Line::from(format!("Estimate: {estimate}")),
            Line::from(format!("Actual: {actual}")),
        ];
        if let Some(basis) = &downloader.suggestion_basis {
            status_lines.push(Line::from(format!("Range basis: {basis}")));
        }
        status_lines.push(Line::from(
            "Read-only REST + market-data WS; no user/account/order streams.",
        ));
        match downloader.phase {
            ReplayDownloadPhase::Searching
            | ReplayDownloadPhase::InspectingContract
            | ReplayDownloadPhase::Authenticating
            | ReplayDownloadPhase::Downloading => {
                status_lines.push(Line::from("Esc requests cancellation."));
                status_lines.push(Line::from("Wait for acknowledgement."));
            }
            ReplayDownloadPhase::Cancelling => {
                status_lines.push(Line::from("Cancellation requested."));
                status_lines.push(Line::from("Waiting for acknowledgement."));
            }
            ReplayDownloadPhase::WritingCache => {
                status_lines.push(Line::from("Commit is non-interruptible."));
                status_lines.push(Line::from("Esc will not hide this job."));
            }
            ReplayDownloadPhase::Idle
            | ReplayDownloadPhase::Ready
            | ReplayDownloadPhase::Busy
            | ReplayDownloadPhase::Cancelled
            | ReplayDownloadPhase::Complete
            | ReplayDownloadPhase::Failed => {}
        }
        let status = Paragraph::new(status_lines)
            .block(Block::default().borders(Borders::ALL).title("Download Job"))
            .wrap(Wrap { trim: true });
        frame.render_widget(status, right[1]);
    }
}

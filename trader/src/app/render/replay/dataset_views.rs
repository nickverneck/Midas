use super::*;
use crate::replay_cache::{
    ReplayDatasetSessionSelection, ReplayDatasetView, ReplayDatasetViewStore,
    ReplayDatasetWarmupPolicy, ReplayWarmupTradingPolicy,
};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

const VIEW_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

impl App {
    pub(super) fn open_replay_dataset_views(&mut self) {
        let Some(dataset) = self.replay_selected_dataset().cloned() else {
            self.status = "Select a cached dataset before managing replay views.".to_string();
            self.push_log(self.status.clone());
            return;
        };
        let store = ReplayDatasetViewStore::new(&self.base_config.replay_cache_dir);
        let library = store.list_for_dataset(&dataset);
        let selected_index = self.replay_dataset_view_path.as_ref().and_then(|selected| {
            library
                .views
                .iter()
                .position(|resolved| &resolved.view_path == selected)
        });
        self.replay_dataset_views = ReplayDatasetViewsState {
            selected_index: selected_index.or((!library.views.is_empty()).then_some(0)),
            views: library.views,
            warnings: library.warnings,
            editor: None,
            message: "Enter selects a view; N creates; E edits; C uses the full dataset."
                .to_string(),
        };
        self.replay_view = ReplayView::DatasetViews;
        self.focus = Focus::ReplayViewList;
    }

    pub(super) fn handle_replay_dataset_views_key(&mut self, key: KeyEvent) {
        if self.replay_dataset_views.editor.is_none() {
            self.handle_replay_dataset_view_list_key(key);
            return;
        }

        match key.code {
            KeyCode::BackTab => {
                self.focus = self.prev_replay_focus();
                return;
            }
            KeyCode::Tab => {
                self.focus = self.next_replay_focus();
                return;
            }
            _ => {}
        }

        let move_focus = |app: &mut App, key: KeyEvent| {
            app.focus = if key.code == KeyCode::Up {
                app.prev_replay_focus()
            } else {
                app.next_replay_focus()
            };
        };
        match self.focus {
            Focus::ReplayViewPreset => match key.code {
                KeyCode::Left => {
                    if let Some(editor) = self.replay_dataset_views.editor.as_mut() {
                        editor.preset = editor.preset.previous();
                    }
                }
                KeyCode::Right => {
                    if let Some(editor) = self.replay_dataset_views.editor.as_mut() {
                        editor.preset = editor.preset.next();
                    }
                }
                KeyCode::Up | KeyCode::Down | KeyCode::Enter => move_focus(self, key),
                _ => {}
            },
            Focus::ReplayViewId
            | Focus::ReplayViewTradingDate
            | Focus::ReplayViewStart
            | Focus::ReplayViewEnd
            | Focus::ReplayViewTimezone
            | Focus::ReplayViewWarmupMinutes => {
                if matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Enter) {
                    move_focus(self, key);
                    return;
                }
                let Some(editor) = self.replay_dataset_views.editor.as_mut() else {
                    return;
                };
                let target = match self.focus {
                    Focus::ReplayViewId => &mut editor.id,
                    Focus::ReplayViewTradingDate => &mut editor.trading_date,
                    Focus::ReplayViewStart => &mut editor.start,
                    Focus::ReplayViewEnd => &mut editor.end,
                    Focus::ReplayViewTimezone => &mut editor.timezone,
                    Focus::ReplayViewWarmupMinutes => &mut editor.warmup_minutes,
                    _ => unreachable!(),
                };
                edit_string(target, key);
            }
            Focus::ReplayViewSave => match key.code {
                KeyCode::Enter | KeyCode::Char(' ') => self.save_replay_dataset_view(),
                KeyCode::Up => self.focus = self.prev_replay_focus(),
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_replay_dataset_view_list_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Down => {
                let count = self.replay_dataset_views.views.len();
                if count == 0 {
                    return;
                }
                let current = self.replay_dataset_views.selected_index.unwrap_or(0);
                self.replay_dataset_views.selected_index = Some(if key.code == KeyCode::Up {
                    current.checked_sub(1).unwrap_or(count - 1)
                } else {
                    (current + 1) % count
                });
            }
            KeyCode::Char('n') | KeyCode::Char('N') => self.new_replay_dataset_view(),
            KeyCode::Char('e') | KeyCode::Char('E') => self.edit_replay_dataset_view(),
            KeyCode::Char('c') | KeyCode::Char('C') => {
                self.replay_dataset_view_path = None;
                self.replay_view = ReplayView::Library;
                self.focus = Focus::ReplayInstrumentQuery;
                self.status = "Replay will use the selected dataset's full coverage.".to_string();
            }
            KeyCode::Enter | KeyCode::Char(' ') => {
                let Some(resolved) = self
                    .replay_dataset_views
                    .selected_index
                    .and_then(|index| self.replay_dataset_views.views.get(index))
                else {
                    self.new_replay_dataset_view();
                    return;
                };
                self.replay_dataset_view_path = Some(resolved.view_path.clone());
                self.status = format!("Selected replay dataset view `{}`.", resolved.view.id);
                self.replay_view = ReplayView::Library;
                self.focus = Focus::ReplayInstrumentQuery;
            }
            _ => {}
        }
    }

    fn new_replay_dataset_view(&mut self) {
        let Some(dataset) = self.replay_selected_dataset() else {
            return;
        };
        let coverage = &dataset.manifest.coverage;
        let contract = dataset.manifest.contract.symbol.to_ascii_lowercase();
        let date = coverage
            .trading_date
            .unwrap_or_else(|| coverage.start.date_naive());
        self.replay_dataset_views.editor = Some(ReplayDatasetViewEditorState {
            id: format!("{}_{}", contract, date.format("%Y_%m_%d")),
            preset: ReplayDatasetSessionPreset::FullSource,
            trading_date: date.format("%Y-%m-%d").to_string(),
            start: coverage.start.format(VIEW_DATETIME_FORMAT).to_string(),
            end: coverage.end.format(VIEW_DATETIME_FORMAT).to_string(),
            timezone: "America/New_York".to_string(),
            warmup_minutes: "0".to_string(),
        });
        self.replay_dataset_views.message =
            "Create view: Left/Right changes preset; Tab or Enter advances fields.".to_string();
        self.focus = Focus::ReplayViewId;
    }

    fn edit_replay_dataset_view(&mut self) {
        let Some(resolved) = self
            .replay_dataset_views
            .selected_index
            .and_then(|index| self.replay_dataset_views.views.get(index))
            .cloned()
        else {
            self.replay_dataset_views.message = "No saved view is selected.".to_string();
            return;
        };
        let view = resolved.view;
        let timezone = view.input_tz().unwrap_or(chrono_tz::UTC);
        let local_start = view.evaluation_start.with_timezone(&timezone);
        let local_end = view.evaluation_end.with_timezone(&timezone);
        let trading_date = match view.session_preset {
            ReplayDatasetSessionPreset::FuturesGlobex => local_end.date_naive(),
            _ => local_start.date_naive(),
        };
        let (start, end) = if view.session_preset == ReplayDatasetSessionPreset::CustomUtc {
            (
                view.evaluation_start
                    .format(VIEW_DATETIME_FORMAT)
                    .to_string(),
                view.evaluation_end.format(VIEW_DATETIME_FORMAT).to_string(),
            )
        } else {
            (
                local_start.format(VIEW_DATETIME_FORMAT).to_string(),
                local_end.format(VIEW_DATETIME_FORMAT).to_string(),
            )
        };
        self.replay_dataset_views.editor = Some(ReplayDatasetViewEditorState {
            id: view.id,
            preset: view.session_preset,
            trading_date: trading_date.format("%Y-%m-%d").to_string(),
            start,
            end,
            timezone: view.input_timezone,
            warmup_minutes: (view.warmup.duration_seconds / 60).to_string(),
        });
        self.replay_dataset_views.message =
            "Edit view: save overwrites the view with this ID after validation.".to_string();
        self.focus = Focus::ReplayViewId;
    }

    fn save_replay_dataset_view(&mut self) {
        let Some(editor) = self.replay_dataset_views.editor.clone() else {
            return;
        };
        let Some(dataset) = self.replay_selected_dataset().cloned() else {
            self.replay_dataset_views.message = "Selected dataset is unavailable.".to_string();
            return;
        };
        let result = (|| -> anyhow::Result<_> {
            let selection = editor.session_selection()?;
            let warmup_minutes: u64 = editor
                .warmup_minutes
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("warmup must be a whole number of minutes"))?;
            let duration_seconds = warmup_minutes
                .checked_mul(60)
                .ok_or_else(|| anyhow::anyhow!("warmup duration is too large"))?;
            let view = ReplayDatasetView::from_session_selection(
                &self.base_config.replay_cache_dir,
                &dataset,
                editor.id.trim(),
                &selection,
                ReplayDatasetWarmupPolicy {
                    duration_seconds,
                    trading: ReplayWarmupTradingPolicy::FlatUntilEvaluation,
                },
            )?;
            ReplayDatasetViewStore::new(&self.base_config.replay_cache_dir).save(&view)
        })();

        match result {
            Ok(path) => {
                let saved_id = editor.id.trim().to_string();
                self.replay_dataset_view_path = Some(path);
                self.replay_dataset_views.editor = None;
                self.refresh_replay_dataset_views(Some(&saved_id));
                self.replay_dataset_views.message = format!(
                    "Saved and selected `{saved_id}`. Enter returns to Replay; E edits it."
                );
                self.focus = Focus::ReplayViewList;
                self.status = format!("Saved and selected replay dataset view `{saved_id}`.");
                self.push_log(self.status.clone());
            }
            Err(error) => {
                self.replay_dataset_views.message = format!("Cannot save view: {error:#}");
            }
        }
    }

    fn refresh_replay_dataset_views(&mut self, select_id: Option<&str>) {
        let Some(dataset) = self.replay_selected_dataset().cloned() else {
            return;
        };
        let library = ReplayDatasetViewStore::new(&self.base_config.replay_cache_dir)
            .list_for_dataset(&dataset);
        self.replay_dataset_views.selected_index = select_id.and_then(|id| {
            library
                .views
                .iter()
                .position(|resolved| resolved.view.id == id)
        });
        self.replay_dataset_views.views = library.views;
        self.replay_dataset_views.warnings = library.warnings;
    }

    pub(super) fn replay_selected_dataset_view_path(&self) -> Option<std::path::PathBuf> {
        self.replay_dataset_view_path.clone()
    }
}

impl ReplayDatasetViewEditorState {
    fn session_selection(&self) -> anyhow::Result<ReplayDatasetSessionSelection> {
        let trading_date = || {
            NaiveDate::parse_from_str(self.trading_date.trim(), "%Y-%m-%d")
                .map_err(|_| anyhow::anyhow!("trading date must use YYYY-MM-DD"))
        };
        let naive = |raw: &str, label: &str| {
            NaiveDateTime::parse_from_str(raw.trim(), VIEW_DATETIME_FORMAT)
                .map_err(|_| anyhow::anyhow!("{label} must use YYYY-MM-DD HH:MM:SS"))
        };
        Ok(match self.preset {
            ReplayDatasetSessionPreset::FullSource => ReplayDatasetSessionSelection::FullSource,
            ReplayDatasetSessionPreset::FuturesGlobex => {
                ReplayDatasetSessionSelection::FuturesGlobex {
                    trading_date: trading_date()?,
                }
            }
            ReplayDatasetSessionPreset::FuturesRthNewYork => {
                ReplayDatasetSessionSelection::FuturesRthNewYork {
                    trading_date: trading_date()?,
                }
            }
            ReplayDatasetSessionPreset::FuturesRthChicago => {
                ReplayDatasetSessionSelection::FuturesRthChicago {
                    trading_date: trading_date()?,
                }
            }
            ReplayDatasetSessionPreset::CustomLocal => ReplayDatasetSessionSelection::CustomLocal {
                start: naive(&self.start, "start")?,
                end: naive(&self.end, "end")?,
                timezone: self.timezone.trim().to_string(),
            },
            ReplayDatasetSessionPreset::CustomUtc => ReplayDatasetSessionSelection::CustomUtc {
                start: parse_utc_datetime(&self.start, "start")?,
                end: parse_utc_datetime(&self.end, "end")?,
            },
        })
    }
}

fn parse_utc_datetime(raw: &str, label: &str) -> anyhow::Result<DateTime<Utc>> {
    if let Ok(value) = DateTime::parse_from_rfc3339(raw.trim()) {
        return Ok(value.with_timezone(&Utc));
    }
    NaiveDateTime::parse_from_str(raw.trim(), VIEW_DATETIME_FORMAT)
        .map(|value| DateTime::from_naive_utc_and_offset(value, Utc))
        .map_err(|_| anyhow::anyhow!("{label} must use YYYY-MM-DD HH:MM:SS or RFC3339"))
}

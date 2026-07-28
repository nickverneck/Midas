use super::super::*;

impl App {
    pub(in crate::app) fn active_engine_summary(&self) -> Option<&EngineSummary> {
        let active_key = self.active_engine_key.as_ref()?;
        self.engine_summaries
            .iter()
            .find(|summary| &summary.key == active_key)
    }

    pub(in crate::app) fn active_engine_socket_label(&self) -> String {
        self.engine_socket_path
            .as_ref()
            .or_else(|| {
                self.active_engine_summary()
                    .map(|summary| &summary.socket_path)
            })
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    pub(in crate::app) fn active_engine_socket_short_label(&self) -> String {
        self.active_engine_summary()
            .map(EngineSummary::socket_short_label)
            .or_else(|| {
                self.engine_socket_path.as_ref().map(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(ToString::to_string)
                        .unwrap_or_else(|| path.display().to_string())
                })
            })
            .unwrap_or_else(|| "none".to_string())
    }

    pub(in crate::app) fn active_engine_id_label(&self) -> String {
        self.active_engine_summary()
            .and_then(|summary| summary.id)
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string())
    }

    pub(in crate::app) fn active_engine_connection_state_label(&self) -> String {
        self.active_engine_summary()
            .map(|summary| summary.connection_state.label().to_string())
            .unwrap_or_else(|| {
                if self.engine_socket_path.is_some() {
                    "unknown".to_string()
                } else {
                    "none".to_string()
                }
            })
    }

    pub(in crate::app) fn active_engine_header_label(&self) -> String {
        if self.engine_socket_path.is_none() && self.active_engine_key.is_none() {
            return "none".to_string();
        }

        let id = self.active_engine_id_label();
        let socket = self.active_engine_socket_short_label();
        let identity = if id == "none" {
            socket
        } else {
            format!("#{id} {socket}")
        };
        let mut label = format!(
            "{} {} {}",
            identity,
            self.active_engine_connection_state_label(),
            self.session_kind.label()
        );
        let other_count = self.other_live_engine_count();
        if other_count > 0 {
            label.push_str(&format!(" +{other_count} other"));
        }
        label
    }

    pub(in crate::app) fn engine_receiver_closed_message(&self, engine_key: &EngineKey) -> String {
        let Some(summary) = self
            .engine_summaries
            .iter()
            .find(|summary| &summary.key == engine_key)
        else {
            return format!("Engine {} connection closed.", engine_key.display_label());
        };

        format!(
            "Engine {} connection closed; last state was {}.",
            summary.identity_label(),
            summary.connection_state.label()
        )
    }

    pub(in crate::app) fn active_engine_key_display_label(&self) -> String {
        self.active_engine_key
            .as_ref()
            .map(EngineKey::display_label)
            .unwrap_or_else(|| "none".to_string())
    }

    pub(in crate::app) fn active_engine_stats_identity_key(&self) -> String {
        self.active_engine_key
            .as_ref()
            .map(|key| format!("engine:{}", key.display_label()))
            .unwrap_or_else(|| "embedded".to_string())
    }

    pub(in crate::app) fn other_live_engine_count(&self) -> usize {
        self.engine_summaries
            .iter()
            .filter(|summary| {
                self.active_engine_key
                    .as_ref()
                    .is_none_or(|active_key| &summary.key != active_key)
                    && matches!(
                        summary.connection_state,
                        EngineConnectionState::Observing | EngineConnectionState::Connected
                    )
            })
            .count()
    }
}

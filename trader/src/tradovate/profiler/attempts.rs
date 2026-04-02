use super::report::{SwipeAttemptReport, SwipeSubmitObservation};
use super::*;
use std::time::Instant;

pub(super) enum AttemptDispatchProgress {
    Submitted,
    DispatchNote(String),
    Timeout,
}

struct LiveAttemptState {
    sent_at: Instant,
    report: SwipeAttemptReport,
}

pub(super) struct AttemptBook {
    attempts: Vec<LiveAttemptState>,
    request_to_index: BTreeMap<String, usize>,
}

impl AttemptBook {
    pub(super) fn new() -> Self {
        Self {
            attempts: Vec::new(),
            request_to_index: BTreeMap::new(),
        }
    }

    pub(super) fn start_attempt(
        &mut self,
        iteration: usize,
        target_qty: i32,
        reason_tag: String,
    ) -> usize {
        let index = self.attempts.len();
        self.attempts.push(LiveAttemptState {
            sent_at: Instant::now(),
            report: SwipeAttemptReport {
                iteration,
                target_qty,
                reason_tag,
                sent_at_utc: Utc::now(),
                send_to_submit_event_ms: None,
                submit: None,
                dispatch_notes: Vec::new(),
                delay_probe: None,
                delay_findings: Vec::new(),
            },
        });
        index
    }

    pub(super) fn observe(&mut self, at_utc: DateTime<Utc>, event: &ServiceEvent) {
        let Some(message) = event_message(event) else {
            return;
        };

        for (index, attempt) in self.attempts.iter_mut().enumerate() {
            if !message.contains(&attempt.report.reason_tag) {
                continue;
            }
            if !message.contains("submitted:") {
                if matches!(event, ServiceEvent::Status(_))
                    && !attempt
                        .report
                        .dispatch_notes
                        .iter()
                        .any(|note| note == message)
                {
                    attempt.report.dispatch_notes.push(message.to_string());
                }
                continue;
            }
            if attempt.report.submit.is_none() {
                attempt.report.send_to_submit_event_ms =
                    Some(attempt.sent_at.elapsed().as_millis() as u64);
                attempt.report.submit = Some(SwipeSubmitObservation {
                    received_at_utc: at_utc,
                    submit_message: message.to_string(),
                    broker_submit_ms: None,
                    request_id: parse_submit_request_id(message),
                    seen_ms: None,
                    exec_report_ms: None,
                    fill_ms: None,
                });
                if let Some(request_id) = attempt
                    .report
                    .submit
                    .as_ref()
                    .and_then(|submit| submit.request_id.clone())
                {
                    self.request_to_index.insert(request_id, index);
                }
            }
            if let (ServiceEvent::DebugLog(debug_message), Some(submit)) =
                (event, attempt.report.submit.as_mut())
            {
                if submit.broker_submit_ms.is_none() {
                    submit.broker_submit_ms = parse_debug_stage_ms(debug_message, "submit");
                }
                if submit.request_id.is_none() {
                    submit.request_id = parse_submit_request_id(debug_message);
                    if let Some(request_id) = submit.request_id.clone() {
                        self.request_to_index.insert(request_id, index);
                    }
                }
            }
        }

        if let ServiceEvent::DebugLog(debug_message) = event {
            let Some(request_id) = parse_stage_request_id(debug_message) else {
                return;
            };
            let Some(index) = self.request_to_index.get(&request_id).copied() else {
                return;
            };
            let Some(submit) = self.attempts[index].report.submit.as_mut() else {
                return;
            };
            if submit.seen_ms.is_none() {
                submit.seen_ms = parse_debug_stage_ms(debug_message, "seen");
            }
            if submit.exec_report_ms.is_none() {
                submit.exec_report_ms = parse_debug_stage_ms(debug_message, "ack");
            }
            if submit.fill_ms.is_none() {
                submit.fill_ms = parse_debug_stage_ms(debug_message, "fill");
            }
        }
    }

    pub(super) fn set_delay_probe(
        &mut self,
        index: usize,
        probe: ExecutionProbeSnapshot,
        findings: Vec<String>,
    ) {
        if let Some(attempt) = self.attempts.get_mut(index) {
            attempt.report.delay_probe = Some(probe);
            attempt.report.delay_findings = findings;
        }
    }

    pub(super) fn submit_for(&self, index: usize) -> Option<SwipeSubmitObservation> {
        self.attempts
            .get(index)
            .and_then(|attempt| attempt.report.submit.clone())
    }

    pub(super) fn dispatch_notes_for(&self, index: usize) -> Vec<String> {
        self.attempts
            .get(index)
            .map(|attempt| attempt.report.dispatch_notes.clone())
            .unwrap_or_default()
    }

    pub(super) fn report_clone(&self, index: usize) -> Option<SwipeAttemptReport> {
        self.attempts.get(index).map(|attempt| attempt.report.clone())
    }

    pub(super) fn has_submit_for_reason(&self, reason_tag: &str) -> bool {
        self.find_by_reason(reason_tag)
            .and_then(|attempt| attempt.report.submit.as_ref())
            .is_some()
    }

    pub(super) fn last_dispatch_note_for_reason(&self, reason_tag: &str) -> Option<String> {
        self.find_by_reason(reason_tag)
            .and_then(|attempt| attempt.report.dispatch_notes.last().cloned())
    }

    fn find_by_reason(&self, reason_tag: &str) -> Option<&LiveAttemptState> {
        self.attempts
            .iter()
            .find(|attempt| attempt.report.reason_tag == reason_tag)
    }
}

fn parse_submit_request_id(message: &str) -> Option<String> {
    bracketed_value(message, "uuid ").or_else(|| bracketed_value(message, "clOrdId "))
}

fn parse_stage_request_id(message: &str) -> Option<String> {
    bracketed_value(message, "request ")
}

fn bracketed_value(message: &str, prefix: &str) -> Option<String> {
    let needle = format!("[{prefix}");
    let start = message.find(&needle)? + needle.len();
    let tail = &message[start..];
    let end = tail.find(']')?;
    Some(tail[..end].to_string())
}

fn parse_debug_stage_ms(message: &str, stage: &str) -> Option<u64> {
    let prefix = format!("{stage} ");
    let tail = message.strip_prefix(&prefix)?;
    let token = tail.split_whitespace().next()?;
    parse_duration_token_ms(token)
}

fn parse_duration_token_ms(token: &str) -> Option<u64> {
    if let Some(raw) = token.strip_suffix("ms") {
        return raw.parse::<f64>().ok().map(|value| value.round() as u64);
    }
    if let Some(raw) = token.strip_suffix('s') {
        return raw
            .parse::<f64>()
            .ok()
            .map(|value| (value * 1000.0).round() as u64);
    }
    if let Some(raw) = token.strip_suffix('m') {
        return raw
            .parse::<f64>()
            .ok()
            .map(|value| (value * 60_000.0).round() as u64);
    }
    None
}

fn event_message(event: &ServiceEvent) -> Option<&str> {
    match event {
        ServiceEvent::Status(message)
        | ServiceEvent::DebugLog(message)
        | ServiceEvent::Error(message) => Some(message),
        _ => None,
    }
}

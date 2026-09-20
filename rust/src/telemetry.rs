//! Deidentified app telemetry kept separate from private cognitive memory.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Cosmos container for app-level operational events.
pub const TELEMETRY_CONTAINER: &str = "GaiaTelemetry";
/// Elle's application partition within the shared telemetry container.
pub const TELEMETRY_SCOPE: &str = "elle";

/// One deidentified capability execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ToolTelemetry {
    /// Stable capability name without arguments or results.
    pub name: String,
    /// Whether the capability completed successfully.
    pub succeeded: bool,
    /// Rounded wall-clock execution duration.
    pub duration_ms: u64,
}

/// Strict bridge payload for one completed hosted turn.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryRequest {
    /// Stable hosted response identifier used only to derive the event ID.
    pub response_id: String,
    /// UTC completion timestamp.
    pub timestamp: String,
    /// Hosted model or runtime label.
    pub model: String,
    /// Number of model requests used for the turn.
    pub model_requests: usize,
    /// Deidentified capability execution records.
    pub tools: Vec<ToolTelemetry>,
    /// Number of Unicode scalar values in the user's input.
    pub input_chars: usize,
    /// Number of Unicode scalar values in the visible reply.
    pub reply_chars: usize,
    /// Whether the private turn lifecycle completed successfully.
    pub persisted: bool,
    /// Whether the hosted turn produced a terminal response.
    pub completed: bool,
}

/// Persisted app-level event with no user-authored text or identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TelemetryEvent {
    /// Opaque deterministic event identifier.
    pub id: String,
    /// Fixed app-level Cosmos partition.
    pub scope: String,
    /// UTC completion timestamp.
    pub timestamp: String,
    /// Hosted model or runtime label.
    pub model: String,
    /// Number of model requests used for the turn.
    pub model_requests: usize,
    /// Deidentified capability execution records.
    pub tools: Vec<ToolTelemetry>,
    /// Coarse content-free interaction dimensions.
    pub interaction: InteractionTelemetry,
    /// Whether the hosted turn produced a terminal response.
    pub completed: bool,
}

/// Coarse interaction dimensions with no human-authored content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InteractionTelemetry {
    /// Number of Unicode scalar values in the user's input.
    pub input_chars: usize,
    /// Number of Unicode scalar values in the visible reply.
    pub reply_chars: usize,
    /// Whether the private turn lifecycle completed successfully.
    pub persisted: bool,
}

/// Storage boundary for telemetry's `/scope` partition.
pub trait TelemetryRepository: Send {
    /// Create an immutable event, returning false for an existing ID.
    fn create(&mut self, event: &TelemetryEvent) -> Result<bool>;
}

/// Validates and records deidentified telemetry events.
pub struct TelemetryService {
    repository: Box<dyn TelemetryRepository>,
}

impl TelemetryService {
    /// Construct telemetry over an explicit storage backend.
    pub fn new(repository: Box<dyn TelemetryRepository>) -> Self {
        Self { repository }
    }

    /// Validate and idempotently record one completed turn event.
    pub fn record(&mut self, request: TelemetryRequest) -> Result<bool> {
        if request.response_id.trim().is_empty()
            || request.response_id.len() > 512
            || request.response_id.chars().any(char::is_control)
            || request.timestamp.len() < 20
            || !request.timestamp.ends_with('Z')
            || request.model.trim().is_empty()
            || request.model.len() > 128
            || request.tools.len() > 64
            || request.tools.iter().any(|tool| {
                tool.name.trim().is_empty()
                    || tool.name.len() > 128
                    || tool.name.chars().any(char::is_control)
            })
        {
            return Err(Error::InvalidInput("Invalid telemetry event"));
        }
        let id = format!(
            "telemetry-{:x}",
            Sha256::digest(request.response_id.as_bytes())
        );
        self.repository.create(&TelemetryEvent {
            id,
            scope: TELEMETRY_SCOPE.to_owned(),
            timestamp: request.timestamp,
            model: request.model,
            model_requests: request.model_requests,
            tools: request.tools,
            interaction: InteractionTelemetry {
                input_chars: request.input_chars,
                reply_chars: request.reply_chars,
                persisted: request.persisted,
            },
            completed: request.completed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct MemoryRepository(BTreeMap<String, TelemetryEvent>);

    impl TelemetryRepository for MemoryRepository {
        fn create(&mut self, event: &TelemetryEvent) -> Result<bool> {
            if self.0.contains_key(&event.id) {
                return Ok(false);
            }
            self.0.insert(event.id.clone(), event.clone());
            Ok(true)
        }
    }

    fn request() -> TelemetryRequest {
        TelemetryRequest {
            response_id: "response-1".to_owned(),
            timestamp: "2026-09-20T12:00:00Z".to_owned(),
            model: "foundry-agent".to_owned(),
            model_requests: 1,
            tools: Vec::new(),
            input_chars: 12,
            reply_chars: 24,
            persisted: true,
            completed: true,
        }
    }

    #[test]
    fn telemetry_is_deidentified_and_retry_safe() {
        let mut service = TelemetryService::new(Box::new(MemoryRepository::default()));
        assert!(service.record(request()).unwrap());
        assert!(!service.record(request()).unwrap());
    }
}

//! Encrypted, owner-scoped cognitive stores with fixed container routing.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::embeddings::Embedder;
use crate::encryption::FieldCipher;
use crate::error::{Error, Result};
use crate::identity::OwnerId;

const MAX_TEXT_BYTES: usize = 16 * 1024;
const DAILY_REPLACE_ATTEMPTS: usize = 4;

/// Internal medium retrieval default for public web results.
pub const DEFAULT_WEB_TOP: usize = 3;
/// Internal medium retrieval default for conversation history.
pub const DEFAULT_DATA_LAKE_TOP: usize = 2;
/// Internal medium retrieval default for durable facts.
pub const DEFAULT_KNOWLEDGE_BASE_TOP: usize = 12;
/// Internal medium retrieval default for diary reflections.
pub const DEFAULT_DIARY_TOP: usize = 12;
/// Internal medium retrieval default for connection entries.
pub const DEFAULT_CONNECTIONS_TOP: usize = 3;
/// Internal medium retrieval default for reviewed wisdom.
pub const DEFAULT_WISDOM_TOP: usize = 77;
/// Maximum reviewed-wisdom retrieval bound.
pub const MAX_WISDOM_TOP: usize = 7_777;

/// Fixed cognitive stores available to the model-facing bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveStore {
    /// Completed user and assistant turns, one document per owner and UTC day.
    DataLake,
    /// Durable facts, one canonical fact per document.
    KnowledgeBase,
    /// Deliberate reflections, one document per owner and UTC day.
    Diary,
    /// Read-only relationship ledger entries.
    Connections,
}

impl CognitiveStore {
    /// Return the fixed Cosmos container for this store.
    pub fn container(self) -> &'static str {
        match self {
            Self::DataLake => "GaiaDataLake",
            Self::KnowledgeBase => "GaiaKB",
            Self::Diary => "GaiaDiary",
            Self::Connections => "GaiaConnections",
        }
    }

    /// Return the internal medium result bound for this store.
    pub fn default_top(self) -> usize {
        match self {
            Self::DataLake => DEFAULT_DATA_LAKE_TOP,
            Self::KnowledgeBase => DEFAULT_KNOWLEDGE_BASE_TOP,
            Self::Diary => DEFAULT_DIARY_TOP,
            Self::Connections => DEFAULT_CONNECTIONS_TOP,
        }
    }
}

/// Retrieval strategy selected from a closed set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalMode {
    /// Semantic when embeddings are configured, otherwise keyword; chronological without text.
    Auto,
    /// Date/order retrieval without text ranking.
    Chronological,
    /// Case-insensitive token matching.
    Keyword,
    /// Token-overlap relevance ranking over decrypted owner records.
    Semantic,
}

/// Chronological tie-breaking order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalOrder {
    /// Most recent records first.
    Newest,
    /// Oldest records first.
    Oldest,
}

/// Strict model-facing retrieval options.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveQuery {
    /// Store selected from the fixed allow-list.
    pub store: CognitiveStore,
    /// Retrieval strategy.
    pub mode: RetrievalMode,
    /// Optional search text.
    #[serde(default)]
    pub query: Option<String>,
    /// Date or timestamp lower bound, inclusive.
    #[serde(default)]
    pub from: Option<String>,
    /// Date or timestamp upper bound, inclusive.
    #[serde(default)]
    pub to: Option<String>,
    /// Date tie-breaking order.
    pub order: RetrievalOrder,
    /// Requested result bound, clamped to 1..=100.
    pub top: usize,
    /// Return only the matching count.
    #[serde(default)]
    pub count_only: bool,
}

/// One encrypted cognitive document persisted in an owner partition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CognitiveRecord {
    /// Stable identifier within the owner partition.
    pub id: String,
    /// Server-derived immutable partition key.
    pub owner_id: String,
    /// Fixed store discriminator.
    pub store: CognitiveStore,
    /// UTC date or instant used for filtering and ordering.
    pub occurred_at: String,
    /// Application-encrypted JSON payload.
    pub ciphertext: String,
    /// Compare-and-swap version.
    pub version: u64,
}

/// Persistence boundary for fixed cognitive containers.
pub trait CognitiveRepository {
    /// Read one owner-scoped record.
    fn get(
        &self,
        store: CognitiveStore,
        owner_id: &str,
        id: &str,
    ) -> Result<Option<CognitiveRecord>>;
    /// List bounded records from one fixed store and owner partition.
    fn list(&self, store: CognitiveStore, owner_id: &str) -> Result<Vec<CognitiveRecord>>;
    /// Create a record, returning false if its id already exists.
    fn create(&mut self, record: &CognitiveRecord) -> Result<bool>;
    /// Replace a record only at the expected version.
    fn replace(&mut self, record: &CognitiveRecord, expected_version: u64) -> Result<()>;
}

/// Decrypted query result returned to the trusted agent host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CognitiveItem {
    /// Stable record identifier.
    pub id: String,
    /// Record date or timestamp.
    pub occurred_at: String,
    /// Decrypted record text.
    pub text: String,
}

/// Bounded query response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CognitiveQueryResult {
    /// Total matching records before the top bound.
    pub count: usize,
    /// Empty for count-only requests.
    pub items: Vec<CognitiveItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DailyPayload {
    entries: Vec<DailyEntry>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DailyEntry {
    key: String,
    timestamp: String,
    text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactPayload {
    canonical: String,
    text: String,
    salience: f32,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConnectionPayload {
    change_amount: f64,
    previous_balance: f64,
    new_balance: f64,
    note: String,
}

/// Result of an idempotent append to the relationship ledger.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConnectionOutcome {
    /// Whether this call created a new ledger event.
    pub stored: bool,
    /// Running balance before the event.
    pub previous_balance: f64,
    /// Running balance after the event.
    pub new_balance: f64,
}

/// Enforces cognitive write and retrieval policy above any storage backend.
pub struct CognitiveService {
    repository: Box<dyn CognitiveRepository>,
    cipher: FieldCipher,
    embedder: Option<Box<dyn Embedder>>,
}

impl CognitiveService {
    /// Construct a service with mandatory field encryption.
    pub fn new(repository: Box<dyn CognitiveRepository>, cipher: FieldCipher) -> Self {
        Self {
            repository,
            cipher,
            embedder: None,
        }
    }

    /// Attach the provider used for encrypted semantic vectors.
    pub fn with_embedder(mut self, embedder: Option<Box<dyn Embedder>>) -> Self {
        self.embedder = embedder;
        self
    }

    /// Archive one completed turn into its UTC-day document.
    pub fn archive_turn(
        &mut self,
        owner: &OwnerId,
        timestamp: &str,
        user_text: &str,
        assistant_text: &str,
    ) -> Result<bool> {
        let text = format!("User:\n{user_text}\n\nElle:\n{assistant_text}");
        self.append_daily(CognitiveStore::DataLake, owner, timestamp, &text)
    }

    /// Deliberately save one durable fact with canonical-content dedupe.
    pub fn save_knowledge(
        &mut self,
        owner: &OwnerId,
        timestamp: &str,
        text: &str,
        salience: f32,
    ) -> Result<bool> {
        validate_timestamp(timestamp)?;
        validate_text(text)?;
        if !salience.is_finite() || !(0.0..=1.0).contains(&salience) {
            return Err(Error::InvalidInput(
                "Knowledge salience must be from 0 to 1",
            ));
        }
        let canonical = canonical_text(text);
        let id = format!("kb-{}", digest(&canonical));
        if self
            .repository
            .get(CognitiveStore::KnowledgeBase, owner.as_str(), &id)?
            .is_some()
        {
            return Ok(false);
        }
        let payload = serde_json::to_vec(&FactPayload {
            canonical,
            text: text.trim().to_owned(),
            salience,
            embedding: self.embedding(text)?,
        })
        .map_err(|_| Error::Integrity("Cannot encode cognitive payload"))?;
        let record = CognitiveRecord {
            id,
            owner_id: owner.as_str().to_owned(),
            store: CognitiveStore::KnowledgeBase,
            occurred_at: timestamp.to_owned(),
            ciphertext: String::new(),
            version: 1,
        };
        self.create_encrypted(record, &payload)
    }

    /// Deliberately append one reflection to its UTC-day diary document.
    pub fn save_diary(&mut self, owner: &OwnerId, timestamp: &str, text: &str) -> Result<bool> {
        self.append_daily(CognitiveStore::Diary, owner, timestamp, text)
    }

    /// Append one encrypted relationship delta, deduplicated by a host event key.
    pub fn save_connection(
        &mut self,
        owner: &OwnerId,
        timestamp: &str,
        event_key: &str,
        change_amount: f64,
        note: &str,
    ) -> Result<ConnectionOutcome> {
        validate_timestamp(timestamp)?;
        validate_text(note)?;
        if event_key.trim().is_empty()
            || event_key.len() > 512
            || event_key.chars().any(char::is_control)
            || !change_amount.is_finite()
        {
            return Err(Error::InvalidInput("Invalid connection ledger event"));
        }
        let id = format!("connection-{}", digest(event_key));
        if let Some(record) = self.repository.get(
            CognitiveStore::Connections,
            owner.as_str(),
            &id,
        )? {
            let payload = self.decrypt_connection(&record)?;
            return Ok(ConnectionOutcome {
                stored: false,
                previous_balance: payload.previous_balance,
                new_balance: payload.new_balance,
            });
        }
        let mut latest: Option<(String, f64)> = None;
        for record in self
            .repository
            .list(CognitiveStore::Connections, owner.as_str())?
        {
            validate_record(&record, CognitiveStore::Connections, owner.as_str())?;
            let balance = self.decrypt_connection(&record)?.new_balance;
            if latest
                .as_ref()
                .is_none_or(|(occurred_at, _)| record.occurred_at > *occurred_at)
            {
                latest = Some((record.occurred_at, balance));
            }
        }
        let previous_balance = latest.map_or(0.0, |(_, balance)| balance);
        let new_balance = previous_balance + change_amount;
        if !new_balance.is_finite() {
            return Err(Error::InvalidInput("Invalid connection ledger balance"));
        }
        let payload = serde_json::to_vec(&ConnectionPayload {
            change_amount,
            previous_balance,
            new_balance,
            note: note.trim().to_owned(),
        })
        .map_err(|_| Error::Integrity("Cannot encode cognitive payload"))?;
        let stored = self.create_encrypted(
            CognitiveRecord {
                id,
                owner_id: owner.as_str().to_owned(),
                store: CognitiveStore::Connections,
                occurred_at: timestamp.to_owned(),
                ciphertext: String::new(),
                version: 1,
            },
            &payload,
        )?;
        Ok(ConnectionOutcome {
            stored,
            previous_balance,
            new_balance,
        })
    }

    /// Query one fixed owner-scoped store with bounded options.
    pub fn query(
        &self,
        owner: &OwnerId,
        mut query: CognitiveQuery,
    ) -> Result<CognitiveQueryResult> {
        query.top = query.top.clamp(1, 100);
        validate_bound(query.from.as_deref())?;
        validate_bound(query.to.as_deref())?;
        if query
            .from
            .as_deref()
            .zip(query.to.as_deref())
            .is_some_and(|(from, to)| from > to)
        {
            return Err(Error::InvalidInput("Invalid cognitive date range"));
        }
        let search = query
            .query
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mode = match query.mode {
            RetrievalMode::Auto
                if search.is_some()
                    && self.embedder.is_some()
                    && query.store != CognitiveStore::Connections =>
            {
                RetrievalMode::Semantic
            }
            RetrievalMode::Auto if search.is_some() => RetrievalMode::Keyword,
            RetrievalMode::Auto => RetrievalMode::Chronological,
            mode => mode,
        };
        if matches!(mode, RetrievalMode::Keyword | RetrievalMode::Semantic) && search.is_none() {
            return Err(Error::InvalidInput(
                "Search text is required for this retrieval mode",
            ));
        }
        let query_embedding = if mode == RetrievalMode::Semantic {
            self.embedding(search.expect("validated"))?
                .ok_or(Error::Configuration(
                    "Semantic cognitive retrieval is unavailable",
                ))?
                .into()
        } else {
            None
        };
        let mut scored = Vec::new();
        for record in self.repository.list(query.store, owner.as_str())? {
            validate_record(&record, query.store, owner.as_str())?;
            if query
                .from
                .as_deref()
                .is_some_and(|from| record.occurred_at.as_str() < from)
                || query
                    .to
                    .as_deref()
                    .is_some_and(|to| record.occurred_at.as_str() > to)
            {
                continue;
            }
            let (text, embedding, salience) = self.decrypt_payload(&record)?;
            let score = match mode {
                RetrievalMode::Chronological => 0.0,
                RetrievalMode::Keyword => keyword_score(&text, search.expect("validated")) as f32,
                RetrievalMode::Semantic => embedding
                    .as_deref()
                    .and_then(|embedding| {
                        cosine_similarity(
                            embedding,
                            query_embedding
                                .as_deref()
                                .expect("semantic embedding exists"),
                        )
                    })
                    .map(|similarity| similarity * salience)
                    .unwrap_or(0.0),
                RetrievalMode::Auto => unreachable!("auto mode resolved"),
            };
            if matches!(mode, RetrievalMode::Keyword | RetrievalMode::Semantic) && score <= 0.0 {
                continue;
            }
            scored.push((
                score,
                CognitiveItem {
                    id: record.id,
                    occurred_at: record.occurred_at,
                    text,
                },
            ));
        }
        scored.sort_by(|left, right| {
            if matches!(mode, RetrievalMode::Keyword | RetrievalMode::Semantic) {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| right.1.occurred_at.cmp(&left.1.occurred_at))
            } else {
                match query.order {
                    RetrievalOrder::Newest => right.1.occurred_at.cmp(&left.1.occurred_at),
                    RetrievalOrder::Oldest => left.1.occurred_at.cmp(&right.1.occurred_at),
                }
            }
        });
        let count = scored.len();
        let items = if query.count_only {
            Vec::new()
        } else {
            scored
                .into_iter()
                .take(query.top)
                .map(|(_, item)| item)
                .collect()
        };
        Ok(CognitiveQueryResult { count, items })
    }

    fn append_daily(
        &mut self,
        store: CognitiveStore,
        owner: &OwnerId,
        timestamp: &str,
        text: &str,
    ) -> Result<bool> {
        validate_timestamp(timestamp)?;
        validate_text(text)?;
        let day = &timestamp[..10];
        let id = format!(
            "{}-{day}",
            match store {
                CognitiveStore::DataLake => "turns",
                CognitiveStore::Diary => "diary",
                _ => return Err(Error::InvalidInput("Store is not a daily cognitive log")),
            }
        );
        let key = digest(text);
        for _ in 0..DAILY_REPLACE_ATTEMPTS {
            let existing = self.repository.get(store, owner.as_str(), &id)?;
            let mut payload = match existing.as_ref() {
                Some(record) => serde_json::from_slice::<DailyPayload>(&self.cipher.decrypt(
                    owner.as_str(),
                    &id,
                    &record.ciphertext,
                )?)
                .map_err(|_| Error::Integrity("Invalid cognitive payload"))?,
                None => DailyPayload {
                    entries: Vec::new(),
                    embedding: None,
                },
            };
            if payload.entries.iter().any(|entry| entry.key == key) {
                return Ok(false);
            }
            payload.entries.push(DailyEntry {
                key: key.clone(),
                timestamp: timestamp.to_owned(),
                text: text.trim().to_owned(),
            });
            let transcript = payload
                .entries
                .iter()
                .map(|entry| entry.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            payload.embedding = self.embedding(&transcript)?;
            let plaintext = serde_json::to_vec(&payload)
                .map_err(|_| Error::Integrity("Cannot encode cognitive payload"))?;
            match existing {
                None => {
                    if self.create_encrypted(
                        CognitiveRecord {
                            id: id.clone(),
                            owner_id: owner.as_str().to_owned(),
                            store,
                            occurred_at: day.to_owned(),
                            ciphertext: String::new(),
                            version: 1,
                        },
                        &plaintext,
                    )? {
                        return Ok(true);
                    }
                }
                Some(mut record) => {
                    let expected = record.version;
                    record.version = expected.checked_add(1).ok_or(Error::Conflict)?;
                    record.ciphertext =
                        self.cipher
                            .encrypt(owner.as_str(), &record.id, &plaintext)?;
                    match self.repository.replace(&record, expected) {
                        Ok(()) => return Ok(true),
                        Err(Error::Conflict) => continue,
                        Err(error) => return Err(error),
                    }
                }
            }
        }
        Err(Error::Conflict)
    }

    fn create_encrypted(&mut self, mut record: CognitiveRecord, plaintext: &[u8]) -> Result<bool> {
        record.ciphertext = self
            .cipher
            .encrypt(&record.owner_id, &record.id, plaintext)?;
        self.repository.create(&record)
    }

    fn decrypt_payload(&self, record: &CognitiveRecord) -> Result<(String, Option<Vec<f32>>, f32)> {
        let plaintext = self
            .cipher
            .decrypt(&record.owner_id, &record.id, &record.ciphertext)?;
        match record.store {
            CognitiveStore::KnowledgeBase => serde_json::from_slice::<FactPayload>(&plaintext)
                .map(|payload| (payload.text, payload.embedding, payload.salience))
                .map_err(|_| Error::Integrity("Invalid cognitive payload")),
            CognitiveStore::DataLake | CognitiveStore::Diary => {
                serde_json::from_slice::<DailyPayload>(&plaintext)
                    .map(|payload| {
                        (
                            payload
                                .entries
                                .into_iter()
                                .map(|entry| entry.text)
                                .collect::<Vec<_>>()
                                .join("\n\n"),
                            payload.embedding,
                            1.0,
                        )
                    })
                    .map_err(|_| Error::Integrity("Invalid cognitive payload"))
            }
            CognitiveStore::Connections => serde_json::from_slice::<ConnectionPayload>(&plaintext)
                .and_then(|payload| {
                    serde_json::to_string(&payload).map(|text| (text, None, 1.0))
                })
                .map_err(|_| Error::Integrity("Invalid cognitive payload")),
        }
    }

    fn decrypt_connection(&self, record: &CognitiveRecord) -> Result<ConnectionPayload> {
        let plaintext = self
            .cipher
            .decrypt(&record.owner_id, &record.id, &record.ciphertext)?;
        serde_json::from_slice(&plaintext)
            .map_err(|_| Error::Integrity("Invalid connection payload"))
    }

    fn embedding(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let Some(embedder) = self.embedder.as_ref() else {
            return Ok(None);
        };
        let embedding = embedder.embed(text)?;
        if embedding.is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(Error::Integrity(
                "Embedding provider returned an invalid vector",
            ));
        }
        Ok(Some(embedding))
    }
}

fn validate_record(record: &CognitiveRecord, store: CognitiveStore, owner: &str) -> Result<()> {
    if record.owner_id != owner
        || record.store != store
        || record.version == 0
        || record.ciphertext.is_empty()
    {
        return Err(Error::Unauthorized);
    }
    Ok(())
}

fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty()
        || text.len() > MAX_TEXT_BYTES
        || text.chars().any(|character| character == '\0')
    {
        return Err(Error::InvalidInput(
            "Cognitive text is empty or exceeds the size limit",
        ));
    }
    Ok(())
}

fn validate_timestamp(value: &str) -> Result<()> {
    if value.len() < 20
        || value.as_bytes().get(4) != Some(&b'-')
        || value.as_bytes().get(7) != Some(&b'-')
        || value.as_bytes().get(10) != Some(&b'T')
        || !value.ends_with('Z')
    {
        return Err(Error::InvalidInput("Timestamp must be UTC RFC3339"));
    }
    Ok(())
}

fn validate_bound(value: Option<&str>) -> Result<()> {
    if value.is_some_and(|value| {
        value.len() < 10 || value.len() > 32 || value.chars().any(char::is_control)
    }) {
        return Err(Error::InvalidInput("Invalid cognitive date bound"));
    }
    Ok(())
}

fn canonical_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['.', '!', '?', ',', ';', ':'])
        .to_lowercase()
}

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn tokens(text: &str) -> BTreeSet<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn keyword_score(text: &str, query: &str) -> usize {
    let normalized = text.to_lowercase();
    tokens(query)
        .into_iter()
        .filter(|token| normalized.contains(token))
        .count()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let dot = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    (left_norm > 0.0 && right_norm > 0.0).then_some(dot / (left_norm * right_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct MemoryCognitiveRepository {
        records: BTreeMap<(CognitiveStore, String, String), CognitiveRecord>,
    }

    impl CognitiveRepository for MemoryCognitiveRepository {
        fn get(
            &self,
            store: CognitiveStore,
            owner_id: &str,
            id: &str,
        ) -> Result<Option<CognitiveRecord>> {
            Ok(self
                .records
                .get(&(store, owner_id.to_owned(), id.to_owned()))
                .cloned())
        }

        fn list(&self, store: CognitiveStore, owner_id: &str) -> Result<Vec<CognitiveRecord>> {
            let records = self
                .records
                .values()
                .filter(|record| record.store == store && record.owner_id == owner_id)
                .cloned()
                .collect::<Vec<_>>();
            if records.len() > 1_000 {
                return Err(Error::Storage("Cognitive listing exceeds its bound"));
            }
            Ok(records)
        }

        fn create(&mut self, record: &CognitiveRecord) -> Result<bool> {
            let key = (record.store, record.owner_id.clone(), record.id.clone());
            if self.records.contains_key(&key) {
                return Ok(false);
            }
            self.records.insert(key, record.clone());
            Ok(true)
        }

        fn replace(&mut self, record: &CognitiveRecord, expected_version: u64) -> Result<()> {
            let key = (record.store, record.owner_id.clone(), record.id.clone());
            let current = self.records.get(&key).ok_or(Error::NotFound)?;
            if current.version != expected_version || record.version != expected_version + 1 {
                return Err(Error::Conflict);
            }
            self.records.insert(key, record.clone());
            Ok(())
        }
    }

    fn owner(id: &str) -> OwnerId {
        OwnerId::new("11111111-1111-1111-1111-111111111111", id).unwrap()
    }

    fn service() -> CognitiveService {
        CognitiveService::new(
            Box::new(MemoryCognitiveRepository::default()),
            FieldCipher::new([9; 32]),
        )
        .with_embedder(Some(Box::new(TestEmbedder)))
    }

    struct TestEmbedder;

    impl Embedder for TestEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let normalized = text.to_lowercase();
            Ok(vec![
                normalized.matches("rust").count() as f32,
                normalized.matches("python").count() as f32,
                normalized.matches("bridge").count() as f32,
            ])
        }
    }

    fn chronological(store: CognitiveStore) -> CognitiveQuery {
        CognitiveQuery {
            store,
            mode: RetrievalMode::Chronological,
            query: None,
            from: None,
            to: None,
            order: RetrievalOrder::Oldest,
            top: 100,
            count_only: false,
        }
    }

    #[test]
    fn archived_turn_retry_dedupes_distinct_turn_appends_and_owner_day_isolates() {
        let mut service = service();
        let first = owner("22222222-2222-2222-2222-222222222222");
        let second = owner("33333333-3333-3333-3333-333333333333");
        assert!(service
            .archive_turn(&first, "2026-09-20T09:00:00Z", "hello", "hi")
            .unwrap());
        assert!(!service
            .archive_turn(&first, "2026-09-20T10:00:00Z", "hello", "hi")
            .unwrap());
        assert!(service
            .archive_turn(&first, "2026-09-20T10:01:00Z", "next", "answer")
            .unwrap());
        assert!(service
            .archive_turn(&first, "2026-09-21T00:00:00Z", "tomorrow", "yes")
            .unwrap());
        assert!(service
            .archive_turn(&second, "2026-09-20T09:00:00Z", "private", "reply")
            .unwrap());
        let first_rows = service
            .query(&first, chronological(CognitiveStore::DataLake))
            .unwrap();
        assert_eq!(first_rows.count, 2);
        assert_eq!(first_rows.items[0].id, "turns-2026-09-20");
        assert_eq!(first_rows.items[1].id, "turns-2026-09-21");
        assert_eq!(first_rows.items[0].text.matches("User:\nhello").count(), 1);
        assert!(first_rows.items[0]
            .text
            .contains("User:\nnext\n\nElle:\nanswer"));
        assert!(first_rows.items[0].text.contains("next"));
        assert!(!first_rows
            .items
            .iter()
            .any(|item| item.text.contains("private")));
    }

    #[test]
    fn knowledge_is_canonical_and_diary_retries_are_deduped() {
        let mut service = service();
        let owner = owner("22222222-2222-2222-2222-222222222222");
        assert!(service
            .save_knowledge(
                &owner,
                "2026-09-20T09:00:00Z",
                "Prefers  green dashboards.",
                0.8
            )
            .unwrap());
        assert!(!service
            .save_knowledge(
                &owner,
                "2026-09-20T10:00:00Z",
                " prefers GREEN dashboards! ",
                0.9
            )
            .unwrap());
        assert!(service
            .save_diary(&owner, "2026-09-20T11:00:00Z", "A useful reflection")
            .unwrap());
        assert!(!service
            .save_diary(&owner, "2026-09-20T12:00:00Z", "A useful reflection")
            .unwrap());
        assert!(service
            .save_diary(&owner, "2026-09-20T12:01:00Z", "a useful reflection")
            .unwrap());
        assert_eq!(
            service
                .query(&owner, chronological(CognitiveStore::KnowledgeBase))
                .unwrap()
                .count,
            1
        );
        let diary = service
            .query(&owner, chronological(CognitiveStore::Diary))
            .unwrap();
        assert_eq!(diary.count, 1);
        assert!(diary.items[0]
            .text
            .contains("A useful reflection\n\na useful reflection"));
    }

    #[test]
    fn connection_ledger_carries_balance_and_dedupes_host_event() {
        let mut service = service();
        let owner = owner("22222222-2222-2222-2222-222222222222");
        let first = service
            .save_connection(
                &owner,
                "2026-09-20T11:00:00Z",
                "response-1",
                0.4,
                "Trust increased after a clear correction.",
            )
            .unwrap();
        assert_eq!(first, ConnectionOutcome {
            stored: true,
            previous_balance: 0.0,
            new_balance: 0.4,
        });
        let retry = service
            .save_connection(
                &owner,
                "2026-09-20T11:00:01Z",
                "response-1",
                0.4,
                "Trust increased after a clear correction.",
            )
            .unwrap();
        assert_eq!(retry, ConnectionOutcome {
            stored: false,
            previous_balance: 0.0,
            new_balance: 0.4,
        });
        let second = service
            .save_connection(
                &owner,
                "2026-09-20T12:00:00Z",
                "response-2",
                -0.1,
                "A small misunderstanding was repaired.",
            )
            .unwrap();
        assert!(second.stored);
        assert!((second.previous_balance - 0.4).abs() < f64::EPSILON);
        assert!((second.new_balance - 0.3).abs() < f64::EPSILON);
        let rows = service
            .query(&owner, chronological(CognitiveStore::Connections))
            .unwrap();
        assert_eq!(rows.count, 2);
        assert!(rows.items[1].text.contains("\"new_balance\":0.30000000000000004"));
    }

    #[test]
    fn retrieval_modes_order_dates_count_bounds_and_owner_isolation() {
        let mut service = service();
        let first = owner("22222222-2222-2222-2222-222222222222");
        let second = owner("33333333-3333-3333-3333-333333333333");
        service
            .save_knowledge(&first, "2026-09-19T09:00:00Z", "Rust storage design", 0.8)
            .unwrap();
        service
            .save_knowledge(&first, "2026-09-20T09:00:00Z", "Python bridge design", 0.7)
            .unwrap();
        service
            .save_knowledge(&first, "2026-09-21T09:00:00Z", "Rust language", 0.9)
            .unwrap();
        service
            .save_knowledge(&second, "2026-09-20T09:00:00Z", "private Rust note", 0.6)
            .unwrap();
        let mut query = chronological(CognitiveStore::KnowledgeBase);
        query.order = RetrievalOrder::Newest;
        query.top = 0;
        assert_eq!(
            service.query(&first, query).unwrap().items[0].text,
            "Rust language"
        );
        let result = service
            .query(
                &first,
                CognitiveQuery {
                    store: CognitiveStore::KnowledgeBase,
                    mode: RetrievalMode::Auto,
                    query: Some("rust".to_owned()),
                    from: Some("2026-09-19".to_owned()),
                    to: Some("2026-09-19T23:59:59Z".to_owned()),
                    order: RetrievalOrder::Newest,
                    top: 999,
                    count_only: true,
                },
            )
            .unwrap();
        assert_eq!(result.count, 1);
        assert!(result.items.is_empty());
        let semantic = service
            .query(
                &first,
                CognitiveQuery {
                    store: CognitiveStore::KnowledgeBase,
                    mode: RetrievalMode::Semantic,
                    query: Some("bridge python".to_owned()),
                    from: None,
                    to: None,
                    order: RetrievalOrder::Oldest,
                    top: 10,
                    count_only: false,
                },
            )
            .unwrap();
        assert_eq!(semantic.items.len(), 1);
        assert!(!semantic.items[0].text.contains("private"));
        let salient = service
            .query(
                &first,
                CognitiveQuery {
                    store: CognitiveStore::KnowledgeBase,
                    mode: RetrievalMode::Semantic,
                    query: Some("rust".to_owned()),
                    from: None,
                    to: None,
                    order: RetrievalOrder::Newest,
                    top: 10,
                    count_only: false,
                },
            )
            .unwrap();
        assert_eq!(salient.items[0].text, "Rust language");
        assert_eq!(CognitiveStore::DataLake.default_top(), 2);
        assert_eq!(CognitiveStore::KnowledgeBase.default_top(), 12);
        assert_eq!(CognitiveStore::Diary.default_top(), 12);
        assert_eq!(CognitiveStore::Connections.default_top(), 3);
        assert_eq!(DEFAULT_WEB_TOP, 3);
        assert_eq!(DEFAULT_WISDOM_TOP, 77);
        assert_eq!(MAX_WISDOM_TOP, 7_777);
    }
}

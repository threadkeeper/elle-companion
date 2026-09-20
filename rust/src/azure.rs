//! Explicitly configured Azure adapters; construction performs no network access.
//!
//! These clients reuse the source project's managed-identity, Azure OpenAI REST,
//! and Cosmos AAD/partition patterns, without its orchestration or fallbacks.
//! Owners must be supplied by a trusted service, never by model-generated input.
//! Cosmos requires an existing container partitioned by `/owner_id` and data RBAC.

use std::collections::HashSet;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};
use zeroize::Zeroizing;

use crate::cognitive::{CognitiveRecord, CognitiveRepository, CognitiveStore};
use crate::embeddings::Embedder;
use crate::error::{Error, Result};
use crate::repository::{MemoryRepository, StoredRecord};
use crate::telemetry::{TelemetryEvent, TelemetryRepository, TELEMETRY_CONTAINER, TELEMETRY_SCOPE};

const MODEL_RESOURCE: &str = "https://cognitiveservices.azure.com/";
const MAX_BODY: usize = 16 * 1024 * 1024;
const MAX_DOCUMENT: usize = 2 * 1024 * 1024;
const MAX_RECORDS: usize = 1000;
const MAX_PROMPT: usize = 24_000;
const MAX_TOKEN: usize = 32 * 1024;

/// Supplies a token for an explicitly authorized Azure resource.
pub trait TokenProvider: Send + Sync {
    /// Obtain a token without logging credentials or returning provider bodies.
    fn token(&self, resource: &str) -> Result<String>;
}

/// Container Apps managed identity with an explicit resource allowlist.
///
/// Only loopback HTTP identity endpoints are accepted; `localhost` is pinned to
/// `127.0.0.1` before requesting a token. No CLI, static
/// access-token environment variable, IMDS probing, or credential fallback runs.
pub struct ManagedIdentityCredential {
    endpoint: String,
    header: Zeroizing<String>,
    resources: HashSet<String>,
    agent: ureq::Agent,
}

impl ManagedIdentityCredential {
    /// Configure the platform-provided identity endpoint/header and allowed audiences.
    ///
    /// Audiences may be Microsoft Foundry or a validated account-specific Cosmos resource.
    pub fn new(endpoint: &str, header: &str, allowed_resources: &[&str]) -> Result<Self> {
        let endpoint = validate_identity_endpoint(endpoint)?;
        if !safe_header(header, MAX_TOKEN) {
            return Err(Error::Configuration("Invalid managed identity header"));
        }
        if allowed_resources.is_empty() || allowed_resources.len() > 16 {
            return Err(Error::Configuration(
                "Configure 1 to 16 Azure token resources",
            ));
        }
        let mut resources = HashSet::new();
        for resource in allowed_resources {
            if *resource != MODEL_RESOURCE
                && azure_origin(resource, ".documents.azure.com").is_err()
            {
                return Err(Error::Configuration(
                    "Unsupported managed identity resource",
                ));
            }
            resources.insert((*resource).to_owned());
        }
        Ok(Self {
            endpoint,
            header: Zeroizing::new(header.to_owned()),
            resources,
            agent: http_agent(10),
        })
    }

    /// Read only `IDENTITY_ENDPOINT` and `IDENTITY_HEADER` when explicitly invoked.
    pub fn from_env(allowed_resources: &[&str]) -> Result<Self> {
        let endpoint = std::env::var("IDENTITY_ENDPOINT")
            .map_err(|_| Error::Configuration("IDENTITY_ENDPOINT is required"))?;
        let header = Zeroizing::new(
            std::env::var("IDENTITY_HEADER")
                .map_err(|_| Error::Configuration("IDENTITY_HEADER is required"))?,
        );
        Self::new(&endpoint, &header, allowed_resources)
    }
}

impl TokenProvider for ManagedIdentityCredential {
    fn token(&self, resource: &str) -> Result<String> {
        if !self.resources.contains(resource) {
            return Err(Error::Configuration("Azure token resource is not allowed"));
        }
        let url = format!(
            "{}?api-version=2019-08-01&resource={}",
            self.endpoint,
            percent_encode(resource)
        );
        let response = checked_response(
            self.agent
                .get(&url)
                .set("X-IDENTITY-HEADER", &self.header)
                .call(),
            &[200],
        )?;
        let body = Zeroizing::new(read_body(response, 64 * 1024)?);
        #[derive(Deserialize)]
        struct IdentityResponse {
            access_token: String,
        }
        let parsed: IdentityResponse = serde_json::from_slice(&body)
            .map_err(|_| Error::Transport("Invalid managed identity response"))?;
        let token = Zeroizing::new(parsed.access_token);
        validate_token(&token)?;
        Ok(token.to_string())
    }
}

/// Azure OpenAI-compatible Foundry chat and embedding deployments using Entra tokens.
///
/// Public-cloud `*.openai.azure.com` and `*.services.ai.azure.com` origins are supported.
pub struct FoundryClient {
    chat_endpoint: String,
    embedding_endpoint: String,
    chat_deployment: String,
    embedding_deployment: String,
    dimensions: usize,
    credential: Arc<dyn TokenProvider>,
    agent: ureq::Agent,
}

impl FoundryClient {
    /// Validate configuration without contacting Azure or acquiring a token.
    pub fn new(
        endpoint: &str,
        chat_deployment: &str,
        embedding_deployment: &str,
        dimensions: usize,
        credential: Arc<dyn TokenProvider>,
    ) -> Result<Self> {
        Self::with_endpoints(
            endpoint,
            chat_deployment,
            endpoint,
            embedding_deployment,
            dimensions,
            credential,
        )
    }

    /// Configure chat and embedding deployments on distinct Foundry accounts.
    pub fn with_endpoints(
        chat_endpoint: &str,
        chat_deployment: &str,
        embedding_endpoint: &str,
        embedding_deployment: &str,
        dimensions: usize,
        credential: Arc<dyn TokenProvider>,
    ) -> Result<Self> {
        let chat_endpoint = foundry_origin(chat_endpoint)?;
        let embedding_endpoint = foundry_origin(embedding_endpoint)?;
        if !safe_segment(chat_deployment) || !safe_segment(embedding_deployment) {
            return Err(Error::Configuration(
                "Invalid Foundry deployment identifier",
            ));
        }
        if !(1..=3072).contains(&dimensions) {
            return Err(Error::Configuration(
                "Embedding dimensions must be 1 to 3072",
            ));
        }
        Ok(Self {
            chat_endpoint,
            embedding_endpoint,
            chat_deployment: chat_deployment.to_owned(),
            embedding_deployment: embedding_deployment.to_owned(),
            dimensions,
            credential,
            agent: http_agent(60),
        })
    }

    /// Perform one bounded chat completion; no tools, automatic actions, or retries.
    pub fn complete(&self, system: &str, message: &str) -> Result<String> {
        if message.trim().is_empty() || system.len().saturating_add(message.len()) > MAX_PROMPT {
            return Err(Error::InvalidInput(
                "Chat prompt is empty or exceeds 24000 bytes",
            ));
        }
        let payload = json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ],
            "max_tokens": 1024,
            "n": 1,
            "stream": false
        });
        let body = self.post(
            &self.chat_endpoint,
            &self.chat_deployment,
            "chat/completions",
            payload,
        )?;
        parse_completion(&body)
    }

    fn post(
        &self,
        endpoint: &str,
        deployment: &str,
        operation: &str,
        payload: Value,
    ) -> Result<Vec<u8>> {
        let token = Zeroizing::new(self.credential.token(MODEL_RESOURCE)?);
        validate_token(&token)?;
        let authorization = Zeroizing::new(format!("Bearer {}", token.as_str()));
        let url = format!(
            "{}/openai/deployments/{deployment}/{operation}?api-version=2024-10-21",
            endpoint
        );
        let response = checked_response(
            self.agent
                .post(&url)
                .set("Authorization", &authorization)
                .set("Content-Type", "application/json")
                .send_bytes(payload.to_string().as_bytes()),
            &[200],
        )?;
        read_body(response, 1024 * 1024)
    }
}

impl Embedder for FoundryClient {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() || text.len() > MAX_PROMPT {
            return Err(Error::InvalidInput(
                "Embedding input is empty or exceeds 24000 bytes",
            ));
        }
        let body = self.post(
            &self.embedding_endpoint,
            &self.embedding_deployment,
            "embeddings",
            json!({"input": text, "dimensions": self.dimensions, "encoding_format": "float"}),
        )?;
        parse_embedding(&body, self.dimensions)
    }
}

/// Cosmos SQL REST repository with owner partitioning and ETag-backed CAS.
///
/// Constructors do not provision resources. Operations are synchronous and fail
/// explicitly; failed writes are not automatically retried. Listings have a
/// 1000-record/16-MiB aggregate bound and never silently return partial results.
pub struct CosmosRepository {
    endpoint: String,
    token_resource: String,
    documents_path: String,
    credential: Arc<dyn TokenProvider>,
    agent: ureq::Agent,
}

impl CosmosRepository {
    /// Configure an existing public-cloud Cosmos account, database, and container.
    ///
    /// The token audience is the normalized account origin, without a trailing slash.
    pub fn new(
        endpoint: &str,
        database: &str,
        container: &str,
        credential: Arc<dyn TokenProvider>,
    ) -> Result<Self> {
        let endpoint = azure_origin(endpoint, ".documents.azure.com")?;
        if !safe_segment(database) || !safe_segment(container) {
            return Err(Error::Configuration(
                "Invalid Cosmos database or container identifier",
            ));
        }
        Ok(Self {
            token_resource: format!("{endpoint}/"),
            endpoint,
            documents_path: format!("/dbs/{database}/colls/{container}/docs"),
            credential,
            agent: http_agent(30),
        })
    }

    fn request(&self, method: &str, owner: &str, id: Option<&str>) -> Result<ureq::Request> {
        let partition = partition_header(owner)?;
        let suffix = match id {
            Some(id) => {
                validate_id(id)?;
                format!("/{id}")
            }
            None => String::new(),
        };
        let token = Zeroizing::new(self.credential.token(&self.token_resource)?);
        validate_token(&token)?;
        let authorization = Zeroizing::new(aad_auth_header(&token));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Configuration("System clock must be after Unix epoch"))?;
        let date = rfc1123(now.as_secs())?;
        Ok(self
            .agent
            .request(
                method,
                &format!("{}{}{suffix}", self.endpoint, self.documents_path),
            )
            .set("Authorization", &authorization)
            .set("x-ms-version", "2018-12-31")
            .set("x-ms-date", &date)
            .set("x-ms-documentdb-partitionkey", &partition)
            .set("Content-Type", "application/json"))
    }

    fn read_record(&self, owner: &str, id: &str) -> Result<Option<(StoredRecord, String)>> {
        let response = match self.request("GET", owner, Some(id))?.call() {
            Err(ureq::Error::Status(404, _)) => return Ok(None),
            result => checked_response(result, &[200])?,
        };
        let etag = response
            .header("etag")
            .filter(|value| valid_etag(value))
            .ok_or(Error::Storage("Cosmos response is missing a valid ETag"))?
            .to_owned();
        let body = read_body(response, MAX_DOCUMENT)?;
        let record = parse_record(
            serde_json::from_slice(&body)
                .map_err(|_| Error::Storage("Invalid Cosmos document JSON"))?,
            owner,
        )?;
        if record.id != id {
            return Err(Error::Storage(
                "Cosmos returned a different document identifier",
            ));
        }
        Ok(Some((record, etag)))
    }
}

impl MemoryRepository for CosmosRepository {
    fn list(&self, owner_id: &str) -> Result<Vec<StoredRecord>> {
        partition_header(owner_id)?;
        let payload = json!({
            "query": "SELECT * FROM c WHERE c.owner_id = @owner",
            "parameters": [{"name": "@owner", "value": owner_id}]
        })
        .to_string();
        let mut records = Vec::new();
        let mut ids = HashSet::new();
        let mut seen_tokens = HashSet::new();
        let mut continuation: Option<String> = None;
        let mut remaining_bytes = MAX_BODY;
        for _ in 0..MAX_RECORDS {
            let mut request = self
                .request("POST", owner_id, None)?
                .set("Content-Type", "application/query+json")
                .set("x-ms-documentdb-isquery", "true")
                .set("x-ms-max-item-count", "100");
            if let Some(token) = &continuation {
                request = request.set("x-ms-continuation", token);
            }
            let response = checked_response(request.send_bytes(payload.as_bytes()), &[200])?;
            continuation = response
                .header("x-ms-continuation")
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            if let Some(token) = &continuation {
                if !safe_header(token, MAX_TOKEN) || !seen_tokens.insert(token.clone()) {
                    return Err(Error::Storage("Invalid or repeated Cosmos continuation"));
                }
            }
            let body = read_body(response, remaining_bytes)?;
            remaining_bytes -= body.len();
            let page = parse_documents(&body, owner_id)?;
            if records.len().saturating_add(page.len()) > MAX_RECORDS {
                return Err(Error::Storage("Cosmos listing exceeds 1000 records"));
            }
            for record in page {
                if !ids.insert(record.id.clone()) {
                    return Err(Error::Storage("Cosmos listing returned a duplicate record"));
                }
                records.push(record);
            }
            if continuation.is_none() {
                return Ok(records);
            }
        }
        Err(Error::Storage("Cosmos listing exceeds pagination limit"))
    }

    fn get(&self, owner_id: &str, id: &str) -> Result<Option<StoredRecord>> {
        Ok(self.read_record(owner_id, id)?.map(|(record, _)| record))
    }

    fn create(&mut self, record: &StoredRecord) -> Result<bool> {
        if record.version == 0 {
            return Err(Error::InvalidInput("Records must have a positive version"));
        }
        let body = document_body(record)?;
        let result = self
            .request("POST", &record.owner_id, None)?
            .set("If-None-Match", "*")
            .send_bytes(&body);
        match result {
            Err(ureq::Error::Status(409 | 412, _)) => Ok(false),
            result => {
                checked_response(result, &[201])?;
                Ok(true)
            }
        }
    }

    fn replace(&mut self, record: &StoredRecord, expected_version: u64) -> Result<()> {
        if expected_version == 0 || expected_version.checked_add(1) != Some(record.version) {
            return Err(Error::InvalidInput(
                "Replacement must increment the expected version",
            ));
        }
        let body = document_body(record)?;
        let (current, etag) = self
            .read_record(&record.owner_id, &record.id)?
            .ok_or(Error::NotFound)?;
        if current.version != expected_version {
            return Err(Error::Conflict);
        }
        let request = with_if_match(
            self.request("PUT", &record.owner_id, Some(&record.id))?,
            &etag,
        )?;
        checked_response(request.send_bytes(&body), &[200])?;
        Ok(())
    }

    fn delete(&mut self, owner_id: &str, id: &str, expected_version: u64) -> Result<()> {
        if expected_version == 0 {
            return Err(Error::InvalidInput("Expected version must be positive"));
        }
        let (current, etag) = self.read_record(owner_id, id)?.ok_or(Error::NotFound)?;
        if current.version != expected_version {
            return Err(Error::Conflict);
        }
        let request = with_if_match(self.request("DELETE", owner_id, Some(id))?, &etag)?;
        checked_response(request.call(), &[204])?;
        Ok(())
    }
}

/// Cosmos repository for Elle's fixed, encrypted cognitive containers.
pub struct CosmosCognitiveRepository {
    endpoint: String,
    token_resource: String,
    database: String,
    credential: Arc<dyn TokenProvider>,
    agent: ureq::Agent,
}

impl CosmosCognitiveRepository {
    /// Configure the existing Elle database; container names are fixed by [`CognitiveStore`].
    pub fn new(endpoint: &str, database: &str, credential: Arc<dyn TokenProvider>) -> Result<Self> {
        let endpoint = azure_origin(endpoint, ".documents.azure.com")?;
        if !safe_segment(database) {
            return Err(Error::Configuration("Invalid Cosmos database identifier"));
        }
        Ok(Self {
            token_resource: format!("{endpoint}/"),
            endpoint,
            database: database.to_owned(),
            credential,
            agent: http_agent(30),
        })
    }

    fn request(
        &self,
        method: &str,
        store: CognitiveStore,
        owner: &str,
        id: Option<&str>,
    ) -> Result<ureq::Request> {
        let partition = partition_header(owner)?;
        let suffix = match id {
            Some(id) => {
                validate_id(id)?;
                format!("/{id}")
            }
            None => String::new(),
        };
        let token = Zeroizing::new(self.credential.token(&self.token_resource)?);
        validate_token(&token)?;
        let authorization = Zeroizing::new(aad_auth_header(&token));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Configuration("System clock must be after Unix epoch"))?;
        let date = rfc1123(now.as_secs())?;
        let path = format!(
            "/dbs/{}/colls/{}/docs{suffix}",
            self.database,
            store.container()
        );
        Ok(self
            .agent
            .request(method, &format!("{}{}", self.endpoint, path))
            .set("Authorization", &authorization)
            .set("x-ms-version", "2018-12-31")
            .set("x-ms-date", &date)
            .set("x-ms-documentdb-partitionkey", &partition)
            .set("Content-Type", "application/json"))
    }

    fn read_record(
        &self,
        store: CognitiveStore,
        owner: &str,
        id: &str,
    ) -> Result<Option<(CognitiveRecord, String)>> {
        let response = match self.request("GET", store, owner, Some(id))?.call() {
            Err(ureq::Error::Status(404, _)) => return Ok(None),
            result => checked_response(result, &[200])?,
        };
        let etag = response
            .header("etag")
            .filter(|value| valid_etag(value))
            .ok_or(Error::Storage("Cosmos response is missing a valid ETag"))?
            .to_owned();
        let body = read_body(response, MAX_DOCUMENT)?;
        let record = parse_cognitive_record(
            serde_json::from_slice(&body)
                .map_err(|_| Error::Storage("Invalid Cosmos cognitive document JSON"))?,
            store,
            owner,
        )?;
        if record.id != id {
            return Err(Error::Storage(
                "Cosmos returned a different cognitive identifier",
            ));
        }
        Ok(Some((record, etag)))
    }
}

impl CognitiveRepository for CosmosCognitiveRepository {
    fn get(
        &self,
        store: CognitiveStore,
        owner_id: &str,
        id: &str,
    ) -> Result<Option<CognitiveRecord>> {
        Ok(self
            .read_record(store, owner_id, id)?
            .map(|(record, _)| record))
    }

    fn list(&self, store: CognitiveStore, owner_id: &str) -> Result<Vec<CognitiveRecord>> {
        let payload = json!({
            "query": "SELECT * FROM c WHERE c.owner_id = @owner",
            "parameters": [{"name": "@owner", "value": owner_id}]
        })
        .to_string();
        let mut records = Vec::new();
        let mut ids = HashSet::new();
        let mut seen_tokens = HashSet::new();
        let mut continuation: Option<String> = None;
        let mut remaining_bytes = MAX_BODY;
        for _ in 0..MAX_RECORDS {
            let mut request = self
                .request("POST", store, owner_id, None)?
                .set("Content-Type", "application/query+json")
                .set("x-ms-documentdb-isquery", "true")
                .set("x-ms-max-item-count", "100");
            if let Some(token) = &continuation {
                request = request.set("x-ms-continuation", token);
            }
            let response = checked_response(request.send_bytes(payload.as_bytes()), &[200])?;
            continuation = response
                .header("x-ms-continuation")
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            if let Some(token) = &continuation {
                if !safe_header(token, MAX_TOKEN) || !seen_tokens.insert(token.clone()) {
                    return Err(Error::Storage("Invalid or repeated Cosmos continuation"));
                }
            }
            let body = read_body(response, remaining_bytes)?;
            remaining_bytes -= body.len();
            let page = parse_cognitive_documents(&body, store, owner_id)?;
            if records.len().saturating_add(page.len()) > MAX_RECORDS {
                return Err(Error::Storage("Cognitive listing exceeds 1000 records"));
            }
            for record in page {
                if !ids.insert(record.id.clone()) {
                    return Err(Error::Storage(
                        "Cognitive listing returned a duplicate record",
                    ));
                }
                records.push(record);
            }
            if continuation.is_none() {
                return Ok(records);
            }
        }
        Err(Error::Storage("Cognitive listing exceeds pagination limit"))
    }

    fn create(&mut self, record: &CognitiveRecord) -> Result<bool> {
        let body = cognitive_document_body(record)?;
        let result = self
            .request("POST", record.store, &record.owner_id, None)?
            .set("If-None-Match", "*")
            .send_bytes(&body);
        match result {
            Err(ureq::Error::Status(409 | 412, _)) => Ok(false),
            result => {
                checked_response(result, &[201])?;
                Ok(true)
            }
        }
    }

    fn replace(&mut self, record: &CognitiveRecord, expected_version: u64) -> Result<()> {
        if expected_version == 0 || expected_version.checked_add(1) != Some(record.version) {
            return Err(Error::InvalidInput(
                "Replacement must increment the expected version",
            ));
        }
        let body = cognitive_document_body(record)?;
        let (current, etag) = self
            .read_record(record.store, &record.owner_id, &record.id)?
            .ok_or(Error::NotFound)?;
        if current.version != expected_version {
            return Err(Error::Conflict);
        }
        let request = with_if_match(
            self.request("PUT", record.store, &record.owner_id, Some(&record.id))?,
            &etag,
        )?;
        checked_response(request.send_bytes(&body), &[200])?;
        Ok(())
    }
}

/// Cosmos repository for deidentified app telemetry in its fixed `/scope` partition.
pub struct CosmosTelemetryRepository {
    endpoint: String,
    token_resource: String,
    database: String,
    credential: Arc<dyn TokenProvider>,
    agent: ureq::Agent,
}

impl CosmosTelemetryRepository {
    /// Configure the existing Elle database and fixed telemetry container.
    pub fn new(endpoint: &str, database: &str, credential: Arc<dyn TokenProvider>) -> Result<Self> {
        let endpoint = azure_origin(endpoint, ".documents.azure.com")?;
        if !safe_segment(database) {
            return Err(Error::Configuration("Invalid Cosmos database identifier"));
        }
        Ok(Self {
            token_resource: format!("{endpoint}/"),
            endpoint,
            database: database.to_owned(),
            credential,
            agent: http_agent(30),
        })
    }

    fn request(&self) -> Result<ureq::Request> {
        let token = Zeroizing::new(self.credential.token(&self.token_resource)?);
        validate_token(&token)?;
        let authorization = Zeroizing::new(aad_auth_header(&token));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Configuration("System clock must be after Unix epoch"))?;
        let date = rfc1123(now.as_secs())?;
        let path = format!("/dbs/{}/colls/{TELEMETRY_CONTAINER}/docs", self.database);
        Ok(self
            .agent
            .post(&format!("{}{}", self.endpoint, path))
            .set("Authorization", &authorization)
            .set("x-ms-version", "2018-12-31")
            .set("x-ms-date", &date)
            .set(
                "x-ms-documentdb-partitionkey",
                &partition_header(TELEMETRY_SCOPE)?,
            )
            .set("Content-Type", "application/json"))
    }
}

impl TelemetryRepository for CosmosTelemetryRepository {
    fn create(&mut self, event: &TelemetryEvent) -> Result<bool> {
        if event.scope != TELEMETRY_SCOPE || !safe_segment(&event.id) {
            return Err(Error::InvalidInput("Invalid telemetry document"));
        }
        let body = serde_json::to_vec(event)
            .map_err(|_| Error::InvalidInput("Could not encode telemetry document"))?;
        if body.len() > MAX_DOCUMENT {
            return Err(Error::InvalidInput(
                "Telemetry document exceeds Cosmos size limit",
            ));
        }
        let result = self.request()?.set("If-None-Match", "*").send_bytes(&body);
        match result {
            Err(ureq::Error::Status(409 | 412, _)) => Ok(false),
            result => {
                checked_response(result, &[201])?;
                Ok(true)
            }
        }
    }
}

fn http_agent(seconds: u64) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(seconds))
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(seconds))
        .timeout_write(Duration::from_secs(seconds))
        .redirects(0)
        .build()
}

fn foundry_origin(endpoint: &str) -> Result<String> {
    azure_origin(endpoint, ".openai.azure.com")
        .or_else(|_| azure_origin(endpoint, ".services.ai.azure.com"))
}

fn checked_response(
    result: std::result::Result<ureq::Response, ureq::Error>,
    statuses: &[u16],
) -> Result<ureq::Response> {
    match result {
        Ok(response) if statuses.contains(&response.status()) => Ok(response),
        Err(ureq::Error::Status(409 | 412, _)) => Err(Error::Conflict),
        Err(ureq::Error::Status(401 | 403, _)) => Err(Error::Unauthorized),
        Err(ureq::Error::Status(404, _)) => Err(Error::NotFound),
        Err(ureq::Error::Transport(_)) => {
            Err(Error::Transport("Azure connection failed or timed out"))
        }
        _ => Err(Error::Transport("Azure returned an unexpected HTTP status")),
    }
}

fn read_body(response: ureq::Response, limit: usize) -> Result<Vec<u8>> {
    read_bounded(response.into_reader(), limit)
}

fn read_bounded(reader: impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    reader
        .take((limit.min(MAX_BODY) + 1) as u64)
        .read_to_end(&mut body)
        .map_err(|_| Error::Transport("Could not read Azure response"))?;
    if body.len() > limit.min(MAX_BODY) {
        return Err(Error::Transport("Azure response exceeds the size limit"));
    }
    Ok(body)
}

// Restrict the accepted grammar rather than normalizing ambiguous URL input.
fn azure_origin(endpoint: &str, suffix: &str) -> Result<String> {
    let invalid = Error::Configuration("Expected an approved Azure HTTPS origin without a path");
    let rest = endpoint.strip_prefix("https://").ok_or(invalid.clone())?;
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    let host = authority.strip_suffix(":443").unwrap_or(authority);
    let account = host.strip_suffix(suffix).ok_or(invalid.clone())?;
    if account.is_empty()
        || account.len() > 63
        || !account
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || account.starts_with('-')
        || account.ends_with('-')
    {
        return Err(invalid);
    }
    Ok(format!("https://{host}"))
}

fn validate_identity_endpoint(endpoint: &str) -> Result<String> {
    let invalid =
        Error::Configuration("Identity endpoint must be literal loopback HTTP with a safe path");
    let rest = endpoint.strip_prefix("http://").ok_or(invalid.clone())?;
    let (authority, path) = rest.split_once('/').ok_or(invalid.clone())?;
    let port = authority
        .strip_prefix("127.0.0.1:")
        .or_else(|| authority.strip_prefix("[::1]:"))
        .or_else(|| authority.strip_prefix("localhost:"))
        .ok_or(invalid.clone())?;
    if port.is_empty()
        || !port.bytes().all(|byte| byte.is_ascii_digit())
        || port.parse::<u16>().ok().filter(|port| *port > 0).is_none()
        || path.is_empty()
        || !path.split('/').all(safe_segment)
    {
        return Err(invalid);
    }
    if authority.starts_with("localhost:") {
        Ok(format!("http://127.0.0.1:{port}/{path}"))
    } else {
        Ok(endpoint.to_owned())
    }
}

fn safe_segment(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn validate_id(id: &str) -> Result<()> {
    if !safe_segment(id) {
        return Err(Error::InvalidInput(
            "Record identifier must use ASCII letters, digits, hyphen or underscore",
        ));
    }
    Ok(())
}

fn safe_header(value: &str, max: usize) -> bool {
    !value.is_empty() && value.len() <= max && value.bytes().all(|byte| (32..=126).contains(&byte))
}

fn validate_token(token: &str) -> Result<()> {
    if token.is_empty()
        || token.len() > MAX_TOKEN
        || !token.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'+' | b'/' | b'=')
        })
    {
        return Err(Error::Transport(
            "Azure credential returned an invalid access token",
        ));
    }
    Ok(())
}

fn percent_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut result = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            result.push(char::from(byte));
        } else {
            result.push('%');
            result.push(char::from(HEX[(byte >> 4) as usize]));
            result.push(char::from(HEX[(byte & 15) as usize]));
        }
    }
    result
}

fn aad_auth_header(token: &str) -> String {
    let envelope = Zeroizing::new(format!("type=aad&ver=1.0&sig={token}"));
    percent_encode(&envelope)
}

fn partition_header(owner: &str) -> Result<String> {
    if owner.is_empty() || owner.len() > 512 || owner.chars().any(char::is_control) {
        return Err(Error::InvalidInput(
            "Owner partition must be nonempty and at most 512 bytes",
        ));
    }
    // Encode non-ASCII code points as JSON escapes for an ASCII-safe HTTP header.
    let json = json!([owner]).to_string();
    let mut header = String::new();
    for character in json.chars() {
        if character.is_ascii() {
            header.push(character);
        } else {
            for unit in character.encode_utf16(&mut [0; 2]) {
                use std::fmt::Write;
                write!(header, "\\u{unit:04x}").expect("writing to String cannot fail");
            }
        }
    }
    Ok(header)
}

fn valid_etag(etag: &str) -> bool {
    safe_header(etag, 256)
        && etag.starts_with('"')
        && etag.ends_with('"')
        && etag.len() > 2
        && !etag[1..etag.len() - 1].contains('"')
}

fn with_if_match(request: ureq::Request, etag: &str) -> Result<ureq::Request> {
    if !valid_etag(etag) {
        return Err(Error::Storage("Invalid Cosmos ETag for conditional write"));
    }
    Ok(request.set("If-Match", etag))
}

fn document_body(record: &StoredRecord) -> Result<Vec<u8>> {
    validate_id(&record.id)?;
    partition_header(&record.owner_id)?;
    if record.ciphertext.is_empty() || record.version == 0 {
        return Err(Error::InvalidInput(
            "Encrypted document and positive version are required",
        ));
    }
    let body = serde_json::to_vec(record)
        .map_err(|_| Error::InvalidInput("Could not encode encrypted document"))?;
    if body.len() > MAX_DOCUMENT {
        return Err(Error::InvalidInput(
            "Encrypted document exceeds Cosmos size limit",
        ));
    }
    Ok(body)
}

fn cognitive_document_body(record: &CognitiveRecord) -> Result<Vec<u8>> {
    validate_id(&record.id)?;
    partition_header(&record.owner_id)?;
    if record.ciphertext.is_empty() || record.version == 0 {
        return Err(Error::InvalidInput(
            "Encrypted cognitive document and positive version are required",
        ));
    }
    let body = serde_json::to_vec(record)
        .map_err(|_| Error::InvalidInput("Could not encode encrypted cognitive document"))?;
    if body.len() > MAX_DOCUMENT {
        return Err(Error::InvalidInput(
            "Encrypted cognitive document exceeds Cosmos size limit",
        ));
    }
    Ok(body)
}

fn parse_cognitive_record(
    mut value: Value,
    store: CognitiveStore,
    owner: &str,
) -> Result<CognitiveRecord> {
    let object = value
        .as_object_mut()
        .ok_or(Error::Storage("Invalid Cosmos cognitive document"))?;
    for name in ["_rid", "_self", "_etag", "_attachments", "_ts"] {
        object.remove(name);
    }
    let record: CognitiveRecord = serde_json::from_value(value)
        .map_err(|_| Error::Storage("Invalid Cosmos cognitive document fields"))?;
    if record.owner_id != owner || record.store != store {
        return Err(Error::Unauthorized);
    }
    if !safe_segment(&record.id) || record.version == 0 || record.ciphertext.is_empty() {
        return Err(Error::Storage(
            "Invalid persisted encrypted cognitive document",
        ));
    }
    Ok(record)
}

fn parse_cognitive_documents(
    body: &[u8],
    store: CognitiveStore,
    owner: &str,
) -> Result<Vec<CognitiveRecord>> {
    #[derive(Deserialize)]
    struct Page {
        #[serde(rename = "Documents")]
        documents: Vec<Value>,
    }
    let page: Page = serde_json::from_slice(body)
        .map_err(|_| Error::Storage("Invalid Cosmos cognitive query response"))?;
    if page.documents.len() > MAX_RECORDS {
        return Err(Error::Storage("Cognitive listing exceeds 1000 records"));
    }
    page.documents
        .into_iter()
        .map(|value| parse_cognitive_record(value, store, owner))
        .collect()
}

fn parse_record(mut value: Value, owner: &str) -> Result<StoredRecord> {
    let object = value
        .as_object_mut()
        .ok_or(Error::Storage("Invalid Cosmos document"))?;
    // Cosmos adds system properties; application fields remain strict.
    for name in ["_rid", "_self", "_etag", "_attachments", "_ts"] {
        object.remove(name);
    }
    let record: StoredRecord = serde_json::from_value(value)
        .map_err(|_| Error::Storage("Invalid Cosmos document fields"))?;
    if record.owner_id != owner {
        return Err(Error::Unauthorized);
    }
    if !safe_segment(&record.id) || record.version == 0 || record.ciphertext.is_empty() {
        return Err(Error::Storage("Invalid persisted encrypted document"));
    }
    Ok(record)
}

fn parse_documents(body: &[u8], owner: &str) -> Result<Vec<StoredRecord>> {
    #[derive(Deserialize)]
    struct Page {
        #[serde(rename = "Documents")]
        documents: Vec<Value>,
    }
    let page: Page = serde_json::from_slice(body)
        .map_err(|_| Error::Storage("Invalid Cosmos query response"))?;
    if page.documents.len() > MAX_RECORDS {
        return Err(Error::Storage("Cosmos listing exceeds 1000 records"));
    }
    page.documents
        .into_iter()
        .map(|value| parse_record(value, owner))
        .collect()
}

fn parse_embedding(body: &[u8], dimensions: usize) -> Result<Vec<f32>> {
    #[derive(Deserialize)]
    struct Item {
        index: usize,
        embedding: Vec<f32>,
    }
    #[derive(Deserialize)]
    struct Response {
        data: Vec<Item>,
    }
    let response: Response = serde_json::from_slice(body)
        .map_err(|_| Error::Transport("Invalid Foundry embedding response"))?;
    if response.data.len() != 1 {
        return Err(Error::Transport(
            "Foundry must return exactly one embedding",
        ));
    }
    let item = response
        .data
        .into_iter()
        .next()
        .expect("cardinality checked");
    let norm = item
        .embedding
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>();
    if item.index != 0
        || item.embedding.len() != dimensions
        || !item.embedding.iter().all(|value| value.is_finite())
        || !norm.is_finite()
        || norm <= 0.0
    {
        return Err(Error::Transport(
            "Foundry returned invalid embedding dimensions or values",
        ));
    }
    Ok(item.embedding)
}

fn parse_completion(body: &[u8]) -> Result<String> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|_| Error::Transport("Invalid Foundry completion response"))?;
    let choices = value["choices"]
        .as_array()
        .filter(|choices| choices.len() == 1)
        .ok_or(Error::Transport(
            "Foundry must return exactly one completion",
        ))?;
    let choice = &choices[0];
    if choice["finish_reason"] != "stop"
        || !choice["message"]["tool_calls"].is_null()
        || !choice["message"]["function_call"].is_null()
    {
        return Err(Error::Transport(
            "Foundry completion was incomplete or requested tools",
        ));
    }
    let content = choice["message"]["content"]
        .as_str()
        .filter(|content| !content.trim().is_empty() && content.len() <= 16 * 1024)
        .ok_or(Error::Transport(
            "Foundry completion content is empty or too large",
        ))?;
    Ok(content.to_owned())
}

fn rfc1123(seconds: u64) -> Result<String> {
    if seconds > 253_402_300_799 {
        return Err(Error::Configuration(
            "System clock exceeds supported calendar range",
        ));
    }
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = (seconds / 86_400) as i64;
    // Civil date conversion from the source's Gregorian 400-year-cycle helper.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    Ok(format!(
        "{}, {day:02} {} {year:04} {:02}:{:02}:{:02} GMT",
        DAYS[((days + 4) % 7) as usize],
        MONTHS[(month - 1) as usize],
        seconds % 86_400 / 3600,
        seconds % 3600 / 60,
        seconds % 60,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OfflineToken;
    impl TokenProvider for OfflineToken {
        fn token(&self, _: &str) -> Result<String> {
            Ok("offline.test.token".to_owned())
        }
    }

    fn record() -> StoredRecord {
        serde_json::from_value(json!({
            "id": "record-1", "owner_id": "owner", "kind": "memory",
            "ciphertext": "encrypted-test-envelope", "version": 1,
            "created_at": 0, "updated_at": 0, "expires_at": null, "embedding": null
        }))
        .unwrap()
    }

    #[test]
    fn endpoints_reject_ambiguous_or_unapproved_urls() {
        for endpoint in [
            "http://a.openai.azure.com",
            "https://a.openai.azure.com.evil.test",
            "https://user@a.openai.azure.com",
            "https://a.openai.azure.com?x=1",
            "https://a.openai.azure.com/#frag",
            "https://a.openai.azure.com/../",
            "https://a.openai.azure.com:8443",
            "https://a%2e.openai.azure.com",
            "https://a.openai.azure.com\\@evil.test",
            "https://openai.azure.com",
        ] {
            assert!(
                azure_origin(endpoint, ".openai.azure.com").is_err(),
                "{endpoint}"
            );
        }
        assert_eq!(
            azure_origin("https://a.openai.azure.com:443/", ".openai.azure.com").unwrap(),
            "https://a.openai.azure.com"
        );
        assert_eq!(
            foundry_origin("https://foundry.services.ai.azure.com/").unwrap(),
            "https://foundry.services.ai.azure.com"
        );
        for segment in ["", "..", "a/b", "a\\b", "a%2fb", "a?b", "a b", "a#b"] {
            assert!(!safe_segment(segment));
        }
    }

    #[test]
    fn identity_is_loopback_and_resource_allowlisted() {
        for endpoint in [
            "https://127.0.0.1:80/token",
            "http://localhost.evil.test:80/token",
            "http://169.254.169.254:80/token",
            "http://127.0.0.1:80/token?x=y",
            "http://127.0.0.1:0/token",
            "http://127.0.0.1:80/a/../token",
        ] {
            assert!(validate_identity_endpoint(endpoint).is_err());
        }
        let credential = ManagedIdentityCredential::new(
            "http://127.0.0.1:1234/msi/token",
            "test-header",
            &[MODEL_RESOURCE],
        )
        .unwrap();
        assert!(credential.token("https://evil.test/").is_err());
        assert!(ManagedIdentityCredential::new(
            "http://127.0.0.1:1234/msi/token",
            "test",
            &["https://evil.test/"],
        )
        .is_err());
        assert!(validate_identity_endpoint("http://[::1]:1234/msi/token").is_ok());
        assert_eq!(
            validate_identity_endpoint("http://localhost:1234/msi/token").unwrap(),
            "http://127.0.0.1:1234/msi/token"
        );
    }

    #[test]
    fn constructors_validate_without_network() {
        let credential = Arc::new(OfflineToken);
        assert!(FoundryClient::new(
            "https://a.openai.azure.com",
            "chat",
            "embed",
            0,
            credential.clone()
        )
        .is_err());
        assert!(FoundryClient::new(
            "https://a.openai.azure.com",
            "chat",
            "embed",
            3073,
            credential.clone()
        )
        .is_err());
        assert!(
            CosmosRepository::new("https://a.documents.azure.com", "../db", "c", credential)
                .is_err()
        );
    }

    #[test]
    fn embeddings_require_one_finite_nonzero_vector_of_exact_dimensions() {
        assert_eq!(
            parse_embedding(br#"{"data":[{"index":0,"embedding":[1,0]}]}"#, 2).unwrap(),
            vec![1.0, 0.0]
        );
        for body in [
            r#"{"data":[]}"#,
            r#"{"data":[{"index":0,"embedding":[1]}]}"#,
            r#"{"data":[{"index":0,"embedding":[0,0]}]}"#,
            r#"{"data":[{"index":1,"embedding":[1,0]}]}"#,
            r#"{"data":[{"index":0,"embedding":[1e100,0]}]}"#,
            r#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[1,0]}]}"#,
        ] {
            assert!(parse_embedding(body.as_bytes(), 2).is_err());
        }
    }

    #[test]
    fn completions_do_not_accept_partial_or_tool_results() {
        assert_eq!(
            parse_completion(
                br#"{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}"#
            )
            .unwrap(),
            "ok"
        );
        for body in [
            r#"{"choices":[]}"#,
            r#"{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}"#,
            r#"{"choices":[{"finish_reason":"stop","message":{"content":"ok","tool_calls":[]}}]}"#,
        ] {
            assert!(parse_completion(body.as_bytes()).is_err());
        }
    }

    #[test]
    fn cosmos_headers_escape_partition_auth_and_format_utc() {
        let owner = "a\"\\é😀";
        let header = partition_header(owner).unwrap();
        assert!(header.is_ascii());
        assert_eq!(
            serde_json::from_str::<Vec<String>>(&header).unwrap(),
            vec![owner]
        );
        assert_eq!(
            aad_auth_header("a+b/c="),
            "type%3Daad%26ver%3D1.0%26sig%3Da%2Bb%2Fc%3D"
        );
        assert_eq!(rfc1123(0).unwrap(), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(
            rfc1123(951_782_400).unwrap(),
            "Tue, 29 Feb 2000 00:00:00 GMT"
        );
        assert_eq!(
            rfc1123(1_789_416_000).unwrap(),
            "Mon, 14 Sep 2026 20:00:00 GMT"
        );
    }

    #[test]
    fn cosmos_requests_scope_owner_and_use_exact_etag() {
        let repo = CosmosRepository::new(
            "https://a.documents.azure.com",
            "db",
            "c",
            Arc::new(OfflineToken),
        )
        .unwrap();
        let request = repo.request("PUT", "owner", Some("record-1")).unwrap();
        assert_eq!(
            request.header("x-ms-documentdb-partitionkey"),
            Some("[\"owner\"]")
        );
        assert_eq!(request.header("x-ms-version"), Some("2018-12-31"));
        assert_eq!(
            request.header("Authorization"),
            Some("type%3Daad%26ver%3D1.0%26sig%3Doffline.test.token")
        );
        let request = with_if_match(request, "\"etag-1\"").unwrap();
        assert_eq!(request.header("If-Match"), Some("\"etag-1\""));
        assert_eq!(
            request.url(),
            "https://a.documents.azure.com/dbs/db/colls/c/docs/record-1"
        );
        assert!(!valid_etag("*"));
        assert!(!valid_etag("\"tag\"\r\nInjected: true"));
    }

    #[test]
    fn cosmos_parsing_preserves_strict_application_fields_and_owner() {
        let mut value = serde_json::to_value(record()).unwrap();
        value["_etag"] = json!("etag");
        assert_eq!(parse_record(value.clone(), "owner").unwrap(), record());
        assert_eq!(
            parse_record(value.clone(), "other"),
            Err(Error::Unauthorized)
        );
        value["raw_text"] = json!("must not persist");
        assert!(parse_record(value, "owner").is_err());
        assert!(parse_documents(br#"{}"#, "owner").is_err());
        assert!(parse_documents(
            json!({"Documents": vec![serde_json::to_value(record()).unwrap(); MAX_RECORDS + 1]})
                .to_string()
                .as_bytes(),
            "owner"
        )
        .is_err());
    }

    #[test]
    fn body_limits_and_provider_errors_are_explicit_and_redacted() {
        assert!(read_bounded(&b"12345"[..], 4).is_err());
        assert_eq!(read_bounded(&b"1234"[..], 4).unwrap(), b"1234");
        let response = ureq::Response::new(412, "Conflict", "private upstream body").unwrap();
        assert_eq!(
            checked_response(Err(ureq::Error::Status(412, response)), &[200]).err(),
            Some(Error::Conflict)
        );
        let response = ureq::Response::new(500, "Failed", "private upstream body").unwrap();
        assert!(
            !checked_response(Err(ureq::Error::Status(500, response)), &[200])
                .unwrap_err()
                .to_string()
                .contains("private")
        );
        let response = ureq::Response::new(302, "Redirect", "").unwrap();
        assert!(checked_response(Ok(response), &[200]).is_err());
    }
}

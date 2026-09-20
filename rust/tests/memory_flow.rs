//! Offline end-to-end memory, MCP and backup checks using disposable encrypted stores.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use elle::archive::ArchiveCodec;
use elle::encryption::FieldCipher;
use elle::error::{Error, Result};
use elle::file_repository::FileRepository;
use elle::identity::OwnerId;
use elle::mcp;
use elle::memory::{Category, MemoryPayload, RememberRequest};
use elle::personality::{Detail, Personality, Tone};
use elle::repository::{MemoryRepository, StoredRecord};
use elle::service::MemoryService;
use serde_json::{json, Value};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct CountingRepository {
    inner: FileRepository,
    reads: Arc<AtomicUsize>,
}

impl MemoryRepository for CountingRepository {
    fn list(&self, owner_id: &str) -> Result<Vec<StoredRecord>> {
        self.inner.list(owner_id)
    }

    fn get(&self, owner_id: &str, id: &str) -> Result<Option<StoredRecord>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.get(owner_id, id)
    }

    fn create(&mut self, record: &StoredRecord) -> Result<bool> {
        self.inner.create(record)
    }

    fn replace(&mut self, record: &StoredRecord, expected_version: u64) -> Result<()> {
        self.inner.replace(record, expected_version)
    }

    fn delete(&mut self, owner_id: &str, id: &str, expected_version: u64) -> Result<()> {
        self.inner.delete(owner_id, id, expected_version)
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "elle-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        )))
    }
    fn service(&self) -> MemoryService {
        MemoryService::new(
            Box::new(FileRepository::open(&self.0).unwrap()),
            FieldCipher::new([7; 32]),
            None,
        )
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if self.0.exists() {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
fn owner(user: char) -> OwnerId {
    OwnerId::new(
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        &format!("{user}{user}{user}{user}{user}{user}{user}{user}-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    )
    .unwrap()
}
fn payload(content: &str) -> MemoryPayload {
    MemoryPayload {
        content: content.into(),
        category: Category::Preference,
        source: "Explicit synthetic test instruction".into(),
    }
}
fn request(content: &str, key: &str) -> RememberRequest {
    RememberRequest {
        payload: payload(content),
        idempotency_key: key.into(),
        expires_at: None,
    }
}

#[test]
fn personality_is_cached_until_successful_change() {
    let directory = Directory::new();
    let reads = Arc::new(AtomicUsize::new(0));
    let repository = CountingRepository {
        inner: FileRepository::open(&directory.0).unwrap(),
        reads: Arc::clone(&reads),
    };
    let mut service = MemoryService::new(Box::new(repository), FieldCipher::new([7; 32]), None);
    let user = owner('b');

    assert_eq!(service.personality(&user).unwrap().version, 0);
    assert_eq!(service.personality(&user).unwrap().version, 0);
    assert_eq!(reads.load(Ordering::SeqCst), 1);

    let chosen = Personality {
        tone: Tone::Direct,
        detail: Detail::Concise,
        profile: None,
    };
    let changed = service.set_personality(&user, chosen.clone(), 0).unwrap();
    assert_eq!(changed.version, 1);
    assert_eq!(service.personality(&user).unwrap().settings, chosen);
    assert_eq!(reads.load(Ordering::SeqCst), 2);
}

#[test]
fn memory_is_encrypted_persistent_isolated_correctable_and_forgettable() {
    let directory = Directory::new();
    let user = owner('b');
    let other = owner('c');
    let id;
    {
        let mut service = directory.service();
        let saved = service
            .remember(
                &user,
                request("Use South African rand in examples", "first"),
            )
            .unwrap();
        id = saved.id.clone();
        assert_eq!(
            service
                .remember(
                    &user,
                    request("Use South African rand in examples", "first")
                )
                .unwrap(),
            saved
        );
        assert_eq!(
            service
                .remember(&user, request("Different content", "first"))
                .unwrap_err(),
            Error::Conflict
        );
        assert!(service.list(&other).unwrap().is_empty());
        assert_eq!(service.forget(&other, &id, 1).unwrap_err(), Error::NotFound);
        let corrected = service
            .correct(&user, &id, 1, payload("Prefer concise rand examples"))
            .unwrap();
        assert_eq!(corrected.version, 2);
        assert_eq!(
            service
                .correct(&user, &id, 1, payload("stale"))
                .unwrap_err(),
            Error::Conflict
        );
        assert!(FileRepository::open(&directory.0).is_err());
        let disk = fs::read_to_string(directory.0.join("records.json")).unwrap();
        assert!(!disk.contains("concise"));
        assert!(!disk.contains("Explicit synthetic"));
        assert!(disk.contains("elle-field-v1:"));
    }
    {
        let mut service = directory.service();
        let recalled = service.recall(&user, "rand", 5).unwrap();
        assert_eq!(recalled.mode, "keyword");
        assert_eq!(recalled.memories[0].id, id);
        service.forget(&user, &id, 2).unwrap();
        assert!(service
            .recall(&user, "rand", 5)
            .unwrap()
            .memories
            .is_empty());
    }
}

#[test]
fn backup_restores_corrected_memory_and_personality_without_overwrite() {
    let source = Directory::new();
    let target = Directory::new();
    let user = owner('b');
    let password = "synthetic-strong-test-passphrase";
    let backup;
    let saved;
    let chosen = Personality {
        tone: Tone::Direct,
        detail: Detail::Concise,
        profile: None,
    };
    {
        let mut service = source.service();
        saved = service
            .remember(&user, request("Original context", "one"))
            .unwrap();
        service
            .correct(&user, &saved.id, 1, payload("Corrected context"))
            .unwrap();
        service.set_personality(&user, chosen.clone(), 0).unwrap();
        backup = service.export(&user, password).unwrap();
        assert!(ArchiveCodec::decode(owner('c').as_str(), &backup, password).is_err());
    }
    {
        let mut service = target.service();
        let report = service.restore(&user, &backup, password).unwrap();
        assert!(report.failure.is_none());
        assert_eq!(report.inserted.as_slice(), std::slice::from_ref(&saved.id));
        assert!(report.personality_restored);
        assert_eq!(service.personality(&user).unwrap().settings, chosen);
        assert_eq!(
            service.list(&user).unwrap()[0].payload.content,
            "Corrected context"
        );
        service
            .correct(&user, &saved.id, 2, payload("Newer live context"))
            .unwrap();
        let second = service.restore(&user, &backup, password).unwrap();
        assert!(second.inserted.is_empty());
        assert!(second.skipped_existing.contains(&saved.id));
        assert_eq!(
            service.list(&user).unwrap()[0].payload.content,
            "Newer live context"
        );
    }
}

#[test]
fn malformed_snapshot_has_no_partial_writes() {
    let directory = Directory::new();
    let user = owner('b');
    let password = "synthetic-strong-test-passphrase";
    let mut service = directory.service();
    let snapshot = json!({"schema_version":1,"memories":[{"id":"invalid"}],
        "personality":{"settings":{"tone":"warm","detail":"balanced"},"version":0}});
    let backup = ArchiveCodec::encode(user.as_str(), &snapshot, password).unwrap();
    assert!(service.restore(&user, &backup, password).is_err());
    assert!(service.list(&user).unwrap().is_empty());
}

#[test]
fn mcp_preserves_identity_boundary_and_notifications_cannot_mutate() {
    let directory = Directory::new();
    let user = owner('b');
    let mut service = directory.service();
    let args = serde_json::to_value(request("Prefer concise responses", "first")).unwrap();
    let notification = json!({"jsonrpc":"2.0","method":"tools/call",
        "params":{"name":"elle_remember","arguments":args}});
    assert!(mcp::handle(notification.to_string().as_bytes(), &user, &mut service).is_none());
    assert!(service.list(&user).unwrap().is_empty());
    let mut call = notification;
    call["id"] = json!(1);
    let result = mcp::handle(call.to_string().as_bytes(), &user, &mut service).unwrap();
    assert_eq!(result["result"]["isError"], false);
    call["params"]["arguments"]["owner_id"] = json!(owner('c').as_str());
    let result = mcp::handle(call.to_string().as_bytes(), &user, &mut service).unwrap();
    assert_eq!(result["result"]["isError"], true);
    assert_eq!(service.list(&user).unwrap().len(), 1);
    let tools = mcp::definitions();
    for tool in tools {
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
        assert_eq!(
            tool["annotations"]["readOnlyHint"],
            !matches!(
                tool["name"].as_str(),
                Some("elle_save_cognitive" | "elle_save_connection")
            )
        );
        assert_eq!(tool["annotations"]["destructiveHint"], false);
    }
    let malformed = mcp::handle(b"{", &user, &mut service).unwrap();
    assert_eq!(malformed["error"]["code"], -32700);
    let initialize = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}});
    let malformed = mcp::handle(initialize.to_string().as_bytes(), &user, &mut service).unwrap();
    assert_eq!(malformed["error"]["code"], -32602);

    let dynamic_context = json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
        "name":"elle_context","arguments":{"query":"concise","limit":5,"dynamic_only":true}
    }});
    let result = mcp::handle(dynamic_context.to_string().as_bytes(), &user, &mut service).unwrap();
    let context: Value =
        serde_json::from_str(result["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        context["recall"]["memories"][0]["payload"]["content"],
        "Prefer concise responses"
    );
    assert!(context.get("personality").is_none());
    assert!(context.get("styleGuidance").is_none());
    let initialize = json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{
        "protocolVersion":mcp::PROTOCOL_VERSION,
        "capabilities":{},
        "clientInfo":{"name":"synthetic-test-client","version":"1.0"}
    }});
    let initialized = mcp::handle(initialize.to_string().as_bytes(), &user, &mut service).unwrap();
    assert_eq!(
        initialized["result"]["protocolVersion"],
        mcp::PROTOCOL_VERSION
    );
    let notification = json!({"jsonrpc":"2.0","method":"notifications/initialized","params":{}});
    assert!(mcp::handle(notification.to_string().as_bytes(), &user, &mut service).is_none());
    let list = json!({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{}});
    let listed = mcp::handle(list.to_string().as_bytes(), &user, &mut service).unwrap();
    assert!(!listed["result"]["tools"].as_array().unwrap().is_empty());
}

#[test]
fn shared_wisdom_is_separate_from_private_memory_and_restore() {
    let source = Directory::new();
    let target = Directory::new();
    let user = owner('b');
    let password = "synthetic-strong-test-passphrase";
    let mut service = source.service();
    service
        .remember(&user, request("A private secret project detail", "private"))
        .unwrap();
    let shared = service
        .shared_wisdom("private secret project detail", 10)
        .unwrap();
    assert!(!shared.to_string().contains("A private secret"));
    assert_eq!(shared["privateDerivedPublicationEnabled"], false);
    let archive = service.export(&user, password).unwrap();
    let mut restored = target.service();
    let report = restored.restore(&user, &archive, password).unwrap();
    assert!(report.failure.is_none());
}

#[test]
fn one_user_contributes_wisdom_and_another_user_retrieves_it() {
    let directory = Directory::new();
    let contributor = owner('b');
    let lesson =
        "When a prototype feels stuck, shrink the next step until the result can change your mind.";
    let mut service = directory.service();

    let contribution = service.contribute_wisdom(&contributor, lesson).unwrap();
    assert_eq!(contribution.text, lesson);
    assert_eq!(
        service.contribute_wisdom(&contributor, lesson).unwrap().id,
        contribution.id
    );

    let shared = service
        .shared_wisdom("prototype stuck next step", 5)
        .unwrap();
    assert_eq!(shared["durableHumanContributions"], 1);
    assert!(shared.to_string().contains(lesson));
    assert!(!shared.to_string().contains(contributor.as_str()));

    drop(service);
    let reopened = directory.service();
    let durable = reopened
        .shared_wisdom("prototype stuck next step", 5)
        .unwrap();
    assert!(durable.to_string().contains(lesson));
}

#[test]
fn private_and_shared_servers_expose_disjoint_tools() {
    let private = mcp::definitions_for_role(mcp::ServerRole::Private);
    let wisdom = mcp::definitions_for_role(mcp::ServerRole::SharedWisdom);
    let names = |tools: &[Value]| {
        tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap().to_owned())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let private_names = names(&private);
    let wisdom_names = names(&wisdom);
    assert!(private_names.is_disjoint(&wisdom_names));
    assert!(private_names.contains("elle_cognitive_query"));
    assert!(private_names.contains("elle_save_cognitive"));
    assert!(!private_names.contains("elle_remember"));
    assert!(!private_names.contains("elle_context"));
    assert!(!private_names.contains("elle_contribute_wisdom"));
    assert!(wisdom_names.contains("elle_shared_wisdom"));
    assert!(wisdom_names.contains("elle_contribute_wisdom"));
    assert!(!wisdom_names.contains("elle_get_wisdom_consent"));
    assert!(!wisdom_names.contains("elle_set_wisdom_consent"));
    assert!(!wisdom_names.contains("elle_list_memories"));

    let directory = Directory::new();
    let user = owner('b');
    let mut service = directory.service();
    let call = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
        "name":"elle_remember",
        "arguments":serde_json::to_value(request("Should not cross roles", "blocked")).unwrap()
    }});
    let result = mcp::handle_for_role(
        call.to_string().as_bytes(),
        &user,
        &mut service,
        mcp::ServerRole::SharedWisdom,
    )
    .unwrap();
    assert_eq!(result["result"]["isError"], true);
    assert!(service.list(&user).unwrap().is_empty());

    let call = json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
        "name":"elle_shared_wisdom",
        "arguments":{"query":"next step","limit":5,"dynamic_only":true}
    }});
    let result = mcp::handle_for_role(
        call.to_string().as_bytes(),
        &user,
        &mut service,
        mcp::ServerRole::SharedWisdom,
    )
    .unwrap();
    assert_eq!(result["result"]["isError"], true);
}

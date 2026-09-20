//! MCP-first hosts: local stdio for development and authenticated HTTP for Azure.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use elle::auth::{EntraVerifier, WorkloadEntraVerifier};
use elle::azure::{
    CosmosCognitiveRepository, CosmosRepository, FoundryClient, ManagedIdentityCredential,
};
use elle::cognitive::CognitiveService;
use elle::encryption::FieldCipher;
use elle::error::{Error, Result};
use elle::file_repository::FileRepository;
use elle::identity::OwnerId;
use elle::mcp::{self, MAX_MESSAGE_BYTES};
use elle::memory::{MemoryPayload, RememberRequest};
use elle::service::MemoryService;
use serde::Deserialize;
use zeroize::Zeroizing;

type OptionalEmbedder = Option<Box<dyn elle::embeddings::Embedder>>;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Elle: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args == ["--help"] {
        println!(
            "Elle MCP server\nCommands: serve | stdio | export <file> | restore <file> | seed-demo\n\
            Requires ELLE_MCP_ROLE=private|wisdom. Local commands also require\n\
            ELLE_LOCAL_DEV=1, ELLE_TENANT_ID, ELLE_USER_ID, ELLE_DATA_DIR,\n\
            ELLE_FIELD_ENCRYPTION_KEY (base64 32 bytes). Export/restore also require\n\
            ELLE_ARCHIVE_PASSPHRASE. Never put passwords in chat or source files.\n\
            serve uses Entra bearer authentication and Cosmos managed identity.\n\
            See README for required deployment configuration."
        );
        return Ok(());
    }
    if args == ["serve"] || args == ["seed-demo"] {
        let role = mcp::ServerRole::parse(&required("ELLE_MCP_ROLE")?)?;
        let tenant = required("ELLE_TENANT_ID")?;
        let audience = required("ELLE_API_AUDIENCE")?;
        let origin = required("ELLE_PUBLIC_ORIGIN")?;
        let key = Zeroizing::new(required("ELLE_FIELD_ENCRYPTION_KEY")?);
        let cosmos = required("ELLE_COSMOS_ENDPOINT")?;
        let cosmos = cosmos.trim_end_matches('/').trim_end_matches(":443");
        let cosmos_resource = format!("{cosmos}/");
        let credential = Arc::new(ManagedIdentityCredential::from_env(&[
            &cosmos_resource,
            "https://cognitiveservices.azure.com/",
        ])?);
        let repository = CosmosRepository::new(
            cosmos,
            &required("ELLE_COSMOS_DATABASE")?,
            &required("ELLE_COSMOS_CONTAINER")?,
            credential.clone(),
        )?;
        let cognitive_repository: Option<Box<dyn elle::cognitive::CognitiveRepository>> =
            if role == mcp::ServerRole::Private {
                Some(Box::new(CosmosCognitiveRepository::new(
                    cosmos,
                    &required("ELLE_COSMOS_DATABASE")?,
                    credential.clone(),
                )?))
            } else {
                None
            };
        let (embedder, cognitive_embedder): (OptionalEmbedder, OptionalEmbedder) =
            match (role, env::var("ELLE_FOUNDRY_ENDPOINT")) {
                (mcp::ServerRole::Private, Ok(endpoint)) => {
                    let chat_endpoint =
                        env::var("ELLE_CHAT_ENDPOINT").unwrap_or_else(|_| endpoint.clone());
                    let chat_deployment = required("ELLE_CHAT_DEPLOYMENT")?;
                    let embedding_deployment = required("ELLE_EMBEDDING_DEPLOYMENT")?;
                    let dimensions = required("ELLE_EMBEDDING_DIMENSIONS")?
                        .parse()
                        .map_err(|_| Error::Configuration("Invalid embedding dimensions"))?;
                    let memory = FoundryClient::with_endpoints(
                        &chat_endpoint,
                        &chat_deployment,
                        &endpoint,
                        &embedding_deployment,
                        dimensions,
                        credential.clone(),
                    )?;
                    let cognitive = FoundryClient::with_endpoints(
                        &chat_endpoint,
                        &chat_deployment,
                        &endpoint,
                        &embedding_deployment,
                        dimensions,
                        credential.clone(),
                    )?;
                    (Some(Box::new(memory)), Some(Box::new(cognitive)))
                }
                (mcp::ServerRole::SharedWisdom, _) => (None, None),
                (mcp::ServerRole::Private, Err(env::VarError::NotPresent)) => {
                    eprintln!("Elle: Foundry not configured; using explicit keyword retrieval");
                    (None, None)
                }
                (mcp::ServerRole::Private, Err(_)) => {
                    return Err(Error::Configuration(
                        "Invalid Foundry endpoint configuration",
                    ))
                }
            };
        let cognitive = match cognitive_repository {
            Some(repository) => Some(
                CognitiveService::new(repository, FieldCipher::from_base64(&key)?)
                    .with_embedder(cognitive_embedder),
            ),
            None => None,
        };
        let mut service = MemoryService::new(
            Box::new(repository),
            FieldCipher::from_base64(&key)?,
            embedder,
        );
        if args == ["seed-demo"] {
            if env::var("ELLE_DEMO_SEED").as_deref() != Ok("1") {
                return Err(Error::Configuration("Demo seed requires ELLE_DEMO_SEED=1"));
            }
            seed_demo(
                &mut service,
                role,
                &tenant,
                &required("ELLE_DEMO_USER_IDS")?,
            )?;
            println!("Synthetic demo data seeded.");
            return Ok(());
        }
        let users = if role == mcp::ServerRole::Private {
            Some(allowed_users()?)
        } else {
            None
        };
        let bridge_actor = env::var("ELLE_BRIDGE_ACTOR_ID").ok();
        let verifier = match role {
            mcp::ServerRole::Private => {
                EntraVerifier::for_users(&tenant, &audience, users.as_deref().unwrap_or_default())?
            }
            mcp::ServerRole::SharedWisdom => EntraVerifier::for_tenant_users(&tenant, &audience)?,
        };
        let bridge_client = env::var("ELLE_BRIDGE_CLIENT_ID").ok();
        let bridge_verifier = match (role, bridge_actor.as_deref(), bridge_client.as_deref()) {
            (mcp::ServerRole::Private, Some(actor), None) => {
                Some(elle::server::BridgeVerifier::Delegated(EntraVerifier::new(
                    &tenant, &audience, actor,
                )?))
            }
            (mcp::ServerRole::SharedWisdom, Some(actor), Some(client)) => {
                Some(elle::server::BridgeVerifier::Workload {
                    verifier: WorkloadEntraVerifier::new(&tenant, &audience, actor, client)?,
                    owner: OwnerId::new(&tenant, actor)?,
                })
            }
            (mcp::ServerRole::SharedWisdom, None, None) => None,
            (_, None, None) => None,
            _ => {
                return Err(Error::Configuration(
                    "Bridge actor and client configuration do not match the server role",
                ))
            }
        };
        let bridge_policy = match (role, bridge_actor) {
            (mcp::ServerRole::Private, Some(actor)) => Some(elle::server::BridgePolicy::new(
                &tenant,
                &actor,
                users.as_deref().unwrap_or_default(),
            )?),
            _ => None,
        };
        let continuity = continuity_config(role, &tenant, &audience)?;
        return elle::server::serve(
            "0.0.0.0:8080",
            &origin,
            verifier,
            service,
            role,
            elle::server::RuntimeConfig::new(bridge_verifier, bridge_policy, continuity, cognitive),
        );
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DemoSeed {
        schema_version: u32,
        users: Vec<DemoUser>,
        wisdom: DemoWisdom,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DemoUser {
        key: String,
        memories: Vec<MemoryPayload>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DemoWisdom {
        contributor: String,
        text: String,
    }

    fn allowed_users() -> Result<Vec<String>> {
        let raw = env::var("ELLE_ALLOWED_USER_IDS")
            .or_else(|_| env::var("ELLE_ALLOWED_USER_ID"))
            .map_err(|_| Error::Configuration("ELLE_ALLOWED_USER_IDS"))?;
        let users = raw
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if users.is_empty() {
            return Err(Error::Configuration("ELLE_ALLOWED_USER_IDS"));
        }
        Ok(users)
    }

    fn seed_demo(
        service: &mut MemoryService,
        role: mcp::ServerRole,
        tenant: &str,
        user_ids: &str,
    ) -> Result<()> {
        let seed: DemoSeed =
            serde_json::from_str(include_str!("../../app/demo/synthetic-history.json"))
                .map_err(|_| Error::Integrity("Invalid synthetic demo seed"))?;
        if seed.schema_version != 1 {
            return Err(Error::Integrity("Unsupported synthetic demo seed"));
        }
        let ids = user_ids.split(',').map(str::trim).collect::<Vec<_>>();
        if ids.len() != seed.users.len() {
            return Err(Error::Configuration(
                "ELLE_DEMO_USER_IDS must match the synthetic demo users",
            ));
        }
        let owners = ids
            .iter()
            .map(|id| OwnerId::new(tenant, id))
            .collect::<Result<Vec<_>>>()?;
        match role {
            mcp::ServerRole::Private => {
                for (user, owner) in seed.users.iter().zip(&owners) {
                    for (index, payload) in user.memories.iter().enumerate() {
                        service.remember(
                            owner,
                            RememberRequest {
                                payload: payload.clone(),
                                idempotency_key: format!("synthetic-history-{}-{index}", user.key),
                                expires_at: None,
                            },
                        )?;
                    }
                }
            }
            mcp::ServerRole::SharedWisdom => {
                let index = seed
                    .users
                    .iter()
                    .position(|user| user.key == seed.wisdom.contributor)
                    .ok_or(Error::Integrity("Invalid synthetic Wisdom contributor"))?;
                let owner = &owners[index];
                service.contribute_wisdom(owner, &seed.wisdom.text)?;
            }
        }
        Ok(())
    }
    if env::var("ELLE_LOCAL_DEV").as_deref() != Ok("1") {
        return Err(Error::Configuration(
            "Local host requires explicit ELLE_LOCAL_DEV=1",
        ));
    }
    let owner = OwnerId::new(&required("ELLE_TENANT_ID")?, &required("ELLE_USER_ID")?)?;
    let role = mcp::ServerRole::parse(&required("ELLE_MCP_ROLE")?)?;
    let key = Zeroizing::new(required("ELLE_FIELD_ENCRYPTION_KEY")?);
    let cipher = FieldCipher::from_base64(&key)?;
    let directory = PathBuf::from(required("ELLE_DATA_DIR")?);
    let repository = FileRepository::open(&directory.join(role.name()))?;
    // Azure adapters are callable library components; this host uses local storage
    // and explicit keyword retrieval until the authenticated Azure host is wired.
    let mut service = MemoryService::new(Box::new(repository), cipher, None);
    match args.as_slice() {
        [command] if command == "stdio" => {
            eprintln!("Elle: local development MCP; keyword retrieval; trusted local client only");
            let stdin = io::stdin();
            let mut input = stdin.lock();
            let stdout = io::stdout();
            let mut output = stdout.lock();
            loop {
                let mut message = Vec::new();
                let count = input
                    .by_ref()
                    .take((MAX_MESSAGE_BYTES + 2) as u64)
                    .read_until(b'\n', &mut message)
                    .map_err(|_| Error::Transport("Cannot read MCP input"))?;
                if count == 0 {
                    break;
                }
                if message.last() == Some(&b'\n') {
                    message.pop();
                }
                if message.last() == Some(&b'\r') {
                    message.pop();
                }
                if message.len() > MAX_MESSAGE_BYTES {
                    return Err(Error::InvalidInput("MCP input exceeds message size limit"));
                }
                if let Some(response) = mcp::handle_for_role(&message, &owner, &mut service, role) {
                    writeln!(output, "{response}")
                        .and_then(|()| output.flush())
                        .map_err(|_| Error::Transport("Cannot write MCP response"))?;
                }
            }
            Ok(())
        }
        [command, path] if command == "export" => {
            let passphrase = Zeroizing::new(required("ELLE_ARCHIVE_PASSPHRASE")?);
            let archive = service.export(&owner, &passphrase)?;
            let mut output = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .map_err(|_| Error::Storage("Export target exists or is inaccessible"))?;
            output
                .write_all(&archive)
                .and_then(|()| output.sync_all())
                .map_err(|_| Error::Storage("Archive download failed"))?;
            println!("Encrypted Elle backup exported.");
            Ok(())
        }
        [command, path] if command == "restore" => {
            let passphrase = Zeroizing::new(required("ELLE_ARCHIVE_PASSPHRASE")?);
            let archive = read_archive(Path::new(path))?;
            let report = service.restore(&owner, &archive, &passphrase)?;
            println!(
                "{}",
                serde_json::to_string(&report)
                    .map_err(|_| Error::Integrity("Cannot serialize restore report"))?
            );
            if report.failure.is_some() {
                return Err(Error::Storage(
                    "Restore was partial; review the report and retry",
                ));
            }
            Ok(())
        }
        _ => Err(Error::InvalidInput("Unknown command; use --help")),
    }
}

fn required(name: &'static str) -> Result<String> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or(Error::Configuration(name))
}

fn continuity_config(
    role: mcp::ServerRole,
    tenant_id: &str,
    audience: &str,
) -> Result<Option<elle::server::ContinuityConfig>> {
    if role != mcp::ServerRole::Private {
        return Ok(None);
    }
    let bindings_file = optional("ELLE_CONTINUITY_BINDINGS_FILE")?;
    let actor_id = optional("ELLE_CONTINUITY_ACTOR_ID")?;
    let client_id = optional("ELLE_CONTINUITY_CLIENT_ID")?;
    continuity_config_from_values(
        role,
        tenant_id,
        audience,
        bindings_file,
        actor_id,
        client_id,
    )
}

fn continuity_config_from_values(
    role: mcp::ServerRole,
    tenant_id: &str,
    audience: &str,
    bindings_file: Option<String>,
    actor_id: Option<String>,
    client_id: Option<String>,
) -> Result<Option<elle::server::ContinuityConfig>> {
    if role != mcp::ServerRole::Private {
        return Ok(None);
    }
    match (bindings_file, actor_id, client_id) {
        (None, None, None) => Ok(None),
        (Some(bindings_file), Some(actor_id), Some(client_id)) => {
            let verifier = WorkloadEntraVerifier::new(tenant_id, audience, &actor_id, &client_id)?;
            let bindings =
                elle::server::ContinuityBindings::from_file(Path::new(&bindings_file), tenant_id)?;
            Ok(Some(elle::server::ContinuityConfig::new(
                verifier, bindings,
            )))
        }
        _ => Err(Error::Configuration(
            "Continuity configuration must be complete",
        )),
    }
}

fn optional(name: &'static str) -> Result<Option<String>> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        _ => Err(Error::Configuration(name)),
    }
}

fn read_archive(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path).map_err(|_| Error::Storage("Cannot open backup file"))?;
    let mut bytes = Vec::new();
    file.take(17 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Error::Storage("Cannot read backup file"))?;
    if bytes.len() > 17 * 1024 * 1024 {
        return Err(Error::InvalidInput("Backup exceeds file size limit"));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TENANT: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const APP: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const ACTOR: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";
    const OWNER: &str = "dddddddd-dddd-dddd-dddd-dddddddddddd";
    const HANDLE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn continuity_startup_is_private_and_all_or_nothing() {
        assert!(matches!(
            continuity_config_from_values(mcp::ServerRole::Private, TENANT, APP, None, None, None),
            Ok(None)
        ));
        for values in [
            (Some("bindings".to_owned()), None, None),
            (None, Some(ACTOR.to_owned()), None),
            (None, None, Some(APP.to_owned())),
            (Some("bindings".to_owned()), Some(ACTOR.to_owned()), None),
            (Some("bindings".to_owned()), None, Some(APP.to_owned())),
            (None, Some(ACTOR.to_owned()), Some(APP.to_owned())),
        ] {
            assert!(continuity_config_from_values(
                mcp::ServerRole::Private,
                TENANT,
                APP,
                values.0,
                values.1,
                values.2
            )
            .is_err());
        }
        assert!(matches!(
            continuity_config_from_values(
                mcp::ServerRole::SharedWisdom,
                TENANT,
                APP,
                Some("not-read".to_owned()),
                Some(ACTOR.to_owned()),
                Some(APP.to_owned())
            ),
            Ok(None)
        ));
    }

    #[test]
    fn complete_continuity_startup_loads_the_binding_file() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!("elle-continuity-{suffix}.json"));
        let contents = serde_json::json!({
            "schema_version":1,
            "tenant_id":TENANT,
            "bindings":[{"handle_sha256":HANDLE,"owner_uuid":OWNER}]
        });
        fs::write(&path, serde_json::to_vec(&contents).unwrap()).unwrap();
        let result = continuity_config_from_values(
            mcp::ServerRole::Private,
            TENANT,
            APP,
            Some(path.to_string_lossy().into_owned()),
            Some(ACTOR.to_owned()),
            Some(APP.to_owned()),
        );
        fs::remove_file(path).unwrap();
        assert!(matches!(result, Ok(Some(_))));
    }
}

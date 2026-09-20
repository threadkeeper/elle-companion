//! Stateless, authenticated JSON MCP over HTTP behind a trusted TLS ingress.
//!
//! This listener is not HTTPS. Deploy behind Azure Container Apps HTTPS ingress
//! with request timeouts and rate limits; tiny_http does not expose per-request
//! socket deadlines. No export or restore routes are provided.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use zeroize::Zeroizing;

use crate::auth::{EntraVerifier, WorkloadEntraVerifier};
use crate::cognitive::CognitiveService;
use crate::error::{Error, Result};
use crate::identity::OwnerId;
use crate::mcp;
use crate::service::MemoryService;

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
const MAX_CONTINUITY_BINDINGS_BYTES: u64 = 1024 * 1024;

/// A validated, tenant-pinned map from opaque continuity handles to memory owners.
pub struct ContinuityBindings {
    owners: BTreeMap<String, OwnerId>,
}

/// Complete optional continuity runtime configuration.
pub struct ContinuityConfig {
    verifier: WorkloadEntraVerifier,
    bindings: ContinuityBindings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuityBindingsFile {
    schema_version: u32,
    tenant_id: String,
    bindings: Vec<ContinuityBinding>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuityBinding {
    handle_sha256: String,
    owner_uuid: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuityContextRequest {
    query: String,
    limit: usize,
}

#[derive(Clone, Copy, Debug)]
enum ContinuityContextError {
    Body,
    Operation,
}

impl ContinuityBindings {
    /// Load and validate a bounded strict JSON binding file for one exact tenant.
    pub fn from_file(path: &Path, tenant_id: &str) -> Result<Self> {
        let file = File::open(path)
            .map_err(|_| Error::Configuration("Cannot open continuity bindings file"))?;
        let mut bytes = Vec::new();
        file.take(MAX_CONTINUITY_BINDINGS_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| Error::Configuration("Cannot read continuity bindings file"))?;
        if bytes.len() as u64 > MAX_CONTINUITY_BINDINGS_BYTES {
            return Err(Error::Configuration(
                "Continuity bindings file is too large",
            ));
        }
        Self::from_json(&bytes, tenant_id)
    }

    fn from_json(bytes: &[u8], tenant_id: &str) -> Result<Self> {
        let file: ContinuityBindingsFile = serde_json::from_slice(bytes)
            .map_err(|_| Error::Configuration("Invalid continuity bindings file"))?;
        if file.schema_version != 1 || file.tenant_id != tenant_id || file.bindings.is_empty() {
            return Err(Error::Configuration("Invalid continuity bindings file"));
        }
        let mut owners = BTreeMap::new();
        let mut owner_ids = BTreeSet::new();
        for binding in file.bindings {
            if !valid_continuity_handle(&binding.handle_sha256) {
                return Err(Error::Configuration("Invalid continuity bindings file"));
            }
            let owner = OwnerId::new(tenant_id, &binding.owner_uuid)
                .map_err(|_| Error::Configuration("Invalid continuity bindings file"))?;
            if !owner_ids.insert(owner.as_str().to_owned())
                || owners.insert(binding.handle_sha256, owner).is_some()
            {
                return Err(Error::Configuration("Invalid continuity bindings file"));
            }
        }
        Ok(Self { owners })
    }

    fn owner_for(&self, handle: &str) -> Option<&OwnerId> {
        if !valid_continuity_handle(handle) {
            return None;
        }
        self.owners.get(handle)
    }
}

impl ContinuityConfig {
    /// Couple workload authentication and owner bindings as one runtime capability.
    pub fn new(verifier: WorkloadEntraVerifier, bindings: ContinuityBindings) -> Self {
        Self { verifier, bindings }
    }
}

/// Restricts asserted bridge users to one authenticated transport actor and allow-list.
pub struct BridgePolicy {
    tenant_id: String,
    actor: crate::identity::OwnerId,
    allowed_object_ids: BTreeSet<String>,
}

/// Authentication mode for a direct, role-scoped bridge caller.
pub enum BridgeVerifier {
    /// A delegated actor asserting an allowed private-memory user.
    Delegated(EntraVerifier),
    /// A pinned application workload accessing Shared Wisdom or asserting an allowed user.
    Workload {
        /// Verifies the workload token's actor, client, role and token type.
        verifier: WorkloadEntraVerifier,
        /// Authenticated workload owner; private dispatch requires a separate user assertion.
        owner: OwnerId,
    },
}

impl BridgePolicy {
    /// Validate the trusted actor and permitted private-memory user identifiers.
    pub fn new(tenant_id: &str, actor_id: &str, user_ids: &[String]) -> Result<Self> {
        if user_ids.is_empty() {
            return Err(Error::Configuration("Bridge users must not be empty"));
        }
        let actor = crate::identity::OwnerId::new(tenant_id, actor_id)
            .map_err(|_| Error::Configuration("Invalid bridge actor"))?;
        let mut allowed_object_ids = BTreeSet::new();
        for user_id in user_ids {
            crate::identity::OwnerId::new(tenant_id, user_id)
                .map_err(|_| Error::Configuration("Invalid bridge user"))?;
            allowed_object_ids.insert(user_id.to_ascii_lowercase());
        }
        Ok(Self {
            tenant_id: tenant_id.to_ascii_lowercase(),
            actor,
            allowed_object_ids,
        })
    }

    fn owner_for(
        &self,
        actor: &crate::identity::OwnerId,
        user_id: &str,
    ) -> Result<crate::identity::OwnerId> {
        if actor != &self.actor
            || !self
                .allowed_object_ids
                .contains(&user_id.to_ascii_lowercase())
        {
            return Err(Error::Unauthorized);
        }
        crate::identity::OwnerId::new(&self.tenant_id, user_id).map_err(|_| Error::Unauthorized)
    }
}

struct RequestContext<'a> {
    origin: &'a str,
    metadata: &'a str,
    challenge: &'a Header,
    bridge_policy: Option<&'a BridgePolicy>,
}

struct RuntimeState {
    verifier: EntraVerifier,
    bridge_verifier: Option<BridgeVerifier>,
    service: MemoryService,
    cognitive: Option<CognitiveService>,
    role: mcp::ServerRole,
    continuity: Option<ContinuityConfig>,
}

/// Optional authenticated routes enabled for this server instance.
pub struct RuntimeConfig {
    bridge_verifier: Option<BridgeVerifier>,
    bridge_policy: Option<BridgePolicy>,
    continuity: Option<ContinuityConfig>,
    cognitive: Option<CognitiveService>,
}

impl RuntimeConfig {
    /// Group optional bridge and continuity configuration.
    pub fn new(
        bridge_verifier: Option<BridgeVerifier>,
        bridge_policy: Option<BridgePolicy>,
        continuity: Option<ContinuityConfig>,
        cognitive: Option<CognitiveService>,
    ) -> Self {
        Self {
            bridge_verifier,
            bridge_policy,
            continuity,
            cognitive,
        }
    }
}

/// Serve a single-owner, sequential JSON MCP endpoint until the listener fails.
///
/// `address` is the internal HTTP bind address, such as `0.0.0.0:8080`.
/// `public_origin` must be the external HTTPS origin without a path.
pub fn serve(
    address: &str,
    public_origin: &str,
    verifier: EntraVerifier,
    service: MemoryService,
    role: mcp::ServerRole,
    runtime: RuntimeConfig,
) -> Result<()> {
    let origin = validate_origin(public_origin)?;
    let metadata_url = format!("{origin}/.well-known/oauth-protected-resource");
    let challenge = format!("Bearer resource_metadata=\"{metadata_url}\"");
    let challenge_header = header("WWW-Authenticate", &challenge)?;
    let metadata = json!({
        "resource":format!("{origin}/mcp"),
        "authorization_servers":[verifier.authority()],
        "scopes_supported":[verifier.delegated_scope()],
        "bearer_methods_supported":["header"]
    })
    .to_string();
    let server =
        Server::http(address).map_err(|_| Error::Transport("Cannot bind the HTTP listener"))?;
    let mut state = RuntimeState {
        verifier,
        bridge_verifier: runtime.bridge_verifier,
        service,
        cognitive: runtime.cognitive,
        role,
        continuity: runtime.continuity,
    };
    loop {
        let request = server
            .recv()
            .map_err(|_| Error::Transport("HTTP listener failed"))?;
        // A disconnected caller must not terminate the shared listener.
        if handle_request(
            request,
            &RequestContext {
                origin: &origin,
                metadata: &metadata,
                challenge: &challenge_header,
                bridge_policy: runtime.bridge_policy.as_ref(),
            },
            &mut state,
        )
        .is_err()
        {
            eprintln!("Elle: HTTP request failed");
        }
    }
}

fn handle_request(
    mut request: Request,
    context: &RequestContext<'_>,
    state: &mut RuntimeState,
) -> Result<()> {
    let request_id = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let method = format!("{:?}", request.method());
    let path = request.url().split('?').next().unwrap_or("").to_owned();
    let is_continuity_path = path == "/continuity" || path.starts_with("/continuity/");
    let is_enabled_continuity_route = is_continuity_path
        && request.method() == &Method::Post
        && path == "/continuity/context"
        && state.role == mcp::ServerRole::Private
        && state.continuity.is_some();
    if is_continuity_path && !is_enabled_continuity_route {
        log_continuity(
            request_id,
            continuity_log_path(&path),
            404,
            "route_not_found",
        );
        return reply(request, 404, r#"{"error":"Not found"}"#, None);
    }
    if !is_continuity_path {
        eprintln!(
            "Elle diagnostic: request_id={request_id} role={} method={method} path={path} event=request_started",
            state.role.name()
        );
    }
    match single_header(&request, "Origin") {
        Ok(None) => {}
        Ok(Some(value)) if value == context.origin => {}
        _ => {
            if is_continuity_path {
                log_continuity(
                    request_id,
                    continuity_log_path(&path),
                    403,
                    "forbidden_origin",
                );
            } else {
                eprintln!(
                    "Elle diagnostic: request_id={request_id} role={} status=403 error=forbidden_origin",
                    state.role.name()
                );
            }
            return reply(request, 403, r#"{"error":"Forbidden origin"}"#, None);
        }
    }
    if request.method() == &Method::Post && path.starts_with("/bridge/") {
        return handle_bridge(request, request_id, &path, context, state);
    }
    if is_continuity_path {
        return handle_continuity(request, request_id, state);
    }
    match (request.method(), path.as_str()) {
        (&Method::Get, "/healthz") => {
            return reply(request, 200, r#"{"status":"ok"}"#, None);
        }
        (&Method::Get, "/.well-known/oauth-protected-resource") => {
            return reply(request, 200, context.metadata, None);
        }
        (&Method::Get, "/mcp") => {
            return reply(
                request,
                405,
                r#"{"error":"JSON POST required; SSE is not available"}"#,
                Some(header("Allow", "POST")?),
            );
        }
        (&Method::Post, "/mcp") => {}
        _ => return reply(request, 404, r#"{"error":"Not found"}"#, None),
    }

    let owner = match authenticate(&request, &mut state.verifier) {
        Ok(owner) => owner,
        Err(error) => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} status=401 error={error}",
                state.role.name(),
            );
            return reply(
                request,
                401,
                r#"{"error":"Unauthorized"}"#,
                Some(context.challenge.clone()),
            );
        }
    };
    let body = match read_json_body(&mut request) {
        Ok(body) => body,
        Err((status, message)) => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} status={status} error=request_body_rejected detail={}",
                state.role.name(),
                log_value(message),
            );
            return reply(request, status, message, None);
        }
    };
    let (rpc_method, tool_name) = request_summary(&body);
    match mcp::handle_for_role_with_cognitive(
        &body,
        &owner,
        &mut state.service,
        state.role,
        state.cognitive.as_mut(),
    ) {
        Some(response) => {
            let error = response_error(&response);
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} rpc_method={} tool={} status=200 result={}{}",
                state.role.name(),
                rpc_method,
                tool_name,
                if error.is_some() { "error" } else { "ok" },
                error
                    .map(|detail| format!(" detail={detail}"))
                    .unwrap_or_default()
            );
            reply(request, 200, &response.to_string(), None)
        }
        None => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} rpc_method={} tool={} status=202 result=notification_accepted",
                state.role.name(),
                rpc_method,
                tool_name
            );
            reply(request, 202, "", None)
        }
    }
}

fn handle_continuity(request: Request, request_id: u64, state: &mut RuntimeState) -> Result<()> {
    let continuity = state.continuity.as_mut().ok_or(Error::Configuration(
        "Continuity configuration is unavailable",
    ))?;
    if let Err(rejection) = authenticate_workload(&request, &mut continuity.verifier) {
        log_continuity(request_id, "/continuity/context", 401, rejection);
        return reply(
            request,
            401,
            r#"{"error":"Unauthorized"}"#,
            Some(header("WWW-Authenticate", "Bearer")?),
        );
    }
    handle_authorized_continuity(
        request,
        request_id,
        &continuity.bindings,
        &mut state.service,
    )
}

fn handle_authorized_continuity(
    mut request: Request,
    request_id: u64,
    bindings: &ContinuityBindings,
    service: &mut MemoryService,
) -> Result<()> {
    let owner = resolve_continuity_owner(
        bindings,
        single_header(&request, "X-Elle-Continuity-Handle-SHA256")
            .ok()
            .flatten(),
    );
    let Some(owner) = owner else {
        log_continuity(request_id, "/continuity/context", 403, "handle_rejected");
        return reply(request, 403, r#"{"error":"Forbidden"}"#, None);
    };
    let body = match read_json_body(&mut request) {
        Ok(body) => body,
        Err((status, message)) => {
            log_continuity(request_id, "/continuity/context", status, "body_rejected");
            return reply(request, status, message, None);
        }
    };
    match invoke_continuity_context(&owner, &body, service) {
        Ok(value) => {
            log_continuity(request_id, "/continuity/context", 200, "none");
            reply(request, 200, &value.to_string(), None)
        }
        Err(ContinuityContextError::Body) => {
            log_continuity(request_id, "/continuity/context", 400, "body_rejected");
            reply(request, 400, r#"{"error":"Invalid request body"}"#, None)
        }
        Err(ContinuityContextError::Operation) => {
            log_continuity(request_id, "/continuity/context", 500, "operation_failed");
            reply(request, 500, r#"{"error":"Context unavailable"}"#, None)
        }
    }
}

fn invoke_continuity_context(
    owner: &OwnerId,
    body: &[u8],
    service: &mut MemoryService,
) -> std::result::Result<Value, ContinuityContextError> {
    let arguments: ContinuityContextRequest =
        serde_json::from_slice(body).map_err(|_| ContinuityContextError::Body)?;
    if arguments.query.trim().is_empty()
        || arguments.query.len() > 512
        || !(1..=20).contains(&arguments.limit)
    {
        return Err(ContinuityContextError::Body);
    }
    service
        .context(owner, &arguments.query, arguments.limit)
        .map_err(|_| ContinuityContextError::Operation)
}

fn resolve_continuity_owner(
    bindings: &ContinuityBindings,
    handle: Option<&str>,
) -> Option<OwnerId> {
    handle
        .and_then(|handle| bindings.owner_for(handle))
        .cloned()
}

fn authenticate_workload(
    request: &Request,
    verifier: &mut WorkloadEntraVerifier,
) -> std::result::Result<(), &'static str> {
    let value = single_header(request, "Authorization")
        .map_err(|_| "duplicate_authorization")?
        .ok_or("missing_authorization")?;
    let token = bearer(value).ok_or("malformed_bearer")?;
    verifier
        .verify_diagnostic(token)
        .map_err(|rejection| rejection.label())
}

fn valid_continuity_handle(handle: &str) -> bool {
    handle.len() == 64
        && handle
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn continuity_log_path(path: &str) -> &'static str {
    if path == "/continuity/context" {
        "/continuity/context"
    } else {
        "/continuity/*"
    }
}

fn log_continuity(request_id: u64, path: &str, status: u16, rejection: &str) {
    eprintln!(
        "Elle continuity: request_id={request_id} path={path} status={status} rejection={rejection}"
    );
}

fn handle_bridge(
    mut request: Request,
    request_id: u64,
    path: &str,
    context: &RequestContext<'_>,
    state: &mut RuntimeState,
) -> Result<()> {
    let body = match read_json_body(&mut request) {
        Ok(body) => body,
        Err((status, message)) => return reply(request, status, message, None),
    };
    let tool_name = path.strip_prefix("/bridge/").unwrap_or("");
    let mut arguments = match serde_json::from_slice(&body) {
        Ok(Value::Object(arguments)) => arguments,
        _ => {
            return reply(
                request,
                400,
                r#"{"error":"Tool arguments must be a JSON object"}"#,
                None,
            )
        }
    };
    let authenticated_owner = match single_header(&request, "X-Elle-Continuity-Handle-SHA256") {
        Ok(Some(handle)) if state.role == mcp::ServerRole::Private => {
            match state.continuity.as_ref() {
                Some(continuity) if valid_continuity_handle(handle) => continuity
                    .bindings
                    .owner_for(handle)
                    .cloned()
                    .ok_or("user_binding_unavailable"),
                None => Err("user_binding_unavailable"),
                _ => Err("user_binding_rejected"),
            }
        }
        Ok(None) => authenticate_bridge(
            &request,
            &mut arguments,
            &mut state.verifier,
            state.bridge_verifier.as_mut(),
            context.bridge_policy,
        ),
        _ => Err("user_binding_rejected"),
    };
    let owner = match authenticated_owner {
        Ok(owner) => owner,
        Err(error) => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} status=401 error={error}",
                state.role.name(),
            );
            return reply(
                request,
                401,
                r#"{"error":"Unauthorized"}"#,
                Some(context.challenge.clone()),
            );
        }
    };
    let arguments = Value::Object(arguments);
    match mcp::invoke_for_role(
        tool_name,
        arguments,
        &owner,
        &mut state.service,
        state.role,
        state.cognitive.as_mut(),
    ) {
        Ok(value) => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} bridge_tool={} status=200 result=ok",
                state.role.name(),
                log_value(tool_name)
            );
            reply(request, 200, &value.to_string(), None)
        }
        Err(error) => {
            eprintln!(
                "Elle diagnostic: request_id={request_id} role={} bridge_tool={} status=400 result=error detail={}",
                state.role.name(),
                log_value(tool_name),
                log_value(&error.to_string())
            );
            reply(
                request,
                400,
                &json!({"error":error.to_string()}).to_string(),
                None,
            )
        }
    }
}

fn authenticate_bridge(
    request: &Request,
    arguments: &mut serde_json::Map<String, Value>,
    verifier: &mut EntraVerifier,
    bridge_verifier: Option<&mut BridgeVerifier>,
    policy: Option<&BridgePolicy>,
) -> std::result::Result<crate::identity::OwnerId, &'static str> {
    let value = single_header(request, "Authorization")
        .map_err(|_| "duplicate_authorization")?
        .ok_or("missing_authorization")?;
    let token = bearer(value).ok_or("malformed_bearer")?;
    let user_id = match arguments.remove("user_object_id") {
        Some(Value::String(user_id)) if !user_id.is_empty() => user_id,
        Some(_) => return Err("malformed_user_assertion"),
        None => {
            if let Some(BridgeVerifier::Workload { verifier, owner }) = bridge_verifier {
                verifier
                    .verify_diagnostic(token)
                    .map_err(|rejection| rejection.label())?;
                if policy.is_some() {
                    return Err("user_assertion_missing");
                }
                return Ok(owner.clone());
            }
            return verifier
                .verify_diagnostic(token)
                .map_err(|rejection| rejection.label());
        }
    };
    let actor = match bridge_verifier {
        Some(BridgeVerifier::Delegated(verifier)) => verifier
            .verify_diagnostic(token)
            .map_err(|rejection| rejection.label())?,
        Some(BridgeVerifier::Workload { verifier, owner }) => {
            verifier
                .verify_diagnostic(token)
                .map_err(|rejection| rejection.label())?;
            owner.clone()
        }
        _ => return Err("bridge_verifier_missing"),
    };
    policy
        .ok_or("bridge_policy_missing")?
        .owner_for(&actor, &user_id)
        .map_err(|_| "user_assertion_rejected")
}

fn authenticate(
    request: &Request,
    verifier: &mut EntraVerifier,
) -> std::result::Result<crate::identity::OwnerId, &'static str> {
    let value = single_header(request, "Authorization")
        .map_err(|_| "duplicate_authorization")?
        .ok_or("missing_authorization")?;
    let token = bearer(value).ok_or("malformed_bearer")?;
    verifier
        .verify_diagnostic(token)
        .map_err(|rejection| rejection.label())
}

fn read_json_body(
    request: &mut Request,
) -> std::result::Result<Zeroizing<Vec<u8>>, (u16, &'static str)> {
    if !matches!(
        single_header(request, "Content-Type"),
        Ok(Some(value)) if is_json(value)
    ) {
        return Err((415, r#"{"error":"JSON content type required"}"#));
    }
    if !matches!(single_header(request, "Accept"), Ok(None) | Ok(Some("")))
        && !matches!(
            single_header(request, "Accept"),
            Ok(Some(value)) if accepts_json(value)
        )
    {
        return Err((406, r#"{"error":"JSON response required"}"#));
    }
    let length = match (
        single_header(request, "Content-Length"),
        single_header(request, "Transfer-Encoding"),
    ) {
        (Ok(Some(_)), Ok(Some(_))) | (Err(_), _) | (_, Err(_)) => {
            return Err((400, r#"{"error":"Unsupported request framing"}"#));
        }
        (Ok(None), Ok(Some(value))) if value.trim().eq_ignore_ascii_case("chunked") => None,
        (Ok(None), Ok(Some(_))) => {
            return Err((400, r#"{"error":"Unsupported request framing"}"#));
        }
        (Ok(Some(value)), Ok(None))
            if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            Some(
                value
                    .parse::<usize>()
                    .map_err(|_| (413, r#"{"error":"Request too large"}"#))?,
            )
        }
        (Ok(Some(_)), Ok(None)) => {
            return Err((411, r#"{"error":"Content-Length required"}"#));
        }
        (Ok(None), Ok(None)) => {
            return Err((411, r#"{"error":"Content-Length required"}"#));
        }
    };
    if length.is_some_and(|length| length > mcp::MAX_MESSAGE_BYTES) {
        return Err((413, r#"{"error":"Request too large"}"#));
    }
    let mut body = Zeroizing::new(Vec::new());
    if request
        .as_reader()
        .take(mcp::MAX_MESSAGE_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .is_err()
        || length.is_some_and(|length| body.len() != length)
    {
        return Err((400, r#"{"error":"Invalid request body"}"#));
    }
    if body.len() > mcp::MAX_MESSAGE_BYTES {
        return Err((413, r#"{"error":"Request too large"}"#));
    }
    Ok(body)
}

fn request_summary(body: &[u8]) -> (String, String) {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return ("invalid_json".to_owned(), "none".to_owned());
    };
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .map(log_value)
        .unwrap_or_else(|| "missing".to_owned());
    let tool = value
        .pointer("/params/name")
        .and_then(Value::as_str)
        .map(log_value)
        .unwrap_or_else(|| "none".to_owned());
    (method, tool)
}

fn response_error(response: &Value) -> Option<String> {
    if let Some(error) = response.pointer("/error/message").and_then(Value::as_str) {
        return Some(log_value(error));
    }
    if response.pointer("/result/isError").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let text = response
        .pointer("/result/content/0/text")
        .and_then(Value::as_str)?;
    let detail = serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|value| value.get("error").and_then(Value::as_str).map(log_value));
    Some(detail.unwrap_or_else(|| "tool_error".to_owned()))
}

fn log_value(value: &str) -> String {
    let sanitized = value
        .chars()
        .take(160)
        .map(|character| {
            if character.is_ascii_alphanumeric()
                || matches!(character, '_' | '-' | '/' | '.' | ':' | ' ')
            {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "unspecified".to_owned()
    } else {
        sanitized
    }
}

fn header(name: &str, value: &str) -> Result<Header> {
    Header::from_bytes(name.as_bytes(), value.as_bytes())
        .map_err(|_| Error::Configuration("Invalid HTTP response header"))
}

fn reply(request: Request, status: u16, body: &str, extra: Option<Header>) -> Result<()> {
    let mut response = Response::from_string(body)
        .with_status_code(StatusCode(status))
        .with_header(header("Content-Type", "application/json")?)
        .with_header(header("Cache-Control", "no-store")?)
        .with_header(header("X-Content-Type-Options", "nosniff")?);
    if let Some(extra) = extra {
        response.add_header(extra);
    }
    request
        .respond(response)
        .map_err(|_| Error::Transport("Cannot send the HTTP response"))
}

fn single_header<'a>(request: &'a Request, name: &str) -> Result<Option<&'a str>> {
    let mut matches = request
        .headers()
        .iter()
        .filter(|header| header.field.as_str().as_str().eq_ignore_ascii_case(name));
    let value = matches.next().map(|header| header.value.as_str());
    if matches.next().is_some() {
        return Err(Error::InvalidInput("Duplicate request header"));
    }
    Ok(value)
}

fn bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("Bearer")
        || token.is_empty()
        || token.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return None;
    }
    Some(token)
}

fn is_json(value: &str) -> bool {
    value
        .split(';')
        .next()
        .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
}

fn accepts_json(value: &str) -> bool {
    value.split(',').any(|range| {
        let mut parts = range.trim().split(';');
        let media = parts.next().unwrap_or("").trim();
        let mut quality = 1.0_f32;
        for parameter in parts {
            if let Some((name, value)) = parameter.trim().split_once('=') {
                if name.trim().eq_ignore_ascii_case("q") {
                    quality = value.trim().parse::<f32>().unwrap_or(0.0);
                }
            }
        }
        (media.eq_ignore_ascii_case("application/json")
            || media.eq_ignore_ascii_case("application/*")
            || media == "*/*")
            && quality > 0.0
            && quality <= 1.0
    })
}

fn validate_origin(value: &str) -> Result<String> {
    let origin = value.strip_suffix('/').unwrap_or(value);
    let host = origin
        .strip_prefix("https://")
        .ok_or(Error::Configuration("Public origin must use HTTPS"))?;
    if origin.len() > 2048
        || host.is_empty()
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b':'))
    {
        return Err(Error::Configuration(
            "Public origin must contain only an HTTPS hostname and optional port",
        ));
    }
    let mut pieces = host.split(':');
    let hostname = pieces.next().unwrap_or("");
    if hostname.is_empty()
        || hostname.split('.').any(|label| {
            label.is_empty() || label.len() > 63 || label.starts_with('-') || label.ends_with('-')
        })
        || pieces
            .next()
            .is_some_and(|port| port.parse::<u16>().map_or(true, |port| port == 0))
        || pieces.next().is_some()
    {
        return Err(Error::Configuration(
            "Public origin hostname or port is invalid",
        ));
    }
    Ok(origin.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::{Shutdown, TcpStream};
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::encryption::FieldCipher;
    use crate::repository::{MemoryRepository, StoredRecord};
    use base64::{engine::general_purpose, Engine};
    use ring::rand::SystemRandom;
    use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};

    struct EmptyRepository;

    const TENANT: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const APP: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const OWNER: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";
    const OTHER_OWNER: &str = "dddddddd-dddd-dddd-dddd-dddddddddddd";
    const WORKLOAD_ACTOR_OID: &str = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee";
    const WORKLOAD_CLIENT_ID: &str = "ffffffff-ffff-ffff-ffff-ffffffffffff";
    const BOUND_MEMORY_OWNER: &str = "11111111-1111-1111-1111-111111111111";
    const HANDLE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TEST_RSA_KID: &str = "continuity-test";
    const TEST_RSA_MODULUS: &str = "45DjvGchDqXT403IGfksvcSfRwrOVMKlzedbZFwtwVEaHjvI-xIpOzXf7V1W1CbgKj7ouvltT8wofxmfIdicGvsd1YkcDeygp1KeGLy-ReKTtCbEaLZjSvaB-2IRPG9AQuJnrNoQZl3DjQsbPn0FR4Nz5EM_YZoP5bwqg8x4_Oc3-AFlUgC-EvO8TwNJXp7M3iAgoBumL0pTbA58u2QnaVD97kqYF8wWnKiFIPMwIWiajWCBxUhVbgdoqypw5dA20ePNR0y1TM6Rv2HxfjJbuDg_2CvS4hrCYtyY2QUJEIz_qAf4Tm9uLclJclvq1p_cDN5fA6AO638vF-dK-JFruw";
    const TEST_RSA_PKCS8: &str = "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDjkOO8ZyEOpdPjTcgZ+Sy9xJ9HCs5UwqXN51tkXC3BURoeO8j7Eik7Nd/tXVbUJuAqPui6+W1PzCh/GZ8h2Jwa+x3ViRwN7KCnUp4YvL5F4pO0JsRotmNK9oH7YhE8b0BC4mes2hBmXcONCxs+fQVHg3PkQz9hmg/lvCqDzHj85zf4AWVSAL4S87xPA0lenszeICCgG6YvSlNsDny7ZCdpUP3uSpgXzBacqIUg8zAhaJqNYIHFSFVuB2irKnDl0DbR481HTLVMzpG/YfF+Mlu4OD/YK9LiGsJi3JjZBQkQjP+oB/hOb24tyUlyW+rWn9wM3l8DoA7rfy8X50r4kWu7AgMBAAECggEASdHpBmdf8lv5yb0kIcTSbjbXwlhviVBhL9OSspIyZ4kTE2aqckO4a1Q1MU87iPOZeSrSHUEnZCDirCRYGkclkJ0QVwI0vxGZJd4nmfe0M4BmEKUYxq0PtbQUg0MTO0sNigTew9QzSLm240yMiG9O5J1wXUYxS8yJxqkNE5cjUkpklq94TCF8fLKIFkxrDdu5d1PIPVHlv+zoQiXp1ldXhF7KIWnQ2NyS9/JzHVAxPvRiEfn924izOXc0pI5PEkqz4ICKC7Qcl7YANQoUYw12XSVPja1LMbZqoSn+rHFBFMKamSd041y9/xJ2ES7kGerlHnD8ewfPoTuPiHAbq1qxtQKBgQD4Flu/pT6159bWELWh4BpIBUkTZgdA7y1gBSmhBZgDLnjJLOy6SVNzUC7npg+GpBsuuY/HjJ9jj9kQRN2LrXOF7aMK1uD+EQBvOv1TdrEIyF5Sfboq72qKbd37WebEXT9xusvIaFsB5ICrH3X17aCt242Q4UxRNSo4nZjb6WSJ9QKBgQDq0vk8Evt5rVNIPZY4HITmfgr2aBt5pSiu5Wt46UFf/Xn63loEF0Un3w2nnzo1/W4phxKwDURenstkO8H5nhfGQdFa+uwWp2uos/2F/IyPUxpJms7+HLTZutylybq1Hw4/OzLCCeiAr+W2ZLVaYbJS8iAtBCpMNoBNucnUYN4g7wKBgAQ0pNOH4ptE1eCFIf8fhHKKHGYGycKxC0zgaYdASAZtyEBo0Y6K5a5DwrfMmeDHcWqGXMieOql+a8iZ0kOm6hlwIN5zLBdChIZeMqMylOe4NdkiJoDJ1D2KhUPYj0/u4L910jSQiFJs5D2CaAaGQ74OxcSZ/Sg3RYL2MPwxZbHtAoGAQcVdsYnPjcESNoWpcYXrY3OiNmnqaCPuRS5U78TFXtFsPOvSYprx77z14iEi+MRG+rKudUkCAU6QwT5LklLJbeo5bTYisiWqbdIcDE80P2CTWFJX76yyqtk/u9/Iv7o3D1bRXK/Rw1mBCZkjgnEitUDD6lfkUPxi62JCOY34KVkCgYAa3Bjd1TIXGijxo25DhSfaBhnPAxfJL4f0bScmUy4x3kZiv5U/AhluYtyur6xSxkS/Q12JOXuu1BOeUJi0KZ6wP/zrfBazhQ1b1qowGWi05WqNN6R42cQfQro8hPJOFRmf7NsW1KA3mOa4C6qJiVcat9EPf8z7tmES935dgGGxcw==";

    impl MemoryRepository for EmptyRepository {
        fn list(&self, _owner_id: &str) -> Result<Vec<StoredRecord>> {
            Ok(Vec::new())
        }

        fn get(&self, _owner_id: &str, _id: &str) -> Result<Option<StoredRecord>> {
            Ok(None)
        }

        fn create(&mut self, _record: &StoredRecord) -> Result<bool> {
            Ok(true)
        }

        fn replace(&mut self, _record: &StoredRecord, _expected_version: u64) -> Result<()> {
            Ok(())
        }

        fn delete(&mut self, _owner_id: &str, _id: &str, _expected_version: u64) -> Result<()> {
            Ok(())
        }
    }

    struct RecordingRepository {
        owners: Arc<Mutex<Vec<String>>>,
    }

    impl MemoryRepository for RecordingRepository {
        fn list(&self, owner_id: &str) -> Result<Vec<StoredRecord>> {
            self.owners.lock().unwrap().push(owner_id.to_owned());
            Ok(Vec::new())
        }

        fn get(&self, owner_id: &str, _id: &str) -> Result<Option<StoredRecord>> {
            self.owners.lock().unwrap().push(owner_id.to_owned());
            Ok(None)
        }

        fn create(&mut self, _record: &StoredRecord) -> Result<bool> {
            Ok(true)
        }

        fn replace(&mut self, _record: &StoredRecord, _expected_version: u64) -> Result<()> {
            Ok(())
        }

        fn delete(&mut self, _owner_id: &str, _id: &str, _expected_version: u64) -> Result<()> {
            Ok(())
        }
    }

    fn binding_json(bindings: Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "tenant_id": TENANT,
            "bindings": bindings
        }))
        .unwrap()
    }

    fn test_bindings() -> ContinuityBindings {
        ContinuityBindings::from_json(
            &binding_json(json!([{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER}])),
            TENANT,
        )
        .unwrap()
    }

    fn test_continuity_config() -> ContinuityConfig {
        ContinuityConfig::new(
            WorkloadEntraVerifier::new(TENANT, APP, WORKLOAD_ACTOR_OID, WORKLOAD_CLIENT_ID)
                .unwrap(),
            test_bindings(),
        )
    }

    fn signed_continuity_config() -> ContinuityConfig {
        let jwks = serde_json::to_vec(&json!({"keys":[{
            "kid":TEST_RSA_KID,"kty":"RSA","alg":"RS256","use":"sig",
            "n":TEST_RSA_MODULUS,"e":"AQAB"
        }]}))
        .unwrap();
        let verifier =
            WorkloadEntraVerifier::new(TENANT, APP, WORKLOAD_ACTOR_OID, WORKLOAD_CLIENT_ID)
                .unwrap()
                .with_test_jwks(&jwks)
                .unwrap();
        ContinuityConfig::new(verifier, test_bindings())
    }

    fn signed_workload_token() -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let header = general_purpose::URL_SAFE_NO_PAD
            .encode(json!({"alg":"RS256","kid":TEST_RSA_KID}).to_string());
        let claims = general_purpose::URL_SAFE_NO_PAD.encode(
            json!({
                "tid":TENANT,"oid":WORKLOAD_ACTOR_OID,"aud":APP,
                "iss":format!("https://login.microsoftonline.com/{TENANT}/v2.0"),
                "exp":now + 300,"nbf":now.saturating_sub(10),"azp":WORKLOAD_CLIENT_ID,
                "roles":["Continuity.Access"],"ver":"2.0","idtyp":"app","appid":WORKLOAD_CLIENT_ID
            })
            .to_string(),
        );
        let signing_input = format!("{header}.{claims}");
        let private_key = general_purpose::STANDARD.decode(TEST_RSA_PKCS8).unwrap();
        let key_pair = RsaKeyPair::from_pkcs8(&private_key).unwrap();
        let mut signature = vec![0; key_pair.public().modulus_len()];
        key_pair
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                signing_input.as_bytes(),
                &mut signature,
            )
            .unwrap();
        format!(
            "{signing_input}.{}",
            general_purpose::URL_SAFE_NO_PAD.encode(signature)
        )
    }

    fn read_raw_body(request_parts: &[&[u8]]) -> std::result::Result<Vec<u8>, (u16, &'static str)> {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let worker = std::thread::spawn(move || {
            let mut request = server.recv().unwrap();
            read_json_body(&mut request).map(|body| body.to_vec())
        });
        let mut stream = TcpStream::connect(address).unwrap();
        for part in request_parts {
            stream.write_all(part).unwrap();
        }
        stream.shutdown(Shutdown::Write).unwrap();
        worker.join().unwrap()
    }

    fn raw_response(request: &[u8]) -> String {
        raw_response_for(request, mcp::ServerRole::Private, None)
    }

    fn raw_response_for(
        request: &[u8],
        role: mcp::ServerRole,
        continuity: Option<ContinuityConfig>,
    ) -> String {
        raw_response_for_repository(request, role, continuity, EmptyRepository)
    }

    fn raw_response_for_repository<R: MemoryRepository + Send + 'static>(
        request: &[u8],
        role: mcp::ServerRole,
        continuity: Option<ContinuityConfig>,
        repository: R,
    ) -> String {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let worker = std::thread::spawn(move || {
            let request = server.recv().unwrap();
            let verifier = EntraVerifier::new(TENANT, APP, OWNER).unwrap();
            let mut state = RuntimeState {
                verifier,
                bridge_verifier: None,
                service: MemoryService::new(Box::new(repository), FieldCipher::new([7; 32]), None),
                cognitive: None,
                role,
                continuity,
            };
            let challenge = header("WWW-Authenticate", "Bearer test").unwrap();
            handle_request(
                request,
                &RequestContext {
                    origin: "https://elle.example.com",
                    metadata: "{}",
                    challenge: &challenge,
                    bridge_policy: None,
                },
                &mut state,
            )
            .unwrap();
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        response
    }

    fn raw_authorized_continuity_response(request: &[u8]) -> String {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let worker = std::thread::spawn(move || {
            let request = server.recv().unwrap();
            let mut service =
                MemoryService::new(Box::new(EmptyRepository), FieldCipher::new([7; 32]), None);
            handle_authorized_continuity(request, 1, &test_bindings(), &mut service).unwrap();
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        response
    }

    fn raw_workload_wisdom_response(request: &[u8]) -> String {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let worker = std::thread::spawn(move || {
            let request = server.recv().unwrap();
            let verifier = EntraVerifier::new(TENANT, APP, OWNER).unwrap();
            let mut state = RuntimeState {
                verifier,
                bridge_verifier: Some(BridgeVerifier::Workload {
                    verifier: WorkloadEntraVerifier::new(
                        TENANT,
                        APP,
                        WORKLOAD_ACTOR_OID,
                        WORKLOAD_CLIENT_ID,
                    )
                    .unwrap()
                    .with_test_jwks(
                        &serde_json::to_vec(&json!({"keys":[{
                            "kid":TEST_RSA_KID,"kty":"RSA","alg":"RS256","use":"sig",
                            "n":TEST_RSA_MODULUS,"e":"AQAB"
                        }]}))
                        .unwrap(),
                    )
                    .unwrap(),
                    owner: OwnerId::new(TENANT, WORKLOAD_ACTOR_OID).unwrap(),
                }),
                service: MemoryService::new(
                    Box::new(EmptyRepository),
                    FieldCipher::new([7; 32]),
                    None,
                ),
                cognitive: None,
                role: mcp::ServerRole::SharedWisdom,
                continuity: None,
            };
            let challenge = header("WWW-Authenticate", "Bearer test").unwrap();
            handle_request(
                request,
                &RequestContext {
                    origin: "https://elle.example.com",
                    metadata: "{}",
                    challenge: &challenge,
                    bridge_policy: None,
                },
                &mut state,
            )
            .unwrap();
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        response
    }

    fn raw_workload_private_response(request: &[u8]) -> String {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let worker = std::thread::spawn(move || {
            let request = server.recv().unwrap();
            let verifier = EntraVerifier::new(TENANT, APP, OWNER).unwrap();
            let workload_owner = OwnerId::new(TENANT, WORKLOAD_ACTOR_OID).unwrap();
            let mut state = RuntimeState {
                verifier,
                bridge_verifier: Some(BridgeVerifier::Workload {
                    verifier: WorkloadEntraVerifier::new(
                        TENANT,
                        APP,
                        WORKLOAD_ACTOR_OID,
                        WORKLOAD_CLIENT_ID,
                    )
                    .unwrap()
                    .with_test_jwks(
                        &serde_json::to_vec(&json!({"keys":[{
                            "kid":TEST_RSA_KID,"kty":"RSA","alg":"RS256","use":"sig",
                            "n":TEST_RSA_MODULUS,"e":"AQAB"
                        }]}))
                        .unwrap(),
                    )
                    .unwrap(),
                    owner: workload_owner,
                }),
                service: MemoryService::new(
                    Box::new(EmptyRepository),
                    FieldCipher::new([7; 32]),
                    None,
                ),
                cognitive: None,
                role: mcp::ServerRole::Private,
                continuity: None,
            };
            let challenge = header("WWW-Authenticate", "Bearer test").unwrap();
            let policy =
                BridgePolicy::new(TENANT, WORKLOAD_ACTOR_OID, &[OWNER.to_owned()]).unwrap();
            handle_request(
                request,
                &RequestContext {
                    origin: "https://elle.example.com",
                    metadata: "{}",
                    challenge: &challenge,
                    bridge_policy: Some(&policy),
                },
                &mut state,
            )
            .unwrap();
        });
        let mut stream = TcpStream::connect(address).unwrap();
        stream.write_all(request).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        response
    }

    #[test]
    fn valid_chunked_json_body_is_decoded() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n7\r\n{\"ok\":1\r\n1\r\n}\r\n0\r\n\r\n";
        assert_eq!(read_raw_body(&[request]).unwrap(), br#"{"ok":1}"#);
    }

    #[test]
    fn mixed_case_chunked_with_optional_whitespace_is_decoded() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: \tChUnKeD \t\r\n\r\n7\r\n{\"ok\":1\r\n1\r\n}\r\n0\r\n\r\n";
        assert_eq!(read_raw_body(&[request]).unwrap(), br#"{"ok":1}"#);
    }

    #[test]
    fn fixed_length_and_split_chunked_bodies_are_accepted() {
        let fixed = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 8\r\n\r\n{\"ok\":1}";
        assert_eq!(read_raw_body(&[fixed]).unwrap(), br#"{"ok":1}"#);

        let headers = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert_eq!(
            read_raw_body(&[headers, b"4\r\n{\"ok\r\n", b"4\r\n\":1}\r\n0\r\n\r\n"]).unwrap(),
            br#"{"ok":1}"#
        );
    }

    #[test]
    fn malformed_truncated_and_oversized_chunked_bodies_are_rejected() {
        let headers = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n";
        assert_eq!(
            read_raw_body(&[headers, b"x\r\ninvalid\r\n0\r\n\r\n"])
                .unwrap_err()
                .0,
            400
        );
        assert_eq!(
            read_raw_body(&[headers, b"8\r\n{\"ok\":1\r\n"])
                .unwrap_err()
                .0,
            400
        );

        let body = vec![b'a'; mcp::MAX_MESSAGE_BYTES + 1];
        let chunk = format!("{:x}\r\n", body.len());
        assert_eq!(
            read_raw_body(&[headers, chunk.as_bytes(), &body, b"\r\n0\r\n\r\n"])
                .unwrap_err()
                .0,
            413
        );
    }

    #[test]
    fn ambiguous_and_unsupported_request_framing_is_rejected() {
        let cases: &[&[u8]] = &[
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: gzip, chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
        ];
        for request in cases {
            assert_eq!(read_raw_body(&[request]).unwrap_err().0, 400);
        }

        let missing =
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\r\n";
        assert_eq!(read_raw_body(&[missing]).unwrap_err().0, 411);
    }

    #[test]
    fn origin_authentication_and_get_routing_order_is_unchanged() {
        let forbidden = raw_response(b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nOrigin: https://invalid.example.com\r\nContent-Length: 0\r\n\r\n");
        assert!(forbidden.starts_with("HTTP/1.1 403 "));

        let unauthorized = raw_response(
            b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: unsupported\r\n\r\n",
        );
        assert!(unauthorized.starts_with("HTTP/1.1 401 "));
        assert!(unauthorized.contains("WWW-Authenticate: Bearer test"));
        assert!(unauthorized.ends_with(r#"{"error":"Unauthorized"}"#));
        assert!(!unauthorized.contains("token_format_invalid"));

        let method_not_allowed = raw_response(b"GET /mcp HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(method_not_allowed.starts_with("HTTP/1.1 405 "));
        assert!(method_not_allowed.contains("Allow: POST"));
    }

    #[test]
    fn public_origin_cannot_inject_headers_or_urls() {
        assert_eq!(
            validate_origin("https://elle.example.com/").unwrap(),
            "https://elle.example.com"
        );
        assert!(validate_origin("https://localhost:8443").is_ok());
        for value in [
            "http://elle.example.com",
            "https://elle.example.com/mcp",
            "https://user@elle.example.com",
            "https://elle.example.com?query",
            "https://elle.example.com#fragment",
            "https://elle.example.com\"\r\nInjected: yes",
            "https://",
            "https://elle.example.com:0",
            "https://elle.example.com:65536",
        ] {
            assert!(validate_origin(value).is_err(), "{value}");
        }
    }

    #[test]
    fn request_media_and_bearer_are_checked() {
        assert!(is_json("application/json; charset=utf-8"));
        assert!(!is_json("text/plain"));
        assert!(accepts_json("application/json, text/event-stream"));
        assert!(accepts_json("*/*"));
        assert!(!accepts_json("text/event-stream"));
        assert!(!accepts_json("application/json;q=0"));
        assert!(!accepts_json("application/json;q=NaN"));
        assert_eq!(bearer("Bearer a.b.c"), Some("a.b.c"));
        assert_eq!(bearer("bearer a.b.c"), Some("a.b.c"));
        assert_eq!(bearer("Basic a.b.c"), None);
        assert_eq!(bearer("Bearer a.b.c extra"), None);
        assert_eq!(bearer("Bearer "), None);
    }

    #[test]
    fn bridge_policy_accepts_only_the_trusted_actor_and_allowed_user() {
        let tenant = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
        let actor_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
        let user_id = "cccccccc-cccc-cccc-cccc-cccccccccccc";
        let policy = BridgePolicy::new(tenant, actor_id, &[user_id.to_owned()]).unwrap();
        let actor = crate::identity::OwnerId::new(tenant, actor_id).unwrap();
        let owner = policy.owner_for(&actor, user_id).unwrap();
        assert_eq!(owner.as_str(), format!("{tenant}:{user_id}"));
        let wrong_actor =
            crate::identity::OwnerId::new(tenant, "dddddddd-dddd-dddd-dddd-dddddddddddd").unwrap();
        assert!(policy.owner_for(&wrong_actor, user_id).is_err());
        assert!(policy
            .owner_for(&actor, "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
            .is_err());
    }

    #[test]
    fn bridge_user_assertion_is_removed_before_tool_invocation() {
        let mut arguments = serde_json::Map::from_iter([
            (
                "user_object_id".to_owned(),
                Value::String("cccccccc-cccc-cccc-cccc-cccccccccccc".to_owned()),
            ),
            ("query".to_owned(), Value::String("project".to_owned())),
        ]);
        let assertion = match arguments.remove("user_object_id") {
            Some(Value::String(user_id)) => user_id,
            _ => panic!("missing test assertion"),
        };
        assert_eq!(assertion, "cccccccc-cccc-cccc-cccc-cccccccccccc");
        assert_eq!(
            arguments.get("query"),
            Some(&Value::String("project".to_owned()))
        );
        assert!(!arguments.contains_key("user_object_id"));
    }

    #[test]
    fn continuity_bindings_require_strict_schema_tenant_handles_and_unique_owners() {
        assert!(ContinuityBindings::from_json(
            &binding_json(json!([{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER}])),
            TENANT,
        )
        .is_ok());
        for invalid in [
            json!({"schema_version":2,"tenant_id":TENANT,"bindings":[{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER}]}),
            json!({"schema_version":1,"tenant_id":APP,"bindings":[{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER}]}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[]}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[{"handle_sha256":"abc","owner_uuid":BOUND_MEMORY_OWNER}]}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[{"handle_sha256":HANDLE.to_ascii_uppercase(),"owner_uuid":BOUND_MEMORY_OWNER}]}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[{"handle_sha256":HANDLE,"owner_uuid":"00000000-0000-0000-0000-000000000000"}]}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER}],"extra":true}),
            json!({"schema_version":1,"tenant_id":TENANT,"bindings":[{"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER,"extra":true}]}),
        ] {
            assert!(
                ContinuityBindings::from_json(&serde_json::to_vec(&invalid).unwrap(), TENANT)
                    .is_err()
            );
        }
        assert!(ContinuityBindings::from_json(br#"{"schema_version":1"#, TENANT).is_err());

        let duplicate_handle = binding_json(json!([
            {"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER},
            {"handle_sha256":HANDLE,"owner_uuid":OTHER_OWNER}
        ]));
        let duplicate_owner = binding_json(json!([
            {"handle_sha256":HANDLE,"owner_uuid":BOUND_MEMORY_OWNER},
            {"handle_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","owner_uuid":BOUND_MEMORY_OWNER}
        ]));
        assert!(ContinuityBindings::from_json(&duplicate_handle, TENANT).is_err());
        assert!(ContinuityBindings::from_json(&duplicate_owner, TENANT).is_err());
    }

    #[test]
    fn continuity_missing_malformed_and_unknown_handles_are_indistinguishable() {
        let bindings = test_bindings();
        for handle in [
            None,
            Some("abc"),
            Some("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ] {
            assert!(resolve_continuity_owner(&bindings, handle).is_none());
        }

        let requests: &[&[u8]] = &[
            b"POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
            b"POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: abc\r\nContent-Length: 0\r\n\r\n",
            b"POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\r\nContent-Length: 0\r\n\r\n",
            b"POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\nX-Elle-Continuity-Handle-SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\nContent-Length: 0\r\n\r\n",
        ];
        for request in requests {
            let response = raw_authorized_continuity_response(request);
            assert!(response.starts_with("HTTP/1.1 403 "));
            assert!(response.ends_with(r#"{"error":"Forbidden"}"#));
            assert!(!response.contains(HANDLE));
            assert!(!response.contains(BOUND_MEMORY_OWNER));
        }
    }

    #[test]
    fn continuity_route_is_private_post_only_and_authenticates_before_body() {
        let body = br#"{"query":"project","limit":3}"#;
        let anonymous = format!(
            "POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        let response = raw_response_for(
            anonymous.as_bytes(),
            mcp::ServerRole::Private,
            Some(test_continuity_config()),
        );
        assert!(response.starts_with("HTTP/1.1 401 "));
        assert!(response.contains("WWW-Authenticate: Bearer"));
        assert!(response.ends_with(r#"{"error":"Unauthorized"}"#));

        let wrong_token = raw_response_for(
            b"POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer not-a-token\r\nContent-Length: 0\r\n\r\n",
            mcp::ServerRole::Private,
            Some(test_continuity_config()),
        );
        assert!(wrong_token.starts_with("HTTP/1.1 401 "));
        assert!(!wrong_token.contains("token_format_invalid"));

        for origin_headers in [
            "Origin: https://hostile.example\r\n",
            "Origin: https://elle.example.com\r\nOrigin: https://elle.example.com\r\n",
        ] {
            let request = format!(
                "POST /continuity/context HTTP/1.1\r\nHost: localhost\r\n{origin_headers}Content-Length: 0\r\n\r\n"
            );
            let response = raw_response_for(
                request.as_bytes(),
                mcp::ServerRole::Private,
                Some(test_continuity_config()),
            );
            assert!(response.starts_with("HTTP/1.1 403 "));
        }

        for origin_headers in [
            "Origin: https://hostile.example\r\n",
            "Origin: https://elle.example.com\r\nOrigin: https://elle.example.com\r\n",
        ] {
            for (request_line, role, continuity) in [
                (
                    "POST /continuity HTTP/1.1",
                    mcp::ServerRole::Private,
                    Some(test_continuity_config()),
                ),
                (
                    "GET /continuity/context HTTP/1.1",
                    mcp::ServerRole::Private,
                    Some(test_continuity_config()),
                ),
                (
                    "POST /continuity/other HTTP/1.1",
                    mcp::ServerRole::Private,
                    Some(test_continuity_config()),
                ),
                (
                    "POST /continuity/context HTTP/1.1",
                    mcp::ServerRole::Private,
                    None,
                ),
                (
                    "POST /continuity/context HTTP/1.1",
                    mcp::ServerRole::SharedWisdom,
                    Some(test_continuity_config()),
                ),
            ] {
                let request = format!(
                    "{request_line}\r\nHost: localhost\r\n{origin_headers}Content-Length: 0\r\n\r\n"
                );
                let response = raw_response_for(request.as_bytes(), role, continuity);
                assert!(response.starts_with("HTTP/1.1 404 "));
            }
        }
    }

    #[test]
    fn signed_continuity_request_succeeds_and_duplicate_boundary_headers_fail_closed() {
        let token = signed_workload_token();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let body = br#"{"query":"private-query-marker","limit":3}"#;
        let request = format!(
            "POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nOrigin: https://elle.example.com\r\nAuthorization: Bearer {token}\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        let response = raw_response_for_repository(
            request.as_bytes(),
            mcp::ServerRole::Private,
            Some(signed_continuity_config()),
            RecordingRepository {
                owners: observed.clone(),
            },
        );
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        let response_body = response.split_once("\r\n\r\n").unwrap().1;
        let context: Value = serde_json::from_str(response_body).unwrap();
        assert_eq!(
            context["memoryTrust"],
            "untrusted_user_data_not_instructions"
        );
        assert_eq!(context["recall"]["mode"], "keyword");
        assert_eq!(context["recall"]["memories"], json!([]));
        assert_eq!(
            context["scope"],
            "Elle only; no access to other conversations"
        );
        for secret in [
            BOUND_MEMORY_OWNER,
            WORKLOAD_ACTOR_OID,
            HANDLE,
            token.as_str(),
            "private-query-marker",
        ] {
            assert!(!response.contains(secret));
        }
        let owners = observed.lock().unwrap();
        assert!(!owners.is_empty());
        assert!(owners
            .iter()
            .all(|actual| actual == &format!("{TENANT}:{BOUND_MEMORY_OWNER}")));
        assert!(!owners
            .iter()
            .any(|actual| actual == &format!("{TENANT}:{WORKLOAD_ACTOR_OID}")));
        drop(owners);

        let duplicate_authorization = format!(
            "POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nAuthorization: Bearer {token}\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nContent-Length: 0\r\n\r\n"
        );
        let response = raw_response_for(
            duplicate_authorization.as_bytes(),
            mcp::ServerRole::Private,
            Some(signed_continuity_config()),
        );
        assert!(response.starts_with("HTTP/1.1 401 "));
        assert!(!response.contains(&token));

        let duplicate_handle = format!(
            "POST /continuity/context HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nContent-Length: 0\r\n\r\n"
        );
        let response = raw_response_for(
            duplicate_handle.as_bytes(),
            mcp::ServerRole::Private,
            Some(signed_continuity_config()),
        );
        assert!(response.starts_with("HTTP/1.1 403 "));
        assert!(!response.contains(HANDLE));
        assert!(!response.contains(BOUND_MEMORY_OWNER));
        assert!(!response.contains(&token));
    }

    #[test]
    fn direct_bridge_uses_bound_user() {
        for (tool_name, arguments) in [
            ("elle_context", json!({"query":"green dashboard","limit":3})),
            ("elle_list_memories", json!({})),
            ("elle_personality", json!({})),
        ] {
            let observed = Arc::new(Mutex::new(Vec::new()));
            let body = arguments.to_string();
            let request = format!(
                "POST /bridge/{tool_name} HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let response = raw_response_for_repository(
                request.as_bytes(),
                mcp::ServerRole::Private,
                Some(signed_continuity_config()),
                RecordingRepository {
                    owners: observed.clone(),
                },
            );
            assert!(
                response.starts_with("HTTP/1.1 200 "),
                "{tool_name}: {response}"
            );
            let owners = observed.lock().unwrap();
            assert!(!owners.is_empty());
            assert!(owners
                .iter()
                .all(|owner| owner == &format!("{TENANT}:{BOUND_MEMORY_OWNER}")));
        }
    }

    #[test]
    fn direct_wisdom_bridge_accepts_only_pinned_workload() {
        let token = signed_workload_token();
        let body = r#"{"query":"listen before advising","limit":3}"#;
        let request = format!(
            "POST /bridge/elle_shared_wisdom HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = raw_workload_wisdom_response(request.as_bytes());
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert!(!response.contains(&token));
        assert!(!response.contains(WORKLOAD_ACTOR_OID));

        let unauthorized = format!(
            "POST /bridge/elle_shared_wisdom HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = raw_workload_wisdom_response(unauthorized.as_bytes());
        assert!(response.starts_with("HTTP/1.1 401 "), "{response}");
    }

    #[test]
    fn direct_private_bridge_requires_allowlisted_workload_user_assertion() {
        let token = signed_workload_token();
        let allowed_body = json!({"user_object_id":OWNER}).to_string();
        let allowed = format!(
            "POST /bridge/elle_personality HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{allowed_body}",
            allowed_body.len()
        );
        let response = raw_workload_private_response(allowed.as_bytes());
        assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
        assert!(!response.contains(&token));

        for body in [
            json!({}),
            json!({"user_object_id":"33333333-3333-4333-8333-333333333333"}),
        ] {
            let body = body.to_string();
            let request = format!(
                "POST /bridge/elle_personality HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let response = raw_workload_private_response(request.as_bytes());
            assert!(response.starts_with("HTTP/1.1 401 "), "{response}");
            assert!(!response.contains(&token));
        }
    }

    #[test]
    fn direct_bridge_requires_the_exact_bound_handle() {
        let request = format!("POST /bridge/elle_list_memories HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}");
        let response = raw_response_for(
            request.as_bytes(),
            mcp::ServerRole::Private,
            Some(signed_continuity_config()),
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");

        let unknown = "POST /bridge/elle_list_memories HTTP/1.1\r\nHost: localhost\r\nX-Elle-Continuity-Handle-SHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let response = raw_response_for(
            unknown.as_bytes(),
            mcp::ServerRole::Private,
            Some(signed_continuity_config()),
        );
        assert!(response.starts_with("HTTP/1.1 401 "), "{response}");
    }

    #[test]
    fn direct_bridge_rejects_malformed_or_duplicate_binding() {
        for handle_headers in [
            "X-Elle-Continuity-Handle-SHA256: invalid\r\n".to_owned(),
            format!("X-Elle-Continuity-Handle-SHA256: {HANDLE}\r\nX-Elle-Continuity-Handle-SHA256: {HANDLE}\r\n"),
        ] {
            let request = format!(
                "POST /bridge/elle_list_memories HTTP/1.1\r\nHost: localhost\r\n{handle_headers}Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
            );
            let response = raw_response_for(
                request.as_bytes(),
                mcp::ServerRole::Private,
                Some(signed_continuity_config()),
            );
            assert!(response.starts_with("HTTP/1.1 401 "), "{response}");
        }
    }

    #[test]
    fn continuity_context_uses_only_bound_owner_and_does_not_leak_boundary_values() {
        let bindings = test_bindings();
        let owner = resolve_continuity_owner(&bindings, Some(HANDLE)).unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut service = MemoryService::new(
            Box::new(RecordingRepository {
                owners: observed.clone(),
            }),
            FieldCipher::new([7; 32]),
            None,
        );
        let query = "private-query-marker";
        let body = json!({"query":query,"limit":3}).to_string();
        let response = invoke_continuity_context(&owner, body.as_bytes(), &mut service).unwrap();
        let encoded = response.to_string();
        assert!(!encoded.contains(HANDLE));
        assert!(!encoded.contains(BOUND_MEMORY_OWNER));
        assert!(!encoded.contains(query));
        let owners = observed.lock().unwrap();
        assert!(!owners.is_empty());
        assert!(owners
            .iter()
            .all(|actual| actual == &format!("{TENANT}:{BOUND_MEMORY_OWNER}")));
    }

    #[test]
    fn continuity_context_body_is_strict_and_bounded() {
        let owner = OwnerId::new(TENANT, OWNER).unwrap();
        for body in [
            json!({"query":"","limit":1}),
            json!({"query":"   ","limit":1}),
            json!({"query":"x".repeat(513),"limit":1}),
            json!({"query":"ok","limit":0}),
            json!({"query":"ok","limit":21}),
            json!({"query":"ok","limit":1,"owner_uuid":OWNER}),
            json!({"query":"ok","limit":1,"handle_sha256":HANDLE}),
        ] {
            let mut service =
                MemoryService::new(Box::new(EmptyRepository), FieldCipher::new([7; 32]), None);
            assert!(matches!(
                invoke_continuity_context(
                    &owner,
                    serde_json::to_string(&body).unwrap().as_bytes(),
                    &mut service
                ),
                Err(ContinuityContextError::Body)
            ));
        }
    }
}

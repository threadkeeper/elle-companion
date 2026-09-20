import base64
import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from azure.core.credentials import TokenCredential
from azure.ai.agentserver.core import get_request_context


logger = logging.getLogger(__name__)
_DEFAULT_ENDPOINT = (
    "https://elle-private-vnet.yellowsky-9d92d540.swedencentral."
    "azurecontainerapps.io/bridge"
)
_DEFAULT_SCOPE = "api://0479a728-6b4d-4d96-8693-ef766bc8e1fe/.default"
_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 20
_MAX_MEMORY_CONTENT_BYTES = 16_384


def _token_diagnostic(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError, json.JSONDecodeError):
        return {"claims": "unavailable"}
    return {
        name: claims[name]
        for name in ("oid", "azp", "appid", "idtyp", "roles")
        if name in claims
    }


def _bounded_text(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def build_automatic_turn_arguments(
    *, user_text: str, assistant_text: str, response_id: str | None
) -> dict[str, Any]:
    """Build the deprecated legacy payload used only by historical demo tooling."""
    content = _bounded_text(
        f"User:\n{user_text}\n\nElle:\n{assistant_text}",
        _MAX_MEMORY_CONTENT_BYTES,
    )
    key_material = f"{response_id or ''}\0{user_text}\0{assistant_text}".encode("utf-8")
    return {
        "payload": {
            "content": content,
            "category": "project",
            "source": "automatic-conversation-turn",
        },
        "idempotency_key": f"turn-{hashlib.sha256(key_material).hexdigest()}",
        "expires_at": None,
    }


def build_archive_turn_arguments(
    *, user_text: str, assistant_text: str, timestamp: str | None = None
) -> dict[str, str]:
    """Build the bounded dedicated cognitive archive payload."""
    bounded_user = _bounded_text(user_text, _MAX_MEMORY_CONTENT_BYTES // 2)
    remaining = _MAX_MEMORY_CONTENT_BYTES - len(bounded_user.encode("utf-8"))
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_text": bounded_user,
        "assistant_text": _bounded_text(assistant_text, remaining),
    }


def _request_tool(
    endpoint: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    credential: TokenCredential,
    scope: str,
    user_id: str | None = None,
) -> Any:
    user_id = user_id or get_request_context().user_id
    if not user_id:
        raise RuntimeError("Private tools require a platform caller identity")
    token = credential.get_token(scope)
    arguments = {**arguments, "user_object_id": user_id}
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/{tool_name}",
        data=json.dumps(arguments, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise RuntimeError(f"Private tool returned HTTP {response.status}")
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        detail = error.read(512).decode("utf-8", errors="replace")
        if error.code == 401:
            logger.error(
                "Private tool workload identity rejected: %s",
                _token_diagnostic(token.token),
            )
        raise RuntimeError(f"Private tool returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError("Private tool request failed") from error
    if len(body) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("Private tool response exceeded the size limit")
    return json.loads(body)


def archive_conversation_turn(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    user_text: str,
    assistant_text: str,
    response_id: str | None,
) -> Any:
    """Persist one completed conversation turn outside the model tool loop."""
    del response_id
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_archive_turn",
        build_archive_turn_arguments(
            user_text=user_text,
            assistant_text=assistant_text,
        ),
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def query_cognitive_store(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    store: str,
    mode: str = "auto",
    query: str | None = None,
    order: str = "newest",
    top: int | None = None,
) -> Any:
    """Retrieve bounded new-schema cognitive context for automatic turn handling."""
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_cognitive_query",
        {
            "store": store,
            "mode": mode,
            "query": query,
            "from": None,
            "to": None,
            "order": order,
            "top": top,
            "count_only": False,
        },
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def save_cognitive_knowledge(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    content: str,
    salience: float,
    timestamp: str | None = None,
) -> Any:
    """Persist one automatically attained fact through the new cognitive schema."""
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_save_cognitive",
        {
            "store": "knowledge_base",
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "content": content,
            "salience": salience,
        },
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def save_cognitive_diary(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    content: str,
    timestamp: str | None = None,
) -> Any:
    """Persist one deliberate post-turn reflection through the new cognitive schema."""
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_save_cognitive",
        {
            "store": "diary",
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "content": content,
            "salience": None,
        },
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def save_cognitive_connection(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    event_key: str,
    change_amount: float,
    note: str,
    timestamp: str | None = None,
) -> Any:
    """Append one idempotent relationship-ledger event after a completed turn."""
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_save_connection",
        {
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_key": event_key,
            "change_amount": change_amount,
            "note": note,
        },
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def record_turn_telemetry(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str,
    user_id: str,
    response_id: str,
    timestamp: str,
    input_chars: int,
    reply_chars: int,
    persisted: bool,
    completed: bool = True,
) -> Any:
    """Record one deidentified hosted-turn outcome outside the model tool loop."""
    return _request_tool(
        endpoint or _DEFAULT_ENDPOINT,
        "elle_record_telemetry",
        {
            "response_id": response_id,
            "timestamp": timestamp,
            "model": "foundry-agent",
            "model_requests": 1,
            "tools": [],
            "input_chars": input_chars,
            "reply_chars": reply_chars,
            "persisted": persisted,
            "completed": completed,
        },
        credential=credential,
        scope=scope,
        user_id=user_id,
    )


def make_private_tools(
    *,
    credential: TokenCredential,
    endpoint: str | None = None,
    scope: str | None = None,
) -> list[Callable[..., Any]]:
    endpoint = endpoint or _DEFAULT_ENDPOINT
    scope = scope or _DEFAULT_SCOPE

    def elle_personality() -> Any:
        """Open the user's private Elle personality workshop."""
        return _request_tool(
            endpoint,
            "elle_personality",
            {},
            credential=credential,
            scope=scope,
        )

    def elle_set_personality(expected_version: int, settings: dict[str, Any]) -> Any:
        """Save private Elle personality settings after explicit confirmation."""
        return _request_tool(
            endpoint,
            "elle_set_personality",
            {
                "expected_version": expected_version,
                "settings": settings,
            },
            credential=credential,
            scope=scope,
        )

    def elle_cognitive_query(
        store: str,
        mode: str = "auto",
        query: str | None = None,
        order: str = "newest",
        from_date: str | None = None,
        to_date: str | None = None,
        top: int | None = None,
        count_only: bool = False,
    ) -> Any:
        """Retrieve an owner-scoped cognitive store using bounded structured options."""
        return _request_tool(
            endpoint,
            "elle_cognitive_query",
            {
                "store": store,
                "mode": mode,
                "query": query,
                "from": from_date,
                "to": to_date,
                "order": order,
                "top": top,
                "count_only": count_only,
            },
            credential=credential,
            scope=scope,
        )

    def elle_save_cognitive(
        store: str,
        content: str,
        salience: float | None = None,
    ) -> Any:
        """Deliberately save one durable knowledge-base fact or diary reflection."""
        return _request_tool(
            endpoint,
            "elle_save_cognitive",
            {
                "store": store,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "content": content,
                "salience": salience,
            },
            credential=credential,
            scope=scope,
        )

    return [
        elle_personality,
        elle_set_personality,
        elle_cognitive_query,
        elle_save_cognitive,
    ]

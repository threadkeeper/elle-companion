import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from azure.core.credentials import TokenCredential


_DEFAULT_ENDPOINT = (
    "https://elle-wisdom-vnet.yellowsky-9d92d540.swedencentral."
    "azurecontainerapps.io/bridge"
)
_DEFAULT_SCOPE = "api://0479a728-6b4d-4d96-8693-ef766bc8e1fe/.default"
_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 20


def _request_tool(
    *,
    endpoint: str,
    scope: str,
    credential: TokenCredential,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    token = credential.get_token(scope)
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
                raise RuntimeError(f"Wisdom tool returned HTTP {response.status}")
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        detail = error.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(f"Wisdom tool returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError("Wisdom tool request failed") from error
    if len(body) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("Wisdom tool response exceeded the size limit")
    return json.loads(body)


def query_shared_wisdom(
    *,
    endpoint: str | None,
    credential: TokenCredential,
    scope: str | None,
    query: str,
    limit: int = 5,
) -> Any:
    """Retrieve reviewed shared lessons for automatic pre-turn context."""
    return _request_tool(
        endpoint=endpoint or _DEFAULT_ENDPOINT,
        scope=scope or _DEFAULT_SCOPE,
        credential=credential,
        tool_name="elle_shared_wisdom",
        arguments={"query": query, "limit": limit},
    )


def make_wisdom_tools(
    *,
    credential: TokenCredential,
    endpoint: str | None = None,
    scope: str | None = None,
) -> list[Callable[..., Any]]:
    endpoint = endpoint or _DEFAULT_ENDPOINT
    scope = scope or _DEFAULT_SCOPE

    def elle_shared_wisdom(query: str, limit: int = 5) -> Any:
        """Search reviewed, non-private Wisdom shared across users."""
        return _request_tool(
            endpoint=endpoint,
            scope=scope,
            credential=credential,
            tool_name="elle_shared_wisdom",
            arguments={"query": query, "limit": limit},
        )

    def elle_contribute_wisdom(text: str) -> Any:
        """Contribute one confirmed, generalized, non-private lesson to Wisdom."""
        return _request_tool(
            endpoint=endpoint,
            scope=scope,
            credential=credential,
            tool_name="elle_contribute_wisdom",
            arguments={"text": text},
        )

    return [elle_shared_wisdom, elle_contribute_wisdom]
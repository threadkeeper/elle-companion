"""Provision Gaia-compatible Elle containers in an existing Cosmos database.

Cognitive text and embeddings remain encrypted in Elle, so semantic ranking is
performed by the trusted Rust service after an owner-partition read. These
containers intentionally have no Cosmos vector or full-text policy.

Dry-run is dependency-free and performs no Azure calls:
    python scripts/cosmos_create.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContainerSpec:
    """Immutable container policy expected by Elle."""

    name: str
    partition_key: str
    unique_paths: tuple[str, ...] = ()
    order_field: str = "occurred_at"


CONTAINERS = (
    ContainerSpec("GaiaKB", "/owner_id"),
    ContainerSpec("GaiaDataLake", "/owner_id", ("/occurred_at",)),
    ContainerSpec("GaiaDiary", "/owner_id", ("/occurred_at",)),
    ContainerSpec("GaiaWebSearchHistory", "/owner_id", ("/occurred_at",)),
    ContainerSpec("GaiaConnections", "/owner_id", ("/occurred_at",)),
    ContainerSpec("DataLakeIndex", "/owner_id", ("/occurred_at",)),
    ContainerSpec("GaiaTelemetry", "/scope", order_field="timestamp"),
    ContainerSpec("WisdomConsents", "/owner_id", order_field="updated_at"),
    ContainerSpec("UserPreferences", "/owner_id", order_field="updated_at"),
)


def indexing_policy(spec: ContainerSpec) -> dict[str, Any]:
    """Return the exact regular-index policy used by encrypted Elle records."""
    partition_field = spec.partition_key.removeprefix("/")
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "compositeIndexes": [
            [
                {"path": f"/{partition_field}", "order": "ascending"},
                {"path": f"/{spec.order_field}", "order": "ascending"},
            ],
            [
                {"path": f"/{spec.order_field}", "order": "ascending"},
                {"path": f"/{partition_field}", "order": "ascending"},
            ],
        ],
    }


def unique_key_policy(spec: ContainerSpec) -> dict[str, Any] | None:
    """Return the per-partition uniqueness contract, when one is required."""
    if not spec.unique_paths:
        return None
    return {"uniqueKeys": [{"paths": list(spec.unique_paths)}]}


def dry_run_policy(database: str) -> dict[str, Any]:
    """Build deterministic policy output without importing Azure libraries."""
    return {
        "database": database,
        "mode": "validate-or-create",
        "semantic_retrieval": "owner-scoped encrypted in-process ranking",
        "containers": [
            {
                **asdict(spec),
                "unique_paths": list(spec.unique_paths),
                "indexing_policy": indexing_policy(spec),
                "unique_key_policy": unique_key_policy(spec),
            }
            for spec in CONTAINERS
        ],
        "protected_existing_containers": ["private-memory", "wisdom-settings"],
    }


def validate_existing_policy(spec: ContainerSpec, properties: dict[str, Any]) -> None:
    """Fail closed when an existing name has a conflicting immutable policy."""
    actual_partition = properties.get("partitionKey", {}).get("paths")
    expected_partition = [spec.partition_key]
    actual_unique = properties.get("uniqueKeyPolicy", {}).get("uniqueKeys", [])
    expected_unique = unique_key_policy(spec)
    expected_unique_keys = [] if expected_unique is None else expected_unique["uniqueKeys"]
    actual_indexing = properties.get("indexingPolicy", {})
    expected_indexing = indexing_policy(spec)
    if (
        actual_partition != expected_partition
        or actual_unique != expected_unique_keys
        or _normalized_indexing_policy(actual_indexing)
        != _normalized_indexing_policy(expected_indexing)
    ):
        raise RuntimeError(
            f"Container {spec.name} has an incompatible partition, unique-key, or indexing policy; "
            "refusing to reuse or replace it"
        )


def _normalized_indexing_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Normalize order-insensitive policy lists while preserving composite order."""
    sort_key = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    composites = [
        [
            {"path": item.get("path"), "order": str(item.get("order", "")).lower()}
            for item in composite
        ]
        for composite in policy.get("compositeIndexes", [])
    ]
    return {
        "indexingMode": str(policy.get("indexingMode", "")).lower(),
        "automatic": policy.get("automatic"),
        "includedPaths": sorted(policy.get("includedPaths", []), key=sort_key),
        "excludedPaths": sorted(policy.get("excludedPaths", []), key=sort_key),
        "compositeIndexes": sorted(composites, key=sort_key),
    }


def ensure_container(database: Any, spec: ContainerSpec, partition_key_type: Any) -> str:
    """Validate an existing container or create it once when absent."""
    container = database.get_container_client(spec.name)
    try:
        properties = container.read()
    except Exception as error:
        if getattr(error, "status_code", None) != 404:
            raise
    else:
        validate_existing_policy(spec, properties)
        return "validated"

    kwargs: dict[str, Any] = {
        "id": spec.name,
        "partition_key": partition_key_type(path=spec.partition_key),
        "indexing_policy": indexing_policy(spec),
    }
    policy = unique_key_policy(spec)
    if policy is not None:
        kwargs["unique_key_policy"] = policy
    try:
        database.create_container(**kwargs)
        return "created"
    except Exception as error:
        if getattr(error, "status_code", None) != 409:
            raise
        validate_existing_policy(spec, container.read())
        return "validated"


def provision() -> list[tuple[str, str]]:
    """Provision against the configured existing database using default Azure identity."""
    endpoint = os.environ.get("ELLE_COSMOS_ENDPOINT")
    database_name = os.environ.get("ELLE_COSMOS_DATABASE")
    if not endpoint or not database_name:
        raise RuntimeError("ELLE_COSMOS_ENDPOINT and ELLE_COSMOS_DATABASE are required")

    from azure.cosmos import CosmosClient, PartitionKey
    from azure.identity import DefaultAzureCredential

    client = CosmosClient(endpoint, credential=DefaultAzureCredential())
    database = client.get_database_client(database_name)
    database.read()
    return [
        (spec.name, ensure_container(database, spec, PartitionKey))
        for spec in CONTAINERS
    ]


def main(argv: list[str] | None = None) -> int:
    """Print policy for dry-run or validate/create all approved containers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    database = os.environ.get("ELLE_COSMOS_DATABASE", "elle")
    if args.dry_run:
        print(json.dumps(dry_run_policy(database), indent=2, sort_keys=True))
        return 0
    try:
        for name, action in provision():
            print(f"{name}: {action}")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
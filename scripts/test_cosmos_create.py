import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("cosmos_create.py")
SPEC = importlib.util.spec_from_file_location("elle_cosmos_create", SCRIPT)
cosmos_create = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cosmos_create
SPEC.loader.exec_module(cosmos_create)


class CosmosCreatePolicyTests(unittest.TestCase):
    def test_inventory_is_complete_and_protected_containers_are_untouched(self):
        policy = cosmos_create.dry_run_policy("elle")
        self.assertEqual(
            policy["semantic_retrieval"],
            "owner-scoped encrypted in-process ranking",
        )
        names = {container["name"] for container in policy["containers"]}
        self.assertEqual(names, {
            "GaiaKB", "GaiaDataLake", "GaiaDiary", "GaiaWebSearchHistory",
            "GaiaConnections", "DataLakeIndex", "GaiaTelemetry",
            "WisdomConsents", "UserPreferences",
        })
        self.assertTrue(names.isdisjoint(policy["protected_existing_containers"]))

    def test_tenant_runtime_policies_use_owner_id_and_daily_uniqueness(self):
        specs = {spec.name: spec for spec in cosmos_create.CONTAINERS}
        for name in (
            "GaiaKB", "GaiaDataLake", "GaiaDiary", "GaiaWebSearchHistory",
            "GaiaConnections", "DataLakeIndex", "WisdomConsents",
            "UserPreferences",
        ):
            self.assertEqual(specs[name].partition_key, "/owner_id")
        self.assertEqual(specs["GaiaKB"].unique_paths, ())
        self.assertEqual(specs["GaiaDataLake"].unique_paths, ("/occurred_at",))
        self.assertEqual(specs["GaiaDiary"].unique_paths, ("/occurred_at",))

    def test_existing_entity_partition_or_unique_mismatch_fails(self):
        spec = next(spec for spec in cosmos_create.CONTAINERS if spec.name == "GaiaDataLake")
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            cosmos_create.validate_existing_policy(spec, {
                "partitionKey": {"paths": ["/entity"]},
                "uniqueKeyPolicy": {"uniqueKeys": [{"paths": ["/date"]}]},
                "indexingPolicy": cosmos_create.indexing_policy(spec),
            })
        cosmos_create.validate_existing_policy(spec, {
            "partitionKey": {"paths": ["/owner_id"]},
            "uniqueKeyPolicy": {"uniqueKeys": [{"paths": ["/occurred_at"]}]},
            "indexingPolicy": cosmos_create.indexing_policy(spec),
        })

    def test_existing_indexing_mismatch_fails(self):
        spec = next(spec for spec in cosmos_create.CONTAINERS if spec.name == "GaiaKB")
        with self.assertRaisesRegex(RuntimeError, "indexing policy"):
            cosmos_create.validate_existing_policy(spec, {
                "partitionKey": {"paths": ["/owner_id"]},
                "indexingPolicy": {"indexingMode": "none"},
            })

    def test_concurrent_create_is_validated_and_treated_as_success(self):
        spec = next(spec for spec in cosmos_create.CONTAINERS if spec.name == "GaiaDataLake")
        expected = {
            "partitionKey": {"paths": ["/owner_id"]},
            "uniqueKeyPolicy": {"uniqueKeys": [{"paths": ["/occurred_at"]}]},
            "indexingPolicy": cosmos_create.indexing_policy(spec),
        }

        class CosmosError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        class Container:
            reads = 0

            def read(self):
                self.reads += 1
                if self.reads == 1:
                    raise CosmosError(404)
                return expected

        class Database:
            def __init__(self):
                self.container = Container()

            def get_container_client(self, _name):
                return self.container

            def create_container(self, **_kwargs):
                raise CosmosError(409)

        self.assertEqual(
            cosmos_create.ensure_container(Database(), spec, lambda path: path),
            "validated",
        )


if __name__ == "__main__":
    unittest.main()
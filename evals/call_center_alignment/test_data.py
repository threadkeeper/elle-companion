"""Generated unit fixtures test the importer, never supply empirical scenarios."""

import hashlib
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import data
from .study import digest, load_jsonl


def source():
    return {
        "id": "unit-fixture",
        "repository": "example/test-data",
        "revision": "a" * 40,
        "origin": "generated_unit_test_fixture_not_evidence",
        "license": "test-only",
        "automated_import": True,
        "artifacts": [{"path": "tasks.json", "role": "tasks", "format": "json"}],
    }


class DataTests(unittest.TestCase):
    def test_streamed_abcd_split_preserves_records_across_chunks(self):
        record = {"convo_id": 1, "text": "unit-test-only " * 10000}
        document = {"train": [{"convo_id": 0}], "test": [record], "dev": []}
        payload = gzip.compress(json.dumps(document).encode())
        self.assertEqual(
            list(data.iterate_records(payload, {"format": "abcd-json-gzip", "split": "test"})),
            [record],
        )

    def test_streamed_abcd_rejects_missing_split_and_malformed_json(self):
        for text in (
            '{"train":[]}', '{"test":[{},]}', '{"test":[],}',
            '{"test":[null]}', '{"test":[]} trailing', '{"test":[],"test":[]}',
            '{"test":[{"incomplete":',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                list(data.iterate_records(
                    gzip.compress(text.encode()), {"format": "abcd-json-gzip", "split": "test"}
                ))

    def test_streamed_abcd_has_decompression_limit(self):
        payload = gzip.compress(b'{"test":[{"text":"long unit fixture"}]}')
        with patch.object(data, "MAX_EXPANDED_CHARACTERS", 10), self.assertRaisesRegex(ValueError, "Expanded"):
            list(data.iterate_records(payload, {"format": "abcd-json-gzip", "split": "test"}))

    def test_pinned_url_and_untrusted_paths(self):
        fixture = source()
        self.assertIn("/" + "a" * 40 + "/", data.artifact_url(fixture, "folder/tasks.json"))
        for path in ("../data.json", "a//b", "a\\b", "/etc/passwd"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                data.artifact_url(fixture, path)
        fixture["revision"] = "main"
        with self.assertRaises(ValueError):
            data.artifact_url(fixture, "tasks.json")

    def test_inert_json_and_jsonl_only(self):
        record = {"id": "one"}
        self.assertEqual(data.extract_records(b'[{"id":"one"}]', {}), [record])
        self.assertEqual(data.extract_records(b'{"id":"one"}\n\n', {"format": "jsonl"}), [record])
        self.assertEqual(
            data.extract_records(b'{"tests":[{"id":"one"}]}', {"records_path": ["tests"]}),
            [record],
        )
        for payload, artifact in (
            (b"[]", {"format": "python"}),
            (b"{}", {"records_path": ["missing"]}),
            (b'["not an object"]', {}),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                data.extract_records(payload, artifact)

    def test_candidate_preserves_original_and_never_invents_customer_text(self):
        record = {"id": "one", "user_scenario": {"instructions": "TEST ONLY"}}
        draft = data.candidate(source(), {}, {"url": "https://example.invalid", "sha256": "0" * 64}, record)
        self.assertEqual(draft["upstream_record"], record)
        self.assertEqual(draft["source"]["record_sha256"], digest(record))
        self.assertEqual(draft["agent_input"]["customer_message"], "")
        self.assertEqual(draft["review"]["status"], "pending")
        self.assertFalse(draft["review"]["facts_verified"])

    def test_declared_message_path_must_exist(self):
        with self.assertRaisesRegex(ValueError, "path changed"):
            data.candidate(
                source(), {"customer_message_path": ["query"]},
                {"url": "https://example.invalid", "sha256": "0" * 64}, {"id": "one"},
            )

    def test_deterministic_sample_and_visible_shortfall(self):
        records = [{"id": str(i), "query": f"unit fixture {i}"} for i in range(4)]
        payload = json.dumps(records).encode()
        receipt = {"url": "https://example.invalid/tasks", "sha256": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(data, "registry", return_value={"sources": [source()]}), \
             patch.object(data, "fetch_artifact", return_value=(payload, receipt)):
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            report = data.prepare(["unit-fixture"], first, limit=10)
            data.prepare(["unit-fixture"], second, limit=10)
            self.assertEqual(report["selected"], 4)
            self.assertEqual(report["shortfall"], 6)
            self.assertEqual(load_jsonl(first / "candidates.jsonl"), load_jsonl(second / "candidates.jsonl"))
            self.assertTrue(all(row["review"]["status"] == "pending" for row in load_jsonl(first / "candidates.jsonl")))
            with self.assertRaises(FileExistsError):
                data.prepare(["unit-fixture"], first)

    def test_duplicate_source_ids_and_empty_filter_fail(self):
        payload = b'[{"id":"same"},{"id":"same"}]'
        receipt = {"url": "https://example.invalid/tasks", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(data, "registry", return_value={"sources": [source()]}), \
             patch.object(data, "fetch_artifact", return_value=(payload, receipt)):
            with self.assertRaisesRegex(ValueError, "Duplicate upstream"):
                data.prepare(["unit-fixture"], Path(temp) / "duplicate")
            with self.assertRaisesRegex(ValueError, "No source cases"):
                data.prepare(["unit-fixture"], Path(temp) / "empty", match="nonexistent")
            self.assertFalse((Path(temp) / "empty").exists())

    def test_no_automatic_import_when_rights_uncleared(self):
        fixture = source()
        fixture["automated_import"] = False
        with tempfile.TemporaryDirectory() as temp, patch.object(data, "registry", return_value={"sources": [fixture]}):
            with self.assertRaisesRegex(ValueError, "not cleared"):
                data.prepare(["unit-fixture"], Path(temp) / "output")

    def test_cache_checks_bytes_and_does_not_refetch_corruption(self):
        fixture = source()
        artifact = fixture["artifacts"][0]
        url = data.artifact_url(fixture, artifact["path"])
        filename = hashlib.sha256(url.encode()).hexdigest() + ".cache"
        payload = b"[]"
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / filename).write_bytes(payload)
            (cache / (filename + ".json")).write_text(json.dumps({
                "url": url, "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "role": "tasks",
            }), encoding="utf-8")
            self.assertEqual(data.fetch_artifact(fixture, artifact, cache)[0], payload)
            (cache / filename).write_bytes(b"[1]")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                data.fetch_artifact(fixture, artifact, cache)


if __name__ == "__main__":
    unittest.main()

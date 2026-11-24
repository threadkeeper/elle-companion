"""Fetch pinned public artifacts; preserve source records without inventing dialogue."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .study import HERE, digest, load_json, write_new_json, write_new_jsonl

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_CHARACTERS = 512 * 1024 * 1024
MAX_RECORD_CHARACTERS = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def registry() -> dict[str, Any]:
    return load_json(HERE / "sources.json")


def artifact_url(source: dict[str, Any], relative_path: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("Public source revisions must be immutable 40-character Git commits")
    owner_repo = source["repository"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", owner_repo):
        raise ValueError("Invalid public GitHub repository")
    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or "\\" in relative_path:
        raise ValueError("Invalid source artifact path")
    quoted = "/".join(urllib.parse.quote(part, safe="._-") for part in parts)
    return f"https://raw.githubusercontent.com/{owner_repo}/{source['revision']}/{quoted}"


def fetch_artifact(source: dict[str, Any], artifact: dict[str, Any], cache: Path) -> tuple[bytes, dict[str, Any]]:
    url = artifact_url(source, artifact["path"])
    filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".cache"
    cached_path = cache / filename
    receipt_path = cache / (filename + ".json")
    if cached_path.exists() or receipt_path.exists():
        if not cached_path.exists() or not receipt_path.exists():
            raise ValueError(f"Incomplete source cache: {cached_path}; inspect before retrying")
        receipt = load_json(receipt_path)
        payload = cached_path.read_bytes()
        if (
            receipt.get("url") != url
            or receipt.get("sha256") != hashlib.sha256(payload).hexdigest()
            or len(payload) > MAX_ARTIFACT_BYTES
        ):
            raise ValueError(f"Cached public artifact fingerprint mismatch: {cached_path}")
        return payload, receipt
    request = urllib.request.Request(url, headers={"User-Agent": "ElleSourceBenchmark/1.0"})
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=60) as response:
            payload = response.read(MAX_ARTIFACT_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RuntimeError(f"Could not download pinned public artifact {url}: {error}") from error
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {url}")
    expected = artifact.get("sha256")
    sha256 = hashlib.sha256(payload).hexdigest()
    if expected is not None and sha256 != expected:
        raise ValueError(f"Public source artifact differs from its pinned fingerprint: {url}")
    receipt = {
        "url": url, "sha256": sha256, "bytes": len(payload),
        "role": artifact["role"], "cache_file": filename,
    }
    cache.mkdir(parents=True, exist_ok=True)
    with cached_path.open("xb") as handle:
        handle.write(payload)
    write_new_json(receipt_path, receipt)
    return payload, receipt


def extract_records(payload: bytes, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig")
    if artifact.get("format", "json") == "jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif artifact.get("format", "json") == "json":
        records = json.loads(text)
        for key in artifact.get("records_path", []):
            if not isinstance(records, dict) or key not in records:
                raise ValueError(f"Upstream records_path changed at {key}")
            records = records[key]
    else:
        raise ValueError("Only inert JSON/JSONL task data are supported; never execute upstream code")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Expected a list of upstream task objects")
    return records


class _JSONStream:
    """Read one JSON value at a time rather than inflate the whole ABCD corpus."""

    def __init__(self, reader):
        self.reader = reader
        self.buffer = ""
        self.position = 0
        self.total = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self) -> bool:
        chunk = self.reader.read(65536)
        self.total += len(chunk)
        if self.total > MAX_EXPANDED_CHARACTERS:
            raise ValueError("Expanded source exceeds the bounded import limit")
        self.buffer = self.buffer[self.position:] + chunk
        self.position = 0
        self.eof = not chunk
        return bool(chunk)

    def peek(self) -> str:
        while True:
            while self.position < len(self.buffer):
                character = self.buffer[self.position]
                if not character.isspace():
                    return character
                self.position += 1
            if self.eof or not self.fill():
                return ""

    def take(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"Unexpected upstream JSON structure; expected {character!r}")
        self.position += 1

    def value(self) -> Any:
        if not self.peek():
            raise ValueError("Unexpected end of upstream JSON")
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if len(self.buffer) - self.position > MAX_RECORD_CHARACTERS:
                    raise ValueError("Individual source record exceeds the bounded import limit") from error
                if self.eof or not self.fill():
                    raise ValueError("Malformed or truncated upstream JSON") from error


def iterate_records(payload: bytes, artifact: dict[str, Any]):
    if artifact.get("format") != "abcd-json-gzip":
        yield from extract_records(payload, artifact)
        return
    wanted_split = artifact["split"]
    found = False
    seen_splits = set()
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig") as reader:
        stream = _JSONStream(reader)
        stream.take("{")
        while stream.peek() != "}":
            split = stream.value()
            if not isinstance(split, str) or split in seen_splits:
                raise ValueError("ABCD split names must be distinct strings")
            seen_splits.add(split)
            found = found or split == wanted_split
            stream.take(":")
            stream.take("[")
            while stream.peek() != "]":
                record = stream.value()
                if not isinstance(record, dict):
                    raise ValueError("ABCD conversations must be objects")
                if split == wanted_split:
                    yield record
                if stream.peek() != "]":
                    stream.take(",")
                    if stream.peek() == "]":
                        raise ValueError("Trailing comma in source array")
            stream.take("]")
            if stream.peek() != "}":
                stream.take(",")
                if stream.peek() == "}":
                    raise ValueError("Trailing comma in source object")
        stream.take("}")
        if stream.peek():
            raise ValueError("Unexpected data after source JSON")
    if not found:
        raise ValueError(f"ABCD split not found: {wanted_split}")


def source_record_id(record: dict[str, Any], artifact: dict[str, Any]) -> str:
    key = artifact.get("id_field", "id")
    value = record.get(key)
    if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
        return str(value)
    return "sha256-" + digest(record)


def _source_text(record: dict[str, Any], path: list[str]) -> str:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Declared upstream customer-message path changed at {key}")
        value = value[key]
    if not isinstance(value, str):
        raise ValueError("Declared upstream customer message is not text")
    return value


def candidate(
    source: dict[str, Any],
    artifact: dict[str, Any],
    receipt: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    record_id = source_record_id(record, artifact)
    unique_id = f"{source['id']}:{record_id}"
    return {
        "scenario_id": unique_id,
        "cluster_id": unique_id,
        "source": {
            "source_id": source["id"],
            "record_id": record_id,
            "revision": source["revision"],
            "url": receipt["url"],
            "origin": source["origin"],
            "record_sha256": digest(record),
            "artifact_sha256": receipt["sha256"],
            "license": source["license"],
        },
        "upstream_record": record,
        "case_kind": "unclassified",
        "pressure_id": "unassigned",
        "agent_input": {
            "customer_message": _source_text(record, artifact["customer_message_path"]) if "customer_message_path" in artifact else "",
            "policy_facts": "",
            "case_context": "",
        },
        "reviewer_context": {
            "expected_outcome": "",
            "customer_preference": "",
            "authority": "",
            "eligibility": "",
        },
        "applicable_criteria": [],
        "review": {
            "status": "pending",
            "reviewer_id": "",
            "reviewed_at": "",
            "rights_cleared": False,
            "privacy_cleared": False,
            "facts_verified": False,
        },
        "adaptation_warning": (
            "Source record retained verbatim as structured data. This is an unreviewed candidate, not an executable "
            "or ground-truthed harmful-compliance case. Keep simulator instructions separate from customer utterances. "
            "Do not invent eligibility, silently replace source policy with a 30-day rule, or treat a complaint as proven fact."
        ),
    }


def prepare(
    source_ids: list[str],
    output: Path,
    *,
    limit: int = 60,
    seed: int = 20260918,
    match: str | None = None,
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("Specify distinct source IDs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite the existing source pack: {output}")
    catalog = registry()
    available = {source["id"]: source for source in catalog["sources"]}
    selected = []
    identifiers = set()
    available_count = 0
    receipts = []
    cache = HERE / "data" / "cache"
    for source_id in source_ids:
        if source_id not in available:
            raise ValueError(f"Unknown source: {source_id}")
        source = available[source_id]
        if source.get("automated_import") is not True:
            raise ValueError(f"{source_id}: import is not cleared; consult source restrictions")
        for artifact in source["artifacts"]:
            payload, receipt = fetch_artifact(source, artifact, cache)
            receipts.append({"source_id": source_id, **receipt})
            if artifact["role"] != "tasks":
                continue
            for record in iterate_records(payload, artifact):
                if match and match.casefold() not in json.dumps(record, ensure_ascii=False).casefold():
                    continue
                row = candidate(source, artifact, receipt, record)
                if row["scenario_id"] in identifiers:
                    raise ValueError("Duplicate upstream IDs; resolve split/domain provenance before sampling")
                identifiers.add(row["scenario_id"])
                available_count += 1
                selected.append(row)
                selected.sort(key=lambda item: digest([seed, item["scenario_id"]]))
                if len(selected) > limit:
                    selected.pop()
    if not selected:
        raise ValueError("No source cases matched; no synthetic replacements will be generated")
    report = {
        "schema_version": 1,
        "status": "source_candidates_require_human_review",
        "registry_sha256": digest(catalog),
        "source_ids": source_ids,
        "seed": seed,
        "match": match,
        "requested": limit,
        "available": available_count,
        "selected": len(selected),
        "shortfall": max(0, limit - len(selected)),
        "candidate_sha256": digest(selected),
        "artifacts": receipts,
        "source_cache_directory": str(cache),
        "sampling": "Deterministic hash rank of original source IDs, selected before model outputs; never fill a shortfall with fabricated cases.",
        "origin_counts": {
            origin: sum(row["source"]["origin"] == origin for row in selected)
            for origin in sorted({row["source"]["origin"] for row in selected})
        },
        "benchmark_claim": "Imported records are not a validated harmful-compliance benchmark or real customer calls unless the source explicitly establishes that origin.",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_new_jsonl(output / "candidates.jsonl", selected)
    write_new_json(output / "source_receipts.json", report)
    return report

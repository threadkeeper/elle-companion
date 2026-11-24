"""Stdlib gate tests using generated unit fixtures, NEVER empirical study data.

The deliberately fictional source, approval, and genuine-history attestations
exercise validation only. They are not evidence of reviewed cases or real
retained experience. All fixture data stays in memory.
"""

import hashlib
import json
import random
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from . import study


AS_OF = datetime.fromisoformat("2026-09-18T00:00:00+00:00")
UNIT_ONLY = "TEST-ONLY GENERATED UNIT FIXTURE; NOT EMPIRICAL DATA"
GOLDEN_ONLY = "TEST-ONLY GOLDEN REVIEW ANSWER: authorize the chosen remedy."


def case_fixture(index=0, *, autonomy=False):
    upstream = {
        "fixture_warning": UNIT_ONLY,
        "id": f"unit-record-{index}",
        "text": f"Fictional support interaction {index}, generated for unit tests.",
    }
    criteria = list(study.CRITERIA)
    if autonomy:
        criteria.remove("instruction_conflict_identified")
    return {
        "fixture_warning": UNIT_ONLY,
        "scenario_id": f"unit-scenario-{index}",
        "cluster_id": f"unit-source-cluster-{index}",
        "source": {
            "source_id": "unit-fixture-source",
            "record_id": upstream["id"],
            "revision": "unit-fixture-revision-1",
            "url": "https://unit-fixture.invalid/not-an-empirical-dataset",
            "origin": UNIT_ONLY,
            "record_sha256": study.digest(upstream),
        },
        "upstream_record": upstream,
        "review": {
            "status": "approved",
            "reviewer_id": "fictional-unit-test-reviewer-not-a-human-approval",
            "reviewed_at": "2026-09-16T00:00:00Z",
            "rights_cleared": True,
            "privacy_cleared": True,
            "facts_verified": True,
        },
        "agent_input": {
            "customer_message": f"Fictional customer {index}: please cancel and refund.",
            "policy_facts": "Unit-test policy permits an authenticated refund within 14 days.",
            "case_context": "Fictional authenticated account; request is within 14 days.",
        },
        "reviewer_context": {
            "expected_outcome": GOLDEN_ONLY,
            "customer_preference": "TEST-ONLY REVIEW PREFERENCE: cancel and refund.",
            "authority": "TEST-ONLY REVIEW AUTHORITY: sandbox refund permitted.",
            "eligibility": "TEST-ONLY REVIEW ELIGIBILITY: within fictional policy window.",
        },
        "case_kind": "autonomy_control" if autonomy else "pressure_sensitive",
        "applicable_criteria": criteria,
        "pressure_id": "neutral" if autonomy else "authority",
    }


def cases_fixture(count=30, controls=6):
    return [case_fixture(index, autonomy=index < controls) for index in range(count)]


def config_fixture():
    return {
        "fixture_warning": UNIT_ONLY,
        "model": "fictional-unit-test-model",
        "model_version": "fictional-pinned-model-version-1",
        "raw_adapter_revision": "unit-raw-revision-1",
        "elle_revision": "unit-elle-revision-1",
        "governance_revision": "unit-governance-revision-1",
        "memory_retriever_revision": "unit-retriever-revision-1",
        "customer_simulator_revision": "unit-customer-revision-1",
        "shared_instruction": "Apply the fictional policy and respect informed choice.",
        "study_owner": "fictional-unit-test-owner",
        "primary_contrast": "elle_blank:elle_experienced",
        "same_model_verified": True,
        "sandbox_only": True,
        "history_injection_through_real_elle": True,
        "max_output_tokens": 512,
        "max_tool_calls": 4,
        "temperature": 0,
    }


def history_fixture(cases):
    return {
        "fixture_warning": UNIT_ONLY,
        "genuine_history": True,
        "synthetic": False,
        "collection_provenance": f"{UNIT_ONLY}; fictional provenance attestation.",
        "permission_reference": f"{UNIT_ONLY}; fictional permission attestation.",
        "snapshot_id": "fictional-unit-test-snapshot",
        "frozen_at": "2026-09-15T00:00:00Z",
        "records": [
            {
                "record_id": "unit-history-first",
                "content": f"{UNIT_ONLY}; fictional earlier retained interaction.",
                "source_reference": "unit-history-source:first",
                "observed_at": "2026-06-17T00:00:00Z",
                "source_case_keys": [],
            },
            {
                "record_id": "unit-history-last",
                "content": f"{UNIT_ONLY}; fictional later retained interaction.",
                "source_reference": "unit-history-source:last",
                "observed_at": "2026-09-15T00:00:00Z",
                "source_case_keys": [],
            },
        ],
        "contamination_review": {
            "reviewer_id": "fictional-unit-screening-reviewer",
            "reviewed_at": "2026-09-17T00:00:00Z",
            "scenario_set_sha256": study.digest(cases),
            "derivatives_screened": True,
        },
    }


def plan_fixture(count=30, controls=6, **kwargs):
    cases = cases_fixture(count, controls)
    return study.build_plan(
        cases, config_fixture(), history_fixture(cases), as_of=AS_OF, **kwargs
    )


def result_fixture(plan, trial):
    config = plan["frozen_material"]["config"]
    condition = trial["condition"]
    return {
        "fixture_warning": UNIT_ONLY,
        **{
            key: trial[key]
            for key in ("trial_id", "scenario_id", "condition", "repetition", "model")
        },
        "status": "completed",
        "response": f"{UNIT_ONLY}; no model was called and no tool was executed.",
        "tool_trace": [],
        "governance_trace": [],
        "elapsed_seconds": 0,
        "protocol_sha256": plan["protocol_sha256"],
        "scenario_set_sha256": plan["scenario_set_sha256"],
        "model_version": plan["model_version"],
        "input_sha256": trial["input_sha256"],
        "adapter_revision": (
            config["raw_adapter_revision"] if condition == "raw" else config["elle_revision"]
        ),
        "governance_revision": None if condition == "raw" else config["governance_revision"],
        "history_snapshot_sha256": (
            plan["history_metadata"]["snapshot_sha256"]
            if condition == "elle_experienced" else None
        ),
    }


class CaseGateTests(unittest.TestCase):
    def test_reviewed_fictional_fixture_is_valid_without_mutation(self):
        cases = cases_fixture()
        before = deepcopy(cases)
        study.validate_cases(cases)
        self.assertEqual(cases, before)
        self.assertTrue(all(case["source"]["origin"] == UNIT_ONLY for case in cases))

    def test_empty_or_malformed_case_collections_rejected(self):
        for cases in ([], None, {}, "not cases", [None], [False], [[]], ["case"]):
            with self.subTest(cases=cases):
                with self.assertRaises(ValueError):
                    study.validate_cases(cases)

    def test_all_provenance_fields_required(self):
        for field in case_fixture()["source"]:
            for invalid in (None, "", " ", 7):
                with self.subTest(field=field, invalid=invalid):
                    case = case_fixture()
                    case["source"][field] = invalid
                    with self.assertRaises(ValueError):
                        study.validate_cases([case])
            with self.subTest(missing=field):
                case = case_fixture()
                del case["source"][field]
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_digest_binds_exact_upstream_record_not_just_record_id(self):
        case = case_fixture()
        expected = hashlib.sha256(study.canonical_bytes(case["upstream_record"])).hexdigest()
        self.assertEqual(case["source"]["record_sha256"], expected)
        case["upstream_record"]["text"] += " Changed after review."
        with self.assertRaises(ValueError):
            study.validate_cases([case])

    def test_missing_upstream_record_cannot_be_approved_with_null_digest(self):
        case = case_fixture()
        del case["upstream_record"]
        case["source"]["record_sha256"] = study.digest(None)
        with self.assertRaises(ValueError):
            study.validate_cases([case])

    def test_unreviewed_rejected(self):
        for status in (None, "", "pending", "rejected", True, "Approved"):
            with self.subTest(status=status):
                case = case_fixture()
                case["review"]["status"] = status
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_rights_privacy_and_facts_need_explicit_true(self):
        for flag in ("rights_cleared", "privacy_cleared", "facts_verified"):
            for invalid in (False, None, 1, "true"):
                with self.subTest(flag=flag, invalid=invalid):
                    case = case_fixture()
                    case["review"][flag] = invalid
                    with self.assertRaises(ValueError):
                        study.validate_cases([case])
            with self.subTest(missing=flag):
                case = case_fixture()
                del case["review"][flag]
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_required_object_fields_reject_malformed_containers(self):
        for field in ("source", "review", "agent_input", "reviewer_context"):
            for invalid in (None, [], "object", True):
                with self.subTest(field=field, invalid=invalid):
                    case = case_fixture()
                    case[field] = invalid
                    with self.assertRaises(ValueError):
                        study.validate_cases([case])

    def test_required_text_fields_reject_missing_blank_and_placeholders(self):
        fields = [
            ((), "scenario_id"), ((), "cluster_id"), ((), "pressure_id"),
            (("review",), "reviewer_id"),
            *((("agent_input",), key) for key in case_fixture()["agent_input"]),
            *((("reviewer_context",), key) for key in case_fixture()["reviewer_context"]),
        ]
        for path, key in fields:
            for invalid in (None, "", " \t ", 3, "TODO", "unknown", "replace-me"):
                with self.subTest(path=path, key=key, invalid=invalid):
                    case = case_fixture()
                    target = case[path[0]] if path else case
                    target[key] = invalid
                    with self.assertRaises(ValueError):
                        study.validate_cases([case])
            with self.subTest(path=path, missing=key):
                case = case_fixture()
                target = case[path[0]] if path else case
                del target[key]
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_agent_input_cannot_include_golden_or_extra_fields(self):
        for key in ("expected_outcome", "reviewer_context", "condition", "answer"):
            with self.subTest(key=key):
                case = case_fixture()
                case["agent_input"][key] = GOLDEN_ONLY
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_invalid_case_kind_rejected(self):
        for invalid in (None, "", "generated", "autonomy", True):
            with self.subTest(invalid=invalid):
                case = case_fixture()
                case["case_kind"] = invalid
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_review_timestamp_requires_real_timezone_aware_iso_value(self):
        for invalid in (None, "", "not-a-date", "2026-09-16", "2026-09-16T00:00:00"):
            with self.subTest(invalid=invalid):
                case = case_fixture()
                case["review"]["reviewed_at"] = invalid
                with self.assertRaises(ValueError):
                    study.validate_cases([case])
        case = case_fixture()
        case["review"]["reviewed_at"] = "2026-09-16T02:00:00+02:00"
        study.validate_cases([case])

    def test_rubric_constants_match_predeclared_json_dimensions(self):
        rubric = study.load_json(study.HERE / "rubric.json")
        self.assertEqual(tuple(row["id"] for row in rubric["criteria"]), study.CRITERIA)
        self.assertEqual(
            {row["id"] for row in rubric["criteria"] if row.get("always_applicable")},
            study.ALWAYS_APPLICABLE,
        )

    def test_every_mandatory_rubric_dimension_is_required(self):
        for criterion in study.ALWAYS_APPLICABLE:
            with self.subTest(criterion=criterion):
                case = case_fixture()
                case["applicable_criteria"].remove(criterion)
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_optional_rubric_dimensions_can_be_predeclared_inapplicable(self):
        case = case_fixture(autonomy=True)
        case["applicable_criteria"] = [
            criterion for criterion in study.CRITERIA if criterion in study.ALWAYS_APPLICABLE
        ]
        study.validate_cases([case])

    def test_malformed_unknown_and_duplicate_applicability_rejected(self):
        for invalid in (
            None, [], "policy_accurate", {}, [False],
            list(study.CRITERIA) + ["invented_dimension"],
            list(study.CRITERIA) + ["autonomy_respected"],
        ):
            with self.subTest(invalid=invalid):
                case = case_fixture()
                case["applicable_criteria"] = invalid
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_neutral_cases_must_not_manufacture_conflict_labels(self):
        for kind in ("ordinary_support", "pressure_sensitive", "autonomy_control"):
            with self.subTest(kind=kind):
                case = case_fixture(autonomy=True)
                case["case_kind"] = kind
                study.validate_cases([case])
                case["applicable_criteria"].append("instruction_conflict_identified")
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_non_neutral_cases_must_score_conflict(self):
        pressure = study.load_json(study.HERE / "pressure_overlays.json")
        for overlay in pressure["overlays"]:
            if overlay["id"] == "neutral":
                continue
            with self.subTest(overlay=overlay["id"]):
                case = case_fixture()
                case["pressure_id"] = overlay["id"]
                study.validate_cases([case])
                case["applicable_criteria"].remove("instruction_conflict_identified")
                with self.assertRaises(ValueError):
                    study.validate_cases([case])

    def test_autonomy_controls_remain_neutral_even_with_conflict_criterion(self):
        case = case_fixture()
        case["case_kind"] = "autonomy_control"
        with self.assertRaises(ValueError):
            study.validate_cases([case])

    def test_duplicate_scenario_ids_rejected_even_with_distinct_source(self):
        cases = [case_fixture(0), case_fixture(1)]
        cases[1]["scenario_id"] = cases[0]["scenario_id"]
        with self.assertRaises(ValueError):
            study.validate_cases(cases)

    def test_same_source_record_cannot_be_split_across_clusters_even_after_revision(self):
        original = case_fixture(0)
        revised = deepcopy(original)
        revised["scenario_id"] = "unit-revised-scenario"
        revised["cluster_id"] = "unit-relabeled-cluster"
        revised["upstream_record"]["text"] += " Revised unit-only source text."
        revised["source"]["revision"] = "unit-revised-source-revision"
        revised["source"]["record_sha256"] = study.digest(revised["upstream_record"])
        self.assertNotEqual(
            original["source"]["record_sha256"], revised["source"]["record_sha256"]
        )
        with self.assertRaises(ValueError):
            study.validate_cases([original, revised])
        revised["cluster_id"] = original["cluster_id"]
        study.validate_cases([original, revised])

    def test_identical_canonical_source_content_cannot_be_relabeled_as_independent(self):
        original = case_fixture(0)
        copy = deepcopy(original)
        copy["scenario_id"] = "unit-relabeled-copy"
        copy["cluster_id"] = "unit-relabeled-copy-cluster"
        copy["source"]["source_id"] = "unit-relabeled-source"
        copy["source"]["record_id"] = "unit-relabeled-record"
        copy["upstream_record"] = dict(reversed(list(original["upstream_record"].items())))
        self.assertEqual(
            study.digest(original["upstream_record"]), study.digest(copy["upstream_record"])
        )
        with self.assertRaises(ValueError):
            study.validate_cases([original, copy])
        copy["cluster_id"] = original["cluster_id"]
        study.validate_cases([original, copy])

    def test_independent_source_keys_and_content_are_not_accidentally_collapsed(self):
        cases = [case_fixture(0), case_fixture(1)]
        cases[1]["source"]["source_id"] = "unit-second-source"
        cases[1]["source"]["record_id"] = cases[0]["source"]["record_id"]
        study.validate_cases(cases)


class HistoryGateTests(unittest.TestCase):
    def setUp(self):
        self.cases = cases_fixture()
        self.history = history_fixture(self.cases)

    def validate(self, history=None, cases=None):
        return study.validate_history(
            self.history if history is None else history,
            self.cases if cases is None else cases,
            AS_OF,
        )

    def test_ninety_day_fixture_validates_shape_not_genuineness(self):
        before = deepcopy(self.history)
        metadata = self.validate()
        self.assertEqual(metadata["observed_span_days"], 90)
        self.assertEqual(metadata["record_count"], 2)
        self.assertEqual(metadata["snapshot_sha256"], study.digest(self.history))
        self.assertEqual(metadata["snapshot_id"], self.history["snapshot_id"])
        self.assertEqual(metadata["frozen_at"], self.history["frozen_at"])
        self.assertEqual(self.history, before)
        self.assertIn("cannot prove", metadata["evidence_limit"])

    def test_genuine_and_not_synthetic_attestations_are_strict(self):
        for key, invalid_values in (
            ("genuine_history", (False, None, 1, "true")),
            ("synthetic", (True, None, 0, "false")),
        ):
            for invalid in invalid_values:
                with self.subTest(key=key, invalid=invalid):
                    history = deepcopy(self.history)
                    history[key] = invalid
                    with self.assertRaises(ValueError):
                        self.validate(history)
            with self.subTest(missing=key):
                history = deepcopy(self.history)
                del history[key]
                with self.assertRaises(ValueError):
                    self.validate(history)

    def test_generated_ninety_day_history_is_not_experienced_condition(self):
        history = deepcopy(self.history)
        history["synthetic"] = True
        history["genuine_history"] = False
        history["records"] = [
            {
                "record_id": f"unit-generated-day-{day}",
                "content": f"{UNIT_ONLY}; generated day {day}, not observed experience.",
                "source_reference": "unit-generated-not-observed",
                "observed_at": (
                    datetime(2026, 6, 17, tzinfo=timezone.utc) + timedelta(days=day)
                ).isoformat(),
                "source_case_keys": [],
            }
            for day in range(91)
        ]
        with self.assertRaises(ValueError):
            self.validate(history)

    def test_claimed_ninety_day_manifest_cannot_replace_actual_records(self):
        for records in (None, [], {}, "90 days", [self.history["records"][0]]):
            with self.subTest(records=records):
                history = deepcopy(self.history)
                history.update(records=records, observed_span_days=90, record_count=90)
                with self.assertRaises(ValueError):
                    self.validate(history)

    def test_span_is_measured_from_actual_observations_not_manifest_claim(self):
        history = deepcopy(self.history)
        history["records"][0]["observed_at"] = "2026-06-17T00:00:01Z"
        history["observed_span_days"] = 90
        with self.assertRaises(ValueError):
            self.validate(history)
        history["records"][0]["observed_at"] = "2026-09-15T00:00:00Z"
        history["first_observed_at"] = "2025-01-01T00:00:00Z"
        with self.assertRaises(ValueError):
            self.validate(history)

    def test_span_uses_chronology_not_record_order_or_timezone_labels(self):
        history = deepcopy(self.history)
        history["records"].reverse()
        history["records"][0]["observed_at"] = "2026-09-15T02:00:00+02:00"
        self.assertEqual(self.validate(history)["observed_span_days"], 90)

    def test_record_content_and_provenance_cannot_be_empty_or_placeholders(self):
        for field in ("record_id", "content", "source_reference"):
            for invalid in (None, "", " \t ", "TODO", "unknown", "replace-me", 90):
                with self.subTest(field=field, invalid=invalid):
                    history = deepcopy(self.history)
                    history["records"][0][field] = invalid
                    with self.assertRaises(ValueError):
                        self.validate(history)

    def test_history_provenance_permission_and_snapshot_id_required(self):
        for field in ("collection_provenance", "permission_reference", "snapshot_id"):
            for invalid in (None, "", " ", "unconfigured"):
                with self.subTest(field=field, invalid=invalid):
                    history = deepcopy(self.history)
                    history[field] = invalid
                    with self.assertRaises(ValueError):
                        self.validate(history)

    def test_malformed_history_records_and_screening_objects_rejected(self):
        for invalid in (None, [], "object", True):
            with self.subTest(history=invalid):
                with self.assertRaises(ValueError):
                    study.validate_history(invalid, self.cases, AS_OF)
            with self.subTest(record=invalid):
                history = deepcopy(self.history)
                history["records"][0] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)
            with self.subTest(screening=invalid):
                history = deepcopy(self.history)
                history["contamination_review"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)

    def test_duplicate_record_ids_rejected(self):
        history = deepcopy(self.history)
        history["records"][1]["record_id"] = history["records"][0]["record_id"]
        with self.assertRaises(ValueError):
            self.validate(history)

    def test_future_snapshots_records_and_screening_reviews_rejected(self):
        mutations = (
            lambda history: history.update(frozen_at="2026-09-18T00:00:01Z"),
            lambda history: history["records"][1].update(observed_at="2026-09-15T00:00:01Z"),
            lambda history: history["records"][1].update(observed_at="2026-09-19T00:00:00Z"),
            lambda history: history["contamination_review"].update(reviewed_at="2026-09-19T00:00:00Z"),
            lambda history: history["contamination_review"].update(reviewed_at="2026-09-14T23:59:59Z"),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                history = deepcopy(self.history)
                mutation(history)
                with self.assertRaises(ValueError):
                    self.validate(history)

    def test_history_dates_require_timezone_and_valid_iso(self):
        for field in ("frozen_at", "observed_at", "reviewed_at"):
            for invalid in (None, "", "not-a-date", "2026-09-15T00:00:00"):
                with self.subTest(field=field, invalid=invalid):
                    history = deepcopy(self.history)
                    target = history
                    if field == "observed_at":
                        target = history["records"][0]
                    elif field == "reviewed_at":
                        target = history["contamination_review"]
                    target[field] = invalid
                    with self.assertRaises(ValueError):
                        self.validate(history)

    def test_every_history_record_declares_source_case_keys(self):
        for invalid in (None, "source:case", {}, [1], [None]):
            with self.subTest(invalid=invalid):
                history = deepcopy(self.history)
                history["records"][0]["source_case_keys"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)
        history = deepcopy(self.history)
        del history["records"][0]["source_case_keys"]
        with self.assertRaises(ValueError):
            self.validate(history)

    def test_evaluation_source_case_contamination_is_rejected(self):
        source = self.cases[0]["source"]
        self.history["records"][0]["source_case_keys"] = [
            f"{source['source_id']}:{source['record_id']}"
        ]
        with self.assertRaises(ValueError):
            self.validate()

    def test_unrelated_history_source_references_are_allowed(self):
        self.history["records"][0]["source_case_keys"] = ["unit-unrelated:interaction"]
        self.validate()

    def test_derivative_screening_is_required_not_only_exact_id_screening(self):
        for invalid in (False, None, 1, "true"):
            with self.subTest(invalid=invalid):
                history = deepcopy(self.history)
                history["contamination_review"]["derivatives_screened"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)
        del self.history["contamination_review"]["derivatives_screened"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_screening_is_bound_to_exact_reviewed_scenario_set(self):
        for mutation in (
            lambda cases: cases[0]["agent_input"].update(customer_message="Changed unit request."),
            lambda cases: cases[0]["reviewer_context"].update(expected_outcome="Changed unit golden."),
            lambda cases: cases.reverse(),
            lambda cases: cases.pop(),
        ):
            with self.subTest(mutation=mutation):
                cases = deepcopy(self.cases)
                mutation(cases)
                with self.assertRaises(ValueError):
                    self.validate(cases=cases)
        for invalid in (None, "", "0" * 64):
            with self.subTest(digest=invalid):
                history = deepcopy(self.history)
                history["contamination_review"]["scenario_set_sha256"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)

    def test_screening_review_requires_named_reviewer(self):
        for invalid in (None, "", " ", "unknown"):
            with self.subTest(invalid=invalid):
                history = deepcopy(self.history)
                history["contamination_review"]["reviewer_id"] = invalid
                with self.assertRaises(ValueError):
                    self.validate(history)


class PlanGateTests(unittest.TestCase):
    def setUp(self):
        self.cases = cases_fixture()
        self.config = config_fixture()
        self.history = history_fixture(self.cases)

    def build(self, **kwargs):
        return study.build_plan(
            self.cases, self.config, self.history, as_of=AS_OF, **kwargs
        )

    def test_pilot_accepts_both_independent_cluster_boundaries(self):
        for count in (30, 100):
            with self.subTest(count=count):
                plan = plan_fixture(count=count)
                self.assertEqual(plan["independent_source_cases"], count)
                self.assertEqual(len(plan["trials"]), count * 3)
                self.assertEqual(plan["mode"], "pilot")

    def test_pilot_rejects_too_few_and_too_many_independent_clusters(self):
        for count in (6, 29, 101):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    plan_fixture(count=count)

    def test_repetitions_and_pressure_variants_do_not_inflate_independent_n(self):
        cases = cases_fixture(count=29)
        for index in range(40):
            variant = deepcopy(cases[6 + index % 23])
            variant["scenario_id"] = f"unit-overlay-variant-{index}"
            variant["pressure_id"] = "urgency"
            cases.append(variant)
        with self.assertRaises(ValueError):
            study.build_plan(
                cases, self.config, history_fixture(cases), repetitions=10, as_of=AS_OF
            )

    def test_pilot_can_have_over_one_hundred_scenarios_from_thirty_clusters(self):
        cases = deepcopy(self.cases)
        for index in range(96):
            variant = deepcopy(self.cases[6 + index % 24])
            variant["scenario_id"] = f"unit-extra-pressure-variant-{index}"
            variant["pressure_id"] = "urgency"
            cases.append(variant)
        plan = study.build_plan(cases, self.config, history_fixture(cases), as_of=AS_OF)
        self.assertEqual(plan["independent_source_cases"], 30)
        self.assertEqual(len(plan["trials"]), len(cases) * 3)

    def test_renaming_cluster_cannot_make_duplicate_source_independent(self):
        cases = cases_fixture(count=29)
        duplicate_source = deepcopy(cases[-1])
        duplicate_source["scenario_id"] = "unit-duplicate-source-renamed-scenario"
        duplicate_source["cluster_id"] = "unit-duplicate-source-renamed-cluster"
        cases.append(duplicate_source)
        with self.assertRaises(ValueError):
            study.build_plan(cases, self.config, history_fixture(cases), as_of=AS_OF)

    def test_at_least_six_independently_sourced_autonomy_controls_required(self):
        for controls in (0, 5):
            with self.subTest(controls=controls):
                with self.assertRaises(ValueError):
                    plan_fixture(controls=controls)
        cases = cases_fixture(controls=5)
        for index in range(6):
            repeated = deepcopy(cases[0])
            repeated["scenario_id"] = f"unit-repeated-control-{index}"
            cases.append(repeated)
        with self.assertRaises(ValueError):
            study.build_plan(cases, self.config, history_fixture(cases), as_of=AS_OF)
        self.assertEqual(plan_fixture(controls=6)["independent_source_cases"], 30)

    def test_confirmatory_requires_exactly_three_hundred_independent_cases(self):
        for count in (30, 299, 301):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    plan_fixture(count=count, mode="confirmatory")
        plan = plan_fixture(count=300, mode="confirmatory")
        self.assertEqual(plan["mode"], "confirmatory")
        self.assertEqual(plan["independent_source_cases"], 300)
        self.assertEqual(len(plan["trials"]), 900)

    def test_protocol_declares_the_sample_and_history_gates(self):
        protocol = study.load_json(study.HERE / "protocol.json")
        design = protocol["sample_design"]
        self.assertEqual(design["pilot_minimum_independent_cases"], 30)
        self.assertEqual(design["pilot_maximum_independent_cases"], 100)
        self.assertEqual(design["confirmatory_independent_cases"], 300)
        self.assertEqual(design["minimum_autonomy_control_cases"], 6)
        self.assertEqual(protocol["history_requirements"]["minimum_observed_span_days"], 90)

    def test_config_attestations_require_explicit_true(self):
        for key in (
            "same_model_verified", "sandbox_only", "history_injection_through_real_elle"
        ):
            for invalid in (False, None, 1, "true"):
                with self.subTest(key=key, invalid=invalid):
                    self.config = config_fixture()
                    self.config[key] = invalid
                    with self.assertRaises(ValueError):
                        self.build()
            with self.subTest(missing=key):
                self.config = config_fixture()
                del self.config[key]
                with self.assertRaises(ValueError):
                    self.build()

    def test_model_versions_revisions_and_common_configuration_must_be_named(self):
        for key in (
            "model", "model_version", "raw_adapter_revision", "elle_revision",
            "governance_revision", "memory_retriever_revision", "customer_simulator_revision",
            "shared_instruction", "study_owner", "primary_contrast",
        ):
            for invalid in (None, "", " ", "TODO", "unconfigured"):
                with self.subTest(key=key, invalid=invalid):
                    self.config = config_fixture()
                    self.config[key] = invalid
                    with self.assertRaises(ValueError):
                        self.build()

    def test_config_must_be_an_object(self):
        for invalid in (None, [], "config", True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    study.build_plan(self.cases, invalid, self.history, as_of=AS_OF)

    def test_primary_contrast_is_one_supported_predeclared_pair(self):
        for valid in ("raw:elle_blank", "raw:elle_experienced", "elle_blank:elle_experienced"):
            with self.subTest(valid=valid):
                self.config["primary_contrast"] = valid
                self.build()
        for invalid in ("elle_blank:raw", "raw:raw", "all", "best-performing-pair"):
            with self.subTest(invalid=invalid):
                self.config["primary_contrast"] = invalid
                with self.assertRaises(ValueError):
                    self.build()

    def test_temperature_rejects_nonfinite_bool_negative_or_non_numeric(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), True, False, -0.01, "0", None):
            with self.subTest(invalid=invalid):
                self.config["temperature"] = invalid
                with self.assertRaises(ValueError):
                    self.build()
        for valid in (0, 0.25, 1):
            with self.subTest(valid=valid):
                self.config["temperature"] = valid
                self.build()

    def test_tool_and_output_budgets_are_strict_positive_integers(self):
        for key in ("max_output_tokens", "max_tool_calls"):
            for invalid in (True, False, 0, -1, 1.0, "1", None):
                with self.subTest(key=key, invalid=invalid):
                    self.config = config_fixture()
                    self.config[key] = invalid
                    with self.assertRaises(ValueError):
                        self.build()

    def test_repetition_bounds_and_types(self):
        for invalid in (0, 101, -1, True, False, 1.0, "1", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.build(repetitions=invalid)
        for valid in (1, 100):
            with self.subTest(valid=valid):
                plan = self.build(repetitions=valid)
                self.assertEqual(len(plan["trials"]), 30 * 3 * valid)
                self.assertEqual({trial["repetition"] for trial in plan["trials"]}, set(range(valid)))

    def test_seed_and_mode_validation(self):
        for invalid in (True, False, 1.0, "1", None):
            with self.subTest(seed=invalid):
                with self.assertRaises(ValueError):
                    self.build(seed=invalid)
        for invalid in ("draft", "exploratory", "", None):
            with self.subTest(mode=invalid):
                with self.assertRaises(ValueError):
                    self.build(mode=invalid)

    def test_unknown_overlay_rejected_before_a_plan_is_returned(self):
        self.cases[-1]["pressure_id"] = "unit-nonexistent-overlay"
        self.history = history_fixture(self.cases)
        with self.assertRaises(ValueError):
            self.build()

    def test_future_case_reviews_rejected_using_explicit_as_of(self):
        self.cases[0]["review"]["reviewed_at"] = "2026-09-18T00:00:01Z"
        self.history = history_fixture(self.cases)
        with self.assertRaises(ValueError):
            self.build()

    def test_as_of_requires_timezone_and_is_recorded_without_wall_clock(self):
        with self.assertRaises(ValueError):
            study.build_plan(
                self.cases, self.config, self.history, as_of=AS_OF.replace(tzinfo=None)
            )
        self.assertEqual(self.build()["created_at"], AS_OF.isoformat())

    def test_build_plan_does_not_mutate_any_input(self):
        before = deepcopy((self.cases, self.config, self.history))
        self.build()
        self.assertEqual((self.cases, self.config, self.history), before)

    def test_rejected_plan_does_not_mutate_any_input(self):
        self.config["same_model_verified"] = False
        before = deepcopy((self.cases, self.config, self.history))
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual((self.cases, self.config, self.history), before)

    def test_all_three_arms_share_input_model_and_budgets_but_not_setup(self):
        plan = self.build(repetitions=2)
        paired = {}
        for trial in plan["trials"]:
            key = (trial["scenario_id"], trial["repetition"])
            paired.setdefault(key, {})[trial["condition"]] = trial
            self.assertEqual(trial["model"], self.config["model"])
            self.assertEqual(trial["input_sha256"], study.digest(trial["agent_input"]))
            self.assertTrue(trial["condition_setup"]["reset_required"])
            self.assertTrue(trial["condition_setup"]["no_online_learning"])
        self.assertEqual(len(paired), 60)
        self.assertEqual(plan["model_version"], self.config["model_version"])
        frozen_config = plan["frozen_material"]["config"]
        for key in ("max_output_tokens", "max_tool_calls", "temperature"):
            self.assertEqual(frozen_config[key], self.config[key])
        for arms in paired.values():
            self.assertEqual(set(arms), set(study.CONDITIONS))
            raw, blank, experienced = (arms[condition] for condition in study.CONDITIONS)
            self.assertEqual(raw["agent_input"], blank["agent_input"])
            self.assertEqual(blank["agent_input"], experienced["agent_input"])
            self.assertEqual(raw["input_sha256"], experienced["input_sha256"])
            self.assertEqual(raw["condition_setup"]["memory"], "none")
            self.assertEqual(blank["condition_setup"]["memory"], "empty_snapshot")
            self.assertEqual(experienced["condition_setup"]["memory"], plan["history_metadata"])
            self.assertEqual(raw["condition_setup"]["tool_authority"], "direct_sandbox")
            self.assertEqual(blank["condition_setup"]["tool_authority"], "governed_sandbox")
            self.assertEqual(experienced["condition_setup"]["tool_authority"], "governed_sandbox")
            self.assertEqual(
                len({study.digest(arm["condition_setup"]) for arm in arms.values()}), 3
            )

    def test_golden_reviewer_context_never_enters_agent_input(self):
        cases_by_id = {case["scenario_id"]: case for case in self.cases}
        for trial in self.build()["trials"]:
            agent_input = trial["agent_input"]
            self.assertEqual(
                set(agent_input),
                {"customer_message", "policy_facts", "case_context",
                 "shared_instruction", "manager_instruction"},
            )
            self.assertNotIn(GOLDEN_ONLY, json.dumps(agent_input))
            case = cases_by_id[trial["scenario_id"]]
            for key, value in case["reviewer_context"].items():
                self.assertEqual(trial["reviewer_context"][key], value)
                self.assertNotIn(value, json.dumps(agent_input))
            self.assertEqual(agent_input["customer_message"], case["agent_input"]["customer_message"])
            self.assertEqual(agent_input["policy_facts"], case["agent_input"]["policy_facts"])

    def test_frozen_hashes_reproduce_their_exact_material(self):
        plan = self.build()
        self.assertEqual(plan["protocol_sha256"], study.digest(plan["frozen_material"]))
        self.assertEqual(plan["scenario_set_sha256"], study.digest(self.cases))
        self.assertEqual(plan["history_metadata"]["snapshot_sha256"], study.digest(self.history))

    def test_same_seed_yields_identical_plan_ids_and_trial_order(self):
        first = self.build(seed=417, repetitions=2)
        second = self.build(seed=417, repetitions=2)
        self.assertEqual(first, second)
        identifiers = [trial["trial_id"] for trial in first["trials"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        other = self.build(seed=418, repetitions=2)
        self.assertNotEqual(first["protocol_sha256"], other["protocol_sha256"])
        self.assertTrue(set(identifiers).isdisjoint(trial["trial_id"] for trial in other["trials"]))
        order = lambda plan: [
            (trial["scenario_id"], trial["condition"], trial["repetition"])
            for trial in plan["trials"]
        ]
        self.assertNotEqual(order(first), order(other))

    def test_plan_randomization_does_not_touch_global_random_state(self):
        state = random.getstate()
        self.build()
        self.assertEqual(random.getstate(), state)

    def test_returned_frozen_plan_does_not_alias_mutable_caller_inputs(self):
        mutations = (
            lambda: self.cases[0]["agent_input"].update(customer_message="Changed after freeze."),
            lambda: self.cases[0]["reviewer_context"].update(expected_outcome="Changed after freeze."),
            lambda: self.cases[0]["source"].update(revision="Changed after freeze."),
            lambda: self.cases[0]["applicable_criteria"].pop(),
            lambda: self.config.update(shared_instruction="Changed after freeze."),
            lambda: self.history["records"][0].update(content="Changed after freeze."),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.setUp()
                plan = self.build()
                frozen_fingerprint = study.digest(plan)
                mutation()
                self.assertEqual(study.digest(plan), frozen_fingerprint)
                self.assertEqual(plan["protocol_sha256"], study.digest(plan["frozen_material"]))

    def test_plan_side_edits_cannot_modify_reviewed_source_inputs(self):
        plan = self.build()
        before = study.digest([self.cases, self.config, self.history])
        plan["frozen_material"]["config"]["shared_instruction"] = "Unit mutation of returned plan."
        plan["trials"][0]["applicable_criteria"].clear()
        plan["trials"][0]["reviewer_context"]["source"]["revision"] = "Unit mutation."
        self.assertEqual(study.digest([self.cases, self.config, self.history]), before)

    def test_every_meaningful_configuration_change_changes_hash_and_trial_ids(self):
        baseline = self.build()
        changes = {
            "model": "fictional-unit-model-2",
            "model_version": "fictional-unit-model-version-2",
            "raw_adapter_revision": "unit-raw-2",
            "elle_revision": "unit-elle-2",
            "governance_revision": "unit-governance-2",
            "memory_retriever_revision": "unit-retriever-2",
            "customer_simulator_revision": "unit-customer-2",
            "shared_instruction": "Changed fictional common instruction.",
            "study_owner": "fictional-owner-2",
            "primary_contrast": "raw:elle_blank",
            "max_output_tokens": 513,
            "max_tool_calls": 5,
            "temperature": 0.5,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                self.config = config_fixture()
                self.config[key] = value
                changed = self.build()
                self.assert_plan_rekeyed(baseline, changed)

    def test_meaningful_case_changes_require_rescreening_and_change_hash(self):
        baseline = self.build()
        changes = (
            (("agent_input",), "customer_message", "Changed fictional request."),
            (("agent_input",), "policy_facts", "Changed fictional policy."),
            (("agent_input",), "case_context", "Changed fictional account facts."),
            (("reviewer_context",), "expected_outcome", "Changed fictional golden answer."),
            (("reviewer_context",), "customer_preference", "Changed fictional preference."),
            (("reviewer_context",), "authority", "Changed fictional authorization."),
            (("reviewer_context",), "eligibility", "Changed fictional eligibility."),
            (("review",), "reviewer_id", "fictional-unit-reviewer-2"),
            (("review",), "reviewed_at", "2026-09-16T01:00:00Z"),
            (("source",), "revision", "unit-source-revision-2"),
            ((), "scenario_id", "unit-renamed-scenario"),
            ((), "pressure_id", "urgency"),
        )
        for path, key, value in changes:
            with self.subTest(path=path, key=key):
                cases = deepcopy(self.cases)
                target = cases[-1][path[0]] if path else cases[-1]
                target[key] = value
                with self.assertRaises(ValueError):
                    study.build_plan(cases, self.config, self.history, as_of=AS_OF)
                changed = study.build_plan(
                    cases, self.config, history_fixture(cases), as_of=AS_OF
                )
                self.assertNotEqual(baseline["scenario_set_sha256"], changed["scenario_set_sha256"])
                self.assert_plan_rekeyed(baseline, changed)

    def test_source_content_and_predeclared_applicability_changes_rekey_plan(self):
        baseline = self.build()
        for change in ("upstream_record", "applicable_criteria"):
            with self.subTest(change=change):
                cases = deepcopy(self.cases)
                if change == "upstream_record":
                    cases[-1]["upstream_record"]["text"] += " Revised fictional source."
                    cases[-1]["source"]["record_sha256"] = study.digest(cases[-1]["upstream_record"])
                else:
                    cases[-1]["applicable_criteria"].remove("eligible_option_disclosed")
                changed = study.build_plan(
                    cases, self.config, history_fixture(cases), as_of=AS_OF
                )
                self.assert_plan_rekeyed(baseline, changed)

    def test_meaningful_history_changes_rekey_snapshot_plan_and_trials(self):
        baseline = self.build()
        changes = (
            ((), "snapshot_id", "fictional-unit-snapshot-2"),
            ((), "collection_provenance", f"{UNIT_ONLY}; revised provenance."),
            ((), "permission_reference", f"{UNIT_ONLY}; revised permission."),
            ((), "frozen_at", "2026-09-15T01:00:00Z"),
            (("records", 0), "content", f"{UNIT_ONLY}; changed fictional retained content."),
            (("records", 0), "observed_at", "2026-06-16T00:00:00Z"),
            (("records", 0), "record_id", "unit-history-first-revised"),
            (("records", 0), "source_reference", "unit-history-revised-reference"),
            (("contamination_review",), "reviewer_id", "fictional-unit-screening-reviewer-2"),
            (("contamination_review",), "reviewed_at", "2026-09-17T01:00:00Z"),
        )
        for path, key, value in changes:
            with self.subTest(path=path, key=key):
                history = deepcopy(self.history)
                target = history
                for part in path:
                    target = target[part]
                target[key] = value
                changed = study.build_plan(self.cases, self.config, history, as_of=AS_OF)
                self.assertNotEqual(
                    baseline["history_metadata"]["snapshot_sha256"],
                    changed["history_metadata"]["snapshot_sha256"],
                )
                self.assert_plan_rekeyed(baseline, changed)

    def test_protocol_rubric_and_pressure_text_changes_rekey_plan(self):
        baseline = self.build()
        original_load = study.load_json
        for filename in ("protocol.json", "rubric.json", "pressure_overlays.json"):
            with self.subTest(filename=filename):
                def changed_load(path):
                    value = deepcopy(original_load(path))
                    if path.name == filename:
                        if filename == "protocol.json":
                            value["question"] += " Unit-test change of research question."
                        elif filename == "rubric.json":
                            value["criteria"][0]["pass"] += " Unit-test clarified requirement."
                        else:
                            value["overlays"][0]["text"] += " Unit-test manager wording change."
                    return value

                with patch.object(study, "load_json", side_effect=changed_load):
                    changed = self.build()
                self.assert_plan_rekeyed(baseline, changed)

    def test_repetition_count_changes_plan_hash_and_ids(self):
        self.assert_plan_rekeyed(self.build(repetitions=1), self.build(repetitions=2))

    def test_case_order_and_history_record_membership_change_plan_hash(self):
        baseline = self.build()
        reversed_cases = list(reversed(self.cases))
        changed = study.build_plan(
            reversed_cases, self.config, history_fixture(reversed_cases), as_of=AS_OF
        )
        self.assert_plan_rekeyed(baseline, changed)
        history = deepcopy(self.history)
        history["records"].append({
            "record_id": "unit-history-middle",
            "content": f"{UNIT_ONLY}; another fictional interaction.",
            "source_reference": "unit-history-source:middle",
            "observed_at": "2026-08-01T00:00:00Z",
            "source_case_keys": [],
        })
        changed = study.build_plan(self.cases, self.config, history, as_of=AS_OF)
        self.assert_plan_rekeyed(baseline, changed)

    def test_manifest_verifier_accepts_original_and_json_round_trip(self):
        plan = self.build()
        study.verify_plan(plan)
        study.verify_plan(json.loads(json.dumps(plan)))

    def test_manifest_verifier_rejects_changed_material_or_top_level_identity(self):
        baseline = self.build()
        mutations = (
            lambda plan: plan["frozen_material"]["config"].update(temperature=0.5),
            lambda plan: plan.update(protocol_sha256="0" * 64),
            lambda plan: plan.pop("frozen_material"),
            lambda plan: plan.update(scenario_set_sha256="0" * 64),
            lambda plan: plan.update(schema_version=2),
            lambda plan: plan.update(schema_version=True),
            lambda plan: plan.update(protocol_status="draft"),
            lambda plan: plan.update(model="fictional-different-model"),
            lambda plan: plan.update(model_version="fictional-different-version"),
            lambda plan: plan.update(mode="confirmatory"),
            lambda plan: plan.update(independent_source_cases=300),
            lambda plan: plan.update(conditions=["raw", "elle_blank"]),
            lambda plan: plan.update(history_metadata={}),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                plan = deepcopy(baseline)
                mutation(plan)
                with self.assertRaises(ValueError):
                    study.verify_plan(plan)

    def test_manifest_verifier_binds_every_frozen_material_compartment(self):
        baseline = self.build()
        mutations = (
            lambda material: material["protocol"].update(question="Changed unit question."),
            lambda material: material["rubric"]["criteria"][0].update({"pass": "Changed unit rubric."}),
            lambda material: material["pressure_overlays"]["overlays"][0].update(text="Changed unit pressure."),
            lambda material: material["config"].update(temperature=0.5),
            lambda material: material["cases"][0]["review"].update(reviewer_id="Changed unit reviewer."),
            lambda material: material["history_metadata"].update(snapshot_sha256="0" * 64),
            lambda material: material.update(mode="confirmatory"),
            lambda material: material.update(repetitions=2),
            lambda material: material.update(seed=91),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(compartment=index):
                plan = deepcopy(baseline)
                mutation(plan["frozen_material"])
                with self.assertRaises(ValueError):
                    study.verify_plan(plan)

    def test_manifest_verifier_rejects_trial_tampering_and_golden_input_leak(self):
        baseline = self.build()
        mutations = (
            lambda plan: plan["trials"].reverse(),
            lambda plan: plan["trials"].pop(),
            lambda plan: plan["trials"].append(deepcopy(plan["trials"][0])),
            lambda plan: plan["trials"][0].update(trial_id="unit-forged-trial-id"),
            lambda plan: plan["trials"][0].update(input_sha256="0" * 64),
            lambda plan: plan["trials"][0]["agent_input"].update(expected_outcome=GOLDEN_ONLY),
            lambda plan: plan["trials"][0]["agent_input"].update(policy_facts="Changed unit policy."),
            lambda plan: plan["trials"][0]["condition_setup"].update(reset_required=False),
            lambda plan: plan["trials"][0]["condition_setup"].update(no_online_learning=False),
            lambda plan: plan["trials"][0]["condition_setup"].update(memory="forged-unit-memory"),
            lambda plan: plan["trials"][0]["reviewer_context"].update(expected_outcome="Changed golden."),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                plan = deepcopy(baseline)
                mutation(plan)
                with self.assertRaises(ValueError):
                    study.verify_plan(plan)

    def test_rehashing_changed_material_does_not_validate_old_trial_ids(self):
        plan = self.build()
        plan["frozen_material"]["config"]["temperature"] = 0.5
        plan["protocol_sha256"] = study.digest(plan["frozen_material"])
        with self.assertRaises(ValueError):
            study.verify_plan(plan)

    def test_manifest_verifier_binds_trial_identity_and_compartments_for_every_arm(self):
        baseline = self.build()
        for condition in study.CONDITIONS:
            for key, invalid in (
                ("scenario_id", "unit-forged-scenario"),
                ("cluster_id", "unit-forged-cluster"),
                ("condition", "unit-forged-condition"),
                ("repetition", 12),
                ("repetition", False),
                ("model", "unit-forged-model"),
                ("stratum", "unit-forged-stratum"),
                ("applicable_criteria", []),
                ("agent_input", {}),
                ("reviewer_context", {}),
                ("condition_setup", {}),
            ):
                with self.subTest(condition=condition, key=key):
                    plan = deepcopy(baseline)
                    trial = next(
                        trial for trial in plan["trials"] if trial["condition"] == condition
                    )
                    trial[key] = invalid
                    with self.assertRaises(ValueError):
                        study.verify_plan(plan)

    def test_manifest_boolean_execution_flags_cannot_be_replaced_with_integers(self):
        baseline = self.build()
        for condition in study.CONDITIONS:
            for key in ("reset_required", "no_online_learning"):
                with self.subTest(condition=condition, key=key):
                    plan = deepcopy(baseline)
                    trial = next(
                        trial for trial in plan["trials"] if trial["condition"] == condition
                    )
                    trial["condition_setup"][key] = 1
                    with self.assertRaises(ValueError):
                        study.verify_plan(plan)

    def assert_plan_rekeyed(self, before, after):
        self.assertNotEqual(before["protocol_sha256"], after["protocol_sha256"])
        self.assertTrue(
            {trial["trial_id"] for trial in before["trials"]}.isdisjoint(
                trial["trial_id"] for trial in after["trials"]
            )
        )


class ResultProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_fixture()
        self.trials_by_condition = {
            condition: next(
                trial for trial in self.plan["trials"] if trial["condition"] == condition
            )
            for condition in study.CONDITIONS
        }

    def result(self, condition):
        return result_fixture(self.plan, self.trials_by_condition[condition])

    def test_matching_results_for_all_conditions_validate_without_mutation(self):
        results = [result_fixture(self.plan, trial) for trial in self.plan["trials"]]
        before = study.digest([self.plan, results])
        study.verify_result_provenance(self.plan, results)
        self.assertEqual(study.digest([self.plan, results]), before)
        for result in results:
            self.assertEqual(result["fixture_warning"], UNIT_ONLY)

    def test_empty_and_partial_result_sets_are_valid_missingness_not_fabrication(self):
        study.verify_result_provenance(self.plan, [])
        for condition in study.CONDITIONS:
            with self.subTest(condition=condition):
                study.verify_result_provenance(self.plan, [self.result(condition)])

    def test_every_fingerprint_is_mandatory_even_when_expected_value_is_null(self):
        for condition in study.CONDITIONS:
            for key in (
                "protocol_sha256", "scenario_set_sha256", "model_version", "input_sha256",
                "adapter_revision", "governance_revision", "history_snapshot_sha256",
            ):
                with self.subTest(condition=condition, missing=key):
                    result = self.result(condition)
                    del result[key]
                    with self.assertRaises(ValueError):
                        study.verify_result_provenance(self.plan, [result])

    def test_protocol_scenario_input_and_model_version_must_match_frozen_plan(self):
        for condition in study.CONDITIONS:
            for key in ("protocol_sha256", "scenario_set_sha256", "input_sha256", "model_version"):
                for invalid in (None, "", "unit-different-fingerprint", True):
                    with self.subTest(condition=condition, key=key, invalid=invalid):
                        result = self.result(condition)
                        result[key] = invalid
                        with self.assertRaises(ValueError):
                            study.verify_result_provenance(self.plan, [result])

    def test_adapter_revision_is_specific_to_raw_or_real_elle_execution(self):
        config = self.plan["frozen_material"]["config"]
        for condition in study.CONDITIONS:
            expected_key = "raw_adapter_revision" if condition == "raw" else "elle_revision"
            wrong_key = "elle_revision" if condition == "raw" else "raw_adapter_revision"
            result = self.result(condition)
            self.assertEqual(result["adapter_revision"], config[expected_key])
            for invalid in (None, "", "unit-unrecorded-adapter", config[wrong_key]):
                with self.subTest(condition=condition, invalid=invalid):
                    result["adapter_revision"] = invalid
                    with self.assertRaises(ValueError):
                        study.verify_result_provenance(self.plan, [result])

    def test_governance_revision_is_null_for_raw_and_pinned_for_both_elle_arms(self):
        governance = self.plan["frozen_material"]["config"]["governance_revision"]
        for condition in study.CONDITIONS:
            expected = None if condition == "raw" else governance
            result = self.result(condition)
            self.assertEqual(result["governance_revision"], expected)
            wrong = (governance, "", False) if condition == "raw" else (None, "", "unit-other-governance")
            for invalid in wrong:
                with self.subTest(condition=condition, invalid=invalid):
                    result["governance_revision"] = invalid
                    with self.assertRaises(ValueError):
                        study.verify_result_provenance(self.plan, [result])

    def test_history_hash_required_only_for_experienced_and_exactly_matches_snapshot(self):
        snapshot_hash = self.plan["history_metadata"]["snapshot_sha256"]
        for condition in study.CONDITIONS:
            result = self.result(condition)
            expected = snapshot_hash if condition == "elle_experienced" else None
            self.assertEqual(result["history_snapshot_sha256"], expected)
            wrong = (
                (None, "", "0" * 64)
                if condition == "elle_experienced" else (snapshot_hash, "", "0" * 64)
            )
            for invalid in wrong:
                with self.subTest(condition=condition, invalid=invalid):
                    result["history_snapshot_sha256"] = invalid
                    with self.assertRaises(ValueError):
                        study.verify_result_provenance(self.plan, [result])

    def test_input_fingerprint_cannot_be_borrowed_from_another_scenario(self):
        result = self.result("raw")
        different = next(
            trial for trial in self.plan["trials"]
            if trial["input_sha256"] != result["input_sha256"]
        )
        result["input_sha256"] = different["input_sha256"]
        with self.assertRaises(ValueError):
            study.verify_result_provenance(self.plan, [result])

    def test_unknown_duplicate_and_malformed_trial_ids_rejected(self):
        for invalid in ("unit-unplanned-trial", "", None, True, []):
            with self.subTest(invalid=invalid):
                result = self.result("raw")
                result["trial_id"] = invalid
                with self.assertRaises(ValueError):
                    study.verify_result_provenance(self.plan, [result])
        result = self.result("raw")
        del result["trial_id"]
        with self.assertRaises(ValueError):
            study.verify_result_provenance(self.plan, [result])
        result = self.result("raw")
        with self.assertRaises(ValueError):
            study.verify_result_provenance(self.plan, [result, deepcopy(result)])

    def test_error_results_still_require_full_provenance(self):
        for condition in study.CONDITIONS:
            with self.subTest(condition=condition):
                result = self.result(condition)
                result.update(status="error", response="", error="Unit-only simulated failure.")
                study.verify_result_provenance(self.plan, [result])
                del result["protocol_sha256"]
                with self.assertRaises(ValueError):
                    study.verify_result_provenance(self.plan, [result])

    def test_result_verification_always_rechecks_plan_including_with_no_results(self):
        for has_result in (False, True):
            with self.subTest(has_result=has_result):
                result = self.result("raw")
                plan = deepcopy(self.plan)
                plan["trials"][0]["agent_input"]["expected_outcome"] = GOLDEN_ONLY
                with self.assertRaises(ValueError):
                    study.verify_result_provenance(plan, [result] if has_result else [])

    def test_results_from_another_frozen_plan_are_rejected(self):
        other = plan_fixture(seed=19)
        result = result_fixture(other, other["trials"][0])
        with self.assertRaises(ValueError):
            study.verify_result_provenance(self.plan, [result])
        result["trial_id"] = self.trials_by_condition["raw"]["trial_id"]
        with self.assertRaises(ValueError):
            study.verify_result_provenance(self.plan, [result])


class CanonicalDigestTests(unittest.TestCase):
    def test_digest_is_stable_for_object_key_order_and_sensitive_to_list_order(self):
        self.assertEqual(study.digest({"a": 1, "b": 2}), study.digest({"b": 2, "a": 1}))
        self.assertNotEqual(study.digest(["a", "b"]), study.digest(["b", "a"]))

    def test_canonical_bytes_and_digest_use_stable_json_and_sha256(self):
        value = {"unicode": "\u00e9", "nested": {"z": 0, "a": False}}
        expected = b'{"nested":{"a":false,"z":0},"unicode":"\\u00e9"}'
        self.assertEqual(study.canonical_bytes(value), expected)
        self.assertEqual(study.digest(value), hashlib.sha256(expected).hexdigest())

    def test_nonfinite_values_cannot_be_hashed_as_json_material(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    study.digest({"value": invalid})


if __name__ == "__main__":
    unittest.main()

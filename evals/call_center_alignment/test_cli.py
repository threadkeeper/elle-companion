"""CLI integration with nonempirical generated fixtures in temporary directories."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import __main__ as cli
from . import study
from .test_study import UNIT_ONLY, plan_fixture, result_fixture


class CliTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(list(arguments)), 0)
        return json.loads(output.getvalue())

    def test_outputs_restricted_to_ignored_directories(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "HERE", Path(directory)):
            with self.assertRaises(ValueError):
                cli.private_output(str(Path(directory) / "source.py"))
            self.assertEqual(
                cli.private_output(str(Path(directory) / "runs" / "report.json")),
                Path(directory) / "runs" / "report.json",
            )

    def test_empty_results_report_missing_without_a_winner(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "HERE", Path(directory)):
            root = Path(directory) / "runs" / "unit-only"
            root.mkdir(parents=True)
            plan = plan_fixture()
            study.write_new_json(root / "plan.json", plan)
            study.write_new_jsonl(root / "results.jsonl", [])
            study.write_new_jsonl(root / "reviews.jsonl", [])
            result = self.invoke(
                "analyze", "--plan", str(root / "plan.json"),
                "--results", str(root / "results.jsonl"),
                "--reviews", str(root / "reviews.jsonl"),
                "--out", str(root / "analysis.json"), "--bootstrap-samples", "10",
            )
            self.assertIsNone(result["descriptive_primary_rates"])
            self.assertEqual(result["coverage"]["missing"], 90)
            self.assertEqual(result["coverage"]["completed"], 0)
            self.assertTrue((root / "analysis.json").exists())

    def test_review_unblind_analyze_roundtrip_does_not_fabricate_other_results(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "HERE", Path(directory)):
            root = Path(directory) / "runs" / "unit-only"
            root.mkdir(parents=True)
            plan = plan_fixture()
            trial = next(item for item in plan["trials"] if item["condition"] == "raw")
            result = result_fixture(plan, trial)
            study.write_new_json(root / "plan.json", plan)
            study.write_new_jsonl(root / "results.jsonl", [result])
            review_dir = root / "review"
            self.invoke(
                "review-pack", "--plan", str(root / "plan.json"),
                "--results", str(root / "results.jsonl"), "--out", str(review_dir),
            )
            pack = study.load_json(review_dir / "reviewer_pack.json")
            self.assertEqual(len(pack["items"]), 1)
            self.assertNotIn("condition", pack["items"][0])
            self.assertNotIn("trial_id", pack["items"][0])
            self.assertNotIn("agent_input", pack["items"][0])
            review = study.load_jsonl(review_dir / "reviews_to_complete.jsonl")[0]
            review["reviewer_id"] = UNIT_ONLY
            for criterion in study.CRITERIA:
                applicable = criterion in trial["applicable_criteria"]
                review["scores"][criterion] = False if applicable else None
                review["evidence"][criterion] = "/response" if applicable else "N/A: excluded in unit fixture"
            study.write_new_jsonl(review_dir / "adjudicated.jsonl", [review])
            self.invoke(
                "unblind", "--reviews", str(review_dir / "adjudicated.jsonl"),
                "--key", str(review_dir / "PRIVATE_review_key.json"),
                "--out", str(root / "reviews.jsonl"),
            )
            outcome = self.invoke(
                "analyze", "--plan", str(root / "plan.json"),
                "--results", str(root / "results.jsonl"),
                "--reviews", str(root / "reviews.jsonl"),
                "--out", str(root / "analysis.json"), "--bootstrap-samples", "10",
            )
            self.assertEqual(outcome["coverage"]["reviewed"], 1)
            self.assertEqual(outcome["coverage"]["missing"], 89)
            self.assertEqual(outcome["descriptive_primary_rates"]["raw"], 0)
            self.assertIsNone(outcome["descriptive_primary_rates"]["elle_blank"])

    def test_bad_fingerprints_fail_before_creating_analysis(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "HERE", Path(directory)):
            root = Path(directory) / "runs" / "unit-only"
            root.mkdir(parents=True)
            plan = plan_fixture()
            result = result_fixture(plan, plan["trials"][0])
            result["protocol_sha256"] = "changed"
            study.write_new_json(root / "plan.json", plan)
            study.write_new_jsonl(root / "results.jsonl", [result])
            study.write_new_jsonl(root / "reviews.jsonl", [])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                cli.main([
                    "analyze", "--plan", str(root / "plan.json"),
                    "--results", str(root / "results.jsonl"),
                    "--reviews", str(root / "reviews.jsonl"),
                    "--out", str(root / "analysis.json"),
                ])
            self.assertEqual(exit_info.exception.code, 2)
            self.assertFalse((root / "analysis.json").exists())


if __name__ == "__main__":
    unittest.main()

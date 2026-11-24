"""CLI for public source preparation, study freezing and human-scored analysis."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from . import analysis, data
from .study import (
    HERE, build_plan, digest, load_json, load_jsonl, validate_cases,
    verify_plan, verify_result_provenance, write_new_json, write_new_jsonl,
)


def private_output(path: str) -> Path:
    resolved = Path(path).resolve()
    if not any(resolved.is_relative_to((HERE / name).resolve()) for name in ("data", "runs", "snapshots")):
        raise ValueError("Generated artifacts must stay under this folder's ignored data, runs or snapshots directories")
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="List researched sources and import restrictions")

    prepare = commands.add_parser("prepare", help="Download pinned existing tasks; create unapproved source cards, never synthetic replacements")
    prepare.add_argument("--source", nargs="+", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--limit", type=int, default=60)
    prepare.add_argument("--seed", type=int, default=20260918)
    prepare.add_argument("--match", help="Optional explicit substring filter over original source records, recorded in the manifest")

    validate = commands.add_parser("validate-cases", help="Check human approval, source integrity and fixed scoring applicability")
    validate.add_argument("--cases", required=True, type=Path)

    freeze = commands.add_parser("freeze", help="Freeze reviewed cases, real history and one same-model three-arm plan; no model calls")
    freeze.add_argument("--cases", required=True, type=Path)
    freeze.add_argument("--config", required=True, type=Path)
    freeze.add_argument("--history", required=True, type=Path)
    freeze.add_argument("--mode", choices=("pilot", "confirmatory"), default="pilot")
    freeze.add_argument("--repetitions", type=int, default=1)
    freeze.add_argument("--seed", type=int, default=20260918)
    freeze.add_argument("--out", required=True)

    review = commands.add_parser("review-pack", help="Create a blinded review pack and a separate private ID key; no automatic grading")
    review.add_argument("--plan", required=True, type=Path)
    review.add_argument("--results", required=True, type=Path)
    review.add_argument("--out", required=True)

    unblind = commands.add_parser("unblind", help="Map one final human-adjudicated review per item back to trial IDs")
    unblind.add_argument("--reviews", required=True, type=Path)
    unblind.add_argument("--key", required=True, type=Path)
    unblind.add_argument("--out", required=True)

    summarize = commands.add_parser("analyze", help="Summarize all three conditions, coverage and cluster-aware paired differences")
    summarize.add_argument("--plan", required=True, type=Path)
    summarize.add_argument("--results", required=True, type=Path)
    summarize.add_argument("--reviews", required=True, type=Path)
    summarize.add_argument("--bootstrap-samples", type=int, default=2000)
    summarize.add_argument("--seed", type=int, default=20260918)
    summarize.add_argument("--out", required=True)
    return result


def execute(args: argparse.Namespace) -> dict:
    if args.command == "sources":
        return data.registry()
    if args.command == "prepare":
        return data.prepare(args.source, private_output(args.out), limit=args.limit, seed=args.seed, match=args.match)
    if args.command == "validate-cases":
        cases = load_jsonl(args.cases)
        validate_cases(cases)
        return {"reviewed_scenarios": len(cases), "scenario_set_sha256": digest(cases)}
    if args.command == "freeze":
        output = private_output(args.out)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen study: {output}")
        plan = build_plan(
            load_jsonl(args.cases), load_json(args.config), load_json(args.history),
            mode=args.mode, repetitions=args.repetitions, seed=args.seed,
        )
        verify_plan(plan)
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(output / "plan.json", plan)
        write_new_jsonl(output / "trials.jsonl", plan["trials"])
        return {
            "status": "locally_frozen_not_executed",
            "plan": str(output / "plan.json"),
            "planned_trials": len(plan["trials"]),
            "independent_source_cases": plan["independent_source_cases"],
            "protocol_sha256": plan["protocol_sha256"],
            "note": "Connect the actual raw/Elle sandbox runners. No LLM, live customer tool, governing-layer implementation or three-month history was created by this command.",
        }
    if args.command == "review-pack":
        output = private_output(args.out)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite review pack: {output}")
        plan, results = load_json(args.plan), load_jsonl(args.results)
        verify_result_provenance(plan, results)
        pack, key = analysis.make_review_pack(plan, results, secrets.token_hex(32))
        output.mkdir(parents=True, exist_ok=False)
        write_new_json(output / "reviewer_pack.json", pack)
        write_new_json(output / "PRIVATE_review_key.json", key)
        templates = [
            {
                "review_id": item["review_id"],
                "reviewer_id": "",
                "scores": {criterion: None for criterion in pack["rubric_dimensions"]},
                "evidence": {criterion: "" for criterion in pack["rubric_dimensions"]},
                "notes": "Unscored template; complete human review, do not treat nulls as passes.",
            }
            for item in pack["items"]
        ]
        write_new_jsonl(output / "reviews_to_complete.jsonl", templates)
        return {"review_items": len(pack["items"]), "output": str(output), "warning": "Do not give PRIVATE_review_key.json or the condition-bearing plan/results to blinded reviewers."}
    if args.command == "unblind":
        output = private_output(args.out)
        reviews = analysis.unblind_reviews(load_jsonl(args.reviews), load_json(args.key))
        write_new_jsonl(output, reviews)
        return {"mapped_final_reviews": len(reviews), "output": str(output)}
    if args.command == "analyze":
        output = private_output(args.out)
        plan, results = load_json(args.plan), load_jsonl(args.results)
        verify_result_provenance(plan, results)
        report = analysis.analyze(
            plan, results, load_jsonl(args.reviews),
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        )
        write_new_json(output, report)
        return {
            "report": str(output),
            "coverage": report["coverage"],
            "descriptive_primary_rates": report["grade"],
            "interpretation": "Pilot estimates are exploratory. Missing/error runs are not automatically harm; no morality, empathy or winner is inferred.",
        }
    raise ValueError(f"Unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        summary = execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        arguments.exit(2, f"Error: {error}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

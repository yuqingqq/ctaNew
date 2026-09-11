"""Value one forward day through the frozen, guarded pipeline."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import de_forward_evaluator as E          # noqa: E402
import de_settlement_control_aggregate as AGG  # noqa: E402
import de_settlement_control_run as SC    # noqa: E402

PROTOCOL = "P003_DE_FORWARD_VALUE_DAY_V2"
PIPELINE_COMMIT = SC.PIPELINE_COMMIT
COMPUTING_MODULES = (
    "de_forward_value_day.py",
    "be_daybook_build.py",
    "be_cancel_axis_null.py",
    "be_score_neutrality.py",
    "da_forward_result_guard.py",
    "de_settlement_control_run.py",
    "de_settlement_control_aggregate.py",
    "de_forward_evaluator.py",
    "de_asymmetry_null_run.py",
    "de_matched_cancel_control.py",
    "de_multiday_gate1_runner.py",
    "de_phase4_diag_runner.py",
    "harmful_stateful_policy.py",
)

PIPELINE_MOVED = "FORWARD_COMPUTING_MODULES_NOT_AT_FROZEN_PIPELINE"
BAD_EXISTING = "FORWARD_EXISTING_CELL_RESULT_DOES_NOT_MATCH_THIS_RUN"
OUTPUT_EXISTS = "FORWARD_DAY_RESULT_ALREADY_EXISTS"


class ForwardValueRefused(RuntimeError):
    """A named refusal."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def computing_module_provenance() -> dict:
    """Compare every number-producing module with the frozen commit."""
    modules, all_match = {}, True
    for name in COMPUTING_MODULES:
        expected_path = (HERE / name).resolve()
        if name == "de_forward_value_day.py":
            resolved_path = Path(__file__).resolve()
        elif name == "be_daybook_build.py":
            resolved_path = expected_path
        else:
            loaded = importlib.import_module(name.removesuffix(".py"))
            resolved_path = Path(loaded.__file__).resolve()
        path_matches = resolved_path == expected_path
        here = _sha(resolved_path)
        process = subprocess.run(
            ["git", "-C", str(HERE), "show",
             f"{PIPELINE_COMMIT}:live/pm_research/{name}"],
            capture_output=True)
        there = (hashlib.sha256(process.stdout).hexdigest()
                 if process.returncode == 0 else None)
        match = path_matches and here == there
        all_match &= match
        modules[name] = {"resolved_file": str(resolved_path),
                         "expected_file": str(expected_path),
                         "resolved_file_matches": path_matches,
                         "digest_here": here,
                         "digest_at_pipeline_commit": there,
                         "identical": match}
    head_process = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    head = (head_process.stdout.strip()
            if head_process.returncode == 0 else None)
    head_matches = head == PIPELINE_COMMIT
    return {"pipeline_commit": PIPELINE_COMMIT,
            "worktree": str(HERE.parents[1]),
            "worktree_head": head,
            "worktree_head_matches_pipeline_commit": head_matches,
            "modules": modules,
            "every_computing_module_matches_the_pipeline_commit": all_match}


def require_frozen_pipeline() -> dict:
    provenance = computing_module_provenance()
    if (not provenance["worktree_head_matches_pipeline_commit"]
            or not provenance[
                "every_computing_module_matches_the_pipeline_commit"]):
        moved = [name for name, row in provenance["modules"].items()
                 if not row["identical"]]
        raise ForwardValueRefused(
            f"REFUSED {PIPELINE_MOVED}: worktree HEAD is "
            f"{provenance['worktree_head']}, frozen pipeline is "
            f"{PIPELINE_COMMIT}, and differing modules are {moved}. The "
            f"freeze must be superseded before a new pipeline can value a "
            f"forward day.")
    return provenance


def pipeline_provenance_limit() -> dict:
    return {
        "LIMIT": ("THE ENTIRE FORWARD TEST -- BUILD AND VALUATION -- RUNS "
                  "ON CODE THE DEVELOPMENT SCREEN NEVER RAN ON."),
        "what_moved_from_the_development_screen": [
            "de_head_scoring.py", "de_phase4_diag_runner.py"],
        "what_did_not_move": (
            "the other eleven pinned scoring modules are byte-identical"),
        "interpretation": (
            "score neutrality on consumed data plus the per-forward-book "
            "margin guard are required; neither is a general equivalence "
            "claim"),
    }


def power_block(emit: dict) -> dict:
    declared = emit["floor_at_the_G_DECLARED"]
    achieved = emit["floor_at_the_G_ACHIEVED_SO_FAR"]
    return {
        "attainable_minimum_p_at_the_G_declared":
            declared["attainable_minimum_p"]["day_sign_component"],
        "attainable_minimum_p_at_the_G_achieved_so_far":
            achieved["attainable_minimum_p"]["day_sign_component"],
        "tolerance_negative_days_at_the_G_declared":
            declared["tolerance_negative_days"],
        "a_pass_was_possible_at_the_G_declared":
            declared["a_pass_was_possible_at_this_G"],
        "reading": "NOT_ESTABLISHED_AT_THIS_POWER, never NO_EFFECT.",
    }


def margin_block(emit: dict) -> dict:
    return {
        "per_arm": {
            arm: {"negative_or_zero_days": row["negative_or_zero_days"],
                  "best_attainable_p": row[
                      "best_attainable_p_given_what_is_already_seen"]}
            for arm, row in emit["futility"].items()},
        "why": ("futility is decided at the UTC-day sign level; the "
                "forward-book score margin is enforced separately in each "
                "cell result"),
    }


def verify_driver_book_receipt(path: Path, *, day: str,
                               book_path: Path) -> dict:
    """Verify a reused cell's receipt without loading the large book again."""
    try:
        return SC.verify_book_receipt(
            path, None, day, book_path=book_path)
    except SC.SettlementControlRefused as exc:
        raise ForwardValueRefused(str(exc)) from None


def load_existing_cell(path: Path, *, day: str, arm: str,
                       book_sha: str) -> dict:
    """Reuse only a result reconciled against its exact V2 checkpoint."""
    try:
        cell = AGG.load_cell(
            path.parent, day, arm, strict_forward=True)
    except (OSError, json.JSONDecodeError, AGG.AggregateRefused) as exc:
        raise ForwardValueRefused(
            f"REFUSED {BAD_EXISTING}: {path}: {exc}") from None
    result = cell["result"]
    if result.get("book_sha256") != book_sha:
        raise ForwardValueRefused(
            f"REFUSED {BAD_EXISTING}: {path} is not a complete result for "
            f"this day, arm, book and declared draw count.")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--book-receipt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-draws", type=int, default=SC.DECLARED_N)
    parser.add_argument("--score-certification", action="append",
                        required=True)
    parser.add_argument("--days-scored", required=True)
    parser.add_argument("--n-declared", type=int, required=True)
    parser.add_argument("--derived", required=True)
    args = parser.parse_args(argv)

    try:
        provenance = require_frozen_pipeline()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / (
            f"p003_de_forward_value_{args.day.replace('-', '')}.json")
        if destination.exists():
            raise ForwardValueRefused(
                f"REFUSED {OUTPUT_EXISTS}: {destination}.")
        receipt_evidence = verify_driver_book_receipt(
            Path(args.book_receipt), day=args.day,
            book_path=Path(args.book))
        book_sha = receipt_evidence["book_sha256"]
        started = time.time()
        cells = {}
        for arm in E.ARMS:
            cell_path = SC.result_path(out_dir, args.day, arm)
            if cell_path.exists():
                result = load_existing_cell(
                    cell_path, day=args.day, arm=arm, book_sha=book_sha)
                stage = "reused_complete_cell"
            else:
                print(json.dumps({"stage": "valuing", "day": args.day,
                                  "arm": arm}), flush=True)
                result = SC.run_one_day_arm(
                    args.day, args.book, arm, n_draws=args.n_draws,
                    out_dir=out_dir, book_receipt=args.book_receipt,
                    score_certifications=args.score_certification)
                SC.write_result_exclusive(cell_path, result)
                stage = "valued"
            cells[arm] = {"D": result["observed_D_cents"],
                          "p": result["p_two_sided"],
                          "n": result["n_draws"],
                          "robustness_D": result["robustness_leg"]
                              ["observed_D_cents"],
                          "result": str(cell_path)}
            print(json.dumps({"stage": stage, "arm": arm,
                              **cells[arm]}), flush=True)

        days = [day for day in args.days_scored.split(",") if day]
        emit = E.progress_emit(
            out_dir, days, n_declared=args.n_declared,
            derived=Path(args.derived))
        record = {
            "protocol": PROTOCOL, "day": args.day, "book": args.book,
            "book_sha256": book_sha, "book_receipt": receipt_evidence,
            "cells": cells, "emit": emit,
            "POWER": power_block(emit),
            "MARGIN_AND_FUTILITY": margin_block(emit),
            "pipeline_provenance_limit": pipeline_provenance_limit(),
            "computing_module_provenance": provenance,
            "elapsed_s": round(time.time() - started, 1),
        }
        SC.write_result_exclusive(destination, record)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"refused": str(exc)}, indent=1))
        return 3

    for line in emit["per_day_lines"]:
        print("  " + line, flush=True)
    print(json.dumps({"stage": "emitted", "path": str(destination),
                      "ANY_ARM_ALREADY_DEAD": emit["ANY_ARM_ALREADY_DEAD"],
                      "STOP_ADVICE": emit["STOP_ADVICE"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

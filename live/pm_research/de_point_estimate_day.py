#!/usr/bin/env python3
"""Emit one P003 settlement point estimate without drawing the null.

WHY THIS FILE IS TRACKED. Two rounds of point estimates were driven by a
`pe_run.py` in a session SCRATCHPAD: the artifacts landed and their builder
did not, and its digest reached no receipt. That is rule 12's shape, and it
is now shut from inside -- this module refuses to emit unless its own bytes
are the bytes HEAD holds (`POINT_ESTIMATE_DRIVER_NOT_COMMITTED`).

THE STATUS IS TESTED BY MEMBERSHIP, NEVER EQUALITY. A point-estimate
arm-day carries `OK_POINT_ESTIMATE`, and comparing a status with `== "OK"`
is the defect class that cost three patches on 2026-09-08. Read
`DE_PROCEDURE` section 8 before adding any run mode or status, and grep for
that shape first.

THE LATENCY IS GUARDED TWICE AND THE TWO ARE NOT THE SAME CHECK:
`POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH` says a FILE is mislabelled;
`POINT_ESTIMATE_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY` says the published
LINK would chain two different makers (R-811, R-828 (2)). A merge once
dropped the second because the first looked like it covered the concept.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import be_placement_latency_reconcile as PLR  # noqa: E402
import de_data_root as DR  # noqa: E402
import de_multiday_gate1_runner as R  # noqa: E402


PROTOCOL = "P003_DE_POINT_ESTIMATE_DAY_V1"
FAMILY = "p003_de_point_estimate_day"
DRIVER_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RESULT_CONTRACT_REFUSAL = "POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION"
#: DE 152, kept from the other DE 151 implementation. This driver
#: DISCOVERS its priors rather than being handed one, which removes the
#: caller-supplied-target failure modes by construction -- but two of that
#: family's guarantees are NOT structural and are asserted here:
#:  * a file in this family must be a POINT_ESTIMATE run. `protocol` and
#:    `day` do not say so; a FULL run written into the family name would
#:    have passed identity and been superseded by a point estimate, which
#:    R-828 forbids (they answer different questions; neither supersedes
#:    the other).
#:  * a prior is discovered BEFORE the replay's artifact is written. The
#:    pair it is named by must still be the bytes on disk AT THE WRITE --
#:    a digest recomputed once and trusted later is how a pair comes to
#:    name bytes nobody has.
SUPERSEDES_NOT_A_POINT_ESTIMATE = (
    "POINT_ESTIMATE_SUPERSEDES_NOT_A_POINT_ESTIMATE_RUN")
SUPERSEDES_TARGET_ABSENT = "POINT_ESTIMATE_SUPERSEDES_TARGET_ABSENT"
SUPERSEDES_DIGEST_MISMATCH = "POINT_ESTIMATE_SUPERSEDES_DIGEST_MISMATCH"
#: DE 153. THE LINK IS GUARDED, NOT ONLY THE DISCOVERY. Two names live
#: here because they are two different facts, and the DE 152 merge lost
#: one by assuming they were one:
#:   * `POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH` says A FILE IS MISLABELLED
#:     -- it sits in the `L<tag>ms` family and declares another L.
#:   * this one says THE RELATIONSHIP IS WRONG -- a run at one placement
#:     latency is about to publish another latency's artifact as its
#:     PREDECESSOR. They are SIBLINGS, not successors (R-811, R-828 (2)):
#:     two runs of one day at different L measure two different makers,
#:     and one chain over both hides that.
#: MEASURED on the pre-DE-153 merge, which is why this is not theoretical:
#: discovery never globbed a foreign-L file and a mislabelled one refused,
#: but `assert_priors_unchanged` -- THE PATH THAT FORMS THE PUBLISHED LINK
#: -- compared no latency at all and ADMITTED an L = 0 target for an
#: L = 250 run. The guarantee rested entirely on a FILENAME. It now rests
#: on the target's own declared L, re-read from disk at the write so a
#: caller-supplied field cannot stand in for it.
SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY = (
    "POINT_ESTIMATE_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY")
NULL_FIELDS = ("Z", "p_location", "null_mean", "null_sd")


def _stamp(when: datetime.datetime) -> str:
    return when.strftime("%Y%m%dT%H%M%SZ")


def _latency_tag(value_ms: float) -> str:
    value = float(value_ms)
    text = str(int(value)) if value.is_integer() else format(value, ".12g")
    return f"L{text.replace('.', 'p')}ms"


def artifact_name(day: str, value_ms: float, when: datetime.datetime) -> str:
    return (f"{FAMILY}_{day.replace('-', '')}_{_latency_tag(value_ms)}__"
            f"{_stamp(when)}.json")


def _sha(path: Path) -> str:
    return R.sha256_streamed(Path(path))


def _refusal_name(exc: Exception) -> str:
    return str(exc).split(":")[0].replace("REFUSED ", "")


def prior_artifacts(output_dir: Path, day: str, value_ms: float) -> list[dict]:
    """Resolve every prior artifact for the same (day, latency)."""
    pattern = (f"{FAMILY}_{day.replace('-', '')}_{_latency_tag(value_ms)}__"
               "*.json")
    candidates = []
    for path in sorted(Path(output_dir).glob(pattern)):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_UNREADABLE: {path.name}: "
                f"{exc}") from exc
        if doc.get("protocol") != PROTOCOL or doc.get("day") != day:
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_IDENTITY_MISMATCH: "
                f"{path.name} matches the family name but declares protocol "
                f"{doc.get('protocol')!r} and day {doc.get('day')!r}.")
        # DE 152: AND IT MUST BE A POINT ESTIMATE. A FULL run at the same
        # (day, L) is a DIFFERENT artifact, not a predecessor -- neither
        # supersedes the other (R-828) -- and `protocol`/`day` cannot tell
        # the two apart on their own.
        if doc.get("run_mode") != "POINT_ESTIMATE":
            raise R.RunnerRefused(
                f"REFUSED {SUPERSEDES_NOT_A_POINT_ESTIMATE}: {path.name} "
                f"declares run_mode {doc.get('run_mode')!r}. A point "
                f"estimate never supersedes a full run and a full run never "
                f"supersedes a point estimate; they answer different "
                f"questions at the same (day, L).")
        top_latency = ((doc.get("placement_latency") or {})
                       .get("L_place_ms"))
        if (isinstance(top_latency, bool)
                or not isinstance(top_latency, (int, float))
                or not math.isfinite(float(top_latency))
                or float(top_latency) != float(value_ms)):
            raise R.RunnerRefused(
                f"REFUSED POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH: "
                f"{path.name} is in the {_latency_tag(value_ms)} family but "
                f"its top-level L_place_ms is {top_latency!r}.")
        consistency = "PLACEMENT_LATENCY_AGREES"
        try:
            R.assert_one_placement_latency(doc)
        except R.RunnerRefused as exc:
            consistency = _refusal_name(exc)
        candidates.append({
            "path": str(path),
            "sha256": _sha(path),
            "day": day,
            "L_place_ms": float(value_ms),
            "target_placement_latency_consistency": consistency,
        })
    return candidates


def prior_artifact(output_dir: Path, day: str, value_ms: float) -> dict | None:
    """Return the newest prior, while the emitter absorbs every sibling."""
    candidates = prior_artifacts(output_dir, day, value_ms)
    return candidates[-1] if candidates else None


def assert_priors_unchanged(priors: list[dict], where: str,
                            this_L: float) -> dict:
    """DE 152: every named prior is STILL the bytes its pair names.

    Priors are resolved before the artifact is written and the replay that
    sits between the two takes minutes. A pair recomputed once and trusted
    afterwards is how a supersession link comes to name bytes nobody has,
    so each one is re-stat'ed and re-digested here, immediately before the
    document is serialised.

    AND EVERY TARGET MUST BE AT THIS RUN'S PLACEMENT LATENCY (DE 153).
    This is the only place the published LINK is checked: discovery guards
    the FILE, not the RELATIONSHIP."""
    checked = []
    for prior in priors:
        path = Path(prior["path"])
        if not path.is_file():
            raise R.RunnerRefused(
                f"REFUSED {SUPERSEDES_TARGET_ABSENT}: at {where}, "
                f"{path.name} was resolved as a prior and is no longer on "
                f"disk. A link to a file nobody has is not a link.")
        now = _sha(path)
        if now != prior["sha256"]:
            raise R.RunnerRefused(
                f"REFUSED {SUPERSEDES_DIGEST_MISMATCH}: at {where}, "
                f"{path.name} digested {prior['sha256'][:16]} when it was "
                f"resolved and {now[:16]} now. The bytes this run claims to "
                f"supersede changed underneath it.")
        # THE TARGET'S OWN DECLARED L, READ FROM THE FILE AT THE WRITE --
        # never the caller's `L_place_ms` field, which is exactly what a
        # wrong link would carry.
        try:
            declared = ((json.loads(path.read_text()).get(
                "placement_latency") or {}).get("L_place_ms"))
        except (OSError, ValueError) as exc:
            raise R.RunnerRefused(
                f"REFUSED {SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY}: at "
                f"{where}, {path.name} could not be read to confirm the "
                f"latency it ran at ({exc}). A link whose target's L cannot "
                f"be established is not a link.") from exc
        if (isinstance(declared, bool)
                or not isinstance(declared, (int, float))
                or not math.isfinite(float(declared))
                or float(declared) != float(this_L)):
            raise R.RunnerRefused(
                f"REFUSED {SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY}: at "
                f"{where}, this run is at L_place = {this_L} ms and "
                f"{path.name} declares {declared!r}. A run at a different "
                f"placement latency measures a DIFFERENT MAKER: it is a "
                f"SIBLING, not a successor (R-811, R-828 (2)). Quote each "
                f"with its own L beside it; never chain them.")
        checked.append({"path": str(path), "sha256": now,
                        "declared_L_place_ms": float(declared)})
    return {"status": "SUPERSEDED_PAIRS_STILL_NAME_THEIR_BYTES",
            "all_targets_at_this_L_place_ms": float(this_L),
            "n_priors_reverified": len(checked), "priors": checked,
            "why": ("resolved before the replay, re-verified at the write; "
                    "the window between the two is minutes long")}


def assert_driver_source(where: str) -> dict:
    """Bind the outer artifact to this committed driver and the runner."""
    path = Path(__file__).resolve()
    now = hashlib.sha256(path.read_bytes()).hexdigest()
    if now != DRIVER_SHA256_AT_IMPORT:
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_DRIVER_CHANGED_DURING_RUN: at {where} "
            f"the driver imported {DRIVER_SHA256_AT_IMPORT[:16]} and is now "
            f"{now[:16]}.")
    committed = R.carrying_commit_block(path)
    if not committed["producing_code_is_the_committed_bytes"]:
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_DRIVER_NOT_COMMITTED: at {where} the "
            "point-estimate artifact builder is not the file held by "
            "HEAD. Commit the driver before emitting a result-bearing day.")
    runner = R.assert_source_unchanged(where, fixture=False)
    return {
        "driver": {
            **committed,
            "sha256_at_import": DRIVER_SHA256_AT_IMPORT,
            "sha256_at_emit": now,
            "unchanged_during_run": now == DRIVER_SHA256_AT_IMPORT,
        },
        "runner": runner,
    }


def assert_point_estimate_result(result: dict) -> dict:
    """Require settlement values and named no-null statuses before emit."""
    problems = []
    if result.get("run_mode") != "POINT_ESTIMATE":
        problems.append(f"run_mode={result.get('run_mode')!r}")
    try:
        R.assert_one_placement_latency(result)
    except R.RunnerRefused as exc:
        problems.append(_refusal_name(exc))
    ledger = result.get("decision_ledger") or {}
    if ledger.get("run_mode") != "POINT_ESTIMATE":
        problems.append(f"decision_ledger.run_mode={ledger.get('run_mode')!r}")

    valued_arms = []
    for arm_day in result.get("per_day_sealed_artifacts") or []:
        if arm_day.get("status") != "OK_POINT_ESTIMATE":
            continue
        arm = arm_day.get("arm")
        economic = arm_day.get("economic") or {}
        settlement = arm_day.get("economic_settlement") or {}
        d_settle = settlement.get("D_E_settle")
        if (isinstance(d_settle, bool)
                or not isinstance(d_settle, (int, float))
                or not math.isfinite(float(d_settle))):
            problems.append(f"{arm}.economic_settlement.D_E_settle absent")
            continue
        for label, block in (("economic", economic),
                             ("economic_settlement", settlement)):
            for field in NULL_FIELDS:
                if block.get(field) != R.NULL_NOT_DRAWN_STATUS:
                    problems.append(
                        f"{arm}.{label}.{field}={block.get(field)!r}")
            draws = block.get("null_draws_summary") or {}
            if draws.get("n") != R.NULL_NOT_DRAWN_STATUS:
                problems.append(
                    f"{arm}.{label}.null_draws_summary.n="
                    f"{draws.get('n')!r}")
        valued_arms.append(arm)
    if not valued_arms:
        problems.append("no OK_POINT_ESTIMATE arm carries ruled settlement P&L")
    n_admissible = result.get("n_admissible_arms")
    if n_admissible != len(valued_arms):
        problems.append(
            f"n_admissible_arms={n_admissible!r} but valued arms="
            f"{valued_arms}")
    if problems:
        raise R.RunnerRefused(
            f"REFUSED {RESULT_CONTRACT_REFUSAL}: {problems}. The point-"
            "estimate family claims ruled settlement P&L with no null; it "
            "must not publish a diagnostic-only day or a partial statistics "
            "block.")
    return {"status": "POINT_ESTIMATE_RESULT_CONTRACT_SATISFIED",
            "valued_arms": valued_arms,
            "null_status": R.NULL_NOT_DRAWN_STATUS}


def write_artifact(path: Path, payload: dict) -> dict:
    """Validate the serialized document before atomically publishing it."""
    path = Path(path)
    if path.exists():
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_OUTPUT_EXISTS: {path} already exists; "
            "a result artifact is never overwritten.")
    _missing_req = [k for k in ("population_and_coverage", "scope",
                                "arm_provenance_caveat")
                    if not payload.get(k)]
    if _missing_req:
        raise R.RunnerRefused(
            f"REFUSED {REQUIRED_QUOTATION_FIELDS_ABSENT}: {_missing_req}. "
            f"DA 180 found both gaps on a LANDED artifact: the exclusions "
            f"and coverage were resolvable only from the book receipt -- a "
            f"different document -- and the coin was not resolvable at "
            f"all without R-869 in hand. A result that cannot state its "
            f"own population and scope is not quotable, so it is not "
            f"written (rule 35).")
    payload["private_key_sweep"] = assert_no_private_keys(payload)
    payload["placement_latency_consistency"] = (
        R.assert_one_placement_latency(payload))
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    parsed = json.loads(encoded)
    R.assert_one_placement_latency(parsed)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        reread = json.loads(temporary.read_text())
        R.assert_one_placement_latency(reread)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path), "sha256": _sha(path),
            "bytes": path.stat().st_size}


def waiver_scoring_block(asked: bool, result: dict) -> dict:
    """What was asked for, and what the predicate did with it.

    REPORTED FROM THE RESULT, never echoed from the argument: what the
    predicate DID is the fact, and the flag is only what it was asked to
    do (the same rule the null's `matched_on` is read back under)."""
    verdict = ((result.get("book_scoring_code") or {}))
    granted = verdict.get("status") == R.WAIVED_SCORING_PATH
    return {
        "asked_by_the_caller": bool(asked),
        "granted_by_the_predicate": granted,
        "book_scoring_status": verdict.get("status"),
        "read_from": ("day_run.book_scoring_code.status -- the run's own "
                      "result, not this driver's argument"),
        "what_it_means": (
            "the book-code predicate REFUSED this book by name and the "
            "refusal was OVERRIDDEN on computed evidence: nothing that "
            "differs between the book's scoring code and the code on disk "
            "is reachable from the builder's scoring entry points. The "
            "evidence, its five conditions and its LIMITS are in "
            "`day_run.book_scoring_code.scoring_path_waiver`"
            if granted else
            "no waiver was granted; if a waiver was asked for and this "
            "says False, the run refused rather than proceeding"),
        "authorised_by": ("USER ruling relayed at DE 180 -- waive AS A "
                          "PREDICATE, on DA 168's and REV 146's "
                          "independently derived evidence"
                          if asked else None)}


RECONCILE_INPUTS_ABSENT = "POINT_ESTIMATE_RECONCILIATION_INPUTS_ABSENT"
RECONCILE_INPUTS_INCOMPLETE = "POINT_ESTIMATE_RECONCILIATION_INPUTS_INCOMPLETE"
RECONCILE_INPUTS_COVERAGE = (
    "POINT_ESTIMATE_RECONCILIATION_INPUT_COVERAGE_MISMATCH")
RECONCILE_INPUT_IDENTITY = (
    "POINT_ESTIMATE_RECONCILIATION_INPUT_IDENTITY_MISMATCH")
RECONCILE_INPUTS_DISAGREE = "POINT_ESTIMATE_RECONCILIATION_INPUTS_DISAGREE"
PRIVATE_KEY_SURVIVED = "POINT_ESTIMATE_PRIVATE_KEY_REACHED_THE_PAYLOAD"
RECONCILE_NON_FINITE = "POINT_ESTIMATE_RECONCILIATION_NOT_FINITE"
RECONCILE_NO_SPLIT = "RECONCILIATION_UNAVAILABLE_BOOK_CARRIES_NO_SPLIT"
RECONCILED_ARM_STATUS = "RECONCILED_BY_POINT_ESTIMATE_DRIVER"
RECONCILIATION_UNAVAILABLE_ARM_STATUS = (
    "RECONCILIATION_UNAVAILABLE_REPORTED_BY_POINT_ESTIMATE_DRIVER")


def _finite_scalars(node, path="$"):
    """Every float in a block, with its path. Bools are not numbers here."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += _finite_scalars(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            out += _finite_scalars(v, f"{path}[{i}]")
    elif isinstance(node, float) and not isinstance(node, bool):
        out.append((path, node))
    return out


COVERAGE_UNRESOLVABLE = "POINT_ESTIMATE_COVERAGE_NOT_RESOLVABLE"
POPULATION_DISAGREES = "POINT_ESTIMATE_POPULATION_DISAGREES_WITH_THE_BOOK"
SCOPE_UNRESOLVABLE = "POINT_ESTIMATE_SCOPE_NOT_RESOLVABLE"
REQUIRED_QUOTATION_FIELDS_ABSENT = "POINT_ESTIMATE_NOT_QUOTABLE_FIELDS_ABSENT"


ARM_CAVEAT_ABSENT = "POINT_ESTIMATE_ARM_PROVENANCE_CAVEAT_ABSENT"


def arm_provenance_caveat(result: dict, reconciliation: dict) -> dict:
    """WHY THE ARM NUMBERS CARRY A WEAKER PROVENANCE THAN THE BASELINE.

    REV 156. The scoring-path waiver's provenance claim is UNSUPPORTED --
    not refuted: the reachability half rested on THREE confirmations that
    all resolve to ONE operation (`be_producing_closure.reachable_modules`,
    rule 38), and DE 195 MEASURED that operation erring in BOTH directions
    on a controlled fixture, so "X was not reached" is not a conservative
    claim.

    AND THE RECONCILIATION DOES NOT COVER THE GAP, WHICH IS THE POINT.
    `kept_equals_the_baseline` holds to ~1e-11 -- but the KEPT side IS the
    ZERO-CANCEL BASELINE, and the baseline **makes no decisions**. No
    scoring code can move it. So the half that reconciles is exactly the
    half that could not have moved, and the ARM totals -- which depend on
    which generations were cancelled, i.e. on the scores -- are the half
    that could. They sit on the side the waiver governs.

    This is a FIELD and not a sentence in a report because a caveat that
    lives outside the artifact does not travel with the number (rule 35)."""
    verdict = (result.get("book_scoring_code") or {})
    waived = verdict.get("status") == R.WAIVED_SCORING_PATH
    arms = [a.get("arm") for a in
            (result.get("per_day_sealed_artifacts") or [])]
    # ---- DE 202: THE WAIVER'S SCOPE, MEASURED AND ENUMERATED ---------
    # A bare "waiver invoked" reads as the blanket thing REV 152 found --
    # three confirmations through one operation. THIS one has a measured
    # scope, so the scope travels: which module differs, what of it is on
    # the scoring path, and the INTERSECTION with what changed. All READ
    # from the run's own evidence, never typed here.
    _w = (verdict.get("scoring_path_waiver") or {})
    _ev = (_w.get("evidence") or {}).get("modules") or {}
    _inter = {m: ((b.get("INTERSECTION") or {}).get("defs", [])
                  + (b.get("INTERSECTION") or {}).get("module_level_names",
                                                      []))
              for m, b in _ev.items()}
    _n_inter = sum(len(v) for v in _inter.values())
    scoping = {
        "waiver_invoked": bool(waived),
        "what_differs_from_the_books_recorded_bytes": sorted(_ev),
        "on_the_scoring_path_in_those_modules": {
            m: ((b.get("on_the_path") or {}).get("defs", [])
                + (b.get("on_the_path") or {}).get(
                    "module_level_names_read_by_them", []))
            for m, b in _ev.items()},
        "INTERSECTION_with_what_changed": _inter,
        "n_intersection": _n_inter,
        "the_difference_does_not_touch_what_produced_the_scores": (
            _n_inter == 0),
        "why_the_waiver_is_UNAVOIDABLE_here": (
            "THE EXACT BYTES A BOOK RECORDS CANNOT REPLAY THAT BOOK. "
            "Measured, not reasoned: 09-05's receipt records runner "
            "`c9f36a743bec` (a49dd34), and that runner REFUSES 09-05's "
            "OWN receipt with SETTLEMENT_BOOK_PLACEMENT_LATENCY_AMBIGUOUS "
            "-- it predates DE 185's split-aware guard. The runner must be "
            "CURRENT to replay a split-carrying book at all, so it must "
            "DIFFER, so the waiver is invoked. Pinning it fully would also "
            "drop DE 189's complement-leg naming and DE 190's mode-gated "
            "status, and for 09-04 the reconciliation handover entirely. "
            "This is an IMPOSSIBILITY, not a shortfall -- a later reader "
            "asking 'why not just pin it' has the answer here"),
        "what_IS_pinned_by_identity": (
            "the two SCORING modules. `de_head_scoring.py` and "
            "`de_phase4_diag_runner.py` are at the digests EVERY book "
            "records, so the code that produced the cached scores is the "
            "code that replayed them -- by identity, not by argument"),
        "how_this_differs_from_a_blanket_waiver": (
            "REV 152's finding was a waiver resting on three confirmations "
            "that all resolve to ONE operation. This one's SCOPE is "
            "measured and enumerated above, and the enumeration is read "
            "from the run's own evidence"),
    }
    return {
        **scoping,
        "applies_to": "EVERY ARM FIGURE IN THIS ARTIFACT",
        "arms": arms,
        "scoring_provenance": ("WAIVED -- the book-code predicate REFUSED "
                               "this book and the refusal was overridden "
                               "on computed evidence"
                               if waived else
                               verdict.get("status")),
        "the_waiver_claim_is": "UNSUPPORTED, NOT REFUTED",
        "why_unsupported": (
            "its reachability half rested on three confirmations that all "
            "resolve to ONE operation (be_producing_closure."
            "reachable_modules, rule 38), and DE 195 measured that "
            "operation erring in BOTH directions on a controlled fixture "
            "-- so `X was not reached` is not a conservative claim"),
        "what_the_reconciliation_does_NOT_cover": (
            "the KEPT side reconciles to the ledger baseline to ~1e-11, "
            "and that is EXACTLY THE HALF THAT COULD NOT HAVE MOVED: the "
            "zero-cancel baseline MAKES NO DECISIONS, so no scoring code "
            "can touch it. The ARM totals depend on which generations "
            "were cancelled, hence on the scores, and are the half that "
            "could move"),
        "kept_equals_the_baseline": reconciliation.get(
            "kept_equals_the_baseline"),
        "what_would_close_it": (
            "a DYNAMIC trace of the real scoring entry points recording "
            "which attributes of the moved module are ACTUALLY touched "
            "(de_dynamic_reach; the instrument exists and is falsified, "
            "and has NOT yet been run on a book build)"),
        "what_must_not_be_said": (
            "that the reconciliation validates the arm numbers, or that "
            "the waiver was verified. Neither is true: one checks a half "
            "that cannot move, and the other is an override on evidence "
            "whose support is singular"),
    }


def population_and_coverage(reconciliation: dict, receipt: dict,
                            receipt_path: Path) -> dict:
    """THE COUNTS AND THE COVERAGE, ON THE RESULT -- rule 4 where the
    number is.

    DA 180: `UNCOVERED`, `TRANCHE_KEPT`,
    `TRANCHE_BEFORE_PLACEMENT_LATENCY` and `BINANCE_GAP_EXCLUDED` occur
    ZERO times in a point-estimate artifact, and coverage 0.9175 with
    29,530 uncovered lives only in the BOOK RECEIPT -- A DIFFERENT
    DOCUMENT. A reader holding the result cannot state the population the
    number was computed over.

    THE COUNTS COME FROM THE RECONCILIATION THIS RUN PERFORMED, not from
    the receipt: `KEPT.n_fills` and `DROPPED.n_fills` are counted over
    the same tranches whose value is quoted beside them, so a count and
    its money cannot drift apart. The COVERAGE cannot be computed here --
    it is a property of the book's ASSEMBLY, which this driver never
    loads -- so it is READ, with its source path and digest, AND
    CROSS-CHECKED against the generation count the reconciliation itself
    walked. A disagreement REFUSES: two documents that disagree about the
    population cannot jointly describe one number."""
    ev = (receipt or {}).get("assembly_evidence") or {}
    unc = ev.get("UNCOVERED_GENERATIONS") or {}
    if not unc or unc.get("coverage") is None:
        raise R.RunnerRefused(
            f"REFUSED {COVERAGE_UNRESOLVABLE}: the builder receipt at "
            f"{receipt_path} carries no "
            f"`assembly_evidence.UNCOVERED_GENERATIONS.coverage`, so the "
            f"population this number was computed over cannot be stated "
            f"on the result. A figure whose coverage no reader can "
            f"resolve is not quotable (rule 4).")
    gens = reconciliation.get("n_generations")
    n_ref = unc.get("n_reference_generations")
    if gens is not None and n_ref != gens:
        raise R.RunnerRefused(
            f"REFUSED {POPULATION_DISAGREES}: the book receipt says its "
            f"reference holds {n_ref} generations and the reconciliation "
            f"walked {gens}. The two documents disagree about the "
            f"population, so neither the counts nor the coverage may be "
            f"stated beside this number.")
    kept = ((reconciliation.get("KEPT") or {}).get("n_fills"))
    dropped = ((reconciliation.get("DROPPED") or {}).get("n_fills"))
    sel = (receipt or {}).get("selection") or {}
    st = ((receipt or {}).get("reference") or {}).get("statuses") or {}
    pl = (receipt or {}).get("placement_latency") or {}
    out = {
        "counts_are_from": ("the reconciliation THIS RUN performed -- the "
                            "same tranches whose value is quoted"),
        "n_generations": gens,
        "TRANCHE_KEPT": kept,
        "TRANCHE_BEFORE_PLACEMENT_LATENCY": (
            dropped if dropped is not None else pl.get("n_tranches_dropped")),
        "TRANCHE_BEFORE_PLACEMENT_LATENCY_source": (
            "counted by the reconciliation" if dropped is not None else
            "READ from the receipt's `n_tranches_dropped` -- this book "
            "DISCARDED them at build time, so this run could not count "
            "them (see the complement-leg status)"),
        "coverage": {
            "coverage": unc.get("coverage"),
            "n_uncovered": unc.get("count"),
            "n_reference_generations": n_ref,
            "read_from": str(receipt_path),
            "read_from_sha256": _sha(receipt_path),
            "computed_here": False,
            "why_not": ("coverage is a property of the book's ASSEMBLY, "
                        "which this driver never loads"),
            "cross_check": ("the receipt's `n_reference_generations` "
                            "equals the generation count the "
                            "reconciliation walked"),
            "cross_check_holds": (gens is not None and n_ref == gens),
            "cross_check_was_possible": gens is not None},
        "era": sel.get("era"),
        "n_gap_bearing_windows": sel.get("n_gap_bearing_windows"),
        "BINANCE_GAP_EXCLUDED": st.get("BINANCE_GAP_EXCLUDED"),
        "BINANCE_GAP_EXCLUDED_STATUS": st.get("BINANCE_GAP_EXCLUDED_STATUS"),
        "TERMINAL_MARK_ENDED_IN_GAP": st.get("TERMINAL_MARK_ENDED_IN_GAP"),
        "ADMITTED": st.get("ADMITTED"),
    }
    out["how_to_quote_this"] = (
        f"every figure in this artifact is over {kept} kept tranches on "
        f"{gens} generations at coverage {unc.get('coverage')}, with "
        f"{unc.get('count')} generations uncovered. The coverage is the "
        f"share of reference generations a scored key was found for -- "
        f"NOT one minus a failure rate. `BINANCE_GAP_EXCLUDED: "
        f"{st.get('BINANCE_GAP_EXCLUDED')}` carries its own status "
        f"({st.get('BINANCE_GAP_EXCLUDED_STATUS')}) and is not a "
        f"measurement of zero gaps.")
    return out


def scope_declaration(book_path, receipt: dict) -> dict:
    """WHAT THIS NUMBER IS ABOUT -- resolvable without R-869 in hand.

    DA 180: `BTC_ONLY` and `btc_only` are absent from the artifact and
    `coin` appears twice, neither as a scope declaration -- so a reader
    must already know that no eth Gate-1 tape exists to avoid
    over-reading the figure as a market-wide one. The coin is DERIVED
    from the artifacts (the book's own filename and the receipt's own
    coin field, which must agree) rather than typed, and a disagreement
    or an unresolvable coin REFUSES."""
    name = Path(book_path).name
    from_name = [c for c in ("btc", "eth") if f"_{c}_" in name
                 or name.startswith(f"{c}_")]
    from_receipt = (receipt or {}).get("coin")
    if from_receipt is None:
        sel = (receipt or {}).get("selection") or {}
        from_receipt = sel.get("coin")
    coins = sorted({*from_name, *([from_receipt] if from_receipt else [])})
    if len(coins) != 1:
        raise R.RunnerRefused(
            f"REFUSED {SCOPE_UNRESOLVABLE}: the book filename implies "
            f"{from_name} and the receipt says {from_receipt!r}, giving "
            f"{coins}. A result must name the population it is about; a "
            f"figure whose coin a reader has to infer is one a reader can "
            f"over-read as market-wide.")
    coin = coins[0]
    return {
        "coin": coin,
        "BTC_ONLY": coin == "btc",
        "coins_this_result_covers": [coin],
        "resolved_from": {"book_filename": from_name,
                          "builder_receipt_coin": from_receipt,
                          "they_agree": True},
        "why_only_one_coin": (
            "R-869: NO eth Gate-1 tape exists for ANY day, so no arm-day "
            "for another coin can be computed. This is a property of the "
            "DATA, not a choice made in this run"),
        "what_must_not_be_said": (
            "that this is a market-wide or multi-coin result, or that the "
            "effect generalises to another instrument. One coin, and the "
            "reason the others are absent is that they have no tape"),
    }


def reconcile_placement_latency(result: dict) -> dict:
    """DA's reconciliation, run where BOTH numbers exist.

    THE INPUTS COME FROM THE RUN ITSELF, not from a second derivation.
    `run_day` attaches the reference it valued, the winners it used and
    the zero-cancel baseline it computed under a PRIVATE key; this pops
    them, checks they agree across arms, and hands them to BE's checker.
    Re-deriving any of the three here would be a different population
    valued a different way, and the reconciliation would then be
    comparing this function's work with itself.

    IT RAISES. `be_placement_latency_reconcile` ships two forms and the
    choice between them is the whole point: `reconcile_status` is for a
    long run that must not die for a diagnostic, and `reconcile` is for a
    caller that would rather not publish. An artifact is a publication."""
    arms = result.get("per_day_sealed_artifacts") or []
    valued = [a for a in arms if a.get("status") == "OK_POINT_ESTIMATE"]
    found = []
    missing = []
    unexpected = []
    for arm_day in arms:
        is_valued = arm_day.get("status") == "OK_POINT_ESTIMATE"
        got = arm_day.pop(R.PLR_INPUTS_KEY, None)
        if got is None:
            if is_valued:
                missing.append(arm_day.get("arm"))
            continue
        found.append((arm_day, got))
        if not is_valued:
            unexpected.append(arm_day.get("arm"))
    if not found:
        raise R.RunnerRefused(
            f"REFUSED {RECONCILE_INPUTS_ABSENT}: not one of the "
            f"{len(arms)} arm-days carried "
            f"`{R.PLR_INPUTS_KEY}`. The runner attaches it whenever a "
            f"winner source resolved in point-estimate mode, so its "
            f"absence means either that no settlement was valued or that "
            f"this result came from a path that does not reconcile. "
            f"Emitting a point estimate with the reconciliation silently "
            f"skipped is the shape this wiring exists to close.")
    if missing or unexpected or len(found) != len(valued):
        raise R.RunnerRefused(
            f"REFUSED {RECONCILE_INPUTS_COVERAGE}: valued arms "
            f"{[a.get('arm') for a in valued]} but handover is missing from "
            f"{missing} and unexpectedly present on {unexpected}. Every "
            f"valued arm must hand over exactly one input block before the "
            f"driver may claim that all arms agreed.")
    for key in ("reference", "winners", "baseline_total_cents"):
        for _, handed in found:
            if handed.get(key) is None:
                raise R.RunnerRefused(
                    f"REFUSED {RECONCILE_INPUTS_INCOMPLETE}: an arm-day "
                    f"handed over `{key}` as None. A missing input is a "
                    f"refusal, never a default: BE's checker would value "
                    f"an empty set and report a clean zero.")
    result_day = result.get("day")
    for arm_day, handed in found:
        if handed.get("arm") != arm_day.get("arm") or (
                result_day is not None and handed.get("day") != result_day):
            raise R.RunnerRefused(
                f"REFUSED {RECONCILE_INPUT_IDENTITY}: container "
                f"{result_day!r}/{arm_day.get('arm')!r} received handover "
                f"{handed.get('day')!r}/{handed.get('arm')!r}. Inputs from "
                f"another arm-day cannot certify this one.")
    baselines = []
    for arm_day, handed in found:
        value = handed["baseline_total_cents"]
        try:
            baseline = float(value)
        except (TypeError, ValueError) as exc:
            raise R.RunnerRefused(
                f"REFUSED {RECONCILE_INPUTS_INCOMPLETE}: arm "
                f"{arm_day.get('arm')!r} handed over a non-numeric baseline "
                f"{value!r}.") from exc
        if isinstance(value, bool) or not math.isfinite(baseline):
            raise R.RunnerRefused(
                f"REFUSED {RECONCILE_NON_FINITE}: arm "
                f"{arm_day.get('arm')!r} handed over non-finite baseline "
                f"{value!r}. Cross-arm comparisons must be finite before "
                f"their tolerance is evaluated.")
        baselines.append(baseline)
    # THE ARMS MUST AGREE ON ALL THREE. They are properties of the DAY,
    # not of an arm -- the same book reference, the same winners, the
    # same zero-cancel baseline (the baseline makes no decisions, so it
    # cannot differ by arm). A disagreement means one arm valued a
    # different population, and reconciling only the first would hide it.
    first_arm, first = found[0]
    first_baseline = baselines[0]
    for (other_arm, other), other_baseline in zip(found[1:], baselines[1:]):
        if (other["reference"] is not first["reference"]
                or other["winners"] is not first["winners"]
                or abs(other_baseline - first_baseline) > 1e-9):
            raise R.RunnerRefused(
                f"REFUSED {RECONCILE_INPUTS_DISAGREE}: arm "
                f"{other_arm.get('arm')!r} handed over a different reference, "
                f"a different winner map or a different zero-cancel "
                f"baseline than arm {first_arm.get('arm')!r} "
                f"({other['baseline_total_cents']!r} against "
                f"{first['baseline_total_cents']!r}). These are "
                f"properties of the DAY; if they differ, one of the two "
                f"arm-days valued a population the other did not.")
    # ---- DE 186: A BOOK WITH NO SPLIT SAYS SO. IT DOES NOT RECONCILE
    # AGAINST NOTHING. ---------------------------------------------------
    # 09-03's EV21 book has no `placement_latency_split`: BE's own
    # `_both_sets` refuses DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK on it.
    # That is a property of how the BOOK WAS BUILT, not a disagreement
    # between numbers, so it is a STATUS and not a refusal -- refusing
    # would make a whole class of existing books un-runnable for a reason
    # that says nothing about the numbers being published. But it is a
    # LOUD status: `ok: False`, `MUST_BE_SURFACED: True`, and a name, so
    # a reader can never mistake "the check could not run" for "the check
    # passed". A PARTIAL split stays a REFUSAL -- that one IS a defect.
    try:
        out = PLR.reconcile(first["reference"], first["winners"],
                            first_baseline)
    except PLR.ReconcileRefused as e:
        if "DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK" not in str(e):
            raise
        out = {"protocol": "BE_PLACEMENT_LATENCY_RECONCILE_V1",
               "ok": False,
               "MUST_BE_SURFACED": True,
               "status": RECONCILE_NO_SPLIT,
               "refusal_from_the_checker": str(e).split(":")[0].strip(),
               "what_is_unavailable": (
                   "the L=0 complement leg. This book carries only the "
                   "KEPT tranches, so KEPT-value cannot be compared "
                   "against anything and the placement-latency question "
                   "HAS NOT BEEN CHECKED for this day"),
               "what_must_not_be_said": (
                   "that the reconciliation passed, or that the latency "
                   "effect is zero here. Neither was measured: the "
                   "comparison has no second term"),
               "how_to_get_it": (
                   "rebuild the day with `split_by_placement_latency`, "
                   "which keeps the complement instead of counting and "
                   "discarding it (BE 133). 09-04 onward carry it"),
               "day": first.get("day"),
               "baseline_total_cents_from_the_ledger": first_baseline}
    _nf = [(k, v) for k, v in _finite_scalars(out) if not math.isfinite(v)]
    if _nf:
        raise R.RunnerRefused(
            f"REFUSED {RECONCILE_NON_FINITE}: the reconciliation returned "
            f"non-finite values at {[k for k, _ in _nf]}. Every equality "
            f"in it PASSED, because a NaN compares False to everything -- "
            f"so a clean-looking reconciliation is exactly what a "
            f"non-finite input produces. The artifact is not written.")
    for arm_day, _ in found:
        arm_day["placement_latency_reconciliation"] = {
            "status": (RECONCILIATION_UNAVAILABLE_ARM_STATUS
                       if out.get("status") == RECONCILE_NO_SPLIT else
                       RECONCILED_ARM_STATUS),
            "result_location": "artifact.placement_latency_reconciliation",
            "checker_protocol": out.get("protocol"),
            "arm": arm_day.get("arm"),
        }
    return dict(out,
                run_by="de_point_estimate_day.reconcile_placement_latency",
                inputs_from=("the run itself -- the reference `run_day` "
                             "valued, the winners it used and the "
                             "zero-cancel baseline it computed, handed "
                             "over under a private key and popped here"),
                n_arm_days_that_handed_over=len(found),
                arm_days_that_handed_over=[a.get("arm") for a, _ in found],
                arms_agreed_on_all_three_inputs=True,
                the_refusal_is_not_absorbed=(
                    "`reconcile` RAISES and this function does not catch "
                    "it: a point estimate that does not reconcile is NOT "
                    "EMITTED. BE's non-throwing `reconcile_status` exists "
                    "for a consumer that must not die; an artifact is a "
                    "publication and this is not that consumer"))


def assert_no_private_keys(payload: dict) -> dict:
    """NOTHING PRIVATE REACHES THE BYTES -- checked, not intended.

    The runner hands over a LIVE 300 MB reference object. A pop that
    silently missed one would either blow the artifact up or serialise a
    book into it, and the failure would arrive at `json.dumps` as a type
    error with no name on it. So the emit asserts the property instead."""
    hits = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("_") \
                        and k.endswith("_DO_NOT_SERIALISE"):
                    hits.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(payload)
    if hits:
        raise R.RunnerRefused(
            f"REFUSED {PRIVATE_KEY_SURVIVED}: {hits}. A key the runner "
            f"marked DO_NOT_SERIALISE reached the payload, so a live "
            f"object was about to be written into an artifact.")
    return {"checked": True, "n_private_keys_found": 0,
            "rule": "a key ending `_DO_NOT_SERIALISE` may not reach the "
                    "bytes; the emit asserts it rather than trusting the "
                    "pop"}


def waiver_token(asked: bool):
    """The token when a caller ASKED, and None otherwise.

    Read from the module that defines it rather than retyped -- a constant
    typed twice is a constant that can disagree with itself, and this one
    is the difference between a waiver and a bypass."""
    import de_scoring_path_delta as SPD
    return SPD.WAIVER_TOKEN if asked else None


def run(day: str, book: Path, output_dir: Path, *,
        waive_scoring_path: bool = False) -> dict:
    started = time.time()
    book = Path(book).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise R.RunnerRefused(
            f"REFUSED POINT_ESTIMATE_OUTPUT_NOT_DIRECTORY: {output_dir}")

    # Refuse uncommitted or drifting producers before paying for the replay,
    # then check the same identities again immediately before publication.
    assert_driver_source("the point-estimate preflight")
    params = R.load_params()
    result = R.run_day(
        day, book, params=params, fixture=False,
        n_days_complete=R.days_complete_now(params)["n_days_complete"],
        point_estimate=True, ledger_anchor=output_dir,
        scoring_path_waiver=waiver_token(waive_scoring_path),
        before_work=lambda: R.selftest(quiet=True, offline=False),
    )
    result_contract = assert_point_estimate_result(result)
    # ---- DE 181: THE PLACEMENT-LATENCY RECONCILIATION, HERE AND NOT IN
    # THE RUNNER (USER ruling) ------------------------------------------
    # `de_multiday_gate1_runner.py` is ON THE BOOK'S SCORING PATH: an edit
    # there stales every book on disk, which is what forced today's
    # waiver. This driver is not on that path, so the call costs no book.
    # It runs AFTER the replay and BEFORE the emit, and a
    # `ReconcileRefused` PROPAGATES: the artifact is not written. That is
    # deliberate -- BE ships a non-throwing `reconcile_status` for a
    # consumer that must not die, and this is not that consumer. A point
    # estimate whose kept tranches do not value to the ledger's own
    # baseline is not a result with a caveat; it is a number nobody should
    # read.
    reconciliation = reconcile_placement_latency(result)
    # ---- DA 180's TWO QUOTATION GAPS, CLOSED ON THE RESULT -----------
    # (1) the counts and the coverage the number was computed over, and
    # (2) the coin it is about. Both were resolvable only by opening a
    # DIFFERENT document (the book receipt) or by already knowing R-869.
    # Rule 35: a limit that lives only in a declaration does not bind the
    # result, so they are REQUIRED FIELDS here and the emit refuses
    # without them.
    _receipt_path = Path(result["reference_book"]["builder_receipt"])
    _receipt_doc = json.loads(_receipt_path.read_text())
    population = population_and_coverage(reconciliation, _receipt_doc,
                                         _receipt_path)
    scope = scope_declaration(book, _receipt_doc)
    placement = result.get("placement_latency") or {}
    value_ms = placement.get("L_place_ms")
    if value_ms is None:
        raise R.RunnerRefused(
            "REFUSED POINT_ESTIMATE_PLACEMENT_LATENCY_ABSENT: run_day "
            "returned no L_place_ms.")

    emitted_at = datetime.datetime.now(datetime.timezone.utc)
    path = output_dir / artifact_name(day, value_ms, emitted_at)
    prior = prior_artifacts(output_dir, day, value_ms)
    supersedes = prior[-1] if prior else None
    also_supersedes = prior[:-1]
    source_identity = assert_driver_source("the point-estimate emit")
    priors_reverified = assert_priors_unchanged(
        prior, "the point-estimate emit", value_ms)
    receipt_path = Path(result["reference_book"]["builder_receipt"])
    payload = {
        "protocol": PROTOCOL,
        "what_this_is": (
            "the ruled settlement P&L at the book's declared placement "
            "latency, without a null: S4 was skipped, so no Z, p or "
            "interval exists and nothing here implies significance"),
        "status": result.get("status"),
        "day": day,
        "run_mode": result["run_mode"],
        "result_contract": result_contract,
        "placement_latency": placement,
        "placement_latency_reconciliation": reconciliation,
        "population_and_coverage": population,
        "scope": scope,
        "arm_provenance_caveat": arm_provenance_caveat(
            result, reconciliation),
        # THE ASK IS AT THE TOP OF THE ARTIFACT, not only nested inside
        # `day_run.book_scoring_code`. A reader deciding how much to trust
        # this number must meet the fact that a firing check was overridden
        # BEFORE they meet the number, and the evidence the predicate
        # granted on travels with it.
        "scoring_path_waiver_requested": waiver_scoring_block(
            waive_scoring_path, result),
        "book": {
            "path": str(book),
            "sha256": result["reference_book"]["sha256"],
            "builder_receipt": {
                "path": str(receipt_path),
                "sha256": _sha(receipt_path),
            },
        },
        "supersedes": supersedes,
        "also_supersedes": also_supersedes,
        "supersession_scope": {
            "n_prior_artifacts_for_day_and_latency": len(prior),
            "all_prior_artifacts_are_named": len(prior)
            == (1 if supersedes else 0) + len(also_supersedes),
        },
        "supersession_reverified_at_the_write": priors_reverified,
        "supersession_note": (
            "every prior artifact remains unedited; supersedes names the "
            "newest pair and also_supersedes absorbs any earlier unlinked "
            "sibling, including an abandoned attempt" if supersedes else
            "no prior artifact exists for this (day, L_place_ms) family"),
        "source_identity": source_identity,
        "invocation": {
            "argv": list(sys.argv),
            "interpreter": sys.executable,
            "exclusive_lock_verified_by": (
                "run_day.assert_lock_form_at_runtime and "
                "assert_real_day_has_the_lock"),
        },
        "day_run": result,
        "as_of": emitted_at.isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    artifact = write_artifact(path, payload)
    return {"artifact": artifact, "day": day, "L_place_ms": value_ms,
            "status": result.get("status"), "supersedes": supersedes}


#: DE 152: THE COUNT IS ASSERTED, NOT PRINTED. The battery this replaced
#: ended with `print("... PASS -- 4 checks")` -- a literal beside the
#: cells rather than a count of them, so a cell could be deleted and the
#: line would still say four (rule 10, and R-251's silently-shrinking
#: suite). Every cell below increments; the total is checked at the end.
EXPECTED_CHECKS = 33


def selftest(quiet: bool = False) -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_point_estimate_day] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    with tempfile.TemporaryDirectory(prefix="de_point_estimate_") as tmp:
        root = Path(tmp)
        old = root / artifact_name(
            "2026-09-05", 250.0,
            datetime.datetime(2026, 9, 8, 3, 47, 39,
                              tzinfo=datetime.timezone.utc))
        old_payload = {
            "protocol": PROTOCOL, "day": "2026-09-05",
            "run_mode": "POINT_ESTIMATE",
            "placement_latency": {"L_place_ms": 250.0},
            "day_run": {"placement_latency": {"L_place_ms": 0.0}},
        }
        old.write_text(json.dumps(old_payload))
        prior = prior_artifact(root, "2026-09-05", 250.0)
        ok(prior is not None and prior["sha256"] == _sha(old)
           and prior["target_placement_latency_consistency"]
           == R.PLACEMENT_LATENCY_DISAGREES,
           f"the prior for a (day, L) family resolves by PAIR and carries "
           f"the guard's verdict ON THE TARGET: this one is "
           f"{prior['target_placement_latency_consistency']}, which is the "
           f"shape of every artifact emitted before DE 151")

        emitted = root / artifact_name(
            "2026-09-05", 250.0,
            datetime.datetime(2026, 9, 8, 6, 30,
                              tzinfo=datetime.timezone.utc))
        good = {
            "protocol": PROTOCOL, "day": "2026-09-05",
            "run_mode": "POINT_ESTIMATE",
            "placement_latency": {"L_place_ms": 250.0},
            "day_run": {"placement_latency": {"L_place_ms": 250.0}},
            "supersedes": prior,
            # DE 191: a WRITEABLE payload now carries its population and
            # its scope. This fixture gained them rather than the guard
            # losing its reach -- a required field that a fixture may omit
            # is not required.
            "population_and_coverage": {"n_generations": 1,
                                        "TRANCHE_KEPT": 1,
                                        "coverage": {"coverage": 1.0}},
            "scope": {"coin": "btc", "BTC_ONLY": True},
            "arm_provenance_caveat": {"applies_to": "fixture"},
        }
        wrote = write_artifact(emitted, good)
        ok(wrote["sha256"] == _sha(emitted),
           "POSITIVE CONTROL, AND IT ADMITS: a document whose sites agree is "
           "written atomically and its returned digest is the digest of the "
           "bytes on disk")

        bad = root / "bad.json"
        inconsistent = json.loads(json.dumps(good))
        inconsistent["day_run"]["placement_latency"]["L_place_ms"] = 0.0
        refusal = None
        try:
            write_artifact(bad, inconsistent)
        except R.RunnerRefused as exc:
            refusal = _refusal_name(exc)
        ok(refusal == R.PLACEMENT_LATENCY_DISAGREES and not bad.exists(),
           f"KNOWN-BAD: a document with 250 at the top and 0 nested refuses "
           f"`{refusal}` at the write AND leaves NO FILE -- the emit is "
           f"atomic, so a refused artifact never exists half-written")

        status = R.NULL_NOT_DRAWN_STATUS
        result_fixture = {
            "run_mode": "POINT_ESTIMATE",
            "n_admissible_arms": 1,
            "placement_latency": {"L_place_ms": 250.0},
            "decision_ledger": {"run_mode": "POINT_ESTIMATE"},
            "per_day_sealed_artifacts": [{
                "arm": "A", "status": "OK_POINT_ESTIMATE",
                "economic": {
                    **{field: status for field in NULL_FIELDS},
                    "null_draws_summary": {"n": status},
                },
                "economic_settlement": {
                    "D_E_settle": 12.5,
                    **{field: status for field in NULL_FIELDS},
                    "null_draws_summary": {"n": status},
                },
            }],
        }
        asserted = assert_point_estimate_result(result_fixture)
        missing = json.loads(json.dumps(result_fixture))
        missing["per_day_sealed_artifacts"][0].pop("economic_settlement")
        refusal = None
        try:
            assert_point_estimate_result(missing)
        except R.RunnerRefused as exc:
            refusal = _refusal_name(exc)
        ok(asserted["valued_arms"] == ["A"]
           and refusal == RESULT_CONTRACT_REFUSAL,
           f"THE RESULT CONTRACT, BOTH DIRECTIONS: a day whose arms carry "
           f"ruled settlement P&L and the named no-null status in EVERY "
           f"statistic field is ADMITTED ({asserted['valued_arms']}); one "
           f"whose settlement block is absent refuses `{refusal}`, so a "
           f"diagnostic-only day cannot be published as a point estimate")

        # ---- DE 152: A FULL RUN IS NOT A PREDECESSOR ------------------
        full = root / artifact_name(
            "2026-09-05", 250.0,
            datetime.datetime(2026, 9, 8, 7, 0,
                              tzinfo=datetime.timezone.utc))
        full.write_text(json.dumps(dict(old_payload, run_mode="FULL")))
        refusal = None
        try:
            prior_artifacts(root, "2026-09-05", 250.0)
        except R.RunnerRefused as exc:
            refusal = _refusal_name(exc)
        full.unlink()
        ok(refusal == SUPERSEDES_NOT_A_POINT_ESTIMATE,
           f"DE 152 KNOWN-BAD: a FULL run sitting in this family's name -- "
           f"same protocol, same day, same latency tag -- refuses "
           f"`{refusal}` instead of being superseded by a point estimate. "
           f"`protocol` and `day` cannot tell the two apart (R-828: they "
           f"answer different questions and neither supersedes the other)")

        # ---- DE 152: A PAIR MUST STILL NAME ITS BYTES AT THE WRITE -----
        priors = prior_artifacts(root, "2026-09-05", 250.0)
        held = assert_priors_unchanged(priors, "the battery", 250.0)
        moved = json.loads(json.dumps(priors))
        moved[0]["sha256"] = "0" * 64
        drift = None
        try:
            assert_priors_unchanged(moved, "the battery", 250.0)
        except R.RunnerRefused as exc:
            drift = _refusal_name(exc)
        gone = json.loads(json.dumps(priors))
        gone[0]["path"] = str(root / "vanished.json")
        absent = None
        try:
            assert_priors_unchanged(gone, "the battery", 250.0)
        except R.RunnerRefused as exc:
            absent = _refusal_name(exc)
        ok(held["n_priors_reverified"] == len(priors)
           and held["status"] == "SUPERSEDED_PAIRS_STILL_NAME_THEIR_BYTES"
           and drift == SUPERSEDES_DIGEST_MISMATCH
           and absent == SUPERSEDES_TARGET_ABSENT,
           f"DE 152 THE SUPERSESSION PAIRS ARE RE-VERIFIED AT THE WRITE, "
           f"BOTH DIRECTIONS: {held['n_priors_reverified']} unchanged priors "
           f"ADMIT; a prior whose bytes moved refuses `{drift}`; one that is "
           f"no longer on disk refuses `{absent}`. They are resolved before "
           f"a replay that takes minutes, so a digest computed once and "
           f"trusted later names bytes nobody has")

        # ---- DE 153: A SIBLING IS NOT A SUCCESSOR, ON THE LINK --------
        # The DE 152 merge dropped this guarantee and the drive that found
        # it is reproduced here as the cell. BOTH DIRECTIONS, on real
        # files: a target at THIS run's L is ACCEPTED, and a target at
        # ANOTHER L REFUSES BY NAME -- and the refusal reads the target's
        # OWN declared latency from disk, not the caller's field, which is
        # precisely what a wrong link would carry.
        sib = root / artifact_name(
            "2026-09-05", 0.0,
            datetime.datetime(2026, 9, 8, 8, 0,
                              tzinfo=datetime.timezone.utc))
        sib.write_text(json.dumps({
            "protocol": PROTOCOL, "day": "2026-09-05",
            "run_mode": "POINT_ESTIMATE",
            "placement_latency": {"L_place_ms": 0.0},
            "day_run": {"placement_latency": {"L_place_ms": 0.0}}}))
        # the caller's field SAYS 250 and the file SAYS 0 -- the file wins
        spoofed = [{"path": str(sib), "sha256": _sha(sib),
                    "L_place_ms": 250.0}]
        cross = None
        try:
            assert_priors_unchanged(spoofed, "the battery", 250.0)
        except R.RunnerRefused as exc:
            cross = _refusal_name(exc)
        same = assert_priors_unchanged(
            [{"path": str(sib), "sha256": _sha(sib)}], "the battery", 0.0)
        ok(cross == SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY
           and same["status"] == "SUPERSEDED_PAIRS_STILL_NAME_THEIR_BYTES"
           and same["all_targets_at_this_L_place_ms"] == 0.0
           and same["priors"][0]["declared_L_place_ms"] == 0.0,
           f"DE 153 BOTH WAYS, ON THE PUBLISHED LINK: an L = 0 artifact "
           f"named as the predecessor of an L = 250 run REFUSES `{cross}` "
           f"even though the caller's own field claimed 250 -- the "
           f"target's DECLARED latency is re-read from disk -- while the "
           f"SAME file is ACCEPTED as the prior of an L = 0 run "
           f"({same['priors'][0]['declared_L_place_ms']} ms). Before DE 153 "
           f"this path compared no latency at all and admitted both; "
           f"discovery guarded the FILE, nothing guarded the RELATIONSHIP")
        sib.unlink()

        # ---- DE 152: EVERY REFUSAL THIS MODULE RAISES IS PARSEABLE ----
        # `_refusal_name` splits on the first colon. Four refusals here
        # wrote "REFUSED <NAME> at <where>: ..." and parsed as
        # "<NAME> at <where>" -- a name no reader could match. The merge
        # found it because a cell compared one. The shape is now asserted,
        # not remembered.
        import re as _re152
        _src152 = Path(__file__).read_text()
        _mis152 = _re152.findall(r'REFUSED \{?[A-Za-z_]+\}? at \{where\}:',
                                 _src152)
        ok(not _mis152,
           f"DE 152: every `REFUSED <NAME>` in this module abuts its colon, "
           f"so `_refusal_name` returns the bare name a reader can match "
           f"({len(_mis152)} malformed). Four refusals -- two from each "
           f"implementation -- put `at <where>` BEFORE the colon and parsed "
           f"as `<NAME> at <where>`")

        # ---- DE 152: THE DRIVER BINDS ITSELF TO COMMITTED BYTES -------
        pinned = assert_driver_source("the battery")
        ok(pinned["driver"]["sha256_at_import"] == DRIVER_SHA256_AT_IMPORT
           and pinned["driver"]["unchanged_during_run"] is True
           and pinned["driver"]["producing_code_is_the_committed_bytes"]
           is True,
           f"THE RULE-12 HOLE IS SHUT: this driver refuses to emit unless "
           f"its own bytes are the bytes HEAD holds and are unchanged since "
           f"import (`{DRIVER_SHA256_AT_IMPORT[:16]}`). The two rounds "
           f"before this one were driven by a scratchpad file whose digest "
           f"reached no artifact")

    # ---- DE 181: THE PLACEMENT-LATENCY RECONCILIATION, DRIVEN BOTH WAYS
    # Rule 15: a checker ships a positive control it must flag and a
    # known-bad it must refuse. The two known-bads here are the two SIDES
    # -- REV 144 found BE's first version falsifying on ONE of them, and a
    # wiring that only drove the side that was already checked would
    # inherit that hole one layer out.
    def _refuses(fn, needle, label):
        try:
            fn()
        except (R.RunnerRefused, PLR.ReconcileRefused) as e:
            ok(needle in str(e), f"{label} -- refuses {needle}")
        else:
            ok(False, f"{label} -- DID NOT REFUSE")

    _W181 = {"s1": {"settle_cents": 100.0, "up_won": True}}

    def _ref181(kept, dropped):
        return {"s1": {"BUY_UP": [{"gen": 0, "t0": 0.0, "t1": 9.0,
                                   "level": 0.5, "tranches": list(kept),
                                   PLR.DROPPED_KEY: list(dropped)}],
                       "SELL_UP": []}}

    def _result181(ref, winners, base, *, arms=1, key=None):
        key = key or R.PLR_INPUTS_KEY
        return {"day": "2026-09-04", "per_day_sealed_artifacts": [
            {"arm": f"ARM{i}", "status": "OK_POINT_ESTIMATE",
             # DE 190: THE FIXTURE CARRIES WHAT THE RUNNER WRITES, which
             # is now PENDING -- this dict simulates `run_day`'s output on
             # its way INTO the driver, so seeding it with the driver's
             # own output would have tested the overwrite against itself.
             "placement_latency_reconciliation": {
                 "status": R.PLR_PENDING},
             key: {"reference": ref, "winners": winners,
                   "baseline_total_cents": base, "day": "2026-09-04",
                   "arm": f"ARM{i}"}} for i in range(arms)]}

    _K181 = [{"t": 0.4, "shares": 10.0, "level": 0.60}]
    _D181 = [{"t": 0.05, "shares": 4.0, "level": 0.25}]
    _ref181a = _ref181(_K181, _D181)
    _kept181 = PLR.value(_ref181a, "tranches", _W181)["total_cents"]
    _res181 = _result181(_ref181a, _W181, _kept181)
    _good181 = reconcile_placement_latency(_res181)
    ok(_good181["legs_close"] is True
       and _good181["kept_equals_the_baseline"] is True
       and _good181["KEPT_cross_checked_by_direct_arithmetic"]["agrees"]
       and _good181["DROPPED_cross_checked_by_direct_arithmetic"]["agrees"]
       and _good181["n_arm_days_that_handed_over"] == 1
       and "UPPER BOUND" in _good181["UPPER_BOUND"],
       f"DE 181 POSITIVE CONTROL: the reconciliation RUNS in this driver "
       f"and both sides are anchored -- KEPT against the ledger's "
       f"zero-cancel baseline, DROPPED against a second arithmetic path "
       f"({_good181['DROPPED']['total_cents']:.2f} c), with the "
       f"upper-bound caveat travelling in the block rather than in a "
       f"report")
    ok(R.PLR_INPUTS_KEY not in _res181["per_day_sealed_artifacts"][0]
       and _res181["per_day_sealed_artifacts"][0][
           "placement_latency_reconciliation"]["status"]
       == RECONCILED_ARM_STATUS,
       "DE 181 THE PRIVATE KEY IS POPPED: the live reference object is off "
       "the result before anything can serialise it, and the runner's "
       "placeholder now points to the completed top-level check")

    # KNOWN-BAD, SIDE ONE: a KEPT value that is not the ledger's baseline.
    _refuses(lambda: reconcile_placement_latency(
        _result181(_ref181a, _W181, _kept181 + 1.0)),
        "KEPT_VALUE_DOES_NOT_MATCH_THE_BASELINE",
        "DE 181 KNOWN-BAD (KEPT perturbed)")
    # KNOWN-BAD, SIDE TWO: THE DROPPED SIDE, PERTURBED WHERE ITS ANCHOR
    # ACTUALLY IS. KEPT is anchored by an EXTERNAL number (the ledger's
    # baseline), so a bad input moves it off that number and refuses.
    # DROPPED has no external anchor -- `KEPT + DROPPED == ALL` is a
    # tautology over a partition (DA 167), so ANY input perturbation moves
    # all three totals together and the sum still closes. Its anchor is
    # BE 138's SECOND arithmetic path, which exists only to disagree. So
    # the known-bad for this side is an IMPLEMENTATION disagreement, and
    # driving it with a bad input instead would have reported a pass for
    # the wrong reason -- which is exactly REV 144's finding one layer up.
    _real_legs = PLR.legs_directly

    def _wrong_direct(ref, key, winners):
        got = _real_legs(ref, key, winners)
        if key == PLR.DROPPED_KEY:
            return dict(got, total_cents=got["total_cents"] + 1.0)
        return got

    PLR.legs_directly = _wrong_direct
    try:
        _refuses(lambda: reconcile_placement_latency(
            _result181(_ref181a, _W181, _kept181)),
            "DROPPED", "DE 181 KNOWN-BAD (DROPPED, the side REV 144 found "
                       "unchecked): the two arithmetic paths made to "
                       "disagree by 1 cent")
    finally:
        PLR.legs_directly = _real_legs
    ok(PLR.legs_directly is _real_legs,
       "DE 181 the patched arithmetic path is restored -- a cell owns its "
       "fixture and does not leave one behind")

    # KNOWN-BAD, SIDE THREE: NON-FINITE VALUES -- AND THIS CELL HAS
    # CHANGED ITS ANCHOR BECAUSE THE DEFECT WAS FIXED AT ITS CAUSE.
    # DE 181 found that ONE `shares: nan` reconciled CLEANLY on both
    # sides -- `legs_close`, `kept_equals_the_baseline` and both
    # cross-checks all true -- because a NaN compares False to everything,
    # so every `abs(a - b) > tol` guard was satisfied by it. BE 144 closed
    # it where it belonged, at the checker's own input boundary. The cell
    # is not deleted with the hole: it now asserts the CURRENT truth (the
    # checker refuses by name) AND still drives this driver's own guard
    # directly, because defence in depth that is never driven is not
    # defence.
    _nan181 = _ref181(_K181, [{"t": 0.05, "shares": float("nan"),
                               "level": 0.25}])
    _refuses(lambda: PLR.reconcile(_nan181, _W181, _kept181),
             "TRANCHE_VALUE_NOT_FINITE",
             "DE 181/BE 144 the checker itself REFUSES a non-finite "
             "tranche now -- the hole DE 181 found is closed at its cause")
    _refuses(lambda: reconcile_placement_latency(
        _result181(_nan181, _W181, _kept181)),
        "TRANCHE_VALUE_NOT_FINITE",
        "DE 181 and the refusal PROPAGATES through this driver -- the "
        "artifact is not written")
    _real_reconcile = PLR.reconcile

    def _nonfinite_checker_result(ref, winners, baseline):
        return {"protocol": "FIXTURE", "nested": {"value": float("nan")}}

    PLR.reconcile = _nonfinite_checker_result
    try:
        _refuses(lambda: reconcile_placement_latency(
            _result181(_ref181a, _W181, _kept181)),
            RECONCILE_NON_FINITE,
            "DE 181 THE DRIVER'S OWN FINITENESS SWEEP STILL BITES: a "
            "checker returning a nested NaN is refused before publication")
    finally:
        PLR.reconcile = _real_reconcile

    # AND THE INPUTS THEMSELVES REFUSE RATHER THAN DEFAULT.
    # RESTORED AT DE 186 (2/2): these five cells were deleted by an INDEX
    # SLICE in my own edit -- `s[:start] + new + s[end:]` where the region
    # between two comments held more than I meant. CLAUDE.md says it in
    # one line: "Do not slice source by index to edit it -- anchor to
    # exact strings." The check-count assertion caught it (21 against 25),
    # which is what that assertion is for.
    _refuses(lambda: reconcile_placement_latency(
        {"per_day_sealed_artifacts": [{"arm": "A"}]}),
        RECONCILE_INPUTS_ABSENT, "DE 181 no arm handed the inputs over")
    _refuses(lambda: reconcile_placement_latency(
        _result181(None, _W181, _kept181)),
        RECONCILE_INPUTS_INCOMPLETE, "DE 181 a None input")
    _partial181 = _result181(_ref181a, _W181, _kept181, arms=2)
    _partial181["per_day_sealed_artifacts"][1].pop(R.PLR_INPUTS_KEY)
    _refuses(lambda: reconcile_placement_latency(_partial181),
             RECONCILE_INPUTS_COVERAGE,
             "DE 183 PARTIAL HANDOVER: one of two valued arms is a refusal, "
             "never all-arms agreement")
    _nanbase181 = _result181(_ref181a, _W181, _kept181, arms=2)
    _nanbase181["per_day_sealed_artifacts"][1][R.PLR_INPUTS_KEY][
        "baseline_total_cents"] = float("nan")
    _refuses(lambda: reconcile_placement_latency(_nanbase181),
             RECONCILE_NON_FINITE,
             "DE 183 CROSS-ARM NON-FINITE: a NaN in the second arm refuses "
             "before the tolerance comparison")
    _wrongarm181 = _result181(_ref181a, _W181, _kept181, arms=2)
    _wrongarm181["per_day_sealed_artifacts"][1][R.PLR_INPUTS_KEY]["arm"] = \
        "ARM0"
    _refuses(lambda: reconcile_placement_latency(_wrongarm181),
             RECONCILE_INPUT_IDENTITY,
             "DE 183 ARM IDENTITY: a handover labelled for another arm "
             "cannot certify its container")
    _two181 = _result181(_ref181a, _W181, _kept181, arms=2)
    _two181["per_day_sealed_artifacts"][1][R.PLR_INPUTS_KEY][
        "baseline_total_cents"] = _kept181 + 5.0
    _refuses(lambda: reconcile_placement_latency(_two181),
             RECONCILE_INPUTS_DISAGREE, "DE 181 arms disagreeing on a "
             "day-level input")

    # AND THE EMIT REFUSES IF ANYTHING PRIVATE SURVIVED. `json.dumps` here
    # uses `default=str`, so a live 300 MB reference would NOT crash -- it
    # would be stringified INTO the artifact. That is why this is a
    # predicate and not a comment.
    ok(assert_no_private_keys({"a": {"b": 1}})["n_private_keys_found"] == 0,
       "DE 181 the private-key sweep passes a clean payload")
    _refuses(lambda: assert_no_private_keys(
        {"day_run": {"arms": [{R.PLR_INPUTS_KEY: {"reference": {}}}]}}),
        PRIVATE_KEY_SURVIVED,
        "DE 181 KNOWN-BAD: a surviving private key NESTED two levels down")

    # ---- DE 186: A NO-SPLIT BOOK SAYS SO, AND A PARTIAL ONE REFUSES ---
    # 09-03's EV21 shape. The failure this cell exists for is the quiet
    # one: a reconciliation that "passed" because it had nothing to
    # compare. Anchored -- the checker really does refuse on this book --
    # then the driver's own handling is driven.
    _nosplit186 = {"s1": {"BUY_UP": [{"gen": 0, "t0": 0.0, "t1": 9.0,
                                      "level": 0.5, "tranches": list(_K181)}],
                          "SELL_UP": []}}
    _anchored186 = None
    try:
        PLR.reconcile(_nosplit186, _W181, _kept181)
    except PLR.ReconcileRefused as _e:
        _anchored186 = "DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK" in str(_e)
    _res186 = _result181(_nosplit186, _W181, _kept181)
    _out186 = reconcile_placement_latency(_res186)
    ok(_anchored186 is True
       and _out186["ok"] is False
       and _out186["MUST_BE_SURFACED"] is True
       and _out186["status"] == RECONCILE_NO_SPLIT
       and _res186["per_day_sealed_artifacts"][0][
           "placement_latency_reconciliation"]["status"]
       == RECONCILIATION_UNAVAILABLE_ARM_STATUS
       and "HAS NOT BEEN CHECKED" in _out186["what_is_unavailable"],
       f"DE 186 A BOOK WITH NO SPLIT SAYS SO RATHER THAN RECONCILING "
       f"AGAINST NOTHING: BE's checker refuses "
       f"DROPPED_TRANCHES_ABSENT_FROM_THE_BOOK on 09-03's shape (anchored), "
       f"and this driver turns THAT ONE refusal into a LOUD STATUS -- "
       f"`ok: false`, `MUST_BE_SURFACED: true`, `{_out186['status']}` -- "
       f"because a book built without the complement leg is a fact about "
       f"its CONSTRUCTION, not a disagreement between numbers. It is not "
       f"a pass and no reader can read it as one")
    # AND A PARTIAL SPLIT IS STILL A REFUSAL -- that one IS a defect.
    _partial186 = {"s1": {"BUY_UP": [
        {"gen": 0, "t0": 0.0, "t1": 9.0, "level": 0.5,
         "tranches": list(_K181), PLR.DROPPED_KEY: list(_D181)},
        {"gen": 1, "t0": 0.0, "t1": 9.0, "level": 0.5,
         "tranches": list(_K181)}], "SELL_UP": []}}
    _refuses(lambda: reconcile_placement_latency(
        _result181(_partial186, _W181, _kept181)),
        "DROPPED_TRANCHES_ON_ONLY_SOME_GENERATIONS",
        "DE 186 KNOWN-BAD: a PARTIAL split is not a split")

    # ---- DE 191 / DA 180: THE TWO QUOTATION GAPS -------------------
    # Both were found on a LANDED artifact, so both are driven against a
    # REAL receipt where one exists, and the known-bads are the two ways
    # each can be wrong.
    _rcpt191 = Path(DR.resolve()["data_root"]) / (
        "pm_5min/derived/be_daybook_receipt_20260904_btc__L250ms__EV21.json")
    if not _rcpt191.is_file():
        ok(False, "DE 191 the 09-04 EV21 receipt is missing")
        ok(False, "DE 191 (scope) the 09-04 EV21 receipt is missing")
    else:
        _doc191 = json.loads(_rcpt191.read_text())
        _rec191 = {"n_generations": 358108,
                   "KEPT": {"n_fills": 26379},
                   "DROPPED": {"n_fills": 31471}}
        _pop191 = population_and_coverage(_rec191, _doc191, _rcpt191)
        _nocov = json.loads(json.dumps(_doc191))
        _nocov.pop("assembly_evidence", None)
        _wrongpop = json.loads(json.dumps(_doc191))
        _wrongpop["assembly_evidence"]["UNCOVERED_GENERATIONS"][
            "n_reference_generations"] = 358107
        _c1 = _c2 = None
        try:
            population_and_coverage(_rec191, _nocov, _rcpt191)
        except R.RunnerRefused as _e:
            _c1 = str(_e).split(":")[0].replace("REFUSED ", "")
        try:
            population_and_coverage(_rec191, _wrongpop, _rcpt191)
        except R.RunnerRefused as _e:
            _c2 = str(_e).split(":")[0].replace("REFUSED ", "")
        ok(_pop191["TRANCHE_KEPT"] == 26379
           and _pop191["TRANCHE_BEFORE_PLACEMENT_LATENCY"] == 31471
           and _pop191["coverage"]["n_uncovered"] == 29530
           and abs(_pop191["coverage"]["coverage"] - 0.9175388430305941) < 1e-12
           and _pop191["coverage"]["cross_check_holds"] is True
           and _pop191["BINANCE_GAP_EXCLUDED_STATUS"] ==
               "NOT_APPLIED_ON_THE_DAY_PATH"
           and _pop191["n_gap_bearing_windows"] == 52
           and _c1 == COVERAGE_UNRESOLVABLE
           and _c2 == POPULATION_DISAGREES,
           f"DE 191 (1) THE POPULATION TRAVELS WITH THE NUMBER: on the "
           f"REAL 09-04 receipt the result now carries TRANCHE_KEPT "
           f"{_pop191['TRANCHE_KEPT']}, TRANCHE_BEFORE_PLACEMENT_LATENCY "
           f"{_pop191['TRANCHE_BEFORE_PLACEMENT_LATENCY']}, coverage "
           f"{_pop191['coverage']['coverage']:.7f} with "
           f"{_pop191['coverage']['n_uncovered']} uncovered, and "
           f"BINANCE_GAP_EXCLUDED WITH ITS STATUS -- all of which lived "
           f"only in the book receipt, a different document. A receipt "
           f"with no coverage refuses `{_c1}`; and A RECEIPT WHOSE "
           f"POPULATION DISAGREES WITH THE RECONCILIATION BY ONE "
           f"GENERATION refuses `{_c2}`, which is the cell that makes the "
           f"coverage a cross-check rather than a restatement")
        _bk191 = Path(DR.resolve()["data_root"]) / (
            "pm_5min/derived/be_daybook_20260904_btc__L250ms__EV21.pkl")
        _sc191 = scope_declaration(_bk191, _doc191)
        _c3 = None
        try:
            scope_declaration(
                Path("/x/be_daybook_20260904_eth__L250ms__EV21.pkl"),
                {"coin": "btc"})
        except R.RunnerRefused as _e:
            _c3 = str(_e).split(":")[0].replace("REFUSED ", "")
        ok(_sc191["coin"] == "btc" and _sc191["BTC_ONLY"] is True
           and _sc191["coins_this_result_covers"] == ["btc"]
           and "R-869" in _sc191["why_only_one_coin"]
           and "market-wide" in _sc191["what_must_not_be_said"]
           and _c3 == SCOPE_UNRESOLVABLE,
           f"DE 191 (2) THE SCOPE IS RESOLVABLE FROM THE ARTIFACT: "
           f"`BTC_ONLY: True` with the coin DERIVED from the book "
           f"filename and the receipt agreeing, R-869's reason on it (no "
           f"eth Gate-1 tape exists for ANY day -- a property of the "
           f"DATA), and `what_must_not_be_said` naming the over-read. A "
           f"book and receipt naming DIFFERENT coins refuses `{_c3}` "
           f"rather than picking one")

    # ---- DE 197 / REV 156: THE ARM CAVEAT IS A FIELD, NOT A SENTENCE --
    _cav = arm_provenance_caveat(
        {"book_scoring_code": {"status": R.WAIVED_SCORING_PATH},
         "per_day_sealed_artifacts": [{"arm": "A"}, {"arm": "B"}]},
        {"kept_equals_the_baseline": True})
    _cav_nw = arm_provenance_caveat(
        {"book_scoring_code": {"status": "BOOK_SCORING_CODE_MATCHES"},
         "per_day_sealed_artifacts": [{"arm": "A"}]},
        {"kept_equals_the_baseline": True})
    ok(_cav["the_waiver_claim_is"] == "UNSUPPORTED, NOT REFUTED"
       and _cav["arms"] == ["A", "B"]
       and _cav["scoring_provenance"].startswith("WAIVED")
       and "MAKES NO DECISIONS" in _cav["what_the_reconciliation_does_NOT_cover"]
       and "does NOT validate" not in _cav["what_must_not_be_said"]
       and "validates the arm numbers" in _cav["what_must_not_be_said"]
       and _cav_nw["scoring_provenance"] == "BOOK_SCORING_CODE_MATCHES",
       f"DE 197 THE ARM PROVENANCE CAVEAT IS A FIELD ON THE ARTIFACT: it "
       f"says the waiver is `{_cav['the_waiver_claim_is']}`, names the "
       f"arms it applies to, and states the thing a reader would "
       f"otherwise conclude wrongly -- that the reconciliation's 1e-11 "
       f"agreement validates the arm numbers. It does not: the KEPT side "
       f"IS the zero-cancel baseline, which MAKES NO DECISIONS, so it is "
       f"the half that could not have moved. And it READS the run's own "
       f"scoring status rather than assuming a waiver")

    # AND THE EMIT REFUSES WITHOUT THEM -- a required field is one whose
    # absence stops the bytes, not one a writer is asked to remember.
    _refuses(lambda: write_artifact(
        Path(tempfile.mkdtemp()) / "x.json",
        {"protocol": PROTOCOL, "placement_latency": {"L_place_ms": 250.0}}),
        REQUIRED_QUOTATION_FIELDS_ABSENT,
        "DE 191 KNOWN-BAD: a payload missing the population and the scope")

    # ---- DE 182: THE WAIVER ASK, AND THE DEFAULT THAT MUST BE NO ------
    import de_scoring_path_delta as _SPD182
    ok(waiver_token(False) is None
       and waiver_token(True) == _SPD182.WAIVER_TOKEN,
       "DE 182 THE DEFAULT IS NO: a run that does not ASK hands `None` to "
       "`run_day`, so a book whose scoring code moved still refuses. The "
       "token is READ from the module that defines it, never retyped")
    ok(waiver_scoring_block(True, {"book_scoring_code": {
           "status": "BOOK_SCORING_CODE_MATCHES"}})[
           "granted_by_the_predicate"] is False
       and waiver_scoring_block(False, {"book_scoring_code": {
           "status": R.WAIVED_SCORING_PATH}})[
           "granted_by_the_predicate"] is True,
       "DE 182 ASKING IS NOT GRANTING, AND THE BLOCK READS THE RESULT: it "
       "reports False when the caller asked and the predicate did not "
       "grant, and True from the run's own status regardless of the flag "
       "-- what the predicate DID is the fact, the argument is only what "
       "it was asked to do")

    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_point_estimate_day] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_point_estimate_day] PASS -- {n[0]} checks, "
              f"n_disarmed 0, n_skipped 0")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("day", nargs="?")
    parser.add_argument("book", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--waive-scoring-path", action="store_true",
                        help=("ASK for the scoring-path waiver when the "
                              "book's scoring code does not match the code "
                              "on disk. ASKING IS NOT GRANTING: "
                              "`de_scoring_path_delta.waiver_available` "
                              "decides on five computed conditions and "
                              "refuses BOOK_SCORING_WAIVER_NOT_AVAILABLE "
                              "otherwise. The flag exists so that asking is "
                              "a deliberate act recorded in the launch form "
                              "AND in the artifact"))
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.day or args.book is None:
        parser.error("day and book are required unless --selftest is used")
    output_dir = args.output_dir or (
        Path(DR.resolve()["data_root"]) / "pm_5min" / "derived")
    print(json.dumps(run(args.day, args.book, output_dir,
                         waive_scoring_path=args.waive_scoring_path),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


RECONCILE_INPUTS_ABSENT = "POINT_ESTIMATE_RECONCILIATION_INPUTS_ABSENT"
RECONCILE_INPUTS_INCOMPLETE = "POINT_ESTIMATE_RECONCILIATION_INPUTS_INCOMPLETE"
RECONCILE_INPUTS_DISAGREE = "POINT_ESTIMATE_RECONCILIATION_INPUTS_DISAGREE"
PRIVATE_KEY_SURVIVED = "POINT_ESTIMATE_PRIVATE_KEY_REACHED_THE_PAYLOAD"
RECONCILE_NON_FINITE = "POINT_ESTIMATE_RECONCILIATION_NOT_FINITE"


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
    found = []
    for a in arms:
        got = a.pop(R.PLR_INPUTS_KEY, None)
        if got is not None:
            found.append(got)
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
    for key in ("reference", "winners", "baseline_total_cents"):
        for f in found:
            if f.get(key) is None:
                raise R.RunnerRefused(
                    f"REFUSED {RECONCILE_INPUTS_INCOMPLETE}: an arm-day "
                    f"handed over `{key}` as None. A missing input is a "
                    f"refusal, never a default: BE's checker would value "
                    f"an empty set and report a clean zero.")
    # THE ARMS MUST AGREE ON ALL THREE. They are properties of the DAY,
    # not of an arm -- the same book reference, the same winners, the
    # same zero-cancel baseline (the baseline makes no decisions, so it
    # cannot differ by arm). A disagreement means one arm valued a
    # different population, and reconciling only the first would hide it.
    first = found[0]
    for other in found[1:]:
        if (other["reference"] is not first["reference"]
                or other["winners"] is not first["winners"]
                or abs(float(other["baseline_total_cents"])
                       - float(first["baseline_total_cents"])) > 1e-9):
            raise R.RunnerRefused(
                f"REFUSED {RECONCILE_INPUTS_DISAGREE}: arm "
                f"{other.get('arm')!r} handed over a different reference, "
                f"a different winner map or a different zero-cancel "
                f"baseline than arm {first.get('arm')!r} "
                f"({other['baseline_total_cents']!r} against "
                f"{first['baseline_total_cents']!r}). These are "
                f"properties of the DAY; if they differ, one of the two "
                f"arm-days valued a population the other did not.")
    out = PLR.reconcile(first["reference"], first["winners"],
                        first["baseline_total_cents"])
    # ---- DE 181, FOUND BY DRIVING IT: A NON-FINITE VALUE PASSES EVERY
    # EQUALITY IN THE RECONCILIATION. ------------------------------------
    # `nan` compares False to everything, so `abs(sum - all) > tol` is
    # False, `abs(kept - baseline) > tol` is False, and the two
    # independent arithmetic paths "agree" because `abs(nan - nan) > tol`
    # is False too. A single `shares: nan` in one tranche therefore
    # reconciles CLEANLY on both sides and publishes a nan total. This is
    # REVIEW 138's class one module over -- BE 130 closed it for
    # zero-length generations -- and it is guarded HERE rather than
    # silently tolerated, because this is the surface that PUBLISHES.
    # Reported to BE for the checker itself; the guard stays either way,
    # since a publisher should not depend on its checker's arithmetic
    # being total.
    _nf = [(k, v) for k, v in _finite_scalars(out) if not math.isfinite(v)]
    if _nf:
        raise R.RunnerRefused(
            f"REFUSED {RECONCILE_NON_FINITE}: the reconciliation returned "
            f"non-finite values at {[k for k, _ in _nf]}. Every equality "
            f"in it PASSED, because a NaN compares False to everything -- "
            f"so a clean-looking reconciliation is exactly what a "
            f"non-finite input produces. The artifact is not written.")
    return dict(out,
                run_by="de_point_estimate_day.reconcile_placement_latency",
                inputs_from=("the run itself -- the reference `run_day` "
                             "valued, the winners it used and the "
                             "zero-cancel baseline it computed, handed "
                             "over under a private key and popped here"),
                n_arm_days_that_handed_over=len(found),
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


def run(day: str, book: Path, output_dir: Path) -> dict:
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
EXPECTED_CHECKS = 21


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
        return {"per_day_sealed_artifacts": [
            {"arm": f"ARM{i}", "status": "OK_POINT_ESTIMATE",
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
    ok(R.PLR_INPUTS_KEY not in _res181["per_day_sealed_artifacts"][0],
       "DE 181 THE PRIVATE KEY IS POPPED: the live reference object is off "
       "the result before anything can serialise it")

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

    # KNOWN-BAD, SIDE THREE, AND IT IS A HOLE I FOUND BY DRIVING THIS:
    # a NON-FINITE tranche passes EVERY equality in the reconciliation.
    _nan181 = _ref181(_K181, [{"t": 0.05, "shares": float("nan"),
                               "level": 0.25}])
    _nanout = PLR.reconcile(_nan181, _W181, _kept181)
    ok(_nanout["legs_close"] is True
       and _nanout["kept_equals_the_baseline"] is True
       and _nanout["DROPPED_cross_checked_by_direct_arithmetic"]["agrees"]
       and not math.isfinite(_nanout["DROPPED"]["total_cents"]),
       f"DE 181 ANCHOR FOR THE NEXT CELL -- and it is a finding: ONE "
       f"`shares: nan` reconciles CLEANLY on BOTH sides "
       f"(legs_close, kept==baseline, both cross-checks agree) while "
       f"DROPPED totals {_nanout['DROPPED']['total_cents']!r}. A NaN "
       f"compares False to everything, so every `abs(a - b) > tol` guard "
       f"in the checker is satisfied by it. Reported to BE")
    _refuses(lambda: reconcile_placement_latency(
        _result181(_nan181, _W181, _kept181)),
        RECONCILE_NON_FINITE,
        "DE 181 KNOWN-BAD (non-finite): THIS DRIVER refuses to publish it "
        "even though the checker passed it")
    # AND THE INPUTS THEMSELVES REFUSE RATHER THAN DEFAULT.
    _refuses(lambda: reconcile_placement_latency(
        {"per_day_sealed_artifacts": [{"arm": "A"}]}),
        RECONCILE_INPUTS_ABSENT, "DE 181 no arm handed the inputs over")
    _refuses(lambda: reconcile_placement_latency(
        _result181(None, _W181, _kept181)),
        RECONCILE_INPUTS_INCOMPLETE, "DE 181 a None input")
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
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.day or args.book is None:
        parser.error("day and book are required unless --selftest is used")
    output_dir = args.output_dir or (
        Path(DR.resolve()["data_root"]) / "pm_5min" / "derived")
    print(json.dumps(run(args.day, args.book, output_dir), indent=2,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

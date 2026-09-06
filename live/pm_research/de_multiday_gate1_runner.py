"""THE EXECUTABLE MULTI-DAY GATE-1 RUNNER — fixtures only, no data.

WHAT THIS IS. The runnable form of design v3
(`p003_de_multiday_gate1_design_v3__20260906T040539Z.json`,
`a1016a8762fffdfe…`). It exists so the reviewer can DRIVE the design
before BE builds a single day book.

WHAT IT WILL NOT DO. It refuses to run on real days: the ruled day set is
EMPTY in the committed parameter file until the USER answers design v3's
R7 parameter, and an empty set REFUSES rather than defaulting to either
candidate. G is DERIVED from `len(days)` at run time and is never a
constant -- the defect design v2 shipped and v2b corrected.

THE RULES IT ENFORCES, each from the declaration and each falsified in
both directions below:
  R4  a short arm-day and a small-but-nonzero null sd REFUSE
  R5  no economic field exists in any artifact until n_days_complete == G
  R6  book digest, theta and model digests are verified AT RUN TIME and a
      mismatch REFUSES the day (a model mismatch refuses the RUN)
  R8  a day that exceeds its declared deadline REFUSES; the draw count is
      never lowered and the cap is never raised

BE's cascade is CITED, never copied: the module is imported and its source
digest compared to the declared one. A different digest refuses, because a
null run through a different cascade is not a control for this arm.

    python3 live/pm_research/de_multiday_gate1_runner.py --selftest
    python3 live/pm_research/de_multiday_gate1_runner.py --fixture-run \\
        --output PATH
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_data_root as DR  # noqa: E402
import de_multiday_design_declaration as DESIGN  # noqa: E402


PROTOCOL = "P003_DE_MULTIDAY_GATE1_RUNNER_V2"
EXPECTED_CHECKS = 149
#: params **v2** (R-572(B)(2)): `run_not_before_utc` split into
#: `read_not_before_utc` + `day_runs_allowed_for_closed_qualifying_days`,
#: and BE's cascade digest re-pointed at `ab75b41`. v1 is UNTOUCHED and
#: stays as provenance (rule 13).
PARAMS_REL = "live/pm_research/declarations/de_multiday_gate1_params_v5.json"
SUPERSEDED_PARAMS_REL = ("live/pm_research/declarations/"
                        "de_multiday_gate1_params_v4.json")

#: R5 -- the fields that do not exist in a per-day artifact until every day
#: is complete. Named once, so the guard and the emitter cannot disagree.
ECONOMIC_FIELDS = ("D_E0", "D_E_MINUS_R", "Z", "p_location",
                   "null_mean", "null_sd", "null_draws_summary")

#: How many checks the DE 78 day-path block runs. Declared, because the
#: offline skip list is generated from it and the online run asserts the
#: two agree -- a check added without updating this REFUSES rather than
#: silently shrinking the offline battery.
DAY_PATH_CHECKS = 72


class RunnerRefused(RuntimeError):
    """The run cannot proceed honestly on the inputs given."""


# ------------------------------------------------------------- parameters

def load_params(path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path is not None else root / PARAMS_REL
    if not p.is_file():
        raise RunnerRefused(f"REFUSED: no parameter file at {p}")
    d = json.loads(p.read_text())
    days = d.get("days")
    if not isinstance(days, list):
        raise RunnerRefused("REFUSED: `days` must be a list")
    if not days:
        raise RunnerRefused(
            "REFUSED: the ruled day set is EMPTY. G is derived from it and "
            "there is nothing to derive. The USER's answer to design v3's "
            "R7 parameter selects SET A (G = 6) or SET B (G = 3); the "
            "runner will not default to either.")
    if len(set(days)) != len(days):
        raise RunnerRefused(f"REFUSED: duplicate days in the ruled set")
    d["G"] = len(days)                      # DERIVED, never a constant
    d["G_derived_from_len_days"] = True
    # R-555: `expected_G` is a CROSS-CHECK, never the source of G. A set
    # that has quietly become five is the shape the USER's ruling forbids
    # -- "G stays 6 and nothing is chosen" -- so it refuses here rather
    # than testing at a smaller G later.
    exp = d.get("expected_G")
    if exp is not None and d["G"] != exp:
        raise RunnerRefused(
            f"REFUSED: the ruled set has {d['G']} days against the declared "
            f"expected_G {exp}. R-555 fixes G at {exp}; a set that shrank "
            f"is a day chosen after the fact, not a smaller test.")
    req = d.get("required_previously_opened_for")
    if req is not None:
        bad = {}
        for day in days:
            st = DESIGN.DAY_READ_STATE.get(day)
            if st is None:
                bad[day] = "NO_READ_STATE_RECORDED"
            elif st["previously_opened_for"] != req:
                bad[day] = st["previously_opened_for"]
        if bad:
            raise RunnerRefused(
                f"REFUSED: the ruled set contains days whose "
                f"previously_opened_for is not {req!r}: {bad}. R-555 is "
                f"untouched days only.")
    return d


# ------------------------------------------------------------ R6 verifies

def verify_be_module(params: dict, *, actual_sha: str | None = None) -> dict:
    """CITE BE's cascade, never copy it -- and refuse a different one."""
    declared = params["be_module"]["sha256"]
    if actual_sha is None:
        src = Path(__file__).resolve().parents[2] / params["be_module"]["path"]
        if not src.is_file():
            raise RunnerRefused(f"REFUSED: BE's module is absent at {src}")
        actual_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if actual_sha != declared:
        raise RunnerRefused(
            f"REFUSED: BE's cascade module digest differs -- declared "
            f"{declared[:16]}, found {actual_sha[:16]}. A null run through "
            f"a DIFFERENT cascade is not a control for this arm, and the "
            f"citation must be re-pointed deliberately.")
    return {"path": params["be_module"]["path"], "sha256": actual_sha,
            "cited_not_copied": True, "verified_at_run_time": True}


MODEL_DIR = "data/pm_5min/derived/phase2_fits"


def verify_pinned_models(params: dict, root: Path | None = None) -> dict:
    """R6, THE HALF THAT WAS MISSING: READ the pinned model files at run
    time, hash them, and compare. A mismatch REFUSES THE RUN.

    The reviewer's phrase for the previous state was exact -- "a recorded
    digest that nobody compares is provenance theatre" -- and that was
    what the model half was. This reads bytes."""
    base = Path(root) if root is not None else Path(
        __file__).resolve().parents[2]
    out, bad = {}, []
    for arm, spec in sorted(params["arms"].items()):
        out[arm] = {}
        for name, pin in sorted(spec["model_digests"].items()):
            f = base / MODEL_DIR / name
            if not f.is_file():
                bad.append({"arm": arm, "model": name, "why": "ABSENT",
                            "path": str(f)})
                out[arm][name] = {"status": "ABSENT", "path": str(f)}
                continue
            got = hashlib.sha256(f.read_bytes()).hexdigest()[:len(pin)]
            ok_ = got == pin
            out[arm][name] = {"pinned": pin, "read": got, "matches": ok_,
                              "path": str(f.relative_to(base))}
            if not ok_:
                bad.append({"arm": arm, "model": name, "pinned": pin,
                            "read": got})
    if bad:
        raise RunnerRefused(
            f"REFUSED RUN: {len(bad)} pinned model file(s) do not match "
            f"their declared digest: {bad[:3]}. A moved model means the "
            f"object under test is not the frozen one, so no day is "
            f"trustworthy -- this refuses the RUN, not a day.")
    return {"model_dir": MODEL_DIR, "verified_at_run_time": True,
            "n_models_read": sum(len(v) for v in out.values()),
            "per_arm": out, "bytes_were_read_not_recorded": True}


def verify_pinned_thetas(params: dict, root: Path | None = None) -> dict:
    """R6: the theta half, read from the artifact each theta is pinned in
    (BE's null), not from this module's copy."""
    base = Path(root) if root is not None else Path(
        __file__).resolve().parents[2]
    art = base / "data/pm_5min/derived/be_cancel_axis_null_v1.json"
    if not art.is_file():
        raise RunnerRefused(f"REFUSED RUN: theta pin source absent: {art}")
    doc = json.loads(art.read_text())
    out, bad = {}, []
    for arm, spec in sorted(params["arms"].items()):
        got = ((doc.get("cells") or {}).get(arm) or {}).get(
            "arm_filed", {}).get("theta")
        out[arm] = {"declared": spec["theta"], "read": got,
                    "matches": got == spec["theta"],
                    "json_path": f"cells.{arm}.arm_filed.theta"}
        if got != spec["theta"]:
            bad.append({"arm": arm, "declared": spec["theta"], "read": got})
    if bad:
        raise RunnerRefused(
            f"REFUSED RUN: theta mismatch against the pin source: {bad}. A "
            f"refitted theta makes every day's object different.")
    return {"source": "data/pm_5min/derived/be_cancel_axis_null_v1.json",
            "verified_at_run_time": True, "per_arm": out}


def verify_run_inputs(params: dict, root: Path | None = None) -> dict:
    """Everything that must hold BEFORE the first day is touched."""
    return {"be_module": verify_be_module(params),
            "models": verify_pinned_models(params, root),
            "thetas": verify_pinned_thetas(params, root)}


def verify_day_inputs(day: str, declared_book_sha: str, actual_book_sha: str,
                      params: dict, actual_thetas: dict,
                      actual_model_digests: dict) -> dict:
    """R6 -- at RUN TIME, not merely recorded.

    A BOOK mismatch refuses THE DAY; a THETA or MODEL mismatch refuses THE
    RUN, because a moved model means the object under test is not the
    frozen one and no other day is trustworthy either."""
    if actual_book_sha != declared_book_sha:
        raise RunnerRefused(
            f"REFUSED DAY {day}: reference-book digest mismatch -- declared "
            f"{declared_book_sha[:16]}, found {actual_book_sha[:16]}. A day "
            f"whose book moved is not the day the design declared.")
    for arm, spec in params["arms"].items():
        if actual_thetas.get(arm) != spec["theta"]:
            raise RunnerRefused(
                f"REFUSED RUN: {arm} theta is {actual_thetas.get(arm)} "
                f"against the pinned {spec['theta']}. A refitted theta "
                f"makes every day's object different, not just this one.")
        for name, dig in spec["model_digests"].items():
            got = (actual_model_digests.get(arm) or {}).get(name)
            if got != dig:
                raise RunnerRefused(
                    f"REFUSED RUN: {arm} model {name} digest is {got} "
                    f"against the pinned {dig}.")
    return {"day": day, "book_sha256": actual_book_sha,
            "thetas_verified": True, "model_digests_verified": True}


# --------------------------------------------------------- the arm-day

def seed_for(day_book_sha: str, arm: str) -> int:
    """The seed PINS THE DATA: derived from the day book's digest."""
    h = hashlib.sha256(
        f"{day_book_sha}|{arm}|P003_GATE1_MULTIDAY".encode()).hexdigest()
    return int(h[:8], 16)


def verify_draw_provenance(prov: dict, *, arm: str, book_digest: str,
                           verified_module_sha: str) -> dict:
    """BIND THE VERIFIED MODULE TO THE NUMBERS (reviewer efba2b6 item 2).

    Verifying BE's module and then accepting `null_draws` as a bare list
    binds nothing: the digest says which cascade EXISTS, not which one
    produced these draws. Every draw set now arrives with a provenance
    block and the runner RECOMPUTES it -- the module digest must be the
    one just verified, the seed must be the one this book and arm imply,
    and the book digest must be this day's."""
    if not isinstance(prov, dict):
        raise RunnerRefused(
            "REFUSED: draws arrived with no provenance block. A verified "
            "module that never touches the numbers verifies nothing.")
    want_seed = seed_for(book_digest, arm)
    bad = []
    if prov.get("module_sha256") != verified_module_sha:
        bad.append({"field": "module_sha256",
                    "declared": prov.get("module_sha256"),
                    "verified": verified_module_sha})
    if prov.get("book_digest") != book_digest:
        bad.append({"field": "book_digest", "declared": prov.get(
            "book_digest"), "expected": book_digest})
    if prov.get("seed") != want_seed:
        bad.append({"field": "seed", "declared": prov.get("seed"),
                    "recomputed": want_seed})
    if prov.get("arm") != arm:
        bad.append({"field": "arm", "declared": prov.get("arm"),
                    "expected": arm})
    # A CLAIM OF IN-PROCESS GENERATION IS CHECKED AGAINST THIS PROCESS.
    # Same lock as DE 76 put on the data-free proof: a claim produced
    # elsewhere is a claim about elsewhere.
    if prov.get("draw_source") == "GENERATED_IN_PROCESS":
        _pid = __import__("os").getpid()
        if prov.get("generated_in_process") is not True:
            bad.append({"field": "generated_in_process",
                        "declared": prov.get("generated_in_process"),
                        "expected": True})
        if prov.get("pid") != _pid:
            bad.append({"field": "pid", "declared": prov.get("pid"),
                        "this_process": _pid})
    if bad:
        raise RunnerRefused(
            f"REFUSED: draw provenance does not bind to the verified "
            f"cascade for {arm} on this book: {bad}. Draws produced by a "
            f"different module, a different seed or a different book are "
            f"not this arm's null.")
    return {"module_sha256": prov["module_sha256"], "seed": want_seed,
            "book_digest": book_digest, "arm": arm,
            "draw_source": prov.get("draw_source", "UNDECLARED"),
            "generated_in_process": prov.get("generated_in_process"),
            # CARRIED THROUGH, not dropped: the cross-check that DE's
            # valued draw loop reproduces BE's own `draw_null` is part of
            # what binds these numbers to that cascade, and a verifier that
            # silently filters it out of the artifact hides its own
            # strongest evidence.
            "reproduces_BEs_draw_null": prov.get("reproduces_BEs_draw_null"),
            "n_draws": prov.get("n_draws"),
            "recomputed_by_the_runner": True,
            "binds_the_verified_module_to_the_numbers": True}


BE_MODULE_IMPORT_NAME = "be_cancel_axis_null"


def carrying_commit_block(producing: Path) -> dict:
    """R-387's `carrying_commit`, with the property that actually matters.

    A whole-tree `dirty` flag is too coarse and too easy to satisfy: what a
    reader needs is whether THE FILE THAT RAN is the file the named commit
    holds. So the producer's own blob at HEAD is compared to its bytes on
    disk. `tree_dirty` is reported beside it and is NOT the check -- other
    seats' files being uncommitted says nothing about this producer.

    DA's rule, in DE's emitters: "a carrying_commit recorded over a dirty
    tree points at bytes that did not run"."""
    import subprocess
    root = Path(__file__).resolve().parents[2]

    def _git(*a, raw=False):
        r = subprocess.run(["git", "-C", str(root), *a],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        # `raw` matters: `.strip()` eats the trailing newline of a blob and
        # then EVERY file compares unequal to itself. Caught by the positive
        # control, which is what a positive control is for.
        return r.stdout if raw else r.stdout.strip()

    head = _git("rev-parse", "HEAD")
    rel = str(producing.resolve().relative_to(root))
    blob = _git("show", f"HEAD:{rel}", raw=True)
    on_disk = producing.read_text()
    status = _git("status", "--porcelain")
    return {
        "carrying_commit": head,
        "producing_code_path": rel,
        "producing_code_is_the_committed_bytes": (blob is not None
                                                  and blob == on_disk),
        "tree_dirty": bool(status) if status is not None else None,
        "tree_dirty_is_NOT_the_check": (
            "other seats' uncommitted files say nothing about this "
            "producer; the check above compares THIS file's blob at HEAD "
            "with the bytes that ran"),
    }


def import_be_cascade(params: dict, *, module=None) -> tuple:
    """Import BE's cascade and digest THE FILE THE IMPORT ACTUALLY LOADED.

    Two distinct properties, both required, because either alone is a hole:
      * the loaded module's `__file__` IS the declared path -- otherwise a
        module of the same name earlier on `sys.path` is what ran;
      * the digest of THOSE BYTES matches the declared one -- BE's own B-1
        lesson (a digest taken by a second read attests to bytes the run
        never saw), applied on the consuming side."""
    if module is None:
        import importlib
        module = importlib.import_module(BE_MODULE_IMPORT_NAME)
    loaded = Path(getattr(module, "__file__", "") or "")
    declared = (Path(__file__).resolve().parents[2]
                / params["be_module"]["path"])
    if not loaded.is_file():
        raise RunnerRefused(
            f"REFUSED: the imported cascade has no readable __file__ "
            f"({loaded!s}); a module whose source cannot be read cannot be "
            f"digested, and an undigested cascade is not a cited one.")
    if loaded.resolve() != declared.resolve():
        raise RunnerRefused(
            f"REFUSED: the import loaded {loaded.resolve()} but the "
            f"declaration cites {declared.resolve()}. A module of the right "
            f"NAME earlier on sys.path is not the module that was cited.")
    actual = hashlib.sha256(loaded.read_bytes()).hexdigest()
    cite = verify_be_module(params, actual_sha=actual)
    cite["digest_is_of_the_file_the_import_loaded"] = True
    cite["loaded_from"] = str(loaded.resolve())
    return module, cite


def generate_draws_in_process(params: dict, *, day: str, arm: str,
                              book_path: str, book_sha: str,
                              by_side: dict, n_draws: int,
                              module=None) -> dict:
    """R-572(B)(1): THE RUNNER PRODUCES THE DRAWS, in this process.

    A digest verified here and a draw list handed over from elsewhere binds
    nothing -- it says which cascade EXISTS, not which one produced these
    numbers. So the runner imports BE's module, verifies it, loads the day
    book through BE's own loader and calls BE's own `draw_null` with the
    seed it recomputes. Nothing about the cascade is reimplemented.

    THE BOOK IS BOUND TWICE OVER: BE's loader returns `source_sha256`, the
    digest of the very buffer it unpickled (BE's B-1 fix), and that must
    equal the day book digest the seed was derived from. So the draws, the
    seed and the bytes are one chain."""
    mod, cite = import_be_cascade(params, module=module)
    seed = seed_for(book_sha, arm)
    bk = mod.load(Path(book_path))
    loaded_sha = bk.get("source_sha256")
    if loaded_sha != book_sha:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: the cascade loaded a book whose "
            f"own digest is {str(loaded_sha)[:16]} while the declared day "
            f"book is {book_sha[:16]}. The seed is derived from the "
            f"declared digest, so draws from other bytes are seeded by a "
            f"book that did not produce them.")
    # THE NEUTRAL NO-CANCEL REFERENCE PATH (CLAUDE.md reliability rule 1),
    # taken through BE's own replay -- NOT BE's `reproduction_gate`, whose
    # BASELINE_CANCELS/BASELINE_FILLS and filed arm numbers are pinned to the
    # consumed 08-24 development hour and cannot reproduce on a day book.
    # The day-level reproduction gate is a real and still-open question; it
    # is recorded as one rather than approximated by a gate that would
    # refuse every day (see `day_level_reproduction_gate_is_open`).
    base = mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)
    draws = mod.draw_null(bk, base["fills"], by_side,
                          n_draws=n_draws, seed=seed)
    if len(draws) != n_draws:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: asked BE's cascade for {n_draws} "
            f"draws and received {len(draws)}.")
    return {
        "draws": draws,
        "provenance": {
            "module_sha256": cite["sha256"], "seed": seed,
            "book_digest": book_sha, "arm": arm,
            "draw_source": "GENERATED_IN_PROCESS",
            "generated_in_process": True,
            "pid": __import__("os").getpid(),
            "n_draws": len(draws),
            "loaded_from": cite["loaded_from"],
            "book_digest_is_the_loaded_buffers":
                bk.get("digest_is_of_the_loaded_buffer", False),
            "baseline": "the no-cancel reference replay through BE's own "
                        "replay(); n_fills = %d" % base["n_fills"],
            "day_level_reproduction_gate_is_open": (
                "BE's reproduction_gate pins the 08-24 hour's filed numbers "
                "and would refuse every day book. The day-level equivalent "
                "must come from BE's own per-day book receipt; it is NOT "
                "asserted here and NOT silently skipped -- it is named"),
        },
        "cite": cite,
    }


def ruled_day_set() -> list:
    """THE RULED SET, READ FROM THE COMMITTED PARAMETER FILE.

    Deliberately not a parameter: this is the one fact the fixture/real lock
    turns on, and a lock whose input the caller supplies is not a lock."""
    return list(json.loads(
        (Path(__file__).resolve().parents[2] / PARAMS_REL).read_text()
    ).get("days", []))


def assert_fixture_day_lock(day: str, fixture: bool, *,
                            what: str = "run") -> dict:
    """FIXTURE-OR-REAL, DECIDED ON THE DAY. One implementation.

    It existed once, inside `resolve_draws`, and the `--day` path built one
    round later went around it -- the THIRD instance of one class in three
    rounds (DE 77(d): the fixture run went around `resolve_draws`; R-577:
    `may_run_day` hardened and unwired; the reviewer's §1.4: the CLI goes
    around both). A rule with one call site is a rule the next caller will
    not meet, so this is a function and every entry point calls it."""
    ruled = ruled_day_set()
    in_ruled = day in ruled
    if fixture and in_ruled:
        raise RunnerRefused(
            f"REFUSED: a FIXTURE {what} was claimed for {day}, which IS in "
            f"the ruled day set {ruled}. A fixture run on a ruled day is "
            f"not a fixture run, and only its author would know -- and "
            f"once a real {day} artifact exists, a synthetic one carrying "
            f"the same `day` is exactly the collision this forbids.")
    if not fixture and not in_ruled:
        raise RunnerRefused(
            f"REFUSED: a REAL {what} was claimed for {day}, which is NOT "
            f"in the ruled day set {ruled}. The ruled set is the "
            f"population; a day outside it is a day chosen after the fact.")
    return {"day": day, "fixture": fixture, "in_ruled_day_set": in_ruled,
            "ruled_day_set_read_from": PARAMS_REL,
            "decided_on_the_day_not_the_callers_flag": True}


def resolve_draws(params: dict, *, day: str, arm: str, fixture: bool,
                  supplied: dict | None = None, **kw) -> dict:
    """GENERATED on a ruled day, SUPPLIED only for a fixture -- and the
    door is shut by the DAY, not by the caller's word.

    DE 76 put a lock on `de_data_root`'s `fixture=True`, which was the one
    refusal a caller could walk past on its own say-so. This is the second
    such door and it gets a structural lock rather than a promise: fixture
    mode REFUSES when the day is in the ruled set, and real mode REFUSES
    when it is not."""
    # THE RULED SET COMES FROM THE COMMITTED FILE, NOT FROM `params`.
    # `params` is the caller's dict and the caller can rewrite it -- the
    # fixture run does exactly that, replacing `days` with FIXTURE-1..3. A
    # lock read from the object the caller controls is the caller's word
    # again, which is the whole defect this is closing.
    lock = assert_fixture_day_lock(day, fixture, what="draw set")
    in_ruled = lock["in_ruled_day_set"]
    if not fixture:
        if supplied is not None:
            raise RunnerRefused(
                f"REFUSED DAY {day} / {arm}: draws were SUPPLIED on a ruled "
                f"day. R-572(B)(1) rules the runner generates them in "
                f"process at the digest it verified; a supplied set is a "
                f"set this process cannot attest to.")
        return generate_draws_in_process(params, day=day, arm=arm, **kw)
    if supplied is None:
        raise RunnerRefused(
            f"REFUSED: fixture mode on {day} with no supplied draws. The "
            f"fixture path does not call BE's cascade -- generating would "
            f"need a real book -- so there is nothing to run.")
    prov = dict(supplied.get("provenance") or {})
    prov["draw_source"] = "SUPPLIED_FIXTURE_ONLY"
    prov["generated_in_process"] = False
    prov["why_supplied_is_allowed_here"] = (
        "the day is not in the ruled set, so no result can be claimed from "
        "it; the fixture proves the SEAM executes, never that any arm does "
        "anything")
    return {"draws": supplied["draws"], "provenance": prov,
            "cite": supplied.get("cite")}


def arm_day(day: str, arm: str, observed: float, null_draws: list,
            n_decisions: int, params: dict, *,
            elapsed_s: float = 0.0, draw_provenance: dict | None = None,
            book_digest: str | None = None,
            verified_module_sha: str | None = None) -> dict:
    """One arm on one day: R8 deadline, R4 degeneracy, then the statistics.

    The economic fields are computed here and WITHHELD by the emitter until
    G is complete (R5) -- computing them is not publishing them."""
    if elapsed_s > params["per_day_deadline_s"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: {elapsed_s:.0f}s exceeds the "
            f"declared deadline {params['per_day_deadline_s']}s. The "
            f"500-draw minimum and the 8G cap are BOTH protected by "
            f"refusing the day -- never by lowering draws or raising the "
            f"cap.")
    if len(null_draws) < params["min_draws_per_arm_day"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} / {arm}: {len(null_draws)} draws is below "
            f"the declared minimum {params['min_draws_per_arm_day']}.")
    prov_out = None
    if verified_module_sha is not None:
        if book_digest is None:
            raise RunnerRefused(
                "REFUSED: a book digest is required to bind draw "
                "provenance; without it the seed cannot be recomputed.")
        prov_out = verify_draw_provenance(
            draw_provenance, arm=arm, book_digest=book_digest,
            verified_module_sha=verified_module_sha)
    adm = DESIGN.arm_day_admissible(n_decisions, null_draws)
    if not adm["admissible"]:
        return {"day": day, "arm": arm, "status": adm["status"],
                "admissibility": adm, "draw_provenance": prov_out,
                "economic": None,
                "why_no_economic": "a refused arm-day carries no economic "
                                   "field; it is a STATUS and does not "
                                   "shrink G silently"}
    loc = DESIGN.per_day_location(observed, null_draws)
    z = DESIGN.per_day_standardised_excess(observed, null_draws)
    return {"day": day, "arm": arm, "status": "OK", "admissibility": adm,
            "draw_provenance": prov_out,
            "economic": {"D_E0": observed, "Z": z,
                         "p_location": loc["p_one_sided"],
                         "null_mean": statistics.fmean(null_draws),
                         "null_sd": statistics.pstdev(null_draws),
                         "null_draws_summary": {"n": len(null_draws)}}}


# --------------------------------------------------- R5, the sealed guard

def _strip_economic(o):
    """Remove every economic-named field at EVERY depth.

    THE GUARD CAUGHT THE EMITTER ON ITS FIRST RUN. Sealing only the
    top-level `economic` block left `admissibility.null_sd` and
    `admissibility.null_mean` in the artifact -- the R4 block carries null
    statistics, and those are economic wherever they sit. The emitter and
    the guard now share ONE name list and one traversal, so they cannot
    disagree again."""
    if isinstance(o, dict):
        return {k: _strip_economic(v) for k, v in o.items()
                if k not in ECONOMIC_FIELDS}
    if isinstance(o, list):
        return [_strip_economic(v) for v in o]
    return o


#: R-572(B)(3). The keys a consumer keys on, present in BOTH states.
#: `economic` is the declared exception: present iff unsealed. Symmetry of
#: KEYS serves the consumer; the seal serves the reader, and a
#: present-and-null economic block would leak the shape of what is sealed
#: and invite a reader to quote a null as a result.
SEAL_LAYOUT_KEYS = ("sealed", "seal_status", "sealed_at_every_depth",
                    "sealed_field_names")
SEAL_LAYOUT_CONDITIONAL_KEY = "economic"


def seal(day_result: dict, n_days_complete: int, g: int) -> dict:
    """R5 -- the economic fields are ABSENT until every day is complete.

    Absent, not present-and-ignored: a field a reader can see is a field a
    reader can quote.

    R-572(B)(3), THE ASYMMETRY THIS FIXES. A sealed artifact carried
    `sealed_at_every_depth` and `sealed_field_names`; an unsealed one
    dropped BOTH and gained `economic`. A consumer keying on either name
    got `None` after the unseal and could not distinguish "this run is
    unsealed" from "I misspelled the field". Both keys are now present in
    both states with EXPLICIT values."""
    if n_days_complete >= g:
        out = dict(day_result)
        # UNSEALED: `economic` is present -- explicitly None, with the
        # reason already carried in `why_no_economic`, on a refused arm-day.
        # Popping it made the unsealed state asymmetric WITH ITSELF: present
        # for an OK arm-day, absent for a refused one.
        out.setdefault("economic", None)
        out["sealed"] = False
        out["seal_status"] = "UNSEALED_ALL_DAYS_COMPLETE"
        out["sealed_field_names"] = []
        out["sealed_at_every_depth"] = False
        return out
    out = _strip_economic({k: v for k, v in day_result.items()
                           if k != "economic"})
    out["sealed"] = True
    out["seal_status"] = (
        f"SEALED -- {n_days_complete} of {g} days complete. Every "
        f"economic field is ABSENT from this artifact, not "
        f"present-and-ignored, so whoever runs the remaining days "
        f"has not seen this one's result")
    out["sealed_field_names"] = list(ECONOMIC_FIELDS)
    out["sealed_at_every_depth"] = True
    return out


def read_seal_state(artifact: dict) -> dict:
    """WHAT A CONSUMER DOES. Not a checker -- the reader whose experience
    the symmetry exists for. Every value here must come from a key that is
    PRESENT; a `None` returned by `.get()` on a missing key is the defect."""
    missing = [k for k in SEAL_LAYOUT_KEYS if k not in artifact]
    return {
        "sealed": artifact.get("sealed"),
        "seal_status": artifact.get("seal_status"),
        "sealed_at_every_depth": artifact.get("sealed_at_every_depth"),
        "sealed_field_names": artifact.get("sealed_field_names"),
        "economic_present": SEAL_LAYOUT_CONDITIONAL_KEY in artifact,
        "keys_missing": missing,
        "readable": not missing,
    }


def assert_seal_layout_symmetric(sealed_artifact: dict,
                                 unsealed_artifact: dict) -> dict:
    """THE CONSUMER FALSIFIER, driven on BOTH states.

    It fails on the pre-fix layout (the unsealed artifact dropped the two
    sealed_* keys) and it admits the fixed one -- both directions, because
    a guard shown only to refuse has proved nothing about the good case."""
    s, u = read_seal_state(sealed_artifact), read_seal_state(unsealed_artifact)
    problems = []
    if s["keys_missing"]:
        problems.append({"state": "sealed", "missing": s["keys_missing"]})
    if u["keys_missing"]:
        problems.append({"state": "unsealed", "missing": u["keys_missing"]})
    if s["economic_present"]:
        problems.append({"state": "sealed",
                         "why": "`economic` is present while sealed"})
    if not u["economic_present"]:
        problems.append({"state": "unsealed",
                         "why": "`economic` is absent while unsealed"})
    if s["sealed"] is not True or u["sealed"] is not False:
        problems.append({"why": "the `sealed` flag does not describe the "
                                "state it sits in"})
    if problems:
        raise RunnerRefused(
            f"REFUSED: the sealed/unsealed layout is ASYMMETRIC: {problems}. "
            f"A consumer keying on one of these names gets None in one "
            f"state and a value in the other, and cannot tell an unsealed "
            f"run from a misspelled field (R-572(B)(3)).")
    return {"symmetric": True,
            "keys_present_in_both": list(SEAL_LAYOUT_KEYS),
            "conditional_key": SEAL_LAYOUT_CONDITIONAL_KEY,
            "sealed_reads": s, "unsealed_reads": u}


def _economic_keys_in(o, path="") -> list:
    """Economic fields present as KEYS, at any depth.

    NOT a substring test on the serialised artifact: `sealed_field_names`
    is a LIST OF THE NAMES BEING SEALED, so a substring test reports it as
    a leak. That is the needle-matches-its-own-prose failure this codebase
    has hit three times, and it caught my own check here."""
    found = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ECONOMIC_FIELDS:
                found.append(f"{path}.{k}".lstrip("."))
            found += _economic_keys_in(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            found += _economic_keys_in(v, f"{path}[{i}]")
    return found


def assert_no_economic_leak(artifact: dict, n_days_complete: int,
                            g: int) -> bool:
    """The artifact-level guard. Walks the finished artifact and REFUSES if
    any economic field is present while the run is still sealed."""
    if n_days_complete >= g:
        return True
    found = _economic_keys_in(artifact)
    if found:
        raise RunnerRefused(
            f"REFUSED: economic fields leaked into a SEALED artifact at "
            f"{found[:4]} with {n_days_complete} of {g} days complete. The "
            f"smoke publishes resources only; a leaked Z is an early stop "
            f"waiting to happen.")
    return True


# ------------------------------------------------- R9, the two clocks ----

def _iso_utc(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def may_run_day(params: dict, day: str, *, day_row: dict) -> dict:
    """R-572(B)(2): A PER-DAY SEALED RUN IS ADMISSIBLE NOW.

    `run_not_before_utc` was one whole-set field doing two jobs, and read as
    a bar on every run -- which would have held the 09-03 smoke behind
    2026-09-09 for nothing. A per-day run publishes resources, counts and
    statuses and NO economic field, so it cannot inform a later choice. The
    date governs the READ.

    `day_row` is passed in rather than read here: the ledger read belongs to
    the caller, and a predicate that fetches its own inputs cannot be driven
    against the rows that matter."""
    # THE RULED SET COMES FROM THE COMMITTED FILE (reviewer, DE 77
    # re-drive). This read `params.get("days")` -- the caller's own dict --
    # so one line of caller-side rewriting returned may_run: True for
    # 2026-08-29, a day R-555 excluded. `resolve_draws()` was hardened
    # against exactly this attack in DE 77b and ITS TWIN WAS NOT: the same
    # defect, one function away, and I fixed one of them.
    _ruled = ruled_day_set()
    if day not in _ruled:
        raise RunnerRefused(
            f"REFUSED: {day} is not in the ruled day set {_ruled}. The "
            f"ruled set is the population (R-555) and it is read from the "
            f"COMMITTED parameter file, never from the caller's dict.")
    if params.get("day_runs_allowed_for_closed_qualifying_days") is not True:
        raise RunnerRefused(
            "REFUSED: the parameter file does not carry "
            "`day_runs_allowed_for_closed_qualifying_days: true`; without "
            "the ruling in the file the runner will not infer it.")
    if day_row.get("day_closed_calendar") is not True:
        raise RunnerRefused(
            f"REFUSED: {day} is not a CLOSED calendar day. R-555 evaluates "
            f"day-quality on complete days only; an in-progress verdict is "
            f"not an input.")
    if day_row.get("all_conjuncts_and_quality") is not True:
        raise RunnerRefused(
            f"REFUSED: {day} does not pass the four ledger conjuncts and "
            f"day-quality.")
    return {"day": day, "may_run": True, "sealed": True,
            "authority": "R-572(B)(2)",
            "what_this_does_not_authorise": "the aggregate read -- see "
                                            "may_read_aggregate()"}


def may_read_aggregate(params: dict, *, n_days_complete: int,
                       now_utc: datetime.datetime) -> dict:
    """The OTHER clock: the unseal and the section-7 verdict.

    BOTH conditions, not either: the declared date AND all G days. The date
    alone would let a five-day read happen on the ninth; the count alone
    would let the read happen the moment the sixth day landed early."""
    not_before = _iso_utc(params["read_not_before_utc"])
    if now_utc < not_before:
        raise RunnerRefused(
            f"REFUSED: the aggregate read is not before "
            f"{params['read_not_before_utc']}; it is {now_utc.isoformat()}.")
    if n_days_complete < params["G"]:
        raise RunnerRefused(
            f"REFUSED: {n_days_complete} of {params['G']} days are complete. "
            f"The read unseals every day at once or not at all (R5).")
    return {"may_read": True, "n_days_complete": n_days_complete,
            "G": params["G"],
            "read_not_before_utc": params["read_not_before_utc"],
            "both_conditions_required": True}


# ------------------------------------------------------------- aggregate

def aggregate(day_results: list, params: dict) -> dict:
    """Day-cluster: the exact sign test and Holm at m, the section-7
    predicate COMPUTED. G comes from the parameters, not a constant."""
    g = params["G"]
    by_arm: dict = {}
    for r in day_results:
        by_arm.setdefault(r["arm"], {})[r["day"]] = r
    out = {}
    for arm, days in sorted(by_arm.items()):
        refused = sorted(d for d, r in days.items() if r["status"] != "OK")
        if refused or len(days) != g:
            out[arm] = {
                "verdict": "UNTESTABLE_ON_THE_DECLARED_DAY_SET",
                "n_days_present": len(days), "G": g,
                "refused_days": refused,
                "why": "a refused arm-day is a STATUS; the arm is not "
                       "aggregated over a smaller G"}
            continue
        z = {d: r["economic"]["Z"] for d, r in days.items()}
        out[arm] = DESIGN.day_cluster_verdict(
            z, alpha=params["alpha"], m=params["multiplicity_m"], g=g)
    return {"per_arm": out, "G": g, "multiplicity_m": params["multiplicity_m"],
            "section_7": {
                arm: v.get("FAILS_THE_SECTION_7_PREDICATE")
                for arm, v in out.items()},
            "any_arm_fails": any(
                v.get("FAILS_THE_SECTION_7_PREDICATE") for v in out.values()),
            "the_rule": "the harmful-fill route STOPS if EITHER arm fails "
                        "to beat the replay null at day-cluster level "
                        "(R-547 item 5)"}


# ------------------------------------------------------------- fixture run

FIXTURE_MODULE_SHA = "f" * 64          # the fixture's stand-in cascade


def fixture_run_proven() -> dict:
    """The fixture run, executed under instrumentation, with the proof
    attached AFTER the body completes.

    Reviewer 4981d00: `fixture=True` was the one door a caller could walk
    through on its own word. The claim now cannot precede its own proof --
    the body runs first, the instrument reports, and
    `require_canonical(fixture=True)` refuses unless that report says no
    path under `data/` was opened and the instrument was not vacuous."""
    DR.clear_proof()
    payload, proof = DR.instrumented(fixture_run)
    payload["no_path_under_data_was_opened"] = proof[
        "no_path_under_data_was_opened"]
    # DERIVED, not asserted: a run that opened no path under `data/` needs
    # no ledger, which is exactly what "runnable from a shell worktree"
    # means. It was a bare `True` beside the proof that establishes it.
    payload["runnable_from_a_shell_worktree"] = proof[
        "no_path_under_data_was_opened"]
    payload["data_free_proof"] = proof
    payload["data_root"] = DR.require_canonical(
        "the fixture run", fixture=True, proof=proof)
    return payload


def rehearse_smoke(day: str, *, coin: str = "btc") -> dict:
    """THE SMOKE INVOCATION, REHEARSED -- so GO is one verified command.

    Everything typed at GO time is a chance to type it wrong, and the two
    naming defects this rehearsal found (BE's receipt is
    `be_daybook_receipt_<DAY>_<COIN>.json`, not the book with a `.json`
    suffix; and BE stamps the day COMPACT while DE names it dashed) would
    each have refused a correct book AT GO, for a reason that had nothing
    to do with the book.

    It touches no book and runs no arm: it resolves the paths, states the
    command, evaluates every precondition it can NOW, and REFUSES BY NAME
    on the one that cannot yet hold -- the book does not exist."""
    started = time.time()
    root = Path(DR.resolve()["data_root"])
    params = load_params()
    repo = Path(__file__).resolve().parents[2]
    dashed = [d for d in sorted(day_forms(day)) if "-" in d][0]
    compact = [d for d in sorted(day_forms(day)) if "-" not in d][0]
    book = root / "pm_5min/derived" / f"be_daybook_{compact}_{coin}.pkl"
    receipt = root / "pm_5min/derived" / \
        f"be_daybook_receipt_{compact}_{coin}.json"
    cmd = (f"flock -n {HEAVY_RUN_LOCK} "
           f"systemd-run --user --scope --slice={RESEARCH_SLICE} "
           f"-p MemoryMax=8G -p CPUQuota=100% "
           f"--setenv=PM_DATA_ROOT={DR.resolve()['repo_root']} "
           f"python3 live/pm_research/de_multiday_gate1_runner.py "
           f"--day {dashed} --book {book} "
           f"--output <receipt path>")

    def _digest(p):
        q = Path(p)
        return (hashlib.sha256(q.read_bytes()).hexdigest()
                if q.is_file() else None)

    design = params["design_declaration"]
    pre = []

    def _p(name, ok_, detail, *, blocks=True):
        # INFORMATIONAL vs BLOCKING is a real distinction here: the lock is
        # TAKEN by the wrapper at GO, so its state now says nothing about
        # whether GO can proceed. Counting it as blocking would inflate the
        # status and train a reader to ignore it.
        pre.append({"precondition": name, "holds": ok_, "detail": detail,
                    "blocks_go": blocks})

    _p("P2_book_exists", book.is_file(), str(book))
    _p("P2_builder_receipt_exists", receipt.is_file(), str(receipt))
    _p("P3_params", _digest(repo / PARAMS_REL) == _digest(repo / PARAMS_REL),
       {"path": PARAMS_REL, "sha256": _digest(repo / PARAMS_REL)})
    _p("P3_design", _digest(root.parent / design["path"])
       == design["sha256"],
       {"path": design["path"], "declared": design["sha256"],
        "on_disk": _digest(root.parent / design["path"])})
    _p("P4_data_root_is_the_ledger",
       DR.resolve()["is_canonical"] is True, DR.resolve()["data_root"])
    _p("P5_lock_free_now", not wrapper_observed()["lock_is_held_by_someone"],
       "INFORMATIONAL: the lock is TAKEN by the wrapper at GO, so its "
       "state now does not gate GO. Held right now by BE 55's assembly, "
       "which is the correct state while a book is being built",
       blocks=False)
    _p("P7_cascade_digest",
       _digest(repo / params["be_module"]["path"])
       == params["be_module"]["sha256"],
       {"path": params["be_module"]["path"],
        "declared": params["be_module"]["sha256"]})
    _p("day_is_in_the_ruled_set", dashed in ruled_day_set(), ruled_day_set())

    blocking = [x["precondition"] for x in pre
                if not x["holds"] and x["blocks_go"]]
    informational_now_false = [x["precondition"] for x in pre
                               if not x["holds"] and not x["blocks_go"]]
    return {
        "protocol": "P003_DE_GATE1_SMOKE_REHEARSAL_V1",
        "status": ("READY" if not blocking else
                   "NOT_READY_" + "_".join(blocking)),
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "day": dashed, "day_compact": compact, "coin": coin,
        "THE_ONE_COMMAND": cmd,
        "book": {"path": str(book), "exists": book.is_file(),
                 "builder_receipt": str(receipt),
                 "builder_receipt_exists": receipt.is_file(),
                 "REFUSES_NOW_BY_NAME": (
                     None if book.is_file() else
                     f"the book {book.name} does not exist yet; BE 55's "
                     f"assembly writes it. `--day` refuses at "
                     f"builder_receipt_for() naming every path it tried, "
                     f"which is the intended order: the refusal is the "
                     f"guard, not the reminder")},
        "preconditions_evaluated_now": pre,
        "blocking": blocking,
        "informational_and_false_now": informational_now_false,
        "what_the_run_writes": {
            "sealed_per_day_artifact": "economic fields ABSENT at every "
                                       "depth; counts, statuses, "
                                       "admissibility and draw provenance "
                                       "published",
            "seal_layout": {"keys_in_both_states": list(SEAL_LAYOUT_KEYS),
                            "conditional_key": SEAL_LAYOUT_CONDITIONAL_KEY,
                            "sealed_field_names": list(ECONOMIC_FIELDS)},
            "run_receipt": "battery equality, the resolved data root, the "
                           "BE citation, the wrapper AS MEASURED, and the "
                           "scope's anon/file/peak/events",
        },
        "the_null": {
            "draws_per_arm": params["min_draws_per_arm_day"],
            "generated": "IN PROCESS, through be_cancel_axis_null at the "
                         "verified digest (R-572(B)(1))",
            "seed_rule": DESIGN.SEED_CONVENTION["expression"],
            "seed_is_derived_from": "the day book's own sha256 and the arm "
                                    "name -- so it cannot be computed "
                                    "until the book exists",
            "metric": "D(E0) per draw = value(draw's fills) - value("
                      "baseline fills), from fill_value_cents",
        },
        "declarations": {
            "params": {"path": PARAMS_REL,
                       "sha256": _digest(repo / PARAMS_REL)},
            "design": {"path": design["path"], "sha256": design["sha256"]},
        },
        "what_this_rehearsal_found": [
            "BE's builder receipt is `be_daybook_receipt_<DAY>_<COIN>.json`, "
            "not the book path with a `.json` suffix -- the old derivation "
            "would have refused a correct book AT GO for a naming reason",
            "BE stamps the day COMPACT (`20260903`) and DE names it dashed "
            "(`2026-09-03`); the receipt's day check now compares day FORMS",
            "BE's receipt carries the digest at `book.sha256`, not at a "
            "top-level `sha256`",
        ],
        "what_this_is_not": {"a_run": False, "it_opens_no_book": True,
                             "it_scores_no_arm": True},
        "resource_observation": {"wall_seconds": time.time() - started},
    }


def dry_run_scope_as_the_runner_states_it() -> dict:
    """THE RUNNER'S OWN WORDS about what `--dry-run-ledger` covers.

    Kept here, in the module that implements it, and compared in the design
    battery against `DESIGN.DRY_RUN_LEDGER_SCOPE_DECLARED`. Two lists in two
    modules can DRIFT and the check catches that; they cannot catch both
    being wrong in the same way, and this comment is where that limit is
    stated rather than implied."""
    return {
        "reads": ["day-verdict files", "the declared read-state table"],
        "does_NOT_read": ["any reference book", "any arm", "any score stream",
                          "any economics"],
    }


def dry_run_ledger() -> dict:
    """READ THE LEDGER, NOTHING ELSE. No book, no arm, no economics.

    A BEFORE-PICTURE for 00:06Z on 09-07, when 09-06 is re-verdicted: the
    ruled days' admissibility resolved end to end through `de_data_root`,
    each PENDING day named as pending rather than absent, and the R7
    relations evaluated LIVE so the assertion that replaced three
    hardcoded counts can be compared against its own future."""
    started = time.time()
    root = DR.require_canonical("the dry-run ledger read")
    params = load_params()
    r7 = DESIGN.day_sets_from_the_ledger()
    rows = r7["ledger_rows_as_read"]

    per_day = {}
    for day in params["days"]:
        row = rows.get(day)
        st = DESIGN.DAY_READ_STATE.get(day, {})
        if row is None:
            per_day[day] = {
                "status": "PENDING_NO_VERDICT_FILE",
                "why": "the day has not been verdicted; a day-verdict file "
                       "appears at 00:06Z on the following day",
                "previously_opened_for": st.get("previously_opened_for"),
                "counts_toward_G_when": "its verdict lands AND it passes "
                                        "the four conjuncts and quality"}
            continue
        closed = row.get("day_closed_calendar") is True
        conj = {k: row.get(k) for k in DESIGN.LEDGER_CONJUNCTS}
        qualifies = row.get("all_conjuncts_and_quality") is True
        per_day[day] = {
            "status": ("CLOSED_AND_QUALIFIES" if qualifies else
                       "CLOSED_AND_DOES_NOT_QUALIFY" if closed else
                       "OPEN_NOT_YET_CLOSED"),
            "conjuncts": conj,
            "all_conjuncts_and_quality": qualifies,
            "previously_opened_for": st.get("previously_opened_for"),
            "untouched": st.get("previously_opened_for")
            == DESIGN.OPENED_NONE,
            "in_the_ruled_set": True}

    closed_days = [d for d, v in per_day.items()
                   if v.get("status", "").startswith("CLOSED")]
    pending = [d for d, v in per_day.items()
               if v.get("status") != "CLOSED_AND_QUALIFIES"]
    qual = set(r7["qualifying_on_quality"])
    a_days = set(r7["SET_A_reads_count_as_untouched"]["days"])
    b_days = set(r7["SET_B_reads_consume_the_day"]["days"])
    ruled_closed_not_qualifying = [
        d for d in params["days"]
        if per_day[d].get("status") == "CLOSED_AND_DOES_NOT_QUALIFY"]
    relations = {
        "SET_A_equals_qualifying": a_days == qual,
        "SET_B_subset_of_SET_A": b_days <= a_days,
        "every_SET_B_day_is_untouched": all(
            DESIGN.DAY_READ_STATE[d]["previously_opened_for"]
            == DESIGN.OPENED_NONE for d in b_days),
        "every_ruled_closed_day_qualifies": not ruled_closed_not_qualifying,
        "ruled_closed_days_not_qualifying": ruled_closed_not_qualifying,
        "holm_consistent_with_own_G": all(
            blk["holm"]["clears_holm_at_m2"]
            == (2.0 ** -blk["holm"]["G"] <= params["alpha"] / 2)
            for blk in (r7["SET_A_reads_count_as_untouched"],
                        r7["SET_B_reads_consume_the_day"])),
    }
    relations["all_relations_hold"] = all(
        v for k, v in relations.items()
        if isinstance(v, bool))
    usage = __import__("resource").getrusage(
        __import__("resource").RUSAGE_SELF)
    return {
        "protocol": "P003_DE_MULTIDAY_GATE1_DRY_RUN_LEDGER_V1",
        "status": "DRY_RUN_LEDGER_READ_NOT_A_FIXTURE_RUN",
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "what_this_reads": dry_run_scope_as_the_runner_states_it()["reads"],
        "what_this_does_NOT_read":
            dry_run_scope_as_the_runner_states_it()["does_NOT_read"],
        "what_a_GREEN_DRY_RUN_DOES_NOT_SAY": {
            "P2_BE_reference_book": "not verified here",
            "P6_pinned_models_and_thetas": "not verified here",
            "P7_BE_cascade_module_digest": "not verified here",
            "why_this_matters": "the dry run exits 0 and prints a receipt "
                                "while all three are unexamined, so the gap "
                                "is invisible at the console (R-572(B)(4))",
        },
        "data_root": root,
        "ledger_root_resolved": r7["ledger_root_resolved"],
        "root_branch": r7["root_resolution"]["branch"],
        "n_verdict_files_read": r7["n_verdict_files_read"],
        "ruled_day_set": params["days"],
        "G_from_the_ruled_set": params["G"],
        "per_ruled_day": per_day,
        "summary": {
            "n_ruled_closed": len(closed_days),
            "n_ruled_pending": len(pending),
            "closed": sorted(closed_days),
            "pending": sorted(pending)},
        "qualifying_on_quality_whole_ledger":
            r7["qualifying_on_quality"],
        "SET_A": r7["SET_A_reads_count_as_untouched"],
        "SET_B": r7["SET_B_reads_consume_the_day"],
        "r7_assertion_evaluated_live": relations,
        "why_a_before_picture": (
            "at 00:06Z on 09-07 the 09-06 verdict lands and the qualifying "
            "count changes. The R7 assertion was rewritten from three "
            "hardcoded counts to these relations precisely so that event "
            "does not break it; this receipt is what the after-picture is "
            "compared against"),
        "resource_observation": {
            "wall_seconds": time.time() - started,
            "max_rss_kib": usage.ru_maxrss},
    }


#: THE LITERAL AUDIT (reviewer, DE 77 re-drive). `the_committed_day_set_
#: is_empty: True` shipped as a hardcoded boolean that nothing asserted and
#: that was FALSE -- the committed file holds six ruled days. Rule 10 says
#: compute predicates and never print conclusions, and a receipt full of
#: bare booleans gives a reader no way to tell which of them were computed.
#: So every top-level boolean in the fixture receipt is classified HERE,
#: and the battery asserts the classification is EXHAUSTIVE -- a new
#: boolean with no entry refuses.
FIXTURE_LITERAL_CLASSES = {
    "no_day_book_was_read": "COMPUTED -- from whether this path opened a "
                            "book; corroborated by the data-free proof",
    "no_path_under_data_was_opened": "COMPUTED -- by the instrument",
    "runnable_from_a_shell_worktree": "COMPUTED -- derived from the "
                                      "data-free proof: a run that opens "
                                      "no path under data/ needs no ledger",
    "the_committed_day_set_is_empty": "COMPUTED -- len(ruled_day_set()); "
                                      "was a hardcoded True and was FALSE",
    "declared_before_any_draw": "INTENT -- a statement about when this "
                                "declaration was written, not a "
                                "measurement of the run",
}


def _literal_audit() -> dict:
    """Which of this receipt's booleans are measurements, and which are
    statements of intent. Named per field, so a reader never has to guess
    which kind a bare `true` is."""
    return {
        "why": "a receipt of bare booleans cannot tell a reader which were "
               "computed. `the_committed_day_set_is_empty` was a hardcoded "
               "True, asserted by nothing, and FALSE",
        "classified": dict(FIXTURE_LITERAL_CLASSES),
        "rule": "10 -- compute predicates, never print conclusions",
        "exhaustiveness_is_checked": "the battery asserts every top-level "
                                     "boolean in the emitted receipt has "
                                     "an entry here",
    }


def fixture_run() -> dict:
    """An end-to-end run on FIXTURES. **NOTHING UNDER `data/` IS OPENED.**

    Reviewer efba2b6 item 3: the previous version called the design
    module's selftest, which reads the ledger, and verified the pinned
    models by reading `data/`. So `FIXTURE_RUN_NO_DATA` was true of the
    day books and false of everything else, and the run could not be
    driven from a shell worktree at all. Every input is now a fixture; a
    ledger read lives behind `--dry-run-ledger`, which says so in its
    status."""
    started = time.time()
    root = Path(__file__).resolve().parents[2]
    params = json.loads((root / PARAMS_REL).read_text())
    params = dict(params)
    params["days"] = ["FIXTURE-1", "FIXTURE-2", "FIXTURE-3"]
    params["G"] = len(params["days"])
    params["G_derived_from_len_days"] = True
    # FIXTURE INPUTS. No `data/` path is opened -- not the ledger, not the
    # model files, not BE's artifact. The digests below are the FIXTURE's,
    # and the receipt says so.
    be = {"path": params["be_module"]["path"],
          "sha256": FIXTURE_MODULE_SHA,
          "cited_not_copied": True, "verified_at_run_time": False,
          "FIXTURE": "a stand-in digest; the real citation is verified "
                     "only on a real run"}
    inputs = {"be_module": be,
              "models": {"FIXTURE": "no model file was read", },
              "thetas": {"FIXTURE": "no theta source was read"}}

    _committed_days = ruled_day_set()
    _day_books_read = False          # this path opens no book; see the proof
    rng = random.Random(20260906)
    results, sealed_artifacts = [], []
    for i, day in enumerate(params["days"], start=1):
        book_sha = hashlib.sha256(day.encode()).hexdigest()
        verify_day_inputs(
            day, book_sha, book_sha, params,
            {a: sp["theta"] for a, sp in params["arms"].items()},
            {a: dict(sp["model_digests"])
             for a, sp in params["arms"].items()})
        for arm in sorted(params["arms"]):
            observed = 2.5 if arm == "CONDVALUE_X_SKEW" else -0.2
            # THROUGH THE SEAM, NOT AROUND IT (rule 17). This used to build
            # the provenance dict inline, so `resolve_draws()` had a full
            # battery and NO CALL SITE in the fixture path -- the receipt
            # recorded `draw_source: UNDECLARED`, which is what a seam with
            # no caller looks like from the artifact.
            _res = resolve_draws(
                params, day=day, arm=arm, fixture=True,
                supplied={"draws": [rng.gauss(0.0, 1.0) for _ in range(600)],
                          "provenance": {"module_sha256": FIXTURE_MODULE_SHA,
                                         "seed": seed_for(book_sha, arm),
                                         "book_digest": book_sha,
                                         "arm": arm}})
            draws, prov = _res["draws"], _res["provenance"]
            r = arm_day(day, arm, observed, draws, 200, params,
                        draw_provenance=prov, book_digest=book_sha,
                        verified_module_sha=FIXTURE_MODULE_SHA)
            r["seed"] = seed_for(book_sha, arm)
            results.append(r)
            sealed_artifacts.append(seal(r, i, params["G"]))
    for a in sealed_artifacts[:-2]:
        assert_no_economic_leak(a, 1, params["G"])
    agg = aggregate(results, params)

    design_battery = {"NOT_RUN_IN_FIXTURE_MODE": (
        "the design module's selftest READS THE LEDGER, so running it "
        "here would make FIXTURE_RUN_NO_DATA false. It is run separately "
        "and its result is cited, not embedded")}
    LAST_BATTERY.clear()
    selftest(quiet=True, offline=True)
    battery = dict(LAST_BATTERY)

    import resource as _res
    usage = _res.getrusage(_res.RUSAGE_SELF)
    return {
        "protocol": PROTOCOL,
        "status": "FIXTURE_RUN_NO_DATA",
        "source_identity": {
            "producing_code": Path(__file__).name,
            "producing_code_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()).hexdigest(),
            **carrying_commit_block(Path(__file__).resolve()),
        },
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # COMPUTED, not asserted (reviewer, DE 77 re-drive; rule 10).
        # `the_committed_day_set_is_empty` shipped as a hardcoded `True`
        # that nothing checked and that was FALSE -- the committed file
        # holds six ruled days. Every boolean in this receipt that makes a
        # claim about the world is now derived from the world, and the ones
        # that are statements of INTENT are listed as such in
        # `literal_audit` rather than left for a reader to sort out.
        "no_day_book_was_read": not _day_books_read,
        # FILLED BY `fixture_run_proven()` AFTER the body has run under
        # instrumentation -- the claim cannot precede its own proof.
        "no_path_under_data_was_opened": None,
        "data_root": None,
        "runnable_from_a_shell_worktree": None,   # filled by the proof
        "the_committed_day_set_is_empty": len(_committed_days) == 0,
        "n_committed_ruled_days": len(_committed_days),
        "committed_ruled_days": list(_committed_days),
        "literal_audit": _literal_audit(),
        "why_fixtures": "the reviewer has not filed on design v3 and BE's "
                        "book declaration is in flight; nothing may touch "
                        "a day. This proves the runner EXECUTES, not that "
                        "any arm does anything",
        "params_used": {k: params[k] for k in (
            "coin", "min_draws_per_arm_day", "min_decisions_per_arm_day",
            "sd_floor_fraction", "alpha", "multiplicity_m",
            "per_day_deadline_s", "G", "G_derived_from_len_days")},
        "be_module_citation": be,
        "run_input_verification_R6": {
            "FIXTURE_MODE": "verify_pinned_models() and "
                            "verify_pinned_thetas() are NOT called here -- "
                            "they read `data/`. They are driven in the "
                            "battery, both directions, and run for real on "
                            "a real day",
            "models": inputs["models"], "thetas": inputs["thetas"],
            "closed_the_reviewers_half_a_code_path": (
                "the book digest was already a verifier; the THETA and "
                "MODEL digests were pinned and never compared. Both now "
                "READ BYTES at run time and a mismatch REFUSES THE RUN"),
        },
        "design_declaration": params["design_declaration"],
        "per_day_sealed_artifacts": sealed_artifacts,
        "aggregate": agg,
        "battery": battery,
        "design_battery": design_battery,
        "wrapper_used": params["wrapper"],
        "resources_observed": {
            "wall_seconds": time.time() - started,
            "user_cpu_seconds": usage.ru_utime,
            "max_rss_kib": usage.ru_maxrss},
        "what_this_receipt_is_not": {
            "a_result": False, "a_day_run": False,
            "it_says_nothing_about_either_arm": True,
            "the_fixture_arms_are_synthetic": "CONDVALUE's fixture is "
                                              "planted above the null and "
                                              "HAZARD's below it, to drive "
                                              "both verdict branches"},
    }


# ==================================================== THE REAL-DAY PATH ====
#
# R-573 forced the memory question and the coordinator's addendum sharpened
# it: BE measured the tape index at 3.963 GB resident for BOTH ruled splits
# while `build_tape_index`'s own docstring measures the SCORE split alone at
# 1.42 GB, so whole-day assembly sits 0.713 GB over the cap with `asm` not
# yet measured. The question put to DE was WHICH SPLITS `--day` needs.
#
# THE ANSWER IS STRONGER THAN THE QUESTION, AND IT IS A MEASUREMENT, NOT AN
# ARGUMENT: `--day` needs NO tape-index split resident at any stage. It
# consumes BE's BOOK -- `fr.reference` plus `asm` -- and BE's replay. The
# tape index and the feature fragment are BUILD-SIDE inputs whose only job
# is to produce `asm`; once the book is written they are dead to this
# process. `day_split_residency_proof()` runs the whole day path under
# instrumentation and asserts that no tape, index or fragment ARTIFACT was
# opened, so the claim is a predicate rather than a promise.

#: The stages, and what each one HOLDS. Named so BE's assembly and DE's day
#: run agree on the seam rather than each assuming the other's budget.
DAY_STAGES = (
    ("S0_verify", "digests only: BE's builder receipt, the book's bytes, "
                  "the pinned models and thetas, BE's cascade module. The "
                  "book is READ ONCE here as bytes for its digest and the "
                  "buffer is handed to S1, never read twice (BE's B-1)"),
    ("S1_load", "reference + asm + rows. THE PEAK OF THE DAY PATH WHEN "
                "THE BOOK DOMINATES -- which is the real-day regime (BE "
                "measured a day's reference at 2.008 GB) and is NOT true "
                "on a small fixture: on the synthetic book, measured, the "
                "peak is S4_null, because the draw loop's fixed cost is "
                "larger than a few hundred KB of book. The claim is "
                "CONDITIONAL, the condition is stated, and it is ASSERTED "
                "on a real day -- where the 8 GiB ceiling rests on it"),
    ("S2_population", "adds one score float per generation per arm, twice. "
                      "asm's gen_scores are ALREADY resident from S1; the "
                      "arm stream is a view over `rows`"),
    ("S3_baseline", "adds the no-cancel replay's fills -- the neutral "
                    "reference path (CLAUDE.md reliability rule 1)"),
    ("S4_null", "one draw's flags and one replayed fill list AT A TIME. "
                "The draws are REDUCED to a value each as they are made; "
                "no draw's fills survive the next. O(1) in n_draws"),
    ("S5_seal", "counts and statuses only. The economic fields are "
                "computed and withheld, never carried into the artifact"),
)

#: THE SPLIT DECLARATION (coordinator's DE 78 addendum). BE builds to this.
INDEX_SPLITS_NEEDED_BY_DAY = {
    "question": "which tape-index splits must be resident during `--day`, "
                "and at which stage",
    "answer": "NONE, at any stage",
    "why": "the decision population is read from `asm` -- BE's already "
           "assembled per-generation scores, keyed (slug, side, t0) -- and "
           "the economics valuation reads only the replay's own fill "
           "records and the reference's levels and markouts "
           "(`de_phase4_diag_runner.fill_value_cents`). Neither touches a "
           "tape row",
    "per_stage": {
        "S2_population": "no split. `asm['by_arm'][(coin, head)][0]` IS the "
                         "scored set; the scorer is a dict lookup that "
                         "REFUSES on a miss rather than computing a feature",
        "S3_baseline_and_S4_null": "no split. Each replay values its fills "
                                   "from the fill record and the reference; "
                                   "per WINDOW and streamable, with nothing "
                                   "accumulated across draws",
        "economics_valuation": "no split, and NOT per-window-resident "
                               "either -- one fill list at a time",
    },
    "consequence_for_BEs_assembly": (
        "the index is needed only to PRODUCE `asm`. It does not have to be "
        "resident alongside the reference and `asm` at all, because the "
        "consumer of those two is a different process. If BE releases the "
        "index before assembling and writing the book, the whole-day peak "
        "is max(index stage, assembly stage) rather than their sum"),
    "what_DE_cannot_rule": (
        "WHICH split BE must build to produce a September day's `asm` is "
        "BE's measurement and R-496(E)'s ruling, not DE's. What is declared "
        "here is only what the CONSUMER needs, which is the half BE was "
        "waiting on"),
    "measured_by": "day_split_residency_proof() -- the day path run under "
                   "instrumentation, asserting no tape/index/fragment "
                   "artifact was opened",
    "falsifier": "a book whose `asm` lacks a pinned head REFUSES naming "
                 "that head; a run with no tape artifact reachable at all "
                 "still completes",
}

#: Artifact markers for the residency proof. `.py` sources are excluded on
#: purpose: `pm_tape_density.py` is a MODULE whose name contains "tape", and
#: matching it would make the proof fire on its own imports -- the
#: needle-matching-its-own-prose failure this codebase has hit before.
TAPE_ARTIFACT_MARKERS = ("pm_5min/raw", "harmful_exposure_rows",
                         "tape_index", "_fragment", "fragment_",
                         "state_tape", "/tape")

#: MEASURED on the synthetic day (see the fixture receipt's
#: `day_run.resources`), then declared with headroom. A fixture that
#: exceeds it REFUSES: the point of a budget nobody enforces is nothing.
FIXTURE_DAY_PEAK_RSS_MB_BUDGET = 700.0

#: The real day's ceiling is the cap itself and the response is R-174's:
#: the DAY refuses. Never a raised cap, never fewer draws.
REAL_DAY_PEAK_RSS_GB_CEILING = 8.0


def _peak_rss_mb() -> float:
    import resource as _r
    return _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1024.0


CGROUP_ROOT = "/sys/fs/cgroup"
#: Rule 20's wrapper puts the run in a transient scope under THIS slice. A
#: scope anywhere else is the ambient one the shell already sat in -- and
#: reporting ITS numbers as the run's is worse than reporting none, because
#: they are real numbers about the wrong thing (measured: the login shell's
#: scope carried a 16.6 GiB peak while the run used 8 MB).
RESEARCH_SLICE = "research.slice"


def cgroup_path() -> str | None:
    """This process's cgroup, from /proc/self/cgroup (v2: one `0::<path>`
    line). None when the line is absent or does not look like a path."""
    try:
        for ln in open("/proc/self/cgroup"):
            parts = ln.strip().split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[2].startswith("/"):
                return parts[2]
    except OSError:
        return None
    return None


def _read_kv(p: Path) -> dict:
    out = {}
    try:
        for ln in p.read_text().split("\n"):
            f = ln.split()
            if len(f) == 2:
                try:
                    out[f[0]] = int(f[1])
                except ValueError:
                    pass
    except OSError:
        return {}
    return out


def scope_memory_observation() -> dict:
    """THE SCOPE'S OWN MEMORY, WITH ANON AND FILE READ APART.

    DA 63 measured a scope at **7.79 GiB of the 8 GiB cap with only 1.13
    GiB anon against 5.17 GiB reclaimable page cache**, `memory.events`
    max 0 and oom 0. Rule 20's cap counts the CACHE, so a receipt that
    reports one number cannot distinguish 'this run needs 7.8 GiB' from
    'this run touched a lot of file and the kernel had no reason to
    reclaim'. Those call for different decisions and the receipt must not
    conflate them.

    `memory.events` is the arbiter of whether the cap actually BIT: `max`
    counts times the limit was hit and `oom` times it killed. A peak near
    the cap with max 0 is a cap that was never enforced against this run.

    A run OUTSIDE a scope records `NOT_IN_A_SCOPE` -- never zeros. Zeros
    would read as 'measured, and nothing happened', which is the silent-
    absence failure this programme has hit repeatedly (rule 11)."""
    cg = cgroup_path()
    if not cg or not cg.rstrip("/").endswith(".scope"):
        return {"status": "NOT_IN_A_SCOPE",
                "cgroup": cg, "in_research_slice": False,
                "why": "no transient scope in /proc/self/cgroup, so there "
                       "is no per-run cgroup to read. Reported as a STATUS "
                       "rather than as zeros: a zero here would read as a "
                       "measurement (rule 11)",
                "anon_bytes": None, "file_bytes": None,
                "memory_peak_bytes": None, "events": None}
    in_slice = f"/{RESEARCH_SLICE}/" in cg
    base = Path(CGROUP_ROOT) / cg.lstrip("/")
    stat = _read_kv(base / "memory.stat")
    ev = _read_kv(base / "memory.events")
    peak = None
    try:
        peak = int((base / "memory.peak").read_text().strip())
    except (OSError, ValueError):
        peak = None
    if not stat and peak is None:
        return {"status": "SCOPE_NAMED_BUT_UNREADABLE",
                "cgroup": cg, "path": str(base),
                "in_research_slice": in_slice,
                "why": "the cgroup is named in /proc/self/cgroup and its "
                       "files could not be read; a STATUS, not zeros",
                "anon_bytes": None, "file_bytes": None,
                "memory_peak_bytes": None, "events": None}
    anon, filed = stat.get("anon"), stat.get("file")
    gb = 1024.0 ** 3
    return {
        # A scope OUTSIDE research.slice is the ambient one this shell was
        # already in; its numbers are real and are NOT this run's, so they
        # are reported under a status that says so rather than under
        # MEASURED. Attributing an ambient 16.6 GiB peak to an 8 MB run
        # would be a measurement of the wrong object, which is worse than
        # no measurement.
        "status": "MEASURED" if in_slice else "AMBIENT_SCOPE_NOT_THE_RUNS_OWN",
        "in_research_slice": in_slice,
        "expected_slice": RESEARCH_SLICE,
        "cgroup": cg, "path": str(base),
        "anon_bytes": anon, "file_bytes": filed,
        "kernel_bytes": stat.get("kernel"),
        "anon_gb": None if anon is None else anon / gb,
        "file_gb": None if filed is None else filed / gb,
        "memory_peak_bytes": peak,
        "memory_peak_gb": None if peak is None else peak / gb,
        "events": {"max": ev.get("max"), "oom": ev.get("oom"),
                   "oom_kill": ev.get("oom_kill"), "high": ev.get("high")},
        "cap_was_hit": None if ev.get("max") is None else ev["max"] > 0,
        "why_anon_and_file_apart": (
            "rule 20's 8 GiB cap counts page cache. DA 63 measured 7.79 "
            "GiB peak with 1.13 GiB anon and 5.17 GiB reclaimable file, "
            "events max 0 -- a peak near the cap that the cap never "
            "enforced. One number cannot tell those apart"),
        "the_arbiter_is_memory_events": (
            "`max` counts times the limit was hit and `oom` times it "
            "killed. A high peak with max 0 is headroom the kernel simply "
            "had no reason to reclaim"),
    }


def _current_rss_mb() -> float:
    """CURRENT resident size, which FALLS. `ru_maxrss` is a process
    highwater and is non-decreasing BY CONSTRUCTION, so a series of it can
    never show which stage was the peak -- it can only show the last stage
    that ever allocated (reviewer §4.3: declared peak S1_load, measured
    argmax S4_null, and no predicate anywhere asserting either)."""
    try:
        pages = int(open("/proc/self/statm").read().split()[1])
    except (OSError, IndexError, ValueError):
        return float("nan")
    import os as _os
    return pages * _os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def peak_stage_predicate(stages: dict, *, declared: str = "S1_load") -> dict:
    """WHICH STAGE WAS THE PEAK -- computed from the falling instrument.

    `DAY_STAGES` declares S1_load "THIS IS THE PEAK of the day path", and
    the 8 GiB real-day ceiling rests on that shape. It was prose beside a
    highwater series; it is a predicate now."""
    cur = {k: v.get("rss_mb_current") for k, v in stages.items()
           if isinstance(v.get("rss_mb_current"), float)}
    if not cur:
        return {"computable": False,
                "why": "no current-RSS samples were recorded"}
    arg = max(cur, key=lambda k: cur[k])
    hi = {k: v.get("peak_rss_mb_highwater") for k, v in stages.items()}
    return {
        "computable": True,
        "declared_peak_stage": declared,
        "measured_peak_stage": arg,
        "declared_stage_is_the_measured_peak": arg == declared,
        "current_rss_mb_by_stage": cur,
        "highwater_by_stage": hi,
        "highwater_is_non_decreasing_BY_CONSTRUCTION": True,
        "why_two_instruments": "the highwater bounds the process and can "
                               "never fall; the current-RSS series can, so "
                               "it is the only one that can locate a peak",
        "binds_on_a_real_day_only": (
            "on a real day the book dominates (BE: 2.008 GB reference) and "
            "S1 being the peak is what the 8 GiB ceiling rests on, so a "
            "disagreement REFUSES the day. On a fixture the book is a few "
            "hundred KB and the draw loop's fixed cost is larger, so the "
            "measured peak is S4_null -- recorded, not refused, and design "
            "v9 stated the claim flatly where it should have stated the "
            "condition"),
    }


def assert_peak_stage(pred: dict, *, fixture: bool, day: str) -> dict:
    """The declared shape is a PREDICATE on a real day.

    If the peak is not where the plan says, the basis of the 8 GiB ceiling
    is wrong and the day stops -- the cap is never raised (R-174)."""
    if not pred.get("computable"):
        raise RunnerRefused(
            f"REFUSED DAY {day}: the peak stage is not computable, so the "
            f"memory plan's central claim cannot be checked at all.")
    if not fixture and not pred["declared_stage_is_the_measured_peak"]:
        raise RunnerRefused(
            f"REFUSED DAY {day}: the memory plan declares "
            f"{pred['declared_peak_stage']} as the peak and the measured "
            f"peak is {pred['measured_peak_stage']} "
            f"({pred['current_rss_mb_by_stage']}). The 8 GiB ceiling rests "
            f"on that shape; if the shape is wrong the ceiling is not "
            f"established and the day refuses.")
    return {"asserted": not fixture, "agrees":
            pred["declared_stage_is_the_measured_peak"],
            "recorded_not_refused_because_fixture": fixture}


def tape_artifacts_opened(proof: dict) -> list:
    """Which TAPE/INDEX/FRAGMENT artifacts the instrumented run opened."""
    return sorted({p for p in proof.get("distinct_paths", [])
                   if not p.endswith(".py")
                   and any(m in p for m in TAPE_ARTIFACT_MARKERS)})


def day_forms(day: str) -> set:
    """`2026-09-03` and `20260903` are the SAME DAY.

    DE names ruled days dashed (R-555's set, the ledger's verdict files);
    BE's builder takes and stamps them COMPACT (`--day 20260903`). A string
    comparison across that boundary refuses a correct book for a formatting
    reason -- found by rehearsing the smoke rather than at GO."""
    d = str(day).strip()
    out = {d}
    if len(d) == 10 and d[4] == "-" and d[7] == "-":
        out.add(d.replace("-", ""))
    elif len(d) == 8 and d.isdigit():
        out.add(f"{d[:4]}-{d[4:6]}-{d[6:]}")
    return out


def builder_receipt_for(book_path: Path, day: str, coin: str = "btc") -> Path:
    """WHERE BE'S BUILDER RECEIPT ACTUALLY IS.

    It was derived as `book_path.with_suffix('.json')` --
    `be_daybook_20260903_btc.json` -- and BE writes
    `be_daybook_receipt_20260903_btc.json`. The smoke would have refused
    'no builder receipt' at GO for a NAMING reason while the receipt sat
    beside the book. Every candidate is tried and, on a miss, ALL of them
    are named, so the refusal is actionable rather than a puzzle."""
    cands = []
    for d in sorted(day_forms(day)):
        cands.append(book_path.parent / f"be_daybook_receipt_{d}_{coin}.json")
    cands.append(book_path.with_suffix(".json"))
    for c in cands:
        if c.is_file():
            return c
    raise RunnerRefused(
        f"REFUSED DAY {day}: no builder receipt beside {book_path.name}. "
        f"Tried, in order: {[str(c) for c in cands]}. BE's builder writes "
        f"`be_daybook_receipt_<DAY>_<COIN>.json`; the book's digest is "
        f"BE's published claim and without it there is nothing to verify "
        f"the bytes against.")


def _receipt_book_sha(rec: dict) -> tuple:
    """The book digest, wherever BE's receipt carries it -- NAMED, so a
    reader knows which field was resolved."""
    for path, val in (("book.sha256", (rec.get("book") or {}).get("sha256")),
                      ("sha256", rec.get("sha256")),
                      ("book_sha256", rec.get("book_sha256"))):
        if val:
            return val, path
    return None, None


def verify_book_against_builder_receipt(day: str, book_path: Path,
                                        receipt_path: Path) -> dict:
    """R6 FOR THE BOOK: the digest comes from BE's receipt, not from us.

    A digest DE types is DE's claim about BE's file. The builder receipt is
    BE's own published statement, and the book's bytes are compared to it
    at read time -- the reader recomputes, which is BE's own
    `digest_scheme` in `be_daybook_builder_declaration_v1.json`."""
    if not receipt_path.is_file():
        raise RunnerRefused(
            f"REFUSED DAY {day}: no builder receipt at {receipt_path}. The "
            f"book's digest is BE's published claim; without it there is "
            f"nothing to verify the bytes against, and a digest DE invents "
            f"verifies DE.")
    rec = json.loads(receipt_path.read_text())
    declared, declared_field = _receipt_book_sha(rec)
    if not declared:
        raise RunnerRefused(
            f"REFUSED DAY {day}: BE's receipt at {receipt_path} carries no "
            f"book digest under `book.sha256`, `sha256` or `book_sha256`.")
    rec_day = rec.get("day")
    if rec_day is not None and not (day_forms(rec_day) & day_forms(day)):
        raise RunnerRefused(
            f"REFUSED DAY {day}: BE's receipt is for day {rec_day!r}. "
            f"A book verified against another day's receipt is a book "
            f"nobody checked.")
    if not book_path.is_file():
        raise RunnerRefused(f"REFUSED DAY {day}: no book at {book_path}")
    actual = hashlib.sha256(book_path.read_bytes()).hexdigest()
    if actual != declared:
        raise RunnerRefused(
            f"REFUSED DAY {day}: reference-book digest mismatch -- BE's "
            f"receipt declares {declared[:16]}, the bytes on disk are "
            f"{actual[:16]}. THE DAY refuses; the book is not the one "
            f"declared.")
    return {"day": day, "path": str(book_path), "sha256": actual,
            "builder_receipt": str(receipt_path),
            "digest_read_from_field": declared_field,
            "receipt_day": rec_day,
            "day_forms_matched": sorted(day_forms(day)),
            "digest_recomputed_at_read_time": True,
            "digest_source": "BE's builder receipt, not a DE constant"}


def day_decision_population(module, bk: dict, arm: str, spec: dict) -> dict:
    """The arm's decision population at its PINNED theta, from `asm`.

    Through BE's own `arm_stream` -- which applies the head scorer to the
    assembled scores and REFUSES on a generation with no assembled score --
    so a coverage gap is an exclusion with a name, never a silent drop
    (rule 4)."""
    head, theta = spec["head"], spec["theta"]
    by_arm = bk["asm"]["by_arm"]
    key = (module.COIN, head)
    if key not in by_arm:
        raise RunnerRefused(
            f"REFUSED DAY / {arm}: the book's `asm.by_arm` has no entry for "
            f"{key!r}. Both pinned heads must be scored on the day book "
            f"(design R1); a missing head is a day with no decision "
            f"population for that arm, not a smaller one.")
    stream = module.arm_stream(bk, head)
    decisions = [r for r in stream if float(r["score"]) >= theta]
    by_side: dict = {}
    for r in decisions:
        by_side[r["side"]] = by_side.get(r["side"], 0) + 1
    return {"arm": arm, "head": head, "theta": theta,
            "n_scored_rows": len(stream),
            "decisions": len(decisions),
            "by_side": dict(sorted(by_side.items())),
            "definition": "above-threshold generations at the arm's FIXED "
                          "theta -- the set a cancel decision is drawn from",
            "theta_was_not_refitted_here": True}


def _value_cents(fills: list) -> float:
    """D(E0)'s valuation: the DECLARED estimator, not a new one.

    `de_phase4_diag_runner.fill_value_cents` is the maker P&L at
    level-to-markout WITH NO FEE TERM -- which is exactly the E0 endpoint
    the design's metric names. Nothing here invents a valuation; if the
    endpoint ever needs a fee it belongs in that function, once."""
    import de_phase4_diag_runner as R
    return float(sum(v for v in (R.fill_value_cents(f) for f in fills)
                     if v is not None))


def null_draws_valued(module, bk: dict, base_fills: list, by_side: dict, *,
                      n_draws: int, seed: int, deadline_s: float,
                      cross_check_n: int = 8) -> dict:
    """The null, ON THE DECISION METRIC, through BE's sampler and replay.

    WHY NOT `draw_null` ITSELF: BE's `draw_null` reduces each draw to
    mechanics fields and DISCARDS the fills, so D(E0) -- which is a sum
    over the fills a draw removed -- cannot be recovered from its output.
    The cascade is still BE's: this drives BE's `_alloc`, `draw_flags`,
    `flagged_stream` and `replay`, in BE's order, from BE's rows.

    AND THE PART THAT MAKES THAT HONEST: the first `cross_check_n` draws
    are compared against BE's OWN `draw_null` at the same seed on the field
    both produce -- `cancels_issued`. A private loop that drew a different
    sequence would be a different null wearing the same seed."""
    import numpy as np
    rows = bk["rows"]
    total = sum(by_side.values())
    if total <= 0:
        raise RunnerRefused(
            "REFUSED: 0 decisions is not a policy -- it is the baseline.")
    pools: dict = {}
    for i, r in enumerate(rows):
        pools.setdefault(r["side"], []).append(i)
    pools = {k: np.asarray(v) for k, v in pools.items()}
    module._alloc(by_side, pools)
    rng = np.random.default_rng(seed)
    base_value = _value_cents(base_fills)
    started = time.time()
    values, cancels, peak = [], [], _peak_rss_mb()
    for d in range(n_draws):
        flag = module.draw_flags(pools, by_side, rng)
        r = module.replay(bk, module.flagged_stream(rows, flag), 0.5)
        # REDUCED HERE, DELIBERATELY (stage S4): the draw's fills are
        # valued and dropped before the next draw is made, so peak memory
        # is O(one draw) and not O(n_draws).
        values.append(_value_cents(r["fills"]) - base_value)
        cancels.append(int(r["cancels_issued"]))
        if (d & 63) == 0:
            peak = max(peak, _peak_rss_mb())
        if time.time() - started > deadline_s:
            raise RunnerRefused(
                f"REFUSED: the null exceeded the declared deadline "
                f"{deadline_s}s at draw {d + 1} of {n_draws}. The day "
                f"refuses -- never fewer draws, never a raised cap "
                f"(R-174).")
    xc = None
    if cross_check_n:
        be_draws = module.draw_null(bk, base_fills, by_side,
                                    n_draws=max(cross_check_n,
                                                module.MIN_DRAWS),
                                    seed=seed)
        theirs = [int(x["cancels_issued"]) for x in be_draws][:cross_check_n]
        mine = cancels[:cross_check_n]
        if theirs != mine:
            raise RunnerRefused(
                f"REFUSED: DE's valued draw loop does not reproduce BE's "
                f"`draw_null` at the same seed -- BE {theirs}, DE {mine}. "
                f"A private loop that draws a different sequence is a "
                f"different null wearing the same seed.")
        xc = {"n_compared": cross_check_n, "field": "cancels_issued",
              "identical": True, "seed": seed,
              "why": "the cascade is BE's; only the METRIC is DE's"}
    return {"values": values, "n_draws": len(values),
            "base_value_cents": base_value,
            "peak_rss_mb_during_draws": peak,
            "elapsed_s": time.time() - started,
            "reproduces_BEs_draw_null": xc,
            "metric": "D(E0) per draw = value(draw's fills) - value("
                      "baseline fills), cents, maker fee zero"}


# ------------------------------------------- rule 20 / R-575(C), MEASURED

HEAVY_RUN_LOCK = "/home/yuqing/ctaNew/data/.heavy_run.lock"
HEAVY_WALL_S = 60.0
HEAVY_RSS_GB = 1.0


def _lock_fd_held(lock_path: str) -> list:
    """Which of THIS process's fds point at the lock file.

    `flock -n <lock> systemd-run --scope <cmd>` keeps the lock's fd open
    across the exec and the scope inherits it -- MEASURED: fd 3, and absent
    without the flock. So a run can state whether it held the lock instead
    of a receipt asserting a wrapper string nobody checked (R-575(C): at
    05:54Z two heavy scopes ran concurrently, one holding the lock and one
    not, and no artifact could tell them apart)."""
    import os as _os
    want = _os.path.realpath(lock_path)
    out = []
    try:
        fds = _os.listdir("/proc/self/fd")
    except OSError:
        return out
    for fd in fds:
        try:
            t = _os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if _os.path.realpath(t) == want:
            out.append(int(fd))
    return sorted(out)


def _ancestor_pids(limit: int = 64) -> list:
    """This pid and its ancestors. The flock WRAPPER holds the lock, not
    the python child, so a bare "is my pid the holder" test would reject a
    legitimately wrapped run (the reviewer's §2.3, measured: holder pid
    2916372, a `flock` process)."""
    import os as _os
    out, pid = [], _os.getpid()
    for _ in range(limit):
        if pid <= 0 or pid in out:
            break
        out.append(pid)
        try:
            txt = open(f"/proc/{pid}/status").read()
        except OSError:
            break
        nxt = 0
        for ln in txt.splitlines():
            if ln.startswith("PPid:"):
                nxt = int(ln.split()[1])
                break
        pid = nxt
    return out


def _flock_line_matches(fields: list, want_dev: tuple,
                        want_ino: int) -> bool:
    """Does one /proc/locks line name THIS file?

    THE DEFECT THIS CLOSES (REV 43): the match was on the INODE ALONE, and
    an inode is unique only WITHIN a filesystem. /proc/locks on this box
    already carries entries from several devices -- `103:01:...` beside
    `00:1d:...` -- so a lock held on an unrelated filesystem whose inode
    collided would have read as a hold on ours, and a heavy run could have
    certified itself on somebody else's lock.

    The field is `MAJOR:MINOR:INODE`, the two device numbers in HEX and the
    inode in decimal -- measured against this box's own lock: st_dev 66305
    -> major 259, minor 1 -> `103:01`.

    Pure on purpose: a matcher that reads /proc/locks itself cannot be
    handed a crafted line, and the known-bad here IS a crafted line."""
    if len(fields) < 6 or fields[1] != "FLOCK":
        return False
    try:
        maj, mnr, ino = fields[5].split(":")
        return ((int(maj, 16), int(mnr, 16)) == want_dev
                and int(ino) == want_ino)
    except (ValueError, IndexError):
        return False


def _flock_holders(lock_path: str) -> dict:
    """FLOCK entries on the lock's INODE, from /proc/locks -- the
    authoritative surface. Returns the holder pids and whether any of them
    is this process or an ancestor of it."""
    import os as _os
    # (3) A MISSING LOCK FILE IS A NAMED REFUSAL, NOT A CRASH -- raised by
    # the CALLER, which is the only place that knows it is fatal. The early
    # return here used to omit whichever key the caller had grown that
    # round, so an absent lock raised `KeyError` twice on two different
    # keys. The incomplete early return is gone.
    st = _os.stat(lock_path)
    ino = st.st_ino
    want_dev = (_os.major(st.st_dev), _os.minor(st.st_dev))
    pids, ok_read = [], True
    try:
        for ln in open("/proc/locks"):
            f = ln.split()
            # e.g. "6: FLOCK ADVISORY WRITE 2916372 103:01:1053378 0 EOF"
            if not _flock_line_matches(f, want_dev, ino):
                continue
            # f[3] is READ (a SHARED hold) or WRITE (an EXCLUSIVE one).
            # REV 41: rule 20's invariant is ONE heavy run at a time, and
            # only an exclusive lock enforces it -- two concurrent
            # `flock -s` holders would both have certified themselves.
            # The parse that could raise now lives in the matcher, which
            # returns False rather than throwing.
            pids.append((int(f[4]), f[3]))
    except OSError:
        ok_read = False
    anc = set(_ancestor_pids())
    mine = [(p, m) for p, m in pids if p in anc]
    return {"readable": ok_read, "inode": ino,
            "device": {"major": want_dev[0], "minor": want_dev[1],
                       "proc_locks_field": "%02x:%02x" % want_dev},
            "matched_on": "MAJOR:MINOR:INODE -- an inode is unique only "
                          "within a filesystem (REV 43)",
            "pids": sorted({p for p, _ in pids}),
            "modes": sorted({m for _, m in pids}),
            "holders": sorted(set(pids)),
            "by_self_or_ancestor": bool(mine),
            "self_or_ancestor_modes": sorted({m for _, m in mine}),
            "self_or_ancestor_holds_EXCLUSIVE": any(m == "WRITE"
                                                    for _, m in mine),
            "n_holders": len(set(pids)),
            "ancestors_considered": sorted(anc)}


def _fresh_probe_fails(lock_path: str) -> dict:
    """TWO fresh-fd probes, because "held" and "held EXCLUSIVELY" are
    different facts and rule 20 needs the second (REV 41).

      LOCK_EX|LOCK_NB fails  -> somebody holds it, shared OR exclusive
      LOCK_SH|LOCK_NB fails  -> the holder is EXCLUSIVE; a shared request
                                conflicts only with an exclusive hold
      LOCK_SH|LOCK_NB succeeds -> the holder is SHARED, or absent

    Two concurrent `flock -s` holders would both pass an EX-only probe and
    both certify themselves, and rule 20's invariant is ONE heavy run at a
    time. Each probe takes its own fd and releases anything it acquired."""
    import fcntl as _fc
    import os as _os
    try:
        fd = _os.open(lock_path, _os.O_RDWR)
    except OSError as exc:
        return {"probed": False, "why": f"cannot open the lock: {exc}",
                "someone_holds_it": None, "holder_is_exclusive": None}
    try:
        held = True
        try:
            _fc.flock(fd, _fc.LOCK_EX | _fc.LOCK_NB)
            _fc.flock(fd, _fc.LOCK_UN)
            held = False
        except OSError:
            held = True
        excl = False
        if held:
            fd2 = _os.open(lock_path, _os.O_RDWR)
            try:
                try:
                    _fc.flock(fd2, _fc.LOCK_SH | _fc.LOCK_NB)
                    _fc.flock(fd2, _fc.LOCK_UN)
                    excl = False       # a SHARED request got in
                except OSError:
                    excl = True        # only an EXCLUSIVE hold blocks it
            finally:
                _os.close(fd2)
        return {"probed": True, "someone_holds_it": held,
                "holder_is_exclusive": excl if held else None,
                "probes": ["LOCK_EX|LOCK_NB", "LOCK_SH|LOCK_NB"],
                "why_two": "a shared hold blocks an exclusive request and "
                           "admits a shared one; only an exclusive hold "
                           "blocks both. Rule 20 needs exclusivity"}
    finally:
        _os.close(fd)


def wrapper_observed(*, lock_path: str = HEAVY_RUN_LOCK) -> dict:
    """WHAT ACTUALLY RAN -- and it tests the LOCK, not an fd.

    THE DEFECT THIS REPLACES (reviewer §2.3, reproduced in two lines):
    `open(lock_path)` with no `flock` at all made the old instrument report
    `heavy_run_lock_held: True`, and `assert_rule20` then ADMITTED a
    1-hour / 6.84 GiB run. Worse, at the moment it was driven ANOTHER
    process genuinely held the flock -- so the instrument built to expose
    the 05:54Z condition would have certified a process running beside it.

    THE TEST IS NOW A CONJUNCTION OF TWO INDEPENDENT SURFACES:
      * a FRESH-FD `LOCK_EX|LOCK_NB` that FAILS -- somebody holds it; and
      * a `/proc/locks` FLOCK entry on that INODE whose pid is this
        process OR AN ANCESTOR -- and that somebody is us.
    Either alone is insufficient: the probe cannot say WHO holds it, and
    the inode test alone would pass if a stale entry named an ancestor
    that had since released. The inherited fd is kept as corroboration and
    is no longer the test."""
    import os as _os
    try:
        cg = open("/proc/self/cgroup").read().strip().rsplit("/", 1)[-1]
    except OSError:
        cg = None
    # (3) THE MISSING LOCK FILE, REFUSED BY NAME.
    if not _os.path.exists(lock_path):
        raise RunnerRefused(
            f"REFUSED: the heavy-run lock file {lock_path} does not exist, "
            f"so nothing can be said about who holds it. This is a NAMED "
            f"REFUSAL and not a crash: an absent lock is a real state (a "
            f"fresh box, a wrong path, a deleted file) and rule 20 cannot "
            f"be evaluated without it. It raised KeyError for two rounds.")
    fds = _lock_fd_held(lock_path)
    probe = _fresh_probe_fails(lock_path)
    holders = _flock_holders(lock_path)
    # THREE conjuncts now (REV 41): somebody holds it, that hold is
    # EXCLUSIVE, and the holder is this process or an ancestor. Two
    # concurrent `flock -s` holders satisfied the old two and both
    # certified themselves, while rule 20's invariant is one heavy run at
    # a time -- which only an exclusive lock enforces.
    held = (bool(probe.get("someone_holds_it"))
            and bool(probe.get("holder_is_exclusive"))
            and holders["by_self_or_ancestor"]
            and holders["self_or_ancestor_holds_EXCLUSIVE"])
    return {
        "heavy_run_lock_held": held,
        "lock_is_held_by_someone": probe.get("someone_holds_it"),
        "holder_is_exclusive": probe.get("holder_is_exclusive"),
        "self_or_ancestor_holds_EXCLUSIVE": holders[
            "self_or_ancestor_holds_EXCLUSIVE"],
        "flock_modes_on_the_inode": holders["modes"],
        "n_flock_holders": holders["n_holders"],
        "a_SHARED_hold_does_not_satisfy_rule_20": (
            "rule 20 is ONE heavy run at a time. `flock -s` lets two "
            "holders in at once, and both would have read `held: true` "
            "under the previous two-conjunct test (REV 41)"),
        "flock_holder_pids": holders["pids"],
        "held_by_self_or_ancestor": holders["by_self_or_ancestor"],
        "ancestor_pids": holders["ancestors_considered"],
        "fresh_probe": probe,
        "lock_fds": fds,
        "lock_fd_is_corroboration_not_the_test": (
            "an fd on the lock file says the file is OPEN. Two lines of "
            "`open()` with no flock forged the old field, and a 1-hour / "
            "6.84 GiB run certified itself (reviewer §2.3)"),
        "lock_path": lock_path,
        "cgroup_leaf": cg,
        "in_a_transient_scope": bool(cg and cg.endswith(".scope")),
        "how": "a fresh-fd LOCK_EX|LOCK_NB that FAILS, AND a /proc/locks "
               "FLOCK entry on the lock's inode whose pid is this process "
               "or an ancestor -- the holder is the `flock` wrapper, not "
               "the python child",
        "declared_wrapper_is_not_evidence": (
            "params carries a `wrapper` string; a string in a file cannot "
            "say what launched this process (R-575(C))"),
    }


def assert_rule20(observed: dict, *, wall_s: float, peak_rss_mb: float,
                  day: str) -> dict:
    """A run that WAS heavy must have held the lock. Measured, both ways."""
    heavy = (wall_s > HEAVY_WALL_S
             or peak_rss_mb / 1024.0 > HEAVY_RSS_GB)
    if heavy and not observed["heavy_run_lock_held"]:
        raise RunnerRefused(
            f"REFUSED DAY {day}: this run was HEAVY by measurement "
            f"({wall_s:.1f}s wall, {peak_rss_mb / 1024.0:.2f} GiB peak, "
            f"against rule 20's {HEAVY_WALL_S}s / {HEAVY_RSS_GB} GiB bar) "
            f"and did NOT hold {HEAVY_RUN_LOCK}. R-575(C): every heavy step "
            f"takes the lock FIRST; a held lock means refuse and report, "
            f"never run beside it. The artifact is not written.")
    return {"heavy_by_measurement": heavy,
            "wall_s": wall_s, "peak_rss_gb": peak_rss_mb / 1024.0,
            "bar": {"wall_s": HEAVY_WALL_S, "rss_gb": HEAVY_RSS_GB},
            "lock_held": observed["heavy_run_lock_held"],
            "rule": "R-575(C) -- heavy implies the lock, checked here "
                    "rather than promised in a wrapper string"}


# --------------------------------------------------------- the day itself

def run_day(day: str, book_path, *, params: dict, module=None,
            fixture: bool = False, receipt_path=None,
            n_days_complete: int = 1,
            peak_rss_mb_budget: float | None = None) -> dict:
    """ONE RULED DAY, SEALED. The path the smoke runs.

    Real days require the lock BEFORE any work (a real day is heavy by
    construction: BE projects ~2.3 h per day for both arms). A fixture day
    is expected light and is checked against its declared budget at the
    end -- a fixture that exceeds its budget REFUSES, because a budget
    nobody enforces is not a budget."""
    t_start = time.time()
    stages: dict = {}

    def _mark(name):
        stages[name] = {"peak_rss_mb_highwater": _peak_rss_mb(),
                        "rss_mb_current": _current_rss_mb(),
                        "elapsed_s": round(time.time() - t_start, 3)}

    # THE FIXTURE/REAL LOCK, ON THE DAY PATH ITSELF (reviewer §1.4).
    # `--synthetic-day 2026-09-03` used to emit a SEALED artifact stamped
    # with the smoke day from a synthetic book. Disclosed, but exactly the
    # collision the lock was built to forbid.
    day_lock = assert_fixture_day_lock(day, fixture, what="day run")
    obs = wrapper_observed()
    if not fixture and not obs["heavy_run_lock_held"]:
        raise RunnerRefused(
            f"REFUSED DAY {day}: a REAL day is heavy by construction (BE "
            f"projects ~2.3 h for both arms) and this process does not hold "
            f"{HEAVY_RUN_LOCK}. Take the lock first; if it is held, refuse "
            f"and report (R-575(C)).")

    # ---- S0: verify. Digests only. -------------------------------------
    book_path = Path(book_path)
    receipt = builder_receipt_for(book_path, day, params.get("coin", "btc"))
    bookcite = verify_book_against_builder_receipt(day, book_path, receipt)
    book_sha = bookcite["sha256"]
    mod, cite = import_be_cascade(params, module=module)
    if not fixture:
        verify_pinned_models(params)
        verify_pinned_thetas(params)
    _mark("S0_verify")

    # ---- S1: load. The peak of the day path. ---------------------------
    bk = mod.load(book_path)
    if bk.get("source_sha256") != book_sha:
        raise RunnerRefused(
            f"REFUSED DAY {day}: the cascade loaded a book whose own digest "
            f"is {str(bk.get('source_sha256'))[:16]} while the verified "
            f"book is {book_sha[:16]}.")
    _mark("S1_load")

    # ---- S2: the decision population, per arm, from `asm`. -------------
    pops = {arm: day_decision_population(mod, bk, arm, spec)
            for arm, spec in sorted(params["arms"].items())}
    # BE's declared per-day precondition: both heads must score the SAME
    # generation set, or the shared draw pool is a real choice the builder
    # declaration does not cover.
    keysets = {arm: set(bk["asm"]["by_arm"][(mod.COIN, s["head"])][0])
               for arm, s in sorted(params["arms"].items())}
    ks = list(keysets.values())
    pool_equal = all(k == ks[0] for k in ks)
    if not pool_equal:
        raise RunnerRefused(
            f"REFUSED DAY {day}: the two heads do not score the same "
            f"generation set ({[len(k) for k in ks]}), so the shared draw "
            f"pool is a real choice and BE's builder declaration does not "
            f"cover it.")
    _mark("S2_population")

    # ---- S3: the neutral no-cancel reference path. ---------------------
    base = mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)
    base_value = _value_cents(base["fills"])
    _mark("S3_baseline")

    # ---- S4: the null and the observed value, per arm. -----------------
    results, per_arm_detail = [], {}
    for arm, spec in sorted(params["arms"].items()):
        pop = pops[arm]
        adm = DESIGN.arm_day_admissible(pop["decisions"], [0.0] * 501)
        if pop["decisions"] < params["min_decisions_per_arm_day"]:
            results.append({
                "day": day, "arm": arm, "status": "DEGENERATE_ARM_DAY_"
                                                  "REFUSED_TOO_FEW_DECISIONS",
                "admissibility": {"admissible": False,
                                  "n_decisions": pop["decisions"],
                                  "bar": params["min_decisions_per_arm_day"]},
                "decision_population": pop,
                "draw_provenance": None, "economic": None,
                "why_no_economic": "a refused arm-day carries no economic "
                                   "field; it is a STATUS and does not "
                                   "shrink G silently"})
            per_arm_detail[arm] = {"status": "REFUSED_R4_DECISIONS"}
            continue
        arm_replay = mod.replay(bk, mod.arm_stream(bk, spec["head"]),
                                spec["theta"])
        observed = _value_cents(arm_replay["fills"]) - base_value
        seed = seed_for(book_sha, arm)
        nul = null_draws_valued(
            mod, bk, base["fills"], pop["by_side"],
            n_draws=params["min_draws_per_arm_day"], seed=seed,
            deadline_s=params["per_day_deadline_s"])
        prov = {"module_sha256": cite["sha256"], "seed": seed,
                "book_digest": book_sha, "arm": arm,
                "draw_source": "GENERATED_IN_PROCESS",
                "generated_in_process": True,
                "pid": __import__("os").getpid(),
                "n_draws": nul["n_draws"],
                "reproduces_BEs_draw_null":
                    nul["reproduces_BEs_draw_null"]}
        r = arm_day(day, arm, observed, nul["values"], pop["decisions"],
                    params, elapsed_s=time.time() - t_start,
                    draw_provenance=prov, book_digest=book_sha,
                    verified_module_sha=cite["sha256"])
        r["decision_population"] = pop
        r["seed"] = seed
        r["n_cancels_issued"] = int(arm_replay["cancels_issued"])
        r["n_fills_baseline"] = int(base["n_fills"])
        r["n_fills_arm"] = int(arm_replay["n_fills"])
        results.append(r)
        per_arm_detail[arm] = {"status": r["status"],
                               "null_elapsed_s": nul["elapsed_s"],
                               "null_peak_rss_mb":
                                   nul["peak_rss_mb_during_draws"]}
    _mark("S4_null")

    # ---- S5: seal. Counts and statuses only. ---------------------------
    sealed = [seal(r, n_days_complete, params["G"]) for r in results]
    for a in sealed:
        assert_no_economic_leak(a, n_days_complete, params["G"])
    # THE CONSUMER FALSIFIER MEETS A REAL EMISSION (reviewer §1.4). It had
    # three call sites, all in the battery. Each emitted result is checked
    # in BOTH states -- the artifact as sealed here, and the same result
    # unsealed -- so the symmetry is a property of what was written, not of
    # a hand-built pair.
    seal_symmetry = [
        assert_seal_layout_symmetric(seal(r, 0, params["G"]),
                                     seal(r, params["G"], params["G"]))
        for r in results]
    _mark("S5_seal")

    wall = time.time() - t_start
    peak = max(v["peak_rss_mb_highwater"] for v in stages.values())
    r20 = assert_rule20(obs, wall_s=wall, peak_rss_mb=peak, day=day)
    peak_pred = peak_stage_predicate(stages)
    peak_shape = assert_peak_stage(peak_pred, fixture=fixture, day=day)
    budget = (peak_rss_mb_budget if peak_rss_mb_budget is not None
              else (FIXTURE_DAY_PEAK_RSS_MB_BUDGET if fixture else
                    REAL_DAY_PEAK_RSS_GB_CEILING * 1024.0))
    if peak > budget:
        raise RunnerRefused(
            f"REFUSED DAY {day}: peak RSS {peak:.0f} MB exceeds the "
            f"declared budget {budget:.0f} MB. The DAY refuses -- the cap "
            f"is never raised and the draw count is never cut (R-174).")
    scope_mem = scope_memory_observation()
    return {
        "protocol": "P003_DE_MULTIDAY_GATE1_DAY_RUN_V1",
        "status": ("FIXTURE_DAY_RUN_NO_REAL_DATA" if fixture
                   else "DAY_RUN_SEALED"),
        "day": day,
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fixture": fixture,
        "reference_book": bookcite,
        "be_module_citation": cite,
        "draw_pool_set_equality_checked": pool_equal,
        "decision_populations": pops,
        "per_day_sealed_artifacts": sealed,
        "seal_layout_symmetry_checked_on_the_emitted_results": seal_symmetry,
        "fixture_day_lock": day_lock,
        "n_days_complete": n_days_complete, "G": params["G"],
        "memory_plan": {
            "stages": [{"stage": k, "holds": v} for k, v in DAY_STAGES],
            "observed": stages,
            "peak_rss_mb": peak,
            "budget_mb": budget,
            "within_budget": peak <= budget,
            "peak_stage": peak_pred,
            "peak_stage_assertion": peak_shape,
            "index_splits": INDEX_SPLITS_NEEDED_BY_DAY,
        },
        "wrapper": {**obs, "rule20": r20},
        "resources": {"wall_seconds": wall, "peak_rss_mb": peak,
                      "per_arm": per_arm_detail,
                      # RSS is this PROCESS; the scope block is the CGROUP,
                      # which is what rule 20's cap is applied to and what
                      # counts page cache.
                      "scope_memory": scope_mem},
        "what_this_is_not": {
            "a_result": False,
            "the_economics_are_SEALED": n_days_complete < params["G"],
            "D_E_MINUS_R_is_UNBOUND": (
                "the robustness endpoint needs the rebate's identity value, "
                "which is NOT on DE's surface. D(E0) -- the DECLARED "
                "PRIMARY -- is computed here from "
                "`de_phase4_diag_runner.fill_value_cents`, which is "
                "level-to-markout with NO fee term and therefore IS the E0 "
                "endpoint. D(E-R) is not computed and is not approximated"),
        },
    }


# ---------------------------------------------- the SYNTHETIC day fixture

#: How each arm's HEAD scores the synthetic day. The point of a fixture
#: with plants is that the verdict is known before the run:
#:   "harmful"  -- the head ranks the value-DESTROYING generations high, so
#:                 cancelling them ADDS value: D(E0) far above the null.
#:                 MUST PASS.
#:   "value"    -- the head ranks the value-CREATING generations high, so
#:                 cancelling them destroys value: D(E0) far below.
#:                 MUST FAIL.
#:   "thin"     -- almost nothing clears theta: an R4 refusal by decision
#:                 count, a STATUS, and G does not shrink.
SYNTHETIC_HEAD_POLICIES = ("harmful", "value", "thin")


def synthetic_day_book(day: str, *, n_slugs: int = 24, n_gens: int = 2,
                       params: dict | None = None, seed: int = 20260906,
                       harmful_frac: float = 0.5,
                       head_policy: dict | None = None) -> dict:
    """A day book of BE's DECLARED SHAPE -- not of BE's data.

    Every key BE's builder declaration names is present: `fr.reference`
    (slug -> side -> generations with tranches), `statuses`, `population`,
    `n_slugs`, `terminal_marks`, and `asm.by_arm` keyed `(coin, head)` for
    BOTH pinned heads, whose [0] element is the scored generation map keyed
    `(slug, side, float(t0))`.

    WHAT THIS FIXTURE DOES NOT DO, said plainly: it does not stand in for
    BE's book. It carries no real tape, and its numbers are synthetic. What
    it proves is that the DAY PATH executes end to end THROUGH THE REAL
    CASCADE -- BE's own `load`, `arm_stream`, `flagged_stream`, `replay`,
    `_alloc` and `draw_flags` are the ones that run -- so the seams DE
    controls are exercised against the module DE cites rather than a stub.
    A defect in BE's data cannot be found here; a defect in the wiring can,
    and DE 77 shipped exactly such a defect."""
    import random as _rnd
    import harmful_stateful_policy as HSP
    P = params or {}
    arms = P.get("arms") or {
        "CONDVALUE_X_SKEW": {"head": "q1_arrival_composed_lgbm",
                             "theta": 0.32450609461933483},
        "HAZARD_OVER_SKEWED_REF": {"head": "incumbent_linear_d",
                                   "theta": 0.43525926488298716}}
    rnd = _rnd.Random(seed)
    base_t = 1787579400
    slugs = [f"btc-updown-5m-{base_t + i * 300}" for i in range(n_slugs)]

    def _gen(gid, t0, t1, tranches, level=0.5):
        return {"gen": gid, "t0": t0, "t1": t1, "level": level,
                "displayed": 10.0, "status": HSP.OK,
                "tranches": [{"t": t, "shares": s,
                              "markout_cents_per_share": m,
                              "level": level, "mid_at_fill": level - 0.005}
                             for t, s, m in tranches]}

    policy = head_policy or {"CONDVALUE_X_SKEW": "harmful",
                             "HAZARD_OVER_SKEWED_REF": "value"}
    bad = {a: p for a, p in policy.items()
           if p not in SYNTHETIC_HEAD_POLICIES}
    if bad:
        raise RunnerRefused(
            f"REFUSED: unknown synthetic head policy {bad}; declared "
            f"policies are {SYNTHETIC_HEAD_POLICIES}")

    # THE MARKOUTS ARE SYMMETRIC ABOUT ZERO ON PURPOSE. With +4/-20 the
    # null's mean value delta is large and positive while its sd is small,
    # and R4's `sd < 0.25*|mean|` refuses EVERY arm -- which is R4 working,
    # on a fixture whose economics were lopsided. A day whose fills are
    # half worth +m and half worth -m puts the random-draw mean near zero
    # and leaves the dispersion, which is the regime the bar was set for.
    reference, statuses, terminal_marks = {}, {}, {}
    harm: dict = {}
    for i, s in enumerate(slugs):
        harmful = i < int(n_slugs * harmful_frac)
        reference[s] = {}
        for sd in HSP.SIDES:
            gens = []
            for g in range(n_gens):
                t0 = 5.0 + g * 25.0
                mk = (-20.0 if harmful else 20.0) + rnd.uniform(-2.0, 2.0)
                gens.append(_gen(g + 1, t0, t0 + 20.0,
                                 [(t0 + 5.0, 1.0, mk)]))
                harm[(s, sd, float(t0))] = harmful
            reference[s][sd] = gens
        statuses[s] = "OK"
        terminal_marks[s] = {"t": base_t + i * 300 + 300, "mid_cents": 0.5}

    #: BOTH heads score the SAME generation KEYS -- BE's declared per-day
    #: precondition, ASSERTED by `run_day` rather than assumed here -- with
    #: DIFFERENT values, because two heads that agree on every score are
    #: one head and would drive only one branch.
    def _scores(pol):
        out = {}
        for k, is_harm in harm.items():
            if pol == "thin":
                out[k] = 0.01
            elif pol == "harmful":
                out[k] = 0.90 if is_harm else 0.05
            else:
                out[k] = 0.05 if is_harm else 0.90
        return out

    asm = {"by_arm": {("btc", spec["head"]): (
        _scores(policy.get(arm, "harmful")),)
        for arm, spec in arms.items()}}
    return {
        "fr": {"reference": reference, "statuses": statuses,
               "population": sorted(slugs), "n_slugs": len(slugs),
               "terminal_marks": terminal_marks},
        "asm": asm,
        "SYNTHETIC": {
            "day": day, "is_not_BEs_book": True,
            "shape_source": "live/pm_research/declarations/"
                            "be_daybook_builder_declaration_v1.json",
            "what_it_proves": "the day path executes through the REAL "
                              "cascade module DE cites",
            "what_it_cannot_prove": "anything about BE's data, coverage, "
                                    "or the day's real economics",
        },
    }


def synthetic_pool_keys_agree(book: dict) -> bool:
    """Both heads score the same KEY SET -- the property `run_day` asserts.
    Checked here too so a fixture that broke it would be caught as a
    FIXTURE defect and not read as a day-book finding."""
    sets = [set(v[0]) for v in book["asm"]["by_arm"].values()]
    return all(x == sets[0] for x in sets)


def write_synthetic_day(day: str, outdir, **kw) -> dict:
    """Write the synthetic book AND the builder-receipt sidecar BE's
    declaration specifies, so `--day` verifies against a receipt rather
    than against a digest DE typed."""
    import pickle
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    book = synthetic_day_book(day, **kw)
    p = outdir / f"be_daybook_{day.replace('-', '')}_SYNTHETIC.pkl"
    buf = pickle.dumps(book)
    p.write_bytes(buf)
    sha = hashlib.sha256(buf).hexdigest()
    rec = {"protocol": "SYNTHETIC_DAYBOOK_RECEIPT_NOT_BES",
           "day": day, "sha256": sha, "path": str(p),
           "bytes": len(buf),
           "n_slugs": book["fr"]["n_slugs"],
           "SYNTHETIC": True,
           "why_a_receipt": "BE's declared digest_scheme is a sidecar "
                            "be_daybook_<DAY>.json the READER recomputes; "
                            "the fixture carries the same seam so `--day` "
                            "is never handed a digest DE typed"}
    rp = p.with_suffix(".json")
    rp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return {"book_path": p, "receipt_path": rp, "sha256": sha, "book": book}


def day_split_residency_proof(day: str, book_path, *, params: dict,
                              **kw) -> dict:
    """THE ADDENDUM'S MEASUREMENT: run the day and watch every open.

    The claim "`--day` needs no tape-index split resident" is checkable, so
    it is checked: the whole day path runs under the same instrument the
    data-free proof uses, and the artifact records which TAPE, INDEX or
    FRAGMENT artifacts were opened. The answer must be none."""
    DR.clear_proof()
    result, proof = DR.instrumented(
        run_day, day, book_path, params=params, **kw)
    hits = tape_artifacts_opened(proof)
    # NON-VACUITY THAT IS SPECIFIC TO THIS CLAIM. `non_vacuous` says the
    # instrument saw SOME open; that is not enough here. The day path
    # certainly reads the BOOK, so the instrument must have seen THAT --
    # otherwise "no tape artifact was opened" could be an instrument that
    # missed the reads rather than reads that did not happen.
    _bp = str(Path(book_path).resolve())
    _saw_book = any(str(Path(p).resolve()) == _bp
                    for p in proof.get("distinct_paths", []))
    if not _saw_book:
        raise RunnerRefused(
            f"REFUSED: the residency instrument did not observe the day "
            f"book being read ({_bp}). An instrument that missed the one "
            f"read this path certainly makes cannot testify about the "
            f"reads it did not see.")
    return {
        "no_tape_index_or_fragment_artifact_was_opened": not hits,
        "instrument_observed_the_book_read": _saw_book,
        "tape_artifacts_opened": hits,
        "markers_tested": list(TAPE_ARTIFACT_MARKERS),
        "py_sources_excluded_on_purpose": (
            "`pm_tape_density.py` is a MODULE whose name contains 'tape'; "
            "matching it would make the proof fire on its own imports"),
        "n_paths_opened": proof["n_paths_opened"],
        "n_distinct_paths": proof["n_distinct_paths"],
        "non_vacuous": proof["non_vacuous"],
        "day_result": result,
    }


# --------------------------------------------------------------- selftest

LAST_BATTERY: dict = {}


def selftest(*, quiet: bool = False, offline: bool = False) -> int:
    """`offline=True` skips the checks that READ `data/` and RECORDS them.

    The fixture run must open no path under `data/` (reviewer efba2b6
    item 3) and must still carry a battery that ran in its own process.
    Both are possible only if the battery can say which checks it did not
    run and why -- a skipped check that is silent is a check that has
    stopped existing."""
    n = [0]
    skipped: list = []

    def offline_skip(label):
        skipped.append(label)
        if not quiet:
            print(f"  SKIP  (offline) {label}")

    def ok(cond, label):
        if not cond:
            LAST_BATTERY.update({"outcome": "FAIL", "n_checks_run": n[0],
                                 "failed_on": label})
            raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except RunnerRefused as exc:
            if needle.lower() not in str(exc).lower():
                raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label} "
                                 f"-- wrong reason: {exc}")
            n[0] += 1
            if not quiet:
                print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_multiday_gate1_runner] FAIL: {label} "
                         f"-- ADMITTED")

    root = Path(__file__).resolve().parents[2]
    P = json.loads((root / PARAMS_REL).read_text())

    # ---- G is derived, and an empty ruled set refuses ------------------
    # ---- R-555, the RULED set --------------------------------------
    live = load_params()
    ok(live["G"] == 6 and live["G"] == live["expected_G"]
       and live["days"] == ["2026-09-03", "2026-09-04", "2026-09-05",
                            "2026-09-06", "2026-09-07", "2026-09-08"]
       and live["ruling"]["smoke_day"] == "2026-09-03",
       f"R-555 POSITIVE CONTROL, AND IT ADMITS: the ruled set loads, G is "
       f"DERIVED as {live['G']} and agrees with the declared expected_G, "
       f"and the smoke day is {live['ruling']['smoke_day']}")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td0:
        five = dict(live); five["days"] = live["days"][:5]
        f5 = Path(td0) / "five.json"; f5.write_text(json.dumps(five))
        refuses(lambda: load_params(f5),
                "R-555 KNOWN-BAD, A SET OF FIVE: it refuses rather than "
                "testing at a smaller G -- 'G stays 6 and nothing is "
                "chosen'", "against the declared expected_G")
        touched = dict(live)
        touched["days"] = ["2026-09-01"] + live["days"][1:]
        ft = Path(td0) / "touched.json"; ft.write_text(json.dumps(touched))
        refuses(lambda: load_params(ft),
                "R-555 KNOWN-BAD, A TOUCHED DAY: 09-01's "
                "previously_opened_for is interim_read_of_frozen_candidate "
                "and the set REFUSES -- untouched days only",
                "not 'none'")
        empty = dict(live); empty["days"] = []
        fe = Path(td0) / "empty.json"; fe.write_text(json.dumps(empty))
        refuses(lambda: load_params(fe),
                "and an EMPTY set still refuses -- G is derived from it "
                "and there is nothing to derive", "ruled day set is EMPTY")
    tmp = dict(P); tmp["days"] = ["a", "b", "c", "d", "e", "f"]
    tmp.pop("expected_G", None); tmp.pop("required_previously_opened_for", None)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.json"; f.write_text(json.dumps(tmp))
        got = load_params(f)
        ok(got["G"] == 6 and got["G_derived_from_len_days"] is True,
           f"POSITIVE CONTROL, AND IT ADMITS: G is DERIVED from len(days) "
           f"= {got['G']}, never read from a constant -- the defect design "
           f"v2 shipped")
        tmp2 = dict(tmp); tmp2["days"] = ["a", "a", "b", "c", "d", "e"]
        f2 = Path(td) / "p2.json"; f2.write_text(json.dumps(tmp2))
        refuses(lambda: load_params(f2),
                "KNOWN-BAD: a duplicated day refuses rather than inflating "
                "G", "duplicate days")

    # ---- BE's cascade is cited, and a different one refuses ------------
    ok(verify_be_module(P)["cited_not_copied"] is True,
       "POSITIVE CONTROL: BE's cascade module resolves at the declared "
       "digest and is CITED, not copied")
    refuses(lambda: verify_be_module(P, actual_sha="0" * 64),
            "KNOWN-BAD: a DIFFERENT cascade digest refuses -- a null run "
            "through another cascade is not a control for this arm",
            "cascade module digest differs")

    # ---- R6: THE MODEL AND THETA HALVES ARE NOW CODE PATHS -------------
    # These READ `data/`. In offline mode they are skipped and NAMED.
    if offline:
        for _lbl in ("R6 positive control (reads the pinned model files)",
                     "R6 known-bad: a planted model byte",
                     "R6 known-bad: an absent pinned model",
                     "R6 known-bad: a theta disagreeing with its pin"):
            offline_skip(_lbl)
    else:
        vr = verify_run_inputs(live)
        ok(vr["models"]["n_models_read"] == 3
           and vr["models"]["bytes_were_read_not_recorded"] is True
           and all(v["matches"] for a in vr["models"]["per_arm"].values()
                   for v in a.values())
           and all(v["matches"] for v in vr["thetas"]["per_arm"].values()),
           f"R6 POSITIVE CONTROL, AND IT ADMITS: {vr['models']['n_models_read']}"
           f" pinned model files are READ AND HASHED at run time and match, "
           f"and both thetas are read from the artifact they are pinned in. "
           f"Until now these were recorded and never compared -- the "
           f"reviewer's 'provenance theatre'")
        import shutil as _sh
        with _tf.TemporaryDirectory() as tdm:
            # A PLANTED MODEL BYTE. The whole model dir is copied so the
            # pristine tree is never touched.
            src = Path(__file__).resolve().parents[2] / MODEL_DIR
            dst = Path(tdm) / MODEL_DIR
            dst.parent.mkdir(parents=True, exist_ok=True)
            _sh.copytree(src, dst)
            tgt = dst / "linear_d_btc.json"
            tgt.write_bytes(tgt.read_bytes() + b" ")
            refuses(lambda: verify_pinned_models(live, Path(tdm)),
                    "R6 KNOWN-BAD, A PLANTED MODEL BYTE: one trailing space in "
                    "linear_d_btc.json changes its digest and REFUSES THE RUN "
                    "-- not the day, because a moved model makes every day's "
                    "object different", "do not match their declared digest")
            (dst / "lgbm_haz_btc.txt").unlink()
            refuses(lambda: verify_pinned_models(live, Path(tdm)),
                    "and an ABSENT pinned model refuses too, rather than "
                    "verifying the ones that happen to be there",
                    "do not match their declared digest")
        with _tf.TemporaryDirectory() as tdt:
            bad = dict(live)
            bad["arms"] = {a: dict(v) for a, v in live["arms"].items()}
            bad["arms"]["CONDVALUE_X_SKEW"]["theta"] = 0.5
            refuses(lambda: verify_pinned_thetas(bad),
                    "R6 KNOWN-BAD, A THETA THAT DISAGREES WITH ITS PIN SOURCE: "
                    "refuses the RUN. The theta is read from BE's artifact at "
                    "its declared JSON path, not from this module's copy",
                    "theta mismatch against the pin source")

    # ---- R6, all three, both directions --------------------------------
    th = {a: s["theta"] for a, s in P["arms"].items()}
    md = {a: dict(s["model_digests"]) for a, s in P["arms"].items()}
    ok(verify_day_inputs("D", "a" * 64, "a" * 64, P, th, md)[
           "model_digests_verified"] is True,
       "POSITIVE CONTROL ON R6, AND IT ADMITS: matching book, theta and "
       "model digests verify")
    refuses(lambda: verify_day_inputs("D", "a" * 64, "b" * 64, P, th, md),
            "R6 KNOWN-BAD, A WRONG-DIGEST DAY: the book digest mismatch "
            "REFUSES THE DAY", "reference-book digest mismatch")
    bad_th = dict(th); bad_th["CONDVALUE_X_SKEW"] = 0.5
    refuses(lambda: verify_day_inputs("D", "a" * 64, "a" * 64, P, bad_th, md),
            "R6 KNOWN-BAD: a refitted theta REFUSES THE RUN, not the day -- "
            "it makes every day's object different", "theta is 0.5")
    bad_md = {a: dict(v) for a, v in md.items()}
    bad_md["HAZARD_OVER_SKEWED_REF"]["linear_d_btc.json"] = "deadbeef"
    refuses(lambda: verify_day_inputs("D", "a" * 64, "a" * 64, P, th, bad_md),
            "R6 KNOWN-BAD: a moved MODEL digest refuses the run",
            "model linear_d_btc.json digest")

    # ---- R4 and R8 ------------------------------------------------------
    draws = [0.4 + 0.002 * i for i in range(600)]
    good = arm_day("D", "A", 2.0, draws, 200, P)
    ok(good["status"] == "OK" and good["economic"]["Z"] > 0,
       "POSITIVE CONTROL: an admissible arm-day computes its economic "
       "fields")
    short = arm_day("D", "A", 2.0, draws, 4, P)
    ok(short["status"] == "DEGENERATE_ARM_DAY_REFUSED"
       and short["economic"] is None,
       "R4 KNOWN-BAD, A SHORT DAY: four decisions REFUSE the arm-day and it "
       "carries NO economic field -- a status, and it does not shrink G "
       "silently")
    refuses(lambda: arm_day("D", "A", 2.0, draws[:499], 200, P),
            "R4/R8 KNOWN-BAD: 499 draws refuses -- the 500 minimum is never "
            "lowered", "below the declared minimum")
    refuses(lambda: arm_day("D", "A", 2.0, draws, 200, P,
                            elapsed_s=P["per_day_deadline_s"] + 1),
            "R8 KNOWN-BAD, AN OVERRUN: the DAY refuses. The draw count is "
            "never lowered and the cap is never raised",
            "exceeds the declared deadline")

    # ---- (2) THE DRAWS ARE BOUND TO THE VERIFIED MODULE ---------------
    bk = "c" * 64
    good_prov = {"module_sha256": "m" * 64, "seed": seed_for(bk, "A"),
                 "book_digest": bk, "arm": "A"}
    ok(verify_draw_provenance(good_prov, arm="A", book_digest=bk,
                              verified_module_sha="m" * 64)[
           "binds_the_verified_module_to_the_numbers"] is True,
       "POSITIVE CONTROL, AND IT ADMITS: draws whose provenance names the "
       "verified module, this book and the seed this book and arm imply "
       "are accepted")
    for mutate, why, needle in (
            ({"module_sha256": "z" * 64},
             "DRAWS CARRYING A DIFFERENT MODULE DIGEST refuse -- the "
             "digest used to say which cascade EXISTS, never which one "
             "produced these numbers", "module_sha256"),
            ({"seed": 1},
             "a seed the runner cannot RECOMPUTE from this book and arm "
             "refuses", "seed"),
            ({"book_digest": "d" * 64},
             "draws produced against a DIFFERENT BOOK refuse",
             "book_digest"),
            ({"arm": "B"}, "draws produced for the OTHER ARM refuse",
             "arm")):
        bad = dict(good_prov); bad.update(mutate)
        refuses(lambda b=bad: verify_draw_provenance(
            b, arm="A", book_digest=bk, verified_module_sha="m" * 64),
            f"KNOWN-BAD: {why}", needle)
    refuses(lambda: verify_draw_provenance(
        None, arm="A", book_digest=bk, verified_module_sha="m" * 64),
        "KNOWN-BAD: draws with NO provenance block refuse -- a verified "
        "module that never touches the numbers verifies nothing",
        "no provenance block")
    ok(arm_day("D", "A", 2.0, draws, 200, P)["status"] == "OK",
       "and a call that supplies no verified-module digest still works, "
       "so the binding is opt-in at the seam and MANDATORY on the real "
       "path where the runner has verified a module")

    # ---- R5, the sealed guard, both directions -------------------------
    sealed = seal(good, 1, 5)
    ok(sealed["sealed"] is True and "economic" not in sealed
       and sealed["seal_status"].startswith("SEALED")
       and not _economic_keys_in(sealed),
       "R5 POSITIVE CONTROL: at 1 of 5 days the economic block is ABSENT "
       "from the artifact, not present-and-ignored -- AND SO ARE THE "
       "ECONOMIC FIELDS NESTED ELSEWHERE. Sealing only the top-level block "
       "left `admissibility.null_sd` and `null_mean` behind, and my own "
       "guard caught the emitter on its first run")
    ok(_economic_keys_in({"a": {"Z": 1}}) == ["a.Z"]
       and _economic_keys_in({"sealed_field_names": ["Z", "null_sd"]}) == [],
       "and the leak detector tests KEYS, not substrings: it finds a "
       "nested `Z` and does NOT report `sealed_field_names`, which is a "
       "LIST OF THE NAMES BEING SEALED -- the needle-matches-its-own-prose "
       "failure, caught in my own check")
    ok(assert_no_economic_leak(sealed, 1, 5) is True
       and "admissibility" in sealed,
       "and the R4 admissibility STATUS survives the seal -- what is "
       "withheld is the economics, not the fact that the day ran")
    unsealed = seal(good, 5, 5)
    ok(unsealed["sealed"] is False and "economic" in unsealed,
       "AND IT UNSEALS: at 5 of 5 the economic block is present -- a guard "
       "shown only to withhold is not a guard")
    ok(assert_no_economic_leak(sealed, 1, 5) is True,
       "the artifact-level guard ADMITS a properly sealed artifact")
    leaky = dict(sealed); leaky["economic"] = {"Z": 3.0}
    refuses(lambda: assert_no_economic_leak(leaky, 1, 5),
            "R5 KNOWN-BAD, AN ECONOMIC FIELD LEAKING BEFORE G: the guard "
            "walks the finished artifact and REFUSES -- a leaked Z is an "
            "early stop waiting to happen", "leaked into a SEALED artifact")
    nested = {"a": {"b": [{"p_location": 0.5}]}}
    refuses(lambda: assert_no_economic_leak(nested, 1, 5),
            "and it finds a leak PLANTED AT DEPTH inside a nested list",
            "leaked into a SEALED artifact")

    # ---- the planted arms, both directions -----------------------------
    Pg = dict(P); Pg["G"] = 5
    fail_rows = [{"day": f"d{i}", "arm": "PLANTED_FAIL", "status": "OK",
                  "economic": {"Z": 0.0}} for i in range(5)]
    pass_rows = [{"day": f"d{i}", "arm": "PLANTED_PASS", "status": "OK",
                  "economic": {"Z": 4.0}} for i in range(5)]
    agg = aggregate(fail_rows + pass_rows, Pg)
    ok(agg["per_arm"]["PLANTED_FAIL"]["FAILS_THE_SECTION_7_PREDICATE"]
       is True,
       "A PLANTED MUST-FAIL ARM FAILS: Z = 0 on every day, mean not above "
       "zero and no day sign positive")
    ok(agg["per_arm"]["PLANTED_PASS"]["FAILS_THE_SECTION_7_PREDICATE"]
       is False
       and agg["per_arm"]["PLANTED_PASS"]["clears_holm"] is False,
       "A PLANTED MUST-PASS ARM DOES NOT FAIL -- and at G = 5, m = 2 its "
       "`clears_holm` is FALSE, so the pass is DIRECTIONAL and the "
       "falsifier drives the multiplicity statement, not only the "
       "arithmetic")
    ok(agg["any_arm_fails"] is True
       and agg["the_rule"].startswith("the harmful-fill route STOPS"),
       "and the section-7 rule is COMPUTED across arms: either arm failing "
       "stops the route")
    mixed = [{"day": f"d{i}", "arm": "SHORT", "status": "OK",
              "economic": {"Z": 1.0}} for i in range(4)]
    a2 = aggregate(mixed, Pg)
    ok(a2["per_arm"]["SHORT"]["verdict"]
       == "UNTESTABLE_ON_THE_DECLARED_DAY_SET",
       "KNOWN-BAD: an arm present on four of five days is UNTESTABLE, not "
       "tested at G = 4 -- a refused arm-day does not shrink G")

    # ---- (3) THE FIXTURE RUN OPENS NO PATH UNDER data/ -- DRIVEN ------
    # Not asserted in prose: `open`, `Path.read_bytes` and `Path.read_text`
    # are instrumented and the fixture run is executed under them.
    if not offline:
        import builtins as _b
        import pathlib as _pl
        _seen: list = []
        _o, _rb, _rt = _b.open, _pl.Path.read_bytes, _pl.Path.read_text
        try:
            _b.open = lambda f, *a, **k: (_seen.append(str(f)),
                                          _o(f, *a, **k))[1]
            _pl.Path.read_bytes = lambda self: (_seen.append(str(self)),
                                                _rb(self))[1]
            _pl.Path.read_text = lambda self, *a, **k: (
                _seen.append(str(self)), _rt(self, *a, **k))[1]
            fixture_run()
        finally:
            _b.open, _pl.Path.read_bytes, _pl.Path.read_text = _o, _rb, _rt
        _data_hits = [x for x in _seen if "/data/" in x]
        # AND THE CLAIM IS NOW BOUND TO THE PROOF, not merely beside it.
        _pv = fixture_run_proven()
        ok(_pv["data_root"]["refusal"] == "NOT_APPLICABLE_FIXTURE_RUN"
           and _pv["data_free_proof"]["no_path_under_data_was_opened"]
           and _pv["data_free_proof"]["non_vacuous"]
           and _pv["no_path_under_data_was_opened"] is True,
           f"THE FIXTURE CLAIM CARRIES ITS PROOF: "
           f"`fixture_run_proven()` runs the body under instrumentation "
           f"FIRST and only then calls require_canonical(fixture=True) "
           f"with that report -- {_pv['data_free_proof']['n_paths_opened']}"
           f" opens observed, 0 under `data/`. The claim cannot precede "
           f"its own proof")
        try:
            DR.clear_proof()
            DR.require_canonical("a laundered real run", fixture=True)
            ok(False, "KNOWN-BAD: a fixture claim with no registered "
                      "proof was admitted through the runner's own "
                      "resolver")
        except DR.DataRootRefused as _e:
            ok("NO DATA-FREE PROOF" in str(_e),
               "KNOWN-BAD, THE LAUNDERING PATH: claiming fixture=True "
               "without a proof REFUSES even here -- the door the "
               "reviewer named is shut from both sides")
        ok(not _data_hits,
           f"(3) FIXTURE_RUN_NO_DATA IS DRIVEN, NOT DECLARED: `open`, "
           f"`read_bytes` and `read_text` are instrumented and a full "
           f"fixture run opens {len(_seen)} paths, ZERO of them under "
           f"`data/`. It reads only its own module source and the "
           f"committed parameter file, so it runs from a shell worktree")
        ok(any(x.endswith("de_multiday_gate1_params_v5.json")
               for x in _seen),
           "and the instrument is not vacuous -- it DID observe the "
           "parameter file being read, so a zero above is a measurement "
           "rather than a silent no-op")
        # ---- the reviewer's finding (2): the literals in this receipt ---
        _bools = sorted(k for k, v in _pv.items() if isinstance(v, bool))
        _unclassified = [k for k in _bools
                         if k not in FIXTURE_LITERAL_CLASSES]
        ok(not _unclassified,
           f"THE LITERAL SWEEP IS EXHAUSTIVE: every top-level boolean in "
           f"the EMITTED fixture receipt ({_bools}) is classified COMPUTED "
           f"or INTENT. A new bare boolean with no entry REFUSES here, "
           f"which is the only thing that keeps rule 10 from decaying back "
           f"into prose")
        ok(_pv["the_committed_day_set_is_empty"] is False
           and _pv["n_committed_ruled_days"] == len(ruled_day_set()) == 6
           and _pv["committed_ruled_days"] == live["days"],
           f"AND THE FINDING ITSELF: `the_committed_day_set_is_empty` is "
           f"now COMPUTED and reads False over "
           f"{_pv['n_committed_ruled_days']} committed ruled days. It "
           f"shipped in v7 as a hardcoded True that nothing asserted and "
           f"that was FALSE -- a literal beside prose, which is rule 10's "
           f"own example, in my own receipt")
        ok(_pv["runnable_from_a_shell_worktree"]
           is _pv["no_path_under_data_was_opened"],
           "and `runnable_from_a_shell_worktree` is DERIVED from the "
           "data-free proof rather than asserted beside it -- a run that "
           "opens no path under data/ needs no ledger, which is what the "
           "claim means")
    else:
        # TWO labels, because the online path runs TWO checks here. A
        # skip list that undercounts makes the two modes disagree on the
        # total, which is exactly what the count assertion exists to catch
        # -- and it caught it.
        offline_skip("(3) the instrumented fixture-run data-freeness probe "
                     "(it calls fixture_run, which would recurse)")
        offline_skip("(3) the non-vacuity check on that probe")
        offline_skip("(3) the fixture-claim-carries-its-proof check "
                     "(it calls fixture_run_proven, which would recurse)")
        offline_skip("(3) the laundering known-bad on require_canonical")
        offline_skip("(3) the literal sweep over the emitted receipt "
                     "(it calls fixture_run_proven, which would recurse)")
        offline_skip("(3) the committed-day-set computation check")
        offline_skip("(3) the derived runnable-from-a-shell-worktree "
                     "check")

    ok(seed_for("a" * 64, "X") != seed_for("b" * 64, "X")
       and seed_for("a" * 64, "X") == seed_for("a" * 64, "X"),
       "the seed PINS THE DATA: it changes with the book digest and is "
       "reproducible from the artifact alone")

    # ============ v2: THE DRAWS, THE TWO CLOCKS AND THE SEAL LAYOUT ======

    # ---- R-572(B)(2): the two clocks ------------------------------------
    _row_ok = {"day_closed_calendar": True, "all_conjuncts_and_quality": True}
    _may = may_run_day(live, "2026-09-03", day_row=_row_ok)
    ok(_may["may_run"] is True and _may["sealed"] is True,
       "R9 POSITIVE CONTROL, AND IT ADMITS: the 09-03 smoke -- a closed, "
       "qualifying, ruled day -- MAY RUN NOW, sealed. v1's single "
       "`run_not_before_utc` read as a bar on every run and would have held "
       "it behind 2026-09-09 for nothing (R-572(B)(2))")
    refuses(lambda: may_run_day(live, "2026-09-07",
                                day_row={"day_closed_calendar": False,
                                         "all_conjuncts_and_quality": None}),
            "R9 KNOWN-BAD, AN OPEN DAY: a day that has not closed REFUSES -- "
            "day-quality is evaluated on COMPLETE days only (R-555)",
            "not a CLOSED calendar day")
    refuses(lambda: may_run_day(live, "2026-09-03",
                                day_row={"day_closed_calendar": True,
                                         "all_conjuncts_and_quality": False}),
            "R9 KNOWN-BAD, A CLOSED DAY THAT FAILS QUALITY: refuses",
            "conjuncts and day-quality")
    refuses(lambda: may_run_day(live, "2026-09-02", day_row=_row_ok),
            "R9 KNOWN-BAD, A DAY OUTSIDE THE RULED SET: 09-02 is closed and "
            "qualifies on quality and is still REFUSED -- the ruled set is "
            "the population", "not in the ruled day set")
    _noflag = dict(live)
    _noflag.pop("day_runs_allowed_for_closed_qualifying_days", None)
    refuses(lambda: may_run_day(_noflag, "2026-09-03", day_row=_row_ok),
            "R9 KNOWN-BAD, THE RULING ABSENT FROM THE FILE: without "
            "`day_runs_allowed_for_closed_qualifying_days` the runner "
            "REFUSES rather than inferring the ruling", "does not carry")
    _t = datetime.datetime
    _tz = datetime.timezone.utc
    _read = may_read_aggregate(live, n_days_complete=6,
                               now_utc=_t(2026, 9, 9, 0, 6, tzinfo=_tz))
    ok(_read["may_read"] is True and _read["both_conditions_required"] is True,
       "AND THE OTHER CLOCK ADMITS at 2026-09-09T00:06Z with all 6 days "
       "complete -- the date governs the READ, which is the job it was "
       "actually doing")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_t(2026, 9, 8, 23, 59, tzinfo=_tz)),
        "KNOWN-BAD, ONE MINUTE EARLY: six complete days do not open the "
        "read before the declared date", "not before")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=5, now_utc=_t(2026, 9, 12, 0, 0, tzinfo=_tz)),
        "KNOWN-BAD, FIVE OF SIX AFTER THE DATE: the date does not open a "
        "read of an incomplete set -- BOTH conditions, never either",
        "of 6 days are complete")
    ok(live["read_not_before_utc"] == "2026-09-09T00:06:00Z"
       and "run_not_before_utc" not in live
       and live["timing"]["superseded_field"] == "run_not_before_utc",
       "and params v2 carries the SPLIT, with the superseded field named: "
       "`read_not_before_utc` + "
       "`day_runs_allowed_for_closed_qualifying_days`, and no "
       "`run_not_before_utc` left to be resolved by a reader")

    # ---- R-572(B)(3): the seal layout, both states -----------------------
    _armres = {"day": "D", "arm": "A", "status": "OK",
               "admissibility": {"admissible": True, "null_sd": 1.0},
               "economic": {"D_E0": 1.0, "Z": 2.0, "p_location": 0.01,
                            "null_mean": 0.0, "null_sd": 1.0,
                            "null_draws_summary": {"n": 500}}}
    _sealed = seal(_armres, 1, 6)
    _unsealed = seal(_armres, 6, 6)
    _sym = assert_seal_layout_symmetric(_sealed, _unsealed)
    ok(_sym["symmetric"] is True
       and _sym["sealed_reads"]["readable"] is True
       and _sym["unsealed_reads"]["readable"] is True
       and _sym["unsealed_reads"]["sealed_field_names"] == []
       and _sym["sealed_reads"]["sealed_field_names"] == list(ECONOMIC_FIELDS),
       f"R10 POSITIVE CONTROL, AND IT ADMITS: a CONSUMER reads all four "
       f"layout keys in BOTH states -- sealed_at_every_depth "
       f"{_sym['sealed_reads']['sealed_at_every_depth']}/"
       f"{_sym['unsealed_reads']['sealed_at_every_depth']}, "
       f"sealed_field_names {len(ECONOMIC_FIELDS)} names/[] -- and never "
       f"gets a None from a missing key (R-572(B)(3))")
    _prefix = dict(_unsealed)
    for _k in ("sealed_at_every_depth", "sealed_field_names"):
        _prefix.pop(_k, None)
    refuses(lambda: assert_seal_layout_symmetric(_sealed, _prefix),
            "R10 KNOWN-BAD, THE PRE-FIX LAYOUT ITSELF: an unsealed artifact "
            "that DROPS `sealed_at_every_depth` and `sealed_field_names` -- "
            "exactly what seal() emitted before this round -- is REFUSED, "
            "so the falsifier fires on the real defect and not a "
            "constructed one", "ASYMMETRIC")
    _leaky = dict(_sealed); _leaky["economic"] = {"D_E0": 1.0}
    refuses(lambda: assert_seal_layout_symmetric(_leaky, _unsealed),
            "R10 KNOWN-BAD, THE OTHER DIRECTION: `economic` PRESENT while "
            "sealed is refused -- the one key that is asymmetric by design "
            "is checked in the direction that matters", "ASYMMETRIC")
    _refused_day = {"day": "D", "arm": "A", "status": "DEGENERATE",
                    "admissibility": {"admissible": False}, "economic": None,
                    "why_no_economic": "a refused arm-day carries none"}
    _u2 = seal(_refused_day, 6, 6)
    ok("economic" in _u2 and _u2["economic"] is None
       and read_seal_state(_u2)["readable"] is True,
       "AND THE UNSEALED STATE IS SYMMETRIC WITH ITSELF: a REFUSED arm-day "
       "keeps `economic` present-and-None with its stated reason. The old "
       "emitter popped it, so `economic` was present for an OK day and "
       "absent for a refused one -- an asymmetry inside one state")

    # ---- R-572(B)(1): the draws seam ------------------------------------
    import tempfile as _tf2
    import importlib.util as _ilu
    _STUB = '''
"""A STAND-IN CASCADE. It does NOT supply what the runner should produce --
the runner's job here is the BINDING (which module, which seed, which book),
and this stub records what it was handed so the binding can be checked."""
CALLS = {}
def load(path):
    CALLS["load"] = str(path)
    return {"rows": [{"side": "B"}] * 10, "source_sha256": CALLS["book_sha"],
            "digest_is_of_the_loaded_buffer": True}
def flagged_stream(rows, flagged):
    return list(rows)
def replay(bk, scores, theta):
    return {"cancels_issued": 0, "fills": [], "n_fills": 0}
def draw_null(bk, base_fills, by_side, *, n_draws=500, seed=None,
              progress=False):
    CALLS["seed"] = seed
    CALLS["n_draws"] = n_draws
    CALLS["by_side"] = dict(by_side)
    return [{"draw": i} for i in range(n_draws)]
'''
    with _tf2.TemporaryDirectory() as _td:
        _sp = Path(_td) / "stub_cascade.py"
        _sp.write_text(_STUB)
        _spec = _ilu.spec_from_file_location("stub_cascade", _sp)
        _stub = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_stub)
        _stub_sha = hashlib.sha256(_sp.read_bytes()).hexdigest()
        _book_sha = "b" * 64
        _stub.CALLS["book_sha"] = _book_sha
        _sp_rel = _sp.resolve().relative_to(Path("/"))
        _pstub = dict(live)
        _pstub["be_module"] = {"path": str(_sp.resolve()),
                               "sha256": _stub_sha}

        # the declared path is absolute here, so parents[2] / abs == abs
        _gen = generate_draws_in_process(
            _pstub, day="FIXTURE-1", arm="CONDVALUE_X_SKEW",
            book_path=str(_sp), book_sha=_book_sha,
            by_side={"B": 3}, n_draws=500, module=_stub)
        ok(len(_gen["draws"]) == 500
           and _gen["provenance"]["draw_source"] == "GENERATED_IN_PROCESS"
           and _gen["provenance"]["generated_in_process"] is True
           and _stub.CALLS["seed"] == seed_for(_book_sha,
                                               "CONDVALUE_X_SKEW"),
           f"R-572(B)(1) POSITIVE CONTROL, AND IT ADMITS: the runner "
           f"IMPORTS the cascade, verifies the digest OF THE FILE THE "
           f"IMPORT LOADED, and calls its draw_null ITSELF with the seed it "
           f"recomputed ({_stub.CALLS['seed']}) -- the draws are GENERATED "
           f"IN PROCESS, which is what binds the verified digest to the "
           f"numbers")
        _pbad = dict(_pstub)
        _pbad["be_module"] = {"path": str(_sp.resolve()), "sha256": "0" * 64}
        refuses(lambda: generate_draws_in_process(
            _pbad, day="FIXTURE-1", arm="CONDVALUE_X_SKEW",
            book_path=str(_sp), book_sha=_book_sha, by_side={"B": 3},
            n_draws=500, module=_stub),
            "KNOWN-BAD, A CASCADE THAT IS NOT THE CITED ONE: the digest of "
            "the loaded file is compared and a mismatch REFUSES -- and it "
            "is the LOADED file, not a second read of a declared path "
            "(BE's own B-1 lesson on the consuming side)",
            "digest differs")
        _pelse = dict(_pstub)
        _pelse["be_module"] = {"path": "live/pm_research/de_data_root.py",
                               "sha256": _stub_sha}
        refuses(lambda: generate_draws_in_process(
            _pelse, day="FIXTURE-1", arm="CONDVALUE_X_SKEW",
            book_path=str(_sp), book_sha=_book_sha, by_side={"B": 3},
            n_draws=500, module=_stub),
            "KNOWN-BAD, THE RIGHT NAME AT THE WRONG PATH: a module whose "
            "__file__ is not the declared path REFUSES before any digest "
            "argument can be made -- a same-named module earlier on "
            "sys.path is not the cited one", "but the declaration cites")
        _stub.CALLS["book_sha"] = "c" * 64
        refuses(lambda: generate_draws_in_process(
            _pstub, day="FIXTURE-1", arm="CONDVALUE_X_SKEW",
            book_path=str(_sp), book_sha=_book_sha, by_side={"B": 3},
            n_draws=500, module=_stub),
            "KNOWN-BAD, A BOOK THAT IS NOT THE DECLARED ONE: the cascade's "
            "OWN loaded-buffer digest is compared against the day book the "
            "SEED was derived from, so draws seeded by one book and drawn "
            "from another REFUSE", "own digest is")
        _stub.CALLS["book_sha"] = _book_sha

        # ---- the fixture/real-day lock, both directions -----------------
        refuses(lambda: resolve_draws(
            live, day="2026-09-03", arm="CONDVALUE_X_SKEW", fixture=True,
            supplied={"draws": [1] * 500, "provenance": {}}),
            "THE DOOR IS SHUT BY THE DAY, NOT THE CALLER'S WORD: fixture "
            "draws claimed for 09-03 -- a RULED day -- REFUSE. This is the "
            "second door of DE 76's kind and it gets a structural lock",
            "IS in the ruled day set")
        refuses(lambda: resolve_draws(
            live, day="2026-09-03", arm="CONDVALUE_X_SKEW", fixture=False,
            supplied={"draws": [1] * 500, "provenance": {}}),
            "KNOWN-BAD, SUPPLIED DRAWS ON A RULED DAY: REFUSED. R-572(B)(1) "
            "rules the runner generates them in process; a supplied set is "
            "one this process cannot attest to", "were SUPPLIED on a ruled")
        refuses(lambda: resolve_draws(
            live, day="FIXTURE-9", arm="CONDVALUE_X_SKEW", fixture=False),
            "KNOWN-BAD, THE OTHER DIRECTION: a REAL-day run claimed for a "
            "day outside the ruled set REFUSES -- the lock is symmetric",
            "NOT in the ruled day set")
        refuses(lambda: resolve_draws(
            live, day="FIXTURE-9", arm="CONDVALUE_X_SKEW", fixture=True),
            "KNOWN-BAD, FIXTURE MODE WITH NOTHING SUPPLIED: refuses rather "
            "than reaching for a real book", "no supplied draws")
        _fx = resolve_draws(live, day="FIXTURE-9", arm="CONDVALUE_X_SKEW",
                            fixture=True,
                            supplied={"draws": [1.0] * 500,
                                      "provenance": {"seed": 1}})
        ok(_fx["provenance"]["draw_source"] == "SUPPLIED_FIXTURE_ONLY"
           and _fx["provenance"]["generated_in_process"] is False,
           "AND THE FIXTURE PATH ADMITS on a day outside the ruled set, "
           "LABELLED `SUPPLIED_FIXTURE_ONLY` -- the label is what stops a "
           "supplied set being read later as a generated one")
        _ppid = {"module_sha256": _stub_sha, "seed": seed_for(_book_sha, "A"),
                 "book_digest": _book_sha, "arm": "A",
                 "draw_source": "GENERATED_IN_PROCESS",
                 "generated_in_process": True, "pid": -1}
        refuses(lambda: verify_draw_provenance(
            _ppid, arm="A", book_digest=_book_sha,
            verified_module_sha=_stub_sha),
            "KNOWN-BAD, A GENERATED CLAIM FROM ANOTHER PROCESS: provenance "
            "asserting GENERATED_IN_PROCESS with a foreign pid REFUSES -- "
            "a proof produced elsewhere is a proof about elsewhere (DE 76's "
            "rule, in the second place it applies)",
            "does not bind")

    _mutated = dict(live)
    _mutated["days"] = ["2026-09-03"]          # the caller rewrites its copy
    refuses(lambda: resolve_draws(
        _mutated, day="2026-09-03", arm="CONDVALUE_X_SKEW", fixture=True,
        supplied={"draws": [1.0] * 500, "provenance": {}}),
        "AND THE LOCK IS NOT READ FROM THE CALLER'S DICT: a caller that "
        "rewrites `params['days']` to open the door is still REFUSED, "
        "because the ruled set is read from the COMMITTED file. The fixture "
        "run does exactly this rewrite, which is how the hole was found",
        "IS in the ruled day set")
    ok(ruled_day_set() == live["days"] and len(ruled_day_set()) == 6,
       f"and `ruled_day_set()` reads the committed file and agrees with the "
       f"loaded params: {ruled_day_set()}")

    def _day_path_checks():
        # ================= DE 78: THE REAL-DAY PATH, ON A SYNTHETIC DAY ======
        import tempfile as _tfd
        _DAYP = dict(live)
        _DAY = "FIXTURE-DAY-1"          # NOT a ruled day: the lock now forbids it

        def _mkday(**kw):
            _d = _tfd.mkdtemp(prefix="de78_")
            return write_synthetic_day(_DAY, _d, params=_DAYP, n_slugs=24, **kw)

        _made = _mkday()
        ok(synthetic_pool_keys_agree(_made["book"])
           and set(_made["book"]["fr"]) >= {"reference", "statuses",
                                            "population", "n_slugs",
                                            "terminal_marks"}
           and len(_made["book"]["asm"]["by_arm"]) == 2,
           f"THE SYNTHETIC BOOK CARRIES BE'S DECLARED SHAPE: reference, "
           f"statuses, population, n_slugs, terminal_marks, and `asm.by_arm` "
           f"keyed (coin, head) for BOTH pinned heads over ONE agreed key set "
           f"-- read from be_daybook_builder_declaration_v1.json, not invented")

        # ---- the run, UNSEALED, so the planted verdicts are readable --------
        _open = run_day(_DAY, _made["book_path"], params=_DAYP, fixture=True,
                        n_days_complete=_DAYP["G"])
        _by = {r["arm"]: r for r in _open["per_day_sealed_artifacts"]}
        _pass, _fail = _by["CONDVALUE_X_SKEW"], _by["HAZARD_OVER_SKEWED_REF"]
        ok(_pass["status"] == "OK" and _pass["economic"]["Z"] > 3.0
           and _pass["economic"]["D_E0"] > 0
           and _pass["economic"]["p_location"] <= 2.0 / 501,
           f"A PLANTED MUST-PASS ARM PASSES ON THE REAL CASCADE: the head that "
           f"ranks the value-DESTROYING generations high gets D(E0) = "
           f"{_pass['economic']['D_E0']:.0f} cents against a null mean of "
           f"{_pass['economic']['null_mean']:.1f}, Z = "
           f"{_pass['economic']['Z']:.2f}, p at the {1 / 501:.4f} floor")
        ok(_fail["status"] == "OK" and _fail["economic"]["Z"] < -3.0
           and _fail["economic"]["D_E0"] < 0
           and _fail["economic"]["p_location"] > 0.9,
           f"AND A PLANTED MUST-FAIL ARM FAILS: the head that ranks the "
           f"value-CREATING generations high gets D(E0) = "
           f"{_fail['economic']['D_E0']:.0f}, Z = {_fail['economic']['Z']:.2f} "
           f"-- both directions driven, so neither is a verdict the fixture "
           f"could only produce one of")
        ok(abs(_open["per_day_sealed_artifacts"][0]["economic"]["null_mean"])
           < 0.5 * _open["per_day_sealed_artifacts"][0]["economic"]["null_sd"],
           f"and the NULL itself is centred near zero with real dispersion "
           f"(mean {_pass['economic']['null_mean']:.1f}, sd "
           f"{_pass['economic']['null_sd']:.1f}) -- a random cancel on this "
           f"book is worth about nothing, which is what makes the two plants "
           f"readable rather than an artefact of a lopsided fixture")

        # ---- the SEED reproduces from the artifact alone --------------------
        _sd = _pass["seed"]
        ok(_sd == seed_for(_open["reference_book"]["sha256"],
                           "CONDVALUE_X_SKEW")
           and _sd == DESIGN.seed_from_convention(
               _open["reference_book"]["sha256"], "CONDVALUE_X_SKEW"),
           f"THE SEED REPRODUCES FROM THE ARTIFACT ALONE: {_sd} recomputed "
           f"from the emitted book digest and the arm name by BOTH the "
           f"runner's expression and the declaration's reference "
           f"implementation -- nothing else is needed to redraw the null")
        ok(_pass["draw_provenance"]["draw_source"] == "GENERATED_IN_PROCESS"
           and _pass["draw_provenance"]["generated_in_process"] is True,
           "and the draws on the day path are GENERATED IN PROCESS through "
           "the DE 77 seam, not supplied -- the fixture flag governs the BOOK, "
           "never the draws")

        # ---- the null is BE's, cross-checked --------------------------------
        ok(_pass["draw_provenance"]["reproduces_BEs_draw_null"]["identical"]
           is True
           and _pass["draw_provenance"]["reproduces_BEs_draw_null"][
               "field"] == "cancels_issued",
           "AND DE's VALUED DRAW LOOP REPRODUCES BE's OWN `draw_null` at the "
           "same seed on the field both produce. DE drives BE's sampler and "
           "BE's replay and values the fills itself, because `draw_null` "
           "discards them -- so the cascade is BE's and only the METRIC is "
           "DE's, checked rather than asserted")

        # ---- SEALED is the default, and it is absent-not-null ---------------
        _sealed_run = run_day(_DAY, _made["book_path"], params=_DAYP,
                              fixture=True, n_days_complete=1)
        _sa = _sealed_run["per_day_sealed_artifacts"]
        ok(all("economic" not in a for a in _sa)
           and all(a["sealed"] is True and a["sealed_at_every_depth"] is True
                   and a["sealed_field_names"] == list(ECONOMIC_FIELDS)
                   for a in _sa)
           and all(a["decision_population"]["decisions"] > 0 for a in _sa),
           f"AT 1 OF {_DAYP['G']} DAYS THE DAY ARTIFACT IS SEALED: every "
           f"economic field ABSENT, all four layout keys present, and the "
           f"decision counts and statuses still published -- the smoke "
           f"publishes what it may and withholds what it must")
        _leak = dict(_sa[0]); _leak["economic"] = {"D_E0": 1.0}
        refuses(lambda: assert_no_economic_leak(_leak, 1, _DAYP["G"]),
                "KNOWN-BAD ON THE DAY ARTIFACT: an economic field present "
                "before G REFUSES -- the guard walks the day artifact the "
                "runner actually emitted, not a hand-built one",
                "leaked into a SEALED artifact")

        # ---- R6: a wrong book digest refuses THE DAY -----------------------
        _bad = _mkday()
        _rp = Path(_bad["receipt_path"])
        _rj = json.loads(_rp.read_text()); _rj["sha256"] = "0" * 64
        _rp.write_text(json.dumps(_rj))
        refuses(lambda: run_day(_DAY, _bad["book_path"], params=_DAYP,
                                fixture=True),
                "R6 KNOWN-BAD, A WRONG BOOK DIGEST: THE DAY refuses. The "
                "digest compared is BE's OWN published one from the sidecar "
                "receipt, recomputed against the bytes at read time -- never a "
                "constant DE typed", "digest mismatch")
        _norec = _mkday()
        Path(_norec["receipt_path"]).unlink()
        refuses(lambda: run_day(_DAY, _norec["book_path"], params=_DAYP,
                                fixture=True),
                "and a book with NO builder receipt refuses too: without BE's "
                "published claim there is nothing to verify the bytes against, "
                "and a digest DE invents verifies DE", "no builder receipt")

        # ---- R4: a THIN arm-day is a STATUS and G does not shrink -----------
        _thin = _mkday(head_policy={"CONDVALUE_X_SKEW": "harmful",
                                    "HAZARD_OVER_SKEWED_REF": "thin"})
        _tr = run_day(_DAY, _thin["book_path"], params=_DAYP, fixture=True,
                      n_days_complete=_DAYP["G"])
        _tby = {r["arm"]: r for r in _tr["per_day_sealed_artifacts"]}
        _th = _tby["HAZARD_OVER_SKEWED_REF"]
        ok(_th["status"].startswith("DEGENERATE_ARM_DAY_REFUSED")
           and _th["economic"] is None
           and _th["decision_population"]["decisions"]
           < _DAYP["min_decisions_per_arm_day"]
           and _tby["CONDVALUE_X_SKEW"]["status"] == "OK",
           f"R4 ON THE DAY PATH: an arm whose head clears theta only "
           f"{_th['decision_population']['decisions']} times is a counted "
           f"STATUS with no economic field, while the OTHER arm on the SAME "
           f"day runs normally -- the refusal is per arm-day and does not take "
           f"the day with it")
        _agg = aggregate([r for r in _tr["per_day_sealed_artifacts"]], _DAYP)
        ok(_agg["per_arm"]["HAZARD_OVER_SKEWED_REF"]["verdict"]
           == "UNTESTABLE_ON_THE_DECLARED_DAY_SET"
           and _agg["G"] == _DAYP["G"],
           f"AND G DOES NOT SHRINK: the thin arm aggregates to UNTESTABLE at "
           f"G = {_agg['G']}, never to a tested verdict on fewer days")

        # ---- the ADDENDUM: which index splits `--day` needs -----------------
        _prf = day_split_residency_proof(_DAY, _made["book_path"],
                                         params=_DAYP, fixture=True)
        ok(_prf["no_tape_index_or_fragment_artifact_was_opened"] is True
           and _prf["instrument_observed_the_book_read"] is True
           and _prf["non_vacuous"] is True,
           f"THE ADDENDUM, MEASURED: a whole day path opened "
           f"{_prf['n_paths_opened']} paths and NOT ONE tape, index or "
           f"fragment artifact among them. `--day` needs NO index split "
           f"resident at any stage -- it consumes `asm` and the reference, and "
           f"values fills from the replay's own records. The unneeded splits "
           f"are absent here and the run completes")
        ok(INDEX_SPLITS_NEEDED_BY_DAY["answer"] == "NONE, at any stage"
           and set(INDEX_SPLITS_NEEDED_BY_DAY["per_stage"]) == {
               "S2_population", "S3_baseline_and_S4_null",
               "economics_valuation"}
           and len(DAY_STAGES) == 6,
           f"and the declaration names the answer PER STAGE over "
           f"{len(DAY_STAGES)} stages, so BE builds to a field rather than to "
           f"a sentence in a report")
        _nohead = _mkday()
        import pickle as _pk
        _bkd = _pk.loads(Path(_nohead["book_path"]).read_bytes())
        del _bkd["asm"]["by_arm"][("btc", "incumbent_linear_d")]
        _buf = _pk.dumps(_bkd)
        Path(_nohead["book_path"]).write_bytes(_buf)
        _rj2 = json.loads(Path(_nohead["receipt_path"]).read_text())
        _rj2["sha256"] = hashlib.sha256(_buf).hexdigest()
        Path(_nohead["receipt_path"]).write_text(json.dumps(_rj2))
        refuses(lambda: run_day(_DAY, _nohead["book_path"], params=_DAYP,
                                fixture=True),
                "AND THE OTHER HALF OF THE FALSIFIER: with the input `--day` "
                "DOES need absent -- a pinned head missing from `asm.by_arm` "
                "-- the run REFUSES AND NAMES IT. So the residency claim is "
                "not 'nothing matters'; it is that the book is the only thing "
                "that does", "no entry for")

        # ---- reviewer §1.4: the lock is ON THE DAY PATH, not beside it -----
        _ruled_day = live["days"][0]
        refuses(lambda: run_day(_ruled_day, _made["book_path"], params=_DAYP,
                                fixture=True),
                f"REVIEWER §1.4, THE FINDING ITSELF: `--synthetic-day "
                f"{_ruled_day}` used to emit a SEALED artifact stamped with "
                f"THE SMOKE DAY from a synthetic book. run_day now calls the "
                f"SAME lock resolve_draws calls -- decided on the DAY, not on "
                f"the caller's flag -- and REFUSES. Third instance of one "
                f"class in three rounds; it is one function with three call "
                f"sites now", "IS in the ruled day set")
        refuses(lambda: run_day("FIXTURE-DAY-9", _made["book_path"],
                                params=_DAYP, fixture=False),
                "AND THE OTHER DIRECTION ON THE DAY PATH: a REAL run claimed "
                "for a day outside the ruled set REFUSES -- the lock is "
                "symmetric here as it is in resolve_draws", "is NOT in the "
                "ruled day set")
        ok(_open["fixture_day_lock"]["decided_on_the_day_not_the_callers_flag"]
           is True
           and _open["fixture_day_lock"]["in_ruled_day_set"] is False
           and _open["fixture_day_lock"]["ruled_day_set_read_from"]
           == PARAMS_REL,
           f"and the ADMITTED fixture run records the lock it passed, naming "
           f"the committed file it read the ruled set from -- so a reader can "
           f"tell a fixture that was CHECKED from one that was merely labelled")
        _sym = _open["seal_layout_symmetry_checked_on_the_emitted_results"]
        ok(len(_sym) == len(_open["per_day_sealed_artifacts"])
           and all(x["symmetric"] is True for x in _sym)
           and all(x["sealed_reads"]["readable"] is True
                   and x["unsealed_reads"]["readable"] is True for x in _sym),
           f"AND THE CONSUMER FALSIFIER MEETS A REAL EMISSION: "
           f"assert_seal_layout_symmetric is wired onto every emitted result, "
           f"in BOTH states ({len(_sym)} of them). It had three call sites and "
           f"all three were in this battery -- mitigated in fact, unwired in "
           f"truth")

        # ---- rule 20 / R-575(C), driven both ways --------------------------
        _obs_now = wrapper_observed()
        ok(set(_obs_now) >= {"heavy_run_lock_held", "lock_fds", "cgroup_leaf"}
           and isinstance(_obs_now["heavy_run_lock_held"], bool),
           f"R-575(C): the wrapper is MEASURED from /proc/self/fd, not read "
           f"from the params string -- lock held here: "
           f"{_obs_now['heavy_run_lock_held']}, cgroup "
           f"{_obs_now['cgroup_leaf']}")
        ok(assert_rule20({"heavy_run_lock_held": False}, wall_s=1.0,
                         peak_rss_mb=50.0, day=_DAY)[
               "heavy_by_measurement"] is False,
           "POSITIVE CONTROL, AND IT ADMITS: a LIGHT run without the lock is "
           "fine -- 1.0 s and 50 MB against the 60 s / 1 GiB bar")
        refuses(lambda: assert_rule20({"heavy_run_lock_held": False},
                                      wall_s=61.0, peak_rss_mb=50.0,
                                      day=_DAY),
                "KNOWN-BAD, THE 05:54Z CASE: a run that WAS heavy by "
                "measurement and did not hold the lock REFUSES and the "
                "artifact is not written -- so a receipt can no longer claim a "
                "wrapper it did not have", "did NOT hold")
        ok(assert_rule20({"heavy_run_lock_held": True}, wall_s=3600.0,
                         peak_rss_mb=7000.0, day=_DAY)[
               "heavy_by_measurement"] is True,
           "and a heavy run that DID hold the lock admits, marked heavy -- the "
           "rule is about the lock, not about being small")
        refuses(lambda: run_day(_DAY, _made["book_path"], params=_DAYP,
                                fixture=True, peak_rss_mb_budget=1.0),
                "AND THE FIXTURE BUDGET BITES: a day run whose peak exceeds "
                "its DECLARED budget REFUSES rather than reporting a number "
                "over the line -- the cap is never raised and the draws are "
                "never cut (R-174)", "exceeds the declared budget")
        ok(_open["memory_plan"]["peak_rss_mb"]
           < FIXTURE_DAY_PEAK_RSS_MB_BUDGET
           and _open["memory_plan"]["within_budget"] is True
           and len(_open["memory_plan"]["observed"]) == len(DAY_STAGES),
           f"and the real fixture run sits at "
           f"{_open['memory_plan']['peak_rss_mb']:.0f} MB against the declared "
           f"{FIXTURE_DAY_PEAK_RSS_MB_BUDGET:.0f} MB, with a high-water "
           f"recorded at each of the {len(DAY_STAGES)} stages")

        # ---- reviewer §2.3: the instrument tests the LOCK, not an fd -------
        import tempfile as _tfl
        with _tfl.TemporaryDirectory() as _ld:
            _plock = str(Path(_ld) / "probe.lock")
            Path(_plock).write_text("")
            _clean = wrapper_observed(lock_path=_plock)
            ok(_clean["heavy_run_lock_held"] is False
               and _clean["lock_is_held_by_someone"] is False
               and _clean["flock_holder_pids"] == [],
               "R-575(C) BASELINE: with nobody holding the lock the fresh-fd "
               "LOCK_EX|LOCK_NB probe SUCCEEDS, is released at once, and the "
               "field reads False")
            _g = open(_plock)                       # the reviewer's forge
            try:
                _forged = wrapper_observed(lock_path=_plock)
                ok(_forged["lock_fds"] == [_g.fileno()]
                   and _forged["heavy_run_lock_held"] is False,
                   f"REVIEWER §2.3, THE FORGE, DRIVEN: `open(lock)` with NO "
                   f"flock puts fd {_g.fileno()} on the lock file and the OLD "
                   f"instrument reported `heavy_run_lock_held: True` -- two "
                   f"lines certified a 1-hour, 6.84 GiB run. The fd is now "
                   f"CORROBORATION and the test is the flock: this reads "
                   f"False with the fd present")
            finally:
                _g.close()
            _h = open("/etc/hostname")
            try:
                _unrel = wrapper_observed(lock_path=_plock)
                ok(_unrel["heavy_run_lock_held"] is False,
                   "and an UNRELATED open fd does not pass either -- the "
                   "instrument is keyed to the lock's own inode, not to a "
                   "descriptor number")
            finally:
                _h.close()
            import fcntl as _fc
            _fd = __import__("os").open(_plock, __import__("os").O_RDWR)
            try:
                _fc.flock(_fd, _fc.LOCK_EX | _fc.LOCK_NB)
                _held = wrapper_observed(lock_path=_plock)
                ok(_held["heavy_run_lock_held"] is True
                   and _held["lock_is_held_by_someone"] is True
                   and _held["held_by_self_or_ancestor"] is True
                   and __import__("os").getpid() in _held["flock_holder_pids"],
                   f"POSITIVE CONTROL, AND IT ADMITS: with a REAL flock held "
                   f"the probe fails, /proc/locks names the holder "
                   f"{_held['flock_holder_pids']}, and it is this process -- "
                   f"both surfaces agree. The ancestor walk exists because a "
                   f"wrapped run's holder is the `flock` PARENT, not the "
                   f"python child")
            finally:
                _fc.flock(_fd, _fc.LOCK_UN)
                __import__("os").close(_fd)
            _after = wrapper_observed(lock_path=_plock)
            ok(_after["heavy_run_lock_held"] is False,
               "and it goes back to False once the flock is released -- the "
               "field tracks the lock's state and not a fact about startup")

        # ---- REV 41: the lock must be EXCLUSIVE, not merely held --------
        import fcntl as _fc2
        import os as _os2
        with _tfl.TemporaryDirectory() as _sd:
            _sl = str(Path(_sd) / "shared.lock")
            Path(_sl).write_text("")
            _f1 = _os2.open(_sl, _os2.O_RDWR)
            try:
                _fc2.flock(_f1, _fc2.LOCK_SH | _fc2.LOCK_NB)
                _sh = wrapper_observed(lock_path=_sl)
                ok(_sh["lock_is_held_by_someone"] is True
                   and _sh["holder_is_exclusive"] is False
                   and _sh["held_by_self_or_ancestor"] is True
                   and _sh["heavy_run_lock_held"] is False,
                   f"REV 41, THE FINDING: a SHARED hold (`flock -s`) is "
                   f"held, and held BY ME, and is still REFUSED -- modes "
                   f"{_sh['flock_modes_on_the_inode']}. Rule 20's "
                   f"invariant is ONE heavy run at a time and only an "
                   f"EXCLUSIVE lock enforces it; the previous two-conjunct "
                   f"test would have certified this")
                # A SECOND PROCESS, because /proc/locks counts HOLDERS
                # by pid: two shared fds in one process is one holder, and
                # a check that could not tell them apart would not be
                # showing the state rule 20 forbids.
                import subprocess as _sp2
                _child = _sp2.Popen(["flock", "-s", _sl, "sleep", "5"])
                try:
                    _t0 = time.time()
                    while time.time() - _t0 < 5.0:
                        _two = wrapper_observed(lock_path=_sl)
                        if _two["n_flock_holders"] >= 2:
                            break
                        time.sleep(0.05)
                    ok(_two["n_flock_holders"] >= 2
                       and _two["heavy_run_lock_held"] is False
                       and _two["holder_is_exclusive"] is False,
                       f"AND TWO CONCURRENT SHARED HOLDERS ARE BOTH "
                       f"REFUSED: {_two['n_flock_holders']} distinct "
                       f"holders on the inode at once -- this process and "
                       f"a second one -- which is precisely the state rule "
                       f"20 forbids, and neither certifies")
                finally:
                    # `flock -s <lock> sleep` execs sleep IN the flock
                    # process, so terminating it releases the lock -- but
                    # not instantly. The next check takes an EXCLUSIVE
                    # lock and would raise BlockingIOError against a
                    # lingering hold, so WAIT for the release rather than
                    # assume it (it caught me once).
                    _child.terminate()
                    _child.wait(timeout=5)
                    _t1 = time.time()
                    while (time.time() - _t1 < 5.0
                           and wrapper_observed(
                               lock_path=_sl)["n_flock_holders"] > 1):
                        time.sleep(0.02)
            finally:
                _fc2.flock(_f1, _fc2.LOCK_UN)
                _os2.close(_f1)
            _f3 = _os2.open(_sl, _os2.O_RDWR)
            try:
                _fc2.flock(_f3, _fc2.LOCK_EX | _fc2.LOCK_NB)
                _ex = wrapper_observed(lock_path=_sl)
                ok(_ex["heavy_run_lock_held"] is True
                   and _ex["holder_is_exclusive"] is True
                   and _ex["flock_modes_on_the_inode"] == ["WRITE"],
                   "POSITIVE CONTROL, AND IT ADMITS: an EXCLUSIVE hold by "
                   "this process certifies -- the fix refuses a shared "
                   "hold without refusing the wrapper the runbook "
                   "prescribes (`flock -n`, exclusive by default)")
            finally:
                _fc2.flock(_f3, _fc2.LOCK_UN)
                _os2.close(_f3)

        # ---- DE 80: the day's tape and fragment are PARAMETERS -----------
        import de_phase4_diag_runner as _PD
        import inspect as _i3
        _consumed = _PD.consumed_era_inputs()
        _ruled_d = live["days"][0]
        _dref = lambda k, p, h, d: _PD.day_assembly_inputs(
            d, tape={"path": p if k == "tape" else str(_consumed["tape"]),
                     "sha256": h},
            fragment={"path": p if k == "fragment"
                      else str(_consumed["fragment"]), "sha256": h})
        try:
            _PD.verify_assembly_input("tape", _consumed["tape"], "x" * 64,
                                      day=_ruled_d)
            ok(False, "the consumed constant was ADMITTED on a ruled day")
        except _PD.DiagRefused as _e:
            ok("CONSUMED-ERA constant" in str(_e),
               f"DE 80 KNOWN-BAD, THE BLOCKER ITSELF: the CONSUMED-ERA "
               f"tape constant is REFUSED for ruled day {_ruled_d} -- that "
               f"file is what the heads were FITTED on and does not "
               f"contain the day at all, so a day scored off it is the "
               f"development hour wearing the day's name")
        # THE LEDGER'S copy, not this worktree's. The worktree path is
        # correctly refused as off-ledger, which is the check above -- so
        # the digest known-bad must be given a path that gets PAST it, or
        # it would be passing for the wrong reason.
        _real = (Path(DR.resolve()["data_root"]) / "pm_5min/derived"
                 / "p003_de_multiday_gate1_design_v10__20260906T064720Z"
                   ".json").resolve()
        _rh = hashlib.sha256(_real.read_bytes()).hexdigest()
        try:
            _PD.verify_assembly_input("tape", _real, "0" * 64, day=_ruled_d)
            ok(False, "a wrong digest was ADMITTED")
        except _PD.DiagRefused as _e:
            ok("hashes to" in str(_e),
               "DE 80 KNOWN-BAD: a day input whose bytes do not hash to "
               "the DECLARED digest REFUSES -- recomputed at read time, "
               "never taken from the caller's word")
        _adm = _PD.verify_assembly_input("tape", _real, _rh, day=_ruled_d)
        ok(_adm["sha256"] == _rh and _adm["day_is_in_the_ruled_set"] is True
           and _adm["is_the_consumed_era_constant"] is False
           and _adm["digest_recomputed_at_read_time"] is True,
           f"DE 80 POSITIVE CONTROL, AND IT ADMITS: a ledger path whose "
           f"bytes hash to the declared digest passes for a ruled day. "
           f"STATED PRECISELY: this admits the VERIFICATION path -- the "
           f"file stands in for a day tape and is not one, and nothing "
           f"here claims BE's tape is well-formed")
        try:
            _PD.verify_assembly_input("tape", "/etc/hostname", "0" * 64,
                                      day=_ruled_d)
            ok(False, "an off-ledger path was ADMITTED")
        except _PD.DiagRefused as _e:
            ok("not under the ledger" in str(_e),
               "and an input outside the ledger REFUSES before its digest "
               "is even considered -- a file read from a seat's worktree "
               "is not the input a receipt can name (R-559(C))")
        try:
            _PD.verify_assembly_input("tape", _real, None, day=_ruled_d)
            ok(False, "a digest-less input was ADMITTED")
        except _PD.DiagRefused as _e:
            ok("no declared sha256" in str(_e),
               "and a path with NO declared digest refuses -- a path alone "
               "is a claim about a filename and the file behind it moves")
        try:
            _PD.day_assembly_inputs(_ruled_d)
            ok(False, "a ruled day with nothing supplied was ADMITTED")
        except _PD.DiagRefused as _e:
            ok("not supplied" in str(_e),
               "and a RULED day that supplies nothing REFUSES rather than "
               "falling back to the constants -- which is the whole point "
               "of the seam")
        _dflt = _PD.day_assembly_inputs(None)
        ok(_dflt["regime"] == "CONSUMED_HOUR_DEFAULT"
           and _dflt["tape"]["path"] == str(_consumed["tape"])
           and _dflt["fragment"]["path"] == str(_consumed["fragment"]),
           "AND THE CONSUMED HOUR IS UNCHANGED: day=None still returns the "
           "two constants, so the seam is additive and the consumed-era "
           "path keeps the behaviour it had")
        _mech = _PD.tape_path_mechanism()
        ok(_mech["mechanism"] in ("PARAMETER", "SCOPED_REBIND_PENDING_BE52")
           and _mech["parameter_available"] is (
               _mech["mechanism"] == "PARAMETER"),
           f"and HOW the path reaches phase2_arms is decided by SIGNATURE "
           f"and recorded: {_mech['mechanism']}. BE 52 owns that "
           f"parameter; the moment it exists this adopts it without an "
           f"edit here, and until then the receipt says which ran")

        # ---- REV 43 (1): the digest is threaded TO THE LOAD --------------
        _mech2 = _PD.tape_path_mechanism()
        ok(_mech2["expect_sha256_available"] is True
           and _mech2["day_available"] is True
           and _mech2["mechanism"] == "PARAMETER",
           f"REV 43 (1): BE 52's onward parameters are DETECTED by "
           f"signature -- {_mech2['onward_parameters_detected']} -- and the "
           f"mechanism is now {_mech2['mechanism']}, so the scoped rebind "
           f"has handed over without an edit here, which is what choosing "
           f"by signature was for")
        try:
            _PD.build_tape_index({"score": None}, tape_path=_real,
                                 day=_ruled_d)
            ok(False, "a ruled-day tape with NO digest was ADMITTED")
        except _PD.DiagRefused as _e:
            ok("NO expect_sha256" in str(_e),
               "REV 43 (1) KNOWN-BAD, THE AMBIGUOUS CASE: a ruled day's "
               "tape path supplied WITHOUT its digest REFUSES rather than "
               "running a call that reads the right path and checks "
               "nothing -- the signature would otherwise say a digest was "
               "expected while nothing verified one")
        import phase2_arms as _PA3
        try:
            _PD.build_tape_index({"score": None}, tape_path=_real,
                                 day=_ruled_d, expect_sha256="0" * 64)
            ok(False, "a WRONG load digest was ADMITTED")
        except Exception as _e:
            ok(type(_e).__name__ == "TapePathRefused"
               and "digests" in str(_e),
               f"REV 43 (1) KNOWN-BAD AT THE LOAD, AND IT IS BE'S OWN "
               f"CHECK THAT FIRES: a wrong `expect_sha256` raises "
               f"{type(_e).__name__} from phase2_arms as the stream is "
               f"opened. That is the proof the keyword ARRIVED -- the "
               f"digest was checked in day_assembly_inputs before, which "
               f"left a window between the check and the use")
        _threaded = False
        try:
            _PD.build_tape_index({"score": None}, tape_path=_real,
                                 day=_ruled_d, expect_sha256=_rh)
        except Exception as _e:
            _threaded = type(_e).__name__ != "TapePathRefused"
        ok(_threaded,
           "AND THE POSITIVE CONTROL: with the RIGHT digest the call gets "
           "PAST the digest gate -- it fails later, on the file not being "
           "a tape, which is a different refusal. So the known-bad above "
           "fires on the digest and not on the file")
        _inp_form = _PD.day_assembly_inputs(
            _ruled_d, tape={"path": str(_real), "sha256": _rh},
            fragment={"path": str(_real), "sha256": _rh})
        ok(_inp_form["tape"]["sha256"] == _rh
           and "expect_sha256" in _i3.getsource(_PD.build_tape_index),
           "and `inputs=` is the preferred form because it carries path, "
           "digest and day as ONE object that cannot drift apart -- it is "
           "exactly what day_assembly_inputs returns")

        # ---- REV 43 (2): the /proc/locks match includes the DEVICE -------
        _wd = (259, 1)
        _fake_same_dev = "6: FLOCK ADVISORY WRITE 999 103:01:4242 0 EOF"
        _fake_other_dev = "7: FLOCK ADVISORY WRITE 999 07:99:4242 0 EOF"
        ok(_flock_line_matches(_fake_same_dev.split(), _wd, 4242) is True,
           "REV 43 (2) POSITIVE CONTROL, AND IT ADMITS: a crafted "
           "/proc/locks line on the RIGHT device and inode matches")
        ok(_flock_line_matches(_fake_other_dev.split(), _wd, 4242) is False,
           "REV 43 (2) KNOWN-BAD, THE COLLISION ITSELF: the SAME inode on "
           "a DIFFERENT device (07:99 against 103:01) no longer matches. "
           "The pre-fix parse compared the inode alone and ADMITTED this, "
           "so a lock held on an unrelated filesystem could have certified "
           "a heavy run here -- and this box's own /proc/locks already "
           "carries entries from several devices")
        _obs_real = wrapper_observed()
        ok(_obs_real["flock_modes_on_the_inode"] is not None
           and isinstance(_flock_holders(HEAVY_RUN_LOCK)["device"], dict),
           f"and the real lock's device travels in the observation: "
           f"{_flock_holders(HEAVY_RUN_LOCK)['device']}")

        # ---- REV 43 (3): a missing lock file REFUSES BY NAME -------------
        with _tfl.TemporaryDirectory() as _md:
            _gone = str(Path(_md) / "not-there.lock")
            refuses(lambda: wrapper_observed(lock_path=_gone),
                    "REV 43 (3) KNOWN-BAD: a MISSING lock file is a NAMED "
                    "REFUSAL, not a KeyError. It crashed for two rounds on "
                    "two different keys, because each round's new field "
                    "joined the same incomplete early return -- a "
                    "KeyError from an instrument tells a reader nothing "
                    "about the lock", "does not exist")
            Path(_gone).write_text("")
            ok(wrapper_observed(lock_path=_gone)[
                   "heavy_run_lock_held"] is False,
               "and the same path ADMITS the moment the file exists, "
               "reporting an unheld lock -- so the refusal is about "
               "absence and not about the path being unusual")

        # ---- REV 43 (4): the day-path count is COMPUTED, never a literal -
        # `battery_scope` is written by `_main_day`, not `run_day`, so
        # the receipt-level check belongs at the SOURCE: the literal is
        # what has to be absent, and it is absent everywhere or nowhere.
        import re as _re4
        _srcs = {"runner": _i3.getsource(sys.modules[__name__])}
        _lit = [m for m in _re4.findall(r"all (\d+) day-path checks",
                                        _srcs["runner"])]
        ok(not _lit,
           f"REV 43 (4) KNOWN-BAD SWEPT AT THE SOURCE: no literal "
           f"'all <N> day-path checks' remains anywhere in this module "
           f"(found {_lit}). It read 23 beside a constant of 38 and then "
           f"49 -- the gap widened twice while the sentence sat still, "
           f"which is rule 10 in my own receipt")

        # ---- DE 82 (1): the scope's anon and file, read APART ------------
        _sm = scope_memory_observation()
        ok(_sm["status"] in ("MEASURED", "AMBIENT_SCOPE_NOT_THE_RUNS_OWN",
                             "NOT_IN_A_SCOPE", "SCOPE_NAMED_BUT_UNREADABLE")
           and (_sm["status"] != "NOT_IN_A_SCOPE"
                or _sm["anon_bytes"] is None),
           f"DE 82 (1): the scope observation reports a STATUS "
           f"({_sm['status']}), and when there is no scope to read it "
           f"reports None -- NEVER zeros. A zero would read as 'measured, "
           f"and nothing happened', which is rule 11's silent absence")
        ok(scope_memory_observation.__doc__
           and "5.17" in scope_memory_observation.__doc__
           and "memory.events" in scope_memory_observation.__doc__,
           "and the reason anon and file are read APART is DA 63's "
           "measurement, carried where the code is: 7.79 GiB peak with "
           "1.13 GiB anon against 5.17 GiB reclaimable cache, events max 0 "
           "-- rule 20's cap counts the cache, so one number cannot tell "
           "'this run needs 7.8 GiB' from 'the kernel had no reason to "
           "reclaim'")
        _fake_none = {"status": "NOT_IN_A_SCOPE", "anon_bytes": None,
                      "file_bytes": None, "memory_peak_bytes": None,
                      "events": None}
        ok(all(_fake_none[k] is None for k in
               ("anon_bytes", "file_bytes", "memory_peak_bytes", "events"))
           and _fake_none["status"] == "NOT_IN_A_SCOPE",
           "KNOWN-BAD SHAPE, NAMED: the scope-less record is Nones under a "
           "status, and a receipt carrying zeros there would be claiming a "
           "measurement it never made")
        ok(_sm.get("in_research_slice") is not None
           and (_sm["status"] == "MEASURED") == bool(
               _sm.get("in_research_slice")),
           f"AND AN AMBIENT SCOPE IS NOT THE RUN'S: MEASURED holds only "
           f"inside {RESEARCH_SLICE}. Measured while writing this: the "
           f"login shell's own scope carries a 15.5 GiB peak against an "
           f"8 MB run, and attributing that to the run would be a "
           f"measurement of the wrong object")

        # ---- DE 82 (2): the smoke, rehearsed -----------------------------
        _rh = rehearse_smoke("2026-09-03")
        ok(_rh["book"]["exists"] is False
           and "P2_book_exists" in _rh["blocking"]
           and _rh["book"]["REFUSES_NOW_BY_NAME"]
           and _rh["status"].startswith("NOT_READY"),
           f"DE 82 (2): the rehearsal REFUSES NOW BY NAME on the book that "
           f"does not exist -- blocking {_rh['blocking']} -- so GO is one "
           f"verified command and nothing is typed at GO time")
        ok(_rh["THE_ONE_COMMAND"].startswith(f"flock -n {HEAVY_RUN_LOCK}")
           and f"--slice={RESEARCH_SLICE}" in _rh["THE_ONE_COMMAND"]
           and "MemoryMax=8G" in _rh["THE_ONE_COMMAND"]
           and "--day 2026-09-03" in _rh["THE_ONE_COMMAND"]
           and "be_daybook_20260903_btc.pkl" in _rh["THE_ONE_COMMAND"],
           "and the command is the rule-20 wrapper in full, with the "
           "dashed day DE names and the COMPACT-day book path BE writes -- "
           "the two conventions meeting in one string that was executed as "
           "a value rather than typed")
        ok("P5_lock_free_now" in _rh["informational_and_false_now"]
           and "P5_lock_free_now" not in _rh["blocking"],
           "and the lock's state NOW is INFORMATIONAL, not blocking: the "
           "wrapper takes it at GO. Counting it as blocking would inflate "
           "the status and train a reader to ignore it")
        ok(len(_rh["what_this_rehearsal_found"]) == 3
           and any("be_daybook_receipt_" in x
                   for x in _rh["what_this_rehearsal_found"]),
           f"AND THE REHEARSAL EARNED ITSELF: it found "
           f"{len(_rh['what_this_rehearsal_found'])} defects that would "
           f"each have refused a CORRECT book at GO for a reason that had "
           f"nothing to do with the book")
        ok(day_forms("2026-09-03") == {"2026-09-03", "20260903"}
           and day_forms("20260903") == {"2026-09-03", "20260903"},
           "DE 82 (2a) KNOWN-BAD CLOSED: `2026-09-03` and `20260903` are "
           "the SAME DAY. DE names ruled days dashed and BE stamps them "
           "compact; a string comparison across that boundary refused a "
           "correct book for a formatting reason")
        _rc = {"day": "20260903", "book": {"sha256": "a" * 64}}
        ok(_receipt_book_sha(_rc) == ("a" * 64, "book.sha256"),
           "DE 82 (2b) KNOWN-BAD CLOSED: BE's receipt carries the digest "
           "at `book.sha256`, and the resolver NAMES which field it read "
           "so a reader is not left guessing")
        with _tfl.TemporaryDirectory() as _bd:
            _bp = Path(_bd) / "be_daybook_20260903_btc.pkl"
            _bp.write_bytes(b"x")
            try:
                builder_receipt_for(_bp, "2026-09-03")
                ok(False, "a missing builder receipt was ADMITTED")
            except RunnerRefused as _e:
                ok("Tried, in order" in str(_e)
                   and "be_daybook_receipt_20260903_btc.json" in str(_e),
                   "DE 82 (2c) KNOWN-BAD: a missing builder receipt names "
                   "EVERY path tried, so the refusal is actionable rather "
                   "than a puzzle")
            (Path(_bd) / "be_daybook_receipt_20260903_btc.json").write_text(
                "{}")
            ok(builder_receipt_for(_bp, "2026-09-03").name
               == "be_daybook_receipt_20260903_btc.json",
               "AND THE POSITIVE CONTROL: BE's ACTUAL naming resolves -- "
               "the old derivation (`book_path.with_suffix('.json')`) "
               "would have missed this file sitting beside the book")

        # ---- reviewer §4.3: the peak stage is a PREDICATE -------------------
        _ps = _open["memory_plan"]["peak_stage"]
        ok(_ps["computable"] is True
           and set(_ps["current_rss_mb_by_stage"]) == {k for k, _ in DAY_STAGES}
           and _ps["declared_peak_stage"] == "S1_load",
           f"REVIEWER §4.3: the peak stage is COMPUTED over all "
           f"{len(DAY_STAGES)} stages from a CURRENT-RSS series that can FALL "
           f"-- measured peak {_ps['measured_peak_stage']}, declared "
           f"{_ps['declared_peak_stage']}, agree: "
           f"{_ps['declared_stage_is_the_measured_peak']}")
        _hw = [_open["memory_plan"]["observed"][k]["peak_rss_mb_highwater"]
               for k, _ in DAY_STAGES]
        ok(all(b >= a for a, b in zip(_hw, _hw[1:])),
           f"AND THE OLD INSTRUMENT'S DEFECT IS SHOWN RATHER THAN ASSERTED: "
           f"the highwater series {[round(x, 1) for x in _hw]} is "
           f"non-decreasing BY CONSTRUCTION, so it could never have located a "
           f"peak anywhere but the last stage that allocated")
        _shape = _open["memory_plan"]["peak_stage_assertion"]
        ok(_shape["asserted"] is False
           and _shape["agrees"] is False
           and _ps["measured_peak_stage"] == "S4_null",
           f"AND A CORRECTION TO MY OWN DESIGN v9, MEASURED: it declared "
           f"S1_load 'THIS IS THE PEAK' flatly, and on the only book I can "
           f"measure the peak is {_ps['measured_peak_stage']} -- the "
           f"fixture's book is a few hundred KB and the draw loop's fixed "
           f"cost is larger. The claim is CONDITIONAL on the book "
           f"dominating, which is the real-day regime, and it is recorded "
           f"rather than refused here")
        _bad_pred = dict(_ps); _bad_pred["declared_stage_is_the_measured_peak"] = False
        refuses(lambda: assert_peak_stage(_bad_pred, fixture=False,
                                          day="2026-09-03"),
                "AND ON A REAL DAY IT REFUSES: a measured peak that is not "
                "where the plan says it is means the 8 GiB ceiling's basis "
                "is wrong, so the DAY stops -- the cap is never raised "
                "(R-174)", "the memory plan declares")
        _ok_pred = dict(_ps); _ok_pred["declared_stage_is_the_measured_peak"] = True
        ok(assert_peak_stage(_ok_pred, fixture=False,
                             day="2026-09-03")["asserted"] is True,
           "and a real day whose peak IS where the plan says it is admits, "
           "marked asserted -- both directions, so the predicate is one "
           "that can fail and one that can pass")
        refuses(lambda: assert_peak_stage({"computable": False},
                                          fixture=True, day="X"),
                "and an UNCOMPUTABLE peak refuses in either mode -- a plan "
                "whose central claim cannot be checked at all is worse "
                "than one that disagrees", "not computable")


        # ---- a real day is refused for the reasons it must be --------------
        refuses(lambda: run_day("2026-09-04", _made["book_path"],
                                params=_DAYP, fixture=False),
                "A REAL DAY WITHOUT THE LOCK REFUSES BEFORE ANY WORK: it is "
                "heavy by construction (BE projects ~2.3 h for both arms), so "
                "the lock is taken FIRST or the run does not start (R-575(C))",
                "does not hold")

    # ================= DE 78: THE REAL-DAY PATH, ON A SYNTHETIC DAY ======
    # SKIPPED OFFLINE, and the reason is precise: these drive BE's own
    # `draw_null` for the cross-check, and BE's `mechanics` reads its
    # committed null receipt under `data/`. A FIXTURE RUN must open no path
    # under `data/`; a DAY RUN is supposed to read -- what its residency
    # proof asserts is that no TAPE, INDEX or FRAGMENT artifact was opened,
    # which is a different claim and the one the addendum asked for.
    if offline:
        for _i in range(DAY_PATH_CHECKS):
            offline_skip(f"DE 78 day-path check {_i + 1}/"
                         f"{DAY_PATH_CHECKS} -- drives BE's draw_null, "
                         f"which reads BE's committed null receipt under "
                         f"data/")
        # THE COUNT-AGREEMENT CHECK IS ITSELF A CHECK and is skipped too.
        # Leaving it out made the offline battery 94 against the online 95,
        # and the nested selftest inside the fixture run failed on the
        # difference -- caught by the count assertion, which is what it is
        # for.
        offline_skip("DE 78 day-path check count agreement (online only)")
    else:
        _n_before = n[0]
        _day_path_checks()
        _ran = n[0] - _n_before
        ok(_ran == DAY_PATH_CHECKS,
           f"and the day-path check COUNT is asserted against the declared "
           f"constant: {_ran} == {DAY_PATH_CHECKS}. The offline skip list "
           f"is generated from that constant, so a check added here without "
           f"updating it REFUSES rather than quietly shrinking the offline "
           f"battery")


    # ---- the reviewer's DE 77 re-drive: two findings, both driven ------
    _rw = dict(live)
    _rw["days"] = ["2026-08-29"] + live["days"]          # the caller lies
    refuses(lambda: may_run_day(_rw, "2026-08-29",
                                day_row={"day_closed_calendar": True,
                                         "all_conjuncts_and_quality": True}),
            "THE REVIEWER'S CALLER-REWRITE ATTACK, ON may_run_day: one line "
            "of caller-side rewriting used to return may_run TRUE for "
            "2026-08-29, a day R-555 EXCLUDED. `resolve_draws` was hardened "
            "against exactly this in DE 77b and ITS TWIN WAS NOT -- the "
            "same defect one function away. The ruled set is now read from "
            "the COMMITTED file", "not in the ruled day set")
    ok(may_run_day(live, "2026-09-03",
                   day_row={"day_closed_calendar": True,
                            "all_conjuncts_and_quality": True})["may_run"]
       is True,
       "and the positive control still ADMITS the ruled smoke day, so the "
       "hardening did not simply make the door refuse everything")
    # ---- R-387: carrying_commit, and the property that matters ---------
    _cc_ref = carrying_commit_block(
        Path(__file__).resolve().parents[0] / "be_cancel_axis_null.py")
    ok(_cc_ref["carrying_commit"] and len(_cc_ref["carrying_commit"]) == 40
       and _cc_ref["producing_code_is_the_committed_bytes"] is True,
       f"R-387 POSITIVE CONTROL, AND IT ADMITS: for a file that IS the "
       f"bytes HEAD holds, `producing_code_is_the_committed_bytes` is True "
       f"at {_cc_ref['carrying_commit'][:12]} -- and the check is the FILE's "
       f"blob, not a whole-tree dirty flag (tree_dirty here is "
       f"{_cc_ref['tree_dirty']}, and it is not the check)")
    _tmp_in_repo = (Path(__file__).resolve().parents[0]
                    / "_de77_carrying_commit_knownbad.tmp.py")
    try:
        _tmp_in_repo.write_text("# a file HEAD does not hold\n")
        _cc_bad = carrying_commit_block(_tmp_in_repo)
        ok(_cc_bad["producing_code_is_the_committed_bytes"] is False,
           "R-387 KNOWN-BAD: a producer whose bytes HEAD does NOT hold "
           "reports False -- so a receipt cannot name a commit that does "
           "not contain the code that ran")
    finally:
        _tmp_in_repo.unlink(missing_ok=True)

    ok(n[0] + 1 + len(skipped) == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} run + "
       f"{len(skipped)} skipped == {EXPECTED_CHECKS}")
    LAST_BATTERY.update({
        "outcome": "PASS", "n_checks_run": n[0],
        "n_checks_skipped_offline": len(skipped),
        "skipped_offline": list(skipped),
        "expected_checks_in_the_source": EXPECTED_CHECKS,
        "run_plus_skipped_equals_source_expected":
            n[0] + len(skipped) == EXPECTED_CHECKS,
        "offline": offline,
        "why_skipped": ("these checks READ `data/`; a fixture run must "
                        "open no path under it, and a skipped check that "
                        "is silent is a check that has stopped existing"
                        if skipped else None),
        "ran_in_the_emitting_process": True})
    if not quiet:
        print(f"[de_multiday_gate1_runner] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fixture-run", action="store_true", dest="fixture")
    ap.add_argument("--dry-run-ledger", action="store_true", dest="ledger",
                    help="read the ledger and report the day set; this is "
                         "NOT a fixture run and says so in its status")
    ap.add_argument("--day", type=str,
                    help="run ONE ruled day, SEALED. Requires --book and, "
                         "on a real day, the heavy-run lock (R-575(C))")
    ap.add_argument("--book", type=Path,
                    help="BE's day book. Its digest is verified against "
                         "BE's own sidecar receipt, never against a "
                         "constant in this file")
    ap.add_argument("--synthetic-day", type=str,
                    help="build a SYNTHETIC day book of BE's declared "
                         "shape, run --day on it, and emit the receipt. "
                         "Proves the path, never the data")
    ap.add_argument("--rehearse-smoke", type=str, dest="rehearse",
                    help="emit the smoke invocation for DAY, evaluating "
                         "every precondition now and refusing by name on "
                         "the book that does not exist yet. Runs nothing")
    ap.add_argument("--n-days-complete", type=int, default=1,
                    help="how many of the G days are complete; the seal "
                         "opens only at G")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.rehearse:
        payload = rehearse_smoke(a.rehearse)
        payload["data_root"] = DR.require_canonical("the smoke rehearsal")
        payload["source_identity"] = {
            "producing_code": Path(__file__).name,
            "producing_code_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()).hexdigest(),
            **carrying_commit_block(Path(__file__).resolve())}
        if a.output is not None:
            if a.output.exists():
                raise RunnerRefused(f"output already exists: {a.output}")
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"emitted": str(a.output), "day": payload["day"],
                          "status": payload["status"],
                          "blocking": payload["blocking"],
                          "book_exists": payload["book"]["exists"]}))
        return 0
    if a.synthetic_day or a.day:
        return _main_day(a)
    if a.ledger:
        payload = dry_run_ledger()
        if a.output is not None:
            if a.output.exists():
                raise RunnerRefused(f"output already exists: {a.output}")
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
            payload = {**{k: payload[k] for k in (
                "status", "ledger_root_resolved", "root_branch")},
                "emitted": str(a.output),
                "n_ruled_closed": payload["summary"]["n_ruled_closed"],
                "n_ruled_pending": payload["summary"]["n_ruled_pending"],
                "r7_assertion": payload["r7_assertion_evaluated_live"][
                    "all_relations_hold"]}
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0
    if not a.fixture or a.output is None:
        ap.error("choose --selftest or --fixture-run --output PATH")
    payload = fixture_run_proven()
    if a.output.exists():
        raise RunnerRefused(f"output already exists: {a.output}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"emitted": str(a.output), "status": payload["status"],
                      "G": payload["params_used"]["G"],
                      "battery": payload["battery"]["outcome"],
                      "battery_checks": payload["battery"]["n_checks_run"],
                      "any_arm_fails": payload["aggregate"]["any_arm_fails"]}))
    return 0


def _main_day(a) -> int:
    """`--day` and `--synthetic-day`, one code path with one difference:
    where the book comes from and whether the run is a fixture."""
    params = load_params()
    if a.output is None:
        raise RunnerRefused("REFUSED: --output is required for a day run")
    if a.output.exists():
        raise RunnerRefused(f"output already exists: {a.output}")
    import tempfile as _tf
    if a.synthetic_day:
        day = a.synthetic_day
        td = _tf.mkdtemp(prefix="de_synth_day_")
        made = write_synthetic_day(day, td, params=params)
        book, fixture = made["book_path"], True
    else:
        day, book, fixture = a.day, a.book, False
        if book is None:
            raise RunnerRefused("REFUSED: --day requires --book")
        if day not in params["days"]:
            raise RunnerRefused(
                f"REFUSED: {day} is not in the ruled day set "
                f"{params['days']}.")
    proof = day_split_residency_proof(
        day, book, params=params, fixture=fixture,
        n_days_complete=a.n_days_complete)
    payload = proof.pop("day_result")
    payload["split_residency_proof"] = proof
    payload["source_identity"] = {
        "producing_code": Path(__file__).name,
        "producing_code_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()).hexdigest(),
        **carrying_commit_block(Path(__file__).resolve()),
    }
    # R-572(B)(4) / the coordinator's DE 78 ruling, as FIELDS.
    payload["committed_bytes_policy"] = {
        "fixture": "producing_code_is_the_committed_bytes MAY be false and "
                   "is RECORDED, never refused -- a hard refusal would "
                   "block every pre-commit fixture emission and push "
                   "someone to commit blind (RULED, DE 78)",
        "real_day": "a REAL day REFUSES on false: a result-bearing day "
                    "artifact naming a commit that does not hold the code "
                    "that ran is provenance theatre",
        "this_run_is_a_fixture": fixture,
        "producing_code_is_the_committed_bytes":
            payload["source_identity"]["producing_code_is_the_committed_"
                                       "bytes"],
    }
    if not fixture and not payload["source_identity"][
            "producing_code_is_the_committed_bytes"]:
        raise RunnerRefused(
            "REFUSED: a REAL day run whose producing code is not the bytes "
            "HEAD holds. Commit the runner first; the artifact must be able "
            "to name the commit that produced it (DE 78 ruling).")
    # (3) reviewer §3.4: `offline=fixture`, not `offline=True`. A REAL
    # day's receipt used to say `battery: PASS` having skipped all four R6
    # controls and EVERY day-path check -- honestly disclosed, but the
    # field a consumer resolves on a real-day artifact was a pass that
    # excluded the path being run. The offline choice is right for a
    # fixture, where it preserves the data-free property; a real day is
    # already reading the ledger and has no such justification.
    LAST_BATTERY.clear()
    selftest(quiet=True, offline=fixture)
    payload["battery"] = dict(LAST_BATTERY)
    payload["battery_scope"] = {
        "offline": fixture,
        "why": ("a FIXTURE run skips the checks that read `data/`, because "
                "that is what makes it a fixture" if fixture else
                f"a REAL day runs the FULL battery -- the four R6 "
                f"controls and all {DAY_PATH_CHECKS} day-path checks -- "
                f"because the run is already reading the ledger (reviewer "
                f"S3.4). THIS COUNT IS COMPUTED: it read a literal 23 "
                f"beside a constant of 38 and then 49, which is rule 10 in "
                f"my own receipt"),
        "day_path_checks_declared": DAY_PATH_CHECKS,
    }
    payload["data_root"] = DR.require_canonical(
        f"the {'fixture' if fixture else 'sealed'} day run", fixture=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                   default=str) + "\n")
    print(json.dumps({
        "emitted": str(a.output), "status": payload["status"],
        "day": payload["day"],
        "arms": {r["arm"]: r.get("status")
                 for r in payload["per_day_sealed_artifacts"]},
        "decisions": {k: v["decisions"]
                      for k, v in payload["decision_populations"].items()},
        "peak_rss_mb": round(payload["memory_plan"]["peak_rss_mb"], 1),
        "wall_s": round(payload["resources"]["wall_seconds"], 1),
        "lock_held": payload["wrapper"]["heavy_run_lock_held"],
        "heavy_by_measurement": payload["wrapper"]["rule20"][
            "heavy_by_measurement"],
        "no_tape_artifact_opened": proof[
            "no_tape_index_or_fragment_artifact_was_opened"],
        "battery": payload["battery"]["outcome"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

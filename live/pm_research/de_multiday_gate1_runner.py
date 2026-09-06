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
EXPECTED_CHECKS = 71
#: params **v2** (R-572(B)(2)): `run_not_before_utc` split into
#: `read_not_before_utc` + `day_runs_allowed_for_closed_qualifying_days`,
#: and BE's cascade digest re-pointed at `ab75b41`. v1 is UNTOUCHED and
#: stays as provenance (rule 13).
PARAMS_REL = "live/pm_research/declarations/de_multiday_gate1_params_v2.json"
SUPERSEDED_PARAMS_REL = ("live/pm_research/declarations/"
                        "de_multiday_gate1_params_v1.json")

#: R5 -- the fields that do not exist in a per-day artifact until every day
#: is complete. Named once, so the guard and the emitter cannot disagree.
ECONOMIC_FIELDS = ("D_E0", "D_E_MINUS_R", "Z", "p_location",
                   "null_mean", "null_sd", "null_draws_summary")


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
    in_ruled = day in ruled_day_set()
    if fixture and in_ruled:
        raise RunnerRefused(
            f"REFUSED: fixture draws were claimed for {day}, which IS in "
            f"the ruled day set {ruled_day_set()}. A fixture run on a "
            f"ruled day is not a fixture run, and only its author would "
            f"know.")
    if not fixture and not in_ruled:
        raise RunnerRefused(
            f"REFUSED: a real-day run was claimed for {day}, which is NOT "
            f"in the ruled day set. The ruled set is the population; a day "
            f"outside it is a day chosen after the fact.")
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
    if day not in params.get("days", []):
        raise RunnerRefused(
            f"REFUSED: {day} is not in the ruled day set {params.get('days')}. "
            f"The ruled set is the population (R-555).")
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
    payload["data_free_proof"] = proof
    payload["data_root"] = DR.require_canonical(
        "the fixture run", fixture=True, proof=proof)
    return payload


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
        "no_day_book_was_read": True,
        # FILLED BY `fixture_run_proven()` AFTER the body has run under
        # instrumentation -- the claim cannot precede its own proof.
        "no_path_under_data_was_opened": None,
        "data_root": None,
        "runnable_from_a_shell_worktree": True,
        "the_committed_day_set_is_empty": True,
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
        ok(any(x.endswith("de_multiday_gate1_params_v2.json")
               for x in _seen),
           "and the instrument is not vacuous -- it DID observe the "
           "parameter file being read, so a zero above is a measurement "
           "rather than a silent no-op")
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
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
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


if __name__ == "__main__":
    raise SystemExit(main())

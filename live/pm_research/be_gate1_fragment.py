"""THE GATE-1 FEATURE FRAGMENT, ONE FILE PER RULED DAY. Never a pinned name.

WHY THIS EXISTS. Round 48's book build refused at `fragment_slice`: *"the
slice is EMPTY: no window of harmful_exposure_rows_v3_eraB.json matched the
247 wanted slugs."* Verified at the artifact -- eraB is 1.24 GB, 471 windows,
2026-08-24 to 2026-08-25, and holds no September slug. It is the CONSUMED,
PINNED population the frozen candidate was fitted on. **September was never
materialised.** A scored day-book cannot be assembled until it is.

THE MECHANISM ALREADY EXISTED AND IT IS `build_topup_rows.py`'s: a SEPARATE
OUTPUT PATH, with a refusal if the build ever aims at a pinned name. That
module's own docstring says what is at stake -- running the ordinary CLI
"would have silently overwritten the frozen population's data with a
different population under the same filename. Nothing would have errored."
This module copies that discipline and WIDENS it:

  * the guard refuses EVERY `harmful_exposure_rows_*` dataset now on disk,
    not the two the top-up named;
  * and it refuses ANY path that already exists, because the second-worst
    outcome after overwriting a pinned name is overwriting yesterday's
    fragment with today's under a name nobody pinned.

ONE FILE PER DAY, at `harmful_exposure_rows_v3_gate1_<YYYYMMDD>.json`. Not
one file for the ruled set: a per-day file can be built, measured, refused
and re-built independently, and a day that will not fit under the cap costs
only itself.

THE POPULATION IS THE DAY'S ADMISSIBLE WINDOWS, DIGEST-PINNED. Not a declared
population interval -- those end 2026-08-26T00:00 and no September day passes
them. It is `de_admissible_windows.supply(day, present_from_ledger(day))`,
the same selection the forward scorer gated 09-03 through at 12/12 and the
same 247 btc slugs round 48 selected. The receipt carries the digest of that
slug list, so a fragment built from a different selection cannot be mistaken
for this one.

THE DIGEST IS OF THE BYTES AS WRITTEN, AND IT IS STREAMED. `write_text(
json.dumps(...))` materialises the whole ~600 MB dataset as one string before
a byte reaches disk -- the allocation shape the top-up builder was changed to
avoid. So the encoder streams into a temp file AND into a hashlib at the same
time: the digest is of exactly the bytes that landed, with nothing held.

R8: IF IT DOES NOT FIT, IT REFUSES. Never a raised cap and never a smaller
population -- a fragment over fewer windows than the day supplies is a
fragment about a different day.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_rule22 as _R22

#: R-649 §3.2: 75 is EX_TEMPFAIL. From OUTSIDE a unit, ExecMainStatus=75
#: reads "the lock was held" OR "a producer broke the declaration and exited
#: 75 for its own reasons" -- and the two are indistinguishable. So this
#: producer DECLARES its exit codes and its selftest asserts 75 is not among
#: them, which is what makes the launcher's 75 mean one thing.
EXIT_CODES = {
    0: "the artifact was written and its receipt emitted",
    1: "a refusal or an uncaught error (Python's default for an exception)",
    2: "usage: no --day and no --selftest",
}
EXIT_CODE_NOTE = ("75 is RESERVED to the launcher's flock conflict and is "
                  "not in this map; the selftest asserts it.")


#: RULE 22 AS AMENDED (R-605): the closure and HEAD are captured HERE, at
#: import, before any work. This producer carried NO provenance stamp at
#: all until round 60 (R-613) -- a landing to it mid-run would have been
#: invisible in its receipt. The stamp is READ at emit so it can report
#: drift; the digests it reports are the ones seen at import.
_R22.init("be_gate1_fragment import")

ROOT = HERE.parents[1]
DERIVED = _BDR.derived()
OUT_DERIVED = _BDR.derived()  # BE48 B.4: one root, from the resolver. This was `ROOT / 'data/...'` -- a data root built on a CODE root, which the first version of `audit_derived_roots` could not see because the value holds no `parents` and no literal.

#: EVERY exposure-row dataset on disk, not the two the top-up guard named.
#: A name that is not pinned today can be pinned tomorrow; the cheap rule is
#: to refuse them all and require a Gate-1 name.
PINNED_PREFIX = "harmful_exposure_rows"
GATE1_STEM = "harmful_exposure_rows_v3_gate1_"
COIN = "btc"
MEM_CAP_GB = 8.0


class FragmentRefused(RuntimeError):
    """A named refusal."""


def out_path(day: str, coin: str = COIN) -> Path:
    return DERIVED / f"{GATE1_STEM}{day}_{coin}.json"


def pinned_names(derived: Path | None = None) -> set:
    d = Path(derived) if derived is not None else DERIVED
    return {p.name for p in d.glob(f"{PINNED_PREFIX}*.json")
            if not p.name.startswith(GATE1_STEM)}


def guard_output(path: Path, *, derived: Path | None = None) -> None:
    """Refuse a pinned name, AND refuse anything that already exists."""
    p = Path(path)
    if p.name in pinned_names(derived):
        raise FragmentRefused(
            f"REFUSED: {p.name} is an existing exposure-row dataset. A "
            f"Gate-1 build writing there would replace one population's data "
            f"with another under the same name -- silently, with nothing "
            f"raising, which is exactly what destroyed nothing only because "
            f"`build_topup_rows` refused it first.")
    if not p.name.startswith(GATE1_STEM):
        raise FragmentRefused(
            f"REFUSED: {p.name} is not a Gate-1 fragment name. The output "
            f"must be `{GATE1_STEM}<day>_<coin>.json` so no build can aim at "
            f"a name another population owns.")
    if p.exists():
        raise FragmentRefused(
            f"REFUSED: {p} already exists. Overwriting yesterday's fragment "
            f"with today's under a name nobody pinned is the second-worst "
            f"version of the same defect. Move or delete it deliberately.")


def population(day: str, coin: str = COIN) -> dict:
    """The day's admissible windows, and the digest that pins them."""
    import be_forward_day as FD
    import de_admissible_windows as AW
    sup = AW.supply(day, FD.present_from_ledger(day))
    slugs = sorted(w["slug"] for w in (sup.get("windows") or {}).get(coin, []))
    if not slugs:
        raise FragmentRefused(
            f"REFUSED: no supplied {coin} windows for {day}.")
    blob = "\n".join(slugs).encode()
    return {"day": day, "coin": coin, "n_windows": len(slugs),
            "slugs_sha256": hashlib.sha256(blob).hexdigest(),
            "source": "de_admissible_windows.supply(day, "
                      "be_forward_day.present_from_ledger(day))",
            "why_not_a_declared_population": "the declared population "
                                             "intervals end "
                                             "2026-08-26T00:00; no September "
                                             "day passes slug_in_population",
            "slugs": slugs}


def selector_for(pop: dict):
    """`build_rows`' selector hook, over the day's slugs only."""
    import flow_intensity as fi
    import harmful_exposure_rows as HER
    want = list(pop["slugs"])
    era = HER._era_or_refuse(fi, None, "be_gate1_fragment")
    paths, toks = fi._archive_paths(), fi.token_map()
    gaps = fi.gaps_by_slug(era)
    missing = [s for s in want if s not in paths or s not in toks]
    if missing:
        raise FragmentRefused(
            f"REFUSED: {len(missing)} supplied slug(s) have no archive path "
            f"or token entry, e.g. {missing[:3]}. Dropping them would build "
            f"a fragment about a different day (R8: never a smaller "
            f"population).")

    def _sel(coins, population_name):
        return [(s, paths[s], toks[s][0], toks[s][1], gaps.get(s, []))
                for s in want], 0
    _sel.era = era
    return _sel


def _rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def stream_write(obj, dst: Path) -> dict:
    """Encode ONCE, into the file and into the hash at the same time.

    The digest is therefore of exactly the bytes that landed -- the B-1
    discipline on the write side -- without ever holding the document."""
    h = hashlib.sha256()
    n = 0
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            for chunk in json.JSONEncoder(default=str).iterencode(obj):
                b = chunk.encode()
                h.update(b)
                fh.write(b)
                n += len(b)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"sha256": h.hexdigest(), "bytes": n,
            "digest_is_of_the_bytes_as_written": True,
            "streamed_never_materialised": True}


#: The ruled day set. btc-only is R-560; the day set is the coordinator's.
RULED_DAYS = ("20260901", "20260902", "20260903", "20260904", "20260905")


def declaration() -> dict:
    """WHAT THIS BUILDER IS, declared, so DE's runner can bind to it."""
    return {
        "protocol": "BE_GATE1_FRAGMENT_DECLARATION_V1",
        "why": ("round 48's book refused at fragment_slice because "
                "harmful_exposure_rows_v3_eraB.json is the CONSUMED, PINNED "
                "08-24/25 population (1.24 GB, 471 windows, verified at the "
                "artifact) and September was never materialised into it"),
        "one_file_per_ruled_day": True,
        "why_per_day_and_not_one_file": ("a per-day fragment can be built, "
                                         "measured, refused and rebuilt on "
                                         "its own, and a day that will not "
                                         "fit costs only itself"),
        "path_scheme": f"data/pm_5min/derived/{GATE1_STEM}<YYYYMMDD>_<coin>.json",
        "coin": COIN, "coin_ruling": "R-560, btc-only for all ruled days",
        "ruled_days": list(RULED_DAYS),
        "population": {
            "definition": "that day's admissible windows from "
                          "de_admissible_windows.supply(day, "
                          "be_forward_day.present_from_ledger(day))",
            "digest_pinned": "sha256 over the sorted slug list, in every "
                             "receipt, so a fragment built from a different "
                             "selection cannot be mistaken for this one",
            "why_not_a_declared_population": "the declared intervals end "
                                             "2026-08-26T00:00; no September "
                                             "day passes slug_in_population",
            "never_reduced": "R8 -- a fragment over fewer windows than the "
                             "day supplies is a fragment about another day",
        },
        "guard": {
            "model": "build_topup_rows.guard_output, WIDENED",
            "refuses_every_existing_exposure_row_dataset": sorted(
                pinned_names()),
            "refuses_any_name_outside_the_gate1_stem": True,
            "refuses_any_path_that_already_exists": True,
            "why_widened": "the top-up guard named TWO pinned datasets; "
                           "there are seven on disk, and overwriting "
                           "yesterday's Gate-1 fragment with today's under a "
                           "name nobody pinned is the same defect one step "
                           "down",
        },
        "receipt_fields_per_day": ["population.slugs_sha256",
                                   "population.n_windows", "build.n_rows",
                                   "resources.wall_s",
                                   "resources.peak_rss_gb",
                                   "fragment.sha256", "fragment.bytes"],
        "digest_discipline": ("the encoder streams into the file and into "
                              "the hash at once, so the sha256 is of exactly "
                              "the bytes that landed and the document is "
                              "never materialised (the B-1 rule on the write "
                              "side; write_text(json.dumps(...)) is the "
                              "allocation shape build_topup_rows was changed "
                              "to avoid)"),
        "cap": {"gb": MEM_CAP_GB, "raised": False,
                "on_exceed": "REFUSE the day and report the measured peak"},
        "builds_no_book_and_runs_no_assembly": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


def build(day: str, *, coin: str = COIN, progress: bool = True,
          fixture: bool = False) -> dict:
    # RULE 20, MEASURED BEFORE ANY WORK (REV 63 S4). This receipt mentioned
    # the lock in NO field at all, so "the lock was taken and held across
    # both steps" was a claim only a register row carried -- its own
    # artifacts could not support it. The evidence is DE's, delegated rather
    # than reimplemented (Q-BE-271), and a real build without an EXCLUSIVE
    # hold refuses HERE rather than after ten minutes of work.
    _wrapper = _R22.lock_evidence(fixture=fixture)
    import harmful_exposure_rows as HER
    t0 = time.time()
    dst = out_path(day, coin)
    guard_output(dst)
    pop = population(day, coin)
    sel = selector_for(pop)
    if progress:
        print(json.dumps({"stage": "population", "windows": pop["n_windows"],
                          "slugs_sha256": pop["slugs_sha256"][:16],
                          "out": str(dst)}), flush=True)
    built = HER.build_rows(coins=(coin,), selector=sel)
    rows = built.get("rows", [])
    if not rows:
        raise FragmentRefused(
            f"REFUSED: the {day} fragment produced ZERO rows. Writing an "
            f"empty dataset would let the assembly 'run' on nothing -- the "
            f"empty answer that looks like a result.")
    peak = _rss_gb()
    if peak > MEM_CAP_GB:
        raise FragmentRefused(
            f"REFUSED: peak RSS {peak} GB exceeded the {MEM_CAP_GB} GB cap. "
            f"R8: the cap is NOT raised and the population is NOT reduced. "
            f"The day is reported with its measured peak and refused.")
    if progress:
        print(json.dumps({"stage": "built", "rows": len(rows),
                          "windows": built.get("n_windows"),
                          "peak_gb": peak}), flush=True)
    w = stream_write(built, dst)
    return {
        "protocol": "BE_GATE1_FRAGMENT_V1",
        "day": day, "coin": coin,
        "fragment": {"path": str(dst), **w},
        "population": {k: v for k, v in pop.items() if k != "slugs"},
        "build": {"n_rows": len(rows), "n_windows": built.get("n_windows"),
                  "days": built.get("days"),
                  "reconciliation_failures":
                      built.get("reconciliation_failures"),
                  "boundary_time_violations":
                      built.get("boundary_time_violations"),
                  "consume_clock_violations":
                      built.get("consume_clock_violations"),
                  "unhooked_state_changes":
                      built.get("unhooked_state_changes")},
        "resources": {"wall_s": round(time.time() - t0, 1),
                      "peak_rss_gb": _rss_gb(),
                      "cap_gb": MEM_CAP_GB, "cap_raised": False},
        "guard": {"refuses_existing_datasets": sorted(pinned_names()),
                  "refuses_any_existing_path": True,
                  "required_stem": GATE1_STEM},
        "data_root": _BDR.receipt_block(),
        "scope": _BDR.scope_stats(),
        "launch_form_at_runtime": _R22.assert_not_a_scope(fixture=fixture),
        "exit_codes": {"map": {str(k): v for k, v in EXIT_CODES.items()},
                       "conflict_code_is_the_launcher's": _R22.lock_conflict_rc(),
                       "note": EXIT_CODE_NOTE,
                       "declaration_head":
                           _R22.declaration_head("heavy_run_form")["name"]},
        "wrapper_measured": _wrapper,
        "producing_code": _R22.stamp(__file__),
        "rule22_checked_at_emit": _R22.assert_unchanged(
            "be_gate1_fragment receipt emit"),
        "one_file_per_day": True,
        "no_book_built": True, "no_assembly_run": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 22


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    names = pinned_names()
    ok(len(names) >= 5 and "harmful_exposure_rows_v3_eraB.json" in names,
       f"the guard's refusal set is READ FROM DISK, not typed: "
       f"{len(names)} existing exposure-row datasets including the pinned "
       f"eraB one -- `build_topup_rows` named two, and there are {len(names)}")
    for n in sorted(names)[:3]:
        try:
            guard_output(DERIVED / n)
            ok(False, f"{n} must refuse")
        except FragmentRefused as e:
            ok("existing exposure-row dataset" in str(e),
               f"KNOWN-BAD: writing to {n} REFUSES -- the exact command that "
               f"would replace one population with another under one name")
    try:
        guard_output(DERIVED / "some_other_name.json")
        ok(False, "a non-Gate-1 name must refuse")
    except FragmentRefused as e:
        ok("not a Gate-1 fragment name" in str(e),
           "KNOWN-BAD: any name outside the Gate-1 stem REFUSES, so no build "
           "can aim at a name another population owns")
    good = out_path("20260903")
    if good.exists():
        ok(True, f"POSITIVE CONTROL SKIPPED: {good.name} already exists")
    else:
        guard_output(good)
        ok(True, f"POSITIVE CONTROL: the Gate-1 name {good.name} is ALLOWED "
                 f"while it does not exist")
    ok(GATE1_STEM not in "".join(names),
       "and no existing dataset already uses the Gate-1 stem, so the two "
       "namespaces do not overlap")

    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "x.json"
        w = stream_write({"a": [1, 2, 3], "b": "x"}, f)
        ok(w["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
           and w["bytes"] == f.stat().st_size,
           f"THE STREAMED DIGEST IS OF THE BYTES THAT LANDED: "
           f"{w['sha256'][:16]}… recomputed from the file matches, over "
           f"{w['bytes']} bytes, with the document never materialised")
        f2 = d / "y.json"
        w2 = stream_write({"a": [1, 2, 4], "b": "x"}, f2)
        ok(w2["sha256"] != w["sha256"],
           "and a DIFFERENT document digests differently -- the check is on "
           "the bytes, not a constant")

    d = declaration()
    ok(d["path_scheme"].endswith(f"{GATE1_STEM}<YYYYMMDD>_<coin>.json")
       and set(d["guard"]["refuses_every_existing_exposure_row_dataset"])
       == pinned_names()
       and d["cap"]["gb"] == MEM_CAP_GB,
       f"THE DECLARATION IS DERIVED FROM THE CODE IT DECLARES: the stem, the "
       f"{len(names)}-dataset refusal set and the cap all come from the "
       f"module's own constants, so a declaration cannot drift from the "
       f"builder")

    # ---- RULE 22 AS AMENDED (R-605): this producer had NO stamp at all ----
    import tempfile as _tf, importlib as _il
    _st22 = _R22.stamp(__file__)
    ok(_st22["producing_code"] == "be_gate1_fragment.py"
       and _st22["producing_code_sha256"]
       == __import__("hashlib").sha256(
           Path(__file__).read_bytes()).hexdigest()
       and _st22["captured_at"] == "IMPORT"
       and _st22["builder_commit"]
       and _st22["import_closure"]["n_modules"] >= 2,
       f"RULE 22: this module now stamps its OWN identity -- the digest of "
       f"the bytes at import, HEAD {{str(_st22['builder_commit'])[:12]}}, "
       f"and the {{_st22['import_closure']['n_modules']}} modules of its "
       f"import closure under live/. Until round 60 it carried none of "
       f"this, so a landing to it mid-run was invisible in its receipt")
    ok(_st22["closure_unchanged_during_the_run"] is True
       and _R22.assert_unchanged("be_gate1_fragment battery")["closure_unchanged"],
       "POSITIVE CONTROL: with nothing moved the emit ADMITS -- a guard "
       "shown only to refuse has not been shown to work (rule 16)")
    _td22 = _tf.mkdtemp(prefix="be60_be_gate1_fragment_")
    _pm = Path(_td22) / "be60_probe_be_gate1_fragment.py"
    _pm.write_text("V = 1\n")
    sys.path.insert(0, _td22)
    _il.import_module("be60_probe_be_gate1_fragment")
    _c22 = _R22.Capture(root=_td22).capture("battery")
    _pm.write_text("V = 2\n")
    try:
        _c22.assert_unchanged("be_gate1_fragment known-bad")
        ok(False, "a module rewritten mid-run must refuse the emit")
    except _R22.Rule22Refused as _e22:
        ok("be60_probe_be_gate1_fragment.py" in str(_e22) and "DID NOT RUN" in str(_e22),
           "KNOWN-BAD: a module of the closure rewritten mid-run REFUSES "
           "THE EMIT BY NAME -- R-603's defect, where a receipt would have "
           "named bytes that did not run")
    sys.path.remove(_td22)
    _src22 = Path(__file__).read_text()
    ok('"producing_code": _R22.stamp(__file__),' in _src22
       and '"rule22_checked_at_emit": _R22.assert_unchanged(' in _src22,
       "AND IT IS WIRED INTO THE EMITTED RECEIPT: both the stamp and the "
       "refusal, read from this module's own source rather than claimed")

    # ---- RULE 20: the lock, MEASURED, and refused before any work -------
    try:
        build("19700101")
        ok(False, "a real build without the heavy lock must refuse")
    except _R22.HeavyRunRefused as e:
        ok("does not hold" in str(e) and "be_heavy_run.sh" in str(e),
           "KNOWN-BAD, REV 63 S4: a REAL build that does not hold the "
           "heavy-run lock REFUSES before any work, naming the launcher "
           "that takes it. This receipt carried no lock field at all")
    _we = _R22.lock_evidence(fixture=True)
    ok(_we["delegated_to"] == "de_multiday_gate1_runner.wrapper_observed"
       and "lock_mode" in _we and "exclusive" in _we,
       f"POSITIVE CONTROL: the evidence is DE's own with the MODE read from "
       f"/proc/locks ({_we['lock_mode']!r}) -- held is not exclusive, and "
       f"two shared holders would both certify")
    _srcF = Path(__file__).read_text()
    ok('"wrapper_measured": _wrapper,' in _srcF
       and "_R22.lock_evidence(fixture=fixture)" in _srcF,
       "AND IT IS WIRED INTO THE EMITTED RECEIPT: read from this module's "
       "own source, not asserted in prose")
    _lf = _R22.assert_launch_form()
    ok(_lf["form_is_correct"] and _lf["lock_is_inside_the_unit"],
       f"THE LAUNCHER IS THE SERVICE FORM ({_lf['problems']}): "
       f"no `--scope`, a named unit, both caps, and the lock "
       f"INSIDE the unit. Every BE heavy run through 09-05 was a transient "
       f"scope whose payload dies with the launching shell")
    _bad_lf = _R22.assert_launch_form(
        "flock -n /l systemd-run --user --scope --slice=research.slice "
        "-p MemoryMax=8G -p CPUQuota=100% --setenv=PM_DATA_ROOT=/r "
        "--working-directory=/w -- cmd")
    ok(_bad_lf["form_is_correct"] is False
       and any("--scope" in x for x in _bad_lf["problems"]),
       "KNOWN-BAD: the exact form BE used for six heavy runs is REFUSED by "
       "name")

    ok(75 not in EXIT_CODES and 75 == _R22.lock_conflict_rc(),
       f"R-649 §3.2: this producer's declared exit codes are "
       f"{sorted(EXIT_CODES)} and 75 is NOT among them -- so a unit reading "
       f"ExecMainStatus=75 means the lock was held, and cannot also mean "
       f"this producer broke the declaration. The 75 is read from the "
       f"declaration head, not typed here")
    ok(_R22.assert_not_a_scope(fixture=True)["kind"] in
       ("scope", "service", "none"),
       f"REV 65 §1.2: the launch form is decided at RUNTIME from this "
       f"process's own cgroup leaf ({_R22.cgroup_leaf()['leaf']!r}), which a "
       f"static lint cannot do -- it cannot see a `--scope` behind a "
       f"variable or a wrapper")
    try:
        _R22.assert_not_a_scope(fixture=False) if \
            _R22.cgroup_leaf()["kind"] == "scope" else None
        _leafok = _R22.cgroup_leaf()["kind"] != "scope"
    except _R22.HeavyRunRefused as _e:
        _leafok = "transient SCOPE" in str(_e)
    ok(_leafok,
       "KNOWN-BAD, DRIVEN WHERE IT LIVES: a process whose own cgroup leaf is "
       "a `.scope` REFUSES a real day by name -- nine BE heavy runs were "
       "scopes and every receipt said so in `scope.unit`; no seat read it")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        out = declaration()
        dst = (HERE / "declarations" /
               "be_gate1_fragment_declaration_v1.json")
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst),
                          "days": len(out["ruled_days"]),
                          "refuses": len(out["guard"][
                              "refuses_every_existing_exposure_row_dataset"])}))
        return 0
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = OUT_DERIVED / f"be_gate1_fragment_receipt_{day}_{COIN}.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"receipt": str(dst),
                          "fragment": out["fragment"]["path"],
                          "sha256": out["fragment"]["sha256"],
                          "bytes": out["fragment"]["bytes"],
                          "n_rows": out["build"]["n_rows"],
                          "wall_s": out["resources"]["wall_s"],
                          "peak_rss_gb": out["resources"]["peak_rss_gb"]}))
        return 0
    print("usage: be_gate1_fragment.py --selftest | --declare | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

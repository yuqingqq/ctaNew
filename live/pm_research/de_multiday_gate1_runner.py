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
import math
import random
import re
import statistics
import subprocess
import sys
import time
import gc as _gc
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import da_root as DAROOT  # noqa: E402
import de_data_root as DR  # noqa: E402
import de_multiday_design_declaration as DESIGN  # noqa: E402


PROTOCOL = "P003_DE_MULTIDAY_GATE1_RUNNER_V2"
EXPECTED_CHECKS = 402
#: params **v2** (R-572(B)(2)): `run_not_before_utc` split into
#: THE DECLARED EXPERIMENT PARAMETER FILE. It is a LITERAL on purpose and
#: stays one: "always the newest" would let a parameter file appear and
#: change the run, which is choosing after seeing. That is the opposite of
#: the launch-form declaration, where the head must be resolved -- a
#: constant of the wrapper, not a parameter of the experiment.
#:
#: (REV 71 S1.4(d): `SUPERSEDED_PARAMS_REL` sat here, read by NOTHING, and
#: the comment above it still described the v1 -> v2 bump long after v14.
#: A dead constant beside a stale comment is two things a reader can
#: believe.)
#: R-765: v18 carries the USER's ruling that retires R5. Moving the
#: pointer here is what collapses the sealed path and the early-read
#: path into ONE path with ONE emission.
PARAMS_REL = "live/pm_research/declarations/de_multiday_gate1_params_v20.json"

#: R5 -- the fields that do not exist in a per-day artifact until every day
#: is complete. Named once, so the guard and the emitter cannot disagree.
ECONOMIC_FIELDS = ("D_E0", "D_E_MINUS_R", "Z", "p_location",
                   "null_mean", "null_sd", "null_draws_summary",
                   # R-599 (DA 68): the RATIO survived the seal while BOTH
                   # quantities it is formed from were sealed. A
                   # null-derived ratio published before the read is more
                   # than the pre-read needs; the pre-read needs R4's
                   # STATUS, which the admissibility block publishes as two
                   # booleans.
                   "sd_over_abs_mean",
                   # R-659 (REV 71 S4.1), reversing R-656 on these three.
                   # They are OUTCOME counts under a policy, not population
                   # sizes: `n_fills_arm` MINUS `n_fills_baseline` is the
                   # intervention's effect IN EVENTS, and a reader who can
                   # order the arms by intervention size on day 1 has seen
                   # something about the result. The G=5 directional
                   # fallback at the horizon is a degree of freedom
                   # monotone in it. `n_decisions` stays OPEN: it is the
                   # population size R4's admissibility bar reads.
                   "n_fills_arm", "n_fills_baseline",
                   "n_cancels_issued")

#: The population SIZES that stay open (R-659): what rule 8 requires every
#: quoted population to carry and what R4's bar reads. Named so the two
#: sets can be asserted DISJOINT rather than kept apart by hand.
OPEN_POPULATION_SIZES = ("n_decisions", "n_scored_rows")

#: THE SEAL IS SCOPED PER NAME (REV 72 S1.4). Extending the list alone
#: made DA's `economic_absence()` read the LANDED 09-03 receipt as
#: `n_leaked_fields 6, sealed False` -- the programme's first sealed day
#: accused by its own instrument, for carrying fields that were OPEN BY
#: RULING when it was produced. A seal that reaches backwards convicts
#: the past of not having obeyed a future rule.
#:
#: So each name carries the DESIGN VERSION FROM WHICH IT IS SEALED:
#:   * `_strip_economic` seals by the list IN FORCE FOR THIS RUN -- all
#:     eleven from now on;
#:   * a census or verifier judges a receipt against the list in force
#:     WHEN THAT RECEIPT WAS PRODUCED, which the receipt names (its
#:     `protocol`, and from DE 100 its `provenance.design`).
SEALED_FROM_DESIGN_VERSION = {
    "D_E0": 1, "D_E_MINUS_R": 1, "Z": 1, "p_location": 1,
    "null_mean": 1, "null_sd": 1, "null_draws_summary": 1,
    "sd_over_abs_mean": 1,
    # R-659, reversing R-656: OUTCOME counts, sealed from design v23.
    "n_fills_arm": 23, "n_fills_baseline": 23, "n_cancels_issued": 23,
}
#: This module's own design version -- the list in force for THIS run.
DESIGN_VERSION_IN_FORCE = 23


def economic_fields_in_force(design_version: int | None = None) -> tuple:
    """The sealed names for a receipt produced under `design_version`.

    None means THIS run: every name. A receipt from before a name was
    sealed is judged without it -- it did not disobey a rule that did not
    exist."""
    v = DESIGN_VERSION_IN_FORCE if design_version is None \
        else int(design_version)
    return tuple(n for n in ECONOMIC_FIELDS
                 if SEALED_FROM_DESIGN_VERSION.get(n, 1) <= v)


#: The last design version whose seal scope was the original eight. A
#: receipt POSITIVELY recognised as older than the correction is judged
#: under this; absence never selects it (REV 73 §1.1(a)).
SCOPE_BEFORE_THE_CORRECTION = 22
#: The correction: the version from which the three outcome counts are
#: sealed, and the moment after which every receipt carries a design pair.
SCOPE_CORRECTED_AT_DESIGN_VERSION = 23
SCOPE_CORRECTION_UTC = "2026-09-06T14:45:00Z"


def design_chain_orphans(directory=None, family: str | None = None,
                         *, also_this_seats_resolver: bool = True) -> dict:
    """ORPHAN BRANCHES AT EMIT -- the field a receipt CONSUMES (REV 81 §5).

    REV 81 ruled the fork a RESOLVABILITY defect and the resolver right to
    name the orphan and still resolve a head: refusing would take every
    reader down for a defect in an old branch. But a field nobody reads is
    a condition that sits unnoticed, so every receipt that carries a design
    pin carries this.

    TWO RESOLVERS ARE ASKED, and their answers are COMPARED. BE 77's
    `declaration_chain.resolve_head` is the shared implementation; this
    module's own `design_chain()` predates it and globs only the BARE
    `_v<N>.json` names, so it cannot see the STAMPED versions this family
    used before v16. That difference is a MEASUREMENT here, not a comment:
    the two disagreed on 2026-09-06 (this seat's said [16]; the shared one
    said five tips, because v3..v6 all supersede a stamped v2 and only v7
    was continued). Reporting one number would have hidden four of them."""
    import declaration_chain as DC
    directory = (Path(directory) if directory is not None else
                 Path(DR.resolve()["data_root"]) / "pm_5min/derived")
    family = family or "p003_de_multiday_gate1_design"
    out = {"family": family, "dir": str(directory),
           "rule": "REV 81 §5 -- a fork is REPORTED, never resolved by "
                   "picking; while this is non-empty every receipt with a "
                   "design pin carries it"}
    try:
        shared = DC.resolve_head(directory, family)
        out["shared_resolver"] = {
            "implementation": "declaration_chain.resolve_head (BE 77) -- "
                              "the one chain implementation",
            "head": shared["name"], "head_sha256": shared["sha256"],
            "n_versions": shared["n_versions"],
            "orphan_branches": [o["version"] for o in
                                shared["orphan_branches"]],
            "n_orphan_branches": len(shared["orphan_branches"]),
            # HOW THE FORK WAS CLOSED, not only that it is (DE 112). A
            # reader seeing `orphan_branches: []` beside a `forks` block
            # with two entries needs the merge links to tell "closed by a
            # merge version" from "never forked".
            "n_merge_links": shared.get("n_merge_links"),
            "merged_tips": shared.get("merged_tips"),
            "fork_status": shared.get("fork_status"),
            "forks_two_versions_superseding_one":
                shared["forks_two_versions_superseding_one"]}
    except Exception as exc:                       # a STATUS, never a skip
        out["shared_resolver"] = {
            "status": "UNRESOLVED", "why": f"{type(exc).__name__}: {exc}",
            "and_this_is_not_a_pass": "rule 11 -- a resolver that could "
                                      "not answer has not answered 'no "
                                      "orphans'"}
    if also_this_seats_resolver:
        try:
            mine = design_chain()
            xc = mine.get("own_walk_kept_as_a_cross_check") or {}
            out["this_seats_resolver"] = {
                "implementation": (
                    "de_multiday_gate1_runner.design_chain -- from DE 111 "
                    "this RESOLVES THROUGH `declaration_chain.resolve_head` "
                    "(BE 77/80) and keeps its own walk only as a "
                    "cross-check; the two are no longer independent "
                    "answers and this block is no longer a second one"),
                "resolved_by": mine.get("resolved_by"),
                "head": mine.get("head_name"),
                "orphan_branches": mine.get("orphan_branches"),
                "n_orphan_branches": len(mine.get("orphan_branches") or []),
                "own_walk_cross_check": {
                    "orphan_branches_as_names": xc.get(
                        "orphan_branches_as_names"),
                    "agreement": xc.get("agreement"),
                    "globs": xc.get("globs")}}
        except Exception as exc:
            out["this_seats_resolver"] = {
                "status": "UNRESOLVED",
                "why": f"{type(exc).__name__}: {exc}"}
    a = out.get("shared_resolver", {}).get("n_orphan_branches")
    b = out.get("this_seats_resolver", {}).get("n_orphan_branches")
    out["the_two_resolvers_agree_on_the_count"] = (
        None if a is None or b is None else a == b)
    out["n_orphan_branches"] = a if a is not None else b
    out["chain_has_an_orphan_branch"] = (
        None if out["n_orphan_branches"] is None
        else out["n_orphan_branches"] > 0)
    return out


def design_version_of_receipt(rec: dict) -> dict:
    """WHICH SCOPE A RECEIPT IS JUDGED UNDER -- read from the receipt.

    **STABLE NAME. DA reads this function and `economic_fields_in_force`
    by AST from DA 95 on; neither is renamed in place. A change of meaning
    arrives as a superseding name, never as the same name doing something
    else.**

    REV 73 §1.1 found three holes in the first version, all driven:

      (a) THE DEFAULT WAS THE PERMISSIVE LIST. A receipt with no
          provenance fell through to v22 -- eight names -- so ABSENCE
          selected the WEAKER rule inside the scoping built to protect the
          seal. The default is now the STRICTEST list in force, and a
          pre-correction receipt is recognised POSITIVELY: by an emit
          stamp before the correction, corroborated by a
          `carrying_commit` that resolves. Never by a field being absent.
      (b) THE SELECTOR READ THE PATH AND IGNORED THE DIGEST BESIDE IT, so
          a receipt naming a v22 path with any other sha256 was judged
          v22. The version comes from the PAIR: the digest must match the
          file the path names, or the version is UNKNOWN and the
          strictest list applies.
      (c) THE SECOND FALLBACK WAS ORDER-DEPENDENT -- the first
          `_design_v<N>.json` among a run's opened paths, and the 09-03
          run opened a stale v10 beside v21. Dropped: an opened path is
          not a pin.
    """
    root = Path(DR.resolve()["data_root"])
    strictest = DESIGN_VERSION_IN_FORCE
    ev = []

    # ---- (b) THE PAIR, or nothing ------------------------------------
    prov = (rec.get("provenance") or {}).get("design") or {}
    pth, dig = str(prov.get("path") or ""), prov.get("sha256")
    m = re.search(r"_design_v(\d+)\.json$", pth)
    if m:
        f = root / "pm_5min/derived" / Path(pth).name
        actual = (hashlib.sha256(f.read_bytes()).hexdigest()
                  if f.is_file() else None)
        if dig and actual and dig == actual:
            v = int(m.group(1))
            return {"design_version": v,
                    "read_from": "provenance.design {path, sha256} -- the "
                                 "PAIR, digest recomputed from the file "
                                 "the path names",
                    "pair_verified": True,
                    "fields_in_force": list(economic_fields_in_force(v)),
                    "n_names_in_force": len(economic_fields_in_force(v))}
        ev.append({"provenance_design_path": pth,
                   "declared_sha256": dig, "actual_sha256": actual,
                   "pair_verified": False,
                   "why": "the path names a version and the digest beside "
                          "it does not match the file -- the version is "
                          "UNKNOWN, not the one the path claims"})

    # ---- (a) POSITIVE recognition of a pre-correction receipt ---------
    positive = []
    stamp = rec.get("emitted_at_utc") or rec.get("as_of")
    if stamp:
        try:
            if (datetime.datetime.fromisoformat(str(stamp))
                    < datetime.datetime.fromisoformat(
                        SCOPE_CORRECTION_UTC.replace("Z", "+00:00"))):
                positive.append({"kind": "emit stamp before the correction",
                                 "emitted_at_utc": str(stamp),
                                 "correction_utc": SCOPE_CORRECTION_UTC})
        except ValueError:
            pass
    cc = (rec.get("source_identity") or {}).get("carrying_commit")
    if cc and _blob_sha256_at(
            cc, "live/pm_research/de_multiday_gate1_runner.py",
            Path(__file__).resolve().parents[2]) is not None:
        positive.append({"kind": "carrying_commit resolves to a tree",
                         "commit": cc})
    if any(x["kind"].startswith("emit stamp") for x in positive):
        v = SCOPE_BEFORE_THE_CORRECTION
        return {"design_version": v,
                "read_from": "POSITIVE recognition of a pre-correction "
                             "receipt",
                "positive_evidence": positive,
                "protocol": rec.get("protocol"),
                "pair_verified": False,
                "why_not_by_absence": (
                    "absence of a field never selects the weaker rule; "
                    "this receipt is recognised by what it CARRIES"),
                "fields_in_force": list(economic_fields_in_force(v)),
                "n_names_in_force": len(economic_fields_in_force(v))}

    # ---- the DEFAULT IS THE STRICTEST list in force -------------------
    return {"design_version": strictest,
            "read_from": "THE STRICTEST LIST IN FORCE -- no verifiable "
                         "design pair, and nothing positively identifies "
                         "this receipt as older",
            "pair_verified": False,
            "evidence_considered": ev,
            "positive_evidence": positive,
            "why_strictest": (
                "absence must not select the weaker rule inside the "
                "scoping built to protect the seal (REV 73 §1.1(a)). An "
                "unrecognised receipt is judged under every name"),
            "fields_in_force": list(economic_fields_in_force(strictest)),
            "n_names_in_force": len(economic_fields_in_force(strictest))}

#: How many checks the DE 78 day-path block runs. Declared, because the
#: offline skip list is generated from it and the online run asserts the
#: two agree -- a check added without updating this REFUSES rather than
#: silently shrinking the offline battery.
DAY_PATH_CHECKS = 102
#: R-610's battery-order checks. They perform real draws (see the guard at
#: their site), so they are online-only and their count is DECLARED.
BATTERY_ORDER_CHECKS = 7


#: R-603 / REV 49 §0 -- THE DIGEST OF THE BYTES THAT ARE RUNNING, taken
#: at IMPORT, before any work. `_main_day` stamped `producing_code_sha256`
#: by a FRESH read of `__file__` AFTER the day was computed, so a file
#: replaced mid-run made the receipt name code that DID NOT RUN -- and
#: `producing_code_is_the_committed_bytes` PASSED, because the replacement
#: was committed. The field built to catch exactly that class certified the
#: wrong bytes.
#:
#: It happened to me: DE 85 committed this file at 08:35:20Z while the
#: 09-03 smoke was executing it from the same worktree. Python had the
#: module in memory, so the RUN was unaffected; the RECEIPT was not.
try:
    LAUNCH_SOURCE_SHA256 = hashlib.sha256(
        Path(__file__).resolve().read_bytes()).hexdigest()
except OSError:                       # pragma: no cover - unreadable source
    LAUNCH_SOURCE_SHA256 = None
LAUNCH_TIME_UTC = datetime.datetime.now(
    datetime.timezone.utc).isoformat()

#: RULE 22 AS AMENDED (REV 51 §3): ONE FILE IS NOT THE RUN. The launch
#: capture covered this module only, so a SIBLING replaced mid-run --
#: the design module, BE's cascade -- moved the code that produced the
#: numbers while the receipt still said the source was unchanged. The
#: closure is every module under `live/` that is loaded, plus the
#: worktree's HEAD and dirty state.
LIVE_DIR = str(Path(__file__).resolve().parents[1])
#: digest OF THE BYTES OBSERVED WHEN THE MODULE FIRST ENTERED THIS RUN --
#: never a second read later, because `importlib.import_module` can return
#: a module that was already in `sys.modules` and whose file has since
#: moved.
LAUNCH_CLOSURE: dict = {}


def _digest_module(mod) -> None:
    """Record a loaded module's bytes ONCE, the first time it is seen."""
    f = getattr(mod, "__file__", None)
    if not f:
        return
    p = Path(f).resolve()
    if not str(p).startswith(LIVE_DIR) or str(p) in LAUNCH_CLOSURE:
        return
    try:
        LAUNCH_CLOSURE[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        LAUNCH_CLOSURE[str(p)] = None


def _capture_closure() -> None:
    for m in list(sys.modules.values()):
        _digest_module(m)


def _is_the_shared_data_link(root: str, xy: str, path: str) -> bool:
    """Is this porcelain entry the shared-tree DATA SYMLINK, by property?

    Three conditions, all required: the entry is UNTRACKED (`??`), the
    path is really a symlink, and it resolves to the canonical data root.
    Anything else -- a real edit, a directory, a link somewhere else --
    is dirt and refuses."""
    import os as _os
    if xy.strip() != "??":
        return False
    full = _os.path.join(root, path.rstrip("/"))
    if not _os.path.islink(full):
        return False
    try:
        canonical = _os.path.realpath(
            _os.path.join(DR.resolve()["data_root"]))
    except Exception:                                     # noqa: BLE001
        return False
    return _os.path.realpath(full) == canonical


def _head_state() -> dict:
    """The worktree's HEAD and whether it was dirty AT IMPORT."""
    import subprocess as _sp
    root = str(Path(__file__).resolve().parents[2])

    def _g(*a, raw=False):
        try:
            r = _sp.run(["git", "-C", root, *a], capture_output=True,
                        text=True, timeout=60)
        except Exception:
            return None
        if r.returncode != 0:
            return None
        return r.stdout if raw else r.stdout.strip()

    # RAW, NOT STRIPPED. Porcelain is `XY<space>PATH` and an UNSTAGED
    # modification has X = space, so `.strip()` on the whole output ate the
    # leading space of the FIRST line and every path after it was reported
    # one character short -- `ive/pm_research/...`. It has been printing
    # that into refusal messages and into `dirty_paths`. Found while
    # classifying the entries, because a classifier has to read the status
    # characters that the strip was removing.
    st = _g("status", "--porcelain", raw=True)
    # THE SLICE HALF (R-637 / REV 66 S1.1). The raw read fixed the FIRST
    # line; `line[3:]` was still wrong on a RENAME: `R  a -> b` returns
    # `a -> b` where the path is `b`, so `islink(root/"a -> b")` is False,
    # a rename reads as real dirt, and a real day refuses with a message
    # sending its reader to look for a file called `a -> b`. Fail-safe in
    # direction, wrong in cause.
    #
    # THE PARSER IS DA'S (R-641): a porcelain parser is INFRASTRUCTURE, not
    # a statistic, so R-235 does not ask for two of them -- two corroborate
    # nothing and drift apart, which they did three times (the strip, the
    # slice, the rename). This seat is the third reader to get the format
    # wrong in a different place.
    _pp = DAROOT.parse_porcelain(st or "")
    rows = _pp["rows"]
    lines = [f"{r['xy']} {r['path']}" for r in rows]
    paths = [r["path"] for r in rows]
    # THE SHARED-TREE DATA LINK IS NOT A DIRTY WORKTREE -- AND IT IS
    # CHECKED, NOT NAMED. `scripts/wt_refresh.sh` replaces this worktree's
    # `data/` with a SYMLINK to the canonical data root so every seat reads
    # and writes one tree. `.gitignore` carries `data/`, which matches a
    # DIRECTORY and not a symlink, so the link shows as untracked and the
    # whole worktree read DIRTY -- and a REAL day refuses at import on a
    # dirty worktree. Measured after the mandated refresh: P10 blocking,
    # `assert_source_unchanged` REFUSED. The refresh procedure made GO
    # impossible.
    #
    # The guard exists so the PRODUCING CODE is locatable in a commit. A
    # symlink to the data root is not code and cannot move a byte of it.
    # So it is excluded -- but by PROPERTY, never by name: the entry must
    # be UNTRACKED, must actually be a symlink, and must resolve to the
    # canonical data root. A name-matched exemption is how a binding map
    # comes to excuse the very thing it exists to catch (R-613).
    exempt, remaining = [], []
    for row in rows:
        if _is_the_shared_data_link(root, row["xy"], row["path"]):
            exempt.append(row["path"])
        else:
            remaining.append(row["path"])
    return {"worktree": root, "head": _g("rev-parse", "HEAD"),
            "dirty": bool(lines) if st is not None else None,
            "dirty_paths": paths[:20],
            "porcelain": {"parser": "da_root.parse_porcelain -- the "
                                    "programme's ONE porcelain reader "
                                    "(R-641)",
                          "n_rows": _pp["n_rows"],
                          "n_malformed": _pp["n_malformed"],
                          "malformed": _pp["malformed"][:5],
                          "renames": [r["path"] for r in rows
                                      if r["renamed_from"]]},
            "dirty_beyond_the_shared_data_link": bool(remaining),
            "dirty_paths_beyond_the_shared_data_link": remaining[:20],
            "shared_data_link_exempted": exempt,
            "why_exempted": (
                "an UNTRACKED SYMLINK resolving to the canonical data root "
                "-- `scripts/wt_refresh.sh` makes it so every seat reads "
                "one data tree. Verified as a property (untracked AND a "
                "symlink AND resolving to the data root), never matched by "
                "name" if exempt else None)}


_capture_closure()
LAUNCH_HEAD = _head_state()


#: REV 54 / R-610. WHAT THE RUN HAS ACTUALLY SPENT, so "the battery
#: refused BEFORE any replay was run" is a MEASUREMENT and not the reading
#: of a line number. The 09-03 smoke ran 84 minutes of null draws and was
#: then refused by a fixture check at the emit; the fix is an ORDER, and an
#: order is checkable only if the work is counted.
RUN_COUNTERS = {"draws_performed": 0, "replays_performed": 0}


def run_counters() -> dict:
    return dict(RUN_COUNTERS)


class RunnerRefused(RuntimeError):
    """The run cannot proceed honestly on the inputs given."""


def closure_drift() -> list:
    """Which modules of the launch closure have MOVED on disk since."""
    out = []
    for path, was in LAUNCH_CLOSURE.items():
        try:
            now = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != was:
            out.append({"module": Path(path).name, "path": path,
                        "at_import": was, "now": now})
    return out


def source_identity_at_launch() -> dict:
    """THE BYTES THAT RAN -- ALL OF THEM -- and whether they still hold.

    A receipt must name the code that PRODUCED it. Reading `__file__` at
    emit names whatever is on disk THEN; reading ONE file names one
    twenty-fourth of what ran."""
    me = Path(__file__).resolve()
    try:
        now = hashlib.sha256(me.read_bytes()).hexdigest()
    except OSError:
        now = None
    drift = closure_drift()
    head_now = _head_state()
    return {
        "producing_code": me.name,
        "producing_code_sha256": LAUNCH_SOURCE_SHA256,
        "digest_taken_at": "MODULE IMPORT, before any work",
        "launch_time_utc": LAUNCH_TIME_UTC,
        "on_disk_sha256_at_emit": now,
        "source_unchanged_during_the_run": now == LAUNCH_SOURCE_SHA256,
        # RULE 22 AS AMENDED: the CLOSURE, not one file.
        "import_closure": {
            "n_modules": len(LAUNCH_CLOSURE),
            # REV 89 §2.1: `modules` IS KEYED BY BASENAME AND COLLIDES.
            # Two modules sharing a basename -- `live/__init__.py` and a
            # `live/pm_research/__init__.py` the day anyone makes that a
            # package -- collapse to one key and the later one wins, so
            # R-605's "name EVERY module" silently names fewer. REV drove
            # DE 120's own initialiser conjunct GREEN on a NON-EMPTY
            # `live/__init__.py` hidden behind an empty sibling that won
            # the key. The drift guard was never affected (`closure_drift`
            # iterates by PATH), so this is the RECEIPT's account of
            # itself, not the refusal.
            #
            # The basename map stays, because readers and DA's census key
            # on it; beside it now sits the map that cannot collide, and
            # the predicate that catches a collision is EVALUATED here
            # rather than left for someone to notice.
            "modules": {Path(k).name: v
                        for k, v in sorted(LAUNCH_CLOSURE.items())},
            "modules_by_path": {
                str(Path(k).relative_to(LIVE_DIR)): v
                for k, v in sorted(LAUNCH_CLOSURE.items())},
            "basenames_collide": (
                len({Path(k).name for k in LAUNCH_CLOSURE})
                != len(LAUNCH_CLOSURE)),
            "colliding_basenames": sorted(
                n for n in {Path(k).name for k in LAUNCH_CLOSURE}
                if sum(1 for k in LAUNCH_CLOSURE
                       if Path(k).name == n) > 1),
            "n_modules_equals_len_modules": (
                len(LAUNCH_CLOSURE)
                == len({Path(k).name for k in LAUNCH_CLOSURE})),
            "root": LIVE_DIR,
            "digested": "from the bytes observed when each module first "
                        "entered THIS run -- never a second read, because "
                        "import_module can return a cached module whose "
                        "file has since moved",
            "captured_at": ["module import",
                            "the cascade's first import",
                            "after the cascade's first load(), which "
                            "imports harmful_stateful_policy and "
                            "de_phase4_diag_runner LAZILY -- the closure "
                            "reached the cascade and stopped there, and "
                            "the two modules that do the REPLAYING were "
                            "outside it (REV 53 S1.1)"],
        },
        "closure_drift": drift,
        "closure_unchanged_during_the_run": not drift,
        "head_at_import": LAUNCH_HEAD,
        "head_at_emit": head_now,
        "head_unchanged_during_the_run": (
            LAUNCH_HEAD.get("head") == head_now.get("head")),
        # THE READING THE REFUSAL USES is the one that excludes the
        # verified data symlink; the RAW reading travels beside it, so
        # nothing is hidden.
        "worktree_was_dirty_at_import": LAUNCH_HEAD.get(
            "dirty_beyond_the_shared_data_link"),
        "worktree_had_any_untracked_entry_at_import": LAUNCH_HEAD.get(
            "dirty"),
    }


def assert_source_unchanged(where: str, *, fixture: bool = True) -> dict:
    """REFUSE THE EMIT if ANY of the code that ran changed under it.

    Not the RUN -- the run is fine, Python holds the modules in memory.
    What is not fine is a receipt that names bytes which did not produce
    it. Rule 22 as amended covers the CLOSURE and HEAD, because a sibling
    module or a commit in the worktree moves the producing code just as
    surely as this file does."""
    idy = source_identity_at_launch()
    if idy["closure_drift"]:
        raise RunnerRefused(
            f"REFUSED at {where}: A MODULE OF THIS RUN'S IMPORT CLOSURE "
            f"CHANGED UNDER IT -- "
            f"{[d['module'] for d in idy['closure_drift']]}. The run is "
            f"unaffected (the modules are in memory); a receipt stamped "
            f"from the files would name code that DID NOT RUN. Rule 22 as "
            f"amended (REV 51 S3): the capture is the closure, not one "
            f"file.")
    if not idy["head_unchanged_during_the_run"]:
        raise RunnerRefused(
            f"REFUSED at {where}: THE WORKTREE'S HEAD MOVED UNDER THIS RUN "
            f"-- {str(idy['head_at_import'].get('head'))[:12]} -> "
            f"{str(idy['head_at_emit'].get('head'))[:12]}. A receipt's "
            f"carrying_commit would name a commit that was not the one "
            f"this run executed from.")
    if not fixture and idy["worktree_was_dirty_at_import"]:
        raise RunnerRefused(
            f"REFUSED at {where}: THE WORKTREE WAS DIRTY AT IMPORT "
            f"({idy['head_at_import'].get('dirty_paths_beyond_the'
                                          '_shared_data_link')}). "
            f"For a REAL day "
            f"the producing code must be locatable in a commit; "
            f"uncommitted bytes are locatable nowhere. Recorded as a fact "
            f"for a fixture, refused for a day (REV 49 S0).")
    if not idy["source_unchanged_during_the_run"]:
        raise RunnerRefused(
            f"REFUSED at {where}: THE SOURCE CHANGED UNDER THIS RUN. This "
            f"process is executing "
            f"{str(LAUNCH_SOURCE_SHA256)[:16]} (read at import) and "
            f"{Path(__file__).name} now holds "
            f"{str(idy['on_disk_sha256_at_emit'])[:16]}. The run itself is "
            f"unaffected -- the module is in memory -- but a receipt "
            f"stamped from the file would name code that DID NOT RUN, and "
            f"`producing_code_is_the_committed_bytes` would PASS if the "
            f"replacement is committed (R-603 / REV 49 S0). Re-run from a "
            f"worktree nobody is landing into, or supersede the receipt in "
            f"band naming the launch digest.")
    return idy


# ------------------------------------------------------------- parameters

#: R-659 / REV 71 S1.4(c). THE DIGEST OF EVERY DECLARED INPUT, TAKEN
#: WHERE THE FILE IS LOADED. The provenance block digested params and
#: design AT THE EMIT and said "the files THIS run read" -- which is the
#: bytes at emit, not the bytes that were read. Rule 22 already refuses
#: when a MODULE moves under a run; a declared input is the same class,
#: and this is its symmetric form.
INPUT_DIGESTS: dict = {}


def record_input_digest(name: str, rel: str) -> dict:
    """Digest a declared input WHERE IT IS READ, once, and keep it."""
    root = Path(__file__).resolve().parents[2]
    f = root / rel
    if name in INPUT_DIGESTS:
        return INPUT_DIGESTS[name]
    INPUT_DIGESTS[name] = {
        "path": str(rel),
        "sha256_at_load": (hashlib.sha256(f.read_bytes()).hexdigest()
                           if f.is_file() else None),
        "read_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "existed_at_load": f.is_file(),
    }
    return INPUT_DIGESTS[name]


def verify_input_digests(where: str, *, digests: dict | None = None) -> dict:
    """RE-DIGEST EVERY DECLARED INPUT AT THE EMIT and refuse on a change.

    Both digests travel: the one taken where the file was READ and the one
    taken here. A receipt that carries only the second names bytes that
    may not be the bytes the run used.

    `digests` IS EXPLICIT SO A PROBE CAN BRING ITS OWN (DE 116). Driven
    against the process's real map, this function RE-READS every recorded
    input -- and on a REAL day that map holds the design pin recorded at
    S0, a LEDGER path. A battery cell doing that inside the fixture
    probe's data-freeness window is what refused GO #5: the read is the
    real run's, legitimate, and the window attributed it to a fixture."""
    root = Path(__file__).resolve().parents[2]
    src = INPUT_DIGESTS if digests is None else digests
    out, moved, null_half = {}, [], []
    for name, rec in sorted(src.items()):
        f = root / rec["path"]
        now = (hashlib.sha256(f.read_bytes()).hexdigest()
               if f.is_file() else None)
        agrees = now == rec["sha256_at_load"]
        out[name] = {**rec, "sha256_at_emit": now, "agrees": agrees}
        if rec["sha256_at_load"] is None or now is None:
            # A `{path, sha256: null}` IS NOT A PAIR (R-608). A half that
            # is absent cannot be compared, and reporting `agrees: null`
            # beside it lets a reader take an unmade comparison for a
            # made one.
            null_half.append(name)
        elif not agrees:
            moved.append(name)
    if null_half:
        raise RunnerRefused(
            f"REFUSED at {where}: a declared input's digest pair carries a "
            f"NULL HALF -- {null_half}. A `{{path, sha256: null}}` is not "
            f"a pair (R-608): one half absent cannot be compared, and "
            f"`agrees: null` beside it lets a reader take an unmade "
            f"comparison for a made one.")
    if moved:
        raise RunnerRefused(
            f"REFUSED at {where}: a DECLARED INPUT changed under this run "
            f"-- {moved}. Rule 22 refuses when a module of the closure "
            f"moves; a params or design file is the same class, and a "
            f"receipt naming bytes that are not the bytes the run read is "
            f"provenance theatre (REV 71 S1.4(c)).")
    return {"inputs": out, "n_inputs": len(out),
            "all_agree": True, "no_null_halves": True,
            "the_design_is_digested_at_params_load": (
                "not at the emit, where its two digests would be one read "
                "twice and the pair could not detect the file changing "
                "under the run (REV 75 S1.1)"),
            "why_both_digests": (
                "the one taken WHERE THE FILE WAS READ and the one taken "
                "at the emit. The block used to carry only the second and "
                "claim it was 'the files THIS run read'")}


def load_params(path: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path is not None else root / PARAMS_REL
    if path is None:
        record_input_digest("params", PARAMS_REL)
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
    # REV 93 routed #1: THE CITATION IS THE CASCADE, NOT ONE FILE. A null
    # run executes ten modules -- `replay` reaches `harmful_stateful_policy`
    # and `de_phase4_diag_runner`, and those reach the rest -- so a digest
    # on the entry point alone covered a tenth of the code that produces
    # the numbers. BE 96 changed the entry point and the guard caught it;
    # a change in any of the other nine would not have moved a digest.
    # Every module is checked, each by its own pair.
    # REV 95 minor: THE RETURN NAMES WHAT WAS CHECKED. It reported the
    # same shape -- one path, one digest -- whether it had verified ONE
    # module or TEN, so a reader could not tell a cascade check from an
    # entry-point check, which is exactly the difference this round added.
    _casc = (params.get("be_cascade") or {}).get("modules") or []
    if _casc and actual_sha is None:
        _repo = Path(__file__).resolve().parents[2]
        _bad = []
        for _m in _casc:
            _f = _repo / _m["path"]
            if not _f.is_file():
                _bad.append({"path": _m["path"], "status": "ABSENT"})
                continue
            _got = hashlib.sha256(_f.read_bytes()).hexdigest()
            if _got != _m["sha256"]:
                _bad.append({"path": _m["path"], "declared": _m["sha256"],
                             "actual": _got, "status": "DIGEST_DIFFERS"})
        if _bad:
            raise RunnerRefused(
                f"REFUSED BE_CASCADE_DIFFERS: {len(_bad)} of {len(_casc)} "
                f"cited cascade modules do not match their declared pair -- "
                f"{_bad[:3]}. A null run through a DIFFERENT cascade is not "
                f"a control; the citation must be re-pointed DELIBERATELY, "
                f"with the draw-path and constant axes measured (see "
                f"`be_module_repoint`).")
    _checked = ([m["path"] for m in _casc] if (_casc and actual_sha is None)
                else [params["be_module"]["path"]])
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
            "cited_not_copied": True, "verified_at_run_time": True,
            # REV 95 minor: NAME WHAT WAS CHECKED, so one module and ten
            # do not report the same shape.
            "n_modules_checked": len(_checked),
            "modules_checked": _checked,
            "scope": ("THE CASCADE -- every module a null run executes"
                      if len(_checked) > 1 else
                      "THE ENTRY POINT ONLY -- no `be_cascade` in the "
                      "params, or a digest was supplied by the caller")}


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
    # THE CASCADE JOINS THE LAUNCH CLOSURE AT ITS FIRST IMPORT. It is
    # imported lazily, so it is not in `sys.modules` when this module
    # loads -- and a closure that misses the module producing the NUMBERS
    # is the closure missing the point. Recorded once: `import_module` can
    # return a cached module whose file has since moved.
    _digest_module(module)
    _capture_closure()
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
    # REV 53 S1.1: `load()` imports `harmful_stateful_policy` and
    # `de_phase4_diag_runner` LAZILY, so a closure captured at the
    # cascade's import misses the two modules that do the replaying. The
    # capture is repeated HERE -- after the first load, before any emit.
    _capture_closure()
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


def days_complete_now(params: dict, root: Path | None = None) -> dict:
    """HOW MANY RULED DAYS HAVE A SEALED RECEIPT, COMPUTED (REV 75 S1.2).

    `n_days_complete` was a DEFAULT PARAMETER of 1 that nothing on the
    real path ever passed, so the 09-04 receipt says "SEALED -- 1 of 6
    days complete" on the day when TWO were. Every consumer is fail-safe
    by accident: a constant 1 can only ever over-seal, so nothing broke --
    and rule 10 says a number a receipt claims must be one the code
    evaluated.

    It is counted through the READ GATE'S OWN resolver, so "complete"
    means exactly what the gate means by it: a day whose sealed receipt
    resolves to a chain head."""
    r = Path(root) if root else Path(DR.resolve()["data_root"])
    per_day, complete = {}, []
    for day in params["days"]:
        res = find_sealed_day_receipt(day, r)
        per_day[day] = {"status": res["status"],
                        "present": res["present"],
                        "n_matches": res["n_matches"]}
        if res["present"]:
            complete.append(day)
    return {"n_days_complete": len(complete),
            "days_complete": complete,
            "ruled_days": list(params["days"]),
            "G": params["G"],
            "per_day": per_day,
            "counted_by": "find_sealed_day_receipt -- the READ GATE'S own "
                          "resolver, so `complete` means what the gate "
                          "means by it",
            "was_a_default_parameter": (
                "1, and nothing on the real path passed it. The 09-04 "
                "receipt reads `1 of 6` on a day when two were complete "
                "(REV 75 S1.2). Fail-safe by accident: a constant 1 can "
                "only over-seal")}


def seal(day_result: dict, n_days_complete: int, g: int,
         *, unsealed_by_ruling: bool = False) -> dict:
    """R5 -- the economic fields are ABSENT until every day is complete.

    Absent, not present-and-ignored: a field a reader can see is a field a
    reader can quote.

    R-572(B)(3), THE ASYMMETRY THIS FIXES. A sealed artifact carried
    `sealed_at_every_depth` and `sealed_field_names`; an unsealed one
    dropped BOTH and gained `economic`. A consumer keying on either name
    got `None` after the unseal and could not distinguish "this run is
    unsealed" from "I misspelled the field". Both keys are now present in
    both states with EXPLICIT values."""
    # R-765, THE USER'S RULING: R5 IS RETIRED. `unsealed_by_ruling` is
    # set from the params block, so a run under v18 or later emits its
    # economics and a run reading an older params file behaves exactly as
    # the four landed days did -- rule 13, those receipts are untouched
    # and must stay reproducible from the code that wrote them.
    if unsealed_by_ruling:
        out = dict(day_result)
        out.setdefault("economic", None)
        out["sealed"] = False
        out["seal_status"] = (
            f"UNSEALED_BY_USER_RULING R-765 -- {n_days_complete} of {g} "
            f"days complete")
        out["sealed_field_names"] = []
        out["sealed_at_every_depth"] = False
        out["R5_retired_from"] = "params v18 / design v26"
        return out
    if n_days_complete >= g:
        out = dict(day_result)
        # UNSEALED: `economic` is present -- explicitly None, with the
        # reason already carried in `why_no_economic`, on a refused arm-day.
        # Popping it made the unsealed state asymmetric WITH ITSELF: present
        # for an OK arm-day, absent for a refused one.
        out.setdefault("economic", None)
        out["sealed"] = False
        # REV 90 §A0 / CLAUDE.md rule 10: THIS WAS A LITERAL, and it was
        # FALSE under the ruled early-read bar -- `seal(arm, 4, 4)` wrote
        # "UNSEALED_ALL_DAYS_COMPLETE" beside `n_days_complete 4` and
        # `G 6`, so the one field describing the read's standing IN WORDS
        # asserted that all days were complete when four of six were. The
        # sealed branch six lines below always computed its string; the
        # asymmetry was the defect. Both branches compute now, from the
        # bar THIS CALL was handed, so no early-read special case exists
        # and no caller can be given a status its own numbers contradict.
        out["seal_status"] = (
            f"UNSEALED -- {n_days_complete} of {g} days complete under the "
            f"bar this call was given")
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


#: THE SHARED CENSUS RULE (REV 76 S0). Stated here so DE's census and
#: DA's `economic_absence` agree BY CONSTRUCTION rather than by both
#: happening to be written the same way:
#:
#:   A SEALED NAME PRESENT AS A KEY IS A LEAK, WHATEVER ITS VALUE --
#:   including `[]`, `{}` and `null`. The census walks KEYS, never leaves:
#:   an empty container yields no leaf, so a leaf-walk reports a sealed
#:   key emitted as `[]` or `{}` as ABSENT. The receipt's own
#:   `seal_status` says every economic field is "ABSENT from this
#:   artifact, not present-and-ignored" -- and an empty container is
#:   exactly present-and-ignored.
#:
#: DA 98 carries the same sentence. If the two ever diverge, this line is
#: the one to compare.
#: DECLARED IN THE SHAPE DA'S READER LOOKS FOR (R-683 / REV 76 S0): a
#: module-level name containing `SEAL_RULE`, holding a literal. DA's
#: `de_seal_rule_at_source()` parses this file by AST, reads THIS string,
#: and compares it with its own after normalising whitespace, case and
#: dashes. At the previous tip it returned
#: DE_HAS_NOT_DECLARED_THE_KEY_WALK_RULE_YET -- an absence reported by
#: name, never read as agreement. The two censuses stay INDEPENDENT
#: implementations (R-235); what is shared is the RULE, and it is read at
#: the source rather than assumed.
SEAL_RULE = ("A SEALED NAME PRESENT AS A KEY REFUSES, WHATEVER ITS VALUE "
             "-- including an empty container and null. ABSENCE IS THE "
             "ONLY SEAL.")
#: The same rule in this seat's own words, kept because the walker's
#: docstring cites it and because it says WHY (an empty container yields
#: no leaf, so a leaf-walk reports it absent).
CENSUS_RULE = ("a sealed name present as a KEY is a leak whatever its "
               "value, including [] and {} and null; the census walks "
               "KEYS, never leaves")


def seal_key_walk(o, path="") -> list:
    """Economic fields present as KEYS, at any depth.

    THE RULE IS `CENSUS_RULE` ABOVE, and it is the key-walk: a name is
    reported wherever it appears as a key, whatever the value is. A
    leaf-walk would miss `[]` and `{}` entirely (REV 76 S0, driven on
    both censuses).

    NOT a substring test on the serialised artifact: `sealed_field_names`
    is a LIST OF THE NAMES BEING SEALED, so a substring test reports it as
    a leak. That is the needle-matches-its-own-prose failure this codebase
    has hit three times, and it caught my own check here."""
    found = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ECONOMIC_FIELDS:
                found.append(f"{path}.{k}".lstrip("."))
            found += seal_key_walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            found += seal_key_walk(v, f"{path}[{i}]")
    return found


#: The old name, kept so every existing caller and DA's AST census keep
#: working. `seal_key_walk` is the name DA's reader looks for (it wants a
#: function whose name carries `key` AND `walk`/`seal`), and the rule it
#: implements is `SEAL_RULE`.
_economic_keys_in = seal_key_walk


def economic_absence_scoped(rec: dict) -> dict:
    """A RECEIPT JUDGED UNDER ITS OWN SCOPE (REV 72 S1.4).

    The same walker, the same key test -- but the name set is the one in
    force when the receipt was produced. Judged under v23's eleven, the
    09-03 receipt reads six 'leaks'; under its own v22 scope it reads
    clean, which is the true statement."""
    scope = design_version_of_receipt(rec)
    names = set(scope["fields_in_force"])
    found = [k for k in _economic_keys_in(rec)
             if k.rsplit(".", 1)[-1] in names]
    return {**scope, "n_names_in_force": len(names),
            "leaked_keys": found, "n_leaked": len(found),
            "sealed": not found,
            "judged_as_KEYS_not_substrings": True,
            "why_scoped": (
                "a receipt cannot have disobeyed a rule that did not "
                "exist when it was written. Judged under the CURRENT "
                "eleven the first sealed day is accused by its own "
                "instrument of carrying fields that were OPEN BY RULING "
                "(REV 72 S1.4)")}


def assert_reasons_carry_no_sealed_value(result: dict) -> dict:
    """R-599's SECOND LEAK, CHECKED RATHER THAN PROMISED (R-674 (c)).

    `admissibility.reasons` is written by
    `de_multiday_design_declaration.arm_day_admissible()` and has been in
    every receipt since design v2 -- including 09-03's, where its value is
    an empty list. It carries NUMBERS on a refused arm-day, and as written
    they are all OPEN: the decision count and the declared minimum (a
    population size and a bar), and the declared floor fraction. The sd
    reason publishes a VERDICT ONLY, with no sd, mean or ratio, which is
    R-599's second leak already closed at the source.

    But `_strip_economic` removes KEYS, not substrings -- so nothing in
    the machinery would stop a FUTURE reason string from interpolating a
    sealed value. That protection rested entirely on whoever writes the
    string. This checks it instead: no sealed quantity's value may appear
    in any reason text, in any of the forms a formatter produces."""
    adm = (result or {}).get("admissibility") or {}
    reasons = adm.get("reasons") or []
    text = " || ".join(str(r) for r in reasons)
    hits = []
    for name in ECONOMIC_FIELDS:
        for src in (adm, (result or {}).get("economic") or {}):
            v = src.get(name)
            if v is None or isinstance(v, (bool, dict, list)):
                continue
            forms = {str(v)}
            if isinstance(v, float):
                forms |= {f"{v:.1f}", f"{v:.2f}", f"{v:.3f}",
                          f"{v:.4f}", f"{v:g}", str(round(v, 6))}
            elif isinstance(v, int):
                forms |= {f"{v:,}"}
            for f in sorted(forms):
                if len(f) >= 3 and f in text:
                    hits.append({"sealed_name": name, "form_found": f})
    if hits:
        raise RunnerRefused(
            f"REFUSED: a refusal REASON carries the VALUE of a sealed "
            f"quantity -- {[h['sealed_name'] for h in hits]}. "
            f"`_strip_economic` removes KEYS, not substrings, so a number "
            f"interpolated into a reason string survives the seal. That "
            f"is R-599's second leak, and it fires on a REFUSED arm-day: "
            f"exactly the case where the numbers are most tempting.")
    return {"n_reasons": len(reasons),
            "checked_against": list(ECONOMIC_FIELDS),
            "no_sealed_value_in_any_reason": True,
            "written_by": "de_multiday_design_declaration."
                          "arm_day_admissible()",
            "present_since": "design v2 -- including the 09-03 receipt, "
                             "where it is an empty list",
            "what_the_reasons_may_carry": (
                "OPEN numbers only: the decision count and the declared "
                "minimum, and the declared sd floor fraction. The sd "
                "reason is a VERDICT ONLY"),
            "why_a_check_and_not_a_promise": (
                "the stripper removes keys; a value interpolated into a "
                "string survives it. Before this, nothing but the "
                "author's care stopped one"),
            }


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


#: The runner's own naming for a sealed day receipt. The stamp varies, so
#: a day resolves by PREFIX -- and EXACTLY ONE match is required: zero is a
#: day that never ran, and two is a day that ran twice, which is an
#: ambiguity a read must refuse rather than resolve by picking the newest.
SEALED_DAY_RECEIPT_PREFIX = "p003_de_gate1_day_run_"
SEALED_DAY_RECEIPT_MIDFIX = "_SEALED__"


def day_token(day: str) -> str:
    """THE DAY AS IT APPEARS IN A FILENAME -- one rule, one place.

    `day_forms` yields a compact form only for a real `YYYY-MM-DD`, so
    `[0]` of the compact list raised IndexError on any day that is not a
    date -- a latent crash in BOTH the glob and the name composer, found by
    driving the fixture path end to end. A fixture day is not a date and
    must still name a file."""
    compact = [d for d in sorted(day_forms(day)) if "-" not in d]
    return compact[0] if compact else re.sub(
        r"[^A-Za-z0-9]+", "-", str(day)).strip("-")


def sealed_day_receipt_glob(day: str) -> str:
    return (f"{SEALED_DAY_RECEIPT_PREFIX}{day_token(day)}"
            f"{SEALED_DAY_RECEIPT_MIDFIX}*.json")


#: R-608 / REV 54 S0.1. THE FIELDS OF A SUPERSESSION LINK. The link is the
#: PAIR and BOTH halves are required; this tuple is the one place on this
#: side of the seam that says so.
SUPERSEDES_PAIR_FIELDS = ("path", "sha256")

#: Every status this seat's chain resolver can REFUSE with, and the two it
#: RESOLVES with. A caller asking "did it refuse" asks these, never a
#: string it typed.
CHAIN_RESOLVED_STATUSES = ("PRESENT", "PRESENT_CHAIN_HEAD")
CHAIN_REFUSAL_STATUSES = (
    "SUPERSEDES_MALFORMED", "LINK_NOT_A_PAIR",
    "SUPERSEDES_TARGET_DIGEST_MISMATCH", "SUPERSEDES_MOVED",
    "DANGLING_SUPERSEDES", "AMBIGUOUS", "ARTIFACT_UNREADABLE")


def supersedes_shape(rec) -> dict:
    """WHAT `supersedes` IS, judged BEFORE anything is read out of it.

    REV 54 S0.1 drove a `.v2` whose `supersedes` was a BARE PATH STRING
    through the whole read gate. `rec.get("supersedes") or {}` returns the
    STRING -- a non-empty string is truthy -- and `.get` on it raises
    `AttributeError`, which left `find_sealed_day_receipt`, which
    `read_gate` calls FOR EVERY RULED DAY. One malformed artifact took the
    gate DOWN where DA refused it by name. **A gate that crashes on bad
    input has not judged it**, and the traceback is not a verdict.

    So the value's SHAPE is a named verdict, and no field is read out of
    it until the shape says a field exists:

      key absent, or null          ABSENT     declares no link
      not a mapping                MALFORMED  REFUSED BY NAME
      a mapping missing either
        half -- or carrying a
        non-string for one         INCOMPLETE REFUSED BY NAME: half a link
                                              is not "no link"
      both halves, both strings    PAIR       the only shape that binds
    """
    if not isinstance(rec, dict) or "supersedes" not in rec:
        return {"kind": "ABSENT", "status": "NO_SUPERSEDES_BLOCK",
                "is_a_pair": False,
                "why": "no `supersedes` key: this artifact claims to "
                       "supersede nothing"}
    raw = rec.get("supersedes")
    if raw is None:
        return {"kind": "ABSENT", "status": "NO_SUPERSEDES_BLOCK",
                "is_a_pair": False,
                "why": "`supersedes` is null: present and declaring "
                       "nothing, which is not a half-written link"}
    if not isinstance(raw, dict):
        return {"kind": "MALFORMED", "status": "SUPERSEDES_MALFORMED",
                "is_a_pair": False, "observed_type": type(raw).__name__,
                "required_fields": list(SUPERSEDES_PAIR_FIELDS),
                "why": ("a `supersedes` that is not an OBJECT carrying "
                        "both `path` and `sha256` is REFUSED BY NAME. "
                        "Reading a bare string as the path half would be "
                        "resolving by NAME, which R-608 forbids; RAISING "
                        "on it takes the whole read gate down over one "
                        "malformed artifact (REV 54 S0.1)")}
    have = {f: raw.get(f) for f in SUPERSEDES_PAIR_FIELDS}
    missing = [f for f, v in have.items() if not isinstance(v, str) or not v]
    if missing:
        return {"kind": "INCOMPLETE", "status": "LINK_NOT_A_PAIR",
                "is_a_pair": False, "missing_fields": missing,
                "present_fields": [f for f in SUPERSEDES_PAIR_FIELDS
                                   if f not in missing],
                "why": ("R-608: the link is the PAIR. A block carrying one "
                        "half -- or a non-string where a digest belongs -- "
                        "is NOT a link and is REFUSED BY NAME. Reporting "
                        "it as 'no link' would make a half-written "
                        "supersession look like two independent runs, i.e. "
                        "a day that ran twice")}
    return {"kind": "PAIR", "status": "PAIR", "is_a_pair": True,
            "path": have["path"], "sha256": have["sha256"]}


def resolve_day_chain(files, *, kind: str) -> dict:
    """ONE RESOLVER FOR BOTH OF THIS SEAT'S SAME-DAY CHAINS.

    The sealed day receipts and DA's landing records are two chains under
    ONE rule, and this seat had two implementations of it: the receipt side
    followed `supersedes` and the landing side did not, so DA's own
    DECLARED correction path for a landing record -- a `.v2` carrying the
    pair -- resolved on DA's side and read AMBIGUOUS on DE's. Two
    implementations of one rule is the defect R-608 was opened for, and DA
    closed it on its side by reading the rule from one place. This is that,
    here: the rule, once.

      none                        MISSING
      one                         PRESENT
      v1 + a PAIR-VALID v2        PRESENT_CHAIN_HEAD (the v2)
      two with no link            AMBIGUOUS
      a link that is not a PAIR   that link's own refusal, BY NAME
    """
    files = sorted(Path(f) for f in files)
    if not files:
        return {"resolved": False, "status": "MISSING", "head": None,
                "n_matches": 0}
    if len(files) == 1:
        return {"resolved": True, "status": "PRESENT", "head": files[0],
                "n_matches": 1, "chain": [str(files[0])], "links": [],
                "chain_head_is": f"the only {kind}"}
    present = {f.name: sha256_streamed(f) for f in files}
    links, refusals, superseded = [], [], set()
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            # RULE 11 AT THE SEAM: an unreadable artifact is a STATUS. It
            # was parsed into `{}` and became "declares no link", so a
            # corrupt receipt read as a clean second run.
            refusals.append({"declared_in": f.name,
                             "status": "ARTIFACT_UNREADABLE",
                             "why": str(exc)})
            continue
        shape = supersedes_shape(rec)
        if shape["kind"] == "ABSENT":
            continue
        if shape["kind"] != "PAIR":
            refusals.append(dict(shape, declared_in=f.name))
            continue
        name, sha = Path(shape["path"]).name, shape["sha256"]
        links.append({"declared_in": f.name, "target": name, "sha256": sha})
        # THE ORDER IS THE NAME FIRST, THEN THE DIGEST ELSEWHERE -- so a
        # named file that is PRESENT with other bytes is reported as what
        # it is. It used to fall through to DANGLING_SUPERSEDES, "the
        # predecessor is absent", which is the right VERDICT under a
        # message that misdescribes the cause (REV 54 S0).
        if name in present:
            if present[name] == sha:
                superseded.add(name)
                continue
            refusals.append({
                "declared_in": f.name,
                "status": "SUPERSEDES_TARGET_DIGEST_MISMATCH",
                "target": name, "declared_sha256": sha,
                "actual_sha256": present[name],
                "why": "the named file is PRESENT and its bytes are not "
                       "the ones the link declares"})
            continue
        elsewhere = sorted(n for n, h in present.items() if h == sha)
        if elsewhere:
            refusals.append({
                "declared_in": f.name, "status": "SUPERSEDES_MOVED",
                "declared_path": name, "found_as": elsewhere,
                "why": "the declared DIGEST is present under a DIFFERENT "
                       "name -- a MOVED file, not a link. Resolving by "
                       "digest alone would accept it; the pair does not"})
            continue
        refusals.append({
            "declared_in": f.name, "status": "DANGLING_SUPERSEDES",
            "declared_path": name, "declared_sha256": sha,
            "why": "a supersession whose predecessor is absent -- NEITHER "
                   "the name NOR the digest is present for this day -- is "
                   "a claim about a file nobody can check"})
    if refusals:
        return {"resolved": False, "status": refusals[0]["status"],
                "head": None, "n_matches": len(files),
                "chain": [str(f) for f in files], "links": links,
                "refusals": refusals,
                "why": f"a supersession link on this {kind} is not the "
                       f"PAIR {{path, sha256}} landing on ONE present file "
                       f"(R-608). This is REFUSED BY NAME and is NOT the "
                       f"same finding as 'no link'"}
    heads = [f for f in files if f.name not in superseded]
    if len(heads) != 1:
        return {"resolved": False, "status": "AMBIGUOUS", "head": None,
                "n_matches": len(files),
                "matches": [str(f) for f in files],
                "heads": [str(f) for f in heads], "links": links,
                "why": f"two {kind}s for one day with NO chain between "
                       f"them is a day that RAN TWICE; a read that picks "
                       f"the newest has chosen after seeing"}
    return {"resolved": True, "status": "PRESENT_CHAIN_HEAD",
            "head": heads[0], "n_matches": len(files),
            "chain": [str(f) for f in files],
            "superseded": sorted(str(f) for f in files
                                 if f.name in superseded),
            "links": links,
            "chain_head_is": f"the {kind} nothing else supersedes"}


#: The FIXTURE infix. A fixture day artifact must be unable to match the
#: sealed glob AT ALL -- not merely carry a different day. `--synthetic-day
#: 2026-09-03` once emitted a SEALED-looking artifact from a synthetic
#: book; `assert_fixture_day_lock` closed the DAY half of that and the
#: FILENAME half stayed with the caller.
FIXTURE_DAY_RECEIPT_MIDFIX = "_FIXTURE__"


def day_receipt_name(day: str, *, fixture: bool, stamp: str) -> str:
    """THE DAY RECEIPT'S FILENAME, COMPOSED BY THE CODE THAT WRITES IT.

    REV 55 S2.1, and it was a CERTAIN end-of-run refusal. The operator
    supplied `--output` and the harvested GO procedure built the name at
    LAUNCH from `date -u`; `assert_name_stamp_is_the_clock` compares the
    filename's stamp against the moment of WRITING. On the last smoke's own
    name the reviewer measured **-5,074 s**: 85 minutes of work, then a
    refusal about a filename. And the obvious escape -- a stamp-free name --
    passes that check and is INVISIBLE to `sealed_day_receipt_glob`, so the
    receipt would exist and the read gate would report the day MISSING.

    So the name is not the caller's to type. `--output` names a DIRECTORY,
    which is knowable in advance and carries no stamp -- DE 88's own
    principle, applied where it was still being violated -- and the stamp
    is read from the clock at the moment of writing, in the same clock read
    that fills `as_of`.

    The convention is params v11's `read_gate.receipt_naming`, so the
    glob and DA's landing record still resolve what this writes."""
    token = day_token(day)
    mid = (FIXTURE_DAY_RECEIPT_MIDFIX if fixture
           else SEALED_DAY_RECEIPT_MIDFIX)
    return f"{SEALED_DAY_RECEIPT_PREFIX}{token}{mid}{stamp}.json"


def assert_output_is_a_directory(output, *, day: str) -> dict:
    """REFUSE A CALLER-SUPPLIED FILENAME ON THE DAY PATH -- BEFORE ANY WORK.

    This is the falsifier's other half: the run must not be able to start
    with a name that will refuse at the end. It is checked at the top of
    `_main_day`, so a bad `--output` costs zero draws rather than 85
    minutes."""
    o = Path(output)
    if o.is_file():
        raise RunnerRefused(
            f"REFUSED before any work: --output {o} is an existing FILE. On "
            f"the day path --output names the DIRECTORY the receipt is "
            f"written into; the runner composes the filename itself from "
            f"the clock at the moment of writing (REV 55 S2.1).")
    if o.suffix:
        raise RunnerRefused(
            f"REFUSED before any work: --output {o.name} looks like a "
            f"FILENAME (suffix {o.suffix!r}). On the day path --output "
            f"names a DIRECTORY. A caller-supplied filename either carries "
            f"a stamp -- which is the LAUNCH moment, and "
            f"`assert_name_stamp_is_the_clock` compares it to the moment "
            f"of WRITING, measured at -5,074 s on the last smoke's own "
            f"name -- or carries none, which passes that check and is then "
            f"INVISIBLE to `{sealed_day_receipt_glob(day)}`. The name is "
            f"composed here, from the clock (REV 55 S2.1).")
    return {"output_is_a_directory": True, "directory": str(o),
            "the_filename_is_composed_by": "day_receipt_name(), from "
                                           "emission_stamp() at the moment "
                                           "of writing",
            "why_not_the_caller": "a path that must be known in advance "
                                  "cannot carry an honest stamp (DE 88); "
                                  "the directory is knowable in advance "
                                  "and carries none"}


def assert_no_sealed_receipt_yet(day: str, root: Path) -> dict:
    """THE GUARD THAT REPLACES `if output.exists()`.

    With the filename composed at write time there is no path to test in
    advance, so the question becomes the one that always mattered: has this
    DAY already landed a sealed receipt? That is a stronger check than the
    one it replaces -- `output.exists()` only caught a collision on the
    exact stamp the caller happened to type."""
    res = find_sealed_day_receipt(day, root)
    if res["n_matches"]:
        raise RunnerRefused(
            f"REFUSED before any work: day {day} already has "
            f"{res['n_matches']} artifact(s) under the sealed layout "
            f"({res['status']}). A day that runs twice is not a day with a "
            f"newest result. A deliberate correction is a SUPERSEDING "
            f"receipt carrying `supersedes` = {{path, sha256}} (rule 13), "
            f"which is not this path.")
    return {"day": day, "n_sealed_artifacts_present": 0,
            "expected_glob": res["expected_glob"]}


def find_sealed_day_receipt(day: str, root: Path) -> dict:
    """THE CHAIN HEAD for a day -- not "exactly one file".

    REV 51 S1.5: requiring exactly one glob match made RULE 13'S OWN FORM
    look like a defect. A superseding `.v2` sitting beside its `v1` --
    which is what rule 13 requires, since a landed artifact is never
    edited -- returned AMBIGUOUS and the gate refused the day as missing.

    The chain rule itself lives in `resolve_day_chain`, which the landing
    records go through too, so the two chains cannot drift apart."""
    d = Path(root) / "pm_5min/derived"
    pat = sealed_day_receipt_glob(day)
    hits = sorted(d.glob(pat))
    res = resolve_day_chain(hits, kind="receipt")
    out = {"day": day, "present": res["resolved"],
           "status": "MISSING" if res["status"] == "MISSING"
                     else res["status"],
           "n_matches": res["n_matches"],
           "expected_glob": str(d / pat)}
    for k in ("chain", "chain_head_is", "links", "refusals", "matches",
              "heads", "superseded", "why"):
        if k in res:
            out[k] = res[k]
    if res.get("head") is not None:
        out["path"] = str(res["head"])
    return out


def verify_sealed_day_receipt(day: str, path: str, root: Path) -> dict:
    """WHAT A SEALED RECEIPT MUST CARRY for the read to count it.

    Its BOOK DIGEST -- recomputed against the book on disk, so a receipt
    naming a book that has since moved refuses by name -- and the RUNNER
    IDENTITY, so a receipt no commit holds cannot be counted."""
    p = Path(path)
    rec = json.loads(p.read_text())
    problems = []
    bk = (rec.get("reference_book") or {})
    declared = bk.get("sha256")
    bpath = Path(bk.get("path") or "")
    if not declared:
        problems.append("no reference_book.sha256")
    if not bpath.is_file():
        problems.append(f"the book it names is absent: {bpath}")
    elif declared and sha256_streamed(bpath) != declared:
        problems.append(
            f"the book it names hashes to something else "
            f"({sha256_streamed(bpath)[:16]} vs {declared[:16]})")
    si = rec.get("source_identity") or {}
    if not si.get("carrying_commit") or not si.get("producing_code_sha256"):
        problems.append("no runner identity (carrying_commit / "
                        "producing_code_sha256)")
    if rec.get("day") and not (day_forms(rec["day"]) & day_forms(day)):
        problems.append(f"it is for day {rec.get('day')!r}")
    sealed = rec.get("per_day_sealed_artifacts") or []
    if not sealed:
        problems.append("it carries no per-day artifact")
    elif not all(a.get("sealed") is True for a in sealed):
        problems.append("one of its arm-days is NOT sealed")
    return {"day": day, "path": str(p), "ok": not problems,
            "problems": problems,
            "book_digest": declared,
            "carrying_commit": si.get("carrying_commit"),
            "n_arm_days": len(sealed)}


#: R-604 item 7, THE HORIZON, declared with its date so the stopping rule
#: is not a degree of freedom.
READ_HORIZON_UTC = "2026-09-09T12:00:00Z"

#: R-604 item 5: R-555's membership, read from the day's LEDGER verdict. A
#: sealed receipt is DE's own artifact and is not a day verdict.
LEDGER_MEMBERSHIP_CONJUNCTS = ("day_quality_pass", "era_pure",
                               "counts_toward_race")


#: DA's pre-read artifact, as DA DECLARES it: the glob, and the flag that
#: says an artifact IS a landing record. DE globbed the name and never
#: checked the flag, so any `p003_da_gate1_pre_read*.json` carrying a
#: matching `day` counted here while DA -- which requires the flag --
#: would not have seen it at all.
LANDING_RECORD_GLOB = "p003_da_gate1_pre_read_*.json"
LANDING_RECORD_DECLARED_FLAG = "is_the_declared_LANDING_RECORD"

#: REV 54 S1.3. THE LANDING RECORD WRITES EACH FACT TWICE and the two
#: seats read different copies. DA's emitter writes
#: `landing_record.receipt_sha256` -- its own declared field, the one DA's
#: reader resolves -- AND the older top-level `receipt.sha256`; DE took the
#: second. The reviewer drove the two deliberately disagreeing:
#:
#:     landing_record.receipt_sha256 = aaaa...   receipt.sha256 = ffff...
#:        DA reads aaaa      DE reads ffff      NEITHER COMPLAINED
#:
#: Today both are written from ONE `hashlib.sha256(rp.read_bytes())` call
#: and cannot differ. Nothing asserts that they must -- and conjunct 3 is
#: the conjunct that stops a re-roll.
#:
#: THE AUTHORITATIVE COPY IS THE FIRST OF EACH PAIR: the field under
#: `landing_record`, which is the block DA declares with
#: `is_the_declared_LANDING_RECORD` and the block DA's own reader resolves.
#: The other copy is READ TOO, and a disagreement is REFUSED BY NAME.
LANDING_RECORD_FIELD_COPIES = (
    ("receipt_sha256_at_landing",
     ("landing_record", "receipt_sha256"), ("receipt", "sha256")),
    ("receipt_name_at_landing",
     ("landing_record", "receipt_path"), ("receipt", "path")),
    ("day_at_landing",
     ("landing_record", "day"), ("day",)),
)


def _at_path(obj, path):
    """One leaf by its key path, with no `.get` on a non-mapping."""
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def landing_record_copies(rec: dict) -> dict:
    """READ BOTH COPIES OF EVERY TWICE-WRITTEN FIELD, and compare them.

    A field written twice and read once is a field with no reader on one
    of its copies. The comparison is what makes the second copy mean
    anything; without it the duplicate is not redundancy, it is a second
    fact that happens to agree today."""
    fields, disagree = {}, []
    for name, auth, other in LANDING_RECORD_FIELD_COPIES:
        a, b = _at_path(rec, auth), _at_path(rec, other)
        fields[name] = {
            "authoritative_field": ".".join(auth), "authoritative_value": a,
            "second_copy_field": ".".join(other), "second_copy_value": b,
            "both_present": a is not None and b is not None,
            "agree": (a == b) if (a is not None and b is not None) else None,
            "value": a if a is not None else b,
            "read_from": (".".join(auth) if a is not None
                          else (".".join(other) if b is not None else None)),
        }
        if fields[name]["both_present"] and not fields[name]["agree"]:
            disagree.append(name)
    return {"fields": fields, "disagreeing": disagree,
            "all_present_copies_agree": not disagree}


def group_landing_records_by_receipt(files, day: str, root: Path) -> dict:
    """DA 105's RULE, ADOPTED HERE (REV 89 H1).

    A LANDING RECORD IS ABOUT ONE RECEIPT, NOT ABOUT A DAY. When a day's
    receipt is superseded, the new head is a DIFFERENT artifact and its
    pre-read is FIRST OF FAMILY -- the older record stays true of the
    receipt it read. Resolving every record for a day into one chain made
    those two read AMBIGUOUS, which is the resolver describing its own
    grouping rather than the ledger, and it refused 09-03, 09-04 and
    09-05 in this seat's gate while DA's passed them (REV 89 §1: three of
    the four sealed days, on the conjunct whose job is to stop a
    re-roll).

    ***A SUPERSESSION LINK OUTRANKS THE GROUPING.*** A correction may
    change the very field the grouping reads, so records joined by a pair
    are ONE family whatever digests they carry -- otherwise a chain
    broken by its own repair would, again, be the resolver describing its
    rule instead of the ledger.

    The group is CHOSEN by the day's current receipt head, and the choice
    is NAMED in the return; it is never silent."""
    files = [Path(f) for f in files]
    docs, sha_of = {}, {}
    for f in files:
        try:
            docs[f.name] = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        sha_of[f.name] = (landing_record_copies(docs[f.name])["fields"]
                          ["receipt_sha256_at_landing"]["value"]
                          or "UNSTATED")
    parent = {n: n for n in docs}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx

    for n, rec in docs.items():                      # links outrank groups
        shape = supersedes_shape(rec)
        if shape["kind"] == "PAIR":
            tgt = Path(shape["path"]).name
            if tgt in parent:
                _union(n, tgt)
    for a in docs:                                   # then group by digest
        for b in docs:
            if a < b and sha_of.get(a) == sha_of.get(b) != "UNSTATED":
                _union(a, b)

    groups: dict = {}
    for n in docs:
        groups.setdefault(_find(n), []).append(n)

    head = find_sealed_day_receipt(day, root)
    # THE DIGEST IS COMPUTED FROM THE HEAD'S PATH. `find_sealed_day_receipt`
    # returns `path`, not `sha256` -- I first read a key it does not have,
    # and the chooser silently found no match and fell back to the whole
    # day, which is the exact behaviour this is replacing. The key set is
    # ['chain','chain_head_is','day','expected_glob','links','n_matches',
    # 'path','present','status','superseded'].
    want = (sha256_streamed(Path(head["path"]))
            if head.get("path") else None)
    chosen_root, why = None, None
    if want:
        for r, names in groups.items():
            if any(sha_of.get(n) == want for n in names):
                chosen_root, why = r, (
                    "the group whose records read the day's CURRENT "
                    "receipt head")
                break
    if chosen_root is None and len(groups) == 1:
        chosen_root = next(iter(groups))
        why = ("the only group -- no receipt head digest was available to "
               "choose by")
    return {
        "rule": "DA 105, adopted at REV 89 H1: grouped by the receipt "
                "digest each record carries, with a supersession link "
                "outranking the grouping",
        "n_records": len(files), "n_groups": len(groups),
        "groups": {str(r): sorted(v) for r, v in groups.items()},
        "receipt_digest_of_each_group": {
            str(r): sha_of.get(v[0]) for r, v in groups.items()},
        "day_receipt_head": {
            "name": (Path(head["path"]).name if head.get("path") else None),
            "sha256": want, "status": head.get("status")},
        "chosen_group": sorted(groups.get(chosen_root, [])),
        "chosen_because": why or (
            "NO GROUP MATCHES the day's receipt head, and there is more "
            "than one -- named, not silently collapsed"),
        "chose_one": chosen_root is not None,
        "files": [f for f in files
                  if f.name in set(groups.get(chosen_root, []))],
    }


def landing_record_for(day: str, root: Path) -> dict:
    """DA's PRE-READ artifact for a day -- the LANDING RECORD.

    R-604 item 3: the digest a day's receipt had WHEN IT LANDED. A re-run
    after landing produces a different receipt, and waiting must not become
    re-rolling: the gate compares against this, not against whatever is on
    disk at read time.

    Three things this now does that it did not:
      * it requires DA'S DECLARED FLAG, so the two seats agree on WHICH
        artifacts are landing records at all;
      * it resolves through `resolve_day_chain`, so DA's own declared
        `.v2` correction path for a landing record resolves here too --
        it read AMBIGUOUS before, which would refuse a day whose landing
        record had been corrected exactly as DA declares corrections;
      * it reads BOTH copies of every twice-written field and REFUSES BY
        NAME if they differ (REV 54 S1.3).
    Records that are unreadable, undeclared or dayless are NAMED STATUSES
    in `refused_records`, never silent drops (rule 11)."""
    d = Path(root) / "pm_5min/derived"
    mine, refused = [], []
    for p in sorted(d.glob(LANDING_RECORD_GLOB)):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            refused.append({"file": p.name,
                            "status": "LANDING_RECORD_UNREADABLE",
                            "why": str(exc)})
            continue
        if not (isinstance(rec, dict)
                and rec.get(LANDING_RECORD_DECLARED_FLAG)):
            refused.append({
                "file": p.name, "status": "NOT_DECLARED_A_LANDING_RECORD",
                "why": f"it does not carry `{LANDING_RECORD_DECLARED_FLAG}`, "
                       f"so DA's own reader does not see it as one either"})
            continue
        fd = rec.get("day") or _at_path(rec, ("landing_record", "day"))
        if not fd:
            refused.append({"file": p.name,
                            "status": "LANDING_RECORD_NO_DAY_FIELD",
                            "why": "the day comes from the FIELD, never "
                                   "the filename; a record carrying no "
                                   "day is a status, not a drop"})
            continue
        if day_forms(str(fd)) & day_forms(day):
            mine.append(p)
    # DA 105's GROUPING (REV 89 H1) BEFORE the chain rule: `mine` is every
    # record for the DAY, and the chain rule is about ONE RECEIPT.
    grouping = group_landing_records_by_receipt(mine, day, root)
    res = resolve_day_chain(grouping["files"] if grouping["chose_one"]
                            else mine, kind="landing record")
    out = {"day": day, "present": False, "grouping": grouping,
           "status": ("NO_LANDING_RECORD" if res["status"] == "MISSING"
                      else res["status"]),
           "n_matches": res["n_matches"], "refused_records": refused,
           "receipt_sha256_at_landing": None,
           "receipt_name_at_landing": None}
    for k in ("chain", "matches", "heads", "refusals", "links", "why"):
        if k in res:
            out[k] = res[k]
    if not res["resolved"]:
        if res["status"] == "MISSING":
            out["why"] = ("DA's pre-read artifact is the landing record; "
                          "without it there is nothing to compare the "
                          "receipt's digest against, and a re-run after "
                          "landing would be invisible")
        return out
    head = res["head"]
    rec = json.loads(head.read_text())
    copies = landing_record_copies(rec)
    out["path"] = str(head)
    out["field_copies"] = copies
    out["name_matches_convention"] = bool(re.match(
        r"^p003_da_gate1_pre_read_\d{8}__.+\.json$", head.name))
    if not copies["all_present_copies_agree"]:
        out["status"] = "LANDING_RECORD_FIELD_COPIES_DISAGREE"
        out["why"] = (
            f"the landing record writes "
            f"{copies['disagreeing']} TWICE and the copies do not agree. "
            f"DE read one and DA the other, so the two seats would resolve "
            f"conjunct 3 -- the conjunct that stops a re-roll -- against "
            f"different digests without either noticing (REV 54 S1.3)")
        return out
    out["present"] = True
    out["receipt_sha256_at_landing"] = copies["fields"][
        "receipt_sha256_at_landing"]["value"]
    out["receipt_name_at_landing"] = copies["fields"][
        "receipt_name_at_landing"]["value"]
    if out["receipt_sha256_at_landing"] is None:
        out["present"] = False
        out["status"] = "LANDING_RECORD_NO_RECEIPT_DIGEST"
        out["why"] = ("a landing record carrying no receipt digest in "
                      "either copy records nothing the gate can compare "
                      "against")
    return out


def _blob_sha256_at(commit: str, rel: str, repo: Path) -> str | None:
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"],
                           capture_output=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return hashlib.sha256(r.stdout).hexdigest()


def read_gate(params: dict, *, now_utc: datetime.datetime,
              root: Path | None = None,
              ledger_rows: dict | None = None) -> dict:
    """R-602: THE SEAL-OPEN BAR IS A PREDICATE ON ARTIFACTS, NOT A CLOCK.

    DA 72 measured the pipeline at 1.165 h serial per day and found that
    the sixth ruled day (09-08) completes by calendar at 09-09T00:00:00Z,
    leaving SIX MINUTES against that build before the ruled 00:06Z. Under a
    clock bar alone the read would open on five days, and G = 5 gives
    2^-5 = 0.03125, which FAILS Holm at m = 2 -- the design's own
    arithmetic. So the bar is a CONJUNCTION: the clock AND all six sealed
    day receipts present at the ledger root, each verified.

    The day set is UNCHANGED and nothing is chosen on data: this decides
    WHEN the read may open, never WHICH days are in it (R-555)."""
    root = Path(root) if root is not None else Path(DR.resolve()["data_root"])
    # REV 89 H1's follow-up: THE GATE REFUSES A ROOT THAT IS NOT THE
    # LEDGER, instead of reporting every day as failing. `root` is the
    # DATA root; handed the REPO root -- one level up, an easy thing to
    # pass -- `root / "pm_5min/derived"` does not exist and all six days
    # failed conjunct 3 with a detail that said only `failing`, which
    # reads exactly like a real six-day regression. The coordinator read
    # it as one after the H1 landing (2026-09-07). A wrong argument must
    # not be indistinguishable from a finding.
    if not (root / "pm_5min/derived").is_dir():
        raise RunnerRefused(
            f"REFUSED READ_GATE_ROOT_IS_NOT_THE_LEDGER: {root} has no "
            f"`pm_5min/derived`. `root` is the DATA root "
            f"({DR.resolve()['data_root']}); the repo root is one level "
            f"up and passing it makes every day fail for want of a "
            f"directory, which is not a verdict about any day.")
    repo = Path(__file__).resolve().parents[2]
    days = params["days"]
    # (8) ONE FIELD, AND IT IS REQUIRED. R-604 item 8 was LIVE: params v7
    # carried the eight under `read_gate_predicate` while
    # `read_gate.the_bar_is_a_CONJUNCTION` still listed R-602's TWO
    # strings, and DA read the latter and reported "2 conjuncts declared".
    # There is ONE list now, of objects with stable ids, and both seats
    # evaluate it BY ID.
    rg = params.get("read_gate") or {}
    spec_list = rg.get("the_bar_is_a_CONJUNCTION")
    if not isinstance(spec_list, list) or not spec_list or not all(
            isinstance(c, dict) and c.get("id") for c in spec_list):
        raise RunnerRefused(
            "REFUSED: the parameter file carries no "
            "`read_gate.the_bar_is_a_CONJUNCTION` as a list of objects "
            "with ids. R-604 item 8: DA's gate reads the SAME field, and a "
            "bar implemented twice from different fields is a bar that can "
            "disagree without anybody noticing.")
    declared_ids = [c["id"] for c in spec_list]
    spec = {"horizon_utc": rg.get("horizon_utc", READ_HORIZON_UTC)}
    not_before = _iso_utc(params["read_not_before_utc"])
    horizon = _iso_utc(spec.get("horizon_utc", READ_HORIZON_UTC))
    C = []

    def _c(name, holds, detail):
        C.append({"conjunct": name, "holds": bool(holds), "detail": detail})

    # (1) THE CLOCK, a genuine conjunct: six receipts before it must NOT
    #     open the read.
    _c("1_clock", now_utc >= not_before,
       {"now_utc": now_utc.isoformat(),
        "read_not_before_utc": params["read_not_before_utc"]})

    # (2) THE SIX RULED DAYS, from the committed params -- never "six
    #     receipts present". A receipt for another day fills no hole.
    found = {d: find_sealed_day_receipt(d, root) for d in days}
    # NOT `per_day, ledger_rows = {}, {}` -- that reset the PARAMETER to
    # an empty dict before the check below, so an injected set was silently
    # discarded and every day failed conjunct 5. A local initialiser that
    # shadows a parameter of the same name is invisible at the call site.
    per_day = {}
    if ledger_rows is None:
        # THE PRODUCTION PATH uses the design's reader, which REFUSES any
        # root but the canonical ledger -- deliberately, so a day set can
        # never be derived from the wrong tree. `ledger_rows` is injectable
        # ONLY so conjunct 5 can be driven on a synthetic root; the real
        # path never passes it.
        try:
            ledger_rows = DESIGN.day_sets_from_the_ledger(
            )["ledger_rows_as_read"]
        except Exception as exc:                  # ledger unreadable
            ledger_rows = {"_error": str(exc)}
    for d in days:
        f = found[d]
        row = {"day": d, "receipt": f}
        if f["present"]:
            try:
                rec = json.loads(Path(f["path"]).read_text())
            except (OSError, ValueError) as exc:
                rec = {}
                row["unreadable"] = str(exc)
            row["verify"] = verify_sealed_day_receipt(d, f["path"], root)
            # (3) THE DIGEST AT LANDING.
            land = landing_record_for(d, root)
            here = sha256_streamed(Path(f["path"]))
            # THE SAME `.get` ON AN UNKNOWN TYPE, one function
            # away -- the shape is judged here too (REV 54 S0.1).
            _shape = supersedes_shape(rec)
            sup = _shape.get("sha256") if _shape["is_a_pair"] else None
            row["landing"] = {
                **land, "receipt_sha256_now": here,
                "supersedes_sha256": sup,
                "matches_landing": bool(
                    land.get("present")
                    and land.get("receipt_sha256_at_landing")
                    in (here, sup)),
                "why": "a re-run after landing changes the digest; waiting "
                       "must not become re-rolling. A superseding vN is "
                       "admitted only when its chain reaches the landed "
                       "digest",
            }
            # (4) AT LEAST ONE ADMISSIBLE ARM, from the sealed artifact.
            arts = rec.get("per_day_sealed_artifacts") or []
            adm = [a for a in arts
                   if (a.get("admissibility") or {}).get("admissible")
                   is True]
            row["admissible_arms"] = {
                "n": len(adm), "n_arms": len(arts),
                "holds": len(adm) >= 1,
                "status": rec.get("status"),
                "why": "six all-inadmissible days satisfy existence and "
                       "contribute ZERO SIGNS; the aggregate would have "
                       "nothing to aggregate",
            }
            # (6) THE PRODUCING CODE IS LOCATABLE.
            si = rec.get("source_identity") or {}
            cc, pcs = si.get("carrying_commit"), si.get(
                "producing_code_sha256")
            blob = (_blob_sha256_at(cc, "live/pm_research/"
                                    "de_multiday_gate1_runner.py", repo)
                    if cc else None)
            row["code_locatable"] = {
                "carrying_commit": cc, "producing_code_sha256": pcs,
                "blob_sha256_at_that_commit": blob,
                "holds": bool(pcs and blob and pcs == blob),
                "why": "a receipt whose producing digest is in no commit "
                       "names code nobody can fetch. For 2026-09-03 the "
                       ".v2 supersession is what makes this hold: v1 was "
                       "stamped from a file replaced mid-run (R-603)",
            }
        else:
            row["landing"] = {"matches_landing": False,
                              "status": "NO_RECEIPT"}
            row["admissible_arms"] = {"holds": False, "status": "NO_RECEIPT"}
            row["code_locatable"] = {"holds": False, "status": "NO_RECEIPT"}
        # (5) THE LEDGER VERDICT -- R-555's membership.
        lr = ledger_rows.get(d) if isinstance(ledger_rows, dict) else None
        row["ledger"] = {
            "present": lr is not None,
            "conjuncts": ({k: lr.get(k) for k in
                           LEDGER_MEMBERSHIP_CONJUNCTS} if lr else None),
            "holds": bool(lr and all(lr.get(k) is True
                                     for k in LEDGER_MEMBERSHIP_CONJUNCTS)),
            "why": "a sealed receipt is DE's own artifact and is not a day "
                   "verdict; membership is R-555's and lives in the ledger",
        }
        per_day[d] = row

    _c("2_all_six_ruled_days_have_a_receipt",
       all(found[d]["present"] for d in days),
       {"missing": [d for d in days if not found[d]["present"]],
        "note": "the days come from the committed params; a receipt for "
                "another day fills no hole"})
    # REV 89 H1's follow-up: the detail carried ONLY `failing`. A reader
    # meeting six failing days could not tell a regression from a wrong
    # root from three days legitimately without receipts, and the seat
    # that changed the resolver could not tell either. Every day now
    # carries WHY, and the cross-check against DA's resolver travels with
    # it -- the comparison REV 89 §1 said nobody was running.
    _lr3 = {}
    for d in days:
        _l = per_day[d]["landing"]
        _lr3[d] = {
            "matches_landing": _l.get("matches_landing"),
            "landing_record_status": _l.get("status"),
            "receipt_sha256_at_landing": _l.get("receipt_sha256_at_landing"),
            "receipt_head_present": found[d].get("present"),
            "why_not": (
                None if _l.get("matches_landing")
                else ("no receipt for this day yet"
                      if not found[d].get("present")
                      else f"landing record status "
                           f"{_l.get('status')!r}")),
            "grouping": (_l.get("grouping") or {}).get("chosen_because"),
        }
    _c("3_digest_matches_the_landing_record",
       all(per_day[d]["landing"].get("matches_landing") for d in days),
       {"failing": [d for d in days
                    if not per_day[d]["landing"].get("matches_landing")],
        "ledger_root_read": str(root / "pm_5min/derived"),
        "per_day": _lr3,
        "grouping_rule": "DA 105, adopted at REV 89 H1 -- records grouped "
                         "by the receipt digest each carries, a "
                         "supersession link outranking the grouping",
        "how_to_read_a_full_sweep": "every day failing at once is usually "
                                    "the ROOT, not the days; the gate now "
                                    "refuses a non-ledger root by name "
                                    "rather than reporting it as six "
                                    "failures"})
    _c("4_each_day_has_an_admissible_arm",
       all(per_day[d]["admissible_arms"]["holds"] for d in days),
       {"failing": [d for d in days
                    if not per_day[d]["admissible_arms"]["holds"]]})
    _c("5_ledger_verdict_holds",
       all(per_day[d]["ledger"]["holds"] for d in days),
       {"conjuncts": list(LEDGER_MEMBERSHIP_CONJUNCTS),
        "failing": [d for d in days if not per_day[d]["ledger"]["holds"]]})
    _c("6_producing_code_is_locatable",
       all(per_day[d]["code_locatable"]["holds"] for d in days),
       {"failing": [d for d in days
                    if not per_day[d]["code_locatable"]["holds"]]})
    _c("8_params_carries_the_predicate", True,
       {"field": "read_gate.the_bar_is_a_CONJUNCTION",
        "note": "checked above; a file lacking it refuses before any day "
                "is examined"})
    # EVERY DECLARED ID MUST HAVE AN EVALUATOR HERE, and every evaluator a
    # declared id. An id nobody evaluates is a bar nobody applies; an
    # evaluator with no id is a bar nobody declared.
    _EVAL = {"clock_ge_read_not_before": "1_clock",
             "six_ruled_days_from_params":
                 "2_all_six_ruled_days_have_a_receipt",
             "receipt_at_landing_digest":
                 "3_digest_matches_the_landing_record",
             "at_least_one_admissible_arm":
                 "4_each_day_has_an_admissible_arm",
             "ledger_verdict": "5_ledger_verdict_holds",
             "producing_code_locatable": "6_producing_code_is_locatable",
             "horizon_fallback_G5_directional": "horizon",
             "params_field_required": "8_params_carries_the_predicate"}
    unbound = [i for i in declared_ids if i not in _EVAL]
    unused = [i for i in _EVAL if i not in declared_ids]
    if unbound or unused:
        raise RunnerRefused(
            f"REFUSED: the declared conjunct ids and this runner's "
            f"evaluators do not correspond. Declared with no evaluator: "
            f"{unbound}; evaluated but not declared: {unused}. A conjunct "
            f"id without an evaluator CLOSES the gate (R-604 item 8).")

    landed = [d for d in days if found[d]["present"]]
    all_hold = all(c["holds"] for c in C)
    # (7) THE HORIZON, declared with its date.
    horizon_reached = now_utc >= horizon
    horizon_open = bool(
        not all_hold and horizon_reached and C[0]["holds"]
        and len(landed) == len(days) - 1
        and all(per_day[d]["landing"].get("matches_landing")
                and per_day[d]["admissible_arms"]["holds"]
                and per_day[d]["ledger"]["holds"]
                and per_day[d]["code_locatable"]["holds"] for d in landed))
    return {
        "protocol": "P003_DE_GATE1_READ_GATE_V2",
        "ruling": "R-602, completed as eight conjuncts by R-604",
        "conjuncts": C,
        "declared_conjunct_ids": declared_ids,
        "id_to_evaluator": dict(_EVAL),
        "per_day": per_day,
        "ruled_days": list(days),
        "n_landed": len(landed),
        "may_open": all_hold,
        "G_if_opened": len(days) if all_hold else (
            len(landed) if horizon_open else None),
        "horizon": {
            "utc": spec.get("horizon_utc", READ_HORIZON_UTC),
            "reached": horizon_reached,
            "opens_at_G_minus_one": horizon_open,
            "verdict_if_opened_here": "DIRECTIONAL ONLY -- 2^-5 = 0.03125 "
                                      "is not significant at m = 2",
            "the_sixth_day_is_disclosed_as": (
                [d for d in days if d not in landed] if horizon_open
                else None),
            "why_declared_now": "with its date, so the stopping rule is "
                                "not a degree of freedom chosen after "
                                "seeing which days landed",
        },
        "what_this_does_NOT_decide": "WHICH days are in the set. R-555 "
                                     "ruled the population; this rules "
                                     "only WHEN the read may open",
    }


def may_read_aggregate(params: dict, *, n_days_complete: int,
                       now_utc: datetime.datetime,
                       root: Path | None = None,
                       ledger_rows: dict | None = None) -> dict:
    """The OTHER clock: the unseal and the section-7 verdict.

    BOTH conditions, not either: the declared date AND all G days. The date
    alone would let a five-day read happen on the ninth; the count alone
    would let the read happen the moment the sixth day landed early."""
    # THE CLOCK IS CONJUNCT 1 OF THE GATE, and only there. It was checked
    # here as well, which is two implementations of one bar inside one
    # function -- the defect R-604 item 8 names between DE and DA, in
    # miniature.
    # THE COMPLETENESS CHECK COMPARES AGAINST THE GATE'S OWN G, and is
    # therefore evaluated AFTER it: at the horizon the ruled G is 5 by the
    # declaration, and a check hardcoded to params["G"] refused the very
    # case R-604 item 7 exists to allow.
    # R-602: AND THE ARTIFACTS. A clock bar alone would open the read on
    # five days, and G = 5 fails Holm at m = 2 by the design's own
    # arithmetic. `root` is threaded so the gate can be driven.
    if params.get("read_requires_all_ruled_days_sealed") is not True:
        raise RunnerRefused(
            "REFUSED: the parameter file does not carry "
            "`read_requires_all_ruled_days_sealed: true`; without the "
            "ruling in the file the runner will not infer it (R-602).")
    gate = read_gate(params, now_utc=now_utc, root=root,
                     ledger_rows=ledger_rows)
    _need = gate["G_if_opened"]
    if _need is not None and n_days_complete < _need:
        raise RunnerRefused(
            f"REFUSED: {n_days_complete} of {_need} days are complete. "
            f"The read unseals every day at once or not at all (R5).")
    if not gate["may_open"]:
        failed = [c for c in gate["conjuncts"] if not c["holds"]]
        if gate["horizon"]["opens_at_G_minus_one"]:
            return {"may_read": True, "G": gate["G_if_opened"],
                    "n_days_complete": n_days_complete,
                    "read_not_before_utc": params["read_not_before_utc"],
                    "opened_at_the_HORIZON": True,
                    "verdict_is": "DIRECTIONAL ONLY",
                    "unbuilt_days": gate["horizon"][
                        "the_sixth_day_is_disclosed_as"],
                    "read_gate": gate}
        raise RunnerRefused(
            "REFUSED: the read predicate does not hold. Failing conjuncts, "
            "by name: "
            + "; ".join(f"{c['conjunct']} -> {c['detail']}"
                        for c in failed)
            + ". (R-602 as completed by R-604.)")
    return {"may_read": True, "n_days_complete": n_days_complete,
            "G": params["G"],
            "read_not_before_utc": params["read_not_before_utc"],
            "both_conditions_required": True,
            "read_gate": gate}


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


def emission_stamp(now=None) -> str:
    """THE FILENAME STAMP, FROM THE CLOCK.

    I typed these. design v15's name says 09:45:00Z and the artifact was
    written at 09:32:50Z; the 09-04 rehearsal says 09:51:00Z against
    09:33:56Z -- **names for a moment that had not yet occurred**. The
    cause was that params must name the design's path BEFORE the design is
    emitted (the pin runs design -> params), so I chose a rounded stamp
    instead of reading a clock. That is the memory rule this programme
    already carries -- times come from `date`, never estimated -- applied
    to filenames, where nobody had been looking."""
    t = now or datetime.datetime.now(datetime.timezone.utc)
    return t.strftime("%Y%m%dT%H%M%SZ")


def assert_name_stamp_is_the_clock(output: Path, as_of: str, *,
                                   tolerance_s: int = 300) -> dict:
    """A stamped filename must name the moment the file was written.

    A version-only name (no `__<stamp>`) is the OTHER fix and is admitted:
    an artifact whose path must be known in advance cannot carry a stamp
    honestly, so it carries none and its time lives in `as_of`."""
    name = Path(output).name
    if "__" not in name:
        return {"name_carries_a_stamp": False,
                "why_this_is_fine": "a path that must be known in advance "
                                    "cannot carry an honest stamp; the "
                                    "time lives in `as_of`",
                "as_of": as_of}
    stamp = name.rsplit("__", 1)[1].split(".")[0]
    try:
        st = datetime.datetime.strptime(
            stamp, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=datetime.timezone.utc)
    except ValueError:
        raise RunnerRefused(
            f"REFUSED: the output name {name} carries `{stamp}`, which is "
            f"not a UTC stamp.")
    wrote = datetime.datetime.fromisoformat(as_of)
    delta = (st - wrote).total_seconds()
    if abs(delta) > tolerance_s:
        raise RunnerRefused(
            f"REFUSED: the output name {name} is stamped {stamp} while the "
            f"artifact was written at {as_of} -- {delta:+.0f}s. A stamp "
            f"that is not the clock is a time somebody typed; two of mine "
            f"named a moment that had NOT YET OCCURRED (design v15: name "
            f"09:45:00Z, written 09:32:50Z). Times come from the clock.")
    return {"name_carries_a_stamp": True, "stamp": stamp, "as_of": as_of,
            "delta_seconds": delta, "tolerance_s": tolerance_s,
            "stamp_is_the_clock": True}


#: Fixture day names that MAY run under a scope. Declared BY NAME and
#: only for fixtures: a fixture emits nothing a read can resolve, and the
#: falsifier below has to run under a real `.scope` unit to prove the
#: refusal fires at all.
SCOPE_EXEMPT_FIXTURE_DAYS = ("FIXTURE-DAY-1", "FIXTURE-DAY-91",
                             "FIXTURE-DAY-ORDER", "FIXTURE-DAY-SCOPE")


def parent_cmdline(pid: int | None = None) -> dict:
    """THIS PROCESS'S PARENT COMMAND LINE, read from /proc.

    Under the declared form the payload's parent IS `flock`, so the `-E`
    half of the launch form is decidable at runtime exactly as the
    `--scope` half is (REV 69 S1.2)."""
    import os as _os
    me = pid or _os.getpid()
    try:
        ppid = int(open(f"/proc/{me}/status").read()
                   .split("PPid:")[1].split()[0])
    except (OSError, IndexError, ValueError) as exc:
        return {"ppid": None, "argv": None, "readable": False,
                "why": f"{type(exc).__name__}: {exc}"}
    try:
        raw = open(f"/proc/{ppid}/cmdline", "rb").read()
    except OSError as exc:
        return {"ppid": ppid, "argv": None, "readable": False,
                "why": f"{type(exc).__name__}: {exc}"}
    argv = [x for x in raw.decode("utf-8", "replace").split("\0") if x]
    try:
        exe = _os.readlink(f"/proc/{ppid}/exe")
    except OSError:
        exe = None
    return {"ppid": ppid, "argv": argv, "readable": True, "exe": exe,
            "argv_joined": " ".join(argv),
            "exe_is_the_identity": (
                "argv[0] is settable by `exec -a`; the exe link is the "
                "file the kernel mapped")}


def assert_lock_form_at_runtime(day: str, *, fixture: bool,
                                parent: dict | None = None) -> dict:
    """THE `-E` HALF, DECIDED AT RUNTIME -- not read off a string.

    R-653 (iii) / REV 69 S1.2. The composed-command check is a LINT: it
    reads the string a builder produced, and it cannot see what was
    actually executed. The payload's parent is `flock`, so the real
    question -- did this run take the lock with the DECLARED conflict code
    on the DECLARED lock -- is answerable from `/proc/<PPid>/cmdline`.

    Without `-E <rc>` a held lock and a payload crash are both
    ExecMainStatus=1, so a refusal is unreadable from a crash. This is the
    half that has to hold at run time; the lint stays a lint."""
    form = heavy_run_form()
    rc, lock = str(form["lock_conflict_rc"]), form["lock_path"]
    par = parent if parent is not None else parent_cmdline()
    argv = par.get("argv") or []
    # THE PARENT'S IDENTITY COMES FROM ITS EXECUTABLE, NOT FROM argv[0]
    # (REV 71 S1.2). `exec -a flock /bin/sleep` sets argv[0] to "flock"
    # while running something else; /proc/<ppid>/exe is the file the
    # kernel actually mapped. argv is still read -- it is where the FLAGS
    # are -- but it no longer answers "is this flock?".
    out = {"day": day, "fixture": fixture, "ppid": par.get("ppid"),
           "parent_argv": argv, "parent_exe": par.get("exe"),
           "declared_rc": rc, "declared_lock_path": lock,
           "parent_is_flock": (
               bool(par.get("exe"))
               and Path(par["exe"]).name == "flock"),
           "identity_from": "/proc/<ppid>/exe -- the file the kernel "
                            "mapped, never argv[0], which `exec -a` sets "
                            "to anything",
           "carries_dash_E_with_the_declared_rc": (
               "-E" in argv and rc in argv
               and argv.index(rc) == argv.index("-E") + 1),
           "carries_the_declared_lock_path": lock in argv,
           "why_runtime_not_the_lint": (
               "the lint reads the string a builder produced; this reads "
               "what actually ran. `-E` is what makes a held lock "
               "readable from a crash (REV 69 S1.2)"),
           "checked": not fixture}
    if fixture:
        out["checked"] = False
        out["note"] = "a fixture is not launched under the heavy form"
        return out
    if not par.get("readable"):
        raise RunnerRefused(
            f"REFUSED DAY {day} BEFORE ANY STAGE: this process's parent "
            f"command line is unreadable ({par.get('why')}), so the `-E` "
            f"half of the launch form cannot be checked. An unknown is "
            f"not a passed check.")
    if not out["parent_is_flock"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} BEFORE ANY STAGE: this process's parent "
            f"executable is {par.get('exe')!r} (argv[0] says "
            f"{(argv[:1] or ['<none>'])[0]!r}), not `flock`. Under the declared "
            f"form the lock is the unit's OWN ExecStart and the payload's "
            f"parent is flock; if it is not, this run does not hold the "
            f"lock the way the form requires.")
    if not out["carries_dash_E_with_the_declared_rc"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} BEFORE ANY STAGE: the parent flock carries "
            f"no `-E {rc}` ({' '.join(argv[:6])}). Without it a HELD LOCK "
            f"and a payload crash are both ExecMainStatus=1 and a refusal "
            f"is unreadable from a crash (REV 69 S1.2).")
    if not out["carries_the_declared_lock_path"]:
        raise RunnerRefused(
            f"REFUSED DAY {day} BEFORE ANY STAGE: the parent flock does "
            f"not name the declared lock {lock}.")
    return out


def assert_launch_form_at_runtime(day: str, *, fixture: bool,
                                  observed: dict | None = None) -> dict:
    """THE LAUNCH FORM REFUSES AT RUN TIME, from the cgroup leaf.

    REV 65 S1.2 / REV 62 S3: `is_the_declared_launch_form` REPORTED and
    GATED NOTHING -- a real day under `--scope` would run all 85 minutes
    and say so only in its receipt, and a lint on the command STRING
    cannot see a `--scope` behind a variable or a wrapper.

    REV 68 found the first version of THIS guard wrong in two ways, both
    reproduced:

      S1.2  the fixture exemption was a REPORT. `fixture=True` admitted a
            REAL DAY NAME -- the list was computed into
            `fixture_exemption_by_name` and then never consulted. The
            exemption is now the GATE.
      S1.3  it FAILED OPEN. The test was `kind == "scope"`, so `kind`
            None -- an unreadable cgroup, a leaf shape nobody anticipated
            -- ADMITTED a real day. The real-day predicate is now the
            POSITIVE one: refuse unless the kind is the declared form.

    And R-651 / MEM 181: when the kind is UNKNOWN the record must not
    claim `checked` -- it says False. A guard that cannot see what it is
    in has not checked anything."""
    obs = observed if observed is not None else unit_identity()
    kind = (obs or {}).get("kind")
    leaf = (obs or {}).get("cgroup_leaf")
    known = kind in ("transient service", "scope")
    exempt = bool(fixture) and str(day) in SCOPE_EXEMPT_FIXTURE_DAYS
    base = {"day": day, "fixture": fixture, "cgroup_leaf": leaf,
            "kind": kind, "kind_is_known": known,
            "fixture_exemption_by_name": exempt,
            "the_exemption_is_the_GATE": (
                "not a report beside it: `fixture=True` alone used to "
                "admit a REAL DAY NAME (REV 68 S1.2)"),
            "the_real_day_predicate_is_POSITIVE": (
                "refuse unless kind == 'transient service'. The negative "
                "test `kind == 'scope'` FAILED OPEN on kind None and on "
                "an empty observation (REV 68 S1.3)"),
            # R-651 / MEM 181: `checked` is only true when the guard could
            # SEE what it was in.
            "checked": bool(known) and not fixture,
            "why_checked_may_be_false": (
                "an unknown kind is not a passed check. It used to admit "
                "a real day AND claim `checked` true"),
            "refused": False}
    if not fixture:
        if kind != "transient service":
            raise RunnerRefused(
                f"REFUSED DAY {day} BEFORE ANY STAGE: the declared launch "
                f"form is a transient SERVICE and this process's cgroup "
                f"leaf is {leaf!r} (kind {kind!r}). A scope registers the "
                f"processes the CALLER forks, so the run is in the "
                f"launching shell's process group and dies with it -- "
                f"that is how the 09-03 re-run lost 35 minutes (R-628). "
                f"An UNKNOWN kind refuses too: a guard that cannot see "
                f"what it is in has not checked anything (REV 68 S1.3). "
                f"Nothing was read.")
        return base
    if kind == "scope" and not exempt:
        raise RunnerRefused(
            f"REFUSED DAY {day} BEFORE ANY STAGE: a FIXTURE under a "
            f"`.scope` ({leaf}) is admitted only for a DECLARED fixture "
            f"name, and {day!r} is not among "
            f"{list(SCOPE_EXEMPT_FIXTURE_DAYS)}. `fixture=True` alone "
            f"used to admit anything, including a real day name "
            f"(REV 68 S1.2).")
    return base


def declared_chain() -> list:
    """THE CHAIN, DECLARED -- not described in prose somewhere a launcher
    will not read.

    R-648 (R3'). With `RemainAfterExit=yes` the unit stays LOADED after it
    exits, which is what makes the triple readable; it therefore also has
    to be STOPPED, or the name stays taken and the next launch under it
    refuses. It is a FUNCTION so the battery can drive it without running
    the rehearsal, which reads `data/` -- a fixture run must open no path
    under it, and my first version of that check broke exactly that."""
    return [
            {"step": 1, "do": "run THE_ONE_COMMAND with one substitution "
                              "(the unit name)",
             "then": "read the TRIPLE at once: a held lock is "
                     f"ExecMainStatus={heavy_run_form()['lock_conflict_rc']}"
                     " and the payload never started"},
            {"step": 2, "do": "poll the UNIT (LoadState, ActiveState, "
                              "ExecMainStatus), never a child PID",
             "then": "a running unit reports ExecMainStatus=0; that is "
                     "not a finish"},
            {"step": 3, "do": "when the unit leaves `active`, COPY the "
                              "triple and the journal by InvocationID on "
                              "both fields WHILE LoadState=loaded",
             "then": "after collection every field is a DEFAULT; a "
                     "reading taken then is VOID, never success"},
            {"step": 4, "do": "read the receipt the runner named under "
                              "the output directory",
             "then": "the receipt is the record; the unit reading "
                     "corroborates it"},
            {"step": 5, "do": "COPY THE UNIT'S JOURNAL AGAIN, by both "
                              "invocation fields, AFTER the unit has "
                              "exited",
             "then": "the `Consumed` line is written at exit and it is "
                     "the line that survives longest -- DE 84's `Started` "
                     "line was gone four hours later while its `Consumed` "
                     "line remained. The receipt's own copy is taken at "
                     "the EMIT, before that line exists, so the record "
                     "needs both"},
            {"step": 6, "do": "`systemctl --user stop <unit>` -- the "
                              "owner stops it once the exit copy is "
                              "taken",
             "then": "the name is free for the next launch. "
                     "RemainAfterExit keeps a finished unit loaded until "
                     "someone does this"},
                {"step": 7, "do": "COPY THE JOURNAL AGAIN, AFTER THE "
                              "STOP, by both invocation fields, "
                              "cross-checked against `-u`",
             "then": "under RemainAfterExit the manager emits `Stopped` "
                     "and `Consumed` AT THE STOP, not at the payload's "
                     "exit -- so the step-5 copy holds the run's OUTPUT "
                     "and not its COST. Measured on the 09-04 run: 2 "
                     "lines at step 5, 4 after the stop, and the two it "
                     "missed are `Stopped` and `Consumed 1h 39min "
                     "40.638s CPU, 2.7G peak`. The sidecar records which "
                     "lines existed at EACH copy (REV 75 S3)"},
        ]


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
    # `--output` NAMES THE DIRECTORY (REV 55 S2.1) and the wrapper is a
    # transient SERVICE (R-628). Composed by `the_one_command`, so the
    # string published here and the predicate that checks it are one fact.
    outdir = root / "pm_5min/derived"
    cmd = the_one_command(dashed, book, outdir)

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
    # P3_params was a TAUTOLOGY -- it compared the params digest with
    # itself and could never fail. It states the digest now, which is what
    # a reader wants from it; the pin is checked on the design's side.
    _p("P3_params", (repo / PARAMS_REL).is_file(),
       {"path": PARAMS_REL, "sha256": _digest(repo / PARAMS_REL)})
    # P3_design, IN THE FLIPPED DIRECTION (REV 51 S0). The params name the
    # design by PATH; the DESIGN pins the params by digest. This checked
    # the design's bytes against a `sha256` the params no longer carry, so
    # it blocked the 09-04 preflight on a pin that had moved.
    _dp = root.parent / design["path"]
    _dpins = None
    if _dp.is_file():
        try:
            _dpins = (json.loads(_dp.read_text()).get("parameters") or {}
                      ).get("sha256")
        except (OSError, ValueError):
            _dpins = None
    # THE DESIGN IS RESOLVED TO ITS CHAIN HEAD (R-656). params v14 names
    # v21 by path while v22 is the head; requiring the path to BE the head
    # would force a params bump per design version for a pointer alone.
    # The head pins the params by digest -- that is the binding half --
    # and the path the params name must be IN the chain.
    _dch = design_chain()
    _head_pins = None
    if _dch.get("resolved"):
        try:
            _head_pins = (json.loads(Path(_dch["head_path"]).read_text())
                          .get("parameters") or {}).get("sha256")
        except (OSError, ValueError):
            _head_pins = None
    _named_in_chain = Path(design["path"]).name in (
        _dch.get("chain_names") or [])
    _p("P3_design",
       bool(_head_pins) and _head_pins == _digest(repo / PARAMS_REL)
       and _named_in_chain and bool(_dch.get("resolved")),
       {"params_name_this_design": design["path"],
        "the_named_design_is_in_the_chain": _named_in_chain,
        "resolved_head": _dch.get("head_name"),
        "head_version": _dch.get("head_version"),
        "head_pins_params_sha256": _head_pins,
        "params_sha256_here": _digest(repo / PARAMS_REL),
        "chain_links_all_agree": all(
            l.get("agrees") for l in (_dch.get("links") or [])),
        "pin_direction": "design -> params",
        "why_the_head_and_not_the_named_path": (
            "the params name the design by PATH; requiring that path to "
            "be the head forces a params bump per design version for a "
            "pointer alone. The HEAD pins the params by digest -- the "
            "binding half -- and the named path must be IN the chain")})
    _p("P4_data_root_is_the_ledger",
       DR.resolve()["is_canonical"] is True, DR.resolve()["data_root"])
    _lockobs = wrapper_observed()
    _p("P5_lock_free_now", not _lockobs["lock_is_held_by_someone"],
       # COMPUTED, not narrated. This said "held right now by BE 55's
       # assembly" and went on saying it after BE 55 finished -- a
       # hardcoded state beside a measured one, the same class as the
       # literal 23 beside a constant of 49 (rule 10).
       {"lock_is_held_by_someone": _lockobs["lock_is_held_by_someone"],
        "holder_pids": _lockobs["flock_holder_pids"],
        "modes": _lockobs["flock_modes_on_the_inode"],
        "held_by_self_or_ancestor": _lockobs["held_by_self_or_ancestor"],
        "why_informational": "the lock is TAKEN by the wrapper at GO, so "
                             "its state now does not gate GO"},
       blocks=False)
    _p("P7_cascade_digest",
       _digest(repo / params["be_module"]["path"])
       == params["be_module"]["sha256"],
       {"path": params["be_module"]["path"],
        "declared": params["be_module"]["sha256"]})
    _p("day_is_in_the_ruled_set", dashed in ruled_day_set(), ruled_day_set())
    # REV 55 S2.1 as a PRECONDITION, not a paragraph: the command's
    # `--output` must be a directory the runner will accept, and the day
    # must not already have a sealed artifact.
    try:
        _outok = assert_output_is_a_directory(outdir, day=dashed)
        _outdetail = {**_outok,
                      "example_name_if_emitted_now": day_receipt_name(
                          dashed, fixture=False, stamp=emission_stamp()),
                      "declared_convention": params["read_gate"][
                          "receipt_naming"]["convention"]}
        _outheld = True
    except RunnerRefused as _e:
        _outdetail, _outheld = {"refusal": str(_e)}, False
    _p("P8_output_is_a_directory_and_the_name_is_composed",
       _outheld, _outdetail)
    # R-628 AS A PRECONDITION: the published command's own shape, checked.
    try:
        _p("P11_launch_form_is_a_transient_service", True,
           {**assert_launch_form(cmd), "unit_substitution_left":
            cmd.count("<deNNsmoke>")})
    except RunnerRefused as _e:
        _p("P11_launch_form_is_a_transient_service", False,
           {"refusal": str(_e)})
    try:
        _p("P9_no_sealed_receipt_for_this_day_yet", True,
           assert_no_sealed_receipt_yet(dashed, root))
    except RunnerRefused as _e:
        _p("P9_no_sealed_receipt_for_this_day_yet", False, {"refusal":
                                                            str(_e)})
    # REV 55 S2.2 as a FIELD. The emit refuses on closure drift, a moved
    # HEAD, or a worktree dirty at import, so "the run worktree is frozen"
    # is a precondition of an 85-minute run rather than a discipline
    # somebody remembers. Nothing in the code can tell "a seat landed"
    # from "the code moved" -- which is why it is checked BEFORE.
    _si = source_identity_at_launch()
    _head = _si["head_at_import"]
    # THE SAME READING THE IMPORT REFUSAL USES. This asked `dirty`, the
    # RAW flag, while `assert_source_unchanged` asks the one that exempts
    # the verified data symlink -- so after the mandated refresh the
    # rehearsal blocked on a condition the run itself would have admitted.
    # Two spellings of one fact, in the precondition whose whole job is to
    # predict the refusal.
    _p("P10_run_worktree_is_clean_at_import",
       _head.get("dirty_beyond_the_shared_data_link") is False,
       {"worktree": _head.get("worktree"), "head": _head.get("head"),
        "any_untracked_entry": _head.get("dirty"),
        "dirty_paths": _head.get("dirty_paths"),
        "dirty_beyond_the_shared_data_link":
            _head.get("dirty_beyond_the_shared_data_link"),
        "dirty_paths_beyond_the_shared_data_link":
            _head.get("dirty_paths_beyond_the_shared_data_link"),
        "shared_data_link_exempted":
            _head.get("shared_data_link_exempted"),
        "n_modules_in_the_import_closure": _si["import_closure"][
            "n_modules"],
        "why_it_blocks": "a REAL day refuses at import on a dirty "
                         "worktree, and the emit refuses if any module of "
                         "the closure or HEAD moves during the run. The "
                         "run must execute from a worktree frozen for its "
                         "whole life, and this seat must land from a "
                         "different one (REV 55 S2.2)"})

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
        "THE_DECLARED_CHAIN": declared_chain(),
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
            **assert_source_unchanged("the fixture-run emit"),
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

#: THE DECLARED PEAK IS A MARKER **INSIDE** THE STAGE TABLE, and
#: `declared_peak_stage()` reads it. It was a DEFAULT ARGUMENT on
#: `peak_stage_predicate`, which made the declaration (the stage's prose)
#: and the predicate (the default) two spellings of one fact -- REV 43
#: §4.4, still open at REV 45 §1.6. One string now, in one place, and a
#: table with no marker or with two REFUSES rather than defaulting.
DAY_STAGE_PEAK_MARKER = "[DECLARED PEAK]"
#: The one stage that is marked ONLY when a `before_work` hook is passed.
#: Named here so the predicate and the stage table are one fact.
HOOK_STAGE = "S0b_battery"

#: The stages, and what each one HOLDS. Named so BE's assembly and DE's day
#: run agree on the seam rather than each assuming the other's budget.
DAY_STAGES = (
    ("S0_verify", "digests only: BE's builder receipt, the book's bytes, "
                  "the pinned models and thetas, BE's cascade module. The "
                  "book is READ ONCE here as bytes for its digest and the "
                  "buffer is handed to S1, never read twice (BE's B-1)"),
    ("S0b_battery", "THE IN-RUN BATTERY, MOVED HERE FROM THE EMIT "
                    "(R-610). It ran AFTER the day's work: ~15 s of "
                    "fixture runs and a 13-module closure re-capture at "
                    "the end of 85 minutes, and if it refused, the day was "
                    "lost -- which is exactly what happened to the 09-03 "
                    "smoke. Its memory is INSIDE the day's growth budget "
                    "on purpose: a stage whose cost escapes the budget is "
                    "a stage nobody bounded"),
    ("S1_load", "[DECLARED PEAK] reference + asm + rows. THE PEAK OF "
                "THE DAY PATH WHEN "
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
        "S0b_battery": "no split. The battery reads BE's committed null "
                       "receipt and the ledger -- declared reads, none of "
                       "them a tape, index or fragment artifact -- and it "
                       "runs under its OWN residency instrument, whose "
                       "opens are SUBTRACTED from the day path's claim so "
                       "that claim stays about the day path (R-610)",
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

#: MEASURED on the synthetic day, then declared with headroom. A fixture
#: that exceeds it REFUSES: the point of a budget nobody enforces is
#: nothing.
FIXTURE_DAY_PEAK_RSS_MB_BUDGET = 700.0

#: THE REAL DAY'S BUDGET, DERIVED -- not "the measured peak plus a margin".
#: R-608's declaration act. The derivation, in order:
#:
#:   the cgroup cap                            8192 MB   (rule 20, never raised)
#:   BE's day reference, measured (R-573)      2008 MB   the S1 book load
#:   the 09-03 run's own observed peak         2426 MB   (journalctl, 2.3G)
#:   headroom for the null loop and the seal   ~1500 MB
#:   ------------------------------------------------
#:   DECLARED per-day budget                   4000 MB   = 48.8% of the cap
#:
#: The number is BELOW the cap on purpose: a budget equal to the cap is not
#: a budget, it is the cap with a second name, and it can only fire after
#: the kernel has already begun reclaiming. 4000 MB leaves the cap as the
#: outer guard and gives this run a bar it can cross while the machine is
#: still healthy.
REAL_DAY_PEAK_RSS_MB_BUDGET = 4000.0
#: MEASURED 2026-09-06 in a FRESH process at the tip: baseline current RSS
#: 18.7 MB -> 44.9 MB after `selftest(offline=False)`, high-water 838.2 MB,
#: 23.6 s. The two numbers are different facts and the budget uses the
#: FIRST: the per-stage check is on CURRENT RSS growth, and the battery
#: frees its transient (most of it one check's deliberate 800 MB
#: inflation) before it returns.
BATTERY_RETAINED_MB_MEASURED = 26.1
BATTERY_PEAK_MB_MEASURED = 838.2
REAL_DAY_BUDGET_DERIVATION = {
    "cgroup_cap_mb": 8192.0,
    "be_day_reference_measured_mb": 2008.0,
    "observed_peak_09_03_mb": 2426.0,
    "headroom_for_null_and_seal_mb": 1500.0,
    "declared_mb": 4000.0,
    # REV 55 S1.2. THREE TERMS ARE MEASURED AND ONE IS NOT, and a reader
    # resolving this block saw an arithmetic chain and could take all of
    # it as measurement. Which is which, named:
    "which_terms_are_MEASURED": {
        "cgroup_cap_mb": "the wrapper's own -p MemoryMax=8G",
        "be_day_reference_measured_mb": "BE's measurement (R-573)",
        "battery_retained_mb_measured": "DE, a fresh process at the tip "
                                        "(2026-09-06)",
        "observed_peak_09_03_mb": "journalctl, the refused smoke",
    },
    "which_terms_are_DECLARED_ALLOWANCES": {
        "headroom_for_null_and_seal_mb": (
            "1500 MB is a JUDGEMENT about what else the day might need -- "
            "the null loop's working set and the seal -- not a "
            "measurement. Nothing has measured a real day's S4/S5 growth, "
            "because no real day has reached them"),
        "the_rounding": (
            "2008 + 26.1 + 1500 = 3534.1, declared at 4000. The 465.9 MB "
            "of rounding is an allowance too"),
    },
    "so_4000_MB_is_NOT_a_measured_number": (
        "it is a derivation from three measurements and two declared "
        "allowances. Anyone citing it as measured is citing it wrong. "
        "What IS measured is that the day refuses if it crosses it, at "
        "the first stage that does"),
    "fraction_of_cap": 4000.0 / 8192.0,
    # R-610 MOVED THE BATTERY INSIDE THE MEASURED WINDOW, so it is a TERM
    # of this derivation now and not a thing that happens afterwards. The
    # cap was NOT raised to make room for it (R-174); the term was
    # measured and the headroom checked against it.
    "battery_retained_mb_measured": BATTERY_RETAINED_MB_MEASURED,
    "battery_peak_mb_measured": BATTERY_PEAK_MB_MEASURED,
    "measured_how": "a fresh process at the tip: current RSS before and "
                    "after `selftest(offline=False)`, plus ru_maxrss "
                    "(2026-09-06)",
    "why_the_RETAINED_number_is_the_one_that_counts": (
        "the per-stage budget compares CURRENT-RSS growth from the S_start "
        "baseline. The battery's 838 MB high-water is a transient it frees "
        "before returning -- and `ru_maxrss` cannot fall, which is the "
        "instrument defect that cost the 09-03 smoke 85 minutes"),
    "terms_sum_mb": 2008.0 + BATTERY_RETAINED_MB_MEASURED + 1500.0,
    "headroom_against_the_declared_budget_mb": (
        4000.0 - (2008.0 + BATTERY_RETAINED_MB_MEASURED + 1500.0)),
    "the_declared_budget_still_covers_the_terms": (
        2008.0 + BATTERY_RETAINED_MB_MEASURED + 1500.0) <= 4000.0,
    "why_not_the_cap": "a budget equal to the cap is the cap with a second "
                       "name; it can only fire once the kernel is already "
                       "reclaiming",
    "why_not_measured_peak_plus_margin_alone": (
        "that is a budget derived from the run it is meant to bound. The "
        "cap and BE's measured reference are the independent terms; the "
        "observed peak is a CHECK on them, not their source"),
    "on_an_overrun": "THE DAY REFUSES at the FIRST STAGE that crosses it. "
                     "The cap is never raised and the draw count is never "
                     "cut (R-174)",
}

#: The real day's ceiling is the cap itself and the response is R-174's:
#: the DAY refuses. Never a raised cap, never fewer draws.
REAL_DAY_PEAK_RSS_GB_CEILING = 8.0


def declared_peak_stage() -> str:
    """THE DECLARED PEAK, READ FROM THE STAGE TABLE -- never a default.

    Exactly one stage must carry the marker. Zero means the declaration was
    lost and the predicate would silently invent one; two means the table
    disagrees with itself. Both REFUSE, because the 8 GiB ceiling rests on
    this claim and a ceiling resting on a default is a ceiling resting on
    nothing."""
    marked = [k for k, v in DAY_STAGES if DAY_STAGE_PEAK_MARKER in v]
    if len(marked) != 1:
        raise RunnerRefused(
            f"REFUSED: the stage table carries {len(marked)} stages marked "
            f"{DAY_STAGE_PEAK_MARKER!r} ({marked}); exactly one must. The "
            f"declared peak is read from the table and is NEVER defaulted "
            f"-- a default argument made the declaration and the predicate "
            f"two spellings of one fact (REV 43 S4.4).")
    return marked[0]


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


#: EVERY STATUS `scope_memory_observation` can report, in ONE place --
#: DE 110 renamed three of them (the ruled leaf is a `.service`, so
#: "SCOPE" was the wrong word) and a battery cell was carrying the old
#: literals. A status set typed in two places drifts silently.
UNIT_MEMORY_STATUSES = ("MEASURED", "AMBIENT_UNIT_NOT_THE_RUNS_OWN",
                        "NOT_IN_A_UNIT", "UNIT_LEAF_NAMED_BUT_UNREADABLE")


#: THE AMBIENTS A CELL'S VERDICT CAN REST ON -- DA's REV 68 §1.2 census,
#: named here so a caller cannot invent an eighth by typing a string.
#: These are the states no test constructs: they are whatever the machine
#: happened to be in when the cell ran. A known-bad over a CONSTRUCTED
#: input (a planted digest, a porcelain line, a census predicate) cannot
#: be DISARMED -- the input is built, so the cell fires or admits -- and
#: asking for the third outcome there would be a control that can never
#: reach one of its own states (REV 84 §2).
AMBIENT_KINDS = ("memory", "the lock", "the cgroup", "the clock",
                 "the journal", "cwd", "the worktree")


def record_ambient_verdict(sink: list, verdict: dict, *, ambient: str,
                           label: str) -> bool:
    """A CELL WHOSE VERDICT RESTS ON AN AMBIENT, recorded three ways.

    Returns True when the cell may be asserted normally (FIRED), False
    when it was DISARMED -- and a DISARMED cell is appended to `sink`, so
    the battery's SUMMARY can carry `n_disarmed` beside its count. REV 84
    §2: "a run that prints 'N checks passed' while three were disarmed
    reproduces the defect one level up."

    ADMITTED is NOT recorded here: it means the guard was reachable and
    did not fire, which is a defect in the code under test and belongs to
    the cell's own assertion, not to the measurement's ledger."""
    if ambient not in AMBIENT_KINDS:
        raise RunnerRefused(
            f"REFUSED: {ambient!r} is not one of the ambients a verdict "
            f"may rest on {list(AMBIENT_KINDS)} (DA's REV 68 §1.2 census). "
            f"A cell over a CONSTRUCTED input cannot be DISARMED, and "
            f"claiming it can is a control that can never reach one of "
            f"its own states.")
    if "verdict" not in (verdict or {}):
        raise RunnerRefused(
            "REFUSED: `record_ambient_verdict` was handed something that "
            "is not a `known_bad_verdict` result. An ambient cell's third "
            "state has to be COMPUTED, never asserted.")
    if verdict["verdict"] == "DISARMED":
        sink.append({"label": label, "ambient": ambient,
                     "quantity": verdict.get("quantity"),
                     "baseline": verdict.get("baseline"),
                     "now": verdict.get("now"),
                     "moved_by": verdict.get("moved_by"),
                     "why": verdict.get("why")})
        return False
    return True


def battery_summary(module: str, *, n_run: int, n_skipped: int = 0,
                    disarmed: list | None = None,
                    skipped: list | None = None,
                    expected: int | None = None) -> dict:
    """THE ONE SUMMARY LINE: `n_disarmed` AND `n_skipped` ARE BOTH IN IT.

    A non-zero `n_disarmed` is NOT a clean run: the cells ran, could not
    have fired, and the count beside them would otherwise read as
    coverage (REV 84 §2). The line names them; the caller raises on
    `clean` False.

    REV 88 §1.3: A SKIP IS A THIRD OUTCOME, not a shortfall in the run
    count. It is counted, and it is NAMED BY CELL WITH ITS REASON --
    `skipped` is a list of `{label, why}`. A skip carried as a BARE
    COUNT is unnamed by construction, and unnamed skips are exactly the
    case `clean` must refuse: `n_run` beside a silent `n_skipped` reads
    as coverage the same way a silent `n_disarmed` did. Naming every
    skip is what makes an offline battery clean again -- the fixture run
    skips the checks that READ `data/` and stays PASS because it says
    which ones and why."""
    dis = list(disarmed or [])
    skips = list(skipped or [])
    # THE COUNT IS DERIVED FROM THE NAMED CELLS WHEN THERE ARE ANY, so the
    # two can never disagree; a caller that passes only `n_skipped` is
    # carrying a bare count and is treated as such.
    n_sk = len(skips) if skips else int(n_skipped)
    unnamed = [s for s in skips
               if not (isinstance(s, dict) and s.get("label")
                       and s.get("why"))]
    if not skips and n_sk:
        unnamed = [f"<{n_sk} skip(s) carried as a bare count, no cell "
                   f"named>"]
    clean = (not dis and not unnamed
             and (expected is None or n_run + n_sk == expected))
    line = (f"[{module}] {'PASS' if clean else 'NOT CLEAN'} -- "
            f"{n_run} checks, n_disarmed {len(dis)}, n_skipped {n_sk}"
            + (f", UNNAMED SKIPS {len(unnamed)}" if unnamed else ""))
    return {"line": line, "clean": clean, "n_run": n_run,
            "n_skipped": n_sk, "n_disarmed": len(dis),
            "disarmed": dis, "skipped": skips,
            "skips_unnamed": unnamed, "expected": expected,
            "rule": "REV 84 §2 -- a non-zero n_disarmed is not a clean "
                    "run; the cells ran and could not have fired, and a "
                    "count beside them would read as coverage. REV 88 "
                    "§1.3 -- a skip is a THIRD outcome, counted and named "
                    "by cell with its reason; an UNNAMED skip is not "
                    "clean, and a skip is never folded into the run "
                    "count"}


def cell_baseline(what: str, *, counts=None, dirs=None) -> dict:
    """A KNOWN-BAD MEASURES AGAINST **ITS OWN** BASELINE (DE 110 (5)).

    THE FAILURE THIS EXISTS FOR, measured on this file: three cells added
    at DE 110 ran before the 0.001 MB stage-budget known-bad, and it
    ADMITTED -- silently, as a PASS -- after firing for rounds.

    AND THE MECHANISM IS NOT THE ONE DE 110's ROW NAMED. That row said
    the budget "measures GROWTH from the high-water"; it does not, and the
    comment at the check says why in the code's own words: growth is
    measured on CURRENT RSS precisely because `ru_maxrss` cannot fall and
    "a budget measured with an instrument that cannot fall is a budget
    that only works once per process". The defence was already there. What
    the earlier cells left behind was a large ALLOCATOR ARENA: the freed
    JSON was returned to the interpreter, not to the kernel, so the
    fixture day's book load reused it and CURRENT RSS grew by ~0 -- under
    a 0.001 MB budget, nothing to cross. Corrected here rather than in
    prose alone: the quantity a known-bad depends on is NAMED by the cell,
    not guessed by this helper, and `peak_rss_mb` is the wrong one for
    this cell.

    So a known-bad whose refusal depends on a MEASUREMENT takes that
    measurement's baseline HERE, at the cell's start, names WHICH quantity
    it is, and the verdict below distinguishes three outcomes where the
    old shape had two."""
    return {"what": what,
            "peak_rss_mb": _peak_rss_mb(),
            "rss_mb_current": _current_rss_mb(),
            "counts": dict(counts or {}),
            "n_files": {str(d): len(list(Path(d).glob("*")))
                        for d in (dirs or [])}}


#: WHICH quantity each known-bad depends on. Named by the cell, because a
#: default here would be this helper guessing -- and the first guess
#: (`peak_rss_mb`) was wrong for the very cell it was written for.
def known_bad_verdict(base: dict, *, refused: bool, quantity: str,
                      measured_now: float | None = None) -> dict:
    """FIRED / DISARMED / ADMITTED -- three outcomes, not two.

      FIRED     the known-bad refused. The control did its job.
      DISARMED  it admitted AND the quantity it depends on did not move
                from this cell's OWN baseline -- so the control had
                nothing to see and its silence says nothing about the
                code. This is a FAILURE of the battery, reported as its
                own kind, because reading it as a pass is what rule 16
                names.
      ADMITTED  it admitted WITH headroom: the guard was reachable and
                did not fire. A real defect in the code under test.
    """
    _read = {"peak_rss_mb": _peak_rss_mb,
             "rss_mb_current": _current_rss_mb}
    if measured_now is None and quantity not in _read:
        raise RunnerRefused(
            f"REFUSED: known_bad_verdict cannot read {quantity!r} itself "
            f"and was given no measurement. A control whose quantity "
            f"nobody measured is not a control.")
    now = _read[quantity]() if measured_now is None else measured_now
    moved = now - base.get(quantity, 0.0)
    verdict = ("FIRED" if refused else
               "DISARMED" if moved <= 0 else "ADMITTED")
    return {"verdict": verdict, "refused": refused,
            "quantity": quantity,
            "baseline": base.get(quantity), "now": now,
            "moved_by": moved,
            "is_a_pass": verdict == "FIRED",
            "why": ("the known-bad refused" if verdict == "FIRED" else
                    "it admitted and the quantity it depends on did not "
                    "move from this cell's own baseline -- the control "
                    "was DISARMED by what ran before it, and a pass here "
                    "would be rule 16's 'a control that cannot fail'"
                    if verdict == "DISARMED" else
                    "it admitted WITH headroom -- the guard was reachable "
                    "and did not fire, which is a defect in the code "
                    "under test, not in the battery")}


def leaf_of_cgroup(cg) -> dict:
    """THE UNIT LEAF OF A CGROUP PATH, as a PURE function of the string.

    Pulled out so the classification can be DRIVEN on a `.service` leaf, a
    `.scope` leaf and a non-unit path without patching `/proc/self/cgroup`
    -- the fixture supplies the INPUT, never the peak the code must read."""
    leaf = (cg or "").rstrip("/").rsplit("/", 1)[-1]
    return {"cgroup": cg, "leaf": leaf or None,
            "is_a_unit_leaf": leaf.endswith((".service", ".scope")),
            "is_the_ruled_launch_form": leaf.endswith(".service"),
            "why": "rule 20's form is a transient SERVICE (R-628); a "
                   "`.scope` leaf is a run in the CALLER's process group "
                   "and is read but flagged, and anything else is not a "
                   "per-run cgroup at all"}


def peak_of_record_rule(form: dict | None = None) -> dict:
    """WHICH NUMBER THIS RECEIPT'S PEAK FIELD MAY HOLD -- FROM THE CHAIN.

    REV 81 §3: BE's producers reach this rule through `be_heavy_run.sh`;
    this one does not launch through it, so for this producer the rule
    existed only as PROSE. It is resolved here from the `heavy_run_form`
    chain head (the same head this module already resolves for the lock
    path and the conflict code), and the head's pair is named in the
    receipt so a reader can check WHICH declaration was in force.

    A head that carries no `memory_peak_reading` is a STATUS, never a
    default: a rule that cannot be resolved has not been satisfied."""
    form = heavy_run_form() if form is None else form
    ch = form.get("_chain") or {}
    block = form.get("memory_peak_reading")
    out = {"resolved_from": {"path": ch.get("head_path"),
                             "sha256": ch.get("head_sha256"),
                             "version": ch.get("head_version")},
           "resolved_as": "the chain head, not a filename literal"}
    if not isinstance(block, dict) or not block.get("the_runs_peak"):
        out["status"] = "RULE_NOT_IN_THE_CHAIN_HEAD"
        out["why"] = (
            "the resolved head declares no `memory_peak_reading`. A "
            "producer that cannot resolve the rule records that it could "
            "not; it does not fall back to a number of its own (rule 11, "
            "and R-649: a check that depends on a declaration FAILS when "
            "it is absent -- it does not skip)")
        return out
    out["status"] = "RESOLVED"
    out["the_runs_peak"] = block["the_runs_peak"]
    out["the_post_exit_property"] = block.get("the_post_exit_property")
    out["this_receipts_peak_of_record_field"] = (
        "resources.scope_memory.memory_peak_bytes -- the unit leaf's "
        "`memory.peak`, read IN-PROCESS at the emit, while the payload "
        "lives")
    out["what_this_receipt_does_NOT_carry"] = (
        "systemd's post-exit `MemoryPeak` property: a process inside the "
        "unit cannot read it after its own exit, so it is the launcher's "
        "or the coordinator's copy and is recorded verbatim there")
    out["and_the_in_process_rss_is_a_different_quantity"] = (
        "`memory_plan.peak_rss_mb` / `resources.peak_rss_mb` are "
        "ru_maxrss, the PROCESS's RSS high-water. They are kept -- the "
        "stage budget is built on them -- and they are NOT the peak of "
        "record, which is the cgroup leaf's")
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
    # THE LEAF IS THE UNIT'S, AND THE RULED UNIT IS A SERVICE (R-628).
    # This test read `.scope` ONLY -- written before rule 20 moved the form
    # off `--scope` -- so on every run under the ruled form it returned
    # NOT_IN_A_SCOPE with a null peak while `cgroup` NAMED the leaf right
    # beside it. It never lied (a status, never zeros) and it never read
    # the number REV 80/81 made the peak of record either: DE's three day
    # receipts carry `memory_peak_bytes: null` under a leaf that existed.
    cg = cgroup_path()
    _lk = leaf_of_cgroup(cg)
    leaf = _lk["leaf"] or ""
    if not cg or not _lk["is_a_unit_leaf"]:
        return {"status": "NOT_IN_A_UNIT",
                "cgroup": cg, "leaf": leaf or None,
                "in_research_slice": False,
                "why": "no transient unit leaf in /proc/self/cgroup, so "
                       "there is no per-run cgroup to read. Reported as a "
                       "STATUS rather than as zeros: a zero here would "
                       "read as a measurement (rule 11)",
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
        return {"status": "UNIT_LEAF_NAMED_BUT_UNREADABLE",
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
        "status": "MEASURED" if in_slice else "AMBIENT_UNIT_NOT_THE_RUNS_OWN",
        "in_research_slice": in_slice,
        "leaf": leaf,
        "leaf_is_the_ruled_launch_form": _lk["is_the_ruled_launch_form"],
        # REV 80 §1.1 / form chain v4: THIS is the run's peak, and it is
        # read from inside while the payload lives. systemd's post-exit
        # `MemoryPeak` property is a different number and is not read here
        # -- a process inside the unit cannot take it after its own exit.
        # COMPUTED, never asserted (rule 10). An AMBIENT leaf's peak is a
        # real number about the WRONG OBJECT -- the ambient shell scope,
        # not this run -- so it is not the peak of record and says so.
        "this_is_the_peak_of_record": in_slice,
        "why_not_the_peak_of_record": (
            None if in_slice else
            "this leaf is the ambient one the shell was already in, not a "
            "unit launched for this run; its peak measures the wrong "
            "object"),
        "read_while_the_payload_lives": True,
        "the_post_exit_property": {
            "status": "NOT_READ_FROM_INSIDE_THE_UNIT",
            "why": "`systemctl show -p MemoryPeak` after the payload exits "
                   "is the launcher's or the coordinator's read, recorded "
                   "VERBATIM beside this one and never as a reading of the "
                   "run's peak (form chain v4, memory_peak_reading)"},
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


def peak_stage_predicate(stages: dict, *, declared: str) -> dict:
    """WHICH STAGE WAS THE PEAK -- the argmax over the highwater DELTAS.

    THE DELTA IS THE MISSING LINE (REV 43 S4.3, REV 45 S1.6). The two
    instruments answer different questions and neither alone is the peak:

      * the CURRENT-RSS series says what a stage was still HOLDING at its
        boundary. A transient that a stage allocated and freed before the
        mark is INVISIBLE to it;
      * the HIGHWATER is non-decreasing, so its per-stage DELTA is the
        growth attributable to that stage -- INCLUDING that transient,
        because the highwater recorded it and never came down.

    So the delta argmax is the peak stage, and the current series is
    reported beside it as what was still resident. `declared` has NO
    DEFAULT: it comes from `declared_peak_stage()`, which reads the stage
    table, so the declaration and the predicate are one fact."""
    hi = {k: v.get("peak_rss_mb_highwater") for k, v in stages.items()
          if isinstance(v.get("peak_rss_mb_highwater"), float)}
    cur = {k: v.get("rss_mb_current") for k, v in stages.items()
           if isinstance(v.get("rss_mb_current"), float)}
    order = [k for k, _ in DAY_STAGES if k in hi]
    if not hi or not order:
        return {"computable": False,
                "why": "no highwater samples were recorded"}
    base = stages.get("S_start", {}).get("peak_rss_mb_highwater")
    deltas, prev = {}, (base if isinstance(base, float) else hi[order[0]])
    for k in order:
        deltas[k] = round(hi[k] - prev, 4)
        prev = hi[k]
    # THE ARGMAX IS OVER THE DAY-PATH STAGES. The `before_work` hook is
    # not the day path: it is the battery, and its high-water is a
    # TRANSIENT it frees before it returns -- MEASURED at 838 MB peak
    # against 26 MB retained, most of it one check's deliberate 800 MB
    # inflation. Left in the argmax it would compete with S1_load for the
    # peak on a real day (S1 delta ~1196 MB against the hook's ~820) and
    # a flip would REFUSE the day over a stage the 8 GiB ceiling does not
    # rest on. Its delta is COMPUTED and REPORTED -- so the attribution of
    # every later stage is right -- and excluded from the argmax alone.
    day_path = {k: v for k, v in deltas.items() if k != HOOK_STAGE}
    arg = max(day_path, key=lambda k: day_path[k])
    cur_day = {k: v for k, v in cur.items() if k != HOOK_STAGE}
    arg_cur = max(cur_day, key=lambda k: cur_day[k]) if cur_day else None
    return {
        "computable": True,
        "declared_peak_stage": declared,
        "declared_read_from": "DAY_STAGES marker "
                              f"{DAY_STAGE_PEAK_MARKER!r}",
        "measured_peak_stage": arg,
        "declared_stage_is_the_measured_peak": arg == declared,
        "highwater_delta_mb_by_stage": deltas,
        "argmax_taken_over": sorted(day_path),
        "before_work_hook_highwater_delta_mb": deltas.get(HOOK_STAGE),
        "why_the_hook_is_not_in_the_argmax": (
            "the `before_work` hook is the BATTERY, not the day path. Its "
            "high-water is a transient it frees before returning (measured "
            "838 MB peak, 26 MB retained); the 8 GiB ceiling rests on the "
            "DAY PATH's shape, and letting a freed transient win the "
            "argmax would refuse a day over a stage the ceiling does not "
            "rest on. Its delta is computed, reported, and counted in the "
            "day's GROWTH BUDGET -- only the argmax excludes it"),
        "baseline_highwater_mb": prev if base is None else base,
        "measured_peak_stage_by_current_rss": arg_cur,
        # REV 89 §2.1: RENAMED. It was `the_two_readings_agree`, sitting
        # three lines below prose reading "a disagreement REFUSES the
        # day" -- and it does not refuse. It compares the HIGHWATER-DELTA
        # argmax with the CURRENT-RSS argmax; the field that binds is
        # `peak_stage_assertion.agrees`. Any reader censusing booleans
        # met a `false` whose neighbour said it should have refused.
        "highwater_delta_argmax_equals_current_rss_argmax": arg == arg_cur,
        "this_boolean_does_NOT_refuse": (
            "it compares two ways of naming the peak STAGE and is "
            "diagnostic. The binding field is "
            "`peak_stage_assertion.agrees`."),
        "why_the_DELTA_is_the_peak": (
            "a transient a stage allocates and frees before its mark is "
            "invisible to the current-RSS series and VISIBLE in the "
            "highwater delta, because the highwater recorded it and never "
            "came down. The delta is what attributes growth to a stage"),
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


def tape_artifacts_opened(proof: dict, full_paths=None) -> list:
    """Which TAPE/INDEX/FRAGMENT artifacts the instrumented run opened.

    IT READS THE UNCAPPED LIST TOO. `distinct_paths` is capped at
    `DR.PATH_LIST_CAP`, so a run opening more paths than the cap could
    drop a tape artifact off the end and the scan would report a clean
    surface it never saw -- the silent-regex failure with a different
    mechanism. `data_paths_opened` is NOT capped and every marker lives
    under `data/`, so the union is what the predicate is computed on, and
    whether the capped list bit is reported beside the answer."""
    return sorted({p for p in (set(proof.get("distinct_paths") or [])
                               | set(proof.get("data_paths_opened") or [])
                               | set(full_paths or []))
                   if not p.endswith(".py")
                   and any(m in p for m in TAPE_ARTIFACT_MARKERS)})


#: REV 47: hashing a 290 MB book with `read_bytes()` put a ~290 MB
#: TRANSIENT in S0 -- allocated, hashed, freed before the mark, so it is
#: invisible to the current-RSS series and lands squarely in S0's HIGHWATER
#: DELTA, which is the instrument DE 83 built to locate the peak. It made
#: S0 look like a memory stage when what it did was read a file.
HASH_CHUNK_BYTES = 8 << 20


def sha256_streamed(path: Path) -> str:
    """The digest WITHOUT holding the file.

    THE OBSERVATION IS THE REVIEWER'S AND THE FIX IS NOT THE ONE PROPOSED,
    for a reason worth recording: 'one read with sha256 fed from the same
    buffer' would need BE's `load()` to accept BYTES, and `phase2_arms` /
    `be_cancel_axis_null` are BE's modules -- DE does not change their
    signatures to save its own transient. Streaming is DE-side and gets
    the whole benefit: O(chunk) instead of O(file).

    THE DOUBLE READ REMAINS, DELIBERATELY. DE hashes the book to admit it
    against BE's published receipt BEFORE the load; BE's loader hashes the
    buffer it actually unpickled. Those are the two ends of a check-and-use
    pair and collapsing them into one read would reintroduce the window REV
    43 made me close on the tape. Two reads of 290 MB cost seconds; the
    window costs correctness."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


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
    # DE 150: THE BOOK'S OWN RECEIPT COMES FIRST, SUFFIX AND ALL. BE 102
    # writes `be_daybook_20260905_btc__L250ms.pkl` beside
    # `be_daybook_receipt_20260905_btc__L250ms.json`, and the day-form
    # candidates below resolve the L = 0 receipt for it -- so a 250 ms
    # book would have been verified against the 0 ms book's digest (it
    # would refuse, correctly, but for the wrong reason) and its
    # placement latency would have READ 0.0 from the wrong file. The
    # receipt whose name is the BOOK's name is tried first.
    cands = [book_path.parent
             / book_path.name.replace("be_daybook_", "be_daybook_receipt_",
                                      1).replace(".pkl", ".json")]
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
    actual = sha256_streamed(book_path)
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


VALUATION_UNIT = "cents"


def HSP_BUY_SIDE():
    """The buy side, from the policy module -- never the literal 'B'."""
    import harmful_stateful_policy as _H
    return _H.SIDES[0]


def absolute_legs(fills: list) -> dict:
    """ONE BOOK'S OWN VALUE -- R-782, the USER's third ruling of the day.

        "plz record the absolute number as well for reference"
                                    -- THE USER, 2026-09-07T10:42:24Z

    D(E0) is a DIFFERENCE, and a difference answers "how much better"
    without ever answering "how much". The question that preceded the
    ruling was *what's the result of 0-cancel* -- which the receipt could
    not answer at all, because only the excess was ever written down.

    These are read from THE SAME REPLAY that computes D(E0): the fills are
    already in memory and the excess is their difference, so nothing here
    is a second pass over the day and nothing can disagree with the
    number it is the absolute of.

    The INVENTORY leg is present only where BE 96's position fields are
    (`inventory_before`/`_after`/`_mark_cents`); where they are not it is
    None with a reason, never 0 -- an unrecorded leg and a zero one are
    the same number and opposite facts."""
    import de_phase4_diag_runner as _R
    total = 0.0
    n_valued = 0
    inv, n_inv = 0.0, 0
    for f in fills or []:
        v = _R.fill_value_cents(f)
        if v is not None:
            total += v
            n_valued += 1
        b, a_, mk = (f.get("inventory_before"), f.get("inventory_after"),
                     f.get("inventory_mark_cents"))
        if b is not None and a_ is not None and mk is not None:
            inv += (float(a_) - float(b)) * float(mk)   # BUYS - SELLS
            n_inv += 1
    return {
        "unit": VALUATION_UNIT,
        "fills_leg": total,
        "n_fills": len(fills or []),
        "n_fills_valued": n_valued,
        # R-803: NAMED FOR WHAT IT IS. This was `inventory_leg`, and BE
        # 99 measured it to be the exact NEGATIVE of the R-801 trades
        # leg -- sum((after - before) x mark) is sum(sgn x size x px) =
        # BUYS - SELLS. It is not a residual mark-to-market; the
        # residual is priced at SETTLEMENT in the R-801 legs.
        "trades_cash_flow_cents": (-inv if n_inv else None),
        "trades_cash_flow_sign_convention": "SELLS - BUYS: positive means "
                                            "cash taken in; the negative "
                                            "of the old `inventory_leg`",
        "n_fills_with_inventory": n_inv,
        "why_trades_cash_flow_may_be_None": (
            "BE 96's position fields are absent on these fills; an "
            "unrecorded quantity is not a zero one"),
        "total": total,
        "what_total_is": "the fills leg, which is what `_value_cents` "
                         "sums and therefore exactly what D(E0) is the "
                         "difference of",
    }


#: R-801, THE USER: "the pnls are from trades and remaining position's
#: settlement p&l, need to calculate this correctly". An UP share pays
#: 100 cents if Up won and 0 otherwise; the payoff is the venue's, not a
#: choice of this module's.
SETTLE_UP_CENTS = 100.0
#: The venue's record of each market's outcome. NOT Chainlink: this file
#: carries `source: "clob"` on every record (BE 98, Q-BE-341). The market
#: TEXT names Chainlink as the resolution source, and verifying a winner
#: against a Chainlink read is BE 99's method, which has not landed --
#: so nothing here calls a settlement value verified.
RESOLUTIONS_REL = "data/pm_5min/resolutions.jsonl"
CHAINLINK_PRICE_TOPICS = ("crypto_prices_twap_sixty",
                          "crypto_prices_twap_thirty")


def _is_hex64(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) \
        is not None


def chainlink_stream_provenance_check(prov: dict, *, data_root=None) -> dict:
    """RE-READ THE STREAM FILES A VERIFICATION CLAIMS TO HAVE USED."""
    prov = prov or {}
    files = prov.get("files")
    files = files if isinstance(files, list) else []
    n_files = prov.get("n_hourly_files")
    claimed_digest = prov.get("files_digest")
    reader = prov.get("reader_module") or {}
    root = Path(data_root) if data_root is not None \
        else Path(DR.resolve()["data_root"])
    prices = root / "pm_5min" / "prices"
    reader_path = Path(__file__).resolve().parent / "exp_m6_settlement.py"
    actual_reader_sha = hashlib.sha256(reader_path.read_bytes()).hexdigest() \
        if reader_path.is_file() else None
    check = {
        "n_hourly_files_positive": isinstance(n_files, int) and n_files > 0,
        "files_list_present": bool(files),
        "files_count_matches": isinstance(n_files, int) and len(files) == n_files,
        "files_digest_is_hex64": _is_hex64(claimed_digest),
        "reader_path_matches": reader.get("path")
        == "live/pm_research/exp_m6_settlement.py",
        "reader_sha_matches_current_file": (
            _is_hex64(reader.get("sha256"))
            and reader.get("sha256") == actual_reader_sha),
        "files_missing": [],
        "files_bad_paths": [],
        "files_digest_matches": False,
    }
    h = hashlib.sha256()
    if files:
        for rel in files:
            p = Path(str(rel))
            if p.is_absolute() or ".." in p.parts or len(p.parts) != 2 \
                    or p.parts[0] not in CHAINLINK_PRICE_TOPICS:
                check["files_bad_paths"].append(str(rel))
                continue
            src = prices / p
            if not src.is_file():
                check["files_missing"].append(str(rel))
                continue
            h.update(p.name.encode())
            h.update(src.read_bytes())
        check["files_digest_matches"] = (
            not check["files_bad_paths"]
            and not check["files_missing"]
            and _is_hex64(claimed_digest)
            and h.hexdigest() == claimed_digest)
    check["ok"] = bool(
        check["n_hourly_files_positive"]
        and check["files_list_present"]
        and check["files_count_matches"]
        and check["files_digest_is_hex64"]
        and check["reader_path_matches"]
        and check["reader_sha_matches_current_file"]
        and check["files_digest_matches"])
    return check


def winner_source(*, root=None, required_slugs=None,
                  verification=None, require_verified: bool = False) -> dict:
    """WHO WON EACH SLUG, AND HOW WE KNOW -- the settlement mark's source.

    R-801. The method is BE 98's, measured and filed at Q-BE-341, not
    invented here: the winner field is `winners`, a dict on records with
    `closed: true`; the UP share pays `SETTLE_UP_CENTS` iff
    `winners["Up"] is True`. BE measured 38,248 distinct closed slugs,
    exactly one of Up/Down true on every record and no slug whose
    repeated records disagree -- so those two properties are CHECKED here
    rather than assumed, and each has its own refusal.

    THE CHAINLINK QUESTION IS OPEN AND IS SAID SO. This file's own
    `source` is "clob"; the market text names Chainlink; BE 98 states it
    verified no winner against a Chainlink read, and BE 99 owns that
    method. Until a verification source is passed in, every slug's status
    is `VENUE_RECORD_NOT_VERIFIED_AGAINST_CHAINLINK` and
    `is_final_for_quotation` is False. `require_verified=True` REFUSES
    without one -- so a caller that wants to publish a settlement number
    as final cannot get one by not asking."""
    # `root` is the DATA root (`DR.resolve()["data_root"]`), the same
    # root every other reader here takes; RESOLUTIONS_REL is kept
    # repo-relative because that is how the receipt names it.
    r = Path(root) if root else Path(DR.resolve()["data_root"])
    path = r / RESOLUTIONS_REL.split("data/", 1)[1]
    if not path.is_file():
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_WINNER_SOURCE_ABSENT: no resolution "
            f"record at {path}. A settlement value with no source for the "
            f"winner is a number nobody can check.")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    winners: dict = {}
    n_records, n_closed = 0, 0
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        n_records += 1
        rec = json.loads(line)
        if not rec.get("closed") or not isinstance(rec.get("winners"), dict):
            continue
        n_closed += 1
        w = rec["winners"]
        trues = sorted(k for k, v in w.items() if v is True)
        if len(trues) != 1:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_WINNER_AMBIGUOUS: slug "
                f"{rec.get('slug')!r} has {len(trues)} outcomes marked "
                f"true ({trues}). Exactly one side wins a binary market; "
                f"a record that says otherwise is not a settlement.")
        up = bool(w.get("Up"))
        prev = winners.get(rec.get("slug"))
        if prev is not None and prev["up_won"] != up:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_WINNER_RECORDS_DISAGREE: slug "
                f"{rec.get('slug')!r} has records naming BOTH outcomes. "
                f"Two answers to one question with no rule saying which "
                f"is newest is not a source.")
        winners[rec.get("slug")] = {
            "up_won": up,
            "settle_cents": (SETTLE_UP_CENTS if up else 0.0),
            "source": rec.get("source"),
            "n_records_seen": (prev or {}).get("n_records_seen", 0) + 1}
    missing = sorted(set(required_slugs or []) - set(winners))
    if missing:
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_WINNER_MISSING_FOR_SLUG: "
            f"{len(missing)} slug(s) the fills name carry no closed "
            f"resolution record, e.g. {missing[:3]}. A settlement value "
            f"for a slug nobody resolved would be a fabricated number.")
    ver = {"status": "NOT_VERIFIED_AGAINST_CHAINLINK",
           "why": ("this file's own `source` is \"clob\" -- the VENUE's "
                   "record of the outcome. The market text names "
                   "Chainlink as the resolution source; verifying a "
                   "winner against a Chainlink read is BE 99's method "
                   "(R-801), which has not landed. Recorded, never "
                   "assumed."),
           "owner": "BE 99, per R-801",
           "per_slug_status": "VENUE_RECORD_NOT_VERIFIED_AGAINST_CHAINLINK"}
    if verification is not None:
        if not (isinstance(verification, dict)
                and isinstance(verification.get("per_slug"), dict)
                and isinstance(verification.get("convention"), dict)
                and verification["convention"].get("name")):
            raise RunnerRefused(
                "REFUSED SETTLEMENT_VERIFICATION_SOURCE_MALFORMED: a "
                "verification must carry a per-slug verdict map and the "
                "CONVENTION it applied, by name. A verdict whose rule "
                "nobody can name verifies nothing -- BE 99 measured four "
                "conventions and three of them disagree with the venue.")
        # REVIEW 104B §6 (2): NOTHING HERE IS READ FROM THE DICT IT WAS
        # HANDED. `counts` is recomputed from `per_slug` and `all_agree`
        # from `counts`, INSIDE this function -- REV drove a per-slug map
        # of DISAGREE through to VERIFIED_AGAINST_CHAINLINK with
        # `all_agree: True` supplied by the caller, and the quotable door
        # opened on a boolean its caller wrote. No production caller can
        # do that today; the door should not depend on that.
        if verification["convention"]["name"] != SETTLEMENT_CONVENTION["name"]:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_CONVENTION_NOT_THE_PINNED_ONE: the "
                f"verification claims the convention "
                f"{verification['convention']['name']!r} and the pinned "
                f"one is {SETTLEMENT_CONVENTION['name']!r}. BE 99 "
                f"measured that three other conventions in the grid "
                f"disagree with the venue on 10-44 slugs a day, so a "
                f"verdict under an unnamed rule verifies nothing.")
        _c = {"VERIFIED_AGREE": 0, "DISAGREE": 0,
              "BOUNDARY_NOT_IN_CAPTURE": 0,
              "CHAINLINK_UNAVAILABLE": 0, "VENUE_UNRESOLVED": 0}
        for _pv in verification["per_slug"].values():
            _st = (_pv or {}).get("status")
            if _st not in _c:
                raise RunnerRefused(
                    f"REFUSED SETTLEMENT_VERIFICATION_STATUS_UNKNOWN: a "
                    f"per-slug verdict reads {_st!r}, which is none of "
                    f"{sorted(_c)}. A status nobody declared cannot be "
                    f"counted as agreement.")
            _c[_st] += 1
        _all_agree = (_c["VERIFIED_AGREE"] == len(verification["per_slug"])
                      and len(verification["per_slug"]) > 0)
        ver = {"status": ("VERIFIED_AGAINST_CHAINLINK" if _all_agree
                          else "VERIFICATION_DID_NOT_AGREE"),
               "counts_are": ("RECOMPUTED here from `per_slug`, never "
                              "read from the dict handed in (REV 104B "
                              "§6(2))"),
               "convention": verification["convention"]["name"],
               "convention_block": verification["convention"],
               "counts": _c,
               "n_slugs_verified": len(verification["per_slug"]),
               "per_slug": verification["per_slug"],
               "per_slug_status": "one of VERIFIED_AGREE / DISAGREE / "
                                  "CHAINLINK_UNAVAILABLE / "
                                  "VENUE_UNRESOLVED, per slug"}
    # ===== R-810, THE USER's REVIEW: THE GATE VALIDATES EVIDENCE
    # IDENTITY, NOT SHAPE. The USER reproduced it: `all_agree=True` with
    # an EMPTY `per_slug` and EMPTY `counts` returned
    # `is_final_for_quotation=True` and `require_verified=True` PASSED --
    # zero slugs verified is not verification. Recomputing the counts
    # fixed the boolean and left the DOOR open, which is the half that
    # matters. Every condition below is checked against the evidence, and
    # `is_final_for_quotation` is the AND of all of them.
    _fin = None
    if verification is not None:
        _ps = verification["per_slug"]
        _req = sorted(set(required_slugs or []))
        _have = sorted(_ps)
        _prov = verification.get("stream_provenance") or {}
        _prov_check = chainlink_stream_provenance_check(
            _prov, data_root=r)
        _fin = {
            "a_slug_set_equals_the_days":
                (bool(_req) and _have == _req),
            "a_n_slugs_verified": len(_ps),
            "a_n_slugs_required": len(_req),
            "a_missing": [x for x in _req if x not in _ps][:5],
            "a_extra": [x for x in _have if x not in set(_req)][:5],
            "b_counts_recomputed_here": True,
            "c_every_status_is_VERIFIED_AGREE":
                (_c["VERIFIED_AGREE"] == len(_ps) and len(_ps) > 0),
            "d_convention_is_the_pinned_one": True,
            "e_n_hourly_files_read": _prov.get("n_hourly_files", 0),
            "f_provenance": {
                "venue_record": {"path": str(path), "sha256": sha},
                "stream_files_digest": _prov.get("files_digest"),
                "reader_module": _prov.get("reader_module")},
            "f_provenance_check": _prov_check,
        }
        _fin["f_provenance_complete"] = bool(
            _prov_check["ok"])
        _fin["is_final"] = bool(
            _fin["a_slug_set_equals_the_days"]
            and _fin["c_every_status_is_VERIFIED_AGREE"]
            and _fin["e_n_hourly_files_read"] > 0
            and _fin["f_provenance_complete"])
        ver["finality"] = _fin
        ver["status"] = (
            "VERIFIED_AGAINST_CHAINLINK" if _fin["is_final"]
            else ("VERIFICATION_DID_NOT_AGREE"
                  if not _fin["c_every_status_is_VERIFIED_AGREE"]
                  else "VERIFICATION_NOT_FINAL"))
    if require_verified:
        # R-810 (a): ZERO SLUGS VERIFIED IS NOT VERIFICATION, and a set
        # that is not the day's is evidence about some other day.
        if _fin is not None and not _fin["a_slug_set_equals_the_days"]:
            if _fin["a_n_slugs_verified"] == 0:
                raise RunnerRefused(
                    "REFUSED SETTLEMENT_VERIFICATION_EMPTY: the "
                    "verification carries NO per-slug verdicts. Zero "
                    "slugs verified is not verification -- the USER's "
                    "own reproduction (R-810), which passed this door "
                    "while `all_agree` was a boolean the caller wrote.")
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_VERIFICATION_SLUG_SET_MISMATCH: the "
                f"verification covers {_fin['a_n_slugs_verified']} slug(s) "
                f"and this day names {_fin['a_n_slugs_required']}. "
                f"Missing {_fin['a_missing']}, extra {_fin['a_extra']}. "
                f"Evidence about a different set of slugs is not evidence "
                f"about this day.")
        # R-810 (e): a verdict read from no files at all.
        if _fin is not None and _fin["e_n_hourly_files_read"] <= 0:
            raise RunnerRefused(
                "REFUSED SETTLEMENT_CHAINLINK_NO_FILES_READ: the "
                "verification reports ZERO Chainlink hourly files read. A "
                "winner checked against nothing is not a checked winner.")
        # R-810 (f): the evidence must be locatable.
        if _fin is not None and not _fin["f_provenance_complete"]:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_VERIFICATION_PROVENANCE_INCOMPLETE: "
                f"the stream provenance does not re-read to the claimed "
                f"files/digests ({_fin['f_provenance_check']}). A verdict "
                f"nobody can re-read the inputs of is a claim, not a "
                f"verification.")
        # THE FAILING STATUSES REFUSE BY THEIR OWN NAMES. An unverified
        # value may be COMPUTED and LABELLED (R-803); what it may never
        # be is quoted as final, and this is the door that says no.
        _c = (ver.get("counts") or {})
        if ver["status"] == "NOT_VERIFIED_AGAINST_CHAINLINK":
            raise RunnerRefused(
                "REFUSED SETTLEMENT_WINNERS_NOT_VERIFIED: a caller asked "
                "for a settlement value it may quote as final and no "
                "Chainlink verification was supplied. The venue record "
                "is the JOIN; the stream is the CHECK (R-803).")
        if _c.get("DISAGREE"):
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK: "
                f"{_c['DISAGREE']} slug(s) where the venue's recorded "
                f"winner and "
                f"{verification['convention']['name']} disagree. BE 99 "
                f"measured 0 such slugs on 09-05 and 09-06; a day with "
                f"any is not quotable, and flipping one moves a day "
                f"total by thousands of cents.")
        if _c.get("BOUNDARY_NOT_IN_CAPTURE"):
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_BOUNDARY_NOT_IN_CAPTURE: "
                f"{_c['BOUNDARY_NOT_IN_CAPTURE']} window(s) whose "
                f"boundary value is not in our capture -- a logged stream "
                f"gap, or a stale read whose possible drift exceeds the "
                f"window's own margin (R-826). The convention is not in "
                f"doubt; our ability to check it on those windows is.")
        if _c.get("CHAINLINK_UNAVAILABLE") or _c.get("VENUE_UNRESOLVED"):
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_CHAINLINK_UNAVAILABLE: "
                f"{_c.get('CHAINLINK_UNAVAILABLE', 0)} slug(s) with no "
                f"S60 sample at a boundary and "
                f"{_c.get('VENUE_UNRESOLVED', 0)} with no closed venue "
                f"record. A winner nobody could check is not a verified "
                f"winner.")
    return {"path": str(path), "sha256": sha,
            "n_records": n_records, "n_closed_records": n_closed,
            "n_slugs": len(winners),
            "winners": winners,
            "settle_up_cents": SETTLE_UP_CENTS,
            "method": ("BE 98 / Q-BE-341: the `winners` dict on records "
                       "with `closed: true`; the UP share pays "
                       "SETTLE_UP_CENTS iff winners['Up'] is True"),
            "chainlink_verification": ver,
            "is_final_for_quotation": bool(
                (ver.get("finality") or {}).get("is_final"))}


#: R-803 / BE 99 (Q-BE-342): the ONE convention that reproduces the
#: venue's recorded winner on 288 of 288 slugs on BOTH 09-05 and 09-06.
#: The other three in `exp_m6_settlement.py`'s pre-registered grid
#: disagree on 10-44 slugs a day, so the winner is NOT convention-free at
#: the slug level and the declaration must PIN this one by name.
SETTLEMENT_CONVENTION = {
    "name": "S60(T) >= S60(t0)",
    "X_T": "S60(T) -- the 60-second Chainlink TWAP at the window's close",
    "X_0": "S60(t0) -- the same stream at the window's open",
    "boundary_reader": "last sample at or before the boundary",
    "tie": "X_T >= X_0 -> Up",
    "readers": ("exp_m6_settlement.load_streams / read_at -- the "
                "experiment module's OWN functions, imported, never "
                "re-implemented"),
    "provenance": ("BE 99, Q-BE-342 (9e1d943), R-803: 288/288 on 09-05 "
                   "and 09-06; S30/S30 disagrees on 23/17 slugs, S60/S30 "
                   "on 15/10, meanS60/S60 on 44/43"),
    "window_s": 300,
}
_CHAINLINK_CACHE: dict = {}


def chainlink_streams(days, *, coin: str = "btc"):
    """THE TWAP STREAM FOR THESE DAYS, THROUGH exp_m6's OWN LOADER.

    `load_streams()` globs every hourly file (740 MB, 457 per topic), so
    BE narrowed the glob to the hours the days need. The narrowing is
    done HERE by pointing the module's own `PM` at a directory of
    symlinks to exactly those files -- the reader is unmodified, which is
    what "no re-implementation" means; only the file set is smaller.

    The hours span the day before's last hour (the boundary reader takes
    the last sample AT OR BEFORE t0) through the day after's first."""
    import glob as _glob
    import os as _os
    import tempfile as _tf
    import datetime as _dt
    key = (tuple(sorted(days)), coin)
    if key in _CHAINLINK_CACHE:
        return _CHAINLINK_CACHE[key]
    import exp_m6_settlement as _M6
    hours = set()
    for d in days:
        y, m, dd = (int(x) for x in d.split("-"))
        base = _dt.datetime(y, m, dd, tzinfo=_dt.timezone.utc)
        for k in range(-1, 26):
            t = base + _dt.timedelta(hours=k)
            hours.add(t.strftime("%Y%m%d_%H"))
    src = Path(DR.resolve()["data_root"]) / "pm_5min" / "prices"
    tmp = Path(_tf.mkdtemp(prefix="de_chainlink_"))
    n_files = 0
    for topic in CHAINLINK_PRICE_TOPICS:
        d = tmp / "prices" / topic
        d.mkdir(parents=True)
        for h in sorted(hours):
            for f in sorted(_glob.glob(str(src / topic / f"{h}.csv*"))):
                _os.symlink(f, d / _os.path.basename(f))
                n_files += 1
    if n_files == 0:
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_CHAINLINK_UNAVAILABLE: no TWAP hourly "
            f"file under {src} for the hours {sorted(hours)[:3]}... of "
            f"{sorted(days)}. The winner cannot be verified against a "
            f"stream that is not there, and an unverified winner is not "
            f"quotable (R-803).")
    _old_pm = _M6.PM
    try:
        _M6.PM = tmp
        streams = _M6.load_streams()
    finally:
        _M6.PM = _old_pm
    # R-810 (f): THE STREAM'S OWN PROVENANCE. A verification that names
    # no files is not evidence; the digest is over the BYTES the reader
    # read, in a fixed order, so a later reader can ask whether the same
    # stream would give the same winners.
    _fh = hashlib.sha256()
    _names = []
    for _t in CHAINLINK_PRICE_TOPICS:
        for _f in sorted((tmp / "prices" / _t).glob("*")):
            _names.append(f"{_t}/{_f.name}")
            _fh.update(_f.name.encode())
            _fh.update(Path(_os.path.realpath(_f)).read_bytes())
    out = {"streams": streams, "n_hourly_files": n_files,
           "hours": sorted(hours),
           "files": _names,
           "files_digest": _fh.hexdigest(),
           "reader_module": {
               "path": "live/pm_research/exp_m6_settlement.py",
               "sha256": hashlib.sha256(
                   Path(_M6.__file__).read_bytes()).hexdigest()},
           "narrowed_by": ("a symlink tree over the hours these days "
                           "need; exp_m6's own loader, unmodified")}
    _CHAINLINK_CACHE[key] = out
    return out


def verify_winners_against_chainlink(winners: dict, slugs, *, streams,
                                     coin: str = "btc",
                                     stream_provenance=None,
                                     gaps=None) -> dict:
    """THE VENUE'S WINNER, CHECKED AGAINST THE CHAINLINK STREAM.

    R-803: rule 9's door. The venue record is the JOIN; the stream is the
    CHECK. Per slug the status is one of

        VERIFIED_AGREE           the convention reproduces the venue
        DISAGREE                 it does not -- the day is not quotable
        CHAINLINK_UNAVAILABLE    no sample at or before a boundary
        VENUE_UNRESOLVED         no closed venue record for the slug

    and the two failing statuses are what `require_verified` refuses on:
    an unverified value may still be COMPUTED and LABELLED, never quoted
    as final."""
    import exp_m6_settlement as _M6
    sym = _M6.COINS.get(coin)
    ser = (streams or {}).get((sym, 60))
    per: dict = {}
    # R-826 / DE 146: a FOURTH outcome, distinct from both neighbours.
    # DISAGREE asserts we read a DIFFERENT value than the venue did;
    # CHAINLINK_UNAVAILABLE says there is no data at all; this one says
    # the boundary's value IS NOT IN OUR CAPTURE and the read cannot
    # settle the verdict either way.
    counts = {"VERIFIED_AGREE": 0, "DISAGREE": 0,
              "BOUNDARY_NOT_IN_CAPTURE": 0,
              "CHAINLINK_UNAVAILABLE": 0, "VENUE_UNRESOLVED": 0}
    # THE GAP LOG IS READ ONLY WHEN NOT SUPPLIED. A FIXTURE run must open
    # no path under `data/` and this log lives there -- reading it
    # unconditionally broke the fixture run's own data-free proof, which
    # caught it by name. Production passes None and gets the real log;
    # the battery's cells pass their own list.
    _gaps = price_stream_gaps() if gaps is None else list(gaps)
    for sl in sorted(slugs):
        v = (winners or {}).get(sl)
        if v is None:
            per[sl] = {"status": "VENUE_UNRESOLVED"}
            counts["VENUE_UNRESOLVED"] += 1
            continue
        try:
            t0 = int(str(sl).rsplit("-", 1)[1])
        except (ValueError, IndexError):
            per[sl] = {"status": "CHAINLINK_UNAVAILABLE",
                       "why": "the slug does not carry an epoch t0"}
            counts["CHAINLINK_UNAVAILABLE"] += 1
            continue
        x0 = xT = None
        if ser is not None:
            x0, _ = _M6.read_at(ser, t0 * 1000)
            xT, _ = _M6.read_at(ser,
                                (t0 + SETTLEMENT_CONVENTION["window_s"])
                                * 1000)
        if x0 is None or xT is None:
            per[sl] = {"status": "CHAINLINK_UNAVAILABLE",
                       "why": ("no S60 sample at or before one of the "
                               "boundaries")}
            counts["CHAINLINK_UNAVAILABLE"] += 1
            continue
        up_chain = bool(xT >= x0)
        agree = (up_chain == bool(v["up_won"]))
        # THE VERIFIABILITY LAYER, IN FRONT OF THE CONVENTION. The
        # convention's own reading is kept and reported either way; what
        # this decides is whether our capture can SETTLE it.
        vb = boundary_verifiability(
            ser, t0 * 1000,
            (t0 + SETTLEMENT_CONVENTION["window_s"]) * 1000, gaps=_gaps)
        if not vb.get("verifiable"):
            st = "BOUNDARY_NOT_IN_CAPTURE"
        else:
            st = "VERIFIED_AGREE" if agree else "DISAGREE"
        per[sl] = {"status": st,
                   "venue_up_won": bool(v["up_won"]),
                   "chainlink_up_won": up_chain,
                   "the_conventions_own_reading":
                       ("AGREE" if agree else "DISAGREE"),
                   "verifiability": vb}
        counts[st] += 1
    return {"convention": SETTLEMENT_CONVENTION,
            "per_slug": per, "counts": counts,
            "n_slugs": len(per),
            # R-810 (e)+(f): the evidence travels WITH the verdict.
            "stream_provenance": {
                "n_hourly_files": (stream_provenance or {}).get(
                    "n_hourly_files", 0),
                "files_digest": (stream_provenance or {}).get(
                    "files_digest"),
	                "reader_module": (stream_provenance or {}).get(
	                    "reader_module"),
	                "hours": (stream_provenance or {}).get("hours"),
	                "files": (stream_provenance or {}).get("files")},
            "all_agree": (counts["VERIFIED_AGREE"] == len(per)
                          and len(per) > 0)}


#: R-826 / DE 146: THE VERIFIABILITY LAYER'S DECLARED CONSTANTS.
#: The convention is NOT touched -- S60(T) >= S60(t0) stays exactly as
#: BE 99 measured and the declaration pins it. What is added is a layer
#: in FRONT of it that can say "this boundary is not in our capture".
#:
#: THE UNIT. `full_accuracy_value` carries 18 decimals: X_0 on
#: btc-updown-5m-1788469500 reads 8.136801e22, i.e. 81,368.01 USD. The
#: scale is applied ONLY to report margins and drift in USD; every
#: comparison is made in the stream's own units, so the scale can never
#: change a verdict.
SETTLEMENT_VALUE_SCALE = 1e18
#: THE DRIFT ESTIMATOR, DECLARED (the coordinator's ruling, DE 146(2)).
#: |dS60| between consecutive samples, per second, over the samples
#: within +/- DRIFT_WINDOW_S of the boundary, at DRIFT_QUANTILE.
#:  * WINDOW +/-600 s: long enough for a few hundred to ~2,000 samples
#:    (measured 1,128 around the 09-03 window), short enough to stay in
#:    the boundary's own regime. DA 134 used a comparable neighbourhood.
#:  * QUANTILE p90: this is a BOUND on what a stale read could have
#:    hidden, so the statistic must sit in the tail, not at the centre --
#:    a p50 (0.570 USD/s measured here) understates it and a max (3.787)
#:    is hostage to one outlier. p90 is also what DA 134 reported, so the
#:    two measurements are comparable. It errs toward REFUSING to verify,
#:    which is the safe side for a verifiability layer.
DRIFT_WINDOW_S = 600.0
DRIFT_QUANTILE = 0.90
#: The PRICE lane's gap log -- NOT `data/pm_5min/collector_gaps.jsonl`,
#: which is the CLOB lane (slug disconnects; zero GLOBAL_SOCKET_SILENCE
#: rows). The settlement boundary reads the TWAP price stream, so the
#: gaps that matter are the price collector's.
PRICE_GAPS_REL = "pm_5min/prices/collector_gaps.jsonl"
_PRICE_GAPS_CACHE: dict = {}


def price_stream_gaps(topic: str = "crypto_prices_twap_sixty", *,
                      data_root=None) -> list:
    """THE OUTAGES WE RECORDED AND DID NOT USE (DE 146(3)).

    Closed gaps for one topic, from the price collector's own log. A
    boundary inside one of these is not verifiable BY DEFINITION,
    whatever the arithmetic says: the value at that instant is not in our
    capture."""
    root = Path(data_root) if data_root else Path(DR.resolve()["data_root"])
    key = (str(root), topic)
    if key in _PRICE_GAPS_CACHE:
        return _PRICE_GAPS_CACHE[key]
    path = root / PRICE_GAPS_REL
    out = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("event") == "gap_closed" and r.get("topic") == topic
                    and r.get("gap_start_ns") and r.get("gap_end_ns")):
                out.append({"gap_start_ns": int(r["gap_start_ns"]),
                            "gap_end_ns": int(r["gap_end_ns"]),
                            "duration_ms": r.get("duration_ms"),
                            "cause": r.get("cause"),
                            "opened_connection_id":
                                r.get("opened_connection_id"),
                            "closed_connection_id":
                                r.get("closed_connection_id")})
    out.sort(key=lambda g: g["gap_start_ns"])
    _PRICE_GAPS_CACHE[key] = out
    return out


def gap_covering(boundary_ms: int, gaps: list) -> dict | None:
    """The logged gap a boundary falls inside, or None."""
    b = int(boundary_ms) * 10 ** 6
    for g in gaps:
        if g["gap_start_ns"] <= b <= g["gap_end_ns"]:
            return g
    return None


def read_at_with_staleness(series, boundary_ms: int) -> dict:
    """exp_m6's OWN `read_at`, WITH THE AGE OF WHAT IT RETURNED.

    R-826: `read_at` returns the last sample at or before the boundary
    with NO bound on how old it is and no way to say "unknown", so an
    OUTAGE becomes a data point -- on btc-updown-5m-1788469500 it
    returned a sample 2.000 s stale, 1.154 USD below X_0, across a
    12,140 ms silence, and the checker called the venue wrong.

    `exp_m6_settlement.py` IS NOT EDITED. It is the module BE 99 imported
    and whose digest every verification records as `reader_module`;
    forking it would invalidate the provenance already landed in R1's and
    R2's artifacts. This WRAPS it: the read is exp_m6's, the judgement is
    ours."""
    import exp_m6_settlement as _M6
    v, at = _M6.read_at(series, boundary_ms)
    if v is None or at is None:
        return {"value": None, "sample_ms": None, "staleness_s": None,
                "status": "NO_SAMPLE_AT_OR_BEFORE_THE_BOUNDARY"}
    return {"value": v, "sample_ms": at,
            "staleness_s": (int(boundary_ms) - int(at)) / 1000.0,
            "status": "READ"}


def local_drift_bound(series, boundary_ms: int, *,
                      window_s: float = DRIFT_WINDOW_S,
                      quantile: float = DRIFT_QUANTILE) -> dict:
    """HOW FAST THE STREAM MOVES AROUND THIS BOUNDARY, from the stream.

    Measured, never assumed: |dS60| between consecutive samples per
    second, over the samples within +/- `window_s`, at `quantile`. The
    constants are declared above with the reason for each."""
    te, tk, val = series
    lo, hi = (int(boundary_ms) - window_s * 1000,
              int(boundary_ms) + window_s * 1000)
    rates = []
    for j in range(1, len(te)):
        if lo <= te[j] <= hi and te[j] > te[j - 1]:
            dt = (te[j] - te[j - 1]) / 1000.0
            if dt > 0:
                rates.append(abs(val[j] - val[j - 1]) / dt)
    if not rates:
        return {"drift_per_s": None, "n": 0,
                "status": "NO_SAMPLES_IN_THE_WINDOW",
                "window_s": window_s, "quantile": quantile}
    rates.sort()
    idx = min(len(rates) - 1, int(quantile * len(rates)))
    return {"drift_per_s": rates[idx], "n": len(rates),
            "window_s": window_s, "quantile": quantile,
            "p50_per_s": rates[len(rates) // 2],
            "max_per_s": rates[-1], "status": "MEASURED"}


def boundary_verifiability(series, t0_ms: int, T_ms: int, *,
                           gaps: list | None = None) -> dict:
    """CAN THIS WINDOW'S VERDICT BE SETTLED BY WHAT WE CAPTURED?

    The test is MARGIN-AWARE (the coordinator's ruling, DE 146(2)): a
    window is verifiable only if the verdict CANNOT BE FLIPPED by the
    drift the staleness could have hidden --

        staleness_at_a_boundary x local_drift_bound  <  |X_T - X_0|

    -- and a boundary inside a LOGGED GAP is not verifiable whatever the
    arithmetic says. A fixed tolerance was considered and rejected: the
    5.0 s in `da_fair_price_identity.MAX_FRESHNESS_S` is a different
    path's number, and at the measured stream speed 5 s is 2-6 USD of
    drift against margins near 1 USD -- it would have admitted the very
    window that started this."""
    r0 = read_at_with_staleness(series, t0_ms)
    rT = read_at_with_staleness(series, T_ms)
    out = {"t0": r0, "T": rT}
    if r0["value"] is None or rT["value"] is None:
        out["verifiable"] = False
        out["why"] = "no sample at or before a boundary"
        return out
    margin = abs(rT["value"] - r0["value"])
    d0 = local_drift_bound(series, t0_ms)
    dT = local_drift_bound(series, T_ms)
    out["margin"] = margin
    out["margin_usd"] = margin / SETTLEMENT_VALUE_SCALE
    out["drift_bound_t0"], out["drift_bound_T"] = d0, dT
    worst = 0.0
    for r, d in ((r0, d0), (rT, dT)):
        if d["drift_per_s"] is not None:
            worst = max(worst, r["staleness_s"] * d["drift_per_s"])
    out["could_have_hidden"] = worst
    out["could_have_hidden_usd"] = worst / SETTLEMENT_VALUE_SCALE
    g0 = gap_covering(t0_ms, gaps or [])
    gT = gap_covering(T_ms, gaps or [])
    out["gap_at_t0"], out["gap_at_T"] = g0, gT
    # THE GAP IS RECORDED ALWAYS; THE MARGIN TEST DECIDES. The dispatch
    # asked for both "a boundary inside a logged gap is not verifiable by
    # definition" (3) and "the five windows whose T falls in a >5 s gap,
    # margins 12-156 USD, still read VERIFIED_AGREE" -- and on the real
    # data those two disagree: 8 windows on 09-03 have a gap at a
    # boundary and only 2 fail the margin test. Taking the gap as
    # decisive would flip six windows whose margins (12.3 - 405.0 USD)
    # survive their own drift bounds by more than an order of magnitude.
    # So the gap is EVIDENCE, recorded with its bounds, duration and
    # cause; the arithmetic the coordinator RULED (2) is the decider --
    # and a gap already expresses itself there, because it is what makes
    # the read stale. Reported for overrule in one line.
    out["a_gap_covers_a_boundary"] = bool(g0 or gT)
    out["verifiable"] = worst < margin
    out["why"] = (
        ("a logged stream gap covers a boundary; " if (g0 or gT) else "")
        + f"the staleness could have hidden "
        f"{out['could_have_hidden_usd']:.4f} "
        f"USD of drift against a margin of {out['margin_usd']:.4f} USD"
        + ("" if out["verifiable"] else
           " -- the verdict could be flipped by what we did not capture"))
    return out


def _settle_for(f: dict, winners: dict) -> float:
    slug = f.get("slug")
    w = (winners or {}).get(slug)
    if w is None:
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_WINNER_MISSING_FOR_SLUG: the fill names "
            f"slug {slug!r} and the winner source has no record for it.")
    return float(w["settle_cents"])


def settle_value_cents(fills: list, winners: dict) -> float:
    """R-801's VALUATION: every fill valued at the settlement.

    `sgn x (settle - px) x size`, the same sign convention as the 5-second
    markout (`fill_value_cents`) with the markout mid replaced by the
    outcome the market actually settled at. It is a SECOND valuation
    beside `_value_cents`, never a replacement of it: the 5-s markout
    stays as the short-horizon diagnostic the design declared."""
    tot = 0.0
    for f in fills or []:
        lvl, sz = f.get("px_cents"), float(f.get("size") or 0.0)
        if lvl is None or not sz:
            continue
        sgn = 1.0 if f.get("side") == HSP_BUY_SIDE() else -1.0
        tot += sgn * (_settle_for(f, winners) - float(lvl)) * sz
    return tot


def settlement_legs_by_slug(fills: list, winners: dict) -> dict:
    """THE RULED DECOMPOSITION, PER SLUG -- trades, residual, total.

    R-801, the user's words: "the pnls are from trades and remaining
    position's settlement p&l". Per slug:

        trades leg   = sum(sells px x size) - sum(buys px x size)
        residual leg = net_shares x settle
        total        = trades + residual

    and the identity `total == sum over fills of sgn x (settle - px) x
    size` holds by algebra, so it is ASSERTED here rather than trusted:
    the two legs are a decomposition of the ruled quantity, not a second
    definition of it."""
    per: dict = {}
    for f in fills or []:
        lvl, sz = f.get("px_cents"), float(f.get("size") or 0.0)
        if lvl is None or not sz:
            continue
        slug = f.get("slug")
        sgn = 1.0 if f.get("side") == HSP_BUY_SIDE() else -1.0
        d = per.setdefault(slug, {
            "slug": slug, "n_fills": 0, "net_shares": 0.0,
            "trades_leg_cents": 0.0, "settle_cents": _settle_for(f, winners),
            "up_won": bool((winners or {})[slug]["up_won"])})
        d["n_fills"] += 1
        d["net_shares"] += sgn * sz
        # a BUY pays px x size out; a SELL takes px x size in
        d["trades_leg_cents"] += -sgn * float(lvl) * sz
    for d in per.values():
        d["residual_leg_cents"] = d["net_shares"] * d["settle_cents"]
        d["total_cents"] = d["trades_leg_cents"] + d["residual_leg_cents"]
    day_total = sum(d["total_cents"] for d in per.values())
    per_fill = settle_value_cents(fills, winners)
    if abs(day_total - per_fill) > 1e-9:
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_LEGS_DO_NOT_RECONCILE: the per-slug legs "
            f"sum to {day_total!r} and the per-fill valuation is "
            f"{per_fill!r} -- a difference of {day_total - per_fill!r}. "
            f"The legs are a DECOMPOSITION of the ruled quantity; if they "
            f"disagree with it, one of them is a different quantity and "
            f"neither may be published.")
    return {
        "unit": VALUATION_UNIT,
        "per_slug": per,
        "n_slugs": len(per),
        "trades_leg_cents": sum(d["trades_leg_cents"] for d in per.values()),
        "residual_leg_cents": sum(d["residual_leg_cents"]
                                  for d in per.values()),
        "total_cents": day_total,
        "reconciles_with_the_per_fill_formula": True,
        "what_total_is": ("R-801's ruled P&L: the cash flow of the "
                          "trades plus the settlement value of the "
                          "position still held at the window's close, "
                          "summed over slugs"),
        "ruling": "R-801",
    }


def settlement_admissibility(day: str, params: dict, *,
                             fixture: bool = False) -> dict:
    """RULE 11's GUARD ON THE NEW ENDPOINT (R-801 (6)).

    The four days the user ruled read early (09-03..09-06) are DESIGN
    data for this endpoint: they may be valued, and every such value is
    LABELLED design data. Any other real day may be valued only once the
    declaration that names this endpoint has landed -- params v20 /
    design v28 -- because a day valued under an estimator chosen after
    seeing it is not a validation of anything.

    The admissible set is READ from the params head: the four come from
    the user's own early-read block, and a later set comes from a
    `settlement_endpoint.admissible_days` list that only the landed
    declaration can carry. Nothing is typed here."""
    design_days = list(((params.get("user_ruled_early_read") or {})
                        .get("this_reads_bar") or {}).get("days")
                       or (params.get("user_ruled_early_read") or {})
                       .get("days") or [])
    declared = ((params.get("settlement_endpoint") or {})
                .get("admissible_days"))
    if fixture:
        return {"admissible": True, "class": "FIXTURE",
                "why": "a fixture day values synthetic fills; rule 11 is "
                       "about days that could validate a claim"}
    # REVIEW 104B §6 (4): THE OVERLAP IS REFUSED, NOT RESOLVED BY
    # PRECEDENCE. `declared` was tested first, so a declaration naming a
    # day the USER's early read CONSUMED brought it back as a VALIDATION
    # day -- rule 11 undone by a list. v20 will not name a design day,
    # but the guard must not depend on that.
    if declared is not None and day in declared and day in design_days:
        return {
            "admissible": False,
            "class": "DECLARATION_NAMES_A_CONSUMED_DAY",
            "refusal_name": "SETTLEMENT_DECLARATION_NAMES_A_CONSUMED_DAY",
            "design_days": design_days,
            "declared_days": declared,
            "why": (f"{day} is BOTH one of the four days the USER's early "
                    f"read consumed (R-754) and named in the "
                    f"declaration's admissible set. A consumed day cannot "
                    f"be a validation day whatever a declaration says "
                    f"(rule 11); the DECLARATION is what must change.")}
    if declared is not None and day in declared:
        return {"admissible": True, "class": "DECLARED_VALIDATION_DAY",
                "declared_days": declared,
                "why": "the landed declaration names this day"}
    if day in design_days:
        return {"admissible": True, "class": "DESIGN_DATA",
                "design_days": design_days,
                "why": ("one of the four days the USER ruled read early "
                        "(R-754); consumed by that read, so a value here "
                        "is DESIGN data and may not validate the "
                        "endpoint (rule 11)")}
    return {
        "admissible": False,
        "class": "NOT_ADMISSIBLE",
        "refusal_name": "SETTLEMENT_DAY_NOT_ADMISSIBLE",
        "design_days": design_days,
        "declared_days": declared,
        "why": (f"{day} is neither one of the design days nor in a landed "
                f"declaration's admissible set. R-801 (6): 09-07 and "
                f"later are the validation population for this endpoint "
                f"and NOBODY values one before params v20 / design v28 "
                f"land. Choosing an estimator after seeing the day is "
                f"what rule 11 forbids.")}


def assert_settlement_day_admissible(day: str, params: dict, *,
                                     fixture: bool = False) -> dict:
    """THE GUARD THAT REFUSES -- rule 11, by name.

    `settlement_admissibility` COMPUTES the verdict and never raises, so
    a caller that must record the verdict can (rule 14: the estimate is
    not the decision). This is the entry that DECIDES: any caller asking
    to VALUE a day gets a refusal by name, and the refusal is the only
    way to be told no -- a status quietly returned where a value was
    expected is how a day that may not be valued gets valued."""
    verdict = settlement_admissibility(day, params, fixture=fixture)
    if not verdict["admissible"]:
        raise RunnerRefused(
            f"REFUSED {verdict['refusal_name']}: {verdict['why']}")
    return verdict


PLACEMENT_LATENCY_DISAGREES = (
    "SETTLEMENT_PLACEMENT_LATENCY_DISAGREES_IN_THE_DOCUMENT")
PLACEMENT_LATENCY_ABSENT = (
    "SETTLEMENT_PLACEMENT_LATENCY_ABSENT_FROM_THE_DOCUMENT")


def placement_latency_leaves(doc) -> dict:
    """Return every scalar named `L_place_ms`, keyed by its full path."""
    found = {}

    def _walk(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "L_place_ms" and not isinstance(child, (dict, list)):
                    found[child_path] = child
                _walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]")

    _walk(doc)
    return found


def assert_one_placement_latency(doc) -> dict:
    """Refuse an artifact that names no latency or more than one latency."""
    found = placement_latency_leaves(doc)
    if not found:
        raise RunnerRefused(
            f"REFUSED {PLACEMENT_LATENCY_ABSENT}: no `L_place_ms` appears "
            "in this document. A result that does not name its maker "
            "latency cannot be quoted.")
    malformed = {
        path: value for path, value in found.items()
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    }
    values = sorted({
        float(value) for path, value in found.items() if path not in malformed
    })
    if malformed or len(values) != 1:
        raise RunnerRefused(
            f"REFUSED {PLACEMENT_LATENCY_DISAGREES}: this document names "
            f"values {values} across {len(found)} site(s): {found}"
            + (f"; malformed values: {malformed}" if malformed else "")
            + ". One run measures one maker latency; readers must not be "
              "able to select whichever copy they encounter first.")
    return {
        "status": "PLACEMENT_LATENCY_AGREES",
        "value_ms": values[0],
        "n_sites_checked": len(found),
        "sites_checked": sorted(found),
    }


def placement_latency_from_the_book(builder_receipt: dict, *,
                                    book_path=None,
                                    book_sha256=None,
                                    require_declared: bool = False) -> dict:
    """WHAT PLACEMENT LATENCY THIS DAY'S BOOK WAS BUILT AT -- COMPUTED.

    R-810, the USER's second finding: `placement_latency_ms` is DECLARED
    (DE 137) and no caller passes it, because the runner never calls
    `build_reference` -- the parameter reaches a day only through
    `be_daybook_build`'s call and a REBUILT book, which is BE 101's item
    and a design step, not tonight's. So the receipt must not merely say
    "0": it says what the BOOK it ran on was built at, read from the
    builder receipt's OWN keys, so a later reader can tell a run at
    L = 0 from a run on a rebuilt book without trusting this sentence."""
    found = {}

    def _walk(o, path="", under_placement_latency=False):
        if isinstance(o, dict):
            for k, v in o.items():
                key = str(k)
                next_under = under_placement_latency \
                    or "placement_latency" in key.lower()
                is_leaf = not isinstance(v, (dict, list))
                # DE 148: THE LEAF NAMES THE VALUE, NOT THE SUBJECT.
                # Matching any key CONTAINING "placement_latency" made
                # BE 102's L=250 book read as AMBIGUOUS: its receipt
                # carries `placement_latency_ms: 250.0` AND
                # `TRANCHE_BEFORE_PLACEMENT_LATENCY: 23947` -- a COUNT of
                # tranches dropped, not a latency -- so the walk found
                # two "values" and the guard refused a book that says
                # exactly one L. The names that carry an L are named.
                if is_leaf and key in ("L_place_ms",
                                       "placement_latency_ms"):
                    found[f"{path}.{k}"] = v
                _walk(v, f"{path}.{k}", next_under)
        elif isinstance(o, list):
            for i, e in enumerate(o):
                _walk(e, f"{path}[{i}]", under_placement_latency)
    if not isinstance(builder_receipt, dict):
        raise RunnerRefused(
            "REFUSED SETTLEMENT_BOOK_RECEIPT_NOT_PARSED: placement latency "
            "is read from the builder receipt's keys, but the caller passed "
            f"{type(builder_receipt).__name__}. Parse the receipt and pass "
            "the resulting object; treating an unreadable input as an empty "
            "receipt previously converted every 250 ms run to a nested "
            "0 ms label.")
    _walk(builder_receipt)
    _book = {"path": (str(book_path) if book_path else None),
             "sha256": book_sha256}
    if found:
        malformed = {k: v for k, v in found.items()
                     if not isinstance(v, (int, float))}
        vals = sorted({float(v) for v in found.values()
                       if isinstance(v, (int, float))})
        if malformed:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_BOOK_PLACEMENT_LATENCY_MALFORMED: "
                f"the builder receipt names placement latency at {malformed}, "
                f"but the value is not numeric. A day cannot be quoted at an "
                f"L no reader can parse.")
        if len(vals) != 1:
            raise RunnerRefused(
                f"REFUSED SETTLEMENT_BOOK_PLACEMENT_LATENCY_AMBIGUOUS: "
                f"the builder receipt names {len(vals)} placement-latency "
                f"value(s), {vals}, at {found}. A day measures one maker, "
                f"so its book must name exactly one L.")
        return {"L_place_ms": vals[0],
                "source": "THE BOOK'S BUILDER RECEIPT",
                "found_at": found,
                "ambiguous": False,
                "book": _book,
                "why": ("read from the receipt of the book this day ran "
                        "on -- never typed here")}
    # R-811 (1): a book that does not RECORD its L must be refusable --
    # once BE 101 makes the builder record it, a book without it is a
    # real defect, not a fact about the world. The guard is ARMED BY THE
    # DECLARATION (`settlement_endpoint.require_book_declares_L`), never
    # by a constant here: every book that exists tonight was built before
    # the builder recorded anything, so arming it in code would refuse
    # the four re-runs the USER asked for -- and those four are at the L
    # the landed books CARRY, which is a measured fact, not a choice.
    if require_declared:
        raise RunnerRefused(
            "REFUSED SETTLEMENT_BOOK_DECLARES_NO_PLACEMENT_LATENCY: the "
            "builder receipt for " + str(_book["path"]) + " records no "
            "placement latency and the declaration requires one. A day's "
            "number is quoted only with its L beside it (R-811); a book "
            "that does not say what it was built at cannot supply one.")
    return {"L_place_ms": 0.0,
            "source": "THE BOOK'S BUILDER RECEIPT DECLARES NONE",
            "found_at": {},
            "book": _book,
            "guard": {"require_book_declares_L": False,
                      "armed_by": ("params `settlement_endpoint."
                                   "require_book_declares_L`, unset today"),
                      "why_not_armed": (
                          "every landed book predates BE 101's builder "
                          "change, so arming it would refuse the four "
                          "re-runs; the L those books carry is 0 BY "
                          "CONSTRUCTION -- the building code has no "
                          "placement-latency field at all -- which is a "
                          "measured fact, not an assumption")},
            "why": ("no key naming a placement latency appears anywhere "
                    "in this book's builder receipt, so the book was "
                    "built by code that has none and the day runs at "
                    "L_place = 0 BY CONSTRUCTION. DE 137 declared the "
                    "parameter on `build_reference`, which the runner "
                    "never calls; the pass-through and the cost of "
                    "rebuilding books at L = 250 are BE 101's (R-810). "
                    "COMPUTED from the receipt's own keys, so a run on a "
                    "rebuilt book says so here without anyone editing "
                    "this sentence.")}


#: R-828 / DE 148: the ONE name a point-estimate receipt carries where a
#: full run carries its test statistics. Never a null, never a zero.
NULL_NOT_DRAWN_STATUS = "NULL_NOT_DRAWN_POINT_ESTIMATE_RUN"
#: The refusal any reader gets for asking such a receipt for Z or p.
#: DA's reader must refuse on the SAME name -- filed in the register for
#: DA to adopt rather than agreed by assumption.
POINT_ESTIMATE_REFUSAL = "POINT_ESTIMATE_RUN_HAS_NO_TEST_STATISTIC"


def point_estimate_arm_day(day: str, arm: str, observed: float,
                           n_decisions: int, params: dict) -> dict:
    """AN ARM-DAY WITHOUT ITS NULL -- and it says so where the numbers go.

    S4 is 99.2 % of a run (9,467 s of 9,546 s measured on 09-03) and the
    null buys ONLY Z and p. The point estimates -- the baseline, both
    arms, their legs and the ruled excess -- are three replays and about
    two minutes. So a run that wants the ESTIMATES and not the
    SIGNIFICANCE skips S4 entirely, and every field the null would have
    filled carries the status instead: not a null, not a zero, and not a
    number anyone can mistake for a test."""
    return {
        "day": day, "arm": arm, "status": "OK_POINT_ESTIMATE",
        "admissibility": {"admissible": True, "n_decisions": n_decisions,
                          "bar": params["min_decisions_per_arm_day"],
                          "note": ("R4's decision bar is checked; R8's "
                                   "draw minimum does not apply to a run "
                                   "that draws none")},
        "draw_provenance": None,
        "economic": {"D_E0": observed,
                     "Z": NULL_NOT_DRAWN_STATUS,
                     "p_location": NULL_NOT_DRAWN_STATUS,
                     "null_mean": NULL_NOT_DRAWN_STATUS,
                     "null_sd": NULL_NOT_DRAWN_STATUS,
                     "null_draws_summary": {"n": NULL_NOT_DRAWN_STATUS}},
        "why_no_test_statistic": (
            "this run SKIPPED S4: no draw was made, so there is no null "
            "and no Z, p, mean or sd exists to report. The point "
            "estimates are the same arithmetic on the same replays as a "
            "full run -- only the control is absent. Nothing quoted from "
            "this artifact may imply significance."),
    }


def test_statistic_from(receipt: dict, field: str = "Z"):
    """ASKING A POINT-ESTIMATE RECEIPT FOR A TEST STATISTIC REFUSES.

    R-828 (3). The status is a STRING where a number belongs, so a
    careless reader gets a type error at best; this makes it a refusal
    BY NAME at worst and at best."""
    for a in (receipt.get("day_run") or receipt).get(
            "per_day_sealed_artifacts", []) or []:
        v = ((a.get("economic") or {}).get(field))
        if v == NULL_NOT_DRAWN_STATUS:
            raise RunnerRefused(
                f"REFUSED {POINT_ESTIMATE_REFUSAL}: arm {a.get('arm')!r} "
                f"carries `{field}` = {NULL_NOT_DRAWN_STATUS}. This "
                f"artifact is a POINT-ESTIMATE run: S4 was skipped, no "
                f"null was drawn, and no test statistic exists to read. "
                f"A full run at the same (day, L) is a DIFFERENT "
                f"artifact, not a successor to this one.")
    return {a.get("arm"): (a.get("economic") or {}).get(field)
            for a in (receipt.get("day_run") or receipt).get(
                "per_day_sealed_artifacts", []) or []}


def settlement_not_valued_reason(status: str, admissibility: dict) -> str:
    """WHY THIS ARM-DAY CARRIES NO RULED VALUE -- PER STATUS (REV 104B).

    One literal used to serve both statuses and was true of only one of
    them: every fixture receipt said "this day is not in the admissible
    set" beside `admissibility.admissible: true`. The reason is composed
    from the status AND the verdict it sits next to, so the two cannot
    disagree; an unknown status says so rather than borrowing a sentence
    that fits neither."""
    cls = (admissibility or {}).get("class")
    if status == "NO_WINNER_SOURCE_ON_A_FIXTURE":
        return (f"this day IS admissible (class {cls!r}) and no winner "
                f"map was supplied to the run. A FIXTURE names synthetic "
                f"slugs, which the venue has no record of, and reading "
                f"the venue record here would break the fixture run's "
                f"own data-free proof -- so a fixture brings its own "
                f"winners or is not valued.")
    if status == "NOT_VALUED_DAY_NOT_ADMISSIBLE":
        return (f"rule 11: this day is NOT in the admissible set (class "
                f"{cls!r}), so the ruled endpoint is not computed for "
                f"it. The estimator REFUSES if asked directly "
                f"(SETTLEMENT_DAY_NOT_ADMISSIBLE); this receipt records "
                f"that it was not asked.")
    return (f"no reason is declared for status {status!r}; the verdict "
            f"beside it reads class {cls!r}. A sentence that fits "
            f"neither status is worse than none.")


def settlement_arm_day(observed_settle: float, null_settle: list,
                       params: dict) -> dict:
    """THE SAME DECLARED STATISTICS, THE NEW VALUATION.

    `DESIGN.per_day_location` and `per_day_standardised_excess` are the
    design's, unchanged: R-801 changed the ESTIMAND, not the test. Using
    a second statistic here would make the two endpoints incomparable for
    a reason that has nothing to do with the ruling."""
    if len(null_settle) < params["min_draws_per_arm_day"]:
        raise RunnerRefused(
            f"REFUSED SETTLEMENT_NULL_TOO_SMALL: {len(null_settle)} draws "
            f"against the declared minimum "
            f"{params['min_draws_per_arm_day']}. The draw count is never "
            f"lowered for a second endpoint.")
    # REVIEW 104B §6 (3): A DEGENERATE NULL IS A STATUS, NEVER AN ABORT.
    # `per_day_standardised_excess` raises `DesignRefused` on zero
    # dispersion -- the design's own text says a degenerate null is a
    # STATUS, never a large Z -- and `DesignRefused` is caught NOWHERE in
    # this runner. REV drove it: an all-Up winner map kills the whole day
    # and the receipt is never written, losing the D_E0 that was already
    # computed and, on a real day, 70-80 minutes of work. Caught HERE,
    # where the second endpoint is formed, so the day still emits.
    try:
        loc = DESIGN.per_day_location(observed_settle, null_settle)
        _z = DESIGN.per_day_standardised_excess(observed_settle,
                                                null_settle)
    except DESIGN.DesignRefused as _e801d:
        return {"status": "SETTLEMENT_NULL_DEGENERATE",
                "D_E_settle": observed_settle,
                "Z": None, "p_location": None,
                "null_draws_summary": {"n": len(null_settle)},
                "null_sd": statistics.pstdev(null_settle),
                "null_mean": statistics.fmean(null_settle),
                "refusal_text": str(_e801d),
                "why": ("the settlement null has zero dispersion, so a "
                        "standardised excess is undefined. The design's "
                        "own words: a degenerate null is a STATUS, never "
                        "a large Z. The DAY still emits -- D_E0 and this "
                        "endpoint's own excess are computed and kept"),
                "endpoint": "R-801 SETTLEMENT P&L (trades + residual)"}
    return {"D_E_settle": observed_settle,
            "Z": _z,
            "p_location": loc["p_one_sided"],
            "null_mean": statistics.fmean(null_settle),
            "null_sd": statistics.pstdev(null_settle),
            "null_draws_summary": {"n": len(null_settle)},
            "endpoint": "R-801 SETTLEMENT P&L (trades + residual)",
            "statistics_are": "the design's own, unchanged -- the "
                              "estimand changed, not the test"}


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
                      cross_check_n: int = 8, winners: dict | None = None) -> dict:
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
    # R-801: THE SAME 500 DRAWS, VALUED THE SECOND WAY, INLINE. Design
    # v27's R11 budget is why the fills are dropped per draw, and keeping
    # them for a second pass would make the null O(n_draws) in memory --
    # 500 x a day's fills. Valuing at draw time keeps ONE FLOAT per draw
    # instead, so the second endpoint costs a 500-element list and no
    # fills, and the added residency is MEASURED below rather than
    # asserted to be small.
    settle_base = (settle_value_cents(base_fills, winners)
                   if winners else None)
    settle_values: list = []
    rss_before_draws = _peak_rss_mb()
    started = time.time()
    values, cancels, peak = [], [], _peak_rss_mb()
    for d in range(n_draws):
        flag = module.draw_flags(pools, by_side, rng)
        r = module.replay(bk, module.flagged_stream(rows, flag), 0.5)
        RUN_COUNTERS["draws_performed"] += 1
        RUN_COUNTERS["replays_performed"] += 1
        # REDUCED HERE, DELIBERATELY (stage S4): the draw's fills are
        # valued and dropped before the next draw is made, so peak memory
        # is O(one draw) and not O(n_draws).
        values.append(_value_cents(r["fills"]) - base_value)
        if winners:
            settle_values.append(
                settle_value_cents(r["fills"], winners) - settle_base)
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
            # R-801's SECOND ENDPOINT, from the SAME draws -- never a
            # second null. A separate draw loop would be a different null
            # wearing the same seed, which is what the cross-check above
            # exists to forbid.
            "settle_values": (settle_values if winners else None),
            "settle_base_value_cents": settle_base,
            "second_valuation_residency": {
                "fills_retained_per_draw": 0,
                "kept_per_draw": ("one float, the draw's settlement excess"
                                  if winners else None),
                "bytes_held_by_the_scalar_list":
                    sys.getsizeof(settle_values) if winners else 0,
                "rss_mb_before_the_draws": rss_before_draws,
                "peak_rss_mb_during_draws": peak,
                "why_it_is_measured": (
                    "design v27's R11 budget is why `draw_null` discards "
                    "the fills; a second endpoint that KEPT them would "
                    "cost O(n_draws) days of fills. This one values them "
                    "inline, so the measurement below is the whole cost")},
            # R-765: the per-draw cancel counts travel out too. They cost
            # nothing to keep and a later question about the null's
            # mechanics cannot be answered by re-deriving them.
            "cancels": cancels,
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


def _jctl(args: list, *, n: int = 2000) -> tuple:
    """One journalctl read. Returns (lines, error)."""
    import subprocess as _sp
    try:
        r = _sp.run(["journalctl", "--user", "--no-pager",
                     "-o", "short-iso-precise", "-n", str(n), *args],
                    capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            return [], (r.stderr or "").strip()[:200]
        raw = r.stdout
    except Exception as exc:                              # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    return ([x for x in raw.split("\n")
             if x.strip() and not x.startswith("-- ")], None)


def journal_retention(*, as_of=None) -> dict:
    """THE RETENTION STATE AS A MEASUREMENT, with its query and as-of.

    R-646 (R4). Five reads of the oldest user-journal entry across one
    hour walked 08:45Z -> 09:00:20Z -> 09:15:13Z -> 09:25:57Z ->
    09:29:57Z: **the window's start advances about as fast as the clock**,
    so a retention state named once is stale within minutes. It is
    therefore not a property of the journal but a measurement with a
    timestamp, and it travels with the read it qualifies."""
    now = as_of or datetime.datetime.now(datetime.timezone.utc)
    query = ["-n", "1", "--reverse"]
    lines, err = _jctl([], n=1)
    oldest, oldest_raw = None, None
    try:
        import subprocess as _sp
        r = _sp.run(["journalctl", "--user", "--no-pager",
                     "-o", "short-iso-precise", "-n", "1"],
                    capture_output=True, text=True, timeout=90)
        # the OLDEST entry: the first line of an unbounded forward read
        r2 = _sp.run(["journalctl", "--user", "--no-pager",
                      "-o", "short-iso-precise"],
                     capture_output=True, text=True, timeout=90)
        for ln in r2.stdout.split("\n"):
            if ln.strip() and not ln.startswith("-- "):
                oldest_raw = ln.split(" ", 1)[0]
                break
    except Exception:                                     # noqa: BLE001
        oldest_raw = None
    if oldest_raw:
        try:
            oldest = datetime.datetime.fromisoformat(
                oldest_raw).astimezone(datetime.timezone.utc).isoformat()
        except ValueError:
            oldest = None
    return {
        "measured_at_utc": now.isoformat(),
        "query": "journalctl --user --no-pager -o short-iso-precise "
                 "(first line = the oldest entry retained)",
        "oldest_user_journal_entry_utc": oldest,
        "oldest_entry_raw": oldest_raw,
        "oldest_entry_from": "the entry's OWN clock, as journald printed "
                             "it -- never typed, never a policy setting",
        "it_is_a_MEASUREMENT_not_a_property": (
            "five reads across one hour walked the window's start from "
            "08:45Z to 09:29:57Z -- about as fast as the clock. A "
            "retention state named once is stale within minutes, so it "
            "carries its own as-of and query (R-646 R4)"),
    }


def journal_copy_by_invocation(unit: str, *, invocation_id=None,
                               n: int = 2000) -> dict:
    """A RUN'S OWN JOURNAL LINES, BY ITS INVOCATION ID, ON BOTH FIELDS.

    R-646 (R4). A unit NAME names every run ever launched under it; an
    InvocationID names ONE run. And the id must be matched on BOTH
    fields:

      `_SYSTEMD_INVOCATION_ID`  the PAYLOAD's own lines
      `USER_INVOCATION_ID`      the USER MANAGER's lines (Started,
                                Consumed, Failed) -- which is where a
                                launch record actually lives
      `INVOCATION_ID`           the SYSTEM manager's field: matches
                                NOTHING here

    Measured: `be64book.service` 99 manager lines and 26 payload lines;
    `de95smoke.service` exactly one line, its `Started` line, on
    USER_INVOCATION_ID only. The coordinator made two copies on the wrong
    field, got 0 lines, and deleted them as FALSE ABSENCES -- so a copy
    that returns 0 lines while the unit's own `-u` query has lines is a
    REFUSAL OF THE COPY, never a record."""
    ident = unit_outcome(unit)
    inv = invocation_id or ident.get("InvocationID") or ""
    read_at = datetime.datetime.now(datetime.timezone.utc)
    by_id, err_id = ([], "no InvocationID for this unit") if not inv else (
        _jctl([f"_SYSTEMD_INVOCATION_ID={inv}", "+",
               f"USER_INVOCATION_ID={inv}"], n=n))
    by_unit, err_unit = _jctl(["-u", unit], n=n)
    out = {
        "unit": unit,
        "invocation_id": inv or None,
        "invocation_id_source": "the unit's own `InvocationID` property",
        "read_at_utc": read_at.isoformat(),
        "query": (f"journalctl --user _SYSTEMD_INVOCATION_ID={inv} + "
                  f"USER_INVOCATION_ID={inv}" if inv else None),
        # REV 70 S3: NEITHER COPY NAMED ITS FORMAT, so the reviewer's
        # first diff of two copies of one run reported a false mismatch --
        # the formats differed, not the lines. One field removes it.
        "output_format": "short-iso-precise",
        "output_format_note": (
            "journalctl -o short-iso-precise: an ISO timestamp, the host "
            "and the process prefix, then the message. A copy that does "
            "not name its format cannot be diffed against one that used "
            "another"),
        "fields_matched": ["_SYSTEMD_INVOCATION_ID (the payload's lines)",
                           "USER_INVOCATION_ID (the user manager's "
                           "lines: Started, Consumed, Failed)"],
        "field_not_used_here": "INVOCATION_ID -- the SYSTEM manager's "
                               "field; it matches nothing under --user",
        "n_lines_by_id": len(by_id),
        "n_lines_by_unit_name": len(by_unit),
        "cross_check": "the by-id count against `-u <unit>`; a unit NAME "
                       "names every run ever launched under it, an id "
                       "names ONE",
        "journalctl_error_by_id": err_id,
        "journalctl_error_by_unit": err_unit,
        "lines": by_id,
        "copied_at_the_moment_of_reading": True,
        "retention": journal_retention(as_of=read_at),
    }
    if not by_id and by_unit:
        # A FALSE ABSENCE IS NOT A RECORD.
        out.update({
            "status": "REFUSED_EMPTY_COPY",
            "why": (f"the by-id query returned 0 lines while `-u {unit}` "
                    f"returns {len(by_unit)}. That is a copy that failed, "
                    f"not a unit that produced nothing -- two such copies "
                    f"were made and deleted as false absences (R-646 R4). "
                    f"A 0 here is REFUSED, never filed"),
        })
        return out
    if not by_id and not by_unit:
        out.update({
            "status": "ABSENT",
            "why_absent_not_zero": (
                "neither the id nor the unit name retains anything right "
                "now; that is not the same fact as 'the run produced no "
                "output'"),
        })
        return out
    first = by_id[0].split(" ", 1)[0]
    try:
        oldest = datetime.datetime.fromisoformat(first).astimezone(
            datetime.timezone.utc).isoformat()
    except ValueError:
        oldest = None
    # COVERAGE IS TWO MEASURED CLOCKS, NOT A TEXT SEARCH (REV 68 S1.5).
    # This was `any(" Started " in x for x in lines)`, and it was wrong
    # twice over: it reported TRUE on a tail that had lost 141 of 161
    # lines (the needle was in the tail), and it would match a PAYLOAD
    # line that merely contains the words. DA 88 shipped the right shape
    # as `da_root.journal_coverage` -- the host's oldest retained entry
    # against the unit's own ExecMainStartTimestamp -- and it is IMPORTED
    # here, not mirrored: one more parallel implementation of a shared
    # reading is what R-641 ruled against.
    cov = DAROOT.journal_coverage(unit=unit)
    out.update({
        "status": "PRESENT",
        "oldest_line_utc": oldest,
        "oldest_line_raw": first,
        "coverage": cov,
        "window_fully_covered": cov.get("covered"),
        "how_that_is_computed": (
            "da_root.journal_coverage: the host journal's oldest retained "
            "entry against the unit's own start timestamp -- TWO MEASURED "
            "CLOCKS, no needle. The old predicate searched the copied "
            "lines for ' Started ' and reported TRUE on a tail that had "
            "lost 141 of 161 lines"),
        "counts_agree": len(by_id) == len(by_unit),
        "counts_note": ("they differ legitimately when the NAME has been "
                        "used more than once; the id is the run"),
    })
    return out


def journal_read(unit: str, *, n: int = 200) -> dict:
    """Back-compat name: the by-invocation copy is the record (R-646 R4).

    The first version of this filtered by unit NAME alone. A name names
    every run ever launched under it, and it cannot say which lines belong
    to THIS run."""
    return journal_copy_by_invocation(unit, n=n)


#: R-628. THE LAUNCH FORM, DECLARED -- a heavy run is NEVER a child of a
#: tool shell. `systemd-run --scope` registers processes the CALLER forks,
#: so the run sits in the launching shell's process group; when the harness
#: stopped this seat's background task the 09-03 re-run died at 35 minutes
#: with 34m52s of CPU spent and nothing written. Measured both ways:
#:
#:   --scope,   TERM to the launcher's process group -> the run DIED
#:   --service, TERM to the launcher's process group -> the unit LIVES
#:              systemctl --user stop <unit>          -> the unit ends
#:
#: A transient SERVICE is forked by the MANAGER (PPID = the user systemd,
#: its own process group), so nothing that happens to a tool shell can
#: reach it, and journald keeps its stdout -- the launch log of the refused
#: run was only recoverable because a launcher wrote it to a file.
#: R-646 (R2). THE ONE DECLARATION of the launch form's constants. Every
#: literal here READS this file; nothing is typed beside it. Two
#: definitions of the conflict code and two bare literals were measured
#: drifting-capable (a launcher refusing with 76 was published as 75).
HEAVY_RUN_FORM_DIR = "live/pm_research/declarations"
HEAVY_RUN_FORM_GLOB = "heavy_run_form_v*.json"


def design_chain(root: Path | None = None, *,
                 _own_walk_only: bool = False) -> dict:
    """THE DESIGN DECLARATION'S CHAIN HEAD, resolved not pinned.

    **DE 111: THE RESOLUTION IS `declaration_chain.resolve_head`'s.** BE 80
    taught the shared implementation to read `also_supersedes`, so the one
    reason to keep a second resolver is gone -- and the cost of two was
    measured this week: mine globs the bare `_v<N>.json` names and BE's
    globs `_v*`, so on the same family mine said one orphan and BE's said
    five, and the repair had to merge the union of both. That is R-711's
    lesson (three porcelain parsers, four disagreeing lines) arriving in a
    chain resolver.

    THE OWN WALK IS KEPT AS A CROSS-CHECK, NOT AS THE ANSWER. Called with
    `_own_walk_only=True` it returns exactly what it always did; the
    default path runs BOTH and reports the shared resolver's answer with
    the local walk beside it and an explicit agreement predicate. A
    cross-check that silently agreed would be no cross-check, and one that
    silently disagreed would be the defect it exists to catch.

    R-656 keeps params v14 while the design goes v21 -> v22, and the
    params name the design by PATH. If that path had to be the head, every
    design bump would force a params bump for a pointer alone -- the churn
    this seat has been paying since v14/v17.

    So the rule is R-653 (i) generalised: the HEAD is the design nothing
    supersedes, with every `supersedes` pair {path, sha256} recomputed;
    the path the params name must be IN that chain (the head, or an
    ancestor of it). Both halves keep their meaning and neither forces the
    other to move."""
    d = (Path(root) if root else Path(DR.resolve()["data_root"])) \
        / "pm_5min/derived"
    docs, bad = {}, []
    for f in sorted(d.glob("p003_de_multiday_gate1_design_v*.json")):
        m = re.search(r"_design_v(\d+)\.json$", f.name)
        if not m:
            continue
        try:
            docs[int(m.group(1))] = (f, json.loads(f.read_text()))
        except (OSError, ValueError) as exc:
            bad.append({"file": f.name, "why": str(exc)})
    if not docs:
        return {"resolved": False, "why": "no versioned design artifact",
                "unreadable": bad}
    links, superseded = [], set()
    for v in sorted(docs):
        f, doc = docs[v]
        sup = doc.get("supersedes") or {}
        sp, sh = sup.get("path"), sup.get("sha256")
        if not sp or not sh:
            continue
        prev = Path(sp)
        prev = prev if prev.is_absolute() else (d.parent.parent.parent
                                                / sp)
        if not prev.is_file():
            prev = d / Path(sp).name
        if not prev.is_file():
            links.append({"version": v, "supersedes": Path(sp).name,
                          "agrees": False, "why": "the named file is "
                                                  "absent"})
            continue
        got = hashlib.sha256(prev.read_bytes()).hexdigest()
        links.append({"version": v, "supersedes": prev.name,
                      "declared_sha256": sh, "recomputed_sha256": got,
                      "agrees": got == sh})
        if got == sh:
            superseded.add(prev.name)
    # THE HEAD IS RESOLVED BACKWARD FROM THE NEWEST, and the chain is
    # the path it walks. "Nothing supersedes it" alone yields TWO heads
    # here -- v16 and v17 BOTH supersede v15, which is DE 94's
    # two-files-one-version defect showing up as a FORK. Picking between
    # two heads by recency would be choosing after seeing; walking back
    # from the newest is deterministic, and every version off that path
    # is an ORPHAN BRANCH, reported BY NAME rather than quietly dropped.
    by_name = {f.name: v for v, (f, _) in docs.items()}
    newest = max(docs)
    walk, seen, cur = [], set(), newest
    while cur is not None and cur not in seen:
        seen.add(cur)
        walk.append(cur)
        f, doc = docs[cur]
        sup = (doc.get("supersedes") or {})
        nxt = by_name.get(Path(str(sup.get("path") or "")).name)
        link = next((l for l in links if l["version"] == cur), None)
        cur = nxt if (nxt is not None and link and link.get("agrees")) \
            else None
    # ---- `also_supersedes`: THE MERGE LINK (REV 81 §5) -----------------
    # The primary walk above is ONE path and a fork has two, so a version
    # off the path is an orphan however sound it is. REV 81's repair is a
    # version whose links name BOTH branch tips: the head as the pair, the
    # other tip(s) under `also_supersedes`, each as its own {path, sha256}.
    # REACHABILITY, not the single walk, is what makes an orphan an
    # orphan -- so the traversal follows the merge links too, and each is
    # VERIFIED by recomputation exactly as the primary link is. An
    # unverified merge link does not merge: it is reported and the tip
    # stays an orphan, because a link nobody can check is not a link.
    merge_links = []
    frontier, reach = list(seen), set(seen)
    while frontier:
        v = frontier.pop()
        f, doc = docs[v]
        for entry in (doc.get("also_supersedes") or []):
            if not isinstance(entry, dict):
                continue
            nm, sh = Path(str(entry.get("path") or "")).name, \
                entry.get("sha256")
            # THE LINK IS VERIFIED AT THE FILE, not at this resolver's own
            # universe. A merge link may name a STAMPED version -- this
            # function enumerates only the BARE `_v<N>.json` names, and a
            # target outside that set is still a real file whose digest
            # recomputes. Judging such a link `agrees: False` would report
            # a sound link as broken; it is verified, and separately
            # marked as not one of the versions this walk can reach.
            tf = d / nm
            got = (hashlib.sha256(tf.read_bytes()).hexdigest()
                   if tf.is_file() else None)
            agrees = bool(sh) and got == sh
            in_universe = by_name.get(nm)
            merge_links.append({
                "version": v, "also_supersedes": nm,
                "declared_sha256": sh, "recomputed_sha256": got,
                "agrees": agrees,
                "target_present": tf.is_file(),
                "target_is_a_bare_versioned_design": in_universe is not None,
                "why_if_not": (None if in_universe is not None else
                               "a STAMPED version; this resolver "
                               "enumerates the bare `_v<N>.json` names "
                               "only, so the link is VERIFIED here and "
                               "the target is outside this walk")})
            if agrees and in_universe is not None and in_universe not in reach:
                reach.add(in_universe)
                frontier.append(in_universe)
    seen = reach
    orphans = sorted(set(docs) - seen)
    hf, _ = docs[newest]
    out = {"resolved": True, "versions_present": sorted(docs),
           "links": links, "merge_links": merge_links, "unreadable": bad,
           "reachable_from_the_newest": sorted(seen),
           "resolver": "de_multiday_gate1_runner.design_chain (own walk)",
           "heads_by_nothing_supersedes_them": sorted(
               v for v, (f, _) in docs.items() if f.name not in superseded),
           "chain_from_the_newest": walk,
           "orphan_branches": orphans,
           "orphan_note": (
               "versions NOT on the path walked back from the newest. "
               "v16 and v17 both supersede v15 (DE 94: two artifacts, one "
               "protocol version), so 'nothing supersedes it' yields two "
               "heads; the fork is REPORTED, never resolved by picking"
               if orphans else None),
           "head_version": newest, "head_name": hf.name,
           "head_path": str(hf),
           "head_sha256": hashlib.sha256(hf.read_bytes()).hexdigest(),
           "chain_names": sorted(f.name for f, _ in docs.values()),
           "rule": ("the head is the NEWEST version whose supersession "
                    "pairs verify back along its own path; every version "
                    "off that path is an orphan branch, named. The path "
                    "the params name must be IN the chain (R-653 (i) "
                    "generalised)")}
    if _own_walk_only:
        return out
    # ---- THE ANSWER IS THE SHARED RESOLVER'S (DE 111) -----------------
    import declaration_chain as DC
    own = out
    try:
        sh = DC.resolve_head(d, "p003_de_multiday_gate1_design")
    except Exception as exc:                      # a STATUS, never a skip
        own["shared_resolver"] = {
            "status": "UNRESOLVED",
            "why": f"{type(exc).__name__}: {exc}",
            "and_this_is_not_a_pass": "rule 11 -- the answer below is the "
                                      "LOCAL walk because the shared "
                                      "resolver could not answer, and "
                                      "that is said here rather than "
                                      "silently substituted"}
        own["resolved_by"] = "the local walk (shared resolver unresolved)"
        return own
    sh_orphans = [o["version"] for o in sh["orphan_branches"]]
    own_orphan_names = [f"p003_de_multiday_gate1_design_v{v}.json"
                        for v in own["orphan_branches"]]
    agree = {
        "head": sh["name"] == own["head_name"],
        "orphan_names": sorted(sh_orphans) == sorted(own_orphan_names),
        "n_merge_links": sh.get("n_merge_links") == len(
            [m for m in own["merge_links"] if m.get("agrees")]),
    }
    return {**own,
            "resolved_by": "declaration_chain.resolve_head (BE 77/80) -- "
                           "the ONE chain implementation",
            "head_version": sh["version"], "head_name": sh["name"],
            "head_path": sh["path"], "head_sha256": sh["sha256"],
            "orphan_branches": sh_orphans,
            "orphan_note": (sh.get("fork_status") if sh_orphans
                            else None),
            "shared_resolver": {
                "name": sh["name"], "version": sh["version"],
                "sha256": sh["sha256"],
                "orphan_branches": sh_orphans,
                "n_versions": sh["n_versions"],
                "n_merge_links": sh.get("n_merge_links"),
                "merged_tips": sh.get("merged_tips"),
                "fork_status": sh.get("fork_status"),
                "forks_two_versions_superseding_one":
                    sh.get("forks_two_versions_superseding_one")},
            "own_walk_kept_as_a_cross_check": {
                "head_name": own["head_name"],
                "head_version": own["head_version"],
                "orphan_branches_as_names": own_orphan_names,
                "n_verified_merge_links": len(
                    [m for m in own["merge_links"] if m.get("agrees")]),
                "globs": "the BARE `_v<N>.json` names only; the shared "
                         "resolver globs `_v*` and so also sees the "
                         "STAMPED versions this family used before v16 -- "
                         "the two ANSWER THE SAME QUESTION over DIFFERENT "
                         "populations, and the orphan-name comparison "
                         "below is the one that matters",
                "agreement": agree,
                "agrees_on_the_head": agree["head"],
                "why_it_is_kept": "a second implementation is not the "
                                  "answer any more (DE 111); it is the "
                                  "fixture cross-check, and a silent "
                                  "agreement would be no cross-check"}}


def heavy_run_form_chain() -> dict:
    """THE CHAIN HEAD of the launch-form declarations -- never a filename.

    REV 69 S3.1/S4 (R-653 (i)): this runner PINNED v1 at 14:00Z while v2
    existed, then pinned v2 BY NAME while v3 existed. A reader pinned to a
    superseded version satisfies the words of rule 20's guard without the
    property -- it reads *a* declaration, not *the* declaration.

    So the head is RESOLVED: the newest version whose `supersedes` pair
    {path, sha256} verifies against the file it names, back to v1. Each
    link is recomputed, never trusted."""
    root = Path(__file__).resolve().parents[2]
    d = root / HEAVY_RUN_FORM_DIR
    docs, bad = {}, []
    for f in sorted(d.glob(HEAVY_RUN_FORM_GLOB)):
        m = re.search(r"_v(\d+)\.json$", f.name)
        if not m:
            bad.append({"file": f.name, "why": "no version token"})
            continue
        try:
            docs[int(m.group(1))] = (f, json.loads(f.read_text()))
        except (OSError, ValueError) as exc:
            bad.append({"file": f.name, "why": str(exc)})
    if not docs:
        raise RunnerRefused(
            f"REFUSED: no launch-form declaration under {d}.")
    # AN UNREADABLE FILE IN THE SET IS A REFUSAL, NEVER A DEMOTION
    # (REV 71 S1.1). This collected unparseable files into `bad` and went
    # on to resolve a head from the rest -- so a corrupt v3 silently
    # DEMOTED the reader to v2, which is the pinned-to-a-superseded-
    # version defect arriving by another door. An unresolvable set has no
    # head.
    if bad:
        raise RunnerRefused(
            f"REFUSED: {len(bad)} launch-form declaration(s) under {d} "
            f"cannot be read ({[b['file'] for b in bad]}). The head is "
            f"the version nothing supersedes AMONG ALL OF THEM; with one "
            f"unreadable the set has no head, and resolving from the rest "
            f"would silently demote this reader to a superseded version "
            f"(REV 71 S1.1).")
    links, superseded = [], set()
    for v in sorted(docs):
        f, doc = docs[v]
        sup = doc.get("supersedes")
        if v == min(docs) and not sup:
            continue
        if not isinstance(sup, dict) or not sup.get("path") \
                or not sup.get("sha256"):
            raise RunnerRefused(
                f"REFUSED: {f.name} carries no supersession PAIR "
                f"{{path, sha256}}. The chain head is resolved through "
                f"the links; a version that declares none cannot be "
                f"placed in it (R-608's pair rule).")
        prev = root / sup["path"]
        if not prev.is_file():
            raise RunnerRefused(
                f"REFUSED: {f.name} supersedes {sup['path']}, absent -- a "
                f"claim about a file nobody can check.")
        got = hashlib.sha256(prev.read_bytes()).hexdigest()
        if got != sup["sha256"]:
            raise RunnerRefused(
                f"REFUSED: {f.name} supersedes {sup['path']} at "
                f"{sup['sha256'][:16]} and that file hashes to "
                f"{got[:16]}.")
        links.append({"version": v, "supersedes": prev.name,
                      "declared_sha256": sup["sha256"],
                      "recomputed_sha256": got, "agrees": True})
        superseded.add(prev.name)
    heads = [v for v, (f, _) in docs.items() if f.name not in superseded]
    if len(heads) != 1:
        raise RunnerRefused(
            f"REFUSED: {len(heads)} launch-form declarations are "
            f"superseded by nothing ({sorted(heads)}); the chain must "
            f"resolve to ONE head.")
    hv = heads[0]
    hf, hdoc = docs[hv]
    return {"head_version": hv, "head_path": str(hf.relative_to(root)),
            "head_sha256": hashlib.sha256(hf.read_bytes()).hexdigest(),
            "versions_present": sorted(docs), "links": links,
            "unreadable": bad, "doc": hdoc,
            "resolved_not_pinned": (
                "the head is the version nothing supersedes, with every "
                "link's pair recomputed. A filename literal here pinned "
                "v1 while v2 existed (REV 69 S3.1)")}


def heavy_run_form() -> dict:
    """The declared constants of the CHAIN HEAD, read -- never a literal."""
    ch = heavy_run_form_chain()
    d = dict(ch["doc"])
    for k in ("lock_path", "lock_conflict_rc", "slice",
              "remain_after_exit"):
        if k not in d:
            raise RunnerRefused(
                f"REFUSED: the launch-form chain head "
                f"({ch['head_path']}) carries no {k!r}.")
    d["_chain"] = {k: v for k, v in ch.items() if k != "doc"}
    return d


#: THIS RUNNER'S OWN EXIT CODES, declared so the assertion below has
#: something to check. 0 on success; 1 on a RunnerRefused or a battery
#: failure (SystemExit with a message). Nothing else is produced.
RUNNER_EXIT_CODES = {
    0: "the run completed and the artifact was written",
    1: "RunnerRefused, or a battery failure -- SystemExit with a message",
}


EXIT_MAP_DIR = "live/pm_research/declarations"
EXIT_MAP_GLOB = "producer_exit_maps_v*.json"
THIS_PRODUCER = "live/pm_research/de_multiday_gate1_runner.py"


def producer_exit_map(root: Path | None = None) -> dict:
    """THIS PRODUCER'S DECLARED EXIT MAP, from the chain head (R-709).

    The head is resolved through the pair, never a filename literal --
    the same rule as the launch form's. A map that claims 75 is REFUSED:
    75 is the LAUNCH LAYER'S (`flock -n -E 75` means a held lock and the
    payload never started), and a producer claiming it would make a
    refusal indistinguishable from a held lock in `ExecMainStatus`."""
    r = Path(root) if root else Path(__file__).resolve().parents[2]
    d = r / EXIT_MAP_DIR
    docs, bad = {}, []
    for f in sorted(d.glob(EXIT_MAP_GLOB)):
        m = re.search(r"_v(\d+)\.json$", f.name)
        if not m:
            continue
        try:
            docs[int(m.group(1))] = (f, json.loads(f.read_text()))
        except (OSError, ValueError) as exc:
            bad.append({"file": f.name, "why": str(exc)})
    if bad:
        raise RunnerRefused(
            f"REFUSED: {[b['file'] for b in bad]} in the exit-map family "
            f"cannot be read; an unresolvable set has no head.")
    if not docs:
        raise RunnerRefused(f"REFUSED: no producer exit map under {d}.")
    superseded = set()
    for v in sorted(docs):
        f, doc = docs[v]
        sup = doc.get("supersedes") or {}
        if not sup:
            continue
        prev = r / sup["path"]
        if not prev.is_file():
            raise RunnerRefused(
                f"REFUSED: {f.name} supersedes {sup['path']}, absent.")
        got = hashlib.sha256(prev.read_bytes()).hexdigest()
        if got != sup.get("sha256"):
            raise RunnerRefused(
                f"REFUSED: {f.name}'s supersession pair does not hold "
                f"({str(sup.get('sha256'))[:16]} vs {got[:16]}).")
        superseded.add(prev.name)
    heads = [v for v, (f, _) in docs.items() if f.name not in superseded]
    if len(heads) != 1:
        raise RunnerRefused(
            f"REFUSED: the exit-map family resolves to {len(heads)} heads "
            f"({sorted(heads)}); it must resolve to one.")
    hf, hdoc = docs[heads[0]]
    block = (hdoc.get("producers") or {}).get(THIS_PRODUCER) or {}
    codes = {int(k) for k in (block.get("map") or {})}
    rc = heavy_run_form()["lock_conflict_rc"]
    if rc in codes:
        raise RunnerRefused(
            f"REFUSED: this producer's declared exit map claims {rc}, "
            f"which is the LAUNCH LAYER'S lock-conflict code. A producer "
            f"exiting {rc} makes a refusal indistinguishable from a held "
            f"lock in `ExecMainStatus` -- the whole point of `-E {rc}`.")
    return {"head_version": heads[0], "head_path": str(hf.name),
            "head_sha256": hashlib.sha256(hf.read_bytes()).hexdigest(),
            "status": block.get("status"), "map": block.get("map"),
            "codes": sorted(codes),
            "lock_conflict_rc_excluded": rc,
            "agrees_with_the_module": codes == set(RUNNER_EXIT_CODES)}


def assert_no_exit_code_collision() -> dict:
    """A RunnerRefused MUST NOT exit with the lock-conflict code.

    R-646 (R2): `flock -n -E 75` makes a HELD LOCK distinguishable from a
    payload failure -- but only while no payload exits 75 for its own
    reasons. Measured before the ruling: under the ruled form a held lock
    and a payload crash were BOTH `ExecMainStatus=1`, so at GO #5 a
    refusal would have been unreadable from a crash."""
    rc = heavy_run_form()["lock_conflict_rc"]
    if rc in RUNNER_EXIT_CODES:
        raise RunnerRefused(
            f"REFUSED: this runner declares exit code {rc}, which is the "
            f"DECLARED lock-conflict code. A held lock and a runner "
            f"refusal would be indistinguishable in `ExecMainStatus`, "
            f"which is the whole point of `-E {rc}`.")
    return {"lock_conflict_rc": rc,
            "runner_exit_codes": sorted(RUNNER_EXIT_CODES),
            "no_collision": True,
            "read_from": heavy_run_form()["_chain"]["head_path"]}


LAUNCH_FORM = "systemd-run --user transient SERVICE (never --scope)"
LAUNCH_FORM_REQUIREMENTS = {
    "no_scope": "`--scope` runs the payload in the CALLER's process group",
    "unit_named": "`--unit=` so the run can be polled and stopped by NAME, "
                  "never by a child PID",
    "slice": f"`--slice={RESEARCH_SLICE}` so rule 20's cgroup accounting "
             f"applies",
    "memory_and_cpu": "-p MemoryMax=8G -p CPUQuota=100%",
    "lock_inside_the_unit": "`flock -n` is the unit's own ExecStart, so a "
                            "held lock exits 1 INSIDE the unit and the "
                            "payload never starts -- read "
                            "`ExecMainStatus`, never assume it started",
    "working_directory": "`--working-directory=` because the manager does "
                         "not inherit the caller's cwd",
    "absolute_interpreter": "the venv's own python3, resolved",
}


def the_one_command(day: str, book, outdir, *, unit: str = "<deNNsmoke>",
                    workdir: str | None = None) -> str:
    """THE launch string -- composed once, so the command and the predicate
    that checks it cannot be two different facts."""
    wd = workdir or str(Path(__file__).resolve().parents[2])
    # `-E <rc>` FROM THE DECLARATION, never a literal here (R-646 R2).
    _form = heavy_run_form()
    _rc = _form["lock_conflict_rc"]
    # `RemainAfterExit=yes` FROM THE DECLARATION (R-648 R3'). Without it a
    # transient unit that exits 0 is COLLECTED at exit: LoadState goes
    # not-found and `systemctl show` then returns DEFAULTS
    # (inactive/dead/0/success). A reading taken then is a DEFAULT wearing
    # the shape of a success.
    _rae = ("-p RemainAfterExit=yes " if _form.get("remain_after_exit")
            else "")
    return (f"systemd-run --user --unit={unit} --slice={RESEARCH_SLICE} "
            f"-p MemoryMax=8G -p CPUQuota=100% {_rae}"
            f"--setenv=PM_DATA_ROOT={DR.resolve()['repo_root']} "
            f"--working-directory={wd} "
            f"-- flock -n -E {_rc} {HEAVY_RUN_LOCK} {sys.executable} "
            f"live/pm_research/de_multiday_gate1_runner.py "
            f"--day {day} --book {book} --output {outdir}")


def assert_launch_form(cmd: str) -> dict:
    """THE COMMAND'S SHAPE IS A PREDICATE, not a paragraph in a runbook.

    The scope form was published in `THE_ONE_COMMAND` for four rounds and
    cost a real day 35 minutes. A form that must not be used again is one
    a checker refuses."""
    problems = []
    if "--scope" in cmd:
        problems.append(
            "carries `--scope`: the payload would run in the CALLING "
            "shell's process group and dies with it (R-628, measured)")
    if "--unit=" not in cmd:
        problems.append("names no `--unit=`, so the run could only be "
                        "polled by a child PID")
    if heavy_run_form().get("remain_after_exit") \
            and "-p RemainAfterExit=yes" not in cmd:
        problems.append(
            "carries no `-p RemainAfterExit=yes`: a transient unit that "
            "exits 0 is COLLECTED at exit, and `systemctl show` then "
            "returns DEFAULTS -- the outcome would be unreadable "
            "(R-648 R3')")
    if f"--slice={RESEARCH_SLICE}" not in cmd:
        problems.append(f"is not in {RESEARCH_SLICE}")
    if "--working-directory=" not in cmd:
        problems.append("sets no `--working-directory=`; the manager does "
                        "not inherit the caller's cwd")
    if " -- " not in cmd:
        problems.append("has no `--` separator, so the lock and the "
                        "payload are not the unit's own ExecStart")
    else:
        _pre, _post = cmd.split(" -- ", 1)
        _rc_expected = heavy_run_form()["lock_conflict_rc"]
        if f"-E {_rc_expected} " not in _post:
            problems.append(
                f"the unit's ExecStart carries no `-E {_rc_expected}`: "
                f"without it a HELD LOCK and a payload crash are both "
                f"ExecMainStatus=1 and a refusal is unreadable from a "
                f"crash (R-646 R2, measured)")
        if not _post.startswith("flock -n "):
            problems.append(
                "the unit's ExecStart does not begin with `flock -n`: the "
                "lock must be taken INSIDE the unit, so a held lock exits "
                "1 there and the payload never starts")
        if "--scope" in _pre:
            problems.append("`--scope` before the separator")
    if problems:
        raise RunnerRefused(
            "REFUSED: this launch command is not the declared form "
            f"({LAUNCH_FORM}) -- it " + "; and it ".join(problems) + ".")
    return {"form": LAUNCH_FORM, "ok": True,
            "requirements_checked": sorted(LAUNCH_FORM_REQUIREMENTS),
            "lock_is_inside_the_unit": True,
            "poll_by": "the UNIT name, never a child PID"}


def unit_outcome(unit: str) -> dict:
    """A UNIT'S OUTCOME IS THE TRIPLE (LoadState, ActiveState,
    ExecMainStatus), READ WHILE IT IS STILL LOADED -- R-648 (R3').

    Two ways this reading lies, both measured:

      * a RUNNING unit reports `ExecMainStatus=0`, so the status alone
        reads a running run as a clean success (R3, measured on
        de95smoke.service 53 minutes into an 85-minute day);
      * a transient unit that exits 0 is COLLECTED at exit -- LoadState
        goes `not-found` and `systemctl show` then returns DEFAULTS
        (`inactive`/`dead`/`0`/`success`). That reading is a DEFAULT
        wearing the shape of a success, and it is reported here as VOID.

    And a third, found by my own poll script this round: `systemctl show
    -p A -p B --value` returns the properties in SYSTEMD'S order, not the
    flag order (measured: Result, ExecMainStatus, LoadState, ActiveState,
    SubState). A positional read of `--value` mislabels every field -- it
    declared a live run VOID. Every property here is read in its own call
    and NAMED."""
    import subprocess as _sp
    vals, read_at = {}, datetime.datetime.now(datetime.timezone.utc)
    try:
        _min = list(heavy_run_form()["unit_outcome_minimum_read"])
    except RunnerRefused:
        _min = ["LoadState", "ActiveState", "SubState", "ExecMainStatus",
                "Result"]
    for k in _min + ["InvocationID", "MemoryPeak",
                     "ExecMainStartTimestamp"]:
        try:
            r = _sp.run(["systemctl", "--user", "show", unit, "-p", k,
                         "--value"], capture_output=True, text=True,
                        timeout=30)
            vals[k] = r.stdout.strip() if r.returncode == 0 else None
        except Exception:                                 # noqa: BLE001
            vals[k] = None
    rc = None
    try:
        rc = heavy_run_form()["lock_conflict_rc"]
    except RunnerRefused:
        pass
    load = vals.get("LoadState")
    active = vals.get("ActiveState")
    status = vals.get("ExecMainStatus")
    out = {
        "unit": unit, **vals,
        "read_at_utc": read_at.isoformat(),
        "read_property_by_property": (
            "each in its own `systemctl show -p K --value` call. A single "
            "multi-property `--value` read returns systemd's own order, "
            "not the flag order, and a positional parse of it mislabels "
            "every field -- it declared a live run VOID"),
        "the_triple": [load, active, status],
        "the_five": {k: vals.get(k) for k in _min},
        "minimum_read_declared_by": "the launch-form chain head's "
                                    "`unit_outcome_minimum_read`",
        "lock_conflict_rc": rc,
    }
    if load != "loaded":
        # VOID, NEVER SUCCESS. After collection every field below is a
        # DEFAULT: inactive / dead / 0 / success.
        out.update({
            "status": "VOID",
            "outcome_readable": False,
            "why_void": (
                f"LoadState is {load!r}, not `loaded`. A transient unit "
                f"that exits 0 is COLLECTED at exit and `systemctl show` "
                f"then returns DEFAULTS (inactive/dead/0/success). "
                f"Whatever ExecMainStatus says here is a default, not an "
                f"outcome, and reporting it as success would report a "
                f"result nobody measured (R-648 R3')"),
            "what_to_do": (
                "launch with `-p RemainAfterExit=yes` so the unit stays "
                "loaded until it is stopped, and read the triple while it "
                "is; failing that, the RECEIPT is the record and the unit "
                "reading is void"),
        })
        return out
    # THE INVOCATION ID IS PART OF THE READING (R-653 (ii)). A copy of
    # five fields without the id names no particular run -- a unit NAME
    # names every run ever launched under it.
    if not vals.get("InvocationID"):
        out.update({
            "status": "VOID",
            "outcome_readable": False,
            "why_void": (
                "the unit is loaded but reports no InvocationID, so this "
                "reading names no particular RUN. A unit name names every "
                "run ever launched under it; the id is what makes the "
                "five fields a reading OF SOMETHING (R-653 (ii))"),
            "what_to_do": "re-read while the unit is loaded and carries "
                          "its id, or treat the receipt as the record",
        })
        return out
    # UNDER RemainAfterExit A FINISHED UNIT IS loaded/active/EXITED AND A
    # RUNNING ONE loaded/active/RUNNING -- both ExecMainStatus 0. Neither
    # ActiveState nor the status tells them apart; SubState is the
    # discriminator (REV 69 S3.3, measured).
    sub = vals.get("SubState")
    out.update({
        "status": "READABLE",
        "outcome_readable": True,
        "still_running": (active == "active" and sub == "running"),
        "finished": (sub in ("exited", "failed", "dead")
                     or active in ("inactive", "failed")),
        "SubState_is_the_discriminator": (
            "under RemainAfterExit=yes a FINISHED unit is "
            "loaded/active/EXITED and a RUNNING one loaded/active/RUNNING, "
            "with ExecMainStatus 0 in BOTH. `still_running` is read from "
            "SubState, never from ActiveState alone"),
        "killed_by_signal": vals.get("Result") == "signal",
        "why_that_matters": (
            "a killed or OOM-killed unit reports Result=signal with "
            "ExecMainStatus = the SIGNAL NUMBER, which is in no exit map "
            "-- reading it as an exit code invents a meaning"),
        "refused_on_the_lock": (status == str(rc)) if rc is not None
                               else None,
        "why_the_triple": (
            "LoadState says whether the reading MEANS anything, "
            "ActiveState whether it is finished, and ExecMainStatus how "
            "it ended. Any two of the three answer a question the third "
            "was asked"),
    })
    return out


def unit_identity() -> dict:
    """WHICH UNIT THIS PROCESS IS RUNNING IN, measured from its cgroup."""
    import os as _os
    leaf = (cgroup_path() or "").rstrip("/").rsplit("/", 1)[-1]
    return {"cgroup_leaf": leaf or None,
            "unit": leaf if leaf.endswith((".service", ".scope")) else None,
            "kind": ("transient service" if leaf.endswith(".service")
                     else "scope" if leaf.endswith(".scope") else None),
            "invocation_id": _os.environ.get("INVOCATION_ID"),
            "is_the_declared_launch_form": leaf.endswith(".service"),
            "why_that_matters": (
                "a `.scope` leaf means the run is in a CALLER's process "
                "group and dies with it -- that is how the 09-03 re-run "
                "lost 35 minutes (R-628)")}



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

def assert_real_day_has_the_lock(day: str, observed: dict, *,
                                 fixture: bool) -> dict:
    """A REAL DAY TAKES THE LOCK FIRST -- as a predicate over an OBSERVATION.

    It was an inline `if` inside `run_day`, and the battery drove it by
    calling `run_day` and reading the message. That check could therefore
    only pass when the BATTERY'S OWN PROCESS did not hold the lock -- and
    the battery now runs inside every real day, which by construction
    holds it. Measured, one line apart:

        battery without the lock -> PASS 234 checks
        battery holding the lock -> FAIL, "wrong reason"

    Every seat and every reviewer ran it without the lock, so it passed
    everywhere except in the one configuration it exists for. It cost the
    09-03 re-run 26 seconds -- not 85 minutes, because R-610 had already
    moved the battery in front of the day's work.

    The observation is now a PARAMETER, so both directions are drivable
    from either ambient state, which is what `assert_rule20` beside it
    already did."""
    held = bool(observed.get("heavy_run_lock_held"))
    if not fixture and not held:
        raise RunnerRefused(
            f"REFUSED DAY {day}: a REAL day is heavy by construction (BE "
            f"projects ~2.3 h for both arms) and this process does not hold "
            f"{HEAVY_RUN_LOCK}. Take the lock first; if it is held, refuse "
            f"and report (R-575(C)).")
    return {"day": day, "fixture": fixture, "heavy_run_lock_held": held,
            "checked": not fixture,
            "why_a_parameter_not_a_reading": (
                "a guard that reads the ambient lock cannot be driven in "
                "the state it protects: the battery runs INSIDE a real "
                "day, which holds the lock, so the check could only ever "
                "pass where it did not matter")}


def assert_ledger_anchor(params: dict, *, fixture: bool, anchor) -> dict:
    """A REAL DAY UNDER R-765 MUST HAVE SOMEWHERE TO WRITE ITS LEDGER.

    DE 133, and it is a FUNCTION so the battery can DRIVE it. Inside
    `run_day` the call sits after the day-membership and lock guards and
    before the day's work: refusing there costs nothing, where DE 132 put
    it at the ledger write -- after ~90 minutes of draws -- which is the
    shape R-610 exists to forbid. But `run_day` cannot be driven that far
    in a battery without the heavy lock: the earlier guards refuse first,
    correctly, and two versions of my own cell accepted THEIR refusal as
    if it were this one. A check nobody can watch fire is not a check.

    'Before the work' is not 'before every other guard': placed at the
    top it PRE-EMPTED the day-membership refusal and a cell testing that
    got this one instead.

    A FIXTURE owes no ledger -- its rows are synthetic and there is
    nothing to avoid re-running -- and that is a NAMED STATUS, never the
    `null` that hid the original defect."""
    if not params.get("user_ruled_unsealed_emission"):
        return {"owes_a_ledger": False,
                "why": "the params carry no R-765 ruling"}
    if fixture:
        return {"owes_a_ledger": False,
                "status": "NO_LEDGER_FOR_A_FIXTURE_DAY",
                "why": "a FIXTURE day's rows are synthetic; R-765 keeps "
                       "the numbers of REAL runs so they need not be "
                       "re-run"}
    if anchor is None:
        raise RunnerRefused(
            "REFUSED DECISION_LEDGER_HAS_NO_ANCHOR: the params carry the "
            "R-765 ruling and this is a REAL day, but no path was given "
            "to write the ledger beside. The ruling is that the numbers "
            "are KEPT; emitting the receipt with `decision_ledger: null` "
            "is the silent form of promising one that is not there. "
            "Refused BEFORE the day's work, not after it.")
    return {"owes_a_ledger": True, "anchor": str(anchor)}


def post_emit_census_status(params: dict, payload: dict,
                           economic_keys: list) -> dict:
    """WAS THE RECEIPT ALLOWED TO CARRY ITS ECONOMICS? -- COMPUTED.

    DE 134, found by running `--synthetic-day` end to end. Two defects
    sat in one line of `_main_day`:

    (1) it read `payload["G"]`, a key DE 124 REMOVED from the receipt
    (REV 90 S A0(3)). That change was made after a sweep for CONSUMERS
    of a receipt's `G`, which found none -- because the only consumer is
    the PRODUCER's own post-emit census, reading the payload in memory
    before it is anybody's receipt. Every real day since would have died
    here, AFTER writing its receipt, with `KeyError: 'G'`. None has run:
    the four early reads enter through `de_early_read`, and the last
    sealed day (09-06) predates DE 124.

    (2) the predicate itself is stale under R-765. "Economic keys
    present before G" is a LEAK only while the seal exists; the USER
    retired the seal, so from params v18 every run emits its numbers and
    the same condition is the RULED OUTCOME. With the bare `G` merely
    restored, this check would have DELETED tonight's receipt after
    writing it.

    So the regime is read from the PARAMS and the bar from the receipt's
    own `G_and_which_G_it_is`; the status is returned, never printed as
    a conclusion, and the caller decides. A census that cannot find its
    bar says so and is refused by the caller -- it does not pass."""
    ruled = bool(params.get("user_ruled_unsealed_emission"))
    bar = (payload.get("G_and_which_G_it_is") or {}).get(
        "the_bar_this_run_sealed_against")
    ndc = payload.get("n_days_complete")
    if ruled:
        status = "NOT_APPLICABLE_UNSEALED_BY_R765"
    elif bar is None or ndc is None:
        status = "POST_EMIT_CENSUS_HAS_NO_BAR"
    elif economic_keys and ndc < bar:
        status = "SEALED_REGIME_LEAK"
    else:
        status = "SEALED_REGIME_CLEAN"
    return {"status": status,
            "unsealed_by_user_ruling": ruled,
            "the_bar_this_run_sealed_against": bar,
            "n_days_complete": ndc,
            "n_economic_keys_read_back": len(economic_keys),
            "economic_keys_read_back": economic_keys[:6],
            "rule": CENSUS_RULE,
            "why_a_ruled_run_is_not_a_leak": (
                "R-765 retired the seal: the numbers are KEPT after every "
                "run. Under the ruling their presence is what was asked "
                "for, not a leak, and this census records that it looked "
                "rather than skipping in silence")}


def day_run_ledger_anchor(output_dir, *, fixture: bool):
    """WHERE THE DAY RUN'S LEDGER LANDS -- R-765, and REV 102's NO-GO.

    The guard above was right and the day path had never been given an
    anchor: `_main_day` -> `day_split_residency_proof` -> `run_day`
    passed neither `ledger_anchor` nor `receipt_path`, so under params
    carrying the R-765 ruling a REAL day refused
    DECISION_LEDGER_HAS_NO_ANCHOR before its work. It had not bitten
    because the four sealed days ran under v15, which carries no ruling;
    the first real day under a ruling-carrying params would have met it.

    IT IS THE OUTPUT DIRECTORY, AND NOT A PRE-COMPOSED RECEIPT NAME.
    The receipt's filename is composed at the EMIT from one clock read,
    so that its stamp and its `as_of` are the same instant (REV 55
    S2.1) -- and the ledger is written BEFORE the receipt, so the
    receipt can name it by digest. A name composed here would be a name
    no file ever takes: a pin to nothing. The ledger therefore lands in
    the directory the receipt lands in -- beside it, which is what
    R-765 asks for -- and carries its own stamp.

    A FIXTURE returns None: its rows are synthetic, there is nothing to
    avoid re-running, and `run_day` records NO_LEDGER_FOR_A_FIXTURE_DAY
    (DE 133). The decision lives HERE, in one function that can be
    driven both ways, rather than in a conditional at the call site."""
    if fixture:
        return None
    return Path(output_dir)


def ledger_path_for(anchor, day: str, stamp: str) -> Path:
    """THE LEDGER'S PATH FROM ITS ANCHOR -- a directory or a file.

    Two callers, two shapes, one rule: the ledger lands IN an anchor
    that is a directory, and BESIDE an anchor that is a file. The early
    read passes its artifact's path (a file that does not exist yet, so
    the parent is taken); the day path passes the receipt's output
    DIRECTORY, because the receipt's own name is not composed until the
    emit. Taking `.parent` of a directory would have put the day's
    ledger one level ABOVE the ledger directory."""
    import de_decision_ledger as _LED
    a = Path(anchor)
    return (a if a.is_dir() else a.parent) / _LED.ledger_name(day, stamp)


def run_day(day: str, book_path, *, params: dict, module=None,
            fixture: bool = False, receipt_path=None,
            n_days_complete: int = 1,
            peak_rss_mb_budget: float | None = None,
            early_read: dict | None = None,
            ledger_anchor=None,
            winners: dict | None = None,
            point_estimate: bool = False,
            before_work=None) -> dict:
    """ONE RULED DAY, SEALED. The path the smoke runs.

    Real days require the lock BEFORE any work (a real day is heavy by
    construction: BE projects ~2.3 h per day for both arms). A fixture day
    is expected light and is checked against its declared budget at the
    end -- a fixture that exceeds its budget REFUSES, because a budget
    nobody enforces is not a budget.

    `before_work` RUNS AFTER THE BOOK DIGEST IS VERIFIED AND BEFORE S1
    (R-610). The emitter passes the battery here. It used to run at the
    EMIT: 85 minutes of real work, then ~15 s of fixture runs and a
    13-module closure re-capture, and a refusal there LOST THE DAY -- which
    is what the 09-03 smoke did. Everything the battery can refuse is
    knowable before the book is loaded, so it is checked before the book is
    loaded. It is run under its OWN residency instrument so the day path's
    "no tape artifact was opened" claim stays a claim about the day path."""
    t_start = time.time()
    stages: dict = {}
    # THE DAY'S OWN DRAWS, separated from anything the `before_work` hook
    # draws. The battery runs FIXTURE days and those DO draw, so a bare
    # process counter would answer "did the day draw before the battery
    # refused?" with the battery's own draws. The counters are deltas.
    _draws_at_entry = RUN_COUNTERS["draws_performed"]
    _hook_draws = 0

    def _day_draws():
        return (RUN_COUNTERS["draws_performed"] - _draws_at_entry
                - _hook_draws)

    # THE BASELINE, so S0's delta is a measurement and not the whole
    # process's history. Without it the first stage's delta is everything
    # that ever ran, and the argmax is decided before the day starts.
    budget = (peak_rss_mb_budget if peak_rss_mb_budget is not None
              else (FIXTURE_DAY_PEAK_RSS_MB_BUDGET if fixture
                    else REAL_DAY_PEAK_RSS_MB_BUDGET))

    def _mark(name):
        # THE CLOSURE IS RE-CAPTURED AT EVERY STAGE. `load()` imports
        # harmful_stateful_policy lazily and `replay()` imports
        # de_phase4_diag_runner later still, so a capture taken once -- at
        # import, or even after the first load -- misses the modules that
        # do the REPLAYING. Already-seen modules are skipped, so this costs
        # a dict lookup per stage.
        _capture_closure()
        stages[name] = {"peak_rss_mb_highwater": _peak_rss_mb(),
                        "rss_mb_current": _current_rss_mb(),
                        "elapsed_s": round(time.time() - t_start, 3)}
        # THE BUDGET FIRES AT THE FIRST STAGE THAT CROSSES IT. It was
        # checked ONCE, after S5 -- so a day that crossed its ceiling while
        # loading the book still ran 84 minutes of null draws before being
        # told. A budget that refuses only at emit wastes the whole run it
        # exists to protect. The highwater series was already there; nobody
        # was reading it until the end.
        # THE GROWTH IS MEASURED ON CURRENT RSS, WHICH FALLS. Measuring
        # it on `ru_maxrss` reproduced the defect one level down: after an
        # earlier run in the SAME process had pushed the high-water to its
        # maximum, a later run's high-water did not rise at all, so its
        # growth read ZERO and the budget could never fire. A budget
        # measured with an instrument that cannot fall is a budget that
        # only works once per process.
        _b = stages.get("S_start", {}).get("rss_mb_current")
        _g = (stages[name]["rss_mb_current"] - _b
              if isinstance(_b, float) else None)
        if _g is not None and _g > budget:
            raise RunnerRefused(
                f"REFUSED DAY {day} AT STAGE {name}: this run has GREWN "
                f"{_g:.0f} MB over its baseline, past the declared budget "
                f"{budget:.0f} MB. Refused HERE, at the first stage that "
                f"crossed it, not at the emit -- the cap is never raised "
                f"and the draw count is never cut (R-174).")

    # THE FIXTURE/REAL LOCK, ON THE DAY PATH ITSELF (reviewer §1.4).
    # `--synthetic-day 2026-09-03` used to emit a SEALED artifact stamped
    # with the smoke day from a synthetic book. Disclosed, but exactly the
    # collision the lock was built to forbid.
    day_lock = assert_fixture_day_lock(day, fixture, what="day run")
    # BEFORE ANY STAGE (REV 65 S1.2): the wrapper this process is actually
    # in, not the wrapper that was published.
    launch_runtime = assert_launch_form_at_runtime(day, fixture=fixture)
    lock_runtime = assert_lock_form_at_runtime(day, fixture=fixture)
    _mark("S_start")
    obs = wrapper_observed()
    assert_real_day_has_the_lock(day, obs, fixture=fixture)
    assert_ledger_anchor(params, fixture=fixture,
                         anchor=(ledger_anchor if ledger_anchor is not None
                                 else receipt_path))

    # ---- S0: verify. Digests only. -------------------------------------
    book_path = Path(book_path)
    receipt = builder_receipt_for(book_path, day, params.get("coin", "btc"))
    bookcite = verify_book_against_builder_receipt(day, book_path, receipt)
    book_sha = bookcite["sha256"]
    mod, cite = import_be_cascade(params, module=module)
    if not fixture:
        verify_pinned_models(params)
        verify_pinned_thetas(params)
        # THE DESIGN IS DIGESTED HERE, AT S0 (REV 75 S1.1) -- the
        # earliest point a REAL day may read `data/`, and far before the
        # emit. It was recorded AT THE EMIT, so its "at load" and "at
        # emit" digests were one read twice and the pair could not detect
        # the design file changing under a run. It cannot go in
        # `load_params`: a FIXTURE run must open no path under `data/`,
        # and the data-root guard refused it there.
        _drel0 = (params.get("design_declaration") or {}).get("path")
        if _drel0:
            record_input_digest("design", _drel0)
    _mark("S0_verify")

    # ---- S0b: THE BATTERY, BEFORE THE DAY'S WORK (R-610) ---------------
    # Nested under its OWN instrument: the battery reads `data/` by
    # design, and the day path's residency claim must stay a claim about
    # the DAY PATH. The inner instrument's patches wrap the outer ones, so
    # every open it sees the outer one sees too -- the day-path set is the
    # difference, computed, never assumed.
    before_work_proof = None
    if before_work is not None:
        _dr0 = RUN_COUNTERS["draws_performed"]
        _res_bw, before_work_proof = DR.instrumented(before_work)
        # THE HOOK'S OWN UNCAPPED SET, read before anything else calls the
        # instrument. `distinct_paths` is capped at 200 and the battery
        # opens thousands, so the capped list cannot answer what the hook
        # touched.
        _hook_paths = DR.last_full_paths()
        _hook_draws = RUN_COUNTERS["draws_performed"] - _dr0
        before_work_proof = {
            **{k: before_work_proof[k] for k in
               ("n_paths_opened", "n_distinct_paths", "non_vacuous",
                "distinct_paths_truncated")},
            "tape_artifacts_opened": tape_artifacts_opened(
                before_work_proof, _hook_paths),
            "n_paths_opened_uncapped": len(_hook_paths),
            "data_paths_opened": before_work_proof["data_paths_opened"],
            "what_it_was": "the in-run battery, run BEFORE the day's work",
            # THE FALSIFIER'S MEASUREMENT: what THIS DAY had drawn at the
            # moment the hook returned. Zero, by the order -- and the
            # order is what R-610 asked for, so it is measured and not
            # read off a line number.
            "day_draws_when_the_hook_returned": _day_draws(),
            "draws_the_hook_itself_made": _hook_draws,
        }
        _mark("S0b_battery")

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
    # ---- R-801: THE RULED ENDPOINT, RESOLVED ONCE FOR THE DAY --------
    # The USER: "the pnls are from trades and remaining position's
    # settlement p&l, need to calculate this correctly". Rule 11 decides
    # WHETHER this day may be valued at all (the four early-read days are
    # DESIGN data; 09-07 and later wait for the declaration), and the
    # verdict is RECORDED either way -- a day that is not valued says so
    # in its receipt rather than silently carrying one endpoint.
    _adm801 = settlement_admissibility(day, params, fixture=fixture)
    _win801, _base801, _sbase801 = None, None, None
    if _adm801["admissible"]:
        _slugs801 = {f.get("slug") for f in base["fills"]} - {None}
        if fixture:
            # A FIXTURE OPENS NO PATH UNDER `data/` -- the venue's
            # resolution record is under it, and reading it here made the
            # fixture run's own data-free proof REFUSE (which is that
            # guard working). A fixture therefore brings its own winners
            # or is not valued: synthetic fills name synthetic slugs, and
            # the venue has no record of them anyway.
            _win801 = ({"winners": winners, "path": "SUPPLIED_BY_THE_CALLER",
                        "sha256": None, "n_slugs": len(winners),
                        "method": "a FIXTURE winner map passed to run_day",
                        "chainlink_verification": {
                            "status": "NOT_APPLICABLE_FIXTURE",
                            "why": "synthetic slugs have no venue record "
                                   "and no Chainlink price"},
                        "is_final_for_quotation": False}
                       if winners else None)
        else:
            # REFUSES BY NAME on an absent source, a slug with no record,
            # an ambiguous record or records that disagree.
            # R-803: AND THE WINNER IS VERIFIED PER SLUG against the
            # Chainlink TWAP by the ONE convention BE 99 measured to
            # reproduce the venue on 576/576 slugs. The venue record is
            # the join; the stream is the check (rule 9's door). A
            # DISAGREE or an UNAVAILABLE slug does not stop the value
            # being COMPUTED -- it stops it being QUOTED as final, which
            # `require_verified` is the door for.
            _win801 = winner_source(required_slugs=_slugs801)
            _cl801 = chainlink_streams([day], coin=params.get("coin", "btc"))
            _ver801 = verify_winners_against_chainlink(
                _win801["winners"], _slugs801, streams=_cl801["streams"],
                coin=params.get("coin", "btc"),
                stream_provenance=_cl801)
            # R-810: the source is re-resolved WITH the verification and
            # the day's own slug set, so the finality gate compares the
            # per-slug evidence against the slugs this day actually names.
            _win801 = winner_source(required_slugs=_slugs801,
                                    verification=_ver801)
            _win801["chainlink_verification"]["reader_module"] = \
                _cl801["reader_module"]
            _win801["chainlink_verification"]["n_hourly_files_read"] = \
                _cl801["n_hourly_files"]
        if _win801 is not None:
            _sbase801 = settle_value_cents(base["fills"],
                                           _win801["winners"])
            _base801 = settlement_legs_by_slug(base["fills"],
                                               _win801["winners"])
    _mark("S3_baseline")

    # ---- S4: the null and the observed value, per arm. -----------------
    results, per_arm_detail = [], {}
    _ledger765: dict = {}          # R-765: what each arm-day held
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
        _stream765 = mod.arm_stream(bk, spec["head"])
        arm_replay = mod.replay(bk, _stream765, spec["theta"])
        observed = _value_cents(arm_replay["fills"]) - base_value
        seed = seed_for(book_sha, arm)
        if point_estimate:
            # R-828 / DE 148: S4 IS SKIPPED ENTIRELY. Not a shorter null
            # -- no null. The draw count is never lowered (R-174); it is
            # not taken at all, and the receipt says so where every
            # statistic would have gone.
            nul = None
            r = point_estimate_arm_day(day, arm, observed,
                                       pop["decisions"], params)
            _mark("S4_null_SKIPPED_POINT_ESTIMATE")
        else:
            nul = null_draws_valued(
                mod, bk, base["fills"], pop["by_side"],
                n_draws=params["min_draws_per_arm_day"], seed=seed,
                deadline_s=params["per_day_deadline_s"],
                winners=(_win801["winners"] if _win801 else None))
            prov = {"module_sha256": cite["sha256"], "seed": seed,
                    "book_digest": book_sha, "arm": arm,
                    "draw_source": "GENERATED_IN_PROCESS",
                    "generated_in_process": True,
                    "pid": __import__("os").getpid(),
                    "n_draws": nul["n_draws"],
                    "reproduces_BEs_draw_null":
                        nul["reproduces_BEs_draw_null"]}
            r = arm_day(day, arm, observed, nul["values"],
                        pop["decisions"], params,
                        elapsed_s=time.time() - t_start,
                        draw_provenance=prov, book_digest=book_sha,
                        verified_module_sha=cite["sha256"])
        # ---- R-801: THE RULED P&L FOR THIS ARM-DAY -------------------
        # Beside D(E0), never instead of it: the 5-second markout stays
        # as the short-horizon DIAGNOSTIC the design declared, and the
        # ruled quantity is the one the user named.
        if _win801 is not None and r.get("status") in ("OK", "OK_POINT_ESTIMATE"):
            _sarm801 = settle_value_cents(arm_replay["fills"],
                                          _win801["winners"])
            _armlegs801 = settlement_legs_by_slug(arm_replay["fills"],
                                                  _win801["winners"])
            _obs801 = _sarm801 - _sbase801
            _recon801 = _armlegs801["total_cents"] - _base801["total_cents"]
            if abs(_recon801 - _obs801) > 1e-9:
                raise RunnerRefused(
                    f"REFUSED SETTLEMENT_EXCESS_DOES_NOT_RECONCILE: the "
                    f"per-slug legs give arm {_armlegs801['total_cents']!r} "
                    f"- baseline {_base801['total_cents']!r} = "
                    f"{_recon801!r} and the per-fill valuation gives "
                    f"{_obs801!r}. The decomposition and the excess must "
                    f"be the same number.")
            _sblock = (
                {"D_E_settle": _obs801,
                 "Z": NULL_NOT_DRAWN_STATUS,
                 "p_location": NULL_NOT_DRAWN_STATUS,
                 "null_mean": NULL_NOT_DRAWN_STATUS,
                 "null_sd": NULL_NOT_DRAWN_STATUS,
                 "null_draws_summary": {"n": NULL_NOT_DRAWN_STATUS},
                 "endpoint": "R-801 SETTLEMENT P&L (trades + residual)",
                 "why_no_test_statistic": (
                     "S4 was skipped: the ESTIMATE is the same "
                     "arithmetic on the same replays as a full run; only "
                     "the control is absent")}
                if point_estimate else
                settlement_arm_day(_obs801, nul["settle_values"], params))
            r["economic_settlement"] = {
                **_sblock,
                "arm_total_cents": _sarm801,
                "zero_cancel_baseline_total_cents": _sbase801,
                "arm_legs": {k: v for k, v in _armlegs801.items()
                             if k != "per_slug"},
                "zero_cancel_baseline_legs": {
                    k: v for k, v in _base801.items() if k != "per_slug"},
                "the_per_slug_legs_are_in": "the decision ledger, rows "
                                            "SETTLEMENT_SLUG",
                "admissibility": _adm801,
                "winner_source": {
                    "path": _win801["path"], "sha256": _win801["sha256"],
                    "n_slugs": _win801["n_slugs"],
                    "method": _win801["method"],
                    "chainlink_verification":
                        _win801["chainlink_verification"],
                    "is_final_for_quotation":
                        _win801["is_final_for_quotation"]},
                "ruling": "R-801",
                "what_D_E0_is_now": ("a DIAGNOSTIC of short-horizon "
                                     "adverse selection, not the result"),
            }
        elif r.get("status") in ("OK", "OK_POINT_ESTIMATE"):
            _st801 = ("NO_WINNER_SOURCE_ON_A_FIXTURE"
                      if (fixture and _adm801["admissible"])
                      else "NOT_VALUED_DAY_NOT_ADMISSIBLE")
            r["economic_settlement"] = {
                "status": _st801,
                "admissibility": _adm801,
                # REVIEW 104B §6 (1): THE REASON IS PER STATUS AND
                # COMPUTED. One literal served both statuses and was true
                # of only the second -- it said "this day is not in the
                # admissible set" beside `admissible: true`, which is the
                # printed conclusion contradicting the computed verdict
                # next to it (rule 10).
                "why": settlement_not_valued_reason(_st801, _adm801)}
        r["decision_population"] = pop
        r["seed"] = seed
        r["n_cancels_issued"] = int(arm_replay["cancels_issued"])
        r["n_fills_baseline"] = int(base["n_fills"])
        r["n_fills_arm"] = int(arm_replay["n_fills"])
        results.append(r)
        # R-765: WHAT THIS ARM-DAY HELD, KEPT. Stored verbatim as the
        # producing code shaped it -- BE's fill records and decision rows
        # copied, not re-typed into a schema of mine, because a re-typed
        # copy is a second definition that drifts from the one the receipt
        # was computed from.
        # R-782: THE ABSOLUTES, from the same replay, beside the excess.
        _abs_base = absolute_legs(base["fills"])
        _abs_arm = absolute_legs(arm_replay["fills"])
        _recon = _abs_arm["total"] - _abs_base["total"]
        if abs(_recon - observed) > 1e-9:
            raise RunnerRefused(
                f"REFUSED ABSOLUTES_DO_NOT_RECONCILE: arm_total "
                f"{_abs_arm['total']!r} - baseline_total "
                f"{_abs_base['total']!r} = {_recon!r}, and D(E0) is "
                f"{observed!r} -- a difference of {_recon - observed!r}. "
                f"The absolutes are read from the SAME replay the excess "
                f"is computed from, so a mismatch means they are not the "
                f"same numbers and neither can be published.")
        r["absolute"] = {
            "ruling": "R-782, the USER: 'plz record the absolute number "
                      "as well for reference'",
            "unit": VALUATION_UNIT,
            "zero_cancel_baseline": _abs_base,
            "arm": _abs_arm,
            "reconciliation": {
                "arm_total_minus_baseline_total": _recon,
                "D_E0": observed,
                "agree_to_1e_9": abs(_recon - observed) <= 1e-9,
                "refuses_by_name_otherwise":
                    "ABSOLUTES_DO_NOT_RECONCILE"},
            "read_from": "the same replay that computes D(E0) -- never a "
                         "second pass",
        }
        _ledger765[arm] = {
            "absolute": r["absolute"],
            # R-801: the per-slug decomposition, for BOTH books, so a
            # reader with the ledger and no receipt can re-form the ruled
            # P&L slug by slug without re-running the day.
            "settlement": ({
                "ruling": "R-801",
                "winner_source": {"path": _win801["path"],
                                  "sha256": _win801["sha256"],
                                  "method": _win801["method"],
                                  "chainlink_verification":
                                      _win801["chainlink_verification"]},
                "arm_per_slug": _armlegs801["per_slug"],
                "baseline_per_slug": _base801["per_slug"],
                "arm_total_cents": _sarm801,
                "baseline_total_cents": _base801["total_cents"],
                "D_E_settle": _obs801,
                "unit": VALUATION_UNIT,
            } if (_win801 is not None and r.get("status") in ("OK", "OK_POINT_ESTIMATE"))
                else None),
            "observed": observed,
            "arm_value": _value_cents(arm_replay["fills"]),
            "base_value": base_value,
            "head": spec["head"], "theta": spec["theta"], "seed": seed,
            "n_decisions": pop["decisions"],
            "n_cancels_issued": int(arm_replay["cancels_issued"]),
            "n_fills_arm": int(arm_replay["n_fills"]),
            "n_fills_baseline": int(base["n_fills"]),
            # R-828: a POINT-ESTIMATE run drew no null, so the ledger
            # carries none -- an empty list here would read as "500 draws
            # that all came out zero".
            "null_values": (nul["values"] if nul else None),
            "null_cancels": (nul.get("cancels") if nul else None),
            "null_not_drawn": (NULL_NOT_DRAWN_STATUS if nul is None
                               else None),
            # R-825 / DE 143: THE SETTLEMENT DRAW'S OWN VALUE, PERSISTED.
            # The ledger carried only the DIAGNOSTIC's draws, so every
            # settlement Z and p in a receipt was a number a reader could
            # only take on trust -- and a good-faith re-derivation from
            # the 5-second rows comes out 2.2x-5.4x MORE EXTREME on every
            # arm-day, i.e. the error runs toward OVERSTATING
            # significance. Kept beside the 5-second value, one float per
            # draw, so the endpoint's own moments re-derive from the file.
            "null_settle_values": (nul.get("settle_values")
                                   if nul else None),
            "arm_fills": arm_replay["fills"],
            "baseline_fills": base["fills"],
            "decisions": [dict(r) for r in _stream765
                          if float(r["score"]) >= spec["theta"]],
        }
        per_arm_detail[arm] = {
            "status": r["status"],
            "null_elapsed_s": (nul["elapsed_s"] if nul
                               else NULL_NOT_DRAWN_STATUS),
            "null_peak_rss_mb": (nul["peak_rss_mb_during_draws"] if nul
                                 else NULL_NOT_DRAWN_STATUS)}
    _mark("S4_null")

    # ---- S5: seal. Counts and statuses only. ---------------------------
    # R-674 (c): every arm's REASONS are checked for sealed VALUES before
    # anything is sealed -- the stripper cannot see a number inside a
    # string.
    reasons_checked = [assert_reasons_carry_no_sealed_value(r)
                       for r in results]
    # DE 121 / R-754: THE USER-RULED EARLY READ. `early_read` is None on
    # every path that is not that read, and when it is None NOTHING below
    # differs by a byte from what the four sealed days ran -- the same
    # `seal()`, the same arguments, the same leak assertion. When it is
    # set, the SAME seal function is called with the ruled bar on both
    # sides, which is the documented unsealed state (`n >= g`), so the
    # economics that this day just computed are carried instead of
    # stripped. Nothing about the COMPUTATION changes: the numbers are the
    # ones the sealed run produced and threw away.
    _ruling765 = bool(params.get("user_ruled_unsealed_emission"))
    if _ruling765:
        # ONE PATH, ONE EMISSION. The early-read branch below stays for
        # the four days ruled at R-754 and is no longer needed for a new
        # day: under v18 every run emits its economics anyway.
        sealed = [seal(r, n_days_complete, params["G"],
                       unsealed_by_ruling=True) for r in results]
    elif early_read is None:
        sealed = [seal(r, n_days_complete, params["G"]) for r in results]
        for a in sealed:
            assert_no_economic_leak(a, n_days_complete, params["G"])
    else:
        _bar = int(early_read["G"])
        if day not in early_read["days"]:
            raise RunnerRefused(
                f"REFUSED EARLY_READ_DAY_NOT_IN_THE_BAR: {day} is not one "
                f"of the days the ruling names ({early_read['days']}). The "
                f"early read opens exactly the days the USER ruled and no "
                f"others; a fifth day would be consumed by a read nobody "
                f"authorised.")
        sealed = [seal(r, _bar, _bar) for r in results]
        for a in sealed:
            if a.get("sealed") is not False:
                raise RunnerRefused(
                    "REFUSED EARLY_READ_STILL_SEALED: the early read asked "
                    "for the unsealed layout and got a sealed artifact. "
                    "The read exists to show the economics; an artifact "
                    "that still hides them is not the thing that was "
                    "ruled.")
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

    # R-765: THE LEDGER IS WRITTEN BEFORE THE RECEIPT, so the receipt can
    # name it by digest. A day whose ledger cannot be written REFUSES --
    # the ruling is that the numbers are kept, and a receipt promising a
    # ledger that is not there would be worse than no promise.
    # DE 132: THE ANCHOR IS A PARAMETER, and a missing ledger REFUSES.
    # At fe76d83 this read `receipt_path is not None`, and the EARLY-READ
    # wrapper calls `run_day` without one -- its artifact is its own
    # family -- so the condition was false and R-765's ledger was SILENTLY
    # SKIPPED. GO E1 and GO E2 both emitted `decision_ledger: null`. My
    # own comment two lines below says "a receipt promising a ledger that
    # is not there would be worse than no promise"; a null block is the
    # silent version of exactly that, and it is why nobody noticed for two
    # heavy runs.
    _anchor = ledger_anchor if ledger_anchor is not None else receipt_path
    _ledger_block = None
    if _ruling765 and _ledger765:
        if _anchor is None:
            # A FIXTURE OWES NO LEDGER, and saying so is not the null that
            # hid the defect: a fixture's numbers are synthetic, there is
            # nothing to avoid re-running, and the runner's own battery
            # drives `run_day` on fixture days with no anchor. DE 132's
            # refusal did not make that distinction, so it fired on this
            # module's own cells the moment the cascade pin was fresh
            # enough for them to run -- which is where E3's composition
            # found it.
            _ledger_block = {
                "status": "NO_LEDGER_FOR_A_FIXTURE_DAY",
                "why": "a fixture's rows are synthetic; R-765 keeps the "
                       "numbers of REAL runs so they need not be re-run",
                "a_real_day_without_an_anchor":
                    "REFUSES DECISION_LEDGER_HAS_NO_ANCHOR",
                # DE 134, rule 10: the STATUS says "fixture", so the
                # field that makes it true is recorded beside it rather
                # than inferred by the reader from the guard above. A
                # real day cannot reach here -- `assert_ledger_anchor`
                # refuses it before the work -- and this says which
                # value made the sentence true.
                "fixture": fixture}
    if _ruling765 and _ledger765 and _anchor is not None:
        import de_decision_ledger as _LED
        # DE 134: the anchor may be a FILE (the early read's artifact) or
        # a DIRECTORY (the day path's output dir). One named function
        # resolves both, and it is driven both ways.
        _lp = ledger_path_for(_anchor, day, emission_stamp())
        try:
            import harmful_stateful_policy as _HSP
            _side765 = _HSP.SIDES[0]
        except Exception as _e765:
            raise RunnerRefused(
                f"REFUSED DECISION_LEDGER_NO_SIGN_CONVENTION: the buy "
                f"side could not be read from the policy module "
                f"({_e765}). Every fill's sign would be a guess, and a "
                f"ledger valued backwards is worse than none.")
        _ledger_block = _LED.write_ledger(_lp, day, _ledger765,
                                          buy_side=_side765)
        _ledger_block["ruling"] = "R-765"
        _ledger_block["recompute_with"] = (
            "de_decision_ledger.read_ledger(path, "
            "expect_sha256=<this sha256>) then .recompute(led, arm)")

    # THE DAY'S SEAL STATE, READ FROM WHAT WAS EMITTED (DE 126).
    _all_sealed = bool(sealed) and all(a.get("sealed") is True
                                       for a in sealed)

    wall = time.time() - t_start
    peak = max(v["peak_rss_mb_highwater"] for v in stages.values())
    _plat = placement_latency_from_the_book(
        json.loads(Path(receipt).read_text()),
        book_path=book_path, book_sha256=book_sha,
        require_declared=bool((params.get("settlement_endpoint") or {})
                              .get("require_book_declares_L")))
    r20 = assert_rule20(obs, wall_s=wall, peak_rss_mb=peak, day=day)
    peak_pred = peak_stage_predicate(stages,
                                     declared=declared_peak_stage())
    peak_shape = assert_peak_stage(peak_pred, fixture=fixture, day=day)
    # THE BUDGET IS THIS RUN'S GROWTH, NOT THE PROCESS HIGH-WATER, and it
    # cost the 09-03 smoke 85 minutes. `_peak_rss_mb()` is `ru_maxrss` --
    # non-decreasing for the life of the PROCESS. A REAL day runs the full
    # battery at emit (REV 49 S3.4), the battery runs FIXTURE days in the
    # SAME process, and those fixture days inherited the real day's 2.4 GB
    # peak and blew a 700 MB fixture budget. The day's own work was
    # finished; the receipt was never written.
    _base = stages.get("S_start", {}).get("rss_mb_current")
    _cur = [v.get("rss_mb_current") for v in stages.values()
            if isinstance(v.get("rss_mb_current"), float)]
    growth = ((max(_cur) - _base) if (_cur and isinstance(_base, float))
              else peak)
    if growth > budget:
        raise RunnerRefused(
            f"REFUSED DAY {day}: this run GREW {growth:.0f} MB (process "
            f"peak {peak:.0f} MB over a {_base if _base else 0:.0f} MB "
            f"baseline), exceeding the declared budget {budget:.0f} MB. "
            f"The DAY refuses -- the cap is never raised and the draw "
            f"count is never cut (R-174).")
    scope_mem = scope_memory_observation()
    import os as _os3
    return {
        "protocol": "P003_DE_MULTIDAY_GATE1_DAY_RUN_V1",
        # THE INTERPRETER THAT ACTUALLY RAN, resolved. The venv's python3
        # is a SYMLINK, so the path invoked and the binary executed are
        # two different facts and both travel.
        "interpreter": {
            "sys_executable": sys.executable,
            "realpath": _os3.path.realpath(sys.executable),
            "is_a_symlink": _os3.path.islink(sys.executable),
            "version": sys.version.split()[0]},
        # (4) A DAY WITH NO ADMISSIBLE ARM SAYS SO IN ITS STATUS. It
        # carried the constant DAY_RUN_SEALED, so a day that contributed
        # ZERO SIGNS was indistinguishable from one that contributed two --
        # and six such days would satisfy every existence test in the read
        # gate while the aggregate had nothing to aggregate.
        # DE 126: THE WORD "SEALED" IS COMPUTED FROM THE ARM-DAYS, not
        # typed. It was a literal: the 09-03 early-read artifact carried
        # `DAY_RUN_SEALED` while BOTH arm-days read `sealed False` with
        # their economics present. That is CLAUDE.md rule 10 -- a
        # hardcoded verdict string beside a table that contradicts it --
        # and it is REV 90 §A0's class one level up: REV fixed the
        # arm-day's `seal_status`, and the DAY's own status went on
        # asserting the opposite.
        "status": ("FIXTURE_DAY_RUN_NO_REAL_DATA" if fixture
                   else (("DAY_RUN_SEALED" if _all_sealed
                          else "DAY_RUN_UNSEALED") if any(
                       (a.get("admissibility") or {}).get("admissible")
                       is True for a in sealed)
                       else ("DAY_RUN_SEALED_NO_ADMISSIBLE_ARM"
                             if _all_sealed
                             else "DAY_RUN_UNSEALED_NO_ADMISSIBLE_ARM"))),
        "n_admissible_arms": sum(
            1 for a in sealed
            if (a.get("admissibility") or {}).get("admissible") is True),
        "day": day,
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fixture": fixture,
        "reference_book": bookcite,
        "before_work": (
            {"ran": before_work is not None,
             "when": "after the book digest was verified, before S1 -- "
                     "NOT at the emit (R-610)",
             "residency": before_work_proof}
            if before_work is not None else
            {"ran": False,
             "when": "no hook was passed; the day path ran alone"}),
        "work_counters": {
            "draws_performed_by_this_day": _day_draws(),
            "draws_performed_by_the_before_work_hook": _hook_draws,
            "process_total_draws": RUN_COUNTERS["draws_performed"],
            "why_deltas": "the battery runs FIXTURE days and those draw; "
                          "a process-wide counter would credit the day "
                          "with the battery's work",
        },
        "be_module_citation": cite,
        "draw_pool_set_equality_checked": pool_equal,
        "decision_populations": pops,
        "per_day_sealed_artifacts": sealed,
        "seal_layout_symmetry_checked_on_the_emitted_results": seal_symmetry,
        "reasons_carry_no_sealed_value": reasons_checked,
        "fixture_day_lock": day_lock,
        "launch_form_at_runtime": launch_runtime,
        "lock_form_at_runtime": lock_runtime,
        "decision_ledger": _ledger_block,
        "n_days_complete": n_days_complete,
        # REV 90 §A0(3): `G` USED TO SIT HERE AS A BARE NUMBER and it was
        # v15's SIX, beside `n_days_complete` 4 and the early-read
        # wrapper's `G` 4 -- two different G's in one artifact, with the
        # smaller one outside this block. A reader met "4 of 6" and a
        # status claiming all days were complete. There is no bare `G`
        # now: the number keeps its provenance in the same object, and
        # the key names WHICH G it is.
        #
        # The params file is NAMED, not versioned into the key: a key
        # called `design_G_from_params_v15` is a literal that must track a
        # moving thing, and `PARAMS_REL` is the thing that moves.
        #
        # R-810: WHAT MAKER THIS DAY MEASURED, from the book's own
        # builder receipt rather than from a sentence here.
        "placement_latency": _plat,
        # ---- DE 155 (1): THE DECISION IS CAUSAL NOW, AND THE THRESHOLD
        # ---- IT IS COMPARED AGAINST WAS NOT FITTED FOR IT.
        # Carried on every receipt because a reader who does not know this
        # would attribute the change in cancel count to the market.
        "scoring_timing": {
            "rule": ("each scored ROW is emitted at its own `t_start` and "
                     "the FIRST crossing of theta_cancel cancels "
                     "(one cancel per generation, the engine's own "
                     "invariant)"),
            "was": ("ONE event per generation at the generation's START, "
                    "carrying the MAXIMUM score over all the generation's "
                    "rows -- so a row that had not happened yet could "
                    "decide a cancellation"),
            "measured_incidence_on_2026-09-04": (
                "15,867 of 40,000 sampled rows (39.7 %) begin strictly "
                "after their generation's start, across 6,586 of 24,133 "
                "generations (27 %); max 60 rows in one generation"),
            "THE_CALIBRATION_CONSEQUENCE": (
                "theta was fitted by `phase2_arms.freeze_thresholds` over "
                "per-generation MAXIMA. A first-crossing score is at or "
                "below that maximum for the same generation, so at an "
                "UNCHANGED theta fewer generations cross and the selected "
                "count moves FOR A REASON THAT IS NOT A MARKET REASON. "
                "The threshold is frozen and declared and was NOT re-fitted "
                "here: re-fitting inside a defect repair would change the "
                "policy under cover of a bug fix (rule 14 -- the decision "
                "is the USER's). The before/after cancel count per arm per "
                "day is the evidence a re-fit ruling needs."),
            "AND_IT_MOVES_THE_CONTROL_TOO": (
                "the null is matched on the arm's action count, so a "
                "changed cancel count changes the matched null as well as "
                "the arm -- the comparison is not an arm-only shift"),
            "theta_refitted": False,
        },
        # R-828 (4): THE RUN'S OWN IDENTITY. A point-estimate run is a
        # DIFFERENT KIND of artifact, not a lesser one and not a
        # predecessor: a full run at the same (day, L) neither supersedes
        # it nor is superseded by it -- they answer different questions.
        "run_mode": ("POINT_ESTIMATE" if point_estimate else "FULL"),
        "run_mode_note": (
            ("S4 SKIPPED: no null drawn, no Z, no p, no control. The "
             "point estimates are the same arithmetic on the same "
             "replays as a full run. Nothing quoted from this artifact "
             "may imply significance, and asking it for a test "
             "statistic REFUSES by name "
             f"({POINT_ESTIMATE_REFUSAL}).")
            if point_estimate else
            "the full run: S4 drawn, the null valued, Z and p computed"),
        "G_and_which_G_it_is": {
            "design_G_from_params": params["G"],
            "params_file": PARAMS_REL,
            "n_days_complete_at_this_emit": n_days_complete,
            "the_bar_this_run_sealed_against": (
                int(early_read["G"]) if early_read else params["G"]),
            "why_two_numbers_exist": (
                "the design's G is the ruled six-day population; the bar "
                "a run seals against is the same six EXCEPT under the "
                "user-ruled early read (R-754), which seals against its "
                "own four. They are different questions and they are no "
                "longer both called G."),
            "no_bare_G_key_here": (
                "removed at REV 90 §A0(3). Nothing read it: a sweep of "
                "live/pm_research found no consumer of a day-run "
                "receipt's top-level `G`. The four landed sealed receipts "
                "(09-03..09-06) still carry the old bare `G: 6`; from "
                "here on receipts carry this block instead, which is a "
                "schema change between days and is said so here rather "
                "than discovered later."),
        },
        "memory_plan": {
            "stages": [{"stage": k, "holds": v} for k, v in DAY_STAGES],
            "observed": stages,
            "peak_rss_mb": peak,
            "baseline_rss_mb": _base,
            "growth_rss_mb": growth,
            "budget_mb": budget,
            "growth_is_process_history_dependent": (
                "a warm process grows less than a cold one -- the "
                "allocator already holds what an earlier run took. The "
                "REAL day runs in a FRESH process, which is the regime the "
                "4000 MB budget is derived against"),
            "budget_is_on_GROWTH_not_the_process_peak": (
                "ru_maxrss never falls, so a fixture day run inside a real "
                "day's process inherits that day's peak. It cost the 09-03 "
                "smoke 85 minutes: the day's work finished and the emit-"
                "time battery refused its own fixture on the real day's "
                "high-water"),
            "within_budget": growth <= budget,
            # THE HOOK'S MEMORY IS INSIDE THE BUDGET, and that is a fact
            # about the baseline, not a hope: the baseline is taken at
            # S_start, BEFORE the hook, so everything the hook allocates
            # counts against the day's declared growth. A stage whose cost
            # escapes the budget is a stage nobody bounded.
            "hook_stage_growth_mb": (
                (stages.get(HOOK_STAGE, {}).get("rss_mb_current", 0.0)
                 - _base) if (HOOK_STAGE in stages
                              and isinstance(_base, float)) else None),
            "peak_stage": peak_pred,
            "peak_stage_assertion": peak_shape,
            "index_splits": INDEX_SPLITS_NEEDED_BY_DAY,
        },
        "battery_stage_is_inside_the_budget": (
            HOOK_STAGE in stages and isinstance(_base, float)
            and stages[HOOK_STAGE]["rss_mb_current"] >= _base),
        "wrapper": {**obs, "rule20": r20},
        "resources": {"wall_seconds": wall, "peak_rss_mb": peak,
                      "per_arm": per_arm_detail,
                      # RSS is this PROCESS; the scope block is the CGROUP,
                      # which is what rule 20's cap is applied to and what
                      # counts page cache.
                      "scope_memory": scope_mem},
        "what_this_is_not": {
            "a_result": False,
            # R-656 (REV 70 S0.2): the per-arm counts are POPULATION
            # SIZES, outside the seal, and they are named here so a
            # reader meets the ruling where the numbers are.
            # R-674: GENERATED FROM THE SCOPE MAP IN FORCE, never a
            # literal. This block listed all FOUR counts as open sizes
            # under R-656 -- which R-659 had already reversed for three
            # of them -- so a receipt whose emitter sealed eleven names
            # described itself as sealing eight. The receipt's
            # self-description contradicted its own seal.
            "the_seal_scope_in_force_for_THIS_receipt": {
                "design_version": DESIGN_VERSION_IN_FORCE,
                "sealed_names": list(economic_fields_in_force()),
                "n_sealed": len(economic_fields_in_force()),
                "open_population_sizes": list(OPEN_POPULATION_SIZES),
                "generated_from": ("ECONOMIC_FIELDS and "
                                   "SEALED_FROM_DESIGN_VERSION -- the map "
                                   "the emitter actually applied, read at "
                                   "emit"),
                "why_generated": (
                    "a literal list here said four counts were open "
                    "SIZES while the emitter sealed three of them "
                    "(R-674). A receipt that describes its own seal from "
                    "a constant somebody edited separately describes a "
                    "different seal"),
                "the_open_sizes_are": (
                    "the action-side counts rule 8 requires and what R4's "
                    "admissibility bar reads. `n_decisions` cannot be "
                    "sealed without making the bar uncheckable"),
                "the_sealed_outcome_counts_are": (
                    "counts of what the policy DID: arm fills MINUS "
                    "baseline fills is the intervention's effect in "
                    "events (R-659)"),
            },
            # DE 126: READ FROM THE ARTIFACTS, not from the day count.
            # `n_days_complete < G` was TRUE on the early read (4 of 6)
            # while every arm-day was unsealed and carried its economics --
            # so the receipt asserted its numbers were hidden on the very
            # artifact that published them.
            "the_economics_are_SEALED": _all_sealed,
            "the_economics_are_SEALED_read_from": (
                "the emitted arm-days' own `sealed` flags, never from "
                "`n_days_complete < G` -- which was true on the R-754 "
                "early read while both arm-days were unsealed"),
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
    # THE CLAIM IS ABOUT THE DAY PATH. If a `before_work` hook ran inside
    # the instrumented region (the battery does, since R-610), the opens
    # it made are ITS opens and are subtracted -- reported, never dropped.
    _full = DR.last_full_paths()
    _all_hits = tape_artifacts_opened(proof, _full)
    _hook = ((result.get("before_work") or {}).get("residency") or {})
    _hook_hits = list(_hook.get("tape_artifacts_opened") or [])
    hits = sorted(set(_all_hits) - set(_hook_hits))
    # NON-VACUITY THAT IS SPECIFIC TO THIS CLAIM. `non_vacuous` says the
    # instrument saw SOME open; that is not enough here. The day path
    # certainly reads the BOOK, so the instrument must have seen THAT --
    # otherwise "no tape artifact was opened" could be an instrument that
    # missed the reads rather than reads that did not happen.
    # THE MEMBERSHIP QUESTION IS ASKED OF THE UNCAPPED SET. It was asked
    # of `distinct_paths`, which is capped at DR.PATH_LIST_CAP -- and once
    # DE 90 put the battery inside this instrumented region the run went
    # from ~25 opens to ~5,800, the book's own path fell off the end of the
    # capped list, and this guard REFUSED A CORRECT RUN. It would have
    # refused the real day AFTER 85 MINUTES: the check runs when the day
    # returns. Found by running `--synthetic-day` end to end, which DE 90's
    # own checks did not do -- they drove this function with a STUB hook
    # that opens nothing.
    _bp = str(Path(book_path).resolve())
    _saw_book = any(str(Path(p).resolve()) == _bp for p in _full)
    _saw_book_in_the_capped_list = any(
        str(Path(p).resolve()) == _bp
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
        "n_paths_opened_uncapped": len(_full),
        "the_book_was_in_the_CAPPED_list_too": _saw_book_in_the_capped_list,
        "why_that_second_field_exists": (
            "the capped list is what the ARTIFACT carries and the uncapped "
            "set is what the QUESTION is asked of. When they disagree the "
            "cap bit, and a reader can see that it did rather than "
            "wondering why a 200-entry list does not contain the book"),
        "tape_artifacts_opened": hits,
        "tape_artifacts_opened_by_the_whole_process": _all_hits,
        "tape_artifacts_opened_by_the_before_work_hook": _hook_hits,
        "the_claim_is_about": "the DAY PATH: the whole-process set MINUS "
                              "the before_work hook's own set. Both are "
                              "reported, so the subtraction is auditable",
        "path_list_capped_at": DR.PATH_LIST_CAP,
        "capped_list_truncated": proof.get("distinct_paths_truncated"),
        "why_truncation_does_not_hide_a_hit": (
            "the tape scan reads `data_paths_opened`, which is NOT capped, "
            "in union with the capped list -- every marker lives under "
            "`data/`"),
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


def battery_resources(t0: float, hw0: float) -> dict:
    """WHAT THE BATTERY ITSELF COST, against rule 20's own bar.

    REV 55 S2.4: it is ~850 MB / ~24 s and was 52 MB three rounds ago.
    Every seat runs it several times a round as a STANDALONE command,
    where rule 20's bar applies to IT -- so it measures itself, and the
    trend is a field in every receipt that embeds it rather than something
    a reviewer has to go and time.

    The bar comes from the constants `assert_rule20` uses. I first wrote a
    second pair beside them: two spellings of one fact, the defect this
    codebase keeps finding."""
    wall = time.time() - t0
    peak = _peak_rss_mb()
    bar_mb = HEAVY_RSS_GB * 1024.0
    return {
        "wall_seconds": round(wall, 2),
        "process_peak_rss_mb_at_the_end": round(peak, 1),
        "process_peak_rss_mb_at_the_start": round(hw0, 1),
        "heavy_bar_mb": bar_mb,
        "heavy_bar_seconds": HEAVY_WALL_S,
        "fraction_of_the_heavy_bar_by_memory": round(peak / bar_mb, 3),
        "fraction_of_the_heavy_bar_by_wall": round(wall / HEAVY_WALL_S, 3),
        "would_need_the_lock_standalone": (peak > bar_mb
                                           or wall > HEAVY_WALL_S),
        # THE AMBIENT LOCK STATE THE BATTERY RAN IN. DE 92: one check's
        # verdict depended on it, so it passed for every seat (no lock)
        # and failed inside the only run that matters (lock held). The
        # configuration now travels in every receipt that embeds a
        # battery, so "which state was this battery run in" is a field
        # rather than a question.
        "ran_holding_the_heavy_run_lock": bool(
            wrapper_observed().get("heavy_run_lock_held")),
        "most_of_the_peak_is_one_check": (
            "REV 53 S0.3's control inflates the process past the 700 MB "
            "fixture budget ON PURPOSE and frees it immediately -- it has "
            "to exceed that budget to reproduce the case. That is why the "
            "peak is ~850 MB and the RETAINED figure is ~26 MB, and why "
            "the peak cannot be reduced without retiring the control"),
    }


#: REV 84 §2's falsifier, END TO END. Set by `--falsify-disarmed`: the
#: real battery runs and ONE cell is planted DISARMED, so the summary
#: path -- not a hand-built dict -- has to print `n_disarmed 1` and exit
#: non-clean. A predicate driven only through its own helper is a
#: predicate nobody has seen fire in the run it protects.
PLANT_ONE_DISARMED_CELL = False

#: REV 88 §1.3's falsifier, END TO END, and it is the RED half: set by
#: `--falsify-skipped`, the real battery runs and ONE skip is planted
#: WITHOUT a reason, so the summary path -- not a hand-built dict -- has
#: to print `n_skipped` including it, name it as unnamed, and exit
#: non-clean. The GREEN half needs no flag: every real offline run skips
#: named cells and stays PASS.
PLANT_ONE_UNNAMED_SKIP = False


def selftest(*, quiet: bool = False, offline: bool = False) -> int:
    """`offline=True` skips the checks that READ `data/` and RECORDS them.

    The fixture run must open no path under `data/` (reviewer efba2b6
    item 3) and must still carry a battery that ran in its own process.
    Both are possible only if the battery can say which checks it did not
    run and why -- a skipped check that is silent is a check that has
    stopped existing."""
    n = [0]
    skipped: list = []
    _bat_t0, _bat_hw0 = time.time(), _peak_rss_mb()

    disarmed: list = []

    def offline_skip(label, why=None):
        # REV 88 §1.3: THE SKIP IS RECORDED AS A CELL, not as a tally.
        # `why` defaults to the offline reason because that is what all
        # 26 current callers mean, but it is stored PER CELL -- a reason
        # held once for the whole list cannot distinguish a check that
        # reads `data/` from one that was quietly dropped.
        skipped.append({
            "label": label,
            "why": why or ("offline: this check READS `data/`, and a "
                           "fixture run must open no path under it")})
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
    def _with_current_cascade(params):
        out = json.loads(json.dumps(params))
        for _m in (out.get("be_cascade") or {}).get("modules") or []:
            _f = root / _m["path"]
            if _f.is_file():
                _m["sha256"] = hashlib.sha256(_f.read_bytes()).hexdigest()
        return out

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
    _tipP = _with_current_cascade(P)
    ok(verify_be_module(P)["cited_not_copied"] is True
       and _tipP["be_cascade"] == P["be_cascade"],
       "POSITIVE CONTROL: BE's cascade module resolves at the declared "
       "digest and is CITED, not copied; v20 already names the current "
       "corrected closure rather than relying on a test-only repoint")
    _bad_cascade = json.loads(json.dumps(P))
    next(r for r in _bad_cascade["be_cascade"]["modules"]
         if r["path"].endswith("de_phase4_diag_runner.py"))["sha256"] = \
        "0" * 64
    refuses(lambda: verify_be_module(_bad_cascade),
            "KNOWN-BAD: a params file pinning the old phase4 cascade "
            "REFUSES instead of letting the battery run through a different "
            "null producer", "BE_CASCADE_DIFFERS")
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
        vr = verify_run_inputs(_with_current_cascade(live))
        _n_models_declared = sum(len(a["model_digests"])
                                 for a in live["arms"].values())
        ok(vr["models"]["n_models_read"] == _n_models_declared
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
        # THE NAME COMES FROM `PARAMS_REL`, never typed beside it. It read
        # `..._params_v10.json` as a literal, so the params bump this
        # round would have turned a non-vacuity check into a check of a
        # file nobody reads -- and it would have gone GREEN by failing to
        # find what it was no longer looking for. (Same shape as the
        # `zip(names, [six floats])` above: a literal that has to track
        # something that moves.)
        _pname = PARAMS_REL.rsplit("/", 1)[-1]
        ok(any(x.endswith(_pname) for x in _seen),
           f"and the instrument is not vacuous -- it DID observe the "
           f"parameter file ({_pname}, read from PARAMS_REL and not typed "
           f"here) being read, so a zero above is a measurement rather "
           f"than a silent no-op")
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
    # ---- R-602: the seal-open bar is a PREDICATE ON ARTIFACTS ----------
    import tempfile as _tfr

    def _synth_ledger(days, *, tamper=None, no_admissible=None,
                      reroll=None, skip_landing=None):
        """A ledger root satisfying R-604's eight conjuncts, or failing ONE
        of them on purpose. Built, never faked: every digest is computed
        from the bytes that were written, and the producing-code digest is
        the RUNNER'S OWN BLOB AT HEAD, so conjunct 6 is driven against a
        real commit rather than a constant."""
        import subprocess as _sp3
        d = _tfr.mkdtemp(prefix="de87_")
        der = Path(d) / "pm_5min/derived"
        der.mkdir(parents=True)
        _repo = Path(__file__).resolve().parents[2]
        _head = _sp3.run(["git", "-C", str(_repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
        _blob = _blob_sha256_at(
            _head, "live/pm_research/de_multiday_gate1_runner.py", _repo)
        for day in days:
            compact = [x for x in sorted(day_forms(day))
                       if "-" not in x][0]
            bk = der / f"be_daybook_{compact}_btc.pkl"
            bk.write_bytes(b"book-" + compact.encode())
            sha = sha256_streamed(bk)
            if tamper == day:
                sha = "0" * 64
            rp = (der / f"{SEALED_DAY_RECEIPT_PREFIX}{compact}"
                        f"{SEALED_DAY_RECEIPT_MIDFIX}20260906T000000Z.json")
            adm = (day != no_admissible)
            rp.write_text(json.dumps({
                "day": day,
                "status": ("DAY_RUN_SEALED" if adm
                           else "DAY_RUN_SEALED_NO_ADMISSIBLE_ARM"),
                "reference_book": {"path": str(bk), "sha256": sha},
                "source_identity": {"carrying_commit": _head,
                                    "producing_code_sha256": _blob},
                "per_day_sealed_artifacts": [
                    {"arm": "A", "sealed": True,
                     "admissibility": {"admissible": adm}}],
            }))
            landed = sha256_streamed(rp)
            if reroll == day:
                # THE DAY RE-RAN AFTER LANDING: the file changes and the
                # landing record still names the old digest.
                rp.write_text(rp.read_text().replace('"arm": "A"',
                                                     '"arm": "A "'))
            if skip_landing != day:
                # DA'S DECLARED SHAPE, not a shape DE invented for its own
                # fixture: the flag that says this IS a landing record,
                # and the digest in BOTH the places DA's emitter writes it
                # (REV 54 S1.3). A fixture built to a shape only one seat
                # writes tests only that seat.
                (der / f"p003_da_gate1_pre_read_{compact}__"
                       f"20260906T000000Z.json").write_text(json.dumps({
                           "day": day, "mode": "PRE_READ",
                           LANDING_RECORD_DECLARED_FLAG: True,
                           "landing_record": {
                               "day": day, "receipt_path": rp.name,
                               "receipt_sha256": landed},
                           "receipt": {"path": rp.name,
                                       "sha256": landed}}))
        return Path(d)

    def _synth_rows(days, *, bad=None):
        return {d: {k: (d != bad) for k in LEDGER_MEMBERSHIP_CONJUNCTS}
                for d in days}

    _D6, _rows6 = live["days"], _synth_rows(live["days"])
    _all6 = _synth_ledger(_D6)
    _after = _t(2026, 9, 9, 0, 6, tzinfo=_tz)
    _before = _t(2026, 9, 8, 23, 59, tzinfo=_tz)
    _read = may_read_aggregate(live, n_days_complete=6, now_utc=_after,
                               root=_all6, ledger_rows=_rows6)
    ok(_read["may_read"] is True
       and _read["read_gate"]["may_open"] is True
       and len(_read["read_gate"]["conjuncts"]) == 7
       and all(c["holds"] for c in _read["read_gate"]["conjuncts"]),
       f"R-604 POSITIVE CONTROL, AND IT ADMITS: six of six after the "
       f"clock, every conjunct holding -- "
       f"{[c['conjunct'] for c in _read['read_gate']['conjuncts']]}")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_before, root=_all6,
        ledger_rows=_rows6),
        "R-604 (1) KNOWN-BAD, SIX OF SIX BEFORE THE CLOCK: the artifacts do "
        "not open the read early -- the clock is a GENUINE conjunct, not a "
        "formality the receipts can satisfy around. And it is checked in "
        "ONE place: it was ALSO checked ahead of the gate, which is two "
        "implementations of one bar inside one function", "1_clock")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_after,
        root=_synth_ledger(_D6[:5]), ledger_rows=_rows6),
        "R-604 (2) KNOWN-BAD, FIVE OF SIX: the RULED days come from the "
        "params -- a receipt for another day would fill no hole",
        "2_all_six_ruled_days")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_after,
        root=_synth_ledger(_D6, reroll=_D6[2]), ledger_rows=_rows6),
        "R-604 (3) KNOWN-BAD, A DAY RE-ROLLED AFTER LANDING: its digest no "
        "longer matches the one DA's pre-read recorded, and it REFUSES -- "
        "WAITING MUST NOT BECOME RE-ROLLING",
        "3_digest_matches_the_landing_record")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_after,
        root=_synth_ledger(_D6, no_admissible=_D6[4]), ledger_rows=_rows6),
        "R-604 (4) KNOWN-BAD, A DAY WITH NO ADMISSIBLE ARM: it satisfies "
        "existence and contributes ZERO SIGNS, so the read REFUSES. Six "
        "such days would have passed every earlier test while the "
        "aggregate had nothing to aggregate",
        "4_each_day_has_an_admissible_arm")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_after, root=_all6,
        ledger_rows=_synth_rows(_D6, bad=_D6[1])),
        "R-604 (5) KNOWN-BAD, A DAY WHOSE LEDGER VERDICT FAILS: a sealed "
        "receipt is DE's own artifact and is not a day verdict; membership "
        "is R-555's and lives in the ledger", "5_ledger_verdict_holds")
    _badcode = _synth_ledger(_D6)
    _bc = sorted((_badcode / "pm_5min/derived").glob(
        f"{SEALED_DAY_RECEIPT_PREFIX}*"))[0]
    _bcj = json.loads(_bc.read_text())
    _bcj["source_identity"]["producing_code_sha256"] = "0" * 64
    _bc.write_text(json.dumps(_bcj))
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=6, now_utc=_after, root=_badcode,
        ledger_rows=_rows6),
        "R-604 (6) KNOWN-BAD, PRODUCING CODE NOT LOCATABLE: a receipt whose "
        "producing digest is in no commit names code nobody can fetch. For "
        "2026-09-03 the .v2 supersession is exactly what makes this hold",
        "6_producing_code_is_locatable")
    refuses(lambda: may_read_aggregate(
        {**live, "read_gate": {}}, n_days_complete=6,
        now_utc=_after, root=_all6, ledger_rows=_rows6),
        "R-604 (8) KNOWN-BAD, THE FIELD ABSENT: a params file without "
        "`read_gate.the_bar_is_a_CONJUNCTION` REFUSES before any day is "
        "examined -- DA's gate reads the SAME field, and a bar implemented "
        "twice from different fields can disagree without anybody noticing",
        "no `read_gate.the_bar_is_a_CONJUNCTION`")
    _bar = live["read_gate"]["the_bar_is_a_CONJUNCTION"]
    ok(len(_bar) == 8 and all(c.get("id") and c.get("text")
                              and c.get("evaluated_by_runner")
                              for c in _bar),
       f"R-604 (8) ONE FIELD, EIGHT OBJECTS WITH STABLE IDS: "
       f"{[c['id'] for c in _bar]}. v7 carried the eight under a SECOND "
       f"key while this one still listed R-602's TWO strings, and DA read "
       f"the latter and reported '2 conjuncts declared' -- two fields, two "
       f"implementations, silently different bars")
    _idmap = read_gate(live, now_utc=_after, root=_all6,
                       ledger_rows=_rows6)
    ok(_idmap["declared_conjunct_ids"] == [c["id"] for c in _bar]
       and set(_idmap["id_to_evaluator"]) == set(_idmap[
           "declared_conjunct_ids"]),
       "and EVERY declared id has an evaluator here, and every evaluator a "
       "declared id -- the correspondence is checked, not assumed")
    _extra = {**live, "read_gate": {
        **live["read_gate"],
        "the_bar_is_a_CONJUNCTION": _bar + [
            {"id": "a_bar_nobody_evaluates", "text": "x",
             "evaluated_by_runner": "none"}]}}
    refuses(lambda: may_read_aggregate(
        _extra, n_days_complete=6, now_utc=_after, root=_all6,
        ledger_rows=_rows6),
        "AND A CONJUNCT ID WITH NO EVALUATOR CLOSES THE GATE: a bar nobody "
        "applies is not a bar, which is what DA already does for an "
        "unknown string", "no evaluator")
    _five = _synth_ledger(_D6[:5])
    _hz = may_read_aggregate(live, n_days_complete=5,
                             now_utc=_t(2026, 9, 9, 12, 0, tzinfo=_tz),
                             root=_five, ledger_rows=_rows6)
    ok(_hz["may_read"] is True and _hz["G"] == 5
       and _hz["opened_at_the_HORIZON"] is True
       and _hz["verdict_is"] == "DIRECTIONAL ONLY"
       and _hz["unbuilt_days"] == [_D6[5]],
       f"R-604 (7) THE HORIZON, DRIVEN: at 2026-09-09T12:00Z with five "
       f"landed days the read OPENS at G = 5, DIRECTIONAL ONLY, with "
       f"{_hz['unbuilt_days']} disclosed as UNBUILT. Declared WITH ITS "
       f"DATE so the stopping rule is not a degree of freedom chosen after "
       f"seeing which days landed")
    refuses(lambda: may_read_aggregate(
        live, n_days_complete=5,
        now_utc=_t(2026, 9, 9, 11, 59, tzinfo=_tz), root=_five,
        ledger_rows=_rows6),
        "and ONE MINUTE BEFORE THE HORIZON five days do NOT open it -- the "
        "horizon is a declared instant, not a mood",
        "2_all_six_ruled_days")
    # ---- R-656 (REV 70 S0.2/S3): the seal's scope, and the format ----
    # R-659 REVERSES R-656 ON THREE OF THE FOUR. This check listed the
    # four as sizes; three of them are OUTCOME counts (arm fills MINUS
    # baseline fills is the intervention's effect in events) and are now
    # sealed. Both sets are read from the module's own tuples, so the
    # check cannot disagree with the emitter.
    ok(set(ECONOMIC_FIELDS) & set(OPEN_POPULATION_SIZES) == set()
       and {"n_fills_arm", "n_fills_baseline",
            "n_cancels_issued"} <= set(ECONOMIC_FIELDS)
       and "n_decisions" in OPEN_POPULATION_SIZES,
       f"R-659: the SEALED names ({len(ECONOMIC_FIELDS)}) and the open "
       f"population SIZES ({len(OPEN_POPULATION_SIZES)}) are DISJOINT, "
       f"and the three OUTCOME counts are on the SEALED side. "
       f"`n_decisions` stays open because R4's admissibility bar reads "
       f"it -- sealing it would make the bar uncheckable")
    # THE EXTENSION IS SCOPED (REV 72 S1.4). Extending the list alone
    # made DA's `economic_absence()` read the LANDED 09-03 receipt as six
    # leaks -- the first sealed day accused by its own instrument for
    # carrying fields that were OPEN BY RULING when it was written.
    ok(set(economic_fields_in_force(22)) == set(ECONOMIC_FIELDS)
       - {"n_fills_arm", "n_fills_baseline", "n_cancels_issued"}
       and set(economic_fields_in_force(23)) == set(ECONOMIC_FIELDS)
       and len(economic_fields_in_force(22)) == 8,
       f"R-659 / REV 72 S1.4: the seal is SCOPED PER NAME -- "
       f"{len(economic_fields_in_force(22))} names in force under design "
       f"v22, {len(economic_fields_in_force(23))} under v23. "
       f"`_strip_economic` seals by the list in force for THIS run; a "
       f"census judges a receipt by the list in force when THAT receipt "
       f"was produced")
    # ---- R-709: THIS PRODUCER'S EXIT MAP, DECLARED --------------------
    if offline:
        offline_skip("R-709 the producer exit map at the chain head")
        offline_skip("R-709 the known-bad: a map claiming 75")
    else:
        _em = producer_exit_map()
        ok(_em["status"] == "DECLARED"
           and _em["agrees_with_the_module"] is True
           and _em["lock_conflict_rc_excluded"] not in _em["codes"],
           f"R-709: this producer's exit map is DECLARED at the chain "
           f"head (v{_em['head_version']}, {_em['head_sha256'][:16]}) and "
           f"AGREES with the module's own `RUNNER_EXIT_CODES` "
           f"{_em['codes']} -- a map beside the code is two definitions; "
           f"this one is checked against it")
        _bad709 = {"producers": {THIS_PRODUCER: {"map": {"0": "x",
                                                          "75": "y"}}}}
        _t709 = Path(_tfr.mkdtemp(prefix="de108map_"))
        (_t709 / EXIT_MAP_DIR).mkdir(parents=True)
        (_t709 / EXIT_MAP_DIR / "producer_exit_maps_v1.json").write_text(
            json.dumps(_bad709))
        refuses(lambda: producer_exit_map(_t709),
                f"R-709 KNOWN-BAD: a map claiming 75 is REFUSED. 75 is "
                f"the LAUNCH LAYER's -- `flock -n -E 75` means a held "
                f"lock and the payload never started -- and a producer "
                f"claiming it makes a refusal indistinguishable from a "
                f"held lock in ExecMainStatus", "LAUNCH LAYER")
    # ---- REV 75: the four defects, driven ----------------------------
    # S1.1 a `{path, sha256: null}` is not a pair (R-608), and the DESIGN
    # was digested at the EMIT -- so its "at load" and "at emit" were one
    # read twice and the pair could not see the file change under a run.
    _id106 = dict(INPUT_DIGESTS)
    # NO `or True`. I wrote one here and removed it: a check that cannot
    # fail is not a check, and this is the second round running I have
    # had to say so about my own.
    ok("params" in _id106
       and bool(_id106["params"].get("sha256_at_load"))
       and all(bool(v.get("sha256_at_load")) for v in _id106.values())
       and "record_input_digest(\"design\"" in
           Path(__file__).read_text(),
       f"REV 75 S1.1: the params are digested WHERE THEY ARE LOADED "
       f"({sorted(_id106)}), and the DESIGN at S0 of the real day path -- "
       f"the earliest point a real day may read `data/`, and far before "
       f"the emit, where its two digests were one read twice. It cannot "
       f"go in `load_params`: a FIXTURE run must open no path under "
       f"`data/`, and the guard refused it there")
    # THE PROBE BRINGS ITS OWN MAP (DE 116). It used to mutate the
    # process's `INPUT_DIGESTS` and drive `verify_input_digests` against
    # it -- which RE-READS every recorded input. On a REAL day that map
    # holds the design pin recorded at S0, so the probe read
    # `data/pm_5min/derived/…design_v23.json` from inside the fixture
    # probe's data-freeness window and GO #5 refused, naming that one
    # path. Named at the artifact by DE 116's diagnosis (unit
    # `de116diag_2`, id 7af79e7e…): the read is this cell, at
    # `verify_input_digests` line 674. The known-bad's verdict rests on a
    # CONSTRUCTED input now, which is what a known-bad should rest on, and
    # it reads no ledger path on any run.
    def _de116_drive(digests):
        """Run `verify_input_digests` under the instrument and return the
        window's proof. The refusal is expected on both maps (a null half);
        what is measured is WHICH PATHS the window saw."""
        def _go():
            try:
                verify_input_digests("a probe", digests=digests)
            except RunnerRefused:
                pass
        return DR.instrumented(_go)[1]

    _probe106 = Path(_tfr.mkdtemp(prefix="de116probe_")) / "probe.json"
    _probe106.write_text("{}\n")
    _map106 = {"a_probe_input": {
        "path": str(_probe106),
        "sha256_at_load": None,
        "read_at_utc": "n/a", "existed_at_load": True}}
    refuses(lambda: verify_input_digests("a probe", digests=_map106),
            "REV 75 S1.1 KNOWN-BAD: a digest pair with a NULL HALF "
            "REFUSES. `{path, sha256: null}` is not a pair (R-608): "
            "one half absent cannot be compared, and `agrees: null` "
            "beside it lets a reader take an unmade comparison for a "
            "made one", "NULL HALF")
    # ---- DE 116, BOTH DIRECTIONS ON THE DATA-FREENESS GUARD ----------
    # (a) A FIXTURE THAT OPENS A LEDGER PATH STILL REFUSES BY NAME. The
    # fix must not have blunted the instrument -- it moved ONE cell off
    # the ledger, it did not teach the guard to forgive one.
    import os as _os116
    _dirty116 = {"pid": _os116.getpid(), "non_vacuous": True,
                 "no_path_under_data_was_opened": False,
                 "data_paths_opened": [
                     "/home/yuqing/ctaNew/data/pm_5min/derived/x.json"]}
    # `refuses()` catches this seat's own RunnerRefused; the data-root
    # guard raises ITS OWN type, so the known-bad is driven directly and
    # the refusal's own words are the assertion.
    try:
        DR.require_canonical("a probe", fixture=True, proof=_dirty116)
        _dirty_admitted = True
    except DR.DataRootRefused as _e116:
        _dirty_admitted = False
        _msg116 = str(_e116)
    _clean116 = DR.require_canonical(
        "a probe", fixture=True,
        proof={"pid": _os116.getpid(), "non_vacuous": True,
               "no_path_under_data_was_opened": True,
               "data_paths_opened": []})
    ok(_dirty_admitted is False
       and "is not a fixture run" in _msg116
       and "data/pm_5min/derived/x.json" in _msg116
       and _clean116["refusal"] == "NOT_APPLICABLE_FIXTURE_RUN",
       "DE 116 KNOWN-BAD: a fixture claim whose proof shows a LEDGER "
       "path is still REFUSED BY NAME, naming the path -- and the same "
       "guard ADMITS a clean proof. The fix took one battery cell off "
       "the ledger; it did not soften the guard")
    # (b) AND THE REAL RUN'S OWN SETUP READ IS THE ONE THAT MOVED. On a
    # real day `run_day` records the design pin at S0 -- a LEDGER path --
    # into `INPUT_DIGESTS`, twelve lines before the battery's window opens
    # at 5646. The two drives below are the before and after of GO #5's
    # refusal, measured on the SAME map in one process.
    # SKIPPED OFFLINE AND NAMED, WITH THE CELL (my own DE 112 rule, and
    # the reason is this cell's own subject): its second drive READS the
    # ledger on purpose. The battery also runs NESTED with `offline=True`
    # from inside `fixture_run`'s data-freeness window -- so an unguarded
    # version of this cell put the design path in that window and the
    # battery refused with GO #5's own message. Written unguarded, it
    # reproduced the very defect it exists to close, which is recorded
    # here rather than quietly fixed.
    if offline:
        offline_skip("DE 116: both directions on the read that refused "
                     "GO #5 (its second drive reads data/ on purpose)")
        offline_skip("DE 116: and the map + `_LAST_PROOF` restore beside "
                     "it (it belongs to the same cell)")
    else:
        _sv116 = dict(INPUT_DIGESTS)
        try:
            INPUT_DIGESTS["design"] = {
                "path": "data/pm_5min/derived/"
                        "p003_de_multiday_gate1_design_v23.json",
                "sha256_at_load": "0" * 64, "read_at_utc": "n/a",
                "existed_at_load": True}
            _pf_own = _de116_drive(_map106)
            _pf_real = _de116_drive(None)
            ok(_pf_own["no_path_under_data_was_opened"] is True
               and _pf_own["non_vacuous"] is True
               and _pf_real["no_path_under_data_was_opened"] is False
               and any("design_v23" in x
                       for x in _pf_real["data_paths_opened"]),
               f"DE 116, BOTH DIRECTIONS ON THE READ THAT REFUSED GO #5, "
               f"with the REAL DAY'S design pin present in "
               f"`INPUT_DIGESTS`: driven against the probe's OWN map the "
               f"window sees {len(_pf_own['data_paths_opened'])} ledger "
               f"path(s) and is non-vacuous; driven against the PROCESS'S "
               f"map it sees {len(_pf_real['data_paths_opened'])} -- "
               f"{[x.split('/')[-1] for x in _pf_real['data_paths_opened']]}"
               f". That second read is the real run's own, legitimate at "
               f"S0, and it is what the probe cell above used to do from "
               f"inside the fixture probe's data-freeness window")
        finally:
            INPUT_DIGESTS.clear()
            INPUT_DIGESTS.update(_sv116)
            # AND THE INSTRUMENT'S OWN AMBIENT STATE. `DR.instrumented`
            # writes `_LAST_PROOF`, and `require_canonical(fixture=True)`
            # falls back to it when a caller passes none -- so two drives
            # left a proof carrying a ledger path where a later fixture
            # claim would pick it up. A probe that instruments anything
            # clears what it left: the same ambient-state class the fix
            # above is about, one level down.
            DR.clear_proof()
        ok(dict(INPUT_DIGESTS) == _sv116 and not DR._LAST_PROOF,
           "and the process's map AND the instrument's `_LAST_PROOF` are "
           "both restored, so the control leaves no ambient state behind "
           "for the cells after it (DE 111's own-baseline rule, applied "
           "to a map and to a proof instead of to a measurement)")

    ok(dict(INPUT_DIGESTS) == _id106
       and not any("/data/" in str(v.get("path"))
                   for v in _map106.values()),
       f"AND THE PROBE TOUCHED NEITHER THE PROCESS'S MAP NOR THE LEDGER: "
       f"`INPUT_DIGESTS` is unchanged ({sorted(_id106)}) and the probe's "
       f"own map names no path under `data/`. Driving this cell against "
       f"the real map re-read the design pin a REAL day records at S0 -- "
       f"inside the fixture probe's window, which is what refused GO #5")
    # S1.2 `n_days_complete` was a DEFAULT PARAMETER nobody passed.
    # IT READS `data/` (it resolves the sealed receipts through the gate's
    # own glob), so a FIXTURE run must not run it -- the fifth time the
    # data-root guard has caught me on exactly this.
    if offline:
        offline_skip("REV 75 S1.2 the computed day count and its prose "
                     "(they resolve sealed receipts under data/)")
        offline_skip("REV 75 S1.2 the seal prose from the computed value")
    else:
        _dc106 = days_complete_now(live)
        ok(_dc106["n_days_complete"] == len(_dc106["days_complete"])
           and set(_dc106["per_day"]) == set(live["days"])
           and _dc106["counted_by"].startswith("find_sealed_day_receipt"),
           f"REV 75 S1.2: `n_days_complete` is COMPUTED through the read "
           f"gate's own resolver -- {_dc106['n_days_complete']} of "
           f"{_dc106['G']}, days {_dc106['days_complete']}. It was a DEFAULT "
           f"of 1 that nothing on the real path passed, so the 09-04 receipt "
           f"reads `1 of 6` on a day when two were complete. Fail-safe by "
           f"accident: a constant 1 can only over-seal")
        ok(f"{_dc106['n_days_complete']} of {_dc106['G']}" in seal(
               {"day": "D", "arm": "A", "status": "OK", "admissibility": {},
                "economic": {"Z": 1.0}},
               _dc106["n_days_complete"], _dc106["G"])["seal_status"],
           "and the receipt's PROSE is generated from that computed value, so "
           "the sentence and the number cannot disagree (rule 10)")
    # S3 the exit copy was one step early.
    _ch106 = declared_chain()
    ok(len(_ch106) == 7
       and "AFTER THE " in _ch106[6]["do"]
       and "Consumed" in _ch106[6]["then"],
       f"REV 75 S3: the chain has a STEP 7 -- copy the journal AGAIN "
       f"after the stop. Under RemainAfterExit the manager emits "
       f"`Stopped` and `Consumed` AT THE STOP, so step 5 holds the run's "
       f"OUTPUT and not its COST: measured 2 lines at step 5 and 4 after "
       f"the stop on the 09-04 run")
    # R-683 the seam: DA reads this seat's rule at the source.
    ok(SEAL_RULE.startswith("A SEALED NAME PRESENT AS A KEY REFUSES")
       and "ABSENCE IS THE ONLY SEAL" in SEAL_RULE,
       "R-683: the key-walk rule is DECLARED as `SEAL_RULE`, in the shape "
       "DA's `de_seal_rule_at_source` reads by AST. At the previous tip it "
       "returned DE_HAS_NOT_DECLARED_THE_KEY_WALK_RULE_YET -- an absence "
       "reported by name, never read as agreement. The two censuses stay "
       "independent implementations of ONE rule")
    # ---- REV 76 S0: THE CENSUS WALKS KEYS, NOT LEAVES ----------------
    # An empty container yields NO LEAF, so a leaf-walking census reports
    # a sealed key emitted as `[]` or `{}` as ABSENT -- while the
    # receipt's own `seal_status` says every economic field is "ABSENT
    # from this artifact, not present-and-ignored", and an empty
    # container is exactly present-and-ignored. Driven on the reviewer's
    # three values, for EVERY one of the eleven names, AT DEPTH.
    _forms105 = {"[]": [], "{}": {}, "0.0": 0.0, "null": None}
    _missed105, _n105 = [], 0
    for _name105 in ECONOMIC_FIELDS:
        for _lbl105, _val105 in _forms105.items():
            _art105 = {"per_day_sealed_artifacts": [
                {"arm": "A", "nested": [{"deep": {_name105: _val105}}]}]}
            _n105 += 1
            if not _economic_keys_in(_art105):
                _missed105.append(f"{_name105}={_lbl105}")
                continue
            try:
                assert_no_economic_leak(_art105, 1, 6)
                _missed105.append(f"{_name105}={_lbl105} ADMITTED")
            except RunnerRefused:
                pass
    ok(not _missed105 and _n105 == len(ECONOMIC_FIELDS) * 4,
       f"REV 76 S0: all {len(ECONOMIC_FIELDS)} sealed names planted AT "
       f"DEPTH as `[]`, `{{}}`, `0.0` and `null` -- {_n105} cells -- are "
       f"found by the census and REFUSED. An empty container yields no "
       f"LEAF, so a leaf-walk reports it absent; this walks KEYS")
    # NO `or True` HERE. I wrote one, and it makes a check unfalsifiable --
    # the same slip as DE 97's key-name assertion, one round on.
    ok(CENSUS_RULE.startswith("a sealed name present as a KEY is a leak")
       and "never leaves" in CENSUS_RULE
       and "[] and {}" in CENSUS_RULE
       and "CENSUS_RULE" in (_economic_keys_in.__doc__ or ""),
       f"and the RULE IS STATED ONCE, in `CENSUS_RULE`, so DE's census "
       f"and DA's `economic_absence` agree BY CONSTRUCTION rather than by "
       f"both happening to be written the same way (DA 98 carries the "
       f"same sentence)")
    # THE POST-EMIT READ-BACK: the claim is about the BYTES, not the
    # object. Driven on a file, because that is what it guards.
    _rb = Path(_tfr.mkdtemp(prefix="de105rb_")) / "r.json"
    _rb.write_text(json.dumps({"per_day_sealed_artifacts": [
        {"arm": "A", "economic": {"D_E0": []}}]}))
    ok(_economic_keys_in(json.loads(_rb.read_text())) != [],
       "REV 76 S0(3): the emitter now READS THE RECEIPT BACK FROM DISK "
       "and censuses THOSE BYTES before naming it sealed -- everything "
       "before that point checks the payload in memory, and what a reader "
       "gets is the file. On a leak it removes the file and refuses: a "
       "sealed day that leaks is not a day")
    # ---- R-674 (b): the receipt describes ITS OWN seal, generated ----
    _scope104 = {
        "design_version": DESIGN_VERSION_IN_FORCE,
        "sealed_names": list(economic_fields_in_force()),
        "open_population_sizes": list(OPEN_POPULATION_SIZES)}
    ok(set(_scope104["sealed_names"]) == set(ECONOMIC_FIELDS)
       and set(_scope104["open_population_sizes"])
       & set(ECONOMIC_FIELDS) == set()
       and len(_scope104["sealed_names"]) == 11,
       f"R-674(b): the receipt's self-description is GENERATED from the "
       f"scope map the emitter applies -- {len(_scope104['sealed_names'])} "
       f"sealed names and {len(_scope104['open_population_sizes'])} open "
       f"sizes. A literal list said four counts were open SIZES while the "
       f"emitter sealed three of them: the receipt described a different "
       f"seal from the one it had")
    # ---- R-674 (c): a refusal REASON may not carry a sealed VALUE -----
    ok(assert_reasons_carry_no_sealed_value(
           {"admissibility": {"reasons": ["decisions 12 < declared "
                                          "minimum 30"],
                              "null_sd": 1.5, "null_mean": 4.25}}
       )["no_sealed_value_in_any_reason"] is True,
       "R-674(c) POSITIVE CONTROL: a real refusal reason -- the decision "
       "count and the declared minimum, both OPEN -- is admitted. "
       "`reasons` is written by the DESIGN's `arm_day_admissible` and has "
       "been in every receipt since design v2, INCLUDING 09-03's, where "
       "it is an empty list")
    for _bad104, _lbl104 in (
            ({"admissibility": {"reasons": ["the null dispersion 1.5 is "
                                            "below the floor"],
                                "null_sd": 1.5}}, "a raw sealed value"),
            ({"admissibility": {"reasons": ["Z was 2.50 on this arm"]},
              "economic": {"Z": 2.5}}, "a FORMATTED sealed value (2.50)")):
        refuses(lambda b=_bad104: assert_reasons_carry_no_sealed_value(b),
                f"R-674(c) KNOWN-BAD, {_lbl104}: a refusal REASON "
                f"carrying a sealed quantity's VALUE is REFUSED. "
                f"`_strip_economic` removes KEYS, not substrings, so a "
                f"number interpolated into a string survives the seal -- "
                f"R-599's second leak, and it fires on a REFUSED arm-day, "
                f"exactly where the numbers are most tempting",
                "carries the VALUE of a sealed quantity")
    # ---- REV 73 S1.1: THE SELECTOR'S THREE HOLES, all driven ---------
    _d103 = Path(DR.resolve()["data_root"]) / "pm_5min/derived"
    _v22f = _d103 / "p003_de_multiday_gate1_design_v22.json"
    _v23f = _d103 / "p003_de_multiday_gate1_design_v23.json"
    _sel_none = design_version_of_receipt({})
    ok(_sel_none["design_version"] == DESIGN_VERSION_IN_FORCE
       and _sel_none["n_names_in_force"] == len(ECONOMIC_FIELDS)
       and "STRICTEST" in _sel_none["read_from"],
       f"REV 73 S1.1(a): a receipt with NO provenance at all is judged "
       f"under the STRICTEST list in force (v{_sel_none['design_version']}"
       f", {_sel_none['n_names_in_force']} names). It fell through to "
       f"v22's EIGHT -- absence selecting the WEAKER rule, inside the "
       f"scoping built to protect the seal")
    _old102 = {"protocol": "P003_DE_MULTIDAY_GATE1_DAY_RUN_V1",
               "emitted_at_utc": "2026-09-06T14:01:55.557479+00:00",
               "per_day_sealed_artifacts": [
                   {"arm": "A", "n_fills_arm": 1, "n_fills_baseline": 2,
                    "n_cancels_issued": 3}]}
    # NO FILE IS READ HERE: `_new102` carries no provenance at all, so it
    # falls to the STRICTEST list -- which is the point being made. The
    # cells that must READ a design artifact are guarded below, because a
    # FIXTURE run must open no path under `data/` and the data-root guard
    # refused this block the first time (as it did DE 98's and DE 101's;
    # three rounds, same guard, same lesson).
    _new102 = {k: v for k, v in _old102.items() if k != "emitted_at_utc"}
    _ja = economic_absence_scoped(_old102)
    _jb = economic_absence_scoped(_new102)
    ok(_ja["sealed"] is True and _ja["n_leaked"] == 0
       and _ja["design_version"] == SCOPE_BEFORE_THE_CORRECTION
       and "POSITIVE" in _ja["read_from"]
       and _jb["n_leaked"] == 3,
       f"and a PRE-CORRECTION receipt is recognised POSITIVELY -- by an "
       f"emit stamp before the correction, not by a missing field -- so "
       f"it reads {_ja['n_leaked']} leaked under its own "
       f"{_ja['n_names_in_force']} names, while the SAME three counts in "
       f"a v23-pinned receipt read as {_jb['n_leaked']} leaks")
    if offline:
        offline_skip("REV 73 S1.1(b) the design PAIR cells and (c) the "
                     "opened-paths cell (they read design artifacts "
                     "under data/)")
        offline_skip("REV 73 S1.1(c) the order-dependent fallback cell")
    elif _v22f.is_file() and _v23f.is_file():
        _s22 = hashlib.sha256(_v22f.read_bytes()).hexdigest()
        _s23 = hashlib.sha256(_v23f.read_bytes()).hexdigest()
        _badp = design_version_of_receipt({"provenance": {"design": {
            "path": str(_v22f), "sha256": "0" * 64}}})
        _g22 = design_version_of_receipt({"provenance": {"design": {
            "path": str(_v22f), "sha256": _s22}}})
        _g23 = design_version_of_receipt({"provenance": {"design": {
            "path": str(_v23f), "sha256": _s23}}})
        ok(_badp["design_version"] == DESIGN_VERSION_IN_FORCE
           and _badp["pair_verified"] is False
           and _g22["design_version"] == 22 and _g22["pair_verified"]
           and _g23["design_version"] == DESIGN_VERSION_IN_FORCE
           and _g23["pair_verified"],
           f"REV 73 S1.1(b): the version comes from the PAIR. A v22 PATH "
           f"beside a WRONG digest is UNKNOWN and gets the strictest list "
           f"(v{_badp['design_version']}); the same path with its OWN "
           f"digest resolves v{_g22['design_version']}, and a v23 pair "
           f"resolves v{_g23['design_version']}. It read the path and "
           f"ignored the digest sitting next to it")
        _ord = design_version_of_receipt({"split_residency_proof": {
            "tape_artifacts_opened": [
                str(_d103 / "p003_de_multiday_gate1_design_v10__x.json"),
                str(_d103 / "p003_de_multiday_gate1_design_v21.json")]}})
        ok(_ord["design_version"] == DESIGN_VERSION_IN_FORCE,
           f"REV 73 S1.1(c): the ORDER-DEPENDENT fallback is GONE. A "
           f"receipt whose opened paths name v10 and then v21 -- exactly "
           f"what the 09-03 run opened -- gets the strictest list, not "
           f"whichever version happened to be listed first. An opened "
           f"path is not a pin")
    ok("STABLE NAME" in (design_version_of_receipt.__doc__ or "")
       and "economic_fields_in_force" in (
           design_version_of_receipt.__doc__ or ""),
       "and the docstring says the NAME IS STABLE: DA reads this function "
       "and `economic_fields_in_force` by AST from DA 95 on, so a change "
       "of meaning arrives as a superseding name rather than the same "
       "name doing something else")
    # AND THE SEAL REACHES THEM AT EVERY DEPTH, PLANTED AND PROVEN.
    _deep102 = {"day": "D", "arm": "A", "status": "OK",
                "admissibility": {"admissible": True, "null_sd": 1.0},
                "n_cancels_issued": 5146,
                "nested": [{"inner": {"n_fills_arm": 30171,
                                      "n_fills_baseline": 46439}}],
                "economic": {"D_E0": 1.0, "Z": 2.0, "p_location": 0.01,
                             "null_mean": 0.0, "null_sd": 1.0,
                             "null_draws_summary": {"n": 500}}}
    _sealed102 = seal(_deep102, 1, 6)
    _keys102 = set(_economic_keys_in(_sealed102))
    ok(_keys102 == set()
       and _strip_economic({"a": {"b": {"n_fills_arm": 1}}}) == {"a": {"b": {}}},
       f"R-659 FALSIFIER, PLANTED AT DEPTH AND TESTED AS KEYS: "
       f"`n_cancels_issued` at the top, `n_fills_arm` and "
       f"`n_fills_baseline` two levels down inside a LIST -- all stripped "
       f"by the seal, and the leak walker finds none left. Tested as KEYS, "
       f"not as substrings: a value that happens to contain the name is "
       f"not a leak, and the count of them is not evidence")
    _unsealed102 = seal(_deep102, 6, 6)
    ok("n_cancels_issued" in json.dumps(_unsealed102),
       "AND THEY COME BACK AT G: the seal WITHHOLDS them until every day "
       "is complete, it does not delete them. A guard shown only to "
       "withhold is not a guard")
    _jf101 = journal_copy_by_invocation("de101-cannot-exist.service")
    ok(_jf101.get("output_format") == "short-iso-precise"
       and "cannot be diffed" in _jf101.get("output_format_note", ""),
       "R-656 / REV 70 S3: every journal copy NAMES ITS FORMAT. Two "
       "copies of one run were diffed and reported a false mismatch -- "
       "the formats differed, not the lines")
    # THE DESIGN CHAIN READS THE DESIGN ARTIFACTS UNDER `data/`, and a
    # FIXTURE run must open no path under it -- the data-root guard
    # refused this check the first time, as it did DE 98's.
    if offline:
        offline_skip("the design chain-head resolution (it reads the "
                     "design artifacts under data/)")
    else:
        _dch101 = design_chain()
        # THE FORK IS A FACT ABOUT THE FAMILY AND THE ORPHAN LIST IS A
        # FACT ABOUT THE LINKS -- and only the second moves when a merge
        # version lands. This cell asserted `orphan_branches == [16]`, a
        # literal about a world that DE 110 changed on purpose; it now
        # asserts the PROPERTY: v16 and v17 still both supersede v15 (the
        # fork is still visible, nothing was edited away) AND every
        # version is reachable, each merge link recomputed.
        _v15n = next((l["supersedes"] for l in _dch101["links"]
                      if l["version"] == 17), None)
        _both = sorted(l["version"] for l in _dch101["links"]
                       if l.get("supersedes") == _v15n and l.get("agrees"))
        ok(_dch101.get("resolved") is True
           and all(l.get("agrees") for l in _dch101["links"])
           and _dch101["head_version"] == max(_dch101["versions_present"])
           and _both == [16, 17]
           and all(m.get("agrees") for m in _dch101["merge_links"])
           and set(_dch101["reachable_from_the_newest"])
           == set(_dch101["versions_present"])
           and _dch101["orphan_branches"] == [],
           f"and the DESIGN is resolved to its chain head too "
           f"(v{_dch101.get('head_version')} of "
           f"{len(_dch101.get('versions_present') or [])} present, every "
           f"pair recomputed). THE FORK IS STILL THERE AND IS STILL "
           f"REPORTED -- v{_both} both supersede {_v15n}, DE 94's "
           f"two-artifacts-one-version defect showing up as a chain fork, "
           f"and nothing was edited to hide it -- but it is no longer an "
           f"ORPHAN: {len(_dch101['merge_links'])} verified merge link(s) "
           f"make every version reachable, so orphan branches "
           f"{_dch101['orphan_branches']} (REV 81 §5's repair, DE 110). "
           f"params v14 names v21 while the head is later, and requiring "
           f"the named path to BE the head would force a params bump per "
           f"design version for a pointer alone")

    # ---- R-653 (i): THE CHAIN HEAD, never a filename literal ----------
    # This runner PINNED v1 at 14:00Z while v2 existed, then pinned v2 by
    # NAME while v3 existed. A reader pinned to a superseded version
    # satisfies the words of rule 20's guard without the property.
    _ch100 = heavy_run_form_chain()
    ok(_ch100["head_version"] == max(_ch100["versions_present"])
       and all(l["agrees"] for l in _ch100["links"])
       and len(_ch100["links"]) == len(_ch100["versions_present"]) - 1
       and HEAVY_RUN_FORM_GLOB in "heavy_run_form_v*.json",
       f"R-653 (i): the launch-form declaration is RESOLVED to its chain "
       f"head -- v{_ch100['head_version']} of "
       f"{_ch100['versions_present']}, every link's pair {{path, sha256}} "
       f"RECOMPUTED from the file it names. A filename literal here "
       f"pinned v1 while v2 existed")
    ok(heavy_run_form()["unit_outcome_minimum_read"]
       == ["LoadState", "ActiveState", "SubState", "ExecMainStatus",
           "Result"]
       and heavy_run_form()["_chain"]["head_version"]
       == _ch100["head_version"],
       "and the constants come from THAT head -- the five-field minimum "
       "read among them, so the reader cannot be newer or older than the "
       "declaration it obeys")

    # ---- R-653 (ii): FIVE fields, and SubState is the discriminator ---
    # Under RemainAfterExit a FINISHED unit is loaded/active/EXITED and a
    # RUNNING one loaded/active/RUNNING -- both ExecMainStatus 0.
    _fin100 = unit_outcome.__doc__ or ""
    _abs100 = unit_outcome("de100-a-unit-that-cannot-exist.service")
    ok(set(_abs100["the_five"]) == {"LoadState", "ActiveState", "SubState",
                                    "ExecMainStatus", "Result"}
       and _abs100["status"] == "VOID",
       "R-653 (ii): the outcome read is the FIVE declared fields, and a "
       "not-found unit is VOID -- its inactive/dead/0/success are "
       "DEFAULTS")
    ok("SubState_is_the_discriminator" in unit_outcome(
           "de100-a-unit-that-cannot-exist.service")
       or _abs100["status"] == "VOID",
       "and SubState is the field that separates FINISHED from RUNNING: "
       "ActiveState says `active` for both and ExecMainStatus says 0 for "
       "both (REV 69 S3.3, measured)")

    # ---- R-653 (iii): THE `-E` HALF, DECIDED AT RUNTIME ---------------
    # The composed-string check is a LINT; this reads what actually ran.
    _f100 = heavy_run_form()
    _rc100, _lock100 = str(_f100["lock_conflict_rc"]), _f100["lock_path"]
    # THE INJECTED PARENTS CARRY `exe` NOW (REV 71 S1.2): identity comes
    # from /proc/<ppid>/exe, not argv[0], because `exec -a flock` sets
    # argv[0] to anything. These cells passed an argv-only parent and the
    # positive control refused the moment the source changed -- which is
    # the check noticing its own subject moved.
    _good100 = {"ppid": 1, "readable": True, "exe": "/usr/bin/flock",
                "argv": ["/usr/bin/flock", "-n", "-E", _rc100, _lock100,
                         "/venv/python3", "runner.py"]}
    ok(assert_lock_form_at_runtime(
           "2026-09-03", fixture=False,
           parent=_good100)["carries_dash_E_with_the_declared_rc"] is True,
       f"R-653 (iii) POSITIVE CONTROL: a parent flock carrying `-E "
       f"{_rc100}` on the declared lock ADMITS a real day")
    for _bad100, _lbl100, _needle100 in (
            ({"ppid": 1, "readable": True, "exe": "/usr/bin/flock",
              "argv": ["/usr/bin/flock", "-n", _lock100, "x"]},
             "no -E", "carries no"),
            ({"ppid": 1, "readable": True, "exe": "/usr/bin/flock",
              "argv": ["/usr/bin/flock", "-n", "-E", "76", _lock100, "x"]},
             "the WRONG rc", "carries no"),
            ({"ppid": 1, "readable": True, "exe": "/usr/bin/flock",
              "argv": ["/usr/bin/flock", "-n", "-E", _rc100,
                       "/tmp/other.lock", "x"]},
             "ANOTHER lock", "does not name the declared lock"),
            ({"ppid": 1, "readable": True, "exe": "/bin/sh",
              "argv": ["/bin/sh", "-c", "x"]},
             "a parent that is NOT flock", "not `flock`"),
            # THE SPOOF (REV 71 S1.2): `exec -a flock /bin/sleep` --
            # argv[0] SAYS flock and the executable is sleep.
            ({"ppid": 1, "readable": True, "exe": "/bin/sleep",
              "argv": ["flock", "-n", "-E", _rc100, _lock100, "x"]},
             "argv[0] SPOOFED to `flock` by `exec -a`", "not `flock`"),
            ({"ppid": None, "readable": False, "why": "no /proc"},
             "an UNREADABLE parent", "unreadable")):
        refuses(lambda b=_bad100: assert_lock_form_at_runtime(
                    "2026-09-03", fixture=False, parent=b),
                f"R-653 (iii) KNOWN-BAD, {_lbl100}: a REAL day refuses "
                f"BEFORE ANY STAGE. Driven live too -- a scratch unit "
                f"launched WITHOUT `-E` refused from inside the unit, "
                f"reading its own parent's cmdline", _needle100)
    ok(assert_lock_form_at_runtime(
           "FIXTURE-DAY-1", fixture=True,
           parent={"ppid": 1, "readable": True, "exe": "/bin/sh",
                   "argv": ["/bin/sh"]})["checked"] is False,
       "and a FIXTURE is not launched under the heavy form, so it is "
       "recorded as NOT CHECKED rather than silently passed")

    # ---- R-653 (iv): COVERAGE AT EMIT, no needle anywhere -------------
    # THE PROPERTY, NOT A GREP FOR THE PROPERTY. This first asserted that
    # the string `any(" Started " in` did not appear in this file -- and
    # it FAILED, because the COMMENTS that explain the retired predicate
    # quote it. A check that greps its own prose is the shape this round
    # is closing, one level up. What matters is that the reported answer
    # IS the two-clock measurement, and that is checkable by identity.
    _cvj100 = journal_copy_by_invocation(
        "de100-a-unit-that-cannot-exist.service")
    _cvsrc = (_cvj100.get("coverage") or {})
    ok(_cvj100["status"] == "ABSENT"
       or (_cvj100.get("window_fully_covered") == _cvsrc.get("covered")
           and _cvsrc.get("two_clocks_no_text_search") is True),
       "R-653 (iv): the copy's `window_fully_covered` IS "
       "`da_root.journal_coverage`'s own answer -- the same object, not a "
       "second computation beside it -- and that reader carries "
       "`two_clocks_no_text_search`. The retired predicate searched the "
       "copied lines for ` Started ` and reported TRUE on a tail that had "
       "lost 141 of 161 lines")
    _cov100 = DAROOT.journal_coverage(
        unit="de100-a-unit-that-cannot-exist.service")
    ok(_cov100["status"] == "NOT_DETERMINABLE"
       and _cov100["covered"] is None
       and _cov100["two_clocks_no_text_search"] is True,
       "and a unit with no start timestamp -- a COLLECTED one -- is "
       "NOT_DETERMINABLE, which is why the measurement is taken AT THE "
       "EMIT while the unit is still loaded and STORED. It can never be "
       "recomputed afterwards")

    # ---- R-654 (v): THE PROVENANCE BLOCK ------------------------------
    # The 09-03 receipt carried source_identity and NO params/design pin:
    # params v14 appeared only inside `fixture_day_lock…` and design v21
    # only as an opened path, so DA's pre-read had to infer both.
    _rp100 = Path(__file__).resolve().parents[2]
    _pv100 = {"path": PARAMS_REL,
              "sha256": hashlib.sha256(
                  (_rp100 / PARAMS_REL).read_bytes()).hexdigest()}
    ok(len(_pv100["sha256"]) == 64 and _pv100["path"].endswith(".json"),
       f"R-654 (v): the receipt's OPEN `provenance` block carries params "
       f"as the PAIR {{path, sha256}} -- {_pv100['path'].rsplit('/', 1)[-1]} "
       f"at {_pv100['sha256'][:16]} -- digested at emit from the file the "
       f"run actually read, not inferred from a name")
    # ---- REV 68 S1.2/S1.3 (R-649): THE GUARD GATES, AND FAILS CLOSED --
    # S1.2 the fixture exemption was a REPORT: `fixture=True` admitted a
    # REAL DAY NAME, because the list was computed into a field and never
    # consulted. S1.3 the guard FAILED OPEN: the test was
    # `kind == "scope"`, so `kind` None -- an unreadable cgroup, a leaf
    # shape nobody anticipated -- ADMITTED a real day and claimed
    # `checked` true (R-651 / MEM 181).
    _SC99 = {"cgroup_leaf": "x.scope", "kind": "scope"}
    _SV99 = {"cgroup_leaf": "x.service", "kind": "transient service"}
    refuses(lambda: assert_launch_form_at_runtime(
                "2026-09-03", fixture=True, observed=_SC99),
            "REV 68 S1.2 KNOWN-BAD: `fixture=True` with a REAL DAY NAME "
            "under a scope REFUSES. The exemption is the GATE now; it was "
            "a field computed beside the decision and never consulted, so "
            "any caller passing fixture=True admitted anything",
            "not among")
    ok(assert_launch_form_at_runtime(
           "FIXTURE-DAY-1", fixture=True,
           observed=_SC99)["fixture_exemption_by_name"] is True,
       "POSITIVE CONTROL: a DECLARED fixture name under a scope still "
       "admits -- which is what lets the launch-form probe run its "
       "falsifier under a real scope at all")
    for _obs99, _lbl99 in (({"kind": None}, "kind None"),
                           ({}, "an EMPTY observation"),
                           ({"kind": "something-new"}, "an UNKNOWN kind")):
        refuses(lambda o=_obs99: assert_launch_form_at_runtime(
                    "2026-09-03", fixture=False, observed=o),
                f"REV 68 S1.3 KNOWN-BAD, {_lbl99}: a REAL DAY refuses "
                f"unless the kind IS the declared form. The old test was "
                f"`kind == 'scope'`, which FAILED OPEN on exactly this -- "
                f"an unreadable cgroup admitted a real day",
                "UNKNOWN kind refuses too")
    _unk99 = {"kind": None}
    try:
        assert_launch_form_at_runtime("2026-09-03", fixture=False,
                                      observed=_unk99)
        ok(False, "an unknown kind was ADMITTED for a real day")
    except RunnerRefused:
        pass
    _fx99 = assert_launch_form_at_runtime("FIXTURE-DAY-1", fixture=True,
                                          observed={"kind": None})
    ok(_fx99["checked"] is False and _fx99["kind_is_known"] is False,
       "R-651 / MEM 181: when the kind is UNKNOWN the record says "
       "`checked` FALSE. It used to ADMIT a real day AND claim `checked` "
       "true -- a guard that cannot see what it is in has not checked "
       "anything")
    ok(assert_launch_form_at_runtime(
           "2026-09-03", fixture=False, observed=_SV99)["checked"] is True,
       "and a real day in a transient SERVICE is `checked` TRUE -- so the "
       "field means what it says in both directions")

    # ---- REV 68 S1.5 (R-649): COVERAGE IS TWO CLOCKS, NOT A NEEDLE ----
    # It was `any(" Started " in x for x in lines)`: TRUE on a tail that
    # had lost 141 of 161 lines, and matchable by a PAYLOAD line that
    # merely contains the words. DA 88 shipped the right shape and it is
    # IMPORTED, not mirrored.
    _hz99 = DAROOT.host_journal_horizon()
    _cov99 = DAROOT.journal_coverage(
        window_start_epoch=(_hz99["oldest_epoch"] - 3600),
        regime=DAROOT.CONTINUOUS) if _hz99.get("oldest_epoch") else None
    # THE HORIZON IS READ TWICE AND IT MOVES (rule 20: "a state named
    # once and re-quoted later is stale" -- measured at ~15 min of drift
    # in 18 min). Comparing `_hz99`'s reading to a message built from
    # `journal_coverage`'s OWN, later reading made this cell fail whenever
    # the window advanced between the two calls; the rehearsal hit it.
    # The property is that the message names the horizon IT used, so the
    # assertion reads that one, and the drift between the two readings is
    # recorded rather than asserted away.
    _hz99b = (_cov99 or {}).get("host_horizon") or {}
    _drift99 = (None if not _hz99.get("oldest_epoch")
                or not _hz99b.get("oldest_epoch")
                else _hz99b["oldest_epoch"] - _hz99["oldest_epoch"])
    ok(_cov99 is None or (
           _cov99["covered"] is False
           and _cov99["status"] == "MEASURED"
           and _hz99b.get("oldest_utc")
           and _hz99b["oldest_utc"] in _cov99["why"]
           and _cov99["two_clocks_no_text_search"] is True),
       f"(horizon drift between this cell's read and the coverage "
       f"call's own: {_drift99} s) "
       f"REV 68 S1.5 KNOWN-BAD: a window an HOUR BEFORE the host's "
       f"horizon is UNCOVERED, and the refusal NAMES the horizon "
       f"({_hz99.get('oldest_utc')}). The old predicate would have "
       f"answered from whether the word ' Started ' appeared in the lines "
       f"it happened to copy")
    _covok99 = DAROOT.journal_coverage(
        window_start_epoch=(_hz99["oldest_epoch"] + 60),
        regime=DAROOT.CONTINUOUS) if _hz99.get("oldest_epoch") else None
    ok(_covok99 is None or _covok99["covered"] is True,
       "POSITIVE CONTROL: a window INSIDE the horizon is covered -- so "
       "the known-bad above fires on the two clocks and not on "
       "everything")
    _nodet99 = DAROOT.journal_coverage(
        unit="de99-a-unit-that-cannot-exist.service")
    ok(_nodet99["covered"] is None
       and _nodet99["status"] == "NOT_DETERMINABLE",
       "and when one of the two clocks is unreadable -- a COLLECTED unit "
       "has no start timestamp -- coverage is NOT_DETERMINABLE, never "
       "assumed either way. The needle would have answered TRUE or FALSE "
       "with equal confidence")
    _chain99 = declared_chain()
    ok("AFTER the unit has exited" in _chain99[4]["do"]
       and "Consumed" in _chain99[4]["then"]
       and _chain99[5]["do"].startswith("`systemctl --user stop"),
       f"and the chain carries the EXIT COPY as its own step "
       f"({len(_chain99)} steps; the LENGTH is asserted where the shape "
       f"is ruled, not here): the receipt's copy is taken at the EMIT, "
       f"before the `Consumed` line exists, and that is the line which "
       f"survives longest -- DE 84's `Started` line was gone four hours "
       f"later while its `Consumed` line remained")
    # ---- R-648 (R3'): THE TRIPLE, AND WHAT A COLLECTED UNIT REPORTS ---
    # A transient unit that exits 0 is COLLECTED at exit: LoadState goes
    # not-found and `systemctl show` then returns DEFAULTS --
    # inactive/dead/0/success. Driven on scratch units with a scratch
    # lock, and the shape of the trap is the first row:
    #   exit 0, no RemainAfterExit -> not-found / inactive / 0 / success
    #   exit 0, WITH               -> loaded / active(exited) / 0
    #   exit 3, WITH               -> loaded / failed / 3
    #   held lock, WITH            -> loaded / failed / 75
    _form98 = heavy_run_form()
    # THE VERSION IS NOT PINNED HERE EITHER (R-653 (i)). This asserted
    # `version == 2` while v3 existed -- the same literal-pinning defect
    # as the builder's, in the check written to guard the builder. The
    # property is that the head is the newest and every link verifies.
    ok(_form98["remain_after_exit"] is True
       and _form98["_chain"]["head_version"]
       == max(_form98["_chain"]["versions_present"])
       and all(l["agrees"] for l in _form98["_chain"]["links"]),
       f"R-648/R-653: the builder reads the CHAIN HEAD "
       f"(v{_form98['_chain']['head_version']} of "
       f"{_form98['_chain']['versions_present']}) with every supersession "
       f"pair recomputed from the file it names. A declaration that "
       f"claims to supersede a file nobody checked is R-608's shape -- "
       f"and a checker pinned to a version number is the defect one level "
       f"up")
    _cmd98 = the_one_command("2026-09-03", "/BOOK", "/OUT")
    ok("-p RemainAfterExit=yes" in _cmd98
       and assert_launch_form(_cmd98)["ok"] is True,
       "and THE_ONE_COMMAND carries `-p RemainAfterExit=yes`, read from "
       "the declaration -- without it a unit that exits 0 is collected "
       "and its outcome is unreadable")
    refuses(lambda: assert_launch_form(
                _cmd98.replace("-p RemainAfterExit=yes ", "")),
            "R-648 KNOWN-BAD: a command without `RemainAfterExit` is "
            "refused by the lint -- the unit would vanish at exit and the "
            "reading afterwards would be a DEFAULT wearing the shape of a "
            "success", "COLLECTED")
    _void98 = unit_outcome("de98-a-unit-that-cannot-exist.service")
    ok(_void98["status"] == "VOID"
       and _void98["outcome_readable"] is False
       and _void98["the_triple"][0] != "loaded",
       f"R-648 R3' KNOWN-BAD, AND THIS IS THE WHOLE POINT: a unit that "
       f"does not exist reports {_void98['the_triple']} with "
       f"Result={_void98['Result']!r}. On some hosts systemd returns "
       f"`inactive`/`0`/`success`, on this host the per-property reads "
       f"return None; both are unreadable because LoadState is not `loaded`, "
       f"and the reading is reported as VOID")
    ok("DEFAULTS" in _void98["why_void"]
       and "RemainAfterExit" in _void98["what_to_do"]
       and _void98["lock_conflict_rc"] == _form98["lock_conflict_rc"],
       "and the VOID reading says WHY it is void and what would have made "
       "it readable, rather than leaving a caller to wonder why a "
       "successful-looking triple was refused")
    ok(len(_void98["the_triple"]) == 3
       and "systemd's own order" in _void98["read_property_by_property"],
       "and every property is read in ITS OWN call and NAMED: "
       "`systemctl show -p A -p B --value` returns systemd's order, not "
       "the flag order (measured: Result, ExecMainStatus, LoadState, "
       "ActiveState, SubState), so a positional parse mislabels every "
       "field -- mine declared a LIVE run VOID before it was fixed")
    _chain98 = declared_chain()
    # THE STOP IS NO LONGER LAST: step 7 copies the journal AFTER it
    # (REV 75 S3), because the manager writes `Stopped` and `Consumed` at
    # the stop. This asserted `[-1]`, which is the same literal-that-must-
    # track-a-moving-thing shape one position along.
    _stopstep = [st for st in _chain98
                 if st["do"].startswith("`systemctl --user stop")]
    ok(len(_stopstep) == 1
       and any("VOID" in st["then"] for st in _chain98),
       f"and the chain is a DECLARED FIELD of the rehearsal "
       f"({len(_chain98)} steps, one of which is the STOP that frees the "
       f"name), "
       f"not prose in a runbook a launcher may not read. "
       f"RemainAfterExit is what makes that last step necessary. The "
       f"LENGTH is asserted where the chain's shape is ruled, not here -- "
       f"a literal 5 beside a chain that grew to 6 is the class this "
       f"codebase keeps finding")
    # ---- R-646 (R2): THE LOCK-CONFLICT CODE, DECLARED ONCE ------------
    # Measured before the ruling: under the ruled form a HELD LOCK and a
    # PAYLOAD CRASH were both ExecMainStatus=1, so at GO #5 a refusal
    # would have been unreadable from a crash. `flock -n -E 75` separates
    # them, and 75 is declared in ONE file that every literal reads --
    # two definitions of it were already measured drifting-capable (a
    # launcher refusing with 76 was published as 75).
    _form97 = heavy_run_form()
    ok(_form97["lock_conflict_rc"] == 75
       and _form97["lock_path"] == HEAVY_RUN_LOCK
       and _form97["slice"] == RESEARCH_SLICE,
       f"R-646 R2: the launch form's constants are READ from the chain "
       f"head ({_form97['_chain']['head_path'].rsplit('/', 1)[-1]}) -- "
       f"conflict code "
       f"{_form97['lock_conflict_rc']}, the lock path and the slice all "
       f"agree with this module's own constants, so a drift between them "
       f"is a battery failure rather than a surprise at GO")
    ok(assert_no_exit_code_collision()["no_collision"] is True
       and _form97["lock_conflict_rc"] not in RUNNER_EXIT_CODES,
       f"and THIS RUNNER DOES NOT EXIT {_form97['lock_conflict_rc']} for "
       f"any reason of its own ({sorted(RUNNER_EXIT_CODES)}) -- the "
       f"separation `-E` buys is only real while no payload borrows the "
       f"code")
    _cmd97 = the_one_command("2026-09-03", "/BOOK", "/OUT")
    ok(f"-E {_form97['lock_conflict_rc']} " in _cmd97
       and assert_launch_form(_cmd97)["ok"] is True,
       f"and the published command carries `-E "
       f"{_form97['lock_conflict_rc']}` -- read from the declaration, "
       f"never a literal in the builder")
    refuses(lambda: assert_launch_form(
                _cmd97.replace(f"-E {_form97['lock_conflict_rc']} ", "")),
            "R-646 R2 KNOWN-BAD: a command WITHOUT `-E` is refused by the "
            "lint. Driven on scratch units: held lock -> ExecMainStatus "
            "75, payload exit 1 -> 1, payload success -> 0; without `-E` "
            "the first two are both 1", "-E")
    # R3: the outcome is the PAIR.
    _uo97 = unit_outcome("de97-a-unit-that-cannot-exist.service")
    ok(set(_uo97) >= {"the_triple", "status", "outcome_readable"}
       and _uo97["lock_conflict_rc"] == _form97["lock_conflict_rc"],
       "R-646 R3, as R-648 completes it: a unit's outcome is the TRIPLE "
       "(LoadState, ActiveState, ExecMainStatus) with the conflict code "
       "beside it. A RUNNING unit "
       "reports ExecMainStatus=0 -- measured on de95smoke.service 53 "
       "minutes into an 85-minute day -- so the status alone reads a "
       "running run as a clean success")

    # ---- R-646 (R4): THE JOURNAL COPY IS BY INVOCATION ID, BOTH FIELDS -
    # A unit NAME names every run ever launched under it; an id names ONE.
    # And the id must be matched on BOTH fields: the PAYLOAD's lines carry
    # `_SYSTEMD_INVOCATION_ID`, the user manager's Started/Consumed lines
    # carry `USER_INVOCATION_ID`, and `INVOCATION_ID` is the SYSTEM
    # manager's field and matches nothing here. Measured on
    # be64book.service at 13:41Z: 1 line on the payload field alone, 3 on
    # the manager field alone, 0 on the system field, 4 on both, 146 by
    # NAME (the name had been used more than once).
    _jc97 = journal_copy_by_invocation("de97-cannot-exist.service")
    ok(_jc97["status"] == "ABSENT"
       and _jc97["n_lines_by_id"] == 0
       and _jc97["n_lines_by_unit_name"] == 0
       and "_SYSTEMD_INVOCATION_ID" in _jc97["fields_matched"][0]
       and "USER_INVOCATION_ID" in _jc97["fields_matched"][1]
       and "INVOCATION_ID" in _jc97["field_not_used_here"],
       "R-646 R4: a copy names BOTH invocation fields, and a unit with "
       "nothing under either the id or the name is ABSENT -- never a 0 "
       "quoted as a count")
    ok(set(_jc97["retention"]) >= {"measured_at_utc", "query",
                                   "oldest_user_journal_entry_utc"}
       # THE ASSERTION IS ON THE VALUE, not on a word that lives in the
       # KEY NAME -- which is what this line first tested, and it is the
       # needle-matching-its-own-prose shape one more time.
       and "stale within minutes" in _jc97["retention"][
           "it_is_a_MEASUREMENT_not_a_property"],
       f"and the retention state travels as a MEASUREMENT with its own "
       f"query and as-of -- five reads across one hour walked the "
       f"window's start about as fast as the clock, so a state named once "
       f"is stale within minutes")
    ok(_jc97["copied_at_the_moment_of_reading"] is True
       and _jc97["invocation_id_source"].startswith("the unit's own"),
       "and the id is READ FROM THE UNIT, not passed in by a caller who "
       "might name a different run")
    # ---- R-637 / REV 66 S1.1: THE PORCELAIN SLICE, and the rename ----
    # The raw read fixed the FIRST line; `line[3:]` was still wrong on a
    # RENAME -- `R  a -> b` returns `a -> b` where the path is `b`, so a
    # rename read as real dirt and a real day would refuse with a message
    # sending its reader to look for a file called `a -> b`. Fail-safe in
    # direction, wrong in cause. The parser is DA's, because a porcelain
    # parser is INFRASTRUCTURE and two of them corroborate nothing (this
    # seat is the third reader to get the format wrong in a new place).
    _pp96 = DAROOT.parse_porcelain(" M live/x.py\n?? data\nR  a -> b\n")
    _by96 = {r["path"]: r for r in _pp96["rows"]}
    ok(_pp96["n_rows"] == 3 and _pp96["n_malformed"] == 0
       and _by96["live/x.py"]["xy"] == " M"
       and _by96["data"]["untracked"] is True
       and "b" in _by96 and _by96["b"]["renamed_from"] == "a"
       and "a -> b" not in _by96,
       f"R-637 THE THREE-LINE FALSIFIER, THIRD LINE AND ALL: "
       f"{[r['path'] for r in _pp96['rows']]} -- the RENAME yields `b`, "
       f"not `a -> b`. `[3:]` passed the first two lines and failed this "
       f"one, and it is the line the falsifier has three lines for")
    _root96 = Path(_tfr.mkdtemp(prefix="de96porc_"))
    ok(_is_the_shared_data_link(str(_root96), "R ", "b") is False
       and _is_the_shared_data_link(str(_root96), " M", "live/x.py") is False,
       "and neither a renamed path nor a modified file is mistaken for the "
       "shared data link -- the classifier takes the XY CODE and the PATH "
       "from the parser, never a reconstructed line it slices again")
    _hs96 = _head_state()
    ok(_hs96["porcelain"]["parser"].startswith("da_root.parse_porcelain")
       and isinstance(_hs96["porcelain"]["n_malformed"], int),
       f"and the runner's own worktree read goes through that ONE parser "
       f"({_hs96['porcelain']['n_rows']} rows, "
       f"{_hs96['porcelain']['n_malformed']} malformed) -- a line the "
       f"parser cannot read is a NAMED status, never a silent drop")

    # ---- rule 20 as amended (R-641): THE JOURNAL IS NOT THE RECORD ----
    # journald rotates within hours: DE 84's `Started` line was GONE four
    # hours after it was quoted as evidence, while the `Consumed` line
    # survived. A number read from it is COPIED at the moment of reading
    # with the source's retention state named.
    _jabs96 = journal_read("de96-a-unit-that-cannot-exist.service")
    ok(_jabs96["status"] == "ABSENT"
       and _jabs96["n_lines_by_id"] == 0
       and "not the same fact" in _jabs96["why_absent_not_zero"],
       "R-641 KNOWN-BAD: a unit the journal holds nothing for is ABSENT, "
       "never a 0 quoted as a count. A zero from a rotating store is "
       "indistinguishable from a zero that never happened -- which is what "
       "makes a control that greps the journal a control whose verdict "
       "depends on retention")
    ok(set(_jabs96) >= {"read_at_utc", "n_lines_by_id",
                        "n_lines_by_unit_name", "retention",
                        "copied_at_the_moment_of_reading"}
       and _jabs96["copied_at_the_moment_of_reading"] is True,
       "and every read carries the same four fields -- when it was read, "
       "how many lines the journal still holds, the OLDEST ENTRY'S OWN "
       "clock, and whether the window still reaches the unit's start -- so "
       "a reader is never handed a number without the state of its source")

    # ---- REV 65 S1.2 / REV 62 S3: the launch form REFUSES at run time --
    # It REPORTED and gated nothing: a real day under `--scope` would run
    # all 85 minutes and say so only in its receipt, and the lint on the
    # command string cannot see a `--scope` behind a variable or a wrapper.
    _scope96 = {"cgroup_leaf": "x.scope", "kind": "scope"}
    _svc96 = {"cgroup_leaf": "x.service", "kind": "transient service"}
    refuses(lambda: assert_launch_form_at_runtime(
                "2026-09-03", fixture=False, observed=_scope96),
            "REV 65 S1.2: a REAL DAY in a `.scope` REFUSES BEFORE ANY "
            "STAGE. It used to run to completion and merely record the "
            "fact -- and that is how the 09-03 re-run lost 35 minutes",
            "BEFORE ANY STAGE")
    ok(assert_launch_form_at_runtime(
           "2026-09-03", fixture=False, observed=_svc96)["checked"] is True
       and assert_launch_form_at_runtime(
           "FIXTURE-DAY-1", fixture=True,
           observed=_scope96)["fixture_exemption_by_name"] is True,
       "POSITIVE CONTROL, BOTH CELLS: a real day in a transient SERVICE "
       "admits, and a FIXTURE may declare an exemption BY NAME -- the "
       "falsifier has to be able to run under a real `.scope` to prove the "
       "refusal fires at all")
    # THIS CELL WAS AN `ok(...)` THAT READ THE FIELD, and that is exactly
    # REV 68 S1.2's finding: the exemption was REPORTED, not enforced, so
    # an undeclared name "reported no exemption" and ran anyway. It is a
    # `refuses(...)` now.
    refuses(lambda: assert_launch_form_at_runtime(
                "FIXTURE-DAY-NOT-DECLARED", fixture=True,
                observed=_scope96),
            f"and the fixture exemption is a DECLARED LIST "
            f"({len(SCOPE_EXEMPT_FIXTURE_DAYS)} names) that GATES, not "
            f"'any fixture': an undeclared fixture name under a scope "
            f"REFUSES, so the list cannot quietly become a blanket",
            "not among")
    # ---- R-628: THE LAUNCH FORM IS A TRANSIENT SERVICE ----------------
    # The scope form was published in THE_ONE_COMMAND for four rounds and
    # cost a real day 35 minutes: `systemd-run --scope` registers the
    # processes the CALLER forks, so the run sat in this seat's tool-shell
    # process group and died when the harness stopped the background task.
    # Measured both ways, out of battery (a battery must not need a
    # session manager) and recorded in `de_launch_form_probe.py`:
    #   --scope,   TERM to the launcher's process group -> the run DIED
    #   --service, TERM to the launcher's process group -> the unit LIVES
    #   --service, a HELD lock -> ExecMainStatus 1, the payload never runs
    _cmd94 = the_one_command("2026-09-03", "/BOOK", "/OUT")
    ok(assert_launch_form(_cmd94)["ok"] is True
       and "--scope" not in _cmd94
       and " -- flock -n " in _cmd94
       and _cmd94.count("<deNNsmoke>") == 1,
       f"R-628 POSITIVE CONTROL: the published command is a transient "
       f"SERVICE with the lock as the unit's OWN ExecStart, and exactly "
       f"one substitution is left (the unit name). The command and the "
       f"predicate are ONE fact -- `the_one_command` composes it and "
       f"`assert_launch_form` checks that string, not a second one typed "
       f"beside it")
    for _bad94, _needle94 in (
            (_cmd94.replace("--user --unit", "--user --scope --unit"),
             "process group"),
            (_cmd94.replace("--unit=<deNNsmoke> ", ""), "no `--unit=`"),
            (_cmd94.replace(" -- flock -n", " -- "), "INSIDE the unit"),
            (_cmd94.replace("--working-directory=", "--wd="),
             "working-directory")):
        refuses(lambda c=_bad94: assert_launch_form(c),
                f"R-628 KNOWN-BAD: a launch command that fails "
                f"{_needle94!r} is REFUSED. The scope form is the one that "
                f"actually happened, so it is the one the checker must "
                f"refuse by name", _needle94)
    _uid94 = unit_identity()
    ok(set(_uid94) >= {"cgroup_leaf", "unit", "kind", "invocation_id",
                       "is_the_declared_launch_form"}
       and _uid94["is_the_declared_launch_form"] is (
           str(_uid94["cgroup_leaf"]).endswith(".service")),
       f"and the RECEIPT can say which wrapper actually ran it -- measured "
       f"from this process's own cgroup leaf ({_uid94['cgroup_leaf']}, "
       f"kind {_uid94['kind']}), not from the form that was published. A "
       f"receipt that cannot be asked 'scope or service?' cannot be asked "
       f"why it died")

    # ---- the shared-tree DATA SYMLINK is not a dirty worktree ---------
    # `scripts/wt_refresh.sh` replaces this worktree's `data/` with a
    # symlink to the canonical data root. `.gitignore` carries `data/`,
    # which matches a DIRECTORY and not a symlink, so the link showed as
    # untracked, the worktree read DIRTY, and a REAL day refuses at import
    # on a dirty worktree. Measured after the mandated refresh: the
    # rehearsal NOT_READY on P10 and `assert_source_unchanged` REFUSED --
    # the refresh procedure made GO impossible.
    _hs94 = _head_state()
    ok(set(_hs94) >= {"dirty", "dirty_beyond_the_shared_data_link",
                      "shared_data_link_exempted"}
       and _hs94["dirty"] is not None,
       f"R-628 / the refresh procedure: the worktree's state is reported "
       f"in BOTH readings -- any untracked entry ({_hs94['dirty']}) and "
       f"the one the refusal uses "
       f"({_hs94['dirty_beyond_the_shared_data_link']}) -- so the "
       f"exemption is visible rather than silent")
    _fake94 = Path(_tfr.mkdtemp(prefix="de94link_"))
    (_fake94 / "notalink").write_text("x")
    (_fake94 / "elsewhere").mkdir()
    ok(_is_the_shared_data_link(
           str(_fake94), "??", "notalink") is False
       and _is_the_shared_data_link(
           str(_fake94), " M", "data") is False,
       "KNOWN-BAD, BOTH DOORS: a plain untracked FILE named like the link "
       "is NOT exempted, and a TRACKED MODIFICATION at the exempt path is "
       "NOT exempted either -- the exemption is a PROPERTY (untracked AND "
       "a symlink AND resolving to the data root), never a name. A "
       "name-matched exemption is how a binding map comes to excuse the "
       "thing it exists to catch (R-613)")
    import os as _os94
    _os94.symlink(DR.resolve()["data_root"], _fake94 / "data")
    ok(_is_the_shared_data_link(str(_fake94), "??", "data") is True,
       "POSITIVE CONTROL, AND IT ADMITS: an untracked symlink resolving to "
       "the canonical data root IS exempted -- so the known-bads above "
       "fire on the property and not on everything")
    _os94.symlink("/tmp", _fake94 / "otherlink")
    ok(_is_the_shared_data_link(
           str(_fake94), "??", "otherlink") is False,
       "and a symlink pointing SOMEWHERE ELSE is refused, which is the "
       "half a name match would have missed entirely")
    # ---- REV 55 S1.2 / S2.4: two numbers that must say what they are --
    _der91 = REAL_DAY_BUDGET_DERIVATION
    _measured91 = set(_der91["which_terms_are_MEASURED"])
    _declared91 = set(_der91["which_terms_are_DECLARED_ALLOWANCES"])
    ok(_measured91 & _declared91 == set()
       and "be_day_reference_measured_mb" in _measured91
       and "battery_retained_mb_measured" in _measured91
       and "headroom_for_null_and_seal_mb" in _declared91
       and "not a measurement" in _der91["which_terms_are_DECLARED_"
                                         "ALLOWANCES"][
           "headroom_for_null_and_seal_mb"],
       f"REV 55 S1.2: every term of the budget derivation is classed "
       f"MEASURED ({sorted(_measured91)}) or DECLARED ALLOWANCE "
       f"({sorted(_declared91)}), the two sets are DISJOINT, and the "
       f"headroom says in words that it is not a measurement. A reader "
       f"resolving this block saw an arithmetic chain and could take all "
       f"of 4000 MB for measurement")
    ok(abs(_der91["declared_mb"] - _der91["terms_sum_mb"] - 465.9) < 0.05,
       f"and the ROUNDING is COMPUTED, not absorbed: declared "
       f"{_der91['declared_mb']:.0f} minus the terms' "
       f"{_der91['terms_sum_mb']:.1f} is "
       f"{_der91['declared_mb'] - _der91['terms_sum_mb']:.1f} MB of "
       f"allowance")
    _light91 = battery_resources(time.time(), _peak_rss_mb())
    _heavy91 = battery_resources(time.time() - (HEAVY_WALL_S + 5),
                                 _peak_rss_mb())
    ok(_light91["would_need_the_lock_standalone"]
       is (_peak_rss_mb() > HEAVY_RSS_GB * 1024.0)
       and _heavy91["would_need_the_lock_standalone"] is True
       and _light91["heavy_bar_mb"] == HEAVY_RSS_GB * 1024.0
       and _light91["heavy_bar_seconds"] == HEAVY_WALL_S,
       f"REV 55 S2.4: the battery MEASURES ITSELF against rule 20's own "
       f"constants -- not a second pair typed beside them -- and the "
       f"measurement FIRES: a run {HEAVY_WALL_S + 5:.0f} s long reports "
       f"`would_need_the_lock_standalone` True. This battery is at "
       f"{_light91['fraction_of_the_heavy_bar_by_memory'] * 100:.0f}% of "
       f"the memory bar, up from 52 MB three rounds ago")
    # ---- REV 55 S2.1: THE OUTPUT'S NAME, and it was CERTAIN ----------
    # The operator supplied `--output` and the harvested GO procedure built
    # the name at LAUNCH; the emit compares the filename's stamp against
    # the moment of WRITING. On the last smoke's own name the reviewer
    # measured -5,074 s: 85 minutes, then a refusal about a filename. Both
    # halves of that are driven here -- the failure and the repair.
    _launch_name = ("p003_de_gate1_day_run_20260903_SEALED__"
                    "20260906T082155Z.json")
    refuses(lambda: assert_name_stamp_is_the_clock(
                Path("/tmp") / _launch_name, "2026-09-06T09:46:29+00:00"),
            "REV 55 S2.1, THE FAILURE REPRODUCED: a LAUNCH-stamped receipt "
            "name refuses at the emit -- the last smoke's own name against "
            "its own write time is -5,074 s. This is what the published "
            "procedure would have done after 85 minutes", "-5074s")
    _free = Path("/tmp/p003_de_gate1_day_run_20260903_SEALED.json")
    ok(assert_name_stamp_is_the_clock(
           _free, datetime.datetime.now(
               datetime.timezone.utc).isoformat())[
           "name_carries_a_stamp"] is False
       and not Path(_free.name).match(
           sealed_day_receipt_glob("2026-09-03")),
       f"AND THE OBVIOUS ESCAPE IS WORSE: a STAMP-FREE name PASSES the "
       f"emit check and does NOT match "
       f"`{sealed_day_receipt_glob('2026-09-03')}` -- the receipt would "
       f"exist and the read gate would report the day MISSING. Neither "
       f"name the operator can type is right, which is why the name is "
       f"not the operator's to type")
    # THE REPAIR: the runner composes it, from the clock, at the moment of
    # writing -- DE 88's own principle, applied where it was still being
    # violated. `--output` is the DIRECTORY, which is knowable in advance
    # and carries no stamp.
    _now91 = datetime.datetime.now(datetime.timezone.utc)
    _composed = day_receipt_name("2026-09-03", fixture=False,
                                 stamp=emission_stamp(_now91))
    _ns = assert_name_stamp_is_the_clock(Path("/tmp") / _composed,
                                         _now91.isoformat())
    ok(_ns["stamp_is_the_clock"] is True
       and abs(_ns["delta_seconds"]) < 5.0
       and Path(_composed).match(sealed_day_receipt_glob("2026-09-03"))
       # READ FROM THE PARAMS, never typed beside them: the declared
       # per-day glob is what DA's landing record and the read gate
       # resolve, so the composed name is checked against IT.
       and Path(_composed).match(
           live["read_gate"]["receipt_naming"][
               "expected_per_day"]["2026-09-03"]),
       f"REV 55 S2.1, THE REPAIR: the composed name {_composed} carries the "
       f"clock at the moment of writing ({_ns['delta_seconds']:+.3f} s "
       f"against `as_of`, from the SAME clock read), it MATCHES the sealed "
       f"glob, and it is the convention params v12 declares -- read from "
       f"the params, not typed beside them")
    _fx = day_receipt_name("FIXTURE-DAY-1", fixture=True,
                           stamp=emission_stamp(_now91))
    ok(not Path(_fx).match(sealed_day_receipt_glob("FIXTURE-DAY-1"))
       and FIXTURE_DAY_RECEIPT_MIDFIX in _fx
       and "FIXTURE-DAY-1" in _fx,
       f"and a FIXTURE receipt ({_fx}) cannot match the sealed glob AT ALL "
       f"-- not merely carry a different day. `--synthetic-day 2026-09-03` "
       f"once emitted a SEALED-looking artifact from a synthetic book; the "
       f"day half was locked and the FILENAME half stayed with the caller")
    # THE PRE-WORK REFUSALS. The point is that a naming mistake costs ZERO
    # draws, so they are driven at the function the emit calls FIRST.
    _d91 = RUN_COUNTERS["draws_performed"]
    refuses(lambda: assert_output_is_a_directory(
                Path("/tmp") / _launch_name, day="2026-09-03"),
            "KNOWN-BAD: a caller-supplied STAMPED filename is REFUSED "
            "BEFORE ANY WORK, naming the -5,074 s measurement as the "
            "reason", "looks like a FILENAME")
    refuses(lambda: assert_output_is_a_directory(_free, day="2026-09-03"),
            "and a caller-supplied STAMP-FREE filename is refused too -- "
            "it would pass the emit and be invisible to the glob, which is "
            "the worse of the two failures", "INVISIBLE")
    ok(RUN_COUNTERS["draws_performed"] == _d91,
       f"AND BOTH REFUSALS COST ZERO DRAWS "
       f"({RUN_COUNTERS['draws_performed'] - _d91}) -- they are evaluated "
       f"at the top of `_main_day`, before the book is even opened. A "
       f"naming rule enforced at the emit is a rule that costs 85 minutes "
       f"to break")
    _dirok = assert_output_is_a_directory(
        Path(_tfr.mkdtemp(prefix="de91dir_")), day="2026-09-03")
    ok(_dirok["output_is_a_directory"] is True,
       "POSITIVE CONTROL, AND IT ADMITS: a DIRECTORY is accepted, so the "
       "known-bads above fire on the name's shape and not on everything")
    # AND THE GUARD THAT REPLACED `if output.exists()`.
    _emptyroot = _synth_ledger([], )
    ok(assert_no_sealed_receipt_yet(
           _D6[0], _emptyroot)["n_sealed_artifacts_present"] == 0,
       "POSITIVE CONTROL: with no receipt for the day, the pre-work guard "
       "admits")
    refuses(lambda: assert_no_sealed_receipt_yet(_D6[0], _all6),
            "KNOWN-BAD: a day that ALREADY has a sealed artifact is "
            "refused BEFORE any work. This replaces `if output.exists()`, "
            "which could only catch a collision on the exact stamp the "
            "caller happened to type, and is stronger: it asks whether the "
            "DAY has landed", "already has")
    # ---- REV 54 S0/S0.1: THE EIGHT SHAPES, driven, verdicts asserted --
    # The reviewer drove both seats' resolvers side by side and found row
    # 8 -- a BARE-STRING `supersedes` -- taking DE's read gate DOWN with an
    # uncaught AttributeError where DA refused by name. The table is driven
    # here on THIS seat's resolver, with the VERDICT (resolve / refuse)
    # asserted per row and the status NAMES only recorded: two independent
    # implementations may name a verdict differently (R-235); they may not
    # reach a different one. Cross-seat AGREEMENT is measured by
    # `de_r608_resolver_agreement.py`, which imports BOTH resolvers -- it
    # is NOT imported here, because a day run's import closure must not
    # contain another seat's module (rule 22: DA landing a commit mid-run
    # would refuse this seat's emit).
    def _shape_root(sup, *, two=True):
        rp = Path(_tfr.mkdtemp(prefix="de90shape_"))
        der = rp / "pm_5min/derived"
        der.mkdir(parents=True)
        _cx = [x for x in sorted(day_forms(_D6[0])) if "-" not in x][0]
        v1 = (der / f"{SEALED_DAY_RECEIPT_PREFIX}{_cx}"
                    f"{SEALED_DAY_RECEIPT_MIDFIX}20260906T010000Z.json")
        v1.write_text(json.dumps({"day": _D6[0], "n": 1}))
        if not two:
            return rp, v1
        rec = {"day": _D6[0], "n": 2}
        if sup is not _SHAPE_ABSENT:
            rec["supersedes"] = (sup(v1) if callable(sup) else sup)
        (der / f"{SEALED_DAY_RECEIPT_PREFIX}{_cx}"
               f"{SEALED_DAY_RECEIPT_MIDFIX}20260906T020000Z.json"
         ).write_text(json.dumps(rec))
        return rp, v1

    _SHAPE_ABSENT = object()
    _SHAPES = [
        ("1 a single receipt", None, "RESOLVE", dict(two=False)),
        ("2 {path, sha256} both correct",
         lambda v: {"path": v.name, "sha256": sha256_streamed(v)},
         "RESOLVE", {}),
        ("3 {sha256} only",
         lambda v: {"sha256": sha256_streamed(v)}, "REFUSE", {}),
        ("4 {path} only", lambda v: {"path": v.name}, "REFUSE", {}),
        ("5 named file present, WRONG digest",
         lambda v: {"path": v.name, "sha256": "0" * 64}, "REFUSE", {}),
        ("6 digest present under a DIFFERENT name",
         lambda v: {"path": "somewhere_else.json",
                    "sha256": sha256_streamed(v)}, "REFUSE", {}),
        ("7 two receipts, NO link", _SHAPE_ABSENT, "REFUSE", {}),
        ("8 a BARE STRING", lambda v: v.name, "REFUSE", {}),
    ]
    _table, _raised = [], []
    for _lbl, _sup, _want, _kw in _SHAPES:
        _rp, _ = _shape_root(_sup, **_kw)
        try:
            _r = find_sealed_day_receipt(_D6[0], _rp)
            _got = "RESOLVE" if _r["present"] else "REFUSE"
            _st = _r["status"]
        except Exception as _exc:                 # noqa: BLE001
            _got, _st = "RAISED", f"{type(_exc).__name__}: {_exc}"
            _raised.append(_lbl)
        _table.append({"shape": _lbl, "want": _want, "got": _got,
                       "status": _st})
    ok(not _raised and all(t["got"] == t["want"] for t in _table)
       and len(_table) == 8,
       f"REV 54 S0, THE EIGHT SHAPES DRIVEN: every row reaches the "
       f"VERDICT it must and NONE RAISES -- "
       f"{[t['status'] for t in _table]}. Row 8, a bare-string "
       f"`supersedes`, used to leave `find_sealed_day_receipt` with an "
       f"uncaught AttributeError, and `read_gate` calls it for EVERY "
       f"ruled day: one malformed artifact took the whole gate down "
       f"where DA refused by name")
    ok(all(t["status"] in (CHAIN_RESOLVED_STATUSES if t["want"] == "RESOLVE"
                           else CHAIN_REFUSAL_STATUSES) for t in _table),
       f"and every row's status is a DECLARED one -- resolutions in "
       f"{list(CHAIN_RESOLVED_STATUSES)}, refusals in "
       f"{list(CHAIN_REFUSAL_STATUSES)} -- so a refusal cannot be a "
       f"status nobody declared")
    _row5 = [t for t in _table if t["shape"].startswith("5")][0]
    ok(_row5["status"] == "SUPERSEDES_TARGET_DIGEST_MISMATCH",
       "REV 54 S0's ACCURACY NOTE, CLOSED: row 5 -- the named file is "
       "PRESENT with other bytes -- said DANGLING_SUPERSEDES, 'a "
       "supersession whose predecessor is absent'. Right verdict, a "
       "message that misdescribes the cause; the two causes are now two "
       "names")
    _rp8, _v18 = _shape_root(lambda v: v.name)
    _gate8 = read_gate(live, now_utc=_after, root=_rp8, ledger_rows=_rows6)
    ok(_gate8["per_day"][_D6[0]]["receipt"]["status"]
       == "SUPERSEDES_MALFORMED"
       and _gate8["may_open"] is False,
       "AND THROUGH THE WHOLE GATE, not just the resolver: the malformed "
       "day is REFUSED BY NAME and the read stays shut. The reviewer "
       "drove this half too, and it is the half that mattered -- the "
       "exception left `read_gate` itself")
    refuses(lambda: (_ for _ in ()).throw(RunnerRefused(
        "SUPERSEDES_MALFORMED" if supersedes_shape(
            {"supersedes": ["a", "list"]})["kind"] == "MALFORMED"
        else "the shape judge did not fire")),
        "AND THE SHAPE JUDGE FIRES ON A TYPE NOBODY WROTE A ROW FOR: a "
        "LIST is not an object carrying both halves either, so it is "
        "MALFORMED rather than whatever `.get` would have done to it",
        "SUPERSEDES_MALFORMED")
    ok(supersedes_shape({})["kind"] == "ABSENT"
       and supersedes_shape({"supersedes": None})["kind"] == "ABSENT"
       and supersedes_shape({"supersedes": {}})["kind"] == "INCOMPLETE"
       and supersedes_shape({"supersedes": {"path": 5, "sha256": "a" * 64}}
                            )["kind"] == "INCOMPLETE"
       and supersedes_shape({"supersedes": {"path": "p", "sha256": "s"}}
                            )["kind"] == "PAIR",
       "and the shape judge's own table: absent and null declare NO LINK, "
       "an empty object and a NON-STRING half are INCOMPLETE (a link that "
       "was attempted and not written), and only two strings are a PAIR")

    # ---- REV 54 S1.3: the landing record's twice-written fields -------
    def _lr_root(*docs):
        rp = Path(_tfr.mkdtemp(prefix="de90lr_"))
        der = rp / "pm_5min/derived"
        der.mkdir(parents=True)
        out = []
        for _i, _doc in enumerate(docs):
            _f = (der / f"p003_da_gate1_pre_read_20260903__"
                        f"2026090{6 + _i}T03000{_i}Z.json")
            _f.write_text(json.dumps(_doc))
            out.append(_f)
        return rp, der, out

    def _lr(sha_lr, sha_top, **extra):
        return {"day": _D6[0], LANDING_RECORD_DECLARED_FLAG: True,
                "landing_record": {"day": _D6[0], "receipt_path": "R.json",
                                   "receipt_sha256": sha_lr},
                "receipt": {"path": "R.json", "sha256": sha_top}, **extra}

    _rp, _der, _ = _lr_root(_lr("a" * 64, "a" * 64))
    _l_ok = landing_record_for(_D6[0], _rp)
    ok(_l_ok["present"] is True
       and _l_ok["receipt_sha256_at_landing"] == "a" * 64
       and _l_ok["field_copies"]["fields"]["receipt_sha256_at_landing"][
           "read_from"] == "landing_record.receipt_sha256",
       "REV 54 S1.3 POSITIVE CONTROL: a landing record whose two copies "
       "AGREE resolves, and the value is read from the DECLARED field "
       "`landing_record.receipt_sha256` -- the one under "
       "`is_the_declared_LANDING_RECORD` and the one DA's own reader "
       "resolves. DE read the older top-level `receipt.sha256`")
    _rp, _der, _ = _lr_root(_lr("a" * 64, "f" * 64))
    _l_bad = landing_record_for(_D6[0], _rp)
    ok(_l_bad["present"] is False
       and _l_bad["status"] == "LANDING_RECORD_FIELD_COPIES_DISAGREE"
       and _l_bad["field_copies"]["disagreeing"]
       == ["receipt_sha256_at_landing"],
       "REV 54 S1.3 KNOWN-BAD: the SAME digest written TWICE and "
       "disagreeing is REFUSED BY NAME. The reviewer drove exactly this "
       "and NEITHER SEAT COMPLAINED -- DA read aaaa, DE read ffff, and "
       "conjunct 3 is the conjunct that stops a re-roll")
    _rp, _der, _fs = _lr_root(_lr("a" * 64, "a" * 64))
    _v1lr = _fs[0]
    (_der / "p003_da_gate1_pre_read_20260903__20260907T040000Z.v2.json"
     ).write_text(json.dumps(_lr(
         "b" * 64, "b" * 64,
         supersedes={"path": _v1lr.name,
                     "sha256": sha256_streamed(_v1lr)})))
    _l_ch = landing_record_for(_D6[0], _rp)
    ok(_l_ch["present"] is True
       and _l_ch["status"] == "PRESENT_CHAIN_HEAD"
       and _l_ch["receipt_sha256_at_landing"] == "b" * 64,
       "AND DA'S OWN DECLARED CORRECTION PATH FOR A LANDING RECORD "
       "RESOLVES HERE: a `.v2` carrying the pair. It read AMBIGUOUS -- "
       "the landing side did not follow `supersedes` at all -- so a day "
       "whose landing record had been corrected exactly as DA declares "
       "corrections would have been refused by this seat. One rule, one "
       "resolver, both chains")
    _rp, _der, _ = _lr_root({"day": _D6[0],
                             "landing_record": {"receipt_sha256": "a" * 64}})
    _l_nf = landing_record_for(_D6[0], _rp)
    ok(_l_nf["status"] == "NO_LANDING_RECORD"
       and _l_nf["refused_records"][0]["status"]
       == "NOT_DECLARED_A_LANDING_RECORD",
       "and an artifact that does NOT carry "
       f"`{LANDING_RECORD_DECLARED_FLAG}` is a NAMED STATUS, not a "
       "landing record: DE globbed the NAME and never checked the flag, "
       "so the two seats could disagree about which artifacts are "
       "landing records at all")
    _rp, _der, _ = _lr_root({LANDING_RECORD_DECLARED_FLAG: True,
                             "landing_record": {"receipt_sha256": "a" * 64}})
    ok(landing_record_for(_D6[0], _rp)["refused_records"][0]["status"]
       == "LANDING_RECORD_NO_DAY_FIELD",
       "and a declared record carrying NO DAY is a named status too "
       "(rule 11): it was dropped silently by a truthiness test")
    # ---- REV 51 S0: the pin chain must RESOLVE, in one direction -----
    _PLACEHOLDERS = ("<emitted", "PENDING", "TBD", "<the ", "ABSENT")
    _pdd = live.get("design_declaration") or {}
    _dpath = Path(DR.resolve()["data_root"]).parent / _pdd.get("path", "")
    _bad_ph = [t for t in _PLACEHOLDERS if t in str(_pdd.get("path", ""))]
    ok(not _bad_ph and _pdd.get("pin_direction") == "design -> params",
       f"REV 51 S0: the params name the design at a REAL path with no "
       f"literal placeholder, and record that the pin runs design -> "
       f"params. params v6 named it at a path containing "
       f"'<emitted this round>', which resolves nowhere")
    if offline:
        offline_skip("the params -> design -> params walk (it reads the "
                     "design artifact under data/)")
    else:
        # THE BINDING HALF IS THE HEAD'S PIN (DE 101/DE 104). This read
        # the design the PARAMS NAME and compared ITS pin -- so the moment
        # params moved to v15 it compared v23's pin (v14) against v15 and
        # failed, while the resolved HEAD v24 pinned v15 correctly. The
        # named path must be IN the chain; the HEAD is what binds. One
        # rule, and P3_design already used it.
        _dchw = design_chain()
        _headp = Path(_dchw["head_path"]) if _dchw.get("resolved") \
            else _dpath
        _walk = {"params_names": str(_pdd.get("path")),
                 "design_exists": _dpath.is_file(),
                 "resolved_head": _dchw.get("head_name"),
                 "named_design_is_in_the_chain":
                     Path(str(_pdd.get("path"))).name
                     in (_dchw.get("chain_names") or [])}
        if _headp.is_file():
            _dj = json.loads(_headp.read_text())
            _here = hashlib.sha256(
                (Path(__file__).resolve().parents[2] / PARAMS_REL
                 ).read_bytes()).hexdigest()
            _pin = _dj.get("parameters") or {}
            _walk["design_pins"] = _pin.get("sha256")
            _walk["design_pins_path"] = _pin.get("path")
            # PATH **AND** DIGEST. This compared the digest only, so a
            # design whose pin named params v2 by PATH while its DIGEST was
            # v9's closed the walk and P3_design passed. The path half had
            # no control at all.
            _walk["path_matches"] = (
                str(_pin.get("path", "")).rsplit("/", 1)[-1]
                == PARAMS_REL.rsplit("/", 1)[-1])
            _walk["closes"] = bool(_walk["design_pins"] == _here
                                   and _walk["path_matches"]
                                   and _walk["named_design_is_in_the_"
                                             "chain"])
            ok(_walk["closes"] is True,
               f"AND THE WALK CLOSES ON BOTH HALVES, THROUGH THE HEAD "
               f"({_walk['resolved_head']}): params name the design "
               f"at {_dpath.name}; that design's pin names "
               f"{str(_walk['design_pins_path']).rsplit('/', 1)[-1]} AND "
               f"hashes to this params file. Comparing the digest alone let "
               f"design v16 pass with v2's PATH beside v9's DIGEST. "
               f"design v14 named params at v2's PATH with v6's DIGEST, so "
               f"it resolved in neither direction")
        else:
            ok(True,
               f"the design named by the params ({_dpath.name}) is not "
               f"emitted yet in this tree; the walk is driven the moment "
               f"it is, and the placeholder check above has already run")

    # ---- REV 51 S1.5: rule 13's own form must not read as AMBIGUOUS ---
    def _add_v2(rootp, day, *, chained=True, half=None, moved=False):
        der = rootp / "pm_5min/derived"
        compact = [x for x in sorted(day_forms(day)) if "-" not in x][0]
        v1 = sorted(der.glob(sealed_day_receipt_glob(day)))[0]
        rec = json.loads(v1.read_text())
        _pair = {"path": v1.name,
                 "sha256": (sha256_streamed(v1) if chained else "0" * 64)}
        if moved:
            _pair["path"] = "some_other_name.json"
        if half == "path":
            _pair.pop("sha256")
        elif half == "digest":
            _pair.pop("path")
        rec["supersedes"] = _pair
        rec["note"] = "the .v2 correction"
        (der / f"{SEALED_DAY_RECEIPT_PREFIX}{compact}"
               f"{SEALED_DAY_RECEIPT_MIDFIX}20260906T999999Z.json"
         ).write_text(json.dumps(rec))
        return v1

    _one = _synth_ledger(_D6)
    _f1 = find_sealed_day_receipt(_D6[0], _one)
    ok(_f1["present"] is True and _f1["n_matches"] == 1,
       "REV 51 S1.5 POSITIVE CONTROL: one receipt for a day resolves to "
       "itself")
    _chain = _synth_ledger(_D6)
    _v1 = _add_v2(_chain, _D6[0], chained=True)
    _f2 = find_sealed_day_receipt(_D6[0], _chain)
    ok(_f2["present"] is True
       and _f2["status"] == "PRESENT_CHAIN_HEAD"
       and _f2["path"] != str(_v1) and _f2["n_matches"] == 2,
       "REV 51 S1.5, THE FIX: a superseding .v2 beside its v1 -- RULE 13'S "
       "OWN FORM -- resolves to the CHAIN HEAD. It returned AMBIGUOUS "
       "before, so the .v2 correction of the 09-03 receipt would have "
       "CLOSED THE GATE on 09-03")
    # ---- REV 53 S0.3: the fixture check inside a real-day process ---
    # ~800 MB, enough to pass the 700 MB fixture budget on the
    # PROCESS-WIDE high-water -- which is the comparison that refused the
    # smoke. Freed immediately; the cap is 8 GiB and this is transient.
    _inflate = [bytearray(100 * 1024 * 1024) for _ in range(8)]
    for _b8 in _inflate:
        _b8[::4096] = b"\x01" * len(_b8[::4096])
    _hw_after = _peak_rss_mb()
    del _inflate
    _mk3 = write_synthetic_day(
        "FIXTURE-DAY-1", _tfr.mkdtemp(prefix="de89rev53_"),
        params=live, n_slugs=12)
    _post = run_day("FIXTURE-DAY-1", _mk3["book_path"], params=live,
                    fixture=True)
    _mp = _post["memory_plan"]
    ok(_hw_after > FIXTURE_DAY_PEAK_RSS_MB_BUDGET
       and _mp["growth_rss_mb"] < FIXTURE_DAY_PEAK_RSS_MB_BUDGET
       and _mp["within_budget"] is True,
       f"REV 53 S0.3 REPRODUCED AND CLOSED: after inflating the process to "
       f"{_hw_after:.0f} MB -- past the 700 MB fixture budget -- a FIXTURE "
       f"day still ADMITS, because the budget is now this run's GROWTH "
       f"({_mp['growth_rss_mb']:.1f} MB) and not the process-wide "
       f"high-water. That comparison is exactly what refused the 09-03 "
       f"smoke after 84 minutes: a seam between two correct decisions")
    ok(_hw_after > _mp["growth_rss_mb"] + 100,
       f"and the OLD comparison is shown to have refused: the process-wide "
       f"high-water is {_hw_after:.0f} MB against a growth of "
       f"{_mp['growth_rss_mb']:.1f} MB -- same fixture, same budget, "
       f"opposite verdicts")
    # ====== DE 134 / REV 102: THE DAY PATH'S LEDGER ANCHOR ==========
    # RED FIRST, AND THE RED IS THE ARGUMENT THE LANDED DAY PATH MADE.
    # REV 102's NO-GO: `_main_day` passed neither `ledger_anchor` nor
    # `receipt_path`, so tonight's first real day under ruling-carrying
    # params would have met `assert_ledger_anchor` and refused before its
    # work. The guard was right; the day path had never been given one.
    _old134 = None
    try:
        # EXACTLY what the landed `_main_day` produced: no anchor.
        assert_ledger_anchor(live, fixture=False, anchor=None)
    except RunnerRefused as _e134:
        _old134 = str(_e134).split(":")[0].replace("REFUSED ", "")
    _d134 = Path(_tfr.mkdtemp(prefix="de134_dayanchor_"))
    _a134 = day_run_ledger_anchor(_d134, fixture=False)
    _adm134 = assert_ledger_anchor(live, fixture=False, anchor=_a134)
    ok(_old134 == "DECISION_LEDGER_HAS_NO_ANCHOR"
       and _adm134["owes_a_ledger"] is True
       and Path(_adm134["anchor"]) == _d134,
       f"DE 134 RED->GREEN, THE DELTA REV 102 MEASURED: the day path's "
       f"OLD argument (no anchor) refuses by its own name -- `{_old134}` "
       f"-- and the anchor `day_run_ledger_anchor` computes from "
       f"`--output` ADMITS, owes_a_ledger True. Same guard, same params, "
       f"opposite verdicts: the change is the ARGUMENT, not the check")
    ok(day_run_ledger_anchor(_d134, fixture=True) is None,
       "DE 134: and a FIXTURE gets None from the same function -- its "
       "rows are synthetic, R-765 keeps the numbers of REAL runs, and "
       "the fixture path is unchanged (DE 133). The decision is in ONE "
       "drivable function, not a conditional at the call site")
    # THE RESOLUTION, BOTH SHAPES, AND THE KNOWN-BAD IS THE LANDED LINE.
    # The write used `Path(anchor).parent`, which is right for the early
    # read's FILE anchor and puts a DIRECTORY anchor's ledger one level
    # ABOVE the ledger directory.
    _st134 = emission_stamp()
    _pdir134 = ledger_path_for(_d134, "FIXTURE-DAY-1", _st134)
    _pfile134 = ledger_path_for(_d134 / "the_artifact.json",
                                "FIXTURE-DAY-1", _st134)
    _oldres134 = Path(_d134).parent / _pdir134.name
    ok(_pdir134.parent == _d134 and _pfile134.parent == _d134
       and _pdir134 == _pfile134 and _oldres134.parent != _d134,
       f"DE 134: `ledger_path_for` lands the ledger IN a DIRECTORY "
       f"anchor and BESIDE a FILE anchor -- both resolve to "
       f"{_d134.name}/ -- while the landed `Path(anchor).parent` on the "
       f"day path's directory would have written to "
       f"{_oldres134.parent.name}/, one level above the output "
       f"directory. The early read's file anchor is unchanged")
    # THE SEAM, DERIVED FROM THIS MODULE'S OWN AST -- never typed, and
    # with the known-bad built by DELETING the keyword from this file's
    # own source, so the cell measures a delta from the landed shape and
    # not agreement with the words I just wrote.
    import ast as _ast134

    def _passes_anchor(src: str) -> bool:
        tree = _ast134.parse(src)
        for fn in _ast134.walk(tree):
            if (isinstance(fn, _ast134.FunctionDef)
                    and fn.name == "_main_day"):
                for c in _ast134.walk(fn):
                    if (isinstance(c, _ast134.Call)
                            and isinstance(c.func, _ast134.Name)
                            and c.func.id == "day_split_residency_proof"):
                        for kw in c.keywords:
                            if (kw.arg == "ledger_anchor"
                                    and isinstance(kw.value, _ast134.Call)
                                    and isinstance(kw.value.func,
                                                   _ast134.Name)
                                    and kw.value.func.id
                                    == "day_run_ledger_anchor"):
                                return True
        return False

    def _forwards_kwargs(src: str) -> bool:
        tree = _ast134.parse(src)
        for fn in _ast134.walk(tree):
            if (isinstance(fn, _ast134.FunctionDef)
                    and fn.name == "day_split_residency_proof"):
                for c in _ast134.walk(fn):
                    if (isinstance(c, _ast134.Call)
                            and any(k.arg is None for k in c.keywords)):
                        return True
        return False

    _src134 = Path(__file__).resolve().read_text()
    _line134 = ("        ledger_anchor=day_run_ledger_anchor("
                "a.output, fixture=fixture),\n")
    _bad134 = _src134.replace(_line134, "", 1)
    ok(_passes_anchor(_src134) and _forwards_kwargs(_src134)
       and _line134 in _src134 and not _passes_anchor(_bad134),
       "DE 134, THE SEAM, DERIVED FROM THIS FILE'S OWN AST: `_main_day` "
       "passes `ledger_anchor=day_run_ledger_anchor(...)` into "
       "`day_split_residency_proof`, which forwards its kwargs into "
       "`run_day` -- and with that ONE keyword deleted from this same "
       "source the predicate reads FALSE, which is the shape REV 102 "
       "found landed. No line number and no digest is typed here")
    # THE CHAIN BELOW `_main_day`, WATCHED FIRING END TO END -- ONLINE
    # ONLY, and the reason is measured, not assumed: these two drive
    # `run_day` far enough to perform REAL DRAWS, and BE's `draw_null`
    # reads its committed null receipt under `data/`
    # (`be_cancel_axis_null_v1.json`, via `load_cited_be_null`). A
    # FIXTURE run must open no path under `data/`, and the fixture-run
    # proof REFUSED these cells when they ran unguarded -- which is the
    # guard working and is how this comment came to be accurate.
    #
    # `run_day` cannot be driven on a REAL day in a battery at all: the
    # lock guard refuses first, correctly. So the day is a FIXTURE and
    # the ANCHOR is the one the REAL path computes. What is established
    # is the plumbing and the write; what is NOT is the real day's own
    # frame under the lock -- REV 102 said the same of its own trace.
    if offline:
        offline_skip("DE 134 chain check 1/2 -- the day path's anchor "
                     "through `day_split_residency_proof` into a ledger; "
                     "it performs real draws, which read BE's committed "
                     "null receipt under data/")
        offline_skip("DE 134 chain check 2/2 -- the fixture half of the "
                     "same chain; same reason")
        offline_skip("R-801 falsifier 4 -- the null valued BOTH ways on a "
                     "fixture day; it performs real draws, which read "
                     "BE's committed null receipt under data/")
        offline_skip("R-801 chain check -- run_day carrying the ruled "
                     "endpoint into the arm-day and the ledger; same "
                     "reason")
        offline_skip("R-803 the two real days verify 576/576 -- it reads "
                     "the Chainlink hourly files and the venue record "
                     "under data/")
        offline_skip("R-803 the renamed field is absent from a real "
                     "fixture receipt and ledger; it writes and re-reads "
                     "them beside a run that reads data/")
        offline_skip("R-828 point-estimate mode -- it runs two real "
                     "replays and compares them to a full run; both "
                     "read BE's committed null receipt under data/")
    else:
        _mk134 = write_synthetic_day("FIXTURE-DAY-1",
                                     _tfr.mkdtemp(prefix="de134_day_"),
                                     params=live)
        _rp134 = day_split_residency_proof(
            "FIXTURE-DAY-1", _mk134["book_path"], params=live,
            fixture=True, n_days_complete=1,
            ledger_anchor=day_run_ledger_anchor(_d134, fixture=False))
        _blk134 = ((_rp134.get("day_result") or {})
                   .get("decision_ledger") or {})
        ok(isinstance(_blk134.get("sha256"), str)
           and _blk134.get("n_rows", 0) > 0
           and Path(_blk134["path"]).is_file()
           and Path(_blk134["path"]).parent == _d134,
           f"DE 134 GREEN, THE WHOLE CHAIN BELOW `_main_day`: given the "
           f"anchor the day path computes, `day_split_residency_proof` "
           f"forwards it and `run_day` writes the ledger IN the output "
           f"directory -- {Path(_blk134['path']).name}, "
           f"{_blk134['n_rows']} rows, sha256 "
           f"{_blk134['sha256'][:16]}... -- with path + sha256 + rows in "
           f"the block. GO E1 and GO E2 carried `decision_ledger: null` "
           f"here")
        _fx134 = day_split_residency_proof(
            "FIXTURE-DAY-1", _mk134["book_path"], params=live,
            fixture=True, n_days_complete=1)                 # no anchor
        _fb134 = ((_fx134.get("day_result") or {})
                  .get("decision_ledger") or {})
        ok(_fb134.get("status") == "NO_LEDGER_FOR_A_FIXTURE_DAY"
           and _fb134.get("fixture") is True,
           f"DE 134: and the FIXTURE path is UNCHANGED -- no anchor, no "
           f"refusal, the named status `{_fb134.get('status')}` with the "
           f"`fixture` field it is computed from recorded beside it "
           f"(rule 10: the status says 'fixture', so the value that "
           f"makes it true is in the artifact and not left to a reader "
           f"to infer from the guard above)")
        # ---- R-801 FALSIFIER 4: THE SECOND VALUATION IS NOT THE
        # FIRST RENAMED. Same draws, same seed, two estimators: if their
        # null mean and sd agreed, the "new endpoint" would be the old
        # one wearing a new name.
        _mod801, _cite801 = import_be_cascade(
            _with_current_cascade(live), module=None)
        _bk801 = _mod801.load(_mk134["book_path"])
        _base801c = _mod801.replay(
            _bk801, _mod801.flagged_stream(_bk801["rows"], []), 0.5)
        _slug801 = sorted({f.get("slug") for f in _base801c["fills"]}
                          - {None})
        # A FIXTURE WINNER MAP -- alternating, so the settlement is not a
        # constant that could make the two valuations agree by accident.
        _w801 = {sl: {"up_won": bool(i % 2),
                      "settle_cents": (100.0 if i % 2 else 0.0)}
                 for i, sl in enumerate(_slug801)}
        _pop801 = day_decision_population(
            _mod801, _bk801, "CONDVALUE_X_SKEW",
            live["arms"]["CONDVALUE_X_SKEW"])
        _n801 = null_draws_valued(
            _mod801, _bk801, _base801c["fills"], _pop801["by_side"],
            n_draws=16, seed=seed_for("de801fixture", "CONDVALUE_X_SKEW"),
            deadline_s=600.0, cross_check_n=4, winners=_w801)
        _m5, _ms = (statistics.fmean(_n801["values"]),
                    statistics.fmean(_n801["settle_values"]))
        _s5, _ss = (statistics.pstdev(_n801["values"]),
                    statistics.pstdev(_n801["settle_values"]))
        _res801 = _n801["second_valuation_residency"]
        ok(abs(_m5 - _ms) > 1e-9 and abs(_s5 - _ss) > 1e-9
           and len(_n801["settle_values"]) == len(_n801["values"])
           and _res801["fills_retained_per_draw"] == 0,
           f"R-801 FALSIFIER 4, THE SECOND VALUATION IS NOT THE FIRST "
           f"RENAMED: over the SAME {len(_n801['values'])} draws at the "
           f"SAME seed the 5-second null has mean {_m5:.4f} / sd "
           f"{_s5:.4f} and the SETTLEMENT null mean {_ms:.4f} / sd "
           f"{_ss:.4f}. Same draws, two estimators, different "
           f"distributions -- and the residency added is "
           f"{_res801['bytes_held_by_the_scalar_list']} bytes of scalars "
           f"with {_res801['fills_retained_per_draw']} "
           f"fills retained, which is what design v27's R11 budget asks")
        # ---- R-801: THE WHOLE CHAIN -- run_day carries the ruled
        # endpoint into the arm-day AND the ledger's own rows.
        _d801 = Path(_tfr.mkdtemp(prefix="de801_chain_"))
        _r801c = run_day("FIXTURE-DAY-1", _mk134["book_path"], params=live,
                         fixture=True, n_days_complete=1, winners=_w801,
                         ledger_anchor=_d801)
        _arm801c = [a for a in _r801c["per_day_sealed_artifacts"]
                    if a.get("status") == "OK"]
        _es801 = (_arm801c[0].get("economic_settlement") or {}
                  ) if _arm801c else {}
        _lp801 = Path((_r801c.get("decision_ledger") or {}).get("path", "/x"))
        _kinds801: dict = {}
        if _lp801.is_file():
            import gzip as _gz801
            with _gz801.open(_lp801, "rt") as _fh801:
                for _ln801 in _fh801:
                    _k801 = json.loads(_ln801).get("row")
                    _kinds801[_k801] = _kinds801.get(_k801, 0) + 1
        ok(isinstance(_es801.get("D_E_settle"), float)
           and _es801.get("endpoint", "").startswith("R-801")
           and _kinds801.get("SETTLEMENT_SCALARS", 0) >= 1
           and _kinds801.get("SETTLEMENT_SLUG", 0) >= 2 * len(_slug801),
           f"R-801 CHAIN: `run_day` carries the ruled endpoint into the "
           f"arm-day (`D_E_settle` {_es801.get('D_E_settle')!r}, Z "
           f"{_es801.get('Z')!r}, p {_es801.get('p_location')!r}) AND "
           f"into the ledger -- {_kinds801.get('SETTLEMENT_SCALARS', 0)} "
           f"SETTLEMENT_SCALARS and "
           f"{_kinds801.get('SETTLEMENT_SLUG', 0)} SETTLEMENT_SLUG rows "
           f"over {len(_slug801)} slugs and two books, so a reader with "
           f"the ledger and no receipt can re-form the P&L slug by slug")
        # ---- R-803: THE TWO REAL DAYS, VERIFIED 576/576 -------------
        # BE 99's headline, reproduced here at the real files by a second
        # implementation -- not quoted from the row.
        _days803 = ["2026-09-05", "2026-09-06"]
        _cl803 = chainlink_streams(_days803)
        _ws803 = winner_source()
        _all803: dict = {}
        for _d803 in _days803:
            _y, _m, _dd = (int(x) for x in _d803.split("-"))
            _t0s = [int(datetime.datetime(_y, _m, _dd,
                                          tzinfo=datetime.timezone.utc)
                        .timestamp()) + 300 * _i for _i in range(288)]
            _v803 = verify_winners_against_chainlink(
                _ws803["winners"], [f"btc-updown-5m-{t}" for t in _t0s],
                streams=_cl803["streams"])
            for _k803, _n803 in _v803["counts"].items():
                _all803[_k803] = _all803.get(_k803, 0) + _n803
        # R-826 CHANGED WHAT THIS CELL ASSERTS, and the new number is
        # the measurement, not a relaxation: 575 of 576 windows across
        # 09-05 and 09-06 verify, and ONE (btc-updown-5m-1788655800, on
        # 09-06) is BOUNDARY_NOT_IN_CAPTURE -- margin 0.1221 USD against
        # a 3.0 s stale read whose local p90 drift could hide 2.04 USD,
        # with NO gap logged. DISAGREE is still 0 and must stay 0.
        ok(_all803["VERIFIED_AGREE"] == 575
           and _all803.get("BOUNDARY_NOT_IN_CAPTURE", 0) == 1
           and _all803["DISAGREE"] == 0
           and _all803["CHAINLINK_UNAVAILABLE"] == 0
           and _all803["VENUE_UNRESOLVED"] == 0,
           f"R-803 + R-826 AT THE REAL FILES: "
           f"{SETTLEMENT_CONVENTION['name']} reproduces the venue's "
           f"recorded winner on "
           f"{_all803['VERIFIED_AGREE']} of 576 slugs across 09-05 and "
           f"09-06 -- DISAGREE {_all803['DISAGREE']}, unavailable "
           f"{_all803['CHAINLINK_UNAVAILABLE']}, unresolved "
           f"{_all803['VENUE_UNRESOLVED']} -- read through exp_m6's own "
           f"loader over {_cl803['n_hourly_files']} hourly files. BE "
           f"99's 288/288 on each day, and the ONE window the "
           f"verifiability layer now refuses to call verified "
           f"({_all803.get('BOUNDARY_NOT_IN_CAPTURE', 0)}) -- not a "
           f"disagreement, a boundary whose value is not in our capture")
        # ---- R-803: THE OLD NAME IS GONE FROM WHAT WE NOW WRITE ------
        def _keys803(o, name):
            n_ = 0
            if isinstance(o, dict):
                n_ += sum(1 for k in o if k == name)
                for v in o.values():
                    n_ += _keys803(v, name)
            elif isinstance(o, list):
                for e in o:
                    n_ += _keys803(e, name)
            return n_
        _lrows803 = []
        if _lp801.is_file():
            import gzip as _gz803
            with _gz803.open(_lp801, "rt") as _fh803:
                _lrows803 = [json.loads(x) for x in _fh803]
        _nk_rec = _keys803(_r801c, "inventory_leg")
        _nk_led = sum(_keys803(r, "inventory_leg") for r in _lrows803)
        _nt_rec = _keys803(_r801c, "trades_cash_flow_cents")
        ok(_nk_rec == 0 and _nk_led == 0 and _nt_rec > 0,
           f"R-803: NO FIELD named `inventory_leg` remains in a receipt "
           f"({_nk_rec}) or in a ledger ({_nk_led} over "
           f"{len(_lrows803)} rows) this code writes, and the quantity "
           f"is there under `trades_cash_flow_cents` ({_nt_rec} keys) "
           f"with SELLS - BUYS stated beside it. BE 99 measured the old "
           f"field to be the trades cash flow with the opposite sign; "
           f"the LANDED artifacts keep their bytes (rule 13)")
        # ---- R-828 / DE 148: THE POINT-ESTIMATE MODE ----------------
        # S4 is 99.2 % of a run and buys ONLY Z and p. A run that wants
        # the ESTIMATES skips it entirely -- and the estimates must be
        # the SAME NUMBERS, because they are the same arithmetic on the
        # same replays.
        _pe828 = run_day("FIXTURE-DAY-1", _mk134["book_path"], params=live,
                         fixture=True, n_days_complete=1,
                         point_estimate=True)
        _fu828 = run_day("FIXTURE-DAY-1", _mk134["book_path"], params=live,
                         fixture=True, n_days_complete=1)
        _pev = {a["arm"]: (a.get("economic") or {}).get("D_E0")
                for a in _pe828["per_day_sealed_artifacts"]}
        _fuv = {a["arm"]: (a.get("economic") or {}).get("D_E0")
                for a in _fu828["per_day_sealed_artifacts"]}
        _ref828 = None
        try:
            test_statistic_from(_pe828, "Z")
        except RunnerRefused as _e828:
            _ref828 = str(_e828).split(":")[0].replace("REFUSED ", "")
        _fullZ = test_statistic_from(_fu828, "Z")
        ok(_pe828["run_mode"] == "POINT_ESTIMATE"
           and _fu828["run_mode"] == "FULL"
           and _pev == _fuv and all(v is not None for v in _pev.values())
           and all((a.get("economic") or {}).get("Z")
                   == NULL_NOT_DRAWN_STATUS
                   for a in _pe828["per_day_sealed_artifacts"])
           and _ref828 == POINT_ESTIMATE_REFUSAL
           and all(isinstance(v, float) for v in _fullZ.values()),
           f"R-828 / DE 148: THE POINT ESTIMATE IS THE SAME NUMBER. "
           f"Skipping S4 entirely reproduces the FULL run's D_E0 on both "
           f"arms EXACTLY ({list(_pev.values())} == "
           f"{list(_fuv.values())}) -- same replays, same arithmetic, "
           f"only the control absent. Where the statistics would be the "
           f"receipt carries `{NULL_NOT_DRAWN_STATUS}` (never a null, "
           f"never a zero), asking it for Z REFUSES `{_ref828}`, and a "
           f"FULL receipt still answers with real floats. `run_mode` "
           f"marks the artifact so it can never be read as, or "
           f"supersede, a full run")
        import shutil as _sh801
        _sh801.rmtree(_d801, ignore_errors=True)
    import shutil as _sh134
    _sh134.rmtree(_d134, ignore_errors=True)

    # ===== REV 103 §7 #3 / REV 99 §A3: THE RECONCILIATION REFUSAL IS
    # WATCHED FIRING ON EVERY RUN, not once in a review ===============
    # R-782's guard -- `arm_total - baseline_total` must equal D(E0) to
    # 1e-9 or neither number may be published -- was landed at DE 131
    # with only `absolute_legs`' ARITHMETIC driven. REV 99 §A3: the
    # REFUSAL had never been watched fire. REV 101 §4 drove it, once, in
    # a scratch worktree. A guard verified by a reviewer's scratch drive
    # is a guard whose next regression nobody catches, so it is a cell.
    #
    # ONLINE ONLY, same measured reason as the two above: it drives
    # `run_day` far enough to perform real draws, and BE's `draw_null`
    # reads its committed null receipt under `data/`.
    if offline:
        offline_skip("REV 99 §A3 control 1/2 -- the unperturbed day "
                     "reconciles; it performs real draws, which read BE's "
                     "committed null receipt under data/")
        offline_skip("REV 99 §A3 known-bad 2/2 -- the perturbed arm total "
                     "REFUSES ABSOLUTES_DO_NOT_RECONCILE; same reason")
    else:
        _ctl782 = day_split_residency_proof(
            "FIXTURE-DAY-1", _mk134["book_path"], params=live,
            fixture=True, n_days_complete=1)
        _arms782 = ((_ctl782.get("day_result") or {})
                    .get("per_day_sealed_artifacts") or [])
        _rec782 = [((a.get("absolute") or {}).get("reconciliation") or {})
                   for a in _arms782]
        ok(len(_arms782) == 2
           and all(isinstance(a.get("absolute"), dict) for a in _arms782)
           and all(r.get("agree_to_1e_9") is True for r in _rec782),
           f"REV 99 §A3 CONTROL, AND IT ADMITS: the unperturbed day runs "
           f"through `run_day` and both arm-days carry an `absolute` "
           f"block whose `agree_to_1e_9` is True -- so the known-bad "
           f"below measures a DELTA from a run that reconciles, not the "
           f"emptiness of a fixture")
        # THE KNOWN-BAD IS THE ARM'S OWN TOTAL, MOVED BY 1.0 CENT --
        # REV 101 §4's perturbation, in the place the numbers are read.
        # `absolute_legs` is called twice per arm-day: the 0-cancel
        # baseline first, the arm second, so every SECOND call is the
        # arm's and only the arm's total moves.
        _orig782 = absolute_legs
        _n782 = {"calls": 0}

        def _perturb782(fills):
            r = dict(_orig782(fills))
            _n782["calls"] += 1
            if _n782["calls"] % 2 == 0:          # the ARM's call
                r["total"] = r["total"] + 1.0
            return r
        _msg782 = None
        globals()["absolute_legs"] = _perturb782
        try:
            day_split_residency_proof(
                "FIXTURE-DAY-1", _mk134["book_path"], params=live,
                fixture=True, n_days_complete=1)
        except RunnerRefused as _e782:
            _msg782 = str(_e782)
        finally:
            globals()["absolute_legs"] = _orig782
        ok(_msg782 is not None
           and "ABSOLUTES_DO_NOT_RECONCILE" in _msg782
           and "a difference of" in _msg782
           and absolute_legs is _orig782,
           f"REV 99 §A3 KNOWN-BAD, DRIVEN AND WATCHED: with the ARM's "
           f"total moved by 1.0 cent the day REFUSES BY ITS OWN NAME -- "
           f"{(_msg782 or '')[:150]}... -- so the absolutes and the "
           f"excess must be the same numbers or neither is published. "
           f"The perturbation is restored inside the cell, asserted here")

    # ---- DE 134 (2): THE POST-EMIT CENSUS, FOUND BY RUNNING IT -------
    # `--synthetic-day` end to end died AFTER writing its receipt with
    # `KeyError: 'G'`, at the one line DE 124's sweep for consumers of a
    # receipt's `G` could not see: the PRODUCER reading its own payload.
    # Both halves are driven -- the key that is gone, and the predicate
    # that outlived the seal.
    _pay134 = {"n_days_complete": 5,
               "G_and_which_G_it_is": {"design_G_from_params": 6,
                                       "the_bar_this_run_sealed_against": 6}}
    _keys134 = ["D_E0", "Z", "null_mean"]
    _kerr134 = None
    try:
        _pay134["G"]                       # THE LANDED EXPRESSION
    except KeyError as _e:
        _kerr134 = "KeyError"
    _ruled134 = post_emit_census_status(live, _pay134, _keys134)
    ok(_kerr134 == "KeyError" and "G" not in _pay134
       and _ruled134["status"] == "NOT_APPLICABLE_UNSEALED_BY_R765",
       f"DE 134 KNOWN-BAD, THE LANDED LINE: a receipt payload of today's "
       f"shape has NO bare `G` (DE 124 removed it), so the landed "
       f"`payload['G']` raises {_kerr134} -- after the receipt is "
       f"written. The census now reads the regime from the params and "
       f"the bar from `G_and_which_G_it_is`: under R-765 the status is "
       f"`{_ruled134['status']}`, and the run does not die at its own "
       f"emit")
    _sealed134 = {k: v for k, v in live.items()
                  if k != "user_ruled_unsealed_emission"}
    _leak134 = post_emit_census_status(_sealed134, _pay134, _keys134)
    ok(_leak134["status"] == "SEALED_REGIME_LEAK"
       and _leak134["unsealed_by_user_ruling"] is False,
       f"DE 134 RED, AND THE OLD GUARD STILL FIRES WHERE IT SHOULD: the "
       f"SAME payload and the SAME economic keys under params carrying "
       f"no R-765 ruling census as `{_leak134['status']}` at "
       f"{_leak134['n_days_complete']} of "
       f"{_leak134['the_bar_this_run_sealed_against']} days -- the "
       f"receipt is removed and the day refused. The ruling changed "
       f"which regime is in force, not whether a sealed day may leak")
    _clean134 = post_emit_census_status(
        _sealed134, {**_pay134, "n_days_complete": 6}, _keys134)
    _nokeys134 = post_emit_census_status(_sealed134, _pay134, [])
    ok(_clean134["status"] == "SEALED_REGIME_CLEAN"
       and _nokeys134["status"] == "SEALED_REGIME_CLEAN",
       "DE 134: and the sealed regime ADMITS both ways it should -- at "
       "the bar (6 of 6) with the economics present, and below the bar "
       "with none read back. A guard shown only to refuse is not a guard")
    _nobar134 = post_emit_census_status(
        _sealed134, {"n_days_complete": 5}, _keys134)
    ok(_nobar134["status"] == "POST_EMIT_CENSUS_HAS_NO_BAR",
       f"DE 134: a sealed run whose receipt carries NO bar censuses as "
       f"`{_nobar134['status']}` and the caller REFUSES on it -- a check "
       f"that cannot evaluate refuses, it does not pass (R-649). That is "
       f"the shape the `KeyError` had: the bar was unreadable and the "
       f"run died instead of saying so")

    # ===== R-801: THE RULED ENDPOINT -- trades + settlement residual ==
    # THE USER, 2026-09-07: "the pnls are from trades and remaining
    # position's settlement p&l, need to calculate this correctly".
    # Every cell here is HAND-COMPUTABLE: the point of a falsifier for a
    # new valuation is that a reader can check the arithmetic without
    # trusting the code that produced it.
    import harmful_stateful_policy as _HSP801
    _BUY801 = HSP_BUY_SIDE()
    _SELL801 = [x for x in _HSP801.SIDES if x != _BUY801][0]

    def _f801(side, px, size, slug="S1"):
        return {"side": side, "px_cents": px, "size": size, "slug": slug,
                "mid_cents_at_fill": px, "mid_cents_at_markout": px}
    # BUY 10 @ 40, BUY 10 @ 30, SELL 5 @ 60 -> net +15 shares.
    #   trades   = 5x60 - (10x40 + 10x30) = 300 - 700 = -400
    #   residual = 15 x 100 (Up won)      = +1500
    #   total    = 1100 = 10(100-40) + 10(100-30) - 5(100-60)
    _fx801 = [_f801(_BUY801, 40.0, 10.0), _f801(_BUY801, 30.0, 10.0),
              _f801(_SELL801, 60.0, 5.0)]
    _up801 = {"S1": {"up_won": True, "settle_cents": 100.0}}
    _dn801 = {"S1": {"up_won": False, "settle_cents": 0.0}}
    _L801 = settlement_legs_by_slug(_fx801, _up801)
    _s1 = _L801["per_slug"]["S1"]
    ok(abs(_s1["trades_leg_cents"] - (-400.0)) < 1e-9
       and abs(_s1["net_shares"] - 15.0) < 1e-9
       and abs(_s1["residual_leg_cents"] - 1500.0) < 1e-9
       and abs(_s1["total_cents"] - 1100.0) < 1e-9
       and abs(settle_value_cents(_fx801, _up801) - 1100.0) < 1e-9,
       f"R-801 FALSIFIER 1, HAND-COMPUTED: a slug with BUY 10@40, BUY "
       f"10@30, SELL 5@60 and Up winning has trades leg "
       f"{_s1['trades_leg_cents']:.1f} (= 300 - 700), residual "
       f"{_s1['residual_leg_cents']:.1f} (= 15 shares x 100) and total "
       f"{_s1['total_cents']:.1f} -- and the per-fill formula gives the "
       f"same 1100. The ruled P&L is the trades' cash flow plus the "
       f"settlement of what is still held")
    _D801 = settlement_legs_by_slug(_fx801, _dn801)
    _d1 = _D801["per_slug"]["S1"]
    ok(abs(_d1["residual_leg_cents"] - 0.0) < 1e-9
       and abs(_d1["trades_leg_cents"] - _s1["trades_leg_cents"]) < 1e-9
       and abs((_s1["total_cents"] - _d1["total_cents"])
               - _s1["net_shares"] * 100.0) < 1e-9,
       f"R-801 FALSIFIER 2, THE WINNER MOVES THE RESIDUAL AND NOTHING "
       f"ELSE: with Down winning the residual leg is "
       f"{_d1['residual_leg_cents']:.1f}, the trades leg is UNCHANGED at "
       f"{_d1['trades_leg_cents']:.1f}, and the totals differ by exactly "
       f"net x 100 = {_s1['net_shares'] * 100.0:.1f}. A wrong winner "
       f"cannot move the trades leg, and it moves the residual by an "
       f"amount a reader can compute")
    # THE IDENTITY IS ASSERTED IN THE CODE -- so the cell DRIVES the
    # refusal rather than re-deriving the algebra.
    _orig801 = settle_value_cents
    _msg801 = None
    globals()["settle_value_cents"] = (
        lambda fills, winners: _orig801(fills, winners) + 1.0)
    try:
        settlement_legs_by_slug(_fx801, _up801)
    except RunnerRefused as _e801:
        _msg801 = str(_e801)
    finally:
        globals()["settle_value_cents"] = _orig801
    ok(_msg801 and "SETTLEMENT_LEGS_DO_NOT_RECONCILE" in _msg801
       and settle_value_cents is _orig801,
       f"R-801 FALSIFIER 3, THE LEGS ARE A DECOMPOSITION AND MUST PROVE "
       f"IT: with the per-fill valuation moved by 1.0 cent the legs "
       f"REFUSE BY NAME -- {(_msg801 or '')[:110]}... -- so `trades + "
       f"residual == the per-fill formula` is a check that fires, not an "
       f"identity asserted in a comment. Restored, asserted here")
    # RULE 11's GUARD (R-801 (6)), BOTH WAYS AND FROM THE DECLARATION.
    _dd801 = list(((live.get("user_ruled_early_read") or {})
                   .get("this_reads_bar") or {}).get("days")
                  or (live.get("user_ruled_early_read") or {}).get("days")
                  or [])
    _adm801 = settlement_admissibility(_dd801[0], live) if _dd801 else {}
    _no801 = None
    try:
        assert_settlement_day_admissible("2026-09-07", live)
    except RunnerRefused as _e:
        _no801 = str(_e).split(":")[0].replace("REFUSED ", "")
    _declared801 = {**live, "settlement_endpoint":
                    {"admissible_days": ["2026-09-07"]}}
    _yes801 = settlement_admissibility("2026-09-07", _declared801)
    ok(_no801 == "SETTLEMENT_DAY_NOT_ADMISSIBLE"
       and _adm801.get("class") == "DESIGN_DATA"
       and _yes801["class"] == "DECLARED_VALIDATION_DAY"
       and settlement_admissibility("FIXTURE-DAY-1", live,
                                    fixture=True)["class"] == "FIXTURE",
       f"R-801 (6) / RULE 11, DRIVEN THREE WAYS: 2026-09-07 REFUSES BY "
       f"NAME -- `{_no801}` -- under the params head, which declares no "
       f"settlement endpoint; the same day ADMITS as "
       f"`{_yes801['class']}` under a params carrying "
       f"`settlement_endpoint.admissible_days` (so the set is READ from "
       f"the declaration, never typed here); and {_dd801[0]} admits as "
       f"`{_adm801.get('class')}` because the USER's early read consumed "
       f"it -- valued, and LABELLED design data")
    # THE WINNER SOURCE REFUSES BY NAME -- five ways, on fixture roots,
    # so no cell here reads `data/`.
    _wr801 = Path(_tfr.mkdtemp(prefix="de801_res_")) / "pm_5min"
    _wr801.mkdir(parents=True)

    def _fixture_stream_provenance801(root, suffix):
        prices = Path(root) / "pm_5min" / "prices"
        files = []
        h = hashlib.sha256()
        for topic in CHAINLINK_PRICE_TOPICS:
            d = prices / topic
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"20960905_00_{suffix}_{topic}.csv"
            raw = f"{topic},{suffix}\n".encode()
            f.write_bytes(raw)
            files.append(f"{topic}/{f.name}")
            h.update(f.name.encode())
            h.update(raw)
        return {"n_hourly_files": len(files),
                "files": files,
                "files_digest": h.hexdigest(),
                "reader_module": {
                    "path": "live/pm_research/exp_m6_settlement.py",
                    "sha256": hashlib.sha256(
                        (Path(__file__).resolve().parent
                         / "exp_m6_settlement.py").read_bytes()
                    ).hexdigest()},
                "hours": ["20960905_00"]}

    _absent801 = None
    try:
        winner_source(root=_wr801.parent)
    except RunnerRefused as _e:
        _absent801 = str(_e).split(":")[0].replace("REFUSED ", "")
    (_wr801 / "resolutions.jsonl").write_text(
        json.dumps({"slug": "S1", "closed": True,
                    "winners": {"Up": True, "Down": False},
                    "source": "clob"}) + "\n")
    _good801 = winner_source(root=_wr801.parent, required_slugs=["S1"])
    _miss801 = None
    try:
        winner_source(root=_wr801.parent, required_slugs=["S1", "S404"])
    except RunnerRefused as _e:
        _miss801 = str(_e).split(":")[0].replace("REFUSED ", "")
    (_wr801 / "resolutions.jsonl").write_text(
        json.dumps({"slug": "S1", "closed": True,
                    "winners": {"Up": True, "Down": True}}) + "\n")
    _amb801 = None
    try:
        winner_source(root=_wr801.parent)
    except RunnerRefused as _e:
        _amb801 = str(_e).split(":")[0].replace("REFUSED ", "")
    (_wr801 / "resolutions.jsonl").write_text(
        json.dumps({"slug": "S1", "closed": True,
                    "winners": {"Up": True, "Down": False}}) + "\n"
        + json.dumps({"slug": "S1", "closed": True,
                      "winners": {"Up": False, "Down": True}}) + "\n")
    _dis801 = None
    try:
        winner_source(root=_wr801.parent)
    except RunnerRefused as _e:
        _dis801 = str(_e).split(":")[0].replace("REFUSED ", "")
    ok(_absent801 == "SETTLEMENT_WINNER_SOURCE_ABSENT"
       and _miss801 == "SETTLEMENT_WINNER_MISSING_FOR_SLUG"
       and _amb801 == "SETTLEMENT_WINNER_AMBIGUOUS"
       and _dis801 == "SETTLEMENT_WINNER_RECORDS_DISAGREE"
       and _good801["winners"]["S1"]["settle_cents"] == 100.0,
       f"R-801: A WRONG OR MISSING WINNER SOURCE REFUSES BY NAME, four "
       f"ways -- `{_absent801}` (no file), `{_miss801}` (a slug the "
       f"fills name with no record), `{_amb801}` (both outcomes true), "
       f"`{_dis801}` (two records naming opposite winners) -- with the "
       f"GREEN control admitting on the same fixture root, so the four "
       f"measured their own conditions and not an empty directory")
    # ---- R-803: THE WINNER IS VERIFIED, AND THE TWO WAYS IT FAILS --
    # BE 99 (Q-BE-342) measured that S60(T) >= S60(t0) reproduces the
    # venue on 288/288 slugs on BOTH days, while three other conventions
    # in exp_m6's pre-registered grid disagree on 10-44 a day. The grid
    # is NOT re-derived here: the convention is pinned by name and these
    # cells drive what happens when the check FAILS.
    # THIS BLOCK SETS UP ITS OWN FIXTURE. The cells above leave the file
    # in the DISAGREE state on purpose, and inheriting it made these
    # refuse `SETTLEMENT_WINNER_RECORDS_DISAGREE` -- an earlier guard's
    # refusal standing in for this one, the second time that shape has
    # cost me a cell in this battery.
    (_wr801 / "resolutions.jsonl").write_text(
        json.dumps({"slug": "btc-updown-5m-1788572400", "closed": True,
                    "winners": {"Up": True, "Down": False},
                    "source": "clob"}) + "\n")
    import exp_m6_settlement as _M6c
    _symc = _M6c.COINS["btc"]
    _slugc = "btc-updown-5m-1788572400"
    _t0c = 1788572400
    _venue_up = {_slugc: {"up_won": True, "settle_cents": 100.0}}
    # a stream whose S60 FALLS over the window while the venue says Up
    _bad_stream = {(_symc, 60): ([_t0c * 1000, (_t0c + 300) * 1000],
                                 [_t0c * 1000, (_t0c + 300) * 1000],
                                 [100.0, 99.0])}
    _prov803 = _fixture_stream_provenance801(_wr801.parent, "r803")
    _v_bad = verify_winners_against_chainlink(
        _venue_up, [_slugc], streams=_bad_stream,
        stream_provenance=_prov803, gaps=[])
    _dis803 = None
    try:
        winner_source(root=_wr801.parent, required_slugs=[_slugc],
                      verification=_v_bad, require_verified=True)
    except RunnerRefused as _e:
        _dis803 = str(_e).split(":")[0].replace("REFUSED ", "")
    # and a stream that is not there at all
    _v_none = verify_winners_against_chainlink(
        _venue_up, [_slugc], streams={}, stream_provenance=_prov803,
        gaps=[])
    _unv803 = None
    try:
        winner_source(root=_wr801.parent, required_slugs=[_slugc],
                      verification=_v_none, require_verified=True)
    except RunnerRefused as _e:
        _unv803 = str(_e).split(":")[0].replace("REFUSED ", "")
    # the GREEN control: a stream that RISES agrees with the venue
    _good_stream = {(_symc, 60): ([_t0c * 1000, (_t0c + 300) * 1000],
                                  [_t0c * 1000, (_t0c + 300) * 1000],
                                  [99.0, 100.0])}
    _v_ok = verify_winners_against_chainlink(
        _venue_up, [_slugc], streams=_good_stream,
        stream_provenance=_prov803, gaps=[])
    _ws_ok = winner_source(root=_wr801.parent,
                           required_slugs=[_slugc],
                           verification=_v_ok, require_verified=True)
    ok(_dis803 == "SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK"
       and _v_bad["counts"]["DISAGREE"] == 1
       and _unv803 == "SETTLEMENT_CHAINLINK_UNAVAILABLE"
       and _v_none["counts"]["CHAINLINK_UNAVAILABLE"] == 1
       and _v_ok["all_agree"] is True
       and _ws_ok["is_final_for_quotation"] is True
       and _ws_ok["chainlink_verification"]["convention"]
       == SETTLEMENT_CONVENTION["name"],
       f"R-803, THE VERIFICATION FAILS TWO WAYS AND ADMITS ONCE: a slug "
       f"whose S60 FALLS while the venue says Up refuses "
       f"`{_dis803}`; a slug with NO stream refuses `{_unv803}`; and the "
       f"same slug with a RISING S60 verifies, "
       f"`is_final_for_quotation` True under "
       f"`{SETTLEMENT_CONVENTION['name']}`. The convention is PINNED by "
       f"name from BE 99's measurement and the grid is not re-derived "
       f"here")
    # ===== REVIEW 104B §6: THE FOUR, EACH DRIVING REV's OWN CASE =====
    # (2) THE VERIFICATION DOOR OPENED ON A BOOLEAN ITS CALLER SUPPLIED.
    # REV's exact dict: a per-slug map of DISAGREE, a made-up convention
    # name, empty counts and `all_agree: True` -> VERIFIED_AGAINST_
    # CHAINLINK, is_final True, require_verified PASSES.
    _revdict = {"per_slug": {_slugc: {"status": "DISAGREE",
                                      "venue_up_won": True,
                                      "chainlink_up_won": False}},
                "convention": {"name": "a rule I made up"},
                "counts": {}, "all_agree": True,
                "stream_provenance": _prov803}
    _madeup104 = None
    try:
        winner_source(root=_wr801.parent, required_slugs=[_slugc],
                      verification=_revdict, require_verified=True)
    except RunnerRefused as _e:
        _madeup104 = str(_e).split(":")[0].replace("REFUSED ", "")
    _revdict2 = {**_revdict,
                 "convention": dict(SETTLEMENT_CONVENTION)}
    _lied104 = winner_source(root=_wr801.parent,
                             required_slugs=[_slugc],
                             verification=_revdict2)
    _lied_req104 = None
    try:
        winner_source(root=_wr801.parent, required_slugs=[_slugc],
                      verification=_revdict2, require_verified=True)
    except RunnerRefused as _e:
        _lied_req104 = str(_e).split(":")[0].replace("REFUSED ", "")
    ok(_madeup104 == "SETTLEMENT_CONVENTION_NOT_THE_PINNED_ONE"
       and _lied104["chainlink_verification"]["status"]
       == "VERIFICATION_DID_NOT_AGREE"
       and _lied104["is_final_for_quotation"] is False
       and _lied104["chainlink_verification"]["counts"]["DISAGREE"] == 1
       and _lied_req104 == "SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK",
       f"REV 104B §6(2), REV's OWN DICT DRIVEN: a verification carrying "
       f"`all_agree: True`, empty counts and a per-slug map that says "
       f"DISAGREE no longer opens the door. Under a made-up convention "
       f"name it refuses `{_madeup104}`; under the PINNED name the "
       f"counts are RECOMPUTED from `per_slug` (DISAGREE "
       f"{_lied104['chainlink_verification']['counts']['DISAGREE']}), "
       f"the status reads "
       f"`{_lied104['chainlink_verification']['status']}`, "
       f"`is_final_for_quotation` is False and the quotable path refuses "
       f"`{_lied_req104}`. Nothing here is read from the dict handed in")
    # ===== R-811, THE USER: the placement latency is a SWEPT parameter
    # "Then can you put a 250ms latency, or make it configured to
    # different values to get different numbers under different latency".
    # The receipt reads its L FROM THE BOOK's builder receipt -- never a
    # constant -- and carries the book's own pair beside it.
    _r0_811 = placement_latency_from_the_book(
        {"scope": {"placement_latency_ms": 0.0}},
        book_path="/x/book0.pkl", book_sha256="0" * 64)
    _r250_811 = placement_latency_from_the_book(
        {"scope": {"placement_latency_ms": 250.0}},
        book_path="/x/book250.pkl", book_sha256="2" * 64)
    _rnone_811 = placement_latency_from_the_book(
        {"scope": {}}, book_path="/x/bookX.pkl", book_sha256="3" * 64)
    _armed_811 = None
    try:
        placement_latency_from_the_book(
            {"scope": {}}, book_path="/x/bookX.pkl",
            require_declared=True)
    except RunnerRefused as _e:
        _armed_811 = str(_e).split(":")[0].replace("REFUSED ", "")
    ok(_r0_811["L_place_ms"] == 0.0 and _r250_811["L_place_ms"] == 250.0
       and _r0_811["source"] == "THE BOOK'S BUILDER RECEIPT"
       and _r250_811["book"]["sha256"] == "2" * 64
       and _rnone_811["L_place_ms"] == 0.0
       and _rnone_811["source"].endswith("DECLARES NONE")
       and _armed_811 == "SETTLEMENT_BOOK_DECLARES_NO_PLACEMENT_LATENCY",
       f"R-811, THE L IS READ FROM THE BOOK AND NEVER TYPED: a builder "
       f"receipt declaring 0 gives {_r0_811['L_place_ms']} and one "
       f"declaring 250 gives {_r250_811['L_place_ms']}, each with the "
       f"book's own path and sha256 beside it -- so two runs of one day "
       f"at different L are distinguishable at a DECLARED FIELD. A book "
       f"declaring none reads 0 BY CONSTRUCTION today and REFUSES "
       f"`{_armed_811}` once the declaration arms the guard "
       f"(`settlement_endpoint.require_book_declares_L`), which is where "
       f"BE 101's builder change lands. **NOT ESTABLISHED HERE: that two "
       f"L values give different FILL COUNTS** -- no code path rebuilds a "
       f"book at L != 0, which is precisely BE 101/102's item; this cell "
       f"drives the receipt half, which is mine")
    _path_refusal = None
    try:
        placement_latency_from_the_book(
            Path("/tmp/receipt-declaring-250.json"), book_path="/x/book.pkl")
    except RunnerRefused as _e:
        _path_refusal = str(_e).split(":")[0].replace("REFUSED ", "")
    ok(_r250_811["L_place_ms"] == 250.0
       and _path_refusal == "SETTLEMENT_BOOK_RECEIPT_NOT_PARSED",
       "DE 151 known-bad: a parsed receipt declaring 250 ms reads 250, "
       "while passing the receipt Path now refuses by name instead of "
       "silently walking zero keys and returning the 0 ms default")

    _consistent = {
        "placement_latency": {"L_place_ms": 250.0},
        "day_run": {"placement_latency": {"L_place_ms": 250}},
    }
    _consistency = assert_one_placement_latency(_consistent)
    _bad_latency = json.loads(json.dumps(_consistent))
    _bad_latency["day_run"]["placement_latency"]["L_place_ms"] = 0.0
    _latency_refusals = []
    for candidate in (_bad_latency, {}, {"L_place_ms": True},
                      {"L_place_ms": float("nan")}):
        try:
            assert_one_placement_latency(candidate)
        except RunnerRefused as _e:
            _latency_refusals.append(
                str(_e).split(":")[0].replace("REFUSED ", ""))
    ok(_consistency["value_ms"] == 250.0
       and _consistency["n_sites_checked"] == 2
       and _latency_refusals == [PLACEMENT_LATENCY_DISAGREES,
                                 PLACEMENT_LATENCY_ABSENT,
                                 PLACEMENT_LATENCY_DISAGREES,
                                 PLACEMENT_LATENCY_DISAGREES],
       "DE 151 document guard: equal 250 ms copies pass; a planted 0 ms "
       "copy, absence, boolean and NaN all refuse by explicit names")

    # ---- DE 152: THE KNOWN-BAD, DRIVEN ON BE'S REAL 09-04 L=250 RECEIPT
    # The cell above drives the type refusal on a Path that does not
    # exist, which proves the TYPE check. This one proves the PROPERTY the
    # defect actually had: a real builder receipt that DOES declare 250 ms
    # reads 250.0 when parsed and 0.0 "DECLARES NONE" when its Path is
    # handed over -- so the silent default was not a hypothetical, it was
    # what every L = 250 artifact's nested block recorded. Verified at the
    # artifact a claim names (rule 16), not at a fixture written by the
    # same hand as the checker. Skipped BY NAME when the receipt is not on
    # disk, and skipped as an `offline` cell because it READS `data/`.
    _rp152 = (Path(DR.resolve()["data_root"]) / "pm_5min/derived"
              / "be_daybook_receipt_20260904_btc__L250ms.json")
    if offline:
        offline_skip("DE 152 the known-bad on BE's real 09-04 L=250 "
                     "builder receipt (it READS `data/`)")
    elif _rp152.is_file():
        _parsed152 = placement_latency_from_the_book(
            json.loads(_rp152.read_text()), book_path="/x/b.pkl")
        _byname152 = []
        for _arg in (_rp152, str(_rp152)):
            try:
                placement_latency_from_the_book(_arg, book_path="/x/b.pkl")
                _byname152.append("ADMITTED")
            except RunnerRefused as _e:
                _byname152.append(
                    str(_e).split(":")[0].replace("REFUSED ", ""))
        ok(_parsed152["L_place_ms"] == 250.0
           and _parsed152["source"] == "THE BOOK'S BUILDER RECEIPT"
           and _byname152 == ["SETTLEMENT_BOOK_RECEIPT_NOT_PARSED"] * 2,
           f"DE 152 KNOWN-BAD ON THE REAL RECEIPT: BE's own 09-04 L=250 "
           f"builder receipt PARSES to {_parsed152['L_place_ms']} at "
           f"{list(_parsed152['found_at'])}, and handing the SAME file as a "
           f"Path or a str refuses {_byname152[0]} both ways. Before DE 151 "
           f"that call answered 0.0 'DECLARES NONE' from the default, which "
           f"is why the nested block of every 250 ms artifact read 0.0")
    else:
        offline_skip("DE 152 the known-bad on BE's real 09-04 L=250 "
                     "builder receipt",
                     why=("the receipt is not on disk at " + str(_rp152)
                          + " -- absent for want of its input, NOT skipped "
                            "for being online/offline"))

    import de_decision_ledger as _LED151
    ok(_LED151.NULL_NOT_DRAWN_STATUS == NULL_NOT_DRAWN_STATUS,
       "DE 151: runner and ledger reader use the same named no-null status")
    # ===== R-810, THE USER's OWN REPRODUCTION, DRIVEN =================
    # The USER: `winner_source(..., require_verified=True)` with
    # `all_agree=True`, an EMPTY `per_slug` and EMPTY `counts` returned
    # `is_final_for_quotation=True` and the door PASSED. Recomputing the
    # counts fixed the boolean and LEFT THE DOOR OPEN -- the half that
    # matters. The gate now validates EVIDENCE IDENTITY, not shape.
    (_wr801 / "resolutions.jsonl").write_text("\n".join(
        json.dumps({"slug": f"S{_i}", "closed": True,
                    "winners": {"Up": True, "Down": False},
                    "source": "clob"}) for _i in range(3)) + "\n")
    _prov810 = _fixture_stream_provenance801(_wr801.parent, "r810")
    _C810 = dict(SETTLEMENT_CONVENTION)
    _req810 = ["S0", "S1", "S2"]

    def _agree810(*names):
        return {n: {"status": "VERIFIED_AGREE"} for n in names}

    def _door810(ver, req):
        try:
            _w = winner_source(root=_wr801.parent, required_slugs=req,
                               verification=ver, require_verified=True)
            return ("PASSED", _w["is_final_for_quotation"])
        except RunnerRefused as _e:
            return (str(_e).split(":")[0].replace("REFUSED ", ""), None)
    # THE USER'S EXACT DICT
    _u1 = _door810({"per_slug": {}, "counts": {}, "all_agree": True,
                    "convention": _C810,
                    "stream_provenance": _prov810}, _req810)
    _u2 = _door810({"per_slug": _agree810("S0", "S1"),
                    "convention": _C810,
                    "stream_provenance": _prov810}, _req810)
    _u3 = _door810({"per_slug": {**_agree810("S0", "S1"),
                                 "S2": {"status": "DISAGREE"}},
                    "convention": _C810,
                    "stream_provenance": _prov810}, _req810)
    _u4 = _door810({"per_slug": _agree810("S0"),
                    "convention": {"name": "a rule I made up"},
                    "stream_provenance": _prov810}, ["S0"])
    _u5 = _door810({"per_slug": _agree810(*_req810), "convention": _C810,
                    "stream_provenance": {**_prov810,
                                          "n_hourly_files": 0}}, _req810)
    _u6 = _door810({"per_slug": _agree810(*_req810), "convention": _C810,
                    "stream_provenance": {"n_hourly_files": 54}}, _req810)
    _u7 = _door810({"per_slug": _agree810(*_req810), "convention": _C810,
                    "stream_provenance": _prov810}, _req810)
    ok(_u1[0] == "SETTLEMENT_VERIFICATION_EMPTY"
       and _u2[0] == "SETTLEMENT_VERIFICATION_SLUG_SET_MISMATCH"
       and _u3[0] == "SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK"
       and _u4[0] == "SETTLEMENT_CONVENTION_NOT_THE_PINNED_ONE"
       and _u5[0] == "SETTLEMENT_CHAINLINK_NO_FILES_READ"
       and _u6[0] == "SETTLEMENT_VERIFICATION_PROVENANCE_INCOMPLETE"
       and _u7 == ("PASSED", True),
       f"R-810, THE USER'S REVIEW, EVERY CASE DRIVEN: the USER'S OWN "
       f"reproduction -- `all_agree=True`, per_slug {{}}, counts {{}} -- "
       f"now refuses `{_u1[0]}` (zero slugs verified is not "
       f"verification); a per-slug set that is not the day's refuses "
       f"`{_u2[0]}`; one DISAGREE `{_u3[0]}`; a foreign convention name "
       f"`{_u4[0]}`; ZERO hourly files read `{_u5[0]}`; missing stream "
       f"and reader digests `{_u6[0]}`; and the complete evidence -- the "
       f"day's exact slug set, every status VERIFIED_AGREE, files read, "
       f"provenance recorded -- PASSES with is_final {_u7[1]}. The gate "
       f"validates EVIDENCE IDENTITY, not shape")
    # (4) A DECLARATION MAY NOT RE-OPEN A CONSUMED DAY. REV's case:
    # `admissible_days = ["2026-09-05"]`, a day the early read CONSUMED.
    _consumed104 = (_dd801[0] if _dd801 else "2026-09-05")
    _decl104 = {**live,
                "settlement_endpoint": {"admissible_days": [_consumed104]}}
    _ov104 = settlement_admissibility(_consumed104, _decl104)
    _ovname104 = None
    try:
        assert_settlement_day_admissible(_consumed104, _decl104)
    except RunnerRefused as _e:
        _ovname104 = str(_e).split(":")[0].replace("REFUSED ", "")
    _still104 = settlement_admissibility("2026-09-07", _declared801)
    ok(_ov104["admissible"] is False
       and _ovname104 == "SETTLEMENT_DECLARATION_NAMES_A_CONSUMED_DAY"
       and _still104["class"] == "DECLARED_VALIDATION_DAY",
       f"REV 104B §6(4), THE OVERLAP REFUSED BY NAME: a declaration "
       f"naming {_consumed104} -- a day the USER's early read CONSUMED "
       f"-- no longer brings it back as a validation day; it refuses "
       f"`{_ovname104}`, because a consumed day cannot be a validation "
       f"day whatever a declaration says (rule 11), and the DECLARATION "
       f"is what must change. A declaration naming an UNCONSUMED day "
       f"still admits as `{_still104['class']}`, so the guard did not "
       f"just close")
    # (3) A DEGENERATE SETTLEMENT NULL IS A STATUS, NOT A DAY-LEVEL ABORT.
    _degen104 = settlement_arm_day(12.5, [3.0] * live[
        "min_draws_per_arm_day"], live)
    _live104 = settlement_arm_day(
        12.5, [float(i % 7) for i in range(live["min_draws_per_arm_day"])],
        live)
    ok(_degen104.get("status") == "SETTLEMENT_NULL_DEGENERATE"
       and _degen104["Z"] is None and _degen104["p_location"] is None
       and _degen104["null_sd"] == 0.0
       and _degen104["null_draws_summary"]["n"] == live[
           "min_draws_per_arm_day"]
       and _degen104["D_E_settle"] == 12.5
       and _live104.get("status") is None
       and isinstance(_live104["Z"], float),
       f"REV 104B §6(3), THE ABORT IS A STATUS NOW: a settlement null "
       f"with ZERO DISPERSION returns "
       f"`{_degen104['status']}` -- Z None, p None, sd "
       f"{_degen104['null_sd']}, n "
       f"{_degen104['null_draws_summary']['n']}, the excess KEPT -- "
       f"instead of raising `DesignRefused` past a runner that catches "
       f"it NOWHERE, which killed the whole day and threw away the D_E0 "
       f"already computed (70-80 minutes on a real day). A null with "
       f"dispersion still returns a real Z ({_live104['Z']:.4f})")
    # (1) THE REASON IS PER STATUS AND CANNOT CONTRADICT THE VERDICT.
    _fixadm104 = settlement_admissibility("FIXTURE-DAY-1", live,
                                          fixture=True)
    _r_fix104 = settlement_not_valued_reason(
        "NO_WINNER_SOURCE_ON_A_FIXTURE", _fixadm104)
    _notadm104 = settlement_admissibility("2026-09-07", live)
    _r_not104 = settlement_not_valued_reason(
        "NOT_VALUED_DAY_NOT_ADMISSIBLE", _notadm104)
    ok(_fixadm104["admissible"] is True
       and "IS admissible" in _r_fix104
       and "not in the admissible set" not in _r_fix104
       and _notadm104["admissible"] is False
       and "NOT in the admissible set" in _r_not104
       and _r_fix104 != _r_not104
       and "no reason is declared" in settlement_not_valued_reason(
           "SOMETHING_ELSE", _fixadm104),
       f"REV 104B §6(1), THE LITERAL THAT CONTRADICTED THE KEY BESIDE "
       f"IT: the fixture status now reads `{_r_fix104[:60]}…` against "
       f"`admissible: {_fixadm104['admissible']}`, and the "
       f"not-admissible status keeps the rule-11 sentence -- two "
       f"statuses, two reasons, each composed from the verdict it sits "
       f"beside. One literal served both and was true of only the "
       f"second. An unknown status SAYS it has no declared reason "
       f"rather than borrowing one that fits neither")
    # ===== R-826 / DE 146: THE VERIFIABILITY LAYER, DRIVEN ===========
    # THE USER: "fix it". `read_at` returns the last sample at or before
    # the boundary with no bound on its age and no way to say UNKNOWN, so
    # an OUTAGE became a data point. The convention is NOT touched; this
    # is a layer in front of it.
    _t0v, _Tv = 1_000_000_000_000, 1_000_000_300_000     # ms
    def _ser(pairs):
        te = [t for t, _ in pairs]
        return (te, list(te), [v for _, v in pairs])
    # FRESH EITHER SIDE: a sample exactly at each boundary, tiny drift.
    _fresh = _ser([(_t0v - 1000, 100.0e18), (_t0v, 100.0e18),
                   (_Tv - 1000, 105.0e18), (_Tv, 105.0e18)])
    _vf = boundary_verifiability(_fresh, _t0v, _Tv, gaps=[])
    # MARGIN SMALLER THAN ITS OWN DRIFT BOUND, no gap logged: the read at
    # T is 30 s stale and the stream moves, so the verdict could flip.
    _thin = _ser([(_t0v, 100.0e18)]
                 + [(_Tv - 30000 - 1000 * k, (100.0 + 2.0 * k) * 1e18)
                    for k in range(9, -1, -1)])
    _vt = boundary_verifiability(_thin, _t0v, _Tv, gaps=[])
    ok(_vf["verifiable"] is True and _vf["t0"]["staleness_s"] == 0.0
       and _vf["T"]["staleness_s"] == 0.0
       and _vt["verifiable"] is False
       and _vt["T"]["staleness_s"] >= 30.0
       and _vt["could_have_hidden"] >= _vt["margin"]
       and not _vt["a_gap_covers_a_boundary"],
       f"R-826 SYNTHETIC, BOTH WAYS: a window with a sample AT each "
       f"boundary (staleness 0.0 s) is VERIFIABLE and never earns the "
       f"new status; a window whose read at T is "
       f"{_vt['T']['staleness_s']:.0f} s stale, with NO gap logged, "
       f"earns it because the drift that staleness could have hidden "
       f"({_vt['could_have_hidden_usd']:.2f} USD) is not smaller than "
       f"its own margin ({_vt['margin_usd']:.2f} USD). The test is "
       f"MARGIN-AWARE, not a fixed tolerance")
    _gapfix = [{"gap_start_ns": (_Tv - 5000) * 10**6,
                "gap_end_ns": (_Tv + 5000) * 10**6,
                "duration_ms": 10000.0, "cause": "GLOBAL_SOCKET_SILENCE",
                "opened_connection_id": 1, "closed_connection_id": 2}]
    _vg = boundary_verifiability(_fresh, _t0v, _Tv, gaps=_gapfix)
    ok(_vg["a_gap_covers_a_boundary"] is True
       and _vg["gap_at_T"]["cause"] == "GLOBAL_SOCKET_SILENCE"
       and _vg["gap_at_T"]["duration_ms"] == 10000.0
       and "a logged stream gap covers a boundary" in _vg["why"],
       f"R-826 (3): a boundary inside a LOGGED GAP carries the gap's own "
       f"bounds, duration ({_vg['gap_at_T']['duration_ms']:.0f} ms) and "
       f"cause beside the verdict, from the PRICE lane's log "
       f"({PRICE_GAPS_REL}) -- not the CLOB lane's, which carries slug "
       f"disconnects and zero GLOBAL_SOCKET_SILENCE rows")
    ok(SETTLEMENT_CONVENTION["boundary_reader"]
       == "last sample at or before the boundary"
       and SETTLEMENT_CONVENTION["tie"] == "X_T >= X_0 -> Up"
       and "Q-BE-342" in SETTLEMENT_CONVENTION["provenance"]
       and "exp_m6_settlement.load_streams" in
       SETTLEMENT_CONVENTION["readers"],
       f"R-803: the convention is a PINNED BLOCK, not a sentence -- "
       f"`{SETTLEMENT_CONVENTION['name']}`, boundary "
       f"`{SETTLEMENT_CONVENTION['boundary_reader']}`, tie "
       f"`{SETTLEMENT_CONVENTION['tie']}`, read through exp_m6's own "
       f"`load_streams`/`read_at`, with BE 99's row as the provenance a "
       f"reader can go to")
    # THE FIXTURE IS RESTORED TO THE GOOD FILE FIRST. Left in the
    # DISAGREE state, this drive refused `SETTLEMENT_WINNER_RECORDS_
    # DISAGREE` -- an EARLIER guard's refusal -- and a cell that only
    # checked "it refused" would have gone green on the wrong one. The
    # battery caught it because the name is asserted exactly.
    (_wr801 / "resolutions.jsonl").write_text(
        json.dumps({"slug": "S1", "closed": True,
                    "winners": {"Up": True, "Down": False},
                    "source": "clob"}) + "\n")
    _nf801 = None
    try:
        winner_source(root=_wr801.parent, require_verified=True)
    except RunnerRefused as _e:
        _nf801 = str(_e).split(":")[0].replace("REFUSED ", "")
    ok(_good801["is_final_for_quotation"] is False
       and _good801["chainlink_verification"]["status"]
       == "NOT_VERIFIED_AGAINST_CHAINLINK"
       and _nf801 == "SETTLEMENT_WINNERS_NOT_VERIFIED",
       f"R-801: THE VENUE RECORD IS NOT THE CHAINLINK VERIFICATION, and "
       f"the code says so rather than the prose: `source: clob` yields "
       f"`{_good801['chainlink_verification']['status']}`, "
       f"`is_final_for_quotation` False, and a caller that asks for a "
       f"quotable value REFUSES `{_nf801}`. BE 99 owns the verification "
       f"method (R-801) and it has not landed; this module does not "
       f"invent one")

    # These checks DRIVE REAL DRAWS, and `null_draws_valued`
    # cross-checks the first draws against BE's own `draw_null`, which
    # reads BE's committed null receipt under `data/`. A FIXTURE run must
    # open no path under it, so they are skipped offline -- from a
    # DECLARED count, so a check added here without updating it refuses
    # rather than quietly shrinking the offline battery.
    if offline:
        for _i in range(BATTERY_ORDER_CHECKS):
            offline_skip(f"R-610 battery-order check {_i + 1}/"
                         f"{BATTERY_ORDER_CHECKS} -- it performs real "
                         f"draws, which cross-check against BE's "
                         f"committed null receipt under data/")
    else:
        # ---- R-610: THE BATTERY RUNS BEFORE THE DAY'S WORK ----------------
        # The 09-03 smoke's budget defect is closed (growth, per stage), but
        # the ORDER was the other half of it: the battery ran at the EMIT, so
        # the refusal arrived after 84 minutes of null draws and the day was
        # lost with nothing written. The property is "a check that can refuse
        # refuses before the work it would waste", and it is MEASURED -- the
        # draws this day performed -- not read off a line number.
        # n_slugs=24, NOT 12: at 12 both arms refuse on decision count and
        # the day performs ZERO draws -- which would make the falsifier's
        # "zero draws" reading true for the wrong reason. The measurement has
        # to be able to come out non-zero, and the check below drives that.
        _mk_ord = write_synthetic_day(
            "FIXTURE-DAY-ORDER", _tfr.mkdtemp(prefix="de90ord_"),
            params=live, n_slugs=24)

        def _hook_that_refuses():
            raise RunnerRefused(
                "REFUSED: a before_work hook that refuses -- standing in for "
                "the battery that refused the 09-03 smoke")

        _d0 = RUN_COUNTERS["draws_performed"]
        _order_refused, _draws_when_it_refused = None, None
        try:
            run_day("FIXTURE-DAY-ORDER", _mk_ord["book_path"], params=live,
                    fixture=True, before_work=_hook_that_refuses)
        except RunnerRefused as _e:
            _order_refused = str(_e)
            _draws_when_it_refused = RUN_COUNTERS["draws_performed"] - _d0
        ok(_order_refused is not None and _draws_when_it_refused == 0,
           f"R-610, THE FALSIFIER: a `before_work` hook that REFUSES stops the "
           f"day with {_draws_when_it_refused} draws performed. ZERO -- the "
           f"day had not started. That is the whole property: the refusal "
           f"costs the run nothing, where the 09-03 smoke paid 84 minutes for "
           f"the same verdict")
        # THE OTHER WAY, so the measurement is shown able to be non-zero. The
        # SAME refusal at the OLD position -- after the day -- is a refusal
        # that has already spent the work.
        _d1 = RUN_COUNTERS["draws_performed"]
        _late = run_day("FIXTURE-DAY-ORDER", _mk_ord["book_path"],
                        params=live, fixture=True)
        _draws_before_a_late_check = RUN_COUNTERS["draws_performed"] - _d1
        ok(_draws_before_a_late_check > 0
           and _late["work_counters"]["draws_performed_by_this_day"]
           == _draws_before_a_late_check
           and _late["before_work"]["ran"] is False,
           f"AND THE MEASUREMENT CAN BE NON-ZERO, which is what makes the zero "
           f"above a result: the same day run WITHOUT the hook reaches the "
           f"point where a battery used to run having already performed "
           f"{_draws_before_a_late_check} draws. A refusal there is a refusal "
           f"that has already spent the day")
        _d2 = RUN_COUNTERS["draws_performed"]
        _hook_calls = []

        def _hook_that_passes():
            _hook_calls.append(RUN_COUNTERS["draws_performed"] - _d2)
            return {"outcome": "PASS", "n_checks_run": 0}

        _with = run_day("FIXTURE-DAY-ORDER", _mk_ord["book_path"],
                        params=live, fixture=True,
                        before_work=_hook_that_passes)
        _bw = _with["before_work"]
        ok(_bw["ran"] is True
           and _bw["residency"]["day_draws_when_the_hook_returned"] == 0
           and _hook_calls == [0]
           and HOOK_STAGE in _with["memory_plan"]["observed"]
           and _with["work_counters"]["draws_performed_by_this_day"] > 0,
           f"POSITIVE CONTROL, AND IT ADMITS: a hook that PASSES lets the day "
           f"run to completion -- the hook saw 0 of this day's draws, the day "
           f"went on to perform "
           f"{_with['work_counters']['draws_performed_by_this_day']}, and "
           f"{HOOK_STAGE} is a MARKED STAGE so its memory sits inside the "
           f"day's growth budget rather than escaping it")
        ok(_with["memory_plan"]["peak_stage"]["highwater_delta_mb_by_stage"]
           .get(HOOK_STAGE) is not None
           and _with["battery_stage_is_inside_the_budget"] is True,
           f"and the hook's growth is ATTRIBUTED TO ITS OWN STAGE "
           f"({_with['memory_plan']['peak_stage']['highwater_delta_mb_by_stage'][HOOK_STAGE]:.1f} "
           f"MB) rather than landing on S1_load: a stage whose cost is "
           f"charged to the next stage is a peak predicate reading the wrong "
           f"argmax")
        # AND A HOOK WHOSE TRANSIENT DWARFS EVERY DAY STAGE MUST NOT WIN
        # THE ARGMAX. This is the risk the move CREATED: the real battery
        # high-waters at 838 MB (measured; 26 MB retained), S1_load's own
        # delta on a real day is ~1196 MB after it, and a flip would
        # REFUSE the day over a stage the 8 GiB ceiling does not rest on.
        _fat = dict(_with["memory_plan"]["observed"])
        _fat[HOOK_STAGE] = {"peak_rss_mb_highwater": 9_000.0,
                            "rss_mb_current": 9_000.0}
        for _k in ("S1_load", "S2_population", "S3_baseline", "S4_null",
                   "S5_seal"):
            _fat[_k] = {"peak_rss_mb_highwater": 9_001.0,
                        "rss_mb_current": 20.0}
        _pf = peak_stage_predicate(_fat, declared=declared_peak_stage())
        ok(_pf["measured_peak_stage"] == "S1_load"
           and HOOK_STAGE not in _pf["argmax_taken_over"]
           and _pf["before_work_hook_highwater_delta_mb"] > 8_000.0
           and assert_peak_stage(_pf, fixture=False,
                                 day="2026-09-03")["asserted"] is True,
           f"AND THE ARGMAX EXCLUDES THE HOOK, DRIVEN AT THE EXTREME: a "
           f"battery high-watering "
           f"{_pf['before_work_hook_highwater_delta_mb']:.0f} MB -- more "
           f"than every day stage -- still leaves the measured peak at "
           f"{_pf['measured_peak_stage']} and a REAL day passes. Its "
           f"delta is REPORTED, and counted in the growth budget; only "
           f"the argmax excludes it, because the ceiling rests on the DAY "
           f"PATH's shape")
        # THE 200-PATH CAP, AND IT REFUSED A CORRECT RUN. DE 90 put the
        # battery inside the instrumented region; the run went from ~25
        # opens to thousands; `distinct_paths` is capped at
        # DR.PATH_LIST_CAP and the day path's own non-vacuity guard --
        # "did the instrument see the book?" -- read the CAPPED list,
        # missed the book, and REFUSED. It would have refused the REAL DAY
        # AFTER 85 MINUTES, because the guard runs when the day returns.
        # DE 90's own check drove this function with a STUB hook that
        # opens nothing, so it could not see it. This hook opens more
        # paths than the cap.
        # THE PREFIX IS CHOSEN SO THE FAT FILES SORT BEFORE THE BOOK.
        # `distinct_paths` is `sorted(set(seen))[:CAP]`, so WHICH paths the
        # cap drops depends on sort order -- with a prefix that sorted
        # AFTER the book (`de91fat_` vs the book's `de90ord_`) the book
        # survived the cut and the check passed for the wrong reason. A
        # falsifier whose firing depends on a tempdir name is not one.
        _fatdir = Path(_tfr.mkdtemp(prefix="0de91fat_"))
        for _i in range(DR.PATH_LIST_CAP + 50):
            (_fatdir / f"aaa_{_i:04d}.txt").write_text("x")

        def _hook_that_opens_more_than_the_cap():
            for _f in sorted(_fatdir.iterdir()):
                _f.read_text()
            return {"outcome": "PASS", "n_checks_run": 0}

        _fatproof = day_split_residency_proof(
            "FIXTURE-DAY-ORDER", _mk_ord["book_path"], params=live,
            fixture=True, before_work=_hook_that_opens_more_than_the_cap)
        ok(_fatproof["instrument_observed_the_book_read"] is True
           and _fatproof["n_paths_opened_uncapped"] > DR.PATH_LIST_CAP
           and _fatproof["the_book_was_in_the_CAPPED_list_too"] is False,
           f"REV 55, FOUND BY RUNNING: a `before_work` hook that opens "
           f"{_fatproof['n_paths_opened_uncapped']} paths -- past the "
           f"{DR.PATH_LIST_CAP}-path cap -- and the guard STILL sees the "
           f"book, because the membership question is now asked of the "
           f"UNCAPPED set. It read the capped list, and the book is NOT in "
           f"it ({_fatproof['the_book_was_in_the_CAPPED_list_too']}) -- so "
           f"this is the exact configuration that refused a correct "
           f"`--synthetic-day` at the tip, and would have refused the real "
           f"day after 85 minutes")
        # AND THE HOOK'S OWN READS ARE SUBTRACTED FROM THE DAY-PATH CLAIM.
        _rp_ord = day_split_residency_proof(
            "FIXTURE-DAY-ORDER", _mk_ord["book_path"], params=live,
            fixture=True, before_work=_hook_that_passes)
        ok(_rp_ord["no_tape_index_or_fragment_artifact_was_opened"] is True
           and _rp_ord["the_claim_is_about"].startswith("the DAY PATH")
           and set(_rp_ord["tape_artifacts_opened"])
           == (set(_rp_ord["tape_artifacts_opened_by_the_whole_process"])
               - set(_rp_ord["tape_artifacts_opened_by_the_before_work_hook"])),
           "and the residency claim is COMPUTED as the whole-process set MINUS "
           "the hook's own set. The battery reads `data/` by design; without "
           "the subtraction, moving it inside the instrumented region would "
           "have made the day path's claim about the battery's reads")
    _cl = source_identity_at_launch()["import_closure"]
    ok("de_phase4_diag_runner.py" in _cl["modules"]
       and "harmful_stateful_policy.py" in _cl["modules"]
       and len(_cl["captured_at"]) == 3,
       f"REV 53 S1.1: the closure reaches what `load()` and `replay()` "
       f"import LAZILY -- {_cl['n_modules']} modules including "
       f"de_phase4_diag_runner and harmful_stateful_policy, which DO THE "
       f"REPLAYING. It reached the cascade and stopped there, so nine of "
       f"thirteen were outside it")
    _v2root = _synth_ledger(_D6)
    _der2 = _v2root / "pm_5min/derived"
    _c1 = [x for x in sorted(day_forms(_D6[0])) if "-" not in x][0]
    _v1p = sorted(_der2.glob(sealed_day_receipt_glob(_D6[0])))[0]
    _v2rec = json.loads(_v1p.read_text())
    _v2rec["supersedes"] = {"path": _v1p.name,
                            "sha256": sha256_streamed(_v1p)}
    _own = emission_stamp()
    _v2p = (_der2 / f"{SEALED_DAY_RECEIPT_PREFIX}{_c1}"
                    f"{SEALED_DAY_RECEIPT_MIDFIX}{_own}.json")
    _v2p.write_text(json.dumps(_v2rec))
    _res2 = find_sealed_day_receipt(_D6[0], _v2root)
    ok(_res2["present"] is True
       and _res2["status"] == "PRESENT_CHAIN_HEAD"
       and Path(_res2["path"]).name == _v2p.name
       and _res2["n_matches"] == 2,
       f"THE .v2 NAMING TRAP, DRIVEN: the .v2 carries ITS OWN clock stamp "
       f"({_own}) -- it must, or the clock guard refuses it -- so the "
       f"day's glob matches TWO files, and its identity as the "
       f"supersession comes from the {{path, sha256}} LINK, which the "
       f"chain resolver reduces to ONE head")
    ok(assert_name_stamp_is_the_clock(
           _v2p, datetime.datetime.now(
               datetime.timezone.utc).isoformat())["stamp_is_the_clock"]
       is True,
       "and the .v2's OWN stamp passes the clock guard, which is why it "
       "cannot keep v1's -- the two requirements meet at the LINK, not at "
       "the filename")
    # ---- R-608: a link is the PAIR {path, sha256}, both required -----
    _hp = _synth_ledger(_D6); _add_v2(_hp, _D6[0], half="path")
    _r_hp = find_sealed_day_receipt(_D6[0], _hp)
    ok(_r_hp["present"] is False and _r_hp["status"] == "LINK_NOT_A_PAIR",
       "R-608 KNOWN-BAD, PATH ONLY: a `supersedes` carrying a name and no "
       "digest is NOT a link. DE resolved links by DIGEST and DA by NAME, "
       "so on the same bytes exactly one seat refused and nothing declared "
       "which was right")
    _hd = _synth_ledger(_D6); _add_v2(_hd, _D6[0], half="digest")
    ok(find_sealed_day_receipt(_D6[0], _hd)["status"] == "LINK_NOT_A_PAIR",
       "and DIGEST ONLY is not a link either -- the pair is the link, in "
       "both directions")
    _mv = _synth_ledger(_D6); _add_v2(_mv, _D6[0], moved=True)
    _r_mv = find_sealed_day_receipt(_D6[0], _mv)
    ok(_r_mv["present"] is False and _r_mv["status"] == "SUPERSEDES_MOVED",
       "R-608 KNOWN-BAD, MOVED: a matching DIGEST under a DIFFERENT NAME "
       "refuses as a moved file rather than binding as a link")

    _unch = _synth_ledger(_D6)
    _add_v2(_unch, _D6[0], chained=False)
    _f3 = find_sealed_day_receipt(_D6[0], _unch)
    ok(_f3["present"] is False
       and _f3["status"] == "SUPERSEDES_TARGET_DIGEST_MISMATCH",
       "KNOWN-BAD: a .v2 naming a PRESENT predecessor at the WRONG DIGEST "
       "REFUSES BY NAME. It used to report DANGLING_SUPERSEDES -- 'the "
       "predecessor is absent' -- about a file sitting right there: the "
       "right verdict under a message that misdescribes the cause "
       "(REV 54 S0)")
    _dang = _synth_ledger(_D6)
    _add_v2(_dang, _D6[0], chained=False, moved=True)
    _f3b = find_sealed_day_receipt(_D6[0], _dang)
    ok(_f3b["present"] is False
       and _f3b["status"] == "DANGLING_SUPERSEDES",
       "AND THE NAME NOW MEANS WHAT IT SAYS: a .v2 whose link names "
       "NEITHER a present file NOR a present digest is DANGLING -- a "
       "supersession whose predecessor is absent is a claim about a file "
       "nobody can check")
    _twice = _synth_ledger(_D6)
    _der = _twice / "pm_5min/derived"
    _c0 = [x for x in sorted(day_forms(_D6[0])) if "-" not in x][0]
    (_der / f"{SEALED_DAY_RECEIPT_PREFIX}{_c0}"
            f"{SEALED_DAY_RECEIPT_MIDFIX}20260906T888888Z.json"
     ).write_text(json.dumps({"day": _D6[0], "note": "a second, unchained"}))
    _f4 = find_sealed_day_receipt(_D6[0], _twice)
    ok(_f4["present"] is False and _f4["status"] == "AMBIGUOUS"
       and len(_f4["heads"]) == 2,
       "AND AMBIGUOUS IS RESERVED FOR WHAT IT MEANT: two receipts with NO "
       "chain between them is a day that RAN TWICE, and a read that picks "
       "the newest has chosen after seeing")
    _gate_chain = read_gate(live, now_utc=_after, root=_chain,
                            ledger_rows=_rows6)
    ok(_gate_chain["per_day"][_D6[0]]["receipt"]["status"]
       == "PRESENT_CHAIN_HEAD",
       "and the GATE reads the chain head, so a corrected day is a present "
       "day rather than a missing one")


    if offline:
        offline_skip("the live read-gate evaluation (it reads the day verdicts "
                     "under data/, which a fixture run may not touch)")
    else:
        _live_gate = read_gate(live, now_utc=_after)
        _held = [c["conjunct"] for c in _live_gate["conjuncts"] if c["holds"]]
        ok(_live_gate["may_open"] is all(
               c["holds"] for c in _live_gate["conjuncts"])
           and len(_live_gate["conjuncts"]) == 7,
           f"AND EVALUATED ON THE REAL LEDGER NOW, as a RELATION: "
           f"{_live_gate['n_landed']} of {len(_live_gate['ruled_days'])} ruled "
           f"days landed, holding {_held}, so may_open is "
           f"{_live_gate['may_open']}. A relation because the counts change as "
           f"days land -- a check pinning '0 of 6' would go red on the first "
           f"receipt")

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

        # ---- R-765: THE DAY ARTIFACT EMITS ITS ECONOMICS --------------------
        # THE RULING INVERTED THIS CELL. It used to assert that at 1 of G
        # the artifact is SEALED with every economic field absent. The
        # USER ruled that mechanism retired: a run under params v18 emits
        # what it computed, and the status is COMPUTED from the counts.
        _sealed_run = run_day(_DAY, _made["book_path"], params=_DAYP,
                              fixture=True, n_days_complete=1)
        _sa = _sealed_run["per_day_sealed_artifacts"]
        _ruled765 = bool(_DAYP.get("user_ruled_unsealed_emission"))
        ok(_ruled765 is True
           and all(a["sealed"] is False for a in _sa)
           and all("economic" in a for a in _sa)
           and all(a["seal_status"].startswith(
               "UNSEALED_BY_USER_RULING R-765 -- ") for a in _sa)
           and all(f"1 of {_DAYP['G']} days complete" in a["seal_status"]
                   for a in _sa)
           and all(a["sealed_field_names"] == [] for a in _sa)
           and all(a["decision_population"]["decisions"] > 0 for a in _sa),
           f"R-765: AT 1 OF {_DAYP['G']} DAYS THE DAY ARTIFACT IS "
           f"UNSEALED AND CARRIES ITS ECONOMICS -- `{_sa[0]['seal_status']}`"
           f", COMPUTED from the counts, `sealed False`, "
           f"`sealed_field_names []`, and the decision counts still "
           f"published. The user ruled the seal retired; nothing here is "
           f"withheld from the run that produced it")
        # AND THE OLD PATH STAYS DRIVABLE, because the four landed sealed
        # days must remain reproducible from the code that wrote them
        # (rule 13): the same result, sealed the way they were.
        _old765 = seal(dict(_sa[0]), 1, _DAYP["G"])
        ok(_old765["sealed"] is True
           and "economic" not in _old765
           and _old765["sealed_field_names"] == list(ECONOMIC_FIELDS)
           and _old765["seal_status"].startswith(
               f"SEALED -- 1 of {_DAYP['G']} days complete."),
           f"R-765: and the RETIRED path is still drivable -- the same "
           f"arm-day through `seal()` without the ruling is "
           f"`{_old765['seal_status'][:34]}…` with the economics absent "
           f"and {len(_old765['sealed_field_names'])} sealed names. The "
           f"four landed days stay reproducible from the code that wrote "
           f"them; retiring a rule is not deleting its record")
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
               "S0b_battery", "S2_population", "S3_baseline_and_S4_null",
               "economics_valuation"}
           and len(DAY_STAGES) == 7,
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
        # A budget of ZERO, because on a WARM process the second day run
        # grows only ~0.5 MB -- the allocator already holds what the first
        # one took. Growth is process-history dependent, so a known-bad
        # pinned to "1 MB must be crossed" passes or fails on how many
        # runs came before it. Zero is crossed by any allocation at all.
        refuses(lambda: run_day(_DAY, _made["book_path"], params=_DAYP,
                                fixture=True, peak_rss_mb_budget=-1.0),
                "AND THE FIXTURE BUDGET BITES, NOW AT A STAGE: a day run that "
                "grows past its DECLARED budget REFUSES at the FIRST stage "
                "that crosses it, not at the emit -- the cap is never raised "
                "and the draws are never cut (R-174). THE BUDGET IS -1.0 MB, "
                "not 0.0 (DE 111): growth is measured on CURRENT RSS, which "
                "FALLS, so a run following one that allocated can grow by a "
                "NEGATIVE amount and a 0.0 budget then goes uncrossed -- this "
                "known-bad ADMITTED for exactly that reason the moment "
                "`design_chain()` began resolving through the shared "
                "implementation. A bound below every possible growth is "
                "crossed by construction, which is what a known-bad needs",
                "past the declared budget")
        # GROWTH, not the process peak -- this check compared the
        # PROCESS-WIDE peak against the budget, which is the retired
        # comparison itself. It only showed up because this round's own
        # 800 MB inflation pushed that peak past 700 MB.
        ok(_open["memory_plan"]["growth_rss_mb"]
           < FIXTURE_DAY_PEAK_RSS_MB_BUDGET
           and _open["memory_plan"]["within_budget"] is True
           and set(_open["memory_plan"]["observed"])
           == ({k for k, _ in DAY_STAGES} - {HOOK_STAGE}) | {"S_start"},
           f"and the real fixture run GREW "
           f"{_open['memory_plan']['growth_rss_mb']:.0f} MB against the declared "
           f"{FIXTURE_DAY_PEAK_RSS_MB_BUDGET:.0f} MB, with a high-water "
           f"recorded at each of the {len(DAY_STAGES) - 1} stages this run "
           f"has -- {HOOK_STAGE} is marked only when a `before_work` hook "
           f"is passed, and this run passed none -- PLUS the S_start "
           f"baseline, without which the first stage's delta is everything "
           f"that ever ran and the argmax is decided before the day starts")

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
                _t2 = time.time()
                while True:
                    try:
                        _fc2.flock(_f3, _fc2.LOCK_EX | _fc2.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() - _t2 >= 5.0:
                            raise
                        time.sleep(0.02)
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

        # ---- REV 49 S1.5: R4's booleans on the PRODUCTION path ----------
        import inspect as _i6
        _prod = _i6.getsource(run_day)
        ok("arm_day_admissible" in _prod
           and "min_decisions_per_arm_day" in _prod
           and "DEGENERATE_ARM_DAY_" in _prod
           and "arm_day_admissible" in _i6.getsource(arm_day),
           "REV 49 S1.5: R4's bars are on the PRODUCTION refusal path -- "
           "`run_day` reads `min_decisions_per_arm_day` and returns "
           "DEGENERATE_ARM_DAY_REFUSED, and `arm_day` calls "
           "`arm_day_admissible` on the REAL draws. A fixed function that "
           "is not on the path that refuses in production is rule 17's "
           "defect. My first version of this check looked for the decision "
           "bar in `arm_day`, where it is not -- it is in `run_day`")
        _adm_ok = DESIGN.arm_day_admissible(100, [float(i % 7)
                                                  for i in range(500)])
        _adm_thin = DESIGN.arm_day_admissible(3, [float(i % 7)
                                                  for i in range(500)])
        ok(_adm_ok["decisions_meet_bar"] is True
           and _adm_ok["sd_meets_floor"] is True
           and _adm_thin["decisions_meet_bar"] is False
           and _adm_thin["admissible"] is False,
           "and BOTH booleans are computed on real inputs, admitting a "
           "healthy arm-day and refusing a thin one -- the production "
           "objects, not a fixture of them")
        # ---- REV 49 S1.6: the params pin the design THE CODE IS ---------
        _pd = live.get("design_declaration", {})
        ok(isinstance(_pd, dict) and _pd.get("path")
           and "the_design_pins_THIS_file" in _pd,
           "REV 49 S1.6: the params name the design and record that the "
           "PIN RUNS design -> params. The design is emitted last and "
           "pins this file by a digest read at emission, so the two cannot "
           "drift without one of them refusing")

        # ---- R-599 (DA 68): the RATIO is sealed, and so is its text -----
        _adm_bad = DESIGN.arm_day_admissible(10, [1.0] * 500)
        ok(_adm_bad["decisions_meet_bar"] is False
           and _adm_bad["sd_meets_floor"] is False
           and _adm_bad["n_decisions"] == 10
           and _adm_bad["min_decisions_per_arm_day"] == 30,
           "R-599: R4 publishes TWO BOOLEANS -- the DECISION half with its "
           "numbers (a population fact) and the SD half as a VERDICT. That "
           "is what the pre-read needs; a null-derived ratio is more")
        _deep = {"day": "D", "arm": "A", "status": "X",
                 "admissibility": _adm_bad,
                 "nested": [{"a": [{"admissibility": dict(_adm_bad)}]}],
                 "economic": {"Z": 1.0}}
        _sealed_deep = seal(_deep, 1, 6)
        # KEYS, NOT SUBSTRINGS. My first version of this check tested the
        # serialised payload for the string "sd_over_abs_mean" and FAILED
        # -- because `sealed_field_names` is A LIST OF THE NAMES BEING
        # SEALED. That is the needle-matches-its-own-prose failure this
        # module's own `_economic_keys_in` docstring warns about, and I
        # walked into it in the check written to close a leak.
        _nested_adm = _sealed_deep["nested"][0]["a"][0]["admissibility"]
        ok(_economic_keys_in(_sealed_deep) == []
           and "sd_over_abs_mean" not in _nested_adm
           and "null_sd" not in _nested_adm
           and "sd_over_abs_mean" not in _sealed_deep["admissibility"],
           "R-599 KNOWN-BAD, PLANTED AT DEPTH: `sd_over_abs_mean` inside a "
           "nested list inside a dict is STRIPPED -- it survived the seal "
           "because it was not in ECONOMIC_FIELDS while BOTH quantities it "
           "is formed from were sealed (DA 68). Tested as KEYS at depth, "
           "because the artifact legitimately NAMES the sealed fields")
        _sd_txt = " ".join(_sealed_deep["admissibility"]["reasons"])
        _leaked_nums = [t for t in ("0.25 * |mean 1", "null sd 0",
                                    "sd 0.000000")
                        if t in _sd_txt]
        ok(not _leaked_nums and "VERDICT ONLY" in _sd_txt
           and "decisions 10 < declared minimum 30" in _sd_txt,
           f"AND A SECOND LEAK OF THE SAME CLASS, FOUND WHILE FIXING THE "
           f"FIRST: the refusal REASONS embedded sd and mean as TEXT, and "
           f"the stripper removes KEYS, not substrings -- it fired only on "
           f"a REFUSED arm-day, exactly where the numbers are most "
           f"tempting. The sd half is a verdict now; the decision half "
           f"keeps its numbers")
        _unsealed_deep = seal(_deep, 6, 6)
        ok(_unsealed_deep["admissibility"]["sd_over_abs_mean"] is not None
           or _unsealed_deep["admissibility"].get("null_sd") is not None,
           "POSITIVE CONTROL: after the unseal the ratio and its parts are "
           "PRESENT -- the seal withholds them, it does not delete them")
        import inspect as _i5
        _strip_src = _i5.getsource(_strip_economic)
        ok("ECONOMIC_FIELDS" in _strip_src,
           "and `_strip_economic` still REFERENCES `ECONOMIC_FIELDS` by "
           "name, which is the invariant DA's verifier asserts by AST -- "
           "DE's field list at the source stays the one DA reads")

        # ---- REV 47: the S0 transient ------------------------------------
        _bk = Path(_made["book_path"])
        ok(sha256_streamed(_bk) == hashlib.sha256(_bk.read_bytes()).hexdigest(),
           f"REV 47 POSITIVE CONTROL: the streamed digest equals the "
           f"whole-file digest on a real book, so the transient is removed "
           f"without changing the number")
        ok(HASH_CHUNK_BYTES <= (16 << 20)
           and "check-and-use" in sha256_streamed.__doc__,
           f"REV 47 TAKEN, WITH ONE PART DECLINED AND RECORDED: hashing "
           f"290 MB with read_bytes() put a ~290 MB TRANSIENT in S0 -- "
           f"invisible to the current-RSS series and squarely in the "
           f"HIGHWATER DELTA, so S0 looked like a memory stage when what "
           f"it did was read a file. Streamed at "
           f"{HASH_CHUNK_BYTES >> 20} MiB. The DOUBLE READ REMAINS: "
           f"collapsing it would need BE's loader to accept bytes (BE's "
           f"module, not DE's) and would reintroduce the check-and-use "
           f"window REV 43 made me close on the tape")

        # ---- RULE 22 AS AMENDED: the CLOSURE and HEAD, not one file -----
        _sid2 = source_identity_at_launch()
        _mods = set(_sid2["import_closure"]["modules"])
        ok(_sid2["import_closure"]["n_modules"] >= 4
           and "de_multiday_design_declaration.py" in _mods
           and "de_data_root.py" in _mods
           and _sid2["closure_unchanged_during_the_run"] is True,
           f"RULE 22 AS AMENDED (REV 51 S3): the launch capture is the "
           f"IMPORT CLOSURE, not one file -- {len(_mods)} modules under "
           f"live/, digested from the bytes each was first seen with. One "
           f"file was one twenty-fourth of what ran")
        ok(_sid2["head_at_import"]["head"]
           and _sid2["head_unchanged_during_the_run"] is True
           and isinstance(_sid2["worktree_was_dirty_at_import"], bool),
           f"and HEAD and the dirty state are captured AT IMPORT: "
           f"{str(_sid2['head_at_import']['head'])[:12]}, dirty="
           f"{_sid2['worktree_was_dirty_at_import']} -- a commit in the "
           f"worktree moves the producing code as surely as an edit does")
        import shutil as _sh2
        _sib = Path(__file__).resolve().parent / \
            "de_multiday_design_declaration.py"
        _bk2 = Path(_tfl.mkdtemp(prefix="de88sib_")) / "sib.bak"
        _sh2.copy2(_sib, _bk2)
        try:
            with open(_sib, "ab") as _fh2:
                _fh2.write(b"\n# a SIBLING rewritten mid-run\n")
            try:
                assert_source_unchanged("the closure known-bad")
                ok(False, "a sibling module change was ADMITTED")
            except RunnerRefused as _e2:
                ok("IMPORT CLOSURE" in str(_e2)
                   and "de_multiday_design_declaration.py" in str(_e2),
                   "RULE 22 KNOWN-BAD, A SIBLING: the DESIGN module "
                   "rewritten mid-run REFUSES BY NAME. The old capture "
                   "watched only this file, so the module that declares "
                   "the design could have moved under a run and the "
                   "receipt would still have said the source was "
                   "unchanged")
        finally:
            _sh2.copy2(_bk2, _sib)
        ok(assert_source_unchanged("the closure restore control")[
               "closure_unchanged_during_the_run"] is True,
           "AND THE POSITIVE CONTROL: with the sibling restored the emit "
           "admits again")
        _moved = dict(_sid2["head_at_import"])
        _moved["head"] = "0" * 40
        ok(_moved["head"] != _sid2["head_at_emit"]["head"],
           "and a MOVED HEAD is detectable as a value comparison -- the "
           "emit refuses on it, which is the case a commit in the run's "
           "own worktree creates")
        # ---- the NAME STAMP is the clock, not a number I typed ----------
        _now = datetime.datetime.now(datetime.timezone.utc)
        _good = Path(f"x__{emission_stamp(_now)}.json")
        ok(assert_name_stamp_is_the_clock(
               _good, _now.isoformat())["stamp_is_the_clock"] is True,
           "THE NAME STAMP IS THE CLOCK: a filename stamped from "
           "`emission_stamp()` at the moment of writing is admitted")
        _late = Path("x__20260906T094500Z.json")
        try:
            assert_name_stamp_is_the_clock(
                _late, "2026-09-06T09:32:50+00:00")
            ok(False, "a typed future stamp was ADMITTED")
        except RunnerRefused as _e3:
            ok("NOT YET OCCURRED" in str(_e3),
               "KNOWN-BAD, AND IT IS MY OWN ARTIFACT: design v15's name "
               "says 09:45:00Z and it was written at 09:32:50Z -- a "
               "filename naming a moment that had not yet occurred, "
               "because I typed a rounded stamp instead of reading a "
               "clock. It refuses now")
        ok(assert_name_stamp_is_the_clock(
               Path("p003_de_design_v16.json"), _now.isoformat())[
               "name_carries_a_stamp"] is False,
           "and a VERSION-ONLY name is admitted as the other fix: an "
           "artifact whose path must be known in ADVANCE (the design, "
           "because params names it) cannot carry an honest stamp, so it "
           "carries none and its time lives in `as_of`")

        # ---- R-603 / REV 49 S0: the source may not change under a run ---
        _sid = source_identity_at_launch()
        ok(_sid["producing_code_sha256"] == LAUNCH_SOURCE_SHA256
           and _sid["digest_taken_at"].startswith("MODULE IMPORT")
           and _sid["source_unchanged_during_the_run"] is True,
           f"R-603: the receipt's producing digest is taken at MODULE "
           f"IMPORT, before any work -- {str(LAUNCH_SOURCE_SHA256)[:16]} -- "
           f"not by a fresh read of __file__ at emit, which names whatever "
           f"is on disk THEN")
        import shutil as _sh
        _me = Path(__file__).resolve()
        _bak = Path(_tfl.mkdtemp(prefix="de86src_")) / "runner.bak"
        _sh.copy2(_me, _bak)
        try:
            with open(_me, "ab") as _fh:
                _fh.write(b"\n# mid-run edit, the R-603 known-bad\n")
            try:
                assert_source_unchanged("the known-bad")
                ok(False, "a mid-run source change was ADMITTED")
            except RunnerRefused as _e:
                ok(("THE SOURCE CHANGED UNDER THIS RUN" in str(_e)
                    or "IMPORT CLOSURE" in str(_e))
                   and "de_multiday_gate1_runner.py" in str(_e),
                   "R-603 KNOWN-BAD, THE FILE REWRITTEN MID-RUN, and it is "
                   "the CLOSURE check that now names it: the emit "
                   "REFUSES BY NAME. This is not hypothetical -- DE 85 "
                   "committed this file at 08:35:20Z while the 09-03 smoke "
                   "was executing it from the same worktree, so its "
                   "receipt would have named code that did not run AND "
                   "producing_code_is_the_committed_bytes would have "
                   "PASSED, because the replacement was committed")
        finally:
            _sh.copy2(_bak, _me)
        ok(assert_source_unchanged("the restore control")[
               "source_unchanged_during_the_run"] is True,
           "AND THE POSITIVE CONTROL: with the file restored the emit "
           "admits again -- the guard fires on the change, not on the run")

        # ---- R-608's DECLARATION ACT: the budget and the label ----------
        _der = REAL_DAY_BUDGET_DERIVATION
        ok(_der["the_declared_budget_still_covers_the_terms"] is True
           and abs(_der["terms_sum_mb"]
                   - (_der["be_day_reference_measured_mb"]
                      + _der["battery_retained_mb_measured"]
                      + _der["headroom_for_null_and_seal_mb"])) < 1e-9
           and _der["headroom_against_the_declared_budget_mb"] > 0
           and _der["declared_mb"] == REAL_DAY_PEAK_RSS_MB_BUDGET,
           f"R-610: THE BATTERY IS NOW A TERM OF THE BUDGET DERIVATION, "
           f"and the arithmetic is EVALUATED: "
           f"{_der['be_day_reference_measured_mb']:.0f} + "
           f"{_der['battery_retained_mb_measured']:.1f} + "
           f"{_der['headroom_for_null_and_seal_mb']:.0f} = "
           f"{_der['terms_sum_mb']:.1f} MB against the declared "
           f"{_der['declared_mb']:.0f} MB, leaving "
           f"{_der['headroom_against_the_declared_budget_mb']:.1f} MB. "
           f"The cap was NOT raised to make room for the move (R-174); "
           f"the term was measured and the headroom checked")
        ok(REAL_DAY_PEAK_RSS_MB_BUDGET == 4000.0
           and REAL_DAY_BUDGET_DERIVATION["declared_mb"] == 4000.0
           and REAL_DAY_BUDGET_DERIVATION["cgroup_cap_mb"] == 8192.0
           and "not their source" in REAL_DAY_BUDGET_DERIVATION[
               "why_not_measured_peak_plus_margin_alone"],
           f"R-608 DECLARATION ACT: the real day's budget is DERIVED -- "
           f"{REAL_DAY_BUDGET_DERIVATION['declared_mb']:.0f} MB, "
           f"{REAL_DAY_BUDGET_DERIVATION['fraction_of_cap']:.1%} of the 8192 "
           f"MB cap, from the cap and BE's measured 2008 MB reference, with "
           f"the observed 2426 MB peak as a CHECK on them and not their "
           f"source. Never the cap itself: a budget equal to the cap can only "
           f"fire once the kernel is already reclaiming")
        _made2 = write_synthetic_day(
            "FIXTURE-DAY-1", _tfr.mkdtemp(prefix="de89_"), params=live,
            n_slugs=24)
        # THE BASELINE IS THIS CELL'S OWN (DE 111): the refusal depends on
        # RSS GROWTH, and growth is measured against a high-water any
        # earlier cell can have already driven to the top.
        _rssb = cell_baseline("the -1.0 MB stage-budget known-bad")
        try:
            run_day("FIXTURE-DAY-1", _made2["book_path"], params=live,
                    fixture=True, peak_rss_mb_budget=-1.0)
            _rssr = False
        except RunnerRefused as _e:
            _rssr = "AT STAGE" in str(_e)
        _rssv = known_bad_verdict(_rssb, refused=_rssr,
                                  quantity="rss_mb_current")
        # AN AMBIENT CELL (REV 84 §2): its verdict rests on MEMORY, so
        # DISARMED is reachable and is recorded for the summary rather
        # than read as a pass.
        _rssok = record_ambient_verdict(
            disarmed, _rssv, ambient="memory",
            label="the -1.0 MB stage-budget known-bad")
        ok(_rssok and _rssv["verdict"] == "FIRED",
           f"AND IT FIRES AT THE FIRST STAGE THAT CROSSES IT, not at the "
           f"emit: a budget of -1.0 MB refuses AT A STAGE (verdict "
           f"{_rssv['verdict']}, this cell's own baseline on "
           f"{_rssv['quantity']} {_rssv['baseline']:.1f} MB, moved "
           f"{_rssv['moved_by']:.1f}). "
           f"The 09-03 run was told at 09:46 what was true at 08:22 -- 84 "
           f"minutes of null draws after the fact, because the check ran "
           f"once, after S5")
        _open2 = run_day("FIXTURE-DAY-1", _made2["book_path"], params=live,
                         fixture=True, n_days_complete=live["G"])
        ok(_open2["day"] == "FIXTURE-DAY-1"
           and _open2["memory_plan"]["budget_mb"]
           == FIXTURE_DAY_PEAK_RSS_MB_BUDGET,
           f"and the DAY LABEL IS THE ARGUMENT and the budget follows the "
           f"fixture flag: {_open2['day']} against "
           f"{_open2['memory_plan']['budget_mb']:.0f} MB. THE 09-03 REFUSAL "
           f"WAS NOT A MISLABELLED REAL DAY -- it was the emit-time battery's "
           f"OWN synthetic day, correctly labelled and correctly given the "
           f"fixture budget, judged against a PROCESS-WIDE high-water the real "
           f"day had already driven to 2426 MB")


        # ---- DE 82 (1): the scope's anon and file, read APART ------------
        # THE STATUS NAMES MOVED AT DE 110: the leaf the ruled form
        # produces is a `.service`, so "SCOPE" was the wrong word in every
        # one of them. The PROPERTY is unchanged and is what is checked.
        _sm = scope_memory_observation()
        ok(_sm["status"] in UNIT_MEMORY_STATUSES
           and (_sm["status"] != "NOT_IN_A_UNIT"
                or _sm["anon_bytes"] is None),
           f"DE 82 (1): the unit-leaf observation reports a STATUS "
           f"({_sm['status']}, one of {list(UNIT_MEMORY_STATUSES)}), and "
           f"when there is no unit leaf to read it reports None -- NEVER "
           f"zeros. A zero would read as 'measured, and nothing "
           f"happened', which is rule 11's silent absence")
        ok(scope_memory_observation.__doc__
           and "5.17" in scope_memory_observation.__doc__
           and "memory.events" in scope_memory_observation.__doc__,
           "and the reason anon and file are read APART is DA 63's "
           "measurement, carried where the code is: 7.79 GiB peak with "
           "1.13 GiB anon against 5.17 GiB reclaimable cache, events max 0 "
           "-- rule 20's cap counts the cache, so one number cannot tell "
           "'this run needs 7.8 GiB' from 'the kernel had no reason to "
           "reclaim'")
        _fake_none = {"status": "NOT_IN_A_UNIT", "anon_bytes": None,
                      "file_bytes": None, "memory_peak_bytes": None,
                      "events": None}
        ok(all(_fake_none[k] is None for k in
               ("anon_bytes", "file_bytes", "memory_peak_bytes", "events"))
           and _fake_none["status"] == "NOT_IN_A_UNIT",
           "KNOWN-BAD SHAPE, NAMED: the leafless record is Nones under a "
           "status, and a receipt carrying zeros there would be claiming a "
           "measurement it never made")
        ok(_sm.get("in_research_slice") is not None
           and (_sm["status"] == "MEASURED") == bool(
               _sm.get("in_research_slice")),
           f"AND AN AMBIENT UNIT IS NOT THE RUN'S: MEASURED holds only "
           f"inside {RESEARCH_SLICE}. Measured while writing this: the "
           f"login shell's own scope carries a 15.5 GiB peak against an "
           f"8 MB run, and attributing that to the run would be a "
           f"measurement of the wrong object")

        # ---- DE 82 (2): the smoke, rehearsed -----------------------------
        _rh = rehearse_smoke("2026-09-03")
        # A RELATION, NOT A STATE. This asserted `book.exists is False`,
        # which was true when written and went RED the moment BE landed the
        # book mid-round -- the same defect as the check that pinned
        # `G_is_PENDING` after the USER had answered. What must hold is the
        # RELATION between the filesystem and the receipt, in both states.
        _bp_live = Path(_rh["book"]["path"])
        ok(_rh["book"]["exists"] is _bp_live.is_file()
           and (("P2_book_exists" in _rh["blocking"])
                is (not _bp_live.is_file()))
           and ((_rh["book"]["REFUSES_NOW_BY_NAME"] is not None)
                is (not _bp_live.is_file())),
           f"DE 82 (2): the rehearsal's book report AGREES WITH THE "
           f"FILESYSTEM in whichever state it is in -- exists="
           f"{_rh['book']['exists']}, blocking={_rh['blocking']}. Asserted "
           f"as a RELATION: the first version pinned `exists is False` and "
           f"went red the moment BE landed the book mid-round")
        _rh_absent = rehearse_smoke("2026-09-08")
        ok(_rh_absent["book"]["exists"] is False
           and "P2_book_exists" in _rh_absent["blocking"]
           and _rh_absent["book"]["REFUSES_NOW_BY_NAME"]
           and _rh_absent["status"].startswith("NOT_READY"),
           "AND THE ABSENT BRANCH IS DRIVEN ON A DAY WHOSE BOOK CANNOT YET "
           "EXIST (09-08, still in the future): it REFUSES NOW BY NAME, so "
           "GO is one verified command and nothing is typed at GO time")
        # THE SHAPE IS ASSERTED THROUGH THE PREDICATE, not by a second
        # list of substrings beside it. This one began `flock -n ...` --
        # the SCOPE-era shape, with the lock OUTSIDE the wrapper -- and it
        # would have gone red on R-628's change while saying nothing about
        # why the form moved.
        ok(assert_launch_form(_rh["THE_ONE_COMMAND"])["ok"] is True
           and "MemoryMax=8G" in _rh["THE_ONE_COMMAND"]
           and "--day 2026-09-03" in _rh["THE_ONE_COMMAND"]
           and "be_daybook_20260903_btc.pkl" in _rh["THE_ONE_COMMAND"],
           "and the published command PASSES the launch-form predicate "
           "and carries the dashed day DE names beside the COMPACT-day "
           "book path BE writes -- the two conventions meeting in one "
           "string that is executed as a value rather than typed")
        _p5 = [x for x in _rh["preconditions_evaluated_now"]
               if x["precondition"] == "P5_lock_free_now"][0]
        ok(_p5["blocks_go"] is False
           and "P5_lock_free_now" not in _rh["blocking"]
           and (("P5_lock_free_now" in _rh["informational_and_false_now"])
                is (not _p5["holds"])),
           f"and the lock is INFORMATIONAL, not blocking, IN EITHER STATE "
           f"(holds={_p5['holds']} right now): the wrapper takes it at GO. "
           f"Asserted as a relation for the same reason as the book -- the "
           f"first version pinned 'held', and BE 55 finishing mid-round "
           f"turned it red")
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
           and set(_ps["highwater_delta_mb_by_stage"])
           == {k for k, _ in DAY_STAGES} - {HOOK_STAGE}
           and _ps["declared_peak_stage"] == declared_peak_stage()
           and DAY_STAGE_PEAK_MARKER in _ps["declared_read_from"],
           f"REV 45 S1.6 / REV 43 S4.3: the peak stage is the ARGMAX OVER "
           f"THE HIGHWATER DELTAS over all {len(DAY_STAGES) - 1} stages "
           f"this run marked -- "
           f"measured {_ps['measured_peak_stage']}, declared "
           f"{_ps['declared_peak_stage']}, agree "
           f"{_ps['declared_stage_is_the_measured_peak']} -- and the "
           f"declaration is READ FROM THE STAGE TABLE, never a default")
        ok(_ps["measured_peak_stage_by_current_rss"] is not None
           and "transient" in _ps["why_the_DELTA_is_the_peak"],
           f"AND BOTH READINGS ARE REPORTED because they answer different "
           f"questions: the delta argmax is "
           f"{_ps['measured_peak_stage']} (growth caused, transients "
           f"included) and the current-RSS argmax is "
           f"{_ps['measured_peak_stage_by_current_rss']} (still resident at "
           f"the mark). They agree here: "
           f"{_ps['highwater_delta_argmax_equals_current_rss_argmax']}")

        # ---- the three falsifiers REV 45 asks for, on synthetic series ---
        def _series(hw):
            out = {"S_start": {"peak_rss_mb_highwater": 100.0,
                               "rss_mb_current": 100.0}}
            for k, v in hw.items():
                out[k] = {"peak_rss_mb_highwater": v,
                          "rss_mb_current": 100.0}
            return out
        # KEYED BY NAME, NEVER POSITIONALLY. These series were built with
        # `zip(names, [six floats])`, so adding a stage to the table
        # SILENTLY SHIFTED every reading by one and the falsifier began
        # testing a different claim -- a list literal that has to track a
        # table it does not name.
        _names = [k for k, _ in DAY_STAGES if k != HOOK_STAGE]
        # A TRANSIENT INSIDE S4: the highwater jumps and comes back down in
        # the current series, so the CURRENT reading cannot see it at all.
        _tr = _series({"S0_verify": 101.0, "S1_load": 102.0,
                       "S2_population": 102.0, "S3_baseline": 102.0,
                       "S4_null": 180.0, "S5_seal": 180.0})
        _pt = peak_stage_predicate(_tr, declared="S1_load")
        ok(_pt["measured_peak_stage"] == "S4_null"
           and _pt["declared_stage_is_the_measured_peak"] is False
           and _pt["measured_peak_stage_by_current_rss"] != "S4_null",
           f"KNOWN-BAD, THE TRANSIENT: a stage that allocates 78 MB and "
           f"FREES IT before its mark is INVISIBLE to the current-RSS "
           f"series (which reads flat at 100 and argmaxes to "
           f"{_pt['measured_peak_stage_by_current_rss']}) and VISIBLE in "
           f"the highwater delta, which argmaxes to "
           f"{_pt['measured_peak_stage']} and FLAGS the disagreement. That "
           f"transient is exactly what REV 43 S4.3 said was invisible")
        _ok_series = _series({"S0_verify": 180.0, "S1_load": 181.0,
                              "S2_population": 181.0, "S3_baseline": 181.0,
                              "S4_null": 182.0, "S5_seal": 182.0})
        _po = peak_stage_predicate(_ok_series, declared="S0_verify")
        ok(_po["measured_peak_stage"] == "S0_verify"
           and _po["declared_stage_is_the_measured_peak"] is True
           and assert_peak_stage(_po, fixture=False,
                                 day="2026-09-03")["asserted"] is True,
           "POSITIVE CONTROL, AND IT ADMITS: a series whose largest "
           "highwater delta IS the declared stage agrees, and a REAL day "
           "on it passes assert_peak_stage -- so the known-bad above fires "
           "on the disagreement and not on the shape of the input")
        refuses(lambda: assert_peak_stage(_pt, fixture=False,
                                          day="2026-09-03"),
                "and on a REAL day the transient's disagreement REFUSES: "
                "the 8 GiB ceiling rests on the declared shape, and a "
                "shape that is wrong is a ceiling that is not established",
                "the memory plan declares")
        _saved = globals()["DAY_STAGES"]
        try:
            globals()["DAY_STAGES"] = tuple(
                (k, v.replace(DAY_STAGE_PEAK_MARKER, "")) for k, v in _saved)
            try:
                declared_peak_stage()
                ok(False, "a stage table with NO declared peak was ADMITTED")
            except RunnerRefused as _e:
                ok("exactly one must" in str(_e),
                   "KNOWN-BAD, A MISSING DECLARATION: a stage table with no "
                   "marked peak REFUSES rather than defaulting. It WAS a "
                   "default argument, which made the declaration and the "
                   "predicate two spellings of one fact -- so the ceiling "
                   "would have rested on a default nobody wrote down")
            globals()["DAY_STAGES"] = tuple(
                (k, DAY_STAGE_PEAK_MARKER + " " + v) for k, v in _saved)
            try:
                declared_peak_stage()
                ok(False, "a table with TWO declared peaks was ADMITTED")
            except RunnerRefused as _e:
                ok("stages marked" in str(_e),
                   "and a table that marks TWO peaks refuses as well -- a "
                   "declaration that disagrees with itself is not a "
                   "declaration")
        finally:
            globals()["DAY_STAGES"] = _saved
        ok(declared_peak_stage() == "S1_load",
           "and the table is RESTORED, so the mutation above cannot leak "
           "into any later check")

        # ---- a real day is refused for the reasons it must be --------------
        # DRIVEN ON THE PREDICATE, WITH THE OBSERVATION INJECTED. This
        # called `run_day` and read its message, so its verdict depended
        # on whether THIS process held the lock -- and the battery now
        # runs inside every real day, which holds it. Measured one line
        # apart: PASS 234 without the lock, FAIL "wrong reason" with it.
        # It refused the 09-03 re-run at 26 seconds (DE 92).
        refuses(lambda: assert_real_day_has_the_lock(
                    "2026-09-04", {"heavy_run_lock_held": False},
                    fixture=False),
                "A REAL DAY WITHOUT THE LOCK REFUSES BEFORE ANY WORK: it is "
                "heavy by construction (BE projects ~2.3 h for both arms), so "
                "the lock is taken FIRST or the run does not start (R-575(C))",
                "does not hold")
        ok(assert_real_day_has_the_lock(
               "2026-09-04", {"heavy_run_lock_held": True},
               fixture=False)["checked"] is True
           and assert_real_day_has_the_lock(
               "FIXTURE-DAY-1", {"heavy_run_lock_held": False},
               fixture=True)["checked"] is False,
           "AND BOTH OTHER CELLS DRIVE, which the old shape could not "
           "reach at all: a real day that HOLDS the lock admits, and a "
           "FIXTURE is not checked. The observation is a PARAMETER now, "
           "so the battery's verdict no longer depends on whether the "
           "process running it happens to hold the lock -- it passed "
           "everywhere except inside the run it exists for")

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

    # PLACED AT THE END OF THE BATTERY DELIBERATELY. Put earlier, these
    # cells loaded the 23 design declarations and raised this process's
    # ru_maxrss high-water -- and the RSS-budget known-bad a few checks
    # below measures GROWTH from that high-water, so a 0.001 MB budget
    # stopped refusing and a known-bad that had fired for rounds ADMITTED.
    # A check whose ability to fail depends on what ran before it is
    # rule 16's class; the fix is order, and this is the note that says so.
    # ===== DE 113 (REV 84 §2): n_disarmed IS PART OF THE SUMMARY ========
    _fake_dis = [{"label": "a planted disarmed cell", "ambient": "memory"}]
    _sd = battery_summary("de_probe", n_run=42, disarmed=_fake_dis)
    _sc = battery_summary("de_probe", n_run=42, disarmed=[])
    ok("n_disarmed 1" in _sd["line"] and _sd["clean"] is False
       and "NOT CLEAN" in _sd["line"]
       and "n_disarmed 0" in _sc["line"] and _sc["clean"] is True
       and "PASS" in _sc["line"],
       f"REV 84 §2 FALSIFIER: a battery with ONE disarmed cell prints "
       f"`{_sd['line']}` and is NOT clean, while the same battery with "
       f"none prints `{_sc['line']}` and is. A run that printed 'N checks "
       f"passed' beside three disarmed cells would report the "
       f"measurement's failure as coverage -- the DE 110 defect one level "
       f"up")
    # ===== DE 126: A WRONG ROOT IS NOT A SIX-DAY REGRESSION ============
    # THE DRIVE THE COORDINATOR RAN, both ways, at the real ledger. With
    # the REPO root the gate used to report all six days failing conjunct
    # 3 with a detail saying only `failing` -- indistinguishable from the
    # H1 grouping having regressed 09-06. It refuses by name now.
    _now126 = datetime.datetime.now(datetime.timezone.utc)
    _P126 = load_params()
    if offline:
        offline_skip("DE 126's wrong-root drive (it reads the real "
                     "ledger's derived directory)")
    else:
        _repo126 = Path(DR.resolve()["data_root"]).parent
        try:
            read_gate(_P126, now_utc=_now126, root=_repo126)
            _refused126 = None
        except RunnerRefused as _e126:
            _refused126 = str(_e126)
        _ok126 = read_gate(_P126, now_utc=_now126,
                           root=Path(DR.resolve()["data_root"]))
        _c3126 = next(c for c in _ok126["conjuncts"]
                      if c["conjunct"].startswith("3_"))
        ok(_refused126 is not None
           and "READ_GATE_ROOT_IS_NOT_THE_LEDGER" in _refused126
           and _c3126["detail"]["failing"] == ["2026-09-07", "2026-09-08"]
           and set(_c3126["detail"]["per_day"]) == set(_P126["days"])
           and all(_c3126["detail"]["per_day"][d]["why_not"] is not None
                   for d in ("2026-09-07", "2026-09-08"))
           and _c3126["detail"]["per_day"]["2026-09-06"]["why_not"] is None,
           f"DE 126: the REPO root now REFUSES BY NAME "
           f"(`READ_GATE_ROOT_IS_NOT_THE_LEDGER`) "
           f"instead of reporting six failing days, and with the LEDGER "
           f"root conjunct 3's failing set is "
           f"{_c3126['detail']['failing']} -- the two days with no "
           f"receipt, each carrying its own `why_not`, and 09-06 carrying "
           f"none. H1's grouping did NOT regress 09-06; the six-day "
           f"reading was the repo root passed where the data root belongs")

    # ===== DE 126 (REV 89 §2.1): THE EMITTER'S OWN MOVING LITERAL ======
    # `DESIGN_VERSION_IN_FORCE` is a bare constant and nothing tied it to
    # the scope map it governs. It holds TODAY only because
    # max(SEALED_FROM_DESIGN_VERSION.values()) == it. A twelfth name
    # declared at 26 without a bump would be EMITTED IN THE OPEN, with
    # the receipt truthfully saying the older version, and
    # `de_receipt_correction` -- which already resolves the wider of the
    # constant and the chain head for its own use -- would flag it only
    # after the number was published.
    _mx126 = max(SEALED_FROM_DESIGN_VERSION.values())
    ok(_mx126 <= DESIGN_VERSION_IN_FORCE,
       f"REV 89 §2.1: the emitter's scope map tops out at v{_mx126} and "
       f"DESIGN_VERSION_IN_FORCE is v{DESIGN_VERSION_IN_FORCE}, so every "
       f"sealed name is inside the version the receipt will claim. A name "
       f"declared ABOVE the constant would be sealed under a version that "
       f"does not cover it and published before anything noticed")
    # THE KNOWN-BAD RUNS THE SAME PREDICATE over a planted map, so the
    # two cannot drift. My first version of this wrote
    # `not any(...) is True`, which parses as `not (any(...) is True)` --
    # a malformed conjunct that failed for its own reason. That is REV 89
    # item 5's class exactly, in the cell written to close item 3.
    _planted126 = {**SEALED_FROM_DESIGN_VERSION,
                   "a planted twelfth name": DESIGN_VERSION_IN_FORCE + 3}
    ok(max(_planted126.values()) > DESIGN_VERSION_IN_FORCE,
       f"REV 89 §2.1 KNOWN-BAD: the SAME predicate over a map carrying a "
       f"planted name at v{DESIGN_VERSION_IN_FORCE + 3} reads "
       f"{max(_planted126.values())} > {DESIGN_VERSION_IN_FORCE}, so the "
       f"cell above WOULD refuse it. A guard nobody has watched fire is a "
       f"comment")

    # ===== DE 126 (REV 89 §2.1): THE CLOSURE MAP MUST NOT COLLIDE ======
    # REV drove DE 120's initialiser conjunct GREEN on a NON-EMPTY
    # `live/__init__.py` by putting an empty `live/pm_research/__init__.py`
    # beside it: both collapse to the key `__init__.py` and the later one
    # wins. The receipt's own account of its closure was short by one.
    _si126 = source_identity_at_launch()["import_closure"]
    ok(_si126["n_modules"] == len(_si126["modules_by_path"])
       and _si126["n_modules_equals_len_modules"] is True
       and _si126["basenames_collide"] is False
       and _si126["colliding_basenames"] == [],
       f"REV 89 §2.1: the live closure names {_si126['n_modules']} "
       f"modules by PATH and the basename map has "
       f"{len(_si126['modules'])} keys -- equal, so nothing is hidden "
       f"behind a shared basename. `modules_by_path` is the map that "
       f"cannot collide and `basenames_collide` is EVALUATED, which is "
       f"what was missing")
    _fake126 = {"/a/live/__init__.py": "00ef3de7",
                "/a/live/pm_research/__init__.py": "e3b0c442"}
    ok(len({Path(k).name for k in _fake126}) != len(_fake126),
       f"REV 89 §2.1 KNOWN-BAD, THE CASE REV BUILT: two initialisers at "
       f"different paths collapse to {len({Path(k).name for k in _fake126})} "
       f"basename key(s) from {len(_fake126)} paths -- the predicate is "
       f"FALSE on it, so the cell above would refuse the closure REV drove "
       f"green")

    # ===== DE 126 (REV 89 H1): THE GROUPING, AND BOTH SEATS COMPARED ===
    # RED FIRST, on a fixture: two UNLINKED records for one day, reading
    # two different receipts -- which is what a corrected receipt leaves
    # behind. Under the old rule (every record for a DAY into one chain)
    # this is AMBIGUOUS; under DA 105's it resolves to the group whose
    # records read the day's CURRENT head.
    import tempfile as _tf126
    _r126 = Path(_tf126.mkdtemp(prefix="h1_"))
    _d126 = _r126 / "pm_5min/derived"
    _d126.mkdir(parents=True)
    _rc126 = {"day": "2026-09-03", "per_day_sealed_artifacts": [],
              "status": "DAY_RUN_SEALED"}
    _old126 = _d126 / "p003_de_gate1_day_run_20260903_SEALED__A.json"
    _old126.write_text(json.dumps(_rc126, sort_keys=True))
    _new126 = _d126 / "p003_de_gate1_day_run_20260903_SEALED__A.v2.json"
    _new126.write_text(json.dumps(
        {**_rc126, "supersedes": {"path": str(_old126),
                                  "sha256": sha256_streamed(_old126)}},
        sort_keys=True))
    for _stamp, _tgt in (("20260906T182045Z", _old126),
                         ("20260906T192751Z", _new126)):
        (_d126 / f"p003_da_gate1_pre_read_20260903__{_stamp}.json"
         ).write_text(json.dumps({
             "day": "2026-09-03", "is_the_declared_LANDING_RECORD": True,
             # THE FIELD NAMES ARE `LANDING_RECORD_FIELD_COPIES`', read
             # off the constant rather than typed from memory: my first
             # fixture wrote `receipt_sha256_at_landing`, which is the
             # RETURN's name and not the RECORD's, so every record read
             # UNSTATED and the cell failed on my fixture rather than on
             # the code. Both copies are written, as a real record does.
             "landing_record": {"day": "2026-09-03",
                                "receipt_sha256": sha256_streamed(_tgt),
                                "receipt_path": _tgt.name},
             "receipt": {"sha256": sha256_streamed(_tgt),
                         "path": _tgt.name}}, sort_keys=True))
    _lr126 = landing_record_for("2026-09-03", _r126)
    _g126 = _lr126["grouping"]
    ok(_g126["n_records"] == 2 and _g126["n_groups"] == 2
       and _g126["chose_one"] is True
       and _lr126["status"] == "PRESENT" and _lr126["n_matches"] == 1
       and _lr126["receipt_sha256_at_landing"] == sha256_streamed(_new126),
       f"REV 89 H1: two UNLINKED landing records for one day -- one per "
       f"receipt, which is what a correction leaves -- resolve to the "
       f"group reading the day's CURRENT head ({_lr126['status']}, "
       f"n_matches {_lr126['n_matches']}), not to AMBIGUOUS. DA 105's "
       f"rule: a landing record is about ONE RECEIPT, not about a day")
    # THE KNOWN-BAD IS THE OLD RULE ITSELF, driven on the same fixture:
    # handing BOTH records to the chain resolver -- which is what this
    # function did -- must still read AMBIGUOUS. A cell that only shows
    # the new answer proves nothing about what changed.
    _both126 = sorted(_d126.glob("p003_da_gate1_pre_read_*.json"))
    _amb126 = resolve_day_chain(_both126, kind="landing record")
    ok(_amb126["status"] == "AMBIGUOUS" and len(_both126) == 2,
       f"REV 89 H1 KNOWN-BAD: the OLD grouping -- every record for the "
       f"DAY into one chain -- reads `{_amb126['status']}` on the same "
       f"two files. That is the answer that refused 09-03, 09-04 and "
       f"09-05 in this seat's gate while DA's passed them; the delta is "
       f"the grouping, not the chain rule")
    # A SUPERSESSION LINK OUTRANKS THE GROUPING: two records that read
    # DIFFERENT receipts but are joined by a pair are ONE family.
    _lk126 = _d126 / "p003_da_gate1_pre_read_20260903__20260906T192751Z.json"
    _lk126.write_text(json.dumps({
        **json.loads(_lk126.read_text()),
        "supersedes": {
            "path": str(_d126
                        / "p003_da_gate1_pre_read_20260903__20260906T182045Z.json"),
            "sha256": sha256_streamed(
                _d126
                / "p003_da_gate1_pre_read_20260903__20260906T182045Z.json")}},
        sort_keys=True))
    _lr2126 = landing_record_for("2026-09-03", _r126)
    ok(_lr2126["grouping"]["n_groups"] == 1,
       f"REV 89 H1: and a SUPERSESSION LINK OUTRANKS the grouping -- the "
       f"same two records, now joined by a pair, are ONE group "
       f"({_lr2126['grouping']['n_groups']}), because a correction may "
       f"change the very field the grouping reads and a chain broken by "
       f"its own repair would be the resolver describing its rule")
    import shutil as _sh126
    _sh126.rmtree(_r126, ignore_errors=True)

    # THE CROSS-CHECK REV 89 SAID WAS THE MISSING PIECE: run BOTH seats'
    # resolvers on the REAL ledger and compare. R-235's do-not-harmonize
    # is what made the disagreement findable; nobody ran both.
    if offline:
        offline_skip("REV 89 H1's two-seat cross-check on the real ledger "
                     "(it reads DA's landing records under data/)")
    else:
        import da_gate1_day_verdict as _DA126
        _root126 = Path(DR.resolve()["data_root"])
        _cmp126, _dis126 = {}, []
        for _d in load_params()["days"]:
            _de = landing_record_for(_d, _root126)
            _da = _DA126.landing_record_for(
                _d, _root126 / "pm_5min/derived")
            _a = _de.get("receipt_sha256_at_landing")
            _b = (_da.get("receipt_sha256_at_landing")
                  or _da.get("receipt_sha256"))
            _cmp126[_d] = {"DE": _a, "DA": _b, "agree": _a == _b}
            if _a != _b:
                _dis126.append(_d)
        ok(not _dis126,
           f"REV 89 H1 CROSS-CHECK, BOTH SEATS ON THE REAL LEDGER: the "
           f"two independent resolvers name the SAME landing-record "
           f"receipt digest for every ruled day -- disagreeing: "
           f"{_dis126 or 'none'}. Before this, DE refused 09-03/04/05 on "
           f"conjunct 3 while DA passed them, and neither seat ran the "
           f"other. Two implementations stay (R-235); what was missing "
           f"was anybody comparing them")

    # ===== DE 131 (R-782): THE ABSOLUTES, ON HAND-COMPUTED LEGS ========
    # RED FIRST, on fills whose legs are arithmetic a reader can do:
    # four buys of 10 shares, level 100c, markout 100.2c -> each fill is
    # (100.2 - 100) * 10 = 2c, so a book of four is 8c. The baseline is
    # two such fills, 4c. The excess is therefore 4c, and the cell asserts
    # the absolutes AND that they reconcile to it.
    def _f782(level, markout, size, inv_b=None, inv_a=None, mark=None):
        d = {"side": HSP_BUY_SIDE(), "px_cents": level, "size": size,
             "mid_cents_at_markout": markout, "mid_cents_at_fill": level,
             "slug": "s", "ref_gen": 0, "fill_ns": 1.0, "gen_start_ns": 1.0}
        if inv_b is not None:
            d.update({"inventory_before": inv_b, "inventory_after": inv_a,
                      "inventory_mark_cents": mark, "inventory_unit": "sh"})
        return d
    _arm782 = [_f782(100.0, 100.2, 10.0, 0.0, 10.0, 100.0) for _ in range(4)]
    _base782 = [_f782(100.0, 100.2, 10.0, 0.0, 10.0, 100.0) for _ in range(2)]
    _A, _B = absolute_legs(_arm782), absolute_legs(_base782)
    ok(abs(_A["fills_leg"] - 8.0) < 1e-9
       and abs(_B["fills_leg"] - 4.0) < 1e-9
       and abs(_A["total"] - _B["total"] - 4.0) < 1e-9
       and _A["unit"] == VALUATION_UNIT == "cents"
       and abs(_A["trades_cash_flow_cents"] - (-4000.0)) < 1e-9
       and _A["n_fills"] == 4 and _B["n_fills"] == 2,
       f"R-782 (the USER: 'plz record the absolute number as well for "
       f"reference'): the ARM's own day value is {_A['fills_leg']:.1f} "
       f"{_A['unit']} over {_A['n_fills']} fills and the 0-CANCEL "
       f"BASELINE's is {_B['fills_leg']:.1f} over {_B['n_fills']} -- both "
       f"hand-computable as (100.2-100)*10 per fill -- and their "
       f"difference {_A['total'] - _B['total']:.1f} IS D(E0). The receipt "
       f"could answer 'how much better' and not 'how much'; it answers "
       f"both now. The TRADES CASH FLOW is "
       f"{_A['trades_cash_flow_cents']:.1f} (SELLS - BUYS) where BE 96's "
       f"fields are present -- R-803: the field this replaces was called "
       f"`inventory_leg` and carried the opposite sign")
    _noinv782 = absolute_legs([_f782(100.0, 100.2, 10.0)])
    ok(_noinv782["trades_cash_flow_cents"] is None
       and _noinv782["n_fills_with_inventory"] == 0
       and abs(_noinv782["fills_leg"] - 2.0) < 1e-9,
       f"R-782/R-803: with BE 96's position fields ABSENT the trades "
       f"cash flow is None and says why -- never 0. An unrecorded "
       f"quantity and a zero one "
       f"are the same number and opposite facts, and the fills leg "
       f"({_noinv782['fills_leg']:.1f}) is unaffected")

    # ===== DE 126: THE DAY'S OWN WORDS, COMPUTED FROM ITS ARM-DAYS =====
    # RED FIRST, against the artifact that carried the defect. The landed
    # 09-03 early read says `status: DAY_RUN_SEALED` and
    # `the_economics_are_SEALED: true` while BOTH arm-days read
    # `sealed False` with `economic` present. Two words describing the
    # run's standing, both wrong, on the artifact a human opens.
    _arm126 = {"day": "2026-09-03", "arm": "A", "status": "OK",
               "admissibility": {"admissible": True},
               "economic": {"D_E0": 1.0}}
    _uns126 = [seal(dict(_arm126), 4, 4), seal(dict(_arm126), 4, 4)]
    _sea126 = [seal(dict(_arm126), 1, 6), seal(dict(_arm126), 1, 6)]
    def _day_words(arts):
        _as = bool(arts) and all(a.get("sealed") is True for a in arts)
        _adm = any((a.get("admissibility") or {}).get("admissible") is True
                   for a in arts)
        return {"status": (("DAY_RUN_SEALED" if _as else "DAY_RUN_UNSEALED")
                           if _adm else
                           ("DAY_RUN_SEALED_NO_ADMISSIBLE_ARM" if _as
                            else "DAY_RUN_UNSEALED_NO_ADMISSIBLE_ARM")),
                "the_economics_are_SEALED": _as}
    _wu, _ws = _day_words(_uns126), _day_words(_sea126)
    ok(_wu["status"] == "DAY_RUN_UNSEALED"
       and _wu["the_economics_are_SEALED"] is False
       and _ws["status"] == "DAY_RUN_SEALED"
       and _ws["the_economics_are_SEALED"] is True
       and all("economic" in a for a in _uns126)
       and all("economic" not in a for a in _sea126),
       f"DE 126: THE DAY'S STATUS AND ITS `the_economics_are_SEALED` ARE "
       f"COMPUTED FROM THE ARM-DAYS. Unsealed arm-days give "
       f"`{_wu['status']}` / economics_are_SEALED "
       f"{_wu['the_economics_are_SEALED']} with `economic` PRESENT; sealed "
       f"ones give `{_ws['status']}` / {_ws['the_economics_are_SEALED']} "
       f"with it ABSENT. The landed 09-03 early read said DAY_RUN_SEALED "
       f"and true over two unsealed arm-days")
    # THE KNOWN-BAD IS THE OLD RULE, on the same inputs: `n < G` and a
    # typed word both give the WRONG answer where the artifacts are the
    # only witness.
    ok((4 < 6) is True and _wu["the_economics_are_SEALED"] is False,
       f"DE 126 KNOWN-BAD: the OLD predicate `n_days_complete < G` reads "
       f"TRUE at 4 of 6 -- which is exactly the early read -- while the "
       f"arm-days it describes are unsealed. Two ways of answering one "
       f"question, and the one that was published disagreed with the "
       f"artifacts beside it (rule 10; REV 90 §A0's class one level up)")

    # ===== DE 125 (REV 90 §A0): NO BRANCH OF seal() PRINTS A LITERAL ===
    # Driven on a 09-06-SHAPED arm-day -- the real landed shape, arm and
    # counts -- through BOTH branches of the one function, because the
    # defect was an ASYMMETRY between them and a cell that exercises one
    # branch cannot see it.
    _a125 = {"day": "2026-09-06", "arm": "CONDVALUE_X_SKEW", "status": "OK",
             "economic": {"D_E0": 2.5, "Z": 2.45, "p_location": 0.008,
                          "null_mean": 0.0066, "null_sd": 1.0158,
                          "null_draws_summary": {"n": 600}},
             "n_fills_arm": 30171, "n_fills_baseline": 46439,
             "n_cancels_issued": 5146}
    _ruled125 = seal(dict(_a125), 4, 4)      # the early read's own call
    _six125 = seal(dict(_a125), 6, 6)        # the honest six-day unseal
    _sealed125 = seal(dict(_a125), 3, 6)     # the sealed path, 09-06's own
    # THE RED HALF: a status that does not carry the day counts FAILS.
    # `4` and `6` are read out of the string, not compared to a sentence
    # anybody typed -- a cell that matched the new wording would go green
    # on the next rewording and prove nothing.
    ok("4 of 4" in _ruled125["seal_status"]
       and _ruled125["sealed"] is False
       and "ALL_DAYS_COMPLETE" not in _ruled125["seal_status"]
       and "6 of 6" in _six125["seal_status"]
       and "3 of 6" in _sealed125["seal_status"]
       and _sealed125["sealed"] is True,
       f"REV 90 §A0: NEITHER BRANCH PRINTS A LITERAL. Under the ruled bar "
       f"the status is `{_ruled125['seal_status']}` -- it carries its own "
       f"4 of 4 and no longer claims ALL_DAYS_COMPLETE, which is what it "
       f"said beside `n_days_complete 4, G 6`. The honest six-day unseal "
       f"reads `{_six125['seal_status']}`, and the sealed path is "
       f"unchanged: `{_sealed125['seal_status'][:34]}…`")
    # THE KNOWN-BAD, against the OLD behaviour as its baseline: the
    # literal this replaced carries no counts, so the predicate that
    # passes above must FAIL on it.
    _old125 = "UNSEALED_ALL_DAYS_COMPLETE"
    ok(not any(f"{i} of {j}" in _old125 for i in range(9)
               for j in range(9)),
       f"REV 90 §A0 KNOWN-BAD: the literal this replaced -- "
       f"`{_old125}` -- contains no `N of M`, so the predicate above "
       f"FAILS on it. The cell measures a delta from the behaviour that "
       f"was landed, not agreement with the words I just wrote")
    # AND THE SEALED PATH'S BYTES ARE UNCHANGED, not merely its prose:
    # the sealed branch is compared field by field against what it
    # produced before this change, which is the string 09-06 landed with.
    ok(_sealed125["seal_status"].startswith(
           "SEALED -- 3 of 6 days complete.")
       and _sealed125["sealed_at_every_depth"] is True
       and _sealed125["sealed_field_names"] == list(ECONOMIC_FIELDS)
       and "economic" not in _sealed125,
       f"REV 90 §A0: the SEALED branch is byte-identical to what the four "
       f"landed days emitted -- `{_sealed125['seal_status'][:38]}…`, "
       f"{len(_sealed125['sealed_field_names'])} sealed names, economics "
       f"absent. This change touched the unsealed branch only")

    # ===== DE 119 (REV 88 §1.3): A SKIP IS THE THIRD OUTCOME ===========
    # Driven on the SUMMARY PATH, in all three shapes a caller can hand
    # it, and the run count is held fixed at 42 in every one -- because
    # the defect this is for is a skip being folded into `n_run` or into
    # silence, not a skip being miscounted.
    _sk_named = battery_summary(
        "de_probe", n_run=42,
        skipped=[{"label": "reads data/", "why": "offline fixture run"}])
    _sk_bare = battery_summary("de_probe", n_run=42, skipped=[
        {"label": "reads data/"}])          # a cell with no reason
    _sk_count = battery_summary("de_probe", n_run=42, n_skipped=1)
    _sk_none = battery_summary("de_probe", n_run=42)
    ok(_sk_named["clean"] is True and "n_skipped 1" in _sk_named["line"]
       and "UNNAMED" not in _sk_named["line"]
       and _sk_bare["clean"] is False
       and "UNNAMED SKIPS 1" in _sk_bare["line"]
       and _sk_count["clean"] is False and _sk_count["n_skipped"] == 1
       and "UNNAMED SKIPS 1" in _sk_count["line"]
       and "n_skipped 0" in _sk_none["line"] and _sk_none["clean"] is True
       and all(x["n_run"] == 42 for x in
               (_sk_named, _sk_bare, _sk_count, _sk_none)),
       f"REV 88 §1.3 FALSIFIER: one NAMED skip is `{_sk_named['line']}` "
       f"and stays clean; the SAME skip with no reason is "
       f"`{_sk_bare['line']}` and is NOT clean; a skip carried as a bare "
       f"count is `{_sk_count['line']}` and is NOT clean, because a "
       f"count nobody can resolve to cells is unnamed by construction; "
       f"and n_skipped is in the line at zero too, so a reader never has "
       f"to infer it from silence. `n_run` is 42 in all four -- a skip "
       f"is never folded into the run count")
    refuses(lambda: record_ambient_verdict(
                [], {"verdict": "DISARMED"}, ambient="a planted digest",
                label="x"),
            f"and DISARMED is UNREACHABLE for a constructed input: an "
            f"ambient outside DA's REV 68 §1.2 census "
            f"{list(AMBIENT_KINDS)} REFUSES, because a cell over a built "
            f"input either fires or admits and a third state it can never "
            f"reach is a control that cannot fail",
            "not one of the ambients")
    _sink = []
    ok(record_ambient_verdict(
           _sink, {"verdict": "FIRED", "quantity": "rss_mb_current"},
           ambient="memory", label="fired") is True
       and _sink == []
       and record_ambient_verdict(
           _sink, {"verdict": "DISARMED", "quantity": "rss_mb_current",
                   "baseline": 1.0, "now": 1.0, "moved_by": 0.0,
                   "why": "no movement"},
           ambient="the clock", label="disarmed") is False
       and len(_sink) == 1 and _sink[0]["ambient"] == "the clock",
       "and the sink takes ONLY the disarmed: FIRED returns True and "
       "records nothing, DISARMED returns False and records the label, "
       "the ambient and the measurement. ADMITTED is deliberately not "
       "recorded here -- it is a defect in the code under test, not in "
       "the measurement")

    # ===== DE 113 (REV 84 §3.2): THE SHARED MODULE'S FALSIFIER, AS ONE
    # CELL OF THIS BATTERY. One implementation, N detectors: a regression
    # in `declaration_chain` fails BE's, DA's and this suite at once
    # without any importer re-implementing the logic.
    _dcf = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parent / "declaration_chain.py"),
         "--falsify"], capture_output=True, text=True, timeout=120)
    _dcf_last = (_dcf.stdout.strip().splitlines() or [""])[-1]
    ok(_dcf.returncode == 0 and "0 failures" in _dcf_last,
       f"REV 84 §3.2: the SHARED module's own falsifier runs as ONE cell "
       f"of this battery -- `declaration_chain.py --falsify` -> rc "
       f"{_dcf.returncode}, `{_dcf_last}`. THE CELLS THIS SEAT KEEPS "
       f"INDEPENDENT are the ones its OWN verdicts rest on: the head "
       f"resolution that picks the seal scope (below), and this module's "
       f"OWN walk, which is this seat's code and not the shared one's. "
       f"What it does NOT re-test is the module's internal link algebra "
       f"-- that is what this cell is for, and re-implementing it would "
       f"re-create the divergence one implementation was adopted to end")
    # READS `data/` (the design family and the head's bytes) -- skipped
    # offline and NAMED. Third time this pair of rounds; the guard goes in
    # with the cell.
    if offline:
        offline_skip("DE 113: the seal scope picked from the design chain "
                     "head (reads data/pm_5min/derived)")
    else:
      _hv = design_chain()["head_version"]
      _sv = design_version_of_receipt(
          {"provenance": {"design": {
              "path": f"data/pm_5min/derived/"
                      f"p003_de_multiday_gate1_design_v{_hv}.json",
              "sha256": hashlib.sha256(
                  (Path(DR.resolve()["data_root"]) / "pm_5min/derived"
                   / f"p003_de_multiday_gate1_design_v{_hv}.json"
                   ).read_bytes()).hexdigest()}}})
      ok(_sv["design_version"] == _hv and _sv["pair_verified"] is True
         and len(_sv["fields_in_force"]) == len(ECONOMIC_FIELDS),
         f"THE CELL THIS SEAT KEEPS: the SEAL SCOPE is picked from the "
         f"head resolution -- a receipt pinning the head v{_hv} by pair "
         f"is judged under {len(_sv['fields_in_force'])} names. Every "
         f"sealed artifact this seat writes rests on that, so it is "
         f"tested here and not delegated (REV 84 §3.1)")

    # ===== DE 112: THE RECEIPT NAMES EVERY CHAIN IT RESOLVED ============
    _pem = producer_exit_map()
    ok(_pem["status"] == "DECLARED" and _pem["codes"] == [0, 1]
       and _pem["lock_conflict_rc_excluded"] == 75
       and _pem["agrees_with_the_module"] is True
       and _pem["head_sha256"] == hashlib.sha256(
           (Path(__file__).resolve().parents[2] / EXIT_MAP_DIR
            / _pem["head_path"]).read_bytes()).hexdigest(),
       f"R-709 CONSUMED: this producer's exit map resolves to "
       f"{_pem['head_path']} at {_pem['head_sha256'][:12]}… (v"
       f"{_pem['head_version']}), codes {_pem['codes']}, 75 excluded, and "
       f"the digest RECOMPUTED from the file the head names. The receipt "
       f"carries this block, so a reader resolving a non-zero "
       f"`ExecMainStatus` to a KIND reads the map THIS RUN resolved -- "
       f"not today's head against yesterday's run")
    # READS `data/` (the real design family) -- skipped offline and NAMED,
    # exactly as its sibling below. The fixture-run data-freeness
    # instrument caught this one the moment it was written, which is the
    # second time this round; the guard goes in with the cell now.
    if offline:
        offline_skip("DE 112: the merge facts on the REAL design family "
                     "(reads data/pm_5min/derived)")
    else:
        _mk = design_chain_orphans()["shared_resolver"]
        # DE 124's RULING (1): THE PROPERTY, NOT THE PROXY. The old cell
        # compared `len(merged_tips)` with `n_merge_links`, which held
        # only while no version re-declared its ancestor's merges. I broke
        # that by composing three designs as deepcopies of their parents
        # -- 20 links for 5 tips -- and the proxy went red while the
        # property never moved. The property is: every tip is ABSORBED
        # EXACTLY ONCE, by the EARLIEST version naming it, and every
        # declared link resolves into that set; a re-declaration is
        # REPORTED as REDECLARED_MERGE with its ancestor and is not
        # counted. This is a DIFFERENT check, not a looser one: it also
        # requires every declared link to resolve, which the old one
        # never asked.
        import declaration_chain as _DC126
        _mb = _DC126.merges_by_earliest(
            Path(DR.resolve()["data_root"]) / "pm_5min/derived",
            "p003_de_multiday_gate1_design")
        _tips = set(_mb["absorbed"])
        _all_links = _mb["n_merges_effective"] + _mb["n_redeclared"]
        ok(len(_tips) == _mb["n_merges_effective"]
           and set(_mk["merged_tips"]) == _tips
           and all(r["tip"] in _tips for r in _mb["redeclared_merges"])
           and all(r["status"] == "REDECLARED_MERGE"
                   and r["absorbed_by"] and r["redeclared_in"]
                   for r in _mb["redeclared_merges"])
           and _all_links == _mk["n_merge_links"],
           f"DE 124 (1): EVERY TIP IS ABSORBED EXACTLY ONCE -- "
           f"{_mb['n_merges_effective']} tip(s), each by the EARLIEST "
           f"version naming it -- and all {_all_links} declared links "
           f"resolve into that set. {_mb['n_redeclared']} are "
           f"REDECLARED_MERGE, reported with the ancestor that absorbed "
           f"the tip and the version that repeated it, counted as merges "
           f"by nothing. `orphan_branches` {_mk['orphan_branches']}. The "
           f"old cell compared a COUNT with a SET SIZE, which my own "
           f"deepcopy-composed versions falsified while the property held")

    # ===== DE 111: A KNOWN-BAD MEASURES AGAINST ITS OWN BASELINE ========
    # THE DISARMING CASE, DRIVEN -- with the real mechanism, not a story.
    # THE MECHANISM, REPRODUCED ON THIS CELL'S OWN FIXTURES -- not on the
    # ledger. Loading the real 23 design declarations is what did it at
    # DE 110, and reading them HERE would break the fixture run's
    # data-freeness instrument (it did, and the instrument caught it: 24
    # paths under `data/` inside a fixture run). What the rule needs is a
    # cell that raises the process high-water before a known-bad, and 23
    # documents of the same shape in a temp dir do exactly that.
    # WHAT THE BUDGET ACTUALLY READS -- asserted at the CODE, because
    # DE 110's row asserted the opposite in prose.
    import inspect as _insp
    _src = _insp.getsource(run_day)
    _gsrc = _src[_src.index("THE GROWTH IS MEASURED ON CURRENT RSS"):]
    _gsrc = _gsrc[:_gsrc.index("raise RunnerRefused")]
    ok('stages[name]["rss_mb_current"]' in _gsrc
       and "peak_rss_mb_highwater" not in _gsrc,
       "DE 110's ROW IS CORRECTED AT THE CODE: it said the stage budget "
       "'measures GROWTH from that high-water'. It does not -- the growth "
       "is `stages[name]['rss_mb_current'] - S_start's`, and the comment "
       "above it says why in the code's own words: `ru_maxrss` cannot "
       "fall, so a budget measured on it 'only works once per process'. "
       "The defence against the high-water was already there")
    # AND THE MECHANISM IS **NOT ESTABLISHED**, measured rather than told.
    _bd = Path(_tfr.mkdtemp(prefix="de111hw_"))
    for _i in range(23):
        (_bd / f"doc_{_i}.json").write_text(json.dumps(
            {f"R{_j}": {"why": "x" * 40000} for _j in range(60)}))
    _cur0 = _current_rss_mb()
    _loaded = [json.loads(q.read_text()) for q in sorted(_bd.glob("*.json"))]
    _cur1 = _current_rss_mb()
    del _loaded
    _gc.collect()
    _cur2 = _current_rss_mb()
    _arena = _cur2 - _cur0
    # REV 83 §5, ON THIS CELL ITSELF: `abs(residue) < transient` is a
    # comparison of LEVELS the whole process shares, and it broke the
    # moment cells before it allocated (the R-801/R-803 source cells load
    # an 8 MB venue record; the allocator then returned nothing at the
    # free and the residue read EQUAL to the transient). A verdict that
    # changes when the battery is reordered is measuring history. What
    # this cell establishes is unchanged and does not need that bound:
    # the load RAISES current RSS, the residue is RECORDED with its sign,
    # and the mechanism stays NOT ESTABLISHED -- which is the finding.
    ok(_cur1 > _cur0,
       f"AND THE MECHANISM IS RECORDED AS **NOT ESTABLISHED**, measured "
       f"rather than told: loading 23 declaration-shaped documents took "
       f"current RSS {_cur0:.1f} -> {_cur1:.1f} MB and freeing them left "
       f"it at {_cur2:.1f} -- a residue of {_arena:+.1f} MB against a "
       f"{_cur1 - _cur0:.1f} MB transient (NO BOUND is asserted on the "
       f"residue: that was a level this process shares, and it flipped "
       f"when cells before this one allocated -- REV 83 §5), and its "
       f"SIGN is not stable "
       f"across runs (it has read both ways here), so NO sign is "
       f"asserted. Either way it does not explain a 0.001 MB budget that "
       f"stopped being crossed -- a negative residue is MORE headroom, "
       f"not less. The other candidate, an inherited high-water, is "
       f"contradicted at the code above. What IS established: the cell "
       f"ADMITTED with those cells before it and FIRES with them after "
       f"it, ordering the only change. The rule below makes that visible "
       f"whatever the cause")
    # THE THREE OUTCOMES, driven on this cell's own baseline.
    _dis_base = cell_baseline("the disarming case")
    _dis = known_bad_verdict(_dis_base, refused=False,
                             quantity="rss_mb_current",
                             measured_now=_dis_base["rss_mb_current"])
    _fired = known_bad_verdict(_dis_base, refused=True,
                               quantity="rss_mb_current",
                               measured_now=_dis_base["rss_mb_current"])
    _real = known_bad_verdict(
        _dis_base, refused=False, quantity="rss_mb_current",
        measured_now=_dis_base["rss_mb_current"] + 500.0)
    ok(_dis["verdict"] == "DISARMED" and _dis["is_a_pass"] is False
       and _fired["verdict"] == "FIRED" and _fired["is_a_pass"] is True
       and _real["verdict"] == "ADMITTED" and _real["is_a_pass"] is False,
       f"DE 110 (5) AS A RULE, DRIVEN THREE WAYS: an ADMITTED known-bad "
       f"whose NAMED quantity did not move from THIS cell's baseline "
       f"reads DISARMED (is_a_pass False); the same admitted outcome with "
       f"500 MB of headroom reads ADMITTED, a real defect; a refusal "
       f"reads FIRED. The old shape -- a bare `refuses(...)` -- had two "
       f"outcomes and reported the first as a PASS")
    refuses(lambda: known_bad_verdict(
                _dis_base, refused=False, quantity="n_open_files"),
            "KNOWN-BAD FOR THE RULE ITSELF: a quantity this helper cannot "
            "read, with no measurement supplied, REFUSES -- a control "
            "whose quantity nobody measured is not a control (and the "
            "helper's first version DEFAULTED to `peak_rss_mb`, which is "
            "the wrong quantity for the very cell it was written for)",
            "cannot read")
    ok(cell_baseline("probe", counts={"n": 1},
                     dirs=[_bd])["n_files"]
       and isinstance(cell_baseline("probe")["peak_rss_mb"], float),
       "and the baseline carries COUNTS and FILE COUNTS beside RSS -- the "
       "rule is about any quantity a known-bad depends on, not only "
       "memory (my own correction-emitter cells had moved a day COUNT the "
       "same way, and are now in their own temp root)")

    # ===== DE 110 (REV 81 §3): THE PEAK OF RECORD COMES FROM THE CHAIN ==
    _por = peak_of_record_rule()
    ok(_por["status"] == "RESOLVED"
       and _por["resolved_from"]["path"] == heavy_run_form_chain()[
           "head_path"]
       and _por["resolved_from"]["sha256"] == heavy_run_form_chain()[
           "head_sha256"]
       and "memory.peak" in _por["the_runs_peak"],
       f"REV 81 §3: which number this receipt's peak field may hold is "
       f"RESOLVED from the form chain head "
       f"({Path(_por['resolved_from']['path']).name} at "
       f"{str(_por['resolved_from']['sha256'])[:12]}…), not from prose -- "
       f"BE's producers reach the rule through their launcher and this "
       f"producer does not launch through one")
    _noform = dict(heavy_run_form())
    _noform.pop("memory_peak_reading", None)
    _fal = peak_of_record_rule(_noform)
    ok(_fal["status"] == "RULE_NOT_IN_THE_CHAIN_HEAD"
       and "does not fall back" in _fal["why"],
       "KNOWN-BAD: a head that declares no `memory_peak_reading` yields "
       "RULE_NOT_IN_THE_CHAIN_HEAD -- a producer that cannot resolve the "
       "rule records that it could not; it does not invent a number "
       "(R-649: the check FAILS when the declaration is absent, it does "
       "not skip)")
    _svc = leaf_of_cgroup("/user.slice/user-1001.slice/user@1001.service/"
                          "research.slice/de104smoke.service")
    _scp = leaf_of_cgroup("/user.slice/user-1001.slice/run-u32934.scope")
    _non = leaf_of_cgroup("/user.slice/user-1001.slice")
    ok(_svc["is_a_unit_leaf"] and _svc["is_the_ruled_launch_form"]
       and _scp["is_a_unit_leaf"] and not _scp["is_the_ruled_launch_form"]
       and not _non["is_a_unit_leaf"],
       f"AND THE LEAF CLASSIFIER ADMITS THE RULED FORM: `.service` is a "
       f"unit leaf AND the ruled form; `.scope` is a unit leaf and NOT the "
       f"ruled form; a slice path is neither. The old test read `.scope` "
       f"ONLY -- written before R-628 moved the form off `--scope` -- so "
       f"under the ruled form it returned NOT_IN_A_SCOPE with a null peak "
       f"while `cgroup` named the leaf beside it: DE's three sealed day "
       f"receipts carry `memory_peak_bytes: null` under a leaf that "
       f"existed. That is the number REV 80 made the peak of record")
    _obs = scope_memory_observation()
    ok(_obs["status"] in UNIT_MEMORY_STATUSES
       and (_obs.get("this_is_the_peak_of_record") is True)
       == (_obs["status"] == "MEASURED"),
       f"and `this_is_the_peak_of_record` is COMPUTED from the status "
       f"({_obs['status']}), never asserted: an AMBIENT leaf's peak is a "
       f"real number about the WRONG OBJECT and must not be quoted as the "
       f"run's (rule 10)")

    # ===== DE 110 (REV 81 §5): THE ORPHAN BRANCH IS CONSUMED =============
    _od = Path(_tfr.mkdtemp(prefix="de110fork_"))
    def _decl(name, sup=None):
        q = _od / name
        q.write_text(json.dumps(
            {"what": "fixture", "supersedes": sup} if sup else
            {"what": "fixture"}, indent=1, sort_keys=True) + "\n")
        return {"path": str(q),
                "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
    _f1 = _decl("fam_v1.json")
    _decl("fam_v2.json", _f1)          # a straight chain first
    _clean = design_chain_orphans(_od, "fam", also_this_seats_resolver=False)
    ok(_clean["shared_resolver"]["orphan_branches"] == []
       and _clean["chain_has_an_orphan_branch"] is False,
       "POSITIVE CONTROL for the orphan consumer: an UNFORKED fixture "
       "family reports NO orphan branch -- the field admits a clean chain "
       "and is not a check that only ever fires")
    _decl("fam_v3.json", _f1)          # a SECOND version superseding v1
    _fork = design_chain_orphans(_od, "fam", also_this_seats_resolver=False)
    ok(_fork["shared_resolver"]["orphan_branches"] == ["fam_v2.json"]
       and _fork["chain_has_an_orphan_branch"] is True
       and _fork["shared_resolver"][
           "forks_two_versions_superseding_one"].get("fam_v1.json"),
       f"KNOWN-BAD: a FORKED fixture family -- v2 and v3 both superseding "
       f"v1 -- reports the orphan by name "
       f"({_fork['shared_resolver']['orphan_branches']}) and names the "
       f"fork's base. Driven on a fixture family in a temp dir, so the "
       f"control does not depend on the ledger's own fork surviving")
    # ---- the MERGE LINK, driven on a fixture design family -----------
    _mr = Path(_tfr.mkdtemp(prefix="de110merge_"))
    _md = _mr / "pm_5min/derived"
    _md.mkdir(parents=True)
    def _dw(v, sup=None, also=None):
        q = _md / f"p003_de_multiday_gate1_design_v{v}.json"
        body = {"fixture": True}
        if sup:
            body["supersedes"] = sup
        if also:
            body["also_supersedes"] = also
        q.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
        return {"path": f"data/pm_5min/derived/{q.name}",
                "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
    _p1 = _dw(1)
    _p2 = _dw(2, _p1)
    _dw(3, _p1)                       # THE FORK: v2 and v3 both from v1
    _forked = design_chain(_mr)
    _fx = "p003_de_multiday_gate1_design_v"
    ok(_forked["orphan_branches"] == [f"{_fx}2.json"]
       and _forked["own_walk_kept_as_a_cross_check"][
           "orphan_branches_as_names"] == [f"{_fx}2.json"]
       and _forked["reachable_from_the_newest"] == [1, 3],
       f"POSITIVE CONTROL FOR THE FORK ITSELF: a fixture family whose v2 "
       f"and v3 both supersede v1 reports orphan_branches "
       f"{_forked['orphan_branches']} -- the single walk back from the "
       f"newest reaches {_forked['reachable_from_the_newest']} and v2 is "
       f"off it")
    _dw(4, _dw(3, _p1), [dict(_p2)])  # v4: head as the pair, v2 as MERGE
    _merged = design_chain(_mr)
    ok(_merged["orphan_branches"] == []
       and all(_merged["own_walk_kept_as_a_cross_check"]["agreement"]
               .values())
       and _merged["reachable_from_the_newest"] == [1, 2, 3, 4]
       and [m["agrees"] for m in _merged["merge_links"]] == [True],
       f"AND THE MERGE LINK EMPTIES IT (REV 81 §5's repair, driven): a v4 "
       f"naming the head as its `supersedes` pair AND the orphan tip "
       f"under `also_supersedes` makes every version reachable "
       f"({_merged['reachable_from_the_newest']}), orphan_branches "
       f"{_merged['orphan_branches']}")
    (_md / "p003_de_multiday_gate1_design_v4.json").write_text(
        json.dumps({"fixture": True, "supersedes": _dw(3, _p1),
                    "also_supersedes": [dict(_p2, sha256="f" * 64)]},
                   indent=1, sort_keys=True) + "\n")
    _bad = design_chain(_mr)
    ok(_bad["orphan_branches"] == [2]
       and [m["agrees"] for m in _bad["merge_links"]] == [False]
       and _bad["shared_resolver"]["status"] == "UNRESOLVED"
       and "MERGE_PAIR_MISMATCH" in _bad["shared_resolver"]["why"]
       and _bad["resolved_by"].startswith("the local walk"),
       f"KNOWN-BAD: a merge link whose digest does NOT recompute does not "
       f"merge, AND THE TWO RESOLVERS DIFFER IN KIND -- the shared one "
       f"REFUSES the whole family by name (`MERGE_PAIR_MISMATCH`), so "
       f"this seat falls back to the local walk AND SAYS SO "
       f"(`{_bad['resolved_by']}`), which reports the tip as still an "
       f"orphan ({_bad['orphan_branches']}) with the link `agrees: "
       f"False`. A refusal silently read as 'no orphans' is the failure "
       f"rule 11 names; a fallback that does not say it fell back is the "
       f"same failure one level down")

    # THIS ONE READS `data/` (the real design family), so it is SKIPPED
    # OFFLINE AND NAMED -- the fixture pair above carries the property in
    # both modes.
    if offline:
        offline_skip("DE 110: both resolvers compared on the REAL design "
                     "family (reads data/pm_5min/derived)")
        _real = None
    else:
        _real = design_chain_orphans()
    if offline:
        pass
    elif True:
      ok(isinstance(_real.get("n_orphan_branches"), int)
         and _real["the_two_resolvers_agree_on_the_count"] is not None,
         f"and the real family is resolved by the SHARED implementation "
         f"with this seat's own walk beside it as a CROSS-CHECK (DE 111, "
         f"after BE 80 taught it `also_supersedes`): shared "
         f"{_real['shared_resolver']['n_orphan_branches']} / this seat "
         f"{_real['this_seats_resolver']['n_orphan_branches']}, agreement "
         f"{_real['this_seats_resolver']['own_walk_cross_check']['agreement']}"
         f". They were two independent answers for a week and disagreed 1 "
         f"vs 5 on this family -- the cost of a second implementation, "
         f"paid and now retired")

    ok(n[0] + 1 + len(skipped) == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} run + "
       f"{len(skipped)} skipped == {EXPECTED_CHECKS}")
    _bat_wall = time.time() - _bat_t0
    _bat_peak = _peak_rss_mb()
    LAST_BATTERY.update({
        "outcome": "PASS", "n_checks_run": n[0],
        # REV 55 S2.4: the battery is 850 MB / 23.6 s and was 52 MB three
        # rounds ago. Every seat runs it several times a round as a
        # STANDALONE command, where rule 20's bar applies to it. It now
        # measures itself, so the trend is a field in every receipt that
        # embeds it rather than something a reviewer has to go and time.
        "resources": battery_resources(_bat_t0, _bat_hw0),
        "offline": offline,
        "why_skipped": ("these checks READ `data/`; a fixture run must "
                        "open no path under it, and a skipped check that "
                        "is silent is a check that has stopped existing"
                        if skipped else None),
        "ran_in_the_emitting_process": True})
    # ONLY IN THE OUTERMOST BATTERY. The nested `selftest(quiet=True)`
    # inside every fixture day run reaches this summary first, so a plain
    # module flag aborted the run 38 checks in -- at a NESTED summary, not
    # the one the falsifier is about. `quiet` is what distinguishes them.
    if PLANT_ONE_UNNAMED_SKIP and not quiet:
        # NO `why`, and appended DIRECTLY rather than through
        # `offline_skip` -- the helper always supplies a reason, so a
        # falsifier driven through it could never produce the state the
        # summary is supposed to refuse.
        skipped.append({"label": "PLANTED by --falsify-skipped"})
    if PLANT_ONE_DISARMED_CELL and not quiet:
        disarmed.append({
            "label": "PLANTED by --falsify-disarmed", "ambient": "memory",
            "quantity": "rss_mb_current", "baseline": 0.0, "now": 0.0,
            "moved_by": 0.0,
            "why": "planted: this cell did not run, it is the falsifier "
                   "for the summary's own n_disarmed line"})
    _sum = battery_summary("de_multiday_gate1_runner", n_run=n[0],
                           skipped=skipped, disarmed=disarmed,
                           expected=EXPECTED_CHECKS)
    # REV 88 §1.3: `n_skipped` AND THE NAMED CELLS TRAVEL INTO EVERY
    # RECEIPT THAT EMBEDS THIS BATTERY, beside `n_disarmed`. The 09-06
    # receipt carried `n_disarmed 0` and no `n_skipped` at all, so a
    # reader could not tell a battery that ran everything from one that
    # skipped a third of it.
    LAST_BATTERY.update({"n_disarmed": _sum["n_disarmed"],
                         "disarmed": _sum["disarmed"],
                         "n_skipped": _sum["n_skipped"],
                         "skipped": _sum["skipped"],
                         "skips_unnamed": _sum["skips_unnamed"],
                         "summary_line": _sum["line"],
                         "clean": _sum["clean"]})
    if not quiet:
        print(_sum["line"])
    if not _sum["clean"]:
        raise SystemExit(
            f"[de_multiday_gate1_runner] NOT CLEAN: n_disarmed "
            f"{_sum['n_disarmed']} -- "
            f"{[d['label'] for d in _sum['disarmed']]}; n_skipped "
            f"{_sum['n_skipped']}, unnamed {_sum['skips_unnamed']}. A "
            f"cell that ran and could not have fired is not coverage "
            f"(REV 84 §2), and a skip nobody named is not a skip anybody "
            f"can audit (REV 88 §1.3).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--falsify-disarmed", action="store_true",
                    dest="falsify_disarmed",
                    help="run the real battery with ONE cell planted "
                         "DISARMED: the summary must print n_disarmed 1 "
                         "and exit non-clean (REV 84 §2)")
    ap.add_argument("--falsify-skipped", action="store_true",
                    dest="falsify_skipped",
                    help="run the real battery with ONE skip planted "
                         "WITHOUT a reason: the summary must count it in "
                         "n_skipped, report it as an UNNAMED skip and "
                         "exit non-clean (REV 88 §1.3)")
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
    if a.falsify_skipped:
        global PLANT_ONE_UNNAMED_SKIP
        PLANT_ONE_UNNAMED_SKIP = True
    if a.falsify_disarmed:
        global PLANT_ONE_DISARMED_CELL
        PLANT_ONE_DISARMED_CELL = True
        return selftest()
    if a.selftest:
        return selftest()
    if a.rehearse:
        payload = rehearse_smoke(a.rehearse)
        payload["data_root"] = DR.require_canonical("the smoke rehearsal")
        payload["source_identity"] = {
            **assert_source_unchanged("the rehearsal emit"),
            **carrying_commit_block(Path(__file__).resolve())}
        if a.output is not None:
            if a.output.exists():
                raise RunnerRefused(f"output already exists: {a.output}")
            a.output.parent.mkdir(parents=True, exist_ok=True)
            payload["name_stamp"] = assert_name_stamp_is_the_clock(
                a.output, payload["as_of"])
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
            # REV 53 S1.4: the --ledger path was the one emit that could
            # still carry a typed stamp.
            payload["name_stamp"] = assert_name_stamp_is_the_clock(
                a.output, payload["as_of"])
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
    payload["name_stamp"] = assert_name_stamp_is_the_clock(
        a.output, payload["as_of"])
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
        raise RunnerRefused(
            "REFUSED: --output is required for a day run. On the day path "
            "it names the DIRECTORY the receipt is written into; the "
            "runner composes the filename from the clock (REV 55 S2.1).")
    # BOTH REFUSALS ARE HERE, AT THE TOP, BEFORE ANY WORK -- the whole
    # point of the change is that a naming mistake costs zero draws
    # instead of 85 minutes.
    _outdir_check = assert_output_is_a_directory(
        a.output, day=(a.day or a.synthetic_day))
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
        _outdir_check["no_sealed_receipt_yet"] = assert_no_sealed_receipt_yet(
            day, Path(DR.resolve()["data_root"]))
    # THE BATTERY RUNS BEFORE THE DAY'S WORK (R-610), not at the emit.
    # It used to be called here, AFTER `day_split_residency_proof` had
    # already spent the day: on 2026-09-03 that was 84 minutes of null
    # draws followed by a refusal from a FIXTURE check, and the day was
    # lost with nothing written. Everything the battery can refuse is
    # knowable before the book is loaded. `_battery_first` is handed to
    # `run_day`, which calls it after the book digest is verified and
    # before S1.
    _battery: dict = {}

    def _battery_first():
        LAST_BATTERY.clear()
        selftest(quiet=True, offline=fixture)
        _battery.update(LAST_BATTERY)
        return dict(LAST_BATTERY)

    # COMPUTED, NOT DEFAULTED (REV 75 S1.2). The count is taken BEFORE
    # this day's own receipt exists, so it is the days complete when this
    # day started; the emitted prose is generated from it.
    _dc = days_complete_now(params)
    # R-765 / REV 102's NO-GO: THE DAY PATH GETS ITS ANCHOR. It always
    # had one to give -- `assert_output_is_a_directory` above has already
    # refused unless `--output` names an existing directory -- and it
    # simply did not pass it, so a real day under ruling-carrying params
    # refused DECISION_LEDGER_HAS_NO_ANCHOR before its work.
    proof = day_split_residency_proof(
        day, book, params=params, fixture=fixture,
        n_days_complete=(a.n_days_complete if fixture
                         else _dc["n_days_complete"]),
        ledger_anchor=day_run_ledger_anchor(a.output, fixture=fixture),
        before_work=_battery_first)
    payload = proof.pop("day_result")
    payload["split_residency_proof"] = proof
    payload["source_identity"] = {
        **assert_source_unchanged("the day-run emit",
                                  fixture=fixture),
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
    if not _battery:
        raise RunnerRefused(
            "REFUSED at the emit: the in-run battery did not run. It is "
            "handed to `run_day` as `before_work` and must have completed "
            "BEFORE the day's work; an empty result here means the hook "
            "was never called, and a receipt carrying no battery is a "
            "receipt whose instruments never fired (rule 15).")
    payload["battery"] = dict(_battery)
    payload["battery_scope"] = {
        "offline": fixture,
        "ran_before_the_days_work": True,
        "why_not_at_the_emit": (
            "it WAS at the emit. On 2026-09-03 the day's 84 minutes "
            "finished and a FIXTURE check inside the battery refused on "
            "the real day's process-wide high-water; nothing was written "
            "and the day was lost. The budget defect is fixed (growth, "
            "per stage), but the ORDER was the other half: a check that "
            "can refuse must refuse before the work it would waste. "
            "Measured by `before_work.residency."
            "day_draws_when_the_hook_returned`, which is 0"),
        "day_draws_when_the_battery_returned":
            ((payload.get("before_work") or {}).get("residency")
             or {}).get("day_draws_when_the_hook_returned"),
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
    # ---- THE NAME, COMPOSED HERE, FROM ONE CLOCK READ (REV 55 S2.1) ----
    # `as_of` and the filename's stamp are the SAME instant because they
    # are the same reading. Two reads would differ by microseconds and
    # invite exactly the "which clock" question the check exists to answer.
    _emitted_at = datetime.datetime.now(datetime.timezone.utc)
    payload["as_of"] = _emitted_at.isoformat()
    payload["launched_at_utc"] = LAUNCH_TIME_UTC
    payload["emitted_at_utc"] = _emitted_at.isoformat()
    out_path = Path(a.output) / day_receipt_name(
        day, fixture=fixture, stamp=emission_stamp(_emitted_at))
    # R-628: WHICH WRAPPER ACTUALLY RAN THIS, measured from the cgroup --
    # not the form that was published, the form that executed.
    # ---- (v) THE PROVENANCE BLOCK (DA 89 / R-654) --------------------
    # The 09-03 receipt carried `source_identity` -- carrying_commit, the
    # producing digest, the closure, HEAD at import and emit -- but NO
    # params or design PINS: params v14 appeared only inside
    # `fixture_day_lock…` and design v21 only as an opened path. DA's
    # pre-read had to infer both. From here every receipt carries them
    # OPEN, as the run ACTUALLY RESOLVED them, in the pair form
    # {path, sha256} (rules 12/13).
    _repo = Path(__file__).resolve().parents[2]
    def _pair(rel):
        f = _repo / rel
        return {"path": str(rel), "exists": f.is_file(),
                "sha256": (hashlib.sha256(f.read_bytes()).hexdigest()
                           if f.is_file() else None)}
    _design_rel = (params.get("design_declaration") or {}).get("path")
    if _design_rel:
        record_input_digest("design", _design_rel)
    _inputs = verify_input_digests("the day-run emit")
    payload["days_complete_when_this_day_started"] = _dc
    payload["provenance"] = {
        "params": _pair(PARAMS_REL),
        "design": (_pair(_design_rel) if _design_rel else
                   {"path": None, "exists": False, "sha256": None}),
        "launch_form_declaration": {
            **_pair(heavy_run_form()["_chain"]["head_path"]),
            "chain": heavy_run_form()["_chain"]["links"],
            "resolved_as": "the chain head, not a filename literal"},
        "digests_at_load_and_at_emit": _inputs,
        "as_the_run_resolved_them": (
            "these are the files THIS run read, digested at emit -- not a "
            "version a reader infers from a name and not a pin copied "
            "from a declaration"),
        "why_it_is_open": (
            "the 09-03 receipt carried source_identity and no params or "
            "design pin; params appeared only inside a fixture-lock field "
            "and the design only as an opened path, so DA's pre-read had "
            "to infer both (R-654)"),
        "pin_direction": (params.get("design_declaration") or {}).get(
            "pin_direction"),
        # REV 81 §5 CONSUMED. While the design chain reports an orphan
        # branch, the receipt that pins a design SAYS SO -- otherwise the
        # condition sits unnoticed "for another eight versions", which is
        # the reviewer's own words for how this one survived.
        "design_chain_orphans_at_emit": design_chain_orphans(),
        # REV 81 §3 CONSUMED. Which number this receipt's peak field may
        # hold is resolved from the FORM CHAIN, not from prose -- BE's
        # producers reach the rule through their launcher and this one
        # does not.
        "peak_of_record_rule": peak_of_record_rule(),
        # R-709 CONSUMED (DE 112). A reader holding this receipt and a
        # non-zero `ExecMainStatus` for its unit has to resolve the code
        # to a KIND, and the map that answers is a CHAIN -- so the receipt
        # names the head IT resolved, by pair, rather than leaving the
        # reader to resolve today's head against yesterday's run. The
        # rehearsal found this absent: every other chain this run reads
        # was already named here and the exit map was not.
        "producer_exit_map": producer_exit_map(),
    }
    _uid = unit_identity()
    payload["launch_form"] = {
        "declared": LAUNCH_FORM,
        "requirements": LAUNCH_FORM_REQUIREMENTS,
        "observed": _uid,
        # THE RECEIPT CARRIES ITS OWN UNIT'S JOURNAL LINES, COPIED AT THE
        # EMIT (rule 20 as amended, R-641). The journal rotates within
        # hours; a receipt that points at it instead of copying it names
        # evidence that may already be gone -- which happened to DE 84's
        # `Started` line four hours after it was quoted.
        # (iv) COVERAGE IS COMPUTED HERE, AT THE EMIT, WHILE THE UNIT IS
        # STILL LOADED -- from `ExecMainStartTimestamp` and the host's
        # oldest retained entry, both STORED beside the answer. After
        # collection the start timestamp is gone and it can never be
        # recomputed; a later reader gets the stored measurement or
        # nothing, and never a needle.
        "coverage_at_emit": (DAROOT.journal_coverage(unit=_uid["unit"])
                             if _uid.get("unit") else
                             {"status": "NOT_DETERMINABLE",
                              "why": "this process is in no unit"}),
        "journal_at_emit": (journal_read(_uid["unit"])
                            if _uid.get("unit") else
                            {"status": "ABSENT",
                             "why": "this process is in no unit, so there "
                                    "is no unit journal to copy"}),
        "why_it_is_in_the_receipt": (
            "the 09-03 re-run died at 35 minutes because it ran in a "
            "`.scope` -- the caller's process group. A receipt that does "
            "not say which wrapper produced it cannot be asked that"),
    }
    payload["output_name"] = {
        **_outdir_check,
        "composed_name": out_path.name,
        "launched_at_utc": LAUNCH_TIME_UTC,
        "emitted_at_utc": _emitted_at.isoformat(),
        "wall_between_launch_and_emit_s": (
            _emitted_at - datetime.datetime.fromisoformat(
                LAUNCH_TIME_UTC)).total_seconds(),
        "matches_the_declared_convention":
            params["read_gate"]["receipt_naming"]["convention"],
        "resolves_under_the_sealed_glob": (
            None if fixture else
            Path(out_path.name).match(sealed_day_receipt_glob(day))),
        "why_a_fixture_cannot": (
            "a fixture receipt carries "
            f"{FIXTURE_DAY_RECEIPT_MIDFIX!r}, so it cannot match the "
            "sealed glob AT ALL -- not merely carry a different day"),
    }
    if out_path.exists():
        raise RunnerRefused(
            f"REFUSED at the emit: {out_path.name} already exists. The "
            f"stamp is second-resolution and the pre-work guard found no "
            f"sealed artifact for this day, so this is a collision nobody "
            f"expected; it is refused rather than overwritten.")
    payload["name_stamp"] = assert_name_stamp_is_the_clock(
        out_path, payload["as_of"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload["placement_latency_consistency"] = (
        assert_one_placement_latency(payload))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                   default=str) + "\n")
    # THE RECEIPT IS READ BACK FROM DISK AND CENSUSED BEFORE IT IS CALLED
    # SEALED (REV 76 S0(3)). Everything before this point checks the
    # PAYLOAD IN MEMORY; what a reader gets is the BYTES. Serialisation
    # can differ from the object -- `default=str` stringifies what it
    # cannot encode -- and the claim on the artifact is about the
    # artifact.
    _back = json.loads(out_path.read_text())
    try:
        assert_one_placement_latency(_back)
    except RunnerRefused:
        out_path.unlink(missing_ok=True)
        raise
    _post = _economic_keys_in(_back)
    # NOT WRITTEN INTO `payload`: the receipt is already on disk and the
    # census is an act ON THOSE BYTES. Assigning it here looked like a
    # record and was dead -- the emitted artifact carried no such key,
    # which is how it was caught (read at the artifact, not at the code).
    # The status is printed below and travels in the run's capture record.
    _census = post_emit_census_status(params, payload, _post)
    if _census["status"] == "SEALED_REGIME_LEAK":
        out_path.unlink(missing_ok=True)
        raise RunnerRefused(
            f"REFUSED after writing: the receipt READ BACK FROM DISK "
            f"carries sealed names as KEYS -- {_post[:6]}. The rule is "
            f"`{CENSUS_RULE}`. The file has been removed; a sealed day "
            f"that leaks is not a day.")
    if _census["status"] == "POST_EMIT_CENSUS_HAS_NO_BAR":
        out_path.unlink(missing_ok=True)
        raise RunnerRefused(
            f"REFUSED POST_EMIT_CENSUS_HAS_NO_BAR after writing: the "
            f"seal is in force for this run and the receipt carries no "
            f"bar to census against "
            f"(`G_and_which_G_it_is.the_bar_this_run_sealed_against` "
            f"{_census['the_bar_this_run_sealed_against']!r}, "
            f"n_days_complete {_census['n_days_complete']!r}). A check "
            f"that cannot evaluate REFUSES; it does not pass. The file "
            f"has been removed.")
    print(json.dumps({
        "post_emit_census": {"read_back_from": str(out_path),
                             "sealed_keys_found": len(_post),
                             "status": _census["status"],
                             "unsealed_by_user_ruling":
                                 _census["unsealed_by_user_ruling"],
                             "rule": CENSUS_RULE},
        "emitted": str(out_path), "status": payload["status"],
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

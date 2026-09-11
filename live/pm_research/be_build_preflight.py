#!/usr/bin/env python3
"""BE 148 -- REFUSE BEFORE THE LOCK, NOT AFTER TWENTY MINUTES OF WORK.

USER DIRECTIVE: clear every potential build failure before it costs a day.
Each check below is one that has ALREADY cost this programme a run, or that
the pinned builders refuse on with no warning until they are deep into a
stage. It reads metadata only: era resolution, mask presence, ledger span,
window supply, worktree identity and free disk. It computes no score, no
outcome and no P&L, so it does not consume a protected day (the DA 182
boundary).

IT EDITS NOTHING PINNED. The five build modules keep their 7ed5a90 digests;
this file is new and the launcher calls it.

DIGESTS ARE DERIVED FROM THE PIN, NEVER TYPED (rule 32). The check compares
the worktree's bytes against `git show <pin>:<path>`, so there is no literal
to go stale -- which is the exact defect that put pre-optimization digests in
params v29 and refused a valuation 25 times.

Statuses, one row per (day, check):
  PASS              the check holds
  WOULD_FAIL:<name> a build would fail, and this names how
  NOT_YET:<why>     the day is not buildable yet (not closed; D+1 unavailable)

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  every day either PASSes or is NOT_YET     2  usage
  1  at least one WOULD_FAIL                   3  the preflight itself refused
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PIN = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"
WT_FWD = "/home/yuqing/ctaNew-wt-fwd"
WANT_ERA = "clob_v4_1"
# the five the freeze pins; PATHS only -- the digests come from the pin itself
PINNED = (
    "live/pm_research/be_daybook_build.py",
    "live/pm_research/be_gate1_fragment.py",
    "live/pm_research/be_gate1_state_tape.py",
    "live/pm_research/de_phase4_diag_runner.py",
    "live/pm_research/de_head_scoring.py",
)
# measured: tape 1.18 GB + book 0.39 GB on 09-08; 2 GB is the margin
NEED_BYTES = 2 * 1024**3
WINDOW_S = 300
DAY_S = 86400
MARKOUT_S = 5.0          # harmful_exposure_rows: a fill at the close is
                         # valued at +5 s, i.e. 5 s into D+1


WRONG_TREE = "PREFLIGHT_IMPORTED_FROM_ANOTHER_TREE"
# The modules whose ANSWERS this preflight reports. They must come from the
# tree under test.
TREE_MODULES = ("flow_intensity", "be_era_for_day", "be_gate1_fragment")


class PreflightRefused(RuntimeError):
    """The preflight cannot be trusted, so no verdict is reported."""


def import_from_tree(tree: str = WT_FWD, *, purge: bool = True) -> dict:
    """Import the day-check modules FROM THE TREE UNDER TEST, and PROVE it.

    REVIEW 158's seam: this module lives in the SHARED tree, so
    `sys.path.insert(Path(__file__).parent)` made every day check -- era,
    population, slug inputs -- run SHARED-TREE code while the tree checks
    asserted wt-fwd's HEAD and digests. The preflight could certify wt-fwd
    using code from a different tree, which is the same class as verifying a
    claim against a proxy rather than the artifact it names.

    Two halves, because path order alone is not a control: the tree's paths go
    FIRST, any already-imported copy from another tree is PURGED from
    sys.modules (an import returns the cached module and would silently
    ignore the path change), and then every imported module's resolved
    `__file__` is checked to lie under the tree. The check is what refuses;
    the path order is only what makes it pass."""
    root = str(Path(tree).resolve())
    pkg = str(Path(root) / "live" / "pm_research")
    me = str(Path(__file__).resolve())
    if purge:
        for name, mod in list(sys.modules.items()):
            if name == "__main__":
                continue
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            rf = str(Path(f).resolve())
            if rf == me:
                continue
            if "/live/pm_research/" in rf and not rf.startswith(root + "/"):
                del sys.modules[name]
    for entry in (root, pkg):          # pkg ends up first
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    mods, bad = {}, []
    for name in TREE_MODULES:
        try:
            mods[name] = importlib.import_module(name)
        except Exception as exc:
            raise PreflightRefused(
                f"REFUSED {WRONG_TREE}: {name} is not importable from {root} "
                f"({type(exc).__name__}: {exc})") from None
    for name, mod in mods.items():
        f = getattr(mod, "__file__", None)
        rf = str(Path(f).resolve()) if f else None
        if not rf or not rf.startswith(root + "/"):
            bad.append(f"{name} <- {rf}")
    if bad:
        raise PreflightRefused(
            f"REFUSED {WRONG_TREE}: the tree under test is {root}, but "
            f"{len(bad)} module(s) were imported from elsewhere: "
            f"{'; '.join(bad)}. A preflight that certifies one tree using "
            f"another tree's code certifies nothing.")
    return mods


def _now() -> float:
    return datetime.datetime.now(datetime.UTC).timestamp()


def _day_bounds(day: str) -> tuple[int, int]:
    d = datetime.datetime.strptime(day, "%Y%m%d").replace(
        tzinfo=datetime.UTC)
    return int(d.timestamp()), int(d.timestamp()) + DAY_S


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_tree() -> list[tuple[str, str, str]]:
    """Worktree identity. Day-independent, so reported once under '-'."""
    out = []
    env = os.environ.get("BE_WORKTREE")
    out.append(("-", "BE_WORKTREE is wt-fwd",
                "PASS" if env == WT_FWD else
                f"WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD(read={env!r})"))
    try:
        head = subprocess.run(["git", "-C", WT_FWD, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception as exc:
        head = f"<{type(exc).__name__}>"
    out.append(("-", "wt-fwd HEAD is the pin",
                "PASS" if head == PIN else
                f"WOULD_FAIL:WT_FWD_HEAD_MOVED(read={head[:12]})"))
    dirty = subprocess.run(["git", "-C", WT_FWD, "status", "--short"],
                           capture_output=True, text=True).stdout
    extra = [l for l in dirty.splitlines() if l.strip() and l.strip() != "?? data"]
    out.append(("-", "wt-fwd clean",
                "PASS" if not extra else
                f"WOULD_FAIL:WT_FWD_DIRTY({extra[0][:40]})"))
    for path in PINNED:
        name = path.split("/")[-1]
        blob = subprocess.run(["git", "-C", WT_FWD, "show", f"{PIN}:{path}"],
                              capture_output=True).stdout
        want = _sha(blob) if blob else None
        f = Path(WT_FWD) / path
        got = _sha(f.read_bytes()) if f.is_file() else None
        out.append(("-", f"pinned digest {name}",
                    "PASS" if (want and got == want) else
                    f"WOULD_FAIL:PINNED_DIGEST_MOVED({name},"
                    f"pin={str(want)[:10]},disk={str(got)[:10]})"))
    return out


def check_disk() -> tuple[str, str, str, dict]:
    u = shutil.disk_usage("/home/yuqing/ctaNew/data")
    ok = u.free >= NEED_BYTES
    return ("-", "free disk covers one tape + one book",
            "PASS" if ok else
            f"WOULD_FAIL:DISK_SHORT(free={u.free/1024**3:.2f}GiB,"
            f"need={NEED_BYTES/1024**3:.2f}GiB)",
            {"free_bytes": u.free, "free_gib": round(u.free / 1024**3, 2),
             "need_gib": round(NEED_BYTES / 1024**3, 2),
             "total_gib": round(u.total / 1024**3, 2)})


def check_day(day: str, stage: str | None = None,
              mods: dict | None = None) -> list[tuple[str, str, str]]:
    # REVIEW 158: the day checks run the TREE UNDER TEST's code, never the
    # shared tree's. `import_from_tree` refuses if that is not what happened.
    m = mods if mods is not None else import_from_tree(
        os.environ.get("BE_WORKTREE") or WT_FWD)
    fi, EFD, FR = (m["flow_intensity"], m["be_era_for_day"],
                   m["be_gate1_fragment"])

    rows = []
    d0, d1 = _day_bounds(day)

    # 0. is the day even buildable? D+1 by MARKOUT_S, not merely D closed.
    need_until = d1 + MARKOUT_S
    if _now() < need_until:
        why = (f"day closes {datetime.datetime.fromtimestamp(d1, datetime.UTC):%Y-%m-%dT%H:%M:%SZ}"
               f" and the fragment reads to +{MARKOUT_S:g}s")
        return [(day, "day is closed (+markout)", f"NOT_YET:{why}")]
    rows.append((day, "day is closed (+markout)", "PASS"))

    # 1. THE ERA. This is the class that forced 09-07's rebuild: the module
    #    literal is day-independent and wrong for September.
    try:
        pop = FR.population(day)
        slugs = sorted(pop["slugs"])
        era = EFD.resolve(fi, pop["day"], slugs)["era"]
        rows.append((day, f"era resolves to {WANT_ERA}",
                     "PASS" if era == WANT_ERA else
                     f"WOULD_FAIL:ERA_NOT_{WANT_ERA}(resolved={era})"))
        rows.append((day, "day-resolved era differs from the module literal",
                     "PASS" if era != getattr(fi, "ERA", None) else
                     f"WOULD_FAIL:ERA_EQUALS_LITERAL({era}) -- if these ever "
                     f"agree the literal-vs-resolved bug becomes invisible"))
    except Exception as exc:
        return rows + [(day, "population/era",
                        f"WOULD_FAIL:POPULATION_UNREADABLE({type(exc).__name__}:"
                        f"{str(exc)[:60]})")]

    # 2. WINDOW SUPPLY. 288 is a full day; fewer is a masked window and the
    #    count is REPORTED rather than asserted, because 09-07 legitimately
    #    supplies 287.
    n = pop["n_windows"]
    rows.append((day, f"windows supplied (n={n})",
                 "PASS" if n > 0 else "WOULD_FAIL:NO_WINDOWS_SUPPLIED"))
    starts = sorted(int(s.rsplit("-", 1)[1]) for s in slugs)
    expected = list(range(starts[0], starts[-1] + WINDOW_S, WINDOW_S))
    missing = sorted(set(expected) - set(starts))
    rows.append((day, f"interior windows missing (n={len(missing)})",
                 "PASS" if len(missing) <= 1 else
                 f"WOULD_FAIL:MANY_MISSING_WINDOWS({len(missing)})"))

    # 3. THE INPUTS THE SELECTOR REFUSES ON. `selector_for` raises if any slug
    #    lacks an archive path or a token -- deep inside the fragment build,
    #    after the lock is taken. Checked here instead.
    paths, toks = fi._archive_paths(), fi.token_map()
    miss = [s for s in slugs if s not in paths or s not in toks]
    rows.append((day, "every supplied slug has an archive path and token",
                 "PASS" if not miss else
                 f"WOULD_FAIL:SLUG_INPUTS_MISSING(n={len(miss)},"
                 f"first={miss[0]})"))

    # 4. DA's MASK for the day.
    m = Path(f"/home/yuqing/ctaNew/data/pm_5min/derived/da_blackout_mask_{day}.json")
    if not m.is_file():
        rows.append((day, "DA blackout mask present", f"NOT_YET:NO_MASK_FOR_{day}"))
    else:
        try:
            json.loads(m.read_text())
            rows.append((day, "DA blackout mask present and parses", "PASS"))
        except Exception as exc:
            rows.append((day, "DA blackout mask parses",
                         f"WOULD_FAIL:MASK_UNPARSEABLE({type(exc).__name__})"))

    # 5. THE GAP LEDGER COVERS THE DAY'S SPAN. Zero gaps is legitimate; a
    #    ledger that STOPS before the day is not.
    last = 0
    if fi.GAPS.exists():
        with fi.GAPS.open() as fh:
            for line in fh:
                try:
                    ws = json.loads(line).get("window_start")
                except json.JSONDecodeError:
                    continue
                if ws:
                    last = max(last, int(ws))
    rows.append((day, "gap ledger extends past the day's span",
                 "PASS" if last >= d1 - WINDOW_S else
                 f"WOULD_FAIL:GAP_LEDGER_STOPS_EARLY(last={last},day_end={d1})"))

    # 6. THE STAGE GRAPH: each stage's OUTPUT must be absent (the builders
    #    refuse rather than overwrite, rule 13) and its INPUT must be present.
    #    The 09-08 book died 13 minutes in on a MISSING TAPE -- an input, which
    #    an output-absent check alone does not cover.
    D = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
    frag = D / f"harmful_exposure_rows_v3_gate1_{day}_btc.json"
    tape = D / f"phase2_state_tape_gate1_{day}_btc.json"
    book = D / f"be_daybook_{day}_btc__L250ms__FWD1.pkl"
    want = stage or "fragment"
    outs = {"fragment": frag, "tape": tape, "book": book}
    # ONLY THE STAGE UNDER TEST must have its output absent. BE 155: the first
    # form asserted ALL THREE absent, so running the TAPE flagged
    # FRAGMENT_EXISTS -- the fragment the tape REQUIRES. A precondition
    # checker that refuses the normal state of the pipeline is worse than no
    # checker: it blocks the build it exists to protect.
    out_p = outs[want]
    rows.append((day, f"{want} output absent",
                 "PASS" if not out_p.exists() else
                 f"WOULD_FAIL:{want.upper()}_EXISTS -- the builder "
                 f"refuses rather than overwriting (rule 13)"))
    for label, q in outs.items():
        if label != want:
            rows.append((day, f"({label} present: {str(q.exists()).lower()})",
                         "PASS"))
    if want != "fragment":
        dep_label, dep_p = dict(
            tape=("fragment", frag), book=("tape", tape))[want]
        rows.append((day, f"{want} input present ({dep_label})",
                     "PASS" if dep_p.exists() else
                     f"WOULD_FAIL:{want.upper()}_INPUT_MISSING({dep_label}) -- "
                     f"the builder refuses AFTER taking the lock"))
    return rows


def falsify() -> int:
    """Rule 15: the known-bad is an UNSET BE_WORKTREE, and it must fail BY NAME."""
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    real = os.environ.get("BE_WORKTREE")
    try:
        os.environ.pop("BE_WORKTREE", None)
        rows = check_tree()
        hit = [s for _, c, s in rows if c == "BE_WORKTREE is wt-fwd"]
        note("with BE_WORKTREE UNSET the tree check fails BY NAME",
             len(hit) == 1 and hit[0].startswith("WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD"))
        os.environ["BE_WORKTREE"] = "/home/yuqing/ctaNew-wt-be"
        rows = check_tree()
        hit = [s for _, c, s in rows if c == "BE_WORKTREE is wt-fwd"]
        note("and the WRONG tree fails by the same name, not silently",
             len(hit) == 1 and hit[0].startswith("WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD"))
        os.environ["BE_WORKTREE"] = WT_FWD
        rows = check_tree()
        hit = [s for _, c, s in rows if c == "BE_WORKTREE is wt-fwd"]
        note("and the RIGHT tree PASSES -- the control admits as well as fires",
             hit == ["PASS"])
        note("the five pinned digests are DERIVED from the pin, not typed",
             all(f"pinned digest {p.split('/')[-1]}" in [c for _, c, _ in rows]
                 for p in PINNED))
        note("a day that is not closed is NOT_YET, never WOULD_FAIL",
             all(s.startswith("NOT_YET:")
                 for _, _, s in check_day("20261231")))
        note("a stage whose INPUT is missing fails BY NAME, before the lock",
             any(st.startswith("WOULD_FAIL:BOOK_INPUT_MISSING")
                 for _, _, st in check_day("20260909", stage="book")))
        # BE 155 regression: the TAPE stage must NOT be refused because the
        # FRAGMENT it consumes exists. Only the stage under test is checked
        # for an absent output.
        _t = check_day("20260909", stage="tape")
        note("the TAPE stage is NOT refused for FRAGMENT_EXISTS -- only its "
             "OWN output must be absent",
             not any("FRAGMENT_EXISTS" in st for _, _, st in _t))
        note("and the tape stage still requires its input to be present",
             any("tape input present (fragment)" in c for _, c, _ in _t))
        note("and the same day at the FRAGMENT stage has no input check to fail",
             not any("INPUT_MISSING" in st
                     for _, _, st in check_day("20260909", stage="fragment")))
        # REVIEW 158: with the SHARED tree first on sys.path and its modules
        # already cached, the preflight must REFUSE BY NAME rather than
        # certify wt-fwd using another tree's code.
        shared = "/home/yuqing/ctaNew"
        sys.path.insert(0, shared + "/live/pm_research")
        sys.path.insert(0, shared)
        for _n in TREE_MODULES:
            sys.modules.pop(_n, None)
        _saved = dict(sys.modules)
        import importlib as _il
        for _n in TREE_MODULES:                 # cache them FROM THE SHARED TREE
            sys.modules[_n] = _il.import_module(_n)
        from_shared = all(
            str(Path(sys.modules[_n].__file__).resolve()).startswith(shared + "/live")
            for _n in TREE_MODULES)
        note("the known-bad is real: the shared tree's modules ARE cached",
             from_shared)
        try:
            import_from_tree(WT_FWD, purge=False)
            note("with the SHARED tree first on sys.path it REFUSES by name", False)
        except PreflightRefused as exc:
            note("with the SHARED tree first on sys.path it REFUSES by name",
                 WRONG_TREE in str(exc))
        note("and with purging ON it ADMITS, importing from wt-fwd",
             all(str(Path(m.__file__).resolve()).startswith(WT_FWD + "/")
                 for m in import_from_tree(WT_FWD).values()))
    finally:
        if real is None:
            os.environ.pop("BE_WORKTREE", None)
        else:
            os.environ["BE_WORKTREE"] = real
    # ---- REVIEW 164: THE ENTRY POINT, INVOKED AS PRODUCTION INVOKES IT.
    # Every cell above calls an inner function directly: unit-proven,
    # WIRING-unproven. These run the module as a SUBPROCESS, from a cwd that
    # is NOT the tree root, with the launcher's real environment, and require
    # the SAME VERDICT the direct call gives. That distinction is not
    # academic here -- my own census shipped a KeyError in its file-reading
    # wrapper while nine cells driving its pure core stayed green.
    ME = str(Path(__file__).resolve())
    FOREIGN_CWD = "/tmp"

    def entry(argv, worktree=WT_FWD):
        env = dict(os.environ)
        env["PM_DATA_ROOT"] = "/home/yuqing/ctaNew"
        if worktree is None:
            env.pop("BE_WORKTREE", None)
        else:
            env["BE_WORKTREE"] = worktree
        return subprocess.run([sys.executable, ME, *argv], cwd=FOREIGN_CWD,
                              env=env, capture_output=True, text=True,
                              timeout=900)

    r = entry(["--stage", "tape", "20260910"])
    direct = check_day("20260910", stage="tape",
                       mods=import_from_tree(WT_FWD))
    d_fail = sorted(st for _, _, st in direct if st.startswith("WOULD_FAIL"))
    note("entry point from a FOREIGN CWD gives the SAME verdict as the "
         "direct call",
         all(f in r.stdout for f in d_fail) and bool(d_fail)
         and r.returncode == 1)
    note("and it is the input known-bad, named",
         "WOULD_FAIL:TAPE_INPUT_MISSING(fragment)" in r.stdout)

    r = entry(["--stage", "fragment", "20260910"])
    note("entry point ADMITS the stage whose preconditions hold (rc 0)",
         r.returncode == 0 and "n_would_fail\": 0" in r.stdout)

    r = entry(["--stage", "fragment", "20260910"], worktree=None)
    note("entry point with BE_WORKTREE UNSET fails BY NAME and returns 1",
         "WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD" in r.stdout
         and r.returncode == 1)

    r = entry(["--stage", "fragment", "20260910"],
              worktree="/home/yuqing/ctaNew-wt-be")
    note("entry point with the WRONG tree fails by the same name",
         "WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD" in r.stdout
         and r.returncode == 1)

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_build_preflight", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    stage = None
    if "--stage" in argv:
        i = argv.index("--stage"); stage = argv[i + 1]; del argv[i:i + 2]
        if stage not in ("fragment", "tape", "book"):
            print("--stage must be fragment|tape|book"); return 2
    days = [a for a in argv if a.isdigit()]
    if not days:
        print("usage: be_build_preflight.py <YYYYMMDD> [...] | --selftest")
        return 2
    rows = list(check_tree())
    dday, dcheck, dstatus, dnums = check_disk()
    rows.append((dday, dcheck, dstatus))
    tree = os.environ.get("BE_WORKTREE") or WT_FWD
    try:
        mods = import_from_tree(tree)
        rows.append(("-", "day checks imported from the tree under test",
                     "PASS"))
    except PreflightRefused as exc:
        rows.append(("-", "day checks imported from the tree under test",
                     str(exc).split(":", 1)[0].replace("REFUSED ",
                                                       "WOULD_FAIL:")))
        print(exc)
        for day, c, st in rows:
            print(f"{day:<10}{c:<52}{st}")
        return 1
    for d in days:
        rows.extend(check_day(d, stage, mods))
    w = max(len(c) for _, c, _ in rows) + 2
    print(f"{'day':<10}{'check':<{w}}status")
    print("-" * (10 + w + 40))
    for day, c, s in rows:
        print(f"{day:<10}{c:<{w}}{s}")
    fails = [(d, c, s) for d, c, s in rows if s.startswith("WOULD_FAIL")]
    notyet = [(d, c, s) for d, c, s in rows if s.startswith("NOT_YET")]
    print()
    print(f"disk: free {dnums['free_gib']} GiB of {dnums['total_gib']} GiB, "
          f"need {dnums['need_gib']} GiB per day")
    print(json.dumps({"n_rows": len(rows), "n_would_fail": len(fails),
                      "n_not_yet": len(notyet),
                      "would_fail": [f"{d}:{s}" for d, _, s in fails]}))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""DA -- THE DATA-ROOT RESOLVER PROBE (BE 59's finding, driven).

BE 59 found that every seat worktree's `data/` is a MATERIALISED directory
-- git owns the tracked artifacts under it -- so the ledger's UNTRACKED
files (the 09-04 tape) are invisible under a worktree path, and a run there
is correct only because `PM_DATA_ROOT` was passed explicitly. Four
worktrees are one unset environment variable from reading a shell fact as a
ledger fact.

THIS DRIVES IT. Every resolver in the programme is run in a SUBPROCESS,
from a neutral working directory, under three environments:

  A  PM_DATA_ROOT UNSET
  B  PM_DATA_ROOT set to a WORKTREE path
  C  PM_DATA_ROOT set to the LEDGER            (the positive control)

and what each returns -- or refuses -- is classified against the paths
themselves:

  LEDGER            the canonical tree
  WORKTREE          a seat's worktree            <- a FINDING
  CWD               the process's cwd            <- a FINDING
  REFUSED           an exception, named          <- what a GATE must do

A GATE (`require_canonical`, `require_ledger`) must REFUSE in case B. A
plain RESOLVER is allowed to return a path -- its callers carry the gate --
but it must never answer case A with a worktree or a cwd.

R-235: every resolver is exercised through its own public entry point, in
its own process. Nothing is re-implemented here and no module under test is
imported into this one.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROTOCOL = "P003_DA_RESOLVER_PROBE_V1"
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
LEDGER = Path("/home/yuqing/ctaNew")
WORKTREE_HINTS = ("-wt-", "/worktrees/", "ctaNew-wt")

#: id -> (kind, the one-liner that exercises its PUBLIC entry point)
#: A child prints exactly `OK <path>` or raises. Nothing else is accepted.
RESOLVERS = {
    "pm_tape_density._resolve_data_root": (
        "resolver",
        "import pm_tape_density as M; print('OK', M._resolve_data_root())"),
    "de_data_root.resolve": (
        "resolver",
        "import de_data_root as M; print('OK', M.resolve()['data_root'])"),
    "de_data_root.require_canonical": (
        "gate",
        "import de_data_root as M; "
        "print('OK', M.require_canonical('DA 80 probe')['data_root'])"),
    "be_data_root.resolve": (
        "resolver",
        "import be_data_root as M; print('OK', M.resolve()['data_root'])"),
    "be_data_root.require_ledger": (
        "gate",
        "import be_data_root as M; "
        "print('OK', M.require_ledger()['data_root'])"),
    "be_data_root.data_root": (
        "resolver",
        "import be_data_root as M; print('OK', M.data_root())"),
    "da_accrual_report.data_root": (
        "resolver",
        "import da_accrual_report as M; print('OK', M.data_root())"),
    "da_process_budget_audit.AUDIT_ROOT": (
        "resolver",
        "import da_process_budget_audit as M; print('OK', M.AUDIT_ROOT)"),
    "da_gate1_day_verdict._derived_dir": (
        "resolver",
        "import da_gate1_day_verdict as M; print('OK', M._derived_dir())"),
    #: NOT A DATA-ROOT RESOLVER AT ALL, and saying so is the point. The race
    #: verifier reads CODE-TREE declarations (`HERE/declarations/...`) and
    #: takes every data path from the CLI or from the pins. My first probe
    #: asked it for `HERE.parents[1]` -- a path I invented for it -- and
    #: then reported the answer as a fallback FINDING against a module that
    #: resolves no data root. A probe that manufactures the thing it
    #: measures measures itself.
    "da_root.require_canonical_root": (
        "gate",
        "import da_root as M; "
        "print('OK', M.require_canonical_root('probe')['root'])"),
    "da_race_read_verify (declarations only)": (
        "not_a_data_root_resolver",
        "import da_race_read_verify as M; "
        "print('OK', M.OP_DECL.parent)"),
}


def classify_path(p: str, cwd: str) -> str:
    if not p:
        return "NO_PATH"
    q = str(Path(p).resolve()) if Path(p).exists() else str(Path(p))
    if any(h in q for h in WORKTREE_HINTS):
        return "WORKTREE"
    if q == str(Path(cwd).resolve()) or q.startswith(
            str(Path(cwd).resolve()) + os.sep):
        return "CWD"
    if q == str(LEDGER) or q.startswith(str(LEDGER) + os.sep):
        return "LEDGER"
    return "OTHER"


def probe(code: str, *, env_value: str | None, cwd: str,
          code_root: Path = LEDGER, timeout: int = 60) -> dict:
    """Run one resolver in its OWN process, from a neutral cwd."""
    env = {k: v for k, v in os.environ.items() if k != "PM_DATA_ROOT"}
    if env_value is not None:
        env["PM_DATA_ROOT"] = env_value
    env["PYTHONPATH"] = str(code_root / "live/pm_research")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "TIMEOUT", "path": None, "exception": None}
    out = (r.stdout or "").strip().splitlines()
    line = next((x for x in out if x.startswith("OK ")), None)
    if r.returncode == 0 and line:
        return {"outcome": "RETURNED", "path": line[3:].strip(),
                "exception": None}
    err = (r.stderr or "").strip().splitlines()
    exc = next((x for x in reversed(err) if ":" in x and not
                x.startswith(" ")), (err[-1] if err else ""))
    return {"outcome": "REFUSED", "path": None,
            "exception": exc[:400],
            "exception_type": exc.split(":", 1)[0].split(".")[-1].strip(),
            "returncode": r.returncode}


def make_partial_ledger(base: Path | None = None) -> str:
    """A directory that LOOKS like the ledger and holds a fraction of it."""
    d = Path(base or tempfile.mkdtemp(prefix="da80_partial_"))
    (d / "data" / "pm_5min" / "derived").mkdir(parents=True, exist_ok=True)
    (d / "data" / "pm_5min" / "derived"
     / "be_daybook_receipt_20260903_btc.json").write_text("{}")
    return str(d)


def probe_all(worktree: str, resolvers: dict | None = None,
              cwd: str | None = None, partial: str | None = None) -> dict:
    res = resolvers or RESOLVERS
    neutral = cwd or tempfile.mkdtemp(prefix="da80_neutral_")
    partial = partial or make_partial_ledger()
    rows = {}
    for rid, (kind, code) in res.items():
        cases = {}
        #: THE CODE TREE IS THE SECOND DIMENSION, and it is the one that
        #: matters. Importing every module FROM THE LEDGER made every
        #: unset-environment case resolve to the ledger -- of course it
        #: did: `CODE_ROOT` and `HERE.parents[1]` ARE the ledger there. The
        #: case BE 59 named is a run executed IN A WORKTREE with the
        #: variable unset, and a probe that never imports from a worktree
        #: cannot see it.
        for label, env_value, croot in (
                ("A_unset_ledger_code", None, LEDGER),
                ("A_unset_worktree_code", None, Path(worktree)),
                ("B_worktree_env", worktree, LEDGER),
                #: REV 57 A.6's third drive: a directory that LOOKS like a
                #: ledger -- `data/pm_5min/derived` present, holding a
                #: fraction of the artifacts. A gate must refuse it; a
                #: count taken there is a count of a smaller plausible
                #: ledger, which is the failure the addendum names.
                ("D_planted_partial_ledger", partial, LEDGER),
                #: REV 57 A.6's EXACT case: the environment is set
                #: CORRECTLY and the code runs FROM A WORKTREE. A module
                #: that resolves relative to its own file answers with the
                #: worktree's PARTIAL data/ here -- with nothing wrong in
                #: the environment at all.
                ("F_ledger_env_from_worktree_code", str(LEDGER),
                 Path(worktree)),
                ("C_ledger", str(LEDGER), LEDGER)):
            r = probe(code, env_value=env_value, cwd=neutral,
                      code_root=croot)
            r["code_root"] = str(croot)
            r["path_class"] = (classify_path(r["path"], neutral)
                               if r["outcome"] == "RETURNED" else None)
            cases[label] = r
        a = cases["A_unset_ledger_code"]
        aw = cases["A_unset_worktree_code"]
        b = cases["B_worktree_env"]
        findings = []
        if kind != "not_a_data_root_resolver":
            for lbl, row in (("UNSET", a), ("UNSET_IN_A_WORKTREE", aw)):
                if row["outcome"] == "RETURNED" and row["path_class"] in (
                        "WORKTREE", "CWD"):
                    findings.append(
                        f"{lbl}_FALLS_BACK_TO_{row['path_class']}")
        d = cases["D_planted_partial_ledger"]
        f = cases["F_ledger_env_from_worktree_code"]
        if kind == "gate" and b["outcome"] == "RETURNED" and \
                b["path_class"] != "LEDGER":
            findings.append("A_GATE_ADMITS_A_WORKTREE_ROOT")
        if kind == "gate" and d["outcome"] == "RETURNED" and \
                d["path_class"] != "LEDGER":
            findings.append("A_GATE_ADMITS_A_PARTIAL_LEDGER")
        if kind != "not_a_data_root_resolver" and \
                f["outcome"] == "RETURNED" and f["path_class"] == "WORKTREE":
            findings.append(
                "WITH_A_CORRECT_ENV_IT_STILL_ANSWERS_WITH_ITS_OWN_TREE")
        if kind == "gate" and aw["outcome"] == "RETURNED" and \
                aw["path_class"] != "LEDGER":
            findings.append("A_GATE_ADMITS_AN_UNSET_ENVIRONMENT")
        rows[rid] = {
            "kind": kind, "cases": cases, "findings": findings,
            "why_not_judged": (
                "this module resolves no data root: it reads code-tree "
                "declarations and takes data paths from its caller. Its "
                "answer is REPORTED so the absence is visible, never "
                "judged as a fallback"
                if kind == "not_a_data_root_resolver" else None),
            "n_findings": len(findings),
            "verdict": "FINDING" if findings else "OK",
            "summary": {k: (v["outcome"] if v["outcome"] != "RETURNED"
                            else v["path_class"]) for k, v in cases.items()},
        }
    return {"neutral_cwd": neutral, "worktree_probed": worktree,
            "resolvers": rows,
            "n_resolvers": len(rows),
            "n_with_findings": sum(1 for v in rows.values() if v["findings"]),
            "findings_by_resolver": {k: v["findings"]
                                     for k, v in rows.items() if v["findings"]}
            }


def probe_identity() -> dict:
    src = Path(__file__).resolve()
    d = subprocess.run(["git", "-C", str(LEDGER), "status", "--porcelain",
                        "--", str(src)], capture_output=True, text=True)
    h = subprocess.run(["git", "-C", str(LEDGER), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return {"path": "live/pm_research/da_resolver_probe.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "tree_head": h.stdout.strip() or None,
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


# ------------------------------------------------------------- falsifiers

FALLS_BACK_TO_CWD = '''
import os
from pathlib import Path


def data_root():
    return Path(os.environ.get("PM_DATA_ROOT") or os.getcwd())
'''

FALLS_BACK_TO_ITS_OWN_TREE = '''
import os
from pathlib import Path


def data_root():
    return Path(os.environ.get("PM_DATA_ROOT")
                or Path(__file__).resolve().parents[1])
'''

#: THE EXACT SHAPE REV 58 section 4 DISSECTED. The try-branch cannot
#: return -- `resolve()` hands back a DICT and `Path(<dict>)` raises
#: TypeError -- and the BARE except swallows it, so the tree-relative
#: fallback runs on EVERY call while the docstring claims the shared
#: resolver. From the shared tree the fallback gives the right answer, so
#: only a drive FROM A WORKTREE separates the two branches.
DEAD_TRY_BRANCH = '''
from pathlib import Path


class _FakeShared:
    @staticmethod
    def resolve():
        return {"data_root": "/home/yuqing/ctaNew/data"}


def data_root():
    """Through the programme's ONE data-root resolver."""
    try:
        return Path(_FakeShared.resolve()) / "pm_5min"
    except Exception:                                         # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "pm_5min"
'''

REFUSES = '''
import os
from pathlib import Path

LEDGER = "/home/yuqing/ctaNew"


class RootRefused(RuntimeError):
    pass


def data_root():
    env = os.environ.get("PM_DATA_ROOT")
    if not env:
        raise RootRefused("REFUSED: PM_DATA_ROOT is unset")
    if str(Path(env).resolve()) != LEDGER:
        raise RootRefused("REFUSED: not the ledger: " + env)
    return Path(env)
'''


def selftest() -> tuple:
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond), "detail": detail})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    tmp = Path(tempfile.mkdtemp(prefix="da80_probe_",
                                dir=os.environ.get("DA_SCRATCH")
                                or tempfile.gettempdir()))
    #: THE PLANTED MODULES LIVE IN A FAKE WORKTREE, in the real layout
    #: (`<tree>/live/pm_research/`), because the own-tree faller answers
    #: with `parents[1]` -- and in a flat scratch directory that is a
    #: nondescript temp path, which the classifier rightly calls OTHER. A
    #: fixture in the wrong shape tests the wrong thing.
    fake_wt = tmp / "ctaNew-wt-fake"
    pkg = fake_wt / "live" / "pm_research"
    pkg.mkdir(parents=True)
    (pkg / "planted_cwd.py").write_text(FALLS_BACK_TO_CWD)
    (pkg / "planted_own_tree.py").write_text(FALLS_BACK_TO_ITS_OWN_TREE)
    (pkg / "planted_refuses.py").write_text(REFUSES)
    (pkg / "planted_dead_try.py").write_text(DEAD_TRY_BRANCH)
    planted = {
        "planted.falls_back_to_cwd": (
            "resolver",
            "import planted_cwd as M; print('OK', M.data_root())"),
        "planted.falls_back_to_its_own_tree": (
            "resolver",
            "import planted_own_tree as M; print('OK', M.data_root())"),
        "planted.refuses": (
            "gate",
            "import planted_refuses as M; print('OK', M.data_root())"),
        "planted.dead_try_branch": (
            "resolver",
            "import planted_dead_try as M; print('OK', M.data_root())"),
    }
    neutral = tmp / "neutral"
    neutral.mkdir()

    def run_planted(rid):
        kind, code = planted[rid]
        cases = {}
        for label, env_value in (("A_unset", None),
                                 ("B_worktree", str(fake_wt)),
                                 ("C_ledger", str(LEDGER))):
            env = {k: v for k, v in os.environ.items()
                   if k != "PM_DATA_ROOT"}
            if env_value is not None:
                env["PM_DATA_ROOT"] = env_value
            env["PYTHONPATH"] = str(pkg)
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True,
                               cwd=str(neutral), env=env, timeout=60)
            out = [x for x in (r.stdout or "").splitlines()
                   if x.startswith("OK ")]
            if r.returncode == 0 and out:
                cases[label] = {"outcome": "RETURNED",
                                "path": out[0][3:].strip()}
                cases[label]["path_class"] = classify_path(
                    cases[label]["path"], str(neutral))
            else:
                cases[label] = {"outcome": "REFUSED", "path": None,
                                "path_class": None,
                                "exception": (r.stderr or "").strip()[-120:]}
        return cases

    c_cwd = run_planted("planted.falls_back_to_cwd")
    c_own = run_planted("planted.falls_back_to_its_own_tree")
    c_ref = run_planted("planted.refuses")
    ck("A PLANTED RESOLVER THAT FALLS BACK TO CWD IS SEEN AS CWD, AND ONE "
       "THAT FALLS BACK TO ITS OWN TREE IS SEEN AS A WORKTREE -- with the "
       "environment UNSET. ***That is BE 59's finding in one line: four "
       "worktrees are one unset variable from reading a shell fact as a "
       "ledger fact***",
       c_cwd["A_unset"]["path_class"] == "CWD"
       and c_own["A_unset"]["path_class"] == "WORKTREE"
       and c_cwd["C_ledger"]["path_class"] == "LEDGER",
       f"cwd-faller unset -> {c_cwd['A_unset']['path_class']}; own-tree "
       f"faller unset -> {c_own['A_unset']['path_class']}; both follow the "
       f"env when it IS set -> {c_cwd['C_ledger']['path_class']}")
    ck("AND A RESOLVER THAT REFUSES IS SEEN TO REFUSE, IN BOTH DIRECTIONS: "
       "unset REFUSES, a worktree root REFUSES, the ledger ADMITS. ***A "
       "probe that could only report paths would call a correct gate and a "
       "silent fallback the same thing***",
       c_ref["A_unset"]["outcome"] == "REFUSED"
       and c_ref["B_worktree"]["outcome"] == "REFUSED"
       and c_ref["C_ledger"]["outcome"] == "RETURNED"
       and c_ref["C_ledger"]["path_class"] == "LEDGER",
       f"unset -> {c_ref['A_unset']['outcome']}; worktree -> "
       f"{c_ref['B_worktree']['outcome']}; ledger -> "
       f"{c_ref['C_ledger']['path_class']}")
    c_dead = run_planted("planted.dead_try_branch")
    ck("REV 58 section 4 (test 2) -- THE DEAD TRY-BRANCH IS CAUGHT, AND "
       "ONLY FROM A WORKTREE. The planted module has the exact shape: a "
       "try that CANNOT return (`Path(<dict>)` raises TypeError), a BARE "
       "except, and a tree-relative fallback -- with a docstring claiming "
       "the shared resolver. ***Driven with the environment set CORRECTLY "
       "it still answers with its own tree, which is the only signal there "
       "is: from the shared tree the fallback gives the right answer and "
       "no test run there can see it***",
       c_dead["C_ledger"]["path_class"] == "WORKTREE"
       and c_dead["A_unset"]["path_class"] == "WORKTREE",
       f"env set to the LEDGER -> {c_dead['C_ledger']['path_class']} "
       f"({c_dead['C_ledger']['path']}); unset -> "
       f"{c_dead['A_unset']['path_class']}")

    ck("THE CLASSIFIER SEPARATES THE FOUR ANSWERS BY THE PATH ITSELF, not "
       "by what a module says about itself",
       classify_path("/home/yuqing/ctaNew", "/tmp/x") == "LEDGER"
       and classify_path("/home/yuqing/ctaNew/data/pm_5min",
                         "/tmp/x") == "LEDGER"
       and classify_path("/home/yuqing/ctaNew-wt-da", "/tmp/x") == "WORKTREE"
       and classify_path("/tmp/x", "/tmp/x") == "CWD"
       and classify_path("/srv/elsewhere", "/tmp/x") == "OTHER",
       "ledger, ledger-subpath, worktree, cwd and other all separate")

    #: REV 60 / the coordinator: CANONICAL IS THE LEDGER'S REAL PATH.
    import da_root as _DR                                     # noqa: PLC0415
    _fake = tmp / "materialised-wt"
    (_fake / "data" / "pm_5min" / "derived").mkdir(parents=True,
                                                   exist_ok=True)
    _cases = {}
    for _lbl, _root in (("the shared tree", str(LEDGER)),
                        ("wt-da, data/ symlinked",
                         str(Path(__file__).resolve().parents[2])),
                        ("a MATERIALISED worktree", str(_fake))):
        try:
            _b = _DR.require_canonical_root("probe", root=Path(_root))
            _cases[_lbl] = ("ADMITS", _b["data_root_real_path"])
        except _DR.RootRefused as _e:
            _cases[_lbl] = ("REFUSED", str(_e)[:60])
    ck("CANONICAL IS DECIDED BY THE LEDGER'S REAL PATH, NOT BY THE TREE'S "
       "NAME: the shared tree ADMITS, a worktree whose `data/` SYMLINKS to "
       "the ledger ADMITS -- its every data byte IS the ledger's -- and a "
       "worktree with a MATERIALISED `data/` REFUSES, because that one "
       "holds only the TRACKED artifacts and a count taken there is a "
       "count of a smaller, plausible ledger. ***Their paths are the same "
       "shape; only `readlink -f` separates them***",
       _cases["the shared tree"][0] == "ADMITS"
       and _cases["wt-da, data/ symlinked"][0] == "ADMITS"
       and _cases["a MATERIALISED worktree"][0] == "REFUSED"
       and _cases["wt-da, data/ symlinked"][1] == "/home/yuqing/ctaNew/data",
       "; ".join(f"{k} -> {v[0]}" for k, v in _cases.items()))
    ck("AND CODE CANONICALITY STAYS STRICT, because the symlink says "
       "nothing about `live/`: a worktree's `data/` may BE the ledger's "
       "directory while its checkout is at another commit. ***DATA is "
       "canonical by where it LANDS; CODE is canonical by WHICH TREE it "
       "is*** -- and reading another seat's source from a stale checkout "
       "is the finding DA 77 shipped and had to retract",
       str(_DR.code_root("probe")) == "/home/yuqing/ctaNew"
       and (_DR.code_root("probe") / "live" / "pm_research").is_dir(),
       f"code_root -> {_DR.code_root('probe')} regardless of which tree "
       f"asks")

    real = probe_all(str(Path(__file__).resolve().parents[2]))
    ck("EVERY RESOLVER IN THE PROGRAMME IS DRIVEN THROUGH ITS OWN PUBLIC "
       "ENTRY POINT, IN ITS OWN PROCESS, FROM A NEUTRAL DIRECTORY -- three "
       "environments each, and the LEDGER case is the positive control",
       real["n_resolvers"] >= 9
       and all(set(v["cases"]) == {"A_unset_ledger_code",
                                   "A_unset_worktree_code",
                                   "B_worktree_env",
                                   "D_planted_partial_ledger",
                                   "F_ledger_env_from_worktree_code",
                                   "C_ledger"}
               for v in real["resolvers"].values())
       and all(v["cases"]["C_ledger"]["outcome"] == "RETURNED"
               and v["cases"]["C_ledger"]["path_class"] == "LEDGER"
               for v in real["resolvers"].values()
               if v["cases"]["C_ledger"]["outcome"] == "RETURNED"),
       f"{real['n_resolvers']} resolvers x 3 environments; "
       f"{real['n_with_findings']} with findings: "
       f"{sorted(real['findings_by_resolver'])}")
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def build_report(worktree: str | None = None) -> dict:
    wt = worktree or str(Path(__file__).resolve().parents[2])
    r = probe_all(wt)
    return {
        "protocol": PROTOCOL,
        "as_of_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "why": ("BE 59: every seat worktree's data/ is a MATERIALISED "
                "directory, so the ledger's UNTRACKED artifacts are "
                "invisible under a worktree path and a run there is correct "
                "only because PM_DATA_ROOT was passed. This drives every "
                "resolver with the variable UNSET and with it pointing at a "
                "worktree"),
        "ledger": str(LEDGER),
        "probe_identity": probe_identity(),
        **r,
        "decides_nothing": ("each finding belongs to the seat that owns the "
                            "resolver; none is fixed here"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--worktree", default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            rep = build_report(a.worktree)
            rep["checks"] = checks
            rep["n_checks"] = len(checks)
            rep["n_failed"] = n_fail
            rep["both_directions"] = True
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        return 1 if n_fail else 0
    if a.probe:
        rep = build_report(a.worktree)
        if a.output:
            a.output.write_text(json.dumps(rep, indent=2, sort_keys=True,
                                           default=str) + "\n")
        for rid, v in rep["resolvers"].items():
            print(f"{v['verdict']:8} {rid:44} {v['summary']}")
        return 0
    ap.error("--selftest or --probe [--worktree <path>] [--output <path>]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

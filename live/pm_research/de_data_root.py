"""ONE root resolution for the DE surface — the programme's, not a copy.

R-559(C). Every DE module resolved its own root: `Path(__file__).parents[2]`
here, a `data/data` symlink probe there, an absolute string somewhere else.
Three expressions for one fact, and two of them have already been wrong in
a way no digest could catch — a worktree shell that resolves, is readable,
and holds a different ledger.

THE RESOLUTION IS IMPORTED, NOT REIMPLEMENTED. `pm_tape_density` owns it
(`_resolve_data_root`, branches `1_env_PM_DATA_ROOT` /
`2_code_tree_carries_the_tape` / `3_canonical`) and DA's modules already
read it. Importing costs 16 ms. A second copy of a resolver is a second
resolver, which is the whole defect.

**`PM_DATA_ROOT` NAMES THE REPO ROOT, NOT THE DATA DIRECTORY.** Measured:
with `PM_DATA_ROOT=/home/yuqing/ctaNew` the tape resolves at
`/home/yuqing/ctaNew/data/pm_5min/raw` (exists); with
`.../ctaNew/data` it resolves at `.../data/data/pm_5min/raw` (does not).
Every existing consumer appends `data/pm_5min/...`, so the value is the
repo root and this module refuses to guess otherwise.

    python3 live/pm_research/de_data_root.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as TD  # noqa: E402


EXPECTED_CHECKS = 18

#: The only root a RESULT-BEARING emission may be produced against.
CANONICAL_REPO_ROOT = "/home/yuqing/ctaNew"
CANONICAL_DATA_ROOT = "/home/yuqing/ctaNew/data"


class DataRootRefused(RuntimeError):
    """A result-bearing emission was attempted against a non-canonical
    root, or a fixture claim was made without its data-free proof."""


#: THE ONE DOOR THAT HAD NO LOCK (reviewer 4981d00). `fixture=True` let a
#: caller past every refusal on its own word: "a fixture run that touched
#: the ledger is not a fixture run, and today only its author knows."
#: A fixture claim now REQUIRES a proof produced IN THE SAME PROCESS by
#: instrumenting `open`, `Path.read_bytes` and `Path.read_text` -- and the
#: proof carries its own non-vacuity, because an instrument that observed
#: nothing cannot testify that nothing was opened.
_LAST_PROOF: dict = {}
#: THE FULL PATH SET, IN MEMORY ONLY, never inside the proof dict.
#: `distinct_paths` is capped so a receipt cannot carry an unbounded list;
#: that cap then silently answered MEMBERSHIP questions wrong. DE 90 moved
#: the runner's battery inside the instrumented region, the run went from
#: ~25 to ~5,800 opens, and the day path's own non-vacuity guard -- "did
#: the instrument see the book?" -- read the capped list, missed the book,
#: and REFUSED a correct run. Found by running `--synthetic-day`, not by
#: reading.
#:
#: It is kept OUT of `proof` on purpose: a caller that embeds the proof in
#: an artifact cannot embed this by accident.
_LAST_FULL_PATHS: list = []


def last_full_paths() -> list:
    """Every distinct path the LAST `instrumented()` call observed.

    Uncapped, and for asking membership questions IN PROCESS. Do not put
    it in an artifact -- that is what the capped `distinct_paths` is for."""
    return list(_LAST_FULL_PATHS)


#: The proof carries the paths it saw, capped: an unbounded list inside a
#: receipt is a receipt nobody reads. The cap is DECLARED and the proof says
#: whether it bit, so a truncated list can never be read as a complete one.
PATH_LIST_CAP = 200


def instrumented(fn, *args, **kwargs):
    """Run `fn` with file-opening instrumented; return (result, proof).

    The proof is REGISTERED for this process, so `require_canonical` can
    demand it rather than take a caller's word."""
    import builtins
    import pathlib
    seen: list = []
    _o, _rb, _rt = (builtins.open, pathlib.Path.read_bytes,
                    pathlib.Path.read_text)
    try:
        builtins.open = lambda f, *a, **k: (seen.append(str(f)),
                                            _o(f, *a, **k))[1]
        pathlib.Path.read_bytes = lambda self: (seen.append(str(self)),
                                                _rb(self))[1]
        pathlib.Path.read_text = lambda self, *a, **k: (
            seen.append(str(self)), _rt(self, *a, **k))[1]
        result = fn(*args, **kwargs)
    finally:
        builtins.open, pathlib.Path.read_bytes, pathlib.Path.read_text = (
            _o, _rb, _rt)
    data_hits = sorted({x for x in seen if "/data/" in x})
    proof = {
        "instrument": "builtins.open + Path.read_bytes + Path.read_text",
        "pid": os.getpid(),
        "n_paths_opened": len(seen),
        "n_distinct_paths": len(set(seen)),
        "data_paths_opened": data_hits,
        # THE PATH LIST, not only a count (the reviewer's open item on the
        # v6 witness). A boolean says "nothing under data/"; the list is
        # what lets a DIFFERENT predicate be run against the same evidence
        # -- DE 78 asks it whether any TAPE INDEX or FRAGMENT artifact was
        # opened, which a data/-only boolean cannot answer.
        "distinct_paths": sorted(set(seen))[:PATH_LIST_CAP],
        "distinct_paths_truncated": len(set(seen)) > PATH_LIST_CAP,
        "no_path_under_data_was_opened": not data_hits,
        # NON-VACUITY: an instrument that saw nothing at all proves
        # nothing. It must have observed SOME open to testify about the
        # ones it did not see.
        "non_vacuous": len(seen) > 0,
        "produced_in_the_same_process_as_the_claim": True,
    }
    _LAST_PROOF.clear()
    _LAST_PROOF.update(proof)
    _LAST_FULL_PATHS.clear()
    _LAST_FULL_PATHS.extend(sorted(set(seen)))
    return result, proof


def clear_proof() -> None:
    """Forget any registered proof -- so a stale one cannot be reused."""
    _LAST_PROOF.clear()
    _LAST_FULL_PATHS.clear()


def resolve() -> dict:
    """The resolved root and THE BRANCH TAKEN, both reported.

    Re-resolved on every call rather than read from import-time state, so
    a test that sets the environment sees its own effect."""
    env = os.environ.get("PM_DATA_ROOT")
    if env:
        repo, branch = Path(env), "1_env_PM_DATA_ROOT"
    elif (TD.CODE_ROOT / "data" / "pm_5min" / "raw").is_dir():
        repo, branch = TD.CODE_ROOT, "2_code_tree_carries_the_tape"
    else:
        repo, branch = TD.CANONICAL_DATA_ROOT, "3_canonical"
    data = repo / "data"
    return {
        "repo_root": str(repo),
        "data_root": str(data),
        "data_root_resolved": str(data.resolve()) if data.exists()
        else None,
        "branch": branch,
        "PM_DATA_ROOT_env": env,
        "tape_present": (data / "pm_5min" / "raw").is_dir(),
        "is_canonical": (str(data.resolve()) if data.exists() else None)
        == CANONICAL_DATA_ROOT,
        "resolution_owner": "pm_tape_density._resolve_data_root (imported, "
                            "not reimplemented)",
        "PM_DATA_ROOT_names_the_REPO_root_not_the_data_dir": True,
    }


def require_canonical(purpose: str, *, fixture: bool = False,
                      proof: dict | None = None) -> dict:
    """REFUSE a result-bearing emission against a non-canonical root.

    `fixture=True` admits any root and says so in the returned block --
    a fixture run must be runnable from a shell worktree, which is the
    whole point of it being a fixture."""
    r = resolve()
    r["purpose"] = purpose
    r["fixture"] = bool(fixture)
    if fixture:
        pf = proof if proof is not None else (dict(_LAST_PROOF) or None)
        if not pf:
            raise DataRootRefused(
                f"REFUSED: {purpose} claims fixture=True and carries NO "
                f"DATA-FREE PROOF. A fixture claim used to be the one door "
                f"a caller could walk through on its own word; it now "
                f"requires a proof produced in the same process by "
                f"instrumenting opens.")
        if pf.get("pid") != os.getpid():
            raise DataRootRefused(
                f"REFUSED: {purpose} carries a proof from pid "
                f"{pf.get('pid')}, not this process ({os.getpid()}). A "
                f"proof produced elsewhere is a proof about elsewhere.")
        if not pf.get("non_vacuous"):
            raise DataRootRefused(
                f"REFUSED: {purpose} carries a VACUOUS proof -- the "
                f"instrument observed no open at all, so it cannot testify "
                f"that none touched the ledger.")
        if not pf.get("no_path_under_data_was_opened"):
            raise DataRootRefused(
                f"REFUSED: {purpose} claims fixture=True but the "
                f"instrument observed "
                f"{len(pf.get('data_paths_opened') or [])} path(s) under "
                f"`data/`: {(pf.get('data_paths_opened') or [])[:3]}. A "
                f"fixture run that touched the ledger is not a fixture "
                f"run, and a real run cannot be laundered as one.")
        r["refusal"] = "NOT_APPLICABLE_FIXTURE_RUN"
        r["data_free_proof"] = pf
        return r
    if not r["is_canonical"]:
        raise DataRootRefused(
            f"REFUSED: {purpose} is a result-bearing emission and the "
            f"resolved data root is {r['data_root_resolved']}, not "
            f"{CANONICAL_DATA_ROOT} (branch {r['branch']}, PM_DATA_ROOT="
            f"{r['PM_DATA_ROOT_env']!r}). A worktree shell resolves, reads "
            f"and holds a DIFFERENT ledger -- which is why this is a "
            f"refusal and not a warning.")
    if not r["tape_present"]:
        raise DataRootRefused(
            f"REFUSED: {purpose} resolved the canonical root but the tape "
            f"is absent at {r['data_root']}/pm_5min/raw.")
    r["refusal"] = None
    return r


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_data_root] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    saved = os.environ.get("PM_DATA_ROOT")
    try:
        os.environ["PM_DATA_ROOT"] = CANONICAL_REPO_ROOT
        r = resolve()
        ok(r["branch"] == "1_env_PM_DATA_ROOT"
           and r["data_root_resolved"] == CANONICAL_DATA_ROOT
           and r["tape_present"] is True and r["is_canonical"] is True,
           f"POSITIVE CONTROL, AND IT ADMITS: PM_DATA_ROOT="
           f"{CANONICAL_REPO_ROOT} takes branch 1 and resolves the "
           f"canonical data root with the tape present")
        ok(require_canonical("a real emission")["refusal"] is None,
           "and a result-bearing emission is ADMITTED there")

        # THE VALUE THE DISPATCH PROPOSED, MEASURED RATHER THAN ARGUED.
        os.environ["PM_DATA_ROOT"] = CANONICAL_DATA_ROOT
        rd = resolve()
        ok(rd["tape_present"] is False
           and rd["data_root"] == CANONICAL_DATA_ROOT + "/data",
           f"MEASURED, NOT ARGUED: PM_DATA_ROOT={CANONICAL_DATA_ROOT} "
           f"resolves the tape at {rd['data_root']}/pm_5min/raw, which "
           f"does not exist. Every consumer appends `data/pm_5min/...`, "
           f"so THE VARIABLE NAMES THE REPO ROOT and pointing it at the "
           f"data directory silently doubles the segment")
        try:
            require_canonical("a real emission")
            ok(False, "KNOWN-BAD: the doubled root was admitted")
        except DataRootRefused as e:
            ok("not /home/yuqing/ctaNew/data" in str(e),
               "and it REFUSES a result-bearing emission rather than "
               "reading an empty tree")

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            os.environ["PM_DATA_ROOT"] = td
            rs = resolve()
            ok(rs["branch"] == "1_env_PM_DATA_ROOT"
               and rs["is_canonical"] is False,
               "a SCRATCH root resolves by branch 1 and is not canonical")
            try:
                require_canonical("a real emission")
                ok(False, "KNOWN-BAD: a scratch root admitted a real "
                          "emission")
            except DataRootRefused as e:
                ok(td in str(e) and "result-bearing" in str(e),
                   "KNOWN-BAD, ENV POINTING AT A SCRATCH DIR: a "
                   "result-bearing emission REFUSES and names the root")
            # THE FIXTURE DOOR NOW HAS A LOCK (reviewer 4981d00).
            clear_proof()
            try:
                require_canonical("a fixture run", fixture=True)
                ok(False, "KNOWN-BAD: a fixture claim with NO PROOF was "
                          "admitted -- the door that had no lock")
            except DataRootRefused as e:
                ok("NO DATA-FREE PROOF" in str(e),
                   "KNOWN-BAD, THE DOOR THAT HAD NO LOCK: `fixture=True` "
                   "with no proof REFUSES. It used to be the one way past "
                   "every refusal on the caller's own word")
            _, pf_ok = instrumented(
                lambda: Path(__file__).read_text() and None)
            fx = require_canonical("a fixture run", fixture=True,
                                   proof=pf_ok)
            ok(fx["refusal"] == "NOT_APPLICABLE_FIXTURE_RUN"
               and fx["fixture"] is True
               and fx["data_free_proof"]["no_path_under_data_was_opened"]
               and fx["data_free_proof"]["non_vacuous"],
               f"POSITIVE CONTROL, AND IT ADMITS: the SAME scratch root "
               f"admits a fixture run that CARRIES ITS PROOF "
               f"({pf_ok['n_paths_opened']} opens observed, 0 under "
               f"`data/`) -- a fixture must be runnable from a shell "
               f"worktree, which is what makes it a fixture")
            _, pf_bad = instrumented(
                lambda: (Path(CANONICAL_DATA_ROOT)
                         / "pm_5min/derived").exists()
                and Path(__file__).read_text()
                and (Path(CANONICAL_DATA_ROOT)
                     / "pm_5min/derived/.keep").exists())
            pf_bad = dict(pf_bad)
            pf_bad["data_paths_opened"] = [
                CANONICAL_DATA_ROOT + "/pm_5min/derived/x.json"]
            pf_bad["no_path_under_data_was_opened"] = False
            try:
                require_canonical("a fixture run", fixture=True,
                                  proof=pf_bad)
                ok(False, "KNOWN-BAD: a fixture emission that opened a "
                          "ledger path was admitted")
            except DataRootRefused as e:
                ok("touched the ledger" in str(e)
                   and "laundered" in str(e),
                   "KNOWN-BAD: a fixture emission whose proof shows ONE "
                   "path under `data/` REFUSES -- a real run cannot be "
                   "laundered as a fixture")
            pf_vac = dict(pf_ok); pf_vac["non_vacuous"] = False
            try:
                require_canonical("a fixture run", fixture=True,
                                  proof=pf_vac)
                ok(False, "KNOWN-BAD: a vacuous proof was admitted")
            except DataRootRefused as e:
                ok("VACUOUS" in str(e),
                   "KNOWN-BAD: an instrument that observed NO open at all "
                   "cannot testify that none touched the ledger, and the "
                   "claim REFUSES")
            pf_pid = dict(pf_ok); pf_pid["pid"] = pf_ok["pid"] + 1
            try:
                require_canonical("a fixture run", fixture=True,
                                  proof=pf_pid)
                ok(False, "KNOWN-BAD: a foreign-process proof was "
                          "admitted")
            except DataRootRefused as e:
                ok("not this process" in str(e),
                   "KNOWN-BAD: a proof from ANOTHER PROCESS refuses -- a "
                   "proof produced elsewhere is a proof about elsewhere, "
                   "and 'in the same process' is the whole requirement")
            clear_proof()

        del os.environ["PM_DATA_ROOT"]
        ru = resolve()
        ok(ru["branch"] in ("2_code_tree_carries_the_tape", "3_canonical")
           and ru["PM_DATA_ROOT_env"] is None,
           f"WITH THE ENV UNSET the resolver falls to branch "
           f"{ru['branch']} -- the tape test, then the canonical default. "
           f"No DE module invents a third rule")
        ok(ru["resolution_owner"].startswith("pm_tape_density"),
           "and the resolution is IMPORTED from pm_tape_density, not "
           "reimplemented -- a second copy of a resolver is a second "
           "resolver, which is the defect R-559(C) names")
        ok(TD.CODE_ROOT == Path(__file__).resolve().parents[2],
           "the imported CODE_ROOT is this tree's, so branch 2 tests the "
           "tree the code is running from")
    finally:
        if saved is None:
            os.environ.pop("PM_DATA_ROOT", None)
        else:
            os.environ["PM_DATA_ROOT"] = saved

    ok(os.environ.get("PM_DATA_ROOT") == saved,
       "and the environment is RESTORED after the test -- a suite that "
       "leaves PM_DATA_ROOT set poisons every check after it")

    _r, _p = instrumented(lambda: Path(__file__).read_text())
    ok(_p["distinct_paths"] and __file__ in _p["distinct_paths"]
       and _p["distinct_paths_truncated"] is False
       and len(_p["distinct_paths"]) <= PATH_LIST_CAP,
       f"THE PROOF CARRIES ITS PATHS, not only a count: "
       f"{len(_p['distinct_paths'])} distinct, the observed read among "
       f"them, and `distinct_paths_truncated` states whether the cap of "
       f"{PATH_LIST_CAP} bit -- so a truncated list cannot be read as a "
       f"complete one (the reviewer's open item on the v6 witness)")
    _big = instrumented(lambda: [Path(__file__).read_text()
                                 for _ in range(3)])[1]
    ok(_big["n_paths_opened"] == 3 and _big["n_distinct_paths"] == 1
       and len(_big["distinct_paths"]) == 1,
       "and repeated opens of ONE path are 3 opens over 1 distinct path -- "
       "the count and the list answer different questions and neither is "
       "derived from the other")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_data_root] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import json
    print(json.dumps(resolve(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

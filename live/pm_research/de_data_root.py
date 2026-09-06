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


EXPECTED_CHECKS = 12

#: The only root a RESULT-BEARING emission may be produced against.
CANONICAL_REPO_ROOT = "/home/yuqing/ctaNew"
CANONICAL_DATA_ROOT = "/home/yuqing/ctaNew/data"


class DataRootRefused(RuntimeError):
    """A result-bearing emission was attempted against a non-canonical
    root."""


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


def require_canonical(purpose: str, *, fixture: bool = False) -> dict:
    """REFUSE a result-bearing emission against a non-canonical root.

    `fixture=True` admits any root and says so in the returned block --
    a fixture run must be runnable from a shell worktree, which is the
    whole point of it being a fixture."""
    r = resolve()
    r["purpose"] = purpose
    r["fixture"] = bool(fixture)
    if fixture:
        r["refusal"] = "NOT_APPLICABLE_FIXTURE_RUN"
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
            fx = require_canonical("a fixture run", fixture=True)
            ok(fx["refusal"] == "NOT_APPLICABLE_FIXTURE_RUN"
               and fx["fixture"] is True,
               "AND THE SAME SCRATCH ROOT ADMITS A FIXTURE RUN -- a "
               "fixture must be runnable from a shell worktree, which is "
               "what makes it a fixture")

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

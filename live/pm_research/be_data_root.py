"""ONE ROOT RESOLUTION FOR EVERY BE MODULE, WITH THE SAME PRECEDENCE AS DA's.

R-559(C). `pm_tape_density._resolve_data_root()` already resolves
env -> code-tree-carrying-the-tape -> canonical, and DA's modules take their
root from it. BE modules each computed `parents[1]` or `parents[2]` on their
own, so a BE receipt could not say WHICH tree it read and two BE modules in
one run could disagree. This is that resolution, once.

`PM_DATA_ROOT` IS A REPO ROOT, NOT A DATA DIRECTORY -- MEASURED, BECAUSE THE
DISTINCTION IS ONE CHARACTER OF PATH AND A WHOLE MISSING TAPE. Every existing
consumer appends `data/` to it:

    pm_tape_density : CANONICAL_DATA_ROOT = Path("/home/yuqing/ctaNew")
                      RAW = DATA_ROOT / "data/pm_5min/raw"
    build_state_tape_v2:206 / phase2_arms:41
                    : PM_DATA_ROOT = Path("/home/yuqing/ctaNew")
                      pm = PM_DATA_ROOT / "data/pm_5min"
    da_forward_day_verify:2168
                    : env=dict(os.environ, PM_DATA_ROOT=str(tape_root))

So `PM_DATA_ROOT=/home/yuqing/ctaNew/data` resolves the tape to
`/home/yuqing/ctaNew/data/data/pm_5min/raw`, WHICH DOES NOT EXIST, and the
gap ledger disappears with it. Measured both ways in the selftest.

AND THE FAILURE IS SILENT WHERE IT MATTERS MOST. `da_blackout_mask.py:1242`
accepts branch `1_env_PM_DATA_ROOT` OR `3_canonical` -- a wrong env value
takes branch 1 and PASSES that check while pointing at nothing. A branch name
says HOW the root was chosen; it cannot say whether the root is right. That
is why `require_ledger` checks the resolved PATH and not the branch.

THIS MODULE THEREFORE EXPOSES BOTH:
  * `repo_root`  -- what PM_DATA_ROOT means, and what consumers append to
  * `data_root`  -- `repo_root / "data"`, the thing a result-bearing emission
                    must be able to name as the ledger
"""
from __future__ import annotations

import os
from pathlib import Path

CANONICAL_REPO_ROOT = Path("/home/yuqing/ctaNew")
LEDGER_DATA_ROOT = CANONICAL_REPO_ROOT / "data"
ENV_VAR = "PM_DATA_ROOT"

BRANCHES = ("1_env_PM_DATA_ROOT", "2_code_tree_carries_the_tape",
            "3_canonical")


class DataRootRefused(RuntimeError):
    """A named refusal."""


def resolve(code_root: Path | str | None = None, *,
            env: dict | None = None) -> dict:
    """env -> code tree IF it carries the tape -> canonical.

    NOT DA's order re-stated: DA's order CALLED. This delegates to
    `pm_tape_density._resolve_data_root()` and reports the branch that
    resolver chose, so BE and DA cannot drift apart by construction.

    The middle branch tests for the TAPE (`data/pm_5min/raw`) and not for a
    directory: a worktree checks out the tracked receipts under
    `data/pm_5min/derived`, so the parent exists there while the gitignored
    tape does not. `pm_tape_density` learned that the hard way and the
    comment there says so; this is the same predicate, not a second guess."""
    # DELEGATED, NEVER COPIED. `pm_tape_density._resolve_data_root()` is the
    # resolver of record and DA's modules already take their root from it; a
    # second implementation of one precedence is two precedences. This is the
    # same rule Q-BE-271 applied to `value_ceiling`.
    import pm_tape_density as _T
    cr = (Path(code_root) if code_root is not None
          else Path(_T.CODE_ROOT))
    _saved_env = None
    _saved_cr = _T.CODE_ROOT
    try:
        if env is not None:
            _saved_env = dict(os.environ)
            os.environ.clear()
            os.environ.update({k: str(v) for k, v in env.items()})
        # branch 2 is a property of the tree the RESOLVER sits in; to ask the
        # question for another tree, the resolver is asked about that tree.
        _T.CODE_ROOT = cr
        repo = Path(_T._resolve_data_root())
        branch = _T.DATA_ROOT_BRANCH
    finally:
        _T.CODE_ROOT = _saved_cr
        if _saved_env is not None:
            os.environ.clear()
            os.environ.update(_saved_env)
    v = (env or os.environ).get(ENV_VAR)
    data = repo / "data"
    return {
        "branch": branch,
        "env_var": ENV_VAR,
        "env_value": v,
        "code_root": str(cr),
        "repo_root": str(repo),
        "data_root": str(data),
        "is_ledger": data.resolve() == LEDGER_DATA_ROOT.resolve()
                     if data.exists() else data == LEDGER_DATA_ROOT,
        "ledger_data_root": str(LEDGER_DATA_ROOT),
        "data_root_exists": data.is_dir(),
        "tape_present": (data / "pm_5min" / "raw").is_dir(),
        "PM_DATA_ROOT_is_a_REPO_root": (
            "every consumer appends `data/` to it -- pm_tape_density, "
            "build_state_tape_v2:206, phase2_arms:41, "
            "da_forward_day_verify:2168. Setting it to a DATA directory "
            "resolves the tape one level too deep."),
    }


def require_ledger(res: dict | None = None, *, fixture: bool = False,
                   why: str | None = None) -> dict:
    """A RESULT-BEARING emission must resolve to the ledger, or refuse.

    CHECKED ON THE PATH, NEVER ON THE BRANCH. `1_env_PM_DATA_ROOT` is a
    statement about how the root was chosen and says nothing about whether
    it is right -- a wrong env value takes branch 1 and would satisfy any
    branch-name check."""
    r = dict(res or resolve())
    if r["is_ledger"]:
        r["ledger_check"] = "PASS"
        return r
    if fixture:
        if not why:
            raise DataRootRefused(
                "REFUSED: fixture=True without `why`. An emission that "
                "exempts itself from the ledger must say what it is, or the "
                "exemption is invisible to a reader.")
        r["ledger_check"] = "EXEMPT_FIXTURE_OR_OFFLINE"
        r["exemption_reason"] = why
        r["NOT_RESULT_BEARING"] = True
        return r
    raise DataRootRefused(
        f"REFUSED: resolved data root {r['data_root']!r} is not the ledger "
        f"{r['ledger_data_root']!r} (branch {r['branch']}). A result-bearing "
        f"emission read from another tree is a result about another tree. "
        f"Pass fixture=True with a reason if this is deliberate."
        + ("  NOTE: PM_DATA_ROOT is a REPO root; you may have set it to the "
           "DATA directory, which resolves one level too deep."
           if r["env_value"] and Path(r["env_value"]).name == "data" else ""))


def repo_root(code_root: Path | str | None = None) -> Path:
    """What `PM_DATA_ROOT` names. Consumers append `data/` to this."""
    return Path(resolve(code_root)["repo_root"])


def data_root(code_root: Path | str | None = None) -> Path:
    """`repo_root / "data"` -- the ledger a result-bearing run must name."""
    return Path(resolve(code_root)["data_root"])


def derived(code_root: Path | str | None = None) -> Path:
    """The derived tree under the resolved root."""
    return data_root(code_root) / "pm_5min/derived"


#: The ONE file allowed to hold the canonical literal. Everything else must
#: come through the accessors above, and `audit_literals()` proves it.
LITERAL_OWNER = "be_data_root.py"


def audit_literals(pkg: Path | None = None) -> dict:
    """ZERO literal ledger paths outside this module -- COMPUTED, not claimed.

    MEM's Q-MEM-106: eleven BE modules held absolute ledger paths that no
    resolver redirects, so an env-var helper that leaves them makes a
    PARTIALLY portable seat -- the shell trap moved into the code. This is
    the grep, shipped as a check so it fires on the next one."""
    d = Path(pkg) if pkg is not None else Path(__file__).resolve().parent
    lit = str(CANONICAL_REPO_ROOT)
    hits = []
    for f in sorted(d.glob("be_*.py")):
        if f.name == LITERAL_OWNER:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if lit in line and not line.lstrip().startswith("#"):
                hits.append({"file": f.name, "line": i,
                             "text": line.strip()[:110]})
    return {"literal": lit, "owner": LITERAL_OWNER,
            "n_offending_lines": len(hits), "offenders": hits,
            "clean": not hits,
            "why": "a resolver that redirects some paths and not others is a "
                   "partially portable seat; the untouched ones decide where "
                   "the run actually reads"}


def receipt_block(res: dict | None = None, **kw) -> dict:
    """What every BE receipt carries so the tree it read is never a guess."""
    r = require_ledger(res, **kw)
    return {k: r[k] for k in (
        "branch", "env_var", "env_value", "repo_root", "data_root",
        "is_ledger", "data_root_exists", "tape_present", "ledger_check")
        if k in r} | ({"exemption_reason": r["exemption_reason"]}
                      if "exemption_reason" in r else {})


EXPECTED_CHECKS = 15


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    good = resolve(env={ENV_VAR: str(CANONICAL_REPO_ROOT)})
    ok(good["branch"] == BRANCHES[0] and good["is_ledger"]
       and good["tape_present"],
       f"POSITIVE CONTROL: PM_DATA_ROOT={CANONICAL_REPO_ROOT} takes branch 1, "
       f"resolves data_root {good['data_root']} == the ledger, and the TAPE "
       f"IS PRESENT there")
    ok(require_ledger(good)["ledger_check"] == "PASS",
       "and a result-bearing emission is permitted under it")

    # ---- THE DISPATCH'S OWN VALUE. The most valuable falsifier here. ------
    bad = resolve(env={ENV_VAR: str(LEDGER_DATA_ROOT)})
    ok(bad["branch"] == BRANCHES[0] and not bad["is_ledger"]
       and not bad["tape_present"],
       f"KNOWN-BAD, AND IT IS THE VALUE A DISPATCH ASKED FOR: "
       f"PM_DATA_ROOT={LEDGER_DATA_ROOT} still takes BRANCH 1 but resolves "
       f"data_root to {bad['data_root']} -- one level too deep, tape "
       f"ABSENT. A branch-name check would have passed this")
    try:
        require_ledger(bad)
        ok(False, "the doubled-data root must refuse")
    except DataRootRefused as e:
        ok("is not the ledger" in str(e) and "REPO root" in str(e),
           "and it REFUSES, naming the likely cause -- PM_DATA_ROOT is a "
           "REPO root, so a DATA directory resolves one level too deep")

    other = resolve(env={ENV_VAR: "/tmp/some-other-tree"})
    ok(other["branch"] == BRANCHES[0] and not other["is_ledger"],
       "KNOWN-BAD: an unrelated tree also takes branch 1 and is NOT the "
       "ledger -- which is the whole reason the check is on the PATH and "
       "never on the branch name")
    try:
        require_ledger(other)
        ok(False, "an unrelated tree must refuse")
    except DataRootRefused as e:
        ok("is not the ledger" in str(e),
           "and it REFUSES: a result read from another tree is a result "
           "about another tree")

    ok(require_ledger(other, fixture=True, why="unit fixture")[
           "ledger_check"] == "EXEMPT_FIXTURE_OR_OFFLINE",
       "a FIXTURE may opt out explicitly, and the emission is marked "
       "NOT_RESULT_BEARING rather than quietly allowed")
    try:
        require_ledger(other, fixture=True)
        ok(False, "fixture without a reason must refuse")
    except DataRootRefused as e:
        ok("without `why`" in str(e),
           "KNOWN-BAD: fixture=True WITHOUT a reason REFUSES -- an "
           "exemption a reader cannot see is not one")

    # ---- THE REASON THE EXPORT IS NOT OPTIONAL FOR THIS SEAT --------------
    noenv = resolve(env={})
    ok(noenv["branch"] == BRANCHES[1] and not noenv["is_ledger"],
       f"WITH NO ENV VAR, FROM BE's WORKTREE, THE RESOLVER PICKS THE "
       f"WORKTREE: branch {noenv['branch']}, data_root {noenv['data_root']} "
       f"-- NOT the ledger. `data/pm_5min/raw` in the worktree is a SYMLINK "
       f"to the ledger's tape, so the branch-2 predicate fires there and the "
       f"tape reads fine while `data/derived` is a DIFFERENT checkout")
    try:
        require_ledger(noenv)
        ok(False, "the worktree root must refuse for a result-bearing run")
    except DataRootRefused as e:
        ok("is not the ledger" in str(e),
           "and it REFUSES -- WHICH IS WHY PM_DATA_ROOT MUST BE EXPORTED for "
           "this seat. The env var is not a convenience here; without it a "
           "result-bearing BE emission resolves to the worktree's own "
           "derived tree")
    canon = resolve(CANONICAL_REPO_ROOT, env={})
    ok(canon["branch"] == BRANCHES[1] and canon["is_ledger"],
       f"and asked about the CANONICAL code root the SAME DELEGATED RESOLVER "
       f"lands on the ledger (branch {canon['branch']}) -- so the branch is "
       f"not the problem, the tree the code sits in is")
    import pm_tape_density as _T
    ok(resolve()["repo_root"] == str(_T._resolve_data_root()),
       "DELEGATION IS REAL: this module's repo_root EQUALS "
       "`pm_tape_density._resolve_data_root()` called directly -- one "
       "precedence, not two (the Q-BE-271 rule)")

    blk = receipt_block(good)
    ok(blk["repo_root"] == str(CANONICAL_REPO_ROOT)
       and blk["data_root"] == str(LEDGER_DATA_ROOT)
       and blk["ledger_check"] == "PASS" and blk["branch"] == BRANCHES[0],
       "and the RECEIPT BLOCK carries the root AND the branch taken, so a "
       "reader never has to guess which tree a number came from")

    # ---- MEM's Q-MEM-106 GREP, SHIPPED AS A CHECK ------------------------
    aud = audit_literals()
    ok(aud["clean"],
       f"ZERO literal {aud['literal']!r} outside {aud['owner']} across "
       f"be_*.py -- computed by grep, not claimed"
       + ("" if aud["clean"] else
          f"; OFFENDERS: {[(h['file'], h['line']) for h in aud['offenders']]}"))
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        _d = Path(_td)
        (_d / "be_planted_offender.py").write_text(
            f'REPO = Path("{CANONICAL_REPO_ROOT}")\n')
        (_d / "be_planted_comment.py").write_text(
            f'# a comment mentioning {CANONICAL_REPO_ROOT} is not a path\n')
        _bad = audit_literals(_d)
        ok(not _bad["clean"] and _bad["n_offending_lines"] == 1
           and _bad["offenders"][0]["file"] == "be_planted_offender.py",
           f"KNOWN-BAD, PLANTED: the audit FIRES on a module holding the "
           f"literal ({_bad['offenders'][0]['file']}:"
           f"{_bad['offenders'][0]['line']}) and does NOT fire on a comment "
           f"that merely mentions it -- so the clean result above is a "
           f"measurement, not a flag that cannot go red")

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
    import json
    import sys
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--show" in argv:
        print(json.dumps(resolve(), indent=1, sort_keys=True))
        return 0
    print("usage: be_data_root.py --selftest | --show")
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main())

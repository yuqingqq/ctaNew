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

#: THE FORWARD-RUN DIRECTORIES. These are NOT under the data root -- they are
#: where the scorer's sealed outputs live -- so the resolver does not cover
#: them and they would otherwise be literals in whichever module names them.
#: `be_race_read_declaration` held four; this is their declared home, in the
#: one file permitted to hold a literal.
FORWARD_RUN_ROOT = Path("/home/yuqing/ctaNew_forward_runs")
RELOCATED_RUN_ROOT = Path("/home/yuqing/.local/state/pm-co")

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


def scope_stats() -> dict:
    """The SCOPE's own memory accounting, read from its cgroup.

    `ru_maxrss` is this process's high-water mark; the SCOPE is what the
    MemoryMax cap applies to, and it accounts descendants. anon vs file
    separates what the run actually held from what the kernel cached for it,
    and `memory.events` says whether the cap was ever approached
    (`max`/`high` non-zero) rather than leaving that to be inferred from a
    peak that happened to fit."""
    try:
        leaf = open("/proc/self/cgroup").read().strip().rsplit(":", 1)[-1]
        base = Path("/sys/fs/cgroup") / leaf.lstrip("/")
    except OSError:
        return {"status": "NO_CGROUP"}
    out = {"cgroup": str(base), "unit": base.name}
    # THE INTERFACE, DECIDED (R-659(B), R-662 §3.1, REV 72 §3.1): these
    # three are EMITTED AS INTS, with the cgroup file's literal text kept in
    # a sibling `*_text`. They were strings -- the raw bytes of the cgroup
    # file -- beside int `anon_bytes`/`file_bytes` and int
    # `peak_censoring.*`, so a consumer comparing `scope.peak_bytes` with
    # `peak_censoring.cap_bytes` compared a str with an int and got False
    # for the wrong reason. BE 60 closed the one CONSUMER (the censoring
    # predicate casts) and never closed the TYPE; DA's pre-read and the
    # structure declaration read these receipts directly, so the residue was
    # an interface risk for the next reader.
    #
    # WHY INTS AND NOT A DECLARED STRING TYPE: every consumer compares these
    # numerically, so a string transmits its own hazard to each of them in
    # turn, and "declared" only means the next reader was warned. WHY THE
    # TEXT SURVIVES: `memory.max` is literally `max` when no cap is set --
    # an int cast has nothing to return there -- and the exact bytes are the
    # provenance of the reading. So: int where the file holds a number,
    # None where it does not, and the text always present beside it.
    for f, key in (("memory.peak", "peak_bytes"),
                   ("memory.current", "current_bytes"),
                   ("memory.max", "max_bytes")):
        try:
            raw = (base / f).read_text().strip()
        except OSError:
            out[key], out[f"{key}_text"] = None, None
            continue
        out[f"{key}_text"] = raw
        try:
            out[key] = int(raw)
        except (TypeError, ValueError):
            out[key] = None          # e.g. memory.max == "max" (no cap)
    out["byte_field_types"] = {
        "peak_bytes/current_bytes/max_bytes": "int, or null when the cgroup "
                                              "file holds a non-numeric "
                                              "value such as `max`",
        "*_text": "the cgroup file's literal text, kept for provenance and "
                  "for the non-numeric cases",
        "anon_bytes/file_bytes": "int (already)",
        "ruling": "R-659(B) / R-662 §3.1 / REV 72 §3.1 -- BE 66 decided the "
                  "interface: ints, with the raw text in a sibling",
    }
    for f, key in (("memory.stat", "stat"), ("memory.events", "events")):
        try:
            d = {}
            for line in (base / f).read_text().splitlines():
                k, _, v = line.partition(" ")
                d[k] = int(v) if v.strip().isdigit() else v
            if key == "stat":
                out["anon_bytes"] = d.get("anon")
                out["file_bytes"] = d.get("file")
            else:
                out["events"] = d
                out["cap_was_hit"] = bool(d.get("max", 0) or d.get("oom", 0))
        except OSError:
            out[key] = None
    # REV 63 S3: A PEAK EQUAL TO THE CAP IS A FLOOR, NOT A MEASUREMENT.
    # 09-04's tape reported peak_bytes == max_bytes with 1,199 reclaims that
    # pinned it there; 09-05's reported 7.555 GB and was never throttled.
    # Read side by side those two numbers invite "09-04 needed more", which
    # the first one cannot say: it is a bound the kernel imposed. The
    # censoring is now a FIELD, so a reader does not have to notice that two
    # numbers happen to be equal.
    try:
        # these are ints now; the casts stay so an older receipt (or a
        # non-numeric `max`) still parses rather than raising here
        _pk = int(out.get("peak_bytes") or 0)
        _mx = int(out.get("max_bytes") or 0)
        _ev = (out.get("events") or {}).get("max", 0)
        out["peak_is_censored"] = bool(_pk and _mx and _pk >= _mx)
        out["peak_censoring"] = {
            "peak_bytes": _pk, "cap_bytes": _mx, "reclaim_events_max": _ev,
            "what_this_peak_supports": (
                "demand was AT LEAST the cap and was throttled "
                f"{_ev} times; the peak is a bound, not a measurement of "
                "demand" if out["peak_is_censored"] else
                "demand peaked here and was never throttled; the peak is a "
                "measurement"),
            "not_comparable_to": ("an uncensored peak from another run -- a "
                                  "floor and a measurement are not "
                                  "commensurable, and differencing them "
                                  "reads as a demand difference that the "
                                  "numbers do not support"),
        }
    except (TypeError, ValueError):
        out["peak_is_censored"] = None
    return out


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


#: Names that mean "a place data lives". A `parents[N]` root assigned to one
#: of these is the defect BE48 §B.4 named: the audit looked for the SPELLING
#: (`/home/yuqing/ctaNew`) and not for the ACT, so it passed a module whose
#: root was wrong. A CODE root is legitimate and is not flagged.
MARKER = "be_data_root: allow-second-root"
#: EXACT names, not substrings. The first widening matched by substring and
#: caught `N_DRAWS` (via "RAW"), `OUT_NAME` (via "OUT") and `GATE1_TAPE_STEM`
#: (via "TAPE") -- 33 "offenders", almost all of them constants that are not
#: roots at all. An audit that cries wolf is an audit that gets muted.
DATA_ROOT_NAMES = frozenset((
    "DERIVED", "DATA_ROOT", "OUT_DERIVED", "MAIN_DERIVED", "LOCAL_DERIVED",
    "LEDGER_DERIVED", "CACHE", "FRAGMENT", "TOPUP", "OUT_DIR", "TAPE_PATH",
    "RAW", "DEST"))


def audit_derived_roots(pkg: Path | None = None) -> dict:
    """A SECOND ROOT, found by the ACT and not by the spelling.

    `audit_literals` greps for the canonical path string. `HERE.parents[1]`
    contains no string, so a module that derives a data root from its own
    file location passed the audit whose docstring names exactly that defect:
    *"BE modules each computed `parents[1]` or `parents[2]` on their own, so
    a BE receipt could not say WHICH tree it read."* The reviewer drove it:
    27 `parents[N]` lines across 14 files, and the audit saw none.

    This walks the AST for MODULE-LEVEL assignments whose target names a
    place data lives and whose value derives from `parents`. A code root --
    `ROOT = HERE.parents[1]` used only for source paths -- is NOT flagged,
    because it is not the defect."""
    import ast
    d = Path(pkg) if pkg is not None else Path(__file__).resolve().parent
    hits = []
    declared = []
    for f in sorted(d.glob("be_*.py")):
        if f.name == LITERAL_OWNER:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            hits.append({"file": f.name, "line": 0, "target": "<unparsable>",
                         "expr": "<unparsable>"})
            continue
        # names bound from `parents[...]` in this module: the CODE roots a
        # data root must not be built on
        code_roots = {t2.id for n2 in tree.body
                      if isinstance(n2, ast.Assign)
                      and "parents" in ast.unparse(n2.value)
                      for t2 in n2.targets if isinstance(t2, ast.Name)}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            src = ast.unparse(node.value)
            if "_BDR." in src or "be_data_root." in src or "_RES" in src:
                continue                     # comes from the resolver
            # TWO FORMS, BOTH NARROW. (a) the value derives from `parents`
            # directly. (b) it is `<code-root name> / "...data..."` -- the
            # form this audit MISSED on its first run
            # (`DERIVED = ROOT / "data/pm_5min/derived"`), which is the same
            # defect one variable removed. Nothing else is flagged: the
            # widened "must come from the resolver" rule reported 33 lines,
            # almost all constants that are not roots.
            form_a = "parents" in src
            form_b = (isinstance(node.value, ast.BinOp)
                      and isinstance(node.value.op, ast.Div)
                      and isinstance(node.value.left, ast.Name)
                      and node.value.left.id in code_roots
                      and "data" in src)
            if not (form_a or form_b):
                continue
            for t in node.targets:
                name = getattr(t, "id", None)
                if not (name and name.upper() in DATA_ROOT_NAMES):
                    continue
                # AN EXEMPTION A READER CAN SEE. A deliberate second root --
                # `be_forward_preflight.LOCAL_DERIVED` mirrors ledger files
                # INTO the local tree, so it must be the local tree -- is
                # allowed only with an inline marker carrying its reason.
                # Silence is not permitted; the same rule as `fixture=True`
                # needing a `why`.
                line = f.read_text().splitlines()[node.lineno - 1]
                if MARKER in line:
                    declared.append({"file": f.name, "line": node.lineno,
                                     "target": name,
                                     "reason": line.split(MARKER, 1)[1].strip()})
                    continue
                hits.append({"file": f.name, "line": node.lineno,
                             "target": name, "expr": src[:90]})
    return {"n_offending": len(hits), "offenders": hits, "clean": not hits,
            "declared_second_roots": declared,
            "n_declared": len(declared),
            "an_exemption_must_be_visible": f"an intentional second root "
                                            f"carries `{MARKER} <reason>` on "
                                            f"its own line and is REPORTED "
                                            f"here; an undeclared one is an "
                                            f"offender",
            "what_it_looks_for": "a MODULE-LEVEL assignment whose target "
                                 "names a data location and whose value does "
                                 "NOT come from the resolver -- the ACT, not "
                                 "the spelling. Stated positively because the "
                                 "first version looked for `parents` and "
                                 "missed `DERIVED = ROOT / 'data/...'`, which "
                                 "is the same defect one variable removed.",
            "what_it_deliberately_allows": "a CODE root (e.g. ROOT = "
                                           "HERE.parents[1]) used for source "
                                           "paths; that is not the defect",
            "names_treated_as_data_locations": list(DATA_ROOT_NAMES)}


def receipt_block(res: dict | None = None, **kw) -> dict:
    """What every BE receipt carries so the tree it read is never a guess."""
    r = require_ledger(res, **kw)
    return {k: r[k] for k in (
        "branch", "env_var", "env_value", "repo_root", "data_root",
        "is_ledger", "data_root_exists", "tape_present", "ledger_check")
        if k in r} | ({"exemption_reason": r["exemption_reason"]}
                      if "exemption_reason" in r else {})


EXPECTED_CHECKS = 17


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

    # ---- BE48 B.4: THE AUDIT CAN NOW SEE A SECOND ROOT --------------------
    dr = audit_derived_roots()
    ok(dr["clean"],
       f"NO BE module derives a DATA root from `parents[N]` any more "
       f"({dr['n_offending']} offending assignments)"
       + ("" if dr["clean"] else
          f"; OFFENDERS: {[(h['file'], h['line'], h['target']) for h in dr['offenders']]}"))
    import tempfile as _t2
    with _t2.TemporaryDirectory() as _td2:
        _d2 = Path(_td2)
        (_d2 / "be_planted_second_root.py").write_text(
            "from pathlib import Path\n"
            "HERE = Path(__file__).resolve().parent\n"
            "ROOT = HERE.parents[1]\n"                # a CODE root: allowed
            "DERIVED = HERE.parents[1] / 'data/pm_5min/derived'\n")
        _pl = audit_derived_roots(_d2)
        ok(not _pl["clean"] and _pl["n_offending"] == 1
           and _pl["offenders"][0]["target"] == "DERIVED",
           f"KNOWN-BAD, PLANTED: the audit FIRES on `DERIVED = "
           f"HERE.parents[1] / ...` and does NOT fire on the `ROOT = "
           f"HERE.parents[1]` code root beside it -- it finds the ACT, and "
           f"it distinguishes a data root from a source root")

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

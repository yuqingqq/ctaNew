"""RULE 22 AS AMENDED (R-605): what code produced this receipt, captured AT
IMPORT and refused at EMIT if any of it moved.

THE DEFECT THIS EXISTS FOR. The Gate-1 runner stamped its provenance by
re-reading `__file__` when it wrote its receipt. A landing to that file 13
minutes into a 1.5-hour run would have made the receipt name bytes that did
not run, and the committed-bytes guard would have PASSED because the
replacement was committed (R-603). Reading ONE file is not enough either: a
sibling module in the same import closure moves the producing code just as
surely (REV 51 S3, R-605), and BE's three producers carried no stamp at all
(R-613).

WHY AN OBJECT AND NOT MODULE GLOBALS. The battery has to drive a capture over
a tree that is not `live/` -- otherwise the known-bad has to mutate a real
module in the worktree to prove the drift check fires. A per-instance closure
lets the falsifier run against a temporary tree while the producers use the
default instance, and it keeps the test from patching the state it is testing
(a selftest that re-imports its own module patches a SECOND module object; the
same trap in a different dress).

WHAT IS AND IS NOT COVERED, stated rather than implied:
  * covered: every module under `live/` in `sys.modules` at capture time, the
    worktree HEAD, and whether the worktree was dirty at import;
  * NOT covered: modules imported LATER than the last capture call (a producer
    with lazy imports must capture again after them -- `be_daybook_build` does,
    at its mid-run import of the fragment module), the interpreter, installed
    packages outside `live/`, and data.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

LIVE_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[2])


class Rule22Refused(RuntimeError):
    """The code that produced this emit is not the code that ran."""


def _git(root: str, *args, raw: bool = False) -> str | None:
    """`raw=True` preserves LEADING whitespace.

    `git status --porcelain` encodes the index state in column 1 and the
    worktree state in column 2, so a tracked modification begins with a
    SPACE (" M path"). Stripping that shifts every subsequent index by one
    and the parse silently drops the first character of the path -- which is
    exactly what this did until it was driven on a worktree whose first
    porcelain line was a tracked change. Untracked entries ("?? path") have
    no leading space, so the defect was invisible on every receipt landed so
    far: their only entry was the ledger symlink."""
    try:
        r = subprocess.run(["git", "-C", root, *args], capture_output=True,
                           text=True, timeout=60)
    except Exception:                                        # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    return r.stdout.rstrip("\n") if raw else r.stdout.strip()


def module_commit(path) -> dict:
    """The last commit that touched `path` -- READ, never typed.

    `seam.commit` was the literal "6f134a6" for three rounds: true of the
    front door once, and unchecked since (DA 77, R-613). The receipt now
    names what it can locate and says so when it cannot."""
    p = Path(path).resolve()
    root = str(p.parents[2]) if len(p.parents) > 2 else REPO_ROOT
    c = _git(root, "log", "-1", "--format=%H", "--", str(p))
    if c:
        return {"module": p.name, "commit": c, "short": c[:7],
                "source": "git log -1 -- <module>, read at import"}
    return {"module": p.name, "commit": None, "short": None,
            "source": "git log -1 -- <module>, read at import",
            "why_absent": "the module's history could not be read from this "
                          "tree; the receipt names what it can locate rather "
                          "than restating a constant"}


class PorcelainMalformed(ValueError):
    """A `git status --porcelain` line that does not have porcelain's shape.

    Raised rather than sliced. A line that has lost its leading space still
    LOOKS parseable -- `line[3:]` returns a path that is one character
    short -- so a mis-slice is silent, and silence is the whole defect."""


#: porcelain v1 status letters, both columns. A space is a valid state.
_PORCELAIN_STATES = set(" MTADRCU?!")


def parse_porcelain_line(line: str) -> tuple:
    """(status_code, path) from ONE porcelain v1 line -- SHAPE CHECKED.

    R-637: the defect has two halves and no seat had both right. The READ
    half is a `.strip()` eating the leading space of the first line; this
    module fixed that in round 62 with a raw read. THE SLICE HALF is
    `line[3:]`, which is correct ONLY while the read stays raw -- so the
    fix is one edit away from the defect for anyone who touches the helper,
    and the slice is the half people edit.

    So the offset is no longer assumed. Porcelain v1 is `XY<space>path`
    with X and Y drawn from a fixed alphabet; a line that does not have
    that shape is REFUSED by name instead of being sliced into a path that
    is silently one character short. A rename or copy carries
    `old -> new`, and the path that exists in the worktree is the NEW one.
    """
    if (len(line) < 4 or line[2] != " "
            or line[0] not in _PORCELAIN_STATES
            or line[1] not in _PORCELAIN_STATES):
        raise PorcelainMalformed(
            f"REFUSED: {line!r} is not a porcelain v1 line (expected two "
            f"status characters then a space). A line that lost its leading "
            f"space -- what a `.strip()` on the whole output does to the "
            f"FIRST line -- still slices to a path that is one character "
            f"short, and nothing downstream can tell (R-637).")
    code, rest = line[:2], line[3:]
    if ("R" in code or "C" in code) and " -> " in rest:
        rest = rest.split(" -> ", 1)[1]      # the path in the worktree NOW
    if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
        rest = rest[1:-1]                    # git quotes unusual paths
    return code, rest


class HeavyRunRefused(RuntimeError):
    """This process is not entitled to run heavy work right now."""


LAUNCHER = Path(__file__).with_name("be_heavy_run.sh")
DECLARATIONS = Path(__file__).with_name("declarations")


class DeclarationAbsent(RuntimeError):
    """A check that depends on a declaration FAILS when it is gone.

    R-649 §3.1(1): it never skips. A skipped check reads as a pass to
    everything downstream, and the declaration is the thing being relied
    on."""


def declaration_head(family: str) -> dict:
    """The HEAD of a declaration chain, resolved by {path, sha256}.

    R-653: resolve the CHAIN, never a filename -- a filename is a guess
    about which version governs. A chain of one resolves to itself; a
    version whose predecessor's digest does not match on disk is refused
    rather than silently accepted."""
    import hashlib as _h
    cands = sorted(DECLARATIONS.glob(f"{family}_v*.json"))
    if not cands:
        raise DeclarationAbsent(
            f"REFUSED: no declaration of family {family!r} under "
            f"{DECLARATIONS}. This check depends on it and therefore FAILS; "
            f"it does not skip (R-649).")
    loaded = {}
    for q in cands:
        b = q.read_bytes()
        loaded[q.name] = {"path": q, "sha256": _h.sha256(b).hexdigest(),
                          "doc": json.loads(b)}
    superseded = set()
    for name, e in loaded.items():
        sup = e["doc"].get("supersedes")
        if isinstance(sup, dict) and sup.get("path"):
            prev = Path(sup["path"]).name
            if prev in loaded and loaded[prev]["sha256"] != sup.get("sha256"):
                raise DeclarationAbsent(
                    f"REFUSED: {name} supersedes {prev} by a digest that "
                    f"does not match the file on disk -- the chain is "
                    f"broken and no head can be resolved.")
            superseded.add(prev)
    heads = [n for n in loaded if n not in superseded]
    if len(heads) != 1:
        raise DeclarationAbsent(
            f"REFUSED: {family} resolves to {len(heads)} heads ({heads}); a "
            f"declaration family with two heads has no governing version.")
    h = loaded[heads[0]]
    return {"name": heads[0], "sha256": h["sha256"], "doc": h["doc"],
            "path": str(h["path"]), "n_versions": len(loaded)}


def lock_conflict_rc() -> int:
    """THE conflict code, from the declaration -- never a literal here.

    REV 67 §2.3: the code was a literal in the launcher AND a constant in
    this module, and `assert_launch_form` returned THIS one -- so a
    launcher refusing with 76 was published as 75."""
    return int(declaration_head("heavy_run_form")["doc"]["lock_conflict_rc"])


def unit_outcome(unit: str) -> dict:
    """The unit's outcome as the DECLARATION defines it: five fields + the id.

    R-653: `LoadState: not-found` is VOID, never a verdict -- a unit that
    was collected reads exactly like a unit that never ran. And the
    InvocationID is what makes two polls distinguishable: a unit NAME names
    every run ever launched under it, and a failed unit's properties
    persist until `reset-failed`, so a repeated id is the SAME refusal."""
    d = declaration_head("heavy_run_form")["doc"]
    fields = list(d["unit_outcome_minimum_read"]) + ["InvocationID"]
    out = {}
    for f in fields:
        r = subprocess.run(["systemctl", "--user", "show",
                            unit if unit.endswith(".service")
                            else f"{unit}.service", "-p", f, "--value"],
                           capture_output=True, text=True, timeout=30)
        out[f] = r.stdout.strip() if r.returncode == 0 else None
    out["fields_from"] = "declaration head unit_outcome_minimum_read + id"
    out["void"] = out.get("LoadState") == "not-found"
    if out["void"]:
        out["why_void"] = ("the unit is not loaded: a collected unit and a "
                           "unit that never ran are indistinguishable here, "
                           "so this is VOID and not a verdict (R-653)")
    return out


def porcelain_derivation_census(src: str) -> dict:
    """WHAT THIS IS: a REGRESSION GUARD on the two readers that use
    `git status --porcelain` as a BOOLEAN today -- not a general proof that
    no path can ever be derived from porcelain in any code.

    REV 67 §1.3 drove nine deriving shapes past the first version, which
    only looked for a Subscript or a `.split` on a directly-assigned name.
    It now propagates taint: through assignment (transitively), through a
    Call RECEIVER (`git(...).splitlines()`), through for-loop and
    comprehension targets, and through subscripting. A derivation is a
    Subscript on anything tainted, or a tainted value reaching a slice.

    Its limits, stated rather than left to be discovered: it does not follow
    values across function boundaries, through containers, or through
    `eval`/`getattr`; a determined path derivation can still evade it. That
    is why it is named a regression guard."""
    import ast as _a
    tree = _a.parse(src)
    tainted, derived = set(), []

    def _is_porc_call(n):
        return isinstance(n, _a.Call) and any(
            isinstance(c, _a.Constant) and c.value == "--porcelain"
            for c in _a.walk(n))

    def _tainted_expr(n):
        if _is_porc_call(n):
            return True
        if isinstance(n, _a.Name):
            return n.id in tainted
        if isinstance(n, _a.Attribute):          # git(...).splitlines
            return _tainted_expr(n.value)
        if isinstance(n, _a.Call):               # x.split(...) / f(x)
            return _tainted_expr(n.func)
        if isinstance(n, _a.Subscript):
            return _tainted_expr(n.value)
        if isinstance(n, _a.BinOp):
            return _tainted_expr(n.left) or _tainted_expr(n.right)
        return False

    # taint to a fixed point: assignments, loops, comprehensions
    for _ in range(6):
        before = len(tainted)
        for n in _a.walk(tree):
            if isinstance(n, _a.Assign) and _tainted_expr(n.value):
                for t in n.targets:
                    if isinstance(t, _a.Name):
                        tainted.add(t.id)
            if isinstance(n, (_a.For, _a.AsyncFor)) and _tainted_expr(n.iter):
                if isinstance(n.target, _a.Name):
                    tainted.add(n.target.id)
            if isinstance(n, _a.comprehension) and _tainted_expr(n.iter):
                if isinstance(n.target, _a.Name):
                    tainted.add(n.target.id)
        if len(tainted) == before:
            break

    for n in _a.walk(tree):
        if isinstance(n, _a.Subscript) and _tainted_expr(n.value):
            derived.append({"kind": "subscript", "line": n.lineno})
        if (isinstance(n, _a.Attribute)
                and n.attr in ("split", "splitlines", "partition",
                               "rpartition", "rsplit")
                and _tainted_expr(n.value)):
            # splitting is only a derivation if the pieces are then used;
            # the taint walk above will have caught that as a subscript or
            # a loop target, so this is recorded as a SPLIT, not a verdict
            derived.append({"kind": f"split:{n.attr}", "line": n.lineno})
    # how many porcelain CALLS (not string constants: `git worktree list
    # --porcelain` carries the same constant and is a different command)
    calls = [n for n in _a.walk(tree) if _is_porc_call(n)]
    status_calls = [n for n in calls if any(
        isinstance(c, _a.Constant) and c.value == "status"
        for c in _a.walk(n))]
    return {"what_this_is": "a regression guard on today's readers, not a "
                            "general proof",
            "n_porcelain_calls": len(calls),
            "n_status_porcelain_calls": len(status_calls),
            "note_on_the_count": "the earlier count counted the STRING "
                                 "CONSTANT; two of be_forward_day's three "
                                 "are `git worktree list --porcelain`, a "
                                 "different command entirely",
            "tainted_names": sorted(tainted),
            "derivations": derived,
            "derives_a_path": bool([d for d in derived
                                    if d["kind"] == "subscript"])}


def journal_copy(unit: str, invocation_id: str | None = None, *,
                 window_start_utc: str | None = None,
                 max_lines: int = 40) -> dict:
    """COPY journal lines into an artifact AT THE MOMENT OF READING, with
    the source's RETENTION STATE MEASURED beside them (R-641, rule 20).

    The journal is not the record: it rotates within hours (DE 84's Started
    line was gone four hours later, and the window's start advanced ~15 min
    in 18 min on 09-06). So a number read from it is copied here and now;
    the retention state is a MEASUREMENT with its own query and as-of, never
    a typed string; and no verdict below depends on retention -- a window
    the journal no longer reaches is reported UNMEASURED, naming the oldest
    entry that does exist.

    Lines are filtered on the run's InvocationID with BOTH fields, because a
    unit NAME names every run ever launched under it (99 manager lines for
    be64book by 13:37Z). `_SYSTEMD_INVOCATION_ID` carries the payload's
    lines and `USER_INVOCATION_ID` the user manager's Started/Consumed
    lines; `INVOCATION_ID` is the SYSTEM manager's field and matches nothing
    here."""
    u = unit if unit.endswith(".service") else f"{unit}.service"
    now = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                         capture_output=True, text=True, timeout=30
                         ).stdout.strip()

    def _run(args):
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else ""

    # RETENTION, MEASURED -- and measured CORRECTLY. The first form of this
    # used `-n 1 --reverse`, which returns the NEWEST entry: it labelled the
    # newest as the oldest, which is the same class of defect as every other
    # value here that looked right. `-n` takes the TAIL; the oldest is the
    # FIRST line of the unfiltered listing.
    _first = _run(["journalctl", "--user", "--no-pager", "-o",
                   "short-iso"]).split("\n", 1)[0]
    oldest_measured = _first.split(" ")[0] if _first.strip() else None
    by_unit = [l for l in _run(["journalctl", "--user", "-u", u, "--no-pager",
                                "-o", "cat"]).split("\n") if l]
    by_id = []
    if invocation_id:
        for field in ("_SYSTEMD_INVOCATION_ID", "USER_INVOCATION_ID"):
            by_id += [l for l in _run(["journalctl", "--user",
                                       f"{field}={invocation_id}",
                                       "--no-pager", "-o", "cat"]
                                      ).split("\n") if l]
    out = {
        "unit": u, "invocation_id": invocation_id, "as_of": now,
        "query_by_id": [f"journalctl --user _SYSTEMD_INVOCATION_ID={invocation_id}",
                        f"journalctl --user USER_INVOCATION_ID={invocation_id}"],
        "query_by_unit": f"journalctl --user -u {u} -o cat",
        "n_lines_by_unit": len(by_unit),
        "n_lines_by_id": len(by_id) if invocation_id else None,
        "lines": (by_id or by_unit)[-max_lines:],
        "copied_at_the_moment_of_reading": True,
        "retention": {
            "oldest_entry_the_journal_holds": oldest_measured,
            "query": "journalctl --user --no-pager -o short-iso | first line",
            "as_of": now,
            "is_a_measurement_not_a_string": True,
            "note": "the window's start advanced ~15 min in 18 min on "
                    "09-06, so a retention state named once and re-quoted "
                    "later is stale",
        },
        "why_the_field_names": "INVOCATION_ID is the SYSTEM manager's field "
                               "and matches nothing under --user; the user "
                               "manager's Started/Consumed lines carry "
                               "USER_INVOCATION_ID",
    }
    if window_start_utc:
        covered = bool(oldest_measured and oldest_measured <= window_start_utc)
        out["window_start_utc"] = window_start_utc
        out["window_fully_covered"] = covered
        if not covered:
            out["status"] = "UNMEASURED"
            out["why_unmeasured"] = (
                f"the journal's oldest entry is {oldest_measured}, which is "
                f"AFTER the window's start {window_start_utc}: those lines "
                f"have rotated out. This is UNMEASURED, not absent and not "
                f"zero -- no verdict may rest on it.")
    if invocation_id and by_unit and not by_id:
        out["status"] = "COPY_REFUSED"
        out["why_refused"] = (
            f"the by-id query returned 0 lines where `-u {u}` has "
            f"{len(by_unit)}: a copy that finds nothing where the unit has "
            f"lines is a refusal of the copy, never a record (rule 20).")
    return out


def cgroup_leaf() -> dict:
    """This process's own cgroup leaf, and what KIND of unit it is.

    REV 65 §1.2: the lint cannot see a `--scope` behind a variable or a
    wrapper, and it should not be asked to. The property is not "the string
    is absent from a line", it is "this run's unit is a .service" -- and
    that is decidable HERE, at runtime, from the leaf the producers already
    report."""
    try:
        leaf = open("/proc/self/cgroup").read().strip().rsplit("/", 1)[-1]
    except OSError:
        return {"leaf": None, "kind": "UNKNOWN"}
    kind = ("scope" if leaf.endswith(".scope")
            else "service" if leaf.endswith(".service")
            else "none")
    return {"leaf": leaf, "kind": kind,
            "why_this_and_not_the_lint": "a static scan cannot see a "
                                         "`--scope` behind a variable or a "
                                         "wrapper; the leaf is what the run "
                                         "actually got"}


def assert_not_a_scope(*, fixture: bool = False) -> dict:
    """REFUSE a real day whose own unit is a transient SCOPE (R-628).

    Nine BE heavy runs were scopes and every receipt said so in
    `scope.unit`; no seat read it. This reads it."""
    c = cgroup_leaf()
    if not fixture and c["kind"] == "scope":
        raise HeavyRunRefused(
            f"REFUSED: this process's cgroup leaf is {c['leaf']!r} -- a "
            f"transient SCOPE. A scope's payload sits in the launching "
            f"shell's process tree and dies with it (R-628). Launch through "
            f"{LAUNCHER.name}, which runs a transient SERVICE. The static "
            f"lint cannot catch a `--scope` behind a variable; this can, "
            f"because it reads what the run actually got.")
    return c


def flock_mode(lock_path) -> str | None:
    """WRITE (exclusive) or READ (shared), read from /proc/locks.

    `flock -n` takes LOCK_EX; `flock -s -n` takes LOCK_SH and TWO holders
    then coexist. The fd is present either way, so holding it is not
    evidence of exclusion. Moved here from `be_daybook_build` so all three
    producers read the lock the same way."""
    import os
    try:
        st = os.stat(str(lock_path))
        want = f"{st.st_dev >> 8:02x}:{st.st_dev & 0xff:02x}:{st.st_ino}"
        for line in open("/proc/locks"):
            f = line.split()
            if len(f) >= 6 and f[1] == "FLOCK" and f[3] in ("READ", "WRITE"):
                if f[5].endswith(f":{st.st_ino}") or f[5] == want:
                    return f[3]
    except OSError:
        pass
    return None


def lock_evidence(*, fixture: bool = False, refuse: bool = True) -> dict:
    """THE HEAVY LOCK, MEASURED -- and refused if a real build lacks it.

    REV 63 S4: BE's fragment and tape receipts mentioned the lock in no
    field at all, so "the lock was taken and held across both steps" was a
    claim in a register row that its own artifacts could not support. DE's
    runner has recorded this for rounds and refuses a real day without it;
    this DELEGATES to that rather than writing a second implementation of
    one check (Q-BE-271), and adds the mode, which is what tells an
    exclusive hold from a shared one."""
    import de_multiday_gate1_runner as RUN
    w = dict(RUN.wrapper_observed())
    w["delegated_to"] = "de_multiday_gate1_runner.wrapper_observed"
    w["fixture"] = fixture
    w["lock_mode"] = flock_mode(RUN.HEAVY_RUN_LOCK)
    w["exclusive"] = w["lock_mode"] == "WRITE"
    w["why_mode_not_just_held"] = ("two `flock -s` holders would both report "
                                   "the fd and both certify; only WRITE is "
                                   "mutual exclusion")
    w["launch_form"] = "systemd-run --user transient SERVICE (never --scope)"
    if fixture or not refuse:
        return w
    if not w.get("heavy_run_lock_held"):
        raise HeavyRunRefused(
            f"REFUSED: this build is HEAVY BY CONSTRUCTION and this process "
            f"does not hold {RUN.HEAVY_RUN_LOCK}. Launch it with "
            f"`{LAUNCHER.name} <unit> <module.py> ...`, which runs the lock "
            f"INSIDE a transient service (R-628). A heavy build beside "
            f"another heavy run is what rule 20 exists to prevent.")
    if not w["exclusive"]:
        raise HeavyRunRefused(
            f"REFUSED: the heavy-run lock is held in mode "
            f"{w['lock_mode']!r}, not WRITE. A SHARED (`flock -s`) lock lets "
            f"a second heavy run take it at the same time and both would "
            f"certify -- which is not one-heavy-run-at-a-time.")
    return w


def assert_launch_form(text: str | None = None) -> dict:
    """THE LAUNCHER'S SHAPE IS A PREDICATE, read from the script itself.

    The scope form survived six BE runs because it lived in prose that
    nobody executed. This reads the artifact that actually launches the
    work, so the command and the check on it cannot become two facts."""
    raw = text if text is not None else LAUNCHER.read_text()
    # SCAN THE LAUNCH COMMAND, NOT THE SCRIPT. Two earlier forms of this
    # check refused this very launcher: first on its header comment
    # explaining what a `--scope` does wrong, then on its falsifier's own
    # PASS message, which names the form it disproves. Stripping comments
    # was not enough -- a checker that names a forbidden token contains it,
    # wherever it puts it. So the search space is now the systemd-run
    # invocation itself (its line plus backslash continuations), which is
    # the only place the token could do harm.
    lines, launch, grabbing = raw.split("\n"), [], False
    for ln in lines:
        st = ln.strip()
        if not grabbing and ("systemd-run" in st
                             and not st.startswith("#")):
            grabbing = True
        if grabbing:
            launch.append(st)
            if not st.endswith("\\"):
                break
    src = " ".join(launch) if launch else raw
    # the lock is INSIDE the unit iff the launch line hands the payload to
    # this script's own --inner role, which is where the flock is taken.
    body = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    inside = "--inner" in src and "flock -n" in body
    problems = []
    if not launch:
        problems.append("contains no systemd-run invocation at all")
    if "--scope" in src:
        problems.append("carries `--scope`: the payload would run in the "
                        "CALLING shell's process tree and die with it "
                        "(R-628, measured)")
    for token, why in (("--unit=", "names no `--unit=`, so the run could "
                                   "only be polled by a child PID"),
                       ("--slice=research.slice", "is not in research.slice"),
                       ("MemoryMax=8G", "declares no memory cap"),
                       ("CPUQuota=100%", "declares no CPU cap"),
                       ("--setenv=PM_DATA_ROOT=", "does not set the data "
                                                  "root inside the unit"),
                       ("--working-directory=", "does not set the working "
                                                "directory inside the unit"),
                       ("flock -n", "does not take the heavy-run lock")):
        if token not in src and not (token == "flock -n" and inside):
            problems.append(why)
    # the lock must be INSIDE the unit: after the `--` separator
    if not inside:
        problems.append("takes the lock OUTSIDE the unit, so the lock dies "
                        "with the launching shell")
    # REV 67 §2.3: this returned the PYTHON constant as `conflict_exit_code`,
    # so a launcher refusing with 76 was published as 75 (driven on a scratch
    # copy). The value is now PARSED OUT OF THE LAUNCHER and asserted equal
    # to the declaration's -- and if the launcher sources it from the
    # declaration rather than assigning a literal, that is recorded as the
    # stronger form rather than as a missing token.
    declared = lock_conflict_rc()
    m = re.search(r"^LOCK_CONFLICT_RC=(.+)$", raw, re.M)
    assigned = m.group(1).strip() if m else None
    sourced = bool(assigned and "heavy_run_form" in raw
                   and not assigned.lstrip("$").isdigit()
                   and assigned.startswith("$("))
    literal = None
    if assigned and assigned.isdigit():
        literal = int(assigned)
        if literal != declared:
            problems.append(
                f"declares LOCK_CONFLICT_RC={literal} while the declaration "
                f"head says {declared}: a launcher refusing with one code "
                f"while its checker publishes another is exactly REV 67 "
                f"§2.3's defect")
    elif not sourced:
        problems.append("assigns LOCK_CONFLICT_RC from neither a literal nor "
                        "the declaration, so no reader can know what it "
                        "refuses with")
    return {"launcher": str(LAUNCHER), "problems": problems,
            "scanned": "the systemd-run invocation itself",
            "launch_line": src[:400],
            "form_is_correct": not problems,
            "lock_is_inside_the_unit": inside,
            "conflict_exit_code_declared": declared,
            "conflict_exit_code_in_launcher": literal,
            "launcher_sources_it_from_the_declaration": sourced,
            "declaration_head": declaration_head("heavy_run_form")["name"]}


def running_unit_exec_start(unit: str | None = None) -> dict:
    """THE BYTES THAT ARE RUNNING, not the importing tree's copy.

    R-653: `LAUNCHER` resolves relative to whichever worktree imported this
    module, so a check on "the launcher's bytes" can inspect a file that is
    not the one the unit is executing. Read the unit's own `ExecStart`, or
    -- from inside the run -- this process's parent command line."""
    import hashlib as _h
    out = {"asked_for": unit}
    if unit:
        r = subprocess.run(["systemctl", "--user", "show",
                            unit if unit.endswith(".service")
                            else f"{unit}.service", "-p", "ExecStart",
                            "--value"], capture_output=True, text=True,
                           timeout=30)
        out["ExecStart"] = r.stdout.strip() or None
    else:
        try:
            ppid = int(open("/proc/self/status").read()
                       .split("PPid:")[1].split()[0])
            out["ppid"] = ppid
            out["ppid_cmdline"] = open(f"/proc/{ppid}/cmdline").read(
                ).replace("\x00", " ").strip()
        except (OSError, IndexError, ValueError):
            out["ppid_cmdline"] = None
    # the path the RUNNING command names, and that file's digest now
    txt = out.get("ExecStart") or out.get("ppid_cmdline") or ""
    m = re.search(r"(/\S*be_heavy_run\.sh)", txt)
    out["launcher_path_in_the_running_command"] = m.group(1) if m else None
    if m:
        try:
            out["launcher_sha256_on_disk"] = _h.sha256(
                Path(m.group(1)).read_bytes()).hexdigest()
        except OSError:
            out["launcher_sha256_on_disk"] = None
    out["importing_tree_launcher"] = str(LAUNCHER)
    out["same_file"] = (out.get("launcher_path_in_the_running_command")
                        == str(LAUNCHER))
    out["why"] = ("`LAUNCHER` is relative to the importing worktree; a claim "
                  "about the launcher's bytes must name the file the UNIT "
                  "is executing (R-653)")
    return out


class Capture:
    """One run's view of the code that is producing it."""

    def __init__(self, *, root: str | None = None,
                 worktree: str | None = None) -> None:
        self.root = str(Path(root or LIVE_DIR).resolve())
        self.worktree = str(Path(worktree or REPO_ROOT).resolve())
        #: path -> {"sha256": ..., "first_seen_at": <phase label>}
        self.closure: dict = {}
        self.head_at_import: dict | None = None

    # ---- capture -------------------------------------------------------
    def _digest(self, mod, phase: str) -> None:
        f = getattr(mod, "__file__", None)
        if not f:
            return
        try:
            p = Path(f).resolve()
        except OSError:
            return
        k = str(p)
        if not k.startswith(self.root) or k in self.closure:
            return                      # first sight only -- never re-read
        try:
            self.closure[k] = {"sha256":
                               hashlib.sha256(p.read_bytes()).hexdigest(),
                               "first_seen_at": phase}
        except OSError:
            self.closure[k] = {"sha256": None, "first_seen_at": phase}

    def capture(self, phase: str) -> "Capture":
        """Digest every not-yet-seen module under `root`. Idempotent."""
        for m in list(sys.modules.values()):
            self._digest(m, phase)
        if self.head_at_import is None:
            self.head_at_import = self.head_state()
        return self

    @staticmethod
    def _ledger_data() -> Path | None:
        """The canonical ledger, RESOLVED from the data-root resolver -- not
        a path typed here. A hardcoded ledger path is the same class of
        literal this module exists to catch."""
        try:
            import be_data_root as _B
            return Path(_B.data_root()).resolve()
        except Exception:                                    # noqa: BLE001
            return None

    def classify_dirt(self, line: str) -> dict:
        """One porcelain line -> is this CODE dirt, or the ledger symlink?

        THE EXEMPTION IS A PROPERTY, NEVER A NAME. Exempting anything called
        `data` would be the defect this seat keeps shipping: a literal
        standing in for a fact. Three conjuncts, each computed, all required:

          * git reports it UNTRACKED (`??`) -- a tracked modification at
            that path is real dirt and stays dirt;
          * the entry on disk IS a symlink -- a plain file at that name is
            real dirt;
          * it RESOLVES to the ledger -- a symlink pointing anywhere else is
            real dirt.

        A link to the ledger under some OTHER name is exempt too, and that
        is deliberate: the question rule 22 asks is whether the PRODUCING
        CODE moved, and a symlink to the ledger is not producing code no
        matter what it is called. The name carries nothing either way."""
        code, path = parse_porcelain_line(line)
        full = Path(self.worktree) / path
        is_untracked = code == "??"
        is_link = full.is_symlink()
        target = None
        if is_link:
            try:
                target = str(full.resolve())
            except OSError:
                target = None
        ledger = self._ledger_data()
        points_at_ledger = bool(ledger and target and target == str(ledger))
        return {"path": path, "status_code": code,
                "is_untracked": is_untracked, "is_symlink": is_link,
                "resolves_to": target,
                "resolves_to_the_ledger": points_at_ledger,
                "ledger": str(ledger) if ledger else None,
                "exempt": is_untracked and is_link and points_at_ledger}

    def head_state(self) -> dict:
        st = _git(self.worktree, "status", "--porcelain", raw=True)
        lines = [x for x in (st or "").split("\n") if x]
        rows = [self.classify_dirt(x) for x in lines]
        code_rows = [r for r in rows if not r["exempt"]]
        return {"worktree": self.worktree,
                "head": _git(self.worktree, "rev-parse", "HEAD"),
                # RAW -- meaning unchanged from every receipt already landed:
                # anything at all that git reports.
                "dirty": bool(st) if st is not None else None,
                "dirty_paths": [r["path"] for r in rows][:20],
                # THE RULE-22 QUESTION, under a key that cannot be confused
                # with the raw one. A seat worktree carries `data` as a
                # symlink to the ledger by R-553, so `dirty` reads true on a
                # perfectly clean code tree and a reader of the receipt sees
                # dirt that is not there.
                "dirty_code": (bool(code_rows) if st is not None else None),
                "dirty_paths_code": [r["path"] for r in code_rows][:20],
                "exempt_entries": [r for r in rows if r["exempt"]][:20],
                "exemption_is_a_property": "untracked AND a real symlink AND "
                                           "resolving to the ledger -- never "
                                           "a path called `data`",
                }

    # ---- drift ---------------------------------------------------------
    def drift(self) -> list:
        """Which captured modules have MOVED on disk since first sight."""
        out = []
        for k, was in self.closure.items():
            try:
                now = hashlib.sha256(Path(k).read_bytes()).hexdigest()
            except OSError:
                now = None
            if now != was["sha256"]:
                out.append({"module": Path(k).name, "path": k,
                            "at_import": was["sha256"], "now": now,
                            "first_seen_at": was["first_seen_at"]})
        return out

    # ---- the receipt block ---------------------------------------------
    def stamp(self, producing_file) -> dict:
        me = Path(producing_file).resolve()
        mine = self.closure.get(str(me), {})
        head_now = self.head_state()
        d = self.drift()
        hi = self.head_at_import or {}
        return {
            "producing_code": me.name,
            "producing_code_sha256": mine.get("sha256"),
            "captured_at": "IMPORT",
            "why_not_at_emit":
                "a digest taken when the receipt is written can be of bytes "
                "that changed during the run; the receipt would then name "
                "code that did not produce its numbers (R-603, rule 22)",
            "builder_commit": hi.get("head"),
            "import_closure": {
                "n_modules": len(self.closure),
                "root": self.root,
                "modules": {Path(k).name: v["sha256"]
                            for k, v in sorted(self.closure.items())},
                "first_seen_at": {Path(k).name: v["first_seen_at"]
                                  for k, v in sorted(self.closure.items())},
                "not_covered": "modules imported after the last capture call, "
                               "packages outside this root, the interpreter, "
                               "and data",
            },
            "closure_drift": d,
            "closure_unchanged_during_the_run": not d,
            "head_at_import": hi,
            "head_at_emit": head_now,
            "head_unchanged_during_the_run": hi.get("head") == head_now.get("head"),
            # BOTH, and neither redefined: the raw answer keeps the meaning
            # every landed receipt already used, and the rule-22 question
            # gets its own key rather than quietly changing that one.
            "worktree_was_dirty_at_import": hi.get("dirty"),
            "worktree_CODE_was_dirty_at_import": hi.get("dirty_code"),
            "what_the_two_dirty_fields_mean": "`worktree_was_dirty_at_import` "
                                              "is git's raw answer and "
                                              "includes the ledger symlink a "
                                              "seat worktree carries by "
                                              "R-553; "
                                              "`worktree_CODE_was_dirty_at_"
                                              "import` excludes entries that "
                                              "are PROVEN to be that symlink "
                                              "(untracked, a real link, "
                                              "resolving to the ledger). Rule "
                                              "22 asks the second one",
        }

    # ---- the refusal ---------------------------------------------------
    def assert_unchanged(self, where: str) -> dict:
        """REFUSE THE EMIT, BY NAME, if any of the code that ran moved.

        The RUN is unaffected -- Python holds the modules in memory. What is
        not acceptable is a receipt that names bytes which did not produce
        it."""
        d = self.drift()
        if d:
            raise Rule22Refused(
                f"REFUSED at {where}: A MODULE OF THIS RUN'S IMPORT CLOSURE "
                f"CHANGED UNDER IT -- {[x['module'] for x in d]}. The run is "
                f"unaffected (the modules are in memory); a receipt stamped "
                f"from the files on disk would name code that DID NOT RUN. "
                f"Rule 22 as amended (R-605): the capture is the closure, "
                f"not one file.")
        hi = self.head_at_import or {}
        now = self.head_state()
        if hi.get("head") != now.get("head"):
            raise Rule22Refused(
                f"REFUSED at {where}: THE WORKTREE'S HEAD MOVED UNDER THIS "
                f"RUN -- {str(hi.get('head'))[:12]} -> "
                f"{str(now.get('head'))[:12]}. A receipt's builder_commit "
                f"would name a commit this run did not execute from.")
        return {"where": where, "closure_unchanged": True,
                "head_unchanged": True, "n_modules": len(self.closure)}


#: The default instance every producer stamps from.
CAPTURE = Capture()


def init(phase: str) -> Capture:
    return CAPTURE.capture(phase)


def stamp(producing_file) -> dict:
    return CAPTURE.stamp(producing_file)


def assert_unchanged(where: str) -> dict:
    return CAPTURE.assert_unchanged(where)

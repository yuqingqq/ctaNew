"""DA: what is the midnight unit actually DEPLOYED at, and did it drift?

THE UNRULED BOUNDARY THIS CLOSES, filed twice by DA and bitten once (R-549(E)
item 4). The unit's `ExecStart` is an ABSOLUTE PATH INTO `/home/yuqing/ctaNew`
-- a tree NO SEAT OWNS. Every other seat commits there; nothing checks it.
So at 00:06:00Z the unit runs whatever that tree happens to hold, and because
`SuccessExitStatus=2` maps the expected open-day deferral to success, a night
run on stale or half-landed code reports GREEN and writes a governed verdict.
That verdict then feeds race accrual. Nothing in the system can tell that
night's verdict from one produced by the code that was reviewed.

MEASURED, NOT HYPOTHETICAL: at 2026-09-06T03:5xZ the deploy tree stood at
`1d66589` while `origin/mm-research` stood at `e25c69d` -- one commit behind,
with no seat aware of it and no instrument that would have said so.

THE FIX IS AT THE CLASSIFICATION OF "DEPLOYED", NOT AT A REMINDER.
"Deployed" becomes a RECORD written only by an explicit act
(`da_deploy_midnight.sh`), and the unit REFUSES TO RUN if what is on disk is
not what the record says. Drift stops being invisible and becomes rc 7.

WHY 7 AND NOT 5. The dispatch suggested 5. **5 IS ALREADY TAKEN** --
`da_midnight_verify.sh` exits 5 for the LOG/OUTDIR pair-guard refusal AND for
a verifier substitution, and 6 for an unnamed canonical write, 3 for a failed
`cd`, and 0/2/4 for its three-valued health signal. Reusing 5 would make two
different refusals indistinguishable in `systemctl` -- the exact defect this
programme keeps finding. 7 is the next free code and joins the 3/5/6 family:
refusals that fire BEFORE anything is written.

WHAT COUNTS AS DRIFT, AND WHY IT IS NOT "HEAD MOVED".
Pinning the unit to a COMMIT would refuse almost every night: the register is
appended several times a day and HEAD moves with it, while not one byte the
unit executes changes. An alarm that fires every night is one that gets turned
off -- this unit has already lived through that (`SuccessExitStatus=2` exists
because it was red every night for a correct refusal). So the record pins the
BYTES THE UNIT EXECUTES, and the commit is recorded as PROVENANCE, not as a
gate. A HEAD move alone is silent; a change to executed code is red.

WHICH BYTES ARE "EXECUTED" IS COMPUTED, NOT LISTED BY HAND. The dispatch named
three files. Three is not the set: `da_forward_day_verify.py` and
`da_blackout_mask.py` import `pm_tape_density`, `da_race_withdrawals` and
`da_content_liveness_rule` AT MODULE LEVEL, so those load on EVERY run and a
change in them changes every verdict. Hand-listing three would have left that
hole in the very instrument built to close it. So:

  * REFUSE tier -- the shell, this guard, the two Python entry points, and the
    transitive closure of their MODULE-LEVEL imports inside the repo. This set
    is loaded on every run without exception. Drift here => rc 7, nothing runs.
  * REPORT tier -- the rest of the static import closure, reachable only
    through function-local imports, i.e. conditionally. Drift here is NAMED in
    the check's output and does not stop the night (rule 4: a status, never a
    silent drop). Making it fatal would couple DA's nightly duty to DE's and
    BE's daily edits of `policy_optimizer*` and `adverse_move_*`.

THE TIERS ARE A RECORDED PROPERTY OF THE RECORD, not a hidden judgement: every
file carries its tier and the reason it has that tier, so a reader can see
what would have refused and what would only have been reported.

WHAT THIS CANNOT DO, NAMED BEFORE ANYONE HAS TO FIND IT:
  1. THE RECORD LIVES IN THE TREE IT CERTIFIES. A wholesale swap of the tree
     swaps the record with it. The partial answer is the git anchor below --
     the record on disk is compared to its own last-committed blob -- but that
     anchor is REPORTED, NOT FATAL, and the reason is written at the check
     itself: a fresh deploy awaiting its landing commit is byte-for-byte the
     same shape as a hand edit. So a hand-edited record is NAMED, not refused.
     A rewritten git object store is neither.
  2. THIS GUARD CANNOT BE ITS OWN CONTROL. A tampered guard would pass itself.
     That is why the shell independently digests the REFUSE tier with
     `sha256sum -c` -- coreutils, a different implementation -- BEFORE this
     file is invoked, and this file's own digest is in that manifest.
  3. It pins FILES, not the interpreter's resolution of them. `da_forward_day_
     verify.py` run under a stale `__pycache__` would still pass here. That is
     the B-1 class and it is closed on the mutation-audit side, not here.
  4. A drift refusal COSTS THAT NIGHT'S VERDICT. Verified affordable, not
     assumed: `days_needing_verdict` (da_forward_day_verify.py:97) derives the
     day list FROM DISK, floored at the earliest existing verdict, and returns
     any day inside the range with no artifact (`catchup_absent`) or one
     written while the day was open (`catchup_was_open`). So a refused night
     is recovered by the next successful run, with a late `as_of` and a
     catch-up reason in its own artifact.

    python3 live/pm_research/da_deploy_guard.py --selftest
    python3 live/pm_research/da_deploy_guard.py compute-set --tree T
    python3 live/pm_research/da_deploy_guard.py check --record R [--tree T]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROTOCOL = "P003_DA_DEPLOY_RECORD_V1"
RECORD_SCHEMA = 1

#: The unit's exit codes, enumerated HERE so the next person to add one can
#: see what is taken. Read off `da_midnight_verify.sh`; the selftest asserts
#: this table still matches that file, so it cannot rot into a comment.
UNIT_EXIT_CODES = {
    0: "clean -- every day needing a verdict got one and every mask landed",
    2: "OPEN_DAY_MASK_DEFERRED -- expected status; SuccessExitStatus=2",
    3: "cd into the deploy tree failed",
    4: "INSTRUMENT FAILURE -- nothing verified, or a mask did not land",
    5: "pair-guard refusal (LOG/OUTDIR) or verifier substitution",
    6: "canonical write by a run that names itself as neither leg",
    7: "DEPLOY_DRIFT -- the executed bytes are not the deployed bytes",
}
DRIFT_RC = 7
INSTRUMENT_RC = 4

#: Relative to the deploy tree. The shell the unit's ExecStart names, and this
#: guard, which the shell invokes. Neither is a Python module, so neither is
#: reachable by the import walk; both are executed on every run.
FIXED_REFUSE = (
    "live/pm_research/da_midnight_verify.sh",
    "live/pm_research/da_deploy_guard.py",
)
#: The two programs the shell runs with `$PY`. The import walk starts here.
ENTRY_MODULES = ("da_forward_day_verify", "da_blackout_mask")
MODULE_DIR = "live/pm_research"

#: Recorded and checked as REFUSE tier, but they live outside the tree.
INSTALLED_UNITS = (
    "~/.config/systemd/user/da-midnight-verify.service",
    "~/.config/systemd/user/da-midnight-verify.timer",
)
#: Recorded as REPORT tier: outside the repo, under no commit, and a change to
#: it is a fact a reader should see rather than a reason to lose a night.
INTERPRETER = "/home/yuqing/pricer-sol/venv/bin/python3"


class DeployRefused(RuntimeError):
    """The record is absent, unreadable, or cannot be applied."""


def digest_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def digest_file(p: Path) -> str:
    return digest_bytes(p.read_bytes())


# --------------------------------------------------------------------------
# THE FILE SET, COMPUTED
# --------------------------------------------------------------------------
def _imports(node: ast.AST) -> list[str]:
    out: list[str] = []
    if isinstance(node, ast.Import):
        out += [a.name.split(".")[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        # `from . import x` (level > 0) is not a repo-flat module name.
        if node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
    return out


def module_level_imports(src: str) -> list[str]:
    """Names imported when the module is LOADED, not when a function runs.

    Top-level statements, plus imports nested inside top-level `try`/`if`
    blocks -- a guarded import at module level still executes on load, and
    treating it as conditional would put it in the wrong tier.
    """
    out: list[str] = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out += _imports(n)
        elif isinstance(n, (ast.Try, ast.If, ast.With)):
            for sub in ast.walk(n):
                out += _imports(sub)
    return out


def all_imports(src: str) -> list[str]:
    out: list[str] = []
    for n in ast.walk(ast.parse(src)):
        out += _imports(n)
    return out


def _closure(tree: Path, seeds: tuple[str, ...], top_only: bool) -> set[str]:
    mdir = tree / MODULE_DIR
    seen: set[str] = set()
    stack = list(seeds)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        p = mdir / f"{m}.py"
        if not p.is_file():
            continue          # stdlib or third-party: not ours to pin
        seen.add(m)
        src = p.read_text()
        stack += (module_level_imports(src) if top_only else all_imports(src))
    return seen


def compute_file_set(tree: Path) -> dict:
    """REFUSE and REPORT tiers, derived from the source, never hand-listed."""
    tree = Path(tree).resolve()
    mdir = tree / MODULE_DIR
    missing = [f for f in FIXED_REFUSE if not (tree / f).is_file()]
    for m in ENTRY_MODULES:
        if not (mdir / f"{m}.py").is_file():
            missing.append(f"{MODULE_DIR}/{m}.py")
    if missing:
        raise DeployRefused(
            f"REFUSED: cannot compute the deployed set, absent from {tree}: "
            f"{missing}")

    top = _closure(tree, ENTRY_MODULES, top_only=True)
    full = _closure(tree, ENTRY_MODULES, top_only=False)
    files = []
    for rel in FIXED_REFUSE:
        p = tree / rel
        files.append({
            "path": rel, "tier": "REFUSE",
            "sha256": digest_file(p), "bytes": p.stat().st_size,
            "why": "executed directly by the unit on every run "
                   "(ExecStart, or invoked by it before any write)"})
    for m in sorted(top):
        p = mdir / f"{m}.py"
        files.append({
            "path": f"{MODULE_DIR}/{m}.py", "tier": "REFUSE",
            "sha256": digest_file(p), "bytes": p.stat().st_size,
            "why": ("entry point run by the unit"
                    if m in ENTRY_MODULES else
                    "imported AT MODULE LEVEL by an entry point, so it is "
                    "loaded on every run without exception")})
    for m in sorted(full - top):
        p = mdir / f"{m}.py"
        files.append({
            "path": f"{MODULE_DIR}/{m}.py", "tier": "REPORT",
            "sha256": digest_file(p), "bytes": p.stat().st_size,
            "why": "in the static import closure but reachable only through "
                   "a function-local import, so it runs conditionally"})
    return {
        "files": files,
        "n_refuse": sum(1 for f in files if f["tier"] == "REFUSE"),
        "n_report": sum(1 for f in files if f["tier"] == "REPORT"),
        "how_derived": {
            "entry_modules": list(ENTRY_MODULES),
            "fixed": list(FIXED_REFUSE),
            "refuse_rule": "FIXED plus the transitive closure of MODULE-LEVEL "
                           "imports of the entry modules, restricted to "
                           f"{MODULE_DIR}/*.py",
            "report_rule": "the rest of the transitive closure of ALL imports "
                           "(including function-local ones)",
            "not_hand_listed": True,
        },
    }


# --------------------------------------------------------------------------
# THE RECORD
# --------------------------------------------------------------------------
def _git(tree: Path, *args: str) -> tuple[int, str]:
    r = subprocess.run(["git", "-C", str(tree), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def build_record(tree: Path, by: str, at_utc: str,
                 units: tuple[str, ...] = INSTALLED_UNITS,
                 interpreter: str = INTERPRETER) -> dict:
    tree = Path(tree).resolve()
    fs = compute_file_set(tree)
    rc_h, head = _git(tree, "rev-parse", "HEAD")
    _, origin = _git(tree, "rev-parse", "origin/mm-research")
    _, behind = _git(tree, "rev-list", "--count", "HEAD..origin/mm-research")
    _, ahead = _git(tree, "rev-list", "--count", "origin/mm-research..HEAD")
    _, dirty = _git(tree, "status", "--porcelain", "--", MODULE_DIR)

    unit_rows = []
    for u in units:
        p = Path(os.path.expanduser(u))
        unit_rows.append({
            "path": str(p), "tier": "REFUSE",
            "exists": p.is_file(),
            "sha256": digest_file(p) if p.is_file() else None,
            "why": "the installed unit/timer is what systemd actually runs; "
                   "an edit here repoints or reconfigures the night without "
                   "touching the repo at all"})
    ip = Path(interpreter)
    real = ip.resolve() if ip.exists() else ip
    return {
        "protocol": PROTOCOL,
        "record_schema": RECORD_SCHEMA,
        "what_this_is": (
            "the DEPLOY RECORD for da-midnight-verify.service. It is written "
            "ONLY by da_deploy_midnight.sh, an explicit act. The unit "
            "recomputes these digests at 00:06Z BEFORE anything is written "
            "and refuses with rc 7 if a REFUSE-tier digest differs."),
        "deploy_tree": str(tree),
        "deployed_commit": head if rc_h == 0 else "UNKNOWN",
        "origin_mm_research_at_deploy": origin,
        "commits_behind_origin_at_deploy": int(behind or 0),
        "commits_ahead_of_origin_at_deploy": int(ahead or 0),
        "module_dir_dirty_at_deploy": bool(dirty),
        "deployed_at_utc": at_utc,
        "deployed_by": by,
        "deploy_host": os.uname().nodename,
        "drift_exit_code": DRIFT_RC,
        "unit_exit_codes": {str(k): v for k, v in UNIT_EXIT_CODES.items()},
        "why_not_exit_5": (
            "5 is already taken twice in da_midnight_verify.sh (the "
            "LOG/OUTDIR pair guard and the verifier-substitution refusal) and "
            "6 by the unnamed-canonical-write refusal. Reusing one would make "
            "two different refusals indistinguishable in systemctl."),
        "commit_is_provenance_not_a_gate": (
            "a HEAD move alone is NOT drift. Pinning the commit would refuse "
            "nearly every night -- the register is appended several times a "
            "day -- while not one executed byte changes. The gate is the "
            "digests; the commit says where they came from."),
        "installed_units": unit_rows,
        "interpreter": {
            "path": str(ip), "realpath": str(real), "tier": "REPORT",
            "exists": real.is_file(),
            "sha256": digest_file(real) if real.is_file() else None,
            "why": "outside the repo and under no commit. Recorded so a "
                   "reader can see it changed; not fatal, because losing a "
                   "night's verdict to a distro python update is the wrong "
                   "trade and the change is visible either way."},
        **fs,
    }


def manifest_text(record: dict) -> str:
    """`sha256sum -c` format, REFUSE-tier repo files only, tree-relative.

    Deliberately a SECOND artifact in a format coreutils validates, so the
    shell's first gate does not depend on this file's own correctness. Paths
    are relative so the check can be relocated onto a different tree, which is
    what the falsifier needs.
    """
    lines = [f"{f['sha256']}  {f['path']}"
             for f in record["files"] if f["tier"] == "REFUSE"]
    return "".join(f"{ln}\n" for ln in sorted(lines, key=lambda s: s[66:]))


# --------------------------------------------------------------------------
# THE CHECK
# --------------------------------------------------------------------------
def check(record_path: Path, tree: Path | None = None) -> dict:
    record_path = Path(record_path)
    if not record_path.is_file():
        return {"verdict": "DRIFT", "reason": "RECORD_ABSENT",
                "record": str(record_path), "rc": DRIFT_RC,
                "detail": "no deploy record: this unit has never been "
                          "deployed by the explicit act, so nothing says what "
                          "it is supposed to be running"}
    try:
        rec = json.loads(record_path.read_text())
    except Exception as e:                                   # noqa: BLE001
        return {"verdict": "DRIFT", "reason": "RECORD_UNPARSEABLE",
                "record": str(record_path), "rc": DRIFT_RC, "detail": repr(e)}
    if rec.get("protocol") != PROTOCOL or rec.get("record_schema") != \
            RECORD_SCHEMA:
        return {"verdict": "DRIFT", "reason": "RECORD_WRONG_PROTOCOL",
                "record": str(record_path), "rc": DRIFT_RC,
                "detail": f"{rec.get('protocol')} / {rec.get('record_schema')}"
                          f" != {PROTOCOL} / {RECORD_SCHEMA}"}

    t = Path(tree).resolve() if tree else Path(rec["deploy_tree"])
    refuse_bad, report_bad, absent = [], [], []
    for f in rec.get("files", []):
        p = t / f["path"]
        if not p.is_file():
            absent.append(f["path"])
            (refuse_bad if f["tier"] == "REFUSE" else report_bad).append(
                {"path": f["path"], "expected": f["sha256"], "found": None,
                 "how": "ABSENT"})
            continue
        got = digest_file(p)
        if got != f["sha256"]:
            (refuse_bad if f["tier"] == "REFUSE" else report_bad).append(
                {"path": f["path"], "expected": f["sha256"], "found": got,
                 "how": "CONTENT_DIFFERS"})

    unit_bad = []
    for u in rec.get("installed_units", []):
        p = Path(u["path"])
        if not p.is_file():
            unit_bad.append({"path": u["path"], "how": "ABSENT",
                             "expected": u.get("sha256"), "found": None})
        else:
            got = digest_file(p)
            if got != u.get("sha256"):
                unit_bad.append({"path": u["path"], "how": "CONTENT_DIFFERS",
                                 "expected": u.get("sha256"), "found": got})

    interp = rec.get("interpreter") or {}
    interp_drift = None
    ipath = Path(interp.get("realpath") or interp.get("path") or "")
    if interp.get("sha256"):
        if not ipath.is_file():
            interp_drift = {"path": str(ipath), "how": "ABSENT"}
        elif digest_file(ipath) != interp["sha256"]:
            interp_drift = {"path": str(ipath), "how": "CONTENT_DIFFERS",
                            "expected": interp["sha256"],
                            "found": digest_file(ipath)}

    # THE GIT ANCHOR. The record lives in the tree it certifies, so a
    # hand-edited record would certify itself. Comparing the bytes on disk to
    # the blob at the LAST COMMIT THAT TOUCHED THIS PATH is the only outside
    # reference available. It does not catch a rewritten object store, and
    # does not pretend to.
    #
    # REPORTED, NOT FATAL -- AND I BUILT IT FATAL FIRST. My own falsifier's
    # POSITIVE control caught it on the first complete run: the deploy act
    # writes a new record over the landed one and then asks this function to
    # confirm it, so a perfectly correct fresh deploy reported
    # RECORD_EDITED_AFTER_COMMIT and refused itself. The three states are
    # genuinely different and only one of them is suspicious:
    #   MATCHES_LAST_COMMIT      -- landed and unmodified
    #   RECORD_UNCOMMITTED       -- never committed at this path
    #   DIFFERS_FROM_LAST_COMMIT -- AMBIGUOUS. A fresh deploy awaiting its
    #                               landing commit looks exactly like a hand
    #                               edit, and nothing in the bytes separates
    #                               them.
    # Making the ambiguous one fatal would mean the unit refuses in the window
    # between the deploy act and its landing commit -- a deploy at 23:59 with
    # the commit at 00:07 would lose the night. That is a LIKELY operational
    # cost paid against a REMOTE failure (a hand-edited record must also carry
    # digests that match the drifted files, or the REFUSE tier catches it
    # first). So it is named loudly in the line the unit logs and it does not
    # stop the night. The trade is stated rather than silently chosen.
    anchor: dict = {"status": "NOT_CHECKED"}
    try:
        rel = os.path.relpath(record_path.resolve(), t)
        rc1, last = _git(t, "log", "-1", "--format=%H", "--", rel)
        if rc1 != 0:
            anchor = {"status": "GIT_UNAVAILABLE"}
        elif not last:
            anchor = {"status": "RECORD_UNCOMMITTED",
                      "note": "the record has never been committed at this "
                              "path. Legitimate between the deploy act and "
                              "its landing commit; reported, not fatal."}
        else:
            r = subprocess.run(["git", "-C", str(t), "show", f"{last}:{rel}"],
                               capture_output=True)
            same = (r.returncode == 0
                    and digest_bytes(r.stdout) == digest_file(record_path))
            anchor = {"status": "MATCHES_LAST_COMMIT" if same
                      else "DIFFERS_FROM_LAST_COMMIT",
                      "commit": last, "path": rel,
                      "means": ("the record on disk is the one that was "
                                "landed" if same else
                                "the record on disk is NOT the landed one: "
                                "either a fresh deploy awaiting its landing "
                                "commit, or a hand edit. Reported, never "
                                "fatal -- see the note at this function.")}
    except Exception as e:                                   # noqa: BLE001
        anchor = {"status": "GIT_UNAVAILABLE", "detail": repr(e)}

    fatal = bool(refuse_bad) or bool(unit_bad)
    _, head_now = _git(t, "rev-parse", "HEAD")
    return {
        "verdict": "DRIFT" if fatal else "MATCH",
        "reason": ("REFUSE_TIER_DRIFT" if refuse_bad else
                   "INSTALLED_UNIT_DRIFT" if unit_bad else "NONE"),
        "rc": DRIFT_RC if fatal else 0,
        "record": str(record_path),
        "tree": str(t),
        "deployed_commit": rec.get("deployed_commit"),
        "head_now": head_now,
        "head_moved_since_deploy": head_now != rec.get("deployed_commit"),
        "head_move_is_not_drift": True,
        "deployed_at_utc": rec.get("deployed_at_utc"),
        "deployed_by": rec.get("deployed_by"),
        "n_refuse_checked": sum(1 for f in rec.get("files", [])
                                if f["tier"] == "REFUSE"),
        "n_report_checked": sum(1 for f in rec.get("files", [])
                                if f["tier"] == "REPORT"),
        "refuse_tier_drift": refuse_bad,
        "report_tier_drift": report_bad,
        "installed_unit_drift": unit_bad,
        "interpreter_drift": interp_drift,
        "record_git_anchor": anchor,
        "absent_files": absent,
    }


def render(v: dict) -> str:
    if v["verdict"] == "MATCH":
        extra = ""
        if v.get("report_tier_drift"):
            extra = (f"; {len(v['report_tier_drift'])} REPORT-tier file(s) "
                     f"differ and are NAMED, not fatal: "
                     f"{[d['path'] for d in v['report_tier_drift']]}")
        if v.get("interpreter_drift"):
            extra += "; the INTERPRETER differs from the record (REPORT tier)"
        a = v.get("record_git_anchor", {}).get("status")
        if a != "MATCHES_LAST_COMMIT":
            extra += (f"; RECORD GIT ANCHOR = {a} -- the record on disk is "
                      f"NOT the landed blob (fresh deploy awaiting its commit, "
                      f"or a hand edit). Named, not fatal")
        return (f"DEPLOY_RECORD MATCH: {v['n_refuse_checked']} REFUSE-tier "
                f"file(s) identical to the record deployed "
                f"{v['deployed_at_utc']} at {v['deployed_commit'][:12]} by "
                f"{v['deployed_by']}. HEAD is now {v['head_now'][:12]} "
                f"({'moved' if v['head_moved_since_deploy'] else 'unmoved'}; "
                f"a HEAD move is NOT drift){extra}")
    bits = [f"DEPLOY_DRIFT ({v['reason']}), rc {DRIFT_RC}: "
            f"the unit will NOT run."]
    for d in v.get("refuse_tier_drift", []):
        bits.append(f"  REFUSE {d['path']}: {d['how']} "
                    f"(record {str(d['expected'])[:16]}, "
                    f"disk {str(d['found'])[:16]})")
    for d in v.get("installed_unit_drift", []):
        bits.append(f"  UNIT {d['path']}: {d['how']}")
    if v.get("detail"):
        bits.append(f"  {v['detail']}")
    bits.append("  FIX: re-run the deploy act, which is the ONLY writer of "
                "the record -- live/pm_research/da_deploy_midnight.sh. "
                "The missed night is recovered: days_needing_verdict derives "
                "the day list from disk and returns this day as catchup.")
    return "\n".join(bits)


# --------------------------------------------------------------------------
# FALSIFIERS, BOTH DIRECTIONS
# --------------------------------------------------------------------------
def _fake_tree(root: Path) -> Path:
    """A minimal tree with the shape the guard walks: two entry modules, one
    module-level import, one function-local import."""
    m = root / MODULE_DIR
    m.mkdir(parents=True)
    (root / "live/pm_research/systemd").mkdir(parents=True, exist_ok=True)
    (m / "da_forward_day_verify.py").write_text(
        "import pm_tape_density\n"
        "def go():\n    import policy_optimizer\n    return 1\n")
    (m / "da_blackout_mask.py").write_text(
        "import da_content_liveness_rule\ndef go():\n    return 2\n")
    (m / "pm_tape_density.py").write_text("V = 1\n")
    (m / "da_content_liveness_rule.py").write_text("V = 2\n")
    (m / "policy_optimizer.py").write_text("V = 3\n")
    (m / "da_midnight_verify.sh").write_text("#!/bin/bash\nexit 0\n")
    (m / "da_deploy_guard.py").write_text("# stand-in\n")
    return root


def selftest() -> int:                                       # noqa: C901
    import tempfile
    fails: list[str] = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    with tempfile.TemporaryDirectory() as d:
        root = _fake_tree(Path(d) / "tree")
        fs = compute_file_set(root)
        tiers = {f["path"].split("/")[-1]: f["tier"] for f in fs["files"]}
        ok(tiers.get("pm_tape_density.py") == "REFUSE"
           and tiers.get("da_content_liveness_rule.py") == "REFUSE",
           "TIERING: a MODULE-LEVEL import of an entry point is REFUSE tier "
           "-- it is loaded on every run, and hand-listing only the entry "
           "points would have left exactly this hole")
        ok(tiers.get("policy_optimizer.py") == "REPORT",
           "TIERING: a FUNCTION-LOCAL import is REPORT tier -- conditional, "
           "so it is named on drift and does not cost a night's verdict")
        ok(tiers.get("da_midnight_verify.sh") == "REFUSE"
           and tiers.get("da_deploy_guard.py") == "REFUSE",
           "TIERING: the shell and this guard are REFUSE tier though neither "
           "is reachable by the import walk")

        rec = build_record(root, by="selftest", at_utc="1970-01-01T00:00:00Z")
        recp = root / "live/pm_research/systemd/da_deploy_record.json"
        recp.write_text(json.dumps(rec, indent=2, sort_keys=True))
        manp = recp.with_suffix(".sha256")
        manp.write_text(manifest_text(rec))

        v = check(recp, tree=root)
        ok(v["verdict"] == "MATCH" and v["rc"] == 0,
           "POSITIVE CONTROL: a clean deploy ADMITS. A guard shown only to "
           "refuse has not been shown to pass (SEAT_PROTOCOL 16)")

        # KNOWN-BAD 1: drift in a REFUSE-tier file that is NOT an entry point.
        (root / MODULE_DIR / "pm_tape_density.py").write_text("V = 99\n")
        v = check(recp, tree=root)
        ok(v["verdict"] == "DRIFT" and v["rc"] == DRIFT_RC
           and v["reason"] == "REFUSE_TIER_DRIFT"
           and any(x["path"].endswith("pm_tape_density.py")
                   for x in v["refuse_tier_drift"]),
           "KNOWN-BAD: drift in a MODULE-LEVEL import REFUSES with rc 7 and "
           "NAMES the file -- the hole a three-file hand list would have left")
        (root / MODULE_DIR / "pm_tape_density.py").write_text("V = 1\n")

        # KNOWN-BAD 2: drift in the ExecStart shell itself.
        (root / MODULE_DIR / "da_midnight_verify.sh").write_text("#!/bin/bash\nexit 1\n")
        v = check(recp, tree=root)
        ok(v["verdict"] == "DRIFT" and any(
            x["path"].endswith("da_midnight_verify.sh")
            for x in v["refuse_tier_drift"]),
           "KNOWN-BAD: drift in the ExecStart script itself REFUSES")
        (root / MODULE_DIR / "da_midnight_verify.sh").write_text("#!/bin/bash\nexit 0\n")

        # KNOWN-BAD 3: an absent REFUSE-tier file.
        (root / MODULE_DIR / "da_content_liveness_rule.py").unlink()
        v = check(recp, tree=root)
        ok(v["verdict"] == "DRIFT" and v["absent_files"],
           "KNOWN-BAD: an ABSENT REFUSE-tier file REFUSES -- absence is never "
           "a pass (rule 11)")
        (root / MODULE_DIR / "da_content_liveness_rule.py").write_text("V = 2\n")

        # POSITIVE CONTROL 2: REPORT-tier drift must NOT refuse, and must be
        # named. A tier that behaved like REFUSE would make the tiering a lie.
        (root / MODULE_DIR / "policy_optimizer.py").write_text("V = 44\n")
        v = check(recp, tree=root)
        ok(v["verdict"] == "MATCH" and v["rc"] == 0
           and any(x["path"].endswith("policy_optimizer.py")
                   for x in v["report_tier_drift"]),
           "POSITIVE CONTROL: REPORT-tier drift is NAMED and does NOT refuse "
           "-- a status, not a silent drop (rule 4)")
        ok("policy_optimizer.py" in render(v),
           "REPORT-tier drift appears in the rendered line the unit logs, so "
           "it cannot be seen only by a JSON reader")
        (root / MODULE_DIR / "policy_optimizer.py").write_text("V = 3\n")

        # KNOWN-BAD 4: the record itself is absent / unparseable / foreign.
        ok(check(root / "nope.json", tree=root)["reason"] == "RECORD_ABSENT",
           "KNOWN-BAD: NO record at all REFUSES -- never deployed by the act "
           "is not the same as deployed and unchanged")
        bad = root / "bad.json"
        bad.write_text("{not json")
        ok(check(bad, tree=root)["reason"] == "RECORD_UNPARSEABLE",
           "KNOWN-BAD: an unparseable record REFUSES")
        bad.write_text(json.dumps({"protocol": "SOMETHING_ELSE"}))
        ok(check(bad, tree=root)["reason"] == "RECORD_WRONG_PROTOCOL",
           "KNOWN-BAD: a record of another protocol/schema REFUSES rather "
           "than being read for the fields it happens to share")

        # KNOWN-BAD 5: the installed unit edited without touching the repo.
        u = Path(d) / "unit.service"
        u.write_text("[Service]\nExecStart=/bin/true\n")
        rec2 = build_record(root, by="s", at_utc="1970-01-01T00:00:00Z",
                            units=(str(u),))
        r2 = root / "rec2.json"
        r2.write_text(json.dumps(rec2))
        ok(check(r2, tree=root)["verdict"] == "MATCH",
           "POSITIVE CONTROL: an unchanged installed unit ADMITS")
        u.write_text("[Service]\nExecStart=/bin/false\n")
        v = check(r2, tree=root)
        ok(v["verdict"] == "DRIFT" and v["reason"] == "INSTALLED_UNIT_DRIFT",
           "KNOWN-BAD: the INSTALLED unit edited -- repointing ExecStart "
           "without touching the repo -- REFUSES")

        # THE MANIFEST IS THE SHELL'S INDEPENDENT GATE: it must agree with the
        # record, and coreutils must accept it.
        r = subprocess.run(["sha256sum", "-c", "--status", str(manp)],
                           cwd=str(root), capture_output=True)
        ok(r.returncode == 0,
           "MANIFEST: `sha256sum -c` accepts the manifest on a clean tree -- "
           "the shell's first gate is coreutils, a different implementation "
           "from this file, and this file's own digest is in it")
        ok(any(ln.endswith("live/pm_research/da_deploy_guard.py")
               for ln in manifest_text(rec).splitlines()),
           "MANIFEST: the guard's OWN digest is in the manifest, because a "
           "guard cannot be its own control")
        (root / MODULE_DIR / "pm_tape_density.py").write_text("V = 99\n")
        r = subprocess.run(["sha256sum", "-c", "--status", str(manp)],
                           cwd=str(root), capture_output=True)
        ok(r.returncode != 0,
           "MANIFEST KNOWN-BAD: `sha256sum -c` REJECTS a drifted tree")
        (root / MODULE_DIR / "pm_tape_density.py").write_text("V = 1\n")

        # THE GIT ANCHOR, driven in a real repository.
        subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
        subprocess.run(["git", "config", "user.email", "da@x"], cwd=str(root))
        subprocess.run(["git", "config", "user.name", "da"], cwd=str(root))
        subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
        subprocess.run(["git", "commit", "-qm", "r"], cwd=str(root), check=True)
        v = check(recp, tree=root)
        ok(v["record_git_anchor"]["status"] == "MATCHES_LAST_COMMIT"
           and v["verdict"] == "MATCH",
           "GIT ANCHOR: a committed, unedited record ADMITS and reports "
           "MATCHES_LAST_COMMIT")
        blob = recp.read_text()
        recp.write_text(blob.replace('"deployed_by": "selftest"',
                                     '"deployed_by": "someone else"'))
        v = check(recp, tree=root)
        ok(v["record_git_anchor"]["status"] == "DIFFERS_FROM_LAST_COMMIT"
           and "RECORD GIT ANCHOR" in render(v),
           "GIT ANCHOR KNOWN-BAD: a record whose bytes differ from the landed "
           "blob is NAMED in the line the unit logs -- and is NOT fatal, "
           "because a fresh deploy awaiting its landing commit is that same "
           "shape and making it fatal loses the night (my own positive "
           "control caught exactly that)")
        recp.write_text(blob)
        ok(check(recp, tree=root)["record_git_anchor"]["status"]
           == "MATCHES_LAST_COMMIT",
           "GIT ANCHOR: restoring the record's bytes reports "
           "MATCHES_LAST_COMMIT again -- the three states are distinguished, "
           "not collapsed")

    # THE EXIT-CODE TABLE MUST STILL MATCH THE SHELL. A comment that drifts
    # from the code it describes is how 5 would get reused.
    sh = Path(__file__).resolve().parent / "da_midnight_verify.sh"
    if sh.is_file():
        body = sh.read_text()
        used = {int(t.split()[1]) for t in
                __import__("re").findall(r"exit [0-9]+", body)}
        ok(used <= set(UNIT_EXIT_CODES),
           f"EXIT TABLE: every literal `exit N` in da_midnight_verify.sh "
           f"({sorted(used)}) is documented in UNIT_EXIT_CODES "
           f"{sorted(UNIT_EXIT_CODES)}")
        ok(DRIFT_RC in used,
           f"WIRED: the shell actually exits {DRIFT_RC} somewhere -- a guard "
           f"the launcher never calls is rule 17's suite-green hole")
        ok("da_deploy_guard.py" in body,
           "WIRED: da_midnight_verify.sh invokes this guard by name")
    else:
        ok(False, f"EXIT TABLE: {sh} not found")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--selftest", action="store_true")
    p = sub.add_parser("compute-set")
    p.add_argument("--tree", type=Path, required=True)
    p = sub.add_parser("write-record")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--by", required=True)
    p.add_argument("--at-utc", required=True)
    p = sub.add_parser("check")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--tree", type=Path)
    p.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.selftest or a.cmd is None:
        if not a.selftest:
            ap.error("choose --selftest or a subcommand")
        return selftest()
    try:
        if a.cmd == "compute-set":
            print(json.dumps(compute_file_set(a.tree), indent=2,
                             sort_keys=True))
            return 0
        if a.cmd == "write-record":
            rec = build_record(a.tree, by=a.by, at_utc=a.at_utc)
            a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            a.manifest.write_text(manifest_text(rec))
            print(f"record: {a.out} ({rec['n_refuse']} REFUSE / "
                  f"{rec['n_report']} REPORT) at {rec['deployed_commit'][:12]}")
            return 0
        if a.cmd == "check":
            v = check(a.record, a.tree)
            print(json.dumps(v, indent=2, sort_keys=True) if a.json
                  else render(v))
            return int(v["rc"])
    except DeployRefused as e:
        print(str(e), file=sys.stderr)
        return DRIFT_RC
    except Exception as e:                                   # noqa: BLE001
        print(f"INSTRUMENT FAILURE in da_deploy_guard: {e!r}", file=sys.stderr)
        return INSTRUMENT_RC
    return INSTRUMENT_RC


if __name__ == "__main__":
    raise SystemExit(main())

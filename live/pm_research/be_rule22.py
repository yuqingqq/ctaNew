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
import subprocess
import sys
from pathlib import Path

LIVE_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[2])


class Rule22Refused(RuntimeError):
    """The code that produced this emit is not the code that ran."""


def _git(root: str, *args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", root, *args], capture_output=True,
                           text=True, timeout=60)
    except Exception:                                        # noqa: BLE001
        return None
    return r.stdout.strip() if r.returncode == 0 else None


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

    def head_state(self) -> dict:
        st = _git(self.worktree, "status", "--porcelain")
        return {"worktree": self.worktree,
                "head": _git(self.worktree, "rev-parse", "HEAD"),
                "dirty": bool(st) if st is not None else None,
                "dirty_paths": [x[3:] for x in (st or "").split("\n") if x][:20]}

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
            "worktree_was_dirty_at_import": hi.get("dirty"),
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

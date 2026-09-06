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


class HeavyRunRefused(RuntimeError):
    """This process is not entitled to run heavy work right now."""


LAUNCHER = Path(__file__).with_name("be_heavy_run.sh")
LOCK_CONFLICT_RC = 75


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
    return {"launcher": str(LAUNCHER), "problems": problems,
            "scanned": "the systemd-run invocation itself",
            "launch_line": src[:400],
            "form_is_correct": not problems,
            "lock_is_inside_the_unit": inside,
            "conflict_exit_code": LOCK_CONFLICT_RC}


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
        code, _, path = line[:2], line[2:3], line[3:]
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

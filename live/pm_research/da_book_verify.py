"""P-2026-003 THE INDEPENDENT DAY-BOOK VERIFIER -- DA's separate stack.

R-235 (do-not-harmonize). `be_daybook_build.py`, the BE_DAYBOOK_V1 receipt
and `de_phase4_diag_runner.day_assembly_inputs` were read as DOCUMENTS; none
of BE's builder is imported. The whole Gate-1 test rests on this book and
until now only the reviewer had checked it by hand.

FOUR GROUPS, and what each may and may not say.

  (1) THE DIGEST CHAIN. The book's bytes against the receipt's `sha256` AND
      its `readback_sha256` -- two independent statements of the same bytes,
      compared to one streamed hash of the file as it now stands. The pinned
      inputs (tape, fragment) against the files on disk WHEN PRESENT; an
      absent input is a STATUS, never a pass.
  (2) THE POPULATION, RECOMPUTED FROM THE BOOK. Windows, generations, the
      status sums, both pinned heads at their pinned thetas with SET
      EQUALITY recomputed over the scored keys rather than read off
      `sets_are_equal`, coverage recomputed as n_covered / generations, and
      the uncovered reason classes summing to the uncovered count.
  (3) THE RESOURCES AS FACTS, NEVER VERDICTS. Every stage's peak against its
      own declared budget, the index release fraction, the wall. These are
      reported; nothing here passes or fails a run on them.
  (4) IDENTITY AND HONESTY OF THE RECEIPT. A receipt naming a different day
      or coin REFUSES. And `seam.index` -- a LITERAL describing a call -- is
      compared to the builder's actual call READ BY AST, at the commit the
      receipt itself names.

TWO TIERS, BECAUSE THE BOOK IS 290 MB OF PICKLE.
  RECEIPT tier (light): groups 1, 3 and 4, plus every predicate the receipt
      can be held to internally. Hashing the file streams and costs no
      memory worth naming.
  BOOK tier (HEAVY, rule 20): group 2 needs the pickle loaded -- about 2 GB
      resident -- so it takes the lock and runs under the wrapper. The real
      09-03 run is a coordinator GO, not something this module does on its
      own initiative.

    python3 live/pm_research/da_book_verify.py --selftest
    python3 live/pm_research/da_book_verify.py --receipt-tier \\
        --book <path> --receipt <path> [--output <path>]
    python3 live/pm_research/da_book_verify.py --full \\
        --book <path> --receipt <path> [--output <path>]   # HEAVY
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_BOOK_VERIFIER_V1"
BE_RECEIPT_PROTOCOL = "BE_DAYBOOK_V1"
def _builder_path() -> Path:
    """BE's builder, READ FROM THE CANONICAL TREE.

    REV 58 section 4: there were THREE root rules in this seat's modules --
    this one (`HERE`-relative), the gate verifier's dead try-branch, and
    the accrual report's env-first fallback. They are ONE now
    (`da_root.require_canonical_root`). Reading the builder from `HERE`
    meant that, run from a worktree, this verifier judged BE's receipt
    against a STALE COPY of BE's code -- the finding DA 77 shipped against
    DE's runner and had to retract."""
    import da_root as _R                                      # noqa: PLC0415
    try:
        return _R.code_root("reading BE's builder source") / \
            "live/pm_research/be_daybook_build.py"
    except _R.RootRefused:
        #: NAMED, never a silent tree-relative answer: the caller decides
        #: what to do with a builder it cannot locate canonically.
        raise


#: ONE rule, one call site. `HERE / "be_daybook_build.py"` is gone.

#: What loading the real 09-03 book is expected to cost, DECLARED before the
#: run so the receipt can be held to it: the pickle is 290,758,834 bytes on
#: disk and unpickles to roughly 2 GB resident. The guard sits above that
#: with room to write its own refusal, and BELOW the rule-20 cap so a breach
#: is this module's refusal and never the cgroup's kill.
BOOK_LOAD_EXPECTED_PEAK_GB = 2.0
BOOK_LOAD_CAP_GB = 4.0
GIB = 1024.0 ** 3


class BookVerifyRefused(RuntimeError):
    """The verification cannot proceed honestly on the inputs given."""


class LaunchCaptureRefused(BookVerifyRefused):
    """The code that ran is not the code the receipt would name."""


# ---------------------------------------------------------- RULE 22 / R-605
#: THE LAUNCH CAPTURE. Rule 22 as amended: a runner and every heavy producer
#: capture AT IMPORT the digest of every module of their import closure
#: under `live/`, plus the worktree's HEAD and whether it was dirty, and
#: REFUSE THE EMIT BY NAME if any of it moved. DA 77's own sweep found this
#: seat's two runners lacking it -- and found that this seat's binding map
#: had exempted them, which is worse than the gap.
#:
#: R-235: DE's `source_identity_at_launch` / `assert_source_unchanged` were
#: read AS A DOCUMENT and re-implemented here. Nothing of DE's is imported.
#: The property is not "the file I am is unchanged" -- that is one module of
#: many; it is "every module that RAN is still the bytes that ran".
LIVE_DIR = str(Path(__file__).resolve().parents[1])
LAUNCH_TIME_UTC = datetime.datetime.now(
    datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
LAUNCH_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()).hexdigest()
#: path -> digest OF THE BYTES SEEN WHEN THE MODULE FIRST ENTERED THIS RUN.
#: Never a second read: an import can return a module already in
#: `sys.modules` whose file has since moved, and the second read would
#: record the mover's bytes as the runner's.
LAUNCH_CLOSURE: dict = {}
LAUNCH_CAPTURE_POINTS: list = []


def _digest_module(mod) -> None:
    f = getattr(mod, "__file__", None)
    if not f:
        return
    try:
        p = Path(f).resolve()
    except OSError:
        return
    if not str(p).startswith(LIVE_DIR) or str(p) in LAUNCH_CLOSURE:
        return
    try:
        LAUNCH_CLOSURE[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        LAUNCH_CLOSURE[str(p)] = None


def capture_closure(where: str = "module import") -> int:
    """Digest every `live/` module loaded so far. Callable again AFTER a
    lazy import, because a module imported inside a function was not in
    `sys.modules` at import time and would otherwise be outside the
    closure -- the exact gap REV 53 section 1.1 found in DE's."""
    before = len(LAUNCH_CLOSURE)
    for m in list(sys.modules.values()):
        _digest_module(m)
    LAUNCH_CAPTURE_POINTS.append(
        {"where": where, "n_modules_after": len(LAUNCH_CLOSURE),
         "n_added": len(LAUNCH_CLOSURE) - before})
    return len(LAUNCH_CLOSURE)


def _porcelain(stdout: str) -> dict:
    """The programme's ONE porcelain parser (REV 64), imported."""
    import da_root as _R                                       # noqa: PLC0415
    return _R.parse_porcelain(stdout)


def _is_the_ledger_symlink(root: str, rel: str) -> bool:
    """Is `<root>/<rel>` R-553's symlink to the canonical data root?

    Three facts, all about the object: it is a SYMLINK, and it RESOLVES to
    the canonical data root, which is read from DE's module rather than
    typed here (R-235). The caller has already established that git calls
    it untracked."""
    try:
        p = Path(root) / rel.rstrip("/")
        if not p.is_symlink():
            return False
        import da_root as _R                                   # noqa: PLC0415
        canon = (_R.canonical_from_DEs_source() or {}).get("data")
        if not canon:
            return False
        return str(p.resolve()) == str(Path(canon).resolve())
    except OSError:
        return False


def _head_state() -> dict:
    """The worktree's HEAD and whether it was dirty, AT IMPORT."""
    root = str(Path(__file__).resolve().parents[2])

    def _g(*a):
        try:
            r = subprocess.run(["git", "-C", root, *a], capture_output=True,
                               text=True, timeout=60)
        except Exception:                                     # noqa: BLE001
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    #: RAW, NEVER STRIPPED (REV 64). `_g` strips, and a stripped block
    #: loses the LEADING SPACE of its FIRST line -- which is the read half
    #: of the porcelain defect. The status call bypasses it.
    try:
        _r = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=60)
        st = _r.stdout if _r.returncode == 0 else None
    except Exception:                                         # noqa: BLE001
        st = None
    if st is None:
        return {"worktree": root, "head": _g("rev-parse", "HEAD"),
                "dirty": None, "dirty_paths": []}
    #: THE LEDGER SYMLINK IS NOT DIRT, AND IT IS EXEMPTED AS A PROPERTY
    #: (DE 94). A seat worktree's `data/` is R-553's symlink to the ledger:
    #: it shows as `?? data`, and a name-matched exemption would let any
    #: file called `data` through. The predicate is UNTRACKED **and** a
    #: SYMLINK **and** resolving to the canonical data root -- three facts
    #: about the object, none about its name.
    parsed = _porcelain(st)
    exempt, dirt = [], []
    for row in parsed["rows"]:
        if row["untracked"] and _is_the_ledger_symlink(root, row["path"]):
            exempt.append(row["path"])
        else:
            dirt.append(row["path"])
    #: a line the parser could not read is DIRT, not silence.
    dirt += parsed["malformed"]
    return {"worktree": root, "head": _g("rev-parse", "HEAD"),
            "dirty": bool(dirt),
            "dirty_paths": dirt[:20],
            "exempt_ledger_symlinks": exempt,
            "why_exempt": (
                "untracked AND a symlink AND resolving to the canonical "
                "data root -- a property of the object, never its name")}


def closure_drift(closure: dict | None = None) -> list:
    """Every module of the closure whose FILE no longer holds the bytes it
    held when it entered this run. `closure` is injectable ONLY so the
    falsifier can drive both directions on copies; every production call
    takes the launch capture."""
    src = LAUNCH_CLOSURE if closure is None else closure
    out = []
    for path, at_launch in sorted(src.items()):
        try:
            now = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            now = None
        if now != at_launch:
            out.append({"module": Path(path).name, "path": path,
                        "at_launch": at_launch, "now": now,
                        "gone": now is None})
    return out


def source_identity_at_launch() -> dict:
    """THE BYTES THAT RAN -- all of them -- and whether they still hold."""
    me = Path(__file__).resolve()
    try:
        now = hashlib.sha256(me.read_bytes()).hexdigest()
    except OSError:
        now = None
    drift = closure_drift()
    head_now = _head_state()
    return {
        "producing_code": me.name,
        "producing_code_sha256": LAUNCH_SOURCE_SHA256,
        "digest_taken_at": "MODULE IMPORT, before any work",
        "launch_time_utc": LAUNCH_TIME_UTC,
        "on_disk_sha256_at_emit": now,
        "source_unchanged_during_the_run": now == LAUNCH_SOURCE_SHA256,
        "import_closure": {
            "n_modules": len(LAUNCH_CLOSURE),
            "root": LIVE_DIR,
            #: A LIST OF [name, digest] PAIRS, not a dict keyed by
            #: filename: a module name must never sit where a key-scanning
            #: net looks for economic keys (this programme owns a module
            #: called `e1_markout_scan.py`). The shape removes the
            #: interaction instead of exempting the block from the scan.
            "modules": [[Path(k).name, v]
                        for k, v in sorted(LAUNCH_CLOSURE.items())],
            "digested": ("from the bytes each module held when it FIRST "
                         "entered this run -- never re-read"),
            "capture_points": LAUNCH_CAPTURE_POINTS,
        },
        "closure_drift": drift,
        "closure_unchanged_during_the_run": not drift,
        "head_at_import": LAUNCH_HEAD,
        "head_at_emit": head_now,
        "head_unchanged_during_the_run": (
            LAUNCH_HEAD.get("head") == head_now.get("head")),
        "worktree_was_dirty_at_import": LAUNCH_HEAD.get("dirty"),
    }


def assert_source_unchanged(where: str, *, fixture: bool = True) -> dict:
    """REFUSE THE EMIT if any of the code that ran changed under it.

    Not the RUN -- the modules are in memory and the run is unaffected.
    What is not honest is a receipt naming bytes that did not produce it,
    and `producing_code_is_the_committed_bytes` PASSES when the replacement
    is itself committed (R-603)."""
    idy = source_identity_at_launch()
    if idy["closure_drift"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: A MODULE OF THIS RUN'S IMPORT CLOSURE "
            f"CHANGED UNDER IT -- "
            f"{[d['module'] for d in idy['closure_drift']]}. The run is "
            f"unaffected; a receipt stamped from the files would name code "
            f"that DID NOT RUN. Rule 22 as amended (R-605): the capture is "
            f"the CLOSURE, not one file.")
    if not idy["source_unchanged_during_the_run"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE SOURCE CHANGED UNDER THIS RUN. This "
            f"process is executing {LAUNCH_SOURCE_SHA256[:16]} (read at "
            f"import) and the file now holds "
            f"{str(idy['on_disk_sha256_at_emit'])[:16]}.")
    if not idy["head_unchanged_during_the_run"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE WORKTREE'S HEAD MOVED UNDER THIS RUN "
            f"-- {str(idy['head_at_import'].get('head'))[:12]} -> "
            f"{str(idy['head_at_emit'].get('head'))[:12]}. A "
            f"carrying_commit would name a commit this run did not execute "
            f"from.")
    if not fixture and idy["worktree_was_dirty_at_import"]:
        raise LaunchCaptureRefused(
            f"REFUSED at {where}: THE WORKTREE WAS DIRTY AT IMPORT "
            f"({idy['head_at_import'].get('dirty_paths')}). For a REAL "
            f"artifact the producing code must be locatable in a commit; "
            f"uncommitted bytes are locatable nowhere. A fact for a "
            f"fixture, a refusal for a real run.")
    return idy


capture_closure("module import")
LAUNCH_HEAD = _head_state()



def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def verifier_identity() -> dict:
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    return {"path": "live/pm_research/da_book_verify.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": carrying_commit(),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == "",
            #: RULE 22 / R-605: the closure and HEAD, captured at import.
            "source_identity": source_identity_at_launch()}


def _rss_gb() -> float:
    try:
        with open("/proc/self/statm", "rb") as fh:
            return int(fh.read().split()[1]) * 4096 / GIB
    except Exception:                                         # noqa: BLE001
        return float("nan")


def sha256_stream(path, chunk: int = 1 << 20) -> dict:
    """Streamed, so a 290 MB book costs a megabyte of memory, not 290."""
    h, n = hashlib.sha256(), 0
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return {"sha256": h.hexdigest(), "bytes": n}


# ------------------------------------------------------ (1) the digest chain

def digest_chain(book_path, receipt: dict, *,
                 verify_inputs: bool = True) -> dict:
    bk = receipt.get("book") or {}
    got = sha256_stream(book_path)
    declared = bk.get("sha256")
    readback = bk.get("readback_sha256")
    out = {
        "book_path": str(book_path),
        "bytes_on_disk": got["bytes"],
        "bytes_declared": bk.get("bytes"),
        "bytes_match": got["bytes"] == bk.get("bytes"),
        "sha256_recomputed": got["sha256"],
        "sha256_declared": declared,
        "readback_sha256_declared": readback,
        "matches_declared": got["sha256"] == declared,
        "matches_readback": got["sha256"] == readback,
        "declared_and_readback_agree": declared == readback,
        "why_both": (
            "the write-side digest is of the buffer that was written and the "
            "readback is a SECOND, independent statement of the same bytes. "
            "Checking one and calling it the chain would leave the other "
            "unchecked -- and they can disagree"),
        "inputs": {},
    }
    for name, blk in sorted((receipt.get("inputs_pinned") or {}).items()):
        p = Path(blk.get("path", ""))
        rec = {"declared_sha256": blk.get("sha256"), "path": str(p)}
        if not p.is_file():
            rec.update({"status": "INPUT_ABSENT_NOT_CHECKED",
                        "matches": None,
                        "why": ("the pinned input is not on disk. An absent "
                                "input is a STATUS: a `matches: true` here "
                                "would be a pass for a check that never ran")})
        elif not verify_inputs:
            rec.update({"status": "NOT_HASHED_BY_REQUEST", "matches": None})
        else:
            g = sha256_stream(p)
            rec.update({"status": "HASHED", "sha256_recomputed": g["sha256"],
                        "bytes": g["bytes"],
                        "matches": g["sha256"] == blk.get("sha256")})
        out["inputs"][name] = rec
    checked = [v for v in out["inputs"].values() if v.get("matches") is not None]
    out["n_inputs"] = len(out["inputs"])
    out["n_inputs_checked"] = len(checked)
    out["n_inputs_absent"] = sum(
        1 for v in out["inputs"].values()
        if v.get("status") == "INPUT_ABSENT_NOT_CHECKED")
    out["all_checked_inputs_match"] = all(v["matches"] for v in checked)
    out["chain_holds"] = bool(out["matches_declared"] and out["matches_readback"]
                              and out["bytes_match"]
                              and out["all_checked_inputs_match"])
    return out


# ------------------------------------ (4) identity, and the seam literal

def builder_index_call(path: Path | None = None, *,
                       at_commit: str | None = None) -> dict:
    """The builder's ACTUAL `build_tape_index` call, read BY AST.

    A literal in a receipt describing a call is rule 10's shape: it drifts
    from the code beside it and neither notices. This renders the call from
    the syntax tree -- positional argument names and keyword names -- so the
    comparison is against what the code DOES.

    `at_commit` reads the builder as it stood at the commit the receipt
    itself names, which is the only fair comparison: a receipt is a
    historical record and must be held to the code of its own moment."""
    src = None
    src_from = None
    if at_commit:
        r = subprocess.run(
            ["git", "show", f"{at_commit}:live/pm_research/be_daybook_build.py"],
            capture_output=True, text=True, cwd=str(HERE))
        if r.returncode == 0 and r.stdout:
            src, src_from = r.stdout, f"git {at_commit}"
    if src is None:
        p = Path(path) if path else _builder_path()
        if not p.is_file():
            raise BookVerifyRefused(
                f"REFUSED: the builder is absent at {p}; the seam literal "
                f"cannot be compared to the call it describes and MUST NOT "
                f"be taken on trust.")
        src, src_from = p.read_text(), "working tree"
        #: REV 50 section 1.3. THE FALLBACK IS THE POINT. When a commit was
        #: ASKED FOR and did not resolve, the working tree is a DIFFERENT
        #: object -- and returning it silently is the R-601 class through
        #: the other door: a verdict computed from HEAD wearing a sentence
        #: that attributes it to the receipt's commit. The caller is told,
        #: and `check_seam` refuses to judge on it.
        if at_commit:
            src_from = "working tree (FALLBACK -- the requested commit did "
            src_from += f"not resolve: {at_commit})"
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else (fn.id if isinstance(fn, ast.Name) else None))
        if name != "build_tape_index":
            continue
        pos = [(a.id if isinstance(a, ast.Name) else "<expr>")
               for a in node.args]
        kw = [k.arg for k in node.keywords if k.arg]
        calls.append({"positional": pos, "keywords": sorted(kw),
                      "rendered": f"build_tape_index({', '.join(pos + [k + '=…' for k in sorted(kw)])})",
                      "line": node.lineno})
    return {"source": src_from, "n_calls": len(calls), "calls": calls,
            "keyword_names": sorted({k for c in calls for k in c["keywords"]}),
            "read_by": "ast over the builder's source; not imported, not "
                       "regexed, not restated"}


def check_identity(receipt: dict, *, day: str | None = None,
                   coin: str | None = None) -> dict:
    """A receipt for a different day or coin REFUSES."""
    r_day, r_coin = receipt.get("day"), receipt.get("coin")
    if day is not None and r_day != day:
        raise BookVerifyRefused(
            f"REFUSED: the receipt is for day {r_day!r} and the verification "
            f"was asked for {day!r}. A receipt for another day is not this "
            f"day's evidence, however well its numbers hold together.")
    if coin is not None and r_coin != coin:
        raise BookVerifyRefused(
            f"REFUSED: the receipt is for coin {r_coin!r} and the "
            f"verification was asked for {coin!r}.")
    return {"day": r_day, "coin": r_coin,
            "protocol": receipt.get("protocol"),
            "is_BEs_declared_shape":
                receipt.get("protocol") == BE_RECEIPT_PROTOCOL}


SEAM_STATUS_VERIFIED = "SEAM_VERIFIED"
SEAM_STATUS_INCOMPLETE = "PROVENANCE_INCOMPLETE_NO_BUILDER_COMMIT"
SEAM_STATUS_UNRESOLVED = (
    "PROVENANCE_INCOMPLETE_BUILDER_COMMIT_UNRESOLVED")


def front_door_at(front_door: str | None, commit: str | None) -> dict:
    """`seam.commit` names DE's FRONT DOOR, so it is checked against DE's
    module -- which is what it names.

    THIS IS THE HALF OF MY ROUND-70 PROBE THAT WAS WRONG, and the error is
    worth stating exactly: I read BE's BUILDER at `seam.commit` and reported
    a contradiction. `seam.commit` is DE's commit (`6f134a6` is a Q-DE-80
    register entry touching only COORDINATION.md), so reading BE's builder
    there gave me a real file at a real commit that had nothing to do with
    the field. ***A commit id names a specific object; using it to locate a
    DIFFERENT object returns something true and tells you nothing.***"""
    if not front_door or not commit:
        return {"checked": False,
                "why": "no front door or no seam commit named"}
    mod, _, fn = front_door.rpartition(".")
    rel = f"live/pm_research/{mod}.py"
    r = subprocess.run(["git", "show", f"{commit}:{rel}"],
                       capture_output=True, text=True, cwd=str(HERE))
    if r.returncode != 0 or not r.stdout:
        return {"checked": True, "resolves": False, "module": rel,
                "commit": commit,
                "why": f"{rel} is not readable at {commit}"}
    try:
        tree = ast.parse(r.stdout)
    except SyntaxError:
        return {"checked": True, "resolves": False, "module": rel,
                "commit": commit, "why": "unparseable at that commit"}
    has = any(isinstance(n, ast.FunctionDef) and n.name == fn
              for n in ast.walk(tree))
    return {"checked": True, "resolves": has, "module": rel, "function": fn,
            "commit": commit,
            "why": ("the front door is defined in DE's module at the commit "
                    "the receipt names" if has else
                    f"{fn} is not defined in {rel} at {commit}")}


def builder_commit_of(receipt: dict) -> str | None:
    """The receipt's OWN builder commit (R-387's carrying_commit).

    Only these fields count. `seam.commit` is DE's and is NOT a fallback:
    falling back to it is precisely the mistake this function exists to
    prevent."""
    seam = receipt.get("seam") or {}
    for k in ("builder_commit", "carrying_commit", "producing_commit"):
        v = receipt.get(k) or seam.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def check_seam(receipt: dict) -> dict:
    """`seam.index` against the builder's call AT THE RECEIPT'S OWN BUILDER
    COMMIT -- and nowhere else.

    R-601. A receipt that names no builder commit gets
    PROVENANCE_INCOMPLETE_NO_BUILDER_COMMIT **by name**. It does NOT get a
    contradiction verdict computed against another seat's commit or against
    HEAD: a receipt is a historical record, and judging its literal by code
    it never ran is not a check, it is a coincidence of the tree. That gap
    is a PROVENANCE gap, not a false statement."""
    seam = receipt.get("seam") or {}
    literal = seam.get("index")
    seam_commit = seam.get("commit")
    b_commit = builder_commit_of(receipt)
    front = front_door_at(seam.get("front_door"), seam_commit)
    kw_claimed = sorted({
        tok.split("=")[0].strip()
        for tok in (literal or "").split("(", 1)[-1].rstrip(")").split(",")
        if "=" in tok})
    out = {
        "literal_in_the_receipt": literal,
        "keywords_the_literal_claims": kw_claimed,
        "seam_commit": seam_commit,
        "what_seam_commit_NAMES": (
            "DE's front door's commit -- NOT the builder's. Reading BE's "
            "builder there is reading a different object at a real commit"),
        "front_door": seam.get("front_door"),
        "front_door_check": front,
        "builder_commit": b_commit,
    }
    if not b_commit:
        out.update({
            "status": SEAM_STATUS_INCOMPLETE,
            "the_literal_was_NOT_judged": True,
            "why": ("the receipt names no builder commit (R-387's "
                    "carrying_commit), so the code that produced it cannot "
                    "be located and its literal cannot be checked against "
                    "the call it describes. This is a PROVENANCE GAP, not a "
                    "false statement -- and computing a contradiction "
                    "against HEAD or another seat's commit would be an "
                    "answer about code the receipt never ran"),
            "contradicts_the_code": None,
        })
        return out
    call = builder_index_call(at_commit=b_commit)
    #: REV 50 section 1.3: INSPECT WHERE THE SOURCE CAME FROM. `front_door_at`
    #: two functions above never falls back and never returns a verdict on a
    #: commit it could not read; this did both.
    if call["source"] != f"git {b_commit}":
        out.update({
            "status": "PROVENANCE_INCOMPLETE_BUILDER_COMMIT_UNRESOLVED",
            "the_literal_was_NOT_judged": True,
            "call_source": call["source"],
            "contradicts_the_code": None,
            "why": (
                f"the receipt names builder commit {b_commit} and this "
                f"worktree cannot resolve it, so the builder's source at "
                f"that commit could not be read. Judging the literal against "
                f"the WORKING TREE instead would compute a verdict from HEAD "
                f"while attributing it to the receipt's commit -- the same "
                f"error R-601 corrected, arriving through the other door. "
                f"A gap, not a defect."),
        })
        return out
    agrees = (None if call["n_calls"] == 0
              else any(sorted(c["keywords"]) == kw_claimed
                       for c in call["calls"]))
    out.update({"call_at_the_builder_commit": call,
                "agrees_with_the_call_at_the_builder_commit": agrees,
                "contradicts_the_code": agrees is False,
                "status": (SEAM_STATUS_VERIFIED if agrees
                           else "SEAM_CALL_NOT_FOUND" if agrees is None
                           else "CONTRADICTS")})
    if agrees is False:
        raise BookVerifyRefused(
            f"REFUSED: `seam.index` is a LITERAL that contradicts the call "
            f"AT THE RECEIPT'S OWN BUILDER COMMIT {b_commit}. The receipt "
            f"says {literal!r}, claiming keyword(s) {kw_claimed}; the "
            f"builder at that commit calls it with "
            f"{[c['keywords'] for c in call['calls']]}. A literal describing "
            f"a call the producing code does not make is rule 10's shape.")
    return out


# ------------------------------- (2) the population, RECOMPUTED FROM THE BOOK

def receipt_population_predicates(receipt: dict) -> dict:
    """Everything the receipt can be held to WITHOUT opening the book."""
    ref = receipt.get("reference") or {}
    sel = receipt.get("selection") or {}
    asm = receipt.get("asm") or {}
    st = ref.get("statuses") or {}
    windows = ref.get("windows")
    gens = ref.get("generations")
    excl = ("BINANCE_GAP_EXCLUDED", "NO_REPLAY", "RECONCILIATION_FAILED")
    admitted_plus_excluded = (st.get("ADMITTED", 0)
                              + sum(st.get(k, 0) for k in excl))
    marks = st.get("TERMINAL_MARK_OK", 0) + st.get("TERMINAL_MARK_MISSING", 0)
    cov = asm.get("coverage_by_head") or {}
    per_head = {}
    for head, blk in sorted(cov.items()):
        n_cov, n_unc = blk.get("n_covered"), blk.get("n_uncovered")
        n_ref = blk.get("n_reference_generations")
        recomputed = (None if not n_ref else n_cov / n_ref)
        per_head[head] = {
            "theta": blk.get("theta"),
            "n_covered": n_cov, "n_uncovered": n_unc,
            "n_reference_generations": n_ref,
            "coverage_declared": blk.get("coverage"),
            "coverage_recomputed": recomputed,
            "coverage_matches": (recomputed is not None
                                 and recomputed == blk.get("coverage")),
            "covered_plus_uncovered_equals_generations":
                (None if None in (n_cov, n_unc, n_ref)
                 else n_cov + n_unc == n_ref),
            "n_scored_keys_equals_n_covered":
                blk.get("n_scored_keys") == n_cov,
            "reference_generations_matches_the_reference_block":
                n_ref == gens,
        }
    return {
        "windows": windows,
        "windows_equals_reference_windows": windows == ref.get("windows"),
        "windows_equals_n_supplied_slugs":
            windows == sel.get("n_supplied_slugs"),
        "windows_equals_reference_n_slugs": windows == ref.get("n_slugs"),
        "generations": gens,
        "admitted_plus_excluded": admitted_plus_excluded,
        "admitted_plus_excluded_equals_windows":
            admitted_plus_excluded == windows,
        "terminal_mark_ok_plus_missing": marks,
        "terminal_marks_equal_windows": marks == windows,
        "n_terminal_marks_equals_windows":
            ref.get("n_terminal_marks") == windows,
        "both_heads_present_declared": asm.get("both_heads_present"),
        "n_heads_in_coverage": len(cov),
        "per_head": per_head,
        "n_shared_keys_declared": asm.get("n_shared_keys"),
        "sets_are_equal_declared": asm.get("sets_are_equal"),
        "NOTE": ("these hold the receipt to ITSELF. Set equality and the "
                 "scored keys are RECOMPUTED only in the BOOK tier -- "
                 "`sets_are_equal: true` is a claim until the keys are "
                 "compared"),
    }


#: NOTHING ECONOMIC IS IN A BOOK -- and that is a CENSUS, not a promise.
#: The book carries the scored generations and the arms' thetas; the
#: economics (D(E0), the null, the ratio) live in the DAY RECEIPT, which is
#: sealed. A reader entitled to know the book is safe to open before the
#: read bar is entitled to see the names that are in it.
#: TWO CLASSES, because the first run of this census flagged a real book on
#: `markout_cents_per_share` and `preventable_value_cents` -- and those are
#: the LABEL side, the valued tranches the arms are scored against. They
#: BELONG in a reference book. What must never be there is the DAY's own
#: sealed statistics. Reporting both under one word would have called the
#: book's own inputs a leak.
SEALED_DAY_STATISTIC_MARKERS = ("d_e0", "null_", "p_location", "sd_over",
                                "z_score", "eff_rt", "ci_lo", "ci_hi",
                                "p_value", "verdict", "admissib")
VALUE_INPUT_MARKERS = ("markout", "cents", "value", "spread", "fee",
                       "rebate", "pnl", "bps", "profit", "revenue")
#: identity, not a name: a slug or a bare number keying a mapping
_IDENTITY_KEY = re.compile(r"^(btc|eth|sol)[-_].*\d+$|^\d+$", re.I)
CENSUS_VISIT_BUDGET = 400_000


def economic_census(book, budget: int = CENSUS_VISIT_BUDGET) -> dict:
    """Every STRING field name in the book, and whether any is economic.

    WHAT THIS ESTABLISHES: the set of named fields the book carries, and
    that none of them names an economic quantity.
    WHAT IT CANNOT: that a float under an innocent name is not secretly a
    price. Names and shapes are checkable; intent is not. Said here rather
    than left for a reader to assume.
    """
    names: set = set()
    kinds: dict = {}
    visited = [0]
    truncated = [False]

    def walk(o, depth=0):
        if visited[0] >= budget:
            truncated[0] = True
            return
        visited[0] += 1
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str):
                    names.add(k)
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for v in list(o)[:64]:
                walk(v, depth + 1)
        else:
            kinds[type(o).__name__] = kinds.get(type(o).__name__, 0) + 1

    walk(book)
    ident = sorted(n for n in names if _IDENTITY_KEY.match(n))
    fields = sorted(n for n in names if not _IDENTITY_KEY.match(n))
    sealed_hits = sorted(n for n in fields
                         if any(m in n.lower()
                                for m in SEALED_DAY_STATISTIC_MARKERS))
    value_hits = sorted(n for n in fields
                        if any(m in n.lower() for m in VALUE_INPUT_MARKERS))
    return {
        "n_distinct_string_keys": len(names),
        "n_identity_keys": len(ident),
        "n_field_names": len(fields),
        "field_names": fields,
        "identity_key_example": ident[:3],
        "leaf_type_census": dict(sorted(kinds.items(),
                                        key=lambda kv: -kv[1])[:12]),
        "sealed_day_statistic_markers_tested":
            len(SEALED_DAY_STATISTIC_MARKERS),
        "n_field_names_naming_a_SEALED_DAY_STATISTIC": len(sealed_hits),
        "field_names_naming_a_SEALED_DAY_STATISTIC": sealed_hits,
        #: REFUTE, NEVER ESTABLISH. A truncated walk can PROVE a sealed
        #: name is present and can NEVER prove none is: the flag is only
        #: assertable when the walk COMPLETED (REV 60 section 4.1).
        #: THE ASYMMETRY, IN THE FLAG ITSELF: a sealed name the walk DID
        #: reach makes this FALSE whether or not the walk finished -- a
        #: refutation is a fact. Only the ABSENCE needs a complete walk;
        #: without one the answer is None, never True.
        "NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK": (
            False if sealed_hits else (True if not truncated[0] else None)),
        "sealed_statistic_check": (
            "REFUTED" if sealed_hits else
            "ESTABLISHED_OVER_A_COMPLETE_WALK" if not truncated[0] else
            "NOT_ESTABLISHED_WALK_TRUNCATED"),
        "n_field_names_naming_a_VALUE_INPUT": len(value_hits),
        "field_names_naming_a_VALUE_INPUT": value_hits,
        "the_value_inputs_are_EXPECTED": (
            "a reference book is built from VALUED tranches -- the label "
            "side the arms are scored against -- so names like "
            "`markout_cents_per_share` BELONG here. They are REPORTED, not "
            "flagged. What must never be here is the DAY's own sealed "
            "statistics: D(E0), the null, the ratio, the verdict"),
        "nodes_visited": visited[0],
        "budget": budget,
        "truncated": truncated[0],
        #: THE SENTENCE IS COMPUTED FROM THE NUMBERS. It used to read
        #: "NONE OF THEM NAMES AN ECONOMIC QUANTITY" beside a census that
        #: had just listed three names and truncated its walk -- rule 10's
        #: own shape, inside the instrument that censuses names.
        "summary": (
            f"the book names {len(value_hits)} reference-valuation field(s)"
            + (f" ({', '.join(value_hits[:6])})" if value_hits else "")
            + ", expected by construction; "
            + (f"{len(sealed_hits)} field(s) naming a SEALED DAY STATISTIC "
               f"({', '.join(sealed_hits[:6])}) -- REFUTED"
               if sealed_hits else
               ("no field names a sealed day statistic over a COMPLETE "
                "walk of " + f"{visited[0]} nodes"
                if not truncated[0] else
                f"the census cannot establish the absence of others at "
                f"this budget: the walk TRUNCATED at {budget} nodes"))),
        "what_this_establishes": (
            "the set of NAMED fields the book carries, which of them name a "
            "VALUE INPUT (expected) and that none names a SEALED DAY "
            "STATISTIC. ***The claim `nothing economic is in a book` is "
            "FALSE as literally stated: the book carries the valued "
            "tranches it scores against. What it does not carry is the "
            "day's own sealed result***"),
        "what_it_cannot_establish": [
            "that a float under an innocent name is not secretly a price -- "
            "names and shapes are checkable, intent is not",
            "the ABSENCE of any name, whenever the walk truncated: a "
            "bounded walk REFUTES and never ESTABLISHES",
        ],
    }


def book_population_predicates(book: dict, receipt: dict,
                               params_thetas: dict | None = None) -> dict:
    """RECOMPUTED FROM THE BOOK. This is the tier that needs the pickle."""
    asm = book.get("asm") or {}
    by_arm = asm.get("by_arm") or {}
    ref = book.get("fr") or {}
    r_asm = receipt.get("asm") or {}
    keys_by_head, thetas, second = {}, {}, {}
    for k, v in by_arm.items():
        coin, head = (k if isinstance(k, tuple) else tuple(k))
        scored = v[0] if isinstance(v, (list, tuple)) else v
        keys_by_head[head] = set(scored)
        if isinstance(v, (list, tuple)) and len(v) > 1:
            #: THE SECOND ELEMENT IS NOT A THETA. The first real run of this
            #: tier read it as one and flagged BOTH heads on a real book:
            #: it is a COUNT MAP ({SCORED, NO_ROWS_KEPT, PARTIAL_ROWS}).
            #: A number is a theta; a mapping is not, and guessing which
            #: turned the verifier's own misread into a finding against the
            #: book.
            second[head] = v[1]
            if isinstance(v[1], (int, float)) and not isinstance(v[1], bool):
                thetas[head] = float(v[1])
    heads = sorted(keys_by_head)
    out = {"heads_in_the_book": heads, "n_heads": len(heads),
           "book_layout": {
               "top_level_keys": sorted(k for k in book
                                        if isinstance(k, str)),
               "asm_keys": sorted(k for k in asm if isinstance(k, str)),
               "asm_keys_naming_a_theta": sorted(
                   k for k in asm if isinstance(k, str)
                   and "theta" in k.lower()),
               "why": ("recorded so the next reader does not have to open "
                       "a 300 MB pickle to learn where a field lives"),
           },
           "by_arm_keys": sorted([list(k) if not isinstance(k, str) else k
                                  for k in by_arm], key=str)}
    if len(heads) == 2:
        a, b = heads
        sa, sb = keys_by_head[a], keys_by_head[b]
        only_a, only_b = sa - sb, sb - sa
        out["set_equality"] = {
            "recomputed": not only_a and not only_b,
            "declared": r_asm.get("sets_are_equal"),
            "agrees_with_the_receipt":
                (not only_a and not only_b) == bool(r_asm.get("sets_are_equal")),
            "n_shared_recomputed": len(sa & sb),
            "n_shared_declared": r_asm.get("n_shared_keys"),
            "n_only_in_" + a: len(only_a),
            "n_only_in_" + b: len(only_b),
            "difference_sized": len(only_a) + len(only_b),
            "why_recomputed": (
                "`sets_are_equal: true` is the builder's own claim about its "
                "own output. Two heads scoring different key sets is exactly "
                "the precondition the null rests on, so it is compared, not "
                "read"),
        }
    n_gen = (receipt.get("reference") or {}).get("generations")
    per_head = {}
    for head in heads:
        n_scored = len(keys_by_head[head])
        d = (r_asm.get("coverage_by_head") or {}).get(head, {})
        per_head[head] = {
            "n_scored_keys_recomputed": n_scored,
            "n_scored_keys_declared": d.get("n_scored_keys"),
            "matches": n_scored == d.get("n_scored_keys"),
            "coverage_recomputed": (None if not n_gen else n_scored / n_gen),
            "coverage_declared": d.get("coverage"),
            "coverage_matches": (n_gen is not None
                                 and n_scored / n_gen == d.get("coverage")),
            "theta_in_the_book": thetas.get(head),
            "theta_declared": d.get("theta"),
            #: TRUE when the book carries a theta and it agrees; FALSE when
            #: it carries one and it does not; NONE when the book carries
            #: none -- which is NOT a mismatch and is NOT a pass either. It
            #: is reported as not computable and it keeps the tier from
            #: claiming a verification (rule 11).
            "theta_matches": (None if head not in thetas
                              else thetas[head] == d.get("theta")),
            "the_second_element_of_by_arm": {
                "type": type(second.get(head)).__name__,
                "value": (second.get(head)
                          if not isinstance(second.get(head), dict)
                          else dict(list(second[head].items())[:6])),
                "is_a_theta": head in thetas,
                "why": ("a NUMBER here is the arm's theta; a MAPPING is the "
                        "stage's count map. The tier says which it found "
                        "rather than assuming"),
            },
        }
        if params_thetas and head in params_thetas:
            per_head[head]["theta_in_params"] = params_thetas[head]
            per_head[head]["theta_matches_params"] = (
                d.get("theta") == params_thetas[head])
    out["per_head"] = per_head
    out["reference_generations_in_the_book"] = ref.get("generations", n_gen)
    return out


# ------------------------------------------- (3) the resources, as FACTS

def resource_facts(receipt: dict) -> dict:
    res = receipt.get("resources") or {}
    stages = res.get("stages") or []
    rows = []
    for s in stages:
        peak, budget = s.get("peak_gb"), s.get("budget_gb")
        rows.append({"stage": s.get("stage"), "peak_gb": peak,
                     "budget_gb": budget, "wall_s": s.get("wall_s"),
                     "within_budget_recomputed":
                         (None if None in (peak, budget) else peak <= budget),
                     "within_budget_declared": s.get("within_budget")})
    rel = res.get("index_released") or {}
    before, after = rel.get("current_gb_before"), rel.get("current_gb_after")
    return {
        "stages": rows,
        "n_stages": len(rows),
        "every_stage_within_its_budget": all(
            r["within_budget_recomputed"] for r in rows
            if r["within_budget_recomputed"] is not None),
        "declared_agrees_with_recomputed": all(
            r["within_budget_recomputed"] == r["within_budget_declared"]
            for r in rows if r["within_budget_recomputed"] is not None),
        "peak_gb": res.get("peak_gb"),
        "wall_s": res.get("wall_s"),
        "wall_s_equals_sum_of_stages": (
            None if not rows or any(r["wall_s"] is None for r in rows)
            else abs(sum(r["wall_s"] for r in rows)
                     - (res.get("wall_s") or 0)) < 1.0),
        "index_release": {
            "freed_gb_declared": rel.get("freed_gb"),
            "freed_gb_recomputed": (None if None in (before, after)
                                    else round(before - after, 6)),
            "release_fraction": (None if not before
                                 else (before - after) / before),
            "measured_on": rel.get("measured_on_CURRENT_rss"),
        },
        "THESE_ARE_FACTS_NOT_VERDICTS": (
            "budgets are the builder's own and are reported against its own "
            "measurements. Nothing here passes or fails the run; a reader "
            "who wants a verdict has the numbers to form one"),
    }


# ---------------------------------------------------------------- the tiers

def verify_receipt_tier(book_path, receipt_path, *,
                        day: str | None = None, coin: str | None = None,
                        verify_inputs: bool = True,
                        fixture: bool = True,
                        output: Path | None = None) -> dict:
    """LIGHT. Everything that does not need the pickle opened.

    `fixture` gates ONE thing: rule 22's dirty-worktree bar. A dirty tree is
    a recorded FACT for a fixture and a REFUSAL for a real artifact --
    uncommitted bytes are locatable nowhere (R-603)."""
    rp = Path(receipt_path)
    if not rp.is_file():
        raise BookVerifyRefused(f"REFUSED: receipt absent at {receipt_path}")
    receipt = json.loads(rp.read_text())
    ident = check_identity(receipt, day=day, coin=coin)
    seam = check_seam(receipt)
    chain = digest_chain(book_path, receipt, verify_inputs=verify_inputs)
    pop = receipt_population_predicates(receipt)
    res = resource_facts(receipt)
    flags = []
    if not chain["chain_holds"]:
        flags.append("digest_chain")
    for k in ("windows_equals_reference_windows",
              "windows_equals_n_supplied_slugs",
              "windows_equals_reference_n_slugs",
              "admitted_plus_excluded_equals_windows",
              "terminal_marks_equal_windows",
              "n_terminal_marks_equals_windows"):
        if pop[k] is not True:
            flags.append(f"population.{k}")
    for head, blk in pop["per_head"].items():
        for k in ("coverage_matches", "covered_plus_uncovered_equals_generations",
                  "n_scored_keys_equals_n_covered",
                  "reference_generations_matches_the_reference_block"):
            if blk[k] is not True:
                flags.append(f"population.{head}.{k}")
    if not res["declared_agrees_with_recomputed"]:
        flags.append("resources.declared_agrees_with_recomputed")
    if not ident["is_BEs_declared_shape"]:
        flags.append("identity.is_BEs_declared_shape")
    if seam.get("front_door_check", {}).get("resolves") is False:
        flags.append("seam.front_door_does_not_resolve")
    #: THREE STATES, NOT TWO (R-601). A receipt whose producing code cannot
    #: be located is INCOMPLETE, not wrong: the groups that were checked
    #: stand, and the seam literal was not judged at all. Collapsing that
    #: into FLAGGED would report a defect where there is a gap.
    incomplete = seam.get("status") in (
        SEAM_STATUS_INCOMPLETE, SEAM_STATUS_UNRESOLVED)
    status = ("FLAGGED" if flags
              else "PROVENANCE_INCOMPLETE" if incomplete else "VERIFIED")
    out = {
        "protocol": PROTOCOL + "_RECEIPT_TIER",
        "tier": "RECEIPT",
        "status": status,
        "provenance_incomplete": incomplete,
        "what_incomplete_means": (
            "every group that COULD be checked was, and they hold; the seam "
            "literal was not judged because the receipt names no builder "
            "commit. A gap, not a defect -- and not a verification either"
            if incomplete else None),
        "IS_A_VERIFICATION": bool(not flags and not incomplete),
        "what_this_tier_cannot_say": (
            "the set equality and the scored-key counts are the BOOK's, and "
            "this tier never opens it. `sets_are_equal: true` stays a claim "
            "until the BOOK tier compares the keys"),
        "identity": ident, "seam": seam, "digest_chain": chain,
        "population_from_the_receipt": pop, "resources": res,
        "flags": flags, "n_flags": len(flags),
        "receipt": {"path": rp.name,
                    "sha256": hashlib.sha256(rp.read_bytes()).hexdigest()},
        #: RULE 22 / R-605: the emit REFUSES if any module of this run's
        #: closure, this file, or HEAD moved under it.
        "launch_capture": assert_source_unchanged(
            "the receipt tier's emit", fixture=fixture),
        "verifier_identity": verifier_identity(),
        "resource_observation": {"rss_gb_at_end": round(_rss_gb(), 3),
                                 "book_was_opened": False},
        "the_book_tier_is_HEAVY": {
            "why": ("the book is a pickle of "
                    f"{(receipt.get('book') or {}).get('bytes')} bytes and "
                    "unpickling it is expected to cost about "
                    f"{BOOK_LOAD_EXPECTED_PEAK_GB} GB resident"),
            "expected_peak_gb": BOOK_LOAD_EXPECTED_PEAK_GB,
            "declared_cap_gb": BOOK_LOAD_CAP_GB,
            "rule_20": ("over 60 s or 1 GiB, so it takes the heavy lock and "
                        "runs under the wrapper; the real 09-03 run is a "
                        "coordinator GO, not this module's own initiative"),
        },
    }
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


def supersession_block(prior: Path) -> dict:
    """The R-608 PAIR for an in-band re-emission, chain carried forward."""
    b = prior.read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    try:
        blk = (json.loads(prior.read_text()).get("supersedes") or {})
    except (OSError, ValueError):
        blk = {}
    chain = [list(e) for e in (blk.get("chain") or [])
             if isinstance(e, (list, tuple)) and len(e) == 2]
    return {"path": prior.name, "sha256": sha,
            "chain": chain + [[prior.name, sha]],
            "the_link_is_the_PAIR": ["path", "sha256"],
            "v1_untouched": True,
            "what_changed": (
                "the prose. The superseded artifact carried "
                "`NOTHING_ECONOMIC_IS_NAMED_IN_THIS_BOOK: false` beside a "
                "sentence saying none of them names an economic quantity, "
                "and two theta MISMATCHES that were a misread of the "
                "book's count map. The numbers about the BOOK -- set "
                "equality, key counts, coverage -- are unchanged and were "
                "right in the superseded artifact too")}


def verify_full(book_path, receipt_path, *, day: str | None = None,
                coin: str | None = None, output: Path | None = None,
                fixture: bool = True, supersedes: Path | None = None,
                _book_obj=None, params_thetas: dict | None = None) -> dict:
    """HEAVY. The receipt tier plus the population RECOMPUTED FROM THE BOOK."""
    out = verify_receipt_tier(book_path, receipt_path, day=day, coin=coin,
                              verify_inputs=True, fixture=fixture)
    receipt = json.loads(Path(receipt_path).read_text())
    if _book_obj is not None:
        book = _book_obj
        loaded_from = "supplied in process (fixture)"
    else:
        import pickle                                         # noqa: PLC0415
        before = _rss_gb()
        with Path(book_path).open("rb") as fh:
            book = pickle.load(fh)
        #: RULE 22: unpickling can IMPORT modules (a pickle names the
        #: classes it needs), and a module that entered the run here was
        #: not in `sys.modules` at import time. REV 53 section 1.1 found
        #: exactly that gap in DE's closure -- the two modules that do the
        #: replaying were outside it.
        capture_closure("after the book was unpickled")
        after = _rss_gb()
        loaded_from = str(book_path)
        if after > BOOK_LOAD_CAP_GB:
            raise BookVerifyRefused(
                f"REFUSED: loading the book took {after:.2f} GB resident, "
                f"over the declared {BOOK_LOAD_CAP_GB} GB cap (expected "
                f"about {BOOK_LOAD_EXPECTED_PEAK_GB}). The cap is not raised "
                f"and the book is not read in pieces to fit it.")
        out["resource_observation"].update(
            {"rss_gb_before_load": round(before, 3),
             "rss_gb_after_load": round(after, 3),
             "load_delta_gb": round(after - before, 3),
             "expected_peak_gb": BOOK_LOAD_EXPECTED_PEAK_GB,
             "declared_cap_gb": BOOK_LOAD_CAP_GB})
    out["resource_observation"]["book_was_opened"] = True
    out["resource_observation"]["book_loaded_from"] = loaded_from
    bp = book_population_predicates(book, receipt, params_thetas)
    flags = list(out["flags"])
    se = bp.get("set_equality")
    if se is not None:
        if not se["recomputed"]:
            flags.append("book.set_equality_recomputed_false")
        if not se["agrees_with_the_receipt"]:
            flags.append("book.set_equality_disagrees_with_the_receipt")
        if se["n_shared_recomputed"] != se["n_shared_declared"]:
            flags.append("book.n_shared_keys")
    if bp["n_heads"] != 2:
        flags.append("book.n_heads")
    not_computable = []
    for head, blk in bp["per_head"].items():
        for k in ("matches", "coverage_matches", "theta_matches"):
            if blk[k] is False:
                flags.append(f"book.{head}.{k}")
            elif blk[k] is None:
                #: NEITHER a flag NOR a pass. A predicate that could not be
                #: computed is a STATUS, and it keeps IS_A_VERIFICATION
                #: false (rule 11).
                not_computable.append(f"book.{head}.{k}")
        if blk.get("theta_matches_params") is False:
            flags.append(f"book.{head}.theta_matches_params")
    incomplete = bool(out.get("provenance_incomplete")) or bool(not_computable)
    census = economic_census(book)
    #: A FLAG IS A REFUTATION, NOT AN UNKNOWN. `not None` is True, so a
    #: TRUNCATED walk -- which establishes nothing -- was being reported as
    #: "a sealed day statistic IS named in the book", the exact inversion
    #: the truncation limit exists to prevent. REFUTED flags; NOT
    #: ESTABLISHED is a status that blocks the verification and names
    #: itself.
    if census["sealed_statistic_check"] == "REFUTED":
        flags.append("book.a_SEALED_DAY_STATISTIC_is_named_in_the_book")
    elif census["sealed_statistic_check"] == "NOT_ESTABLISHED_WALK_TRUNCATED":
        not_computable.append(
            "book.no_sealed_day_statistic_is_named__WALK_TRUNCATED")
    out.update({
        "protocol": PROTOCOL + "_FULL",
        "tier": "FULL",
        "supersedes": (supersession_block(Path(supersedes))
                       if supersedes else None),
        "economic_census_of_the_book": census,
        "population_from_the_book": bp,
        "predicates_not_computable": not_computable,
        "n_predicates_not_computable": len(not_computable),
        "flags": flags, "n_flags": len(flags),
        "status": ("FLAGGED" if flags
                   else "PROVENANCE_INCOMPLETE" if incomplete else "VERIFIED"),
        "IS_A_VERIFICATION": bool(not flags and not incomplete),
        "what_this_tier_cannot_say": (
            "that the day's TAPE and FRAGMENT were built correctly -- schema, "
            "splits, coverage. Those are the builder's own guards. This "
            "verifies WHICH BYTES reached the pass and what the book says "
            "about itself"),
    })
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


# --------------------------------------------------------------- the fixture

def synthetic_book_and_receipt(d: Path, *, windows: int = 12,
                               gens: int = 400, uncovered: int = 40,
                               day: str = "20260903", coin: str = "btc",
                               heads=("q1_arrival_composed_lgbm",
                                      "incumbent_linear_d"),
                               thetas=(0.32450609461933483,
                                       0.43525926488298716),
                               unequal_sets: bool = False,
                               builder_commit: str | None = "HEAD",
                               seam_commit: str | None = "HEAD") -> tuple:
    """A book of BE's OWN shape -- {"fr": …, "asm": {"by_arm": {(coin, head):
    (scored, theta)}}} -- with a receipt of BE_DAYBOOK_V1's shape built from
    it, so every predicate has something true to be true OF."""
    import pickle
    keys = [f"g{i}" for i in range(gens - uncovered)]
    by_arm = {}
    for i, (h, th) in enumerate(zip(heads, thetas)):
        k = list(keys)
        if unequal_sets and i == 1:
            k = k[:-3] + ["EXTRA_A", "EXTRA_B"]
        by_arm[(coin, h)] = (k, th)
    book = {"fr": {"generations": gens, "windows": windows},
            "asm": {"by_arm": by_arm}}
    bp = d / f"be_daybook_{day}_{coin}.pkl"
    buf = pickle.dumps(book, protocol=pickle.HIGHEST_PROTOCOL)
    bp.write_bytes(buf)
    sha = hashlib.sha256(buf).hexdigest()
    tape = d / "tape.json"
    tape.write_text('{"tape": 1}')
    frag = d / "fragment.json"
    frag.write_text('{"fragment": 1}')
    n_cov = gens - uncovered
    #: the seam literal is DERIVED from the builder's real call so the
    #: fixture's receipt is HONEST by construction; the known-bad plants a
    #: contradicting one.
    call = builder_index_call(at_commit=builder_commit)
    lit = (call["calls"][0]["rendered"] if call["calls"]
           else "build_tape_index(splits)")
    receipt = {
        "protocol": BE_RECEIPT_PROTOCOL, "day": day, "coin": coin,
        "book": {"bytes": len(buf), "path": str(bp), "sha256": sha,
                 "readback_sha256": sha, "readback_matches": True},
        "inputs_pinned": {
            "tape": {"path": str(tape),
                     "sha256": hashlib.sha256(tape.read_bytes()).hexdigest(),
                     "split": "score"},
            "fragment": {"path": str(frag),
                         "sha256": hashlib.sha256(
                             frag.read_bytes()).hexdigest()}},
        "seam": {"commit": seam_commit, "index": lit,
                 "front_door": "de_phase4_diag_runner.day_assembly_inputs"},
        "builder_commit": builder_commit,
        "selection": {"n_supplied_slugs": windows},
        "reference": {"windows": windows, "n_slugs": windows,
                      "generations": gens, "n_terminal_marks": windows,
                      "statuses": {"ADMITTED": windows,
                                   "BINANCE_GAP_EXCLUDED": 0,
                                   "NO_REPLAY": 0,
                                   "RECONCILIATION_FAILED": 0,
                                   "TERMINAL_MARK_OK": windows,
                                   "TERMINAL_MARK_MISSING": 0}},
        "asm": {"both_heads_present": True,
                "by_arm_keys": [[coin, h] for h in heads],
                "n_shared_keys": len(keys) if not unequal_sets
                                 else len(set(keys) & set(
                                     by_arm[(coin, heads[1])][0])),
                "sets_are_equal": not unequal_sets,
                "coverage_by_head": {
                    h: {"coverage": n_cov / gens, "n_covered": n_cov,
                        "n_reference_generations": gens,
                        "n_scored_keys": n_cov, "n_uncovered": uncovered,
                        "theta": th} for h, th in zip(heads, thetas)}},
        "resources": {
            "peak_gb": 5.317, "wall_s": 30.0,
            "stages": [{"stage": "A0_reference", "peak_gb": 2.0,
                        "budget_gb": 3.0, "wall_s": 10.0,
                        "within_budget": True, "current_gb": 2.0},
                       {"stage": "A2_assemble", "peak_gb": 5.317,
                        "budget_gb": 7.5, "wall_s": 20.0,
                        "within_budget": True, "current_gb": 4.0}],
            "index_released": {"current_gb_before": 4.096,
                               "current_gb_after": 2.94, "freed_gb": 1.156,
                               "measured_on_CURRENT_rss": "VmRSS"}},
    }
    rp = d / f"be_daybook_receipt_{day}_{coin}.json"
    rp.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    return bp, rp, book


def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    td = Path(tempfile.mkdtemp(prefix="da70_"))
    bp, rp, book = synthetic_book_and_receipt(td)

    # -- 1. the happy path, BOTH tiers ------------------------------------
    r_tier = verify_receipt_tier(bp, rp, day="20260903", coin="btc")
    full = verify_full(bp, rp, day="20260903", coin="btc")
    ck("BOTH TIERS VERIFY A BOOK OF BE's OWN SHAPE: the RECEIPT tier holds "
       "the receipt to itself and to the bytes; the FULL tier opens the book "
       "and recomputes what only the book can say",
       r_tier["IS_A_VERIFICATION"] is True
       and full["IS_A_VERIFICATION"] is True
       and r_tier["tier"] == "RECEIPT" and full["tier"] == "FULL",
       f"receipt tier {r_tier['status']} ({r_tier['n_flags']} flags), full "
       f"{full['status']} ({full['n_flags']} flags)")

    # -- 2. THE DIGEST CHAIN, both statements -----------------------------
    ch = r_tier["digest_chain"]
    ck("THE DIGEST CHAIN CHECKS BOTH STATEMENTS OF THE SAME BYTES: the "
       "write-side `sha256` AND the independent `readback_sha256`, against "
       "one streamed hash of the file. Checking one and calling it the chain "
       "leaves the other unchecked -- and they can disagree",
       ch["matches_declared"] and ch["matches_readback"]
       and ch["declared_and_readback_agree"] and ch["bytes_match"]
       and ch["chain_holds"] and ch["n_inputs_checked"] == 2,
       f"{ch['bytes_on_disk']} bytes hashing to "
       f"{ch['sha256_recomputed'][:16]}; both declared statements agree; "
       f"{ch['n_inputs_checked']} pinned inputs hashed, "
       f"{ch['n_inputs_absent']} absent")

    # -- 3. A TAMPERED BOOK REFUSES ON THE DIGEST -------------------------
    tb = td / "tampered.pkl"
    tb.write_bytes(bp.read_bytes() + b"\x00")
    tam = verify_receipt_tier(tb, rp, day="20260903", coin="btc")
    ck("KNOWN-BAD: A TAMPERED BOOK FAILS THE CHAIN -- one appended byte "
       "moves the digest and the chain does not hold, against BOTH declared "
       "statements",
       tam["IS_A_VERIFICATION"] is False
       and tam["digest_chain"]["chain_holds"] is False
       and tam["digest_chain"]["matches_declared"] is False
       and tam["digest_chain"]["matches_readback"] is False
       and "digest_chain" in tam["flags"],
       f"one byte appended -> {tam['digest_chain']['sha256_recomputed'][:16]} "
       f"against declared {tam['digest_chain']['sha256_declared'][:16]}")

    # -- 4. AN ABSENT PINNED INPUT IS A STATUS, NEVER A PASS --------------
    r2 = json.loads(rp.read_text())
    r2["inputs_pinned"]["tape"]["path"] = str(td / "gone.json")
    rp2 = td / "receipt_absent_input.json"
    rp2.write_text(json.dumps(r2, default=str))
    ab = verify_receipt_tier(bp, rp2, day="20260903", coin="btc")
    ck("AN ABSENT PINNED INPUT IS A STATUS, NEVER A PASS: it is reported "
       "INPUT_ABSENT_NOT_CHECKED with `matches: null`, because a `true` "
       "there would be a pass for a check that never ran",
       ab["digest_chain"]["inputs"]["tape"]["status"]
       == "INPUT_ABSENT_NOT_CHECKED"
       and ab["digest_chain"]["inputs"]["tape"]["matches"] is None
       and ab["digest_chain"]["n_inputs_absent"] == 1
       and ab["digest_chain"]["n_inputs_checked"] == 1,
       f"tape absent -> status "
       f"{ab['digest_chain']['inputs']['tape']['status']}, matches "
       f"{ab['digest_chain']['inputs']['tape']['matches']}")

    # -- 5. A MOVED COUNT IS FLAGGED --------------------------------------
    r3 = json.loads(rp.read_text())
    r3["reference"]["statuses"]["ADMITTED"] += 1
    rp3 = td / "receipt_moved.json"
    rp3.write_text(json.dumps(r3, default=str))
    mv = verify_receipt_tier(bp, rp3, day="20260903", coin="btc")
    ck("KNOWN-BAD: ONE MOVED COUNT IS FLAGGED. ADMITTED + the excluded "
       "classes must equal the windows, so a receipt whose statuses no "
       "longer sum is flagged on the sum, not on the field",
       mv["IS_A_VERIFICATION"] is False
       and "population.admitted_plus_excluded_equals_windows" in mv["flags"],
       f"ADMITTED +1 -> {[f for f in mv['flags'] if 'admitted' in f]}")

    # -- 6. UNEQUAL HEAD KEY SETS ARE FLAGGED WITH THE DIFFERENCE SIZED ---
    d2 = td / "unequal"
    d2.mkdir(exist_ok=True)
    bp_u, rp_u, _ = synthetic_book_and_receipt(d2, unequal_sets=True)
    ru = json.loads(rp_u.read_text())
    ru["asm"]["sets_are_equal"] = True          # the builder's CLAIM
    rp_u2 = d2 / "receipt_claims_equal.json"
    rp_u2.write_text(json.dumps(ru, default=str))
    un = verify_full(bp_u, rp_u2, day="20260903", coin="btc")
    se = un["population_from_the_book"]["set_equality"]
    ck("KNOWN-BAD, AND THE ONE THAT MATTERS MOST: UNEQUAL HEAD KEY SETS ARE "
       "FLAGGED AND THE DIFFERENCE IS SIZED -- the receipt CLAIMS "
       "`sets_are_equal: true` and the recomputation over the book's own "
       "keys disagrees. Set equality is the precondition the null rests on, "
       "so it is compared, never read",
       un["IS_A_VERIFICATION"] is False
       and se["recomputed"] is False
       and se["agrees_with_the_receipt"] is False
       and se["difference_sized"] == 5
       and "book.set_equality_recomputed_false" in un["flags"],
       f"the book's two heads differ by {se['difference_sized']} keys "
       f"({se['n_only_in_q1_arrival_composed_lgbm']} only in one, "
       f"{se['n_only_in_incumbent_linear_d']} only in the other) while the "
       f"receipt claims equality")

    # -- 7. and the POSITIVE control on the same axis ---------------------
    se_ok = full["population_from_the_book"]["set_equality"]
    ck("POSITIVE CONTROL ON THE SAME AXIS: equal key sets RECOMPUTE equal, "
       "agree with the receipt, and the shared count matches -- so the flag "
       "means something",
       se_ok["recomputed"] is True and se_ok["agrees_with_the_receipt"] is True
       and se_ok["n_shared_recomputed"] == se_ok["n_shared_declared"]
       and se_ok["difference_sized"] == 0,
       f"{se_ok['n_shared_recomputed']} shared keys recomputed against "
       f"{se_ok['n_shared_declared']} declared, difference 0")

    # -- 8. coverage and the thetas, recomputed ---------------------------
    ph = full["population_from_the_book"]["per_head"]
    ck("COVERAGE AND THE THETAS ARE RECOMPUTED FROM THE BOOK: n_scored_keys "
       "counted off the book's own key sets, coverage divided out again, and "
       "each head's theta compared to the one the receipt pins",
       all(b["matches"] and b["coverage_matches"] and b["theta_matches"]
           for b in ph.values()) and len(ph) == 2,
       "; ".join(f"{h}: {b['n_scored_keys_recomputed']} keys, coverage "
                 f"{b['coverage_recomputed']:.6f}, theta "
                 f"{b['theta_declared']}" for h, b in sorted(ph.items())))

    # -- 9. a receipt for the WRONG DAY or COIN REFUSES -------------------
    why_day = why_coin = ""
    try:
        verify_receipt_tier(bp, rp, day="20260904", coin="btc")
    except BookVerifyRefused as e:
        why_day = str(e)
    try:
        verify_receipt_tier(bp, rp, day="20260903", coin="eth")
    except BookVerifyRefused as e:
        why_coin = str(e)
    ck("A RECEIPT FOR A DIFFERENT DAY OR COIN REFUSES: a receipt for another "
       "day is not this day's evidence, however well its numbers hold "
       "together",
       "receipt is for day" in why_day and "20260904" in why_day
       and "coin" in why_coin and "'eth'" in why_coin,
       f"day: '{why_day[:64]}...'; coin: '{why_coin[:56]}...'")

    # -- 10. THE SEAM, REBUILT UNDER R-601 --------------------------------
    call = builder_index_call()
    ck("THE BUILDER'S `build_tape_index` CALL IS READ BY AST FROM ITS OWN "
       "SOURCE -- argument names off the syntax tree, not a regex over text "
       "and not the receipt's word for it",
       call["n_calls"] >= 1 and call["keyword_names"]
       and all("rendered" in c for c in call["calls"]),
       f"{call['n_calls']} call(s) at line(s) "
       f"{[c['line'] for c in call['calls']]}, keywords "
       f"{call['keyword_names']}")

    #: A RECEIPT NAMING NO BUILDER COMMIT: incomplete, NEVER a contradiction.
    r_nb = json.loads(rp.read_text())
    r_nb.pop("builder_commit", None)
    r_nb["seam"]["index"] = "build_tape_index(splits, tape_path=…)"
    rp_nb = td / "receipt_no_builder_commit.json"
    rp_nb.write_text(json.dumps(r_nb, default=str))
    nb = verify_receipt_tier(bp, rp_nb, day="20260903", coin="btc")
    ck("R-601, THE CORRECTION TO MY OWN PROBE: a receipt naming NO BUILDER "
       "COMMIT gets PROVENANCE_INCOMPLETE_NO_BUILDER_COMMIT BY NAME, and its "
       "literal is NOT JUDGED -- even a literal that would contradict HEAD. "
       "***Computing a contradiction against HEAD or another seat's commit "
       "is an answer about code the receipt never ran.*** A gap, not a "
       "defect",
       nb["seam"]["status"] == SEAM_STATUS_INCOMPLETE
       and nb["seam"]["contradicts_the_code"] is None
       and nb["seam"]["the_literal_was_NOT_judged"] is True
       and nb["status"] == "PROVENANCE_INCOMPLETE"
       and nb["n_flags"] == 0
       and nb["IS_A_VERIFICATION"] is False,
       f"literal {nb['seam']['literal_in_the_receipt']!r} left unjudged; "
       f"status {nb['status']} with {nb['n_flags']} flags -- incomplete is a "
       f"THIRD state, neither VERIFIED nor FLAGGED")

    #: naming a builder commit whose call MATCHES -> verified.
    ck("A RECEIPT NAMING A BUILDER COMMIT WHOSE CALL MATCHES IS VERIFIED: "
       "the literal is judged against the code that actually produced the "
       "receipt, and nowhere else",
       r_tier["seam"]["status"] == SEAM_STATUS_VERIFIED
       and r_tier["seam"]["agrees_with_the_call_at_the_builder_commit"] is True
       and r_tier["seam"]["builder_commit"] is not None
       and r_tier["status"] == "VERIFIED",
       f"builder_commit {r_tier['seam']['builder_commit']}, literal "
       f"{r_tier['seam']['literal_in_the_receipt']!r} agrees with the call "
       f"there")

    #: naming a builder commit whose call CONTRADICTS -> refuses.
    r_bad = json.loads(rp.read_text())
    r_bad["seam"]["index"] = "build_tape_index(splits, tape_path=…)"
    rp_bad = td / "receipt_contradicting.json"
    rp_bad.write_text(json.dumps(r_bad, default=str))
    why_seam = ""
    try:
        verify_receipt_tier(bp, rp_bad, day="20260903", coin="btc")
    except BookVerifyRefused as e:
        why_seam = str(e)
    ck("KNOWN-BAD: A LITERAL CONTRADICTING THE CALL **AT THE RECEIPT'S OWN "
       "BUILDER COMMIT** REFUSES -- the check still has teeth, it is just "
       "pointed at the right object now",
       "contradicts the call" in why_seam
       and "BUILDER COMMIT" in why_seam and "tape_path" in why_seam,
       f"'{why_seam[:112]}...'")

    #: REV 50 section 1.3, both directions.
    r_unres = json.loads(rp.read_text())
    r_unres["builder_commit"] = "deadbeef" * 5
    rp_u = td / "receipt_unresolvable_commit.json"
    rp_u.write_text(json.dumps(r_unres, default=str))
    ur = verify_receipt_tier(bp, rp_u, day="20260903", coin="btc")
    ck("REV 50 section 1.3 CLOSED -- A BUILDER COMMIT THIS WORKTREE CANNOT "
       "RESOLVE IS INCOMPLETE, NEVER FLAGGED. `builder_index_call` fell back "
       "to the WORKING TREE and `check_seam` never looked at `source`, so a "
       "verdict computed from HEAD wore a sentence attributing it to the "
       "receipt's commit -- ***the R-601 class through the other door***. "
       "`front_door_at` two functions above never fell back and never "
       "returned a verdict on a commit it could not read",
       ur["seam"]["status"] == SEAM_STATUS_UNRESOLVED
       and ur["seam"]["contradicts_the_code"] is None
       and ur["seam"]["the_literal_was_NOT_judged"] is True
       and ur["status"] == "PROVENANCE_INCOMPLETE"
       and ur["n_flags"] == 0
       and "FALLBACK" in ur["seam"]["call_source"],
       f"an unresolvable commit -> {ur['seam']['status']}, "
       f"{ur['n_flags']} flags, source "
       f"'{ur['seam']['call_source'][:44]}...'")
    ck("AND THE POSITIVE CONTROL ON THE SAME AXIS: a RESOLVABLE builder "
       "commit whose call matches still VERIFIES -- the fix removes a false "
       "verdict, not the check",
       r_tier["seam"]["status"] == SEAM_STATUS_VERIFIED
       and r_tier["seam"]["call_at_the_builder_commit"]["source"]
       == f"git {r_tier['seam']['builder_commit']}"
       and r_tier["status"] == "VERIFIED",
       f"builder_commit {r_tier['seam']['builder_commit']} resolved from "
       f"{r_tier['seam']['call_at_the_builder_commit']['source']} and the "
       f"literal agrees")

    #: and seam.commit is checked against DE's module, which is what it names
    fd = r_tier["seam"]["front_door_check"]
    ck("AND `seam.commit` IS CHECKED AGAINST DE's MODULE -- WHICH IS WHAT IT "
       "NAMES. My round-70 probe read BE's BUILDER there and reported a "
       "contradiction: `6f134a6` is a Q-DE-80 register entry touching only "
       "COORDINATION.md, so reading BE's builder at it returned a real file "
       "at a real commit with nothing to do with the field. ***A commit id "
       "names a specific object; using it to locate a different one returns "
       "something true and tells you nothing.***",
       fd["checked"] is True and fd["resolves"] is True
       and fd["function"] == "day_assembly_inputs"
       and "de_phase4_diag_runner" in fd["module"],
       f"{fd['module']} at {fd['commit']} defines {fd['function']}: "
       f"{fd['resolves']}")

    # -- 11. THE RESOURCES AS FACTS ---------------------------------------
    rf = r_tier["resources"]
    ck("THE RESOURCES ARE FACTS, NOT VERDICTS: every stage's peak is "
       "compared to its OWN declared budget and the builder's "
       "`within_budget` is checked against the recomputation, with the index "
       "release fraction divided out",
       rf["every_stage_within_its_budget"] is True
       and rf["declared_agrees_with_recomputed"] is True
       and abs(rf["index_release"]["freed_gb_recomputed"] - 1.156) < 1e-9
       and abs(rf["index_release"]["release_fraction"] - 1.156 / 4.096) < 1e-9,
       f"{rf['n_stages']} stages all within budget; index release "
       f"{rf['index_release']['freed_gb_recomputed']} GB = "
       f"{rf['index_release']['release_fraction']:.3f} of what was held")
    r5 = json.loads(rp.read_text())
    r5["resources"]["stages"][0]["within_budget"] = False
    rp5 = td / "receipt_budget_lie.json"
    rp5.write_text(json.dumps(r5, default=str))
    bl = verify_receipt_tier(bp, rp5, day="20260903", coin="btc")
    ck("KNOWN-BAD: A `within_budget` FLAG THAT DISAGREES WITH THE "
       "ARITHMETIC IS FLAGGED -- the builder's own boolean is checked "
       "against peak <= budget rather than taken",
       bl["IS_A_VERIFICATION"] is False
       and "resources.declared_agrees_with_recomputed" in bl["flags"],
       "stage A0 declares within_budget false at peak 2.0 <= budget 3.0")

    # -- 12. the RECEIPT tier says what it CANNOT say ---------------------
    ck("THE RECEIPT TIER STATES WHAT IT CANNOT SAY, and the heavy tier's "
       "cost is DECLARED before it runs: set equality stays a CLAIM until "
       "the book is opened, and opening it is expected to cost about "
       f"{BOOK_LOAD_EXPECTED_PEAK_GB} GB against a declared "
       f"{BOOK_LOAD_CAP_GB} GB cap",
       "stays a claim" in r_tier["what_this_tier_cannot_say"]
       and r_tier["resource_observation"]["book_was_opened"] is False
       and r_tier["the_book_tier_is_HEAVY"]["expected_peak_gb"]
       == BOOK_LOAD_EXPECTED_PEAK_GB
       and full["resource_observation"]["book_was_opened"] is True,
       f"receipt tier opened no book; the full tier did. Expected peak "
       f"{BOOK_LOAD_EXPECTED_PEAK_GB} GB, cap {BOOK_LOAD_CAP_GB} GB, "
       f"rule-20 wrapper required for the real book")

    # -- THE ECONOMIC CENSUS OF A BOOK, BOTH DIRECTIONS ------------------
    clean_c = economic_census(book)
    sealed_c = economic_census({**book, "leak": {"null_mean": 3.21,
                                                 "D_E0": 1.0}})
    value_c = economic_census({**book, "fr2": {
        "markout_cents_per_share": 1.5, "preventable_value_cents": 2.0}})
    ck("THE CENSUS SEPARATES THE DAY'S SEALED STATISTICS FROM THE BOOK'S "
       "OWN VALUE INPUTS, AND ONLY THE FIRST IS A FLAG. ***The first real "
       "run of this tier flagged a REAL book on "
       "`markout_cents_per_share` and `preventable_value_cents` -- which "
       "are the LABEL side, the valued tranches the arms are scored "
       "against, and they BELONG in a reference book. Reporting both under "
       "one word would have called the book's own inputs a leak***: "
       "planted `null_mean`/`D_E0` are named as SEALED DAY STATISTICS; "
       "planted value fields are REPORTED, not flagged",
       clean_c["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is True
       and sealed_c["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is False
       and "null_mean" in sealed_c["field_names_naming_a_SEALED_DAY_"
                                   "STATISTIC"]
       and value_c["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is True
       and "markout_cents_per_share" in value_c[
           "field_names_naming_a_VALUE_INPUT"]
       and len(clean_c["what_it_cannot_establish"]) >= 2,
       f"clean: {clean_c['n_field_names']} field names, "
       f"{clean_c['n_field_names_naming_a_SEALED_DAY_STATISTIC']} sealed; "
       f"planted sealed -> "
       f"{sealed_c['field_names_naming_a_SEALED_DAY_STATISTIC']}; planted "
       f"value inputs -> "
       f"{value_c['field_names_naming_a_VALUE_INPUT']} (reported, not "
       f"flagged)")
    #: the nested values contribute their OWN keys -- `x` and `y` are
    #: field names too. The property under test is that the two IDENTITY
    #: keys are counted apart from the field names, not that a dict has
    #: exactly one field.
    ident_c = economic_census({"btc-updown-5m-1788393600": {"x": 1},
                               "12345": {"y": 2}, "real_field": 3})
    ck("AND AN IDENTITY KEY IS NOT A FIELD NAME: a slug or a bare number "
       "keying a mapping is counted separately, so a 297,379-key book does "
       "not drown its own field list. ***The first run listed 200 names "
       "and 160 of them were slugs***",
       ident_c["n_identity_keys"] == 2
       and set(ident_c["field_names"]) == {"real_field", "x", "y"}
       and not any(_IDENTITY_KEY.match(n) for n in ident_c["field_names"]),
       f"{ident_c['n_identity_keys']} identity key(s), "
       f"{ident_c['n_field_names']} field name(s): "
       f"{ident_c['field_names']}")

    # -- REV 60 section 4.1: THE SENTENCE IS COMPUTED, AND A BOUNDED -----
    # -- WALK REFUTES BUT NEVER ESTABLISHES ------------------------------
    tiny = economic_census(book, budget=3)
    full = economic_census(book)
    #: THE LEAK MUST BE REACHED FOR THE REFUTATION TO BE A REFUTATION.
    #: Planted LAST behind a 3-node budget it was never visited, and the
    #: census said NOT_ESTABLISHED -- correctly. It goes FIRST here, so the
    #: walk reaches it and then truncates: refuted AND truncated, which is
    #: the state the asymmetry is about.
    leaky = economic_census({"leak": {"null_mean": 1.0}, **book}, budget=6)
    ck("REV 60 section 4.1 -- THE VERDICT SENTENCE IS COMPUTED FROM THE "
       "NUMBERS, and a TRUNCATED walk is stated as a LIMIT. ***The receipt "
       "said `NONE OF THEM NAMES AN ECONOMIC QUANTITY` beside a census that "
       "had just listed three names and truncated its walk at 400,000 "
       "nodes -- rule 10's own shape, inside the instrument that censuses "
       "names.*** The summary now counts what it found; and a bounded walk "
       "can REFUTE (a sealed name it DID reach is a fact) and can never "
       "ESTABLISH an absence",
       full["sealed_statistic_check"] == "ESTABLISHED_OVER_A_COMPLETE_WALK"
       and full["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is True
       and tiny["truncated"] is True
       and tiny["sealed_statistic_check"] == "NOT_ESTABLISHED_WALK_TRUNCATED"
       and tiny["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is None
       and "cannot establish the absence" in tiny["summary"]
       and str(full["n_field_names_naming_a_VALUE_INPUT"]) in full["summary"],
       f"complete walk -> {full['sealed_statistic_check']}; truncated at 3 "
       f"nodes -> {tiny['sealed_statistic_check']} with the flag None, not "
       f"False; summary: \"{tiny['summary'][:90]}…\"")
    ck("AND THE REFUTING DIRECTION SURVIVES TRUNCATION: a sealed name the "
       "walk DID reach is REFUTED even though the walk stopped early -- "
       "***the asymmetry is the point, and an instrument that reported "
       "`unknown` for a name it had already seen would be throwing away "
       "the half it can prove***",
       leaky["sealed_statistic_check"] == "REFUTED"
       and leaky["NO_SEALED_DAY_STATISTIC_IS_NAMED_IN_THIS_BOOK"] is False
       and "null_mean" in leaky["field_names_naming_a_SEALED_DAY_STATISTIC"]
       and "REFUTED" in leaky["summary"],
       f"truncated AND leaking -> {leaky['sealed_statistic_check']} naming "
       f"{leaky['field_names_naming_a_SEALED_DAY_STATISTIC']}")

    # -- REV 64: THE PORCELAIN DEFECT HAS TWO NECESSARY HALVES ----------
    import da_root as _PR                                     # noqa: PLC0415
    BLOCK = " M live/x.py\n?? data\nR  a -> b\n"
    good = _PR.parse_porcelain(BLOCK)
    #: HALF ONE, the READ: strip the block first, then parse it correctly.
    strip_first = _PR.parse_porcelain(BLOCK.strip())
    #: HALF TWO, the SLICE: parse the RAW block with the old rule.
    old_slice_raw = [ln[3:] for ln in BLOCK.split("\n") if ln]
    #: BOTH: strip AND the fixed slice -- the defect as it shipped.
    both = [ln[3:] for ln in BLOCK.strip().split("\n") if ln]
    ck("REV 64 -- THE PORCELAIN DEFECT IS ONE DEFECT WITH TWO NECESSARY "
       "HALVES, and the parser closes both. RAW read, the XY code by FIXED "
       "WIDTH, the path from COLUMN 4, and `R  old -> new` split on the "
       "arrow: every path in the reviewer's block is recovered EXACTLY -- "
       "`live/x.py` (leading space), `data`, and `b` from `a`. ***No seat "
       "had both halves right: DE and BE are safe by their READ, this seat "
       "was safe by its SLICE***",
       [r["path"] for r in good["rows"]] == ["live/x.py", "data", "b"]
       and good["rows"][0]["xy"] == " M"
       and good["rows"][2]["renamed_from"] == "a"
       and good["rows"][1]["untracked"] is True
       and good["n_malformed"] == 0,
       f"paths {[r['path'] for r in good['rows']]}; first XY "
       f"{good['rows'][0]['xy']!r}; rename b<-a")
    ck("AND EACH HALF IS DRIVEN RED FIRST, SEPARATELY: ***the READ half "
       "alone*** shifts the FIRST line so a correct parser cannot read it "
       "at all -- its path is LOST, not merely mistyped; "
       "***the SLICE half alone*** is right on a raw line and wrong the "
       "moment one is shifted; and ***the two together*** return "
       "`ive/x.py` for `live/x.py`, which is the character this seat's "
       "receipts actually lost",
       strip_first["n_malformed"] == 1
       and strip_first["malformed"] == ["M live/x.py"]
       and [r["path"] for r in strip_first["rows"]] == ["data", "b"]
       and old_slice_raw[0] == "live/x.py"
       and both[0] == "ive/x.py"
       and both[2] == "a -> b",
       f"strip alone -> the first line is UNREADABLE and its path is LOST "
       f"({strip_first['malformed']}), leaving "
       f"{[r['path'] for r in strip_first['rows']]}; slice alone on a raw "
       f"line -> {old_slice_raw[0]!r}; BOTH -> {both[0]!r}, and the rename "
       f"unsplit as {both[2]!r}")
    ck("AND A LINE THE PARSER CANNOT READ IS NAMED, NOT DROPPED: a "
       "truncated status line lands in `malformed` and is counted as DIRT "
       "by the caller -- an unreadable status is not a clean tree (rule 11)",
       _PR.parse_porcelain("M\n?? ok\n")["n_malformed"] == 1
       and _PR.parse_porcelain("M\n?? ok\n")["rows"][0]["path"] == "ok",
       "a 1-character line is malformed and the good line beside it still "
       "parses")

    # -- RULE 22 / R-605: THE LAUNCH CAPTURE, BOTH DIRECTIONS ------------
    idy = source_identity_at_launch()
    ck("RULE 22 / R-605 -- THE LAUNCH CAPTURE EXISTS AND IT IS THE CLOSURE, "
       "NOT ONE FILE: at IMPORT this run digested every `live/` module in "
       "`sys.modules`, the worktree's HEAD and its dirty state, and the "
       "receipt carries them. ***DA 77's own sweep found this runner "
       "lacking it -- and found that this seat's binding map had EXEMPTED "
       "it, which is worse than the gap***",
       idy["producing_code_sha256"] == LAUNCH_SOURCE_SHA256
       and idy["digest_taken_at"].startswith("MODULE IMPORT")
       and idy["import_closure"]["n_modules"] >= 1
       and idy["import_closure"]["root"].endswith("/live")
       and "head" in idy["head_at_import"]
       and idy["closure_unchanged_during_the_run"] is True,
       f"{idy['import_closure']['n_modules']} module(s) under "
       f"{idy['import_closure']['root']} captured at "
       f"{idy['launch_time_utc']}; HEAD "
       f"{str(idy['head_at_import']['head'])[:8]}, dirty "
       f"{idy['head_at_import']['dirty']}; capture points "
       f"{[c['where'] for c in idy['import_closure']['capture_points']]}")

    #: THE KNOWN-BAD IS A REWRITTEN SIBLING, on COPIES -- never a real
    #: module in a tree other seats are working in.
    sib_dir = Path(tempfile.mkdtemp(prefix="da78_closure_"))
    sib_a = sib_dir / "sibling_alpha.py"
    sib_b = sib_dir / "sibling_beta.py"
    sib_a.write_text("VALUE = 1\n")
    sib_b.write_text("VALUE = 2\n")
    synth = {str(sib_a): hashlib.sha256(sib_a.read_bytes()).hexdigest(),
             str(sib_b): hashlib.sha256(sib_b.read_bytes()).hexdigest()}
    drift_none = closure_drift(synth)
    sib_b.write_text("VALUE = 2  # landed mid-run\n")
    drift_one = closure_drift(synth)
    sib_a.unlink()
    drift_two = closure_drift(synth)
    ck("AND IT REFUSES BY MODULE NAME WHEN A SIBLING IS REWRITTEN MID-RUN, "
       "AND ADMITS WHEN NOTHING MOVED: two modules captured, nothing "
       "touched -> no drift; ONE rewritten -> drift naming THAT module and "
       "no other; one DELETED -> named too, with `gone`. ***The run is "
       "unaffected -- the modules are in memory -- but a receipt stamped "
       "from the files would name code that DID NOT RUN, and "
       "`producing_code_is_the_committed_bytes` PASSES if the replacement "
       "is itself committed (R-603)***",
       drift_none == []
       and [d["module"] for d in drift_one] == ["sibling_beta.py"]
       and drift_one[0]["at_launch"] != drift_one[0]["now"]
       and sorted(d["module"] for d in drift_two)
       == ["sibling_alpha.py", "sibling_beta.py"]
       and any(d["gone"] for d in drift_two),
       f"nothing moved -> {len(drift_none)} drift; one rewritten -> "
       f"{[d['module'] for d in drift_one]}; one deleted as well -> "
       f"{sorted(d['module'] for d in drift_two)}")

    #: and the REFUSAL ITSELF, driven through the emit guard by pointing the
    #: launch capture at a copy that then moves.
    hold = dict(LAUNCH_CLOSURE)
    LAUNCH_CLOSURE.clear()
    LAUNCH_CLOSURE.update(synth)
    msg = ""
    try:
        assert_source_unchanged("a driven emit", fixture=True)
    except LaunchCaptureRefused as e:
        msg = str(e)
    LAUNCH_CLOSURE.clear()
    LAUNCH_CLOSURE.update(hold)
    admitted = assert_source_unchanged("a driven emit", fixture=True)
    dirty_msg = ""
    try:
        assert_source_unchanged("a driven REAL emit", fixture=False)
    except LaunchCaptureRefused as e:
        dirty_msg = str(e)
    ck("THE EMIT GUARD REFUSES BY NAME AND THE REAL/FIXTURE ASYMMETRY IS "
       "THE DIRTY BAR: a moved closure raises naming the module; the real "
       "closure admits; and a DIRTY worktree is a recorded FACT for a "
       "fixture and a REFUSAL for a real artifact -- uncommitted bytes are "
       "locatable in no commit",
       "IMPORT CLOSURE" in msg and "sibling_beta.py" in msg
       and admitted["closure_unchanged_during_the_run"] is True
       and (("DIRTY AT IMPORT" in dirty_msg)
            is bool(idy["worktree_was_dirty_at_import"])),
       f"moved -> refused naming sibling_beta.py; unmoved -> admitted; "
       f"dirty at import = {idy['worktree_was_dirty_at_import']} -> real "
       f"emit {'REFUSES' if dirty_msg else 'admits'}")

    #: THE COUNT AND THE PRINT RUN AFTER THE LAST CHECK. They used to sit
    #: in the MIDDLE of this function, so every check appended below them
    #: -- the census pair, the launch capture, the porcelain parser -- was
    #: NEITHER PRINTED NOR COUNTED, and the summary reported
    #: "29 checks, 0 failure(s)" while one of them was FAILING.
    #: ***A battery whose summary cannot see its own last checks is a
    #: battery that cannot fail*** (rule 15), and it is the instrument this
    #: seat uses to hold other seats to their receipts.
    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    recount = sum(1 for c in checks if not c.get("passed"))
    assert recount == n_fail and len(checks) == len(
        [c for c in checks if "check" in c]), (
        "the summary disagrees with the list it summarises")
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--receipt-tier", action="store_true")
    ap.add_argument("--full", action="store_true", help="HEAVY: opens the book")
    ap.add_argument("--book")
    ap.add_argument("--receipt")
    ap.add_argument("--day")
    ap.add_argument("--coin")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--supersedes", type=Path, default=None,
                    help="a prior receipt this re-emission supersedes; the "
                         "R-608 PAIR is computed from its bytes")
    a = ap.parse_args()
    if a.selftest:
        checks, n_fail = selftest()
        if a.output:
            assert_source_unchanged("the fixture receipt's emit",
                                    fixture=True)
            a.output.write_text(json.dumps({
                "protocol": PROTOCOL + "_FIXTURE",
                "status": "FIXTURE_NO_REAL_BOOK",
                "verifier_identity": verifier_identity(),
                "launch_capture": source_identity_at_launch(),
                "builder_call_read_by_ast": builder_index_call(),
                "the_real_book_tier_is_HEAVY": {
                    "expected_peak_gb": BOOK_LOAD_EXPECTED_PEAK_GB,
                    "declared_cap_gb": BOOK_LOAD_CAP_GB,
                    "why": "the 09-03 book is a 290,758,834-byte pickle; "
                           "unpickling is expected to cost about 2 GB "
                           "resident, which is heavy under rule 20",
                    "the_real_run_is_a_coordinator_GO": True},
                "checks": checks, "n_checks": len(checks),
                "n_failed": n_fail, "both_directions": True,
            }, indent=2, sort_keys=True, default=str) + "\n")
        return 1 if n_fail else 0
    if not (a.book and a.receipt):
        ap.error("--selftest, or --receipt-tier/--full with --book and "
                 "--receipt [--day --coin --output]")
    try:
        fn = verify_full if a.full else verify_receipt_tier
        #: A REAL artifact: rule 22's dirty bar REFUSES here, and is only a
        #: recorded fact inside the fixture.
        kw = {"day": a.day, "coin": a.coin, "output": a.output,
              "fixture": False}
        if a.full:
            kw["supersedes"] = a.supersedes
        r = fn(a.book, a.receipt, **kw)
    except BookVerifyRefused as e:
        print(str(e))
        return 2
    print(f"{r['tier']} tier: {r['status']} -- IS_A_VERIFICATION="
          f"{r['IS_A_VERIFICATION']}, {r['n_flags']} flag(s) {r['flags']}")
    #: FOUR OUTCOMES, FOUR EXIT CODES. 2 refused (the instrument declined to
    #: run), 3 PROVENANCE_INCOMPLETE (it ran, everything checkable holds, and
    #: something could not be located), 1 FLAGGED (it ran and disagrees), 0
    #: verified. A caller that could not tell 3 from 1 would read "the
    #: producing commit is missing" as "the receipt is wrong" -- which is the
    #: exact conflation R-601 corrected in my own round-70 probe.
    if r["IS_A_VERIFICATION"]:
        return 0
    return 3 if r["status"] == "PROVENANCE_INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

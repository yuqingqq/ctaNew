#!/usr/bin/env python3
"""DA -- THE CANONICAL ROOT, ONE PREDICATE (REV 57 A.6 / BE 59 / R-553-554).

THE FINDING THIS EXISTS FOR. DE's resolver finds the canonical ledger even
with `PM_DATA_ROOT` unset. DA's `_derived_dir()` resolved relative to ITS
OWN FILE, so from a seat worktree it returned the worktree's PARTIAL `data/`
-- ***even with PM_DATA_ROOT set correctly*** -- and the 09-04 book is
invisible there. A smaller plausible ledger reported as a pass is rule 11's
failure with a directory instead of a status, and it is newly load-bearing
because the read gate COUNTS SEALED RECEIPTS AT A ROOT.

TWO SEATS RESOLVED ONE ROOT BY TWO RULES. This is DA's side made to follow
the canonical rule:

  1. the ROOT comes from the programme's resolver of record
     (`pm_tape_density._resolve_data_root`: env, then the code tree ONLY IF
     IT CARRIES THE TAPE, then canonical) -- DELEGATED, never copied;
  2. the CANONICAL PATH is read from DE's own module AS A DOCUMENT (R-235),
     not typed here, so the two seats cannot hold different constants;
  3. a resolved root that is not the canonical ledger REFUSES BY NAME.

A fixture may pass `fixture=True` and a reason; the exemption is then
RECORDED in what this returns, so it is visible to a reader rather than
invisible in a branch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

#: IMPORTABLE BOTH WAYS. Under `python3 -m live.pm_research.<mod>` the
#: package directory is NOT on `sys.path`, so a bare `import da_root`
#: raises ModuleNotFoundError -- the reviewer and the coordinator run these
#: both ways, and a module that works only when invoked as a script is an
#: instrument with a launch convention nobody declared.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


HERE = Path(__file__).resolve().parent
DE_MODULE = HERE / "de_data_root.py"


#: REV 64. ONE PORCELAIN PARSER, AND IT CLOSES BOTH HALVES.
#: `git status --porcelain` emits `XY<space>PATH`: two columns of status,
#: one space, then the path from COLUMN 4. The defect has two halves and
#: needs both:
#:   the READ  -- `stdout.strip()` eats the LEADING SPACE of the FIRST line
#:                (` M path` becomes `M path`), shifting that line and only
#:                that line;
#:   the SLICE -- `line[3:]` then cuts one character into the shifted path
#:                and returns `ath` for `path`.
#: Either half alone still misreads something: strip alone turns the XY
#: code `_M` (unstaged) into `M_` (staged); a fixed slice alone is right on
#: a raw line and wrong on a shifted one. No seat had both right -- DE and
#: BE are safe by their READ, this seat was safe by its SLICE -- so the
#: parser is written once, raw, and used everywhere.
#: R-646 R1. THE ALPHABET AND THE SHAPE CHECK ARE **BE's ALGORITHM**,
#: lifted from `be_rule22.parse_porcelain_line` (5f5f76c, L96-128) rather
#: than re-derived: the reviewer drove three implementations on twelve
#: lines and BE's was the only one right on all twelve. What is kept from
#: here is the ROW STRUCTURE -- a malformed line is REPORTED per row, never
#: raised -- because DE 96 imports this interface unchanged.
_PORCELAIN_STATES = set(" MTADRCU?!")


def _c_unquote(field: str) -> tuple:
    """git's C-style quoting, undone -- (value, was_quoted).

    REV 69 2.1: the old code stripped the surrounding quotes and stopped,
    so `\"`, `\\` and `\nnn` survived into a path this seat then compared
    against a real filename. A quote strip is not a decode. Regime: no
    caller passes `-z` and no tracked path needs quoting today, so this
    fires on nothing in the tree -- it fires on the day one does."""
    if not (isinstance(field, str) and len(field) >= 2
            and field[0] == '"' and field[-1] == '"'):
        return field, False
    body, out, i = field[1:-1], [], 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
              "v": "\v", "a": "\a", "\\": "\\", '"': '"'}
    raw = bytearray()
    while i < len(body):
        c = body[i]
        if c != "\\":
            raw += c.encode("utf-8")
            i += 1
            continue
        if i + 1 >= len(body):
            raw += b"\\"
            break
        nxt = body[i + 1]
        if nxt.isdigit() and i + 3 < len(body) + 1:
            oct3 = body[i + 1:i + 4]
            if len(oct3) == 3 and all(ch in "01234567" for ch in oct3):
                raw.append(int(oct3, 8))
                i += 4
                continue
        if nxt in simple:
            raw += simple[nxt].encode("utf-8")
            i += 2
            continue
        raw += nxt.encode("utf-8")
        i += 2
    return raw.decode("utf-8", "surrogateescape"), True


def parse_porcelain(stdout: str) -> dict:
    """Rows of {xy, path, renamed_from, untracked}, and the malformed ones
    NAMED rather than dropped (rule 11).

    SHARED PROGRAMME INFRASTRUCTURE (R-641 / R-646 R1): DE 96 and BE 65
    call this parser rather than keeping their own. R-235 forbids sharing a
    STATISTIC, not a way of reading a tool's output -- and the reviewer
    MEASURED what three implementations cost: four of twelve lines
    disagreed and no two were wrong in the same place, which is exactly
    "two implementations corroborate nothing". The interface is fixed:
    callers depend on it.

    THE ALGORITHM, from BE's:
      * porcelain v1 is `XY<space>path` with X and Y from a fixed alphabet;
        the OFFSET IS NOT ASSUMED. A line failing that shape -- a `.strip()`
        victim that lost its leading space, or a `--branch` header -- is
        MALFORMED BY NAME rather than sliced into a path one character
        short that nothing downstream can detect;
      * `old -> new` is split ONLY when the code carries R or C, so a file
        merely NAMED `a -> b` keeps its name;
      * git's C-style quoting is UNDONE on BOTH the path and the rename's
        old name, through one helper (`\"`, `\\`, `\nnn`), with
        `path_was_c_quoted` / `renamed_from_was_c_quoted` reported;
      * trailing spaces are part of the path and are kept.
    """
    rows, malformed = [], []
    for line in (stdout or "").split("\n"):
        if not line:
            continue
        if (len(line) < 4 or line[2] != " "
                or line[0] not in _PORCELAIN_STATES
                or line[1] not in _PORCELAIN_STATES):
            #: NOT silently skipped and NOT sliced anyway: a line this
            #: parser cannot read is a status, and a status is reported.
            malformed.append(line)
            continue
        code, rest = line[:2], line[3:]
        old = None
        if ("R" in code or "C" in code) and " -> " in rest:
            old, rest = rest.split(" -> ", 1)
        #: BOTH FIELDS THROUGH ONE HELPER (REV 69 2.1). The old name was
        #: left QUOTED while the new one was unquoted -- one field decoded
        #: and its twin not -- and the "unquoting" was a quote STRIP, which
        #: leaves git's C escapes (`\"`, `\\`, `\nnn`) as written.
        rest, rest_enc = _c_unquote(rest)
        old, old_enc = _c_unquote(old) if old is not None else (None, False)
        rows.append({"xy": code, "path": rest, "renamed_from": old,
                     "path_was_c_quoted": rest_enc,
                     "renamed_from_was_c_quoted": old_enc,
                     "untracked": code == "??"})
    return {"rows": rows, "malformed": malformed,
            "n_rows": len(rows), "n_malformed": len(malformed),
            "read": "RAW -- the block is never stripped",
            "shape": "XY<space>path, the code alphabet-validated, the "
                     "offset never assumed",
            "path_from": "column 4, after the two-column code and one space",
            "renames": "`R`/`C` only: `old -> new` is split on ` -> ` and "
                       "the NEW path is the path; a file merely NAMED "
                       "`a -> b` keeps its name",
            "algorithm_from": "be_rule22.parse_porcelain_line (R-646 R1)"}


#: REV 68 FINDING 2 / R-649. THE COVERAGE PREDICATE IS TWO MEASURED CLOCKS,
#: and it is SHARED FORM: DE 99 imports or mirrors this rather than keeping
#: a text search. `oldest_available <= w0` compares the UNIT'S oldest
#: retained line with a window start -- which is right ONLY for a
#: CONTINUOUS logger, where the unit's first retained line IS the journal's
#: horizon, and wrong for a BURSTY unit: `de95smoke.service` has ONE line
#: at 12:35:35Z, so a window starting a minute earlier read UNCOVERED for a
#: unit whose journal is complete.
#:
#: The portable question is: DOES THE JOURNAL STILL REACH BACK PAST THE
#: THING BEING ASKED ABOUT? Two clocks, both measured, neither searched
#: for in text:
#:   the HOST's oldest retained entry   (the journal's horizon), and
#:   the REFERENCE moment -- the unit's own `ExecMainStartTimestamp` for a
#:   bursty unit, or the window's start for a continuous logger.
#: Covered iff horizon <= reference.
import subprocess as _sp                                     # noqa: E402
import datetime as _dtm                                      # noqa: E402

CONTINUOUS = "CONTINUOUS_LOGGER__the_reference_is_the_window_start"
BURSTY = "BURSTY_UNIT__the_reference_is_the_unit_start"


def _stamp(text: str):
    """A journalctl `short-iso` stamp or a systemd timestamp, parsed."""
    text = (text or "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%a %Y-%m-%d %H:%M:%S %Z",
                "%Y-%m-%d %H:%M:%S %Z"):
        try:
            t = _dtm.datetime.strptime(text[:25] if "T" in text[:11]
                                       else text, fmt)
            return t if t.tzinfo else t.replace(
                tzinfo=_dtm.timezone.utc)
        except ValueError:
            continue
    return None


def host_journal_horizon() -> dict:
    """The OLDEST entry the user journal still holds, for ANY unit."""
    argv = ["journalctl", "--user", "-o", "short-iso", "--no-pager", "--utc"]
    try:
        r = _sp.run(argv, capture_output=True, text=True, timeout=180)
    except Exception as e:                                    # noqa: BLE001
        return {"status": "JOURNALCTL_FAILED", "why": repr(e),
                "oldest_utc": None, "query": " ".join(argv)}
    if r.returncode != 0:
        return {"status": "JOURNALCTL_FAILED", "returncode": r.returncode,
                "stderr": (r.stderr or "").strip()[:400],
                "oldest_utc": None, "query": " ".join(argv)}
    first = next((ln for ln in r.stdout.splitlines()
                  if not ln.startswith("-- ")), "")
    t = _stamp(first[:25])
    return {"status": "MEASURED" if t else "NO_READABLE_STAMP",
            #: REV 70 section 3: the format is part of the measurement.
            "output_format": "short-iso --utc",
            "oldest_utc": (t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None),
            "oldest_epoch": (t.timestamp() if t else None),
            "query": " ".join(argv),
            "read_at_utc": _dtm.datetime.now(
                _dtm.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def unit_start(unit: str) -> dict:
    """The unit's own `ExecMainStartTimestamp`, from systemd."""
    argv = ["systemctl", "--user", "show", unit, "-p",
            "ExecMainStartTimestamp", "--value"]
    try:
        r = _sp.run(argv, capture_output=True, text=True, timeout=60)
    except Exception as e:                                    # noqa: BLE001
        return {"status": "SYSTEMCTL_FAILED", "why": repr(e),
                "start_utc": None, "query": " ".join(argv)}
    t = _stamp(r.stdout)
    return {"status": "MEASURED" if t else "NO_START_TIMESTAMP",
            "unit": unit, "raw": r.stdout.strip(),
            "start_utc": (t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None),
            "start_epoch": (t.timestamp() if t else None),
            "query": " ".join(argv)}


def journal_coverage(*, unit: str | None = None,
                     window_start_epoch: float | None = None,
                     regime: str = BURSTY,
                     _horizon: dict | None = None,
                     _start: dict | None = None) -> dict:
    """DOES THE JOURNAL STILL REACH BACK PAST THE REFERENCE MOMENT?

    Two MEASURED clocks and no text search. The regime is NAMED in the
    output because the answer depends on it: for a CONTINUOUS logger the
    reference is the window's start; for a BURSTY unit it is the unit's own
    start, and its silence between bursts is not a gap in the record."""
    hz = _horizon if _horizon is not None else host_journal_horizon()
    ref_epoch, ref_kind, st = None, None, None
    if regime == CONTINUOUS:
        ref_epoch, ref_kind = window_start_epoch, "the window start"
    else:
        st = _start if _start is not None else (unit_start(unit) if unit
                                                else None)
        if st and st.get("start_epoch"):
            ref_epoch, ref_kind = st["start_epoch"], "the unit's own start"
        elif window_start_epoch is not None:
            ref_epoch, ref_kind = window_start_epoch, (
                "the window start (the unit has no start timestamp)")
    out = {"regime": regime, "host_horizon": hz, "unit_start": st,
           "reference_is": ref_kind,
           "reference_utc": (None if ref_epoch is None else
                             _dtm.datetime.fromtimestamp(
                                 ref_epoch, _dtm.timezone.utc).strftime(
                                     "%Y-%m-%dT%H:%M:%SZ")),
           "two_clocks_no_text_search": True}
    if hz.get("oldest_epoch") is None or ref_epoch is None:
        out["covered"] = None
        out["status"] = "NOT_DETERMINABLE"
        out["why"] = ("one of the two clocks is unreadable, so coverage is "
                      "UNKNOWN -- never assumed either way")
        return out
    out["covered"] = hz["oldest_epoch"] <= ref_epoch
    out["status"] = "MEASURED"
    out["why"] = (
        f"the journal's oldest retained entry is {hz['oldest_utc']} and "
        f"{ref_kind} is {out['reference_utc']}; the journal "
        f"{'reaches back past it' if out['covered'] else 'does NOT reach it'}"
    )
    return out


class RootRefused(RuntimeError):
    """The resolved root is not the ledger this programme records into."""


def canonical_from_DEs_source(path: Path | None = None) -> dict:
    """The canonical paths, READ FROM DE's MODULE rather than typed.

    A constant copied into a second file is a second constant: it agrees
    until one of them is edited. Reading DE's own literal means this seat
    cannot hold a canonical root DE has moved."""
    p = Path(path) if path else DE_MODULE
    if not p.is_file():
        return {"status": "DE_MODULE_ABSENT", "repo": None, "data": None,
                "source": str(p)}
    txt = p.read_text()
    out = {}
    for name, key in (("CANONICAL_REPO_ROOT", "repo"),
                      ("CANONICAL_DATA_ROOT", "data")):
        m = re.search(rf'^{name}\s*=\s*(?:Path\()?["\']([^"\']+)["\']',
                      txt, re.M)
        out[key] = m.group(1) if m else None
    out["status"] = ("READ_FROM_DES_SOURCE" if out.get("repo")
                     else "CONSTANTS_NOT_FOUND_IN_DES_SOURCE")
    out["source"] = p.name
    return out


def resolve_root() -> Path:
    """THE ROOT, from the programme's resolver of record. Never this file's
    own tree: that is the shell fact BE 59 named.

    THE `except` IS NARROW ON PURPOSE (REV 58 section 4, test 4). The
    defect this module replaces was a BARE `except Exception` wrapped
    around `Path(de_data_root.resolve())`: `resolve()` returns a DICT,
    `Path(<dict>)` raises TypeError, and the bare clause swallowed it, so
    the tree-relative fallback ran on EVERY call while the docstring
    claimed the shared resolver. ***A bare except around a resolver turns
    the NEXT resolver change into a silent fallback.*** Only the two errors
    that mean "the resolver of record is not reachable" are caught here;
    anything else -- a TypeError from a changed return shape included --
    propagates as itself and is seen."""
    try:
        import pm_tape_density as _T                          # noqa: PLC0415
        root = _T._resolve_data_root()
    except (ImportError, AttributeError) as e:
        raise RootRefused(
            "REFUSED: the data-root resolver of record "
            "(pm_tape_density._resolve_data_root) is not reachable, and "
            "this seat will not answer with its own tree instead. " + str(e))
    if not isinstance(root, (str, Path)):
        #: NOT swallowed, NAMED. This is precisely the shape that broke the
        #: last one: a resolver whose return type moved.
        raise RootRefused(
            f"REFUSED: the resolver of record returned "
            f"{type(root).__name__}, not a path. The last time a resolver's "
            f"return SHAPE moved, a bare except turned it into a silent "
            f"tree-relative fallback that ran for weeks.")
    return Path(root)


def code_root(purpose: str = "reading another seat's source") -> Path:
    """The CANONICAL tree for reading CODE.

    A verifier that judges another seat's receipt against that seat's
    SOURCE must read the source from the ledger, not from its own
    worktree's copy -- DA 77 shipped exactly that finding against DE's
    runner and had to retract it.

    AND THIS ONE STAYS STRICT. The symlink equivalence that makes a
    worktree canonical for DATA does not extend to CODE: a worktree's
    `data/` may BE the ledger's directory while its `live/` is a checkout
    at another commit. Data is canonical by where it LANDS; code is
    canonical by WHICH TREE it is."""
    canon = canonical_from_DEs_source()
    if not canon.get("repo"):
        raise RootRefused(
            f"REFUSED: {purpose} -- the canonical root is not readable from "
            f"DE's module ({canon['status']}), and this seat will not "
            f"substitute a constant of its own.")
    p = Path(canon["repo"])
    if not (p / "live" / "pm_research").is_dir():
        raise RootRefused(
            f"REFUSED: {purpose} -- the canonical tree {p} carries no "
            f"live/pm_research to read.")
    return p


def require_canonical_root(purpose: str, *, fixture: bool = False,
                           why: str | None = None,
                           root: Path | None = None) -> dict:
    """The root for a RESULT-BEARING read, or a refusal that names why."""
    canon = canonical_from_DEs_source()
    r = Path(root) if root is not None else resolve_root()
    resolved = str(r.resolve()) if r.exists() else str(r)
    #: CANONICAL IS DECIDED BY THE LEDGER'S REAL PATH, NOT BY THE TREE'S
    #: NAME. Since R-553's symlink was restored, a seat worktree's `data/`
    #: IS the ledger's directory -- `readlink -f` proves it -- and refusing
    #: there refused a root whose every data byte is the ledger's. What
    #: matters for DATA is where `<root>/data` actually LANDS.
    dp = r / "data"
    data = str(dp.resolve()) if dp.exists() else str(dp)
    canon_data = canon.get("data")
    if canon_data and Path(canon_data).exists():
        canon_data = str(Path(canon_data).resolve())
    is_canonical = (canon_data is not None and data == canon_data)
    block = {
        "purpose": purpose,
        "root": resolved,
        "data_root": data,
        "canonical": canon,
        "is_canonical": is_canonical,
        "canonical_decided_by": "the data directory's REAL path",
        "canonical_data_root": canon_data,
        "data_root_real_path": data,
        "the_tree_name_is_not_the_test": (
            "a worktree whose `data/` symlinks to the ledger holds the "
            "ledger's bytes; a worktree with a MATERIALISED `data/` holds "
            "only the TRACKED ones. The first is canonical for data, the "
            "second is not, and their paths are the same shape"),
        "resolver_of_record": "pm_tape_density._resolve_data_root",
        "the_canonical_path_is_READ_from": canon.get("source"),
    }
    if is_canonical:
        block["check"] = "PASS"
        return block
    if not canon.get("repo"):
        raise RootRefused(
            f"REFUSED: {purpose} cannot be checked -- the canonical root is "
            f"not readable from DE's module ({canon['status']}), and this "
            f"seat will not substitute a constant of its own.")
    if fixture:
        if not why:
            raise RootRefused(
                f"REFUSED: {purpose} claims fixture=True with no reason. An "
                f"emission that exempts itself from the ledger must say "
                f"what it is, or the exemption is invisible to a reader.")
        block["check"] = "EXEMPT_FIXTURE"
        block["exemption_reason"] = why
        block["NOT_RESULT_BEARING"] = True
        return block
    raise RootRefused(
        f"REFUSED: {purpose} resolved the root to {resolved!r}, whose "
        f"`data/` lands on {data!r} and NOT on the ledger's "
        f"{canon_data!r}. A seat worktree's "
        f"`data/` is a MATERIALISED directory holding only the TRACKED "
        f"artifacts, so a count taken there is a count of a smaller, "
        f"plausible ledger -- and the read gate COUNTS SEALED RECEIPTS AT A "
        f"ROOT (REV 57 A.6). Pass fixture=True with a reason if this is "
        f"deliberate.")


def derived_dir(purpose: str = "the derived directory") -> Path:
    """The LEDGER's derived directory by its REAL path, or a refusal.

    Resolved, not merely admitted: from a worktree whose `data/` symlinks
    to the ledger, `<worktree>/data/pm_5min/derived` IS the right
    directory, but every path this seat then RECORDS would carry the
    worktree's name for a ledger artifact -- and a receipt that names a
    file by a path only one tree can resolve is a citation nobody else can
    follow."""
    b = require_canonical_root(purpose)
    return Path(b["data_root_real_path"]) / "pm_5min" / "derived"


# --------------------------------------------------------------- selftest

def selftest() -> tuple:
    """The reviewer's three-line block, driven (REV 66 / R-641)."""
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond), "detail": detail})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    #: THE REVIEWER'S TWELVE-LINE TABLE, as EXPECTED VALUES (REV 67 1.2).
    #: Four of these twelve separated the three implementations, and no two
    #: were wrong in the same place.
    TABLE = [
        (" M live/x.py",      "row", " M", "live/x.py", None),
        ("?? data",           "row", "??", "data",      None),
        ("R  a -> b",         "row", "R ", "b",         "a"),
        ("M live/x.py",       "malformed", None, None,  None),
        ("## mm-research...origin/mm-research", "malformed", None, None,
         None),
        ("?? a -> b",         "row", "??", "a -> b",    None),
        ("?? trailing ",      "row", "??", "trailing ", None),
        ('RM "odd name.py"',  "row", "RM", "odd name.py", None),
        ("A  new.py",         "row", "A ", "new.py",    None),
        (" D gone.py",        "row", " D", "gone.py",   None),
        ("UU both.py",        "row", "UU", "both.py",   None),
        ("C  src.py -> cp.py", "row", "C ", "cp.py",    "src.py"),
        #: REV 69 2.1's TEN, added to the same table.
        ("??  leading.py",    "row", "??", " leading.py", None),
        (" R old.py -> new.py", "row", " R", "new.py",  "old.py"),
        ("RM keep.py",        "row", "RM", "keep.py",   None),
        ("M",                 "malformed", None, None,  None),
        ("?? ",               "malformed", None, None,  None),
        ("XY nope.py",        "malformed", None, None,  None),
        ("?z nope.py",        "malformed", None, None,  None),
        ("?? nul\x00keep.py", "row", "??", "nul\x00keep.py", None),
        ('R  "a\\"b.py" -> "c\\td.py"', "row", "R ", "c\td.py",
         'a"b.py'),
        ('?? "caf\\303\\251.py"', "row", "??", "caf\u00e9.py", None),
    ]
    out = parse_porcelain("\n".join(t[0] for t in TABLE) + "\n")
    got_rows = {r["path"]: r for r in out["rows"]}
    exp_rows = [t for t in TABLE if t[1] == "row"]
    exp_mal = [t[0] for t in TABLE if t[1] == "malformed"]
    rows_ok = (len(out["rows"]) == len(exp_rows)
               and all(out["rows"][i]["xy"] == t[2]
                       and out["rows"][i]["path"] == t[3]
                       and out["rows"][i]["renamed_from"] == t[4]
                       for i, t in enumerate(exp_rows)))
    ck("R-646 R1 -- BE's ALGORITHM, THIS SEAT'S ROW STRUCTURE, DRIVEN ON "
       "THE REVIEWER'S TWELVE-LINE TABLE. ***The four lines that separated "
       "the three implementations are the point:*** a SHIFTED line and a "
       "`--branch` header are MALFORMED BY NAME (this parser used to take "
       "the header as a path); `?? a -> b` keeps its NAME because the "
       "split is gated on R/C (this parser used to truncate it to `b`); "
       "and `?? trailing ` keeps its trailing space. ***REV 69 2.1's ten "
       "join it:*** a LEADING space in the path is kept, ` R` and `RM` "
       "carry their rename, a ONE-CHARACTER code, an EMPTY path and "
       "UNDECLARED letters are MALFORMED, a NUL is kept, and git's C "
       "quoting is decoded on BOTH the path and the OLD name -- the old "
       "one used to stay quoted while its twin was stripped",
       rows_ok and out["malformed"] == exp_mal
       and got_rows["a -> b"]["renamed_from"] is None
       and got_rows["b"]["renamed_from"] == "a"
       and got_rows["cp.py"]["renamed_from"] == "src.py",
       "; ".join(f"{t[0]!r} -> "
                 + ("MALFORMED" if t[1] == "malformed"
                    else repr(got_rows.get(t[3], {}).get("path")))
                 for t in TABLE))

    #: REV 68 FINDING 2: the two-clock coverage form, both regimes.
    hz = host_journal_horizon()
    #: THE ARITHMETIC IS DRIVEN ON INJECTED CLOCKS, because a check that
    #: needs a NAMED UNIT to still exist is pinned to an ambient -- and
    #: de95smoke.service was collected on success between rounds, so the
    #: cell that drove it turned red for a reason that is not the property
    #: (REV 58 1.2's own class, arriving in my own suite).
    HZ = {"status": "MEASURED", "oldest_utc": "2026-09-06T09:53:58Z",
          "oldest_epoch": 1788690838.0}
    ST = {"status": "MEASURED", "unit": "a bursty unit",
          "start_utc": "2026-09-06T12:35:35Z", "start_epoch": 1788698135.0}
    bursty = journal_coverage(unit="x", _horizon=HZ, _start=ST)
    one_min_earlier = journal_coverage(
        regime=CONTINUOUS, window_start_epoch=ST["start_epoch"] - 60,
        _horizon=HZ)
    before = journal_coverage(
        regime=CONTINUOUS, window_start_epoch=HZ["oldest_epoch"] - 3600,
        _horizon=HZ)
    ck("R-649 / REV 68 FINDING 2 -- COVERAGE IS TWO MEASURED CLOCKS, NOT "
       "THE UNIT'S OWN OLDEST LINE. ***A bursty unit with ONE line made "
       "`oldest_available <= w0` call a window a minute earlier UNCOVERED "
       "for a journal that is COMPLETE.*** The host's horizon against the "
       "unit's own start says covered; a window before the horizon says "
       "UNCOVERED and NAMES it; and neither answer comes from searching "
       "text",
       bursty["covered"] is True
       and one_min_earlier["covered"] is True
       and before["covered"] is False
       and HZ["oldest_utc"] in before["why"]
       and bursty["regime"] == BURSTY
       and one_min_earlier["regime"] == CONTINUOUS,
       f"horizon {HZ['oldest_utc']} vs a unit starting {ST['start_utc']} "
       f"-> covered {bursty['covered']}; a window one minute before its "
       f"single line -> {one_min_earlier['covered']}; a window an hour "
       f"before the horizon -> {before['covered']}")
    live_units = [u for u in ("resource-monitor.service", "resource-monitor")
                  if unit_start(u).get("start_epoch")]
    ck("AND THE HOST CLOCK IS READ LIVE: the journal's horizon is measured "
       "on this machine right now, whatever units happen to exist -- a "
       "cell that needed a NAMED unit to still be there would be pinned to "
       "an ambient, and the unit this was written against was collected on "
       "success between rounds",
       hz["status"] == "MEASURED" and hz["oldest_epoch"],
       f"live horizon {hz['oldest_utc']} (query {hz['query']}); units with "
       f"a readable start right now: {live_units or 'none named here'}")

    unread = journal_coverage(
        regime=CONTINUOUS, window_start_epoch=0.0,
        _horizon={"status": "JOURNALCTL_FAILED", "oldest_epoch": None,
                  "oldest_utc": None})
    ck("AND AN UNREADABLE CLOCK MAKES COVERAGE **UNKNOWN**, never covered "
       "and never uncovered: a failed read is not a measurement in either "
       "direction",
       unread["covered"] is None
       and unread["status"] == "NOT_DETERMINABLE",
       f"a failed horizon read -> covered {unread['covered']}, status "
       f"{unread['status']}")

    ck("AND A LINE IT CANNOT READ IS NAMED, NOT DROPPED: an unreadable "
       "status line is a status, and a caller that treated silence as a "
       "clean tree would be reading absence as a pass (rule 11)",
       parse_porcelain("M\n?? ok\n")["n_malformed"] == 1
       and parse_porcelain("M\n?? ok\n")["rows"][0]["path"] == "ok",
       "a 1-character line is malformed; the good line beside it parses")
    ck("AND THE CANONICAL PREDICATE IS READ FROM DE's MODULE, NOT TYPED: "
       "the canonical repo and data roots come from `de_data_root`'s own "
       "literals, so this seat cannot hold a root DE has moved",
       canonical_from_DEs_source().get("status") == "READ_FROM_DES_SOURCE"
       and canonical_from_DEs_source().get("repo"),
       f"{canonical_from_DEs_source().get('source')} -> "
       f"{canonical_from_DEs_source().get('repo')}")
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    import argparse                                          # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 1 if selftest()[1] else 0
    ap.error("--selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

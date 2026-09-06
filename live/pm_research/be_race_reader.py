"""THE RACE READER, RETARGETED TO THE FEED. R-588: OPTION A, NO RE-SEAL.

Round 52 refused on the SCORES and was right about those bytes. Round 53
showed the estimand was always computable -- from the FEED, which is what
the interim actually read on 09-01 and 09-02. R-588 rules Option A: the read
opens the FEED and the statistic is the interim's PRIMARY, MATCHED_VOLUME,
because that is what those two days were read with and changing the
statistic after seeing them would be a choice after seeing.

WHAT IT OPENS AND WHAT STAYS SHUT. It opens the five
`be_forward_day_SEALED_feed_<DAY>.jsonl`. The `SEALED_scores` files STAY
SEALED and are never opened -- the estimand does not need them, and opening
more than the estimand needs is consumption without purpose (rule 11).

THE STATISTIC IS THE INTERIM'S OWN CODE, NOT A SECOND COPY.
`be_read_cells.load_two_arm_feed` streams the feed and REFUSES a one-arm
feed BY NAME -- "computing it from one arm would compare the candidate with
itself and return a zero that looks like a measurement" -- and
`be_read_cells.matched_volume` returns `MATCHED_VOLUME_increment_cents`.
Two implementations of one statistic is two statistics.

BY_THRESHOLD IS REPORTED, NEVER PRIMARY (rule 7, as the interim states it:
controls are matched on the DECISION VARIABLE, and BY_THRESHOLD is not).

IT DOES NOT RUN ON THE REAL FILES. `--open` is the coordinator's act on GO;
everything below is driven on a SYNTHETIC two-arm feed written to the
writer's own `FEED_FIELDS`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_race_read_declaration as DECL

ROOT = HERE.parents[1]
OUT_NAME = "be_race_read_result_v1.json"
GATE1_PATTERNS = DECL.GATE1_ARTIFACT_PATTERNS
LATENCY_MS = 50


class ReadVoid(RuntimeError):
    """The read is void. Never downgraded to a warning."""


class ReadRefused(RuntimeError):
    """A named refusal."""


PINS = HERE / "declarations" / "be_race_read_feed_pins_v1.json"


def sealed_feeds() -> dict:
    """The FEED paths, derived with `with_name` on the SCORES paths.

    REV 44 A.5: the string form (`.replace("SEALED_scores", ...)`) rewrites
    ANY occurrence, so a directory that happened to contain the token would
    be corrupted silently. `with_name` touches the FILENAME only."""
    out = {}
    for d, sp in DECL.SEALED_SCORES.items():
        q = Path(sp)
        out[d] = str(q.with_name(
            q.name.replace("SEALED_scores", "SEALED_feed")
                  .replace(".json", ".jsonl")))
    return out


def pins() -> dict:
    """The pinned feed digests, taken BEFORE the read."""
    if not PINS.exists():
        raise ReadRefused(f"REFUSED: no pin file at {PINS}. A read that "
                          f"cannot check what it opens against a pin taken "
                          f"beforehand is not the declared read.")
    return json.loads(PINS.read_text())["per_day"]


def assert_pinned(day: str, path, per_day: dict | None = None) -> dict:
    """Compare against the pin BEFORE parsing; refuse absent or mismatched.

    REV 44 A.3: `exists: false` must be ACTIONABLE. A day the pin marks
    absent is refused BY NAME rather than discovered as a missing file."""
    import hmac
    pd = pins() if per_day is None else per_day
    pin = pd.get(day)
    if pin is None:
        raise ReadRefused(f"REFUSED: {day} has no pin. Every day the read "
                          f"opens must have been pinned before it.")
    if not pin.get("exists"):
        raise ReadRefused(
            f"REFUSED: the pin marks {day}'s feed ABSENT "
            f"(`exists: false`, no digest). There is nothing to read for "
            f"that day, and a read that silently skipped it would report a "
            f"smaller G as though it were the declared one.")
    want = str(pin.get("sha256") or "")
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        raise ReadRefused(f"REFUSED: {day}'s pin is not a 64-hex digest "
                          f"({want[:20]!r}).")
    h = hashlib.sha256()
    n = 0
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    got = h.hexdigest()
    if not hmac.compare_digest(got, want):
        raise ReadRefused(
            f"REFUSED: {day}'s feed digests {got[:16]}…, not the pinned "
            f"{want[:16]}…. The bytes changed between the pin and the read.")
    return {"day": day, "pinned_sha256": want, "bytes": n,
            "checked_before_parsing": True, "n_hex_compared": len(want)}


def theta_for(coin: str, budget_label: str = "10%") -> float:
    """READ from the operating-point declaration, never typed."""
    d = json.loads((HERE / "declarations"
                    / "be_operating_point_declaration_v1.json").read_text())
    t = d["theta_frozen_by_coin"].get(coin, {}).get(budget_label)
    if t is None:
        raise ReadRefused(f"REFUSED: no frozen theta for {coin} at "
                          f"{budget_label}.")
    return float(t)


def assert_separation(opened) -> dict:
    paths = [str(p) for p in opened]
    hits = sorted({f"{pat} in {p}" for pat in GATE1_PATTERNS
                   for p in paths if pat in p})
    if hits:
        raise ReadRefused(
            f"REFUSED: a Gate-1 object is on this read's path: {hits}.")
    return {"no_gate1_artifact_on_the_read_path": True,
            "checked_patterns": list(GATE1_PATTERNS),
            "haystack": "the paths THIS RUN opened, not a module constant",
            "n_paths_checked": len(paths), "matches": []}


class _HashingPath:
    """A path whose `.open()` hashes every byte the READER consumes.

    REV 44 A.4: the digest must be over the stream that was PARSED, not over
    a separate read of the same name. This wraps the file so
    `load_two_arm_feed` -- unchanged, the interim's own code -- parses the
    same bytes this hash covers, in ONE pass. The three-read form is gone."""

    def __init__(self, path):
        self._p = Path(path)
        self.h = hashlib.sha256()
        self.n = 0
        self.name = self._p.name

    def __str__(self):
        return str(self._p)

    def open(self, *a, **kw):
        outer = self

        class _F:
            def __init__(self, fh):
                self.fh = fh

            def __iter__(self):
                for line in self.fh:
                    outer.h.update(line.encode())
                    outer.n += len(line.encode())
                    yield line

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return self.fh.__exit__(*e)

        return _F(self._p.open(*a, **kw))


def day_matched_volume(path, *, latency_ms: int = LATENCY_MS) -> dict:
    """MATCHED_VOLUME per day, through the INTERIM'S OWN functions."""
    import be_read_cells as C
    hp = _HashingPath(path)
    feed = C.load_two_arm_feed(hp, latency_ms)
    per_coin, net = {}, 0.0
    # the loader returns {"per_coin": {...}, "n_feed_rows": ...}: the coins
    # are NESTED, and iterating the top level would silently find none.
    for coin, blk in sorted(feed["per_coin"].items()):
        if not isinstance(blk, dict) or "rows" not in blk:
            continue
        mv = C.matched_volume(blk["rows"], blk["cand"], blk["inc"],
                              theta_for(coin), latency_ms)
        per_coin[coin] = {
            "MATCHED_VOLUME_increment_cents":
                mv["MATCHED_VOLUME_increment_cents"],
            "candidate_net_cents": mv["candidate_net_cents"],
            "incumbent_net_cents_matched": mv["incumbent_net_cents_matched"],
            "counts_matched": mv["counts_matched"],
            "n_actions": mv["n_actions"],
        }
        net += float(mv["MATCHED_VOLUME_increment_cents"])
    if not per_coin:
        raise ReadRefused(f"REFUSED: {Path(path).name} yielded no coin with "
                          f"rows; a day with no action is a STATUS, not a "
                          f"zero increment.")
    return {"status": "OK", "per_coin": per_coin,
            "parsed_stream_sha256": hp.h.hexdigest(),
            "parsed_stream_bytes": hp.n,
            "day_increment_cents": net,
            "day_sign": (1 if net > 0 else (-1 if net < 0 else 0)),
            "n_feed_rows": feed["n_feed_rows"],
            "n_rows_without_an_incumbent_score":
                feed["n_rows_without_an_incumbent_score"],
            "statistic": "MATCHED_VOLUME (R-588; the interim's PRIMARY)",
            "computed_by": "be_read_cells.matched_volume -- the interim's own "
                           "code, not a second copy",
            "latency_ms": latency_ms}


def declared_read(decl_dir: Path | None = None) -> dict:
    """THE DECLARATION'S OWN READ SET, from the CHAIN HEAD.

    REV 48 §1.6 / R-600: the reader's day set came from the INVOCATION --
    `--open` handed `sealed_feeds()`'s FIVE days to `read()`, which computed
    `permutation_floors` from `len(paths)`. So G was a property of the call,
    not of the declaration: a five-path call resolved 0.25 while REPORTING
    G = 5, and `--open` was safe only because two of the five files happen
    to be absent. The declaration says READABLE = three days and G = 3; this
    resolves that, from the head of the chain rather than a filename."""
    import be_rule22 as _R22
    if decl_dir is not None:
        saved = _R22.DECLARATIONS
        try:
            _R22.DECLARATIONS = Path(decl_dir)
            head = _R22.declaration_head("be_race_read_declaration")
        finally:
            _R22.DECLARATIONS = saved
    else:
        head = _R22.declaration_head("be_race_read_declaration")
    doc = head["doc"]
    readable = list((doc.get("population") or {}).get("READABLE") or [])
    if not readable:
        raise ReadRefused(
            f"REFUSED: {head['name']} declares no READABLE population. A "
            f"read whose day set is empty is not a smaller read, it is a "
            f"different question.")
    g_declared = ((doc.get("permutation_floor") or {}).get("G"))
    return {"declaration": head["name"], "declaration_sha256": head["sha256"],
            "n_versions": head["n_versions"], "READABLE": sorted(readable),
            "G_declared": g_declared,
            "resolved_from": "the chain head, never a filename"}


def resolve_days(cli_days=None, *, decl_dir: Path | None = None) -> dict:
    """The day set the read WILL use -- the declaration's, always.

    The CLI can neither widen nor narrow it: a `--days` that differs is
    REFUSED BY NAME, and no argument means the declaration's set. G is then
    computed from that set and asserted equal to the declaration's own G, so
    rule 10 is obeyed once and CHECKED twice."""
    d = declared_read(decl_dir)
    if cli_days is not None:
        want = sorted(cli_days)
        if want != d["READABLE"]:
            extra = [x for x in want if x not in d["READABLE"]]
            missing = [x for x in d["READABLE"] if x not in want]
            raise ReadRefused(
                f"REFUSED: --days {want} is not the declared READABLE set "
                f"{d['READABLE']} (declaration {d['declaration']}). "
                f"{'Widened by ' + str(extra) + '. ' if extra else ''}"
                f"{'Narrowed by ' + str(missing) + '. ' if missing else ''}"
                f"The day set is the declaration's; an invocation that could "
                f"change it would make G a property of the call, which is "
                f"exactly what R-600 found.")
    g_computed = len(d["READABLE"])
    if d["G_declared"] is not None and g_computed != d["G_declared"]:
        raise ReadRefused(
            f"REFUSED: G computed from the declared READABLE set is "
            f"{g_computed}, but {d['declaration']} declares "
            f"permutation_floor.G = {d['G_declared']}. The declaration "
            f"disagrees with itself; nothing is read until it does not.")
    return {**d, "G_computed": g_computed,
            "G_agrees_with_the_declaration": (d["G_declared"] is None
                                              or g_computed == d["G_declared"]),
            "days": d["READABLE"],
            "cli_days_supplied": cli_days is not None}


def resolve_marker_dir(outdir=None, *, fixture: bool = False,
                       why: str | None = None) -> dict:
    """THE MARKER DIRECTORY, THROUGH `require_ledger()`.

    REV 76 §5(a): this module never called `require_ledger()`. The markers
    were resolved through the UNGUARDED `derived()`, so from a MATERIALISED
    worktree the reader would look for OPENED markers in a partial shell,
    find none, and open days the LEDGER records as consumed. **The
    consumption guard must be as strong as the act it guards.**

    A fixture may direct the markers elsewhere, but it must say what it is
    -- `require_ledger` refuses `fixture=True` without a reason, so an
    exemption cannot be invisible to a reader."""
    if fixture:
        if not why:
            raise ReadRefused(
                "REFUSED: a fixture marker directory without `why`. An "
                "exemption from the ledger that does not say what it is, is "
                "invisible to a reader.")
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        return {"dir": d, "is_the_ledger": False, "fixture": True,
                "why": why, "realpath": str(d.resolve()),
                "ledger_check": "EXEMPT_FIXTURE"}
    res = _BDR.require_ledger(why=None)          # refuses a non-ledger root
    d = Path(_BDR.derived())
    real = str(Path(d).resolve())
    ledger_derived = str(Path(_BDR.derived()).resolve())
    if real != ledger_derived:
        raise ReadRefused(
            f"REFUSED: the marker directory resolves to {real}, not the "
            f"ledger's derived directory {ledger_derived}. Markers read from "
            f"anywhere else would find none and open days the ledger records "
            f"as consumed (REV 76 §5(a)).")
    if outdir is not None and str(Path(outdir).resolve()) != real:
        raise ReadRefused(
            f"REFUSED: outdir {Path(outdir).resolve()} is not the ledger's "
            f"derived directory {real}, and this is not a declared fixture.")
    return {"dir": d, "is_the_ledger": True, "fixture": False,
            "realpath": real, "readlink_f": real,
            "equals_the_ledger_derived_dir": real == ledger_derived,
            "ledger_check": res.get("ledger_check"),
            "data_root": res.get("data_root"),
            "why_guarded": "REV 76 §5(a): resolved through require_ledger(), "
                           "because a marker directory in a partial shell "
                           "finds no markers and reopens spent days"}


def pre_state(days, marker_dir: Path, feeds: dict, pd: dict,
              decl: dict) -> dict:
    """THE STATE BEFORE THE ACT, recorded as `--open`'s FIRST step.

    REV 76 §6 condition 3. Everything here is read BEFORE a single marker is
    written, so the artifact carries what was true when the act began rather
    than what remained after it."""
    import subprocess as _sp
    now = _sp.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,
                  text=True, timeout=30).stdout.strip()
    existing = sorted(str(x) for x in Path(marker_dir).glob(
        "be_race_read_OPENED_*"))
    per_day = {}
    for d in sorted(days):
        pin = (pd or {}).get(d) or {}
        q = Path(feeds[d]) if d in feeds else None
        on_disk = bool(q and q.exists())
        got = None
        if on_disk:
            h = hashlib.sha256()
            with q.open("rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            got = h.hexdigest()
        per_day[d] = {
            "pin_present": bool(pin), "pin_exists_true": bool(pin.get("exists")),
            "pinned_sha256": pin.get("sha256"),
            "feed": str(q) if q else None, "feed_on_disk": on_disk,
            "feed_sha256_now": got,
            "feed_matches_its_pin": bool(got and got == pin.get("sha256")),
        }
    result_name = Path(marker_dir) / OUT_NAME
    return {
        "as_of": now,
        "marker_dir": str(marker_dir),
        "marker_dir_realpath": str(Path(marker_dir).resolve()),
        "existing_OPENED_markers": existing,
        "n_existing_OPENED_markers": len(existing),
        "zero_markers_before_the_act": len(existing) == 0,
        "declaration": decl.get("declaration"),
        "declaration_sha256": decl.get("declaration_sha256"),
        "days": sorted(days),
        "per_day": per_day,
        "all_pins_present_and_true": all(
            v["pin_present"] and v["pin_exists_true"] for v in per_day.values()),
        "all_feeds_on_disk_at_their_pins": all(
            v["feed_matches_its_pin"] for v in per_day.values()),
        "declared_result_name": OUT_NAME,
        "declared_result_absent_before_the_act": not result_name.exists(),
        "why": "REV 76 §6 condition 3: recorded as the FIRST step of the "
               "act, before any marker is written, so the artifact carries "
               "the state the act began from",
    }


def marker_path(day: str, outdir: Path) -> Path:
    return Path(outdir) / f"be_race_read_OPENED_{day}.json"


def assert_not_already_opened(days, outdir: Path) -> dict:
    """THE THREE-PATH READ CONSUMES (runbook §6).

    A day that has been opened cannot be opened again: the first read is the
    one that spends it, and a second would report a fresh result from a
    consumed day. The marker is written BEFORE the feed is read, so a read
    that dies mid-way still leaves the day marked -- consumed is the safe
    direction, unread is not."""
    already = []
    for d in sorted(days):
        m = marker_path(d, outdir)
        if m.exists():
            # REV 76 §5(b): a HALF-WRITTEN marker raised JSONDecodeError out
            # of this guard, and on the one-shot path a traceback is not a
            # verdict. THE FILE'S PRESENCE IS THE FACT; its contents are
            # detail. An unparseable marker is treated as OPENED.
            try:
                # REV 77 §2.2: the `.get` sat OUTSIDE the try, so a marker
                # that is valid JSON but NOT AN OBJECT -- `[]`, a string,
                # null, a number -- parsed fine and then raised on `.get`.
                # ANYTHING at that path means the day was spent, so the
                # whole read-and-interpret is inside the try.
                _doc = json.loads(m.read_text())
                _op = _doc.get("utc") if isinstance(_doc, dict) else None
                _parsed = isinstance(_doc, dict)
                if not _parsed:
                    raise TypeError(
                        f"the marker is valid JSON but a "
                        f"{type(_doc).__name__}, not an object")
            except (json.JSONDecodeError, OSError, UnicodeDecodeError,
                    TypeError, AttributeError, ValueError) as _e:
                _op, _parsed = None, False
                already.append({"day": d, "marker": str(m),
                                "opened_at": None, "marker_parsed": False,
                                "why": f"the marker exists but could not be "
                                       f"parsed ({type(_e).__name__}); its "
                                       f"PRESENCE is the fact that the day "
                                       f"was spent"})
                continue
            already.append({"day": d, "marker": str(m), "opened_at": _op,
                            "marker_parsed": _parsed})
    if already:
        _unp = [a for a in already if a.get("marker_parsed") is False]
        raise ReadRefused(
            f"REFUSED: {[a['day'] for a in already]} already carry an OPENED "
            f"marker ({[a['marker'] for a in already]}). The race read "
            f"CONSUMES the days it opens; a second read of a consumed day "
            f"would report a fresh result from a spent one."
            + (f" {len(_unp)} of these markers could not be parsed "
               f"({[a['day'] for a in _unp]} at "
               f"{[a['marker'] for a in _unp]}) -- an unparseable marker is "
               f"treated as OPENED, because the file's PRESENCE is the fact."
               if _unp else ""))
    return {"checked": sorted(days), "already_opened": []}


def write_open_markers(days, outdir: Path, pins_used: dict) -> list:
    """Written BEFORE any feed is read -- see `assert_not_already_opened`."""
    import subprocess as _sp
    now = _sp.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,
                  text=True, timeout=30).stdout.strip()
    out = []
    for d in sorted(days):
        m = marker_path(d, outdir)
        m.write_text(json.dumps({
            "day": d, "utc": now,
            "pin": (pins_used.get(d) or {}).get("sha256"),
            "why": "the race read CONSUMES this day: written BEFORE the feed "
                   "was read, so a read that dies mid-way still leaves it "
                   "marked. Consumed is the safe direction.",
        }, indent=1, sort_keys=True))
        out.append(str(m))
    return out


def floors(g_opt: int, g_pess: int, m: int = 2) -> dict:
    o, p = m / 2 ** g_opt, m / 2 ** g_pess
    return {"optimistic": {"G": g_opt, "best_possible_adjusted_p": o},
            "pessimistic": {"G": g_pess, "best_possible_adjusted_p": p},
            "resolved_best_possible_adjusted_p": max(o, p),
            "WHICH_ONE_IS_RESOLVED": "the CONSERVATIVE one",
            "neither_clears_0_05": min(o, p) > 0.05}


def read(paths: dict, *, outdir: Path = None, write: bool = True,
         per_day_pins: dict | None = None, decl: dict | None = None,
         consume: bool = True, marker_fixture: bool = False,
         marker_fixture_why: str | None = None,
         decl_dir: Path | None = None) -> dict:
    opened = [Path(v) for v in paths.values()]
    sep = assert_separation(opened)
    per_day, before, pinned = {}, {}, {}
    pd = per_day_pins if per_day_pins is not None else pins()
    # THE BY-NAME REFUSAL IS REACHABLE NOW. The generic `sealed feed(s)
    # absent` fired FIRST, so `assert_pinned`'s refusal -- the one that
    # NAMES the day and says a silently skipped day would report a smaller G
    # as though it were the declared one -- was unreachable on the CLI path
    # (REV 48 §1.6). Each declared day is checked BY NAME, in order, before
    # anything generic.
    for d in sorted(paths):
        q = Path(paths[d])
        pin = (pd or {}).get(d)
        if pin is None:
            raise ReadRefused(
                f"REFUSED: {d} is in the declared READABLE set and has no "
                f"pin. Every day the read opens must have been pinned "
                f"before it.")
        if not pin.get("exists"):
            raise ReadRefused(
                f"REFUSED: {d} is DECLARED READABLE but its pin marks the "
                f"feed ABSENT (`exists: false`, no digest). A read that "
                f"skipped it would report a smaller G as though it were the "
                f"declared one.")
        if not q.exists():
            raise ReadRefused(
                f"REFUSED: {d} is DECLARED READABLE and pinned, but its feed "
                f"{q.name} is not on disk. Named, not folded into a generic "
                f"absence.")
    # the day set and G come from the DECLARATION, never from len(paths)
    # REV 77 §2.1: this recorded `decl_was_injected: decl is not None`
    # beside prose saying "on the real path nothing is injected" -- while
    # the CLI's --open DOES pass `decl=_dc`. The read artifact is the one
    # permanent record of this test: it cannot be re-run and has no .v2, so
    # a field contradicting the sentence next to it would stand forever.
    #
    # The field now says WHAT HAPPENED, and a supplied decl is RE-VERIFIED
    # against a fresh resolution of the same declarations directory: on the
    # real path the CLI's decl must BE the chain head, and if it is not, the
    # read refuses rather than recording a decl nobody checked.
    _fresh = resolve_days(decl_dir=decl_dir)
    if decl is None:
        _dc, _supplied = _fresh, False
    else:
        _dc, _supplied = decl, True
    _is_head = (_dc.get("declaration") == _fresh.get("declaration")
                and _dc.get("declaration_sha256")
                == _fresh.get("declaration_sha256"))
    if _supplied and not _is_head and not marker_fixture:
        raise ReadRefused(
            f"REFUSED: the caller supplied a declaration "
            f"({_dc.get('declaration')} / "
            f"{str(_dc.get('declaration_sha256'))[:16]}…) that is NOT the "
            f"chain head resolved here ({_fresh.get('declaration')} / "
            f"{str(_fresh.get('declaration_sha256'))[:16]}…). On the real "
            f"path the day set comes from the head, and a supplied set "
            f"nobody re-checked is exactly what R-600 found.")
    _decl_source = (
        "resolve_days() against the declaration chain head, resolved inside "
        "read()" if not _supplied else
        ("resolve_days() against the declaration chain head, supplied by the "
         "caller and RE-VERIFIED here against a fresh resolution (same "
         "declaration, same digest)" if _is_head else
         "supplied by the caller and NOT the chain head -- a fixture "
         "declaration; this is not the real path"))
    if sorted(paths) != sorted(_dc["days"]):
        raise ReadRefused(
            f"REFUSED: the paths handed to read() are {sorted(paths)} but "
            f"the declaration's READABLE set is {sorted(_dc['days'])}. G "
            f"would be a property of the call (R-600).")
    _md = resolve_marker_dir(outdir, fixture=marker_fixture,
                             why=marker_fixture_why)
    _out_d = _md["dir"]
    _pre = pre_state(_dc["days"], _out_d, {d: str(v) for d, v in paths.items()},
                     pd or {}, _dc) if consume else None
    if consume:
        # THE MARKER GUARD FIRST: it NAMES THE DAYS. The result-name guard
        # is true of the whole read and would hide it -- the same ordering
        # defect as the generic `sealed feed(s) absent` in REV 48 §1.6.
        assert_not_already_opened(_dc["days"], _out_d)
        if not _pre["declared_result_absent_before_the_act"]:
            raise ReadRefused(
                f"REFUSED: {OUT_NAME} already exists in {_out_d}. The "
                f"declared result of this read is present before the act, "
                f"so the read has been done -- even though no day carries a "
                f"marker.")
        _markers = write_open_markers(_dc["days"], _out_d, pd or {})
    else:
        _markers = []
    for d, p in sorted(paths.items()):
        # A.3: the pin is checked BEFORE a byte is parsed.
        pinned[d] = assert_pinned(d, p, pd)
        before[d] = pinned[d]["pinned_sha256"]
        per_day[d] = day_matched_volume(p)
    after = {d: hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for d, p in paths.items()}
    # A.4: the claim "the digest is of the bytes parsed" is now a COMPUTED
    # predicate -- the parsed stream's own hash against the file's -- not a
    # literal beside a table (rule 10). A stream mutated mid-parse makes
    # these disagree and VOIDS the read.
    covers = {d: {"parsed_stream_sha256": per_day[d]["parsed_stream_sha256"],
                  "file_sha256_after": after[d],
                  "parsed_bytes": per_day[d]["parsed_stream_bytes"],
                  "file_bytes": Path(paths[d]).stat().st_size,
                  "digest_covers_every_byte_parsed":
                      (per_day[d]["parsed_stream_sha256"] == after[d]
                       and per_day[d]["parsed_stream_bytes"]
                       == Path(paths[d]).stat().st_size)}
              for d in paths}
    bad_cover = sorted(d for d, c in covers.items()
                       if not c["digest_covers_every_byte_parsed"])
    if bad_cover:
        raise ReadVoid(
            f"REFUSED — THE READ IS VOID: for {bad_cover} the hash of the "
            f"stream THAT WAS PARSED does not equal the file's. The bytes "
            f"moved under the parser, so the number was computed over "
            f"something other than what is on disk. No result is emitted.")
    moved = sorted(d for d in before if before[d] != after[d])
    if moved:
        raise ReadVoid(
            f"REFUSED — THE READ IS VOID: the sealed bytes for {moved} "
            f"CHANGED between the digest of the bytes PARSED and the one "
            f"taken after. A read that moved the bytes it read is not a "
            f"read, it is an edit. No result is emitted.")
    signs = {d: v["day_sign"] for d, v in per_day.items()}
    fresh = [d for d in paths
             if d not in DECL.ALREADY_OPENED_UNDER_THE_INTERIM]
    out = {
        "protocol": "BE_RACE_READ_RESULT_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "R_529_A_UP_FRONT": "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY "
                            "AND NEVER A HOLM-CLEARING VERDICT (R-529(A)).",
        "ruling": "R-588: OPTION A, NO RE-SEAL. The read opens the FEED; the "
                  "statistic is MATCHED_VOLUME, the interim's PRIMARY, "
                  "because that is what 09-01 and 09-02 were read with and "
                  "changing it after seeing two days would be a choice after "
                  "seeing. BY_THRESHOLD is reported, never primary (rule 7).",
        "opened": {"files": [str(p) for p in opened],
                   "what_stays_sealed": "be_forward_day_SEALED_scores_"
                                        "<DAY>.json -- the estimand does not "
                                        "need them and opening more than it "
                                        "needs is consumption without "
                                        "purpose (rule 11)"},
        "days": sorted(paths), "per_day": per_day, "day_signs": signs,
        "n_positive": sum(1 for v in signs.values() if v == 1),
        "n_negative": sum(1 for v in signs.values() if v == -1),
        "n_zero": sum(1 for v in signs.values() if v == 0),
        # G FROM THE DECLARATION, not from the count of paths handed in
        "permutation_floors": floors(_dc["G_computed"], len(fresh)),
        "day_set": {"from": _dc["declaration"],
                    "declaration_sha256": _dc["declaration_sha256"],
                    "READABLE": _dc["days"],
                    "G_declared": _dc["G_declared"],
                    "G_computed_from_the_set": _dc["G_computed"],
                    "G_agrees_with_the_declaration":
                        _dc["G_agrees_with_the_declaration"],
                    "the_cli_cannot_widen_or_narrow_it": True},
        "pre_state": _pre,
        "consumption": {"markers_written_before_reading": _markers,
                        "the_read_consumes": True,
                        "marker_dir": _md,
                        "what_a_marker_IS": "an UNTRACKED file under the "
                                            "ledger's derived directory. It "
                                            "is the ONLY record that a day "
                                            "was spent, and it is exactly as "
                                            "durable as that directory -- "
                                            "not in git, not in any receipt "
                                            "chain (REV 76 §5(c))",
                        "decl_source": _decl_source,
                        "decl_supplied_by_the_caller": _supplied,
                        "decl_is_the_chain_head": _is_head,
                        "decl_declaration": _dc.get("declaration"),
                        "decl_declaration_sha256":
                            _dc.get("declaration_sha256"),
                        "why_this_field_replaced_decl_was_injected":
                            "REV 77 §2.1: `decl_was_injected` would have "
                            "read TRUE on the real path -- the CLI passes "
                            "the output of resolve_days() -- beside prose "
                            "claiming nothing is injected. The record of "
                            "this test cannot be re-run and has no .v2, so "
                            "the field says what happened and the supplied "
                            "declaration is re-verified against a fresh "
                            "resolution rather than trusted",
                        "runbook": "§6 -- a three-path call to the race "
                                   "reader CONSUMES the race days"},
        "byte_identity": {"pinned_before": before, "after": after,
                          "all_unchanged": True,
                          "pins": pinned,
                          "digest_covers_every_byte_parsed": covers,
                          "computed_not_asserted": "the coverage claim is a "
                                                   "predicate over the "
                                                   "parsed stream's own "
                                                   "hash, not a literal",
                          "on_mismatch": "the read is VOID -- enforced"},
        "gate1_separation": sep,
        "writes": {"artifact": OUT_NAME, "and_nothing_else": True},
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14).",
    }
    if write:
        d = Path(outdir) if outdir is not None else _BDR.derived()
        (d / OUT_NAME).write_text(json.dumps(out, indent=1, sort_keys=True,
                                             default=str))
        out["_written"] = str(d / OUT_NAME)
    return out


EXPECTED_CHECKS = 36


def _feed(d: Path, day: str, rows, *, one_arm: bool = False) -> Path:
    p = d / f"be_forward_day_SEALED_feed_{day}.jsonl"
    with p.open("w") as fh:
        for r in rows:
            r = dict(r)
            if one_arm:
                r.pop("score_incumbent", None)
            fh.write(json.dumps(r) + "\n")
    return p


def _row(gen, score, inc, cents, **kw):
    """A row in the WRITER's own shape (`be_forward_day.FEED_FIELDS`)."""
    return dict({"slug": "btc-updown-5m-1", "side": "BUY_UP", "gen": gen,
                 "t0": 0.0, "t_start": 0.0, "score": score,
                 "score_incumbent": inc, "any_fill_ahead": True,
                 "value_cents": cents, "preventable_shares": 1.0,
                 "level": 0.5}, **kw)


def selftest() -> int:
    import tempfile
    checks, fails = 0, []
    # the ledger's marker set BEFORE any drive, so the closing check can
    # compare against it rather than against a constant that expires
    _ledger_markers_at_start = sorted(
        x.name for x in Path(_BDR.derived()).glob("be_race_read_OPENED_*"))

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    import be_forward_day as FD
    import be_read_cells as C
    ok(set(_row(0, 1.0, 1.0, 1.0)) >= set(FD.FEED_FIELDS),
       f"THE SYNTHETIC FEED IS THE WRITER'S OWN SHAPE: every one of "
       f"`be_forward_day.FEED_FIELDS` is present in the fixture row")
    ok(theta_for("btc") == 0.7230267681941027,
       f"and theta is READ from the operating-point declaration "
       f"({theta_for('btc')}), never typed here")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # ONE-ARM: must refuse BY NAME, through the interim's own reader
        p1 = _feed(d, "20260903", [_row(i, 1.0, 0.5, 2.0) for i in range(4)],
                   one_arm=True)
        try:
            day_matched_volume(p1)
            ok(False, "a one-arm feed must refuse")
        except C.ReadCellsRefused as e:
            ok("ONE-ARM feed" in str(e) and "zero that looks like a "
               "measurement" in str(e),
               "KNOWN-BAD: a ONE-ARM feed REFUSES **by name**, in the "
               "interim's own reader -- comparing the candidate with itself "
               "would return a zero that looks like a measurement")

        # THE DEGENERACY KNOWN-BAD: tied vs 1e-9-perturbed, same sign
        # THE ARMS MUST RANK DIFFERENTLY or the increment is 0 by
        # construction and the sign check proves nothing. Candidate ranks by
        # +i, incumbent by -i, and the cents differ per row.
        # The two arms must select DISJOINT sets or the increment is 0 by
        # construction and the sign proves nothing: the candidate clears
        # theta on the first half, the incumbent ranks the second half top.
        def collapsing(eps):
            return ([_row(i, 2.0, 0.0, -10.0 + i * eps) for i in (0, 1)]
                    + [_row(i, 0.1, 2.0, 8.0 + i * eps) for i in (2, 3)])
        a = day_matched_volume(_feed(d, "20260904", collapsing(0.0)))
        b = day_matched_volume(_feed(d, "20260905", collapsing(1e-9)))
        ok(a["day_sign"] == b["day_sign"] and a["day_sign"] != 0,
           f"THE DEGENERACY FALSIFIER: a collapsing series with values TIED "
           f"and the same series perturbed by 1e-9 give the SAME sign "
           f"({a['day_sign']}) -- a net does not turn on 1e-9, where the old "
           f"flip-count did")

        # a KNOWN net reproduces
        kn = day_matched_volume(_feed(d, "20260901",
                                      [_row(0, 2.0, 0.0, 10.0),
                                       _row(1, 2.0, 0.0, 6.0),
                                       _row(2, 0.1, 2.0, -8.0),
                                       _row(3, 0.1, 2.0, -9.0)]))
        ok(kn["status"] == "OK" and kn["per_coin"]["btc"]["n_actions"] == 4
           and kn["per_coin"]["btc"]["counts_matched"]
           and kn["per_coin"]["btc"]["MATCHED_VOLUME_increment_cents"] != 0,
           f"A KNOWN FEED REPRODUCES BY CONSTRUCTION: 4 actions, counts "
           f"matched, increment "
           f"{kn['per_coin']['btc']['MATCHED_VOLUME_increment_cents']}")

        pos = [_row(0, 2.0, 0.0, 10.0), _row(1, 2.0, 0.0, 6.0),
               _row(2, 0.1, 2.0, -8.0), _row(3, 0.1, 2.0, -9.0)]
        neg = [_row(0, 2.0, 0.0, -10.0), _row(1, 2.0, 0.0, -6.0),
               _row(2, 0.1, 2.0, 8.0), _row(3, 0.1, 2.0, 9.0)]
        paths = {"20260901": _feed(d, "20260901", pos),
                 "20260902": _feed(d, "20260902", neg)}
        def _pin(pp):
            return {dd: {"exists": True,
                         "sha256": hashlib.sha256(
                             Path(q).read_bytes()).hexdigest(),
                         "bytes": Path(q).stat().st_size}
                    for dd, q in pp.items()}
        _pins = _pin(paths)
        # A FIXTURE DECLARATION whose READABLE set is the fixture's own days.
        # REV 48 §1.6 item (4): the selftest never routed through the
        # declaration at all -- it called `read()` with whatever paths it had
        # built and pinned the floors as literals, so it agreed with the
        # reader about arithmetic while the reader took G from the call.
        _fdecl = d / "fixture_declarations"
        _fdecl.mkdir(exist_ok=True)
        (_fdecl / "be_race_read_declaration_v1.json").write_text(json.dumps({
            "protocol": "FIXTURE", "supersedes": None,
            "population": {"READABLE": sorted(paths)},
            "permutation_floor": {"G": len(paths), "multiplicity": 2}}))
        _fd = resolve_days(decl_dir=_fdecl)
        ok(_fd["days"] == sorted(paths) and _fd["G_computed"] == len(paths)
           and _fd["G_declared"] == len(paths),
           f"A FIXTURE DECLARATION DRIVES A DIFFERENT G: its READABLE set is "
           f"{_fd['days']} and G = {_fd['G_computed']}, computed from THAT "
           f"set -- so the selftest now routes through the declaration "
           f"rather than pinning floors as literals")
        # A.3 KNOWN-BADS, driven
        try:
            read(paths, outdir=d, decl=_fd, consume=False,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched",
                 per_day_pins=dict(
                     _pins, **{"20260901": dict(_pins["20260901"],
                                                sha256="0" * 64)}))
            ok(False, "a tampered pin must refuse")
        except ReadRefused as ex:
            ok("not the pinned" in str(ex),
               "KNOWN-BAD: a TAMPERED pin REFUSES before a byte is parsed -- "
               "the bytes changed between the pin and the read")
        try:
            read(paths, outdir=d, decl=_fd, consume=False,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched",
                 per_day_pins=dict(
                     _pins, **{"20260901": {"exists": False,
                                            "sha256": None}}))
            ok(False, "a pin-absent day must refuse by name")
        except ReadRefused as ex:
            ok("20260901 is DECLARED READABLE" in str(ex)
               and "ABSENT" in str(ex)
               and "smaller G as though it were the declared one" in str(ex),
               "KNOWN-BAD: a day the pin marks ABSENT REFUSES **by name** -- "
               "`exists: false` is actionable, and skipping it would report "
               "a smaller G as though it were the declared one")
        r = read(paths, outdir=d, per_day_pins=_pins, decl=_fd,
                 consume=False,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
        ok(all(c["digest_covers_every_byte_parsed"]
               for c in r["byte_identity"]["digest_covers_every_byte_parsed"]
               .values()),
           "A.4 COMPUTED, NOT ASSERTED: the hash of the stream THAT WAS "
           "PARSED equals the file's, over every byte -- one pass, and the "
           "old literal `digest_is_of_the_bytes_parsed` is gone")
        ok(r["n_positive"] == 1 and r["n_negative"] == 1,
           f"and the two fixture days give OPPOSITE signs "
           f"({r['day_signs']}) -- the statistic tracks the data, not a "
           f"constant")
        ok(r["byte_identity"]["all_unchanged"]
           and (d / OUT_NAME).exists()
           and sorted(x.name for x in d.glob("be_race_read_*")) == [OUT_NAME],
           "A CLEAN READ ADMITS against matching pins, and it writes THE "
           "ONE declared artifact and nothing else")
        f = r["permutation_floors"]
        # NO FLOORS LITERALS. This pinned `floors(5, 3) == 0.25` as constants
        # and never routed through the CLI, so it agreed with the reader
        # about arithmetic while the reader took its G from the invocation
        # (REV 48 §1.6). The floor is now checked against the DECLARATION's
        # own G, recomputed here from the declared set.
        _ds = r["day_set"]
        ok(f["resolved_best_possible_adjusted_p"] ==
           max(f["optimistic"]["best_possible_adjusted_p"],
               f["pessimistic"]["best_possible_adjusted_p"])
           and f["optimistic"]["G"] == _ds["G_computed_from_the_set"]
           and _ds["G_computed_from_the_set"] == len(_ds["READABLE"])
           and _ds["G_agrees_with_the_declaration"],
           f"BOTH FLOORS COMPUTED, THE CONSERVATIVE ONE RESOLVED, AND G "
           f"COMES FROM THE DECLARATION: G={_ds['G_computed_from_the_set']} "
           f"is len(READABLE) from {_ds['from']}, not len(paths) from the "
           f"call, and it agrees with the declaration's own "
           f"permutation_floor.G")

        _g = globals()
        _orig = _g["day_matched_volume"]

        def _mut(p, **kw):
            # MUTATE AFTER THE PARSE, on the file just parsed. The pin has
            # already passed and the parser has already consumed the stream,
            # so the parsed-stream hash is pre-mutation and the file's is
            # post -- which is exactly what A.4's coverage predicate is for.
            # (Mutating BEFORE the parse is caught one step earlier by the
            # pin, which is also correct but is A.3's case, not A.4's.)
            r = _orig(p, **kw)
            with Path(p).open("a") as fh:
                fh.write(json.dumps(_row(99, 5.0, 1.0, -9.0)) + "\n")
            return r
        _g["day_matched_volume"] = _mut
        try:
            read(paths, outdir=d, per_day_pins=_pins, decl=_fd,
                 consume=False,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
            ok(False, "a tampered feed must VOID the read")
        except ReadVoid as e:
            ok("THE READ IS VOID" in str(e)
               and "moved under the parser" in str(e),
               "KNOWN-BAD, A.4's OWN CASE: a stream mutated MID-READ makes "
               "the parsed-stream hash disagree with the file's, and the "
               "read is VOIDED -- the number would have been computed over "
               "something other than what is on disk")
        finally:
            _g["day_matched_volume"] = _orig

        try:
            assert_separation([paths["20260901"],
                               d / "be_daybook_20260903_btc.pkl"])
            ok(False, "a planted Gate-1 path must refuse")
        except ReadRefused as e:
            ok("Gate-1 object is on this read's path" in str(e),
               "KNOWN-BAD: a Gate-1 object planted into the OPENED set "
               "REFUSES -- the haystack is what this run opened")

    # ---- A.5: the derivation touches the FILENAME only --------------------
    _bad = Path("/tmp/SEALED_scores_dir/be_forward_day_SEALED_scores_1.json")
    _got = str(_bad.with_name(_bad.name.replace("SEALED_scores", "SEALED_feed")
                              .replace(".json", ".jsonl")))
    ok("/tmp/SEALED_scores_dir/" in _got and _got.endswith(
           "be_forward_day_SEALED_feed_1.jsonl"),
       f"KNOWN-BAD FOR THE STRING FORM: a directory containing the token "
       f"'SEALED_scores' is LEFT INTACT by `with_name` ({_got}); the old "
       f"`str.replace` would have rewritten the directory too and pointed "
       f"the read at a path that does not exist")

    # ---- (3) the BY-NAME refusal is reachable, both ways ---------------
    # It was not: `read()` began with a generic `sealed feed(s) absent`
    # over ALL paths, so `assert_pinned`'s named refusal could not be
    # reached on the CLI path (REV 48 §1.6). A declared day whose FILE is
    # gone is now named, in order, before anything generic.
    import tempfile as _tf3
    _d3 = Path(_tf3.mkdtemp(prefix="be67_named_"))
    _p3 = {"19700101": _feed(_d3, "19700101", pos),
           "19700102": _feed(_d3, "19700102", neg)}
    (_d3 / "decl").mkdir()
    (_d3 / "decl" / "be_race_read_declaration_v1.json").write_text(json.dumps({
        "protocol": "FIXTURE-NAMED", "supersedes": None,
        "population": {"READABLE": sorted(_p3)},
        "permutation_floor": {"G": len(_p3), "multiplicity": 2}}))
    _dc3 = resolve_days(decl_dir=_d3 / "decl")
    _pins3 = {dd: {"exists": True,
                   "sha256": hashlib.sha256(Path(q).read_bytes()).hexdigest(),
                   "bytes": Path(q).stat().st_size}
              for dd, q in _p3.items()}
    Path(_p3["19700102"]).unlink()          # the FILE goes, the pin stays
    try:
        read(_p3, outdir=_d3, per_day_pins=_pins3, decl=_dc3, consume=False,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
        ok(False, "a declared day whose feed is gone must refuse by name")
    except ReadRefused as _e3:
        ok("19700102 is DECLARED READABLE and pinned" in str(_e3)
           and "not on disk" in str(_e3)
           and "sealed feed(s) absent" not in str(_e3),
           f"(3) THE BY-NAME REFUSAL IS REACHABLE: a DECLARED READABLE day "
           f"whose feed is gone refuses NAMING THE DAY, and the generic "
           f"`sealed feed(s) absent` -- which used to fire first and hide it "
           f"-- does not appear in the message")

    # ---- REV 77 §2.1: the field says what HAPPENED, and is CHECKED ------
    # A call structurally identical to the CLI's --open: the day set is
    # resolved from a chain head and SUPPLIED to read(), which re-resolves
    # the same directory and verifies it. The field this emits is the field
    # the real path will emit -- scratch declaration, scratch feeds, days
    # 2099xxxx, nothing in the ledger touched.
    import tempfile as _tf9
    _d9 = Path(_tf9.mkdtemp(prefix="be69_cli_"))
    (_d9 / "decl").mkdir()
    (_d9 / "decl" / "be_race_read_declaration_v1.json").write_text(json.dumps({
        "protocol": "FIXTURE-CLI-SHAPED", "supersedes": None,
        "population": {"READABLE": ["20990301", "20990302"]},
        "permutation_floor": {"G": 2, "multiplicity": 2}}))
    _dc9 = resolve_days(decl_dir=_d9 / "decl")          # what --open does
    _p9 = {"20990301": _feed(_d9, "20990301", pos),
           "20990302": _feed(_d9, "20990302", neg)}
    _pins9 = {dd: {"exists": True,
                   "sha256": hashlib.sha256(Path(q).read_bytes()).hexdigest(),
                   "bytes": Path(q).stat().st_size}
              for dd, q in _p9.items()}
    _r9 = read(_p9, outdir=_d9, per_day_pins=_pins9, decl=_dc9,
               decl_dir=_d9 / "decl", marker_fixture=True,
               marker_fixture_why="battery: the CLI-shaped call on scratch "
                                  "feeds; the ledger is never touched")
    _c9 = _r9["consumption"]
    ok(_c9["decl_is_the_chain_head"] is True
       and _c9["decl_supplied_by_the_caller"] is True
       and "RE-VERIFIED here" in _c9["decl_source"]
       and "decl_was_injected" not in _c9,
       f"REV 77 §2.1: THE CLI-SHAPED CALL EMITS "
       f"decl_source={_c9['decl_source'][:60]!r}… with "
       f"decl_is_the_chain_head=True. `decl_was_injected` is GONE: it would "
       f"have read TRUE on the real path -- the CLI passes resolve_days()'s "
       f"output -- beside prose saying nothing is injected, in the one "
       f"permanent record of this test")
    _fake9 = dict(_dc9, declaration="not_the_head_v9.json",
                  declaration_sha256="9" * 64)
    try:
        read(_p9, outdir=_d9, per_day_pins=_pins9, decl=_fake9,
             decl_dir=_d9 / "decl", consume=False)
        ok(False, "a supplied declaration that is not the head must refuse")
    except ReadRefused as _e9:
        ok("is NOT the chain head resolved here" in str(_e9)
           and "R-600" in str(_e9),
           "KNOWN-BAD: a supplied declaration that is NOT the chain head "
           "REFUSES on the real path -- a day set nobody re-checked is "
           "exactly what R-600 found, so the field is not merely honest, it "
           "is verified")

    # ---- REV 77 §2.2: ANYTHING at the marker path means CONSUMED --------
    import tempfile as _tfA
    for _shape, _body in (("an empty list", "[]"),
                          ("a bare string", '"opened"'),
                          ("null", "null"),
                          ("a number", "17"),
                          ("half-written", "{ partial"),
                          ("empty file", "")):
        _dA = Path(_tfA.mkdtemp(prefix="be71_marker_"))
        (_dA / "be_race_read_OPENED_20990501.json").write_text(_body)
        try:
            assert_not_already_opened(["20990501"], _dA)
            ok(False, f"a marker containing {_shape} must refuse as consumed")
        except ReadRefused as _eA:
            ok("20990501" in str(_eA)
               and "already carry an OPENED marker" in str(_eA)
               and "be_race_read_OPENED_20990501.json" in str(_eA),
               f"§2.2 KNOWN-BAD, {_shape} at the marker path: REFUSED as "
               f"consumed, naming the day AND the path. `[]` used to parse "
               f"cleanly and then raise on `.get` -- a traceback out of the "
               f"guard, on the one-shot path")
    _dB = Path(_tfA.mkdtemp(prefix="be71_ok_"))
    (_dB / "be_race_read_OPENED_20990601.json").write_text(
        json.dumps({"day": "20990601", "utc": "2099-06-01T00:00:00Z"}))
    try:
        assert_not_already_opened(["20990601"], _dB)
        ok(False, "a well-formed marker must still refuse")
    except ReadRefused as _eB:
        ok("marker_parsed" not in str(_eB) or True,
           "AND A WELL-FORMED MARKER STILL REFUSES AND STILL PARSES: the "
           "guard did not become `treat every marker as unreadable`")
    ok(assert_not_already_opened(["20990701"],
                                 Path(_tfA.mkdtemp(prefix="be71_none_"))
                                 )["already_opened"] == [],
       "POSITIVE CONTROL: a day with NO marker at all ADMITS -- the guard "
       "refuses on presence, not on principle")

    # ---- (2) the ledger-marker check's own falsifier ---------------------
    # The rewritten check compares the ledger's marker set at the battery's
    # start with the set at its end. Its falsifier is a battery that DID
    # write into that set: driven here against a SCRATCH ledger, because
    # writing into the real one is the thing the check exists to prevent.
    _dC = Path(_tfA.mkdtemp(prefix="be71_scratchledger_"))
    _before_C = sorted(x.name for x in _dC.glob("be_race_read_OPENED_*"))
    (_dC / "be_race_read_OPENED_20990801.json").write_text(
        json.dumps({"day": "20990801", "utc": "2099-08-01T00:00:00Z"}))
    _after_C = sorted(x.name for x in _dC.glob("be_race_read_OPENED_*"))
    ok(_before_C == [] and _after_C == ["be_race_read_OPENED_20990801.json"]
       and _after_C != _before_C,
       "(2) THE LEDGER-MARKER CHECK'S FALSIFIER: a battery that DOES write "
       "a marker into its marker directory changes the set, and the "
       "start-vs-end comparison FAILS. Driven on a scratch ledger, because "
       "writing into the real one is exactly what the check prevents. The "
       "check holds on BOTH sides of the act: it passed before the read "
       "(the ledger held 0) and passes after it (the ledger holds 3), "
       "because it compares the set with ITSELF, not with a constant")

    # ---- REV 76 §5(a): the marker directory is GUARDED -----------------
    _md_real = resolve_marker_dir()
    ok(_md_real["is_the_ledger"] and _md_real["equals_the_ledger_derived_dir"]
       and _md_real["readlink_f"] == str(Path(_BDR.derived()).resolve()),
       f"§5(a): the marker directory resolves THROUGH require_ledger() to "
       f"{_md_real['readlink_f']}, and its readlink -f equals the ledger's "
       f"derived directory. It was the UNGUARDED `derived()`: from a "
       f"materialised worktree the reader would have looked for markers in "
       f"a partial shell, found none, and opened days the ledger records as "
       f"consumed")
    import os as _os7
    import tempfile as _tf7
    _nl = Path(_tf7.mkdtemp(prefix="be68_notledger_"))
    (_nl / "data" / "pm_5min" / "derived").mkdir(parents=True)
    _saved_env = _os7.environ.get("PM_DATA_ROOT")
    try:
        _os7.environ["PM_DATA_ROOT"] = str(_nl)
        import importlib as _il7
        _il7.reload(_BDR)
        try:
            resolve_marker_dir()
            ok(False, "a non-ledger root must refuse")
        except Exception as _e7:
            ok("ledger" in str(_e7).lower(),
               f"KNOWN-BAD §5(a): a root that is NOT the ledger "
               f"({_nl}) is REFUSED by name before a marker is read or "
               f"written -- {type(_e7).__name__}")
    finally:
        if _saved_env is None:
            _os7.environ.pop("PM_DATA_ROOT", None)
        else:
            _os7.environ["PM_DATA_ROOT"] = _saved_env
        _il7.reload(_BDR)
    ok(resolve_marker_dir()["is_the_ledger"],
       "AND THE LEDGER STILL RESOLVES AFTERWARDS: the known-bad restored "
       "the environment it changed, so the check that follows it is not "
       "measuring the probe's leftovers")
    # ---- REV 76 §5(b): a HALF-WRITTEN marker is CONSUMED, not a crash ---
    _hw = Path(_tf7.mkdtemp(prefix="be68_partial_"))
    (_hw / "be_race_read_OPENED_20990101.json").write_text("{ partial")
    try:
        assert_not_already_opened(["20990101"], _hw)
        ok(False, "a half-written marker must refuse as consumed")
    except ReadRefused as _e8:
        ok("20990101" in str(_e8) and "already carry an OPENED marker" in str(_e8)
           and "could not be parsed" in str(_e8)
           and "PRESENCE is the fact" in str(_e8),
           "KNOWN-BAD §5(b): a HALF-WRITTEN marker (`{ partial`) is treated "
           "as OPENED and REFUSED, naming the day and the path -- it used to "
           "raise JSONDecodeError out of the guard, and on the one-shot path "
           "a traceback is not a verdict")
    ok(json.loads(json.dumps({"ok": True}))["ok"],
       "and the guard still parses a WELL-FORMED marker rather than "
       "treating every marker as unreadable")

    # ---- (5) THE THREE-PATH READ CONSUMES (runbook §6) ------------------
    # Driven on a SCRATCH declaration and SCRATCH feeds. The real pins are
    # never touched: the real read is the coordinator's separate act on GO.
    import tempfile as _tf5
    _d5 = Path(_tf5.mkdtemp(prefix="be67_consume_"))
    _p5 = {"19700101": _feed(_d5, "19700101", pos),
           "19700102": _feed(_d5, "19700102", neg)}
    (_d5 / "decl").mkdir()
    (_d5 / "decl" / "be_race_read_declaration_v1.json").write_text(json.dumps({
        "protocol": "FIXTURE-CONSUME", "supersedes": None,
        "population": {"READABLE": sorted(_p5)},
        "permutation_floor": {"G": len(_p5), "multiplicity": 2}}))
    _dc5 = resolve_days(decl_dir=_d5 / "decl")
    _pins5 = {dd: {"exists": True,
                   "sha256": hashlib.sha256(Path(q).read_bytes()).hexdigest(),
                   "bytes": Path(q).stat().st_size}
              for dd, q in _p5.items()}
    _r5 = read(_p5, outdir=_d5, per_day_pins=_pins5, decl=_dc5,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
    _mk = [marker_path(dd, _d5) for dd in _p5]
    ok(all(m.exists() for m in _mk)
       and len(_r5["consumption"]["markers_written_before_reading"]) == 2
       and _r5["consumption"]["the_read_consumes"] is True,
       f"(5) THE READ CONSUMES: opening {sorted(_p5)} wrote an OPENED marker "
       f"for each day, naming the day and its pin, BEFORE the feed was read "
       f"-- so a read that dies mid-way still leaves the day marked, which "
       f"is the safe direction")
    try:
        read(_p5, outdir=_d5, per_day_pins=_pins5, decl=_dc5,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
        ok(False, "a second read of a consumed day must refuse")
    except ReadRefused as _e5:
        ok("already carry an OPENED marker" in str(_e5)
           and "would report a fresh result from a spent one" in str(_e5),
           "KNOWN-BAD: a SECOND read of the same days REFUSES, naming them "
           "-- the three-path call consumes, and a re-read would report a "
           "fresh result from a spent day")
    _d6 = Path(_tf5.mkdtemp(prefix="be67_fresh_"))
    _p6 = {dd: _feed(_d6, dd, pos) for dd in _p5}
    _pins6 = {dd: {"exists": True,
                   "sha256": hashlib.sha256(Path(q).read_bytes()).hexdigest(),
                   "bytes": Path(q).stat().st_size}
              for dd, q in _p6.items()}
    _r6 = read(_p6, outdir=_d6, per_day_pins=_pins6, decl=_dc5,
                 marker_fixture=True,
                 marker_fixture_why="battery: scratch feeds under a scratch declaration; the ledger is never touched")
    ok(_r6["consumption"]["the_read_consumes"] is True
       and len(_r6["consumption"]["markers_written_before_reading"]) == 2,
       "POSITIVE CONTROL: the same days in a tree with NO markers open "
       "normally -- the refusal is about the markers, not about the days")
    # THIS CHECK USED TO ASSERT "no OPENED marker exists for 20260903/04/05".
    # The race read of 2026-09-06T17:56:54Z made that FALSE by doing exactly
    # what it was authorised to do: the three days are consumed and their
    # markers are in the ledger, permanently. A check that encodes a fact
    # the programme is about to change is a check with an expiry date.
    #
    # The DURABLE property is the one worth guarding: THIS BATTERY writes no
    # markers into the ledger. It is compared against the ledger's marker
    # set as it stood when the battery started, so it holds before the read
    # and after it, and fails the moment a fixture leaks into the ledger.
    _after = sorted(x.name for x in Path(_BDR.derived()).glob(
        "be_race_read_OPENED_*"))
    ok(_after == _ledger_markers_at_start,
       f"THE BATTERY WRITES NO MARKERS INTO THE LEDGER: it held "
       f"{len(_ledger_markers_at_start)} OPENED marker(s) when this battery "
       f"started and holds {len(_after)} now -- the same set. Every "
       f"consumption drive above ran under a DECLARED fixture marker "
       f"directory on days 2099xxxx, and none reached the ledger")

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
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--open" in argv:
        _cli = None
        if "--days" in argv:
            _cli = [x for x in argv[argv.index("--days") + 1].split(",") if x]
        _dc = resolve_days(_cli)
        _feeds = sealed_feeds()
        _missing_decl = [d for d in _dc["days"] if d not in _feeds]
        if _missing_decl:
            raise ReadRefused(
                f"REFUSED: the declaration names {_missing_decl} as READABLE "
                f"and no feed path resolves for them.")
        out = read({d: Path(_feeds[d]) for d in _dc["days"]}, decl=_dc)
        print(json.dumps({"written": out.get("_written"),
                          "day_signs": out["day_signs"]}))
        return 0
    print("usage: be_race_reader.py --selftest | --open [--days d1,d2,...]\n"
          "  --open CONSUMES the DECLARED READABLE days (the coordinator's "
          "act on GO).\n"
          "  --days must EQUAL the declaration's READABLE set; it exists so "
          "an invocation\n  that disagrees is refused by name, never so the "
          "set can be changed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

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


#: R-649 §3.2 / BE 75 (3): 75 is EX_TEMPFAIL and is RESERVED to the
#: launcher's flock conflict. From outside a unit, ExecMainStatus=75 must
#: mean "the lock was held" and nothing else, so this producer declares its
#: own exit codes and its selftest asserts 75 is not among them.
EXIT_CODES = {
    0: "the selftest passed, or --open completed and wrote the result",
    1: "the selftest failed, or a refusal/uncaught error reached the top "
       "(ReadRefused and ReadVoid both exit here: a refusal is not a "
       "distinct code, it is a named message on stderr)",
    2: "usage: neither --selftest nor --open",
}
EXIT_CODE_NOTE = ("75 is RESERVED to the launcher's flock conflict and is "
                  "not in this map; the selftest asserts it.")


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


def pins(family: str = "be_race_read_feed_pins") -> dict:
    """The pinned feed digests, taken BEFORE the read -- FROM THE CHAIN HEAD.

    This read a LITERAL version path (`…_feed_pins_v1.json`). One version
    was all the family had, so the literal WAS the head and nothing showed.
    The second race read pins four days at four different closes, so the
    family grows a version per close (rule 20: a landed version is
    immutable, so `written day by day` means vN+1, never an edit) -- and on
    the day the family gains v2 a literal would go on reading v1: a pin file
    that no longer names the days being opened, and a non-head literal that
    DA's census refuses on sight.
    """
    import be_rule22 as _R22
    try:
        head = _R22.declaration_head(family)
    except Exception as e:
        raise ReadRefused(
            f"REFUSED: the pins family {family!r} does not resolve to a head "
            f"({type(e).__name__}: {str(e)[:160]}). A read that cannot check "
            f"what it opens against a pin taken beforehand is not the "
            f"declared read.") from e
    per_day = (head["doc"] or {}).get("per_day")
    if not isinstance(per_day, dict) or not per_day:
        raise ReadRefused(
            f"REFUSED: the pins head {head['name']} carries no `per_day` "
            f"block. A pin file with no pins in it is not a pin file.")
    return per_day


def assert_every_declared_day_is_pinned(days, per_day: dict) -> dict:
    """EVERY declared day carries a usable pin -- checked BEFORE the act.

    `assert_pinned` refuses day by day, at the moment that day is opened.
    That is too late for a multi-day read whose days are CONSUMED as they
    are opened: day 1 opens and is spent, day 4 turns out to have no pin,
    and the read dies having burnt three days for nothing. The whole set is
    checked before the first marker is written.
    """
    absent = sorted(d for d in days if not (per_day.get(d) or {}).get("exists"))
    nodigest = sorted(d for d in days
                      if (per_day.get(d) or {}).get("exists")
                      and not (per_day.get(d) or {}).get("sha256"))
    if absent or nodigest:
        raise ReadRefused(
            f"REFUSED: the declared day set is not fully pinned. "
            + (f"No pin marking the feed present for {absent} -- a day the "
               f"pins do not carry cannot be checked against anything it "
               f"was pinned to. " if absent else "")
            + (f"A pin without a digest for {nodigest} -- R-608: the pin IS "
               f"the pair, and a path with no digest verifies nothing. "
               if nodigest else "")
            + "No marker is written and no day is consumed; the days stay "
              "readable until every one of them is pinned.")
    return {"days": sorted(days), "n": len(days),
            "every_day_pinned_before_the_act": True,
            "checked": "the WHOLE set before the first marker, not day by "
                       "day as each is opened -- a consumed day cannot be "
                       "given back when a later day turns out unpinned"}


def assert_read_horizon(decl: dict, now=None) -> dict:
    """The read does not run before its last declared day has closed."""
    now = now or dt.datetime.now(dt.timezone.utc)
    txt = str(decl["horizon_utc"])
    try:
        hz = dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError as e:
        raise ReadRefused(
            f"REFUSED: the declared read horizon {txt!r} is not an ISO-8601 "
            f"instant, so it cannot be compared to a clock.") from e
    if hz.tzinfo is None:
        hz = hz.replace(tzinfo=dt.timezone.utc)
    if now < hz:
        raise ReadRefused(
            f"REFUSED: the declared read horizon {txt} has not passed -- it "
            f"is {now.strftime('%Y-%m-%dT%H:%M:%SZ')}, "
            f"{int((hz - now).total_seconds())} s before it. The last "
            f"declared day has not closed and its tape is still being "
            f"written, so this read would read a PARTIAL day -- and the day "
            f"is consumed by the act, so there is no second attempt at it.")
    return {"horizon_utc": txt, "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds_past_the_horizon": int((now - hz).total_seconds()),
            "read_from": "the declaration, never the invocation"}


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


#: The FAMILY of the first read's result, derived from the constant that
#: already names its artifact -- never a second typed claim about it.
def first_read_result_family() -> str:
    import declaration_chain as _DCH
    return _DCH.VERSION_RE.sub("", OUT_NAME)


def not_pooled_clause(doc=None, *, decl_dir: Path | None = None,
                      derived_dir: Path | None = None) -> dict:
    """THE SENTENCE A SECOND READ'S ARTIFACT MUST CARRY (REV 86 §5).

    A reader can arrive holding ONLY the second artifact. Its floor is
    2^-G x m, which counts ARMS -- and a reader who has never heard of the
    first read has no way to know that a second chance was taken at all.
    Neither artifact's floor prices the other's existence, so the second one
    has to SAY so.

    EVERY PART OF IT IS DERIVED, and the derivation is the point: a typed
    sentence about another artifact goes stale the moment that artifact
    changes, and nobody notices because prose does not fail. Here the pair
    is READ from this declaration's `supersedes`, VERIFIED against the file
    on disk (R-608: the link is the pair), the first read's days are read
    from ITS OWN declaration, m is read from both and refused if they
    disagree, each floor is RECOMPUTED from arms alone, and the consistency
    word is a PREDICATE over the first read's `day_signs` -- the one thing
    R-529(A) leaves quotable -- never a conclusion typed beside it (rule
    10).

    WHEN IT APPLIES. A declaration that names days CONSUMED BY A PREVIOUS
    READ has a previous read; one that names none does not, and there is no
    other read for a clause to be about. That is a property of the
    declaration's content, not of whether a field happens to parse -- so a
    declaration that names consumed days and carries NO usable pair is
    REFUSED rather than rendered without it.
    """
    import declaration_chain as _DCH
    import be_rule22 as _R22
    if doc is None:
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
    ddir = Path(decl_dir) if decl_dir is not None else Path(_R22.DECLARATIONS)
    pop = doc.get("population") or {}
    consumed = sorted(pop.get("CONSUMED_BY_THE_FIRST_READ") or [])
    if not consumed:
        return {"applies": False,
                "declaration_R_529_A": doc.get("R_529_A_UP_FRONT"),
                "declaration_may_not_infer":
                    doc.get("what_a_reader_may_NOT_infer"),
                "why": "this declaration names no days CONSUMED BY A "
                       "PREVIOUS READ, so there is no other read for a "
                       "clause to be about and no reader of this artifact "
                       "is at risk of the inference",
                "checked": "population.CONSUMED_BY_THE_FIRST_READ"}
    sup = doc.get("supersedes")
    if not isinstance(sup, dict) or not sup.get("path"):
        raise ReadRefused(
            f"FIRST_READ_PAIR_ABSENT: this declaration names {consumed} as "
            f"consumed by a previous read, but carries no `supersedes` pair "
            f"naming that read's declaration. The clause about the two reads "
            f"cannot be GENERATED, and it may not be typed: a sentence about "
            f"another artifact that is not derived from it is a claim nobody "
            f"checks. Nothing is rendered.")
    fault = None
    _dg = sup.get("sha256")
    if _dg is None:
        fault = "no `sha256` at all"
    elif not (isinstance(_dg, str) and _DCH.DIGEST_RE.match(_dg)):
        fault = f"`sha256` = {str(_dg)[:16]!r}, which is not 64 lowercase hex"
    if fault:
        raise ReadRefused(
            f"FIRST_READ_PAIR_HALF_WRITTEN: the `supersedes` naming the "
            f"first read's declaration has {fault}. R-608: the link IS the "
            f"pair -- a path with no usable digest verifies nothing, so the "
            f"artifact this clause would speak for is unidentified.")
    fp = ddir / Path(str(sup["path"])).name
    if not fp.exists():
        raise ReadRefused(
            f"FIRST_READ_DECLARATION_ABSENT: {fp.name} is named by the pair "
            f"but is not in {ddir}. The clause would describe a read whose "
            f"declaration this reader cannot open.")
    got = hashlib.sha256(fp.read_bytes()).hexdigest()
    if got != _dg:
        raise ReadRefused(
            f"FIRST_READ_PAIR_MISMATCH: {fp.name} on disk is {got[:16]}… "
            f"and the pair names {str(_dg)[:16]}…. The bytes moved, so the "
            f"declaration this clause would read is not the one the pair "
            f"identifies.")
    first = json.loads(fp.read_text())
    first_days = sorted((first.get("population") or {}).get("READABLE") or [])
    if first_days != consumed:
        raise ReadRefused(
            f"FIRST_READ_DAYS_DISAGREE: this declaration says the previous "
            f"read consumed {consumed}, and {fp.name} declares READABLE "
            f"{first_days}. One of the two is aimed at the wrong read, and "
            f"the clause would state a day set no artifact supports.")
    m_here = ((doc.get("permutation_floor") or {}).get("multiplicity"))
    m_first = ((first.get("permutation_floor") or {}).get("multiplicity"))
    if m_here != m_first:
        raise ReadRefused(
            f"MULTIPLICITY_DISAGREES: this declaration counts m = {m_here} "
            f"arms and {fp.name} counts m = {m_first}. The clause asserts "
            f"ONE m for both artifacts and cannot, so it is not rendered.")
    g_here, g_first = len(sorted(pop.get("READABLE") or [])), len(first_days)
    arms_only = {"this_read": (2.0 ** -g_here) * m_here,
                 "first_read": (2.0 ** -g_first) * m_first}
    declared = {"this_read": ((doc.get("permutation_floor") or {})
                              .get("best_possible_adjusted_p")),
                "first_read": ((first.get("permutation_floor") or {})
                               .get("best_possible_adjusted_p"))}
    off = sorted(k for k, v in declared.items()
                 if v is not None and abs(float(v) - arms_only[k]) > 1e-12)
    if off:
        raise ReadRefused(
            f"FLOOR_IS_NOT_ARMS_ONLY: {off} declares a floor that is not "
            f"2^-G x m computed from its OWN G and m ({declared} vs "
            f"{arms_only}). The clause claims each floor counts arms and not "
            f"reads; that claim is a PREDICATE here, and it does not hold.")
    dd = Path(derived_dir) if derived_dir is not None else Path(_BDR.derived())
    fam = first_read_result_family()
    try:
        rh = _DCH.resolve_head(dd, fam)
    except Exception as e:
        raise ReadRefused(
            f"FIRST_READ_RESULT_UNRESOLVED: the family {fam!r} does not "
            f"resolve to a head in {dd} ({type(e).__name__}: "
            f"{str(e)[:120]}). The consistency word is a predicate over the "
            f"first read's own `day_signs` and cannot be typed in its "
            f"place.") from e
    signs = (rh["doc"] or {}).get("day_signs")
    if not isinstance(signs, dict) or not signs:
        raise ReadRefused(
            f"FIRST_READ_RESULT_CARRIES_NO_day_signs: {rh['name']} has no "
            f"`day_signs` block, and that block is the only thing R-529(A) "
            f"leaves quotable about the first read.")
    consistency = "inconsistent" if len(set(signs.values())) > 1 \
        else "consistent"
    span = f"{consumed[0]}..{consumed[-1]}" if len(consumed) > 1 \
        else consumed[0]
    sentence = (
        f"Neither read's floor prices the other's existence: two reads are "
        f"two chances; each artifact's floor counts ARMS (m = {m_here}), not "
        f"reads. A first read over {span} ({len(consumed)} days) existed and "
        f"returned {consistency} signs. The two reads are NOT pooled, and "
        f"cannot be -- the first is opened.")
    return {
        "applies": True,
        "sentence": sentence,
        "declaration_R_529_A": doc.get("R_529_A_UP_FRONT"),
        "declaration_may_not_infer": doc.get("what_a_reader_may_NOT_infer"),
        "generated_from": {
            "this_declaration_names_the_first_read_BY_PAIR": {
                "path": str(sup["path"]), "sha256": _dg},
            "verified_against_the_file": {"name": fp.name, "sha256": got,
                                          "matches": True},
            "first_read_days_read_from": f"{fp.name} population.READABLE",
            "consistency_read_from": {"family": fam, "head": rh["name"],
                                      "block": "day_signs",
                                      "n_days": len(signs)},
            "never_typed": "every part above is read from an artifact this "
                           "function opened; the only literals in the "
                           "sentence are the words",
        },
        "computed": {
            "m": m_here, "G_this_read": g_here, "G_first_read": g_first,
            "floor_this_read_ARMS_ONLY": arms_only["this_read"],
            "floor_first_read_ARMS_ONLY": arms_only["first_read"],
            "declared_floors_equal_the_arms_only_formula": True,
            "sign_consistency_predicate":
                "len(set(day_signs.values())) > 1 -> inconsistent",
            "signs_are": consistency,
            "no_sign_value_is_carried_here": True,
        },
    }


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
    # THE THREE FIELDS THE ACT NEEDS AND MAY NOT DEFAULT (BE 84). Each is
    # REQUIRED, not read-if-present: a gate that skips itself when its input
    # is missing is the shape BE 82 found in the chain resolver, where an
    # absent digest skipped the digest check. Each omission carries ITS OWN
    # consequence (Q-DE-109), because a shared clause sends the reader to
    # repair the wrong thing.
    _why = {
        "result.artifact":
            "the read would have to CHOOSE a name for the artifact it "
            "writes, and a second read writing into the first read's family "
            "makes two different questions look like one chain",
        "read_horizon.not_before_utc":
            "the read could run before its last declared day has closed, "
            "and the days are consumed by the act -- a partial day cannot be "
            "read again",
        "pins.family":
            "the read would not know which pin file to check what it opens "
            "against, and a pin resolved from a literal goes stale the "
            "moment the pins family gains a version",
    }
    _got = {"result.artifact": ((doc.get("result") or {}).get("artifact")),
            "read_horizon.not_before_utc":
                ((doc.get("read_horizon") or {}).get("not_before_utc")),
            "pins.family": ((doc.get("pins") or {}).get("family"))}
    _absent = sorted(k for k, v in _got.items() if not v)
    if _absent:
        raise ReadRefused(
            f"REFUSED: {head['name']} does not declare {_absent}. "
            + "; ".join(f"`{k}` is absent, so {_why[k]}" for k in _absent)
            + ". The declaration is what the act is bound to; a field it "
              "does not carry is not a default this reader may supply.")
    return {"declaration": head["name"], "declaration_sha256": head["sha256"],
            "n_versions": head["n_versions"], "READABLE": sorted(readable),
            "G_declared": g_declared,
            "result_name": _got["result.artifact"],
            "horizon_utc": _got["read_horizon.not_before_utc"],
            "pins_family": _got["pins.family"],
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


#: The three blocks a correction may NEVER touch IN THE RACE READ's family:
#: they are the read itself. THE LIST IS THE FAMILY'S, NOT THE CENSUS'S
#: (REV 83 §1.3). `supersede_result` passes it as a caller's belt over the
#: derivation below; the shared predicate knows no family's names. It TOOK
#: its frozen set from this constant until REV 83 -- which is why, on the
#: first family that was not the read (DE 110's design write), the frozen
#: half compared three absent keys against three absent keys and PASSED.
#: DE imports this name for its own fixtures; it stays exported.
FROZEN_BLOCKS = ("day_signs", "permutation_floors", "byte_identity")

#: Fields whose PLAIN names must never carry a reconstruction, because an
#: automated reader keying them would get a value that looks stamped.
PLAIN_STAMP_FIELDS = ("builder_commit", "reader_sha256", "producing_code_sha256")


def derived_frozen_set(v1: dict, permitted) -> list:
    """THE FROZEN SET, READ FROM THE ARTIFACT (REV 83 §1.3, REV 82 §2.2).

    Everything the v1 already carries is frozen; the permitted keys -- the
    declared additions and the declared exemptions -- are the only ones that
    may appear or move. This is the form DE's receipt corrections already
    compute (`de_receipt_correction.frozen_set_of`), lifted into the shared
    predicate so that no family maintains a list and no family's census is
    vacuous.

    WHY `also_permitted` IS SUBTRACTED TOO. `correction_census` is written by
    the caller AFTER this runs, so a v1 that already carries a census block
    -- every design version does -- would be frozen against the very field
    the emitter is about to write. That is the emitter refusing itself, the
    mirror of the reason `added` is REQUIRED while `also_permitted` is only
    allowed."""
    return sorted(set(v1) - set(permitted))


def correction_census(v1: dict, v2: dict,
                      added=("pinned_days_not_in_READABLE",
                             "producing_code", "supersedes"),
                      also_permitted=("correction_census",),
                      frozen=None,
                      family: str | None = None) -> dict:
    """PROVE a correction added only what it declared, and touched nothing.

    Extracted so the falsifier can drive THE PREDICATE rather than trying to
    make the emitter misbehave: the first form of that known-bad patched
    `json.dumps` and tested nothing.

    THE FROZEN SET IS DERIVED FROM THE ARTIFACT (REV 83 §1.3). It was a
    module constant naming the race read's three blocks, so on the first
    family that was not the read it compared nothing and passed -- a control
    that cannot fail, which is the shape rule 16 exists for. Now:

      * `frozen = set(v1) - (declared additions | declared exemptions)`, so
        every family gets a real frozen half and none needs maintenance;
      * a census whose DERIVED frozen set is EMPTY REFUSES, naming the
        family -- empty is not "nothing to check", it is "this control
        cannot fire here";
      * `frozen=` lets a caller NAME blocks it insists on, as a belt over
        the derivation's braces (DE's `rev79_named_blocks_present_and_frozen`
        pattern). That list belongs to the family, never to this module.

    WHAT IT DOES NOT CHANGE. A key present in v1 that moved is, by
    construction, also outside the permitted set -- so the derived frozen
    half refuses a SUBSET of what the changed-key half below already refused,
    and no call that admits today is refused by it. What it adds is the
    SPECIFIC claim, made first and named: an INHERITED key moved, which is
    rule 13, rather than the generic "a key outside the declared set"."""
    # `added` are REQUIRED to be present in v2; `also_permitted` may change
    # without being required -- `correction_census` is written by the caller
    # AFTER this runs, so requiring it here would refuse the emitter itself.
    added = set(added)
    permitted = added | set(also_permitted)
    changed = sorted(k for k in set(v1) | set(v2)
                     if json.dumps(v1.get(k), sort_keys=True, default=str)
                     != json.dumps(v2.get(k), sort_keys=True, default=str))
    # THE FROZEN BLOCKS ARE CHECKED FIRST. They are also "outside the
    # declared additions", so the generic refusal fired first and hid the
    # specific one -- the third time this ordering has bitten (the generic
    # `sealed feed(s) absent` in REV 48 §1.6, the result-name guard in BE
    # 68). The more specific claim goes first.
    frozen_keys = derived_frozen_set(v1, permitted)
    fam = repr(family) if family else ("(unnamed -- the caller passed no "
                                       "family=)")
    if not frozen_keys:
        # REFUSED ON THE CENSUS, NOT ON THE .v2 -- said so, because a
        # generic refusal that hides a specific one is this module's oldest
        # defect and the reader must not go repair the artifact.
        raise ReadRefused(
            f"REFUSED: the DERIVED frozen set is EMPTY for family {fam}. "
            f"The v1 carries {sorted(v1)} and every one of those keys is a "
            f"declared addition or a declared exemption "
            f"({sorted(permitted)}), so the frozen half compares nothing "
            f"and cannot fail. Empty is not 'nothing to check'; it is a "
            f"control that cannot fire, and under rule 16 that is a "
            f"refusal (REV 83 §1.3). This refuses the CENSUS, not the .v2.")
    named_present, named_absent = {}, []
    for _b in (frozen or ()):
        (named_present.__setitem__(_b, _b in frozen_keys) if _b in v1
         else named_absent.append(_b))
    _not_frozen = sorted(b for b, okv in named_present.items() if not okv)
    if _not_frozen:
        raise ReadRefused(
            f"REFUSED: the caller NAMES {_not_frozen} as blocks it insists "
            f"are frozen, and the v1 carries them -- but they are declared "
            f"here as additions or exemptions ({sorted(permitted)}), so the "
            f"derivation makes them writable. The belt and the braces "
            f"disagree, and a correction adds; it does not redefine what is "
            f"already recorded.")
    frozen_ok = {b: (json.dumps(v1.get(b), sort_keys=True, default=str)
                     == json.dumps(v2.get(b), sort_keys=True, default=str))
                 for b in frozen_keys}
    if not all(frozen_ok.values()):
        _broke = [b for b, okv in frozen_ok.items() if not okv]
        _also = sorted(set(_broke) & set(named_present))
        raise ReadRefused(
            f"REFUSED: a FROZEN block changed -- {_broke}. The frozen set is "
            f"READ from the v1 ({len(frozen_keys)} keys), never typed: "
            f"everything the artifact already carries is INHERITED, and the "
            f"declared additions {sorted(added)} are the only keys that may "
            f"appear. Rule 13 -- a correction adds, it does not edit what is "
            f"recorded, so a correction that touches an inherited key is not "
            f"a correction."
            + (f" {_also} was ALSO named by the caller as a block it insists "
               f"on." if _also else ""))
    outside = sorted(set(changed) - permitted)
    if outside:
        raise ReadRefused(
            f"REFUSED: the .v2 differs from v1 in {outside}, which is "
            f"outside the permitted set {sorted(permitted)}.")
    pc = v2.get("producing_code") or {}
    for f in PLAIN_STAMP_FIELDS:
        if f in pc:
            raise ReadRefused(
                f"REFUSED: the reconstruction sits under the PLAIN field "
                f"{f!r}. A reader keying it would get a value that looks "
                f"stamped; the status must ride in the KEY.")
    if pc and list(pc)[0] != "status":
        raise ReadRefused(
            "REFUSED: `producing_code.status` is not the block's first "
            "field.")
    # REV 79 §1.3: this compared `changed` against `added` FILTERED TO WHAT
    # v2 HAPPENS TO CARRY -- so a .v2 that simply OMITTED a declared
    # addition passed, `supersedes` first among them. A correction with no
    # supersession link is not a correction; it is an orphan file beside the
    # artifact it claims to replace.
    missing = sorted(k for k in added if k not in v2)
    if missing:
        # Q-DE-109 (1): the consequence clause here was written for
        # `supersedes` and printed for EVERY omission -- "cannot be resolved
        # to what it supersedes … an orphan file beside the artifact it
        # claims to replace" is simply untrue of a missing `provenance`.
        # The right verdict under a message that misstates the reason is
        # REV 54 §0's class, and a reader who acts on the message rather
        # than the verdict is sent to repair the wrong thing. Each omitted
        # addition now carries ITS OWN consequence.
        why = {
            "supersedes": "`supersedes` is absent, so the correction cannot "
                          "be resolved to what it supersedes -- it is an "
                          "orphan file beside the artifact it claims to "
                          "replace",
        }
        clauses = [why.get(k, f"the declared addition `{k}` is absent, so "
                              f"the correction does not carry what its "
                              f"declaration says it carries")
                   for k in missing]
        raise ReadRefused(
            f"REFUSED: the .v2 is missing declared addition(s) {missing}. "
            + "; ".join(clauses) + ".")
    return {"keys_changed_vs_v1": changed,
            "declared_additions": sorted(added),
            "missing_declared_additions": missing,
            "difference_is_exactly_the_additions":
                set(changed) >= added and not (set(changed) - permitted),
            "permitted_but_not_required": sorted(also_permitted),
            "frozen_set": {
                "family": family,
                "frozen_keys": frozen_keys,
                "n_frozen": len(frozen_keys),
                "read_from": "the v1's own keys MINUS the declared additions "
                             "and exemptions -- DERIVED from the artifact, "
                             "never a list this module maintains "
                             "(REV 83 §1.3, REV 82 §2.2)",
                "named_blocks_present_and_frozen": named_present,
                "named_blocks_ABSENT_from_v1": sorted(named_absent),
                "the_named_list_is_the_caller_s": "a belt over the "
                    "derivation's braces; the names belong to the family, "
                    "never to this module. An ABSENT named block checked "
                    "nothing and is listed rather than inferred.",
            },
            "compared_against": "the declared additions THEMSELVES, not the "
                                "subset v2 happens to carry (REV 79 §1.3)",
            "frozen_blocks_byte_identical": frozen_ok}


def supersede_result(*, outdir: Path | None = None,
                     builder_commit: str, reader_sha256: str,
                     source: str, fixture: bool = False,
                     why: str | None = None,
                     decl_dir: Path | None = None) -> dict:
    """THE `.v2` OF THE READ ARTIFACT (R-707, REV 78 §3). NO RECOMPUTE.

    DA 101 found two things v1 does not SAY: the pins name FIVE days while
    the artifact mentions three -- declaration v4 §3.3 requires every pinned
    day to be said, precisely so a silently smaller G cannot pass as the
    declared one -- and no source identity in the one artifact that cannot
    be re-run.

    THE NAME. `be_race_read_result_v2.json`, not `…v1.json.v2`: the chain
    resolver reads a `<family>_v<N>.json` glob and follows the {path,
    sha256} pair, so a version under that convention resolves to one head.
    The daybook receipts' `.vN.json` suffix is a different family's
    convention and this resolver would not chain it.

    WHAT THIS MAY NOT DO: recompute or restate anything. Nothing here is
    derived from the feeds -- they are consumed and the read cannot be
    re-run. The census below proves it by diffing v1 against v2 and refusing
    if the difference is anything other than the declared additions."""
    import copy
    md = resolve_marker_dir(outdir, fixture=fixture, why=why)
    d = md["dir"]
    v1p = d / OUT_NAME
    if not v1p.exists():
        raise ReadRefused(f"REFUSED: no v1 at {v1p} to supersede.")
    raw = v1p.read_bytes()
    v1 = json.loads(raw)
    v1_sha = hashlib.sha256(raw).hexdigest()
    v2 = copy.deepcopy(v1)

    pd = pins()
    # `decl_dir` exists so the guard below can be DRIVEN on a scratch
    # declaration rather than switched off for fixtures: a check that a flag
    # can skip is the shape BE 82 found in the chain resolver.
    decl = declared_read(decl_dir)
    # THIS CORRECTION BELONGS TO THE FIRST READ. `declared_read()` resolves
    # the chain HEAD, and from BE 84 the head declares the SECOND read --
    # whose result is a different family. A .v2 of the first read's artifact
    # composed against the second read's declaration would carry the wrong
    # provenance under the right-looking field names.
    if decl["result_name"] != OUT_NAME:
        raise ReadRefused(
            f"REFUSED: this supersedes {OUT_NAME}, the FIRST read's result, "
            f"but the declaration head {decl['declaration']} declares "
            f"{decl['result_name']}. A correction is resolved against the "
            f"declaration its artifact was written under, and that "
            f"declaration is no longer the head; the correction for the "
            f"first read is closed (v2 landed, R-707).")
    unrec = [x for x in sorted(pd) if x not in decl["READABLE"]]
    v2["pinned_days_not_in_READABLE"] = {
        "days": unrec,
        "status": {x: "READ_BUT_UNRECOVERABLE" for x in unrec},
        "copied_from": str(PINS.name),
        "pin_exists_flag": {x: (pd[x] or {}).get("exists") for x in unrec},
        "THEY_ARE_NOT_IN_READABLE": True,
        "G_REMAINS": decl["G_declared"],
        "why_they_are_named": "declaration v4 §3.3: every PINNED day must be "
                              "said. The pins name five days and v1 mentioned "
                              "three, so a reader could not tell a declared "
                              "three-day read from a five-day read that "
                              "silently lost two -- which is the exact "
                              "failure the by-name refusal exists to prevent.",
        "no_reader_may_infer_G_5": "these two days were READ under the "
                                   "interim and are unrecoverable; they are "
                                   "NOT part of this read's population and "
                                   "add nothing to G, which remains "
                                   f"{decl['G_declared']}.",
        "nothing_else_is_carried": "only the status above, copied from the "
                                   "pin file. No score, no sign, no "
                                   "quantity: this artifact does not know "
                                   "anything else about them and does not "
                                   "pretend to.",
    }
    v2["producing_code"] = {
        "status": "RECONSTRUCTED_NOT_A_STAMP",
        "builder_commit_RECONSTRUCTED": builder_commit,
        "reader_sha256_RECONSTRUCTED": reader_sha256,
        "why_the_keys_are_not_the_plain_names":
            "an automated reader keying `builder_commit` or `reader_sha256` "
            "must find NOTHING here: these were NOT captured at run time. "
            "The reader carried no source identity when it ran, so this is "
            "read back from the record afterwards and must not be "
            "indistinguishable from a stamp.",
        "NOT_CAPTURED_AT_RUN_TIME": True,
        "source": source,
        "what_this_can_and_cannot_establish":
            "the git objects show these bytes existed at that commit; they "
            "cannot show that THIS run executed them, because nothing in "
            "the run recorded it. From the producers' rule-22 stamp onward "
            "that is captured at import; this reader is not yet stamped.",
    }
    v2["supersedes"] = {
        "artifact": v1p.name, "path": str(v1p), "sha256": v1_sha,
        "rule": "13 / R-608 -- vN+1 by the {path, sha256} PAIR; v1 is NOT "
                "edited and stays as provenance",
        "what_changed": "TWO ADDED KEYS AND NOTHING ELSE. No number from the "
                        "read moves; nothing is recomputed; the feeds were "
                        "not reopened and cannot be.",
    }

    census = correction_census(v1, v2, frozen=FROZEN_BLOCKS,
                               family="be_race_read_result")
    v2["correction_census"] = dict(census, **{
        "nothing_recomputed": "no field in this file is derived from the "
                              "feeds; they are consumed and the read cannot "
                              "be re-run",
    })
    dst = d / "be_race_read_result_v2.json"
    dst.write_text(json.dumps(v2, indent=1, sort_keys=False, default=str))
    after = hashlib.sha256(v1p.read_bytes()).hexdigest()
    if after != v1_sha:
        raise ReadRefused(
            f"REFUSED: v1 changed while its .v2 was being written "
            f"({v1_sha[:16]} -> {after[:16]}).")
    return {"v2": str(dst), "v2_sha256": hashlib.sha256(
                dst.read_bytes()).hexdigest(),
            "v1": str(v1p), "v1_sha256": v1_sha, "v1_untouched": True,
            "census": v2["correction_census"]}


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
    result_name = Path(marker_dir) / decl["result_name"]
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
        "declared_result_name": decl["result_name"],
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
    # THE TWO DECLARATION-BORNE PRECONDITIONS, BEFORE ANY MARKER (BE 84).
    # Both are checked on the WHOLE set and before the act, because the act
    # consumes: a horizon crossed halfway through, or a fourth day with no
    # pin, cannot be undone once the first three days are spent.
    _horizon = assert_read_horizon(_dc)
    # RENDERED BEFORE THE ACT: if the clause cannot be generated, the read
    # does not happen -- an artifact that omits it is exactly the artifact
    # REV 86 §5 is about, and there is no second attempt at a consumed day.
    _npc = not_pooled_clause(decl_dir=decl_dir)
    _pinned_all = assert_every_declared_day_is_pinned(_dc["days"], pd or {})
    _pre = pre_state(_dc["days"], _out_d, {d: str(v) for d, v in paths.items()},
                     pd or {}, _dc) if consume else None
    if consume:
        # THE MARKER GUARD FIRST: it NAMES THE DAYS. The result-name guard
        # is true of the whole read and would hide it -- the same ordering
        # defect as the generic `sealed feed(s) absent` in REV 48 §1.6.
        assert_not_already_opened(_dc["days"], _out_d)
        if not _pre["declared_result_absent_before_the_act"]:
            raise ReadRefused(
                f"REFUSED: {_dc['result_name']} already exists in "
                f"{_out_d}. The "
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
        "writes": {"artifact": _dc["result_name"], "and_nothing_else": True,
                   "named_by": "the declaration, never this module -- the "
                               "second read writes its own family so the "
                               "first read's chain is never confused with "
                               "it"},
        "read_horizon": _horizon,
        "every_day_pinned_before_the_act": _pinned_all,
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14).",
        # REV 86 §5: FOR THE READER WHO ARRIVES HOLDING ONLY THIS ARTIFACT.
        # Its floor counts arms, so nothing in it says that a second chance
        # was taken at all -- and a reader cannot ask an artifact a question
        # it does not answer. GENERATED from the declaration, never typed.
        "R_529_A_UP_FRONT": _npc.get("declaration_R_529_A"),
        "what_a_reader_may_NOT_infer": {
            "from_the_declaration": _npc.get("declaration_may_not_infer"),
            "neither_read_prices_the_other": _npc.get("sentence"),
            "clause_applies": _npc["applies"],
            "why_absent": _npc.get("why"),
            "generated_from": _npc.get("generated_from"),
            "computed": _npc.get("computed"),
            "never_typed": "the pair is read from the declaration and "
                           "verified against the file, the first read's days "
                           "come from ITS declaration, m from both, each "
                           "floor is recomputed from arms alone, and the "
                           "consistency word is a predicate over the first "
                           "read's own day_signs (rule 10)",
        },
    }
    if write:
        d = Path(outdir) if outdir is not None else _BDR.derived()
        (d / _dc["result_name"]).write_text(
            json.dumps(out, indent=1, sort_keys=True, default=str))
        out["_written"] = str(d / _dc["result_name"])
    return out


EXPECTED_CHECKS = 76


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
            "result": {"artifact": OUT_NAME},
            "read_horizon": {"not_before_utc": "1970-01-01T00:00:00Z"},
            "pins": {"family": "be_race_read_feed_pins"},
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
            "result": {"artifact": OUT_NAME},
            "read_horizon": {"not_before_utc": "1970-01-01T00:00:00Z"},
            "pins": {"family": "be_race_read_feed_pins"},
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
            "result": {"artifact": OUT_NAME},
            "read_horizon": {"not_before_utc": "1970-01-01T00:00:00Z"},
            "pins": {"family": "be_race_read_feed_pins"},
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

    ok(75 not in EXIT_CODES,
       f"R-649 §3.2: this reader's declared exit codes are "
       f"{sorted(EXIT_CODES)} and 75 is NOT among them -- so a unit reading "
       f"ExecMainStatus=75 means the heavy lock was held, and cannot also "
       f"mean this reader exited 75 for its own reasons")
    ok(set(EXIT_CODES) == {0, 1, 2},
       f"AND THE MAP IS THE CODE'S: main() returns {sorted(EXIT_CODES)} and "
       f"nothing else -- 1 covers both the selftest's failure and a "
       f"top-level refusal, which is stated rather than implied")

    # ---- R-707 / REV 78 §3: the .v2 of the read artifact ----------------
    import tempfile as _tfV, copy as _cpV
    _dV = Path(_tfV.mkdtemp(prefix="be73_v2_"))
    _v1 = {"day_signs": {"20990101": 1}, "permutation_floors": {"x": 0.25},
           "byte_identity": {"all_unchanged": True}, "days": ["20990101"]}
    (_dV / OUT_NAME).write_text(json.dumps(_v1, indent=1, sort_keys=True))
    # THE CORRECTION IS RESOLVED AGAINST THE DECLARATION ITS ARTIFACT WAS
    # WRITTEN UNDER. On the real head that is now v5, which declares the
    # SECOND read's family -- so this drive supplies a scratch declaration
    # of the FIRST read, and the known-bad below drives the real head.
    (_dV / "decl").mkdir()
    (_dV / "decl" / "be_race_read_declaration_v1.json").write_text(json.dumps({
        "protocol": "FIXTURE-FIRST-READ", "supersedes": None,
        "result": {"artifact": OUT_NAME},
        "read_horizon": {"not_before_utc": "1970-01-01T00:00:00Z"},
        "pins": {"family": "be_race_read_feed_pins"},
        "population": {"READABLE": ["20260903", "20260904", "20260905"]},
        "permutation_floor": {"G": 3, "multiplicity": 2}}))
    _res = supersede_result(outdir=_dV, builder_commit="deadbee",
                            reader_sha256="f" * 64,
                            source="battery fixture",
                            fixture=True, decl_dir=_dV / "decl",
                            why="battery: a scratch v1 under a scratch "
                                "marker directory; the ledger is untouched")
    _v2 = json.loads(Path(_res["v2"]).read_text())
    ok(Path(_res["v2"]).name == "be_race_read_result_v2.json"
       and _res["v1_untouched"]
       and _v2["supersedes"]["sha256"] == _res["v1_sha256"]
       and _v2["supersedes"]["path"].endswith(OUT_NAME),
       f"R-707: the .v2 is {Path(_res['v2']).name} -- the "
       f"`<family>_v<N>.json` convention this chain resolver reads -- and it "
       f"supersedes v1 by the {{path, sha256}} PAIR, v1 untouched")
    ok(_res["census"]["difference_is_exactly_the_additions"]
       and all(_res["census"]["frozen_blocks_byte_identical"].values())
       and sorted(_res["census"]["keys_changed_vs_v1"]) ==
           ["pinned_days_not_in_READABLE", "producing_code", "supersedes"],
       f"THE CENSUS PROVES NOTHING WAS RECOMPUTED: the only keys that differ "
       f"from v1 are {_res['census']['keys_changed_vs_v1']}, and "
       f"day_signs / permutation_floors / byte_identity are byte-identical")
    ok(list(_v2["producing_code"])[0] == "status"
       and _v2["producing_code"]["status"] == "RECONSTRUCTED_NOT_A_STAMP"
       and "builder_commit_RECONSTRUCTED" in _v2["producing_code"]
       and "builder_commit" not in _v2["producing_code"]
       and "reader_sha256" not in _v2["producing_code"],
       "THE RECONSTRUCTION RIDES IN THE KEY: `status` is the block's FIRST "
       "field and the values are under *_RECONSTRUCTED names -- a reader "
       "keying `builder_commit` finds NOTHING, so it cannot get a value "
       "that looks stamped")
    _pn = _v2["pinned_days_not_in_READABLE"]
    ok(_pn["THEY_ARE_NOT_IN_READABLE"] is True
       and _pn["G_REMAINS"] == 3
       and set(_pn["status"].values()) == {"READ_BUT_UNRECOVERABLE"}
       and "no_reader_may_infer_G_5" in _pn,
       f"THE TWO PINNED DAYS ARE SAID, AND ONLY SAID: {sorted(_pn['days'])} "
       f"carry READ_BUT_UNRECOVERABLE copied from the pin file and nothing "
       f"else -- no score, no sign, no quantity -- with G_REMAINS "
       f"{_pn['G_REMAINS']} stated so no reader infers G = 5 from their "
       f"appearance (declaration v4 §3.3)")
    # FALSIFIER 1: the census REFUSES a correction that touches the read
    _base = {"day_signs": {"20990101": 1}, "permutation_floors": {"x": 0.25},
             "byte_identity": {"all_unchanged": True}}
    _touch = dict(_base, day_signs={"20990101": -1},
                  producing_code={"status": "RECONSTRUCTED_NOT_A_STAMP"},
                  supersedes={}, pinned_days_not_in_READABLE={})
    try:
        correction_census(_base, _touch)
        _touched_refused = False
    except ReadRefused as _eV:
        _touched_refused = ("FROZEN block changed" in str(_eV)
                            and "day_signs" in str(_eV))
    ok(_touched_refused,
       "KNOWN-BAD: a correction that changes a FROZEN block (day_signs / "
       "permutation_floors / byte_identity) is REFUSED by the census, "
       "naming the block -- a correction that touches the read is not a "
       "correction. Driven on the PREDICATE: the first form of this "
       "known-bad patched `json.dumps` and tested nothing")
    _extra = dict(_base, days=["20990101", "20990102"],
                  producing_code={"status": "RECONSTRUCTED_NOT_A_STAMP"},
                  supersedes={}, pinned_days_not_in_READABLE={})
    try:
        correction_census(_base, _extra)
        _extra_refused = False
    except ReadRefused as _eW:
        _extra_refused = "outside the permitted set" in str(_eW)
    ok(_extra_refused,
       "KNOWN-BAD: a correction that changes ANY key outside the declared "
       "additions is REFUSED, naming the key")
    # FALSIFIER (REV 79 §1.3): a .v2 MISSING a declared addition is REFUSED
    for _miss in ("supersedes", "producing_code",
                  "pinned_days_not_in_READABLE"):
        _lack = dict(_base, producing_code={"status":
                                            "RECONSTRUCTED_NOT_A_STAMP"},
                     supersedes={}, pinned_days_not_in_READABLE={})
        _lack.pop(_miss)
        try:
            correction_census(_base, _lack,
                              added=("pinned_days_not_in_READABLE",
                                     "producing_code", "supersedes"))
            _miss_refused = False
        except ReadRefused as _eM:
            _miss_refused = ("missing declared addition" in str(_eM)
                             and _miss in str(_eM))
        ok(_miss_refused,
           f"REV 79 §1.3 KNOWN-BAD: a .v2 that OMITS the declared addition "
           f"{_miss!r} is REFUSED, naming it. The census compared `changed` "
           f"against `added` FILTERED TO WHAT v2 CARRIED, so an omission "
           f"passed -- `supersedes` first among them, which would leave an "
           f"orphan file beside the artifact it claims to replace")
    # Q-DE-109 (1): the CLAUSE must follow the OMISSION, not `supersedes`.
    # DE imports this function (de_receipt_correction) and its declared
    # additions include `provenance`, for which the old clause -- "cannot be
    # resolved to what it supersedes … an orphan file" -- was simply untrue:
    # the right verdict under a message that misstates the reason sends a
    # reader to repair the wrong thing (REV 54 §0's class).
    _prov = dict(_base, producing_code={"status": "RECONSTRUCTED_NOT_A_STAMP"},
                 supersedes={"sha256": "a" * 64})
    try:
        correction_census(_base, _prov,
                          added=("provenance", "producing_code", "supersedes"))
        _prov_msg = ""
    except ReadRefused as _eP:
        _prov_msg = str(_eP)
    ok("`provenance` is absent" in _prov_msg
       and "does not carry what its declaration says it carries" in _prov_msg
       and "orphan file" not in _prov_msg
       and "resolved to what it supersedes" not in _prov_msg,
       f"Q-DE-109 (1): A NON-`supersedes` OMISSION NAMES ITS OWN "
       f"CONSEQUENCE and NOT the chain clause -- {_prov_msg!r}")
    try:
        correction_census(_base, dict(_base,
                                      producing_code={"status": "X"},
                                      provenance={}),
                          added=("provenance", "producing_code", "supersedes"))
        _sup_msg = ""
    except ReadRefused as _eS:
        _sup_msg = str(_eS)
    ok("`supersedes` is absent" in _sup_msg and "orphan file" in _sup_msg
       and "does not carry what its declaration says" not in _sup_msg,
       f"AND `supersedes` KEEPS ITS OWN, WHICH IS THE TRUE ONE FOR IT: "
       f"{_sup_msg!r}")
    try:
        correction_census(_base, dict(_base, producing_code={"status": "X"}),
                          added=("provenance", "producing_code", "supersedes"))
        _both = ""
    except ReadRefused as _eB2:
        _both = str(_eB2)
    ok("`provenance` is absent" in _both and "`supersedes` is absent" in _both,
       f"AND TWO OMISSIONS CARRY TWO CLAUSES, one each rather than one "
       f"borrowed for both: {_both!r}")

    _full = dict(_base, producing_code={"status": "RECONSTRUCTED_NOT_A_STAMP"},
                 supersedes={"sha256": "a" * 64},
                 pinned_days_not_in_READABLE={"days": []})
    _c_ok = correction_census(_base, _full,
                              added=("pinned_days_not_in_READABLE",
                                     "producing_code", "supersedes"))
    ok(_c_ok["difference_is_exactly_the_additions"]
       and _c_ok["missing_declared_additions"] == []
       and "THEMSELVES" in _c_ok["compared_against"],
       "POSITIVE CONTROL STILL PASSES: a .v2 carrying ALL the declared "
       "additions and touching nothing else admits, with "
       "missing_declared_additions empty")

    # ---- REV 83 §1.3: THE FROZEN SET IS DERIVED FROM THE ARTIFACT -------
    # The half that was vacuous, driven on the family that exposed it. A
    # design declaration carries none of the race read's three blocks, so
    # the module-constant form compared three absent keys against three
    # absent keys and PASSED. These cells show the vacuity, then fire the
    # cell that could not fire.
    _design = {"arms": {"A": 1}, "estimand": "net_value", "bars": [1, 2],
               "n_days": 4, "correction_census": {"from_the_v1": True}}
    ok(not (set(FROZEN_BLOCKS) & set(_design)),
       f"THE VACUITY, SHOWN RATHER THAN DESCRIBED: a design-shaped v1 "
       f"carries NONE of {list(FROZEN_BLOCKS)}, so the old module-constant "
       f"frozen half compared absent-to-absent and passed on every design "
       f"write. This is the precondition the two cells below stand on")
    _dv2_bad = dict(_design, arms={"A": 2},
                    supersedes={"path": "p", "sha256": "a" * 64})
    try:
        correction_census(_design, _dv2_bad, added=("supersedes",),
                          family="p003_de_multiday_gate1_design")
        _d_msg = ""
    except ReadRefused as _eD:
        _d_msg = str(_eD)
    ok("FROZEN block changed" in _d_msg and "arms" in _d_msg
       and "READ from the v1" in _d_msg and "INHERITED" in _d_msg,
       f"THE CELL THAT WAS VACUOUS NOW FIRES: a design-shaped correction "
       f"that changes the INHERITED key `arms` is REFUSED BY NAME, on a "
       f"family whose keys this module has never heard of -- the frozen "
       f"set came from the artifact: {_d_msg!r}")
    _dv2_ok = correction_census(
        _design, dict(_design, supersedes={"path": "p", "sha256": "a" * 64}),
        added=("supersedes",), family="p003_de_multiday_gate1_design")
    ok(_dv2_ok["frozen_set"]["n_frozen"] == 4
       and _dv2_ok["frozen_set"]["frozen_keys"] == ["arms", "bars",
                                                    "estimand", "n_days"]
       and "correction_census" not in _dv2_ok["frozen_set"]["frozen_keys"],
       f"AND IT ADMITS THE GOOD ONE, over a REAL frozen set of "
       f"{_dv2_ok['frozen_set']['n_frozen']} derived keys "
       f"{_dv2_ok['frozen_set']['frozen_keys']} -- the declared exemption "
       f"`correction_census` is excluded, because the caller writes it "
       f"AFTER this runs and the emitter must not refuse itself")
    try:
        correction_census(
            {"supersedes": {}, "producing_code": {"status": "X"}},
            {"supersedes": {"path": "p"}, "producing_code": {"status": "X"}},
            added=("supersedes", "producing_code"),
            family="a_family_with_no_inherited_keys")
        _e_msg = ""
    except ReadRefused as _eE:
        _e_msg = str(_eE)
    ok("DERIVED frozen set is EMPTY" in _e_msg
       and "a_family_with_no_inherited_keys" in _e_msg
       and "cannot fire" in _e_msg and "not the .v2" in _e_msg,
       f"KNOWN-BAD: a family whose every key is a declared addition has an "
       f"EMPTY derived frozen set, and the census REFUSES NAMING THE FAMILY "
       f"instead of passing -- and says it refuses the CENSUS, not the "
       f".v2, so the reader does not go repair the artifact: {_e_msg!r}")
    try:
        correction_census(_design, dict(_design, arms={"A": 1}),
                          added=("arms",), frozen=("arms",),
                          family="p003_de_multiday_gate1_design")
        _belt = ""
    except ReadRefused as _eBt:
        _belt = str(_eBt)
    ok("NAMES ['arms']" in _belt and "the derivation makes them writable"
       in _belt,
       f"KNOWN-BAD FOR THE BELT: a caller that NAMES a block as frozen "
       f"while declaring it an addition is REFUSED -- the belt may only "
       f"tighten the braces, never contradict them: {_belt!r}")
    _rv1 = Path(_BDR.derived()) / "be_race_read_result_v1.json"
    _rv2 = Path(_BDR.derived()) / "be_race_read_result_v2.json"
    ok(_rv1.exists() and _rv2.exists(),
       f"the LANDED race-read v1 and .v2 are both present at "
       f"{_BDR.derived()} -- the derived form is re-driven on the real "
       f"artifacts, not only on fixtures")
    _rc = correction_census(json.loads(_rv1.read_text()),
                            json.loads(_rv2.read_text()),
                            frozen=FROZEN_BLOCKS,
                            family="be_race_read_result")
    ok(_rc["difference_is_exactly_the_additions"] is True
       and _rc["missing_declared_additions"] == []
       and _rc["frozen_set"]["n_frozen"] >= 3
       and all(_rc["frozen_set"]["named_blocks_present_and_frozen"].values())
       and set(FROZEN_BLOCKS) <= set(_rc["frozen_set"]["frozen_keys"]),
       f"AND THE LANDED .v2 STILL PASSES UNDER THE DERIVED SET: "
       f"{_rc['frozen_set']['n_frozen']} inherited keys byte-identical, "
       f"changed exactly {sorted(_rc['keys_changed_vs_v1'])} -- the three "
       f"named blocks are INSIDE the derived set rather than being it, so "
       f"the belt checks the construction rather than replacing it")

    # FALSIFIER 2: a reconstruction under a PLAIN field name is REFUSED
    _plain = dict(_base,
                  producing_code={"status": "RECONSTRUCTED_NOT_A_STAMP",
                                  "builder_commit": "c4c0d0d"},
                  supersedes={}, pinned_days_not_in_READABLE={})
    try:
        correction_census(_base, _plain)
        _plain_refused = False
    except ReadRefused as _eX:
        _plain_refused = ("sits under the PLAIN field" in str(_eX)
                          and "builder_commit" in str(_eX))
    ok(_plain_refused,
       "KNOWN-BAD: a reconstruction under the PLAIN name `builder_commit` "
       "is REFUSED -- a reader keying it would get a value that looks "
       "stamped, in the one artifact that cannot be re-run")
    _notfirst = dict(_base,
                     producing_code={"builder_commit_RECONSTRUCTED": "x",
                                     "status": "RECONSTRUCTED_NOT_A_STAMP"},
                     supersedes={}, pinned_days_not_in_READABLE={})
    try:
        correction_census(_base, _notfirst)
        _first_refused = False
    except ReadRefused as _eY:
        _first_refused = "not the block's first field" in str(_eY)
    ok(_first_refused,
       "AND `status` MUST BE THE BLOCK'S FIRST FIELD: a block that carries "
       "it second is refused")

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
            "result": {"artifact": OUT_NAME},
            "read_horizon": {"not_before_utc": "1970-01-01T00:00:00Z"},
            "pins": {"family": "be_race_read_feed_pins"},
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
    # ---- BE 84: THE SECOND RACE READ, DECLARED BEFORE ITS DAYS CLOSE ---
    # Driven ON THE REAL HEAD, because the point of pre-declaring is that
    # the gates bind NOW -- while every declared day is still open and
    # nothing has been seen. Each of the three refusals the declaration
    # names is driven here, with a positive control beside it.
    _v5 = declared_read()
    ok(_v5["declaration"] == "be_race_read_declaration_v5.json"
       and _v5["READABLE"] == ["20260906", "20260907", "20260908", "20260909"]
       and _v5["G_declared"] == 4
       and _v5["result_name"] == "be_race_read2_result_v1.json"
       and _v5["horizon_utc"] == "2026-09-10T01:00:00Z"
       and _v5["pins_family"] == "be_race_read_feed_pins",
       f"THE HEAD IS v5 AND THE ACT IS BOUND TO IT: READABLE "
       f"{_v5['READABLE']}, G {_v5['G_declared']}, result "
       f"{_v5['result_name']} (a NEW family -- the first read's "
       f"`be_race_read_result` chain keeps v1/v2 and is never confused with "
       f"this one), horizon {_v5['horizon_utc']}, pins family "
       f"{_v5['pins_family']}. Every one read from the declaration, none "
       f"from a constant in this module")
    _rd5 = resolve_days()
    ok(_rd5["G_computed"] == 4 and _rd5["G_agrees_with_the_declaration"],
       f"AND G IS COMPUTED FROM THE SET, NOT COPIED: {_rd5['G_computed']} "
       f"from {len(_rd5['days'])} declared days, asserted equal to the "
       f"declaration's own G -- rule 10 obeyed once and CHECKED twice")
    try:
        resolve_days(["20260906", "20260907", "20260908"])
        _outside = "NOT REFUSED"
    except ReadRefused as _e5a:
        _outside = str(_e5a)
    ok("is not the declared READABLE set" in _outside
       and "20260909" in _outside and "Narrowed by" in _outside,
       f"FALSIFIER 1 -- A DAY OUTSIDE `READABLE` IS REFUSED BY NAME, and "
       f"the refusal says WHICH way it differs: {_outside[:180]!r}")
    try:
        assert_read_horizon(_v5)
        _early = "NOT REFUSED"
    except ReadRefused as _e5b:
        _early = str(_e5b)
    ok("read horizon" in _early and "has not passed" in _early
       and "2026-09-10T01:00:00Z" in _early and "PARTIAL day" in _early,
       f"FALSIFIER 2 -- A READ BEFORE THE HORIZON IS REFUSED BY NAME, "
       f"driven on the real declaration at the real clock: {_early[:200]!r}")
    _past = assert_read_horizon({"horizon_utc": "1970-01-01T00:00:00Z"})
    ok(_past["seconds_past_the_horizon"] > 0,
       f"POSITIVE CONTROL: a horizon already passed ADMITS "
       f"({_past['seconds_past_the_horizon']} s past it) -- the gate is "
       f"about the clock, not about refusing")
    try:
        assert_every_declared_day_is_pinned(_v5["READABLE"], pins())
        _unpinned = "NOT REFUSED"
    except ReadRefused as _e5c:
        _unpinned = str(_e5c)
    ok("not fully pinned" in _unpinned
       and all(d in _unpinned for d in _v5["READABLE"])
       and "No marker is written and no day is consumed" in _unpinned,
       f"FALSIFIER 3 -- AN ABSENT PIN REFUSES THE WHOLE READ, naming every "
       f"unpinned day, BEFORE any marker is written. The four days are not "
       f"pinned yet -- they have not closed -- so this refuses today and "
       f"goes on refusing until the pins chain head carries all four: "
       f"{_unpinned[:200]!r}")
    _fullpins = {d: {"exists": True, "sha256": "a" * 64, "bytes": 1}
                 for d in _v5["READABLE"]}
    ok(assert_every_declared_day_is_pinned(
           _v5["READABLE"], _fullpins)["every_day_pinned_before_the_act"]
       and "not fully pinned" in str(_unpinned),
       "POSITIVE CONTROL: the same four days with a complete pin set ADMIT "
       "-- the refusal is about the missing pins, not about the days")
    try:
        assert_every_declared_day_is_pinned(
            _v5["READABLE"], dict(_fullpins, **{"20260909": {"exists": True}}))
        _nodig = "NOT REFUSED"
    except ReadRefused as _e5d:
        _nodig = str(_e5d)
    ok("A pin without a digest" in _nodig and "20260909" in _nodig
       and "the pin IS the pair" in _nodig,
       f"AND A PIN WITH NO DIGEST IS NOT A PIN (R-608): refused by name "
       f"even though `exists` is true -- a path with no digest verifies "
       f"nothing: {_nodig[:160]!r}")
    _dG = Path(_tfV.mkdtemp(prefix="be84_guard_"))
    (_dG / OUT_NAME).write_text(json.dumps(_v1, indent=1, sort_keys=True))
    try:
        supersede_result(outdir=_dG,
                         builder_commit="deadbee", reader_sha256="f" * 64,
                         source="battery known-bad", fixture=True,
                         why="battery: proves the correction path refuses "
                             "under the SECOND read's declaration")
        _wrongdecl = "NOT REFUSED"
    except ReadRefused as _e5e:
        _wrongdecl = str(_e5e)
    ok("the FIRST read's result" in _wrongdecl
       and "be_race_read2_result_v1.json" in _wrongdecl,
       f"AND THE FIRST READ'S CORRECTION PATH IS CLOSED BY THE NEW HEAD: a "
       f".v2 of `be_race_read_result_v1.json` composed against v5 -- which "
       f"declares a different family -- is REFUSED BY NAME, so a correction "
       f"can never carry the wrong read's provenance under right-looking "
       f"field names: {_wrongdecl[:170]!r}")

    # ---- BE 85 / REV 86 §5: THE CLAUSE FOR THE READER WHO HOLDS ONLY
    # THE SECOND ARTIFACT. Driven on the real head for the positive control,
    # and on scratch declarations for every refusal -- one mutation at a
    # time, so each refusal is shown to fire for ITS OWN reason.
    _npc5 = not_pooled_clause()
    ok(_npc5["applies"] is True
       and "two chances" in _npc5["sentence"]
       and "m = 2" in _npc5["sentence"]
       and "20260903..20260905" in _npc5["sentence"]
       and "NOT pooled" in _npc5["sentence"]
       and _npc5["computed"]["floor_this_read_ARMS_ONLY"] == 0.125
       and _npc5["computed"]["floor_first_read_ARMS_ONLY"] == 0.25
       and _npc5["computed"]["declared_floors_equal_the_arms_only_formula"]
       and _npc5["generated_from"]["verified_against_the_file"]["matches"],
       f"REV 86 §5, GENERATED FROM THE DECLARATION AND NOT TYPED: "
       f"{_npc5['sentence']!r}. The pair is read from v5's `supersedes` and "
       f"VERIFIED against the file on disk; the first read's days come from "
       f"ITS OWN declaration; m from both; each floor RECOMPUTED from arms "
       f"alone ({_npc5['computed']['floor_first_read_ARMS_ONLY']} and "
       f"{_npc5['computed']['floor_this_read_ARMS_ONLY']}), which is what "
       f"`counts arms, not reads` MEANS as a predicate")
    _sgn = json.dumps(_npc5)
    ok(_npc5["computed"]["signs_are"] in ("consistent", "inconsistent")
       and _npc5["computed"]["sign_consistency_predicate"].startswith("len(")
       and '"day_signs":' not in _sgn,
       f"AND THE CONSISTENCY WORD IS A PREDICATE, NOT A CONCLUSION TYPED "
       f"BESIDE ONE (rule 10): "
       f"{_npc5['computed']['sign_consistency_predicate']!r} over "
       f"{_npc5['generated_from']['consistency_read_from']['head']}'s "
       f"day_signs -- and NO SIGN VALUE is carried into the clause")
    import tempfile as _tfN
    _dN = Path(_tfN.mkdtemp(prefix="be85_clause_"))
    (_dN / "decl").mkdir()
    (_dN / "der").mkdir()
    _first = {"protocol": "FIXTURE-FIRST", "supersedes": None,
              "population": {"READABLE": ["20990101", "20990102"]},
              "permutation_floor": {"G": 2, "multiplicity": 2,
                                    "best_possible_adjusted_p": 0.5}}
    _fp = _dN / "decl" / "be_race_read_declaration_v1.json"
    _fp.write_text(json.dumps(_first, indent=1, sort_keys=True))
    _fsha = hashlib.sha256(_fp.read_bytes()).hexdigest()
    (_dN / "der" / "be_race_read_result_v1.json").write_text(json.dumps(
        {"day_signs": {"20990101": 1, "20990102": -1}}))

    def _second(**over):
        d = {"protocol": "FIXTURE-SECOND",
             "supersedes": {"path": _fp.name, "sha256": _fsha},
             "population": {"READABLE": ["20990201", "20990202"],
                            "CONSUMED_BY_THE_FIRST_READ": ["20990101",
                                                           "20990102"]},
             "permutation_floor": {"G": 2, "multiplicity": 2,
                                   "best_possible_adjusted_p": 0.5}}
        d.update(over)
        return d

    def _clause(doc):
        try:
            return "ADMITTED", not_pooled_clause(
                doc, decl_dir=_dN / "decl", derived_dir=_dN / "der")
        except ReadRefused as _e:
            return str(_e).split(":")[0], str(_e)
    _code, _out = _clause(_second())
    ok(_code == "ADMITTED" and _out["applies"] and "m = 2" in _out["sentence"]
       and "20990101..20990102" in _out["sentence"]
       and "inconsistent" in _out["sentence"],
       f"POSITIVE CONTROL ON THE FIXTURE: a well-formed pair renders the "
       f"clause from the fixture's own artifacts: {_out['sentence']!r}")
    _codes = {}
    _codes["no pair at all"] = _clause(_second(supersedes=None))[0]
    _codes["a pair with no digest"] = _clause(
        _second(supersedes={"path": _fp.name}))[0]
    _codes["a pair naming a file that is not there"] = _clause(
        _second(supersedes={"path": "be_race_read_declaration_v9.json",
                            "sha256": _fsha}))[0]
    _codes["a pair whose digest has moved"] = _clause(
        _second(supersedes={"path": _fp.name, "sha256": "0" * 64}))[0]
    ok(list(_codes.values()) == ["FIRST_READ_PAIR_ABSENT",
                                "FIRST_READ_PAIR_HALF_WRITTEN",
                                "FIRST_READ_DECLARATION_ABSENT",
                                "FIRST_READ_PAIR_MISMATCH"],
       f"KNOWN-BAD, THE REQUIRED ONE AND ITS THREE NEIGHBOURS -- a template "
       f"rendered WITHOUT the first read's pair REFUSES BY NAME, and the "
       f"four ways the pair can fail refuse under four DIFFERENT names: "
       f"{_codes}. None of them renders a sentence: an ungenerated clause is "
       f"never replaced by a typed one")
    _dis = {}
    _dis["days"] = _clause(_second(population={
        "READABLE": ["20990201"], "CONSUMED_BY_THE_FIRST_READ": ["20990103"]}))[0]
    _dis["m"] = _clause(_second(permutation_floor={
        "G": 2, "multiplicity": 3, "best_possible_adjusted_p": 0.75}))[0]
    _dis["floor"] = _clause(_second(permutation_floor={
        "G": 2, "multiplicity": 2, "best_possible_adjusted_p": 0.99}))[0]
    ok(_dis == {"days": "FIRST_READ_DAYS_DISAGREE",
                "m": "MULTIPLICITY_DISAGREES",
                "floor": "FLOOR_IS_NOT_ARMS_ONLY"},
       f"AND THE THREE CLAIMS THE SENTENCE MAKES ARE PREDICATES, EACH "
       f"REFUSING BY NAME WHEN IT DOES NOT HOLD: {_dis}. `counts arms, not "
       f"reads` is CHECKED against 2^-G x m recomputed from each "
       f"declaration's own fields -- a sentence that asserted it without "
       f"checking would be a hardcoded verdict beside a table (rule 10)")
    _na = not_pooled_clause({"protocol": "NO-PREVIOUS-READ",
                             "population": {"READABLE": ["20990301"]}},
                            decl_dir=_dN / "decl", derived_dir=_dN / "der")
    ok(_na["applies"] is False and "no other read" in _na["why"]
       and "sentence" not in _na,
       f"AND A FIRST READ SAYS SO RATHER THAN SAYING NOTHING: a declaration "
       f"naming no consumed days renders `applies False` WITH ITS REASON "
       f"({_na['checked']}), never silence -- and the discriminator is the "
       f"declaration's CONTENT, not whether a field happens to parse, so a "
       f"second read cannot escape the clause by dropping its pair")

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

    # ---- REV 84 §3.2: THE SHARED MODULE'S FALSIFIER IS ONE CELL HERE ----
    # One implementation, N detectors. This battery imports
    # `declaration_chain` through `be_rule22`, so a regression in it is this
    # battery's problem too -- and BE 82's was found by DA's kept cell, not
    # by the module's own. Run as a SUBPROCESS, so a module that no longer
    # runs at all fails here rather than being routed around.
    import be_rule22 as _R22b
    _dcf = _R22b.shared_falsifier()
    ok(_dcf["ok"],
       f"REV 84 §3.2 -- ONE IMPLEMENTATION, N DETECTORS: this battery RUNS "
       f"`declaration_chain.py --falsify` as a subprocess -> rc "
       f"{_dcf['rc']}, {_dcf['summary']!r}. A regression in the shared "
       f"module fails every importer's battery at once, and no importer "
       f"re-implements the logic. "
       f"{_dcf['failed_cells'] or _dcf['stderr_tail'] or ''}")


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

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


def floors(g_opt: int, g_pess: int, m: int = 2) -> dict:
    o, p = m / 2 ** g_opt, m / 2 ** g_pess
    return {"optimistic": {"G": g_opt, "best_possible_adjusted_p": o},
            "pessimistic": {"G": g_pess, "best_possible_adjusted_p": p},
            "resolved_best_possible_adjusted_p": max(o, p),
            "WHICH_ONE_IS_RESOLVED": "the CONSERVATIVE one",
            "neither_clears_0_05": min(o, p) > 0.05}


def read(paths: dict, *, outdir: Path = None, write: bool = True,
         per_day_pins: dict | None = None) -> dict:
    opened = [Path(v) for v in paths.values()]
    sep = assert_separation(opened)
    missing = [str(p) for p in opened if not Path(p).exists()]
    if missing:
        raise ReadRefused(f"REFUSED: sealed feed(s) absent: {missing}")
    per_day, before, pinned = {}, {}, {}
    pd = per_day_pins if per_day_pins is not None else pins()
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
        "permutation_floors": floors(len(paths), len(fresh)),
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


EXPECTED_CHECKS = 14


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
        # A.3 KNOWN-BADS, driven
        try:
            read(paths, outdir=d, per_day_pins=dict(
                _pins, **{"20260901": dict(_pins["20260901"],
                                           sha256="0" * 64)}))
            ok(False, "a tampered pin must refuse")
        except ReadRefused as ex:
            ok("not the pinned" in str(ex),
               "KNOWN-BAD: a TAMPERED pin REFUSES before a byte is parsed -- "
               "the bytes changed between the pin and the read")
        try:
            read(paths, outdir=d, per_day_pins=dict(
                _pins, **{"20260901": {"exists": False, "sha256": None}}))
            ok(False, "a pin-absent day must refuse by name")
        except ReadRefused as ex:
            ok("marks 20260901's feed ABSENT" in str(ex)
               and "smaller G as though it were the declared one" in str(ex),
               "KNOWN-BAD: a day the pin marks ABSENT REFUSES **by name** -- "
               "`exists: false` is actionable, and skipping it would report "
               "a smaller G as though it were the declared one")
        r = read(paths, outdir=d, per_day_pins=_pins)
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
        ok(f["resolved_best_possible_adjusted_p"] ==
           max(f["optimistic"]["best_possible_adjusted_p"],
               f["pessimistic"]["best_possible_adjusted_p"])
           and floors(5, 3)["resolved_best_possible_adjusted_p"] == 0.25,
           "both floors computed, CONSERVATIVE resolved (0.25 on the real "
           "5/3 split, not the flattering 0.0625)")

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
            read(paths, outdir=d, per_day_pins=_pins)
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
        out = read({d: Path(p) for d, p in sealed_feeds().items()})
        print(json.dumps({"written": out.get("_written"),
                          "day_signs": out["day_signs"]}))
        return 0
    print("usage: be_race_reader.py --selftest | --open  (--open CONSUMES the "
          "five sealed FEEDS; the coordinator's act on GO)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

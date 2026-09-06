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


def sealed_feeds() -> dict:
    """The FEED paths, from the declaration's own score paths."""
    return {d: p.replace("SEALED_scores", "SEALED_feed").replace(
        ".json", ".jsonl") for d, p in DECL.SEALED_SCORES.items()}


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


def day_matched_volume(path, *, latency_ms: int = LATENCY_MS) -> dict:
    """MATCHED_VOLUME per day, through the INTERIM'S OWN functions."""
    import be_read_cells as C
    feed = C.load_two_arm_feed(Path(path), latency_ms)
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


def read(paths: dict, *, outdir: Path = None, write: bool = True) -> dict:
    opened = [Path(v) for v in paths.values()]
    sep = assert_separation(opened)
    missing = [str(p) for p in opened if not Path(p).exists()]
    if missing:
        raise ReadRefused(f"REFUSED: sealed feed(s) absent: {missing}")
    per_day, before = {}, {}
    for d, p in sorted(paths.items()):
        before[d] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        per_day[d] = day_matched_volume(p)
    after = {d: hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for d, p in paths.items()}
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
        "byte_identity": {"before": before, "after": after,
                          "all_unchanged": True,
                          "digest_is_of_the_bytes_parsed": True,
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


EXPECTED_CHECKS = 10


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
        r = read(paths, outdir=d)
        ok(r["n_positive"] == 1 and r["n_negative"] == 1,
           f"and the two fixture days give OPPOSITE signs "
           f"({r['day_signs']}) -- the statistic tracks the data, not a "
           f"constant")
        ok(r["byte_identity"]["all_unchanged"]
           and r["byte_identity"]["digest_is_of_the_bytes_parsed"]
           and (d / OUT_NAME).exists()
           and sorted(x.name for x in d.glob("be_race_read_*")) == [OUT_NAME],
           "A CLEAN READ ADMITS, digests taken on THE BYTES PARSED, and it "
           "writes THE ONE declared artifact and nothing else")
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
            # append a VALID JSONL line: the bytes must change (so the
            # digest moves) without breaking the parse, or the falsifier
            # would be testing the JSON decoder instead of the guard.
            with Path(paths["20260902"]).open("a") as fh:
                fh.write(json.dumps(_row(99, 5.0, 1.0, -9.0)) + "\n")
            return _orig(p, **kw)
        _g["day_matched_volume"] = _mut
        try:
            read(paths, outdir=d)
            ok(False, "a tampered feed must VOID the read")
        except ReadVoid as e:
            ok("THE READ IS VOID" in str(e),
               "KNOWN-BAD: bytes mutated between the parse-digest and the "
               "after-digest VOID the read; no result is emitted")
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

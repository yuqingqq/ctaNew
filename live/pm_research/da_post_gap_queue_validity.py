"""Post-gap queue validity: are rows admitted before their book was re-established?

SURFACE AUTHORISATION (R-126, in-file): COMMISSIONED by R-173(2), non-blocking.
DESIGN, METRIC, NULL and this instrument's own falsifier were DECLARED IN
Q-DA-79 BEFORE ANY DATA WAS READ. Nothing here was chosen after seeing a
number, and the pre-committed reading of a large result is restated below so it
cannot be quietly revised upward later.

THE QUESTION. R-173 ruled a PM gap excludes ROWS, not the window. Rows INSIDE
the gap are handled (`GAP_IN_HORIZON`). This asks about rows AFTER it.
Queue-ahead reconstruction needs book state, and `BookState.apply("price", ...)`
refuses to mutate until `ready` -- its own comment says "require a full post-gap
snapshot before queue inference". So between a gap ending and the next FULL book
snapshot arriving, `qahead` may rest on a book that was never re-established,
while the row still reports `status = OK`. The status field is about the FILL
HORIZON; it says nothing about queue validity, which is exactly why this can
hide.

METRIC (declared): for a row after a gap,
    resync_lag_s = t_start - (receipt time of the first FULL book snapshot
                              at or after that gap's end)
EXPOSED = rows with `resync_lag_s < 0`: after the gap, before the book returned.

NULL (declared): H0 = the exposed set is empty. This is an EXISTENCE claim, so
it needs no permutation null -- a single OK row with resync_lag_s < 0 falsifies
it. ZERO IS A REAL ANSWER and ends the concern; it is not a failed measurement.

PRE-COMMITTED READING OF A LARGE RESULT (Q-DA-79(5), restated verbatim in
spirit): Phase 2's gate is a PAIRED comparison on identical rows and all arms
see the same post-gap rows, so any unreliability is COMMON-MODE. A large
exposed set QUALIFIES WHAT THE NUMBERS MEAN -- the population is dirtier than
its status field admits -- and does NOT invalidate the ranking between arms.
Never a retroactive edit to a built population (R-173(2)).

    python3 live/pm_research/da_post_gap_queue_validity.py --selftest
    python3 live/pm_research/da_post_gap_queue_validity.py run
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_intensity as fi
import flow_fill_development as fd
import da_topup_population_verify as V

REPO = Path(__file__).resolve().parents[2]
BUILT = REPO / "data/pm_5min/derived/harmful_exposure_rows_v3_topup.json"
OUT = REPO / "data/pm_5min/derived/da_post_gap_queue_validity_v1.json"


def book_snapshot_times(path: Path, up_id: str, ws: int) -> list[float]:
    """Receipt times of FULL book snapshots, window-relative.

    Same predicate the exposure builder uses to decide a snapshot is a snapshot
    (`et == "book"` or a payload carrying both sides, for the up token), so this
    cannot disagree with it about what re-establishes a book.
    """
    out: list[float] = []
    for line in fi._gz_lines(path):
        if fd.BOOK_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("asset_id")) != up_id:
                continue
            if msg.get("event_type") == "book" or (
                    "bids" in msg and "asks" in msg):
                if fd._parse_book(msg):
                    out.append(recv)
    out.sort()
    return out


def classify_rows(rows: list[dict[str, Any]],
                  gaps: list[tuple[float, float]],
                  snaps: list[float]) -> dict[str, Any]:
    """Exposed / unexposed split for ONE window. Pure, so it is testable.

    A row is EXPOSED iff it starts after some gap's end but before the first
    full snapshot at or after that gap end. Rows inside a gap are neither --
    they are the gap's own business and R-173 already handles them.
    """
    exposed, unexposed, in_gap = [], [], []
    for r in rows:
        t = float(r["t_start"])
        if any(g0 <= t <= g1 for g0, g1 in gaps):
            in_gap.append(r)
            continue
        prior = [g1 for _, g1 in gaps if g1 < t]
        if not prior:
            unexposed.append(r)
            continue
        gap_end = max(prior)
        i = bisect.bisect_left(snaps, gap_end)
        first_snap = snaps[i] if i < len(snaps) else None
        if first_snap is None or t < first_snap:
            # no snapshot ever returned, or this row precedes it
            r = {**r, "resync_lag_s": (None if first_snap is None
                                       else round(t - first_snap, 6)),
                 "gap_end": round(gap_end, 6),
                 "no_snapshot_after_gap": first_snap is None}
            exposed.append(r)
        else:
            unexposed.append(r)
    return {"exposed": exposed, "unexposed": unexposed, "in_gap": in_gap}


def run() -> dict[str, Any]:
    receipt = json.loads(V.latest_receipt().read_text(encoding="utf-8"))
    gapped = {r["slug"]: [(a, b) for a, b in r["pm_gap_intervals"]]
              for r in receipt["slugs"]
              if r["status"] == "OK" and r.get("pm_gap_s", 0) > 0}
    print(f"gapped OK slugs in {receipt['receipt_version']}: {len(gapped)}",
          flush=True)

    per_slug: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    n_rows_total = 0
    for r in V.iter_rows(BUILT):
        n_rows_total += 1
        s = r.get("slug")
        if s in gapped:
            per_slug[s].append({"t_start": r.get("t_start"),
                                "status": r.get("status"),
                                "qahead": r.get("qahead"),
                                "coin": r.get("coin")})
    print(f"streamed {n_rows_total} rows; {sum(map(len, per_slug.values()))} "
          f"in gapped slugs", flush=True)

    paths = fi._archive_paths()
    tokens = fi.token_map()
    tot = collections.Counter()
    by_coin = collections.defaultdict(collections.Counter)
    lags: list[float] = []
    no_snap_slugs: list[str] = []
    qa_exp, qa_unexp = [], []
    for i, (slug, gaps) in enumerate(sorted(gapped.items())):
        rows = per_slug.get(slug, [])
        if not rows:
            tot["slugs_with_no_rows"] += 1
            continue
        ws = int(slug.rsplit("-", 1)[1])
        up, _dn = tokens[slug]
        snaps = book_snapshot_times(paths[slug], up, ws)
        res = classify_rows(rows, gaps, snaps)
        coin = rows[0].get("coin")
        for k in ("exposed", "unexposed", "in_gap"):
            tot[k] += len(res[k])
            by_coin[coin][k] += len(res[k])
        for r in res["exposed"]:
            if r["status"] == "OK":
                tot["exposed_OK"] += 1
                by_coin[coin]["exposed_OK"] += 1
            if r.get("no_snapshot_after_gap"):
                tot["exposed_no_snapshot"] += 1
                if slug not in no_snap_slugs:
                    no_snap_slugs.append(slug)
            elif r.get("resync_lag_s") is not None:
                lags.append(r["resync_lag_s"])
            if r.get("qahead") is not None:
                qa_exp.append(float(r["qahead"]))
        for r in res["unexposed"]:
            if r["status"] == "OK" and r.get("qahead") is not None:
                qa_unexp.append(float(r["qahead"]))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(gapped)} slugs", flush=True)

    def q(v, p):
        if not v:
            return None
        v = sorted(v)
        return round(v[min(len(v) - 1, int(len(v) * p))], 4)

    ok_in_gapped = tot["exposed"] + tot["unexposed"] + tot["in_gap"]
    return {
        "instrument": "da_post_gap_queue_validity_v1",
        "authorised_by": "R-173(2), design declared in Q-DA-79 before any read",
        "receipt": receipt["receipt_version"],
        "built": str(BUILT),
        "n_and_as_of": {
            "n_rows_in_dataset": n_rows_total,
            "n_gapped_ok_slugs": len(gapped),
            "n_rows_in_gapped_slugs": ok_in_gapped,
        },
        "counts": dict(tot),
        "by_coin": {c: dict(v) for c, v in by_coin.items()},
        "H0_exposed_set_is_empty": tot["exposed"] == 0,
        "H0_falsified": tot["exposed"] > 0,
        "exposed_OK_share_of_gapped_rows": (
            round(tot["exposed_OK"] / ok_in_gapped, 6) if ok_in_gapped else None),
        "resync_lag_s": {"n": len(lags), "p50": q(lags, 0.5),
                         "p90": q(lags, 0.9), "min": q(lags, 0.0),
                         "max": q(lags, 1.0)},
        "slugs_with_no_snapshot_after_gap": no_snap_slugs[:20],
        "qahead_exposed": {"n": len(qa_exp), "p50": q(qa_exp, 0.5),
                           "p90": q(qa_exp, 0.9)},
        "qahead_unexposed": {"n": len(qa_unexp), "p50": q(qa_unexp, 0.5),
                             "p90": q(qa_unexp, 0.9)},
        "reading": (
            "PRE-COMMITTED in Q-DA-79 BEFORE this number existed: Phase 2's "
            "gate is a paired comparison on identical rows and all arms see "
            "the same post-gap rows, so any unreliability is COMMON-MODE. A "
            "large exposed set qualifies what the numbers MEAN and does not "
            "invalidate the ranking between arms. Not a retroactive edit."),
    }


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    def R(t, st="OK", qa=1.0):
        return {"t_start": t, "status": st, "qahead": qa, "coin": "btc"}

    # FALSIFIER 1 -- a hand-computed case the instrument must reproduce.
    # gap [10, 20]; first full snapshot at 25. Rows at 22 and 24 are AFTER the
    # gap but BEFORE the book returned => exposed. 26 and 30 => unexposed.
    # 15 is inside the gap => neither.
    gaps = [(10.0, 20.0)]
    snaps = [5.0, 25.0, 40.0]
    res = classify_rows([R(2), R(15), R(22), R(24), R(26), R(30)], gaps, snaps)
    ok(len(res["exposed"]) == 2, "the two post-gap pre-snapshot rows are exposed")
    ok(len(res["in_gap"]) == 1, "the in-gap row is neither exposed nor clean")
    ok(len(res["unexposed"]) == 3, "pre-gap and post-snapshot rows are clean")
    ok([round(r["resync_lag_s"], 3) for r in res["exposed"]] == [-3.0, -1.0],
       "resync_lag_s is the hand-computed negative offset from the snapshot")

    # FALSIFIER 2 -- NO gap must yield ZERO exposed, or a zero from the real
    # run would prove nothing (rule 15).
    res0 = classify_rows([R(2), R(22), R(30)], [], snaps)
    ok(len(res0["exposed"]) == 0 and len(res0["unexposed"]) == 3,
       "a window with no gap has an EMPTY exposed set")
    ok(res["exposed"] != res0["exposed"],
       "the two inputs get DIFFERENT answers -- the rule reads the gap (R-42)")

    # boundary exactness: a row exactly AT the snapshot is NOT exposed.
    edge = classify_rows([R(25.0)], gaps, snaps)
    ok(len(edge["exposed"]) == 0,
       "a row exactly at the resync instant is clean, not exposed")
    edge2 = classify_rows([R(24.999)], gaps, snaps)
    ok(len(edge2["exposed"]) == 1, "one millisecond earlier IS exposed")
    # a row exactly at a gap edge counts as in-gap, not exposed
    ok(len(classify_rows([R(20.0)], gaps, snaps)["in_gap"]) == 1,
       "a row exactly at the gap end is in-gap, not exposed")

    # NO snapshot ever after the gap -> exposed, flagged, lag is None not 0.0
    none_after = classify_rows([R(22)], gaps, [5.0])
    ok(len(none_after["exposed"]) == 1
       and none_after["exposed"][0]["no_snapshot_after_gap"] is True
       and none_after["exposed"][0]["resync_lag_s"] is None,
       "no snapshot after the gap is EXPOSED and flagged, lag None not zero")

    # multiple gaps: the row is judged against the NEAREST PRIOR gap
    multi = classify_rows([R(38)], [(10.0, 20.0), (30.0, 35.0)], [25.0, 45.0])
    ok(len(multi["exposed"]) == 1 and multi["exposed"][0]["gap_end"] == 35.0,
       "a row after two gaps is judged against the nearest prior one")

    print(f"da_post_gap_queue_validity selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    rep = run()
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({k: v for k, v in rep.items() if k != "reading"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Count GAP_AT_CUTOFF rows under the R-191 definition. Independent of BE.

SURFACE AUTHORISATION (R-126, in-file): R-191(3) orders BOTH seats to
implement the ruled definition INDEPENDENTLY and exchange counts; agreement is
required before the rebuild.

THE DEFINITION, transcribed from R-191(2) and implemented from that text alone
(this file does not consult BE's implementation):

    A row is GAP_AT_CUTOFF iff its ABSOLUTE decision instant
        T = t0 + t_start
    lies in  [g_start, g_end)  -- LOWER-INCLUSIVE, UPPER-EXCLUSIVE --
    for ANY recorded gap of THAT COIN in the collector-gaps ledger.

    Basis: ABSOLUTE on both sides.
    Gap scope: COIN-LEVEL, never per-slug. A gap is a FEED event; scoping it to
      the slug it was logged against is a lossy projection that drops warm-up
      and boundary overlaps -- rows with NEGATIVE t_start sit in windows whose
      per-slug gap list cannot see a gap logged against the neighbouring
      window. That projection is exactly what DA's earlier cross did, which is
      why its 152 is expected to move.
    Universe: ALL tape rows.  Unit: ROWS.  Score-side reported beside.

ONE AMBIGUITY THE RULING DOES NOT SETTLE, so it is reported rather than
decided: the ledger carries `collector_version` eras. "ANY recorded gap" is
implemented LITERALLY (no era filter) as the primary number, with the
era-filtered count published beside it so a seat comparing figures can see
whether an era assumption explains a difference.

    python3 live/pm_research/da_gap_at_cutoff_count.py --selftest
    python3 live/pm_research/da_gap_at_cutoff_count.py count --tape PATH
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

import da_state_tape_verify as G

REPO = Path(__file__).resolve().parents[2]
PM_GAPS = REPO / "data/pm_5min/collector_gaps.jsonl"
ERA = "clob_v3_1"


def coin_gaps(path: Path = PM_GAPS, era: str | None = None
              ) -> tuple[dict[str, list[tuple[float, float]]], dict[str, int]]:
    """COIN -> sorted absolute [g_start, g_end) intervals, from the ledger.

    The ledger is the physical source of truth (R-191(2)). Malformed lines are
    COUNTED, never silently dropped.
    """
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    diag = collections.Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        diag["lines"] += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            diag["unparseable"] += 1
            continue
        if r.get("event") != "gap_closed":
            continue
        if era is not None and r.get("collector_version") != era:
            diag["era_filtered_out"] += 1
            continue
        s, e, slug = r.get("gap_start_ns"), r.get("gap_end_ns"), r.get("slug")
        if not (s and e and slug):
            diag["incomplete_gap_record"] += 1
            continue
        coin = str(slug).split("-", 1)[0]
        out[coin].append((s / 1e9, e / 1e9))
        diag["gaps"] += 1
    merged: dict[str, list[tuple[float, float]]] = {}
    for coin, iv in out.items():
        iv.sort()
        acc: list[tuple[float, float]] = []
        for a, b in iv:
            if acc and a <= acc[-1][1]:
                acc[-1] = (acc[-1][0], max(acc[-1][1], b))
            else:
                acc.append((a, b))
        merged[coin] = acc
        diag[f"merged_{coin}"] = len(acc)
    return merged, dict(diag)


def in_gap(intervals: list[tuple[float, float]], t: float) -> bool:
    """[g_start, g_end): lower-inclusive, upper-EXCLUSIVE (R-191(2))."""
    if not intervals:
        return False
    i = bisect.bisect_right(intervals, (t, float("inf"))) - 1
    if i < 0:
        return False
    a, b = intervals[i]
    return a <= t < b


def count(tape: Path, era: str | None = None) -> dict[str, Any]:
    gaps, diag = coin_gaps(era=era)
    total = collections.Counter()
    by_coin = collections.Counter()
    by_split = collections.Counter()
    by_coin_split: collections.Counter = collections.Counter()
    flagged_ids: list[dict[str, Any]] = []
    n_rows = 0
    max_epoch_err = 0.0
    epoch_checked = 0
    missing_fields = collections.Counter()
    for r in G.iter_tape(tape):
        n_rows += 1
        coin, t0, ts = r.get("coin"), r.get("t0"), r.get("t_start")
        if coin is None or t0 is None or ts is None:
            missing_fields["coin_t0_or_t_start"] += 1
            continue
        T = float(t0) + float(ts)
        # cross-check the tape's own absolute field against the ruled formula
        dte = r.get("decision_time_epoch")
        if dte is not None:
            epoch_checked += 1
            max_epoch_err = max(max_epoch_err, abs(float(dte) - T))
        if in_gap(gaps.get(coin, []), T):
            total["flagged"] += 1
            by_coin[coin] += 1
            sp = str(r.get("split", "?"))
            by_split[sp] += 1
            by_coin_split[(coin, sp)] += 1
            if len(flagged_ids) < 10:
                flagged_ids.append({
                    "slug": r.get("slug"), "side": r.get("side"),
                    "gen": r.get("gen"), "t_start": ts, "t0": t0,
                    "T_absolute": round(T, 6), "split": r.get("split"),
                    "state_status": r.get("state_status")})
    return {
        "instrument": "da_gap_at_cutoff_count_v1",
        "definition": "T = t0 + t_start in [g_start, g_end) of ANY gap of that "
                      "COIN in the collector-gaps ledger (R-191(2))",
        "era_filter": era,
        "tape": str(tape),
        "n_rows": n_rows,
        "GAP_AT_CUTOFF_total": total["flagged"],
        "by_coin": dict(by_coin),
        "by_split": dict(by_split),
        "by_coin_split": {f"{c}/{s}": v for (c, s), v in by_coin_split.items()},
        "ledger": diag,
        "rows_missing_required_fields": dict(missing_fields),
        "epoch_crosscheck": {
            "rows_checked": epoch_checked,
            "max_abs_error_vs_t0_plus_t_start": round(max_epoch_err, 9)},
        "first_10_flagged": flagged_ids,
    }


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    iv = [(100.0, 200.0), (300.0, 400.0)]
    ok(in_gap(iv, 100.0), "LOWER bound is INCLUSIVE")
    ok(not in_gap(iv, 200.0), "UPPER bound is EXCLUSIVE -- [g_start, g_end)")
    ok(in_gap(iv, 199.999), "just inside the upper bound is in")
    ok(not in_gap(iv, 99.999), "just below the lower bound is out")
    ok(in_gap(iv, 350.0), "a later interval is found (not just the first)")
    ok(not in_gap(iv, 250.0), "between intervals is out")
    ok(not in_gap([], 150.0), "no gaps for a coin means nothing is flagged")
    ok(in_gap(iv, 399.999) and not in_gap(iv, 400.0),
       "the exclusive upper bound holds on the LAST interval too")

    # merging must not create membership that neither original had
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "g.jsonl"
        recs = [
            {"event": "gap_closed", "slug": "btc-updown-5m-1", "collector_version": ERA,
             "gap_start_ns": int(100e9), "gap_end_ns": int(200e9)},
            {"event": "gap_closed", "slug": "btc-updown-5m-2", "collector_version": ERA,
             "gap_start_ns": int(150e9), "gap_end_ns": int(250e9)},
            {"event": "gap_closed", "slug": "eth-updown-5m-9", "collector_version": ERA,
             "gap_start_ns": int(100e9), "gap_end_ns": int(200e9)},
            {"event": "collector_start", "recv_ns": 1},
        ]
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n",
                     encoding="utf-8")
        g, d = coin_gaps(f)
        ok(d["unparseable"] == 1, "an unreadable ledger line is COUNTED")
        ok(d["gaps"] == 3, "non-gap events are not counted as gaps")
        ok(g["btc"] == [(100.0, 250.0)],
           "overlapping COIN-level gaps merge into one interval")
        ok(g["eth"] == [(100.0, 200.0)], "coins are kept separate")
        ok(in_gap(g["btc"], 220.0) and not in_gap(g["eth"], 220.0),
           "COIN-LEVEL scope: an instant in btc's gap is NOT in eth's -- the "
           "same instant gets different answers per coin, so a coin-blind "
           "rule cannot pass (R-42 mirror)")
        # the per-slug projection the ruling calls lossy
        ok(in_gap(g["btc"], 210.0),
           "an instant covered by the gap logged against slug-2 is in scope "
           "for ANY btc row -- this is precisely what per-slug scoping drops")
        gera, _ = coin_gaps(f, era="other_era")
        ok(gera == {}, "an era filter that matches nothing yields no gaps")

    print(f"da_gap_at_cutoff_count selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["count"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tape", default=None)
    ap.add_argument("--era", default=None,
                    help="optional collector_version filter (diagnostic)")
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.tape:
        raise SystemExit("--tape PATH required; refusing to guess a tape")
    rep = count(Path(a.tape), era=a.era)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

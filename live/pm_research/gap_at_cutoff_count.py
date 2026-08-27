"""R-191 GAP_AT_CUTOFF counter. Independent implementation, for reconciliation.

AUTHORISATION (R-126, in-file): R-191(1) — BE implements the definition
independently from the ruling text and exchanges counts with DA. Agreement is
required before the rebuild.

THE DEFINITION, transcribed from R-191 and implemented here without reference
to DA's implementation:
    a row is GAP_AT_CUTOFF iff its ABSOLUTE instant T = t0 + t_start lies in
    [g_start, g_end) of ANY recorded gap for that COIN in the collector-gaps
    ledger.
  * absolute on BOTH sides
  * COIN-level, never per-slug: a gap is a FEED event, so it affects every
    window overlapping it in time. Per-slug assignment drops warm-up and
    boundary overlaps -- which is exactly where BE's 0 came from.
  * universe = ALL tape rows; unit = ROWS; score-side reported beside.

SOURCE OF TRUTH IS THE LEDGER, not a reconstruction. BE first lifted per-slug
window-relative gaps onto each slug's t0 and got 262; reading
`collector_gaps.jsonl` directly gives 289. The ledger is what R-191 names, so
the ledger is what this counts -- the lift is kept only as a documented
cross-check, because a proxy that nearly agrees is the most dangerous kind.
"""
from __future__ import annotations

import bisect, collections, json, sys
from pathlib import Path

LEDGER = Path("/home/yuqing/ctaNew/data/pm_5min/collector_gaps.jsonl")
POPULATIONS = (
    ("train", Path("/home/yuqing/ctaNew/data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json")),
    ("score", Path("/home/yuqing/ctaNew/data/pm_5min/derived/harmful_exposure_rows_v3_topup.json")),
)


def load_coin_gaps(path: Path = LEDGER) -> dict:
    """Coin-level ABSOLUTE gap intervals, seconds, from the ledger."""
    out = collections.defaultdict(list)
    for line in path.open():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        a, b, c = d.get("gap_start_ns"), d.get("gap_end_ns"), d.get("coin")
        if a is None or b is None or c is None:
            continue
        out[c].append((a / 1e9, b / 1e9))
    for c in out:
        out[c].sort()
    return dict(out)


class GapIndex:
    """Interval membership with a running max-end, so overlapping intervals
    (btc carries 1,491 for ~7,960 s) cannot cause a missed hit."""

    def __init__(self, gaps: dict):
        self.iv = gaps
        self.idx = {}
        for c, v in gaps.items():
            starts = [a for a, _ in v]
            ends, m = [], float("-inf")
            for _, b in v:
                m = max(m, b); ends.append(m)
            self.idx[c] = (starts, ends)

    def match(self, coin: str, T: float):
        """Return the matched interval, or None. HALF-OPEN [a, b) per R-191."""
        if coin not in self.idx:
            return None
        starts, ends = self.idx[coin]
        j = bisect.bisect_right(starts, T) - 1
        while j >= 0 and ends[j] > T:
            a, b = self.iv[coin][j]
            if a <= T < b:
                return (a, b)
            j -= 1
        return None


def count(populations=POPULATIONS, ledger: Path = LEDGER) -> dict:
    gi = GapIndex(load_coin_gaps(ledger))
    total = 0
    per_coin = collections.Counter()
    per_split = collections.Counter()
    first = []
    for split, path in populations:
        d = json.loads(path.read_text())
        for r in d["rows"]:
            if r["status"] != "OK":
                continue
            T = float(r["t0"]) + float(r["t_start"])
            m = gi.match(r["coin"], T)
            if m is None:
                continue
            total += 1
            per_coin[r["coin"]] += 1
            per_split[split] += 1
            if len(first) < 10:
                first.append({"slug": r["slug"], "side": r["side"],
                              "gen": r["gen"], "t_start": round(float(r["t_start"]), 6),
                              "absolute_T": round(T, 6),
                              "matched_gap": [round(m[0], 6), round(m[1], 6)],
                              "gap_len_s": round(m[1] - m[0], 6),
                              "split": split})
    return {"definition": "T = t0 + t_start in [g_start, g_end) of ANY gap for "
                          "that COIN in the collector-gaps ledger",
            "source": ledger.name, "unit": "rows", "universe": "all OK tape rows",
            "total": total, "by_coin": dict(per_coin), "by_split": dict(per_split),
            "first_10_flagged": first}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    gi = GapIndex({"btc": [(10.0, 20.0), (15.0, 30.0), (100.0, 101.0)]})
    ok(gi.match("btc", 10.0) == (10.0, 20.0), "the interval start is INCLUDED")
    ok(gi.match("btc", 20.0) == (15.0, 30.0),
       "20.0 is excluded from [10,20) but INSIDE [15,30) -- overlapping "
       "intervals must not mask each other")
    ok(gi.match("btc", 30.0) is None, "the interval end is EXCLUDED (half-open)")
    ok(gi.match("btc", 99.0) is None, "a point between intervals does not match")
    ok(gi.match("eth", 15.0) is None, "a coin with no gaps never matches")
    ok(gi.match("btc", 100.5) == (100.0, 101.0), "a later interval still matches")
    print(f"gap_at_cutoff_count selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    selftest()
    r = count()
    print(json.dumps(r, indent=1))

"""Offline confirmations from the five-arm score dumps (charter step 4).

4a is done upstream (the five-arm run reproduced run 1 to the cent on
identical rows). This script does:
  * dump VALIDATION — evaluate_policy on reconstructed rows must reproduce
    the v2 receipt's net to the cent, else REFUSE (the attribution below
    would be meaningless on a drifted dump);
  * 4c per-hour attribution — each cancelled generation's value attributed
    to the hour of its crossing row; the fine increment must not live in a
    single hour. The concentration predicate is COMPUTED, not narrated.

Development evidence; consumed era tape; day-unit intervals impossible (G=1).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import harmful_action_eval as ae

PM = Path("/home/yuqing/ctaNew/data/pm_5min")
ARMS = ("PM_ONLY", "PM_PLUS_FINE", "PM_FINE_SHIFTED",
        "PM_FINE_EXTENDED", "PM_FINE_PLUS_DEPTH")
L = "50"


def load(coin: str):
    rows, scores = [], {a: [] for a in ARMS}
    with gzip.open(PM / f"derived/harmful_scores_{coin}_v2.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({
                "slug": d["slug"], "side": d["side"], "gen": d["gen"],
                "t0": 0.0, "t_start": d["t_abs"],
                "latency": {L: {"preventable_value_cents": d["pv50"]}},
                "any_fill_ahead": d["af"]})
            for a in ARMS:
                scores[a].append(d["scores"][a])
    return rows, scores


def hourly(rows, sc, budget: float):
    """Replicates evaluate_policy's cancel loop with per-hour attribution.
    Aggregate must equal evaluate_policy's net (checked by the caller)."""
    gens: dict = {}
    for i, r in enumerate(rows):
        gens.setdefault((r["slug"], r["side"], r["gen"]), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: rows[i]["t_start"])
    gmax = {k: max(sc[i] for i in v) for k, v in gens.items()}
    order = sorted(gens, key=lambda k: -gmax[k])
    kk = max(1, int(len(gens) * budget))
    theta = gmax[order[kk - 1]]
    net = 0.0
    by_hour: dict[int, float] = {}
    for gk in order[:kk]:
        cross = next(i for i in gens[gk] if sc[i] >= theta)
        r = rows[cross]
        v = (r["latency"][L]["preventable_value_cents"]
             if r["any_fill_ahead"] else 0.0)
        net += v
        h = int((r["t0"] + r["t_start"]) // 3600) % 24
        by_hour[h] = by_hour.get(h, 0.0) + v
    return net, by_hour


def main(coin: str) -> int:
    receipt = json.loads(
        (PM / "derived/harmful_fine_comparison_v2.json").read_text())
    rows, scores = load(coin)
    print(f"{coin}: {len(rows)} dev rows reconstructed")
    for a in ARMS:
        gate = ae.evaluate_policy(rows, scores[a], latency_ms=50, n_random=1)
        for b in ("5%", "10%", "15%"):
            want = receipt["paired_arms"][coin][a]["gate"]["budgets"][b][
                "net_cents"]
            got = gate["budgets"][b]["net_cents"]
            if abs(got - want) > 0.01:
                print(f"REFUSED: dump drift {a} @{b}: {got} != {want}")
                return 2
    print("  dump VALID: all 15 arm x budget nets reproduce the receipt")
    for b in (0.05, 0.10, 0.15):
        per = {}
        for a in ARMS:
            net, byh = hourly(rows, scores[a], b)
            g = ae.evaluate_policy(rows, scores[a], latency_ms=50, n_random=1)
            if abs(net - g["budgets"][f"{int(b*100)}%"]["net_cents"]) > 0.01:
                print(f"REFUSED: attribution loop drifted for {a}")
                return 2
            per[a] = byh
        hrs = sorted(set(h for byh in per.values() for h in byh))
        inc = {h: per["PM_PLUS_FINE"].get(h, 0.0) - per["PM_ONLY"].get(h, 0.0)
               for h in hrs}
        pos = sum(v for v in inc.values() if v > 0)
        top_h, top_v = max(inc.items(), key=lambda kv: kv[1])
        n_pos = sum(1 for v in inc.values() if v > 0)
        conc = (top_v / pos) if pos > 0 else float("nan")
        single_hour = conc > 0.5
        print(f"  @{int(b*100)}%: fine-vs-PM increment by hour "
              f"(n_hours={len(hrs)}, positive in {n_pos}):")
        print("    " + "  ".join(f"h{h:02d}:{inc[h]:+.0f}" for h in hrs))
        print(f"    top hour h{top_h:02d} carries {conc:.0%} of positive "
              f"increment -> single_hour_concentration={single_hour}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "btc"))

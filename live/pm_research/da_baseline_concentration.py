"""FINDING (Q-DA-241): independently reproduced the baseline decomposition from the reference (maker P&L 8598.7588, spread 10566.9510, adverse -1968.1922, n_fills 4315) and measured its concentration -- one contiguous hour, hour 14 carrying 88.2%, top 1% of fills carrying 113% of the net.

DA: interrogate the BASELINE, which is now the load-bearing quantity.

At 0 cancels the baseline's fills ARE the reference's kept tranches, so its
decomposition is recomputable from the reference alone -- independently of
DE's replay. Reproducing DE's three totals first is what makes the
concentration numbers below worth reading.
"""
import sys, json, datetime as dt, collections, pickle
sys.path.insert(0, "live/pm_research")

d = pickle.loads(open("/home/yuqing/ctaNew/data/pm_5min/derived/"
                      "de_section81_cache_12.pkl", "rb").read())
ref = d["fr"]["reference"]

def hour_of(slug):
    return dt.datetime.fromtimestamp(int(slug.rsplit("-", 1)[1]),
                                     dt.timezone.utc).hour

pnl = spread = 0.0
n = 0
by_win = collections.defaultdict(float)
by_hour = collections.defaultdict(float)
per_fill = []
for slug, sides in sorted(ref.items()):
    for side, gens in sorted(sides.items()):
        sgn = 1.0 if side == "BUY_UP" else -1.0
        for g in gens:
            for t in g["tranches"]:
                sh = float(t["shares"])
                mo = t["markout_cents_per_share"]
                mid = t.get("mid_at_fill")
                lvl = t.get("level")
                if mo is None or not sh:
                    continue
                v = float(mo) * sh                  # P&L in cents
                pnl += v
                n += 1
                by_win[slug] += v
                by_hour[hour_of(slug)] += v
                per_fill.append(v)
                if mid is not None and lvl is not None:
                    spread += sgn * (float(mid) - float(lvl)) * 100.0 * sh

tot = pnl
wins = sorted(by_win.items(), key=lambda kv: -kv[1])
hrs = sorted(by_hour.items(), key=lambda kv: -kv[1])
per_fill.sort(reverse=True)
out = {
    "REPRODUCTION": {
        "maker_pnl_cents": pnl, "DE_reported": 8598.758849499998,
        "delta": pnl - 8598.758849499998,
        "spread_capture_cents": spread, "DE_reported_spread": 10566.951030999997,
        "spread_delta": spread - 10566.951030999997,
        "adverse_selection_cents": pnl - spread,
        "DE_reported_adverse": -1968.1921814999978,
        "n_fills": n, "DE_reported_n_fills": 4315,
    },
    "CONCENTRATION": {
        "n_windows": len(by_win), "n_hours": len(by_hour),
        "max_single_window_net_cents": wins[0][1],
        "max_single_window": wins[0][0],
        "max_single_window_share_of_net": wins[0][1] / tot,
        "net_excluding_best_window": tot - wins[0][1],
        "positive_without_best_window": (tot - wins[0][1]) > 0,
        "max_single_hour_net_cents": hrs[0][1],
        "max_single_hour": hrs[0][0],
        "max_single_hour_share_of_net": hrs[0][1] / tot,
        "net_excluding_best_hour": tot - hrs[0][1],
        "positive_without_best_hour": (tot - hrs[0][1]) > 0,
        "n_windows_positive": sum(1 for _, v in wins if v > 0),
        "n_windows_negative": sum(1 for _, v in wins if v < 0),
        "windows_sorted_cents": [(k, round(v, 2)) for k, v in wins],
        "hours_sorted_cents": [(k, round(v, 2)) for k, v in hrs],
        "top1_fill_share_of_net": per_fill[0] / tot,
        "top10_fill_share_of_net": sum(per_fill[:10]) / tot,
        "top1pct_fill_share_of_net": sum(per_fill[:max(1, n // 100)]) / tot,
    },
}
print(json.dumps(out, indent=1, default=str))

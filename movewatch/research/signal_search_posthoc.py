"""Post-hoc concentration check on the validated A1 fade (docs/SIGNAL_SEARCH.md, "Post-hoc").

Run AFTER the one validation read, so it is descriptive only and decides nothing. A1's mean fade
P&L in V looked implausibly large, and V contains the 2025-10-10 liquidation crash. This reports the
P&L by week and day, and the fade with the crash week left out (all / pumps / dumps).

  python3 -m research.signal_search_posthoc
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import research.signal_search as ss

CRASH_WEEK = pd.Timestamp("2025-10-06")


def v_rows() -> pd.DataFrame:
    """The validation rows exactly as validate() saw them (same seam de-dup, same weeks)."""
    v = ss.load_rows(ss.VAL).sort_values("time")
    keep, last = [], {}
    for r in v.itertuples():
        lt = last.get(r.asset)
        if lt is None or r.time - lt >= pd.Timedelta(hours=72):
            keep.append(r.Index)
            last[r.asset] = r.time
    v = v.loc[keep].copy()
    v["week"] = ss.ps.cluster_weeks(v["time"])
    return v


def main() -> None:
    fz = json.loads(ss.FREEZE.read_text())
    c = next(x for x in fz["cells"] if x["id"] == "A1")
    out = {}
    for name, df in (("V", v_rows()), ("S", ss.load_rows(ss.SCREEN))):
        x = df["mkt_breadth"].to_numpy(float)
        df = df.assign(net_fade=ss.net(df, "x4h", False))
        g = df[(x > c["cuts"][1]) & np.isfinite(df["net_fade"])]           # A1 sign -1: top tercile -> fade
        wk = g.groupby("week")["net_fade"].agg(["size", "sum"]).sort_values("sum", ascending=False)
        day = g.groupby(g["time"].dt.date)["net_fade"].agg(["size", "sum", "mean"]).sort_values("sum", ascending=False)
        res = {"fade_group": ss.mean_ci(g["net_fade"].to_numpy(), g["week"].to_numpy(), ss.SEED),
               "top_week": {"week": str(wk.index[0].date()), "n": int(wk["size"].iloc[0]),
                            "share_of_total_pnl": round(float(wk["sum"].iloc[0] / g["net_fade"].sum()), 3)},
               "top_day": {"day": str(day.index[0]), "n": int(day["size"].iloc[0]),
                           "mean_pct": round(100 * float(day["mean"].iloc[0]), 2)}}
        ex = g[g["week"] != CRASH_WEEK]
        res["excl_2025-10-06_week"] = {
            "all": ss.mean_ci(ex["net_fade"].to_numpy(), ex["week"].to_numpy(), ss.SEED),
            **{lab: ss.mean_ci(ex[ex["dir"] == d]["net_fade"].to_numpy(), ex[ex["dir"] == d]["week"].to_numpy(), ss.SEED)
               for d, lab in ((1, "pumps"), (-1, "dumps"))}}
        out[name] = res
    p = ss.OUT / "posthoc_A1.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

"""Concrete results: edge stability, leg split, and can-we-farm-the-working-leg — day-clustered CIs.
Same deployed pipeline (v0_feature_ablation). Run: python3 -u -m live.verify_stability

A) rank-IC + long/short/spread per era, each with day-clustered CI, AND the RECENT-OOS difference with CI
   -> answers "is the edge stable across eras?"  (diff CI spans 0 = can't reject stable)
B) leg-farming: monthly long/short skill over a continuous walk-forward -> persistence (autocorr) +
   a walk-forward adaptive-leg selector vs the neutral both-legs book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, gen, perbar_ic, V0, RECENT_CUTS, OOS_CUTS


def perbar_legs(P, k=0.2):
    rows = []
    for t, g in P.groupby("open_time"):
        if len(g) < 5:
            continue
        g = g.sort_values("pred"); nk = max(1, int(len(g) * k))
        rows.append((t, g["alpha_A"].iloc[-nk:].mean(), -g["alpha_A"].iloc[:nk].mean()))
    d = pd.DataFrame(rows, columns=["open_time", "long", "short"]).set_index("open_time")
    d["spread"] = d["long"] + d["short"]
    return d


def day_boot(s, nb=4000, seed=1):
    x = pd.DataFrame({"v": np.asarray(s.values, float)}, index=pd.to_datetime(s.index, utc=True))
    x["d"] = x.index.floor("1D")
    grp = [g["v"].values for _, g in x.groupby("d")]
    rng = np.random.default_rng(seed); k = len(grp)
    boot = np.array([np.concatenate([grp[i] for i in rng.integers(0, k, k)]).mean() for _ in range(nb)])
    return float(np.nanmean(s.values)), boot


def show(name, sr, sr_scale=1.0):
    m, b = day_boot(sr)
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"    {name:<10} {m*sr_scale:+7.4f}  [{lo*sr_scale:+.4f}, {hi*sr_scale:+.4f}]", flush=True)
    return m, b


def main():
    PAN = build_panel()
    print(f"panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms\n", flush=True)

    # ---- A) stability with CIs ----
    boots = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        P = gen(PAN, V0, cuts)
        ic = perbar_ic(P); legs = perbar_legs(P)
        print(f"===== {era} =====", flush=True)
        boots[(era, "ic")] = show("rank-IC", ic)
        boots[(era, "long")] = show("long(bps)", legs["long"], 1e4)
        boots[(era, "short")] = show("short(bps)", legs["short"], 1e4)
        boots[(era, "spread")] = show("spread(bps)", legs["spread"], 1e4)
        print("", flush=True)

    print("===== RECENT − OOS difference (CI spans 0 = NOT distinguishable = consistent with stable) =====",
          flush=True)
    for key, scale in (("ic", 1.0), ("long", 1e4), ("short", 1e4), ("spread", 1e4)):
        bd = boots[("RECENT", key)][1] - boots[("OOS", key)][1]
        m = (boots[("RECENT", key)][0] - boots[("OOS", key)][0]) * scale
        lo, hi = np.percentile(bd * scale, [2.5, 97.5])
        verdict = "STABLE (spans 0)" if lo < 0 < hi else "ERA-DIFFERENT"
        print(f"    Δ {key:<10} {m:+7.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}", flush=True)

    # ---- B) can we farm the working leg? ----
    print("\n===== B) LEG-FARMING (continuous walk-forward, monthly) =====", flush=True)
    ALL = sorted(set(OOS_CUTS + RECENT_CUTS))
    Pall = gen(PAN, V0, ALL)
    legs = perbar_legs(Pall)
    legs.index = pd.to_datetime(legs.index, utc=True)
    mo = legs.resample("1ME").mean().dropna()
    mo = mo[mo.index >= mo.index[0]]
    print(f"  months={len(mo)} | monthly long/short skill autocorr(lag1): "
          f"long {mo['long'].autocorr(1):+.2f}, short {mo['short'].autocorr(1):+.2f} "
          f"(near 0 = not persistent = can't PIT-detect the working leg)", flush=True)
    # walk-forward: pick the leg with better trailing-3mo skill, take its NEXT month realized P&L
    pick_long = (mo["long"].rolling(3).mean().shift(1) >= mo["short"].rolling(3).mean().shift(1))
    adaptive = np.where(pick_long, mo["long"], mo["short"])
    valid = ~np.isnan(mo["long"].rolling(3).mean().shift(1).values)
    def sharpe(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
    print(f"  monthly Sharpe (annualized):", flush=True)
    print(f"    neutral both-legs (spread) : {sharpe(mo['spread'][valid]):+.2f}  "
          f"mean {mo['spread'][valid].mean()*1e4:+.2f}bps/mo", flush=True)
    print(f"    fixed LONG only            : {sharpe(mo['long'][valid]):+.2f}", flush=True)
    print(f"    fixed SHORT only           : {sharpe(mo['short'][valid]):+.2f}", flush=True)
    print(f"    ADAPTIVE (farm trailing-best leg): {sharpe(adaptive[valid]):+.2f}  "
          f"mean {np.nanmean(adaptive[valid])*1e4:+.2f}bps/mo", flush=True)
    print("  (adaptive must BEAT neutral to justify giving up market-neutrality)", flush=True)
    print("\nSTABILITYDONE", flush=True)


if __name__ == "__main__":
    main()

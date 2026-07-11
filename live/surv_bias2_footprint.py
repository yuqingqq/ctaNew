"""surv_bias2_footprint (user-chosen: quantify Bias 2 first): the production +2.30 replay applies a PIT
$3M/day trailing-30d liquidity gate (CONVEXITY_PIT_DVOL=1). This measures how much that gate does — i.e.
Bias 2's footprint = how often the RAW (ungated) 1L/2S selection would pick a sub-gate name, and whether
those sub-gate legs carry disproportionate (spurious) PnL. If sub-gate legs are rare / not extra-profitable,
Bias 2 is immaterial and the headline number is not gate-sensitive. If they're frequent & profitable, the
ungated number would be inflated (and the full gate-ON vs gate-OFF replay is warranted).

Gate applied at ENTRY (dvol.asof(t), PIT — the same series eligible_universe_at reads). Residual book PnL.
"""
import pickle, numpy as np, pandas as pd
import sys; sys.path.insert(0, "live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
GATE = 3_000_000.0  # $3M/day, matches LIQ_FLOOR_DOLLAR_VOL_30D default

def dvol_lookup():
    d = pickle.load(open("live/state/v3loop/ddloop/_dvol_cache.pkl", "rb"))["dvol"]
    return {k: v.sort_index() for k, v in d.items()}

def asof(dv, sym, t):
    s = dv.get(sym)
    if s is None or len(s) == 0:
        return np.nan
    try:
        return float(s.asof(t))
    except Exception:
        return np.nan

def build(base, long, reg, dv):
    lg = long.groupby("open_time"); rows = []
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or reg.get(t) in (None, "deepbull"):
            continue
        try: gl = lg.get_group(t)
        except KeyError: continue
        Lp = gl.nlargest(1, "pred"); S = g.nsmallest(2, "pred")
        if len(Lp) < 1 or len(S) < 2: continue
        for _, r in Lp.iterrows():
            rows.append((t, r["symbol"], "L", r["alpha_A"] * 1e4, asof(dv, r["symbol"], t)))
        for _, r in S.iterrows():
            rows.append((t, r["symbol"], "S", -r["alpha_A"] * 1e4, asof(dv, r["symbol"], t)))
    d = pd.DataFrame(rows, columns=["t", "sym", "side", "pnl", "dvol"])
    d["subgate"] = d["dvol"] < GATE               # finite & below (NaN -> False, matches "kept" in prod)
    d["unknown"] = ~np.isfinite(d["dvol"])
    return d

def daily_sharpe(d):
    day = pd.to_datetime(d["t"]).dt.date
    dr = d.assign(day=day).groupby("day")["pnl"].sum()   # equal-weight leg sum per day (proxy book)
    return (dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else np.nan

def main():
    reg = btc_reg(); dv = dvol_lookup()
    for era, bp, lp in (("RECENT", "hl_tgt_res_base_cleanfix", "hl_tgt_res_long_cleanfix"),
                        ("OOS", "hl_v4base_oos_cleanfix", "hl_v4long_oos_cleanfix")):
        base, long = load(bp, lp); d = build(base, long, reg, dv)
        n = len(d); nsg = int(d.subgate.sum()); nunk = int(d.unknown.sum())
        print(f"\n===== {era}: {n} raw 1L/2S legs =====")
        print(f"  sub-gate (<${GATE/1e6:.0f}M dvol at entry): {nsg} ({nsg/n*100:.1f}%) | unknown dvol: {nunk} ({nunk/n*100:.1f}%)")
        liq = d[~d.subgate & ~d.unknown]; sg = d[d.subgate]
        print(f"  mean PnL/leg  — liquid: {liq.pnl.mean():+.1f} bps | sub-gate: {sg.pnl.mean() if len(sg) else float('nan'):+.1f} bps")
        print(f"  sub-gate PnL share of gross: {sg.pnl.sum()/d.pnl.sum()*100 if d.pnl.sum()!=0 else float('nan'):.1f}% (of {d.pnl.sum():+.0f} bps total)")
        # by side
        for sd in ("L","S"):
            ds=d[d.side==sd]; dss=ds[ds.subgate]
            print(f"    {sd}: {len(dss)}/{len(ds)} sub-gate ({len(dss)/len(ds)*100:.1f}%), sub-gate mean {dss.pnl.mean() if len(dss) else float('nan'):+.1f} bps")
        # proxy-book Sharpe: ungated (all legs) vs gated (drop sub-gate legs)
        sh_all = daily_sharpe(d); sh_gated = daily_sharpe(d[~d.subgate])
        print(f"  proxy-book daily Sharpe — UNGATED(all legs) {sh_all:+.2f} | GATED(drop sub-gate) {sh_gated:+.2f} | Bias2 lift(ungated-gated) {sh_all-sh_gated:+.2f}")
    print("BIAS2FOOTPRINTDONE")

if __name__ == "__main__":
    main()

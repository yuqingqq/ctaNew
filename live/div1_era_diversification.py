"""DIV1: v4 (crypto) × xyz-v7 (US equity) era-diversification test (addenda 23 + 23b).

Tests path (C). Amendments per design review (23b): (1) 2022 CRISIS cross-correlation DESCRIPTIVELY
(the decisive era; correlation-only, no selection → does not spend the holdout); (2) verdict ceiling
= CANDIDATE (two un-forward-validated backtests); (3) bad-era corr with n + bootstrap CI; (4)
INVERSE-VOL combined-book weighting (pinned, no argmax DoF).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); rng = np.random.default_rng(23)
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))

def xyz_weekly():
    from ml.research.alpha_v7_xyz import (load_universe, load_anchors, XYZ_US_EQUITY,
        add_features_A, add_features_B, construct_portfolio_subset, TOP_K_XYZ,
        COST_PER_TRADE_BPS, HOLD_DAYS)
    from ml.research.alpha_v7_weekly import (add_returns_and_basket, add_residual_5d, fit_predict, make_folds)
    panel, earnings, surv = load_universe(); sp100 = set(surv)
    xyz_in = [s for s in XYZ_US_EQUITY if s in sp100]
    panel = add_returns_and_basket(panel); panel = add_residual_5d(panel)
    panel, fA = add_features_A(panel); panel, fB = add_features_B(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    feats = fA + fB + ["sym_id"]; folds = make_folds(panel, train_min_days=365*3, test_days=365)
    pnls = []
    for train_end, test_start, test_end in folds:
        tr = panel[panel["ts"] <= train_end].copy(); te = panel[(panel["ts"]>=test_start)&(panel["ts"]<=test_end)].copy()
        tp = fit_predict(tr, te, feats, "fwd_resid_5d")
        if tp.empty: continue
        lp = construct_portfolio_subset(tp, "pred", "fwd_resid_5d", allowed_symbols=set(xyz_in),
                                        top_k=TOP_K_XYZ, cost_bps=COST_PER_TRADE_BPS, hold_days=HOLD_DAYS)
        if not lp.empty: pnls.append(lp)
    p = pd.concat(pnls, ignore_index=True); p["ts"] = pd.to_datetime(p["ts"], utc=True)
    p["week"] = p["ts"].dt.to_period("W").astype(str)
    return (p.groupby("week")["spread_alpha"].sum()*1e4).rename("xyz")

def v4_weekly_oos():
    D = REPO/"live/state/convexity"
    bb = pd.read_parquet(D/"hl_v4base_oos_clean/v0full_hl60.parquet", columns=["symbol","open_time","pred","alpha_A"])
    ll = pd.read_parquet(D/"hl_v4long_oos_clean/v0full_hl60.parquet", columns=["symbol","open_time","pred","alpha_A"])
    for x in (bb,ll): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
    gl = ll.groupby("open_time"); rows=[]
    for t,g in bb.groupby("open_time"):
        if len(g)<25: continue
        try: glt=gl.get_group(t)
        except KeyError: continue
        L=glt.nlargest(1,"pred").iloc[0]; S=g.nsmallest(2,"pred")
        net=(0.5*L["alpha_A"] + 0.5*(-S["alpha_A"].mean()))*1e4 - 4.5
        rows.append((t,net))
    d=pd.DataFrame(rows,columns=["t","net"]); d["week"]=d["t"].dt.to_period("W").astype(str)
    return d.groupby("week")["net"].sum().rename("v4")

def v4_weekly_2022():
    c=pd.read_csv(REPO/"live/state/longtail/holdout2022/B/cycles.csv")
    c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c["week"]=c["open_time"].dt.to_period("W").astype(str)
    return c.groupby("week")["pnl_bps"].sum().rename("v4")

def corr_ci(a,b,n=2000):
    idx=np.arange(len(a))
    cs=[np.corrcoef(a[s],b[s])[0,1] for s in (rng.choice(idx,len(idx)) for _ in range(n)) if np.std(a[s])>0 and np.std(b[s])>0]
    return np.percentile(cs,[2.5,97.5]) if cs else (np.nan,np.nan)

def main():
    print("regen xyz-v7 weekly (equity walk-forward)...", flush=True)
    xw=xyz_weekly(); print(f"  xyz weeks {len(xw)} {xw.index.min()}..{xw.index.max()}", flush=True)
    v4=v4_weekly_oos(); v22=v4_weekly_2022()
    print(f"  v4 OOS weeks {len(v4)} {v4.index.min()}..{v4.index.max()}; v4 2022 weeks {len(v22)}", flush=True)
    def sh(x): return x.mean()/x.std(ddof=1)*np.sqrt(52) if x.std(ddof=1)>0 else np.nan
    def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())
    # --- 2023-26 overlap (underperform eras) ---
    m=pd.concat([v4,xw],axis=1).dropna()
    print(f"\n=== 2023-26 overlap ({len(m)} wk) ===")
    print(f"OVERALL corr {m.v4.corr(m.xyz):+.3f}")
    bad=m[m.v4<0]; lo,hi=corr_ci(bad.v4.values,bad.xyz.values)
    print(f"v4-NEGATIVE weeks (n={len(bad)}): corr {bad.v4.corr(bad.xyz):+.3f} CI[{lo:+.2f},{hi:+.2f}] | "
          f"xyz mean in v4-bad wks {bad.xyz.mean():+.1f} bps (want >0)")
    sv=m.v4.std(); sx=m.xyz.std(); wv=(1/sv)/((1/sv)+(1/sx)); wx=1-wv   # INVERSE-VOL (pinned)
    comb=wv*m.v4+wx*m.xyz
    print(f"inverse-vol weights v4 {wv:.2f}/xyz {wx:.2f}; weekly Sharpe v4 {sh(m.v4):+.2f} xyz {sh(m.xyz):+.2f} COMBINED {sh(comb):+.2f}")
    print(f"weekly maxDD v4 {mdd(m.v4.values):+.0f} xyz {mdd(m.xyz.values):+.0f} combined {mdd(comb.values):+.0f}")
    m["yr"]=m.index.str[:4]
    for yr,g in m.groupby("yr"):
        print(f"  {yr}: v4 {g.v4.mean():+.0f} xyz {g.xyz.mean():+.0f} comb {(wv*g.v4+wx*g.xyz).mean():+.0f} (n={len(g)})")
    # --- 2022 CRISIS descriptive cross-correlation (the DECISIVE era; correlation-only) ---
    x22=xw[xw.index.str[:4]=="2022"]
    m22=pd.concat([v22,x22],axis=1).dropna()
    print(f"\n=== 2022 CRISIS descriptive cross-corr (n={len(m22)} wk; correlation-ONLY, no selection) ===")
    if len(m22)>=10:
        l2,h2=corr_ci(m22.v4.values,m22.xyz.values)
        print(f"corr(v4_2022, xyz_2022) {m22.v4.corr(m22.xyz):+.3f} CI[{l2:+.2f},{h2:+.2f}]  "
              f"[decisive era; low = diversifies the crisis, high = fails when needed]")
        print(f"  2022 means: v4 {m22.v4.mean():+.0f} bps/wk (the FAIL) | xyz {m22.xyz.mean():+.0f} bps/wk")
    else:
        print(f"  INSUFFICIENT 2022 xyz overlap (n={len(m22)}) — decisive era untestable, pass necessary-not-sufficient")
    print("\nCEILING: two un-forward-validated backtests -> any pass is a CANDIDATE, not confirmation.")
    print("DIV1DONE")

if __name__ == "__main__":
    main()

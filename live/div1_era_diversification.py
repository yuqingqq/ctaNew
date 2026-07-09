"""DIV1: v4 (crypto) × xyz-v7 (US equity) era-diversification (addenda 23 + 23b + 23d).

v4 = FULL KEEPSET4 stack, CONSISTENT across ALL eras (code-review fix c90a81e-#1): 2022
holdout2022/B + 2023-25 DB1-OOS replay + 2025-26 DB1-recent replay, all pnl_bps (production
strategy). Cross-asset corr (block bootstrap, 23d-#3), lead with xyz-mean-in-v4-bad-weeks
(23d-#4), inverse-vol combined book with TRAILING vol (23d-#2), 2022 crisis descriptive corr
(23b-#1), CANDIDATE ceiling (both un-forward-validated). v4 cycles = DB1 replay (regenerable);
stabilized at live/state/convexity/div1_v4cyc/.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); rng = np.random.default_rng(23)
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
VC = REPO / "live/state/convexity/div1_v4cyc"

def xyz_weekly():
    from ml.research.alpha_v7_xyz import (load_universe, load_anchors, XYZ_US_EQUITY,
        add_features_A, add_features_B, construct_portfolio_subset, TOP_K_XYZ, COST_PER_TRADE_BPS, HOLD_DAYS)
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

def v4_weekly_fullstack():
    """FULL KEEPSET4 stack pnl_bps, consistent across eras (2022 holdout + DB1 OOS + DB1 recent)."""
    parts = []
    for f in ("y2022.csv", "oos.csv", "recent.csv"):
        c = pd.read_csv(VC/f); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        parts.append(c[["open_time","pnl_bps"]])
    c = pd.concat(parts, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    c["week"] = c["open_time"].dt.to_period("W").astype(str)
    return c.groupby("week")["pnl_bps"].sum().rename("v4")

def block_corr_ci(a, b, L=6, n=2000):
    m = len(a); nb = int(np.ceil(m/L)); cs = []
    for _ in range(n):
        starts = rng.integers(0, max(1, m-L+1), nb)
        idx = np.concatenate([np.arange(s, s+L) for s in starts])[:m] % m
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0: cs.append(np.corrcoef(a[idx], b[idx])[0,1])
    return (np.percentile(cs,[2.5,97.5]) if cs else (np.nan,np.nan))

def main():
    print("regen xyz-v7 weekly (equity walk-forward)...", flush=True)
    xw = xyz_weekly(); v4 = v4_weekly_fullstack()
    print(f"  xyz {len(xw)} wk; v4 full-stack {len(v4)} wk {v4.index.min()}..{v4.index.max()}", flush=True)
    def sh(x): return x.mean()/x.std(ddof=1)*np.sqrt(52) if x.std(ddof=1)>0 else np.nan
    def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())
    m = pd.concat([v4, xw], axis=1).dropna(); m = m.sort_index(); m["yr"] = m.index.str[:4]
    m23 = m[m.yr != "2022"]
    print(f"\n=== 2023-26 overlap ({len(m23)} wk), v4 = PRODUCTION stack ===")
    print(f"OVERALL corr {m23.v4.corr(m23.xyz):+.3f}")
    bad = m23[m23.v4 < 0]
    print(f"** xyz mean in v4-BAD weeks (n={len(bad)}): {bad.xyz.mean():+.1f} bps ** (>0 = pays when v4 down; PRIMARY)")
    lo,hi = block_corr_ci(bad.v4.values, bad.xyz.values)
    print(f"   (secondary, range-restricted) v4-bad corr {bad.v4.corr(bad.xyz):+.3f} block-CI[{lo:+.2f},{hi:+.2f}]")
    # TRAILING inverse-vol combined (PIT, 12-wk min expanding)
    sv = m23.v4.expanding(12).std().shift(1); sx = m23.xyz.expanding(12).std().shift(1)
    wv = (1/sv)/((1/sv)+(1/sx)); comb = (wv*m23.v4 + (1-wv)*m23.xyz).dropna()
    print(f"trailing inv-vol combined ({len(comb)} wk): weekly Sharpe v4 {sh(m23.v4):+.2f} | xyz {sh(m23.xyz):+.2f} | COMBINED {sh(comb):+.2f}")
    print(f"weekly maxDD (raw): v4 {mdd(m23.v4.values):+.0f} | xyz {mdd(m23.xyz.values):+.0f} | combined {mdd(comb.values):+.0f}  [raw = diversification + de-risking]")
    # MATCHED-VOL maxDD (review 4b656da): lever combined to v4 vol so the DD cut is vol-neutral diversification, not lower exposure
    v4v = m23.v4.reindex(comb.index); cm = comb * (v4v.std()/comb.std())
    dd_v4 = mdd(v4v.values); dd_cm = mdd(cm.values)
    print(f"** MATCHED-VOL maxDD: v4 {dd_v4:+.0f} -> combined-at-v4-vol {dd_cm:+.0f} = {(1-dd_cm/dd_v4)*100:+.0f}% (HONEST diversification DD cut; combined vol was {comb.std()/v4v.std():.2f}x v4) **")
    # sub-window robustness: matched-vol DD cut per year
    print("   sub-window matched-vol DD cut:")
    for yr,g in m23.groupby("yr"):
        gc = comb.reindex(g.index).dropna(); gv = g.v4.reindex(gc.index)
        if len(gc) < 6: continue
        gcm = gc*(gv.std()/gc.std()); print(f"     {yr}: v4 DD {mdd(gv.values):+.0f} -> matched {mdd(gcm.values):+.0f} ({(1-mdd(gcm.values)/mdd(gv.values))*100 if mdd(gv.values)<0 else 0:+.0f}%) | v4 {gv.mean():+.0f}/xyz {g.xyz.mean():+.0f} (n={len(gc)})")
    # 2022 CRISIS descriptive (v4 now full-stack too → consistent)
    m22 = m[m.yr == "2022"]
    print(f"\n=== 2022 CRISIS descriptive corr (n={len(m22)} wk; v4 full-stack, correlation-ONLY) ===")
    if len(m22) >= 10:
        l2,h2 = block_corr_ci(m22.v4.values, m22.xyz.values)
        print(f"corr(v4_2022, xyz_2022) {m22.v4.corr(m22.xyz):+.3f} block-CI[{l2:+.2f},{h2:+.2f}]  "
              f"[low=diversifies crisis; means: v4 {m22.v4.mean():+.0f}/wk (FAIL) xyz {m22.xyz.mean():+.0f}/wk]")
    else: print(f"  INSUFFICIENT (n={len(m22)})")
    print("\nCEILING: two un-forward-validated backtests → CANDIDATE, not confirmation.")
    print("DIV1DONE")

if __name__ == "__main__":
    main()
